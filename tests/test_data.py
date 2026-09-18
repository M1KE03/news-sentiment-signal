"""B09b: the loader's universe filter, and structural RQ2 suppression.

The filter is not hygiene. FNSPID's `All_external.csv` concatenates five-plus
sub-corpora, one of which is a Russian-language general news site; an English
financial sentiment model will happily score it and return numbers that mean
nothing. The fixture below reproduces that file's shape, embedded newlines and
all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import data as sdata
from src import inference

# One row per source, plus the two pathologies the audit found: a record whose
# Article field contains a newline, and a Cyrillic headline.
FNSPID_CSV = '''Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,Lsa_summary,Luhn_summary,Textrank_summary,Lexrank_summary
2015-03-02 00:00:00 UTC,Apple beats on earnings,AAPL,https://www.benzinga.com/news/1,Benzinga,A,,,,,
2015-03-03 00:00:00 UTC,Stocks That Hit 52-Week Highs,MSFT,https://www.benzinga.com/news/2,Benzinga,B,,,,,
2015-03-04 00:00:00 UTC,Zacks rank upgrade for XYZ,XYZ,https://www.zacks.com/news/3,Zacks,C,,,,,
2015-03-05 00:00:00 UTC,Китай раскрыл планы по исследованию Луны,,https://lenta.ru/news/4,Lenta,D,,,,,
2015-03-06 09:14:00 UTC,Fashion comes first at Royal Ascot,,https://www.reuters.com/news/5,,,,,,,
2015-03-07 00:00:00 UTC,Multi-line article body here,GE,https://www.benzinga.com/news/6,Benzinga,E,"First line of the body
second line of the body",,,,
2016-01-04 00:00:00 UTC,Later headline in the window,AAPL,https://www.benzinga.com/news/7,Benzinga,F,,,,,
'''


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "all_external.csv"
    p.write_text(FNSPID_CSV, encoding="utf-8")
    return p


def test_embedded_newlines_do_not_corrupt_records(csv_path):
    """Naive line-splitting would turn the body's second line into a record."""
    df = sdata.load_news(csv_path, domains=())
    assert len(df) == 7, "the multi-line record must stay one row"
    assert df.attrs["load_stats"]["n_bad_timestamp"] == 0
    assert "second line of the body" not in set(df["text"])


def test_source_filter_removes_the_russian_corpus(csv_path):
    df = sdata.load_news(csv_path, domains=("benzinga.com",))
    assert set(df["source"]) == {"benzinga.com"}
    assert not df["text"].str.contains("Китай").any()
    stats = df.attrs["load_stats"]
    assert stats["n_wrong_source"] == 3            # zacks, lenta.ru, reuters
    assert "lenta.ru" in stats["rejected_hosts"]
    assert stats["n_kept"] == 4


def test_filter_defaults_to_the_configured_universe(csv_path):
    """Filtering is mandatory: the default comes from config, not from nothing."""
    df = sdata.load_news(csv_path)                 # no domains argument
    assert config.NEWS_SOURCE_DOMAINS == ("benzinga.com",)
    assert set(df["source"]) == {"benzinga.com"}


def test_disabling_the_filter_is_an_explicit_act(csv_path):
    df = sdata.load_news(csv_path, domains=())
    assert len(set(df["source"])) > 1
    assert df.attrs["load_stats"]["n_wrong_source"] == 0


def test_pooled_universe_can_be_selected(csv_path):
    df = sdata.load_news(csv_path, domains=config.NEWS_SOURCE_DOMAINS_POOLED)
    assert set(df["source"]) == {"benzinga.com", "zacks.com"}


def test_malformed_timestamps_are_dropped_and_counted(tmp_path):
    """Body text bleeding across a record boundary, as seen in the real file."""
    p = tmp_path / "bad.csv"
    p.write_text(
        FNSPID_CSV
        + "with the matter.  Bakrie & Brothers,frag,,,,,,,,,\n"
        + "$1.34 billion bridge loan,frag,,,,,,,,,\n",
        encoding="utf-8",
    )
    df = sdata.load_news(p, domains=())
    stats = df.attrs["load_stats"]
    assert stats["n_bad_timestamp"] == 2
    assert any("Bakrie" in e for e in stats["bad_timestamp_examples"])
    assert len(df) == 7


def test_timezone_is_applied_not_inferred(csv_path):
    df = sdata.load_news(csv_path, domains=("benzinga.com",))
    assert str(df["ts_utc"].dt.tz) == "UTC"
    assert str(df["ts_et"].dt.tz) == config.TZ_MARKET
    # 2015-03-02 00:00 UTC is 2015-03-01 19:00 ET -- the previous calendar day.
    first = df.iloc[0]
    assert first["ts_utc"].strftime("%Y-%m-%d") == "2015-03-02"
    assert first["ts_et"].strftime("%Y-%m-%d") == "2015-03-01"


def test_window_filter(csv_path):
    df = sdata.load_news(csv_path, domains=("benzinga.com",), start="2015-01-01", end="2015-12-31")
    assert len(df) == 3
    assert df["ts_utc"].max().year == 2015


def test_chunked_and_unchunked_agree(csv_path):
    a = sdata.load_news(csv_path, domains=("benzinga.com",), chunksize=None)
    b = sdata.load_news(csv_path, domains=("benzinga.com",), chunksize=2)
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs["load_stats"]["n_kept"] == b.attrs["load_stats"]["n_kept"]


def test_obsolete_standalone_benzinga_source_is_rejected(csv_path):
    with pytest.raises(ValueError, match="sub-corpus of"):
        sdata.load_news(csv_path, source="benzinga")


def test_schema_matches_the_contract(csv_path):
    df = sdata.load_news(csv_path, domains=("benzinga.com",))
    assert list(df.columns) == sdata.HEADLINE_COLUMNS
    assert df["headline_id"].is_unique
    assert df["ts_utc"].is_monotonic_increasing


# ------------------------------------------------- structural RQ2 suppression


def _panel():
    from src import align

    import numpy as np

    rng = np.random.default_rng(config.SEED)
    cal = pd.bdate_range("2015-01-05", periods=60)
    n = len(cal)
    daily = pd.DataFrame({"date": cal, "n_headlines": 5})
    for s in config.SCORERS:                      # vary, or the design is singular
        daily[f"s_{s}"] = rng.normal(0, 0.1, n)
        daily[f"d_{s}"] = rng.uniform(0.1, 0.4, n)
    mkt = pd.DataFrame(
        {"date": cal, "close_adj": 100 + np.arange(n, dtype=float),
         "ret": rng.normal(0, 0.01, n), "rv_parkinson": rng.uniform(1e-5, 5e-4, n),
         "volume": rng.uniform(5e7, 2e8, n), "log_volume": rng.normal(18.4, 0.2, n),
         "vix_close": rng.uniform(10, 30, n)}
    )
    return align.build_panel(daily, mkt)


def test_contemporaneous_refuses_to_run_when_rq2_is_inadmissible():
    assert config.RQ2_ADMISSIBLE is False
    with pytest.raises(RuntimeError, match="inadmissible"):
        inference.contemporaneous(_panel(), "finbert")


def test_contemporaneous_names_the_reason_and_the_evidence():
    with pytest.raises(RuntimeError) as exc:
        inference.contemporaneous(_panel(), "finbert")
    msg = str(exc.value)
    assert "intraday timestamps" in msg
    assert "data-audit-fnspid" in msg


def test_deliberate_override_is_available_but_must_be_asked_for():
    """Suppression is structural, not a prohibition: a labelled exhibit can opt in."""
    res = inference.contemporaneous(_panel(), "finbert", allow_inadmissible=True)
    assert res.nobs > 0


def test_fallback_is_marked_implemented():
    """The run_all guard must stand down now that the mapping exists."""
    assert config.DATE_ONLY_FALLBACK is True
    assert config.DATE_ONLY_FALLBACK_IMPLEMENTED is True


# ------------------------------------------------- near-duplicate detection


def _frame(texts, dates, tickers=None):
    """A minimal headline frame carrying everything `dedup` now requires.

    `source_row_id` is explicit rather than implied by row order: that is the
    whole point of R03b's representative rule.
    """
    return pd.DataFrame(
        {
            "source_row_id": range(len(texts)),
            "headline_id": [f"h{i}" for i in range(len(texts))],
            "text": texts,
            "text_norm": [t.lower() for t in texts],
            "ts_utc": [pd.Timestamp(d, tz="UTC") for d in dates],
            "tickers": tickers if tickers is not None else [[] for _ in texts],
        }
    )


def _pairwise_keep(df, window_days=3, overlap=0.9):
    """The obvious O(n x window) implementation, as a reference oracle."""
    import numpy as np

    ts, norms = df["ts_utc"].to_numpy(), df["text_norm"].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    w = pd.Timedelta(days=window_days).to_timedelta64()
    j0 = 0
    for i in range(len(df)):
        while ts[i] - ts[j0] > w:
            j0 += 1
        for j in range(j0, i):
            if keep[j] and sdata._token_set_overlap(norms[i], norms[j]) >= overlap:
                keep[i] = False
                break
    return keep


def test_blocked_dedup_matches_the_pairwise_oracle():
    """Blocking is an optimisation; it must not change the answer."""
    import numpy as np

    rng = np.random.default_rng(config.SEED)
    tmpl = [
        "cape bancorp inc cbnj ceo michael d devlin buys {} shares",
        "stocks that hit 52 week highs on friday",
        "apple reports quarterly profit above expectations for the period",
        "an unrelated story number {} about something else entirely today",
    ]
    texts, dates = [], []
    for d in pd.date_range("2015-01-01", periods=30):
        for k in range(20):
            t = tmpl[k % len(tmpl)]
            texts.append(t.format(rng.integers(1000, 9999)) if "{}" in t else t)
            dates.append(d)
    df = _frame(texts, dates)
    blocked = sdata.near_duplicate_keep(df["text_norm"], df["ts_utc"])
    assert (blocked == _pairwise_keep(df)).all()


def test_near_duplicate_catches_a_one_token_difference():
    """The real pattern: a CEO-purchase template differing only in a number.

    Eleven unique tokens, one differing: 10/11 = 0.909, which clears 0.9.
    """
    df = _frame(
        [
            "cape bancorp inc cbnj ceo michael d devlin buys 1081 shares",
            "cape bancorp inc cbnj ceo michael d devlin buys 1832 shares",
        ],
        ["2015-01-05", "2015-01-05"],
    )
    keep = sdata.near_duplicate_keep(df["text_norm"], df["ts_utc"])
    assert keep.tolist() == [True, False]


def test_threshold_is_strict_on_short_headlines():
    """Not a defect -- a property of the threshold, pinned so it is not a surprise.

    Nine unique tokens differing by one scores 8/9 = 0.889 and is BELOW 0.9, so
    both rows are kept. The 0.9 threshold demands near-identity on short text.
    """
    df = _frame(
        [
            "cape bancorp cbnj ceo michael devlin buys 1081 shares",
            "cape bancorp cbnj ceo michael devlin buys 1832 shares",
        ],
        ["2015-01-05", "2015-01-05"],
    )
    assert sdata.near_duplicate_keep(df["text_norm"], df["ts_utc"]).all()


def test_near_duplicate_keeps_genuinely_different_headlines():
    df = _frame(
        [
            "apple beats on earnings and raises guidance",
            "microsoft misses on revenue and cuts outlook",
        ],
        ["2015-01-05", "2015-01-05"],
    )
    assert sdata.near_duplicate_keep(df["text_norm"], df["ts_utc"]).all()


def test_near_duplicate_respects_the_time_window():
    """The same template outside the window is a fresh headline, not a repeat."""
    df = _frame(
        [
            "cape bancorp cbnj ceo michael devlin buys 1081 shares",
            "cape bancorp cbnj ceo michael devlin buys 1832 shares",
        ],
        ["2015-01-05", "2015-01-20"],
    )
    assert sdata.near_duplicate_keep(df["text_norm"], df["ts_utc"], window_days=3).all()


def test_dedup_reports_the_signature_limit_exposure():
    """The blocking's known blind spot is quoted, not assumed away.

    At overlap 0.9 a two-token difference needs >= 20 unique tokens, so only
    rows that long are exposed.
    """
    long_text = " ".join(f"token{i}" for i in range(40))
    mid_text = " ".join(f"word{i}" for i in range(12))
    df = _frame(
        ["short headline here", mid_text, long_text],
        ["2015-01-05", "2015-01-05", "2015-01-05"],
    )
    _, stats = sdata.dedup(df)
    assert stats["share_above_signature_limit"] == pytest.approx(1 / 3)


def test_dedup_removes_the_cross_ticker_repeat():
    """One roundup emitted once per tagged ticker must count once."""
    texts = ["stocks that hit 52 week highs on friday"] * 8 + ["apple beats on earnings"]
    df = _frame(texts, ["2015-01-05"] * 9)
    out, stats = sdata.dedup(df)
    assert len(out) == 2
    assert stats["n_exact_dropped"] == 7
    assert stats["dedup_rate"] == pytest.approx(7 / 9)


def test_window_end_is_inclusive_of_its_last_day_and_nothing_more(tmp_path):
    """A date-only corpus makes an off-by-one here worth a whole extra day.

    Stamps sit at 00:00 UTC, so an inclusive `<= end + 1 day` bound admits the
    next day's midnight stamp. The bound must be strict.
    """
    p = tmp_path / "edge.csv"
    p.write_text(
        "Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,"
        "Lsa_summary,Luhn_summary,Textrank_summary,Lexrank_summary\n"
        "2019-12-30 00:00:00 UTC,inside the window,A,https://www.benzinga.com/1,B,,,,,,\n"
        "2019-12-31 00:00:00 UTC,the last day itself,A,https://www.benzinga.com/2,B,,,,,,\n"
        "2020-01-01 00:00:00 UTC,one day too far,A,https://www.benzinga.com/3,B,,,,,,\n",
        encoding="utf-8",
    )
    df = sdata.load_news(p, domains=("benzinga.com",), start="2010-01-01", end="2019-12-31")
    assert len(df) == 2
    assert df["ts_utc"].max().strftime("%Y-%m-%d") == "2019-12-31"
    assert "one day too far" not in set(df["text"])


# --------------------------------------------------------------------------
# R03b: the dedup lineage contract's acceptance tests (docs/archive/dedup-lineage-contract.md
# section 5). Each corresponds to a defect reproduced against the running code
# in R03a and quantified on the real corpus.
# --------------------------------------------------------------------------


def _clustered_frame():
    """One story filed under three tickers, all on the same date.

    A date-only corpus stamps every headline on a date at 00:00 UTC, so all
    three rows tie on `ts_utc`. This is the ordinary case here, not an edge one.
    """
    return _frame(
        ["Fed holds rates steady"] * 3,
        ["2015-03-18"] * 3,
        tickers=[["SPY"], ["QQQ"], ["DIA"]],
    )


def test_dedup_output_is_invariant_to_input_order():
    """A12/D-2. The survivor must be a function of the input SET.

    Before R03b, `dedup` sorted on `ts_utc` alone with pandas' default
    quicksort, which is not stable, so the survivor of a tied group was
    whichever row arrived first. On the real corpus, permuting the input changed
    2,363 surviving ids and the retained tickers of 10.78% of survivors.
    """
    import itertools

    base = _clustered_frame()
    seen = set()
    for perm in itertools.permutations(range(3)):
        shuffled = base.iloc[list(perm)].reset_index(drop=True)
        # source_row_id travels WITH the row; only the frame order changes.
        out, _ = sdata.dedup(shuffled)
        assert len(out) == 1
        seen.add(
            (
                out["source_row_id"].iloc[0],
                tuple(out["tickers"].iloc[0]),
                out["cluster_id"].iloc[0],
                int(out["n_cluster_rows"].iloc[0]),
            )
        )
    assert len(seen) == 1, f"output depends on input order: {seen}"
    # The stated rule picks the smallest source_row_id among the timestamp ties.
    assert next(iter(seen))[0] == 0


def test_dedup_unions_ticker_tags_across_the_cluster():
    """A12/D-3. 75.7% of distinct (cluster, ticker) pairs were being destroyed."""
    out, _ = sdata.dedup(_clustered_frame())
    assert len(out) == 1
    assert list(out["tickers"].iloc[0]) == ["DIA", "QQQ", "SPY"]
    assert int(out["n_cluster_rows"].iloc[0]) == 3


def test_exact_window_reanchors_on_the_last_kept_occurrence():
    """A12/D-1, the audit's exact probe.

    Days 0, 1, 10, 11. Days 10 and 11 are an identical pair one day apart. The
    old window was measured from day 0, so both cleared it: day 11 was then
    removed by the NEAR pass and counted as a near duplicate, and with
    `near_dupe=False` it survived outright. On the real corpus this
    misattributed 107,485 rows -- 96.2% of the reported near count.
    """
    df = _frame(
        ["Apple beats on earnings"] * 4,
        ["2015-01-05", "2015-01-06", "2015-01-15", "2015-01-16"],
    )
    for near_dupe in (False, True):
        out, stats = sdata.dedup(df.copy(), near_dupe=near_dupe)
        kept = list(out["ts_utc"].dt.strftime("%Y-%m-%d"))
        assert kept == ["2015-01-05", "2015-01-15"], (near_dupe, kept)
        assert stats["n_exact_dropped"] == 2
        assert stats["n_near_dropped"] == 0, "an exact repeat is not a near one"


def test_exact_and_near_are_attributed_to_their_own_relation():
    """The two counts must each describe what they name."""
    # Ten unique tokens, one replaced: 9/10 = 0.90, exactly at the threshold.
    ten = "apple beats on earnings and raises its full year outlook"
    near = "apple beats on earnings and raises its full year guidance"
    df = _frame(
        [ten, ten, near, "microsoft cuts its dividend"],
        ["2015-01-05"] * 4,
    )
    out, stats, lineage = sdata.dedup(df, return_lineage=True)
    assert stats["n_exact_dropped"] == 1
    assert stats["n_near_dropped"] == 1
    assert set(lineage.loc[~lineage["kept"], "relation"]) == {"exact", "near"}
    assert len(out) == 2


def test_signature_limit_boundary_is_twenty_tokens_at_the_configured_threshold():
    """A12/D-4. `ceil(2/(1-0.90))` returns 21 in binary floating point.

    A 20-unique-token pair differing by two tokens scores exactly 18/20 = 0.90
    and clears the threshold, so 20-token rows ARE exposed. The float form
    excluded 14,494 real rows from the quoted figure.
    """
    assert sdata._missable_token_length(0.90) == 20
    assert sdata._missable_token_length(0.95) == 40
    assert sdata._missable_token_length(0.80) == 10

    exactly_twenty = " ".join(f"tok{i:02d}" for i in range(20))
    nineteen = " ".join(f"word{i:02d}" for i in range(19))
    df = _frame([exactly_twenty, nineteen], ["2015-01-05", "2015-01-06"])
    _, stats = sdata.dedup(df)
    assert stats["min_len_missable"] == 20
    assert stats["share_above_signature_limit"] == pytest.approx(0.5)


def test_lineage_accounts_for_every_input_row():
    """The dedup rate must be recomputable from the artifact, not only re-run."""
    df = _frame(
        ["Apple beats on earnings"] * 3 + ["Microsoft cuts its outlook"],
        ["2015-01-05"] * 3 + ["2015-01-06"],
        tickers=[["AAPL"], ["AAPL"], ["SPY"], ["MSFT"]],
    )
    out, stats, lineage = sdata.dedup(df, return_lineage=True)

    assert list(lineage.columns) == sdata.LINEAGE_COLUMNS
    assert len(lineage) == len(df)
    assert set(lineage["source_row_id"]) == set(df["source_row_id"])

    # Every eliminated row names a survivor, and every cluster resolves to one.
    kept_ids = set(out["source_row_id"])
    dropped = lineage[~lineage["kept"]]
    assert len(dropped) == len(df) - len(out)
    # `eliminated_by` records the IMMEDIATE eliminator, which need not itself
    # have survived: an exact repeat's anchor can later be removed as a
    # near-duplicate of an earlier headline, so elimination forms short chains.
    # `cluster_id` is the resolved root, and that is what must be a survivor.
    assert set(dropped["eliminated_by"]).issubset(set(df["source_row_id"]))
    kept_cluster_ids = set(out["cluster_id"])
    assert set(dropped["cluster_id"]).issubset(kept_cluster_ids)
    assert lineage.loc[lineage["kept"], "eliminated_by"].isna().all()
    assert lineage.loc[lineage["kept"], "relation"].isna().all()
    assert set(lineage["cluster_id"]).issubset(kept_cluster_ids)

    # The rate recomputed from lineage alone equals the reported one.
    assert 1 - lineage["kept"].mean() == pytest.approx(stats["dedup_rate"])


def test_dedup_preserves_headline_id_uniqueness_for_the_r04b_gate():
    """R04b requires unique, non-null ids on both sides of the score join."""
    df = _frame(
        ["Apple beats on earnings"] * 3 + ["Microsoft cuts its outlook"],
        ["2015-01-05"] * 3 + ["2015-01-06"],
    )
    out, _ = sdata.dedup(df)
    assert out["headline_id"].is_unique
    assert out["headline_id"].notna().all()


def test_representative_text_is_chosen_by_the_stated_rule():
    """Cluster members share `text_norm` but may differ in raw `text`.

    The representative's raw text is what gets scored, so which row wins is a
    measurement decision, not a formatting detail.
    """
    df = _frame(
        ["APPLE BEATS ON EARNINGS", "Apple beats on earnings"],
        ["2015-01-05", "2015-01-05"],
    )
    df["text_norm"] = "apple beats on earnings"
    # Give the lower-cased row the smaller id, so the rule must select it.
    df["source_row_id"] = [1, 0]
    out, _ = sdata.dedup(df)
    assert len(out) == 1
    assert out["text"].iloc[0] == "Apple beats on earnings"
    assert out["source_row_id"].iloc[0] == 0


def test_dedup_refuses_a_frame_without_source_row_id():
    """The strict schema is what keeps an unnumbered frame off the corpus path."""
    df = _frame(["Apple beats on earnings"], ["2015-01-05"]).drop(columns=["source_row_id"])
    with pytest.raises(ValueError, match="source_row_id"):
        sdata.dedup(df)


# ------------------------------------- R12 follow-up: the source filter boundary


def test_the_source_filter_matches_exact_hosts_and_true_subdomains():
    assert sdata._host_matches("benzinga.com", ("benzinga.com",))
    assert sdata._host_matches("www2.benzinga.com", ("benzinga.com",))
    assert sdata._host_matches("BENZINGA.COM", ("benzinga.com",))
    assert sdata._host_matches("benzinga.com.", ("benzinga.com",))   # trailing dot


def test_the_source_filter_rejects_substring_impostors():
    """The defect this closes: `any(d in h ...)` admitted both of these.

    The filter is the corpus boundary, not hygiene -- `All_external.csv`
    concatenates five-plus corpora including Russian-language `lenta.ru`, so a
    leak is a different corpus rather than a little extra noise.
    """
    for host in ("notbenzinga.com", "benzinga.com.evil.example",
                 "xbenzinga.com", "fakebenzinga.community"):
        assert not sdata._host_matches(host, ("benzinga.com",)), host


def test_the_source_filter_still_admits_every_pooled_candidate():
    for host in ("benzinga.com", "zacks.com", "seekingalpha.com"):
        assert sdata._host_matches(host, config.NEWS_SOURCE_DOMAINS_POOLED)
    assert not sdata._host_matches("lenta.ru", config.NEWS_SOURCE_DOMAINS_POOLED)


def test_an_empty_or_missing_host_is_not_admitted():
    assert not sdata._host_matches("", ("benzinga.com",))
