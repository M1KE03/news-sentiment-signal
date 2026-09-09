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
         "ret": rng.normal(0, 0.01, n), "parkinson": rng.uniform(1e-5, 5e-4, n),
         "volume": rng.uniform(5e7, 2e8, n), "log_turnover": rng.normal(18.4, 0.2, n),
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
