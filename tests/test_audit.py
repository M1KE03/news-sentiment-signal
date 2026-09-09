"""Audit tests, on a synthetic corpus with FNSPID's actual failure modes.

The fixture deliberately reproduces the three traps the real B03 audit hit:

  * a concatenated file whose pooled timestamp profile looks nothing like any
    of its parts;
  * an intraday source that passes every mechanical check and is still the
    wrong content;
  * a non-English source hiding inside a financial dataset.

The tests assert that the module separates sources, refuses to produce
corpus-wide rates from a cluster sample, and does not pretend to judge
relevance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import audit

RNG = np.random.default_rng(config.SEED)
DAYS = pd.bdate_range("2015-01-01", "2019-12-31")


def _rows(source_url, titles, stamps, ticker=True):
    n = len(stamps)
    return pd.DataFrame(
        {
            "Date": [f"{t:%Y-%m-%d %H:%M:%S} UTC" for t in stamps],
            "Article_title": [titles[i % len(titles)] for i in range(n)],
            "Stock_symbol": (["AAPL"] * n) if ticker else [None] * n,
            "Url": [f"https://www.{source_url}/story/{i}" for i in range(n)],
        }
    )


@pytest.fixture(scope="module")
def corpus():
    """Three sources concatenated, mirroring the real file's structure."""
    # (a) tagged US-equity source, DATE-ONLY -- like FNSPID's Benzinga block
    bz = _rows("benzinga.com",
               ["Apple beats on earnings", "Stocks That Hit 52-Week Highs On Friday"],
               [pd.Timestamp(d, tz="UTC") for d in DAYS for _ in range(4)])
    # (b) untagged wire, GENUINELY INTRADAY -- like FNSPID's Reuters block
    rt = _rows("reuters.com",
               ["Fashion comes first at Royal Ascot Ladies' Day",
                "REG-Baillie Gifford Japan - Net Asset Value(s)"],
               [pd.Timestamp(d, tz="UTC") + pd.Timedelta(minutes=int(RNG.integers(0, 1440)))
                for d in DAYS for _ in range(6)],
               ticker=False)
    # (c) Russian-language general news, date-only -- like lenta.ru
    ln = _rows("lenta.ru",
               ["Китай раскрыл планы по исследованию Луны",
                "«Зеленый день» заменит в России «Черную пятницу»"],
               [pd.Timestamp(d, tz="UTC") for d in DAYS[:600] for _ in range(3)],
               ticker=False)
    return pd.concat([bz, rt, ln], ignore_index=True)


@pytest.fixture(scope="module")
def loaded(corpus, tmp_path_factory):
    p = tmp_path_factory.mktemp("audit") / "sample.parquet"
    # Inject the malformed rows that embedded newlines produce in the real file.
    bad = pd.DataFrame({
        "Date": ["with the matter.  Bakrie & Brothers", "$1.34 billion bridge loan"],
        "Article_title": ["frag", "frag"], "Stock_symbol": [None, None],
        "Url": [None, None],
    })
    pd.concat([corpus, bad], ignore_index=True).to_parquet(p, index=False)
    return audit.load_sample(p)


# ----------------------------------------------------------------- loading


def test_load_drops_and_counts_malformed_dates(loaded):
    df, stats = loaded
    assert stats["n_dropped_malformed_date"] == 2
    assert stats["n_kept"] == stats["n_read"] - 2
    assert any("Bakrie" in e for e in stats["malformed_examples"])


def test_load_classifies_sources(loaded):
    df, _ = loaded
    assert set(df["source"].unique()) == {"benzinga", "reuters", "lenta.ru"}


def test_load_makes_timestamps_tz_aware(loaded):
    df, _ = loaded
    assert df["ts_utc"].dt.tz is not None and df["ts_et"].dt.tz is not None


# ------------------------------------------- the pooling trap, per source


def test_pooled_profile_would_have_been_misleading(loaded):
    """The whole point of the rework: the file's pooled view is not any source's."""
    df, _ = loaded
    pooled = audit.timestamp_profile(df["ts_utc"], df["ts_et"])
    per_source = audit.profile_by_source(df)

    # Pooled, the corpus looks intraday because Reuters dominates the row count.
    assert pooled["looks_date_only"] is False
    # Per source, two of the three are date-only -- including the only usable one.
    assert per_source.loc["benzinga", "looks_date_only"]
    assert per_source.loc["lenta.ru", "looks_date_only"]
    assert not per_source.loc["reuters", "looks_date_only"]


def test_profile_by_source_separates_the_three_regimes(loaded):
    df, _ = loaded
    p = audit.profile_by_source(df)
    assert p.loc["benzinga", "midnight_share"] == pytest.approx(1.0)
    assert p.loc["reuters", "distinct_minutes"] > audit.MIN_DISTINCT_MINUTES
    assert p.loc["reuters", "midnight_share"] < 0.05
    assert p.loc["benzinga", "ticker_share"] == pytest.approx(1.0)
    assert p.loc["reuters", "ticker_share"] == pytest.approx(0.0)


def test_non_english_source_is_flagged(loaded):
    df, _ = loaded
    p = audit.profile_by_source(df)
    assert p.loc["lenta.ru", "likely_non_english"]
    assert p.loc["lenta.ru", "cyrillic_share"] > 0.9
    assert not p.loc["benzinga", "likely_non_english"]
    assert not p.loc["reuters", "likely_non_english"]


def test_timestamp_profile_rejects_naive_input(loaded):
    df, _ = loaded
    with pytest.raises(ValueError):
        audit.timestamp_profile(df["ts_et"].dt.tz_localize(None))


def test_session_shares_sum_to_one(loaded):
    df, _ = loaded
    g = df[df.source == "reuters"]
    p = audit.timestamp_profile(g["ts_utc"], g["ts_et"])
    assert p["share_before_open"] + p["share_in_session"] + p["share_after_close"] == pytest.approx(1.0)


def test_midnight_share_by_year_detects_a_regime_change():
    """A source that switches instrument mid-history must show up per year."""
    stamps = []
    for d in pd.bdate_range("2018-01-01", "2018-12-31"):
        stamps += [pd.Timestamp(d, tz="UTC")] * 5
    for d in pd.bdate_range("2019-01-01", "2019-12-31"):
        stamps += [pd.Timestamp(d, tz="UTC") + pd.Timedelta(minutes=int(RNG.integers(1, 1400)))] * 5
    df = _rows("benzinga.com", ["headline"], stamps)
    df["ts_utc"] = pd.to_datetime(df["Date"].str.removesuffix(" UTC"), utc=True)
    df["source"] = "benzinga"
    piv = audit.midnight_share_by_year(df)
    assert piv.loc[2018, "benzinga"] == pytest.approx(1.0)
    assert piv.loc[2019, "benzinga"] < 0.05


# ------------------------------------------- cluster-sample rate withholding


def test_corpus_rates_are_withheld_on_a_cluster_sample(loaded):
    df, _ = loaded
    cov = audit.coverage_profile(df, calendar=pd.DatetimeIndex(DAYS), clustered=True)
    for name in ("per_session_mean", "zero_news_share", "thin_days_share"):
        rate = cov[name]
        assert not rate.available
        assert "cluster sample" in rate.withheld_reason
        assert "WITHHELD" in repr(rate)


def test_corpus_rates_are_computed_when_representative(loaded):
    df, _ = loaded
    cov = audit.coverage_profile(df[df.source == "benzinga"],
                                 calendar=pd.DatetimeIndex(DAYS), clustered=False)
    assert cov["per_session_mean"].available
    assert cov["per_session_mean"].value > 0
    assert 0.0 <= cov["zero_news_share"].value <= 1.0


def test_span_and_per_year_are_always_available(loaded):
    """Facts that survive clustering must not be withheld along with the rates."""
    df, _ = loaded
    cov = audit.coverage_profile(df, clustered=True)
    assert cov["n_headlines"] > 0
    assert not cov["per_year"].empty
    assert cov["first"] < cov["last"]


def test_duplication_rate_is_withheld_but_examples_survive(loaded):
    df, _ = loaded
    dup = audit.duplication_profile(df, clustered=True)
    assert not dup["dedup_rate"].available
    # The structural evidence is still useful and is still returned.
    assert dup["most_repeated"].iloc[0] > 1
    assert dup["share_in_repeated_texts"] > 0.5


def test_duplication_rate_computed_when_representative(loaded):
    df, _ = loaded
    dup = audit.duplication_profile(df[df.source == "benzinga"].head(200), clustered=False)
    assert dup["dedup_rate"].available


# ------------------------------------------------------------- screening


def test_screen_reports_the_split_between_timestamps_and_relevance(loaded):
    df, _ = loaded
    screen = audit.screen_sources(audit.profile_by_source(df))
    assert screen.passes_intraday == ["reuters"]
    assert "reuters" not in screen.passes_tagging
    assert "benzinga" in screen.passes_tagging
    joined = " ".join(screen.notes)
    assert "carry no ticker tags" in joined


def test_screen_flags_the_non_english_source(loaded):
    df, _ = loaded
    screen = audit.screen_sources(audit.profile_by_source(df))
    assert "lenta.ru" not in screen.passes_language
    assert "NON-ENGLISH" in " ".join(screen.notes)


def test_screen_never_selects_a_universe(loaded):
    """Relevance is a human judgement; the screen must say so and not guess."""
    df, _ = loaded
    screen = audit.screen_sources(audit.profile_by_source(df))
    assert not hasattr(screen, "recommendation")
    assert "RELEVANCE IS NOT SCREENED HERE" in " ".join(screen.notes)


def test_screen_reports_the_fallback_when_nothing_is_intraday(loaded):
    df, _ = loaded
    only_date_only = df[df.source != "reuters"]
    screen = audit.screen_sources(audit.profile_by_source(only_date_only))
    assert screen.passes_intraday == []
    assert "date-only fallback" in " ".join(screen.notes)


# ------------------------------------------------------------ window / misc


def test_suggest_window_is_always_marked_provisional(loaded):
    df, _ = loaded
    w = audit.suggest_window(df[df.source == "benzinga"], min_per_year=500)
    assert w["provisional"] is True
    assert w["start"] == "2015-01-01" and w["end"] == "2019-12-31"


def test_suggest_window_excludes_a_thin_ramp_up_year():
    stamps = []
    for year, count in ((2016, 40), (2017, 3000), (2018, 3100), (2019, 2900)):
        stamps += [pd.Timestamp(f"{year}-01-01", tz="UTC") + pd.Timedelta(hours=i)
                   for i in range(count)]
    df = _rows("benzinga.com", ["h"], stamps)
    df["ts_utc"] = pd.to_datetime(df["Date"].str.removesuffix(" UTC"), utc=True)
    w = audit.suggest_window(df, ts_col="ts_utc", min_per_year=500)
    assert w["start"] == "2017-01-01" and 2016 in w["excluded_years"]


def test_content_sample_is_seeded_and_bounded(loaded):
    df, _ = loaded
    a = audit.content_sample(df, "reuters", n=10)
    b = audit.content_sample(df, "reuters", n=10)
    assert a == b and len(a) == 10
    assert audit.content_sample(df, "nonexistent") == []


def test_hour_histogram_covers_all_24_hours(loaded):
    df, _ = loaded
    h = audit.hour_histogram(df[df.source == "reuters"])
    assert list(h.index) == list(range(24))
    assert h.sum() == (df.source == "reuters").sum()
