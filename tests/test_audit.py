"""Stage 0 audit checks, on synthetic candidates with known properties.

The point of these tests is the gate: a date-only dataset must be *caught*, not
scored well on coverage and waved through. So one synthetic candidate is a
genuine intraday feed and the other is a date dump wearing a timestamp column,
and the audit has to tell them apart and pick the right one.
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

CALENDAR = pd.DatetimeIndex(pd.bdate_range("2020-01-01", "2023-12-29"))


def _make(stamps, texts=None, tickers=None) -> pd.DataFrame:
    stamps = pd.Series(pd.to_datetime(stamps))
    n = len(stamps)
    texts = texts or [f"headline number {i}" for i in range(n)]
    return pd.DataFrame(
        {
            "headline_id": [f"h{i:06d}" for i in range(n)],
            "text": texts,
            "text_norm": [t.lower() for t in texts],
            "ts_utc": stamps.dt.tz_convert("UTC"),
            "ts_et": stamps,
            "tickers": tickers if tickers is not None else [["SPY"] for _ in range(n)],
        }
    )


@pytest.fixture(scope="module")
def intraday():
    """A believable feed: ~20 headlines per weekday, spread across the clock."""
    rng = np.random.default_rng(config.SEED)
    stamps = []
    for day in CALENDAR:
        for _ in range(20):
            minute = int(rng.integers(4 * 60, 22 * 60))
            stamps.append(
                pd.Timestamp(day, tz=config.TZ_MARKET) + pd.Timedelta(minutes=minute)
            )
    return _make(pd.Series(stamps).sort_values().reset_index(drop=True))


@pytest.fixture(scope="module")
def date_only():
    """The trap: a date column parsed to datetimes, so every stamp is midnight."""
    stamps = []
    for day in CALENDAR:
        for _ in range(20):
            stamps.append(pd.Timestamp(day, tz=config.TZ_MARKET))
    return _make(pd.Series(stamps))


# ---------------------------------------------------------------- criterion 1


def test_intraday_feed_is_recognised(intraday):
    p = audit.timestamp_profile(intraday)
    assert p["looks_date_only"] is False
    assert p["distinct_minutes_of_day"] >= audit.MIN_DISTINCT_MINUTES
    assert p["modal_share"] < audit.DATE_ONLY_SHARE
    assert "INTRADAY" in p["verdict"]


def test_date_only_dump_is_caught(date_only):
    p = audit.timestamp_profile(date_only)
    assert p["looks_date_only"] is True
    assert p["midnight_share"] == pytest.approx(1.0)
    assert "D2" in p["verdict"]


def test_almost_date_only_is_still_caught():
    """95% at one clock time and a 5% intraday tail must not read as intraday."""
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2021-03-01", tz=config.TZ_MARKET)
    stamps = [base + pd.Timedelta(days=int(i % 300)) for i in range(1900)]
    stamps += [
        base + pd.Timedelta(days=int(i % 300), minutes=int(rng.integers(1, 1400)))
        for i in range(100)
    ]
    p = audit.timestamp_profile(_make(pd.Series(stamps).sort_values().reset_index(drop=True)))
    assert p["looks_date_only"] is True


def test_naive_timestamps_are_rejected(intraday):
    naive = intraday.copy()
    naive["ts_et"] = naive["ts_et"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        audit.timestamp_profile(naive)


def test_session_shares_sum_to_one(intraday):
    p = audit.timestamp_profile(intraday)
    total = p["share_before_open"] + p["share_in_session"] + p["share_after_close"]
    assert total == pytest.approx(1.0)


# ---------------------------------------------------------------- criterion 2


def test_coverage_counts_sessions_and_zero_news_days(intraday):
    c = audit.coverage_profile(intraday, CALENDAR)
    assert c["n_sessions"] == len(CALENDAR)
    assert c["per_day_median"] > 0
    # Every session got headlines, but the last session's post-close news is
    # unassignable and must be reported, not silently dropped.
    assert c["zero_news_days"] < len(CALENDAR)
    assert c["n_unassignable"] >= 0


def test_coverage_flags_a_gap():
    """A dataset that stops for a year must show up as zero-news days."""
    days = CALENDAR[(CALENDAR < "2021-01-01") | (CALENDAR >= "2022-01-01")]
    stamps = [pd.Timestamp(d, tz=config.TZ_MARKET) + pd.Timedelta(hours=10) for d in days]
    c = audit.coverage_profile(_make(pd.Series(stamps)), CALENDAR)
    assert c["zero_news_share"] > 0.2


def test_duplication_profile_reports_a_rate():
    texts = ["apple beats on earnings"] * 40 + [f"unique story {i}" for i in range(60)]
    stamps = [
        pd.Timestamp("2021-06-01", tz=config.TZ_MARKET) + pd.Timedelta(hours=i % 48)
        for i in range(100)
    ]
    stats = audit.duplication_profile(_make(pd.Series(stamps), texts=texts))
    assert stats["n_in"] == 100
    assert stats["dedup_rate"] > 0.3
    assert stats["n_exact_dropped"] > 0


def test_ticker_sanity_returns_a_readable_sample(intraday):
    sample = audit.ticker_sanity(intraday, n=50)
    assert len(sample) == 50
    assert list(sample.columns) == ["text", "ts_et", "tickers", "n_tickers"]
    assert sample.attrs["untagged_share"] == 0.0
    assert sample.attrs["most_covered"].index[0] == "SPY"


def test_ticker_sanity_is_seeded(intraday):
    a = audit.ticker_sanity(intraday, n=25)
    b = audit.ticker_sanity(intraday, n=25)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- criterion 3


def test_suggest_window_finds_the_stable_run(intraday):
    w = audit.suggest_window(intraday, min_headlines_per_year=1000)
    assert w["start"] == "2020-01-01"
    assert w["end"] == "2023-12-31"
    assert w["years"] == 4


def test_suggest_window_excludes_a_thin_ramp_up_year():
    """Coverage that ramps from nothing must not drag the window backwards."""
    stamps = []
    for year, count in ((2018, 50), (2019, 60), (2020, 6000), (2021, 6200), (2022, 5900)):
        for i in range(count):
            stamps.append(
                pd.Timestamp(f"{year}-01-01", tz=config.TZ_MARKET) + pd.Timedelta(hours=i)
            )
    w = audit.suggest_window(_make(pd.Series(stamps)), min_headlines_per_year=1000)
    assert w["start"] == "2020-01-01"
    assert 2018 in w["excluded_years"] and 2019 in w["excluded_years"]


def test_suggest_window_reports_failure_rather_than_guessing():
    stamps = [pd.Timestamp("2021-01-01", tz=config.TZ_MARKET) + pd.Timedelta(hours=i)
              for i in range(10)]
    w = audit.suggest_window(_make(pd.Series(stamps)), min_headlines_per_year=5000)
    assert w["ok"] is False and "reason" in w


# ---------------------------------------------------------------- the choice


def test_comparison_picks_the_intraday_candidate(intraday, date_only):
    audits = {
        "benzinga": audit.audit_candidate(date_only, "benzinga", CALENDAR),
        "fnspid": audit.audit_candidate(intraday, "fnspid", CALENDAR),
    }
    table = audit.compare_candidates(audits)
    assert table.attrs["recommendation"] == "fnspid"
    assert table.loc["benzinga", "intraday"] == False  # noqa: E712
    assert "intraday timestamps confirmed" in table.attrs["reason"]


def test_intraday_gate_beats_better_coverage(intraday, date_only):
    """The gate is not a score: a date-only set with cleaner data still loses."""
    audits = {
        "date_only_but_clean": audit.audit_candidate(date_only, "date_only_but_clean", CALENDAR),
        "intraday": audit.audit_candidate(intraday, "intraday", CALENDAR),
    }
    assert audit.compare_candidates(audits).attrs["recommendation"] == "intraday"


def test_all_candidates_date_only_recommends_the_d2_fallback(date_only):
    audits = {
        "a": audit.audit_candidate(date_only, "a", CALENDAR),
        "b": audit.audit_candidate(date_only, "b", CALENDAR),
    }
    table = audit.compare_candidates(audits)
    assert table.attrs["recommendation"] == "D2"
    assert "RQ2" in table.attrs["reason"]
