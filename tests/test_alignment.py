"""The look-ahead firewall.

D6 says a headline stamped s belongs to trading day t iff s is in
(close(t-1), close(t)], with close = 16:00 ET. Every case below is one way that
rule can be got wrong, and the last test is the one that matters: no row of the
panel at date t may have been built from a headline stamped after close(t).

These tests need no dataset. They run on a synthetic calendar and synthetic
headlines, which is the point -- the firewall must be checkable before there is
any data to be wrong about.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import align


# A week with a Thursday holiday, so the tests cover weekend and holiday rolls.
#   Mon 2024-07-01, Tue 07-02, Wed 07-03, [Thu 07-04 holiday], Fri 07-05,
#   Mon 07-08, Tue 07-09
CALENDAR = pd.DatetimeIndex(
    ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08", "2024-07-09"]
)


def et(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz=config.TZ_MARKET)


def assign(stamp: str) -> pd.Timestamp:
    return align.map_to_trading_day(pd.Series([et(stamp)]), CALENDAR).iloc[0]


def test_before_close_belongs_to_same_day():
    assert assign("2024-07-02 15:59") == pd.Timestamp("2024-07-02")


def test_after_close_rolls_to_next_trading_day():
    assert assign("2024-07-02 16:01") == pd.Timestamp("2024-07-03")


def test_exactly_at_close_belongs_to_that_day():
    # The interval is half-open on the right: s == close(t) is day t.
    assert assign("2024-07-02 16:00") == pd.Timestamp("2024-07-02")


def test_overnight_belongs_to_next_session():
    assert assign("2024-07-03 06:30") == pd.Timestamp("2024-07-03")


def test_saturday_rolls_to_monday():
    assert assign("2024-07-06 10:00") == pd.Timestamp("2024-07-08")


def test_sunday_rolls_to_monday():
    assert assign("2024-07-07 23:30") == pd.Timestamp("2024-07-08")


def test_holiday_rolls_to_next_trading_day():
    # 2024-07-04 is a market holiday; it is absent from CALENDAR.
    assert assign("2024-07-04 11:00") == pd.Timestamp("2024-07-05")


def test_after_last_session_is_unassignable():
    # Must be NaT, not clipped onto the final session -- clipping would silently
    # invent a day's news out of the future.
    assert pd.isna(assign("2024-07-09 16:30"))


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError):
        align.map_to_trading_day(pd.Series([pd.Timestamp("2024-07-02 10:00")]), CALENDAR)


def test_utc_input_is_converted_not_reinterpreted():
    # 20:00 UTC on 2024-07-02 is 16:00 EDT -- the boundary, so day 07-02.
    ts = pd.Series([pd.Timestamp("2024-07-02 20:00", tz="UTC")])
    assert align.map_to_trading_day(ts, CALENDAR).iloc[0] == pd.Timestamp("2024-07-02")
    ts = pd.Series([pd.Timestamp("2024-07-02 20:01", tz="UTC")])
    assert align.map_to_trading_day(ts, CALENDAR).iloc[0] == pd.Timestamp("2024-07-03")


# --------------------------------------------------------------------------
# The firewall assertion: nothing in the panel at date t saw the future.
# --------------------------------------------------------------------------


def _synthetic_headlines(n_per_day: int = 7, seed: int = config.SEED):
    rng = np.random.default_rng(seed)
    stamps = []
    for day in CALENDAR:
        base = pd.Timestamp(day, tz=config.TZ_MARKET)
        for _ in range(n_per_day):
            # Spread across the 24h ending at that day's close, plus some after
            # it, so both sides of the boundary are exercised.
            offset = pd.Timedelta(minutes=int(rng.integers(-600, 900)))
            stamps.append(base + pd.Timedelta(hours=12) + offset)
    headlines = pd.DataFrame(
        {
            "headline_id": [f"h{i:04d}" for i in range(len(stamps))],
            "ts_et": pd.Series(stamps),
        }
    )
    scores = pd.DataFrame(
        {"headline_id": headlines["headline_id"]}
        | {f"score_{s}": rng.uniform(-1, 1, len(headlines)) for s in config.SCORERS}
    )
    return headlines, scores


def test_no_headline_after_close_contributes_to_that_day():
    headlines, scores = _synthetic_headlines()
    headlines = headlines.copy()
    headlines["assigned"] = align.map_to_trading_day(headlines["ts_et"], CALENDAR)
    used = headlines.dropna(subset=["assigned"])

    assigned_close = align.session_closes(pd.DatetimeIndex(used["assigned"]))
    violations = used["ts_et"].to_numpy() > assigned_close.to_numpy()
    assert not violations.any(), (
        f"{int(violations.sum())} headline(s) assigned to a day whose close they "
        "postdate -- this is the look-ahead bug the panel exists to prevent"
    )


def test_aggregate_daily_respects_the_dispersion_floor():
    headlines, scores = _synthetic_headlines(n_per_day=3)
    daily = align.aggregate_daily(scores, headlines, CALENDAR)
    thin = daily["n_headlines"] < config.MIN_HEADLINES_FOR_DISPERSION
    assert daily.loc[thin, [f"d_{s}" for s in config.SCORERS]].isna().all().all()


def test_aggregate_daily_covers_every_session():
    headlines, scores = _synthetic_headlines()
    daily = align.aggregate_daily(scores, headlines, CALENDAR)
    assert list(daily["date"]) == list(CALENDAR)
    assert daily["n_headlines"].ge(0).all()


def test_leads_are_pure_shifts_and_never_leak():
    headlines, scores = _synthetic_headlines()
    daily = align.aggregate_daily(scores, headlines, CALENDAR)
    market = pd.DataFrame(
        {
            "date": CALENDAR,
            "close_adj": np.linspace(100, 105, len(CALENDAR)),
            "ret": np.linspace(0.001, 0.006, len(CALENDAR)),
            "parkinson": np.linspace(1e-4, 6e-4, len(CALENDAR)),
            "volume": np.linspace(1e8, 1.5e8, len(CALENDAR)),
            "log_turnover": np.log(np.linspace(1e8, 1.5e8, len(CALENDAR))),
            "vix_close": np.linspace(12, 18, len(CALENDAR)),
        }
    )
    panel = align.build_panel(daily, market)

    for h in config.HORIZONS:
        lead = panel[f"ret_lead{h}"]
        expected = panel["ret"].shift(-h)
        pd.testing.assert_series_equal(lead, expected, check_names=False)
        # The last h rows must be missing, not filled: there is no future there.
        assert lead.tail(h).isna().all()

    assert list(panel.columns) == align.PANEL_COLUMNS


# --------------------------------------------------------------------------
# B07/B08: calendar edges and the wrong-mapper guard.
# --------------------------------------------------------------------------


def test_pre_window_headlines_are_dropped_not_piled_on_the_first_session():
    """B05 defect D-1, as a regression test.

    The first session's window opens at the *previous* session's close, which
    lies outside the calendar. Without that bound every earlier headline
    searchsorted to index 0 and landed on the first session -- five years of
    pre-window news on one day.
    """
    stamps = pd.Series(
        [
            et("2019-03-01 10:00"),   # years before the calendar
            et("2024-06-14 10:00"),   # weeks before
            et("2024-06-28 15:00"),   # the session before the calendar starts
            et("2024-07-01 15:00"),   # inside the first session's day
            et("2024-07-02 15:00"),   # comfortably inside the calendar
        ]
    )
    got = align.map_to_trading_day(stamps, CALENDAR)
    assert got.iloc[:4].isna().all(), "everything at or before session 0 must be NaT"
    assert got.iloc[4] == pd.Timestamp("2024-07-02")


def test_prior_close_populates_the_first_session():
    """Supplying the previous session's close makes session 0 usable again."""
    prior = pd.Timestamp("2024-06-28 16:00", tz=config.TZ_MARKET)
    stamps = pd.Series(
        [
            et("2019-03-01 10:00"),   # still out of window
            et("2024-06-28 15:59"),   # before the prior close -> still out
            et("2024-06-28 16:30"),   # after the prior close -> session 0
            et("2024-07-01 09:00"),   # session 0
        ]
    )
    got = align.map_to_trading_day(stamps, CALENDAR, prior_close=prior)
    assert got.iloc[0] is pd.NaT or pd.isna(got.iloc[0])
    assert pd.isna(got.iloc[1])
    assert got.iloc[2] == pd.Timestamp("2024-07-01")
    assert got.iloc[3] == pd.Timestamp("2024-07-01")


def test_prior_close_must_be_timezone_aware():
    with pytest.raises(ValueError, match="prior_close must be timezone-aware"):
        align.map_to_trading_day(
            pd.Series([et("2024-07-02 10:00")]),
            CALENDAR,
            prior_close=pd.Timestamp("2024-06-28 16:00"),
        )


def test_intraday_mapper_refuses_date_only_input():
    """B07 descope guard: the two mappers must refuse each other's data."""
    days = pd.date_range("2015-01-01", periods=400, freq="D", tz="UTC")
    date_only = pd.Series(days).dt.tz_convert(config.TZ_MARKET)
    cal = pd.DatetimeIndex(pd.bdate_range("2015-01-01", "2016-12-31"))
    with pytest.raises(ValueError, match="date-only timestamps"):
        align.map_to_trading_day(date_only, cal)


def test_date_only_guard_is_skipped_on_small_samples():
    """Concentration is not evidence when there are only a few timestamps."""
    small = pd.Series([et("2024-07-02 10:00"), et("2024-07-03 10:00")])
    got = align.map_to_trading_day(small, CALENDAR)
    assert got.notna().all()


def test_date_only_guard_passes_genuine_intraday_data():
    rng = np.random.default_rng(config.SEED)
    cal = pd.DatetimeIndex(pd.bdate_range("2015-01-01", "2016-12-31"))
    stamps = pd.Series(
        [
            pd.Timestamp(d, tz=config.TZ_MARKET) + pd.Timedelta(minutes=int(rng.integers(0, 1440)))
            for d in cal[:300]
        ]
    )
    assert align.map_to_trading_day(stamps, cal).notna().any()


def test_trading_calendar_raises_without_the_calendar_package(monkeypatch):
    """B08: no silent degradation to business days."""
    import builtins

    from src import data as sdata

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "pandas_market_calendars":
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="pandas_market_calendars is required"):
        sdata.trading_calendar("2015-01-01", "2015-12-31")


def test_trading_calendar_excludes_market_holidays():
    from src import data as sdata

    cal = sdata.trading_calendar("2015-01-01", "2015-12-31")
    assert pd.Timestamp("2015-07-03") not in cal      # Independence Day observed
    assert pd.Timestamp("2015-12-25") not in cal      # Christmas
    assert pd.Timestamp("2015-11-26") not in cal      # Thanksgiving
    assert pd.Timestamp("2015-11-27") in cal          # half-day, still a session
    assert len(cal) == 252
