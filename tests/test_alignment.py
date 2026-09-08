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
