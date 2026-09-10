"""B06: controls and leads must mean trading sessions, not surviving rows.

The defect these guard against is invisible in output: a column still named
`ret_lag1` or `ret_lead1` that silently refers to a different session for some
subset of rows. Both fixtures below construct the exact situation that produced
it and assert the number, not just the absence of an exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import align, inference

# Mon .. next Tue, no holidays. Distinct returns so a wrong row is detectable.
CAL = pd.DatetimeIndex(
    ["2015-01-05", "2015-01-06", "2015-01-07", "2015-01-08", "2015-01-09",
     "2015-01-12", "2015-01-13"]
)
RETS = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007])


def _market(dates=CAL, rets=RETS) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "close_adj": 100 + np.arange(n, dtype=float),
            "ret": rets[:n],
            "rv_parkinson": np.linspace(1e-4, 6e-4, n),
            "volume": np.full(n, 1e8),
            "log_volume": np.log(np.linspace(1e8, 1.5e8, n)),
            "vix_close": np.linspace(12, 18, n),
        }
    )


def _daily(n_headlines) -> pd.DataFrame:
    n = len(CAL)
    out = pd.DataFrame({"date": CAL, "n_headlines": n_headlines})
    for s in config.SCORERS:
        out[f"s_{s}"] = np.linspace(-0.2, 0.2, n)
        out[f"d_{s}"] = 0.3
    return out


# ------------------------------------------------- lags over the calendar


def test_lag_columns_are_built_on_the_full_calendar():
    panel = align.build_panel(_daily([5] * 7), _market())
    pd.testing.assert_series_equal(
        panel["ret_lag1"], panel["ret"].shift(1), check_names=False
    )
    assert pd.isna(panel["ret_lag1"].iloc[0])


def test_zero_news_tuesday_still_supplies_wednesdays_control():
    """The B06 defect, stated as a number.

    Tuesday 2015-01-06 has no headlines and is excluded from the analysis
    sample. Wednesday's lagged return must still be *Tuesday's* 0.002 -- not
    Monday's 0.001, which is what shifting inside the reduced frame would give.
    """
    panel = align.build_panel(_daily([5, 0, 5, 5, 5, 5, 5]), _market())
    sample, stats = inference.analysis_sample(panel)

    assert stats["n_zero_news_days"] == 1
    assert pd.Timestamp("2015-01-06") not in set(sample["date"])

    wed = sample[sample["date"] == pd.Timestamp("2015-01-07")].iloc[0]
    assert wed["ret_lag1"] == pytest.approx(0.002)      # Tuesday's return
    assert wed["ret_lag1"] != pytest.approx(0.001)      # not Monday's

    # And the naive alternative really would have been wrong: in the reduced
    # frame Wednesday sits at position 1, so a local shift hands it Monday's
    # 0.001 -- the exact substitution this test exists to prevent.
    assert sample["date"].iloc[1] == pd.Timestamp("2015-01-07")
    assert sample["ret"].shift(1).iloc[1] == pytest.approx(0.001)


def test_contemporaneous_refuses_a_panel_without_full_calendar_lags():
    # allow_inadmissible: RQ2 is suppressed for this corpus (B09b), and the
    # guard fires first. Here we are testing the lag contract, not that one.
    panel = align.build_panel(_daily([5] * 7), _market())
    stripped = panel.drop(columns=["ret_lag1"])
    with pytest.raises(KeyError, match="full-calendar lag"):
        inference.contemporaneous(stripped, "finbert", allow_inadmissible=True)


def test_contemporaneous_uses_the_panel_lag_not_a_local_shift():
    """Corrupt only the panel's lag column; the fit must move.

    If the function recomputed the lag internally it would ignore the column
    entirely and the coefficient would be unchanged.
    """
    rng = np.random.default_rng(config.SEED)
    long_cal = pd.bdate_range("2015-01-05", periods=120)
    daily = pd.DataFrame({"date": long_cal, "n_headlines": 5})
    for s in config.SCORERS:
        daily[f"s_{s}"] = rng.normal(0, 0.1, len(long_cal))
        daily[f"d_{s}"] = 0.3
    mkt = _market(long_cal, rng.normal(0, 0.01, len(long_cal)))
    panel = align.build_panel(daily, mkt)

    base = inference.contemporaneous(
        panel, "finbert", allow_inadmissible=True
    ).params["ret_lag1"]
    tampered = panel.copy()
    tampered["ret_lag1"] = tampered["ret_lag1"] * -1
    after = inference.contemporaneous(
        tampered, "finbert", allow_inadmissible=True
    ).params["ret_lag1"]
    assert base != pytest.approx(after)


# ------------------------------------ leads must not span a missing session


def test_missing_market_row_does_not_become_a_two_session_lead():
    """B05 defect D-2, as a regression test.

    2015-01-08 has no price row. Under the old inner join, 2015-01-07's
    `ret_lead1` became the 2015-01-09 return (0.005). It must now be NaN, so the
    observation is dropped at the eligibility stage instead of silently
    changing the estimand.
    """
    gapped = _market().drop(index=3).reset_index(drop=True)   # remove 2015-01-08
    panel = align.build_panel(_daily([5] * 7), gapped)

    assert len(panel) == len(CAL), "the missing session must survive the join"
    row = panel[panel["date"] == pd.Timestamp("2015-01-07")].iloc[0]
    assert pd.isna(row["ret_lead1"]), "lead must be NaN, not the next-but-one return"
    assert row["ret_lead1"] != pytest.approx(0.005)

    # The session after the hole keeps a correct, non-null lead.
    thu = panel[panel["date"] == pd.Timestamp("2015-01-09")].iloc[0]
    assert thu["ret_lead1"] == pytest.approx(0.006)


def test_missing_market_sessions_are_counted_not_hidden():
    gapped = _market().drop(index=3).reset_index(drop=True)
    panel = align.build_panel(_daily([5] * 7), gapped)
    stats = panel.attrs["build_stats"]
    assert stats["n_sessions"] == 7
    assert stats["n_missing_market"] == 1
    assert stats["missing_market_dates"] == ["2015-01-08"]


def test_leads_are_pure_session_shifts():
    panel = align.build_panel(_daily([5] * 7), _market())
    for h in config.HORIZONS:
        pd.testing.assert_series_equal(
            panel[f"ret_lead{h}"], panel["ret"].shift(-h), check_names=False
        )
        assert panel[f"ret_lead{h}"].tail(h).isna().all()


# --------------------------------------------------- calendar assertions


def test_calendar_assertion_accepts_a_matching_panel():
    panel = align.build_panel(_daily([5] * 7), _market())
    align.assert_sessions_match_calendar(panel, CAL)


def test_calendar_assertion_catches_a_dropped_session():
    panel = align.build_panel(_daily([5] * 7), _market())
    short = panel[panel["date"] != pd.Timestamp("2015-01-08")]
    with pytest.raises(AssertionError, match="1 session\\(s\\) missing"):
        align.assert_sessions_match_calendar(short, CAL)


def test_build_panel_rejects_duplicate_sessions():
    daily = pd.concat([_daily([5] * 7), _daily([5] * 7).head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate sessions"):
        align.build_panel(daily, _market())


def test_close_adj_reaches_the_panel():
    """The context figure needs a price series; the schema must carry one (P29)."""
    panel = align.build_panel(_daily([5] * 7), _market())
    assert "close_adj" in panel.columns
    assert panel["close_adj"].notna().all()
