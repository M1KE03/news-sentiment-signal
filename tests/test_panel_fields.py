"""R07a: panel field names must match their formulas, and stale names must fail.

Audit A07 / decision P22. Two columns were misnamed. `log_turnover` is
`log(share volume)` with no share-count denominator anywhere in the pipeline, so
it was never turnover; `parkinson` was described as volatility when it is a
high-low range estimator of variance that cannot see the intraday path or
overnight moves. The inference protocol Section 10 fixes the names; this file
holds them to their formulas and closes the silent-failure path the rename
opens.

The failure being closed is not the wrong name -- it is what a *stale* artifact
does after the rename. `build_panel` fills any declared panel column its inputs
did not supply with NaN, so a market frame still carrying `log_turnover` would
produce an entirely missing `log_volume`, and the regression would be fitted on
whatever rows survived, reporting a small sample rather than an error.

Offline throughout: yfinance is replaced with fixture frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import align, data as sdata, plots

CAL = pd.DatetimeIndex(
    ["2015-01-05", "2015-01-06", "2015-01-07", "2015-01-08", "2015-01-09"]
)


def _bars(dates):
    """Bars whose High/Low/Volume all differ per row, so a formula check bites."""
    idx = pd.DatetimeIndex(dates)
    n = len(idx)
    closes = np.linspace(100.0, 104.0, n)
    return pd.DataFrame(
        {
            "Close": closes,
            "High": closes * np.linspace(1.005, 1.02, n),
            "Low": closes * np.linspace(0.995, 0.98, n),
            "Volume": np.linspace(5e7, 2e8, n),
        },
        index=idx,
    )


@pytest.fixture
def yf(monkeypatch):
    import types

    state = {"frames": {}}

    def download(ticker, start=None, end=None, **kw):
        return state["frames"].get(ticker, pd.DataFrame())

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    return state


def _market(cal=CAL):
    """A market frame in the current contract, built by hand."""
    n = len(cal)
    rng = np.random.default_rng(config.SEED)
    return pd.DataFrame(
        {
            "date": cal,
            "close_adj": 100 + np.arange(n, dtype=float),
            "ret": rng.normal(0, 0.01, n),
            "rv_parkinson": rng.uniform(1e-5, 5e-4, n),
            "volume": rng.uniform(5e7, 2e8, n),
            "log_volume": rng.normal(18.4, 0.2, n),
            "vix_close": rng.uniform(10, 30, n),
        }
    )


def _daily(cal=CAL):
    n = len(cal)
    rng = np.random.default_rng(config.SEED + 1)
    daily = pd.DataFrame({"date": cal, "n_headlines": 5})
    for s in config.SCORERS:
        daily[f"s_{s}"] = rng.normal(0, 0.1, n)
        daily[f"d_{s}"] = rng.uniform(0.1, 0.4, n)
    return daily


# --------------------------------------------------------- names vs formulas


def test_log_volume_is_log_share_volume(yf):
    """The name claims log(volume) and nothing more; check it against the bars."""
    bars = _bars(CAL)
    yf["frames"]["SPY"] = bars
    out = sdata.load_market("2015-01-05", "2015-01-09", calendar=CAL).set_index("date")

    expected = np.log(bars["Volume"].to_numpy())
    np.testing.assert_allclose(out["log_volume"].to_numpy(), expected)
    assert "log_turnover" not in out.columns, (
        "turnover needs a share-count denominator this pipeline never obtains"
    )


def test_rv_parkinson_is_the_range_variance_estimator(yf):
    """(log(H/L))^2 / (4 log 2) -- a range estimator, not total daily variance."""
    bars = _bars(CAL)
    yf["frames"]["SPY"] = bars
    out = sdata.load_market("2015-01-05", "2015-01-09", calendar=CAL).set_index("date")

    expected = (np.log(bars["High"] / bars["Low"]).to_numpy() ** 2) / (4 * np.log(2))
    np.testing.assert_allclose(out["rv_parkinson"].to_numpy(), expected)
    assert "parkinson" not in out.columns


def test_detrended_log_volume_removes_a_trailing_mean_only():
    """The name says detrended; the window must not reach past t (leakage)."""
    cal = pd.bdate_range("2015-01-05", periods=40)
    window = 5
    panel = align.build_panel(_daily(cal), _market(cal), volume_window=window)

    lv = panel["log_volume"]
    trailing = lv.rolling(window, min_periods=window).mean()
    np.testing.assert_allclose(
        panel["log_volume_detrended"].to_numpy(),
        (lv - trailing).to_numpy(),
        equal_nan=True,
    )
    assert panel["log_volume_detrended"].iloc[: window - 1].isna().all(), (
        "the first sessions have no complete trailing window and must stay missing"
    )


def test_the_panel_carries_the_new_names_and_none_of_the_old_ones():
    panel = align.build_panel(_daily(), _market())
    for col in ("rv_parkinson", "rv_parkinson_lag1", "rv_parkinson_lead1",
                "log_volume", "log_volume_lag1",
                "log_volume_detrended", "log_volume_detrended_lead1"):
        assert col in panel.columns
    for gone in align.LEGACY_COLUMN_RENAMES:
        assert gone not in panel.columns


# ------------------------------------------------- stale artifacts must fail


def test_a_market_frame_with_the_old_names_is_refused():
    """The NaN-fill would otherwise turn this into a silently empty control."""
    stale = _market().rename(columns={"log_volume": "log_turnover",
                                      "rv_parkinson": "parkinson"})
    with pytest.raises(ValueError, match="pre-B16 column name"):
        align.build_panel(_daily(), stale)


def test_the_refusal_names_the_column_and_its_replacement():
    stale = _market().rename(columns={"log_volume": "log_turnover"})
    with pytest.raises(ValueError) as exc:
        align.build_panel(_daily(), stale)
    assert "log_turnover -> log_volume" in str(exc.value)


def test_a_market_frame_missing_a_control_is_refused_not_nan_filled():
    """An all-NaN control empties the sample and gets reported as a sample size."""
    short = _market().drop(columns=["rv_parkinson"])
    with pytest.raises(ValueError, match="missing required column"):
        align.build_panel(_daily(), short)


def test_assert_panel_schema_rejects_a_panel_saved_before_the_rename():
    """`run_all.py --skip-panel` reuses whatever is on disk."""
    panel = align.build_panel(_daily(), _market())
    stale = panel.rename(columns={"log_volume_detrended_lead1":
                                  "log_turnover_detrended_lead1"})
    with pytest.raises(ValueError, match="pre-B16 column name"):
        align.assert_panel_schema(stale)


def test_assert_panel_schema_rejects_a_panel_missing_a_contract_column():
    panel = align.build_panel(_daily(), _market())
    with pytest.raises(ValueError, match="missing contract column"):
        align.assert_panel_schema(panel.drop(columns=["rv_parkinson_lead1"]))


def test_assert_panel_schema_accepts_a_freshly_built_panel():
    align.assert_panel_schema(align.build_panel(_daily(), _market()))


# ------------------------------------------- P29: prices reach the context figure


def test_the_context_figure_plots_the_panel_prices():
    """P29: it used to fall back to a scalar NaN, so no price curve appeared."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal = pd.bdate_range("2015-01-05", periods=40)
    panel = align.build_panel(_daily(cal), _market(cal))
    fig = plots.figure4_context(panel)
    try:
        price_lines = [
            ln for ax in fig.axes for ln in ax.get_lines()
            if ln.get_label() == "SPY"
        ]
        assert len(price_lines) == 1
        drawn = np.asarray(price_lines[0].get_ydata(), dtype=float)
        assert np.isfinite(drawn).all(), (
            "the price curve must carry real prices, not a broadcast NaN"
        )
        np.testing.assert_allclose(drawn, panel["close_adj"].to_numpy())
    finally:
        plt.close(fig)


def test_the_context_figure_refuses_a_panel_without_prices():
    import matplotlib
    matplotlib.use("Agg")

    panel = align.build_panel(_daily(), _market()).drop(columns=["close_adj"])
    with pytest.raises(KeyError, match="close_adj"):
        plots.figure4_context(panel)
