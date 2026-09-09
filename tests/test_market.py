"""R04a: a return must span exactly one exchange session.

Audit A04: `load_market` computed `diff(log(close))` over whatever rows the
price source returned, *before* any calendar reindex. A session missing from
that response produced a two-session return sitting in the row labelled with the
later date, and `build_panel`'s left-join then inherited it. The earlier B06
work fixed the panel-side shift only; the number was already wrong upstream.

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
from src import align, data as sdata

CAL = pd.DatetimeIndex(
    ["2015-01-05", "2015-01-06", "2015-01-07", "2015-01-08", "2015-01-09"]
)
CLOSES = [100.0, 101.0, 102.0, 103.0, 104.0]


def _bars(dates, closes):
    idx = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {
            "Close": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Volume": [1e8] * len(closes),
        },
        index=idx,
    )


@pytest.fixture
def yf(monkeypatch):
    """Serve fixture bars through the yfinance API, recording the request."""
    import types

    state = {"frames": {}, "calls": []}

    def download(ticker, start=None, end=None, **kw):
        state["calls"].append({"ticker": ticker, "start": start, "end": end})
        return state["frames"].get(ticker, pd.DataFrame())

    fake = types.SimpleNamespace(download=download)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return state


# ------------------------------------------------------- the core defect


def test_return_across_a_missing_session_is_undefined():
    """The A04 fixture, stated as a number.

    Monday 100, Tuesday absent, Wednesday 102. The naive computation gives
    Wednesday log(102/100) = 0.019803 -- a two-session return wearing a
    one-session label. It must be NaN instead.
    """
    present = pd.Series([100.0, 102.0], index=pd.DatetimeIndex(["2015-01-05", "2015-01-07"]))
    cal = pd.DatetimeIndex(["2015-01-05", "2015-01-06", "2015-01-07", "2015-01-08"])

    naive = np.log(present).diff().iloc[-1]
    assert naive == pytest.approx(0.0198026, abs=1e-6), "the defect, for reference"

    ret = sdata.returns_on_calendar(present, cal)
    assert pd.isna(ret.loc[pd.Timestamp("2015-01-06")]), "missing session has no return"
    assert pd.isna(ret.loc[pd.Timestamp("2015-01-07")]), "nor does the session after it"


def test_returns_are_correct_when_no_session_is_missing():
    closes = pd.Series(CLOSES, index=CAL)
    ret = sdata.returns_on_calendar(closes, CAL)
    expected = np.log(pd.Series(CLOSES, index=CAL)).diff()
    # check_freq=False: reindexing sets the index's inferred freq; only values matter.
    pd.testing.assert_series_equal(ret, expected, check_names=False, check_freq=False)
    assert pd.isna(ret.iloc[0]), "the first session has no predecessor"


def test_return_resumes_after_the_gap():
    """Only the two affected returns are lost, not everything downstream."""
    present = pd.Series([100.0, 102.0, 103.0],
                        index=pd.DatetimeIndex(["2015-01-05", "2015-01-07", "2015-01-08"]))
    ret = sdata.returns_on_calendar(present, CAL[:4])
    assert pd.isna(ret.loc[pd.Timestamp("2015-01-07")])
    assert ret.loc[pd.Timestamp("2015-01-08")] == pytest.approx(np.log(103 / 102))


# ------------------------------------------------------- loader integration


def test_loader_places_prices_on_the_calendar(yf):
    yf["frames"]["SPY"] = _bars(CAL, CLOSES)
    out = sdata.load_market("2015-01-05", "2015-01-09", calendar=CAL)
    assert list(pd.DatetimeIndex(out["date"])) == list(CAL)
    assert out.attrs["market_stats"]["n_missing_price"] == 0


def test_loader_leaves_a_missing_session_present_but_null(yf):
    """The session survives as a row; its price and returns are NaN."""
    partial = _bars([d for d in CAL if d != pd.Timestamp("2015-01-07")],
                    [100.0, 101.0, 103.0, 104.0])
    yf["frames"]["SPY"] = partial
    out = sdata.load_market("2015-01-05", "2015-01-09", calendar=CAL).set_index("date")

    assert len(out) == len(CAL)
    assert pd.isna(out.loc[pd.Timestamp("2015-01-07"), "close_adj"])
    assert pd.isna(out.loc[pd.Timestamp("2015-01-07"), "ret"])
    assert pd.isna(out.loc[pd.Timestamp("2015-01-08"), "ret"]), (
        "the session after the gap must not carry a two-session return"
    )
    stats = out.attrs["market_stats"] if hasattr(out, "attrs") else {}
    assert stats.get("n_missing_price") == 1


def test_yfinance_end_is_treated_as_exclusive(yf):
    """The configured end is inclusive; yfinance's is not."""
    yf["frames"]["SPY"] = _bars(CAL, CLOSES)
    sdata.load_market("2015-01-05", "2015-01-09", calendar=CAL)
    call = yf["calls"][0]
    assert call["start"] == "2015-01-05"
    assert call["end"] == "2015-01-10", "one day past the inclusive end, or the last session is lost"


def test_warmup_sessions_are_flagged_and_do_not_widen_the_window(yf):
    full = pd.DatetimeIndex(pd.bdate_range("2014-12-01", "2015-01-09"))
    yf["frames"]["SPY"] = _bars(full, list(np.linspace(90, 104, len(full))))
    out = sdata.load_market("2015-01-05", "2015-01-09", calendar=CAL, warmup_sessions=5)

    in_window = out[out["in_window"]]
    assert list(pd.DatetimeIndex(in_window["date"])) == list(CAL)
    assert (~out["in_window"]).sum() == 5
    # The first analysis session now has a defined lagged return.
    first = out[out["in_window"]].iloc[0]
    assert not pd.isna(first["ret"])


def test_missing_prices_are_reported_not_hidden(yf):
    partial = _bars([d for d in CAL if d != pd.Timestamp("2015-01-07")],
                    [100.0, 101.0, 103.0, 104.0])
    yf["frames"]["SPY"] = partial
    out = sdata.load_market("2015-01-05", "2015-01-09", calendar=CAL)
    stats = out.attrs["market_stats"]
    assert stats["n_missing_price"] == 1
    assert stats["missing_dates"] == ["2015-01-07"]


# ------------------------------------------------------- loader -> panel


def test_gap_does_not_reach_the_panel_as_a_valid_lead(yf):
    """End to end: the defect must not survive into `ret_lead1`."""
    partial = _bars([d for d in CAL if d != pd.Timestamp("2015-01-07")],
                    [100.0, 101.0, 103.0, 104.0])
    yf["frames"]["SPY"] = partial
    market = sdata.load_market("2015-01-05", "2015-01-09", calendar=CAL)

    daily = pd.DataFrame({"date": CAL, "n_headlines": 5})
    for s in config.SCORERS:
        daily[f"s_{s}"] = 0.1
        daily[f"d_{s}"] = 0.2
    panel = align.build_panel(daily, market.drop(columns=["in_window"]))

    by_date = panel.set_index("date")
    # 2015-01-06's lead is the missing session: undefined.
    assert pd.isna(by_date.loc[pd.Timestamp("2015-01-06"), "ret_lead1"])
    # 2015-01-07's lead would have been the contaminated two-session return.
    assert pd.isna(by_date.loc[pd.Timestamp("2015-01-07"), "ret_lead1"])
    # Clean sessions are unaffected.
    assert not pd.isna(by_date.loc[pd.Timestamp("2015-01-08"), "ret_lead1"])
