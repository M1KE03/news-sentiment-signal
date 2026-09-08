"""The look-ahead firewall: timestamps -> trading days -> the one analysis table.

D6 is implemented in exactly one function (`map_to_trading_day`) and the lead
columns are created by exactly one shift (`build_panel`). Both are deliberate:
these are the two places a look-ahead bug can hide in plain sight, so there is
only one of each to audit, and `tests/test_alignment.py` audits them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

PANEL_COLUMNS = (
    ["date", "n_headlines"]
    + [f"s_{n}" for n in config.SCORERS]
    + [f"d_{n}" for n in config.SCORERS]
    + ["ret"]
    + [f"ret_lead{h}" for h in config.HORIZONS]
    + ["parkinson", "parkinson_lead1"]
    + ["log_turnover", "log_turnover_detrended", "log_turnover_detrended_lead1"]
    + ["vix_close"]
)


def session_closes(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """close(t) for each session, as tz-aware ET timestamps. D6: 16:00 ET."""
    cal = pd.DatetimeIndex(calendar).normalize()
    if cal.tz is not None:
        cal = cal.tz_localize(None)
    closes = cal + pd.Timedelta(
        hours=config.MARKET_CLOSE_ET.hour, minutes=config.MARKET_CLOSE_ET.minute
    )
    return closes.tz_localize(config.TZ_MARKET)


def map_to_trading_day(ts_et: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """D6. Headline stamped s belongs to trading day t iff s in (close(t-1), close(t)].

    News arriving after the close, at the weekend, or on a holiday rolls forward
    into the next session's window. Headlines falling after the last session in
    `calendar` get NaT -- they have no assignable trading day and must be dropped
    rather than clipped onto the final day.
    """
    ts = pd.to_datetime(pd.Series(ts_et).reset_index(drop=True))
    if ts.dt.tz is None:
        raise ValueError("ts_et must be timezone-aware; localize at load time (§4)")
    ts = ts.dt.tz_convert(config.TZ_MARKET)

    closes = session_closes(calendar).sort_values()
    # side="left" -> first close >= s, which is exactly the (close(t-1), close(t)]
    # half-open interval, with s == close(t) landing on t.
    idx = closes.searchsorted(ts.to_numpy(), side="left")

    days = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    ok = idx < len(closes)
    sessions = pd.DatetimeIndex(closes).tz_localize(None).normalize()
    days.loc[ok] = sessions[idx[ok]]
    return days


def aggregate_daily(
    scores_df: pd.DataFrame, headlines_df: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """D8/D9. Per trading day: n_t, S_t per scorer, and dispersion d_t.

    d_t is the within-day standard deviation of that scorer's headline scores,
    defined only when n_t >= 5 (config.MIN_HEADLINES_FOR_DISPERSION), NaN below.
    Sessions with no headlines appear with n_headlines = 0 so the panel keeps a
    complete calendar and the zero-news day count can be reported (D9).
    """
    df = headlines_df[["headline_id", "ts_et"]].merge(scores_df, on="headline_id", how="inner")
    df["date"] = map_to_trading_day(df["ts_et"], calendar)
    df = df[df["date"].notna()]

    score_cols = [f"score_{n}" for n in config.SCORERS]
    grouped = df.groupby("date")
    daily = pd.DataFrame({"n_headlines": grouped.size()})
    for n in config.SCORERS:
        daily[f"s_{n}"] = grouped[f"score_{n}"].mean()
        daily[f"d_{n}"] = grouped[f"score_{n}"].std(ddof=1)

    thin = daily["n_headlines"] < config.MIN_HEADLINES_FOR_DISPERSION
    for n in config.SCORERS:
        daily.loc[thin, f"d_{n}"] = np.nan

    sessions = pd.DatetimeIndex(calendar).normalize()
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    daily = daily.reindex(sessions)
    daily["n_headlines"] = daily["n_headlines"].fillna(0).astype(int)
    daily.index.name = "date"
    return daily.reset_index()


def build_panel(
    daily_scores: pd.DataFrame, market: pd.DataFrame, turnover_window: int = 63
) -> pd.DataFrame:
    """Join the daily scores to market data and add every lead. §4 schema.

    Turnover is detrended against a **trailing-only** rolling mean: a centred
    window would leak future volume into day t, which is the same class of bug
    the lead columns exist to make visible.
    """
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    daily_scores = daily_scores.copy()
    daily_scores["date"] = pd.to_datetime(daily_scores["date"]).dt.normalize()

    panel = market.merge(daily_scores, on="date", how="inner").sort_values("date")
    panel = panel.reset_index(drop=True)

    roll = panel["log_turnover"].rolling(turnover_window, min_periods=turnover_window).mean()
    panel["log_turnover_detrended"] = panel["log_turnover"] - roll

    # The single shift. Every forward-looking column in the project is made here.
    for h in config.HORIZONS:
        panel[f"ret_lead{h}"] = panel["ret"].shift(-h)
    panel["parkinson_lead1"] = panel["parkinson"].shift(-1)
    panel["log_turnover_detrended_lead1"] = panel["log_turnover_detrended"].shift(-1)

    for c in PANEL_COLUMNS:
        if c not in panel.columns:
            panel[c] = np.nan
    return panel.loc[:, PANEL_COLUMNS]
