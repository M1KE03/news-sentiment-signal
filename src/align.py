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
    + ["ret", "ret_lag1"]
    + [f"ret_lead{h}" for h in config.HORIZONS]
    + ["parkinson", "parkinson_lag1", "parkinson_lead1"]
    + ["log_turnover", "log_turnover_lag1"]
    + ["log_turnover_detrended", "log_turnover_detrended_lead1"]
    + ["close_adj", "vix_close"]
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


def assert_sessions_match_calendar(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> None:
    """The panel's dates must be exactly the exchange calendar, in order.

    `build_panel` builds every lag and lead by shifting rows, which is only the
    same thing as shifting *sessions* when the frame is the complete calendar.
    That premise is checked here rather than assumed, wherever a calendar is in
    scope (`run_all.py`).
    """
    sessions = pd.DatetimeIndex(calendar).normalize()
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"])).normalize()
    if not dates.equals(sessions):
        missing = sessions.difference(dates)
        extra = dates.difference(sessions)
        raise AssertionError(
            "panel dates are not the exchange calendar: "
            f"{len(missing)} session(s) missing (first: "
            f"{missing[0].date() if len(missing) else '-'}), "
            f"{len(extra)} unexpected date(s) (first: "
            f"{extra[0].date() if len(extra) else '-'}). "
            "Row shifts would not equal session shifts."
        )


def build_panel(
    daily_scores: pd.DataFrame, market: pd.DataFrame, turnover_window: int = 63
) -> pd.DataFrame:
    """Join market data onto the calendar-indexed daily scores; build lags and leads.

    `daily_scores` must span the **complete exchange calendar** -- `aggregate_daily`
    guarantees that by reindexing to it. That is what makes a row shift equal a
    session shift, and it is why the join below is a LEFT join onto the daily
    frame rather than an inner join.

    An inner join was the earlier bug (B05, defect D-2): a session missing from
    the market frame was dropped, and the subsequent `.shift(-1)` reached across
    the hole, so `ret_lead1` silently became a two-session return for an unknown
    subset of rows. With a left join the session survives with NaN market
    columns, the lead for the preceding row is NaN, and the observation is
    excluded at the eligibility stage where the exclusion is counted.

    Lags are built here too, on the full calendar, for the same reason: a
    regression function that shifts inside its own frame would compute the lag
    *after* zero-news sessions had been excluded, so a Wednesday whose Tuesday
    was dropped would silently take Monday's return as its lag (B05/P12).

    Turnover is detrended against a **trailing-only** rolling mean; a centred
    window would leak future volume into day t.

    `panel.attrs["build_stats"]` carries the missing-market-session counts.
    """
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    daily_scores = daily_scores.copy()
    daily_scores["date"] = pd.to_datetime(daily_scores["date"]).dt.normalize()
    daily_scores = daily_scores.sort_values("date").reset_index(drop=True)

    if daily_scores["date"].duplicated().any():
        raise ValueError("daily_scores contains duplicate sessions")
    if not daily_scores["date"].is_monotonic_increasing:
        raise ValueError("daily_scores is not sorted by date")

    n_sessions = len(daily_scores)
    panel = daily_scores.merge(market, on="date", how="left")
    if len(panel) != n_sessions:
        raise AssertionError(
            f"joining market data changed the session count "
            f"({n_sessions} -> {len(panel)}); the market frame has duplicate dates"
        )
    panel = panel.reset_index(drop=True)

    missing_market = panel["ret"].isna() if "ret" in panel else pd.Series(True, index=panel.index)
    panel.attrs["build_stats"] = {
        "n_sessions": int(n_sessions),
        "n_missing_market": int(missing_market.sum()),
        "missing_market_dates": [
            str(d.date()) for d in panel.loc[missing_market, "date"].head(10)
        ],
    }

    roll = panel["log_turnover"].rolling(turnover_window, min_periods=turnover_window).mean()
    panel["log_turnover_detrended"] = panel["log_turnover"] - roll

    # Lags and leads: the only place either is constructed. Every shift below is
    # a shift over the complete calendar, so it is a shift in trading sessions.
    panel["ret_lag1"] = panel["ret"].shift(1)
    panel["parkinson_lag1"] = panel["parkinson"].shift(1)
    panel["log_turnover_lag1"] = panel["log_turnover"].shift(1)

    for h in config.HORIZONS:
        panel[f"ret_lead{h}"] = panel["ret"].shift(-h)
    panel["parkinson_lead1"] = panel["parkinson"].shift(-1)
    panel["log_turnover_detrended_lead1"] = panel["log_turnover_detrended"].shift(-1)

    for c in PANEL_COLUMNS:
        if c not in panel.columns:
            panel[c] = np.nan
    out = panel.loc[:, PANEL_COLUMNS]
    out.attrs["build_stats"] = panel.attrs["build_stats"]
    return out
