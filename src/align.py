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


# A concentration check needs enough observations to mean anything: a handful of
# timestamps legitimately covers only a handful of distinct minutes.
DATE_ONLY_CHECK_MIN_N = 200


def _reject_date_only(ts: pd.Series) -> None:
    """Refuse date-only timestamps in the intraday mapper (B05 §6, B07).

    B07's actual-session-close work is descoped because the selected corpus
    carries no times. That descope is only safe if the intraday rule can never
    be applied to date-only stamps by accident, so this mapper and
    `map_date_to_session` refuse each other's inputs.

    Skipped below `DATE_ONLY_CHECK_MIN_N` rows, where concentration is not
    evidence of anything.
    """
    if len(ts) < DATE_ONLY_CHECK_MIN_N:
        return
    from src import audit  # local import: audit imports data, which imports config

    prof = audit.timestamp_profile(ts)
    if prof["looks_date_only"]:
        raise ValueError(
            "map_to_trading_day received date-only timestamps "
            f"({prof['modal_share']:.1%} of stamps at {prof['modal_time']}, "
            f"{prof['distinct_minutes']} distinct minutes-of-day). Applying the "
            "intraday close rule to them would assign each headline to the "
            "session it was dated rather than deferring it. Use "
            "map_date_to_session instead."
        )


def map_to_trading_day(
    ts_et: pd.Series,
    calendar: pd.DatetimeIndex,
    prior_close: pd.Timestamp | None = None,
    check_date_only: bool = True,
) -> pd.Series:
    """Headline stamped s belongs to session t iff s in (close(t-1), close(t)].

    News arriving after the close, at the weekend, or on a holiday rolls forward
    into the next session's window.

    **Both** calendar edges return NaT, so out-of-window news is dropped rather
    than clipped:

    * after the last session -- no assignable session;
    * at or before the start of the first session's window -- the first
      session's window opens at the *previous* session's close, which is outside
      `calendar` and therefore unknown. Without it, everything earlier would pile
      onto the first session (B05 defect D-1: five years of pre-window headlines
      landed on one day). Pass `prior_close` -- the close of the session
      immediately before `calendar[0]` -- to populate the first session properly;
      leave it None to drop that session's news, which costs one session out of
      the sample and cannot be wrong.
    """
    ts = pd.to_datetime(pd.Series(ts_et).reset_index(drop=True))
    if ts.dt.tz is None:
        raise ValueError("ts_et must be timezone-aware; localize at load time")
    ts = ts.dt.tz_convert(config.TZ_MARKET)

    if check_date_only:
        _reject_date_only(ts)

    closes = session_closes(calendar).sort_values()
    # side="left" -> first close >= s, which is exactly the (close(t-1), close(t)]
    # half-open interval, with s == close(t) landing on t.
    idx = closes.searchsorted(ts.to_numpy(), side="left")

    days = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    ok = idx < len(closes)                                    # upper edge

    if prior_close is None:
        ok &= idx > 0                                         # lower edge: drop session 0
    else:
        lower = pd.Timestamp(prior_close)
        if lower.tz is None:
            raise ValueError("prior_close must be timezone-aware")
        ok &= (idx > 0) | (ts.to_numpy() > lower.tz_convert(config.TZ_MARKET))

    sessions = pd.DatetimeIndex(closes).tz_localize(None).normalize()
    days.loc[ok] = sessions[idx[ok]]
    return days


def _reject_intraday(ts: pd.Series) -> None:
    """Mirror of `_reject_date_only`: the deferred mapper refuses real times.

    Thresholds are `audit`'s, so the two mappers agree on what "date-only"
    means. The test is written out here rather than calling
    `audit.timestamp_profile` because these timestamps may be tz-naive dates.
    """
    if len(ts) < DATE_ONLY_CHECK_MIN_N:
        return
    from src import audit

    minute = ts.dt.hour * 60 + ts.dt.minute
    modal_share = float(minute.value_counts().iloc[0] / len(ts))
    looks_date_only = (
        modal_share >= audit.DATE_ONLY_SHARE
        or minute.nunique() < audit.MIN_DISTINCT_MINUTES
    )
    if not looks_date_only:
        raise ValueError(
            f"map_date_to_session received intraday timestamps "
            f"({minute.nunique()} distinct minutes-of-day, modal minute only "
            f"{modal_share:.1%}). Deferring them would discard a time-of-day the "
            "data actually has. Use map_to_trading_day instead."
        )


def map_date_to_session(
    dates: pd.Series,
    calendar: pd.DatetimeIndex,
    defer: bool = True,
    prior_session: pd.Timestamp | str | None = None,
    check_intraday: bool = True,
) -> pd.Series:
    """Date-only fallback mapping (B05 §2). A headline dated d -> a trading session.

    **`dates` must be in the zone the source published in, not market time.**
    The function reads the wall-clock date and never converts. FNSPID's stamps
    are `00:00 UTC`; converting them to America/New_York would make them 19:00
    or 20:00 on the *previous* calendar day, shifting every headline back a day
    before the mapping even begins. Pass `ts_utc`, not `ts_et`.

    **Primary rule (`defer=True`)** -- the first session beginning *strictly
    after* date d. Session t therefore receives dates in `[prev_session, t)`.

    With a date-only stamp the headline may have been published at 23:59 on d,
    so only the close of the first session after d is guaranteed to postdate all
    of day d. That is what keeps S_t inside the information set at close(t),
    which is where the return window for r_(t+1) begins. The cost is real and is
    stated in the report rather than hidden: the session in which the reaction
    most plausibly occurs is skipped, so the specification tests whether one-to-
    three-day-old news is associated with returns.

    **Secondary rule (`defer=False`)** -- the first session at or after d, so a
    date that is itself a session maps to that session. This is the
    specification most of the literature uses, and it assumes every headline
    dated d preceded that session's close, which the B03 audit contradicts for a
    share of the corpus. It is a **sensitivity exhibit only** and never the
    headline result.

    Both calendar edges return NaT, as in `map_to_trading_day`, and for the same
    reason: the first session's window opens at the previous session, which lies
    outside `calendar`. Supply `prior_session` to populate it.
    """
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    if getattr(d.dt, "tz", None) is not None:
        if check_intraday:
            _reject_intraday(d)
        # Keep the wall-clock date exactly as published; do not convert.
        d = d.dt.tz_localize(None)
    elif check_intraday:
        _reject_intraday(d)
    d = d.dt.normalize()

    sessions = pd.DatetimeIndex(calendar).normalize()
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    sessions = sessions.sort_values()

    # defer  -> first session strictly after d   (searchsorted "right")
    # else   -> first session at or after d      (searchsorted "left")
    idx = sessions.searchsorted(d.to_numpy(), side="right" if defer else "left")

    out = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    ok = idx < len(sessions)                                    # upper edge

    # Lower edge. A date needs the previous session only when it falls before
    # the calendar's first session -- not merely when it maps to it. Under the
    # secondary rule a date that *is* the first session maps there legitimately
    # and requires no outside knowledge, so the test is on the date itself
    # rather than on the resulting index.
    needs_prior = d.to_numpy() < sessions[0].to_numpy()
    if prior_session is None:
        ok &= ~needs_prior
    else:
        prev = pd.Timestamp(prior_session).normalize()
        if prev.tz is not None:
            prev = prev.tz_localize(None)
        # Session 0 takes [prev, s0) when deferring, (prev, s0] otherwise.
        within = (
            (d.to_numpy() >= prev.to_numpy()) if defer else (d.to_numpy() > prev.to_numpy())
        )
        ok &= (~needs_prior) | within

    out.loc[ok] = sessions[idx[ok]]
    return out


def aggregate_daily(
    scores_df: pd.DataFrame,
    headlines_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    date_only: bool | None = None,
    defer: bool = True,
    prior_session: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Per trading session: n_t, S_t per scorer, and dispersion d_t.

    d_t is the within-day standard deviation of that scorer's headline scores,
    defined only when n_t >= 5 (config.MIN_HEADLINES_FOR_DISPERSION), NaN below.
    Sessions with no headlines appear with n_headlines = 0 so the panel keeps a
    complete calendar and the zero-news count can be reported.

    `date_only` selects the mapper; None reads `config.DATE_ONLY_FALLBACK`.

    * **date-only** -- `map_date_to_session` on `ts_utc`, the source clock. It
      must be the source clock: `ts_et` would move a `00:00 UTC` stamp to the
      previous calendar day and shift every headline back one day.
    * **intraday** -- `map_to_trading_day` on `ts_et`, where market time is the
      only clock in which "after the close" means anything.

    Returns a frame whose `date` column is exactly `calendar`, which is what
    makes a row shift in `build_panel` equal a session shift.
    """
    if date_only is None:
        date_only = config.DATE_ONLY_FALLBACK

    ts_col = "ts_utc" if date_only else "ts_et"
    if ts_col not in headlines_df.columns:
        raise KeyError(
            f"headlines are missing {ts_col!r}, required in "
            f"{'date-only' if date_only else 'intraday'} mode. "
            + (
                "The deferred mapper reads the date in the source's own zone; "
                "passing ts_et would shift a 00:00 UTC stamp back one day."
                if date_only
                else "The intraday mapper compares timestamps against the close."
            )
        )

    df = headlines_df[["headline_id", ts_col]].merge(scores_df, on="headline_id", how="inner")
    if date_only:
        df["date"] = map_date_to_session(
            df[ts_col], calendar, defer=defer, prior_session=prior_session
        )
    else:
        df["date"] = map_to_trading_day(df[ts_col], calendar, prior_close=prior_session)
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
