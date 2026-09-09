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
    daily = align.aggregate_daily(scores, headlines, CALENDAR, date_only=False)
    thin = daily["n_headlines"] < config.MIN_HEADLINES_FOR_DISPERSION
    assert daily.loc[thin, [f"d_{s}" for s in config.SCORERS]].isna().all().all()


def test_aggregate_daily_covers_every_session():
    headlines, scores = _synthetic_headlines()
    daily = align.aggregate_daily(scores, headlines, CALENDAR, date_only=False)
    assert list(daily["date"]) == list(CALENDAR)
    assert daily["n_headlines"].ge(0).all()


def test_leads_are_pure_shifts_and_never_leak():
    headlines, scores = _synthetic_headlines()
    daily = align.aggregate_daily(scores, headlines, CALENDAR, date_only=False)
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


# --------------------------------------------------------------------------
# B09a: the date-only fallback mapper.
# --------------------------------------------------------------------------


def dated(day: str, **kw) -> pd.Timestamp:
    return align.map_date_to_session(pd.Series([pd.Timestamp(day)]), CALENDAR, **kw).iloc[0]


def test_deferred_rule_sends_a_session_date_to_the_following_session():
    # Mon 07-01 is itself a session; deferring means Tue 07-02, never 07-01.
    assert dated("2024-07-01") == pd.Timestamp("2024-07-02")


def test_deferred_rule_skips_a_holiday():
    # Wed 07-03; Thu 07-04 is a holiday, so the next session is Fri 07-05.
    assert dated("2024-07-03") == pd.Timestamp("2024-07-05")


def test_deferred_rule_sends_friday_to_monday():
    assert dated("2024-07-05") == pd.Timestamp("2024-07-08")


def test_deferred_rule_rolls_the_weekend_forward():
    assert dated("2024-07-06") == pd.Timestamp("2024-07-08")   # Saturday
    assert dated("2024-07-07") == pd.Timestamp("2024-07-08")   # Sunday


def test_deferred_rule_has_no_session_after_the_last_one():
    assert pd.isna(dated("2024-07-09"))
    assert pd.isna(dated("2024-07-20"))


def test_deferred_rule_drops_pre_window_dates():
    """The D-1 edge, in the date-only mapper too."""
    for day in ("2010-01-04", "2024-06-14", "2024-06-27"):
        assert pd.isna(dated(day)), f"{day} must not land on the first session"


def test_prior_session_populates_the_first_session():
    # Session 0 takes dates in [prior_session, session_0).
    assert dated("2024-06-28", prior_session="2024-06-28") == pd.Timestamp("2024-07-01")
    assert pd.isna(dated("2024-06-27", prior_session="2024-06-28"))


def test_secondary_rule_maps_a_session_date_to_itself():
    """The as-if-intraday sensitivity: assumes same-date news precedes the close."""
    assert dated("2024-07-01", defer=False) == pd.Timestamp("2024-07-01")
    assert dated("2024-07-03", defer=False) == pd.Timestamp("2024-07-03")
    # A non-session date still rolls forward under either rule.
    assert dated("2024-07-04", defer=False) == pd.Timestamp("2024-07-05")
    assert dated("2024-07-06", defer=False) == pd.Timestamp("2024-07-08")


def test_the_two_rules_differ_by_exactly_one_session_on_session_dates():
    dates = pd.Series([pd.Timestamp(d) for d in ("2024-07-01", "2024-07-02", "2024-07-03")])
    primary = align.map_date_to_session(dates, CALENDAR, defer=True)
    secondary = align.map_date_to_session(dates, CALENDAR, defer=False)
    assert (primary > secondary).all(), "deferring must never be the earlier session"


def test_utc_midnight_stamps_keep_their_published_date():
    """The timezone trap the mapper exists to avoid.

    A 00:00 UTC stamp is 20:00 ET on the *previous* day. Reading the date in
    market time would move every headline back one calendar day before the
    mapping starts, so the mapper must read the source clock.
    """
    utc = pd.Series([pd.Timestamp("2024-07-02 00:00", tz="UTC")])
    assert align.map_date_to_session(utc, CALENDAR).iloc[0] == pd.Timestamp("2024-07-03")

    # What converting first would have done: 2024-07-01 20:00 ET -> dated 07-01
    # -> mapped to 07-02. One session early, silently.
    et_converted = utc.dt.tz_convert(config.TZ_MARKET)
    assert (
        align.map_date_to_session(et_converted, CALENDAR).iloc[0]
        == pd.Timestamp("2024-07-02")
    )


def test_deferred_mapper_refuses_intraday_input():
    rng = np.random.default_rng(config.SEED)
    cal = pd.DatetimeIndex(pd.bdate_range("2015-01-01", "2016-12-31"))
    stamps = pd.Series(
        [
            pd.Timestamp(d, tz="UTC") + pd.Timedelta(minutes=int(rng.integers(0, 1440)))
            for d in cal[:300]
        ]
    )
    with pytest.raises(ValueError, match="intraday timestamps"):
        align.map_date_to_session(stamps, cal)


def test_deferred_mapper_accepts_a_real_date_only_series():
    days = pd.date_range("2015-01-01", periods=400, freq="D", tz="UTC")
    cal = pd.DatetimeIndex(pd.bdate_range("2015-01-01", "2016-12-31"))
    got = align.map_date_to_session(pd.Series(days), cal)
    assert got.notna().sum() > 350


# ------------------------------------------------ aggregate_daily dispatch


def _date_only_headlines(n_per_day: int = 6, seed: int = config.SEED):
    """FNSPID-shaped rows: every stamp at 00:00 UTC, both clocks present."""
    rng = np.random.default_rng(seed)
    days = [d for d in pd.date_range("2024-06-28", "2024-07-09", freq="D")
            for _ in range(n_per_day)]
    ts_utc = pd.Series([pd.Timestamp(d, tz="UTC") for d in days])
    headlines = pd.DataFrame(
        {
            "headline_id": [f"h{i:04d}" for i in range(len(ts_utc))],
            "ts_utc": ts_utc,
            "ts_et": ts_utc.dt.tz_convert(config.TZ_MARKET),
        }
    )
    scores = pd.DataFrame(
        {"headline_id": headlines["headline_id"]}
        | {f"score_{s}": rng.uniform(-1, 1, len(headlines)) for s in config.SCORERS}
    )
    return headlines, scores


def test_aggregate_daily_uses_the_deferred_mapper_in_date_only_mode():
    headlines, scores = _date_only_headlines()
    daily = align.aggregate_daily(scores, headlines, CALENDAR, date_only=True)
    assert list(daily["date"]) == list(CALENDAR)
    # Session t receives dates in [prev_session, t), so 07-02 takes exactly the
    # 07-01 headlines. The first session gets nothing: its window opens at a
    # session outside the calendar, and 06-28..06-30 are therefore dropped.
    counts = daily.set_index("date")["n_headlines"]
    assert counts.loc[pd.Timestamp("2024-07-01")] == 0
    assert counts.loc[pd.Timestamp("2024-07-02")] == 6
    # 07-05 takes [07-03, 07-05): the 07-03 headlines *and* the 07-04 holiday's.
    assert counts.loc[pd.Timestamp("2024-07-05")] == 12
    assert counts.loc[pd.Timestamp("2024-07-08")] == 18     # 07-05, Sat 07-06, Sun 07-07
    # Nothing is lost or double-counted: every non-dropped headline lands once.
    assert counts.sum() == 6 * 8                            # 07-01 .. 07-08 inclusive


def test_aggregate_daily_requires_the_source_clock_in_date_only_mode():
    headlines, scores = _date_only_headlines()
    with pytest.raises(KeyError, match="ts_utc"):
        align.aggregate_daily(
            scores, headlines.drop(columns=["ts_utc"]), CALENDAR, date_only=True
        )


def test_aggregate_daily_intraday_mode_is_unchanged():
    headlines, scores = _synthetic_headlines()
    daily = align.aggregate_daily(scores, headlines, CALENDAR, date_only=False)
    assert list(daily["date"]) == list(CALENDAR)


def test_aggregate_daily_defaults_to_the_configured_fallback():
    headlines, scores = _date_only_headlines()
    daily = align.aggregate_daily(scores, headlines, CALENDAR)   # no date_only arg
    assert config.DATE_ONLY_FALLBACK is True
    assert daily.set_index("date")["n_headlines"].loc[pd.Timestamp("2024-07-01")] == 0


# --------------------------------------------------------------------------
# R04b: the score-to-panel completeness gate (audit A03).
# --------------------------------------------------------------------------


def _scored_headlines(n=5, day="2024-07-02"):
    heads = pd.DataFrame({
        "headline_id": [f"g{i}" for i in range(n)],
        "ts_utc": [pd.Timestamp(day, tz="UTC")] * n,
    })
    scores = pd.DataFrame({"headline_id": heads["headline_id"]}
                          | {f"score_{s}": np.full(n, 0.5) for s in config.SCORERS})
    return heads, scores


def test_partial_scoring_is_refused_by_default():
    """The A03 fixture: five headlines, one scored, previously reported n=1."""
    heads, scores = _scored_headlines(5)
    with pytest.raises(align.IncompleteScoring, match="have no score row"):
        align.aggregate_daily(scores.head(1), heads, CALENDAR, date_only=True)


def test_refusal_names_the_shortfall():
    heads, scores = _scored_headlines(5)
    with pytest.raises(align.IncompleteScoring) as exc:
        align.aggregate_daily(scores.head(2), heads, CALENDAR, date_only=True)
    msg = str(exc.value)
    assert "3 of 5 headlines have no score row" in msg
    assert "within-day sampling" in msg or "different subset" in msg


def test_missing_value_within_a_scored_row_is_refused():
    """A row can exist and still be unusable."""
    heads, scores = _scored_headlines(5)
    scores.loc[0, "score_lm"] = np.nan
    with pytest.raises(align.IncompleteScoring, match="missing value"):
        align.aggregate_daily(scores, heads, CALENDAR, date_only=True)


def test_out_of_range_score_is_refused():
    heads, scores = _scored_headlines(5)
    scores.loc[2, "score_finbert"] = 1.4
    with pytest.raises(align.IncompleteScoring, match=r"outside \[-1, 1\]"):
        align.aggregate_daily(scores, heads, CALENDAR, date_only=True)


def test_duplicate_score_rows_are_refused():
    """A duplicate score row would multiply that headline's weight in the mean."""
    heads, scores = _scored_headlines(5)
    doubled = pd.concat([scores, scores.head(1)], ignore_index=True)
    with pytest.raises(align.IncompleteScoring, match="duplicate headline_id"):
        align.aggregate_daily(doubled, heads, CALENDAR, date_only=True)


def test_complete_scoring_passes_and_counts_every_headline():
    heads, scores = _scored_headlines(5)
    daily = align.aggregate_daily(scores, heads, CALENDAR, date_only=True)
    row = daily[daily["date"] == pd.Timestamp("2024-07-03")].iloc[0]
    assert row["n_headlines"] == 5, "all five headlines must be counted"
    assert row["s_lm"] == pytest.approx(0.5)
    assert daily.attrs["coverage"]["n_headlines_scored"] == 5


def test_declared_partial_pass_drops_incomplete_sessions():
    """Bounded scoring yields whole sessions or none -- never a subset of one."""
    heads = pd.DataFrame({
        "headline_id": [f"g{i}" for i in range(6)],
        "ts_utc": [pd.Timestamp("2024-07-02", tz="UTC")] * 3
                  + [pd.Timestamp("2024-07-03", tz="UTC")] * 3,
    })
    # 07-02's three are all scored; 07-03 has only two of three.
    scored_ids = ["g0", "g1", "g2", "g3", "g4"]
    scores = pd.DataFrame({"headline_id": scored_ids}
                          | {f"score_{s}": np.full(5, 0.25) for s in config.SCORERS})

    daily = align.aggregate_daily(scores, heads, CALENDAR, date_only=True,
                                  require_complete=False)
    # 07-02 defers to 07-03; 07-03 defers past the 07-04 holiday to 07-05.
    by_date = daily.set_index("date")["n_headlines"]
    assert by_date.loc[pd.Timestamp("2024-07-03")] == 3, "the complete session survives"
    assert by_date.loc[pd.Timestamp("2024-07-05")] == 0, "the incomplete one is dropped"

    cov = daily.attrs["coverage"]
    assert cov["n_sessions_complete"] == 1
    assert cov["n_sessions_incomplete"] == 1
    assert cov["incomplete_sessions"] == ["2024-07-05"]


def test_validate_scores_returns_stats_when_everything_is_present():
    heads, scores = _scored_headlines(4)
    stats = align.validate_scores(heads, scores)
    assert stats["ok"] and stats["n_unscored"] == 0
    assert stats["n_headlines"] == 4 and stats["n_scored_rows"] == 4


def test_extra_cached_scores_are_harmless():
    """The cache legitimately holds rows for headlines outside this request."""
    heads, scores = _scored_headlines(3)
    extra = pd.DataFrame({"headline_id": ["zz"]}
                         | {f"score_{s}": [0.9] for s in config.SCORERS})
    daily = align.aggregate_daily(pd.concat([scores, extra], ignore_index=True),
                                  heads, CALENDAR, date_only=True)
    assert daily.set_index("date")["n_headlines"].loc[pd.Timestamp("2024-07-03")] == 3
