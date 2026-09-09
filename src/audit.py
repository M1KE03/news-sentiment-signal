"""Stage 0: the dataset audit that locks D1 and D4.

SUPERSEDED IN PART -- read before reusing (B03; docs/data-audit-fnspid.md).

    `timestamp_profile` and `compare_candidates` apply the intraday gate to a
    *whole file*. The B03 audit showed that is the wrong unit: FNSPID's
    All_external.csv concatenates five-plus sub-corpora with different
    languages, schemas and timestamp behaviour. Judged as one file it looks
    partly intraday, when in fact one block (Reuters) is intraday and every
    other block is date-only. The gate must be applied per source.

    `coverage_profile` and `suggest_window` further assume the input sample is
    representative of the corpus. The bounded audit sample is a *cluster*
    sample over tickers, so market-wide per-session rates cannot be estimated
    from it.

    Reworking this module to profile per source is the next increment. Until
    then these functions are sound only for a single-source input, and the
    conclusions in docs/data-audit-fnspid.md were produced by the analysis
    recorded there, not by compare_candidates.

The plan is explicit that no description of either candidate dataset is to be
trusted -- including its own. So every claim a dataset makes about itself is
checked here against the data, and each check returns a verdict rather than a
number for the reader to interpret.

The audit answers, per candidate, in the plan's priority order:
  1. are the intraday timestamps real?   `timestamp_profile`
  2. how good is coverage / duplication? `coverage_profile`, `duplication_profile`
  3. how long is the usable sample?      `coverage_profile`, `suggest_window`
and `compare_candidates` reduces those to a single choice with a stated reason.

Nothing here plots. Figures live in `src/plots.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import data as sdata

# A dataset whose stamps are really dates carries a huge mass at one clock time
# (usually midnight). Above this share we call it date-only regardless of what
# the column is named or what the documentation says.
DATE_ONLY_SHARE = 0.50
# Even a genuine intraday feed clusters on the minute; below this many distinct
# minutes-of-day the "intraday" stamps are not resolving anything useful.
MIN_DISTINCT_MINUTES = 60


def timestamp_profile(df: pd.DataFrame, ts_col: str = "ts_et") -> dict:
    """Is this an intraday feed, or a date-only one wearing a timestamp column?

    The tell is concentration: a date-only dump converted to datetimes puts
    every row at the same clock time. Returns the evidence and a verdict.
    """
    ts = pd.to_datetime(df[ts_col])
    if getattr(ts.dt, "tz", None) is None:
        raise ValueError(f"{ts_col} is timezone-naive; localize at load time (§4)")

    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    counts = minute_of_day.value_counts()
    modal_minute = int(counts.index[0])
    modal_share = float(counts.iloc[0] / len(ts))
    midnight_share = float((minute_of_day == 0).mean())
    distinct_minutes = int(minute_of_day.nunique())

    # Where does the news actually land relative to the 16:00 ET close? If
    # almost none of it is intraday, D6 is doing very little work.
    close_minute = config.MARKET_CLOSE_ET.hour * 60 + config.MARKET_CLOSE_ET.minute
    open_minute = 9 * 60 + 30

    looks_date_only = modal_share >= DATE_ONLY_SHARE or distinct_minutes < MIN_DISTINCT_MINUTES

    return {
        "n": int(len(ts)),
        "modal_minute": modal_minute,
        "modal_time": f"{modal_minute // 60:02d}:{modal_minute % 60:02d}",
        "modal_share": modal_share,
        "midnight_share": midnight_share,
        "distinct_minutes_of_day": distinct_minutes,
        "share_in_session": float(
            ((minute_of_day >= open_minute) & (minute_of_day <= close_minute)).mean()
        ),
        "share_after_close": float((minute_of_day > close_minute).mean()),
        "share_before_open": float((minute_of_day < open_minute).mean()),
        "looks_date_only": bool(looks_date_only),
        "verdict": (
            f"DATE-ONLY: {modal_share:.1%} of stamps land at "
            f"{modal_minute // 60:02d}:{modal_minute % 60:02d}; D6 cannot be applied "
            "-- invoke D2 (news dated d -> trading day d+1, RQ2 dropped)"
            if looks_date_only
            else f"INTRADAY: {distinct_minutes} distinct minutes-of-day, modal minute "
            f"holds only {modal_share:.1%}; D6 applies"
        ),
    }


def hour_histogram(df: pd.DataFrame, ts_col: str = "ts_et") -> pd.Series:
    """Headlines per hour-of-day (ET). The input to the audit's first figure."""
    ts = pd.to_datetime(df[ts_col])
    return ts.dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()


def coverage_profile(
    df: pd.DataFrame, calendar: pd.DatetimeIndex | None = None, ts_col: str = "ts_et"
) -> dict:
    """Headlines per year and per trading day, and the zero-news day share.

    Zero-news days are the D9 exclusion, so their count is a sample-size fact,
    not a footnote. `per_year` doubles as the D4 evidence: the window is the
    longest span with *stable* coverage, not simply the longest span available.
    """
    ts = pd.to_datetime(df[ts_col])
    per_year = ts.dt.year.value_counts().sort_index()

    out = {
        "n_headlines": int(len(df)),
        "first": ts.min(),
        "last": ts.max(),
        "per_year": per_year,
    }

    if calendar is None:
        return out

    from src import align

    day = align.map_to_trading_day(ts, calendar)
    unassignable = int(day.isna().sum())
    per_day = day.value_counts().reindex(pd.DatetimeIndex(calendar), fill_value=0)

    out.update(
        {
            "n_sessions": int(len(calendar)),
            "n_unassignable": unassignable,
            "per_day_mean": float(per_day.mean()),
            "per_day_median": float(per_day.median()),
            "per_day_p10": float(per_day.quantile(0.10)),
            "zero_news_days": int((per_day == 0).sum()),
            "zero_news_share": float((per_day == 0).mean()),
            "thin_days_share": float(
                (per_day < config.MIN_HEADLINES_FOR_DISPERSION).mean()
            ),  # these lose d_t under D9
            "per_day": per_day,
        }
    )
    return out


def duplication_profile(df: pd.DataFrame, **dedup_kwargs) -> dict:
    """Exact and near-duplicate rates, from the same `dedup` the pipeline uses.

    Syndicated headlines inflate n_t and distort S_t, so this rate is reported
    in the write-up rather than silently absorbed.
    """
    _, stats = sdata.dedup(df, **dedup_kwargs)
    return stats


def ticker_sanity(df: pd.DataFrame, n: int = 50, seed: int = config.SEED) -> pd.DataFrame:
    """A hand-checkable sample of headline/ticker pairs, plus tag-rate stats.

    Ticker tags are used only in the Stage 6 single-name spot check, so this is
    a sanity check, not a validation: read the 50 rows, do not score them.
    """
    rng = np.random.default_rng(seed)
    take = min(n, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    sample = df.iloc[np.sort(idx)][["text", "ts_et", "tickers"]].copy()
    sample["n_tickers"] = sample["tickers"].map(len)

    tagged = df["tickers"].map(len)
    sample.attrs["untagged_share"] = float((tagged == 0).mean())
    sample.attrs["mean_tags"] = float(tagged.mean())
    sample.attrs["most_covered"] = (
        pd.Series([t for tags in df["tickers"] for t in tags]).value_counts().head(10)
        if tagged.sum()
        else pd.Series(dtype=int)
    )
    return sample


def suggest_window(
    df: pd.DataFrame,
    ts_col: str = "ts_et",
    min_headlines_per_year: int = 5000,
    tol: float = 0.25,
) -> dict:
    """Propose D4: the longest *recent* run of years with stable coverage.

    Stability, not length, is the binding criterion -- a year with a tenth of
    the neighbouring coverage makes S_t a different measurement, which is the
    coverage-drift limitation the plan names. A proposal, not a decision: D4 is
    locked by hand after reading this and `per_year`.
    """
    ts = pd.to_datetime(df[ts_col])
    per_year = ts.dt.year.value_counts().sort_index()
    if per_year.empty:
        return {"ok": False, "reason": "no dated headlines"}

    median = float(per_year.median())
    floor = max(min_headlines_per_year, median * tol)
    ok_years = per_year[per_year >= floor].index.tolist()

    # Longest consecutive run of acceptable years; ties go to the later run.
    best: list[int] = []
    run: list[int] = []
    for y in ok_years:
        run = run + [y] if run and y == run[-1] + 1 else [y]
        if len(run) >= len(best):
            best = run

    if not best:
        return {"ok": False, "reason": f"no year clears {floor:,.0f} headlines"}

    n_years = len(best)
    return {
        "ok": n_years * 252 >= config.MIN_TRADING_DAYS,
        "start": f"{best[0]}-01-01",
        "end": f"{best[-1]}-12-31",
        "years": n_years,
        "approx_trading_days": n_years * 252,
        "floor_used": floor,
        "excluded_years": [int(y) for y in per_year.index if y not in best],
        "reason": (
            f"{n_years} consecutive years ({best[0]}-{best[-1]}) clear {floor:,.0f} "
            f"headlines/yr; ~{n_years * 252:,} trading days vs. the "
            f"{config.MIN_TRADING_DAYS:,} target"
        ),
    }


def audit_candidate(
    df: pd.DataFrame, name: str, calendar: pd.DatetimeIndex | None = None
) -> dict:
    """Run every check on one candidate and return the evidence in one dict."""
    return {
        "name": name,
        "timestamps": timestamp_profile(df),
        "coverage": coverage_profile(df, calendar),
        "duplication": duplication_profile(df),
        "window": suggest_window(df),
    }


def compare_candidates(audits: dict[str, dict]) -> pd.DataFrame:
    """Side-by-side summary and the D1 recommendation, in the plan's priority order.

    Criterion 1 is a gate, not a score: a date-only dataset is disqualified from
    D1 no matter how good its coverage is. If every candidate fails the gate,
    the recommendation is D2 -- and the deadline for that is the end of day 1.
    """
    rows = []
    for name, a in audits.items():
        ts, cov, dup, win = a["timestamps"], a["coverage"], a["duplication"], a["window"]
        rows.append(
            {
                "candidate": name,
                "intraday": not ts["looks_date_only"],
                "modal_share": ts["modal_share"],
                "distinct_minutes": ts["distinct_minutes_of_day"],
                "n_headlines": cov["n_headlines"],
                "dedup_rate": dup["dedup_rate"],
                "zero_news_share": cov.get("zero_news_share", np.nan),
                "median_per_day": cov.get("per_day_median", np.nan),
                "window_years": win.get("years", 0),
                "window_ok": win.get("ok", False),
            }
        )
    out = pd.DataFrame(rows).set_index("candidate")

    passing = out[out["intraday"]]
    if passing.empty:
        out.attrs["recommendation"] = "D2"
        out.attrs["reason"] = (
            "No candidate has usable intraday timestamps. Invoke D2 now: take the "
            "better date-only dataset, map news dated d to trading day d+1, and drop "
            "RQ2. Do not spend day 2 hunting for better data."
        )
        return out

    # Among candidates that clear the gate: fewer duplicates, then fewer
    # zero-news days, then a longer window.
    ranked = passing.sort_values(
        ["dedup_rate", "zero_news_share", "window_years"], ascending=[True, True, False]
    )
    pick = ranked.index[0]
    out.attrs["recommendation"] = pick
    out.attrs["reason"] = (
        f"{pick}: intraday timestamps confirmed ({ranked.loc[pick, 'distinct_minutes']} "
        f"distinct minutes-of-day, modal minute {ranked.loc[pick, 'modal_share']:.1%}), "
        f"dedup rate {ranked.loc[pick, 'dedup_rate']:.1%}, "
        f"{ranked.loc[pick, 'zero_news_share']:.1%} zero-news sessions, "
        f"{ranked.loc[pick, 'window_years']} usable years."
    )
    return out
