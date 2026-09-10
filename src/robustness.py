"""Act 2 robustness exhibits (protocol §4 sensitivity set, §7; B26/R15).

These are **sensitivity analyses, not additional tests.** Protocol §6 is explicit:
they enter no correction family and cannot supply a significant result the
primary specification did not. Their job is to say whether the primary number
depends on a choice that could have gone another way.

The rule that governs how they are read: **divergence is reported as a finding
about the inference's fragility, not resolved by picking one.** If the bandwidth
sensitivity disagrees with the primary, the report says so; it does not adopt
whichever bandwidth is most convenient.

Every exhibit below re-runs the frozen primary specification with exactly one
choice changed, on the same eligibility rules, so a difference is attributable to
that choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import align, inference


def bandwidth_sensitivity(panel: pd.DataFrame, scorer: str = inference.PRIMARY_SCORER,
                          horizon: int = inference.PRIMARY_HORIZON) -> pd.DataFrame:
    """Protocol §4: `L ∈ {0, 1, 10}` and the data-driven plug-in, beside `L = 5`.

    `L = 0` is heteroskedasticity-robust only. The plug-in is Newey–West (1994),
    `floor(4 * (n/100)^(2/9))`, computed from `n` and never chosen by looking at
    which bandwidth gives the tidiest answer.
    """
    rows = []
    n = int(inference.primary(panel, scorer, horizon).n)
    plug_in = int(np.floor(4 * (n / 100) ** (2 / 9)))
    # `is_primary` marks the prespecified entry, not every row that happens to
    # share its bandwidth: the data-driven plug-in can coincide with L = 5 on a
    # small sample, and flagging two rows primary would misattribute which one
    # the protocol fixed in advance.
    plan = (("0 (HC only)", 0, False), ("1", 1, False),
            (f"{config.NW_MAXLAGS} (primary)", config.NW_MAXLAGS, True),
            ("10", 10, False), (f"plug-in ({plug_in})", plug_in, False))
    for label, maxlags, is_primary in plan:
        fit = inference.primary(panel, scorer, horizon, maxlags=maxlags)
        point, lo, hi = fit.bps
        rows.append({
            "bandwidth": label, "maxlags": maxlags, "n": fit.n,
            "bps_per_sd": point, "bps_lo95": lo, "bps_hi95": hi,
            "se_bps": (hi - point) / 1.959963984540054,
            "p": float(fit.fit.pvalues[fit.tone_term]),
            "is_primary": is_primary,
            "plug_in_equals_primary": plug_in == config.NW_MAXLAGS,
        })
    out = pd.DataFrame(rows)
    # Classify with the protocol's own rule (M2: non-strict bounds on endpoints
    # rounded to 0.1 bps, boundary cases flagged), never a second criterion
    # invented here. An earlier version compared unrounded endpoints directly and
    # disagreed with `classify_effect_interval` at L = 0 and L = 1, which would
    # have been reported as a conclusion flip that the stated rule does not make.
    verdicts = [inference.classify_effect_interval(r.bps_lo95, r.bps_hi95)
                for r in out.itertuples()]
    out["association_detected"] = [v["association_detected"] for v in verdicts]
    out["inside_sesoi"] = [v["effect_established_small"] for v in verdicts]
    out["boundary_case"] = [v["boundary_case"] for v in verdicts]
    out["conclusion"] = [v["conclusion"] for v in verdicts]
    return out


def aggregation_sensitivity(scores: pd.DataFrame, headlines: pd.DataFrame,
                            market: pd.DataFrame, calendar: pd.DatetimeIndex,
                            scorer: str = inference.PRIMARY_SCORER) -> pd.DataFrame:
    """D8 variant: median instead of mean daily aggregation.

    Rebuilds the panel from scratch under each rule, because `S_t` is defined at
    aggregation and cannot be recovered from a panel built the other way.
    """
    rows = []
    for agg in config.AGG_CHOICES:
        daily = align.aggregate_daily(scores, headlines, calendar, agg=agg)
        panel = align.build_panel(daily, market)
        fit = inference.primary(panel, scorer)
        point, lo, hi = fit.bps
        rows.append({
            "aggregation": agg, "n": fit.n, "tone_sd": fit.tone_sd,
            "bps_per_sd": point, "bps_lo95": lo, "bps_hi95": hi,
            "p": float(fit.fit.pvalues[fit.tone_term]),
            "is_primary": agg == config.AGG,
        })
    return pd.DataFrame(rows)


def subperiod_split(panel: pd.DataFrame, scorer: str = inference.PRIMARY_SCORER,
                    horizon: int = inference.PRIMARY_HORIZON) -> pd.DataFrame:
    """First half against second half of the eligible sample.

    The protocol's answer to non-stationarity: estimate on everything, then
    *show* whether the halves disagree, rather than shortening the window to
    avoid the question. Each half is standardized on its own rows, so the two
    coefficients are per-half-SD and comparable in the same units the primary
    uses.

    Returns a fourth row: the **difference** between the halves with its own
    interval. Comparing the two halves' intervals for overlap would be the wrong
    reading -- halving `n` widens each by about sqrt(2), so overlap is nearly
    guaranteed and would be mistaken for stability.
    """
    keep, _ = inference.eligibility(panel, scorer=scorer, horizon=horizon)
    positions = np.flatnonzero(keep)
    cut = positions[len(positions) // 2]
    dates = pd.to_datetime(panel["date"])

    rows = []
    for label, mask in (("full", np.ones(len(panel), dtype=bool)),
                        ("first half", (np.arange(len(panel)) < cut)),
                        ("second half", (np.arange(len(panel)) >= cut))):
        # Blank the tone outside the half rather than dropping rows: the panel
        # must stay the complete calendar or row shifts stop being session shifts.
        sub = panel.copy()
        sub.loc[~mask, f"s_{scorer}"] = np.nan
        try:
            fit = inference.primary(sub, scorer, horizon)
        except ValueError as exc:
            rows.append({"period": label, "n": 0, "note": str(exc)[:60]})
            continue
        point, lo, hi = fit.bps
        rows.append({
            "period": label, "n": fit.n,
            "start": str(dates[fit.mask].min().date()),
            "end": str(dates[fit.mask].max().date()),
            "bps_per_sd": point, "bps_lo95": lo, "bps_hi95": hi,
            "se_bps": (hi - point) / 1.959963984540054,
            "p": float(fit.fit.pvalues[fit.tone_term]),
        })
    out = pd.DataFrame(rows)

    # Test the DIFFERENCE, not two intervals for overlap.
    #
    # Reading a split by whether two intervals overlap is a weak and biased
    # reading: halving `n` widens each by ~sqrt(2), so overlap is close to
    # guaranteed and "the halves agree" would be an artefact of low power rather
    # than evidence of stability. The exhibit's question is whether the halves
    # DIFFER, and that has its own interval.
    #
    # The two halves are disjoint samples, so the estimators share no
    # observations and cov(b1, b2) = 0 up to dependence across the single split
    # boundary -- negligible at L = 5 against n > 1200 per half. Hence
    # var(diff) = var(b1) + var(b2), with no stacking required.
    halves = out[out["period"].isin(("first half", "second half"))]
    if len(halves) == 2 and "se_bps" in halves:
        first, second = halves.iloc[0], halves.iloc[1]
        diff = float(first["bps_per_sd"] - second["bps_per_sd"])
        se = float(np.hypot(first["se_bps"], second["se_bps"]))
        z = 1.959963984540054
        from scipy import stats as _stats
        out = pd.concat([out, pd.DataFrame([{
            "period": "difference (first - second)",
            "n": int(first["n"] + second["n"]),
            "bps_per_sd": diff, "bps_lo95": diff - z * se, "bps_hi95": diff + z * se,
            "se_bps": se,
            "p": float(2 * _stats.norm.sf(abs(diff / se))) if se > 0 else np.nan,
        }])], ignore_index=True)
    return out


def dispersion_floor_sensitivity(panel: pd.DataFrame,
                                 scorer: str = inference.PRIMARY_SCORER) -> pd.DataFrame:
    """Drop `n_t < 5` sessions from the primary specification too.

    `d_t` already requires it; the `S_t` specification does not. This asks
    whether thin-news sessions, where `S_t` is a mean over very few headlines and
    therefore noisy, are influencing the primary result.
    """
    rows = []
    for label, floor in (("none (primary)", 1),
                         (f"n_t >= {config.MIN_HEADLINES_FOR_DISPERSION}",
                          config.MIN_HEADLINES_FOR_DISPERSION)):
        sub = panel.copy()
        thin = pd.to_numeric(sub["n_headlines"], errors="coerce").fillna(0) < floor
        sub.loc[thin, f"s_{scorer}"] = np.nan
        fit = inference.primary(sub, scorer)
        point, lo, hi = fit.bps
        rows.append({
            "floor": label, "n": fit.n, "n_dropped": int(thin.sum()),
            "bps_per_sd": point, "bps_lo95": lo, "bps_hi95": hi,
            "p": float(fit.fit.pvalues[fit.tone_term]),
            "is_primary": floor == 1,
        })
    return pd.DataFrame(rows)


def residual_autocorrelation(panel: pd.DataFrame, lags: int = 20,
                             scorer: str = inference.PRIMARY_SCORER) -> pd.DataFrame:
    """Protocol §4 diagnostic: residual ACF to lag 20, and the regressor's own ACF.

    **Reported, not acted upon.** These describe whether `L = 5` was a reasonable
    prespecification. They do not license re-choosing `L` after the fact.
    """
    fit = inference.primary(panel, scorer)
    resid = np.asarray(fit.fit.ols.resid, dtype=float)
    tone = fit.design[fit.tone_term].to_numpy(dtype=float)

    def acf(x, k):
        x = x - x.mean()
        denom = float(x @ x)
        return float((x[k:] @ x[:-k]) / denom) if denom else np.nan

    n = len(resid)
    band = 1.959963984540054 / np.sqrt(n)
    return pd.DataFrame([{
        "lag": k,
        "residual_acf": acf(resid, k),
        "tone_acf": acf(tone, k),
        "abs_band_95": band,
        "residual_outside_band": abs(acf(resid, k)) > band,
    } for k in range(1, lags + 1)])


def summarise(panel: pd.DataFrame, scorer: str = inference.PRIMARY_SCORER) -> dict:
    """Does any sensitivity change the primary conclusion? Report, do not resolve."""
    primary_fit = inference.primary(panel, scorer)
    base_point, base_lo, base_hi = primary_fit.bps
    base = inference.classify_effect_interval(base_lo, base_hi)

    bands = bandwidth_sensitivity(panel, scorer)
    floors = dispersion_floor_sensitivity(panel, scorer)
    halves = subperiod_split(panel, scorer)

    verdicts = [inference.classify_effect_interval(r["bps_lo95"], r["bps_hi95"])
                for _, r in pd.concat([bands, floors], ignore_index=True).iterrows()]
    conclusions = [v["conclusion"] for v in verdicts]
    n_boundary = sum(bool(v["boundary_case"]) for v in verdicts)

    return {
        "primary_conclusion": base["conclusion"],
        "primary_boundary_case": bool(base["boundary_case"]),
        "conclusions_across_sensitivities": sorted(set(conclusions)),
        "any_sensitivity_changes_the_conclusion": len(set(conclusions)) > 1,
        "n_sensitivities_that_are_boundary_cases": n_boundary,
        "n_sensitivities": len(verdicts),
        "bandwidth_se_min_bps": float(bands["se_bps"].min()),
        "bandwidth_se_max_bps": float(bands["se_bps"].max()),
        "halves_intervals_overlap": bool(
            len(halves) == 3 and halves.iloc[1]["bps_lo95"] <= halves.iloc[2]["bps_hi95"]
            and halves.iloc[2]["bps_lo95"] <= halves.iloc[1]["bps_hi95"]
        ),
        "note": "Sensitivities enter no correction family and cannot supply a "
                "result the primary specification did not (protocol section 6). "
                "Divergence is reported as fragility, never resolved by selection.",
    }
