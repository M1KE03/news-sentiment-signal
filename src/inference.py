"""Act 2: HAC regressions, the FDR-controlled lag family, the placebo, effect sizes.

Every regression in the project is fitted here, with Newey-West standard errors
(D10) every time. Notebooks call these functions and display the result; they do
not fit models themselves.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import config


def nw_ols(y, X, maxlags: int = config.NW_MAXLAGS):
    """OLS with a constant and HAC (Newey-West) covariance.

    Rows with any missing value in y or X are dropped, which is the only
    listwise deletion in the project and is reported as `nobs`.
    """
    y = pd.Series(y).astype(float).reset_index(drop=True)
    X = pd.DataFrame(X).astype(float).reset_index(drop=True)
    frame = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if frame.empty:
        raise ValueError("no complete observations after dropping missing rows")
    yy = frame["__y__"]
    XX = sm.add_constant(frame.drop(columns="__y__"), has_constant="add")
    return sm.OLS(yy, XX).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


LAG_COLUMNS = ("ret_lag1", "rv_parkinson_lag1", "log_volume_lag1")


def _require_lags(panel: pd.DataFrame) -> None:
    """Refuse to run if the lag columns were not built on the full calendar.

    These must come from `align.build_panel`, which constructs them before any
    exclusion. Shifting inside a regression function computes the lag over the
    *retained* rows, so a Wednesday whose Tuesday was dropped as a zero-news
    session would take Monday's return as its lag -- a silently different
    control (B05/P12). Recomputing them here would reintroduce exactly that.
    """
    missing = [c for c in LAG_COLUMNS if c not in panel.columns]
    if missing:
        raise KeyError(
            f"panel is missing full-calendar lag column(s) {missing}. Build the "
            "panel with align.build_panel; do not shift inside a regression, "
            "which would lag over retained rows rather than trading sessions."
        )


def contemporaneous(
    panel: pd.DataFrame,
    scorer: str,
    maxlags: int = config.NW_MAXLAGS,
    allow_inadmissible: bool = False,
):
    """Section 6.1: ret_t on S_t plus lagged return, range variance and volume.

    Reported as association, not causation, in that language. Its absence is
    not evidence that the pipeline is broken (P23): a weak contemporaneous
    association can equally reflect aggregation, the chosen universe, or
    measurement noise, and is investigated as such rather than by adjusting the
    pipeline until it appears.

    Controls are read from the panel, never shifted here -- see `_require_lags`.

    **Refuses to run when `config.RQ2_ADMISSIBLE` is False.** The B03 audit found
    no sub-corpus that is both intraday-stamped and relevant, so for this corpus
    there is no verified within-day information boundary and a same-day
    coefficient cannot be interpreted. Suppression is structural rather than
    editorial: a number that is never computed cannot be quoted by accident,
    whereas one computed "for reference" reliably escapes into a table.
    """
    if not config.RQ2_ADMISSIBLE and not allow_inadmissible:
        raise RuntimeError(
            "the contemporaneous (same-day) specification is inadmissible for "
            "this corpus: config.RQ2_ADMISSIBLE is False because no selected "
            "source has usable intraday timestamps (docs/data-audit-fnspid.md). "
            "Without a within-day information boundary a same-day coefficient "
            "has no interpretation. Pass allow_inadmissible=True only for a "
            "deliberate, labelled methodological exhibit."
        )
    _require_lags(panel)
    X = pd.DataFrame(
        {
            f"s_{scorer}": panel[f"s_{scorer}"],
            "ret_lag1": panel["ret_lag1"],
            "rv_parkinson_lag1": panel["rv_parkinson_lag1"],
            "log_volume_lag1": panel["log_volume_lag1"],
        }
    )
    return nw_ols(panel["ret"], X, maxlags=maxlags)


def predictive(panel: pd.DataFrame, scorer: str, horizon: int, maxlags: int = config.NW_MAXLAGS):
    """Section 6.2: ret_{t+h} on S_t, controlling for momentum, volatility, attention.

    The control set is deliberately short. A kitchen-sink control set is a
    specification search wearing a lab coat.
    """
    df = panel
    X = pd.DataFrame(
        {
            f"s_{scorer}": df[f"s_{scorer}"],
            "ret": df["ret"],
            "rv_parkinson": df["rv_parkinson"],
            "log_volume": df["log_volume"],
        }
    )
    return nw_ols(df[f"ret_lead{horizon}"], X, maxlags=maxlags)


def lag_family(
    panel: pd.DataFrame,
    scorer: str,
    horizons: Sequence[int] = config.HORIZONS,
    maxlags: int = config.NW_MAXLAGS,
) -> pd.DataFrame:
    """Unadjusted return estimates for one scorer, ready for family assembly.

    Correction belongs to `adjust_return_families` after all three scorers have
    been collected. A scorer's five horizons are not a correction family.
    """
    rows = []
    key = f"z_s_{scorer}"
    for h in horizons:
        fitted = primary(panel, scorer, h, maxlags=maxlags)
        res = fitted.fit
        rows.append(
            {
                "scorer": scorer,
                "horizon": h,
                "coef": float(res.params[key]),
                "nw_se": float(res.bse[key]),
                "t": float(res.tvalues[key]),
                "p": float(res.pvalues[key]),
                "nobs": int(res.nobs),
                "coefficient_scale": "standardized_tone",
                "tone_sd": fitted.tone_sd,
                "hac_convention": fitted.convention,
                "alternate_nw_se": float(fitted.alternate.bse[key]),
            }
        )
    return pd.DataFrame(rows)


RETURN_TESTS = tuple((sc, h) for sc in ("finbert", "lm", "vader") for h in range(1, 6))
PRIMARY_RETURN_TEST = ("finbert", 1)


def adjust_return_families(estimates: pd.DataFrame, q: float = config.FDR_Q) -> pd.DataFrame:
    """Protocol section 6: one primary and exactly 14 secondary return tests.

    Accepts already computed estimates without fitting anything. Missing,
    duplicate or extra tests are errors, including failed fits with missing p.
    The primary has no adjusted p-value or correction decision. Secondary
    claims use BY when BH and BY disagree; intervals remain pointwise.
    """
    required = {"scorer", "horizon", "p"}
    if not required.issubset(estimates.columns):
        raise ValueError(f"return estimates require columns {sorted(required)}")
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must lie strictly between zero and one")
    if "outcome" in estimates and not estimates["outcome"].eq("ret").all():
        raise ValueError("the secondary family contains return tests only (outcome='ret')")
    horizons = estimates["horizon"]
    if any(isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer))
           for h in horizons):
        raise ValueError("return horizons must be integers from 1 through 5")
    pairs = list(zip(estimates["scorer"], horizons))
    if len(pairs) != 15 or len(set(pairs)) != 15 or set(pairs) != set(RETURN_TESTS):
        raise ValueError("expected exactly the 15 declared return tests: one primary and 14 secondary")
    try:
        p = estimates["p"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("return p-values must be finite numbers in [0, 1]") from exc
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("return p-values must be finite numbers in [0, 1]")
    # Canonical order makes exports invariant to caller ordering/index labels.
    out = estimates.assign(p=p).set_index(["scorer", "horizon"]).loc[list(RETURN_TESTS)].reset_index()
    secondary = ~((out["scorer"] == "finbert") & (out["horizon"] == 1))
    out["family"] = np.where(secondary, "secondary_return", "primary_return")
    out["family_size"] = np.where(secondary, 14, 1)
    out["interval_scope"] = "pointwise"
    out["alpha"] = q
    for method, prefix in (("fdr_bh", "bh"), ("fdr_by", "by")):
        reject, adjusted = multipletests(out.loc[secondary, "p"], alpha=q, method=method)[:2]
        out[f"{prefix}_q"] = np.nan
        out[f"{prefix}_reject"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        out.loc[secondary, f"{prefix}_q"] = adjusted
        out.loc[secondary, f"{prefix}_reject"] = reject
    out["bh_by_disagree"] = (out["bh_reject"] != out["by_reject"]).fillna(False)
    out["claim_reject"] = out["p"] <= q
    out.loc[secondary, "claim_reject"] = out.loc[secondary, "by_reject"].astype(bool)
    out["claim_basis"] = np.where(secondary, "BY", "unadjusted")
    return out


def _circular_block_permute(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Reassemble x from randomly placed circular blocks of fixed length.

    Preserves the series' own short-run autocorrelation while destroying its
    alignment with the outcome -- an assumption-free null (D12). The same
    instinct as a block bootstrap: never let a resampling scheme destroy the
    dependence you are worried about.
    """
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return x[idx[:n]]


def permutation_pvalue(
    panel: pd.DataFrame,
    scorer: str,
    horizon: int = config.PLACEBO_HORIZON,
    n: int = config.PERMUTATION_DRAWS,
    block: int = config.PERMUTATION_BLOCK,
    seed: int = config.SEED,
    maxlags: int = config.NW_MAXLAGS,
) -> dict:
    """D12. Two-sided permutation p-value for the horizon-h t-statistic."""
    rng = np.random.default_rng(seed)
    key = f"s_{scorer}"
    observed = float(predictive(panel, scorer, horizon, maxlags=maxlags).tvalues[key])

    df = panel.copy()
    s = df[key].to_numpy(dtype=float)
    null = np.empty(n)
    for i in range(n):
        df[key] = _circular_block_permute(s, block, rng)
        null[i] = predictive(df, scorer, horizon, maxlags=maxlags).tvalues[key]

    # +1 top and bottom: under the null the observed statistic is itself one
    # draw, so an exact zero is not an available p-value.
    p = (1 + np.sum(np.abs(null) >= abs(observed))) / (n + 1)
    return {
        "scorer": scorer,
        "horizon": horizon,
        "t_observed": observed,
        "p_permutation": float(p),
        "n_draws": n,
        "block": block,
        "seed": seed,
        "null": null,
    }


def advance_precision(
    outcome, tone, controls: pd.DataFrame, *, kappa: float, hac_spacing: str,
) -> dict:
    """Compute the two section-5 planning widths without fitting tone to returns.

    Inputs must already be the identical eligible rows, in identical order.
    No row is dropped here. R07c must supply its controls-only residual HAC/OLS
    SE ratio at L=5 and its selected spacing explicitly; neither is guessed.
    The returned JSON-compatible record must be persisted before estimation.
    This utility does not enforce that sequencing in a caller.
    """
    expected = ["ret", "rv_parkinson", "log_volume_detrended"]
    if set(controls.columns) != set(expected) or len(controls.columns) != 3:
        raise ValueError(f"precision controls must be exactly {expected}")
    for value in (outcome, tone):
        if isinstance(value, pd.Series) and not value.index.equals(controls.index):
            raise ValueError("precision inputs must have identical row indices")
    y, s = np.asarray(outcome, dtype=float), np.asarray(tone, dtype=float)
    x = controls[expected].to_numpy(dtype=float)
    n = len(x)
    if y.shape != (n,) or s.shape != (n,) or n <= 4:
        raise ValueError("precision requires more than four identical, complete rows")
    if not all(np.isfinite(a).all() for a in (y, s, x)):
        raise ValueError("precision inputs must be finite; determine eligibility first")
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be a positive finite controls-only HAC/OLS SE ratio")
    if hac_spacing not in {"retained_position", "session_indexed"}:
        raise ValueError("declare hac_spacing as retained_position or session_indexed")
    sd_s, sd_y = float(s.std(ddof=1)), float(y.std(ddof=1))
    if sd_s <= 0 or sd_y <= 0:
        raise ValueError("tone and outcome must have positive sample standard deviations")
    design = np.column_stack([np.ones(n), x])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("precision controls and constant must have full column rank")
    z = (s - s.mean()) / sd_s
    # Deliberately separate fits: the outcome is never regressed on tone.
    u = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    v = z - design @ np.linalg.lstsq(design, z, rcond=None)[0]
    sd_u, sd_v = float(u.std(ddof=1)), float(v.std(ddof=1))
    if sd_v <= np.sqrt(np.finfo(float).eps) or sd_u <= sd_y * np.sqrt(np.finfo(float).eps):
        raise ValueError("precision is undefined for a degenerate residual fit")
    multiplier = 1.96 * config.BPS_PER_UNIT / np.sqrt(n)
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "n": n, "sd_r": sd_y, "sd_u": sd_u, "sd_v": sd_v,
        "tone_controls_r_squared": float(1 - sd_v ** 2),
        "kappa": float(kappa), "kappa_source": "caller_supplied_controls_only",
        "hac_spacing": hac_spacing, "hac_maxlags": 5, "sd_ddof": 1,
        "h1_bps": float(multiplier * sd_y),
        "h2_bps": float(multiplier * sd_u / sd_v * kappa),
        "purpose": "advance_planning_estimates_not_bounds",
    }


def primary_precision(panel: pd.DataFrame, *, convention: str = config.HAC_CONVENTION) -> dict:
    """Prepare the primary advance record on R07b's rows, without estimating beta.

    Operational kappa: HAC SE of the mean of controls-only residuals divided
    by their OLS mean SE, sd(u, ddof=1)/sqrt(n). The HAC sandwich uses an
    intercept-only design, L=5, and R07c's selected session pairing.
    """
    keep, ledger = eligibility(panel)
    positions = np.flatnonzero(keep)
    _assert_lead_adjacency(panel, positions, PRIMARY_HORIZON)
    rows = panel.loc[keep]
    y = rows["ret_lead1"].astype(float)
    tone = rows["s_finbert"].astype(float)
    controls = rows[list(PRIMARY_CONTROLS)].astype(float)
    record = advance_precision(y, tone, controls, kappa=1.0, hac_spacing=convention)
    design = np.column_stack([np.ones(len(rows)), controls.to_numpy()])
    u = y.to_numpy() - design @ np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0]
    ols_mean_se = record["sd_u"] / np.sqrt(len(rows))
    ratios = {}
    for spacing in HAC_CONVENTIONS:
        cov = hac_sandwich(np.ones((len(rows), 1)), u, 5,
                           positions if spacing == "session_indexed" else None)
        variance = float(cov[0, 0])
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("controls-only residual mean HAC variance must be positive and finite")
        ratios[spacing] = float(np.sqrt(variance) / ols_mean_se)
    record["kappa"] = ratios[convention]
    record["kappa_source"] = "controls_only_residual_mean_hac_se_over_sample_sd_mean_se"
    record["h2_bps"] *= ratios[convention]
    record["kappa_by_spacing"] = ratios
    record["h2_bps_by_spacing"] = {
        spacing: record["h2_bps"] * ratio / ratios[convention]
        for spacing, ratio in ratios.items()
    }
    record["eligibility"] = ledger
    record["session_positions"] = positions.tolist()
    record["session_dates"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d").tolist()
    return record


def classify_effect_interval(lo: float, hi: float) -> dict:
    """Two evidence dimensions on closed, 0.1-bps rounded pointwise endpoints.

    NumPy's nearest-even tie rounding is used. Original endpoints are retained
    separately; proximity to zero or +/-SESOI is checked before rounding.
    """
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError("interval endpoints must be finite and ordered")
    margin = config.SESOI_BPS
    lower, upper = float(np.round(lo, 1)), float(np.round(hi, 1))
    association = lower > 0 or upper < 0
    small = lower >= -margin and upper <= margin
    conclusions = {
        (True, True): "Association detected, and small against the 5 bps yardstick",
        (True, False): "Association detected; magnitude not established as small",
        (False, True): "No association detected, and precise enough to exclude effects beyond +/-5 bps",
        (False, False): "Inconclusive",
    }
    boundary = any(abs(endpoint - mark) <= 0.1 + 1e-12
                   for endpoint in (lo, hi) for mark in (-margin, 0, margin))
    return {
        "interval_scope": "pointwise", "sesoi_bps": margin,
        "bps_lo95_rounded": lower, "bps_hi95_rounded": upper,
        "association_detected": association, "effect_established_small": small,
        "conclusion": conclusions[(association, small)], "boundary_case": boundary,
        "boundary_endpoints_bps": f"[{lo:.4f}, {hi:.4f}]" if boundary else "",
        "boundary_note": (
            "Boundary case: classification is sensitive at the 0.1 bps reporting precision."
            if boundary else ""
        ),
    }


def effect_size_bps(coef: float, se: float, sigma_s: float = 1.0) -> tuple[float, float, float]:
    """D15. Basis points of return per 1 sigma move in S_t, with a 95% NW CI.

    The interval is transformed identically to the point estimate, so a null
    reads as a ruled-out interval -- effects larger than X bps per 1 sigma are
    excluded at 95% -- rather than as a shrug.
    """
    if not np.isfinite([coef, se, sigma_s]).all() or se < 0 or sigma_s <= 0:
        raise ValueError("effect scale requires finite values, nonnegative SE and positive tone SD")
    scale = sigma_s * config.BPS_PER_UNIT
    point = coef * scale
    half = 1.96 * se * scale
    return point, point - half, point + half


def effect_size_table(family: pd.DataFrame, sigma_s: float = 1.0) -> pd.DataFrame:
    """Pointwise effects and both section-5 evidence dimensions.

    Standardized coefficients use the default scale of one. For legacy raw
    coefficients the caller must supply the fitted sample's tone SD.
    """
    out = family.copy()
    vals = [effect_size_bps(c, s, sigma_s) for c, s in zip(out["coef"], out["nw_se"])]
    out[["bps_per_sd", "bps_lo95", "bps_hi95"]] = pd.DataFrame(vals, index=out.index)
    classifications = pd.DataFrame(
        [classify_effect_interval(lo, hi) for lo, hi in zip(out["bps_lo95"], out["bps_hi95"])],
        index=out.index,
    )
    out = out.drop(columns=["clears_costs", "ruled_out_above_bps"], errors="ignore")
    for column in classifications:
        out[column] = classifications[column]
    return out


def attenuation_comparison(
    panel: pd.DataFrame,
    scorers: Sequence[str] = ("finbert", "lm"),
    horizon: int = 1,
    maxlags: int = config.NW_MAXLAGS,
) -> pd.DataFrame:
    """Section 6.3. Identical specification, one scorer swapped for the other.

    Errors-in-variables predicts the better classifier carries the larger |beta|
    and the relatively smaller standard error. Read this next to the daily
    correlation of the two S_t series (reported in `.attrs["corr_s"]`): if they
    are near-collinear the comparison is bounded and inconclusive, and is
    reported that way rather than redesigned until it separates them.
    """
    rows = []
    for sc in scorers:
        res = predictive(panel, sc, horizon, maxlags=maxlags)
        key = f"s_{sc}"
        rows.append(
            {
                "scorer": sc,
                "coef": float(res.params[key]),
                "abs_coef": abs(float(res.params[key])),
                "nw_se": float(res.bse[key]),
                "t": float(res.tvalues[key]),
                "p": float(res.pvalues[key]),
                "sd_s": float(panel[key].std()),
                "nobs": int(res.nobs),
            }
        )
    out = pd.DataFrame(rows)
    if len(scorers) == 2:
        out.attrs["corr_s"] = float(panel[f"s_{scorers[0]}"].corr(panel[f"s_{scorers[1]}"]))
    return out


def horse_race(panel: pd.DataFrame, horizon: int = 1, maxlags: int = config.NW_MAXLAGS):
    """Section 6.3 secondary exhibit: both scores in one regression.

    Expect multicollinearity to blur this. Quote the correlation alongside it
    rather than reading a muddy horse race as evidence of anything.
    """
    X = pd.DataFrame(
        {
            "s_finbert": panel["s_finbert"],
            "s_lm": panel["s_lm"],
            "ret": panel["ret"],
            "rv_parkinson": panel["rv_parkinson"],
            "log_volume": panel["log_volume"],
        }
    )
    return nw_ols(panel[f"ret_lead{horizon}"], X, maxlags=maxlags)


def volatility_spec(panel: pd.DataFrame, scorer: str, maxlags: int = config.NW_MAXLAGS):
    """Section 6.4, range variance: next-day rv_parkinson on level, intensity and dispersion.

    `rv_parkinson` is the intraday range-based variance proxy, not total daily
    volatility: it uses the high-low range only, so variation within the range
    and overnight moves are invisible to it (P22).
    """
    X = pd.DataFrame(
        {
            f"s_{scorer}": panel[f"s_{scorer}"],
            f"abs_s_{scorer}": panel[f"s_{scorer}"].abs(),
            f"d_{scorer}": panel[f"d_{scorer}"],
            "rv_parkinson": panel["rv_parkinson"],
        }
    )
    return nw_ols(panel["rv_parkinson_lead1"], X, maxlags=maxlags)


def volume_spec(panel: pd.DataFrame, scorer: str, maxlags: int = config.NW_MAXLAGS):
    """Section 6.4, volume: next-day detrended log volume, same regressors plus |ret_t|.

    The dispersion coefficient is the one to read first: disagreement predicting
    volume is the most plausible positive finding in the project, and it is a
    documented one (Tetlock 2007).
    """
    X = pd.DataFrame(
        {
            f"s_{scorer}": panel[f"s_{scorer}"],
            f"abs_s_{scorer}": panel[f"s_{scorer}"].abs(),
            f"d_{scorer}": panel[f"d_{scorer}"],
            "log_volume_detrended": panel["log_volume_detrended"],
            "abs_ret": panel["ret"].abs(),
        }
    )
    return nw_ols(panel["log_volume_detrended_lead1"], X, maxlags=maxlags)


def analysis_sample(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """D9. Drop zero-news trading days and report how many were dropped."""
    keep = panel["n_headlines"] > 0
    stats = {
        "n_sessions": int(len(panel)),
        "n_zero_news_days": int((~keep).sum()),
        "n_analysis_days": int(keep.sum()),
    }
    return panel[keep].reset_index(drop=True), stats


# =====================================================================
# R07b / R07c: the primary specification, its sample, and its HAC spacing
# =====================================================================
#
# Audit A07: `predictive` below fits raw `log_volume` where the inference
# protocol Sections 1-3 specify the trailing-63-session-detrended series, does not
# standardize tone at fit time, and reports zero-news exclusions only. That is
# a different model from the frozen one, so this section implements the frozen
# one rather than patching that function; `predictive` is retained for the
# secondary/exploratory paths R08 replaces.
#
# Audit A09 / amendment M7: what the prespecified bandwidth `L = 5` counts.
# See `hac_sandwich` and `HAC_CONVENTIONS`.

PRIMARY_SCORER = "finbert"
PRIMARY_HORIZON = 1

# Inference protocol Section 2: frozen before any coefficient is examined. The
# volume control is the DETRENDED series -- `log_volume` raw is a different
# regressor with a strong secular trend, not a renamed column (A07).
PRIMARY_CONTROLS = ("ret", "rv_parkinson", "log_volume_detrended")

# Ordered, and the order is part of the contract: a row is charged to the FIRST
# reason that applies, so the counts partition the excluded rows exactly and the
# ledger reconciles. Per-reason counts ignoring the order are reported too, so
# overlap between reasons stays visible rather than being hidden by precedence.
EXCLUSION_REASONS = ("no_lead_return", "zero_news", "missing_tone", "missing_control")


def _require_full_calendar_panel(panel: pd.DataFrame, horizon: int) -> None:
    """Refuse a panel whose rows are not the complete exchange calendar.

    The lead and lag columns are built once, by `align.build_panel`, over every
    session. That is what makes a row shift a session shift. Passing an
    already-filtered frame here -- the output of `analysis_sample`, say -- would
    leave those columns correct but make row `i + h` no longer the `h`-th next
    session, so the adjacency assertion below would be checking the wrong rows.

    Detection is cheap and does not need the calendar: on a complete frame the
    stored `ret_lead{h}` is exactly `ret.shift(-h)`. Drop any interior row and
    the two disagree at the hole.
    """
    dates = pd.to_datetime(panel["date"])
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("panel dates must be unique and sorted ascending")

    lead = f"ret_lead{horizon}"
    if lead not in panel.columns:
        raise KeyError(
            f"panel has no '{lead}'; build it with align.build_panel rather than "
            "shifting here, which would shift over retained rows"
        )
    stored = pd.to_numeric(panel[lead], errors="coerce")
    reshift = pd.to_numeric(panel["ret"], errors="coerce").shift(-horizon)
    both = stored.notna() & reshift.notna()
    disagree = int((both & ~np.isclose(stored.where(both, 0.0), reshift.where(both, 0.0))).sum())
    disagree += int((stored.isna() != reshift.isna()).sum())
    if disagree:
        raise ValueError(
            f"'{lead}' does not equal ret.shift(-{horizon}) on {disagree} row(s). "
            "This panel is not the complete exchange calendar -- rows have been "
            "removed since it was built, so row i+h is no longer the h-th next "
            "session. Pass the full panel from align.build_panel; eligibility is "
            "applied here."
        )


def _assert_lead_adjacency(panel: pd.DataFrame, positions: np.ndarray, horizon: int) -> None:
    """Inference protocol Section 3: assert, do not assume, what supplies `r_(t+h)`.

    For each retained row the target must be the return of the session `horizon`
    places later in the exchange calendar -- not merely a non-missing number
    sitting in a column with that name (B05 defect D-2).
    """
    lead = pd.to_numeric(panel[f"ret_lead{horizon}"], errors="coerce").to_numpy()
    ret = pd.to_numeric(panel["ret"], errors="coerce").to_numpy()
    n = len(panel)
    for p in positions:
        q = int(p) + horizon
        if q >= n:
            raise AssertionError(
                f"retained row {p} has no session {horizon} ahead inside the panel; "
                "the window's last sessions must be excluded, not extrapolated"
            )
        if not np.isclose(lead[p], ret[q]):
            raise AssertionError(
                f"row {p}: ret_lead{horizon} is {lead[p]!r} but the return of the "
                f"session {horizon} ahead (row {q}) is {ret[q]!r}. The target is "
                "not the adjacent session's return."
            )


def eligibility(
    panel: pd.DataFrame,
    scorer: str = PRIMARY_SCORER,
    horizon: int = PRIMARY_HORIZON,
) -> tuple[np.ndarray, dict]:
    """Inference protocol Section 3, as an auditable ledger rather than a dropna.

    The three eligibility rules become four recorded reasons, because "the
    controls are defined" and "the tone is defined" fail for different causes
    and a single count would not distinguish the 62-session detrend warm-up from
    an unscored session.

    Returns the boolean mask and a ledger whose first-reason counts sum exactly
    to the number of excluded rows.
    """
    _require_full_calendar_panel(panel, horizon)

    tone_col = f"s_{scorer}"
    for col in (tone_col, "n_headlines", *PRIMARY_CONTROLS):
        if col not in panel.columns:
            raise KeyError(f"panel is missing '{col}', required by the primary specification")

    num = lambda c: pd.to_numeric(panel[c], errors="coerce")

    # Rule 1: the next session's return must exist. This is also where the
    # window's final session and any missing SPY row are removed.
    no_lead = num(f"ret_lead{horizon}").isna().to_numpy()
    # Rule 2: at least one headline assigned to t.
    zero_news = (num("n_headlines").fillna(0) < 1).to_numpy()
    # A session with headlines but no score is not a zero-news session; it is an
    # incomplete scoring run, and R04b's gate should have caught it upstream.
    missing_tone = num(tone_col).isna().to_numpy()
    # Rule 3: every control defined. log_volume_detrended is missing for the
    # first 62 sessions by construction (trailing 63-session window).
    missing_control = np.zeros(len(panel), dtype=bool)
    for c in PRIMARY_CONTROLS:
        missing_control |= num(c).isna().to_numpy()

    flags = {
        "no_lead_return": no_lead,
        "zero_news": zero_news,
        "missing_tone": missing_tone,
        "missing_control": missing_control,
    }

    keep = ~(no_lead | zero_news | missing_tone | missing_control)

    first_reason = {}
    claimed = np.zeros(len(panel), dtype=bool)
    for reason in EXCLUSION_REASONS:
        hit = flags[reason] & ~claimed
        first_reason[reason] = int(hit.sum())
        claimed |= flags[reason]

    ledger = {
        "n_panel_rows": int(len(panel)),
        "n_eligible": int(keep.sum()),
        "n_excluded": int((~keep).sum()),
        "reason_order": list(EXCLUSION_REASONS),
        "excluded_first_reason": first_reason,
        "excluded_any_reason": {r: int(flags[r].sum()) for r in EXCLUSION_REASONS},
        "scorer": scorer,
        "horizon": int(horizon),
        "controls": list(PRIMARY_CONTROLS),
    }
    assert sum(first_reason.values()) == ledger["n_excluded"], "exclusion ledger does not reconcile"
    return keep, ledger


# ------------------------------------------------------ R07c: HAC spacing

# Audit A09, amendment M7. `L = 5` was prespecified as "one trading week"
# (inference protocol Section 4). Whether the code delivers that depends on what a
# lag counts, and the two available answers differ wherever the analysis sample
# has gaps:
#
#   retained_position -- lag `l` is `l` ROWS of the analysis sample. This is
#       what `nw_ols` does: it resets the index, so statsmodels pairs
#       observation i with observation i-l regardless of how far apart in
#       calendar time they sit. Across a gap, "one week" silently becomes more.
#       The installed statsmodels `cov_hac_simple` documents an assumption of
#       consecutive, equally spaced periods, which a gapped sample violates.
#
#   session_indexed -- lag `l` is `l` EXCHANGE SESSIONS. Pairs further apart
#       than `L` sessions get zero weight even when they are adjacent rows, and
#       pairs exactly `l` sessions apart get the Bartlett weight for `l` even
#       when rows between them were excluded.
#
# The two are the SAME computation with a different pairing rule, which is why
# `hac_sandwich` takes the session index as an argument rather than
# implementing two kernels: passing `arange(n)` reproduces retained-position
# exactly, so any difference between the conventions is attributable to spacing
# and to nothing else.
HAC_CONVENTIONS = ("session_indexed", "retained_position")


def _bartlett_weights(maxlags: int) -> np.ndarray:
    """w_0 = 1, w_l = 1 - l/(L+1) -- the triangular kernel statsmodels uses."""
    if maxlags < 0:
        raise ValueError("maxlags must be non-negative")
    return 1.0 - np.arange(maxlags + 1) / (maxlags + 1.0)


def hac_sandwich(
    X: np.ndarray,
    resid: np.ndarray,
    maxlags: int,
    session_index: Sequence[int] | None = None,
) -> np.ndarray:
    """Newey-West sandwich covariance with an explicit spacing rule.

    `(X'X)^-1 S (X'X)^-1`, with

        S = sum_l w_l * (Omega_l + Omega_l')

    where `Omega_l` accumulates `xu_i' xu_j` over the pairs whose SESSION
    distance is `l`. With `session_index=None` the session index is the row
    position, every pair `l` rows apart is `l` apart in session distance, and
    this reduces term by term to `statsmodels.stats.sandwich_covariance`'s
    `S_hac_simple` -- verified against the library in `tests/test_hac_spacing.py`
    rather than asserted here. statsmodels applies no small-sample correction on
    this path, so neither does this.

    The Bartlett kernel is positive definite on the line, so the irregular-grid
    form is still a positive semi-definite estimator; the caller checks the
    realised diagonal rather than trusting that.
    """
    X = np.asarray(X, dtype=float)
    u = np.asarray(resid, dtype=float).ravel()
    n, _ = X.shape
    if len(u) != n:
        raise ValueError(f"resid has {len(u)} rows, X has {n}")

    if session_index is None:
        s = np.arange(n, dtype=np.int64)
    else:
        s = np.asarray(session_index, dtype=np.int64).ravel()
        if len(s) != n:
            raise ValueError(f"session_index has {len(s)} entries, X has {n} rows")
        if len(np.unique(s)) != n:
            raise ValueError("session_index must be unique: one row per session")
        if not np.all(np.diff(s) > 0):
            raise ValueError("session_index must be strictly increasing")

    xu = X * u[:, None]
    XtXi = np.linalg.inv(X.T @ X)
    w = _bartlett_weights(maxlags)

    S = xu.T @ xu                      # lag 0, every observation with itself
    where = {int(v): i for i, v in enumerate(s)}
    for lag in range(1, maxlags + 1):
        rows_i, rows_j = [], []
        for i, v in enumerate(s):
            j = where.get(int(v) - lag)
            if j is not None:
                rows_i.append(i)
                rows_j.append(j)
        if not rows_i:
            continue
        omega = xu[rows_i].T @ xu[rows_j]
        S += w[lag] * (omega + omega.T)

    return XtXi @ S @ XtXi


class HACFit:
    """An OLS fit carrying a named HAC convention alongside its covariance.

    The two conventions must be comparable coefficient by coefficient, so both
    come back as this type: identical point estimates (OLS does not depend on
    the covariance), and standard errors that differ only through the spacing
    rule. Inference is asymptotic-normal, matching statsmodels' `cov_type="HAC"`
    (`use_t=False`).
    """

    def __init__(self, ols, cov: np.ndarray, convention: str, maxlags: int):
        self.ols = ols
        self.convention = convention
        self.maxlags = int(maxlags)
        self.names = list(ols.params.index)
        self.params = pd.Series(np.asarray(ols.params, dtype=float), index=self.names)
        self.cov = pd.DataFrame(np.asarray(cov, dtype=float), index=self.names, columns=self.names)
        self.nobs = int(ols.nobs)
        self.df_resid = int(ols.df_resid)
        self.rsquared = float(ols.rsquared)

    @property
    def bse(self) -> pd.Series:
        var = np.diag(self.cov.to_numpy())
        if np.any(var < 0):
            bad = [n for n, v in zip(self.names, var) if v < 0]
            raise ValueError(
                f"HAC variance is negative for {bad} under the {self.convention} "
                "convention; the covariance estimate is not positive semi-definite"
            )
        return pd.Series(np.sqrt(var), index=self.names)

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        from scipy import stats as _stats

        return pd.Series(
            2.0 * _stats.norm.sf(np.abs(self.tvalues.to_numpy())), index=self.names
        )

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        from scipy import stats as _stats

        z = _stats.norm.ppf(1.0 - alpha / 2.0)
        half = z * self.bse
        return pd.DataFrame({"lower": self.params - half, "upper": self.params + half})


def fit_hac(
    y,
    X,
    maxlags: int = config.NW_MAXLAGS,
    session_index: Sequence[int] | None = None,
    convention: str = "session_indexed",
) -> HACFit:
    """Fit OLS on complete rows and attach the requested HAC covariance.

    Unlike `nw_ols`, this does NOT silently drop missing rows: the sample is
    decided by `eligibility` and recorded in a ledger, so a missing value
    reaching this point is a defect upstream and says so.
    """
    if convention not in HAC_CONVENTIONS:
        raise ValueError(f"convention must be one of {HAC_CONVENTIONS}, got {convention!r}")
    y = pd.Series(y).astype(float)
    X = pd.DataFrame(X).astype(float)
    if y.isna().any() or X.isna().any().any():
        raise ValueError(
            "fit_hac received missing values; the analysis sample must be "
            "complete by construction (see `eligibility`), not by dropna"
        )
    if convention == "retained_position":
        session_index = None

    Xc = sm.add_constant(X.reset_index(drop=True), has_constant="add")
    ols = sm.OLS(y.reset_index(drop=True), Xc).fit()
    cov = hac_sandwich(Xc.to_numpy(), ols.resid.to_numpy(), maxlags, session_index)
    return HACFit(ols, cov, convention, maxlags)


# ------------------------------------------- R07b: the primary specification


def bps_interval(coef: float, se: float, alpha: float = 0.05) -> tuple[float, float, float]:
    """Inference protocol Section 5, for an ALREADY standardized regressor.

    `beta` is log return per 1 SD of tone because `z(S_t)` was standardized at
    fit time, so the scale is `beta * 10,000` and nothing else. Multiplying by
    an `sd(S)` computed on some other sample -- which `effect_size_bps` below
    does, and which the runner fed the broader positive-news sample's SD (A07)
    -- would rescale the coefficient a second time.
    """
    from scipy import stats as _stats

    z = _stats.norm.ppf(1.0 - alpha / 2.0)
    point = coef * config.BPS_PER_UNIT
    half = z * se * config.BPS_PER_UNIT
    return point, point - half, point + half


class PrimaryFit:
    """The frozen primary specification, its sample, and both HAC conventions.

    Inference protocol Section 1:  `r_(t+1) = a + beta z(S_t) + gamma' X_t + e`,
    `X_t = {r_t, RV_t, lv_t}`, FinBERT, `h = 1`, HAC `L = 5`.

    Every quantity that depends on the sample -- the fit, `sd(S)` used to
    standardize, and therefore the basis-point scale -- is computed from the
    SAME retained rows. That was the A07 defect: tone was not standardized at
    fit time and the bps conversion used a different sample's SD, so the
    reported effect size was not the fitted coefficient in different units.
    """

    def __init__(self, *, panel, mask, ledger, design, target, dates, positions,
                 tone_mean, tone_sd, fits, convention, scorer, horizon, maxlags):
        self.mask = mask
        self.ledger = ledger
        self.design = design
        self.target = target
        self.dates = dates
        self.positions = positions
        self.tone_mean = float(tone_mean)
        self.tone_sd = float(tone_sd)
        self.fits = fits
        self.convention = convention
        self.scorer = scorer
        self.horizon = int(horizon)
        self.maxlags = int(maxlags)
        self.tone_term = f"z_s_{scorer}"

    @property
    def fit(self) -> HACFit:
        """The primary convention's fit."""
        return self.fits[self.convention]

    @property
    def alternate(self) -> HACFit:
        """The other convention, reported alongside and never substituted."""
        other = [c for c in HAC_CONVENTIONS if c != self.convention][0]
        return self.fits[other]

    @property
    def n(self) -> int:
        return int(self.fit.nobs)

    @property
    def beta(self) -> float:
        return float(self.fit.params[self.tone_term])

    @property
    def se(self) -> float:
        return float(self.fit.bse[self.tone_term])

    @property
    def bps(self) -> tuple[float, float, float]:
        return bps_interval(self.beta, self.se)

    def spacing_comparison(self) -> pd.DataFrame:
        """Both conventions side by side, for every term. Section 4 requires both."""
        rows = []
        for name in self.fit.names:
            a = self.fits["session_indexed"]
            b = self.fits["retained_position"]
            rows.append(
                {
                    "term": name,
                    "coef": float(a.params[name]),
                    "se_session_indexed": float(a.bse[name]),
                    "se_retained_position": float(b.bse[name]),
                    "se_ratio": float(a.bse[name] / b.bse[name]),
                }
            )
        out = pd.DataFrame(rows)
        out.attrs["primary_convention"] = self.convention
        out.attrs["maxlags"] = self.maxlags
        return out


def primary(
    panel: pd.DataFrame,
    scorer: str = PRIMARY_SCORER,
    horizon: int = PRIMARY_HORIZON,
    maxlags: int = config.NW_MAXLAGS,
    convention: str = config.HAC_CONVENTION,
) -> PrimaryFit:
    """Fit the frozen primary specification on the eligible sessions.

    Takes the FULL panel from `align.build_panel`, not a pre-filtered one:
    eligibility is applied here so it can be counted, and the session positions
    it retains are what the session-indexed HAC needs (R07c).

    Sequence note. This computes `beta`. The protocol's requirement that the two
    anticipated half-widths be recorded BEFORE the tone coefficient is inspected
    is a requirement on the reporting path (R07d), which owns the precision
    contract; it is not enforced by this function.
    """
    keep, ledger = eligibility(panel, scorer=scorer, horizon=horizon)
    if not keep.any():
        raise ValueError(f"no eligible sessions; exclusion ledger: {ledger}")

    positions = np.flatnonzero(keep)
    _assert_lead_adjacency(panel, positions, horizon)

    rows = panel.loc[keep]
    tone = pd.to_numeric(rows[f"s_{scorer}"], errors="coerce").astype(float)
    tone_mean = float(tone.mean())
    tone_sd = float(tone.std(ddof=1))
    if not np.isfinite(tone_sd) or tone_sd == 0.0:
        raise ValueError(
            f"s_{scorer} has zero or undefined standard deviation on the "
            f"{int(keep.sum())} eligible sessions; z(S_t) is undefined"
        )

    design = pd.DataFrame(
        {f"z_s_{scorer}": (tone - tone_mean) / tone_sd},
        index=rows.index,
    )
    for c in PRIMARY_CONTROLS:
        design[c] = pd.to_numeric(rows[c], errors="coerce").astype(float)
    target = pd.to_numeric(rows[f"ret_lead{horizon}"], errors="coerce").astype(float)

    fits = {
        conv: fit_hac(target, design, maxlags=maxlags, session_index=positions, convention=conv)
        for conv in HAC_CONVENTIONS
    }
    ledger = dict(ledger)
    ledger["n_fitted"] = int(fits[convention].nobs)
    ledger["tone_mean"] = tone_mean
    ledger["tone_sd"] = tone_sd
    ledger["hac_convention"] = convention
    ledger["hac_maxlags"] = int(maxlags)

    return PrimaryFit(
        panel=panel, mask=keep, ledger=ledger, design=design, target=target,
        dates=pd.to_datetime(rows["date"]).reset_index(drop=True),
        positions=positions, tone_mean=tone_mean, tone_sd=tone_sd,
        fits=fits, convention=convention, scorer=scorer, horizon=horizon,
        maxlags=maxlags,
    )
