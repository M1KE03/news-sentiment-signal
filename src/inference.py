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
            "source has usable intraday timestamps (docs/archive/data-audit-fnspid.md). "
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


# `_circular_block_permute` and `permutation_pvalue` were removed at R08c
# (audit A08 / B19). They drew blocks **with replacement**, so some observations
# appeared twice and others not at all -- not a permutation -- and the `+1`
# correction on the resulting p-value did not repair a null distribution built
# the wrong way. The output was also labelled `p_permutation` and plotted as a
# p-value, which protocol Section 11 forbids. `timing_diagnostic` replaces them
# with a full circular shift and a percentile rank. The functions were deleted
# rather than deprecated: a callable that returns a field named `p_permutation`
# is an invitation to quote it.


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


# `attenuation_comparison` and `horse_race` were removed at R08b (audit A08).
#
# The first fitted each scorer separately and tabulated `abs_coef` beside two
# separate t-statistics, then applied a correlation threshold that declared the
# comparison "inconclusive" on its own. None of that tests `beta_A - beta_B`,
# which is the quantity a "one beats the other" claim is about (P07), and
# protocol Section 11 forbids the claim without an interval for that difference.
# `paired_scorer_contrast` supplies the interval, with the cross-equation
# covariance in it, and reports collinearity as a diagnostic instead of a rule.
#
# The second put unstandardized scores against raw `log_volume`, so it was
# neither the frozen control set nor a comparison of comparable coefficients
# (A07). `incremental_contribution` replaces it, labelled as the different
# estimand it is.
#
# Both were deleted rather than deprecated, for the same reason as
# `permutation_pvalue`: a callable that returns `abs_coef` and a per-scorer
# p-value is an invitation to quote it as a comparison.


# `volatility_spec` and `volume_spec` were removed when RQ4 was implemented to
# the protocol (audit A08). They fitted **unstandardized** `s_` terms, printed a
# per-coefficient t and p for each with no joint test and no correction, and ran
# on a different sample from the primary -- the arrangement in which "the
# dispersion coefficient" becomes a headline after someone has looked at it.
# `rq4_family` (joint HAC Wald, BH-corrected) and `rq4_coefficients`
# (descriptive) replace them. Deleted rather than deprecated: a function
# returning a per-coefficient p-value for an exploratory term is an invitation
# to quote it.


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

    meat = hac_meat(X * u[:, None], maxlags, session_index)
    XtXi = np.linalg.inv(X.T @ X)
    return XtXi @ meat @ XtXi


def _session_positions(n: int, session_index: Sequence[int] | None) -> np.ndarray:
    """Validate the spacing rule's index. `None` means: rows are the sessions."""
    if session_index is None:
        return np.arange(n, dtype=np.int64)
    s = np.asarray(session_index, dtype=np.int64).ravel()
    if len(s) != n:
        raise ValueError(f"session_index has {len(s)} entries, X has {n} rows")
    if len(np.unique(s)) != n:
        raise ValueError("session_index must be unique: one row per session")
    if not np.all(np.diff(s) > 0):
        raise ValueError("session_index must be strictly increasing")
    return s


def hac_meat(
    scores: np.ndarray,
    maxlags: int,
    session_index: Sequence[int] | None = None,
) -> np.ndarray:
    """The `S` in the sandwich: Bartlett-weighted moment autocovariances.

        S = sum_l w_l * (Omega_l + Omega_l'),
        Omega_l = sum over pairs (i, j) at SESSION distance l of  g_i' g_j

    `scores` holds one moment vector per observation. For a single equation
    that is `x_t * u_t`; for the stacked two-equation system of protocol
    Section 7(a) it is the two equations' moment vectors concatenated, which is
    exactly how the cross-equation covariance enters `delta`'s standard error.

    **This is the only place the spacing rule is implemented.** R07c's argument
    rests on the two conventions being one computation with one differing input
    (`docs/archive/hac-spacing-decision.md`); a second copy of this loop would quietly make
    that false, so the stacked estimator calls this rather than reimplementing
    the pairing.
    """
    g = np.asarray(scores, dtype=float)
    if g.ndim != 2:
        raise ValueError("scores must be a 2-D array with one row per observation")
    n = g.shape[0]
    s = _session_positions(n, session_index)
    w = _bartlett_weights(maxlags)

    S = g.T @ g                        # lag 0, every observation with itself
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
        omega = g[rows_i].T @ g[rows_j]
        S += w[lag] * (omega + omega.T)
    return S


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


# =====================================================================
# R08b: comparing scorers (protocol Section 7)
# =====================================================================
#
# Audit A08. `attenuation_comparison` compared raw coefficient magnitudes and
# separate t-statistics, and `horse_race` fitted unstandardized scores against
# raw `log_volume`. Neither supports the claim it invites. Protocol Section 11
# forbids claiming one scorer beats another without an interval for their
# paired difference, and Section 7 says how to build one.
#
# Two different estimands, kept apart:
#   (a) `paired_scorer_contrast` -- delta = beta_A - beta_B, each from its own
#       regression, on the identical observation set, with the cross-equation
#       HAC covariance in delta's standard error.
#   (b) `incremental_contribution` -- both standardized scores in one
#       regression: does one add information given the other?


def common_eligibility(
    panel: pd.DataFrame,
    scorers: Sequence[str] = ("finbert", "lm"),
    horizon: int = PRIMARY_HORIZON,
) -> tuple[np.ndarray, dict]:
    """Rows eligible for **every** named scorer, with each scorer's own ledger.

    Protocol Section 7(a) requires the identical observation set. A scorer missing
    tone on a session it would otherwise be eligible for removes that session
    from the comparison for both, and the count is recorded rather than
    absorbed: a difference computed on two different samples is not a paired
    difference at all.
    """
    scorers = tuple(scorers)
    if len(scorers) < 2 or len(set(scorers)) != len(scorers):
        raise ValueError(f"need at least two distinct scorers, got {scorers}")

    masks, ledgers = {}, {}
    for sc in scorers:
        masks[sc], ledgers[sc] = eligibility(panel, scorer=sc, horizon=horizon)

    keep = np.logical_and.reduce([masks[sc] for sc in scorers])
    ledger = {
        "scorers": list(scorers),
        "horizon": int(horizon),
        "n_panel_rows": int(len(panel)),
        "n_common": int(keep.sum()),
        "per_scorer": {sc: ledgers[sc] for sc in scorers},
        "n_lost_to_common_sample": {
            sc: int(masks[sc].sum() - keep.sum()) for sc in scorers
        },
    }
    return keep, ledger


def _design_for(rows: pd.DataFrame, scorer: str) -> tuple[pd.DataFrame, float, float]:
    """[const, z(S), controls] on the given rows, standardized on those rows."""
    tone = pd.to_numeric(rows[f"s_{scorer}"], errors="coerce").astype(float)
    mean, sd = float(tone.mean()), float(tone.std(ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        raise ValueError(
            f"s_{scorer} has zero or undefined standard deviation on the "
            f"{len(rows)} common sessions; z(S) is undefined"
        )
    design = pd.DataFrame({f"z_s_{scorer}": (tone - mean) / sd}, index=rows.index)
    for c in PRIMARY_CONTROLS:
        design[c] = pd.to_numeric(rows[c], errors="coerce").astype(float)
    design = sm.add_constant(design, has_constant="add")
    return design, mean, sd


def _vif(design: pd.DataFrame, term: str) -> float:
    """Variance inflation for one regressor against the rest of its own design."""
    others = [c for c in design.columns if c not in (term, "const")]
    if not others:
        return 1.0
    y = design[term].to_numpy(dtype=float)
    X = sm.add_constant(design[others].to_numpy(dtype=float), has_constant="add")
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("inf")
    r2 = 1.0 - float((resid**2).sum()) / ss_tot
    return float("inf") if r2 >= 1.0 else float(1.0 / (1.0 - r2))


class PairedContrast:
    """`delta = beta_A - beta_B` with the cross-equation covariance included.

    The two equations are fitted separately -- stacking a block-diagonal design
    gives exactly the separate OLS estimates -- but the covariance is computed
    over the **full** parameter vector, from each observation's concatenated
    moment vector. That is what puts `cov(beta_A, beta_B)` into `delta`'s
    standard error. Because the two tone series are strongly correlated, that
    covariance is large and positive, so ignoring it (as two separate fits do)
    substantially overstates the uncertainty about their difference.
    """

    def __init__(self, *, scorers, horizon, maxlags, convention, ledger, mask,
                 positions, dates, designs, target, fits, params, cov, tone_sd,
                 diagnostics):
        self.scorers = tuple(scorers)
        self.horizon = int(horizon)
        self.maxlags = int(maxlags)
        self.convention = convention
        self.ledger = ledger
        self.mask = mask
        self.positions = positions
        self.dates = dates
        self.designs = designs
        self.target = target
        self.fits = fits
        self.params = params          # Series, keys "<scorer>::<term>"
        self.cov = cov                # DataFrame over the same keys
        self.tone_sd = tone_sd
        self.diagnostics = diagnostics

    @property
    def n(self) -> int:
        return len(self.target)

    def _tone_key(self, scorer: str) -> str:
        return f"{scorer}::z_s_{scorer}"

    def beta(self, scorer: str) -> float:
        return float(self.params[self._tone_key(scorer)])

    @property
    def delta(self) -> float:
        a, b = self.scorers[0], self.scorers[1]
        return self.beta(a) - self.beta(b)

    @property
    def se_delta(self) -> float:
        a, b = (self._tone_key(sc) for sc in self.scorers[:2])
        var = (
            float(self.cov.loc[a, a])
            + float(self.cov.loc[b, b])
            - 2.0 * float(self.cov.loc[a, b])
        )
        if var < 0:
            if var > -1e-12 * max(abs(float(self.cov.loc[a, a])), 1.0):
                return 0.0
            raise ValueError(
                f"variance of delta is negative ({var:.3g}); the stacked HAC "
                f"covariance is not positive semi-definite under {self.convention}"
            )
        return float(np.sqrt(var))

    @property
    def degenerate(self) -> bool:
        """True when the two scorers carry the same information on this sample.

        Then `delta` and its variance are both exactly zero and no Wald
        statistic exists. Reported as a fact about the inputs rather than
        rescued with a floor.
        """
        return self.se_delta == 0.0

    def wald(self) -> dict:
        """Two-sided test of `delta = 0`, plus the pointwise interval in bps."""
        from scipy import stats as _stats

        delta, se = self.delta, self.se_delta
        if self.degenerate:
            return {
                "delta": delta, "se": se, "z": float("nan"), "p": float("nan"),
                "wald_chi2": float("nan"), "degenerate": True,
                "bps": (0.0, 0.0, 0.0), "interval_scope": "pointwise",
            }
        z = delta / se
        return {
            "delta": delta, "se": se, "z": float(z),
            "p": float(2.0 * _stats.norm.sf(abs(z))),
            "wald_chi2": float(z**2), "degenerate": False,
            "bps": bps_interval(delta, se), "interval_scope": "pointwise",
        }

    def table(self) -> pd.DataFrame:
        """One row per scorer plus the contrast, all on the same sample."""
        rows = []
        for sc in self.scorers:
            key = self._tone_key(sc)
            se = float(np.sqrt(self.cov.loc[key, key]))
            point, lo, hi = bps_interval(self.beta(sc), se)
            rows.append({
                "quantity": sc, "estimate": self.beta(sc), "se": se,
                "bps_per_sd": point, "bps_lo95": lo, "bps_hi95": hi,
                "n": self.n, "tone_sd": self.tone_sd[sc],
            })
        w = self.wald()
        point, lo, hi = w["bps"]
        rows.append({
            "quantity": f"delta({self.scorers[0]} - {self.scorers[1]})",
            "estimate": w["delta"], "se": w["se"],
            "bps_per_sd": point, "bps_lo95": lo, "bps_hi95": hi,
            "n": self.n, "tone_sd": np.nan,
        })
        out = pd.DataFrame(rows)
        out["interval_scope"] = "pointwise"
        out["coefficient_scale"] = "standardized_tone"
        out["hac_convention"] = self.convention
        out.attrs["wald"] = w
        out.attrs["diagnostics"] = self.diagnostics
        out.attrs["ledger"] = self.ledger
        return out


def paired_scorer_contrast(
    panel: pd.DataFrame,
    scorers: Sequence[str] = ("finbert", "lm"),
    horizon: int = PRIMARY_HORIZON,
    maxlags: int = config.NW_MAXLAGS,
    convention: str = config.HAC_CONVENTION,
) -> PairedContrast:
    """Protocol Section 7(a). The interval Section 11 requires before any "beats" claim.

    Both scorers standardized on the **common** sample, so `delta` is a
    difference of comparable quantities: replacing `S` with `2S` changes neither
    coefficient, which raw magnitudes cannot promise (Section 7, last paragraph).

    Collinearity is reported, never acted on. `corr(z_A, z_B)` and the tone
    VIFs go into `.diagnostics`; a high correlation widens `delta`'s interval
    and that widening is the honest statement of what the data can distinguish.
    There is no threshold at which this function declares itself inconclusive.
    """
    if convention not in HAC_CONVENTIONS:
        raise ValueError(f"convention must be one of {HAC_CONVENTIONS}, got {convention!r}")
    scorers = tuple(scorers)
    keep, ledger = common_eligibility(panel, scorers, horizon)
    if int(keep.sum()) <= len(PRIMARY_CONTROLS) + 2:
        raise ValueError(f"too few common sessions to fit the contrast; ledger: {ledger}")

    positions = np.flatnonzero(keep)
    _assert_lead_adjacency(panel, positions, horizon)
    rows = panel.loc[keep]
    target = pd.to_numeric(rows[f"ret_lead{horizon}"], errors="coerce").astype(float)
    if target.isna().any():
        raise ValueError("the common sample contains a missing target; eligibility failed")

    designs, tone_sd, params, moments, breads, fits = {}, {}, {}, [], [], {}
    for sc in scorers:
        design, _, sd = _design_for(rows, sc)
        if design.isna().any().any():
            raise ValueError(f"design for {sc} contains missing values on the common sample")
        designs[sc], tone_sd[sc] = design, sd

        Xa = design.to_numpy(dtype=float)
        beta = np.linalg.lstsq(Xa, target.to_numpy(dtype=float), rcond=None)[0]
        resid = target.to_numpy(dtype=float) - Xa @ beta
        for term, value in zip(design.columns, beta):
            params[f"{sc}::{term}"] = float(value)
        moments.append(Xa * resid[:, None])
        breads.append(np.linalg.inv(Xa.T @ Xa))
        fits[sc] = {"beta": beta, "resid": resid, "design": design}

    session_index = positions if convention == "session_indexed" else None
    meat = hac_meat(np.hstack(moments), maxlags, session_index)
    from scipy.linalg import block_diag as _block_diag

    bread = _block_diag(*breads)
    cov = bread @ meat @ bread

    keys = list(params)
    cov = pd.DataFrame(cov, index=keys, columns=keys)

    z_a = designs[scorers[0]][f"z_s_{scorers[0]}"].to_numpy(dtype=float)
    z_b = designs[scorers[1]][f"z_s_{scorers[1]}"].to_numpy(dtype=float)
    diagnostics = {
        "corr_tone": float(np.corrcoef(z_a, z_b)[0, 1]),
        "vif_tone": {sc: _vif(designs[sc], f"z_s_{sc}") for sc in scorers},
        "n_common": int(len(target)),
        "note": "collinearity is a diagnostic; it does not decide the comparison",
    }

    return PairedContrast(
        scorers=scorers, horizon=horizon, maxlags=maxlags, convention=convention,
        ledger=ledger, mask=keep, positions=positions,
        dates=pd.to_datetime(rows["date"]).reset_index(drop=True),
        designs=designs, target=target, fits=fits,
        params=pd.Series(params), cov=cov, tone_sd=tone_sd, diagnostics=diagnostics,
    )


def incremental_contribution(
    panel: pd.DataFrame,
    scorers: Sequence[str] = ("finbert", "lm"),
    horizon: int = PRIMARY_HORIZON,
    maxlags: int = config.NW_MAXLAGS,
    convention: str = config.HAC_CONVENTION,
) -> HACFit:
    """Protocol Section 7(b). Both standardized scores in one regression.

    A **different estimand** from the Section 7(a) contrast and labelled as such:
    this asks whether one scorer adds information *given* the other, not which
    marginal association is larger. Expect the correlation between the two
    series to blur it; the paired contrast, not this, is what supports a
    comparison claim.

    Supersedes `horse_race`, which used unstandardized scores and raw
    `log_volume` (A07/A08).
    """
    scorers = tuple(scorers)
    keep, ledger = common_eligibility(panel, scorers, horizon)
    positions = np.flatnonzero(keep)
    _assert_lead_adjacency(panel, positions, horizon)
    rows = panel.loc[keep]

    design = pd.DataFrame(index=rows.index)
    for sc in scorers:
        tone = pd.to_numeric(rows[f"s_{sc}"], errors="coerce").astype(float)
        sd = float(tone.std(ddof=1))
        if not np.isfinite(sd) or sd == 0.0:
            raise ValueError(f"s_{sc} has zero or undefined standard deviation on the common sample")
        design[f"z_s_{sc}"] = (tone - float(tone.mean())) / sd
    for c in PRIMARY_CONTROLS:
        design[c] = pd.to_numeric(rows[c], errors="coerce").astype(float)

    target = pd.to_numeric(rows[f"ret_lead{horizon}"], errors="coerce").astype(float)
    session_index = positions if convention == "session_indexed" else None
    fit = fit_hac(target, design, maxlags=maxlags, session_index=session_index,
                  convention=convention)
    fit.ledger = ledger
    fit.estimand = "incremental_contribution_given_the_other_scorer"
    return fit


# =====================================================================
# R08c: the timing diagnostic (protocol Section 9)
# =====================================================================
#
# Audit A08 / B19. The superseded `permutation_pvalue` drew blocks **with
# replacement**, so some observations appeared twice and others not at all. That
# is not a permutation, and the `+1` correction on its p-value did not repair a
# null distribution built the wrong way. It also labelled its output
# `p_permutation` and plotted it as a p-value, which Section 11 forbids: shifting
# tone destroys its relationship with the controls, so the resulting spread is
# not the null distribution of the *conditional* coefficient.
#
# What replaces it is a full circular shift -- a bijection, every observation
# used exactly once, the tone series' autocorrelation preserved exactly -- and
# a percentile rank that is never called a p-value.


def _shift_coefficients(z: np.ndarray, y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Tone coefficient for every circular shift `k = 0 .. n-1`, via Frisch-Waugh.

    The controls and the outcome are identical for every `k` -- only the tone
    column moves -- so partialling them out once is exact and turns `n` full
    refits into `n` inner products. `tests/test_timing_diagnostic.py` checks the
    result against direct refits rather than taking the algebra on trust.
    """
    n = len(z)
    Xc = np.column_stack([np.ones(n), controls])
    # Residualize once; the controls and the outcome never move.
    solve = np.linalg.pinv(Xc)
    y_r = y - Xc @ (solve @ y)

    # The shifted tone must be residualized AFTER shifting: projection and
    # rotation do not commute, so partialling out first and rolling the
    # residual afterwards gives a different -- wrong -- coefficient.
    out = np.empty(n)
    for k in range(n):
        zk = np.roll(z, -k)
        zt = zk - Xc @ (solve @ zk)
        denom = float(zt @ zt)
        out[k] = float(zt @ y_r) / denom if denom > 0 else np.nan
    return out


def timing_diagnostic(
    panel: pd.DataFrame,
    scorer: str = PRIMARY_SCORER,
    horizon: int = PRIMARY_HORIZON,
    convention: str = config.HAC_CONVENTION,
    round_bps: float = 0.1,
) -> dict:
    """Protocol Section 9. A descriptive percentile rank -- never a p-value.

    The standardized tone of the `n` retained rows, **in session order after
    eligibility**, is circularly shifted by every `k` in `1 .. n-1`; outcomes and
    controls stay in place; the tone coefficient is recorded for each. The
    observed `k = 0` coefficient is ranked within that reference distribution.

    Shifting is a permutation of the retained rows, so `n` is identical for
    every `k` and the coefficients are comparable across `k`. The alternative --
    shifting on the full calendar and re-applying eligibility -- was considered
    and rejected in Section 9 because it changes `n` with `k`. The cost of that
    choice is recorded in the output rather than left implicit: **when the
    retained rows have gaps, a shift of `k` positions is not a shift of `k`
    calendar sessions**, so `n_gaps` and `largest_gap_sessions` are returned for
    reporting beside the percentile.

    Ties use the midrank convention, evaluated at the reporting precision of the
    coefficient (0.1 bps), and the tie count is returned. On continuous data it
    should be zero; a nonzero count means something in the fit is degenerate and
    is surfaced rather than absorbed.
    """
    # Check the sample size before fitting: an under-identified design would
    # otherwise emit a rank-deficiency warning from inside `primary` and reach
    # this guard with a less useful message.
    keep, _ = eligibility(panel, scorer=scorer, horizon=horizon)
    if int(keep.sum()) < 3:
        raise ValueError(
            f"the timing diagnostic needs at least three retained rows, got "
            f"{int(keep.sum())}"
        )

    fitted = primary(panel, scorer=scorer, horizon=horizon, convention=convention)
    z = fitted.design[fitted.tone_term].to_numpy(dtype=float)
    y = fitted.target.to_numpy(dtype=float)
    controls = fitted.design[list(PRIMARY_CONTROLS)].to_numpy(dtype=float)
    n = len(z)
    coefs = _shift_coefficients(z, y, controls)
    observed = float(coefs[0])
    shifted = coefs[1:]                      # k = 1 .. n-1
    if not np.isfinite(observed) or not np.isfinite(shifted).all():
        raise ValueError("a shifted fit was degenerate; the diagnostic is not reported")

    # Frisch-Waugh gives the same coefficient as the full fit; assert it rather
    # than assume it, because the whole diagnostic hangs off this equality.
    if not np.isclose(observed, fitted.beta, rtol=1e-8, atol=1e-14):
        raise AssertionError(
            f"partialled-out coefficient {observed!r} does not match the fitted "
            f"{fitted.beta!r}; the shift path is not the primary specification"
        )

    scale = config.BPS_PER_UNIT
    quantum = round_bps / scale
    obs_r = np.round(observed / quantum) * quantum
    sh_r = np.round(shifted / quantum) * quantum
    below = int((sh_r < obs_r).sum())
    ties = int((sh_r == obs_r).sum())
    percentile = (below + 0.5 * ties) / n

    positions = fitted.positions
    steps = np.diff(positions)
    gaps = steps[steps > 1] - 1

    return {
        "scorer": scorer,
        "horizon": int(horizon),
        "statistic": "tone coefficient under circular shift of standardized tone",
        "observed_coef": observed,
        "observed_bps": observed * scale,
        "percentile": float(percentile),
        "n": int(n),
        "n_shifts": int(len(shifted)),
        "n_below": below,
        "n_ties": ties,
        "tie_precision_bps": float(round_bps),
        "shift_domain": "retained analysis rows in session order, after eligibility",
        "shifted_coef_min": float(shifted.min()),
        "shifted_coef_max": float(shifted.max()),
        "shifted_coef_mean": float(shifted.mean()),
        "shifted_bps_p05": float(np.quantile(shifted, 0.05) * scale),
        "shifted_bps_p95": float(np.quantile(shifted, 0.95) * scale),
        "n_gaps": int(len(gaps)),
        "largest_gap_sessions": int(gaps.max()) if len(gaps) else 0,
        "hac_convention": convention,
        "is_p_value": False,
        "reporting_note": (
            "Descriptive timing diagnostic reported as a percentile rank. It is "
            "NOT a p-value for H0: beta = 0 and is not assumption-free: shifting "
            "tone also destroys its relationship with the controls, so this is "
            "not the null distribution of the conditional coefficient. The "
            "inferential statement comes from the HAC interval alone. A shift of "
            "k positions is not a shift of k calendar sessions where the "
            "retained rows have gaps; n_gaps and largest_gap_sessions quantify that."
        ),
        "shifted_coefficients": shifted,
    }


# =====================================================================
# RQ4: the exploratory family (protocol section 6, P21)
# =====================================================================
#
# Audit A08: `volatility_spec` and `volume_spec` fitted **unstandardized** `s_`
# terms, reported a per-coefficient t and p for each, applied no correction, and
# ran on a different sample from the primary. That is precisely the arrangement
# in which "the dispersion coefficient" gets promoted to a headline after
# someone has looked at it -- which section 6 and P21 exist to prevent.
#
# What the protocol specifies instead:
#
#   RV_(t+1) = a + b1 z(S_t) + b2 z(|S_t|) + b3 z(d_t) + phi RV_t + e
#   lv_(t+1) = a + b1 z(S_t) + b2 z(|S_t|) + b3 z(d_t) + phi lv_t + psi |r_t| + e
#
# **The primary test for each is a HAC Wald test of the joint null
# b1 = b2 = b3 = 0.** Individual coefficients are descriptive afterwards, with
# pointwise intervals. BH within the family of Wald tests.

RQ4_OUTCOMES = ("rv_parkinson_lead1", "log_volume_detrended_lead1")
RQ4_TONE_TERMS = ("z_s", "z_abs_s", "z_d")


def rq4_design(panel: pd.DataFrame, scorer: str, outcome: str) -> tuple[np.ndarray, dict]:
    """Eligible rows for one RQ4 equation, with its own reason ledger.

    `d_t` is defined only when `n_t >= 5`, so this sample is smaller than the
    primary's and the reduction is reported rather than absorbed (section 6).
    """
    if outcome not in RQ4_OUTCOMES:
        raise ValueError(f"outcome must be one of {RQ4_OUTCOMES}, got {outcome!r}")
    _require_full_calendar_panel(panel, PRIMARY_HORIZON)

    num = lambda c: pd.to_numeric(panel[c], errors="coerce")
    control = "rv_parkinson" if outcome.startswith("rv_") else "log_volume_detrended"
    needed = [f"s_{scorer}", f"d_{scorer}", outcome, control]
    if outcome.startswith("log_volume"):
        needed.append("ret")

    flags = {
        "no_outcome": num(outcome).isna().to_numpy(),
        "zero_news": (num("n_headlines").fillna(0) < 1).to_numpy(),
        "missing_tone": num(f"s_{scorer}").isna().to_numpy(),
        # d_t is NaN below the dispersion floor by construction, not by defect
        "dispersion_undefined": num(f"d_{scorer}").isna().to_numpy(),
        "missing_control": np.zeros(len(panel), dtype=bool),
    }
    for c in needed:
        if c not in (f"d_{scorer}", outcome, f"s_{scorer}"):
            flags["missing_control"] |= num(c).isna().to_numpy()

    keep = ~np.logical_or.reduce(list(flags.values()))
    order = ("no_outcome", "zero_news", "missing_tone", "dispersion_undefined",
             "missing_control")
    first, claimed = {}, np.zeros(len(panel), dtype=bool)
    for reason in order:
        hit = flags[reason] & ~claimed
        first[reason] = int(hit.sum())
        claimed |= flags[reason]

    ledger = {
        "scorer": scorer, "outcome": outcome, "control": control,
        "n_panel_rows": int(len(panel)), "n_eligible": int(keep.sum()),
        "n_excluded": int((~keep).sum()), "reason_order": list(order),
        "excluded_first_reason": first,
        "n_lost_to_dispersion_floor": int(flags["dispersion_undefined"].sum()),
        "dispersion_floor": int(config.MIN_HEADLINES_FOR_DISPERSION),
    }
    assert sum(first.values()) == ledger["n_excluded"], "RQ4 ledger does not reconcile"
    return keep, ledger


def rq4_fit(panel: pd.DataFrame, scorer: str, outcome: str,
            maxlags: int = config.NW_MAXLAGS,
            convention: str = config.HAC_CONVENTION):
    """One RQ4 equation, all three tone terms standardized on its own sample."""
    keep, ledger = rq4_design(panel, scorer, outcome)
    if int(keep.sum()) <= 8:
        raise ValueError(f"too few eligible sessions for RQ4; ledger: {ledger}")
    positions = np.flatnonzero(keep)
    rows = panel.loc[keep]

    tone = pd.to_numeric(rows[f"s_{scorer}"], errors="coerce").astype(float)
    disp = pd.to_numeric(rows[f"d_{scorer}"], errors="coerce").astype(float)
    control = ledger["control"]

    design = pd.DataFrame(index=rows.index)
    for name, series in (("z_s", tone), ("z_abs_s", tone.abs()), ("z_d", disp)):
        sd = float(series.std(ddof=1))
        if not np.isfinite(sd) or sd == 0.0:
            raise ValueError(f"{name} has zero or undefined standard deviation")
        design[name] = (series - float(series.mean())) / sd
    design[control] = pd.to_numeric(rows[control], errors="coerce").astype(float)
    if outcome.startswith("log_volume"):
        design["abs_ret"] = pd.to_numeric(rows["ret"], errors="coerce").abs().astype(float)

    target = pd.to_numeric(rows[outcome], errors="coerce").astype(float)
    session_index = positions if convention == "session_indexed" else None
    fit = fit_hac(target, design, maxlags=maxlags, session_index=session_index,
                  convention=convention)
    fit.ledger = ledger
    fit.estimand = f"exploratory_rq4:{outcome}"
    return fit


def hac_wald(fit: HACFit, terms: Sequence[str]) -> dict:
    """Joint HAC Wald test that every named coefficient is zero.

    `W = (R b)' (R V R')^-1 (R b)`, chi-square with `rank(R)` degrees of
    freedom. This is the RQ4 primary test: it asks whether tone carries *any*
    information about the outcome, which is a different and prior question to
    which of the three terms carries it.
    """
    from scipy import stats as _stats

    missing = [t for t in terms if t not in fit.names]
    if missing:
        raise KeyError(f"fit has no term(s) {missing}")
    idx = [fit.names.index(t) for t in terms]
    b = fit.params.to_numpy()[idx]
    V = fit.cov.to_numpy()[np.ix_(idx, idx)]
    if np.linalg.matrix_rank(V) < len(idx):
        raise ValueError("the restricted covariance is singular; the joint test is undefined")
    stat = float(b @ np.linalg.solve(V, b))
    df = len(idx)
    return {"wald_chi2": stat, "df": df,
            "p": float(_stats.chi2.sf(stat, df)), "terms": list(terms)}


def rq4_family(panel: pd.DataFrame, scorers: Sequence[str] = ("finbert", "lm"),
               q: float = config.FDR_Q, maxlags: int = config.NW_MAXLAGS,
               convention: str = config.HAC_CONVENTION) -> pd.DataFrame:
    """The exploratory family: one joint Wald test per (scorer, outcome), BH-corrected.

    Returns the **Wald tests**, which are the family. Individual coefficients
    come from `rq4_coefficients` and are explicitly descriptive: the ordering is
    what stops a single coefficient becoming the headline after inspection
    (section 6, P21).
    """
    rows = []
    for scorer in scorers:
        for outcome in RQ4_OUTCOMES:
            fit = rq4_fit(panel, scorer, outcome, maxlags=maxlags, convention=convention)
            wald = hac_wald(fit, RQ4_TONE_TERMS)
            rows.append({
                "scorer": scorer, "outcome": outcome,
                "wald_chi2": wald["wald_chi2"], "df": wald["df"], "p": wald["p"],
                "nobs": int(fit.nobs),
                "n_lost_to_dispersion_floor": fit.ledger["n_lost_to_dispersion_floor"],
                "hac_convention": fit.convention, "hac_maxlags": int(fit.maxlags),
            })
    out = pd.DataFrame(rows)
    reject, adjusted = multipletests(out["p"], alpha=q, method="fdr_bh")[:2]
    out["bh_q"] = adjusted
    out["bh_reject"] = reject
    out["family"] = "exploratory_rq4"
    out["family_size"] = len(out)
    out["alpha"] = q
    out["test"] = "HAC Wald, joint null b1 = b2 = b3 = 0"
    return out


def rq4_coefficients(panel: pd.DataFrame, scorers: Sequence[str] = ("finbert", "lm"),
                     maxlags: int = config.NW_MAXLAGS,
                     convention: str = config.HAC_CONVENTION) -> pd.DataFrame:
    """Individual RQ4 coefficients, **descriptive only**.

    Carries no q-value and no reject flag by design. Section 6 makes the joint
    Wald test the primary; these are read afterwards, with pointwise intervals,
    and none is promoted to a headline. `b2` and `b3` are additionally not
    independent dimensions -- bounded scores tie the mean and dispersion
    mechanically, `d_t^2 <= [n_t/(n_t-1)] * (1 - S_t^2)`.
    """
    from scipy import stats as _stats

    z = float(_stats.norm.ppf(0.975))
    rows = []
    for scorer in scorers:
        for outcome in RQ4_OUTCOMES:
            fit = rq4_fit(panel, scorer, outcome, maxlags=maxlags, convention=convention)
            for term in fit.names:
                coef, se = float(fit.params[term]), float(fit.bse[term])
                rows.append({
                    "scorer": scorer, "outcome": outcome, "term": term,
                    "coef": coef, "nw_se": se,
                    "lo95": coef - z * se, "hi95": coef + z * se,
                    "nobs": int(fit.nobs),
                    "standardized": term in RQ4_TONE_TERMS,
                })
    out = pd.DataFrame(rows)
    out["interval_scope"] = "pointwise"
    out["role"] = "descriptive; the joint Wald test in rq4_family is the primary"
    return out
