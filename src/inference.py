"""Act 2: HAC regressions, the FDR-controlled lag family, the placebo, effect sizes.

Every regression in the project is fitted here, with Newey-West standard errors
(D10) every time. Notebooks call these functions and display the result; they do
not fit models themselves.
"""

from __future__ import annotations

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


LAG_COLUMNS = ("ret_lag1", "parkinson_lag1", "log_turnover_lag1")


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


def contemporaneous(panel: pd.DataFrame, scorer: str, maxlags: int = config.NW_MAXLAGS):
    """Section 6.1: ret_t on S_t plus lagged return, range variance and volume.

    Reported as association, not causation, in that language. Its absence is
    not evidence that the pipeline is broken (P23): a weak contemporaneous
    association can equally reflect aggregation, the chosen universe, or
    measurement noise, and is investigated as such rather than by adjusting the
    pipeline until it appears.

    Controls are read from the panel, never shifted here -- see `_require_lags`.
    """
    _require_lags(panel)
    X = pd.DataFrame(
        {
            f"s_{scorer}": panel[f"s_{scorer}"],
            "ret_lag1": panel["ret_lag1"],
            "parkinson_lag1": panel["parkinson_lag1"],
            "log_turnover_lag1": panel["log_turnover_lag1"],
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
            "parkinson": df["parkinson"],
            "log_turnover": df["log_turnover"],
        }
    )
    return nw_ols(df[f"ret_lead{horizon}"], X, maxlags=maxlags)


def lag_family(
    panel: pd.DataFrame,
    scorer: str,
    horizons: Sequence[int] = config.HORIZONS,
    maxlags: int = config.NW_MAXLAGS,
    q: float = config.FDR_Q,
) -> pd.DataFrame:
    """Section 6.2 across the declared horizon family, with BH q-values.

    BH is applied within each scorer's family (D11). The five horizons are the
    family declared in advance, and they are correlated -- which is exactly why
    Benjamini-Hochberg and not Bonferroni.
    """
    rows = []
    key = f"s_{scorer}"
    for h in horizons:
        res = predictive(panel, scorer, h, maxlags=maxlags)
        rows.append(
            {
                "scorer": scorer,
                "horizon": h,
                "coef": float(res.params[key]),
                "nw_se": float(res.bse[key]),
                "t": float(res.tvalues[key]),
                "p": float(res.pvalues[key]),
                "nobs": int(res.nobs),
            }
        )
    out = pd.DataFrame(rows)
    reject, qvals = multipletests(out["p"], alpha=q, method="fdr_bh")[:2]
    out["bh_q"] = qvals
    out["bh_reject"] = reject
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


def effect_size_bps(coef: float, se: float, sigma_s: float) -> tuple[float, float, float]:
    """D15. Basis points of return per 1 sigma move in S_t, with a 95% NW CI.

    The interval is transformed identically to the point estimate, so a null
    reads as a ruled-out interval -- effects larger than X bps per 1 sigma are
    excluded at 95% -- rather than as a shrug.
    """
    scale = sigma_s * config.BPS_PER_UNIT
    point = coef * scale
    half = 1.96 * se * scale
    return point, point - half, point + half


def effect_size_table(family: pd.DataFrame, sigma_s: float) -> pd.DataFrame:
    """Table 4: every headline coefficient in bps per 1 sigma, vs. the cost bar."""
    out = family.copy()
    vals = [effect_size_bps(c, s, sigma_s) for c, s in zip(out["coef"], out["nw_se"])]
    out[["bps_per_sd", "bps_lo95", "bps_hi95"]] = pd.DataFrame(vals, index=out.index)
    lo, hi = out["bps_lo95"], out["bps_hi95"]
    # Economic significance: does the whole interval clear one-way costs?
    out["clears_costs"] = (lo > config.TRANSACTION_COST_BPS) | (hi < -config.TRANSACTION_COST_BPS)
    out["ruled_out_above_bps"] = np.maximum(lo.abs(), hi.abs())
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
            "parkinson": panel["parkinson"],
            "log_turnover": panel["log_turnover"],
        }
    )
    return nw_ols(panel[f"ret_lead{horizon}"], X, maxlags=maxlags)


def volatility_spec(panel: pd.DataFrame, scorer: str, maxlags: int = config.NW_MAXLAGS):
    """Section 6.4, volatility: next-day Parkinson on level, intensity and dispersion."""
    X = pd.DataFrame(
        {
            f"s_{scorer}": panel[f"s_{scorer}"],
            f"abs_s_{scorer}": panel[f"s_{scorer}"].abs(),
            f"d_{scorer}": panel[f"d_{scorer}"],
            "parkinson": panel["parkinson"],
        }
    )
    return nw_ols(panel["parkinson_lead1"], X, maxlags=maxlags)


def volume_spec(panel: pd.DataFrame, scorer: str, maxlags: int = config.NW_MAXLAGS):
    """Section 6.4, volume: next-day detrended turnover, same regressors plus |ret_t|.

    The dispersion coefficient is the one to read first: disagreement predicting
    volume is the most plausible positive finding in the project, and it is a
    documented one (Tetlock 2007).
    """
    X = pd.DataFrame(
        {
            f"s_{scorer}": panel[f"s_{scorer}"],
            f"abs_s_{scorer}": panel[f"s_{scorer}"].abs(),
            f"d_{scorer}": panel[f"d_{scorer}"],
            "log_turnover_detrended": panel["log_turnover_detrended"],
            "abs_ret": panel["ret"].abs(),
        }
    )
    return nw_ols(panel["log_turnover_detrended_lead1"], X, maxlags=maxlags)


def analysis_sample(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """D9. Drop zero-news trading days and report how many were dropped."""
    keep = panel["n_headlines"] > 0
    stats = {
        "n_sessions": int(len(panel)),
        "n_zero_news_days": int((~keep).sum()),
        "n_analysis_days": int(keep.sum()),
    }
    return panel[keep].reset_index(drop=True), stats
