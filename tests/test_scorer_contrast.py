"""R08b: comparing scorers with an interval for their difference (audit A08).

`attenuation_comparison` fitted each scorer separately and tabulated `abs_coef`
alongside two separate t-statistics. That cannot support the claim it invites:
"FinBERT's coefficient is larger" is a statement about `delta = beta_A - beta_B`,
and two marginal significance verdicts are not a test of it (P07). Protocol
Section 11 forbids the claim without an interval for the paired difference.

The interval has to carry the **cross-equation covariance**. The two daily tone
series are strongly correlated, so `cov(beta_A, beta_B)` is large and positive
and `var(delta) = var_A + var_B - 2 cov` is much smaller than the sum of the two
marginal variances. Treating the fits as independent would overstate the
uncertainty about their difference, which is a different error from the usual
direction and just as wrong.

Everything here is synthetic; no claim is made about any coefficient's value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import align, inference as inf


def _market(cal, seed=4):
    n = len(cal)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "date": cal,
            "close_adj": 100 + np.arange(n, dtype=float),
            "ret": rng.normal(0, 0.01, n),
            "rv_parkinson": rng.uniform(1e-5, 5e-4, n),
            "volume": rng.uniform(5e7, 2e8, n),
            "log_volume": 18.0 + np.linspace(0, 1.5, n) + rng.normal(0, 0.05, n),
            "vix_close": rng.uniform(10, 30, n),
        }
    )


def _daily(cal, seed=5, corr=0.8):
    """Two correlated tone series, which is the situation the contrast is for."""
    n = len(cal)
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 0.1, n)
    daily = pd.DataFrame({"date": cal, "n_headlines": rng.integers(5, 50, n)})
    for s in config.SCORERS:
        own = rng.normal(0, 0.1, n)
        daily[f"s_{s}"] = corr * shared + np.sqrt(1 - corr**2) * own
        daily[f"d_{s}"] = rng.uniform(0.1, 0.4, n)
    return daily


def _panel(periods=220, **kw):
    cal = pd.bdate_range("2015-01-05", periods=periods)
    return align.build_panel(_daily(cal, **kw), _market(cal))


# ------------------------------------------------------- the common sample


def test_the_contrast_uses_only_sessions_eligible_for_both_scorers():
    panel = _panel()
    panel.loc[120, "s_lm"] = np.nan          # eligible for finbert, not for lm

    contrast = inf.paired_scorer_contrast(panel)
    assert not contrast.mask[120]
    assert contrast.ledger["n_lost_to_common_sample"]["finbert"] >= 1
    assert contrast.ledger["n_lost_to_common_sample"]["lm"] == 0
    assert contrast.n == contrast.ledger["n_common"]


def test_both_equations_are_fitted_on_exactly_the_same_rows():
    """A difference across two different samples is not a paired difference."""
    contrast = inf.paired_scorer_contrast(_panel())
    a, b = contrast.scorers
    assert len(contrast.designs[a]) == len(contrast.designs[b]) == contrast.n
    assert contrast.designs[a].index.equals(contrast.designs[b].index)
    assert contrast.designs[a].index.equals(contrast.target.index)


def test_each_scorers_ledger_is_kept_separately():
    _, ledger = inf.common_eligibility(_panel(), ("finbert", "lm"))
    for sc in ("finbert", "lm"):
        per = ledger["per_scorer"][sc]
        assert sum(per["excluded_first_reason"].values()) == per["n_excluded"]


def test_fewer_than_two_scorers_is_refused():
    with pytest.raises(ValueError, match="at least two distinct scorers"):
        inf.common_eligibility(_panel(), ("finbert",))


def test_a_repeated_scorer_is_refused():
    with pytest.raises(ValueError, match="at least two distinct scorers"):
        inf.common_eligibility(_panel(), ("finbert", "finbert"))


# ------------------------------------------ the stacked covariance is correct


def test_the_diagonal_blocks_reproduce_each_single_equation_fit():
    """Stacking must not change either equation -- only add the cross terms."""
    panel = _panel()
    contrast = inf.paired_scorer_contrast(panel)

    for sc in contrast.scorers:
        alone = inf.fit_hac(
            contrast.target,
            contrast.designs[sc].drop(columns="const"),
            maxlags=contrast.maxlags,
            session_index=contrast.positions,
            convention=contrast.convention,
        )
        for term in alone.names:
            key = f"{sc}::{term}"
            assert contrast.params[key] == pytest.approx(float(alone.params[term]))
            assert float(contrast.cov.loc[key, key]) == pytest.approx(
                float(alone.cov.loc[term, term]), rel=1e-10
            )


def test_the_stacked_covariance_matches_an_independent_pairwise_calculation():
    """The definition, written out as a double loop over every pair."""
    panel = _panel(periods=140)
    contrast = inf.paired_scorer_contrast(panel)
    a, b = contrast.scorers
    L, sessions = contrast.maxlags, contrast.positions

    blocks, breads = [], []
    for sc in (a, b):
        X = contrast.designs[sc].to_numpy(dtype=float)
        u = contrast.fits[sc]["resid"]
        blocks.append(X * u[:, None])
        breads.append(np.linalg.inv(X.T @ X))
    g = np.hstack(blocks)

    n, k = g.shape
    S = np.zeros((k, k))
    for i in range(n):
        for j in range(n):
            d = abs(int(sessions[i]) - int(sessions[j]))
            if d > L:
                continue
            S += (1.0 - d / (L + 1.0)) * np.outer(g[i], g[j])
    from scipy.linalg import block_diag

    bread = block_diag(*breads)
    np.testing.assert_allclose(contrast.cov.to_numpy(), bread @ S @ bread,
                               rtol=1e-10, atol=1e-20)


def test_the_cross_equation_covariance_is_actually_used():
    """If it were dropped, var(delta) would be the sum of the two variances."""
    contrast = inf.paired_scorer_contrast(_panel())
    a, b = (f"{sc}::z_s_{sc}" for sc in contrast.scorers)
    var_a, var_b = float(contrast.cov.loc[a, a]), float(contrast.cov.loc[b, b])
    cov_ab = float(contrast.cov.loc[a, b])

    assert cov_ab != pytest.approx(0.0, abs=1e-12)
    assert contrast.se_delta**2 == pytest.approx(var_a + var_b - 2 * cov_ab)
    # correlated scorers => the paired SE is smaller than the independent one
    assert contrast.se_delta < np.sqrt(var_a + var_b)


# ------------------------------------------------ the acceptance properties


def test_identical_scorers_give_exactly_zero_contrast_and_zero_uncertainty():
    panel = _panel()
    panel["s_lm"] = panel["s_finbert"]

    contrast = inf.paired_scorer_contrast(panel)
    assert contrast.delta == pytest.approx(0.0, abs=1e-12)
    assert contrast.se_delta == pytest.approx(0.0, abs=1e-12)
    assert contrast.degenerate

    w = contrast.wald()
    assert w["degenerate"] is True
    assert np.isnan(w["z"]) and np.isnan(w["p"])
    assert w["bps"] == (0.0, 0.0, 0.0)


def test_rescaling_one_score_leaves_the_standardized_comparison_unchanged():
    """Section 7's reason for specifying (a) on z(S): 2S carries the same information."""
    panel = _panel()
    doubled = panel.copy()
    doubled["s_lm"] = doubled["s_lm"] * 2.0

    base = inf.paired_scorer_contrast(panel)
    after = inf.paired_scorer_contrast(doubled)

    assert after.delta == pytest.approx(base.delta)
    assert after.se_delta == pytest.approx(base.se_delta)
    np.testing.assert_allclose(after.wald()["bps"], base.wald()["bps"])
    assert after.tone_sd["lm"] == pytest.approx(2 * base.tone_sd["lm"])


def test_shifting_one_score_by_a_constant_also_leaves_it_unchanged():
    panel = _panel()
    shifted = panel.copy()
    shifted["s_finbert"] = shifted["s_finbert"] + 0.4

    assert inf.paired_scorer_contrast(shifted).delta == pytest.approx(
        inf.paired_scorer_contrast(panel).delta
    )


def test_swapping_the_scorers_flips_the_sign_and_keeps_the_standard_error():
    panel = _panel()
    ab = inf.paired_scorer_contrast(panel, scorers=("finbert", "lm"))
    ba = inf.paired_scorer_contrast(panel, scorers=("lm", "finbert"))
    assert ba.delta == pytest.approx(-ab.delta)
    assert ba.se_delta == pytest.approx(ab.se_delta)


def test_the_wald_statistic_is_the_squared_z_of_the_same_difference():
    contrast = inf.paired_scorer_contrast(_panel())
    w = contrast.wald()
    assert w["z"] == pytest.approx(contrast.delta / contrast.se_delta)
    assert w["wald_chi2"] == pytest.approx(w["z"] ** 2)
    point, lo, hi = w["bps"]
    assert point == pytest.approx(contrast.delta * config.BPS_PER_UNIT)
    assert lo < point < hi or contrast.delta == 0


# ------------------------------------------ collinearity is a diagnostic only


def test_correlation_and_vifs_are_reported_and_nothing_is_declared_inconclusive():
    """Section 7: a high correlation widens the interval; it triggers no rule."""
    strong = inf.paired_scorer_contrast(_panel(corr=0.99))
    weak = inf.paired_scorer_contrast(_panel(corr=0.1))

    assert strong.diagnostics["corr_tone"] > 0.9
    assert set(strong.diagnostics["vif_tone"]) == set(strong.scorers)
    assert all(v >= 1.0 for v in strong.diagnostics["vif_tone"].values())

    # both produce a full result; neither is short-circuited
    for c in (strong, weak):
        assert np.isfinite(c.wald()["p"])
    table = strong.table()
    assert "inconclusive" not in " ".join(map(str, table.to_numpy().ravel())).lower()


def test_the_table_reports_both_marginals_and_the_contrast_on_one_sample():
    contrast = inf.paired_scorer_contrast(_panel())
    table = contrast.table()
    assert list(table["quantity"]) == ["finbert", "lm", "delta(finbert - lm)"]
    assert table["n"].nunique() == 1
    assert (table["interval_scope"] == "pointwise").all()
    assert (table["coefficient_scale"] == "standardized_tone").all()
    assert table.attrs["wald"]["delta"] == pytest.approx(contrast.delta)


def test_no_output_frames_the_comparison_as_profitability():
    """Section 11 forbids strategy-profitability claims from a coefficient."""
    table = inf.paired_scorer_contrast(_panel()).table()
    forbidden = {"clears_costs", "profit", "profitable", "strategy_return"}
    assert not forbidden & set(table.columns)


# ------------------------------------- 7(b) is a different estimand, labelled


def test_incremental_contribution_puts_both_standardized_scores_in_one_fit():
    fit = inf.incremental_contribution(_panel())
    assert "z_s_finbert" in fit.names and "z_s_lm" in fit.names
    assert "log_volume_detrended" in fit.names
    assert "log_volume" not in fit.names          # the raw series is not a control
    assert fit.estimand == "incremental_contribution_given_the_other_scorer"


def test_incremental_contribution_is_not_the_paired_contrast():
    """Different estimands must not be quietly interchangeable."""
    panel = _panel()
    joint = inf.incremental_contribution(panel)
    contrast = inf.paired_scorer_contrast(panel)
    joint_delta = float(joint.params["z_s_finbert"]) - float(joint.params["z_s_lm"])
    assert joint_delta != pytest.approx(contrast.delta, rel=1e-6)


def test_incremental_contribution_uses_the_common_sample_too():
    panel = _panel()
    panel.loc[130, "s_lm"] = np.nan
    joint = inf.incremental_contribution(panel)
    assert joint.nobs == inf.paired_scorer_contrast(panel).n


def test_the_superseded_comparison_functions_no_longer_exist():
    """Deleted, not deprecated. `abs_coef` beside two separate t-statistics is
    an invitation to read a comparison the data has not been asked for."""
    assert not hasattr(inf, "attenuation_comparison")
    assert not hasattr(inf, "horse_race")


def test_no_correlation_threshold_declares_the_comparison_inconclusive():
    """The removed version short-circuited above corr > 0.9. Section 7 says the
    interval already carries that information, quantitatively."""
    near_identical = _panel(corr=0.999)
    contrast = inf.paired_scorer_contrast(near_identical)
    assert contrast.diagnostics["corr_tone"] > 0.9
    result = contrast.wald()
    assert not result["degenerate"]
    assert np.isfinite(result["p"])
    assert np.isfinite(result["se"]) and result["se"] > 0
