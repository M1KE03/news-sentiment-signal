"""RQ4 as the protocol specifies it: a joint test first (section 6, P21, A08).

The removed `volatility_spec` / `volume_spec` fitted unstandardized `s_` terms,
printed a per-coefficient t and p with no joint test and no correction, and ran
on a different sample from the primary. That is the arrangement in which "the
dispersion coefficient" becomes a headline after someone has looked at it.

What replaces it: the family is four **HAC Wald tests** of `b1 = b2 = b3 = 0`,
BH-corrected; individual coefficients are descriptive and carry no q-value.

Synthetic panels throughout. No claim is made about any coefficient's value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import align, inference as inf


def _market(cal, seed=4):
    n = len(cal)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": cal,
        "close_adj": 100 + np.arange(n, dtype=float),
        "ret": rng.normal(0, 0.01, n),
        "rv_parkinson": rng.uniform(1e-5, 5e-4, n),
        "volume": rng.uniform(5e7, 2e8, n),
        "log_volume": 18.0 + np.linspace(0, 1.5, n) + rng.normal(0, 0.05, n),
        "vix_close": rng.uniform(10, 30, n),
    })


def _daily(cal, seed=5, thin_rows=()):
    n = len(cal)
    rng = np.random.default_rng(seed)
    daily = pd.DataFrame({"date": cal, "n_headlines": rng.integers(8, 50, n)})
    for s in config.SCORERS:
        daily[f"s_{s}"] = rng.normal(0, 0.1, n)
        daily[f"d_{s}"] = rng.uniform(0.1, 0.4, n)
    for row in thin_rows:                       # below the dispersion floor
        daily.loc[row, "n_headlines"] = 2
        for s in config.SCORERS:
            daily.loc[row, f"d_{s}"] = np.nan
    return daily


def _panel(periods=220, **kw):
    cal = pd.bdate_range("2015-01-05", periods=periods)
    return align.build_panel(_daily(cal, **kw), _market(cal))


# ------------------------------------------------- the superseded path is gone


def test_the_old_per_coefficient_specs_no_longer_exist():
    assert not hasattr(inf, "volatility_spec")
    assert not hasattr(inf, "volume_spec")


# --------------------------------------------------------------- the design


@pytest.mark.parametrize("outcome", inf.RQ4_OUTCOMES)
def test_all_three_tone_terms_are_standardized_on_the_equations_own_sample(outcome):
    fit = inf.rq4_fit(_panel(), "finbert", outcome)
    for term in inf.RQ4_TONE_TERMS:
        col = fit.ols.model.exog[:, fit.names.index(term)]
        assert abs(col.mean()) < 1e-10
        assert abs(col.std(ddof=1) - 1.0) < 1e-10


def test_each_outcome_carries_its_own_lagged_control():
    rv = inf.rq4_fit(_panel(), "finbert", "rv_parkinson_lead1")
    lv = inf.rq4_fit(_panel(), "finbert", "log_volume_detrended_lead1")
    assert "rv_parkinson" in rv.names and "abs_ret" not in rv.names
    assert "log_volume_detrended" in lv.names and "abs_ret" in lv.names


def test_the_volume_equation_uses_the_detrended_series():
    fit = inf.rq4_fit(_panel(), "finbert", "log_volume_detrended_lead1")
    assert "log_volume" not in fit.names, "the raw series is not an RQ4 control"


def test_an_unknown_outcome_is_refused():
    with pytest.raises(ValueError, match="outcome must be one of"):
        inf.rq4_fit(_panel(), "finbert", "ret_lead1")


# ------------------------------------------------- the dispersion-floor sample


def test_sessions_below_the_dispersion_floor_are_excluded_and_counted():
    """d_t is undefined below n_t >= 5 by construction; section 6 requires the
    reduction to be reported, not absorbed."""
    panel = _panel(thin_rows=(120, 121, 122))
    _, ledger = inf.rq4_design(panel, "finbert", "rv_parkinson_lead1")
    assert ledger["n_lost_to_dispersion_floor"] >= 3
    assert ledger["excluded_first_reason"]["dispersion_undefined"] >= 3
    assert ledger["dispersion_floor"] == config.MIN_HEADLINES_FOR_DISPERSION


def test_the_rq4_ledger_reconciles():
    panel = _panel(thin_rows=(120,))
    for outcome in inf.RQ4_OUTCOMES:
        _, ledger = inf.rq4_design(panel, "finbert", outcome)
        assert sum(ledger["excluded_first_reason"].values()) == ledger["n_excluded"]
        assert ledger["n_eligible"] + ledger["n_excluded"] == ledger["n_panel_rows"]


def test_the_rq4_sample_is_smaller_than_the_primary_and_says_so():
    panel = _panel(thin_rows=(120, 121))
    primary = inf.primary(panel)
    fit = inf.rq4_fit(panel, "finbert", "log_volume_detrended_lead1")
    assert fit.nobs < primary.n
    assert fit.ledger["n_lost_to_dispersion_floor"] >= 2


def test_a_prefiltered_panel_is_refused():
    panel = _panel()
    with pytest.raises(ValueError, match="not the complete exchange calendar"):
        inf.rq4_fit(panel.drop(index=range(100, 110)).reset_index(drop=True),
                    "finbert", "rv_parkinson_lead1")


# ----------------------------------------------------------- the joint test


def test_the_wald_statistic_is_the_quadratic_form_computed_by_hand():
    fit = inf.rq4_fit(_panel(), "finbert", "rv_parkinson_lead1")
    got = inf.hac_wald(fit, inf.RQ4_TONE_TERMS)

    idx = [fit.names.index(t) for t in inf.RQ4_TONE_TERMS]
    b = fit.params.to_numpy()[idx]
    V = fit.cov.to_numpy()[np.ix_(idx, idx)]
    assert got["wald_chi2"] == pytest.approx(float(b @ np.linalg.inv(V) @ b))
    assert got["df"] == 3


def test_a_single_term_wald_is_the_squared_t_statistic():
    """The chi-square with one degree of freedom must agree with the z test."""
    fit = inf.rq4_fit(_panel(), "finbert", "rv_parkinson_lead1")
    got = inf.hac_wald(fit, ("z_d",))
    t = float(fit.params["z_d"] / fit.bse["z_d"])
    assert got["wald_chi2"] == pytest.approx(t**2)
    assert got["df"] == 1


def test_the_joint_test_uses_the_hac_covariance_not_the_ols_one():
    fit = inf.rq4_fit(_panel(), "finbert", "rv_parkinson_lead1")
    hac = inf.hac_wald(fit, inf.RQ4_TONE_TERMS)["wald_chi2"]
    idx = [fit.names.index(t) for t in inf.RQ4_TONE_TERMS]
    b = fit.params.to_numpy()[idx]
    V_ols = np.asarray(fit.ols.cov_params())[np.ix_(idx, idx)]
    assert hac != pytest.approx(float(b @ np.linalg.inv(V_ols) @ b), rel=1e-6)


def test_an_unknown_term_is_refused():
    fit = inf.rq4_fit(_panel(), "finbert", "rv_parkinson_lead1")
    with pytest.raises(KeyError, match="no term"):
        inf.hac_wald(fit, ("z_s", "not_a_term"))


# --------------------------------------------------------------- the family


def test_the_family_is_four_wald_tests_with_bh_applied_across_them():
    family = inf.rq4_family(_panel())
    assert len(family) == 4                      # 2 scorers x 2 outcomes
    assert set(family["outcome"]) == set(inf.RQ4_OUTCOMES)
    assert (family["family_size"] == 4).all()
    assert (family["test"].str.contains("joint null")).all()
    assert (family["bh_q"] >= family["p"] - 1e-12).all(), "BH q must not fall below raw p"


def test_the_family_reports_the_dispersion_floor_loss_per_equation():
    family = inf.rq4_family(_panel(thin_rows=(120, 121)))
    assert (family["n_lost_to_dispersion_floor"] >= 2).all()


# -------------------------------------- coefficients are descriptive, and only that


def test_individual_coefficients_carry_no_q_value_and_no_reject_flag():
    """The ordering is what stops one coefficient becoming the headline."""
    coefs = inf.rq4_coefficients(_panel())
    for forbidden in ("bh_q", "by_q", "bh_reject", "claim_reject", "p"):
        assert forbidden not in coefs.columns, f"{forbidden} must not appear here"
    assert (coefs["role"].str.contains("descriptive")).all()
    assert (coefs["interval_scope"] == "pointwise").all()


def test_the_coefficient_table_marks_which_terms_are_standardized():
    coefs = inf.rq4_coefficients(_panel())
    tone = coefs[coefs["term"].isin(inf.RQ4_TONE_TERMS)]
    other = coefs[~coefs["term"].isin(inf.RQ4_TONE_TERMS)]
    assert tone["standardized"].all()
    assert not other["standardized"].any()


def test_the_coefficient_intervals_are_the_hac_ones():
    coefs = inf.rq4_coefficients(_panel()).set_index(["scorer", "outcome", "term"])
    row = coefs.loc[("finbert", "rv_parkinson_lead1", "z_d")]
    assert row["lo95"] == pytest.approx(row["coef"] - 1.959963984540054 * row["nw_se"], rel=1e-6)
    assert row["hi95"] == pytest.approx(row["coef"] + 1.959963984540054 * row["nw_se"], rel=1e-6)


def test_rescaling_a_score_leaves_the_standardized_rq4_result_unchanged():
    panel = _panel()
    doubled = panel.copy()
    doubled["s_finbert"] = doubled["s_finbert"] * 2.0
    doubled["d_finbert"] = doubled["d_finbert"] * 2.0

    base = inf.hac_wald(inf.rq4_fit(panel, "finbert", "rv_parkinson_lead1"), inf.RQ4_TONE_TERMS)
    after = inf.hac_wald(inf.rq4_fit(doubled, "finbert", "rv_parkinson_lead1"), inf.RQ4_TONE_TERMS)
    assert after["wald_chi2"] == pytest.approx(base["wald_chi2"], rel=1e-6)
