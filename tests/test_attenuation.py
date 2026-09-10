"""R12: the attenuation derivation, checked against simulation (B21, P06, P32).

These test the *mathematics* in `docs/mathematical-appendix.md`, not project
data. Each case has a true parameter set by construction, so a failure means the
derivation or the code is wrong -- there is no empirical quantity here to be
uncertain about.

Sample sizes are smaller than `attenuation_study.py` uses, and tolerances are
set from the Monte-Carlo standard error rather than by tuning until green.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import attenuation_study as study

N = 60_000
SEED = 20260830


def rng():
    return np.random.default_rng(SEED)


# ------------------------------------------------------------ 1. classical


@pytest.mark.parametrize("var_u,expected_lambda", [(0.0, 1.0), (0.25, 0.8), (1.0, 0.5), (3.0, 0.25)])
def test_classical_attenuation_matches_the_reliability_ratio(var_u, expected_lambda):
    """plim(beta) = theta * Var(Z) / (Var(Z) + Var(u))."""
    g, theta, var_z = rng(), 0.5, 1.0
    z = g.normal(0, np.sqrt(var_z), N)
    u = g.normal(0, np.sqrt(var_u), N)
    r = theta * z + g.normal(0, 1.0, N)

    lam = var_z / (var_z + var_u)
    assert lam == pytest.approx(expected_lambda)
    assert study.ols_slope(z + u, r) == pytest.approx(theta * lam, abs=0.02)


def test_attenuation_never_flips_the_sign_or_exceeds_theta():
    """lambda is in (0, 1], so the bias is toward zero and one-directional."""
    g, theta, var_z = rng(), 0.5, 1.0
    z = g.normal(0, np.sqrt(var_z), N)
    r = theta * z + g.normal(0, 1.0, N)
    previous = np.inf
    for var_u in (0.0, 0.5, 2.0, 8.0, 32.0):
        got = study.ols_slope(z + g.normal(0, np.sqrt(var_u), N), r)
        assert 0 < got < theta + 0.02, "must stay between zero and theta"
        assert got < previous, "more noise must attenuate further"
        previous = got


# --------------------------------------------------------- 2. standardized


@pytest.mark.parametrize("var_u", [0.0, 0.25, 1.0])
def test_standardized_coefficient_is_attenuated_by_sqrt_lambda(var_u):
    """beta_z = theta * sd(Z) * sqrt(lambda) -- exponent 1/2, not 1."""
    g, theta, var_z = rng(), 0.5, 1.0
    z = g.normal(0, np.sqrt(var_z), N)
    u = g.normal(0, np.sqrt(var_u), N)
    r = theta * z + g.normal(0, 1.0, N)

    lam = var_z / (var_z + var_u)
    predicted = theta * np.sqrt(var_z) * np.sqrt(lam)
    assert study.standardized_slope(z + u, r) == pytest.approx(predicted, abs=0.02)


def test_sqrt_lambda_shrinks_less_than_lambda():
    """Standardizing makes the attenuation milder, so scorer gaps are smaller."""
    for lam in (0.05, 0.25, 0.5, 0.93):
        assert np.sqrt(lam) > lam


def test_standardization_removes_the_scorer_scale():
    """A shared [-1, 1] range is not a shared scale; z(S) makes it one.

    With no measurement error, doubling a score must leave the standardized
    coefficient unchanged -- the formal reason section 7(a) is specified on z(S).
    """
    g, theta = rng(), 0.5
    z = g.normal(0, 1.0, N)
    r = theta * z + g.normal(0, 1.0, N)
    single = study.standardized_slope(z, r)
    doubled = study.standardized_slope(2.0 * z, r)
    assert doubled == pytest.approx(single, rel=1e-9)


def test_the_raw_coefficient_does_not_survive_rescaling():
    """The property standardization exists to fix."""
    g = rng()
    z = g.normal(0, 1.0, N)
    r = 0.5 * z + g.normal(0, 1.0, N)
    assert study.ols_slope(2.0 * z, r) == pytest.approx(study.ols_slope(z, r) / 2, rel=1e-9)


# ---------------------------------------------------------- 3. aggregation


def test_daily_reliability_approaches_one_as_headline_count_grows():
    """Independent per-headline error averages down as Var(u)/n; the signal does not."""
    var_zbar, var_u = 0.01, 0.25
    lams = [var_zbar / (var_zbar + var_u / n) for n in (1, 25, 337, 5000)]
    assert lams == sorted(lams), "more headlines must mean higher reliability"
    assert lams[0] < 0.05
    assert lams[2] == pytest.approx(0.9309, abs=1e-3)
    assert lams[3] > 0.99


def test_aggregation_compresses_a_two_fold_noise_difference_to_a_few_percent():
    """The quantitative content of P06.

    A 2x per-headline measurement-noise gap becomes ~3.4% in the standardized
    daily coefficient at the census median of 337 headlines per session. That is
    a reason internal to the design for expecting a small paired contrast.
    """
    var_zbar, better, worse, n = 0.01, 0.25, 0.50, 337
    lam_b = var_zbar / (var_zbar + better / n)
    lam_w = var_zbar / (var_zbar + worse / n)

    assert lam_b / lam_w == pytest.approx(1.069, abs=0.005)
    assert np.sqrt(lam_b) / np.sqrt(lam_w) == pytest.approx(1.034, abs=0.005)

    # and the same thing simulated end to end
    g = rng()
    zbar = g.normal(0, np.sqrt(var_zbar), N)
    r = 0.5 * zbar + g.normal(0, 1.0, N)
    ratio = (study.standardized_slope(zbar + g.normal(0, np.sqrt(better / n), N), r)
             / study.standardized_slope(zbar + g.normal(0, np.sqrt(worse / n), N), r))
    assert ratio == pytest.approx(1.034, abs=0.03)


def test_the_per_headline_gap_is_large_before_aggregation():
    """The compression is what aggregation does, not an absence of difference."""
    var_zbar, better, worse = 0.01, 0.25, 0.50
    lam_b = var_zbar / (var_zbar + better)
    lam_w = var_zbar / (var_zbar + worse)
    assert lam_b / lam_w > 1.9      # ~1.96x at n = 1


# ------------------------------------------------------- 4. correlated error


def test_correlated_within_day_error_destroys_the_compression():
    """A5 is load-bearing: the divisor is n only when errors are independent."""
    n, var_u = 337, 0.25
    divisors = {}
    for rho in (0.0, 0.01, 0.1, 1.0):
        var_ubar = var_u * (1 + (n - 1) * rho) / n
        divisors[rho] = var_u / var_ubar

    assert divisors[0.0] == pytest.approx(n)
    assert divisors[0.1] == pytest.approx(9.7, abs=0.2)
    assert divisors[1.0] == pytest.approx(1.0)
    assert divisors[0.0] > divisors[0.01] > divisors[0.1] > divisors[1.0]


def test_perfectly_correlated_error_is_the_single_headline_case():
    """rho = 1 means averaging removes nothing at all."""
    n, var_u = 337, 0.25
    assert var_u * (1 + (n - 1) * 1.0) / n == pytest.approx(var_u)


# ------------------------------------------------------------ 5. bounded score


def test_bounding_a_score_makes_the_error_negatively_correlated_with_the_signal():
    """A2 fails by construction: near +1 the error cannot point outward."""
    g = rng()
    z = g.normal(0, 1.0, N)
    noise = g.normal(0, np.sqrt(0.5), N)
    u_bounded = np.clip(z + noise, -1.0, 1.0) - z

    assert np.cov(z, noise)[0, 1] == pytest.approx(0.0, abs=0.02)
    assert np.cov(z, u_bounded)[0, 1] < -0.2


def test_the_classical_formula_mispredicts_a_bounded_score():
    """And in the amplifying direction, so 'shrunk toward zero' is not safe."""
    g, theta, var_z, var_u = rng(), 0.5, 1.0, 0.5
    z = g.normal(0, np.sqrt(var_z), N)
    r = theta * z + g.normal(0, 1.0, N)
    s = np.clip(z + g.normal(0, np.sqrt(var_u), N), -1.0, 1.0)

    classical = theta * var_z / (var_z + var_u)
    observed = study.ols_slope(s, r)
    assert observed > classical + 0.1, "bounded case is amplified, not attenuated"
    assert observed > theta * 0.9, "the plim is not bounded above by theta*lambda"


def test_the_corrected_formula_recovers_the_bounded_case():
    """plim = theta*(Var(Z)+Cov(Z,u)) / (Var(Z)+2Cov(Z,u)+Var(u))."""
    g, theta, var_z = rng(), 0.5, 1.0
    z = g.normal(0, np.sqrt(var_z), N)
    r = theta * z + g.normal(0, 1.0, N)
    s = np.clip(z + g.normal(0, np.sqrt(0.5), N), -1.0, 1.0)
    u = s - z

    cov_zu = float(np.cov(z, u)[0, 1])
    corrected = theta * (var_z + cov_zu) / (var_z + 2 * cov_zu + float(np.var(u)))
    assert study.ols_slope(s, r) == pytest.approx(corrected, abs=0.03)


# ------------------------------------------------------------- 6. theta = 0


@pytest.mark.parametrize("var_u", [0.0, 0.25, 4.0])
def test_no_amount_of_accuracy_creates_a_signal_that_is_not_there(var_u):
    """theta = 0 absorbs every reliability ratio."""
    g = rng()
    z = g.normal(0, 1.0, N)
    r = 0.0 * z + g.normal(0, 1.0, N)
    assert study.ols_slope(z + g.normal(0, np.sqrt(var_u), N), r) == pytest.approx(0.0, abs=0.02)


# ------------------------------------------------------------- the guardrail


def test_the_study_reads_no_project_data():
    """P32/P06: nothing here may become an estimate about SPY or these scorers."""
    source = Path(study.__file__).read_text(encoding="utf-8")
    for forbidden in ("read_parquet", "HEADLINES", "SCORES_PARQUET", "PANEL",
                      "MARKET", "import config", "from src"):
        assert forbidden not in source, f"the simulation must not touch {forbidden}"
