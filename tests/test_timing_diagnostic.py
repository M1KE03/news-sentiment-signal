"""R08c: the circular-shift timing diagnostic (audit A08 / B19).

What was removed and why. `permutation_pvalue` drew blocks **with replacement**,
so some observations appeared twice in a draw and others not at all. That is not
a permutation of the series, and the `+1` correction applied to its p-value does
not repair a null distribution built the wrong way. Its output was named
`p_permutation`, returned as a p-value, and annotated on Figure 2 as one --
which protocol Section 11 forbids, because shifting tone also destroys its
relationship with the controls, so the spread is not the null distribution of
the *conditional* coefficient.

What replaces it is a full circular shift: a bijection on the retained rows,
every observation used exactly once, the tone series' autocorrelation preserved
exactly, and the result reported as a percentile rank that is never called a
p-value.

Synthetic fixtures throughout.
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


def _daily(cal, seed=5):
    n = len(cal)
    rng = np.random.default_rng(seed)
    daily = pd.DataFrame({"date": cal, "n_headlines": rng.integers(5, 50, n)})
    for s in config.SCORERS:
        daily[f"s_{s}"] = rng.normal(0, 0.1, n)
        daily[f"d_{s}"] = rng.uniform(0.1, 0.4, n)
    return daily


def _panel(periods=120, **kw):
    cal = pd.bdate_range("2015-01-05", periods=periods)
    return align.build_panel(_daily(cal, **kw), _market(cal))


# ------------------------------------------------- the superseded path is gone


def test_the_block_resampling_placebo_no_longer_exists():
    """Deleted rather than deprecated: a function returning `p_permutation` is
    an invitation to quote it."""
    assert not hasattr(inf, "permutation_pvalue")
    assert not hasattr(inf, "_circular_block_permute")


def test_its_configuration_went_with_it():
    assert not hasattr(config, "PERMUTATION_DRAWS")
    assert not hasattr(config, "PERMUTATION_BLOCK")


def test_the_output_carries_no_p_value_and_says_so():
    d = inf.timing_diagnostic(_panel())
    assert d["is_p_value"] is False          # the disclosure flag itself
    reported = set(d) - {"is_p_value"}
    assert not {"p", "pvalue", "p_value", "p_permutation"} & reported
    assert not any("p_value" in k for k in reported)
    assert "not a p-value" in d["reporting_note"].lower()
    assert "percentile" in d


# ------------------------------------------------------ the shift is a shift


def test_every_shift_uses_every_observation_exactly_once():
    """The property block resampling did not have."""
    n = 9
    z = np.arange(n, dtype=float)
    for k in range(n):
        shifted = np.roll(z, -k)
        assert sorted(shifted) == sorted(z)
        assert len(set(shifted)) == n


def test_the_number_of_shifts_is_n_minus_one_and_the_denominator_is_n():
    d = inf.timing_diagnostic(_panel())
    assert d["n_shifts"] == d["n"] - 1
    # midrank over the n-1 shifts plus the observed value
    expected = (d["n_below"] + 0.5 * d["n_ties"]) / d["n"]
    assert d["percentile"] == pytest.approx(expected)


def test_the_shift_domain_is_the_retained_rows_in_session_order():
    """M6: not the full calendar. n must be identical for every k."""
    panel = _panel()
    panel.loc[70, "n_headlines"] = 0
    d = inf.timing_diagnostic(panel)
    fit = inf.primary(panel)
    assert d["n"] == fit.n
    assert "after eligibility" in d["shift_domain"]


def test_the_observed_value_is_the_primary_specifications_coefficient():
    panel = _panel()
    d = inf.timing_diagnostic(panel)
    assert d["observed_coef"] == pytest.approx(inf.primary(panel).beta)
    assert d["observed_bps"] == pytest.approx(d["observed_coef"] * config.BPS_PER_UNIT)


def test_the_partialled_out_shift_coefficients_equal_direct_refits():
    """Frisch-Waugh is used for speed; check the algebra rather than trust it."""
    import statsmodels.api as sm

    panel = _panel(periods=90)
    fit = inf.primary(panel)
    z = fit.design[fit.tone_term].to_numpy(dtype=float)
    y = fit.target.to_numpy(dtype=float)
    controls = fit.design[list(inf.PRIMARY_CONTROLS)].to_numpy(dtype=float)

    fast = inf._shift_coefficients(z, y, controls)
    for k in (0, 1, 5, len(z) // 2, len(z) - 1):
        X = sm.add_constant(np.column_stack([np.roll(z, -k), controls]))
        direct = np.linalg.lstsq(X, y, rcond=None)[0][1]
        assert fast[k] == pytest.approx(direct, rel=1e-9, abs=1e-14)


def test_a_full_rotation_returns_the_observed_coefficient():
    """k = n is the identity, so the sweep is closed."""
    panel = _panel(periods=80)
    fit = inf.primary(panel)
    z = fit.design[fit.tone_term].to_numpy(dtype=float)
    y = fit.target.to_numpy(dtype=float)
    controls = fit.design[list(inf.PRIMARY_CONTROLS)].to_numpy(dtype=float)
    coefs = inf._shift_coefficients(z, y, controls)
    assert np.roll(z, -len(z)) == pytest.approx(z)
    assert coefs[0] == pytest.approx(fit.beta)


# --------------------------------------------------------- rank and ties


def test_a_dominant_observed_coefficient_ranks_near_the_top():
    """A controlled case with a known answer: make tone genuinely predictive."""
    panel = _panel(periods=160)
    fit = inf.primary(panel)
    rows = np.flatnonzero(fit.mask)
    # inject a strong alignment between tone at t and the return at t+1
    panel = panel.copy()
    for pos in rows:
        panel.loc[pos + 1, "ret"] = 5.0 * panel.loc[pos, "s_finbert"]
    panel["ret_lead1"] = panel["ret"].shift(-1)
    for h in config.HORIZONS:
        panel[f"ret_lead{h}"] = panel["ret"].shift(-h)
    panel["ret_lag1"] = panel["ret"].shift(1)

    d = inf.timing_diagnostic(panel)
    assert d["percentile"] > 0.97, (
        "a coefficient built to be the largest must rank near the top of its "
        "own shift distribution, or the ranking is wired up wrong"
    )


def test_ties_are_counted_at_the_reporting_precision_and_reported():
    d = inf.timing_diagnostic(_panel())
    assert d["n_ties"] >= 0
    assert d["tie_precision_bps"] == pytest.approx(0.1)
    assert d["n_below"] + d["n_ties"] <= d["n_shifts"]


def test_a_coarser_tie_precision_can_only_create_ties_not_destroy_them():
    panel = _panel()
    fine = inf.timing_diagnostic(panel, round_bps=0.1)
    coarse = inf.timing_diagnostic(panel, round_bps=50.0)
    assert coarse["n_ties"] >= fine["n_ties"]


# ------------------------------------------------- gaps are disclosed, not hidden


def test_a_gapless_sample_reports_no_gaps():
    d = inf.timing_diagnostic(_panel())
    assert d["n_gaps"] == 0
    assert d["largest_gap_sessions"] == 0


def test_gaps_in_the_retained_rows_are_counted_and_sized():
    """Section 9: k positions is not k sessions once the rows have holes, and the
    reader is given the numbers needed to judge how far apart those are."""
    panel = _panel()
    # well clear of the 62-session detrend warm-up, or these would only move
    # the sample's start rather than punch a hole in it
    for row in (70, 71, 72):
        panel.loc[row, "n_headlines"] = 0
    panel.loc[95, "n_headlines"] = 0

    d = inf.timing_diagnostic(panel)
    assert d["n_gaps"] == 2
    assert d["largest_gap_sessions"] == 3
    assert "not a shift of k calendar sessions" in d["reporting_note"]


def test_the_reference_distribution_summary_is_reported():
    d = inf.timing_diagnostic(_panel())
    assert d["shifted_coef_min"] <= d["shifted_coef_mean"] <= d["shifted_coef_max"]
    assert d["shifted_bps_p05"] <= d["shifted_bps_p95"]
    assert len(d["shifted_coefficients"]) == d["n_shifts"]


def test_a_panel_with_too_few_rows_is_refused():
    panel = _panel(periods=65)   # the 62-session warm-up leaves exactly two rows
    assert int(inf.eligibility(panel)[0].sum()) == 2
    with pytest.raises(ValueError, match="at least three retained rows"):
        inf.timing_diagnostic(panel)


def test_a_single_eligible_row_is_refused_by_the_same_guard():
    """The sample size is checked before fitting, so an under-identified design
    never reaches statsmodels and the message names the real problem."""
    assert int(inf.eligibility(_panel(periods=64))[0].sum()) == 1
    with pytest.raises(ValueError, match="at least three retained rows"):
        inf.timing_diagnostic(_panel(periods=64))
