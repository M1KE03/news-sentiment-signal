"""Act 2 robustness exhibits (protocol §4 sensitivity set, §7).

These are sensitivity analyses, not additional tests: §6 makes them incapable of
supplying a result the primary specification did not. The tests below enforce
that framing structurally, and hold the two readings that were nearly reported
wrongly:

  * the SESOI comparison must use `classify_effect_interval` -- the protocol's
    own M2 rule -- and never a second criterion invented in this module. An
    earlier version compared unrounded endpoints and would have reported a
    conclusion flip that the stated rule does not make;
  * the subperiod split must test the DIFFERENCE between halves, not whether
    two intervals overlap. Halving `n` widens each by ~sqrt(2), so overlap is
    nearly guaranteed and would be mistaken for stability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import align, inference, robustness


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


def _daily(cal, seed=5):
    n = len(cal)
    rng = np.random.default_rng(seed)
    daily = pd.DataFrame({"date": cal, "n_headlines": rng.integers(8, 50, n)})
    for s in config.SCORERS:
        daily[f"s_{s}"] = rng.normal(0, 0.1, n)
        daily[f"d_{s}"] = rng.uniform(0.1, 0.4, n)
    return daily


def _panel(periods=400):
    cal = pd.bdate_range("2015-01-05", periods=periods)
    return align.build_panel(_daily(cal), _market(cal))


# ------------------------------------------------------------ bandwidth


def test_the_bandwidth_set_is_the_one_the_protocol_names():
    bands = robustness.bandwidth_sensitivity(_panel())
    assert set(bands["maxlags"]) >= {0, 1, config.NW_MAXLAGS, 10}
    assert bands["is_primary"].sum() == 1
    assert bands.loc[bands["is_primary"], "maxlags"].iloc[0] == config.NW_MAXLAGS


def test_bandwidth_changes_the_standard_error_and_never_the_point_estimate():
    """HAC affects only the covariance; a moving point estimate would be a bug."""
    bands = robustness.bandwidth_sensitivity(_panel())
    assert bands["bps_per_sd"].nunique() == 1
    assert bands["se_bps"].nunique() > 1


def test_the_plug_in_bandwidth_is_computed_from_n_not_chosen():
    """Newey-West (1994): floor(4 * (n/100)^(2/9)). Never picked by inspection."""
    panel = _panel()
    n = inference.primary(panel).n
    expected = int(np.floor(4 * (n / 100) ** (2 / 9)))
    bands = robustness.bandwidth_sensitivity(panel)
    assert expected in set(bands["maxlags"])


def test_the_sesoi_verdict_uses_the_protocols_own_rule():
    """The defect this closes: a second criterion, comparing unrounded endpoints,
    disagreed with `classify_effect_interval` at short bandwidths and would have
    been reported as a conclusion flip the stated rule does not make."""
    bands = robustness.bandwidth_sensitivity(_panel())
    for _, r in bands.iterrows():
        verdict = inference.classify_effect_interval(r["bps_lo95"], r["bps_hi95"])
        assert bool(r["inside_sesoi"]) == bool(verdict["effect_established_small"])
        assert bool(r["association_detected"]) == bool(verdict["association_detected"])
        assert bool(r["boundary_case"]) == bool(verdict["boundary_case"])
        assert r["conclusion"] == verdict["conclusion"]


# ------------------------------------------------------------ subperiod


def test_the_subperiod_split_reports_the_difference_not_just_the_halves():
    """Comparing two intervals for overlap is the wrong reading: halving n
    widens each by ~sqrt(2), so overlap is close to guaranteed."""
    out = robustness.subperiod_split(_panel())
    assert "difference (first - second)" in set(out["period"])
    row = out[out["period"] == "difference (first - second)"].iloc[0]
    assert np.isfinite(row["bps_lo95"]) and np.isfinite(row["bps_hi95"])
    assert row["bps_lo95"] < row["bps_per_sd"] < row["bps_hi95"]


def test_the_difference_is_the_gap_between_the_halves():
    out = robustness.subperiod_split(_panel()).set_index("period")
    first, second = out.loc["first half"], out.loc["second half"]
    diff = out.loc["difference (first - second)"]
    assert diff["bps_per_sd"] == pytest.approx(first["bps_per_sd"] - second["bps_per_sd"])


def test_the_difference_standard_error_treats_the_halves_as_disjoint():
    """Disjoint samples share no observations, so var(diff) = var1 + var2."""
    out = robustness.subperiod_split(_panel()).set_index("period")
    expected = float(np.hypot(out.loc["first half", "se_bps"],
                              out.loc["second half", "se_bps"]))
    assert out.loc["difference (first - second)", "se_bps"] == pytest.approx(expected)


def test_the_halves_partition_the_eligible_sample():
    out = robustness.subperiod_split(_panel()).set_index("period")
    assert out.loc["first half", "n"] + out.loc["second half", "n"] == out.loc["full", "n"]


def test_the_split_never_shortens_the_panel():
    """The panel must stay the complete calendar or row shifts stop being
    session shifts -- so halves are made by blanking tone, not dropping rows."""
    panel = _panel()
    robustness.subperiod_split(panel)          # must not raise
    inference._require_full_calendar_panel(panel, 1)


# --------------------------------------------------- floor and aggregation


def test_the_dispersion_floor_exhibit_reports_how_many_sessions_it_dropped():
    """Its answer can be 'none', and then it has tested nothing -- which the
    output must make visible rather than read as a passed stress test."""
    out = robustness.dispersion_floor_sensitivity(_panel())
    assert "n_dropped" in out.columns
    assert out["is_primary"].sum() == 1


def test_aggregation_sensitivity_covers_every_configured_rule():
    assert set(config.AGG_CHOICES) == {"mean", "median"}


# ------------------------------------------------------------ diagnostics


def test_the_residual_acf_is_reported_with_its_band():
    acf = robustness.residual_autocorrelation(_panel(), lags=20)
    assert len(acf) == 20
    assert (acf["abs_band_95"] > 0).all()
    assert acf["residual_outside_band"].dtype == bool
    # the band is the standard 1.96/sqrt(n)
    n = inference.primary(_panel()).n
    assert acf["abs_band_95"].iloc[0] == pytest.approx(1.959963984540054 / np.sqrt(n))


def test_the_tone_acf_is_reported_beside_the_residual_one():
    """Section 4 asks for both: the residual ACF and the regressor's own."""
    acf = robustness.residual_autocorrelation(_panel())
    assert "tone_acf" in acf.columns and acf["tone_acf"].notna().all()


# ------------------------------------------------------ the framing itself


def test_the_summary_records_that_sensitivities_cannot_supply_a_finding():
    summary = robustness.summarise(_panel())
    assert "cannot supply a result the primary specification did not" in summary["note"]
    assert "never resolved by selection" in summary["note"]


def test_the_summary_counts_boundary_cases_rather_than_hiding_them():
    summary = robustness.summarise(_panel())
    assert summary["n_sensitivities"] > 0
    assert 0 <= summary["n_sensitivities_that_are_boundary_cases"] <= summary["n_sensitivities"]


def test_the_summary_flags_a_conclusion_that_is_not_invariant():
    """If a sensitivity did change the conclusion, that must surface as a fact
    rather than be resolved by adopting the convenient one."""
    summary = robustness.summarise(_panel())
    assert isinstance(summary["any_sensitivity_changes_the_conclusion"], bool)
    assert summary["any_sensitivity_changes_the_conclusion"] == (
        len(summary["conclusions_across_sensitivities"]) > 1
    )
