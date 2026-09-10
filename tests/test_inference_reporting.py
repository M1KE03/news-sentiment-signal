"""R07d/R08a: protocol conclusions and multiplicity, using synthetic inputs only."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import inference


@pytest.mark.parametrize("lo,hi,association,small", [
    (-2, 2, False, True), (1, 3, True, True),
    (-8, 8, False, False), (6, 8, True, False),
    (-5, 5, False, True), (0, 5, False, True),
    (-5, 0, False, True), (-3, -1, True, True),
    (0.04, 5.04, False, True), (0.06, 5.06, True, False),
])
def test_interval_dimensions_and_closed_rounded_boundaries(lo, hi, association, small):
    result = inference.classify_effect_interval(lo, hi)
    assert result["association_detected"] is association
    assert result["effect_established_small"] is small
    assert result["interval_scope"] == "pointwise"
    assert (result["conclusion"] == "Inconclusive") == (not association and not small)


def test_boundary_disclosure_preserves_unrounded_values():
    result = inference.classify_effect_interval(0.0491, 5.0491)
    assert result["boundary_case"]
    assert result["boundary_endpoints_bps"] == "[0.0491, 5.0491]"
    assert result["bps_lo95_rounded"] == 0
    assert result["bps_hi95_rounded"] == 5
    assert "0.1 bps" in result["boundary_note"]
    assert inference.classify_effect_interval(0.1, 4)["boundary_case"]
    assert not inference.classify_effect_interval(0.1001, 4)["boundary_case"]


@pytest.mark.parametrize("lo,hi", [(3, 1), (np.nan, 1), (-1, np.inf)])
def test_invalid_intervals_are_not_classified(lo, hi):
    with pytest.raises(ValueError, match="finite and ordered"):
        inference.classify_effect_interval(lo, hi)


def test_effect_table_keeps_pointwise_intervals_and_discards_profitability():
    family = pd.DataFrame({"coef": [0.0002], "nw_se": [0.00005],
                           "bh_q": [0.02], "clears_costs": [True]}, index=[9])
    result = inference.effect_size_table(family)
    row = result.loc[9]
    assert row["bps_per_sd"] == pytest.approx(2)
    assert row["bps_lo95"] == pytest.approx(1.02)
    assert row["bps_hi95"] == pytest.approx(2.98)
    assert row["association_detected"] and row["effect_established_small"]
    assert row["interval_scope"] == "pointwise"
    assert "clears_costs" not in result
    assert "ruled_out_above_bps" not in result
    assert "clears_costs" in family  # caller data was not mutated
    np.testing.assert_allclose(inference.effect_size_bps(0.0004, 0.0001, 0.5), [2, 1.02, 2.98])


@pytest.mark.parametrize("coef,se,sd", [(0, -1, 1), (0, 1, 0), (np.inf, 1, 1)])
def test_effect_scale_rejects_invalid_inputs(coef, se, sd):
    with pytest.raises(ValueError):
        inference.effect_size_bps(coef, se, sd)


def _precision_inputs():
    # Orthogonal, mean-zero +/-1 contrasts of length 8 permit hand arithmetic.
    x1 = np.array([-1, 1] * 4)
    x2 = np.array([-1, -1, 1, 1] * 2)
    x3 = np.array([-1] * 4 + [1] * 4)
    u = x1 * x2
    v = x1 * x3
    controls = pd.DataFrame({"ret": x1, "rv_parkinson": x2,
                             "log_volume_detrended": x3})
    return pd.Series(0.002 * x1 + 0.003 * u), pd.Series(2 * x1 + v), controls


def test_advance_widths_match_orthogonal_fixture_without_tone_outcome_fit(monkeypatch):
    y, tone, controls = _precision_inputs()
    original = np.linalg.lstsq
    targets = []

    def spy(design, target, **kwargs):
        assert design.shape == (8, 4)
        np.testing.assert_array_equal(design[:, 1:], controls)
        targets.append(target.copy())
        return original(design, target, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", spy)
    result = inference.advance_precision(y, tone, controls, kappa=1.2, hac_spacing="session_indexed")
    assert len(targets) == 2
    np.testing.assert_allclose(targets[0], y)
    np.testing.assert_allclose(targets[1], (tone - tone.mean()) / tone.std(ddof=1))
    assert result["n"] == 8
    assert result["tone_controls_r_squared"] == pytest.approx(0.8)
    assert result["sd_v"] == pytest.approx(np.sqrt(0.2))
    assert result["h1_bps"] == pytest.approx(1.96 * np.sqrt(13e-6 / 7) * 10000)
    assert result["h2_bps"] == pytest.approx(1.96 * 0.003 / np.sqrt(7 * 0.2) * 1.2 * 10000)
    assert result["purpose"] == "advance_planning_estimates_not_bounds"
    assert result["recorded_at_utc"].endswith("+00:00")
    assert not any("coef" in key or key == "p" for key in result)


def test_advance_precision_rejects_missing_and_misaligned_rows():
    y, tone, controls = _precision_inputs()
    with pytest.raises(ValueError, match="indices"):
        inference.advance_precision(y.iloc[::-1], tone, controls, kappa=1, hac_spacing="session_indexed")
    y.iloc[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        inference.advance_precision(y, tone, controls, kappa=1, hac_spacing="session_indexed")


def test_advance_precision_requires_kappa_and_declared_spacing():
    y, tone, controls = _precision_inputs()
    with pytest.raises(TypeError, match="kappa"):
        inference.advance_precision(y, tone, controls, hac_spacing="session_indexed")
    with pytest.raises(ValueError, match="hac_spacing"):
        inference.advance_precision(y, tone, controls, kappa=1, hac_spacing="automatic")
    with pytest.raises(ValueError, match="kappa"):
        inference.advance_precision(y, tone, controls, kappa=0, hac_spacing="session_indexed")
    with pytest.raises(ValueError, match="degenerate"):
        inference.advance_precision(y, controls["ret"], controls, kappa=1, hac_spacing="session_indexed")


def test_precision_widths_are_invariant_to_tone_units():
    y, tone, controls = _precision_inputs()
    base = inference.advance_precision(y, tone, controls, kappa=1, hac_spacing="retained_position")
    scaled = inference.advance_precision(y, 300 - 1000 * tone, controls, kappa=1, hac_spacing="retained_position")
    for key in ("h1_bps", "h2_bps", "tone_controls_r_squared"):
        assert scaled[key] == pytest.approx(base[key])


def _return_estimates():
    pairs = [(sc, h) for sc in ("finbert", "lm", "vader") for h in range(1, 6)]
    return pd.DataFrame({
        "scorer": [p[0] for p in pairs], "horizon": [p[1] for p in pairs],
        "p": [1e-8, 0.0001, 0.001, 0.01, 0.02, 0.2, 0.3, 0.4,
              0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        "coef": np.zeros(15), "nw_se": np.ones(15),
    })


def test_exact_family_and_independent_bh_by_reference_values():
    result = inference.adjust_return_families(_return_estimates())
    primary = result.iloc[0]
    assert primary["scorer"] == "finbert" and primary["horizon"] == 1
    assert primary["family_size"] == 1 and primary["claim_basis"] == "unadjusted"
    assert primary["p"] == 1e-8
    assert pd.isna(primary["bh_q"]) and pd.isna(primary["by_reject"])
    secondary = result.iloc[1:]
    assert len(secondary) == 14 and secondary["family_size"].eq(14).all()
    # Hand-computed first three ranks: m*p/r; later ranks cannot lower them.
    expected_bh = np.array([0.0014, 0.007, 0.14 / 3, 0.07, 0.56, 0.7, 0.8,
                            0.875, 8.4 / 9, 0.98, 0.99, 0.99, 0.99, 0.99])
    np.testing.assert_allclose(secondary["bh_q"], expected_bh)
    harmonic14 = sum(1 / rank for rank in range(1, 15))
    np.testing.assert_allclose(secondary["by_q"], np.minimum(1, expected_bh * harmonic14))
    assert secondary["bh_reject"].sum() == 3
    assert secondary["by_reject"].sum() == 2
    disagreement = secondary.iloc[2]
    assert disagreement["bh_by_disagree"] and not disagreement["claim_reject"]
    assert result["interval_scope"].eq("pointwise").all()


def test_corrections_ignore_primary_p_and_input_order():
    estimates = _return_estimates()
    reference = inference.adjust_return_families(estimates)
    shuffled = estimates.sample(frac=1, random_state=9)
    shuffled.index = [99] * 15
    pd.testing.assert_frame_equal(reference, inference.adjust_return_families(shuffled))
    estimates.loc[0, "p"] = 0.99
    changed = inference.adjust_return_families(estimates)
    pd.testing.assert_frame_equal(reference.iloc[1:], changed.iloc[1:])
    assert not changed.iloc[0]["claim_reject"]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "extra", "scorer", "horizon", "classification", "nan", "infinity", "negative"])
def test_wrong_or_failed_return_tests_cannot_shrink_the_family(problem):
    estimates = _return_estimates()
    if problem == "missing":
        estimates = estimates.iloc[:-1]
    elif problem == "duplicate":
        estimates.iloc[-1] = estimates.iloc[0]
    elif problem == "extra":
        estimates = pd.concat([estimates, estimates.iloc[:1]])
    elif problem == "scorer":
        estimates.loc[0, "scorer"] = "unknown"
    elif problem == "horizon":
        estimates["horizon"] = estimates["horizon"].astype(float) + 0.5
    elif problem == "classification":
        estimates["outcome"] = "accuracy"
    else:
        estimates.loc[1, "p"] = {"nan": np.nan, "infinity": np.inf, "negative": -0.1}[problem]
    with pytest.raises(ValueError):
        inference.adjust_return_families(estimates)


def test_lag_family_defers_correction_until_all_scorers_are_assembled(monkeypatch):
    from types import SimpleNamespace

    def fake_fit(panel, scorer, horizon, maxlags):
        fit = SimpleNamespace(params={f"z_s_{scorer}": 0}, bse={f"z_s_{scorer}": 1},
                              tvalues={f"z_s_{scorer}": 0}, pvalues={f"z_s_{scorer}": 0.5}, nobs=80)
        return SimpleNamespace(fit=fit, alternate=fit, tone_sd=0.4, convention="session_indexed")

    monkeypatch.setattr(inference, "primary", fake_fit)
    family = inference.lag_family(pd.DataFrame(), "finbert")
    assert len(family) == 5
    assert "p" in family and "bh_q" not in family and "by_q" not in family
    with pytest.raises(ValueError, match="15 declared"):
        inference.adjust_return_families(family)


def _full_panel():
    rng = np.random.default_rng(810)
    n = 100
    panel = pd.DataFrame({"date": pd.bdate_range("2015-01-05", periods=n),
                          "ret": rng.normal(0, 0.01, n),
                          "rv_parkinson": rng.uniform(0, 0.001, n),
                          "log_volume_detrended": rng.normal(size=n),
                          "n_headlines": np.full(n, 5)})
    for scorer in ("finbert", "lm", "vader"):
        panel[f"s_{scorer}"] = rng.normal(0, 0.2, n)
    for horizon in range(1, 6):
        panel[f"ret_lead{horizon}"] = panel["ret"].shift(-horizon)
    panel.loc[[8, 11, 12, 40], "n_headlines"] = 0
    panel.loc[:5, "log_volume_detrended"] = np.nan
    return panel


def test_primary_precision_uses_fit_rows_and_hand_computed_residual_mean_hac():
    panel = _full_panel()
    record = inference.primary_precision(panel)
    fitted = inference.primary(panel)
    assert record["n"] == fitted.n
    assert record["session_positions"] == fitted.positions.tolist()
    assert record["session_dates"] == fitted.dates.dt.strftime("%Y-%m-%d").tolist()
    rows = panel.iloc[fitted.positions]
    x = np.column_stack([np.ones(len(rows)), rows[list(inference.PRIMARY_CONTROLS)]])
    y = rows["ret_lead1"].to_numpy()
    u = y - x @ np.linalg.solve(x.T @ x, x.T @ y)
    for spacing, ratio in record["kappa_by_spacing"].items():
        times = fitted.positions if spacing == "session_indexed" else np.arange(len(rows))
        # Direct double sum over all pairs: independent of the lag-loop helper.
        long_run_sum = sum(
            u[i] * u[j] * max(0, 1 - abs(int(times[i]) - int(times[j])) / 6)
            for i in range(len(rows)) for j in range(len(rows))
        )
        expected_ratio = np.sqrt(long_run_sum / len(rows) ** 2) / (u.std(ddof=1) / np.sqrt(len(rows)))
        assert ratio == pytest.approx(expected_ratio)
    assert record["kappa_by_spacing"]["retained_position"] != pytest.approx(record["kappa_by_spacing"]["session_indexed"])


def test_return_family_coefficients_are_standardized_with_detrended_controls():
    panel = _full_panel()
    family = inference.lag_family(panel, "finbert")
    scaled = panel.copy()
    scaled["s_finbert"] *= 100
    after = inference.lag_family(scaled, "finbert")
    np.testing.assert_allclose(family[["coef", "nw_se", "p"]], after[["coef", "nw_se", "p"]])
    assert family["coefficient_scale"].eq("standardized_tone").all()
    effects = inference.effect_size_table(family)
    np.testing.assert_allclose(effects["bps_per_sd"], family["coef"] * 10000)
    assert family["nobs"].iloc[-1] < family["nobs"].iloc[0]


def test_runner_persists_advance_record_before_first_tone_fit(tmp_path, monkeypatch):
    import json
    import run_all

    monkeypatch.setattr(run_all, "TABLES", tmp_path)

    class ReachedFirstToneFit(Exception):
        pass

    def stop_at_first_fit(panel, scorer):
        record = json.loads((tmp_path / "advance_precision.json").read_text(encoding="utf-8"))
        assert record["n"] > 0 and record["kappa_source"].startswith("controls_only")
        assert not list(tmp_path.glob("*.tmp"))
        raise ReachedFirstToneFit

    monkeypatch.setattr(inference, "lag_family", stop_at_first_fit)
    with pytest.raises(ReachedFirstToneFit):
        run_all.run_analysis(_full_panel(), draws=1)


def test_failed_advance_publication_prevents_any_regression(tmp_path, monkeypatch):
    import run_all

    monkeypatch.setattr(run_all, "TABLES", tmp_path)
    def refuse_replace(*args):
        raise OSError("fixture publication failure")
    monkeypatch.setattr(run_all.os, "replace", refuse_replace)
    def unexpected_fit(*args, **kwargs):
        pytest.fail("a tone coefficient was estimated before the advance record was published")
    monkeypatch.setattr(inference, "primary", unexpected_fit)
    with pytest.raises(OSError, match="publication failure"):
        run_all.run_analysis(_full_panel(), draws=1)
    assert not (tmp_path / "advance_precision.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
