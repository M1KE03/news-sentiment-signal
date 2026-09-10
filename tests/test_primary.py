"""R07b: the primary specification, its sample, and its scale (audit A07).

`inference.predictive` did not implement the frozen specification. It fitted
**raw** `log_volume` where the inference protocol Sections 1-3 specify the
trailing-63-session-detrended series -- a different regressor, not a renamed
column; it did not standardize tone at fit time, so the coefficient was not "per
1 SD" of anything; and the basis-point conversion multiplied by an `sd(S)`
computed on a different sample, rescaling a coefficient that the fit had already
scaled. `inference.primary` implements the frozen specification instead.

The three things these tests exist to hold:

  * the design matrix and target are what the protocol says, checkable by hand;
  * the fit, the standard deviation used to standardize, and the basis-point
    scale are all computed from the SAME retained rows;
  * the exclusion ledger reconciles -- every dropped session is charged to
    exactly one reason, and the reasons sum to the number dropped.

No real data is involved and no claim is made about any coefficient's value.
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
    # Log volume gets a deliberate upward trend, so raw and detrended are
    # unmistakably different regressors rather than near-copies.
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


def _panel(periods=200, **kw):
    cal = pd.bdate_range("2015-01-05", periods=periods)
    return align.build_panel(_daily(cal, **kw), _market(cal))


# ------------------------------------------------- the design matrix and target


def test_the_design_is_the_frozen_control_set_and_nothing_else():
    fit = inf.primary(_panel())
    assert list(fit.design.columns) == ["z_s_finbert", "ret", "rv_parkinson",
                                        "log_volume_detrended"]
    assert inf.PRIMARY_CONTROLS == ("ret", "rv_parkinson", "log_volume_detrended")


def test_the_volume_control_is_the_detrended_series_not_the_raw_one():
    """A07's concrete mismatch. These are distinguishable designs, not renames."""
    panel = _panel()
    fit = inf.primary(panel)
    rows = panel.loc[fit.mask]

    np.testing.assert_allclose(
        fit.design["log_volume_detrended"].to_numpy(),
        rows["log_volume_detrended"].to_numpy(),
    )
    raw = rows["log_volume"].to_numpy()
    assert not np.allclose(fit.design["log_volume_detrended"].to_numpy(), raw)
    # The trend is exactly what the detrend removes, so the raw series is far
    # more persistent than the control the protocol specifies.
    assert abs(np.corrcoef(raw, np.arange(len(raw)))[0, 1]) > 0.9
    assert abs(np.corrcoef(fit.design["log_volume_detrended"], np.arange(len(raw)))[0, 1]) < 0.5


def test_tone_is_standardized_at_fit_time_on_the_retained_rows():
    panel = _panel()
    fit = inf.primary(panel)
    z = fit.design["z_s_finbert"]
    assert abs(float(z.mean())) < 1e-12
    assert abs(float(z.std(ddof=1)) - 1.0) < 1e-12

    rows = panel.loc[fit.mask]
    assert fit.tone_mean == pytest.approx(float(rows["s_finbert"].mean()))
    assert fit.tone_sd == pytest.approx(float(rows["s_finbert"].std(ddof=1)))


def test_the_target_is_the_next_sessions_return():
    panel = _panel()
    fit = inf.primary(panel)
    rows = panel.loc[fit.mask]
    np.testing.assert_allclose(fit.target.to_numpy(), rows["ret_lead1"].to_numpy())
    # and that column is the return of the row one session later
    ret = panel["ret"].to_numpy()
    for p in fit.positions:
        assert fit.target.loc[p] == pytest.approx(ret[p + 1])


def test_a_corrupted_lead_column_fails_the_adjacency_assertion():
    """Section 3: assert what supplies r_(t+1); do not assume the column name."""
    panel = _panel()
    panel.loc[80, "ret_lead1"] = panel.loc[80, "ret_lead1"] + 0.05
    with pytest.raises(ValueError, match="does not equal ret.shift"):
        inf.primary(panel)


def test_a_prefiltered_panel_is_refused():
    """Row i+1 is only the next session while the panel is the whole calendar."""
    panel = _panel()
    filtered = panel.drop(index=range(100, 110)).reset_index(drop=True)
    with pytest.raises(ValueError, match="not the complete exchange calendar"):
        inf.primary(filtered)


def test_the_output_of_analysis_sample_is_refused_by_name():
    """The most likely way to make that mistake in a notebook."""
    panel = _panel()
    panel.loc[5, "n_headlines"] = 0
    sample, _ = inf.analysis_sample(panel)
    with pytest.raises(ValueError, match="not the complete exchange calendar"):
        inf.primary(sample)


# ------------------------------------------------------- the exclusion ledger


def test_the_ledger_reconciles():
    panel = _panel()
    _, ledger = inf.eligibility(panel)
    assert sum(ledger["excluded_first_reason"].values()) == ledger["n_excluded"]
    assert ledger["n_eligible"] + ledger["n_excluded"] == ledger["n_panel_rows"]
    assert ledger["n_panel_rows"] == len(panel)


def test_every_reason_is_recorded_separately_and_fires_when_it_should():
    panel = _panel()
    panel.loc[100, "n_headlines"] = 0            # zero-news
    panel.loc[120, "s_finbert"] = np.nan         # scored corpus, unscored session
    panel.loc[140, "rv_parkinson"] = np.nan      # a control undefined

    keep, ledger = inf.eligibility(panel)
    first = ledger["excluded_first_reason"]

    assert first["zero_news"] >= 1
    assert first["missing_tone"] >= 1
    # the 62-session detrend warm-up plus the injected hole
    assert first["missing_control"] >= 63
    # the window's final session has no lead
    assert first["no_lead_return"] >= 1
    assert not keep[100] and not keep[120] and not keep[140]
    assert sum(first.values()) == ledger["n_excluded"]


def test_the_detrend_warmup_is_charged_to_missing_control_not_to_zero_news():
    """A single count would not distinguish these, and they mean different things."""
    panel = _panel()
    _, ledger = inf.eligibility(panel)
    assert ledger["excluded_first_reason"]["missing_control"] >= 62
    assert ledger["excluded_first_reason"]["zero_news"] == 0


def test_per_reason_counts_are_also_reported_without_precedence():
    """First-reason counts partition; any-reason counts show the overlap."""
    panel = _panel()
    panel.loc[100, "n_headlines"] = 0
    panel.loc[100, "s_finbert"] = np.nan       # both reasons apply to this row
    _, ledger = inf.eligibility(panel)
    assert ledger["excluded_any_reason"]["zero_news"] >= 1
    assert ledger["excluded_any_reason"]["missing_tone"] >= 1
    assert sum(ledger["excluded_any_reason"].values()) >= ledger["n_excluded"]


def test_the_fitted_row_count_equals_the_ledgers_eligible_count():
    fit = inf.primary(_panel())
    assert fit.n == fit.ledger["n_eligible"] == int(fit.mask.sum())
    assert len(fit.design) == fit.n
    assert len(fit.target) == fit.n


# -------------------------------------------------- one sample, one scale


def test_the_bps_scale_is_the_fitted_coefficient_and_no_second_rescaling():
    """`z(S_t)` was standardized at fit time, so beta is already per 1 SD."""
    fit = inf.primary(_panel())
    point, lo, hi = fit.bps
    assert point == pytest.approx(fit.beta * config.BPS_PER_UNIT)
    half = 1.959963984540054 * fit.se * config.BPS_PER_UNIT
    assert lo == pytest.approx(point - half, rel=1e-6)
    assert hi == pytest.approx(point + half, rel=1e-6)


def test_rescaling_the_tone_series_leaves_the_standardized_result_unchanged():
    """The invariance A07's conversion broke: 2*S carries the same information.

    With standardization at fit time the coefficient and its basis-point scale
    are untouched. Multiplying a raw coefficient by an externally supplied
    `sd(S)` would not have this property.
    """
    panel = _panel()
    doubled = panel.copy()
    doubled["s_finbert"] = doubled["s_finbert"] * 2.0

    a = inf.primary(panel)
    b = inf.primary(doubled)

    assert b.tone_sd == pytest.approx(2 * a.tone_sd)
    assert b.beta == pytest.approx(a.beta)
    assert b.se == pytest.approx(a.se)
    np.testing.assert_allclose(b.bps, a.bps)


def test_the_standard_deviation_comes_from_the_retained_rows_only():
    """Excluding a session must move the SD; a whole-panel SD would not notice."""
    panel = _panel()
    base = inf.primary(panel)

    nudged = panel.copy()
    eligible_rows = np.flatnonzero(base.mask)
    nudged.loc[eligible_rows[10], "n_headlines"] = 0      # drop one eligible session

    after = inf.primary(nudged)
    assert after.n == base.n - 1
    assert after.tone_sd != pytest.approx(base.tone_sd)


def test_a_constant_tone_series_is_refused_rather_than_dividing_by_zero():
    panel = _panel()
    panel["s_finbert"] = 0.25
    with pytest.raises(ValueError, match="standard deviation"):
        inf.primary(panel)


# ------------------------------------------ both HAC conventions are available


def test_both_spacing_conventions_are_computed_and_the_primary_one_is_named():
    fit = inf.primary(_panel())
    assert set(fit.fits) == set(inf.HAC_CONVENTIONS)
    assert fit.convention == config.HAC_CONVENTION
    assert fit.alternate.convention != fit.convention
    # OLS does not depend on the covariance, so the point estimates must agree
    assert fit.fit.params[fit.tone_term] == pytest.approx(
        fit.alternate.params[fit.tone_term]
    )


def test_the_spacing_comparison_reports_both_standard_errors_per_term():
    fit = inf.primary(_panel())
    cmp = fit.spacing_comparison()
    assert list(cmp["term"]) == fit.fit.names
    assert {"se_session_indexed", "se_retained_position", "se_ratio"} <= set(cmp.columns)
    assert cmp.attrs["primary_convention"] == config.HAC_CONVENTION
    assert (cmp["se_ratio"] > 0).all()


def test_on_a_gapless_sample_the_two_conventions_agree_exactly():
    """The reduction, end to end: no gaps means nothing for spacing to change."""
    panel = _panel()
    fit = inf.primary(panel)
    # the eligible rows are one contiguous run (warm-up at the front, no interior
    # holes), so session distance equals row distance throughout
    assert np.all(np.diff(fit.positions) == 1)
    np.testing.assert_allclose(
        fit.fits["session_indexed"].bse.to_numpy(),
        fit.fits["retained_position"].bse.to_numpy(),
        rtol=1e-12,
    )


def test_an_interior_gap_separates_the_two_conventions():
    panel = _panel()
    for row in (100, 101, 102):
        panel.loc[row, "n_headlines"] = 0
    fit = inf.primary(panel)
    assert not np.all(np.diff(fit.positions) == 1)
    assert fit.fits["session_indexed"].bse[fit.tone_term] != pytest.approx(
        fit.fits["retained_position"].bse[fit.tone_term], rel=1e-12
    )


def test_the_ledger_records_the_convention_and_bandwidth_actually_used():
    fit = inf.primary(_panel(), maxlags=3)
    assert fit.ledger["hac_maxlags"] == 3
    assert fit.ledger["hac_convention"] == config.HAC_CONVENTION
    assert fit.ledger["n_fitted"] == fit.n
    assert fit.ledger["tone_sd"] == pytest.approx(fit.tone_sd)


def test_the_per_observation_adjacency_assertion_is_exercised_directly():
    """`_assert_lead_adjacency` restates the invariant per retained row.

    On a panel that passed `_require_full_calendar_panel` it cannot fire -- that
    check already compares the whole lead column against `ret.shift(-h)`. It is
    kept because the protocol asks for the assertion at the observation level
    (Section 3), and it is exercised here directly so it is not merely decorative:
    a future change to the panel-level check would still be caught by this one.
    """
    panel = _panel()
    broken = panel.copy()
    broken.loc[50, "ret_lead1"] = broken.loc[50, "ret_lead1"] + 0.1
    with pytest.raises(AssertionError, match="not the adjacent session"):
        inf._assert_lead_adjacency(broken, np.array([50]), 1)


def test_adjacency_refuses_to_run_off_the_end_of_the_panel():
    panel = _panel()
    with pytest.raises(AssertionError, match="no session 1 ahead"):
        inf._assert_lead_adjacency(panel, np.array([len(panel) - 1]), 1)


def test_the_last_session_is_excluded_for_having_no_lead():
    panel = _panel()
    keep, ledger = inf.eligibility(panel)
    assert not keep[len(panel) - 1]
    assert ledger["excluded_first_reason"]["no_lead_return"] >= 1
