"""R07c: what the prespecified bandwidth `L = 5` counts (audit A09, amendment M7).

`L = 5` was prespecified as "one trading week". Whether the code delivers that
depends on what a lag counts. `nw_ols` resets the index before handing the data
to statsmodels, so lag `l` means `l` *rows of the analysis sample*, which spans
more than `l` sessions wherever the sample has gaps -- and the installed
`cov_hac_simple` documents an assumption of consecutive, equally spaced periods.

`hac_sandwich` takes the session index as an argument so that the two
conventions are the same computation with one input changed. These tests hold it
to that: with `arange(n)` it must reproduce the library term for term, and on a
gapped index it must match lag products computed here by hand.

No model of the real data is involved, and no coefficient of interest is
estimated. The generating processes are fixtures with known structure.
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
from src import inference as inf

L = 5


def _fixture(n, seed=11):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "z_tone": rng.normal(size=n),
            "c1": rng.normal(size=n),
            "c2": rng.uniform(0.0, 1.0, size=n),
        }
    )
    y = pd.Series(0.3 * X["c1"] - 0.2 * X["c2"] + rng.normal(scale=0.5, size=n))
    return y, X


# ------------------------------------- the contiguous case must match the library


def test_session_indexed_reduces_to_the_library_when_no_session_is_missing():
    """With no gaps the two conventions are the same estimator, so both must agree."""
    n = 150
    y, X = _fixture(n)
    lib = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": L})

    mine = inf.fit_hac(y, X, maxlags=L, session_index=np.arange(n),
                       convention="session_indexed")

    np.testing.assert_allclose(mine.params.to_numpy(), lib.params.to_numpy())
    np.testing.assert_allclose(mine.bse.to_numpy(), lib.bse.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(mine.cov.to_numpy(), np.asarray(lib.cov_params()), rtol=1e-12)


def test_retained_position_reduces_to_the_library_too():
    n = 150
    y, X = _fixture(n)
    lib = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    mine = inf.fit_hac(y, X, maxlags=L, convention="retained_position")
    np.testing.assert_allclose(mine.bse.to_numpy(), lib.bse.to_numpy(), rtol=1e-12)


def test_passing_arange_is_exactly_the_retained_position_convention():
    """The claim the design rests on: one computation, one differing input."""
    n = 90
    y, X = _fixture(n, seed=3)
    a = inf.fit_hac(y, X, maxlags=L, session_index=np.arange(n),
                    convention="session_indexed")
    b = inf.fit_hac(y, X, maxlags=L, convention="retained_position")
    np.testing.assert_allclose(a.cov.to_numpy(), b.cov.to_numpy(), rtol=1e-14)


def test_maxlags_zero_is_the_white_sandwich_under_both_conventions():
    n = 80
    y, X = _fixture(n, seed=5)
    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()
    xu = Xc.to_numpy() * ols.resid.to_numpy()[:, None]
    XtXi = np.linalg.inv(Xc.to_numpy().T @ Xc.to_numpy())
    hc0 = XtXi @ (xu.T @ xu) @ XtXi

    sessions = np.arange(0, 4 * n, 4)  # spread out; with L = 0 spacing cannot matter
    for conv, si in (("session_indexed", sessions), ("retained_position", None)):
        got = inf.fit_hac(y, X, maxlags=0, session_index=si, convention=conv)
        np.testing.assert_allclose(got.cov.to_numpy(), hc0, rtol=1e-12)


# --------------------------------------------- the gapped case, checked by hand


def _hand_sandwich(Xc, resid, maxlags, sessions):
    """The estimator written out independently of the implementation."""
    xu = Xc * resid[:, None]
    n = len(sessions)
    S = np.zeros((Xc.shape[1], Xc.shape[1]))
    for i in range(n):
        for j in range(n):
            d = abs(int(sessions[i]) - int(sessions[j]))
            if d > maxlags:
                continue
            w = 1.0 - d / (maxlags + 1.0)
            S += w * np.outer(xu[i], xu[j])
    XtXi = np.linalg.inv(Xc.T @ Xc)
    return XtXi @ S @ XtXi


def test_gapped_covariance_matches_lag_products_computed_independently():
    """Every pair, weighted by its session distance -- the definition, spelled out."""
    rng = np.random.default_rng(23)
    sessions = np.array(sorted(rng.choice(200, size=70, replace=False)))
    y, X = _fixture(len(sessions), seed=9)

    got = inf.fit_hac(y, X, maxlags=L, session_index=sessions,
                      convention="session_indexed")

    Xc = sm.add_constant(X).to_numpy()
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    want = _hand_sandwich(Xc, ols.resid.to_numpy(), L, sessions)
    np.testing.assert_allclose(got.cov.to_numpy(), want, rtol=1e-11, atol=1e-18)


def test_rows_more_than_L_sessions_apart_contribute_nothing():
    """The property that makes the convention mean what `L` says.

    Space every retained session more than `L` apart. No pair is then within one
    trading week, so the session-indexed estimator must collapse to the White
    sandwich -- while retained-position, which still sees adjacent *rows*, does
    not.
    """
    n = 60
    y, X = _fixture(n, seed=17)
    sessions = np.arange(n) * (L + 1)  # neighbouring rows are L+1 sessions apart

    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()
    xu = Xc.to_numpy() * ols.resid.to_numpy()[:, None]
    XtXi = np.linalg.inv(Xc.to_numpy().T @ Xc.to_numpy())
    hc0 = XtXi @ (xu.T @ xu) @ XtXi

    spaced = inf.fit_hac(y, X, maxlags=L, session_index=sessions,
                         convention="session_indexed")
    np.testing.assert_allclose(spaced.cov.to_numpy(), hc0, rtol=1e-12)

    rows = inf.fit_hac(y, X, maxlags=L, convention="retained_position")
    assert not np.allclose(rows.cov.to_numpy(), hc0), (
        "retained-position must still weight adjacent rows; if it did not, the "
        "two conventions could never differ and there would be nothing to decide"
    )


def test_one_interior_gap_changes_only_the_pairs_that_straddle_it():
    """A single hole, hand-checked: the estimator is local, so the change is local."""
    n = 40
    y, X = _fixture(n, seed=31)
    sessions = np.arange(n)
    gapped = sessions.copy()
    gapped[20:] += 3          # a three-session hole between rows 19 and 20

    a = inf.fit_hac(y, X, maxlags=L, session_index=gapped, convention="session_indexed")
    b = inf.fit_hac(y, X, maxlags=L, session_index=sessions, convention="session_indexed")
    assert not np.allclose(a.cov.to_numpy(), b.cov.to_numpy())

    Xc = sm.add_constant(X).to_numpy()
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    np.testing.assert_allclose(
        a.cov.to_numpy(),
        _hand_sandwich(Xc, ols.resid.to_numpy(), L, gapped),
        rtol=1e-11, atol=1e-18,
    )


# --------------------------------------------------------------------- guards


def test_a_session_index_with_duplicates_is_refused():
    y, X = _fixture(10)
    with pytest.raises(ValueError, match="unique"):
        inf.fit_hac(y, X, maxlags=L, session_index=[0, 1, 1, 3, 4, 5, 6, 7, 8, 9])


def test_an_unsorted_session_index_is_refused():
    y, X = _fixture(6)
    with pytest.raises(ValueError, match="strictly increasing"):
        inf.fit_hac(y, X, maxlags=L, session_index=[0, 2, 1, 3, 4, 5])


def test_a_session_index_of_the_wrong_length_is_refused():
    y, X = _fixture(10)
    with pytest.raises(ValueError, match="session_index has"):
        inf.fit_hac(y, X, maxlags=L, session_index=np.arange(9))


def test_an_unknown_convention_is_refused():
    y, X = _fixture(10)
    with pytest.raises(ValueError, match="convention must be one of"):
        inf.fit_hac(y, X, maxlags=L, convention="whatever")


def test_fit_hac_refuses_missing_values_rather_than_dropping_them():
    """The sample is decided by `eligibility` and counted; a dropna here would
    silently make the fitted rows differ from the ledger's."""
    y, X = _fixture(20)
    X.loc[3, "c1"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        inf.fit_hac(y, X, maxlags=L, convention="retained_position")


def test_the_configured_convention_is_one_the_code_implements():
    assert config.HAC_CONVENTION in inf.HAC_CONVENTIONS
