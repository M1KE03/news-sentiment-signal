"""R07c / M7: measure the two HAC spacing conventions on synthetic gapped data.

Audit A09 asked what the prespecified bandwidth `L = 5` counts once the analysis
sample has gaps. Two answers are available and the inference protocol requires
both to be computed; this script measures how far apart they are, so the choice
of primary convention is made against evidence rather than intuition.

    python hac_spacing_study.py

Everything here is synthetic. The generating process is fixed and known, and no
sentiment-return coefficient is involved: the regressor named `z_tone` has a
true coefficient of exactly zero, and only its STANDARD ERROR is compared. This
script cannot be used to look at the primary result early.

The design isolates spacing. The retained rows are the same data in both
branches; only the session index differs, and `inference.hac_sandwich` takes
that index as an argument. So any divergence is attributable to the pairing rule
and to nothing else.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import config
from src import inference as inf

SEED = 20260830
N_SESSIONS = 2516        # the locked window's session count
N_REPS = 60
RHO_TONE = 0.85          # daily aggregate tone is persistent


def retained_sessions(n_sessions: int, gap_frac: float, rng) -> np.ndarray:
    """Session indices that survive to the analysis sample."""
    if gap_frac <= 0:
        return np.arange(n_sessions)
    keep = rng.random(n_sessions) >= gap_frac
    keep[0] = keep[-1] = True
    return np.flatnonzero(keep)


def simulate(sessions: np.ndarray, rho_resid: float, rng):
    """Build the sample. Dependence is generated in SESSION time, then sampled.

    This is the case that separates the conventions: the residual process is
    autocorrelated over the exchange calendar, so two rows that are adjacent in
    the analysis sample but far apart in session time are genuinely much less
    dependent than retained-position pairing assumes.
    """
    n_full = int(sessions[-1]) + 1

    def ar1(rho):
        e = rng.normal(size=n_full)
        out = np.empty(n_full)
        out[0] = e[0]
        for t in range(1, n_full):
            out[t] = rho * out[t - 1] + np.sqrt(1 - rho**2) * e[t]
        return out

    u = ar1(rho_resid)
    tone = ar1(RHO_TONE)
    c1 = rng.normal(size=n_full)
    c2 = rng.normal(size=n_full)
    y_full = 0.0 * tone + 0.3 * c1 - 0.15 * c2 + 0.01 * u   # true beta on tone: 0

    X = pd.DataFrame(
        {"z_tone": tone[sessions], "c1": c1[sessions], "c2": c2[sessions]}
    ).reset_index(drop=True)
    return pd.Series(y_full[sessions]).reset_index(drop=True), X


def compare(sessions, rho_resid, rng, maxlags):
    y, X = simulate(sessions, rho_resid, rng)
    si = inf.fit_hac(y, X, maxlags=maxlags, session_index=sessions,
                     convention="session_indexed")
    rp = inf.fit_hac(y, X, maxlags=maxlags, convention="retained_position")
    return float(si.bse["z_tone"] / rp.bse["z_tone"])


def sweep(maxlags: int = config.NW_MAXLAGS, reps: int = N_REPS) -> pd.DataFrame:
    """Gap fraction x residual persistence."""
    rng = np.random.default_rng(SEED)
    rows = []
    for gap_frac in (0.0, 0.0004, 0.01, 0.05, 0.15, 0.30, 0.50):
        for rho in (0.0, 0.3, 0.6):
            ratios, kept = [], []
            for _ in range(reps):
                s = retained_sessions(N_SESSIONS, gap_frac, rng)
                ratios.append(compare(s, rho, rng, maxlags))
                kept.append(len(s))
            r = np.asarray(ratios)
            rows.append({
                "gap_frac": gap_frac,
                "rho_resid": rho,
                "n_retained": int(np.mean(kept)),
                "se_ratio_mean": r.mean(),
                "se_ratio_min": r.min(),
                "se_ratio_max": r.max(),
                "max_abs_pct_diff": float(np.abs(r - 1).max() * 100),
            })
    return pd.DataFrame(rows)


def anticipated_structure(maxlags: int = config.NW_MAXLAGS, reps: int = 200) -> pd.DataFrame:
    """The gap structure this corpus is actually expected to produce.

    A 62-session leading warm-up (the trailing 63-session volume detrend), one
    interior zero-news session -- the census found exactly one, and it is the
    window's first session, so this places it in the interior as the harsher
    case -- and the final session dropped for having no lead.
    """
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for rho in (0.0, 0.3, 0.6):
        ratios = []
        for _ in range(reps):
            s = np.arange(62, N_SESSIONS - 1)
            s = s[s != rng.integers(200, N_SESSIONS - 200)]
            ratios.append(compare(s, rho, rng, maxlags))
        r = np.asarray(ratios)
        rows.append({
            "rho_resid": rho,
            "n_retained": int(len(s)),
            "se_ratio_mean": r.mean(),
            "se_ratio_min": r.min(),
            "se_ratio_max": r.max(),
            "max_abs_pct_diff": float(np.abs(r - 1).max() * 100),
        })
    return pd.DataFrame(rows)


def main() -> None:
    fmt = lambda v: f"{v:.6g}"
    pd.set_option("display.width", 200)
    print(f"HAC spacing study -- L = {config.NW_MAXLAGS}, seed {SEED}, "
          f"{N_REPS} replications per cell")
    print("ratio = se(session_indexed) / se(retained_position) on the tone term\n")
    print("Gap sweep")
    print(sweep().to_string(index=False, float_format=fmt))
    print("\nAnticipated structure for this corpus "
          "(62-session warm-up, one interior gap, last session dropped)")
    print(anticipated_structure().to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
