"""R12 / B21: attenuation, aggregation, and where the classical model fails.

    python attenuation_study.py

Six demonstrations on generating processes specified here, with the true
parameter known by construction in every case. Nothing in this file reads
project data, and no quantity produced is an estimate of anything about SPY or
about these scorers. See docs/archive/mathematical-appendix.md.

The point of a simulation with a known answer is that it can be wrong. Each
case prints the analytic prediction beside the simulated value; a mismatch is a
defect in the derivation or the code, not a finding.
"""

from __future__ import annotations

import numpy as np

SEED = 20260830
N_DAYS = 200_000          # large enough that Monte-Carlo error is well under
RNG = np.random.default_rng(SEED)   # the differences being demonstrated


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Simple regression slope through centred data."""
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x @ x))


def standardized_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope on z(x): the protocol's scale, log return per 1 SD of tone."""
    return ols_slope((x - x.mean()) / x.std(ddof=1), y)


def banner(n: int, title: str) -> None:
    print(f"\n{'=' * 72}\n{n}. {title}\n{'=' * 72}")


# ---------------------------------------------------------------- 1. classical


def case_1_classical(theta: float = 0.5, var_z: float = 1.0) -> None:
    banner(1, "Classical attenuation: plim(beta) = theta * lambda")
    print(f"   r = theta*Z + eps,  S = Z + u,  theta = {theta}, Var(Z) = {var_z}")
    print(f"\n   {'Var(u)':>8}{'lambda':>10}{'predicted':>12}{'simulated':>12}{'error':>10}")
    for var_u in (0.0, 0.25, 1.0, 4.0):
        z = RNG.normal(0, np.sqrt(var_z), N_DAYS)
        u = RNG.normal(0, np.sqrt(var_u), N_DAYS)
        r = theta * z + RNG.normal(0, 1.0, N_DAYS)
        lam = var_z / (var_z + var_u)
        got = ols_slope(z + u, r)
        print(f"   {var_u:>8.2f}{lam:>10.4f}{theta * lam:>12.4f}{got:>12.4f}{got - theta*lam:>10.4f}")
    print("\n   lambda <= 1 always, so the estimate is biased toward zero and never")
    print("   changes sign. Attenuation cannot manufacture an association.")


# ------------------------------------------------------------ 2. standardized


def case_2_standardized(theta: float = 0.5, var_z: float = 1.0) -> None:
    banner(2, "Standardized regressor: the exponent becomes 1/2")
    print("   beta_z = theta * sd(Z) * sqrt(lambda), and the scale factor a drops out")
    print(f"\n   {'a':>5}{'Var(u)':>8}{'lambda_a':>10}{'predicted':>12}{'simulated':>12}")
    for a in (1.0, 2.0):
        for var_u in (0.0, 0.25, 1.0):
            z = RNG.normal(0, np.sqrt(var_z), N_DAYS)
            u = RNG.normal(0, np.sqrt(var_u), N_DAYS)
            r = theta * z + RNG.normal(0, 1.0, N_DAYS)
            lam_a = a**2 * var_z / (a**2 * var_z + var_u)
            pred = theta * np.sqrt(var_z) * np.sqrt(lam_a)
            got = standardized_slope(a * z + u, r)
            print(f"   {a:>5.1f}{var_u:>8.2f}{lam_a:>10.4f}{pred:>12.4f}{got:>12.4f}")
    print("\n   At Var(u) = 0 both scales give the same coefficient: standardization")
    print("   removes a. sqrt(lambda) >= lambda, so shrinkage is milder than the")
    print("   textbook formula and scorer differences are correspondingly smaller.")


# -------------------------------------------------------------- 3. aggregation


def case_3_aggregation(theta: float = 0.5, var_zbar: float = 0.01) -> None:
    banner(3, "Aggregation compresses a large per-headline difference")
    print("   S_t is the mean of n_t headline scores. Independent per-headline")
    print("   error averages down as Var(u)/n; the day-to-day signal Var(Zbar) does not.")
    better, worse = 0.25, 0.50
    print(f"\n   Var(Zbar) = {var_zbar}, per-headline Var(u): better {better}, worse {worse} (2x)")
    print(f"\n   {'n_t':>6}{'lam better':>12}{'lam worse':>11}{'raw ratio':>11}{'std ratio':>11}")
    for n in (1, 5, 25, 100, 337, 1000):
        lam_b = var_zbar / (var_zbar + better / n)
        lam_w = var_zbar / (var_zbar + worse / n)
        print(f"   {n:>6}{lam_b:>12.4f}{lam_w:>11.4f}{lam_b/lam_w:>11.3f}"
              f"{np.sqrt(lam_b)/np.sqrt(lam_w):>11.3f}")

    n = 337
    zbar = RNG.normal(0, np.sqrt(var_zbar), N_DAYS)
    r = theta * zbar + RNG.normal(0, 1.0, N_DAYS)
    sim = {}
    for name, var_u in (("better", better), ("worse", worse)):
        ubar = RNG.normal(0, np.sqrt(var_u / n), N_DAYS)
        sim[name] = standardized_slope(zbar + ubar, r)
    print(f"\n   simulated standardized coefficients at n = {n}:")
    print(f"     better {sim['better']:.4f}   worse {sim['worse']:.4f}   "
          f"ratio {sim['better']/sim['worse']:.3f}")
    print("\n   A 2x gap in per-headline noise becomes ~1.03x in the standardized")
    print("   daily coefficient. The Act 1 -> Act 2 bridge is compressed by the")
    print("   design itself, before any question of whether the scorers differ.")


# -------------------------------------------------------- 4. correlated errors


def case_4_correlated(var_zbar: float = 0.01, var_u: float = 0.25, n: int = 337) -> None:
    banner(4, "Correlated within-day error removes the compression")
    print("   A scorer that systematically misreads one kind of headline makes")
    print("   correlated mistakes on days such headlines cluster.")
    print(f"\n   Var(ubar) = Var(u) * (1 + (n-1)*rho) / n,  n = {n}, Var(u) = {var_u}")
    print(f"\n   {'rho':>7}{'Var(ubar)':>12}{'eff. divisor':>14}{'lambda':>10}")
    for rho in (0.0, 0.01, 0.05, 0.10, 0.25, 1.0):
        var_ubar = var_u * (1 + (n - 1) * rho) / n
        lam = var_zbar / (var_zbar + var_ubar)
        print(f"   {rho:>7.2f}{var_ubar:>12.5f}{var_u/var_ubar:>14.1f}{lam:>10.4f}")
    print("\n   At rho = 0.1 the effective divisor is ~10, not 337. Independence")
    print("   (A5) is load-bearing and is not established for these scorers.")


# ------------------------------------------------------------ 5. bounded score


def case_5_bounded(theta: float = 0.5, var_z: float = 1.0, var_u: float = 0.5) -> None:
    banner(5, "Bounded scores break A2: Cov(Z, u) != 0 by construction")
    print("   Every scorer maps to [-1, 1]. Near a bound the error cannot point")
    print("   outward, so it is negatively correlated with what is measured.")
    z = RNG.normal(0, np.sqrt(var_z), N_DAYS)
    noise = RNG.normal(0, np.sqrt(var_u), N_DAYS)
    r = theta * z + RNG.normal(0, 1.0, N_DAYS)

    s_unbounded = z + noise
    s_bounded = np.clip(z + noise, -1.0, 1.0)
    u_eff = s_bounded - z                      # the error that actually occurred

    cov_zu = float(np.cov(z, u_eff)[0, 1])
    lam = var_z / (var_z + var_u)
    classical = theta * lam
    corrected = theta * (var_z + cov_zu) / (var_z + 2 * cov_zu + float(np.var(u_eff)))

    print(f"\n   Cov(Z, u) unbounded : {float(np.cov(z, noise)[0,1]):+.4f}  (A2 holds)")
    print(f"   Cov(Z, u) bounded   : {cov_zu:+.4f}  (A2 fails)")
    print(f"\n   {'':<22}{'predicted':>12}{'simulated':>12}")
    print(f"   {'unbounded, classical':<22}{classical:>12.4f}{ols_slope(s_unbounded, r):>12.4f}")
    print(f"   {'bounded, classical':<22}{classical:>12.4f}{ols_slope(s_bounded, r):>12.4f}  <- mispredicts")
    print(f"   {'bounded, corrected':<22}{corrected:>12.4f}{ols_slope(s_bounded, r):>12.4f}")
    print("\n   With Cov(Z,u) < 0 the plim is not bounded above by theta, so the")
    print("   coefficient can be amplified rather than attenuated. 'Attenuation")
    print("   toward zero' is not a safe reading for a bounded score.")


# -------------------------------------------------------------- 6. theta = 0


def case_6_no_signal(var_z: float = 1.0) -> None:
    banner(6, "theta = 0: better measurement of an unrelated quantity")
    print("   If aggregate tone has no conditional relation to the next return,")
    print("   no reduction in measurement error creates one.")
    print(f"\n   {'Var(u)':>8}{'lambda':>10}{'simulated beta':>16}")
    for var_u in (4.0, 1.0, 0.25, 0.0):
        z = RNG.normal(0, np.sqrt(var_z), N_DAYS)
        u = RNG.normal(0, np.sqrt(var_u), N_DAYS)
        r = 0.0 * z + RNG.normal(0, 1.0, N_DAYS)
        lam = var_z / (var_z + var_u)
        print(f"   {var_u:>8.2f}{lam:>10.4f}{ols_slope(z + u, r):>16.4f}")
    print("\n   Act 1 improvements matter for Act 2 only conditional on theta != 0,")
    print("   which is what Act 2 is testing. The bridge cannot be assumed in advance.")


def main() -> None:
    print(__doc__.strip().split("\n\n")[0])
    print(f"seed = {SEED}, {N_DAYS:,} simulated sessions per case")
    case_1_classical()
    case_2_standardized()
    case_3_aggregation()
    case_4_correlated()
    case_5_bounded()
    case_6_no_signal()
    print("\n" + "=" * 72)
    print("Every process above is invented to isolate one mechanism. None of these")
    print("numbers estimates anything about SPY or about FinBERT, LM or VADER.")
    print("=" * 72)


if __name__ == "__main__":
    main()
