# Mathematical appendix: attenuation, aggregation, and what they do not establish

Date: 2026-09-10. Increment: **R12** (B21). Decisions: **P06**, **P32**.

Status: **derivation and simulation**. No empirical result is used here and none is produced. Every number in this document comes from a generating process specified in advance, with the true parameter known by construction.

Related: [inference protocol](inference-protocol.md) §8 · [decision log](research-review-decision-log.md) P06, P32 · simulation: `attenuation_study.py`.

---

## 1. What this exists to settle

The original plan asserted a bridge between the two acts: a better classifier has less measurement error, therefore it produces a larger and better-determined return coefficient on the same specification. P06 replaced that with a conditional hypothesis.

This appendix does three things:

1. Derives classical attenuation, so the assumptions it needs are on the page.
2. Shows that **three of those assumptions fail or are unverified for this design**, and derives what replaces the formula in each case.
3. Demonstrates each result numerically on a known process.

The conclusion is stated up front because it is not a null result: **under this study's own design, a large difference in per-headline classification quality produces a small difference in the daily return coefficient.** The bridge is weak by construction, not merely unproven. §4 gives the magnitude.

## 2. Classical attenuation

Let `Z_t` be the latent, market-relevant information content of session `t`'s news. Centred variables throughout.

```text
r_(t+1) = theta * Z_t + eps_(t+1)        (outcome equation)
S_t     = Z_t + u_t                       (measurement equation)
```

Assumptions **A1–A4**:

- **A1** `E[Z] = E[u] = E[eps] = 0`
- **A2** `Cov(Z, u) = 0` — the measurement error is unrelated to what is measured
- **A3** `Cov(Z, eps) = 0` — the outcome equation is correctly specified
- **A4** `Cov(u, eps) = 0` — scoring error is unrelated to the return shock

The OLS slope of `r_(t+1)` on `S_t` converges to

```text
plim beta_hat = Cov(S, r) / Var(S)
              = Cov(Z + u, theta*Z + eps) / Var(Z + u)
              = theta * Var(Z) / (Var(Z) + Var(u))
              = theta * lambda
```

with the **reliability ratio**

```text
lambda = Var(Z) / (Var(Z) + Var(u))  in (0, 1].
```

Two consequences. `lambda <= 1`, so the estimate is biased **toward zero** and never changes sign: attenuation cannot manufacture an association, only shrink one. And `lambda` is a *variance ratio*, not an accuracy: nothing in the derivation refers to a classification metric.

## 3. The regressor is standardized, which changes the exponent

The protocol fits `z(S_t)`, not `S_t` (§1, §7). Allow also a scorer-specific scale `a`, since a shared `[-1, 1]` range is not a shared scale:

```text
S_t = a * Z_t + u_t
```

Then

```text
beta_z = Cov(z(S), r) = Cov(S, r) / sd(S)
       = a * theta * Var(Z) / sqrt(a^2 * Var(Z) + Var(u))
       = theta * sd(Z) * sqrt(lambda_a),    lambda_a = a^2 Var(Z) / (a^2 Var(Z) + Var(u))
```

Two results follow, both used later:

- **Standardization removes the scale factor.** As `Var(u) -> 0`, `beta_z -> theta * sd(Z)` regardless of `a`. This is the formal reason §7(a) specifies the paired contrast on `z(S)`: without it, doubling a score halves its raw coefficient with no change in information.
- **The attenuation exponent is 1/2, not 1.** The standardized coefficient is attenuated by `sqrt(lambda)`, not `lambda`. Since `lambda <= 1`, `sqrt(lambda) >= lambda`: the shrinkage is **milder** than the textbook formula implies, and differences between scorers are correspondingly **smaller**.

## 4. Aggregation: the assumption the two acts do not share

Act 1 measures classification quality **per headline**. Act 2 regresses on `S_t`, the equal-weighted mean of that session's `n_t` headline scores. These are not the same measurement error.

Write headline `i` on session `t` as `s_(t,i) = z_(t,i) + u_(t,i)`, so

```text
S_t = Zbar_t + ubar_t,     ubar_t = (1/n_t) * sum_i u_(t,i)
```

and let the outcome depend on the aggregate, `r_(t+1) = theta * Zbar_t + eps`.

**If per-headline errors are independent within a session** (assumption **A5**):

```text
Var(ubar_t) = Var(u) / n_t
lambda_daily = Var(Zbar) / ( Var(Zbar) + Var(u)/n_t )   ->  1  as n_t grows
```

`Var(Zbar)` — the day-to-day variation in average news tone — does **not** shrink with `n_t`; it is a real quantity. Only the error term shrinks. The census puts `n_t` at a median of **337**.

**The magnitude, which is the point of this section.** Take `Var(Zbar) = 0.01` and compare a scorer with per-headline `Var(u) = 0.25` against a much worse one at `Var(u) = 0.50`:

| | per-headline | daily, `n = 337` | daily + standardized |
|---|---:|---:|---:|
| `lambda`, better scorer | 0.0385 | 0.9309 | `sqrt` = 0.9648 |
| `lambda`, worse scorer | 0.0196 | 0.8709 | `sqrt` = 0.9332 |
| ratio of coefficients | **1.96x** | **1.069x** | **1.034x** |

A **2x difference in per-headline measurement noise becomes a 3.4% difference in the standardized daily coefficient.** Averaging 337 headlines removes most of the independent error for *both* scorers, and the square root from standardization halves what remains in log terms.

This is the quantitative content of P06. With `n_t` in the hundreds, the Act 1 → Act 2 bridge is compressed by the design itself. A macro-F1 gap that is large and real at the sentence level need not survive aggregation, and the paired contrast §7(a) is estimating a difference that this model predicts to be small.

**A5 is the load-bearing assumption and it is not established.** If within-session errors share a correlation `rho`:

```text
Var(ubar_t) = Var(u) * (1 + (n_t - 1) * rho) / n_t  ->  rho * Var(u)   as n_t grows
```

The error floor is `rho * Var(u)`, which averaging cannot remove. At `rho = 0.1` and `n = 337` the effective divisor is about 9.8 rather than 337. Correlated error is the realistic case: a scorer that systematically misreads one class of headline — earnings-guidance formats, negations, sector jargon — makes correlated mistakes on the days such headlines cluster. So aggregation compresses the difference between scorers, but by an unknown factor between 1 and `n_t`.

## 5. Where the classical model fails outright

**Bounded scores make A2 false.** Every scorer maps to `[-1, 1]`. If `Z` is unbounded, `S = Z + u` cannot hold globally: near `+1` the error cannot be positive. So `Cov(Z, u) < 0` by construction near the bounds, and

```text
plim beta_hat = theta * (Var(Z) + Cov(Z, u)) / (Var(Z) + 2*Cov(Z, u) + Var(u))
```

which is not bounded above by `theta`. Negatively correlated error can **amplify** rather than attenuate. The direction is no longer guaranteed, and "attenuation toward zero" stops being a safe reading.

**A common latent `Z` is assumed, not shown.** The derivation compares two scorers by assuming both measure the *same* `Z` with different noise. If FinBERT and Loughran-McDonald measure partially different constructs — one closer to analyst sentiment, one to a legal-risk vocabulary — then `theta` differs between the two equations and their coefficients are not comparable through `lambda` at all.

**`theta = 0` absorbs everything.** If aggregate news tone has no conditional relation to next-session returns, then `plim beta_hat = 0` for every `lambda`. Better measurement of a quantity unrelated to the outcome produces no association. Act 1 improvements matter for Act 2 **only conditional on `theta != 0`**, which is the very thing Act 2 is testing.

**No mapping from macro-F1 to `Var(u)` exists.** `lambda` is a variance ratio on a continuous score. Macro-F1 is a three-class classification metric computed after thresholding. Nothing in this derivation converts one into the other, and this study does not attempt it.

## 6. What the simulation demonstrates

`attenuation_study.py`, seed 20260830. The true `theta` is set by construction in every case, so each result is checked against a known answer rather than an expectation.

| # | Demonstrates | Expected |
|---|---|---|
| 1 | Classical attenuation | `beta_hat -> theta * lambda` |
| 2 | Standardized regressor | `beta_z -> theta * sd(Z) * sqrt(lambda)` |
| 3 | Aggregation compresses scorer differences | 2x noise gap -> ~1.03x coefficient gap at `n = 337` |
| 4 | Correlated within-day error | compression collapses as `rho` rises |
| 5 | Bounded scores | `Cov(Z, u) < 0`; classical formula mispredicts, can amplify |
| 6 | `theta = 0` | `beta_hat = 0` at every noise level |

## 7. What this does and does not establish

**Establishes**, conditional on the stated generating processes:

- Classical attenuation shrinks a coefficient toward zero and cannot flip its sign or create an association.
- Standardization removes the scale factor and changes the attenuation exponent to 1/2.
- Daily aggregation over hundreds of headlines compresses per-headline measurement differences into small coefficient differences, **when** within-day errors are independent.
- Each of the failure modes in §5 breaks the classical prediction in the direction shown.

**Does not establish** anything about this study's data:

- No simulated quantity is an estimate of `theta`, `lambda`, `Var(u)` or any coefficient for SPY or for these scorers. The generating processes are inventions chosen to isolate one mechanism at a time.
- The `n = 337` figure enters only as a plausible aggregation count taken from the census. It is not used to predict what the paired contrast will find.
- **No inference runs from this simulation to the observed market.** If §7(a)'s `delta` comes out positive, that is consistent with attenuation and with several other explanations, and the report says so (§8 of the inference protocol).

**What it changes about how a result is read.** If the paired contrast finds a small `delta`, §4 gives a reason internal to the design — aggregation over hundreds of headlines — that does not require FinBERT and Loughran-McDonald to be similarly good classifiers. A small `delta` is therefore weak evidence about relative classification quality, and Act 1 remains the place that question is answered.
