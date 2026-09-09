# Inference protocol (Act 2)

Date: 2026-09-09. Increment: **B02**.

Status: **specification**. No model has been fitted and no result has been seen. Everything here is fixed before estimation. Items whose resolution genuinely requires the data are listed in §12 as feasibility checks with a stated decision rule, rather than silently assumed.

Related: [implementation plan](implementation-plan.md) · [decision log](research-review-decision-log.md) (P06–P10, P12, P15, P18–P23) · [validation protocol](validation-protocol.md) · [scope draft](protocol-revision-draft.md).

The rule this document exists to enforce: **every claim the report makes has a named estimand and a named uncertainty procedure for that same estimand.** Where those two do not match — a p-value about one quantity attached to a statement about another — the claim is not made.

---

## 1. The primary claim

Exactly one primary claim. Everything else is secondary or exploratory and is labelled as such.

**Estimand.** `beta`, the coefficient on standardized FinBERT daily tone in

```text
r_(t+1) = alpha + beta * z(S_t) + gamma' X_t + e_(t+1)
```

**Null and alternative.** `H0: beta = 0` against a two-sided alternative, at the 5% level. No correction is applied to the primary claim, because the primary family has one member (§6).

**Interpretation.** An **in-sample conditional association** between tone measured at session `t` and the return realised over the next session, given `X_t`. It is not a forecast, not a causal effect, and not a strategy return (P15). The report uses the word "association" and does not use "predicts" for this quantity.

### Variable definitions

| Symbol | Definition | Notes |
|---|---|---|
| `r_(t+1)` | `log(P_(t+1) / P_t)`, adjusted closes of SPY, from the close of session `t` to the close of the **next exchange session** | A single day's return, not a cumulative multi-day return (P15) |
| `S_t` | Equal-weighted mean FinBERT tone over headlines assigned to session `t` | `P(pos) − P(neg)`. "Average model-assigned tone in this news collection", not market sentiment (P20) |
| `z(S_t)` | `(S_t − mean(S)) / sd(S)` over the analysis sample | See §3 |
| `r_t` | Session `t` log return | Momentum/reversal control |
| `RV_t` | `(log(H_t / L_t))^2 / (4 log 2)` — Parkinson **intraday range-based variance proxy** | Not "volatility"; it captures range, not total daily variation (P22) |
| `lv_t` | `log(volume_t)` minus its **trailing** 63-session mean ending at `t` | **Log trading volume**, not turnover — there is no share-count denominator (P22). Trailing only; a centred window would leak |
| `X_t` | `{ r_t, RV_t, lv_t }` | The full control set. Frozen here (§2) |

`z(S_t)` uses the analysis sample's own mean and standard deviation. This is a full-sample transformation, which is legitimate for an association study and would not be for a forecasting claim — one more reason the claim in §1 is stated as association. The choice is recorded here so it is not mistaken for a forecasting design later.

## 2. The control set is frozen now

`X_t = { r_t, RV_t, lv_t }` — return, range variance, log volume, all observable at the close of session `t`. Three controls: momentum/reversal, volatility state, attention.

This set, and the transformations above, are **fixed before any sentiment–return coefficient is examined**. Adding, dropping, or re-transforming a control after seeing `beta` is a specification search and is not available. If a data-integrity problem forces a change (for example the 63-session detrend window proves unusable, §12), the change is made, dated, and logged with a statement of what had already been seen (P25).

A deliberately short control set is not an oversight. A larger one would add researcher degrees of freedom without adding identification, since nothing here identifies a causal channel regardless of how many controls are included.

## 3. Analysis sample and the calendar contract

The eligibility rules below decide which sessions enter the primary regression. They are stated before estimation because the sample is part of the specification.

A session `t` is **eligible** when all of:

1. `t` and the next exchange session both lie inside the locked window **[B04]**, with SPY prices present for both;
2. `n_t >= 1` — at least one headline assigned to `t` (zero-news sessions are excluded, and the excluded count is reported);
3. `r_t`, `RV_t` and `lv_t` are all defined, which requires 63 prior sessions of volume history.

Three requirements on how the panel is built, which exist because getting them wrong changes what `r_(t+1)` means without changing any code that looks wrong (P12):

- **Lags and leads are constructed on the complete exchange calendar, before any exclusion.** If Tuesday has no news, Wednesday's control `r_t` is still Tuesday's return, and Monday's `r_(t+1)` is still Tuesday's return. Excluding a row must never re-point a neighbour's lag at a different session.
- **Adjacency is asserted, not assumed.** For every retained observation, the code asserts that the session supplying `r_(t+1)` is the immediate next session in the exchange calendar. A missing price row must fail this assertion rather than silently turn a one-day lead into a two-day one.
- **The window's edges are enforced.** News before the window's first session cannot accumulate into that session, and the last session has no lead and is dropped.

Reported for the primary specification, always: eligible sessions, sessions dropped for each of the three reasons above, and the final `n` used by the regression.

The exact treatment of session closes, date-only fallback mapping, and missing market rows is the **timing contract, B05**, and is implemented at B06–B09. This protocol states what those must deliver; it does not restate their internals.

## 4. Uncertainty for the primary claim

**Estimator.** OLS. **Standard errors.** Newey–West HAC, **bandwidth `L = 5` prespecified**, reported as the primary number regardless of what other bandwidths give.

Justification, and its limits. The one-day horizon is non-overlapping, so the leading source of residual autocorrelation present in overlapping-return regressions is absent here. HAC is used anyway because the regressor is persistent and return residuals are heteroskedastic and mildly dependent, and because a bandwidth choice made *after* seeing which one produces significance is not a choice at all. `L = 5` is one trading week and is carried over from the original specification, so it is not a fresh degree of freedom.

**Sensitivity, reported alongside and never substituted for the primary:** `L ∈ {0 (heteroskedasticity-robust only), 1, 10}` and the Newey–West (1994) data-driven plug-in bandwidth. Divergence across these is reported as a finding about the inference's fragility, not resolved by picking one.

**Diagnostics, reported, not acted upon:** residual autocorrelation function to lag 20, and the regressor's own ACF. These describe whether `L = 5` was a reasonable prespecification. They do not license re-choosing `L` after the fact.

**Intervals are pointwise.** Every reported 95% interval is a pointwise interval for a single coefficient and is labelled with that word. A BH-adjusted p-value elsewhere in the report does not convert any interval into a simultaneous one (P10). If a joint statement across several horizons is ever required, it uses sup-t simultaneous bands simulated from the HAC covariance of the coefficient vector, and says so explicitly.

## 5. Effect scale, the smallest effect of interest, and the three permitted conclusions

**Scale.** Because the regressor is standardized, `beta` is already "log return per 1 standard deviation of tone". Reported as basis points:

```text
bps per 1 SD      = beta * 10,000
95% pointwise CI  = (beta -/+ 1.96 * se_HAC) * 10,000
```

The interval is transformed identically to the point estimate.

**Smallest effect of interest (SESOI): 5 bps per 1 SD, prespecified.**

Rationale, and its boundaries. 5 bps is the order of magnitude commonly quoted for one-way execution cost in a large liquid ETF. It is used here **as a yardstick for what counts as small** — a daily conditional association below this is small relative to frictions any user of it would face. It is **not** a profitability threshold. A coefficient above 5 bps does not establish that a strategy makes money, and one below it does not establish that the information is useless: realised economic value depends on signal use, timing, turnover, holding period and capacity, none of which this design measures (P19).

**The three conclusions, and the rule that picks between them** (P18). Let `CI` be the pointwise 95% interval in bps.

| Conclusion | Condition |
|---|---|
| **Evidence of association** | `0` is outside `CI` |
| **Evidence the effect is small** | `CI` lies entirely inside `(−5, +5)` bps |
| **Inconclusive** | `0` is inside `CI` **and** `CI` extends beyond `±5` bps |

An interval crossing zero is not by itself evidence of absence. If the third row obtains, the report says the study could not distinguish a null from an economically meaningful effect, and gives the interval. That is a legitimate outcome and is written as one.

**Precision is assessed before `beta` is looked at.** Once the analysis sample exists but before the sentiment coefficient is examined, compute the anticipated half-width from the return residual scale and sample size:

```text
half-width ≈ 1.96 * sd(r) / sqrt(n)      (bps, standardized regressor, ignoring controls and dependence)
```

If that half-width already exceeds the 5 bps SESOI, the study **cannot** deliver the "effect is small" conclusion no matter what is estimated. That fact is recorded at that point, in advance, so a wide interval is reported as a known precision limit rather than discovered afterwards and framed as a finding.

## 6. Testing families

Family membership is fixed here. Nothing moves between families after results are seen.

**Primary — one test.** FinBERT, `h = 1`. No multiplicity correction.

**Secondary return family — 14 tests.** All remaining (scorer, horizon) pairs: `{finbert, lm, vader} × {1..5}` minus the primary. Correction: **Benjamini–Hochberg at `q = 0.05`** across all 14.

BH's guarantee holds under independence and under positive regression dependence. These tests share regressors and outcomes and are positively correlated by construction, which makes PRDS plausible but not established (P09). So **Benjamini–Yekutieli is reported alongside** as a sensitivity: it is valid under arbitrary dependence at the cost of power. Where BH and BY disagree, both are shown and the claim is made at the BY level.

`r_(t+h)` is the single-day return over session `h` steps ahead, not the cumulative `h`-day return.

**Exploratory family — volume and range variance (RQ4).** Per outcome, per scorer:

```text
RV_(t+1) = alpha + b1 z(S_t) + b2 z(|S_t|) + b3 z(d_t) + phi RV_t + e
lv_(t+1) = alpha + b1 z(S_t) + b2 z(|S_t|) + b3 z(d_t) + phi lv_t + psi |r_t| + e
```

**The primary test for each is a HAC Wald test of the joint null `b1 = b2 = b3 = 0`** — this resolves the question the plan left open. Individual coefficients, `b3` included, are then reported **descriptively** with pointwise intervals; a single one is not promoted to the headline after inspection. BH within this family of Wald tests.

`d_t` is the **within-day standard deviation of model-assigned headline tone**, defined only when `n_t >= 5`. It is not investor disagreement: these are headlines about different events, scored by a model, not observations of belief dispersion (P21). Its relation to volume is an extension to test, not a replication of Tetlock (2007), which links unusually high or low pessimism — not dispersion — to volume. Note also that bounded scores tie the mean and dispersion mechanically,

```text
d_t^2 <= [ n_t / (n_t - 1) ] * ( 1 - S_t^2 )
```

so `b2` and `b3` cannot be read as independent dimensions. The `n_t >= 5` requirement shrinks this sample; the reduction is reported.

**Same-day association (RQ2) — secondary context, conditional.** Admissible only if B03 establishes that timestamps reflect publication availability rather than update or collection time. If admissible: `r_t` on `z(S_t)` with lagged controls, reported descriptively as a contemporaneous association, in no correction family, with no causal direction claimed — prices moving the news is as consistent with a positive coefficient as the reverse. **Its absence is not evidence that the pipeline is broken** (P23); a weak contemporaneous association can equally reflect aggregation, the chosen universe, or measurement noise, and it is investigated as such rather than by adjusting the pipeline until it appears.

**Robustness exhibits and the classification agreement subsets are sensitivity analyses, not additional tests.** They do not enter any family and cannot supply a significant result that the primary specification did not.

## 7. Comparing scorers

Two different questions, two different estimands. The report keeps them apart.

**(a) Difference of marginal associations.** `delta = beta_finbert − beta_lm`, each from its own regression, both standardized, both fitted on the **identical observation set**.

A claim that one exceeds the other requires an interval for `delta` itself. Two separate t-statistics, or significance for one scorer and not the other, do not test the difference (P07). `delta` is estimated by stacking the two equations into one system and computing a HAC covariance over the full parameter vector, so the cross-equation covariance — which is large here, since the two scores are correlated — enters the standard error. Wald test on `delta = 0`, pointwise interval reported in bps per 1 SD.

**(b) Incremental contribution.** Both standardized scores in one regression. This asks whether one adds information given the other. It is a **different estimand** from (a) and is labelled as such.

**Collinearity is a diagnostic, not a decision rule.** Report `corr(z(S_finbert), z(S_lm))` and the VIFs. A high correlation widens the interval on `delta`, and that widening is the honest statement of what the data can distinguish. It does not trigger a rule that declares the comparison inconclusive at some threshold — the interval already says that, quantitatively.

Standardization matters here and is the reason (a) is specified on `z(S)`: replacing `S` with `2S` halves its raw coefficient without changing its information content, so raw coefficient magnitudes are not comparable across scorers. A shared `[−1, 1]` range is not a shared scale.

## 8. The attenuation argument is a hypothesis, not a mechanism

Under a classical measurement-error model with `Z` latent, centred variables, and the usual zero-covariance assumptions,

```text
r_(t+1) = theta Z_t + e_(t+1)
S_t     = Z_t + u_t
plim(beta_hat) = theta * Var(Z) / ( Var(Z) + Var(u) )
```

so a noisier measurement attenuates the coefficient toward zero.

This study has **not** established that the three scorers measure a common latent `Z` on a common scale, that their errors are classical (bounded scores with a nonlinear map to any latent quantity are not), or that sentence-level macro-F1 maps onto `Var(u)` for a *daily aggregate*. Better measurement of a quantity unrelated to returns produces no signal at all. So the relation between Act 1 and Act 2 is stated as a **conditional hypothesis with its assumptions on display**, examined in the derivation and simulation at B21 — never as an implication that a better classifier must produce a larger coefficient (P06).

Concretely: if `delta` from §7(a) is positive, that is *consistent with* attenuation and also with several other explanations, and the report says so.

## 9. The timing diagnostic

Demoted from "placebo test" to what it can actually support (P08).

**Procedure.** Circular **shift** of the standardized tone series by `k` sessions, with outcomes and all controls left in place, refitting the primary specification for every `k` in `1 .. n−1`:

```text
z(S)_shifted[i] = z(S)[(i + k) mod n]
```

**Output.** The distribution of the shifted coefficient across `k`, and the rank of the observed `k = 0` value within it, reported as a percentile.

**Why a full circular shift and not block resampling.** A circular shift is a bijection: every observation appears exactly once, and the tone series' entire autocorrelation structure is preserved exactly. Drawing blocks with replacement — what the current `src/inference.py` does — duplicates some observations and omits others, so it is not a permutation, and the `+1` correction on its p-value does not repair a null distribution built the wrong way (B19 fixes this).

**What it is not, stated in the report next to the number.** Shifting tone also destroys its relationship with the controls, so the resulting distribution is not the null distribution of the *conditional* coefficient. It is therefore reported as a **descriptive timing diagnostic** and as a **percentile rank**, never as a p-value for `H0: beta = 0`, and never as "assumption-free". The regression's inferential statement comes from the HAC interval in §4 and from nowhere else.

A formal resampling test remains optional and conditional on specifying a procedure that handles the controls and the temporal dependence defensibly, and on checking its behaviour on synthetic data with a known answer. Absent that, it is not run.

## 10. Naming corrections carried into the schema

Enforced when the panel contract is revised at B16. Names must match formulas (P22).

| Current | Corrected | Reason |
|---|---|---|
| `log_turnover` | `log_volume` | Log share volume; no denominator makes it turnover |
| `log_turnover_detrended` | `log_volume_detrended` | Same, trailing 63-session mean removed |
| `parkinson` described as "volatility" | `rv_parkinson`, "intraday range-based variance proxy" | It is a range estimator of variance, not total daily volatility |
| `d_t` described as "disagreement" | "dispersion of model-assigned headline tone" | See §6 |

## 11. Claims this protocol forbids

- Calling any §1 or §6 result a forecast, a prediction of future performance, or a causal effect.
- Reading `r_(t+h)` as a cumulative `h`-day return.
- Attaching the timing diagnostic's percentile to `H0: beta = 0` as a p-value.
- Describing any procedure here as assumption-free.
- Presenting a pointwise interval as a simultaneous one, or as ruling out effects across a family.
- Concluding an effect is absent because an interval contains zero, without the SESOI comparison in §5.
- Claiming or denying strategy profitability from a regression coefficient and a cost figure.
- Claiming one scorer beats another without an interval for their paired difference.
- Reading `d_t` as investor disagreement.
- Treating a weak same-day association as proof of a pipeline defect.
- Changing the control set, sample rules, families, bandwidth, or SESOI after seeing a coefficient, without a dated log entry recording what had been seen.

## 12. Feasibility checks that depend on the data

Identified in advance, each with the decision it drives. None is assumed to pass.

| Check | When | Decision it drives |
|---|---|---|
| Timestamps reflect publication availability, not update or collection time | B03 | Whether RQ2 is admissible at all (§6); whether the date-only fallback applies |
| Anticipated half-width vs. the 5 bps SESOI | B25, before inspecting `beta` | Whether "effect is small" is attainable; recorded in advance either way |
| Zero-news session frequency | B25 | Sample-size reporting; whether adjacency assertions fire often enough to matter |
| Volume series admits a 63-session trailing detrend (no regime break, no stale-zero rows) | B03/B25 | Whether `lv_t` needs a different window; a change here is logged with what had been seen |
| Coverage stability across the window | B03 | Window selection at B04 — decided before any return relationship is examined |
| Fraction of sessions with `n_t >= 5` | B25 | Whether the RQ4 dispersion specification has usable support |
| Residual and regressor ACFs | B25 | Reported as diagnostics for the `L = 5` prespecification; do **not** license re-choosing `L` |
| `corr(z(S_finbert), z(S_lm))` and VIFs | B25 | Reported alongside the §7(a) interval as a diagnostic; does not gate the comparison |

## 13. Open items for later increments

| Item | Resolved at |
|---|---|
| Session-close, sample-edge and missing-row contract | B05, implemented B06–B09 |
| Date-only fallback mapping, if the audit selects it | B09 |
| Panel field renames in §10 | B16 |
| Primary regression implementation and its fixture tests | B17 |
| Secondary family correction and the stacked `delta` comparison | B18 |
| Timing-diagnostic reimplementation | B19 |
| Standardized effects, precision report, three-conclusion logic | B20 |
| Attenuation derivation and simulation | B21 |
