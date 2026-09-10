# What `L = 5` counts: the HAC spacing decision

Date: 2026-09-10. Increment: **R07c**. Resolves audit finding **A09** and the deferred protocol amendment **M7**.

Status: **decided and implemented**. No sentiment–return coefficient has been estimated, and nothing in this document was informed by one.

Related: [inference protocol](inference-protocol.md) §4 · [project audit](project-audit-2026-09-09.md) A09 · [repair sequence](audit-implementation-plan-2026-09-09.md) · [decision log](research-review-decision-log.md).

---

## 1. The question

The inference protocol prespecifies a Newey–West bandwidth of `L = 5`, justified in words as **one trading week**. Whether the code delivers that depends on something the protocol did not say: what a lag counts.

`nw_ols` resets the index before handing the data to statsmodels. statsmodels then pairs observation `i` with observation `i − ℓ`, so lag `ℓ` means **ℓ rows of the analysis sample**. Wherever the sample has a gap — a zero-news session, a missing SPY row, an unscored session — those rows sit further apart on the exchange calendar than their row distance suggests, and "one trading week" quietly becomes more than one. The installed `statsmodels.stats.sandwich_covariance.cov_hac_simple` documents an assumption of consecutive, equally spaced periods, which a gapped sample violates.

| Convention | Lag `ℓ` means |
|---|---|
| `retained_position` | `ℓ` rows of the analysis sample — what the code did |
| `session_indexed` | `ℓ` exchange sessions; pairs more than `L` sessions apart get zero weight even when they are adjacent rows |

## 2. What was implemented

Both, as **one computation with one differing input**. `inference.hac_sandwich` takes the session index as an argument:

```text
S = Σ_ℓ w_ℓ (Ω_ℓ + Ω_ℓ'),   w_ℓ = 1 − ℓ/(L+1)
Ω_ℓ = Σ over pairs (i, j) whose SESSION distance is exactly ℓ  of  xu_i' xu_j
cov = (X'X)^-1 S (X'X)^-1
```

Passing `arange(n)` makes session distance equal row distance and reproduces the retained-position convention term for term. This is not an assertion: `tests/test_hac_spacing.py` checks that both conventions reproduce statsmodels' own HAC covariance **exactly** (to 1e-12 relative) on a contiguous sample, that the two agree with each other bit for bit when the index is `arange`, and that the gapped case matches lag products written out independently as a double loop over every pair.

Because the two differ only in the pairing rule, any divergence between them is attributable to spacing and to nothing else. That is what makes the measurement below interpretable.

The Bartlett kernel is positive definite on the line, so the irregular-grid form remains a positive semi-definite estimator; `HACFit.bse` checks the realised diagonal rather than relying on that.

## 3. The measurement

`hac_spacing_study.py`, seed 20260830, `L = 5`, 60 replications per cell. Synthetic throughout, with a **true tone coefficient of exactly zero**; only the standard error is compared, so the script cannot be used to see the primary result early.

The generating process is chosen to be the case that separates the conventions: residual dependence is generated in **session time** over the full calendar and then sampled at the retained sessions, so two rows adjacent in the analysis sample but far apart on the calendar are genuinely far less dependent than retained-position pairing assumes.

Reported as `se(session_indexed) / se(retained_position)` on the tone term.

### Gap sweep

| Gap fraction | `rho` resid | n retained | ratio, mean | ratio, min | ratio, max | max abs % diff |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 2516 | 1.000000 | 1.000000 | 1.000000 | **0** |
| 0 | 0.3 | 2516 | 1.000000 | 1.000000 | 1.000000 | **0** |
| 0 | 0.6 | 2516 | 1.000000 | 1.000000 | 1.000000 | **0** |
| 0.0004 | 0.0 | 2515 | 1.00004 | 0.99946 | 1.00274 | 0.27 |
| 0.0004 | 0.3 | 2514 | 1.00000 | 0.99850 | 1.00060 | 0.15 |
| 0.0004 | 0.6 | 2515 | 0.99997 | 0.99910 | 1.00051 | 0.09 |
| 0.01 | 0.0 | 2491 | 0.99974 | 0.99324 | 1.00479 | 0.68 |
| 0.01 | 0.3 | 2489 | 1.00037 | 0.99756 | 1.00567 | 0.57 |
| 0.01 | 0.6 | 2490 | 0.99925 | 0.99571 | 1.00194 | 0.43 |
| 0.05 | 0.0 | 2389 | 1.00083 | 0.99471 | 1.00811 | 0.81 |
| 0.05 | 0.3 | 2389 | 1.00018 | 0.99378 | 1.00692 | 0.69 |
| 0.05 | 0.6 | 2389 | 0.99618 | 0.98633 | 1.00451 | 1.37 |
| 0.15 | 0.0 | 2139 | 0.99984 | 0.98676 | 1.02421 | 2.42 |
| 0.15 | 0.3 | 2137 | 0.99949 | 0.98522 | 1.01219 | 1.48 |
| 0.15 | 0.6 | 2140 | 0.99014 | 0.97211 | 1.00259 | 2.79 |
| 0.30 | 0.0 | 1760 | 1.00198 | 0.97483 | 1.03120 | 3.12 |
| 0.30 | 0.3 | 1757 | 0.99591 | 0.94870 | 1.02299 | 5.13 |
| 0.30 | 0.6 | 1763 | 0.98472 | 0.95573 | 1.01572 | 4.43 |
| 0.50 | 0.0 | 1257 | 1.00446 | 0.95878 | 1.06951 | 6.95 |
| 0.50 | 0.3 | 1255 | 0.99988 | 0.95021 | 1.04154 | 4.98 |
| 0.50 | 0.6 | 1264 | 0.97567 | 0.94230 | 1.03284 | 5.77 |

### The structure this corpus is expected to produce

A 62-session leading warm-up from the trailing 63-session volume detrend (a truncation, not an interior hole, so it contributes nothing), one interior zero-news session, and the final session dropped for having no lead.

| `rho` resid | n retained | ratio, mean | ratio, min | ratio, max | max abs % diff |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 2452 | 1.000020 | 0.998554 | 1.001450 | 0.15 |
| 0.3 | 2452 | 0.999984 | 0.998654 | 1.000660 | 0.13 |
| 0.6 | 2452 | 0.999933 | 0.996723 | 1.000550 | **0.33** |

### What the numbers say

1. **With no gaps the conventions are numerically identical** — ratio exactly 1, difference exactly 0. This is the reduction, confirmed end to end rather than argued.
2. **At this corpus's anticipated gap structure the difference is at most 0.33%** on the tone standard error, and the mean ratio is 1.000 to four decimals. The protocol's expectation that "the two are expected to be close" is confirmed, and the alternative — that R07c would find otherwise and that would itself be the finding — did not occur.
3. **Divergence grows with the gap fraction and with genuine session-time residual dependence**, reaching ~7% at 50% gaps. The direction is informative: where residuals are actually autocorrelated in session time, session-indexed standard errors are *smaller* on average (mean ratio 0.976 at 50% gaps, `rho = 0.6`), because retained-position is crediting dependence between pairs that are not in fact within one week.

Point 3 is why the convention still matters even though point 2 makes it immaterial here. It says the retained-position convention degrades in a specific, predictable direction as a sample becomes gappier — which is the situation a later corpus, a shorter window, or a stricter eligibility rule would create.

## 4. The decision

**`session_indexed` is the primary convention.** `retained_position` is computed on every fit and reported alongside as a sensitivity, never substituted.

The reason is interpretive, not empirical. `L = 5` was prespecified as *one trading week*, and only one of these two conventions makes that sentence true. The retained-position convention answers a different question — "five rows of whatever survived eligibility" — which has no prespecified justification behind it and which changes meaning whenever the eligibility rules change. Choosing the convention that matches the bandwidth's own stated rationale is a choice the protocol had already effectively made in §4's wording; R07c is making the code agree with it.

The measurement's role is to bound the exposure, not to pick the winner. Had the two diverged materially at this corpus's gap structure, the divergence itself would have been the reportable finding (protocol §4). It does not: at most 0.33%.

### On the deferral, stated plainly

The inference protocol recorded an exposure when it deferred this: *"the convention will be selected after its effect on the sample has been measured."* Three things bound it, and the first is stronger than the protocol anticipated.

- **The selection was made on interpretive grounds, not on measured outcomes.** The argument in this section would be unchanged if every number in §3 were different. That is a stricter standard than the deferral required.
- **No tone coefficient existed to inform it.** R07c is infrastructure and completes before any scoring run; the study's own regressor has a true coefficient of zero by construction.
- **Both conventions are always reported**, so the choice cannot conceal a divergence.

### What is not settled here

The measurement is synthetic. The protocol's plan also called for measuring both conventions **on the real analysis sample**, and that sample does not exist — no text has been scored and no panel has been built. That measurement is therefore a **required step at R13b**: `PrimaryFit.spacing_comparison()` produces it for every fit, and the realised ratio goes into the run manifest and the report alongside the primary interval. If the real sample's gap structure turns out to be worse than §3's anticipated case, the comparison will show it.

This is a weaker claim than "the conventions agree on this study's data", and it is deliberately not written as the stronger one.

## 5. Where this lives in the code

| Piece | Location |
|---|---|
| The estimator, spacing rule as an argument | `inference.hac_sandwich` |
| Convention names | `inference.HAC_CONVENTIONS` |
| Fit carrying a named convention | `inference.fit_hac`, `inference.HACFit` |
| Both conventions on the primary fit | `inference.primary` → `PrimaryFit.fits`, `.fit`, `.alternate` |
| Side-by-side report | `PrimaryFit.spacing_comparison()` |
| The selected primary | `config.HAC_CONVENTION` |
| Tests | `tests/test_hac_spacing.py` (13), `tests/test_primary.py` spacing section |
| This measurement | `hac_spacing_study.py` |

`nw_ols` is unchanged and still used by the secondary and exploratory paths, which R08 replaces. Those retain the retained-position convention until then; that is recorded here so it is not mistaken for a considered choice.
