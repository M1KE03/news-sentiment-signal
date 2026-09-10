# R07d reporting and R08a return families

2026-09-10. Implements the reporting rules in [inference protocol](inference-protocol.md) sections 5–6 on synthetic inputs. No empirical regression was run.

## Status and prerequisites

The user requested a few tasks after R07c. R07b/R07c interfaces were absent at the start, then appeared in the shared workspace during implementation. This increment preserves that work and integrates with `eligibility`, `primary` and `hac_sandwich`. R07c's decision/measurement record is owned by that concurrent work; this increment estimates no real tone coefficient.

**R08a is implemented and tested:** `lag_family` produces unadjusted estimates; `adjust_return_families` assembles and corrects the frozen family; `run_all.run_analysis` uses that assembly for Table 3 and passes its family labels into Table 4.

**R07d is implemented and tested:** effect reporting, the two-width calculator, a controls-only HAC ratio on the eligible rows, and atomic persistence before the first tone coefficient are integrated into the runner. Return families now use `primary` on the full panel, so each fit standardizes its own eligible sample with detrended-volume controls. Table 4 converts standardized coefficients directly to bps, removing the old overall-sample SD rescaling. The runner's other paths still require R08b/R08c and the later readiness work before empirical use.

## Effect reporting

`classify_effect_interval(lo, hi)` takes finite, ordered endpoints in bps. It reports both association evidence and practical smallness, using closed bounds and endpoints rounded to one decimal place, with NumPy's nearest-even rule for exact ties. Comparisons follow the frozen 2×2 table, including intervals that show a small, nonzero association.

Boundary proximity is evaluated on the original endpoints, within 0.1 bps of zero or either SESOI margin. A boundary row includes both original endpoints formatted to four decimal places and an explicit note. Original numeric endpoints also remain in the effect table. Every interval is labelled `pointwise`; correction of a p-value does not alter its endpoints or make it simultaneous.

`effect_size_table` defaults to already standardized coefficients (`sigma_s=1`). Legacy raw coefficients require the caller to pass the fitted sample's tone SD. `clears_costs` and the old `ruled_out_above_bps` field are removed, including when supplied by an old table. `config.SESOI_BPS=5` names the actual interpretation; the old transaction-cost constant remains only as a compatibility alias.

## Advance precision

`advance_precision(outcome, tone, controls, *, kappa, hac_spacing)` accepts the same complete eligible rows in the same order; Series indices must match. It refuses missing/nonfinite inputs, rank-deficient controls and degenerate residual fits. The exact three controls are `ret`, `rv_parkinson` and `log_volume_detrended`. Standard deviations use `ddof=1`.

The calculator fits outcome on controls and standardized tone on controls separately. It never regresses the outcome on tone. Its JSON-compatible return value includes the UTC computation time, n, outcome/residual SDs, tone-on-controls R-squared, kappa, spacing, bandwidth 5, and h1/h2 in bps, labelled as planning estimates rather than bounds.

The lower-level calculator's `kappa` and `hac_spacing` have **no defaults**. `primary_precision(panel)` supplies them using R07b's exact eligible rows and R07c's covariance helper and configured convention. Both spacing conventions are recorded alongside their h2 values.

**Operational definition fixed before empirical estimation:** kappa is the HAC standard error of the mean of controls-only residuals divided by `sd(u, ddof=1)/sqrt(n)`. HAC uses an intercept-only design, bandwidth 5 and the declared spacing, without a small-sample sandwich correction. This specifies what the protocol's previously underspecified “computed from controls-only residuals” means. It is a residual-dependence planning multiplier, not the eventual tone coefficient's HAC/OLS ratio.

`run_analysis` atomically publishes `report/tables/advance_precision.json` before invoking any tone/outcome fit. A publication failure prevents regression and cleans up the temporary file. The record includes both spacing sensitivities, eligible session dates/positions and the exclusion ledger. Its timestamp records computation; the runner's ordering supplies the advance-sequencing guarantee for this execution. No advance empirical manifest was created in this increment; tests publish only under temporary directories.

## Return family assembly

`adjust_return_families` accepts exactly the 15 `(scorer, horizon)` pairs in `{finbert,lm,vader} × {1,2,3,4,5}`, with a finite p-value in [0,1] for each. Missing, repeated, extra, unknown, or failed tests raise an error. If an `outcome` column is supplied, every value must be `ret`; classifier and RQ4 tests are excluded.

The output has deterministic ordering and carries:

- FinBERT h=1: primary, size 1, original p-value, no BH/BY adjusted value or correction decision.
- Other 14 rows: secondary return family, size 14, BH and BY adjusted p-values and decisions.
- `bh_by_disagree`, `claim_basis` and `claim_reject`: the primary uses its unadjusted test; secondary claims use BY as required when it disagrees with BH.
- `interval_scope=pointwise` and the chosen alpha (default 0.05).

`lag_family` now requires the full panel, delegates eligibility and standardization to `primary`, and carries both HAC standard errors. It no longer accepts a correction threshold or applies per-scorer BH. Callers concatenate all three scorers and use `adjust_return_families(..., q=...)`. The repository runner and figure's coefficient-unit labels are migrated. This is a bounded reporting-function change within the existing module.

## Verification

`tests/test_inference_reporting.py` has 38 synthetic checks. They cover the four conclusion combinations, exact margins, zero boundaries, reporting precision, invalid intervals, effect-unit conversion, orthogonal hand-calculated precision widths, separate-fit designs, tone-unit invariance, and refusal of missing/misaligned rows. Family tests check full BH/BY reference vectors, primary exclusion, input-order invariance, disagreement and malformed family rejection. Integration checks compare controls-only residual HAC against an independent all-pairs sum, verify that precision and the primary use identical rows, verify standardized return-family effects, and force a manifest-publication failure to prove no regression follows it.

R08b's paired covariance and R08c's shift diagnostic remain untouched. The existing block-resampling placebo is still a known defect owned by R08c.
