# Revised research scope: Increment 1

Date: 2026-09-09.

Status: scope draft implementing the direction accepted in P01-P33 of the [research review decision log](research-review-decision-log.md). This increment records the research question, priorities, and limits of the intended conclusions. It does not finalize the validation or inference procedures, change analysis code, or report empirical findings.

Later planning checkpoint: the [implementation plan](implementation-plan.md) now gives the current execution sequence B01-B28 and awaits the user's go-ahead. This scope draft is retained to document the completed first increment; its earlier scheduling table is historical context.

The original implementation plan is preserved as the historical baseline. It still contains assumptions and implementation instructions identified for correction. This draft and the decision log identify the accepted direction; the replacement plan organizes its implementation into bounded steps as detailed specifications are resolved.

## Purpose and central question

Working title: **Financial headline sentiment: measurement quality and subsequent market associations**.

The project will demonstrate mathematical and statistical understanding through a carefully defined measurement, justified comparisons, temporal data integrity, and honest uncertainty. Its contribution does not depend on discovering a profitable trading signal.

> How do financial sentiment measurements differ in classification quality, and what additional information do they provide about subsequent market outcomes?

The initial scope is a retrospective measurement and inference study. Chronological forecast evaluation remains an optional later extension if claims about performance on unseen future periods are desired. Accepting that conditional proposal does not add a forecasting or trading experiment to the initial deliverable.

## The two acts and their priorities

Original research-question IDs are retained so that later edits can be traced to the previous plan.

| Question | Revised role | Initial scope |
|---|---|---|
| RQ1: Classification quality | Core measurement question | Compare FinBERT, LM, and VADER on evaluation examples whose independence from model training is established. The primary classification contrast is FinBERT minus LM macro-F1; other comparisons are explicitly labeled. |
| RQ2: Same-day return association | Secondary context | Describe contemporaneous association only when timestamp provenance supports it. Its presence is not required for the pipeline to be correct. Omit it if the date-only fallback is used. |
| RQ3: Subsequent return association | Primary market question | Estimate the association between FinBERT daily tone and the next trading day's SPY log return, conditional on a prespecified control set. Additional scorers and horizons are secondary. |
| RQ4: Subsequent volume and range variance | Exploratory extension | Assess whether these outcomes add enough value after the core study is complete. Define hypotheses and testing families before looking at their results. |

Act 1 evaluates classification of financial tone. Act 2 studies the association of aggregated measurements with market outcomes. The first does not establish the answer to the second.

## What is being measured and estimated

**Classification:** use macro-F1 as the headline metric, with uncertainty for the paired difference between classifiers. Report accuracy separately; a McNemar result concerns paired correctness, not the macro-F1 difference. The validation source, sampling, annotation, calibration split, and dependence treatment are the next increment's specifications. PhraseBank is supplementary unless independence from the exact checkpoint's training examples can be established.

**Daily tone:** interpret the aggregate as average model-assigned financial tone in the chosen news collection. Do not assume it represents all investors or the entire market. The universe, aggregation details, coverage requirements, and stable sample window depend on the data audit. The equal-headline mean is the existing starting method, not proof of representativeness.

**Primary market association:** estimate the coefficient on FinBERT daily tone in a prespecified next-day regression, reporting basis points per standard deviation and an appropriate interval. The intended primary null is that this coefficient is zero, with a two-sided alternative. Exact controls, analysis-sample rules, standardization, and the treatment of secondary tests will be specified before outcome analysis in Increment 3.

For a session t, next-day return means the log price change from that session's close to the next trading session's close. A later horizon r_(t+h), if retained, is one day's return h trading days ahead; it is not a cumulative h-day return. The price adjustment convention and alignment details will be recorded in the protocol before estimation.

News eligibility must use documented publication availability and actual exchange session closes. All market lags and leads must refer to the complete trading calendar before analysis exclusions. These are required behaviors, not claims about what the current scaffold already implements.

## Mathematical argument and statistical direction

Classical measurement error is a motivation to examine, not an established explanation for differences between the scorers. Its attenuation formula requires assumptions about the latent variable, scale, and error covariances that classifier macro-F1 does not establish. A higher classification score need not produce a larger return coefficient, and raw coefficient size alone cannot compare measurements on different scales.

The accepted direction is to use justified HAC inference for the regressions and a timing shuffle as a descriptive diagnostic. A formal additional resampling test is conditional on a defensible procedure for temporal dependence and the control variables. No procedure will be described as assumption-free.

Primary and secondary claims will be distinguished explicitly. Testing families, dependence assumptions, and pointwise versus simultaneous uncertainty will be resolved in the inference increment. A comparison of scorer effects requires uncertainty for their paired difference if superiority is claimed.

A focused mathematical appendix and one controlled simulation will connect the methods to the application objective. Attenuation and its failure under changed assumptions, or interval coverage under temporal dependence, are suitable choices. The specific demonstration remains to be selected; this scope does not require every suggested simulation.

## Permitted conclusions

The report may find evidence of an association, evidence that an effect is sufficiently small relative to a prespecified threshold, or an interval too wide to distinguish those cases. The smallest effect of interest and attainable precision must be addressed before promising a useful exclusion bound.

The study will not infer the following from its initial design:

- Independent classifier generalization from a benchmark with unresolved training overlap.
- A causal division of classifier gains into vocabulary and context contributions.
- Causation from contemporaneous or subsequent regression associations.
- Out-of-sample forecasting performance from in-sample regression estimates.
- Strategy profitability from a coefficient exceeding an illustrative trading cost.
- Investor disagreement directly from dispersion across headlines about different events.
- A failure of the pipeline merely because the expected positive association is absent.

Directions and methods can be specified in advance; results remain open. Figure titles and captions must follow the evidence rather than state predetermined outcomes.

## Smallest complete deliverable

The first complete version should contain:

1. A documented data and provenance audit, including news composition, timestamps, coverage, and sample selection.
2. Independent classification evaluation with interpretable baselines, class-level diagnostics, and paired uncertainty for the primary comparison.
3. The primary next-day market regression, its exact analysis sample, standardized effect, justified uncertainty, and precision limitations.
4. Focused checks for the temporal and data-integrity failures identified in the review.
5. One mathematical derivation and simulation, with stated assumptions and a clear interpretation.
6. A reproducible route to the selected outputs, a short overview, and accessible technical details.

Additional horizons, volume/variance analyses, a single-name check, a dashboard, and chronological forecasting will not delay completion of that core. They remain later scope decisions under the accepted conditional recommendations. No new framework, major module, core API, or data-model redesign is authorized by this draft.

## Specifications scheduled for later increments

These are details to resolve within the accepted proposals, not requests to accept P01-P33 again.

| Increment | Specifications to resolve | Outcome |
|---|---|---|
| 2: Validation protocol | Primary evaluation source; annotation rubric and workload; class balance; calibration/evaluation separation; pairing and cluster units | A concrete, defensible evaluation procedure |
| 3: Inference protocol | Exact controls; primary and secondary families; HAC and diagnostic details; standardized comparisons; precision and practical-effect criterion | Hypotheses and inference rules that can be implemented and checked |
| 4: Bounded data audit | Publication versus update/collection timestamps; source relevance; coverage and concentration; candidate window and scale | Verified facts and documented unknowns before large scoring runs |
| 5: Timing contract | Session closes; sample edges; missing rows; date-only mapping; affected interfaces | Expected behavior and any concrete architecture proposal before implementation |
| 6: First isolated timing correction | Previous-trading-day controls despite zero-news exclusions | A narrow patch and a meaningful regression test |

If the data audit invalidates an earlier draft choice, record the evidence and revision before analyzing outcomes. Do not choose a replacement based on which version produces significance.

## Increment 1 completion and boundary

Completed: recorded acceptance of P01-P33 and created this revised scope draft. The existing implementation plan, analysis code, data, and outputs are unchanged. The next suggested increment is the validation protocol.

Work stops here for explicit instructions. Any architecture change requires its own concrete explanation and approval before implementation. Version-control operations remain read-only for the assistant.
