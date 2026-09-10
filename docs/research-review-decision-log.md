# Research review and decision log

Recorded: 2026-09-09.

Last updated: 2026-09-10, dictionary acquisition and pilot handoff. R01d and later repairs are recorded below; the old PyTorch blocker is resolved.

## Dependency resolution — dictionary and annotation pilot, 2026-09-10

**Authorization:** user requested “please fix the blocked by side” and answered “I will label the pilot.”

**Completed:** downloaded the CSV linked by the official Notre Dame SRAF page, validated it, placed the unmodified bytes at `config.LM_DICT_PATH`, and recorded the 1993–2025/March 2026 release plus SHA-256. Dictionary: 86,553 entries; 347 positive and 2,345 negative terms. The existing scorer passed a synthetic positive/negative/no-hit check. [Full acquisition provenance](lm-dictionary-provenance.md). This release is retrospective and was not selected using sentiment–return results.

**Pilot handoff:** prepared `data/annotation/pilot_worksheet.csv` and instructions from existing frozen IDs, preserving blind presentation order. Verified all 60 IDs are unique calibration items; labels and flags remain blank. No draw/split was repeated and no annotation headline was scored. The user committed to this pilot, not yet to full annotation or a second annotator.

**Verification:** suite after acquisition/configuration update: **246 passed, 0 skipped**. Rechecked dictionary SHA-256, worksheet order/calibration membership, blank response fields and new documentation links. Diff whitespace check passed. Additional source edits appeared in the shared working tree during this task; this dependency update did not modify those analysis modules.

**Corrected dependency statement:** R06b/B15 implementation does not require real labels; synthetic paired fixtures can verify it. R13a requires actual independent labels and the implemented validation path. R11's dictionary prerequisite is resolved, but readiness/preceding repairs and a measured pilot remain. No full scoring, real regression, environment changes or Git writes were performed.

## Audit repair checkpoint — R01c

**Authorization:** user requested "do r01c". This is the contract-proposal increment, not its implementation.

**Specification:** wrote [scoring-checkpoint-contract.md](scoring-checkpoint-contract.md). Proposed one Parquet artifact containing versioned embedded metadata, one same-directory replacement commit point, a process-held writer lock, consistent single-open reads, explicit failure/recovery states, legacy refusal without automatic migration, and the narrow runner reader change needed to enforce the new contract. Preserves public score columns and cache path. Defines R01d interruption and process-contention tests; explicitly excludes power-loss and disk-corruption recovery guarantees.

**Verification:** inspected existing writers/readers; a disposable one-row probe confirmed metadata embedding and pandas/Arrow round-trip compatibility on installed PyArrow 25.0.1. This does not verify crash behavior or the declared environment. Documentation links and diff whitespace checked. No production code, empirical data, environment or Git writes changed.

**Decision pending:** approval of this concrete contract under implementation-plan §5. R01d is next after approval; it may split format/reader implementation from locking and fault-injection verification to preserve bounded increments.

## Audit repair checkpoint — R01b

**Authorization:** user requested "do the next", continuing the [audit implementation plan](audit-implementation-plan-2026-09-09.md).

**Implementation:** default completed-cache calls compare all configured identities before reuse, with no FinBERT model loading. LM/VADER identity still requires their local lexicons. All selected identities are preflighted before any scorer writes. Fingerprint changes under explicit rescore invalidate the full cached column, including unrequested rows. Raw UTF-8 headline text is hashed into a new cache-only `text_sha256` column; changed text invalidates all scorers for affected IDs under explicit rescore. Missing legacy text identity refuses reuse and requires a rebuild to a new cache path. Public returned score columns are unchanged. Duplicate/null IDs are rejected to prevent ambiguous cache joins.

**Verification:** eight regression cases added. Full suite **141 passed, 7 skipped** (Windows PyTorch DLL block); whitespace diff check passed. Tests include interrupted subset resumption and a default completed-cache path that refuses to load model weights. No substantive scoring, corpus changes, environment changes or Git writes.

**Remaining:** next is R01c's concrete checkpoint consistency proposal, then R01d implementation. Atomic consistency between parquet and JSON is still unresolved; successful batch resumption tests do not establish safety between those writes. The analysis runner's own provenance gate and broader measurement fingerprint completeness remain later work. Stop at this reviewable increment.

## Audit repair checkpoint — R01a

**Authorization:** user requested, "please follow the plan now to make some changes". Started the concrete first increment in the [audit implementation plan](audit-implementation-plan-2026-09-09.md).

**Implementation:** `src/scoring.py` canonicalizes fingerprints through JSON before comparing, records the revision passed to the FinBERT constructor, rejects populated cache columns without a nonempty recorded identity, and requires an identity before a scorer can write scores. Missing provenance is checked before default scorer construction. Restore original metadata or rebuild to a new cache path; unknown values are never assigned the current scorer's identity.

**Verification:** regression cases reproduced the old failures before repair. Added 13 cases in `tests/test_scoring.py`; full suite **133 passed, 7 skipped** (PyTorch DLL blocked). Constructor tests use fake imports and exercise revision forwarding without downloads. Existing identity-change and checkpoint tests continue to pass. Diff whitespace check passed.

**Remaining:** default-path comparison with current artifacts, input-content identity and subset invalidation are R01b; checkpoint data/metadata transaction safety is R01c/R01d. Earlier statements that B13/B14 were complete must be read with these reopened audit findings. No empirical analysis, corpus changes, or Git writes. Stop at this reviewable increment.

Purpose: preserve the previous project structure, all changes proposed in the research review, the reasons for those proposals, and the decisions actually made. The project is intended to demonstrate mathematical and statistical understanding in a data science master's application portfolio.

The historical baseline is preserved in [implementation-plan-original.md](implementation-plan-original.md). The [replacement implementation plan](implementation-plan.md) organizes the accepted changes into bounded increments; the user gave the go-ahead and B01-B09 are complete, Phase B is closed, the corpus has been assembled and censused, D4 is frozen, and B12-B14 are complete. B22 is blocked on the environment. B06 is the first code change to the analysis path. This log preserves the prior design, recommendations, and actual decisions. Findings about the scaffold came from reading the code, not from running the empirical study. No empirical results were available to assess.

## Decision status

The user initially approved creating this document and subsequently **accepted all proposals P01-P33**, stating: "i accept all your proposals, now lets start building it slowly". The earlier pending status is preserved in the decision history. Implementation proceeds in short increments, with a stop for explicit instructions after each increment.

| Item | Decision | Implementation status | Reason or evidence |
|---|---|---|---|
| DOC-01: Create a document in `docs/` recording the previous structure, proposals, decisions, and reasoning | Accepted | This document created | Explicit user request following the review |
| P01-P33: Research, implementation, and presentation proposals below | Accepted on 2026-09-09 | Initial scope documented; methods and code fixes remain unimplemented | Explicit blanket acceptance after reviewing the proposals and incremental plan; no additional proposal-specific rationale supplied |
| DOC-02: Record acceptance and draft the revised research scope | Authorized as Increment 1 | Decision log updated; [scope draft](protocol-revision-draft.md) created | User instructed starting the accepted work slowly, following the proposed incremental sequence |
| DOC-03: Provide an implementation plan in chat and `docs/` | Accepted documentation request | [Plan](implementation-plan.md) written; original reference copy and documentation links maintained | The user explicitly requested the plan before giving the go-ahead to implement; B01-B28 have not started |
| B01: Independent-validation protocol | Authorized and executed | [validation-protocol.md](validation-protocol.md) written; this log updated | User instructed starting the replacement plan in small increments they would validate and commit themselves |
| B02: Inference protocol | Authorized and executed | [inference-protocol.md](inference-protocol.md) written; this log updated | User instructed continuing to B02 and deferred B01's unresolved items until they arise |
| B03: Bounded news audit | Authorized and executed | [data-audit-fnspid.md](data-audit-fnspid.md) written; `data/raw/download.py` rewritten to fetch a bounded range-slice sample; 144 MB of FNSPID downloaded and analysed | User instructed downloading the dataset and running the audit |
| B04: Universe, window, artifact provenance | Authorized and executed | `config.py` records the source, source-domain filter, active date-only fallback, provisional window and pinned artifact revisions; `run_all.py` gained a guard that refuses to build a panel while the fallback is inert | Depends on B03; values correspond to inspected artifacts |
| B12: FinBERT class predictions | Authorized and executed | `src/scoring.py` (`Classifier` protocol, `predict_proba`, `predict`, pinned-label-order check), `src/validate.predictions_for` | Act 1 needs labels; Act 2 needs the continuous score, and one does not determine the other |
| B13: Cache provenance | Authorized and executed | `fingerprint` on every scorer; `scores.parquet.meta.json` sidecar; `IncompatibleCache` | Changing a scorer artifact can no longer silently reuse scores from a different measurement |
| B14: Resumable scoring | Authorized and executed | atomic checkpointed writes in `score_all` | An interrupted pass resumes and matches an uninterrupted one |
| B22: Scoring pilot | **BLOCKED (environment)** | Not run | Smart App Control refuses to load `torch/lib/c10.dll`, so FinBERT cannot be instantiated. VADER measured at 47,060 headlines/s (~0.3 min for the corpus); LM blocked separately on the un-downloaded dictionary |
| Corpus assembly and census | Authorized and executed | `interim/headlines_raw.parquet` (1,412,524) and `interim/headlines.parquet` (869,205); census in [data-audit-fnspid.md](data-audit-fnspid.md) §11; `src/data.near_duplicate_keep`; `src/audit.concentration_profile`, `window_stability`, `corpus_census` | The four measurements the cluster sample withheld |
| D4 sample window | **FROZEN** at 2010-01-01..2019-12-31 | `config.SAMPLE_WINDOW_PROVISIONAL = False`, `SAMPLE_WINDOW_FROZEN_ON = 2026-09-09` | Frozen on coverage evidence alone, before any sentiment-return coefficient was examined. 2,516 sessions; no year below the stability tolerance. The drop-2010 variant is declared in advance as a prespecified sensitivity |
| B09b: RQ2 suppression, universe filter, fallback flag | Authorized and executed | `src/data.py` (`load_news` rewritten with a mandatory source filter, explicit malformed-stamp rejection, chunked reading), `src/inference.py` (contemporaneous refuses to run), `run_all.py` (no Table 2), `config.py` (`DATE_ONLY_FALLBACK_IMPLEMENTED = True`), new `tests/test_data.py` (15 tests) | Completes P14 and the B04 universe decision |
| B09a: Date-only fallback mapper | Authorized and executed | `src/align.py` (`map_date_to_session`, `_reject_intraday`, `aggregate_daily` dispatch), `tests/test_alignment.py` (+16 tests) | Implements the B05 §2 mapping rule; the fallback is still not complete until B09b |
| B07-B08: Calendar edges and mapper guards | Authorized and executed | `src/align.py` (lower-edge guard, optional `prior_close`, date-only rejection), `src/data.py` (`trading_calendar` raises), `src/audit.py` (one scoped opt-out), `tests/test_alignment.py` (+8 tests) | Implements B05 defects D-1 and D-3 and the B07 descope guard |
| B06: Full-calendar lags, leads and adjacency | Authorized and executed | `src/align.py` (left join, lag columns, `assert_sessions_match_calendar`, build stats), `src/inference.py` (`contemporaneous` reads panel lags and refuses a panel without them), `run_all.py` (calendar assertion + missing-session report), new `tests/test_inference.py` (11 tests) | Implements B05 defects D-2 and the P12 lag-after-exclusion defect |
| B05: Timing and missing-data contract | Authorized and executed | [timing-contract.md](timing-contract.md) written; `src/data.py` and `src/align.py` interfaces reviewed; three defects reproduced against the current code | Suggested as the next increment and approved |
| Calendar-handoff architecture gate | **Not opened** | No core signature change required; every defect is fixable inside the current interfaces | `aggregate_daily` already reindexes to the full calendar, so `daily_scores['date']` is the session list and `build_panel` needs no calendar argument |
| P11 / B07 actual session closes | **Descoped for the selected corpus**, requirement retained | Recorded in [timing-contract.md](timing-contract.md) §6 with the trigger that reopens it | The selected corpus is date-only, so no time-of-day is ever compared against a close |
| DEV-01 follow-up | **Resolved** | `src/audit.py` rewritten to profile per source and to withhold corpus-wide rates on a cluster sample; `tests/test_audit.py` rewritten (22 tests); `notebooks/01_data_audit.ipynb` regenerated and executed end to end against the downloaded sample | The B03 findings, not a change of plan |
| DEV-01: Out-of-sequence scaffold work under the superseded plan | Recorded as a deviation; disposition not yet decided | `src/audit.py`, `tests/test_audit.py` created and `src/plots.py`, `notebooks/01_data_audit.ipynb` modified in the working tree; uncommitted | The assistant began Stage 0 audit work from [implementation-plan-original.md](implementation-plan-original.md) before discovering the replacement plan in the working tree. The code assumes the fixed 16:00 ET close that P11/B07 removes, and omits the publication-availability (P13) and source/company-concentration (P20) checks B03 requires. Superseded as written; retain only as raw material for B03 |
| Version-control writes | Prohibited for the assistant under the operating instructions | None performed | Git inspection only; no commits, pushes, branches, tags, or other Git writes in this increment |

Every proposal below is marked **Accepted**. None is recorded as rejected. Acceptance of a conditional recommendation does not make every alternative mandatory: chronological forecasting, formal resampling beyond the descriptive placebo, and other extensions remain conditional as described. Their concrete specifications will be resolved in the scheduled increments. Correcting an existing implementation gap and changing the research design remain distinguished below.

Decision and implementation status are separate. Acceptance does not establish completion or verification. The reasons below explain the recommendations; the user accepted them collectively without giving an additional rationale for each. Increment 1 was documentation only. The later planning checkpoint replaced the implementation-plan document and preserved its historical source; analysis code, data, and results remain unchanged.

Concrete architecture changes still require the separate current-approach/proposed-approach/files explanation and explicit approval specified in the user's operating instructions. General acceptance of P01-P33 does not authorize an unreviewed redesign.

## Previous structure and design

The original project was titled **Reading the News with a Machine: Does FinBERT's Classification Skill Survive as a Market Signal?** Its stated budget was approximately 15-20 hours over one week.

It had two linked acts:

- **Act 1, measurement:** compare VADER, Loughran-McDonald (LM), and `ProsusAI/finbert` on Financial PhraseBank. Macro-F1 was the headline metric, with exact McNemar comparisons. Lexicon thresholds were fitted on a seeded 20% split and evaluated on the other 80%, with FinBERT evaluated on those same sentences.
- **Act 2, market inference:** aggregate daily headline scores and regress SPY outcomes on those measurements. RQ2 concerned same-day returns, RQ3 concerned returns on days t+1 through t+5, and RQ4 concerned next-day volatility and volume.

The connecting argument was that a better classifier should have less measurement error and therefore produce a larger, more precisely estimated return coefficient through classical errors-in-variables attenuation. The plan expected FinBERT to win classification, a positive contemporaneous association, weak or null future-return results, and a plausible positive volume/volatility result.

The planned data flow was:

```text
Raw news -> load, deduplicate, timezone-normalize -> headlines.parquet
                                                     |
                                      LM / VADER / FinBERT
                                                     |
                                                scores.parquet

SPY OHLCV and VIX -> market.parquet

headlines + scores + market
          |
          v
timestamp alignment and daily aggregation
          |
          v
daily_panel.parquet -> regressions, tables, figures, report

Independent validation branch:
Financial PhraseBank -> shared scorers -> classification metrics and comparisons
```

The main repository responsibilities were:

| Location | Previous responsibility |
|---|---|
| `config.py` | Locked decisions D1-D16, paths, seeds, model and dictionary settings |
| `src/data.py` | News loading, deduplication, market data, calendar handling |
| `src/scoring.py` | Common scorer interface and score caching |
| `src/align.py` | Timestamp mapping, daily aggregation, panel and lead construction |
| `src/validate.py` | PhraseBank split, threshold fitting, metrics, McNemar comparisons |
| `src/inference.py` | HAC regressions, horizon families, placebo, effect-size calculations |
| `src/plots.py` | Functions for the numbered figures |
| `rescore.py` | Separate expensive scoring pass |
| `run_all.py` | Rebuild the panel and reproduce the analysis outputs |
| `notebooks/01` through `05` | Data audit, validation, signal, volume/volatility, robustness |
| `tests/` | Alignment and scorer checks |
| `report/report.md` | Two-page write-up skeleton |
| `future-work.md` | Out-of-scope ideas and deviations from locked decisions |

The original stages were:

| Stage | Previous work |
|---|---|
| 0 | Scaffold; audit two news candidates; choose source and sample; obtain market and validation data |
| 1 | Implement and run all three scorers; cache scores; inspect approximately 30 qualitative examples |
| 2 | PhraseBank validation, agreement-subset comparisons, Table 1 and Figure 1 |
| 3 | Timestamp alignment, daily panel, exploratory plots, alignment tests |
| 4 | Contemporaneous and lead-return regressions, FDR, placebo, attenuation comparison, effect sizes |
| 5 | Volume and volatility regressions |
| 6 | Median aggregation, HAC bandwidth sensitivity, subperiods, coverage restriction, single-name check |
| 7 | Report, README, figures, and clean-clone reproduction |

The planned inventory included five main tables, four main figures, a qualitative scoring exhibit, and five notebooks. The design excluded model training, long-document analysis, a cross-sectional panel, and a trading-strategy backtest.

Important original settings included a fixed 16:00 ET news cutoff, equal headline weights, dropping zero-news days, dispersion only when at least five headlines were present, HAC bandwidth 5 with 10 as sensitivity, BH correction separately within each scorer's five horizons, and a 1,000-draw circular-block placebo with block length 21. Source, sample boundaries, and model/dictionary versions were still unresolved in the scaffold.

## Research proposals

### P01: Reframe the central question while retaining the two acts

**Decision: Accepted (2026-09-09). Type: Research design and framing.**

**Previous:** The question and narrative assumed that better classification was established and that its measurement advantage should survive as a stronger market coefficient.

**Proposed:** Retain classification validation followed by market analysis, but use the question: “How do financial sentiment measurements differ in classification quality, and what additional information do they provide about subsequent market outcomes?”

**Reason:** This allows classification quality and market informativeness to be tested separately. It supports the application goal without requiring a positive market finding or an unsupported mathematical link. The recommendation is to improve this project, not replace it.

### P02: Establish an independent primary classification evaluation

**Decision: Accepted (2026-09-09). Type: Research design. References: D13; Stage 2.**

**Previous:** Fit lexicon thresholds on 20% of PhraseBank and compare all scorers on the other 80%. Treat FinBERT's possible advantage from shared provenance as a limitation.

**Proposed:** Prefer an independently labeled sample of the actual news collection as the primary evaluation. Use a written annotation rubric, assign labels without seeing model predictions, separate calibration and evaluation examples, and report class balance and uncertainty. Several hundred examples may be a practical starting point, subject to precision. Ideally, a second person independently labels a subset and agreement is reported. Alternatives are an independently sourced benchmark with verified training provenance, or a verified original held-out partition for the exact checkpoint. Keep PhraseBank supplementary unless its evaluation independence is established.

**Reason:** The [official FinBERT model card](https://huggingface.co/ProsusAI/finbert) explicitly identifies PhraseBank as fine-tuning data. A new random split does not remove the checkpoint's previous exposure. This is a direct overlap risk, not merely similar subject matter. Validation on the target news also makes Act 1 more relevant to Act 2. The exact contaminated examples were not established in the review.

### P03: Match classifier uncertainty to the metric being compared

**Decision: Accepted (2026-09-09). Type: Statistical inference. References: D14; Section 6.6.**

**Previous:** Macro-F1 was the headline metric, while the proposed result sentence attached a McNemar p-value to the F1 gap.

**Proposed:** Report a paired bootstrap interval for the macro-F1 difference, keeping both models' predictions on each sampled example together. Retain exact McNemar as a separately labeled paired accuracy comparison. Respect article or duplicate clusters when relevant rather than assuming every sentence is independent.

**Reason:** McNemar applied to paired correctness tests an accuracy difference, not a macro-F1 difference. The uncertainty statement must refer to the statistic actually being interpreted. See the [McNemar documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).

### P04: Handle overlapping agreement subsets consistently

**Decision: Accepted (2026-09-09). Type: Validation protocol. References: D13; Stage 2.**

**Previous:** Repeat the comparison on all four PhraseBank agreement subsets using the stated split procedure.

**Proposed:** If these analyses are retained, assign calibration/evaluation membership consistently by unique sentence and describe agreement subsets as overlapping sensitivity analyses, not independent replications.

**Reason:** The subsets share examples. Independent splitting can give the same sentence different roles across exhibits, and repeated results on overlapping samples do not provide four independent confirmations.

### P05: Remove the causal interpretation of the classifier ladder

**Decision: Accepted (2026-09-09). Type: Interpretation. References: Section 1; Stage 2.**

**Previous:** Attribute VADER-to-LM improvement to vocabulary and LM-to-FinBERT improvement to context; report each step's contribution to the total gain.

**Proposed:** Present three approaches and their performance differences descriptively. Do not assign percentages of the improvement to vocabulary or context without controlled ablations.

**Reason:** Training exposure, architecture, scoring rules, and other properties differ simultaneously. An arithmetic decomposition is not an identified explanation. [VADER itself includes linguistic rules](https://github.com/cjhutto/vaderSentiment), so the starting comparison is not simply general versus domain word counting.

### P06: Replace the asserted attenuation bridge with a conditional hypothesis

**Decision: Accepted (2026-09-09). Type: Mathematical argument. References: Section 1; Section 6.3.**

**Previous:** Higher classification accuracy implies less measurement error and hence a larger, more precisely estimated return coefficient on the same regression specification.

**Proposed:** Explain classical attenuation under its assumptions, then treat its relevance to these scorers as a hypothesis rather than an established implication. Separate financial tone, latent market-relevant information, and future returns.

For a simple centered model,

```text
r_(t+1) = theta * Z_t + epsilon_(t+1)
S_t     = Z_t + u_t
```

with the necessary zero-covariance assumptions,

```text
plim(beta_hat) = theta * Var(Z) / (Var(Z) + Var(u)).
```

**Reason:** The study has not established a common latent variable and scale, classical measurement errors, or a mapping from sentence macro-F1 to daily measurement-error variance. Better measurement of a quantity with no predictive relation to returns need not create a signal. These assumptions are part of what the mathematical discussion should make explicit.

### P07: Compare standardized effects and directly quantify scorer differences

**Decision: Accepted (2026-09-09). Type: Statistical comparison. References: Section 6.3; D15.**

**Previous:** Compare raw coefficient magnitudes and individual t-statistics; near-collinearity above a correlation threshold would trigger an inconclusive interpretation.

**Proposed:** Compare standardized effects on the same observations. If claiming that one scorer's effect exceeds another's, estimate a paired contrast and its uncertainty using a justified procedure. Report correlation and collinearity as diagnostics rather than substituting an arbitrary threshold for the actual uncertainty.

**Reason:** Replacing S with 2S halves its coefficient without changing its information. A common bounded range does not create a common measurement scale. Comparing two individual t-statistics, or finding significance for one scorer but not another, does not test the difference between them.

### P08: Correct the placebo algorithm and its inferential claim

**Decision: Accepted (2026-09-09). Type: Statistical method and implementation. References: D12; Sections 6.6 and 12.**

**Previous:** The plan alternated between circular block permutation and a whole-series circular shift and described an assumption-free null. The code instead draws circular blocks with replacement and substitutes sentiment while keeping regression controls fixed.

**Proposed:** Clearly distinguish those procedures. The simplest recommended route is a descriptive timing placebo with HAC as the stated regression inference. An additional formal resampling test remains an alternative: specify an appropriate regression procedure, justify its treatment of controls and temporal dependence, and check its behavior on synthetic data. Do not call any procedure assumption-free.

**Reason:** Resampling with replacement can duplicate and omit observations; it is not a permutation. Changing sentiment alone also changes its relationship with the controls, so validity for the conditional regression null is not automatic. Stability and dependence assumptions matter. The +1 p-value correction does not fix an inappropriate null distribution.

### P09: Define primary hypotheses and complete testing families

**Decision: Accepted (2026-09-09). Type: Statistical protocol. References: D11; Sections 6.2 and 6.4.**

**Previous:** Apply BH separately to each scorer's five return horizons, while also examining contemporaneous, volume, volatility, agreement-subset, and robustness results. RQ4 includes several sentiment terms without a clearly specified primary test.

**Proposed:** Nominate a primary question, such as FinBERT's next-day return coefficient; define the secondary return family and exploratory volume/volatility family. If the headline claim is that any scorer predicts returns at any horizon, use a family consistent with that search. Specify whether RQ4 tests dispersion alone or a joint null for the sentiment terms. Treat robustness and overlapping validation subsets as sensitivity analyses rather than extra opportunities to select significance.

**Reason:** Separate families can be legitimate, but their guarantees do not automatically extend to the full project. Correlation alone is not sufficient justification for BH; its assumptions must be considered. See this [mathematical treatment of FDR under dependence](https://arxiv.org/abs/2201.09350).

### P10: Distinguish pointwise intervals from joint conclusions

**Decision: Accepted (2026-09-09). Type: Statistical interpretation. References: D15; Figure 2.**

**Previous:** Plot ordinary 95% HAC intervals alongside BH-adjusted p-values and frame the output as ruling out effects across horizons.

**Proposed:** Label those intervals pointwise. Limit exclusion statements to the inference actually constructed; use suitable simultaneous inference if a joint claim about all effects is required.

**Reason:** Adjusting p-values with BH does not transform ordinary confidence intervals into simultaneous intervals.

### P11: Use actual exchange session closes

**Decision: Accepted (2026-09-09). Type: Design correction and implementation. References: D6; Stage 3.**

**Previous:** Set every close to 16:00 ET, explicitly including half-days. Actual closes were parked as future work. `USE_ACTUAL_SESSION_CLOSE` exists but does not activate different behavior.

**Proposed:** Map news against each session's actual close, implement that behavior, and test early closes and daylight-saving transitions.

**Reason:** A 14:00 headline on a 13:00-close session arrives after the closing price. Including it before that price cutoff violates the intended information boundary. The [exchange calendar](https://www.nyse.com/trade/hours-calendars) explicitly includes early closes. This is necessary timing correctness, not merely additional polish.

### P12: Preserve trading-day meaning through exclusions and sample boundaries

**Decision: Accepted (2026-09-09). Type: Data integrity and implementation. References: D9; Stage 3.**

**Previous:** The common panel stores leads, but zero-news days are removed before contemporaneous lag controls are constructed. The first-session mapping lacks a prior-close lower-bound check, and missing market rows can compress lead construction.

**Proposed:** Construct market lags and leads on the full trading calendar before exclusions. Enforce the sample's lower and upper boundaries, preserve the required preceding close, and check missing market observations. Test zero-news gaps, missing price rows, and sample-edge headlines. Define how remaining gaps affect temporal inference rather than assuming row distance always equals trading-day distance.

**Reason:** If Tuesday is excluded, Wednesday's previous row can be Monday. Pre-sample news can otherwise accumulate in the first session, while a missing market row can make a one-row lead span multiple sessions. Tests should verify the intended financial time interval, not only array shifts.

### P13: Audit publication availability, not only timezone formatting

**Decision: Accepted (2026-09-09). Type: Data audit. References: D1; Stage 0.**

**Previous:** Emphasize usable intraday timestamps, a documented timezone, timestamp distributions, duplication, and coverage.

**Proposed:** Also establish whether each timestamp represents original publication, a later update, or collection. Record what is and is not known about historical availability before assigning information cutoffs.

**Reason:** A correctly localized timestamp can still describe a time other than when the content became available to a reader. Timezone correctness alone cannot establish a valid historical information set.

### P14: Implement and test the date-only fallback

**Decision: Accepted (2026-09-09). Type: Existing-plan implementation gap. References: D2.**

**Previous:** D2 promises deferring date-only news and dropping the contemporaneous analysis. The configuration flag is defined, but downstream code does not implement either behavior.

**Proposed:** If the fallback is selected after the audit, define its exact calendar mapping, implement it throughout the pipeline, test its behavior, and suppress RQ2 as specified. Do not treat switching the current flag as sufficient.

**Reason:** A documented fallback provides no protection until the code implements it. Date-only data also cannot support the same contemporaneous interpretation as verified intraday availability.

### P15: State exactly what prediction and horizon mean

**Decision: Accepted (2026-09-09). Type: Research interpretation. References: RQ3; Section 6.2.**

**Previous:** Call the fitted lead regressions predictive and refer to horizons t+1 through t+5 without an independent forecast evaluation.

**Proposed:** For the inference-focused version, describe results as in-sample conditional associations with subsequent returns. Label r_(t+h) as a single day's return observed h trading days ahead, not the cumulative return over the next h days.

**Reason:** Those regressions and cumulative-return models target different quantities. A coefficient estimated on the analysis sample does not by itself establish forecasting performance on future unseen observations.

### P16: Consider chronological forecast evaluation only if making forecasting claims

**Decision: Accepted (2026-09-09). Type: Optional scope alternative. References: RQ3; Section 1 exclusions.**

**Previous:** No held-out forecast comparison is specified; a trading-strategy backtest is out of scope.

**Proposed:** If forecasting performance becomes a central claim, add a modest chronological evaluation against a controls-only baseline, fitting preprocessing and model choices on the training portion. Otherwise retain the narrower inference interpretation in P15.

**Reason:** A forecast evaluation measures performance on later observations. It does not require positions, portfolio construction, or a trading strategy. This is an alternative to narrowing the claim, not an additional mandatory experiment for an association study.

### P17: Distinguish retrospective measurement from historical deployment

**Decision: Accepted (2026-09-09). Type: Provenance and interpretation. References: D4; D7.**

**Previous:** Apply a published pretrained model to a historical news window without separately discussing when that model became available.

**Proposed:** Identify the work as retrospective measurement unless the model's availability is consistent with the claimed historical deployment period. Record the checkpoint and its provenance.

**Reason:** Using a later model on earlier news can be valid retrospective research, but it does not recreate the tools available to an investor at that earlier date.

### P18: Assess precision and permit an inconclusive result

**Decision: Accepted (2026-09-09). Type: Statistical design. References: D4; D15; Section 6.5.**

**Previous:** Target at least 1,250 trading days and expect a null finding to rule out economically meaningful effects below a cost threshold.

**Proposed:** Define a smallest effect of interest, assess attainable precision using the chosen sample, and allow three conclusions: evidence of association, evidence the effect is sufficiently small, or an inconclusive interval. Do not equate an interval crossing zero with practical absence.

**Reason:** Sample length alone does not guarantee a useful bound. The review's illustrative calculation assumed independent observations, a standardized predictor, and residual return SD of 100 bps: SE approximately 100/sqrt(1250) = 2.83 bps, a 95% half-width of approximately 5.5 bps, and an approximately 7.9 bps effect for 80% power in a two-sided 5% test. These are illustrations, not measured SPY results; dependence, controls, exclusions, and multiplicity alter them.

### P19: Keep effect sizes but narrow the transaction-cost interpretation

**Decision: Accepted (2026-09-09). Type: Economic interpretation. References: D15; Section 6.5.**

**Previous:** Treat bps per sentiment standard deviation versus approximately 5 bps one-way costs as sufficient economic significance, including statements that useful effects are ruled out below costs.

**Proposed:** Keep bps effect sizes and their intervals. Treat the cost figure as illustrative context rather than evidence of profitability or universal lack of usefulness. Keep a trading backtest optional for this study.

**Reason:** A conditional return coefficient is not strategy profit. Signal use, timing, turnover, holding periods, and other information affect realized economic value. Neither a coefficient above costs nor one below costs settles those questions.

### P20: Define the news collection's measured quantity and audit its composition

**Decision: Accepted (2026-09-09). Type: Measurement and data design. References: D1; D5; D8.**

**Previous:** Equally weight all selected headlines and describe the aggregate as market sentiment tested against SPY. Plot coverage drift and inspect subperiods.

**Proposed:** Define the object as, for example, average financial tone in the selected news collection. Audit concentration by company, publisher/source, and period. Choose the universe on relevance and coverage criteria before examining return relationships.

**Reason:** More-covered companies receive more weight, source mixtures can change, and company-level news need not represent SPY. Plotting coverage drift and naming it as a limitation do not establish that the aggregate represents a stable market-wide quantity. The review did not prescribe a specific replacement weighting scheme.

### P21: Interpret dispersion as variation in scored tone, and correct its literature link

**Decision: Accepted (2026-09-09). Type: Construct definition and literature interpretation. References: RQ4; D8; Section 6.4.**

**Previous:** Describe within-day score dispersion as disagreement and present dispersion predicting volume as a documented Tetlock finding.

**Proposed:** Describe it as dispersion of model-assigned headline tone. Discuss company/event composition, scorer behavior, and sample support. Present its connection to volume as an extension to test, not a direct replication. Consider the relationship between mean and dispersion when interpreting the specification.

For bounded scores and the usual sample variance,

```text
d_t^2 <= [n_t / (n_t - 1)] * (1 - S_t^2).
```

**Reason:** Headlines about different events are not direct observations of investor beliefs. Bounded scores mechanically constrain mean and dispersion. [Tetlock (2007)](https://www.uts.edu.au/globalassets/sites/default/files/adg_cons2015_tetlock-journal-of-finance-2007.pdf) links unusually high or low pessimism to volume; this differs from the proposed dispersion measure.

### P22: Name and interpret the outcome variables accurately

**Decision: Accepted (2026-09-09). Type: Measurement and implementation naming. References: Section 4; Section 6.4.**

**Previous:** The code computes log share volume under the name `log_turnover`; the range-based Parkinson variance is often described simply as volatility.

**Proposed:** Rename the former log trading volume unless a denominator supporting actual turnover is obtained. Describe Parkinson's measure as an intraday range-based variance proxy.

**Reason:** Variable names should match their formulas and the claims they support. Share volume is not automatically turnover, and intraday range variance does not capture every component of total daily volatility.

### P23: Prespecify hypotheses without requiring the expected results

**Decision: Accepted (2026-09-09). Type: Research protocol and interpretation. References: Section 1; Stages 4-5; risk register.**

**Previous:** State that FinBERT wins clearly, a same-day association should be present, missing same-day significance indicates a broken pipeline, and RQ4 carries a likely positive finding.

**Proposed:** Preserve expected directions as hypotheses, allow contrary or inconclusive outcomes, and investigate absent associations without tuning the pipeline until the expected result appears. Remove the assertion that market efficiency guarantees this particular contemporaneous coefficient.

**Reason:** Weak association can reflect aggregation, the chosen universe, measurement noise, or the absence of the relationship. Prespecification should constrain analytical choices while leaving outcomes open.

### P24: Remove conclusions embedded in figure titles before results exist

**Decision: Accepted (2026-09-09). Type: Presentation and implementation. References: src/plots.py; Stage 7.**

**Previous:** Plot titles assert “Same-day association, next-day nothing,” heavier trading following disagreement, and lexicons over-predicting neutral regardless of the estimated results.

**Proposed:** Use neutral titles until findings have been reviewed; write conclusion-bearing titles and captions only when supported by the actual estimates and their uncertainty.

**Reason:** The current titles would remain unchanged if the findings reversed. A presentation intended to demonstrate statistical judgment must not predetermine its conclusions.

### P25: Treat transparent protocol corrections as legitimate research

**Decision: Accepted (2026-09-09). Type: Research governance. References: Section 3; future-work.md.**

**Previous:** Describe deviations from locked decisions as look-ahead and an empty change log as the goal. Some correctness fixes are parked as future work.

**Proposed:** Record the date, reason, affected decisions, and exposure to results when changing a protocol. Distinguish correcting an error from undisclosed outcome-driven specification changes. Update the authoritative plan and change log when an approved revision is actually adopted.

**Reason:** A transparent methodological correction is not inherently look-ahead bias. Keeping a known error solely to preserve a frozen plan undermines validity. This log records proposals; the original plan and future-work log have not been changed by creating it.

## Implementation and reproduction proposals

### P26: Make the reproduction path match the promised outputs

**Decision: Accepted (2026-09-09). Type: Existing-plan implementation gap. References: D16; Stage 7.**

**Previous:** The plan promises all tables and figures from a clean clone with one analysis command. `run_all.py` delegates Act 1 to an unfinished validation notebook, robustness remains incomplete, and required intermediate inputs are absent from a clean clone.

**Proposed:** Complete the validation and selected robustness workflows, orchestrate the promised outputs, and document prerequisites and the distinction between raw-data rebuilding and reproduction from available prepared inputs. Verify the final supported workflow from a clean clone.

**Reason:** Modular files and placeholders are useful foundations but do not establish end-to-end reproduction. The advertised command should describe an executable path and its inputs accurately.

### P27: Expose the FinBERT predictions needed for classification validation

**Decision: Accepted (2026-09-09). Type: Existing-plan implementation gap. References: src/scoring.py; src/validate.py.**

**Previous:** FinBERT exposes only P(positive) - P(negative), while the evaluation function expects class labels for FinBERT.

**Proposed:** Provide full class probabilities or an argmax prediction interface and connect it to the validation workflow, sharing inference where practical.

**Reason:** The continuous difference does not uniquely determine the winning class when neutral probability varies. Act 1 needs actual classifier predictions; Act 2 needs the continuous score.

### P28: Make cached scoring resumable and sensitive to provenance

**Decision: Accepted (2026-09-09). Type: Existing-plan implementation gap. References: Stage 1; src/scoring.py; rescore.py.**

**Previous:** Cache reuse is based on headline identifiers, and the outstanding scoring pass is written only after scoring completes. Changes to model or dictionary settings do not reliably invalidate populated scores.

**Proposed:** Include a model/dictionary/scoring-settings fingerprint in cache provenance, invalidate or explicitly rebuild incompatible scores, and checkpoint completed batches so interrupted work can resume.

**Reason:** Reusing stale scores mixes measurement definitions. Losing a long inference pass contradicts the advertised resumability and makes the time budget less reliable.

### P29: Repair the context figure's panel contract

**Decision: Accepted (2026-09-09). Type: Existing-plan implementation gap. References: src/align.py; src/plots.py.**

**Previous:** The panel omits `close_adj`, but the SPY context figure expects it and otherwise passes a scalar missing value against the date series.

**Proposed:** Make the panel and figure agree on a valid price series or another explicitly supplied, aligned context input, consistent with the intended architecture.

**Reason:** The current contract cannot produce the promised price curve and may raise a plotting error. This was identified statically; a runtime failure was not claimed to have been reproduced.

### P30: Resolve the pinned PhraseBank loader incompatibility

**Decision: Accepted (2026-09-09). Type: Dependency and data-loading correction. References: requirements.txt; src/validate.py.**

**Previous:** Pin `datasets==4.0.0` while calling the legacy PhraseBank loading path with `trust_remote_code=True`.

**Proposed:** Use a supported, pinned data artifact and compatible loading method for any retained PhraseBank branch, then verify it in the supported environment.

**Reason:** The [official Datasets 4.0.0 release notes](https://github.com/huggingface/datasets/releases/tag/4.0.0) remove dataset-script support and `trust_remote_code`. The mismatch was established from source and documentation, not by claiming an executed runtime test.

### P31: Complete the unresolved data and scoring metadata

**Decision: Accepted (2026-09-09). Type: Existing-plan completion. References: D1; D4; D7; config.py.**

**Previous:** Source, sample boundaries, FinBERT revision, and dictionary version remain unresolved in the scaffold despite the planned reproducibility guarantees.

**Proposed:** Complete the audit, record the selected source and window, and pin the actual model and dictionary artifacts before the substantive scoring and analysis runs. Keep those records consistent with cache provenance and reproduction instructions.

**Reason:** Unset metadata is expected at scaffold stage, but results cannot be reproducibly tied to a measurement procedure until those choices are resolved. This completes existing intentions rather than necessarily changing them.

## Application value, scope, and presentation proposals

### P32: Add a focused mathematical derivation and simulation

**Decision: Accepted (2026-09-09). Type: Recommended addition. References: Application objective; Section 12.**

**Previous:** Statistical depth is conveyed mainly through regression specifications, named tests, sensitivity checks, and interview explanations. There is no focused simulation assessing a method's assumptions or behavior.

**Proposed:** Add a compact mathematical appendix and one well-defined simulation tied to the research question. Suitable choices are attenuation under its assumptions and failure under scaling/nonclassical error, or interval coverage and precision under temporal dependence. Explain the estimand, assumptions, data-generating process, and what the demonstration does and does not establish.

**Reason:** The application goal is mathematical and statistical understanding. A derivation that the applicant can explain, paired with a controlled numerical demonstration, provides stronger evidence of that understanding than adding several unmotivated regressions. These are possible choices, not a requirement to implement every simulation suggested.

### P33: Narrow the initial deliverable and separate the overview from technical detail

**Decision: Accepted (2026-09-09). Type: Scope, sequence, and presentation. References: Stages 4-7; Sections 9-10.**

**Previous:** Fit all three scorers across five horizons plus contemporaneous, volume, volatility, and robustness analyses within roughly one week; fit the full write-up into two pages. The cut list puts VADER first, while a dashboard remains a stretch item.

**Proposed:** Start with independent classification validation and a primary next-day return specification, with uncertainty and a clear baseline. Add secondary horizons and outcomes only when they contribute enough to justify their cost. Keep a short overview and place derivations, protocol details, and supplementary results in an accessible appendix. Reassess the budget after data and labeling requirements are known. If time is tight, cut the dashboard, single-name extension, and selected secondary outcomes/horizons before independent validation or inference checks.

**Reason:** The stated time budget is optimistic for provenance checks, labeling, method validation, and full reproduction. A smaller completed study is easier to defend. Since the other portfolio projects were not reviewed, complementarity remains conditional; if the completed project is the referenced volatility study, emphasize measurement and inference here.

## Protocol amendments (R05, 2026-09-09)

Seven amendments to the two frozen protocols, arising from [project audit](project-audit-2026-09-09.md) findings A10 and A09. They correct defects in the specifications themselves. **They do not touch the research question, the frozen control set, the sample window, the 5 bps yardstick, the sampling design or the rubric.**

The condition that makes these legitimate rather than post hoc is recorded here explicitly: **at the time of amendment no text had been scored, no labels collected, no panel built and no coefficient estimated.** Nothing below could have been chosen by its result, because no result of any kind existed. Six were accepted and implemented; the seventh was deliberately deferred.

### M1: The advance precision check is a planning estimate, not a bound

**Decision: Accepted and implemented (2026-09-09). Type: Inference protocol, §5. Finding: A10.**

**Previous:** If `1.96*sd(r)/sqrt(n)` exceeded the 5 bps SESOI, the study "**cannot** deliver the 'effect is small' conclusion no matter what is estimated."

**Proposed:** Record two anticipated half-widths in advance: the crude outcome-scale one, and a sharper one that fits `r ~ X` and `z(S) ~ X` separately — using tone and the outcome but never jointly, so `beta` remains unseen. Both are labelled planning quantities and stored in the run manifest with `n`, `sd(r)`, `R²` and the HAC ratio. The conclusion rule consults only the realised HAC interval.

**Reason:** The original statement was false in both directions. Conditioning on controls can narrow the realised interval below the crude estimate; collinearity between tone and the controls widens it; dependence at `L = 5` moves it either way. What the advance record actually buys is evidence of sequence — that a wide interval was anticipated rather than discovered and reframed.

### M2: Replace three overlapping conclusions with two reported dimensions

**Decision: Accepted and implemented (2026-09-09). Type: Inference protocol, §5. Finding: A10. Supersedes part of P18.**

**Previous:** Three conclusion categories — evidence of association, evidence the effect is small, inconclusive.

**Proposed:** Two independent binary facts, both always reported: `A`, whether `0` is outside the interval; `B`, whether the interval is contained in `[−5, +5]` bps. A 2×2 table names the four combinations, including the informative null (`A` no, `B` yes). Boundary rule: both comparisons non-strict, evaluated at 0.1 bps reporting precision, with any endpoint within 0.1 bps of `0` or `±5` flagged as a boundary case and printed unrounded.

**Reason:** The categories overlapped with no rule to choose between them — an interval of `[1, 3]` bps satisfied two of them simultaneously — and exact `±5` endpoints had no defined treatment. `A` and `B` answer different questions, so collapsing them into one label discards information rather than resolving it. P18's substance, that an inconclusive result is a permitted and legitimate outcome, is retained as the bottom-right cell.

### M3: The 14-test secondary family contains return tests only

**Decision: Accepted and implemented (2026-09-09). Type: Both protocols — inference §6, validation §7. Finding: A10.**

**Previous:** The validation protocol delegated its secondary classifier contrasts to "the secondary family defined in the inference protocol (B02)".

**Proposed:** Close the inference protocol's secondary family at exactly the 14 (scorer, horizon) return tests, and forbid classification tests from joining it. Specify the two Act 1 contrasts in the validation protocol as pointwise paired-bootstrap intervals, labelled descriptive-secondary, with no p-value and no correction.

**Reason:** The delegation pointed at a family that never enumerated classifier contrasts, so their multiplicity treatment was undefined in both documents. The two contrasts are deterministic functions of the same three macro-F1 values as the primary and are near-collinear with it; correcting across them would imply a family structure the design does not have, while the estimand is an interval and thresholds nothing. Act 1 and Act 2 use different samples and different resampling units, so pooling them would be a category error either way.

### M4: The primary accuracy comparison must respect article groups

**Decision: Accepted and implemented (2026-09-09). Type: Validation protocol, §7. Finding: A10. Refines P04.**

**Previous:** Exact item-level McNemar as the accuracy comparison, reported separately.

**Proposed:** Make the paired **group** bootstrap on the accuracy difference primary — identical machinery, `B = 10,000`, `seed = 20260830`, to the macro-F1 interval. Demote McNemar to supplementary, printed with the number of evaluation groups, the group-size distribution, and an explicit statement that its p-value is valid only under item independence, which this design does not assert.

**Reason:** §4 assigns whole article groups to a part precisely because near-duplicate headlines are not independent. A group contributing several evaluation items therefore violates McNemar's independence assumption in the direction that matters — the p-value is too small. The pairing across scorers was never the problem; dependence within the item set is. The fix reuses machinery the protocol already specifies rather than introducing new theory.

### M5: Narrow the independence claim to labels, and drop "upper bound"

**Decision: Accepted and implemented (2026-09-09). Type: Validation protocol, §2. Finding: A10. Refines P02 and P17.**

**Previous:** New annotation has "verifiable independence from the checkpoints"; a PhraseBank fallback number is "an upper bound of unknown tightness".

**Proposed:** State that new annotation establishes **label** independence — verifiable from the annotation provenance — and that **text exposure is unknown** for all three scorers symmetrically, recorded as an Act 1 limitation in the results section. Replace the PhraseBank wording with "an optimistically biased estimate with no guarantee that it bounds generalization".

**Reason:** These are 2010–2019 headlines from the open web, and no scorer's pretraining corpus is auditable at that granularity, so prior text exposure cannot be excluded for anyone. Independent labelling does not establish it. Separately, documented contamination makes a score expected-optimistic, but it is not a mathematical bound — a contaminated score can in principle sit below a model's performance elsewhere. Both claims were stronger than the evidence supports.

### M6: Fix the circular-shift domain and the tie convention

**Decision: Accepted and implemented (2026-09-09). Type: Inference protocol, §9. Finding: A10. Completes P08.**

**Previous:** Circular shift "by `k` sessions" over `1 .. n−1`, with no stated domain and no tie handling.

**Proposed:** Shift the standardized tone of the `n` retained analysis rows in session order, after eligibility — not the full calendar — so `n` is constant across `k` and each shift is a bijection. State that with gaps, `k` positions is not `k` calendar sessions, and report the gap count and largest gap beside the percentile. Record the calendar-domain alternative as considered and rejected, because it changes `n` with `k` and breaks the bijection. Percentile uses the midrank convention with the exact-tie count reported.

**Reason:** The bijection property is the entire reason the shift replaced block resampling (P08), so the domain that preserves it must be named rather than left to the implementation. The rejected alternative would have restored the exact calendar interpretation at the cost of the property the diagnostic exists to have; that trade is recorded now so it is not silently re-made later. Ties are expected to be zero on continuous data, so a nonzero count signals a degenerate fit and is surfaced rather than absorbed.

### M7: HAC lag spacing — deferred to R07c, not decided

**Decision: Deferred by the user (2026-09-09), against the assistant's recommendation. Type: Inference protocol, §4. Finding: A09.**

**Previous:** Silence. `nw_ols` resets the index before computing HAC, so lag `ℓ` means `ℓ` retained observations, which can span more than `ℓ` calendar sessions wherever the sample has gaps. The installed statsmodels `cov_hac_simple` documents an assumption of consecutive, equally spaced periods.

**Proposed by the assistant:** Prespecify the session-indexed convention as primary now — pairs weighted by session distance, zero weight beyond `L` sessions even when within `L` rows — with retained-position HAC as a reported sensitivity. Reason offered: the corpus has one zero-news session, so the two should barely differ, which makes the choice cheap now and indefensible after its effect has been measured.

**User decision:** Leave it open until **R07c** measures the difference. No user reason was supplied beyond the instruction itself; the reasoning above is the assistant's and is not attributed to the user.

**What was recorded instead.** §4 of the inference protocol now documents both conventions, states that the choice is deferred, and states the exposure the deferral carries in plain terms: the convention will be selected after its effect on the sample has been measured. Two constraints bound that exposure and are written into the protocol — R07c is infrastructure that completes **before** any tone coefficient is estimated, so no sentiment–return result can inform the choice; and whichever convention becomes primary, the other is reported alongside, so the selection cannot conceal a divergence. If R07c finds the two conventions differ materially, that is itself the finding and is reported as one.

## Foundations recommended for retention

The following foundations remain part of the accepted direction. Acceptance does not imply that every corresponding implementation guarantee is already satisfied.

- Two linked acts: classification measurement and subsequent market analysis.
- Interpretable baselines alongside a pretrained transformer.
- Separate treatment of contemporaneous association and later outcomes.
- A common analysis table, modular code, explicit seeds, and auditable data preparation.
- Effect sizes and uncertainty alongside significance tests.
- Timestamp and duplicate audits, expanded to cover the issues above.
- No requirement to train another model or construct a trading strategy.
- Acceptance of a supported null or inconclusive result as a legitimate research outcome.

## Accepted implementation direction

The user accepted the review's direction and instructed incremental implementation. Each work session ends with a summary and a stop for explicit instructions; the list below is a roadmap, not authorization to run all stages in one turn. Detailed specifications and any architecture proposals are handled in their own increments.

1. Revise the protocol: validation independence, measurement-error argument, hypothesis families, timing, and permitted conclusions.
2. Complete the data audit: historical availability, news-universe relevance, composition, coverage, and attainable sample size/precision.
3. Build the smallest complete study: independent classification validation plus the primary next-day specification and justified uncertainty.
4. Add the selected mathematical demonstration and decide whether forecast evaluation is needed for the intended claim.
5. Complete reproduction, verify outputs, and write the overview and appendix; then assess the value of secondary analyses.

## Decision history

Record actual decisions here by proposal ID, with the user's reason when one is given. If no user reason is supplied, say so rather than attributing the review's reasoning to the user. Partial acceptance should specify which parts were accepted and leave alternatives unresolved. Record implementation and verification separately when they occur.

| Date | Item | Decision | Reason or scope | Implementation |
|---|---|---|---|---|
| 2026-09-09 | DOC-01 | Accepted | User requested a document in `docs/` covering prior structure, all proposals, acceptance status, and reasoning | This document created |
| 2026-09-09 | P01-P33 | Pending at initial recording | At document creation, the user had not yet accepted or rejected the reviewed changes | No reviewed changes implemented at that point |
| 2026-09-09 | P01-P33 | Accepted, superseding the earlier pending status | User: "i accept all your proposals, now lets start building it slowly". No additional proposal-specific rationale supplied | Approval recorded; this does not mark the methods or code fixes complete |
| 2026-09-09 | DOC-02 / Increment 1 | Completed documentation increment | Record acceptance and draft the revised scope; retain the original specification as a historical baseline | Updated this log and created [protocol-revision-draft.md](protocol-revision-draft.md); no analysis-code changes |
| 2026-09-09 | DOC-03 / Implementation plan | Completed planning checkpoint; execution awaiting go-ahead | User requested the implementation plan in chat and `docs/` before implementation | Created the replacement [implementation plan](implementation-plan.md), preserved the original from Git history in [implementation-plan-original.md](implementation-plan-original.md), and updated documentation links; B01-B28 remain unimplemented |
| 2026-09-09 | DEV-01 | Recorded, not authorized in advance | Assistant worked from the superseded original plan before reading the replacement; no user instruction to do so | Working-tree files listed in the status table above. No Git writes. Disposition (keep as B03 input, or revert) left to the user |
| 2026-09-09 | B01 | Completed | User: "start on the implementation plan, the one that is not the original one... small increments that i can validate and commit myself, do not push" | Wrote [validation-protocol.md](validation-protocol.md): estimand, primary annotated-headline source with a documented fallback order, stratified seeded sampling, group-level calibration/evaluation separation, rubric v1 with decision rules and blindness requirements, annotator/agreement/provenance requirements, paired group bootstrap for the macro-F1 difference with McNemar kept separate, and a 60-item pilot. No labels collected; no analysis code changed |
| 2026-09-09 | B03 | Completed | User: "do bo3 and bo4 downlaod the dataset and run it" | Downloaded 24 x 6 MB evenly spaced HTTP range slices of FNSPID `All_external.csv` (356,123 rows, 2.5% of 5.7 GB) plus an 8 x 4 MB probe of `nasdaq_exteral_data.csv`. Findings in [data-audit-fnspid.md](data-audit-fnspid.md): the file concatenates five-plus sub-corpora; only Reuters is intraday (1440 distinct minutes, 0.2% midnight) and it is a global general newswire with no ticker tags weighted to European hours; Benzinga is 96-100% date-only through 2019; `lenta.ru` is Russian-language non-financial text present in both files; `Article` contains embedded newlines; timezone is explicit (`UTC` suffix) but timestamp *semantics* are undocumented upstream and recorded as unknown. No return relationships examined |
| 2026-09-09 | B04 | Completed, with the window provisional | Depends on B03 | Selected the Benzinga sub-corpus on relevance and coverage; recorded a mandatory source-domain filter; activated `DATE_ONLY_FALLBACK` and set `RQ2_ADMISSIBLE = False`; set a provisional 2010-01-01..2019-12-31 window flagged `SAMPLE_WINDOW_PROVISIONAL`; pinned FNSPID repo revision, file sha256 and byte length, and the FinBERT revision plus its verified non-conventional label order. Loughran-McDonald release remains UNRESOLVED. Added a `run_all.py` guard so dependent scoring and panel building stop until B09 implements the fallback |
| 2026-09-09 | B12-B14 | Completed | User: "yes continue" | **B12**: added a `Classifier` protocol distinct from `Scorer`, with `predict_proba` (columns re-ordered into `config.LABELS`, so no caller needs to know the checkpoint's published order) and `predict` (argmax, retaining the neutral mass). A test states the reason the two paths cannot be merged: `(0.10, 0.40, 0.50)` and `(0.05, 0.50, 0.45)` share `P(pos)-P(neg) = 0.4` but differ by argmax, so no threshold on the continuous score recovers the label. `FinbertScorer` now also refuses to construct if the checkpoint's label order differs from the pinned one. `validate.predictions_for` refuses to threshold a classifier or to argmax a lexicon. **B13**: every scorer publishes a `fingerprint` naming what changes its output -- model revision and truncation length for FinBERT, a hash of the actual word lists for LM, so a swapped dictionary is caught even if `LM_DICT_VERSION` was not updated. Stored in a `.meta.json` sidecar; a mismatch raises `IncompatibleCache` naming what changed and how many values are affected, with `on_fingerprint_change='rescore'` as the only alternative -- silent reuse is not offered. **B14**: scoring checkpoints every 50,000 rows with atomic temp-file-and-replace writes; a test interrupts a 250-row pass at 200, resumes, rescores only the missing 50, and asserts the result equals an uninterrupted run. 21 new tests, 120 passing |
| 2026-09-09 | B22 | **Blocked, not attempted** | Environment | `torch` installs but its DLLs are refused by Windows Smart App Control (`WinError 4551`, `torch/lib/c10.dll`), so FinBERT cannot be instantiated and the pilot cannot be timed. `transformers` and `vaderSentiment` import normally. Resolving this requires the user to disable Smart App Control; it is not fixable from the code |
| 2026-09-09 | Corpus assembly + census | Completed | User: "you can continue" | Streamed the full 5.73 GB `All_external.csv` once (13,057,514 rows, 11.5 min) through a filtering parser, storing only the result. **The selected source occupies two separate blocks** -- kept rows plateaued at 0.55 GB, then resumed at 2.44 GB -- so reading only the block the audit sample found would have silently dropped ~122,000 headlines (~9%). 0 malformed timestamps across the whole file, which **corrects** the earlier 0.05% figure: those 180 rows were an artifact of byte-range slicing cutting records mid-field, not a property of the data; the embedded-newline finding is unchanged. Replaced the O(n x window) near-duplicate search with sliding-window deletion-signature blocking, verified identical to the pairwise oracle and ~160x faster. Census: dedup rate **38.5%** (431,602 exact + 111,717 near), 345 headlines/session median 339, **2 zero-news sessions in ten years**, 5,707 tickers with a top-10 share of 2.1% and an effective 1,839 names. Fixed an off-by-one: the window's end bound was inclusive of the following midnight, admitting a whole extra day for a date-only corpus; 16 rows were affected. 19 new tests, 107 passing |
| 2026-09-09 | B09b | Completed | User: "you can continue" | `load_news` rewritten: source-domain filtering is now mandatory and defaults to `config.NEWS_SOURCE_DOMAINS`, so the Russian-language `lenta.ru` sub-corpus cannot reach an English financial scorer by omission; disabling the filter requires passing an empty tuple explicitly. The obsolete standalone `benzinga` spec is removed with an error message explaining that Benzinga is a sub-corpus selected by URL host. Malformed timestamps are rejected against the documented pattern and counted separately from missing values, since in FNSPID they are article body text crossing a record boundary. CSV reading is chunked (default 500k rows) for the 5.7 GB file. Load counts, including rejected hosts, are attached to `.attrs['load_stats']`. RQ2 suppression is structural: `inference.contemporaneous` raises while `config.RQ2_ADMISSIBLE` is False, with `allow_inadmissible=True` reserved for a deliberate labelled exhibit, and `run_all.py` writes no Table 2, deletes any stale one, and passes no t=0 point to Figure 2. `DATE_ONLY_FALLBACK_IMPLEMENTED` set True; the `run_all.py` guard is retained as a standing check. A defect found by the new tests: the headline schema's string dtypes depended on `chunksize`, so the schema varied with a performance knob and a later merge on `headline_id` could mismatch; dtypes are now pinned after concatenation. 15 new tests; two B06 tests updated to opt in explicitly now that the guard fires first. 94 passing overall |
| 2026-09-09 | B09a | Completed | User: "do the next increment" | Added `align.map_date_to_session`: primary rule maps a headline dated d to the first session strictly after d (session t receives dates in [prev_session, t)), so all of day d precedes close(t); the secondary as-if-intraday rule sits behind `defer=False` and is fixed in advance as a sensitivity exhibit only. Reads the wall-clock date in the source's own zone and never converts -- a test documents the trap, since converting a 00:00 UTC stamp to market time moves it to the previous calendar day and shifts every headline one session early. Both calendar edges return NaT, with an optional `prior_session`. Added `_reject_intraday`, the mirror of B07's guard, so the two mappers refuse each other's input. `aggregate_daily` now dispatches on `config.DATE_ONLY_FALLBACK` and requires `ts_utc` in date-only mode. A defect found by the new tests: the lower-edge guard keyed on the resulting index, which wrongly dropped a date that *is* the first session under the secondary rule; it now keys on the date itself. 16 new tests, 79 passing overall |
| 2026-09-09 | B07-B08 | Completed | User: "onto the next" | D-1 fixed: `map_to_trading_day` now returns NaT at the *lower* calendar edge as well as the upper, so pre-window headlines are dropped rather than piled onto the first session; the first session's window opens at the previous session's close, which lies outside the calendar, so an optional timezone-aware `prior_close` populates it and the default drops one session's news rather than guessing. D-3 fixed: `trading_calendar` raises instead of falling back to `pd.bdate_range`, which had been treating ~9 market holidays a year as sessions behind a warning. B07 descope guard added: `map_to_trading_day` refuses date-only input (concentration test, skipped below 200 rows so small fixtures still work), so the descope of actual session closes cannot be undone by accident. One scoped opt-out in `audit.coverage_profile`, which bins headlines only to count coverage and where a one-session shift leaves the distribution materially unchanged; documented inline. 8 new tests, 63 passing overall |
| 2026-09-09 | B06 | Completed | User: "yes do b06" | Panel is now built by LEFT-joining market onto the calendar-indexed daily frame, so a missing price row no longer lets `.shift(-1)` span two sessions; it leaves NaN and the row is excluded at eligibility, with the count reported in `panel.attrs['build_stats']` and printed by `run_all.py`. Full-calendar lag columns (`ret_lag1`, `parkinson_lag1`, `log_turnover_lag1`) are built in `build_panel`; `inference.contemporaneous` now reads them and raises if they are absent, instead of shifting inside a frame from which zero-news sessions had already been removed. Added `align.assert_sessions_match_calendar`, called by `run_all.py`. `close_adj` added to the panel schema (partially addresses P29; the figure contract itself is B16). 11 new tests, 55 passing overall. Verified end to end on the real NYSE calendar (1,258 sessions over 2015-2019, 46 holidays correctly excluded) with synthetic scores. Environment note: the Smart App Control block on scipy has cleared, so `statsmodels` and the inference path are now executable here |
| 2026-09-09 | B05 | Completed | User: "proceed" | Wrote [timing-contract.md](timing-contract.md). Reproduced three defects against the current code before specifying them away: (D-1) `map_to_trading_day` guards its upper calendar edge but not its lower, so all pre-window headlines land on the first session; (D-2) `build_panel`'s inner join lets a missing price row turn `ret_lead1` into a two-session lead with no warning; (D-3) `trading_calendar` falls back to business days on ImportError, and the fallback is live in the current environment. Specified the date-only mapping rule (primary: first session strictly after date d, with the staleness cost and weekday variation stated; secondary as-if-intraday rule fixed in advance as a sensitivity only), sample-edge and missing-session handling, structural RQ2 suppression, and the B06-B09 test list. Architecture gate closed without opening; B07 descoped for this corpus with a recorded trigger. No code changed |
| 2026-09-09 | DEV-01 rework | Completed | User: "yeah rework it" | Rewrote `src/audit.py` around the source as the audit unit: `profile_by_source`, `midnight_share_by_year`, `script_profile` (catches non-English sources), `screen_sources` (mechanical criteria only -- it explicitly refuses to judge relevance), and a `CorpusRate` type that withholds duplication and per-session rates on a cluster sample with the reason attached. Fixed a real defect found by the new tests: concentration was being measured in market time, but FNSPID's date-only stamps are 00:00 UTC = 19:00 ET, so `midnight_share` read 0% -- concentration is now measured in the source's published zone and session position in market time. Rewrote `tests/test_audit.py` (22 tests) on a synthetic three-source corpus reproducing the real failure modes; regenerated `notebooks/01_data_audit.ipynb` and executed every cell against the downloaded sample, reproducing the B03 findings |
| 2026-09-09 | B02 | Completed | User: "dont mind the unresolved until they come. Continue the B02". B01's open items (second annotator, final sample size) deferred by that instruction, not resolved | Wrote [inference-protocol.md](inference-protocol.md): primary estimand and two-sided null; frozen control set and variable definitions; eligibility and full-calendar lag/lead contract with an adjacency assertion; HAC L=5 prespecified with a sensitivity set; bps-per-1SD scale, a 5 bps smallest effect of interest, an advance precision check and the three permitted conclusions; primary/secondary/exploratory families with BH plus BY under arbitrary dependence; RQ4 resolved to a joint HAC Wald null; stacked paired contrast for scorer comparison; attenuation stated as a conditional hypothesis; the placebo demoted to a circular-shift percentile diagnostic; naming corrections; a forbidden-claims list; and eight data-dependent feasibility checks. No model fitted; no analysis code changed |
| 2026-09-09 | R01a | Completed | Audit finding A01; user approved the repair sequence | Fingerprints compared by JSON representation so identity survives save/load; FinBERT records the revision actually passed to its constructor rather than rereading global config; populated cache columns with missing, null or empty identities are rejected before scorer construction. Unknown-provenance caches require original metadata or a rebuild into a new path and are never relabelled by the rescore option. 13 new tests; 133 passing, 7 skipped |
| 2026-09-09 | R01b | Completed | Audit finding A01 | Default calls now validate every configured scorer identity even when all values are cached, without loading model weights. The cache Parquet carries `text_sha256` of the exact scoring text, so changed headlines refuse reuse; explicit rescore invalidates all scorer values for the affected ids, and a changed scorer fingerprint invalidates that entire cached column including rows outside the request. Populated legacy caches without text identity require rebuilding to a new path. 8 new tests; 141 passing, 7 skipped |
| 2026-09-09 | R01c | Completed (specification only) | Audit findings A02/A15 | Wrote the [scoring checkpoint contract](scoring-checkpoint-contract.md): scores and provenance in one Parquet file, one replacement commit point, a process-held writer lock, explicit legacy-cache refusal, named metadata fields, enumerated interruption states and recovery, and a no-GPU read-validation path. A disposable Arrow metadata round-trip succeeded; no production code changed |
| 2026-09-09 | R01d | Completed | User approved the checkpoint contract under the existing architecture gate | The contract was already implemented in `src/scoring.py`; verification confirmed all 30 cases in `tests/test_checkpoints.py` cover its stated failure states, and the "approval pending" status in both documents was corrected. No code change was required. Also fixed a test that asserted cache contents positionally — a checkpoint writes unrequested cache rows before the scored frame, so on-disk order is not request order; the assertion now indexes by `headline_id` |
| 2026-09-09 | R02 | Completed | Audit finding A05 | `assemble_corpus` defaults to `HEADLINES_RAW_PARQUET` and raises `AcquisitionError` if pointed at `HEADLINES_PARQUET`, so a documented command can no longer replace 869,205 deduplicated rows with 1,412,524 undeduplicated ones under the same name. The download URL is built from `config.NEWS_HF_REVISION` instead of `/resolve/main/`; the stream is hashed as consumed and checked against the pinned SHA-256 and byte length **before** anything is published; publication is atomic with the manifest written after the data, so a crash leaves data marked unverified rather than the reverse. Retrospective manifests for the existing artifacts record `verified_against_pin: false`, since they were produced from `/resolve/main/` with no digest computed — `--dedup` consequently refuses to rebuild from them, which is correct. 10 new offline tests; 181 passing, 7 skipped |
| 2026-09-09 | R03c | Completed | Audit finding A06 | `corpus_census` assigned sessions with the deferred rule and then passed the rows to `coverage_profile`, which re-mapped them with the intraday close rule; the two disagreed on 2,502 of 2,516 sessions. `coverage_profile` now accepts pre-assigned sessions and every census count derives from one assignment: **869,114 assigned + 91 unassignable = 869,205**, reconciling with the corpus exactly. The corpus has **one** zero-news session, 2010-01-04, and it is structural — under the deferred rule the window's first session cannot receive a headline — so there is no genuine news outage in ten years and the D9 exclusion is inert here. The earlier median of 339 and "2 zero-news sessions" were artifacts of the mismatch and are corrected to 337 and 1. The yearly stability table always used the correct mapping, so the D4 freeze is unaffected |
| 2026-09-09 | R04a | Completed | Audit finding A04 | `load_market` computed `diff(log(close))` over the rows the price source returned, before any calendar reindex, so a missing session produced a two-session return in the row labelled with the later date. New `returns_on_calendar` places prices on the exchange calendar first and requires **both** adjacent closes, so a gap makes the returns on either side NaN rather than wrong. `load_market` now takes `calendar` and `warmup_sessions`, adds one day to yfinance's exclusive `end` (the configured inclusive end would otherwise drop the final session), flags warm-up rows `in_window=False`, and reports missing sessions in `.attrs["market_stats"]`. 9 new offline tests including the audit's exact fixture — Monday 100, Tuesday absent, Wednesday 102, where the naive computation gives 0.019803 and it is now NaN on both sides; 190 passing, 7 skipped |
| 2026-09-09 | R04b | Completed | Audit findings A03/A15 | `aggregate_daily` inner-joined headlines to scores and let the group mean skip missing values, so five headlines with one score row reported `n_headlines = 1` and each scorer could summarise a different subset of the same session. New `align.validate_scores` requires, and `aggregate_daily` enforces by default, exactly one finite in-range score per headline per configured scorer; ids must be unique and non-null on both sides and the join is `validate="one_to_one"`. The join is now LEFT and every headline is mapped, because an inner join makes an unscored headline indistinguishable from one that never existed. For a declared partial pass, `require_complete=False` keeps only sessions whose headlines are all scored and reports the rest in `.attrs["coverage"]`, so bounded scoring produces whole sessions or none. `run_all.py` refuses a score cache declaring no provenance for a configured scorer. 9 new tests; 199 passing, 7 skipped |
| 2026-09-09 | R05 / M1–M6 | Accepted and implemented | User: "do R05 as specified above, M7 left open until R07c measures the difference" | Six amendments to the two frozen protocols, recorded in full above and in each document's own amendment record. **Inference protocol:** M1 replaces the false "cannot deliver no matter what is estimated" precision bound with two advance planning half-widths that never combine tone and the outcome in one fit; M2 replaces three overlapping conclusion categories with two always-reported dimensions, a 2×2 naming rule and a non-strict 0.1 bps boundary rule; M3 closes the 14-test secondary family at return tests only; M6 fixes the circular-shift domain to the retained analysis rows, records the rejected calendar alternative, and specifies midrank ties with a reported tie count. **Validation protocol:** M4 makes the paired group bootstrap the primary accuracy comparison and demotes exact McNemar to supplementary with its validity condition and the group-size distribution printed beside it; M5 narrows independence to labels, records text exposure as unknown for all three scorers symmetrically, and withdraws the "upper bound" description of a contaminated PhraseBank score. Documentation only — no analysis code changed, no test behaviour affected, 199 passing and 7 skipped before and after. Research question, frozen controls, window, 5 bps yardstick, sampling design and rubric untouched |
| 2026-09-09 | R05 / M7 | **Deferred by the user** | User: "M7 left open until R07c measures the difference". No further reason supplied; the recommendation to prespecify was the assistant's | The HAC lag-spacing convention (retained-position vs session-indexed, audit A09) is **not decided**. Inference protocol §4 now documents both conventions, marks the choice open, and states the exposure plainly: the convention will be selected after its effect on the sample has been measured. Two constraints are written into the protocol to bound that — R07c completes before any tone coefficient is estimated, so no sentiment–return result can inform the choice, and the non-primary convention is reported alongside whichever is chosen. Listed in §13's open items as owned by R07c |

| 2026-09-09 | R03a | Completed (specification + measurement) | User: "go", after agreeing the corpus track runs first because annotation is coming but not soon | Wrote the [dedup lineage contract](dedup-lineage-contract.md) for audit finding A12 and A11's missing article-group ID. Five defects reproduced against the running code, then quantified read-only on the real 1,412,524-row raw corpus. **`dedup()` does reproduce the saved clean corpus exactly** (869,205 rows, symmetric difference 0, identical on repeat) — an earlier probe suggesting otherwise had used a different sort and its conclusion is withdrawn — **but the output is not order-invariant**: permuting the input changes 2,363 headline ids and the retained ticker tags of 93,552 of 868,023 survivors (10.78%), because `dedup` sorts on `ts_utc` with an unstable quicksort and a date-only corpus makes every same-day duplicate group a tie. Ticker-tag loss is 501,957 of 663,074 distinct (group, ticker) pairs (75.7%) across 96.8% of multi-row groups; recomputing concentration with tags unioned moves the top-10 share from 2.09% to 2.03% and effective names from 1,839 to 1,810, so **the D4 universe justification is unaffected and marginally better supported** — reported because the check had to be reported whichever way it came out. Re-anchoring the exact window on the last kept occurrence corrects the split from 431,602 exact / 111,717 near to **539,087 / 4,245**, meaning 96.2% of the reported near count is actually exact duplication, while corpus membership moves by only **93 ids (0.011%)**. The blocking-exposure boundary is confirmed: an exact integer formulation gives 20 at `overlap=0.90` where `ceil` gives 21, omitting 14,494 rows and understating exposure as 5.956% instead of 6.982%. `headline_id` is a group key rather than a row key (518,332 raw rows share one), so lineage requires a new `source_row_id`. The contract separates **dedup clusters** (windowed, corpus-wide) from **article groups** (unwindowed, computed within the drawn sample at R06a), since leakage only matters between the calibration and evaluation parts and all-pairs comparison over ~800 items is exact and instant. Specification only: no production code changed, no artifact written or replaced, 199 passing and 7 skipped unchanged. Four decisions requested in §7 before R03b implements it |

| 2026-09-09 | Environment / B22 blocker | **Resolved** | The user disabled Smart App Control | `torch 2.14.0+cpu` and `transformers 5.16.1` now load and FinBERT runs. The seven tests skipped since B22 execute for the first time; the suite reports 0 skipped. B22, R10 and R11 are unblocked. The Loughran-McDonald dictionary and a named annotator remain outstanding |
| 2026-09-09 | Exhibit A premise | Assertion removed, not inverted | User: "you can convert it to a characterization test and continue from there" | On its first real execution `test_finbert_gets_the_documented_hard_cases_right` failed. FinBERT calls "Costs fell sharply in the third quarter" **negative with P = 0.932** and "Profit warning smaller than feared" **negative with P = 0.924** — the two cases the fixture was built to show a context model getting right, and on both it agrees with the word counter it was supposed to beat. Not a code defect: the B12 label-order pin is confirmed correct against the real checkpoint ("shares plunge" → negative, "profit beats" → positive), which had never been executed before. What failed was a test asserting a research expectation on six invented sentences before Act 1 has run — the predetermined-finding pattern P24 removed from figure titles. Replaced with a characterization record pinned to `config.FINBERT_REVISION` that detects a changed model, revision or label mapping and claims nothing about quality, plus a second test isolating the label-order contract. The comment "This is the gap FinBERT should close" was removed from the LM test. Six invented sentences establish nothing in either direction; Act 1 measures classification quality on independently annotated corpus text. The same unverified premise still stands in `README.md:5` and `report/report.md:12` (R09/B27) |
| 2026-09-09 | R03b | Completed | Continuation of the corpus track under the same "go" | Implemented the [dedup lineage contract](dedup-lineage-contract.md) in `src/data.py`, with `data/raw/download.py` writing lineage after the clean corpus so a crash leaves a corpus with missing lineage rather than the reverse. Representative selection is now the total order `(ts_utc, source_row_id)`; the exact window re-anchors on the last kept occurrence; ticker tags are unioned across the cluster; the blocking boundary is computed in exact rational arithmetic (20 at `overlap=0.90`, where the float form gave 21); and a lineage artifact records one row per raw row. **Verified on the real 1,412,524-row corpus, read-only:** permuting the input now yields a membership symmetric difference of **0** with **0** differing representatives and **0** differing tag sets, against 2,363 ids and 93,552 tag sets (10.78%) before; exact drops **539,087**, matching R03a's prediction exactly; distinct tickers 5,707 → **6,235** and tag slots 869,205 → **1,386,040**; top-10 share 2.09% → 2.04% and effective names 1,839 → 1,809, so D4 is unaffected; the dedup rate recomputed from lineage alone equals the reported rate to six decimal places; `headline_id` stays unique on survivors so R04b's gate is untouched. Membership differs from the saved artifact by 2,557 ids (0.29%), net −11 — R03a's figure of 93 measured the window repair alone holding the old sort fixed, while this is the combined effect of the window repair and the total-order rule. **A design decision came out of implementation:** the first version asserted that every eliminated row points at a survivor, and that assertion failed on the real corpus — an exact repeat's anchor can itself be removed later as a near-duplicate, so elimination forms chains (12,074 rows, 2.2%). Lineage now records the immediate eliminator and `cluster_id` resolves to the surviving root; the contract and its acceptance test were corrected to the invariant that actually holds. 9 new tests; **216 passing, 0 skipped**. A second defect surfaced the same way: the first implementation wrote lineage to the absolute `config.DEDUP_LINEAGE_PARQUET`, so the acquisition test published a corpus into `tmp_path` while depositing its lineage in the real `data/interim`, beside the frozen corpus. The writer now derives the path from the corpus it accompanies (`download.lineage_path`, mirroring `manifest_path`), and the test asserts both that lineage lands beside `out` and that the configured path stays absent. No artifact written or replaced — that is R03d |

| 2026-09-09 | Artifact pin correction (B04 / R03d) | **Corrected**, with the file unchanged | Surfaced by R02's digest verification on the first assembly run that exercised it | `config.NEWS_FILE_SHA256` held `dde529189c87048a8be1f73d17ecd5e211fc7809032e60c72cd22645f111c470`, which is **not** a SHA-256 of the file. B04 recorded it from the HTTP **ETag** — `docs/data-audit-fnspid.md` labelled the row "File sha256 (etag)" — and HuggingFace serves its `xetHash` as the ETag for Xet-backed files, so a Xet content-addressing digest was pinned under a SHA-256 name. The real SHA-256 is `5d4c018036bd82ca821da71b7a9c0c7db3289642e0fc6f897ea69f4a0c5135c3`, confirmed two independent ways: HF's `paths-info` API reports it as the Git LFS `oid` for revision `bf9189c4…` (LFS OIDs are sha256-of-content), and streaming the 5.73 GB file and hashing it as consumed produced the same value. The revision, the byte length (5,731,397,037, matching the served `Content-Length`) and the content are all unchanged — **only the pin was mislabelled**. `NEWS_FILE_XET_HASH` now records the original value for traceability. **This is R02's guard doing exactly what it was built for:** the assembly refused to publish and wrote nothing, rather than silently accepting a file it could not verify. It also means the pre-R02 corpus had never had its digest checked at all, which is why its manifest records `verified_against_pin: false`. No corpus membership decision, universe choice or protocol is affected |

| 2026-09-09 | R03d | Completed; **frozen corpus replaced** | User: "yeah continue and also write in handoff and catch up the docs", after being told explicitly that this step replaces the frozen artifact and is not reversible from the repo | Rebuilt raw -> clean end to end through R02's verified acquisition path and replaced `interim/headlines.parquet`; the previous artifacts were copied aside first. **The raw corpus is the same data**: 13,057,514 rows read, 1,412,524 kept, identical `headline_id` set (894,192 distinct) and identical `text_norm` multiset to the corpus it replaced, so the rebuild changed provenance and row identifiers rather than content. Its manifest records **`verified_against_pin: true` for the first time**. Clean corpus **869,205 -> 869,183** (1,168 dropped, 1,146 added, symmetric difference 2,314 ids = 0.27%), split **539,087 exact / 4,254 near**; these differ slightly from R03b's dry run (869,194 / 4,243) because that run numbered `source_row_id` from the stored artifact's order while acquisition numbers it in file order, which is the contract's definition. Ticker tags 5,707 -> **6,235** distinct and 869,205 -> **1,385,810** slots; top-10 share 2.09% -> **2.04%**, effective names 1,839 -> **1,809**. Lineage written beside the corpus, 1,412,524 rows, kept count equal to the clean corpus, every `cluster_id` resolving to a survivor, rate recomputable at 0.384660. **D4 holds**: census reconciles at **869,092 assigned + 91 unassignable = 869,183**, no year below the coverage tolerance, and one structural zero-news session (2010-01-04). The corpus track is closed; R06a's annotation draw is no longer blocked |

| 2026-09-09 | R06a | Completed; **the draw is made** | User: "yes continue" | New `src/annotate.py` and `tests/test_validation.py` (30 tests) draw Act 1's blind evaluation sample from the rebuilt 869,183-row corpus into `data/annotation/`: **800 headlines, 200 calibration / 600 evaluation**, proportional allocation across 10 strata (year x ticker-tag presence), simple random sampling without replacement per cell, seed `20260830`. Per protocol section 3 **the draw is made once**; repeating it requires a dated entry here with the reason (P25). Article groups are computed **within the sample** by exact all-pairs comparison and are **unwindowed** — the relation R03a's contract separated from the windowed dedup cluster, because a story republished ten days later survives dedup twice but is still one article for leakage purposes. 799 groups, one holding two items; a whole group goes to one part, and `load_split` re-checks that invariant when reading `split_assignment.csv` and refuses a file where any group spans both parts, which is protocol section 4's stated reason for storing the split rather than reseeding it. The blind export carries `headline_id` and `text` only, guarded against a named forbidden-column list, shuffled under a **separate** order seed (`20260831`) so that reshuffling the running order can never disturb which rows were drawn. 60-item pilot and 160-item (20%) second-annotator subset identified. `provenance.md` is written at draw time with annotator, dates, blindness confirmation and deviations left as explicit **TO BE COMPLETED** blanks, and a test asserts they stay blank — a pre-filled template would produce exactly the reconstructed-after-the-fact record section 6 forbids. Two strata collapsed to one level on this corpus and are recorded rather than dropped: every headline carries a ticker tag (0.00% untagged) and the universe is a single publisher by construction. `data/annotation/` is deliberately not gitignored, unlike `data/raw|interim|processed`. No labels were produced and no metric computed; **246 tests pass, 0 skipped**. The remaining dependency for Act 1 is a named annotator |


The work so far records acceptance, the scope draft, the implementation plan, B01's validation protocol, B02's inference protocol, the B03-B04 data audit and provenance decisions, B05's timing contract, the B06-B09 timing repairs, the corpus assembly and census, the B12-B14 scoring infrastructure, and the R01-R05 audit repairs.

**Current state, stated once.** Phase B is closed. The corpus **is** assembled and censused (869,205 deduplicated Benzinga headlines over 2,516 sessions), D4 is frozen on coverage evidence, the scoring and checkpoint infrastructure is complete and tested, and the acquisition, census, market-loader and score-to-panel defects found by the 2026-09-09 audit are repaired. The two frozen protocols have been amended once, at R05, before any result existed.

**No text has been scored, no labels have been collected, no annotation sample has been drawn, no panel has been built, no model has been fitted, and no Act 1 or Act 2 result exists.** B22 cannot proceed until Smart App Control stops blocking torch; Act 1 cannot proceed without the Loughran-McDonald dictionary and a named annotator.

> **Corrected 2026-09-09 (R05).** This paragraph previously asserted both that the corpus was assembled and that "no corpus has been assembled" — the second clause was left over from the B03-B04 entry written before assembly and was never updated. It also described B09 as a live blocker after B09 had been completed. Audit finding A14 named this as documentation that could send a later session down the wrong path; the single-status statement above replaces it.

P01-P33 are accepted and M1-M6 are accepted and implemented; M7 is deferred to R07c. Acceptance of a proposal is not implementation of it, and implementation is not empirical verification — the three are recorded separately in the table above. The assistant treats version control as read-only and waits for explicit instructions before starting the next increment.
