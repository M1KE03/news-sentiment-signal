# Implementation plan: financial headline sentiment

Date: 2026-09-09.

**Status: in execution. B01-B09 and B12-B14 are complete; the corpus is assembled and D4 is frozen.** A [project audit](project-audit-2026-09-09.md) on 2026-09-09 found fifteen issues in the work done under this plan, and the repair sequence is now the authoritative execution order — it sits inside this plan's B01-B28 rather than replacing it. Current state is in `handover.md`.

> **Corrected 2026-09-09 (A14).** This line previously read "implementation under this plan has not started" and remained there through fourteen completed increments, so a reader picking the project up was told to begin at B01.

Related documents: [accepted proposals and decision history](research-review-decision-log.md), [completed scope draft](protocol-revision-draft.md), and original implementation plan.

The original plan was absent from the working tree when this replacement was requested. Its reference copy was recovered by reading Git history at commit `b25bdb90093b7da8b3b68434f00808b6c83ec644`, blob `6f8361794d6cd43b4bbb71e32b5933a7f1cdd2ec`. That copy is historical context, not the current execution specification. No Git state was modified to recover it.

## 1. Goal and scope

Build a defensible retrospective study answering:

> How do financial sentiment measurements differ in classification quality, and what additional information do they provide about subsequent market outcomes?

The core has two acts:

1. **Independent classification validation:** compare FinBERT, LM, and VADER on independently labeled financial headlines. The primary classification contrast is FinBERT minus LM macro-F1, with paired uncertainty. Report McNemar separately as an accuracy comparison.
2. **Market inference:** estimate the association between FinBERT daily tone and the next trading day's SPY log return, conditional on a prespecified control set. Report the effect in basis points per sentiment standard deviation, its uncertainty, and the study's precision limits.

Same-day association is secondary and requires trustworthy intraday timestamps. Other scorers/horizons and volume/range-variance outcomes are explicitly secondary or exploratory. The initial core does not require chronological forecasting, a dashboard, a single-name study, or a trading strategy.

Retain the current Python modules, common analysis table, separate expensive scoring step, and existing framework. Correctness repairs do not justify an unrelated restructure.

## 2. How execution will work

- Work in **15-30-minute increments**, including reasoning, edits, and focused verification. A phase contains several increments; it is not one long session.
- Each execution turn addresses one bounded increment. If it cannot finish within the time window, leave a clear checkpoint and stop rather than expanding its scope.
- After every increment, summarize changed files, checks and results, trade-offs or unresolved issues, and the suggested next increment. Wait for explicit instructions before continuing.
- A go-ahead to begin this plan starts **B01 only**, unless the user explicitly names another bounded task. It does not authorize automatic execution of the full table.
- Keep decisions, implementation, and verification separate in the decision log. Do not mark a proposal complete because a plan or function stub exists.
- No Git writes: no staging, commits, pushes, branch/tag creation, resets, restores, or Git configuration changes. Read-only status, diffs, and history are permitted. The human handles version-control writes.
- Do not change production credentials, environment files, or CI/CD. Stop for explicit approval before a concrete architecture change. No destructive cleanup is part of this plan.
- Annotation and full scoring are separate workload dependencies. Do not pretend either necessarily fits one increment. Repeated scoring chunks require their own instructed sessions and durable checkpoints.

## 3. Scientific decisions and remaining specifications

The accepted direction is settled; the following operational details must be written down before fitting the substantive outcome models. These are specifications to resolve, not a request to accept P01-P33 again.

| Area | Starting direction | Must be specified before use |
|---|---|---|
| Validation source | Prefer independently annotated headlines from the selected news collection | Sampling, rubric, annotators, uncertainty, label provenance, and calibration/evaluation separation |
| Alternative validation | An external benchmark is acceptable if independence from the exact checkpoint's training data is established | Evidence for that independence; PhraseBank is otherwise supplementary |
| Model comparison | Macro-F1 difference with paired uncertainty; accuracy and McNemar reported separately | Sentence/article cluster unit, resampling settings, class balance, and threshold fitting |
| News universe | Measure average tone in the chosen collection, not all investor sentiment | Company/source concentration, duplication, availability, coverage stability, and sample window |
| Primary market model | FinBERT tone at session t versus SPY return from close(t) to close(t+1); two-sided null of zero coefficient | Exact controls and transformations, eligible observations, standardization, and price convention |
| Control-set starting point | Return, intraday range variance, and log trading volume available at session t, following the short original specification | Freeze the final definitions before examining sentiment-return significance |
| Inference | Justified HAC with a prespecified bandwidth and sensitivity check | Handling of missing trading sessions and dependence; no automatic claim that retained-row lags equal trading-day lags |
| Multiplicity | Separate primary claims from enumerated secondary families | Exact family membership, correction and dependence justification; whether RQ4 targets dispersion or a joint null |
| Timing placebo | Descriptive timing diagnostic | Exact algorithm and output labels; no conditional-null p-value without a justified procedure |
| Practical absence | A prespecified smallest effect of interest and a precision assessment | Margin and rationale; pointwise versus joint claims; allow inconclusive intervals |
| Mathematical demonstration | Derive attenuation and its assumptions, then demonstrate behavior under controlled changes | One bounded simulation question, data-generating process, parameter grid, seed, and interpretation |

Independent human annotation is an external dependency. Provide a manageable labeling sample and instructions; do not use the evaluated model's own labels as ground truth or describe model-generated labels as independent human judgments. Ideally obtain a second annotator on a subset; if unavailable, report that limitation. Separate calibration and evaluation by unique text or article group before tuning thresholds.

Selecting a dataset, window, control transformation, or testing family must not depend on which version gives the desired p-value. If a feasibility audit requires a protocol change, document why and what results had been seen.

## 4. Execution roadmap

Every row is a planned increment with a 15-30-minute work limit. File lists identify expected touch points, not permission for a broad refactor. `tests/test_inference.py` and `tests/test_validation.py` are small supporting test files if needed, not new application layers. Paths described as planned artifacts do not exist merely because they are listed here.

### Phase A: Specify and audit before substantive analysis

| ID | Bounded task | Expected files/artifact | Dependency and completion check |
|---|---|---|---|
| B01 | Write the independent-validation protocol: rubric, sampling approach, calibration/evaluation separation, and label provenance | Planned `docs/validation-protocol.md`; decision-log progress entry | Uses accepted scope. Finish with an executable annotation/evaluation procedure and explicit human-labeling requirements; source-specific details may await B03-B04. |
| B02 | Write the inference protocol: primary model, families, HAC, placebo role, effect scale, practical margin and precision method | Planned `docs/inference-protocol.md` | Every intended claim has an estimand and a corresponding uncertainty procedure. Data-dependent feasibility checks are identified rather than silently assumed. |
| B03 | Audit one bounded news sample and its documentation; compare a second candidate in another instructed session if needed | `notebooks/01_data_audit.ipynb`; source notes | Check publication versus update/collection time, timezone evidence, duplication, company/source concentration, and coverage. Output verified facts and unknowns, without return-significance comparisons. |
| B04 | Select and record the feasible universe/window; identify and pin actual source/model/dictionary artifacts as available | `config.py`; data-audit records and provenance documentation | Depends on B03. Values must correspond to inspected/downloaded artifacts, not invented identifiers. If provenance remains unresolved, record it and stop dependent scoring. |
| B05 | Specify the exact timing and missing-data contract; review whether any interface change is needed | Timing section of the protocol; relevant `src/data.py` / `src/align.py` interfaces reviewed | Define actual closes, date-only mapping, sample edges, missing rows, and full-calendar lags. Any required architecture decision stops here before implementation. |

### Phase B: Repair timing in isolated patches

| ID | Bounded task | Expected files/artifact | Dependency and completion check |
|---|---|---|---|
| B06 | Correct contemporaneous controls being shifted after zero-news exclusions | `run_all.py`, relevant function in `src/inference.py`, focused inference test | Depends on B05. If Tuesday has no news, Wednesday still uses Tuesday's market controls. Preserve the complete calendar through feature construction. |
| B07 | Use actual session closes, including half-days | `src/align.py`; calendar helper in `src/data.py` if necessary; `tests/test_alignment.py` | Depends on B05 and any required approval. Verify a headline after a 13:00 close moves to the next session; retain normal-close, weekend, holiday, and DST cases. |
| B08 | Protect sample boundaries and missing market sessions | `src/align.py`, relevant market loading checks, alignment tests | Depends on B06-B07. Pre-window history cannot accumulate on day one; missing prices cannot silently make a one-day lead mean two trading sessions. Split into two increments if needed. |
| B09 | Implement the date-only fallback if the audit selects it | Relevant `config.py`, loading/alignment path, `run_all.py`, focused tests | Conditional on B03-B05. Confirm deferred availability mapping and absence of RQ2 output. If unused, record that status rather than claim the currently inert flag works. |

### Phase C: Prepare validation and reliable scoring

| ID | Bounded task | Expected files/artifact | Dependency and completion check |
|---|---|---|---|
| B10 | Resolve the validation-data loader compatibility issue for any retained benchmark | `src/validate.py`; `requirements.txt` only if necessary | Depends on B01. Exercise the chosen supported loading path with a small pinned artifact. Do not run a general dependency upgrade. |
| B11 | Prepare a small prediction-blind annotation sample and stable split assignments | Data-audit/validation preparation; planned annotation files and rubric | Depends on B01, B03-B04. Unique text/article groups cannot cross calibration/evaluation boundaries. Hand off labels to the user/annotator; the labeling workload is not declared complete here. |
| B12 | Expose FinBERT class probabilities or argmax predictions alongside the continuous tone score | `src/scoring.py`, `src/validate.py`, focused scorer tests | Depends on an approved contract if the core API changes. Verify class-label mapping and that neutral probability is retained when deriving classification predictions. |
| B13 | Add scoring-provenance checks and incompatible-cache detection | `src/scoring.py`, `rescore.py`, relevant metadata | Depends on B04 and an approved cache-format proposal if needed. Changing a scorer artifact or setting must not silently reuse incompatible scores. |
| B14 | Add batch checkpoints and interruption-safe resumption to the existing scoring workflow | `src/scoring.py`, `rescore.py`, focused cache tests | Depends on B13. A tiny interrupted run resumes completed work correctly and produces the same final scores as an uninterrupted run. No full corpus run yet. |
| B15 | Implement paired classification uncertainty and consistent subset handling | `src/validate.py`, `tests/test_validation.py` | Depends on B01-B02 and B12. Use synthetic paired predictions with known behavior; check calibration/evaluation separation and label McNemar as an accuracy test. Actual evaluation waits for independent labels. |

### Phase D: Make the panel and inference match the claims

| ID | Bounded task | Expected files/artifact | Dependency and completion check |
|---|---|---|---|
| B16 | Reconcile volume naming and the price/context-plot contract | `src/data.py`, `src/align.py`, affected consumers, `src/plots.py` | Depends on an approved data-contract proposal if schemas change. Keep the patch bounded; split naming and context-price repairs into separate increments if necessary. Verify names match formulas and date-aligned prices reach the figure. |
| B17 | Implement or repair the primary next-day regression on a controlled fixture | `src/inference.py`, focused inference tests | Depends on B02, B06-B08 and B16. Verify target horizon, controls, exclusions, full-calendar meaning, and reported sample size before using real results. |
| B18 | Implement the declared secondary-family correction and scorer-comparison outputs | `src/inference.py`, selected output assembly | Depends on B02 and B17. Confirm the exact reported family is corrected and comparisons use identical observations. A superiority claim requires paired uncertainty; side-by-side t-statistics are insufficient. Split a formal paired-comparison implementation into its own session if needed. |
| B19 | Correct the timing diagnostic's algorithm, terminology, and outputs | `src/inference.py`, `run_all.py`, relevant figure annotation | Depends on B02. Demonstrate the specified transformation on a small known series; no assumption-free or valid conditional-null p-value claim is attached to a descriptive shuffle. |
| B20 | Implement standardized effects, precision reporting, and permitted null interpretations | `src/inference.py`, result tables and focused checks | Depends on B02 and B17. Verify bps conversion and interval endpoints; distinguish practical-smallness from an inconclusive interval. Remove profitability conclusions based only on a cost threshold. |

### Phase E: Demonstrate the mathematics and run bounded analysis

| ID | Bounded task | Expected files/artifact | Dependency and completion check |
|---|---|---|---|
| B21 | Write the selected derivation and a compact simulation specification; implement the simulation in a separate instructed session if needed | Planned `docs/mathematical-appendix.md`; a small reproducible simulation artifact using the existing project structure | Depends on B02. Show attenuation assumptions and the scaling counterexample; any simulated coverage claim is checked against the known generating process. Label simulated quantities clearly. |
| B22 | Run a small scoring pilot and extrapolate compute/storage requirements | Existing `rescore.py`; timing/provenance notes | Depends on B04 and B12-B14. Verify actual artifact identities, complete score coverage, truncation behavior, and recovery before scaling. Present measured runtime and a bounded next chunk. |
| B23 | Score one explicitly bounded, resumable portion of the selected corpus | Existing scoring workflow and approved local cache | Depends on B22. Preserve complete daily headline sets for any analyzed days. Stop at a durable checkpoint. Repeat only in later instructed increments until the chosen corpus is covered. |
| B24 | Produce the independent classification results | `notebooks/02_validation.ipynb`, classification tables/figures | Depends on B11 labels, B12 and B15. Check label provenance, frozen thresholds, paired intervals, class balance, and agreement reporting. If labels are unavailable, report that dependency; do not substitute evaluated-model labels. |
| B25 | Produce the core daily panel and primary market results | Existing panel workflow, `notebooks/03_signal.ipynb`, primary tables/figures | Depends on completed required scoring, timing repairs, and B17-B20. Verify availability cutoffs, analysis counts, outcome horizon, precision, and absence of predetermined result text. Secondary empirical analyses use a separate instructed session. |

### Phase F: Reproduce and present the finished core

| ID | Bounded task | Expected files/artifact | Dependency and completion check |
|---|---|---|---|
| B26 | Integrate the selected Act 1 and Act 2 outputs into the documented reproduction path | `run_all.py`, validation integration, existing notebooks as needed | Depends on B24-B25. One documented workflow produces every promised core output from its stated inputs. Separate expensive scoring from inexpensive result reproduction. Split integration changes if they exceed the increment. |
| B27 | Write evidence-based captions, the short report, and the README | `src/plots.py`, `report/report.md`, `README.md`, mathematical appendix | Depends on actual results. Match every claim to an output; use accurate tone/dispersion/variance terminology and distinguish association, uncertainty, and limitations. Edit one coherent presentation slice per session. |
| B28 | Perform one clean-directory reproduction check and reconcile the progress log | Prepared clean working directory; documented commands; decision log | Depends on B26-B27. Use a file copy or a human-prepared clone, not Git write commands. Run the promised workflow with documented inputs; report missing prerequisites and actual outputs. Do not claim a clean run if labels/data/artifacts are unavailable. |

## 5. Architecture approval gates

The following are potential contract changes to evaluate during the named increments, not architecture implementations authorized by this document. Prefer a bounded repair within the current interfaces where it correctly solves the problem.

| Gate | Current approach | Candidate change and purpose | Files/areas to review |
|---|---|---|---|
| Calendar handoff, B05/B07 | Session dates are passed around and a helper constructs a fixed close | Supply or obtain actual exchange closes without introducing a new calendar layer; if a core signature must change, present it first | `src/data.py`, `src/align.py`, `run_all.py`, alignment tests |
| Classification interface, B12 | `score()` exposes a scalar tone score; classification needs more information | Add a probability/prediction path while retaining the scalar path, so Act 1 can use actual class predictions | `src/scoring.py`, `src/validate.py`, scorer/validation tests |
| Cache provenance, B13-B14 | Headline-keyed scores lack settings/version identity and durable batch progress | Add provenance validation and resumable batch writes to the existing local cache; no new cache service | `src/scoring.py`, `rescore.py`, local cache metadata and tests |
| Panel contract, B16 | Volume is named turnover; the context figure expects a price absent from the panel | Propose a consistent field mapping and aligned price input with a bounded consumer update | `src/data.py`, `src/align.py`, `src/inference.py`, `src/plots.py`, `run_all.py`, affected tests |

Before changing a core contract, present the concrete current behavior, proposed behavior, benefit, affected files, compatibility implications, and smallest implementation slice. Then stop and ask: **“Do you want me to proceed with this architecture change?”** Wait for explicit approval. If the proposal consumes the session, implementation belongs to a later instructed increment.

General acceptance of the research fixes, or approval to start B01, does not waive these gates. No new database, queue, service, framework, or major application module is proposed.

## 6. Verification and completion criteria

Use meaningful checks tied to the defect or claim being changed. Documentation edits need link/status/consistency checks, not a new test suite. Code changes get focused relevant tests; broaden testing only for new integration concerns or failures.

The core is complete when:

- Evaluation labels are independently sourced, calibration is separate, and the primary classifier difference has correctly paired uncertainty.
- Source, window, dictionary, and checkpoint provenance are recorded; the measured collection and its coverage limitations are explicit.
- Actual session cutoffs, boundaries, missing market rows, and news exclusions cannot silently change the intended horizon.
- Regression claims match their estimands, samples, inference assumptions, and testing families.
- Effect intervals support the stated conclusion, including an inconclusive result where appropriate.
- The mathematical artifact explains assumptions and reproduces its controlled demonstration.
- The documented reproduction workflow produces all selected outputs from available, documented prerequisites.
- The README and report state supported results without vocabulary/context causal attribution, investor-disagreement overclaims, or profitability claims from regression slopes.

Do not run a full scoring job or broad environment migration as a side effect of a small fix. Measure workload first. Do not promise a fixed total completion time until data access, annotation effort, and the scoring pilot are known.

## 7. Conditional extensions

After the core is working, separately decide whether to run additional return horizons/scorers, RQ4 volume/range-variance analyses, and the selected robustness exhibits. Define families before inspecting their results. Explain the bounded-score mean/dispersion constraint and describe RQ4 as an extension rather than a direct replication of the cited pessimism result.

Chronological forecasting against a controls-only baseline, a formal additional temporal-resampling procedure, a dashboard, and a single-name study remain optional. A trading-strategy backtest is not required. These conditional proposals were accepted as conditional; they do not expand the initial core automatically.

## 8. Coverage of the accepted proposals

| Proposal(s) | Implementation home |
|---|---|
| P01: Question and two-act framing | Current scope; B02; B27 |
| P02-P04: Independent validation, paired metrics, overlapping subsets | B01; B10-B12; B15; B24 |
| P05: Remove causal classifier-ladder attribution | B15 output labels; B27 |
| P06-P07: Conditional attenuation argument and valid scorer-effect comparison | B02; B18; B21; B27 |
| P08: Correct the temporal placebo | B02; B19; optional formal procedure only if justified |
| P09-P10: Testing families and interval scope | B02; B18; B20; B27 |
| P11-P14: Closes, gaps, boundaries, timestamp provenance, date-only fallback | B03-B09 |
| P15: Meaning of prediction and horizon | B02; B17; B25; B27 |
| P16: Conditional chronological forecast evaluation | Section 7; not required for the core |
| P17: Model availability and retrospective interpretation | B04; B22; B27 |
| P18-P19: Precision, practical absence, economic interpretation | B02; B20; B25; B27 |
| P20-P22: Corpus meaning, dispersion, outcome naming | B03-B04; B16; B27; RQ4 branch in Section 7 |
| P23-P25: Open outcomes, supported captions, transparent corrections | Every increment's log; B02; B27 |
| P26: Complete reproduction | B26; B28 |
| P27: Classification predictions | B12 |
| P28: Cache identity and resumption | B13-B14; B22-B23 |
| P29: Context-figure contract | B16 |
| P30: Loader compatibility | B10 |
| P31: Resolve source/window/artifact metadata | B03-B04; B22 |
| P32: Mathematical demonstration | B21 |
| P33: Narrow scope, presentation, realistic workload | Sections 1-2 and 7; B22; B27 |

## 9. Current stopping point

Completed before this plan: the research review, proposal log, acceptance record, and initial scope draft. Completed now: this implementation plan and its documentation links/reference copy. None of B01-B28 is marked implemented by writing this document.

**Next action after the user's go-ahead: B01, the independent-validation protocol, then stop.** No analysis-code changes, data acquisition, scoring, architecture implementation, or Git writes are authorized by the current planning-only request.
