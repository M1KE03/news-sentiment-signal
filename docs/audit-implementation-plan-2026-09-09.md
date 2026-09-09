# Audit implementation proposal — 2026-09-09

Based on the [project audit](project-audit-2026-09-09.md). This repair sequence sits within the existing B01–B28 roadmap. Initially prepared for review; the user subsequently requested execution in bounded increments.

## Execution checkpoint — 2026-09-09

**R01c specification complete; architecture approval pending (latest).** The [checkpoint contract](scoring-checkpoint-contract.md) proposes embedding provenance in the score Parquet file, one replacement commit point, a process-held writer lock, explicit legacy-cache refusal and a narrow validated reader update in `run_all.py`. It specifies metadata fields, interruption states, recovery and R01d acceptance tests. A disposable Arrow metadata round-trip succeeded; no production code changed. R01d implements the proposal after approval under the existing architecture gate.

### Previous checkpoint: R01b

**R01b completed and fixture-tested.** Default calls validate all configured scorer identities even when every value is cached. LM/VADER identities require their local lexicons; FinBERT's expected identity uses its pinned configuration without importing torch or loading weights. Any loaded scorer is checked against the preflight identity before writes. Explicit scorer subsets validate those selected measurements.

The cache parquet now carries `text_sha256`, the SHA-256 of the exact UTF-8 scoring text. The returned score frame retains its existing columns. Populated legacy caches without text identity require rebuilding to a new path; their identity is never inferred from current text. Changed text refuses reuse by default; explicit rescore invalidates all scorer values for the affected IDs. Changed scorer fingerprints invalidate that entire cached column, including rows outside the request. All selected identities are checked before any scoring/checkpoint, and duplicate/null IDs are rejected.

Verification: eight new regression cases, including default completed-cache checks without model loading, selective scorer construction, raw-text change detection, missing text identity, preflight ordering and interrupted subset resumption. Full suite **141 passed, 7 skipped** (the existing PyTorch DLL block); `git diff --check` passed. No corpus scoring or Git writes.

R01c is now specified above. Data and metadata remain separate writes until R01d implements the approved contract. Full input and scorer checks in the analysis runner remain R04b/R15. Broader fingerprint completeness (e.g. VADER lexicon-content hashing and implementation versioning) remains an audit follow-up.

### Previous checkpoint: R01a

**R01a completed and fixture-tested.** Fingerprints now use their JSON representation for comparison; FinBERT records the constructor's revision rather than rereading global config; populated cache columns with missing/null/empty identities are rejected before scorer construction. New scorers must supply a nonempty JSON-compatible identity. Unknown-provenance caches require original metadata or a rebuild into a new cache path; they are not relabelled by the rescore option.

Verification: 13 new test cases; scorer suite **34 passed, 7 skipped**; full suite **133 passed, 7 skipped**. The seven skips remain the Windows PyTorch DLL block. Revision tests use mocked model/tokenizer loaders and run without PyTorch. `git diff --check` passed. No corpus scoring, environment changes or Git writes.

At the end of R01a, default-path validation, input-content identity and subset invalidation remained open; those are now addressed by R01b above. Two-file checkpoint consistency remains R01c/R01d.

**Recommendation:** start with cache identity repair, not B22's scoring pilot. Meanwhile, the annotation-preparation and inference-fixture work can proceed independently of the PyTorch blocker once their own contracts are settled. Human labels and working model inference are dependencies for results, not reasons to postpone all code work.

## Working constraints

- Keep the current modules and local-file architecture. No new service, dashboard, model training or trading system.
- Use the project's 15–30-minute reviewable increments. Rows below with suffixes are separate increments; split further if necessary. These are scope targets, not promised completion times.
- No Git writes. Record changed files, focused verification, outstanding issues and the next increment for human review.
- Preserve the frozen universe/window. Any corpus-membership or scientific-protocol change receives a dated record of its reason and what outputs had been inspected.
- Contract proposals precede consequential cache, panel or uncertainty-interface changes. They should name exact fields and failure behavior so approval can be concrete.
- No full scoring, environment/security changes or empirical analysis are bundled into an infrastructure repair.

## Proposed increments and acceptance checks

| Increment | Work and likely files | Acceptance check | Dependencies / audit findings |
|---|---|---|---|
| R01a — cache identity | `src/scoring.py`, `tests/test_scoring.py`: canonical JSON-compatible fingerprints; actual constructor revision; missing-metadata refusal | Same identity survives save/load; changed identity and legacy unknown identity fail; no model download required | Reopen B13; A01 |
| R01b — cache validation scope | Validate completed columns on the default path; handle input content identity and subset invalidation | Complete-cache reuse checks provenance; changing A in an A/B cache cannot bless B's old values | R01a; A01 |
| R01c — checkpoint contract | Write exact proposal for versioned data/provenance commit and recovery within the existing cache | Defined interruption states, migration behavior and no-GPU read-validation path | B13/B14 contract checkpoint; A02/A15 |
| R01d — durable recovery | Implement approved checkpoint contract; inject failures at file-write/commit boundaries | Every interrupted state recovers the last consistent checkpoint or refuses explicitly; resumed scores equal uninterrupted scores | R01c; A02 |
| R02 — assembly contract | `data/raw/download.py` plus data tests: raw/clean destination separation, pinned revision, stream digest/length, atomic final publication | Small mocked download cannot overwrite the clean corpus as raw; wrong digest refuses publication; offline deterministic test | B04/B26; A05 |
| R03a — dedup lineage | Specify representative ordering, group lineage and unioned ticker tags; quantify impact on small fixtures | Repeated multi-ticker stories retain provenance and correct tags; grouping semantics explicit | B04/B11; A12 |
| R03b — dedup repairs | `src/data.py`, `tests/test_data.py`: later exact-repeat windows, stable ties, 20-token exposure boundary | Repeated clusters handled correctly with near-dedup on/off; boundary exposure correct | R03a; A12 |
| R03c — census correction | `src/audit.py`, audit tests/docs: derive all counts from one session assignment | Totals reconcile; real saved corpus reports 869,114 assigned headlines and one boundary zero; yearly window decision reviewed without outcome fitting | A06; independent of model setup |
| R03d — corpus verification checkpoint | Run the repaired raw-to-clean path only after membership effects are reviewable; persist manifest and lineage | Compare IDs/counts against old corpus, explain changes, revalidate coverage and record decision before replacement | R02/R03a–c; required before final annotation draw |
| R04a — market/calendar repair | `src/data.py`, `src/align.py`, loader/panel tests: calendar before returns, inclusive end handling, explicit warmup | Missing Tuesday makes Wednesday's return undefined; no one-session target spans a gap; final session requested; warmup does not expand analysis dates | Reopen B08, prepare B16; A04 |
| R04b — score-to-panel gate | `src/align.py`, `run_all.py`, alignment tests: unique IDs, complete finite scores, required scorer set, artifact validation | Partial values, missing rows, duplicate IDs and incompatible measurements fail before aggregation | R01, A03/A15 |
| R05 — protocol amendments | `docs/inference-protocol.md`, `docs/validation-protocol.md`, decision log | Resolve precision approximation, overlapping conclusions/boundaries, classifier multiplicity, group accuracy uncertainty, shift domain and training-overlap wording | A09/A10; no real results required |
| R06a — blind sample preparation | Existing data/validation modules, new `tests/test_validation.py`, `data/annotation/` artifacts | Stable group IDs and stored calibration/evaluation split; no overlap; prediction-blind exports; 60-item calibration pilot identified | B11; R03/R05; no labels required to build tooling |
| R06b — label ingestion and paired metrics | `src/validate.py`, validation tests: validate labels/provenance, fit calibration-only thresholds, paired group bootstrap | Known paired fixtures; identical predictions yield zero difference; all resamples preserve groups/pairing; unusable items counted | B15; R05/R06a; empirical use waits for human labels |
| R07a — panel fields | Concrete B16 field mapping, then bounded rename of volume/range fields and consumers | Formulas match names; adjusted prices reach context figure; obsolete consumers fail tests | A07; agreed panel contract |
| R07b — primary fixture | `src/inference.py`, inference tests: explicit eligibility, detrended-volume control, standardization and counts | Hand-checkable design matrix/target; identical retained rows determine fit, SD and bps scale; exclusion ledger reconciles | B17/B20; R04/R05/R07a |
| R07c — uncertainty spacing | Specify session-indexed HAC behavior and implement after review; diagnostics keep time meaning | Contiguous case agrees with library calculation; gapped fixture agrees with independently calculated lag products | B17/B20 extension; A09 |
| R07d — precision/results contract | Advance planning estimate, pointwise CI, two evidence dimensions or approved conclusion precedence | Null, small-nonzero, wide and exact-margin fixture intervals classified consistently; no profitability field | B20; R05/R07b–c |
| R08a — return families | Assemble one primary and the exact 14-test secondary family with BH/BY | Membership/size asserted; primary excluded from adjustment; correction matches reference values | B18; R07 |
| R08b — paired scorer effect | Implement standardized common-sample comparison with stacked HAC covariance | Identical scorers yield zero contrast; rescaling a score leaves standardized comparison unchanged; covariance checked on a controlled fixture | B18; R07 |
| R08c — timing diagnostic | Replace block-resampling p-value path with specified circular shifts and percentile output | Each shift uses each value once; coefficient/rank semantics checked; no p-value label in outputs | B19; R05/R07 |
| R09 — honest current documentation | Reconcile handover/current-plan status, decision-log tail, README/notebook/report framing and neutral plot titles | No stale “B01 next”, unsupported `--all`, predetermined finding, wrong timing rule or completed-output claim | B27 preliminary pass; A14; can be done early |
| R10 — preflight and pilot readiness | Dependency inventory, separate unit/integration checks, explicit artifact prerequisites; bounded CLI pilot/chunk controls | Missing packages/artifacts fail clearly; genuine model-code failures cannot silently skip; chunk limits/checkpoint location recorded | B22 readiness; A13/A15; no security change |
| R11 — pilot and bounded scoring | Measure actual model throughput, token truncation and checkpoint recovery; then separately authorized complete-session chunks | Measured rate/storage/truncation; verified fingerprints; complete per-session coverage; durable stop/resume | B22/B23; R01–R04/R10; usable model environment and dictionary |
| R12 — mathematics | Write attenuation/scaling derivation, then one bounded reproducible simulation | Known generating process; conditional claims and simulated quantities explicit; no inference from simulation to observed market effect | B21; can proceed before real scores |
| R13a — empirical validation | Run frozen independent evaluation and paired uncertainty | Human-label provenance, frozen thresholds, class balance, agreement limitations and interval outputs verified | B24; R06/R11 and human labels |
| R13b — empirical primary analysis | Build verified panel; record precision estimate before coefficient inspection; run primary plus separately selected secondary work | Eligibility, timing, scale, pointwise uncertainty and manifest complete; no same-day claim | B25; completed required scoring and R07/R08 |
| R14 — integrated reproduction | Runner + thin notebooks, run manifests, staged output publication, supported acquisition commands | Documented workflow rebuilds every promised core table/figure from stated inputs; stale panel or incomplete run cannot appear current | B26; R13 |
| R15 — presentation and clean-directory check | Evidence-based README/report/captions, dependency record, clean copy verification | Every empirical claim links to an output; actual reproduction succeeds or precisely names missing prerequisites | B27/B28; R14 |

R08's full secondary family remains the accepted specification if retained. If the initial deliverable is narrowed to primary-only results, record that scope decision before inspecting results; do not silently publish a selected subset of the 14 as the complete family.

RQ4 implementation is a separate optional increment after the core: standardized regressors, joint HAC Wald tests, exact exploratory-family membership and descriptive component intervals. PhraseBank loading is also a separate conditional increment. Neither needs to delay cache, panel, annotation or primary-model repairs.

## Concrete first increment

**R01a is my recommended starting task.** Change `src/scoring.py` and focused scorer tests only. Normalize fingerprint key/value representation; record the revision actually passed to the scorer; reject existing populated scores whose identity is absent. Verify the three corresponding failure cases with fake scorers, so neither PyTorch nor the dictionary is needed.

Stop with a reviewable diff and test results. R01b then closes the default-call bypass and unsafe subset invalidation. This corrects the handover's most consequential overstatement before any costly scoring creates artifacts dependent on it.

## What I can implement without the external blockers

Cache repairs; pinned acquisition tooling on mocked inputs; census and dedup provenance; market-loader tests; complete-score gates; protocol amendment drafts; annotation tooling; synthetic paired validation; panel and inference fixtures; precision and family logic; the mathematical demonstration; current documentation; preflight and reproduction scaffolding.

External dependencies remain a usable FinBERT execution environment, the actual LM dictionary and release identity, and independent human labels. Market acquisition also needs yfinance installed in the selected environment and a recorded download. I can prepare and validate those workflows; none of their empirical outputs is claimed complete here.

The repair program spans multiple reviewable sessions. A credible estimate for full scoring and annotation should follow the measured pilot and annotator throughput, rather than repeat the original one-week estimate.
