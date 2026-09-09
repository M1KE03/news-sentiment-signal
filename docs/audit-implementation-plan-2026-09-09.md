# Audit implementation proposal — 2026-09-09

Based on the [project audit](project-audit-2026-09-09.md). This repair sequence sits within the existing B01–B28 roadmap. Initially prepared for review; the user subsequently requested execution in bounded increments.

## Execution checkpoint — 2026-09-09

**R03b complete (latest). R01a–R01d, R02, R03a, R03c, R04a, R04b and R05 complete.**

### Environment: the scoring blocker has cleared

`torch 2.14.0+cpu` and `transformers 5.16.1` now load and FinBERT runs. The
seven tests that had skipped since B22 execute for the first time, so the suite
reports **0 skipped**. **B22, R10 and R11 are unblocked.** Two external
dependencies remain: the Loughran–McDonald dictionary and a named annotator.

On its first real execution one of those tests failed, and it was not a code
defect. `test_finbert_gets_the_documented_hard_cases_right` asserted that
FinBERT would *not* call "Costs fell sharply in the third quarter" negative —
Exhibit A's premise that a context model sees what a word counter cannot.
FinBERT calls it **negative with P = 0.932**, and calls "Profit warning smaller
than feared" **negative with P = 0.924**. Both are the cases the fixture was
built to demonstrate, and on both FinBERT agrees with the word counter it was
supposed to beat.

The B12 label-order pin is confirmed **correct** against the real checkpoint —
"shares plunge" → negative, "profit beats" → positive — which had never actually
been executed. What failed was a test asserting a research expectation on six
invented sentences before Act 1 has run: the same predetermined-finding pattern
P24 removed from the figure titles. On the user's instruction the assertion was
**removed rather than inverted** and replaced with a characterization record
pinned to `config.FINBERT_REVISION`, which detects a changed model, revision or
label mapping and claims nothing about quality. Six invented sentences measure
nothing in either direction; Act 1 measures classification quality on
independently annotated corpus text. The same unverified premise still stands in
`README.md:5` and `report/report.md:12` and is R09/B27 scope.

### R03b — dedup lineage implemented

*R03b (A12)* — implements the [dedup lineage contract](dedup-lineage-contract.md)
in `src/data.py`, with `data/raw/download.py` writing the lineage artifact after
the clean corpus so a crash leaves a corpus with missing lineage rather than
lineage describing a corpus that was never published. **216 passed, 0 skipped**;
9 new tests cover the contract's eight acceptance checks plus the refusal of an
unnumbered frame.

Verified on the real 1,412,524-row corpus, read-only, nothing written to `data/`:

- **Order invariance achieved.** Permuting the corpus and re-running now gives a
  membership symmetric difference of **0**, with **0** differing representatives
  and **0** differing tag sets. Before R03b the same permutation changed 2,363
  ids and the retained tags of **93,552 survivors (10.78%)**.
- **Exact drops 539,087**, matching R03a's prediction exactly; near drops
  **4,243** against a predicted 4,245, the small difference being that R03a's
  probe held the old unstable sort while this uses the total order.
- **Tag recovery:** distinct tickers 5,707 → **6,235**, tag slots 869,205 →
  **1,386,040**, top-10 share 2.09% → **2.04%**, effective names 1,839 →
  **1,809**. Concentration is marginally lower, so D4 is unaffected.
- **Lineage integrity:** rows equal raw rows; the dedup rate recomputed from
  lineage alone equals the reported rate to six decimal places; `headline_id`
  remains unique on survivors, so R04b's gate is untouched.
- **Blocking boundary** now 20 at `overlap=0.90`, exposure **7.70%** measured
  over the exact-survivors the near pass actually sees — its correct denominator,
  so not directly comparable to R03a's 6.98% over all raw rows.

Membership against the saved artifact differs by **2,557 ids (0.29%), net −11**.
R03a's headline figure of 93 measured the window repair alone, holding the old
sort fixed; the larger number is the combined effect of the window repair and
the total-order rule replacing the unstable sort. Both measure different things
and both are correct.

**One design decision came out of implementation, not specification.** The first
implementation asserted that every eliminated row points at a survivor. That
assertion **failed on the real corpus**: an exact repeat's anchor can itself be
removed later as a near-duplicate of an earlier headline, so elimination forms
chains — **12,074 rows, 2.2% of all eliminations**. Lineage now records the
immediate eliminator, because that is what happened, and `cluster_id` resolves
to the root, which is always a survivor. The contract and its acceptance test
were corrected to test the invariant that actually holds.

**A second defect was found by the same discipline.** The first implementation
wrote lineage to `config.DEDUP_LINEAGE_PARQUET`, an absolute path. The
acquisition test publishes into `tmp_path` — so it published a corpus there and
deposited its lineage in the **real `data/interim`**, beside the frozen corpus.
The writer now derives the path from the corpus it accompanies
(`download.lineage_path`, mirroring the existing `manifest_path`), and the
acquisition test asserts both that lineage lands beside `out` and that the
configured path stays absent.

### Previous checkpoint: R03a

*R03a (A12, with A11's missing article-group ID)* — wrote the
[dedup lineage contract](dedup-lineage-contract.md), specifying `source_row_id`,
a total representative ordering, unioned ticker tags, the re-anchored exact
window, the corrected blocking boundary and a full lineage artifact. All five
defects were reproduced against the running code and then **quantified on the
real 1.4M-row raw corpus**, read-only; nothing was written and no artifact
replaced.

What the measurement found, and it changes how R03b/R03d should be judged:

- **The saved artifact is reproducible.** `dedup()` on the pinned raw corpus
  returns the saved clean corpus exactly — 869,205 rows, symmetric difference
  **0** — and repeats identically. An earlier probe in this increment suggested
  otherwise; it had sorted with `kind="stable"` while `dedup` uses pandas'
  default, and that conclusion is withdrawn.
- **But the output is not order-invariant.** Permuting the raw corpus changes
  2,363 headline ids and changes the retained ticker tags of **93,552 of
  868,023 surviving headlines (10.78%)**. `dedup` sorts on `ts_utc` with an
  unstable quicksort, and in a date-only corpus every same-day duplicate group
  is a tie on that key. Reproducibility currently rests on the raw file's row
  order, not on any stated rule.
- **Tag loss is large:** 501,957 of 663,074 distinct (group, ticker) pairs are
  destroyed — **75.7%** — across 96.8% of multi-row groups.
- **Its effect on the universe justification is negligible and benign.**
  Recomputing with tags unioned moves the top-10 share from 2.09% to **2.03%**
  and effective names from 1,839 to **1,810**. Concentration is marginally
  *lower*, so the D4 decision is unaffected. Reported because the check had to
  be reported whichever way it came out.
- **The published exact/near split is wrong.** Re-anchoring gives
  **539,087 exact / 4,245 near**, not 431,602 / 111,717 — **96.2% of the
  reported near count is actually exact duplication** — while corpus membership
  moves by only **93 headlines (0.011%)**.
- **The boundary defect is confirmed and bites at the configured threshold.**
  An exact integer formulation gives 20 at `overlap=0.90` where `ceil` gives 21,
  omitting **14,494 rows** at exactly 20 unique tokens; exposure rises from
  5.956% to 6.982%.
- **`headline_id` is a group key, not a row key** — 518,332 raw rows share an id
  with another row — so lineage needs the new `source_row_id`.

The contract also separates two things the audit had merged: **dedup clusters**
(windowed, corpus-wide, for lineage and tag union) and **article groups**
(unwindowed, computed within the drawn annotation sample at R06a, for leakage
control and the bootstrap resampling unit). Leakage only matters between the
calibration and evaluation parts, both subsets of the ~800-item sample, so
all-pairs comparison there is exact and instant and no corpus-wide clustering
is needed.

Verification: documentation and read-only measurement only. No production code
changed — **199 passed, 7 skipped**, unchanged. §7 of the contract asks for four
decisions before R03b implements it.

### Previous checkpoint: R05

*R05 (A10/A09)* — six amendments to the two frozen protocols, plus one deferral, all
made while **no text had been scored, no labels collected, no panel built and no
coefficient estimated**. That condition is what makes them corrections rather than
post hoc adjustment, and it is recorded in each document. The research question,
frozen control set, window, 5 bps yardstick, sampling design and rubric are untouched.

[Inference protocol](inference-protocol.md): **M1** withdraws the claim that a wide
`1.96·sd(r)/√n` means the study "cannot" deliver the smallness conclusion "no matter
what is estimated" — false in both directions, since controls can narrow the realised
interval and collinearity widens it — and replaces it with two advance planning
half-widths, the sharper of which fits `r ~ X` and `z(S) ~ X` separately so `beta`
is never seen. **M2** replaces three conclusion categories that overlapped (a `[1, 3]`
bps interval satisfied two, with no rule to choose) with two always-reported
dimensions, a 2×2 naming rule that includes an informative null, and a non-strict
boundary rule evaluated at 0.1 bps with boundary cases printed unrounded. **M3**
closes the 14-test secondary family at return tests only. **M6** fixes the
circular-shift domain to the retained analysis rows in session order, states that
`k` positions is not `k` calendar sessions when gaps exist, records the calendar
alternative as considered and rejected because it breaks the bijection, and
specifies midrank ties with the tie count reported.

[Validation protocol](validation-protocol.md): **M4** makes the paired **group**
bootstrap on the accuracy difference primary and demotes exact McNemar to
supplementary — §4 assigns whole article groups precisely because near-duplicates are
not independent, which is the assumption McNemar needs — printed with the group count,
the group-size distribution and an explicit validity condition. **M5** narrows
"verifiable independence from the checkpoints" to **label** independence, records text
exposure as unknown for all three scorers symmetrically, and withdraws "upper bound of
unknown tightness" for a contaminated PhraseBank score, since contamination makes a
number expected-optimistic but does not make it a bound.

**M7 (A09) is deferred to R07c by the user's decision**, against the recommendation to
prespecify session-indexed HAC now. §4 documents both conventions, marks the choice
open, and states the exposure plainly: the convention will be selected after its effect
has been measured. Two constraints bound it — R07c completes before any tone
coefficient is estimated, and the non-primary convention is reported alongside.

Verification: documentation only. No analysis code changed and no test behaviour is
affected — **199 passed, 7 skipped** before and after. The stale census median in the
handover (339, superseded by R03c's 337) and the decision log's missing R01a–R04b
history rows were corrected in the same increment; the log's closing paragraph, which
asserted both that the corpus was assembled and that "no corpus has been assembled",
is replaced by a single status statement (A14, partial).

### Previous checkpoint: R01d, R03c, R04a and R04b

*R04b (A03/A15)* — `aggregate_daily` inner-joined headlines to scores and let the
group mean skip missing values, so five headlines with one score row reported
`n_headlines = 1` and each scorer could summarise a different subset of the same
session. New `align.validate_scores` requires, and `aggregate_daily` enforces by
default, that every headline carries exactly one finite in-range score for every
configured scorer; ids must be unique and non-null on both sides, and the join is
`validate="one_to_one"` so a duplicate score row cannot multiply a headline's
weight. The join is now LEFT and every headline is mapped, because an inner join
makes an unscored headline indistinguishable from one that never existed.

For a *declared* partial pass, `require_complete=False` keeps only sessions whose
headlines are all scored and reports the rest in `.attrs["coverage"]`, so bounded
scoring produces whole sessions or none — never a subset of one. `run_all.py`
additionally refuses a score cache that declares no provenance for a configured
scorer, before any aggregation.

Verification: 9 new cases in `tests/test_alignment.py` covering the audit's exact
probe, missing values inside a present row, out-of-range scores, duplicate score
rows, the complete-coverage path, the declared-partial path, and cached scores for
headlines outside the request. Full suite **199 passed, 7 skipped**.

*R01d* — the user approved the checkpoint contract under the architecture gate.
It was already implemented in `src/scoring.py`; verification confirmed all 30
cases in `tests/test_checkpoints.py` cover the contract's stated failure states,
and both documents' "approval pending" status was corrected. No code change was
required.

*R03c (A06)* — `corpus_census` assigned sessions with the deferred rule and then
passed the rows to `coverage_profile`, which re-mapped them with the intraday
close rule. `coverage_profile` now accepts pre-assigned `sessions`, and every
census count derives from one assignment: **869,114 assigned + 91 unassignable =
869,205**, reconciling with the corpus exactly. Zero-news sessions are now split
into boundary exclusions and real outages: the corpus has **one** zero-news
session, 2010-01-04, and it is structural — under the deferred rule the window's
first session cannot receive a headline. There is **no genuine news outage in
ten years**, so the D9 exclusion is inert here. The previously reported median
of 339 and "2 zero-news sessions" were artifacts of the mapping mismatch and are
corrected in `docs/data-audit-fnspid.md` §11 and the handover. The yearly
stability table always used the correct mapping, so the D4 freeze is unaffected.

*R04a (A04)* — `load_market` computed `diff(log(close))` over the rows the price
source returned, before any calendar reindex, so a missing session produced a
two-session return in the row labelled with the later date. New
`returns_on_calendar` places prices on the exchange calendar first and requires
**both** adjacent closes, so a gap makes the returns on either side NaN rather
than wrong. `load_market` now takes `calendar` and `warmup_sessions`, adds one
day to yfinance's exclusive `end` (the configured inclusive end would otherwise
drop the final session), flags warm-up rows `in_window=False` so they cannot
widen the analysis window, and reports missing sessions in
`.attrs["market_stats"]`. `fetch_market` passes a 70-session warm-up, covering
the 63-session volume detrend plus the first row's lagged return.

Verification: 9 new offline cases in `tests/test_market.py`, including the
audit's exact fixture (Monday 100, Tuesday absent, Wednesday 102 — the naive
computation gives 0.019803; it is now NaN on both sides), the exclusive-end
request, warm-up flagging, and an end-to-end check that the contaminated value
cannot reach `ret_lead1`. Full suite **190 passed, 7 skipped**. No network
access, corpus scoring, environment change or Git write.

### Previous checkpoint: R02

**R02 completed and fixture-tested.** Acquisition now writes the raw
corpus only: `assemble_corpus` defaults to `HEADLINES_RAW_PARQUET` and raises
`AcquisitionError` if pointed at `HEADLINES_PARQUET`, so a documented command
can no longer replace 869,205 deduplicated rows with 1,412,524 undeduplicated
ones under the same name. The download URL is built from
`config.NEWS_HF_REVISION` instead of `/resolve/main/`. The stream is hashed as
it is consumed and checked against the pinned SHA-256 and byte length **before**
anything is published; a mismatch writes nothing. Publication is atomic, with
the manifest written after the data so a crash leaves data marked unverified
rather than the reverse. `--dedup` and `--census` complete the documented
assembly → dedup → census sequence, and `--dedup` refuses a raw corpus whose
manifest is absent or records no verification.

Retrospective manifests were written for the existing artifacts recording what
is actually known: they were produced from `/resolve/main/` with no digest
computed, so both are marked `verified_against_pin: false`. `--dedup`
consequently refuses to rebuild the analysis input from them, which is correct —
a verified corpus requires re-running `--assemble` (~12 min). The existing files
are not asserted to be wrong; their provenance is simply unproven.

Verification: 10 new offline regression cases in `tests/test_acquisition.py`
(HTTP replaced by an in-memory CSV), covering the clean-destination refusal,
default destination, pinned URL, wrong-digest and wrong-length refusal with no
partial writes, manifest contents, the full assemble → dedup sequence, and the
two unverified-source refusals. Full suite **181 passed, 7 skipped**. No
network access, corpus scoring, environment change or Git write.

Also fixed: `tests/test_checkpoints.py::test_hard_stop_around_commit_and_resumption[after_text]`
asserted cache contents positionally, but a checkpoint writes unrequested cache
rows before the scored frame, so on-disk order is not request order. The values
were correct by `headline_id`; the assertion now indexes by id. Only
`score_all`'s return order follows the input frame.

### Previous checkpoint: R01c

**R01c/R01d complete.** The user approved the checkpoint contract under the architecture gate; it is implemented in `src/scoring.py` and verified by 30 cases in `tests/test_checkpoints.py`. The [checkpoint contract](scoring-checkpoint-contract.md) proposes embedding provenance in the score Parquet file, one replacement commit point, a process-held writer lock, explicit legacy-cache refusal and a narrow validated reader update in `run_all.py`. It specifies metadata fields, interruption states, recovery and R01d acceptance tests. A disposable Arrow metadata round-trip succeeded; no production code changed. R01d implements the proposal after approval under the existing architecture gate.

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
| ✅ R03a — dedup lineage | **Done 2026-09-09** — [dedup lineage contract](dedup-lineage-contract.md). Representative ordering, `source_row_id`, group lineage, unioned tags, re-anchored exact window, corrected blocking boundary; dedup clusters separated from article groups | Impact quantified on the real corpus rather than fixtures: 75.7% of (group, ticker) pairs destroyed; exact/near split wrong by 96.2% of the near count; membership effect only 93 ids; output not order-invariant (10.78% tag churn). Four decisions requested in §7 | B04/B11; A12 |
| ✅ R03b — dedup repairs | **Done 2026-09-09.** `src/data.py`, `data/raw/download.py`, `tests/test_data.py`: re-anchored exact window, total representative order `(ts_utc, source_row_id)`, unioned tags, lineage artifact, corrected boundary; elimination chains resolved to a surviving root | 9 new tests, 216 passing 0 skipped. On the real corpus: order invariance verified (symmetric difference 0, was 2,363 ids and 10.78% tag churn), exact drops 539,087 as predicted, tickers 5,707 → 6,235, rate recomputable from lineage | R03a; A12 |
| R03c — census correction | `src/audit.py`, audit tests/docs: derive all counts from one session assignment | Totals reconcile; real saved corpus reports 869,114 assigned headlines and one boundary zero; yearly window decision reviewed without outcome fitting | A06; independent of model setup |
| R03d — corpus verification checkpoint | Run the repaired raw-to-clean path only after membership effects are reviewable; persist manifest and lineage | Compare IDs/counts against old corpus, explain changes, revalidate coverage and record decision before replacement | R02/R03a–c; required before final annotation draw |
| R04a — market/calendar repair | `src/data.py`, `src/align.py`, loader/panel tests: calendar before returns, inclusive end handling, explicit warmup | Missing Tuesday makes Wednesday's return undefined; no one-session target spans a gap; final session requested; warmup does not expand analysis dates | Reopen B08, prepare B16; A04 |
| R04b — score-to-panel gate | `src/align.py`, `run_all.py`, alignment tests: unique IDs, complete finite scores, required scorer set, artifact validation | Partial values, missing rows, duplicate IDs and incompatible measurements fail before aggregation | R01, A03/A15 |
| ✅ R05 — protocol amendments | `docs/inference-protocol.md`, `docs/validation-protocol.md`, decision log | **Done 2026-09-09.** M1 precision approximation, M2 overlapping conclusions and boundaries, M3 classifier multiplicity, M4 group accuracy uncertainty, M5 training-overlap wording, M6 shift domain. **M7 (HAC spacing, A09) deferred to R07c by decision** | A09/A10; no real results required |
| R06a — blind sample preparation | Existing data/validation modules, new `tests/test_validation.py`, `data/annotation/` artifacts | Stable group IDs and stored calibration/evaluation split; no overlap; prediction-blind exports; 60-item calibration pilot identified | B11; R03/R05; no labels required to build tooling |
| R06b — label ingestion and paired metrics | `src/validate.py`, validation tests: validate labels/provenance, fit calibration-only thresholds, paired group bootstrap | Known paired fixtures; identical predictions yield zero difference; all resamples preserve groups/pairing; unusable items counted | B15; R05/R06a; empirical use waits for human labels |
| R07a — panel fields | Concrete B16 field mapping, then bounded rename of volume/range fields and consumers | Formulas match names; adjusted prices reach context figure; obsolete consumers fail tests | A07; agreed panel contract |
| R07b — primary fixture | `src/inference.py`, inference tests: explicit eligibility, detrended-volume control, standardization and counts | Hand-checkable design matrix/target; identical retained rows determine fit, SD and bps scale; exclusion ledger reconciles | B17/B20; R04/R05/R07a |
| R07c — uncertainty spacing | Specify session-indexed HAC behavior and implement after review; diagnostics keep time meaning. **Also owns the deferred M7 decision**: measure retained-position vs session-indexed HAC on synthetic gapped data and on the real sample, then select the primary convention and report the other alongside | Contiguous case agrees with library calculation; gapped fixture agrees with independently calculated lag products; both conventions computed and the choice recorded with its measured difference **before any tone coefficient is estimated** | B17/B20 extension; A09; R05/M7 |
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
