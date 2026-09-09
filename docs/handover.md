# Handover

Date: 2026-09-09. Written at the close of Phase B; updated after corpus assembly, and again after the 2026-09-09 audit repairs (R01–R05, R03a/R03b).

Read this first if you are picking the project up. It records what exists, what was decided and why, what is deliberately absent, and what blocks the next step. Authority for each decision lives in the linked document; this file is a map, not a substitute.

Related: [implementation plan](implementation-plan.md) (B01–B28) · [decision log](research-review-decision-log.md) · [scope draft](protocol-revision-draft.md) · [original plan](implementation-plan-original.md).

---

## 1. Where the project stands, in one paragraph

The protocols are written, the candidate dataset has been audited, every timing defect found in the review has been fixed with a regression test behind it, and **the corpus is assembled**: **869,183** deduplicated Benzinga headlines over 2,516 trading sessions, with D4 frozen on coverage evidence. A full project audit on 2026-09-09 found fifteen issues (A01–A15); **eleven of the thirty repair increments are complete**, covering scoring-cache identity and durable checkpoints, verified acquisition, the census mapping, the market-return calendar, the score-to-panel gate, the protocol amendments and the deduplication lineage. **No text has been scored, no labels have been collected, no panel has been built, no model has been fitted, and no empirical result exists.**

**216 tests pass, 0 skip.** The seven long-standing skips were FinBERT and VADER; Smart App Control has since been disabled, so `torch` loads and every test executes.

The live plan is the [audit repair sequence](audit-implementation-plan-2026-09-09.md), which sits inside the B01–B28 roadmap below and is the authoritative execution order.

## 2. What the study is

> How do financial sentiment measurements differ in classification quality, and what additional information do they provide about subsequent market outcomes?

Two acts. **Act 1** compares FinBERT, Loughran–McDonald and VADER on independently labelled financial headlines; the primary contrast is `macroF1(FinBERT) − macroF1(LM)` with paired uncertainty. **Act 2** estimates the association between FinBERT daily tone and the *next* trading session's SPY log return, conditional on a prespecified control set, reported in basis points per standard deviation with its precision limits.

This replaced an earlier framing in which better classification was assumed to imply a stronger market coefficient. That implication is now stated as a **conditional hypothesis with its assumptions on display**, not a mechanism. The contribution does not depend on finding a signal.

## 3. What was done, in order

Nine increments, each ending with a stop for instructions. Git history: `b25bdb9` … `75c98f4` plus uncommitted work listed in §8.

### Scaffold (before the plan revision)

Built the repository against the original specification: `config.py`, six `src/` modules, `run_all.py`, `rescore.py`, README, report skeleton, notebook stubs, and the alignment/scoring tests. Superseded in parts by everything below, but the structure survives.

> **Note for the record.** A slice of Stage-0 audit work (`src/audit.py`, `tests/test_audit.py`) was written against the *original* plan before the replacement plan was discovered in the working tree. It was logged as DEV-01 and has since been fully reworked. Its original whole-file intraday gate was exactly the mistake the B03 audit later disproved.

### B01 — validation protocol → [`validation-protocol.md`](validation-protocol.md)

The estimand is written first, and everything follows from it. Uncertainty for the macro-F1 difference is a **paired bootstrap resampling article groups**, recomputing both models on the same resampled items (B = 10,000, seed 20260830) — macro-F1 is not a mean of per-item scores, so it cannot be bootstrapped from a per-item vector, and near-duplicate headlines are not independent draws. McNemar is retained but relabelled a *paired accuracy comparison*. **Amended at R05 (M4):** the primary accuracy comparison is now the same paired **group** bootstrap applied to the accuracy difference, and exact McNemar is supplementary — its independence assumption is contradicted by the protocol's own article-group design, so it is printed with the group-size distribution and an explicit validity condition.

Primary evaluation is annotated headlines from this study's own collection, with a documented fallback order; PhraseBank is contaminated for `ProsusAI/finbert` (its own model card names PhraseBank as fine-tuning data) and is supplementary only. Calibration/evaluation separation is **by article group and stored as a file**, not recomputed from a seed. Rubric v1 has ten decision rules; a 60-item pilot precedes the full annotation.

### B02 — inference protocol → [`inference-protocol.md`](inference-protocol.md)

Primary estimand and two-sided null; frozen control set; eligibility rules; HAC bandwidth `L = 5` prespecified with a sensitivity set; effect scale in bps per 1σ; **SESOI = 5 bps** with a precision check run *before* the coefficient is inspected; and three permitted conclusions including an explicit *inconclusive*. **Amended at R05 (M1, M2):** the three conclusions overlapped — `[1, 3]` bps satisfied two of them — and are replaced by two always-reported dimensions (does the interval exclude zero; is it inside ±5 bps) with a 2×2 naming rule and a non-strict boundary rule at 0.1 bps. The precision check is now two advance planning half-widths; the earlier claim that a wide one means the study *cannot* deliver the smallness conclusion "no matter what is estimated" was false in both directions and is withdrawn.

Three questions the plan had left open were settled here:

- **RQ4 targets a joint null.** HAC Wald test of `b1 = b2 = b3 = 0` across tone, intensity and dispersion; individual coefficients are descriptive afterwards. This is what stops the dispersion coefficient being promoted to the headline once someone has looked at it.
- **BH plus BY.** BH at `q = 0.05` across the 14-test secondary family, with Benjamini–Yekutieli reported alongside — valid under arbitrary dependence. Where they disagree, the claim is made at the BY level.
- **The placebo is demoted** to a descriptive timing diagnostic reported as a percentile rank, never a p-value.

### B03 — data audit → [`data-audit-fnspid.md`](data-audit-fnspid.md)

Downloaded 144 MB of FNSPID as 24 evenly spaced HTTP range slices (356,123 rows, 2.5% of 5.7 GB) plus an 8 × 4 MB probe of the 23 GB second file. **The decisive finding: `All_external.csv` is five-plus corpora concatenated**, with different languages, schemas and timestamp behaviour.

| Source | Rows | Midnight | Distinct minutes | Ticker | Span |
|---|---:|---:|---:|---:|---|
| reuters.com | 213,568 | **0.2%** | **1440** | 0% | 2007–2016 |
| benzinga.com | 57,207 | 96.7% | 697 | 100% | 2009–2020 |
| seekingalpha | 29,665 | 100% | 1 | 100% | 2014–2019 |
| **lenta.ru** | 19,075 | 100% | 1 | 0% | 2003–2018 |
| zacks / bloomberg / 13 others | ~36,400 | 100% | 1 | mixed | 2007–2020 |

Three consequences. The intraday gate must be applied **per source** — pooled, the file looks partly intraday when only one block is. `lenta.ru` is a **Russian-language general news site** present in both files. And the only intraday block is the wrong content: Reuters is a global newswire (Royal Ascot, Austrian rail politics, UK RNS filings) with no ticker tags, peaking at European hours.

Also found: the `Article` field contains **embedded newlines** (180 sampled `Date` values are body-text fragments), and FinBERT's label order is `{0: positive, 1: negative, 2: neutral}` — not the conventional order, so index-based code inverts every score.

What the sample **cannot** support, and was therefore withheld rather than estimated: dedup rate, headlines per session, company concentration. It is a cluster sample over tickers.

### B04 — universe, window, provenance → `config.py`

- **RQ2 inadmissible.** No sub-corpus is both intraday-stamped and relevant. The date-only fallback is active.
- **Universe:** Benzinga sub-corpus, chosen on relevance and coverage before any return relationship was examined. Zacks/SeekingAlpha recorded *now* as a later pooled sensitivity so that choice cannot be made after seeing a coefficient.
- **Window:** 2010-01-01 … 2019-12-31, flagged `SAMPLE_WINDOW_PROVISIONAL`.
- **Pinned:** FNSPID repo `bf9189c4…`, file sha256 `5d4c0180…` (5,731,397,037 bytes), FinBERT `4556d130…`. Licence CC BY-NC 4.0, non-commercial.
  **Corrected 2026-09-09 (R03d):** the sha256 was originally recorded as `dde52918…`, taken from the HTTP **ETag**. HuggingFace serves its `xetHash` as the ETag for Xet-backed files, so the pin held a different hash function's output under a SHA-256 name. The file itself never changed.

A guard was added to `run_all.py` so nothing could run while the fallback flag was inert.

### DEV-01 rework — `src/audit.py`

Rewritten around the **source** as the audit unit. Adds `profile_by_source`, `midnight_share_by_year`, `script_profile` (catches non-English sources), `screen_sources` (mechanical criteria only — it explicitly refuses to judge relevance), and a `CorpusRate` type that **withholds** duplication and per-session rates on a cluster sample with the reason attached.

A real defect surfaced: concentration was measured in market time, but FNSPID's date-only stamps are `00:00 UTC` = 19:00 ET, so `midnight_share` read 0%. Concentration is now measured in the source's published zone and session position in market time.

### B05 — timing contract → [`timing-contract.md`](timing-contract.md)

Three defects reproduced against the running code before being specified away:

- **D-1** `map_to_trading_day` guarded its upper calendar edge but not its lower, so *all* pre-window headlines landed on the first session.
- **D-2** `build_panel`'s inner join let a missing price row turn `ret_lead1` into a two-session return, with no warning and the column still named `ret_lead1`.
- **D-3** `trading_calendar` fell back to business days on ImportError — live in the environment at the time.

The **architecture gate was closed without opening**: `aggregate_daily` already reindexes to the full calendar, so `daily_scores["date"]` *is* the session list and `build_panel` needs no calendar argument. **B07 was descoped** for this corpus (actual session closes cannot matter when no stamp carries a time), with a recorded trigger to reopen it.

### B06 — full-calendar lags and leads

Panel now **left-joins** market onto the calendar-indexed daily frame. Full-calendar lag columns are built in `build_panel`, and `contemporaneous` reads them and raises if absent, so a zero-news Tuesday still supplies Wednesday's control instead of Monday's. `assert_sessions_match_calendar` added and called from `run_all.py`. Verified on the real NYSE calendar: 1,258 sessions over 2015–2019, 46 holidays correctly excluded.

### B07 + B08 — edges and guards

Lower calendar edge returns `NaT`, with an optional `prior_close` to populate the first session properly. `trading_calendar` raises instead of degrading. `map_to_trading_day` refuses date-only input, so the B07 descope cannot be undone by accident.

### B09a — the date-only mapper

`map_date_to_session`: a headline dated `d` goes to the **first session strictly after `d`**, so session `t` receives dates in `[prev_session, t)` and all of day `d` precedes `close(t)`. The as-if-intraday rule sits behind `defer=False`, fixed in advance as a sensitivity only.

It reads the wall-clock date **in the source's own zone and never converts** — converting a `00:00 UTC` stamp to market time moves it to the previous calendar day and shifts every headline one session early. A test documents the trap. `_reject_intraday` mirrors B07's guard, so the two mappers refuse each other's input in both directions.

### B09b — suppression, filter, flag

`load_news` rewritten: source filtering is **mandatory** and defaults to `config.NEWS_SOURCE_DOMAINS`; disabling it requires an explicit empty tuple. Malformed stamps are rejected against the documented pattern and counted separately from missing values. CSV reading is chunked for the 5.7 GB file. RQ2 suppression is **structural** — `contemporaneous` raises, `run_all.py` writes no Table 2 and deletes any stale one. `DATE_ONLY_FALLBACK_IMPLEMENTED` set True.

A defect the tests caught: the headline schema's string dtypes depended on `chunksize`, so the schema varied with a performance knob and a later merge on `headline_id` could mismatch.

## 4. Checklist against the implementation plan

Legend: ✅ done · 🟡 partial · ⬜ not started · ⛔ descoped (with trigger)

### Phase A — specify and audit

| ID | Task | Status | Evidence |
|---|---|---|---|
| B01 | Independent-validation protocol | ✅ | [`validation-protocol.md`](validation-protocol.md) |
| B02 | Inference protocol | ✅ | [`inference-protocol.md`](inference-protocol.md) |
| B03 | Bounded news audit | ✅ | [`data-audit-fnspid.md`](data-audit-fnspid.md), `notebooks/01_data_audit.ipynb`, `src/audit.py` |
| B04 | Universe, window, artifact provenance | 🟡 | `config.py`. Window **frozen** 2026-09-09 on the census; Loughran–McDonald release still **unresolved** |
| B05 | Timing and missing-data contract | ✅ | [`timing-contract.md`](timing-contract.md) |

### Phase B — timing repairs

| ID | Task | Status | Evidence |
|---|---|---|---|
| B06 | Full-calendar lags/leads; adjacency | ✅ | `src/align.py`, `src/inference.py`, `tests/test_inference.py` |
| B07 | Actual session closes | ⛔ | Descoped in [`timing-contract.md`](timing-contract.md) §6; guard in `align._reject_date_only`. Reopens if the universe becomes intraday |
| B08 | Sample boundaries, missing sessions | ✅ | `src/align.py`, `src/data.py`, `tests/test_alignment.py` |
| B09 | Date-only fallback | ✅ | `align.map_date_to_session`, `load_news` filter, structural RQ2 suppression |

### Phase C — validation and scoring

| ID | Task | Status | Notes |
|---|---|---|---|
| B10 | PhraseBank loader compatibility | ⬜ | `datasets==4.0.0` removed dataset scripts and `trust_remote_code`; `src/validate.load_phrasebank` will not run as pinned. Needed only if the §2 fallback is used |
| B11 | Prediction-blind annotation sample | ✅ | **R06a, 2026-09-09.** `src/annotate.py`; 800 items drawn into `data/annotation/`, 200 calibration / 600 evaluation by article group, blind export, 60-item pilot. **The only remaining dependency is a human annotator** |
| B12 | FinBERT class probabilities | ✅ | `scoring.Classifier`, `predict_proba`/`predict`, `validate.predictions_for`; label order checked against the pin at construction |
| B13 | Scoring-provenance / cache invalidation | ✅ | **Reopened by audit A01, closed again by R01a/R01b.** Fingerprints compare by JSON representation; FinBERT records the revision actually passed to its constructor; `text_sha256` detects changed headlines; a changed scorer fingerprint invalidates that whole cached column |
| B14 | Batch checkpoints, resumable scoring | ✅ | **Reopened by audit A02, closed again by R01c/R01d.** Scores and provenance in one Parquet file, one replacement commit point, a process-held writer lock, explicit legacy refusal — [checkpoint contract](scoring-checkpoint-contract.md), 30 tests |
| B15 | Paired classification uncertainty | ⬜ | Specified in B01 §7 |

### Phase D — panel and inference

| ID | Task | Status | Notes |
|---|---|---|---|
| B16 | Volume naming; context-plot contract | 🟡 | `close_adj` now reaches the panel (part of P29). Renames `log_turnover`→`log_volume`, `parkinson`→`rv_parkinson` outstanding — **R07a** |
| B17 | Primary next-day regression on a fixture | 🟡 | `inference.predictive` exists and runs; not yet verified against the B02 eligibility rules |
| B18 | Secondary family + paired scorer contrast | 🟡 | BH implemented; **BY sensitivity and the stacked `delta` contrast are not** |
| B19 | Timing diagnostic corrected | ⬜ | Still block-resamples with replacement; B02 §9 specifies the circular shift |
| B20 | Standardized effects, precision, null logic | 🟡 | `effect_size_bps` exists; standardization, the advance precision check and the three-conclusion rule are not implemented |

### Phase E — mathematics and analysis

| ID | Task | Status |
|---|---|---|
| B21 | Derivation + simulation | ⬜ |
| B22 | Scoring pilot and extrapolation | ⬜ **unblocked 2026-09-09** — Smart App Control disabled; `torch 2.14.0+cpu` loads and FinBERT runs. Pilot not yet measured — **R10/R11** |
| B23 | Bounded, resumable scoring run | ⬜ |
| B24 | Independent classification results | ⬜ |
| B25 | Core panel and primary market results | ⬜ |

### Phase F — reproduce and present

| ID | Task | Status |
|---|---|---|
| B26 | Integrated reproduction path | ⬜ |
| B27 | Evidence-based captions, report, README | ⬜ |
| B28 | Clean-directory reproduction check | ⬜ |

### The audit repair sequence (the live plan)

Fifteen findings (A01–A15) from the 2026-09-09 [project audit](project-audit-2026-09-09.md), repaired in the thirty increments of the [repair plan](audit-implementation-plan-2026-09-09.md). **Eleven complete, nineteen remaining.**

| Increment | Finding | State |
|---|---|---|
| R01a–R01d | A01, A02 | ✅ Cache identity, validation scope, checkpoint contract, durable recovery |
| R02 | A05 | ✅ Verified acquisition: pinned revision, stream digest, atomic publication, raw/clean separation |
| R03a, R03b | A12, A11 | ✅ Dedup lineage contract and its implementation |
| R03c | A06 | ✅ Census derived from one session assignment |
| R04a, R04b | A04, A03 | ✅ Returns on the calendar; complete-score gate before aggregation |
| R05 | A10, A09 | ✅ Six protocol amendments; M7 (HAC spacing) deferred to R07c |
| R03d | — | ✅ Verified rebuild; corpus replaced, `verified_against_pin: true` |
| R06a | A11 | ✅ Blind sample drawn: 800 items in `data/annotation/`, awaiting an annotator |
| R06b | A11 | ▶ **Next once labels exist.** Label ingestion and paired metrics |
| R07a–R07d | A07, A09 | ⬜ Panel fields, primary fixture, HAC spacing, precision contract |
| R08a–R08c | A08 | ⬜ Return families, paired scorer contrast, timing diagnostic |
| R09, R10 | A14, A13, A15 | ⬜ Documentation, preflight and pilot readiness |
| R11, R12 | — | ⬜ Pilot and bounded scoring; the mathematical demonstration |
| R13a, R13b | — | ⬜ Empirical validation and primary analysis |
| R14, R15 | — | ⬜ Reproduction and presentation |

Six findings are closed (A01–A06), A15 is partly closed, and eight remain open (A07–A14 less A12).

### Carried defects not yet fixed

| Item | Where | Increment |
|---|---|---|
| Figure titles assert conclusions before results exist (P24) | `src/plots.py` — "Same-day association, next-day nothing"; "heavier trading"; "lexicons over-predict neutral" | B27 / R09 |
| README and report assert that a transformer reads financial sentences better than a word counter, as a premise | `README.md:5`, `report/report.md:12` | B27 / R09 |
| README states findings as filled-in placeholders | `README.md` | B27 / R09 |
| Report skeleton still uses original-plan language in places | `report/report.md` | B27 / R09 |
| Placebo resamples blocks with replacement | `inference.permutation_pvalue` | B19 / R08c |
| Primary model uses raw `log_turnover`, not the protocol's 63-session detrend | `src/inference.py:95` | R07a/R07b |
| `clears_costs` frames a coefficient as strategy profitability, which the inference protocol §11 forbids | `src/inference.py:225` | R07d / R08 |
| Source filter matches by substring, so `notbenzinga.com` would pass | `src/data.py:94` | audit follow-up |
| `config.AGG` is never read by `aggregate_daily` — an inert flag | `config.py:138` | audit follow-up |
| `requirements.txt` versions are declared, not `pip freeze`d | | B28 / R10 |

**One carried defect was closed by measurement rather than by code.** A test
asserted that FinBERT would not call "Costs fell sharply in the third quarter"
negative — Exhibit A's premise that a context model sees what a word counter
cannot. It had never executed, because torch was blocked. On its first real run
it failed: FinBERT calls that headline **negative with P = 0.932**, and "Profit
warning smaller than feared" **negative with P = 0.924**. Both are the cases the
fixture was built to demonstrate. The B12 label-order pin is confirmed correct
against the real checkpoint, so this is not a code defect; the assertion was
removed rather than inverted and replaced with a characterization record. Six
invented sentences measure nothing in either direction — Act 1 measures
classification quality on independently annotated corpus text — but the premise
in the README and report is no longer free.

## 5. Corpus assembly and census — done

Streamed the whole 5.73 GB file once (13,057,514 rows, 11.5 min), storing only the filtered result. Full findings in [`data-audit-fnspid.md`](data-audit-fnspid.md) §11.

**The selected source occupies two separate blocks.** Kept rows plateaued at 0.55 GB, then resumed at 2.44 GB. Reading only the first block — which the audit's offset map would have suggested was enough — would have silently dropped ~122,000 headlines, about 9%.

| Measurement | Result |
|---|---|
| Dedup rate | **38.5%** — corrected split **539,087 exact + 4,254 near** (R03b) |
| Headlines per session | mean 345.4, median **337** (corrected by R03c) |
| Zero-news sessions | **1**, structural (the window's first session); no real outage |
| Distinct tickers | **6,235**; top-10 share **2.04%**; effective **1,809** names (R03b, tags unioned) |
| Coverage stability | no year below tolerance → **D4 frozen** |

> **Three of these figures were wrong when first published, and all three were
> corrected by measurement rather than by re-reading the code.**
>
> - The **median and zero-news count** (339 and 2) were artifacts of the census
>   assigning sessions with one mapping rule and then re-mapping with another;
>   the two disagreed on 2,502 of 2,516 sessions (A06 / R03c).
> - The **exact/near split** was 431,602 / 111,717 because the exact-duplicate
>   window anchored on a text's first-ever occurrence rather than its last kept
>   one, so later clusters of exact repeats were charged to the near count.
>   **96.2% of the reported near-duplicate count was actually exact
>   duplication** (A12 / R03b).
> - The **ticker figures** counted surviving tags, not companies covered: a
>   story filed under three tickers kept one. 75.7% of distinct (cluster,
>   ticker) pairs were being discarded. Unioning them *lowers* concentration, so
>   the D4 universe decision is unaffected and marginally better supported
>   (A12 / R03b).

`interim/headlines_raw.parquet` (1,412,524) keeps the pre-dedup corpus so the rate stays checkable; `interim/headlines.parquet` (869,183) is the analysis input; `interim/dedup_lineage.parquet` (1,412,524) records what happened to every raw row, so the rate is recomputable from the artifacts rather than only reproducible by re-running the pass.

**Both artifacts were rebuilt at R03d (2026-09-09) through the verified acquisition path**, and the raw manifest now records `verified_against_pin: true` for the first time. The raw corpus is byte-for-byte the same data as before — identical row count, identical id set, identical `text_norm` multiset — so the rebuild changed provenance and row identifiers, not content.

Two corrections came out of it, both recorded: the earlier "0.05% malformed timestamps" was an artifact of byte-range slicing, not a property of the file (the full read produced **zero**); and the window's end bound was inclusive of the following midnight, which for a date-only corpus admitted a whole extra day.

## 5a. The next increment

**R03d — the corpus verification checkpoint**, then **R06a**. R03d rebuilds raw → clean through R02's verified acquisition path with R03b's rules, diffs membership against the previous artifact, recomputes the census and concentration, and replaces the frozen corpus. It must complete **before any annotation sample is drawn and before any scoring**, because a sample drawn from a corpus that later changes is a sample thrown away.

R06a then draws the blind annotation sample and hands it to an annotator. That ordering is deliberate: annotation is the only dependency with human latency, so it starts as early as possible and the inference track (R07a–R08c, seven fixture-based increments needing nothing external) fills the wait.

**B22, the scoring pilot, is now unblocked** — `torch` loads. `rescore.py --time-only` times FinBERT on 1,000 headlines and extrapolates before a full pass is launched. If the projected time is unacceptable, the plan's rule is to shorten the *window* and record it — **never** to subsample headlines within days, which would bias both `S_t` and `d_t`. Shortening the window means unfreezing D4, which requires a dated log entry.

Still needed from outside the code: the Loughran–McDonald dictionary and a named annotator.

## 6. Blockers and external dependencies

| Blocker | Blocks | Action |
|---|---|---|
| **Loughran–McDonald dictionary** not obtained | Any LM scoring; Act 1 | Download by hand from the Notre Dame SRAF site (no stable link) into `data/raw/LoughranMcDonald_MasterDictionary.csv`, then record the release in `config.LM_DICT_VERSION` |
| **Human annotation** not started | B11, B15, B24, all of Act 1 | Needs a named annotator; ideally a second on a 20% subset for kappa. Protocol is written and executable |
| ~~Window not frozen~~ | — | **Resolved**: frozen 2026-09-09 on the census |
| ~~Smart App Control blocks `torch`~~ | — | **Resolved 2026-09-09**: the user disabled Smart App Control. `torch 2.14.0+cpu` and `transformers 5.16.1` load, FinBERT runs, and the seven long-standing skips now execute |
| `datasets==4.0.0` loader | B10, PhraseBank fallback only | Not on the critical path unless annotation fails |

## 7. Environment notes

- Python 3.12, venv at `.venv/`. `pandas 3.0.5` — note the string-dtype behaviour that produced the `chunksize` schema bug.
- **Smart App Control blocked scipy's DLLs** on this machine for several increments, making `statsmodels` and `scikit-learn` unimportable. It cleared during B06 and the inference path now runs. If it recurs, the symptom is `ImportError: DLL load failed while importing _comb: An Application Control policy has blocked this file`.
- `pandas-market-calendars` is now a **hard requirement** — `trading_calendar` raises without it rather than degrading.
- `data/raw/` is gitignored. The audit sample (`fnspid_sample.parquet`) is local only; regenerate with `python data/raw/download.py --fnspid-sample --slices 24 --slice-mb 6`.
- `torch 2.14.0+cpu`, `transformers 5.16.1` and `vaderSentiment` are installed and working. The suite has **no skips**. FinBERT downloads its checkpoint on first use, so the scorer tests need network the first time.
- **The declared environment is not the tested one** (audit A13). `requirements.txt` pins numpy 2.3.3 / pandas 2.3.2 / torch 2.8.0 / transformers 4.56.1; installed are 2.5.3 / 3.0.5 / 2.14.0 / 5.16.1. The passing suite establishes behaviour here, not in the pinned environment. R10 separates offline unit verification from model integration checks.

## 8. Repository state

The repair increments R01–R05, R03a and R03b are committed. The working tree at the time of writing carries the R03b/R03d work and this documentation pass; the standing rule is that **the assistant does not perform Git writes**, so staging and commits are the human's.

```
docs/
  implementation-plan.md            the execution specification, B01-B28
  implementation-plan-original.md   historical baseline, superseded
  research-review-decision-log.md   P01-P33, decisions, per-increment history
  protocol-revision-draft.md        revised scope
  validation-protocol.md            B01
  inference-protocol.md             B02
  data-audit-fnspid.md              B03 findings + B04 decisions
  timing-contract.md                B05
  scoring-checkpoint-contract.md    R01c, implemented R01d
  dedup-lineage-contract.md         R03a, implemented R03b
  project-audit-2026-09-09.md       the audit, findings A01-A15
  audit-implementation-plan-2026-09-09.md   the live repair sequence R01-R15
  handover.md                       this file
config.py                           every locked decision and pinned artifact
src/data.py       loading, mandatory source filter, dedup + lineage, market, calendar
src/audit.py      per-source profiling; withholds rates a cluster sample cannot support
src/align.py      both mappers, daily aggregation, panel, all lags and leads
src/scoring.py    three scorers behind one protocol, hash-keyed cache
src/validate.py   Act 1 metrics (not yet reworked to the B01 protocol)
src/inference.py  HAC regressions, BH, placebo, effect sizes
src/plots.py      one function per figure (titles still assert conclusions — B27)
tests/            216 passing, 0 skipped: alignment · audit · data · inference ·
                  scoring · checkpoints · acquisition · market
run_all.py        panel -> tables and figures, with the standing fallback guard
rescore.py        the one-off scoring pass (--time-only times it first)
```

## 9. Conventions worth keeping

Four habits account for most of the defects caught so far, and abandoning them will cost more than they save.

1. **Reproduce a defect before specifying it away.** Every fix in Phase B started from a printed demonstration against the running code, not from reading it. Two of the three were worse than they looked.
2. **Encode constraints where they bind, not in prose.** `CorpusRate` withholds a statistic; `_require_lags` refuses a panel; the mappers refuse each other's input. A caveat in a document does not survive contact with a future session.
3. **Fix decisions before they can be chosen by their results** — the pooled-universe sensitivity, the secondary mapping rule, the SESOI, family membership. Each is recorded now precisely because it would be indefensible later.
4. **Separate a decision from its implementation and its verification** in the decision log. Acceptance is not completion; a stub is not a feature.

And the standing rule for this project: **the assistant does not perform Git writes.** Staging, commits and pushes are the human's.
