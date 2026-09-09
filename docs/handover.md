# Handover

Date: 2026-09-09. Written at the close of Phase B; updated after corpus assembly.

Read this first if you are picking the project up. It records what exists, what was decided and why, what is deliberately absent, and what blocks the next step. Authority for each decision lives in the linked document; this file is a map, not a substitute.

Related: [implementation plan](implementation-plan.md) (B01–B28) · [decision log](research-review-decision-log.md) · [scope draft](protocol-revision-draft.md) · [original plan](implementation-plan-original.md).

---

## 1. Where the project stands, in one paragraph

The protocols are written, the candidate dataset has been audited, every timing defect found in the review has been fixed with a regression test behind it, and **the corpus is assembled**: 869,205 deduplicated Benzinga headlines over 2,516 trading sessions, with D4 now frozen on coverage evidence. **No text has been scored, no labels have been collected, no panel has been built, no model has been fitted, and no empirical result exists.**

**120 tests pass, 7 skip** (the skips are VADER and FinBERT, which are optional until scoring begins).

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

The estimand is written first, and everything follows from it. Uncertainty for the macro-F1 difference is a **paired bootstrap resampling article groups**, recomputing both models on the same resampled items (B = 10,000, seed 20260830) — macro-F1 is not a mean of per-item scores, so it cannot be bootstrapped from a per-item vector, and near-duplicate headlines are not independent draws. McNemar is retained but relabelled a *paired accuracy comparison*.

Primary evaluation is annotated headlines from this study's own collection, with a documented fallback order; PhraseBank is contaminated for `ProsusAI/finbert` (its own model card names PhraseBank as fine-tuning data) and is supplementary only. Calibration/evaluation separation is **by article group and stored as a file**, not recomputed from a seed. Rubric v1 has ten decision rules; a 60-item pilot precedes the full annotation.

### B02 — inference protocol → [`inference-protocol.md`](inference-protocol.md)

Primary estimand and two-sided null; frozen control set; eligibility rules; HAC bandwidth `L = 5` prespecified with a sensitivity set; effect scale in bps per 1σ; **SESOI = 5 bps** with a precision check run *before* the coefficient is inspected; and three permitted conclusions including an explicit *inconclusive*.

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
- **Pinned:** FNSPID repo `bf9189c4…`, file sha256 `dde52918…` (5,731,397,037 bytes), FinBERT `4556d130…`. Licence CC BY-NC 4.0, non-commercial.

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
| B11 | Prediction-blind annotation sample | ⬜ | Blocked on corpus assembly. **External dependency: human labelling** |
| B12 | FinBERT class probabilities | ✅ | `scoring.Classifier`, `predict_proba`/`predict`, `validate.predictions_for`; label order checked against the pin at construction |
| B13 | Scoring-provenance / cache invalidation | ✅ | `fingerprint` per scorer, `.meta.json` sidecar, `IncompatibleCache` |
| B14 | Batch checkpoints, resumable scoring | ✅ | atomic checkpointed writes; interruption test |
| B15 | Paired classification uncertainty | ⬜ | Specified in B01 §7 |

### Phase D — panel and inference

| ID | Task | Status | Notes |
|---|---|---|---|
| B16 | Volume naming; context-plot contract | 🟡 | `close_adj` now reaches the panel (part of P29). Renames `log_turnover`→`log_volume`, `parkinson`→`rv_parkinson` outstanding |
| B17 | Primary next-day regression on a fixture | 🟡 | `inference.predictive` exists and runs; not yet verified against the B02 eligibility rules |
| B18 | Secondary family + paired scorer contrast | 🟡 | BH implemented; **BY sensitivity and the stacked `delta` contrast are not** |
| B19 | Timing diagnostic corrected | ⬜ | Still block-resamples with replacement; B02 §9 specifies the circular shift |
| B20 | Standardized effects, precision, null logic | 🟡 | `effect_size_bps` exists; standardization, the advance precision check and the three-conclusion rule are not implemented |

### Phase E — mathematics and analysis

| ID | Task | Status |
|---|---|---|
| B21 | Derivation + simulation | ⬜ |
| B22 | Scoring pilot and extrapolation | ⛔ **blocked** — torch DLLs refused by Smart App Control |
| B23 | Bounded, resumable scoring run | ⬜ |
| B24 | Independent classification results | ⬜ |
| B25 | Core panel and primary market results | ⬜ |

### Phase F — reproduce and present

| ID | Task | Status |
|---|---|---|
| B26 | Integrated reproduction path | ⬜ |
| B27 | Evidence-based captions, report, README | ⬜ |
| B28 | Clean-directory reproduction check | ⬜ |

### Carried defects not yet fixed

| Item | Where | Increment |
|---|---|---|
| Figure titles assert conclusions before results exist (P24) | `src/plots.py` — "Same-day association, next-day nothing"; "heavier trading"; "lexicons over-predict neutral" | B27 |
| README states findings as filled-in placeholders | `README.md` | B27 |
| Report skeleton still uses original-plan language in places | `report/report.md` | B27 |
| Placebo resamples blocks with replacement | `inference.permutation_pvalue` | B19 |
| `requirements.txt` versions are declared, not `pip freeze`d | | B28 |

## 5. Corpus assembly and census — done

Streamed the whole 5.73 GB file once (13,057,514 rows, 11.5 min), storing only the filtered result. Full findings in [`data-audit-fnspid.md`](data-audit-fnspid.md) §11.

**The selected source occupies two separate blocks.** Kept rows plateaued at 0.55 GB, then resumed at 2.44 GB. Reading only the first block — which the audit's offset map would have suggested was enough — would have silently dropped ~122,000 headlines, about 9%.

| Measurement | Result |
|---|---|
| Dedup rate | **38.5%** (431,602 exact + 111,717 near) |
| Headlines per session | mean 345, median 339 |
| Zero-news sessions | **2 in ten years** |
| Distinct tickers | 5,707; top-10 share 2.1%; effective 1,839 names |
| Coverage stability | no year below tolerance → **D4 frozen** |

`interim/headlines_raw.parquet` (1,412,524) keeps the pre-dedup corpus so the rate stays checkable; `interim/headlines.parquet` (869,205) is the analysis input.

Two corrections came out of it, both recorded: the earlier "0.05% malformed timestamps" was an artifact of byte-range slicing, not a property of the file (the full read produced **zero**); and the window's end bound was inclusive of the following midnight, which for a date-only corpus admitted a whole extra day.

## 5a. The next increment

**B22 — the scoring pilot.** `rescore.py --time-only` times FinBERT on 1,000 headlines and extrapolates. 869,205 headlines is the real workload, and the budget question must be answered by measurement before a full pass is launched. `torch` and `transformers` are not yet installed.

If the projected time is unacceptable, the plan's rule is to shorten the *window* and record it — **never** to subsample headlines within days, which would bias both `S_t` and `d_t`. Note that shortening the window now means unfreezing D4, which requires a dated log entry.

In parallel and independent of scoring: **B12** (FinBERT class probabilities, preserving the verified label order) and **B10** (the PhraseBank loader), plus obtaining the Loughran–McDonald dictionary.

## 6. Blockers and external dependencies

| Blocker | Blocks | Action |
|---|---|---|
| **Loughran–McDonald dictionary** not obtained | Any LM scoring; Act 1 | Download by hand from the Notre Dame SRAF site (no stable link) into `data/raw/LoughranMcDonald_MasterDictionary.csv`, then record the release in `config.LM_DICT_VERSION` |
| **Human annotation** not started | B11, B15, B24, all of Act 1 | Needs a named annotator; ideally a second on a 20% subset for kappa. Protocol is written and executable |
| ~~Window not frozen~~ | — | **Resolved**: frozen 2026-09-09 on the census |
| **Smart App Control blocks `torch`** | B22, B23, all FinBERT scoring | `torch` is installed but `WinError 4551` refuses `torch/lib/c10.dll`. Turn off Smart App Control in Windows Security -> App & browser control. Not fixable from the code. `transformers` and `vaderSentiment` work |
| `datasets==4.0.0` loader | B10, PhraseBank fallback only | Not on the critical path unless annotation fails |

## 7. Environment notes

- Python 3.12, venv at `.venv/`. `pandas 3.0.5` — note the string-dtype behaviour that produced the `chunksize` schema bug.
- **Smart App Control blocked scipy's DLLs** on this machine for several increments, making `statsmodels` and `scikit-learn` unimportable. It cleared during B06 and the inference path now runs. If it recurs, the symptom is `ImportError: DLL load failed while importing _comb: An Application Control policy has blocked this file`.
- `pandas-market-calendars` is now a **hard requirement** — `trading_calendar` raises without it rather than degrading.
- `data/raw/` is gitignored. The audit sample (`fnspid_sample.parquet`) is local only; regenerate with `python data/raw/download.py --fnspid-sample --slices 24 --slice-mb 6`.
- `torch`, `transformers` and `vaderSentiment` are **not installed**; two scorer tests skip cleanly until they are.

## 8. Repository state

42 files tracked as of the last commit. Uncommitted at the time of writing: `config.py`, `run_all.py`, `src/data.py`, `src/inference.py`, `tests/test_inference.py`, `docs/research-review-decision-log.md` (modified) and `tests/test_data.py` (new) — the B09b work.

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
  handover.md                       this file
config.py                           every locked decision and pinned artifact
src/data.py       loading, mandatory source filter, dedup, market, calendar
src/audit.py      per-source profiling; withholds rates a cluster sample cannot support
src/align.py      both mappers, daily aggregation, panel, all lags and leads
src/scoring.py    three scorers behind one protocol, hash-keyed cache
src/validate.py   Act 1 metrics (not yet reworked to the B01 protocol)
src/inference.py  HAC regressions, BH, placebo, effect sizes
src/plots.py      one function per figure (titles still assert conclusions — B27)
tests/            alignment 38 · audit 22 · data 15 · inference 11 · scoring 10
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
