# Handover

Date: 2026-09-09. Written at the close of Phase B; updated after corpus assembly, and again after the 2026-09-09 audit repairs (R01–R05, R03a/R03b).

Read this first if you are picking the project up. It records what exists, what was decided and why, what is deliberately absent, and what blocks the next step. Authority for each decision lives in the linked document; this file is a map, not a substitute.

Related: [implementation plan](implementation-plan.md) (B01–B28) · [decision log](research-review-decision-log.md) · [scope draft](protocol-revision-draft.md) · [original plan](implementation-plan-original.md).

---

## 1. Where the project stands, in one paragraph

**Inference update, 2026-09-10:** the Act 2 track is complete — R07a–R07d and R08a–R08c. Panel field names match their formulas; the frozen primary specification fits on an auditable eligibility ledger with tone standardized at fit time; `L = 5` counts exchange sessions (M7 decided, both conventions reported); advance precision is atomically recorded before any tone fit; effects report both evidence dimensions with boundary disclosure; one unadjusted primary plus the closed 14-test secondary family under BH with BY alongside; the paired scorer contrast carries the cross-equation covariance; and the timing diagnostic is an exhaustive circular shift reported as a percentile. The block-resampling placebo, `attenuation_comparison` and `horse_race` were **deleted**. See the [reporting contract](inference-reporting-contract.md) and the [spacing decision](hac-spacing-decision.md). **A08 is closed. No empirical coefficient was estimated. 388 tests pass, 0 skipped.**

**Dependency update, 2026-09-10:** the LM dictionary is now acquired and verified (1993–2025, March 2026 release; [provenance](lm-dictionary-provenance.md)). The user volunteered to label the 60-item pilot; [worksheet](../data/annotation/pilot_worksheet.csv) and [instructions](../data/annotation/PILOT_README.md) are ready. Labels are still pending. R06b implementation can proceed on synthetic fixtures without labels; R13a empirical evaluation cannot. R11 no longer waits for the dictionary, but still follows its implementation/readiness prerequisites. The frozen sample was not redrawn.

The protocols are written, the candidate dataset has been audited, every timing defect found in the review has been fixed with a regression test behind it, and **the corpus is assembled**: **869,183** deduplicated Benzinga headlines over 2,516 trading sessions, with D4 frozen on coverage evidence. A full project audit on 2026-09-09 found fifteen issues (A01–A15). Completed repairs cover scoring-cache identity and durable checkpoints, verified acquisition, census mapping, the market-return calendar, score-to-panel checks, protocol amendments, deduplication lineage and its verified rebuild, Act 1's blind sample, and the inference reporting work above. **No corpus-wide scoring or empirical market regression was performed in this increment; independent labels and empirical results remain pending.**

The seven long-standing skips were FinBERT and VADER; Smart App Control has since been disabled, so `torch` loads and every test executes.

The live plan is the [audit repair sequence](audit-implementation-plan-2026-09-09.md), which sits inside the B01–B28 roadmap below and records each increment's current state.

### Start here

1. Read §1–§2 for what the study is, then **§3a** for what the last session changed.
2. **§5a** names the next increment and why it is next.
3. **§6** is what you cannot do without a human.
4. Run `python preflight.py` to see what is installed and what artifacts exist, then `pytest`: **546 passing, 0 skipped**, is the baseline.

Three facts that will save you an hour:

- **The corpus is final.** It was rebuilt through the verified acquisition path at R03d; `interim/headlines.parquet` is 869,183 rows and its manifest records `verified_against_pin: true`. Do not redraw or rebuild it without a dated decision-log entry.
- **Act 1's sample is drawn and waiting on a person.** `data/annotation/` holds 800 blind items. The draw is made **once** (P25).
- **The Act 2 inference track is built.** Primary, return-family, paired-contrast and timing paths all use the accepted controls, standardization and eligibility. The superseded `attenuation_comparison`, `horse_race` and `permutation_pvalue` were **deleted** at R08b/R08c. The remaining old-plan callers are the RQ4 exploratory specs (`volatility_spec`, `volume_spec`), which still use raw-tone `predictive` conventions and are not on the primary path.

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

## 3a. The 2026-09-09 audit repair session

A full [project audit](project-audit-2026-09-09.md) found fifteen issues (A01–A15). The [repair sequence](audit-implementation-plan-2026-09-09.md) addresses them in twenty-nine increments. Twenty-one are done. What follows is what changed and what it cost, because several of these were found by measurement rather than by reading code.

### R01a–R01d — scoring cache and checkpoints (A01, A02)

Fingerprints compare by JSON representation so identity survives save/load; FinBERT records the revision actually passed to its constructor; `text_sha256` detects changed headlines; a changed scorer fingerprint invalidates that entire cached column. Scores and provenance now live in **one** Parquet file with a single replacement commit point, a process-held writer lock and explicit legacy refusal — the [checkpoint contract](scoring-checkpoint-contract.md), 30 tests.

### R02 — verified acquisition (A05)

`assemble_corpus` writes the **raw** destination only and refuses to be pointed at the clean one, so a documented command can no longer replace the deduplicated corpus with the undeduplicated one under the same name. The URL is built from the pinned revision; the stream is hashed as consumed and checked **before** anything is published; publication is atomic with the manifest written after the data.

### R05 — protocol amendments (A10, A09)

Six amendments to the two frozen protocols, made while no result of any kind existed — which is what makes them corrections rather than post hoc adjustment. Each protocol carries its own amendment record.

| | Change |
|---|---|
| M1 | The advance precision check is two **planning** half-widths, not a bound. The old claim that a wide `1.96·sd(r)/√n` means the study "cannot" deliver the smallness conclusion "no matter what is estimated" was false in both directions and is withdrawn |
| M2 | Three overlapping conclusion categories (a `[1,3]` bps interval satisfied two of them) replaced by two always-reported dimensions, a 2×2 naming rule and a non-strict 0.1 bps boundary rule |
| M3 | The 14-test secondary family is closed and contains **return tests only**; Act 1's contrasts are specified in the validation protocol and cannot join it |
| M4 | The primary accuracy comparison is the paired **group** bootstrap; exact McNemar is supplementary, printed with its validity condition, because §4's own grouping rule contradicts its independence assumption |
| M5 | Independence narrowed to **label** independence; text exposure recorded as unknown for all three scorers; PhraseBank's number is "optimistically biased", not an upper bound |
| M6 | The circular shift's domain is the retained analysis rows in session order, with midrank ties and a reported tie count |

**M7 (HAC lag spacing, A09) — decided at R07c, 2026-09-10.** **Session-indexed is primary; retained-position is computed on every fit and reported alongside.** The choice was made on interpretive grounds — `L = 5` was prespecified as *one trading week*, and only that convention makes the sentence true — so it does not depend on any measured quantity, which is stricter than the deferral required. Measurement bounds the exposure rather than making the choice: identical with no gaps, **at most 0.33%** difference on the tone standard error at this corpus's anticipated gap structure, about 7% at 50% gaps. The comparison on the **real** sample is still outstanding (that sample does not exist) and is a required manifest entry at R13b. Full record: [`hac-spacing-decision.md`](hac-spacing-decision.md).

### R03a/R03b — deduplication lineage (A12, A11)

Five defects, each reproduced against the running code and then quantified on the real 1.4M-row corpus. The [dedup lineage contract](dedup-lineage-contract.md) has the full evidence.

- **The output was not order-invariant.** `dedup` sorted on `ts_utc` with an unstable quicksort, and in a date-only corpus *every* same-day duplicate group ties on that key. Permuting the corpus changed 2,363 surviving ids and the retained ticker tags of **93,552 survivors (10.78%)**. The representative is now the total order `(ts_utc, source_row_id)`, and permuting changes **nothing**.
- **Ticker tags were being destroyed** — 501,957 of 663,074 distinct (cluster, ticker) pairs, **75.7%** — because a story filed under three tickers kept one. Tags are now unioned across the cluster. Concentration measured on unioned tags is marginally *lower*, so the D4 universe decision is unaffected.
- **The exact/near split was badly wrong.** The exact window anchored on a text's first-ever occurrence rather than its last kept one, so later clusters of exact repeats were charged to the near count: **96.2% of the reported near-duplicate count was actually exact duplication**. Corrected from 431,602 / 111,717 to **539,087 / 4,254**.
- **The blocking-exposure boundary was off by one.** `ceil(2/(1-0.90))` returns 21 in binary floating point; the documented boundary is 20, and a 20-token pair differing by two tokens clears the threshold exactly. Now computed in exact rational arithmetic.
- **Lineage is recorded.** `interim/dedup_lineage.parquet` holds one row per raw row, so the dedup rate is recomputable from the artifacts rather than only reproducible by re-running the pass.

`headline_id` is `sha1(text_norm | ts_utc)`, so it identifies a *group*, not a row — 518,332 raw rows share an id with another. Lineage therefore needed the new `source_row_id`.

**A design decision came out of implementation, not specification.** The first version asserted that every eliminated row points at a survivor. That assertion **failed on the real corpus**: an exact repeat's anchor can itself be removed later as a near-duplicate, so elimination forms chains — 12,074 rows, 2.2%. Lineage records the immediate eliminator; `cluster_id` resolves to the root.

### R03d — verified rebuild, and a pin that was wrong

**The pinned digest did not match, and R02's guard caught it.** The first assembly refused to publish and wrote nothing: the streamed file hashed to `5d4c0180…`, not the pinned `dde52918…`.

HuggingFace's `paths-info` API for the pinned revision settles what happened. It reports `lfs.oid` = `5d4c0180…` — Git LFS OIDs are sha256-of-content, matching the independent computation — and `xetHash` = `dde52918…`. **B04 recorded the value from the HTTP ETag**, which HF serves as the Xet hash for Xet-backed files; the data audit even labelled the row "File sha256 (etag)". The file never changed: same revision, same 5,731,397,037 bytes, same content. Only the pin named the wrong hash function. `config.NEWS_FILE_SHA256` is corrected and `NEWS_FILE_XET_HASH` keeps the original for traceability.

This also means **the pre-R02 corpus had never had its digest checked at all**, which is why its manifest recorded `verified_against_pin: false`.

The rebuild produced **the same data**: 1,412,524 raw rows, identical `headline_id` set and identical `text_norm` multiset. Clean corpus 869,205 → **869,183** (symmetric difference 2,314 ids, 0.27%). Census reconciles at **869,092 + 91 = 869,183** and **D4 holds** — no year below tolerance, one structural zero-news session.

### R06a — the blind annotation sample

`src/annotate.py` and 30 tests. `data/annotation/` holds **800 headlines, 200 calibration / 600 evaluation**, drawn proportionally across 10 strata with seed `20260830`, **once**.

Article groups are computed within the sample by exact all-pairs comparison and are **unwindowed** — the relation the dedup contract separated from the windowed cluster, because a story republished ten days later survives dedup twice but is still one article for leakage purposes. A whole group goes to one part, and `load_split` re-checks that invariant on read.

The blind export carries `headline_id` and `text` only, guarded against a forbidden-column list, shuffled under a **separate** order seed so reshuffling the running order cannot disturb which rows were drawn. `provenance.md` leaves annotator, dates, blindness confirmation and deviations as explicit `TO BE COMPLETED` blanks, and a test asserts they stay blank.

### The FinBERT spot-check, which is not a code defect

With `torch` finally loading, six scorer tests executed for the first time and one failed. It asserted that FinBERT would *not* call "Costs fell sharply in the third quarter" negative — Exhibit A's premise that a context model sees what a word counter cannot.

| Headline | FinBERT | P(neg) |
|---|---|---|
| Costs fell sharply in the third quarter | **negative** | 0.932 |
| Profit warning smaller than feared | **negative** | 0.924 |
| Quarterly profit beats expectations | positive | — |
| Shares plunge after weak guidance | negative | 0.933 |

Both headlines built to demonstrate the premise are ones FinBERT gets wrong, agreeing with the word counter it was supposed to beat. **The B12 label-order pin is confirmed correct** against the real checkpoint, so the code is fine; what failed was a test asserting a research expectation on six invented sentences before Act 1 has run. The assertion was **removed rather than inverted** and replaced with a characterization record pinned to the revision.

Six invented sentences establish nothing in either direction — that is what Act 1's independently annotated evaluation is for — but the premise is no longer free, and it still stands in `README.md` and `report/report.md` (R09/B27).

### The documentation pass (A14)

`README.md`, `report/report.md`, `src/plots.py`, notebooks 01–03, `implementation-plan.md` and the decision log were brought to what the code actually does. Removed: the "domain transformer reads financial sentences far better" premise; the Findings section's placeholder bullets that pre-wrote the expected answer's shape; a block literally headed **"Expected outcomes, declared before any result was seen"**; the "no same-day association means something upstream is broken" sanity check (P23); six figure titles that asserted conclusions; the `--all` flag that does not exist; and the D6 intraday rule that is descoped.

### R09 — documentation reconciled against the code (A14)

Every documentation surface was read against what the code now does, rather than against what it did when the text was written. What R09 found is worth recording, because none of it was failing loudly:

- **The decision log's standing status paragraph had decayed again.** It still quoted the pre-R03d corpus count (869,205, not 869,183), still said "no annotation sample has been drawn" after R06a had drawn 800 items, and still named Smart App Control and the missing LM dictionary as live blockers after both were resolved. This is the second time this one paragraph has gone stale — R05 corrected it for the same reason — so it is now dated on every edit and carries a note explaining why.
- **The README still listed Act 2's *three* conclusions**, which M2 withdrew for overlapping (an interval of `[1, 3]` bps satisfied two of them at once). It also gave McNemar as the primary accuracy comparison (M4 replaced it with the group bootstrap), described block permutation as "a null that assumes nothing about the error process" — protocol §11 forbids describing any procedure here as assumption-free, and the block path was deleted at R08c — and called the 5 bps SESOI a "transaction-cost benchmark" delivering "economic significance", which P19 forbids.
- **The report contradicted itself two pages apart.** §5 correctly said the yardstick "is a measure of *smallness*, not a profitability threshold"; §9 called it a "transaction-cost benchmark". Its §2 inference bullet was two generations stale (per-scorer BH; "circular block permutation, block 21, 1,000 draws, seed 20260830" — a procedure and three settings that no longer exist), and §6.3 still carried the correlation-above-0.9 decision rule R08b removed.
- **Notebook 04 still instructed the predetermined finding.** It called RQ4 "the act most likely to yield a positive result" and told the reader to "report the d_t coefficient prominently: disagreement predicting volume is the project's most plausible positive finding" — both the promotion the joint-null ordering exists to prevent, and the "disagreement" reading P21 forbids.
- **The repair table disagreed with its own checkpoints.** Nine increments completed in earlier sessions carried no tick, and every prose count said "thirty increments" when the table has always listed **29**.

Four rows of the carried-defects table were already fixed and still listed as open. Listing a closed defect as open is the same failure as the reverse: it costs a later session the time to rediscover it. They are struck through with what closed them rather than deleted.

**Two code defects survive R09 and are not documentation problems.** The source filter matches by substring, so `notbenzinga.com` would pass the mandatory domain filter; and `config.AGG` is an inert flag that `aggregate_daily` never reads. Both are re-verified open and left for the audit follow-up, because R09 changes prose, not behaviour.

### R10 — preflight and pilot readiness (A13, A15)

**A13 was quantified by the tool built to close it.** On its first run `preflight.py` reported that of fourteen declared requirements, **one matched what was installed, eleven differed, and three were absent**. Every pin was a version the project had never executed against, so a green suite established behaviour in *some* environment rather than the pinned one. `requirements.txt` is now pinned to the environment the suite actually runs in, with `yfinance`, `datasets` and `jupyterlab` — genuinely not installed — in a separate block that says nothing has been run against them.

The repair is a command and a guard rather than a promise to stay in step: `python preflight.py` reports declared-vs-installed on demand, `--freeze` prints correct pin lines from the live environment, and a standing test fails when the file stops describing the environment. Restoring the original `numpy==2.3.3` makes it fail, so it is the assertion that would have caught A13.

**Newly visible:** `yfinance` is absent, so `data/raw/download.py --market` cannot run here and the market artifact cannot be built in this environment. Previously that surfaced only as an ImportError partway through a command.

**A15's readiness half.** `require_artifacts(stage)` runs before anything expensive in `run_all.py` and `rescore.py`, naming the artifact, its expected path, and the command that produces it — as a plain message, because a stack trace says "this program broke" when the truth is "run that first". The registry separates what a **command** builds from what a **person** must supply. Output staging and the completion manifest remain R14's.

**Bounding is by whole session, never by headline count.** `rescore.py --sessions N` takes the first `N` calendar days entire. A partial day is not a smaller sample — it is a different measurement, because `S_t` is a within-day mean and `d_t` a within-day standard deviation. Also added: `--dry-run` (scope and checkpoint location, nothing scored) and `--checkpoint-every`.

**Skips were already narrow; the docstring promising them was not.** `tests/test_scoring.py` still said the model tests "skip cleanly" when the model is absent — the behaviour A13 named, and the reason a failing label-order assertion read as passing for weeks. It now records what conftest actually does: marked `integration`, errors normally, deselected explicitly with `-m "not integration"`.

**A file was overwritten and then merged back.** A root `preflight.py` already existed at HEAD, written by concurrent work on this same increment, and it was overwritten before being read — my error. It was recovered from git and **merged rather than discarded**, because it carried the substantive half: parquet schema validation, the LM dictionary SHA-256 check against `config.LM_DICT_SHA256`, complete-score-cache verification with scorer-identity comparison, optional model loading, and a JSON report R14's run manifest will want. What this pass added on top is the artifact registry with `produced_by`/`external`, the `require_artifacts`/`require_packages` gates wired into both runners, the dependency status taxonomy, `--freeze`, and 30 tests where there had been none. The two stage vocabularies were reconciled onto the one already committed — `unit`, `pilot`, `scoring`, `analysis`, `validation` — so the project has one set of stage names, and a test asserts it.


### R11, pilot half — the expensive step, measured before it is spent

The scoring pass is the one step in the project that costs hours, so it was measured before being run. **FinBERT scores 54 headlines/s on this CPU**, which projects to **4.49 h** for the 869,183-headline corpus — over the plan's 2 h budget. **Truncation at 64 tokens hits 0.45 %** of headlines (3,921), measured by tokenizing the whole corpus rather than asserting that headlines are short; the tail is guidance and option-alert lines that chain several figures. Storage is 128 B/row, about 111 MB.

A **10-session bounded chunk was scored** (546 headlines, 17.4 s) and then checked from the artifact: every headline of every touched day is present exactly once, all three scorer fingerprints match the configured scorers, and re-running the identical command took 0.2 s and rescored nothing. `data/interim/scores.parquet` therefore now holds real study data — 546 rows — and the score-to-panel gate will refuse it until every in-window session is scored.

**The full pass has not been run.** The plan's rule for an over-budget projection is to shorten the window (D4), which is frozen and needs a dated decision-log entry from the user. The alternatives on the table: accept 4.5 h as a one-time cost; or first add length-sorted batching, which changes no output and plausibly recovers 1.5–2×. Neither is taken here.

### R13b, RQ4 and R14 — the first results, and the machinery that dates them

**Act 2's primary claim: −1.74 bps per 1 SD, 95% pointwise [−4.90, +1.42], n = 2,453.** No association detected, and precise enough to exclude effects beyond ±5 bps — the informative null. **Flagged as a boundary case**: the lower endpoint is −4.9031, within 0.1 bps of −5, so the label turns on less than the reporting precision. That flag travels with the number wherever it is quoted.

Nothing rejects anywhere. Secondary family: smallest BH q 0.946, BY 1.000. RQ4 family: smallest BH q 0.087. Paired contrast `delta = −2.43 bps [−6.88, +2.03]`.

**RQ4 was rebuilt before being reported.** The old path gave per-coefficient p-values of 0.005–0.03 on individual dispersion and tone terms, unstandardized, uncorrected, on a different sample from the primary. That is the arrangement P21 exists to prevent. The protocol's joint HAC Wald test of `b1 = b2 = b3 = 0` replaces it; the coefficient table now carries no q-value or reject flag by construction, and lives in a separate file so the two cannot be read as one.

**R14 closed A15's remaining half.** A failed run now publishes nothing and leaves the previous results intact; the manifest is written last and is the commit point; a results directory without one cannot be quoted as current. Panels carry the digests and settings they were built under, so a stale panel is refused rather than relabelled with current config — which was A15's specific concern and is invisible to a schema check.

### R13a — Act 1 measured, and the two acts do not connect

**`macroF1(FinBERT) − macroF1(LM) = +0.102, 95% [+0.050, +0.153]`**, n = 596 over 595 article groups; **+0.104** on the `hard = 0` subset, agreeing to 0.002. The only interval in this study that excludes zero.

**The entire gap is the positive class.** FinBERT and LM are within 0.02 on negative and neutral; LM's positive F1 is 0.263 against FinBERT's 0.546, and it calls **83% of genuinely positive headlines neutral**. The cause was already visible at calibration: LM takes three distinct values on headline text (78.3% exactly zero), and its dictionary carries 2,345 negative terms to 347 positive. This is a finding about applying a 10-K risk dictionary to headlines, not about transformers reading better — and the distinction is enforced by a test that requires the explanation to travel with the number.

**Act 1 separates the scorers; Act 2 cannot** (`delta = −2.43 bps [−6.88, +2.03]`). R12 derived that before either ran: aggregation over ~337 headlines per session compresses a twofold noise difference into 1.034× of coefficient. The bridge is untestable at this aggregation rather than refuted.

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
| B16 | Volume naming; context-plot contract | ✅ | **R07a, 2026-09-10.** `log_turnover`→`log_volume`, `parkinson`→`rv_parkinson` across code, tests and docs; field contract in [`timing-contract.md`](timing-contract.md) §10; legacy-name and required-column guards; the context figure now requires `close_adj` instead of plotting a scalar NaN (P29 closed) |
| B17 | Primary next-day regression on a fixture | ✅ | **R07b, 2026-09-10.** `inference.primary`/`PrimaryFit` fit the frozen specification on `inference.eligibility`'s rows; detrended volume control, tone standardized at fit time, four-reason exclusion ledger, full-calendar and per-observation adjacency guards. 24 tests in `tests/test_primary.py`. `predictive` remains for the secondary/exploratory paths R08 replaces |
| B18 | Secondary family + paired scorer contrast | ✅ | **R08a + R08b, 2026-09-10.** One unadjusted primary and the exact 14-test secondary family with BH **and BY** (R08a); the stacked `delta` contrast on the common sample, with the cross-equation HAC covariance in its standard error (R08b) |
| B19 | Timing diagnostic corrected | ✅ | **R08c, 2026-09-10.** `inference.timing_diagnostic`: full circular shift over the retained rows in session order, midrank percentile at 0.1 bps, tie count, gap disclosure. The block-resampling path and its config were deleted |
| B20 | Standardized effects, precision, null logic | ✅ | **R07b + R07d, 2026-09-10.** Tone standardized at fit time so `beta` is per 1 SD and bps is `beta * 10,000` with no second rescaling (R07b); both advance half-widths recorded before any tone fit, and M2's two evidence dimensions with the boundary rule (R07d). The withdrawn three-conclusion rule is not implemented, by design |

### Phase E — mathematics and analysis

| ID | Task | Status |
|---|---|---|
| B21 | Derivation + simulation | ✅ **R12, 2026-09-10** — [`mathematical-appendix.md`](mathematical-appendix.md), `attenuation_study.py` |
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

Fifteen findings (A01–A15) from the 2026-09-09 [project audit](project-audit-2026-09-09.md), repaired in the twenty-nine increments of the [repair plan](audit-implementation-plan-2026-09-09.md). **Twenty-one complete, eight remaining** (R09 in progress; R10–R15 outstanding).

| Increment | Finding | State |
|---|---|---|
| R01a–R01d | A01, A02 | ✅ Cache identity, validation scope, checkpoint contract, durable recovery |
| R02 | A05 | ✅ Verified acquisition: pinned revision, stream digest, atomic publication, raw/clean separation |
| R03a, R03b | A12, A11 | ✅ Dedup lineage contract and its implementation |
| R03c | A06 | ✅ Census derived from one session assignment |
| R04a, R04b | A04, A03 | ✅ Returns on the calendar; complete-score gate before aggregation |
| R05 | A10, A09 | ✅ Six protocol amendments; M7 (HAC spacing) deferred to R07c |
| R03d | — | ✅ Verified rebuild; corpus replaced, `verified_against_pin: true` |
| R06a | A11 | ✅ Blind sample drawn; user will label the 60-item pilot in `pilot_worksheet.csv` |
| R06b | A11 | **Ready for implementation on synthetic fixtures.** Empirical use awaits human labels |
| R07b | A07 | ✅ Frozen primary specification: eligibility ledger, detrended volume control, fit-time standardization, one sample for fit/SD/bps scale |
| R07c | A09 | ✅ HAC spacing: **M7 decided** — session-indexed primary, retained-position reported alongside; [`hac-spacing-decision.md`](hac-spacing-decision.md) |
| R07d | A10 | ✅ Pointwise effects, two evidence dimensions, boundaries, advance widths and controls-only kappa; atomic manifest publication before tone estimation |
| R08a | A08 | ✅ Exact primary + 14 secondary return family; BH/BY, validation and runner migration |
| R08b | A08 | ✅ Stacked `delta` contrast on the common sample; collinearity reported, never acted on; `attenuation_comparison`/`horse_race` deleted |
| R08c | A08 | ✅ Circular-shift timing diagnostic as a percentile; `permutation_pvalue` deleted. **A08 closed** |
| R09 | A14 | ✅ Documentation reconciled against the code; withdrawn procedures and a predetermined finding removed |
| R10 | A13, A15 | ✅ Preflight inventory and artifact gates; `requirements.txt` re-pinned to the tested environment; bounded whole-session scoring |
| R11 | — | 🟡 **Pilot done.** 54 headlines/s, 4.49 h projected, 0.45 % truncation, 111 MB; 10-session chunk scored, verified, resumed. **Full pass awaits the user's D4/budget decision** |
| R12 | — | ✅ Attenuation derivation + six-case simulation. Standardization gives exponent 1/2; aggregation over 337 headlines compresses a 2x noise gap to 1.034x; bounded scores **amplify** rather than attenuate |
| — | — | **8 of 15 findings' repairs remain**: R11–R15. A13 and A14 closed at R10/R09 |
| R13a | — | ✅ **800 headlines labelled.** `macroF1(FinBERT) − macroF1(LM) = +0.102 [+0.050, +0.153]`, n = 596; +0.104 on the confident subset. Entire gap is the positive class — LM calls 83% of positives neutral |
| R13b | — | ✅ Act 2 run. Primary **−1.74 bps [−4.90, +1.42]**, informative null, **boundary case flagged**. Nothing rejects in any family |
| R14 | A15 | ✅ Staged publication, run manifest, panel provenance. A failed run publishes nothing |
| R15 | — | ✅ Report written against the published tables; **Act 2 robustness suite run** — 2 of 5 exhibits informative, 1 power-bounded, 2 inapplicable, all reported as such; README Findings replaced; [`reproduction.md`](reproduction.md) + 22 tests holding the docs to the code |

**Twelve findings are closed (A01–A12)**, A14 is closed by this R09 pass, A15 is partly closed, and A13 remains open (preflight and dependency inventory — R10).

### Carried defects not yet fixed

Verified line by line at R09 (2026-09-10). Four rows had already been fixed in earlier increments and were still listed as open — which is the same failure mode as listing an open defect as closed, and is recorded rather than quietly deleted.

| Item | Where | Increment |
|---|---|---|
| ~~Figure titles assert conclusions before results exist (P24)~~ | **Already closed**; every title in `src/plots.py` names its axes, and the three quoted strings no longer exist | — |
| ~~README and report assert that a transformer reads financial sentences better than a word counter~~ | **Already closed**; both carry a dated correction note instead | — |
| ~~README states findings as filled-in placeholders~~ | **Already closed**; the Findings section says there are none | — |
| ~~Report skeleton still uses original-plan language in places~~ | **Closed at R09** — the stale inference bullet (per-scorer BH, block permutation with block/draws/seed), the §6.3 correlation decision rule, the "transaction-cost benchmark" framing in §9 that contradicted §5 two pages earlier, and P22 naming in §6 | R09 |
| Findings and method notes in the README described withdrawn procedures | `README.md` — the three overlapping Act 2 conclusions (M2 withdrew them), McNemar as the primary accuracy test (M4), block permutation as "assumes nothing" (§11 forbids "assumption-free"), the 5 bps SESOI as a "transaction-cost benchmark" (P19) | **Closed at R09** |
| Notebook 04 instructed a predetermined finding | `notebooks/04_volume_vol.ipynb` — "the act most likely to yield a positive result" and "report the d_t coefficient prominently" | **Closed at R09** |
| ~~Placebo resamples blocks with replacement~~ | **Closed at R08c** — function deleted; `timing_diagnostic` replaces it | — |
| RQ4 exploratory specs still use old-plan conventions; every primary, return-family, paired and timing path now uses `primary`/`eligibility` | `volatility_spec`, `volume_spec` (and the `predictive` helper they lean on) | optional RQ4 increment |
| ~~Source filter matches by **substring**~~ | **Closed 2026-09-10.** `data._host_matches` requires an exact host or a true subdomain; `notbenzinga.com`, `benzinga.com.evil.example` and `fakebenzinga.community` are now rejected. 4 tests | — |
| ~~`config.AGG` is never read~~ | **Closed 2026-09-10.** `aggregate_daily(agg=...)` reads it, validates against `config.AGG_CHOICES`, and records the choice in `coverage["aggregation"]` so a median run cannot be mistaken for the primary. `d_t` stays a standard deviation under either rule. 5 tests | — |
| VADER's cache fingerprint records `version: unknown` | `src/scoring.py` — the package exposes no version attribute; the lexicon size (7,506) pins the word list, but the field should read the installed distribution version via `importlib.metadata`, as `src/preflight.py` already does | unassigned |
| ~~`requirements.txt` versions are declared, not `pip freeze`d~~ | **Closed at R10** — re-pinned to the tested environment; `python preflight.py` reports drift and a standing test fails on it | — |
| `test_checkpoints.py::test_hard_stop_around_commit_and_resumption[after_text]` fails intermittently **under heavy machine load** | Kills a real subprocess at a commit boundary; a race in the test's own timing, not in the checkpoint contract. Measured 2026-09-10: 2 observed failures during concurrent work, then **0 in 30 isolated, 12 whole-file and 10 full-suite runs**. Nothing in R08–R10 touches `src/scoring.py` | unassigned |

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

**Nothing is blocked.** All 29 increments are complete and both acts are estimated. Remaining work is optional: a second annotator for Cohen's kappa (needs another person, 160 items), an event-category breakdown of Act 1 (mechanical, no annotator cost), and committing the session.

The corpus track is closed and Act 1's sample is drawn, so what remains on the critical path with **no external dependency** is the inference track: seven fixture-based increments that gate every Act 2 result.

| Increment | What it does | Why it is not optional |
|---|---|---|
| ~~R07a~~ | ✅ **Done 2026-09-10.** Renamed across code, tests and docs; field contract written; stale-artifact guards added | Names must match formulas (P22); obsolete consumers should fail tests |
| ~~R07b~~ | ✅ **Done 2026-09-10.** `inference.primary`/`PrimaryFit` on `eligibility`'s rows: detrended volume control, fit-time standardization, reconciling exclusion ledger, full-calendar guards | The old `predictive` fitted **raw** `log_volume` and rescaled the coefficient a second time (A07). R07d/R08a build on this interface |
| ~~R07c~~ | ✅ **Done 2026-09-10.** **M7 decided:** session-indexed primary, retained-position always reported alongside; `hac_spacing_study.py`, [`hac-spacing-decision.md`](hac-spacing-decision.md) | Chosen on interpretive grounds before any tone coefficient existed. Real-sample comparison still due at R13b |
| ~~R07d~~ | ✅ Precision and conclusion reporting integrated | Both advance widths, controls-only kappa, eligible-row record and atomic publication before any tone coefficient |
| ~~R08a~~ | ✅ Return family correction implemented and runner migrated | One unadjusted primary + 14 secondary, BH/BY and exact membership checks |
| ~~R08b~~ | ✅ **Done 2026-09-10.** Stacked `delta` on the common sample; the cross-equation covariance is large and positive here, so ignoring it **overstated** the difference's uncertainty | Section 11 forbids a "beats" claim without an interval for the difference (P07) |
| ~~R08c~~ | ✅ **Done 2026-09-10.** Circular shift, midrank percentile, tie and gap disclosure; the block-resampling path deleted | It was not a permutation, and its output was labelled and plotted as a p-value |

**Do this before scoring, not after.** R11's bounded scoring pass is the expensive step; discovering afterwards that the panel contract was wrong means re-deriving everything downstream of it.

**B22, the scoring pilot, is unblocked** — `torch` loads. `rescore.py --time-only` times FinBERT on 1,000 headlines and extrapolates before a full pass is launched. If the projected time is unacceptable, the plan's rule is to shorten the *window* and record it — **never** to subsample headlines within days, which would bias both `S_t` and `d_t`. Shortening the window means unfreezing D4, which requires a dated log entry.

### What is waiting on a human, and what it unblocks

| Hand this over | Unblocks |
|---|---|
| User labels the 60 items in `data/annotation/pilot_worksheet.csv`; check the pilot before the remaining annotation | Pilot assessment first; full labels later unblock R13a. R06b code does not require labels |
| LM dictionary | **Resolved 2026-09-10:** acquired, validated and release/hash recorded |

Neither blocks R07 or R08.

## 6. Blockers and external dependencies

| Blocker | Blocks | Action |
|---|---|---|
| ~~Loughran–McDonald dictionary missing~~ | — | **Resolved 2026-09-10.** Official 1993–2025 CSV at `LM_DICT_PATH`; release and SHA-256 in config; 347 positive / 2,345 negative terms verified |
| **Human labels pending** | R13a/B24 empirical validation and empirical threshold fitting; **not R06b/B15 code** | User agreed to label the 60-item pilot. Fill `pilot_worksheet.csv`, follow `PILOT_README.md`, then assess the pilot before proceeding. Full 800-item labeling and a second annotator are not yet committed/completed |
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

The repair increments R01–R05, R03a, R03b and the documentation pass are committed. The working tree at the time of writing carries R03d's config change and R06a (`src/annotate.py`, `tests/test_validation.py`, `data/annotation/`). The standing rule is that **the assistant does not perform Git writes**, so staging and commits are the human's.

### Data artifacts and their status

| Artifact | Rows | Status |
|---|---:|---|
| `interim/headlines_raw.parquet` | 1,412,524 | Rebuilt R03d; manifest `verified_against_pin: true` |
| `interim/headlines.parquet` | 869,183 | The analysis input. **Frozen** — do not rebuild without a dated log entry |
| `interim/dedup_lineage.parquet` | 1,412,524 | One row per raw row; makes the dedup rate recomputable |
| `data/annotation/` | 800 | **Tracked in Git**, unlike everything under `raw|interim|processed` |
| `interim/scores.parquet` | — | Does not exist. No text has been scored |
| `interim/market.parquet` | — | Does not exist. Needs `yfinance` |
| `processed/daily_panel.parquet` | — | Does not exist |

`data/annotation/` is deliberately excluded from the gitignore rules: the labels will be the study's own experimental data and the most expensive artifact in the project to reproduce.

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
  dedup-lineage-contract.md         R03a, implemented R03b, executed R03d
  project-audit-2026-09-09.md       the audit, findings A01-A15
  audit-implementation-plan-2026-09-09.md   the live repair sequence R01-R15
  handover.md                       this file
config.py                           every locked decision and pinned artifact
src/data.py       loading, mandatory source filter, dedup + lineage, market, calendar
src/annotate.py   the blind Act 1 sample: strata, article groups, split, exports
src/audit.py      per-source profiling; withholds rates a cluster sample cannot support
src/align.py      both mappers, daily aggregation, panel, all lags and leads
src/scoring.py    three scorers behind one protocol, hash-keyed cache
src/validate.py   Act 1: label ingestion, calibration-only thresholds, group bootstrap
src/inference.py  eligibility ledger, frozen primary spec, session-indexed HAC,
                  14-test secondary family (BH/BY), paired contrast, timing shift
src/plots.py      one function per figure; titles name their axes, not conclusions
tests/            418 passing, 0 skipped: alignment · audit · data · market ·
                  panel_fields · primary · hac_spacing · scorer_contrast ·
                  timing_diagnostic · inference · inference_reporting · scoring ·
                  checkpoints · acquisition · validation · paired_validation
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
