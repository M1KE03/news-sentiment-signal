# Audit implementation proposal — 2026-09-09

Based on the [project audit](project-audit-2026-09-09.md). This repair sequence sits within the existing B01–B28 roadmap. Initially prepared for review; the user subsequently requested execution in bounded increments.

## Execution checkpoint — 2026-09-10

### Latest: R11 — length-sorted batching, and three provenance findings

**Authorized by the user** ("i accept 1-2"): implement the batching optimisation, then run the full pass. The pass is running against 869,183 headlines. 426 tests pass, 0 skip.

**The optimisation is 2.2x, and my claim that it "changes no output" was wrong.** Before implementing it I measured whether batch composition moves the numbers, and it does. Scoring the same 512 real headlines three ways — natural order, length-sorted, and a random permutation, each restored to input order — gives max differences of **4.2e-6**, **1.5e-6**, and 4.2e-6 against `batch_size=1`. Padding is masked out of the attention, but float32 accumulation over different tensor shapes is not bit-identical. Only 159 of 512 scores were exactly equal between natural and sorted order.

4.2e-6 on a score in `[-1, 1]` is immaterial to everything this study reports: it survives a ~345-headline daily mean, standardization, and conversion to basis points printed at 0.1 bps. But "immaterial" and "no change" are different claims, and the second one was the one I made.

**Finding 1 — the fingerprint was already incomplete.** `FinbertScorer.fingerprint` documents itself as "every setting that changes the number". `batch_size` was not in it, and `batch_size` demonstrably changes the number: `batch_size=1` and `batch_size=32` disagree by up to 4.2e-6 on the same text. That gap predates this increment; measuring the batching question is what exposed it. `batch_size` and the new `batch_order` are both recorded now, so the docstring's claim is true.

**Finding 2 — resume was never bit-identical for FinBERT.** The checkpoint contract says an interrupted pass "produces the same result as an uninterrupted one". For the cache mechanics — which rows carry which committed values — that holds. For FinBERT's float32 output it does not and never did: resuming scores only the surviving unscored rows, which form different batches. `tests/test_checkpoints.py` could not catch this because it resumes with a Stub scorer returning a constant. The guarantee is over rows and values, not over the last few ulps, and the fingerprint docstring now says so.

**Finding 3 — the two fingerprint paths diverged, and the guard caught it mid-launch.** `_configured_fingerprints()` derives FinBERT's expected identity from config **without importing torch**, and `score_all` compares it against the loaded scorer before writing anything. I added the two new fields to the scorer and not to the config path, so the first full-pass launch aborted with `IncompatibleCache: loaded finbert identity differs from preflight` — after loading the corpus and the model, before writing a single row. That is the R01a/R01b guard working exactly as designed. Both paths now carry the fields, and a test asserts they agree field for field, which is what makes the next such mistake cheap instead of a wasted launch.

**Measured, on real headlines.**

| | natural order | length-sorted |
|---|---:|---:|
| Throughput | 40–54 headlines/s | **113 headlines/s** |
| Token positions computed | 2.33x the real tokens | **1.01x** |
| Projected full pass | 4.49 h | **2.14 h** |

Padding is what a mixed-length batch pays for: every batch is padded to its longest member, and on a corpus averaging 21 tokens with a p99 of 55, that more than doubles the arithmetic. Grouping by length removes 56.5% of it. Results are returned in input order; the sort is internal, and a test asserts a long headline cannot swap places with short ones on its way through the batcher.

`FINBERT_BATCH_ORDER = "length_sorted"` is in `config.py` and in the fingerprint, so changing it invalidates the cached column rather than silently mixing two orderings.

### Previous: R11 (pilot half) — measured, not estimated

**R11's pilot is complete; the full pass awaits separate authorization.** 418 tests pass, 0 skip. The scoring cache now holds one bounded, complete-session chunk; no panel, no model fit, no result.

Every number below was measured on this machine on 2026-09-10, against the pinned `ProsusAI/finbert` revision `4556d130…`, on CPU (8 torch threads of 16 logical CPUs). None is an estimate carried over from the original plan.

| Quantity | Measured | Note |
|---|---:|---|
| FinBERT throughput | **54 headlines/s** | 2,000 headlines in 37.2 s, batch 32, dynamic padding |
| Projected full pass | **4.49 h** for 869,183 headlines | over the plan's 2 h budget |
| Truncation at `max_length = 64` | **3,921 headlines, 0.45 %** | mean 21.2 tokens, median 19, p95 42, p99 55, max 139; whole corpus tokenized in 17 s |
| Storage | **128 B/row → ≈ 111 MB** for the full cache | three float32 score columns plus provenance |
| Bounded chunk | 10 calendar days, **546 headlines**, 3 scorers, 17.4 s | `rescore.py --sessions 10 --checkpoint-every 200` |
| Per-session coverage | **complete**: 546 of 546, 0 missing, 0 extra | every headline of every touched day, exactly once |
| Fingerprints | **all three verified** against the configured scorers | LM `1993–2025 (March 2026)`, wordlist SHA-1 `35fe7553…`; VADER lexicon 7,506; FinBERT revision + `max_length` |
| Resume | re-running the identical command: **0.2 s, zero rescoring** | durable stop/resume on real data, not only on injected failures |

**Truncation is now a number, not an assurance.** Audit A13 named "truncation is safe: these are headlines" as premature. It is 0.45 %, and the truncated tail is a specific kind of text: earnings-guidance and option-alert headlines that string several figures together (`…Q4 Adj. EPS $0.48–$0.54 vs $0.49 Est., Sales $1.719B–$1.769B vs…`). For those, the sentiment-bearing words are almost always in the first 64 tokens and the cut removes trailing numbers. Raising the cap to 96 would lower truncation to 0.07 % at roughly 1.5× the compute. **The cap is a fingerprinted setting**, so changing it is a different measurement and a new cache, not a tweak; it is left at 64 and reported.

**The 2 h budget is exceeded, and that is a decision, not a repair.** The plan's rule when the projection is unacceptable is to shorten the *window* (D4) and record it — never to subsample within days. D4 is frozen on coverage evidence and unfreezing it needs a dated decision-log entry from the user. Two facts bear on that decision: 4.5 h is a **one-time** cost (the cache resumes across interruptions and never needs re-running unless a fingerprint changes), and the 2 h figure was written into the original plan before the corpus existed. One pure-performance option changes no output: sorting each chunk by token length before batching so batches are homogeneous, which on a mean of 21 tokens against batch maxima near 40 would plausibly recover 1.5–2×. It is not implemented, because it touches the scoring path and belongs to a deliberate choice rather than a side effect of measuring.

**One provenance gap surfaced.** VADER's fingerprint records `version: unknown` — the package exposes no version attribute the scorer can read. The lexicon size (7,506) is recorded and does identify the word list, so the measurement is still pinned; but the field should read the installed distribution version from `importlib.metadata`, which `src/preflight.py` already does for the dependency report. Carried, not fixed here.

**What the bounded pass proves that the tests could not.** `tests/test_checkpoints.py` injects failures at commit boundaries; this pass exercised the same contract with a real transformer, a real 869k-row corpus and a real filesystem, and the re-run found nothing to do. The lock file `scores.parquet.lock` remains after a clean exit by design (R01d) and is reacquired on the next run.

### Previous: R10 — preflight and pilot readiness

**R10 complete. 23 of 29 increments done. A13 closed; A15's readiness half closed.** 418 tests pass, 0 skip.

**A13, quantified before it was repaired.** The new `preflight.py` reported the finding as a number on its first run: of fourteen declared requirements, **one matched what was installed, eleven differed, and three were absent entirely**. Every pin was a version the project had never executed against, so a green suite established behaviour in *some* environment rather than the pinned one. `requirements.txt` is now pinned to the environment the 409-test suite actually runs in (Python 3.12.10), with the three packages that are genuinely not installed — `yfinance`, `datasets`, `jupyterlab` — moved into a clearly separated block that says nothing has been executed against them.

The repair is not a promise to keep them in step. It is `python preflight.py`, which reports the difference on demand, `--freeze`, which prints correct pin lines from the live environment, and a **standing test** that fails when `requirements.txt` stops describing the environment. Restoring the original `numpy==2.3.3` pin makes it fail, so it is the assertion that would have caught A13 in the first place.

**One consequence worth stating plainly:** `yfinance` is absent, so `data/raw/download.py --market` cannot run here and the market artifact cannot be built in this environment. That was previously invisible — it would have surfaced as an ImportError partway through a command. It is now a line in the readiness report.

**A15's readiness half.** `_check_locked_decisions` read three config constants and stopped, so a missing corpus surfaced as a parquet read error several frames inside a loader. `require_artifacts(stage)` now runs first in both `run_all.py` and `rescore.py` and names the artifact, where it was expected, and the command that produces it — as a plain message, not a stack trace, because a traceback says "this program broke" when the truth is "run that first". The registry distinguishes artifacts a **command** produces from ones a **person** must supply (the LM dictionary has no stable URL; the labels need an annotator), because those are different instructions.

**Bounded scoring is bounded by session, never by headline.** `rescore.py --sessions N` takes the first `N` calendar days *whole*. Slicing at an arbitrary headline count would leave a partial day in the cache, and a partial day is not a smaller sample — it is a different measurement, since `S_t` is a within-day mean and `d_t` a within-day standard deviation. `--dry-run` reports scope and checkpoint location without scoring; `--checkpoint-every` exposes the commit interval. A bounded pass prints that the cache will be incomplete and that the score-to-panel gate will refuse it until every in-window session is scored.

**Skips were already narrow; the docstring promising them was not.** `tests/conftest.py` marks model-dependent tests `integration` and they error normally, so `-m "not integration"` is a choice the person running the suite makes rather than one the suite makes silently. `tests/test_scoring.py` still promised that those tests "skip cleanly" when the model is absent — the behaviour A13 named, and the reason a failing label-order assertion had been reported as passing for weeks. The docstring now records what actually happens and why it changed.

**A file was overwritten and then merged back.** A root `preflight.py` already existed at HEAD, written by concurrent work on this same increment, and it was overwritten before being read — my error. It was recovered from git and **merged rather than discarded**, because it carried the substantive half: parquet schema validation, the LM dictionary SHA-256 check against `config.LM_DICT_SHA256`, complete-score-cache verification with scorer-identity comparison, optional model loading, and a JSON report R14's run manifest will want. What this pass added on top is the artifact registry with `produced_by`/`external`, the `require_artifacts`/`require_packages` gates wired into both runners, the dependency status taxonomy, `--freeze`, and 30 tests where there had been none. The two stage vocabularies were reconciled onto the one already committed — `unit`, `pilot`, `scoring`, `analysis`, `validation` — so the project has one set of stage names, and a test asserts it.

**Two defects in the new module, caught by its own tests.** `parse_requirements` took `path=REQUIREMENTS` as a default argument, which binds once at import, so the module could not be pointed at a fixture file — exactly the coupling to the ambient machine it exists to remove. The path is resolved at call time now.

**A known-flaky test, measured rather than dismissed.** `test_checkpoints.py::test_hard_stop_around_commit_and_resumption[after_text]` failed twice during this session's work, both times while the machine was under heavy concurrent load. Measured since: **0 failures in 30 isolated runs, 12 whole-file runs and 10 full-suite runs.** It kills a real subprocess at a commit boundary, so the failure is a load-sensitive race in the test's own timing rather than anything in the checkpoint contract, and nothing in R08–R10 touched `src/scoring.py`. Recorded in the handover's carried-defects table rather than waved through as flake — a test that fails one run in ten is how a real defect hides.

### Previous: R09 — documentation reconciled against the code


**R09 complete. 22 of 29 increments done. A14 closed.** 388 tests pass, 0 skip. Prose only; no behaviour changed.

Every documentation surface was read against what the code does now, rather than against what it did when the text was written. Nothing here was failing loudly, which is the point — stale documentation is silent, and the cost lands on whoever reads it next and believes it.

**Withdrawn procedures still described as current.** The README listed Act 2's *three* conclusions, which M2 withdrew for overlapping — an interval of `[1, 3]` bps satisfied two of them at once and no rule chose between them. It gave McNemar as the primary accuracy comparison, which M4 replaced with the group bootstrap. It described block permutation as "a null that assumes nothing about the error process", which is doubly wrong: §11 forbids calling any procedure here assumption-free, and the block path was deleted at R08c. It called the 5 bps SESOI a "transaction-cost benchmark" delivering "economic significance", which P19 forbids.

**The report contradicted itself two pages apart.** §5 correctly called the yardstick "a measure of *smallness*, not a profitability threshold"; §9 called it a transaction-cost benchmark. Its §2 inference bullet was two generations stale — per-scorer BH, and "circular block permutation, block 21, 1,000 draws, seed 20260830", a procedure and three settings that no longer exist. §6.3 still carried the correlation-above-0.9 rule R08b removed. Its subtitle asked whether FinBERT's classification skill "survives as a market signal", presupposing what Act 1 exists to measure.

**Notebook 04 still instructed the predetermined finding** the audit set out to remove: RQ4 as "the act most likely to yield a positive result", and "report the d_t coefficient prominently: disagreement predicting volume is the project's most plausible positive finding". Both the promotion the joint-null ordering exists to prevent, and the "disagreement" reading P21 forbids.

**The decision log's standing status paragraph had decayed for the second time.** It quoted the pre-R03d corpus count (869,205, not 869,183), said "no annotation sample has been drawn" after R06a drew 800 items, and named Smart App Control and the missing LM dictionary as live blockers after both were resolved. R05 had corrected this same paragraph for the same reason. A standing status statement decays silently because nothing fails when it is wrong; it is now dated on every edit, with a note saying why.

**This plan disagreed with itself.** Nine increments completed in earlier sessions carried no tick, and every prose count said "thirty increments" for a table that has always listed **29**.

**Four carried defects were already fixed and still listed as open.** That is the same failure as the reverse — it costs a later session the time to rediscover them — so they are struck through with what closed them rather than deleted.

**Two code defects survive and were deliberately not touched**, because R09 changes prose and not behaviour: the source filter matches by substring, so a host like `notbenzinga.com` would pass the mandatory domain filter (`src/data.py:112`); and `config.AGG` is inert, since `aggregate_daily` hardcodes `.mean()`. Both are re-verified open in the handover's carried-defects table.

### Previous: R08b and R08c — A08 closed


**R08b and R08c complete. 21 of 29 increments done.** 388 tests pass, 0 skip. No empirical result was computed.

**R08b — comparing scorers.** "FinBERT's coefficient is larger" is a statement about `delta = beta_A - beta_B`. The removed `attenuation_comparison` never estimated it: it fitted each scorer separately, tabulated `abs_coef` beside two marginal t-statistics, and applied a correlation threshold that declared the comparison inconclusive on its own. Protocol Section 11 forbids the claim without an interval for the difference, and Section 7 says how to build one.

`paired_scorer_contrast` fits both equations on the **common** sample — sessions eligible for every scorer, with each scorer's own ledger kept and the sessions lost to the intersection counted — and computes the covariance over the **full** parameter vector from each observation's concatenated moment vector. That is what puts `cov(beta_A, beta_B)` into `delta`'s standard error. It matters in an unusual direction: the two tone series are strongly correlated, so the covariance is large and positive and `var(delta) = var_A + var_B - 2cov` is much **smaller** than the sum of the marginal variances. Two separate fits would have overstated the uncertainty about the difference, not understated it.

The stacked estimator calls the same `hac_meat` as everything else. That was deliberate: R07c's argument is that the two spacing conventions are one computation with one differing input, and a second copy of the pairing loop would quietly make that false. `hac_sandwich` was refactored to share it, so the single-equation path is unchanged and verified by its existing tests.

Collinearity is now reported and not acted on — `corr(z_A, z_B)` and the tone VIFs go into the diagnostics, and no threshold short-circuits the result. Section 7(b)'s different estimand is `incremental_contribution`, replacing `horse_race`, which had fitted unstandardized scores against raw `log_volume`.

**R08c — the timing diagnostic.** The old `permutation_pvalue` drew blocks **with replacement**: some observations appeared twice in a draw and others not at all, so it was not a permutation, and the `+1` correction did not repair a null distribution built the wrong way. It returned `p_permutation` and Figure 2 annotated it as `placebo p = ...`, which Section 11 forbids — shifting tone also destroys its relationship with the controls, so the spread is not the null distribution of the conditional coefficient.

`timing_diagnostic` replaces it with a full circular shift of standardized tone over the `n` retained rows in session order (M6's domain), every `k` in `1..n-1`, controls and outcomes left in place; a midrank percentile at the 0.1 bps reporting precision with the tie count returned; and the gap disclosure Section 9 requires, because **a shift of `k` positions is not a shift of `k` calendar sessions** once the retained rows have holes. Nothing in the output is a p-value and the record carries `is_p_value: False` plus the reporting note.

**Both superseded paths were deleted rather than deprecated**, along with `PERMUTATION_DRAWS`, `PERMUTATION_BLOCK` and the runner's `--draws` flag. A callable that returns a field named `p_permutation`, or one that returns `abs_coef` and a per-scorer p-value, is an invitation to quote it; leaving them importable would have preserved exactly the defect the increment removed. The runner now writes `table_paired_scorer_contrast.csv`, `table_incremental_contribution.csv` and `table_timing_diagnostic.csv`, and deletes a stale `table_attenuation.csv` so a table from a superseded method cannot sit beside a current one.

**A bug the tests caught during implementation.** The shift sweep uses Frisch-Waugh so the controls are partialled out once instead of `n` times. The first version rolled the *residualized* tone — but projection and rotation do not commute, so `M(roll z) != roll(M z)` and every shifted coefficient was wrong. The test that compares the fast path against direct refits failed immediately; residualizing after shifting fixed it. The check is kept, and the mutation is confirmed to fail it.

### Previous: R07d and R08a implemented and integrated


The user requested a few tasks after R07c. Its prerequisite code was absent at the start, then appeared in the shared workspace during this task. R07d/R08a now use those `eligibility`, `primary` and `hac_sandwich` interfaces. [Reporting contract](inference-reporting-contract.md) records the APIs, compatibility changes and verification. R07b/R07c's own completion and M7 decision records are tracked by that concurrent work.

R08a replaces per-scorer BH with one unadjusted primary and the exact 14-test secondary return family, with BH/BY, membership validation and BY-based secondary claim decisions. Return fits use the full panel and sample-specific standardization. R07d adds pointwise bps intervals, both evidence dimensions, boundary disclosures, two advance widths and a controls-only residual-mean HAC ratio. The runner atomically writes the advance record before fitting any tone coefficient. **333 tests passed, 0 skipped**, including 38 reporting/family checks. No empirical result was computed. Next: R08b and R08c after the prerequisite decision record is reconciled.

### R07b and R07c — the primary specification, and what `L = 5` counts

**R01a–R01d, R02, R03a–R03d, R04a, R04b, R05, R06a, R07a, R07b and R07c complete — 16 of 29.**

**R07b — the frozen primary specification.** A07's mismatch was not a naming problem. `predictive` fitted **raw** `log_volume` where the protocol specifies the trailing-63-session-**detrended** series, did not standardize tone at fit time, and the basis-point conversion then multiplied by an `sd(S)` from a different sample — rescaling a coefficient the fit had already scaled. `inference.primary` implements the frozen model instead; `predictive` stays in place for the secondary and exploratory paths R08 replaces, and is now documented as not being the primary.

Three properties are held by test rather than by intention. The design is exactly `{z(S_t), r_t, RV_t, lv_t}`. The fit, the standard deviation used to standardize, and the basis-point scale all come from the **same** retained rows — so replacing `S` with `2S` leaves `beta`, its standard error and the whole interval unchanged, which is the invariance the old conversion broke. And the exclusion ledger reconciles: every dropped session is charged to the first of four ordered reasons, the counts partition the excluded rows exactly, and per-reason counts ignoring precedence are reported alongside so overlaps stay visible. The 62-session detrend warm-up is charged to `missing_control`, never confused with `zero_news`.

Two guards protect the timing contract at the point of use. `_require_full_calendar_panel` refuses a panel whose rows are no longer the complete calendar — detected without needing the calendar, because on a complete frame `ret_lead{h}` is exactly `ret.shift(-h)`, and removing any interior row breaks that. This catches the most likely notebook mistake: passing `analysis_sample`'s output, where row `i+1` is no longer the next session. `_assert_lead_adjacency` then restates the invariant per retained observation, as protocol Section 3 asks.

**R07c — M7 decided.** Full argument and measurements: [`hac-spacing-decision.md`](hac-spacing-decision.md).

`L = 5` was prespecified as *one trading week*, and only one of the two available conventions makes that sentence true. `hac_sandwich` takes the session index as an argument, so session-indexed and retained-position are the same computation with one differing input, and any divergence is attributable to spacing alone. Verified rather than asserted: with `arange(n)` both reproduce statsmodels' own HAC covariance to 1e-12 on a contiguous sample; on a gapped index the result matches lag products written out independently as a double loop over every pair; and when every retained session is more than `L` apart the estimator collapses exactly to the White sandwich, which is the property that makes the bandwidth mean what it says.

Measured on synthetic gapped data (`hac_spacing_study.py`, true tone coefficient zero by construction): **identical with no gaps**, **at most 0.33%** difference on the tone standard error at this corpus's anticipated gap structure, growing to about 7% at 50% gaps with genuine session-time residual dependence. The protocol's expectation that the two would be close is confirmed.

**The selection was made on interpretive grounds, not on the measured outcome** — the argument would be unchanged if every measured number were different. That is a stricter standard than the deferral required, and it was made before any tone coefficient existed. Retained-position is computed on every fit and reported alongside, never substituted. What is **not** settled: the same comparison on the **real** analysis sample, which does not exist yet; `PrimaryFit.spacing_comparison()` produces it, and it is a required manifest entry at R13b.

### Previous checkpoint: R07a — panel field names now match their formulas

**R07a complete (latest). R01a–R01d, R02, R03a–R03d, R04a, R04b, R05, R06a and R07a complete — 14 of 29.** 258 tests pass, 0 skip.

Two panel columns claimed more than their formulas delivered (P22, A07). `log_turnover` is `log(share volume)` — no share-count denominator exists anywhere in this pipeline, so it was never a turnover ratio. `parkinson` was described as volatility when it is a high-low **range** estimator of variance, blind to the intraday path and to the overnight gap.

**The mapping was written before the rename**, as the increment required: [`timing-contract.md`](timing-contract.md) §10 now records every panel column, its formula, where it is built, and what it may not be read as.

Renamed across `src/data.py`, `src/align.py`, `src/inference.py`, `src/plots.py`, `run_all.py` and four test modules: `log_turnover` → `log_volume`, `parkinson` → `rv_parkinson`, with the derived `_lag1`, `_lead1` and `_detrended` columns following their stems, and `build_panel`'s `turnover_window` parameter → `volume_window`.

**The rename's own hazard was the reason for two new guards.** `build_panel` NaN-fills any declared panel column its inputs did not supply. A market frame or saved panel written before this increment would therefore have produced an entirely empty `log_volume`, and every downstream regression would have fitted on a shorter sample and reported it as a sample size — not as an error. `align.reject_legacy_columns` refuses either input by name and says what each stale column became; `align.assert_panel_schema` does the same for a panel read from disk, and `run_all.py --skip-panel` now calls it. `REQUIRED_MARKET_COLUMNS` likewise raises instead of NaN-filling a missing control.

**P29 closed.** The context figure used `df["close_adj"] if "close_adj" in df else np.nan`, which would plot a scalar against a date series. It now requires the column and says which artifact is wrong when it is absent.

Verification: 12 new tests in `tests/test_panel_fields.py` check `log_volume` and `rv_parkinson` against fixture bars arithmetic rather than against themselves, check the detrend window is trailing-only and leaves the first `window - 1` sessions missing, and check that the figure draws the panel's actual prices. Both guards were mutation-checked — disabling either makes four tests fail.

**This increment renamed; it did not remodel.** `inference.predictive` still uses **raw** `log_volume` where the protocol specifies the trailing-63-session-detrended series, and tone is still not standardized at fit time. That is a different model, not a different column name, and it is R07b.

### Dependency update — 2026-09-10 (dictionary and pilot worksheet)

The user requested resolution of the “Blocked by” items and volunteered to label the pilot. Acquired and validated the official LM 1993–2025 CSV (March 2026), pinned release and SHA-256 in `config.py`; see [dictionary provenance](lm-dictionary-provenance.md). Created a blank [60-item pilot worksheet](../data/annotation/pilot_worksheet.csv) and [instructions](../data/annotation/PILOT_README.md) from the frozen sample. No sample redraw, model labels or empirical results.

| Increment | Current dependency |
|---|---|
| ✅ R06b — label ingestion, paired metrics | **Implementation ready:** use synthetic fixtures. Actual calibration/evaluation waits for human labels |
| R13a — empirical validation | Dictionary resolved; independent calibration/evaluation labels and R06b implementation still required. User is starting with the 60-item pilot |
| R11 — pilot/bounded scoring | Dictionary and model-environment blockers resolved; readiness and preceding implementation checks remain. No full scoring authorized or run in this update |

This changes dependency status, not the count of completed research increments.

### Previous checkpoint: 2026-09-09

**R06a complete. R01a–R01d, R02, R03a–R03d, R04a, R04b, R05 and R06a complete — 13 of 29 at that point.**

### R06a — the blind annotation sample is drawn

New `src/annotate.py` and `tests/test_validation.py` (30 tests). The sample is
**drawn and on disk**: `data/annotation/` holds 800 headlines, 200 calibration
and 600 evaluation, ready to hand to an annotator. Nothing here needed a label,
which was the point — the audit found B11 described as "blocked on corpus
assembly" long after assembly finished, when what actually blocks it is human
annotation (A11).

- **Frame:** the 869,183-row rebuilt corpus. **Draw:** proportional allocation
  across 10 strata (year × ticker-tag presence), simple random sampling without
  replacement inside each cell, seed `20260830`, made once.
- **Article groups**, computed within the sample by exact all-pairs comparison
  and **unwindowed** — the relation R03a's contract separated from the dedup
  cluster. 799 groups; one holds two items. A whole group goes to one part, so
  a syndicated headline cannot be threshold-fitted on Monday and evaluated on
  Tuesday.
- **Blind export:** `headline_id` and `text`, and the export refuses to run if
  any of a named forbidden list would reach it. Presentation order is shuffled
  under a **separate** seed (`20260831`), so reshuffling the running order can
  never disturb which rows were drawn.
- **`split_assignment.csv` is the authority.** `load_split` re-checks the
  invariant on the way in and refuses a file where any group spans both parts —
  protocol §4's stated reason for storing rather than reseeding.
- **Pilot:** 60 calibration items identified. **Second annotator:** 160 items
  (20%) for kappa.
- **`provenance.md`** is written at draw time with annotator, dates, blindness
  confirmation and deviations left as explicit **TO BE COMPLETED** blanks — a
  template that pre-fills them produces exactly the reconstructed-after-the-fact
  record §6 forbids. A test asserts they stay blank.

Two strata collapsed to one level on this corpus and are recorded rather than
silently dropped: every headline carries a ticker tag (0.00% untagged), and the
universe is a single publisher by construction.

`data/annotation/` is deliberately **not** gitignored, unlike everything under
`data/raw|interim|processed`: the labels will be the study's own experimental
data and the most expensive artifact in the project to reproduce.

**246 tests pass, 0 skipped.**

### Previous checkpoint: R03d

### R03d — verified rebuild, and a pin that was wrong

The corpus was rebuilt end to end through R02's verified acquisition path and the
artifact replaced. Full numbers in the [dedup lineage contract](dedup-lineage-contract.md) §9.

**The pinned digest was wrong, and R02's guard caught it.** The first assembly
refused to publish: the streamed file's SHA-256 was `5d4c0180…`, not the pinned
`dde52918…`. HuggingFace's `paths-info` API for the pinned revision reports
`lfs.oid` = `5d4c0180…` — Git LFS OIDs are sha256-of-content — and `xetHash` =
`dde52918…`. B04 had recorded the value from the HTTP **ETag**, which HF serves
as the Xet hash for Xet-backed files; `data-audit-fnspid.md` even labelled the
row "File sha256 (etag)". **The file never changed**: same revision, same
5,731,397,037 bytes, same content. Only the pin was mislabelled. `config` now
holds the real SHA-256 with `NEWS_FILE_XET_HASH` kept for traceability. Nothing
was written on the failed run — which is precisely the behaviour R02 was built
for, and it also means the pre-R02 corpus had never had its digest checked.

**The rebuilt raw corpus is byte-identical data**: 1,412,524 rows, identical
`headline_id` set and identical `text_norm` multiset. **Clean corpus 869,205 →
869,183** (−22; symmetric difference 2,314 ids, 0.27%), split **539,087 exact /
4,254 near**, tickers 5,707 → **6,235**, tag slots → **1,385,810**, top-10
2.09% → **2.04%**. Census reconciles at **869,092 + 91 = 869,183** and **D4
holds** — no year below tolerance, one structural zero-news session.

### Previous checkpoint: R03b

### Environment: the scoring blocker has cleared

`torch 2.14.0+cpu` and `transformers 5.16.1` now load and FinBERT runs. The
seven tests that had skipped since B22 execute for the first time, so the suite
reports **0 skipped**. **B22, R10 and R11 are unblocked.** Two external
dependencies remained at that checkpoint: the dictionary and a named annotator. The 2026-09-10 update above records dictionary acquisition and the user's pilot-labeling commitment; completed labels remain pending.

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
| ✅ R01a — cache identity | `src/scoring.py`, `tests/test_scoring.py`: canonical JSON-compatible fingerprints; actual constructor revision; missing-metadata refusal | Same identity survives save/load; changed identity and legacy unknown identity fail; no model download required | Reopen B13; A01 |
| ✅ R01b — cache validation scope | Validate completed columns on the default path; handle input content identity and subset invalidation | Complete-cache reuse checks provenance; changing A in an A/B cache cannot bless B's old values | R01a; A01 |
| ✅ R01c — checkpoint contract | Write exact proposal for versioned data/provenance commit and recovery within the existing cache | Defined interruption states, migration behavior and no-GPU read-validation path | B13/B14 contract checkpoint; A02/A15 |
| ✅ R01d — durable recovery | Implement approved checkpoint contract; inject failures at file-write/commit boundaries | Every interrupted state recovers the last consistent checkpoint or refuses explicitly; resumed scores equal uninterrupted scores | R01c; A02 |
| ✅ R02 — assembly contract | `data/raw/download.py` plus data tests: raw/clean destination separation, pinned revision, stream digest/length, atomic final publication | Small mocked download cannot overwrite the clean corpus as raw; wrong digest refuses publication; offline deterministic test | B04/B26; A05 |
| ✅ R03a — dedup lineage | **Done 2026-09-09** — [dedup lineage contract](dedup-lineage-contract.md). Representative ordering, `source_row_id`, group lineage, unioned tags, re-anchored exact window, corrected blocking boundary; dedup clusters separated from article groups | Impact quantified on the real corpus rather than fixtures: 75.7% of (group, ticker) pairs destroyed; exact/near split wrong by 96.2% of the near count; membership effect only 93 ids; output not order-invariant (10.78% tag churn). Four decisions requested in §7 | B04/B11; A12 |
| ✅ R03b — dedup repairs | **Done 2026-09-09.** `src/data.py`, `data/raw/download.py`, `tests/test_data.py`: re-anchored exact window, total representative order `(ts_utc, source_row_id)`, unioned tags, lineage artifact, corrected boundary; elimination chains resolved to a surviving root | 9 new tests, 216 passing 0 skipped. On the real corpus: order invariance verified (symmetric difference 0, was 2,363 ids and 10.78% tag churn), exact drops 539,087 as predicted, tickers 5,707 → 6,235, rate recomputable from lineage | R03a; A12 |
| ✅ R03c — census correction | `src/audit.py`, audit tests/docs: derive all counts from one session assignment | Totals reconcile; real saved corpus reports 869,114 assigned headlines and one boundary zero; yearly window decision reviewed without outcome fitting | A06; independent of model setup |
| ✅ R03d — corpus verification checkpoint | **Done 2026-09-09.** Rebuilt raw → clean through the verified path; manifest now records `verified_against_pin: true`; lineage persisted | Raw data identical (same id set and `text_norm` multiset); clean corpus 869,205 → 869,183, symmetric difference 2,314 (0.27%); census reconciles 869,092 + 91; **D4 holds**. Surfaced and corrected a mislabelled `NEWS_FILE_SHA256` (HF `xetHash` from the ETag, not a SHA-256) | R02/R03a–c; required before final annotation draw |
| ✅ R04a — market/calendar repair | `src/data.py`, `src/align.py`, loader/panel tests: calendar before returns, inclusive end handling, explicit warmup | Missing Tuesday makes Wednesday's return undefined; no one-session target spans a gap; final session requested; warmup does not expand analysis dates | Reopen B08, prepare B16; A04 |
| ✅ R04b — score-to-panel gate | `src/align.py`, `run_all.py`, alignment tests: unique IDs, complete finite scores, required scorer set, artifact validation | Partial values, missing rows, duplicate IDs and incompatible measurements fail before aggregation | R01, A03/A15 |
| ✅ R05 — protocol amendments | `docs/inference-protocol.md`, `docs/validation-protocol.md`, decision log | **Done 2026-09-09.** M1 precision approximation, M2 overlapping conclusions and boundaries, M3 classifier multiplicity, M4 group accuracy uncertainty, M5 training-overlap wording, M6 shift domain. **M7 (HAC spacing, A09) deferred to R07c by decision** | A09/A10; no real results required |
| ✅ R06a — blind sample preparation | **Done 2026-09-09.** New `src/annotate.py`, `tests/test_validation.py` (30 tests), `data/annotation/` artifacts drawn and on disk | 800 drawn (200/600) across 10 strata; 799 unwindowed article groups, none spanning both parts; blind export is id+text only with a forbidden-column guard and a separate order seed; `load_split` re-checks the invariant; 60-item pilot and 160-item second-annotator subset identified; provenance leaves annotator fields blank by design | B11; R03/R05; no labels required to build tooling |
| R06b — label ingestion and paired metrics | `src/validate.py`, validation tests: validate labels/provenance, fit calibration-only thresholds, paired group bootstrap | Known paired fixtures; identical predictions yield zero difference; all resamples preserve groups/pairing; unusable items counted | B15; R05/R06a; empirical use waits for human labels |
| ✅ R07a — panel fields | **Done 2026-09-10.** Field contract written to [`timing-contract.md`](timing-contract.md) §10; `log_turnover`→`log_volume`, `parkinson`→`rv_parkinson` (and derived lag/lead/detrend columns) across `src/`, `tests/`, `run_all.py`; `volume_window` parameter; legacy-name and required-market-column guards; context figure requires `close_adj` | 12 new tests (`tests/test_panel_fields.py`), 258 passing 0 skipped. Formulas checked against fixture bars; detrend verified trailing-only; guards mutation-checked (removing either makes 4 tests fail). **Renames only** — `predictive` still fits raw `log_volume`, which is R07b | A07; agreed panel contract |
| ✅ R07b — primary fixture | **Done 2026-09-10.** `inference.eligibility` (four-reason ledger), `inference.primary`/`PrimaryFit`, `bps_interval`; `_require_full_calendar_panel` and `_assert_lead_adjacency` | 24 tests (`tests/test_primary.py`). Design is `z(S), ret, rv_parkinson, log_volume_detrended`; tone standardized at fit time on the retained rows, so doubling `S` leaves `beta`, its SE and the bps interval unchanged; bps is `beta * 10,000` with no second rescaling; the ledger's first-reason counts partition the excluded rows exactly. Mutation-checked: reverting the control to raw `log_volume` fails 4 tests, standardizing on the whole panel fails 2 | B17/B20; R04/R05/R07a |
| ✅ R07c — uncertainty spacing | **Done 2026-09-10.** `inference.hac_sandwich` takes the session index as an argument, so both conventions are one computation with one differing input; `fit_hac`/`HACFit`; `config.HAC_CONVENTION = "session_indexed"`; measurement in `hac_spacing_study.py`; **M7 decided** in [`hac-spacing-decision.md`](hac-spacing-decision.md) | 13 tests (`tests/test_hac_spacing.py`). Contiguous case matches statsmodels to 1e-12 under **both** conventions; gapped case matches an independent all-pairs double loop; rows spaced more than `L` sessions apart collapse to the White sandwich. Measured: identical with no gaps, **at most 0.33%** SE difference at this corpus's anticipated gap structure, growing to about 7% at 50% gaps. Chosen on interpretive grounds — only session-indexed makes `L = 5` the one trading week it was prespecified as — **before any tone coefficient existed**. Real-sample comparison remains due at R13b | B17/B20 extension; A09; R05/M7 |
| ✅ R07d — precision/results contract | Pointwise intervals, both evidence dimensions, boundaries, two advance widths and controls-only residual-mean kappa; atomic advance-manifest publication before any tone fit | Synthetic arithmetic and gapped covariance references, identical eligible rows and publication-failure sequencing verified; no profitability field. See [contract](inference-reporting-contract.md) | B20; integrates R07b–c interfaces |
| ✅ R08a — return families | One unadjusted primary and exact 14-test secondary family with BH/BY; runner migrated | Membership/size asserted; primary excluded; full reference corrections and order invariance checked. No empirical fits | B18; R07 remains prerequisite for real estimates |
| ✅ R08b — paired scorer effect | **Done 2026-09-10.** `common_eligibility`, `paired_scorer_contrast`/`PairedContrast` (stacked moment vectors through the shared `hac_meat`, block-diagonal bread), `incremental_contribution` for Section 7(b); `attenuation_comparison` and `horse_race` **deleted** | 21 tests (`tests/test_scorer_contrast.py`). Identical scorers give delta and se both exactly 0 and are flagged degenerate rather than floored; doubling or shifting a score leaves delta, its SE and the bps interval unchanged; the stacked covariance matches an independent all-pairs double loop and its diagonal blocks reproduce each single-equation `fit_hac` exactly. Mutation-checked: dropping the cross-equation term fails 2, abandoning the common sample fails 2 | B18; R07 |
| ✅ R08c — timing diagnostic | **Done 2026-09-10.** `timing_diagnostic`: full circular shift over the retained rows in session order, midrank percentile, tie count, gap disclosure; `permutation_pvalue`, `_circular_block_permute` and their config settings **deleted**; Figure 2's "placebo p" annotation replaced | 17 tests (`tests/test_timing_diagnostic.py`). Every shift is a bijection; Frisch-Waugh coefficients checked against direct refits; a coefficient built to dominate ranks above the 97th percentile of its own shift distribution; no output key is a p-value and the record says so. Mutation-checked: rolling the residualized tone instead of residualizing the rolled tone fails the refit check — a real bug this caught during implementation | B19; R05/R07 |
| ✅ R09 — honest current documentation | **Done 2026-09-10.** README, `report/report.md`, notebooks 02–05, the decision-log status paragraph, this plan's counts and tick marks, and the handover's status, module map and carried-defects table | Verified against the code, not against the prose's own history. Removed: the withdrawn three-conclusion rule (M2), McNemar as the primary accuracy test (M4), "a null that assumes nothing" and the deleted block permutation with its block/draws/seed (R08c, §11), the correlation>0.9 decision rule (R08b), the 5 bps SESOI as a "transaction-cost benchmark" (P19), per-scorer BH (M3/R08a), and notebook 04's instruction to report `d_t` prominently as "the project's most plausible positive finding" (P21/P23). Corrected: the decision log's stale corpus count, drawn-sample and blocker claims; "thirty increments" for a table of 29; nine missing tick marks. **A14 closed.** Two *code* defects re-verified open and left for the audit follow-up: substring domain matching and the inert `config.AGG` | B27 preliminary pass; A14; can be done early |
| ✅ R10 — preflight and pilot readiness | **Done 2026-09-10.** New `src/preflight.py` and root `preflight.py` CLI (`--stage`, `--load-models`, `--cache`, `--output` JSON, `--freeze`, `--strict`), merged with the root script already committed by concurrent work on the same increment); `requirements.txt` re-pinned to the tested environment; `rescore.py` gains `--sessions`, `--checkpoint-every`, `--dry-run`; `run_all.py` and `rescore.py` gate on artifacts before doing anything expensive | 30 tests (`tests/test_preflight.py`), 418 passing 0 skipped. A13 quantified then closed: **1 ok, 11 mismatch, 3 absent** before, 12 ok after, with a standing test that fails if the pins drift again — mutation-checked by restoring the original `numpy` pin. Missing artifacts now name themselves, their path, and the command that produces them, and separate "run this" from "a person must do this". Bounding is **by whole session**, never by headline count | B22 readiness; A13/A15; no security change |
| 🟡 R11 — pilot and bounded scoring | **Pilot done 2026-09-10; full pass awaits authorization.** Measured: 54 headlines/s on CPU, **4.49 h projected** (over the 2 h budget — a D4 decision for the user); truncation at 64 tokens **0.45 %** (3,921), whole corpus tokenized; 128 B/row → ≈ 111 MB. Bounded chunk `--sessions 10`: 546 headlines, complete per session, all three fingerprints verified, resume in 0.2 s with zero rescoring | Rate/storage/truncation measured; fingerprints verified; per-session coverage complete; stop/resume durable on real data. Carried: VADER fingerprint records `version: unknown` | B22/B23; R01–R04/R10; usable model environment and dictionary |
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

The model environment is usable and the actual LM dictionary/release are now available. Independent human labels remain pending, starting with the user's 60-item pilot. Market acquisition still requires a recorded download in the selected environment. R06b's implementation can proceed with synthetic fixtures; its empirical outputs wait for labels.

**Counting note (R09, 2026-09-10).** The table below lists **29** increments; prose in earlier checkpoints said "thirty", which was never true of the table and is corrected throughout. Nine rows completed in earlier sessions carried no tick, so the table disagreed with its own checkpoints; they are marked now. **23 of 29 complete.** Outstanding: R11, R12, R13a, R13b, R14, R15.

The repair program spans multiple reviewable sessions. A credible estimate for full scoring and annotation should follow the measured pilot and annotator throughput, rather than repeat the original one-week estimate.
