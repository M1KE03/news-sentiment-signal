# Project audit — 2026-09-09

The project has a useful corpus, explicit research protocols, and substantial tested infrastructure. It is **not ready for substantive scoring or inference**. Several safeguards described as complete still fail under realistic inputs, and the executable analysis largely follows the superseded specification.

This is an audit and implementation proposal, not implementation approval. No production code, protocol decisions, corpus files, environment settings, or Git state were changed. See the [proposed implementation sequence](audit-implementation-plan-2026-09-09.md).

## Scope and evidence

Reviewed the current implementation plan, original plan as historical context, handover, decision log, scope draft, validation/inference/timing protocols, corpus audit, configuration, all source modules, entry points, tests, notebook cells, README, report skeleton, requirements and local artifact inventory. The unrelated master's-program reference open in the IDE is outside this project audit.

Baseline: commit `0543d36` (`Enhance scoring and classification infrastructure`), initially clean working tree. No repository `AGENTS.md` was found in the inspected project/ancestor paths.

Verification performed:

- Existing suite: `.venv\Scripts\python.exe -m pytest -ra --tb=short` → **120 passed, 7 skipped**. Initial sandbox execution could not access pytest temporary directories; the approved rerun succeeded. All seven skips are FinBERT/PyTorch application-control failures, not VADER skips.
- Read-only corpus inspection: **869,205 rows**, **zero duplicate headline IDs**, all retained source values `benzinga.com`. Both raw and deduplicated parquet files exist. Scores, market parquet, daily panel and annotation files are absent.
- Small synthetic probes exercised real cache and aggregation functions, and `load_market` with mocked price downloads. They used disposable caches, no model inference or real sentiment–return estimation.
- Recomputed the census on the existing deduplicated corpus without rerunning deduplication or changing the window.
- Checked upstream documentation for yfinance end-date semantics and FinBERT training provenance. Installed statsmodels source was also inspected for HAC's spacing assumption.

Labels below distinguish **reproduced defects**, **static findings**, **known unfinished work**, and **protocol issues**. Unfinished B10–B28 work is not presented as a regression.

## Priority findings

### A01 — High: cache provenance does not reliably protect reuse

**Reproduced; reopens B13.** Evidence: [scoring.py](../src/scoring.py), `score_all`, `load_cache`, `_checkpoint`, and scorer fingerprints.

Four independent cases failed:

| Probe | Observed behavior | Required behavior |
|---|---|---|
| All scorer columns populated; default `score_all` call | No scorer is constructed or validated; old values are reused | Validate measurement identity even when no score is missing |
| Same FinBERT-shaped fingerprint before and after saving JSON | Raises `IncompatibleCache`: integer `id2label` keys reload as strings | Canonical serialization must round-trip without changing identity |
| Populated legacy cache with no metadata | Existing `0.8` values accepted; replacement scorer called zero times | Unknown provenance must require explicit rebuild/migration |
| Version 1 cache of A/B; rescore only A under version 2; resume A/B | Cache contains `[-0.5, +0.5]` under version-2 metadata; B is never recomputed | Invalidate every affected row or retain per-generation provenance |

Additional static defects: FinBERT's fingerprint uses `config.FINBERT_REVISION`, even when a different `revision` was supplied to its constructor. VADER uses a possibly absent module `__version__` and lexicon size, which cannot identify same-size lexicon changes. LM identifies its word lists but not the scoring/tokenization implementation. Input IDs derive from normalized text and date while models score raw text; a raw-text edit preserving the ID can also reuse an incompatible measurement.

**Repair:** define canonical, versioned fingerprints from actual loaded artifacts/settings and input content; validate all required columns without requiring GPU inference; reject missing provenance; make subset invalidation safe. This is the highest-value first code increment.

### A02 — High: checkpoint safety covers only one of two files

**Static finding; B14 verification incomplete.** `_atomic_write_parquet` replaces parquet atomically, then `_checkpoint` writes JSON directly. A crash between those operations can pair new data with old metadata; interruption during JSON writing can leave unparsable metadata. The existing interruption test stops between scoring batches, and the atomicity test only checks that temporary files are gone after success.

**Repair:** propose one consistent checkpoint generation containing data and provenance, with a recoverable commit mechanism. Inject interruption before/after each write boundary and verify recovery or an explicit refusal. Atomic writes of two separate files alone do not form a transaction.

### A03 — High: the daily panel accepts incomplete scoring

**Reproduced.** [align.py](../src/align.py), `aggregate_daily`, inner-joins headlines and scores and lets group means skip missing values. There is no one-to-one join validation, per-scorer completeness gate, or finite/range validation. [run_all.py](../run_all.py), `build_panel`, reads parquet directly without validating score provenance.

Probe: a five-headline session with only one finite LM value yields `n_headlines=5`, `s_lm=1.0`. Supplying just one score row silently changes the headline count from five to one. Each scorer can therefore summarize a different subset while the output appears to be one common daily sample. Duplicate score IDs could multiply weights.

**Repair:** establish required scorer coverage before aggregation, verify unique IDs and the input/cache manifest, and distinguish a zero-news session from an incompletely scored session. Bounded scoring must produce explicit completed-session coverage; it must not silently become within-day sampling.

### A04 — High: upstream return construction still crosses missing price sessions

**Reproduced; reopens part of B08.** [data.py](../src/data.py), `load_market`, calculates log-price differences on received rows **before** calendar reindexing. `build_panel` restores missing rows later, which cannot repair a return already computed across a gap.

Mocked closes: Monday 100, Tuesday missing, Wednesday 102. Wednesday `ret` becomes `log(102/100)=0.019803`, despite covering two exchange sessions. After panel construction, Tuesday's `ret_lead1` is also that non-null value. Primary listwise deletion excludes Tuesday for missing controls, but Wednesday can still enter with an invalid current-return control; higher-horizon targets can inherit the same defect. Existing tests supply precomputed returns, so they do not exercise this path.

Also, `fetch_market` passes the inclusive configured end directly to yfinance, whose end is exclusive. The current call would omit 2019-12-31. No pre-window warmup is fetched. [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).

**Repair:** construct prices on the exchange calendar before returns, require both adjacent prices, make fetch bounds explicit, and specify volume/return warmup without extending the analysis window. Test the loader-to-panel path, including the session after a missing price.

### A05 — High: reproduction can replace the clean corpus with undeduplicated input

**Static finding.** [download.py](../data/raw/download.py), `assemble_corpus`, defaults to `HEADLINES_PARQUET`, writes the filtered frame directly, and never calls deduplication. Running `--assemble` can overwrite the existing analysis input with its pre-dedup counterpart. The raw-to-clean/census sequence used to create the current artifacts is not an integrated documented command.

The same file downloads `/resolve/main/` instead of using `NEWS_HF_REVISION`; configured byte length and SHA-256 are not enforced by assembly. Thus the pin records intent but does not bind reproduction to that artifact. This does **not** establish that the existing corpus has the wrong provenance; the missing guarantee concerns the executable reconstruction path.

**Repair:** separate raw filtered and clean destinations, enforce the pinned download and streaming hash/length validation, persist acquisition/dedup manifests, and make assembly → dedup → census a documented sequence. Publish final artifacts only after validation.

### A06 — Medium: corpus coverage is computed using two different mappings

**Reproduced on the real saved corpus.** [audit.py](../src/audit.py), `corpus_census`, first uses the correct deferred date mapping, but passes the retained rows into `coverage_profile`, which remaps them with the intraday close rule. Its claim that this leaves the distribution materially unchanged does not establish that equivalence.

| Quantity | Deferred session counts | Returned coverage counts |
|---|---:|---:|
| Assigned headlines | 869,114 | 869,028 |
| Zero-news sessions | **1** | **2** |
| Zero-news dates | 2010-01-04 | 2010-01-04 and 2019-12-31 |

Counts differ on **2,502 of 2,516 sessions**. The primary mapping has 91 unassignable headlines, as documented. The extra zero on 2019-12-31 is a reporting artifact. The true zero on the first session is also affected by the deliberately conservative lower-edge rule, so it should be identified as a boundary exclusion rather than evidence of a news outage.

The yearly stability table uses the correct mapping. This finding does not by itself justify changing the frozen window.

**Repair:** calculate all census statistics from the single assigned-session vector; distinguish boundary loss from absent coverage; reconcile handover/config/audit numbers.

### A07 — High: the primary model does not implement the frozen specification

**Known unfinished B16/B17/B20 work, with a concrete mismatch.** [inference.py](../src/inference.py), `predictive`, uses raw `log_turnover`; [inference protocol](inference-protocol.md) §§1–3 specifies trailing-63-session-detrended log volume. This changes the model, not merely its column names.

Tone is not standardized at fit time. Later bps conversion uses the broader positive-news sample's SD, not necessarily the regression's complete-case sample. `analysis_sample` reports zero-news exclusions only; it does not produce the promised per-reason eligibility ledger. The runner neither isolates the primary unadjusted test nor performs the advance precision assessment.

**Repair:** explicit primary eligibility/standardization, frozen transformed controls, exact observation IDs and exclusion counts, plus fixture tests where raw and detrended volume produce distinguishable designs. Finish this before inspecting any real coefficient.

### A08 — High: remaining inferential outputs still follow the old plan

**Known unfinished B18–B20 work.** Evidence: `lag_family`, `permutation_pvalue`, `attenuation_comparison`, `volatility_spec`, `volume_spec`, `effect_size_table`, and `run_analysis`.

| Current behavior | Accepted protocol |
|---|---|
| BH separately on each scorer's five horizons, including FinBERT h=1 | One unadjusted primary; BH and BY across the other 14 return tests |
| Blocks drawn with replacement; result named and plotted as permutation p-value | Circular-shift coefficient distribution and descriptive percentile |
| Compare raw coefficient magnitudes and separate t-statistics | Standardized paired difference on identical observations with cross-equation HAC covariance |
| Correlation >0.9 automatically described as inconclusive | Correlation/VIF diagnostics; paired interval determines precision |
| Individual volume/range coefficients receive prominence | Joint HAC Wald tests of tone/intensity/dispersion; separate exploratory family |
| `clears_costs` and cost-bar framing | SESOI comparison and pointwise uncertainty, without strategy-profitability claims |

**Repair:** implement these as separately testable increments. Core result production should wait until its required methods are ready; optional exploratory outputs should not run automatically as a side effect of the primary analysis.

### A09 — Medium: HAC and diagnostics still compress missing sessions

**Static finding; magnitude not yet measured.** `analysis_sample` drops zero-news rows; `nw_ols` drops missing values and resets indices before HAC. Consequently HAC lag 5 means five retained observations, which may span more than five sessions. `figure_acf` similarly drops missing values and labels the axis trading days. Fixing feature shifts does not fix covariance time spacing.

The installed statsmodels `cov_hac_simple` documentation explicitly assumes consecutive, equally spaced periods (`.venv/Lib/site-packages/statsmodels/stats/sandwich_covariance.py`).

**Repair:** preserve session positions in the uncertainty/diagnostic contract; specify and verify how gaps contribute to HAC products, or explicitly justify a retained-observation interpretation. Assess this on synthetic gapped data before results. The current corpus has few news gaps, but missing prices/scores could make the issue larger.

### A10 — High before inference: the protocols themselves need a small correction pass

**Protocol issues; do not silently implement literal errors.**

- Inference §5 treats `1.96*sd(r)/sqrt(n)` as an impossibility bound. It is a rough planning approximation, not a bound on the eventual controlled HAC interval. Controls can reduce residual variance, regressor collinearity can inflate uncertainty, and dependence matters. Report an anticipated width and its assumptions, not “cannot ... no matter what is estimated.”
- Its three conclusion categories overlap: a CI of `[1,3]` bps excludes zero **and** lies within ±5 bps. Boundary cases at exactly ±5 also need a rule. Preserve two properties—association evidence and practical smallness—or specify reporting precedence.
- Validation §7 puts secondary classifier contrasts into a family supposedly defined by B02. B02 enumerates only the 14 return tests and RQ4 Wald family. Define a separate classification treatment; do not quietly add classifier tests to the 14.
- Validation defines article-group dependence but retains ordinary exact item-level McNemar. If multiple dependent items survive per group, pairing alone does not address that dependence. Specify the supplementary test's unit/limitations or a group-aware accuracy comparison.
- Validation §2 equates new human labels with verifiable checkpoint independence. Independent annotation does not establish that the underlying historical text was absent from pretraining/fine-tuning. PhraseBank exposure is explicit in the [FinBERT model card](https://huggingface.co/ProsusAI/finbert); broader non-overlap requires evidence or a qualified statement. A contaminated benchmark score is also not a mathematically guaranteed upper bound on generalization.
- Circular shifting preserves the multiset and circular dependence structure; it does not exactly preserve every conventional finite-sample ACF after a changed boundary. Specify the shift domain when eligible dates have gaps and how percentile ties are handled.

**Repair:** document these limited amendments before implementing B15/B18–B20. Retain the accepted research question, frozen controls, window and 5 bps yardstick.

### A11 — Medium: annotation is ready to prepare, but not implemented

**Known unfinished B11/B15/B24 work.** The handover says B11 is blocked on corpus assembly even though assembly is complete. Human labels block empirical evaluation, not preparation of the blind sample or synthetic bootstrap tests.

`validate.split` remains a row-level 20/80 split, versus the protocol's stored group assignments and target 200/600 calibration/evaluation allocation. The corpus schema has no article-group ID or dedup lineage. `ladder_table` still labels score differences as vocabulary/context gains. Notebooks 02–05 are mostly imports and commented calls, not output-producing workflows.

**Repair:** implement persistent group-aware sampling, blind exports, pilot/label provenance and import validation; implement paired bootstrap independently of label collection. Human judgments remain an external dependency. Keep the incompatible PhraseBank loader off the critical path unless that supplementary branch is selected.

### A12 — Medium: deduplication loses evidence used to justify the universe

**Static finding and bounded implementation defect.** [data.py](../src/data.py), `dedup`, retains one representative's ticker list when duplicate rows represent the same headline under different tickers. [audit.py](../src/audit.py), `concentration_profile`, then treats retained tags as company representation. Without merging tags or retaining source-row lineage, the documented 2.1% top-10 share measures surviving tags, not necessarily the companies actually covered by the text. This does not prove high concentration; it weakens the current assurance of low concentration.

Exact dedup uses the first-ever occurrence as the window anchor, so later clusters of exact repeats are not removed by the exact pass; default near-dedup catches them but misattributes them to the near count. With `near_dupe=False`, later exact clusters remain. Equal-timestamp ordering should also be stable with an explicit representative rule.

The exposure calculation `ceil(2/(1-0.9))` evaluates to **21**, although the documented missable length is 20 unique tokens. The quoted long-headline exposure therefore omits the 20-token boundary. The approximate near-dedup limitation itself is already disclosed and need not trigger an expensive full pairwise replacement.

**Repair:** retain representative/group lineage and union ticker tags; measure concentration at a clearly named unit; fix the exact-window and threshold-boundary logic. Quantify changed corpus membership before replacing frozen artifacts or drawing annotation samples.

### A13 — Medium: the declared environment is not the tested environment

**Observed.** Selected installed vs declared versions:

| Package | requirements.txt | Installed |
|---|---|---|
| numpy | 2.3.3 | 2.5.3 |
| pandas | 2.3.2 | 3.0.5 |
| pyarrow | 21.0.0 | 25.0.1 |
| scipy | 1.16.1 | 1.18.1 |
| statsmodels | 0.14.5 | 0.15.0 |
| scikit-learn | 1.7.2 | 1.9.0 |
| torch | 2.8.0 | 2.14.0+cpu, DLL blocked |
| transformers | 4.56.1 | 5.16.1 |
| pandas-market-calendars | 5.1.1 | 5.4.0 |
| pytest | 8.4.2 | 9.1.1 |
| yfinance / datasets | 0.2.66 / 4.0.0 | Both absent |

The 120 passing tests establish behavior in this local environment, not in the pinned environment. FinBERT fixtures catch broad exceptions and skip, which can hide a code or label-contract defect as “unavailable.” Runtime/truncation checks have not been performed; describing truncation as safe merely because inputs are headlines is premature.

**Repair:** separate offline unit verification from explicit model integration checks; narrow skip conditions; provide a preflight inventory and validate a chosen environment in isolation. Do not change Windows security settings as an automatic project repair. A usable scoring environment remains an external prerequisite.

### A14 — Medium: current documentation can send the next session down the wrong path

**Observed.** The replacement plan still says implementation has not started and B01 is next. The handover simultaneously says torch is installed/blocked and not installed; lists B12 as both complete and future work; and describes old uncommitted files despite a clean initial working tree. The decision-log ending includes both “corpus assembled” and “no corpus assembled.” Notebook 01 still calls the fallback inert and the window provisional.

README/report/notebooks preserve the old PhraseBank-first narrative, classifier-gain attribution, expected results, same-day claims, per-scorer FDR and placebo p-values. Figure titles assert conclusions before results, including “Same-day association, next-day nothing” when same-day estimation is suppressed. README documents `download.py --all`, which the parser does not support. `run_all` does not produce Act 1, and notebook 02 does not yet do so either.

**Repair:** update current-status surfaces and remove predetermined conclusions now; write empirical captions later. Keep the original plan clearly historical. Publish one current status ledger distinguishing specified, implemented, fixture-tested and empirically verified.

### A15 — Medium: readiness and result provenance stop short of the runner

**Static finding.** `_check_locked_decisions` checks a few non-null constants and the fallback implementation flag. It does not check frozen-window status, unresolved dictionary provenance, complete scores, current method readiness, or panel provenance. `--skip-panel` accepts any existing parquet and can label old timing/scoring semantics using current config. Outputs are written sequentially into shared paths, so a failed run can leave a mixture of old and new tables.

**Repair:** preflight validated artifacts and protocol identity before estimation; embed input hashes, transformation settings, eligibility counts and version identifiers in a run manifest; validate reused panels; stage outputs and publish a completion manifest only after all requested outputs succeed. Scoring need not be rerun to reproduce results from verified cached measurements.

## Additional bounded improvements

- Source filtering uses substring membership (`any(d in h ...)`), accepting e.g. `notbenzinga.com`; use parsed, normalized exact domains or true subdomains. No such host was observed in the saved clean corpus.
- Preserve failed/missing timestamp and empty-text counts separately; current loading statistics do not fully reconcile every exclusion reason and their comments retain the corrected byte-slice corruption interpretation.
- Treat the 2010 “roughly doubles sampling variance” statement as conditional on independence and similar per-headline variance, not an established result from counts alone.
- `config.AGG` is not consulted by `aggregate_daily`; a future median sensitivity must explicitly change the operation and record it rather than changing an inert flag.
- Keep timestamp semantics qualified: deferring a reported date cannot prove actual historical availability if that date represents later update/collection. The retrospective interpretation remains essential.

## What should be retained

Keep the existing module structure, separate expensive scoring command, common panel, source-specific audit, conservative date-only mapping, structural same-day suppression, and frozen research question/window. Keep the distinction between classification quality and market association, paired uncertainty, explicit null/inconclusive outcomes, and the modest initial scope.

Actual intraday closes remain legitimately descoped for this date-only corpus. Dashboards, trading backtests, single-name expansion and chronological forecasting are not repairs required by this audit.

## Audit limits

No live market download, model pilot, full scoring, human annotation, substantive regression, clean-environment install, or full artifact reconstruction was run. The existing raw-to-clean dedup pass was inspected but not recomputed over 1.4 million rows. Corpus concentration bias and HAC-gap sensitivity were identified structurally, not quantified as empirical effects. The audit identifies implementation risks; it does not establish any market finding.
