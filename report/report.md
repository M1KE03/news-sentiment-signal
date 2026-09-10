# Reading the News with a Machine
## How three sentiment measurements differ, and what they add about the next session

Act 2 is complete. Act 1 is not: it requires independent human labels that have not been collected, and no classification result is reported below. Every Act 2 number here is traceable to a file in `report/tables/`, and `report/tables/run_manifest.json` records the inputs, settings and code state that produced them.

---

### 1. Question and the two-act logic

*(≈200 words.)* How do financial sentiment measurements differ in classification quality, and what additional information do they provide about subsequent market outcomes? The two questions are kept apart: classification quality is measured against independent human labels, and market association is estimated separately. Neither answer is assumed by the other.

**The attenuation bridge, as a conditional hypothesis.** Sentiment scores measure a latent quantity — the information content of the news — with error, and classical errors-in-variables shrinks a regression coefficient toward zero in proportion to that error. That motivates a hypothesis: on the *identical* specification, a better classifier *may* show a larger, better-determined coefficient. The assumptions it requires are stated with it. It is not a mechanism the design establishes, and a null or inconclusive Act 2 does not falsify Act 1.

The [mathematical appendix](../docs/mathematical-appendix.md) sharpens this, and the result is not favourable to the bridge. Under this design, `S_t` is a mean over a median of 337 headlines, and standardization puts the attenuation exponent at ½ rather than 1. A **twofold** difference in per-headline measurement noise becomes a **1.034×** difference in the standardized daily coefficient. The bridge is compressed by the design before any question of whether the scorers differ.

> **Corrected 2026-09-09 (P05/P06, audit A14).** This section previously opened by asserting that "a domain transformer reads financial sentences better than a word counter" as established fact, and described the bridge as a "falsifiable prediction" that made Act 1 "predictive of Act 2". The first is what Act 1 exists to measure and is not yet measured. The second overstated a conditional argument as a mechanism.

**Prespecified hypotheses, with no expected direction attached.** Act 2 tests `H0: beta = 0` two-sided at the 5% level on one primary specification. RQ4 tests a joint null across tone, intensity and dispersion. Each has a stated estimand and a stated uncertainty procedure for that same estimand, and each may return an explicit **inconclusive**.

> **Removed 2026-09-09 (P23, audit A14).** This paragraph previously declared the expected outcomes in advance — "FinBERT wins Act 1 clearly; the same-day association is positive; next-day prediction is weak or null after correction; the volume/volatility act carries the most plausible positive finding". Prespecifying a *hypothesis* is the discipline; prespecifying the *answer* is what the discipline exists to prevent.

### 2. Protocol

*(Written before any result. Every decision below was fixed before estimation; any deviation is logged with its date in the [decision log](../docs/research-review-decision-log.md).)*

- **News source (D1):** the **Benzinga sub-corpus of FNSPID's `All_external.csv`**, selected on relevance and coverage. The file concatenates five-plus sub-corpora with different languages and timestamp behaviour, including the Russian-language `lenta.ru`, so the source-domain filter is mandatory rather than hygiene. The audit's decisive finding: the only intraday-stamped block is Reuters, a global newswire with no ticker tags weighted to European hours — the wrong content.
- **Window (D4):** 2010-01-01 to 2019-12-31, **2,516 trading sessions**, frozen 2026-09-09 on the corpus census and not revisited.
- **Timestamp rule:** the corpus is **date-only**, so the intraday close rule is descoped. A headline dated *d* belongs to the **first session strictly after *d***, so session *t* receives dates in [prev_session, *t*) and all of day *d* precedes close(*t*). The date is read in the source's own zone and never converted.
- **Scorers (D7):** LM (pos−neg)/(pos+neg); VADER compound; FinBERT P(pos) − P(neg). All in [−1, 1]. FinBERT pinned to revision `4556d130…`, `max_length = 64`, batch order `length_sorted`; all three are recorded in the score cache's fingerprint.
- **Aggregation (D8/D9):** equal-weighted daily mean `S_t`; count `n_t`; within-day dispersion `d_t` defined only when `n_t ≥ 5`. Zero-news sessions dropped and counted: **1**.
- **Inference (D10–D12):** Newey–West `L = 5`, prespecified as one trading week and counted in **exchange sessions** ([spacing decision](../docs/hac-spacing-decision.md)). One **primary** test — FinBERT, h = 1, unadjusted — and a closed secondary family of the remaining 14 (scorer, horizon) pairs under BH at q = 0.05 with Benjamini–Yekutieli alongside. The timing diagnostic is an exhaustive **circular shift** reported as a percentile rank.
- **Dedup:** exact `text_norm` matches within 3 days, plus token-set overlap ≥ 0.9. Dedup rate **38.5%** — 539,087 exact and 4,254 near, from 1,412,524 raw to **869,183** analysis headlines.

### 3. Data

The corpus is 869,183 deduplicated Benzinga headlines over 2,516 sessions, at a mean of 345.4 and median of 337 per session (Figure: coverage). Coverage drifts — 52,198 headlines in 2010 against 101,811 in 2011 — which is why the robustness stage splits the sample in half. `n_t` enters no specification, but it shapes the precision of `S_t` and `d_t`.

Three findings from assembly are worth stating because each changed a number. The selected source occupies **two separate blocks** of the source file, and reading only the first would have silently dropped ~122,000 headlines. The exact/near dedup split was initially wrong by 96.2% of the near count, because the exact-duplicate window anchored on a text's first-ever occurrence rather than its last kept one. And ticker coverage was counting surviving tags rather than companies: unioning them raised distinct tickers from 5,707 to 6,235 and *lowered* concentration.

**What the candidate dataset lost on.** FNSPID's other sub-corpora were rejected on stated criteria before any return relationship was examined: `lenta.ru` is Russian-language general news; `seekingalpha` and the smaller blocks are 100% midnight-stamped with a single distinct minute; Reuters is the only intraday block and carries no ticker tags. A rejected dataset with a stated reason is worth more than a chosen one without.

**Truncation, measured rather than assumed.** At `max_length = 64`, **0.45%** of headlines (3,921) are truncated — mean length 21.2 tokens, p99 55, max 139. The truncated tail is guidance and option-alert headlines that chain several figures; their sentiment-bearing words sit in the first 64 tokens.

### 4. Act 1 — classification quality

**Not run. No result is reported.**

The evaluation sample exists: 800 headlines drawn 2026-09-09 across 10 strata, 200 calibration / 600 evaluation, split by article group and stored as a file. The blind export carries `headline_id` and `text` only. What does not exist is labels, which require a person; the tooling to ingest them, fit calibration-only thresholds and compute the paired group bootstrap is implemented and tested against synthetic fixtures.

Until labels exist, **Table 1 and Figure 1 are not produced and no claim about relative classification quality is made anywhere in this report.** Two of the three scorers cannot even produce class predictions: LM and VADER need neutral thresholds fitted on the calibration part, and there is nothing to fit them against.

When it runs, the primary contrast is `macroF1(FinBERT) − macroF1(LM)` with a paired bootstrap resampling **article groups**, B = 10,000. The accuracy comparison uses the same group bootstrap; exact McNemar is supplementary, printed with the group-size distribution and the explicit note that its independence assumption is contradicted by this design's own grouping.

**One limitation is already fixed and must be stated.** The full scoring pass completed on 2026-09-10, *before* any label was written. Validation protocol §5 permits labelling after scoring provided the cache is not consulted, and requires recording which case obtained; it is recorded in `data/annotation/provenance.md`, and the annotator's blindness confirmation must assert that the cache was not opened rather than merely that no score was shown.

### 5. Act 2 — association

**Table 2 — not produced.** Same-day association (RQ2) is **structurally suppressed**: no sub-corpus is both intraday-stamped and relevant, so `inference.contemporaneous` raises and the runner writes no Table 2. Its earlier description as "the pipeline's sanity check — no same-day association means something upstream is broken" is withdrawn (P23).

**Sample.** 2,516 sessions, of which **2,453 eligible**. The 63 exclusions reconcile exactly: 61 to the trailing-63-session volume detrend warm-up, 1 to the window's final session having no lead, 1 zero-news session.

**Precision, recorded before the coefficient was seen.** `report/tables/advance_precision.json`, written 2026-09-10T12:33:32Z: `h1 = 3.67` bps, `h2 = 3.56` bps, `sd(r) = 0.00928`, `R²(tone ~ controls) = 0.0096`, `κ = 0.964`. Both sit inside the ±5 bps SESOI, so an informative null was reachable in advance rather than discovered to be.

**Table 3 / Figure 2 (headline). The primary result.**

> **`β = −1.74 bps per 1 SD of FinBERT tone, 95% pointwise Newey–West interval [−4.90, +1.41], n = 2,453, p = 0.279.`**

The realised half-width is 3.16 bps, slightly narrower than `h2` anticipated. Under the two dimensions that are always reported together:

| | Result |
|---|---|
| Does the interval exclude zero? | **No** |
| Is the interval inside ±5 bps? | **Yes** |
| Conclusion | **No association detected, and precise enough to exclude effects beyond ±5 bps** |

**This is flagged as a boundary case and the flag is part of the result.** The lower endpoint is **−4.9031**, within 0.1 bps of −5. The classification turns on a difference smaller than the reporting precision. Had the interval been one-tenth of a basis point wider, the conclusion would have been *inconclusive* instead. The finding is therefore an informative null that only just qualifies as informative, and it should not be quoted without that qualification.

**Nothing rejects in the secondary family.** All 14 remaining (scorer, horizon) pairs, corrected together: smallest raw p = 0.249, smallest BH q = **0.946**, smallest BY q = **1.000**. Of the 15 return tests, 13 land on the informative null and 2 are inconclusive (LM h = 3, VADER h = 1), their intervals extending just past ±5.

**Table 4.** Effect sizes in bps per 1σ with 95% **pointwise** intervals, against the prespecified 5 bps SESOI. The yardstick is a measure of *smallness*, not a profitability threshold — no claim about strategy returns is made in either direction, since realised value depends on signal use, timing, turnover, holding period and capacity, none of which this design measures (P19).

**Timing diagnostic.** Circular shift over the 2,453 retained rows, all 2,452 shifts. FinBERT's observed coefficient sits at the **15.5th percentile** of its own shift distribution (VADER 15.9th, LM 68.6th). The retained rows have **0 interior gaps**, so a shift of *k* positions is a shift of *k* sessions here. Reported as a percentile rank and **not** as a p-value: shifting tone destroys its relationship with the controls, so this is not the null distribution of the conditional coefficient. Ties at the 0.1 bps reporting precision: 35–51 per scorer.

**The bridge, tested (§7a).** `delta = β_FinBERT − β_LM`, both standardized, both on the identical 2,453 observations, with the cross-equation HAC covariance in the standard error:

> **`delta = −2.43 bps per 1 SD, 95% pointwise [−6.88, +2.03], p = 0.285.`**

Daily correlation of the two standardized series: **0.298**; tone VIFs 1.010 and 1.003. These are diagnostics, not a decision rule: a high correlation would widen `delta`'s interval, and the widened interval is already the quantitative statement of what the data can distinguish. The interval is wide enough to neither establish nor exclude a difference between the two scorers.

**§7(b), a different estimand.** Both standardized scores in one regression: FinBERT −0.000215 (p = 0.235), LM +0.000133 (p = 0.506). This asks whether one adds information given the other, not which marginal association is larger, and is labelled as such.

### 6. Volume and range variance (RQ4, exploratory)

This family is exploratory and cannot supply a result the primary specification did not. **The primary test for each outcome is a HAC Wald test of the joint null `b1 = b2 = b3 = 0`** across level, intensity (|S_t|) and dispersion (d_t), BH-corrected across the four (scorer, outcome) pairs.

| Scorer | Outcome | χ²(3) | p | BH q | n |
|---|---|---:|---:|---:|---:|
| FinBERT | range variance | 7.78 | 0.051 | 0.102 | 2,514 |
| FinBERT | detrended log volume | 1.47 | 0.689 | 0.689 | 2,453 |
| LM | range variance | 3.64 | 0.303 | 0.403 | 2,514 |
| LM | detrended log volume | 9.66 | 0.022 | **0.087** | 2,453 |

**Nothing rejects at q = 0.05.**

Individual coefficients are read descriptively afterwards, with pointwise intervals, in `table5b_rq4_coefficients.csv`. Two carry intervals excluding zero — LM's dispersion term on next-day volume, and all three FinBERT terms on next-day range variance — and **neither is promoted to a finding**, because the joint test that governs them does not reject. That ordering is the entire point: it is what stops one coefficient becoming the headline once someone has looked at it (P21).

`d_t` is the within-day standard deviation of **model-assigned** tone, not investor disagreement: these are headlines about different events, scored by a model, not observations of belief. Its relation to volume is an extension to test, not a replication of Tetlock (2007), which links unusually high or low pessimism — not dispersion — to volume. Bounded scores also tie the mean and dispersion mechanically, `d_t² ≤ [n_t/(n_t−1)](1 − S_t²)`, so `b2` and `b3` cannot be read as independent dimensions. The `n_t ≥ 5` requirement costs the range-variance equations nothing here and the volume equations the 62-session warm-up.

### 7. Robustness

Specified but not yet run: median aggregation (now selectable rather than an inert flag), NW bandwidth sensitivity `L ∈ {0, 1, 10}` and the data-driven plug-in, first/second-half split, dropping `n_t < 5` from the `S_t` specifications, and a single-name spot check. The retained-position HAC convention is already reported alongside the session-indexed primary and agrees to 16 significant figures on every term, because the analysis sample has no interior gaps.

### 8. Limitations

- **Text exposure is unknown.** New human annotation would establish that the **labels** are independent of every model. It does **not** establish that the headline *text* was unseen: these are 2010–2019 headlines from the open web and no scorer's pretraining corpus is auditable at that granularity. This applies symmetrically to all three scorers.
- **No Act 1 result.** Without labels, nothing in this report speaks to classification quality, and the attenuation bridge cannot be examined empirically at all.
- **Coverage drift.** Headline volume is not stationary over the sample; `n_t` enters no specification, but it shapes the precision of `S_t` and `d_t`.
- **One market, one instrument.** SPY only. No cross-section, so nothing here speaks to single-name predictability.
- **Association, not causation.** Nothing identifies a causal channel.
- **No intraday confirmation.** Daily aggregation cannot distinguish a signal that decays within hours from one that never existed.
- **Headlines, not articles.** The measured object is a headline, written to be read rather than scored.
- **A boundary classification.** The primary conclusion depends on an interval endpoint 0.1 bps from the SESOI. A slightly different sample, bandwidth or control transformation could move it to *inconclusive*.

### 9. Why there is no trading backtest

A backtest converts an inference question into a specification search over costs, sizing and rebalancing rules, every one of them p-hackable. What Table 4 delivers instead is a **yardstick for smallness** — basis points per 1σ against the prespecified 5 bps SESOI — which is not a profitability threshold and is not read as one in either direction (P19).

> **Corrected 2026-09-10 (R09).** This section previously called the 5 bps figure a "transaction-cost benchmark" against which economic significance was "delivered", contradicting §5 two pages earlier and the inference protocol's §11.

### 10. What this study found

Stated plainly, because a null is a result and deserves to be written as one:

1. **No detectable association** between daily aggregate FinBERT tone and the next session's SPY return, conditional on a frozen control set — and the interval is narrow enough to exclude effects beyond ±5 bps per 1σ, *just*.
2. **Nothing rejects in any family** — 14 secondary return tests, 4 exploratory Wald tests. The picture is uniform rather than mixed.
3. **The two scorers cannot be separated** on their market association: `delta`'s interval spans −6.88 to +2.03 bps. The mathematical appendix gives a reason internal to the design — aggregation over hundreds of headlines compresses per-headline differences — so this is weak evidence about relative classification quality, not strong evidence of similarity.
4. **Act 1 is unanswered** and is the study's main outstanding piece of work.

---

### References

Araci (2019), *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*.
Benjamini & Hochberg (1995), *Controlling the False Discovery Rate*.
Benjamini & Yekutieli (2001), *The Control of the False Discovery Rate under Dependency*.
Loughran & McDonald (2011), *When Is a Liability Not a Liability?*, **Journal of Finance**.
Malo et al. (2014), *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts*.
Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*.
Parkinson (1980), *The Extreme Value Method for Estimating the Variance of the Rate of Return*.
Tetlock (2007), *Giving Content to Investor Sentiment*, **Journal of Finance**.
