# Reading the News with a Machine
## How three sentiment measurements differ, and what they add about the next session

Both acts are complete. Every number here is traceable to a file in `report/tables/`, and `report/tables/run_manifest.json` records the inputs, settings and code state that produced the Act 2 results.

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

**Table 1.** Macro-F1 and accuracy on **600 independently labelled headlines from this study's own collection** — not PhraseBank. 596 usable; 4 excluded as `unusable` and reported as a count. Class balance: 292 neutral (49%), 195 positive (33%), 109 negative (18%).

| Scorer | macro-F1 | accuracy | negative F1 | neutral F1 | **positive F1** |
|---|---:|---:|---:|---:|---:|
| **FinBERT** | **0.594** | 0.609 | 0.573 | 0.664 | **0.546** |
| Loughran–McDonald | 0.493 | 0.560 | 0.557 | 0.658 | **0.263** |
| VADER | 0.424 | 0.472 | 0.323 | 0.558 | 0.390 |

**The primary contrast.**

> **`macroF1(FinBERT) − macroF1(LM) = +0.102, 95% pointwise [+0.050, +0.153]`**
> Paired bootstrap resampling **article groups**, B = 10,000, n = 596 over 595 groups.

The interval excludes zero: FinBERT classifies these headlines better than the domain lexicon, by roughly 5 to 15 macro-F1 points.

**Reported on both sets, as §6a requires** — the full evaluation set and the confident subset the annotator did not flag `hard`:

| Set | n | difference | 95% pointwise |
|---|---:|---:|---|
| Full | 596 | +0.102 | [+0.050, +0.153] |
| Confident (`hard = 0`) | 541 | +0.104 | [+0.052, +0.156] |

The two agree to 0.002. **The gap does not come from the headlines a careful human could not resolve** — it holds on the ones whose direction was clear. That is the question the two-set split was prespecified to answer.

**Figure 1.** Confusion matrices, three scorers side by side — and they carry the finding. In LM's true-positive row, **159 of 195 genuinely positive headlines are called neutral**: 83%. FinBERT's same row is 100 positive, 72 neutral, 23 negative.

**Where the gap actually comes from, which is narrower than the headline number suggests.** FinBERT and LM are close on negative (0.573 vs 0.557) and neutral (0.664 vs 0.658). **The entire macro-F1 difference is the positive class.** The reason is visible in the calibration diagnostic rather than inferred: LM takes only **three distinct values** on headline-length text — `−1`, `0`, `+1` — because its score is `(pos − neg)/(pos + neg)` over dictionary matches, and a short headline usually contains either no sentiment word (78.3% of the calibration set, scoring exactly 0) or words of one sign only. The Loughran–McDonald lists carry **2,345 negative terms against 347 positive**, an asymmetry built for 10-K risk language. Applied to headlines, that asymmetry collapses the positive class.

So the honest statement is not "the transformer reads financial language better". It is that **a document-level risk dictionary, applied to headlines, has almost no positive vocabulary to fire on**, and macro-F1 — which weights all three classes equally — makes that visible where accuracy does not.

**Secondary contrasts**, descriptive and in no correction family (M3): FinBERT − VADER **+0.171** [+0.117, +0.226]; LM − VADER **+0.069** [+0.015, +0.125]. The ordering FinBERT > LM > VADER holds on every set and every comparison.

**Accuracy is the weaker claim and is reported as such.** FinBERT − LM accuracy is **+0.049, 95% [0.000, 0.097]** — the lower bound sits exactly at zero. Accuracy is dominated by the 49% neutral majority, which both scorers handle similarly; macro-F1 is what surfaces the positive-class collapse. Where the two metrics disagree in strength, the report says so rather than quoting whichever is larger.

**Exact McNemar is supplementary**, printed with its validity condition: statistic 95.0, **p = 0.058**, group sizes 594 singletons and 1 pair. It does not agree with the group bootstrap. Its independence assumption is contradicted by this design's own article-group structure (M4), so the bootstrap is primary and McNemar is shown for completeness, not as a tie-breaker.

**Thresholds were fitted on the 200 calibration items only** and frozen in `config.VALIDATION_THRESHOLDS` before the evaluation set was scored: LM `[−1.00, 0.00]`, VADER `[−0.125, +0.225]`. FinBERT is never fitted; its class is the argmax over the three probabilities, never a threshold on the tone scalar, so the neutral probability stays decisive (B12).

**One property of that fit belongs in the record.** LM's band is one of **1,600 grid pairs (48.2%) tied at the best calibration macro-F1** — but all 1,600 produce an *identical* classification, so the arbitrariness of the band never reaches the predictions. LM on headlines is effectively `sign(score)`. VADER's band is genuinely determined (3 tied pairs, same classification). Recorded in `report/tables/calibration_record.json`.

**Limitations specific to Act 1, stated here and not only in §8.**

- **The scorers are not compared at equal resolution.** On 198 calibration items FinBERT produces 198 distinct values, VADER 54, LM **3**. Part of any macro-F1 gap is that difference in granularity rather than a difference in reading.
- **The rubric's framing is closer to FinBERT's training objective than to the lexicons'** (§6a). FinBERT was fine-tuned on PhraseBank, whose annotation instruction is the same investor-perspective question this rubric asks — *would this move the price?* — while LM and VADER score textual valence. This is separate from text contamination, which §2 treats on its own, and it is **not fixable by reframing**: a textual-valence rubric would favour the lexicons symmetrically. Some unknown part of +0.102 is this.
- **One annotator, not independent of the analyst.** No second annotator, so **no Cohen's kappa and no measured label error rate**. Rubric v3 was adopted mid-project after a 60-item pilot under v1 flagged 34 of 60 `hard`; the v3 rate is 9.2% on the evaluation set. Labelling followed the full scoring pass, which §5 permits provided the cache is not consulted; the annotator confirms it was not. All sessions, deviations and the reconstructed timings are in `data/annotation/provenance_v2.json`.
- **Ambiguity is real and measured rather than suppressed.** 55 of 600 items are flagged `hard`. For comparison, Financial PhraseBank achieves unanimous agreement on only 2,264 of 4,846 sentences (46.7%) using 5–8 annotators each.

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

**The robustness suite (§7) sharpens that qualification rather than relieving it.** The conclusion is invariant across all five prespecified bandwidths, but **5 of 7 sensitivities are boundary cases**, and standard errors here are *smaller* at longer bandwidths — so the prespecified `L = 5` happens to sit in the range producing the narrow interval that places the result inside ±5. Separately, under median rather than mean daily aggregation the point estimate collapses to **−0.17 bps [−3.18, +2.84]**. So of the two dimensions M2 reports:

- **"No association detected" is robust** — every bandwidth, both aggregation rules, both subperiods, both floors. Nothing rejects anywhere in Act 2.
- **"Precise enough to exclude ±5 bps" is fragile** — it survives the prespecified rule, but on a margin of 0.1 bps and with most sensitivities sitting on the boundary.

The report states both, and does not present the second with the confidence of the first.

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

All five prespecified exhibits were run. They are **sensitivity analyses, not additional tests**: they enter no correction family and cannot supply a result the primary specification did not (§6). Divergence is reported as fragility, never resolved by adopting whichever choice is most convenient.

**Two are informative, one is bounded, and two turned out to be inapplicable to this corpus.** That is a weaker suite than "five robustness checks" suggests, and the distinction is drawn here rather than left for a reader to discover.

**(a) Bandwidth — informative. The conclusion holds; the margin is thin.**

| `L` | Interval (bps) | Conclusion | Boundary case |
|---|---|---|---|
| 0 (HC only) | [−5.02, +1.54] | informative null | yes |
| 1 | [−5.04, +1.56] | informative null | yes |
| **5 (primary)** | [−4.90, +1.41] | informative null | yes |
| 10 | [−4.71, +1.22] | informative null | no |
| plug-in (8) | [−4.77, +1.28] | informative null | no |

The point estimate is identical throughout — bandwidth affects only the standard error — and the conclusion is invariant under the M2 rule. But **5 of 7 sensitivities are boundary cases**, and the direction deserves stating: standard errors here are *smaller* at longer bandwidths (`κ = 0.964 < 1`), so the prespecified `L = 5` sits in the range producing the narrower interval, and the narrow interval is what places it inside ±5. The residual ACF's largest value is **−0.082 at lag 5** — the prespecified bandwidth — and that negative autocorrelation is why HAC shrinks the interval. `L = 5` was fixed before any estimation, so this is coincidence rather than selection; the chain is exposed because a reader cannot check it otherwise.

**(b) Aggregation — informative, and the point estimate is not robust.** Under median rather than mean daily aggregation: **−0.17 bps, 95% [−3.18, +2.84], p = 0.91**. The conclusion is unchanged and in fact cleaner — that interval is not a boundary case — but the point estimate collapses by roughly 90%. Anyone quoting "−1.74 bps" should know it becomes approximately zero under a defensible alternative. Mean was prespecified at D8; median is a sensitivity and is not promoted.

**(c) Subperiod split — bounded.** First half −0.92 [−6.66, +4.83]; second half −2.81 [−6.81, +1.20]. Reading these for *overlap* would be worthless, since halving `n` widens each by about √2 and overlap is nearly guaranteed — an artefact of low power that would be mistaken for stability. The exhibit's actual question is whether the halves differ, which has its own interval:

> **difference (first − second) = +1.89 bps, 95% [−5.11, +8.89], p = 0.60**

Disjoint samples share no observations, so `var(diff) = var₁ + var₂`. The honest reading: the data **cannot distinguish a stable coefficient from one that moved by 5 bps in either direction**. That is a statement about power, not about stability.

**(d) Dispersion floor — inapplicable.** Requiring `n_t ≥ 5` in the `S_t` specification changes nothing: exactly **one** session falls below the floor, and it is the zero-news session already excluded. The concern the check addresses — that thin-news sessions with a noisy `S_t` distort the estimate — does not arise on a corpus with a median of 337 headlines per session. Reporting this as "robust to the dispersion floor" would imply a stress test that was never applied.

**(e) Single-name spot check — untestable on this corpus.** Its purpose is to remove the mismatch §6b concedes: Act 2 aggregates *company* news to a *market* outcome. Run on the three most-covered tickers:

| Ticker | Headlines | Per session | n | bps per 1 SD | 95% interval | Empty sessions |
|---|---:|---:|---:|---:|---|---:|
| MRK | 3,068 | 1.22 | 1,382 | −3.46 | [−9.37, +2.46] | 1,105 |
| MU | 2,934 | 1.17 | 1,101 | +6.06 | [−12.00, +24.12] | 1,414 |
| MS | 2,905 | 1.15 | 1,235 | −5.31 | [−17.28, +6.67] | 1,237 |

At roughly 1.2 headlines per session against the market aggregate's 345, `S_t` for a single name is far noisier and **44–56% of sessions carry no headlines at all**. All three intervals include zero, run up to 36 bps wide, and disagree in sign. This weakness was recorded in the decision log *before* any return was fetched, so it is a confirmed prediction rather than an excuse. The useful conclusion is about design, not about these firms: a properly powered single-name study needs a denser per-ticker feed, and pooling across tickers would change the estimand to cross-sectional, which this design descopes.

**Also reported.** The retained-position HAC convention agrees with the session-indexed primary to 16 significant figures on every term, because the analysis sample has no interior gaps. The tone series is highly persistent (ACF 0.52 at lag 1, 0.45 at lag 5), which is why HAC is used at all. The residual ACF exceeds its 95% band at 3 of 20 lags against roughly 1 expected by chance.

**Not run, and deliberately not added.** A coverage-quartile split would test the concern (d) was aiming at and *is* answerable here, but it is not prespecified, and adding an exhibit after seeing results is the pattern this design exists to prevent. It is recorded as available for a future increment.

### 8. Limitations

- **Text exposure is unknown.** New human annotation would establish that the **labels** are independent of every model. It does **not** establish that the headline *text* was unseen: these are 2010–2019 headlines from the open web and no scorer's pretraining corpus is auditable at that granularity. This applies symmetrically to all three scorers.
- **The attenuation bridge remains untested empirically.** Act 1 separates the scorers and Act 2 cannot, but §6 of the appendix shows the design compresses a twofold measurement-noise difference into ~3% of coefficient. So the bridge is not refuted by the null in Act 2 — it is simply not testable at daily aggregation with this corpus.
- **Coverage drift.** Headline volume is not stationary over the sample; `n_t` enters no specification, but it shapes the precision of `S_t` and `d_t`.
- **One market, one instrument.** SPY only. No cross-section, so nothing here speaks to single-name predictability.
- **Association, not causation.** Nothing identifies a causal channel.
- **No intraday confirmation.** Daily aggregation cannot distinguish a signal that decays within hours from one that never existed.
- **Headlines, not articles.** The measured object is a headline, written to be read rather than scored.
- **A boundary classification.** The primary conclusion depends on an interval endpoint 0.1 bps from the SESOI. A slightly different sample, bandwidth or control transformation could move it to *inconclusive*.
- **The robustness suite is weaker than its length suggests.** Of five prespecified exhibits, two are informative (bandwidth, aggregation), one is bounded by power (the subperiod difference spans ±5 bps and cannot distinguish stability from a substantial shift), and **two are inapplicable to this corpus**: the dispersion floor excludes exactly one session, and the single-name check runs on names averaging 1.2 headlines per session with up to 56% of sessions empty. Absence of divergence in the last three is not evidence of stability.

### 9. Why there is no trading backtest

A backtest converts an inference question into a specification search over costs, sizing and rebalancing rules, every one of them p-hackable. What Table 4 delivers instead is a **yardstick for smallness** — basis points per 1σ against the prespecified 5 bps SESOI — which is not a profitability threshold and is not read as one in either direction (P19).

> **Corrected 2026-09-10 (R09).** This section previously called the 5 bps figure a "transaction-cost benchmark" against which economic significance was "delivered", contradicting §5 two pages earlier and the inference protocol's §11.

### 10. What this study found

Stated plainly, because a null is a result and deserves to be written as one:

1. **No detectable association** between daily aggregate FinBERT tone and the next session's SPY return, conditional on a frozen control set. This part is robust: it holds at every bandwidth, under both aggregation rules, in both halves of the sample.
2. **The claim that the study is precise enough to exclude effects beyond ±5 bps is fragile, and is reported as such.** It rests on a 0.1 bps margin; 5 of 7 sensitivities are boundary cases; and the point estimate falls to −0.17 bps under median aggregation. It survives the prespecified rule and is stated — but a reader should not treat it as established with the same confidence as point 1.
3. **Nothing rejects in any family** — 14 secondary return tests, 4 exploratory Wald tests. The picture is uniform rather than mixed.
4. **The two scorers cannot be separated** on their market association: `delta`'s interval spans −6.88 to +2.03 bps. The mathematical appendix gives a reason internal to the design — aggregation over hundreds of headlines compresses per-headline differences — so this is weak evidence about relative classification quality, not strong evidence of similarity.
5. **FinBERT classifies these headlines better than the domain lexicon**, by +0.102 macro-F1 [+0.050, +0.153] — the one interval in this study that excludes zero. It holds on the confident subset too (+0.104). But the entire gap is the **positive class**: Loughran–McDonald calls 83% of genuinely positive headlines neutral, because a 10-K risk dictionary carries 2,345 negative terms against 347 positive and takes only three distinct values on headline-length text. That is a statement about applying a document-level dictionary to headlines, not about transformers reading language better.

6. **The two acts do not connect, and the appendix predicted that.** Act 1 separates the scorers cleanly; Act 2 cannot separate them at all (`delta = −2.43 bps [−6.88, +2.03]`). That is not a contradiction — [`mathematical-appendix.md`](../docs/mathematical-appendix.md) derives it in advance: standardization puts the attenuation exponent at ½, and averaging a median of 337 headlines per session compresses a **twofold** per-headline noise difference into a **1.034×** coefficient difference. A classification gap of this size was never going to survive daily aggregation, and the conditional hypothesis in §1 is therefore neither confirmed nor refuted — the design cannot test it at this aggregation.

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
