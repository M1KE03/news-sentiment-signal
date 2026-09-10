# Reading the News with a Machine
## How three sentiment measurements differ, and what they add about the next session

> **Title corrected 2026-09-10 (R09, P06).** The subtitle previously asked whether FinBERT's
> classification skill "survives as a market signal", which presupposes the skill Act 1 exists to
> measure and frames Act 2 as a test of survival rather than of association. The bridge between the
> two acts is a conditional hypothesis, not the report's spine.

> **Skeleton, and nothing more.** Section order is fixed so that writing is not deciding. Every
> `[…]` is filled from `report/tables/` and `figures/`. Target: two pages.
>
> **No result exists yet.** No text has been scored, no labels collected, no panel built and no model
> fitted. Nothing in this file is a finding, and the sections below are argument structure awaiting
> evidence.

---

### 1. Question and the two-act logic

*(≈200 words.)* How do financial sentiment measurements differ in classification quality, and what
additional information do they provide about subsequent market outcomes? The two questions are kept
apart: classification quality is measured against independent human labels, and market association
is estimated separately. Neither answer is assumed by the other.

**The attenuation bridge, as a conditional hypothesis.** Sentiment scores measure a latent quantity —
the information content of the news — with error, and classical errors-in-variables shrinks a
regression coefficient toward zero in proportion to that error. That motivates a hypothesis: on the
*identical* specification, a better classifier *may* show a larger, better-determined coefficient.
The assumptions it requires are stated with it. It is not a mechanism the design establishes, and a
null or inconclusive Act 2 does not falsify Act 1.

> **Corrected 2026-09-09 (P05/P06, audit A14).** This section previously opened by asserting that "a
> domain transformer reads financial sentences better than a word counter" as established fact, and
> described the bridge as a "falsifiable prediction" that made Act 1 "predictive of Act 2". The first
> is what Act 1 exists to measure and is not yet measured — and on the project's own six spot-check
> headlines FinBERT agrees with the word counter on both hard cases. The second overstated a
> conditional argument as a mechanism.

**Prespecified hypotheses, with no expected direction attached.** Act 1 tests whether the three
scorers differ in macro-F1, two-sided. Act 2 tests `H0: beta = 0` against a two-sided alternative,
at the 5% level, on one primary specification. RQ4 tests a joint null across tone, intensity and
dispersion. Each has a stated estimand and a stated uncertainty procedure for that same estimand,
and each may return an explicit **inconclusive**.

> **Removed 2026-09-09 (P23, audit A14).** This paragraph previously declared the expected outcomes
> in advance — "FinBERT wins Act 1 clearly; the same-day association is positive; next-day
> prediction is weak or null after correction; the volume/volatility act carries the most plausible
> positive finding". Prespecifying a *hypothesis* is the discipline; prespecifying the *answer* is
> the thing the discipline exists to prevent, and a reader cannot tell a confirmed prediction from a
> result written to match one.

### 2. Protocol

*(Written before any result. Changing a decision after seeing a result is the look-ahead this design
exists to prevent; any deviation is logged with its date in `future-work.md`.)*

- **News source (D1):** the **Benzinga sub-corpus of FNSPID's `All_external.csv`**, selected on
  relevance and coverage. The file concatenates five-plus sub-corpora with different languages and
  timestamp behaviour, including the Russian-language `lenta.ru`, so the source-domain filter is
  mandatory rather than hygiene. The audit's decisive finding: the only intraday-stamped block is
  Reuters, a global newswire with no ticker tags weighted to European hours — the wrong content.
- **Window (D4):** 2010-01-01 to 2019-12-31, **2,516 trading sessions**, frozen 2026-09-09 on the
  corpus census.
- **Timestamp rule:** the corpus is **date-only**, so the intraday close rule is descoped. A
  headline dated *d* belongs to the **first session strictly after *d***, so session *t* receives
  dates in [prev_session, *t*) and all of day *d* precedes close(*t*). The date is read in the
  source's own zone and never converted.
- **Scorers (D7):** LM (pos−neg)/(pos+neg); VADER compound; FinBERT P(pos) − P(neg). All in [−1, 1].
- **Aggregation (D8/D9):** equal-weighted daily mean S_t; count n_t; within-day dispersion d_t
  defined only when n_t ≥ 5. Zero-news days dropped and counted: […].
- **Inference (D10–D12):** Newey–West `L = 5`, prespecified as one trading week and counted in
  **exchange sessions** rather than retained rows, with the row-counting convention reported
  alongside ([spacing decision](../docs/hac-spacing-decision.md)). One **primary** test — FinBERT,
  h = 1, unadjusted — and a closed secondary family of the remaining 14 (scorer, horizon) pairs
  under BH at q = 0.05, with Benjamini–Yekutieli alongside; where they disagree the claim is made at
  the BY level. The timing diagnostic is an exhaustive **circular shift** of standardized tone over
  the retained rows, reported as a percentile rank.
- **Dedup:** exact `text_norm` matches within 3 days, plus token-set overlap ≥ 0.9. Dedup rate: […].

### 3. Data

*(≈250 words.)* The audit, what it found, and what it ruled out. Coverage caveats stated plainly:
headline volume drifts over the sample (Figure: coverage), which is why Stage 6 splits the sample in
half. Say which candidate dataset lost and why — a rejected dataset with a stated reason is worth
more than a chosen one without.

### 4. Act 1 — classification quality

**Table 1.** Accuracy, macro-F1, per-class precision/recall/F1 and support per scorer, on
**independently annotated headlines from this study's own collection** — not PhraseBank, which
`ProsusAI/finbert`'s model card names as fine-tuning data and which appears only as a supplementary
exhibit with a contamination statement. Class balance is reported alongside.
**Figure 1.** Confusion matrices, three scorers side by side.

The primary contrast is `macroF1(FinBERT) − macroF1(LM)` with a paired bootstrap resampling
**article groups**, B = 10,000. The accuracy comparison uses the same group bootstrap; exact
McNemar is supplementary, printed with the group-size distribution and the explicit note that its
p-value is valid only under item independence, which this design does not assert.

Differences between scorers are reported as **differences**, not decomposed into "domain vocabulary"
and "context" contributions: the three scorers differ in more than one way at once, so that
attribution is not identified (P05).

### 5. Act 2 — association, then prediction

**Table 2 — not produced.** Same-day association (RQ2) is **structurally suppressed**: no
sub-corpus is both intraday-stamped and relevant, so `inference.contemporaneous` raises and the
runner writes no Table 2. Its earlier description as "the pipeline's sanity check — no same-day
association means something upstream is broken" is withdrawn (P23): a weak contemporaneous
association can equally reflect aggregation, the chosen universe or measurement noise, and treating
its absence as a defect licenses adjusting the pipeline until the expected result appears.

**Table 3 / Figure 2 (headline).** One **primary** test — FinBERT, h = 1, no correction — reported
separately from the 14-test secondary family of remaining (scorer, horizon) pairs under BH at
q = 0.05, with Benjamini–Yekutieli alongside as a sensitivity valid under arbitrary dependence. The
timing diagnostic is a **circular shift** reported as a percentile rank, never as a p-value.

**Table 4.** Effect sizes in bps per 1σ of S_t with 95% **pointwise** NW intervals, against the
prespecified 5 bps smallest effect of interest. Two facts are always reported together: whether the
interval excludes zero, and whether it lies inside ±5 bps. The yardstick is a measure of *smallness*,
not a profitability threshold — no claim about strategy returns is made in either direction, since
realised value depends on signal use, timing, turnover, holding period and capacity, none of which
this design measures (P19).

**The bridge, tested (§7a).** `delta = beta_FinBERT − beta_LM`, both standardized, both fitted on
the **identical** observation set, with the cross-equation HAC covariance in `delta`'s standard
error: […]. Two separate t-statistics are not a test of the difference (P07), and no claim that one
scorer beats the other is made without this interval.

Daily correlation of the two standardized series, and the tone VIFs: […]. These are **diagnostics,
not a decision rule.** An earlier version of this paragraph declared the comparison "bounded and
inconclusive" whenever the correlation exceeded 0.9. That threshold was removed (R08b): a high
correlation widens `delta`'s interval, and the widened interval is already the honest, quantitative
statement of what the data can distinguish.

### 6. Volume and volatility

**Table 5 / Figure 3.** Next-day range variance (`rv_parkinson` — a high–low **range** estimator of
variance, not total daily volatility) and next-day **detrended log volume** (log share volume; no
denominator makes it turnover — P22) on level, intensity (|S_t|) and dispersion (d_t). **The primary test for each is a HAC Wald test of the joint null
`b1 = b2 = b3 = 0`**; individual coefficients, `d_t`'s included, are reported descriptively
afterwards with pointwise intervals. That ordering is what stops one coefficient being promoted to
the headline once someone has looked at it (P21), and it replaces the instruction to "lead with the
d_t coefficient" as the most plausible positive result.

`d_t` is the within-day standard deviation of **model-assigned** tone, not investor disagreement:
these are headlines about different events, scored by a model, not observations of belief. Its
relation to volume is an extension to test, not a replication of Tetlock (2007), which links
unusually high or low pessimism — not dispersion — to volume. Bounded scores also tie the mean and
dispersion mechanically, so `b2` and `b3` cannot be read as independent dimensions. Note the sample
reduction from the n_t ≥ 5 requirement.

### 7. Robustness

Five verdicts, one sentence each: median aggregation; NW maxlags = 10; first/second half split;
dropping n_t < 5 from the S_t specifications; the single-name spot check.

### 8. Limitations

- **Text exposure is unknown.** New human annotation establishes that the **labels** are
  independent of every model — they were produced by people, after training, and cannot have been in
  any training set. It does **not** establish that the headline *text* was unseen: these are
  2010–2019 headlines from the open web and no scorer's pretraining corpus is auditable at that
  granularity. This applies symmetrically to all three scorers. Where PhraseBank appears as a
  supplementary exhibit, its FinBERT number is an optimistically biased estimate of unknown
  tightness — contamination is documented on the model card — and not a bound on generalization.
- **Coverage drift.** Headline volume is not stationary over the sample; n_t enters no
  specification, but it shapes the precision of S_t and d_t.
- **One market, one instrument.** SPY only. No cross-section, so nothing here speaks to
  single-name predictability.
- **Association, not causation.** Nothing identifies a causal channel; the contemporaneous result in
  particular is as consistent with prices moving the news as the reverse.
- **No intraday confirmation.** Daily aggregation cannot distinguish a signal that decays within
  hours from one that never existed.
- **Headlines, not articles.** The measured object is a headline, which is written to be read, not
  to be scored.

### 9. Why there is no trading backtest

Two sentences. A backtest converts an inference question into a specification search over costs,
sizing and rebalancing rules, every one of them p-hackable. What Table 4 delivers instead is a
**yardstick for smallness** — basis points per 1σ against the prespecified 5 bps SESOI — which is
not a profitability threshold and is not read as one in either direction (P19).

> **Corrected 2026-09-10 (R09).** This section previously called the 5 bps figure a
> "transaction-cost benchmark" against which economic significance was "delivered", contradicting
> §5's own statement two pages earlier and the inference protocol's §11, which forbids claiming or
> denying strategy profitability from a coefficient and a cost figure.

---

### References

Araci (2019), *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*.
Benjamini & Hochberg (1995), *Controlling the False Discovery Rate*.
Loughran & McDonald (2011), *When Is a Liability Not a Liability?*, **Journal of Finance**.
Malo et al. (2014), *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts*.
Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation
Consistent Covariance Matrix*.
Tetlock (2007), *Giving Content to Investor Sentiment*, **Journal of Finance**.
