# Reading the News with a Machine
## Does FinBERT's classification skill survive as a market signal?

> **Skeleton.** Section order and the argument are fixed here at Stage 0 so that Stage 7 is writing,
> not deciding. Every `[…]` is filled from `report/tables/` and `figures/`. Target: two pages.
> Numbers are never written before the protocol section below is complete.

---

### 1. Question and the two-act logic

*(≈200 words.)* A domain transformer reads financial sentences better than a word counter. Whether
that better *measurement* becomes a better *signal* is a separate question carrying a heavier
evidentiary burden. This report answers both, and states the mechanism connecting them up front.

**The attenuation bridge.** Sentiment scores measure a latent quantity — the information content of
the news — with error. Classical errors-in-variables shrinks a regression coefficient toward zero in
proportion to that measurement error. So Act 1's answer makes a falsifiable prediction about Act 2:
on the *identical* specification, the better classifier should show a larger, better-determined
coefficient. That is what makes Act 1 predictive of Act 2 rather than decorative.

**Expected outcomes, declared before any result was seen:** FinBERT wins Act 1 clearly; the
same-day association is positive; next-day prediction is weak or null after correction; the
volume/volatility act carries the most plausible positive finding.

### 2. Protocol

*(Written before any result. Changing a decision after seeing a result is the look-ahead this design
exists to prevent; any deviation is logged with its date in `future-work.md`.)*

- **News source (D1):** […], chosen at Stage 0 on the timestamp audit — usable intraday timestamps
  with a documented timezone first, then coverage quality, then sample length.
- **Window (D4):** […] to […], […] trading days.
- **Timestamp rule (D6):** a headline stamped *s* (America/New_York) belongs to trading day *t* iff
  *s* ∈ (close(*t*−1), close(*t*)], close = 16:00 ET. Weekend and holiday news rolls forward.
- **Scorers (D7):** LM (pos−neg)/(pos+neg); VADER compound; FinBERT P(pos) − P(neg). All in [−1, 1].
- **Aggregation (D8/D9):** equal-weighted daily mean S_t; count n_t; within-day dispersion d_t
  defined only when n_t ≥ 5. Zero-news days dropped and counted: […].
- **Inference (D10–D12):** Newey–West maxlags = 5; horizons t+1…t+5 declared in advance with
  Benjamini–Hochberg at q = 0.05 within each scorer; circular block permutation, block 21, 1,000
  draws, seed 20260830.
- **Dedup:** exact `text_norm` matches within 3 days, plus token-set overlap ≥ 0.9. Dedup rate: […].

### 3. Data

*(≈250 words.)* The audit, what it found, and what it ruled out. Coverage caveats stated plainly:
headline volume drifts over the sample (Figure: coverage), which is why Stage 6 splits the sample in
half. Say which candidate dataset lost and why — a rejected dataset with a stated reason is worth
more than a chosen one without.

### 4. Act 1 — the classification ladder

**Table 1.** Accuracy and macro-F1 per scorer, per PhraseBank agreement subset; McNemar p and the
disagreement cells (b, c).
**Figure 1.** Confusion matrices, three scorers side by side.

One sentence to land: *FinBERT exceeds LM by […] macro-F1 points (McNemar exact p = […]); the
VADER→LM step contributes […] of the total gain* — that is, how much is domain vocabulary and how
much is context.

Does the FinBERT–LM gap widen as label agreement rises? […]

### 5. Act 2 — association, then prediction

**Table 2.** Contemporaneous (§6.1), all scorers. Reported as association, not causation. This is
also the pipeline's sanity check: no same-day association means something upstream is broken.

**Table 3 / Figure 2 (headline).** The lag family: β_h, NW se, t, raw p, BH q for h = 1…5 per
scorer, with the placebo p annotated on h = 1.

**Table 4.** Effect sizes in bps per 1σ of S_t with 95% NW intervals, against the ~5 bps one-way
cost bar. If the finding is null, it is reported *with content*: effects larger than […] bps per 1σ
are ruled out at 95% — below transaction costs.

**The bridge, tested (§6.3).** Identical specification, FinBERT vs LM: […]. Daily correlation of the
two S_t series: […]. *If that correlation exceeds 0.9 the comparison has little room to separate the
scorers, and is reported as bounded and inconclusive — not redesigned until it separates them.*

### 6. Volume and volatility

**Table 5 / Figure 3.** Next-day Parkinson volatility and detrended turnover on level, intensity
(|S_t|) and dispersion (d_t). Lead with the d_t coefficient: disagreement across the day's headlines
predicting next-day volume is the most plausible positive result here, and a documented one
(Tetlock 2007). Note the sample reduction from the n_t ≥ 5 requirement.

### 7. Robustness

Five verdicts, one sentence each: median aggregation; NW maxlags = 10; first/second half split;
dropping n_t < 5 from the S_t specifications; the single-name spot check.

### 8. Limitations

- **Shared provenance.** PhraseBank sentences are annotated from an investor's perspective and
  FinBERT was fine-tuned on related financial text; a shared-provenance advantage in Act 1 cannot be
  ruled out.
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
sizing and rebalancing rules, every one of them p-hackable. Economic significance is delivered
instead by Table 4 — basis points per 1σ against a transaction-cost benchmark — with none of that
surface area.

---

### References

Araci (2019), *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*.
Benjamini & Hochberg (1995), *Controlling the False Discovery Rate*.
Loughran & McDonald (2011), *When Is a Liability Not a Liability?*, **Journal of Finance**.
Malo et al. (2014), *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts*.
Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation
Consistent Covariance Matrix*.
Tetlock (2007), *Giving Content to Investor Sentiment*, **Journal of Finance**.
