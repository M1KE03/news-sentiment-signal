# Project 2 — Implementation Plan
## Reading the News with a Machine: Does FinBERT's Classification Skill Survive as a Market Signal?

**Budget:** ~15–20 hours over ~1 week. Stage hours sum to 16.5–20.5h.
**Status of this document:** it is the build specification. Every decision needed to write code is fixed here or explicitly marked ⏳ (resolved once, at Stage 0, then frozen). If something is ambiguous while building, the answer is in §3, §4 or §6 — not in your judgement on the day.

---

## 1. Objective and research questions

The project runs in two linked acts. The first has a definite, positive answer; the second is where the honest inference lives; a stated statistical mechanism connects them.

**Act 1 — Validation (measurement).**
> RQ1: On labeled financial sentences, how much better does FinBERT classify sentiment than a general-purpose lexicon (VADER) and a domain lexicon (Loughran–McDonald), and is the gap statistically significant?

Three scorers form a ladder — generic lexicon → domain lexicon → domain transformer — so the gain decomposes into "domain vocabulary matters" (VADER→LM) and "context matters beyond vocabulary" (LM→FinBERT).

**Act 2 — Signal (inference).**
> RQ2: Is daily aggregate headline sentiment associated with *same-day* SPY returns?
> RQ3: Does it *predict* next-day (t+1…t+5) returns once standard errors are HAC-corrected, the lag family is FDR-controlled, and the result is checked against a block-permutation placebo?
> RQ4: Does it predict next-day *volatility and volume* — the outcomes where media-sentiment effects are documented (Tetlock 2007)?

**The bridge between acts.** Sentiment scores are noisy measurements of a latent quantity (the information content of the news). Classical errors-in-variables attenuates a regression coefficient toward zero in proportion to measurement noise. So RQ1's answer makes a testable prediction about RQ2/RQ3: on the *identical* specification, the better classifier should produce a larger, better-determined coefficient. That prediction is tested explicitly in Stage 5.

**Expected outcomes, declared in advance** (so the write-up is not written to fit whatever appeared):
- RQ1: FinBERT wins, clearly. LM beats VADER on financial text.
- RQ2: clear positive contemporaneous association. This is what efficient markets predict and is a pipeline sanity check as much as a finding.
- RQ3: weak or null after correction. The deliverable is then a *ruled-out interval*, not a shrug.
- RQ4: the most plausible positive result in the project, particularly sentiment *dispersion* → volume.

**Out of scope, deliberately:** no model training or fine-tuning; no long documents (10-K, transcripts); no cross-sectional panel; no trading-strategy backtest. The report states the reason for the last one (a backtest turns an inference question into a specification search over costs, sizing and rebalancing; economic significance is delivered instead by the bps-per-1σ vs. transaction-cost comparison in §6.5).

---

## 2. System architecture

### 2.1 Pipeline

```
  raw news dump                         yfinance
  (FNSPID or Benzinga)                  (SPY OHLCV+volume, ^VIX)
        │                                     │
        ▼                                     ▼
 ┌──────────────────────┐            ┌──────────────────────┐
 │ src/data.py          │            │ src/data.py          │
 │  load → dedup →      │            │  pull → calendar     │
 │  tz-normalize to ET  │            │  hygiene → returns,  │
 │                      │            │  Parkinson, turnover │
 └──────────┬───────────┘            └──────────┬───────────┘
            │ headlines.parquet                 │ market.parquet
            ▼                                   │
 ┌──────────────────────┐                       │
 │ src/scoring.py       │                       │
 │  LM | VADER | FinBERT│                       │
 │  (cached, run once)  │                       │
 └──────────┬───────────┘                       │
            │ scores.parquet                    │
            ▼                                   │
 ┌──────────────────────────────────────────────┴───────────┐
 │ src/align.py                                              │
 │  close-to-close timestamp map → per-day aggregation →     │
 │  join to market data                                      │
 └──────────────────────────┬───────────────────────────────┘
                            │ daily_panel.parquet  ◄── THE single analysis table
                            │                          nothing downstream touches
                            │                          raw text or raw prices
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌────────────────────┐      ┌────────────────────────┐
   │ src/inference.py   │      │ notebooks/03_signal    │
   │  NW regressions,   │─────►│ notebooks/05_robustness│
   │  BH-FDR, placebo   │      │  tables + figures      │
   └────────────────────┘      └────────────────────────┘

  ── independent branch, shares only scoring.py ──
  PhraseBank ──► src/validate.py (threshold fit, macro-F1, McNemar) ──► notebooks/02_validation
```

Two properties this architecture buys, both worth a sentence in the README:
1. **One analysis table.** Every Act-2 number comes from `daily_panel.parquet`. If a result is wrong, it is wrong in the panel or in the regression — never in an ad-hoc join inside a notebook.
2. **Scoring runs exactly once.** FinBERT inference is the expensive step; scores are cached keyed by headline hash, so `run_all.py` reproduces every result in minutes without a GPU.

### 2.2 Repository layout

**Repository name: `news-sentiment-signal`.** It names the data (news), the measured quantity (sentiment) and the question (is it a signal), it is readable to someone scanning a GitHub profile, and it carries no coursework numbering — a public repo called `project-2-anything` reads as an assignment rather than a piece of work. Keep a `project-2-` prefix only if all three projects end up as folders inside one portfolio repo; as standalone repos they should be named for what they contain (Project 1 by the same rule would be `volatility-forecast-calibration`, not `project-1-volatility-uq`). The Python package inside stays `src/`, so nothing in §5 changes.

```
news-sentiment-signal/
├── README.md                     question, headline figure, 3 findings, repro command
├── requirements.txt              pinned: transformers, torch, vaderSentiment,
│                                 statsmodels, pandas, pyarrow, yfinance, matplotlib
├── run_all.py                    rebuilds panel → all tables → all figures
├── rescore.py                    the one-off FinBERT pass (not in run_all.py)
├── config.py                     every constant from §3, imported everywhere
├── data/
│   ├── raw/                      news dump + download script; SPY/VIX cache
│   ├── interim/                  headlines.parquet, scores.parquet, market.parquet
│   └── processed/                daily_panel.parquet
├── src/
│   ├── data.py
│   ├── scoring.py
│   ├── align.py
│   ├── validate.py
│   ├── inference.py
│   └── plots.py
├── tests/
│   ├── test_alignment.py         the look-ahead firewall (§7, Stage 3)
│   └── test_scoring.py           scorer range/determinism checks
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_validation.ipynb
│   ├── 03_signal.ipynb
│   ├── 04_volume_vol.ipynb
│   └── 05_robustness.ipynb
├── figures/
├── report/report.md
└── future-work.md                where scope-creep ideas go to die quietly
```

Rule: notebooks contain no analysis logic. They import from `src/`, call, and display. This is what makes `run_all.py` possible and what reads as engineering discipline.

---

## 3. Locked decisions

Write these into the report's protocol section before producing any results. Changing one after seeing a result is the look-ahead this design exists to prevent; if you must, log the change and the date in `future-work.md`.

| # | Decision | Value |
|---|---|---|
| D1 | News source | ⏳ Chosen at Stage 0 from **(A) FNSPID** (HuggingFace, Nasdaq-ecosystem news, intraday timestamps, ticker tags) or **(B) Kaggle "Daily Financial News for 6000+ Stocks"** (Benzinga headlines, timestamps, ticker tags). Criteria in priority order: usable intraday timestamps with known timezone → coverage/duplication quality → sample length. Verify availability by downloading; do not trust any description of these datasets, including this one |
| D2 | Fallback if neither passes the timestamp audit | Use the better date-only dataset, map news dated d to trading day **d+1**, and drop the contemporaneous act (RQ2) entirely. Decision deadline: end of day 1 |
| D3 | Market data | SPY daily OHLCV + volume via `yfinance`; `^VIX` close for context plots only |
| D4 | Sample window | ⏳ Longest span with stable news coverage; target ≥ 5 years / ≥ 1,250 trading days. Locked at Stage 0 |
| D5 | Unit of analysis | Market-level: all headlines in the window aggregated per trading day, tested against SPY. Single-name analysis appears only as a robustness spot check |
| D6 | Timestamp → trading day | Headline stamped s (converted to America/New_York) belongs to trading day t iff s ∈ (close(t−1), close(t)], close = 16:00 ET. News on non-trading days rolls forward into the next trading day's window |
| D7 | Scorers | **LM**: (pos−neg)/(pos+neg) on Loughran–McDonald word counts, 0 when no hits. **VADER**: `vaderSentiment` compound. **FinBERT**: `ProsusAI/finbert`, P(positive) − P(negative). All in [−1, 1]. Dictionary version and model revision hash pinned in `config.py` |
| D8 | Daily aggregation | S_t = equal-weighted mean of that day's headline scores; n_t = headline count; d_t = within-day standard deviation of scores, defined only when n_t ≥ 5 (else NaN) |
| D9 | Minimum coverage | Trading days with n_t = 0 are dropped from all regressions and the count reported. Days with 1 ≤ n_t < 5 are kept for S_t, excluded from d_t specs |
| D10 | Standard errors | Newey–West (HAC), maxlags = 5, for every regression. maxlags = 10 as sensitivity in Stage 6 |
| D11 | Lag family | Horizons t+1, t+2, t+3, t+4, t+5, per scorer. Declared now; Benjamini–Hochberg FDR at q = 0.05 applied within each scorer's family |
| D12 | Placebo | Circular block permutation of the S_t series, block length 21 trading days, 1,000 draws, seed 20260830. Permutation p-value for the t+1 t-statistic |
| D13 | PhraseBank protocol | All four agreement subsets loaded; headline metric = **macro-F1**; primary comparison on the ≥75%-agreement subset; LM/VADER neutral-band thresholds fit on a stratified 20% split and evaluated on the remaining 80%; FinBERT evaluated on the same 80% |
| D14 | Classifier comparison test | McNemar (exact binomial variant), paired predictions: FinBERT vs LM (primary), FinBERT vs VADER (secondary) |
| D15 | Effect-size unit | Every headline coefficient reported as **basis points of next-day return per 1σ move in S_t**, with 95% Newey–West CI, benchmarked against ~5 bps one-way transaction costs |
| D16 | Seeds & reproducibility | Global seed 20260830 in `config.py`; pinned `requirements.txt`; `python run_all.py` from a clean clone reproduces every table and figure |

---

## 4. Data contracts

Exact schemas. Anything not listed here does not belong in the file.

**`interim/headlines.parquet`** — one row per deduplicated headline
| column | dtype | note |
|---|---|---|
| `headline_id` | str | sha1 of normalized text + original timestamp |
| `text` | str | original headline, untouched |
| `text_norm` | str | lowercased, whitespace/punctuation-collapsed — dedup key only, never scored |
| `ts_utc` | datetime64[ns, UTC] | source timestamp, timezone made explicit at load |
| `ts_et` | datetime64[ns, America/New_York] | derived |
| `tickers` | list[str] | may be empty; used only in the robustness spot check |

**`interim/scores.parquet`** — one row per headline, one column per scorer
`headline_id`, `score_lm`, `score_vader`, `score_finbert` (float32, all in [−1, 1]).

**`interim/market.parquet`** — one row per trading day
`date` (date), `close_adj`, `ret` (log return), `parkinson` (variance, `(ln(H/L))² / (4 ln 2)`), `volume`, `log_turnover`, `vix_close`.

**`processed/daily_panel.parquet`** — one row per trading day in the locked window; **the only input to Act 2**
| column | note |
|---|---|
| `date` | trading day t |
| `n_headlines` | n_t |
| `s_lm`, `s_vader`, `s_finbert` | S_t per scorer (D8) |
| `d_lm`, `d_vader`, `d_finbert` | dispersion d_t, NaN when n_t < 5 |
| `ret`, `ret_lead1` … `ret_lead5` | contemporaneous and lead returns |
| `parkinson`, `parkinson_lead1` | volatility outcome |
| `log_turnover`, `log_turnover_detrended`, `log_turnover_detrended_lead1` | volume outcome; detrended = residual from a 63-day rolling mean, **trailing only** |
| `vix_close` | context/plots |

Every lead column is constructed by a single shift in `align.py` and nowhere else. This is deliberate: leads are the one place a look-ahead bug can hide in plain sight.

---

## 5. Module contracts

Write these signatures first, then fill them in.

**`src/data.py`**
```python
load_news(path, source: Literal["fnspid","benzinga"]) -> pd.DataFrame   # → headlines schema
dedup(df, near_dupe=True) -> tuple[pd.DataFrame, dict]                  # returns df + dedup stats
load_market(start, end) -> pd.DataFrame                                 # → market schema
```
`dedup` removes exact `text_norm` matches within a 3-day window; near-duplicate matching uses token-set overlap ≥ 0.9. It returns the counts so the report can quote a dedup rate.

**`src/scoring.py`**
```python
class Scorer(Protocol):
    name: str
    def score(self, texts: list[str]) -> np.ndarray: ...   # shape (n,), values in [-1, 1]

LMScorer(dict_path)      # word counts
VaderScorer()
FinbertScorer(batch_size=32, max_length=64)                # truncation is safe: headlines
score_all(headlines_df, cache_path) -> pd.DataFrame        # hash-keyed cache, skips scored rows
```

**`src/align.py`**
```python
map_to_trading_day(ts_et: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series   # D6, the firewall
aggregate_daily(scores_df, headlines_df, calendar) -> pd.DataFrame              # D8/D9
build_panel(daily_scores, market) -> pd.DataFrame                               # + leads → panel schema
```

**`src/validate.py`**
```python
fit_thresholds(scores, labels) -> tuple[float, float]        # neutral band, on the 20% split only
evaluate(scores, labels, thresholds) -> dict                 # accuracy, macro-F1, confusion
mcnemar_test(pred_a, pred_b, truth) -> tuple[float, float]   # statistic, exact p-value
```

**`src/inference.py`**
```python
nw_ols(y, X, maxlags=5) -> RegressionResults                 # statsmodels HAC
lag_family(panel, scorer, horizons=(1,2,3,4,5)) -> pd.DataFrame   # coef, NW se, t, p, BH-q
permutation_pvalue(panel, scorer, horizon=1, n=1000, block=21, seed=SEED) -> float
effect_size_bps(coef, se, sigma_s) -> tuple[float, float, float]  # point + 95% CI, in bps per 1σ
```

**`src/plots.py`** — one function per numbered figure in §8. No plotting code anywhere else.

---

## 6. Statistical specifications

Exact regressions. Fit with `nw_ols`, D10 standard errors, every time.

**6.1 Contemporaneous (RQ2)** — one per scorer
```
ret_t = α + β·S_t + φ·ret_{t−1} + γ·parkinson_{t−1} + δ·log_turnover_{t−1} + ε_t
```
Reported as association, not causation, in that language.

**6.2 Predictive (RQ3)** — one per scorer per horizon h ∈ {1…5}
```
ret_{t+h} = α + β_h·S_t + φ·ret_t + γ·parkinson_t + δ·log_turnover_t + ε_t
```
Controls are momentum/reversal, volatility, attention. The specification is deliberately short: a kitchen-sink control set is a specification search wearing a lab coat. BH-FDR (D11) is applied across the five β_h within each scorer. The placebo (D12) is run on β_1.

**6.3 Attenuation comparison (the bridge)**
Run 6.2 at h = 1 with `s_finbert`, then with `s_lm`, identical otherwise. Compare coefficient magnitude and t-statistic. Prediction from errors-in-variables: FinBERT's |β| is larger and its standard error relatively smaller.
Secondary exhibit: both scores in one regression. Expect multicollinearity to blur the result — report the correlation from Stage 4 alongside it and say so rather than treating a muddy horse race as evidence of anything.

**6.4 Volume and volatility (RQ4)** — the act most likely to yield a positive result
```
parkinson_{t+1}      = α + β₁·S_t + β₂·|S_t| + β₃·d_t + φ·parkinson_t + ε_t
turnover_dt_{t+1}    = α + β₁·S_t + β₂·|S_t| + β₃·d_t + φ·turnover_dt_t + γ·|ret_t| + ε_t
```
`|S_t|` captures intensity irrespective of direction; `d_t` is disagreement across the day's headlines. These are estimated for FinBERT and LM; VADER only if it survives the cut list.

**6.5 Effect-size translation (D15)**
For every headline β: `bps per 1σ = β × sd(S_t) × 10,000`, with the CI transformed identically. A null is then reported with content: *"effects larger than X bps per 1σ are ruled out at 95% — below one-way transaction costs."*

**6.6 Tests used, and what each is for**
| Test | Where | Answers |
|---|---|---|
| McNemar (exact) | Act 1 | Is the classifier gap real, given the predictions are paired on the same sentences? |
| Newey–West HAC | Every regression | Standard errors under serial correlation and heteroskedasticity |
| Benjamini–Hochberg | Lag family | What share of declared discoveries are false, across 5 correlated tests? |
| Circular block permutation | β_1 | Assumption-free null that preserves S_t's own autocorrelation |

---

## 7. Build stages

Each stage ends with a **Done when** that is checkable, not a feeling.

### Stage 0 — Scaffold, dataset audit, market data (2–2.5h)
1. Create the §2.2 tree; `config.py` with every §3 constant; pinned `requirements.txt`.
2. Port from Project 1: SPY pull, calendar hygiene, Parkinson computation.
3. Download both D1 candidates. For each, in `01_data_audit.ipynb`:
   - plot the intraday distribution of timestamps (a spike at 00:00 means date-only wearing a timestamp column);
   - confirm the source timezone from documentation, not inference;
   - headlines per year, and per trading day (mean, median, share of zero-news days);
   - exact and near-duplicate rate;
   - ticker-tag sanity on a 50-row sample.
4. **Lock D1 and D4.** Fill the ⏳ cells. If neither passes, invoke D2 now.
5. Download the Financial PhraseBank; record citation and license.

**Done when:** §3 has no placeholders, `headlines.parquet` and `market.parquet` exist for the locked window, and the audit notebook justifies the choice in writing.

### Stage 1 — Scoring pipeline (2.5–3h)
1. Implement the three `Scorer` classes against the §5 protocol. Get the LM master dictionary from the Loughran–McDonald SRAF site.
2. **Time FinBERT on 1,000 headlines and extrapolate before launching the full pass.** If projected CPU time exceeds ~2h, shorten the *window* (D4) and record it. Never subsample headlines within days — that biases both S_t and d_t.
3. Run `rescore.py` → `scores.parquet`, hash-keyed and resumable.
4. Build the qualitative spot-check table: ~30 hand-picked headlines scored by all three, deliberately including cases where the lexicons should fail — `"company reports increased liability provisions"` (LM-neutral, VADER-negative), `"costs fell sharply"` (word-count-negative, actually positive), `"profit warning smaller than feared"`.
5. `tests/test_scoring.py`: every score in [−1, 1]; scoring is deterministic across two runs.

**Done when:** all headlines in the locked window are scored by all three scorers, cached, tests pass, and the spot-check table renders.

### Stage 2 — Act 1: validation (2–2.5h)
1. Load PhraseBank, all four agreement subsets. Stratified 20/80 split, seeded.
2. Fit LM and VADER neutral bands on the 20% (maximize macro-F1). FinBERT untouched.
3. Evaluate all three on the 80%: accuracy, macro-F1, per-class confusion matrices.
4. McNemar per D14. Report the disagreement cells (b, c) alongside the p-value.
5. Repeat the headline comparison on each agreement subset — does the FinBERT–LM gap widen as label quality rises?
6. Write the limitation now, while it is fresh: PhraseBank sentences are annotated from an investor's perspective and FinBERT was fine-tuned on related financial text, so a shared-provenance advantage cannot be ruled out. Better in your limitations section than in a reader's question.

**Done when:** Table 1 and Figure 1 exist, and you can state in one sentence: "FinBERT exceeds LM by X macro-F1 points (McNemar p = …); the VADER→LM step contributes Y of the total gain."

### Stage 3 — Alignment and the daily panel (2h)
1. Implement `map_to_trading_day` per D6 against the NYSE calendar.
2. **Write `tests/test_alignment.py` before trusting any output.** Required cases:
   - headline at 15:59 ET on day t → day t;
   - the same headline at 16:01 ET → day t+1;
   - Saturday 10:00 ET → the following Monday;
   - a headline dated on a market holiday → the next trading day;
   - assertion that no row in `daily_panel` at date t was built from a headline with `ts_et` > close(t).
3. `aggregate_daily` (D8/D9) → `build_panel` → `daily_panel.parquet`.
4. EDA in `01_data_audit.ipynb` (extend it): S_t per scorer over time against SPY, with major episodes marked; n_t over time (coverage drift is a named limitation); **ACF of S_t** (its persistence is the written justification for D10 and D12); pairwise correlation of the three daily S_t series.

**Done when:** all alignment tests pass, `daily_panel.parquet` matches the §4 schema exactly, and the panel's row count, n_t = 0 day count, and scorer correlations are recorded in the notebook.

> Note from Stage 3 EDA that changes what you claim later: if `corr(s_finbert, s_lm) > 0.9` daily, the attenuation comparison (§6.3) has little room to separate the scorers. Report it as bounded and inconclusive in that case. Do not redesign the comparison to manufacture a difference.

### Stage 4 — Act 2: inference (3–4h) ← the intellectual core
1. Run §6.1 for all scorers. This is the pipeline's sanity check as much as a result: if same-day association is absent, something upstream is broken — check alignment before believing it.
2. Run §6.2 for all scorers × 5 horizons. Assemble the coefficient table: β, NW se, t, raw p, BH q.
3. Permutation placebo (D12) on β_1 for each scorer.
4. Attenuation comparison per §6.3, with the Stage 3 correlation quoted next to it.
5. Effect sizes per §6.5 for every headline coefficient.

**Done when:** `03_signal.ipynb` produces Tables 2–4 and Figure 2 from `daily_panel.parquet` alone — no re-reading of raw text or prices anywhere in the notebook.

### Stage 5 — Volume and volatility (1.5–2h)
1. Run both §6.4 specifications for FinBERT and LM.
2. Report the d_t coefficient prominently — disagreement predicting volume is the project's most plausible positive finding, and it is a documented one in the literature.
3. Note the sample reduction from the n_t ≥ 5 requirement on d_t.

**Done when:** Table 5 and Figure 3 exist, each with a one-sentence verdict.

### Stage 6 — Robustness (1–1.5h)
Each item is a short section with a one-sentence verdict, not a discussion.
1. Median instead of mean aggregation (D8 variant).
2. NW maxlags = 10.
3. First-half / second-half subperiod split.
4. Drop days with n_t < 5 from the S_t specifications too.
5. Single-name spot check: most-covered ticker, its own headlines vs. its own returns.

**Done when:** `05_robustness.ipynb` renders with five verdicts and no result reverses silently.

### Stage 7 — Write-up and polish (3–4h)
1. **Report, 2 pages:** (i) question and the two-act logic, with the attenuation bridge stated up front; (ii) data — the audit, the D6 timestamp rule, coverage caveats; (iii) Act 1 — the ladder and McNemar; (iv) Act 2 — contemporaneous vs. predictive, FDR, placebo, ruled-out effect sizes; (v) volume/volatility; (vi) limitations — shared provenance of FinBERT and PhraseBank, coverage drift, single market, association not causation, no intraday confirmation; (vii) two sentences on why there is no trading backtest.
2. **README:** the question in one sentence, the headline figure, three bullet findings, `pip install -r requirements.txt && python run_all.py`, repo map, and one line pointing at `tests/test_alignment.py` as the look-ahead firewall.
3. Figure pass: consistent style, labeled axes, captions that state the takeaway rather than describing the axes.
4. Clean-clone run of `run_all.py`.

**Done when:** a stranger gets the finding from the README in 90 seconds and reproduces every figure with one command.

---

## 8. Deliverable inventory

Build exactly these. Anything else is scope creep.

| ID | Artifact | Stage |
|---|---|---|
| Table 1 | PhraseBank: accuracy, macro-F1 per scorer, per agreement subset; McNemar p | 2 |
| Table 2 | Contemporaneous regression, all scorers | 4 |
| Table 3 | Lag family: β_h, NW se, t, p, BH q, for h = 1…5 per scorer | 4 |
| Table 4 | Effect sizes: bps per 1σ with 95% CI, vs. transaction-cost benchmark | 4 |
| Table 5 | Volatility and volume regressions | 5 |
| Figure 1 | Confusion matrices, three scorers side by side | 2 |
| Figure 2 | **Headline.** Coefficient ± NW CI by horizon (t, t+1…t+5), one panel per scorer, zero line marked, placebo p annotated on h = 1 | 4 |
| Figure 3 | Sentiment dispersion d_t vs. next-day detrended volume, with fitted line | 5 |
| Figure 4 | S_t (FinBERT) and SPY price over the sample, episodes marked — context, not evidence | 3 |
| Exhibit A | The 30-headline qualitative scoring table | 1 |

---

## 9. Week layout

| Day | Stages | Hours |
|---|---|---|
| 1 | Stage 0 | 2–2.5 |
| 2 | Stage 1 (start the FinBERT pass early, let it run) | 2.5–3 |
| 3 | Stage 2 | 2–2.5 |
| 4 | Stage 3 | 2 |
| 5 | Stage 4 | 3–4 |
| 6 | Stages 5 + 6 | 2.5–3.5 |
| 7 | Stage 7 | 3–4 |

**Checkpoint rule:** if no dataset passes the timestamp audit by the end of day 1, invoke D2 immediately. Do not spend day 2 hunting for better data.

---

## 10. Cut list (in order, if over budget)

1. VADER (LM remains the sole lexicon; the ladder loses its bottom rung, Act 1 survives intact).
2. Horizons t+4 and t+5 (family shrinks to three; FDR still applies).
3. Single-name spot check and median-aggregation robustness.
4. The joint horse-race regression (keep the side-by-side attenuation comparison).
5. The volume outcome (keep the volatility one — it is a single regression on ported code).

**Never cut:** the timestamp audit, `tests/test_alignment.py`, Newey–West, BH-FDR, the permutation placebo, the effect-size CIs, or the limitations section. These are the maturity signals the project exists to send.

**Stretch, only if under budget:** a Streamlit view of S_t against price; scoring a small sample of EDGAR 8-K headlines as an out-of-domain check.

---

## 11. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| News dataset has absent or unreliable intraday timestamps | Medium-high | Two candidates, explicit Stage 0 audit, D2 fallback, day-1 deadline |
| FinBERT inference too slow on CPU | Medium | Time-and-extrapolate before the full pass; batch + truncate at 64 tokens; shorten the window, never the within-day sample; cache so it runs once |
| RQ3 comes back null | High, and planned for | Declared in advance; ruled-out-interval framing (§6.5); RQ4 carries the positive finding |
| FinBERT and LM daily series nearly collinear | Medium | Detected at Stage 3; the attenuation comparison is then reported as bounded — an honest sentence, not a redesign |
| Duplicate or syndicated headlines inflate n_t and distort S_t | Medium | `dedup` at Stage 0, exact + near-duplicate; dedup rate reported |
| Coverage drift over the sample (more headlines in later years) | Likely | Plotted at Stage 3, named as a limitation, subperiod split at Stage 6 |
| Scope creep — transcripts, more tickers, a trading rule | Self-inflicted | §3 is the contract; ideas go to `future-work.md` |

---

## 12. Interview ammunition

- **Why Newey–West.** Sentiment is persistent and regression residuals are serially correlated and heteroskedastic; OLS standard errors assume neither, so naive t-statistics overstate significance. The lag length is a bandwidth choice, sensitivity-checked, not a constant handed down from anywhere.
- **Why Benjamini–Hochberg over Bonferroni.** Bonferroni controls the probability of *any* false positive and sacrifices power badly when tests are correlated — which five adjacent lags certainly are. BH controls the expected *proportion* of false discoveries, the right trade-off for a small exploratory family declared in advance.
- **Why McNemar.** The classifiers score the *same* sentences, so predictions are paired; McNemar tests the asymmetry of the disagreement cells. Comparing two accuracies as if they came from independent samples discards the pairing and gets the variance wrong — the same error as comparing two forecasts without Diebold–Mariano.
- **Contemporaneous versus predictive.** News moving prices the same day is what an efficient market predicts; next-day predictability from public headlines is the anomalous claim and therefore carries the heavier evidentiary burden (FDR plus placebo). Knowing which of your two results would be surprising is the domain-maturity signal.
- **The attenuation bridge.** Sentiment scores measure a latent quantity with error, and classical errors-in-variables shrinks β toward zero in proportion to that error. So a better classifier should show a larger, sharper coefficient on an identical specification — which is what makes Act 1 predictive of Act 2 rather than decorative.
- **Why block permutation.** Circularly shifting S_t preserves its own autocorrelation while destroying its alignment with returns, giving a null distribution that assumes nothing about the error process. The same instinct as a block bootstrap: never let a resampling scheme destroy the dependence you are worried about.
- **Why no trading backtest.** A backtest converts an inference question into a search over costs, sizing and rebalancing rules — every one of them p-hackable. The bps-per-1σ comparison against transaction costs delivers economic significance with none of that surface area.
- **The Loughran–McDonald point.** In general English, *liability*, *tax* and *vice* read negative; in filings they are neutral boilerplate. That is why domain lexicons exist — and *"costs fell sharply"* is why context models beat word counts, because no dictionary sees the verb.

---

## 13. Resources

- Tetlock (2007), *Giving Content to Investor Sentiment* — the canonical media-sentiment paper; §6.4 mirrors its outcome variables. Working-paper version free online.
- Loughran & McDonald (2011), *When Is a Liability Not a Liability?* — the domain-lexicon argument; dictionary free from the authors' SRAF site.
- Malo et al. (2014) — Financial PhraseBank; dataset on HuggingFace (`financial_phrasebank`).
- Araci (2019), *FinBERT* (arXiv); model at `ProsusAI/finbert`.
- FNSPID (paper + HuggingFace dataset) and the Kaggle Benzinga headline dataset — the two D1 candidates.
- `statsmodels`: `OLS.fit(cov_type="HAC", cov_kwds={"maxlags": 5})`; `stats.multitest.multipletests(method="fdr_bh")`; `stats.contingency_tables.mcnemar`.
- Newey & West (1987); Benjamini & Hochberg (1995) — read for the idea, use the library.
