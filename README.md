# news-sentiment-signal

**How do financial sentiment measurements differ in classification quality, and what additional
information do they provide about subsequent market outcomes?**

Three scorers — a general-purpose lexicon, a domain lexicon and a domain transformer — are compared
on independently labelled financial headlines, and their daily aggregate tone is then tested against
the *next* session's SPY return. The two questions are kept apart on purpose: better classification
is not assumed to imply a stronger market coefficient, and the contribution does not depend on
finding a signal.

> **Status: infrastructure, no results.** The protocols are written and frozen, the corpus is
> assembled and censused, and the timing, caching, acquisition and panel defects found by the
> [2026-09-09 audit](docs/project-audit-2026-09-09.md) are being repaired in sequence — 11 of 30
> increments done. **No text has been scored, no labels collected, no panel built, no model fitted,
> and no empirical result of any kind exists.** 216 tests pass, 0 skip.
>
> Current state lives in [`docs/handover.md`](docs/handover.md); the live plan is the
> [repair sequence](docs/audit-implementation-plan-2026-09-09.md).

---

## The two acts

**Act 1 — classification quality.** On headlines from this study's own collection, labelled by
humans who cannot see any model's output, how do FinBERT, Loughran–McDonald and VADER differ in
macro-F1? The primary contrast is `macroF1(FinBERT) − macroF1(LM)` with a paired bootstrap that
resamples **article groups**, since near-duplicate headlines are not independent draws.
Financial PhraseBank is *not* the primary evaluation: `ProsusAI/finbert`'s own model card names it
as fine-tuning data, so it is retained only as a supplementary exhibit carrying a contamination
statement. Full specification: [`docs/validation-protocol.md`](docs/validation-protocol.md).

**Act 2 — market association.** Is daily aggregate FinBERT tone associated with the *next* trading
session's SPY log return, conditional on a frozen control set? One primary test, reported in basis
points per standard deviation with a pointwise Newey–West interval; 14 secondary (scorer, horizon)
tests under BH with Benjamini–Yekutieli alongside. Same-day association is **structurally
suppressed** — no FNSPID sub-corpus is both intraday-stamped and relevant, so the timestamps cannot
support it. Full specification: [`docs/inference-protocol.md`](docs/inference-protocol.md).

**The bridge, and its status as a hypothesis.** Sentiment scores are noisy measurements of a latent
quantity, and classical errors-in-variables attenuates a coefficient toward zero in proportion to
measurement noise. That motivates a **conditional hypothesis** — on an identical specification, a
better classifier *might* produce a larger, better-determined coefficient — and the assumptions it
needs are stated on display rather than asserted as a mechanism. It is not a prediction the design
guarantees, and a null or inconclusive Act 2 does not falsify Act 1.

## Findings

**There are none yet, and this section stays empty until there are.**

No text has been scored, no labels collected, no panel built and no model fitted. This section
previously held placeholder bullets with the shape of the expected answer already written in — an
FDR-controlled five-horizon family, a block-permutation placebo, a transaction-cost bar. All three
describe procedures the accepted protocols replaced, and writing the conclusion's skeleton before
the evidence is the pattern this project spent the audit removing.

What can be said now is what the design permits: three conclusions are available to Act 2 —
evidence of association, evidence the effect is small against a prespecified 5 bps yardstick, and an
explicit **inconclusive** — and the last is a legitimate result that will be reported as one.

## Reproduce

```bash
pip install -r requirements.txt
python data/raw/download.py --assemble   # stream the pinned 5.7 GB FNSPID file, verify its digest
python data/raw/download.py --dedup     # raw -> analysis corpus, with lineage
python data/raw/download.py --census    # coverage, duplication, concentration
python data/raw/download.py --market    # SPY/^VIX; the LM dictionary is obtained by hand
python rescore.py                   # once: the FinBERT pass, cached by headline hash
python run_all.py                   # minutes, no GPU: panel -> every table and figure
pytest                              # the firewall
```

`rescore.py` is deliberately **not** part of `run_all.py`. FinBERT inference over the full headline
set is the only expensive step in the project; scores are cached keyed on headline hash, so the
command a reader actually runs takes minutes and needs no GPU.

## Two properties of the design

**One analysis table.** Every Act-2 number comes from `data/processed/daily_panel.parquet`. If a
result is wrong, it is wrong in the panel or in the regression — never in an ad-hoc join inside a
notebook. Notebooks contain no analysis logic: they import from `src/`, call, and display.

**One firewall.** [`tests/test_alignment.py`](tests/test_alignment.py) is where look-ahead bias goes
to die. The corpus is **date-only** — the audit established that no FNSPID sub-corpus is both
intraday-stamped and relevant — so the intraday close rule is descoped and the active rule is
`map_date_to_session`: a headline dated *d* belongs to the **first session strictly after *d***, so
session *t* receives dates in [prev_session, *t*) and all of day *d* precedes close(*t*). It reads
the date in the source's own zone and never converts, because converting a `00:00 UTC` stamp to
market time moves it back a calendar day and shifts every headline one session early; a test
documents that trap. The two mappers refuse each other's input in both directions, so the descope
cannot be undone by accident. Every forward-looking column is created by exactly one shift, and the
tests assert that no row of the panel at date *t* was built from a headline that postdates
close(*t*). These run on a synthetic calendar with no dataset present, which is the point: the
firewall must be checkable before there is any data to be wrong about.

## Repo map

```
config.py            every locked decision (D1–D16) from the plan; imported everywhere
run_all.py           panel -> Tables 2-5, Figures 2-4
rescore.py           the one-off FinBERT pass (--time-only times it first)
src/data.py          news load + mandatory source filter, dedup with lineage, SPY/^VIX, NYSE calendar
src/scoring.py       LM | VADER | FinBERT, behind one Scorer protocol; hash-keyed cache
src/align.py         both mappers, daily aggregation, the panel, all lags and leads, score gate
src/validate.py      Act 1 metrics (not yet reworked to the validation protocol -- R06b)
src/inference.py     Newey-West OLS, BH lag family, bps effect sizes
                     (still the original plan's methods -- R07/R08 replace them)
src/plots.py         one function per numbered figure; no plotting code anywhere else
tests/               the firewall + scorer range/determinism/cache checks
notebooks/           01 audit · 02 validation · 03 signal · 04 volume+vol · 05 robustness
docs/                the frozen implementation plan
report/report.md     the two-page write-up
future-work.md       where scope creep goes to die quietly
```

## Method notes

- **Newey–West everywhere.** Sentiment is persistent and residuals are serially correlated and
  heteroskedastic; naive OLS standard errors overstate significance. maxlags = 5, with 10 as a
  sensitivity check.
- **Benjamini–Hochberg, not Bonferroni.** Five adjacent lags are strongly correlated; Bonferroni
  controls the probability of *any* false positive and sacrifices power badly there. BH controls the
  expected *proportion* of false discoveries — the right trade-off for a small family declared in
  advance.
- **McNemar, not two accuracies.** The classifiers score the *same* sentences, so the predictions
  are paired. Comparing two accuracies as if they came from independent samples discards the pairing
  and gets the variance wrong.
- **Circular block permutation.** Shifting S_t in blocks preserves its own autocorrelation while
  destroying its alignment with returns — a null that assumes nothing about the error process.
- **No trading backtest, on purpose.** A backtest turns an inference question into a specification
  search over costs, sizing and rebalancing, every one of them p-hackable. Economic significance is
  delivered instead by the bps-per-1σ figure against a transaction-cost benchmark.

## Data and credits

- Financial PhraseBank — Malo et al. (2014), via HuggingFace `financial_phrasebank`.
- FinBERT — Araci (2019), model `ProsusAI/finbert`. Used as published; nothing is fine-tuned here.
- Loughran–McDonald master dictionary — Loughran & McDonald (2011), from the Notre Dame SRAF site.
- Headlines — FNSPID or the Kaggle Benzinga headline set (D1, locked at Stage 0).
- Prices — SPY and ^VIX via `yfinance`.

Nothing under `data/` is tracked: the news dump is large and redistribution-restricted.
