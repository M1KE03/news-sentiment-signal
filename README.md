# news-sentiment-signal

**How do financial sentiment measurements differ in classification quality, and what additional
information do they provide about subsequent market outcomes?**

Three scorers — a general-purpose lexicon, a domain lexicon and a domain transformer — are compared
on independently labelled financial headlines, and their daily aggregate tone is then tested against
the *next* session's SPY return. The two questions are kept apart on purpose: better classification
is not assumed to imply a stronger market coefficient, and the contribution does not depend on
finding a signal.

> **Status: infrastructure, no results.** The protocols are written and frozen, the corpus is
> assembled and censused, and the timing, caching, acquisition, panel and inference defects found by
> the [2026-09-09 audit](docs/project-audit-2026-09-09.md) are being repaired in sequence — 22 of 29
> increments done. **No text has been scored, no labels collected, no panel built, no model fitted,
> and no empirical result of any kind exists.** 388 tests pass, 0 skip.
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

What can be said now is what the design permits. Act 2 reports **two independent facts about the
same interval**, always together: whether it excludes zero, and whether it lies inside ±5 bps. An
earlier version of this section listed *three* conclusions — association, smallness, inconclusive —
which overlapped: an interval of [1, 3] bps satisfied the first two at once and no rule chose
between them. That framing was withdrawn (M2). Of the four combinations the two facts produce, one
is an informative null and one is an explicit **inconclusive**; both are legitimate results and will
be reported as such.

## Reproduce

```bash
pip install -r requirements.txt
python data/raw/download.py --assemble   # stream the pinned 5.7 GB FNSPID file, verify its digest
python data/raw/download.py --dedup     # raw -> analysis corpus, with lineage
python data/raw/download.py --census    # coverage, duplication, concentration
python data/raw/download.py --market    # SPY/^VIX; LM acquisition is documented below
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
src/validate.py      Act 1: label ingestion, calibration-only thresholds, paired group bootstrap
src/inference.py     eligibility ledger, the frozen primary spec, session-indexed HAC,
                     the 14-test secondary family (BH/BY), paired scorer contrast, timing shift
src/plots.py         one function per numbered figure; no plotting code anywhere else
tests/               the firewall + scorer range/determinism/cache checks
notebooks/           01 audit · 02 validation · 03 signal · 04 volume+vol · 05 robustness
docs/                the frozen protocols, the audit, and the repair sequence
report/report.md     the two-page write-up
future-work.md       where scope creep goes to die quietly
```

## Method notes

- **Newey–West everywhere.** Sentiment is persistent and residuals are serially correlated and
  heteroskedastic; naive OLS standard errors overstate significance. `L = 5` is prespecified as one
  trading week, with `L ∈ {0, 1, 10}` and the data-driven plug-in bandwidth reported alongside as
  sensitivities rather than substituted for it. A lag counts **exchange sessions**, not rows of the
  analysis sample — the two differ wherever the sample has gaps, and only the first makes "one
  trading week" true ([the spacing decision](docs/hac-spacing-decision.md)).
- **One primary test, then a closed family of 14.** FinBERT at h = 1 is the primary and carries no
  correction, because a family of one needs none. The remaining 14 (scorer, horizon) pairs are
  corrected together: Bonferroni controls the probability of *any* false positive and sacrifices
  power badly on tests this correlated, so Benjamini–Hochberg at q = 0.05 is used. BH's guarantee
  needs positive dependence, which is plausible here but not established, so **Benjamini–Yekutieli**
  — valid under arbitrary dependence — is reported alongside, and where they disagree the claim is
  made at the BY level.
- **Group bootstrap, not two accuracies.** The classifiers score the *same* headlines, so the
  predictions are paired — but pairing alone is not enough, because near-duplicate headlines from
  one story are not independent draws either. Both the macro-F1 and the accuracy comparisons
  resample whole **article groups**. Exact McNemar is retained as a supplementary exhibit, printed
  with the group-size distribution and the explicit note that its independence assumption is
  contradicted by this design's own grouping (M4).
- **A circular shift, reported as a percentile.** Every shift of the standardized tone series is a
  bijection: each observation is used exactly once and the series' autocorrelation is preserved
  exactly. An earlier version drew blocks **with replacement**, which duplicated some observations
  and omitted others, so it was not a permutation at all. It is a **descriptive timing diagnostic**
  and never a p-value: shifting tone also destroys its relationship with the controls, so the
  resulting spread is not the null distribution of the conditional coefficient — and no procedure
  here is assumption-free.
- **No trading backtest, on purpose.** A backtest turns an inference question into a specification
  search over costs, sizing and rebalancing, every one of them p-hackable. What replaces it is a
  **yardstick for smallness**, not a profitability test: 5 bps per 1σ is the order of magnitude of
  one-way execution cost in a large liquid ETF, so an association below it is small relative to
  frictions any user would face. A coefficient above it does not establish that a strategy makes
  money, and one below it does not establish the information is useless — realised value depends on
  signal use, timing, turnover, holding period and capacity, none of which this design measures.

## Data and credits

- Financial PhraseBank — Malo et al. (2014), via HuggingFace `financial_phrasebank`.
- FinBERT — Araci (2019), model `ProsusAI/finbert`. Used as published; nothing is fine-tuned here.
- Loughran–McDonald master dictionary — Loughran & McDonald (2011), from Notre Dame SRAF; [acquired release, hash and reproduction notes](docs/lm-dictionary-provenance.md).
- Headlines — FNSPID or the Kaggle Benzinga headline set (D1, locked at Stage 0).
- Prices — SPY and ^VIX via `yfinance`.

Nothing under `data/` is tracked: the news dump is large and redistribution-restricted.
