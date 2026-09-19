# news-sentiment-signal

**How do financial sentiment measurements differ in classification quality, and
what additional information do they provide about subsequent market outcomes?**
Three scorers (a general-purpose lexicon, a domain lexicon and a domain
transformer) are compared on independently labelled financial headlines, and
their daily aggregate tone is then tested against the *next* session's SPY
return. The two questions are kept apart on purpose: better classification is
not assumed to imply a stronger market coefficient, and the contribution does
not depend on finding a signal.

**The classifiers separate, and the whole gap sits in one class.** On 596 usable
hand-labelled headlines from this study's own collection, macro-F1 is 0.594
(FinBERT), 0.493 (Loughran–McDonald) and 0.424 (VADER); the FinBERT−LM gap is
**+0.102, 95% pointwise [+0.050, +0.153]** by paired bootstrap over article
groups, and +0.104 on the confident subset. But FinBERT and LM are within 0.02
on negative and neutral: LM calls 81.5% of genuinely positive headlines
neutral, because its score takes only three values on headline-length text and
the dictionary carries 2,345 negative terms against 347 positive, an asymmetry
built for 10-K risk language. That says something about applying a
document-level dictionary to headlines; it is not evidence that a transformer
reads financial language better.

**The market association is an informative null.** Daily FinBERT tone against
the next session's SPY log return, conditional on a frozen control set, over
2,453 eligible sessions: **β = −1.74 bps per 1σ of tone, 95% pointwise
Newey–West interval [−4.90, +1.41], p = 0.279.** No association detected, and
precise enough to exclude effects beyond the prespecified 5 bps yardstick.
**Caution: that second half is a boundary case, and the flag is part of the
result.** The lower endpoint is −4.9031, within 0.1 bps of −5. *No association
detected* is robust across every prespecified bandwidth, both aggregation rules
and both sample halves; *precise enough to exclude ±5 bps* is fragile: 5 of 7
sensitivities are boundary cases, and under median rather than mean aggregation
the estimate collapses to −0.17 bps [−3.18, +2.84].

**The two acts do not connect, and that was derived in advance.** Act 1
separates the scorers; Act 2 cannot (`delta = −2.43 bps [−6.88, +2.03]`).
Standardization puts the attenuation exponent at ½, and averaging ~337
headlines per session turns a *twofold* per-headline noise difference into a
*1.034×* coefficient difference. The bridge is not refuted. It is not testable
at this aggregation, so Act 2 is weak evidence about relative classification
quality, not evidence that the scorers are equivalent.

**Status: both acts complete.** 869,183 headlines scored by all three scorers,
800 labelled by hand, every choice prespecified; 546 tests pass, 0 skip.
Write-up: [`report/report.md`](report/report.md). Clean-clone procedure:
[`docs/archive/reproduction.md`](docs/archive/reproduction.md).

---

## Findings

Per-scorer detail behind the opening: 600 independently labelled headlines from this study's own
collection, 596 usable, thresholds fitted on a separate 200-item calibration part.

| Scorer | macro-F1 | positive-class F1 |
|---|---:|---:|
| **FinBERT** | **0.594** | **0.546** |
| Loughran–McDonald | 0.493 | 0.263 |
| VADER | 0.424 | 0.390 |

On the confident subset (`hard = 0`, n = 541) the gap is +0.104, so it does not come from the
headlines a careful human could not resolve. LM's three values on headline-length text are `−1`,
`0` and `+1`; 78% score exactly 0.

Act 2 reports two independent facts about the same interval, always together. Here the interval
includes zero and lies inside ±5 bps, which is the informative null. The 0.1 bps margin quoted
above is the reporting precision: one-tenth of a basis point wider and the conclusion would have
been inconclusive. Quoting the null without this qualification would overstate it.

The two halves of that conclusion do not carry equal weight, and the robustness suite is why.
Beyond the sensitivities summarised above, standard errors are smaller at longer bandwidths, so
the prespecified L = 5 sits in the range producing the narrow interval. Both halves are reported;
the second is not presented with the confidence of the first.

The robustness suite is also weaker than a list of five robustness checks implies. Two exhibits are
informative, one is bounded by power, and two are inapplicable to this corpus: the dispersion floor
excludes exactly one session, and the single-name check runs on tickers averaging 1.2 headlines per
session with up to 56% of sessions empty. Absence of divergence in those is not evidence of
stability. Detail in [`report/report.md`](report/report.md) §7.

Nothing rejects anywhere else. The 14-test secondary family: smallest BH q = 0.946, BY q = 1.000.
The four exploratory RQ4 joint Wald tests: smallest BH q = 0.087. Two individual RQ4 coefficients
have intervals excluding zero and neither is reported as a finding, because the joint test that
governs them does not reject; that ordering is what stops a coefficient becoming a headline after
someone has looked at it.

The two scorers cannot be separated. `delta = β_FinBERT − β_LM = −2.43 bps, 95% [−6.88, +2.03]`,
estimated on identical observations with the cross-equation HAC covariance. The
[mathematical appendix](docs/archive/mathematical-appendix.md) gives a reason internal to the design:
averaging a median of 337 headlines per session compresses a *twofold* per-headline noise difference
into a *1.034×* coefficient difference. So this is weak evidence about relative classification
quality, not strong evidence of similarity.

Full write-up with every number traced to a file: [`report/report.md`](report/report.md).

## The two acts

**Act 1: classification quality.** On headlines from this study's own collection, labelled by
humans who cannot see any model's output, how do FinBERT, Loughran–McDonald and VADER differ in
macro-F1? The primary contrast is `macroF1(FinBERT) − macroF1(LM)` with a paired bootstrap that
resamples *article groups*, since near-duplicate headlines are not independent draws.
Financial PhraseBank is *not* the primary evaluation: `ProsusAI/finbert`'s own model card names it
as fine-tuning data, so it is retained only as a supplementary exhibit carrying a contamination
statement. Full specification: [`docs/validation-protocol.md`](docs/validation-protocol.md).

**Act 2: market association.** Is daily aggregate FinBERT tone associated with the *next* trading
session's SPY log return, conditional on a frozen control set? One primary test, reported in basis
points per standard deviation with a pointwise Newey–West interval; 14 secondary (scorer, horizon)
tests under BH with Benjamini–Yekutieli alongside. Same-day association is structurally
suppressed: no FNSPID sub-corpus is both intraday-stamped and relevant, so the timestamps cannot
support it. Full specification: [`docs/archive/inference-protocol.md`](docs/archive/inference-protocol.md).

**The bridge, and its status as a hypothesis.** Sentiment scores are noisy measurements of a latent
quantity, and classical errors-in-variables attenuates a coefficient toward zero in proportion to
measurement noise. That motivates a conditional hypothesis: on an identical specification, a
better classifier *might* produce a larger, better-determined coefficient. The assumptions it
needs are stated on display rather than asserted as a mechanism. It is not a prediction the design
guarantees, and a null or inconclusive Act 2 does not falsify Act 1.

## Reproduce

```bash
pip install -r requirements.txt
python preflight.py --stage scoring      # what is installed, and which artifacts exist
python data/raw/download.py --assemble   # stream the pinned 5.7 GB FNSPID file, verify its digest
python data/raw/download.py --dedup     # raw -> analysis corpus, with lineage
python data/raw/download.py --census    # coverage, duplication, concentration
python data/raw/download.py --market    # SPY/^VIX; LM acquisition is documented below
python rescore.py --dry-run              # scope and checkpoint location, nothing scored
python rescore.py                        # once: the FinBERT pass, cached by headline hash
python run_all.py                        # minutes, no GPU: panel -> every table and figure
pytest                                   # the firewall
```

Results are published only if the whole run succeeds. `run_all.py` writes to a
staging area and moves nothing into `report/tables/` or `figures/` until the last
output is produced; `run_manifest.json` is written last and is therefore the
commit point. A directory without a current manifest is an incomplete run, not a
set of results. `src.publish.verify_published()` checks a published directory
against its manifest and reports anything edited, deleted, or whose inputs have
changed since. The manifest records input digests, every frozen setting that
changes a number, the scorer identities, the eligibility ledger, and whether the
git tree was dirty.

`--skip-panel` reuses a saved panel, and refuses one whose recorded inputs or
settings no longer match. That check is not a schema check: a panel built under
different timing or scoring semantics has identical columns.

Every command above refuses to start when its inputs are absent, and says which
artifact is missing and what produces it. `python preflight.py --stage analysis`
answers that question without running anything; `--strict` exits non-zero, which
is the form for a check before an expensive pass.

Bounded scoring is bounded by whole sessions. `rescore.py --sessions 50`
scores fifty calendar days entire. There is deliberately no headline-count
limit: a partial day is not a smaller sample but a different measurement, since
`S_t` is a within-day mean and `d_t` a within-day standard deviation. If the
full pass is too expensive the rule is to shorten the *window* (D4) and record
it.

`rescore.py` is deliberately *not* part of `run_all.py`. FinBERT inference over the full headline
set is the only expensive step in the project; scores are cached keyed on headline hash, so the
command a reader actually runs takes minutes and needs no GPU.

## Two properties of the design

**One analysis table.** Every Act-2 number comes from `data/processed/daily_panel.parquet`. If a
result is wrong, it is wrong in the panel or in the regression, never in an ad-hoc join inside a
notebook. Notebooks contain no analysis logic: they import from `src/`, call, and display.

**One firewall.** [`tests/test_alignment.py`](tests/test_alignment.py) is where look-ahead bias goes
to die. The corpus is date-only, since the audit established that no FNSPID sub-corpus is both
intraday-stamped and relevant, so the intraday close rule is descoped and the active rule is
`map_date_to_session`: a headline dated *d* belongs to the *first session strictly after d*, so
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
preflight.py         what is installed, what artifacts exist, what produces them
rescore.py           the one-off FinBERT pass (--time-only, --dry-run, --sessions N)
src/data.py          news load + mandatory source filter, dedup with lineage, SPY/^VIX, NYSE calendar
src/scoring.py       LM | VADER | FinBERT, behind one Scorer protocol; hash-keyed cache
src/align.py         both mappers, daily aggregation, the panel, all lags and leads, score gate
src/validate.py      Act 1: label ingestion, calibration-only thresholds, paired group bootstrap
src/inference.py     eligibility ledger, the frozen primary spec, session-indexed HAC,
                     the 14-test secondary family (BH/BY), paired scorer contrast, timing shift
src/preflight.py     the dependency inventory and the artifact registry
src/plots.py         one function per numbered figure; no plotting code anywhere else
tests/               the firewall + scorer range/determinism/cache checks
notebooks/           01 audit · 02 validation · 03 signal · 04 volume+vol · 05 robustness
docs/specification.md   frozen decisions, code map, invariants, safe-change procedure
docs/validation-protocol.md  the annotation rubric (hash-locked; do not edit)
docs/archive/        the decision log, the audit, the execution record, the contracts
report/report.md     the two-page write-up
FINDINGS.md          what the study found, in plain language with the mathematics
```

## Method notes

- **Newey–West everywhere.** Sentiment is persistent and residuals are serially correlated and
  heteroskedastic; naive OLS standard errors overstate significance. `L = 5` is prespecified as one
  trading week, with `L ∈ {0, 1, 10}` and the data-driven plug-in bandwidth reported alongside as
  sensitivities rather than substituted for it. A lag counts *exchange sessions*, not rows of the
  analysis sample: the two differ wherever the sample has gaps, and only the first makes "one
  trading week" true ([the spacing decision](docs/archive/hac-spacing-decision.md)).
- **One primary test, then a closed family of 14.** FinBERT at h = 1 is the primary and carries no
  correction, because a family of one needs none. The remaining 14 (scorer, horizon) pairs are
  corrected together: Bonferroni controls the probability of *any* false positive and sacrifices
  power badly on tests this correlated, so Benjamini–Hochberg at q = 0.05 is used. BH's guarantee
  needs positive dependence, which is plausible here but not established, so Benjamini–Yekutieli,
  valid under arbitrary dependence, is reported alongside, and where they disagree the claim is
  made at the BY level.
- **Group bootstrap, not two accuracies.** The classifiers score the *same* headlines, so the
  predictions are paired. Pairing alone is not enough, because near-duplicate headlines from
  one story are not independent draws either. Both the macro-F1 and the accuracy comparisons
  resample whole article groups. Exact McNemar is retained as a supplementary exhibit, printed
  with the group-size distribution and the explicit note that its independence assumption is
  contradicted by this design's own grouping (M4).
- **A circular shift, reported as a percentile.** Every shift of the standardized tone series is a
  bijection: each observation is used exactly once and the series' autocorrelation is preserved
  exactly. An earlier version drew blocks with replacement, which duplicated some observations
  and omitted others, so it was not a permutation at all. It is a descriptive timing diagnostic
  and never a p-value: shifting tone also destroys its relationship with the controls, so the
  resulting spread is not the null distribution of the conditional coefficient; no procedure
  here is assumption-free.
- **No trading backtest, on purpose.** A backtest turns an inference question into a specification
  search over costs, sizing and rebalancing, every one of them p-hackable. What replaces it is a
  yardstick for smallness, not a profitability test: 5 bps per 1σ is the order of magnitude of
  one-way execution cost in a large liquid ETF, so an association below it is small relative to
  frictions any user would face. A coefficient above it does not establish that a strategy makes
  money, and one below it does not establish the information is useless: realised value depends on
  signal use, timing, turnover, holding period and capacity, none of which this design measures.

## Project history

The protocols were written and frozen, the corpus assembled, scored and censused, and the defects
found by the [2026-09-09 audit](docs/archive/project-audit-2026-09-09.md) repaired (29 of 29
increments). The clean-clone procedure is
[`docs/archive/reproduction.md`](docs/archive/reproduction.md).

## Data and credits

- Financial PhraseBank: Malo et al. (2014), via HuggingFace `financial_phrasebank`.
- FinBERT: Araci (2019), model `ProsusAI/finbert`. Used as published; nothing is fine-tuned here.
- Loughran–McDonald master dictionary: Loughran & McDonald (2011), from Notre Dame SRAF; [acquired release, hash and reproduction notes](docs/archive/lm-dictionary-provenance.md).
- Headlines: FNSPID or the Kaggle Benzinga headline set (D1, locked at Stage 0).
- Prices: SPY and ^VIX via `yfinance`.

Nothing under `data/` is tracked: the news dump is large and redistribution-restricted.
