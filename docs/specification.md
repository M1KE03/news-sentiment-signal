# Specification and maintenance guide

Everything you need to change this project safely, in one file. Written 2026-09-11, after both acts completed.

**Read this before altering anything that produces a number.** The study's credibility rests on decisions having been fixed *before* results were seen. Changing one now is allowed — it just has to be recorded as a change made with knowledge of the results, which is a weaker claim than a prespecification. That distinction is the whole point.

Two other documents remain live:

- [`validation-protocol.md`](validation-protocol.md) — the **annotation rubric**. Hash-locked: `data/annotation/provenance_v2.json` records its SHA-256 as the instrument the 800 labels were produced under, and a test enforces the match. **Do not edit it without re-labelling.**

  *This bit me during the archive reorganisation: a bulk link-repointing pass edited this file and broke the hash, and the test caught it immediately. The file was reverted rather than the recorded hash updated — changing the hash would have claimed the annotator read a version that did not exist when they labelled. Its internal links therefore still point at pre-archive paths. That is deliberate: the instrument is frozen, and a stale link inside it is a smaller problem than a provenance record that lies.*
- [`../FINDINGS.md`](../FINDINGS.md) — what the study found, in plain language with the mathematics.

Everything else is in [`archive/`](archive/): the full decision log, the audit, the increment-by-increment execution record, and the individual contracts. Nothing was deleted — that material is the evidence that decisions preceded results, and it is what makes the study defensible. It is archived rather than removed because it is a *record*, not a reference.

---

## 1. The frozen decisions

Changing any of these changes what the study measured. Each is in `config.py`.

| Decision | Value | Why it is what it is |
|---|---|---|
| **Source** | Benzinga sub-corpus of FNSPID | The source file concatenates five-plus corpora with different languages and timestamp behaviour, including Russian-language `lenta.ru`. The domain filter is a **boundary, not hygiene**. |
| **Window** | 2010-01-01 … 2019-12-31, 2,516 sessions | Frozen on coverage evidence: no year fell below tolerance. Shortening it costs precision *and* strands drawn annotation items. |
| **Timestamps** | Date-only; a headline dated *d* maps to the **first session strictly after *d*** | No sub-corpus is both intraday-stamped and relevant, so same-day analysis is structurally suppressed — `inference.contemporaneous` raises. |
| **Scorers** | LM, VADER, FinBERT (`ProsusAI/finbert`, revision `4556d130…`, `max_length=64`, `batch_order=length_sorted`) | All three settings change the numbers and are recorded in the score cache's fingerprint. |
| **Aggregation** | Equal-weighted daily **mean**; dispersion `d_t` only when `n_t ≥ 5` | Median is the robustness variant, selected by passing `agg="median"`, never by editing config. |
| **Controls (frozen)** | previous return, range variance, **detrended** log volume | Fixed before any coefficient was examined. Adding one now is a specification search. |
| **HAC bandwidth** | `L = 5`, counted in **exchange sessions** | "One trading week" is only true under session counting. The row-counting alternative is computed and reported alongside. |
| **Effect yardstick** | 5 basis points per 1 SD | Order of magnitude of round-trip cost in a large ETF. **A measure of smallness, never a profitability threshold.** |
| **Multiplicity** | 1 unadjusted primary + exactly 14 secondary return tests, BH with BY alongside | Family membership is closed. Classification contrasts are a separate family and never join it. |
| **Annotation thresholds** | LM `[−1.00, 0.00]`, VADER `[−0.125, +0.225]` | Fitted on the 200 **calibration** items only and frozen before the evaluation set was scored. |

### Two things the code will refuse to do

- **Evaluate before thresholds are frozen.** `validate.evaluate_frozen` raises unless `config.VALIDATION_THRESHOLDS` matches the calibration record.
- **Estimate a tone coefficient before precision is recorded.** `run_all.py` writes `advance_precision.json` atomically first; if that write fails, no regression runs.

### How the conclusion rule works

Every Act 2 interval reports **two independent facts**, always together:

- **A** — does the interval exclude zero?
- **B** — does it lie inside ±5 bps?

Four combinations, four names. An earlier three-way rule was withdrawn because the categories overlapped. Comparisons are non-strict on endpoints rounded to 0.1 bps, and anything within 0.1 bps of a boundary is **flagged**, never silently resolved. The primary result is such a case.

---

## 2. Code map

```
config.py              every frozen decision; imported everywhere
preflight.py           what is installed, which artifacts exist, what produces them
rescore.py             the one expensive step (FinBERT), resumable
run_all.py             panel -> every table and figure, staged
attenuation_study.py   the measurement-error simulation (no project data)
hac_spacing_study.py   the HAC session-vs-row measurement

src/preflight.py       dependency inventory + artifact registry
src/data.py            news loading, mandatory domain filter, dedup, market data
src/scoring.py         three scorers behind one protocol; fingerprinted cache
src/align.py           timestamp mapping, daily aggregation, the panel, all lags/leads
src/inference.py       eligibility, primary spec, session-indexed HAC, families,
                       paired contrast, timing shift, RQ4 joint Wald
src/robustness.py      the five sensitivity exhibits
src/validate.py        Act 1: label ingestion, calibration, group bootstrap
src/annotate.py        the blind sample draw (made once)
src/publish.py         staged publication, run manifest, panel provenance
src/plots.py           one function per figure
```

## 3. Invariants — the things that will break silently if you remove them

Each exists because the failure it prevents is **invisible**, not because it is tidy.

**The panel must be the complete exchange calendar.** Row shifts are session shifts only if no row is missing. `inference._require_full_calendar_panel` detects a filtered panel without needing the calendar: on a complete frame `ret_lead{h}` is exactly `ret.shift(-h)`, and removing any interior row breaks that. *Feeding it `analysis_sample`'s output is the mistake it exists to catch.*

**Lags and leads are built once, before any exclusion.** Shifting inside a regression would lag over *retained* rows, so a Wednesday whose Tuesday was dropped would silently take Monday's return.

**Scores are fingerprinted.** Model revision, truncation length, batch size and batch order all change the numbers, so all four are recorded. A changed fingerprint invalidates that column rather than mixing two definitions. *Measured: batch composition moves scores by up to 4.2e-6 — immaterial to every reported digit, but real, which is why it is recorded rather than assumed away.*

**Sessions must be completely scored before aggregation.** Averaging a session's scored subset is within-day sampling and biases both `S_t` and `d_t` — a *different* measurement, not a noisier one.

**Nothing is published until the whole run succeeds.** Outputs stage; `run_manifest.json` is written last and is the commit point. A results directory without a current manifest is an incomplete run, not results.

**A reused panel is checked against the world.** `--skip-panel` verifies input digests, frozen settings and the score-cache generation. A schema check cannot see stale semantics — the columns are identical.

**The domain filter matches exact hosts or true subdomains.** Substring matching admitted `notbenzinga.com`.

**The annotation draw is made once.** Redrawing needs a dated entry recording why.

## 4. Making a change safely

1. **Check readiness.** `python preflight.py --stage analysis --strict`
2. **Make the change.** If it touches a fingerprinted setting, expect a rescore.
3. **Run the tests.** `pytest` — 546 passing, 0 skipped, is the baseline. Several tests exist specifically to fail when documentation drifts from the code, or when a document quotes a number its table does not contain.
4. **Rebuild and publish.** `python run_all.py`
5. **Verify.** `python -c "from src import publish; print(publish.verify_published())"`
6. **Record the decision** — what changed, why, and **what had already been seen**. That last part is what separates a prespecification from a post-hoc choice. Append to `archive/research-review-decision-log.md`.

### Reproducing from nothing

```bash
pip install -r requirements.txt
python preflight.py
python data/raw/download.py --assemble   # 5.7 GB stream, digest verified before publishing
python data/raw/download.py --dedup      # 1,412,524 -> 869,183
python data/raw/download.py --market     # SPY + ^VIX on the NYSE calendar
python rescore.py                        # ~3.1 h, resumable
python run_all.py                        # minutes
```

Two prerequisites no command can satisfy: the **Loughran–McDonald dictionary** (no stable URL; digest pinned in `config.py`) and the **annotation labels** (a person, and using a model would make the answer key an output of the system under test).

### What reproduction cannot check

- The 5.7 GB source file is not redistributed. A digest change is detectable, not repairable.
- `yfinance` serves live data; its loader has no regression test against it.
- FinBERT is not bit-reproducible across batch settings (4.2e-6), so a *resumed* pass differs from an uninterrupted one at that scale. The cache guarantee is over which rows carry which committed values, not the last few ulps.
- `test_checkpoints.py::test_hard_stop_around_commit_and_resumption[after_text]` is load-sensitive: two failures under heavy concurrent load, then zero across 52 quiet runs.

## 5. Known limitations, if you are extending this

- **One annotator, who is also the analyst.** No Cohen's kappa, no measured label error rate. A second annotator on the prepared 160-item subset is the single largest available improvement.
- **The rubric's framing favours FinBERT** — it asks "would this move the price?", which is FinBERT's training objective, while the lexicons measure textual valence. Not fixable by reframing; the opposite framing would favour the lexicons.
- **Market-level aggregation is a blunt test.** Company news moves companies. The single-name check could not fix this: the three most-covered tickers average 1.2 headlines a day and leave 44–56% of sessions empty. A powered version needs a denser per-ticker feed.
- **Two robustness exhibits are inapplicable** rather than reassuring: the dispersion floor excludes exactly one session, and the single-name check is underpowered. Absence of divergence there is not evidence of stability.
- **`config.AGG_CHOICES`** offers median aggregation, which moves the point estimate from −1.74 to −0.17 bps. Both are nulls; the mean was prespecified.
- **VADER's fingerprint records `version: unknown`** — the package exposes no version attribute. The lexicon size (7,506) identifies the word list; fixing the string would force a rescore to change no number.
- **PhraseBank is descoped.** `datasets==4.0.0` removed dataset scripts. It was only ever the fallback evaluation source, and Act 1 ran on the preferred one.

## 6. Scope boundaries

These were declared out of scope before the study ran. Moving one in is a change to the design, not an extension of it, and needs a dated line in `archive/research-review-decision-log.md`.

- **No model training or fine-tuning.** FinBERT is used as published.
- **No long documents.** 10-Ks and transcripts are a different measurement problem (length, boilerplate, section structure).
- **No cross-sectional panel.** One market-level series, tested against SPY.
- **No trading backtest.** A backtest turns an inference question into a specification search over costs, sizing and rebalancing rules, each of them p-hackable. Economic significance is delivered by the bps-per-1 SD versus transaction-cost comparison instead.

Parked, in rough order of value:

- **A second annotator** on the prepared 160-item subset (Cohen's kappa). The largest available improvement; needs another person.
- **Event-category breakdown of Act 1** — which headline types LM misses. Mechanical, ~20 minutes.
- **Half-day sessions.** `config.USE_ACTUAL_SESSION_CLOSE` (currently `False`) fixes the close at 16:00 ET even on early-close days. Using each session's actual close is strictly more accurate.
- **Intraday confirmation** at the headline-to-next-15-minutes horizon. Needs intraday SPY data.
- **Volume- or attention-weighted daily aggregation** instead of equal weighting.
- **An out-of-domain check** on EDGAR 8-K headlines.
