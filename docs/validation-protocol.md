# Validation protocol (Act 1)

Date: 2026-09-09. Increment: **B01**. Amended 2026-09-09 (**R05**, audit finding A10) — see §12.

Status: **specification**. No labels have been collected and no evaluation has been run. This document defines the procedure to be executed at B11 (sample preparation), B15 (paired uncertainty implementation) and B24 (results). Source-specific details that depend on the news collection are marked **[B03/B04]** and are resolved by the data audit, not here.

Related: [implementation plan](implementation-plan.md) · [decision log](research-review-decision-log.md) (P02, P03, P04, P05, P17) · [scope draft](protocol-revision-draft.md).

---

## 1. What Act 1 estimates

The estimand is the **difference in macro-F1 between FinBERT and Loughran–McDonald on headlines drawn from the news collection this study actually analyses**, together with an interval for that difference.

```text
Delta = macroF1(FinBERT) - macroF1(LM)        [primary]
```

Two things follow from writing the estimand this way, and both constrain everything below.

1. The uncertainty statement must be about `Delta`. A McNemar test on paired correctness is a statement about an **accuracy** difference, not about `Delta`; it is reported, and labelled, separately (P03).
2. The evaluation examples must be independent of the compared checkpoints' training data. Financial PhraseBank is not, for `ProsusAI/finbert` — the [model card](https://huggingface.co/ProsusAI/finbert) identifies PhraseBank as fine-tuning data, and re-splitting it does not undo the checkpoint's prior exposure (P02).

Secondary, each labelled as such: `macroF1(FinBERT) - macroF1(VADER)`, `macroF1(LM) - macroF1(VADER)`, per-class F1, accuracy, and the confusion matrices.

**Not estimated, and not to be reported:** any decomposition of the FinBERT–LM gap into a "vocabulary" part and a "context" part. The three scorers differ in training exposure, architecture and scoring rule simultaneously; the arithmetic difference of three numbers is not an identified attribution (P05). Describe the three approaches and their measured differences, and stop there.

## 2. Primary evaluation source: annotated headlines from this study's own collection

The primary evaluation set is a sample of headlines from the selected news collection, labelled by humans who cannot see any model's output.

This is preferred over any public benchmark for two reasons, **stated precisely** (amended 2026-09-09, M5). It has verifiable **label** independence, and it measures classification quality on the exact text distribution that Act 2 aggregates — headline register, length, and subject mix — rather than on a different corpus.

**What "independence" does and does not mean here.** The earlier wording claimed "verifiable independence from the checkpoints", which overclaims and is withdrawn (A10). What new human annotation establishes is that **the labels** were produced by people, after the models were trained, and therefore cannot have been in any model's training data. That is the property the comparison needs, and it is genuinely verifiable from the annotation provenance in §6.

What it does **not** establish is that the **headline text** was unseen. These are historical headlines from 2010–2019, published on the open web, and no scorer's pretraining corpus is auditable at that granularity. Prior exposure to the text cannot be excluded for any of the three scorers, FinBERT included. This is recorded as a limitation of Act 1 as a whole, applying symmetrically, and is stated in the results section rather than only in the limitations — it bounds how far any of these numbers generalize to genuinely unseen text.

The two are different claims and the report keeps them apart: **labels are independent; text exposure is unknown.**

**Fallback, in order.** If independent annotation cannot be obtained:

1. An external benchmark for which independence from the exact checkpoint's training data can be **established from documentation**, not assumed. Record the evidence.
2. PhraseBank, retained only as a **supplementary** exhibit carrying an explicit contamination statement, with the FinBERT number described as **an optimistically biased estimate with no guarantee that it bounds generalization**, rather than as a measurement of generalization. (Amended 2026-09-09, M5: the earlier phrase "upper bound of unknown tightness" is withdrawn. `ProsusAI/finbert`'s own model card names PhraseBank as fine-tuning data, so the score is expected to be optimistic — but contamination does not make a number a mathematical bound on anything, and a contaminated score can in principle sit below a model's performance on other text.)

Falling back is a reportable limitation, not a silent substitution. If Act 1 rests on option 2, the report says so in the results section, not only in the limitations.

**Prohibited under every option:** using any model's predictions — FinBERT's included — as ground truth, or describing model-generated labels as human judgements.

## 3. Sampling

**Frame.** All headlines surviving deduplication within the locked window, i.e. the rows of `interim/headlines.parquet` **[B03/B04]**.

**Strata.** Draw proportionally to the frame within cells of:

- **calendar period** — one stratum per year, or per half-year if the window is short, so the label set is not concentrated in the most heavily covered years;
- **source/publisher**, if the collection carries one **[B03]**;
- **ticker-tag presence** (tagged / untagged), if tags exist, since untagged headlines are often macro rather than single-name and are the ones the lexicons handle worst.

Proportional allocation, not equal allocation: the evaluation set should look like the corpus being measured. Record realised cell counts.

**Draw.** Simple random sample without replacement inside each cell, `numpy.random.default_rng(config.SEED)`, seed `20260830`. Record the drawn `headline_id`s. The draw is made **once**; if it is ever repeated, the reason and date go in the decision log (P25).

**Size.** Target **800 headlines: 200 calibration + 600 evaluation.** The split is by unique text/article group (§4), not by row.

The 600 figure is a starting target from the illustrative calculation below, to be revised once the pilot in §8 measures the actual disagreement rate. It is not a power guarantee.

> *Illustrative only.* For a paired difference, the bootstrap interval's width is driven by the rate at which the two classifiers disagree, not by the number of examples alone. If FinBERT and LM disagree on a fraction `p_d` of evaluation items and the accuracy difference among those is `delta`, the standard error of the paired accuracy difference is roughly `sqrt(p_d / n)`. At `n = 600` and `p_d = 0.30`, that is about 2.2 points — so a true gap of ~6 points would be resolvable and a ~2-point gap would not. Macro-F1's standard error is not identical to this and is affected by class balance, which is why §7 obtains the interval by bootstrap rather than from a formula. The pilot in §8 replaces `p_d` with a measured value before the full sample is annotated.

## 4. Calibration / evaluation separation

Two disjoint parts, assigned **before any labelling begins** and never reassigned:

| Part | Size | Sole permitted use |
|---|---|---|
| **Calibration** | 200 | Fitting the LM and VADER neutral-band thresholds; refining rubric wording; the pilot in §8 |
| **Evaluation** | 600 | Every reported number in Act 1. Touched once. |

**The assignment unit is the article group, not the row.** Two headlines are in the same group when they share a normalized text (`text_norm`) or are near-duplicates of each other under the same rule the pipeline uses (token-set overlap ≥ 0.9 within a 3-day window). A whole group goes to one part. Otherwise a syndicated headline can be threshold-fitted on Monday and evaluated on Tuesday, which is leakage even though the row ids differ.

The split is stratified on the §3 strata and seeded. It is stored as a file — `data/annotation/split_assignment.csv`, columns `headline_id`, `group_id`, `part` — and that file is the authority. Recomputing the split from a seed at analysis time is not sufficient: a change to the grouping rule would silently move items across the boundary.

**Threshold fitting.** LM and VADER need a neutral band; FinBERT does not. Bands are fitted on calibration labels only, by maximising macro-F1 over a grid on `[-1, 1]`, and are then **frozen and recorded in `config.py`** before the evaluation part is scored. FinBERT's predictions come from its own class argmax (B12) and involve no fitting at all — which is itself an asymmetry to state in the report: the lexicons receive a tuning step that the transformer does not.

**If the calibration labels turn out to be unusable** (e.g. the rubric changes materially after the pilot), the calibration part is re-labelled. The evaluation part is not, and is not inspected in the meantime.

## 5. Annotation rubric

Version `v1`, dated 2026-09-09. The rubric is versioned because a mid-annotation change to it is a protocol change and must be logged with the date and the number of items already labelled.

### The question the annotator answers

> Read the headline alone. For the company or market it concerns, does this headline read as **good news**, **bad news**, or **neither**, for the value of the relevant equity?

Judged **from the headline text only**. Not from the article, not from outside knowledge of what happened next, not from what the market did that day.

### Labels

| Label | Use when |
|---|---|
| `positive` | The headline's content would, on its face, be received as favourable — beats, raises, wins, upgrades, better-than-expected outcomes, resolved problems. |
| `negative` | Unfavourable on its face — misses, cuts, losses, downgrades, investigations, worse-than-expected outcomes, new problems. |
| `neutral` | No directional valence: scheduling and administrative notices, factual descriptions with no evaluative content, headlines whose direction genuinely cannot be read from the text. |
| `unusable` | Not a financial-news headline at all, truncated to unintelligibility, non-English, or a bare ticker/number string. **Excluded from the evaluation and reported as a count.** |

### Decision rules

These exist so that two annotators resolve the same hard cases the same way. They are rules, not suggestions.

1. **The comparative is the signal.** *"Profit warning smaller than feared"* is `positive`: the operative content is the beat against expectation, not the word "warning". Likewise *"loss narrows"* → `positive`, *"growth slows"* → `negative`.
2. **Direction attaches to the subject, not the word.** *"Costs fell sharply"* is `positive` — a fall in costs is good. *"Provisions rose"* is `negative`. Score what moved and in which direction, never the sentiment of the noun in isolation.
3. **Reported price moves are `neutral`.** *"Shares fall 4%"* describes the market's own reaction; it is not news about the company. This is a deliberate rule with a cost — it will disagree with all three scorers — and it exists because the alternative makes the label set partly a function of returns, which is the outcome variable in Act 2.
4. **Analyst actions take the analyst's direction.** *"Upgraded to buy"* → `positive`; *"price target cut"* → `negative`.
5. **Guidance takes the guidance's direction**, judged against the expectation named in the headline if one is named, otherwise against the prior level.
6. **Mixed headlines take the dominant clause.** If the two halves are genuinely balanced — *"revenue beats, margins miss"* — label `neutral` and tick the `mixed` flag.
7. **Questions and speculation are `neutral`** unless the headline asserts the answer. *"Is X in trouble?"* → `neutral`. *"X is in trouble, says regulator"* → `negative`.
8. **Macro headlines take the direction for equities**, not for the indicator. *"Jobless claims fall"* → `positive` for equities.
9. **Do not resolve ambiguity by guessing.** If rules 1–8 do not decide it, the answer is `neutral` plus the `hard` flag. `neutral` is a real category here, not a bin for indecision — but the flag lets §6 measure how often it was used that way.
10. **One headline, one pass.** Do not revisit earlier labels after seeing later ones, and do not re-read a batch to make it consistent.

### Blindness requirements

- The annotation file contains `headline_id` and `text` and nothing else. No scores, no model predictions, no dates, no tickers, no returns.
- Presentation order is shuffled with a recorded seed, so that neither time order nor source clusters cue the annotator.
- Annotation happens **before** any scorer is run on the evaluation part — or, if scores already exist in the cache, the annotator must not have access to them. Record which was the case.

## 6. Annotators, agreement, and label provenance

**Primary annotator:** one person labels all 800. **[B11 — the user, or a named annotator.]**

**Second annotator:** an independent person labels a random **20% subset (160 items)**, drawn with the recorded seed, without seeing the first annotator's labels. Report:

- Cohen's kappa on the three real classes, with a bootstrap interval;
- the raw agreement rate;
- the full 3×3 annotator-by-annotator confusion matrix, not just the summary statistic.

Disagreements are **not** silently reconciled. Either keep the primary annotator's labels and report kappa as a measurement-quality bound, or adjudicate by a stated rule agreed in advance — and say which was done. Reconciling after seeing which choice helps a model is not available.

If a second annotator cannot be obtained, say so plainly in the report. Single-annotator labels with an unmeasured error rate are a real limitation and a stated one is defensible; an unmentioned one is not.

**Provenance record** — `data/annotation/provenance.md`, written at labelling time, not reconstructed afterwards:

| Field | Content |
|---|---|
| Rubric version | `v1`, plus the date and sha256 of this file at the time of labelling |
| Annotator(s) | Role, not name, if the repository is public; whether independent of the analyst |
| Dates | Start and end of each annotation session |
| Blindness | Confirmation that no model output was visible, and whether scores existed at the time |
| Instrument | The exact file/tool used, and the presentation-order seed |
| Deviations | Every rubric question that arose and how it was resolved |

### Files

```text
data/annotation/
  sample_frame.csv        headline_id, group_id, stratum, part          (B11, from the seeded draw)
  split_assignment.csv    headline_id, group_id, part                    (B11, the authority for §4)
  to_label_primary.csv    headline_id, text                              (blind: these two columns only)
  to_label_second.csv     headline_id, text                              (the 20% subset)
  labels_primary.csv      headline_id, label, mixed, hard, notes
  labels_second.csv       headline_id, label, mixed, hard, notes
  provenance.md
```

These are small and are **tracked in Git** — unlike everything under `data/raw|interim|processed`. The labels are the study's own experimental data and the most expensive artifact in the project to reproduce.

## 7. Metrics and uncertainty

Computed on the **evaluation part only**, excluding `unusable`, over the three classes `{negative, neutral, positive}`.

**Reported per scorer:** macro-F1 (headline); per-class precision, recall, F1, support; accuracy; the 3×3 confusion matrix. Class balance of the label set is reported alongside, because macro-F1's behaviour under imbalance is what makes it the right headline metric and also what makes its variance awkward to write in closed form.

**Primary interval — paired bootstrap on `Delta`.**

```text
resample evaluation ARTICLE GROUPS with replacement, B = 10,000, seed 20260830
  for each resample: recompute macroF1(FinBERT) and macroF1(LM) on the SAME resampled items
  record Delta_b = macroF1(FinBERT)_b - macroF1(LM)_b
report the 2.5th and 97.5th percentiles of {Delta_b}
```

Three properties of that procedure are the point of it:

- **Both models are recomputed on the same resampled items**, so the pairing — the fact that the two scorers saw identical sentences — is preserved, and the interval is for the difference rather than for two independent quantities.
- **The resampling unit is the article group**, not the row, matching §4. Near-duplicate headlines are not independent observations, and treating them as such would make the interval too narrow.
- **Macro-F1 is recomputed inside each resample.** It is not an average of per-item scores, so it cannot be bootstrapped by resampling a vector of per-item values.

Report the interval, the point estimate, and B. If the interval contains zero, that is the result; it is not a reason to look for a subset where it does not.

**Accuracy comparison** (amended 2026-09-09, M4). Accuracy is compared two ways, and the **primary** of the two is the one whose assumptions this design actually satisfies.

*Primary — paired group bootstrap on the accuracy difference.* Identical machinery to the macro-F1 interval above: resample evaluation **article groups** with replacement, `B = 10,000`, `seed = 20260830`, recompute both scorers' accuracy on the same resampled items, and report the 2.5th and 97.5th percentiles of `accuracy(FinBERT)_b − accuracy(LM)_b`. This respects group dependence for the same reason the macro-F1 interval does, and it needs no new theory.

*Supplementary — exact McNemar*, on the paired correctness table, labelled *"paired accuracy comparison"*. The `b` and `c` disagreement cells are reported with the p-value, since the p-value alone hides how much the two classifiers actually differ.

**Why McNemar is supplementary and not primary.** Exact McNemar treats the discordant pairs as independent Bernoulli trials. §4 assigns whole **article groups** to a part precisely because near-duplicate headlines are not independent, so a group contributing several evaluation items violates that assumption in the direction that matters: the p-value is too small. Pairing across scorers is not the issue — that part McNemar handles correctly — the issue is dependence *within* the item set. The earlier text presented McNemar without this qualification (A10).

Three things are therefore reported beside it: **the number of evaluation article groups**, **the group-size distribution** (how many groups contribute 1, 2, 3+ evaluation items), and the explicit statement that **the McNemar p-value is valid only under item independence, which this design does not assert**. If the realised evaluation set turns out to contain one item per group, that fact is reported and McNemar's assumption is satisfied — but that is an outcome to be checked, not assumed in advance.

**Secondary contrasts** (FinBERT−VADER, LM−VADER) use the identical bootstrap and are labelled secondary.

**Their multiplicity treatment is specified here** (amended 2026-09-09, M3). The earlier text delegated it to "the secondary family defined in the inference protocol (B02)" — but that family enumerates exactly 14 (scorer, horizon) **return** tests and never contained a classification contrast (A10). The delegation pointed at nothing, and [inference protocol](inference-protocol.md) §6 now closes that family explicitly.

The treatment: the two secondary contrasts are reported as **pointwise paired-bootstrap intervals, labelled descriptive-secondary, with no p-value and no multiplicity correction.** Three reasons, fixed here before any label exists:

- There are exactly **two** of them, and both are deterministic functions of the same three macro-F1 values as the primary contrast. They are near-collinear with it and with each other; a correction across two such quantities adjusts almost nothing while implying a family structure the design does not have.
- The estimand of Act 1 is a **difference with an interval**, not a decision at a level. Nothing here is thresholded, so there is no error rate to control.
- Act 1 and Act 2 are different estimands on different samples with different resampling units. Pooling them into one correction family would be a category error in either direction.

What this forbids: no significance claim is made for either secondary contrast, and neither may be promoted to a headline result after inspection. If a future increment wants a *tested* classification family, it must define one — membership, level and correction — before the labels are seen, and record it here.

## 8. Pilot before the full annotation

Before committing to 800 labels, label **60 calibration items** and check:

1. **Rubric adequacy** — how many needed the `hard` flag, and which rules were the source of the trouble. If more than ~20% are `hard`, the rubric is underspecified and gets a `v2` before the rest is labelled.
2. **Class balance** — if one class is under ~10%, macro-F1 will be dominated by noise in that class. Consider whether the target size needs to rise. Do **not** rebalance by oversampling: the evaluation set must reflect the corpus.
3. **Disagreement rate `p_d`** between FinBERT and LM on those 60 — this replaces the assumed 0.30 in §3 and gives a measured basis for the final evaluation size.
4. **Throughput** — items per hour, which is what makes the annotation workload a schedulable quantity rather than an estimate.

The pilot uses calibration items only, and its results may change §3's size and §5's wording. They may not change the evaluation set's membership.

## 9. PhraseBank, if retained as supplementary

Only under §2's fallback, and then:

- Loaded through a supported, pinned artifact — `datasets==4.0.0` removed dataset scripts and `trust_remote_code`, so the current `src/validate.py` call does not work as pinned (B10).
- The four agreement subsets share sentences. They are **overlapping sensitivity analyses**, not four independent replications, and calibration/evaluation membership is assigned once by unique sentence across all of them, exactly as in §4 (P04).
- Every FinBERT number carries the contamination statement inline.

## 10. What this protocol forbids

- Fitting any threshold, or choosing any rubric rule, on evaluation items.
- Looking at evaluation results before the thresholds are frozen and recorded.
- Using model output as ground truth, or as a tie-breaker between annotators.
- Reporting a McNemar p-value as the uncertainty of the macro-F1 difference.
- Attributing parts of the classifier gap to vocabulary versus context.
- Treating rows as independent when they belong to one article group.
- Re-drawing the sample, re-splitting, or re-labelling the evaluation part without a dated decision-log entry stating what had already been seen.

## 11. Open items for later increments

| Item | Resolved at |
|---|---|
| The news collection, its window, and whether it carries source/publisher fields | B03, B04 |
| Whether the collection's headlines are English-only and single-name-dominated | B03 |
| Named annotator(s), and whether a second is available | B11 |
| Final evaluation size, after the §8 pilot measures `p_d` and throughput | B11 |
| FinBERT class-probability interface | B12 |
| Implementation of the paired bootstrap and group-aware resampling | B15 / R06b |
| Implementation of the group-aware accuracy comparison (§7, M4) | R06b |
| Whether the PhraseBank fallback is needed at all | B10, B24 |
| Realised group-size distribution of the evaluation set, and whether McNemar's independence assumption holds on it | R06a, reported at B24 |

---

## 12. Amendment record

This document is a frozen specification, so every change to it is listed here with its date, its reason and what it replaced. Amendments correct defects in the specification; the estimand, the primary contrast, the sampling design, the calibration/evaluation separation and the rubric are untouched.

**2026-09-09 — R05, from [project audit](project-audit-2026-09-09.md) A10.** No labels had been collected and no evaluation had been run at the time of amendment. That is why these corrections are legitimate rather than post hoc.

| ID | § | Change | Replaced |
|---|---|---|---|
| M3 | 7 | The two secondary contrasts are specified here as descriptive pointwise intervals with no correction, with the reasoning fixed in advance | A delegation to "the secondary family defined in the inference protocol", which enumerates 14 return tests and never contained a classifier contrast |
| M4 | 7 | Paired **group** bootstrap on the accuracy difference becomes primary; exact McNemar demoted to supplementary, with group counts, group-size distribution and an explicit validity condition printed beside it | Exact item-level McNemar presented without noting that §4's own grouping rule contradicts its independence assumption |
| M5 | 2 | Independence narrowed to **label** independence, with text exposure recorded as unknown for all three scorers symmetrically; PhraseBank's number described as optimistically biased rather than as a bound | "Verifiable independence from the checkpoints", and "an upper bound of unknown tightness" |

M1, M2, M6 and the deferred M7 amend the [inference protocol](inference-protocol.md) and are recorded there.
