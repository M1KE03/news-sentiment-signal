# Deduplication lineage contract

Date: 2026-09-09. Increment: **R03a**. Audit finding: **A12** (with A11's missing article-group ID).

Status: **proposal**. No production code has changed and no artifact has been replaced. This document specifies what R03b implements and what R03d then verifies before the frozen corpus is rebuilt. It is written to be approved or rejected on its stated rules, in the pattern of the [timing contract](timing-contract.md) and the [scoring checkpoint contract](scoring-checkpoint-contract.md).

Related: [project audit](project-audit-2026-09-09.md) A11/A12 · [repair plan](audit-implementation-plan-2026-09-09.md) R03a–R03d · [data audit](data-audit-fnspid.md) §11 · [validation protocol](validation-protocol.md) §4.

---

## 1. What this contract covers

Four defects in `data.dedup` and one missing capability, all reproduced against the running code before being specified away (project convention 1).

| | Defect | Consequence |
|---|---|---|
| **D-1** | The exact-repeat window anchors on the first-ever occurrence of a text, not on the last kept one | Later clusters of exact repeats are attributed to the near-duplicate count, or survive entirely when `near_dupe=False` |
| **D-2** | The surviving representative is selected by an unspecified, unstable sort | Which row survives — and therefore which ticker tags and which raw text reach scoring — depends on upstream row order |
| **D-3** | Duplicate rows' ticker tags are discarded with the row | `audit.concentration_profile` reads surviving tags as "companies covered" |
| **D-4** | `ceil(2/(1-overlap))` evaluates to 21 at `overlap=0.90` | The quoted blocking exposure omits the documented 20-unique-token boundary |
| **D-5** | No lineage survives dedup; the corpus has no article-group identifier | [Validation protocol](validation-protocol.md) §4 assigns whole article groups to the calibration/evaluation parts, and there is no column to key that on — **R06a cannot be implemented against the current schema** |

## 2. Evidence

All figures below are measured on the real saved artifacts (`headlines_raw.parquet`, 1,412,524 rows; `headlines.parquet`, 869,205 rows), read-only. Nothing was written.

### 2.1 What is *not* wrong: the artifact is reproducible

Stated first, because an earlier probe in this increment suggested otherwise and was wrong. Calling the real `dedup()` on the real raw corpus reproduces the saved clean corpus **exactly** — 869,205 rows, 431,602 exact and 111,717 near drops, symmetric difference of **0** headline ids — and does so identically on a repeat run. The saved artifact is not corrupt and is not unreproducible from its pinned input.

The first probe reported a mismatch only because it sorted with `kind="stable"` while `dedup` uses pandas' default. That difference is itself D-2, but the conclusion "the pipeline does not reproduce its own artifact" was an artifact of the probe and is withdrawn.

### 2.2 D-2 — output depends on input order

Permuting the raw corpus (seed 20260909) and re-running the same `dedup`:

| | As stored | Permuted |
|---|---|---:|
| Rows out | 869,205 | 869,204 |
| Headline ids differing | — | **2,363** |
| Surviving headlines whose **retained ticker tags differ** | — | **93,552 of 868,023 (10.78%)** |

`dedup` sorts on `ts_utc` alone with pandas' default `kind="quicksort"`, which is not stable. In a **date-only corpus every headline on a date carries the identical `00:00 UTC` stamp**, so this is not a rare tie — every same-day duplicate group is a tie, and the representative is whichever row the sort happened to place first.

Reproduced on a three-row fixture: the same input *set* in six different input *orders* yields three different survivors, each keeping its own single ticker.

The corpus is therefore reproducible from the pinned raw file, but only accidentally — the property rests on the raw file's row order rather than on any stated rule. Pinning the input (R02) preserved the accident; it did not remove the dependence.

### 2.3 D-3 — the scale of ticker-tag loss

| | |
|---|---:|
| Exact same-stamp groups | 894,192 |
| — multi-row | 161,117 (18.0%) |
| Rows inside multi-row groups | 679,449 |
| Multi-row groups losing at least one tag | 155,906 (**96.8%**) |
| Distinct (group, ticker) pairs in the union | 663,074 |
| Retained by the single representative | 161,117 |
| **Destroyed** | **501,957 (75.7%)** |

Three quarters of the company-coverage evidence is discarded.

### 2.4 D-3's consequence for the universe justification — measured, and benign

The audit noted that this "weakens the current assurance of low concentration". It does not. Recomputing the published figures with tags unioned:

| Measure | Surviving tags (published) | Unioned tags |
|---|---:|---:|
| Distinct tickers | 5,707 | 6,235 |
| Top-10 share | 2.09% | **2.03%** |
| Effective names | 1,839 | **1,810** |

Concentration is marginally **lower** once the destroyed tags are restored, so the D4 universe decision is unaffected and, if anything, better supported. This is recorded because it is the kind of check whose result must be reported whichever way it comes out. **The defect is real and large; its effect on the specific claim that justified the universe is negligible.**

### 2.5 D-1 — the exact/near split is badly wrong

Re-anchoring the window on the last kept occurrence, measured against the true baseline:

| | Published | Corrected |
|---|---:|---:|
| Exact drops | 431,602 | **539,087** |
| Near drops | 111,717 | **4,245** |
| Final rows | 869,205 | 869,192 |

**96.2% of the reported near-duplicate count is actually exact duplication.** The corpus barely moves — 53 headlines dropped, 40 added, **net −13, or 0.011%** — but the published characterisation of *how* this corpus duplicates is wrong, and the data audit uses it.

With `near_dupe=False`, the 107,485 misattributed rows survive entirely, so the exact pass does not currently do what its own docstring says.

### 2.6 D-4 — the omitted boundary

`1 - 0.90` is `0.09999999999999998` in binary floating point, so `2/(1-overlap)` is `20.000000000000004` and `ceil` gives 21. The docstring says 20.

| | |
|---|---:|
| Rows with ≥ 21 unique tokens (counted) | 84,126 (5.956%) |
| Rows with ≥ 20 unique tokens (documented) | 98,620 (6.982%) |
| **Rows at exactly 20, omitted** | **14,494** |

A 20-unique-token pair differing by 2 tokens scores exactly `18/20 = 0.90`, clears the threshold, and can be missed by the blocking — confirmed directly. The quoted exposure understates itself by about one percentage point.

### 2.7 D-5 — `headline_id` is a group key, not a row key

`headline_id = sha(text_norm | ts_utc)`, so the same headline filed under three tickers on one date produces **one id for three rows**. In the raw corpus **518,332 rows share an id with another row**. After dedup ids are unique (869,205 of 869,205), which is what R04b's uniqueness gate relies on.

The consequence is that the existing id cannot double as a source-row identifier: lineage needs a new key.

## 3. Specification

### 3.1 A source-row identifier

Add `source_row_id`: the row's **0-based ordinal position in the verified raw corpus**, assigned at acquisition. This is well-defined precisely because R02 pins the source file by revision, SHA-256 and byte length, so position is a property of a verified artifact rather than of a particular read.

It is the only new identifier required, and it is what makes every rule below order-invariant.

### 3.2 Representative selection — a total order

The surviving row of a dedup cluster is the minimum under this order, applied in sequence until one row remains:

1. **Earliest `ts_utc`.** The existing intent, kept.
2. **Smallest `source_row_id`.**

Rule 2 is a total order on its own, so the composite is total and no tie can reach an unspecified sort. Because it references the raw file's position rather than the incoming frame's, **the output no longer depends on input order** — which is the property D-2 lacks.

The representative determines which raw `text` is scored, since members of an exact group share `text_norm` but may differ in punctuation or case. That choice is now made by a stated rule rather than by a sort.

### 3.3 Ticker tags are unioned

`tickers` on a surviving row becomes the **sorted union of the tags of every row in its dedup cluster**, deduplicated. Add `n_cluster_rows`, the number of raw rows the survivor represents.

`audit.concentration_profile` must then name its unit explicitly: it measures **companies tagged on the text**, not surviving tag rows. The published concentration figures are recomputed against the unioned column and §2.4's values replace them.

### 3.4 Dedup clusters and the exact window

A **dedup cluster** is a survivor together with every row eliminated against it. Two rules produce them, in order:

- **Exact:** identical `text_norm` within `window_days` **of the last kept occurrence**, not of the first-ever one (fixes D-1). Walking a text's occurrences in time order, each survivor re-anchors the window.
- **Near:** token-set overlap ≥ `overlap` against a kept row inside the window, unchanged in substance.

Because near-elimination is greedy against *kept* rows, two survivors are never near-duplicates of each other within the window — which is the property the clusters need, and it holds without computing a transitive closure over a non-transitive relation.

### 3.5 The blocking-exposure boundary

Replace `int(np.ceil(2 / (1 - overlap)))` with an integer-safe formulation that returns the smallest `L` with `(L - 2) / L >= overlap`, so `overlap=0.90` gives **20**, not 21. `share_above_signature_limit` is then quoted against the documented boundary. The reported exposure rises from 5.956% to 6.982%; the blocking itself does not change.

### 3.6 The lineage record

A new artifact, `data/interim/dedup_lineage.parquet`, one row per **raw** row (1,412,524):

| Column | Meaning |
|---|---|
| `source_row_id` | The raw row |
| `headline_id` | Its own content id (not unique in this table) |
| `cluster_id` | The representative's `headline_id` |
| `kept` | Whether this row survived |
| `eliminated_by` | `source_row_id` of the row it was eliminated against; null when `kept` |
| `relation` | `exact` \| `near` \| null when `kept` |

This makes the whole dedup auditable and the 38.5% rate recomputable without re-running the pass, which the data audit currently cannot do.

Clean-corpus schema additions: `source_row_id`, `cluster_id`, `n_cluster_rows`. `HEADLINE_COLUMNS` grows accordingly, and the R04b uniqueness gate continues to key on `headline_id`.

### 3.7 Article groups are a *different* relation, and are computed later

This is the part the audit collapsed into one item, and the distinction matters.

| | Dedup cluster | Article group |
|---|---|---|
| Purpose | Which rows were removed; tag union | Leakage control and the bootstrap resampling unit |
| Domain | The raw corpus | The **drawn annotation sample** |
| Windowed | Yes, 3 days | **No** |
| Members | Survivor + eliminated rows | Surviving headlines related to each other |

A story published 10 days apart survives dedup twice, because the window is 3 days. Those two survivors are different rows of the clean corpus but the **same article group** for annotation purposes, and [validation protocol](validation-protocol.md) §4 requires them assigned to the same part.

**Article groups are computed at R06a, within the drawn sample, by exact pairwise comparison — no blocking, no window.** The reasoning is the point:

- Leakage only matters **between the calibration and evaluation parts**, and both are subsets of the drawn sample. A group straddling sampled and unsampled items is irrelevant to that property.
- The sample is ~800 items, so all-pairs is ~320,000 comparisons: instantaneous, exact, and free of the blocking's one-token limitation. The corpus-wide alternative would need an unwindowed signature index over 869,205 rows to answer a question nothing asks.
- The same groups are the resampling unit for the paired bootstrap in validation §7 and for the M4 accuracy comparison, both of which also operate within the evaluation part.

So R03a creates no corpus-wide `group_id`, and R06a writes `group_id` into `data/annotation/split_assignment.csv` where the protocol already requires it. A11's "no article-group ID" is answered by that file, not by a column on 869,205 rows.

## 4. What this changes, quantified

| Quantity | Now | After R03b | Source |
|---|---:|---:|---|
| Corpus rows | 869,205 | 869,192 | §2.5 |
| Membership change | — | **93 ids (0.011%)** | §2.5 |
| Exact / near split | 431,602 / 111,717 | 539,087 / 4,245 | §2.5 |
| Dedup rate | 38.5% | 38.5% | unchanged |
| Distinct tickers | 5,707 | 6,235 | §2.4 |
| Top-10 share | 2.09% | 2.03% | §2.4 |
| Effective names | 1,839 | 1,810 | §2.4 |
| Blocking exposure | 5.956% | 6.982% | §2.6 |
| Order-invariant output | No (10.78% tag churn) | Yes | §2.2 |

The membership change is 93 headlines. **The corpus is not meaningfully different; its documented characterisation is.** That is the honest summary, and it is why R03d is a verification exercise rather than a re-freeze of the universe: D4 was frozen on coverage stability, and 93 rows across 2,516 sessions cannot move it. R03d confirms that rather than assuming it.

## 5. Acceptance tests for R03b

Offline, fixture-based, no network and no corpus rebuild.

1. **Order invariance.** The three-row same-timestamp fixture from §2.2, run under all six input permutations, yields the identical survivor, the identical unioned tags and the identical `cluster_id`.
2. **Re-anchored window.** The four-date fixture (days 0, 1, 10, 11) drops days 1 and 11 as **exact**, with `near_dupe=False` and with it, and `n_exact_dropped` counts both.
3. **Attribution.** On a fixture mixing exact repeats and genuine near-duplicates, `n_exact_dropped` and `n_near_dropped` each count only their own relation.
4. **Tag union.** A story filed under three tickers survives once with all three tags and `n_cluster_rows == 3`.
5. **Boundary.** `min_len_missable == 20` at `overlap=0.90` and `40` at `0.95`; a 20-unique-token row is counted in `share_above_signature_limit`.
6. **Lineage completeness.** Lineage rows == raw rows; every `kept=False` row names an `eliminated_by` that is itself kept; every `cluster_id` resolves to a kept row; the dedup rate recomputed from lineage equals `stats["dedup_rate"]`.
7. **Uniqueness preserved.** `headline_id` remains unique on the survivor frame, so the R04b gate is unaffected.
8. **Representative text.** Where cluster members share `text_norm` but differ in raw `text`, the scored text is the one the §3.2 order selects.

## 6. What R03d then does

Rebuild raw → clean through R02's verified path with R03b's rules; diff membership against the current artifact and confirm the 93-id change; recompute the census and coverage; recompute concentration on unioned tags; record the decision and replace the artifact. **Before any annotation sample is drawn and before any scoring.**

## 7. Decisions this contract asks for

1. **`source_row_id` as raw ordinal position.** The alternative is a content hash including the ticker field, which would be independent of file order but would not distinguish two genuinely identical raw rows. Position is available and cheap because the file is pinned.
2. **Representative = earliest timestamp, then smallest `source_row_id`.** The alternative tie-break — most ticker tags — is content-aware but no longer needed once tags are unioned, and it would still need a final deterministic tie-break.
3. **Article groups computed within the drawn sample, not corpus-wide** (§3.7). This is the one place where a cheaper design is also the more exact one, but it does mean the clean corpus carries no `group_id`.
4. **R03d replaces the frozen artifact for a 93-row change.** The alternative is to keep the current corpus and correct only the documentation. My recommendation is to replace: the order-dependence in §2.2 is a reproducibility defect independent of the row count, and the corpus should be the one the stated rules produce.
