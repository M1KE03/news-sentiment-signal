# Annotation provenance

Sample drawn 2026-09-09 by `src/annotate.build`. Fields marked TO BE COMPLETED are filled in at labelling time by the annotator, not reconstructed afterwards.

| Field | Content |
|---|---|
| Rubric version | `v1`, `docs/validation-protocol.md` sha256 `b124df2d20b8991b...` |
| Annotator(s) | **TO BE COMPLETED** - role not name; state whether independent of the analyst |
| Dates | **TO BE COMPLETED** - start and end of each session |
| Blindness | Export contains `headline_id` and `text` only. Scores in cache at draw time: **no**. Annotator confirmation **TO BE COMPLETED** |
| Instrument | `to_label_primary.csv`, presentation-order seed `20260831` |
| Deviations | **TO BE COMPLETED** - every rubric question that arose and how it was resolved |

## The draw

- Frame: 869,183 deduplicated headlines (`interim/headlines.parquet`)
- Drawn: **800** across 10 strata (year x ticker-tag presence), proportional allocation, seed `20260830`
- Article groups: 799, of which 1 hold more than one item (largest 2)
- Calibration **200** / evaluation **600**, assigned by whole group
- Pilot: 60 calibration items (`pilot_items.csv`)
- Second annotator: 160 items (`to_label_second.csv`)

## Realised stratum counts

| Stratum | Drawn |
|---|---:|
| 2010|tagged | 48 |
| 2011|tagged | 94 |
| 2012|tagged | 92 |
| 2013|tagged | 79 |
| 2014|tagged | 84 |
| 2015|tagged | 84 |
| 2016|tagged | 87 |
| 2017|tagged | 70 |
| 2018|tagged | 72 |
| 2019|tagged | 90 |

### Strata that collapsed to one level

- **ticker-tag presence**: every headline in this corpus carries at least one tag (0.00% untagged), so this dimension has one level
- **source/publisher**: the universe is a single publisher by construction (the Benzinga sub-corpus, mandatory domain filter)

These are recorded because a stratum with one level stratifies nothing, and a reader counting dimensions would otherwise assume a balance that was never at stake.

## What must not happen to these files

- The evaluation part is touched **once**, for the reported numbers. Thresholds are fitted on calibration only.
- `split_assignment.csv` is the authority for the separation. It is not recomputed from a seed at analysis time: a change to the grouping rule would silently move items across the boundary.
- The draw is made once. Repeating it requires a dated entry in the decision log with the reason (P25).
