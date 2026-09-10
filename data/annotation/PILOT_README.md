# Label the 60-headline pilot

Open [pilot_worksheet.csv](pilot_worksheet.csv) in Excel or another spreadsheet editor. You agreed on 2026-09-10 to label this pilot. All answer cells are blank; no machine predictions were added.

Fill these columns for each row:

| Column | What to enter |
|---|---|
| `label` | Exactly `positive`, `negative`, `neutral`, or `unusable` |
| `mixed` | `1` if the headline contains balanced positive/negative content; otherwise `0` |
| `hard` | `1` if the rubric leaves the label ambiguous; otherwise `0` |
| `notes` | Optional short note about a rubric question or difficulty |

**Scores now exist in the cache.** The full scoring pass finished on 2026-09-10, so `interim/scores.parquet` holds model output for every drawn item. The protocol permits labelling after that (§5) provided the annotator does not consult it. Nothing in this worksheet exposes a score, and your blindness confirmation in [provenance.md](provenance.md) must state that you did not look at the cache.

Read the headline alone and apply the [frozen rubric, §5](../../docs/validation-protocol.md#5-annotation-rubric). Particularly: reported share-price moves are **neutral** under this study's rubric; “costs fell” is favorable; a smaller-than-expected loss is favorable. Do not consult models, returns, article bodies or later events. Do not revise earlier answers merely to match later ones.

## After labelling: two files, not one

The ingestion code reads a **JSON** provenance record, not the markdown one.

1. Save the worksheet as `labels_pilot.csv` (UTF-8 CSV, same folder). Excel's default UTF-8 is fine; the BOM is handled.
2. Fill [`provenance_pilot.json`](provenance_pilot.json). Every `null` and `TO BE COMPLETED` field is a human declaration and the file is **rejected until all are replaced** - that refusal is deliberate, not a bug. The `_comment` keys explain each field and are ignored by the reader.
3. Optionally mirror the same facts into [`provenance.md`](provenance.md), which is the human-readable record.

Note `scores_existed` must be `true`: the scoring pass completed 2026-09-10. `blind_to_model_outputs` then asserts you did not open the cache, not merely that nothing was shown to you.

Keep `headline_id` and `text` unchanged and preserve row order. Save as UTF-8 CSV in the same file. Use `unusable` for text that cannot be evaluated, not an empty answer. Record session start/end and rubric questions in [provenance.md](provenance.md), and confirm whether you saw any model predictions. Do not claim a session or blindness confirmation before it occurred.

**Stop after these 60.** We will check rubric difficulties, class balance and time per item before proceeding with the remaining calibration items or evaluation. The full 800-item sample includes evaluation headlines; leave those untouched until the rubric and calibration procedure are ready. Agreeing to this pilot does not commit you to labeling all 800.

## How this worksheet was prepared

No draw or split was repeated. It filters the existing shuffled `to_label_primary.csv` to the IDs in `pilot_items.csv`, preserving their presentation order. All 60 IDs are unique and belong to calibration according to the stored split. The six worksheet columns expose only the original blind inputs and empty response fields.

Source file SHA-256 values at preparation (2026-09-10):

| File | SHA-256 |
|---|---|
| `pilot_items.csv` | `4662fbdf4d657b64148a65b943792cba367b7a50d590dc232a284fdfcf3e70a7` |
| `to_label_primary.csv` | `b796819ded51945647e4574b58ae75ca889fe15631b716c9b9f559c06edc6681` |
| `split_assignment.csv` | `2505743700ea85ba7081c0161d689e1482acc1e6ef0beb7d4a0bda5efa1ce0c0` |

After labeling, the ingestion step will validate answers by ID and export the protocol's label-only schema; this worksheet is not yet a completed `labels_primary.csv`.
