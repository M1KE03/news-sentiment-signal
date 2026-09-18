# Reproducing this study from a clean directory

Date: 2026-09-10. Increment: **R15** (B26/B28).

This is the procedure a reader follows to rebuild every published number from a fresh clone. It is written as a manual procedure rather than a test because it needs a 5.7 GB download, a hand-acquired dictionary, network access and roughly three and a half hours of CPU — none of which belongs in a test suite. What *is* tested is that every command below exists and accepts the flags used here (`tests/test_reproduction.py`), which is the part that silently rots.

**What reproduces exactly, and what does not.** Act 2's tables and figures reproduce from the pinned inputs. Act 1 does not reproduce at all without independent human labels, which no command can create. Say so plainly rather than describing a workflow that stops halfway.

---

## 0. Check before starting

```bash
pip install -r requirements.txt
python preflight.py                 # what is installed, what artifacts exist
```

`preflight.py` reports declared-versus-installed for every pin and lists each artifact with the command that produces it. `--stage {unit,pilot,scoring,analysis,validation}` narrows it to one stage's prerequisites; `--strict` exits non-zero when anything is missing, which is the form for a check before an expensive step.

Two prerequisites no command in this repository can satisfy:

| Prerequisite | Why it needs a person |
|---|---|
| `data/raw/LoughranMcDonald_MasterDictionary.csv` | No stable download URL. Obtain from the Notre Dame SRAF site; the acquired release and its SHA-256 are pinned in `config.py` and recorded in [`lm-dictionary-provenance.md`](lm-dictionary-provenance.md). `preflight.py --stage scoring` verifies the digest. |
| `data/annotation/labels_primary.csv` | Independent human labels. See [§5](#5-act-1-needs-a-person). |

## 1. Acquire and assemble the corpus

```bash
python data/raw/download.py --assemble   # stream the pinned 5.7 GB FNSPID file
python data/raw/download.py --dedup      # raw -> analysis corpus, with lineage
python data/raw/download.py --census     # coverage, duplication, concentration
python data/raw/download.py --market     # SPY and ^VIX on the NYSE calendar
```

`--assemble` streams the file once (about 11.5 minutes on a normal connection), hashes it as it reads, and **checks the digest against the pinned revision before publishing anything**. It writes only the raw destination and refuses to be pointed at the clean one, so a documented command cannot replace the deduplicated corpus with the undeduplicated one under the same name.

Expected results: `headlines_raw.parquet` 1,412,524 rows; `headlines.parquet` **869,183** rows; dedup rate 38.5% (539,087 exact, 4,254 near); `market.parquet` 2,586 sessions of which 2,516 in window, with **zero** missing closes.

**One pin was wrong once and is worth knowing about.** `NEWS_FILE_SHA256` originally held HuggingFace's `xetHash` taken from the HTTP ETag, under a SHA-256 name. The file never changed; the recorded identity was a different hash function's output. Corrected at R03d.

## 2. Score the corpus

```bash
python preflight.py --stage scoring --strict   # refuse to start if anything is missing
python rescore.py --time-only                  # measure throughput first
python rescore.py --dry-run                    # scope and checkpoint location
python rescore.py                              # the full pass
```

This is the only expensive step. Measured on an 8-thread CPU: **113 headlines/s**, **3.13 hours** wall time for all three scorers over 869,183 headlines, producing a 97 MB cache at 112 B/row.

Interrupting is safe. The cache commits atomically every `--checkpoint-every` rows (default 50,000) and a resumed pass recomputes nothing. `--sessions N` scores the first `N` calendar days **whole**: bounding is by session because `S_t` is a within-day mean and `d_t` a within-day standard deviation, so a partial day is a different measurement rather than a smaller sample.

**Reproducibility caveat, stated precisely.** FinBERT's output is not bit-identical across batch compositions: scoring the same headlines with a different `batch_size` or `batch_order` moves individual scores by up to **4.2e-6**. Both settings are therefore part of the recorded fingerprint. The consequence is that a *resumed* pass is not bit-identical to an uninterrupted one, because the surviving unscored rows form different batches. At 4.2e-6 on a `[-1, 1]` score this is immaterial to every reported quantity — it survives a ~345-headline daily mean, standardization, and a figure printed to 0.1 bps — but the cache contract's guarantee is over *which rows carry which committed values*, not over the last few ulps of a float32.

## 3. Build the panel and run Act 2

```bash
python preflight.py --stage analysis --strict
python run_all.py
```

This builds `processed/daily_panel.parquet` and publishes 13 outputs to `report/tables/` and `figures/`.

**Nothing is published until the whole run succeeds.** Outputs are staged and moved into place only after the last one is produced; `run_manifest.json` is written last and is therefore the commit point. A results directory without a current manifest is an incomplete run, not a set of results.

```python
from src import publish
publish.verify_published()      # {'ok': True, 'problems': []}
```

`--skip-panel` reuses a saved panel and **refuses one whose recorded inputs or settings no longer match**. That is not a schema check: a panel built under different timing or scoring semantics has identical columns. Override deliberately with `--allow-stale-panel`, which prints what differs.

Expected results, from `run_manifest.json` of the 2026-09-10 run:

| Quantity | Value |
|---|---|
| Panel | 2,516 sessions, 0 missing market rows |
| Eligible | 2,453 (63 excluded: 61 detrend warm-up, 1 no lead, 1 zero-news) |
| Primary, FinBERT h=1 | **−1.74 bps per 1 SD, 95% [−4.90, +1.41]**, p = 0.279 |
| Advance half-widths | h1 = 3.67, h2 = 3.56 bps, published before any fit |
| Secondary family | 14 tests, smallest BH q = 0.946, BY q = 1.000 |
| RQ4 family | 4 joint Wald tests, smallest BH q = 0.087 |
| Paired contrast | δ = −2.43 bps, 95% [−6.88, +2.03] |

The primary result is a **flagged boundary case**: its lower endpoint, −4.9031, sits within 0.1 bps of the ±5 SESOI, so the informative-null classification turns on less than the reporting precision. Any quotation of the result carries that flag.

## 4. Verify the reproduction

```bash
pytest                                   # 525 tests, 0 skipped
python preflight.py --strict             # every pin matches, every artifact present
```

Then compare `report/tables/run_manifest.json` against the one published here. It records input digests, every frozen setting that changes a number, the scorer identities and generation id, the eligibility ledger, and a digest per output. Two runs that disagree either read different inputs or ran different code, and the manifest distinguishes those without anyone having to remember.

The manifest also records **whether the git tree was dirty**, because a commit hash alone does not identify the code that ran.

## 5. Act 1 needs a person

Act 1 cannot be reproduced from artifacts. It requires independent human labels, and using a model to produce them would make the ground truth an output of the same class of system under evaluation.

The sample is drawn and frozen: 800 headlines, 200 calibration / 600 evaluation, split by article group and stored as a file. **The draw is made once** (P25); re-running it requires a dated decision-log entry. The blind export carries `headline_id` and `text` only.

1. Label the 60-item pilot: `data/annotation/pilot_worksheet.csv`, instructions in `data/annotation/PILOT_README.md`, rubric v1 in [`validation-protocol.md`](../validation-protocol.md) §5.
2. Fill `data/annotation/provenance_pilot.json`. It is **rejected until every human declaration is replaced** — that refusal is deliberate: `validate_annotation_provenance` never manufactures a declaration.
3. Assess the pilot — rubric difficulties, class balance, throughput — before committing to the remaining 740.

**A blindness condition applies and is recorded.** The scoring pass completed on 2026-09-10, before any label was written, so scores exist for every drawn item. Validation protocol §5 permits this and requires recording which case obtained; the annotator's confirmation must assert that the cache was **not consulted**, rather than merely that no score was displayed.

## 6. What a clean-directory run cannot check

Stated so that nobody reads §4 as stronger than it is:

- **The 5.7 GB source file is not redistributed.** Reproduction depends on the pinned FNSPID revision remaining available. The digest check will detect a changed file; it cannot restore one.
- **`yfinance` returns live data.** A re-run months later fetches whatever the provider serves then. The market artifact's digest is recorded, so a difference is detectable, not preventable.
- **The `yfinance` loader has no regression test against live data.** It was absent from this environment until 2026-09-10 and produced a complete series on its first run; that is one observation, not a guarantee.
- **Scoring is not bit-reproducible across batch settings** — see §2. It is reproducible to well beyond every reported digit.
- **`test_checkpoints.py::test_hard_stop_around_commit_and_resumption[after_text]` is load-sensitive.** It kills a real subprocess at a commit boundary and failed twice under heavy concurrent load, then passed 0/30 isolated, 0/12 file-level and 0/10 full-suite runs. Recorded rather than dismissed.
