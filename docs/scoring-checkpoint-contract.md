# Scoring checkpoint contract — R01c

Date: 2026-09-09. **Status: proposed; implementation awaits approval for R01d.**

Related: [audit implementation plan](audit-implementation-plan-2026-09-09.md), [audit A02](project-audit-2026-09-09.md), [architecture approval gate](implementation-plan.md#5-architecture-approval-gates).

## Decision proposed

Store scores, input-text hashes and scorer provenance in **one Parquet file**, embedding versioned JSON in Arrow schema metadata. Publish a complete checkpoint with one same-directory temporary-file replacement. Keep `config.SCORES_PARQUET`, the public score return columns and `load_cache(path) -> (frame, metadata)` unchanged. A writer lock prevents concurrent scoring processes from overwriting one another's progress.

This is a local-file change within `src/scoring.py`; no database, service, generation directory or external locking dependency is required. The generation ID below identifies a snapshot; it is not a pointer to another file.

## Current failure and benefit

`_checkpoint` currently replaces `scores.parquet`, then writes `scores.parquet.meta.json` directly. A stop between writes can pair new measurements with old provenance. A stop during JSON writing can make a previously usable cache unreadable. R01a/R01b validate identities and invalidate stale rows but cannot make this two-file publication consistent.

Embedding provenance binds the values and their identity to the same replacement. Readers obtain one complete snapshot, or fail explicitly. Making both existing writes individually atomic would still leave a gap between them.

## File contract

| Path | Role |
|---|---|
| `scores.parquet` | Sole authoritative checkpoint: rows plus embedded provenance |
| `scores.parquet.<unique>.tmp` | Uncommitted candidate, created in the destination directory |
| `scores.parquet.lock` | Persistent advisory lock file; existence alone does not mean locked |
| `scores.parquet.meta.json` | Legacy artifact only; never read as provenance for the new format |

The cache table retains `headline_id`, `text_sha256` and `score_lm`, `score_vader`, `score_finbert`. Score columns are float32 with nulls for unfinished work. The return from `score_all` remains `headline_id` plus score columns; the text hash is cache-only. No raw headline text is added to the cache.

Preserve Arrow's existing pandas metadata and add the byte key `b"sentiment_signal.score_cache"`. Its value is UTF-8 JSON, serialized with sorted keys and `allow_nan=False`:

```json
{
  "format_version": 1,
  "generation_id": "719e6ca69a22476898550b19d42fca89",
  "n_rows": 2,
  "fingerprints": {
    "lm": {"scorer": "lm", "dict_version": "example", "wordlist_sha1": "example"}
  }
}
```

The example fingerprint is illustrative, not a real artifact identity. Actual fingerprints are those supplied and canonicalized by R01a/R01b.

- `format_version`: integer `1`, excluding booleans; unknown versions fail explicitly.
- `generation_id`: fresh UUID4 hexadecimal string per checkpoint, including a checkpoint containing only invalidations. It is an identifier, not a checksum or scientific result.
- `n_rows`: nonnegative integer equal to the actual full table's row count, including unfinished rows and rows outside the current request.
- `fingerprints`: dictionary keyed by configured scorer name. Every column containing a score must have a nonempty fingerprint. Entries for entirely null columns are allowed; absent entries for null columns are also allowed.
- Text hashes remain SHA-256 of the exact UTF-8 scorer input. Existing R01b uniqueness, non-null ID and populated-row text-hash checks continue to apply.

Format 1 requires the listed table columns. Missing required columns, invalid JSON, non-object metadata, invalid IDs/hashes or row-count mismatch cause `IncompatibleCache` with the path and reason. No file is silently treated as an empty cache because it is unreadable. Retain exception chaining for underlying I/O/Parquet errors. I/O permission failures must remain distinguishable from an incompatible format.

This format change does not claim to solve every fingerprint-completeness issue, nor authenticate against malicious file edits. Full score coverage and finite/range validation before analysis remain R04b.

## Writer and publication sequence

`score_all` acquires the writer lock **before loading the current cache** and holds it until it returns or raises. Locking only during replacement is insufficient: two processes could otherwise both load the old state and publish conflicting updates.

For each checkpoint:

1. Apply R01b's text/measurement invalidations and merge retained rows in memory. Build one Arrow table using `preserve_index=False`.
2. Construct the complete metadata object from that same snapshot, canonicalize fingerprints, and attach the JSON to its schema. Validate the table/metadata invariants before publication.
3. Create a uniquely named temporary file in the destination directory. Write the table through PyArrow, finish the Parquet footer, flush the file and call `os.fsync` on its writable file descriptor. Close all writer handles before replacement.
4. Validate that the candidate's footer is readable and its embedded generation, version, row count and schema match the intended snapshot. Full round-trip equality belongs in tests; a second full data read is not required at every checkpoint.
5. Call `os.replace(candidate, cache_path)`. **This is the commit point.** Never unlink the existing destination first, copy over it in place, or fall back to a cross-filesystem copy.
6. Report checkpoint success only after replacement succeeds. No JSON sidecar write follows the commit. Release the writer lock when the scoring call ends.

Cleanup removes only this call's own uncommitted temporary file on an ordinary failure, and must not obscure the original error. Process termination can bypass cleanup; orphan temporaries are ignored, never promoted on restart. No automatic directory-wide cleanup is introduced.

Keeping the candidate on the destination filesystem is essential to the replacement approach. Python documents the behavior of [`os.replace`](https://docs.python.org/3/library/os.html#os.replace). R01d must verify success/failure boundaries on this Windows environment; the documentation alone is not a Windows crash test.

## Writer lock

Use a private context manager in `src/scoring.py`, backed by a nonblocking OS lock on a persistent sibling lock file. On Windows use `msvcrt.locking` over a fixed one-byte region from offset zero; on POSIX use `fcntl.flock` with exclusive/nonblocking flags. Open without truncation, initialize the one-byte region if needed, and retain the open handle throughout the scoring call. Use the resolved cache path when deriving the lock path.

A second writer fails promptly with the cache path and “another scoring writer holds this cache”; it does not poll indefinitely. On ordinary exit unlock/close in `finally`; on process death the OS releases the held lock. A leftover lock file is harmless and is not deleted or treated as a stale lock by age/PID heuristics. OS lock acquisition failures other than contention must retain their real cause.

Readers do not acquire this lock. The lock coordinates cooperating project writers on a local filesystem; it does not cover external programs rewriting files, hard-link aliases or network/synchronization services. Unsupported platforms fail explicitly rather than silently omit locking. The platform-specific branch must be verified before claiming it supported.

## Reader contract and consumers

Open the committed Parquet file once and read it into one Arrow table. Extract the custom metadata from that same table and then convert it to pandas. Do not separately reopen the pathname for a metadata read and a data read: a replacement between those opens could recreate a mixed-generation observation.

`load_cache` performs structural and recorded-provenance validation without loading scorers, dictionaries, torch or transformer weights. `score_all` separately compares that provenance against the requested identities using R01b's preflight. Reading a complete snapshot is distinct from establishing that it is the desired measurement or complete enough for analysis.

Reader cases:

| Files/state | Required result |
|---|---|
| No committed parquet, no legacy sidecar | Empty initial cache; lock/temp files do not count as checkpoints |
| Valid format-1 parquet | Return its rows and embedded metadata |
| Valid format-1 parquet plus arbitrary old sidecar | Use embedded metadata only; leave the sidecar untouched |
| Existing parquet without embedded contract | Refuse as legacy; never attach a sidecar to it automatically |
| No parquet but a legacy sidecar exists | Refuse the incomplete legacy pair with rebuild guidance |
| Missing/invalid embedded metadata, corrupt file, unsupported version | Explicit failure; no fallback to sidecar, orphan temporary or guessed identity |

Minimal consumer change in R01d: route `run_all.build_panel`'s score read through `scoring.load_cache`, so it cannot bypass this format/provenance validation. Preserve the existing missing-input message. This is not the full panel-readiness gate; R04b still handles exact input coverage, required scorers and current identities before aggregation. `rescore.py` already delegates cache access to `score_all` and needs only accurate help/docstrings if they mention sidecars.

Ordinary `pd.read_parquet` still reads the numeric columns for inspection, but it is not a validated project ingestion path. A pandas read/write cycle can drop the custom metadata; such rewritten files must fail validation. Metadata preservation is supported by [Arrow's table schema metadata API](https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#pyarrow.Table.replace_schema_metadata).

## Interruption and recovery matrix

Let G0 be the previous committed file and G1 the candidate. If there was no G0, “G0 remains” means no committed file exists.

| Interruption/failure point | Visible checkpoint after restart | Action |
|---|---|---|
| Before candidate creation | G0 | Resume missing work using G0 |
| During serialization or before footer completion | G0; possibly incomplete temporary | Ignore temporary; recompute uncommitted batch |
| After write/flush, before validation | G0 plus uncommitted candidate | Ignore candidate |
| Candidate validation fails | G0 | Raise with reason; ordinary cleanup removes own candidate |
| Replacement fails, e.g. access denied/disk error | G0 | Raise; do not claim success or remove G0 |
| Process stopped at replacement boundary | G0 or G1, subject to filesystem replacement semantics | Validate whichever committed file is present; never mix data/provenance |
| Replacement succeeds, process stops before progress print | G1 | Resume G1, avoiding a repeat of its completed rows |
| Reader already holds old file when replacement occurs | Coherent G0 read, or Windows replacement refusal | Reader never combines it with G1 metadata |
| Corruption found on restart | No accepted checkpoint | Explicit refusal; recover from a verified copy or rebuild |

This is an application-interruption consistency guarantee on supported local filesystems. It is **not** a promise of power-loss durability, directory-entry persistence, disk-failure recovery or arbitrary network-filesystem behavior. Flushing file contents reduces one failure window but does not establish all those guarantees. Automatic rollback copies and multi-generation retention are outside this increment.

## Legacy handling

Do **not** automatically migrate even an apparently matching R01b parquet/JSON pair: the unresolved two-file crash window means their consistency cannot be proven from their mere presence. This applies even when text hashes exist. Existing files remain untouched, with an error directing the user to a new cache destination and fresh scoring.

If valuable legacy scores exist later, an explicit migration would be a separate reviewed task with independently verified provenance; it is not part of R01d. Current project inventory contains no real score cache, so the proposal does not require rerunning an existing corpus pass. Test fixtures are regenerated in temporary directories. No destructive cleanup is proposed.

Keep `meta_path` only if needed to recognize/report legacy sidecars; it must no longer determine a read or write source for the new format.

## R01d implementation and acceptance

Expected files: `src/scoring.py`, `tests/test_scoring.py`, a narrow `run_all.py` reader change and focused runner verification; progress documentation. No changes to scorer formulas, calibration, research decisions or corpus contents.

If necessary split R01d into reviewable sub-increments: first the embedded-format reader/writer and consumer change, then locking/failure-injection verification. Do not mark checkpoint consistency complete or launch a scoring pass until both are verified.

Required checks use fake scorers and temporary files:

1. Metadata and score rows round-trip together; pandas can still inspect unchanged public score columns. Existing R01a/R01b cache tests adapt to embedded metadata and continue to pass.
2. Inject failure during write, after footer completion, before replace and at replacement failure. Assert G0's bytes/generation remain valid, or absence on a first checkpoint. Include a valid-looking orphan candidate and confirm it is never loaded.
3. Stop immediately after successful replacement; restart sees G1, and resumed output matches uninterrupted scoring with exactly the expected missing-row calls.
4. Repeat interruption during measurement invalidation and changed-text handling. No retained out-of-request score may survive under an incompatible fingerprint or text identity.
5. Tampered/missing metadata, bad version, wrong row count, legacy pairs and malformed legacy sidecars follow the reader table. A legacy sidecar must never override a valid embedded identity.
6. Reader obtains data and metadata from one opened file/table. Exercise replacement/open-handle behavior on Windows, accepting a clear replacement refusal while retaining G0.
7. Separate processes contend for the same writer lock: the second fails before loading/scoring; after terminating the first, a new writer acquires the lock without deleting the lock file. Do not substitute a same-process mock for this check.
8. `run_all` refuses an invalid score artifact before aggregation; a valid read imports no torch and performs no scoring. Completeness checks remain separately identified as R04b.

Use assertions on bytes/generation, rows and scorer call counts, not merely “no temporary file remains.” Exception injection checks code paths; a controlled subprocess stop checks cleanup-independent behavior. Neither is evidence about sudden power failure.

## R01c verification and approval boundary

Reviewed all current score-cache readers and writers. A disposable one-row probe on installed PyArrow **25.0.1** attached JSON schema metadata, wrote/read Parquet, and verified both pandas and Arrow-to-pandas round trips preserved the frame. This establishes API feasibility in the current environment, not atomicity, lock behavior or compatibility with the declared PyArrow 21.0.0 environment. Those remain R01d/environment checks.

No production code or cache was changed by R01c. Approving this proposal authorizes implementing this single-file contract, the writer lock, legacy refusal and the narrow runner-reader update in R01d. It does not authorize full corpus scoring, dependency changes, Git writes or the remaining audit roadmap.
