"""Staged output publication and the run manifest (R14, audit A15).

Audit A15: *"Outputs are written sequentially into shared paths, so a failed run
can leave a mixture of old and new tables."* A run that dies after Table 3 left
Tables 4 and 5 from an earlier configuration sitting beside it, with nothing on
disk saying so. Every file looked equally current.

Three rules fix that, and they are the same three the scoring cache already
follows:

1. **Nothing is published until everything succeeds.** Outputs are written to a
   staging directory and moved into place only after the last one is produced.
2. **The manifest is the commit point.** It is written last, after every file is
   in place. A results directory whose manifest is missing, or whose manifest
   does not list a file, is an incomplete run and says so.
3. **A run owns what it published.** The previous manifest records which files
   this runner produced, so an output that a later configuration stops producing
   is removed rather than left to look current. Files the runner never claimed --
   a figure from a notebook, say -- are not touched.

Moving several files cannot be made atomic on Windows, so the guarantee offered
is precisely the one the manifest supports: **the manifest is present and
consistent only if every promised output was produced by the same run.**
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import config

MANIFEST_NAME = "run_manifest.json"
MANIFEST_VERSION = 1


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """Digest a file without reading it all into memory (scores.parquet is ~97 MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    """The commit the run was made from, when there is one. Never fabricated."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def git_dirty() -> bool | None:
    """Whether the working tree had uncommitted changes. `None` if unknown.

    Recorded because a commit hash alone does not identify the code that ran
    when the tree is dirty, and claiming it does would be false provenance.
    """
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=config.ROOT,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


def environment() -> dict:
    """What ran, and where. Enough to explain a number that will not reproduce."""
    from src import preflight

    deps = preflight.dependency_report()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "packages": {r["package"]: r["installed"] for _, r in deps.iterrows()},
        "dependency_drift": [r["package"] for _, r in deps.iterrows()
                             if r["status"] != "ok"],
    }


def input_digests(paths: dict[str, Path] | None = None) -> dict:
    """Digest every artifact the results were computed from.

    This is what makes a result checkable later: two runs that disagree either
    read different inputs or ran different code, and the manifest distinguishes
    those cases without anyone having to remember.
    """
    if paths is None:
        paths = {
            "headlines": config.HEADLINES_PARQUET,
            "scores": config.SCORES_PARQUET,
            "market": config.MARKET_PARQUET,
            "panel": config.PANEL_PARQUET,
        }
    out = {}
    for key, path in paths.items():
        path = Path(path)
        out[key] = ({"path": str(path), "present": False} if not path.exists() else
                    {"path": str(path), "present": True,
                     "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return out


def frozen_settings() -> dict:
    """Every prespecified choice that changes a reported number.

    Recorded per run so that a table cannot be attributed to a specification it
    was not produced under. A15's `--skip-panel` concern was exactly this: a
    panel built under one set of semantics being labelled with current config.
    """
    return {
        "window": [str(config.SAMPLE_START), str(config.SAMPLE_END)],
        "news_source": config.NEWS_SOURCE,
        "news_source_domains": list(config.NEWS_SOURCE_DOMAINS),
        "aggregation": config.AGG,
        "dispersion_floor": config.MIN_HEADLINES_FOR_DISPERSION,
        "horizons": list(config.HORIZONS),
        "nw_maxlags": config.NW_MAXLAGS,
        "hac_convention": config.HAC_CONVENTION,
        "fdr_q": config.FDR_Q,
        "sesoi_bps": config.SESOI_BPS,
        "rq2_admissible": config.RQ2_ADMISSIBLE,
        "date_only_fallback": config.DATE_ONLY_FALLBACK,
        "finbert": {
            "model": config.FINBERT_MODEL,
            "revision": config.FINBERT_REVISION,
            "max_length": config.FINBERT_MAX_LENGTH,
            "batch_size": config.FINBERT_BATCH_SIZE,
            "batch_order": config.FINBERT_BATCH_ORDER,
        },
        "lm_dictionary": {
            "version": config.LM_DICT_VERSION,
            "sha256": config.LM_DICT_SHA256,
        },
        "seed": config.SEED,
    }


def scorer_fingerprints() -> dict | None:
    """The identities recorded in the score cache, read without loading models."""
    from src import scoring

    if not Path(config.SCORES_PARQUET).exists():
        return None
    try:
        _, meta = scoring.load_cache(config.SCORES_PARQUET)
    except Exception:                      # noqa: BLE001 - reported as unknown
        return None
    return {"fingerprints": meta.get("fingerprints"),
            "generation_id": meta.get("generation_id")}


class IncompleteRun(RuntimeError):
    """A published results directory whose manifest is missing or inconsistent."""


class StagedRun:
    """Collect outputs in a staging area; publish only if the run completes.

    Use as a context manager. `path_for` hands back a staging path; on clean
    exit every staged file is moved into its destination, outputs the previous
    run owned but this one did not produce are removed, and the manifest is
    written last. On an exception nothing is moved and the previous results are
    left exactly as they were.
    """

    def __init__(self, destinations: dict[str, Path], manifest_dir: Path | None = None):
        self.destinations = {k: Path(v) for k, v in destinations.items()}
        self.manifest_dir = Path(manifest_dir or next(iter(self.destinations.values())))
        self._staging: Path | None = None
        self._staged: list[tuple[str, str]] = []      # (kind, filename)
        self.extra: dict = {}
        self.published: list[str] | None = None

    # ------------------------------------------------------------- lifecycle

    def __enter__(self) -> "StagedRun":
        self._staging = Path(tempfile.mkdtemp(prefix=".run-", dir=self.manifest_dir.parent))
        for kind in self.destinations:
            (self._staging / kind).mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is None:
                self._publish()
        finally:
            if self._staging and self._staging.exists():
                shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = None
        return False                       # never swallow the original failure

    # ---------------------------------------------------------------- writing

    def path_for(self, kind: str, filename: str) -> Path:
        """A staging path. The file appears in its destination only on success."""
        if self._staging is None:
            raise RuntimeError("StagedRun must be used as a context manager")
        if kind not in self.destinations:
            raise KeyError(f"unknown output kind {kind!r}; known: {sorted(self.destinations)}")
        self._staged.append((kind, filename))
        return self._staging / kind / filename

    # ------------------------------------------------------------ publication

    def previous_manifest(self) -> dict | None:
        path = self.manifest_dir / MANIFEST_NAME
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def _owned_previously(self) -> set[tuple[str, str]]:
        previous = self.previous_manifest()
        if not previous:
            return set()
        return {(o["kind"], o["file"]) for o in previous.get("outputs", [])}

    def _publish(self) -> None:
        assert self._staging is not None
        produced = list(dict.fromkeys(self._staged))     # de-duplicate, keep order
        missing = [(k, f) for k, f in produced if not (self._staging / k / f).exists()]
        if missing:
            raise IncompleteRun(
                f"{len(missing)} output(s) were requested but never written: {missing}. "
                "Nothing was published; the previous results are untouched."
            )

        # The manifest is written last, so it is the commit point.
        stale = self._owned_previously() - set(produced)
        for kind, name in produced:
            destination = self.destinations[kind]
            destination.mkdir(parents=True, exist_ok=True)
            os.replace(self._staging / kind / name, destination / name)
        removed = []
        for kind, name in sorted(stale):
            target = self.destinations.get(kind)
            if target and (target / name).exists():
                (target / name).unlink()
                removed.append(f"{kind}/{name}")

        outputs = [{"kind": k, "file": f,
                    "sha256": sha256_file(self.destinations[k] / f),
                    "bytes": (self.destinations[k] / f).stat().st_size}
                   for k, f in produced]
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "environment": environment(),
            "settings": frozen_settings(),
            "inputs": input_digests(),
            "scorer_identity": scorer_fingerprints(),
            "outputs": outputs,
            "removed_superseded_outputs": removed,
            **self.extra,
        }
        payload = json.dumps(manifest, indent=2, default=str)
        temporary = self.manifest_dir / f".{MANIFEST_NAME}.tmp"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, self.manifest_dir / MANIFEST_NAME)
        self.published = [f"{k}/{f}" for k, f in produced]


# ------------------------------------------------------------- verification


def verify_published(manifest_dir: Path | None = None,
                     destinations: dict[str, Path] | None = None) -> dict:
    """Check a results directory against its manifest.

    The question this answers is not "did a run happen" but "is what is on disk
    what that run produced". A missing manifest, a missing file or a changed
    digest all mean the directory cannot be quoted as a coherent set of results.
    """
    manifest_dir = Path(manifest_dir or (config.REPORT / "tables"))
    destinations = destinations or {"tables": config.REPORT / "tables",
                                    "figures": config.FIGURES}
    path = manifest_dir / MANIFEST_NAME
    if not path.exists():
        raise IncompleteRun(
            f"no {MANIFEST_NAME} in {manifest_dir}. Either no run has completed, "
            "or one failed partway and published nothing. Any files present are "
            "from an earlier configuration and must not be quoted as current."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for entry in manifest.get("outputs", []):
        target = Path(destinations[entry["kind"]]) / entry["file"]
        if not target.exists():
            problems.append(f"missing: {entry['kind']}/{entry['file']}")
        elif sha256_file(target) != entry["sha256"]:
            problems.append(f"modified since publication: {entry['kind']}/{entry['file']}")
    for key, record in manifest.get("inputs", {}).items():
        if not record.get("present"):
            continue
        current = Path(record["path"])
        if not current.exists():
            problems.append(f"input gone: {key}")
        elif sha256_file(current) != record["sha256"]:
            problems.append(f"input changed since the run: {key}")
    return {"manifest": manifest, "ok": not problems, "problems": problems}


# ------------------------------------------------------- panel provenance


PANEL_PROVENANCE_KEY = "panel_provenance"


def panel_provenance() -> dict:
    """What a panel was built from, recorded inside the panel itself.

    A15: `--skip-panel` accepted any existing parquet, so a panel built under
    earlier timing or scoring semantics could be reported under current config.
    A schema check cannot detect that -- the columns are the same. The digests
    of the three inputs and the frozen settings can.
    """
    return {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": input_digests({
            "headlines": config.HEADLINES_PARQUET,
            "scores": config.SCORES_PARQUET,
            "market": config.MARKET_PARQUET,
        }),
        "settings": frozen_settings(),
        "scorer_identity": scorer_fingerprints(),
        "git_commit": git_commit(),
    }


def check_panel_provenance(recorded: dict | None, *, strict: bool = True) -> list[str]:
    """Compare a reused panel's recorded provenance against the world now.

    Returns the discrepancies. An unprovenanced panel is itself a discrepancy:
    it was written before this check existed, so nothing is known about the
    semantics it carries and it must be rebuilt rather than trusted.
    """
    if not recorded:
        return ["the panel carries no provenance; it predates this check. Rebuild it."]

    problems = []
    current_inputs = input_digests({
        "headlines": config.HEADLINES_PARQUET,
        "scores": config.SCORES_PARQUET,
        "market": config.MARKET_PARQUET,
    })
    for key, now in current_inputs.items():
        was = recorded.get("inputs", {}).get(key, {})
        if not was.get("present"):
            problems.append(f"{key}: absent when the panel was built")
        elif not now.get("present"):
            problems.append(f"{key}: present at build time, missing now")
        elif was.get("sha256") != now.get("sha256"):
            problems.append(f"{key}: changed since the panel was built")

    was_settings = recorded.get("settings", {})
    now_settings = frozen_settings()
    for key in sorted(set(was_settings) | set(now_settings)):
        if was_settings.get(key) != now_settings.get(key):
            problems.append(
                f"setting {key!r}: panel built under {was_settings.get(key)!r}, "
                f"config now says {now_settings.get(key)!r}"
            )
    if strict:
        was_id = (recorded.get("scorer_identity") or {}).get("generation_id")
        now_id = (scorer_fingerprints() or {}).get("generation_id")
        if was_id != now_id:
            problems.append(
                f"score cache generation differs: panel built from {was_id}, "
                f"cache is now {now_id}"
            )
    return problems
