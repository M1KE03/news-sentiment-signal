"""What must be true before a stage runs, checked and reported (R10).

Audit A13: the declared environment was not the tested environment. Every pin in
`requirements.txt` was a version the project had never been run against, and
three declared packages were not installed at all. A green test suite therefore
established behaviour in *some* environment, not in the pinned one.

Audit A15: readiness checking stopped at `_check_locked_decisions`, which reads
three config constants. It did not ask whether the artifacts a stage consumes
exist, so a missing corpus or an unscored cache surfaced as a traceback from
deep inside a loader rather than as a statement of what to run first.

This module answers two questions and refuses to guess at either:

    what is installed, against what is declared      -> `dependency_report`
    what a stage needs, against what is on disk      -> `artifact_report`

Neither installs anything, downloads anything, or changes any setting. A
missing prerequisite is reported with the command that produces it; deciding to
run that command is the operator's, not this module's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

import config

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"

# PEP 503 normalization: `scikit-learn`, `scikit_learn` and `Scikit.Learn` are
# one project, and the installed-distribution name is not always the import name.
_IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "pandas-market-calendars": "pandas_market_calendars",
    "vadersentiment": "vaderSentiment",
}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path: Path | str | None = None) -> dict[str, str]:
    """`{normalized name: declared version}` from a pinned requirements file.

    Only `==` pins are recognised. Anything looser is reported as unpinned
    rather than silently accepted, because D16's reproduction promise is that a
    clean clone installs *these* versions.
    """
    out: dict[str, str] = {}
    # Resolved here, not in the signature: a default argument binds once at
    # import and would pin this module to one file for the life of the process.
    path = REQUIREMENTS if path is None else path
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" in line:
            name, version = line.split("==", 1)
            out[normalize(name)] = version.strip()
        else:
            out[normalize(re.split(r"[<>=!~\[]", line, 1)[0])] = ""
    return out


def installed_version(name: str) -> str | None:
    """The installed distribution version, or None. Never imports the package."""
    for candidate in (name, _IMPORT_NAME.get(name, name)):
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def dependency_report(path: Path | str | None = None) -> pd.DataFrame:
    """Declared vs installed, one row per requirement.

    `status` is one of:
      ok        -- installed version equals the pin
      mismatch  -- installed, but not the version this project claims to pin
      absent    -- declared and not installed
      unpinned  -- declared without an `==` version
    """
    rows = []
    for name, declared in parse_requirements(path).items():
        found = installed_version(name)
        if not declared:
            status = "unpinned"
        elif found is None:
            status = "absent"
        elif found == declared:
            status = "ok"
        else:
            status = "mismatch"
        rows.append(
            {"package": name, "declared": declared or "(unpinned)",
             "installed": found or "(absent)", "status": status}
        )
    out = pd.DataFrame(rows).sort_values(
        ["status", "package"], key=lambda c: c.map(
            {"absent": 0, "mismatch": 1, "unpinned": 2, "ok": 3}).fillna(c)
    ).reset_index(drop=True)
    return out


# --------------------------------------------------------------- artifacts


@dataclass(frozen=True)
class Artifact:
    """A file a stage consumes, and the command that produces it.

    `external` marks something no command in this repository can create -- the
    Loughran-McDonald dictionary has no stable download URL and the annotation
    labels need a person. Naming that distinction is the point: an operator can
    act on "run this command" immediately and needs to plan for the other.
    """

    key: str
    path: Path
    produced_by: str
    why: str
    external: bool = False

    def exists(self) -> bool:
        return Path(self.path).exists()


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact("headlines_raw", config.HEADLINES_RAW_PARQUET,
             "python data/raw/download.py --assemble",
             "the verified pre-dedup corpus; keeps the dedup rate checkable"),
    Artifact("headlines", config.HEADLINES_PARQUET,
             "python data/raw/download.py --dedup",
             "the analysis corpus: deduplicated headlines with lineage"),
    Artifact("market", config.MARKET_PARQUET,
             "python data/raw/download.py --market",
             "SPY and ^VIX on the NYSE calendar; needs yfinance installed"),
    Artifact("scores", config.SCORES_PARQUET,
             "python rescore.py",
             "the scoring cache; the only expensive step in the project"),
    Artifact("panel", config.PANEL_PARQUET,
             "python run_all.py",
             "the single analysis table every Act 2 number is read from"),
    Artifact("lm_dictionary", config.LM_DICT_PATH,
             "download by hand from the Notre Dame SRAF site "
             "(see docs/lm-dictionary-provenance.md)",
             "Loughran-McDonald word lists; no stable URL exists",
             external=True),
    Artifact("annotation_sample", config.ANNOTATION_DIR / "to_label_primary.csv",
             "python -c \"from src import annotate; annotate.draw_sample()\"",
             "Act 1's blind evaluation sample; the draw is made once (P25)"),
    Artifact("annotation_labels", config.ANNOTATION_DIR / "labels_primary.csv",
             "a human annotator fills data/annotation/pilot_worksheet.csv, "
             "then the full sheet (see data/annotation/PILOT_README.md)",
             "independent human labels; Act 1 cannot be measured without them",
             external=True),
)

_BY_KEY = {a.key: a for a in ARTIFACTS}

# What each stage consumes. A stage lists only its *direct* inputs: the message
# should name the next command to run, not the whole chain behind it.
#
# The vocabulary is the one the root `preflight.py` CLI already used, so there
# is exactly one set of stage names in the project rather than a pipeline set
# and a readiness set that drift apart.
STAGES: dict[str, tuple[str, ...]] = {
    "unit": (),                                     # offline tests need nothing
    "pilot": ("headlines", "lm_dictionary"),        # timing a scorer
    "scoring": ("headlines", "lm_dictionary"),      # the full pass
    "analysis": ("headlines", "scores", "market"),  # panel and every Act 2 number
    "validation": ("annotation_sample", "annotation_labels", "scores"),
}


def artifact_report(stage: str | None = None) -> pd.DataFrame:
    """One row per artifact, with whether it is present and what produces it."""
    keys = _stage_keys(stage) if stage else tuple(a.key for a in ARTIFACTS)
    rows = []
    for key in keys:
        a = _BY_KEY[key]
        rows.append({
            "artifact": a.key,
            "present": a.exists(),
            "path": str(a.path),
            "source": "external" if a.external else "command",
            "produced_by": a.produced_by,
            "why": a.why,
        })
    # An empty stage (`unit` needs nothing) must still carry the columns, or
    # every caller that reads one has to special-case the empty frame.
    columns = ["artifact", "present", "path", "source", "produced_by", "why"]
    return pd.DataFrame(rows, columns=columns).astype({"present": bool})


def _stage_keys(stage: str) -> tuple[str, ...]:
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; known stages: {sorted(STAGES)}")
    return STAGES[stage]


def missing_artifacts(stage: str) -> list[Artifact]:
    return [_BY_KEY[k] for k in _stage_keys(stage) if not _BY_KEY[k].exists()]


class PreflightError(RuntimeError):
    """A prerequisite is absent. The message says which, and what produces it."""


def require_artifacts(stage: str) -> None:
    """Refuse to start a stage whose inputs are not on disk.

    The failure this replaces is not a crash -- it is a crash *from the wrong
    place*. A missing corpus previously surfaced as a parquet read error inside
    a loader, several frames from anything the operator could act on.
    """
    missing = missing_artifacts(stage)
    if not missing:
        return
    lines = [f"cannot run stage {stage!r}: {len(missing)} prerequisite(s) missing.", ""]
    for a in missing:
        lines.append(f"  {a.key}")
        lines.append(f"    expected at : {a.path}")
        lines.append(f"    {'obtain by  ' if a.external else 'produced by'} : {a.produced_by}")
        lines.append(f"    why         : {a.why}")
        lines.append("")
    if any(a.external for a in missing):
        lines.append(
            "One or more of these cannot be produced by any command in this "
            "repository and needs a person; see docs/handover.md section 6."
        )
    raise PreflightError("\n".join(lines))


def require_packages(names: Sequence[str]) -> None:
    """Refuse to start when a package a stage genuinely needs is absent.

    Separate from `dependency_report` on purpose: a *mismatch* is a
    reproducibility fact to report, while an *absence* is a hard stop for the
    stage that imports it. Reporting both through one path would either block
    on every version drift or stay silent on a package that is not there.
    """
    absent = [n for n in names if installed_version(n) is None]
    if absent:
        raise PreflightError(
            f"missing required package(s): {', '.join(absent)}.\n"
            "Install the pinned environment first: pip install -r requirements.txt"
        )


def summary(stage: str | None = None) -> str:
    """A human-readable readiness report. Reports; never raises on findings."""
    deps = dependency_report()
    counts = deps["status"].value_counts().to_dict()
    arts = artifact_report(stage)

    lines = [
        f"python            : {_python_version()}",
        f"requirements      : {REQUIREMENTS}",
        "dependencies      : "
        + ", ".join(f"{counts.get(k, 0)} {k}"
                    for k in ("ok", "mismatch", "absent", "unpinned")),
    ]
    off = deps[deps["status"] != "ok"]
    if not off.empty:
        lines.append("")
        lines.append("  declared but not matched:")
        for _, r in off.iterrows():
            lines.append(f"    {r['package']:<26} declared {r['declared']:<12} "
                         f"installed {r['installed']}")
    lines.append("")
    lines.append(f"artifacts{'' if stage is None else f' for stage {stage!r}'} :")
    for _, r in arts.iterrows():
        mark = "present" if r["present"] else "MISSING"
        lines.append(f"    [{mark:>7}] {r['artifact']:<20} {r['path']}")
        if not r["present"]:
            label = "obtain by" if r["source"] == "external" else "produced by"
            lines.append(f"                {label}: {r['produced_by']}")
    return "\n".join(lines)


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def freeze(path: Path | str | None = None) -> str:
    """The installed versions of every declared requirement, as pin lines.

    A13's repair is to re-pin against the environment the tests actually ran in,
    rather than against versions nothing was ever executed against. This
    produces those lines; writing them into `requirements.txt` stays a
    deliberate act.
    """
    lines = []
    for name in parse_requirements(path):
        found = installed_version(name)
        lines.append(f"{name}=={found}" if found else f"# {name}: NOT INSTALLED")
    return "\n".join(lines)


# ------------------------------------------------- verification beyond existence
#
# A file being present is not the same as it being the right file. These are
# separated from `require_artifacts` on purpose: existence is cheap and runs on
# every command, while reading a schema, digesting a dictionary or validating a
# score cache costs real time and is asked for deliberately.


def check_parquet(path: Path | str, required: Iterable[str]) -> str:
    """The columns a consumer will read must actually be in the file."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    missing = sorted(set(required) - set(schema.names))
    if missing:
        raise PreflightError(
            f"{path} is missing column(s) {missing}. It was written by older "
            "code; rebuild the artifact rather than patching it in place."
        )
    return f"{path}: {pq.read_metadata(path).num_rows:,} rows"


def check_lm_dictionary() -> str:
    """The dictionary has no stable URL, so its digest is the only identity."""
    import hashlib

    actual = hashlib.sha256(Path(config.LM_DICT_PATH).read_bytes()).hexdigest()
    if actual != config.LM_DICT_SHA256:
        raise PreflightError(
            f"{config.LM_DICT_PATH} has SHA-256 {actual}, but config.LM_DICT_SHA256 "
            f"pins {config.LM_DICT_SHA256}. A different release measures different "
            "words; see docs/lm-dictionary-provenance.md."
        )
    return f"{config.LM_DICT_PATH}: verified {actual}"


def check_score_cache(cache_path: Path | str | None = None) -> str:
    """Complete, and produced by the scorers currently configured.

    This is the check `--skip-panel` could not make: a cache can be complete and
    still have been produced by a different measurement, which is what the
    fingerprint comparison catches (B13/R01a).
    """
    import pandas as pd

    from src import align, scoring

    path = Path(config.SCORES_PARQUET if cache_path is None else cache_path)
    scores, meta = scoring.load_cache(path)
    headlines = pd.read_parquet(config.HEADLINES_PARQUET)
    align.validate_scores(headlines, scores)
    for name, fingerprint in scoring._configured_fingerprints().items():
        recorded = meta["fingerprints"].get(name)
        if scoring._canonical_fingerprint(fingerprint) != recorded:
            raise PreflightError(
                f"the cached {name} column was produced by a different scorer "
                f"identity than the one configured now; reusing it would mix two "
                f"definitions inside one column. Recorded: {recorded}"
            )
    return f"{path}: complete, verified scorer identities"


def verify(stage: str, *, load_models: bool = False,
           cache_path: Path | str | None = None) -> list[dict]:
    """Run every content check a stage warrants. Reports; never raises on findings.

    Failures are recorded with their original exception type. A model that fails
    to load is reported as the error it raised, never converted into a skip --
    that conversion is what let a real label-order defect read as "unavailable"
    for weeks (A13).
    """
    results: list[dict] = []

    def run(name, operation):
        try:
            results.append({"check": name, "ok": True, "detail": str(operation())})
        except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
            results.append({"check": name, "ok": False,
                            "detail": f"{type(exc).__name__}: {exc}"})

    if stage == "unit":
        return results

    run("headlines", lambda: check_parquet(
        config.HEADLINES_PARQUET, ["headline_id", "text", "ts_utc"]))
    run("lm_dictionary", check_lm_dictionary)

    if load_models:
        from src import scoring

        for name in config.SCORERS:
            run(f"model:{name}",
                lambda name=name: scoring.build_scorers([name])[0].fingerprint)

    if stage == "analysis":
        run("market", lambda: check_parquet(
            config.MARKET_PARQUET, ["date", "ret", "log_volume", "rv_parkinson"]))
        run("scores", lambda: check_score_cache(cache_path))

    if stage == "validation":
        from src import annotate, validate

        run("frozen_split", lambda: len(annotate.load_split()))
        run("human_labels", lambda: len(validate.load_annotations(
            config.ANNOTATION_DIR / "labels_primary.csv",
            config.ANNOTATION_DIR / "provenance_primary.json", part="evaluation")))
        results.append({
            "check": "frozen_thresholds",
            "ok": config.VALIDATION_THRESHOLDS is not None,
            "detail": "config.VALIDATION_THRESHOLDS must be recorded after calibration",
        })
    return results


def report(stage: str = "unit", *, load_models: bool = False,
           cache_path: Path | str | None = None) -> dict:
    """The whole readiness picture, JSON-serialisable for a run manifest (R14)."""
    import platform
    import sys

    checks = verify(stage, load_models=load_models, cache_path=cache_path)
    deps = dependency_report()
    arts = artifact_report(stage)
    return {
        "stage": stage,
        "ok": all(c["ok"] for c in checks) and bool(arts["present"].all()),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "checks": checks,
        "dependencies": deps.to_dict("records"),
        "artifacts": arts.to_dict("records"),
        "scope": "dependency and artifact readiness only; does not certify "
                 "unfinished research methods",
    }
