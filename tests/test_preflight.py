"""R10: readiness checks that name what is missing (audit A13, A15).

A13: the declared environment was not the tested environment. Eleven of fourteen
pins differed from what was installed and three declared packages were absent, so
a green suite established behaviour in *some* environment rather than the pinned
one. The repair is not a promise to keep them in step -- it is a command that
reports the difference whenever anyone asks.

A15: readiness stopped at three config constants, so a missing corpus surfaced as
a parquet read error several frames inside a loader. `require_artifacts` fails at
the top instead, naming the artifact, where it was expected, and the command that
produces it.

These tests use temporary requirements files and monkeypatched artifact
registries: nothing here depends on what happens to be installed on the machine
running the suite, which is the same discipline the module exists to enforce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import preflight


def write_requirements(tmp_path, body: str) -> Path:
    p = tmp_path / "requirements.txt"
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------- parsing the pins


def test_parse_reads_versions_and_skips_comments_and_blanks(tmp_path):
    req = write_requirements(tmp_path, """
# a comment
numpy==2.5.3

pandas==3.0.5   # trailing comment
""")
    assert preflight.parse_requirements(req) == {"numpy": "2.5.3", "pandas": "3.0.5"}


def test_names_are_normalized_so_one_project_is_one_row(tmp_path):
    req = write_requirements(tmp_path, "scikit_learn==1.9.0\nPandas_Market.Calendars==5.4.0\n")
    parsed = preflight.parse_requirements(req)
    assert "scikit-learn" in parsed
    assert "pandas-market-calendars" in parsed


def test_a_loose_requirement_is_reported_as_unpinned_not_accepted(tmp_path):
    """D16 promises a clean clone installs *these* versions, not compatible ones."""
    req = write_requirements(tmp_path, "numpy>=2.0\n")
    report = preflight.dependency_report(req)
    assert report.loc[0, "status"] == "unpinned"


# ------------------------------------------------------ the dependency report


def test_the_report_separates_ok_mismatch_and_absent(tmp_path, monkeypatch):
    req = write_requirements(
        tmp_path, "goodpkg==1.0.0\nwrongpkg==1.0.0\nabsentpkg==1.0.0\n")
    versions = {"goodpkg": "1.0.0", "wrongpkg": "9.9.9"}
    monkeypatch.setattr(preflight, "installed_version", versions.get)

    report = preflight.dependency_report(req).set_index("package")
    assert report.loc["goodpkg", "status"] == "ok"
    assert report.loc["wrongpkg", "status"] == "mismatch"
    assert report.loc["absentpkg", "status"] == "absent"
    assert report.loc["absentpkg", "installed"] == "(absent)"


def test_problems_sort_before_healthy_rows(tmp_path, monkeypatch):
    """The reason to run this is to see what is wrong; put that at the top."""
    req = write_requirements(tmp_path, "aaa==1.0\nzzz==1.0\n")
    monkeypatch.setattr(preflight, "installed_version",
                        {"aaa": "1.0", "zzz": None}.get)
    report = preflight.dependency_report(req)
    assert report.loc[0, "package"] == "zzz"       # absent first, despite the name
    assert report.loc[1, "status"] == "ok"


def test_the_projects_own_requirements_file_parses():
    """A pin nobody can read is not a pin."""
    parsed = preflight.parse_requirements()
    assert parsed, "requirements.txt produced no pins"
    assert all(v for v in parsed.values()), f"unpinned entries: {parsed}"


def test_the_declared_environment_is_the_installed_one(tmp_path):
    """A13 itself, as a standing check.

    This is the assertion that would have caught the original finding: eleven
    mismatches and three absent packages, while the suite reported green. It is
    deliberately strict -- if it fails, either re-pin with
    `python preflight.py --freeze` or state why the drift is acceptable.
    """
    report = preflight.dependency_report()
    bad = report[report["status"] != "ok"]
    assert bad.empty, (
        "requirements.txt no longer describes the environment the tests run in:\n"
        + bad.to_string(index=False)
    )


# ------------------------------------------------------------ artifact gates


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A two-artifact registry: one that a command builds, one that a person does."""
    present = tmp_path / "present.parquet"
    present.write_text("x", encoding="utf-8")
    made = preflight.Artifact("made", tmp_path / "missing.parquet",
                              "python data/raw/download.py --dedup", "the corpus")
    human = preflight.Artifact("human", tmp_path / "labels.csv",
                               "a human annotator fills the worksheet",
                               "independent labels", external=True)
    here = preflight.Artifact("here", present, "python run_all.py", "already built")
    monkeypatch.setattr(preflight, "ARTIFACTS", (made, human, here))
    monkeypatch.setattr(preflight, "_BY_KEY", {a.key: a for a in (made, human, here)})
    monkeypatch.setattr(preflight, "STAGES",
                        {"ready": ("here",), "blocked": ("here", "made", "human")})
    return tmp_path


def test_a_stage_whose_inputs_exist_does_not_raise(registry):
    preflight.require_artifacts("ready")


def test_a_missing_artifact_names_itself_its_path_and_its_command(registry):
    with pytest.raises(preflight.PreflightError) as exc:
        preflight.require_artifacts("blocked")
    message = str(exc.value)
    assert "made" in message
    assert "missing.parquet" in message
    assert "python data/raw/download.py --dedup" in message


def test_an_artifact_no_command_can_produce_is_marked_as_needing_a_person(registry):
    """"Run this" and "plan for this" are different instructions."""
    with pytest.raises(preflight.PreflightError) as exc:
        preflight.require_artifacts("blocked")
    message = str(exc.value)
    assert "obtain by" in message
    assert "cannot be produced by any command in this repository" in message


def test_a_present_artifact_is_not_listed_as_missing(registry):
    with pytest.raises(preflight.PreflightError) as exc:
        preflight.require_artifacts("blocked")
    assert "already built" not in str(exc.value)


def test_an_unknown_stage_is_refused_rather_than_silently_passing(registry):
    with pytest.raises(KeyError, match="unknown stage"):
        preflight.require_artifacts("nonexistent")


def test_every_declared_stage_names_known_artifacts():
    known = {a.key for a in preflight.ARTIFACTS}
    for stage, keys in preflight.STAGES.items():
        unknown = set(keys) - known
        assert not unknown, f"stage {stage!r} names unknown artifact(s) {unknown}"


def test_the_artifact_report_covers_the_whole_registry():
    report = preflight.artifact_report()
    assert len(report) == len(preflight.ARTIFACTS)
    assert set(report["artifact"]) == {a.key for a in preflight.ARTIFACTS}


# ------------------------------------------------------------ package gates


def test_absent_packages_stop_a_stage_and_name_the_install_command(monkeypatch):
    monkeypatch.setattr(preflight, "installed_version", lambda n: None)
    with pytest.raises(preflight.PreflightError) as exc:
        preflight.require_packages(["torch", "transformers"])
    message = str(exc.value)
    assert "torch" in message and "transformers" in message
    assert "pip install -r requirements.txt" in message


def test_installed_packages_pass_the_gate(monkeypatch):
    monkeypatch.setattr(preflight, "installed_version", lambda n: "1.0")
    preflight.require_packages(["anything"])


def test_a_version_mismatch_alone_does_not_block_a_stage(monkeypatch):
    """A mismatch is a reproducibility fact to report; an absence is a stop.

    Collapsing the two would either block every run on ordinary version drift or
    stay silent about a package that is not there at all.
    """
    monkeypatch.setattr(preflight, "installed_version", lambda n: "0.0.1-not-the-pin")
    preflight.require_packages(["torch"])


# ----------------------------------------------------------------- reporting


def test_the_summary_reports_and_never_raises(registry):
    text = preflight.summary()
    assert "dependencies" in text and "artifacts" in text
    assert "MISSING" in text          # the fixture registry has one


def test_the_summary_can_be_narrowed_to_one_stage(registry):
    assert "MISSING" not in preflight.summary("ready")


def test_freeze_emits_a_pin_line_per_declared_requirement(monkeypatch, tmp_path):
    req = write_requirements(tmp_path, "alpha==1.0\nbeta==2.0\n")
    monkeypatch.setattr(preflight, "REQUIREMENTS", req)
    monkeypatch.setattr(preflight, "installed_version",
                        {"alpha": "1.5", "beta": None}.get)
    lines = preflight.freeze().splitlines()
    assert lines[0] == "alpha==1.5"
    assert lines[1].startswith("# beta") and "NOT INSTALLED" in lines[1]


def test_nothing_in_preflight_imports_a_scorer(monkeypatch):
    """Readiness must be checkable without loading a 400 MB transformer."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    preflight.dependency_report()
    preflight.artifact_report()
    assert preflight.installed_version("torch") is not None


# -------------------------------------- verification beyond mere existence
#
# The committed root `preflight.py` already carried these deeper checks when
# this module was written, and they are the substantive half: a file being
# present is not the same as it being the right file.


def test_a_parquet_missing_a_required_column_is_refused(tmp_path):
    p = tmp_path / "x.parquet"
    pd.DataFrame({"a": [1], "b": [2]}).to_parquet(p)
    preflight.check_parquet(p, ["a", "b"])
    with pytest.raises(preflight.PreflightError, match=r"missing column\(s\) \['c'\]"):
        preflight.check_parquet(p, ["a", "c"])


def test_the_parquet_check_reports_the_row_count(tmp_path):
    p = tmp_path / "x.parquet"
    pd.DataFrame({"a": range(7)}).to_parquet(p)
    assert "7 rows" in preflight.check_parquet(p, ["a"])


def test_a_dictionary_whose_digest_differs_is_refused(tmp_path, monkeypatch):
    """No stable URL exists for it, so the digest is the only identity it has."""
    fake = tmp_path / "lm.csv"
    fake.write_text("not the real dictionary", encoding="utf-8")
    monkeypatch.setattr(config, "LM_DICT_PATH", fake)
    with pytest.raises(preflight.PreflightError, match="SHA-256"):
        preflight.check_lm_dictionary()


def test_the_real_dictionary_matches_its_pin():
    if not Path(config.LM_DICT_PATH).exists():
        pytest.skip("dictionary not acquired in this checkout")
    assert config.LM_DICT_SHA256 in preflight.check_lm_dictionary()


def test_verify_reports_failures_with_their_type_and_never_skips(monkeypatch):
    """A13: converting a model error into a skip is what hid a real defect."""
    def boom():
        raise RuntimeError("weights are corrupt")

    monkeypatch.setattr(preflight, "check_lm_dictionary", boom)
    monkeypatch.setattr(preflight, "check_parquet", lambda *a, **k: "fine")
    results = {c["check"]: c for c in preflight.verify("pilot")}
    assert results["lm_dictionary"]["ok"] is False
    assert "RuntimeError: weights are corrupt" in results["lm_dictionary"]["detail"]


def test_the_unit_stage_verifies_nothing_and_needs_nothing():
    assert preflight.verify("unit") == []
    assert preflight.STAGES["unit"] == ()


def test_the_report_is_json_serialisable_for_a_run_manifest(monkeypatch):
    monkeypatch.setattr(preflight, "verify", lambda *a, **k: [])
    import json

    payload = preflight.report("unit")
    json.loads(json.dumps(payload, default=str))
    assert set(payload) >= {"stage", "ok", "python", "platform", "checks",
                            "dependencies", "artifacts", "scope"}


def test_the_report_is_not_ok_when_an_artifact_is_missing(registry, monkeypatch):
    monkeypatch.setattr(preflight, "verify", lambda *a, **k: [])
    assert preflight.report("blocked")["ok"] is False
    assert preflight.report("ready")["ok"] is True


def test_one_stage_vocabulary_is_shared_by_the_cli_and_the_gates():
    """Two sets of stage names would drift; there is deliberately only one."""
    import preflight as cli

    assert set(preflight.STAGES) == {"unit", "pilot", "scoring", "analysis", "validation"}
    assert cli.preflight is preflight
