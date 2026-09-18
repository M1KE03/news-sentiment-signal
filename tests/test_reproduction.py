"""R15 / B28: does the documented workflow describe the code that exists?

The clean-directory check has two halves. The half a test can do is verify that
every command the README promises exists and every claim the report makes is
traceable to a published file. The half it cannot do is run the pipeline from an
empty directory -- that needs a 5.7 GB download and hours of scoring, so it is a
documented manual procedure (see `docs/archive/reproduction.md`) rather than a test.

What this catches is the failure that actually recurs: documentation drifting
from the code until a reader follows an instruction that no longer works.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src import publish


def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def report() -> str:
    return (ROOT / "report" / "report.md").read_text(encoding="utf-8")


# ------------------------------------------- every promised command must exist


def test_every_script_the_readme_tells_a_reader_to_run_exists():
    for script in re.findall(r"^python ([\w./\\-]+\.py)", readme(), re.MULTILINE):
        assert (ROOT / script).exists(), f"README tells a reader to run {script}, which is absent"


@pytest.mark.parametrize("script,flag", [
    ("data/raw/download.py", "--assemble"),
    ("data/raw/download.py", "--dedup"),
    ("data/raw/download.py", "--census"),
    ("data/raw/download.py", "--market"),
    ("rescore.py", "--dry-run"),
    ("rescore.py", "--time-only"),
    ("rescore.py", "--sessions"),
    ("run_all.py", "--skip-panel"),
    ("run_all.py", "--allow-stale-panel"),
    ("preflight.py", "--stage"),
    ("preflight.py", "--strict"),
    ("preflight.py", "--freeze"),
])
def test_every_documented_flag_is_accepted_by_its_script(script, flag):
    """A flag in the docs that argparse does not define is a broken instruction."""
    out = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                         capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert out.returncode == 0, f"{script} --help failed: {out.stderr[:400]}"
    assert flag in out.stdout, f"{script} does not accept {flag}, but the docs use it"


def test_the_readme_does_not_promise_a_flag_that_was_removed():
    for gone in ("--all", "--draws"):
        assert gone not in readme(), f"README still documents the removed {gone}"


# --------------------------------------- every empirical claim must be traceable


def published_manifest() -> dict:
    path = config.REPORT / "tables" / publish.MANIFEST_NAME
    if not path.exists():
        pytest.skip("no completed run in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_published_outputs_match_their_manifest():
    published_manifest()
    result = publish.verify_published()
    assert result["ok"], f"published results do not match the manifest: {result['problems']}"


def test_every_table_the_report_names_was_actually_published():
    manifest = published_manifest()
    files = {o["file"] for o in manifest["outputs"]}
    for named in re.findall(r"`(table\w*\.csv|advance_precision\.json)`", report()):
        assert named in files or named == "advance_precision.json", (
            f"the report cites {named}, which no completed run published"
        )


def test_the_reports_primary_number_matches_the_published_table():
    """The claim a reader will quote must equal the file it came from."""
    path = config.REPORT / "tables" / "table4_effect_sizes.csv"
    if not path.exists():
        pytest.skip("no completed run in this checkout")
    table = pd.read_csv(path)
    row = table[(table["scorer"] == "finbert") & (table["horizon"] == 1)].iloc[0]

    text = report()
    assert f"{row['bps_per_sd']:.2f}" in text.replace("−", "-"), "primary point estimate not in the report"
    for endpoint in (row["bps_lo95"], row["bps_hi95"]):
        assert f"{abs(endpoint):.2f}" in text, f"interval endpoint {endpoint:.2f} not in the report"


def test_the_boundary_flag_travels_with_the_primary_result():
    """M2 requires disclosure; a boundary case quoted bare overstates the null."""
    path = config.REPORT / "tables" / "table4_effect_sizes.csv"
    if not path.exists():
        pytest.skip("no completed run in this checkout")
    table = pd.read_csv(path)
    row = table[(table["scorer"] == "finbert") & (table["horizon"] == 1)].iloc[0]
    if not row["boundary_case"]:
        pytest.skip("the primary result is not a boundary case in this run")
    for document in (readme(), report()):
        assert "boundary" in document.lower(), (
            "the primary result is a flagged boundary case and every document "
            "quoting it must say so"
        )


# ------------------------------------------- what must NOT be claimed yet


def test_the_act_1_numbers_match_the_published_table():
    """Superseded 2026-09-10. These two tests previously asserted that NO
    macro-F1 value appeared anywhere, because Act 1 had no labels and any such
    number would have been fabricated. Act 1 has now run, so the guard flips
    from "claims nothing" to "claims exactly what the file says"."""
    path = config.REPORT / "tables" / "table1_act1_metrics.csv"
    if not path.exists():
        pytest.skip("Act 1 has not been evaluated in this checkout")
    table = pd.read_csv(path)
    full = table[table["set"] == "full"].set_index("scorer")

    for name, document in (("README", readme()), ("report", report())):
        for scorer in ("finbert", "lm", "vader"):
            value = f"{full.loc[scorer, 'macro_f1']:.3f}"
            assert value in document, (
                f"{name} does not carry {scorer}'s published macro-F1 {value}"
            )


def test_the_act_1_primary_contrast_matches_its_table():
    path = config.REPORT / "tables" / "table1b_act1_contrasts.csv"
    if not path.exists():
        pytest.skip("Act 1 has not been evaluated in this checkout")
    contrasts = pd.read_csv(path)
    row = contrasts[(contrasts["set"] == "full") &
                    (contrasts["contrast"] == "finbert-lm")].iloc[0]
    for document in (readme(), report()):
        assert f"{row['macro_f1_difference']:.3f}" in document
        for endpoint in (row["macro_f1_lo95"], row["macro_f1_hi95"]):
            assert f"{endpoint:.3f}" in document, f"interval endpoint {endpoint:.3f} missing"


def test_the_act_1_confounds_travel_with_the_result():
    """A +0.102 macro-F1 gap quoted bare would overstate what it establishes:
    the entire difference is one class, and the rubric's framing is closer to
    FinBERT's training objective than to the lexicons'."""
    for name, document in (("README", readme()), ("report", report())):
        lowered = document.lower()
        assert "positive" in lowered and "class" in lowered
        assert re.search(r"2,345|347", document), (
            f"{name} should state the LM dictionary's negative/positive asymmetry"
        )


def _is_correction(line: str) -> bool:
    """A note recording what was removed may quote the phrase it removed."""
    lowered = line.lower()
    return line.lstrip().startswith(">") or any(
        marker in lowered for marker in
        ("previously", "corrected", "withdrawn", "removed", "earlier version")
    )


def test_no_document_frames_the_yardstick_as_profitability():
    """P19: the SESOI is a yardstick for smallness, never a profitability test."""
    for name, document in (("README", readme()), ("report", report())):
        for line in document.splitlines():
            if _is_correction(line):
                continue
            lowered = line.lower()
            if "transaction-cost benchmark" in lowered:
                pytest.fail(f"{name} frames the SESOI as a cost benchmark: {line.strip()!r}")
            for phrase in ("profitable", "strategy return"):
                if phrase in lowered and "not" not in lowered:
                    pytest.fail(f"{name} profitability framing: {line.strip()!r}")


def test_the_dependency_record_matches_the_environment():
    """B28: a clean clone installs these versions, so they must be the tested ones."""
    from src import preflight

    drift = preflight.dependency_report()
    bad = drift[drift["status"] != "ok"]
    assert bad.empty, f"requirements.txt no longer describes this environment:\n{bad}"
