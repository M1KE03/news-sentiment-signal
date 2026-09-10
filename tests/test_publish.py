"""R14: staged publication, the run manifest, and panel provenance (audit A15).

A15: *"Outputs are written sequentially into shared paths, so a failed run can
leave a mixture of old and new tables."* And separately: *"`--skip-panel`
accepts any existing parquet and can label old timing/scoring semantics using
current config."*

Both failures are silent. A half-updated results directory looks exactly like a
complete one, and a stale panel has the right columns. These tests are about
making each of them announce itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import publish


@pytest.fixture
def dests(tmp_path):
    return {"tables": tmp_path / "tables", "figures": tmp_path / "figures"}


def write(run, kind, name, body="x"):
    path = run.path_for(kind, name)
    path.write_text(body, encoding="utf-8")
    return path


# ------------------------------------------------------- nothing until success


def test_a_completed_run_publishes_everything_and_a_manifest(dests, tmp_path):
    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "tables", "t1.csv", "a")
        write(run, "figures", "f1.png", "b")

    assert (dests["tables"] / "t1.csv").read_text() == "a"
    assert (dests["figures"] / "f1.png").read_text() == "b"
    manifest = json.loads((dests["tables"] / publish.MANIFEST_NAME).read_text())
    assert manifest["status"] == "complete"
    assert {o["file"] for o in manifest["outputs"]} == {"t1.csv", "f1.png"}


def test_a_failed_run_publishes_nothing_and_leaves_the_previous_results(dests):
    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "tables", "t1.csv", "first run")
    first = json.loads((dests["tables"] / publish.MANIFEST_NAME).read_text())

    with pytest.raises(RuntimeError, match="deliberate"):
        with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
            write(run, "tables", "t1.csv", "second run")
            write(run, "tables", "t2.csv", "half written")
            raise RuntimeError("deliberate failure partway through")

    # the old table survives untouched, the new one never appeared
    assert (dests["tables"] / "t1.csv").read_text() == "first run"
    assert not (dests["tables"] / "t2.csv").exists()
    assert json.loads((dests["tables"] / publish.MANIFEST_NAME).read_text()) == first


def test_nothing_is_visible_in_the_destination_before_the_run_ends(dests):
    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "tables", "t1.csv")
        assert not (dests["tables"] / "t1.csv").exists(), (
            "a file appearing mid-run is exactly the mixture A15 describes"
        )
    assert (dests["tables"] / "t1.csv").exists()


def test_a_requested_output_that_was_never_written_stops_publication(dests):
    with pytest.raises(publish.IncompleteRun, match="never written"):
        with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
            run.path_for("tables", "promised.csv")      # path taken, file not written
    assert not (dests["tables"] / "promised.csv").exists()
    assert not (dests["tables"] / publish.MANIFEST_NAME).exists()


def test_the_staging_directory_is_removed_either_way(dests, tmp_path):
    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "tables", "t1.csv")
    assert not list(tmp_path.glob(".run-*"))

    with pytest.raises(RuntimeError):
        with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
            write(run, "tables", "t1.csv")
            raise RuntimeError("boom")
    assert not list(tmp_path.glob(".run-*"))


def test_an_unknown_output_kind_is_refused(dests):
    with pytest.raises(KeyError, match="unknown output kind"):
        with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
            run.path_for("spreadsheets", "x.csv")


def test_path_for_outside_a_context_is_refused(dests):
    run = publish.StagedRun(dests, manifest_dir=dests["tables"])
    with pytest.raises(RuntimeError, match="context manager"):
        run.path_for("tables", "x.csv")


# ------------------------------------------------- a run owns what it published


def test_an_output_a_later_run_stops_producing_is_removed(dests):
    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "tables", "keep.csv")
        write(run, "tables", "superseded.csv")
    assert (dests["tables"] / "superseded.csv").exists()

    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "tables", "keep.csv")
    assert (dests["tables"] / "keep.csv").exists()
    assert not (dests["tables"] / "superseded.csv").exists(), (
        "a table from a superseded method must not sit beside a current one"
    )
    manifest = json.loads((dests["tables"] / publish.MANIFEST_NAME).read_text())
    assert manifest["removed_superseded_outputs"] == ["tables/superseded.csv"]


def test_files_the_runner_never_claimed_are_left_alone(dests):
    """A figure from a notebook is not this runner's to delete."""
    dests["figures"].mkdir(parents=True, exist_ok=True)
    (dests["figures"] / "figure0_from_a_notebook.png").write_text("not mine")

    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "figures", "figure2.png")
    assert (dests["figures"] / "figure0_from_a_notebook.png").read_text() == "not mine"


# --------------------------------------------------------------- verification


def test_verification_passes_on_a_freshly_published_directory(dests):
    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "tables", "t1.csv")
    result = publish.verify_published(dests["tables"], dests)
    assert result["ok"] and result["problems"] == []


def test_verification_detects_a_file_edited_after_publication(dests):
    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "tables", "t1.csv", "published")
    (dests["tables"] / "t1.csv").write_text("edited by hand")

    result = publish.verify_published(dests["tables"], dests)
    assert not result["ok"]
    assert any("modified since publication" in p for p in result["problems"])


def test_verification_detects_a_deleted_output(dests):
    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "tables", "t1.csv")
    (dests["tables"] / "t1.csv").unlink()
    assert any("missing" in p for p in publish.verify_published(dests["tables"], dests)["problems"])


def test_a_directory_with_no_manifest_cannot_be_quoted_as_current(dests):
    dests["tables"].mkdir(parents=True, exist_ok=True)
    (dests["tables"] / "leftover.csv").write_text("from some earlier run")
    with pytest.raises(publish.IncompleteRun, match="no run_manifest"):
        publish.verify_published(dests["tables"], dests)


# ------------------------------------------------------------ the manifest


def test_the_manifest_records_what_the_results_were_computed_from(dests):
    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "tables", "t1.csv")
    manifest = json.loads((dests["tables"] / publish.MANIFEST_NAME).read_text())

    assert set(manifest) >= {"manifest_version", "completed_at_utc", "status",
                             "environment", "settings", "inputs",
                             "scorer_identity", "outputs"}
    settings = manifest["settings"]
    for key in ("window", "hac_convention", "nw_maxlags", "aggregation",
                "sesoi_bps", "fdr_q", "finbert", "lm_dictionary", "seed"):
        assert key in settings, f"{key} changes a reported number and must be recorded"
    assert settings["finbert"]["batch_order"] == config.FINBERT_BATCH_ORDER
    assert "python" in manifest["environment"]


def test_extra_fields_reach_the_manifest(dests):
    with publish.StagedRun(dests, manifest_dir=dests["tables"]) as run:
        write(run, "tables", "t1.csv")
        run.extra["eligibility"] = {"n_eligible": 2453}
    manifest = json.loads((dests["tables"] / publish.MANIFEST_NAME).read_text())
    assert manifest["eligibility"]["n_eligible"] == 2453


def test_a_dirty_tree_is_recorded_rather_than_implied_clean():
    """A commit hash alone does not identify the code that ran."""
    dirty = publish.git_dirty()
    assert dirty is None or isinstance(dirty, bool)


# ----------------------------------------------------- panel provenance (A15)


def test_an_unprovenanced_panel_is_refused_rather_than_trusted():
    """A panel written before this check exists says nothing about its semantics."""
    problems = publish.check_panel_provenance(None)
    assert problems and "no provenance" in problems[0]


def test_a_matching_panel_reports_no_problems():
    recorded = publish.panel_provenance()
    assert publish.check_panel_provenance(recorded) == []


def test_a_changed_setting_is_detected_even_though_the_columns_match():
    """The exact A15 failure: same schema, different semantics."""
    recorded = publish.panel_provenance()
    recorded["settings"] = dict(recorded["settings"], hac_convention="retained_position",
                                aggregation="median")
    problems = publish.check_panel_provenance(recorded)
    assert any("hac_convention" in p for p in problems)
    assert any("aggregation" in p for p in problems)


def test_a_changed_input_digest_is_detected():
    recorded = publish.panel_provenance()
    if not recorded["inputs"]["scores"]["present"]:
        pytest.skip("no score cache in this checkout")
    recorded["inputs"]["scores"]["sha256"] = "0" * 64
    assert any("scores: changed" in p for p in publish.check_panel_provenance(recorded))


def test_a_rescored_cache_is_detected_by_generation_id():
    recorded = publish.panel_provenance()
    if not recorded.get("scorer_identity"):
        pytest.skip("no score cache in this checkout")
    recorded["scorer_identity"] = dict(recorded["scorer_identity"],
                                       generation_id="different-generation")
    assert any("generation differs" in p for p in publish.check_panel_provenance(recorded))
