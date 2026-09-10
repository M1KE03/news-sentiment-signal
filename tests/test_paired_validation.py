"""Independent-label and paired group-bootstrap contracts, using invented labels."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src import validate


def provenance():
    return {"rubric_version": "v1", "rubric_sha256": "a" * 64, "annotator_role": "synthetic fixture",
            "instrument": "fixture", "label_source": "human", "blind_to_model_outputs": True,
            "independent_of_analyst": False, "scores_existed": False, "presentation_order_seed": 1,
            "deviations": [], "sessions": [{"start": "2026-09-09T10:00:00Z", "end": "2026-09-09T11:00:00Z"}]}


@pytest.fixture
def annotation_files(tmp_path):
    ids = [f"h{i}" for i in range(8)]
    split = pd.DataFrame({"headline_id": ids, "group_id": [f"g{i}" for i in range(8)],
                          "part": ["calibration"] * 4 + ["evaluation"] * 4})
    split.to_csv(tmp_path / "split_assignment.csv", index=False)
    pd.DataFrame({"headline_id": ids, "text": [f"Invented headline {i}" for i in range(8)]}).to_csv(tmp_path / "to_label_primary.csv", index=False)
    labels = pd.DataFrame({"headline_id": ids, "label": ["negative", "neutral", "positive", "unusable"] * 2,
                           "mixed": 0, "hard": 0, "notes": ""})
    labels.to_csv(tmp_path / "labels.csv", index=False)
    (tmp_path / "provenance.json").write_text(json.dumps(provenance()))
    return tmp_path


def load(folder, part="calibration", **kwargs):
    return validate.load_annotations(folder / "labels.csv", folder / "provenance.json", part=part, annotation_dir=folder, **kwargs)


def test_ingestion_joins_ids_without_touching_blank_evaluation(annotation_files):
    path = annotation_files / "labels.csv"
    labels = pd.read_csv(path).sample(frac=1, random_state=8)
    labels.loc[labels["headline_id"].isin(["h4", "h5", "h6", "h7"]), "label"] = ""
    labels.to_csv(path, index=False)
    result = load(annotation_files)
    assert result["headline_id"].tolist() == ["h0", "h1", "h2", "h3"]
    assert result["label"].tolist() == ["negative", "neutral", "positive", "unusable"]
    assert len(result.attrs["source_hashes"]["labels"]) == 64
    with pytest.raises(ValueError, match="blanks"):
        load(annotation_files, "evaluation")


@pytest.mark.parametrize("defect", ["duplicate", "unknown", "missing", "label", "flag", "text"])
def test_label_defects_refuse_ingestion(annotation_files, defect):
    path = annotation_files / "labels.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    if defect == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    elif defect == "unknown": frame.loc[0, "headline_id"] = "outsider"
    elif defect == "missing": frame = frame.iloc[1:]
    elif defect == "label": frame.loc[0, "label"] = "model-positive"
    elif defect == "flag": frame.loc[0, "hard"] = 2
    else: frame["text"] = "edited source"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError): load(annotation_files)


@pytest.mark.parametrize("field,value", [("label_source", "model"), ("blind_to_model_outputs", False),
                                         ("sessions", []), ("rubric_sha256", "short"), ("annotator_role", "")])
def test_unfinished_or_model_provenance_is_not_human_ground_truth(field, value):
    record = provenance(); record[field] = value
    with pytest.raises(ValueError): validate.validate_annotation_provenance(record)


def test_group_bootstrap_matches_row_level_reference_with_unequal_groups():
    truth = np.array(["negative", "negative", "neutral", "positive", "positive", "neutral"])
    a = np.array(["negative", "neutral", "neutral", "positive", "negative", "neutral"])
    b = np.array(["neutral", "neutral", "positive", "positive", "positive", "neutral"])
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    result = validate.paired_group_bootstrap(truth, a, b, groups, draws=100, seed=9)
    rng = np.random.default_rng(9)
    differences = []
    for _ in range(100):
        indices = np.concatenate([np.flatnonzero(groups == name) for name in rng.choice(["a", "b", "c"], size=3)])
        f1 = lambda pred: f1_score(truth[indices], pred[indices], labels=list(config.LABELS), average="macro", zero_division=0)
        differences.append([f1(a) - f1(b), accuracy_score(truth[indices], a[indices]) - accuracy_score(truth[indices], b[indices])])
    lo, hi = np.quantile(differences, [0.025, 0.975], axis=0)
    np.testing.assert_allclose([result["macro_f1_lo95"], result["accuracy_lo95"]], lo)
    np.testing.assert_allclose([result["macro_f1_hi95"], result["accuracy_hi95"]], hi)


def test_identical_predictions_have_zero_paired_uncertainty():
    truth = np.array(["positive", "negative", "neutral"] * 2)
    pred = np.array(["neutral"] * 6)
    result = validate.paired_group_bootstrap(truth, pred, pred, [0, 0, 1, 2, 3, 3], draws=100)
    assert all(result[key] == 0 for key in ("macro_f1_difference", "macro_f1_lo95", "macro_f1_hi95", "accuracy_difference", "accuracy_lo95", "accuracy_hi95"))


def test_calibration_cannot_accept_evaluation_or_missing_classes(annotation_files):
    labels = load(annotation_files)
    scores = pd.DataFrame({"headline_id": labels["headline_id"], "lm": [-1, 0, 1, 0], "vader": [-1, 0, 1, 0]})
    record = validate.calibrate_lexicons(labels, scores, grid=5)
    assert record["n_unusable"] == 1
    np.testing.assert_array_equal(validate.classify(np.array([-1, 0, 1]), record["thresholds"]["lm"]), ["negative", "neutral", "positive"])
    labels["part"] = "evaluation"
    with pytest.raises(ValueError, match="calibration rows only"): validate.calibrate_lexicons(labels, scores, grid=5)
    with pytest.raises(ValueError, match="all three classes"): validate.fit_thresholds(np.array([0, 1]), np.array(["neutral", "positive"]), grid=5)


def test_frozen_evaluation_requires_registry_and_counts_unusable(annotation_files, monkeypatch):
    monkeypatch.setattr(config, "ANNOTATION_DIR", annotation_files)
    cal = load(annotation_files)
    calibration = validate.calibrate_lexicons(cal, pd.DataFrame({"headline_id": cal["headline_id"], "lm": [-1, 0, 1, 0], "vader": [-1, 0, 1, 0]}), grid=5)
    labels = load(annotation_files, "evaluation")
    scores = pd.DataFrame({"headline_id": labels["headline_id"], "lm": [-1, 0, 1, 0], "vader": [0, 0, 1, 0],
                           "finbert_prediction": ["negative", "neutral", "positive", "neutral"]})
    monkeypatch.setattr(config, "VALIDATION_THRESHOLDS", None)
    with pytest.raises(ValueError, match="freeze"): validate.evaluate_frozen(labels, scores, calibration, draws=10)
    monkeypatch.setattr(config, "VALIDATION_THRESHOLDS", calibration["thresholds"])
    result = validate.evaluate_frozen(labels, scores.sample(frac=1, random_state=8), calibration, draws=100)
    assert result["n_unusable"] == 1 and result["n_usable"] == 3
    assert result["per_scorer"]["finbert"]["macro_f1"] == 1
    assert result["contrasts"][0]["macro_f1_difference"] == 0
    assert all("p" not in contrast for contrast in result["contrasts"])
    assert [c["role"] for c in result["contrasts"]] == ["primary", "descriptive-secondary", "descriptive-secondary"]
    assert "item independence" in result["mcnemar_supplementary"]["condition"]


def test_agreement_retains_primary_and_counts_unusable():
    a = pd.DataFrame({"headline_id": ["a", "b", "c"], "label": ["positive", "negative", "unusable"]})
    b = pd.DataFrame({"headline_id": ["b", "a", "c"], "label": ["neutral", "positive", "positive"]})
    before = a.copy()
    result = validate.annotation_agreement(a, b)
    assert result["n_unusable_either"] == 1 and result["raw_agreement"] == 0.5
    assert result["cohen_kappa"] == pytest.approx(1 / 3)
    pd.testing.assert_frame_equal(a, before)


# ------------------------- R13a readiness: the pilot handoff must actually work


def pilot_paths():
    import config as _c
    return Path(_c.ANNOTATION_DIR)


def test_the_pilot_provenance_template_exists_and_is_rejected_until_filled():
    """The refusal is the feature. `validate_annotation_provenance` never
    manufactures a human declaration, so the shipped template must fail."""
    import json as _json

    path = pilot_paths() / "provenance_pilot.json"
    assert path.exists(), "the ingestion code reads JSON; a template must be provided"
    record = _json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        validate.validate_annotation_provenance(record)


def test_the_template_carries_the_verifiable_facts_already():
    """Only human declarations are left blank; facts about the draw are filled."""
    import hashlib
    import json as _json
    import config as _c

    record = _json.loads((pilot_paths() / "provenance_pilot.json").read_text(encoding="utf-8"))
    assert record["rubric_version"] == "v1"
    assert record["presentation_order_seed"] == 20260831
    assert record["instrument"] == "pilot_worksheet.csv"
    actual = hashlib.sha256(Path("docs/validation-protocol.md").read_bytes()).hexdigest()
    assert record["rubric_sha256"] == actual, (
        "the recorded rubric hash no longer matches the protocol file; a "
        "mid-annotation rubric change is a protocol change and must be logged"
    )


def test_the_filled_template_passes_and_ingests_the_real_pilot_worksheet():
    """End-to-end dry run on the exact file the annotator will hand back.

    Synthetic labels, discarded. This exists so a schema surprise surfaces now
    rather than after someone has spent half an hour labelling.
    """
    import json as _json
    import shutil
    import tempfile

    import config as _c

    folder = pilot_paths()
    tmp = Path(tempfile.mkdtemp())
    for f in folder.glob("*.csv"):
        shutil.copy(f, tmp / f.name)

    sheet = pd.read_csv(folder / "pilot_worksheet.csv", dtype=str,
                        keep_default_na=False, encoding="utf-8-sig")
    cycle = ["positive", "negative", "neutral"]
    sheet["label"] = [cycle[i % 3] for i in range(len(sheet))]
    sheet["mixed"] = "0"
    sheet["hard"] = "0"
    sheet["notes"] = ""
    labels = tmp / "labels_pilot.csv"
    sheet.to_csv(labels, index=False, encoding="utf-8-sig")   # Excel's default on Windows

    record = _json.loads((folder / "provenance_pilot.json").read_text(encoding="utf-8"))
    record.update(annotator_role="project analyst", label_source="human",
                  blind_to_model_outputs=True, independent_of_analyst=False,
                  scores_existed=True,
                  sessions=[{"start": "2026-09-10T14:00:00Z", "end": "2026-09-10T14:35:00Z"}])
    prov = tmp / "provenance_pilot.json"
    prov.write_text(_json.dumps(record), encoding="utf-8")

    ids = pd.read_csv(folder / "pilot_items.csv", dtype=str)["headline_id"].tolist()
    out = validate.load_annotations(labels, prov, part="calibration",
                                    annotation_dir=tmp, expected_ids=ids)
    assert len(out) == 60
    assert set(out["label"]) <= {"positive", "negative", "neutral", "unusable"}
    assert out.attrs["provenance"]["scores_existed"] is True


def test_a_worksheet_with_a_blank_label_is_refused():
    """A missed row must stop ingestion, not be silently dropped."""
    import json as _json
    import shutil
    import tempfile

    folder = pilot_paths()
    tmp = Path(tempfile.mkdtemp())
    for f in folder.glob("*.csv"):
        shutil.copy(f, tmp / f.name)

    sheet = pd.read_csv(folder / "pilot_worksheet.csv", dtype=str,
                        keep_default_na=False, encoding="utf-8-sig")
    sheet["label"] = "neutral"
    sheet.loc[7, "label"] = ""          # one row missed
    sheet["mixed"] = "0"
    sheet["hard"] = "0"
    sheet["notes"] = ""
    labels = tmp / "labels_pilot.csv"
    sheet.to_csv(labels, index=False, encoding="utf-8-sig")

    record = _json.loads((folder / "provenance_pilot.json").read_text(encoding="utf-8"))
    record.update(annotator_role="project analyst", label_source="human",
                  blind_to_model_outputs=True, independent_of_analyst=False,
                  scores_existed=True,
                  sessions=[{"start": "2026-09-10T14:00:00Z", "end": "2026-09-10T14:35:00Z"}])
    prov = tmp / "provenance_pilot.json"
    prov.write_text(_json.dumps(record), encoding="utf-8")

    ids = pd.read_csv(folder / "pilot_items.csv", dtype=str)["headline_id"].tolist()
    with pytest.raises(ValueError, match="blanks are incomplete"):
        validate.load_annotations(labels, prov, part="calibration",
                                  annotation_dir=tmp, expected_ids=ids)
