"""Act 1: independent human labels, calibration, and paired group uncertainty.

PhraseBank helpers remain supplementary legacy utilities, not the primary study.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import config

LABELS = list(config.LABELS)


def load_phrasebank(subset: str = config.PHRASEBANK_PRIMARY_SUBSET) -> pd.DataFrame:
    """Financial PhraseBank (Malo et al. 2014) -> columns `text`, `label`.

    Labels are mapped to the strings in `config.LABELS` so that lexicon
    thresholds and FinBERT predictions live in one label space.
    """
    from datasets import load_dataset

    ds = load_dataset("financial_phrasebank", subset, trust_remote_code=True)["train"]
    names = ds.features["label"].names  # ["negative", "neutral", "positive"]
    return pd.DataFrame(
        {"text": ds["sentence"], "label": [names[i] for i in ds["label"]], "subset": subset}
    )


def split(df: pd.DataFrame, seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified fit/eval split. `fit` is 20% and exists only for thresholds."""
    fit, evalset = train_test_split(
        df,
        train_size=config.THRESHOLD_FIT_FRACTION,
        stratify=df["label"],
        random_state=seed,
        shuffle=True,
    )
    return fit.reset_index(drop=True), evalset.reset_index(drop=True)


def classify(scores: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    """Continuous score -> label, using a neutral band (lo, hi]."""
    lo, hi = thresholds
    out = np.full(len(scores), "neutral", dtype=object)
    out[np.asarray(scores) <= lo] = "negative"
    out[np.asarray(scores) > hi] = "positive"
    return out


def fit_thresholds(
    scores: np.ndarray, labels: np.ndarray, grid: int = 81
) -> tuple[float, float]:
    """Choose the neutral band maximising macro-F1 on the fit split only.

    A grid search over (lo, hi) with lo <= hi on [-1, 1]. This is the only place
    in the project where anything is fitted to labels, and it never sees the
    evaluation 80%.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if scores.ndim != 1 or scores.shape != labels.shape or not len(scores):
        raise ValueError("threshold calibration requires nonempty paired scores and labels")
    if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1):
        raise ValueError("calibration scores must be finite and in [-1, 1]")
    if set(labels) != set(LABELS):
        raise ValueError("calibration must contain all three classes; review class balance")
    if not isinstance(grid, int) or grid < 2:
        raise ValueError("threshold grid must contain at least two points")
    cand = np.linspace(-1.0, 1.0, grid)
    best, best_f1 = (0.0, 0.0), -1.0
    for lo in cand:
        for hi in cand[cand >= lo]:
            f1 = f1_score(labels, classify(scores, (lo, hi)), average="macro", labels=LABELS, zero_division=0)
            if f1 > best_f1:
                best_f1, best = f1, (float(lo), float(hi))
    return best


def predictions_for(scorer, texts, thresholds: tuple[float, float] | None = None):
    """Class labels for one scorer, by the route that scorer actually supports (B12).

    A classifier (FinBERT) predicts by argmax over its own class probabilities.
    A lexicon predicts by a neutral band fitted on the calibration split. The
    two routes are not interchangeable and this function refuses to swap them:

    * thresholding a classifier would discard the neutral probability -- which
      is exactly what decides the label when `P(pos) - P(neg)` is ambiguous --
      and would additionally impose a tuned band on the one scorer that needs
      no tuning, quietly removing the asymmetry the report is meant to state;
    * argmaxing a lexicon is not defined: it produces a single number, not a
      distribution over classes.
    """
    if hasattr(scorer, "predict"):
        if thresholds is not None:
            raise ValueError(
                f"{getattr(scorer, 'name', scorer)!r} is a classifier: it predicts by "
                "argmax over its own class probabilities. Passing a neutral band "
                "would discard the neutral probability and impose a fitted "
                "threshold on a scorer that requires none."
            )
        return np.asarray(scorer.predict(texts))

    if thresholds is None:
        raise ValueError(
            f"{getattr(scorer, 'name', scorer)!r} is a lexicon: it needs a neutral "
            "band fitted on the calibration split (fit_thresholds), because a "
            "continuous score alone does not determine a class."
        )
    return classify(np.asarray(scorer.score(texts)), thresholds)


def evaluate(
    scores: np.ndarray, labels: np.ndarray, thresholds: tuple[float, float] | None = None
) -> dict:
    """Accuracy, macro-F1 and the confusion matrix on the evaluation split.

    Pass `thresholds=None` when `scores` are already labels (FinBERT's argmax).
    """
    pred = np.asarray(scores) if thresholds is None else classify(scores, thresholds)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", labels=LABELS)),
        "confusion": confusion_matrix(labels, pred, labels=LABELS),
        "labels": LABELS,
        "thresholds": thresholds,
        "n": int(len(labels)),
        "pred": pred,
    }


def mcnemar_test(pred_a, pred_b, truth) -> tuple[float, float]:
    """D14. Exact McNemar on paired predictions over the same sentences.

    Returns (statistic, p). The classifiers score the *same* sentences, so the
    predictions are paired; McNemar tests the asymmetry of the disagreement
    cells. Comparing two accuracies as though they came from independent samples
    discards the pairing and gets the variance wrong.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    a_ok = np.asarray(pred_a) == np.asarray(truth)
    b_ok = np.asarray(pred_b) == np.asarray(truth)
    table = np.array(
        [
            [int(np.sum(a_ok & b_ok)), int(np.sum(a_ok & ~b_ok))],
            [int(np.sum(~a_ok & b_ok)), int(np.sum(~a_ok & ~b_ok))],
        ]
    )
    res = mcnemar(table, exact=config.MCNEMAR_EXACT)
    return float(res.statistic), float(res.pvalue)


def disagreement_cells(pred_a, pred_b, truth) -> dict:
    """The b and c cells behind the McNemar p-value, reported alongside it."""
    a_ok = np.asarray(pred_a) == np.asarray(truth)
    b_ok = np.asarray(pred_b) == np.asarray(truth)
    return {
        "b_a_only_correct": int(np.sum(a_ok & ~b_ok)),
        "c_b_only_correct": int(np.sum(~a_ok & b_ok)),
        "both_correct": int(np.sum(a_ok & b_ok)),
        "both_wrong": int(np.sum(~a_ok & ~b_ok)),
    }


def ladder_table(results: dict[str, dict]) -> pd.DataFrame:
    """Table 1's descriptive rows; scorer differences do not identify mechanisms."""
    rows = [
        {"scorer": name, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "n": r["n"]}
        for name, r in results.items()
    ]
    out = pd.DataFrame(rows).set_index("scorer")
    return out.reset_index()


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique_ids(frame: pd.DataFrame, name: str) -> None:
    if "headline_id" not in frame or frame["headline_id"].isna().any() or frame["headline_id"].eq("").any() or frame["headline_id"].duplicated().any():
        raise ValueError(f"{name} requires unique, nonempty headline_id values")


def validate_annotation_provenance(record: dict) -> None:
    """Validate human declarations; never manufacture or infer their truth."""
    for key in ("rubric_version", "rubric_sha256", "annotator_role", "instrument"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip() or "TO BE COMPLETED" in value:
            raise ValueError(f"annotation provenance requires {key}")
    digest = record["rubric_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("rubric_sha256 must be a full lowercase SHA-256")
    if record.get("label_source") != "human" or record.get("blind_to_model_outputs") is not True:
        raise ValueError("independent validation requires human labels and confirmed model blindness")
    for key in ("independent_of_analyst", "scores_existed"):
        if type(record.get(key)) is not bool:
            raise ValueError(f"annotation provenance requires boolean {key}")
    if type(record.get("presentation_order_seed")) is not int or not isinstance(record.get("deviations"), list):
        raise ValueError("annotation provenance requires presentation_order_seed and deviations list")
    sessions = record.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("annotation provenance requires actual annotation sessions")
    for session in sessions:
        start, end = pd.Timestamp(session["start"]), pd.Timestamp(session["end"])
        if pd.isna(start) or pd.isna(end) or start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError("annotation sessions require ordered timestamps with time zones")


def load_annotations(labels_path, provenance_path, *, part: str, annotation_dir=None,
                     expected_ids=None) -> pd.DataFrame:
    """Load one frozen part (or declared pilot/subset) by ID, with human provenance.

    Blank evaluation answers are not inspected while loading calibration. A
    worksheet's optional text column is checked against its blind source.
    """
    from src.annotate import load_split
    folder = Path(config.ANNOTATION_DIR if annotation_dir is None else annotation_dir)
    if part not in {"calibration", "evaluation"}:
        raise ValueError("part must be calibration or evaluation")
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8-sig"))
    validate_annotation_provenance(provenance)
    assignment = load_split(folder)
    if assignment[["headline_id", "group_id", "part"]].isna().any().any():
        raise ValueError("frozen split has missing identifiers or parts")
    required = assignment.loc[assignment["part"] == part]
    if expected_ids is not None:
        requested = list(expected_ids)
        if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(required["headline_id"]):
            raise ValueError("requested IDs must be a unique nonempty subset of the declared part")
        required = required[required["headline_id"].isin(requested)]
    raw = pd.read_csv(labels_path, keep_default_na=False, dtype=str)
    _unique_ids(raw, "labels")
    if not set(raw["headline_id"]) <= set(assignment["headline_id"]):
        raise ValueError("labels contain IDs outside the frozen sample")
    needed = {"label", "mixed", "hard", "notes"}
    if not needed <= set(raw):
        raise ValueError(f"labels require columns {sorted(needed)}")
    if not set(required["headline_id"]) <= set(raw["headline_id"]):
        raise ValueError("labels are incomplete for the requested part/subset")
    selected = required.merge(raw, on="headline_id", validate="one_to_one", how="left")
    if not selected["label"].isin([*LABELS, "unusable"]).all():
        raise ValueError("labels must be negative, neutral, positive or unusable; blanks are incomplete")
    for key in ("mixed", "hard"):
        if not selected[key].isin(["0", "1"]).all():
            raise ValueError(f"{key} must be 0 or 1")
        selected[key] = selected[key].astype(int)
    blind = pd.read_csv(folder / "to_label_primary.csv", keep_default_na=False, dtype=str)
    _unique_ids(blind, "blind export")
    source = blind.set_index("headline_id")["text"].reindex(selected["headline_id"])
    if source.isna().any() or ("text" in selected and not np.array_equal(selected["text"], source)):
        raise ValueError("worksheet text differs from the frozen blind export")
    selected["text"] = source.to_numpy()
    selected.attrs["provenance"] = provenance
    selected.attrs["source_hashes"] = {"labels": _sha256(labels_path), "provenance": _sha256(provenance_path),
                                       "split": _sha256(folder / "split_assignment.csv")}
    return selected


def _join_measurements(labels: pd.DataFrame, measurements: pd.DataFrame) -> pd.DataFrame:
    _unique_ids(labels, "labels")
    _unique_ids(measurements, "measurements")
    if set(labels["headline_id"]) != set(measurements["headline_id"]):
        raise ValueError("measurements must cover exactly the requested label IDs")
    overlap = (set(labels) & set(measurements)) - {"headline_id"}
    if overlap:
        raise ValueError(f"measurements collide with label columns: {sorted(overlap)}")
    return labels.merge(measurements, on="headline_id", validate="one_to_one", how="left")


def calibrate_lexicons(labels: pd.DataFrame, scores: pd.DataFrame, *, grid: int = 81) -> dict:
    """Return a calibration record; register its thresholds in config before evaluation."""
    if not labels["part"].eq("calibration").all() or labels.empty:
        raise ValueError("threshold fitting accepts calibration rows only")
    joined = _join_measurements(labels, scores)
    usable = joined[joined["label"] != "unusable"]
    thresholds = {sc: list(fit_thresholds(usable[sc].to_numpy(), usable["label"].to_numpy(), grid))
                  for sc in ("lm", "vader")}
    return {"thresholds": thresholds, "grid": grid, "tie_rule": "first ascending (lo, hi) grid pair",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_ids": sorted(labels["headline_id"].tolist()),
            "source_hashes": labels.attrs.get("source_hashes", {}),
            "n_calibration": len(labels), "n_unusable": int((joined["label"] == "unusable").sum())}


def paired_group_bootstrap(truth, pred_a, pred_b, groups, *, draws: int = 10000,
                           seed: int = config.SEED) -> dict:
    """Pointwise percentile intervals for macro-F1 and accuracy differences.

    Confusion counts are aggregated by group, then whole groups are sampled
    with replacement. All three classes stay in macro-F1 even when a draw
    omits one (undefined class F1 contributes zero).
    """
    arrays = [np.asarray(a) for a in (truth, pred_a, pred_b, groups)]
    if any(a.ndim != 1 or len(a) != len(arrays[0]) for a in arrays) or not len(arrays[0]):
        raise ValueError("bootstrap inputs must be nonempty paired vectors")
    truth, pred_a, pred_b, groups = arrays
    if any(not set(a) <= set(LABELS) for a in (truth, pred_a, pred_b)) or pd.isna(groups).any():
        raise ValueError("bootstrap requires three-class labels and nonmissing groups")
    if type(draws) is not int or draws < 2:
        raise ValueError("draws must be an integer >= 2")
    codes, names = pd.factorize(groups, sort=True)
    if len(names) < 2:
        raise ValueError("at least two article groups are needed for uncertainty")
    encode = {name: i for i, name in enumerate(LABELS)}
    y = np.array([encode[v] for v in truth])
    matrices = np.zeros((len(names), 2, 3, 3), dtype=np.int64)
    for scorer, pred in enumerate((pred_a, pred_b)):
        np.add.at(matrices, (codes, scorer, y, [encode[v] for v in pred]), 1)

    def differences(cm):
        diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
        denom = cm.sum(axis=-1) + cm.sum(axis=-2)
        f1 = np.divide(2 * diagonal, denom, out=np.zeros_like(diagonal, dtype=float), where=denom != 0).mean(axis=-1)
        accuracy = diagonal.sum(axis=-1) / cm.sum(axis=(-2, -1))
        return np.array([f1[0] - f1[1], accuracy[0] - accuracy[1]])

    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 2))
    for b in range(draws):
        samples[b] = differences(matrices[rng.integers(len(names), size=len(names))].sum(axis=0))
    point = differences(matrices.sum(axis=0))
    low, high = np.quantile(samples, [0.025, 0.975], axis=0, method="linear")
    return {"macro_f1_difference": float(point[0]), "macro_f1_lo95": float(low[0]), "macro_f1_hi95": float(high[0]),
            "accuracy_difference": float(point[1]), "accuracy_lo95": float(low[1]), "accuracy_hi95": float(high[1]),
            "n": len(truth), "n_groups": len(names), "draws": draws, "seed": seed,
            "interval_scope": "pointwise", "resampling_unit": "article_group"}


def evaluate_independent(labels: pd.DataFrame, predictions: pd.DataFrame, *, draws: int = 10000) -> dict:
    """Paired evaluation metrics from frozen evaluation IDs and class predictions."""
    if labels.empty or not labels["part"].eq("evaluation").all():
        raise ValueError("reported metrics require evaluation rows only")
    joined = _join_measurements(labels, predictions)
    if not joined["label"].isin([*LABELS, "unusable"]).all():
        raise ValueError("invalid human labels")
    for sc in config.SCORERS:
        if not joined[sc].isin(LABELS).all():
            raise ValueError(f"{sc} must supply actual three-class predictions")
    usable = joined[joined["label"] != "unusable"]
    if usable.empty:
        raise ValueError("no usable evaluation labels")
    per_scorer = {}
    from sklearn.metrics import precision_recall_fscore_support
    for sc in config.SCORERS:
        precision, recall, f1, support = precision_recall_fscore_support(usable["label"], usable[sc], labels=LABELS, zero_division=0)
        per_scorer[sc] = {"accuracy": float(accuracy_score(usable["label"], usable[sc])), "macro_f1": float(f1.mean()),
                          "confusion": confusion_matrix(usable["label"], usable[sc], labels=LABELS).tolist(),
                          "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                                 "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(LABELS)}}
    contrasts = []
    for a, b in (("finbert", "lm"), ("finbert", "vader"), ("lm", "vader")):
        result = paired_group_bootstrap(usable["label"], usable[a], usable[b], usable["group_id"], draws=draws)
        result.update({"contrast": f"{a}-{b}", "role": "primary" if b == "lm" else "descriptive-secondary"})
        contrasts.append(result)
    sizes = usable.groupby("group_id").size()
    stat, p = mcnemar_test(usable["finbert"], usable["lm"], usable["label"])
    return {"n_requested": len(labels), "n_usable": len(usable), "n_unusable": len(labels) - len(usable),
            "class_balance": {label: int((usable["label"] == label).sum()) for label in LABELS},
            "per_scorer": per_scorer, "contrasts": contrasts,
            "group_sizes": {"1": int((sizes == 1).sum()), "2": int((sizes == 2).sum()), "3+": int((sizes >= 3).sum())},
            "mcnemar_supplementary": {"statistic": stat, "p": p, **disagreement_cells(usable["finbert"], usable["lm"], usable["label"]),
                                      "condition": "valid only under item independence, which this design does not assert"}}


def evaluate_frozen(labels: pd.DataFrame, scores: pd.DataFrame, calibration_record: dict, *, draws: int = 10000) -> dict:
    """Evaluation entry point: require thresholds already registered in config.

    Scores contain lm/vader scalars and finbert_prediction from FinBERT.predict;
    a FinBERT tone scalar cannot substitute for its class argmax.
    """
    from src.annotate import load_split
    validate_annotation_provenance(labels.attrs.get("provenance", {}))
    assignment = load_split()
    calibration_ids = set(assignment.loc[assignment["part"] == "calibration", "headline_id"])
    evaluation_ids = set(assignment.loc[assignment["part"] == "evaluation", "headline_id"])
    if set(calibration_record.get("calibration_ids", [])) != calibration_ids or set(labels["headline_id"]) != evaluation_ids:
        raise ValueError("final evaluation requires the complete frozen calibration/evaluation parts")
    registered = config.VALIDATION_THRESHOLDS
    thresholds = calibration_record.get("thresholds")
    if registered is None or thresholds != registered or set(registered) != {"lm", "vader"}:
        raise ValueError("freeze the calibration thresholds in config.VALIDATION_THRESHOLDS before evaluation")
    if set(calibration_record.get("calibration_ids", [])) & set(labels["headline_id"]):
        raise ValueError("calibration/evaluation IDs overlap")
    if not calibration_record.get("calibration_ids"):
        raise ValueError("calibration record requires its source IDs")
    joined = _join_measurements(labels, scores)
    predictions = joined[["headline_id"]].copy()
    for sc in ("lm", "vader"):
        values = joined[sc].to_numpy(dtype=float)
        band = np.asarray(registered[sc], dtype=float)
        if band.shape != (2,) or not np.isfinite(band).all() or not -1 <= band[0] <= band[1] <= 1:
            raise ValueError("invalid frozen neutral band")
        if not np.isfinite(values).all() or np.any(np.abs(values) > 1):
            raise ValueError("evaluation scores must be finite and in [-1, 1]")
        predictions[sc] = classify(values, tuple(band))
    predictions["finbert"] = joined["finbert_prediction"]
    return evaluate_independent(labels, predictions, draws=draws)


def annotation_agreement(primary_labels: pd.DataFrame, second_labels: pd.DataFrame) -> dict:
    """Report agreement on matched usable items; do not adjudicate either label."""
    from sklearn.metrics import cohen_kappa_score
    _unique_ids(primary_labels, "primary labels")
    _unique_ids(second_labels, "second labels")
    if not set(second_labels["headline_id"]) <= set(primary_labels["headline_id"]):
        raise ValueError("second annotator IDs must be a subset of the primary annotation")
    for frame in (primary_labels, second_labels):
        if not frame["label"].isin([*LABELS, "unusable"]).all():
            raise ValueError("invalid annotation labels")
    paired = primary_labels[["headline_id", "label"]].merge(second_labels[["headline_id", "label"]], on="headline_id", suffixes=("_a", "_b"), validate="one_to_one")
    usable = paired[(paired["label_a"] != "unusable") & (paired["label_b"] != "unusable")]
    if usable.empty:
        raise ValueError("no jointly usable double-annotated items")
    # Kappa is undefined when both annotators assign only the same single class.
    constant_same = len(set(usable["label_a"]) | set(usable["label_b"])) == 1
    kappa = None if constant_same else float(cohen_kappa_score(usable["label_a"], usable["label_b"], labels=LABELS))
    return {"n_matched": len(paired), "n_jointly_usable": len(usable), "n_unusable_either": len(paired) - len(usable),
            "cohen_kappa": kappa, "kappa_undefined": constant_same,
            "raw_agreement": float((usable["label_a"] == usable["label_b"]).mean()),
            "adjudication": "none; primary labels retained"}
