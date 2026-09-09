"""Act 1: the PhraseBank comparison (D13, D14).

An independent branch of the project -- it shares only `scoring.py` with Act 2.
Neutral bands for the two lexicons are fitted on a 20% stratified split and
every reported number comes from the held-out 80%. FinBERT is never fitted.
"""

from __future__ import annotations

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
    cand = np.linspace(-1.0, 1.0, grid)
    best, best_f1 = (0.0, 0.0), -1.0
    for lo in cand:
        for hi in cand[cand >= lo]:
            f1 = f1_score(labels, classify(scores, (lo, hi)), average="macro", labels=LABELS)
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
    """Table 1's core rows, plus the decomposition of the gain.

    The three scorers form a ladder -- generic lexicon, domain lexicon, domain
    transformer -- so the total gain splits into 'domain vocabulary matters'
    (VADER -> LM) and 'context matters beyond vocabulary' (LM -> FinBERT).
    """
    rows = [
        {"scorer": name, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "n": r["n"]}
        for name, r in results.items()
    ]
    out = pd.DataFrame(rows).set_index("scorer")
    if {"vader", "lm", "finbert"} <= set(out.index):
        total = out.loc["finbert", "macro_f1"] - out.loc["vader", "macro_f1"]
        out.attrs["gain_vocabulary"] = out.loc["lm", "macro_f1"] - out.loc["vader", "macro_f1"]
        out.attrs["gain_context"] = out.loc["finbert", "macro_f1"] - out.loc["lm", "macro_f1"]
        out.attrs["gain_total"] = total
    return out.reset_index()
