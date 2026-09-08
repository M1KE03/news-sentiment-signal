"""The three scorers (D7) and the hash-keyed cache that makes them run once.

Every scorer returns floats in [-1, 1] for a list of raw headline strings.
FinBERT inference is the expensive step in the project; `score_all` keys the
cache on `headline_id` and skips rows already scored, so a re-run costs nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

import config


class Scorer(Protocol):
    name: str

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Shape (n,), values in [-1, 1]."""


class LMScorer:
    """(pos - neg) / (pos + neg) over Loughran-McDonald word counts; 0 on no hits.

    Reads the master dictionary CSV from the SRAF site, which carries per-word
    `Negative` / `Positive` columns holding the year a word entered the list
    (0 = not in that list).
    """

    name = "lm"

    def __init__(self, dict_path: str | Path = config.LM_DICT_PATH):
        import re

        self._token = re.compile(r"[a-z']+")
        df = pd.read_csv(dict_path)
        cols = {c.lower(): c for c in df.columns}
        word_col = cols.get("word")
        if word_col is None:
            raise KeyError(f"{dict_path} has no 'Word' column; columns={list(df.columns)}")
        words = df[word_col].astype(str).str.lower()
        self.positive = frozenset(words[df[cols["positive"]].fillna(0).astype(int) > 0])
        self.negative = frozenset(words[df[cols["negative"]].fillna(0).astype(int) > 0])
        if not self.positive or not self.negative:
            raise ValueError("Loughran-McDonald dictionary loaded empty word lists")

    @classmethod
    def from_word_lists(cls, positive: set[str], negative: set[str]) -> "LMScorer":
        """Construct without a CSV. Used by the tests."""
        import re

        obj = cls.__new__(cls)
        obj._token = re.compile(r"[a-z']+")
        obj.positive = frozenset(w.lower() for w in positive)
        obj.negative = frozenset(w.lower() for w in negative)
        return obj

    def score(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(texts), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = self._token.findall(str(t).lower())
            pos = sum(1 for w in toks if w in self.positive)
            neg = sum(1 for w in toks if w in self.negative)
            total = pos + neg
            out[i] = 0.0 if total == 0 else (pos - neg) / total
        return out


class VaderScorer:
    """`vaderSentiment` compound score, already in [-1, 1]."""

    name = "vader"

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [self._analyzer.polarity_scores(str(t))["compound"] for t in texts],
            dtype=np.float32,
        )


class FinbertScorer:
    """`ProsusAI/finbert`, reported as P(positive) - P(negative).

    Truncation at 64 tokens is safe here: the inputs are headlines. Model is put
    in eval mode and run under `no_grad` -- inference only, no fine-tuning
    anywhere in this project.
    """

    name = "finbert"

    def __init__(
        self,
        model_name: str = config.FINBERT_MODEL,
        revision: str | None = config.FINBERT_REVISION,
        batch_size: int = config.FINBERT_BATCH_SIZE,
        max_length: int = config.FINBERT_MAX_LENGTH,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        kw = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kw)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Resolve label positions from the model config rather than assuming an
        # index order that a future revision could change.
        id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self._pos = next(i for i, l in id2label.items() if l.startswith("pos"))
        self._neg = next(i for i, l in id2label.items() if l.startswith("neg"))

    def score(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        out = np.empty(len(texts), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = [str(t) for t in texts[start : start + self.batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
            diff = (probs[:, self._pos] - probs[:, self._neg]).cpu().numpy()
            out[start : start + len(batch)] = diff
        return out


def build_scorers(names: Sequence[str] = config.SCORERS) -> list[Scorer]:
    factories = {"lm": LMScorer, "vader": VaderScorer, "finbert": FinbertScorer}
    return [factories[n]() for n in names]


def score_all(
    headlines_df: pd.DataFrame,
    cache_path: str | Path = config.SCORES_PARQUET,
    scorers: Sequence[Scorer] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Score every headline with every scorer, resuming from the cache.

    The cache is keyed on `headline_id`; a row already carrying a finite score
    for a scorer is never re-scored. This is what lets `run_all.py` reproduce
    every result in minutes without a GPU.
    """
    cache_path = Path(cache_path)
    cols = [f"score_{n}" for n in config.SCORERS]

    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
    else:
        cache = pd.DataFrame({"headline_id": pd.Series(dtype="object")})
    for c in cols:
        if c not in cache.columns:
            cache[c] = np.nan

    out = headlines_df[["headline_id", "text"]].merge(cache, on="headline_id", how="left")

    if scorers is None:
        needed = [n for n in config.SCORERS if out[f"score_{n}"].isna().any()]
        scorers = build_scorers(needed) if needed else []

    for scorer in scorers:
        col = f"score_{scorer.name}"
        todo = out[col].isna().to_numpy()
        if not todo.any():
            if verbose:
                print(f"[score_all] {scorer.name}: cached, nothing to do")
            continue
        texts = out.loc[todo, "text"].tolist()
        if verbose:
            print(f"[score_all] {scorer.name}: scoring {len(texts):,} headlines")
        out.loc[todo, col] = scorer.score(texts)

    scores = out[["headline_id"] + cols].copy()
    for c in cols:
        scores[c] = scores[c].astype("float32")

    merged = pd.concat([cache[~cache["headline_id"].isin(scores["headline_id"])], scores])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.reset_index(drop=True).to_parquet(cache_path, index=False)
    return scores.reset_index(drop=True)
