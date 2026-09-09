"""The three scorers (D7) and the hash-keyed cache that makes them run once.

Every scorer returns floats in [-1, 1] for a list of raw headline strings.
FinBERT inference is the expensive step in the project; `score_all` keys the
cache on `headline_id` and skips rows already scored, so a re-run costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

import config


class Scorer(Protocol):
    """The continuous path: one number per headline, used by Act 2."""

    name: str

    @property
    def fingerprint(self) -> dict:
        """Nonempty JSON-compatible identity required for cached scoring."""

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Shape (n,), values in [-1, 1]."""


class Classifier(Protocol):
    """The discrete path: a class label per headline, used by Act 1 (B12).

    The two paths are separate because they answer different questions and the
    continuous score does not determine the label. `P(pos) - P(neg)` is the same
    for `(0.5, 0.1, 0.4)` and `(0.45, 0.05, 0.5)`, yet the first is positive by
    argmax and the second neutral: the neutral mass, which the difference throws
    away, is what decides. Act 1 needs actual predictions; Act 2 needs the
    continuous measurement.
    """

    name: str
    labels: tuple[str, ...]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Shape (n, 3), columns ordered as `config.LABELS`, rows summing to 1."""

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Shape (n,), values drawn from `config.LABELS`."""


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

    @property
    def fingerprint(self) -> dict:
        """Identity of the measurement this scorer produces (B13).

        The word lists themselves are hashed, not just the recorded version
        string: a dictionary swapped without updating `config.LM_DICT_VERSION`
        would otherwise reuse scores from a different measurement.
        """
        digest = hashlib.sha1(
            ("|".join(sorted(self.positive)) + "#" + "|".join(sorted(self.negative)))
            .encode("utf-8")
        ).hexdigest()
        return {
            "scorer": "lm",
            "dict_version": config.LM_DICT_VERSION,
            "wordlist_sha1": digest,
            "n_positive": len(self.positive),
            "n_negative": len(self.negative),
        }

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

    @property
    def fingerprint(self) -> dict:
        import vaderSentiment

        return {
            "scorer": "vader",
            "version": getattr(vaderSentiment, "__version__", "unknown"),
            "lexicon_size": len(self._analyzer.lexicon),
        }

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
        self.revision = revision
        kw = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kw)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Resolve label positions from the model config rather than assuming an
        # index order. This checkpoint publishes {0: positive, 1: negative,
        # 2: neutral} -- NOT the conventional negative/neutral/positive -- so
        # any code indexing by position silently inverts every score.
        id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self._pos = next(i for i, l in id2label.items() if l.startswith("pos"))
        self._neg = next(i for i, l in id2label.items() if l.startswith("neg"))
        self._neu = next(i for i, l in id2label.items() if l.startswith("neu"))
        self.labels = tuple(config.LABELS)
        # Column order for predict_proba: config.LABELS, not the model's order.
        self._col = {"negative": self._neg, "neutral": self._neu, "positive": self._pos}

        if config.FINBERT_ID2LABEL is not None:
            published = {int(k): v for k, v in config.FINBERT_ID2LABEL.items()}
            if published != id2label:
                raise RuntimeError(
                    "FinBERT label order differs from the pinned one: checkpoint "
                    f"says {id2label}, config.FINBERT_ID2LABEL says {published}. "
                    "Scores would flip sign. Re-verify the checkpoint before use."
                )

    @property
    def fingerprint(self) -> dict:
        """Every setting that changes the number, including truncation length."""
        return {
            "scorer": "finbert",
            "model": self.model.config._name_or_path,
            "revision": self.revision,
            "max_length": self.max_length,
            "id2label": {int(k): v.lower() for k, v in self.model.config.id2label.items()},
        }

    def _probs(self, texts: Sequence[str]) -> np.ndarray:
        """Softmax probabilities in the model's own column order."""
        torch = self._torch
        out = np.empty((len(texts), 3), dtype=np.float32)
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
            out[start : start + len(batch)] = probs.cpu().numpy()
        return out

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """P(positive) - P(negative), the continuous measurement for Act 2."""
        p = self._probs(texts)
        return (p[:, self._pos] - p[:, self._neg]).astype(np.float32)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Full class probabilities, columns ordered as `config.LABELS` (B12).

        Re-ordered into the project's label order so that no downstream caller
        has to know the checkpoint's published order -- which is the whole
        hazard this method exists to contain.
        """
        p = self._probs(texts)
        return np.stack([p[:, self._col[l]] for l in config.LABELS], axis=1)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Argmax class label. Act 1's prediction; never derived from `score`.

        The neutral probability is retained and decisive here. Thresholding the
        continuous score instead would both discard it and impose on FinBERT the
        tuned neutral band that only the lexicons need.
        """
        proba = self.predict_proba(texts)
        return np.asarray(config.LABELS, dtype=object)[proba.argmax(axis=1)]


def build_scorers(names: Sequence[str] = config.SCORERS) -> list[Scorer]:
    factories = {"lm": LMScorer, "vader": VaderScorer, "finbert": FinbertScorer}
    return [factories[n]() for n in names]


def meta_path(cache_path: str | Path) -> Path:
    """Sidecar holding the fingerprint of whatever produced each cached column."""
    cache_path = Path(cache_path)
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file and replace, so an interrupted write cannot truncate.

    A long scoring pass checkpoints repeatedly; a partial parquet left behind by
    a kill signal would be worse than no cache at all, because it would look
    loadable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _canonical_fingerprint(fingerprint: dict) -> dict:
    """Compare the representation actually persisted in JSON (R01a).

    JSON turns integer object keys into strings and tuples into lists. Apply
    that conversion before comparison as well as writing, without a lossy
    default=str fallback for unsupported objects.
    """
    if not isinstance(fingerprint, dict) or not fingerprint:
        raise ValueError("fingerprint must be a nonempty JSON-compatible dictionary")
    return json.loads(json.dumps(fingerprint, allow_nan=False))


def load_cache(cache_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Cached scores and their recorded identities; refuse unknown provenance.

    This checks identity presence even on the default completed-cache path.
    Comparing those identities to current artifacts is separate (R01b).
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

    mp = meta_path(cache_path)
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    if not isinstance(meta, dict) or not isinstance(meta.get("fingerprints", {}), dict):
        raise IncompatibleCache(f"invalid fingerprint metadata in {mp}")
    fingerprints = meta.get("fingerprints", {})
    for name in config.SCORERS:
        populated = int(cache[f"score_{name}"].notna().sum())
        recorded = fingerprints.get(name)
        if populated and (not isinstance(recorded, dict) or not recorded):
            raise IncompatibleCache(
                f"cached {name!r} has {populated:,} values with a missing or invalid "
                f"fingerprint in {mp}. Their provenance cannot be inferred. "
                "Restore the original metadata or rebuild into a new cache path; "
                "on_fingerprint_change='rescore' cannot establish unknown provenance."
            )
    return cache, meta


class IncompatibleCache(RuntimeError):
    """Raised when cached scores were produced by a different measurement (B13)."""


def score_all(
    headlines_df: pd.DataFrame,
    cache_path: str | Path = config.SCORES_PARQUET,
    scorers: Sequence[Scorer] | None = None,
    verbose: bool = True,
    checkpoint_every: int = 50_000,
    on_fingerprint_change: str = "raise",
) -> pd.DataFrame:
    """Score every headline with every scorer, resuming from the cache.

    **Provenance (B13).** Each scorer publishes a `fingerprint` naming everything
    that changes the number it produces -- model revision and truncation length
    for FinBERT, a hash of the actual word lists for LM. The fingerprint is
    stored beside the cache. If it no longer matches, the cached column was
    produced by a *different measurement*, and reusing it would silently mix two
    definitions inside one column. That raises `IncompatibleCache` by default;
    `on_fingerprint_change="rescore"` discards the stale column and recomputes
    it. Populated columns without recorded identity are refused at load time;
    rebuild into a new cache path rather than inventing their provenance.

    **Resumption (B14).** Scoring proceeds in batches of `checkpoint_every` rows
    and the cache is written atomically after each. An interrupted pass resumes
    from the last checkpoint and produces the same result as an uninterrupted
    one. This matters because the FinBERT pass over the assembled corpus is
    hours long and the plan's time budget assumes it is not repeated.
    """
    cache_path = Path(cache_path)
    cols = [f"score_{n}" for n in config.SCORERS]
    cache, meta = load_cache(cache_path)
    fingerprints = dict(meta.get("fingerprints", {}))

    out = headlines_df[["headline_id", "text"]].merge(cache, on="headline_id", how="left")

    if scorers is None:
        needed = [n for n in config.SCORERS if out[f"score_{n}"].isna().any()]
        scorers = build_scorers(needed) if needed else []

    for scorer in scorers:
        name, col = scorer.name, f"score_{scorer.name}"
        current = _canonical_fingerprint(getattr(scorer, "fingerprint", None))
        stored = fingerprints.get(name)
        if stored is not None:
            stored = _canonical_fingerprint(stored)

        if stored is not None and current is not None and stored != current:
            already = int(out[col].notna().sum())
            message = (
                f"cached {name!r} scores were produced by a different measurement.\n"
                f"  cached : {stored}\n"
                f"  current: {current}\n"
                f"  {already:,} cached values affected."
            )
            if on_fingerprint_change == "raise":
                raise IncompatibleCache(
                    message
                    + "\nReusing them would mix two definitions inside one column. "
                    "Re-run with on_fingerprint_change='rescore' to discard and "
                    "recompute, or restore the previous artifact."
                )
            if on_fingerprint_change != "rescore":
                raise ValueError(
                    f"on_fingerprint_change must be 'raise' or 'rescore', "
                    f"got {on_fingerprint_change!r}"
                )
            if verbose:
                print(f"[score_all] {message}\n[score_all] discarding and rescoring")
            out[col] = np.nan

        todo = np.flatnonzero(out[col].isna().to_numpy())
        if not len(todo):
            if verbose:
                print(f"[score_all] {name}: cached, nothing to do")
            fingerprints.setdefault(name, current)
            continue

        if verbose:
            print(f"[score_all] {name}: scoring {len(todo):,} headlines")

        for start in range(0, len(todo), checkpoint_every):
            idx = todo[start : start + checkpoint_every]
            texts = out.loc[out.index[idx], "text"].tolist()
            out.iloc[idx, out.columns.get_loc(col)] = scorer.score(texts)
            fingerprints[name] = current
            _checkpoint(out, cols, cache, cache_path, fingerprints)
            if verbose:
                done = min(start + checkpoint_every, len(todo))
                print(f"[score_all]   {name}: {done:,}/{len(todo):,} checkpointed",
                      flush=True)

    scores = out[["headline_id"] + cols].copy()
    for c in cols:
        scores[c] = scores[c].astype("float32")
    _checkpoint(out, cols, cache, cache_path, fingerprints)
    return scores.reset_index(drop=True)


def _checkpoint(out, cols, cache, cache_path: Path, fingerprints: dict) -> None:
    """Merge the in-progress scores into the cache and write both files."""
    scores = out[["headline_id"] + cols].copy()
    for c in cols:
        scores[c] = scores[c].astype("float32")
    merged = pd.concat(
        [cache[~cache["headline_id"].isin(scores["headline_id"])], scores]
    ).reset_index(drop=True)
    _atomic_write_parquet(merged, cache_path)
    meta_path(cache_path).write_text(
        json.dumps(
            {"fingerprints": fingerprints, "n_rows": int(len(merged))},
            indent=2, sort_keys=True, allow_nan=False,
        ),
        encoding="utf-8",
    )
