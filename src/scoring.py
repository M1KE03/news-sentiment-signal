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
import errno
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

    Truncation at the configured token limit is measured in the pilot. Model is put
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
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
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


def _configured_fingerprints() -> dict[str, dict]:
    """Expected default measurements, without importing torch or loading weights.

    Lexicons must be inspected to establish their identity. FinBERT's expected
    identity comes from the pinned checkpoint contract; its loaded identity is
    checked again if inference is needed.
    """
    if not config.FINBERT_REVISION or not config.FINBERT_ID2LABEL:
        raise ValueError("default cached FinBERT scoring requires a pinned revision and labels")
    return {
        "lm": LMScorer().fingerprint,
        "vader": VaderScorer().fingerprint,
        "finbert": {
            "scorer": "finbert", "model": config.FINBERT_MODEL,
            "revision": config.FINBERT_REVISION,
            "max_length": config.FINBERT_MAX_LENGTH,
            "id2label": {int(k): v.lower() for k, v in config.FINBERT_ID2LABEL.items()},
        },
    }


def meta_path(cache_path: str | Path) -> Path:
    """Legacy sidecar location, used only to diagnose an incomplete old cache."""
    cache_path = Path(cache_path)
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


CACHE_METADATA_KEY = b"sentiment_signal.score_cache"
CACHE_FORMAT_VERSION = 1


class CacheLockedError(RuntimeError):
    """Another cooperating process holds the cache's writer lock."""


@contextmanager
def _writer_lock(cache_path: str | Path):
    """Hold an OS lock from the initial cache read through the final checkpoint.

    The sibling file persists; only the OS lock denotes ownership. Closing the
    handle (also on process death) releases it, without stale-lock deletion.
    """
    path = Path(cache_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name not in {"nt", "posix"}:
        raise RuntimeError(f"cache writer locking unsupported on {os.name}")
    with open(path.with_suffix(path.suffix + ".lock"), "a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise CacheLockedError(f"another scoring writer holds this cache: {path}") from exc
            raise
        yield  # the context closes the descriptor and releases the OS lock


def _decode_metadata(schema: pa.Schema, path: Path) -> dict:
    raw = (schema.metadata or {}).get(CACHE_METADATA_KEY)
    if raw is None:
        raise IncompatibleCache(
            f"{path}: legacy cache or missing embedded metadata; rebuild into a new cache path"
        )
    try:
        meta = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise IncompatibleCache(f"{path}: invalid embedded metadata JSON") from exc
    if not isinstance(meta, dict):
        raise IncompatibleCache(f"{path}: embedded metadata must be an object")
    if type(meta.get("format_version")) is not int or meta["format_version"] != CACHE_FORMAT_VERSION:
        raise IncompatibleCache(f"{path}: unsupported cache format version")
    generation = meta.get("generation_id")
    try:
        parsed = uuid.UUID(hex=generation) if isinstance(generation, str) else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.version != 4 or parsed.hex != generation:
        raise IncompatibleCache(f"{path}: invalid generation_id")
    if type(meta.get("n_rows")) is not int or meta["n_rows"] < 0:
        raise IncompatibleCache(f"{path}: invalid n_rows")
    if not isinstance(meta.get("fingerprints"), dict):
        raise IncompatibleCache(f"{path}: invalid fingerprint metadata")
    return meta


def _validate_candidate(path: Path, schema: pa.Schema, meta: dict) -> None:
    """Check the finished footer before the single publication operation."""
    footer = pq.read_metadata(path)
    actual_schema = footer.schema.to_arrow_schema()
    if (footer.num_rows != meta["n_rows"] or not actual_schema.equals(schema)
            or _decode_metadata(actual_schema, path) != meta):
        raise IncompatibleCache(f"{path}: candidate footer does not match the intended snapshot")


def _atomic_write_parquet(df: pd.DataFrame, path: Path, meta: dict) -> None:
    """Publish data and provenance together; interruption leaves one generation.

    This does not promise power-loss durability or disk-corruption recovery.
    """
    _validate_cache(df, meta, path)
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[CACHE_METADATA_KEY] = json.dumps(meta, sort_keys=True, allow_nan=False).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    _decode_metadata(table.schema, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            pq.write_table(table, stream)
            stream.flush()
            os.fsync(stream.fileno())
        _validate_candidate(Path(tmp), table.schema, meta)
        os.replace(tmp, path)
    finally:
        # A hard stop may leave this temporary behind; readers never promote it.
        try:
            os.unlink(tmp)
        except OSError:
            pass  # cleanup must not replace the original write/replace exception


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

    Current-artifact comparisons are performed by score_all before any writes.
    """
    cache_path = Path(cache_path)
    cols = [f"score_{n}" for n in config.SCORERS]
    try:
        stream = cache_path.open("rb")
    except FileNotFoundError:
        if meta_path(cache_path).exists():
            raise IncompatibleCache(f"{cache_path}: incomplete legacy pair; rebuild to a new cache path")
        return pd.DataFrame({"headline_id": pd.Series(dtype="string"),
                             "text_sha256": pd.Series(dtype="string"),
                             **{c: pd.Series(dtype="float32") for c in cols}}), {}
    try:
        with stream:
            table = pq.read_table(stream)
        meta = _decode_metadata(table.schema, cache_path)
        cache = table.to_pandas()
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise IncompatibleCache(f"{cache_path}: unreadable Parquet checkpoint") from exc
    _validate_cache(cache, meta, cache_path)
    cache["headline_id"] = cache["headline_id"].astype("string")
    cache["text_sha256"] = cache["text_sha256"].astype("string")
    return cache, meta


def _validate_cache(cache: pd.DataFrame, meta: dict, path: Path) -> None:
    cols = [f"score_{n}" for n in config.SCORERS]
    missing = set(["headline_id", "text_sha256"] + cols) - set(cache.columns)
    if missing:
        raise IncompatibleCache(f"{path}: missing required cache columns {sorted(missing)} (text identity required)")
    if cache.columns.duplicated().any():
        raise IncompatibleCache(f"{path}: duplicate cache columns")
    if meta.get("n_rows") != len(cache):
        raise IncompatibleCache(f"{path}: n_rows does not match the checkpoint")
    fingerprints = meta["fingerprints"]
    for name, fp in fingerprints.items():
        if name not in config.SCORERS:
            raise IncompatibleCache(f"{path}: unconfigured scorer {name!r}")
        try:
            _canonical_fingerprint(fp)
        except (ValueError, TypeError) as exc:
            raise IncompatibleCache(f"{path}: missing or invalid fingerprint for {name}") from exc
    for name in config.SCORERS:
        if cache[f"score_{name}"].dtype != np.dtype("float32"):
            raise IncompatibleCache(f"{path}: score_{name} must be float32")
        populated = int(cache[f"score_{name}"].notna().sum())
        recorded = fingerprints.get(name)
        if populated and (not isinstance(recorded, dict) or not recorded):
            raise IncompatibleCache(
                f"cached {name!r} has {populated:,} values with a missing or invalid "
                f"fingerprint in {path}. Their provenance cannot be inferred. "
                "Restore the original metadata or rebuild into a new cache path; "
                "on_fingerprint_change='rescore' cannot establish unknown provenance."
            )
    populated = cache[cols].notna().any(axis=1)
    valid_hash = cache["text_sha256"].astype("string").str.fullmatch(r"[0-9a-f]{64}").fillna(False)
    if (populated & ~valid_hash).any():
        raise IncompatibleCache(
            f"{path}: cached scores have missing or invalid text identity; rebuild into a new "
            "cache path rather than assuming the current text produced old scores"
        )
    if cache["headline_id"].isna().any() or cache["headline_id"].duplicated().any():
        raise IncompatibleCache(f"{path}: cache must contain unique, non-null headline_id values")


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
    embedded in the cache. If it no longer matches, the cached column was
    produced by a *different measurement*, and reusing it would silently mix two
    definitions inside one column. That raises `IncompatibleCache` by default;
    `on_fingerprint_change="rescore"` discards the stale column and recomputes
    it across the entire cache, including rows outside the requested subset.
    Changed raw text invalidates every scorer for that headline. Populated
    columns without recorded identity are refused at load time;
    rebuild into a new cache path rather than inventing their provenance.

    **Resumption (B14).** Scoring proceeds in batches of `checkpoint_every` rows
    and the cache is written atomically after each. An interrupted pass resumes
    from the last checkpoint and produces the same result as an uninterrupted
    one. This matters because the FinBERT pass over the assembled corpus is
    hours long and the plan's time budget assumes it is not repeated.
    """
    cache_path = Path(cache_path).resolve()
    with _writer_lock(cache_path):
        return _score_all_locked(headlines_df, cache_path, scorers, verbose,
                                 checkpoint_every, on_fingerprint_change)


def _score_all_locked(headlines_df, cache_path, scorers, verbose,
                      checkpoint_every, on_fingerprint_change):
    if on_fingerprint_change not in {"raise", "rescore"}:
        raise ValueError("on_fingerprint_change must be 'raise' or 'rescore'")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    cols = [f"score_{n}" for n in config.SCORERS]
    cache, meta = load_cache(cache_path)
    fingerprints = dict(meta.get("fingerprints", {}))
    heads = headlines_df[["headline_id", "text"]].copy()
    if heads["headline_id"].isna().any() or heads["headline_id"].duplicated().any():
        raise ValueError("headlines must contain unique, non-null headline_id values")
    heads["headline_id"] = heads["headline_id"].astype("string")
    if not heads["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("headline text must be a non-null string")
    heads["text_sha256"] = heads["text"].map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    out = heads.merge(cache, on="headline_id", how="left", validate="one_to_one",
                      suffixes=("", "_cached"))
    changed_text = (out["text_sha256_cached"].notna()
                    & out["text_sha256"].ne(out["text_sha256_cached"]))
    if changed_text.any():
        if on_fingerprint_change == "raise":
            raise IncompatibleCache(
                f"headline text changed for {int(changed_text.sum())} cached IDs; "
                "use on_fingerprint_change='rescore' to invalidate all their scores"
            )
        out.loc[changed_text, cols] = np.nan
    out = out.drop(columns="text_sha256_cached")

    supplied = None if scorers is None else list(scorers)
    if supplied is not None:
        names = [s.name for s in supplied]
        if len(set(names)) != len(names) or not set(names) <= set(config.SCORERS):
            raise ValueError("scorers must have unique configured names")
        expected = {s.name: getattr(s, "fingerprint", None) for s in supplied}
    else:
        expected = _configured_fingerprints()
    expected = {name: _canonical_fingerprint(fp) for name, fp in expected.items()}

    # Preflight every selected identity before scoring/checkpointing any column.
    for name, current in expected.items():
        col = f"score_{name}"
        stored = fingerprints.get(name)
        if stored:
            stored = _canonical_fingerprint(stored)
        if stored and stored != current:
            already = int(cache[col].notna().sum())
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
            if verbose:
                print(f"[score_all] {message}\n[score_all] discarding and rescoring")
            out[col] = np.nan
            cache[col] = np.nan  # including rows not present in this request
        fingerprints[name] = current

    if supplied is None:
        needed = [name for name in expected if out[f"score_{name}"].isna().any()]
        supplied = build_scorers(needed) if needed else []
    # Check loaded objects against the preflight snapshot before any writes.
    for scorer in supplied:
        if _canonical_fingerprint(scorer.fingerprint) != expected[scorer.name]:
            raise IncompatibleCache(f"loaded {scorer.name} identity differs from preflight")

    for scorer in supplied:
        name, col = scorer.name, f"score_{scorer.name}"
        current = expected[name]
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
    """Commit one snapshot containing in-progress scores and their provenance."""
    scores = out[["headline_id", "text_sha256"] + cols].copy()
    for c in cols:
        scores[c] = scores[c].astype("float32")
    merged = pd.concat(
        [cache[~cache["headline_id"].isin(scores["headline_id"])], scores]
    ).reset_index(drop=True)
    merged["headline_id"] = merged["headline_id"].astype("string")
    merged["text_sha256"] = merged["text_sha256"].astype("string")
    for c in cols:
        merged[c] = merged[c].astype("float32")
    _atomic_write_parquet(merged, cache_path, {
        "format_version": CACHE_FORMAT_VERSION, "generation_id": uuid.uuid4().hex,
        "fingerprints": fingerprints, "n_rows": int(len(merged)),
    })
