"""Scorer range and determinism checks (Stage 1).

The LM tests run on a small hand-built word list, so they need neither the
Loughran-McDonald CSV nor a network. The VADER and FinBERT tests skip cleanly
when the optional dependency or the model weights are not present, so `pytest`
is green on a fresh clone before Stage 1 has been run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import scoring

# Deliberately includes the cases the lexicons are expected to fail (Stage 1,
# step 4): the LM-neutral "liability provisions", the word-count-negative but
# actually-positive "costs fell sharply", and the negated "smaller than feared".
SPOT_CHECK = [
    "Company reports increased liability provisions",
    "Costs fell sharply in the third quarter",
    "Profit warning smaller than feared",
    "Quarterly profit beats expectations",
    "Shares plunge after weak guidance",
    "Board announces no change to the dividend",
]

POSITIVE_WORDS = {"beats", "profit", "gain", "strong", "record", "upgrade"}
NEGATIVE_WORDS = {"loss", "plunge", "weak", "warning", "liability", "fell", "downgrade"}


@pytest.fixture(scope="module")
def lm():
    return scoring.LMScorer.from_word_lists(POSITIVE_WORDS, NEGATIVE_WORDS)


def test_lm_scores_are_in_range(lm):
    s = lm.score(SPOT_CHECK)
    assert s.shape == (len(SPOT_CHECK),)
    assert np.all(s >= -1.0) and np.all(s <= 1.0)


def test_lm_is_deterministic_across_runs(lm):
    np.testing.assert_array_equal(lm.score(SPOT_CHECK), lm.score(SPOT_CHECK))


def test_lm_returns_zero_with_no_dictionary_hits(lm):
    assert lm.score(["Board announces no change to the dividend"])[0] == 0.0


def test_lm_sign_follows_word_counts(lm):
    assert lm.score(["Quarterly profit beats expectations"])[0] > 0
    assert lm.score(["Shares plunge after weak guidance"])[0] < 0


def test_lm_fails_on_costs_fell_sharply_as_documented(lm):
    # Not a bug -- the point of Exhibit A. A word counter cannot see the verb,
    # so a fall in *costs* reads negative. This is the gap FinBERT should close.
    assert lm.score(["Costs fell sharply in the third quarter"])[0] < 0


def test_lm_handles_empty_and_punctuation_only_input(lm):
    assert lm.score(["", "!!! ???"]).tolist() == [0.0, 0.0]


@pytest.mark.parametrize("name", ["vader", "finbert"])
def test_optional_scorers_range_and_determinism(name):
    try:
        scorer = {"vader": scoring.VaderScorer, "finbert": scoring.FinbertScorer}[name]()
    except Exception as exc:  # dependency or weights absent on a fresh clone
        pytest.skip(f"{name} unavailable: {type(exc).__name__}: {exc}")
    first = scorer.score(SPOT_CHECK)
    assert first.shape == (len(SPOT_CHECK),)
    assert np.all(first >= -1.0) and np.all(first <= 1.0)
    np.testing.assert_allclose(first, scorer.score(SPOT_CHECK), atol=1e-6)


def test_score_all_caches_and_never_rescores(tmp_path, lm):
    """The cache is what makes FinBERT a one-off cost. Prove it skips scored rows."""

    class CountingScorer:
        name = "lm"

        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def score(self, texts):
            self.calls += len(texts)
            return self.inner.score(texts)

    headlines = pd.DataFrame(
        {"headline_id": [f"h{i}" for i in range(len(SPOT_CHECK))], "text": SPOT_CHECK}
    )
    cache = tmp_path / "scores.parquet"

    counter = CountingScorer(lm)
    first = scoring.score_all(headlines, cache_path=cache, scorers=[counter], verbose=False)
    assert counter.calls == len(SPOT_CHECK)
    assert cache.exists()

    counter2 = CountingScorer(lm)
    second = scoring.score_all(headlines, cache_path=cache, scorers=[counter2], verbose=False)
    assert counter2.calls == 0, "cached headlines were re-scored"
    pd.testing.assert_series_equal(first["score_lm"], second["score_lm"])

    # A new headline is scored; the old ones are still not.
    grown = pd.concat(
        [headlines, pd.DataFrame({"headline_id": ["h99"], "text": ["Record profit"]})]
    )
    counter3 = CountingScorer(lm)
    scoring.score_all(grown, cache_path=cache, scorers=[counter3], verbose=False)
    assert counter3.calls == 1


def test_score_all_writes_every_scorer_column(tmp_path, lm):
    headlines = pd.DataFrame({"headline_id": ["a", "b"], "text": SPOT_CHECK[:2]})
    out = scoring.score_all(headlines, cache_path=tmp_path / "s.parquet", scorers=[lm],
                            verbose=False)
    assert list(out.columns) == ["headline_id"] + [f"score_{n}" for n in config.SCORERS]


# --------------------------------------------------------------------------
# B12: the classification path. Separate from the continuous path, on purpose.
# --------------------------------------------------------------------------


class _FakeClassifier:
    """A classifier with controllable probabilities, for the contract tests."""

    name = "fake"
    labels = tuple(config.LABELS)

    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, texts):
        return self._proba[: len(texts)]

    def predict(self, texts):
        return np.asarray(config.LABELS, dtype=object)[
            self.predict_proba(texts).argmax(axis=1)
        ]

    def score(self, texts):
        p = self.predict_proba(texts)
        return (p[:, 2] - p[:, 0]).astype(np.float32)   # positive - negative


def test_continuous_score_does_not_determine_the_class():
    """The reason Act 1 cannot reuse Act 2's number.

    Both rows have P(pos) - P(neg) = 0.4, yet the first is positive by argmax
    and the second neutral. The neutral mass, which the difference discards, is
    what decides -- so a threshold on the score cannot recover the label.
    """
    #                 neg   neu   pos
    proba = np.array([[0.10, 0.40, 0.50],
                      [0.05, 0.50, 0.45]])
    clf = _FakeClassifier(proba)
    texts = ["a", "b"]

    np.testing.assert_allclose(clf.score(texts), [0.4, 0.4], atol=1e-6)
    assert clf.predict(texts).tolist() == ["positive", "neutral"]


def test_predictions_for_uses_argmax_on_a_classifier():
    from src import validate

    clf = _FakeClassifier([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    assert validate.predictions_for(clf, ["a", "b"]).tolist() == ["negative", "positive"]


def test_predictions_for_refuses_to_threshold_a_classifier():
    from src import validate

    clf = _FakeClassifier([[0.7, 0.2, 0.1]])
    with pytest.raises(ValueError, match="discard the neutral probability"):
        validate.predictions_for(clf, ["a"], thresholds=(-0.1, 0.1))


def test_predictions_for_requires_a_band_for_a_lexicon(lm):
    from src import validate

    with pytest.raises(ValueError, match="needs a neutral band"):
        validate.predictions_for(lm, SPOT_CHECK)
    out = validate.predictions_for(lm, SPOT_CHECK, thresholds=(-0.1, 0.1))
    assert set(out) <= set(config.LABELS)


@pytest.fixture(scope="module")
def finbert():
    try:
        return scoring.FinbertScorer()
    except Exception as exc:
        pytest.skip(f"FinBERT unavailable: {type(exc).__name__}: {exc}")


def test_finbert_label_order_matches_the_pinned_one(finbert):
    """The checkpoint publishes positive/negative/neutral, not the usual order."""
    published = {i: l.lower() for i, l in finbert.model.config.id2label.items()}
    assert published == {int(k): v for k, v in config.FINBERT_ID2LABEL.items()}


def test_finbert_proba_is_in_config_label_order(finbert):
    proba = finbert.predict_proba(SPOT_CHECK)
    assert proba.shape == (len(SPOT_CHECK), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # Column order is config.LABELS, whatever the checkpoint's own order is.
    raw = finbert._probs(SPOT_CHECK)
    np.testing.assert_allclose(proba[:, 0], raw[:, finbert._neg], atol=1e-6)
    np.testing.assert_allclose(proba[:, 1], raw[:, finbert._neu], atol=1e-6)
    np.testing.assert_allclose(proba[:, 2], raw[:, finbert._pos], atol=1e-6)


def test_finbert_score_is_consistent_with_its_probabilities(finbert):
    proba = finbert.predict_proba(SPOT_CHECK)
    np.testing.assert_allclose(
        finbert.score(SPOT_CHECK), proba[:, 2] - proba[:, 0], atol=1e-5
    )


def test_finbert_predict_is_argmax_over_the_probabilities(finbert):
    proba = finbert.predict_proba(SPOT_CHECK)
    expected = np.asarray(config.LABELS, dtype=object)[proba.argmax(axis=1)]
    assert finbert.predict(SPOT_CHECK).tolist() == expected.tolist()


def test_finbert_predict_retains_the_neutral_mass(finbert, monkeypatch):
    """Synthetic probabilities through the real object: neutral must be able to win."""
    fake = np.zeros((2, 3), dtype=np.float32)
    fake[:, finbert._neg] = [0.10, 0.05]
    fake[:, finbert._neu] = [0.40, 0.50]
    fake[:, finbert._pos] = [0.50, 0.45]
    monkeypatch.setattr(finbert, "_probs", lambda texts: fake[: len(texts)])
    assert finbert.predict(["a", "b"]).tolist() == ["positive", "neutral"]
    np.testing.assert_allclose(finbert.score(["a", "b"]), [0.4, 0.4], atol=1e-6)


def test_finbert_gets_the_documented_hard_cases_right(finbert):
    """Exhibit A's point: context models see what a word counter cannot."""
    got = dict(zip(SPOT_CHECK, finbert.predict(SPOT_CHECK)))
    # "Costs fell sharply" is word-count-negative but actually good news.
    assert got["Costs fell sharply in the third quarter"] != "negative"
    assert got["Quarterly profit beats expectations"] == "positive"
    assert got["Shares plunge after weak guidance"] == "negative"


# --------------------------------------------------------------------------
# B13/B14: cache provenance and interruption-safe resumption.
# --------------------------------------------------------------------------


class _StubScorer:
    """A scorer with a settable fingerprint and a call counter."""

    def __init__(self, name="lm", value=0.5, fingerprint=None, fail_after=None):
        self.name = name
        self.value = value
        self._fp = fingerprint or {"scorer": name, "version": "v1"}
        self.fail_after = fail_after
        self.scored = 0

    @property
    def fingerprint(self):
        return self._fp

    def score(self, texts):
        if self.fail_after is not None and self.scored >= self.fail_after:
            raise KeyboardInterrupt("simulated interruption")
        self.scored += len(texts)
        return np.full(len(texts), self.value, dtype=np.float32)


def _headlines(n):
    return pd.DataFrame(
        {"headline_id": [f"h{i:05d}" for i in range(n)], "text": [f"story {i}" for i in range(n)]}
    )


def test_cache_records_the_fingerprint_that_produced_it(tmp_path):
    cache = tmp_path / "scores.parquet"
    scoring.score_all(_headlines(10), cache, [_StubScorer()], verbose=False)
    meta = json.loads(scoring.meta_path(cache).read_text(encoding="utf-8"))
    assert meta["fingerprints"]["lm"] == {"scorer": "lm", "version": "v1"}
    assert meta["n_rows"] == 10


def test_changed_artifact_refuses_to_reuse_cached_scores(tmp_path):
    """B13: a different measurement must not be mixed into one column."""
    cache = tmp_path / "scores.parquet"
    scoring.score_all(_headlines(10), cache, [_StubScorer(value=0.5)], verbose=False)

    changed = _StubScorer(value=-0.5, fingerprint={"scorer": "lm", "version": "v2"})
    with pytest.raises(scoring.IncompatibleCache, match="different measurement"):
        scoring.score_all(_headlines(10), cache, [changed], verbose=False)


def test_the_refusal_names_what_changed_and_how_much_is_affected(tmp_path):
    cache = tmp_path / "scores.parquet"
    scoring.score_all(_headlines(7), cache, [_StubScorer()], verbose=False)
    changed = _StubScorer(fingerprint={"scorer": "lm", "version": "v2"})
    with pytest.raises(scoring.IncompatibleCache) as exc:
        scoring.score_all(_headlines(7), cache, [changed], verbose=False)
    msg = str(exc.value)
    assert "v1" in msg and "v2" in msg and "7" in msg


def test_rescore_discards_the_stale_column(tmp_path):
    cache = tmp_path / "scores.parquet"
    scoring.score_all(_headlines(10), cache, [_StubScorer(value=0.5)], verbose=False)
    changed = _StubScorer(value=-0.5, fingerprint={"scorer": "lm", "version": "v2"})
    out = scoring.score_all(
        _headlines(10), cache, [changed], verbose=False, on_fingerprint_change="rescore"
    )
    np.testing.assert_allclose(out["score_lm"], -0.5)
    assert changed.scored == 10
    meta = json.loads(scoring.meta_path(cache).read_text(encoding="utf-8"))
    assert meta["fingerprints"]["lm"]["version"] == "v2"


def test_unchanged_fingerprint_still_reuses_the_cache(tmp_path):
    cache = tmp_path / "scores.parquet"
    scoring.score_all(_headlines(10), cache, [_StubScorer()], verbose=False)
    again = _StubScorer()
    scoring.score_all(_headlines(10), cache, [again], verbose=False)
    assert again.scored == 0


def test_interrupted_run_resumes_and_matches_an_uninterrupted_one(tmp_path):
    """B14: the expensive pass must survive a kill signal."""
    heads = _headlines(250)

    clean_cache = tmp_path / "clean.parquet"
    clean = scoring.score_all(
        heads, clean_cache, [_StubScorer(value=0.25)], verbose=False, checkpoint_every=100
    )

    broken_cache = tmp_path / "broken.parquet"
    interrupted = _StubScorer(value=0.25, fail_after=200)
    with pytest.raises(KeyboardInterrupt):
        scoring.score_all(
            heads, broken_cache, [interrupted], verbose=False, checkpoint_every=100
        )

    # Two checkpoints survived the interruption.
    partial, meta = scoring.load_cache(broken_cache)
    assert partial["score_lm"].notna().sum() == 200
    assert meta["fingerprints"]["lm"]["version"] == "v1"

    resumed_scorer = _StubScorer(value=0.25)
    resumed = scoring.score_all(
        heads, broken_cache, [resumed_scorer], verbose=False, checkpoint_every=100
    )
    assert resumed_scorer.scored == 50, "only the unfinished rows should be rescored"
    pd.testing.assert_frame_equal(
        clean.sort_values("headline_id").reset_index(drop=True),
        resumed.sort_values("headline_id").reset_index(drop=True),
    )


def test_checkpoint_writes_are_atomic(tmp_path):
    """A partial parquet would look loadable, which is worse than none."""
    cache = tmp_path / "scores.parquet"
    scoring.score_all(_headlines(120), cache, [_StubScorer()], verbose=False,
                      checkpoint_every=50)
    assert not list(tmp_path.glob("*.tmp")), "temp files must not survive"
    assert pd.read_parquet(cache)["score_lm"].notna().sum() == 120


def test_real_scorers_expose_a_fingerprint(lm):
    fp = lm.fingerprint
    assert fp["scorer"] == "lm" and len(fp["wordlist_sha1"]) == 40
    # The hash follows the actual word lists, not just the recorded version.
    other = scoring.LMScorer.from_word_lists({"gain"}, {"loss"})
    assert other.fingerprint["wordlist_sha1"] != fp["wordlist_sha1"]
