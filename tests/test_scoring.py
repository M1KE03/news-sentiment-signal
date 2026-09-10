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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import scoring

# Deliberately includes the cases the lexicons were EXPECTED to fail (Stage 1,
# step 4): the LM-neutral "liability provisions", the word-count-negative but
# arguably-positive "costs fell sharply", and the negated "smaller than feared".
#
# Recorded 2026-09-09, on the first run with torch executable: FinBERT fails the
# second and third of those too, both with P(negative) > 0.92. These are six
# invented sentences and they measure nothing; Act 1 measures classification
# quality on independently annotated corpus text.
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
    # Not a bug -- a word counter cannot see the verb, so a fall in *costs*
    # reads negative. The behaviour of the lexicon is the claim being tested.
    #
    # The sentence "This is the gap FinBERT should close" was removed on
    # 2026-09-09: FinBERT also calls this headline negative (P = 0.932), so the
    # gap is not closed here. See
    # `test_finbert_predictions_match_the_recorded_characterization`.
    assert lm.score(["Costs fell sharply in the third quarter"])[0] < 0


def test_lm_handles_empty_and_punctuation_only_input(lm):
    assert lm.score(["", "!!! ???"]).tolist() == [0.0, 0.0]


@pytest.mark.integration
@pytest.mark.parametrize("name", ["vader", "finbert"])
def test_optional_scorers_range_and_determinism(name):
    scorer = {"vader": scoring.VaderScorer, "finbert": scoring.FinbertScorer}[name]()
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

        @property
        def fingerprint(self):
            return self.inner.fingerprint

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
    return scoring.FinbertScorer()


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


# What FinBERT actually predicts on SPOT_CHECK, recorded 2026-09-09 against
# config.FINBERT_REVISION on the first run in which torch was executable.
# This is a CHARACTERIZATION record, not a statement that these labels are
# correct. Its job is to fail when the checkpoint, the revision or the label
# ordering changes -- nothing more.
FINBERT_OBSERVED = {
    "Company reports increased liability provisions": "negative",
    "Costs fell sharply in the third quarter": "negative",
    "Profit warning smaller than feared": "negative",
    "Quarterly profit beats expectations": "positive",
    "Shares plunge after weak guidance": "negative",
    "Board announces no change to the dividend": "neutral",
}


def test_finbert_label_order_contract_holds_on_real_text(finbert):
    """B12's label-order pin, checked against the running checkpoint.

    `ProsusAI/finbert` publishes `{0: positive, 1: negative, 2: neutral}`, which
    is not the conventional order, so index-based code inverts every score
    (B03). These two headlines are unambiguous in opposite directions, so if the
    published order ever changed underneath the pin, one of them would flip.
    That is the whole claim being made here.
    """
    got = dict(zip(SPOT_CHECK, finbert.predict(SPOT_CHECK)))
    assert got["Quarterly profit beats expectations"] == "positive"
    assert got["Shares plunge after weak guidance"] == "negative"


def test_finbert_predictions_match_the_recorded_characterization(finbert):
    """Detects a changed model, revision or label mapping. Asserts no quality.

    **This test previously asserted that FinBERT would NOT call "Costs fell
    sharply in the third quarter" negative** -- Exhibit A's premise that a
    context model sees what a word counter cannot. It had never executed,
    because Smart App Control blocked torch. On its first real run it failed:
    FinBERT calls that headline negative with P(neg) = 0.932, and calls
    "Profit warning smaller than feared" negative with P(neg) = 0.924. Both are
    the cases the fixture was built to demonstrate, and on both FinBERT agrees
    with the word counter it was supposed to beat.

    The assertion was removed rather than inverted. Six invented sentences
    cannot establish classification quality in either direction; that is what
    Act 1's independently annotated evaluation exists to measure, under the
    [validation protocol](../docs/validation-protocol.md). Asserting the
    expected outcome here was the same predetermined-finding pattern that P24
    removed from the figure titles.
    """
    got = dict(zip(SPOT_CHECK, finbert.predict(SPOT_CHECK)))
    assert got == FINBERT_OBSERVED


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
    _, meta = scoring.load_cache(cache)
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
    _, meta = scoring.load_cache(cache)
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


def test_finbert_fingerprint_round_trip_reuses_scores(tmp_path):
    cache = tmp_path / "scores.parquet"
    fp = {"scorer": "finbert", "revision": "pinned-revision",
          "id2label": {0: "positive", 1: "negative", 2: "neutral"}}
    scoring.score_all(_headlines(3), cache,
                      [_StubScorer(name="finbert", fingerprint=fp)], verbose=False)
    again = _StubScorer(name="finbert", fingerprint=fp)
    scoring.score_all(_headlines(3), cache, [again], verbose=False)
    assert again.scored == 0
    changed = _StubScorer(name="finbert", fingerprint={**fp, "revision": "other"})
    with pytest.raises(scoring.IncompatibleCache, match="different measurement"):
        scoring.score_all(_headlines(3), cache, [changed], verbose=False)


@pytest.mark.parametrize("revision", ["explicit-revision", None])
def test_finbert_fingerprint_records_constructor_revision(monkeypatch, revision):
    calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(_name_or_path="fake-finbert",
                               id2label=dict(config.FINBERT_ID2LABEL)),
        eval=lambda: None, to=lambda device: None,
    )

    def load_model(name, **kwargs):
        calls.append(kwargs)
        return model

    def load_tokenizer(name, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
    ))
    scorer = scoring.FinbertScorer("fake-finbert", revision=revision, device="cpu")
    monkeypatch.setattr(config, "FINBERT_REVISION", "unrelated-config-change")
    assert scorer.fingerprint["revision"] == revision
    assert calls == [{"revision": revision} if revision else {}] * 2


@pytest.mark.parametrize("metadata", [None, {}, {"fingerprints": {"lm": None}},
                                      {"fingerprints": {"lm": {}}}])
@pytest.mark.parametrize("explicit_scorer", [False, True])
def test_populated_cache_without_identity_is_rejected(
    tmp_path, monkeypatch, metadata, explicit_scorer
):
    def no_model_loading(names):
        pytest.fail("unknown cache provenance must be rejected before loading models")

    monkeypatch.setattr(scoring, "build_scorers", no_model_loading)
    cache = tmp_path / "scores.parquet"
    scoring.score_all(_headlines(1), cache, [_StubScorer()], verbose=False)
    table = pq.read_table(cache)
    schema_meta = dict(table.schema.metadata)
    embedded = json.loads(schema_meta[scoring.CACHE_METADATA_KEY])
    embedded["fingerprints"] = (metadata or {}).get("fingerprints", {})
    schema_meta[scoring.CACHE_METADATA_KEY] = json.dumps(embedded).encode()
    pq.write_table(table.replace_schema_metadata(schema_meta), cache)
    before = cache.read_bytes()
    scorer = _StubScorer(value=-0.8)
    with pytest.raises(scoring.IncompatibleCache, match="missing.*fingerprint"):
        scoring.score_all(_headlines(1), cache,
                          [scorer] if explicit_scorer else None, verbose=False)
    assert scorer.scored == 0
    assert cache.read_bytes() == before


def test_empty_score_column_needs_no_previous_identity(tmp_path):
    cache = tmp_path / "scores.parquet"
    scoring.score_all(_headlines(1), cache, scorers=[], verbose=False)
    scorer = _StubScorer()
    scoring.score_all(_headlines(1), cache, [scorer], verbose=False)
    assert scorer.scored == 1


def test_scorer_without_identity_cannot_create_cache(tmp_path):
    cache = tmp_path / "scores.parquet"
    scorer = _StubScorer()
    scorer._fp = None
    with pytest.raises(ValueError, match="fingerprint"):
        scoring.score_all(_headlines(1), cache, [scorer], verbose=False)
    assert scorer.scored == 0
    assert not cache.exists()


def test_default_completed_cache_validates_current_identity(tmp_path, monkeypatch):
    cache = tmp_path / "scores.parquet"
    scorers = [_StubScorer(name=n) for n in config.SCORERS]
    scoring.score_all(_headlines(2), cache, scorers, verbose=False)
    identities = {s.name: s.fingerprint for s in scorers}
    monkeypatch.setattr(scoring, "_configured_fingerprints", lambda: identities)
    monkeypatch.setattr(scoring, "build_scorers",
                        lambda names: pytest.fail("completed cache must not load models"))
    scoring.score_all(_headlines(2), cache, verbose=False)
    identities["finbert"] = {"scorer": "finbert", "version": "v2"}
    before = cache.read_bytes()
    with pytest.raises(scoring.IncompatibleCache, match="different measurement"):
        scoring.score_all(_headlines(2), cache, verbose=False)
    assert cache.read_bytes() == before


def test_subset_rescore_invalidates_other_rows_and_resumes(tmp_path):
    cache = tmp_path / "scores.parquet"
    heads = _headlines(3)
    scoring.score_all(heads, cache, [_StubScorer()], verbose=False)
    fp = {"scorer": "lm", "version": "v2"}
    scoring.score_all(heads.iloc[:1], cache, [_StubScorer(value=-.5, fingerprint=fp)],
                      verbose=False, on_fingerprint_change="rescore")
    saved, _ = scoring.load_cache(cache)
    assert saved.loc[saved.headline_id != "h00000", "score_lm"].isna().all()
    again = _StubScorer(value=-.5, fingerprint=fp)
    result = scoring.score_all(heads, cache, [again], verbose=False)
    assert again.scored == 2
    np.testing.assert_allclose(result.score_lm, -.5)


def test_changed_text_refuses_reuse_and_invalidates_all_scorers(tmp_path):
    cache = tmp_path / "scores.parquet"
    heads = _headlines(2)
    scoring.score_all(heads, cache, [_StubScorer(name=n) for n in config.SCORERS],
                      verbose=False)
    heads.loc[0, "text"] = "STORY 0!"
    before = cache.read_bytes()
    with pytest.raises(scoring.IncompatibleCache, match="headline text"):
        scoring.score_all(heads, cache, [_StubScorer()], verbose=False)
    assert cache.read_bytes() == before
    lm = _StubScorer(value=-.5)
    result = scoring.score_all(heads, cache, [lm], verbose=False,
                               on_fingerprint_change="rescore")
    assert lm.scored == 1
    assert result.loc[0, ["score_vader", "score_finbert"]].isna().all()
    assert result.loc[1, "score_finbert"] == .5


def test_missing_text_identity_is_not_blessed(tmp_path):
    cache = tmp_path / "scores.parquet"
    scoring.score_all(_headlines(2), cache, [_StubScorer()], verbose=False)
    saved = pq.read_table(cache).drop(["text_sha256"])
    pq.write_table(saved, cache)
    with pytest.raises(scoring.IncompatibleCache, match="text identity"):
        scoring.score_all(_headlines(2), cache, [_StubScorer()], verbose=False)


def test_all_identities_checked_before_any_checkpoint(tmp_path):
    cache = tmp_path / "scores.parquet"
    scoring.score_all(_headlines(2), cache,
                      [_StubScorer(), _StubScorer(name="vader")], verbose=False)
    before = cache.read_bytes()
    lm = _StubScorer()
    changed = _StubScorer(name="vader", fingerprint={"version": "v2"})
    with pytest.raises(scoring.IncompatibleCache):
        scoring.score_all(_headlines(3), cache, [lm, changed], verbose=False)
    assert lm.scored == 0
    assert cache.read_bytes() == before


def test_configured_identity_needs_no_finbert_import(monkeypatch):
    monkeypatch.setattr(scoring, "LMScorer", lambda: _StubScorer())
    monkeypatch.setattr(scoring, "VaderScorer", lambda: _StubScorer(name="vader"))
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    expected = scoring._configured_fingerprints()
    assert expected["finbert"]["revision"] == config.FINBERT_REVISION
    assert expected["finbert"]["id2label"] == config.FINBERT_ID2LABEL
    monkeypatch.setattr(config, "FINBERT_MAX_LENGTH", 128)
    assert scoring._configured_fingerprints()["finbert"]["max_length"] == 128


def test_default_rescore_loads_only_changed_scorer(tmp_path, monkeypatch):
    cache = tmp_path / "scores.parquet"
    old = [_StubScorer(name=n) for n in config.SCORERS]
    scoring.score_all(_headlines(2), cache, old, verbose=False)
    changed = _StubScorer(name="finbert", value=-.5, fingerprint={"version": "v2"})
    identities = {s.name: s.fingerprint for s in old}
    identities["finbert"] = changed.fingerprint
    monkeypatch.setattr(scoring, "_configured_fingerprints", lambda: identities)

    def build(names):
        assert names == ["finbert"]
        return [changed]

    monkeypatch.setattr(scoring, "build_scorers", build)
    result = scoring.score_all(_headlines(2), cache, verbose=False,
                               on_fingerprint_change="rescore")
    assert changed.scored == 2
    np.testing.assert_allclose(result.score_finbert, -.5)
    np.testing.assert_allclose(result.score_lm, .5)


def test_interrupted_subset_rescore_does_not_restore_old_values(tmp_path):
    cache = tmp_path / "scores.parquet"
    heads = _headlines(4)
    scoring.score_all(heads, cache, [_StubScorer()], verbose=False)
    fp = {"version": "v2"}
    interrupted = _StubScorer(value=-.5, fingerprint=fp, fail_after=1)
    with pytest.raises(KeyboardInterrupt):
        scoring.score_all(heads.iloc[:2], cache, [interrupted], verbose=False,
                          checkpoint_every=1, on_fingerprint_change="rescore")
    saved, _ = scoring.load_cache(cache)
    assert saved.score_lm.notna().sum() == 1
    again = _StubScorer(value=-.5, fingerprint=fp)
    result = scoring.score_all(heads, cache, [again], verbose=False)
    assert again.scored == 3
    np.testing.assert_allclose(result.score_lm, -.5)
