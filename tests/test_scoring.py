"""Scorer range and determinism checks (Stage 1).

The LM tests run on a small hand-built word list, so they need neither the
Loughran-McDonald CSV nor a network. The VADER and FinBERT tests skip cleanly
when the optional dependency or the model weights are not present, so `pytest`
is green on a fresh clone before Stage 1 has been run.
"""

from __future__ import annotations

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
