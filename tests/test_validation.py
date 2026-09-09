"""R06a: the blind annotation sample.

Everything here runs without a single human label. That is the point of the
increment: the audit found B11 described as "blocked on corpus assembly" long
after assembly finished, when what actually blocks it is human annotation --
and none of the tooling below needs any (A11).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import annotate


def _frame(texts, dates, tickers=None, ids=None):
    n = len(texts)
    return pd.DataFrame(
        {
            "headline_id": ids or [f"h{i:04d}" for i in range(n)],
            "text": texts,
            "text_norm": [t.lower() for t in texts],
            "ts_utc": [pd.Timestamp(d, tz="UTC") for d in dates],
            "tickers": tickers if tickers is not None else [["AAPL"]] * n,
            "source": ["benzinga.com"] * n,
        }
    )


def _corpus(n=4000, seed=7):
    """A synthetic frame spanning several years, with a few repeated stories."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2015-01-01", "2019-12-31", freq="D")
    texts, dates = [], []
    for i in range(n):
        d = days[rng.integers(len(days))]
        texts.append(f"company {i % 900} reports quarterly results number {i}")
        dates.append(d)
    return _frame(texts, dates)


# --------------------------------------------------------------- grouping


def test_article_groups_joins_exact_repeats():
    df = _frame(
        ["Apple beats on earnings", "Apple beats on earnings", "Totally other news"],
        ["2015-01-05", "2015-01-06", "2015-01-07"],
    )
    g = annotate.article_groups(df)
    assert g.iloc[0] == g.iloc[1]
    assert g.iloc[2] != g.iloc[0]


def test_article_groups_are_not_windowed():
    """The dedup window is three days; leakage does not expire after three days.

    A story republished ten days later survives deduplication twice and is two
    rows of the corpus. It is still one article: if one row fits a threshold
    and the other evaluates it, that is leakage.
    """
    df = _frame(
        ["Fed holds rates steady", "Fed holds rates steady"],
        ["2015-03-01", "2015-03-20"],
    )
    assert annotate.article_groups(df).nunique() == 1


def test_article_groups_join_near_duplicates():
    ten = "apple beats on earnings and raises its full year outlook"
    near = "apple beats on earnings and raises its full year guidance"   # 9/10
    df = _frame([ten, near, "microsoft cuts its dividend"],
                ["2015-01-05"] * 3)
    g = annotate.article_groups(df)
    assert g.iloc[0] == g.iloc[1]
    assert g.iloc[2] != g.iloc[0]


def test_article_groups_are_connected_components():
    """A and C need not be similar; if both match B they are one article."""
    a = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    b = "alpha beta gamma delta epsilon zeta eta theta iota lambda"   # 9/10 with a
    c = "alpha beta gamma delta epsilon zeta eta theta lambda mu"     # 8/10 with a
    df = _frame([a, b, c], ["2015-01-05"] * 3)
    from src.data import _token_set_overlap

    assert _token_set_overlap(a.lower(), c.lower()) < 0.9    # not directly similar
    assert annotate.article_groups(df).nunique() == 1


# ------------------------------------------------------------------ draw


def test_draw_is_reproducible_and_seed_dependent():
    frame = _corpus()
    a = annotate.draw_sample(frame, n_total=200, seed=config.SEED)
    b = annotate.draw_sample(frame, n_total=200, seed=config.SEED)
    c = annotate.draw_sample(frame, n_total=200, seed=config.SEED + 1)
    assert list(a["headline_id"]) == list(b["headline_id"])
    assert set(a["headline_id"]) != set(c["headline_id"])


def test_draw_does_not_depend_on_frame_row_order():
    """The sample must be a function of the frame's CONTENT and the seed.

    This is the defect R03b removed from `dedup`, in a different place: if the
    draw keys on incoming row order, then the sample silently depends on how
    the corpus happened to be sorted.
    """
    frame = _corpus()
    shuffled = frame.sample(frac=1.0, random_state=99).reset_index(drop=True)
    a = annotate.draw_sample(frame, n_total=200)
    b = annotate.draw_sample(shuffled, n_total=200)
    assert set(a["headline_id"]) == set(b["headline_id"])


def test_allocation_is_proportional_and_sums_exactly():
    counts = pd.Series({"a": 500, "b": 300, "c": 200})
    alloc = annotate._allocate(counts, 100)
    assert alloc.sum() == 100
    assert alloc["a"] == 50 and alloc["b"] == 30 and alloc["c"] == 20


def test_allocation_distributes_remainders_deterministically():
    """3 strata, 2 slots: two cells get one each, and which two is not random."""
    counts = pd.Series({"a": 10, "b": 10, "c": 10})
    first = annotate._allocate(counts, 2)
    assert first.sum() == 2
    # Same counts in a different index order must give the same allocation.
    shuffled = pd.Series({"c": 10, "a": 10, "b": 10})
    second = annotate._allocate(shuffled, 2)
    assert first.to_dict() == second.to_dict()


def test_allocation_never_asks_a_stratum_for_more_rows_than_it_has():
    """A tiny cell cannot supply its proportional share of a large draw."""
    counts = pd.Series({"big": 1000, "tiny": 2})
    alloc = annotate._allocate(counts, 100)
    assert alloc["tiny"] <= 2
    assert (alloc <= counts).all()


def test_draw_refuses_a_frame_with_duplicate_ids():
    frame = _frame(["a", "b"], ["2015-01-05", "2015-01-06"], ids=["x", "x"])
    with pytest.raises(ValueError, match="duplicate"):
        annotate.draw_sample(frame, n_total=1)


def test_draw_refuses_to_oversample_the_frame():
    with pytest.raises(ValueError, match="cannot draw"):
        annotate.draw_sample(_corpus(n=50), n_total=800)


def test_strata_cross_year_with_tag_presence():
    frame = _frame(
        ["a", "b", "c", "d"],
        ["2015-01-05", "2015-01-06", "2016-01-05", "2016-01-06"],
        tickers=[["AAPL"], [], ["MSFT"], []],
    )
    s = annotate.stratum_labels(frame)
    assert list(s) == ["2015|tagged", "2015|untagged", "2016|tagged", "2016|untagged"]


# ----------------------------------------------------------------- split


def test_no_article_group_straddles_the_calibration_boundary():
    """The property the whole split exists for."""
    drawn = annotate.draw_sample(_corpus(), n_total=400)
    assigned = annotate.assign_parts(drawn, n_calibration=100)
    parts_per_group = assigned.groupby("group_id")["part"].nunique()
    assert (parts_per_group == 1).all()


def test_parts_are_disjoint_and_exhaustive():
    drawn = annotate.draw_sample(_corpus(), n_total=400)
    assigned = annotate.assign_parts(drawn, n_calibration=100)
    cal = set(assigned.loc[assigned["part"] == "calibration", "headline_id"])
    ev = set(assigned.loc[assigned["part"] == "evaluation", "headline_id"])
    assert not (cal & ev)
    assert cal | ev == set(drawn["headline_id"])


def test_split_keeps_a_multi_row_group_together():
    """Built so that grouping actually binds, rather than hoping it does."""
    texts = ["Fed holds rates steady"] * 6 + [f"unique story {i}" for i in range(20)]
    dates = ["2015-03-0%d" % (i + 1) for i in range(6)] + ["2016-01-05"] * 20
    drawn = _frame(texts, dates)
    drawn["stratum"] = annotate.stratum_labels(drawn)
    drawn["group_id"] = annotate.article_groups(drawn)
    assert (drawn.groupby("group_id").size() == 6).any()
    assigned = annotate.assign_parts(drawn, n_calibration=5)
    assert (assigned.groupby("group_id")["part"].nunique() == 1).all()


# ------------------------------------------------------------ blindness


def test_blind_export_carries_only_id_and_text():
    drawn = annotate.draw_sample(_corpus(), n_total=100)
    assigned = annotate.assign_parts(drawn, n_calibration=25)
    out = annotate.blind_export(assigned)
    assert list(out.columns) == ["headline_id", "text"]
    for col in ("ts_utc", "tickers", "source", "part", "group_id", "stratum"):
        assert col not in out.columns


def test_blind_export_shuffles_presentation_order():
    """Neither time order nor source clusters may cue the annotator."""
    drawn = annotate.draw_sample(_corpus(), n_total=200)
    assigned = annotate.assign_parts(drawn, n_calibration=50)
    out = annotate.blind_export(assigned)
    ordered = sorted(out["headline_id"])
    assert list(out["headline_id"]) != ordered          # actually shuffled
    assert set(out["headline_id"]) == set(assigned["headline_id"])   # nothing lost


def test_blind_export_order_is_reproducible():
    drawn = annotate.draw_sample(_corpus(), n_total=100)
    assigned = annotate.assign_parts(drawn, n_calibration=25)
    a = annotate.blind_export(assigned)
    b = annotate.blind_export(assigned)
    assert list(a["headline_id"]) == list(b["headline_id"])


def test_shuffling_the_running_order_cannot_change_who_was_drawn():
    """The order seed and the draw seed are separate on purpose."""
    frame = _corpus()
    drawn = annotate.draw_sample(frame, n_total=200)
    assigned = annotate.assign_parts(drawn, n_calibration=50)
    a = annotate.blind_export(assigned, seed=1)
    b = annotate.blind_export(assigned, seed=2)
    assert list(a["headline_id"]) != list(b["headline_id"])
    assert set(a["headline_id"]) == set(b["headline_id"])


# -------------------------------------------------- pilot / second annotator


def test_pilot_items_come_only_from_calibration():
    drawn = annotate.draw_sample(_corpus(), n_total=400)
    assigned = annotate.assign_parts(drawn, n_calibration=100)
    pilot = annotate.pilot_items(assigned, n=60)
    cal = set(assigned.loc[assigned["part"] == "calibration", "headline_id"])
    assert len(pilot) == 60
    assert set(pilot).issubset(cal)


def test_pilot_refuses_when_calibration_is_too_small():
    drawn = annotate.draw_sample(_corpus(), n_total=100)
    assigned = annotate.assign_parts(drawn, n_calibration=10)
    with pytest.raises(ValueError, match="need"):
        annotate.pilot_items(assigned, n=60)


def test_second_annotator_subset_is_twenty_percent_of_the_sample():
    drawn = annotate.draw_sample(_corpus(), n_total=400)
    assigned = annotate.assign_parts(drawn, n_calibration=100)
    second = annotate.second_annotator_subset(assigned, fraction=0.2)
    assert len(second) == 80
    assert set(second).issubset(set(assigned["headline_id"]))


# ------------------------------------------------------------------ build


def test_build_writes_the_protocol_file_set(tmp_path):
    stats = annotate.build(_corpus(n=5000), out_dir=tmp_path)

    expected = {
        "sample_frame.csv": annotate.SAMPLE_FRAME_COLUMNS,
        "split_assignment.csv": annotate.SPLIT_COLUMNS,
        "to_label_primary.csv": annotate.BLIND_COLUMNS,
        "to_label_second.csv": annotate.BLIND_COLUMNS,
    }
    for name, cols in expected.items():
        got = pd.read_csv(tmp_path / name)
        assert list(got.columns) == cols, name
    assert (tmp_path / "provenance.md").exists()
    assert (tmp_path / "pilot_items.csv").exists()

    assert stats["n_drawn"] == config.ANNOTATION_N_TOTAL
    assert stats["n_calibration"] + stats["n_evaluation"] == stats["n_drawn"]
    assert stats["n_pilot"] == config.ANNOTATION_PILOT_N


def test_split_file_agrees_with_the_sample_frame(tmp_path):
    """`split_assignment.csv` is the authority; it must not contradict the frame."""
    annotate.build(_corpus(n=5000), out_dir=tmp_path)
    frame = pd.read_csv(tmp_path / "sample_frame.csv")
    split = pd.read_csv(tmp_path / "split_assignment.csv")
    merged = frame.merge(split, on="headline_id", suffixes=("_f", "_s"))
    assert len(merged) == len(frame)
    assert (merged["part_f"] == merged["part_s"]).all()
    assert (merged["group_id_f"] == merged["group_id_s"]).all()


def test_provenance_leaves_annotator_fields_blank(tmp_path):
    """Written at labelling time by the person doing it, not pre-filled.

    A template that invents an annotator, a date or a blindness confirmation is
    a reconstructed record, which is the one thing protocol section 6 says this
    file must not be.
    """
    annotate.build(_corpus(n=5000), out_dir=tmp_path)
    text = (tmp_path / "provenance.md").read_text(encoding="utf-8")
    assert text.count("TO BE COMPLETED") >= 4
    for field in ("Annotator(s)", "Dates", "Blindness", "Deviations"):
        assert field in text
    assert "presentation-order seed" in text


def test_build_records_whether_scores_already_existed(tmp_path):
    annotate.build(_corpus(n=5000), out_dir=tmp_path, scores_exist=True)
    assert "Scores in cache at draw time: **yes**" in (
        tmp_path / "provenance.md"
    ).read_text(encoding="utf-8")


# ------------------------------------------------------- the split as authority


def test_load_split_reads_the_stored_file(tmp_path):
    annotate.build(_corpus(n=5000), out_dir=tmp_path)
    split = annotate.load_split(tmp_path)
    assert list(split.columns) == annotate.SPLIT_COLUMNS
    assert len(split) == config.ANNOTATION_N_TOTAL


def test_load_split_refuses_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="stored artifact"):
        annotate.load_split(tmp_path)


def test_load_split_rejects_a_group_spanning_both_parts(tmp_path):
    """The one corruption that silently destroys the separation."""
    annotate.build(_corpus(n=5000), out_dir=tmp_path)
    path = tmp_path / "split_assignment.csv"
    split = pd.read_csv(path)
    # Force one group to hold a row from each part -- exactly the corruption a
    # changed grouping rule would introduce.
    cal_idx = split.index[split["part"] == "calibration"][0]
    ev_idx = split.index[split["part"] == "evaluation"][0]
    split.loc[ev_idx, "group_id"] = split.loc[cal_idx, "group_id"]
    split.to_csv(path, index=False)
    with pytest.raises(ValueError, match="span both parts"):
        annotate.load_split(tmp_path)


def test_load_split_rejects_a_duplicated_headline(tmp_path):
    annotate.build(_corpus(n=5000), out_dir=tmp_path)
    path = tmp_path / "split_assignment.csv"
    split = pd.read_csv(path)
    pd.concat([split, split.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="more than one part"):
        annotate.load_split(tmp_path)
