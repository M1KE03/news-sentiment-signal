"""B11 / R06a: draw the blind annotation sample and export it.

Act 1's evaluation set. This module draws the sample, assigns article groups,
separates calibration from evaluation **by group**, and writes the annotator's
files with nothing in them but an id and a headline.

It produces no labels and computes no metric. Labelling is a human dependency;
everything here is the tooling that has to exist before a person can start, and
all of it is testable without a single label (audit finding A11).

The authority for every rule here is
[the validation protocol](../docs/validation-protocol.md) sections 3-6.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data import _token_set_overlap

SAMPLE_FRAME_COLUMNS = ["headline_id", "group_id", "stratum", "part"]
SPLIT_COLUMNS = ["headline_id", "group_id", "part"]
BLIND_COLUMNS = ["headline_id", "text"]

# Columns that must never reach an annotator's file. Presence of any of these
# is what "prediction-blind" actually means, so it is checked rather than
# trusted: dates cue the market outcome, tickers cue the subject, and a score
# column would simply hand over the answer.
FORBIDDEN_IN_BLIND_EXPORT = (
    "ts_utc", "ts_et", "date", "tickers", "source", "text_norm",
    "score", "label", "pred", "finbert", "lm", "vader",
    "ret", "ret_lead1", "stratum", "part", "group_id",
)


def article_groups(
    df: pd.DataFrame, overlap: float = None, text_col: str = "text_norm"
) -> pd.Series:
    """Group headlines that are the same article, as an integer label per row.

    Two headlines join the same group when they share a normalized text, or
    when their token sets overlap by at least `overlap`. Groups are the
    connected components of that relation.

    **This is not the dedup cluster, and the difference is the point.** Dedup
    is windowed at three days, so a story republished ten days later survives
    it twice and appears as two rows of the corpus. Those two rows are one
    article for leakage purposes: if one lands in calibration and the other in
    evaluation, a threshold is fitted on a sentence that is then used to
    evaluate. So this relation carries **no window**.

    It is computed over the drawn sample rather than the corpus, by exact
    all-pairs comparison. That is deliberate: leakage can only occur between
    the calibration and evaluation parts, and both are subsets of the ~800
    drawn items, so all-pairs is ~320k comparisons -- instant, exact, and free
    of the one-token blind spot the corpus-wide blocking has. A corpus-wide
    index over 869,183 rows would answer a question nobody asks.
    """
    overlap = config.ANNOTATION_GROUP_OVERLAP if overlap is None else overlap
    n = len(df)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    texts = df[text_col].fillna("").astype(str).to_numpy()

    # Exact matches first, by hashing rather than by comparison.
    seen: dict[str, int] = {}
    for i, t in enumerate(texts):
        if t in seen:
            union(seen[t], i)
        else:
            seen[t] = i

    # Then the near relation, all pairs.
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if _token_set_overlap(texts[i], texts[j]) >= overlap:
                union(i, j)

    roots = [find(i) for i in range(n)]
    # Relabel to dense 0..k-1 in order of first appearance, so group ids do not
    # depend on the frame's row order beyond the order itself.
    remap: dict[int, int] = {}
    out = []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        out.append(remap[r])
    return pd.Series(out, index=df.index, name="group_id")


def stratum_labels(df: pd.DataFrame) -> pd.Series:
    """Protocol section 3's strata, as one string per row.

    Calendar year, crossed with ticker-tag presence. Untagged headlines are
    often macro rather than single-name and are the ones the lexicons handle
    worst, so they are stratified rather than left to chance.

    The protocol also lists source/publisher "if the collection carries one".
    This corpus is a single publisher by construction -- the Benzinga
    sub-corpus, with a mandatory domain filter -- so that stratum would have
    one level and is omitted. Recorded here rather than silently dropped.
    """
    year = pd.to_datetime(df["ts_utc"], utc=True).dt.year.astype(str)
    tagged = df["tickers"].map(lambda v: "tagged" if len(v) else "untagged")
    return (year + "|" + tagged).rename("stratum")


def _allocate(counts: pd.Series, total: int) -> pd.Series:
    """Proportional allocation with largest-remainder rounding.

    Proportional, not equal: the evaluation set should look like the corpus
    being measured. Largest-remainder makes the parts sum to `total` exactly
    without letting rounding quietly favour any stratum.
    """
    share = counts / counts.sum() * total
    base = np.floor(share).astype(int)
    remainder = total - int(base.sum())
    if remainder:
        # Ties broken by stratum name so the result does not depend on the
        # order pandas happened to produce the groups in.
        order = sorted(counts.index, key=lambda k: (-(share[k] - base[k]), str(k)))
        for k in order[:remainder]:
            base[k] += 1
    return base.clip(upper=counts)


def draw_sample(
    frame: pd.DataFrame, n_total: int = None, seed: int = None
) -> pd.DataFrame:
    """Stratified draw without replacement. Made once (protocol section 3).

    Returns the drawn rows with `stratum` and `group_id` attached.
    """
    n_total = config.ANNOTATION_N_TOTAL if n_total is None else n_total
    seed = config.SEED if seed is None else seed

    if frame["headline_id"].duplicated().any():
        raise ValueError("frame has duplicate headline_id; draw would double-count")
    if len(frame) < n_total:
        raise ValueError(f"frame has {len(frame)} rows, cannot draw {n_total}")

    work = frame.copy()
    work["stratum"] = stratum_labels(work)
    counts = work["stratum"].value_counts().sort_index()
    alloc = _allocate(counts, n_total)

    rng = np.random.default_rng(seed)
    picks = []
    for stratum in counts.index:            # sorted, so the draw is reproducible
        k = int(alloc[stratum])
        if not k:
            continue
        cell = work[work["stratum"] == stratum]
        # Sort by headline_id first: the draw must not depend on the frame's
        # row order, only on the seed and the cell's membership.
        cell = cell.sort_values("headline_id")
        idx = rng.choice(len(cell), size=k, replace=False)
        picks.append(cell.iloc[np.sort(idx)])

    drawn = pd.concat(picks, ignore_index=True).sort_values("headline_id")
    drawn = drawn.reset_index(drop=True)
    drawn["group_id"] = article_groups(drawn)
    drawn.attrs["allocation"] = alloc.to_dict()
    return drawn


def assign_parts(
    drawn: pd.DataFrame, n_calibration: int = None, seed: int = None
) -> pd.DataFrame:
    """Split into calibration and evaluation **by article group**.

    A whole group goes to one part. Otherwise a syndicated headline can be
    threshold-fitted on Monday and evaluated on Tuesday, which is leakage even
    though the row ids differ.

    Groups are shuffled and taken until the calibration target is reached, so
    the realised sizes can differ from 200/600 by less than the largest group.
    The realised counts are what the split file records; the target is not
    enforced by breaking a group.
    """
    n_calibration = (
        config.ANNOTATION_N_CALIBRATION if n_calibration is None else n_calibration
    )
    seed = config.SEED if seed is None else seed

    sizes = drawn.groupby("group_id").size()
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.to_numpy())

    calibration, running = set(), 0
    for g in order:
        if running >= n_calibration:
            break
        calibration.add(int(g))
        running += int(sizes[g])

    out = drawn.copy()
    out["part"] = np.where(
        out["group_id"].isin(calibration), "calibration", "evaluation"
    )
    return out


def pilot_items(assigned: pd.DataFrame, n: int = None, seed: int = None) -> pd.Index:
    """The 60 calibration items labelled first (protocol section 8).

    Calibration only. The pilot may change the target size and the rubric's
    wording; it may not change the evaluation set's membership.
    """
    n = config.ANNOTATION_PILOT_N if n is None else n
    seed = config.ANNOTATION_ORDER_SEED if seed is None else seed
    cal = assigned[assigned["part"] == "calibration"].sort_values("headline_id")
    if len(cal) < n:
        raise ValueError(f"calibration part has {len(cal)} items, need {n}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cal), size=n, replace=False)
    return pd.Index(cal.iloc[np.sort(idx)]["headline_id"])


def second_annotator_subset(
    assigned: pd.DataFrame, fraction: float = None, seed: int = None
) -> pd.Index:
    """The independent second annotator's 20% subset, for kappa (section 6)."""
    fraction = (
        config.ANNOTATION_SECOND_FRACTION if fraction is None else fraction
    )
    seed = config.ANNOTATION_ORDER_SEED if seed is None else seed
    ordered = assigned.sort_values("headline_id")
    k = int(round(len(ordered) * fraction))
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(len(ordered), size=k, replace=False)
    return pd.Index(ordered.iloc[np.sort(idx)]["headline_id"])


def blind_export(
    assigned: pd.DataFrame, ids: pd.Index = None, seed: int = None
) -> pd.DataFrame:
    """`headline_id` and `text`, shuffled. Nothing else reaches the annotator.

    Presentation order is shuffled with its own recorded seed so that neither
    time order nor source clusters cue the annotator, and so that reshuffling
    can never disturb which rows were drawn.
    """
    seed = config.ANNOTATION_ORDER_SEED if seed is None else seed
    sub = assigned if ids is None else assigned[assigned["headline_id"].isin(ids)]
    sub = sub.sort_values("headline_id").reset_index(drop=True)

    leaked = [c for c in sub.columns if c.lower() in FORBIDDEN_IN_BLIND_EXPORT]
    out = sub.loc[:, BLIND_COLUMNS].copy()
    if any(c in out.columns for c in leaked):
        raise AssertionError(f"blind export would leak {leaked}")

    rng = np.random.default_rng(seed)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    return out


def load_split(out_dir: Path = None) -> pd.DataFrame:
    """Read `split_assignment.csv`. **This file is the authority, not a seed.**

    Protocol section 4 is explicit that the separation is stored rather than
    recomputed: "Recomputing the split from a seed at analysis time is not
    sufficient: a change to the grouping rule would silently move items across
    the boundary." So every consumer reads this file, and the invariant it
    exists to protect is re-checked on the way in rather than assumed.
    """
    out_dir = Path(config.ANNOTATION_DIR if out_dir is None else out_dir)
    path = out_dir / "split_assignment.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The calibration/evaluation separation is a "
            "stored artifact, not something to recompute: run "
            "`src.annotate.build` once, commit the result, and never redraw it "
            "without a dated decision-log entry (P25)."
        )
    split = pd.read_csv(path)

    missing = [c for c in SPLIT_COLUMNS if c not in split.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}")
    if split["headline_id"].duplicated().any():
        raise ValueError(f"{path} assigns some headline to more than one part")
    bad = set(split["part"]) - {"calibration", "evaluation"}
    if bad:
        raise ValueError(f"{path} has unknown part(s) {sorted(bad)}")
    straddling = split.groupby("group_id")["part"].nunique()
    if (straddling > 1).any():
        offenders = list(straddling[straddling > 1].index[:5])
        raise ValueError(
            f"{path} lets article group(s) {offenders} span both parts, so a "
            "threshold could be fitted on one member and evaluated on another"
        )
    return split


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(
    frame: pd.DataFrame, out_dir: Path = None, scores_exist: bool = False
) -> dict:
    """Draw, group, split, export. Writes the protocol section 6 file set.

    `scores_exist` is recorded, not enforced: the protocol requires stating
    whether scores were already in the cache when labelling began, since the
    blindness claim is different in each case.
    """
    out_dir = Path(config.ANNOTATION_DIR if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drawn = draw_sample(frame)
    assigned = assign_parts(drawn)
    pilot = pilot_items(assigned)
    second = second_annotator_subset(assigned)

    assigned.loc[:, SAMPLE_FRAME_COLUMNS].to_csv(
        out_dir / "sample_frame.csv", index=False
    )
    assigned.loc[:, SPLIT_COLUMNS].to_csv(
        out_dir / "split_assignment.csv", index=False
    )
    blind_export(assigned).to_csv(out_dir / "to_label_primary.csv", index=False)
    blind_export(assigned, second).to_csv(
        out_dir / "to_label_second.csv", index=False
    )
    pd.DataFrame({"headline_id": pilot}).to_csv(
        out_dir / "pilot_items.csv", index=False
    )

    counts = assigned["part"].value_counts()
    stats = {
        "n_drawn": len(assigned),
        "n_calibration": int(counts.get("calibration", 0)),
        "n_evaluation": int(counts.get("evaluation", 0)),
        "n_groups": int(assigned["group_id"].nunique()),
        "n_multi_row_groups": int(
            (assigned.groupby("group_id").size() > 1).sum()
        ),
        "largest_group": int(assigned.groupby("group_id").size().max()),
        "n_pilot": len(pilot),
        "n_second": len(second),
        "n_strata": int(assigned["stratum"].nunique()),
        "seed_draw": config.SEED,
        "seed_order": config.ANNOTATION_ORDER_SEED,
        "frame_rows": len(frame),
        "scores_existed_at_draw": bool(scores_exist),
    }
    _write_provenance(out_dir, stats, assigned)
    return stats


def _write_provenance(out_dir: Path, stats: dict, assigned: pd.DataFrame) -> None:
    """The section 6 record, written at draw time.

    Annotator identity, dates and deviations are left as explicit blanks: they
    are written at labelling time by the person doing it, and a template that
    pre-fills them invites reconstruction after the fact, which is the one
    thing the protocol says this record must not be.
    """
    proto = Path(config.ROOT) / "docs" / "validation-protocol.md"
    alloc = assigned["stratum"].value_counts().sort_index()
    lines = [
        "# Annotation provenance",
        "",
        f"Sample drawn {pd.Timestamp.utcnow().date().isoformat()} by "
        "`src/annotate.build`. Fields marked TO BE COMPLETED are filled in at "
        "labelling time by the annotator, not reconstructed afterwards.",
        "",
        "| Field | Content |",
        "|---|---|",
        f"| Rubric version | `v1`, `docs/validation-protocol.md` sha256 "
        f"`{_sha256(proto)[:16]}...` |",
        "| Annotator(s) | **TO BE COMPLETED** - role not name; state whether "
        "independent of the analyst |",
        "| Dates | **TO BE COMPLETED** - start and end of each session |",
        "| Blindness | Export contains `headline_id` and `text` only. Scores "
        f"in cache at draw time: **{'yes' if stats['scores_existed_at_draw'] else 'no'}**. "
        "Annotator confirmation **TO BE COMPLETED** |",
        f"| Instrument | `to_label_primary.csv`, presentation-order seed "
        f"`{stats['seed_order']}` |",
        "| Deviations | **TO BE COMPLETED** - every rubric question that arose "
        "and how it was resolved |",
        "",
        "## The draw",
        "",
        f"- Frame: {stats['frame_rows']:,} deduplicated headlines "
        "(`interim/headlines.parquet`)",
        f"- Drawn: **{stats['n_drawn']}** across {stats['n_strata']} strata "
        f"(year x ticker-tag presence), proportional allocation, seed "
        f"`{stats['seed_draw']}`",
        f"- Article groups: {stats['n_groups']}, of which "
        f"{stats['n_multi_row_groups']} hold more than one item "
        f"(largest {stats['largest_group']})",
        f"- Calibration **{stats['n_calibration']}** / evaluation "
        f"**{stats['n_evaluation']}**, assigned by whole group",
        f"- Pilot: {stats['n_pilot']} calibration items (`pilot_items.csv`)",
        f"- Second annotator: {stats['n_second']} items (`to_label_second.csv`)",
        "",
        "## Realised stratum counts",
        "",
        "| Stratum | Drawn |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in alloc.items()]

    # A stratum dimension with one level stratifies nothing. Say so, rather
    # than letting the count imply a balance that was never at stake.
    collapsed = []
    if assigned["tickers"].map(len).gt(0).all():
        collapsed.append(
            "**ticker-tag presence**: every headline in this corpus carries at "
            "least one tag (0.00% untagged), so this dimension has one level"
        )
    elif assigned["tickers"].map(len).eq(0).all():
        collapsed.append("**ticker-tag presence**: no headline carries a tag")
    if assigned["source"].nunique() == 1:
        collapsed.append(
            "**source/publisher**: the universe is a single publisher by "
            "construction (the Benzinga sub-corpus, mandatory domain filter)"
        )
    if collapsed:
        lines += ["", "### Strata that collapsed to one level", ""]
        lines += [f"- {c}" for c in collapsed]
        lines += [
            "",
            "These are recorded because a stratum with one level stratifies "
            "nothing, and a reader counting dimensions would otherwise assume "
            "a balance that was never at stake.",
        ]
    lines += [
        "",
        "## What must not happen to these files",
        "",
        "- The evaluation part is touched **once**, for the reported numbers. "
        "Thresholds are fitted on calibration only.",
        "- `split_assignment.csv` is the authority for the separation. It is "
        "not recomputed from a seed at analysis time: a change to the grouping "
        "rule would silently move items across the boundary.",
        "- The draw is made once. Repeating it requires a dated entry in the "
        "decision log with the reason (P25).",
    ]
    (out_dir / "provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
