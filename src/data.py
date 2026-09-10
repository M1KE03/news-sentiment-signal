"""Loading and hygiene for the two raw inputs: the news dump and SPY/VIX.

Produces `interim/headlines.parquet` and `interim/market.parquet` exactly to the
schemas in §4. Nothing here knows about scoring, alignment or regressions.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import config

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

HEADLINE_COLUMNS = [
    "source_row_id", "headline_id", "text", "text_norm",
    "ts_utc", "ts_et", "tickers", "source",
]

# Added by `dedup`. `cluster_id` is the representative's `headline_id`, which
# joins a surviving row to its lineage record; `n_cluster_rows` is how many raw
# rows it stands for.
CLEAN_COLUMNS = HEADLINE_COLUMNS + ["cluster_id", "n_cluster_rows"]

LINEAGE_COLUMNS = [
    "source_row_id", "headline_id", "cluster_id", "kept",
    "eliminated_by", "relation",
]

# What a single parsed chunk carries. `source_row_id` is assigned once, after
# every chunk has been concatenated, so that it numbers the whole filtered read
# rather than restarting per chunk.
_PARSED_COLUMNS = [c for c in HEADLINE_COLUMNS if c != "source_row_id"]

# Column names, source timezone and timestamp format per dataset. Verified by
# reading the file's bytes in the B03 audit, not taken from documentation --
# FNSPID's dataset card documents no column semantics at all.
#
# The standalone "benzinga" spec is gone: there is no separate Benzinga dataset
# in this study. Benzinga is a *sub-corpus* of FNSPID's All_external.csv,
# selected by URL host, and the file also carries Reuters, Bloomberg, Zacks,
# SeekingAlpha and the Russian-language lenta.ru. See docs/archive/data-audit-fnspid.md.
_SOURCE_SPEC = {
    "fnspid": {
        "text": "Article_title",
        "ts": "Date",
        "tickers": "Stock_symbol",
        "url": "Url",
        "tz": "UTC",
        # Every conforming value is "YYYY-MM-DD HH:MM:SS UTC". The ~0.05% that
        # do not are fragments of article body text, which is how the embedded
        # newlines in the Article field announce themselves.
        "ts_pattern": r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$",
        "ts_suffix": " UTC",
    },
}


def normalize_text(s: pd.Series) -> pd.Series:
    """Dedup key only. Never scored -- scorers see the original `text`."""
    out = s.fillna("").str.lower()
    out = out.map(lambda t: _PUNCT.sub(" ", t))
    return out.map(lambda t: _WS.sub(" ", t).strip())


def _headline_id(text_norm: str, ts_utc: pd.Timestamp) -> str:
    key = f"{text_norm}|{ts_utc.isoformat()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


_SENTINEL: tuple[str, ...] = ("__use_config__",)


def _host_matches(host: str, domains: tuple[str, ...]) -> bool:
    """Exact host or true subdomain -- never a substring.

    The filter is mandatory rather than hygiene: `All_external.csv` concatenates
    five-plus corpora including the Russian-language `lenta.ru`, so a leak is a
    different corpus, not a little extra noise. Substring membership
    (`any(d in h for d in domains)`) accepted any host that merely *contained*
    an approved domain, so `notbenzinga.com` and `benzinga.com.evil.example`
    both passed. No such host was observed in the saved corpus; the check is
    tightened because the filter's job is to be the boundary, not to be
    approximately right (audit A15, closed R12 follow-up).

    `host` has already had a leading `www.` stripped by the extraction regex.
    """
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in (x.lower() for x in domains))


def _parse_chunk(raw: pd.DataFrame, spec: dict, domains: tuple[str, ...],
                 start, end, stats: dict) -> pd.DataFrame:
    """One chunk: validate stamps, filter by source, build the schema."""
    stats["n_read"] += len(raw)

    text_raw = raw[spec["text"]].astype("string").str.strip()
    ts_raw = raw[spec["ts"]].astype("string")

    # Reject malformed stamps explicitly rather than letting `errors="coerce"`
    # turn them into NaT alongside genuinely missing values -- these are body
    # text bleeding through a record boundary, and their count is evidence.
    conforms = ts_raw.str.match(spec["ts_pattern"]).fillna(False)
    stats["n_bad_timestamp"] += int((~conforms).sum())
    if stats["n_bad_timestamp"] and len(stats["bad_timestamp_examples"]) < 3:
        stats["bad_timestamp_examples"].extend(
            ts_raw[~conforms].dropna().astype(str).head(3).tolist()
        )
    raw, text_raw = raw[conforms], text_raw[conforms]

    ts_utc = pd.to_datetime(
        ts_raw[conforms].str.removesuffix(spec["ts_suffix"]), errors="coerce"
    ).dt.tz_localize(spec["tz"]).dt.tz_convert("UTC")

    url_col = spec.get("url")
    host = (
        raw[url_col].astype("string").str.extract(r"https?://(?:www\.)?([^/]+)")[0]
        if url_col and url_col in raw.columns
        else pd.Series(pd.NA, index=raw.index, dtype="string")
    )

    if domains:
        keep = host.fillna("").apply(lambda h: _host_matches(h, domains))
        stats["n_wrong_source"] += int((~keep).sum())
        for h in host[~keep].fillna("(no url)").value_counts().head(20).items():
            stats["rejected_hosts"][h[0]] = stats["rejected_hosts"].get(h[0], 0) + int(h[1])
        raw, text_raw, ts_utc, host = raw[keep], text_raw[keep], ts_utc[keep], host[keep]

    tick_col = spec.get("tickers")
    if tick_col and tick_col in raw.columns:
        tickers = raw[tick_col].fillna("").map(
            lambda v: [t.strip().upper() for t in str(v).split(",") if t.strip()]
        )
    else:
        tickers = pd.Series([[] for _ in range(len(raw))], index=raw.index)

    df = pd.DataFrame(
        {"text": text_raw, "ts_utc": ts_utc, "tickers": tickers, "source": host}
    )
    df = df[df["text"].notna() & (df["text"].str.len() > 0) & df["ts_utc"].notna()]

    if start is not None:
        df = df[df["ts_utc"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        # Strictly less than the day after `end`, so the whole of `end` is kept
        # and nothing later is. The inclusive form admitted the next day's
        # 00:00 stamp, which for a date-only corpus is a whole extra day.
        df = df[df["ts_utc"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]

    df["text_norm"] = normalize_text(df["text"])
    df = df[df["text_norm"].str.len() > 0]
    df["ts_et"] = df["ts_utc"].dt.tz_convert(config.TZ_MARKET)
    df["headline_id"] = [
        _headline_id(tn, t) for tn, t in zip(df["text_norm"], df["ts_utc"])
    ]
    df["text"] = df["text"].astype(str)
    df["source"] = df["source"].astype(str)
    return df.loc[:, _PARSED_COLUMNS]


def load_news(
    path,
    source: Literal["fnspid"] = "fnspid",
    domains: tuple[str, ...] | None = _SENTINEL,
    start: str | None = None,
    end: str | None = None,
    chunksize: int | None = 500_000,
    on_chunk=None,
) -> pd.DataFrame:
    """Read a raw news dump into the `headlines` schema, filtered to its universe.

    **Source filtering is mandatory, not hygiene.** FNSPID's `All_external.csv`
    concatenates five-plus sub-corpora, one of which (`lenta.ru`) is a
    Russian-language general news site. Scoring that text with an English
    financial model returns numbers that look fine and mean nothing. `domains`
    therefore defaults to `config.NEWS_SOURCE_DOMAINS`; pass an empty tuple to
    disable filtering, which is a deliberate act rather than an oversight.

    The source timezone comes from `_SOURCE_SPEC`, applied explicitly, never
    inferred from the data.

    Records whose timestamp does not match the documented pattern are dropped
    and counted separately from missing values: in FNSPID they are article body
    text bleeding across a record boundary, and the count is evidence about
    parsing rather than noise.

    `path` may be a filesystem path or any binary file-like object, which is
    what lets the 5.7 GB corpus be parsed straight off an HTTP stream without
    ever landing on disk.

    `chunksize` streams the CSV, since the real file is 5.7 GB. Set it to None
    to read at once. Chunking changes nothing about the result.

    `on_chunk(i, kept_frame, stats)` is called after each chunk, for progress
    reporting on a long run.

    Counts are attached to `.attrs["load_stats"]` -- what was read, what was
    rejected for a bad timestamp, and what was rejected as the wrong source,
    with the rejected hosts named.
    """
    if source not in _SOURCE_SPEC:
        raise ValueError(
            f"unknown source {source!r}; expected one of {list(_SOURCE_SPEC)}. "
            "Note that 'benzinga' is no longer a source: it is a sub-corpus of "
            "FNSPID selected via `domains`."
        )
    spec = _SOURCE_SPEC[source]
    if domains is _SENTINEL:
        domains = tuple(config.NEWS_SOURCE_DOMAINS)

    is_buffer = hasattr(path, "read")
    if not is_buffer:
        path = Path(path)
    stats = {
        "n_read": 0, "n_bad_timestamp": 0, "n_wrong_source": 0,
        "bad_timestamp_examples": [], "rejected_hosts": {},
        "domains": tuple(domains), "path": "<stream>" if is_buffer else str(path),
    }

    if not is_buffer and path.suffix in {".parquet", ".pq"}:
        chunks = [pd.read_parquet(path)]
    elif chunksize:
        chunks = pd.read_csv(path, dtype=str, chunksize=chunksize)
    else:
        chunks = [pd.read_csv(path, dtype=str)]

    frames, checked = [], False
    for i, raw in enumerate(chunks):
        if not checked:
            missing = [spec[k] for k in ("text", "ts") if spec[k] not in raw.columns]
            if missing:
                name = "stream" if is_buffer else path.name
                raise KeyError(f"{name} is missing expected column(s) {missing}")
            checked = True
        kept = _parse_chunk(raw, spec, domains, start, end, stats)
        frames.append(kept)
        if on_chunk is not None:
            on_chunk(i, kept, stats)

    out = (
        pd.concat(frames, ignore_index=True) if frames
        else pd.DataFrame(columns=_PARSED_COLUMNS)
    )
    # Pin the string dtypes after concatenation. Without this the result depends
    # on how many chunks were read -- a single frame keeps pandas' inferred
    # dtype while a concat of several falls back to object -- so the schema
    # would vary with a performance knob, and a later merge on headline_id
    # could mismatch. Chunking must change nothing observable.
    for col in ("headline_id", "text", "text_norm", "source"):
        out[col] = out[col].astype("string")

    # `source_row_id` is assigned in FILE order, before any sort, and is the
    # only row identifier the pipeline has: `headline_id` is sha1(text_norm |
    # ts_utc), so a story filed under three tickers on one date yields ONE id
    # for THREE rows (518,332 such rows in the real raw corpus). Position is
    # well defined here because R02 pins the source by revision, SHA-256 and
    # byte length, so the filtered read is reproducible.
    out["source_row_id"] = np.arange(len(out), dtype="int64")

    # Stable sort. The default quicksort is not stable, and in a date-only
    # corpus every headline on a date shares the identical 00:00 UTC stamp, so
    # an unstable sort permutes same-day rows arbitrarily (A12/D-2).
    out = out.sort_values("ts_utc", kind="stable").reset_index(drop=True)
    out = out.loc[:, HEADLINE_COLUMNS]
    stats["n_kept"] = len(out)
    out.attrs["load_stats"] = stats
    return out


def assign_source_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Number rows 0..n-1 in their current order, as `source_row_id`.

    `load_news` does this itself. This helper exists for callers holding a frame
    that predates the column -- a raw artifact written before R03b, or a test
    fixture -- so that the identifier is always assigned at one explicit place
    rather than inferred from whatever order a frame happens to be in.
    """
    out = df.copy()
    out["source_row_id"] = np.arange(len(out), dtype="int64")
    return out


def _missable_token_length(overlap: float) -> int:
    """Smallest unique-token count at which the blocking can miss a duplicate.

    Deletion signatures catch token sets differing by one element, so the blind
    spot starts at a two-token difference, which clears `overlap` when
    `(L - 2) / L >= overlap`, i.e. `L >= 2 / (1 - overlap)`.

    Computed in exact rational arithmetic. `int(np.ceil(2 / (1 - 0.90)))`
    returns **21**, because `1 - 0.90` is `0.09999999999999998` in binary
    floating point and the quotient lands a hair above 20. The documented
    boundary is 20, and a 20-token pair differing by two tokens scores exactly
    `18/20 = 0.90` and does clear the threshold, so the float form silently
    excluded 14,494 real rows from the quoted exposure (A12/D-4).
    """
    if overlap >= 1.0:
        return 10 ** 9
    from fractions import Fraction

    q = 2 / (1 - Fraction(str(overlap)))
    return -(-q.numerator // q.denominator)


def _token_set_overlap(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _deletion_signatures(tokens: tuple[str, ...]) -> list[int]:
    """Hashes of the token set, and of the set with any one token removed.

    Two token sets differing by at most one element always share one of these,
    so they generate candidate pairs in O(len) instead of by comparing every
    pair. Standard deletion-neighbourhood blocking.
    """
    sigs = [hash(tokens)]
    if len(tokens) > 1:
        sigs.extend(hash(tokens[:i] + tokens[i + 1 :]) for i in range(len(tokens)))
    return sigs


def near_duplicate_keep(
    text_norm: pd.Series,
    ts: pd.Series,
    window_days: int = 3,
    overlap: float = 0.90,
    return_eliminator: bool = False,
):
    """Boolean keep-mask: drop a headline that near-repeats a kept earlier one.

    **Why blocking rather than pairwise.** The obvious implementation compares
    each row with every earlier row still inside the time window. With date-only
    stamps a 3-day window holds every headline from three days -- thousands of
    them -- so on a multi-million-row corpus that is billions of comparisons and
    simply does not finish.

    Candidates are instead generated by deletion signatures and only then
    verified with the exact `_token_set_overlap` test, so **no pair is ever
    accepted without being checked**. Signatures are held in a sliding window and
    evicted once they age out, which keeps the index at a few thousand entries
    regardless of corpus size.

    **Known limit, and its exposure.** Signatures catch token sets differing by
    at most one element. A pair differing by d tokens clears the threshold when
    `(L - d) / L >= overlap`, so at 0.9 a two-token difference needs 20 or more
    unique tokens; below that length the blocking is exact. Headlines are short,
    and `dedup` reports the share of rows at or above that length, so the
    exposure is quoted rather than assumed away.

    Note the flip side of the same arithmetic: at 0.9 a *nine*-token headline
    differing by one token scores 8/9 = 0.889 and is **kept**. The threshold is
    strict on short text by design, and most of the real duplication in this
    corpus is exact rather than near, so it is caught upstream.

    With `return_eliminator=True`, also returns the positional index of the kept
    row each dropped row was eliminated against, `-1` where the row was kept.
    That is what lets `dedup` record lineage instead of only a count (R03b).
    """
    from collections import deque

    order = np.argsort(ts.to_numpy(), kind="stable")
    times = ts.to_numpy()[order]
    texts = text_norm.to_numpy()[order]
    window = np.timedelta64(window_days, "D")

    keep = np.ones(len(order), dtype=bool)
    elim = np.full(len(order), -1, dtype=np.int64)   # sorted-position of the eliminator
    index: dict[int, list[int]] = {}
    live: deque = deque()          # (position, signatures) of kept rows in window
    sets: dict[int, set] = {}

    for pos in range(len(order)):
        now = times[pos]
        while live and now - times[live[0][0]] > window:
            old_pos, old_sigs = live.popleft()
            for s in old_sigs:
                bucket = index.get(s)
                if bucket:
                    bucket.remove(old_pos)
                    if not bucket:
                        del index[s]
            sets.pop(old_pos, None)

        toks = tuple(sorted(set(str(texts[pos]).split())))
        if not toks:
            keep[pos] = False
            continue

        sigs = _deletion_signatures(toks)
        this = set(toks)
        seen: set[int] = set()
        duplicate = -1
        for sig in sigs:
            for cand in index.get(sig, ()):
                if cand in seen:
                    continue
                seen.add(cand)
                other = sets[cand]
                if len(this & other) / max(len(this), len(other)) >= overlap:
                    duplicate = cand
                    break
            if duplicate >= 0:
                break

        if duplicate >= 0:
            keep[pos] = False
            elim[pos] = duplicate
        else:
            for s in sigs:
                index.setdefault(s, []).append(pos)
            sets[pos] = this
            live.append((pos, sigs))

    out = np.ones(len(order), dtype=bool)
    out[order] = keep
    if not return_eliminator:
        return out
    # Translate eliminator positions from sorted space back to input space.
    elim_in = np.full(len(order), -1, dtype=np.int64)
    has = elim >= 0
    elim_in[order[has]] = order[elim[has]]
    return out, elim_in


def dedup(
    df: pd.DataFrame,
    near_dupe: bool = True,
    window_days: int = 3,
    overlap: float = 0.90,
    return_lineage: bool = False,
):
    """Collapse duplicate headlines, keeping a stated representative and lineage.

    Implements the [dedup lineage contract](../docs/archive/dedup-lineage-contract.md)
    (R03b, audit finding A12). Four things changed from the previous version,
    each of which was reproduced against the running code first.

    **The representative is chosen by a total order**, `(ts_utc,
    source_row_id)`, not by an unspecified sort. The previous version sorted on
    `ts_utc` alone with pandas' default quicksort, which is not stable -- and in
    a date-only corpus *every* headline on a date carries the identical
    `00:00 UTC` stamp, so every same-day duplicate group was a tie broken by
    whatever order the rows arrived in. Permuting the real 1.4M-row corpus
    changed 2,363 surviving ids and the retained ticker tags of 10.78% of
    survivors. Referencing `source_row_id` -- a position in the pinned raw
    artifact -- makes the output a function of the input *set*.

    **The exact window re-anchors on the last kept occurrence**, not on the
    first-ever one. Previously a text repeated on days 0, 1, 10 and 11 kept both
    day 10 and day 11, because both lie outside three days of day 0; the near
    pass then removed day 11 and counted it as a *near* duplicate, and with
    `near_dupe=False` it survived outright. On the real corpus this
    misattributed 107,485 rows: the true split is 539,087 exact / 4,245 near,
    not 431,602 / 111,717.

    **Ticker tags are unioned across the cluster.** A story filed under three
    tickers previously survived with one tag, discarding 75.7% of all distinct
    (cluster, ticker) pairs corpus-wide, after which `audit.concentration_profile`
    read the survivors as "companies covered".

    **Lineage is recorded** rather than only counted, so the dedup rate is
    recomputable from the artifact instead of only reproducible by re-running.

    Returns `(clean, stats)`, or `(clean, stats, lineage)` when
    `return_lineage=True`. `lineage` has one row per input row.
    """
    if "source_row_id" not in df.columns:
        raise ValueError(
            "dedup requires a 'source_row_id' column: the representative rule is "
            "(ts_utc, source_row_id), and without it ties fall back to input "
            "order, which is exactly the defect this implements away (A12). "
            "Frames from load_news carry it; for one that predates it, call "
            "data.assign_source_row_ids first, at a point where the row order is "
            "known to be the pinned artifact's."
        )
    if df["source_row_id"].duplicated().any():
        raise ValueError("source_row_id must be unique; it identifies a raw row")

    n_in = len(df)
    df = df.sort_values(["ts_utc", "source_row_id"], kind="stable").reset_index(drop=True)

    if not n_in:
        empty_lineage = pd.DataFrame(columns=LINEAGE_COLUMNS)
        stats = {
            "n_in": 0, "n_out": 0, "n_exact_dropped": 0, "n_near_dropped": 0,
            "dedup_rate": 0.0, "window_days": window_days,
            "overlap_threshold": overlap, "share_above_signature_limit": 0.0,
            "min_len_missable": _missable_token_length(overlap),
        }
        return (df, stats, empty_lineage) if return_lineage else (df, stats)

    ts = df["ts_utc"].to_numpy()
    win = np.timedelta64(window_days, "D")
    keep = np.ones(n_in, dtype=bool)
    elim = np.full(n_in, -1, dtype=np.int64)
    relation = np.array([None] * n_in, dtype=object)

    # --- Exact pass, re-anchored on the last KEPT occurrence.
    for positions in df.groupby("text_norm", sort=False).indices.values():
        if len(positions) == 1:
            continue
        # The frame is sorted by (ts_utc, source_row_id), so these are already
        # in time order; sorting again costs nothing and states the assumption.
        anchor = -1
        for i in np.sort(positions):
            if anchor < 0 or (ts[i] - ts[anchor]) > win:
                anchor = i
            else:
                keep[i] = False
                elim[i] = anchor
                relation[i] = "exact"
    n_exact_dropped = int((~keep).sum())

    # --- Near pass, over the exact survivors only.
    n_near_dropped, long_share = 0, 0.0
    surv_pos = np.flatnonzero(keep)
    if near_dupe and len(surv_pos):
        sub = df.iloc[surv_pos]
        near_keep, near_elim = near_duplicate_keep(
            sub["text_norm"], sub["ts_utc"], window_days, overlap,
            return_eliminator=True,
        )
        dropped = ~near_keep
        n_near_dropped = int(dropped.sum())
        keep[surv_pos[dropped]] = False
        elim[surv_pos[dropped]] = surv_pos[near_elim[dropped]]
        relation[surv_pos[dropped]] = "near"

        min_len = _missable_token_length(overlap)
        n_tokens = sub["text_norm"].str.split().map(lambda x: len(set(x)))
        long_share = float((n_tokens >= min_len).mean())

    # --- Clusters. `elim` records the row each duplicate was eliminated
    # AGAINST, which is not always a survivor: an exact repeat's anchor can
    # itself be removed later, as a near-duplicate of some earlier headline. So
    # elimination forms short chains, and the cluster representative is the root
    # of the chain rather than the immediate eliminator.
    #
    # `eliminated_by` in the lineage keeps the immediate step, because that is
    # what actually happened; `cluster_id` resolves to the survivor.
    dropped_pos = np.flatnonzero(~keep)
    if len(dropped_pos) and (elim[dropped_pos] < 0).any():
        raise AssertionError(
            "dedup lineage is broken: an eliminated row names no eliminator"
        )
    cluster_pos = np.where(keep, np.arange(n_in), elim)
    for _ in range(64):
        if not len(dropped_pos) or keep[cluster_pos[dropped_pos]].all():
            break
        cluster_pos[dropped_pos] = cluster_pos[cluster_pos[dropped_pos]]
    else:
        raise AssertionError(
            "dedup lineage did not resolve to survivors within 64 hops"
        )

    hid = df["headline_id"].to_numpy()
    srid = df["source_row_id"].to_numpy()
    cluster_id = hid[cluster_pos]

    # --- Union the ticker tags over each cluster.
    tag_union: dict[int, set] = {}
    for pos, tags in zip(cluster_pos, df["tickers"]):
        bucket = tag_union.get(pos)
        if bucket is None:
            tag_union[pos] = set(tags)
        else:
            bucket.update(tags)
    sizes = np.bincount(cluster_pos, minlength=n_in)

    clean = df.iloc[surv_all := np.flatnonzero(keep)].copy()
    clean["tickers"] = [sorted(tag_union[p]) for p in surv_all]
    clean["cluster_id"] = cluster_id[surv_all]
    clean["n_cluster_rows"] = sizes[surv_all].astype("int64")
    clean = clean.reset_index(drop=True)
    clean = clean.loc[:, [c for c in CLEAN_COLUMNS if c in clean.columns]]

    stats = {
        "n_in": n_in,
        "n_out": len(clean),
        "n_exact_dropped": n_exact_dropped,
        "n_near_dropped": n_near_dropped,
        "dedup_rate": 0.0 if n_in == 0 else 1 - len(clean) / n_in,
        "window_days": window_days,
        "overlap_threshold": overlap,
        # Share of rows where the blocking could miss a genuine near-duplicate.
        "share_above_signature_limit": long_share,
        "min_len_missable": _missable_token_length(overlap),
    }
    if not return_lineage:
        return clean, stats

    lineage = pd.DataFrame(
        {
            "source_row_id": srid,
            "headline_id": hid,
            "cluster_id": cluster_id,
            "kept": keep,
            "eliminated_by": np.where(elim >= 0, srid[elim], -1).astype("int64"),
            "relation": relation,
        }
    ).loc[:, LINEAGE_COLUMNS]
    lineage.loc[lineage["kept"], "eliminated_by"] = pd.NA
    lineage["eliminated_by"] = lineage["eliminated_by"].astype("Int64")
    return clean, stats, lineage


def trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """NYSE session dates in [start, end], tz-naive dates.

    Raises if `pandas_market_calendars` is unavailable. It previously fell back
    to `pd.bdate_range` with a warning, which treats every market holiday as a
    session -- roughly nine phantom sessions a year, silently shifting what
    "the next trading day" means, behind a warning that scrolls past (B05
    defect D-3). A wrong calendar is not a degraded mode of this pipeline; it
    is a wrong answer, so it is a hard failure.
    """
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "pandas_market_calendars is required: every lag, lead and "
            "session mapping in this project is defined against the exchange "
            "calendar, and substituting business days would silently treat ~9 "
            "market holidays a year as trading sessions.\n"
            "    pip install pandas-market-calendars"
        ) from exc

    sched = mcal.get_calendar(config.MARKET_CALENDAR).schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(sched.index).normalize()


def returns_on_calendar(close: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Log returns defined only between **adjacent exchange sessions** (R04a).

    The defect this exists to prevent (audit A04): computing `diff(log(close))`
    on whatever rows the price source returned. If a session is missing from
    that response, the difference silently spans two sessions and still occupies
    the row labelled with the later date. Reindexing afterwards cannot repair a
    number that was already computed across the gap, and `build_panel`'s
    left-join therefore inherited it.

    Prices are placed on the calendar first, so a missing session becomes NaN
    and both `ret` at that session and `ret` at the following one are NaN --
    which is the correct answer, because neither is a one-session return.
    """
    cal = pd.DatetimeIndex(calendar).normalize()
    if cal.tz is not None:
        cal = cal.tz_localize(None)
    on_cal = pd.Series(close).copy()
    on_cal.index = pd.DatetimeIndex(on_cal.index).normalize()
    on_cal = on_cal.reindex(cal)

    logp = np.log(on_cal.astype(float))
    ret = logp.diff()
    # diff() propagates NaN forward one row, which is what we want, but make the
    # requirement explicit: a return needs BOTH adjacent closes present.
    ret[on_cal.isna() | on_cal.shift(1).isna()] = np.nan
    return ret


def load_market(
    start: str,
    end: str,
    calendar: pd.DatetimeIndex | None = None,
    warmup_sessions: int = 0,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Daily bars for `ticker` (default SPY) + ^VIX close, on the exchange calendar.

    `ticker` exists only for the protocol's single-name robustness spot check.
    It defaults to `config.MARKET_TICKER`, so every existing caller is unchanged,
    and the column names stay the same because the downstream panel contract is
    defined on names rather than on which instrument produced them.

    Three corrections over the naive version (R04a / audit A04):

    * **Prices are placed on the calendar before returns are computed**, so no
      return can span a missing session. See `returns_on_calendar`.
    * **yfinance's `end` is exclusive**, so the configured inclusive end date
      would silently omit the final session. One day is added to the request.
    * **`warmup_sessions`** fetches earlier sessions so that the first analysis
      session has a defined lagged return, and the 63-session volume detrend is
      defined from the window's start. Warm-up rows are returned and flagged
      `in_window = False`; they must not widen the analysis window.

    The returned frame spans the calendar, one row per session, with NaN where
    the price source had no data. Missing sessions are reported in
    `.attrs["market_stats"]` rather than silently dropped.
    """
    import yfinance as yf

    if calendar is None:
        calendar = trading_calendar(start, end)
    cal = pd.DatetimeIndex(calendar).normalize()

    fetch_from = start
    if warmup_sessions:
        warm = trading_calendar(
            (pd.Timestamp(start) - pd.Timedelta(days=warmup_sessions * 2 + 20)).strftime("%Y-%m-%d"),
            start,
        )
        if len(warm) > warmup_sessions:
            fetch_from = warm[-(warmup_sessions + 1)].strftime("%Y-%m-%d")
            cal = pd.DatetimeIndex(warm[-(warmup_sessions + 1) : -1]).append(cal).unique().sort_values()

    # yfinance's `end` is exclusive; without +1 day the final session is lost.
    fetch_to = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    symbol = config.MARKET_TICKER if ticker is None else ticker
    spy = yf.download(
        symbol, start=fetch_from, end=fetch_to, auto_adjust=True, progress=False
    )
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if spy.empty:
        raise RuntimeError(
            f"no {symbol} data returned for {fetch_from}..{fetch_to}"
        )
    spy.index = pd.DatetimeIndex(spy.index).normalize()

    vix = yf.download(config.VIX_TICKER, start=fetch_from, end=fetch_to, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    if not vix.empty:
        vix.index = pd.DatetimeIndex(vix.index).normalize()

    on_cal = spy.reindex(cal)
    out = pd.DataFrame(index=cal)
    out.index.name = "date"
    out["close_adj"] = on_cal["Close"].astype(float)
    out["ret"] = returns_on_calendar(spy["Close"], cal)
    out["rv_parkinson"] = (np.log(on_cal["High"] / on_cal["Low"]) ** 2) / (4 * np.log(2))
    out["volume"] = on_cal["Volume"]
    out["log_volume"] = np.log(out["volume"].replace(0, np.nan))
    out["vix_close"] = (
        vix["Close"].reindex(cal).astype(float) if not vix.empty else np.nan
    )
    out["in_window"] = (cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))

    missing = out.index[out["close_adj"].isna()]
    out.attrs["market_stats"] = {
        "n_sessions": int(len(cal)),
        "n_warmup": int((~out["in_window"]).sum()),
        "n_missing_price": int(len(missing)),
        "missing_dates": [str(d.date()) for d in missing[:10]],
        "requested": [fetch_from, fetch_to],
    }
    return out.reset_index()
