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

HEADLINE_COLUMNS = ["headline_id", "text", "text_norm", "ts_utc", "ts_et", "tickers", "source"]

# Column names, source timezone and timestamp format per dataset. Verified by
# reading the file's bytes in the B03 audit, not taken from documentation --
# FNSPID's dataset card documents no column semantics at all.
#
# The standalone "benzinga" spec is gone: there is no separate Benzinga dataset
# in this study. Benzinga is a *sub-corpus* of FNSPID's All_external.csv,
# selected by URL host, and the file also carries Reuters, Bloomberg, Zacks,
# SeekingAlpha and the Russian-language lenta.ru. See docs/data-audit-fnspid.md.
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
        keep = host.fillna("").apply(lambda h: any(d in h for d in domains))
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
    return df.loc[:, HEADLINE_COLUMNS]


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
        else pd.DataFrame(columns=HEADLINE_COLUMNS)
    )
    # Pin the string dtypes after concatenation. Without this the result depends
    # on how many chunks were read -- a single frame keeps pandas' inferred
    # dtype while a concat of several falls back to object -- so the schema
    # would vary with a performance knob, and a later merge on headline_id
    # could mismatch. Chunking must change nothing observable.
    for col in ("headline_id", "text", "text_norm", "source"):
        out[col] = out[col].astype("string")
    out = out.sort_values("ts_utc").reset_index(drop=True)
    stats["n_kept"] = len(out)
    out.attrs["load_stats"] = stats
    return out


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
    text_norm: pd.Series, ts: pd.Series, window_days: int = 3, overlap: float = 0.90
) -> np.ndarray:
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
    """
    from collections import deque

    order = np.argsort(ts.to_numpy(), kind="stable")
    times = ts.to_numpy()[order]
    texts = text_norm.to_numpy()[order]
    window = np.timedelta64(window_days, "D")

    keep = np.ones(len(order), dtype=bool)
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
        duplicate = False
        for s in sigs:
            for cand in index.get(s, ()):
                if cand in seen:
                    continue
                seen.add(cand)
                other = sets[cand]
                if len(this & other) / max(len(this), len(other)) >= overlap:
                    duplicate = True
                    break
            if duplicate:
                break

        if duplicate:
            keep[pos] = False
        else:
            for s in sigs:
                index.setdefault(s, []).append(pos)
            sets[pos] = this
            live.append((pos, sigs))

    out = np.ones(len(order), dtype=bool)
    out[order] = keep
    return out


def dedup(
    df: pd.DataFrame,
    near_dupe: bool = True,
    window_days: int = 3,
    overlap: float = 0.90,
) -> tuple[pd.DataFrame, dict]:
    """Drop exact `text_norm` repeats within `window_days`, then near-duplicates.

    Returns the surviving frame plus the counts, so the report can quote a dedup
    rate instead of asserting one.
    """
    n_in = len(df)
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # Exact repeats within the window: keep the first occurrence.
    first_seen = df.groupby("text_norm")["ts_utc"].transform("min")
    within = (df["ts_utc"] - first_seen) <= pd.Timedelta(days=window_days)
    is_first = ~df.duplicated("text_norm", keep="first")
    keep_exact = is_first | ~within
    df_exact = df[keep_exact].reset_index(drop=True)
    n_exact_dropped = n_in - len(df_exact)

    n_near_dropped, long_share = 0, 0.0
    if near_dupe and len(df_exact):
        keep = near_duplicate_keep(
            df_exact["text_norm"], df_exact["ts_utc"], window_days, overlap
        )
        n_near_dropped = int((~keep).sum())
        # Exposure of the one-token-difference limit. A pair differing by d
        # tokens clears the threshold when (L - d) / L >= overlap, i.e. when
        # L >= d / (1 - overlap). Signatures catch d = 1, so the blind spot is
        # d >= 2, which needs L >= 2 / (1 - overlap) -- 20 unique tokens at 0.9.
        min_len_missable = int(np.ceil(2 / (1 - overlap))) if overlap < 1 else 10**9
        n_tokens = df_exact["text_norm"].str.split().map(lambda x: len(set(x)))
        long_share = float((n_tokens >= min_len_missable).mean())
        df_exact = df_exact[keep].reset_index(drop=True)

    stats = {
        "n_in": n_in,
        "n_out": len(df_exact),
        "n_exact_dropped": int(n_exact_dropped),
        "n_near_dropped": n_near_dropped,
        "dedup_rate": 0.0 if n_in == 0 else 1 - len(df_exact) / n_in,
        "window_days": window_days,
        "overlap_threshold": overlap,
        # Share of rows where the blocking could miss a genuine near-duplicate.
        "share_above_signature_limit": long_share,
    }
    return df_exact, stats


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


def load_market(start: str, end: str) -> pd.DataFrame:
    """SPY daily bars + ^VIX close -> the §4 `market` schema."""
    import yfinance as yf

    spy = yf.download(
        config.MARKET_TICKER, start=start, end=end, auto_adjust=True, progress=False
    )
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if spy.empty:
        raise RuntimeError(f"no {config.MARKET_TICKER} data returned for {start}..{end}")

    vix = yf.download(config.VIX_TICKER, start=start, end=end, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    out = pd.DataFrame(index=pd.DatetimeIndex(spy.index).normalize())
    out.index.name = "date"
    out["close_adj"] = spy["Close"].to_numpy()
    out["ret"] = np.log(out["close_adj"]).diff()
    out["parkinson"] = (np.log(spy["High"] / spy["Low"]).to_numpy() ** 2) / (4 * np.log(2))
    out["volume"] = spy["Volume"].to_numpy()
    out["log_turnover"] = np.log(out["volume"].replace(0, np.nan))
    out["vix_close"] = vix["Close"].reindex(out.index).to_numpy() if not vix.empty else np.nan

    return out.reset_index()
