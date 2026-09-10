"""Dataset audit (B03). The unit of audit is the **source**, not the file.

Rewritten after the B03 audit of FNSPID (docs/archive/data-audit-fnspid.md). The earlier
version applied the intraday gate to a whole file, which is the wrong unit:
`All_external.csv` concatenates five-plus sub-corpora with different languages,
schemas and timestamp behaviour. Judged as one file it looks partly intraday,
when in fact one block is intraday and every other block is date-only. Pooling
them hides exactly the property the gate exists to detect.

Two rules are enforced in code rather than left as caveats in prose.

1. **Per-source profiling.** Every timestamp, language, coverage and tagging
   statistic is computed within a source. There is no whole-file verdict.
2. **Cluster samples cannot produce corpus-wide rates.** A bounded audit sample
   drawn as byte-range slices of a ticker-ordered file returns whole ticker
   blocks. Statistics that need a representative sample -- duplication rate,
   headlines per session, company concentration -- are *withheld* on such a
   sample, with the reason recorded, instead of being computed and quietly
   believed. See `CorpusRate`.

What this module does not do: decide relevance. Whether a source's content
belongs in the study is a human judgement made against a written criterion and
recorded in the audit document. `screen_sources` reports the mechanical
criteria and marks relevance as a decision to be supplied, because a screen
that silently invented that judgement would be the most dangerous thing here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config

# A source whose stamps are really dates puts a large mass at one clock time.
DATE_ONLY_SHARE = 0.50
# Even a genuine feed clusters on the minute; below this the stamps resolve
# nothing usable.
MIN_DISTINCT_MINUTES = 60

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_DOMAIN = re.compile(r"https?://(?:www\.)?([^/]+)")

# Canonical names for the sub-corpora seen in FNSPID. Order matters: the first
# substring that matches a URL host wins.
KNOWN_SOURCES = (
    "benzinga", "seekingalpha", "zacks", "reuters", "bloomberg", "lenta.ru",
    "gurufocus", "investors.com", "thestreet", "nasdaq",
)


@dataclass
class CorpusRate:
    """A statistic that is only meaningful on a representative sample.

    Carries either a value or the reason it was withheld, so a downstream
    reader cannot mistake "not computable from this sample" for "zero".
    """

    name: str
    value: float | None = None
    withheld_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.value is not None

    def __repr__(self) -> str:
        if self.available:
            return f"{self.name}={self.value:.4g}"
        return f"{self.name}=WITHHELD({self.withheld_reason})"


CLUSTER_REASON = (
    "sample is a cluster sample (byte-range slices of a ticker-ordered file), "
    "so this rate is an artifact of which offsets were drawn; compute it on the "
    "assembled corpus instead"
)


# --------------------------------------------------------------- loading ----


def source_of(url: pd.Series) -> pd.Series:
    """URL host -> canonical source name; '(no url)' when absent."""
    host = url.astype("string").str.extract(_DOMAIN)[0].fillna("(no url)")

    def canon(h: str) -> str:
        for k in KNOWN_SOURCES:
            if k in h:
                return k
        return h

    return host.map(canon)


def load_sample(path: str | Path, date_col: str = "Date") -> tuple[pd.DataFrame, dict]:
    """Read a downloaded audit sample and normalise it, reporting what was dropped.

    Rows whose `Date` does not match `YYYY-MM-DD HH:MM:SS UTC` are dropped and
    counted. They are not noise: in FNSPID they are fragments of article body
    text, which is how the embedded-newline problem in the `Article` field
    announces itself. The count is returned so it stays visible.
    """
    df = pd.read_parquet(path)
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$"
    conforms = df[date_col].astype("string").str.match(pattern).fillna(False)

    stats = {
        "n_read": int(len(df)),
        "n_dropped_malformed_date": int((~conforms).sum()),
        "malformed_examples": df.loc[~conforms, date_col].astype(str).head(3).tolist(),
    }

    out = df[conforms].copy()
    out["ts_utc"] = pd.to_datetime(
        out[date_col].astype(str).str.removesuffix(" UTC"), utc=True
    )
    out["ts_et"] = out["ts_utc"].dt.tz_convert(config.TZ_MARKET)
    if "Url" in out.columns:
        out["source"] = source_of(out["Url"])
    else:
        out["source"] = "(no url)"
    stats["n_kept"] = int(len(out))
    stats["sources"] = out["source"].value_counts().to_dict()
    return out.reset_index(drop=True), stats


# ------------------------------------------------------ per-source audit ----


def timestamp_profile(ts: pd.Series, ts_market: pd.Series | None = None) -> dict:
    """Timestamp evidence for **one source**. Never call this on a pooled file.

    Two clocks, deliberately, because conflating them hides the answer.

    *Concentration* -- the tell for a date-only feed -- is measured on `ts` in
    **the zone the source published in**. A date column parsed to datetimes
    puts every row at midnight *of that zone*. FNSPID stamps are UTC, so its
    date-only rows sit at 00:00 UTC, which is 19:00 or 20:00 in New York: read
    in market time they look like an evening publication spike rather than the
    absence of a time, and `midnight_share` would report 0%.

    *Session position* -- before open, in session, after close -- is measured on
    `ts_market`, which must be market time, since that is the only clock in
    which "after the close" means anything. Defaults to `ts` converted.
    """
    ts = pd.to_datetime(pd.Series(ts).reset_index(drop=True))
    if getattr(ts.dt, "tz", None) is None:
        raise ValueError("timestamps must be timezone-aware; localize at load time")

    if ts_market is None:
        mkt = ts.dt.tz_convert(config.TZ_MARKET)
    else:
        mkt = pd.to_datetime(pd.Series(ts_market).reset_index(drop=True))
        if getattr(mkt.dt, "tz", None) is None:
            raise ValueError("ts_market must be timezone-aware")
        mkt = mkt.dt.tz_convert(config.TZ_MARKET)

    minute = ts.dt.hour * 60 + ts.dt.minute          # source clock
    counts = minute.value_counts()
    modal, modal_share = int(counts.index[0]), float(counts.iloc[0] / len(ts))
    distinct = int(minute.nunique())
    date_only = modal_share >= DATE_ONLY_SHARE or distinct < MIN_DISTINCT_MINUTES

    m_minute = mkt.dt.hour * 60 + mkt.dt.minute      # market clock
    close = config.MARKET_CLOSE_ET.hour * 60 + config.MARKET_CLOSE_ET.minute
    open_ = 9 * 60 + 30
    return {
        "n": int(len(ts)),
        "source_tz": str(ts.dt.tz),
        "modal_minute": modal,
        "modal_time": f"{modal // 60:02d}:{modal % 60:02d}",
        "modal_share": modal_share,
        "midnight_share": float((minute == 0).mean()),
        "distinct_minutes": distinct,
        "share_before_open": float((m_minute < open_).mean()),
        "share_in_session": float(((m_minute >= open_) & (m_minute <= close)).mean()),
        "share_after_close": float((m_minute > close).mean()),
        "looks_date_only": bool(date_only),
        "verdict": (
            f"DATE-ONLY: {modal_share:.1%} of stamps at "
            f"{modal // 60:02d}:{modal % 60:02d} {ts.dt.tz}; the intraday close "
            "rule cannot be applied to this source"
            if date_only
            else f"INTRADAY: {distinct} distinct minutes-of-day, modal minute holds "
            f"only {modal_share:.1%}"
        ),
    }


def script_profile(text: pd.Series) -> dict:
    """Is this source's text actually English?

    Cheap and specific: the share of headlines containing Cyrillic characters,
    plus the non-ASCII share. This is what separates a Russian-language general
    news site from a financial newswire without loading a language model, and
    scoring non-English text with an English financial model produces numbers
    that look fine and mean nothing.
    """
    s = text.astype("string").fillna("")
    return {
        "cyrillic_share": float(s.str.contains(CYRILLIC, regex=True).mean()),
        "non_ascii_share": float(s.str.contains(r"[^\x00-\x7F]", regex=True).mean()),
        "likely_non_english": bool(s.str.contains(CYRILLIC, regex=True).mean() > 0.05),
    }


def profile_by_source(
    df: pd.DataFrame,
    text_col: str = "Article_title",
    ticker_col: str = "Stock_symbol",
    ts_col: str = "ts_utc",
    ts_market_col: str = "ts_et",
) -> pd.DataFrame:
    """One row per source: the whole B03 evidence table.

    This replaces the old whole-file profile. Every column here is computed
    within a source, because that is the only unit at which the numbers mean
    anything for a concatenated corpus.
    """
    rows = []
    for name, g in df.groupby("source", sort=False):
        tp = timestamp_profile(
            g[ts_col], g[ts_market_col] if ts_market_col in g else None
        )
        sp = script_profile(g[text_col]) if text_col in g else {}
        rows.append(
            {
                "source": name,
                "rows": len(g),
                "midnight_share": tp["midnight_share"],
                "modal_time": tp["modal_time"],
                "modal_share": tp["modal_share"],
                "distinct_minutes": tp["distinct_minutes"],
                "looks_date_only": tp["looks_date_only"],
                "ticker_share": (
                    float(g[ticker_col].notna().mean()) if ticker_col in g else np.nan
                ),
                "cyrillic_share": sp.get("cyrillic_share", np.nan),
                "likely_non_english": sp.get("likely_non_english", False),
                "first": g[ts_col].min(),
                "last": g[ts_col].max(),
                "verdict": tp["verdict"],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("rows", ascending=False)
        .set_index("source")
    )


def midnight_share_by_year(df: pd.DataFrame, ts_col: str = "ts_utc") -> pd.DataFrame:
    """Midnight share per source per year -- the timestamp *regime* over time.

    Measured on the source clock, for the reason given in `timestamp_profile`.

    A source can change instrument mid-history. FNSPID's Benzinga block sits at
    96-100% midnight through 2019 and drops to 75% in 2020, which makes 2020 a
    different measurement from the years before it and is the reason the sample
    window excludes it. A single pooled share would have hidden that entirely.
    """
    ts = pd.to_datetime(df[ts_col])
    frame = pd.DataFrame(
        {
            "year": ts.dt.year,
            "source": df["source"].to_numpy(),
            "is_midnight": ((ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)),
        }
    )
    return frame.pivot_table(
        index="year", columns="source", values="is_midnight", aggfunc="mean"
    )


def hour_histogram(df: pd.DataFrame, ts_col: str = "ts_et") -> pd.Series:
    """Headlines per hour-of-day (ET), for one source's rows."""
    return (
        pd.to_datetime(df[ts_col]).dt.hour.value_counts()
        .reindex(range(24), fill_value=0).sort_index()
    )


def content_sample(
    df: pd.DataFrame,
    source: str,
    n: int = 20,
    text_col: str = "Article_title",
    seed: int = config.SEED,
) -> list[str]:
    """Titles to read by eye. Relevance is judged by a human, not by a statistic."""
    g = df[df["source"] == source][text_col].dropna()
    if g.empty:
        return []
    return g.sample(min(n, len(g)), random_state=seed).astype(str).tolist()


# --------------------------------------------- statistics needing a census ----


def coverage_profile(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex | None = None,
    clustered: bool = True,
    ts_col: str = "ts_et",
    sessions: pd.Series | None = None,
) -> dict:
    """Span and per-year counts, plus per-session rates **only if representative**.

    `clustered=True` (the default, and the truth for a byte-range sample of a
    ticker-ordered file) withholds every per-session rate. Those numbers are a
    property of the assembled corpus, not of an arbitrary set of ticker blocks,
    and returning them anyway would be the quiet kind of wrong.

    **Pass `sessions` when the caller has already assigned them** (R03c). This
    function otherwise falls back to the intraday close rule, which is a
    *different mapping* from the deferred one the analysis path uses. Reporting
    coverage under one mapping and running the study under another produced
    counts that disagreed on 2,502 of 2,516 sessions and an extra zero-news day
    that was purely an artifact of the mismatch.

    Zero-news sessions are split into two kinds, because they mean different
    things. Under the deferred rule a session's window opens at the *previous*
    session, so the first session of the calendar can never receive a headline:
    that zero is a **boundary exclusion**, structural and expected. Any other
    zero is an actual absence of news.
    """
    ts = pd.to_datetime(df[ts_col])
    out: dict = {
        "n_headlines": int(len(df)),
        "first": ts.min(),
        "last": ts.max(),
        "per_year": ts.dt.year.value_counts().sort_index(),
        "clustered": clustered,
    }

    rate_names = ("per_session_mean", "per_session_median", "zero_news_share", "thin_days_share")
    if clustered or calendar is None:
        reason = CLUSTER_REASON if clustered else "no trading calendar supplied"
        for r in rate_names:
            out[r] = CorpusRate(r, withheld_reason=reason)
        return out

    if sessions is not None:
        day = pd.Series(sessions).reset_index(drop=True)
        out["mapping"] = "supplied by caller"
    else:
        from src import align

        # Fallback only. This is the intraday rule and is NOT the mapping the
        # analysis path uses for a date-only corpus; callers that have already
        # assigned sessions must pass them.
        day = align.map_to_trading_day(ts, calendar, check_date_only=False)
        out["mapping"] = "intraday close rule (fallback)"

    cal = pd.DatetimeIndex(calendar).normalize()
    per_day = day.value_counts().reindex(cal, fill_value=0)

    zero = per_day.index[per_day == 0]
    # Structural under the deferred rule: session 0's window opens before the
    # calendar starts, so it cannot receive headlines.
    boundary = [d for d in zero if d == cal[0]]
    true_zero = [d for d in zero if d != cal[0]]

    out["n_assigned"] = int(day.notna().sum())
    out["n_unassignable"] = int(day.isna().sum())
    out["per_session_mean"] = CorpusRate("per_session_mean", float(per_day.mean()))
    out["per_session_median"] = CorpusRate("per_session_median", float(per_day.median()))
    out["zero_news_share"] = CorpusRate("zero_news_share", float((per_day == 0).mean()))
    out["thin_days_share"] = CorpusRate(
        "thin_days_share",
        float((per_day < config.MIN_HEADLINES_FOR_DISPERSION).mean()),
    )
    out["zero_news_dates"] = list(zero)
    out["boundary_sessions"] = boundary
    out["true_zero_news_sessions"] = true_zero
    out["n_boundary_sessions"] = len(boundary)
    out["n_true_zero_news"] = len(true_zero)
    out["per_day"] = per_day
    return out


def duplication_profile(df: pd.DataFrame, clustered: bool = True, **dedup_kwargs) -> dict:
    """Duplication, withheld on a cluster sample.

    A market-wide roundup headline is emitted once per tagged ticker, so its
    repeats are largely invisible in a sample containing only some of those
    tickers: the measured rate would understate the corpus's and there is no
    way to correct it from the sample. Structural examples are still returned,
    since those are informative regardless.
    """
    out = {"clustered": clustered}
    if "text_norm" not in df.columns and "Article_title" in df.columns:
        from src.data import normalize_text

        df = df.assign(text_norm=normalize_text(df["Article_title"]))

    if "text_norm" in df.columns:
        vc = df["text_norm"].value_counts()
        out["most_repeated"] = vc.head(10)
        out["share_in_repeated_texts"] = float((vc[vc > 1].sum()) / max(len(df), 1))

    if clustered:
        out["dedup_rate"] = CorpusRate("dedup_rate", withheld_reason=CLUSTER_REASON)
        return out

    from src import data as sdata

    # An audit sample is a slice of a raw file, not a headline artifact: it may
    # carry `Article_title` and nothing else `dedup` needs. Build the minimum
    # `dedup` requires, here, rather than loosening `dedup` itself -- the strict
    # schema is what stops an unnumbered frame reaching the real corpus path.
    #
    # `source_row_id` is assigned from the sample's own order. That is sound for
    # a *rate*, which does not depend on which member of a duplicate group
    # survives, and it is not a corpus identifier: nothing here is written to an
    # artifact.
    work = df.copy()
    if "ts_utc" not in work.columns:
        for cand in ("ts_utc", "Date", "date"):
            if cand in df.columns:
                work["ts_utc"] = pd.to_datetime(df[cand], errors="coerce", utc=True)
                break
    work = work[work["ts_utc"].notna()]
    if "tickers" not in work.columns:
        work["tickers"] = [[] for _ in range(len(work))]
    if "headline_id" not in work.columns:
        work["headline_id"] = [
            sdata._headline_id(t, x)
            for t, x in zip(work["text_norm"], work["ts_utc"])
        ]
    work = sdata.assign_source_row_ids(work)
    _, stats = sdata.dedup(work, **dedup_kwargs)
    out.update(stats)
    out["dedup_rate"] = CorpusRate("dedup_rate", float(stats["dedup_rate"]))
    return out


def concentration_profile(
    df: pd.DataFrame, ticker_col: str = "tickers", top: int = 15
) -> dict:
    """How much of the corpus a few names carry. Requires a census, not a sample.

    The daily aggregate S_t is an equal-weighted mean over headlines, so a
    company covered ten times more often carries ten times the weight. That is
    a property of the measured quantity -- "average tone in this collection" --
    and it has to be reported rather than discovered by a reader. It cannot be
    estimated from a ticker-clustered sample, which is why it waited for the
    assembled corpus.
    """
    tags = df[ticker_col]
    exploded = tags.explode().dropna()
    counts = exploded.value_counts()
    total = int(counts.sum())
    share = counts / max(total, 1)
    return {
        "n_headlines": int(len(df)),
        "n_tagged": int((tags.map(len) > 0).sum()),
        "untagged_share": float((tags.map(len) == 0).mean()),
        "n_distinct_tickers": int(counts.size),
        "top": counts.head(top),
        "top_share": share.head(top),
        "share_top10": float(share.head(10).sum()),
        "share_top50": float(share.head(50).sum()),
        "share_top100": float(share.head(100).sum()),
        # Herfindahl over ticker mentions: 1/HHI is the "effective number" of
        # names the aggregate really averages over.
        "hhi": float((share**2).sum()),
        "effective_n_tickers": float(1 / max((share**2).sum(), 1e-12)),
    }


def window_stability(
    per_session: pd.Series, calendar: pd.DatetimeIndex, tol: float = 0.25
) -> pd.DataFrame:
    """Per-year coverage on the assembled corpus, for the D4 freeze decision.

    Reports, per year: sessions, headlines, mean and median per session, the
    zero-news share, and the share of sessions too thin for dispersion. A year
    whose median falls below `tol` of the sample median is flagged: coverage
    that unstable makes S_t a different measurement in that year, which is the
    coverage-drift limitation the review named.
    """
    idx = pd.DatetimeIndex(calendar).normalize()
    s = per_session.reindex(idx, fill_value=0)
    frame = pd.DataFrame({"n": s.to_numpy()}, index=idx)
    frame["year"] = frame.index.year

    g = frame.groupby("year")["n"]
    out = pd.DataFrame(
        {
            "sessions": g.size(),
            "headlines": g.sum(),
            "mean": g.mean().round(1),
            "median": g.median(),
            "zero_news": g.apply(lambda x: int((x == 0).sum())),
            "zero_news_share": g.apply(lambda x: float((x == 0).mean())).round(4),
            "thin_share": g.apply(
                lambda x: float((x < config.MIN_HEADLINES_FOR_DISPERSION).mean())
            ).round(4),
        }
    )
    overall_median = float(out["median"].median())
    out["vs_median"] = (out["median"] / max(overall_median, 1e-9)).round(2)
    out["unstable"] = out["vs_median"] < tol
    out.attrs["overall_median"] = overall_median
    out.attrs["tolerance"] = tol
    return out


def suggest_window(
    df: pd.DataFrame, ts_col: str = "ts_et", min_per_year: int = 5000, tol: float = 0.25
) -> dict:
    """Propose a window from one source's per-year counts. A proposal, not a decision.

    Stability, not length, is binding: a year with a fraction of the neighbouring
    coverage makes the daily aggregate a different measurement. On a cluster
    sample the absolute counts are not the corpus's, so the result is marked
    provisional and the caller must confirm it on the assembled corpus.
    """
    per_year = pd.to_datetime(df[ts_col]).dt.year.value_counts().sort_index()
    if per_year.empty:
        return {"ok": False, "reason": "no dated headlines", "provisional": True}

    floor = max(min_per_year, float(per_year.median()) * tol)
    ok_years = [int(y) for y in per_year[per_year >= floor].index]

    best: list[int] = []
    run: list[int] = []
    for y in ok_years:
        run = run + [y] if run and y == run[-1] + 1 else [y]
        if len(run) >= len(best):
            best = run

    if not best:
        return {"ok": False, "reason": f"no year clears {floor:,.0f} headlines",
                "provisional": True}

    return {
        "ok": len(best) * 252 >= config.MIN_TRADING_DAYS,
        "start": f"{best[0]}-01-01",
        "end": f"{best[-1]}-12-31",
        "years": len(best),
        "approx_sessions": len(best) * 252,
        "excluded_years": [int(y) for y in per_year.index if y not in best],
        "provisional": True,
        "reason": (
            f"{len(best)} consecutive years ({best[0]}-{best[-1]}) clear "
            f"{floor:,.0f} headlines/yr; ~{len(best) * 252:,} sessions vs. the "
            f"{config.MIN_TRADING_DAYS:,} target. Provisional: confirm coverage "
            "stability on the assembled corpus before freezing."
        ),
    }


def corpus_census(
    headlines: pd.DataFrame, calendar: pd.DatetimeIndex, dedup: bool = True
) -> dict:
    """Everything the bounded audit sample was unable to measure.

    The B03 sample was a cluster sample over tickers, so duplication, per-session
    coverage and company concentration were withheld rather than estimated. All
    three are properties of the assembled corpus and are computed here, on a
    census, with `clustered=False`.

    Returns the deduplicated frame alongside the statistics, since the dedup
    step is what the per-session counts must be computed on -- counting
    syndicated repeats would inflate n_t and distort S_t.
    """
    from src import align
    from src import data as sdata

    out: dict = {"n_raw": int(len(headlines))}

    if dedup:
        clean, dup_stats = sdata.dedup(headlines)
    else:
        clean, dup_stats = headlines, {}
    out["duplication"] = dup_stats
    out["clean"] = clean

    sessions = align.map_date_to_session(clean["ts_utc"], calendar)
    assigned = clean.assign(session=sessions)
    out["n_unassignable"] = int(sessions.isna().sum())
    per_session = assigned.dropna(subset=["session"]).groupby("session").size()

    # One assignment, reused. Re-mapping here was audit finding A06.
    out["coverage"] = coverage_profile(
        clean, calendar, clustered=False, ts_col="ts_utc", sessions=sessions
    )
    out["per_session"] = per_session
    out["stability"] = window_stability(per_session, calendar)
    out["concentration"] = concentration_profile(clean)
    return out


# ------------------------------------------------------------- screening ----


@dataclass
class SourceScreen:
    """Mechanical screen results, plus the relevance decision a human must supply."""

    table: pd.DataFrame
    passes_intraday: list[str] = field(default_factory=list)
    passes_language: list[str] = field(default_factory=list)
    passes_tagging: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SourceScreen(sources={len(self.table)}, "
            f"intraday={self.passes_intraday}, english={len(self.passes_language)}, "
            f"tagged={self.passes_tagging})"
        )


def screen_sources(
    profile: pd.DataFrame, min_rows: int = 1000, min_ticker_share: float = 0.5
) -> SourceScreen:
    """Apply the mechanical criteria per source. Relevance is deliberately absent.

    Reports which sources have usable intraday timestamps, which are plausibly
    English, and which carry ticker tags. It does **not** pick a universe:
    whether a source's *content* belongs in the study is a judgement against a
    written criterion, made by a person and recorded in the audit document. A
    screen that guessed it would be the most dangerous function in this file --
    FNSPID's one intraday source passes every mechanical test and is still
    unusable, because it is a global general newswire.
    """
    t = profile[profile["rows"] >= min_rows]
    screen = SourceScreen(table=t)
    screen.passes_intraday = t.index[~t["looks_date_only"]].tolist()
    screen.passes_language = t.index[~t["likely_non_english"].astype(bool)].tolist()
    screen.passes_tagging = t.index[t["ticker_share"].fillna(0) >= min_ticker_share].tolist()

    non_english = t.index[t["likely_non_english"].astype(bool)].tolist()
    if non_english:
        screen.notes.append(
            f"NON-ENGLISH sources present and must be filtered out explicitly: "
            f"{non_english}. Scoring them with an English financial model yields "
            "numbers with no meaning."
        )
    if not screen.passes_intraday:
        screen.notes.append(
            "No source has usable intraday timestamps. The date-only fallback "
            "applies: defer each headline to a later session and drop the "
            "same-day question."
        )
    else:
        both = set(screen.passes_intraday) & set(screen.passes_tagging)
        if not both:
            screen.notes.append(
                f"Intraday sources {screen.passes_intraday} carry no ticker tags, "
                f"while tagged sources {screen.passes_tagging} are date-only. "
                "Timestamp quality and relevance point at different sources -- "
                "this is a design decision, not a screening result."
            )
    screen.notes.append(
        "RELEVANCE IS NOT SCREENED HERE. Read content_sample() output for each "
        "candidate and record the judgement in the audit document."
    )
    return screen
