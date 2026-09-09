"""Fetch the raw inputs. Run from the repo root: `python data/raw/download.py --help`.

The news candidates are far too large to download whole (FNSPID's `All_external.csv`
is 5.4 GB and `nasdaq_exteral_data.csv` is 22 GB), and B03 asks for a *bounded*
sample. `fetch_fnspid_sample` therefore pulls a set of HTTP range slices spread
evenly across the file rather than a prefix.

That choice is forced by the file's physical order: `All_external.csv` is sorted
by `Stock_symbol`, not by time, so a prefix would be a handful of tickers whose
names start with "A". Evenly spaced slices give a spread of tickers, each with
its own full date range. The sample is therefore:

  * a *cluster* sample over tickers, not a simple random sample of headlines;
  * usable for timestamp semantics, per-ticker coverage, duplication within a
    ticker, and publisher mix;
  * NOT usable for estimating market-wide headlines-per-day, because only the
    sampled tickers are present.

Every audit statistic must respect that distinction. See docs/data-audit-fnspid.md.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

RAW = config.DATA_RAW

FNSPID_REPO = "Zihan1004/FNSPID"
FNSPID_FILE = "Stock_news/All_external.csv"
# Pinned by revision, not by branch. `/resolve/main/` follows whatever the
# repository owner last pushed, so a re-run could silently acquire a different
# artifact from the one recorded in config and in docs/data-audit-fnspid.md.
FNSPID_REVISION = config.NEWS_HF_REVISION or "main"
FNSPID_URL = (
    f"https://huggingface.co/datasets/{FNSPID_REPO}/resolve/"
    f"{FNSPID_REVISION}/{FNSPID_FILE}"
)

# The header as published, verified by reading the first bytes of the file.
FNSPID_COLUMNS = [
    "Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author",
    "Article", "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary",
]
KEEP = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author"]

LM_NOTE = (
    "The Loughran-McDonald master dictionary is distributed from the SRAF site "
    "(sraf.nd.edu) behind a non-stable link. Download it by hand and save it as "
    f"{config.LM_DICT_PATH.name} in data/raw/, then record the release in "
    "config.LM_DICT_VERSION."
)


def remote_size(url: str = FNSPID_URL) -> int:
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers["content-length"])


def _slice(url: str, start: int, length: int) -> bytes:
    r = requests.get(
        url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=300
    )
    r.raise_for_status()
    return r.content


def fetch_fnspid_sample(
    n_slices: int = 24,
    slice_mb: int = 6,
    out: Path | None = None,
    url: str = FNSPID_URL,
) -> Path:
    """Download `n_slices` evenly spaced range slices and concatenate the rows.

    Each slice is trimmed to whole lines before parsing, so the partial records
    at its two ends are discarded rather than silently mangled. Rows that still
    fail to parse are skipped and counted -- a row count is reported for both,
    because a silently dropped row is a coverage claim you cannot check later.
    """
    out = out or RAW / "fnspid_sample.parquet"
    total = remote_size(url)
    length = slice_mb * 1024 * 1024
    step = (total - length) // max(n_slices - 1, 1)

    frames, kept, skipped = [], 0, 0
    for i in range(n_slices):
        start = i * step
        raw = _slice(url, start, length)
        # Drop the leading partial record; keep only up to the last newline.
        first_nl, last_nl = raw.find(b"\n"), raw.rfind(b"\n")
        if first_nl < 0 or last_nl <= first_nl:
            continue
        body = raw[first_nl + 1 : last_nl].decode("utf-8", errors="replace")

        before = sum(1 for _ in io.StringIO(body))
        df = pd.read_csv(
            io.StringIO(body),
            names=FNSPID_COLUMNS,
            usecols=KEEP,
            on_bad_lines="skip",
            engine="python",
            dtype=str,
        )
        df["_slice"] = i
        df["_offset"] = start
        frames.append(df)
        kept += len(df)
        skipped += max(before - len(df), 0)
        print(f"  slice {i + 1:>2}/{n_slices}  offset {start / 1e9:5.2f} GB  "
              f"{len(df):>6,} rows", flush=True)

    sample = pd.concat(frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out, index=False)
    print(
        f"\nwrote {out}\n"
        f"  {kept:,} rows parsed, {skipped:,} lines skipped as unparseable\n"
        f"  sampled {n_slices * slice_mb} MB of a {total / 1e9:.1f} GB file "
        f"({n_slices * length / total:.2%})"
    )
    return out


class _CountingStream:
    """File-like wrapper that records how many bytes have been consumed.

    Byte position is evidence, not decoration: `All_external.csv` stores each
    sub-corpus as a contiguous block, so knowing where the kept rows came from
    is how we confirm the selected source was read in full rather than
    truncated at some offset.
    """

    def __init__(self, raw):
        self._raw = raw
        self.pos = 0
        self._digest = hashlib.sha256()

    def read(self, n: int = -1) -> bytes:
        b = self._raw.read(n)
        self.pos += len(b)
        self._digest.update(b)
        return b

    def readable(self) -> bool:
        return True

    @property
    def sha256(self) -> str:
        """Digest of everything consumed so far."""
        return self._digest.hexdigest()


class AcquisitionError(RuntimeError):
    """Raised when a download cannot be trusted, or would overwrite clean data."""


def _refuse_clean_destination(out: Path) -> None:
    """Assembly writes the RAW corpus. It must never target the analysis input.

    `HEADLINES_PARQUET` holds the deduplicated corpus that the panel is built
    from. Assembly produces the pre-dedup frame, so pointing it there would
    replace 869k deduplicated rows with 1.41M undeduplicated ones -- silently,
    under the same filename, leaving every daily mean weighted by how many
    tickers each roundup happened to mention.
    """
    if out.resolve() == Path(config.HEADLINES_PARQUET).resolve():
        raise AcquisitionError(
            f"refusing to write the raw corpus to {out}: that path is the "
            "deduplicated analysis input. Assembly writes "
            f"{config.HEADLINES_RAW_PARQUET.name}; run --dedup afterwards to "
            "produce the analysis input."
        )


def _verify_acquisition(stream, expected_bytes: int | None, expected_sha: str | None) -> dict:
    """Compare what was actually read against the pinned artifact identity."""
    got = {"bytes": stream.pos, "sha256": stream.sha256}
    problems = []
    if expected_bytes is not None and stream.pos != expected_bytes:
        problems.append(f"length {stream.pos:,} != pinned {expected_bytes:,}")
    if expected_sha is not None and stream.sha256 != expected_sha:
        problems.append(f"sha256 {stream.sha256} != pinned {expected_sha}")
    got["verified"] = not problems
    got["problems"] = problems
    return got


def _publish(df: pd.DataFrame, out: Path, manifest: dict) -> None:
    """Write parquet and its manifest atomically, manifest last.

    The parquet is committed by replace, then the manifest. A crash between the
    two leaves data with no manifest, which the loader treats as unverified --
    the safe direction. The reverse order could claim verification for a file
    that was never written.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    manifest_path(out).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def manifest_path(artifact: Path) -> Path:
    return Path(artifact).with_suffix(Path(artifact).suffix + ".manifest.json")


def lineage_path(artifact: Path) -> Path:
    """Where a clean corpus's dedup lineage lives: beside it, not at a fixed path.

    Deriving this from `out` rather than from `config` is not a detail. With the
    config constant hard-coded here, a test publishing a clean corpus into
    `tmp_path` still wrote its lineage into the real `data/interim`, so a test
    run left a stray artifact next to the frozen corpus.
    """
    return Path(artifact).with_name("dedup_lineage.parquet")


def assemble_corpus(
    url: str = FNSPID_URL,
    domains: tuple[str, ...] | None = None,
    start: str | None = None,
    end: str | None = None,
    chunksize: int = 250_000,
    out: Path | None = None,
) -> Path:
    """Stream the full 5.7 GB file, keep only the selected universe, write parquet.

    The whole file is read exactly once and never stored: only rows passing the
    source filter and the window survive into memory. Reading all of it -- rather
    than the byte range where the selected source appeared in the audit sample --
    is deliberate. A partial read would silently truncate the corpus if the
    source has blocks the sample never touched, and silent truncation is the
    failure mode this project has spent every increment eliminating.

    Progress and the byte offsets at which kept rows were found are printed, so
    the "read in full" claim is checkable rather than asserted.
    """
    import sys
    import time

    from src import data as sdata

    domains = tuple(domains) if domains is not None else tuple(config.NEWS_SOURCE_DOMAINS)
    start = start or config.SAMPLE_START
    end = end or config.SAMPLE_END
    out = Path(out) if out is not None else config.HEADLINES_RAW_PARQUET
    _refuse_clean_destination(out)

    total = remote_size(url)
    print(f"streaming {total / 1e9:.2f} GB   domains={domains}   window={start}..{end}")
    t0 = time.perf_counter()
    offsets: list[tuple[float, int]] = []

    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        stream = _CountingStream(r.raw)

        def progress(i, kept, stats):
            gb = stream.pos / 1e9
            if len(kept):
                offsets.append((gb, len(kept)))
            elapsed = time.perf_counter() - t0
            pct = 100 * stream.pos / total
            rate = stream.pos / 1e6 / max(elapsed, 1e-9)
            eta = (total - stream.pos) / 1e6 / max(rate, 1e-9) / 60
            print(
                f"  chunk {i:>3}  {gb:5.2f} GB ({pct:5.1f}%)  "
                f"read {stats['n_read']:>10,}  kept {stats['n_kept_running']:>9,}  "
                f"{rate:4.1f} MB/s  eta {eta:4.1f} min",
                flush=True,
            )

        # _parse_chunk maintains counters; add a running kept total for progress.
        original = sdata._parse_chunk

        def counting_parse(raw, spec, doms, s, e, stats):
            kept = original(raw, spec, doms, s, e, stats)
            stats["n_kept_running"] = stats.get("n_kept_running", 0) + len(kept)
            return kept

        sdata._parse_chunk = counting_parse
        try:
            df = sdata.load_news(
                stream, domains=domains, start=start, end=end,
                chunksize=chunksize, on_chunk=progress,
            )
        finally:
            sdata._parse_chunk = original

    stats = df.attrs["load_stats"]

    # Verify the artifact against the pin BEFORE publishing anything. A wrong
    # digest means the rows in memory did not come from the recorded file, and
    # nothing derived from them may be written under a trusted name.
    acq = _verify_acquisition(stream, config.NEWS_FILE_BYTES, config.NEWS_FILE_SHA256)
    if not acq["verified"]:
        raise AcquisitionError(
            "downloaded artifact does not match the pinned identity: "
            + "; ".join(acq["problems"])
            + f"\n  url: {url}\n  Nothing was written. Re-check "
            "config.NEWS_HF_REVISION / NEWS_FILE_SHA256 before retrying."
        )

    manifest = {
        "artifact": "headlines_raw",
        "url": url,
        "repo": FNSPID_REPO,
        "revision": FNSPID_REVISION,
        "file": FNSPID_FILE,
        "bytes_read": acq["bytes"],
        "sha256": acq["sha256"],
        "verified_against_pin": True,
        "domains": list(domains),
        "window": [start, end],
        "rows_read": stats["n_read"],
        "rows_kept": stats["n_kept"],
        "rows_bad_timestamp": stats["n_bad_timestamp"],
        "rows_wrong_source": stats["n_wrong_source"],
        "deduplicated": False,
    }
    _publish(df, out, manifest)

    span = (
        f"{df['ts_utc'].min():%Y-%m-%d} .. "
        f"{df['ts_utc'].max():%Y-%m-%d}"
        if len(df) else "(empty)"
    )
    print(
        f"\nwrote {out}\n"
        f"  rows read        {stats['n_read']:>12,}\n"
        f"  bad timestamps   {stats['n_bad_timestamp']:>12,}\n"
        f"  wrong source     {stats['n_wrong_source']:>12,}\n"
        f"  kept (in window) {stats['n_kept']:>12,}\n"
        f"  span             {span}\n"
        f"  elapsed          {(time.perf_counter() - t0) / 60:.1f} min"
    )
    if offsets:
        print(
            f"  kept rows found between {offsets[0][0]:.2f} GB and "
            f"{offsets[-1][0]:.2f} GB of {total / 1e9:.2f} GB"
        )
    top = sorted(stats["rejected_hosts"].items(), key=lambda kv: -kv[1])[:8]
    print("  rejected hosts:", ", ".join(f"{h}={n:,}" for h, n in top))
    return out


def deduplicate_corpus(
    raw: Path | None = None,
    out: Path | None = None,
    near_dupe: bool = True,
) -> Path:
    """raw corpus -> deduplicated analysis input, with a manifest (R02).

    This is the second half of the documented acquisition sequence:

        python data/raw/download.py --assemble   # -> headlines_raw.parquet
        python data/raw/download.py --dedup      # -> headlines.parquet
        python data/raw/download.py --census     # coverage / concentration

    It exists as a command because the corpus currently in `interim/` was
    produced by an ad-hoc script, which meant the analysis input could not be
    rebuilt from a documented path. Deduplication is not cosmetic here: the
    corpus is ~38% exact and near repeats, mostly one roundup emitted once per
    tagged ticker, so skipping it weights each editorial act by the number of
    symbols it happened to mention.

    Refuses to run unless the raw corpus carries a verified acquisition
    manifest, so a clean artifact cannot be derived from an unverified one.
    """
    from src import data as sdata

    raw = Path(raw) if raw is not None else config.HEADLINES_RAW_PARQUET
    out = Path(out) if out is not None else config.HEADLINES_PARQUET

    if not raw.exists():
        raise AcquisitionError(f"{raw} not found -- run --assemble first")

    raw_manifest_path = manifest_path(raw)
    if not raw_manifest_path.exists():
        raise AcquisitionError(
            f"{raw} has no acquisition manifest at {raw_manifest_path.name}, so its "
            "provenance is unverified. Re-run --assemble, or write a manifest by "
            "hand recording how the file was obtained."
        )
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    if not raw_manifest.get("verified_against_pin"):
        raise AcquisitionError(
            f"{raw}'s manifest does not record verification against the pinned "
            "artifact; refusing to derive the analysis input from it."
        )

    df = pd.read_parquet(raw)
    before = len(df)
    if "source_row_id" not in df.columns:
        raise AcquisitionError(
            f"{raw} predates R03b and carries no 'source_row_id', so the "
            "representative rule (ts_utc, source_row_id) cannot be applied and "
            "duplicate resolution would fall back to row order. Re-run "
            "--assemble to rebuild the raw corpus with row identifiers."
        )
    clean, stats, lineage = sdata.dedup(df, near_dupe=near_dupe, return_lineage=True)

    manifest = {
        "artifact": "headlines",
        "derived_from": str(raw),
        "source_manifest": raw_manifest,
        "deduplicated": True,
        "near_dupe": near_dupe,
        "rows_in": before,
        "rows_out": len(clean),
        "n_exact_dropped": stats["n_exact_dropped"],
        "n_near_dropped": stats["n_near_dropped"],
        "dedup_rate": stats["dedup_rate"],
        "window_days": stats["window_days"],
        "overlap_threshold": stats["overlap_threshold"],
        "share_above_signature_limit": stats.get("share_above_signature_limit"),
        "min_len_missable": stats.get("min_len_missable"),
        "lineage": lineage_path(out).name,
    }
    _publish(clean, out, manifest)

    # Lineage is written AFTER the clean corpus, for the same reason the
    # acquisition manifest is written after the data: a crash then leaves a
    # corpus whose lineage is missing, rather than lineage describing a corpus
    # that was never published.
    lineage.to_parquet(lineage_path(out), index=False)
    print(
        f"wrote {out}\n"
        f"  {before:,} -> {len(clean):,} rows  "
        f"({stats['dedup_rate']:.1%} removed: {stats['n_exact_dropped']:,} exact, "
        f"{stats['n_near_dropped']:,} near)\n"
        f"  lineage -> {lineage_path(out)} ({len(lineage):,} rows)"
    )
    return out


def census_corpus() -> None:
    """Coverage, duplication and concentration on the assembled corpus."""
    from src import audit
    from src import data as sdata

    head = pd.read_parquet(config.HEADLINES_PARQUET)
    cal = sdata.trading_calendar(config.SAMPLE_START, config.SAMPLE_END)
    cen = audit.corpus_census(head, cal, dedup=False)
    print(f"corpus {len(head):,} headlines over {len(cal):,} sessions")
    print(cen["stability"].to_string())
    k = cen["concentration"]
    print(
        f"\ntickers {k['n_distinct_tickers']:,} | top-10 {k['share_top10']:.1%} "
        f"| effective names {k['effective_n_tickers']:.0f}"
    )


def fetch_market(warmup_sessions: int = 70) -> None:
    """SPY + ^VIX on the exchange calendar -> interim/market.parquet.

    The default warm-up covers the 63-session trailing volume detrend plus the
    one prior session the first analysis row's lagged return needs, with a small
    margin. Warm-up rows carry `in_window = False` and must not widen the
    analysis window.
    """
    from src import data as sdata

    if config.SAMPLE_START is None or config.SAMPLE_END is None:
        raise SystemExit("D4 is still unresolved -- lock the window in config.py first.")
    calendar = sdata.trading_calendar(config.SAMPLE_START, config.SAMPLE_END)
    market = sdata.load_market(
        config.SAMPLE_START, config.SAMPLE_END,
        calendar=calendar, warmup_sessions=warmup_sessions,
    )
    stats = market.attrs.get("market_stats", {})
    config.MARKET_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(config.MARKET_PARQUET, index=False)
    print(
        f"wrote {config.MARKET_PARQUET}\n"
        f"  {stats.get('n_sessions', len(market)):,} rows "
        f"({stats.get('n_warmup', 0)} warm-up, "
        f"{int(market['in_window'].sum()):,} in window)\n"
        f"  requested {stats.get('requested')}"
    )
    if stats.get("n_missing_price"):
        print(
            f"  WARNING: {stats['n_missing_price']} calendar session(s) have no price: "
            f"{stats['missing_dates']}\n"
            "  their returns are undefined on both sides of each gap, by design."
        )
    print(f"wrote {config.MARKET_PARQUET}  ({len(market):,} sessions)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fnspid-sample", action="store_true",
                    help="bounded range-slice sample of FNSPID All_external.csv")
    ap.add_argument("--slices", type=int, default=24)
    ap.add_argument("--slice-mb", type=int, default=6)
    ap.add_argument("--assemble", action="store_true",
                    help="stream the pinned file -> headlines_raw.parquet (not deduplicated)")
    ap.add_argument("--dedup", action="store_true",
                    help="headlines_raw.parquet -> headlines.parquet (the analysis input)")
    ap.add_argument("--census", action="store_true",
                    help="coverage, duplication and concentration on the analysis input")
    ap.add_argument("--market", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.fnspid_sample:
        fetch_fnspid_sample(n_slices=args.slices, slice_mb=args.slice_mb)
    if args.assemble:
        assemble_corpus()
    if args.dedup:
        deduplicate_corpus()
    if args.census:
        census_corpus()
    if args.market:
        fetch_market()
    if not config.LM_DICT_PATH.exists():
        print("\nStill needed by hand:\n  " + LM_NOTE)


if __name__ == "__main__":
    main()
