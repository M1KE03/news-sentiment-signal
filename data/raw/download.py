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
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

RAW = config.DATA_RAW

FNSPID_REPO = "Zihan1004/FNSPID"
FNSPID_FILE = "Stock_news/All_external.csv"
FNSPID_URL = f"https://huggingface.co/datasets/{FNSPID_REPO}/resolve/main/{FNSPID_FILE}"

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

    def read(self, n: int = -1) -> bytes:
        b = self._raw.read(n)
        self.pos += len(b)
        return b

    def readable(self) -> bool:
        return True


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
    out = out or config.HEADLINES_PARQUET

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
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

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


def fetch_market() -> None:
    """SPY + ^VIX for the locked window -> interim/market.parquet."""
    from src import data as sdata

    if config.SAMPLE_START is None or config.SAMPLE_END is None:
        raise SystemExit("D4 is still unresolved -- lock the window in config.py first.")
    market = sdata.load_market(config.SAMPLE_START, config.SAMPLE_END)
    config.MARKET_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(config.MARKET_PARQUET, index=False)
    print(f"wrote {config.MARKET_PARQUET}  ({len(market):,} sessions)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fnspid-sample", action="store_true",
                    help="bounded range-slice sample of FNSPID All_external.csv")
    ap.add_argument("--slices", type=int, default=24)
    ap.add_argument("--slice-mb", type=int, default=6)
    ap.add_argument("--assemble", action="store_true",
                    help="stream the full file and write the filtered corpus")
    ap.add_argument("--market", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.fnspid_sample:
        fetch_fnspid_sample(n_slices=args.slices, slice_mb=args.slice_mb)
    if args.assemble:
        assemble_corpus()
    if args.market:
        fetch_market()
    if not config.LM_DICT_PATH.exists():
        print("\nStill needed by hand:\n  " + LM_NOTE)


if __name__ == "__main__":
    main()
