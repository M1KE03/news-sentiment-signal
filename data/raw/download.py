"""Fetch the raw inputs. Run from the repo root: `python data/raw/download.py --all`.

The two D1 candidates are downloaded side by side so that Stage 0 can audit both
and choose. Do not trust any description of either dataset -- including the
implementation plan's -- verify by downloading and looking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

RAW = config.DATA_RAW

LM_URL = (
    "https://drive.google.com/drive/folders/"
    "  -- the Loughran-McDonald master dictionary is distributed from the SRAF site "
    "(sraf.nd.edu). It is not a stable direct link, so download it by hand and save "
    f"it as {config.LM_DICT_PATH.name} in data/raw/."
)


def fetch_fnspid() -> None:
    """FNSPID: Nasdaq-ecosystem news with intraday timestamps and ticker tags."""
    from datasets import load_dataset

    ds = load_dataset("Zihan1004/FNSPID", split="train")
    out = RAW / "fnspid.parquet"
    ds.to_parquet(out)
    print(f"wrote {out}  ({len(ds):,} rows)")


def fetch_benzinga() -> None:
    """Kaggle 'Daily Financial News for 6000+ Stocks' (Benzinga headlines).

    Needs a Kaggle API token at ~/.kaggle/kaggle.json.
    """
    import subprocess

    slug = "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(RAW), "--unzip"],
        check=True,
    )
    print(f"unzipped {slug} into {RAW}")


def fetch_market() -> None:
    """SPY + ^VIX for the locked window -> interim/market.parquet."""
    from src import data as sdata

    if config.SAMPLE_START is None or config.SAMPLE_END is None:
        raise SystemExit("D4 is still PENDING -- lock the window in config.py first.")
    market = sdata.load_market(config.SAMPLE_START, config.SAMPLE_END)
    config.MARKET_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(config.MARKET_PARQUET, index=False)
    print(f"wrote {config.MARKET_PARQUET}  ({len(market):,} sessions)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fnspid", action="store_true")
    ap.add_argument("--benzinga", action="store_true")
    ap.add_argument("--market", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.all or args.fnspid:
        fetch_fnspid()
    if args.all or args.benzinga:
        fetch_benzinga()
    if args.all or args.market:
        fetch_market()
    if not config.LM_DICT_PATH.exists():
        print("\nStill needed by hand:\n  " + LM_URL)


if __name__ == "__main__":
    main()
