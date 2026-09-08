"""The one-off scoring pass. Deliberately NOT part of `run_all.py`.

FinBERT inference over the full headline set is the only expensive step in the
project. It is separated so that `run_all.py` -- the reproduction command a
reader will actually run -- takes minutes and needs no GPU.

    python rescore.py --time-only      # time 1,000 headlines and extrapolate
    python rescore.py                  # score everything, resuming from cache
    python rescore.py --scorers lm     # re-run one scorer

The cache is keyed on `headline_id`, so interrupting this is safe: rerun and it
picks up where it stopped.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

import config
from src import scoring


def _load_headlines() -> pd.DataFrame:
    if not config.HEADLINES_PARQUET.exists():
        raise SystemExit(
            f"{config.HEADLINES_PARQUET} not found -- run Stage 0 "
            "(notebooks/01_data_audit.ipynb) to build it first."
        )
    return pd.read_parquet(config.HEADLINES_PARQUET)


def time_and_extrapolate(headlines: pd.DataFrame, n: int = 1000) -> None:
    """Stage 1, step 2. Time the expensive scorer before committing to the pass.

    If the projected wall time exceeds ~2h, shorten the *window* (D4) and record
    the change. Never subsample headlines within days -- that biases both S_t
    and d_t.
    """
    sample = headlines["text"].head(n).tolist()
    scorer = scoring.FinbertScorer()
    t0 = time.perf_counter()
    scorer.score(sample)
    elapsed = time.perf_counter() - t0

    rate = len(sample) / elapsed
    total_hours = len(headlines) / rate / 3600
    print(f"device               : {scorer.device}")
    print(f"timed                : {len(sample):,} headlines in {elapsed:.1f}s")
    print(f"rate                 : {rate:,.0f} headlines/s")
    print(f"headlines to score   : {len(headlines):,}")
    print(f"projected full pass  : {total_hours:.2f} h")
    if total_hours > 2:
        print(
            "\nOver the 2h budget. Shorten the sample window (D4) and record the "
            "change in the report's protocol section. Do NOT subsample within days."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time-only", action="store_true", help="time 1,000 headlines and stop")
    ap.add_argument("--n", type=int, default=1000, help="sample size for --time-only")
    ap.add_argument(
        "--scorers", nargs="*", default=None,
        help=f"subset of {list(config.SCORERS)}; default = whatever the cache is missing",
    )
    args = ap.parse_args()

    headlines = _load_headlines()
    print(f"headlines: {len(headlines):,}  ({headlines['ts_et'].min()} .. {headlines['ts_et'].max()})")

    if args.time_only:
        time_and_extrapolate(headlines, args.n)
        return

    scorers = scoring.build_scorers(args.scorers) if args.scorers else None
    t0 = time.perf_counter()
    scores = scoring.score_all(headlines, config.SCORES_PARQUET, scorers=scorers)
    print(f"\nwrote {config.SCORES_PARQUET}  ({len(scores):,} rows) "
          f"in {time.perf_counter() - t0:.1f}s")
    print(scores.describe().T)


if __name__ == "__main__":
    main()
