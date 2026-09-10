"""The one-off scoring pass. Deliberately NOT part of `run_all.py`.

FinBERT inference over the full headline set is the only expensive step in the
project. It is separated so that `run_all.py` -- the reproduction command a
reader will actually run -- takes minutes and needs no GPU.

    python rescore.py --time-only        # time a sample and extrapolate
    python rescore.py --dry-run          # what would be scored, and where
    python rescore.py --sessions 50      # a bounded, complete-session chunk
    python rescore.py                    # everything, resuming from cache
    python rescore.py --scorers lm       # re-run one scorer

The cache is keyed on `headline_id`, so interrupting this is safe: rerun and it
picks up where it stopped.

**Bounding is by session, never by headline (R10).** `--sessions N` takes the
first `N` calendar days that the corpus covers and scores *all* of their
headlines. Slicing the corpus at an arbitrary headline count would leave a
partial day in the cache, and a partial day is not a smaller sample -- it is a
different measurement, because `S_t` is a within-day mean and `d_t` a within-day
standard deviation. The plan's rule when the full pass is too expensive is to
shorten the *window* (D4) and record it, never to subsample within days.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

import config
from src import preflight, scoring


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


def session_date(headlines: pd.DataFrame) -> pd.Series:
    """The published calendar date in the source's own zone.

    Not the trading session -- `align.map_date_to_session` owns that mapping and
    needs the exchange calendar. This is only the unit that chunking must not
    split: under the date-only rule every headline sharing a date shares a
    session, so whole dates are the smallest safe chunk.
    """
    return pd.to_datetime(headlines["ts_et"]).dt.tz_localize(None).dt.normalize()


def bounded_chunk(headlines: pd.DataFrame, sessions: int) -> pd.DataFrame:
    """The first `sessions` calendar days, whole. Never a partial day (R10)."""
    if sessions < 1:
        raise SystemExit("--sessions must be at least 1")
    days = session_date(headlines)
    keep = days.isin(sorted(days.unique())[:sessions])
    return headlines.loc[keep]


def describe_plan(headlines, chunk, scorers) -> None:
    """What a pass would do, before it does it. `--dry-run` stops here."""
    names = list(config.SCORERS) if scorers is None else [s.name for s in scorers]
    print(f"scorers            : {', '.join(names)}")
    print(f"headlines in scope : {len(chunk):,} of {len(headlines):,}")
    if len(chunk):
        days = session_date(chunk)
        print(f"calendar days      : {days.nunique():,} "
              f"({days.min().date()} .. {days.max().date()})")
    print(f"checkpoint file    : {config.SCORES_PARQUET}")
    print(f"checkpoint exists  : {config.SCORES_PARQUET.exists()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time-only", action="store_true",
                    help="time a sample of headlines, extrapolate, and stop")
    ap.add_argument("--n", type=int, default=1000, help="sample size for --time-only")
    ap.add_argument(
        "--scorers", nargs="*", default=None,
        help=f"subset of {list(config.SCORERS)}; default = whatever the cache is missing",
    )
    ap.add_argument("--sessions", type=int, default=None,
                    help="score only the first N calendar days, whole. Bounding is "
                         "by session because a partial day is a different "
                         "measurement, not a smaller one")
    ap.add_argument("--checkpoint-every", type=int, default=50_000,
                    help="rows between atomic cache commits (default 50,000)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report scope and checkpoint location, then stop")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="run without checking prerequisites (not recommended)")
    args = ap.parse_args()

    if not args.skip_preflight:
        # Fail before loading the corpus or a transformer, and name what is
        # missing rather than surfacing it from inside a loader (A15).
        try:
            preflight.require_packages(["torch", "transformers"])
            preflight.require_artifacts("scoring")
        except preflight.PreflightError as missing:
            raise SystemExit(str(missing)) from None

    headlines = _load_headlines()
    print(f"headlines: {len(headlines):,}  ({headlines['ts_et'].min()} .. {headlines['ts_et'].max()})")

    if args.time_only:
        time_and_extrapolate(headlines, args.n)
        return

    scorers = scoring.build_scorers(args.scorers) if args.scorers else None
    chunk = bounded_chunk(headlines, args.sessions) if args.sessions else headlines
    describe_plan(headlines, chunk, scorers)

    if args.dry_run:
        print("\ndry run: nothing was scored.")
        return
    if args.sessions:
        print(f"\nBOUNDED PASS: {args.sessions} session(s) only. The cache will be "
              "incomplete, and the score-to-panel gate will refuse it until every "
              "in-window session is scored.")

    t0 = time.perf_counter()
    scores = scoring.score_all(chunk, config.SCORES_PARQUET, scorers=scorers,
                               checkpoint_every=args.checkpoint_every)
    print(f"\nwrote {config.SCORES_PARQUET}  ({len(scores):,} rows) "
          f"in {time.perf_counter() - t0:.1f}s")
    print(scores.describe().T)


if __name__ == "__main__":
    main()
