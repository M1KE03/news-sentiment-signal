# Future work

Where scope-creep ideas go to die quietly. Nothing here is part of the project;
moving an item out of this file is a change to the §3 contract and needs a dated
line in the log at the bottom.

## Deliberately out of scope

- **No model training or fine-tuning.** FinBERT is used as published.
- **No long documents.** 10-Ks and earnings-call transcripts are a different
  measurement problem (length, boilerplate, section structure).
- **No cross-sectional panel.** One market-level series, tested against SPY.
- **No trading backtest.** A backtest turns an inference question into a
  specification search over costs, sizing and rebalancing rules, every one of
  them p-hackable. Economic significance is delivered instead by the
  bps-per-1σ vs. transaction-cost comparison (§6.5).

## Parked ideas

- Streamlit view of S_t against price (stretch item, only if under budget).
- Scoring a small sample of EDGAR 8-K headlines as an out-of-domain check.
- Intraday confirmation: does the same-day association survive at the
  headline-to-next-15-minutes horizon? Needs intraday SPY data.
- Half-day sessions: D6 fixes the close at 16:00 ET even on early-close days.
  Using each session's actual close is strictly more accurate.
  (`config.USE_ACTUAL_SESSION_CLOSE` is the switch, currently False.)
- Volume-weighted or attention-weighted daily aggregation instead of D8's
  equal weighting.

## Change log

Any deviation from a locked decision in §3 gets a dated line here, with the
reason. An empty log is the goal.

| date | decision | change | reason |
|---|---|---|---|
| — | — | — | — |
