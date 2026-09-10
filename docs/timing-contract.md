# Timing and missing-data contract

Date: 2026-09-09. Increment: **B05**.

Status: **specification and interface review**. No code changed in this increment. It defines what B06–B09 must implement and records the outcome of the calendar-handoff architecture gate.

Related: [inference protocol](inference-protocol.md) §3 · [data audit](data-audit-fnspid.md) · [implementation plan](implementation-plan.md) §5 · [decision log](research-review-decision-log.md) (P11, P12, P14).

The corpus selected at B04 is **date-only** (FNSPID's Benzinga block, 96–100% of stamps at `00:00 UTC`). That single fact reshapes this contract: the mapping rule becomes the centre of it, and the actual-session-close work loses its urgency. Both consequences are spelled out below rather than left implicit.

---

## 1. Three defects, demonstrated

Each was reproduced against the current code before being specified away. They are not hypothetical.

### D-1 Pre-window headlines pile onto the first session

`align.map_to_trading_day` finds the first session close at or after a timestamp with `searchsorted(..., side="left")`. Above the calendar it guards correctly (`idx < len(closes)` → `NaT`). **Below it there is no guard**: any timestamp earlier than the first session returns index 0.

```text
2010-03-01 10:00  ->  2015-01-05     <- 5 years of pre-window news
2013-07-04 10:00  ->  2015-01-05
2014-12-31 10:00  ->  2015-01-05
2015-01-05 10:00  ->  2015-01-05     <- the only correct one
```

The first session of the sample silently absorbs the entire history preceding it, inflating `n_t`, distorting `S_t`, and doing so at exactly the point a reader is least likely to look.

### D-2 A missing price row turns a one-day lead into a two-day lead

`align.build_panel` inner-joins market onto daily scores and then shifts. If a session is absent from the market frame, the join drops it and `.shift(-1)` reaches across the hole:

```text
full calendar : ret_lead1 at 2015-01-07 = 0.0030   (= the 2015-01-08 return)
one row gone  : ret_lead1 at 2015-01-07 = 0.0040   (= the 2015-01-09 return)
```

No error, no warning, and the column still reads `ret_lead1`. The estimand quietly changes from a one-session return to a two-session one for an unknown subset of rows.

### D-3 The calendar helper degrades to business days

`data.trading_calendar` catches `ImportError` on `pandas_market_calendars`, warns, and returns `pd.bdate_range`. That treats every market holiday as a trading session. **The fallback is live in the current environment** — the package is not installed — so a run today would produce a calendar with roughly nine phantom sessions a year, and the warning would scroll past.

## 2. The mapping rule for a date-only corpus

A headline carries a date `d` and no usable time.

> **Primary rule.** A headline dated `d` is assigned to the **first exchange session that begins strictly after calendar date `d`**.

Weekends and holidays roll forward by construction. Equivalently: the next session at or after `d + 1` calendar day.

**Why strictly after.** The regression is `r_(t+1) = f(S_t)`, so everything in `S_t` must be in the information set at `close(t)` — the start of the return window. With a date-only stamp the headline may have been published at 23:59 on `d`. Only the close of the first session after `d` is guaranteed to postdate the whole of day `d`. Assigning `d` to the session *on* `d` would require assuming every headline preceded that session's close, and the audit shows that assumption is false for a share of Benzinga's output.

**What it costs, stated plainly.** The session in which the reaction most plausibly occurs is skipped. A headline dated Monday is assigned to Tuesday and is used to explain the Tuesday-close→Wednesday-close return; the Monday→Tuesday move is not examined at all. So the primary specification tests whether **one-to-three-day-old news** is associated with returns — a weaker claim than "next-day", and the report must use that language rather than borrowing the stronger phrase.

**The lag is not constant, and varies by weekday.** Friday news is assigned to Monday and explains Monday→Tuesday: roughly three calendar days of staleness, against one for Tuesday news. A weekday breakdown is therefore a **prespecified sensitivity exhibit**, not a result to go looking for if the headline number disappoints.

### Secondary rule, reported only as a sensitivity

> A headline dated `d` is assigned to the session on `d` when `d` is a session, else the next session.

This recovers the skipped reaction window and is the specification most of the literature would use. It rests on an assumption the audit contradicts, so it is **never the headline result**, is labelled *"assumes all same-date news precedes that session's close — known to be false for a share of the corpus"*, and is fixed here so it cannot be promoted after the fact.

### RQ2 suppression

Same-day association is inadmissible (B04). Suppression must be **structural, not editorial**: with `config.RQ2_ADMISSIBLE = False` the contemporaneous specification does not run, produces no table, and no figure panel. A result that is not computed cannot be quoted by accident.

## 3. Sample edges

1. A headline whose assigned session falls **before the first session of the window** is dropped, not clipped. `map_*` returns `NaT` below the calendar exactly as it already does above it (fixes **D-1**).
2. A headline whose assigned session falls **after the last session** is dropped — already correct.
3. Dropped counts are reported separately for each edge. A silent drop and a silent clip are the same class of error.
4. The corpus loader also filters to the window, so the mapper's guard is a second line of defence rather than the only one. Both exist because D-1 shows what a single unguarded edge costs.

## 4. Missing market sessions

1. **Leads and lags are built on the complete exchange calendar, before any exclusion.** Concretely: left-join market onto the calendar-indexed daily frame, then shift. Never inner-join first (fixes **D-2**).
2. A session in the calendar with no price row leaves `NaN` in the market columns. Those rows are then excluded at the eligibility stage of §3 of the inference protocol, where the exclusion is counted and reported.
3. **Adjacency is asserted.** After construction, every non-null `ret_lead1` must come from the immediately following session in the calendar. A violation raises; it does not warn.
4. Missing-session counts are reported with the panel. If the count is not small, that is a data-quality finding about `yfinance` coverage and belongs in the write-up.

## 5. Zero-news sessions

Retained in the panel with `n_headlines = 0` so the panel spans the full calendar and the lag/lead structure stays intact. They are excluded at regression time, and the count is reported. This is already the inference protocol's rule; it is restated here because it is what makes rule §4.1 possible.

## 6. Actual session closes: descoped, with a trigger to reopen

P11/B07 requires mapping against each session's real close, including early closes. **For the selected corpus that work has no effect**: every stamp is `00:00 UTC`, so no time-of-day is ever compared against a close, and the mapping in §2 uses session *dates* only.

Decision: **descope B07 for the selected corpus, keep the requirement, record the trigger.** It reopens immediately if the universe changes to an intraday source, if a later FNSPID release adds times, or if the Benzinga 2020+ intraday tail is ever used.

To prevent the fixed 16:00 rule being applied to intraday data by accident once B07 is descoped, `map_to_trading_day` gains a guard: it **raises** when its input looks date-only (the `audit.timestamp_profile` concentration test), rather than mapping midnight stamps as if they were times. The two mappers then refuse each other's inputs, which is the cheapest available protection against using the wrong one.

## 7. Interface review, and the architecture gate

Reviewed: `src/data.trading_calendar`, `src/data.load_market`, `src/align.session_closes`, `src/align.map_to_trading_day`, `src/align.aggregate_daily`, `src/align.build_panel`.

> **Gate outcome: no core signature change is required.** Every defect above is fixable inside the current interfaces. The plan's calendar-handoff gate is **not opened**.

The one place a signature change looked necessary was adjacency checking in `build_panel(daily_scores, market)`, which has no calendar argument. It does not need one: `aggregate_daily` already reindexes the daily frame to the full calendar, so `daily_scores["date"]` **is** the session list. Building the panel by left-joining market onto that frame gives both the full-calendar lead construction and the adjacency check, with no new parameter and no new calendar layer.

| Change | Kind | Signature |
|---|---|---|
| `map_date_to_session(dates, calendar, defer=True)` | new function | additive |
| `map_to_trading_day` — guard lower edge; raise on date-only input | bounded repair | unchanged |
| `aggregate_daily` — dispatch on `config.DATE_ONLY_FALLBACK` | bounded repair | unchanged |
| `build_panel` — left-join not inner; leads on full calendar; assert adjacency; report missing sessions | bounded repair | unchanged |
| `trading_calendar` — raise instead of falling back to business days | bounded repair | unchanged |
| `load_market` — report sessions absent from the calendar | bounded repair | unchanged |
| `load_news` — source-domain filter; drop the obsolete standalone `benzinga` spec | bounded repair | unchanged |

`pandas_market_calendars` must be installed for any real run. Given §1's D-3, it moves from an optional import to a hard requirement.

## 8. What B06–B09 must implement, and the tests that prove it

| Increment | Change | Test that must fail before and pass after |
|---|---|---|
| **B06** | Full-calendar lags/leads; drop the inner join | Zero-news Tuesday still supplies Wednesday's `r_t` control |
| **B06** | Adjacency assertion | The D-2 fixture — a missing price row — raises instead of producing a two-session lead |
| **B07** | Descoped; `map_to_trading_day` raises on date-only input | Feeding a midnight-stamped series raises rather than mapping it |
| **B08** | Lower-edge guard | The D-1 fixture: three pre-window headlines return `NaT`, not the first session |
| **B08** | `trading_calendar` raises without the calendar package | Import failure raises; no business-day frame is returned |
| **B09** | `map_date_to_session` primary rule | Mon→Tue; Fri→Mon; holiday-eve→next session; a date on a session→the *following* session |
| **B09** | Secondary rule under an explicit flag | Same dates map to the session on `d` when `d` is a session |
| **B09** | RQ2 structurally suppressed | With `RQ2_ADMISSIBLE = False`, no contemporaneous table or figure panel is produced |
| **B09** | Source-domain filter | A frame containing `lenta.ru` rows loses them, and the count is reported |
| **B09** | Set `DATE_ONLY_FALLBACK_IMPLEMENTED = True` | The `run_all.py` guard stops firing |

Suggested split, since B08 and B09 each carry two independent behaviours: B06 (calendar + adjacency), B07+B08 (guards and edges), B09a (the mapper), B09b (suppression, filter, flag).

## 9. Open items

| Item | Where |
|---|---|
| Whether the weekday-staleness sensitivity becomes a reported exhibit or a footnote | B25 |
| `yfinance` session coverage against the NYSE calendar over 2010–2019 | measured at B06 |
| Whether the secondary mapping rule is reported at all | B25, decided before results are seen |
| ~~Field renames (`log_turnover` → `log_volume`, etc.)~~ | **Closed at R07a, 2026-09-10** — §10 below |

## 10. Panel field contract (B16 / R07a, 2026-09-10)

`align.PANEL_COLUMNS` is the whole analysis table. Every Act-2 number is read
from it, so a name that does not match its formula is a claim the data does not
support. This section is the mapping the audit's B16 row asked for: what each
column is, how it is computed, and where it is built.

**Two names were wrong** and are corrected here (P22, [inference
protocol](inference-protocol.md) §10):

| Was | Is | Why the old name overclaimed |
|---|---|---|
| `log_turnover` | `log_volume` | `log(share volume)`. Turnover needs a share-count or float denominator; nothing in this pipeline supplies one, so the series was never a turnover ratio |
| `log_turnover_detrended` | `log_volume_detrended` | Same series, trailing mean removed |
| `parkinson` | `rv_parkinson` | A high-low **range** estimator of variance. It cannot see the intraday path or the overnight gap, so it is not daily volatility |

The derived columns follow their stems: `log_volume_lag1`,
`log_volume_detrended_lead1`, `rv_parkinson_lag1`, `rv_parkinson_lead1`.
`build_panel`'s window parameter is `volume_window` (was `turnover_window`).

### The columns

| Column | Formula | Built in | Notes |
|---|---|---|---|
| `date` | exchange session | `aggregate_daily` | the complete calendar; a row shift is a session shift |
| `n_headlines` | count mapped to session `t` | `aggregate_daily` | 0 on a zero-news session; the D9 eligibility field |
| `s_<scorer>` | mean headline tone on `t` | `aggregate_daily` | |
| `d_<scorer>` | within-day SD of headline tone | `aggregate_daily` | defined only when `n_t >= 5`; **dispersion**, not disagreement (P21) |
| `ret` | `log(C_t / C_(t-1))` over adjacent **sessions** | `load_market` | computed on the calendar, not on whatever rows the price source returned (R04a) |
| `ret_lag1` | `ret.shift(1)` | `build_panel` | full-calendar shift, before any exclusion |
| `ret_lead<h>` | `ret.shift(-h)` | `build_panel` | the return **of** session `t+h`, not a cumulative `h`-day return |
| `rv_parkinson` | `(log(H_t / L_t))^2 / (4 log 2)` | `load_market` | intraday range-based variance proxy |
| `rv_parkinson_lag1` / `_lead1` | `shift(1)` / `shift(-1)` | `build_panel` | |
| `log_volume` | `log(V_t)`, `V_t = 0` treated as missing | `load_market` | log share volume |
| `log_volume_lag1` | `shift(1)` | `build_panel` | |
| `log_volume_detrended` | `log_volume` minus its **trailing** 63-session mean ending at `t` | `build_panel` | trailing only; a centred window leaks future volume into `t`. Missing for the first 62 sessions by construction |
| `log_volume_detrended_lead1` | `shift(-1)` | `build_panel` | |
| `close_adj` | split/dividend-adjusted close | `load_market` | context figure only; never a regressor (P29) |
| `vix_close` | VIX close on `t` | `load_market` | recorded, not in the frozen control set |

`ret`, `rv_parkinson`, `log_volume` and `close_adj` are
`align.REQUIRED_MARKET_COLUMNS`: `build_panel` raises if the market frame lacks
any of them rather than NaN-filling, because an all-missing control does not
announce itself — it empties the regression sample and is then reported as a
sample size.

### Why a rename needs guards

`build_panel` fills any declared panel column its inputs did not supply with
`NaN`. That is deliberate for genuinely optional fields, and it is exactly what
makes a rename dangerous: a market frame or saved panel written before this
section would carry `log_turnover`, `log_volume` would be created empty, and
every downstream regression would silently fit on a shorter sample instead of
failing. Two guards close that path:

- `align.reject_legacy_columns` — called on both `build_panel` inputs; raises
  naming each stale column and its replacement.
- `align.assert_panel_schema` — called by `run_all.py` on the `--skip-panel`
  path, where a panel is read from disk and its provenance is whatever wrote it.

`tests/test_panel_fields.py` (12 tests) checks each formula against fixture
bars, checks that the detrend window is trailing-only, and checks that both
guards fire.

### What R07a did **not** change

The rename is a rename. `inference.predictive` still uses **raw** `log_volume`
where the protocol specifies the detrended series, and tone is still not
standardized at fit time. That is a different model, not a different column name
(A07), and it is R07b's work.
