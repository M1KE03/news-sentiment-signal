"""Rebuild the panel, then every table and every figure. D16.

    pip install -r requirements.txt
    python rescore.py        # once: the expensive FinBERT pass, cached
    python run_all.py        # minutes, no GPU

This script reads `interim/headlines.parquet`, `interim/scores.parquet` and
`interim/market.parquet`, builds `processed/daily_panel.parquet`, and writes
Tables 1-5 to `report/tables/` and Figures 1-4 to `figures/`.

Every Act-2 number comes from the panel. If a result is wrong it is wrong in the
panel or in the regression -- never in an ad-hoc join inside a notebook.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

import config
from src import align, data, inference, plots, scoring

TABLES = config.REPORT / "tables"


def _require(path, what: str):
    if not path.exists():
        raise SystemExit(f"missing {path} -- {what}")
    return path


def _check_locked_decisions() -> None:
    pending = [
        name
        for name in ("NEWS_SOURCE", "SAMPLE_START", "SAMPLE_END")
        if getattr(config, name) is None
    ]
    if pending:
        raise SystemExit(
            "config.py still has unresolved decisions: "
            + ", ".join(pending)
            + "\nResolve the source and window in the dataset audit (B03/B04) first."
        )

    # Standing guard. Satisfied since B09, and kept because the failure it
    # prevents is silent: applying the intraday 16:00 ET rule to timestamps that
    # carry no time would put every headline on the session it was dated instead
    # of deferring it, inventing an information boundary the data cannot support.
    # If the fallback is ever switched on again for a new corpus, this stops the
    # pipeline until the mapping is actually wired up.
    if config.DATE_ONLY_FALLBACK and not config.DATE_ONLY_FALLBACK_IMPLEMENTED:
        raise SystemExit(
            "STOP: config.DATE_ONLY_FALLBACK is active but not implemented.\n"
            "The selected corpus has date-only timestamps; applying the intraday\n"
            "close rule to them would assign each headline to the session it was\n"
            "dated rather than deferring it.\n"
            "Wire up align.map_date_to_session and the RQ2 suppression, set\n"
            "DATE_ONLY_FALLBACK_IMPLEMENTED = True, then re-run."
        )


def build_panel() -> pd.DataFrame:
    """headlines + scores + market -> the single analysis table."""
    headlines = pd.read_parquet(_require(config.HEADLINES_PARQUET, "run Stage 0"))
    scores, _ = scoring.load_cache(_require(config.SCORES_PARQUET, "run `python rescore.py`"))
    market = pd.read_parquet(_require(config.MARKET_PARQUET, "run Stage 0"))

    calendar = data.trading_calendar(config.SAMPLE_START, config.SAMPLE_END)
    daily = align.aggregate_daily(scores, headlines, calendar)
    panel = align.build_panel(daily, market)

    # Row shifts are session shifts only if the panel is the complete calendar.
    # Check it here, where the calendar is in scope, rather than trusting it.
    align.assert_sessions_match_calendar(panel, calendar)

    stats = panel.attrs.get("build_stats", {})
    print(
        f"panel built: {stats.get('n_sessions', len(panel))} sessions, "
        f"{stats.get('n_missing_market', 0)} with no market row"
    )
    if stats.get("n_missing_market"):
        print(
            "  sessions absent from the market data (first 10): "
            f"{stats['missing_market_dates']}\n"
            "  their leads are NaN and they are excluded at the eligibility "
            "stage; if this count is not small it is a data-quality finding "
            "about the price source and belongs in the write-up."
        )

    config.PANEL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.PANEL_PARQUET, index=False)
    return panel


def run_analysis(panel: pd.DataFrame, draws: int) -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    sample, coverage = inference.analysis_sample(panel)
    print(
        f"panel: {coverage['n_sessions']} sessions, "
        f"{coverage['n_zero_news_days']} zero-news days dropped (D9), "
        f"{coverage['n_analysis_days']} in the analysis sample"
    )

    # --- Table 2: contemporaneous (RQ2) ---------------------------------
    # Structurally suppressed when the corpus cannot support a same-day claim.
    # No table is written and no t=0 point reaches Figure 2, so the absence is
    # visible in the outputs rather than resting on someone remembering it.
    contemp: dict | None = None
    if config.RQ2_ADMISSIBLE:
        contemp, rows = {}, []
        for sc in config.SCORERS:
            res = inference.contemporaneous(sample, sc)
            key = f"s_{sc}"
            contemp[sc] = {"coef": float(res.params[key]), "nw_se": float(res.bse[key])}
            rows.append(
                {"scorer": sc, "coef": res.params[key], "nw_se": res.bse[key],
                 "t": res.tvalues[key], "p": res.pvalues[key], "nobs": int(res.nobs)}
            )
        pd.DataFrame(rows).to_csv(TABLES / "table2_contemporaneous.csv", index=False)
    else:
        stale = TABLES / "table2_contemporaneous.csv"
        if stale.exists():
            stale.unlink()      # never leave a table from an earlier configuration
        print(
            "RQ2 suppressed: no selected source has usable intraday timestamps, "
            "so the same-day specification is not estimated and Table 2 is not "
            "produced (docs/data-audit-fnspid.md)."
        )

    # --- Table 3: the lag family, BH-corrected (RQ3) ---------------------
    families = {sc: inference.lag_family(sample, sc) for sc in config.SCORERS}
    table3 = pd.concat(families.values(), ignore_index=True)
    table3.to_csv(TABLES / "table3_lag_family.csv", index=False)

    # --- placebo (D12) ---------------------------------------------------
    placebo = {}
    for sc in config.SCORERS:
        r = inference.permutation_pvalue(sample, sc, n=draws)
        placebo[sc] = r["p_permutation"]
        print(f"placebo {sc}: t = {r['t_observed']:.2f}, p = {r['p_permutation']:.3f}")

    # --- Table 4: effect sizes in bps per 1 sigma (D15) ------------------
    table4 = pd.concat(
        [inference.effect_size_table(fam, sample[f"s_{sc}"].std())
         for sc, fam in families.items()],
        ignore_index=True,
    )
    table4.to_csv(TABLES / "table4_effect_sizes.csv", index=False)

    # --- the attenuation bridge (6.3) ------------------------------------
    atten = inference.attenuation_comparison(sample)
    atten.to_csv(TABLES / "table_attenuation.csv", index=False)
    corr = atten.attrs.get("corr_s")
    print(f"corr(s_finbert, s_lm) = {corr:.3f}")
    if corr is not None and corr > 0.9:
        print(
            "  -> the two daily series are near-collinear; the attenuation "
            "comparison is bounded and must be reported as inconclusive, not "
            "redesigned until it separates them."
        )

    # --- Table 5: volatility and volume (RQ4) ----------------------------
    rows = []
    for sc in ("finbert", "lm"):
        for outcome, fit in (
            ("parkinson_lead1", inference.volatility_spec(sample, sc)),
            ("log_turnover_detrended_lead1", inference.volume_spec(sample, sc)),
        ):
            for term in fit.params.index:
                rows.append(
                    {"scorer": sc, "outcome": outcome, "term": term,
                     "coef": fit.params[term], "nw_se": fit.bse[term],
                     "t": fit.tvalues[term], "p": fit.pvalues[term],
                     "nobs": int(fit.nobs)}
                )
    table5 = pd.DataFrame(rows)
    table5.to_csv(TABLES / "table5_vol_volume.csv", index=False)

    # --- figures ----------------------------------------------------------
    plots.save(plots.figure2_lag_family(families, contemp, placebo), "figure2_lag_family")
    plots.save(plots.figure3_dispersion_volume(sample), "figure3_dispersion_volume")
    plots.save(plots.figure4_context(panel), "figure4_context")
    plots.save(plots.figure_acf(sample), "figure_acf_sentiment")
    plots.save(plots.figure_coverage(panel), "figure_coverage")

    summary = {
        "coverage": coverage,
        "placebo": placebo,
        "corr_finbert_lm": corr,
        "rq2_admissible": config.RQ2_ADMISSIBLE,
        "date_only_fallback": config.DATE_ONLY_FALLBACK,
        "mapping_rule": (
            "first session strictly after the headline date"
            if config.DATE_ONLY_FALLBACK
            else "intraday close rule"
        ),
    }
    (TABLES / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-panel", action="store_true",
                    help="reuse processed/daily_panel.parquet instead of rebuilding it")
    ap.add_argument("--draws", type=int, default=config.PERMUTATION_DRAWS,
                    help="permutation draws (lower it for a smoke run)")
    args = ap.parse_args()

    _check_locked_decisions()

    if args.skip_panel:
        panel = pd.read_parquet(_require(config.PANEL_PARQUET, "drop --skip-panel"))
    else:
        panel = build_panel()
        print(f"wrote {config.PANEL_PARQUET}  ({len(panel):,} rows)")

    run_analysis(panel, args.draws)
    print("\nAct 1 (Table 1, Figure 1) is produced by notebooks/02_validation.ipynb "
          "-- it is an independent branch and needs the PhraseBank, not the panel.")


if __name__ == "__main__":
    sys.exit(main())
