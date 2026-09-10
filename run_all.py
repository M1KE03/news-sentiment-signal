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
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

import config
from src import align, data, inference, plots, preflight, publish, scoring

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
    scores, meta = scoring.load_cache(_require(config.SCORES_PARQUET, "run `python rescore.py`"))
    market = pd.read_parquet(_require(config.MARKET_PARQUET, "run Stage 0"))

    # R04b: the score artifact must declare a measurement identity for every
    # configured scorer before anything is aggregated. A cache column with no
    # recorded provenance cannot be attributed to a measurement, and a panel
    # built from it would carry numbers nobody can trace to an artifact.
    declared = (meta or {}).get("fingerprints", {})
    undeclared = [s for s in config.SCORERS if not declared.get(s)]
    if undeclared:
        raise SystemExit(
            f"score cache declares no provenance for {undeclared}.\n"
            f"  cache: {config.SCORES_PARQUET}\n"
            "  Re-run scoring so the cache records which measurement produced "
            "each column; see docs/scoring-checkpoint-contract.md."
        )

    calendar = data.trading_calendar(config.SAMPLE_START, config.SAMPLE_END)
    # require_complete=True: refuse a partially scored corpus rather than let
    # each scorer summarise a different subset of each session (audit A03).
    daily = align.aggregate_daily(scores, headlines, calendar)
    cov = daily.attrs.get("coverage", {})
    print(
        f"scores: {cov.get('n_headlines_scored', 0):,} of "
        f"{cov.get('n_headlines_assigned', 0):,} assigned headlines scored"
    )
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
    # A15: record what this panel was built from, so `--skip-panel` can tell
    # whether it still matches the world rather than only that its columns do.
    panel.attrs[publish.PANEL_PROVENANCE_KEY] = json.dumps(
        publish.panel_provenance(), default=str)
    panel.to_parquet(config.PANEL_PARQUET, index=False)
    return panel


def write_advance_precision(record: dict) -> None:
    """Publish the planning record atomically; failure must stop estimation."""
    payload = json.dumps(record, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=TABLES,
                                         prefix=".advance-precision-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, TABLES / "advance_precision.json")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_analysis(panel: pd.DataFrame, run: publish.StagedRun) -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    # Persist advance planning before any tone/outcome regression is called.
    precision = inference.primary_precision(panel)
    write_advance_precision(precision)
    sample, coverage = inference.analysis_sample(panel)
    _, eligibility_ledger = inference.eligibility(panel)
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
        pd.DataFrame(rows).to_csv(run.path_for("tables", "table2_contemporaneous.csv"), index=False)
    else:
        print(
            "RQ2 suppressed: no selected source has usable intraday timestamps, "
            "so the same-day specification is not estimated and Table 2 is not "
            "produced (docs/data-audit-fnspid.md)."
        )

    # --- Table 3: unadjusted primary + 14 secondary return tests (BH/BY) --
    families = {sc: inference.lag_family(panel, sc) for sc in config.SCORERS}
    table3 = inference.adjust_return_families(pd.concat(families.values(), ignore_index=True))
    families = {sc: table3.loc[table3["scorer"] == sc].copy() for sc in config.SCORERS}
    table3.to_csv(run.path_for("tables", "table3_lag_family.csv"), index=False)

    # --- timing diagnostic (protocol section 9) --------------------------
    # A percentile rank, never a p-value: the old block-resampling "placebo"
    # was removed at R08c. Full circular shift of standardized tone over the
    # retained rows, controls and outcomes left in place.
    timing = {}
    for sc in config.SCORERS:
        d = inference.timing_diagnostic(panel, sc)
        timing[sc] = {k: v for k, v in d.items() if k != "shifted_coefficients"}
        print(
            f"timing diagnostic {sc}: observed {d['observed_bps']:.2f} bps, "
            f"percentile {d['percentile']:.3f} of {d['n_shifts']} shifts "
            f"({d['n_ties']} ties); {d['n_gaps']} gap(s), largest "
            f"{d['largest_gap_sessions']} session(s). Not a p-value."
        )
    pd.DataFrame(timing.values()).to_csv(run.path_for("tables", "table_timing_diagnostic.csv"), index=False)

    # --- Table 4: effect sizes in bps per 1 sigma (D15) ------------------
    table4 = pd.concat(
        [inference.effect_size_table(fam) for fam in families.values()],
        ignore_index=True,
    )
    table4.to_csv(run.path_for("tables", "table4_effect_sizes.csv"), index=False)

    # --- comparing scorers (protocol section 7) --------------------------
    # (a) the paired difference, with the cross-equation HAC covariance. This
    # is the only quantity that supports a "one beats the other" statement
    # (section 11); two separate t-statistics do not.
    contrast = inference.paired_scorer_contrast(panel)
    contrast.table().to_csv(run.path_for("tables", "table_paired_scorer_contrast.csv"), index=False)
    wald = contrast.wald()
    corr = contrast.diagnostics["corr_tone"]
    point, lo, hi = wald["bps"]
    print(
        f"delta(finbert - lm) = {point:.2f} bps per 1 SD, 95% pointwise "
        f"[{lo:.2f}, {hi:.2f}], n = {contrast.n}"
        + ("  [degenerate: the two scorers carry identical information]"
           if wald["degenerate"] else f", p = {wald['p']:.3f}")
    )
    # Reported, not acted on: a high correlation widens delta's interval, and
    # that widening is already the honest statement of what can be distinguished.
    print(f"  corr(z_finbert, z_lm) = {corr:.3f}; "
          f"tone VIFs {contrast.diagnostics['vif_tone']}")

    # (b) a different estimand: does one add information given the other?
    incremental = inference.incremental_contribution(panel)
    pd.DataFrame({
        "term": incremental.names,
        "coef": incremental.params.to_numpy(),
        "nw_se": incremental.bse.to_numpy(),
        "p": incremental.pvalues.to_numpy(),
    }).assign(estimand=incremental.estimand, nobs=incremental.nobs,
              interval_scope="pointwise").to_csv(
        run.path_for("tables", "table_incremental_contribution.csv"), index=False)

    # --- Table 5: RQ4 exploratory family ---------------------------------
    # The FAMILY is the joint Wald tests; individual coefficients are
    # descriptive and are written to a separate file so the two cannot be read
    # as one table (protocol section 6, P21).
    table5 = inference.rq4_family(panel)
    table5.to_csv(run.path_for("tables", "table5_rq4_wald.csv"), index=False)
    inference.rq4_coefficients(panel).to_csv(
        run.path_for("tables", "table5b_rq4_coefficients.csv"), index=False)
    for _, r in table5.iterrows():
        print(f"RQ4 {r['scorer']}/{r['outcome']}: Wald chi2({int(r['df'])}) = "
              f"{r['wald_chi2']:.2f}, p = {r['p']:.4f}, BH q = {r['bh_q']:.4f}, "
              f"n = {int(r['nobs'])}")

    # --- figures ----------------------------------------------------------
    for name, figure in (
        ("figure2_lag_family", plots.figure2_lag_family(families, contemp, timing)),
        ("figure3_dispersion_volume", plots.figure3_dispersion_volume(sample)),
        ("figure4_context", plots.figure4_context(panel)),
        ("figure_acf_sentiment", plots.figure_acf(sample)),
        ("figure_coverage", plots.figure_coverage(panel)),
    ):
        figure.savefig(run.path_for("figures", f"{name}.png"), bbox_inches="tight")

    summary = {
        "advance_precision": "advance_precision.json",
        "coverage": coverage,
        "timing_diagnostic": timing,
        "corr_finbert_lm": corr,
        "rq2_admissible": config.RQ2_ADMISSIBLE,
        "date_only_fallback": config.DATE_ONLY_FALLBACK,
        "mapping_rule": (
            "first session strictly after the headline date"
            if config.DATE_ONLY_FALLBACK
            else "intraday close rule"
        ),
    }
    run.path_for("tables", "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    run.extra["eligibility"] = eligibility_ledger
    run.extra["coverage"] = coverage
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-stale-panel", action="store_true",
                    help="reuse a panel whose provenance does not match current "
                         "inputs or settings; prints what differs")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="run without checking prerequisites (not recommended)")
    ap.add_argument("--skip-panel", action="store_true",
                    help="reuse processed/daily_panel.parquet instead of rebuilding it")
    args = ap.parse_args()

    _check_locked_decisions()

    if not args.skip_preflight:
        # A15: readiness stopped at three config constants, so a missing corpus
        # or an unscored cache surfaced as a traceback from inside a loader
        # rather than as a statement of what to run first. Reported as a plain
        # message, like the locked-decision guard beside it -- a stack trace
        # here says "this program broke" when the truth is "run that first".
        try:
            preflight.require_artifacts("analysis")
        except preflight.PreflightError as missing:
            raise SystemExit(str(missing)) from None

    if args.skip_panel:
        panel = pd.read_parquet(_require(config.PANEL_PARQUET, "drop --skip-panel"))
        # A reused panel is an artifact of whatever code wrote it, which may
        # predate the current field contract (B16/P22 renames).
        align.assert_panel_schema(panel)
        # ... and whose columns can match while its semantics do not (A15).
        recorded = panel.attrs.get(publish.PANEL_PROVENANCE_KEY)
        problems = publish.check_panel_provenance(
            json.loads(recorded) if recorded else None)
        if problems and not args.allow_stale_panel:
            raise SystemExit(
                "the reused panel does not match the current inputs or settings:\n  "
                + "\n  ".join(problems)
                + "\n\nRebuild it by dropping --skip-panel, or pass "
                  "--allow-stale-panel to proceed deliberately. Results from a "
                  "mismatched panel would be labelled with settings that did not "
                  "produce them."
            )
        if problems:
            print("WARNING: reusing a panel that does not match current inputs:")
            for problem in problems:
                print(f"  {problem}")
    else:
        panel = build_panel()
        print(f"wrote {config.PANEL_PARQUET}  ({len(panel):,} rows)")

    destinations = {"tables": TABLES, "figures": config.FIGURES}
    with publish.StagedRun(destinations, manifest_dir=TABLES) as run:
        run_analysis(panel, run)
    print(f"\npublished {len(run.published)} output(s) and "
          f"{publish.MANIFEST_NAME}; nothing was written until the run completed.")
    print("\nAct 1 (Table 1, Figure 1) is produced by notebooks/02_validation.ipynb "
          "-- it is an independent branch and needs the PhraseBank, not the panel.")


if __name__ == "__main__":
    sys.exit(main())
