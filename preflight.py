"""Read-only readiness checks: no acquisition, no scoring, no regression (R10).

    python preflight.py                                   # dependencies + artifacts
    python preflight.py --stage analysis                  # what Act 2 consumes
    python preflight.py --stage pilot --load-models       # also build each scorer
    python preflight.py --stage analysis --output report/preflight.json
    python preflight.py --freeze                          # pin lines from this env
    python preflight.py --strict                          # exit 1 if not ready

Reports; it installs nothing, downloads nothing and changes no setting. A
missing prerequisite is printed with the command that produces it, and deciding
to run that command is yours.

Two depths, deliberately separate. **Existence** is cheap and runs on every
command through `preflight.require_artifacts`. **Verification** -- parquet
schemas, the LM dictionary digest, a complete score cache with matching scorer
identities, and optionally loading each model -- costs real time and is asked
for here. A file being present is not the same as it being the right file.

Failures are reported with their original exception type. A model that fails to
load is reported as the error it raised and never converted into a skip: that
conversion is what let a real FinBERT label-order defect read as "unavailable"
for weeks (audit A13).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import preflight


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=sorted(preflight.STAGES), default="unit",
                    help="which stage's prerequisites to report (default: unit)")
    ap.add_argument("--load-models", action="store_true",
                    help="also construct every configured scorer and print its "
                         "fingerprint; slow, and downloads weights if absent")
    ap.add_argument("--cache", type=Path, default=None,
                    help="score cache to verify instead of config.SCORES_PARQUET")
    ap.add_argument("--output", type=Path, default=None,
                    help="write the full report as JSON (for a run manifest)")
    ap.add_argument("--freeze", action="store_true",
                    help="print pin lines for every declared requirement, from "
                         "the installed environment")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when anything is absent, missing or failing")
    args = ap.parse_args()

    if args.freeze:
        print(preflight.freeze())
        return 0

    print(preflight.summary(args.stage))

    checks = preflight.verify(args.stage, load_models=args.load_models,
                              cache_path=args.cache)
    if checks:
        print()
        print(f"verification ({args.stage}):")
        for c in checks:
            print(f"    [{'OK' if c['ok'] else '   FAIL':>7}] {c['check']}: {c['detail']}")

    deps = preflight.dependency_report()
    absent = deps.loc[deps["status"] == "absent", "package"].tolist()
    drift = deps.loc[deps["status"] == "mismatch", "package"].tolist()
    missing = preflight.missing_artifacts(args.stage)
    failed = [c["check"] for c in checks if not c["ok"]]

    print()
    if drift:
        print(f"environment differs from requirements.txt: {', '.join(drift)}")
    if absent:
        print(f"absent packages   : {', '.join(absent)}")
    if missing:
        print(f"missing artifacts : {', '.join(a.key for a in missing)}")
    if failed:
        print(f"failed checks     : {', '.join(failed)}")
    if not (absent or missing or failed):
        print(f"ready for stage {args.stage!r}.")

    if args.output:
        report = preflight.report(args.stage, load_models=args.load_models,
                                  cache_path=args.cache)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n",
                               encoding="utf-8")
        print(f"wrote {args.output}")

    # Version drift is a reproducibility fact to look at, not a reason to block a
    # run; absence, a missing artifact or a failed check is a genuine stop.
    return 1 if args.strict and (absent or missing or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
