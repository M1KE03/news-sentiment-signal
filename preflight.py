"""Read-only dependency/artifact checks; no acquisition, scoring or regression.

python preflight.py --stage pilot --load-models --output report/preflight.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

import config

CORE = {"numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow", "scipy": "scipy",
        "statsmodels": "statsmodels", "scikit-learn": "sklearn", "matplotlib": "matplotlib",
        "pandas-market-calendars": "pandas_market_calendars", "pytest": "pytest"}
MODELS = {"torch": "torch", "transformers": "transformers", "vaderSentiment": "vaderSentiment.vaderSentiment"}


def inspect_environment(stage="unit", *, load_models=False, strict_versions=False, cache_path=None) -> dict:
    if stage not in {"unit", "pilot", "scoring", "validation", "analysis"}:
        raise ValueError("unknown preflight stage")
    pins = {}
    for line in (config.ROOT / "requirements.txt").read_text().splitlines():
        requirement = line.split("#", 1)[0].strip()
        if "==" in requirement:
            name, version = requirement.split("==", 1); pins[name] = version.strip()
    inventory = []
    for name, pin in pins.items():
        try: installed = metadata.version(name)
        except metadata.PackageNotFoundError: installed = None
        inventory.append({"package": name, "pinned": pin, "installed": installed, "matches_pin": installed == pin})
    checks = []

    def check(name, operation):
        try:
            detail = operation()
            checks.append({"check": name, "ok": True, "detail": str(detail)})
        except Exception as exc:
            # Report failures with their original type, never turn model bugs into skips.
            checks.append({"check": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})

    modules = dict(CORE)
    if stage != "unit" or load_models: modules.update(MODELS)
    for package, module in modules.items():
        check(f"import:{package}", lambda module=module: importlib.import_module(module).__name__)
    if strict_versions:
        for item in inventory:
            checks.append({"check": f"pin:{item['package']}", "ok": item["matches_pin"], "detail": str(item)})

    def parquet(path, required):
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        missing = set(required) - set(schema.names)
        if missing: raise ValueError(f"missing columns {sorted(missing)}; rebuild the named artifact")
        return f"{path}: {pq.read_metadata(path).num_rows} rows"

    if stage != "unit":
        check("headlines", lambda: parquet(config.HEADLINES_PARQUET, ["headline_id", "text", "ts_utc"]))
        def dictionary():
            actual = hashlib.sha256(config.LM_DICT_PATH.read_bytes()).hexdigest()
            if actual != config.LM_DICT_SHA256: raise ValueError("LM dictionary SHA-256 differs from config pin")
            return f"{config.LM_DICT_PATH}: verified {actual}"
        check("dictionary", dictionary)
    if load_models:
        from src import scoring
        for name in config.SCORERS:
            check(f"model:{name}", lambda name=name: scoring.build_scorers([name])[0].fingerprint)
    if stage == "validation":
        from src import annotate, validate
        check("frozen_split", lambda: len(annotate.load_split()))
        def labels():
            return len(validate.load_annotations(config.ANNOTATION_DIR / "labels_primary.csv",
                       config.ANNOTATION_DIR / "provenance_primary.json", part="evaluation"))
        check("human_evaluation_labels", labels)
        checks.append({"check": "frozen_thresholds", "ok": config.VALIDATION_THRESHOLDS is not None,
                       "detail": "config.VALIDATION_THRESHOLDS must be recorded after calibration"})
    if stage == "analysis":
        from src import scoring, align
        check("market", lambda: parquet(config.MARKET_PARQUET, ["date", "ret", "log_volume", "rv_parkinson"]))
        def score_cache():
            import pandas as pd
            path = config.SCORES_PARQUET if cache_path is None else Path(cache_path)
            scores, meta = scoring.load_cache(path)
            headlines = pd.read_parquet(config.HEADLINES_PARQUET)
            align.validate_scores(headlines, scores)
            expected = scoring._configured_fingerprints()
            for name, fp in expected.items():
                if scoring._canonical_fingerprint(fp) != meta["fingerprints"].get(name):
                    raise ValueError(f"{name} cache fingerprint differs from configured scorer")
            return f"{path}: complete, verified scorer identities"
        check("complete_scores", score_cache)
    return {"stage": stage, "ok": all(c["ok"] for c in checks), "python": sys.version,
            "platform": platform.platform(), "executable": sys.executable,
            "checks": checks, "dependencies": inventory,
            "scope": "dependency/artifact readiness only; does not certify unfinished research methods"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["unit", "pilot", "scoring", "validation", "analysis"], default="unit")
    parser.add_argument("--load-models", action="store_true")
    parser.add_argument("--strict-versions", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_environment(args.stage, load_models=args.load_models, strict_versions=args.strict_versions, cache_path=args.cache)
    for check in report["checks"]: print(f"{'OK' if check['ok'] else 'FAIL'} {check['check']}: {check['detail']}")
    drift = [d["package"] for d in report["dependencies"] if not d["matches_pin"]]
    if drift: print("Environment differs from requirements.txt: " + ", ".join(drift))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
