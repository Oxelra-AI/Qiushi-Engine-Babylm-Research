#!/usr/bin/env python3
"""research: rebuild FW comparison eval summary from existing per-target JSONs.

Use after a research evaluation timed out after writing some valid per-target files.
This script does not run model evaluation; it imports the validated research parser,
parses all available `step086_{arm}_chck_{checkpoint}.json` files, and rewrites the
standard summary/markdown so `fw_absolute_decision.py` can read a merged
view of 70M/80M/100M results.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
import json
import pathlib
import re
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
EVAL_SCRIPT = WORKSPACE / "scripts/eval_fw_comparison.py"
OUT_ROOT = WORKSPACE / "data/fw_comparison_eval"
PER_TARGET = OUT_ROOT / "per_target"
SUMMARY_JSON = OUT_ROOT / "fw_comparison_eval_summary.json"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_fw_comparison", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EVAL_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def checkpoint_sort_key(ckpt: str) -> tuple[int, str]:
    if ckpt.endswith("M") and ckpt[:-1].isdigit():
        return (int(ckpt[:-1]), ckpt)
    return (10**9, ckpt)


def main() -> None:
    mod = load_eval_module()
    parser_validation = mod.validate_parser()
    if parser_validation.get("status") != "PARSER_VALIDATION_OK":
        raise SystemExit(f"Parser validation failed: {parser_validation}")

    pat = re.compile(r"^step086_(compact_view|source_breadth)_chck_(.+)\.json$")
    results: dict[str, dict[str, Any]] = {}
    checkpoints: set[str] = set()
    incomplete: dict[str, Any] = {}

    for path in sorted(PER_TARGET.glob("step086_*_chck_*.json")):
        m = pat.match(path.name)
        if not m:
            continue
        arm, ckpt = m.group(1), m.group(2)
        parsed = mod.parse_target_scores(path)
        parsed.update({
            "label": f"{arm}_{ckpt}",
            "arm": arm,
            "checkpoint": ckpt,
            "recovered_from_per_target": str(path),
        })
        results[f"{arm}_{ckpt}"] = parsed
        checkpoints.add(ckpt)
        if parsed.get("status") != "OK":
            incomplete[f"{arm}_{ckpt}"] = {"status": parsed.get("status"), "missing_scores": parsed.get("missing_scores")}

    ordered = sorted(checkpoints, key=checkpoint_sort_key)
    comparisons: dict[str, Any] = {}
    for ckpt in ordered:
        comp = mod.compare_checkpoint(results, ckpt)
        if comp is not None:
            comparisons[ckpt] = comp

    summary = {
        "status": "FW_COMPARISON_EVAL_REBUILT_FROM_PER_TARGETS",
        "created_utc": now_utc(),
        "checkpoints": ordered,
        "arms": {k: str(v) for k, v in mod.ARMS.items()},
        "columns": mod.CHEAP_COLUMNS,
        "cheap7_columns": mod.SCALAR_COLUMNS,
        "parser_validation": parser_validation,
        "results": results,
        "comparisons": comparisons,
        "incomplete_results": incomplete,
        "source": "experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/*.json",
        "no_model_eval_run": True,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md = mod.write_markdown(summary)
    print(json.dumps({
        "status": summary["status"],
        "summary_json": str(SUMMARY_JSON),
        "summary_md": str(out_md),
        "checkpoints": ordered,
        "ok_results": [k for k, v in results.items() if v.get("status") == "OK"],
        "incomplete_results": incomplete,
        "comparisons": {k: v for k, v in comparisons.items() if v.get("status") == "OK"},
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
