#!/usr/bin/env python3
"""Evaluate research whole-word copied-content control on the same cheap7-style surface.

Run after `target_selective_drop_copied_content_wholeword_20M` has produced
`hf_model/chck_20M`.  This calls the current official-compatible endpoint evaluator and
then appends a comparison against the research repaired full/drop_abs/token-copied results.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
EVALUATOR = pathlib.Path("experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py")
RUN_DIR = ROOT / "training/runs/target_selective_drop_copied_content_wholeword_20M"
OUT_ROOT = ROOT / "data/wholeword_control_cheap7_eval/wholeword20"
COLLATE_ROOT = ROOT / "data/wholeword_control_cheap7_eval/collated"
TARGET = "dropcopied_word20"
JSONS = {
    "full": ROOT / "data/target_selective_cheap7_eval_repaired/full20/per_target/full20.json",
    "drop_abs": ROOT / "data/target_selective_cheap7_eval_repaired/dropabs20/per_target/dropabs20.json",
    "drop_copied_tok": ROOT / "data/target_selective_cheap7_eval_repaired/dropcopied20/per_target/dropcopied20.json",
}
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def load_scores(path: pathlib.Path) -> dict[str, float]:
    j = json.loads(path.read_text(encoding="utf-8"))
    sc = {c: j["official_overall"]["scores"][c] for c in COLUMNS}
    sc["cheap7_equal_mean"] = round(sum(sc[c] for c in COLUMNS) / len(COLUMNS), 6)
    return sc


def compare(out_json: pathlib.Path, compare_out: pathlib.Path) -> None:
    scores = {k: load_scores(v) for k, v in JSONS.items()}
    scores["drop_copied_word"] = load_scores(out_json)
    deltas = {}
    for lhs, rhs in [
        ("drop_abs", "drop_copied_word"),
        ("drop_copied_word", "full"),
        ("drop_abs", "full"),
        ("drop_abs", "drop_copied_tok"),
        ("drop_copied_word", "drop_copied_tok"),
    ]:
        key = f"{lhs}_minus_{rhs}"
        deltas[key] = {c: round(scores[lhs][c] - scores[rhs][c], 6) for c in COLUMNS + ["cheap7_equal_mean"]}
    payload = {
        "status": "WHOLEWORD_EXTERNAL_COMPARISON",
        "meaning": "20M official-style surface for the whole-word copied-content control. Use as early external coupling only, not as route survival.",
        "scores": scores,
        "deltas": deltas,
        "wholeword_json": str(out_json),
        "jsons": {k: str(v) for k, v in JSONS.items()},
    }
    compare_out.parent.mkdir(parents=True, exist_ok=True)
    compare_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(compare_out), "scores": scores, "deltas": deltas}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()
    model = RUN_DIR / "hf_model/chck_20M"
    metrics = RUN_DIR / "scientific_metrics.json"
    if not model.exists() or not metrics.exists():
        raise FileNotFoundError(f"Need completed training before eval: model={model.exists()} metrics={metrics.exists()}")
    if not args.skip_eval:
        cmd = [
            sys.executable, "-B", str(EVALUATOR),
            "--arm", "reinvest",
            "--target", TARGET,
            "--run-dir", str(RUN_DIR),
            "--endpoint", "chck_20M",
            "--out-root", str(OUT_ROOT),
            "--collate-root", str(COLLATE_ROOT),
            "--gpu", str(args.gpu),
            "--columns", "BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading",
            "--force",
        ]
        print(json.dumps({"event": "run_eval", "cmd": cmd}), flush=True)
        subprocess.run(cmd, check=True)
    out_json = OUT_ROOT / "per_target" / f"{TARGET}.json"
    if not out_json.exists():
        raise FileNotFoundError(out_json)
    compare(out_json, ROOT / "data/wholeword_control_cheap7_eval/wholeword_external_comparison.json")


if __name__ == "__main__":
    main()
