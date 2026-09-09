#!/usr/bin/env python3
"""research: conflict-free evaluator/merger for Muon→AdamW switch screens.

Use this after the switch 80M trainings finish. It can evaluate one arm/checkpoint
and one or more official-compatible cheap columns using a unique part target, so
multiple jobs can run in parallel on two GPUs without writing the same JSON. The
--summarize mode merges existing part JSONs into one comparison table against the
research legal reference and the continuous matched-decay Muon mature readout.

Examples:
  python eval_muon_switch_mature.py --run --arm muon20 --ckpt 80M --gpu 0 --columns BLiMP Supplement EWoK Entity
  python eval_muon_switch_mature.py --run --arm muon20 --ckpt 80M --gpu 1 --columns COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading
  python eval_muon_switch_mature.py --summarize
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval')
PART_ROOT = _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval/parts')
MERGED_ROOT = _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval/merged')
COLLATE_ROOT = _public_path('experiments/archive/frontier_consolidation/data/muon_switch_collate')

ARMS = {
    "muon20": {
        "label": "Muon20M_to_AdamW",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/muon20toadamw_seed43022_80M'),
        "base_target": "muon20toadamw_seed43022",
    },
    "muon40": {
        "label": "Muon40M_to_AdamW",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/muon40toadamw_seed43022_80M'),
        "base_target": "muon40toadamw_seed43022",
    },
}
CKPTS = {
    "70M": "chck_70M",
    "80M": "chck_80M",
}
ZERO = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GP = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
CHEAP_COMPONENTS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ALL_EVAL_COLUMNS = ZERO + GP + ["Reading"]
BASELINES = {
    "70M": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json'),
    "80M": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json'),
}
CONTINUOUS_MUON = {
    "70M": _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/per_target/muon_lr008_wd00125_70M.json'),
    "80M": _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/per_target/muon_lr008_wd00125_80M.json'),
}


def safe_suffix(cols: list[str]) -> str:
    return "cols_" + "_".join(re.sub(r"[^A-Za-z0-9]+", "", c) for c in cols)


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def base_target(arm: str, ckpt: str) -> str:
    return f"{ARMS[arm]['base_target']}_{ckpt}"


def part_target(arm: str, ckpt: str, cols: list[str]) -> str:
    return f"{base_target(arm, ckpt)}_{safe_suffix(cols)}"


def part_json_path(target: str) -> Path:
    return _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval/parts/per_target') / f"{target}.json"


def run_part(arm: str, ckpt: str, cols: list[str], gpu: int) -> Path | None:
    spec = ARMS[arm]
    run_dir = spec["run_dir"]
    endpoint = CKPTS[ckpt]
    ck_path = run_dir / "hf_model" / endpoint
    if not (ck_path / "model.safetensors").exists():
        print(f"MISSING_CHECKPOINT {arm} {ckpt}: {ck_path}", flush=True)
        return None
    tgt = part_target(arm, ckpt, cols)
    out_json = part_json_path(tgt)
    if out_json.exists():
        print(f"EXISTS {arm} {ckpt} {cols}: {out_json}", flush=True)
        return out_json
    PART_ROOT.mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-B", str(EVALUATOR), "--arm", "reinvest",
        "--run-dir", str(run_dir), "--target", tgt, "--endpoint", endpoint,
        "--out-root", str(PART_ROOT), "--collate-root", str(COLLATE_ROOT / tgt),
        "--gpu", str(gpu), "--columns", *cols,
    ]
    print(json.dumps({"event": "run_part", "arm": arm, "ckpt": ckpt, "gpu": gpu, "columns": cols, "target": tgt, "cmd": cmd}), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr, flush=True)
        raise SystemExit(proc.returncode)
    if not out_json.exists():
        raise FileNotFoundError(out_json)
    return out_json


def extract_scores(payload: dict[str, Any] | None) -> dict[str, float | None] | None:
    if payload is None:
        return None
    tasks = payload.get("tasks", {})
    out: dict[str, float | None] = {}
    for col in ZERO:
        rec = tasks.get(col, {})
        out[col] = float(rec["score"]) if rec.get("score") is not None else None
    gp_vals = []
    for col in GP:
        rec = tasks.get(col, {})
        if rec.get("score") is not None:
            gp_vals.append(float(rec["score"]))
    out["GlobalPIQA"] = mean(gp_vals) if len(gp_vals) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores: dict[str, float | None] | None) -> float | None:
    if scores is None:
        return None
    vals = [scores.get(c) for c in CHEAP_COMPONENTS]
    if any(v is None for v in vals):
        return None
    return mean(float(v) for v in vals)


def merge_parts(arm: str, ckpt: str) -> dict[str, Any]:
    merged_tasks: dict[str, Any] = {}
    files = sorted((_public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval/parts/per_target')).glob(f"{base_target(arm, ckpt)}_cols_*.json")) if (_public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval/parts/per_target')).exists() else []
    for path in files:
        payload = read_json(path)
        if not payload:
            continue
        for key, val in payload.get("tasks", {}).items():
            # Column names are unique across intended part files. If a repeated
            # part exists, later identical recomputation should not silently
            # conflict with a different score.
            if key in merged_tasks and merged_tasks[key].get("score") != val.get("score"):
                raise RuntimeError(f"conflicting score for {arm}/{ckpt}/{key}: {files}")
            merged_tasks[key] = val
    payload = {
        "target": base_target(arm, ckpt),
        "description": f"research {ARMS[arm]['label']} merged cheap-column evaluation from nonconflicting part targets",
        "family": "muon_to_adamw_switch_repair",
        "run_dir": str(ARMS[arm]["run_dir"]),
        "model_path": str(ARMS[arm]["run_dir"] / "hf_model" / CKPTS[ckpt]),
        "endpoint": CKPTS[ckpt],
        "part_files": [str(p) for p in files],
        "tasks": merged_tasks,
    }
    MERGED_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = MERGED_ROOT / f"{base_target(arm, ckpt)}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def summarize() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: dict[str, Any] = {}
    for ckpt in CKPTS:
        baseline = extract_scores(read_json(BASELINES[ckpt]))
        continuous = extract_scores(read_json(CONTINUOUS_MUON[ckpt]))
        baseline_c7 = cheap7(baseline)
        continuous_c7 = cheap7(continuous)
        for arm in ARMS:
            merged = merge_parts(arm, ckpt)
            scores = extract_scores(merged)
            c7 = cheap7(scores)
            delta_vs_base = None
            delta_vs_cont = None
            if scores and baseline:
                delta_vs_base = {c: (scores[c] - baseline[c]) if scores[c] is not None and baseline[c] is not None else None for c in CHEAP_COMPONENTS}
                delta_vs_base["cheap7"] = c7 - baseline_c7 if c7 is not None and baseline_c7 is not None else None
            if scores and continuous:
                delta_vs_cont = {c: (scores[c] - continuous[c]) if scores[c] is not None and continuous[c] is not None else None for c in CHEAP_COMPONENTS}
                delta_vs_cont["cheap7"] = c7 - continuous_c7 if c7 is not None and continuous_c7 is not None else None
            rows[f"{arm}_{ckpt}"] = {
                "arm_label": ARMS[arm]["label"],
                "ckpt": ckpt,
                "merged_path": str(MERGED_ROOT / f"{base_target(arm, ckpt)}.json"),
                "scores": scores,
                "cheap7": c7,
                "baseline_scores": baseline,
                "baseline_cheap7": baseline_c7,
                "continuous_muon_scores": continuous,
                "continuous_muon_cheap7": continuous_c7,
                "delta_vs_step35": delta_vs_base,
                "delta_vs_continuous_muon": delta_vs_cont,
                "complete_cheap7": c7 is not None,
            }
    out = {"status": "MUON_SWITCH_MATURE_MERGE", "rows": rows}
    (_public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval/muon_switch_mature_comparison.json')).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# research Muon→AdamW switch mature comparison",
        "",
        "Merged from nonconflicting part-target evaluations. Cheap7 uses BLiMP, Supplement, EWoK, Entity, COMPS, mean GlobalPIQA, Reading.",
        "",
        "| ckpt | arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δ vs research | Δ vs continuous Muon |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ckpt in CKPTS:
        # Include references once per checkpoint.
        b = extract_scores(read_json(BASELINES[ckpt])); bc7 = cheap7(b)
        c = extract_scores(read_json(CONTINUOUS_MUON[ckpt])); cc7 = cheap7(c)
        for label, sc, c7 in [("research", b, bc7), ("continuous Muon", c, cc7)]:
            if sc:
                lines.append(f"| {ckpt} | {label} | {sc['BLiMP']:.2f} | {sc['Supplement']:.2f} | {sc['EWoK']:.2f} | {sc['Entity']:.2f} | {sc['COMPS']:.2f} | {sc['GlobalPIQA']:.2f} | {sc['Reading']:.3f} | {c7:.4f} |  |  |")
        for arm in ARMS:
            r = rows[f"{arm}_{ckpt}"]
            sc = r["scores"]
            if sc and r["cheap7"] is not None:
                db = r["delta_vs_step35"]["cheap7"] if r["delta_vs_step35"] else None
                dc = r["delta_vs_continuous_muon"]["cheap7"] if r["delta_vs_continuous_muon"] else None
                lines.append(f"| {ckpt} | {r['arm_label']} | {sc['BLiMP']:.2f} | {sc['Supplement']:.2f} | {sc['EWoK']:.2f} | {sc['Entity']:.2f} | {sc['COMPS']:.2f} | {sc['GlobalPIQA']:.2f} | {sc['Reading']:.3f} | {r['cheap7']:.4f} | {db:+.4f} | {dc:+.4f} |")
            else:
                lines.append(f"| {ckpt} | {r['arm_label']} | incomplete |  |  |  |  |  |  |  |  |  |")
    (_public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval/muon_switch_mature_comparison.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--arm", choices=sorted(ARMS), default="muon20")
    ap.add_argument("--ckpt", choices=sorted(CKPTS), default="80M")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--columns", nargs="*", default=[])
    args = ap.parse_args()

    if args.run:
        cols = args.columns or ALL_EVAL_COLUMNS
        unknown = [c for c in cols if c not in ALL_EVAL_COLUMNS]
        if unknown:
            raise SystemExit(f"Unknown columns: {unknown}")
        run_part(args.arm, args.ckpt, cols, args.gpu)
    if args.summarize:
        summarize()
    if not args.run and not args.summarize:
        ap.error("choose --run and/or --summarize")


if __name__ == "__main__":
    main()
