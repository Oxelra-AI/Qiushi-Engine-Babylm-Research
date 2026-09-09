#!/usr/bin/env python3
"""Merge the actual research switch-evaluation part outputs.

The research part evals intentionally used separate out-root directories to avoid
concurrent writes. This script reads those per-target JSONs, validates expected
columns, merges them into complete cheap-column payloads for each arm/checkpoint,
and compares against research and continuous matched-decay Muon references.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
PART_BASE = _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval_parts')
OUT = _public_path('experiments/archive/frontier_consolidation/data/muon_switch_actual_merged')

PARTS = {
    "muon20_70M": {
        "heavy": _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval_parts/muon20_70M_heavy/per_target/muon20toadamw_70M_heavy.json'),
        "light": _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval_parts/muon20_70M_light/per_target/muon20toadamw_70M_light.json'),
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/muon20toadamw_seed43022_80M'),
        "endpoint": "chck_70M",
        "label": "Muon20M_to_AdamW",
        "ckpt": "70M",
    },
    "muon20_80M": {
        "heavy": _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval_parts/muon20_80M_heavy/per_target/muon20toadamw_80M_heavy.json'),
        "light": _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval_parts/muon20_80M_light/per_target/muon20toadamw_80M_light.json'),
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/muon20toadamw_seed43022_80M'),
        "endpoint": "chck_80M",
        "label": "Muon20M_to_AdamW",
        "ckpt": "80M",
    },
    "muon40_70M": {
        "heavy": _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval_parts/muon40_70M_heavy/per_target/muon40toadamw_70M_heavy.json'),
        "light": _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval_parts/muon40_70M_light/per_target/muon40toadamw_70M_light.json'),
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/muon40toadamw_seed43022_80M'),
        "endpoint": "chck_70M",
        "label": "Muon40M_to_AdamW",
        "ckpt": "70M",
    },
    "muon40_80M": {
        "heavy": _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval_parts/muon40_80M_heavy/per_target/muon40toadamw_80M_heavy.json'),
        "light": _public_path('experiments/archive/frontier_consolidation/data/muon_switch_eval_parts/muon40_80M_light/per_target/muon40toadamw_80M_light.json'),
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/muon40toadamw_seed43022_80M'),
        "endpoint": "chck_80M",
        "label": "Muon40M_to_AdamW",
        "ckpt": "80M",
    },
}

EXPECTED_HEAVY = {"BLiMP", "Entity"}
EXPECTED_LIGHT = {"Supplement", "EWoK", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"}
EXPECTED_ALL = EXPECTED_HEAVY | EXPECTED_LIGHT
ZERO = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GP = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

BASELINES = {
    "70M": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json'),
    "80M": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json'),
}
CONTINUOUS = {
    "70M": _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/per_target/muon_lr008_wd00125_70M.json'),
    "80M": _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/per_target/muon_lr008_wd00125_80M.json'),
}


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def task_score(tasks: dict[str, Any], key: str) -> float | None:
    rec = tasks.get(key, {})
    if key == "Reading":
        if isinstance(rec.get("scores"), dict) and rec["scores"].get("Reading") is not None:
            return float(rec["scores"]["Reading"])
    if rec.get("score") is not None:
        return float(rec["score"])
    return None


def extract_scores(payload: dict[str, Any] | None) -> dict[str, float | None] | None:
    if payload is None:
        return None
    tasks = payload.get("tasks", {})
    out: dict[str, float | None] = {}
    for c in ZERO:
        out[c] = task_score(tasks, c)
    gp_vals = [task_score(tasks, c) for c in GP]
    out["GlobalPIQA"] = mean([float(v) for v in gp_vals]) if all(v is not None for v in gp_vals) else None
    out["Reading"] = task_score(tasks, "Reading")
    return out


def cheap7(sc: dict[str, float | None] | None) -> float | None:
    if sc is None:
        return None
    vals = [sc.get(c) for c in CHEAP]
    if any(v is None for v in vals):
        return None
    return mean(float(v) for v in vals)


def merge_one(key: str, spec: dict[str, Any]) -> dict[str, Any]:
    missing_files = [name for name in ["heavy", "light"] if not spec[name].exists()]
    tasks: dict[str, Any] = {}
    present_by_file = {}
    for name in ["heavy", "light"]:
        payload = read_json(spec[name])
        present = set(payload.get("tasks", {}).keys()) if payload else set()
        present_by_file[name] = sorted(present)
        if payload:
            for col, rec in payload.get("tasks", {}).items():
                if col in tasks:
                    raise RuntimeError(f"duplicate column {col} in {key}")
                tasks[col] = rec
    missing_cols = sorted(EXPECTED_ALL - set(tasks))
    extra_cols = sorted(set(tasks) - EXPECTED_ALL)
    heavy_bad = sorted((set(present_by_file.get("heavy", [])) - EXPECTED_HEAVY) | (EXPECTED_HEAVY - set(present_by_file.get("heavy", [])))) if spec["heavy"].exists() else sorted(EXPECTED_HEAVY)
    light_bad = sorted((set(present_by_file.get("light", [])) - EXPECTED_LIGHT) | (EXPECTED_LIGHT - set(present_by_file.get("light", [])))) if spec["light"].exists() else sorted(EXPECTED_LIGHT)
    payload = {
        "target": f"step098_{key}_merged",
        "description": "research Muon-to-AdamW switch merged from actual nonconflicting heavy/light eval roots",
        "family": "muon_to_adamw_switch_repair",
        "run_dir": str(spec["run_dir"]),
        "model_path": str(spec["run_dir"] / "hf_model" / spec["endpoint"]),
        "endpoint": spec["endpoint"],
        "part_files": {"heavy": str(spec["heavy"]), "light": str(spec["light"])},
        "missing_files": missing_files,
        "present_by_file": present_by_file,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "part_assignment_mismatches": {"heavy_symmetric_diff": heavy_bad, "light_symmetric_diff": light_bad},
        "tasks": tasks,
    }
    (_public_path('experiments/archive/frontier_consolidation/data/muon_switch_actual_merged/per_target')).mkdir(parents=True, exist_ok=True)
    (_public_path('experiments/archive/frontier_consolidation/data/muon_switch_actual_merged/per_target') / f"step098_{key}_merged.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for key, spec in PARTS.items():
        merged = merge_one(key, spec)
        ckpt = spec["ckpt"]
        scores = extract_scores(merged)
        c7 = cheap7(scores)
        base = extract_scores(read_json(BASELINES[ckpt]))
        base_c7 = cheap7(base)
        cont = extract_scores(read_json(CONTINUOUS[ckpt]))
        cont_c7 = cheap7(cont)
        delta_base = None
        delta_cont = None
        if scores and base:
            delta_base = {c: (scores[c] - base[c]) if scores[c] is not None and base[c] is not None else None for c in CHEAP}
            delta_base["cheap7"] = c7 - base_c7 if c7 is not None and base_c7 is not None else None
        if scores and cont:
            delta_cont = {c: (scores[c] - cont[c]) if scores[c] is not None and cont[c] is not None else None for c in CHEAP}
            delta_cont["cheap7"] = c7 - cont_c7 if c7 is not None and cont_c7 is not None else None
        rows[key] = {
            "label": spec["label"],
            "ckpt": ckpt,
            "merged_path": str(_public_path('experiments/archive/frontier_consolidation/data/muon_switch_actual_merged/per_target') / f"step098_{key}_merged.json"),
            "scores": scores,
            "cheap7": c7,
            "baseline_scores": base,
            "baseline_cheap7": base_c7,
            "continuous_muon_scores": cont,
            "continuous_muon_cheap7": cont_c7,
            "delta_vs_step35": delta_base,
            "delta_vs_continuous_muon": delta_cont,
            "complete": c7 is not None and not merged["missing_files"] and not merged["missing_columns"],
            "missing_files": merged["missing_files"],
            "missing_columns": merged["missing_columns"],
            "part_assignment_mismatches": merged["part_assignment_mismatches"],
        }
    result = {"status": "ACTUAL_SWITCH_EVAL_MERGED", "rows": rows}
    (_public_path('experiments/archive/frontier_consolidation/data/muon_switch_actual_merged/actual_switch_eval_merged.json')).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# research actual Muon→AdamW switch eval merge",
        "",
        "Rows are complete only when both heavy and light part outputs exist with all cheap columns.",
        "",
        "| ckpt | arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δ research | Δ cont.Muon | complete |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for ckpt in ["70M", "80M"]:
        b = extract_scores(read_json(BASELINES[ckpt])); bc7 = cheap7(b)
        c = extract_scores(read_json(CONTINUOUS[ckpt])); cc7 = cheap7(c)
        for lab, sc, c7 in [("research", b, bc7), ("continuous Muon", c, cc7)]:
            if sc:
                lines.append(f"| {ckpt} | {lab} | {sc['BLiMP']:.2f} | {sc['Supplement']:.2f} | {sc['EWoK']:.2f} | {sc['Entity']:.2f} | {sc['COMPS']:.2f} | {sc['GlobalPIQA']:.2f} | {sc['Reading']:.3f} | {c7:.4f} |  |  | ref |")
        for key in [f"muon20_{ckpt}", f"muon40_{ckpt}"]:
            r = rows[key]
            sc = r["scores"] or {}
            if r["cheap7"] is None:
                lines.append(f"| {ckpt} | {r['label']} | incomplete |  |  |  |  |  |  |  |  |  | {r['complete']} |")
            else:
                db = r["delta_vs_step35"]["cheap7"] if r["delta_vs_step35"] else None
                dc = r["delta_vs_continuous_muon"]["cheap7"] if r["delta_vs_continuous_muon"] else None
                lines.append(f"| {ckpt} | {r['label']} | {sc['BLiMP']:.2f} | {sc['Supplement']:.2f} | {sc['EWoK']:.2f} | {sc['Entity']:.2f} | {sc['COMPS']:.2f} | {sc['GlobalPIQA']:.2f} | {sc['Reading']:.3f} | {r['cheap7']:.4f} | {db:+.4f} | {dc:+.4f} | {r['complete']} |")
    (_public_path('research/documents/frontier_consolidation/data/muon_switch_actual_merged/actual_switch_eval_merged.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
