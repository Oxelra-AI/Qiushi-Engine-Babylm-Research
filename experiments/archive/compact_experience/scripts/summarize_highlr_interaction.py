#!/usr/bin/env python3
"""research: summarize high-LR optimizer × paired-data interaction from no-AoA eval.

This script reads per-target JSON payloads produced by
scripts/eval_highlr_10M_interaction.sh and computes the scientific object
needed for route judgment:

  data effect under AdamW:   Qwen - Official
  data effect under LAMB:    Qwen - Official
  optimizer×data interaction: (LAMB_Qwen - LAMB_Official) - (AdamW_Qwen - AdamW_Official)

It intentionally excludes SuperGLUE and AoA; this is a compact screen only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
CHECKPOINTS = ["chck_3M", "chck_5M", "chck_10M"]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": repr(exc), "_path": str(path)}


def extract_scores(payload: dict[str, Any]) -> dict[str, float]:
    """Extract official-overall merged scores or reconstruct them from tasks."""
    out: dict[str, float] = {}
    overall = payload.get("official_overall") if isinstance(payload, dict) else None
    if isinstance(overall, dict) and isinstance(overall.get("scores"), dict):
        for c in COLUMNS:
            v = overall["scores"].get(c)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out[c] = float(v)
    if len(out) == len(COLUMNS):
        return out

    tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
    if isinstance(tasks, dict):
        for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]:
            rec = tasks.get(c)
            if isinstance(rec, dict):
                v = rec.get("score")
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    out[c] = float(v)
        p = tasks.get("GlobalPIQA_parallel", {}).get("score") if isinstance(tasks.get("GlobalPIQA_parallel"), dict) else None
        n = tasks.get("GlobalPIQA_nonparallel", {}).get("score") if isinstance(tasks.get("GlobalPIQA_nonparallel"), dict) else None
        if isinstance(p, (int, float)) and isinstance(n, (int, float)) and math.isfinite(float(p)) and math.isfinite(float(n)):
            out["GlobalPIQA"] = (float(p) + float(n)) / 2.0
    return out


def equal7(scores: dict[str, float]) -> float | None:
    vals = [scores.get(c) for c in COLUMNS]
    if any(v is None for v in vals):
        return None
    return mean(float(v) for v in vals if v is not None)


def rnd(x: Any, nd: int = 4) -> Any:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return round(float(x), nd)
    return x


def sign_structure(cols: dict[str, float]) -> dict[str, Any]:
    vals = [v for v in cols.values() if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals:
        return {"n_positive": 0, "n_negative": 0, "max_abs_column": None, "dominance_fraction": None}
    abs_vals = {k: abs(v) for k, v in cols.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}
    max_col = max(abs_vals, key=abs_vals.get)
    abs_sum = sum(abs_vals.values())
    return {
        "n_positive": sum(1 for v in vals if v > 0),
        "n_negative": sum(1 for v in vals if v < 0),
        "max_abs_column": max_col,
        "max_abs_value": rnd(cols[max_col]),
        "dominance_fraction": rnd(abs_vals[max_col] / abs_sum if abs_sum else None),
    }


def main() -> None:
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/optimizer_10M_noaoa_eval_lr0.007")
    lr = sys.argv[2] if len(sys.argv) > 2 else "0.007"
    per_target = out_root / "per_target"

    base_names = {
        "adamw_official": f"adamw_lr{lr}_official_10M_seed43022",
        "lamb_official": f"lamb_lr{lr}_official_10M_seed43022",
        "adamw_qwen": f"adamw_lr{lr}_qwen_10M_seed43022",
        "lamb_qwen": f"lamb_lr{lr}_qwen_10M_seed43022",
    }

    results: dict[str, Any] = {}
    interactions: dict[str, Any] = {}

    for ckpt in CHECKPOINTS:
        results[ckpt] = {}
        for key, base in base_names.items():
            target = f"{base}_{ckpt}"
            payload_path = per_target / f"{target}.json"
            payload = load_json(payload_path)
            if payload is None:
                results[ckpt][key] = {"status": "missing", "target": target, "payload_path": str(payload_path)}
                continue
            if "_load_error" in payload:
                results[ckpt][key] = {"status": "load_error", "target": target, "error": payload["_load_error"], "payload_path": str(payload_path)}
                continue
            scores = extract_scores(payload)
            eq = equal7(scores)
            results[ckpt][key] = {
                "status": "complete" if eq is not None else "incomplete_scores",
                "target": target,
                "payload_path": str(payload_path),
                "scores": {c: scores.get(c) for c in COLUMNS},
                "equal7_noaoa": eq,
                "run_summary": payload.get("run_summary", {}),
            }

        r = results[ckpt]
        needed = ["adamw_official", "adamw_qwen", "lamb_official", "lamb_qwen"]
        if any(r[k].get("status") != "complete" for k in needed):
            interactions[ckpt] = {"status": "incomplete", "available": {k: r[k].get("status") for k in needed}}
            continue

        ao = r["adamw_official"]["scores"]
        aq = r["adamw_qwen"]["scores"]
        lo = r["lamb_official"]["scores"]
        lq = r["lamb_qwen"]["scores"]

        adamw_data_raw = {c: aq[c] - ao[c] for c in COLUMNS}
        lamb_data_raw = {c: lq[c] - lo[c] for c in COLUMNS}
        interaction_raw = {c: lamb_data_raw[c] - adamw_data_raw[c] for c in COLUMNS}
        optimizer_official_raw = {c: lo[c] - ao[c] for c in COLUMNS}
        optimizer_qwen_raw = {c: lq[c] - aq[c] for c in COLUMNS}

        adamw_data = {c: rnd(v) for c, v in adamw_data_raw.items()}
        lamb_data = {c: rnd(v) for c, v in lamb_data_raw.items()}
        interaction = {c: rnd(v) for c, v in interaction_raw.items()}
        optimizer_official = {c: rnd(v) for c, v in optimizer_official_raw.items()}
        optimizer_qwen = {c: rnd(v) for c, v in optimizer_qwen_raw.items()}

        ae = r["adamw_qwen"]["equal7_noaoa"] - r["adamw_official"]["equal7_noaoa"]
        le = r["lamb_qwen"]["equal7_noaoa"] - r["lamb_official"]["equal7_noaoa"]
        ie = le - ae
        interaction_structure = sign_structure(interaction_raw)

        # Interpretation support: broad if interaction is not just a one-column swing.
        interactions[ckpt] = {
            "status": "complete",
            "adamw_data_effect": adamw_data,
            "adamw_equal7_data_effect": rnd(ae),
            "lamb_data_effect": lamb_data,
            "lamb_equal7_data_effect": rnd(le),
            "interaction_columns": interaction,
            "interaction_equal7": rnd(ie),
            "interaction_sign_structure": interaction_structure,
            "optimizer_effect_official_LAMB_minus_AdamW": optimizer_official,
            "optimizer_effect_qwen_LAMB_minus_AdamW": optimizer_qwen,
            "optimizer_main_effect_equal7_official": rnd(r["lamb_official"]["equal7_noaoa"] - r["adamw_official"]["equal7_noaoa"]),
            "optimizer_main_effect_equal7_qwen": rnd(r["lamb_qwen"]["equal7_noaoa"] - r["adamw_qwen"]["equal7_noaoa"]),
            "collapse_flags": {
                "lamb_qwen_minus_adamw_qwen_Supplement_lt_minus1": bool((lq["Supplement"] - aq["Supplement"]) < -1.0),
                "lamb_qwen_minus_adamw_qwen_Reading_lt_minus0p5": bool((lq["Reading"] - aq["Reading"]) < -0.5),
                "interaction_dominated_by_single_column_gt0p55": bool(interaction_structure.get("dominance_fraction") is not None and interaction_structure["dominance_fraction"] > 0.55),
            },
        }

    # Trajectory view for the equal7 interaction.
    traj = []
    for ckpt in CHECKPOINTS:
        rec = interactions.get(ckpt, {})
        if rec.get("status") == "complete":
            traj.append({
                "checkpoint": ckpt,
                "interaction_equal7": rec["interaction_equal7"],
                "adamw_data_effect": rec["adamw_equal7_data_effect"],
                "lamb_data_effect": rec["lamb_equal7_data_effect"],
                "n_positive_interaction_cols": rec["interaction_sign_structure"]["n_positive"],
                "max_abs_column": rec["interaction_sign_structure"]["max_abs_column"],
                "dominance_fraction": rec["interaction_sign_structure"]["dominance_fraction"],
            })

    summary = {
        "status": "HIGH_LR_OPTIMIZER_10M_INTERACTION_SUMMARY",
        "learning_rate": float(lr),
        "out_root": str(out_root),
        "columns": COLUMNS,
        "checkpoints": CHECKPOINTS,
        "results_by_checkpoint": results,
        "interactions_by_checkpoint": interactions,
        "interaction_trajectory": traj,
        "scientific_question": "Does LAMB at matched high LR amplify the clean-Qwen same-window paired-data effect relative to AdamW, rather than merely improving both corpora generically?",
        "readout": {
            "promising_pattern": "positive and persistent optimizer×data interaction across several capability columns, especially not carried solely by GlobalPIQA, with no Supplement/Reading collapse",
            "negative_pattern": "interaction absent, negative, one-column-dominated, or just a generic LAMB main effect present on official and Qwen alike",
            "limits": "No SuperGLUE or AoA are measured here; any 100M candidate still requires complete nine-column evaluation and exact official-compatible checkpoint ladder.",
        },
    }
    out_path = out_root / "highlr_10M_interaction_summary.json"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
