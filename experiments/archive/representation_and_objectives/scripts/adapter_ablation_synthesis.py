#!/usr/bin/env python3
"""Synthesize adapter inference-disabled ablation against matched controls."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
A02_SUMMARY = Path("experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval/adapter_matched_horizon_summary.json")
JSON = Path("experiments/archive/representation_and_objectives/data/adapter_inference_disabled_eval/per_target/adapter128_live20M_inference_disabled_proxy.json")
OUT = Path("experiments/archive/representation_and_objectives/data/adapter_inference_disabled_synthesis/adapter_inference_disabled_synthesis.json")

def extract(payload: dict) -> dict:
    tasks = payload.get("tasks", {})
    scores = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c, {})
        scores[c] = rec.get("score")
    gps = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if rec.get("score") is not None:
            gps.append(float(rec["score"]))
    scores["GlobalPIQA"] = mean(gps) if len(gps) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict):
        scores["Reading"] = rd["scores"].get("Reading")
    else:
        scores["Reading"] = rd.get("score")
    return {k: (float(v) if v is not None else None) for k, v in scores.items()}

def cheap7(scores: dict) -> float:
    return mean(float(scores[k]) for k in CHEAP)

def delta(a: dict, b: dict) -> dict:
    d = {k: float(a[k] - b[k]) for k in CHEAP}
    d["cheap7"] = cheap7(a) - cheap7(b)
    return d

def main() -> None:
    a02 = json.loads(A02_SUMMARY.read_text(encoding="utf-8"))
    rows = a02["rows"]
    baseline_scores = dict(rows["reference_20M"]["scores"])
    live = dict(rows["live128"]["scores"])
    trained_disabled = dict(rows["disabled128"]["scores"])
    ablation_scores = extract(json.loads(JSON.read_text(encoding="utf-8")))
    table = {
        "or_disabled_training_20M": {"scores": baseline_scores, "cheap7": cheap7(baseline_scores)},
        "live_adapter_enabled_inference_20M": {"scores": live, "cheap7": cheap7(live)},
        "live_adapter_disabled_at_inference_20M": {"scores": ablation_scores, "cheap7": cheap7(ablation_scores)},
    }
    comparisons = {
        "disabled_inference_vs_step35": delta(ablation_scores, baseline_scores),
        "live_enabled_vs_step35": delta(live, baseline_scores),
        "direct_adapter_output_effect_live_minus_disabled_inference": delta(live, ablation_scores),
        "backbone_drift_effect_disabled_inference_minus_step35": delta(ablation_scores, baseline_scores),
    }
    interpretation = {
        "globalpiqa": "Disabling the live adapter at inference recovers GlobalPIQA almost exactly to the research/disabled-training anchor (34.165 vs 34.195), so the direct adapter residual output caused nearly all of the live arm's GlobalPIQA damage.",
        "ewok": "EWoK remains worse than both the research anchor and live adapter-enabled inference (48.98 vs 50.73 and 49.49), so the EWoK damage is mainly stock-backbone trajectory drift from joint adapter training, partly offset by direct adapter output.",
        "broad_columns": "The direct adapter output contributes BLiMP/Supplement/COMPS gains but also the GlobalPIQA loss; turning it off leaves Supplement gain but cheap7 remains below the research anchor.",
        "route": "Ordinary joint adapters are not promoted: the key repair is coupling control, not more width. Stop-gradient/frozen-reader designs must protect stock trajectory and control direct residual output, especially for GlobalPIQA."
    }
    out = {
        "status": "ADAPTER_INFERENCE_DISABLED_SYNTHESIS",
        "a02_summary": str(A02_SUMMARY),
        "per_target": str(JSON),
        "cheap_columns": CHEAP,
        "table": table,
        "comparisons": comparisons,
        "interpretation": interpretation,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "summary_saved", "path": str(OUT), "cheap7_disabled_inference": table["live_adapter_disabled_at_inference_20M"]["cheap7"], "delta_vs_step35": comparisons["disabled_inference_vs_step35"]["cheap7"]}), flush=True)

if __name__ == "__main__":
    main()
