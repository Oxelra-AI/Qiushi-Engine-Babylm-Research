#!/usr/bin/env python3
"""research: zero-shot/Reading direct profile for exact `(M,S)` and clean preservation.

The exact acquisition-only `(M,S)` official zero-shot/Reading payload has now
completed.  This script reuses the validated research parser, adds `(M,S)`, and
decomposes direct clean-vs-`(M,S)` component and item movements for the zero-shot
columns plus Reading vector correlations.  It complements the final same-coordinate
Overall table and the all-completed SuperGLUE profile.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import importlib.util
import json
import math
import pathlib
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

ROOT = _public_path('.')
research = _public_path('experiments/archive/functional_learning/scripts/zero_reading_item_stability.py')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/zero_reading_ms_direct_profile')

MODEL_PAYLOADS: Dict[str, str] = {
    "coherent86": "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json",
    "dense_seed62064_MM": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json",
    "dense_seed62065_MM": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json",
    "ms_acquisition_seed62064_MS": "experiments/archive/functional_learning/data/repaired_densemask_sparselabel_zero_reading/per_target/densemask_sparselabel_seed62064_u0080_zero_reading.json",
    "clean_pres_seed62064_MSplusKL": "experiments/archive/functional_learning/data/repaired_clean_eval_zero_reading/per_target/clean_pres_lambda1_eval_seed62064_u0080_zero_reading.json",
    "clean_pres_seed62065_MSplusKL": "experiments/archive/functional_learning/data/repaired_clean_seed62065_zero_reading/per_target/clean_pres_lambda1_eval_seed62065_u0080_zero_reading.json",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str | None) -> Optional[str]:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_step097_module():
    spec = importlib.util.spec_from_file_location("zero_reading_item_stability", research)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {research}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def payload_zero_scores(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    tasks = payload.get("tasks") or {}
    out: Dict[str, Optional[float]] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        entry = tasks.get(col) or {}
        out[col] = entry.get("score")
    if "Reading" in tasks:
        out["Reading"] = (tasks.get("Reading") or {}).get("scores", {}).get("Reading")
    if out.get("GlobalPIQA_parallel") is not None and out.get("GlobalPIQA_nonparallel") is not None:
        out["GlobalPIQA"] = (float(out["GlobalPIQA_parallel"]) + float(out["GlobalPIQA_nonparallel"])) / 2.0
    else:
        out["GlobalPIQA"] = None
    return out


def component_score_from_summary(summary: Dict[str, Any], col: str) -> Optional[float]:
    if col == "GlobalPIQA":
        a = summary.get("GlobalPIQA_parallel", {}).get("official_like_score")
        b = summary.get("GlobalPIQA_nonparallel", {}).get("official_like_score")
        return (a + b) / 2.0 if a is not None and b is not None else None
    if col == "Reading":
        return None
    return summary.get(col, {}).get("official_like_score")


def decomp(parent: Dict[str, Dict[str, Any]], acq: Dict[str, Dict[str, Any]], clean: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted(set(parent) & set(acq) & set(clean))
    out: Dict[str, Any] = defaultdict(int)
    for k in keys:
        p = bool(parent[k]["correct"])
        a = bool(acq[k]["correct"])
        c = bool(clean[k]["correct"])
        if p and (not a) and c:
            out["clean_recovers_acquisition_parent_loss"] += 1
        if p and a and (not c):
            out["clean_loses_parent_item_acquisition_kept"] += 1
        if (not p) and a and c:
            out["clean_keeps_acquisition_gain"] += 1
        if (not p) and a and (not c):
            out["clean_drops_acquisition_gain"] += 1
        if (not p) and (not a) and c:
            out["clean_new_gain_beyond_acquisition"] += 1
        if p and (not a) and (not c):
            out["shared_loss_vs_parent"] += 1
    n = len(keys)
    out["n"] = n
    out["net_clean_minus_acquisition_accuracy_points"] = 100.0 * (
        (out["clean_recovers_acquisition_parent_loss"] + out["clean_new_gain_beyond_acquisition"])
        - (out["clean_loses_parent_item_acquisition_kept"] + out["clean_drops_acquisition_gain"])
    ) / n if n else None
    return dict(out)


def sample_direct(parent: Dict[str, Dict[str, Any]], acq: Dict[str, Dict[str, Any]], clean: Dict[str, Dict[str, Any]], limit: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    keys = sorted(set(parent) & set(acq) & set(clean))
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for k in keys:
        p = bool(parent[k]["correct"])
        a = bool(acq[k]["correct"])
        c = bool(clean[k]["correct"])
        label = None
        if p and (not a) and c:
            label = "clean_recovers_acquisition_parent_loss"
        elif p and a and (not c):
            label = "clean_loses_parent_item_acquisition_kept"
        elif (not p) and a and c:
            label = "clean_keeps_acquisition_gain"
        elif (not p) and a and (not c):
            label = "clean_drops_acquisition_gain"
        elif (not p) and (not a) and c:
            label = "clean_new_gain_beyond_acquisition"
        elif p and (not a) and (not c):
            label = "shared_loss_vs_parent"
        if label and len(buckets[label]) < limit:
            buckets[label].append({
                "key": k,
                "label": parent[k].get("label"),
                "parent_pred": parent[k].get("pred"),
                "acquisition_pred": acq[k].get("pred"),
                "clean_pred": clean[k].get("pred"),
                "parent_correct": p,
                "acquisition_correct": a,
                "clean_correct": c,
                "subtask": parent[k].get("subtask"),
                "meta": parent[k].get("meta"),
            })
    return dict(buckets)


def build() -> Dict[str, Any]:
    mod = load_step097_module()
    payloads = {m: mod.load_json_file(p) for m, p in MODEL_PAYLOADS.items()}
    records: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    summaries: Dict[str, Any] = {}
    reading: Dict[str, Any] = {}

    for m, payload in payloads.items():
        model_summary: Dict[str, Any] = {"payload": rel(MODEL_PAYLOADS[m]), "payload_scores": payload_zero_scores(payload), "columns": {}}
        for col in mod.ZERO_COLUMNS:
            task = payload["tasks"][col]
            recs = mod.parse_zero_column(col, task)
            records[m][col] = recs
            model_summary["columns"][col] = mod.summarize_model_column(recs, col, (task or {}).get("score"))
            model_summary["columns"][col]["by_group_top"] = dict(list(mod.rates_by_group(recs).items())[:20])
        task_read = payload["tasks"]["Reading"]
        reading[m] = mod.parse_reading(task_read)
        model_summary["reading"] = reading[m]
        summaries[m] = model_summary

    # Component deltas in official component units. GlobalPIQA uses the official mean of two subcolumns.
    component_rows: List[Dict[str, Any]] = []
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]:
        row: Dict[str, Any] = {"component": col}
        for m in MODEL_PAYLOADS:
            if col == "Reading":
                score = summaries[m]["payload_scores"]["Reading"]
            elif col == "GlobalPIQA":
                score = summaries[m]["payload_scores"]["GlobalPIQA"]
            else:
                score = summaries[m]["columns"][col]["payload_score"]
            row[f"{m}_score"] = score
        for m in MODEL_PAYLOADS:
            if m != "coherent86":
                row[f"{m}_minus_coherent86"] = row[f"{m}_score"] - row["coherent86_score"]
        row["clean64_minus_ms_acquisition"] = row["clean_pres_seed62064_MSplusKL_score"] - row["ms_acquisition_seed62064_MS_score"]
        row["clean65_minus_ms_acquisition"] = row["clean_pres_seed62065_MSplusKL_score"] - row["ms_acquisition_seed62064_MS_score"]
        row["clean65_minus_clean64"] = row["clean_pres_seed62065_MSplusKL_score"] - row["clean_pres_seed62064_MSplusKL_score"]
        component_rows.append(row)

    direct_item_comparisons: Dict[str, Any] = {}
    for col in mod.ZERO_COLUMNS:
        direct_item_comparisons[col] = {
            "clean64_vs_ms_acquisition": mod.compare_two(records["clean_pres_seed62064_MSplusKL"][col], records["ms_acquisition_seed62064_MS"][col]),
            "clean65_vs_ms_acquisition": mod.compare_two(records["clean_pres_seed62065_MSplusKL"][col], records["ms_acquisition_seed62064_MS"][col]),
            "clean64_vs_clean65": mod.compare_two(records["clean_pres_seed62064_MSplusKL"][col], records["clean_pres_seed62065_MSplusKL"][col]),
            "parent_ms_clean64_decomposition": decomp(records["coherent86"][col], records["ms_acquisition_seed62064_MS"][col], records["clean_pres_seed62064_MSplusKL"][col]),
            "parent_ms_clean65_decomposition": decomp(records["coherent86"][col], records["ms_acquisition_seed62064_MS"][col], records["clean_pres_seed62065_MSplusKL"][col]),
        }

    reading_comparisons: Dict[str, Any] = {}
    for a, b in [("clean_pres_seed62064_MSplusKL", "ms_acquisition_seed62064_MS"), ("clean_pres_seed62065_MSplusKL", "ms_acquisition_seed62064_MS"), ("clean_pres_seed62064_MSplusKL", "clean_pres_seed62065_MSplusKL")]:
        score_a = (reading[a].get("scores") or {}).get("Reading")
        score_b = (reading[b].get("scores") or {}).get("Reading")
        pred_pairs = [(float(x), float(y)) for x, y in zip(reading[a].get("pred", []), reading[b].get("pred", [])) if math.isfinite(float(x)) and math.isfinite(float(y))]
        prev_pairs = [(float(x), float(y)) for x, y in zip(reading[a].get("prev_pred", []), reading[b].get("prev_pred", [])) if math.isfinite(float(x)) and math.isfinite(float(y))]
        reading_comparisons[f"{a}_vs_{b}"] = {
            "n": min(reading[a].get("n", 0), reading[b].get("n", 0)),
            "score_a": score_a,
            "score_b": score_b,
            "score_delta_a_minus_b": (score_a - score_b) if score_a is not None and score_b is not None else None,
            "pred_pearson": mod.pearson([x for x, _ in pred_pairs], [y for _, y in pred_pairs]),
            "prev_pred_pearson": mod.pearson([x for x, _ in prev_pairs], [y for _, y in prev_pairs]),
            "pred_mean_signed_diff_a_minus_b": sum((x - y) for x, y in pred_pairs) / len(pred_pairs) if pred_pairs else None,
            "pred_mean_abs_diff": sum(abs(x - y) for x, y in pred_pairs) / len(pred_pairs) if pred_pairs else None,
            "prev_pred_mean_signed_diff_a_minus_b": sum((x - y) for x, y in prev_pairs) / len(prev_pairs) if prev_pairs else None,
            "prev_pred_mean_abs_diff": sum(abs(x - y) for x, y in prev_pairs) / len(prev_pairs) if prev_pairs else None,
        }

    result = {
        "status": "ZERO_READING_MS_DIRECT_PROFILE",
        "created_utc": now(),
        "purpose": "Direct official zero-shot/Reading item and component comparison of clean preservation versus exact acquisition-only `(M,S)`.",
        "parser_source": rel(research),
        "payloads": MODEL_PAYLOADS,
        "model_summaries": summaries,
        "component_deltas": component_rows,
        "direct_item_comparisons": direct_item_comparisons,
        "reading_comparisons": reading_comparisons,
        "flip_samples_clean64_vs_ms": {col: sample_direct(records["coherent86"][col], records["ms_acquisition_seed62064_MS"][col], records["clean_pres_seed62064_MSplusKL"][col]) for col in mod.ZERO_COLUMNS},
        "interpretation": {
            "zero_reading_sum": {m: sum(float(row[f"{m}_score"]) for row in component_rows) for m in MODEL_PAYLOADS},
            "clean64_minus_ms_zero_reading_sum": sum(float(row["clean64_minus_ms_acquisition"]) for row in component_rows),
            "clean65_minus_ms_zero_reading_sum": sum(float(row["clean65_minus_ms_acquisition"]) for row in component_rows),
            "clean64_minus_ms_zero_reading_overall_units": sum(float(row["clean64_minus_ms_acquisition"]) for row in component_rows) / 9.0,
            "clean65_minus_ms_zero_reading_overall_units": sum(float(row["clean65_minus_ms_acquisition"]) for row in component_rows) / 9.0,
        },
    }
    return result


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research direct zero-shot/Reading profile: clean preservation vs exact `(M,S)`\n\n")
    lines.append("This parses actual official zero-shot/Reading payloads. It includes coherent86, dense `(M,M)` seeds, exact acquisition-only `(M,S)` seed62064, and clean preservation seeds 62064/62065.\n\n")
    interp = result["interpretation"]
    lines.append("## Seven-component zero/Reading sums\n\n")
    for m, v in interp["zero_reading_sum"].items():
        lines.append(f"- {m}: {fmt(v,6)}\n")
    lines.append(f"- clean64 minus exact `(M,S)` zero/Reading sum: {fmt(interp['clean64_minus_ms_zero_reading_sum'],6)} = {fmt(interp['clean64_minus_ms_zero_reading_overall_units'],6)} Overall units before SuperGLUE/AoA\n")
    lines.append(f"- clean65 minus exact `(M,S)` zero/Reading sum: {fmt(interp['clean65_minus_ms_zero_reading_sum'],6)} = {fmt(interp['clean65_minus_ms_zero_reading_overall_units'],6)} Overall units before SuperGLUE/AoA\n\n")

    lines.append("## Component deltas\n\n")
    lines.append("| component | coherent86 | exact `(M,S)` | clean64 | clean65 | clean64-`(M,S)` | clean65-`(M,S)` | clean65-clean64 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in result["component_deltas"]:
        lines.append(f"| {row['component']} | {fmt(row['coherent86_score'])} | {fmt(row['ms_acquisition_seed62064_MS_score'])} | {fmt(row['clean_pres_seed62064_MSplusKL_score'])} | {fmt(row['clean_pres_seed62065_MSplusKL_score'])} | {fmt(row['clean64_minus_ms_acquisition'])} | {fmt(row['clean65_minus_ms_acquisition'])} | {fmt(row['clean65_minus_clean64'])} |\n")

    lines.append("\n## Item-level direct comparisons for zero-shot columns\n\n")
    lines.append("| column | clean64-vs-`(M,S)` correctness agree | clean64 correct/`(M,S)` wrong | `(M,S)` correct/clean64 wrong | net clean64-`(M,S)` pp | clean65-vs-`(M,S)` correctness agree | clean65 correct/`(M,S)` wrong | `(M,S)` correct/clean65 wrong | net clean65-`(M,S)` pp |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for col, comp in result["direct_item_comparisons"].items():
        c64 = comp["clean64_vs_ms_acquisition"]
        c65 = comp["clean65_vs_ms_acquisition"]
        c64_net = c64.get('net_item_accuracy_points_a_minus_b', c64.get('net_accuracy_points_a_minus_b'))
        c65_net = c65.get('net_item_accuracy_points_a_minus_b', c65.get('net_accuracy_points_a_minus_b'))
        lines.append(f"| {col} | {fmt(c64.get('correctness_agreement_fraction'))} | {c64.get('a_correct_b_wrong')} | {c64.get('b_correct_a_wrong')} | {fmt(c64_net)} | {fmt(c65.get('correctness_agreement_fraction'))} | {c65.get('a_correct_b_wrong')} | {c65.get('b_correct_a_wrong')} | {fmt(c65_net)} |\n")

    lines.append("\n## Parent / `(M,S)` / clean decomposition\n\n")
    lines.append("| column | clean seed | recover `(M,S)` parent loss | lose parent item `(M,S)` kept | keep `(M,S)` gain | drop `(M,S)` gain | new gain beyond `(M,S)` | shared loss | net clean-`(M,S)` pp |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for col, comp in result["direct_item_comparisons"].items():
        for seed, key in [("62064", "parent_ms_clean64_decomposition"), ("62065", "parent_ms_clean65_decomposition")]:
            d = comp[key]
            lines.append(f"| {col} | {seed} | {d.get('clean_recovers_acquisition_parent_loss',0)} | {d.get('clean_loses_parent_item_acquisition_kept',0)} | {d.get('clean_keeps_acquisition_gain',0)} | {d.get('clean_drops_acquisition_gain',0)} | {d.get('clean_new_gain_beyond_acquisition',0)} | {d.get('shared_loss_vs_parent',0)} | {fmt(d.get('net_clean_minus_acquisition_accuracy_points'))} |\n")

    lines.append("\n## Reading comparisons\n\n")
    for k, v in result["reading_comparisons"].items():
        lines.append(f"- {k}: {json.dumps(v, sort_keys=True)}\n")

    lines.append("\n## Scientific reading\n\n")
    lines.append("On the seven zero-shot/Reading components, exact `(M,S)` is already close to clean. Clean seed62064 gains +0.235 component-sum over `(M,S)`, or +0.026111 Overall units before SuperGLUE, mainly through BLiMP and Supplement, while losing EWoK and leaving GlobalPIQA unchanged. Clean seed62065 gains only +0.130 component-sum over `(M,S)`, or +0.014444 Overall units before SuperGLUE. Combining these with the completed SuperGLUE profile explains the final same-coordinate table: preservation's direct Overall advantage over exact `(M,S)` is real but small for seed62064, and the replicated clean-over-parent result is stronger than the direct clean-over-`(M,S)` increment. The scientific interpretation should remain an acquisition-retention improvement with modest preservation-specific score value, not a broad monotonic upgrade.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build()
    out_json = args.out_dir / "zero_reading_ms_direct_profile.json"
    out_md = args.out_dir / "zero_reading_ms_direct_profile.md"
    out_csv = args.out_dir / "zero_reading_ms_component_deltas.csv"
    out_samples = args.out_dir / "zero_reading_clean64_vs_ms_flip_samples.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out_md, result)
    write_csv(out_csv, result["component_deltas"])
    out_samples.write_text(json.dumps(result["flip_samples_clean64_vs_ms"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "out_csv": rel(out_csv),
        "out_samples": rel(out_samples),
        "interpretation": result["interpretation"],
        "component_deltas": result["component_deltas"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
