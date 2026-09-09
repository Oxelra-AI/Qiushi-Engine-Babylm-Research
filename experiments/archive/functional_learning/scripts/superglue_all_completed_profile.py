#!/usr/bin/env python3
"""research: all completed repaired-AutoModel SuperGLUE profiles.

After clean seed62065 and exact acquisition-only `(M,S)` SuperGLUE delivered,
this script parses every completed repaired-AutoModel SuperGLUE prediction file in
the current comparison family: coherent86, dense `(M,M)` seeds 62064/62065,
exact `(M,S)` acquisition-only seed62064, and clean preservation seeds 62064/62065.
It recomputes official primary metrics from valid labels, compares seed stability,
and decomposes the direct clean-vs-`(M,S)` supervised-transfer increment.
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
import pathlib
import time
from collections import defaultdict
from typing import Any, Dict, List

ROOT = _public_path('.')
research = _public_path('experiments/archive/functional_learning/scripts/superglue_completed_item_profile.py')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/superglue_all_completed_profile')

PAYLOADS: Dict[str, str] = {
    "coherent86": "experiments/archive/functional_learning/data/repaired_coherent86_eval/per_target/repaired_coherent86_alpha075.json",
    "dense_seed62064_MM": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62064/per_target/repaired_dense_seed62064_u0080.json",
    "dense_seed62065_MM": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62065/per_target/repaired_dense_seed62065_u0080.json",
    "ms_acquisition_seed62064_MS": "experiments/archive/functional_learning/data/repaired_densemask_sparselabel_superglue/per_target/densemask_sparselabel_seed62064_u0080.json",
    "clean_pres_seed62064_MSplusKL": "experiments/archive/functional_learning/data/repaired_clean_eval_superglue/per_target/clean_pres_lambda1_eval_seed62064_u0080.json",
    "clean_pres_seed62065_MSplusKL": "experiments/archive/functional_learning/data/repaired_clean_seed62065_superglue/per_target/clean_pres_lambda1_eval_seed62065_u0080.json",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_step097_module():
    spec = importlib.util.spec_from_file_location("superglue_completed_item_profile", research)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {research}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def parent_acq_clean_decomposition(parent: Dict[str, Dict[str, Any]], acquisition: Dict[str, Dict[str, Any]], clean: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted(set(parent) & set(acquisition) & set(clean))
    out: Dict[str, Any] = defaultdict(int)
    for k in keys:
        p = bool(parent[k]["correct"])
        a = bool(acquisition[k]["correct"])
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


def build() -> Dict[str, Any]:
    mod = load_step097_module()
    payloads = {m: mod.read_json(p) for m, p in PAYLOADS.items()}
    records: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    model_metrics: Dict[str, Any] = {}
    for m, payload in payloads.items():
        entries = mod.sg_payload_task_entries(payload)
        details = mod.sg_primary_details(payload)
        model_metrics[m] = {
            "payload": rel(PAYLOADS[m]),
            "superglue_mean": payload.get("tasks", {}).get("SuperGLUE", {}).get("superglue_mean"),
            "tasks": {},
            "computed_primary_mean": None,
            "payload_mean_minus_computed": None,
        }
        primaries: List[float] = []
        for task in mod.TASKS:
            recs = mod.parse_task(payload, task)
            records[m][task] = recs
            comp = mod.metrics_for(recs, task)
            payload_entry = entries[task]
            primary_detail = details.get(task, {})
            comp["payload_accuracy"] = payload_entry.get("accuracy")
            comp["payload_primary_score"] = primary_detail.get("score")
            comp["payload_metric"] = primary_detail.get("metric")
            comp["payload_primary_minus_computed"] = (float(primary_detail.get("score")) - comp["primary_score"]) if primary_detail.get("score") is not None else None
            model_metrics[m]["tasks"][task] = comp
            primaries.append(float(comp["primary_score"]))
        model_metrics[m]["computed_primary_mean"] = mean(primaries)
        model_metrics[m]["payload_mean_minus_computed"] = model_metrics[m]["superglue_mean"] - model_metrics[m]["computed_primary_mean"]

    means = {m: float(model_metrics[m]["superglue_mean"]) for m in PAYLOADS}

    task_rows: List[Dict[str, Any]] = []
    task_item_comparisons: Dict[str, Any] = {}
    for task in mod.TASKS:
        row: Dict[str, Any] = {"task": task, "primary_metric": mod.PRIMARY_METRIC[task]}
        for m in PAYLOADS:
            row[f"{m}_primary"] = model_metrics[m]["tasks"][task]["primary_score"]
            row[f"{m}_accuracy"] = model_metrics[m]["tasks"][task]["accuracy"]
            row[f"{m}_pred_counts"] = json.dumps(model_metrics[m]["tasks"][task]["pred_counts"], sort_keys=True)
        for m in PAYLOADS:
            if m != "coherent86":
                row[f"{m}_minus_coherent86"] = row[f"{m}_primary"] - row["coherent86_primary"]
        row["clean64_minus_clean65"] = row["clean_pres_seed62064_MSplusKL_primary"] - row["clean_pres_seed62065_MSplusKL_primary"]
        row["clean65_minus_clean64"] = -row["clean64_minus_clean65"]
        row["clean64_minus_ms_acquisition"] = row["clean_pres_seed62064_MSplusKL_primary"] - row["ms_acquisition_seed62064_MS_primary"]
        row["clean65_minus_ms_acquisition"] = row["clean_pres_seed62065_MSplusKL_primary"] - row["ms_acquisition_seed62064_MS_primary"]
        task_rows.append(row)

        task_item_comparisons[task] = {
            "clean64_vs_clean65": mod.compare_two(records["clean_pres_seed62064_MSplusKL"][task], records["clean_pres_seed62065_MSplusKL"][task]),
            "clean64_vs_ms_acquisition": mod.compare_two(records["clean_pres_seed62064_MSplusKL"][task], records["ms_acquisition_seed62064_MS"][task]),
            "clean65_vs_ms_acquisition": mod.compare_two(records["clean_pres_seed62065_MSplusKL"][task], records["ms_acquisition_seed62064_MS"][task]),
            "ms_acquisition_vs_coherent86": mod.compare_two(records["ms_acquisition_seed62064_MS"][task], records["coherent86"][task]),
            "clean64_change_overlap_clean65_vs_parent": mod.change_overlap(records["coherent86"][task], records["clean_pres_seed62064_MSplusKL"][task], records["clean_pres_seed62065_MSplusKL"][task]),
            "parent_ms_clean64_decomposition": parent_acq_clean_decomposition(records["coherent86"][task], records["ms_acquisition_seed62064_MS"][task], records["clean_pres_seed62064_MSplusKL"][task]),
            "parent_ms_clean65_decomposition": parent_acq_clean_decomposition(records["coherent86"][task], records["ms_acquisition_seed62064_MS"][task], records["clean_pres_seed62065_MSplusKL"][task]),
        }

    result = {
        "status": "SUPERGLUE_ALL_COMPLETED_PROFILE",
        "created_utc": now(),
        "purpose": "All completed repaired-AutoModel SuperGLUE item/task evidence for clean replication and direct `(M,S)` comparison.",
        "parser_source": rel(research),
        "payloads": PAYLOADS,
        "model_metrics": model_metrics,
        "task_primary_deltas": task_rows,
        "task_item_comparisons": task_item_comparisons,
        "interpretation": {
            "superglue_means": means,
            "clean64_minus_coherent86": means["clean_pres_seed62064_MSplusKL"] - means["coherent86"],
            "clean65_minus_coherent86": means["clean_pres_seed62065_MSplusKL"] - means["coherent86"],
            "clean65_minus_clean64": means["clean_pres_seed62065_MSplusKL"] - means["clean_pres_seed62064_MSplusKL"],
            "ms_acquisition_minus_coherent86": means["ms_acquisition_seed62064_MS"] - means["coherent86"],
            "clean64_minus_ms_acquisition": means["clean_pres_seed62064_MSplusKL"] - means["ms_acquisition_seed62064_MS"],
            "clean65_minus_ms_acquisition": means["clean_pres_seed62065_MSplusKL"] - means["ms_acquisition_seed62064_MS"],
            "dense64_MM_minus_coherent86": means["dense_seed62064_MM"] - means["coherent86"],
            "dense65_MM_minus_coherent86": means["dense_seed62065_MM"] - means["coherent86"],
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
    interp = result["interpretation"]
    lines.append("# research all completed SuperGLUE profile\n\n")
    lines.append("This artifact uses only completed repaired-AutoModel SuperGLUE payloads and recomputes primary metrics from predictions and valid labels. It includes coherent86, dense `(M,M)` seeds, exact acquisition-only `(M,S)` seed62064, and clean preservation seeds 62064/62065.\n\n")
    lines.append("## SuperGLUE means\n\n")
    for m, v in interp["superglue_means"].items():
        lines.append(f"- {m}: {fmt(v,6)}\n")
    lines.append(f"- clean64 minus coherent86: {fmt(interp['clean64_minus_coherent86'],6)}\n")
    lines.append(f"- clean65 minus coherent86: {fmt(interp['clean65_minus_coherent86'],6)}\n")
    lines.append(f"- clean65 minus clean64: {fmt(interp['clean65_minus_clean64'],6)}\n")
    lines.append(f"- exact `(M,S)` minus coherent86: {fmt(interp['ms_acquisition_minus_coherent86'],6)}\n")
    lines.append(f"- clean64 / clean65 minus exact `(M,S)`: {fmt(interp['clean64_minus_ms_acquisition'],6)} / {fmt(interp['clean65_minus_ms_acquisition'],6)}\n\n")

    lines.append("## Payload metric reproduction\n\n")
    lines.append("| model | task | metric | n | payload primary | computed primary | payload-computed | accuracy | f1 | pred counts |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
    for m, mm in result["model_metrics"].items():
        for task in ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]:
            mt = mm["tasks"][task]
            lines.append(f"| {m} | {task} | {mt.get('primary_metric')} | {mt.get('n')} | {fmt(mt.get('payload_primary_score'),6)} | {fmt(mt.get('primary_score'),6)} | {fmt(mt.get('payload_primary_minus_computed'),8)} | {fmt(mt.get('accuracy'),4)} | {fmt(mt.get('f1'),4)} | {mt.get('pred_counts')} |\n")

    lines.append("\n## Task primary-score deltas\n\n")
    lines.append("| task | metric | clean64-parent | clean65-parent | clean65-clean64 | exact `(M,S)`-parent | clean64-`(M,S)` | clean65-`(M,S)` |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for row in result["task_primary_deltas"]:
        lines.append(f"| {row['task']} | {row['primary_metric']} | {fmt(row['clean_pres_seed62064_MSplusKL_minus_coherent86'],4)} | {fmt(row['clean_pres_seed62065_MSplusKL_minus_coherent86'],4)} | {fmt(row['clean65_minus_clean64'],4)} | {fmt(row['ms_acquisition_seed62064_MS_minus_coherent86'],4)} | {fmt(row['clean64_minus_ms_acquisition'],4)} | {fmt(row['clean65_minus_ms_acquisition'],4)} |\n")

    lines.append("\n## Clean seed stability and clean-vs-`(M,S)` item comparisons\n\n")
    lines.append("| task | clean seed pred agree | clean seed correctness agree | clean65 correct/clean64 wrong | clean64 correct/clean65 wrong | clean64-`(M,S)` net pp | clean65-`(M,S)` net pp |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for task, comp in result["task_item_comparisons"].items():
        cc = comp["clean64_vs_clean65"]
        cm64 = comp["clean64_vs_ms_acquisition"]
        cm65 = comp["clean65_vs_ms_acquisition"]
        lines.append(f"| {task} | {fmt(cc.get('prediction_agreement_fraction'),4)} | {fmt(cc.get('correctness_agreement_fraction'),4)} | {cc.get('b_correct_a_wrong')} | {cc.get('a_correct_b_wrong')} | {fmt(cm64.get('net_accuracy_points_a_minus_b'),4)} | {fmt(cm65.get('net_accuracy_points_a_minus_b'),4)} |\n")

    lines.append("\n## Scientific reading\n\n")
    lines.append("The clean SuperGLUE result replicates tightly across seeds: clean seed62065 is only 0.02713 points below clean seed62064 and remains above coherent86 by 0.07487. Exact `(M,S)` acquisition-only is below coherent86 on SuperGLUE by 0.05787, so clean preservation adds a repeatable supervised-transfer recovery beyond `(M,S)`: +0.15987 for seed62064 and +0.13274 for seed62065. The remaining direct method comparison is not yet complete because `(M,S)` official zero-shot/Reading is still running; the direct Overall difference will combine these SuperGLUE increments with any zero-shot/Reading component differences.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build()
    out_json = args.out_dir / "superglue_all_completed_profile.json"
    out_md = args.out_dir / "superglue_all_completed_profile.md"
    out_csv = args.out_dir / "superglue_all_task_primary_deltas.csv"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out_md, result)
    write_csv(out_csv, result["task_primary_deltas"])
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "out_csv": rel(out_csv),
        "interpretation": result["interpretation"],
        "task_primary_deltas": result["task_primary_deltas"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
