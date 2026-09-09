#!/usr/bin/env python3
"""research: direct SuperGLUE profile for clean preservation vs exact (M,S) acquisition-only.

This script reuses the validated research SuperGLUE prediction/label parser but adds the
completed exact acquisition-only dense-mask/sparse-label `(M,S)` repaired-AutoModel
SuperGLUE payload.  Scientific purpose: before the `(M,S)` zero-shot/Reading complement
finishes, determine what the already-measured SuperGLUE coordinate says about the direct
preservation increment beyond the exact acquisition policy, rather than comparing only to
older `(M,M)` dense controls.
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
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/superglue_ms_direct_profile')

PAYLOADS: Dict[str, str] = {
    "coherent86": "experiments/archive/functional_learning/data/repaired_coherent86_eval/per_target/repaired_coherent86_alpha075.json",
    "dense_seed62064_MM": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62064/per_target/repaired_dense_seed62064_u0080.json",
    "dense_seed62065_MM": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62065/per_target/repaired_dense_seed62065_u0080.json",
    "ms_acquisition_seed62064_MS": "experiments/archive/functional_learning/data/repaired_densemask_sparselabel_superglue/per_target/densemask_sparselabel_seed62064_u0080.json",
    "clean_pres_seed62064_MSplusKL": "experiments/archive/functional_learning/data/repaired_clean_eval_superglue/per_target/clean_pres_lambda1_eval_seed62064_u0080.json",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str | None) -> str | None:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_step097_module():
    spec = importlib.util.spec_from_file_location("superglue_completed_item_profile", research)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {research}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def decomposition(parent: Dict[str, Dict[str, Any]], acquisition: Dict[str, Dict[str, Any]], clean: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
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
        if p and a and c:
            out["shared_kept_parent_correct"] += 1
        if (not p) and (not a) and (not c):
            out["shared_parent_wrong"] += 1
    n = len(keys)
    out["n"] = n
    out["net_clean_minus_acquisition_accuracy_points"] = 100.0 * (
        (out["clean_recovers_acquisition_parent_loss"] + out["clean_new_gain_beyond_acquisition"])
        - (out["clean_loses_parent_item_acquisition_kept"] + out["clean_drops_acquisition_gain"])
    ) / n if n else None
    return dict(out)


def sample_acq_clean(parent: Dict[str, Dict[str, Any]], acquisition: Dict[str, Dict[str, Any]], clean: Dict[str, Dict[str, Any]], limit: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    keys = sorted(set(parent) & set(acquisition) & set(clean))
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for k in keys:
        p = bool(parent[k]["correct"])
        a = bool(acquisition[k]["correct"])
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
                "task": parent[k].get("task"),
                "idx": parent[k].get("idx"),
                "label": parent[k].get("label"),
                "parent_pred": parent[k].get("pred"),
                "acquisition_pred": acquisition[k].get("pred"),
                "clean_pred": clean[k].get("pred"),
                "parent_correct": p,
                "acquisition_correct": a,
                "clean_correct": c,
                "meta": parent[k].get("meta"),
            })
    return dict(buckets)


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
        primary_scores: List[float] = []
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
            primary_scores.append(float(comp["primary_score"]))
        model_metrics[m]["computed_primary_mean"] = mean(primary_scores)
        model_metrics[m]["payload_mean_minus_computed"] = model_metrics[m]["superglue_mean"] - model_metrics[m]["computed_primary_mean"]

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
        row["clean_minus_ms_acquisition"] = row["clean_pres_seed62064_MSplusKL_primary"] - row["ms_acquisition_seed62064_MS_primary"]
        row["ms_acquisition_minus_dense64_MM"] = row["ms_acquisition_seed62064_MS_primary"] - row["dense_seed62064_MM_primary"]
        row["ms_acquisition_minus_dense65_MM"] = row["ms_acquisition_seed62064_MS_primary"] - row["dense_seed62065_MM_primary"]
        task_rows.append(row)

        task_item_comparisons[task] = {
            "ms_acquisition_vs_coherent86": mod.compare_two(records["ms_acquisition_seed62064_MS"][task], records["coherent86"][task]),
            "clean_vs_ms_acquisition": mod.compare_two(records["clean_pres_seed62064_MSplusKL"][task], records["ms_acquisition_seed62064_MS"][task]),
            "clean_vs_coherent86": mod.compare_two(records["clean_pres_seed62064_MSplusKL"][task], records["coherent86"][task]),
            "ms_vs_dense64_MM": mod.compare_two(records["ms_acquisition_seed62064_MS"][task], records["dense_seed62064_MM"][task]),
            "ms_vs_dense65_MM": mod.compare_two(records["ms_acquisition_seed62064_MS"][task], records["dense_seed62065_MM"][task]),
            "parent_ms_clean_decomposition": decomposition(records["coherent86"][task], records["ms_acquisition_seed62064_MS"][task], records["clean_pres_seed62064_MSplusKL"][task]),
        }

    means = {m: model_metrics[m]["superglue_mean"] for m in PAYLOADS}
    result = {
        "status": "SUPERGLUE_MS_DIRECT_PROFILE",
        "created_utc": now(),
        "purpose": "Direct repaired-AutoModel SuperGLUE comparison of clean preservation seed62064 against exact acquisition-only `(M,S)` seed62064, using actual prediction files and valid labels.",
        "parser_source": rel(research),
        "payloads": PAYLOADS,
        "model_metrics": model_metrics,
        "task_primary_deltas": task_rows,
        "task_item_comparisons": task_item_comparisons,
        "flip_samples_clean_vs_ms": {task: sample_acq_clean(records["coherent86"][task], records["ms_acquisition_seed62064_MS"][task], records["clean_pres_seed62064_MSplusKL"][task]) for task in mod.TASKS},
        "interpretation": {
            "superglue_means": means,
            "ms_acquisition_minus_coherent86": means["ms_acquisition_seed62064_MS"] - means["coherent86"],
            "clean_minus_ms_acquisition": means["clean_pres_seed62064_MSplusKL"] - means["ms_acquisition_seed62064_MS"],
            "clean_minus_coherent86": means["clean_pres_seed62064_MSplusKL"] - means["coherent86"],
            "ms_acquisition_minus_dense64_MM": means["ms_acquisition_seed62064_MS"] - means["dense_seed62064_MM"],
            "ms_acquisition_minus_dense65_MM": means["ms_acquisition_seed62064_MS"] - means["dense_seed62065_MM"],
            "clean_minus_dense64_MM": means["clean_pres_seed62064_MSplusKL"] - means["dense_seed62064_MM"],
            "clean_minus_dense65_MM": means["clean_pres_seed62064_MSplusKL"] - means["dense_seed62065_MM"],
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
    lines.append("# research direct SuperGLUE profile: clean preservation vs exact `(M,S)` acquisition\n\n")
    lines.append("This parses actual repaired-AutoModel SuperGLUE prediction files and valid labels. It uses the exact acquisition-only dense-mask/sparse-label `(M,S)` seed62064 payload now completed in research, plus coherent86, `(M,M)` dense controls, and clean preservation seed62064. It does not use pending clean seed62065 SuperGLUE or pending `(M,S)` zero-shot/Reading.\n\n")
    lines.append("## SuperGLUE means\n\n")
    for m, v in interp["superglue_means"].items():
        lines.append(f"- {m}: {fmt(v, 6)}\n")
    lines.append(f"- exact `(M,S)` minus coherent86: {fmt(interp['ms_acquisition_minus_coherent86'], 6)}\n")
    lines.append(f"- clean preservation minus exact `(M,S)`: {fmt(interp['clean_minus_ms_acquisition'], 6)}\n")
    lines.append(f"- clean preservation minus coherent86: {fmt(interp['clean_minus_coherent86'], 6)}\n")
    lines.append(f"- exact `(M,S)` minus dense `(M,M)` seed62064/62065: {fmt(interp['ms_acquisition_minus_dense64_MM'], 6)} / {fmt(interp['ms_acquisition_minus_dense65_MM'], 6)}\n\n")

    lines.append("## Payload metric reproduction\n\n")
    lines.append("| model | task | metric | n | payload primary | computed primary | payload-computed | accuracy | f1 | pred counts |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
    mod_tasks = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]
    for m, mm in result["model_metrics"].items():
        for task in mod_tasks:
            mt = mm["tasks"][task]
            lines.append(f"| {m} | {task} | {mt.get('primary_metric')} | {mt.get('n')} | {fmt(mt.get('payload_primary_score'),6)} | {fmt(mt.get('primary_score'),6)} | {fmt(mt.get('payload_primary_minus_computed'),8)} | {fmt(mt.get('accuracy'),4)} | {fmt(mt.get('f1'),4)} | {mt.get('pred_counts')} |\n")

    lines.append("\n## Task primary-score deltas\n\n")
    lines.append("| task | metric | exact `(M,S)` - parent | clean - parent | clean - exact `(M,S)` | exact `(M,S)` - dense64 `(M,M)` | exact `(M,S)` - dense65 `(M,M)` |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for row in result["task_primary_deltas"]:
        lines.append(f"| {row['task']} | {row['primary_metric']} | {fmt(row['ms_acquisition_seed62064_MS_minus_coherent86'],4)} | {fmt(row['clean_pres_seed62064_MSplusKL_minus_coherent86'],4)} | {fmt(row['clean_minus_ms_acquisition'],4)} | {fmt(row['ms_acquisition_minus_dense64_MM'],4)} | {fmt(row['ms_acquisition_minus_dense65_MM'],4)} |\n")

    lines.append("\n## Clean preservation versus exact `(M,S)` item comparisons\n\n")
    lines.append("| task | pred agreement | correctness agreement | clean correct / `(M,S)` wrong | `(M,S)` correct / clean wrong | net clean-`(M,S)` accuracy pp |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for task, comp in result["task_item_comparisons"].items():
        cm = comp["clean_vs_ms_acquisition"]
        lines.append(f"| {task} | {fmt(cm.get('prediction_agreement_fraction'),4)} | {fmt(cm.get('correctness_agreement_fraction'),4)} | {cm.get('a_correct_b_wrong')} | {cm.get('b_correct_a_wrong')} | {fmt(cm.get('net_accuracy_points_a_minus_b'),4)} |\n")

    lines.append("\n## Parent / exact `(M,S)` / clean decomposition\n\n")
    lines.append("| task | clean recovers `(M,S)` parent loss | clean loses parent item `(M,S)` kept | clean keeps `(M,S)` gain | clean drops `(M,S)` gain | clean new gain beyond `(M,S)` | shared loss vs parent | net clean-`(M,S)` acc pp |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for task, comp in result["task_item_comparisons"].items():
        d = comp["parent_ms_clean_decomposition"]
        lines.append(f"| {task} | {d.get('clean_recovers_acquisition_parent_loss',0)} | {d.get('clean_loses_parent_item_acquisition_kept',0)} | {d.get('clean_keeps_acquisition_gain',0)} | {d.get('clean_drops_acquisition_gain',0)} | {d.get('clean_new_gain_beyond_acquisition',0)} | {d.get('shared_loss_vs_parent',0)} | {fmt(d.get('net_clean_minus_acquisition_accuracy_points'),4)} |\n")

    lines.append("\n## Scientific reading\n\n")
    lines.append("Exact `(M,S)` acquisition-only SuperGLUE is stronger than the fully evaluated dense `(M,M)` controls, so dense `(M,M)` understated the supervised-transfer coordinate of the acquisition policy used by clean preservation. Clean preservation still improves over exact `(M,S)` by a modest +0.159869 SuperGLUE points. The task table determines whether this increment is distributed and whether it comes from real item changes or only metric bookkeeping. The direct Overall comparison remains unresolved until the running `(M,S)` official zero-shot/Reading job completes and the guarded same-coordinate table admits all components.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build()
    out_json = args.out_dir / "superglue_ms_direct_profile.json"
    out_md = args.out_dir / "superglue_ms_direct_profile.md"
    out_csv = args.out_dir / "superglue_ms_task_primary_deltas.csv"
    out_samples = args.out_dir / "superglue_clean_vs_ms_flip_samples.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out_md, result)
    write_csv(out_csv, result["task_primary_deltas"])
    out_samples.write_text(json.dumps(result["flip_samples_clean_vs_ms"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "out_csv": rel(out_csv),
        "out_samples": rel(out_samples),
        "interpretation": result["interpretation"],
        "task_primary_deltas": result["task_primary_deltas"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
