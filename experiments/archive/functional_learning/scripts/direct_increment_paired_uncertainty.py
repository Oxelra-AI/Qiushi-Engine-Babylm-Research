#!/usr/bin/env python3
"""research: paired uncertainty analysis for clean preservation vs exact (M,S).

This script reviews the small direct increment reported in research using actual
prediction files.  It keeps BabyLM's component definitions: BLiMP/Supplement/EWoK/
Entity/COMPS are averages over their official subtasks, GlobalPIQA is the mean of
parallel and nonparallel accuracies, Reading uses the official regression score,
and SuperGLUE is the mean of seven primary task metrics.  The bootstrap resamples
paired validation items within each fixed official subtask/task so it is a
measurement-sensitivity analysis for the completed files, not a replacement for
training-seed or downstream-finetuning-seed repeats.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = _public_path('.')
ZERO_MOD_PATH = _public_path('experiments/archive/functional_learning/scripts/zero_reading_item_stability.py')
SG_MOD_PATH = _public_path('experiments/archive/functional_learning/scripts/superglue_completed_item_profile.py')
READING_DATA = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/direct_increment_paired_uncertainty')

ZERO_PAYLOADS = {
    "coherent86": "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json",
    "ms_acquisition_seed62064_MS": "experiments/archive/functional_learning/data/repaired_densemask_sparselabel_zero_reading/per_target/densemask_sparselabel_seed62064_u0080_zero_reading.json",
    "clean_pres_seed62064_MSplusKL": "experiments/archive/functional_learning/data/repaired_clean_eval_zero_reading/per_target/clean_pres_lambda1_eval_seed62064_u0080_zero_reading.json",
    "clean_pres_seed62065_MSplusKL": "experiments/archive/functional_learning/data/repaired_clean_seed62065_zero_reading/per_target/clean_pres_lambda1_eval_seed62065_u0080_zero_reading.json",
}
SG_PAYLOADS = {
    "coherent86": "experiments/archive/functional_learning/data/repaired_coherent86_eval/per_target/repaired_coherent86_alpha075.json",
    "ms_acquisition_seed62064_MS": "experiments/archive/functional_learning/data/repaired_densemask_sparselabel_superglue/per_target/densemask_sparselabel_seed62064_u0080.json",
    "clean_pres_seed62064_MSplusKL": "experiments/archive/functional_learning/data/repaired_clean_eval_superglue/per_target/clean_pres_lambda1_eval_seed62064_u0080.json",
    "clean_pres_seed62065_MSplusKL": "experiments/archive/functional_learning/data/repaired_clean_seed62065_superglue/per_target/clean_pres_lambda1_eval_seed62065_u0080.json",
}
ZERO_COMPONENTS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
SG_TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]
PRIMARY = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str | None) -> Optional[str]:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def pct(x: float) -> float:
    return 100.0 * float(x)


def finite_summary(samples: np.ndarray) -> Dict[str, Any]:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"n_boot": 0}
    return {
        "n_boot": int(s.size),
        "mean": float(np.mean(s)),
        "std": float(np.std(s, ddof=1)) if s.size > 1 else 0.0,
        "ci_2p5": float(np.quantile(s, 0.025)),
        "ci_50": float(np.quantile(s, 0.5)),
        "ci_97p5": float(np.quantile(s, 0.975)),
        "fraction_gt_0": float(np.mean(s > 0)),
        "fraction_lt_0": float(np.mean(s < 0)),
    }


def zero_payload_scores(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    tasks = payload.get("tasks") or {}
    out: Dict[str, Optional[float]] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        out[col] = (tasks.get(col) or {}).get("score")
    if out.get("GlobalPIQA_parallel") is not None and out.get("GlobalPIQA_nonparallel") is not None:
        out["GlobalPIQA"] = (float(out["GlobalPIQA_parallel"]) + float(out["GlobalPIQA_nonparallel"])) / 2.0
    else:
        out["GlobalPIQA"] = None
    if "Reading" in tasks:
        out["Reading"] = (tasks.get("Reading") or {}).get("scores", {}).get("Reading")
        out["Reading_eye"] = (tasks.get("Reading") or {}).get("scores", {}).get("Reading_eye")
        out["Reading_self_paced"] = (tasks.get("Reading") or {}).get("scores", {}).get("Reading_self_paced")
    return out


def grouped_pair_deltas(records_a: Dict[str, Dict[str, Any]], records_b: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for k in sorted(set(records_a) & set(records_b)):
        ra, rb = records_a[k], records_b[k]
        if ra.get("idx") == -1 or rb.get("idx") == -1:
            continue
        # a minus b in percentage-point item correctness units before averaging over groups.
        groups[str(ra.get("subtask"))].append(float(bool(ra["correct"])) - float(bool(rb["correct"])))
    return {g: np.asarray(v, dtype=float) for g, v in sorted(groups.items()) if v}


def official_grouped_delta(groups: Dict[str, np.ndarray]) -> float:
    return 100.0 * float(np.mean([np.mean(v) for v in groups.values()])) if groups else float("nan")


def bootstrap_grouped(groups: Dict[str, np.ndarray], rng: np.random.Generator, n_boot: int) -> np.ndarray:
    keys = list(groups)
    out = np.empty(n_boot, dtype=float)
    if not keys:
        out.fill(np.nan)
        return out
    # Fixed official subtask set: resample paired items within each subtask, then average subtask means.
    for b in range(n_boot):
        vals = []
        for k in keys:
            arr = groups[k]
            idx = rng.integers(0, arr.size, size=arr.size)
            vals.append(float(np.mean(arr[idx])))
        out[b] = 100.0 * float(np.mean(vals))
    return out


def parse_zero_records(zmod, payloads: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, Any]]]], Dict[str, Any], Dict[str, Any]]:
    records: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(dict)
    reading: Dict[str, Any] = {}
    scores: Dict[str, Any] = {}
    for m, payload in payloads.items():
        scores[m] = zero_payload_scores(payload)
        for col in zmod.ZERO_COLUMNS:
            records[m][col] = zmod.parse_zero_column(col, payload["tasks"][col])
        reading[m] = zmod.parse_reading(payload["tasks"]["Reading"])
    return records, reading, scores


def design_eye(df: pd.DataFrame, pred: np.ndarray, dv: str, model: bool) -> Tuple[np.ndarray, np.ndarray]:
    d = df[[dv, "Subtlex_log10", "length", "context_length"]].copy()
    d["pred"] = pred
    d = d.dropna()
    y = d[dv].to_numpy(dtype=float)
    s = d["Subtlex_log10"].to_numpy(dtype=float)
    l = d["length"].to_numpy(dtype=float)
    c = d["context_length"].to_numpy(dtype=float)
    cols = [np.ones_like(y), s, l, c, s * l, s * c, l * c]
    if model:
        cols.append(d["pred"].to_numpy(dtype=float))
    return np.column_stack(cols), y


def design_spr(df: pd.DataFrame, pred: np.ndarray, prev_pred: np.ndarray, model: bool) -> Tuple[np.ndarray, np.ndarray]:
    d = df[["self_paced_reading_time", "Subtlex_log10", "length", "context_length", "prev_length"]].copy()
    d["prev_pred"] = prev_pred
    d["pred"] = pred
    d = d.dropna()
    y = d["self_paced_reading_time"].to_numpy(dtype=float)
    base_names = ["Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]
    arrs = [d[name].to_numpy(dtype=float) for name in base_names]
    cols = [np.ones_like(y)] + arrs
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            cols.append(arrs[i] * arrs[j])
    if model:
        cols.append(d["pred"].to_numpy(dtype=float))
    return np.column_stack(cols), y


def r2_lstsq(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.dot(resid, resid))
    yc = y - float(np.mean(y))
    ss_tot = float(np.dot(yc, yc))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def improvement_score(r2_base: float, r2_model: float) -> float:
    return ((r2_model - r2_base) / (1.0 - r2_base)) * 100.0


def reading_score_from_arrays(df: pd.DataFrame, pred: np.ndarray, prev_pred: np.ndarray) -> Dict[str, float]:
    eyes = []
    for dv in ["RTfirstfix", "RTfirstpass", "RTgopast", "RTrightbound"]:
        Xb, yb = design_eye(df, pred, dv, model=False)
        Xm, ym = design_eye(df, pred, dv, model=True)
        # Same row filtering except pred; use y from each design for safety.
        eyes.append(improvement_score(r2_lstsq(Xb, yb), r2_lstsq(Xm, ym)))
    Xb, yb = design_spr(df, pred, prev_pred, model=False)
    Xm, ym = design_spr(df, pred, prev_pred, model=True)
    spr = improvement_score(r2_lstsq(Xb, yb), r2_lstsq(Xm, ym))
    eye = float(np.mean(eyes))
    return {"Reading_self_paced_unrounded": float(spr), "Reading_eye_unrounded": eye, "Reading_unrounded": float((spr + eye) / 2.0)}


def bootstrap_reading_delta(df: pd.DataFrame, read_a: Dict[str, Any], read_b: Dict[str, Any], rng: np.random.Generator, n_boot: int) -> np.ndarray:
    pred_a = np.asarray(read_a["pred"], dtype=float)
    prev_a = np.asarray(read_a["prev_pred"], dtype=float)
    pred_b = np.asarray(read_b["pred"], dtype=float)
    prev_b = np.asarray(read_b["prev_pred"], dtype=float)
    n = len(df)
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        dfi = df.iloc[idx].reset_index(drop=True)
        sa = reading_score_from_arrays(dfi, pred_a[idx], prev_a[idx])["Reading_unrounded"]
        sb = reading_score_from_arrays(dfi, pred_b[idx], prev_b[idx])["Reading_unrounded"]
        out[i] = sa - sb
    return out


def labels_preds(records: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    keys = sorted(records, key=lambda k: str(records[k].get("position", k)))
    labels = np.asarray([int(records[k]["label"]) for k in keys], dtype=int)
    preds = np.asarray([int(records[k]["pred"]) for k in keys], dtype=int)
    return labels, preds


def f1_binary_np(labels: np.ndarray, preds: np.ndarray) -> float:
    tp = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels != 1) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds != 1)))
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else 100.0 * (2 * tp) / den


def primary_metric_np(task: str, labels: np.ndarray, preds: np.ndarray) -> float:
    if PRIMARY[task] == "accuracy":
        return 100.0 * float(np.mean(labels == preds))
    return f1_binary_np(labels, preds)


def bootstrap_task_primary_delta(task: str, labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    n = labels.size
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[i] = primary_metric_np(task, labels[idx], pred_a[idx]) - primary_metric_np(task, labels[idx], pred_b[idx])
    return out


def exact_sg_and_boot(smod, sg_payloads: Dict[str, Any], a: str, b: str, rng: np.random.Generator, n_boot: int) -> Tuple[Dict[str, Any], np.ndarray]:
    task_rows: List[Dict[str, Any]] = []
    boots = []
    exact_diffs = []
    for task in SG_TASKS:
        ra = smod.parse_task(sg_payloads[a], task)
        rb = smod.parse_task(sg_payloads[b], task)
        keys = sorted(set(ra) & set(rb), key=lambda k: str(ra[k].get("position", k)))
        labels = np.asarray([int(ra[k]["label"]) for k in keys], dtype=int)
        pred_a = np.asarray([int(ra[k]["pred"]) for k in keys], dtype=int)
        pred_b = np.asarray([int(rb[k]["pred"]) for k in keys], dtype=int)
        exact_a = primary_metric_np(task, labels, pred_a)
        exact_b = primary_metric_np(task, labels, pred_b)
        exact = exact_a - exact_b
        exact_diffs.append(exact)
        bs = bootstrap_task_primary_delta(task, labels, pred_a, pred_b, rng, n_boot)
        boots.append(bs)
        task_rows.append({
            "task": task,
            "primary_metric": PRIMARY[task],
            "n": int(labels.size),
            "exact_a": exact_a,
            "exact_b": exact_b,
            "exact_delta_a_minus_b": exact,
            "bootstrap": finite_summary(bs),
        })
    boot_sg = np.mean(np.vstack(boots), axis=0)
    return {"tasks": task_rows, "exact_superglue_delta": float(np.mean(exact_diffs)), "bootstrap": finite_summary(boot_sg)}, boot_sg


def analyze_pair(zmod, smod, zero_payloads: Dict[str, Any], sg_payloads: Dict[str, Any], pair_name: str, a: str, b: str, rng: np.random.Generator, n_boot_zero: int, n_boot_sg: int, n_boot_reading: int) -> Dict[str, Any]:
    zero_records, reading, payload_scores = parse_zero_records(zmod, zero_payloads)
    component_rows: List[Dict[str, Any]] = []
    zero_boot_arrays: List[np.ndarray] = []
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        groups = grouped_pair_deltas(zero_records[a][col], zero_records[b][col])
        exact = official_grouped_delta(groups)
        boot = bootstrap_grouped(groups, rng, n_boot_zero)
        zero_boot_arrays.append(boot)
        component_rows.append({
            "component": col,
            "official_fixed_subtask_exact_delta": exact,
            "payload_delta": float(payload_scores[a][col]) - float(payload_scores[b][col]),
            "n_subtasks": len(groups),
            "n_items": int(sum(v.size for v in groups.values())),
            "bootstrap": finite_summary(boot),
        })
    # GlobalPIQA as fixed mean of parallel and nonparallel.
    gp_boots = []
    gp_exact_parts = []
    gp_item_counts = 0
    for split in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        groups = grouped_pair_deltas(zero_records[a][split], zero_records[b][split])
        ex = official_grouped_delta(groups)
        gp_exact_parts.append(ex)
        gp_item_counts += int(sum(v.size for v in groups.values()))
        gp_boots.append(bootstrap_grouped(groups, rng, n_boot_zero))
    gp_boot = np.mean(np.vstack(gp_boots), axis=0)
    zero_boot_arrays.append(gp_boot)
    component_rows.append({
        "component": "GlobalPIQA",
        "official_fixed_split_exact_delta": float(np.mean(gp_exact_parts)),
        "payload_delta": float(payload_scores[a]["GlobalPIQA"]) - float(payload_scores[b]["GlobalPIQA"]),
        "n_subtasks": 2,
        "n_items": gp_item_counts,
        "bootstrap": finite_summary(gp_boot),
    })
    # Reading row bootstrap with raw unrounded regression score; exact Overall arithmetic still uses payload-rounded component.
    df = pd.read_csv(READING_DATA, dtype={"item": str})
    score_a_un = reading_score_from_arrays(df, np.asarray(reading[a]["pred"], dtype=float), np.asarray(reading[a]["prev_pred"], dtype=float))
    score_b_un = reading_score_from_arrays(df, np.asarray(reading[b]["pred"], dtype=float), np.asarray(reading[b]["prev_pred"], dtype=float))
    read_boot = bootstrap_reading_delta(df, reading[a], reading[b], rng, n_boot_reading)
    # Resize reading boot to n_boot_zero for combining by sampling from its empirical distribution.
    if n_boot_reading != n_boot_zero:
        read_for_sum = rng.choice(read_boot, size=n_boot_zero, replace=True)
    else:
        read_for_sum = read_boot
    zero_boot_arrays.append(read_for_sum)
    component_rows.append({
        "component": "Reading",
        "raw_unrounded_delta": score_a_un["Reading_unrounded"] - score_b_un["Reading_unrounded"],
        "payload_delta": float(payload_scores[a]["Reading"]) - float(payload_scores[b]["Reading"]),
        "raw_unrounded_a": score_a_un,
        "raw_unrounded_b": score_b_un,
        "n_items": int(len(df)),
        "bootstrap_raw_unrounded": finite_summary(read_boot),
    })
    zero_sum_boot = np.sum(np.vstack(zero_boot_arrays), axis=0)
    zero_payload_sum_delta = sum(float((float(payload_scores[a][c]) if c != "Reading" else float(payload_scores[a]["Reading"])) - (float(payload_scores[b][c]) if c != "Reading" else float(payload_scores[b]["Reading"]))) for c in ZERO_COMPONENTS)
    sg_result, sg_boot = exact_sg_and_boot(smod, sg_payloads, a, b, rng, n_boot_sg)
    # Combine zero and SG empirical draws. Reading was already resampled into zero_sum_boot.
    sg_for_overall = sg_boot if n_boot_sg == n_boot_zero else rng.choice(sg_boot, size=n_boot_zero, replace=True)
    overall_boot = (zero_sum_boot + sg_for_overall) / 9.0
    exact_overall_payload_units = (zero_payload_sum_delta + sg_result["exact_superglue_delta"]) / 9.0
    return {
        "pair_name": pair_name,
        "a": a,
        "b": b,
        "interpretation_scope": "Paired resampling of existing official prediction files; it measures sensitivity to finite validation items and shared item structure, not private-training or downstream-finetuning randomness.",
        "zero_reading_components": component_rows,
        "zero_reading_payload_sum_delta": zero_payload_sum_delta,
        "zero_reading_payload_overall_units": zero_payload_sum_delta / 9.0,
        "zero_reading_bootstrap_sum": finite_summary(zero_sum_boot),
        "zero_reading_bootstrap_overall_units": finite_summary(zero_sum_boot / 9.0),
        "superglue": sg_result,
        "superglue_overall_units": sg_result["exact_superglue_delta"] / 9.0,
        "overall_direct_delta_payload_components": exact_overall_payload_units,
        "overall_direct_bootstrap": finite_summary(overall_boot),
    }


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            flat = {}
            for k, v in r.items():
                flat[k] = json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
            w.writerow(flat)


def fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return "None"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def write_md(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research paired uncertainty: clean preservation versus exact `(M,S)`\n\n")
    lines.append("This analysis uses completed official prediction files. It keeps the official component definitions and resamples paired validation items within the fixed official subtasks/tasks. It does not estimate private-training randomness or downstream fine-tuning seed variation.\n\n")
    for pair in result["pairs"]:
        lines.append(f"## {pair['pair_name']}\n\n")
        lines.append(f"Compared `{pair['a']}` minus `{pair['b']}`.\n\n")
        lines.append(f"- zero/Reading payload sum delta: `{fmt(pair['zero_reading_payload_sum_delta'])}` = `{fmt(pair['zero_reading_payload_overall_units'])}` Overall units\n")
        lines.append(f"- SuperGLUE primary delta: `{fmt(pair['superglue']['exact_superglue_delta'])}` = `{fmt(pair['superglue_overall_units'])}` Overall units\n")
        lines.append(f"- direct Overall delta from payload components: `{fmt(pair['overall_direct_delta_payload_components'])}`\n")
        ob = pair["overall_direct_bootstrap"]
        lines.append(f"- paired item/bootstrap direct Overall interval: median `{fmt(ob.get('ci_50'))}`, 2.5--97.5% `{fmt(ob.get('ci_2p5'))}` to `{fmt(ob.get('ci_97p5'))}`, fraction > 0 `{fmt(ob.get('fraction_gt_0'),4)}`\n\n")
        lines.append("### Component contributions\n\n")
        lines.append("| component/task group | exact/payload delta | n units | 2.5% | 50% | 97.5% | fraction > 0 |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in pair["zero_reading_components"]:
            boot = row.get("bootstrap") or row.get("bootstrap_raw_unrounded") or {}
            exact = row.get("payload_delta", row.get("official_fixed_subtask_exact_delta", row.get("raw_unrounded_delta")))
            n_units = row.get("n_subtasks", row.get("n_items"))
            lines.append(f"| {row['component']} | {fmt(float(exact))} | {n_units} | {fmt(boot.get('ci_2p5'))} | {fmt(boot.get('ci_50'))} | {fmt(boot.get('ci_97p5'))} | {fmt(boot.get('fraction_gt_0'),4)} |\n")
        lines.append("\n### SuperGLUE task contributions\n\n")
        lines.append("| task | metric | exact clean-minus-reference | n | 2.5% | 50% | 97.5% | fraction > 0 |\n")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in pair["superglue"]["tasks"]:
            boot = row["bootstrap"]
            lines.append(f"| {row['task']} | {row['primary_metric']} | {fmt(row['exact_delta_a_minus_b'])} | {row['n']} | {fmt(boot.get('ci_2p5'))} | {fmt(boot.get('ci_50'))} | {fmt(boot.get('ci_97p5'))} | {fmt(boot.get('fraction_gt_0'),4)} |\n")
        lines.append("\n")
    lines.append("## Scientific reading\n\n")
    lines.append("The completed files establish a higher fixed-coordinate score for clean seed62064 over coherent86, but this paired resampling does not establish broad superiority beyond the fixed validation pools: the clean64-vs-coherent86 interval also crosses zero under this conditional item perturbation. The preservation-specific increment over exact `(M,S)` is small. The direct clean64-minus-`(M,S)` score is carried by a narrow positive zero/Reading sum plus a SuperGLUE difference of about 0.160 component points. The paired item resampling asks whether those finite validation files make the sign fragile; it should be read together with the still-needed same-downstream-seed SuperGLUE comparison, because item resampling cannot stand in for fine-tuning-seed variation. Endpoint selection remains separate: seed62064 is the higher complete fixed-coordinate endpoint, while seed62065 is the training-seed replication.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--n-boot-zero", type=int, default=1500)
    ap.add_argument("--n-boot-sg", type=int, default=1500)
    ap.add_argument("--n-boot-reading", type=int, default=500)
    ap.add_argument("--seed", type=int, default=99099)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zmod = load_module(ZERO_MOD_PATH, "zero_reading_item_stability_for_step099")
    smod = load_module(SG_MOD_PATH, "superglue_completed_item_profile_for_step099")
    zero_payloads = {m: zmod.load_json_file(p) for m, p in ZERO_PAYLOADS.items()}
    sg_payloads = {m: smod.read_json(p) for m, p in SG_PAYLOADS.items()}
    rng = np.random.default_rng(args.seed)
    pairs = [
        analyze_pair(zmod, smod, zero_payloads, sg_payloads, "clean64_vs_exact_MS", "clean_pres_seed62064_MSplusKL", "ms_acquisition_seed62064_MS", rng, args.n_boot_zero, args.n_boot_sg, args.n_boot_reading),
        analyze_pair(zmod, smod, zero_payloads, sg_payloads, "clean65_vs_exact_MS_training_seed_replication_context", "clean_pres_seed62065_MSplusKL", "ms_acquisition_seed62064_MS", rng, args.n_boot_zero, args.n_boot_sg, args.n_boot_reading),
        analyze_pair(zmod, smod, zero_payloads, sg_payloads, "clean64_vs_coherent86_policy_gain", "clean_pres_seed62064_MSplusKL", "coherent86", rng, args.n_boot_zero, args.n_boot_sg, args.n_boot_reading),
    ]
    result = {
        "status": "DIRECT_INCREMENT_PAIRED_UNCERTAINTY",
        "created_utc": now(),
        "bootstrap_seed": args.seed,
        "n_boot_zero": args.n_boot_zero,
        "n_boot_sg": args.n_boot_sg,
        "n_boot_reading": args.n_boot_reading,
        "zero_payloads": ZERO_PAYLOADS,
        "superglue_payloads": SG_PAYLOADS,
        "reading_data": rel(READING_DATA),
        "pairs": pairs,
        "important_distinction": "Endpoint selection, policy gain over coherent86, preservation-specific increment over exact acquisition, and downstream fine-tuning seed variation are different questions.",
    }
    out_json = args.out_dir / "direct_increment_paired_uncertainty.json"
    out_md = args.out_dir / "direct_increment_paired_uncertainty.md"
    out_comp = args.out_dir / "component_bootstrap_summary.csv"
    out_sg = args.out_dir / "superglue_task_bootstrap_summary.csv"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out_md, result)
    comp_rows = []
    sg_rows = []
    for pair in pairs:
        for r in pair["zero_reading_components"]:
            row = dict(r)
            row["pair_name"] = pair["pair_name"]
            comp_rows.append(row)
        for r in pair["superglue"]["tasks"]:
            row = dict(r)
            row["pair_name"] = pair["pair_name"]
            sg_rows.append(row)
    write_csv(out_comp, comp_rows)
    write_csv(out_sg, sg_rows)
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "out_component_csv": rel(out_comp),
        "out_superglue_csv": rel(out_sg),
        "pair_summaries": [{
            "pair_name": p["pair_name"],
            "zero_reading_payload_overall_units": p["zero_reading_payload_overall_units"],
            "superglue_overall_units": p["superglue_overall_units"],
            "overall_direct_delta_payload_components": p["overall_direct_delta_payload_components"],
            "overall_bootstrap": p["overall_direct_bootstrap"],
        } for p in pairs],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
