#!/usr/bin/env python3
"""research official-coordinate prediction transition comparison.

This is a repaired version of the research item-transition script.  The main
scientific reason is that BabyLM zero-shot columns are scored as macro averages
across subtasks, while raw item flips are micro counts.  A private-scale or
continuation comparison can therefore look large or small depending on whether
one reads unweighted item counts or the official column arithmetic.  This script
recomputes both from the prediction files, validates the recomputed official
scores against the payload, and keeps item flips as a separate diagnostic.

Use it to compare two official-style payloads, optionally each assembled from a
zero-shot/Reading payload plus a separate SuperGLUE payload.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
PRISTINE_FULL = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
SUPERGLUE_TASKS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]
SUPERGLUE_PRIMARY = {
    "boolq": "accuracy",
    "mnli": "accuracy",
    "mrpc": "f1",
    "multirc": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "wsc": "accuracy",
}
COMPS_FILE = {
    "base": "comps_base",
    "wugs_dist_before": "comps_wugs_dist-before",
    "wugs_dist_in_between": "comps_wugs_dist-in-between",
    "wugs": "comps_wugs",
}
ZERO_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
OFFICIAL9_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
CHEAP7_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def as_path(x: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(x)
    return p if p.is_absolute() else ROOT / p


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def clean(x: Any) -> Any:
    return x.strip() if isinstance(x, str) else x


def pred_list(block: dict[str, Any], key: str) -> list[dict[str, Any]]:
    if key not in block or not isinstance(block[key], dict) or not isinstance(block[key].get("predictions"), list):
        raise KeyError(f"prediction subtask missing or malformed: {key}")
    return block[key]["predictions"]


def load_payloads(paths: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {"tasks": {}, "source_payloads": []}
    for pstr in paths:
        if not pstr:
            continue
        path = as_path(pstr)
        payload = load_json(path)
        merged["source_payloads"].append(rel(path))
        for k, v in (payload.get("tasks") or {}).items():
            merged["tasks"][k] = v
        for k in ["target", "model_path", "run_dir", "endpoint", "official_overall"]:
            if k in payload and k not in merged:
                merged[k] = payload[k]
    return merged


def task_rec(payload: dict[str, Any], column: str) -> dict[str, Any]:
    rec = (payload.get("tasks") or {}).get(column)
    if not isinstance(rec, dict):
        raise KeyError(f"payload missing task {column}")
    return rec


def prediction_path(payload: dict[str, Any], column: str) -> pathlib.Path:
    rec = task_rec(payload, column)
    p = rec.get("predictions")
    if not p:
        raise KeyError(f"task {column} lacks predictions path")
    return as_path(p)


def data_path(payload: dict[str, Any], column: str, fallback: pathlib.Path) -> pathlib.Path:
    rec = task_rec(payload, column)
    p = rec.get("data_path")
    return as_path(p) if p else fallback


def make_item(column: str, subtask: str, index: int, item_id: Any, pred: Any, target: Any, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return {
        "column": column,
        "subtask": str(subtask),
        "index": int(index),
        "id": str(item_id),
        "pred": clean(pred),
        "target": clean(target),
        "correct": clean(pred) == clean(target),
        "metadata": metadata or {},
    }


def blimp_like_items(payload: dict[str, Any], column: str, fallback_dir: pathlib.Path) -> list[dict[str, Any]]:
    preds = load_json(prediction_path(payload, column))
    ddir = data_path(payload, column, fallback_dir)
    rows: list[dict[str, Any]] = []
    for subtask in preds.keys():
        data = read_jsonl((ddir / subtask).with_suffix(".jsonl"))
        plist = pred_list(preds, subtask)
        for i, (pr, gold) in enumerate(zip(plist, data)):
            rows.append(make_item(
                column,
                subtask,
                i,
                pr.get("id", f"{subtask}_{i}"),
                pr.get("pred"),
                gold["sentence_good"],
                {k: gold.get(k) for k in ["field", "linguistics_term", "UID", "contrast", "pair_id", "row"] if k in gold},
            ))
    return rows


def ewok_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    preds = load_json(prediction_path(payload, "EWoK"))
    ddir = data_path(payload, "EWoK", PRISTINE_FULL / "ewok_filtered")
    rows: list[dict[str, Any]] = []
    for subtask in preds.keys():
        data = read_jsonl((ddir / subtask).with_suffix(".jsonl"))
        plist = pred_list(preds, subtask)
        for i, (pr, gold) in enumerate(zip(plist, data)):
            target = " ".join([gold["Context1"], gold["Target1"]]).strip()
            rows.append(make_item(
                "EWoK",
                subtask,
                i,
                pr.get("id", f"{subtask}_{i}"),
                pr.get("pred"),
                target,
                {k: gold.get(k) for k in ["Domain", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff"] if k in gold},
            ))
    return rows


def entity_targets(ddir: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    # The official generation pipeline drops examples whose options include a
    # literal "nothing" answer.  The patched REPRESENTATION_FRONTIER_STUDIES collator comments document the
    # same filtering before scoring.
    for p in sorted(ddir.glob("*.jsonl")):
        for gold in read_jsonl(p):
            if any("nothing" in str(opt) for opt in gold["options"]):
                continue
            out[f'{p.stem}_{gold["numops"]}_ops'].append(gold)
    return dict(out)


def entity_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    preds = load_json(prediction_path(payload, "Entity"))
    ddir = data_path(payload, "Entity", PRISTINE_FULL / "entity_tracking")
    targets = entity_targets(ddir)
    rows: list[dict[str, Any]] = []
    for subtask in preds.keys():
        plist = pred_list(preds, subtask)
        data = targets[subtask]
        for i, (pr, gold) in enumerate(zip(plist, data)):
            rows.append(make_item(
                "Entity",
                subtask,
                i,
                pr.get("id", f"{subtask}_{i}"),
                pr.get("pred"),
                gold["options"][0],
                {"numops": gold.get("numops"), "example_id": gold.get("example_id"), "sample_id": gold.get("sample_id")},
            ))
    return rows


def comps_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    preds = load_json(prediction_path(payload, "COMPS"))
    ddir = data_path(payload, "COMPS", PRISTINE_FULL / "comps")
    rows: list[dict[str, Any]] = []
    for subtask in preds.keys():
        data = read_jsonl((ddir / COMPS_FILE[subtask]).with_suffix(".jsonl"))
        plist = pred_list(preds, subtask)
        for i, (pr, gold) in enumerate(zip(plist, data)):
            target = " ".join([gold["prefix_acceptable"], gold["property_phrase"]]).strip()
            rows.append(make_item(
                "COMPS",
                subtask,
                i,
                pr.get("id", f"{subtask}_{i}"),
                pr.get("pred"),
                target,
                {k: gold.get(k) for k in ["condition", "pair_id", "property_phrase", "prefix_acceptable"] if k in gold},
            ))
    return rows


def globalpiqa_items(payload: dict[str, Any], column: str, default_task_name: str) -> list[dict[str, Any]]:
    preds = load_json(prediction_path(payload, column))
    ddir = data_path(payload, column, PRISTINE_FULL / default_task_name)
    data = read_jsonl(ddir / "eng_latn.jsonl")
    rows: list[dict[str, Any]] = []
    for i, gold in enumerate(data):
        eid = gold["example_id"]
        plist = pred_list(preds, eid)
        pr = plist[0]
        target = gold[f"solution{gold['label']}"]
        rows.append(make_item(
            column,
            default_task_name,
            i,
            eid,
            pr.get("pred"),
            target,
            {k: gold.get(k) for k in ["categories", "label", "language", "supplement"] if k in gold},
        ))
    return rows


def superglue_pred_path(payload: dict[str, Any], task: str) -> pathlib.Path:
    rec = task_rec(payload, "SuperGLUE")
    for tr in rec.get("tasks") or []:
        if isinstance(tr, dict) and tr.get("task") == task and tr.get("predictions"):
            return as_path(tr["predictions"])
    raise KeyError(f"SuperGLUE task missing predictions: {task}")


def superglue_items(payload: dict[str, Any], task: str) -> list[dict[str, Any]]:
    preds_json = load_json(superglue_pred_path(payload, task))
    preds = [int(x["pred"]) for x in preds_json[task]["predictions"]]
    data = read_jsonl(PRISTINE_FULL / "glue_filtered" / f"{task}.valid.jsonl")
    rows: list[dict[str, Any]] = []
    for i, (p, gold) in enumerate(zip(preds, data)):
        target = int(gold["label"])
        rows.append(make_item("SuperGLUE", task, i, gold.get("idx", gold.get("guid", f"{task}_{i}")), p, target, {}))
    return rows


def all_items(payload: dict[str, Any], include_superglue: bool) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for col, fallback in [("BLiMP", PRISTINE_FULL / "blimp_filtered"), ("Supplement", PRISTINE_FULL / "supplement_filtered")]:
        if col in (payload.get("tasks") or {}):
            out[col] = blimp_like_items(payload, col, fallback)
    if "EWoK" in (payload.get("tasks") or {}):
        out["EWoK"] = ewok_items(payload)
    if "Entity" in (payload.get("tasks") or {}):
        out["Entity"] = entity_items(payload)
    if "COMPS" in (payload.get("tasks") or {}):
        out["COMPS"] = comps_items(payload)
    if "GlobalPIQA_parallel" in (payload.get("tasks") or {}):
        out["GlobalPIQA_parallel"] = globalpiqa_items(payload, "GlobalPIQA_parallel", "global_piqa_parallel")
    if "GlobalPIQA_nonparallel" in (payload.get("tasks") or {}):
        out["GlobalPIQA_nonparallel"] = globalpiqa_items(payload, "GlobalPIQA_nonparallel", "global_piqa_nonparallel")
    if include_superglue and "SuperGLUE" in (payload.get("tasks") or {}):
        for task in SUPERGLUE_TASKS:
            out[f"SuperGLUE/{task}"] = superglue_items(payload, task)
    return out


def key_item(x: dict[str, Any]) -> tuple[str, int, str]:
    return (str(x["subtask"]), int(x["index"]), str(x.get("id", "")))


def binary_f1(labels: list[int], preds: list[int]) -> float:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else 100.0 * (2 * tp / den)


def acc(rows: list[dict[str, Any]]) -> float:
    return 100.0 * sum(int(bool(r["correct"])) for r in rows) / len(rows)


def subtask_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        groups[str(r["subtask"])].append(r)
    return dict(groups)


def payload_score(payload: dict[str, Any], column: str) -> Optional[float]:
    tasks = payload.get("tasks") or {}
    if column == "GlobalPIQA":
        vals = []
        for col in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            rec = tasks.get(col) or {}
            if rec.get("score") is not None:
                vals.append(float(rec["score"]))
        return sum(vals) / 2.0 if len(vals) == 2 else None
    if column == "Reading":
        sc = (tasks.get("Reading") or {}).get("scores") or {}
        return None if sc.get("Reading") is None else float(sc.get("Reading"))
    if column == "SuperGLUE":
        rec = tasks.get("SuperGLUE") or {}
        return None if rec.get("superglue_mean") is None else float(rec.get("superglue_mean"))
    if column == "AoA":
        rec = tasks.get("AoA") or {}
        if rec.get("aoa_leaderboard_score") is not None:
            return float(rec["aoa_leaderboard_score"])
        if rec.get("aoa_for_provisional_overall") is not None:
            return float(rec["aoa_for_provisional_overall"])
        return None
    rec = tasks.get(column) or {}
    return None if rec.get("score") is None else float(rec.get("score"))


def score_table(payload: dict[str, Any], computed: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA", "Reading", "SuperGLUE", "AoA"]:
        p = payload_score(payload, col)
        c = computed.get(col)
        out[col] = {
            "payload_score": p,
            "computed_from_predictions": c,
            "payload_minus_computed": None if p is None or c is None else p - float(c),
        }
    final: dict[str, float] = {}
    for col, rec in out.items():
        val = rec["computed_from_predictions"] if rec["computed_from_predictions"] is not None else rec["payload_score"]
        if val is not None:
            final[col] = float(val)
    if all(k in final for k in CHEAP7_KEYS):
        final["cheap7_mean"] = sum(final[k] for k in CHEAP7_KEYS) / len(CHEAP7_KEYS)
    provisional = dict(final)
    if "AoA" not in provisional and all(k in provisional for k in OFFICIAL9_KEYS if k != "AoA"):
        provisional["AoA"] = 0.0
    if all(k in provisional for k in OFFICIAL9_KEYS):
        final["projected_overall_with_aoa0_if_missing"] = sum(provisional[k] for k in OFFICIAL9_KEYS) / len(OFFICIAL9_KEYS)
    out["__final_values__"] = final
    return out


def compare_standard_column(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    amap = {key_item(x): x for x in a_rows}
    bmap = {key_item(x): x for x in b_rows}
    keys = sorted(set(amap) & set(bmap))
    missing = {"a_only": len(set(amap) - set(bmap)), "b_only": len(set(bmap) - set(amap))}
    sub: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {"n": 0, "a_correct": 0, "b_correct": 0, "b_minus_a": 0, "a_only_correct": 0, "b_only_correct": 0})
    examples_gain: list[dict[str, Any]] = []
    examples_loss: list[dict[str, Any]] = []
    both_correct = a_only = b_only = both_wrong = 0
    for k in keys:
        ar = amap[k]
        br = bmap[k]
        ac = bool(ar["correct"])
        bc = bool(br["correct"])
        if ac and bc:
            both_correct += 1
        elif ac and not bc:
            a_only += 1
            if len(examples_loss) < max_examples:
                examples_loss.append({"subtask": ar["subtask"], "id": ar.get("id"), "index": ar["index"], "target": ar["target"], "a_pred": ar["pred"], "b_pred": br["pred"], "metadata": ar.get("metadata", {})})
        elif (not ac) and bc:
            b_only += 1
            if len(examples_gain) < max_examples:
                examples_gain.append({"subtask": ar["subtask"], "id": ar.get("id"), "index": ar["index"], "target": ar["target"], "a_pred": ar["pred"], "b_pred": br["pred"], "metadata": ar.get("metadata", {})})
        else:
            both_wrong += 1
        sr = sub[str(ar["subtask"])]
        sr["n"] += 1
        sr["a_correct"] += int(ac)
        sr["b_correct"] += int(bc)
        sr["b_minus_a"] += int(bc) - int(ac)
        sr["a_only_correct"] += int(ac and not bc)
        sr["b_only_correct"] += int((not ac) and bc)
    sub_rows = []
    for name, sr in sub.items():
        nsub = sr["n"]
        a_acc = 100.0 * sr["a_correct"] / nsub if nsub else None
        b_acc = 100.0 * sr["b_correct"] / nsub if nsub else None
        sub_rows.append({
            "subtask": name,
            "n": nsub,
            "a_acc": a_acc,
            "b_acc": b_acc,
            "delta_acc_b_minus_a": None if a_acc is None or b_acc is None else b_acc - a_acc,
            "net_item_delta_b_minus_a": sr["b_minus_a"],
            "a_only_correct": sr["a_only_correct"],
            "b_only_correct": sr["b_only_correct"],
        })
    sub_rows.sort(key=lambda r: (abs(float(r["delta_acc_b_minus_a"])), r["n"]), reverse=True)
    n = len(keys)
    a_micro = 100.0 * (both_correct + a_only) / n if n else None
    b_micro = 100.0 * (both_correct + b_only) / n if n else None
    a_macro = sum(float(r["a_acc"]) for r in sub_rows) / len(sub_rows) if sub_rows else None
    b_macro = sum(float(r["b_acc"]) for r in sub_rows) / len(sub_rows) if sub_rows else None
    return {
        "n_common": n,
        "missing": missing,
        "a_accuracy_micro": a_micro,
        "b_accuracy_micro": b_micro,
        "delta_accuracy_micro_b_minus_a": None if a_micro is None or b_micro is None else b_micro - a_micro,
        "a_accuracy_official_macro": a_macro,
        "b_accuracy_official_macro": b_macro,
        "delta_accuracy_official_macro_b_minus_a": None if a_macro is None or b_macro is None else b_macro - a_macro,
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "net_item_delta_b_minus_a": b_only - a_only,
        "subtasks_by_abs_official_delta": sub_rows[:60],
        "example_gains_b_over_a": examples_gain,
        "example_losses_b_vs_a": examples_loss,
    }


def compare_superglue_task(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    base = compare_standard_column(a_rows, b_rows, max_examples)
    task = a_rows[0]["subtask"] if a_rows else b_rows[0]["subtask"]
    metric = SUPERGLUE_PRIMARY[str(task)]
    a_labels = [int(r["target"]) for r in a_rows]
    a_preds = [int(r["pred"]) for r in a_rows]
    b_preds = [int(r["pred"]) for r in b_rows]
    a_acc = 100.0 * sum(int(y == p) for y, p in zip(a_labels, a_preds)) / len(a_labels)
    b_acc = 100.0 * sum(int(y == p) for y, p in zip(a_labels, b_preds)) / len(a_labels)
    a_primary = binary_f1(a_labels, a_preds) if metric == "f1" else a_acc
    b_primary = binary_f1(a_labels, b_preds) if metric == "f1" else b_acc
    base.update({
        "primary_metric": metric,
        "a_primary_score": a_primary,
        "b_primary_score": b_primary,
        "delta_primary_b_minus_a": b_primary - a_primary,
        "a_accuracy": a_acc,
        "b_accuracy": b_acc,
        "delta_accuracy_b_minus_a": b_acc - a_acc,
    })
    return base


def computed_scores_from_comparisons(comps: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        if col in comps:
            out[col] = float(comps[col]["b_accuracy_official_macro"])
    if "GlobalPIQA_parallel" in out and "GlobalPIQA_nonparallel" in out:
        out["GlobalPIQA"] = (out["GlobalPIQA_parallel"] + out["GlobalPIQA_nonparallel"]) / 2.0
    sg_vals = []
    for task in SUPERGLUE_TASKS:
        key = f"SuperGLUE/{task}"
        if key in comps:
            sg_vals.append(float(comps[key]["b_primary_score"]))
    if len(sg_vals) == len(SUPERGLUE_TASKS):
        out["SuperGLUE"] = sum(sg_vals) / len(sg_vals)
    return out


def computed_scores_for_payload(items: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        if col in items:
            groups = subtask_groups(items[col])
            out[col] = sum(acc(g) for g in groups.values()) / len(groups)
    if "GlobalPIQA_parallel" in out and "GlobalPIQA_nonparallel" in out:
        out["GlobalPIQA"] = (out["GlobalPIQA_parallel"] + out["GlobalPIQA_nonparallel"]) / 2.0
    sg_vals = []
    for task in SUPERGLUE_TASKS:
        key = f"SuperGLUE/{task}"
        if key in items:
            rows = items[key]
            labels = [int(r["target"]) for r in rows]
            preds = [int(r["pred"]) for r in rows]
            a = 100.0 * sum(int(y == p) for y, p in zip(labels, preds)) / len(labels)
            score = binary_f1(labels, preds) if SUPERGLUE_PRIMARY[task] == "f1" else a
            out[f"SuperGLUE/{task}"] = score
            sg_vals.append(score)
    if len(sg_vals) == len(SUPERGLUE_TASKS):
        out["SuperGLUE"] = sum(sg_vals) / len(sg_vals)
    return out


def scores_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading", "SuperGLUE", "AoA"]:
        v = payload_score(payload, col)
        if v is not None:
            out[col] = float(v)
    if "GlobalPIQA_parallel" in out and "GlobalPIQA_nonparallel" in out:
        out["GlobalPIQA"] = (out["GlobalPIQA_parallel"] + out["GlobalPIQA_nonparallel"]) / 2.0
    if all(k in out for k in CHEAP7_KEYS):
        out["cheap7_mean"] = sum(out[k] for k in CHEAP7_KEYS) / len(CHEAP7_KEYS)
    provisional = dict(out)
    if "AoA" not in provisional and all(k in provisional for k in OFFICIAL9_KEYS if k != "AoA"):
        provisional["AoA"] = 0.0
    if all(k in provisional for k in OFFICIAL9_KEYS):
        out["projected_overall_with_aoa0_if_missing"] = sum(provisional[k] for k in OFFICIAL9_KEYS) / len(OFFICIAL9_KEYS)
    return out


def combined_score_summary(a_payload: dict[str, Any], b_payload: dict[str, Any], a_items: dict[str, list[dict[str, Any]]], b_items: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    a_payload_scores = scores_from_payload(a_payload)
    b_payload_scores = scores_from_payload(b_payload)
    a_comp = computed_scores_for_payload(a_items)
    b_comp = computed_scores_for_payload(b_items)
    # Reading and AoA are not item-accuracy columns; preserve payload values.
    for source, dest in [(a_payload_scores, a_comp), (b_payload_scores, b_comp)]:
        for col in ["Reading", "Reading_eye", "Reading_self_paced", "AoA"]:
            if col in source and col not in dest:
                dest[col] = source[col]
    for dest in [a_comp, b_comp]:
        if all(k in dest for k in CHEAP7_KEYS):
            dest["cheap7_mean"] = sum(dest[k] for k in CHEAP7_KEYS) / len(CHEAP7_KEYS)
        provisional = dict(dest)
        if "AoA" not in provisional and all(k in provisional for k in OFFICIAL9_KEYS if k != "AoA"):
            provisional["AoA"] = 0.0
        if all(k in provisional for k in OFFICIAL9_KEYS):
            dest["projected_overall_with_aoa0_if_missing"] = sum(provisional[k] for k in OFFICIAL9_KEYS) / len(OFFICIAL9_KEYS)
    keys = sorted(set(a_payload_scores) | set(b_payload_scores) | set(a_comp) | set(b_comp))
    rows = []
    for k in keys:
        rows.append({
            "column": k,
            "a_payload": a_payload_scores.get(k),
            "b_payload": b_payload_scores.get(k),
            "a_computed": a_comp.get(k),
            "b_computed": b_comp.get(k),
            "payload_delta_b_minus_a": None if a_payload_scores.get(k) is None or b_payload_scores.get(k) is None else b_payload_scores[k] - a_payload_scores[k],
            "computed_delta_b_minus_a": None if a_comp.get(k) is None or b_comp.get(k) is None else b_comp[k] - a_comp[k],
            "a_payload_minus_computed": None if a_payload_scores.get(k) is None or a_comp.get(k) is None else a_payload_scores[k] - a_comp[k],
            "b_payload_minus_computed": None if b_payload_scores.get(k) is None or b_comp.get(k) is None else b_payload_scores[k] - b_comp[k],
        })
    return {"rows": rows, "a_payload_scores": a_payload_scores, "b_payload_scores": b_payload_scores, "a_computed_scores": a_comp, "b_computed_scores": b_comp}


def write_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    lines: list[str] = []
    a = result["a_label"]
    b = result["b_label"]
    lines.append(f"# research official-coordinate transition comparison: {b} vs {a}\n")
    lines.append("## Score summary\n")
    lines.append("Scores labelled `computed` are recomputed from prediction files using the official macro over subtasks or SuperGLUE primary metrics where applicable. `payload` is the saved evaluator score. Item flips below are micro counts and should not be read as official score movement when subtasks have unequal sizes.\n")
    lines.append("| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A | A payload-computed | B payload-computed |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    preferred_order = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA", "Reading", "SuperGLUE", "AoA", "cheap7_mean", "projected_overall_with_aoa0_if_missing"] + [f"SuperGLUE/{t}" for t in SUPERGLUE_TASKS]
    rows_by_col = {r["column"]: r for r in result["score_summary"]["rows"]}
    for k in preferred_order:
        if k not in rows_by_col:
            continue
        r = rows_by_col[k]
        def fmt(x: Any) -> str:
            return "" if x is None else f"{float(x):.6f}"
        def fmtd(x: Any) -> str:
            return "" if x is None else f"{float(x):+.6f}"
        lines.append(f"| {k} | {fmt(r['a_payload'])} | {fmt(r['b_payload'])} | {fmtd(r['payload_delta_b_minus_a'])} | {fmt(r['a_computed'])} | {fmt(r['b_computed'])} | {fmtd(r['computed_delta_b_minus_a'])} | {fmtd(r['a_payload_minus_computed'])} | {fmtd(r['b_payload_minus_computed'])} |")

    lines.append("\n## Prediction-level transitions\n")
    lines.append("| column | n | official-macro A | official-macro B | official-macro B-A | micro A | micro B | micro B-A | B-only | A-only | net item |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for col, rec in result["column_comparisons"].items():
        if col.startswith("SuperGLUE/"):
            continue
        lines.append(f"| {col} | {rec['n_common']} | {rec['a_accuracy_official_macro']:.6f} | {rec['b_accuracy_official_macro']:.6f} | {rec['delta_accuracy_official_macro_b_minus_a']:+.6f} | {rec['a_accuracy_micro']:.6f} | {rec['b_accuracy_micro']:.6f} | {rec['delta_accuracy_micro_b_minus_a']:+.6f} | {rec['b_only_correct']} | {rec['a_only_correct']} | {rec['net_item_delta_b_minus_a']:+d} |")
    if result.get("include_superglue"):
        lines.append("\n## SuperGLUE task transitions\n")
        lines.append("| task | metric | n | A primary | B primary | B-A primary | A acc | B acc | B-A acc | B-only | A-only | net item |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for task in SUPERGLUE_TASKS:
            rec = result["column_comparisons"].get(f"SuperGLUE/{task}")
            if not rec:
                continue
            lines.append(f"| {task} | {rec['primary_metric']} | {rec['n_common']} | {rec['a_primary_score']:.6f} | {rec['b_primary_score']:.6f} | {rec['delta_primary_b_minus_a']:+.6f} | {rec['a_accuracy']:.6f} | {rec['b_accuracy']:.6f} | {rec['delta_accuracy_b_minus_a']:+.6f} | {rec['b_only_correct']} | {rec['a_only_correct']} | {rec['net_item_delta_b_minus_a']:+d} |")

    lines.append("\n## Largest subtask movements by official-macro component\n")
    for col, rec in result["column_comparisons"].items():
        if col.startswith("SuperGLUE/"):
            continue
        lines.append(f"\n### {col}\n")
        lines.append("| subtask | n | A acc | B acc | B-A acc | net items |\n|---|---:|---:|---:|---:|---:|")
        for sr in rec["subtasks_by_abs_official_delta"][:15]:
            lines.append(f"| {sr['subtask']} | {sr['n']} | {sr['a_acc']:.6f} | {sr['b_acc']:.6f} | {sr['delta_acc_b_minus_a']:+.6f} | {sr['net_item_delta_b_minus_a']:+d} |")

    lines.append("\n## Interpretation note\n")
    lines.append(result.get("interpretation_note", ""))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a-label", required=True)
    ap.add_argument("--a-payload", nargs="+", required=True)
    ap.add_argument("--b-label", required=True)
    ap.add_argument("--b-payload", nargs="+", required=True)
    ap.add_argument("--out-root", default="experiments/archive/functional_learning/data/official_transition_compare")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--include-superglue", action="store_true")
    ap.add_argument("--max-examples", type=int, default=20)
    args = ap.parse_args()

    a_payload = load_payloads(args.a_payload)
    b_payload = load_payloads(args.b_payload)
    a_items = all_items(a_payload, args.include_superglue)
    b_items = all_items(b_payload, args.include_superglue)
    common_cols = [c for c in a_items.keys() if c in b_items]
    comparisons: dict[str, Any] = {}
    for col in common_cols:
        if col.startswith("SuperGLUE/"):
            comparisons[col] = compare_superglue_task(a_items[col], b_items[col], args.max_examples)
        else:
            comparisons[col] = compare_standard_column(a_items[col], b_items[col], args.max_examples)
    result = {
        "status": "OFFICIAL_TRANSITION_COMPARE",
        "a_label": args.a_label,
        "b_label": args.b_label,
        "a_payloads": [rel(as_path(p)) for p in args.a_payload],
        "b_payloads": [rel(as_path(p)) for p in args.b_payload],
        "include_superglue": bool(args.include_superglue),
        "score_summary": combined_score_summary(a_payload, b_payload, a_items, b_items),
        "column_comparisons": comparisons,
        "interpretation_note": "Official BabyLM columns use macro averages over subtasks and SuperGLUE primary metrics; micro item flips diagnose which decisions changed but do not directly equal leaderboard movement. Treat deltas as measurement evidence, not as a causal mechanism without training provenance and controls.",
    }
    out_root = as_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / f"{args.tag}_official_transition.json"
    out_md = out_root / f"{args.tag}_official_transition.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps({"status": result["status"], "tag": args.tag, "out_json": rel(out_json), "out_md": rel(out_md), "columns": len(comparisons)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
