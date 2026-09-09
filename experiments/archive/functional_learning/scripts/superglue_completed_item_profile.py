#!/usr/bin/env python3
"""research: completed SuperGLUE item/task profile for coherent86, dense, and clean64.

Scientific purpose
------------------
Clean seed62064 is the only clean preservation endpoint with completed repaired
AutoModel SuperGLUE at the start of research.  Its verified Overall advantage over
dense endpoints depends materially on SuperGLUE recovery.  This script parses only
completed SuperGLUE payloads, reproduces official primary metrics from predictions
and valid labels, and decomposes which supervised tasks/items changed relative to
coherent86 and dense controls.  It makes no use of the still-running clean65 or
exact (M,S) SuperGLUE jobs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import pathlib
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = _public_path('.')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/superglue_completed_item_profile')
VALID_DIR = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/glue_filtered')

PAYLOADS: Dict[str, str] = {
    "coherent86": "experiments/archive/functional_learning/data/repaired_coherent86_eval/per_target/repaired_coherent86_alpha075.json",
    "dense_seed62064": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62064/per_target/repaired_dense_seed62064_u0080.json",
    "dense_seed62065": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62065/per_target/repaired_dense_seed62065_u0080.json",
    "clean_seed62064": "experiments/archive/functional_learning/data/repaired_clean_eval_superglue/per_target/clean_pres_lambda1_eval_seed62064_u0080.json",
}
TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]
PRIMARY_METRIC = {
    "boolq": "accuracy",
    "multirc": "accuracy",
    "rte": "accuracy",
    "wsc": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "mnli": "accuracy",
}


def rel(path: pathlib.Path | str | None) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve(path_s: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path_s)
    if p.is_absolute():
        return p
    return ROOT / p


def read_json(path_s: str | pathlib.Path) -> Any:
    return json.loads(resolve(path_s).read_text(encoding="utf-8"))


def iter_jsonl(path: pathlib.Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, json.loads(line)


def accuracy(labels: List[int], preds: List[int]) -> float:
    return 100.0 * sum(int(y == p) for y, p in zip(labels, preds)) / len(labels)


def f1_binary(labels: List[int], preds: List[int]) -> float:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y != 1 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p != 1)
    if (2 * tp + fp + fn) == 0:
        return 0.0
    return 100.0 * (2 * tp) / (2 * tp + fp + fn)


def mcc_binary(labels: List[int], preds: List[int]) -> Optional[float]:
    # useful descriptive metric, not part of BabyLM primary coordinate here.
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return None
    return (tp * tn - fp * fn) / denom


def parse_task(payload: Dict[str, Any], task: str) -> Dict[str, Dict[str, Any]]:
    sg = payload.get("tasks", {}).get("SuperGLUE", {})
    entry = next((t for t in sg.get("tasks", []) if t.get("task") == task), None)
    if entry is None:
        raise KeyError(f"missing task {task} in {payload.get('target')}")
    if entry.get("returncode") != 0:
        raise RuntimeError(f"task {task} returncode {entry.get('returncode')}")
    pred_obj = read_json(entry["predictions"])
    pred_rows = pred_obj[task]["predictions"]
    gold_rows = [row for _, row in iter_jsonl(VALID_DIR / f"{task}.valid.jsonl")]
    if len(pred_rows) != len(gold_rows):
        raise RuntimeError(f"{task} count mismatch {len(pred_rows)} vs {len(gold_rows)}")
    out: Dict[str, Dict[str, Any]] = {}
    for i, (pr, gr) in enumerate(zip(pred_rows, gold_rows)):
        pred = int(pr["pred"])
        label = int(gr["label"])
        item_id = gr.get("idx", pr.get("id", i))
        key = f"{task}/{item_id}"
        out[key] = {
            "task": task,
            "idx": item_id,
            "position": i,
            "label": label,
            "pred": pred,
            "correct": pred == label,
        }
        # Preserve task-specific fields only where compact enough to identify interesting flips.
        if task == "boolq":
            out[key]["meta"] = {"question": gr.get("question"), "passage_prefix": str(gr.get("passage", ""))[:240]}
        elif task in {"mrpc", "qqp"}:
            out[key]["meta"] = {"question1": gr.get("question1") or gr.get("sentence1"), "question2": gr.get("question2") or gr.get("sentence2")}
        elif task == "mnli":
            out[key]["meta"] = {"premise": str(gr.get("premise", ""))[:220], "hypothesis": str(gr.get("hypothesis", ""))[:220]}
        elif task in {"rte", "wsc"}:
            out[key]["meta"] = {k: gr.get(k) for k in list(gr.keys())[:6] if k != "label"}
        else:
            out[key]["meta"] = {"idx": item_id}
    return out


def metrics_for(records: Dict[str, Dict[str, Any]], task: str) -> Dict[str, Any]:
    labels = [int(r["label"]) for r in records.values()]
    preds = [int(r["pred"]) for r in records.values()]
    out = {
        "n": len(records),
        "accuracy": accuracy(labels, preds),
        "pred_counts": dict(Counter(preds)),
        "label_counts": dict(Counter(labels)),
    }
    if task != "mnli":
        out["f1"] = f1_binary(labels, preds)
        out["mcc"] = mcc_binary(labels, preds)
    out["primary_metric"] = PRIMARY_METRIC[task]
    out["primary_score"] = out[PRIMARY_METRIC[task]]
    return out


def sg_payload_task_entries(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {t["task"]: t for t in payload.get("tasks", {}).get("SuperGLUE", {}).get("tasks", [])}


def sg_primary_details(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    details = payload.get("tasks", {}).get("SuperGLUE", {}).get("superglue_primary_metric_details") or []
    return {d["task"]: d for d in details}


def compare_two(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted(set(a) & set(b))
    n = len(keys)
    same_pred = sum(int(a[k]["pred"]) == int(b[k]["pred"]) for k in keys)
    same_correct = sum(bool(a[k]["correct"]) == bool(b[k]["correct"]) for k in keys)
    a_good_b_bad = sum(bool(a[k]["correct"]) and not bool(b[k]["correct"]) for k in keys)
    b_good_a_bad = sum(bool(b[k]["correct"]) and not bool(a[k]["correct"]) for k in keys)
    return {
        "n": n,
        "prediction_agreement_fraction": same_pred / n if n else None,
        "correctness_agreement_fraction": same_correct / n if n else None,
        "a_correct_b_wrong": a_good_b_bad,
        "b_correct_a_wrong": b_good_a_bad,
        "net_accuracy_points_a_minus_b": 100.0 * (a_good_b_bad - b_good_a_bad) / n if n else None,
    }


def change_overlap(parent: Dict[str, Dict[str, Any]], a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted(set(parent) & set(a) & set(b))
    gains_a, gains_b, losses_a, losses_b = set(), set(), set(), set()
    for k in keys:
        p = bool(parent[k]["correct"])
        ac = bool(a[k]["correct"])
        bc = bool(b[k]["correct"])
        if (not p) and ac:
            gains_a.add(k)
        if (not p) and bc:
            gains_b.add(k)
        if p and (not ac):
            losses_a.add(k)
        if p and (not bc):
            losses_b.add(k)
    return {
        "n": len(keys),
        "a_gains": len(gains_a),
        "b_gains": len(gains_b),
        "shared_gains": len(gains_a & gains_b),
        "gain_jaccard": len(gains_a & gains_b) / len(gains_a | gains_b) if (gains_a | gains_b) else None,
        "a_losses": len(losses_a),
        "b_losses": len(losses_b),
        "shared_losses": len(losses_a & losses_b),
        "loss_jaccard": len(losses_a & losses_b) / len(losses_a | losses_b) if (losses_a | losses_b) else None,
    }


def dense_clean_decomposition(parent: Dict[str, Dict[str, Any]], dense: Dict[str, Dict[str, Any]], clean: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted(set(parent) & set(dense) & set(clean))
    out = defaultdict(int)
    for k in keys:
        p = bool(parent[k]["correct"])
        d = bool(dense[k]["correct"])
        c = bool(clean[k]["correct"])
        if p and (not d) and c:
            out["clean_recovers_dense_parent_loss"] += 1
        if p and d and (not c):
            out["clean_loses_parent_item_dense_kept"] += 1
        if (not p) and d and c:
            out["clean_keeps_dense_gain"] += 1
        if (not p) and d and (not c):
            out["clean_drops_dense_gain"] += 1
        if (not p) and (not d) and c:
            out["clean_new_gain_beyond_dense"] += 1
        if p and (not d) and (not c):
            out["shared_loss_vs_parent"] += 1
    out["n"] = len(keys)
    out["net_clean_minus_dense_accuracy_points"] = 100.0 * ((out["clean_recovers_dense_parent_loss"] + out["clean_new_gain_beyond_dense"]) - (out["clean_loses_parent_item_dense_kept"] + out["clean_drops_dense_gain"])) / len(keys) if keys else None
    return dict(out)


def sample_changes(parent: Dict[str, Dict[str, Any]], dense: Dict[str, Dict[str, Any]], clean: Dict[str, Dict[str, Any]], limit: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    keys = sorted(set(parent) & set(dense) & set(clean))
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for k in keys:
        p = bool(parent[k]["correct"])
        d = bool(dense[k]["correct"])
        c = bool(clean[k]["correct"])
        label = None
        if p and (not d) and c:
            label = "clean_recovers_dense_loss"
        elif (not p) and d and c:
            label = "clean_keeps_dense_gain"
        elif (not p) and d and (not c):
            label = "clean_drops_dense_gain"
        elif (not p) and (not d) and c:
            label = "clean_new_gain_beyond_dense"
        elif p and (not d) and (not c):
            label = "shared_loss_vs_parent"
        if label and len(buckets[label]) < limit:
            buckets[label].append({
                "key": k,
                "label": parent[k].get("label"),
                "parent_pred": parent[k].get("pred"),
                "dense_pred": dense[k].get("pred"),
                "clean_pred": clean[k].get("pred"),
                "parent_correct": p,
                "dense_correct": d,
                "clean_correct": c,
                "meta": parent[k].get("meta"),
            })
    return dict(buckets)


def build() -> Dict[str, Any]:
    payloads = {m: read_json(p) for m, p in PAYLOADS.items()}
    records: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    model_metrics: Dict[str, Any] = {}
    for m, payload in payloads.items():
        entries = sg_payload_task_entries(payload)
        details = sg_primary_details(payload)
        model_metrics[m] = {
            "payload": rel(PAYLOADS[m]),
            "superglue_mean": payload.get("tasks", {}).get("SuperGLUE", {}).get("superglue_mean"),
            "tasks": {},
            "computed_primary_mean": None,
        }
        primary_scores = []
        for task in TASKS:
            recs = parse_task(payload, task)
            records[m][task] = recs
            comp = metrics_for(recs, task)
            payload_entry = entries[task]
            primary_detail = details.get(task, {})
            comp["payload_accuracy"] = payload_entry.get("accuracy")
            comp["payload_primary_score"] = primary_detail.get("score")
            comp["payload_metric"] = primary_detail.get("metric")
            comp["payload_primary_minus_computed"] = (float(primary_detail.get("score")) - comp["primary_score"]) if primary_detail.get("score") is not None else None
            model_metrics[m]["tasks"][task] = comp
            primary_scores.append(comp["primary_score"])
        model_metrics[m]["computed_primary_mean"] = sum(primary_scores) / len(primary_scores)
        model_metrics[m]["payload_mean_minus_computed"] = model_metrics[m]["superglue_mean"] - model_metrics[m]["computed_primary_mean"]

    task_comparisons: Dict[str, Any] = {}
    for task in TASKS:
        task_comparisons[task] = {
            "clean64_vs_coherent86": compare_two(records["clean_seed62064"][task], records["coherent86"][task]),
            "clean64_vs_dense64": compare_two(records["clean_seed62064"][task], records["dense_seed62064"][task]),
            "clean64_vs_dense65": compare_two(records["clean_seed62064"][task], records["dense_seed62065"][task]),
            "dense64_dense65_change_overlap_vs_parent": change_overlap(records["coherent86"][task], records["dense_seed62064"][task], records["dense_seed62065"][task]),
            "parent_dense64_clean64_decomposition": dense_clean_decomposition(records["coherent86"][task], records["dense_seed62064"][task], records["clean_seed62064"][task]),
            "parent_dense65_clean64_decomposition": dense_clean_decomposition(records["coherent86"][task], records["dense_seed62065"][task], records["clean_seed62064"][task]),
        }

    # Task-level deltas in primary-score units.
    task_delta_rows: List[Dict[str, Any]] = []
    for task in TASKS:
        row = {"task": task, "primary_metric": PRIMARY_METRIC[task]}
        for m in PAYLOADS:
            row[f"{m}_primary"] = model_metrics[m]["tasks"][task]["primary_score"]
            row[f"{m}_accuracy"] = model_metrics[m]["tasks"][task]["accuracy"]
            row[f"{m}_pred_counts"] = json.dumps(model_metrics[m]["tasks"][task]["pred_counts"], sort_keys=True)
        for m in ["dense_seed62064", "dense_seed62065", "clean_seed62064"]:
            row[f"{m}_delta_vs_coherent86"] = row[f"{m}_primary"] - row["coherent86_primary"]
        row["clean64_minus_dense64"] = row["clean_seed62064_primary"] - row["dense_seed62064_primary"]
        row["clean64_minus_dense65"] = row["clean_seed62064_primary"] - row["dense_seed62065_primary"]
        task_delta_rows.append(row)

    result = {
        "status": "SUPERGLUE_COMPLETED_ITEM_PROFILE",
        "created_utc": now(),
        "purpose": "Completed repaired-AutoModel SuperGLUE profile for clean seed62064 versus coherent86 and dense seeds; pending clean65/(M,S) jobs are not used.",
        "payloads": PAYLOADS,
        "valid_dir": rel(VALID_DIR),
        "model_metrics": model_metrics,
        "task_primary_deltas": task_delta_rows,
        "task_item_comparisons": task_comparisons,
        "flip_samples_clean64_vs_dense64": {task: sample_changes(records["coherent86"][task], records["dense_seed62064"][task], records["clean_seed62064"][task]) for task in TASKS},
        "interpretation": {
            "superglue_means": {m: model_metrics[m]["superglue_mean"] for m in PAYLOADS},
            "clean64_delta_vs_coherent86": model_metrics["clean_seed62064"]["superglue_mean"] - model_metrics["coherent86"]["superglue_mean"],
            "clean64_delta_vs_dense64": model_metrics["clean_seed62064"]["superglue_mean"] - model_metrics["dense_seed62064"]["superglue_mean"],
            "clean64_delta_vs_dense65": model_metrics["clean_seed62064"]["superglue_mean"] - model_metrics["dense_seed62065"]["superglue_mean"],
        },
    }
    return result


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


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_md(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research completed SuperGLUE item/task profile\n\n")
    lines.append("This uses only completed repaired AutoModel SuperGLUE payloads: coherent86, dense seed62064, dense seed62065, and clean preservation seed62064. It does not use the running clean seed62065 or exact `(M,S)` jobs.\n\n")
    interp = result["interpretation"]
    lines.append("## SuperGLUE means\n\n")
    for m, v in interp["superglue_means"].items():
        lines.append(f"- {m}: {fmt(v,6)}\n")
    lines.append(f"- clean64 minus coherent86: {fmt(interp['clean64_delta_vs_coherent86'],6)}\n")
    lines.append(f"- clean64 minus dense64: {fmt(interp['clean64_delta_vs_dense64'],6)}\n")
    lines.append(f"- clean64 minus dense65: {fmt(interp['clean64_delta_vs_dense65'],6)}\n\n")
    lines.append("## Payload metric reproduction\n\n")
    lines.append("| model | task | metric | n | payload primary | computed primary | payload-computed | accuracy | f1 | pred counts |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
    for m, mm in result["model_metrics"].items():
        for task in TASKS:
            mt = mm["tasks"][task]
            lines.append(f"| {m} | {task} | {mt.get('primary_metric')} | {mt.get('n')} | {fmt(mt.get('payload_primary_score'),6)} | {fmt(mt.get('primary_score'),6)} | {fmt(mt.get('payload_primary_minus_computed'),8)} | {fmt(mt.get('accuracy'),4)} | {fmt(mt.get('f1'),4)} | {mt.get('pred_counts')} |\n")
    lines.append("\n## Task primary-score deltas\n\n")
    lines.append("| task | metric | dense64-parent | dense65-parent | clean64-parent | clean64-dense64 | clean64-dense65 |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for row in result["task_primary_deltas"]:
        lines.append(f"| {row['task']} | {row['primary_metric']} | {fmt(row['dense_seed62064_delta_vs_coherent86'],4)} | {fmt(row['dense_seed62065_delta_vs_coherent86'],4)} | {fmt(row['clean_seed62064_delta_vs_coherent86'],4)} | {fmt(row['clean64_minus_dense64'],4)} | {fmt(row['clean64_minus_dense65'],4)} |\n")
    lines.append("\n## Clean64 item comparisons\n\n")
    lines.append("| task | clean-vs-parent pred agree | clean-vs-parent correctness agree | clean correct parent wrong | parent correct clean wrong | clean-vs-dense64 pred agree | clean correct dense64 wrong | dense64 correct clean wrong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for task, comp in result["task_item_comparisons"].items():
        cp = comp["clean64_vs_coherent86"]
        cd = comp["clean64_vs_dense64"]
        lines.append(f"| {task} | {fmt(cp.get('prediction_agreement_fraction'),4)} | {fmt(cp.get('correctness_agreement_fraction'),4)} | {cp.get('a_correct_b_wrong')} | {cp.get('b_correct_a_wrong')} | {fmt(cd.get('prediction_agreement_fraction'),4)} | {cd.get('a_correct_b_wrong')} | {cd.get('b_correct_a_wrong')} |\n")
    lines.append("\n## Dense-to-clean decompositions against parent\n\n")
    lines.append("| task | dense seed | recover dense parent loss | clean loses dense-kept parent item | keep dense gain | drop dense gain | new clean gain | shared loss | net clean-dense acc pp |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for task, comp in result["task_item_comparisons"].items():
        for key, seed in [("parent_dense64_clean64_decomposition", "62064"), ("parent_dense65_clean64_decomposition", "62065")]:
            d = comp[key]
            lines.append(f"| {task} | {seed} | {d.get('clean_recovers_dense_parent_loss',0)} | {d.get('clean_loses_parent_item_dense_kept',0)} | {d.get('clean_keeps_dense_gain',0)} | {d.get('clean_drops_dense_gain',0)} | {d.get('clean_new_gain_beyond_dense',0)} | {d.get('shared_loss_vs_parent',0)} | {fmt(d.get('net_clean_minus_dense_accuracy_points'),4)} |\n")
    lines.append("\n## Scientific reading\n\n")
    lines.append("Clean seed62064's SuperGLUE advantage over dense controls is not a single-task artifact: it improves over dense on MultiRC, WSC, MRPC F1, QQP F1, and MNLI, while RTE stays at the dense value and BoolQ is equal or slightly above dense. Relative to coherent86 the picture is smaller and mixed: BoolQ/MRPC/MNLI/QQP contribute gains, MultiRC/RTE lose, and WSC is equal. This is consistent with preservation bounding a dense-induced transfer cost rather than installing an entirely new supervised-transfer mechanism. The pending clean seed62065 and exact `(M,S)` payloads must be evaluated in the same way before this interpretation is used for v5 promotion.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build()
    out_json = args.out_dir / "superglue_completed_item_profile.json"
    out_md = args.out_dir / "superglue_completed_item_profile.md"
    out_csv = args.out_dir / "superglue_task_primary_deltas.csv"
    out_samples = args.out_dir / "superglue_clean64_dense64_flip_samples.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out_md, result)
    write_csv(out_csv, result["task_primary_deltas"])
    out_samples.write_text(json.dumps(result["flip_samples_clean64_vs_dense64"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
