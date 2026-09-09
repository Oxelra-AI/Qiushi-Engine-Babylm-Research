#!/usr/bin/env python3
"""research paired item-transition comparison for BabyLM official-style outputs.

The script compares two evaluated checkpoints at the prediction level. It is used
here to separate a private-scale readout change from additional training and, once
pa87 official-style predictions arrive, to identify which task families gained or
lost items relative to coherent86.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import json
import math
import pathlib
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
PRISTINE_FULL = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
GLOBALPIQA_FULL = ROOT / "experiments/archive/representation_and_objectives/data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval"
SUPERGLUE_TASKS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]
SUPERGLUE_PRIMARY = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}
COMPS_FILE = {"base": "comps_base", "wugs_dist_before": "comps_wugs_dist-before", "wugs_dist_in_between": "comps_wugs_dist-in-between", "wugs": "comps_wugs"}
ZERO_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> List[dict[str, Any]]:
    out = []
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
    merged: dict[str, Any] = {"tasks": {}}
    for pstr in paths:
        if not pstr:
            continue
        payload = load_json(pathlib.Path(pstr))
        for k, v in (payload.get("tasks") or {}).items():
            merged["tasks"][k] = v
        for k in ["target", "model_path", "run_dir", "endpoint", "official_overall"]:
            if k in payload and k not in merged:
                merged[k] = payload[k]
    return merged


def prediction_path(payload: dict[str, Any], column: str) -> pathlib.Path:
    rec = (payload.get("tasks") or {}).get(column)
    if not isinstance(rec, dict):
        raise KeyError(f"payload missing task {column}")
    p = rec.get("predictions")
    if not p:
        raise KeyError(f"task {column} lacks predictions path")
    return pathlib.Path(p)


def blimp_like_items(payload: dict[str, Any], column: str, data_dir: pathlib.Path) -> list[dict[str, Any]]:
    preds = load_json(prediction_path(payload, column))
    rows = []
    for subtask, block in preds.items():
        data = read_jsonl((data_dir / subtask).with_suffix(".jsonl"))
        plist = pred_list(preds, subtask)
        for i, (pr, gold) in enumerate(zip(plist, data)):
            rows.append({
                "column": column,
                "subtask": subtask,
                "index": i,
                "id": pr.get("id", f"{subtask}_{i}"),
                "pred": clean(pr.get("pred")),
                "target": clean(gold["sentence_good"]),
                "correct": clean(pr.get("pred")) == clean(gold["sentence_good"]),
                "metadata": {k: gold.get(k) for k in ["field", "linguistics_term", "UID", "contrast", "pair_id", "row"] if k in gold},
            })
    return rows


def ewok_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    preds = load_json(prediction_path(payload, "EWoK"))
    rows = []
    for subtask, block in preds.items():
        data = read_jsonl((PRISTINE_FULL / "ewok_filtered" / subtask).with_suffix(".jsonl"))
        plist = pred_list(preds, subtask)
        for i, (pr, gold) in enumerate(zip(plist, data)):
            target = " ".join([gold["Context1"], gold["Target1"]]).strip()
            rows.append({
                "column": "EWoK",
                "subtask": subtask,
                "index": i,
                "id": pr.get("id", f"{subtask}_{i}"),
                "pred": clean(pr.get("pred")),
                "target": target,
                "correct": clean(pr.get("pred")) == target,
                "metadata": {k: gold.get(k) for k in ["Domain", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff"] if k in gold},
            })
    return rows


def entity_targets() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for p in sorted((PRISTINE_FULL / "entity_tracking").glob("*.jsonl")):
        for gold in read_jsonl(p):
            if any("nothing" in str(opt) for opt in gold["options"]):
                continue
            out[f'{p.stem}_{gold["numops"]}_ops'].append(gold)
    return dict(out)


def entity_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    preds = load_json(prediction_path(payload, "Entity"))
    targets = entity_targets()
    rows = []
    for subtask, block in preds.items():
        plist = pred_list(preds, subtask)
        data = targets[subtask]
        for i, (pr, gold) in enumerate(zip(plist, data)):
            target = clean(gold["options"][0])
            rows.append({
                "column": "Entity",
                "subtask": subtask,
                "index": i,
                "id": pr.get("id", f"{subtask}_{i}"),
                "pred": clean(pr.get("pred")),
                "target": target,
                "correct": clean(pr.get("pred")) == target,
                "metadata": {"numops": gold.get("numops"), "example_id": gold.get("example_id"), "sample_id": gold.get("sample_id")},
            })
    return rows


def comps_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    preds = load_json(prediction_path(payload, "COMPS"))
    rows = []
    for subtask, block in preds.items():
        data = read_jsonl((PRISTINE_FULL / "comps" / COMPS_FILE[subtask]).with_suffix(".jsonl"))
        plist = pred_list(preds, subtask)
        for i, (pr, gold) in enumerate(zip(plist, data)):
            target = " ".join([gold["prefix_acceptable"], gold["property_phrase"]]).strip()
            rows.append({
                "column": "COMPS",
                "subtask": subtask,
                "index": i,
                "id": pr.get("id", f"{subtask}_{i}"),
                "pred": clean(pr.get("pred")),
                "target": target,
                "correct": clean(pr.get("pred")) == target,
                "metadata": {k: gold.get(k) for k in ["condition", "pair_id", "property_phrase", "prefix_acceptable"] if k in gold},
            })
    return rows


def globalpiqa_items(payload: dict[str, Any], column: str, task_name: str) -> list[dict[str, Any]]:
    preds = load_json(prediction_path(payload, column))
    data = read_jsonl(GLOBALPIQA_FULL / task_name / "eng_latn.jsonl")
    rows = []
    for i, gold in enumerate(data):
        eid = gold["example_id"]
        plist = pred_list(preds, eid)
        pr = plist[0]
        target = clean(gold[f"solution{gold['label']}"])
        rows.append({
            "column": column,
            "subtask": task_name,
            "index": i,
            "id": eid,
            "pred": clean(pr.get("pred")),
            "target": target,
            "correct": clean(pr.get("pred")) == target,
            "metadata": {k: gold.get(k) for k in ["categories", "label", "language", "supplement"] if k in gold},
        })
    return rows


def superglue_pred_path(payload: dict[str, Any], task: str) -> pathlib.Path:
    rec = (payload.get("tasks") or {}).get("SuperGLUE")
    if not isinstance(rec, dict):
        raise KeyError("payload missing SuperGLUE")
    for tr in rec.get("tasks") or []:
        if isinstance(tr, dict) and tr.get("task") == task and tr.get("predictions"):
            return pathlib.Path(tr["predictions"])
    raise KeyError(f"SuperGLUE task missing predictions: {task}")


def binary_f1(labels: list[int], preds: list[int]) -> float:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else 100.0 * (2 * tp / den)


def superglue_items(payload: dict[str, Any], task: str) -> list[dict[str, Any]]:
    preds_json = load_json(superglue_pred_path(payload, task))
    preds = [int(x["pred"]) for x in preds_json[task]["predictions"]]
    data = read_jsonl(PRISTINE_FULL / "glue_filtered" / f"{task}.valid.jsonl")
    rows = []
    for i, (p, gold) in enumerate(zip(preds, data)):
        target = int(gold["label"])
        rows.append({
            "column": "SuperGLUE",
            "subtask": task,
            "index": i,
            "id": gold.get("idx", gold.get("guid", f"{task}_{i}")),
            "pred": p,
            "target": target,
            "correct": p == target,
            "metadata": {},
        })
    return rows


def all_items(payload: dict[str, Any], include_superglue: bool) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    out["BLiMP"] = blimp_like_items(payload, "BLiMP", PRISTINE_FULL / "blimp_filtered")
    out["Supplement"] = blimp_like_items(payload, "Supplement", PRISTINE_FULL / "supplement_filtered")
    out["EWoK"] = ewok_items(payload)
    out["Entity"] = entity_items(payload)
    out["COMPS"] = comps_items(payload)
    out["GlobalPIQA_parallel"] = globalpiqa_items(payload, "GlobalPIQA_parallel", "global_piqa_parallel")
    out["GlobalPIQA_nonparallel"] = globalpiqa_items(payload, "GlobalPIQA_nonparallel", "global_piqa_nonparallel")
    if include_superglue:
        for task in SUPERGLUE_TASKS:
            out[f"SuperGLUE/{task}"] = superglue_items(payload, task)
    return out


def key_item(x: dict[str, Any]) -> tuple[str, int, str]:
    return (str(x["subtask"]), int(x["index"]), str(x.get("id", "")))


def compare_column(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    amap = {key_item(x): x for x in a_rows}
    bmap = {key_item(x): x for x in b_rows}
    keys = sorted(set(amap) & set(bmap))
    missing = {"a_only": len(set(amap) - set(bmap)), "b_only": len(set(bmap) - set(amap))}
    both_correct = a_only = b_only = both_wrong = 0
    sub = collections.defaultdict(lambda: {"n": 0, "a_correct": 0, "b_correct": 0, "b_minus_a": 0, "a_only_correct": 0, "b_only_correct": 0})
    examples_gain = []
    examples_loss = []
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
    n = len(keys)
    a_acc = 100.0 * (both_correct + a_only) / n if n else None
    b_acc = 100.0 * (both_correct + b_only) / n if n else None
    sub_rows = []
    for name, sr in sub.items():
        nsub = sr["n"]
        sub_rows.append({
            "subtask": name,
            "n": nsub,
            "a_acc": 100.0 * sr["a_correct"] / nsub if nsub else None,
            "b_acc": 100.0 * sr["b_correct"] / nsub if nsub else None,
            "delta_acc_b_minus_a": 100.0 * sr["b_minus_a"] / nsub if nsub else None,
            "net_item_delta_b_minus_a": sr["b_minus_a"],
            "a_only_correct": sr["a_only_correct"],
            "b_only_correct": sr["b_only_correct"],
        })
    sub_rows.sort(key=lambda r: (abs(float(r["delta_acc_b_minus_a"])), r["n"]), reverse=True)
    return {
        "n_common": n,
        "missing": missing,
        "a_accuracy_micro": a_acc,
        "b_accuracy_micro": b_acc,
        "delta_accuracy_micro_b_minus_a": (None if a_acc is None or b_acc is None else b_acc - a_acc),
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "net_item_delta_b_minus_a": b_only - a_only,
        "subtasks_by_abs_delta": sub_rows[:40],
        "example_gains_b_over_a": examples_gain,
        "example_losses_b_vs_a": examples_loss,
    }


def superglue_primary_summary(items_by_key: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    vals = []
    out = {}
    for task in SUPERGLUE_TASKS:
        rows = items_by_key[f"SuperGLUE/{task}"]
        labels = [int(r["target"]) for r in rows]
        preds = [int(r["pred"]) for r in rows]
        acc = 100.0 * sum(int(y == p) for y, p in zip(labels, preds)) / len(labels)
        metric = SUPERGLUE_PRIMARY[task]
        primary = binary_f1(labels, preds) if metric == "f1" else acc
        out[task] = {"primary_metric": metric, "primary_score": primary, "accuracy": acc, "n": len(rows)}
        vals.append(primary)
    return {"tasks": out, "primary_mean": sum(vals) / len(vals)}


def scores_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tasks = payload.get("tasks") or {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(col) or {}
        if isinstance(rec, dict):
            out[col] = rec.get("score")
    vals = []
    for col in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(col) or {}
        if isinstance(rec, dict) and rec.get("score") is not None:
            vals.append(float(rec["score"]))
            out[col] = rec.get("score")
    if len(vals) == 2:
        out["GlobalPIQA"] = sum(vals) / 2.0
    rec = tasks.get("Reading") or {}
    if isinstance(rec, dict):
        sc = rec.get("scores") or {}
        if sc.get("Reading") is not None:
            out["Reading"] = sc.get("Reading")
            out["Reading_eye"] = sc.get("Reading_eye")
            out["Reading_self_paced"] = sc.get("Reading_self_paced")
    rec = tasks.get("SuperGLUE") or {}
    if isinstance(rec, dict) and rec.get("superglue_mean") is not None:
        out["SuperGLUE"] = rec.get("superglue_mean")
    rec = tasks.get("AoA") or {}
    if isinstance(rec, dict):
        if rec.get("aoa_leaderboard_score") is not None:
            out["AoA"] = rec.get("aoa_leaderboard_score")
        elif rec.get("aoa_for_provisional_overall") is not None:
            out["AoA"] = rec.get("aoa_for_provisional_overall")
    keys7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    if all(out.get(k) is not None for k in keys7):
        out["cheap7_or_zero_readout_mean"] = sum(float(out[k]) for k in keys7) / len(keys7)
    keys9 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
    if all(out.get(k) is not None for k in keys9):
        out["provisional_overall"] = sum(float(out[k]) for k in keys9) / len(keys9)
    return out


def write_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    a = result["a_label"]
    b = result["b_label"]
    lines = []
    lines.append(f"# research item-transition comparison: {b} vs {a}\n")
    lines.append("## Score summary\n")
    lines.append("| column | A | B | B-A |\n|---|---:|---:|---:|")
    keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "SuperGLUE", "AoA", "cheap7_or_zero_readout_mean", "provisional_overall"]
    for k in keys:
        av = result["a_scores"].get(k)
        bv = result["b_scores"].get(k)
        if av is None and bv is None:
            continue
        delta = "" if av is None or bv is None else f"{float(bv)-float(av):+.4f}"
        lines.append(f"| {k} | {'' if av is None else f'{float(av):.4f}'} | {'' if bv is None else f'{float(bv):.4f}'} | {delta} |")
    lines.append("\n## Prediction-level transitions\n")
    lines.append("| column | n | A acc | B acc | B-A acc | B-only | A-only | net items |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for col, rec in result["column_comparisons"].items():
        lines.append(f"| {col} | {rec['n_common']} | {rec['a_accuracy_micro']:.4f} | {rec['b_accuracy_micro']:.4f} | {rec['delta_accuracy_micro_b_minus_a']:+.4f} | {rec['b_only_correct']} | {rec['a_only_correct']} | {rec['net_item_delta_b_minus_a']:+d} |")
    lines.append("\n## Largest subtask movements by absolute accuracy difference\n")
    for col, rec in result["column_comparisons"].items():
        lines.append(f"\n### {col}\n")
        lines.append("| subtask | n | A acc | B acc | B-A acc | net items |\n|---|---:|---:|---:|---:|---:|")
        for sr in rec["subtasks_by_abs_delta"][:15]:
            lines.append(f"| {sr['subtask']} | {sr['n']} | {sr['a_acc']:.3f} | {sr['b_acc']:.3f} | {sr['delta_acc_b_minus_a']:+.3f} | {sr['net_item_delta_b_minus_a']:+d} |")
    lines.append("\n## Interpretation note\n")
    lines.append(result.get("interpretation_note", ""))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a-label", required=True)
    ap.add_argument("--a-payload", nargs="+", required=True)
    ap.add_argument("--b-label", required=True)
    ap.add_argument("--b-payload", nargs="+", required=True)
    ap.add_argument("--out-root", default="experiments/archive/functional_learning/data/item_transitions")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--include-superglue", action="store_true")
    ap.add_argument("--max-examples", type=int, default=20)
    args = ap.parse_args()

    a_payload = load_payloads(args.a_payload)
    b_payload = load_payloads(args.b_payload)
    include_sg = bool(args.include_superglue)
    a_items = all_items(a_payload, include_sg)
    b_items = all_items(b_payload, include_sg)
    cols = list(a_items.keys())
    result = {
        "status": "ITEM_TRANSITION_COMPARE",
        "a_label": args.a_label,
        "b_label": args.b_label,
        "a_payloads": args.a_payload,
        "b_payloads": args.b_payload,
        "include_superglue": include_sg,
        "a_scores": scores_from_payload(a_payload),
        "b_scores": scores_from_payload(b_payload),
        "column_comparisons": {},
        "interpretation_note": "Positive B-A item counts show changed decisions on this evaluation coordinate, not by themselves a causal mechanism; compare with training provenance, scale-only controls, and full-score movement before attributing gains to new experience.",
    }
    for col in cols:
        result["column_comparisons"][col] = compare_column(a_items[col], b_items[col], args.max_examples)
    if include_sg:
        result["a_superglue_primary_from_predictions"] = superglue_primary_summary(a_items)
        result["b_superglue_primary_from_predictions"] = superglue_primary_summary(b_items)
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / f"{args.tag}_item_transition.json"
    out_md = out_root / f"{args.tag}_item_transition.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps({"status": result["status"], "tag": args.tag, "out_json": rel(out_json), "out_md": rel(out_md), "columns": len(result["column_comparisons"])}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
