#!/usr/bin/env python3
"""research: item/subtask movement reader for BabyLM selected zero-shot predictions.

Reads two existing per-target evaluation JSON files and their saved predictions; does
not run any model/evaluation/training.  The output is meant to sharpen scientific
interpretation of compact-vs-control experiments by exposing which official columns
and subtasks gain/loss items instead of relying only on aggregate score deltas.

Convention: the comparison is RIGHT minus LEFT.  For compact-order, use
LEFT=scrambled and RIGHT=ordered; for RoBERTa transfer, use LEFT=repeat and
RIGHT=compact.
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
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


COLUMNS_STABLE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GLOBAL_COLUMNS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
ALL_ITEM_COLUMNS = COLUMNS_STABLE + GLOBAL_COLUMNS


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_path(x: str | Path | None) -> Path | None:
    if x is None:
        return None
    p = Path(str(x))
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return ROOT / p


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def short(x: Any, n: int = 220) -> str:
    s = norm(x)
    return s if len(s) <= n else s[: n - 1] + "…"


def pct(num: float, den: float) -> float | None:
    return 100.0 * num / den if den else None


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        v = float(x)
        if not math.isfinite(v):
            return "NA"
        return f"{v:.{nd}f}"
    except Exception:
        return str(x)


def load_eval_record(per_target_path: Path) -> dict[str, Any]:
    obj = read_json(per_target_path)
    if isinstance(obj, dict) and "record" in obj and isinstance(obj["record"], dict):
        return obj["record"]
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict per-target JSON at {per_target_path}")
    return obj


def prediction_list(block: Any) -> list[dict[str, Any]]:
    if isinstance(block, dict) and isinstance(block.get("predictions"), list):
        return block["predictions"]
    if isinstance(block, list):
        return block
    return []


def pred_text(rec: Any) -> str:
    if isinstance(rec, dict):
        return norm(rec.get("pred", ""))
    return norm(rec)


def mk_item(
    column: str,
    subtask: str,
    idx: int,
    pred: Any,
    target: Any,
    meta: dict[str, Any] | None = None,
    item_id: str | None = None,
) -> dict[str, Any]:
    p = pred_text(pred)
    t = norm(target)
    return {
        "item_id": item_id or f"{column}:{subtask}:{idx}",
        "column": column,
        "subtask": subtask,
        "index": idx,
        "pred": p,
        "target": t,
        "correct": p == t,
        "meta": meta or {},
    }


def load_blimp_like(column: str, task: dict[str, Any], ewok: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred_path = resolve_path(task.get("predictions"))
    data_dir = resolve_path(task.get("data_path"))
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if pred_path is None or data_dir is None or not pred_path.exists() or not data_dir.exists():
        warnings.append({"column": column, "problem": "missing_prediction_or_data", "predictions": str(pred_path), "data_path": str(data_dir)})
        return rows, warnings
    preds = read_json(pred_path)
    for subtask, block in preds.items():
        gold_file = data_dir / f"{subtask}.jsonl"
        if not gold_file.exists():
            warnings.append({"column": column, "subtask": subtask, "problem": "missing_gold_file", "gold_file": str(gold_file)})
            continue
        plist = prediction_list(block)
        gold_rows = list(iter_jsonl(gold_file))
        if len(plist) != len(gold_rows):
            warnings.append({"column": column, "subtask": subtask, "problem": "length_mismatch", "predictions": len(plist), "gold": len(gold_rows)})
        for i, (pr, gold) in enumerate(zip(plist, gold_rows)):
            if ewok:
                target = f"{gold.get('Context1', '')} {gold.get('Target1', '')}".strip()
                meta = {
                    "Domain": gold.get("Domain", subtask),
                    "ConceptA": gold.get("ConceptA"),
                    "ConceptB": gold.get("ConceptB"),
                    "ContextType": gold.get("ContextType"),
                    "ContextDiff": gold.get("ContextDiff"),
                    "TargetDiff": gold.get("TargetDiff"),
                    "target_snippet": short(target, 180),
                }
            else:
                target = gold.get("sentence_good", "")
                meta = {
                    "field": gold.get("field"),
                    "linguistics_term": gold.get("linguistics_term"),
                    "UID": gold.get("UID", subtask),
                    "contrast": gold.get("contrast"),
                    "template": gold.get("template"),
                    "target_snippet": short(target, 180),
                }
            rows.append(mk_item(column, subtask, i, pr, target, meta=meta))
    return rows, warnings


def load_entity(column: str, task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred_path = resolve_path(task.get("predictions"))
    data_dir = resolve_path(task.get("data_path"))
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if pred_path is None or data_dir is None or not pred_path.exists() or not data_dir.exists():
        warnings.append({"column": column, "problem": "missing_prediction_or_data", "predictions": str(pred_path), "data_path": str(data_dir)})
        return rows, warnings
    preds = read_json(pred_path)
    gold_by_sub: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stem in ["ambiref", "regular", "move_contents"]:
        gold_file = data_dir / f"{stem}.jsonl"
        if not gold_file.exists():
            continue
        for gold in iter_jsonl(gold_file):
            # Matches earlier official-compatible local readers: entity evaluation excludes rows
            # whose answer options contain "nothing" from the filtered prediction set.
            if any("nothing" in str(opt).lower() for opt in gold.get("options", [])):
                continue
            subtask = f"{stem}_{int(gold.get('numops', 0))}_ops"
            gold_by_sub[subtask].append(gold)
    for subtask, block in preds.items():
        plist = prediction_list(block)
        gold_list = gold_by_sub.get(subtask, [])
        if len(plist) != len(gold_list):
            warnings.append({"column": column, "subtask": subtask, "problem": "length_mismatch", "predictions": len(plist), "gold": len(gold_list)})
        for i, (pr, gold) in enumerate(zip(plist, gold_list)):
            options = [norm(o) for o in gold.get("options", [])]
            target = options[0] if options else ""
            meta = {
                "sample_id": gold.get("sample_id"),
                "example_id": gold.get("example_id"),
                "numops": gold.get("numops"),
                "input_prefix": short(gold.get("input_prefix", ""), 180),
                "target_snippet": short(target, 120),
                "n_options": len(options),
            }
            rows.append(mk_item(column, subtask, i, pr, target, meta=meta))
    return rows, warnings


def load_comps(column: str, task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred_path = resolve_path(task.get("predictions"))
    data_dir = resolve_path(task.get("data_path"))
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if pred_path is None or data_dir is None or not pred_path.exists() or not data_dir.exists():
        warnings.append({"column": column, "problem": "missing_prediction_or_data", "predictions": str(pred_path), "data_path": str(data_dir)})
        return rows, warnings
    preds = read_json(pred_path)
    subtask_to_file = {
        "base": "comps_base",
        "wugs": "comps_wugs",
        "wugs_dist_before": "comps_wugs_dist-before",
        "wugs_dist_in_between": "comps_wugs_dist-in-between",
    }
    for subtask, block in preds.items():
        stem = subtask_to_file.get(subtask, subtask)
        gold_file = data_dir / f"{stem}.jsonl"
        if not gold_file.exists():
            warnings.append({"column": column, "subtask": subtask, "problem": "missing_gold_file", "gold_file": str(gold_file)})
            continue
        plist = prediction_list(block)
        gold_rows = list(iter_jsonl(gold_file))
        if len(plist) != len(gold_rows):
            warnings.append({"column": column, "subtask": subtask, "problem": "length_mismatch", "predictions": len(plist), "gold": len(gold_rows)})
        for i, (pr, gold) in enumerate(zip(plist, gold_rows)):
            target = f"{gold.get('prefix_acceptable', '')} {gold.get('property_phrase', '')}".strip()
            meta = {
                "property": gold.get("property"),
                "negative_sample_type": gold.get("negative_sample_type"),
                "similarity": gold.get("similarity"),
                "target_snippet": short(target, 180),
            }
            rows.append(mk_item(column, subtask, i, pr, target, meta=meta))
    return rows, warnings


def load_globalpiqa(column: str, task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred_path = resolve_path(task.get("predictions"))
    data_dir = resolve_path(task.get("data_path"))
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if pred_path is None or data_dir is None or not pred_path.exists() or not data_dir.exists():
        warnings.append({"column": column, "problem": "missing_prediction_or_data", "predictions": str(pred_path), "data_path": str(data_dir)})
        return rows, warnings
    preds = read_json(pred_path)
    gold_file = data_dir / "eng_latn.jsonl"
    if not gold_file.exists():
        warnings.append({"column": column, "problem": "missing_gold_file", "gold_file": str(gold_file)})
        return rows, warnings
    for i, gold in enumerate(iter_jsonl(gold_file)):
        exid = str(gold.get("example_id"))
        block = preds.get(exid)
        if block is None:
            warnings.append({"column": column, "problem": "missing_prediction_item", "example_id": exid})
            continue
        plist = prediction_list(block)
        if not plist:
            warnings.append({"column": column, "problem": "empty_prediction_item", "example_id": exid})
            continue
        label = gold.get("label")
        target = gold.get(f"solution{label}", "")
        meta = {
            "example_id": exid,
            "categories": gold.get("categories"),
            "language": gold.get("language"),
            "prompt": short(gold.get("prompt", ""), 180),
            "target_snippet": short(target, 120),
        }
        rows.append(mk_item(column, data_dir.name, i, plist[0], target, meta=meta, item_id=f"{column}:{exid}"))
    return rows, warnings


def load_items_from_record(record: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = record.get("tasks", {})
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    task_scores: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for key, task in tasks.items():
        column = task.get("column", key)
        if column == "GlobalPIQA_parallel" or key == "GlobalPIQA_parallel":
            got, warn = load_globalpiqa("GlobalPIQA_parallel", task)
        elif column == "GlobalPIQA_nonparallel" or key == "GlobalPIQA_nonparallel":
            got, warn = load_globalpiqa("GlobalPIQA_nonparallel", task)
        elif column == "BLiMP":
            got, warn = load_blimp_like("BLiMP", task, ewok=False)
        elif column == "Supplement":
            got, warn = load_blimp_like("Supplement", task, ewok=False)
        elif column == "EWoK":
            got, warn = load_blimp_like("EWoK", task, ewok=True)
        elif column == "Entity":
            got, warn = load_entity("Entity", task)
        elif column == "COMPS":
            got, warn = load_comps("COMPS", task)
        elif column == "Reading" or key == "Reading":
            got, warn = [], []
            skipped["Reading"] = "continuous reading correlations, not item correctness flips"
        else:
            got, warn = [], []
            skipped[column] = "unsupported_or_nonselected_column"
        rows.extend(got)
        warnings.extend(warn)
        if "score" in task:
            task_scores[column] = task.get("score")
        elif "scores" in task:
            task_scores[column] = task.get("scores")
    official = (record.get("official_overall") or {}).get("scores") or {}
    return rows, {
        "target": record.get("target"),
        "endpoint": record.get("endpoint"),
        "model_path": record.get("model_path"),
        "run_dir": record.get("run_dir"),
        "task_scores": task_scores,
        "official_scores": official,
        "run_summary": record.get("run_summary"),
        "warnings": warnings,
        "skipped": skipped,
    }


def summarize_items(rows: list[dict[str, Any]], left_label: str, right_label: str) -> dict[str, Any]:
    n = len(rows)
    lc = sum(1 for r in rows if r["left_correct"])
    rc = sum(1 for r in rows if r["right_correct"])
    both_c = sum(1 for r in rows if r["left_correct"] and r["right_correct"])
    both_w = sum(1 for r in rows if (not r["left_correct"]) and (not r["right_correct"]))
    left_only = sum(1 for r in rows if r["left_correct"] and not r["right_correct"])
    right_only = sum(1 for r in rows if (not r["left_correct"]) and r["right_correct"])
    pred_agree = sum(1 for r in rows if norm(r["left_pred"]) == norm(r["right_pred"]))
    union_c = lc + rc - both_c
    flip_count = left_only + right_only
    net = rc - lc
    return {
        "n_common": n,
        f"{left_label}_correct": lc,
        f"{right_label}_correct": rc,
        f"{left_label}_item_acc": pct(lc, n),
        f"{right_label}_item_acc": pct(rc, n),
        f"delta_item_acc_{right_label}_minus_{left_label}": pct(net, n),
        "net_correct_delta_count": net,
        f"{left_label}_only_correct": left_only,
        f"{right_label}_only_correct": right_only,
        "both_correct": both_c,
        "both_wrong": both_w,
        "flip_count": flip_count,
        "flip_fraction_pct": pct(flip_count, n),
        "prediction_agreement_pct": pct(pred_agree, n),
        "correct_set_jaccard": (both_c / union_c) if union_c else None,
        "churn_to_abs_net": (flip_count / abs(net)) if net else None,
    }


def summarize_subtasks(rows: list[dict[str, Any]], left_label: str, right_label: str) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["column"], r["subtask"])].append(r)
    out: list[dict[str, Any]] = []
    delta_key = f"delta_item_acc_{right_label}_minus_{left_label}"
    for (column, subtask), subset in sorted(by.items()):
        s = summarize_items(subset, left_label, right_label)
        s.update({"column": column, "subtask": subtask})
        out.append(s)
    out.sort(key=lambda r: (r.get(delta_key) if r.get(delta_key) is not None else 0.0, r["net_correct_delta_count"], -r["n_common"]))
    return out


def official_delta(left_meta: dict[str, Any], right_meta: dict[str, Any], column: str) -> dict[str, Any]:
    loff = left_meta.get("official_scores") or {}
    roff = right_meta.get("official_scores") or {}
    # Per-task score is useful for GlobalPIQA split rows; official score has aggregated GlobalPIQA.
    ltask = left_meta.get("task_scores") or {}
    rtask = right_meta.get("task_scores") or {}
    if column in loff or column in roff:
        l = loff.get(column)
        r = roff.get(column)
    elif column in ltask or column in rtask:
        l = ltask.get(column)
        r = rtask.get(column)
    elif column == "GlobalPIQA":
        l = loff.get("GlobalPIQA")
        r = roff.get("GlobalPIQA")
    else:
        l = None; r = None
    try:
        delta = float(r) - float(l) if l is not None and r is not None else None
    except Exception:
        delta = None
    return {"left_official_score": l, "right_official_score": r, "official_delta_right_minus_left": delta}


def compare(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], left_label: str, right_label: str, left_meta: dict[str, Any], right_meta: dict[str, Any], max_examples: int) -> dict[str, Any]:
    lmap = {r["item_id"]: r for r in left_rows}
    rmap = {r["item_id"]: r for r in right_rows}
    ids = sorted(set(lmap) & set(rmap))
    item_rows: list[dict[str, Any]] = []
    for item_id in ids:
        l = lmap[item_id]; r = rmap[item_id]
        movement = "same_correct" if l["correct"] and r["correct"] else "same_wrong" if (not l["correct"] and not r["correct"]) else f"{right_label}_gain" if r["correct"] else f"{right_label}_loss"
        item_rows.append({
            "item_id": item_id,
            "column": l["column"],
            "subtask": l["subtask"],
            "index": l["index"],
            "movement": movement,
            "left_correct": bool(l["correct"]),
            "right_correct": bool(r["correct"]),
            "left_pred": l["pred"],
            "right_pred": r["pred"],
            "target": l["target"],
            "prediction_agree": norm(l["pred"]) == norm(r["pred"]),
            "meta": l.get("meta") or {},
        })
    missing = {
        "left_only_item_ids": len(set(lmap) - set(rmap)),
        "right_only_item_ids": len(set(rmap) - set(lmap)),
        "n_left_items": len(left_rows),
        "n_right_items": len(right_rows),
        "n_common_items": len(ids),
    }
    by_column: dict[str, Any] = {}
    for col in ALL_ITEM_COLUMNS:
        subset = [r for r in item_rows if r["column"] == col]
        if not subset:
            continue
        s = summarize_items(subset, left_label, right_label)
        s.update(official_delta(left_meta, right_meta, col))
        by_column[col] = s
    # Aggregated GlobalPIQA item and official view.
    gp_subset = [r for r in item_rows if r["column"] in GLOBAL_COLUMNS]
    if gp_subset:
        s = summarize_items(gp_subset, left_label, right_label)
        s.update(official_delta(left_meta, right_meta, "GlobalPIQA"))
        # Equal-split score delta mirrors challenge column formation when both splits exist.
        split_deltas = []
        for c in GLOBAL_COLUMNS:
            d = by_column.get(c, {}).get("official_delta_right_minus_left")
            if d is not None:
                split_deltas.append(float(d))
        s["mean_split_official_delta"] = sum(split_deltas) / len(split_deltas) if split_deltas else None
        by_column["GlobalPIQA"] = s
    # Stable composite item-level view.
    stable_subset = [r for r in item_rows if r["column"] in COLUMNS_STABLE]
    if stable_subset:
        by_column["stable_five_item_pool"] = summarize_items(stable_subset, left_label, right_label)
    subtasks = summarize_subtasks(item_rows, left_label, right_label)
    delta_key = f"delta_item_acc_{right_label}_minus_{left_label}"
    subtasks_min5 = [r for r in subtasks if r["n_common"] >= 5]
    worst = sorted(subtasks_min5, key=lambda r: (r.get(delta_key) if r.get(delta_key) is not None else 0.0, r["net_correct_delta_count"], -r["n_common"]))[:25]
    best = sorted(subtasks_min5, key=lambda r: (r.get(delta_key) if r.get(delta_key) is not None else 0.0, r["net_correct_delta_count"], r["n_common"]), reverse=True)[:25]
    examples = {
        "right_gains": [example_for_json(r) for r in item_rows if r["movement"] == f"{right_label}_gain"][:max_examples],
        "right_losses": [example_for_json(r) for r in item_rows if r["movement"] == f"{right_label}_loss"][:max_examples],
    }
    # Add high-value stable-family examples first, then fallback naturally sorted examples.
    examples["stable_right_gains"] = [example_for_json(r) for r in item_rows if r["movement"] == f"{right_label}_gain" and r["column"] in COLUMNS_STABLE][:max_examples]
    examples["stable_right_losses"] = [example_for_json(r) for r in item_rows if r["movement"] == f"{right_label}_loss" and r["column"] in COLUMNS_STABLE][:max_examples]
    return {
        "missing_item_diagnostics": missing,
        "by_column": by_column,
        "subtasks_sorted_by_delta": subtasks,
        "worst_subtasks_by_right_minus_left": worst,
        "best_subtasks_by_right_minus_left": best,
        "examples": examples,
        "per_item_rows": item_rows,
    }


def example_for_json(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": r["item_id"],
        "column": r["column"],
        "subtask": r["subtask"],
        "movement": r["movement"],
        "target": short(r["target"], 220),
        "left_pred": short(r["left_pred"], 180),
        "right_pred": short(r["right_pred"], 180),
        "meta": r.get("meta") or {},
    }


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "item_id", "column", "subtask", "index", "movement", "left_correct", "right_correct",
        "prediction_agree", "target", "left_pred", "right_pred", "meta_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "item_id": r["item_id"],
                "column": r["column"],
                "subtask": r["subtask"],
                "index": r["index"],
                "movement": r["movement"],
                "left_correct": int(r["left_correct"]),
                "right_correct": int(r["right_correct"]),
                "prediction_agree": int(r["prediction_agree"]),
                "target": r["target"],
                "left_pred": r["left_pred"],
                "right_pred": r["right_pred"],
                "meta_json": json.dumps(r.get("meta") or {}, ensure_ascii=False, sort_keys=True),
            })


def make_md(payload: dict[str, Any], left_label: str, right_label: str) -> str:
    delta_key = f"delta_item_acc_{right_label}_minus_{left_label}"
    lines: list[str] = []
    lines.append(f"# research selected-prediction movement: {right_label} minus {left_label}")
    lines.append("")
    lines.append(f"Created: `{payload['created_utc']}`")
    lines.append("")
    lines.append("This is a CPU/file-only reading of existing predictions; it does not train, evaluate, upload, or submit. Item accuracies are direct prediction-vs-gold correctness; official scores are quoted separately because several BabyLM columns average by subtask/UID rather than by raw item count.")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- LEFT `{left_label}`: `{payload['inputs']['left_per_target']}`")
    lines.append(f"- RIGHT `{right_label}`: `{payload['inputs']['right_per_target']}`")
    lines.append("")
    lines.append("## Column movement")
    lines.append(f"| column | n | official LEFT | official RIGHT | official Δ | item LEFT | item RIGHT | item Δ | net | flips | Jaccard | pred agree |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for col, s in payload["movement"]["by_column"].items():
        if not isinstance(s, dict) or "n_common" not in s:
            continue
        lines.append(
            f"| {col} | {s['n_common']} | {fmt(s.get('left_official_score'), 3)} | {fmt(s.get('right_official_score'), 3)} | {fmt(s.get('official_delta_right_minus_left'), 3)} | "
            f"{fmt(s.get(f'{left_label}_item_acc'), 3)} | {fmt(s.get(f'{right_label}_item_acc'), 3)} | {fmt(s.get(delta_key), 3)} | "
            f"{s.get('net_correct_delta_count')} | {s.get('flip_count')} | {fmt(s.get('correct_set_jaccard'), 4)} | {fmt(s.get('prediction_agreement_pct'), 2)} |"
        )
    lines.append("")
    lines.append("## Largest subtask movements (item accuracy, RIGHT minus LEFT)")
    lines.append("### Worst for RIGHT")
    for r in payload["movement"]["worst_subtasks_by_right_minus_left"][:12]:
        lines.append(f"- {r['column']} / {r['subtask']}: n={r['n_common']}, Δ={fmt(r.get(delta_key), 3)}, net={r['net_correct_delta_count']}, flips={r['flip_count']}, agree={fmt(r.get('prediction_agreement_pct'), 2)}%")
    lines.append("")
    lines.append("### Best for RIGHT")
    for r in payload["movement"]["best_subtasks_by_right_minus_left"][:12]:
        lines.append(f"- {r['column']} / {r['subtask']}: n={r['n_common']}, Δ={fmt(r.get(delta_key), 3)}, net={r['net_correct_delta_count']}, flips={r['flip_count']}, agree={fmt(r.get('prediction_agreement_pct'), 2)}%")
    lines.append("")
    diag = payload["movement"]["missing_item_diagnostics"]
    lines.append("## Coverage")
    lines.append(f"- common items `{diag['n_common_items']}`, left-only `{diag['left_only_item_ids']}`, right-only `{diag['right_only_item_ids']}`.")
    warn_n = len(payload["left_meta"].get("warnings", [])) + len(payload["right_meta"].get("warnings", []))
    lines.append(f"- loader warnings `{warn_n}`; skipped columns: left {payload['left_meta'].get('skipped')}, right {payload['right_meta'].get('skipped')}.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- JSON: `{payload['outputs']['json']}`")
    lines.append(f"- per-item CSV: `{payload['outputs']['per_item_csv']}`")
    return "\n".join(lines) + "\n"


def strip_payload_for_json(payload: dict[str, Any]) -> dict[str, Any]:
    # Keep full per-item rows in CSV; JSON gets summaries and examples only.
    out = dict(payload)
    out["movement"] = dict(payload["movement"])
    out["movement"].pop("per_item_rows", None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left-per-target", required=True, type=Path)
    ap.add_argument("--right-per-target", required=True, type=Path)
    ap.add_argument("--left-label", default="left")
    ap.add_argument("--right-label", default="right")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-examples", type=int, default=24)
    args = ap.parse_args()

    left_path = resolve_path(args.left_per_target)
    right_path = resolve_path(args.right_per_target)
    if left_path is None or right_path is None or not left_path.exists() or not right_path.exists():
        raise FileNotFoundError({"left": str(left_path), "right": str(right_path)})
    left_record = load_eval_record(left_path)
    right_record = load_eval_record(right_path)
    left_items, left_meta = load_items_from_record(left_record)
    right_items, right_meta = load_items_from_record(right_record)
    movement = compare(left_items, right_items, args.left_label, args.right_label, left_meta, right_meta, args.max_examples)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "selected_prediction_movement.json"
    md_path = args.out_dir / "selected_prediction_movement.md"
    csv_path = args.out_dir / "per_item_movement.csv"
    payload = {
        "status": "SELECTED_PREDICTION_MOVEMENT",
        "created_utc": now(),
        "meaning": "RIGHT-minus-LEFT item/subtask movement over existing BabyLM selected prediction files. This is interpretive file work, not model evaluation.",
        "inputs": {
            "left_per_target": str(left_path),
            "right_per_target": str(right_path),
            "left_label": args.left_label,
            "right_label": args.right_label,
        },
        "left_meta": left_meta,
        "right_meta": right_meta,
        "movement": movement,
        "outputs": {
            "json": str(json_path),
            "md": str(md_path),
            "per_item_csv": str(csv_path),
        },
        "boundary": "No training, selected evaluation, upload, SuperGLUE/AoA, or leaderboard submission is performed.",
    }
    write_csv_rows(csv_path, movement["per_item_rows"])
    json_path.write_text(json.dumps(strip_payload_for_json(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(make_md(strip_payload_for_json(payload), args.left_label, args.right_label), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(json_path),
        "out_md": str(md_path),
        "per_item_csv": str(csv_path),
        "n_common_items": movement["missing_item_diagnostics"]["n_common_items"],
        "by_column_keys": list(movement["by_column"].keys()),
        "warnings": len(left_meta.get("warnings", [])) + len(right_meta.get("warnings", [])),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
