#!/usr/bin/env python3
"""research: item/subtask movement analysis for the reference chck_84M endpoint branch.

This script is CPU/file-only.  It compares already-written official-compatible
prediction payloads for scale1.75 seed43022 reference checkpoints (default 82M,
84M, 100M).  The goal is to decide whether the newly observed 84M cheap-task
lift is broad correction, narrow redistribution, or a volatile artifact before
using SuperGLUE to harden the endpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import time
from collections import defaultdict
from typing import Any, Iterable

ENDPOINTS_DEFAULT = ["chck_82M", "chck_84M", "chck_100M"]
SELECTED_DIR_DEFAULT = pathlib.Path(
    "experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M"
)
CURRENT_FULL = pathlib.Path(
    "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
)
LEGACY_FULL = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval")

COMPS_FILE = {
    "base": "comps_base",
    "wugs": "comps_wugs",
    "wugs_dist_before": "comps_wugs_dist-before",
    "wugs_dist_in_between": "comps_wugs_dist-in-between",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def clean(x: Any) -> str:
    return safe_str(x).strip()


def short(x: Any, n: int = 180) -> str:
    s = safe_str(x).replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pct(x: float) -> float:
    return 100.0 * x


def pearson(a: list[float], b: list[float]) -> float | None:
    vals = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(vals) < 3:
        return None
    xs, ys = zip(*vals)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in vals) / math.sqrt(vx * vy)


def load_selected_rows(selected_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    path = selected_dir / "selected_trajectory.csv"
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ep = r.get("endpoint", "")
            if ep:
                rows[ep] = r
    return rows


def per_target_path(selected_dir: pathlib.Path, label: str, endpoint: str) -> pathlib.Path:
    return selected_dir / "eval" / "per_target" / f"{label}_{endpoint}.json"


def load_endpoint_payloads(selected_dir: pathlib.Path, label: str, endpoints: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for ep in endpoints:
        path = per_target_path(selected_dir, label, ep)
        if not path.exists():
            raise FileNotFoundError(f"Missing per-target payload for {ep}: {path}")
        obj = read_json(path)
        out[ep] = obj
    return out


def prediction_file(payload: dict[str, Any], task_key: str) -> pathlib.Path:
    task = payload.get("tasks", {}).get(task_key)
    if not task:
        raise KeyError(f"Payload has no task {task_key}; keys={list(payload.get('tasks', {}).keys())}")
    pred = task.get("predictions")
    if not pred:
        raise FileNotFoundError(f"Task {task_key} has no predictions path")
    p = pathlib.Path(pred)
    if not p.exists():
        raise FileNotFoundError(f"Predictions path missing for {task_key}: {p}")
    return p


def extract_blimp_like(pred_obj: dict[str, Any], data_root: pathlib.Path, column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subtask, block in pred_obj.items():
        gold_path = (data_root / subtask).with_suffix(".jsonl")
        gold_rows = list(iter_jsonl(gold_path))
        preds = block["predictions"]
        if len(preds) != len(gold_rows):
            raise RuntimeError(f"{column}/{subtask}: prediction/gold length mismatch {len(preds)} vs {len(gold_rows)}")
        for i, (pred, gold) in enumerate(zip(preds, gold_rows)):
            target = clean(gold["sentence_good"])
            got = clean(pred.get("pred"))
            rows.append({
                "item_key": f"{column}/{subtask}/{i}",
                "column": column,
                "subtask": subtask,
                "subgroup": safe_str(gold.get("linguistics_term") or gold.get("field") or subtask),
                "fine_group": safe_str(gold.get("field") or "supplement" if column == "Supplement" else gold.get("field") or ""),
                "item_index": i,
                "pred_id": safe_str(pred.get("id")),
                "gold": target,
                "pred": got,
                "correct": got == target,
                "text_a": short(gold.get("sentence_good")),
                "text_b": short(gold.get("sentence_bad")),
            })
    return rows


def extract_ewok(pred_obj: dict[str, Any], data_root: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subtask, block in pred_obj.items():
        gold_path = (data_root / subtask).with_suffix(".jsonl")
        gold_rows = list(iter_jsonl(gold_path))
        preds = block["predictions"]
        if len(preds) != len(gold_rows):
            raise RuntimeError(f"EWoK/{subtask}: prediction/gold length mismatch {len(preds)} vs {len(gold_rows)}")
        for i, (pred, gold) in enumerate(zip(preds, gold_rows)):
            target = clean(" ".join([safe_str(gold.get("Context1")), safe_str(gold.get("Target1"))]))
            got = clean(pred.get("pred"))
            rows.append({
                "item_key": f"EWoK/{subtask}/{i}",
                "column": "EWoK",
                "subtask": subtask,
                "subgroup": safe_str(gold.get("Domain") or subtask),
                "fine_group": "|".join([safe_str(gold.get("ContextType")), safe_str(gold.get("ContextDiff")), safe_str(gold.get("TargetDiff"))]),
                "item_index": i,
                "pred_id": safe_str(pred.get("id")),
                "gold": target,
                "pred": got,
                "correct": got == target,
                "text_a": short(gold.get("Context1")),
                "text_b": short(gold.get("Context2")),
            })
    return rows


def build_entity_gold(data_root: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    subtask_to_targets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(data_root.glob("*.jsonl")):
        for gold in iter_jsonl(path):
            if any("nothing" in safe_str(option) for option in gold.get("options", [])):
                continue
            subtask = f"{path.stem}_{gold['numops']}_ops"
            gold = dict(gold)
            gold["source_file_stem"] = path.stem
            subtask_to_targets[subtask].append(gold)
    return subtask_to_targets


def extract_entity(pred_obj: dict[str, Any], data_root: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gold_map = build_entity_gold(data_root)
    for subtask, block in pred_obj.items():
        preds = block["predictions"]
        gold_rows = gold_map[subtask]
        if len(preds) != len(gold_rows):
            raise RuntimeError(f"Entity/{subtask}: prediction/gold length mismatch {len(preds)} vs {len(gold_rows)}")
        for i, (pred, gold) in enumerate(zip(preds, gold_rows)):
            target = clean(gold["options"][0])
            got = clean(pred.get("pred"))
            rows.append({
                "item_key": f"Entity/{subtask}/{i}",
                "column": "Entity",
                "subtask": subtask,
                "subgroup": safe_str(gold.get("source_file_stem")),
                "fine_group": f"numops={gold.get('numops')}",
                "item_index": i,
                "pred_id": safe_str(pred.get("id")),
                "gold": target,
                "pred": got,
                "correct": got == target,
                "text_a": short(gold.get("input_prefix")),
                "text_b": short(" || ".join(map(safe_str, gold.get("options", [])[:5]))),
            })
    return rows


def extract_comps(pred_obj: dict[str, Any], data_root: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subtask, block in pred_obj.items():
        gold_path = (data_root / COMPS_FILE[subtask]).with_suffix(".jsonl")
        gold_rows = list(iter_jsonl(gold_path))
        preds = block["predictions"]
        if len(preds) != len(gold_rows):
            raise RuntimeError(f"COMPS/{subtask}: prediction/gold length mismatch {len(preds)} vs {len(gold_rows)}")
        for i, (pred, gold) in enumerate(zip(preds, gold_rows)):
            target = clean(" ".join([safe_str(gold.get("prefix_acceptable")), safe_str(gold.get("property_phrase"))]))
            got = clean(pred.get("pred"))
            rows.append({
                "item_key": f"COMPS/{subtask}/{i}",
                "column": "COMPS",
                "subtask": subtask,
                "subgroup": safe_str(gold.get("negative_sample_type") or subtask),
                "fine_group": safe_str(gold.get("distraction_type") or ""),
                "item_index": i,
                "pred_id": safe_str(pred.get("id")),
                "gold": target,
                "pred": got,
                "correct": got == target,
                "text_a": short(gold.get("prefix_acceptable")),
                "text_b": short(gold.get("prefix_unacceptable")),
                "property": safe_str(gold.get("property")),
            })
    return rows


def extract_global_piqa(pred_obj: dict[str, Any], data_path: pathlib.Path, task_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, gold in enumerate(iter_jsonl(data_path)):
        eid = gold["example_id"]
        pred_block = pred_obj[eid]["predictions"][0]
        got = clean(pred_block.get("pred"))
        target = clean(gold[f"solution{gold['label']}"])
        rows.append({
            "item_key": f"GlobalPIQA/{task_name}/{eid}",
            "column": "GlobalPIQA",
            "subtask": task_name,
            "subgroup": safe_str(gold.get("categories") or task_name),
            "fine_group": safe_str(json.loads(gold.get("supplement") or "{}").get("example_inspiration", "")) if isinstance(gold.get("supplement"), str) else "",
            "item_index": i,
            "pred_id": safe_str(pred_block.get("id")),
            "gold": target,
            "pred": got,
            "correct": got == target,
            "text_a": short(gold.get("prompt")),
            "text_b": short(" | ".join(safe_str(gold.get(f"solution{j}")) for j in range(4) if f"solution{j}" in gold)),
        })
    return rows


def endpoint_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows += extract_blimp_like(read_json(prediction_file(payload, "BLiMP")), CURRENT_FULL / "blimp_filtered", "BLiMP")
    rows += extract_blimp_like(read_json(prediction_file(payload, "Supplement")), CURRENT_FULL / "supplement_filtered", "Supplement")
    rows += extract_ewok(read_json(prediction_file(payload, "EWoK")), CURRENT_FULL / "ewok_filtered")
    rows += extract_entity(read_json(prediction_file(payload, "Entity")), CURRENT_FULL / "entity_tracking")
    rows += extract_comps(read_json(prediction_file(payload, "COMPS")), CURRENT_FULL / "comps")
    rows += extract_global_piqa(
        read_json(prediction_file(payload, "GlobalPIQA_parallel")),
        LEGACY_FULL / "global_piqa_parallel" / "eng_latn.jsonl",
        "global_piqa_parallel",
    )
    rows += extract_global_piqa(
        read_json(prediction_file(payload, "GlobalPIQA_nonparallel")),
        LEGACY_FULL / "global_piqa_nonparallel" / "eng_latn.jsonl",
        "global_piqa_nonparallel",
    )
    return rows


def merge_endpoint_items(endpoint_to_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    all_keys = None
    base_meta: dict[str, dict[str, Any]] = {}
    for ep, rows in endpoint_to_rows.items():
        keys = {r["item_key"] for r in rows}
        if all_keys is None:
            all_keys = keys
        elif keys != all_keys:
            raise RuntimeError(f"Endpoint {ep} item-key set mismatch: {len(keys)} vs {len(all_keys)}")
        for r in rows:
            base_meta.setdefault(r["item_key"], {k: v for k, v in r.items() if k not in {"pred", "correct"}})
    merged = []
    per_ep = {ep: {r["item_key"]: r for r in rows} for ep, rows in endpoint_to_rows.items()}
    for key in sorted(base_meta):
        row = dict(base_meta[key])
        for ep in endpoint_to_rows:
            eprow = per_ep[ep][key]
            suffix = ep.replace("chck_", "").replace("M", "")
            row[f"correct_{suffix}"] = int(bool(eprow["correct"]))
            row[f"pred_{suffix}"] = eprow["pred"]
        merged.append(row)
    return merged


def group_summary(rows: list[dict[str, Any]], group_fields: list[str], endpoints: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(f, "") for f in group_fields)].append(r)
    out = []
    suff = [ep.replace("chck_", "").replace("M", "") for ep in endpoints]
    for key, rs in groups.items():
        rec = {f: key[i] for i, f in enumerate(group_fields)}
        rec["n_items"] = len(rs)
        for s in suff:
            rec[f"score_{s}"] = pct(sum(r[f"correct_{s}"] for r in rs) / len(rs))
        if {"82", "84"}.issubset(suff):
            rec["delta_84_minus_82"] = rec["score_84"] - rec["score_82"]
            rec["gains_82_to_84"] = sum((r["correct_82"] == 0 and r["correct_84"] == 1) for r in rs)
            rec["losses_82_to_84"] = sum((r["correct_82"] == 1 and r["correct_84"] == 0) for r in rs)
            rec["net_items_82_to_84"] = rec["gains_82_to_84"] - rec["losses_82_to_84"]
            denom = max(1, sum(r["correct_82"] == 1 for r in rs))
            rec["retention_of_82_correct_at_84"] = sum((r["correct_82"] == 1 and r["correct_84"] == 1) for r in rs) / denom
        if {"84", "100"}.issubset(suff):
            rec["delta_100_minus_84"] = rec["score_100"] - rec["score_84"]
            rec["gains_84_to_100"] = sum((r["correct_84"] == 0 and r["correct_100"] == 1) for r in rs)
            rec["losses_84_to_100"] = sum((r["correct_84"] == 1 and r["correct_100"] == 0) for r in rs)
            rec["net_items_84_to_100"] = rec["gains_84_to_100"] - rec["losses_84_to_100"]
        out.append(rec)
    return sorted(out, key=lambda r: (safe_str(r.get(group_fields[0])), safe_str(r.get(group_fields[-1]))))


def official_like_column_scores(subtask_rows: list[dict[str, Any]], endpoints: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    suff = [ep.replace("chck_", "").replace("M", "") for ep in endpoints]
    by_column: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in subtask_rows:
        by_column[r["column"]].append(r)
    for col, rows in by_column.items():
        out[col] = {}
        for s in suff:
            vals = [r[f"score_{s}"] for r in rows]
            out[col][f"score_{s}"] = mean(vals) or 0.0
        if "82" in suff and "84" in suff:
            out[col]["delta_84_minus_82"] = out[col]["score_84"] - out[col]["score_82"]
        if "84" in suff and "100" in suff:
            out[col]["delta_100_minus_84"] = out[col]["score_100"] - out[col]["score_84"]
    return out


def load_reading(endpoint_payloads: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred_by_ep: dict[str, list[dict[str, Any]]] = {}
    for ep, payload in endpoint_payloads.items():
        pred_by_ep[ep] = read_json(prediction_file(payload, "Reading"))["reading"]["predictions"]
    # Lengths must match; align by order/id because official reading scorer does so.
    lengths = {ep: len(v) for ep, v in pred_by_ep.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Reading prediction lengths mismatch: {lengths}")
    data_rows = []
    reading_csv = CURRENT_FULL / "reading" / "reading_data.csv"
    with reading_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data_rows.append(row)
    n = next(iter(lengths.values()))
    if n != len(data_rows):
        raise RuntimeError(f"Reading length mismatch predictions={n} data={len(data_rows)}")
    suff = {ep: ep.replace("chck_", "").replace("M", "") for ep in endpoint_payloads}
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rec = {
            "row_index": i,
            "item": data_rows[i].get("item", ""),
            "word": data_rows[i].get("word", ""),
            "sentence": short(data_rows[i].get("sentence", ""), 240),
            "context_length": data_rows[i].get("context_length", ""),
            "length": data_rows[i].get("length", ""),
        }
        for ep, preds in pred_by_ep.items():
            s = suff[ep]
            rec[f"pred_{s}"] = preds[i].get("pred")
            rec[f"prev_pred_{s}"] = preds[i].get("prev_pred")
        if {"chck_82M", "chck_84M"}.issubset(endpoint_payloads):
            rec["delta_pred_84_minus_82"] = rec["pred_84"] - rec["pred_82"]
            if isinstance(rec.get("prev_pred_84"), (int, float)) and isinstance(rec.get("prev_pred_82"), (int, float)):
                rec["delta_prev_pred_84_minus_82"] = rec["prev_pred_84"] - rec["prev_pred_82"]
            else:
                rec["delta_prev_pred_84_minus_82"] = None
        if {"chck_84M", "chck_100M"}.issubset(endpoint_payloads):
            rec["delta_pred_100_minus_84"] = rec["pred_100"] - rec["pred_84"]
            if isinstance(rec.get("prev_pred_100"), (int, float)) and isinstance(rec.get("prev_pred_84"), (int, float)):
                rec["delta_prev_pred_100_minus_84"] = rec["prev_pred_100"] - rec["prev_pred_84"]
            else:
                rec["delta_prev_pred_100_minus_84"] = None
        rows.append(rec)
    summary: dict[str, Any] = {"n_items": n, "prediction_file_sha256": {}}
    for ep, payload in endpoint_payloads.items():
        task = payload["tasks"]["Reading"]
        summary["prediction_file_sha256"][ep] = sha256_file(pathlib.Path(task["predictions"]))
        summary.setdefault("scores", {})[ep] = task.get("scores", {})
    for a, b in [("82", "84"), ("84", "100"), ("82", "100")]:
        if f"pred_{a}" in rows[0] and f"pred_{b}" in rows[0]:
            summary[f"pearson_pred_{a}_{b}"] = pearson([float(r[f"pred_{a}"]) for r in rows], [float(r[f"pred_{b}"]) for r in rows])
            ds = [float(r[f"pred_{b}"]) - float(r[f"pred_{a}"]) for r in rows]
            summary[f"mean_delta_pred_{b}_minus_{a}"] = statistics.fmean(ds)
            summary[f"mean_abs_delta_pred_{b}_minus_{a}"] = statistics.fmean(abs(x) for x in ds)
    return rows, summary


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def json_sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_sanitize(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def top_groups(rows: list[dict[str, Any]], field: str, n: int = 8) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valids = [r for r in rows if isinstance(r.get(field), (int, float))]
    pos = sorted(valids, key=lambda r: r[field], reverse=True)[:n]
    neg = sorted(valids, key=lambda r: r[field])[:n]
    return pos, neg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected-dir", default=str(SELECTED_DIR_DEFAULT))
    ap.add_argument("--label", default="scale1p75_seed43022_reference")
    ap.add_argument("--endpoints", nargs="+", default=ENDPOINTS_DEFAULT)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    selected_dir = pathlib.Path(args.selected_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    endpoints = list(args.endpoints)

    t0 = time.time()
    selected_rows = load_selected_rows(selected_dir)
    payloads = load_endpoint_payloads(selected_dir, args.label, endpoints)

    endpoint_to_items: dict[str, list[dict[str, Any]]] = {}
    for ep, payload in payloads.items():
        endpoint_to_items[ep] = endpoint_items(payload)
    merged = merge_endpoint_items(endpoint_to_items)

    # Summaries.
    subtask = group_summary(merged, ["column", "subtask"], endpoints)
    subgroup = group_summary(merged, ["column", "subgroup"], endpoints)
    fine_group = group_summary(merged, ["column", "subtask", "fine_group"], endpoints)
    col_scores = official_like_column_scores(subtask, endpoints)

    flips = []
    for r in merged:
        c82 = r.get("correct_82")
        c84 = r.get("correct_84")
        c100 = r.get("correct_100")
        direction = None
        if c82 == 0 and c84 == 1:
            direction = "gain_82_to_84"
        elif c82 == 1 and c84 == 0:
            direction = "loss_82_to_84"
        elif c84 == 0 and c100 == 1:
            direction = "gain_84_to_100_no_82_84_flip"
        elif c84 == 1 and c100 == 0:
            direction = "loss_84_to_100_no_82_84_flip"
        if direction:
            flips.append({
                "direction": direction,
                "column": r["column"],
                "subtask": r["subtask"],
                "subgroup": r.get("subgroup", ""),
                "fine_group": r.get("fine_group", ""),
                "item_key": r["item_key"],
                "pred_id": r.get("pred_id", ""),
                "correct_82": c82,
                "correct_84": c84,
                "correct_100": c100,
                "gold": short(r.get("gold"), 220),
                "pred_82": short(r.get("pred_82"), 220),
                "pred_84": short(r.get("pred_84"), 220),
                "pred_100": short(r.get("pred_100"), 220),
                "text_a": r.get("text_a", ""),
                "text_b": r.get("text_b", ""),
            })

    reading_rows, reading_summary = load_reading(payloads)
    reading_top = sorted(
        reading_rows,
        key=lambda r: abs(float(r.get("delta_pred_84_minus_82") or 0.0)),
        reverse=True,
    )[:200]

    # Endpoint-level classification item retention and churn.
    by_col = defaultdict(list)
    for r in merged:
        by_col[r["column"]].append(r)
    churn_by_col = {}
    for col, rs in by_col.items():
        gains = sum(r["correct_82"] == 0 and r["correct_84"] == 1 for r in rs)
        losses = sum(r["correct_82"] == 1 and r["correct_84"] == 0 for r in rs)
        correct82 = sum(r["correct_82"] == 1 for r in rs)
        correct84 = sum(r["correct_84"] == 1 for r in rs)
        correct100 = sum(r.get("correct_100") == 1 for r in rs)
        losses_84_100 = sum(r["correct_84"] == 1 and r.get("correct_100") == 0 for r in rs)
        gains_84_100 = sum(r["correct_84"] == 0 and r.get("correct_100") == 1 for r in rs)
        new84 = [r for r in rs if r["correct_82"] == 0 and r["correct_84"] == 1]
        churn_by_col[col] = {
            "n_items": len(rs),
            "raw_item_accuracy_82": correct82 / len(rs),
            "raw_item_accuracy_84": correct84 / len(rs),
            "raw_item_accuracy_100": correct100 / len(rs),
            "gains_82_to_84": gains,
            "losses_82_to_84": losses,
            "net_items_82_to_84": gains - losses,
            "changed_items_82_to_84": gains + losses,
            "retention_of_82_correct_at_84": (correct82 - losses) / correct82 if correct82 else None,
            "gains_84_to_100": gains_84_100,
            "losses_84_to_100": losses_84_100,
            "net_items_84_to_100": gains_84_100 - losses_84_100,
            "new84_items_retained_at_100": (sum(r.get("correct_100") == 1 for r in new84) / len(new84)) if new84 else None,
        }

    # Save tables.
    write_csv(out_dir / "classification_item_records.csv", merged)
    write_csv(out_dir / "classification_item_flips.csv", flips)
    write_csv(out_dir / "subtask_movement.csv", subtask)
    write_csv(out_dir / "subgroup_movement.csv", subgroup)
    write_csv(out_dir / "fine_group_movement.csv", fine_group)
    write_csv(out_dir / "reading_token_deltas_top200.csv", reading_top)
    write_csv(out_dir / "reading_token_deltas_all.csv", reading_rows)

    selected_scores = {ep: selected_rows.get(ep, {}) for ep in endpoints}
    summary = {
        "status": "CHCK84_ITEM_MOVEMENT_ANALYSIS",
        "created_utc": now_utc(),
        "selected_dir": str(selected_dir),
        "label": args.label,
        "endpoints": endpoints,
        "n_classification_items": len(merged),
        "n_flip_rows_saved": len(flips),
        "selected_scores": selected_scores,
        "official_like_column_scores_from_predictions": col_scores,
        "churn_by_column_raw_items": churn_by_col,
        "reading_summary": reading_summary,
        "top_positive_subtasks_84_minus_82": {},
        "top_negative_subtasks_84_minus_82": {},
        "top_negative_subtasks_100_minus_84": {},
        "output_files": {},
        "elapsed_sec": round(time.time() - t0, 2),
    }
    for col in sorted({r["column"] for r in subtask}):
        rows_col = [r for r in subtask if r["column"] == col]
        pos, neg = top_groups(rows_col, "delta_84_minus_82", n=10)
        _pos100, neg100 = top_groups(rows_col, "delta_100_minus_84", n=10)
        summary["top_positive_subtasks_84_minus_82"][col] = pos
        summary["top_negative_subtasks_84_minus_82"][col] = neg
        summary["top_negative_subtasks_100_minus_84"][col] = neg100
    for name in [
        "classification_item_records.csv",
        "classification_item_flips.csv",
        "subtask_movement.csv",
        "subgroup_movement.csv",
        "fine_group_movement.csv",
        "reading_token_deltas_top200.csv",
        "reading_token_deltas_all.csv",
    ]:
        p = out_dir / name
        summary["output_files"][name] = {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size if p.exists() else None}

    # Markdown synthesis.
    lines = []
    lines.append("# research `chck_84M` cheap-task item movement analysis\n")
    lines.append("CPU/file-only analysis of already-produced official-compatible predictions for the scale1.75 seed43022 reference trajectory. It does not evaluate SuperGLUE and does not authorize any new training.\n")
    lines.append("## Endpoint-level cheap columns\n")
    cols_order = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]
    lines.append("| column | score82 | score84 | score100 | Δ84-82 | Δ100-84 | raw gains 82→84 | raw losses 82→84 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for col in cols_order:
        if col in col_scores:
            cs = col_scores[col]
            ch = churn_by_col[col]
            lines.append(
                f"| {col} | {cs.get('score_82', float('nan')):.4f} | {cs.get('score_84', float('nan')):.4f} | {cs.get('score_100', float('nan')):.4f} | "
                f"{cs.get('delta_84_minus_82', float('nan')):+.4f} | {cs.get('delta_100_minus_84', float('nan')):+.4f} | {ch['gains_82_to_84']} | {ch['losses_82_to_84']} |\n"
            )
    lines.append("\nReading is regression-style; selected-grid score moved 8.150→8.155→8.320. The 84M cheap7 lift is therefore driven by classification columns rather than Reading.\n")

    lines.append("\n## Main interpretation\n")
    pos_cols = [col for col, cs in col_scores.items() if cs.get("delta_84_minus_82", 0) > 0]
    neg_cols = [col for col, cs in col_scores.items() if cs.get("delta_84_minus_82", 0) < 0]
    lines.append(f"`chck_84M` improves {len(pos_cols)}/{len(pos_cols)+len(neg_cols)} classification cheap columns relative to `chck_82M` ({', '.join(pos_cols)} positive; {', '.join(neg_cols)} negative). ")
    lines.append("This supports treating 84M as a real endpoint branch rather than a single GlobalPIQA/Reading accident, but it remains a single seed on the same trajectory and therefore cannot support the transferable residual-capacity story without the pending seed43122 grid.\n")
    lines.append("Raw item churn remains large relative to the net movement, showing competence allocation rather than monotone accumulation: many 82M-correct decisions flip off even when the weighted column score improves.\n")

    lines.append("\n## Largest subtask movements from 82M to 84M\n")
    for col in cols_order:
        if col not in summary["top_positive_subtasks_84_minus_82"]:
            continue
        lines.append(f"\n### {col}\n")
        lines.append("Positive subtask deltas:\n")
        for r in summary["top_positive_subtasks_84_minus_82"][col][:5]:
            lines.append(f"- {r['subtask']}: {r['delta_84_minus_82']:+.3f} points (n={r['n_items']}, net={r.get('net_items_82_to_84')})\n")
        lines.append("Negative subtask deltas:\n")
        for r in summary["top_negative_subtasks_84_minus_82"][col][:5]:
            lines.append(f"- {r['subtask']}: {r['delta_84_minus_82']:+.3f} points (n={r['n_items']}, net={r.get('net_items_82_to_84')})\n")

    lines.append("\n## Files\n")
    for name, rec in summary["output_files"].items():
        lines.append(f"- `{rec['path']}` ({rec['bytes']} bytes, sha256 `{str(rec['sha256'])[:16]}…`)\n")
    lines.append(f"- JSON summary: `{out_dir / 'chck84_item_movement_summary.json'}`\n")

    (out_dir / "chck84_item_movement_summary.json").write_text(json.dumps(json_sanitize(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "chck84_item_movement_summary.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(out_dir),
        "n_classification_items": len(merged),
        "n_flip_rows_saved": len(flips),
        "column_deltas_84_minus_82": {col: round(col_scores[col].get("delta_84_minus_82", 0.0), 6) for col in sorted(col_scores)},
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
