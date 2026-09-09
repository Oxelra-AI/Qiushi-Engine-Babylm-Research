#!/usr/bin/env python3
"""research item/subtask reader for the HS/LS/HD/LD selected-eval factorial.

Given selected-evaluation per-target JSONs for all four arms, compute item-level
factorial movement:

  same_anchor_effect(item) = correct(HS) - correct(LS)
  deranged_effect(item)    = correct(HD) - correct(LD)
  interaction(item)        = same_anchor_effect - deranged_effect

Scores here are raw item correctness over columns where the research loader can
reconstruct gold answers.  They supplement, but do not replace, official score
contrasts from `factorial_selected_eval_and_interaction.py`.
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
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CHECKPOINTS_DEFAULT = ["chck_20M", "chck_40M", "chck_60M", "chck_80M", "chck_100M"]
COLUMNS_STABLE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GLOBAL_COLUMNS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
ALL_COLUMNS = COLUMNS_STABLE + GLOBAL_COLUMNS


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
BASE_READER = ROOT / "experiments/archive/frontier_consolidation/scripts/selected_prediction_movement_reader.py"
spec = importlib.util.spec_from_file_location("a02_step218_selected_reader", BASE_READER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import selected reader from {BASE_READER}")
reader = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reader
spec.loader.exec_module(reader)  # type: ignore[union-attr]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def per_target_path(eval_dir: Path, arm: str, ck: str) -> Path:
    return eval_dir / arm / ck / "eval" / "per_target" / f"factorial_{arm}_{ck}.json"


def load_arm_items(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rec = reader.load_eval_record(path)
    items, meta = reader.load_items_from_record(rec)
    return {it["item_id"]: it for it in items}, meta


def summarize_vals(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "mean_pp": None, "sum": 0}
    return {
        "n": len(vals),
        "mean_pp": 100.0 * sum(vals) / len(vals),
        "sum": int(sum(vals)) if all(float(v).is_integer() for v in vals) else sum(vals),
        "median": statistics.median(vals),
        "counts": dict(Counter(vals)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    same = [r["same_anchor_effect"] for r in rows]
    der = [r["deranged_effect"] for r in rows]
    inter = [r["interaction"] for r in rows]
    return {
        "n": len(rows),
        "same_anchor_effect": summarize_vals(same),
        "deranged_effect": summarize_vals(der),
        "interaction": summarize_vals(inter),
        "same_anchor_gain_count": sum(1 for r in rows if r["same_anchor_effect"] > 0),
        "same_anchor_loss_count": sum(1 for r in rows if r["same_anchor_effect"] < 0),
        "deranged_gain_count": sum(1 for r in rows if r["deranged_effect"] > 0),
        "deranged_loss_count": sum(1 for r in rows if r["deranged_effect"] < 0),
        "positive_interaction_count": sum(1 for r in rows if r["interaction"] > 0),
        "negative_interaction_count": sum(1 for r in rows if r["interaction"] < 0),
    }


def compute_checkpoint(eval_dir: Path, ck: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    problems: list[str] = []
    paths = {arm: per_target_path(eval_dir, arm, ck) for arm in ["hs", "ls", "hd", "ld"]}
    for arm, p in paths.items():
        if not p.exists():
            problems.append(f"{ck}/{arm}: missing {p}")
    if problems:
        return {"ck": ck, "missing": True, "paths": {k: str(v) for k, v in paths.items()}, "problems": problems}, [], problems
    arm_items: dict[str, dict[str, dict[str, Any]]] = {}
    metas: dict[str, Any] = {}
    for arm, p in paths.items():
        arm_items[arm], metas[arm] = load_arm_items(p)
    common = set.intersection(*(set(arm_items[a]) for a in ["hs", "ls", "hd", "ld"]))
    all_union = set.union(*(set(arm_items[a]) for a in ["hs", "ls", "hd", "ld"]))
    missing_counts = {arm: len(all_union - set(arm_items[arm])) for arm in ["hs", "ls", "hd", "ld"]}
    rows: list[dict[str, Any]] = []
    for item_id in sorted(common):
        hsi, lsi, hdi, ldi = (arm_items[a][item_id] for a in ["hs", "ls", "hd", "ld"])
        hs = 1 if hsi["correct"] else 0
        ls = 1 if lsi["correct"] else 0
        hd = 1 if hdi["correct"] else 0
        ld = 1 if ldi["correct"] else 0
        same = hs - ls
        der = hd - ld
        rows.append({
            "ck": ck,
            "item_id": item_id,
            "column": hsi["column"],
            "subtask": hsi["subtask"],
            "index": hsi["index"],
            "hs_correct": hs,
            "ls_correct": ls,
            "hd_correct": hd,
            "ld_correct": ld,
            "same_anchor_effect": same,
            "deranged_effect": der,
            "interaction": same - der,
            "hs_pred": hsi["pred"],
            "ls_pred": lsi["pred"],
            "hd_pred": hdi["pred"],
            "ld_pred": ldi["pred"],
            "target": hsi["target"],
            "meta_json": json.dumps(hsi.get("meta") or {}, ensure_ascii=False, sort_keys=True),
        })
    by_column = {}
    for col in ALL_COLUMNS:
        by_column[col] = summarize_rows([r for r in rows if r["column"] == col])
    gp = [r for r in rows if r["column"] in GLOBAL_COLUMNS]
    if gp:
        by_column["GlobalPIQA_item_pool"] = summarize_rows(gp)
    by_column["stable_five_item_pool"] = summarize_rows([r for r in rows if r["column"] in COLUMNS_STABLE])
    by_subtask = []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[(r["column"], r["subtask"])].append(r)
    for (col, sub), subset in buckets.items():
        s = summarize_rows(subset)
        s.update({"column": col, "subtask": sub})
        by_subtask.append(s)
    by_subtask.sort(key=lambda x: (x.get("interaction", {}).get("mean_pp") if x.get("interaction", {}).get("mean_pp") is not None else 0.0, x.get("interaction", {}).get("sum", 0), -x.get("n", 0)))
    payload = {
        "ck": ck,
        "paths": {k: str(v) for k, v in paths.items()},
        "n_common_items": len(common),
        "missing_counts_against_union": missing_counts,
        "by_column": by_column,
        "worst_interaction_subtasks": by_subtask[:25],
        "best_interaction_subtasks": list(reversed(by_subtask[-25:])),
        "warnings_by_arm": {arm: metas[arm].get("warnings", []) for arm in metas},
        "skipped_by_arm": {arm: metas[arm].get("skipped", {}) for arm in metas},
    }
    return payload, rows, problems


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ck", "item_id", "column", "subtask", "index", "hs_correct", "ls_correct", "hd_correct", "ld_correct", "same_anchor_effect", "deranged_effect", "interaction", "target", "hs_pred", "ls_pred", "hd_pred", "ld_pred", "meta_json"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path("experiments/archive/representation_and_objectives/data/factorial_selected_eval"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/archive/representation_and_objectives/data/factorial_item_interaction"))
    ap.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS_DEFAULT)
    args = ap.parse_args()

    eval_dir = args.eval_dir if args.eval_dir.is_absolute() else ROOT / args.eval_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_payloads = []
    all_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for ck in args.checkpoints:
        payload, rows, probs = compute_checkpoint(eval_dir, ck)
        ck_payloads.append(payload)
        all_rows.extend(rows)
        problems.extend(probs)
    overall = summarize_rows(all_rows)
    by_ck = {p["ck"]: p for p in ck_payloads}
    by_col_all = {}
    for col in ALL_COLUMNS + ["GlobalPIQA_item_pool", "stable_five_item_pool"]:
        if col == "GlobalPIQA_item_pool":
            subset = [r for r in all_rows if r["column"] in GLOBAL_COLUMNS]
        elif col == "stable_five_item_pool":
            subset = [r for r in all_rows if r["column"] in COLUMNS_STABLE]
        else:
            subset = [r for r in all_rows if r["column"] == col]
        by_col_all[col] = summarize_rows(subset)
    out_json = out_dir / "factorial_item_interaction_summary.json"
    out_csv = out_dir / "factorial_item_interaction_rows.csv"
    payload = {
        "status": "FACTORIAL_ITEM_INTERACTION",
        "created_utc": now(),
        "meaning": "Item/subtask correctness interaction for existing selected-evaluation outputs. Positive interaction means compact-over-repeat is larger when the second view shares the source anchor than when it is deranged.",
        "eval_dir": str(eval_dir),
        "checkpoints": args.checkpoints,
        "per_checkpoint": ck_payloads,
        "overall_item_summary": overall,
        "overall_by_column": by_col_all,
        "problems": problems,
        "outputs": {"json": str(out_json), "csv": str(out_csv)},
        "boundary": "This reader performs no training/evaluation/upload; it reads existing per-target prediction files.",
    }
    write_rows_csv(out_csv, all_rows)
    write_json(out_json, payload)
    md = [
        "# research factorial item interaction",
        "",
        f"Created UTC: `{payload['created_utc']}`",
        "",
        f"Checkpoints: `{args.checkpoints}`",
        f"Rows: `{len(all_rows)}`; problems: `{len(problems)}`",
        f"Overall item interaction: `{json.dumps(overall.get('interaction'), sort_keys=True)}`",
        f"Stable-five item interaction: `{json.dumps(by_col_all['stable_five_item_pool'].get('interaction'), sort_keys=True)}`",
        "",
        "This item-level readout supplements official score interactions; it should not override official selected aggregate movement.",
    ]
    (out_dir / "factorial_item_interaction_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(all_rows), "problems": len(problems), "out_json": str(out_json), "out_csv": str(out_csv)}, indent=2), flush=True)
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
