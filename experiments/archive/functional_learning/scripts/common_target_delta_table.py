#!/usr/bin/env python3
"""Compute per-task common-target margins and model-vs-parent deltas."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
from collections import defaultdict

ROOT = _public_path('.')
S = _public_path('experiments/archive/functional_learning')
INP = _public_path('experiments/archive/functional_learning/data/common_target_probe/scores_all_models.jsonl')
OUT = _public_path('experiments/archive/functional_learning/data/common_target_probe')


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def margin_rows(scores):
    cand = {}
    meta = {}
    for r in scores:
        key = (r["model"], r["task_id"], r["condition"], r["candidate_role"])
        cand[key] = r
        meta[(r["model"], r["task_id"])] = {k: r.get(k) for k in ["model", "task_id", "pair_id", "split", "axis", "frame"]}
    rows = []
    for (model, task), m in sorted(meta.items()):
        for cond in ["source_original", "source_altered", "no_source"]:
            o = cand.get((model, task, cond, "original_answer"))
            a = cand.get((model, task, cond, "altered_answer"))
            if not o or not a:
                continue
            if cond == "source_altered":
                expected_margin = o["nll"] - a["nll"]
            else:
                expected_margin = a["nll"] - o["nll"]
            rows.append({**m, "condition": cond, "original_nll": o["nll"], "altered_nll": a["nll"], "expected_margin": expected_margin, "expected_correct": expected_margin > 0})
    return rows


def main():
    scores = load_jsonl(INP)
    rows = margin_rows(scores)
    parent = {(r["task_id"], r["condition"]): r for r in rows if r["model"] == "parent"}
    out = []
    for r in rows:
        rr = dict(r)
        p = parent.get((r["task_id"], r["condition"]))
        rr["delta_vs_parent"] = None if p is None else r["expected_margin"] - p["expected_margin"]
        rr["parent_expected_correct"] = None if p is None else p["expected_correct"]
        rr["changed_correctness_vs_parent"] = None if p is None else (r["expected_correct"] != p["expected_correct"])
        out.append(rr)
    with (_public_path('experiments/archive/functional_learning/data/common_target_probe/common_target_margin_deltas.csv')).open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "task_id", "pair_id", "split", "axis", "condition", "expected_margin", "delta_vs_parent", "expected_correct", "parent_expected_correct", "changed_correctness_vs_parent", "original_nll", "altered_nll", "frame"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k) for k in fieldnames})
    with (_public_path('experiments/archive/functional_learning/data/common_target_probe/common_target_margin_deltas.jsonl')).open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Concise ranked summaries.
    models = sorted(set(r["model"] for r in out if r["model"] != "parent"))
    summaries = {}
    for model in models:
        rs = [r for r in out if r["model"] == model and r["delta_vs_parent"] is not None]
        rs_sorted = sorted(rs, key=lambda x: x["delta_vs_parent"], reverse=True)
        changed = [r for r in rs if r["changed_correctness_vs_parent"]]
        summaries[model] = {
            "top_positive_deltas": [{k: r[k] for k in ["task_id", "split", "axis", "condition", "expected_margin", "delta_vs_parent", "expected_correct", "parent_expected_correct"]} for r in rs_sorted[:10]],
            "top_negative_deltas": [{k: r[k] for k in ["task_id", "split", "axis", "condition", "expected_margin", "delta_vs_parent", "expected_correct", "parent_expected_correct"]} for r in rs_sorted[-10:]],
            "changed_correctness": [{k: r[k] for k in ["task_id", "split", "axis", "condition", "expected_margin", "delta_vs_parent", "expected_correct", "parent_expected_correct"]} for r in changed],
        }
    (_public_path('experiments/archive/functional_learning/data/common_target_probe/common_target_delta_inspection.json')).write_text(json.dumps({"status": "COMMON_TARGET_DELTA_TABLE", "rows": len(out), "summaries": summaries}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMMON_TARGET_DELTA_TABLE", "rows": len(out), "csv": str((_public_path('experiments/archive/functional_learning/data/common_target_probe/common_target_margin_deltas.csv')).relative_to(ROOT)), "inspection": str((_public_path('experiments/archive/functional_learning/data/common_target_probe/common_target_delta_inspection.json')).relative_to(ROOT)), "models": models}, indent=2), flush=True)


if __name__ == "__main__":
    main()
