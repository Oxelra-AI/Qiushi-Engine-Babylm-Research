#!/usr/bin/env python3
"""research: re-summarize research common-target evidence by contextual direction.

The research summary averaged source-original, source-altered, and no-source margins in
some fields.  For the route decision, separate: (i) original-source direction,
(ii) altered-source direction, and (iii) no-source prior shift.  This prevents a
small held-source average from being mistaken for symmetric evidence use.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
from collections import defaultdict
from statistics import mean

ROOT = _public_path('.')
S = _public_path('experiments/archive/functional_learning')
INPUTS = {
    "models": _public_path('experiments/archive/functional_learning/data/common_target_probe/scores_all_models.jsonl'),
    "exact_continuation": _public_path('experiments/archive/functional_learning/data/common_target_probe_continuation/scores_all_models.jsonl'),
}
OUT = _public_path('experiments/archive/functional_learning/data/route_synthesis')


def load_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def margin_rows(scores):
    cand = {}
    meta = {}
    for r in scores:
        key = (r["model"], r["task_id"], r["condition"], r["candidate_role"])
        cand[key] = r
        meta[(r["model"], r["task_id"])] = {k: r.get(k) for k in ["model", "task_id", "pair_id", "split", "axis", "semantic_label", "frame"]}
    rows = []
    for (model, task_id), m in sorted(meta.items()):
        for cond in ["source_original", "source_altered", "no_source"]:
            orig = cand.get((model, task_id, cond, "original_answer"))
            alt = cand.get((model, task_id, cond, "altered_answer"))
            if not orig or not alt:
                continue
            # Positive means the condition-supported answer beats the other answer.
            # For no-source, keep original-vs-altered as a prior diagnostic.
            if cond == "source_altered":
                expected_margin = orig["nll"] - alt["nll"]
            else:
                expected_margin = alt["nll"] - orig["nll"]
            rows.append({**m, "condition": cond, "expected_margin": float(expected_margin), "expected_correct": expected_margin > 0})
    return rows


def summarize_group(rows):
    if not rows:
        return {"n": 0, "mean_delta": None, "mean_margin": None, "success": 0, "changed_correctness": 0}
    return {
        "n": len(rows),
        "mean_delta": mean([r["delta_vs_parent"] for r in rows]),
        "mean_margin": mean([r["expected_margin"] for r in rows]),
        "success": sum(1 for r in rows if r["expected_correct"]),
        "changed_correctness": sum(1 for r in rows if r["changed_correctness_vs_parent"]),
    }


def summarize_dataset(name, path):
    scores = load_jsonl(path)
    rows = margin_rows(scores)
    parent = {(r["task_id"], r["condition"]): r for r in rows if r["model"] == "parent"}
    out_rows = []
    for r in rows:
        rr = dict(r)
        p = parent.get((r["task_id"], r["condition"]))
        if p is None:
            continue
        rr["parent_expected_margin"] = p["expected_margin"]
        rr["delta_vs_parent"] = rr["expected_margin"] - p["expected_margin"]
        rr["parent_expected_correct"] = p["expected_correct"]
        rr["changed_correctness_vs_parent"] = rr["expected_correct"] != p["expected_correct"]
        out_rows.append(rr)

    models = sorted(set(r["model"] for r in out_rows if r["model"] != "parent"))
    summary = {}
    for model in models:
        rs = [r for r in out_rows if r["model"] == model]
        by_split = {}
        for split in sorted(set(r["split"] for r in rs)):
            srs = [r for r in rs if r["split"] == split]
            by_cond = {cond: summarize_group([r for r in srs if r["condition"] == cond]) for cond in ["source_original", "source_altered", "no_source"]}
            source_only = [r for r in srs if r["condition"] in {"source_original", "source_altered"}]
            by_split[split] = {"source_only": summarize_group(source_only), "by_condition": by_cond}
        by_cond_all = {cond: summarize_group([r for r in rs if r["condition"] == cond]) for cond in ["source_original", "source_altered", "no_source"]}
        summary[model] = {"all_source_only": summarize_group([r for r in rs if r["condition"] in {"source_original", "source_altered"}]), "all_by_condition": by_cond_all, "by_split": by_split}
    return {"name": name, "input": str(path.relative_to(ROOT)), "rows": out_rows, "summary": summary}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_results = {name: summarize_dataset(name, path) for name, path in INPUTS.items()}
    # Flatten rows for inspection.
    flat_rows = []
    for name, res in all_results.items():
        for r in res["rows"]:
            flat_rows.append({"dataset": name, **r})
    csv_path = _public_path('experiments/archive/functional_learning/data/route_synthesis/common_target_directional_deltas.csv')
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "model", "task_id", "pair_id", "split", "axis", "semantic_label", "condition", "expected_margin", "parent_expected_margin", "delta_vs_parent", "expected_correct", "parent_expected_correct", "changed_correctness_vs_parent", "frame"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in flat_rows:
            w.writerow({k: r.get(k) for k in fields})
    summary = {
        "status": "COMMON_TARGET_DIRECTIONAL_SUMMARY",
        "purpose": "Separate original-source, altered-source, and no-source prior shifts before deciding the next allocation experiment.",
        "inputs": {k: str(v.relative_to(ROOT)) for k, v in INPUTS.items()},
        "directional_summary": {name: res["summary"] for name, res in all_results.items()},
        "inspection_csv": str(csv_path.relative_to(ROOT)),
        "interpretation": {
            "held_source_exact_epoch80": "For the exact compact epoch80 continuation record, the held-source average is not symmetric evidence-use: source_original improves much more than source_altered, while no_source original-answer preference also increases.",
            "route_consequence": "The next experiment should test allocation of saved budget to additional support under explicit resources, not simply seek a larger common-target number."
        }
    }
    out_path = _public_path('experiments/archive/functional_learning/data/route_synthesis/common_target_directional_summary.json')
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(out_path.relative_to(ROOT)), "csv": str(csv_path.relative_to(ROOT))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
