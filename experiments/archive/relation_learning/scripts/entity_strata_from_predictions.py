#!/usr/bin/env python3
"""research: Entity operation-stratum summary from saved official prediction files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(".").resolve()
SCRIPTS = ROOT / "experiments/archive/relation_learning/scripts"
sys.path.insert(0, str(SCRIPTS))
import eval_state_update_entity as S53  # noqa: E402

KEY_GROUPS = [
    "ALL", "rel_eq0", "rel_eq0_total_ops0", "rel_eq0_irrelevant_ops_gt0",
    "rel_eq0_irrelevant_ops_1to3", "rel_eq0_irrelevant_ops_4to6", "rel_eq0_irrelevant_ops_ge7",
    "rel_ge1", "rel_ge1_postrel_ops0", "rel_ge1_postrel_ops_gt0", "rel_updates_1", "rel_updates_2",
    "rel_updates_3", "rel_updates_4", "rel_updates_5", "rel_ge3", "rel_ge3_postrel_ops0",
    "rel_ge3_postrel_ops_gt0", "stale_available_not_gold",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    pp = pathlib.Path(p)
    try:
        return str(pp.resolve().relative_to(ROOT))
    except Exception:
        return str(pp)


def norm_path(p: str) -> pathlib.Path:
    q = pathlib.Path(p)
    return q if q.is_absolute() else ROOT / q


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def mean(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def se(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    if len(xs) <= 1:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs) / (len(xs)-1)) / math.sqrt(len(xs))


def parse_target(spec: str) -> dict[str, str]:
    # label,seed,arm,checkpoint,path -- comma is safe for these paths.
    parts = spec.split(",", 4)
    if len(parts) != 5:
        raise ValueError("target must be label,seed,arm,checkpoint,predictions_path")
    return {"label": parts[0], "seed": parts[1], "arm": parts[2], "checkpoint": parts[3], "predictions": parts[4]}


def rows_for_target(t: dict[str, str], meta: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    pred_path = norm_path(t["predictions"])
    out = []
    for uid, item_index, pred_id, pred_text in S53.flatten_predictions(pred_path):
        m = meta.get((uid, item_index))
        if not m:
            continue
        gold = m.get("gold", "")
        correct = int(S53.norm_answer(pred_text) == S53.norm_answer(gold))
        pred_is_stale = int(bool(m.get("stale_initial")) and S53.norm_answer(pred_text) == S53.norm_answer(m.get("stale_initial")))
        r = {
            "label": t["label"], "seed": t["seed"], "arm": t["arm"], "checkpoint": t["checkpoint"],
            "uid": uid, "item_index": item_index, "pred_id": pred_id, "pred": pred_text,
            "correct": correct, "pred_is_stale_initial": pred_is_stale,
        }
        r.update(m)
        out.append(r)
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for g in S53.item_groups(r):
            groups[(r["label"], r["checkpoint"], g)].append(r)
    out = []
    for (label, ck, g), vals in sorted(groups.items()):
        wrong = [v for v in vals if not int(v["correct"])]
        stale_cand = [v for v in vals if int(v.get("stale_available", 0)) and not int(v.get("stale_is_gold", 0))]
        wrong_stale = [v for v in stale_cand if not int(v["correct"])]
        out.append({
            "label": label,
            "checkpoint": ck,
            "group": g,
            "n": len(vals),
            "accuracy_pct": 100.0 * sum(int(v["correct"]) for v in vals) / len(vals),
            "mean_relevant_updates": mean([v["relevant_updates"] for v in vals]),
            "mean_total_ops": mean([v["total_ops"] for v in vals]),
            "mean_irrelevant_ops": mean([v["irrelevant_ops"] for v in vals]),
            "mean_ops_after_last_relevant": mean([v["ops_after_last_relevant"] for v in vals]),
            "stale_available_not_gold_n": len(stale_cand),
            "stale_pick_pct_among_wrong_stale_available_not_gold": 100.0 * sum(int(v["pred_is_stale_initial"]) for v in wrong_stale) / len(wrong_stale) if wrong_stale else float("nan"),
        })
    return out


def pairwise(summary: list[dict[str, Any]], baseline_label: str) -> list[dict[str, Any]]:
    idx = {(r["label"], r["checkpoint"], r["group"]): r for r in summary}
    out = []
    labels = sorted({r["label"] for r in summary if r["label"] != baseline_label})
    groups = sorted({r["group"] for r in summary})
    cks = sorted({r["checkpoint"] for r in summary})
    for label in labels:
        for ck in cks:
            for g in groups:
                b = idx.get((baseline_label, ck, g))
                a = idx.get((label, ck, g))
                if not b or not a:
                    continue
                out.append({
                    "label": label,
                    "baseline": baseline_label,
                    "checkpoint": ck,
                    "group": g,
                    "n": int(a["n"]),
                    "baseline_accuracy_pct": float(b["accuracy_pct"]),
                    "label_accuracy_pct": float(a["accuracy_pct"]),
                    "delta_accuracy_pct": float(a["accuracy_pct"]) - float(b["accuracy_pct"]),
                    "mean_relevant_updates": float(a["mean_relevant_updates"]),
                    "mean_total_ops": float(a["mean_total_ops"]),
                })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", required=True, help="label,seed,arm,checkpoint,predictions_path")
    ap.add_argument("--baseline-label", default="chck82")
    ap.add_argument("--out-dir", default="experiments/archive/relation_learning/data/entity_strata")
    args = ap.parse_args()
    out_dir = norm_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = S53.load_entity_metadata()
    targets = [parse_target(x) for x in args.target]
    rows: list[dict[str, Any]] = []
    for t in targets:
        rows.extend(rows_for_target(t, meta))
    summary = summarize(rows)
    deltas = pairwise(summary, args.baseline_label)
    write_csv(out_dir / "prediction_rows.csv", rows)
    write_csv(out_dir / "strata_summary.csv", summary)
    write_csv(out_dir / "pairwise_deltas.csv", deltas)
    obj = {"status": "ENTITY_STRATA_FROM_PREDICTIONS", "created_utc": now(), "targets": targets, "n_prediction_rows": len(rows), "summary_rows": len(summary), "delta_rows": len(deltas), "summary_csv": rel(out_dir / "strata_summary.csv"), "deltas_csv": rel(out_dir / "pairwise_deltas.csv")}
    (out_dir / "summary.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research Entity operation strata", "", "| label | group | n | accuracy |", "|---|---|---:|---:|"]
    for r in summary:
        if r["group"] in KEY_GROUPS:
            lines.append(f"| {r['label']} | {r['group']} | {r['n']} | {r['accuracy_pct']:.2f} |")
    lines += ["", "## Deltas", "", f"Baseline label: `{args.baseline_label}`.", "", "| label | group | n | base | label | delta |", "|---|---|---:|---:|---:|---:|"]
    for r in deltas:
        if r["group"] in KEY_GROUPS:
            lines.append(f"| {r['label']} | {r['group']} | {r['n']} | {r['baseline_accuracy_pct']:.2f} | {r['label_accuracy_pct']:.2f} | {r['delta_accuracy_pct']:+.2f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(obj, indent=2), flush=True)


if __name__ == "__main__":
    main()
