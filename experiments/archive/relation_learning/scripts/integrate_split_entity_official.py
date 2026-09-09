#!/usr/bin/env python3
"""research: integrate official Entity predictions for seed43022 split arms.

Combines existing original V/R/C relevant-update rows with new official RS/VS
predictions and summarizes accuracy by queried-entity update depth. This tests
whether the behavioral face of local identity practice is itself localized.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import re
import statistics
import time
from collections import defaultdict
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/integrate_split_entity_official.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
ORIG_ROWS = _public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_prediction_augmented_rows.csv')
META_ROWS = _public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_item_metadata.csv')
SPLIT_EVAL_ROWS = _public_path('experiments/archive/relation_learning/data/split_entity_official/split_entity_eval_rows.csv')
OUT_DIR = _public_path('experiments/archive/relation_learning/data/split_entity_official_integration')
NOTE = _public_path('research/notes/relation_learning/split_entity_official_integration.md')
CKS = ["chck_80M", "chck_90M", "chck_100M"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    preferred = ["seed", "arm", "checkpoint", "group", "contrast", "n", "n_checkpoints", "accuracy_pct", "late_mean_accuracy_pct", "delta_accuracy_pct", "late_mean_delta_accuracy_pct", "acc_a", "acc_b", "late_mean_acc_a", "late_mean_acc_b", "relevant_updates", "mean_relevant_updates", "mean_total_ops", "mean_prefix_words", "prediction_path"]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else float("nan")


def load_original_rows() -> list[dict[str, Any]]:
    if not ORIG_ROWS.exists():
        raise FileNotFoundError(ORIG_ROWS)
    rows: list[dict[str, Any]] = []
    with ORIG_ROWS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("seed")) != "43022":
                continue
            if r.get("checkpoint") not in CKS:
                continue
            if r.get("arm") not in {"V", "C", "R"}:
                continue
            rows.append({
                "seed": 43022,
                "arm": r["arm"],
                "checkpoint": r["checkpoint"],
                "uid": r["uid"],
                "item_index": int(r["item_index"]),
                "correct": int(r["correct"]),
                "reported_numops": int(r["reported_numops"]),
                "relevant_updates": int(r["relevant_updates"]),
                "total_ops": int(r["total_ops"]),
                "prefix_words": int(r["prefix_words"]),
                "stale_available": int(r.get("stale_available", 0)),
                "stale_is_gold": int(r.get("stale_is_gold", 0)),
                "pred_is_stale_initial": int(r.get("pred_is_stale_initial", 0)),
                "source_family": "original",
            })
    return rows


def load_meta() -> dict[tuple[str, int], dict[str, Any]]:
    if not META_ROWS.exists():
        raise FileNotFoundError(META_ROWS)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with META_ROWS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["uid"], int(r["item_index"]))
            out[key] = r
    return out


def split_group_rows() -> list[dict[str, Any]]:
    if not SPLIT_EVAL_ROWS.exists():
        raise FileNotFoundError(SPLIT_EVAL_ROWS)
    rows: list[dict[str, Any]] = []
    with SPLIT_EVAL_ROWS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("predictions") and r.get("returncode") == "0":
                rows.append(r)
    needed = {(role, ck) for role in ["RS", "VS"] for ck in CKS}
    seen = {(r["role"], r["checkpoint"]) for r in rows}
    if not needed.issubset(seen):
        raise RuntimeError({"missing_split_predictions": sorted(needed - seen), "seen": sorted(seen)})
    return rows


def load_split_prediction_rows(meta: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for er in split_group_rows():
        arm = er["role"]
        ck = er["checkpoint"]
        pred_path = ROOT / er["predictions"]
        obj = read_json(pred_path)
        for uid, block in obj.items():
            preds = block.get("predictions", [])
            for i, pr in enumerate(preds):
                key = (uid, i)
                if key not in meta:
                    raise RuntimeError(f"metadata missing for {key}")
                m = meta[key]
                pred = str(pr.get("pred", ""))
                gold = str(m.get("gold", ""))
                stale = str(m.get("stale_initial", ""))
                rows.append({
                    "seed": 43022,
                    "arm": arm,
                    "checkpoint": ck,
                    "uid": uid,
                    "item_index": i,
                    "correct": int(norm(pred) == norm(gold)),
                    "reported_numops": int(m["reported_numops"]),
                    "relevant_updates": int(m["relevant_updates"]),
                    "total_ops": int(m["total_ops"]),
                    "prefix_words": int(m["prefix_words"]),
                    "stale_available": int(m.get("stale_available", 0)),
                    "stale_is_gold": int(m.get("stale_is_gold", 0)),
                    "pred_is_stale_initial": int(norm(pred) == norm(stale) and stale != ""),
                    "source_family": "split",
                    "prediction_path": rel(pred_path),
                })
    return rows


def groups_for(r: dict[str, Any]) -> list[str]:
    relu = int(r["relevant_updates"])
    total = int(r["total_ops"])
    groups = ["ALL", f"rel_updates_{relu}", f"total_ops_{total}"]
    if relu == 0:
        groups.append("rel_eq0")
        groups.append(f"rel0_total_ops_{total}")
    if relu >= 1:
        groups.append("rel_ge1")
    if relu >= 2:
        groups.append("rel_ge2")
    if relu >= 3:
        groups.append("rel_ge3")
    if relu >= 4:
        groups.append("rel_ge4")
    return groups


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for g in groups_for(r):
            bins[(r["arm"], r["checkpoint"], g)].append(r)
    out: list[dict[str, Any]] = []
    for (arm, ck, group), vals in sorted(bins.items()):
        wrong = [v for v in vals if not int(v["correct"])]
        stale_wrong = [v for v in wrong if int(v.get("stale_available", 0)) and not int(v.get("stale_is_gold", 0))]
        out.append({
            "seed": 43022,
            "arm": arm,
            "checkpoint": ck,
            "group": group,
            "n": len(vals),
            "accuracy_pct": 100.0 * sum(int(v["correct"]) for v in vals) / len(vals),
            "mean_relevant_updates": mean([v["relevant_updates"] for v in vals]),
            "mean_total_ops": mean([v["total_ops"] for v in vals]),
            "mean_prefix_words": mean([v["prefix_words"] for v in vals]),
            "stale_pick_pct_among_wrong_stale_available_not_gold": 100.0 * sum(int(v["pred_is_stale_initial"]) for v in stale_wrong) / len(stale_wrong) if stale_wrong else float("nan"),
        })
    return out


def late_summary(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in summary:
        if r["checkpoint"] in CKS:
            bins[(r["arm"], r["group"])].append(r)
    out: list[dict[str, Any]] = []
    for (arm, group), vals in sorted(bins.items()):
        out.append({
            "seed": 43022,
            "arm": arm,
            "group": group,
            "n_checkpoints": len(vals),
            "n": int(vals[0]["n"]),
            "late_mean_accuracy_pct": mean([v["accuracy_pct"] for v in vals]),
            "mean_relevant_updates": mean([v["mean_relevant_updates"] for v in vals]),
            "mean_total_ops": mean([v["mean_total_ops"] for v in vals]),
            "mean_prefix_words": mean([v["mean_prefix_words"] for v in vals]),
            "late_mean_stale_pick_pct_among_wrong_stale_available_not_gold": mean([v["stale_pick_pct_among_wrong_stale_available_not_gold"] for v in vals]),
        })
    return out


def contrasts(late: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = {(r["arm"], r["group"]): r for r in late}
    groups = sorted({r["group"] for r in late})
    pairs = [
        ("RminusC", "R", "C"), ("RSminusC", "RS", "C"), ("RSminusR", "RS", "R"),
        ("VminusC", "V", "C"), ("VSminusC", "VS", "C"), ("VSminusV", "VS", "V"),
        ("RminusV", "R", "V"), ("RSminusVS", "RS", "VS"),
    ]
    out: list[dict[str, Any]] = []
    for group in groups:
        for cname, a, b in pairs:
            if (a, group) not in idx or (b, group) not in idx:
                continue
            ra, rb = idx[(a, group)], idx[(b, group)]
            out.append({
                "seed": 43022,
                "group": group,
                "contrast": cname,
                "n": int(ra["n"]),
                "n_checkpoints": min(int(ra["n_checkpoints"]), int(rb["n_checkpoints"])),
                "late_mean_delta_accuracy_pct": float(ra["late_mean_accuracy_pct"]) - float(rb["late_mean_accuracy_pct"]),
                "late_mean_acc_a": float(ra["late_mean_accuracy_pct"]),
                "late_mean_acc_b": float(rb["late_mean_accuracy_pct"]),
                "mean_relevant_updates": float(ra["mean_relevant_updates"]),
                "mean_total_ops": float(ra["mean_total_ops"]),
                "mean_prefix_words": float(ra["mean_prefix_words"]),
            })
    return out


def make_note(late: list[dict[str, Any]], con: list[dict[str, Any]]) -> None:
    def get_con(group: str, name: str) -> dict[str, Any] | None:
        for r in con:
            if r["group"] == group and r["contrast"] == name:
                return r
        return None
    lines: list[str] = []
    lines.append("# research split official Entity integration")
    lines.append("")
    lines.append("This readout uses official Entity predictions for seed43022 split arms and the existing original V/R/C prediction rows. It asks whether the behavioral pattern associated with exact local recurrence is also removed by splitting source and companion rows.")
    lines.append("")
    lines.append("## Late accuracy by relevant queried-entity updates")
    lines.append("")
    lines.append("| group | R−C | RS−C | RS−R | V−C | VS−C | VS−V | RS−VS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for group in ["ALL", "rel_eq0", "rel_ge1", "rel_ge2", "rel_ge3", "rel_updates_0", "rel_updates_1", "rel_updates_2", "rel_updates_3", "rel_updates_4", "rel_updates_5"]:
        vals = []
        for cname in ["RminusC", "RSminusC", "RSminusR", "VminusC", "VSminusC", "VSminusV", "RSminusVS"]:
            r = get_con(group, cname)
            vals.append(f"{float(r['late_mean_delta_accuracy_pct']):+.2f}" if r else "")
        if any(vals):
            lines.append(f"| {group} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("A localized behavioral reading would show RS losing most of original R's zero-update advantage and update-depth cost relative to CLEAN, while VS stays closer to CLEAN than original V on the same buckets. If RS retains original R's zero-update advantage, the behavioral retrieval benefit can arise from spaced repetition or broad fit even when the source-specific NLL residual disappears.")
    lines.append("")
    lines.append(f"Data outputs: `{rel(OUT_DIR)}`")
    _public_path('research/notes/relation_learning').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    rows = load_original_rows() + load_split_prediction_rows(meta)
    summary = summarize(rows)
    late = late_summary(summary)
    con = contrasts(late)
    write_csv(_public_path('experiments/archive/relation_learning/data/split_entity_official_integration/entity_rows_original_and_split_seed43022.csv'), rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/split_entity_official_integration/entity_summary_by_checkpoint.csv'), summary)
    write_csv(_public_path('experiments/archive/relation_learning/data/split_entity_official_integration/entity_late_accuracy_by_group.csv'), late)
    write_csv(_public_path('experiments/archive/relation_learning/data/split_entity_official_integration/entity_late_contrasts_by_group.csv'), con)
    make_note(late, con)
    payload = {
        "status": "SPLIT_ENTITY_INTEGRATION_DONE",
        "created_utc": now(),
        "note": rel(NOTE),
        "rows": len(rows),
        "summary": rel(_public_path('experiments/archive/relation_learning/data/split_entity_official_integration/entity_late_accuracy_by_group.csv')),
        "contrasts": rel(_public_path('experiments/archive/relation_learning/data/split_entity_official_integration/entity_late_contrasts_by_group.csv')),
    }
    (_public_path('experiments/archive/relation_learning/data/split_entity_official_integration/split_entity_integration_summary.json')).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
