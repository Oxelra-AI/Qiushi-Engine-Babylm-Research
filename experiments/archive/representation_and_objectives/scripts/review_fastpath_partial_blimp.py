#!/usr/bin/env python3
"""Partial BLiMP comparison for the fast-path arms.

This is deliberately limited to the only completed official output currently present
for Arm2/Arm3: BLiMP predictions. It does not decide the route, because
relations, Entity, COMPS, GlobalPIQA, Reading, Supplement, SuperGLUE and AoA remain
unscored for the coherent/spanbreak pair.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path.cwd()
OUT_DIR = ROOT / "experiments/archive/representation_and_objectives/data/fastpath_partial_blimp_review"
OUT_MD = ROOT / "research/notes/representation_and_objectives/fastpath_partial_blimp_review.md"
DATA_ROOT = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/blimp_filtered"

PRED_PATHS = {
    "chck82": ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/parts/BLiMP/eval/official_outputs/scale1p75_chck82_independent__BLiMP/BLiMP/chck_82M/full_scale1p75_chck82_independent__BLiMP_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json",
    "shuffled86": ROOT / "experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/official_outputs/frozen82_tail4M_shuffled/BLiMP/final/full_frozen82_tail4M_shuffled_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json",
    "coherent4M": ROOT / "experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/official_outputs/fastpath4M_coherent/BLiMP/final/full_fastpath4M_coherent_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json",
    "spanbreak4M": ROOT / "experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_eval/official_outputs/fastpath4M_spanbreak/BLiMP/final/full_fastpath4M_spanbreak_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json",
}


def load_items() -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for path in sorted(DATA_ROOT.glob("*.jsonl")):
        uid = path.stem
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                row = json.loads(line)
                item_id = f"{uid}_{i}"
                row["id"] = item_id
                row["UID"] = row.get("UID", uid)
                items[item_id] = row
    return items


def load_predictions(path: Path) -> dict[str, str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    preds: dict[str, str] = {}
    for uid, block in obj.items():
        for r in block.get("predictions", []):
            preds[r["id"]] = r["pred"]
    return preds


def score_arm(items: dict[str, dict[str, Any]], preds: dict[str, str]) -> tuple[dict[str, bool], dict[str, Any]]:
    corr: dict[str, bool] = {}
    unknown = []
    for item_id, row in items.items():
        pred = preds.get(item_id)
        if pred is None:
            unknown.append(item_id)
            continue
        if pred == row["sentence_good"]:
            corr[item_id] = True
        elif pred == row["sentence_bad"]:
            corr[item_id] = False
        else:
            # The official runner returns one of the two sentences. If not exact,
            # use a strict false plus preserve examples to avoid silently scoring fuzzy text.
            corr[item_id] = False
            unknown.append(item_id)
    summary = {
        "n_items": len(items),
        "n_predictions": len(preds),
        "n_scored": len(corr),
        "n_missing_or_nonexact": len(unknown),
        "accuracy": mean(corr.values()) * 100.0 if corr else None,
        "nonexact_examples": unknown[:10],
    }
    return corr, summary


def group_scores(items: dict[str, dict[str, Any]], corr: dict[str, bool], group_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for item_id, ok in corr.items():
        groups[str(items[item_id].get(group_key, ""))].append(ok)
    rows = []
    for g, vals in sorted(groups.items()):
        rows.append({"group": g, "n": len(vals), "accuracy": mean(vals) * 100.0})
    return rows


def compare(items: dict[str, dict[str, Any]], a_name: str, a: dict[str, bool], b_name: str, b: dict[str, bool]) -> dict[str, Any]:
    common = sorted(set(a) & set(b))
    gain = [i for i in common if (not a[i]) and b[i]]
    loss = [i for i in common if a[i] and (not b[i])]
    both_correct = [i for i in common if a[i] and b[i]]
    both_wrong = [i for i in common if (not a[i]) and (not b[i])]
    by_uid = []
    uid_to_ids: dict[str, list[str]] = defaultdict(list)
    for item_id in common:
        uid_to_ids[items[item_id]["UID"]].append(item_id)
    for uid, ids in sorted(uid_to_ids.items()):
        g = sum((not a[i]) and b[i] for i in ids)
        l = sum(a[i] and (not b[i]) for i in ids)
        if g or l:
            by_uid.append({
                "uid": uid,
                "n": len(ids),
                "gain": g,
                "loss": l,
                "net": g - l,
                f"{a_name}_accuracy": mean(a[i] for i in ids) * 100.0,
                f"{b_name}_accuracy": mean(b[i] for i in ids) * 100.0,
            })
    by_uid.sort(key=lambda r: (r["net"], r["gain"] + r["loss"]), reverse=True)
    by_field = []
    field_to_ids: dict[str, list[str]] = defaultdict(list)
    for item_id in common:
        field_to_ids[items[item_id].get("field", "")].append(item_id)
    for field, ids in sorted(field_to_ids.items()):
        g = sum((not a[i]) and b[i] for i in ids)
        l = sum(a[i] and (not b[i]) for i in ids)
        by_field.append({
            "field": field,
            "n": len(ids),
            "gain": g,
            "loss": l,
            "net": g - l,
            f"{a_name}_accuracy": mean(a[i] for i in ids) * 100.0,
            f"{b_name}_accuracy": mean(b[i] for i in ids) * 100.0,
        })
    return {
        "from": a_name,
        "to": b_name,
        "n_common": len(common),
        "gain": len(gain),
        "loss": len(loss),
        "net": len(gain) - len(loss),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "from_accuracy": mean(a[i] for i in common) * 100.0,
        "to_accuracy": mean(b[i] for i in common) * 100.0,
        "delta_accuracy": (mean(b[i] for i in common) - mean(a[i] for i in common)) * 100.0,
        "by_field": by_field,
        "top_uid_positive": by_uid[:12],
        "top_uid_negative": sorted(by_uid, key=lambda r: (r["net"], -(r["gain"] + r["loss"])))[:12],
    }


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    items = load_items()
    preds = {name: load_predictions(path) for name, path in PRED_PATHS.items()}
    corr = {}
    arm_summary = {}
    for name, p in preds.items():
        c, s = score_arm(items, p)
        corr[name] = c
        arm_summary[name] = s
    comparisons = {
        "coherent_minus_spanbreak": compare(items, "spanbreak4M", corr["spanbreak4M"], "coherent4M", corr["coherent4M"]),
        "coherent_minus_chck82": compare(items, "chck82", corr["chck82"], "coherent4M", corr["coherent4M"]),
        "spanbreak_minus_chck82": compare(items, "chck82", corr["chck82"], "spanbreak4M", corr["spanbreak4M"]),
        "shuffled_minus_chck82": compare(items, "chck82", corr["chck82"], "shuffled86", corr["shuffled86"]),
        "coherent_minus_shuffled": compare(items, "shuffled86", corr["shuffled86"], "coherent4M", corr["coherent4M"]),
    }
    out = {
        "purpose": "Partial reviewer comparison of only completed A02 research official outputs: BLiMP predictions.",
        "not_route_decision": True,
        "items": {"n": len(items), "data_root": str(DATA_ROOT.relative_to(ROOT))},
        "prediction_paths": {k: str(v.relative_to(ROOT)) for k, v in PRED_PATHS.items()},
        "arm_summary": arm_summary,
        "comparisons": comparisons,
    }
    out_path = OUT_DIR / "fastpath_partial_blimp_review.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# research reviewer partial BLiMP comparison for A02 research fast-path arms",
        "",
        "Status: **PARTIAL_ONLY_BLIMP**. This file preserves the only completed research Arm2/Arm3 official output currently present. It is not enough to judge the fast-path route, because the load-bearing relation/state tasks and the rest of cheap7 are missing for coherent/spanbreak.",
        "",
        "## Arm BLiMP scores from prediction files",
        "",
        "| arm | scored items | accuracy | missing/nonexact |",
        "|---|---:|---:|---:|",
    ]
    for name, s in arm_summary.items():
        lines.append(f"| {name} | {s['n_scored']} | {fmt(s['accuracy'])} | {s['n_missing_or_nonexact']} |")
    lines += ["", "## Pair comparisons", "", "| comparison | from acc | to acc | delta | gains | losses | net |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, c in comparisons.items():
        lines.append(f"| {name} | {fmt(c['from_accuracy'])} | {fmt(c['to_accuracy'])} | {fmt(c['delta_accuracy'])} | {c['gain']} | {c['loss']} | {c['net']} |")
    c = comparisons["coherent_minus_spanbreak"]
    lines += ["", "## Coherent versus spanbreak by BLiMP field", "", "| field | n | spanbreak acc | coherent acc | gains | losses | net |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in c["by_field"]:
        lines.append(f"| {r['field']} | {r['n']} | {fmt(r['spanbreak4M_accuracy'])} | {fmt(r['coherent4M_accuracy'])} | {r['gain']} | {r['loss']} | {r['net']} |")
    lines += ["", "## Largest UID movements for coherent over spanbreak", "", "Positive net:"]
    for r in c["top_uid_positive"][:8]:
        lines.append(f"- {r['uid']}: net {r['net']} (gain {r['gain']}, loss {r['loss']}), {fmt(r['spanbreak4M_accuracy'])} -> {fmt(r['coherent4M_accuracy'])}")
    lines.append("")
    lines.append("Negative net:")
    for r in c["top_uid_negative"][:8]:
        lines.append(f"- {r['uid']}: net {r['net']} (gain {r['gain']}, loss {r['loss']}), {fmt(r['spanbreak4M_accuracy'])} -> {fmt(r['coherent4M_accuracy'])}")
    lines += [
        "",
        "## Reviewer interpretation",
        "",
        "The BLiMP-only signal is weakly compatible with coherent natural replay preserving more syntax than spanbreak (coherent 68.52, spanbreak 68.18, net +318 items) and roughly preserving chck82 BLiMP (68.49 -> 68.52). But it is far below shuffled86 BLiMP (69.23) and does not touch the route's load-bearing relation/state families. The research route therefore remains unjudged until at least EWoK, Entity, GlobalPIQA, COMPS, Reading, Supplement, and preferably SuperGLUE outputs are produced for both coherent and spanbreak with identical collation.",
        "",
        f"JSON: `{out_path.relative_to(ROOT)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DONE", "out_json": str(out_path.relative_to(ROOT)), "out_md": str(OUT_MD.relative_to(ROOT)), "coherent_blimp": arm_summary["coherent4M"]["accuracy"], "spanbreak_blimp": arm_summary["spanbreak4M"]["accuracy"]}))


if __name__ == "__main__":
    main()
