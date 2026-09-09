#!/usr/bin/env python3
"""research micro-audit of the GlobalPIQA contribution to private-scale alpha gains.

research showed alpha0.5's cheap7 advantage is dominated by GlobalPIQA. research
manifest showed only 5 GlobalPIQA examples are alpha-sensitive. This script makes
that exact: which examples flip, at which alpha, and how many score points/cheap7
points they account for.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys
import time
from collections import OrderedDict
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import PayloadLoader  # noqa: E402

ALPHA_PAYLOADS = OrderedDict([
    ("a0_anchor", _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json')),
    ("a0p5", _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5.json')),
    ("a0p75", _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json')),
    ("a1", _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json')),
])
MANIFEST = _public_path('experiments/archive/frontier_consolidation/data/anchor_margin_alpha_census/anchor_margin_item_manifest.jsonl')
OUT = _public_path('experiments/archive/frontier_consolidation/data/globalpiqa_alpha_microaudit')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_manifest_changed() -> list[dict[str, Any]]:
    rows = []
    with MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("column") == "GlobalPIQA":
                rows.append(r)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    maps = {}
    for label, path in ALPHA_PAYLOADS.items():
        rows, _ = PayloadLoader(path).load_column("GlobalPIQA")
        maps[label] = {r.item_id: r for r in rows}
    common = sorted(set.intersection(*(set(m) for m in maps.values())))
    changed = read_manifest_changed()
    changed_ids = [r["item_id"] for r in changed]
    denom = len(common)
    if denom == 0:
        raise RuntimeError("No common GlobalPIQA items")

    # Verify manifest matches actual changed set.
    actual_changed = []
    for item_id in common:
        bits = "".join("1" if maps[a][item_id].correct else "0" for a in ALPHA_PAYLOADS)
        if bits not in {"0000", "1111"}:
            actual_changed.append(item_id)
    if set(actual_changed) != set(changed_ids):
        raise RuntimeError({"manifest_changed": changed_ids, "actual_changed": actual_changed})

    alpha_scores = {}
    alpha_deltas = {}
    for a in ALPHA_PAYLOADS:
        correct = sum(1 for item_id in common if maps[a][item_id].correct)
        score = 100.0 * correct / denom
        alpha_scores[a] = {"correct": correct, "score_points": score}
        alpha_deltas[a] = {
            "net_items_vs_anchor": correct - alpha_scores["a0_anchor"]["correct"],
            "score_delta_points_vs_anchor": score - alpha_scores["a0_anchor"]["score_points"],
            "cheap7_contribution_points_vs_anchor": (score - alpha_scores["a0_anchor"]["score_points"]) / 7.0,
        }

    rows_out = []
    for r in changed:
        item_id = r["item_id"]
        correctness = {a: bool(maps[a][item_id].correct) for a in ALPHA_PAYLOADS}
        preds = {a: str(maps[a][item_id].pred) for a in ALPHA_PAYLOADS}
        gold = str(maps["a0_anchor"][item_id].gold)
        pattern = "".join("1" if correctness[a] else "0" for a in ALPHA_PAYLOADS)
        rows_out.append({
            "item_id": item_id,
            "group": r.get("group"),
            "pattern": pattern,
            "pattern_class": r.get("pattern_class"),
            "anchor_correct": r.get("anchor_correct"),
            "gold": gold,
            "correctness": correctness,
            "predictions": preds,
            "candidates": r.get("candidates"),
        })

    reading = []
    reading.append(f"GlobalPIQA has {denom} common examples; only {len(rows_out)} change across alpha0/0.5/0.75/1.0.")
    reading.append(f"Alpha0.5 GlobalPIQA gain is {alpha_deltas['a0p5']['net_items_vs_anchor']} net examples = {alpha_deltas['a0p5']['score_delta_points_vs_anchor']:+.6f} score points = {alpha_deltas['a0p5']['cheap7_contribution_points_vs_anchor']:+.6f} cheap7 points.")
    reading.append(f"Alpha0.75 GlobalPIQA gain is {alpha_deltas['a0p75']['net_items_vs_anchor']} net examples = {alpha_deltas['a0p75']['score_delta_points_vs_anchor']:+.6f} score points = {alpha_deltas['a0p75']['cheap7_contribution_points_vs_anchor']:+.6f} cheap7 points.")
    reading.append("This supports the interpretation that GlobalPIQA-driven alpha ranking is a few-example endpoint effect, not broad commonsense acquisition.")

    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "manifest": rel(MANIFEST),
        "alpha_payloads": {k: rel(v) for k, v in ALPHA_PAYLOADS.items()},
        "globalpiqa_common_examples": denom,
        "changed_examples": len(rows_out),
        "alpha_scores": alpha_scores,
        "alpha_deltas_vs_anchor": alpha_deltas,
        "changed_rows": rows_out,
        "scientific_reading": reading,
    }
    js = _public_path('experiments/archive/frontier_consolidation/data/globalpiqa_alpha_microaudit/globalpiqa_alpha_microaudit.json')
    md = _public_path('research/documents/frontier_consolidation/data/globalpiqa_alpha_microaudit/globalpiqa_alpha_microaudit.md')
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research GlobalPIQA alpha micro-audit",
        "",
        f"Status: **{out['status']}**",
        f"Common GlobalPIQA examples: `{denom}`; alpha-sensitive examples: `{len(rows_out)}`.",
        "",
        "## Alpha scores",
        "",
        "| alpha | correct | score | net items vs anchor | score delta | cheap7 contribution |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in ALPHA_PAYLOADS:
        s = alpha_scores[a]
        d = alpha_deltas[a]
        lines.append(f"| {a} | {s['correct']} | {s['score_points']:.6f} | {d['net_items_vs_anchor']:+d} | {d['score_delta_points_vs_anchor']:+.6f} | {d['cheap7_contribution_points_vs_anchor']:+.6f} |")
    lines += ["", "## Changed examples", "", "| item | pattern | gold | alpha predictions |", "|---|---|---|---|"]
    for r in rows_out:
        pred_short = "; ".join(f"{a}:{'✓' if r['correctness'][a] else '✗'}" for a in ALPHA_PAYLOADS)
        lines.append(f"| {r['item_id']} | {r['pattern']} | {str(r['gold'])[:120]} | {pred_short} |")
    lines += ["", "## Scientific reading", ""]
    for x in reading:
        lines.append(f"- {x}")
    lines += ["", f"JSON: `{rel(js)}`"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(js), "out_md": rel(md), "changed_examples": len(rows_out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
