#!/usr/bin/env python3
"""Synthesize ordinary86 control evidence.

Reads the item-level comparison JSON and writes a focused synthesis of
what the exposure-matched ordinary scale1.75 chck_86M control changes about the
coherent86 frozen-anchor fast-path interpretation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
SRC = _public_path('experiments/archive/representation_and_objectives/data/fastpath_vs_ordinary86_item_review/fastpath_vs_ordinary86_item_review.json')
OUTDIR = _public_path('experiments/archive/frontier_consolidation/data/fastpath_ordinary86_control_synthesis')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/fastpath_ordinary86_control_synthesis/fastpath_ordinary86_control_synthesis.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/fastpath_ordinary86_control_synthesis/fastpath_ordinary86_control_synthesis.md')
COMPARE_KEYS = ["coherent_minus_ordinary86", "ordinary86_minus_chck82", "coherent_minus_chck82"]
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def compact_comparison(c: dict[str, Any]) -> dict[str, Any]:
    comp = {
        "aggregate": c.get("aggregate"),
        "column_payload_deltas": {},
        "column_item_nets": {},
        "column_flip_counts": {},
    }
    for col, v in c.get("columns", {}).items():
        comp["column_payload_deltas"][col] = v.get("delta_score_payload")
        comp["column_item_nets"][col] = v.get("gain_minus_loss_items")
        comp["column_flip_counts"][col] = v.get("flip_counts")
    for name in ["entity_high_operation_total", "ewok_relation_total", "globalpiqa_groups"]:
        if name in c:
            comp[name] = c[name]
    return comp


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    j = json.loads(SRC.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "status": "COMPLETE",
        "source": rel(SRC),
        "route_read": j.get("route_read"),
        "cheap7": j.get("cheap7"),
        "scores": j.get("scores"),
        "comparisons": {},
        "scientific_reading": (
            "A01 ordinary86 is the exposure-matched ordinary-continuation control for coherent86. "
            "Coherent86 has a robust aggregate endpoint edge, but common official-item transitions "
            "show negative item balance versus ordinary86 and versus the anchor. The current evidence "
            "therefore supports coherent86 as a stronger endpoint candidate, not as a demonstrated "
            "general slow-fast learning principle."
        ),
        "next_execution_reading": (
            "Before training a retention-modified fast path, test whether the learned private residual "
            "has a useful inference-time amplitude: alpha 0 is the protected anchor, alpha 1 is coherent86, "
            "and intermediate alpha can reveal whether score gains and item erosion are separable without "
            "new training."
        ),
    }
    for k in COMPARE_KEYS:
        summary["comparisons"][k] = compact_comparison(j["comparisons"][k])
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research fast-path ordinary86 control synthesis",
        "",
        f"Status: **{summary['status']}**",
        f"Source: `{rel(SRC)}`",
        "",
        "## Cheap7 table",
        "",
        "| arm | cheap7 |",
        "|---|---:|",
    ]
    for arm, val in summary["cheap7"].items():
        lines.append(f"| {arm} | {val} |")
    lines += ["", "## Decisive comparisons", ""]
    for k in COMPARE_KEYS:
        comp = summary["comparisons"][k]
        lines.append(f"### {k}")
        lines.append("")
        lines.append(f"Aggregate: `{comp['aggregate']}`")
        lines.append("")
        lines.append("| column | payload Δ | item net |")
        lines.append("|---|---:|---:|")
        for col in COLUMNS:
            lines.append(f"| {col} | {comp['column_payload_deltas'].get(col)} | {comp['column_item_nets'].get(col)} |")
        if "entity_high_operation_total" in comp:
            lines.append("")
            lines.append(f"Entity high-operation total: `{comp['entity_high_operation_total']}`")
        if "ewok_relation_total" in comp:
            lines.append("")
            lines.append(f"EWoK relation total: `{comp['ewok_relation_total']}`")
        lines.append("")
    lines += [
        "## Scientific reading",
        "",
        summary["scientific_reading"],
        "",
        summary["next_execution_reading"],
        "",
        f"JSON: `{rel(OUT_JSON)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
