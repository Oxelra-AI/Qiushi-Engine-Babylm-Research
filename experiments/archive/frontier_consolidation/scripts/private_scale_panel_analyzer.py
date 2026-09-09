#!/usr/bin/env python3
"""research panel analysis for frozen-anchor private-path endpoints.

This script compares saved official-compatible prediction payloads at the common
item level.  It is intentionally inference-free: it consumes prediction payloads
already produced by official-compatible evaluation jobs.  The immediate use is to
put the protected anchor, coherent86, shuffled86, ordinary86, spanbreak86, and any
private-scale alpha payloads on one comparable decision surface.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
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

from pairwise_item_flip_analysis import (  # noqa: E402
    CHEAP_COLS,
    DISCRETE_COLUMNS,
    PayloadLoader,
    compare_column,
)

DEFAULT_ARMS = OrderedDict([
    (
        "chck82_anchor",
        _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json'),
    ),
    (
        "ordinary86_backbone",
        _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json'),
    ),
    (
        "shuffled86_private",
        _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json'),
    ),
    (
        "coherent86_private_alpha1",
        _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json'),
    ),
    (
        "spanbreak86_private",
        _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_eval/per_target/fastpath4M_spanbreak.json'),
    ),
    (
        "coherent86_private_alpha0p5",
        _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5.json'),
    ),
    (
        "coherent86_private_alpha0p75",
        _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json'),
    ),
])
DEFAULT_OUT = _public_path('experiments/archive/frontier_consolidation/data/private_scale_panel_analysis')

FOCUS_EWOK = [
    "material", "spatial", "space", "quantity", "quantitative", "number",
    "physical", "social", "support", "contact", "contained", "size", "mass",
    "volume", "distance", "dynamics", "interaction",
]
FOCUS_ENTITY = ["regular_5", "move_contents_4", "move_contents_3", "ambiref_3", "5_ops", "4_ops", "3_ops"]
FOCUS_SUPPLEMENT = ["qa_congruence", "hypernym", "turn_taking", "ellipsis", "binding", "passive"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def payload_ready(path: pathlib.Path) -> bool:
    if not path.exists():
        return False
    try:
        obj = read_json(path)
    except Exception:
        return False
    tasks = obj.get("tasks", {}) if isinstance(obj, dict) else {}
    needed = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
    return all(k in tasks for k in needed)


def scores_for(path: pathlib.Path | None) -> dict[str, float | None] | None:
    if path is None:
        return None
    obj = read_json(path)
    scores = obj.get("official_overall", {}).get("scores", {})
    return {c: (None if scores.get(c) is None else float(scores.get(c))) for c in CHEAP_COLS}


def cheap7(scores: dict[str, float | None] | None) -> float | None:
    if not scores:
        return None
    vals = [scores.get(c) for c in CHEAP_COLS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def group_focus(groups: list[dict[str, Any]], tokens: list[str]) -> dict[str, Any]:
    rows = []
    for g in groups:
        name = str(g.get("group", "")).lower()
        if any(tok in name for tok in tokens):
            rows.append(g)
    return {
        "n_groups": len(rows),
        "total_n": int(sum(int(g.get("n", 0)) for g in rows)),
        "gain": int(sum(int(g.get("gain", 0)) for g in rows)),
        "loss": int(sum(int(g.get("loss", 0)) for g in rows)),
        "net_gain_minus_loss": int(sum(int(g.get("net_gain_minus_loss", 0)) for g in rows)),
        "rows": rows[:40],
    }


def compact_column(c: dict[str, Any]) -> dict[str, Any]:
    flips = c.get("flip_counts", {})
    both_correct = int(flips.get("both_correct", 0))
    both_wrong = int(flips.get("both_wrong", 0))
    gain = int(flips.get("gain", 0))
    loss = int(flips.get("loss", 0))
    base_correct = both_correct + loss
    cand_correct = both_correct + gain
    changed = gain + loss
    return {
        "delta_payload": None if c.get("delta_score_payload") is None else float(c.get("delta_score_payload")),
        "delta_reconstructed_common": float(c.get("delta_score_reconstructed_common")),
        "n_common": int(c.get("n_common", 0)),
        "gain": gain,
        "loss": loss,
        "net_gain_minus_loss": gain - loss,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "base_correct_items": base_correct,
        "candidate_correct_items": cand_correct,
        "changed_items": changed,
        "anchor_correct_retention_fraction": None if base_correct == 0 else both_correct / base_correct,
        "gain_to_loss_ratio": None if loss == 0 else gain / loss,
        "best_groups_by_item_net": c.get("best_groups_by_item_net", [])[:12],
        "worst_groups_by_item_net": c.get("worst_groups_by_item_net", [])[:12],
        "reconstruction_errors": c.get("payload_reconstruction_errors", {}),
    }


def compare_payloads(base_name: str, base_path: pathlib.Path, cand_name: str, cand_path: pathlib.Path) -> dict[str, Any]:
    base_loader = PayloadLoader(base_path)
    cand_loader = PayloadLoader(cand_path)
    columns: dict[str, Any] = {}
    compact: dict[str, Any] = {}
    for col in DISCRETE_COLUMNS:
        full = compare_column(base_loader, cand_loader, col)
        columns[col] = full
        compact[col] = compact_column(full)
    total_gain = sum(compact[c]["gain"] for c in DISCRETE_COLUMNS)
    total_loss = sum(compact[c]["loss"] for c in DISCRETE_COLUMNS)
    total_both_correct = sum(compact[c]["both_correct"] for c in DISCRETE_COLUMNS)
    total_base_correct = sum(compact[c]["base_correct_items"] for c in DISCRETE_COLUMNS)
    total_common = sum(compact[c]["n_common"] for c in DISCRETE_COLUMNS)
    focus = {}
    if "EWoK" in columns:
        focus["EWoK_relation_like"] = group_focus(columns["EWoK"].get("groups_all_by_item_net", []), FOCUS_EWOK)
    if "Entity" in columns:
        focus["Entity_high_operation"] = group_focus(columns["Entity"].get("groups_all_by_item_net", []), FOCUS_ENTITY)
    if "Supplement" in columns:
        focus["Supplement_selected"] = group_focus(columns["Supplement"].get("groups_all_by_item_net", []), FOCUS_SUPPLEMENT)
    aggregate = {
        "total_common_items": int(total_common),
        "total_gain_items": int(total_gain),
        "total_loss_items": int(total_loss),
        "total_gain_minus_loss": int(total_gain - total_loss),
        "total_changed_items": int(total_gain + total_loss),
        "total_base_correct_items": int(total_base_correct),
        "anchor_correct_retention_fraction": None if total_base_correct == 0 else float(total_both_correct / total_base_correct),
        "gain_to_loss_ratio": None if total_loss == 0 else float(total_gain / total_loss),
        "discrete_payload_mean_delta": float(mean(float(compact[c]["delta_payload"]) for c in DISCRETE_COLUMNS if compact[c]["delta_payload"] is not None)),
        "discrete_reconstructed_mean_delta": float(mean(float(compact[c]["delta_reconstructed_common"]) for c in DISCRETE_COLUMNS)),
    }
    return {
        "label": f"{cand_name}_minus_{base_name}",
        "base": base_name,
        "candidate": cand_name,
        "base_payload": rel(base_path),
        "candidate_payload": rel(cand_path),
        "aggregate": aggregate,
        "columns_compact": compact,
        "focus": focus,
        "columns_full": columns,
    }


def parse_arm(arg: str) -> tuple[str, pathlib.Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError("arms must be label=path")
    label, path = arg.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("empty arm label")
    return label, pathlib.Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--arm", action="append", type=parse_arm, default=[], help="extra or replacement arm: label=payload.json")
    ap.add_argument("--only-default", action="store_true", help="ignore --arm additions")
    args = ap.parse_args()

    arms = OrderedDict((k, pathlib.Path(v)) for k, v in DEFAULT_ARMS.items())
    if not args.only_default:
        for label, path in args.arm:
            arms[label] = path

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ready: OrderedDict[str, pathlib.Path] = OrderedDict()
    pending: OrderedDict[str, str] = OrderedDict()
    for label, path in arms.items():
        if payload_ready(path):
            ready[label] = path
        else:
            pending[label] = rel(path)

    score_table = OrderedDict()
    for label, path in ready.items():
        sc = scores_for(path)
        score_table[label] = {"payload": rel(path), "scores": sc, "cheap7": cheap7(sc)}

    comparisons = OrderedDict()
    anchor = "chck82_anchor"
    if anchor in ready:
        for cand in ready:
            if cand != anchor:
                comp = compare_payloads(anchor, ready[anchor], cand, ready[cand])
                comparisons[comp["label"]] = comp
    # Direct contrasts that clarify whether private replay differs from ordinary continuation or controls.
    direct_pairs = [
        ("ordinary86_backbone", "coherent86_private_alpha1"),
        ("shuffled86_private", "coherent86_private_alpha1"),
        ("spanbreak86_private", "coherent86_private_alpha1"),
        ("coherent86_private_alpha1", "coherent86_private_alpha0p5"),
        ("coherent86_private_alpha1", "coherent86_private_alpha0p75"),
        ("ordinary86_backbone", "coherent86_private_alpha0p5"),
        ("ordinary86_backbone", "coherent86_private_alpha0p75"),
    ]
    for base, cand in direct_pairs:
        if base in ready and cand in ready:
            comp = compare_payloads(base, ready[base], cand, ready[cand])
            comparisons[comp["label"]] = comp

    frontier_rows = []
    if anchor in score_table:
        anchor_cheap = score_table[anchor]["cheap7"]
        for label, row in score_table.items():
            if label == anchor:
                continue
            comp = comparisons.get(f"{label}_minus_{anchor}")
            frontier_rows.append({
                "arm": label,
                "cheap7": row["cheap7"],
                "cheap7_delta_vs_anchor": None if row["cheap7"] is None or anchor_cheap is None else float(row["cheap7"] - anchor_cheap),
                "discrete_net_items_vs_anchor": None if comp is None else comp["aggregate"]["total_gain_minus_loss"],
                "changed_items_vs_anchor": None if comp is None else comp["aggregate"]["total_changed_items"],
                "retention_fraction_vs_anchor": None if comp is None else comp["aggregate"]["anchor_correct_retention_fraction"],
                "gain_to_loss_ratio_vs_anchor": None if comp is None else comp["aggregate"]["gain_to_loss_ratio"],
                "EWoK_focus_net": None if comp is None else comp["focus"].get("EWoK_relation_like", {}).get("net_gain_minus_loss"),
                "Entity_focus_net": None if comp is None else comp["focus"].get("Entity_high_operation", {}).get("net_gain_minus_loss"),
                "Supplement_focus_net": None if comp is None else comp["focus"].get("Supplement_selected", {}).get("net_gain_minus_loss"),
            })
        frontier_rows.sort(key=lambda r: (float("-inf") if r["cheap7_delta_vs_anchor"] is None else r["cheap7_delta_vs_anchor"]), reverse=True)

    reading = []
    if "coherent86_private_alpha1" in score_table and "ordinary86_backbone" in score_table:
        coh = score_table["coherent86_private_alpha1"]["cheap7"]
        ord86 = score_table["ordinary86_backbone"]["cheap7"]
        if coh is not None and ord86 is not None:
            reading.append(f"coherent86 alpha1 cheap7 exceeds ordinary86 by {coh - ord86:+.6f}; this keeps ordinary continued backbone training from explaining the endpoint edge by itself.")
    if "coherent86_private_alpha1_minus_chck82_anchor" in comparisons:
        agg = comparisons["coherent86_private_alpha1_minus_chck82_anchor"]["aggregate"]
        reading.append(f"coherent86 alpha1 changes {agg['total_changed_items']} discrete common items vs the anchor and has net {agg['total_gain_minus_loss']}; this is a redistribution surface, not uniform added competence.")
    if pending:
        reading.append(f"Pending arms are present only as paths: {list(pending)}. No score or item conclusion is made for them here.")

    out = {
        "status": "COMPLETE_WITH_PENDING_ARMS" if pending else "COMPLETE",
        "created_utc": now(),
        "ready_arms": {k: rel(v) for k, v in ready.items()},
        "pending_arms": pending,
        "score_table": score_table,
        "frontier_vs_anchor": frontier_rows,
        "comparisons": comparisons,
        "scientific_reading": reading,
    }
    out_json = out_dir / "private_scale_panel_analysis.json"
    out_md = out_dir / "private_scale_panel_analysis.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research private-scale and fast-path panel analysis",
        "",
        f"Status: **{out['status']}**",
        "",
        "This analysis uses saved official-compatible prediction payloads only; it runs no model inference.",
        "",
        "## Arms",
        "",
        "| arm | payload | cheap7 | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in score_table.items():
        sc = row["scores"] or {}
        lines.append(
            f"| {label} | `{row['payload']}` | {row['cheap7']} | {sc.get('BLiMP')} | {sc.get('Supplement')} | {sc.get('EWoK')} | {sc.get('Entity')} | {sc.get('COMPS')} | {sc.get('GlobalPIQA')} | {sc.get('Reading')} |"
        )
    if pending:
        lines += ["", "Pending payload paths:"]
        for k, v in pending.items():
            lines.append(f"- {k}: `{v}`")
    lines += [
        "",
        "## Anchor comparison surface",
        "",
        "| arm | Δcheap7 vs anchor | net discrete items | changed items | retention fraction | gain/loss | EWoK focus net | Entity focus net | Supplement focus net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in frontier_rows:
        lines.append(
            f"| {r['arm']} | {r['cheap7_delta_vs_anchor']} | {r['discrete_net_items_vs_anchor']} | {r['changed_items_vs_anchor']} | {r['retention_fraction_vs_anchor']} | {r['gain_to_loss_ratio_vs_anchor']} | {r['EWoK_focus_net']} | {r['Entity_focus_net']} | {r['Supplement_focus_net']} |"
        )
    lines += ["", "## Direct compact comparisons", ""]
    for label, comp in comparisons.items():
        agg = comp["aggregate"]
        lines += [
            f"### {label}",
            "",
            f"Aggregate: net {agg['total_gain_minus_loss']} from {agg['total_gain_items']} gains / {agg['total_loss_items']} losses over {agg['total_common_items']} common discrete items; retention fraction {agg['anchor_correct_retention_fraction']}; mean payload Δ {agg['discrete_payload_mean_delta']:+.4f}.",
            "",
            "| column | payload Δ | gains | losses | net | retention | best groups | worst groups |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for col in DISCRETE_COLUMNS:
            c = comp["columns_compact"][col]
            best = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in c["best_groups_by_item_net"][:3])
            worst = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in c["worst_groups_by_item_net"][:3])
            lines.append(f"| {col} | {c['delta_payload']} | {c['gain']} | {c['loss']} | {c['net_gain_minus_loss']} | {c['anchor_correct_retention_fraction']} | {best} | {worst} |")
        focus = comp.get("focus", {})
        if focus:
            lines += ["", "Focus nets:"]
            for fname, fobj in focus.items():
                lines.append(f"- {fname}: net {fobj.get('net_gain_minus_loss')} from gains {fobj.get('gain')} / losses {fobj.get('loss')} over n={fobj.get('total_n')}")
        lines.append("")
    lines += ["## Scientific reading", ""]
    for x in reading:
        lines.append(f"- {x}")
    lines += ["", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "ready": list(ready), "pending": list(pending), "n_comparisons": len(comparisons), "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
