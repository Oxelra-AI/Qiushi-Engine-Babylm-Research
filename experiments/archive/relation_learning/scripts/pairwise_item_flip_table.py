#!/usr/bin/env python3
"""research: paired item-level flip table for private-phase BabyLM endpoints.

This script adapts the validated frontier_consolidation research item parser to the current
Stage-III reading problem.  Scalar cheap7 movements around coherent86 have proven
seed-sensitive; the needed object is a common-item flip table showing which official
items are retained, lost, and gained by each private-phase endpoint relative to the
shared chck_82M trunk.
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
from typing import Any

ROOT = _public_path('.')
REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS))

from pairwise_item_flip_analysis import (  # noqa: E402
    CHEAP_COLS,
    DISCRETE_COLUMNS,
    PayloadLoader,
    compare_column,
)

DEFAULT_PAYLOADS = {
    "chck82": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json'),
    "coherent86_s43022": _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json'),
    "coherent_s43122": _public_path('experiments/archive/relation_learning/data/coherent_seed43122_eval/eval/per_target/coherent_seed43122_alpha0p75.json'),
}
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/pairwise_item_flips')


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scores_from_payload(path: pathlib.Path) -> dict[str, Any]:
    obj = load_json(path)
    raw = obj.get("official_overall", {}).get("scores", {}) if isinstance(obj, dict) else {}
    out = {c: raw.get(c) for c in CHEAP_COLS}
    # Some fast-eval summaries call the combined GlobalPIQA column GlobalPIQA_mean;
    # the full-eval per_target payloads use GlobalPIQA.
    if out.get("GlobalPIQA") is None and raw.get("GlobalPIQA_mean") is not None:
        out["GlobalPIQA"] = raw.get("GlobalPIQA_mean")
    return out


def cheap7(scores: dict[str, Any]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLS]
    if any(v is None for v in vals):
        return None
    return sum(float(v) for v in vals) / len(vals)


def compare_payloads(base_label: str, cand_label: str, base_path: pathlib.Path, cand_path: pathlib.Path) -> dict[str, Any]:
    base_loader = PayloadLoader(base_path)
    cand_loader = PayloadLoader(cand_path)
    cols: dict[str, Any] = {}
    for col in DISCRETE_COLUMNS:
        cols[col] = compare_column(base_loader, cand_loader, col)
    aggregate = {
        "total_gain_items": int(sum(cols[c]["flip_counts"].get("gain", 0) for c in DISCRETE_COLUMNS)),
        "total_loss_items": int(sum(cols[c]["flip_counts"].get("loss", 0) for c in DISCRETE_COLUMNS)),
        "total_common_items": int(sum(cols[c]["n_common"] for c in DISCRETE_COLUMNS)),
        "discrete_payload_mean_delta": float(sum(float(cols[c]["delta_score_payload"]) for c in DISCRETE_COLUMNS) / len(DISCRETE_COLUMNS)),
        "discrete_reconstructed_mean_delta": float(sum(float(cols[c]["delta_score_reconstructed_common"]) for c in DISCRETE_COLUMNS) / len(DISCRETE_COLUMNS)),
    }
    aggregate["total_gain_minus_loss"] = aggregate["total_gain_items"] - aggregate["total_loss_items"]
    aggregate["gain_loss_balance_pct_common"] = 100.0 * aggregate["total_gain_minus_loss"] / max(1, aggregate["total_common_items"])
    column_table = []
    for col in DISCRETE_COLUMNS:
        c = cols[col]
        column_table.append({
            "column": col,
            "n_common": c["n_common"],
            "payload_delta": c["delta_score_payload"],
            "reconstructed_delta": c["delta_score_reconstructed_common"],
            "gains": int(c["flip_counts"].get("gain", 0)),
            "losses": int(c["flip_counts"].get("loss", 0)),
            "both_correct": int(c["flip_counts"].get("both_correct", 0)),
            "both_wrong": int(c["flip_counts"].get("both_wrong", 0)),
            "net_gain_minus_loss": int(c["gain_minus_loss_items"]),
            "worst_groups": c.get("worst_groups_by_item_net", [])[:10],
            "best_groups": c.get("best_groups_by_item_net", [])[:10],
            "examples": c.get("examples", {}),
        })
    return {
        "label": f"{cand_label}_minus_{base_label}",
        "base_label": base_label,
        "candidate_label": cand_label,
        "base_payload": rel(base_path),
        "candidate_payload": rel(cand_path),
        "columns": cols,
        "column_table": column_table,
        "aggregate": aggregate,
    }


def md_table_for(comp: dict[str, Any]) -> list[str]:
    lines = [
        f"## {comp['label']}",
        "",
        f"Aggregate: gains {comp['aggregate']['total_gain_items']}, losses {comp['aggregate']['total_loss_items']}, net {comp['aggregate']['total_gain_minus_loss']} over {comp['aggregate']['total_common_items']} common discrete item rows; discrete payload mean Δ {comp['aggregate']['discrete_payload_mean_delta']:+.3f}.",
        "",
        "| column | payload Δ | reconstructed Δ | common | gains | losses | net | best groups | worst groups |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in comp["column_table"]:
        best = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in r["best_groups"][:4])
        worst = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in r["worst_groups"][:4])
        lines.append(f"| {r['column']} | {float(r['payload_delta']):+.3f} | {float(r['reconstructed_delta']):+.3f} | {r['n_common']} | {r['gains']} | {r['losses']} | {r['net_gain_minus_loss']:+d} | {best} | {worst} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--payload", action="append", default=[], help="label=path; default loads chck82/coherent86_s43022/coherent_s43122")
    ap.add_argument("--anchor", default="chck82")
    args = ap.parse_args()

    payloads = {k: v for k, v in DEFAULT_PAYLOADS.items()}
    for spec in args.payload:
        if "=" not in spec:
            raise SystemExit(f"--payload must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        payloads[label] = pathlib.Path(path)
    payloads = {k: pathlib.Path(v) for k, v in payloads.items()}
    missing = {k: rel(v) for k, v in payloads.items() if not v.exists()}
    if missing:
        print(json.dumps({"status": "MISSING_PAYLOADS", "missing": missing}, indent=2), flush=True)
        raise SystemExit(2)
    if args.anchor not in payloads:
        raise SystemExit(f"anchor {args.anchor!r} not in payload labels {sorted(payloads)}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = {k: scores_from_payload(v) for k, v in payloads.items()}
    cheap = {k: cheap7(scores[k]) for k in payloads}
    comparisons: dict[str, Any] = {}
    anchor_path = payloads[args.anchor]
    for label, path in payloads.items():
        if label == args.anchor:
            continue
        comparisons[f"{label}_minus_{args.anchor}"] = compare_payloads(args.anchor, label, anchor_path, path)
    # Add direct seed contrast if both coherent seeds are present; this reads whether
    # the favorable seed is a stable item set or a different draw.
    if "coherent86_s43022" in payloads and "coherent_s43122" in payloads:
        comparisons["coherent86_s43022_minus_coherent_s43122"] = compare_payloads(
            "coherent_s43122", "coherent86_s43022", payloads["coherent_s43122"], payloads["coherent86_s43022"])

    deltas_vs_anchor = {}
    for label in payloads:
        if label == args.anchor:
            continue
        deltas_vs_anchor[label] = {
            c: None if scores[label].get(c) is None or scores[args.anchor].get(c) is None else float(scores[label][c]) - float(scores[args.anchor][c])
            for c in CHEAP_COLS
        }
        if cheap[label] is not None and cheap[args.anchor] is not None:
            deltas_vs_anchor[label]["cheap7"] = cheap[label] - cheap[args.anchor]

    out = {
        "status": "PAIRWISE_ITEM_FLIPS_DONE",
        "created_utc": now(),
        "scientific_question": "Do private-phase endpoints that move scalar cheap7 retain and gain the same official items relative to the shared chck_82M trunk, or are the changes seed-specific exchanges?",
        "anchor": args.anchor,
        "payloads": {k: rel(v) for k, v in payloads.items()},
        "scores": scores,
        "cheap7": cheap,
        "deltas_vs_anchor": deltas_vs_anchor,
        "comparisons": comparisons,
    }
    out_json = out_dir / "pairwise_item_flips.json"
    out_md = out_dir / "pairwise_item_flips.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research paired item-level flips",
        "",
        out["scientific_question"],
        "",
        "## Payload scores",
        "",
        "| label | cheap7 | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in payloads:
        sc = scores[label]
        lines.append(f"| {label} | {cheap[label]:.4f} | {sc.get('BLiMP')} | {sc.get('Supplement')} | {sc.get('EWoK')} | {sc.get('Entity')} | {sc.get('COMPS')} | {sc.get('GlobalPIQA')} | {sc.get('Reading')} |")
    lines += ["", "## Deltas versus anchor", "", "```json", json.dumps(deltas_vs_anchor, indent=2, ensure_ascii=False), "```", ""]
    for comp in comparisons.values():
        lines += md_table_for(comp) + [""]
    lines += [f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "cheap7": cheap,
        "deltas_vs_anchor": deltas_vs_anchor,
        "comparison_aggregates": {k: v["aggregate"] for k, v in comparisons.items()},
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
