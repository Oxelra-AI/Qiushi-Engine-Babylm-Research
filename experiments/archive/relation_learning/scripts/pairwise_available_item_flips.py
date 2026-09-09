#!/usr/bin/env python3
"""research: paired item flips on the discrete official columns available in each payload.

The research table assumes all six discrete columns are present. Dense official
payloads currently contain zero-shot/Reading columns but no completed SuperGLUE/AoA;
this script compares only the intersection of available discrete columns, preserving
exact item-gain/loss counts without pretending the missing columns are known.
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
}


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def payload_scores(path: pathlib.Path) -> dict[str, Any]:
    obj = read_json(path)
    raw = obj.get("official_overall", {}).get("scores", {}) if isinstance(obj, dict) else {}
    tasks = obj.get("tasks", {}) if isinstance(obj, dict) else {}
    out = {c: raw.get(c) for c in CHEAP_COLS}
    if out.get("GlobalPIQA") is None and raw.get("GlobalPIQA_mean") is not None:
        out["GlobalPIQA"] = raw.get("GlobalPIQA_mean")
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        if out.get(c) is None and isinstance(tasks.get(c), dict):
            out[c] = tasks[c].get("score")
    if out.get("GlobalPIQA") is None:
        gp = []
        for sub in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            rec = tasks.get(sub)
            if isinstance(rec, dict) and rec.get("score") is not None:
                gp.append(float(rec["score"]))
        if len(gp) == 2:
            out["GlobalPIQA"] = sum(gp) / 2.0
    if out.get("Reading") is None and isinstance(tasks.get("Reading"), dict):
        rs = tasks["Reading"].get("scores") or {}
        if isinstance(rs, dict):
            out["Reading"] = rs.get("Reading")
    return out


def available_discrete(path: pathlib.Path) -> list[str]:
    obj = read_json(path)
    tasks = obj.get("tasks", {}) if isinstance(obj, dict) else {}
    cols = []
    for c in DISCRETE_COLUMNS:
        if c == "GlobalPIQA":
            ok = all(isinstance(tasks.get(sub), dict) and bool(tasks[sub].get("predictions"))
                     for sub in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"])
            if ok:
                cols.append(c)
        else:
            rec = tasks.get(c)
            if isinstance(rec, dict) and bool(rec.get("predictions")):
                cols.append(c)
    return cols


def compare_available(base_label: str, cand_label: str, base_path: pathlib.Path, cand_path: pathlib.Path) -> dict[str, Any]:
    common_cols = [c for c in DISCRETE_COLUMNS if c in available_discrete(base_path) and c in available_discrete(cand_path)]
    base_loader = PayloadLoader(base_path)
    cand_loader = PayloadLoader(cand_path)
    cols: dict[str, Any] = {}
    for c in common_cols:
        cols[c] = compare_column(base_loader, cand_loader, c)
    total_common = int(sum(cols[c]["n_common"] for c in common_cols))
    gains = int(sum(cols[c]["flip_counts"].get("gain", 0) for c in common_cols))
    losses = int(sum(cols[c]["flip_counts"].get("loss", 0) for c in common_cols))
    payload_deltas = [float(cols[c]["delta_score_payload"]) for c in common_cols if cols[c].get("delta_score_payload") is not None]
    recon_deltas = [float(cols[c]["delta_score_reconstructed_common"]) for c in common_cols if cols[c].get("delta_score_reconstructed_common") is not None]
    column_table = []
    for c in common_cols:
        x = cols[c]
        column_table.append({
            "column": c,
            "n_common": x["n_common"],
            "payload_delta": x.get("delta_score_payload"),
            "reconstructed_delta": x.get("delta_score_reconstructed_common"),
            "gains": int(x["flip_counts"].get("gain", 0)),
            "losses": int(x["flip_counts"].get("loss", 0)),
            "net_gain_minus_loss": int(x["gain_minus_loss_items"]),
            "best_groups": x.get("best_groups_by_item_net", [])[:10],
            "worst_groups": x.get("worst_groups_by_item_net", [])[:10],
        })
    return {
        "label": f"{cand_label}_minus_{base_label}",
        "base_label": base_label,
        "candidate_label": cand_label,
        "base_payload": rel(base_path),
        "candidate_payload": rel(cand_path),
        "available_discrete_columns": common_cols,
        "missing_discrete_columns": [c for c in DISCRETE_COLUMNS if c not in common_cols],
        "columns": cols,
        "column_table": column_table,
        "aggregate": {
            "total_gain_items": gains,
            "total_loss_items": losses,
            "total_gain_minus_loss": gains - losses,
            "total_common_items": total_common,
            "available_discrete_payload_mean_delta": None if not payload_deltas else sum(payload_deltas) / len(payload_deltas),
            "available_discrete_reconstructed_mean_delta": None if not recon_deltas else sum(recon_deltas) / len(recon_deltas),
            "gain_loss_balance_pct_common": 100.0 * (gains - losses) / max(1, total_common),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--payload", action="append", default=[], help="label=path")
    ap.add_argument("--anchor", default="coherent86_s43022")
    args = ap.parse_args()

    payloads = dict(DEFAULT_PAYLOADS)
    for spec in args.payload:
        label, path = spec.split("=", 1)
        payloads[label] = pathlib.Path(path)
    payloads = {k: pathlib.Path(v) for k, v in payloads.items()}
    missing = {k: rel(v) for k, v in payloads.items() if not v.exists()}
    if missing:
        print(json.dumps({"status": "MISSING_PAYLOADS", "missing": missing}, indent=2), flush=True)
        raise SystemExit(2)
    if args.anchor not in payloads:
        raise SystemExit(f"anchor {args.anchor!r} not in payloads")
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = {k: payload_scores(v) for k, v in payloads.items()}
    comparisons = {}
    for label, path in payloads.items():
        if label == args.anchor:
            continue
        comparisons[f"{label}_minus_{args.anchor}"] = compare_available(args.anchor, label, payloads[args.anchor], path)
    out = {
        "status": "AVAILABLE_PAIRWISE_ITEM_FLIPS_DONE",
        "created_utc": now(),
        "anchor": args.anchor,
        "payloads": {k: rel(v) for k, v in payloads.items()},
        "scores": scores,
        "comparisons": comparisons,
        "note": "Only the intersection of available discrete columns is compared; missing SuperGLUE/AoA or missing per-target tasks are not inferred.",
    }
    out_json = out_dir / "available_pairwise_item_flips.json"
    out_md = out_dir / "available_pairwise_item_flips.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research available-column paired item flips", "", out["note"], ""]
    lines.append("## Scores")
    lines.append("| label | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, sc in scores.items():
        lines.append(f"| {label} | {sc.get('BLiMP')} | {sc.get('Supplement')} | {sc.get('EWoK')} | {sc.get('Entity')} | {sc.get('COMPS')} | {sc.get('GlobalPIQA')} | {sc.get('Reading')} |")
    for comp in comparisons.values():
        agg = comp["aggregate"]
        lines += ["", f"## {comp['label']}", f"Available discrete columns: `{comp['available_discrete_columns']}`; missing: `{comp['missing_discrete_columns']}`.", f"Aggregate over available discrete items: gains `{agg['total_gain_items']}`, losses `{agg['total_loss_items']}`, net `{agg['total_gain_minus_loss']}` over `{agg['total_common_items']}` items; available-column payload mean Δ `{agg['available_discrete_payload_mean_delta']}`.", "", "| column | payload Δ | reconstructed Δ | common | gains | losses | net | best groups | worst groups |", "|---|---:|---:|---:|---:|---:|---:|---|---|"]
        for r in comp["column_table"]:
            best = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in r["best_groups"][:4])
            worst = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in r["worst_groups"][:4])
            pdel = "NA" if r.get("payload_delta") is None else f"{float(r['payload_delta']):+.3f}"
            rdel = "NA" if r.get("reconstructed_delta") is None else f"{float(r['reconstructed_delta']):+.3f}"
            lines.append(f"| {r['column']} | {pdel} | {rdel} | {r['n_common']} | {r['gains']} | {r['losses']} | {r['net_gain_minus_loss']:+d} | {best} | {worst} |")
    lines.append(f"\nJSON: `{rel(out_json)}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(out_json), "out_md": rel(out_md), "aggregates": {k: v["aggregate"] for k, v in comparisons.items()}}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
