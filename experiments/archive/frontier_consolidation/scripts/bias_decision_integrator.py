#!/usr/bin/env python3
"""research: integrate cheap changed-state-bias evidence.

Reads the research file-only Entity numops split, sampled margin split, and the
late counterbalanced binding probes. Produces one compact research-facing record
for route decisions before spending an H100 on the permuted companion arm.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import time
from collections import defaultdict
from typing import Any

ROOT = _public_path('.')
WS = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/bias_decision_integrator')
ENTITY_DELTAS = _public_path('experiments/archive/frontier_consolidation/data/entity_numops_bias_readout/entity_numops_view_minus_repeat_deltas.csv')
MARGIN_KEY = _public_path('experiments/archive/frontier_consolidation/data/entity_margin_numops_readout/entity_margin_numops_key_table.csv')
BIND80 = _public_path('experiments/archive/frontier_consolidation/data/max_binding_change_bias_probe/binding_view_minus_repeat_deltas.csv')
BIND90100 = _public_path('experiments/archive/frontier_consolidation/data/max_binding_change_bias_probe_late90_100/binding_view_minus_repeat_deltas.csv')
NOTE = _public_path('research/notes/frontier_consolidation/changed_state_bias_threat_and_route_decision.md')


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    ent = read_csv(ENTITY_DELTAS)
    ent_key = [r for r in ent if r.get("basin") == "seed43022_first_basin" and r.get("checkpoint") in {"chck_80M", "chck_90M"} and r.get("group") in {"all_18_subtasks", "zero_ops", "nonzero_ops"}]
    ent_by_group: dict[str, list[float]] = defaultdict(list)
    for r in ent_key:
        v = f(r.get("macro_delta_pp_view_minus_repeat"))
        if v is not None:
            ent_by_group[r["group"]].append(v)
    first_basin_late_entity = {k: mean(vs) for k, vs in sorted(ent_by_group.items())}
    if first_basin_late_entity.get("zero_ops") is not None and first_basin_late_entity.get("nonzero_ops") is not None:
        first_basin_late_entity["nonzero_minus_zero"] = first_basin_late_entity["nonzero_ops"] - first_basin_late_entity["zero_ops"]

    margin_rows = read_csv(MARGIN_KEY)
    max_margin = next((r for r in margin_rows if r.get("dose_name") == "dose2p64" and r.get("checkpoint") == "chck_80M"), {})
    margin_summary = {
        "max_all_mean": f(max_margin.get("all_margin_delta_mean")),
        "max_zero_mean": f(max_margin.get("zero_margin_delta_mean")),
        "max_nonzero_mean": f(max_margin.get("nonzero_margin_delta_mean")),
        "max_nonzero_minus_zero": f(max_margin.get("nonzero_minus_zero_margin_delta")),
    }

    bind_rows: list[dict[str, str]] = []
    for path in [BIND80, BIND90100]:
        for r in read_csv(path):
            if r.get("status") == "ok":
                bind_rows.append(r)
    bind_by_basin: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in bind_rows:
        bind_by_basin[r["basin"]].append(r)
    binding_summary: dict[str, dict[str, Any]] = {}
    for basin, rs in sorted(bind_by_basin.items()):
        binding_summary[basin] = {
            "checkpoints": [r["checkpoint"] for r in rs],
            "affected_delta_pp_mean": mean([100.0 * f(r["affected_delta_view_minus_repeat"]) for r in rs if f(r.get("affected_delta_view_minus_repeat")) is not None]),
            "unaffected_delta_pp_mean": mean([100.0 * f(r["unaffected_delta_view_minus_repeat"]) for r in rs if f(r.get("unaffected_delta_view_minus_repeat")) is not None]),
            "EEBF_delta_pp_mean": mean([100.0 * f(r["EEBF_delta_view_minus_repeat"]) for r in rs if f(r.get("EEBF_delta_view_minus_repeat")) is not None]),
            "signed_margin_shift_mean": mean([f(r["signed_bias_margin_delta"]) for r in rs if f(r.get("signed_bias_margin_delta")) is not None]),
            "bias_like_count": sum(1 for r in rs if str(r.get("change_bias_signature")).lower() == "true"),
            "pair_count": len(rs),
        }

    interpretation = {
        "core_update": "The first-basin MAX Entity carrier is no longer safe to read as source-view record addressability. Official Entity and sampled margins both split sharply by operation count: nonzero-operation rows improve while zero-operation rows deteriorate.",
        "cross_basin_binding": "The counterbalanced binding substrate is basin-dependent: seed43022 has a strong affected-gain/unaffected-loss pattern, while seed43122 late checkpoints improve EEBF mostly through unaffected preservation and do not show the same changed-state signature.",
        "permuted_arm_status": "Keep the research permuted companion arm as a prepared mechanism instrument, but do not launch it until pending second-basin official Entity numops and breadth V-B/B-R splits show an aligned carrier not mostly explained by nonzero-op gain plus zero-op loss.",
        "breadth_status": "Breadth training completed exactly. Late EWoK+Entity CPU scoring remains pending and is required before a new training decision.",
        "worker_timeout_status": "s267_t32 and s267_t33 timed out during full CPU Entity scoring; their partial Supplement/EWoK rows are usable only as completed column data, not complete stable-family or Entity evidence.",
    }
    payload = {
        "status": "BIAS_DECISION_INTEGRATED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "entity_numops": rel(ENTITY_DELTAS),
            "entity_margin_numops": rel(MARGIN_KEY),
            "binding_chck80": rel(BIND80),
            "binding_chck90_100": rel(BIND90100),
            "route_note": rel(NOTE),
        },
        "first_basin_late_entity_80_90_official_delta_pp": first_basin_late_entity,
        "sampled_margin_max_80M": margin_summary,
        "binding_summary": binding_summary,
        "interpretation": interpretation,
        "no_training_model_inference_gpu_upload_or_leaderboard": True,
    }
    (_public_path('experiments/archive/frontier_consolidation/data/bias_decision_integrator/bias_decision_integrated_summary.json')).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research changed-state bias decision integration\n\n"]
    lines.append("## First-basin official Entity numops split, late 80/90M\n\n")
    lines.append(f"- all_18_subtasks Δ pp: {first_basin_late_entity.get('all_18_subtasks')}\n")
    lines.append(f"- zero_ops Δ pp: {first_basin_late_entity.get('zero_ops')}\n")
    lines.append(f"- nonzero_ops Δ pp: {first_basin_late_entity.get('nonzero_ops')}\n")
    lines.append(f"- nonzero_minus_zero Δ pp: {first_basin_late_entity.get('nonzero_minus_zero')}\n\n")
    lines.append("## Sampled margin MAX 80M\n\n")
    for k, v in margin_summary.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## Binding late summary\n\n")
    for basin, s in binding_summary.items():
        lines.append(f"- {basin}: affected Δ pp mean {s['affected_delta_pp_mean']}, unaffected Δ pp mean {s['unaffected_delta_pp_mean']}, EEBF Δ pp mean {s['EEBF_delta_pp_mean']}, bias-like {s['bias_like_count']}/{s['pair_count']}.\n")
    lines.append("\n## Route consequence\n\n")
    for k, v in interpretation.items():
        lines.append(f"- **{k}**: {v}\n")
    (_public_path('experiments/archive/frontier_consolidation/data/bias_decision_integrator/bias_decision_integrated_summary.md')).write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": rel(OUT), "first_basin_late_entity": first_basin_late_entity, "binding_summary": binding_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
