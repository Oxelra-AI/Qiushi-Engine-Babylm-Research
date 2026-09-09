#!/usr/bin/env python3
"""Saved-prediction item-family analysis for research fast-path replay arms.

Compares coherent_replay, spanbreak_replay, and the existing shuffled86 null to the
protected chck_82M anchor using the validated research item parser.  The main output
is not just aggregate score: it records retained anchor-correct items, lost anchor-
correct items, and gained new correct items, with focused readouts for EWoK relation
families and high-operation Entity families.
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
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import (  # noqa: E402
    DISCRETE_COLUMNS,
    PayloadLoader,
    compare_column,
)

DEFAULT_CHCK82 = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json')
DEFAULT_SHUFFLED86 = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json')
DEFAULT_COHERENT = _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json')
DEFAULT_SPANBREAK = _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_eval/per_target/fastpath4M_spanbreak.json')
DEFAULT_OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/fastpath_item_family_analysis')
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
FRAGILE_EWOK_TOKENS = [
    "material", "spatial", "space", "quantity", "quantitative", "number", "physical", "social",
    "support", "contact", "contained", "size", "mass", "volume", "distance", "dynamics", "interaction",
]
FRAGILE_ENTITY_TOKENS = ["regular_5", "move_contents_4", "move_contents_3", "ambiref_3", "5_ops", "4_ops", "3_ops"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def payload_has_discrete_tasks(path: pathlib.Path) -> bool:
    data = read_json(path)
    if not isinstance(data, dict):
        return False
    tasks = data.get("tasks", {})
    required = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
    return all(c in tasks for c in required)


def payload_scores(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    raw = read_json(path)
    scores = raw.get("official_overall", {}).get("scores", {}) if isinstance(raw, dict) else {}
    return {c: scores.get(c) for c in CHEAP_COLS}


def cheap7(scores: dict[str, Any] | None) -> float | None:
    if not scores:
        return None
    vals = [scores.get(c) for c in CHEAP_COLS]
    if any(v is None for v in vals):
        return None
    return sum(float(v) for v in vals) / len(vals)


def group_net_sum(groups: list[dict[str, Any]], tokens: list[str]) -> dict[str, Any]:
    tagged = []
    net = 0
    n = 0
    gains = 0
    losses = 0
    for g in groups:
        name = str(g.get("group", "")).lower()
        if any(tok in name for tok in tokens):
            tagged.append(g)
            net += int(g.get("net_gain_minus_loss", 0))
            n += int(g.get("n", 0))
            gains += int(g.get("gain", g.get("gains", 0)) or 0)
            losses += int(g.get("loss", g.get("losses", 0)) or 0)
    return {"tagged_groups": tagged, "net_gain_minus_loss": net, "n": n, "gains_reported": gains, "losses_reported": losses}


def fragile_readouts(cols: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ew = cols.get("EWoK")
    if ew:
        groups = ew.get("groups_all_by_item_net", [])
        out["EWoK_fragile_tagged"] = group_net_sum(groups, FRAGILE_EWOK_TOKENS)
        out["EWoK_worst_12"] = ew.get("worst_groups_by_item_net", [])[:12]
        out["EWoK_best_12"] = ew.get("best_groups_by_item_net", [])[:12]
    ent = cols.get("Entity")
    if ent:
        groups = ent.get("groups_all_by_item_net", [])
        out["Entity_highop_tagged"] = group_net_sum(groups, FRAGILE_ENTITY_TOKENS)
        out["Entity_worst_12"] = ent.get("worst_groups_by_item_net", [])[:12]
        out["Entity_best_12"] = ent.get("best_groups_by_item_net", [])[:12]
    supp = cols.get("Supplement")
    if supp:
        out["Supplement_worst_12"] = supp.get("worst_groups_by_item_net", [])[:12]
        out["Supplement_best_12"] = supp.get("best_groups_by_item_net", [])[:12]
    return out


def compare_payloads(label: str, base_payload: pathlib.Path, cand_payload: pathlib.Path) -> dict[str, Any]:
    base_loader = PayloadLoader(base_payload)
    cand_loader = PayloadLoader(cand_payload)
    cols: dict[str, Any] = {}
    for col in DISCRETE_COLUMNS:
        cols[col] = compare_column(base_loader, cand_loader, col)
    aggregate = {
        "total_gain_items": int(sum(cols[c]["flip_counts"].get("gain", 0) for c in DISCRETE_COLUMNS)),
        "total_loss_items": int(sum(cols[c]["flip_counts"].get("loss", 0) for c in DISCRETE_COLUMNS)),
        "total_common_items": int(sum(cols[c]["n_common"] for c in DISCRETE_COLUMNS)),
        "discrete_payload_mean_delta": sum(float(cols[c]["delta_score_payload"]) for c in DISCRETE_COLUMNS) / len(DISCRETE_COLUMNS),
        "discrete_reconstructed_mean_delta": sum(float(cols[c]["delta_score_reconstructed_common"]) for c in DISCRETE_COLUMNS) / len(DISCRETE_COLUMNS),
    }
    aggregate["total_gain_minus_loss"] = aggregate["total_gain_items"] - aggregate["total_loss_items"]
    aggregate["loss_to_gain_ratio"] = aggregate["total_loss_items"] / max(1, aggregate["total_gain_items"])
    return {
        "label": label,
        "base_payload": rel(base_payload),
        "candidate_payload": rel(cand_payload),
        "columns": cols,
        "aggregate": aggregate,
        "fragile_readouts": fragile_readouts(cols),
    }


def compact_column_table(comp: dict[str, Any]) -> list[str]:
    lines = ["| column | payload Δ | reconstructed Δ | gains | losses | net | worst groups |", "|---|---:|---:|---:|---:|---:|---|"]
    for col in DISCRETE_COLUMNS:
        c = comp["columns"][col]
        worst = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in c.get("worst_groups_by_item_net", [])[:4])
        lines.append(f"| {col} | {float(c['delta_score_payload']):+.3f} | {float(c['delta_score_reconstructed_common']):+.3f} | {c['flip_counts'].get('gain',0)} | {c['flip_counts'].get('loss',0)} | {c['gain_minus_loss_items']:+d} | {worst} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chck82-payload", default=str(DEFAULT_CHCK82))
    ap.add_argument("--coherent-payload", default=str(DEFAULT_COHERENT))
    ap.add_argument("--spanbreak-payload", default=str(DEFAULT_SPANBREAK))
    ap.add_argument("--shuffled86-payload", default=str(DEFAULT_SHUFFLED86))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "fastpath_item_family_analysis.json"
    out_md = out_dir / "fastpath_item_family_analysis.md"

    payloads = {
        "chck82": pathlib.Path(args.chck82_payload),
        "coherent": pathlib.Path(args.coherent_payload),
        "spanbreak": pathlib.Path(args.spanbreak_payload),
        "shuffled86": pathlib.Path(args.shuffled86_payload),
    }
    payloads = {k: (v if v.exists() and payload_has_discrete_tasks(v) else None) for k, v in payloads.items()}
    scores = {name: payload_scores(path) for name, path in payloads.items()}
    cheap = {name: cheap7(sc) for name, sc in scores.items()}
    available = [k for k, v in payloads.items() if v is not None]
    pending = [k for k, v in payloads.items() if v is None]
    comparisons: dict[str, Any] = {}
    if payloads.get("chck82"):
        for cand in ["coherent", "spanbreak", "shuffled86"]:
            if payloads.get(cand):
                comparisons[f"{cand}_minus_chck82"] = compare_payloads(f"{cand}_minus_chck82", payloads["chck82"], payloads[cand])
    if payloads.get("spanbreak") and payloads.get("coherent"):
        comparisons["coherent_minus_spanbreak"] = compare_payloads("coherent_minus_spanbreak", payloads["spanbreak"], payloads["coherent"])
    if payloads.get("shuffled86") and payloads.get("coherent"):
        comparisons["coherent_minus_shuffled86"] = compare_payloads("coherent_minus_shuffled86", payloads["shuffled86"], payloads["coherent"])
    if payloads.get("shuffled86") and payloads.get("spanbreak"):
        comparisons["spanbreak_minus_shuffled86"] = compare_payloads("spanbreak_minus_shuffled86", payloads["shuffled86"], payloads["spanbreak"])

    route_read = "pending_step150_payloads"
    if not any(k in pending for k in ["coherent", "spanbreak"]) and "chck82" not in pending:
        comp = comparisons.get("coherent_minus_chck82", {})
        span = comparisons.get("spanbreak_minus_chck82", {})
        coh_fr = comp.get("fragile_readouts", {})
        sp_fr = span.get("fragile_readouts", {})
        coh_losses = comp.get("aggregate", {}).get("total_loss_items")
        sp_losses = span.get("aggregate", {}).get("total_loss_items")
        coh_delta = cheap.get("coherent") - cheap.get("chck82") if cheap.get("coherent") is not None and cheap.get("chck82") is not None else None
        sp_delta = cheap.get("spanbreak") - cheap.get("chck82") if cheap.get("spanbreak") is not None and cheap.get("chck82") is not None else None
        route_read = "needs_scientific_interpretation"
        if coh_delta is not None and sp_delta is not None:
            if coh_delta <= sp_delta + 0.05:
                route_read = "coherent_not_distinct_from_spanbreak_on_cheap7"
            if coh_losses is not None and sp_losses is not None and coh_losses > sp_losses:
                route_read += "_and_coherent_loses_more_anchor_items"

    out = {
        "status": "PENDING" if pending else "COMPLETE",
        "created_utc": now(),
        "payloads": {k: None if v is None else rel(v) for k, v in payloads.items()},
        "available": available,
        "pending": pending,
        "scores": scores,
        "cheap7": cheap,
        "comparisons": comparisons,
        "route_read": route_read,
        "scientific_reading": "Success requires private-ON coherent replay to retain anchor-correct fragile relation/state decisions while adding reproducible new decisions. Private-OFF equality and aggregate gains alone are not sufficient.",
    }
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research fast-path item-family analysis", "", f"Status: **{out['status']}**", f"Route read: `{route_read}`", "", f"Available: `{available}`; pending: `{pending}`", "", "## Cheap scores", "", "| arm | payload | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["chck82", "shuffled86", "coherent", "spanbreak"]:
        sc = scores.get(name) or {}
        lines.append(f"| {name} | `{out['payloads'].get(name)}` | {cheap.get(name)} | {sc.get('BLiMP')} | {sc.get('Supplement')} | {sc.get('EWoK')} | {sc.get('Entity')} | {sc.get('COMPS')} | {sc.get('GlobalPIQA')} | {sc.get('Reading')} |")
    for cname, comp in comparisons.items():
        lines += ["", f"## {cname}", "", f"Aggregate: `{comp['aggregate']}`", ""]
        lines += compact_column_table(comp)
        fr = comp.get("fragile_readouts", {})
        if fr:
            lines += ["", "### Fragile-family readouts", "", "```json", json.dumps(fr, indent=2)[:10000], "```"]
    lines += ["", out["scientific_reading"], "", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "route_read": route_read, "available": available, "pending": pending, "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
