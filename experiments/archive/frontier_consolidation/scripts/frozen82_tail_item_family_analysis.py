#!/usr/bin/env python3
"""No-inference item-family analysis for the frozen-82M tail branches.

This script consumes saved official-compatible prediction payloads only.  It compares
aligned and shuffled 4M private-tail branches against the protected chck_82M payload
and against each other, reusing the validated research parser.  It is safe to run
before pending evaluations finish; it records which payloads are present.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any

ROOT = pathlib.Path(".").resolve()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import (  # noqa: E402
    DISCRETE_COLUMNS,
    PayloadLoader,
    compare_column,
)

OUT_DIR = WORKSPACE / "data/frozen82_tail_item_family_analysis"
OUT_JSON = OUT_DIR / "frozen82_tail_item_family_analysis.json"
OUT_MD = (ROOT / 'research/documents/frontier_consolidation/data/frozen82_tail_item_family_analysis/frozen82_tail_item_family_analysis.md')

CHCK82_PAYLOAD = ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json"
SUMMARY_PATHS = {
    "aligned": WORKSPACE / "data/frozen82_tail4M_aligned_summary/frozen82_tail4M_aligned_summary.json",
    "shuffled": WORKSPACE / "data/frozen82_tail4M_shuffled_summary/frozen82_tail4M_shuffled_summary.json",
}
FALLBACK_PAYLOADS = {
    "aligned": WORKSPACE / "data/frozen82_tail4M_aligned_eval/per_target/frozen82_tail4M_aligned.json",
    "shuffled": WORKSPACE / "data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json",
}
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
FRAGILE_EWOK_TOKENS = [
    "material", "spatial", "space", "quantity", "quantitative", "number", "physical", "social",
    "support", "contact", "contained", "size", "mass", "volume", "distance",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def payload_has_discrete_tasks(path: pathlib.Path) -> bool:
    data = load_json(path)
    if not isinstance(data, dict):
        return False
    tasks = data.get("tasks", {})
    required = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
    return all(c in tasks for c in required)


def payload_for(name: str) -> pathlib.Path | None:
    if name == "chck82":
        return CHCK82_PAYLOAD if CHCK82_PAYLOAD.exists() and payload_has_discrete_tasks(CHCK82_PAYLOAD) else None
    # A running evaluator can create a partial per_target JSON before all tasks are
    # populated.  The summary file is written only after frozen82_tail_eval_one
    # has successfully completed and extracted all cheap7 scores, so prefer it as the
    # readiness signal.  Fallback is accepted only if the payload itself already has all
    # discrete columns.
    summ = load_json(SUMMARY_PATHS[name])
    if summ and summ.get("payload_path"):
        p = ROOT / str(summ["payload_path"])
        if p.exists() and payload_has_discrete_tasks(p):
            return p
    p = FALLBACK_PAYLOADS[name]
    return p if p.exists() and payload_has_discrete_tasks(p) else None


def payload_scores(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    raw = load_json(path)
    scores = raw.get("official_overall", {}).get("scores", {}) if isinstance(raw, dict) else {}
    return {c: scores.get(c) for c in CHEAP_COLS}


def cheap7(scores: dict[str, Any] | None) -> float | None:
    if not scores:
        return None
    vals = [scores.get(c) for c in CHEAP_COLS]
    if any(v is None for v in vals):
        return None
    return sum(float(v) for v in vals) / len(vals)


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
    aggregate["total_gain_minus_loss_pct"] = 100.0 * aggregate["total_gain_minus_loss"] / max(1, aggregate["total_common_items"])
    return {
        "label": label,
        "base_payload": rel(base_payload),
        "candidate_payload": rel(cand_payload),
        "columns": cols,
        "aggregate": aggregate,
        "fragile_readouts": fragile_readouts(cols),
    }


def fragile_readouts(cols: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ew = cols.get("EWoK")
    if ew:
        groups = ew.get("groups_all_by_item_net", [])
        tagged = []
        for g in groups:
            name = str(g.get("group", "")).lower()
            if any(tok in name for tok in FRAGILE_EWOK_TOKENS):
                tagged.append(g)
        out["EWoK_tagged_fragile_groups"] = tagged
        out["EWoK_worst_10"] = ew.get("worst_groups_by_item_net", [])[:10]
        out["EWoK_best_10"] = ew.get("best_groups_by_item_net", [])[:10]
    ent = cols.get("Entity")
    if ent:
        out["Entity_worst_10"] = ent.get("worst_groups_by_item_net", [])[:10]
        out["Entity_best_10"] = ent.get("best_groups_by_item_net", [])[:10]
    supp = cols.get("Supplement")
    if supp:
        out["Supplement_worst_10"] = supp.get("worst_groups_by_item_net", [])[:10]
        out["Supplement_best_10"] = supp.get("best_groups_by_item_net", [])[:10]
    return out


def compact_column_table(comp: dict[str, Any]) -> list[str]:
    lines = ["| column | payload Δ | reconstructed Δ | gains | losses | net | worst groups |", "|---|---:|---:|---:|---:|---:|---|"]
    for col in DISCRETE_COLUMNS:
        c = comp["columns"][col]
        worst = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in c.get("worst_groups_by_item_net", [])[:4])
        lines.append(f"| {col} | {float(c['delta_score_payload']):+.3f} | {float(c['delta_score_reconstructed_common']):+.3f} | {c['flip_counts'].get('gain',0)} | {c['flip_counts'].get('loss',0)} | {c['gain_minus_loss_items']:+d} | {worst} |")
    return lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {name: payload_for(name) for name in ["chck82", "aligned", "shuffled"]}
    scores = {name: payload_scores(path) for name, path in payloads.items()}
    available = [name for name, path in payloads.items() if path is not None]
    pending = [name for name, path in payloads.items() if path is None]
    comparisons: dict[str, Any] = {}
    if payloads.get("chck82") and payloads.get("aligned"):
        comparisons["aligned_minus_chck82"] = compare_payloads("aligned_minus_chck82", payloads["chck82"], payloads["aligned"])
    if payloads.get("chck82") and payloads.get("shuffled"):
        comparisons["shuffled_minus_chck82"] = compare_payloads("shuffled_minus_chck82", payloads["chck82"], payloads["shuffled"])
    if payloads.get("shuffled") and payloads.get("aligned"):
        comparisons["aligned_minus_shuffled"] = compare_payloads("aligned_minus_shuffled", payloads["shuffled"], payloads["aligned"])

    cheap = {name: cheap7(sc) for name, sc in scores.items()}
    route_read = "pending_tail_item_payloads"
    if not pending:
        a = cheap.get("aligned")
        s = cheap.get("shuffled")
        r = cheap.get("chck82")
        if a is not None and s is not None and r is not None:
            if a > r and a > s:
                route_read = "aligned_tail_improves_or_preserves_score_surface"
            elif a > s and a <= r:
                route_read = "aligned_tail_uses_correspondence_but_does_not_exceed_protected_surface"
            else:
                route_read = "tail_branch_not_supported_by_item_surface"

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
        "scientific_reading": "This is a saved-prediction item-family analysis. It is intended to show whether a private tail changes broad families by adding competence to the protected chck_82M function or by redistributing within fragile Supplement/EWoK/Entity/COMPS/GlobalPIQA families.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research frozen-82M tail item-family analysis", "", f"Status: **{out['status']}**", f"Route read: `{route_read}`", "", f"Available: `{available}`; pending: `{pending}`", "", "## Payload cheap scores", "", "| arm | payload | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["chck82", "aligned", "shuffled"]:
        sc = scores.get(name) or {}
        lines.append(f"| {name} | `{out['payloads'].get(name)}` | {cheap.get(name)} | {sc.get('BLiMP')} | {sc.get('Supplement')} | {sc.get('EWoK')} | {sc.get('Entity')} | {sc.get('COMPS')} | {sc.get('GlobalPIQA')} | {sc.get('Reading')} |")
    for cname, comp in comparisons.items():
        lines += ["", f"## {cname}", "", f"Aggregate: `{comp['aggregate']}`", ""]
        lines += compact_column_table(comp)
        fr = comp.get("fragile_readouts", {})
        if fr:
            lines += ["", "### Fragile-family readouts", "", "```json", json.dumps(fr, indent=2)[:8000], "```"]
    lines += ["", out["scientific_reading"], "", f"JSON: `{rel(OUT_JSON)}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "route_read": route_read, "available": available, "pending": pending, "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
