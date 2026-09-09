#!/usr/bin/env python3
"""Paired item/family analysis for frozen-82M fast-path replay.

Consumes saved official-compatible per-target payloads only. It is pending-safe:
if research coherent/spanbreak cheap7 evaluations have not finished, it records which
payloads are missing. Once both are present, it compares:
  - coherent4M vs spanbreak4M: coherent-context structure effect;
  - coherent4M/spanbreak4M/shuffled86 vs protected chck82: retention and acquisition;
  - coherent4M vs shuffled86: whether coherent replay beats generic private-tail redistribution.

This script performs no model inference or training.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from statistics import mean
from typing import Any

ROOT = pathlib.Path(".").resolve()
A01 = ROOT / "experiments/archive/representation_and_objectives"
A02 = ROOT / "experiments/archive/frontier_consolidation"
A02_SCRIPTS = A02 / "scripts"
if str(A02_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A02_SCRIPTS))

from pairwise_item_flip_analysis import DISCRETE_COLUMNS, PayloadLoader, compare_column  # noqa: E402

OUT_DIR = A01 / "data/fastpath_pair_item_family_review"
OUT_JSON = OUT_DIR / "fastpath_pair_item_family_review.json"
OUT_MD = (ROOT / 'research/notes/representation_and_objectives/fastpath_pair_item_family_review.md')

PAYLOADS = {
    "chck82": A01 / "data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json",
    "shuffled86": A02 / "data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json",
    "coherent4M": A01 / "data/fastpath4M_coherent_cheap_eval/per_target/fastpath4M_coherent.json",
    "spanbreak4M": A01 / "data/fastpath4M_spanbreak_cheap_eval/per_target/fastpath4M_spanbreak.json",
}
SUMMARIES = {
    "coherent4M": A01 / "data/fastpath4M_coherent_summary/fastpath4M_coherent_summary.json",
    "spanbreak4M": A01 / "data/fastpath4M_spanbreak_summary/fastpath4M_spanbreak_summary.json",
}
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
FRAGILE_EWOK_GROUPS = {
    "material-dynamics", "physical-dynamics", "physical-interactions", "physical-relations",
    "spatial-relations", "social-relations", "social-interactions", "agent-properties",
}
ENTITY_FOCUS_PREFIXES = ("regular_5", "regular_4", "move_contents_5", "move_contents_4", "move_contents_3", "ambiref_4", "ambiref_3")


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


def payload_has_required(path: pathlib.Path) -> bool:
    j = load_json(path)
    if not isinstance(j, dict):
        return False
    tasks = j.get("tasks", {})
    required = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
    for c in required:
        rec = tasks.get(c)
        if not isinstance(rec, dict):
            return False
        if c != "Reading" and not rec.get("predictions"):
            return False
        if c == "Reading" and not rec.get("predictions"):
            return False
    scores = j.get("official_overall", {}).get("scores", {})
    return all(scores.get(c) is not None for c in CHEAP_COLS)


def payload_status(name: str, path: pathlib.Path) -> dict[str, Any]:
    exists = path.exists()
    j = load_json(path) if exists else None
    tasks = sorted(list(j.get("tasks", {}).keys())) if isinstance(j, dict) and isinstance(j.get("tasks"), dict) else []
    scores = j.get("official_overall", {}).get("scores", {}) if isinstance(j, dict) else {}
    summary = load_json(SUMMARIES[name]) if name in SUMMARIES else None
    return {
        "path": rel(path),
        "exists": exists,
        "ready": exists and payload_has_required(path),
        "task_keys": tasks,
        "scores": {c: scores.get(c) for c in CHEAP_COLS},
        "summary_path": rel(SUMMARIES[name]) if name in SUMMARIES else None,
        "summary_exists": bool(summary),
        "summary_status": summary.get("status") if isinstance(summary, dict) else None,
    }


def scores_for(path: pathlib.Path) -> dict[str, float] | None:
    j = load_json(path)
    if not isinstance(j, dict):
        return None
    scores = j.get("official_overall", {}).get("scores", {})
    out = {}
    for c in CHEAP_COLS:
        v = scores.get(c)
        if v is None:
            return None
        out[c] = float(v)
    return out


def cheap7(sc: dict[str, float] | None) -> float | None:
    if not sc:
        return None
    return float(mean(sc[c] for c in CHEAP_COLS))


def compare_payloads(label: str, base_name: str, cand_name: str, base: pathlib.Path, cand: pathlib.Path) -> dict[str, Any]:
    base_loader = PayloadLoader(base)
    cand_loader = PayloadLoader(cand)
    cols: dict[str, Any] = {}
    for col in DISCRETE_COLUMNS:
        cols[col] = compare_column(base_loader, cand_loader, col)
    aggregate = {
        "total_gain_items": int(sum(cols[c]["flip_counts"].get("gain", 0) for c in DISCRETE_COLUMNS)),
        "total_loss_items": int(sum(cols[c]["flip_counts"].get("loss", 0) for c in DISCRETE_COLUMNS)),
        "total_common_items": int(sum(cols[c]["n_common"] for c in DISCRETE_COLUMNS)),
        "discrete_payload_mean_delta": float(mean(float(cols[c]["delta_score_payload"]) for c in DISCRETE_COLUMNS)),
        "discrete_reconstructed_mean_delta": float(mean(float(cols[c]["delta_score_reconstructed_common"]) for c in DISCRETE_COLUMNS)),
    }
    aggregate["total_gain_minus_loss"] = int(aggregate["total_gain_items"] - aggregate["total_loss_items"])
    aggregate["loss_to_gain_ratio"] = None if aggregate["total_gain_items"] == 0 else float(aggregate["total_loss_items"] / aggregate["total_gain_items"])
    return {
        "label": label,
        "base": base_name,
        "candidate": cand_name,
        "base_payload": rel(base),
        "candidate_payload": rel(cand),
        "aggregate": aggregate,
        "columns": cols,
        "focus_readouts": focus_readouts(cols),
    }


def focus_readouts(cols: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ew = cols.get("EWoK")
    if ew:
        groups = ew.get("groups_all_by_item_net", [])
        out["EWoK_fragile_groups"] = [g for g in groups if str(g.get("group")) in FRAGILE_EWOK_GROUPS]
        out["EWoK_worst_12"] = ew.get("worst_groups_by_item_net", [])[:12]
        out["EWoK_best_12"] = ew.get("best_groups_by_item_net", [])[:12]
    ent = cols.get("Entity")
    if ent:
        groups = ent.get("groups_all_by_item_net", [])
        out["Entity_focus_high_op"] = [g for g in groups if str(g.get("group", "")).startswith(ENTITY_FOCUS_PREFIXES)]
        out["Entity_worst_12"] = ent.get("worst_groups_by_item_net", [])[:12]
        out["Entity_best_12"] = ent.get("best_groups_by_item_net", [])[:12]
    gp = cols.get("GlobalPIQA")
    if gp:
        out["GlobalPIQA_groups"] = gp.get("groups_all_by_item_net", [])
    supp = cols.get("Supplement")
    if supp:
        out["Supplement_worst_8"] = supp.get("worst_groups_by_item_net", [])[:8]
        out["Supplement_best_8"] = supp.get("best_groups_by_item_net", [])[:8]
    return out


def compact_comp_table(comp: dict[str, Any]) -> list[str]:
    lines = ["| column | payload Δ | reconstructed Δ | gains | losses | net | worst groups |", "|---|---:|---:|---:|---:|---:|---|"]
    for col in DISCRETE_COLUMNS:
        c = comp["columns"][col]
        worst = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in c.get("worst_groups_by_item_net", [])[:4])
        lines.append(f"| {col} | {float(c['delta_score_payload']):+.3f} | {float(c['delta_score_reconstructed_common']):+.3f} | {c['flip_counts'].get('gain',0)} | {c['flip_counts'].get('loss',0)} | {c['gain_minus_loss_items']:+d} | {worst} |")
    return lines


def route_read(statuses: dict[str, dict[str, Any]], scores: dict[str, dict[str, float] | None], comps: dict[str, Any]) -> str:
    if not statuses["coherent4M"]["ready"] or not statuses["spanbreak4M"]["ready"]:
        return "pending_coherent_spanbreak_cheap7"
    ch = cheap7(scores.get("chck82"))
    sh = cheap7(scores.get("shuffled86"))
    co = cheap7(scores.get("coherent4M"))
    sp = cheap7(scores.get("spanbreak4M"))
    if None in (ch, sh, co, sp):
        return "incomplete_score_extraction"
    c_vs_s = comps.get("coherent_minus_spanbreak", {}).get("aggregate", {}).get("discrete_payload_mean_delta")
    co_vs_ch = co - ch
    co_vs_sh = co - sh
    sp_vs_ch = sp - ch
    if co_vs_ch > 0 and co_vs_sh > 0 and (c_vs_s is not None and c_vs_s > 0):
        return "coherent_fastpath_promising_needs_superglue_and_item_retention_check"
    if c_vs_s is not None and c_vs_s <= 0:
        return "coherent_replay_not_better_than_spanbreak_control"
    if co_vs_ch <= 0:
        return "coherent_structure_effect_insufficient_vs_protected_chck82"
    if co_vs_sh <= 0:
        return "coherent_does_not_exceed_generic_shuffled_private_tail"
    return "fastpath_unresolved_mixed"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    statuses = {name: payload_status(name, p) for name, p in PAYLOADS.items()}
    scores = {name: scores_for(p) if statuses[name]["ready"] else None for name, p in PAYLOADS.items()}
    comparisons: dict[str, Any] = {}
    def ready(name: str) -> bool:
        return statuses[name]["ready"]
    if ready("spanbreak4M") and ready("coherent4M"):
        comparisons["coherent_minus_spanbreak"] = compare_payloads("coherent_minus_spanbreak", "spanbreak4M", "coherent4M", PAYLOADS["spanbreak4M"], PAYLOADS["coherent4M"])
    if ready("chck82") and ready("coherent4M"):
        comparisons["coherent_minus_chck82"] = compare_payloads("coherent_minus_chck82", "chck82", "coherent4M", PAYLOADS["chck82"], PAYLOADS["coherent4M"])
    if ready("chck82") and ready("spanbreak4M"):
        comparisons["spanbreak_minus_chck82"] = compare_payloads("spanbreak_minus_chck82", "chck82", "spanbreak4M", PAYLOADS["chck82"], PAYLOADS["spanbreak4M"])
    if ready("chck82") and ready("shuffled86"):
        comparisons["shuffled86_minus_chck82"] = compare_payloads("shuffled86_minus_chck82", "chck82", "shuffled86", PAYLOADS["chck82"], PAYLOADS["shuffled86"])
    if ready("shuffled86") and ready("coherent4M"):
        comparisons["coherent_minus_shuffled86"] = compare_payloads("coherent_minus_shuffled86", "shuffled86", "coherent4M", PAYLOADS["shuffled86"], PAYLOADS["coherent4M"])
    rr = route_read(statuses, scores, comparisons)
    out = {
        "status": "COMPLETE" if ready("coherent4M") and ready("spanbreak4M") else "PENDING",
        "created_utc": now(),
        "purpose": "Reviewer-side paired item/family analysis of frozen-82M fast-path replay arms using saved official-compatible predictions only.",
        "payload_statuses": statuses,
        "scores": scores,
        "cheap7": {k: cheap7(v) for k, v in scores.items()},
        "comparisons": comparisons,
        "route_read": rr,
        "interpretation": "Use coherent_minus_spanbreak as the structure control, coherent_minus_chck82 for retention/acquisition relative to protected endpoint, and coherent_minus_shuffled86 for generic private-tail redistribution. Do not infer a general principle from cheap7 alone.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research fast-path paired item/family review",
        "",
        f"Status: **{out['status']}**",
        f"Route read: `{rr}`",
        "",
        "This reviewer artifact consumes saved official-compatible payloads only. It is designed to judge the already-trained coherent/spanbreak research pair after cheap7 scoring, without new training.",
        "",
        "## Payload readiness and cheap scores",
        "",
        "| arm | ready | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading | payload |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ["chck82", "shuffled86", "coherent4M", "spanbreak4M"]:
        st = statuses[name]
        sc = scores.get(name) or {}
        lines.append(f"| {name} | {st['ready']} | {cheap7(scores.get(name))} | {sc.get('BLiMP')} | {sc.get('Supplement')} | {sc.get('EWoK')} | {sc.get('Entity')} | {sc.get('COMPS')} | {sc.get('GlobalPIQA')} | {sc.get('Reading')} | `{st['path']}` |")
    for cname, comp in comparisons.items():
        lines += ["", f"## {cname}", "", f"Aggregate: `{comp['aggregate']}`", ""]
        lines += compact_comp_table(comp)
        fr = comp.get("focus_readouts", {})
        lines += ["", "Focused readouts excerpt:", "", "```json", json.dumps(fr, indent=2)[:12000], "```"]
    lines += [
        "",
        "## How to use this file",
        "",
        "Continue the fast-path route only if coherent4M beats spanbreak4M on the load-bearing relation/state families, retains most chck82-correct fragile items while adding new correct items, and also beats the generic shuffled86 private-tail reference beyond expected seed/task noise. If coherent and spanbreak have similar turnover, or coherent only gives BLiMP/COMPS broad redistribution while losing Supplement/EWoK/Entity/GlobalPIQA, the route should stop before retention machinery or longer tails.",
        "",
        f"JSON: `{rel(OUT_JSON)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "route_read": rr, "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
