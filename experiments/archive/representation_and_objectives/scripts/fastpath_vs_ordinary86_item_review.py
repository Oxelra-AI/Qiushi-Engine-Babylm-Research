#!/usr/bin/env python3
"""research: pending-safe item/family review including ordinary scale1.75 chck_86M.

This script extends the research paired reviewer with the exposure-matched ordinary
scale1.75 chck_86M comparator.  It consumes saved official-compatible prediction
payloads only.  Its purpose is to distinguish coherent frozen-anchor fast-path replay
from (i) ordinary 82M->86M exposure, (ii) generic shuffled private-tail redistribution,
and (iii) merely avoiding a destructive spanbreak control.
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

OUT_DIR = A01 / "data/fastpath_vs_ordinary86_item_review"
OUT_JSON = OUT_DIR / "fastpath_vs_ordinary86_item_review.json"
OUT_MD = (ROOT / 'research/notes/representation_and_objectives/fastpath_vs_ordinary86_item_review.md')

PAYLOADS = {
    "chck82": A01 / "data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json",
    "ordinary86": A01 / "data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json",
    "shuffled_private86": A02 / "data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json",
    "coherent4M": A01 / "data/fastpath4M_coherent_cheap_eval/per_target/fastpath4M_coherent.json",
    "spanbreak4M": A01 / "data/fastpath4M_spanbreak_cheap_eval/per_target/fastpath4M_spanbreak.json",
}
SUMMARY_PATHS = {
    "ordinary86": A01 / "data/scale1p75_chck86_cheap7_summary/scale1p75_chck86_cheap7_summary.json",
    "coherent4M": A01 / "data/fastpath4M_coherent_summary/fastpath4M_coherent_summary.json",
    "spanbreak4M": A01 / "data/fastpath4M_spanbreak_summary/fastpath4M_spanbreak_summary.json",
    "coherent_superglue": A01 / "data/fastpath4M_coherent_superglue_summary/fastpath4M_coherent_superglue_summary.json",
}
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ENTITY_FOCUS_PREFIXES = ("move_contents_5", "move_contents_4", "move_contents_3", "ambiref_4", "ambiref_3", "regular_5", "regular_4")
RELATION_EWOK_GROUPS = {"material-dynamics", "physical-dynamics", "physical-interactions", "physical-relations", "spatial-relations", "social-relations", "social-interactions", "agent-properties"}


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
        if not isinstance(rec, dict) or not rec.get("predictions"):
            return False
    scores = j.get("official_overall", {}).get("scores", {})
    return all(scores.get(c) is not None for c in CHEAP_COLS)


def payload_status(name: str, path: pathlib.Path) -> dict[str, Any]:
    exists = path.exists()
    j = load_json(path) if exists else None
    scores = j.get("official_overall", {}).get("scores", {}) if isinstance(j, dict) else {}
    summary = load_json(SUMMARY_PATHS[name]) if name in SUMMARY_PATHS else None
    return {
        "path": rel(path),
        "exists": exists,
        "ready": exists and payload_has_required(path),
        "task_keys": sorted(list(j.get("tasks", {}).keys())) if isinstance(j, dict) and isinstance(j.get("tasks"), dict) else [],
        "scores": {c: scores.get(c) for c in CHEAP_COLS},
        "summary_path": rel(SUMMARY_PATHS[name]) if name in SUMMARY_PATHS else None,
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


def compare_payloads(label: str, base_name: str, cand_name: str) -> dict[str, Any]:
    base = PAYLOADS[base_name]
    cand = PAYLOADS[cand_name]
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
        "focus": focus_readouts(cols),
    }


def filter_groups(groups: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    return [g for g in groups if predicate(str(g.get("group", "")))]


def focus_readouts(cols: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ent = cols.get("Entity")
    if ent:
        groups = ent.get("groups_all_by_item_net", [])
        high = filter_groups(groups, lambda x: x.startswith(ENTITY_FOCUS_PREFIXES))
        out["Entity_high_operation_total"] = {
            "n": int(sum(int(g.get("n", 0)) for g in high)),
            "gain": int(sum(int(g.get("gain", 0)) for g in high)),
            "loss": int(sum(int(g.get("loss", 0)) for g in high)),
            "net_gain_minus_loss": int(sum(int(g.get("net_gain_minus_loss", 0)) for g in high)),
        }
        out["Entity_high_operation_groups"] = sorted(high, key=lambda g: int(g.get("net_gain_minus_loss", 0)), reverse=True)
        out["Entity_best_12"] = ent.get("best_groups_by_item_net", [])[:12]
        out["Entity_worst_12"] = ent.get("worst_groups_by_item_net", [])[:12]
    ew = cols.get("EWoK")
    if ew:
        groups = ew.get("groups_all_by_item_net", [])
        out["EWoK_relation_groups"] = filter_groups(groups, lambda x: x in RELATION_EWOK_GROUPS)
        out["EWoK_best_12"] = ew.get("best_groups_by_item_net", [])[:12]
        out["EWoK_worst_12"] = ew.get("worst_groups_by_item_net", [])[:12]
    gp = cols.get("GlobalPIQA")
    if gp:
        out["GlobalPIQA_groups"] = gp.get("groups_all_by_item_net", [])
    for col in ["BLiMP", "Supplement", "COMPS"]:
        c = cols.get(col)
        if c:
            out[f"{col}_best_8"] = c.get("best_groups_by_item_net", [])[:8]
            out[f"{col}_worst_8"] = c.get("worst_groups_by_item_net", [])[:8]
    return out


def compact_comp_lines(comp: dict[str, Any]) -> list[str]:
    lines = ["| column | payload Δ | reconstructed Δ | gains | losses | net |", "|---|---:|---:|---:|---:|---:|"]
    for col in DISCRETE_COLUMNS:
        c = comp["columns"][col]
        lines.append(f"| {col} | {float(c['delta_score_payload']):+.3f} | {float(c['delta_score_reconstructed_common']):+.3f} | {c['flip_counts'].get('gain',0)} | {c['flip_counts'].get('loss',0)} | {c['gain_minus_loss_items']:+d} |")
    return lines


def route_read(statuses: dict[str, dict[str, Any]], scores: dict[str, dict[str, float] | None], comps: dict[str, Any]) -> str:
    if not statuses["ordinary86"]["ready"]:
        return "pending_ordinary86_cheap7_payload"
    if not statuses["coherent4M"]["ready"]:
        return "missing_coherent_payload"
    co = cheap7(scores["coherent4M"])
    o86 = cheap7(scores["ordinary86"])
    ch = cheap7(scores["chck82"])
    if co is None or o86 is None or ch is None:
        return "score_extraction_incomplete"
    comp = comps.get("coherent_minus_ordinary86", {})
    ent_focus = comp.get("focus", {}).get("Entity_high_operation_total", {}) if comp else {}
    ent_net = ent_focus.get("net_gain_minus_loss")
    aggregate_net = comp.get("aggregate", {}).get("total_gain_minus_loss") if comp else None
    ewok_delta = comp.get("columns", {}).get("EWoK", {}).get("delta_score_payload") if comp else None
    if co <= ch:
        return "coherent_not_above_protected_chck82_after_measured_columns"
    if co <= o86:
        return "ordinary_extra_exposure_matches_or_exceeds_coherent_cheap7"
    if aggregate_net is not None and aggregate_net <= 0:
        return "coherent_beats_ordinary86_score_but_item_net_nonpositive"
    if ent_net is not None and ent_net > 0 and (ewok_delta is not None and float(ewok_delta) >= -0.2):
        return "coherent_exceeds_ordinary86_with_high_operation_entity_gain_and_limited_ewok_loss"
    return "coherent_vs_ordinary86_mixed_requires_critical_review"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    statuses = {name: payload_status(name, path) for name, path in PAYLOADS.items()}
    scores = {name: scores_for(path) if statuses[name]["ready"] else None for name, path in PAYLOADS.items()}
    def ready(name: str) -> bool:
        return statuses[name]["ready"]
    comparisons: dict[str, Any] = {}
    planned_pairs = [
        ("coherent_minus_ordinary86", "ordinary86", "coherent4M"),
        ("ordinary86_minus_chck82", "chck82", "ordinary86"),
        ("coherent_minus_chck82", "chck82", "coherent4M"),
        ("coherent_minus_spanbreak", "spanbreak4M", "coherent4M"),
        ("coherent_minus_shuffled_private86", "shuffled_private86", "coherent4M"),
        ("ordinary86_minus_shuffled_private86", "shuffled_private86", "ordinary86"),
    ]
    for label, base, cand in planned_pairs:
        if ready(base) and ready(cand):
            comparisons[label] = compare_payloads(label, base, cand)
    rr = route_read(statuses, scores, comparisons)
    summaries = {name: (load_json(path) if path.exists() else None) for name, path in SUMMARY_PATHS.items()}
    out = {
        "status": "COMPLETE" if ready("ordinary86") and ready("coherent4M") else "PENDING",
        "created_utc": now(),
        "purpose": "Compare coherent fast-path replay against exposure-matched ordinary scale1.75 chck_86M using saved prediction payloads and item/family transitions.",
        "payload_statuses": statuses,
        "scores": scores,
        "cheap7": {k: cheap7(v) for k, v in scores.items()},
        "summaries": {k: {"path": rel(SUMMARY_PATHS[k]), "exists": summaries[k] is not None, "status": summaries[k].get("status") if isinstance(summaries[k], dict) else None} for k in SUMMARY_PATHS},
        "comparisons": comparisons,
        "route_read": rr,
        "interpretation": "The key comparison is coherent_minus_ordinary86. A real fast-path mechanism should add high-operation Entity/multi-step tracking without broad compensated item losses, and should not only beat spanbreak or shuffled-private baselines.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research fast-path vs ordinary86 item review",
        "",
        f"Status: **{out['status']}**",
        f"Route read: `{rr}`",
        "",
        "## Payload readiness and cheap7",
        "",
        "| arm | ready | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading | payload |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ["chck82", "ordinary86", "shuffled_private86", "coherent4M", "spanbreak4M"]:
        st = statuses[name]
        sc = scores.get(name) or {}
        lines.append(f"| {name} | {st['ready']} | {cheap7(scores.get(name))} | {sc.get('BLiMP')} | {sc.get('Supplement')} | {sc.get('EWoK')} | {sc.get('Entity')} | {sc.get('COMPS')} | {sc.get('GlobalPIQA')} | {sc.get('Reading')} | `{st['path']}` |")
    lines += ["", "## Comparisons"]
    for label, comp in comparisons.items():
        lines += ["", f"### {label}", "", f"Aggregate: `{comp['aggregate']}`", ""]
        lines += compact_comp_lines(comp)
        focus = comp.get("focus", {})
        # Compact high-operation and relation focus for the markdown; full detail is in JSON.
        subset = {
            "Entity_high_operation_total": focus.get("Entity_high_operation_total"),
            "Entity_high_operation_groups": focus.get("Entity_high_operation_groups", [])[:12],
            "EWoK_relation_groups": focus.get("EWoK_relation_groups", []),
            "GlobalPIQA_groups": focus.get("GlobalPIQA_groups", []),
        }
        lines += ["", "Focused readouts:", "", "```json", json.dumps(subset, indent=2), "```"]
    lines += [
        "",
        "## Scientific use",
        "",
        "The fast-path route should continue only if coherent4M beats ordinary86, not just spanbreak, and the item/family evidence shows added high-operation Entity tracking without compensating losses in EWoK/GlobalPIQA/COMPS/Supplement. If ordinary86 matches the gains, then coherent replay is ordinary extra exposure/private-tail redistribution rather than a distinct protected learning mechanism.",
        "",
        f"JSON: `{rel(OUT_JSON)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "route_read": rr, "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
