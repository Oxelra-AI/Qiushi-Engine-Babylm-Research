#!/usr/bin/env python3
"""Select the smallest useful full-evaluation follow-up for FineWeb seqsafe96.

This script reads the repaired no-AoA trajectory summary and chooses which frozen
checkpoint(s), if any, deserve the more expensive SuperGLUE+AoA completion.  It does
not run evaluation.  It preserves the arithmetic that should decide whether the
cached 1.753M-word source-breadth contrast is worth spending full-eval time on.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

DEFAULT_SUMMARY = pathlib.Path(
    "experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval_repairseq"
    "fineweb_seqsafe96_delta_summary.json"
)
DEFAULT_OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_seqsafe96_full_eval")
DEFAULT_NOTE = pathlib.Path("research/notes/representation_and_objectives/fineweb_full_eval_endpoint_plan.md")
PUBLIC = pathlib.Path("experiments/archive/representation_and_objectives/data/public_component_tradeoffs/public_component_tradeoffs.json")

KNOWLEDGE_COLS = ["EWoK", "Entity", "COMPS", "GlobalPIQA"]
PROTECT_COLS = ["BLiMP", "Supplement", "Reading"]
NOAOA_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def f(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def ck_order(name: str) -> int:
    try:
        return int(name.split("_")[1].rstrip("M"))
    except Exception:
        return 10**9


def sum_cols(row: dict[str, Any], cols: list[str]) -> float | None:
    vals = [f(row.get(c)) for c in cols]
    if any(v is None for v in vals):
        return None
    return float(sum(v for v in vals if v is not None))


def load_refs() -> dict[str, Any]:
    refs: dict[str, Any] = {
        "public_leader_overall": 41.8,
        "public_leader_superglue": 69.79,
        "public_leader_aoa": 0.0,
        "inherited_cleanqwen_overall": 41.3443,
        "inherited_cleanqwen_superglue": None,
    }
    if not PUBLIC.exists():
        return refs
    pub = json.loads(PUBLIC.read_text(encoding="utf-8"))
    refs["public_leader_overall"] = f((pub.get("strict_small_leader") or {}).get("overall")) or refs["public_leader_overall"]
    for r in pub.get("top15", []):
        if r.get("rank") == 1:
            refs["public_leader_superglue"] = f(r.get("SuperGLUE")) or refs["public_leader_superglue"]
            refs["public_leader_aoa"] = f(r.get("AoA")) or 0.0
            break
    ours = pub.get("ours") or {}
    refs["inherited_cleanqwen_superglue"] = f(ours.get("(Super)GLUE"))
    refs["inherited_cleanqwen_overall"] = f(ours.get("Overall Average")) or refs["inherited_cleanqwen_overall"]
    return refs


def full_estimate(noaoa_equal7: float | None, superglue: float | None, aoa: float | None) -> float | None:
    if noaoa_equal7 is None or superglue is None or aoa is None:
        return None
    return (7.0 * noaoa_equal7 + superglue + aoa) / 9.0


def required_superglue_for_target(noaoa_equal7: float | None, target_overall: float, aoa: float = 0.0) -> float | None:
    if noaoa_equal7 is None:
        return None
    return 9.0 * target_overall - 7.0 * noaoa_equal7 - aoa


def build_rows(summary: dict[str, Any], refs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for ck, pair in sorted((summary.get("matched_rows") or {}).items(), key=lambda kv: ck_order(kv[0])):
        tr = pair.get("treatment") or {}
        cr = pair.get("control") or {}
        d = pair.get("delta_treatment_minus_control") or {}
        tr_eq = f(tr.get("equal7_full_eval"))
        cr_eq = f(cr.get("equal7_full_eval"))
        rows[ck] = {
            "checkpoint": ck,
            "treatment_equal7": tr_eq,
            "control_equal7": cr_eq,
            "delta_equal7": f(d.get("equal7_full_eval")),
            "knowledge_delta_sum": sum_cols(d, KNOWLEDGE_COLS),
            "protected_delta_sum": sum_cols(d, PROTECT_COLS),
            "delta_components": {c: f(d.get(c)) for c in NOAOA_COLS},
            "treatment_full_if_leader_sg_aoa0": full_estimate(tr_eq, refs.get("public_leader_superglue"), 0.0),
            "treatment_full_if_inherited_sg_aoa0": full_estimate(tr_eq, refs.get("inherited_cleanqwen_superglue"), 0.0),
            "required_superglue_for_41p8_if_aoa0": required_superglue_for_target(tr_eq, float(refs["public_leader_overall"]), 0.0),
        }
    return rows


def choose(summary: dict[str, Any], rows: dict[str, dict[str, Any]], refs: dict[str, Any]) -> dict[str, Any]:
    valid = {ck: r for ck, r in rows.items() if r.get("treatment_equal7") is not None and r.get("delta_equal7") is not None}
    if not valid:
        return {
            "should_run_full_eval": False,
            "reason": "No matched no-AoA rows with treatment/control scores were found.",
            "selected_checkpoints": [],
            "eval_targets": [],
        }

    best_treat = max(valid.items(), key=lambda kv: kv[1]["treatment_equal7"])
    best_delta = max(valid.items(), key=lambda kv: kv[1]["delta_equal7"])
    best_knowledge = max(valid.items(), key=lambda kv: kv[1]["knowledge_delta_sum"] if kv[1].get("knowledge_delta_sum") is not None else -10**9)
    primary_ck = best_treat[0]
    primary = best_treat[1]

    # Full completion is deliberately limited to one same-endpoint pair.  The no-AoA
    # trajectory already carries the component-pattern evidence; SuperGLUE+AoA should
    # answer whether the best frozen treatment endpoint is a real Overall candidate,
    # not spend extra compute on every interesting diagnostic endpoint.
    selected = [primary_ck]

    # Decide whether full completion can change the scientific judgment.  Positive route
    # evidence is either a material source-breadth gain in the intended component cluster
    # or a treatment endpoint that is at least competitive with the inherited no-AoA range.
    source_signal = (best_knowledge[1].get("knowledge_delta_sum") is not None and best_knowledge[1]["knowledge_delta_sum"] >= 1.5)
    overall_signal = primary["delta_equal7"] >= 0.20 or primary["treatment_equal7"] >= 41.6
    should_run = bool(source_signal or overall_signal)

    eval_targets: list[dict[str, str]] = []
    if should_run:
        for ck in selected:
            eval_targets.append({"source_target": summary.get("treatment_target", "fineweb_seqsafe96_treatment_repairseq"), "endpoint": ck, "eval_target": f"fineweb_seqsafe96_treatment_repairseq__{ck}"})
            eval_targets.append({"source_target": summary.get("control_target", "fineweb_seqsafe96_control_repairseq"), "endpoint": ck, "eval_target": f"fineweb_seqsafe96_control_repairseq__{ck}"})

    return {
        "should_run_full_eval": should_run,
        "reason": (
            "Run SuperGLUE+AoA for the selected same-endpoint treatment/control pair(s): the frozen no-AoA trajectory shows source-breadth movement large enough that full Overall and SuperGLUE/AoA tradeoffs can change the next route."
            if should_run else
            "Do not spend SuperGLUE+AoA time yet: the frozen no-AoA trajectory does not show enough source-breadth movement or endpoint competitiveness."
        ),
        "selected_checkpoints": selected,
        "primary_checkpoint": primary_ck,
        "best_by_treatment_equal7": {"checkpoint": best_treat[0], "row": best_treat[1]},
        "best_by_delta_equal7": {"checkpoint": best_delta[0], "row": best_delta[1]},
        "best_by_knowledge_delta_sum": {"checkpoint": best_knowledge[0], "row": best_knowledge[1]},
        "eval_targets": eval_targets,
        "selection_logic": {
            "primary": "best treatment equal7 across frozen no-AoA checkpoints",
            "endpoint_count": "one same-endpoint pair only; component-pattern alternatives remain in the no-AoA table unless the primary full result creates a new reason",
            "full_eval_use": "SuperGLUE+AoA only when no-AoA source-breadth movement can change route or Overall judgment",
        },
    }


def fmt(x: Any) -> str:
    return "" if x is None else f"{float(x):.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    args = ap.parse_args()

    summary_path = pathlib.Path(args.summary)
    out_dir = pathlib.Path(args.out_dir)
    note_path = pathlib.Path(args.note)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        payload = {
            "status": "AWAITING_NOAOA_SUMMARY",
            "summary": str(summary_path),
            "should_run_full_eval": False,
            "reason": "The repaired FineWeb no-AoA summary has not been produced yet.",
        }
        out_json = out_dir / "fineweb_full_eval_endpoint_plan.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(f"# FineWeb full-eval endpoint plan\n\nNo no-AoA summary yet: `{summary_path}`.\n\nJSON: `{out_json}`\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "out_json": str(out_json), "note": str(note_path)}, indent=2))
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    refs = load_refs()
    rows = build_rows(summary, refs)
    choice = choose(summary, rows, refs)
    payload = {
        "status": "FINEWEB_FULL_EVAL_ENDPOINT_PLAN",
        "summary": str(summary_path),
        "references": refs,
        "knowledge_cols": KNOWLEDGE_COLS,
        "protected_cols": PROTECT_COLS,
        "rows": rows,
        **choice,
    }
    out_json = out_dir / "fineweb_full_eval_endpoint_plan.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# FineWeb seqsafe96 full-evaluation endpoint plan\n\n"]
    lines.append(f"No-AoA summary: `{summary_path}`\n\n")
    lines.append(f"Run full completion now: `{payload['should_run_full_eval']}` — {payload['reason']}\n\n")
    if payload.get("selected_checkpoints"):
        lines.append("Selected checkpoint(s): " + ", ".join(f"`{x}`" for x in payload["selected_checkpoints"]) + "\n\n")
    lines.append("| checkpoint | treat eq7 | ctrl eq7 | delta eq7 | knowledge Δsum | protected Δsum | req SuperGLUE for 41.8 if AoA=0 | full est leader SG AoA0 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for ck, r in sorted(rows.items(), key=lambda kv: ck_order(kv[0])):
        lines.append(
            f"| {ck} | {fmt(r.get('treatment_equal7'))} | {fmt(r.get('control_equal7'))} | {fmt(r.get('delta_equal7'))} | "
            f"{fmt(r.get('knowledge_delta_sum'))} | {fmt(r.get('protected_delta_sum'))} | {fmt(r.get('required_superglue_for_41p8_if_aoa0'))} | {fmt(r.get('treatment_full_if_leader_sg_aoa0'))} |\n"
        )
    if payload.get("eval_targets"):
        lines.append("\nFull-eval targets to run sequentially after this plan:\n\n")
        for t in payload["eval_targets"]:
            lines.append(f"- `{t['eval_target']}` from `{t['source_target']}` at `{t['endpoint']}`\n")
    lines.append(f"\nJSON: `{out_json}`\n")
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "should_run_full_eval": payload["should_run_full_eval"], "selected_checkpoints": payload.get("selected_checkpoints"), "out_json": str(out_json), "note": str(note_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
