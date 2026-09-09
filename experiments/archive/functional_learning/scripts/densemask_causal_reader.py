#!/usr/bin/env python3
"""research: read the dense-mask/sparse-label causal arm against sparse and dense focus.

This script is intentionally a reader, not an evaluator.  It consumes completed
train/eval summaries and compares the exact causal contrast established in research:

  sparse (S,S): sparse focus labels and sparse masks
  dense-mask/sparse-label (M,S): sparse focus labels, dense masks
  dense (M,M): dense focus labels and dense masks

The causal interpretation depends on real output files.  If the dense-mask
run has not landed, the script records missing status rather than inferring a result.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import statistics
import time
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/densemask_causal_reader')
VERIFY_PATH = _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_fast_verify/densemask_sparselabel_fast_verify.json')
TRUSTED_COMPARISON_PATH = _public_path('experiments/archive/functional_learning/data/trusted_comparison_with_aoa/trusted_comparison_with_aoa.json')

ARMS = {
    "sparse_seed62064_SS": {
        "role": "sparse labels, sparse masks",
        "eval_summary": _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_eval/eval_summary.json'),
        "train_summary": _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/correspondence_focus_weighted/train_summary.json'),
        "expected_model_key": "unchanged_correspondence_focus_weighted_u0080",
        "seed": 62064,
    },
    "dense_seed62064_MM": {
        "role": "dense labels, dense masks",
        "eval_summary": _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_eval/eval_summary.json'),
        "train_summary": _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/train_summary.json'),
        "expected_model_key": "unchanged_correspondence_focus_weighted_u0080",
        "seed": 62064,
    },
    "dense_seed62065_MM": {
        "role": "dense labels, dense masks; fixed-policy replication",
        "eval_summary": _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_eval/eval_summary.json'),
        "train_summary": _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/train_summary.json'),
        "expected_model_key": "unchanged_correspondence_focus_weighted_u0080",
        "seed": 62065,
    },
    "densemask_sparselabel_seed62064_MS": {
        "role": "dense masks, sparse labels; causal arm",
        "eval_summary": _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_eval_seed62064/eval_summary.json'),
        "train_summary": _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/train_summary.json'),
        "pipeline_report": _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_pipeline_seed62064/pipeline_report.json'),
        "expected_model_key": "unchanged_correspondence_focus_weighted_u0080",
        "seed": 62064,
        "optional": True,
    },
}

FAST_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading", "equal_valid_mean"]


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_json(path: pathlib.Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def sub(a: Any, b: Any) -> float | None:
    if finite(a) and finite(b):
        return float(a) - float(b)
    return None


def get_model_key(eval_summary: dict[str, Any], expected: str) -> str | None:
    if not eval_summary:
        return None
    models = eval_summary.get("common_target", {}).get("model_summaries", {})
    if expected in models:
        return expected
    # Prefer non-parent trained endpoint if names differ in a future run.
    candidates = [k for k in models.keys() if k != "parent"]
    if len(candidates) == 1:
        return candidates[0]
    # Fall back to first cheap7 name when available.
    cheap = eval_summary.get("cheap7_results", [])
    if cheap:
        name = cheap[0].get("name")
        if name in models:
            return name
    return None


def extract_arm(label: str, spec: dict[str, Any]) -> dict[str, Any]:
    eval_path = spec["eval_summary"]
    train_path = spec["train_summary"]
    pipe_path = spec.get("pipeline_report")
    ev = load_json(eval_path)
    tr = load_json(train_path)
    pipe = load_json(pipe_path) if isinstance(pipe_path, pathlib.Path) else None
    out: dict[str, Any] = {
        "label": label,
        "role": spec.get("role"),
        "seed": spec.get("seed"),
        "paths": {
            "eval_summary": rel(eval_path),
            "train_summary": rel(train_path),
            "pipeline_report": rel(pipe_path) if isinstance(pipe_path, pathlib.Path) else None,
        },
        "present": bool(ev is not None and tr is not None),
        "missing": [rel(p) for p, obj in [(eval_path, ev), (train_path, tr)] if obj is None],
    }
    if ev is None or tr is None:
        if pipe is not None:
            out["pipeline_status"] = pipe.get("status")
        return out
    model_key = get_model_key(ev, spec.get("expected_model_key", ""))
    out["model_key"] = model_key
    out["train"] = {
        "status": tr.get("status"),
        "completed_updates": tr.get("completed_updates"),
        "total_words_consumed": tr.get("total_words_consumed"),
        "total_targets": tr.get("total_targets"),
        "total_focus_targets": tr.get("total_focus_targets"),
        "total_ordinary_targets": tr.get("total_ordinary_targets"),
        "focus_selected_groups_total": tr.get("focus_selected_groups_total"),
        "focus_candidate_groups_total": tr.get("focus_candidate_groups_total"),
        "aggregate_focus_target_fraction": tr.get("aggregate_focus_target_fraction"),
        "focus_lambda_arg": tr.get("focus_lambda_arg"),
        "mean_lambda_focus_optimized": tr.get("mean_lambda_focus_optimized"),
        "final_optimized_loss": (tr.get("final_update") or {}).get("optimized_loss"),
        "final_focus_loss": (tr.get("final_update") or {}).get("focus_loss"),
        "final_ordinary_loss": (tr.get("final_update") or {}).get("ordinary_loss"),
        "model_identity": tr.get("model_identity"),
    }
    # Cheap7 fast scores.
    cheap_scores: dict[str, Any] = {}
    cheap_name = None
    cheap = ev.get("cheap7_results", [])
    if cheap:
        # If multiple arms in an old eval summary, choose the expected model key if possible.
        chosen = None
        for r in cheap:
            if r.get("name") == model_key:
                chosen = r
                break
        if chosen is None:
            chosen = cheap[-1]
        cheap_name = chosen.get("name")
        cheap_scores = chosen.get("scores", {}) or {}
    out["fast"] = {"cheap_name": cheap_name, "scores": {k: cheap_scores.get(k) for k in FAST_COLS}}
    ref = ev.get("reference_tail_cheap7_scores", {}) or {}
    out["fast"]["delta_vs_tail_reference"] = {k: sub(cheap_scores.get(k), ref.get(k)) for k in FAST_COLS}

    # Qwen trained-pair view-surface readout.
    q_models = ev.get("qwen_view_surface", {}).get("model_summaries", {})
    q = q_models.get(model_key, {}) if model_key else {}
    q_parent = q_models.get("parent", {})
    by_cond = q.get("by_condition", {})
    by_parent = q_parent.get("by_condition", {})
    out["qwen_view_surface"] = {
        "view_only_mean_nll": (by_cond.get("view_only") or {}).get("mean_nll"),
        "with_source_mean_nll": (by_cond.get("with_source") or {}).get("mean_nll"),
        "view_only_delta_nll_vs_parent": (by_cond.get("view_only") or {}).get("mean_delta_nll_vs_parent"),
        "with_source_delta_nll_vs_parent": (by_cond.get("with_source") or {}).get("mean_delta_nll_vs_parent"),
        "source_help": (q.get("source_help") or {}).get("mean_source_help"),
        "source_help_delta_vs_parent": (q.get("source_help") or {}).get("mean_delta_source_help_vs_parent"),
        "parent_source_help": (q_parent.get("source_help") or {}).get("mean_source_help"),
        "n_targets": (by_cond.get("view_only") or {}).get("n_targets"),
        "n_pairs": (by_cond.get("view_only") or {}).get("n_pairs"),
    }

    # Common-target controlled source movement.
    common = ev.get("common_target", {})
    deltas = common.get("deltas_vs_parent", {}).get(model_key, {}) if model_key else {}
    models = common.get("model_summaries", {})
    cm = models.get(model_key, {}) if model_key else {}
    parent = models.get("parent", {})
    out["common_target"] = {
        "mean_delta_expected_margin_vs_parent": deltas.get("mean_delta_expected_margin_vs_parent"),
        "condition_deltas_vs_parent": {k: (v or {}).get("mean_delta") for k, v in (deltas.get("by_condition") or {}).items()},
        "split_deltas_vs_parent": {k: (v or {}).get("mean_delta") for k, v in (deltas.get("by_split") or {}).items()},
        "source_follow": cm.get("source_follow"),
        "condition_success": {k: {"success": (v or {}).get("success"), "n": (v or {}).get("n"), "mean_expected_margin": (v or {}).get("mean_expected_margin")} for k, v in (cm.get("by_condition") or {}).items()},
        "parent_source_follow": parent.get("source_follow"),
    }
    if pipe is not None:
        out["pipeline_status"] = pipe.get("status")
        out["pipeline_finished_utc"] = pipe.get("finished_utc")
    return out


def pairwise_delta(arms: dict[str, dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    aa = arms.get(a, {})
    bb = arms.get(b, {})
    out = {"a": a, "b": b, "present": bool(aa.get("present") and bb.get("present"))}
    if not out["present"]:
        out["reason"] = "one_or_both_arms_missing"
        return out
    out["fast_delta_a_minus_b"] = {k: sub((aa.get("fast") or {}).get("scores", {}).get(k), (bb.get("fast") or {}).get("scores", {}).get(k)) for k in FAST_COLS}
    out["qwen_delta_a_minus_b"] = {
        "view_only_delta_nll_vs_parent": sub(aa.get("qwen_view_surface", {}).get("view_only_delta_nll_vs_parent"), bb.get("qwen_view_surface", {}).get("view_only_delta_nll_vs_parent")),
        "with_source_delta_nll_vs_parent": sub(aa.get("qwen_view_surface", {}).get("with_source_delta_nll_vs_parent"), bb.get("qwen_view_surface", {}).get("with_source_delta_nll_vs_parent")),
        "source_help_delta_vs_parent": sub(aa.get("qwen_view_surface", {}).get("source_help_delta_vs_parent"), bb.get("qwen_view_surface", {}).get("source_help_delta_vs_parent")),
    }
    ca = aa.get("common_target", {})
    cb = bb.get("common_target", {})
    conds = sorted(set((ca.get("condition_deltas_vs_parent") or {}).keys()) | set((cb.get("condition_deltas_vs_parent") or {}).keys()))
    splits = sorted(set((ca.get("split_deltas_vs_parent") or {}).keys()) | set((cb.get("split_deltas_vs_parent") or {}).keys()))
    out["common_delta_a_minus_b"] = {
        "mean_delta_expected_margin_vs_parent": sub(ca.get("mean_delta_expected_margin_vs_parent"), cb.get("mean_delta_expected_margin_vs_parent")),
        "condition_deltas": {k: sub((ca.get("condition_deltas_vs_parent") or {}).get(k), (cb.get("condition_deltas_vs_parent") or {}).get(k)) for k in conds},
        "split_deltas": {k: sub((ca.get("split_deltas_vs_parent") or {}).get(k), (cb.get("split_deltas_vs_parent") or {}).get(k)) for k in splits},
        "both_source_conditions_correct_delta": sub((ca.get("source_follow") or {}).get("both_source_conditions_correct"), (cb.get("source_follow") or {}).get("both_source_conditions_correct")),
        "mean_source_follow_swing_delta": sub((ca.get("source_follow") or {}).get("mean_source_follow_swing"), (cb.get("source_follow") or {}).get("mean_source_follow_swing")),
    }
    return out


def load_trusted_context() -> dict[str, Any]:
    tc = load_json(TRUSTED_COMPARISON_PATH)
    if not tc:
        return {"present": False, "path": rel(TRUSTED_COMPARISON_PATH)}
    # Keep only compact parts; schema can evolve.
    comparisons = tc.get("comparisons", {}) if isinstance(tc.get("comparisons"), dict) else tc.get("comparisons", [])
    return {
        "present": True,
        "path": rel(TRUSTED_COMPARISON_PATH),
        "status": tc.get("status"),
        "comparisons": comparisons,
    }


def write_md(path: pathlib.Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research dense-mask/sparse-label causal reader\n\n")
    lines.append(f"Created: {report['created_utc']}\n\n")
    lines.append("This reader compares sparse `(S,S)`, dense-mask/sparse-label `(M,S)`, and dense `(M,M)` when real outputs exist. Missing dense-mask files are not interpreted.\n\n")
    lines.append("## Arm status\n\n")
    lines.append("| arm | present | role | focus targets | focus groups | fast mean | Entity | source-help Δ | common Δ | both-source |\n")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, arm in report["arms"].items():
        tr = arm.get("train", {})
        fs = (arm.get("fast") or {}).get("scores", {})
        q = arm.get("qwen_view_surface", {})
        c = arm.get("common_target", {})
        sf = c.get("source_follow") or {}
        lines.append(
            f"| {label} | {arm.get('present')} | {arm.get('role')} | {tr.get('total_focus_targets')} | {tr.get('focus_selected_groups_total')} | {fs.get('equal_valid_mean')} | {fs.get('Entity')} | {q.get('source_help_delta_vs_parent')} | {c.get('mean_delta_expected_margin_vs_parent')} | {sf.get('both_source_conditions_correct')} |\n"
        )
    lines.append("\n## Core comparisons\n\n")
    for name, comp in report.get("pairwise", {}).items():
        lines.append(f"### {name}\n\n")
        lines.append(f"Present: `{comp.get('present')}`\n\n")
        if not comp.get("present"):
            lines.append(f"Reason: {comp.get('reason')}\n\n")
            continue
        lines.append("Fast Δ a-b: `" + json.dumps(comp.get("fast_delta_a_minus_b"), ensure_ascii=False) + "`\n\n")
        lines.append("Qwen Δ a-b: `" + json.dumps(comp.get("qwen_delta_a_minus_b"), ensure_ascii=False) + "`\n\n")
        lines.append("Common Δ a-b: `" + json.dumps(comp.get("common_delta_a_minus_b"), ensure_ascii=False) + "`\n\n")
    lines.append("## Verified implementation geometry\n\n")
    vg = report.get("verified_geometry", {})
    lines.append(f"Verification present: `{vg.get('present')}` path `{vg.get('path')}`\n\n")
    if vg.get("present"):
        inv = vg.get("core_invariants", {})
        lines.append("Core invariants: `" + json.dumps(inv, ensure_ascii=False) + "`\n\n")
        oa = vg.get("objective_arithmetic", {})
        lines.append(f"Dense/sparse target ratio: `{oa.get('dense64_to_sparse_total_focus_target_ratio')}`; dense-mask/dense target ratio: `{oa.get('dm64_to_dense64_total_focus_target_ratio')}`; per-label focus-weight ratio dense-to-dm: `{oa.get('dense64_to_dm64_mean_per_token_focus_weight_ratio')}`.\n\n")
    lines.append("## Interpretation handle\n\n")
    lines.append(report.get("interpretation_handle", "") + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = {label: extract_arm(label, spec) for label, spec in ARMS.items()}
    pairwise = {
        "MS_minus_SS_input_mask_at_sparse_labels": pairwise_delta(arms, "densemask_sparselabel_seed62064_MS", "sparse_seed62064_SS"),
        "MM64_minus_SS_dense_total_effect": pairwise_delta(arms, "dense_seed62064_MM", "sparse_seed62064_SS"),
        "MS_minus_MM64_sparse_labels_vs_dense_coverage": pairwise_delta(arms, "densemask_sparselabel_seed62064_MS", "dense_seed62064_MM"),
        "MM65_minus_MM64_dense_replication": pairwise_delta(arms, "dense_seed62065_MM", "dense_seed62064_MM"),
    }
    vf = load_json(VERIFY_PATH)
    verified_geometry = {
        "present": vf is not None,
        "path": rel(VERIFY_PATH),
    }
    if vf is not None:
        verified_geometry.update({
            "status": vf.get("status"),
            "core_invariants": vf.get("core_invariants"),
            "policy_totals_seed62064": {
                "sparse": (vf.get("policy_totals") or {}).get("sparse_seed62064"),
                "dense": (vf.get("policy_totals") or {}).get("dense_seed62064"),
                "densemask_sparselabel": (vf.get("policy_totals") or {}).get("densemask_sparselabel_seed62064"),
            },
            "objective_arithmetic": vf.get("objective_arithmetic"),
        })
    dm_present = bool(arms["densemask_sparselabel_seed62064_MS"].get("present"))
    interpretation = (
        "Dense-mask/sparse-label outputs are present, so compare `(M,S)-(S,S)` as the isolated effect of dense input masking under sparse labels/weight schedule, and `(M,S)-(M,M)` as sparse target sampling/higher per-label focus weight versus broad dense supervision and lower per-label weight."
        if dm_present else
        "Dense-mask/sparse-label outputs are not yet present. The script currently preserves the exact baselines and verified implementation geometry so that the causal-arm result can be interpreted when complete; it does not infer whether input masking or dense supervision caused dense gains."
    )
    report = {
        "status": "DENSEMASK_CAUSAL_READER_READY_WITH_DENSEMASK" if dm_present else "DENSEMASK_CAUSAL_READER_BASELINES_ONLY",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/densemask_causal_reader.py')),
        "out_dir": rel(args.out_dir),
        "arms": arms,
        "pairwise": pairwise,
        "verified_geometry": verified_geometry,
        "trusted_repaired_superglue_context": load_trusted_context(),
        "interpretation_handle": interpretation,
    }
    out_json = args.out_dir / "densemask_causal_reader.json"
    out_md = args.out_dir / "densemask_causal_reader.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_md(out_md, report)
    print(json.dumps({"status": report["status"], "out_json": rel(out_json), "out_md": rel(out_md), "densemask_present": dm_present}, indent=2), flush=True)


if __name__ == "__main__":
    main()
