#!/usr/bin/env python3
"""research: dense-focus seed replication synthesis.

This script reads the completed seed62065 fixed-policy dense-focus replication and the
existing seed62064 dense-focus result, then writes a compact evidence object for the
frontier-validation decision.  It deliberately separates fast-screen replication,
trained-Qwen source-help, controlled original/altered-source common-target movement,
and future causal controls.  Official-compatible endpoints are tracked separately.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from typing import Any, Dict, Optional

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
OUT = _public_path('experiments/archive/functional_learning/data/dense_replication_synthesis')

PATHS = {
    "parent_fast_payload": _public_path('experiments/archive/functional_learning/data/dense_fast_payloads/coherent86_alpha075_fast_payload.json'),
    "seed62064_fast_payload": _public_path('experiments/archive/functional_learning/data/dense_fast_payloads/dense_focus_seed62064_u0080_fast_payload.json'),
    "seed62065_fast_payload": _public_path('experiments/archive/functional_learning/data/dense_seed62065_fast_payloads/dense_focus_seed62065_u0080_fast_payload.json'),
    "seed62064_train": _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/train_summary.json'),
    "seed62065_train": _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/train_summary.json'),
    "seed62064_eval": _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_eval/eval_summary.json'),
    "seed62065_eval": _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_eval/eval_summary.json'),
    "seed62064_view": _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_eval/qwen_view_surface/summary.json'),
    "seed62065_view": _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_eval/qwen_view_surface/summary.json'),
    "seed62065_vs_parent_transition": _public_path('experiments/archive/functional_learning/data/dense_seed62065_vs_coherent86_fast_transition/dense62065_vs_coherent86_fast_official_transition.json'),
    "seed62065_vs_seed62064_transition": _public_path('experiments/archive/functional_learning/data/dense_seed62065_vs_seed62064_fast_transition/dense62065_vs_dense62064_fast_official_transition.json'),
    "densemask_sparse_label_dryrun": _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_dryrun/densemask_sparse_label_patch_stats.json'),
    "alignment_copy_dryrun_plan": _public_path('experiments/archive/functional_learning/data/qwen_alignment_clue_factorial_copy_dryrun/plan.json'),
    "alignment_copy_smoke": _public_path('experiments/archive/functional_learning/data/qwen_alignment_clue_factorial_copy_cpu_smoke/summary.json'),
}
COHERENT86_OVERALL_AOA0 = 42.1210247099666
COHERENT86_OFFICIAL_REF = {
    "zero_reading_payload": "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json",
    "superglue_payload": "experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json",
    "overall_aoa0": COHERENT86_OVERALL_AOA0,
    "cheap7": 44.18142857142857,
    "superglue": 69.81922238969935,
}


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    x: Any = d
    for k in keys:
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return x


def scores_from_payload(p: pathlib.Path) -> Dict[str, Any]:
    js = load(p)
    s = dict(js.get("scores") or {})
    if "GlobalPIQA_mean" in s:
        s["GlobalPIQA"] = s["GlobalPIQA_mean"]
    if "equal_valid_mean" in s:
        s["cheap7_mean"] = s["equal_valid_mean"]
    return s


def delta_scores(seed_scores: Dict[str, Any], parent_scores: Dict[str, Any]) -> Dict[str, Optional[float]]:
    cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "cheap7_mean"]
    out: Dict[str, Optional[float]] = {}
    for c in cols:
        a = parent_scores.get(c)
        b = seed_scores.get(c)
        out[c] = float(b) - float(a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
    return out


def train_extract(p: pathlib.Path) -> Dict[str, Any]:
    j = load(p)
    return {
        "path": rel(p),
        "completed_updates": j.get("completed_updates"),
        "total_words_consumed": j.get("total_words_consumed"),
        "prefix_words": get(j, "prefix_info", "prefix_words"),
        "qwen_rows": get(j, "prefix_info", "qwen_rows"),
        "qwen_pair_segments": get(j, "prefix_info", "qwen_pair_segments"),
        "total_targets": j.get("total_targets"),
        "total_focus_targets": j.get("total_focus_targets"),
        "total_ordinary_targets": j.get("total_ordinary_targets"),
        "aggregate_focus_target_fraction": j.get("aggregate_focus_target_fraction"),
        "focus_selected_groups_total": j.get("focus_selected_groups_total"),
        "focus_candidate_groups_total": j.get("focus_candidate_groups_total"),
        "focus_selected_pair_refs_total": j.get("focus_selected_pair_refs_total"),
        "final_optimized_loss": get(j, "final_update", "optimized_loss"),
        "final_focus_loss": get(j, "final_update", "focus_loss"),
        "final_ordinary_loss": get(j, "final_update", "ordinary_loss"),
        "model_identity": j.get("model_identity"),
    }


def view_extract(p: pathlib.Path, model_key: str = "unchanged_correspondence_focus_weighted_u0080") -> Dict[str, Any]:
    j = load(p)
    m = get(j, "model_summaries", model_key, default={}) or {}
    return {
        "path": rel(p),
        "view_only_mean_nll": get(m, "by_condition", "view_only", "mean_nll"),
        "view_only_delta_nll_vs_parent": get(m, "by_condition", "view_only", "mean_delta_nll_vs_parent"),
        "with_source_mean_nll": get(m, "by_condition", "with_source", "mean_nll"),
        "with_source_delta_nll_vs_parent": get(m, "by_condition", "with_source", "mean_delta_nll_vs_parent"),
        "source_help_mean": get(m, "source_help", "mean_source_help"),
        "source_help_delta_vs_parent": get(m, "source_help", "mean_delta_source_help_vs_parent"),
        "n_targets": get(m, "source_help", "n_targets"),
        "n_pairs": get(m, "source_help", "n_pairs"),
    }


def common_extract(p: pathlib.Path, model_key: str = "unchanged_correspondence_focus_weighted_u0080") -> Dict[str, Any]:
    j = load(p)
    d = get(j, "common_target", "deltas_vs_parent", model_key, default={}) or {}
    abs_model = get(j, "common_target", "model_summaries", model_key, default={}) or {}
    return {
        "path": rel(p),
        "mean_delta_expected_margin_vs_parent": d.get("mean_delta_expected_margin_vs_parent"),
        "delta_no_source": get(d, "by_condition", "no_source", "mean_delta"),
        "delta_source_original": get(d, "by_condition", "source_original", "mean_delta"),
        "delta_source_altered": get(d, "by_condition", "source_altered", "mean_delta"),
        "delta_held_source": get(d, "by_split", "held_source", "mean_delta"),
        "delta_trained_content": get(d, "by_split", "trained_content", "mean_delta"),
        "both_source_conditions_correct": get(abs_model, "source_follow", "both_source_conditions_correct"),
        "source_follow_n": get(abs_model, "source_follow", "n"),
        "mean_source_follow_swing": get(abs_model, "source_follow", "mean_source_follow_swing"),
        "held_source_both_correct": get(abs_model, "source_follow", "by_split", "held_source", "both_correct"),
        "trained_content_both_correct": get(abs_model, "source_follow", "by_split", "trained_content", "both_correct"),
    }


def transition_extract(p: pathlib.Path) -> Dict[str, Any]:
    j = load(p)
    rows = get(j, "score_summary", "rows", default=[]) or []
    out = {r["column"]: r.get("computed_delta_b_minus_a") for r in rows if isinstance(r, dict) and "column" in r}
    comps = j.get("column_comparisons") or {}
    return {
        "path": rel(p),
        "computed_deltas": out,
        "net_item_delta": {k: v.get("net_item_delta_b_minus_a") for k, v in comps.items() if isinstance(v, dict)},
        "a_label": j.get("a_label"),
        "b_label": j.get("b_label"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parent_scores = scores_from_payload(PATHS["parent_fast_payload"])
    s64_scores = scores_from_payload(PATHS["seed62064_fast_payload"])
    s65_scores = scores_from_payload(PATHS["seed62065_fast_payload"])
    obj: Dict[str, Any] = {
        "status": "DENSE_REPLICATION_SYNTHESIS",
        "created_utc": now(),
        "coherent86_official_reference": COHERENT86_OFFICIAL_REF,
        "paths": {k: rel(v) for k, v in PATHS.items() if v.exists()},
        "frontier_status": "Dense focus has replicated the fast-screen and local source-conditioned pattern across seed62064/62065, but official-compatible seed62064 is still pending and seed62065 official-compatible evaluation has only been launched; no v5 is established.",
        "train": {
            "seed62064": train_extract(PATHS["seed62064_train"]),
            "seed62065": train_extract(PATHS["seed62065_train"]),
        },
        "fast_scores": {
            "parent": parent_scores,
            "seed62064": s64_scores,
            "seed62065": s65_scores,
            "seed62064_delta_vs_parent": delta_scores(s64_scores, parent_scores),
            "seed62065_delta_vs_parent": delta_scores(s65_scores, parent_scores),
        },
        "trained_qwen_view_surface": {
            "seed62064": view_extract(PATHS["seed62064_view"]),
            "seed62065": view_extract(PATHS["seed62065_view"]),
        },
        "common_target": {
            "seed62064": common_extract(PATHS["seed62064_eval"]),
            "seed62065": common_extract(PATHS["seed62065_eval"]),
        },
        "fast_transition_tables": {
            "seed62065_vs_parent": transition_extract(PATHS["seed62065_vs_parent_transition"]),
            "seed62065_vs_seed62064": transition_extract(PATHS["seed62065_vs_seed62064_transition"]),
        },
        "mechanism_controls_prepared": {
            "alignment_copy_dryrun_plan": load(PATHS["alignment_copy_dryrun_plan"]) if PATHS["alignment_copy_dryrun_plan"].exists() else None,
            "alignment_copy_smoke_summary": load(PATHS["alignment_copy_smoke"]) if PATHS["alignment_copy_smoke"].exists() else None,
            "densemask_sparse_label_dryrun": load(PATHS["densemask_sparse_label_dryrun"]) if PATHS["densemask_sparse_label_dryrun"].exists() else None,
        },
        "interpretation": {
            "replication": "The dense-focus policy is no longer a one-seed fast-screen anomaly: seed62065 closely reproduces seed62064 on fast Cheap7, Entity/GlobalPIQA gains, trained-Qwen source-help, and original/altered-source common-target movement.",
            "not_yet_frontier": "The decisive frontier coordinate is the official-compatible payload including full zero-shot, Reading, SuperGLUE and AoA=0 arithmetic against coherent86 42.1210; fast-screen replication is necessary practical evidence but not sufficient.",
            "mechanism": "Dense focus masks and supervises all selected Qwen second-view content groups at the same macro lambda=0.15. The mechanism is not established by the alignment-clue interaction alone; controlled original/altered-source correctness and official score preservation remain separate evidence streams.",
        },
    }
    out_json = _public_path('experiments/archive/functional_learning/data/dense_replication_synthesis/dense_replication_synthesis.json')
    out_json.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fs = obj["fast_scores"]
    qv = obj["trained_qwen_view_surface"]
    ct = obj["common_target"]
    trans = obj["fast_transition_tables"]
    dm = obj["mechanism_controls_prepared"].get("densemask_sparse_label_dryrun") or {}
    plan = obj["mechanism_controls_prepared"].get("alignment_copy_dryrun_plan") or {}
    lines = []
    lines.append("# research dense-focus seed replication synthesis\n")
    lines.append("## Frontier coordinate\n")
    lines.append(f"- Coherent86/v4 official projected Overall(AoA0): `{COHERENT86_OVERALL_AOA0}`.\n")
    lines.append("- At the time of this note, seed62064 official-compatible evaluation was pending; seed62065 official evaluation had started after replication. Neither pending evaluation supplied the fast-screen results below.\n")
    lines.append("## Fast-screen replication\n")
    for seed in ["seed62064", "seed62065"]:
        s = fs[seed]
        d = fs[f"{seed}_delta_vs_parent"]
        lines.append(f"- {seed}: cheap7 `{s.get('cheap7_mean')}` (delta `{d.get('cheap7_mean')}`), Entity `{s.get('Entity')}` (delta `{d.get('Entity')}`), GlobalPIQA `{s.get('GlobalPIQA')}` (delta `{d.get('GlobalPIQA')}`), BLiMP delta `{d.get('BLiMP')}`, Supplement delta `{d.get('Supplement')}`.\n")
    seed_delta = trans["seed62065_vs_seed62064"]["computed_deltas"]
    lines.append(f"- Seed62065 minus seed62064 on identical fast items: cheap7 `{seed_delta.get('cheap7_mean')}`, BLiMP `{seed_delta.get('BLiMP')}`, EWoK `{seed_delta.get('EWoK')}`, Entity `{seed_delta.get('Entity')}`, COMPS `{seed_delta.get('COMPS')}`, GlobalPIQA `{seed_delta.get('GlobalPIQA')}`, Reading `{seed_delta.get('Reading')}`.\n")
    lines.append("## Trained-Qwen view and source-help\n")
    for seed in ["seed62064", "seed62065"]:
        s = qv[seed]
        lines.append(f"- {seed}: view-only ΔNLL `{s.get('view_only_delta_nll_vs_parent')}`, with-source ΔNLL `{s.get('with_source_delta_nll_vs_parent')}`, source-help Δ `{s.get('source_help_delta_vs_parent')}` over `{s.get('n_targets')}` targets / `{s.get('n_pairs')}` pairs.\n")
    lines.append("## Controlled original/altered-source common-target movement\n")
    for seed in ["seed62064", "seed62065"]:
        s = ct[seed]
        lines.append(f"- {seed}: mean delta `{s.get('mean_delta_expected_margin_vs_parent')}`, no-source `{s.get('delta_no_source')}`, source-original `{s.get('delta_source_original')}`, source-altered `{s.get('delta_source_altered')}`, held-source `{s.get('delta_held_source')}`, trained-content `{s.get('delta_trained_content')}`, both source conditions correct `{s.get('both_source_conditions_correct')}/{s.get('source_follow_n')}`.\n")
    lines.append("## Mechanism controls prepared, not substituted for official validation\n")
    ts = get(plan, "task_stats", default={}) or {}
    lines.append(f"- Alignment×clue script now records copied/noncopied target classes; dry run sampled `{ts.get('n_base_targets')}` base targets with copy counts `{ts.get('base_target_copy_binary_counts')}` and length-matched target-absent shuffled sources.\n")
    dc = dm.get("counter") or {}
    lines.append(f"- Dense-mask/sparse-label training wrapper dry run: first macro labelled `{dc.get('label_selected_groups')}` groups / `{dc.get('label_target_tokens')}` target tokens while masking `{dc.get('mask_selected_groups')}` groups / `{dc.get('mask_target_tokens')}` tokens; label-to-mask token ratio `{dm.get('label_to_mask_token_ratio')}`. This separates dense input masking from dense supervised coverage for a future causal arm.\n")
    lines.append("## Interpretation\n")
    lines.append("Seed62065 reproduces seed62064 closely enough that dense unchanged-Qwen focus remains the leading practical candidate and merits official-compatible seed62065 evaluation. The common-target and Qwen view probes now look seed-stable, but they do not by themselves prove a general learning principle or a lawful v5. The official-compatible endpoint can still fail if full BLiMP/Supplement/EWoK/SuperGLUE costs erase the fast gains. Mechanism work should preserve the stronger original/altered-source evidence and use shuffled-source alignment only as a complementary trained-material readout; if the official signal survives, the dense-mask/sparse-label arm is the focused training contrast for input-side masking versus added targets.\n")
    out_md = _public_path('research/documents/functional_learning/data/dense_replication_synthesis/dense_replication_synthesis.md')
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": obj["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
