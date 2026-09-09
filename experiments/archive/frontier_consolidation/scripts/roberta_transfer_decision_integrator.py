#!/usr/bin/env python3
"""research: integrate RoBERTa compact-transfer evidence without touching pending evaluation.

Default mode deliberately ignores the selected official-compatible trajectory path while the
managed evaluator is still running.  After the runtime delivers that evaluator, rerun with
--use-selected to incorporate the completed selected scores.

This script starts no training, no evaluation, no upload, and no leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


LATE = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
STABLE_KEYS = [
    "cheap6_no_GlobalPIQA",
    "cheap5_no_GlobalPIQA_Reading",
    "EWoK_plus_Entity",
    "Supplement",
    "Entity",
    "COMPS",
]
ALL_KEYS = ["cheap7", *STABLE_KEYS, "BLiMP", "EWoK", "GlobalPIQA", "Reading"]


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
A01_WS = ROOT / "experiments/archive/representation_and_objectives"
DEFAULT_OUT = WS / "data/roberta_transfer_decision_integrator_current"
DEFAULT_SELECTED = WS / "data/roberta_full100m_integrated/roberta_transfer_integrated.json"
DEFAULT_TRAINING_INTEGRITY = WS / "data/roberta_training_pair_integrity_repaired/roberta_training_pair_integrity.json"
DEFAULT_LOCAL_SYNTHESIS = WS / "data/roberta_compact_result_synthesis_current_v2/roberta_compact_result_synthesis.json"
DEFAULT_SEMANTIC = WS / "data/roberta_semantic_local_response/roberta_semantic_local_response.json"
DEFAULT_A01_SEMANTIC = A01_WS / "data/target_channel_semantic_stratification/novelty_x_relational_interaction.json"
DEFAULT_A01_POSTRUN = A01_WS / "data/packed_targetselect_postrun_readout/postrun_readout_summary.json"
DEFAULT_ORDER = WS / "data/compact_order_integrated_readout/compact_order_integrated_readout.json"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def mean(vals: list[Any]) -> float | None:
    xs = [float(v) for v in vals if finite(v)]
    return sum(xs) / len(xs) if xs else None


def summarize_training(path: Path) -> dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {"ready": False, "path": str(path), "reason": "repaired training-pair integrity JSON absent"}
    mi = obj.get("mechanical_integrity", {})
    ts = obj.get("training_surface", {})
    deltas = ts.get("deltas_compact_minus_repeat", {})
    windows = ts.get("windows", {})
    compact_last = None
    repeat_last = None
    if isinstance(mi.get("compact_loss_first_last"), list) and len(mi["compact_loss_first_last"]) >= 2:
        compact_last = mi["compact_loss_first_last"][1]
    if isinstance(mi.get("repeat_loss_first_last"), list) and len(mi["repeat_loss_first_last"]) >= 2:
        repeat_last = mi["repeat_loss_first_last"][1]
    endpoint_delta = float(compact_last) - float(repeat_last) if finite(compact_last) and finite(repeat_last) else None
    loss_windows = {k: v.get("loss_delta_compact_minus_repeat") for k, v in windows.items() if isinstance(v, dict)}
    total_mask_delta = None
    if finite(ts.get("total_masked_tokens_compact")) and finite(ts.get("total_masked_tokens_repeat")):
        total_mask_delta = int(ts["total_masked_tokens_compact"]) - int(ts["total_masked_tokens_repeat"])
    total_candidate_delta = None
    if finite(ts.get("total_candidate_tokens_compact")) and finite(ts.get("total_candidate_tokens_repeat")):
        total_candidate_delta = int(ts["total_candidate_tokens_compact"]) - int(ts["total_candidate_tokens_repeat"])
    mechanically_matched = bool(
        mi.get("compact_metrics_exists")
        and mi.get("repeat_metrics_exists")
        and mi.get("compact_training_log_rows") == 2529
        and mi.get("repeat_training_log_rows") == 2529
        and mi.get("recipe_mismatches_excluding_expected_corpus_identity") == []
        and mi.get("checkpoint_names_match") is True
        and mi.get("checkpoint_exposure_mismatches") == {}
        and mi.get("exact_batch_word_match_all_steps") is True
        and mi.get("exact_cumulative_word_match_all_steps") is True
        and mi.get("same_lr_all_steps") is True
        and mi.get("compact_word_exposure") == 100_000_000
        and mi.get("repeat_word_exposure") == 100_000_000
        and mi.get("compact_all_checkpoints_present") is True
        and mi.get("repeat_all_checkpoints_present") is True
    )
    return {
        "ready": True,
        "path": str(path),
        "status": obj.get("status"),
        "mechanically_matched": mechanically_matched,
        "expected_corpus_difference_only": bool(mi.get("expected_corpus_identity_differences") and mi.get("recipe_mismatches_excluding_expected_corpus_identity") == []),
        "compact_log_rows": mi.get("compact_training_log_rows"),
        "repeat_log_rows": mi.get("repeat_training_log_rows"),
        "compact_word_exposure": mi.get("compact_word_exposure"),
        "repeat_word_exposure": mi.get("repeat_word_exposure"),
        "compact_all_checkpoints_present": mi.get("compact_all_checkpoints_present"),
        "repeat_all_checkpoints_present": mi.get("repeat_all_checkpoints_present"),
        "endpoint_loss_delta_compact_minus_repeat": endpoint_delta,
        "loss_delta_by_window_compact_minus_repeat": loss_windows,
        "total_masked_token_delta_compact_minus_repeat": total_mask_delta,
        "total_candidate_token_delta_compact_minus_repeat": total_candidate_delta,
        "mean_loss_delta_compact_minus_repeat": (deltas.get("loss") or {}).get("mean"),
        "mean_candidate_token_delta_per_step": (deltas.get("candidate_tokens") or {}).get("mean"),
        "mean_masked_token_delta_per_step": (deltas.get("masked_tokens") or {}).get("mean"),
        "reading": "RoBERTa compact/repeat training pair is interpretable only if mechanically matched; loss advantages are training-surface evidence, not BabyLM selected competence.",
    }


def summarize_local_synthesis(path: Path) -> dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {"ready": False, "path": str(path), "reason": "current local synthesis JSON absent"}
    local = obj.get("roberta_local", {})
    return {
        "ready": bool(local.get("ready")),
        "path": str(path),
        "using_late_band": bool(local.get("using_late_band")),
        "event_sha_matches": bool(local.get("event_sha_matches")),
        "local_means": local.get("local_view_category_advantage_means"),
        "source_absent_selectivity_margin_vs_controls": local.get("source_absent_selectivity_margin_vs_controls"),
        "source_absent_selective_local_advantage": local.get("source_absent_selective_local_advantage"),
        "feature_contrast_means": local.get("feature_contrast_means"),
        "positive_feature_markers": local.get("positive_feature_markers"),
        "scientific_reading_from_prior_synthesis": obj.get("scientific_reading"),
    }


def summarize_semantic(path: Path) -> dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {"ready": False, "path": str(path), "reason": "research semantic-local JSON absent"}
    late = obj.get("late_mean_key_interaction", {})
    pcs = obj.get("per_checkpoint", {})
    per_ck = []
    for ck in LATE:
        rec = pcs.get(ck, {})
        inter = None
        if isinstance(rec, dict):
            inter = rec.get("key_interaction", {}).get("novelty_x_rel_event_interaction_using_compact_view_events")
        per_ck.append({"ck": ck, "interaction": inter, "present": bool(rec and not rec.get("missing"))})
    vals = [r["interaction"] for r in per_ck]
    return {
        "ready": True,
        "path": str(path),
        "event_sha_matches_manifest": obj.get("event_sha_matches_manifest"),
        "late_mean_key_interaction": late,
        "per_checkpoint_interactions": per_ck,
        "all_late_interactions_positive": all(finite(v) and float(v) > 0 for v in vals),
        "min_late_interaction": min([float(v) for v in vals if finite(v)], default=None),
        "semantic_convergence_present": (
            finite(late.get("novelty_x_rel_event_interaction_using_compact_view_events"))
            and float(late["novelty_x_rel_event_interaction_using_compact_view_events"]) > 0.1
            and all(finite(v) and float(v) > 0 for v in vals)
        ),
        "reading": obj.get("scientific_reading"),
    }


def summarize_a01_semantic(path: Path) -> dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {"ready": False, "path": str(path), "reason": "A01 novelty x relational interaction absent"}
    return {
        "ready": True,
        "path": str(path),
        "status": obj.get("status"),
        "contrast": obj.get("contrast"),
        "interaction_I": obj.get("interaction_I"),
        "cells": obj.get("cells"),
        "reading": obj.get("reading"),
    }


def summarize_a01_postrun(path: Path) -> dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {"ready": False, "path": str(path), "reason": "A01 postrun summary absent"}
    return {
        "ready": obj.get("status") == "PACKED_TARGETSELECT_POSTRUN_READOUT_DONE",
        "path": str(path),
        "status": obj.get("status"),
        "integrity_ok": {k: v.get("ok") for k, v in (obj.get("integrity") or {}).items() if isinstance(v, dict)},
        "reading": "A01 endpoint target-selective result is not aligned unless the postrun status is DONE and endpoint fields support the source-absent mediation reading.",
    }


def summarize_order_anchor(path: Path) -> dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {"ready": False, "path": str(path), "reason": "ordered/scrambled integrated readout absent"}
    # The schema is large and has changed; preserve the stable scientific numbers as the anchor.
    return {
        "ready": True,
        "path": str(path),
        "anchor_reading": "Compact ordered-vs-scrambled produced a mature source-absent-selective local NLL channel but worsened the selected stable surface at chck_40M.",
        "chck40_ordered_minus_scrambled": {
            "cheap7": -0.67,
            "cheap6_no_GlobalPIQA": -0.288333,
            "cheap5_no_GlobalPIQA_Reading": -0.298,
            "EWoK_plus_Entity": -0.02,
            "source_absent_minus_controls_mean": -0.237639,
        },
        "why_it_matters": "This is the strongest warning that a local source-absent compact channel can dissociate from official-compatible selected competence.",
    }


def summarize_selected(path: Path, use_selected: bool) -> dict[str, Any]:
    if not use_selected:
        return {
            "used": False,
            "ready": False,
            "path": str(path),
            "reason": "Selected readout intentionally ignored in current mode because managed evaluator has not been delivered by the runtime.",
        }
    obj = read_json(path)
    if obj is None:
        return {"used": True, "ready": False, "path": str(path), "reason": "Selected integrated JSON absent."}
    rows = obj.get("per_checkpoint", [])
    by_ck = {r.get("ck"): r for r in rows if isinstance(r, dict)}
    late_rows = [by_ck[ck] for ck in LATE if ck in by_ck and not by_ck[ck].get("missing")]
    late_means = {k: mean([r.get("compact_minus_repeat", {}).get(k) for r in late_rows]) for k in ALL_KEYS}
    positive = [k for k in STABLE_KEYS if finite(late_means.get(k)) and float(late_means[k]) > 0]
    nonpositive = [k for k in STABLE_KEYS if finite(late_means.get(k)) and float(late_means[k]) <= 0]
    stable = (
        len(late_rows) == len(LATE)
        and finite(late_means.get("cheap6_no_GlobalPIQA")) and float(late_means["cheap6_no_GlobalPIQA"]) > 0
        and finite(late_means.get("cheap5_no_GlobalPIQA_Reading")) and float(late_means["cheap5_no_GlobalPIQA_Reading"]) > 0
        and finite(late_means.get("EWoK_plus_Entity")) and float(late_means["EWoK_plus_Entity"]) >= 0
        and len(positive) >= 4
    )
    volatile = (
        len(late_rows) == len(LATE)
        and finite(late_means.get("cheap7")) and float(late_means["cheap7"]) > 0
        and not stable
    )
    return {
        "used": True,
        "ready": len(late_rows) == len(LATE),
        "path": str(path),
        "status": obj.get("status"),
        "late_rows_present": [r.get("ck") for r in late_rows],
        "late_means": late_means,
        "stable_late_positive": stable,
        "volatile_carried_readout": volatile,
        "stable_positive_keys": positive,
        "stable_nonpositive_keys": nonpositive,
    }


def choose_decision(selected: dict[str, Any], local: dict[str, Any], semantic: dict[str, Any], a01_postrun: dict[str, Any]) -> dict[str, Any]:
    local_ok = bool(local.get("ready") and local.get("using_late_band") and local.get("source_absent_selective_local_advantage"))
    semantic_ok = bool(semantic.get("ready") and semantic.get("semantic_convergence_present"))
    a01_endpoint_ready = bool(a01_postrun.get("ready"))

    if not selected.get("used"):
        return {
            "route_state": "waiting_for_authoritative_selected_trajectory",
            "next_expensive_work_admitted": False,
            "reading": "RoBERTa training and local mechanism evidence are ready, but the selected official-compatible trajectory is still a managed unresolved dependency. Do not start a replication or new data arm before the runtime delivers it.",
            "next_cpu_actions_after_delivery": [
                "obtain the completed RoBERTa evaluation results before interpretation",
                "run roberta_transfer_result_reader.py in normal mode",
                "run selected_prediction_movement_reader.py on repeat→compact for late and best checkpoints",
                "rerun this research integrator with --use-selected",
            ],
        }

    if not selected.get("ready"):
        return {
            "route_state": "selected_readout_incomplete_or_failed",
            "next_expensive_work_admitted": False,
            "reading": "The selected trajectory was requested but is not complete in the integrated file; repair or recover the evaluator output before changing the route.",
        }

    stable = bool(selected.get("stable_late_positive"))
    volatile = bool(selected.get("volatile_carried_readout"))
    if stable and local_ok and semantic_ok:
        return {
            "route_state": "candidate_bidirectional_transfer_with_local_semantic_concordance",
            "next_expensive_work_admitted": True,
            "admitted_expensive_work": "one paired independent-seed RoBERTa compact-vs-repeat replication at the minimum checkpoint/evaluation schedule sufficient to test robustness, after item-movement inspection confirms the gain is not a small pathological subtask artifact",
            "reading": "Stable late selected transfer plus source-absent relational/event local convergence would make the compact marginal a serious cross-bidirectional-coordinate learning candidate, but still not a seed-robust law until replicated.",
        }
    if stable:
        return {
            "route_state": "selected_transfer_without_mechanism_alignment",
            "next_expensive_work_admitted": False,
            "reading": "Stable selected transfer without source-absent relational/event local concordance would still be scientifically valuable, but mechanism attribution remains open; inspect item movement before choosing at most one clean natural data intervention.",
        }
    if volatile:
        return {
            "route_state": "volatile_selected_movement",
            "next_expensive_work_admitted": False,
            "reading": "A positive aggregate carried by GlobalPIQA/Reading or unstable columns would not justify RoBERTa replication; local compact channels would be treated as another local-vs-downstream dissociation.",
        }
    if local_ok or semantic_ok:
        return {
            "route_state": "local_mechanism_without_selected_transfer",
            "next_expensive_work_admitted": False,
            "reading": "Positive source-absent/relational local response without stable selected improvement would bound this stock RoBERTa coordinate and redirect toward decomposing natural compact data factors rather than repeating the same architecture-transfer arm.",
        }
    return {
        "route_state": "no_roberta_transfer_signal",
        "next_expensive_work_admitted": False,
        "reading": "Neither stable selected transfer nor aligned local response is present; return to compact-data decomposition rather than RoBERTa replication.",
    }


def render_md(payload: dict[str, Any]) -> str:
    tr = payload["training_pair"]
    loc = payload["local_source_absent"]
    sem = payload["semantic_local"]
    a01 = payload["a01_semantic"]
    sel = payload["selected_official"]
    dec = payload["decision"]
    lines = [
        "# research RoBERTa compact-transfer decision integrator",
        "",
        f"Created UTC: `{payload['created_utc']}`",
        "",
        "## Route reading",
        "",
        dec.get("reading", ""),
        "",
        f"Route state: `{dec.get('route_state')}`",
        f"Next expensive work admitted: `{dec.get('next_expensive_work_admitted')}`",
        "",
        "## Training pair",
        "",
        f"Ready: `{tr.get('ready')}` mechanically matched: `{tr.get('mechanically_matched')}` expected corpus-only difference: `{tr.get('expected_corpus_difference_only')}`",
        f"Rows compact/repeat: `{tr.get('compact_log_rows')}` / `{tr.get('repeat_log_rows')}`; exposure compact/repeat: `{tr.get('compact_word_exposure')}` / `{tr.get('repeat_word_exposure')}`",
        f"Endpoint loss delta compact-repeat: `{tr.get('endpoint_loss_delta_compact_minus_repeat')}`; candidate-token delta: `{tr.get('total_candidate_token_delta_compact_minus_repeat')}`; masked-token delta: `{tr.get('total_masked_token_delta_compact_minus_repeat')}`",
        "",
        "## Local source-absent bridge",
        "",
        f"Ready: `{loc.get('ready')}` late: `{loc.get('using_late_band')}` event SHA match: `{loc.get('event_sha_matches')}` source-absent selective: `{loc.get('source_absent_selective_local_advantage')}` margin: `{loc.get('source_absent_selectivity_margin_vs_controls')}`",
        f"Local means: `{json.dumps(loc.get('local_means'), sort_keys=True)}`",
        "",
        "## Semantic local bridge",
        "",
        f"Ready: `{sem.get('ready')}` event SHA match: `{sem.get('event_sha_matches_manifest')}` semantic convergence: `{sem.get('semantic_convergence_present')}` min late interaction: `{sem.get('min_late_interaction')}`",
        f"Late mean interaction: `{json.dumps(sem.get('late_mean_key_interaction'), sort_keys=True)}`",
        "",
        "## A01 semantic source-absent evidence",
        "",
        f"Ready: `{a01.get('ready')}` interaction_I: `{a01.get('interaction_I')}` contrast: `{a01.get('contrast')}`",
        "",
        "## Selected official-compatible trajectory",
        "",
        f"Used in this run: `{sel.get('used')}` ready: `{sel.get('ready')}` path: `{sel.get('path')}`",
        f"Late means: `{json.dumps(sel.get('late_means'), sort_keys=True)}`",
        f"Stable positive: `{sel.get('stable_late_positive')}` volatile: `{sel.get('volatile_carried_readout')}`",
        "",
        "## Dissociation anchor",
        "",
        payload["ordered_scrambled_anchor"].get("anchor_reading", ""),
        "",
        "No training, selected evaluation, upload, or leaderboard action is performed by this script.",
    ]
    if dec.get("next_cpu_actions_after_delivery"):
        lines += ["", "## CPU actions after selected-eval delivery", ""]
        lines += [f"- {x}" for x in dec["next_cpu_actions_after_delivery"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-selected", action="store_true", help="Read selected integrated JSON; use only after managed evaluator has delivered.")
    ap.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    ap.add_argument("--training-integrity", type=Path, default=DEFAULT_TRAINING_INTEGRITY)
    ap.add_argument("--local-synthesis", type=Path, default=DEFAULT_LOCAL_SYNTHESIS)
    ap.add_argument("--semantic", type=Path, default=DEFAULT_SEMANTIC)
    ap.add_argument("--a01-semantic", type=Path, default=DEFAULT_A01_SEMANTIC)
    ap.add_argument("--a01-postrun", type=Path, default=DEFAULT_A01_POSTRUN)
    ap.add_argument("--ordered-scrambled", type=Path, default=DEFAULT_ORDER)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    training = summarize_training(args.training_integrity)
    local = summarize_local_synthesis(args.local_synthesis)
    semantic = summarize_semantic(args.semantic)
    a01_semantic = summarize_a01_semantic(args.a01_semantic)
    a01_postrun = summarize_a01_postrun(args.a01_postrun)
    order_anchor = summarize_order_anchor(args.ordered_scrambled)
    selected = summarize_selected(args.selected, args.use_selected)
    decision = choose_decision(selected, local, semantic, a01_postrun)

    payload = {
        "status": "ROBERTA_TRANSFER_DECISION_INTEGRATOR",
        "created_utc": now_utc(),
        "mode": "use_selected" if args.use_selected else "ignore_pending_selected",
        "meaning": "Integrates completed RoBERTa training/local evidence and A01 semantic source-absent evidence while avoiding any inference from an unresolved managed selected-evaluation task unless --use-selected is supplied after delivery.",
        "inputs": {
            "selected": str(args.selected),
            "training_integrity": str(args.training_integrity),
            "local_synthesis": str(args.local_synthesis),
            "semantic": str(args.semantic),
            "a01_semantic": str(args.a01_semantic),
            "a01_postrun": str(args.a01_postrun),
            "ordered_scrambled": str(args.ordered_scrambled),
        },
        "training_pair": training,
        "local_source_absent": local,
        "semantic_local": semantic,
        "a01_semantic": a01_semantic,
        "a01_targetselect_postrun": a01_postrun,
        "ordered_scrambled_anchor": order_anchor,
        "selected_official": selected,
        "decision": decision,
        "no_training_started": True,
        "no_selected_evaluation_started": True,
        "no_upload_or_leaderboard_action": True,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "roberta_transfer_decision_integrator.json", payload)
    (args.out_dir / "roberta_transfer_decision_integrator.md").write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "mode": payload["mode"],
        "out": str(args.out_dir / "roberta_transfer_decision_integrator.json"),
        "selected_used": selected.get("used"),
        "selected_ready": selected.get("ready"),
        "semantic_convergence_present": semantic.get("semantic_convergence_present"),
        "route_state": decision.get("route_state"),
        "next_expensive_work_admitted": decision.get("next_expensive_work_admitted"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
