#!/usr/bin/env python3
"""research: prepare explicit pseudo-run metadata for late-average run views.

The selected evaluator requires run_dir/scientific_metrics.json before scoring. For a
post-hoc same-trajectory averaged checkpoint, that file must not pretend the average
is a chronological trained checkpoint. This helper writes a metadata file whose
purpose is only to satisfy the evaluator's provenance guard while recording:

- the candidate is an arithmetic average of named same-trajectory checkpoints;
- no additional training exposure was consumed by the averaged weights;
- the candidate should be scored only if a Lead decision makes stabilization active;
- the pseudo endpoint label is a wrapper alias, not a chronological checkpoint.

This script does not run evaluation and does not submit anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
from typing import Any, Dict


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-json", type=pathlib.Path, default=pathlib.Path("experiments/archive/frontier_consolidation/data/average_candidate_selected_eval_plan/average_candidate_selected_eval_plan.json"))
    ap.add_argument("--source-run-dir", type=pathlib.Path, default=pathlib.Path("experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"))
    ap.add_argument("--write", action="store_true", help="Actually write run_view/scientific_metrics.json; otherwise dry-run only.")
    args = ap.parse_args()

    plan = json.loads(args.plan_json.read_text(encoding="utf-8"))
    cand_manifest_path = pathlib.Path(plan["candidate_manifest"])
    cand_manifest = json.loads(cand_manifest_path.read_text(encoding="utf-8"))
    source_metrics_path = args.source_run_dir / "scientific_metrics.json"
    if not source_metrics_path.exists():
        raise FileNotFoundError(source_metrics_path)
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    run_view = pathlib.Path(plan["run_view"])
    endpoint_view = pathlib.Path(plan["endpoint_view"])
    candidate_dir = pathlib.Path(plan["candidate_dir"])
    model_path = candidate_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    # Preserve evaluator-friendly model fields while overwriting chronological
    # semantics with explicit averaging provenance.
    keep_fields = [
        "variant", "backend", "model_family", "parameter_count", "vocab_size",
        "tokenizer_label", "tokenizer_path", "loss_first", "loss_last", "seed",
        "extra_init_seed", "train_rng_seed", "mask_mode", "mask_switch_mode",
        "mask_switch_words", "seq_length", "batch_size", "optimizer",
        "learning_rate", "n_layer", "hidden_size", "n_head",
        "intermediate_size", "max_seq_length", "ffn_mult", "masking_curriculum",
    ]
    meta: Dict[str, Any] = {k: source_metrics.get(k) for k in keep_fields if k in source_metrics}
    meta.update({
        "variant": "posthoc_same_trajectory_weight_average_selected_eval_view",
        "backend": source_metrics.get("backend", "mlm"),
        "model_family": "AdapterDebertaV2ForMaskedLM",
        "parameter_count": cand_manifest.get("validation", {}).get("param_count", source_metrics.get("parameter_count")),
        "vocab_size": cand_manifest.get("validation", {}).get("vocab_size", source_metrics.get("vocab_size")),
        "word_exposure": None,
        "chronological_word_exposure": None,
        "source_endpoint_labels": cand_manifest.get("plan", {}).get("endpoints"),
        "additional_training_word_exposure": 0,
        "actual_training_steps": 0,
        "saved_checkpoints": [{
            "name": plan.get("pseudo_endpoint"),
            "path": str(endpoint_view),
            "target_word_exposure": None,
            "actual_cumulative_word_exposure": None,
            "pseudo_endpoint": True,
        }],
        "created_utc": now_utc(),
        "status": "PSEUDO_RUN_METADATA_FOR_POSTHOC_AVERAGE",
        "candidate": cand_manifest.get("candidate"),
        "candidate_dir": str(candidate_dir),
        "candidate_manifest": str(cand_manifest_path),
        "candidate_model_sha256": sha256(model_path),
        "averaged_source_endpoints": cand_manifest.get("plan", {}).get("endpoints"),
        "averaging_weights": cand_manifest.get("plan", {}).get("normalized_weights"),
        "source_model_files": cand_manifest.get("plan", {}).get("model_files"),
        "source_training_run_dir": str(args.source_run_dir),
        "source_training_metrics_sha256": sha256(source_metrics_path),
        "pseudo_endpoint_label": plan.get("pseudo_endpoint"),
        "endpoint_view": str(endpoint_view),
        "official_evaluation_performed_by_this_script": False,
        "leaderboard_submission_performed": False,
        "scientific_boundary": "This file permits selected-evaluator provenance capture for a post-hoc same-trajectory average. It must not be read as a trained chronological checkpoint or as additional BabyLM exposure. Score only after seed43122 evidence makes stabilization the active question.",
    })

    out_path = run_view / "scientific_metrics.json"
    summary = {
        "status": "AVERAGE_RUN_VIEW_METADATA_READY" if args.write else "AVERAGE_RUN_VIEW_METADATA_DRY_RUN",
        "run_view": str(run_view),
        "out_path": str(out_path),
        "would_write": bool(args.write),
        "candidate_model_sha256": meta["candidate_model_sha256"],
        "additional_training_word_exposure": meta["additional_training_word_exposure"],
        "actual_training_steps": meta["actual_training_steps"],
        "pseudo_endpoint_label": meta["pseudo_endpoint_label"],
        "scientific_boundary": meta["scientific_boundary"],
    }
    if args.write:
        run_view.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        md = run_view / "scientific_metrics_pseudo_average_note.md"
        md.write_text(
            "# Pseudo-run metadata for selected evaluation of same-trajectory average\n\n"
            f"Candidate: `{meta['candidate']}`\n\n"
            f"Model SHA256: `{meta['candidate_model_sha256']}`\n\n"
            "This average has zero additional training exposure and is not a chronological checkpoint. "
            "The pseudo endpoint exists only so the selected evaluator can reuse its normal model-loading path.\n\n"
            f"Source endpoints: `{meta['averaged_source_endpoints']}` with weights `{meta['averaging_weights']}`.\n\n"
            "Do not run or interpret selected scoring unless the cross-seed grid makes stabilization the active scientific question.\n",
            encoding="utf-8",
        )
        summary["note_path"] = str(md)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
