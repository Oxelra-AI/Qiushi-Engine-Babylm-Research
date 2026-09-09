#!/usr/bin/env python3
"""Create a truthful shadow run directory for evaluating LAMB lr0.005 chck_14M.

The original LAMB lr0.005 task timed out before final trainer cleanup and therefore
lacks `scientific_metrics.json`; however it saved legal HF checkpoints through
chck_14M. research evaluator refuses to run without metrics. This script creates a
separate shadow directory that symlinks the existing checkpoint and records the run
as partial/timed-out, preserving original provenance.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
import shutil
from pathlib import Path

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
SRC = _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M')
DST = _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_partial14M_shadow')
SRC_CKPT = _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M/hf_model/chck_14M')
DST_CKPT = _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_partial14M_shadow/hf_model/chck_14M')


def load_log():
    rows = []
    with (_public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M/training_log.jsonl')).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    if not (_public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M/hf_model/chck_14M/model.safetensors')).exists():
        raise FileNotFoundError(_public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M/hf_model/chck_14M/model.safetensors'))
    rows = load_log()
    if not rows:
        raise RuntimeError("no source training log rows")
    DST.mkdir(parents=True, exist_ok=True)
    (_public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_partial14M_shadow/hf_model')).mkdir(parents=True, exist_ok=True)
    if DST_CKPT.exists() or DST_CKPT.is_symlink():
        if DST_CKPT.is_symlink():
            DST_CKPT.unlink()
        elif DST_CKPT.is_dir():
            shutil.rmtree(DST_CKPT)
        else:
            DST_CKPT.unlink()
    # Use an absolute symlink target so evaluator subprocesses never depend on cwd.
    DST_CKPT.symlink_to(_public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M/hf_model/chck_14M'), target_is_directory=True)

    # Preserve exact training log/manifest copies for inspectability.
    shutil.copy2(_public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M/training_log.jsonl'), _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_partial14M_shadow/training_log.jsonl'))
    shutil.copy2(_public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M/example_order_manifest.json'), _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_partial14M_shadow/example_order_manifest.json'))

    last = rows[-1]
    metrics = {
        "variant": "masking_curriculum_wwm_fixed",
        "backend": "mlm",
        "model_family": "DebertaV2ForMaskedLM",
        "parameter_count": 34467424,
        "vocab_size": 16384,
        "tokenizer_label": "compliant16k_reinvest10M",
        "word_exposure": int(last["cumulative_word_exposure"]),
        "checkpoint_evaluated": "chck_14M",
        "checkpoint_target_word_exposure": 14000000,
        "partial_shadow_for_evaluation": True,
        "source_run": str(SRC),
        "source_checkpoint": str(SRC_CKPT),
        "recorded_outcome": "timed_out at max_runtime_sec=900",
        "loss_first": float(rows[0]["loss"]),
        "loss_last_logged": float(last["loss"]),
        "last_logged_step": int(last["step"]),
        "actual_training_steps_logged": len(rows),
        "masking_curriculum": "wwm_fixed",
        "mask_prob_start": 0.15,
        "mask_prob_end": 0.15,
        "switch_frac": 0.7,
        "n_amlm_updates": 0,
        "hidden_size": 480,
        "n_layer": 8,
        "n_head": 8,
        "ffn_mult": 4,
        "seed": 43,
        "extra_init_seed": 43022,
        "train_rng_seed": 43023,
        "seq_length": 256,
        "max_seq_length": 256,
        "seq_len_schedule": "",
        "saved_checkpoints": [{
            "name": "chck_14M",
            "target_word_exposure": 14000000,
            "actual_cumulative_word_exposure": 14000474,
            "path": str(DST_CKPT),
            "source_path": str(SRC_CKPT),
        }],
        "data_source_type": "example_jsonl",
        "example_jsonl": "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl",
        "example_jsonl_label": "lamb_lr005_partial14M_shadow",
    }
    (_public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_partial14M_shadow/scientific_metrics.json')).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "LAMB005_PARTIAL_SHADOW_CREATED", "dst": str(DST), "checkpoint_symlink": str(DST_CKPT), "last_logged": last}, indent=2))


if __name__ == "__main__":
    main()
