#!/usr/bin/env python3
"""research corrected paired replay for context-credit BabyLM intervention.

This wrapper reuses the research trainer but repairs the consequential pairing issue in
carrier_residual training: the frozen carrier scoring forward must not run with dropout
or consume the student RNG stream.  It replaces compute_carrier_weights with a version
that:
  * temporarily switches the model to eval mode;
  * disables the private adapter for the carrier distribution;
  * isolates teacher-side RNG with torch.random.fork_rng;
  * restores the model training/private-adapter state before the student forward.

The resulting STANDARD and CARRIER_RESIDUAL arms can be compared as a cleaner paired
replay from the same chck_82M endpoint, same tail rows, same WWM masks, same seed and
matched student dropout sequence up to differences caused by the loss itself.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPTS))

import context_credit_trainer as prev  # noqa: E402


def _private_enabled_flags(model):
    flags = []
    for layer in model.deberta.encoder.layer:
        flags.append(bool(layer.private_adapter.enabled))
    return flags


def _restore_private_enabled_flags(model, flags):
    for layer, flag in zip(model.deberta.encoder.layer, flags):
        layer.private_adapter.enabled = bool(flag)
    model.config.private_adapter_enabled = bool(any(flags))


def compute_carrier_weights_deterministic(model, masked_inputs, attention_mask, labels, device):
    """Deterministic carrier-error weights without perturbing student dropout RNG."""
    was_training = bool(model.training)
    private_flags = _private_enabled_flags(model)
    target_mask = labels != -100
    devices = []
    if masked_inputs.is_cuda:
        idx = masked_inputs.device.index
        if idx is None:
            idx = torch.cuda.current_device()
        devices = [idx]

    try:
        # fork_rng restores CPU/CUDA RNG states even if future carrier code becomes stochastic.
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(990626)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(990626)
            model.eval()
            model.set_private_enabled(False)
            with torch.no_grad():
                carrier_logits = model(input_ids=masked_inputs, attention_mask=attention_mask).logits
                carrier_probs = F.softmax(carrier_logits, dim=-1)
                safe_labels = labels.clone()
                safe_labels[~target_mask] = 0
                p_correct = carrier_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
                weights = (1.0 - p_correct) * target_mask.float()
                n_targets = target_mask.sum()
                if n_targets > 0:
                    mean_w = weights.sum() / n_targets
                    if mean_w > 1e-8:
                        weights = weights / mean_w
                weights = weights.detach()
                del carrier_logits, carrier_probs, safe_labels, p_correct
    finally:
        _restore_private_enabled_flags(model, private_flags)
        if was_training:
            model.train()
        else:
            model.eval()

    return weights


# Monkey-patch the research trainer before invoking its training loop.
prev.compute_carrier_weights = compute_carrier_weights_deterministic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credit_mode", default="standard", choices=["standard", "carrier_residual"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(_public_path('experiments/archive/functional_learning/training/runs') / f"paired_{args.credit_mode}_seed{prev.TRAIN_SEED}")

    if args.smoke:
        prev.MAX_TAIL_CHARGED_WORDS = 60_000
        prev.SCHEDULE_TOTAL = 4
        prev.LOG_EVERY = 1
        prev.CHECKPOINT_WORDS = 30_000
        prev.ACCUM_STEPS = 2
        prev.MICRO_BATCH = 16

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "CORRECTED_PAIRED_REPLAY_CONFIG",
        "created_unix": time.time(),
        "wrapped_trainer": str((_public_path('experiments/archive/functional_learning/scripts/context_credit_trainer.py')).relative_to(ROOT)),
        "credit_mode": args.credit_mode,
        "pairing_repair": {
            "carrier_forward_eval_mode": True,
            "private_disabled_for_carrier": True,
            "teacher_rng_isolated_with_fork_rng": True,
            "teacher_seed_inside_fork_rng": 990626,
            "restores_student_train_state_before_main_forward": True,
            "motivation": "Avoid dropout-stochastic carrier weights and avoid consuming the student dropout RNG before the weighted arm's main training forward."
        },
        "expected_pair": {
            "standard_output_dir": str(_public_path('experiments/archive/functional_learning/training/runs/paired_standard_seed43023')),
            "carrier_residual_output_dir": str(_public_path('experiments/archive/functional_learning/training/runs/paired_carrier_residual_seed43023'))
        }
    }
    (out / "pairing_repair_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    prev.train(args)


if __name__ == "__main__":
    main()
