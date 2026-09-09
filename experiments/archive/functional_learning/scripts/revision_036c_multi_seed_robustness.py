#!/usr/bin/env python3
"""Step036c: Multi-seed robustness test of paired-context objective.

The template single-seed test (research) showed ce_only and combined both at 3/12
held flips with seed=43036. The prior history shows research seed=43033 achieved
10/12 but research seed=43035 achieved only 2/12. The key scientific question is
whether the paired loss improves ROBUSTNESS across seeds, not just one seed.

This script runs ce_only and combined arms across 5 seeds and reports per-seed
and aggregate held (U+R)/2 and held flip counts.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import sys

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the training infrastructure from research
from paired_context_objective import (
    seed_all, train_arm, add_decomposition,
    prepare_template_batch, compute_paired_losses,
    load_private_model, freeze_to_private_adapters,
    evaluate_groups, compact_eval,
)
from corruption_vs_loss_factorial import (
    TRAIN_ENTITY_PAIRS, HELD_ENTITY_PAIRS, STATE_WORDS, TEMPLATES,
)
from paired_factorial_and_update_probe import (
    build_groups_decoupled_unique, group_balance,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str,
                        default="models/frontier")
    parser.add_argument("--lambda_paired", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[43033, 43034, 43035, 43036, 43037])
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--private_bottleneck", type=int, default=128)
    parser.add_argument("--private_scale", type=float, default=0.75)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--held_offset", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    import torch
    device = torch.device(args.device)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    model_path = pathlib.Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    print(f"Tokenizer: vocab={tokenizer.vocab_size}, device={device}", flush=True)

    train_groups = build_groups_decoupled_unique(TRAIN_ENTITY_PAIRS, "train", TEMPLATES, STATE_WORDS)
    held_groups = build_groups_decoupled_unique(HELD_ENTITY_PAIRS, "held", TEMPLATES, STATE_WORDS,
                                                start_offset=args.held_offset)
    print(f"Groups: {len(train_groups)} train, {len(held_groups)} held", flush=True)

    all_results = {}
    for seed in args.seeds:
        for arm_name in ["ce_only", "combined"]:
            key = f"{arm_name}_seed{seed}"
            print(f"\n{'='*50}\n=== {key} ===\n{'='*50}", flush=True)
            args.seed = seed
            result = train_arm(arm_name, model_path, tokenizer, train_groups, held_groups, args, device)
            ft = result["final_train"]
            fh = result["final_held"]
            all_results[key] = {
                "seed": seed,
                "arm": arm_name,
                "held_flips": fh["n_recipient_flip_correct"],
                "held_n": fh["n_groups"],
                "held_recip_dep": fh["recipient_dependent_half"],
                "held_shared_pref": fh["shared_preference_half"],
                "held_U": fh["mean_update_new_minus_source"],
                "held_R": fh["mean_retain_source_minus_new"],
                "train_flips": ft["n_recipient_flip_correct"],
                "train_recip_dep": ft["recipient_dependent_half"],
                "trajectory": result["trajectory"],
            }
            torch.cuda.empty_cache()

    # Aggregate by arm
    for arm_name in ["ce_only", "combined"]:
        results_for_arm = [v for k, v in all_results.items() if v["arm"] == arm_name]
        held_deps = [r["held_recip_dep"] for r in results_for_arm]
        held_flips = [r["held_flips"] for r in results_for_arm]
        print(f"\n{arm_name}: held (U+R)/2 per seed: {[f'{d:.3f}' for d in held_deps]}", flush=True)
        print(f"{arm_name}: held flips per seed: {held_flips}", flush=True)
        print(f"{arm_name}: mean held (U+R)/2 = {sum(held_deps)/len(held_deps):.3f}", flush=True)
        print(f"{arm_name}: mean held flips = {sum(held_flips)/len(held_flips):.1f}/12", flush=True)

    # Save
    summary = {"status": "STEP036C_MULTI_SEED_ROBUSTNESS", "results": {}}
    for key, r in all_results.items():
        summary["results"][key] = {k: v for k, v in r.items() if k != "trajectory"}

    with open(out / "multi_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Compact table
    for key, r in sorted(all_results.items()):
        traj = r.get("trajectory", [])
        # Save trajectory
        with open(out / f"trajectory_{key}.json", "w") as f:
            json.dump(traj, f, indent=2)

    print("\n" + "=" * 50, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
