#!/usr/bin/env python3
"""research: Paired-context contrastive objective for recipient-dependent binding.

Scientific motivation
---------------------
Decomposition across all prior experimental conditions shows:
  (U+R)/2 ≈ 0 in every condition except one lucky seed (research seed=43033).
  (U-R)/2 >> 0 in trained conditions — the model learns shared new-phrase preference
  but NOT context-dependent recipient tracking.

Standard answer-only CE on both contexts (update and retain) can be satisfied by
learning a general preference for the correct candidate without distinguishing
which entity was updated. Only the lucky seed happened to find the context-dependent
solution; other seeds converge to the degenerate shared-preference solution.

This script implements a paired-context loss that directly targets (U+R)/2:

  L_paired = -log σ(d_update - d_retain)

where d(ctx) = log P(new_state | ctx) - log P(source_state | ctx).

Since d_update - d_retain = U + R = 2*(U+R)/2, this loss DIRECTLY optimizes the
recipient-dependent component, canceling shared candidate preference.

Combined: L = L_ce + λ * L_paired

Three arms:
  ce_only:   L = L_ce  (standard both-context answer CE)
  paired_only: L = L_paired  (context-contrastive only)
  combined:  L = L_ce + λ * L_paired  (full intervention)

Mode:
  template: one-token answers, research-style counterbalanced groups
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import copy
import json
import math
import pathlib
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))

from corruption_vs_loss_factorial import (
    TRAIN_ENTITY_PAIRS, HELD_ENTITY_PAIRS, STATE_WORDS, TEMPLATES,
    ContrastGroup, evaluate_groups, compact_eval,
    load_private_model, freeze_to_private_adapters,
    one_token_candidate_id, masked_final_text,
)
from paired_factorial_and_update_probe import (
    build_groups_decoupled_unique, group_balance,
)


# ---------------------------------------------------------------------------
# Decomposition helper
# ---------------------------------------------------------------------------

def add_decomposition(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Add (U+R)/2 and (U-R)/2 decomposition to evaluation result."""
    mean_U = ev["mean_update_new_minus_source"]
    mean_R = ev["mean_retain_source_minus_new"]
    ev["recipient_dependent_half"] = (mean_U + mean_R) / 2
    ev["shared_preference_half"] = (mean_U - mean_R) / 2
    ev["recipient_sum_UR"] = mean_U + mean_R  # = 2 * recipient_dependent_half
    for g in ev.get("per_group", []):
        u = g["update_new_minus_source"]
        r = g["retain_source_minus_new"]
        g["recipient_dependent_half"] = (u + r) / 2
        g["shared_preference_half"] = (u - r) / 2
    return ev


# ---------------------------------------------------------------------------
# Batched template preparation
# ---------------------------------------------------------------------------

def prepare_template_batch(groups: Sequence[ContrastGroup], tokenizer,
                           max_length: int = 256, device: torch.device = torch.device("cpu")):
    """Pre-tokenize all groups for batched forward passes.

    Returns update_ids, retain_ids, attention masks, mask positions, and token IDs.
    """
    update_encs = []
    retain_encs = []
    mask_pos_u_list = []
    mask_pos_r_list = []
    new_ids = []
    src_ids = []

    for g in groups:
        mt_u = masked_final_text(g.source_sentence, g.update_target_sentence, g.final_frame, tokenizer)
        mt_r = masked_final_text(g.source_sentence, g.update_distractor_sentence, g.final_frame, tokenizer)

        enc_u = tokenizer(mt_u, add_special_tokens=True, return_tensors="pt",
                          max_length=max_length, truncation=True, padding="max_length")
        enc_r = tokenizer(mt_r, add_special_tokens=True, return_tensors="pt",
                          max_length=max_length, truncation=True, padding="max_length")

        ids_u = enc_u["input_ids"].squeeze(0).tolist()
        ids_r = enc_r["input_ids"].squeeze(0).tolist()
        mp_u = [i for i, t in enumerate(ids_u) if t == tokenizer.mask_token_id]
        mp_r = [i for i, t in enumerate(ids_r) if t == tokenizer.mask_token_id]
        assert len(mp_u) == 1, f"Expected 1 mask in update for {g.group_id}, got {len(mp_u)}"
        assert len(mp_r) == 1, f"Expected 1 mask in retain for {g.group_id}, got {len(mp_r)}"

        new_id, _ = one_token_candidate_id(tokenizer, mt_u, g.new_state)
        source_id, _ = one_token_candidate_id(tokenizer, mt_u, g.source_state)

        update_encs.append(enc_u)
        retain_encs.append(enc_r)
        mask_pos_u_list.append(mp_u[0])
        mask_pos_r_list.append(mp_r[0])
        new_ids.append(new_id)
        src_ids.append(source_id)

    batch = {
        "update_ids": torch.cat([e["input_ids"] for e in update_encs], dim=0).to(device),
        "update_mask": torch.cat([e["attention_mask"] for e in update_encs], dim=0).to(device),
        "retain_ids": torch.cat([e["input_ids"] for e in retain_encs], dim=0).to(device),
        "retain_mask": torch.cat([e["attention_mask"] for e in retain_encs], dim=0).to(device),
        "mask_pos_u": torch.tensor(mask_pos_u_list, dtype=torch.long, device=device),
        "mask_pos_r": torch.tensor(mask_pos_r_list, dtype=torch.long, device=device),
        "new_ids": torch.tensor(new_ids, dtype=torch.long, device=device),
        "src_ids": torch.tensor(src_ids, dtype=torch.long, device=device),
    }
    return batch


# ---------------------------------------------------------------------------
# Vectorized paired loss computation
# ---------------------------------------------------------------------------

def compute_paired_losses(model, batch: Dict[str, torch.Tensor], arm: str, lambda_paired: float):
    """Compute losses for a batch of template pairs.

    Args:
        model: masked LM
        batch: from prepare_template_batch
        arm: 'ce_only', 'paired_only', 'combined'
        lambda_paired: weight for paired loss in combined arm

    Returns:
        total_loss (for backward), and diagnostic dict
    """
    N = batch["update_ids"].shape[0]
    idx = torch.arange(N, device=batch["update_ids"].device)

    # Forward both contexts
    out_u = model(input_ids=batch["update_ids"], attention_mask=batch["update_mask"])
    out_r = model(input_ids=batch["retain_ids"], attention_mask=batch["retain_mask"])

    # Extract logits at mask positions: [N, vocab]
    logits_u = out_u.logits[idx, batch["mask_pos_u"]]
    logits_r = out_r.logits[idx, batch["mask_pos_r"]]

    # Log softmax
    lp_u = F.log_softmax(logits_u, dim=-1)
    lp_r = F.log_softmax(logits_r, dim=-1)

    # Gather token log probs
    lp_new_u = lp_u[idx, batch["new_ids"]]     # log P(new | update)
    lp_src_u = lp_u[idx, batch["src_ids"]]      # log P(source | update)
    lp_new_r = lp_r[idx, batch["new_ids"]]      # log P(new | retain)
    lp_src_r = lp_r[idx, batch["src_ids"]]      # log P(source | retain)

    # d values (new-preference direction in each context)
    d_update = lp_new_u - lp_src_u   # want positive (new correct in update)
    d_retain = lp_new_r - lp_src_r   # want negative (source correct in retain)

    # CE on correct answers
    ce_update = -lp_new_u.mean()     # CE of new_state in update context
    ce_retain = -lp_src_r.mean()     # CE of source_state in retain context
    mean_ce = (ce_update + ce_retain) / 2

    # Paired loss: -log σ(d_update - d_retain) = -log σ(U + R)
    # d_update - d_retain = U + R (the recipient-dependent component × 2)
    mean_paired = (-F.logsigmoid(d_update - d_retain)).mean()

    # Select loss based on arm
    if arm == "ce_only":
        total_loss = mean_ce
    elif arm == "paired_only":
        total_loss = mean_paired
    elif arm == "combined":
        total_loss = mean_ce + lambda_paired * mean_paired
    else:
        raise ValueError(f"Unknown arm: {arm}")

    diag = {
        "loss": float(total_loss.item()),
        "ce_loss": float(mean_ce.item()),
        "paired_loss": float(mean_paired.item()),
        "mean_d_update": float(d_update.mean().item()),
        "mean_d_retain": float(d_retain.mean().item()),
        "mean_UR": float((d_update - d_retain).mean().item()),
        "mean_recip_dep_half": float((d_update - d_retain).mean().item() / 2),
        "mean_shared_pref_half": float((d_update + d_retain).mean().item() / 2),
    }
    return total_loss, diag


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_arm(arm_name: str, model_path: pathlib.Path, tokenizer,
              train_groups: Sequence[ContrastGroup], held_groups: Sequence[ContrastGroup],
              args, device: torch.device) -> Dict[str, Any]:
    """Train one arm and return trajectory + final evaluation."""
    seed_all(args.seed)
    model, load_info = load_private_model(model_path, args.private_bottleneck,
                                          args.private_scale, device)
    trainable = freeze_to_private_adapters(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # Pre-tokenize batches
    train_batch = prepare_template_batch(train_groups, tokenizer, args.seq_length, device)
    held_batch = prepare_template_batch(held_groups, tokenizer, args.seq_length, device)

    trajectory = []
    eval_epochs = set(range(0, args.epochs + 1, args.eval_every))
    eval_epochs.add(args.epochs)

    for epoch in range(args.epochs + 1):
        if epoch in eval_epochs:
            model.eval()
            with torch.no_grad():
                ev_train = add_decomposition(evaluate_groups(model, tokenizer, train_groups, device, f"train_e{epoch}"))
                ev_held = add_decomposition(evaluate_groups(model, tokenizer, held_groups, device, f"held_e{epoch}"))
            ce_t = compact_eval(ev_train)
            ce_h = compact_eval(ev_held)
            point = {
                "epoch": epoch,
                "train_U": ce_t["mean_update_new_minus_source"],
                "train_R": ce_t["mean_retain_source_minus_new"],
                "train_recip_dep": ev_train["recipient_dependent_half"],
                "train_shared_pref": ev_train["shared_preference_half"],
                "train_flips": ce_t["n_recipient_flip_correct"],
                "train_n": ce_t["n_groups"],
                "held_U": ce_h["mean_update_new_minus_source"],
                "held_R": ce_h["mean_retain_source_minus_new"],
                "held_recip_dep": ev_held["recipient_dependent_half"],
                "held_shared_pref": ev_held["shared_preference_half"],
                "held_flips": ce_h["n_recipient_flip_correct"],
                "held_n": ce_h["n_groups"],
            }
            trajectory.append(point)

            # Print
            print(f"[{arm_name}] e{epoch:04d}"
                  f" | tr flip={ce_t['n_recipient_flip_correct']}/{ce_t['n_groups']}"
                  f" U={ce_t['mean_update_new_minus_source']:+.3f}"
                  f" R={ce_t['mean_retain_source_minus_new']:+.3f}"
                  f" dep={(ev_train['recipient_dependent_half']):+.4f}"
                  f" shared={(ev_train['shared_preference_half']):+.4f}"
                  f" | he flip={ce_h['n_recipient_flip_correct']}/{ce_h['n_groups']}"
                  f" U={ce_h['mean_update_new_minus_source']:+.3f}"
                  f" R={ce_h['mean_retain_source_minus_new']:+.3f}"
                  f" dep={(ev_held['recipient_dependent_half']):+.4f}"
                  f" shared={(ev_held['shared_preference_half']):+.4f}",
                  flush=True)

        if epoch == args.epochs:
            break

        # Training step
        model.train()
        optimizer.zero_grad()
        loss, diag = compute_paired_losses(model, train_batch, arm_name, args.lambda_paired)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

    # Final detailed evaluation
    model.eval()
    with torch.no_grad():
        final_train = add_decomposition(evaluate_groups(model, tokenizer, train_groups, device, "final_train"))
        final_held = add_decomposition(evaluate_groups(model, tokenizer, held_groups, device, "final_held"))

    return {
        "arm": arm_name,
        "trajectory": trajectory,
        "final_train": final_train,
        "final_held": final_held,
        "args": {
            "seed": args.seed, "lr": args.lr, "epochs": args.epochs,
            "lambda_paired": args.lambda_paired, "weight_decay": args.weight_decay,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="research paired-context contrastive objective")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str,
                        default="models/frontier")
    parser.add_argument("--lambda_paired", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=43036)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--private_bottleneck", type=int, default=128)
    parser.add_argument("--private_scale", type=float, default=0.75)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--held_offset", type=int, default=7)
    parser.add_argument("--arms", nargs="+", default=["ce_only", "paired_only", "combined"],
                        help="Arms to run")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    model_path = pathlib.Path(args.model_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    print(f"Tokenizer loaded from {model_path}, vocab={tokenizer.vocab_size}, "
          f"mask={tokenizer.mask_token}, device={device}", flush=True)

    # Build groups
    train_groups = build_groups_decoupled_unique(TRAIN_ENTITY_PAIRS, "train", TEMPLATES, STATE_WORDS)
    held_groups = build_groups_decoupled_unique(HELD_ENTITY_PAIRS, "held", TEMPLATES, STATE_WORDS,
                                                start_offset=args.held_offset)
    print(f"Built {len(train_groups)} train / {len(held_groups)} held groups", flush=True)
    print(f"Train balance: {json.dumps(group_balance(train_groups))}", flush=True)
    print(f"Held balance: {json.dumps(group_balance(held_groups))}", flush=True)

    # Baseline evaluation
    seed_all(args.seed)
    model, _ = load_private_model(model_path, args.private_bottleneck, args.private_scale, device)
    model.eval()
    with torch.no_grad():
        base_train = add_decomposition(evaluate_groups(model, tokenizer, train_groups, device, "baseline_train"))
        base_held = add_decomposition(evaluate_groups(model, tokenizer, held_groups, device, "baseline_held"))
    del model
    torch.cuda.empty_cache()

    print(f"\n=== BASELINE DECOMPOSITION ===", flush=True)
    for tag, ev in [("train", base_train), ("held", base_held)]:
        print(f"  {tag}: U={ev['mean_update_new_minus_source']:+.4f} "
              f"R={ev['mean_retain_source_minus_new']:+.4f} "
              f"(U+R)/2={ev['recipient_dependent_half']:+.4f} "
              f"(U-R)/2={ev['shared_preference_half']:+.4f} "
              f"flip={ev['n_recipient_flip_correct']}/{ev['n_groups']}", flush=True)
    print(flush=True)

    # Run arms
    all_results = {"baseline_train": base_train, "baseline_held": base_held, "arms": {}}

    for arm_name in args.arms:
        print(f"\n{'=' * 60}", flush=True)
        print(f"=== ARM: {arm_name} (λ_paired={args.lambda_paired}) ===", flush=True)
        print(f"{'=' * 60}", flush=True)
        result = train_arm(arm_name, model_path, tokenizer, train_groups, held_groups, args, device)
        all_results["arms"][arm_name] = result
        torch.cuda.empty_cache()

    # Save results
    def compact_result(r):
        ft = r["final_train"]
        fh = r["final_held"]
        return {
            "arm": r["arm"],
            "final_train_flips": f"{ft['n_recipient_flip_correct']}/{ft['n_groups']}",
            "final_held_flips": f"{fh['n_recipient_flip_correct']}/{fh['n_groups']}",
            "final_train_recip_dep": ft["recipient_dependent_half"],
            "final_train_shared_pref": ft["shared_preference_half"],
            "final_held_recip_dep": fh["recipient_dependent_half"],
            "final_held_shared_pref": fh["shared_preference_half"],
            "final_train_U": ft["mean_update_new_minus_source"],
            "final_train_R": ft["mean_retain_source_minus_new"],
            "final_held_U": fh["mean_update_new_minus_source"],
            "final_held_R": fh["mean_retain_source_minus_new"],
        }

    summary = {
        "status": "PAIRED_CONTEXT_OBJECTIVE",
        "baseline_train_recip_dep": base_train["recipient_dependent_half"],
        "baseline_held_recip_dep": base_held["recipient_dependent_half"],
        "baseline_train_shared_pref": base_train["shared_preference_half"],
        "baseline_held_shared_pref": base_held["shared_preference_half"],
        "arms": {name: compact_result(r) for name, r in all_results["arms"].items()},
    }

    with open(out / "paired_context_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full results (per-group details)
    with open(out / "paired_context_full_results.json", "w") as f:
        # Remove per-group details from the large dump to keep file reasonable
        save_results = {
            "baseline_train": {k: v for k, v in base_train.items() if k != "per_group"},
            "baseline_held": {k: v for k, v in base_held.items() if k != "per_group"},
            "arms": {},
        }
        for name, r in all_results["arms"].items():
            save_results["arms"][name] = {
                "arm": r["arm"],
                "trajectory": r["trajectory"],
                "final_train_compact": compact_eval(r["final_train"]),
                "final_held_compact": compact_eval(r["final_held"]),
                "final_train_decomp": {
                    "recipient_dependent_half": r["final_train"]["recipient_dependent_half"],
                    "shared_preference_half": r["final_train"]["shared_preference_half"],
                },
                "final_held_decomp": {
                    "recipient_dependent_half": r["final_held"]["recipient_dependent_half"],
                    "shared_preference_half": r["final_held"]["shared_preference_half"],
                },
                "args": r["args"],
            }
        json.dump(save_results, f, indent=2)

    # Save trajectories for plotting
    for name, r in all_results["arms"].items():
        with open(out / f"trajectory_{name}.json", "w") as f:
            json.dump(r["trajectory"], f, indent=2)

    # Write summary markdown
    md_lines = [
        "# research paired-context contrastive objective",
        "",
        "## Scientific question",
        "",
        "Does a paired-context loss that directly targets (U+R)/2 (the recipient-dependent",
        "component) improve held-out generalization compared to standard CE on both contexts?",
        "",
        "## Decomposition baseline",
        "",
        f"- Train: (U+R)/2 = {base_train['recipient_dependent_half']:+.4f}, (U-R)/2 = {base_train['shared_preference_half']:+.4f}",
        f"- Held: (U+R)/2 = {base_held['recipient_dependent_half']:+.4f}, (U-R)/2 = {base_held['shared_preference_half']:+.4f}",
        "",
        "## Arms",
        "",
        "| Arm | Train flip | Held flip | Train (U+R)/2 | Held (U+R)/2 | Train (U-R)/2 | Held (U-R)/2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in args.arms:
        if name in all_results["arms"]:
            r = compact_result(all_results["arms"][name])
            md_lines.append(
                f"| {name} | {r['final_train_flips']} | {r['final_held_flips']} | "
                f"{r['final_train_recip_dep']:+.4f} | {r['final_held_recip_dep']:+.4f} | "
                f"{r['final_train_shared_pref']:+.4f} | {r['final_held_shared_pref']:+.4f} |"
            )
    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "If the combined arm achieves higher held (U+R)/2 than ce_only, the paired loss",
        "successfully directed optimization toward the context-dependent circuit rather than",
        "the degenerate shared-preference solution. This would be a specific mechanism for",
        "data-efficient learning: explicit paired-context supervision can install recipient-",
        "dependent computation that standard answer CE (even on both contexts) fails to",
        "generalize reliably.",
        "",
    ])

    with open(out / "paired_context_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    # Final print
    print("\n" + "=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
