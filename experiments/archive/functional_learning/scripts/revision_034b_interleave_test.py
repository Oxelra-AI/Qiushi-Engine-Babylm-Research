#!/usr/bin/env python3
"""Step034b: Sequential gradient isolation test.

The factorial showed that SIMULTANEOUS bg gradients destroy binding even at
full answer weight. This script tests whether SEQUENTIAL bg gradients (applied
in separate optimization steps) also destroy binding.

Design:
Phase 1 (install): Answer-only with 15% corruption for 200 epochs (builds binding)
Phase 2 (interleave test): Two arms from the Phase 1 checkpoint:
  Arm A (continue_answer_only): 300 more answer-only epochs (control)
  Arm B (interleaved): Alternates between answer-only and bg-only steps
    - Odd steps: answer-only loss on corrupted input
    - Even steps: bg-only loss on corrupted input (no answer label)
    Total: 300 epochs of each = same total steps as arm A × 2

If Arm B preserves binding comparably to Arm A → sequential isolation works
If Arm B destroys binding → need full separation (separate data streams)
"""

from __future__ import annotations
import argparse, copy, json, pathlib, sys, time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import shared infrastructure from research
sys.path.insert(0, str(pathlib.Path("experiments/archive/functional_learning/scripts")))
from corruption_vs_loss_factorial import (
    TRAIN_ENTITY_PAIRS, HELD_ENTITY_PAIRS, STATE_WORDS, TEMPLATES,
    ContrastGroup, build_groups, groups_to_training_packets,
    is_word_start, masked_final_text, one_token_candidate_id,
    locate_span_token_positions, evaluate_groups, compact_eval,
    RecipientPacketDataset, collate_fn, apply_corruption_and_labels,
    load_private_model, freeze_to_private_adapters,
    validate_groups_tokenization, write_jsonl,
)


def train_phase1(model_path, tokenizer, train_groups, held_groups, args, device):
    """Phase 1: Install binding with answer-only + corruption."""
    model, load_info = load_private_model(model_path, args.private_bottleneck,
                                          args.private_scale, device)
    trainable = freeze_to_private_adapters(model)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    packets = groups_to_training_packets(train_groups)
    ds = RecipientPacketDataset(packets, tokenizer, seq_length=args.seq_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    mask_gen = torch.Generator(device=device)
    mask_gen.manual_seed(args.seed + 42)

    trajectory = []

    def record(epoch, loss_val=None):
        ev_tr = evaluate_groups(model, tokenizer, train_groups, device, tag=f"p1_tr_e{epoch}")
        ev_he = evaluate_groups(model, tokenizer, held_groups, device, tag=f"p1_he_e{epoch}")
        entry = {"epoch": epoch, "train": compact_eval(ev_tr), "held": compact_eval(ev_he)}
        if loss_val is not None:
            entry["loss"] = loss_val
        trajectory.append(entry)
        print(
            f"[phase1] e{epoch:04d}" + (f" L={loss_val:.4f}" if loss_val is not None else "") +
            f" | tr flip={ev_tr['n_recipient_flip_correct']}/{ev_tr['n_groups']}"
            f" | he flip={ev_he['n_recipient_flip_correct']}/{ev_he['n_groups']}"
            f" U={ev_he['mean_update_new_minus_source']:+.3f}"
            f" R={ev_he['mean_retain_source_minus_new']:+.3f}"
            f" N={ev_he['mean_neutral_source_minus_new']:+.3f}",
            flush=True,
        )

    record(0)
    model.train()
    for epoch in range(1, args.phase1_epochs + 1):
        epoch_loss = 0.0
        epoch_n = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_group = batch["word_group"].to(device)
            answer_pos = batch["answer_pos"].to(device)
            answer_gid = batch["answer_gid"].to(device)

            masked_inputs, answer_labels, bg_labels, st = apply_corruption_and_labels(
                input_ids, attention_mask, word_group, answer_pos, answer_gid,
                tokenizer, bg_mask_prob=0.15, label_mode="answer_only", gen=mask_gen,
            )
            outputs = model(input_ids=masked_inputs, attention_mask=attention_mask)
            logits = outputs.logits
            answer_mask = (answer_labels != -100)
            n_answer = int(answer_mask.sum().item())
            if n_answer > 0:
                loss = F.cross_entropy(
                    logits[answer_mask].view(-1, logits.size(-1)),
                    answer_labels[answer_mask].view(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                epoch_loss += float(loss.detach()) * n_answer
                epoch_n += n_answer

        if epoch % args.eval_every == 0 or epoch == args.phase1_epochs:
            record(epoch, epoch_loss / max(1, epoch_n))
            model.train()

    return model, opt, trainable, trajectory, mask_gen


def train_phase2_arm(arm_name, mode, model, opt, trainable, tokenizer,
                     train_groups, held_groups, mask_gen, args, device):
    """Phase 2: Continue from Phase 1 checkpoint.

    mode='answer_only': continue answer-only training (control)
    mode='interleaved': alternate answer-only and bg-only steps
    """
    packets = groups_to_training_packets(train_groups)
    ds = RecipientPacketDataset(packets, tokenizer, seq_length=args.seq_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    trajectory = []
    cum_stats = defaultdict(int)

    def record(epoch, loss_val=None):
        ev_tr = evaluate_groups(model, tokenizer, train_groups, device, tag=f"{arm_name}_tr_e{epoch}")
        ev_he = evaluate_groups(model, tokenizer, held_groups, device, tag=f"{arm_name}_he_e{epoch}")
        entry = {"epoch": epoch, "train": compact_eval(ev_tr), "held": compact_eval(ev_he)}
        if loss_val is not None:
            entry["loss"] = loss_val
        trajectory.append(entry)
        print(
            f"[{arm_name}] e{epoch:04d}" + (f" L={loss_val:.4f}" if loss_val is not None else "") +
            f" | tr flip={ev_tr['n_recipient_flip_correct']}/{ev_tr['n_groups']}"
            f" | he flip={ev_he['n_recipient_flip_correct']}/{ev_he['n_groups']}"
            f" U={ev_he['mean_update_new_minus_source']:+.3f}"
            f" R={ev_he['mean_retain_source_minus_new']:+.3f}"
            f" N={ev_he['mean_neutral_source_minus_new']:+.3f}",
            flush=True,
        )

    record(0)
    model.train()
    step_counter = 0

    for epoch in range(1, args.phase2_epochs + 1):
        epoch_loss = 0.0
        epoch_n = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_group = batch["word_group"].to(device)
            answer_pos = batch["answer_pos"].to(device)
            answer_gid = batch["answer_gid"].to(device)

            masked_inputs, answer_labels, bg_labels, st = apply_corruption_and_labels(
                input_ids, attention_mask, word_group, answer_pos, answer_gid,
                tokenizer, bg_mask_prob=0.15, label_mode="answer_plus_bg", gen=mask_gen,
            )
            outputs = model(input_ids=masked_inputs, attention_mask=attention_mask)
            logits = outputs.logits

            if mode == "answer_only":
                # Always answer-only
                answer_mask = (answer_labels != -100)
                n = int(answer_mask.sum().item())
                if n > 0:
                    loss = F.cross_entropy(
                        logits[answer_mask].view(-1, logits.size(-1)),
                        answer_labels[answer_mask].view(-1),
                    )
                else:
                    loss = None
                cum_stats["answer_steps"] += 1

            elif mode == "interleaved":
                # Alternate: even steps = answer-only, odd steps = bg-only
                if step_counter % 2 == 0:
                    # Answer-only step
                    answer_mask = (answer_labels != -100)
                    n = int(answer_mask.sum().item())
                    if n > 0:
                        loss = F.cross_entropy(
                            logits[answer_mask].view(-1, logits.size(-1)),
                            answer_labels[answer_mask].view(-1),
                        )
                    else:
                        loss = None
                    cum_stats["answer_steps"] += 1
                else:
                    # Bg-only step
                    bg_mask = (bg_labels != -100)
                    n = int(bg_mask.sum().item())
                    if n > 0:
                        loss = F.cross_entropy(
                            logits[bg_mask].view(-1, logits.size(-1)),
                            bg_labels[bg_mask].view(-1),
                        )
                    else:
                        loss = None
                    cum_stats["bg_steps"] += 1
            else:
                raise ValueError(f"Unknown mode {mode}")

            if loss is not None:
                n_targets = n
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                epoch_loss += float(loss.detach()) * n_targets
                epoch_n += n_targets

            step_counter += 1

        if epoch % args.eval_every == 0 or epoch == args.phase2_epochs:
            record(epoch, epoch_loss / max(1, epoch_n))
            model.train()

    final_tr = evaluate_groups(model, tokenizer, train_groups, device, tag=f"{arm_name}_final_tr")
    final_he = evaluate_groups(model, tokenizer, held_groups, device, tag=f"{arm_name}_final_he")
    return {
        "arm_name": arm_name,
        "mode": mode,
        "trajectory": trajectory,
        "cumulative_stats": dict(cum_stats),
        "final_train": compact_eval(final_tr),
        "final_held": compact_eval(final_he),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/frontier")
    ap.add_argument("--out-dir", default="experiments/archive/functional_learning/data/interleave_test")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--phase1-epochs", type=int, default=200)
    ap.add_argument("--phase2-epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--private-bottleneck", type=int, default=128)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=43034)
    ap.add_argument("--held-pattern-offset", type=int, default=7)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = pathlib.Path(args.model_path)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    print(f"Tokenizer loaded, device={device}", flush=True)

    train_groups = build_groups(TRAIN_ENTITY_PAIRS, "train", TEMPLATES, STATE_WORDS, start_offset=0)
    held_groups = build_groups(HELD_ENTITY_PAIRS, "held", TEMPLATES, STATE_WORDS,
                               start_offset=args.held_pattern_offset)
    tv = {"train": validate_groups_tokenization(tokenizer, train_groups),
          "heldout": validate_groups_tokenization(tokenizer, held_groups)}
    assert tv["train"]["n_issues"] == 0 and tv["heldout"]["n_issues"] == 0

    print(f"\n=== PHASE 1: Install binding (answer-only + corruption, {args.phase1_epochs} epochs) ===",
          flush=True)
    model, opt, trainable, p1_trajectory, mask_gen = train_phase1(
        model_path, tokenizer, train_groups, held_groups, args, device)

    # Save Phase 1 state for both arms
    p1_state = {k: v.clone() for k, v in model.state_dict().items()}
    p1_opt_state = copy.deepcopy(opt.state_dict())

    print(f"\n=== PHASE 2a: Continue answer-only (control, {args.phase2_epochs} epochs) ===",
          flush=True)
    mask_gen_a = torch.Generator(device=device)
    mask_gen_a.manual_seed(args.seed + 200)
    arm_a = train_phase2_arm(
        "continue_answer_only", "answer_only", model, opt, trainable,
        tokenizer, train_groups, held_groups, mask_gen_a, args, device)

    # Restore Phase 1 state for Arm B
    model.load_state_dict(p1_state)
    opt.load_state_dict(p1_opt_state)
    # Re-identify trainable params after state restoration
    trainable = []
    for name, p in model.named_parameters():
        if "private_adapter" in name:
            p.requires_grad_(True)
            trainable.append(p)
        else:
            p.requires_grad_(False)

    print(f"\n=== PHASE 2b: Interleaved answer-only + bg-only ({args.phase2_epochs} epochs) ===",
          flush=True)
    mask_gen_b = torch.Generator(device=device)
    mask_gen_b.manual_seed(args.seed + 300)
    arm_b = train_phase2_arm(
        "interleaved_answer_bg", "interleaved", model, opt, trainable,
        tokenizer, train_groups, held_groups, mask_gen_b, args, device)

    # Save results
    summary = {
        "status": "INTERLEAVE_TEST",
        "design": {
            "phase1": f"Answer-only + 15% corruption for {args.phase1_epochs} epochs (install binding)",
            "phase2a": f"Continue answer-only for {args.phase2_epochs} epochs (control)",
            "phase2b": f"Interleave answer-only and bg-only steps for {args.phase2_epochs} epochs (test)",
            "key_question": "Do SEQUENTIAL bg gradients destroy binding installed in Phase 1?",
        },
        "model_path": str(model_path),
        "seed": args.seed,
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "phase1_trajectory": p1_trajectory,
        "arm_a": arm_a,
        "arm_b": arm_b,
    }

    # Interpretation
    a_he = arm_a["final_held"]
    b_he = arm_b["final_held"]
    p1_he = p1_trajectory[-1]["held"] if p1_trajectory else {}
    interp = []
    interp.append(f"Phase 1 end: held flip={p1_he.get('n_recipient_flip_correct', '?')}/{p1_he.get('n_groups', '?')}")
    interp.append(f"Arm A (continue answer-only): held flip={a_he['n_recipient_flip_correct']}/{a_he['n_groups']}, "
                  f"U={a_he['mean_update_new_minus_source']:+.3f}, R={a_he['mean_retain_source_minus_new']:+.3f}")
    interp.append(f"Arm B (interleaved): held flip={b_he['n_recipient_flip_correct']}/{b_he['n_groups']}, "
                  f"U={b_he['mean_update_new_minus_source']:+.3f}, R={b_he['mean_retain_source_minus_new']:+.3f}")
    interp.append(f"Arm B stats: {arm_b['cumulative_stats']}")

    if b_he["n_recipient_flip_correct"] >= a_he["n_recipient_flip_correct"] - 2:
        interp.append("→ Sequential bg gradients preserve binding. Interleaved training is viable for BabyLM.")
    else:
        interp.append("→ Sequential bg gradients also damage binding. Need full gradient separation.")

    summary["interpretation"] = "\n".join(interp)
    (out_dir / "interleave_test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80, flush=True)
    print(json.dumps({
        "status": summary["status"],
        "phase1_held_flip": p1_he.get("n_recipient_flip_correct", "?"),
        "arm_a_held_flip": a_he["n_recipient_flip_correct"],
        "arm_b_held_flip": b_he["n_recipient_flip_correct"],
        "arm_a_held_U": a_he["mean_update_new_minus_source"],
        "arm_b_held_U": b_he["mean_update_new_minus_source"],
        "arm_a_held_R": a_he["mean_retain_source_minus_new"],
        "arm_b_held_R": b_he["mean_retain_source_minus_new"],
        "interpretation": summary["interpretation"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
