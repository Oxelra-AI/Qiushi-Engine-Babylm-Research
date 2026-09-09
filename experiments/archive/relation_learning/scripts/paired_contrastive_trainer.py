#!/usr/bin/env python3
"""research: Paired contrastive private-phase binding trainer.

Extends the research composition trainer with a paired contrastive term:
  softplus(L_a - L_b)

where L_a = mean CE loss on unchanged-entity (row A) answer tokens
      L_b = mean CE loss on updated-entity (row B) answer tokens

A global recency shift (increase P(recently_stated_content) everywhere):
  - Increases L_a (model outputs new state instead of correct source state)
  - Decreases L_b (model correctly outputs the new state)
  - Therefore L_a - L_b increases → softplus increases → gradient opposes the shift

Entity-conditioned correct model:
  - L_a ≈ L_b ≈ small → softplus(0) = constant → no gradient from contrastive term

The contrastive term makes the global recency shift unprofitable while preserving
the reward for entity-conditioned assignment.

All other features from research are preserved: non-answer KL on binding rows,
frame-varied rows, reference-mixed frame sampling, ordinary text interleaving.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import pathlib
import random
import shutil
import sys
import time
from collections import Counter, defaultdict, deque
from typing import Any

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/relation_learning')
WS = _public_path('experiments/archive/relation_learning')
REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
FUNCTIONAL_RELATION_STUDIES_SCRIPTS = _public_path('experiments/archive/relation_learning/scripts')
sys.path.insert(0, str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS))
sys.path.insert(0, str(FUNCTIONAL_RELATION_STUDIES_SCRIPTS))

import frozen82_fastpath_replay_trainer as S150  # noqa: E402
S73 = None  # imported inside main

CHCK82 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
COHERENT86 = _public_path('models/frontier')
DEFAULT_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
FRAME_DIR = _public_path('experiments/archive/relation_learning/data/frame_varied_recombination_rows')
FRAME_TRAIN_ROWS = _public_path('experiments/archive/relation_learning/data/frame_varied_recombination_rows/recombination_train_frame_seen.jsonl')
FRAME_HELD_ROWS = _public_path('experiments/archive/relation_learning/data/frame_varied_recombination_rows/recombination_heldout_frame_all.jsonl')
FRAME_BINDING_PAIRS = _public_path('experiments/archive/relation_learning/data/frame_varied_recombination_rows/binding_pairs_heldout_frame_all.jsonl')
CHCK82_WORDS = 82012495
COHERENT86_WORDS = 86005295
BINDING_EPOCH_WORDS_REFERENCE = 156634  # from research

# Import from the research/083 trainer for shared utilities
import private_binding_composition_trainer as S082  # noqa: E402


def binding_update_contrastive(
    model, tokenizer, optimizer, rows, device: torch.device, pad_id: int,
    gen: torch.Generator, args: argparse.Namespace, update_i: int, total_updates: int
) -> dict[str, Any]:
    """Binding update with paired contrastive loss.
    
    Splits the batch into A (unchanged_entity) and B (updated_entity) rows,
    computes separate answer CE losses, and adds softplus(L_a - L_b).
    """
    rows_a = [r for r in rows if str(r.get("row_id", "")).endswith("unchanged_entity")]
    rows_b = [r for r in rows if str(r.get("row_id", "")).endswith("updated_entity")]
    
    # Fall back to standard if no proper pairing
    if not rows_a or not rows_b:
        return S082.binding_update(model, tokenizer, optimizer, rows, device, pad_id, gen, args, update_i, total_updates)
    
    lr = S082.lr_at(update_i, total_updates, float(args.warmup_fraction), float(args.learning_rate))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    S150.set_private_enabled(model, True)
    
    # Process A rows (unchanged entity, correct answer = source state)
    batch_a = S73.collate(rows_a, pad_id)
    batch_a = {k: v.to(device) for k, v in batch_a.items()}
    masked_a, labels_a, st_a = S73.apply_answer_arm(batch_a, tokenizer, "answer_clean")
    out_a = model(input_ids=masked_a, attention_mask=batch_a["attention_mask"], labels=labels_a)
    loss_a = out_a.loss
    
    # Process B rows (updated entity, correct answer = new state)
    batch_b = S73.collate(rows_b, pad_id)
    batch_b = {k: v.to(device) for k, v in batch_b.items()}
    masked_b, labels_b, st_b = S73.apply_answer_arm(batch_b, tokenizer, "answer_clean")
    out_b = model(input_ids=masked_b, attention_mask=batch_b["attention_mask"], labels=labels_b)
    loss_b = out_b.loss
    
    # Combined answer CE
    answer_loss = 0.5 * (loss_a + loss_b)
    
    # Paired contrastive: penalize loss_a > loss_b (recency shortcut)
    contrastive_loss = F.softplus(loss_a - loss_b)
    contrastive_lambda = float(getattr(args, "contrastive_lambda", 1.0))
    
    total_loss = answer_loss + contrastive_lambda * contrastive_loss
    
    # Non-answer KL on all binding rows (both A and B)
    kl_value_a = kl_value_b = 0.0
    kl_positions_a = kl_positions_b = 0
    if float(args.binding_neutral_lambda) > 0:
        kl_loss_a, kl_value_a, kl_positions_a = S082.binding_nonanswer_kl(
            model, masked_a, batch_a["attention_mask"], batch_a["answer_mask"],
            int(args.binding_neutral_subsample)
        )
        if kl_loss_a is not None:
            total_loss = total_loss + float(args.binding_neutral_lambda) * kl_loss_a
        
        kl_loss_b, kl_value_b, kl_positions_b = S082.binding_nonanswer_kl(
            model, masked_b, batch_b["attention_mask"], batch_b["answer_mask"],
            int(args.binding_neutral_subsample)
        )
        if kl_loss_b is not None:
            total_loss = total_loss + float(args.binding_neutral_lambda) * kl_loss_b
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    
    labeled_a = int((labels_a != -100).sum().item())
    labeled_b = int((labels_b != -100).sum().item())
    frames = Counter(str(r.get("frame_id") or "single_frame") for r in rows)
    
    return {
        "kind": "binding_contrastive",
        "lr": lr,
        "loss": float(total_loss.detach().cpu()),
        "answer_loss": float(answer_loss.detach().cpu()),
        "loss_a": float(loss_a.detach().cpu()),
        "loss_b": float(loss_b.detach().cpu()),
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
        "contrastive_lambda": contrastive_lambda,
        "binding_nonanswer_kl_loss": kl_value_a + kl_value_b,
        "binding_nonanswer_kl_positions": kl_positions_a + kl_positions_b,
        "binding_neutral_lambda": float(args.binding_neutral_lambda),
        "labeled_positions_a": labeled_a,
        "labeled_positions_b": labeled_b,
        "n_rows_a": len(rows_a),
        "n_rows_b": len(rows_b),
        "frames_in_batch": dict(frames),
        **st_a,
    }


def main() -> None:
    """Wrapper that patches the binding_update in S082 trainer and runs it."""
    # Parse args first to add contrastive_lambda
    import importlib
    
    # Add contrastive_lambda to argument parser
    # We'll monkey-patch by modifying sys.argv before S082.main() or by
    # running the full trainer loop with our modified update function
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start", choices=["chck82_fresh", "coherent86"], default="chck82_fresh")
    ap.add_argument("--main-words", type=int, default=3992800)
    ap.add_argument("--kl-words", type=int, default=0)
    ap.add_argument("--binding-train-rows", default=str(S082.default_binding_train_rows()))
    ap.add_argument("--binding-held-rows", default=str(S082.default_binding_held_rows()))
    ap.add_argument("--binding-pairs", default=str(S082.default_binding_pairs()))
    ap.add_argument("--binding-frame-schedule", default="reference_mixed")
    ap.add_argument("--binding-reference-pairs-per-epoch", type=int, default=0)
    ap.add_argument("--binding-epochs", type=int, default=25)
    ap.add_argument("--binding-neutral-lambda", type=float, default=1.0)
    ap.add_argument("--binding-neutral-subsample", type=int, default=0)
    ap.add_argument("--contrastive-lambda", type=float, default=1.0, help="Weight for softplus(L_a - L_b) paired contrastive term")
    ap.add_argument("--example-jsonl", default=str(DEFAULT_STREAM))
    ap.add_argument("--skip-rows", type=int, default=530944)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--pair-batch-size", type=int, default=8)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--warmup-fraction", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--main-lambda", type=float, default=1.0)
    ap.add_argument("--neutral-lambda", type=float, default=1.0)
    ap.add_argument("--neutral-subsample", type=int, default=64)
    ap.add_argument("--private-adapter-bottleneck", type=int, default=128)
    ap.add_argument("--private-adapter-scale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=85085)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    S082.setup_cache(out)
    os.environ["CACHE_BASE"] = str((out / "import_cache").resolve())
    import binding_factorial_train as _S73  # noqa: E402
    global S73
    S73 = _S73
    S082.S73 = _S73  # Patch S082's reference too
    S082.reset(int(args.seed))
    if args.smoke:
        args.main_words = min(int(args.main_words), 2000)
        args.kl_words = min(int(args.kl_words), 2000)
        args.binding_epochs = min(int(args.binding_epochs), 1)
        args.batch_size = min(int(args.batch_size), 8)
        args.pair_batch_size = min(int(args.pair_batch_size), 2)
        args.neutral_subsample = min(int(args.neutral_subsample), 4)
        if int(args.binding_reference_pairs_per_epoch) <= 0:
            args.binding_reference_pairs_per_epoch = 4
        args.binding_neutral_subsample = min(int(args.binding_neutral_subsample), 4) if int(args.binding_neutral_subsample) > 0 else 4
        args.device = "cpu"

    from transformers import AutoTokenizer

    endpoint = CHCK82 if args.start == "chck82_fresh" else COHERENT86
    initial_words = CHCK82_WORDS if args.start == "chck82_fresh" else COHERENT86_WORDS
    tokenizer = AutoTokenizer.from_pretrained(str(endpoint), local_files_only=True, use_fast=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, private_names, identity = S082.load_start_model(endpoint, args, device)
    optimizer = S082.optimizer_for(model, private_names, args)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))
    rng = random.Random(int(args.seed))

    ordinary_batches = S082.load_ordinary_batches(tokenizer, args)
    main_q: deque[dict[str, Any]] = deque()
    kl_q: deque[dict[str, Any]] = deque()
    w_main = 0
    w_kl = 0
    for b in ordinary_batches:
        bw = int(b["words"].sum().item())
        if w_main < int(args.main_words) and w_main + bw <= int(args.main_words):
            main_q.append(b); w_main += bw
        if w_kl < int(args.kl_words) and w_kl + bw <= int(args.kl_words):
            kl_q.append(b); w_kl += bw
    
    binding_batches_by_epoch = []
    binding_words_by_epoch = []
    binding_audits = []
    for ep in range(int(args.binding_epochs)):
        batches, charged, audit = S082.binding_epoch_batches(tokenizer, args, ep, rng)
        if args.smoke:
            batches = batches[:2]
            charged = sum(sum(int(r.get("context_words") or 0) for r in batch) for batch in batches)
            audit["smoke_truncated_batches"] = len(batches)
            audit["charged_words_this_epoch"] = charged
        binding_batches_by_epoch.append(batches)
        binding_words_by_epoch.append(charged)
        binding_audits.append(audit)

    n_binding_updates = sum(len(x) for x in binding_batches_by_epoch)
    total_updates = n_binding_updates + len(main_q) + len(kl_q)
    if total_updates <= 0:
        raise SystemExit("No updates requested")

    config = {
        "status": "PAIRED_CONTRASTIVE_CONFIG",
        "created_utc": S082.now(),
        "start": args.start,
        "start_endpoint": S082.rel(endpoint),
        "output_dir": S082.rel(out),
        "initial_consumed_words": initial_words,
        "selected_main_words": w_main,
        "selected_kl_words": w_kl,
        "binding_train_rows": S082.rel(S082.norm_path(args.binding_train_rows)),
        "binding_held_rows": S082.rel(S082.norm_path(args.binding_held_rows)),
        "binding_pairs": S082.rel(S082.norm_path(args.binding_pairs)),
        "binding_frame_schedule": str(args.binding_frame_schedule),
        "binding_epochs": int(args.binding_epochs),
        "binding_total_words": int(sum(binding_words_by_epoch)),
        "contrastive_lambda": float(args.contrastive_lambda),
        "planned_updates": total_updates,
        "identity": identity,
        "objective": "paired contrastive: CE on A+B answer tokens plus softplus(L_a - L_b) plus non-answer KL",
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "config", "contrastive_lambda": float(args.contrastive_lambda), **{k: config[k] for k in ["start", "initial_consumed_words", "selected_main_words", "selected_kl_words", "binding_total_words", "planned_updates", "binding_train_rows", "binding_frame_schedule"]}, "identity": identity}, indent=2), flush=True)

    log_path = out / "training_log.jsonl"
    losses = Counter()
    counts = Counter()
    update_i = 0
    charged_main = charged_kl = charged_binding_running = 0

    bind_items: deque[tuple[int, list[dict[str, Any]], int]] = deque()
    for ep, batches in enumerate(binding_batches_by_epoch, 1):
        for b in batches:
            bind_items.append((ep, b, sum(int(r.get("context_words") or 0) for r in b)))

    def run_main_one(logf) -> bool:
        nonlocal update_i, charged_main
        if not main_q: return False
        b = main_q.popleft()
        words = int(b["words"].sum().item())
        model.train(); S150.set_private_enabled(model, True)
        rec = S082.ordinary_update(model, tokenizer, optimizer, b, device, gen, args, update_i, total_updates, main_ce=True, neutral_only=False)
        update_i += 1; charged_main += words
        for k in ["main_loss", "neutral_loss"]:
            if k in rec and math.isfinite(float(rec[k])):
                losses[k] += float(rec[k]); counts[k] += 1
        rec.update({"update": update_i, "charged_main_words": charged_main, "charged_kl_words": charged_kl, "charged_binding_words": charged_binding_running})
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if update_i == 1 or update_i % int(args.log_every) == 0:
            print(json.dumps({"event": "train", **rec}), flush=True)
        return True

    def run_kl_one(logf) -> bool:
        nonlocal update_i, charged_kl
        if not kl_q: return False
        b = kl_q.popleft()
        words = int(b["words"].sum().item())
        model.train(); S150.set_private_enabled(model, True)
        rec = S082.ordinary_update(model, tokenizer, optimizer, b, device, gen, args, update_i, total_updates, main_ce=False, neutral_only=True)
        update_i += 1; charged_kl += words
        if "neutral_loss" in rec and math.isfinite(float(rec["neutral_loss"])):
            losses["kl_only_loss"] += float(rec["neutral_loss"]); counts["kl_only_loss"] += 1
        rec.update({"update": update_i, "charged_main_words": charged_main, "charged_kl_words": charged_kl, "charged_binding_words": charged_binding_running})
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if update_i == 1 or update_i % int(args.log_every) == 0:
            print(json.dumps({"event": "train", **rec}), flush=True)
        return True

    def run_bind_one(logf) -> bool:
        nonlocal update_i, charged_binding_running
        if not bind_items: return False
        ep, rows, bwords = bind_items.popleft()
        model.train(); S150.set_private_enabled(model, True)
        # Use the CONTRASTIVE update instead of standard
        rec = binding_update_contrastive(model, tokenizer, optimizer, rows, device, pad_id, gen, args, update_i, total_updates)
        update_i += 1; charged_binding_running += int(bwords)
        rec.update({"update": update_i, "binding_epoch": ep, "charged_main_words": charged_main, "charged_kl_words": charged_kl, "charged_binding_words": charged_binding_running})
        for k, name in [("answer_loss", "binding_answer_loss"), ("binding_nonanswer_kl_loss", "binding_nonanswer_kl_loss"), ("loss", "binding_loss"), ("contrastive_loss", "contrastive_loss")]:
            if k in rec and math.isfinite(float(rec[k])):
                losses[name] += float(rec[k]); counts[name] += 1
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if update_i == 1 or update_i % int(args.log_every) == 0:
            print(json.dumps({"event": "train", **rec}), flush=True)
        return True

    n_binding_updates_for_schedule = max(1, n_binding_updates)
    main_interval = max(1, math.ceil(n_binding_updates_for_schedule / max(1, len(main_q)))) if main_q else 10**9
    kl_interval = max(1, math.ceil(n_binding_updates_for_schedule / max(1, len(kl_q)))) if kl_q else 10**9
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        bind_seen = 0
        while bind_items or main_q or kl_q:
            did = False
            if bind_items:
                did = run_bind_one(logf)
                bind_seen += 1
                if bind_seen % main_interval == 0:
                    run_main_one(logf)
                if bind_seen % kl_interval == 0:
                    run_kl_one(logf)
            else:
                did = run_main_one(logf) or run_kl_one(logf)
            if not did:
                break

    elapsed = time.time() - t0
    # Save checkpoint
    S150.set_private_enabled(model, True)
    if not args.smoke:
        S150.save_checkpoint(model, tokenizer, out / "hf_model" / "final")
    
    # Evaluate held-out binding
    eval_result = None
    if not args.smoke:
        held_rows = S73.read_jsonl(S082.norm_path(args.binding_held_rows))
        pairs = S73.read_jsonl(S082.norm_path(args.binding_pairs))
        eval_result = S73.score_margin_eval(model, tokenizer, held_rows, pairs, device, int(args.seq_length), batch_size=32)
        S150.set_private_enabled(model, True)
    
    metrics = {
        "status": "PAIRED_CONTRASTIVE",
        "total_consumed_words": initial_words + w_main + w_kl + sum(binding_words_by_epoch),
        "tail_charged_words": w_main + w_kl + sum(binding_words_by_epoch),
        "identity": identity,
        "contrastive_lambda": float(args.contrastive_lambda),
        "mean_main_loss": losses["main_loss"] / counts["main_loss"] if counts["main_loss"] else None,
        "mean_binding_loss": losses["binding_loss"] / counts["binding_loss"] if counts["binding_loss"] else None,
        "mean_binding_answer_loss": losses["binding_answer_loss"] / counts["binding_answer_loss"] if counts["binding_answer_loss"] else None,
        "mean_contrastive_loss": losses["contrastive_loss"] / counts["contrastive_loss"] if counts["contrastive_loss"] else None,
        "mean_binding_nonanswer_kl_loss": losses["binding_nonanswer_kl_loss"] / counts["binding_nonanswer_kl_loss"] if counts["binding_nonanswer_kl_loss"] else None,
        "binding_joint_correct": eval_result.get("binding_joint_correct") if eval_result else None,
        "binding_a_correct": eval_result.get("binding_a_correct") if eval_result else None,
        "binding_b_correct": eval_result.get("binding_b_correct") if eval_result else None,
        "elapsed_sec": elapsed,
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PAIRED_CONTRASTIVE", "out": S082.rel(out), "tail_charged_words": metrics["tail_charged_words"], "total_consumed_words": metrics["total_consumed_words"], "contrastive_lambda": float(args.contrastive_lambda), "mean_contrastive_loss": metrics["mean_contrastive_loss"], "binding_joint": eval_result.get("binding_joint_correct") if eval_result else None, "elapsed_sec": elapsed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
