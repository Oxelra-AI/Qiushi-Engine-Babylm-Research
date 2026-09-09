#!/usr/bin/env python3
"""Phase 2 SOTA trainer: DeBERTa-v2 12×384 with LAMB optimizer and 40k tokenizer.

This is a targeted modification of the INITIAL_MODEL_STUDIES fullcycle trainer that adds:
1. LAMB optimizer (with layer-wise trust ratio)
2. Custom intermediate_size (for 1280 instead of hidden*ffn_mult)
3. share_att_key configuration
4. position_biased_input control

Usage:
  python phase2_trainer.py \
    --example_jsonl <path_to_100M_jsonl> \
    --output_dir <output_path> \
    --tokenizer_path <40k_tokenizer_dir> \
    --tokenizer_label sp40k \
    --model_type deberta_v2 \
    --n_layer 12 --hidden_size 384 --n_head 12 \
    --intermediate_size 1280 \
    --share_att_key true \
    --position_biased_input false \
    --optimizer lamb --learning_rate 0.007 \
    --batch_size 256 --max_seq_length 256 \
    --seq_len_schedule "0.0:64,0.3:128,0.6:256" \
    --mask_mode wwm --mask_prob 0.15 \
    --max_word_exposure 100000000 \
    --warmup_fraction 0.05 --weight_decay 0.01 \
    --seed 43 --num_workers 0 \
    --checkpoint_words 10000000 --log_every 50

Key differences from INITIAL_MODEL_STUDIES fullcycle trainer:
- Adds --optimizer {adam,lamb} flag
- Adds --intermediate_size (overrides hidden_size * ffn_mult)
- Adds --share_att_key and --position_biased_input to DeBERTa config
- LAMB implementation with proper exclusions for bias/LayerNorm
"""
from __future__ import annotations
import sys
import importlib
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Tuple

import torch
from torch.optim import Optimizer

# ========== LAMB Optimizer (inlined for portability) ==========

class LAMB(Optimizer):
    """LAMB: Layer-wise Adaptive Moments for Batch training (You et al., ICLR 2020)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.01, exclude_from_layer_adaptation=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._exclude = exclude_from_layer_adaptation or set()
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                update = (exp_avg / bc1) / ((exp_avg_sq / bc2).sqrt() + eps)
                if wd > 0:
                    update = update.add(p, alpha=wd)
                # Layer-wise trust ratio
                pname = getattr(p, "_lamb_name", "")
                if pname in self._exclude:
                    trust = 1.0
                else:
                    w_norm = p.norm(2).item()
                    u_norm = update.norm(2).item()
                    trust = (w_norm / u_norm) if (w_norm > 0 and u_norm > 0) else 1.0
                p.add_(update, alpha=-lr * trust)
        return loss


def make_lamb_optimizer(model, lr=0.007, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6):
    """Create LAMB with standard bias/LN exclusions."""
    decay_params, no_decay_params = [], []
    excluded = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param._lamb_name = name
        if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            no_decay_params.append(param)
            excluded.add(name)
        else:
            decay_params.append(param)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return LAMB(groups, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay, exclude_from_layer_adaptation=excluded)


# ========== Patch: Modified build_model with extra config options ==========

def build_model_phase2(args, tokenizer):
    """Build DeBERTa-v2 with leader-matching config options."""
    from transformers import DebertaV2Config, DebertaV2ForMaskedLM

    max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
    
    # Use intermediate_size directly if provided, else hidden * ffn_mult
    intermediate = args.intermediate_size if args.intermediate_size > 0 else args.hidden_size * args.ffn_mult
    
    pos_att_type = [x.strip() for x in args.deberta_pos_att_type.split(",") if x.strip()]
    
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=intermediate,
        max_position_embeddings=max_pos,
        max_relative_positions=args.max_relative_positions,
        position_buckets=args.position_buckets,
        relative_attention=(args.deberta_relative_attention.lower() == "true"),
        pos_att_type=pos_att_type,
        # Phase 2 additions:
        share_att_key=(args.share_att_key.lower() == "true"),
        position_biased_input=(args.position_biased_input.lower() == "true"),
        # Standard settings:
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return DebertaV2ForMaskedLM(cfg)


# ========== Main: Import and patch the INITIAL_MODEL_STUDIES trainer ==========

def main():
    """Run the INITIAL_MODEL_STUDIES trainer with Phase 2 patches applied."""
    # Add INITIAL_MODEL_STUDIES trainer path for importing
    trainer_path = Path("experiments/archive/initial_model_studies/training/scripts")
    sys.path.insert(0, str(trainer_path))
    
    # Import the original trainer module
    spec = importlib.util.spec_from_file_location(
        "babylm_masked_train_fullcycle",
        trainer_path / "babylm_masked_train_fullcycle.py"
    )
    trainer_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer_mod)
    
    # Patch build_model
    original_build_model = trainer_mod.build_model
    
    # Patch build_args to add new arguments
    original_build_args = trainer_mod.build_args
    
    def patched_build_args():
        args = original_build_args()
        # Add Phase 2 arguments (may already exist in namespace from original parse)
        if not hasattr(args, 'optimizer'):
            args.optimizer = 'adam'
        if not hasattr(args, 'intermediate_size'):
            args.intermediate_size = 0
        if not hasattr(args, 'share_att_key'):
            args.share_att_key = 'false'
        if not hasattr(args, 'position_biased_input'):
            args.position_biased_input = 'true'
        return args
    
    # Parse our extended args
    p = argparse.ArgumentParser()
    # Copy all original args + add new ones
    p.add_argument("--optimizer", choices=["adam", "lamb"], default="adam")
    p.add_argument("--intermediate_size", type=int, default=0,
                   help="Override intermediate_size (0 = use hidden*ffn_mult)")
    p.add_argument("--share_att_key", default="false")
    p.add_argument("--position_biased_input", default="true")
    
    # Parse known args to get Phase 2 additions, pass rest to original trainer
    phase2_args, remaining = p.parse_known_args()
    
    # Now run original trainer with patched components
    # Re-parse with full original arg set
    sys.argv = [sys.argv[0]] + remaining
    args = original_build_args()
    
    # Merge Phase 2 args
    args.optimizer = phase2_args.optimizer
    args.intermediate_size = phase2_args.intermediate_size
    args.share_att_key = phase2_args.share_att_key
    args.position_biased_input = phase2_args.position_biased_input
    
    # Monkey-patch the module's build_model
    trainer_mod.build_model = lambda a, t: build_model_phase2(a, t) if a.model_type == "deberta_v2" else original_build_model(a, t)
    
    # Monkey-patch optimizer creation in the main function
    # We need to modify how the trainer creates the optimizer
    # The cleanest way: override torch.optim.AdamW temporarily
    if args.optimizer == "lamb":
        original_adamw = torch.optim.AdamW
        
        class LAMBWrapper:
            """Intercept AdamW creation and return LAMB instead."""
            def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98), **kwargs):
                # The trainer passes model.parameters() directly
                # We need the model to get named_parameters for exclusions
                # Workaround: collect params, find their names from the model later
                self._params_list = list(params)
                self._lr = lr
                self._wd = weight_decay
                self._betas = betas
                
            def _as_lamb(self, model):
                return make_lamb_optimizer(model, lr=self._lr, weight_decay=self._wd, betas=self._betas)
        
        # Actually, the monkey-patch approach for the optimizer is too fragile.
        # Instead, let me just note that Phase 2 needs a slightly modified main().
        # The approach below directly runs the training loop with our modifications.
        pass
    
    print(json.dumps({
        "event": "phase2_config",
        "optimizer": args.optimizer,
        "intermediate_size": args.intermediate_size,
        "share_att_key": args.share_att_key,
        "position_biased_input": args.position_biased_input,
        "n_layer": args.n_layer,
        "hidden_size": args.hidden_size,
        "n_head": args.n_head,
    }), flush=True)
    
    # For clean implementation, run the training directly
    # (avoiding fragile monkey-patching of the original main)
    run_phase2_training(args, trainer_mod)


def run_phase2_training(args, trainer_mod):
    """Direct Phase 2 training loop using the INITIAL_MODEL_STUDIES infrastructure."""
    import random
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    
    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = trainer_mod.make_portable_tokenizer(args.tokenizer_path)
    
    # Load data (reuse INITIAL_MODEL_STUDIES data loading)
    pool_words = args.example_pool_words
    selected_words = args.max_word_exposure
    
    if args.example_jsonl:
        examples, _, _, sample_rows = trainer_mod.load_examples_jsonl(
            Path(args.example_jsonl), selected_words)
        actual_words = sum(ex.words for ex in examples)
        pool_words = actual_words  # All from JSONL
    else:
        raise RuntimeError("Phase 2 requires --example_jsonl")
    
    if actual_words != selected_words:
        raise RuntimeError(f"word mismatch {actual_words} vs {selected_words}")
    
    # Build dataset
    trainer_mod.reset_all_rng(args.seed)
    dataset = trainer_mod.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    
    def collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                       collate_fn=collate, num_workers=args.num_workers,
                       pin_memory=torch.cuda.is_available())
    
    # Build model (Phase 2 config)
    model = build_model_phase2(args, tokenizer)
    param_count = sum(p.numel() for p in model.parameters())
    input_embedding_params = model.get_input_embeddings().weight.numel()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create optimizer
    if args.optimizer == "lamb":
        optim = make_lamb_optimizer(model, lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.98))
    
    total_steps = len(loader)
    schedule_total = args.lr_total_steps if args.lr_total_steps and args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup,
                                            num_training_steps=schedule_total)
    
    # Sequence length schedule
    seq_schedule = []
    if args.seq_len_schedule:
        for part in args.seq_len_schedule.split(","):
            t, L = part.split(":")
            seq_schedule.append((float(t), int(L)))
        seq_schedule.sort()
    
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    
    print(json.dumps({
        "event": "phase2_model_ready",
        "param_count": param_count,
        "embedding_params": input_embedding_params,
        "vocab_size": len(tokenizer),
        "total_steps": total_steps,
        "optimizer": args.optimizer,
        "lr": args.learning_rate,
        "warmup_steps": warmup,
    }), flush=True)
    
    # Training loop
    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values = []
    saved_checkpoints = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words and args.checkpoint_words > 0 else None
    
    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            
            frac = (step - 1) / max(1, schedule_total)
            cur_len = trainer_mod.seq_length_for_progress(frac, seq_schedule, args.seq_length) if seq_schedule else args.seq_length
            cur_len = min(cur_len, args.max_seq_length)
            
            input_ids = input_ids[:, :cur_len].contiguous()
            attention_mask = attention_mask[:, :cur_len].contiguous()
            word_group = word_group[:, :cur_len].contiguous()
            
            masked_inputs, labels = trainer_mod.apply_masking(
                input_ids, attention_mask, word_group, tokenizer,
                args.mask_mode, args.mask_prob, gen)
            
            optim.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            loss = out_model.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            
            cumulative_words += words
            loss_float = float(loss.detach().cpu())
            loss_values.append(loss_float)
            n_pred = int((labels != -100).sum().item())
            
            rec = {"step": step, "loss": loss_float, "lr": float(sched.get_last_lr()[0]),
                   "batch_words": words, "cumulative_word_exposure": cumulative_words,
                   "seq_len": cur_len, "masked_tokens": n_pred, "elapsed_sec": time.time() - start_time}
            logf.write(json.dumps(rec) + "\n")
            
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            
            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                name = f"chck_{next_ckpt // 1_000_000}M" if next_ckpt >= 1_000_000 else "chck_1M"
                cp = out / "hf_model" / name
                trainer_mod.save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({"name": name, "target_word_exposure": next_ckpt,
                                          "actual_cumulative_word_exposure": cumulative_words, "path": str(cp)})
                print(json.dumps({"event": "checkpoint_saved", "name": name, "cum_words": cumulative_words}), flush=True)
                next_ckpt += args.checkpoint_words
    
    # Save final model
    trainer_mod.save_hf_checkpoint(model, tokenizer, out / "hf_model")
    
    # Save metrics
    metrics = {
        "variant": f"masked_{args.mask_mode}",
        "backend": "mlm",
        "model_family": model.__class__.__name__,
        "model_type": args.model_type,
        "parameter_count": param_count,
        "embedding_parameter_count": input_embedding_params,
        "non_embedding_parameter_count": param_count - input_embedding_params,
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_path": args.tokenizer_path,
        "word_exposure": cumulative_words,
        "example_pool_words_actual": pool_words,
        "selected_for_training_words": actual_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": total_steps,
        "mask_mode": args.mask_mode,
        "mask_prob": args.mask_prob,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "share_att_key": args.share_att_key,
        "position_biased_input": args.position_biased_input,
        "intermediate_size": args.intermediate_size if args.intermediate_size > 0 else args.hidden_size * args.ffn_mult,
        "saved_checkpoints": saved_checkpoints,
        "seq_len_schedule": args.seq_len_schedule,
    }
    
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(json.dumps({"event": "done", "param_count": param_count,
                      "loss_first": metrics["loss_first"], "loss_last": metrics["loss_last"],
                      "word_exposure": cumulative_words,
                      "checkpoints": [c["name"] for c in saved_checkpoints]}), flush=True)


if __name__ == "__main__":
    main()
