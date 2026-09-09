#!/usr/bin/env python3
"""research — RecGPT-style causal training on official BabyLM Strict-Small corpus.

Faithfully implements the public RecGPT recipe:
- RecGPTForCausalLM (recursive_depth=16, hidden=768, embed=192, FFN=12288)
- Aurora/Muon optimizer for 2D matrices, Adam for embeddings/norms
- NextLat auxiliary loss (hidden-state prediction, nl_mult=2.0)
- Token-based LR scheduling (no warmup, 20% linear cooldown)
- 10 epochs, sequence_len=512, microbatch_tok=32768
- BF16 autocast, torch.compile for FlexAttention
- Checkpoints at chck_1M through chck_9M

Uses:
- Official corpus: data/custom_corpus/official_only_10M.txt
- Tokenizer: data/recgpt_official_tokenizer
- Model code: data/recgpt_local/patched_model/modeling_recgpt.py
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, random, shutil, sys, time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from transformers import PreTrainedTokenizerFast

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
sys.path.insert(0, str(ROOT / "data/recgpt_local/patched_model"))
from modeling_recgpt import RecGPTConfig, RecGPTForCausalLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════════
# Aurora/Muon Optimizer (simplified from RecGPT optimizer.py)
# ═══════════════════════════════════════════════════════════════════════════════

def newton_schulz_5(G: torch.Tensor, steps: int = 12, eps: float = 1e-7) -> torch.Tensor:
    """Approximate polar decomposition via Newton-Schulz iteration.

    Compute the square product on the smaller dimension. RecGPT FFN matrices
    include tall shapes such as 12288 x 768; using X @ X.T there would allocate
    a huge 12288 x 12288 matrix. Transpose tall matrices before iteration.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    original_shape = G.shape
    X = G.reshape(G.shape[0], -1) if G.ndim > 2 else G
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    X = X.bfloat16()
    X /= (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.reshape(original_shape).to(G.dtype)


class AuroraOptimizer(torch.optim.Optimizer):
    """Aurora/Muon + Adam optimizer with cautious weight decay."""

    def __init__(self, params, defaults=None):
        if defaults is None:
            defaults = {"lr": 0.02, "use_muon": True, "weight_decay": 0.1,
                        "momentum": 0.95, "betas": (0.9, 0.997)}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            use_muon = group.get("use_muon", False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if use_muon:
                    # Muon/Aurora update with momentum + polar decomposition
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    mom = group.get("momentum", 0.95)
                    buf.mul_(mom).add_(grad)
                    # Nesterov
                    update = grad + mom * buf
                    # Polar decomposition for 2D matrices
                    if update.ndim >= 2:
                        original_shape = update.shape
                        if update.ndim > 2:
                            update = update.view(update.shape[0], -1)
                        update = newton_schulz_5(update)
                        update = update.view(original_shape)
                    # Cautious weight decay
                    mask = (update * grad > 0).float()
                    ratio = mask.mean()
                    if ratio > 0:
                        update = update * mask / ratio
                    p.add_(update, alpha=-lr)
                    # Weight decay
                    if wd > 0:
                        decay_mask = (p * grad > 0).float()
                        p.add_(p * decay_mask, alpha=-lr * wd)
                else:
                    # Adam update
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    beta1, beta2 = group.get("betas", (0.9, 0.997))
                    state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias1 = 1 - beta1 ** state["step"]
                    bias2 = 1 - beta2 ** state["step"]
                    m_hat = state["exp_avg"] / bias1
                    v_hat = state["exp_avg_sq"] / bias2
                    update = m_hat / (v_hat.sqrt() + 1e-8)
                    # Cautious weight decay
                    if wd > 0:
                        mask = (update * grad > 0).float()
                        ratio = mask.mean()
                        if ratio > 0:
                            update = update * mask / ratio
                        p.add_(p * mask, alpha=-lr * wd)
                    p.add_(update, alpha=-lr)


# ═══════════════════════════════════════════════════════════════════════════════
# NextLat Auxiliary Model
# ═══════════════════════════════════════════════════════════════════════════════

class NextLatAux(nn.Module):
    """Predicts next-token hidden state from current hidden + next token embedding."""

    def __init__(self, hidden_size: int, intermediate: int = 5120):
        super().__init__()
        self.input_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, intermediate), nn.ReLU(),
                         nn.Linear(intermediate, hidden_size))
            for _ in range(3)
        ])
        self.norm = nn.RMSNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, next_embed: torch.Tensor,
                target_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """hidden, next_embed, target_hidden: (B, T, H). mask: (B, T) bool."""
        x = self.input_proj(torch.cat([hidden, next_embed], dim=-1))
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm(x)
        loss = F.smooth_l1_loss(x[mask], target_hidden[mask], reduction="none").sum(-1).mean()
        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_tokenize(corpus_path: pathlib.Path, tok_path: pathlib.Path, seq_len: int):
    """Tokenize corpus into packed sequences with segment boundaries."""
    tok = PreTrainedTokenizerFast.from_pretrained(str(tok_path))
    text = corpus_path.read_text(encoding="utf-8")
    # Split into documents (paragraphs separated by double newline, or single lines)
    docs = [d.strip() for d in text.split("\n") if d.strip()] if "\n" in text else [text]
    if len(docs) <= 1:
        # Single-line file: split by periods for document boundaries
        docs = [s.strip() + "." for s in text.split(". ") if s.strip()]

    all_ids = []
    all_segments = []
    current_ids = []
    current_segs = []
    seg_id = 0

    for doc in docs:
        enc = tok(doc, add_special_tokens=False)["input_ids"]
        if not enc:
            continue
        if len(current_ids) + len(enc) > seq_len:
            # Pack current buffer
            if current_ids:
                # Pad to seq_len
                pad_len = seq_len - len(current_ids)
                all_ids.append(current_ids + [tok.pad_token_id] * pad_len)
                all_segments.append(current_segs + [-1] * pad_len)
            current_ids = []
            current_segs = []
            seg_id = 0
        current_ids.extend(enc)
        current_segs.extend([seg_id] * len(enc))
        seg_id += 1

    if current_ids:
        pad_len = seq_len - len(current_ids)
        all_ids.append(current_ids + [tok.pad_token_id] * pad_len)
        all_segments.append(current_segs + [-1] * pad_len)

    return (torch.tensor(all_ids, dtype=torch.long),
            torch.tensor(all_segments, dtype=torch.long),
            tok)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def token_to_lr(tokens_seen: int, total_tokens: int, base_lr: float,
                cooldown_ratio: float = 0.2, min_lr: float = 0.0) -> float:
    cooldown_start = total_tokens * (1 - cooldown_ratio)
    if tokens_seen >= cooldown_start:
        progress = min(1.0, (tokens_seen - cooldown_start) / (total_tokens * cooldown_ratio))
        return base_lr - progress * (base_lr - min_lr)
    return base_lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(ROOT / "data/custom_corpus/official_only_10M.txt"))
    p.add_argument("--tokenizer", default=str(ROOT / "data/recgpt_official_tokenizer"))
    p.add_argument("--output_dir", default=str(ROOT / "training/runs/recgpt_official_train"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_tokens", type=int, default=32768)
    p.add_argument("--lr_embed", type=float, default=0.005)
    p.add_argument("--lr_block", type=float, default=0.02)
    p.add_argument("--nl_mult", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--max_steps", type=int, default=-1, help="Stop after this many optimizer steps for dry-run/debug; -1 means full run")
    p.add_argument("--dry_run_no_save", action="store_true", help="Do not save checkpoints/final model; write only dry_run_result.json")
    args = p.parse_args()
    if args.no_compile:
        args.compile = False

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading and tokenizing corpus...")
    all_ids, all_segments, tok = load_and_tokenize(
        pathlib.Path(args.corpus), pathlib.Path(args.tokenizer), args.seq_len)
    n_sequences = all_ids.shape[0]
    tokens_per_epoch = int((all_segments >= 0).sum().item())
    total_tokens = tokens_per_epoch * args.epochs
    batch_size = args.batch_tokens // args.seq_len
    n_batches_per_epoch = math.ceil(n_sequences / batch_size)
    print(f"Sequences: {n_sequences}, Tokens/epoch: {tokens_per_epoch:,}, "
          f"Total tokens: {total_tokens:,}, Batch size: {batch_size}, "
          f"Batches/epoch: {n_batches_per_epoch}")

    # Model
    print("Initializing model...")
    config = RecGPTConfig(
        vocab_size=tok.vocab_size,
        hidden_size=768,
        embedding_size=192,
        head_dim=64,
        intermediate_size=12288,
        recursive_depth=16,
        max_position_embeddings=args.seq_len,
        pad_token_id=tok.pad_token_id,
        tie_word_embeddings=False,
    )
    model = RecGPTForCausalLM(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # NextLat auxiliary
    nl_model = NextLatAux(config.hidden_size, intermediate=5120).to(DEVICE)

    # Optimizer
    adam_params = []
    muon_params = []
    nl_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or any(k in name for k in ["embed_tokens", "lm_head", "norm", "e_to_h", "h_to_e"]):
            adam_params.append(param)
        else:
            muon_params.append(param)
    for param in nl_model.parameters():
        if param.requires_grad:
            nl_params.append(param)

    optimizer = AuroraOptimizer([
        {"params": adam_params, "lr": args.lr_embed, "use_muon": False,
         "weight_decay": 0.005, "betas": (0.9, 0.997)},
        {"params": muon_params, "lr": args.lr_block, "use_muon": True,
         "weight_decay": 0.1, "momentum": 0.95},
        {"params": nl_params, "lr": 0.02, "use_muon": True,
         "weight_decay": 0.1, "momentum": 0.95},
    ])

    # Compile
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Checkpoint schedule (chck_1M through chck_9M in first epoch)
    words_per_token = 10_000_000 / tokens_per_epoch  # ~0.79
    chck_schedule = [(m, int(m * 1_000_000 / words_per_token)) for m in range(1, 10)]

    # Training loop
    print(f"\nStarting training: {args.epochs} epochs, lr_block={args.lr_block}, "
          f"lr_embed={args.lr_embed}, nl_mult={args.nl_mult}")
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    tokens_seen = 0
    t0 = time.time()
    log_entries = []

    for epoch in range(args.epochs):
        perm = torch.randperm(n_sequences)
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch_idx in range(n_batches_per_epoch):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_sequences)
            indices = perm[start:end]
            input_ids = all_ids[indices].to(DEVICE)
            segment_ids = all_segments[indices].to(DEVICE)
            attention_mask = (segment_ids >= 0).long()

            # Count real tokens
            real_tokens = attention_mask.sum().item()
            tokens_seen += real_tokens

            # LR scheduling
            for group, base_lr in zip(optimizer.param_groups,
                                       [args.lr_embed, args.lr_block, 0.02]):
                group["lr"] = token_to_lr(tokens_seen, total_tokens, base_lr)

            # Forward. Mask padding labels and keep NextLat inside real, same-document transitions.
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                               segment_ids=segment_ids, labels=labels,
                               output_hidden_states=True)
                ce_loss = outputs.loss

                hidden = outputs.hidden_states[-1]  # (B, T, H)
                unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
                next_embed = unwrapped.e_to_h(unwrapped.embed_tokens(input_ids[:, 1:]))
                nl_mask = (attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool() &
                           (segment_ids[:, :-1] == segment_ids[:, 1:]) & (segment_ids[:, :-1] >= 0))
                nl_loss = nl_model(hidden[:, :-1], next_embed, hidden[:, 1:].detach(), nl_mask)
                loss = ce_loss + args.nl_mult * nl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(nl_model.parameters()), 2.0)
            optimizer.step()

            epoch_loss_sum += ce_loss.item()
            epoch_steps += 1

            # Logging
            if epoch_steps % 50 == 0:
                elapsed = time.time() - t0
                entry = {"epoch": epoch, "step": epoch_steps, "tokens_seen": tokens_seen,
                         "ce_loss": round(ce_loss.item(), 4), "nl_loss": round(nl_loss.item(), 4),
                         "lr": optimizer.param_groups[1]["lr"], "elapsed_sec": round(elapsed, 1)}
                log_entries.append(entry)
                print(f"  ep{epoch} step{epoch_steps}: ce={ce_loss.item():.4f} "
                      f"nl={nl_loss.item():.4f} lr={entry['lr']:.5f} "
                      f"tok={tokens_seen:,} ({elapsed:.0f}s)")

            # Checkpoints (first epoch only for strict-small schedule)
            if (not args.dry_run_no_save) and epoch == 0:
                for m, tok_mark in chck_schedule:
                    if tokens_seen >= tok_mark and not (out_dir / f"chck_{m}M").exists():
                        save_checkpoint(model, tok, config, out_dir / f"chck_{m}M")
                        print(f"  → Saved chck_{m}M at {tokens_seen:,} tokens")

            global_step = epoch * n_batches_per_epoch + epoch_steps
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"Reached max_steps={args.max_steps}; stopping early.")
                dry_payload = {
                    "status": "dry_run_ok",
                    "tokens_seen": tokens_seen,
                    "last_ce_loss": float(ce_loss.item()),
                    "last_nl_loss": float(nl_loss.item()),
                    "last_total_loss": float(loss.item()),
                    "n_params": n_params,
                    "batch_size": batch_size,
                    "seq_len": args.seq_len,
                    "compile": bool(args.compile),
                    "dry_run_no_save": bool(args.dry_run_no_save),
                }
                (out_dir / "dry_run_result.json").write_text(json.dumps(dry_payload, indent=2) + "\n")
                if args.dry_run_no_save:
                    return

        avg_loss = epoch_loss_sum / max(1, epoch_steps)
        print(f"Epoch {epoch} complete: avg_ce={avg_loss:.4f}, tokens_seen={tokens_seen:,}")
        if not args.dry_run_no_save:
            save_checkpoint(model, tok, config, out_dir / f"epoch_{epoch}")

    # Final save
    if not args.dry_run_no_save:
        save_checkpoint(model, tok, config, out_dir / "final")
    (out_dir / "training_log.json").write_text(json.dumps(log_entries, indent=2) + "\n")
    print(f"\nTraining complete. Total time: {time.time()-t0:.0f}s")
    print(f"Final model: {out_dir / 'final'}")


def save_checkpoint(model, tok, config, out_dir: pathlib.Path):
    """Save HF-compatible checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Unwrap compiled model
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m.config.auto_map = {
        "AutoConfig": "modeling_recgpt.RecGPTConfig",
        "AutoModelForCausalLM": "modeling_recgpt.RecGPTForCausalLM",
    }
    m.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    # Copy modeling file
    src = ROOT / "data/recgpt_local/patched_model/modeling_recgpt.py"
    shutil.copy2(src, out_dir / "modeling_recgpt.py")


if __name__ == "__main__":
    main()
