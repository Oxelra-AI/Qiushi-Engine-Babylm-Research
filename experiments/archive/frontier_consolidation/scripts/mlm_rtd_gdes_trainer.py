#!/usr/bin/env python3
"""research: GDES-wired MLM+RTD bounded training on compact-view-reinvest.

Dense RTD auxiliary objective with gradient-disentangled embedding sharing.
Tests whether all-token plausibility supervision improves learning efficiency
on the protected compact-view reinvest corpus.

Design from research probe results:
  - Trunk cosine +0.043 (all 8 layers positive) → RTD aligned with MLM
  - AUROC gap +0.191 (hard vs random) → context-requiring corruptions
  - Balanced accuracy 0.684 → learnable, not saturated
  - Embedding RTD/MLM ratio 0.094 → block embedding + rel_embeddings

Training procedure per step:
  1. Standard WWM masking → MLM forward → MLM loss backward → save GDES grads
  2. Self-corruption: sample from MLM logits at masked positions (no_grad)
  3. RTD forward on corrupted input → balanced CE → backward (accumulates)
  4. Restore GDES-blocked params to MLM-only gradients
  5. Optimizer step

GDES-blocked parameters: word_embeddings, rel_embeddings
Everything else (encoder layers, MLM head, RTD head) gets MLM + λ*RTD gradients.

λ=1.0 → RTD contributes ~24% of MLM gradient norm (from probe measurement).

Decision rule for 20M:
  - Broad improvement (Supplement/EWoK/Reading rising with BLiMP) → 100M
  - Familiar rotation (GPIQA/BLiMP up, Supplement/EWoK down) → STOP
  - Neutral but MLM loss preserved → inconclusive (late emergence possible)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

# ─── RTD Head ───────────────────────────────────────────────────────
class RTDHead(nn.Module):
    """Binary replaced-token-detection head."""
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_size, 2)
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, h):
        return self.out(self.act(self.dense(h)))


# ─── Data utilities ─────────────────────────────────────────────────
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_examples_jsonl(path, selected_words):
    examples = []
    total_words = 0
    total_rows = 0
    selected = 0
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            total_rows += 1
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"row {total_rows}: field={words} actual={actual}")
            total_words += words
            if selected < selected_words:
                if selected + words > selected_words:
                    raise RuntimeError(f"partial at row {total_rows}")
                examples.append({"text": text, "words": words,
                                 "example_id": int(obj.get("example_id", total_rows - 1))})
                selected += words
    if selected != selected_words:
        raise RuntimeError(f"selected {selected} != target {selected_words}")
    return examples, total_words, total_rows


def is_word_start(token_str):
    return token_str.startswith("Ġ") or token_str.startswith("▁")


class ChunkDataset(Dataset):
    def __init__(self, examples, tokenizer, seq_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._ws = {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"], add_special_tokens=False, truncation=True,
            max_length=self.seq_length, padding="max_length", return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        grp = torch.full_like(ids, -1)
        gid = -1
        for i in range(ids.shape[0]):
            if mask[i] == 0:
                continue
            tid = int(ids[i])
            if tid in self.special_ids:
                continue
            if tid not in self._ws:
                s = self.tokenizer.convert_ids_to_tokens(tid)
                self._ws[tid] = bool(s and is_word_start(str(s)))
            if gid < 0 or self._ws[tid] or i == 0:
                gid += 1
            grp[i] = gid
        return ids, mask, grp, ex["words"]


def collate(batch):
    ids, masks, grps, words = zip(*batch)
    return (torch.stack(ids), torch.stack(masks), torch.stack(grps),
            torch.tensor(words, dtype=torch.long))


# ─── Masking ────────────────────────────────────────────────────────
def apply_wwm(input_ids, attention_mask, word_group, mask_token_id,
              special_ids_t, mask_prob, vocab_size, gen):
    device = input_ids.device
    bsz, seq = input_ids.shape
    cand = attention_mask.bool() & ~torch.isin(input_ids, special_ids_t)
    sel = torch.zeros_like(cand)

    for b in range(bsz):
        g = word_group[b]
        vg = torch.unique(g[g >= 0])
        if vg.numel() == 0:
            continue
        chosen = vg[torch.rand(vg.numel(), generator=gen, device=device) < mask_prob]
        if chosen.numel() > 0:
            sel[b] = torch.isin(g, chosen) & cand[b]

    if sel.sum() == 0:
        flat = cand.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            sel.view(-1)[flat[0, 0]] = True

    labels = input_ids.clone()
    labels[~sel] = -100
    masked = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    masked[sel & (r < 0.8)] = mask_token_id
    rt = sel & (r >= 0.8) & (r < 0.9)
    if rt.any():
        masked[rt] = torch.randint(0, vocab_size, (int(rt.sum()),),
                                   generator=gen, device=device)
    return masked, labels, sel


# ─── Corruption ─────────────────────────────────────────────────────
def self_corrupt(mlm_logits, input_ids, select, temperature=1.0):
    """Generate corruptions from model's own MLM distribution."""
    device = input_ids.device
    bsz, seq = input_ids.shape
    corrupted = input_ids.clone()
    rtd_labels = torch.zeros(bsz, seq, dtype=torch.long, device=device)

    if select.any():
        probs = F.softmax(mlm_logits[select] / temperature, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        corrupted[select] = sampled
        rtd_labels[select] = (sampled != input_ids[select]).long()

    return corrupted, rtd_labels


# ─── Balanced CE ────────────────────────────────────────────────────
def balanced_ce(logits, labels, valid_mask):
    """Class-balanced cross-entropy for RTD."""
    y = labels[valid_mask]
    x = logits[valid_mask]
    n0 = (y == 0).sum().float().clamp_min(1)
    n1 = (y == 1).sum().float().clamp_min(1)
    n = n0 + n1
    w = torch.stack([n / (2 * n0), n / (2 * n1)]).to(x.device)
    return F.cross_entropy(x, y, weight=w)


# ─── Model build ────────────────────────────────────────────────────
def build_model(args, tokenizer):
    max_pos = max(args.max_position_embeddings, args.seq_length + 8)
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * args.ffn_mult,
        max_position_embeddings=max_pos,
        max_relative_positions=args.max_relative_positions,
        position_buckets=args.position_buckets,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return DebertaV2ForMaskedLM(cfg)


def force_portable_tokenizer_config(dst):
    tok_cfg = dst / "tokenizer_config.json"
    if tok_cfg.exists():
        cfg = json.loads(tok_cfg.read_text())
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        for k, v in [("bos_token", "<s>"), ("eos_token", "</s>"),
                     ("unk_token", "<unk>"), ("pad_token", "<pad>"),
                     ("mask_token", "<mask>")]:
            cfg.setdefault(k, v)
        tok_cfg.write_text(json.dumps(cfg, indent=2) + "\n")


def save_hf(model, tokenizer, dst):
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    force_portable_tokenizer_config(dst)


# ─── GDES parameter classification ─────────────────────────────────
def is_gdes_blocked(name):
    """Parameters whose gradients are blocked from RTD."""
    return ("word_embeddings" in name or "rel_embeddings" in name)


# ─── Main training ──────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--example_jsonl", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--max_word_exposure", type=int, default=20_000_000)
    ap.add_argument("--checkpoint_words", type=int, default=1_000_000)
    ap.add_argument("--hidden_size", type=int, default=480)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--ffn_mult", type=int, default=4)
    ap.add_argument("--max_position_embeddings", type=int, default=512)
    ap.add_argument("--max_relative_positions", type=int, default=256)
    ap.add_argument("--position_buckets", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seq_length", type=int, default=256)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--warmup_fraction", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--extra_init_seed", type=int, default=43022)
    ap.add_argument("--train_rng_seed", type=int, default=43023)
    ap.add_argument("--lr_total_steps", type=int, default=0,
                    help="If >0, schedule spans this many steps (match 100M)")
    ap.add_argument("--rtd_lambda", type=float, default=1.0)
    ap.add_argument("--rtd_temperature", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    mask_token_id = tokenizer.mask_token_id
    vocab_size = len(tokenizer)
    special_ids_t = torch.tensor(sorted(tokenizer.all_special_ids))

    # Load data
    examples, total_file_words, total_rows = load_examples_jsonl(
        Path(args.example_jsonl), args.max_word_exposure
    )
    actual_words = sum(e["words"] for e in examples)
    print(f"Loaded {len(examples)} examples, {actual_words} words from "
          f"{total_rows} total rows ({total_file_words} file words)", flush=True)

    # Dataset
    ds = ChunkDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    total_steps = len(loader)
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps

    # Build model
    def reset_rng(s):
        random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    reset_rng(args.seed)
    reset_rng(args.extra_init_seed)
    model = build_model(args, tokenizer)
    reset_rng(args.train_rng_seed)
    model.to(device)
    n_model_params = sum(p.numel() for p in model.parameters())

    # RTD head
    rtd_head = RTDHead(args.hidden_size).to(device)
    n_rtd_params = sum(p.numel() for p in rtd_head.parameters())

    print(f"Model: {n_model_params:,} params, RTD head: {n_rtd_params:,} params",
          flush=True)
    print(f"Steps: {total_steps}, schedule: {schedule_total}, "
          f"λ_RTD={args.rtd_lambda}, T={args.rtd_temperature}", flush=True)

    # Optimizer (includes RTD head)
    all_params = list(model.parameters()) + list(rtd_head.parameters())
    optim = torch.optim.AdamW(all_params, lr=args.learning_rate,
                              weight_decay=args.weight_decay, betas=(0.9, 0.98))
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, warmup, schedule_total)

    # Masking generator
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed)
    special_ids_dev = special_ids_t.to(device)

    # GDES blocked parameter names
    gdes_blocked = set()
    for name, _ in model.named_parameters():
        if is_gdes_blocked(name):
            gdes_blocked.add(name)
    print(f"GDES-blocked parameters: {sorted(gdes_blocked)}", flush=True)

    # Training
    log_path = out / "training_log.jsonl"
    cum_words = 0
    losses_mlm = []
    losses_rtd = []
    saved_ckpts = []
    next_ckpt = args.checkpoint_words

    # Manifest
    (out / "training_manifest.json").write_text(json.dumps({
        "mode": "mlm_rtd_gdes",
        "rtd_lambda": args.rtd_lambda,
        "rtd_temperature": args.rtd_temperature,
        "gdes_blocked": sorted(gdes_blocked),
        "model_params": n_model_params,
        "rtd_head_params": n_rtd_params,
        "total_steps": total_steps,
        "schedule_total": schedule_total,
        "warmup": warmup,
        "data": str(args.example_jsonl),
        "tokenizer": str(args.tokenizer_path),
        "max_word_exposure": args.max_word_exposure,
        "actual_words": actual_words,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
    }, indent=2) + "\n")

    model.train()
    rtd_head.train()

    with log_path.open("w") as logf:
        for step, (ids, amask, wgrp, words_t) in enumerate(loader, 1):
            batch_words = int(words_t.sum().item())
            ids = ids.to(device, non_blocking=True)
            amask = amask.to(device, non_blocking=True)
            wgrp = wgrp.to(device, non_blocking=True)

            # ─── research: WWM masking ───
            masked, labels, sel = apply_wwm(
                ids, amask, wgrp, mask_token_id, special_ids_dev,
                args.mask_prob, vocab_size, gen
            )

            # ─── research: MLM forward + backward ───
            optim.zero_grad(set_to_none=True)
            mlm_out = model(input_ids=masked, attention_mask=amask, labels=labels)
            mlm_loss = mlm_out.loss
            mlm_loss.backward()

            # Save GDES-blocked gradients (MLM only)
            gdes_saved = {}
            for name, p in model.named_parameters():
                if name in gdes_blocked and p.grad is not None:
                    gdes_saved[name] = p.grad.clone()

            # ─── research: Self-corruption (no grad) ───
            with torch.no_grad():
                corrupted, rtd_labels = self_corrupt(
                    mlm_out.logits, ids, sel, args.rtd_temperature
                )

            # ─── research: RTD forward + backward (accumulates with MLM) ───
            enc = model.deberta(input_ids=corrupted, attention_mask=amask)
            rtd_logits = rtd_head(enc.last_hidden_state)
            valid = amask.bool()
            rtd_loss = args.rtd_lambda * balanced_ce(rtd_logits, rtd_labels, valid)
            rtd_loss.backward()

            # ─── research: GDES restore ───
            for name, p in model.named_parameters():
                if name in gdes_blocked:
                    p.grad = gdes_saved.get(name, torch.zeros_like(p))

            # ─── research: Clip + optimize ───
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optim.step()
            sched.step()

            cum_words += batch_words
            lm = mlm_loss.item()
            lr = rtd_loss.item() / max(args.rtd_lambda, 1e-8)  # unscaled
            losses_mlm.append(lm)
            losses_rtd.append(lr)

            n_pred = int((labels != -100).sum().item())
            n_repl = int(rtd_labels.sum().item())
            n_valid = int(valid.sum().item())

            rec = {
                "step": step, "mlm_loss": round(lm, 5),
                "rtd_loss": round(lr, 5),
                "lr": float(sched.get_last_lr()[0]),
                "cum_words": cum_words,
                "n_masked": n_pred, "n_replaced": n_repl,
                "n_valid": n_valid,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)

            # Checkpoint
            while next_ckpt and cum_words >= next_ckpt and \
                    next_ckpt <= args.max_word_exposure:
                name = f"chck_{next_ckpt // 1_000_000}M"
                cp = out / "hf_model" / name
                save_hf(model, tokenizer, cp)
                # Also save RTD head state
                torch.save(rtd_head.state_dict(),
                           cp / "rtd_head.pt")
                saved_ckpts.append({
                    "name": name, "target_words": next_ckpt,
                    "actual_words": cum_words, "path": str(cp),
                    "mlm_loss": lm, "rtd_loss": lr,
                })
                print(json.dumps({"event": "checkpoint", "name": name,
                                  "cum_words": cum_words}), flush=True)
                next_ckpt += args.checkpoint_words

    # Final save
    save_hf(model, tokenizer, out / "hf_model")
    torch.save(rtd_head.state_dict(), out / "hf_model" / "rtd_head.pt")

    # Metrics
    metrics = {
        "status": "MLM_RTD_TRAINING_COMPLETE",
        "mode": "mlm_rtd_gdes",
        "backend": "mlm",  # official eval uses MLM head
        "model_family": "DebertaV2ForMaskedLM",
        "parameter_count": n_model_params,
        "rtd_head_params": n_rtd_params,
        "rtd_lambda": args.rtd_lambda,
        "rtd_temperature": args.rtd_temperature,
        "word_exposure": cum_words,
        "mlm_loss_first": losses_mlm[0] if losses_mlm else None,
        "mlm_loss_last": losses_mlm[-1] if losses_mlm else None,
        "rtd_loss_first": losses_rtd[0] if losses_rtd else None,
        "rtd_loss_last": losses_rtd[-1] if losses_rtd else None,
        "actual_steps": total_steps,
        "saved_checkpoints": saved_ckpts,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in
                      ("status", "word_exposure", "mlm_loss_first",
                       "mlm_loss_last", "rtd_loss_first", "rtd_loss_last",
                       "elapsed_sec")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
