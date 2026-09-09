#!/usr/bin/env python3
"""Matched DeBERTa-v2 trainer for bidirectional MLM + causal next-token learning.

Scientific intervention
-----------------------
Only the learning signal changes relative to the research clean-Qwen MLM recipe.
Architecture, tokenizer, initialization, corpus order, word exposure, optimizer,
LR schedule, batch size, and checkpoint policy remain matched.

For each batch, a deterministic balanced schedule selects one objective:
  * MLM: standard 15% whole-word masking and bidirectional attention.
  * Causal next-token: uncorrupted input, lower-triangular attention, and
    logits at position t trained to predict token t+1.

The mixed schedule alternates objective assignments in blocks using an exact
integer accumulator, so --causal_fraction 0.5 gives causal batches 2,4,6,...
without stochastic objective-allocation noise. Every consumed word instance is
counted once regardless of objective. No AoA/CDI information is read or used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

# Reuse the exact data/tokenizer/masking/checkpoint primitives of the established trainer.
import masking_curriculum_trainer as base


def objective_for_step(step_zero: int, causal_fraction: float) -> str:
    """Deterministic low-discrepancy allocation; exact for rational fractions."""
    if causal_fraction <= 0.0:
        return "mlm"
    if causal_fraction >= 1.0:
        return "causal"
    before = math.floor(step_zero * causal_fraction + 1e-12)
    after = math.floor((step_zero + 1) * causal_fraction + 1e-12)
    return "causal" if after > before else "mlm"


def causal_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return [B,L,L] boolean mask: valid query/key and key_position <= query_position."""
    bsz, seq = attention_mask.shape
    valid = attention_mask.bool()
    lower = torch.ones((seq, seq), dtype=torch.bool, device=attention_mask.device).tril()
    return lower.unsqueeze(0).expand(bsz, -1, -1) & valid.unsqueeze(1) & valid.unsqueeze(2)


def causal_next_token_loss(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Compute h_t -> x_{t+1} loss without input corruption or future attention.

    DeBERTa-v2's embeddings layer multiplies embeddings by the attention mask (2D)
    to zero out padding, while the encoder uses a separate mask for self-attention.
    We call them separately: embeddings with 2D mask, encoder with 3D causal mask.
    """
    att3 = causal_attention_mask(attention_mask)  # [B, L, L] bool
    # Embeddings: uses 2D mask internally via unsqueeze(2)
    embedding_output = model.deberta.embeddings(input_ids, mask=attention_mask)
    # Encoder: uses 3D causal mask via get_attention_mask which unsqueezes dim=1
    encoder_output = model.deberta.encoder(embedding_output, att3)
    hidden_states = encoder_output[0]
    # Prediction head (shared with MLM)
    logits = model.cls(hidden_states)
    # Next-token loss: position t predicts token at t+1
    target_valid = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    labels = shifted_targets.masked_fill(~target_valid, -100)
    loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        labels.reshape(-1), ignore_index=-100,
    )
    return loss, logits, labels, att3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--example_jsonl_meta", required=True)
    p.add_argument("--example_jsonl_label", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--causal_fraction", type=float, default=0.5)
    p.add_argument("--max_word_exposure", type=int, default=100_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--warmup_fraction", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=0, help="Smoke-test cap; 0 means full run")
    p.add_argument("--save_checkpoints", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = build_args()
    if not 0.0 <= args.causal_fraction <= 1.0:
        raise ValueError("causal_fraction must be in [0,1]")
    start = time.time()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output_dir must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.example_jsonl)
    meta_path = Path(args.example_jsonl_meta)
    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    examples, file_words, file_rows, sample_rows = base.load_examples_jsonl(
        data_path, args.max_word_exposure
    )
    actual_words = sum(x.words for x in examples)
    if actual_words != args.max_word_exposure:
        raise RuntimeError(f"word exposure mismatch {actual_words} != {args.max_word_exposure}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    dataset = base.MaskedChunkDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=base.collate, num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())
    planned_steps = len(loader)
    effective_steps = min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps

    def reset_rng(s: int):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

    reset_rng(args.extra_init_seed)
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer), hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer, num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * args.ffn_mult,
        max_position_embeddings=max(512, args.seq_length + 8),
        max_relative_positions=256, position_buckets=256,
        relative_attention=True, pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = DebertaV2ForMaskedLM(cfg)
    init_probe = {
        "word_embedding_sha256": hashlib.sha256(
            model.deberta.embeddings.word_embeddings.weight.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    reset_rng(args.train_rng_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay, betas=(0.9, 0.98))
    # Full planned trajectory controls LR even in a capped smoke run.
    warmup = max(1, int(planned_steps * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, warmup, planned_steps)
    mask_gen = torch.Generator(device=device); mask_gen.manual_seed(args.train_rng_seed)
    curriculum = base.MaskingCurriculumState(
        curriculum="wwm_fixed", mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob
    )
    curriculum.initialize(len(tokenizer), planned_steps)

    data_manifest = {
        "status": "MIXED_OBJECTIVE_MANIFEST",
        "data": str(data_path), "data_sha256": sha256_file(data_path),
        "metadata": str(meta_path), "metadata_sha256": sha256_file(meta_path),
        "example_jsonl_label": args.example_jsonl_label,
        "file_words": file_words, "file_rows": file_rows,
        "selected_word_exposure": actual_words, "planned_steps": planned_steps,
        "effective_steps": effective_steps, "sample_rows": sample_rows,
        "causal_fraction": args.causal_fraction,
        "objective_policy": "deterministic_low_discrepancy_interleaving",
        "causal_definition": "uncorrupted x_<=t, triangular attention, h_t predicts x_(t+1)",
        "mlm_definition": "15pct whole-word masking, bidirectional attention",
        "word_count_policy": "each consumed row word counted once independent of objective",
        "corpus_metadata_status": meta.get("status"),
        **init_probe,
    }
    (out / "run_manifest.json").write_text(json.dumps(data_manifest, indent=2) + "\n")

    cumulative_words = 0
    losses = {"mlm": [], "causal": []}
    counts = {"mlm_batches": 0, "causal_batches": 0,
              "mlm_targets": 0, "causal_targets": 0}
    saved = []
    next_ckpt = args.checkpoint_words
    model.train()
    with (out / "training_log.jsonl").open("w") as logf:
        for step, batch in enumerate(loader, 1):
            if step > effective_steps: break
            words = int(batch.pop("words").sum())
            ids = batch["input_ids"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)
            wg = batch["word_group"].to(device, non_blocking=True)
            obj = objective_for_step(step - 1, args.causal_fraction)
            optim.zero_grad(set_to_none=True)
            if obj == "mlm":
                curriculum.current_step = step - 1
                masked, labels = base.apply_masking_curriculum(ids, am, wg, tokenizer, curriculum, mask_gen)
                outputs = model(input_ids=masked, attention_mask=am, labels=labels)
                loss = outputs.loss
                n_targets = int((labels != -100).sum())
                counts["mlm_batches"] += 1; counts["mlm_targets"] += n_targets
            else:
                loss, _, causal_labels, _ = causal_next_token_loss(model, ids, am)
                n_targets = int((causal_labels != -100).sum())
                counts["causal_batches"] += 1; counts["causal_targets"] += n_targets
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss at step {step}: {loss}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            cumulative_words += words
            losses[obj].append(float(loss.detach().cpu()))
            rec = {"step": step, "objective": obj, "loss": losses[obj][-1],
                   "targets": n_targets, "batch_words": words,
                   "cumulative_word_exposure": cumulative_words,
                   "lr": float(sched.get_last_lr()[0]), "elapsed_sec": round(time.time()-start, 1)}
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == effective_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            if args.save_checkpoints and args.max_steps == 0:
                while cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                    name = f"chck_{next_ckpt // 1_000_000}M"
                    cp = out / "hf_model" / name
                    base.save_hf_checkpoint(model, tokenizer, cp)
                    saved.append({"name": name, "target_word_exposure": next_ckpt,
                                  "actual_cumulative_word_exposure": cumulative_words})
                    next_ckpt += args.checkpoint_words

    if args.max_steps == 0:
        base.save_hf_checkpoint(model, tokenizer, out / "hf_model")
    summary = {
        "status": "MIXED_OBJECTIVE_TRAINING_COMPLETE",
        **data_manifest, "actual_word_exposure": cumulative_words,
        "actual_steps": sum(counts[k] for k in ("mlm_batches", "causal_batches")),
        **counts,
        "realized_causal_batch_fraction": counts["causal_batches"] / max(1, counts["mlm_batches"] + counts["causal_batches"]),
        "mlm_loss_first": losses["mlm"][0] if losses["mlm"] else None,
        "mlm_loss_last": losses["mlm"][-1] if losses["mlm"] else None,
        "causal_loss_first": losses["causal"][0] if losses["causal"] else None,
        "causal_loss_last": losses["causal"][-1] if losses["causal"] else None,
        "saved_checkpoints": saved,
    }
    (out / "scientific_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "done", **{k: summary[k] for k in ["actual_word_exposure", "actual_steps", "mlm_batches", "causal_batches", "realized_causal_batch_fraction", "mlm_loss_last", "causal_loss_last"]}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
