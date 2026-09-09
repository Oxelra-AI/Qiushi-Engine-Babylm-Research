#!/usr/bin/env python3
"""research: enrichment-weighted masking calibration trainer.

A compact DeBERTa-v2 trainer that uses per-token CHILDES-minus-whole enrichment
weights for WWM masking, with a linear taper to uniform masking.

The principle: acquisition order follows cumulative credited exposure.  Frequency
governs it by default because uniform masking makes credit proportional to
frequency.  This trainer decouples the two: early in training, child-enriched
tokens are masked with higher probability, giving them more prediction credit per
occurrence.  By the taper point, masking reverts to uniform.

This is a 30M-word calibration screen, not a full trunk run.  It produces 1M-word
checkpoints compatible with the official AoA surprisal extractor.

Reuses model/tokenizer architecture from v4 but has its own data loading and
masking logic to keep the shared trainer untouched.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, hashlib, json, math, pathlib, random, time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, PreTrainedTokenizerFast,
    DebertaV2Config, DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

ROOT = _public_path('.')


# ── Data loading ──────────────────────────────────────────────────────────
def load_jsonl_examples(path: pathlib.Path, max_words: int):
    examples = []
    total_words = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            words = int(obj.get("words", len(text.split())))
            if total_words + words > max_words and total_words > 0:
                # Take partial
                take = max_words - total_words
                if take > 0:
                    text = " ".join(text.split()[:take])
                    words = take
                else:
                    break
            examples.append({"text": text, "words": words,
                             "source": str(obj.get("source", "")),
                             "example_id": obj.get("example_id", len(examples))})
            total_words += words
            if total_words >= max_words:
                break
    return examples, total_words


class ChunkDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=256):
        self.items = []
        for ex in examples:
            enc = tokenizer(ex["text"], truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt",
                            add_special_tokens=False)
            ids = enc["input_ids"].squeeze(0)
            att = enc["attention_mask"].squeeze(0)
            # Build word groups for WWM
            wg = torch.full_like(ids, -1)
            tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
            gid = 0
            for i, tok in enumerate(tokens):
                if att[i] == 0:
                    continue
                if tok.startswith("▁") or i == 0 or att[i-1] == 0:
                    gid += 1
                wg[i] = gid
            self.items.append({"input_ids": ids, "attention_mask": att,
                               "word_group": wg, "words": ex["words"]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) if isinstance(batch[0][k], torch.Tensor)
            else torch.tensor([b[k] for b in batch])
            for k in batch[0]}


# ── Enrichment-weighted WWM masking ──────────────────────────────────────
def apply_enrichment_masking(
    input_ids, attention_mask, word_group, tokenizer,
    token_probs, base_prob, gen
):
    """WWM masking with per-token enrichment probabilities."""
    device = input_ids.device
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    labels = input_ids.clone()
    select = torch.zeros_like(candidate)

    probs = token_probs[input_ids.cpu()].to(device)  # per-position probs

    for b in range(bsz):
        groups = word_group[b]
        valid_groups = torch.unique(groups[groups >= 0])
        if valid_groups.numel() == 0:
            continue
        # Mean per-token prob within each word group
        group_probs = torch.zeros(valid_groups.max().item() + 1, device=device)
        for gid in valid_groups:
            g_mask = (groups == gid) & candidate[b]
            if g_mask.any():
                group_probs[gid] = probs[b][g_mask].mean()
        gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
        chosen = valid_groups[gp < group_probs[valid_groups]]
        if chosen.numel() > 0:
            sel_b = torch.isin(groups, chosen) & candidate[b]
            select[b] = sel_b

    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True

    labels[~select] = -100
    masked_inputs = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = mask_token_id
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),),
                                 generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids
    return masked_inputs, labels


def build_enrichment_probs(enrichment_path, alpha, base_prob):
    """Build per-token masking probabilities from enrichment z-scores."""
    data = json.loads(pathlib.Path(enrichment_path).read_text(encoding="utf-8"))
    z = np.array(data["z_enrichment"], dtype=np.float64)
    vocab_size = len(z)
    # p_i = base_prob * exp(alpha * z_i) / E[exp(alpha * z_i)]
    # To normalize so mean masking rate = base_prob
    log_weights = alpha * z
    log_weights -= log_weights.max()  # numerical stability
    weights = np.exp(log_weights)
    # Normalize over all tokens (uniform prior for inactive tokens)
    weights_mean = weights.mean()
    probs = base_prob * weights / weights_mean
    # Clamp to [0.02, 0.60] for training stability
    probs = np.clip(probs, 0.02, 0.60)
    return torch.tensor(probs, dtype=torch.float32), vocab_size


# ── Model builder (matches v4 architecture) ──────────────────────────────
def build_model(tokenizer, hidden_size=480, n_layer=8, n_head=8, ffn_mult=4,
                max_position_embeddings=264):
    vocab_size = len(tokenizer)
    config = DebertaV2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        intermediate_size=hidden_size * ffn_mult,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=0,
        position_biased_input=False,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
    )
    model = DebertaV2ForMaskedLM(config)
    return model


def save_checkpoint(model, tokenizer, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", required=True, help="JSONL training stream")
    parser.add_argument("--enrichment-weights", required=True, help="Token enrichment JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-word-exposure", type=int, default=30_000_000)
    parser.add_argument("--checkpoint-words", type=int, default=1_000_000)
    parser.add_argument("--tokenizer-path", default=str(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_1M')))
    parser.add_argument("--seed", type=int, default=43, help="Match v4 init seed")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--lr-total-steps", type=int, default=2529,
                        help="LR schedule total steps; match v4 for comparable dynamics")
    parser.add_argument("--base-prob", type=float, default=0.15)
    parser.add_argument("--alpha-start", type=float, default=1.5,
                        help="Initial enrichment weight strength (z-score units)")
    parser.add_argument("--taper-frac", type=float, default=0.67,
                        help="Fraction of training at which enrichment fully tapers to uniform")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # Data
    examples, actual_words = load_jsonl_examples(pathlib.Path(args.stream), args.max_word_exposure)
    print(json.dumps({"event": "data_loaded", "examples": len(examples), "words": actual_words}), flush=True)

    dataset = ChunkDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0,
                        pin_memory=(args.device == "cuda"))
    total_steps = len(loader)

    # Model (same init as v4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(tokenizer)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(json.dumps({"event": "model_built", "params": param_count,
                      "device": str(device)}), flush=True)

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay, betas=(0.9, 0.98))
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup,
                                            num_training_steps=schedule_total)

    # Enrichment weights
    enr_probs_full, vocab_size = build_enrichment_probs(
        args.enrichment_weights, args.alpha_start, args.base_prob)
    uniform_probs = torch.full((vocab_size,), args.base_prob, dtype=torch.float32)
    taper_step = int(total_steps * args.taper_frac)
    print(json.dumps({"event": "enrichment_loaded", "vocab_size": vocab_size,
                      "alpha_start": args.alpha_start, "taper_step": taper_step,
                      "total_steps": total_steps,
                      "enr_prob_mean": float(enr_probs_full.mean()),
                      "enr_prob_std": float(enr_probs_full.std()),
                      "enr_prob_min": float(enr_probs_full.min()),
                      "enr_prob_max": float(enr_probs_full.max())}), flush=True)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    # Training loop
    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    next_ckpt = args.checkpoint_words
    saved_checkpoints = []
    source_words: Counter[str] = Counter()

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            # Taper enrichment weights
            if step <= taper_step:
                frac = (step - 1) / max(1, taper_step)
                # Linear interpolation from enriched to uniform
                token_probs = (1 - frac) * enr_probs_full + frac * uniform_probs
            else:
                token_probs = uniform_probs

            masked_inputs, labels = apply_enrichment_masking(
                input_ids, attention_mask, word_group, tokenizer,
                token_probs, args.base_prob, gen
            )

            optim.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            loss = out_model.loss
            if loss is None:
                raise RuntimeError("model returned no loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            cumulative_words += words
            loss_float = float(loss.detach().cpu())
            n_pred = int((labels != -100).sum().item())
            n_cand = int(attention_mask.bool().sum().item())
            eff_rate = n_pred / max(1, n_cand)

            rec = {"step": step, "loss": loss_float, "lr": float(sched.get_last_lr()[0]),
                   "batch_words": words, "cumulative_word_exposure": cumulative_words,
                   "masked_tokens": n_pred, "effective_mask_rate": round(eff_rate, 4),
                   "enrichment_active": step <= taper_step,
                   "elapsed_sec": round(time.time() - t0, 1)}
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)

            # Checkpoint saving
            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                if next_ckpt % 1_000_000 == 0:
                    name = f"chck_{next_ckpt // 1_000_000}M"
                else:
                    name = f"chck_{next_ckpt}w"
                cp = out / "hf_model" / name
                save_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({"name": name, "target_word_exposure": next_ckpt,
                                          "actual_cumulative_word_exposure": cumulative_words,
                                          "path": str(cp)})
                print(json.dumps({"event": "checkpoint_saved", "name": name,
                                  "cum_words": cumulative_words}), flush=True)
                next_ckpt += args.checkpoint_words

    # Final save
    save_checkpoint(model, tokenizer, out / "hf_model")

    # Manifest
    for ex in examples:
        source_words[ex["source"]] = source_words.get(ex["source"], 0) + ex["words"]
    (out / "example_order_manifest.json").write_text(json.dumps({
        "seed": args.seed, "selected_for_training_words": actual_words,
        "num_consumed_examples": len(examples),
        "masking_curriculum": "enrichment_weighted_wwm",
        "mask_prob_start": args.base_prob, "mask_prob_end": args.base_prob,
        "alpha_start": args.alpha_start, "taper_frac": args.taper_frac,
        "source_words_consumed": dict(source_words),
        "data_source_type": "example_jsonl",
        "example_jsonl": args.stream,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "variant": "enrichment_weighted_wwm",
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "word_exposure": cumulative_words,
        "actual_training_steps": total_steps,
        "alpha_start": args.alpha_start,
        "taper_frac": args.taper_frac,
        "taper_step": taper_step,
        "seed": args.seed,
        "lr_total_steps": args.lr_total_steps,
        "saved_checkpoints": saved_checkpoints,
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"event": "done", "param_count": param_count,
                      "word_exposure": cumulative_words,
                      "checkpoints": len(saved_checkpoints),
                      "elapsed_sec": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
