#!/usr/bin/env python3
"""research: Pair-aware causal GPT2 trainer for the prediction-geometry 2×2 experiment.

Key design:
 - Pair rows: tokenize segment_a and segment_b separately, concat, pad to seq_len
 - Filler rows: tokenize, concatenate, chunk to seq_len
 - No cross-row mixing for pairs (clean topology guarantee)
 - Causal LM loss, masked on padding tokens
 - Tracks copied-vs-noncopied loss on pair rows
 - GPT2 8×480, 16k BPE neutral tokenizer (same as the causal transfer reference)
 - Fixed model init seed across all arms for controlled comparison
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, json, math, os, pathlib, random, sys, time, hashlib
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

ROOT = _public_path('experiments/archive/representation_and_objectives')  # 


def build_model(vocab_size, seq_len, n_layer=8, n_embd=480, n_head=8, seed=43):
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size, n_positions=seq_len,
        n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        activation_function="gelu_new",
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=0, eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    return model


def load_pair_sequences(pair_jsonl, tokenizer, seq_len, pad_id):
    """Build fixed-length sequences from pair JSONL. Each pair is one sequence."""
    seqs = []
    for line in open(pair_jsonl):
        row = json.loads(line)
        a_tok = tokenizer.encode(row["segment_a"], add_special_tokens=False)
        b_tok = tokenizer.encode(row["segment_b"], add_special_tokens=False)
        combined = a_tok + b_tok
        boundary = len(a_tok)
        if len(combined) > seq_len:
            combined = combined[:seq_len]
            boundary = min(boundary, seq_len)
        real_len = len(combined)
        pad_len = seq_len - real_len
        tokens = combined + [pad_id] * pad_len
        # copied mask: for tokens in segment_b, 1 if token_id also in segment_a
        a_set = set(a_tok)
        copied = [0] * seq_len
        for i in range(boundary, real_len):
            copied[i] = 1 if tokens[i] in a_set else 0
        seqs.append(dict(tokens=tokens, real_len=real_len, boundary=boundary,
                         copied=copied, is_pair=True))
    return seqs


def load_filler_sequences(filler_jsonl, tokenizer, seq_len, max_words=None):
    """Tokenize filler rows, concatenate, chunk to seq_len."""
    all_tokens = []
    total_words = 0
    for line in open(filler_jsonl):
        row = json.loads(line)
        toks = tokenizer.encode(row["text"], add_special_tokens=False)
        all_tokens.extend(toks)
        total_words += row.get("words", len(row["text"].split()))
        if max_words and total_words >= max_words:
            break
    seqs = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        chunk = all_tokens[i:i + seq_len]
        seqs.append(dict(tokens=chunk, real_len=seq_len, boundary=-1,
                         copied=[0]*seq_len, is_pair=False))
    return seqs, total_words


def compute_loss_breakdown(logits, tokens_t, real_lens, copied_masks, boundaries):
    """Compute causal LM loss with padding masking and copied/noncopied breakdown."""
    B, T, V = logits.shape
    # Shift: predict next token
    pred = logits[:, :-1, :].contiguous()
    target = tokens_t[:, 1:].contiguous()
    T2 = T - 1
    per_tok = F.cross_entropy(pred.view(-1, V), target.view(-1), reduction='none').view(B, T2)
    # Build loss mask (1 for real tokens, 0 for padding)
    mask = torch.zeros(B, T2, device=logits.device)
    for i in range(B):
        mask[i, :min(real_lens[i]-1, T2)] = 1.0
    masked_loss = per_tok * mask
    total_loss = masked_loss.sum() / mask.sum().clamp(min=1)
    # Copied / noncopied / pair / filler breakdown
    copied_m = torch.tensor([c[1:T2+1] for c in copied_masks], device=logits.device, dtype=torch.float)
    pair_m = torch.tensor([[1.0 if b >= 0 else 0.0]*T2 for b in boundaries], device=logits.device)
    filler_m = 1.0 - pair_m
    # segment_b mask: within pair rows, tokens after boundary
    segb_m = torch.zeros(B, T2, device=logits.device)
    for i in range(B):
        if boundaries[i] >= 0:
            start = max(boundaries[i] - 1, 0)  # shifted by 1 for prediction
            end = min(real_lens[i] - 1, T2)
            segb_m[i, start:end] = 1.0
    copied_loss = (masked_loss * segb_m * copied_m).sum()
    copied_count = (mask * segb_m * copied_m).sum().clamp(min=1)
    noncopied_loss = (masked_loss * segb_m * (1 - copied_m)).sum()
    noncopied_count = (mask * segb_m * (1 - copied_m)).sum().clamp(min=1)
    pair_loss = (masked_loss * pair_m).sum()
    pair_count = (mask * pair_m).sum().clamp(min=1)
    return total_loss, {
        "loss": total_loss.item(),
        "pair_loss": (pair_loss / pair_count).item(),
        "filler_loss": ((masked_loss * filler_m * mask).sum() / (mask * filler_m).sum().clamp(min=1)).item(),
        "segb_copied_loss": (copied_loss / copied_count).item(),
        "segb_noncopied_loss": (noncopied_loss / noncopied_count).item(),
        "n_tokens": mask.sum().item(),
        "n_copied": copied_count.item(),
        "n_noncopied": noncopied_count.item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_pair_jsonl", required=True, help="Arm pair JSONL from scaffold")
    ap.add_argument("--filler_jsonl", required=True, help="Filler rows JSONL")
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--total_words", type=int, default=20_000_000, help="Total training budget (words)")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--model_seed", type=int, default=43, help="Model init seed (shared across arms)")
    ap.add_argument("--train_seed", type=int, default=43022, help="Training shuffle seed")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--checkpoint_words", type=str, default="5000000,10000000,15000000,20000000",
                    help="Comma-separated word counts for checkpoints")
    ap.add_argument("--save_hf_at", type=str, default="", help="Word counts for HF model saves")
    ap.add_argument("--max_filler_words", type=int, default=0, help="Limit filler (0=all)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: vocab={vocab_size}, pad_id={pad_id}")

    # Build model
    model = build_model(vocab_size, args.seq_len, seed=args.model_seed)
    model = model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: GPT2 8×480, {n_params:,} parameters")

    # Load pair sequences
    pair_seqs = load_pair_sequences(args.arm_pair_jsonl, tokenizer, args.seq_len, pad_id)
    print(f"Pair sequences: {len(pair_seqs)}")

    # Compute pair words (approximate from scaffold)
    pair_words_per_epoch = sum(s["real_len"] for s in pair_seqs) / 1.3  # rough token-to-word ratio

    # Calculate epochs and filler budget
    filler_words_per_epoch = max(0, args.total_words / 2 - pair_words_per_epoch)  # rough split
    n_epochs = max(1, round(args.total_words / (pair_words_per_epoch + filler_words_per_epoch + 1)))
    # More precise: load filler to fill budget
    filler_limit = args.max_filler_words if args.max_filler_words > 0 else None
    filler_seqs, filler_words_loaded = load_filler_sequences(
        args.filler_jsonl, tokenizer, args.seq_len, max_words=filler_limit)
    print(f"Filler sequences: {len(filler_seqs)} ({filler_words_loaded:,} words)")

    # Total sequences per epoch
    all_seqs = pair_seqs + filler_seqs
    total_tokens_per_epoch = sum(s["real_len"] for s in all_seqs)
    est_words_per_epoch = total_tokens_per_epoch / 1.3
    n_epochs = max(1, round(args.total_words / est_words_per_epoch))
    print(f"Estimated words/epoch: {est_words_per_epoch:,.0f}, epochs: {n_epochs}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    effective_batch = args.batch_size * args.grad_accum
    total_steps = n_epochs * math.ceil(len(all_seqs) / effective_batch)
    warmup_steps = int(total_steps * args.warmup_frac)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    print(f"Training: {total_steps} steps, warmup={warmup_steps}, eff_batch={effective_batch}")

    # Checkpoint schedule
    ckpt_words = [int(x) for x in args.checkpoint_words.split(",") if x.strip()]
    hf_save_words = [int(x) for x in args.save_hf_at.split(",") if x.strip()] if args.save_hf_at else ckpt_words

    # Training loop
    train_rng = random.Random(args.train_seed)
    log_path = out_dir / "training_log.jsonl"
    log_f = open(log_path, "w")
    words_seen = 0
    global_step = 0
    ckpt_idx = 0
    hf_idx = 0
    t0 = time.time()

    config_manifest = {
        "arm_pair_jsonl": args.arm_pair_jsonl,
        "filler_jsonl": args.filler_jsonl,
        "tokenizer_dir": args.tokenizer_dir,
        "total_words": args.total_words,
        "n_epochs": n_epochs,
        "n_pair_seqs": len(pair_seqs),
        "n_filler_seqs": len(filler_seqs),
        "total_seqs_per_epoch": len(all_seqs),
        "total_steps": total_steps,
        "model_seed": args.model_seed,
        "train_seed": args.train_seed,
        "n_params": n_params,
        "device": args.device,
    }
    (out_dir / "train_config.json").write_text(json.dumps(config_manifest, indent=2))

    model.train()
    for epoch in range(n_epochs):
        epoch_seqs = list(all_seqs)
        train_rng.shuffle(epoch_seqs)
        accum_loss = 0
        accum_steps = 0
        breakdown_accum = {}

        for batch_start in range(0, len(epoch_seqs), args.batch_size):
            batch = epoch_seqs[batch_start:batch_start + args.batch_size]
            tokens_t = torch.tensor([s["tokens"] for s in batch], device=args.device)
            real_lens = [s["real_len"] for s in batch]
            copied_masks = [s["copied"] for s in batch]
            boundaries = [s["boundary"] for s in batch]

            outputs = model(tokens_t)
            loss, breakdown = compute_loss_breakdown(
                outputs.logits, tokens_t, real_lens, copied_masks, boundaries)
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            accum_loss += loss.item()
            accum_steps += 1
            # Rough word tracking
            batch_words = sum(r / 1.3 for r in real_lens)
            words_seen += batch_words

            # Accumulate breakdown
            for k, v in breakdown.items():
                if k not in breakdown_accum:
                    breakdown_accum[k] = []
                breakdown_accum[k].append(v)

            if accum_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0 or global_step <= 5:
                    avg_bd = {k: sum(v)/len(v) for k, v in breakdown_accum.items() if v}
                    entry = {"step": global_step, "epoch": epoch, "words_seen": int(words_seen),
                             "lr": scheduler.get_last_lr()[0],
                             "loss": accum_loss / args.grad_accum, **avg_bd}
                    log_f.write(json.dumps(entry) + "\n")
                    log_f.flush()
                    if global_step % 200 == 0:
                        elapsed = time.time() - t0
                        print(f"  step {global_step}/{total_steps}  epoch {epoch}  "
                              f"words {int(words_seen):,}  loss {entry['loss']:.4f}  "
                              f"copied {avg_bd.get('segb_copied_loss',0):.4f}  "
                              f"noncopied {avg_bd.get('segb_noncopied_loss',0):.4f}  "
                              f"lr {entry['lr']:.6f}  {elapsed:.0f}s")
                    breakdown_accum = {}
                    accum_loss = 0

            # Checkpoint
            while ckpt_idx < len(ckpt_words) and words_seen >= ckpt_words[ckpt_idx]:
                ckpt_name = f"chck_{ckpt_words[ckpt_idx] // 1_000_000}M"
                ckpt_dir = out_dir / "checkpoints" / ckpt_name
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / "model.pt")
                print(f"  >> Checkpoint {ckpt_name} at {int(words_seen):,} words")
                ckpt_idx += 1

            # Save HF model for evaluation
            while hf_idx < len(hf_save_words) and words_seen >= hf_save_words[hf_idx]:
                hf_name = f"chck_{hf_save_words[hf_idx] // 1_000_000}M"
                hf_dir = out_dir / "hf_model" / hf_name
                hf_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(hf_dir)
                tokenizer.save_pretrained(hf_dir)
                print(f"  >> HF model saved: {hf_name}")
                hf_idx += 1

    # Final save
    final_dir = out_dir / "hf_model" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    elapsed = time.time() - t0
    summary = {
        "status": "CAUSAL_TOPOLOGY_TRAINING_COMPLETE",
        "total_words_seen": int(words_seen),
        "total_steps": global_step,
        "epochs": n_epochs,
        "elapsed_sec": elapsed,
        "final_model": str(final_dir),
        "checkpoints_saved": ckpt_idx,
    }
    (out_dir / "training_manifest.json").write_text(json.dumps(summary, indent=2))
    log_f.close()
    print(f"\nDone: {global_step} steps, {int(words_seen):,} words, {elapsed:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
