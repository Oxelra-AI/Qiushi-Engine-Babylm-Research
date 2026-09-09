#!/usr/bin/env python3
"""research: pair-aware causal GPT2 trainer with identifiable directional schedules.

Scientific purpose
------------------
The research forward+forward vs forward+reverse contrast cannot identify reciprocal
learning, because a gain could come from reverse prediction alone. This trainer
therefore exposes four two-epoch schedules over the exact same pair-token multiset:

  ff: source -> side, source -> side
  rr: side   -> source, side   -> source
  fr: source -> side, side   -> source
  rf: side   -> source, source -> side

The reciprocal interaction should be read against both one-direction baselines, e.g.
0.5*(fr + rf) - 0.5*(ff + rr), and then contrasted between compact views and
copy-matched extractive controls. The trainer keeps pair boundaries intact and logs
segment-B copied/noncopied losses, so later analysis can separate literal-copy from
noncopy target learning.

Implementation guard
--------------------
Use the same model seed, train seed, shuffling, optimizer schedule, and first-epoch
direction for paired arms (ff vs fr; rr vs rf). Their first-epoch logs and saved
weights should match before any later divergence is interpreted.

Boundary guard
--------------
No source/side tokens are silently truncated from pair rows. Pair rows are tokenized
as source+side or side+source and split into within-pair chunks of length <= seq_len;
chunks never mix different pairs. This preserves the pair-token multiset while
making long-pair boundary effects explicit in the sequence counts and copied vs
noncopied target accounting. Filler is still ordinary concatenated/chunked text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import shutil
import time
from typing import Dict, List, Tuple

# Determinism knobs must be set before torch initializes CUDA/cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_dir_safetensors(hf_dir: pathlib.Path) -> str | None:
    return sha256_file(hf_dir / "model.safetensors") or sha256_file(hf_dir / "pytorch_model.bin")


def build_model(vocab_size: int, seq_len: int, hidden: int, layers: int, heads: int, seed: int) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
        n_embd=hidden,
        n_layer=layers,
        n_head=heads,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=0,
        eos_token_id=0,
    )
    return GPT2LMHeadModel(cfg)


def direction_for(schedule: str, epoch: int) -> str:
    if schedule == "ff":
        return "forward"
    if schedule == "rr":
        return "reverse"
    if schedule == "fr":
        return "forward" if epoch == 0 else "reverse"
    if schedule == "rf":
        return "reverse" if epoch == 0 else "forward"
    raise ValueError(f"Unknown schedule {schedule!r}")


def pre_tokenize_pairs(pair_jsonl: pathlib.Path, tokenizer, max_pairs: int = 0) -> Tuple[List[dict], int, Dict[str, float]]:
    rows: List[dict] = []
    charged_words = 0
    max_src = max_side = max_total = 0
    n_total_gt_256 = 0
    n_src_gt_256 = 0
    n_side_gt_256 = 0
    with pair_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src_tok = tokenizer.encode(r["source_text"], add_special_tokens=False)
            side_tok = tokenizer.encode(r["side_text"], add_special_tokens=False)
            total_len = len(src_tok) + len(side_tok)
            max_src = max(max_src, len(src_tok))
            max_side = max(max_side, len(side_tok))
            max_total = max(max_total, total_len)
            n_total_gt_256 += int(total_len > 256)
            n_src_gt_256 += int(len(src_tok) > 256)
            n_side_gt_256 += int(len(side_tok) > 256)
            rows.append(
                {
                    "pair_id": r["pair_id"],
                    "src_tok": src_tok,
                    "side_tok": side_tok,
                    "source_words": int(r["source_words"]),
                    "side_words": int(r["side_words"]),
                    "data_type": r.get("data_type", "unknown"),
                }
            )
            charged_words += int(r["source_words"]) + int(r["side_words"])
            if max_pairs > 0 and len(rows) >= max_pairs:
                break
    stats = {
        "max_source_tokens": max_src,
        "max_side_tokens": max_side,
        "max_total_pair_tokens": max_total,
        "n_source_gt_seq_len_256": n_src_gt_256,
        "n_side_gt_seq_len_256": n_side_gt_256,
        "n_total_pair_gt_seq_len_256": n_total_gt_256,
    }
    return rows, charged_words, stats


def pre_tokenize_filler(filler_jsonl: pathlib.Path, tokenizer, seq_len: int, max_words: int = 0) -> Tuple[List[List[int]], int, int]:
    all_tok: List[int] = []
    charged_words = 0
    with filler_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            all_tok.extend(tokenizer.encode(r["text"], add_special_tokens=False))
            charged_words += int(r.get("words", len(r["text"].split())))
            if max_words > 0 and charged_words >= max_words:
                break
    full = len(all_tok) // seq_len
    chunks = [all_tok[i * seq_len : (i + 1) * seq_len] for i in range(full)]
    return chunks, charged_words, len(all_tok) - full * seq_len


def make_pair_seqs(pair: dict, epoch: int, schedule: str, seq_len: int, pad_id: int) -> List[dict]:
    direction = direction_for(schedule, epoch)
    if direction == "forward":
        first, second = pair["src_tok"], pair["side_tok"]
    else:
        first, second = pair["side_tok"], pair["src_tok"]
    first_len = len(first)
    combined = first + second
    first_set = set(first)
    side_mask_full = [0] * first_len + [1] * len(second)
    copied_full = [0] * first_len + [1 if t in first_set else 0 for t in second]
    out: List[dict] = []
    for start in range(0, len(combined), seq_len):
        toks = combined[start : start + seq_len]
        sm = side_mask_full[start : start + seq_len]
        cp = copied_full[start : start + seq_len]
        real = len(toks)
        if real < seq_len:
            toks = toks + [pad_id] * (seq_len - real)
            sm = sm + [0] * (seq_len - real)
            cp = cp + [0] * (seq_len - real)
        out.append(
            {
                "tokens": toks,
                "real": real,
                "side_mask": sm,
                "copied": cp,
                "is_pair": True,
                "pair_id": pair["pair_id"],
                "direction": direction,
                "chunk_start": start,
            }
        )
    return out


def make_filler_seq(chunk: List[int], seq_len: int, pad_id: int) -> dict:
    real = min(len(chunk), seq_len)
    tokens = chunk[:seq_len] + [pad_id] * max(seq_len - len(chunk), 0)
    return {"tokens": tokens, "real": real, "side_mask": [0] * seq_len, "copied": [0] * seq_len, "is_pair": False}


def loss_with_stats(logits: torch.Tensor, tok_t: torch.Tensor, reals: List[int], copieds: List[List[int]], side_masks: List[List[int]]) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = tok_t.device
    bsz, tseq, vocab = logits.shape
    pred = logits[:, :-1].contiguous().view(-1, vocab)
    targ = tok_t[:, 1:].contiguous().view(-1)
    per = F.cross_entropy(pred, targ, reduction="none").view(bsz, tseq - 1)
    t2 = tseq - 1

    active_mask = torch.zeros(bsz, t2, device=device)
    for i in range(bsz):
        active = max(min(reals[i] - 1, t2), 0)
        active_mask[i, :active] = 1.0
    # Loss position k predicts token k+1; copy and segment-B labels are properties of target token k+1.
    copied_mask = torch.tensor([c[1 : t2 + 1] for c in copieds], device=device, dtype=torch.float32)
    side_mask = torch.tensor([m[1 : t2 + 1] for m in side_masks], device=device, dtype=torch.float32)

    total_nll_tensor = per * active_mask
    total_nll = total_nll_tensor.sum()
    total_targets = active_mask.sum().clamp(min=1.0)
    seg_copied_mask = active_mask * side_mask * copied_mask
    seg_noncopied_mask = active_mask * side_mask * (1.0 - copied_mask)
    copied_nll = (per * seg_copied_mask).sum()
    noncopied_nll = (per * seg_noncopied_mask).sum()
    copied_targets = seg_copied_mask.sum()
    noncopied_targets = seg_noncopied_mask.sum()
    loss = total_nll / total_targets
    stats = {
        "total_nll": float(total_nll.detach().cpu()),
        "total_targets": float(total_targets.detach().cpu()),
        "copied_nll": float(copied_nll.detach().cpu()),
        "copied_targets": float(copied_targets.detach().cpu()),
        "noncopied_nll": float(noncopied_nll.detach().cpu()),
        "noncopied_targets": float(noncopied_targets.detach().cpu()),
    }
    return loss, stats


def merge_stats(acc: Dict[str, float], add: Dict[str, float]) -> None:
    for k, v in add.items():
        acc[k] = acc.get(k, 0.0) + float(v)


def summarise_stats(acc: Dict[str, float]) -> Dict[str, float | None]:
    total_targets = max(acc.get("total_targets", 0.0), 1.0)
    copied_targets = acc.get("copied_targets", 0.0)
    noncopied_targets = acc.get("noncopied_targets", 0.0)
    return {
        "loss": acc.get("total_nll", 0.0) / total_targets,
        "copied_loss": (acc.get("copied_nll", 0.0) / copied_targets) if copied_targets > 0 else None,
        "noncopied_loss": (acc.get("noncopied_nll", 0.0) / noncopied_targets) if noncopied_targets > 0 else None,
        "n_targets": total_targets,
        "n_copied": copied_targets,
        "n_noncopied": noncopied_targets,
    }


def save_hf(model, tokenizer, out_dir: pathlib.Path) -> str | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return sha256_dir_safetensors(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_pair_jsonl", required=True)
    ap.add_argument("--filler_jsonl", required=True)
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--schedule", choices=["ff", "rr", "fr", "rf"], required=True)
    ap.add_argument("--n_epochs", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=480)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--model_seed", type=int, default=43)
    ap.add_argument("--train_seed", type=int, default=43022)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_pairs", type=int, default=0, help="Smoke-only: use first N pairs")
    ap.add_argument("--max_filler_words", type=int, default=0, help="Smoke-only: stop tokenizing filler after this many words")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_epochs", default="1,2")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--deterministic", action="store_true", default=True)
    args = ap.parse_args()

    if args.n_epochs != 2:
        raise SystemExit("This research schedule screen is defined for exactly two epochs.")

    out = pathlib.Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"Output dir {out} already exists and is not empty; pass --overwrite for smoke/restarts.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if args.deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    print(f"Tokenizer: vocab={tokenizer.vocab_size} pad={pad_id}", flush=True)

    t0 = time.time()
    model = build_model(tokenizer.vocab_size, args.seq_len, args.hidden, args.layers, args.heads, args.model_seed).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: GPT2 {args.layers}x{args.hidden}, heads={args.heads}, params={n_params:,}, device={args.device}", flush=True)

    # After initializing the model, set the training RNG. This makes paired first-epoch arms
    # compare the same dropout/shuffle stream when their first-epoch direction is the same.
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)

    pair_path = pathlib.Path(args.arm_pair_jsonl)
    filler_path = pathlib.Path(args.filler_jsonl)
    print("Tokenizing pairs...", flush=True)
    pairs, pair_words, pair_token_stats = pre_tokenize_pairs(pair_path, tokenizer, args.max_pairs)
    print(f"  pairs={len(pairs):,}; charged_pair_words={pair_words:,}; token_stats={pair_token_stats}", flush=True)
    print("Tokenizing filler...", flush=True)
    filler_chunks, filler_words, filler_tail_tokens = pre_tokenize_filler(filler_path, tokenizer, args.seq_len, args.max_filler_words)
    print(f"  filler_chunks={len(filler_chunks):,}; charged_filler_words={filler_words:,}; dropped_tail_tokens={filler_tail_tokens}; tokenization_elapsed={time.time()-t0:.1f}s", flush=True)

    # Build one epoch once for sequence-count statistics. It is rebuilt per epoch with the correct direction.
    ep0_pair_seqs = []
    for p in pairs:
        ep0_pair_seqs.extend(make_pair_seqs(p, 0, args.schedule, args.seq_len, pad_id))
    max_pair_chunks = 0
    if pairs:
        # Number of chunks is invariant to direction because it depends on total token length.
        max_pair_chunks = max(math.ceil((len(p["src_tok"]) + len(p["side_tok"])) / args.seq_len) for p in pairs)
    pair_chunks_per_epoch = len(ep0_pair_seqs)

    eff_batch = args.batch_size * args.grad_accum
    seqs_per_epoch = pair_chunks_per_epoch + len(filler_chunks)
    steps_per_epoch = math.ceil(seqs_per_epoch / eff_batch)
    total_steps = args.n_epochs * steps_per_epoch
    warmup = int(total_steps * args.warmup_frac)
    save_epochs = {int(x) for x in args.save_epochs.split(",") if x.strip()}

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    cfg = {
        "status": "CAUSAL_TRAINING_CONFIG",
        "arm_pair_jsonl": str(pair_path),
        "arm_pair_sha256": sha256_file(pair_path),
        "filler_jsonl": str(filler_path),
        "filler_sha256": sha256_file(filler_path),
        "tokenizer_dir": args.tokenizer_dir,
        "schedule": args.schedule,
        "schedule_directions": [direction_for(args.schedule, e) for e in range(args.n_epochs)],
        "n_epochs": args.n_epochs,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": eff_batch,
        "lr": args.lr,
        "warmup_frac": args.warmup_frac,
        "warmup_steps": warmup,
        "weight_decay": args.weight_decay,
        "model_seed": args.model_seed,
        "train_seed": args.train_seed,
        "max_pairs": args.max_pairs,
        "max_filler_words": args.max_filler_words,
        "charged_pair_words_per_epoch": pair_words,
        "charged_filler_words_per_epoch": filler_words,
        "charged_words_per_epoch": pair_words + filler_words,
        "charged_total_words": args.n_epochs * (pair_words + filler_words),
        "pairs": len(pairs),
        "pair_chunks_per_epoch": pair_chunks_per_epoch,
        "max_pair_chunks_per_row": max_pair_chunks,
        "pair_token_stats": pair_token_stats,
        "filler_chunks": len(filler_chunks),
        "filler_tail_tokens_dropped": filler_tail_tokens,
        "seqs_per_epoch": seqs_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "model": {"params": n_params, "hidden": args.hidden, "layers": args.layers, "heads": args.heads, "vocab_size": tokenizer.vocab_size},
        "deterministic": bool(args.deterministic),
        "device": args.device,
    }
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps({k: cfg[k] for k in ["schedule", "schedule_directions", "charged_words_per_epoch", "pair_chunks_per_epoch", "steps_per_epoch", "total_steps"]}, indent=2), flush=True)

    log_f = (out / "training_log.jsonl").open("w", encoding="utf-8")
    rng = random.Random(args.train_seed)
    global_step = 0
    model.train()
    checkpoint_hashes: Dict[str, str | None] = {}
    epoch_summaries = []

    for epoch in range(args.n_epochs):
        direction = direction_for(args.schedule, epoch)
        epoch_start = time.time()
        ep_seqs: List[dict] = []
        for p in pairs:
            ep_seqs.extend(make_pair_seqs(p, epoch, args.schedule, args.seq_len, pad_id))
        ep_seqs += [make_filler_seq(c, args.seq_len, pad_id) for c in filler_chunks]
        rng.shuffle(ep_seqs)
        print(f"\nEpoch {epoch+1}/{args.n_epochs}: direction={direction}, seqs={len(ep_seqs):,}, pair_chunks={len(ep_seqs)-len(filler_chunks):,}", flush=True)
        accum = 0
        acc_stats: Dict[str, float] = {}
        epoch_stats: Dict[str, float] = {}
        epoch_first_step = global_step + 1

        for batch_i in range(0, len(ep_seqs), args.batch_size):
            batch = ep_seqs[batch_i : batch_i + args.batch_size]
            tok_t = torch.tensor([s["tokens"] for s in batch], dtype=torch.long, device=args.device)
            reals = [s["real"] for s in batch]
            copieds = [s["copied"] for s in batch]
            side_masks = [s["side_mask"] for s in batch]

            logits = model(tok_t).logits
            loss, stats = loss_with_stats(logits, tok_t, reals, copieds, side_masks)
            (loss / args.grad_accum).backward()
            merge_stats(acc_stats, stats)
            merge_stats(epoch_stats, stats)
            accum += 1

            if accum % args.grad_accum == 0 or batch_i + args.batch_size >= len(ep_seqs):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "direction": direction,
                    "lr": sched.get_last_lr()[0],
                    **summarise_stats(acc_stats),
                    "elapsed_sec": time.time() - t0,
                }
                if global_step <= 5 or global_step % args.log_every == 0 or batch_i + args.batch_size >= len(ep_seqs):
                    log_f.write(json.dumps(entry) + "\n")
                    log_f.flush()
                if global_step <= 3 or global_step % max(args.log_every, 1) == 0:
                    print(
                        f"  step {global_step}/{total_steps} loss={entry['loss']:.4f} "
                        f"copy={entry['copied_loss']} noncopy={entry['noncopied_loss']} lr={entry['lr']:.6g}",
                        flush=True,
                    )
                acc_stats = {}

        ep_num = epoch + 1
        ep_summary = {
            "epoch": ep_num,
            "direction": direction,
            "first_step": epoch_first_step,
            "last_step": global_step,
            **summarise_stats(epoch_stats),
            "elapsed_sec": time.time() - epoch_start,
        }
        epoch_summaries.append(ep_summary)
        if ep_num in save_epochs:
            ck = out / "hf_model" / f"epoch_{ep_num}"
            checkpoint_hashes[f"epoch_{ep_num}"] = save_hf(model, tokenizer, ck)
            print(f"  saved {ck} sha={checkpoint_hashes[f'epoch_{ep_num}']}", flush=True)

    final_dir = out / "hf_model" / "final"
    checkpoint_hashes["final"] = save_hf(model, tokenizer, final_dir)
    elapsed = time.time() - t0
    manifest = {
        "status": "CAUSAL_TRAINING_COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": elapsed,
        "steps": global_step,
        "epochs": args.n_epochs,
        "schedule": args.schedule,
        "schedule_directions": [direction_for(args.schedule, e) for e in range(args.n_epochs)],
        "epoch_summaries": epoch_summaries,
        "checkpoint_hashes": checkpoint_hashes,
        "final": str(final_dir),
        "config_path": str(out / "train_config.json"),
    }
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_f.close()
    print(f"\nDone: {global_step} steps in {elapsed:.1f}s")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
