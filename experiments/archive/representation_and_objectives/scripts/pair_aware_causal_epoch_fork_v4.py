#!/usr/bin/env python3
"""research v4: one-epoch forkable pair-aware causal trainer.

This script fixes the research v3 smoke problem: separate ff/fr processes had nearly
identical first-epoch logs but different first-epoch model hashes. For the scientific
screen we need the paired forward prefix (ff vs fr) and reverse prefix (rr vs rf) to
be exactly identical before the second-direction intervention.

The v4 protocol is therefore forked:
  prefix --direction forward  -> shared F1 state
     continue --direction forward -> FF
     continue --direction reverse -> FR
  prefix --direction reverse  -> shared R1 state
     continue --direction reverse -> RR
     continue --direction forward -> RF

The same optimizer, scheduler, Python RNG, Torch CPU/CUDA RNG, model weights, and
first-epoch shuffled sequence stream are saved in the prefix state. Continuations
load that exact state and differ only in epoch-2 direction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import random
import shutil
import time
from typing import Any, Dict, List, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoTokenizer

V3_PATH = pathlib.Path("experiments/archive/representation_and_objectives/scripts/pair_aware_causal_trainer_v3.py")
spec = importlib.util.spec_from_file_location("v3", V3_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import {V3_PATH}")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dir_model_hash(hf_dir: pathlib.Path) -> str | None:
    return sha256_file(hf_dir / "model.safetensors") or sha256_file(hf_dir / "pytorch_model.bin")


def schedule_for_direction(direction: str) -> str:
    return "ff" if direction == "forward" else "rr"


def set_train_rng(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_sequences(pairs: List[dict], filler_chunks: List[List[int]], epoch_direction: str, seq_len: int, pad_id: int) -> List[dict]:
    # Use v3's boundary-preserving pair chunker; schedule ff/rr gives explicit direction.
    schedule = schedule_for_direction(epoch_direction)
    seqs: List[dict] = []
    for p in pairs:
        seqs.extend(v3.make_pair_seqs(p, 0, schedule, seq_len, pad_id))
    seqs += [v3.make_filler_seq(c, seq_len, pad_id) for c in filler_chunks]
    return seqs


def make_optimizer_and_scheduler(model, lr: float, weight_decay: float, total_steps: int, warmup: int):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


def train_one_epoch(
    *,
    model,
    opt,
    sched,
    rng: random.Random,
    pairs: List[dict],
    filler_chunks: List[List[int]],
    epoch_index_zero_based: int,
    epoch_direction: str,
    args,
    pad_id: int,
    global_step: int,
    t0: float,
    log_f,
) -> Tuple[int, Dict[str, Any]]:
    epoch_start = time.time()
    ep_seqs = build_sequences(pairs, filler_chunks, epoch_direction, args.seq_len, pad_id)
    rng.shuffle(ep_seqs)
    print(
        f"Epoch {epoch_index_zero_based+1}: direction={epoch_direction}, seqs={len(ep_seqs):,}, "
        f"pair_chunks={len(ep_seqs)-len(filler_chunks):,}",
        flush=True,
    )
    accum = 0
    acc_stats: Dict[str, float] = {}
    epoch_stats: Dict[str, float] = {}
    first_step = global_step + 1
    model.train()
    for batch_i in range(0, len(ep_seqs), args.batch_size):
        batch = ep_seqs[batch_i : batch_i + args.batch_size]
        tok_t = torch.tensor([s["tokens"] for s in batch], dtype=torch.long, device=args.device)
        reals = [s["real"] for s in batch]
        copieds = [s["copied"] for s in batch]
        side_masks = [s["side_mask"] for s in batch]
        logits = model(tok_t).logits
        loss, stats = v3.loss_with_stats(logits, tok_t, reals, copieds, side_masks)
        (loss / args.grad_accum).backward()
        v3.merge_stats(acc_stats, stats)
        v3.merge_stats(epoch_stats, stats)
        accum += 1
        if accum % args.grad_accum == 0 or batch_i + args.batch_size >= len(ep_seqs):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            entry = {
                "step": global_step,
                "epoch": epoch_index_zero_based + 1,
                "direction": epoch_direction,
                "lr": sched.get_last_lr()[0],
                **v3.summarise_stats(acc_stats),
                "elapsed_sec": time.time() - t0,
            }
            if global_step <= 5 or global_step % args.log_every == 0 or batch_i + args.batch_size >= len(ep_seqs):
                log_f.write(json.dumps(entry) + "\n")
                log_f.flush()
            if global_step <= 3 or global_step % max(args.log_every, 1) == 0:
                print(
                    f"  step {global_step} loss={entry['loss']:.4f} copy={entry['copied_loss']} "
                    f"noncopy={entry['noncopied_loss']} lr={entry['lr']:.6g}",
                    flush=True,
                )
            acc_stats = {}
    summary = {
        "epoch": epoch_index_zero_based + 1,
        "direction": epoch_direction,
        "first_step": first_step,
        "last_step": global_step,
        **v3.summarise_stats(epoch_stats),
        "elapsed_sec": time.time() - epoch_start,
    }
    return global_step, summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prefix", "continue"], required=True)
    ap.add_argument("--direction", choices=["forward", "reverse"], required=True)
    ap.add_argument("--arm_pair_jsonl", required=True)
    ap.add_argument("--filler_jsonl", required=True)
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--resume_state", default="")
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
    ap.add_argument("--max_pairs", type=int, default=0)
    ap.add_argument("--max_filler_words", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--deterministic", action="store_true", default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = pathlib.Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"Output dir {out} already exists and is not empty; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "continue" and not args.resume_state:
        raise SystemExit("--resume_state is required for phase=continue")

    if args.deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    t0 = time.time()

    print("Tokenizing pairs/filler...", flush=True)
    pairs, pair_words, pair_token_stats = v3.pre_tokenize_pairs(pathlib.Path(args.arm_pair_jsonl), tokenizer, args.max_pairs)
    filler_chunks, filler_words, filler_tail = v3.pre_tokenize_filler(pathlib.Path(args.filler_jsonl), tokenizer, args.seq_len, args.max_filler_words)
    ep0_pair_chunks = sum(len(v3.make_pair_seqs(p, 0, "ff", args.seq_len, pad_id)) for p in pairs)
    seqs_per_epoch = ep0_pair_chunks + len(filler_chunks)
    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(seqs_per_epoch / eff_batch)
    total_steps = 2 * steps_per_epoch
    warmup = int(total_steps * args.warmup_frac)
    print(
        f"pairs={len(pairs):,} pair_chunks={ep0_pair_chunks:,} filler_chunks={len(filler_chunks):,} "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}",
        flush=True,
    )

    model = v3.build_model(tokenizer.vocab_size, args.seq_len, args.hidden, args.layers, args.heads, args.model_seed).to(args.device)
    opt, sched = make_optimizer_and_scheduler(model, args.lr, args.weight_decay, total_steps, warmup)
    n_params = sum(p.numel() for p in model.parameters())
    rng = random.Random(args.train_seed)
    global_step = 0
    prefix_info: Dict[str, Any] | None = None

    if args.phase == "prefix":
        set_train_rng(args.train_seed)
        epoch_index = 0
        status_name = "V4_PREFIX_COMPLETE"
    else:
        state_path = pathlib.Path(args.resume_state)
        # Load the fork state on CPU so the CPU RNG state remains a CPU ByteTensor.
        # Model parameters are copied into the already-CUDA model by load_state_dict;
        # optimizer moments are then moved to the target device explicitly.
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_state"])
        for opt_state in opt.state.values():
            for k, v in list(opt_state.items()):
                if torch.is_tensor(v):
                    opt_state[k] = v.to(args.device)
        sched.load_state_dict(state["scheduler_state"])
        torch.set_rng_state(state["torch_cpu_rng_state"])
        if torch.cuda.is_available() and state.get("torch_cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
        rng.setstate(state["python_rng_state"])
        global_step = int(state["global_step"])
        prefix_info = state["prefix_info"]
        epoch_index = 1
        status_name = "V4_CONTINUATION_COMPLETE"

    config = {
        "phase": args.phase,
        "direction": args.direction,
        "arm_pair_jsonl": args.arm_pair_jsonl,
        "arm_pair_sha256": sha256_file(pathlib.Path(args.arm_pair_jsonl)),
        "filler_jsonl": args.filler_jsonl,
        "filler_sha256": sha256_file(pathlib.Path(args.filler_jsonl)),
        "tokenizer_dir": args.tokenizer_dir,
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
        "model": {"params": n_params, "hidden": args.hidden, "layers": args.layers, "heads": args.heads, "vocab_size": tokenizer.vocab_size},
        "max_pairs": args.max_pairs,
        "max_filler_words": args.max_filler_words,
        "charged_pair_words_per_epoch": pair_words,
        "charged_filler_words_per_epoch": filler_words,
        "charged_words_per_epoch": pair_words + filler_words,
        "full_protocol_total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "seqs_per_epoch": seqs_per_epoch,
        "pair_chunks_per_epoch": ep0_pair_chunks,
        "pair_token_stats": pair_token_stats,
        "filler_tail_tokens_dropped": filler_tail,
        "resume_state": args.resume_state,
        "prefix_info": prefix_info,
        "device": args.device,
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    log_f = (out / "training_log.jsonl").open("w", encoding="utf-8")
    global_step, epoch_summary = train_one_epoch(
        model=model,
        opt=opt,
        sched=sched,
        rng=rng,
        pairs=pairs,
        filler_chunks=filler_chunks,
        epoch_index_zero_based=epoch_index,
        epoch_direction=args.direction,
        args=args,
        pad_id=pad_id,
        global_step=global_step,
        t0=t0,
        log_f=log_f,
    )
    log_f.close()

    hf_name = "prefix_epoch1" if args.phase == "prefix" else "final"
    hf_dir = out / "hf_model" / hf_name
    model_hash = v3.save_hf(model, tokenizer, hf_dir)
    state_file = None
    if args.phase == "prefix":
        prefix_info = {
            "prefix_direction": args.direction,
            "prefix_model_hash": model_hash,
            "prefix_hf_dir": str(hf_dir),
            "prefix_global_step": global_step,
            "prefix_epoch_summary": epoch_summary,
        }
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": sched.state_dict(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng_state": rng.getstate(),
            "global_step": global_step,
            "prefix_info": prefix_info,
            "config": config,
        }
        state_file = out / "fork_state_epoch1.pt"
        torch.save(state, state_file)

    elapsed = time.time() - t0
    manifest = {
        "status": status_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": elapsed,
        "phase": args.phase,
        "direction": args.direction,
        "global_step": global_step,
        "epoch_summary": epoch_summary,
        "model_hash": model_hash,
        "hf_dir": str(hf_dir),
        "state_file": str(state_file) if state_file else None,
        "prefix_info": prefix_info,
        "config_path": str(out / "train_config.json"),
    }
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
