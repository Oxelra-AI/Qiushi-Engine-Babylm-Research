#!/usr/bin/env python3
"""research: clean-ancestry untouched tail-restart trainer with full AoA ladder.

The research E3/E4 controls suggested that the shared active ingredient was not
cluster content but restarting the optimizer/LR phase from clean-Qwen chck_80M.
This script makes that route measurable by continuing the exact clean-Qwen pool
from the true 80M parent and writing a model root that contains the parent
chck_1M..chck_80M ladder plus newly generated chck_85M/chck_90M/chck_95M/chck_100M.

Scientific constraints:
  * train text remains the clean Qwen-aligned 10M pool; no downstream/AoA text is
    used for training;
  * total exposure target is 100M words (80M parent + <=20M continuation);
  * parent checkpoints are symlinked by default, not rewritten, preserving ancestry;
  * the final chck_100M label records actual exposure, which may be slightly under
    100M to avoid overexposure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

SEQ_LENGTH = 256
MASK_PROB = 0.15
ORIGINAL_TOTAL_STEPS = 2515
WARMUP_FRACTION = 0.06
BASE_LR = 0.001
WEIGHT_DECAY = 0.01
PARENT_EXPOSURE = 80_000_000
CONTINUATION_BUDGET = 20_000_000
CHECKPOINT_INTERVAL = 5_000_000
AOA_PARENT_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 9)]
AOA_NEW_STEPS = ["chck_85M", "chck_90M", "chck_95M", "chck_100M"]
AOA_REQUIRED_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ContinuationDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, seq_length: int):
        self.chunks: list[torch.Tensor] = []
        buffer: list[int] = []
        for ex in examples:
            ids = tokenizer.encode(ex["text"], add_special_tokens=False)
            buffer.extend(ids)
            while len(buffer) >= seq_length:
                self.chunks.append(torch.tensor(buffer[:seq_length], dtype=torch.long))
                buffer = buffer[seq_length:]
        if len(buffer) > seq_length // 4:
            padded = buffer + [tokenizer.pad_token_id] * (seq_length - len(buffer))
            self.chunks.append(torch.tensor(padded[:seq_length], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def wwm_mask_batch(input_ids: torch.Tensor, tokenizer, mask_prob: float, gen: torch.Generator, device: torch.device):
    batch_size, seq_len = input_ids.shape
    special_ids = set(tokenizer.all_special_ids)
    mask_id = tokenizer.mask_token_id
    masked_inputs = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    masked_words = 0
    masked_tokens = 0
    for i in range(batch_size):
        word_groups: list[list[int]] = []
        current: list[int] = []
        for j in range(seq_len):
            tid = int(input_ids[i, j].item())
            if tid in special_ids or tid == tokenizer.pad_token_id:
                if current:
                    word_groups.append(current)
                    current = []
                continue
            tok = str(tokenizer.convert_ids_to_tokens(tid))
            if tok and is_word_start(tok) and current:
                word_groups.append(current)
                current = []
            current.append(j)
        if current:
            word_groups.append(current)
        if not word_groups:
            continue
        n_mask = min(len(word_groups), max(1, int(len(word_groups) * mask_prob)))
        for wi in torch.randperm(len(word_groups), generator=gen)[:n_mask].tolist():
            masked_words += 1
            for pos in word_groups[wi]:
                masked_tokens += 1
                labels[i, pos] = input_ids[i, pos]
                r = torch.rand(1, generator=gen).item()
                if r < 0.8:
                    masked_inputs[i, pos] = mask_id
                elif r < 0.9:
                    masked_inputs[i, pos] = torch.randint(len(tokenizer), (1,), generator=gen).item()
    return masked_inputs.to(device), attention_mask.to(device), labels.to(device), {"masked_words": masked_words, "masked_tokens": masked_tokens}


def compute_schedule(dataset_len: int, pool_words: int, batch_size: int, exposure_budget: int) -> dict:
    steps_per_pass = math.ceil(dataset_len / batch_size)
    passes_needed = math.ceil(exposure_budget / pool_words)
    total_steps = steps_per_pass * passes_needed
    approx_words_per_step = pool_words / steps_per_pass
    warmup_steps_orig = int(ORIGINAL_TOTAL_STEPS * WARMUP_FRACTION)
    post_warmup_progress = (ORIGINAL_TOTAL_STEPS * 0.8 - warmup_steps_orig) / (ORIGINAL_TOTAL_STEPS - warmup_steps_orig)
    tail_lr_factor = 0.5 * (1 + math.cos(math.pi * post_warmup_progress))
    continuation_lr = BASE_LR * tail_lr_factor
    return {
        "steps_per_pass": steps_per_pass,
        "passes_needed": passes_needed,
        "total_steps": total_steps,
        "approx_words_per_step": approx_words_per_step,
        "continuation_lr": continuation_lr,
        "tail_lr_factor": tail_lr_factor,
    }


def link_or_copy(src: Path, dst: Path, *, copy_parent: bool = False) -> str:
    if dst.exists() or dst.is_symlink():
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_parent:
        shutil.copytree(src, dst, symlinks=True)
        return "copied"
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel, target_is_directory=True)
    return "symlinked"


def prepare_parent_ladder(parent_model_root: Path, out_model_root: Path, *, copy_parent: bool = False) -> dict:
    records = []
    for name in AOA_PARENT_STEPS:
        src = parent_model_root / name
        if not src.exists():
            raise FileNotFoundError(f"missing parent checkpoint for AoA ladder: {src}")
        dst = out_model_root / name
        action = link_or_copy(src, dst, copy_parent=copy_parent)
        records.append({"name": name, "source": str(src), "path": str(dst), "action": action})
    return {"parent_steps": records}


def ckpt_name_for_total_word_target(total_words: int) -> str:
    return f"chck_{int(round(total_words / 1_000_000))}M"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent_model_root", required=True, help="Clean-Qwen hf_model root with chck_1M..chck_80M.")
    ap.add_argument("--init_checkpoint", required=True, help="Usually parent_model_root/chck_80M.")
    ap.add_argument("--train_file", required=True, help="Clean-Qwen 10M pool jsonl.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--label", default="untouched_restart_ladder")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_word_exposure", type=int, default=CONTINUATION_BUDGET)
    ap.add_argument("--start_word_exposure", type=int, default=PARENT_EXPOSURE)
    ap.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL)
    ap.add_argument("--train_rng_seed", type=int, default=43044)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--copy_parent_ladder", action="store_true", help="Copy parent checkpoint dirs instead of symlinking.")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_dir = Path(args.run_dir)
    out_model_root = run_dir / "hf_model"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_model_root.mkdir(parents=True, exist_ok=True)
    parent_model_root = Path(args.parent_model_root)
    init_path = Path(args.init_checkpoint)
    train_path = Path(args.train_file)
    if not parent_model_root.exists():
        raise FileNotFoundError(parent_model_root)
    if not init_path.exists():
        raise FileNotFoundError(init_path)
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    ladder_record = prepare_parent_ladder(parent_model_root, out_model_root, copy_parent=args.copy_parent_ladder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({
        "event": "start", "label": args.label, "init": str(init_path),
        "parent_model_root": str(parent_model_root), "train_file": str(train_path),
        "device": str(device), "started_utc": now(),
    }, indent=2), flush=True)

    model = DebertaV2ForMaskedLM.from_pretrained(str(init_path))
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(init_path), use_fast=True)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count:,}; tokenizer={len(tokenizer)}")

    examples: list[dict] = []
    pool_words = 0
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            actual = len(str(obj["text"]).split())
            words = int(obj.get("words", actual))
            if words != actual:
                raise RuntimeError(f"word mismatch row={len(examples)} field={words} actual={actual}")
            examples.append(obj)
            pool_words += words
    if pool_words != 10_000_000:
        print(json.dumps({"warning": "pool_words_not_10M", "pool_words": pool_words}), flush=True)
    dataset = ContinuationDataset(examples, tokenizer, SEQ_LENGTH)
    sched_info = compute_schedule(len(dataset), pool_words, args.batch_size, args.max_word_exposure)
    print(json.dumps({"event": "schedule", "pool_rows": len(examples), "pool_words": pool_words,
                      "dataset_chunks": len(dataset), **sched_info}, indent=2), flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=sched_info["continuation_lr"], weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=sched_info["total_steps"])
    random.seed(args.train_rng_seed)
    torch.manual_seed(args.train_rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_rng_seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.train_rng_seed)

    model.train()
    cumulative_words = 0
    processed_chunks_total = 0
    global_step = 0
    next_ckpt = args.checkpoint_interval
    loss_values: list[float] = []
    saved_checkpoints: list[dict] = []
    saved_names: set[str] = {r["name"] for r in ladder_record["parent_steps"]}
    mask_totals = {"masked_words": 0, "masked_tokens": 0}
    log_path = run_dir / "training_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    def save_new_checkpoint(name: str, *, target_continuation_words: int, loss_val: float) -> None:
        ckpt_path = out_model_root / name
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_path, safe_serialization=True)
        tokenizer.save_pretrained(ckpt_path)
        if name not in saved_names:
            saved_names.add(name)
            saved_checkpoints.append({
                "name": name,
                "path": str(ckpt_path),
                "step": global_step,
                "target_continuation_word_exposure": target_continuation_words,
                "actual_continuation_word_exposure": cumulative_words,
                "target_total_word_exposure": args.start_word_exposure + target_continuation_words,
                "actual_total_word_exposure": args.start_word_exposure + cumulative_words,
                "processed_chunks": processed_chunks_total,
                "loss": round(loss_val, 4),
            })
        print(f"  >> Saved {name} at step {global_step} actual_total={args.start_word_exposure + cumulative_words}", flush=True)

    stop_training = False
    for pass_i in range(sched_info["passes_needed"]):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        pos = 0
        while pos < len(indices):
            batch_indices: list[int] = []
            while pos < len(indices) and len(batch_indices) < args.batch_size:
                projected_chunks = processed_chunks_total + len(batch_indices) + 1
                projected_words = int(round((projected_chunks / max(1, len(dataset))) * pool_words))
                if projected_words > args.max_word_exposure:
                    break
                batch_indices.append(indices[pos])
                pos += 1
            if not batch_indices:
                stop_training = True
                break
            input_ids = torch.stack([dataset[i] for i in batch_indices])
            masked_inputs, attention_mask, labels, mask_stats = wwm_mask_batch(input_ids, tokenizer, MASK_PROB, gen, device)
            for k in mask_totals:
                mask_totals[k] += int(mask_stats.get(k, 0))
            optim.zero_grad(set_to_none=True)
            out = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            if out.loss is None:
                raise RuntimeError("Model returned no loss")
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            global_step += 1
            loss_val = float(out.loss.item())
            loss_values.append(loss_val)
            processed_chunks_total += len(batch_indices)
            cumulative_words = min(args.max_word_exposure, int(round((processed_chunks_total / max(1, len(dataset))) * pool_words)))
            if global_step % args.log_every == 0 or cumulative_words >= args.max_word_exposure:
                entry = {
                    "step": global_step,
                    "loss": round(loss_val, 4),
                    "lr": round(sched.get_last_lr()[0], 8),
                    "continuation_words": cumulative_words,
                    "total_words_with_parent": args.start_word_exposure + cumulative_words,
                    "pass": pass_i + 1,
                    "batch_chunks": len(batch_indices),
                    "processed_chunks": processed_chunks_total,
                    **mask_totals,
                }
                log_f.write(json.dumps(entry) + "\n")
                log_f.flush()
                print(f"  step {global_step}/{sched_info['total_steps']} loss={loss_val:.4f} lr={sched.get_last_lr()[0]:.2e} cont_words={cumulative_words}", flush=True)
            while next_ckpt and cumulative_words >= next_ckpt and next_ckpt < args.max_word_exposure:
                save_new_checkpoint(ckpt_name_for_total_word_target(args.start_word_exposure + next_ckpt),
                                    target_continuation_words=next_ckpt, loss_val=loss_val)
                next_ckpt += args.checkpoint_interval
            if cumulative_words >= args.max_word_exposure:
                stop_training = True
                break
        if stop_training:
            break
    log_f.close()

    # Final 100M target checkpoint.  This may be slightly under exact 100M actual exposure
    # because the chunked trainer refuses overexposure; the manifest records the actual total.
    final_name = ckpt_name_for_total_word_target(args.start_word_exposure + args.max_word_exposure)
    save_new_checkpoint(final_name, target_continuation_words=args.max_word_exposure,
                        loss_val=float(loss_values[-1] if loss_values else 0.0))

    available = sorted([p.name for p in out_model_root.iterdir() if p.is_dir() or p.is_symlink()])
    missing_required = [name for name in AOA_REQUIRED_STEPS if name not in set(available)]
    metrics = {
        "variant": f"step045_{args.label}",
        "mechanism": "clean_parent_80M_tail_optimizer_restart_uniform_wwm",
        "parent_model_root": str(parent_model_root),
        "init_checkpoint": str(init_path),
        "train_file": str(train_path),
        "train_file_sha256": sha256_file(train_path),
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "start_word_exposure": args.start_word_exposure,
        "continuation_word_exposure": cumulative_words,
        "actual_total_word_exposure": args.start_word_exposure + cumulative_words,
        "target_total_word_exposure": args.start_word_exposure + args.max_word_exposure,
        "max_word_exposure_requested": args.max_word_exposure,
        "actual_training_steps": global_step,
        "processed_chunks_total": processed_chunks_total,
        "pool_words": pool_words,
        "pool_rows": len(examples),
        "dataset_chunks": len(dataset),
        "batch_size": args.batch_size,
        "seq_length": SEQ_LENGTH,
        "mask_prob": MASK_PROB,
        "optimizer": "AdamW_fresh_state_betas_0.9_0.98",
        "continuation_lr": sched_info["continuation_lr"],
        "tail_lr_factor": sched_info["tail_lr_factor"],
        "lr_schedule": "fresh_cosine_decay_no_warmup_over_continuation",
        "loss_first": round(loss_values[0], 4) if loss_values else None,
        "loss_last": round(loss_values[-1], 4) if loss_values else None,
        "loss_mean": round(sum(loss_values) / len(loss_values), 4) if loss_values else None,
        "mask_stats_totals": mask_totals,
        "train_rng_seed": args.train_rng_seed,
        "created_utc": now(),
        "parent_ladder": ladder_record,
        "new_tail_checkpoints": saved_checkpoints,
        "available_checkpoint_count": len(available),
        "available_checkpoints": available,
        "missing_required_aoa_steps": missing_required,
        "aoa_ladder_ready": not missing_required,
        "non_leakage_statement": "Continuation uses only clean-Qwen training pool and parent checkpoints; no downstream labels/items, official AoA/CDI words, child curves, or leaderboard scores were used for training.",
    }
    (run_dir / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "finished", "label": args.label,
        "steps": global_step, "continuation_words": cumulative_words,
        "total_words_with_parent": args.start_word_exposure + cumulative_words,
        "loss_last": metrics["loss_last"], "new_checkpoints": [c["name"] for c in saved_checkpoints],
        "aoa_ladder_ready": metrics["aoa_ladder_ready"], "metrics": str(run_dir / "scientific_metrics.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
