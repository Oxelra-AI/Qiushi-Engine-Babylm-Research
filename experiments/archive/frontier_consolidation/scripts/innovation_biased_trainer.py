#!/usr/bin/env python3
"""research: Innovation-biased WWM trainer for compact-view reinvest.

Keeps the legal research tokenizer, compact-view-reinvest 100M stream, model shape,
optimizer, seeds, and checkpoint cadence from the COMPACT_EXPERIENCE base trainer, but replaces
standard uniform WWM group selection with innovation-biased selection on the 3,005
changed rows:

  - Innovation groups (source-absent rewrite targets): selected at p_innov (default 0.5)
  - Copyable groups (source-present rewrite targets): selected at p_copy (default 0.0)
  - Other groups (source spans, filler text): selected at p_other computed to match
    overall 0.15 total group budget per row

On unchanged rows (61,735 of 64,740), standard WWM at 0.15 applies unmodified.

The scientific hypothesis: biasing toward harder, source-dependent innovation targets
forces the model to learn conditional content prediction rather than easy copying,
strengthening the redundancy-reduced semantic second-view mechanism.

"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import math
import pathlib
import random
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
BASE_TRAINER_PATH = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
INNOVATION_MAP_PATH = USER_ROOT / "experiments/archive/frontier_consolidation/data/innovation_group_map/innovation_group_map.json"


def load_base_trainer():
    spec = importlib.util.spec_from_file_location("compact_experience_base_trainer", BASE_TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base trainer from {BASE_TRAINER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_base_trainer()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ─── Innovation map loading ──────────────────────────────────────────────────
def load_innovation_map(path: pathlib.Path) -> dict[int, dict[str, Any]]:
    """Load the precomputed innovation group map. Keys are example_id (int)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict[str, Any]] = {}
    for k, v in raw.items():
        eid = int(k)
        out[eid] = {
            "innovation_gids": set(v["innovation_gids"]),
            "copyable_gids": set(v["copyable_gids"]),
            "n_total_groups": int(v["n_total_groups"]),
        }
    return out


# ─── Extended dataset returning example_id ────────────────────────────────────
class InnovationMaskedChunkDataset(Dataset):
    """MaskedChunkDataset that also returns example_id for innovation map lookup."""

    def __init__(self, examples, tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start = {}

    def _word_start_flag(self, token_id: int) -> bool:
        v = self._word_start.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and base.is_word_start(str(s)))
            self._word_start[token_id] = v
        return v

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text, add_special_tokens=False, truncation=True,
            max_length=self.seq_length, padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if attention_mask[i] == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._word_start_flag(tid) or i == 0:
                gid += 1
            group[i] = gid
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": group,
            "words": ex.words,
            "example_id": ex.example_id,
        }


def innovation_collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
        "example_ids": torch.tensor([x["example_id"] for x in batch], dtype=torch.long),
    }


# ─── Innovation-biased masking ────────────────────────────────────────────────
def apply_innovation_biased_masking(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    example_ids: torch.Tensor,
    tokenizer,
    mask_prob: float,
    p_innov: float,
    p_copy: float,
    innovation_map: dict[int, dict[str, Any]],
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """WWM with innovation-biased group selection on changed rows.

    Returns (masked_inputs, labels, stats_dict).
    """
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)

    stats = defaultdict(int)

    for b in range(bsz):
        groups = word_group[b]
        valid_groups = torch.unique(groups[groups >= 0])
        if valid_groups.numel() == 0:
            continue

        eid = int(example_ids[b].item())

        if eid in innovation_map:
            # Changed row: innovation-biased selection
            info = innovation_map[eid]
            innov_set = info["innovation_gids"]
            copy_set = info["copyable_gids"]

            n_total = valid_groups.numel()
            gid_list = valid_groups.tolist()

            # Categorize each group
            innov_mask = torch.tensor([int(g) in innov_set for g in gid_list], device=device)
            copy_mask = torch.tensor([int(g) in copy_set for g in gid_list], device=device)
            other_mask = ~innov_mask & ~copy_mask

            n_innov = int(innov_mask.sum().item())
            n_copy = int(copy_mask.sum().item())
            n_other = int(other_mask.sum().item())

            # Compute p_other to match total budget
            target_total = mask_prob * n_total
            expected_innov = p_innov * n_innov
            expected_copy = p_copy * n_copy
            remaining = target_total - expected_innov - expected_copy
            p_other = max(0.0, min(1.0, remaining / max(1, n_other)))

            # Generate random values for all groups at once (preserves RNG consumption count)
            gp = torch.rand(n_total, generator=gen, device=device)

            # Apply per-category thresholds
            thresholds = torch.full((n_total,), mask_prob, device=device)
            thresholds[innov_mask] = p_innov
            thresholds[copy_mask] = p_copy
            thresholds[other_mask] = p_other

            chosen = valid_groups[gp < thresholds]

            stats["changed_rows"] += 1
            stats["innov_available"] += n_innov
            stats["copy_available"] += n_copy
            # Count what was actually selected
            if chosen.numel() > 0:
                chosen_set = set(chosen.tolist())
                stats["innov_selected"] += sum(1 for g in gid_list if g in chosen_set and int(g) in innov_set)
                stats["copy_selected"] += sum(1 for g in gid_list if g in chosen_set and int(g) in copy_set)
                stats["other_selected"] += sum(1 for g in gid_list if g in chosen_set and int(g) not in innov_set and int(g) not in copy_set)
        else:
            # Unchanged row: standard WWM
            gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
            chosen = valid_groups[gp < mask_prob]
            stats["unchanged_rows"] += 1

        if chosen.numel() == 0:
            stats["forced_fallback"] += 1
            continue
        sel_b = torch.isin(groups, chosen) & candidate[b]
        select[b] = sel_b

    # Force at least one selected token if any candidates exist
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
            stats["global_forced_fallback"] += 1

    labels[~select] = -100

    # Standard 80/10/10 corruption
    masked_inputs = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = mask_token_id
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids

    stats["total_selected_tokens"] += int(select.sum().item())
    stats["total_candidate_tokens"] += int(candidate.sum().item())

    return masked_inputs, labels, dict(stats)


# ─── Argument parsing ─────────────────────────────────────────────────────────
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Innovation-biased WWM Trainer")
    # Core data args (matching base trainer)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--example_jsonl_label", default="compact_view_reinvest")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--tokenizer_label", default="legal16k")
    p.add_argument("--max_word_exposure", type=int, default=100_000_000)
    p.add_argument("--example_pool_words", type=int, default=100_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    # Innovation-biased masking
    p.add_argument("--innovation_map", default=str(INNOVATION_MAP_PATH))
    p.add_argument("--p_innov", type=float, default=0.5,
                   help="Mask probability for innovation groups on changed rows")
    p.add_argument("--p_copy", type=float, default=0.0,
                   help="Mask probability for copyable groups on changed rows")
    p.add_argument("--mask_prob", type=float, default=0.15,
                   help="Standard mask probability (unchanged rows + budget target)")
    # Model (DeBERTa-v2 8x480)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--seq_length", type=int, default=256)
    # Optimization (matching base)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    # Seeds (matching the winning recipe)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    p.add_argument("--log_every", type=int, default=50)
    # Masking curriculum (fixed to wwm_fixed for this experiment)
    p.add_argument("--masking_curriculum", default="wwm_fixed")
    return p.parse_args()


# ─── Main training loop ──────────────────────────────────────────────────────
def main() -> None:
    args = build_args()
    start_time = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load innovation map
    innov_path = pathlib.Path(args.innovation_map)
    if not innov_path.exists():
        raise FileNotFoundError(f"Innovation map not found: {innov_path}")
    innovation_map = load_innovation_map(innov_path)
    n_changed = len(innovation_map)
    n_innov_total = sum(len(v["innovation_gids"]) for v in innovation_map.values())
    n_copy_total = sum(len(v["copyable_gids"]) for v in innovation_map.values())
    print(json.dumps({"event": "innovation_map_loaded", "changed_rows": n_changed,
                       "innovation_groups_total": n_innov_total,
                       "copyable_groups_total": n_copy_total,
                       "p_innov": args.p_innov, "p_copy": args.p_copy}), flush=True)

    # Tokenizer
    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)

    # Load data from 100M JSONL (matching base trainer)
    jsonl_path = pathlib.Path(args.example_jsonl)
    examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(
        jsonl_path, args.max_word_exposure
    )
    actual_words = sum(ex.words for ex in examples)
    if actual_words != args.max_word_exposure:
        raise RuntimeError(f"word mismatch {actual_words} vs {args.max_word_exposure}")

    # Dataset and DataLoader
    dataset = InnovationMaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=innovation_collate, num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())
    total_steps = len(loader)

    # Model (exactly matching base)
    def reset_all_rng(s):
        random.seed(s)
        np.random.seed(s % (2**32 - 1))
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    reset_all_rng(args.seed)
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = base.build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)

    param_count = sum(p.numel() for p in model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer (exactly matching base)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay, betas=(0.9, 0.98))
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup,
                                            num_training_steps=schedule_total)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    # Save manifest
    manifest = {
        "session": "frontier_consolidation",
        "step": "research",
        "description": "innovation-biased WWM on compact-view reinvest",
        "innovation_map": str(innov_path),
        "p_innov": args.p_innov,
        "p_copy": args.p_copy,
        "mask_prob": args.mask_prob,
        "changed_rows_in_map": n_changed,
        "innovation_groups_per_pass": n_innov_total,
        "copyable_groups_per_pass": n_copy_total,
        "tokenizer_path": args.tokenizer_path,
        "example_jsonl": args.example_jsonl,
        "max_word_exposure": args.max_word_exposure,
        "param_count": param_count,
        "total_steps": total_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "created_utc": now_utc(),
    }
    (out / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # Training loop
    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict] = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    cumulative_mask_stats = defaultdict(int)

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            example_ids_batch = batch.pop("example_ids")
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            # Innovation-biased masking
            masked_inputs, labels, mask_stats = apply_innovation_biased_masking(
                input_ids, attention_mask, word_group,
                example_ids_batch,
                tokenizer, args.mask_prob, args.p_innov, args.p_copy,
                innovation_map, gen,
            )
            for k, v in mask_stats.items():
                cumulative_mask_stats[k] += v

            # Forward pass (standard token-mean MLM loss)
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
            loss_values.append(loss_float)
            n_pred = int((labels != -100).sum().item())
            n_candidate = int(attention_mask.bool().sum().item())
            effective_mask_rate = n_pred / max(1, n_candidate)

            if step % args.log_every == 0 or step == 1:
                avg_loss = sum(loss_values[-args.log_every:]) / len(loss_values[-args.log_every:])
                entry = {
                    "step": step, "loss": round(loss_float, 6),
                    "avg_loss": round(avg_loss, 6), "words": cumulative_words,
                    "lr": round(sched.get_last_lr()[0], 8),
                    "mask_rate": round(effective_mask_rate, 4),
                    "n_pred": n_pred,
                }
                logf.write(json.dumps(entry) + "\n")
                logf.flush()
                print(json.dumps({"event": "log", **entry, "time": now_utc()}), flush=True)

            # Checkpointing (every 1M words)
            if next_ckpt is not None and cumulative_words >= next_ckpt:
                ckpt_name = f"chck_{cumulative_words // 1_000_000}M"
                ckpt_dir = out / "hf_model" / ckpt_name
                base.save_hf_checkpoint(model, tokenizer, ckpt_dir)
                ckpt_info = {
                    "name": ckpt_name, "step": step, "words": cumulative_words,
                    "loss": round(loss_float, 6), "path": str(ckpt_dir),
                    "mask_rate": round(effective_mask_rate, 4),
                }
                saved_checkpoints.append(ckpt_info)
                print(json.dumps({"event": "checkpoint", **ckpt_info, "time": now_utc()}), flush=True)
                next_ckpt += args.checkpoint_words

    elapsed = time.time() - start_time

    # Final summary (matching base trainer format for evaluation compatibility)
    final_loss = loss_values[-1] if loss_values else 0.0
    avg_last_50 = sum(loss_values[-50:]) / max(1, len(loss_values[-50:]))
    sci_metrics = {
        "session": "frontier_consolidation",
        "step_label": "innovation_biased_wwm",
        "description": manifest["description"],
        "p_innov": args.p_innov,
        "p_copy": args.p_copy,
        "final_loss": round(final_loss, 6),
        "avg_last_50_loss": round(avg_last_50, 6),
        "total_steps": total_steps,
        "param_count": param_count,
        "total_words": cumulative_words,
        "checkpoints": [c["name"] for c in saved_checkpoints],
        "cumulative_mask_stats": dict(cumulative_mask_stats),
        "elapsed_sec": round(elapsed, 1),
        "finished_utc": now_utc(),
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(sci_metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "done", **{k: sci_metrics[k] for k in
                       ["final_loss", "total_steps", "param_count", "total_words",
                        "elapsed_sec", "p_innov", "p_copy"]}}), flush=True)


if __name__ == "__main__":
    main()
