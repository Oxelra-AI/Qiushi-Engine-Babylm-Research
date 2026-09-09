#!/usr/bin/env python3
"""research strict content-innovation WWM trainer.

This is the repaired, isolated version of research's innovation-biased masking.
It targets only the repaired research object:
  * source-absent, row-unique, content-like rewrite groups.
It does NOT suppress all copyable rewrite masks and does NOT raise source/filler masks.
Instead, it shifts expected token-label mass from copyable rewrite groups into strict
content innovations while keeping source, filler, duplicate source-absent, and
function-like source-absent groups at the ordinary WWM rate.

Everything else follows the COMPACT_EXPERIENCE base trainer recipe used by the research legal16k
compact-view reinvest baseline: 8x480 DeBERTa-v2, token-mean MLM loss, legal 16k
tokenizer, AdamW lr=1e-3, fixed seq256, fixed WWM corruption.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import pathlib
import random
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
BASE_TRAINER_PATH = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
STRICT_MAP_PATH = USER_ROOT / "experiments/archive/frontier_consolidation/data/strict_innovation_group_map/strict_innovation_group_map.json"


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


def load_strict_map(path: pathlib.Path) -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict[str, Any]] = {}
    totals = defaultdict(float)
    for k, v in raw.items():
        eid = int(k)
        rec = {
            "strict_gids": set(int(x) for x in v["strict_content_innovation_gids"]),
            "copyable_gids": set(int(x) for x in v["copyable_gids"]),
            "source_gids": set(int(x) for x in v.get("source_gids", [])),
            "rewrite_other_gids": set(int(x) for x in v.get("rewrite_other_gids", [])),
            "filler_gids": set(int(x) for x in v.get("filler_gids", [])),
            "duplicate_source_absent_gids": set(int(x) for x in v.get("duplicate_source_absent_gids", [])),
            "function_source_absent_unique_gids": set(int(x) for x in v.get("function_source_absent_unique_gids", [])),
            "short_or_empty_source_absent_gids": set(int(x) for x in v.get("short_or_empty_source_absent_gids", [])),
            "strict_tokens": int(v.get("strict_tokens", 0)),
            "copyable_tokens": int(v.get("copyable_tokens", 0)),
            "ordinary_tokens": int(v.get("ordinary_tokens", 0)),
            "n_total_groups": int(v["n_total_groups"]),
        }
        out[eid] = rec
        totals["strict_groups"] += len(rec["strict_gids"])
        totals["copyable_groups"] += len(rec["copyable_gids"])
        totals["source_groups"] += len(rec["source_gids"])
        totals["rewrite_other_groups"] += len(rec["rewrite_other_gids"])
        totals["filler_groups"] += len(rec["filler_gids"])
        totals["strict_tokens"] += rec["strict_tokens"]
        totals["copyable_tokens"] += rec["copyable_tokens"]
        totals["ordinary_tokens"] += rec["ordinary_tokens"]
    return out, dict(totals)


class StrictInnovationMaskedChunkDataset(Dataset):
    """Base masked chunk dataset plus example_id for strict-map lookup."""

    def __init__(self, examples, tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start: dict[int, bool] = {}

    def _word_start_flag(self, token_id: int) -> bool:
        val = self._word_start.get(token_id)
        if val is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            val = bool(s is not None and base.is_word_start(str(s)))
            self._word_start[token_id] = val
        return val

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.seq_length,
            padding="max_length",
            return_tensors="pt",
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


def strict_innovation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
        "example_ids": torch.tensor([x["example_id"] for x in batch], dtype=torch.long),
    }


def compute_auto_p_copy(mask_prob: float, p_strict: float, map_totals: dict[str, float], basis: str = "tokens") -> float:
    if p_strict < mask_prob:
        raise ValueError(f"p_strict {p_strict} must be >= mask_prob {mask_prob}")
    if basis == "tokens":
        strict_mass = float(map_totals["strict_tokens"])
        copy_mass = float(map_totals["copyable_tokens"])
    elif basis == "groups":
        strict_mass = float(map_totals["strict_groups"])
        copy_mass = float(map_totals["copyable_groups"])
    else:
        raise ValueError(f"unknown p_copy basis: {basis}")
    if copy_mass <= 0:
        raise ValueError("no copyable mass available")
    p_copy = mask_prob - (p_strict - mask_prob) * strict_mass / copy_mass
    if not (0.0 <= p_copy <= mask_prob):
        raise ValueError(f"auto p_copy out of range: {p_copy}; strict_mass={strict_mass} copy_mass={copy_mass}")
    return float(p_copy)


def apply_strict_innovation_masking(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    example_ids: torch.Tensor,
    tokenizer,
    mask_prob: float,
    p_strict: float,
    p_copy: float,
    strict_map: dict[int, dict[str, Any]],
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Apply isolated strict innovation WWM reallocation.

    Changed rows:
      strict content-innovation groups -> p_strict
      copyable rewrite groups -> p_copy
      all other groups -> mask_prob
    Unchanged rows:
      standard WWM -> mask_prob
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
        gid_list = [int(x) for x in valid_groups.tolist()]

        if eid in strict_map:
            rec = strict_map[eid]
            strict_set = rec["strict_gids"]
            copy_set = rec["copyable_gids"]
            source_set = rec.get("source_gids", set())
            rewrite_other_set = rec.get("rewrite_other_gids", set())
            filler_set = rec.get("filler_gids", set())

            strict_mask = torch.tensor([g in strict_set for g in gid_list], device=device)
            copy_mask = torch.tensor([g in copy_set for g in gid_list], device=device)
            thresholds = torch.full((valid_groups.numel(),), float(mask_prob), device=device)
            thresholds[strict_mask] = float(p_strict)
            thresholds[copy_mask] = float(p_copy)
            gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
            chosen = valid_groups[gp < thresholds]
            chosen_set = set(int(x) for x in chosen.tolist())

            stats["changed_rows"] += 1
            stats["strict_available_groups"] += int(strict_mask.sum().item())
            stats["copy_available_groups"] += int(copy_mask.sum().item())
            stats["ordinary_available_groups"] += int((~strict_mask & ~copy_mask).sum().item())
            stats["strict_selected_groups"] += sum(1 for g in gid_list if g in chosen_set and g in strict_set)
            stats["copy_selected_groups"] += sum(1 for g in gid_list if g in chosen_set and g in copy_set)
            stats["ordinary_selected_groups"] += sum(1 for g in gid_list if g in chosen_set and g not in strict_set and g not in copy_set)
            stats["source_available_groups"] += sum(1 for g in gid_list if g in source_set)
            stats["source_selected_groups"] += sum(1 for g in gid_list if g in chosen_set and g in source_set)
            stats["rewrite_other_available_groups"] += sum(1 for g in gid_list if g in rewrite_other_set)
            stats["rewrite_other_selected_groups"] += sum(1 for g in gid_list if g in chosen_set and g in rewrite_other_set)
            stats["filler_available_groups"] += sum(1 for g in gid_list if g in filler_set)
            stats["filler_selected_groups"] += sum(1 for g in gid_list if g in chosen_set and g in filler_set)
        else:
            gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
            chosen = valid_groups[gp < mask_prob]
            chosen_set = set(int(x) for x in chosen.tolist())
            stats["unchanged_rows"] += 1

        if chosen.numel() == 0:
            stats["empty_group_selection_rows"] += 1
            continue
        sel_b = torch.isin(groups, chosen) & candidate[b]
        select[b] = sel_b

        if eid in strict_map:
            rec = strict_map[eid]
            strict_set = rec["strict_gids"]
            copy_set = rec["copyable_gids"]
            source_set = rec.get("source_gids", set())
            rewrite_other_set = rec.get("rewrite_other_gids", set())
            filler_set = rec.get("filler_gids", set())
            for cat_name, cat_set in [
                ("strict", strict_set),
                ("copy", copy_set),
                ("source", source_set),
                ("rewrite_other", rewrite_other_set),
                ("filler", filler_set),
            ]:
                if not cat_set:
                    continue
                cat_token_mask = sel_b & torch.isin(groups, torch.tensor(sorted(cat_set), device=device))
                cat_avail_mask = candidate[b] & torch.isin(groups, torch.tensor(sorted(cat_set), device=device))
                stats[f"{cat_name}_selected_tokens"] += int(cat_token_mask.sum().item())
                stats[f"{cat_name}_available_tokens"] += int(cat_avail_mask.sum().item())
            ordinary_set = set(gid_list) - rec["strict_gids"] - rec["copyable_gids"]
            if ordinary_set:
                ordinary_t = torch.tensor(sorted(ordinary_set), device=device)
                stats["ordinary_selected_tokens"] += int((sel_b & torch.isin(groups, ordinary_t)).sum().item())
                stats["ordinary_available_tokens"] += int((candidate[b] & torch.isin(groups, ordinary_t)).sum().item())

    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
            stats["global_forced_fallback"] += 1

    labels[~select] = -100

    masked_inputs = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = mask_token_id
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum().item()),), generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids

    stats["total_selected_tokens"] += int(select.sum().item())
    stats["total_candidate_tokens"] += int(candidate.sum().item())
    return masked_inputs, labels, dict(stats)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict content-innovation WWM trainer")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--example_jsonl_label", default="compact_view_reinvest")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--tokenizer_label", default="legal16k")
    p.add_argument("--strict_map", default=str(STRICT_MAP_PATH))
    p.add_argument("--max_word_exposure", type=int, default=100_000_000)
    p.add_argument("--example_pool_words", type=int, default=100_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--p_strict", type=float, default=0.5)
    p.add_argument("--p_copy", type=float, default=-1.0,
                   help="Copyable rewrite group probability; <0 computes isolated mass match")
    p.add_argument("--p_copy_basis", choices=["tokens", "groups"], default="tokens")
    # Model shape matching research legal16k reinvest.
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
    # Optimization matching research.
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--masking_curriculum", default="wwm_fixed")
    return p.parse_args()


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = build_args()
    start = time.time()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strict_map, map_totals = load_strict_map(pathlib.Path(args.strict_map))
    effective_p_copy = args.p_copy
    if effective_p_copy < 0:
        effective_p_copy = compute_auto_p_copy(args.mask_prob, args.p_strict, map_totals, args.p_copy_basis)

    print(json.dumps({
        "event": "strict_map_loaded",
        "changed_rows": len(strict_map),
        "strict_groups": int(map_totals["strict_groups"]),
        "copyable_groups": int(map_totals["copyable_groups"]),
        "strict_tokens": int(map_totals["strict_tokens"]),
        "copyable_tokens": int(map_totals["copyable_tokens"]),
        "mask_prob": args.mask_prob,
        "p_strict": args.p_strict,
        "p_copy_effective": effective_p_copy,
        "p_copy_basis": args.p_copy_basis,
        "time": now_utc(),
    }), flush=True)

    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(
        pathlib.Path(args.example_jsonl), args.max_word_exposure
    )
    actual_words = sum(ex.words for ex in examples)
    if actual_words != args.max_word_exposure:
        raise RuntimeError(f"word mismatch {actual_words} vs {args.max_word_exposure}")

    dataset = StrictInnovationMaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=strict_innovation_collate, num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())
    total_steps = len(loader)

    reset_all_rng(args.seed)
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = base.build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)
    param_count = sum(p.numel() for p in model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    manifest = {
        "session": "frontier_consolidation",
        "step_label": "strict_content_innovation_wwm",
        "description": "strict repaired-research content-innovation WWM reallocation on compact-view reinvest",
        "strict_map": str(args.strict_map),
        "map_totals": {k: int(v) for k, v in map_totals.items()},
        "mask_prob": args.mask_prob,
        "p_strict": args.p_strict,
        "p_copy_effective": effective_p_copy,
        "p_copy_basis": args.p_copy_basis,
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
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    log_path = out_dir / "training_log.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict[str, Any]] = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    cumulative_mask_stats = defaultdict(int)

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            example_ids = batch.pop("example_ids")
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            masked_inputs, labels, mask_stats = apply_strict_innovation_masking(
                input_ids, attention_mask, word_group, example_ids,
                tokenizer, args.mask_prob, args.p_strict, effective_p_copy,
                strict_map, gen,
            )
            for k, v in mask_stats.items():
                cumulative_mask_stats[k] += int(v)

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
                    "step": step,
                    "loss": round(loss_float, 6),
                    "avg_loss": round(avg_loss, 6),
                    "words": cumulative_words,
                    "lr": round(sched.get_last_lr()[0], 8),
                    "mask_rate": round(effective_mask_rate, 4),
                    "n_pred": n_pred,
                    "changed_rows_cumulative": cumulative_mask_stats.get("changed_rows", 0),
                }
                logf.write(json.dumps(entry) + "\n")
                logf.flush()
                print(json.dumps({"event": "log", **entry, "time": now_utc()}), flush=True)

            if next_ckpt is not None and cumulative_words >= next_ckpt:
                ckpt_name = f"chck_{cumulative_words // 1_000_000}M"
                ckpt_dir = out_dir / "hf_model" / ckpt_name
                base.save_hf_checkpoint(model, tokenizer, ckpt_dir)
                ckpt_info = {
                    "name": ckpt_name,
                    "step": step,
                    "words": cumulative_words,
                    "loss": round(loss_float, 6),
                    "path": str(ckpt_dir),
                    "mask_rate": round(effective_mask_rate, 4),
                }
                saved_checkpoints.append(ckpt_info)
                print(json.dumps({"event": "checkpoint", **ckpt_info, "time": now_utc()}), flush=True)
                next_ckpt += args.checkpoint_words

    elapsed = time.time() - start
    final_loss = loss_values[-1] if loss_values else 0.0
    avg_last_50 = sum(loss_values[-50:]) / max(1, len(loss_values[-50:]))
    sci_metrics = {
        "session": "frontier_consolidation",
        "step_label": "strict_content_innovation_wwm",
        "description": manifest["description"],
        "mask_prob": args.mask_prob,
        "p_strict": args.p_strict,
        "p_copy_effective": effective_p_copy,
        "p_copy_basis": args.p_copy_basis,
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
    (out_dir / "scientific_metrics.json").write_text(json.dumps(sci_metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "done",
        "final_loss": sci_metrics["final_loss"],
        "total_steps": total_steps,
        "total_words": cumulative_words,
        "param_count": param_count,
        "elapsed_sec": sci_metrics["elapsed_sec"],
        "p_strict": args.p_strict,
        "p_copy_effective": effective_p_copy,
    }), flush=True)


if __name__ == "__main__":
    main()
