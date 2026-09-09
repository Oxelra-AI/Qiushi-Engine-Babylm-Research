#!/usr/bin/env python3
"""research: packed-geometry target-selective trainer with event-level copied deletion.

This is a narrow fallback for the research historical packed-geometry intervention.
It reuses the same model, tokenizer, optimizer, scheduler, WWM masking, checkpointing,
and activation-checkpointed DeBERTa construction as packed_target_selective_trainer,
but it can drop copied-content labels at the realized exposure-event level using a map
from 100M stream_index -> word_group ids emitted by realized_wwm_selection_audit.py.

Use this only if the research pool-level copied whole-word selection is not matched well
enough under the actual ten-epoch CUDA WWM schedule.  For the primary source-absent arm,
continue using research drop_abs_content; source-absent deletion naturally covers all
realized rw_abs_content masked events.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

import packed_target_selective_trainer as base

MODES = ["full", "drop_copied_content_eventmatched"]


class EventMatchedAnnotatedChunkDataset(Dataset):
    """research dataset plus stream_index so deletion can be keyed to realized exposure events."""

    def __init__(self, examples, tokenizer, seq_length, annotations):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.annotations = annotations
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start = {}

    def _word_start_flag(self, token_id: int) -> bool:
        v = self._word_start.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and base.is_word_start(str(s)))
            self._word_start[token_id] = v
        return v

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"], add_special_tokens=False, truncation=True,
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

        word_cats = self.annotations.get(ex["example_id"])
        token_cat = torch.zeros_like(input_ids)
        if word_cats is not None:
            for i in range(input_ids.shape[0]):
                g = group[i].item()
                if g >= 0 and g < len(word_cats):
                    token_cat[i] = word_cats[g]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": group,
            "token_cat": token_cat,
            "words": ex["words"],
            "example_id": ex["example_id"],
            "stream_index": ex["stream_index"],
        }


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [x[k] for x in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals)
        else:
            out[k] = torch.tensor(vals, dtype=torch.long)
    return out


def apply_eventmatched_filter(labels, token_cat, word_group, stream_indices, mode, event_map):
    if mode == "full":
        return labels, {}
    if mode != "drop_copied_content_eventmatched":
        raise ValueError(mode)

    filtered = labels.clone()
    counts = collections.Counter()
    bsz, seq = labels.shape
    for b in range(bsz):
        sidx = str(int(stream_indices[b].item()))
        groups_to_drop = event_map.get(sidx)
        if not groups_to_drop:
            # Count kept masked labels by stratum for bookkeeping.
            for i in range(seq):
                if filtered[b, i] != -100:
                    cat = base.IDX_TO_CAT.get(int(token_cat[b, i].item()), base.CAT_FILLER)
                    counts[f"kept_{cat}"] += 1
            continue
        groups_to_drop = set(int(x) for x in groups_to_drop)
        for i in range(seq):
            if filtered[b, i] == -100:
                continue
            gid = int(word_group[b, i].item())
            cat = base.IDX_TO_CAT.get(int(token_cat[b, i].item()), base.CAT_FILLER)
            if gid >= 0 and gid in groups_to_drop:
                filtered[b, i] = -100
                counts["dropped_copied_eventmatched"] += 1
                counts[f"dropped_original_cat_{cat}"] += 1
            else:
                counts[f"kept_{cat}"] += 1
    return filtered, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", required=True, help="100M stream JSONL")
    p.add_argument("--annotation", required=True, help="Pool annotation JSONL")
    p.add_argument("--event_map", required=True, help="stream_index -> word_group ids JSON from research audit")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_mode", required=True, choices=MODES)
    p.add_argument("--max_word_exposure", type=int, default=100_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.06)
    p.add_argument("--lr_total_steps", type=int, default=2529)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)

    print("Loading annotations...", flush=True)
    annotations = {}
    with open(args.annotation, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            a = json.loads(line)
            annotations[int(a["example_id"])] = [base.CAT_TO_IDX.get(c, 0) for c in a["word_categories"]]

    event_map = {}
    if args.target_mode == "drop_copied_content_eventmatched":
        event_map = {str(k): [int(x) for x in v] for k, v in json.loads(Path(args.event_map).read_text(encoding="utf-8")).items()}
        print(f"Loaded event map rows={len(event_map)} groups={sum(len(v) for v in event_map.values())}", flush=True)

    print("Loading stream...", flush=True)
    selected_words = min(500_000, args.max_word_exposure) if args.smoke else args.max_word_exposure
    examples = []
    total_words = 0
    with open(args.stream, encoding="utf-8") as f:
        for sidx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if total_words + words > selected_words:
                break
            examples.append({
                "text": text,
                "words": words,
                "example_id": int(obj.get("example_id", len(examples))),
                "stream_index": sidx,
            })
            total_words += words
    print(f"Loaded {len(examples)} examples, {total_words} words", flush=True)

    dataset = EventMatchedAnnotatedChunkDataset(examples, tokenizer, args.max_seq_length, annotations)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=torch.cuda.is_available())
    total_steps = len(loader)
    print(f"Total steps: {total_steps}", flush=True)

    def reset_all_rng(s):
        random.seed(s)
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
    device = torch.device(args.device)
    model.to(device)

    init_sha = hashlib.sha256()
    for p_tensor in model.parameters():
        init_sha.update(p_tensor.data.cpu().numpy().tobytes())
    init_sha_hex = init_sha.hexdigest()
    print(f"Model: {param_count} params on {device}; init_sha={init_sha_hex}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    cumulative_words = 0
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    saved_checkpoints = []
    stratum_count_accum = collections.Counter()
    total_dropped = collections.Counter()

    model.train()
    for step, batch in enumerate(loader, 1):
        words = int(batch.pop("words").sum().item())
        example_ids = batch.pop("example_id")
        stream_indices = batch.pop("stream_index")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        word_group = batch["word_group"].to(device, non_blocking=True)
        token_cat = batch["token_cat"].to(device, non_blocking=True)

        masked_inputs, labels = base.apply_wwm_masking(input_ids, attention_mask, word_group, tokenizer, args.mask_prob, gen)
        filtered_labels, drop_counts = apply_eventmatched_filter(labels, token_cat, word_group, stream_indices, args.target_mode, event_map)
        for k, v in drop_counts.items():
            total_dropped[k] += int(v)

        with torch.no_grad():
            mask_pos = labels != -100
            for cat_idx, cat_name in base.IDX_TO_CAT.items():
                cat_mask = mask_pos & (token_cat == cat_idx)
                if cat_mask.any():
                    stratum_count_accum[cat_name] += int(cat_mask.sum().item())

        optim.zero_grad(set_to_none=True)
        out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=filtered_labels)
        loss = out_model.loss
        if loss is None:
            raise RuntimeError("model returned no loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        cumulative_words += words

        if step % args.log_every == 0 or step == total_steps:
            kept = int((filtered_labels != -100).sum().item())
            print(json.dumps({
                "e": "t",
                "mode": args.target_mode,
                "s": step,
                "l": round(loss.item(), 5),
                "c": cumulative_words,
                "lr": sched.get_last_lr()[0],
                "kept": kept,
            }, ensure_ascii=False), flush=True)

        if next_ckpt is not None and cumulative_words >= next_ckpt:
            ckpt_name = f"chck_{next_ckpt // 1_000_000}M"
            ckpt_path = out / "hf_model" / ckpt_name
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            saved_checkpoints.append({"name": ckpt_name, "target": next_ckpt, "actual": cumulative_words, "step": step, "path": str(ckpt_path)})
            print(json.dumps({"e": "ckpt", "mode": args.target_mode, "n": ckpt_name, "c": cumulative_words}), flush=True)
            next_ckpt += args.checkpoint_words

        if args.smoke and step >= 3:
            break

    metrics = {
        "status": f"PACKED_TARGET_SELECTIVE_{args.target_mode.upper()}",
        "target_mode": args.target_mode,
        "event_map": args.event_map,
        "parameter_count": param_count,
        "init_sha": init_sha_hex,
        "word_exposure": cumulative_words,
        "actual_training_steps": step,
        "loss_last": round(loss.item(), 6) if 'loss' in dir() else None,
        "stratum_masked_counts": dict(stratum_count_accum),
        "total_dropped_counts": dict(total_dropped),
        "saved_checkpoints": saved_checkpoints,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
