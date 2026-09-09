#!/usr/bin/env python3
"""Explicit cross-view masking trainer for BabyLM rewrite-pair experiments.

This script is intentionally separate from the stable `babylm_masked_train.py`.
It consumes pre-materialized JSONL only. Official examples and pair examples with
no anchors use standard WWM; pair examples with `anchor_source_word_indices` mask
those source-side word groups deterministically while leaving the target view
visible.

Matched control: true-pair and shuffled-pair arms must not reselect
anchors independently. The materializer chooses mask side/anchors from the true
pair once and copies them into the shuffled arm; this trainer records actual
selected groups/tokens per example so the smoke can verify per-sample supervision
identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_portable_tokenizer(tokenizer_path: str = "") -> PreTrainedTokenizerFast:
    if tokenizer_path:
        base = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    else:
        base = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True)
    backend = getattr(base, "_tokenizer", None) or getattr(base, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError("expected a fast tokenizer backend")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=base.bos_token if base.bos_token else "<s>",
        eos_token=base.eos_token if base.eos_token else "</s>",
        unk_token=base.unk_token if base.unk_token else "<unk>",
        pad_token=base.pad_token if base.pad_token else "<pad>",
        mask_token=getattr(base, "mask_token", None) or "<mask>",
        model_max_length=getattr(base, "model_max_length", 1024) or 1024,
    )
    tok.init_kwargs.pop("tokenizer_class", None)
    tok.init_kwargs["tokenizer_class"] = "PreTrainedTokenizerFast"
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})
    return tok


def force_portable_tokenizer_config(dst: Path) -> None:
    tok_cfg = dst / "tokenizer_config.json"
    if tok_cfg.exists():
        cfg = json.loads(tok_cfg.read_text(encoding="utf-8"))
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg.setdefault("bos_token", "<s>")
        cfg.setdefault("eos_token", "</s>")
        cfg.setdefault("unk_token", "<unk>")
        cfg.setdefault("pad_token", "<pad>")
        cfg.setdefault("mask_token", "<mask>")
        tok_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


@dataclass
class Example:
    text: str
    words: int
    example_id: int
    source: str
    kind: str
    mode: str
    source_text: str = ""
    target_text: str = ""
    source_pair_id: int = -1
    target_pair_id: int = -1
    true_target_pair_id: int = -1
    anchor_source_word_indices: list[int] | None = None
    anchor_norms: list[str] | None = None


def load_examples_jsonl(path: Path, selected_words: int) -> tuple[list[Example], int, int, list[dict[str, Any]]]:
    examples: list[Example] = []
    total_file_words = 0
    total_rows = 0
    sample_rows = []
    selected = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            total_rows += 1
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"word-count mismatch row {total_rows}: field={words} actual={actual}")
            total_file_words += words
            if selected < selected_words:
                if selected + words > selected_words:
                    raise RuntimeError(f"partial JSONL example would be needed at row {total_rows}: selected={selected} words={words} target={selected_words}")
                ex = Example(
                    text=text,
                    words=words,
                    example_id=int(obj.get("example_id", total_rows - 1)),
                    source=str(obj.get("source", "example_jsonl")),
                    kind=str(obj.get("kind", "official")),
                    mode=str(obj.get("mode", "")),
                    source_text=str(obj.get("source_text", "")),
                    target_text=str(obj.get("target_text", "")),
                    source_pair_id=int(obj.get("source_pair_id", -1)),
                    target_pair_id=int(obj.get("target_pair_id", -1)),
                    true_target_pair_id=int(obj.get("true_target_pair_id", obj.get("source_pair_id", -1))),
                    anchor_source_word_indices=[int(x) for x in obj.get("anchor_source_word_indices", [])],
                    anchor_norms=[str(x) for x in obj.get("anchor_norms", [])],
                )
                examples.append(ex)
                selected += words
                if len(sample_rows) < 10:
                    sample_rows.append({k: obj[k] for k in obj.keys() if k not in ("text", "source_text", "target_text", "true_target_text")})
    if selected != selected_words:
        raise RuntimeError(f"selected words {selected} != target {selected_words}")
    return examples, total_file_words, total_rows, sample_rows


class CrossViewDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start: dict[int, bool] = {}

    def _word_start_flag(self, token_id: int) -> bool:
        v = self._word_start.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start[token_id] = v
        return v

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
        anchor_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        anchor_group_count = 0
        anchor_token_count = 0
        source_word_count = len(ex.source_text.split()) if ex.kind == "pair_crossview" else 0
        if ex.kind == "pair_crossview" and ex.anchor_source_word_indices:
            desired = {int(x) for x in ex.anchor_source_word_indices if int(x) >= 0}
            for g in sorted(desired):
                if g < source_word_count:
                    hit = (group == g) & attention_mask.bool()
                    if bool(hit.any()):
                        anchor_mask |= hit
                        anchor_group_count += 1
                        anchor_token_count += int(hit.sum().item())
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": group,
            "anchor_mask": anchor_mask,
            "words": torch.tensor(ex.words, dtype=torch.long),
            "is_pair": torch.tensor(1 if ex.kind == "pair_crossview" else 0, dtype=torch.long),
            "has_anchor": torch.tensor(1 if anchor_token_count > 0 else 0, dtype=torch.long),
            "anchor_group_count": torch.tensor(anchor_group_count, dtype=torch.long),
            "anchor_token_count": torch.tensor(anchor_token_count, dtype=torch.long),
            "example_id": torch.tensor(ex.example_id, dtype=torch.long),
            "source_pair_id": torch.tensor(ex.source_pair_id, dtype=torch.long),
            "target_pair_id": torch.tensor(ex.target_pair_id, dtype=torch.long),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0].keys()
    return {k: torch.stack([x[k] for x in batch]) for k in keys}


def apply_crossview_masking(input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor,
                            anchor_mask: torch.Tensor, tokenizer, mask_prob: float, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)
    crossview_rows = 0
    crossview_tokens = 0
    wwm_rows = 0
    wwm_tokens = 0
    for b in range(bsz):
        am = anchor_mask[b].to(device) & candidate[b]
        if bool(am.any()):
            select[b] = am
            crossview_rows += 1
            crossview_tokens += int(am.sum().item())
        else:
            groups = word_group[b]
            valid_groups = torch.unique(groups[groups >= 0])
            if valid_groups.numel() == 0:
                continue
            gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
            chosen = valid_groups[gp < mask_prob]
            if chosen.numel() == 0:
                continue
            sel_b = torch.isin(groups, chosen) & candidate[b]
            select[b] = sel_b
            wwm_rows += 1
            wwm_tokens += int(sel_b.sum().item())
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
            wwm_tokens += 1
    labels[~select] = -100
    masked_inputs = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = tokenizer.mask_token_id
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids
    return masked_inputs, labels, {
        "crossview_rows": crossview_rows,
        "crossview_tokens": crossview_tokens,
        "wwm_rows": wwm_rows,
        "wwm_tokens": wwm_tokens,
        "total_selected_tokens": int(select.sum().item()),
    }


def summarize_tokenization_and_supervision(dataset: CrossViewDataset) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = {
        "num_examples": len(dataset),
        "total_words": 0,
        "pair_examples": 0,
        "official_examples": 0,
        "pair_examples_with_anchor_tokens": 0,
        "anchor_groups_total": 0,
        "anchor_tokens_total": 0,
        "truncated_examples_at_max_seq_length": 0,
        "total_untruncated_tokens": 0,
        "total_kept_tokens": 0,
        "total_tokens_lost_to_truncation": 0,
    }
    manifest = []
    for i, ex in enumerate(dataset.examples):
        ids = dataset.tokenizer(ex.text, add_special_tokens=False, truncation=False)["input_ids"]
        item = dataset[i]
        kept = int(item["attention_mask"].sum().item())
        lost = max(0, len(ids) - kept)
        summary["total_words"] += ex.words
        summary["total_untruncated_tokens"] += len(ids)
        summary["total_kept_tokens"] += kept
        summary["total_tokens_lost_to_truncation"] += lost
        if lost > 0:
            summary["truncated_examples_at_max_seq_length"] += 1
        if ex.kind == "pair_crossview":
            summary["pair_examples"] += 1
        else:
            summary["official_examples"] += 1
        ag = int(item["anchor_group_count"].item())
        at = int(item["anchor_token_count"].item())
        summary["anchor_groups_total"] += ag
        summary["anchor_tokens_total"] += at
        if at > 0:
            summary["pair_examples_with_anchor_tokens"] += 1
        manifest.append({
            "row_index": i,
            "example_id": ex.example_id,
            "kind": ex.kind,
            "mode": ex.mode,
            "words": ex.words,
            "source_pair_id": ex.source_pair_id,
            "target_pair_id": ex.target_pair_id,
            "anchor_group_count": ag,
            "anchor_token_count": at,
            "anchor_source_word_indices": ex.anchor_source_word_indices or [],
            "anchor_norms": ex.anchor_norms or [],
            "kept_tokens": kept,
            "untruncated_tokens": len(ids),
            "tokens_lost_to_truncation": lost,
        })
    summary["kept_tokens_per_word"] = summary["total_kept_tokens"] / max(1, summary["total_words"])
    summary["anchor_tokens_per_word"] = summary["anchor_tokens_total"] / max(1, summary["total_words"])
    return summary, manifest


def build_model(args, tokenizer):
    cfg = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * args.ffn_mult,
        max_position_embeddings=max(args.max_position_embeddings, args.max_seq_length + 8),
        pad_token_id=tokenizer.pad_token_id,
        type_vocab_size=1,
    )
    return BertForMaskedLM(cfg)


def save_hf_checkpoint(model, tokenizer, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    force_portable_tokenizer_config(dst)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--example_jsonl_label", default="")
    p.add_argument("--example_jsonl_meta", default="")
    p.add_argument("--max_word_exposure", type=int, required=True)
    p.add_argument("--example_pool_words", type=int, default=0)
    p.add_argument("--checkpoint_words", type=int, default=0)
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--tokenizer_label", default="baseline16k")
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra_init_seed", type=int, default=-1)
    p.add_argument("--train_rng_seed", type=int, default=-1)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = build_args()
    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected_words = args.max_word_exposure
    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    jsonl_path = Path(args.example_jsonl)
    examples, total_file_words, total_rows, sample_rows = load_examples_jsonl(jsonl_path, selected_words)
    actual_words = sum(ex.words for ex in examples)
    if actual_words != selected_words:
        raise RuntimeError(f"word selection mismatch {actual_words} vs {selected_words}")
    dataset = CrossViewDataset(examples, tokenizer, args.max_seq_length)
    supervision_summary, supervision_manifest = summarize_tokenization_and_supervision(dataset)
    (out / "crossview_supervision_summary.json").write_text(json.dumps(supervision_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "crossview_supervision_manifest.json").write_text(json.dumps(supervision_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    example_jsonl_meta = {}
    if args.example_jsonl_meta:
        meta_path = Path(args.example_jsonl_meta)
        example_jsonl_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        example_jsonl_meta["meta_path"] = str(meta_path)
        example_jsonl_meta["meta_sha256"] = sha256_file(meta_path)
    manifest_files = [{
        "path": str(jsonl_path),
        "name": jsonl_path.name,
        "bytes": jsonl_path.stat().st_size,
        "sha256": sha256_file(jsonl_path),
        "whitespace_words": total_file_words,
        "rows": total_rows,
    }]
    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    example_selection_metadata = {
        "data_source_type": "explicit_crossview_jsonl",
        "example_jsonl": str(jsonl_path),
        "example_jsonl_label": args.example_jsonl_label,
        "example_jsonl_total_words": total_file_words,
        "example_jsonl_total_rows": total_rows,
        "example_jsonl_sample_rows_without_text": sample_rows,
        "example_jsonl_meta": example_jsonl_meta,
    }
    (out / "example_order_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "selected_for_training_words": actual_words,
        "num_consumed_examples": len(examples),
        "source_words_consumed": source_words,
        "consumed_example_ids_in_order": [ex.example_id for ex in examples],
        "objective": "crossview_anchor_mask_plus_standard_wwm",
        "mask_prob_for_wwm_rows": args.mask_prob,
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_vocab_size": len(tokenizer),
        **example_selection_metadata,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=2, pin_memory=torch.cuda.is_available())
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)
    param_count = sum(p.numel() for p in model.parameters())
    input_embedding_params = model.get_input_embeddings().weight.numel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total_steps = len(loader)
    schedule_total = args.lr_total_steps if args.lr_total_steps and args.lr_total_steps > 0 else total_steps
    if schedule_total < total_steps:
        raise RuntimeError(f"lr_total_steps {schedule_total} < actual steps {total_steps}")
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    selected_tokens_total = 0
    crossview_tokens_total = 0
    wwm_tokens_total = 0
    crossview_rows_total = 0
    wwm_rows_total = 0
    saved_checkpoints = []
    checkpoint_words = args.checkpoint_words if args.checkpoint_words > 0 else args.max_word_exposure
    next_ckpt = checkpoint_words
    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch.pop("input_ids").to(device, non_blocking=True)
            attention_mask = batch.pop("attention_mask").to(device, non_blocking=True)
            word_group = batch.pop("word_group").to(device, non_blocking=True)
            anchor_mask = batch.pop("anchor_mask").to(device, non_blocking=True)
            # Remaining metadata tensors are only for manifests.
            masked_inputs, labels, st = apply_crossview_masking(input_ids, attention_mask, word_group, anchor_mask, tokenizer, args.mask_prob, gen)
            optim.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            loss = out_model.loss
            if loss is None:
                raise RuntimeError("model returned no loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            cumulative_words += words
            lf = float(loss.detach().cpu())
            loss_values.append(lf)
            selected_tokens_total += st["total_selected_tokens"]
            crossview_tokens_total += st["crossview_tokens"]
            wwm_tokens_total += st["wwm_tokens"]
            crossview_rows_total += st["crossview_rows"]
            wwm_rows_total += st["wwm_rows"]
            rec = {"step": step, "loss": lf, "lr": float(sched.get_last_lr()[0]), "batch_words": words,
                   "cumulative_word_exposure": cumulative_words, "selected_tokens": st["total_selected_tokens"],
                   "crossview_tokens": st["crossview_tokens"], "wwm_tokens": st["wwm_tokens"],
                   "crossview_rows": st["crossview_rows"], "wwm_rows": st["wwm_rows"],
                   "elapsed_sec": time.time() - start_time}
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            while cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                name = "chck_1M" if next_ckpt < 1_000_000 else (f"chck_{next_ckpt // 1_000_000}M" if next_ckpt % 1_000_000 == 0 else f"chck_{next_ckpt}w")
                cp = out / "hf_model" / name
                save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({"name": name, "target_word_exposure": next_ckpt, "actual_cumulative_word_exposure": cumulative_words, "path": str(cp)})
                print(json.dumps({"event": "checkpoint_saved", "name": name, "cum_words": cumulative_words}), flush=True)
                next_ckpt += checkpoint_words
    save_hf_checkpoint(model, tokenizer, out / "hf_model")
    if not saved_checkpoints:
        cp = out / "hf_model" / "chck_1M"
        save_hf_checkpoint(model, tokenizer, cp)
        saved_checkpoints.append({"name": "chck_1M", "target_word_exposure": checkpoint_words, "actual_cumulative_word_exposure": cumulative_words, "path": str(cp)})
    metrics = {
        "variant": "crossview_anchor_mask_plus_wwm",
        "backend": "mlm",
        "parameter_count": param_count,
        "embedding_parameter_count": input_embedding_params,
        "non_embedding_parameter_count": param_count - input_embedding_params,
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "word_exposure": cumulative_words,
        "selected_for_training_words": actual_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": total_steps,
        "lr_schedule_total_steps": schedule_total,
        "mask_prob_for_wwm_rows": args.mask_prob,
        "selected_tokens_total": selected_tokens_total,
        "selected_tokens_per_whitespace_word": selected_tokens_total / max(1, cumulative_words),
        "crossview_tokens_total": crossview_tokens_total,
        "wwm_tokens_total": wwm_tokens_total,
        "crossview_rows_total": crossview_rows_total,
        "wwm_rows_total": wwm_rows_total,
        "crossview_supervision_summary": supervision_summary,
        "seq_length": args.seq_length,
        "max_seq_length": args.max_seq_length,
        "max_position_embeddings": max(args.max_position_embeddings, args.max_seq_length + 8),
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "ffn_mult": args.ffn_mult,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "saved_checkpoints": saved_checkpoints,
        "source_words_consumed": source_words,
        **example_selection_metadata,
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "data_manifest.json").write_text(json.dumps({
        "files": manifest_files,
        "selected_for_this_run_whitespace_words": actual_words,
        "objective": "crossview_anchor_mask_plus_standard_wwm",
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_vocab_size": len(tokenizer),
        "crossview_supervision_summary_file": str(out / "crossview_supervision_summary.json"),
        "crossview_supervision_manifest_file": str(out / "crossview_supervision_manifest.json"),
        **example_selection_metadata,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "param_count": param_count, "loss_first": metrics["loss_first"], "loss_last": metrics["loss_last"], "word_exposure": cumulative_words, "selected_tokens_total": selected_tokens_total, "checkpoints": [c["name"] for c in saved_checkpoints]}), flush=True)


if __name__ == "__main__":
    main()
