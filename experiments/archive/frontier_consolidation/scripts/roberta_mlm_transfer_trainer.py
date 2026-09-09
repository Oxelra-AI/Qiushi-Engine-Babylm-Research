#!/usr/bin/env python3
"""research: RoBERTa/BERT-style bidirectional MLM trainer for compact-data transfer tests.

This is a deliberately narrow trainer scaffold, written after the DeBERTa-only
research compact ordered-vs-scrambled experiment was already running.  It is not a
new launched route by itself.  Its purpose is to make the next *model-coordinate*
transfer test cheap to start only if the matched compact-source-absent
results justify it.

Scientific role:
  - keep the same legal JSONL corpus accounting, tokenizer, seeds, WWM p=0.15,
    100M LR horizon, batch/seq geometry, and checkpoint format used by the
    DeBERTa compact experiments;
  - replace DeBERTa-v2 relative/disentangled attention with a stock absolute-
    position RoBERTa-like masked LM of comparable size;
  - produce ordinary HuggingFace checkpoints under hf_model/chck_* that the
    existing official-compatible selected evaluator can read.

The script supports preflight and CPU smoke-forward modes so the trainer can be
validated without consuming H100 training time.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    get_cosine_schedule_with_warmup,
)


TRAIN_MODEL_FAMILIES = ("roberta", "bert")


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
DEFAULT_TOKENIZER = WORKSPACE / "data" / "compliant_tokenizer"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_jsonl(path: Path) -> dict[str, Any]:
    rows = 0
    words = 0
    first_rows: list[dict[str, Any]] = []
    source_words: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows += 1
            text = str(obj["text"])
            w = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if w != actual:
                raise RuntimeError(f"word mismatch at row {rows}: field={w} actual={actual}")
            words += w
            source = str(obj.get("source", "example_jsonl"))
            source_words[source] = source_words.get(source, 0) + w
            if len(first_rows) < 5:
                meta = {k: v for k, v in obj.items() if k != "text"}
                meta["text_prefix"] = text[:160]
                first_rows.append(meta)
    return {"rows": rows, "words": words, "source_words": source_words, "first_rows": first_rows}


@dataclass
class Example:
    text: str
    words: int
    example_id: int
    source: str


def load_examples_jsonl(path: Path, selected_words: int, max_rows: int | None = None) -> tuple[list[Example], dict[str, Any]]:
    examples: list[Example] = []
    total_file_words = 0
    total_rows = 0
    selected = 0
    sample_rows: list[dict[str, Any]] = []
    source_words: dict[str, int] = {}
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
                raise RuntimeError(f"JSONL word-count mismatch at row {total_rows}: field={words} actual={actual}")
            total_file_words += words
            if selected < selected_words and (max_rows is None or len(examples) < max_rows):
                if selected + words > selected_words and max_rows is None:
                    raise RuntimeError(
                        f"JSONL selection would require partial example at row {total_rows}: "
                        f"selected={selected} words={words} target={selected_words}"
                    )
                ex_id = int(obj.get("example_id", total_rows - 1))
                source = str(obj.get("source", "example_jsonl"))
                examples.append(Example(text=text, words=words, example_id=ex_id, source=source))
                selected += words
                source_words[source] = source_words.get(source, 0) + words
                if len(sample_rows) < 10:
                    meta = {k: obj[k] for k in obj.keys() if k != "text"}
                    meta["text_prefix"] = text[:160]
                    sample_rows.append(meta)
            if max_rows is not None and len(examples) >= max_rows:
                # Smoke mode intentionally does not scan the whole file.
                break
    if max_rows is None and selected != selected_words:
        raise RuntimeError(f"JSONL selected words {selected} != target {selected_words}")
    meta = {
        "jsonl_total_words_scanned": total_file_words,
        "jsonl_total_rows_scanned": total_rows,
        "selected_words": selected,
        "selected_rows": len(examples),
        "sample_rows": sample_rows,
        "source_words_consumed": source_words,
    }
    return examples, meta


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def make_portable_tokenizer(tokenizer_path: str | Path) -> PreTrainedTokenizerFast:
    base = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    backend = getattr(base, "_tokenizer", None) or getattr(base, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError(f"Tokenizer at {tokenizer_path} has no fast backend")
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
        cfg = read_json(tok_cfg)
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg.setdefault("bos_token", "<s>")
        cfg.setdefault("eos_token", "</s>")
        cfg.setdefault("unk_token", "<unk>")
        cfg.setdefault("pad_token", "<pad>")
        cfg.setdefault("mask_token", "<mask>")
        write_json(tok_cfg, cfg)


class MaskedChunkDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer: PreTrainedTokenizerFast, seq_length: int):
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
            "words": torch.tensor(ex.words, dtype=torch.long),
            "example_id": torch.tensor(ex.example_id, dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0]}


def apply_wwm_masking(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    mask_prob: float,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)
    selected_groups = 0
    total_groups = 0
    for b in range(bsz):
        groups = word_group[b]
        valid_groups = torch.unique(groups[groups >= 0])
        if valid_groups.numel() == 0:
            continue
        total_groups += int(valid_groups.numel())
        gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
        chosen = valid_groups[gp < mask_prob]
        selected_groups += int(chosen.numel())
        if chosen.numel() == 0:
            continue
        select[b] = torch.isin(groups, chosen) & candidate[b]
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
    labels[~select] = -100
    masked_inputs = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = int(mask_token_id)
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids
    stats = {
        "masked_tokens": int(select.sum().item()),
        "candidate_tokens": int(candidate.sum().item()),
        "selected_groups": selected_groups,
        "total_groups": total_groups,
    }
    return masked_inputs, labels, stats


def reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace, tokenizer: PreTrainedTokenizerFast):
    max_pos = max(args.max_position_embeddings, args.max_seq_length + int(tokenizer.pad_token_id or 0) + 4)
    if args.model_family == "roberta":
        cfg = RobertaConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.n_layer,
            num_attention_heads=args.n_head,
            intermediate_size=args.hidden_size * args.ffn_mult,
            max_position_embeddings=max_pos,
            type_vocab_size=1,
            hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            layer_norm_eps=args.layer_norm_eps,
        )
        model = RobertaForMaskedLM(cfg)
    elif args.model_family == "bert":
        cfg = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.n_layer,
            num_attention_heads=args.n_head,
            intermediate_size=args.hidden_size * args.ffn_mult,
            max_position_embeddings=max_pos,
            type_vocab_size=1,
            hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            pad_token_id=tokenizer.pad_token_id,
            layer_norm_eps=args.layer_norm_eps,
        )
        model = BertForMaskedLM(cfg)
    else:
        raise ValueError(f"unknown model_family {args.model_family}")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model


def save_hf_checkpoint(model, tokenizer: PreTrainedTokenizerFast, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    force_portable_tokenizer_config(dst)


def summarize_recipe(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_family": args.model_family,
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "ffn_mult": args.ffn_mult,
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
        "layer_norm_eps": args.layer_norm_eps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "warmup_fraction": args.warmup_fraction,
        "weight_decay": args.weight_decay,
        "mask_prob": args.mask_prob,
        "max_word_exposure": args.max_word_exposure,
        "checkpoint_words": args.checkpoint_words,
        "lr_total_steps": args.lr_total_steps,
        "num_workers": args.num_workers,
    }


def preflight(args: argparse.Namespace) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    reset_rng(args.extra_init_seed if args.extra_init_seed >= 0 else args.seed)
    model = build_model(args, tokenizer)
    param_count = sum(p.numel() for p in model.parameters())
    data_summary = count_jsonl(args.example_jsonl)
    ok = data_summary["words"] >= args.max_word_exposure
    if data_summary["words"] != args.max_word_exposure:
        # For intended exact-prefix training files this should usually be exact.
        ok = ok and bool(args.allow_larger_jsonl)
    payload = {
        "status": "ROBERTA_TRANSFER_PREFLIGHT",
        "ok": ok,
        "created_utc": now_utc(),
        "scientific_purpose": "Dry validation for a possible post-compact-order model-coordinate transfer test; no H100 training is launched by this preflight.",
        "example_jsonl": str(args.example_jsonl),
        "example_jsonl_sha256": sha256_file(args.example_jsonl),
        "data_summary": {k: v for k, v in data_summary.items() if k != "source_words"},
        "source_word_types": len(data_summary["source_words"]),
        "source_words_top20": sorted(data_summary["source_words"].items(), key=lambda kv: -kv[1])[:20],
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_json_sha256": sha256_file(Path(args.tokenizer_path) / "tokenizer.json"),
        "vocab_size": len(tokenizer),
        "model_parameter_count": param_count,
        "model_config": model.config.to_dict(),
        "recipe": summarize_recipe(args),
        "intended_selected_checkpoints": checkpoint_names(args.max_word_exposure, args.checkpoint_words),
        "no_training_started": True,
    }
    write_json(out / "preflight.json", payload)
    print(json.dumps({"status": payload["status"], "ok": ok, "params": param_count, "words": data_summary["words"], "out": str(out / "preflight.json")}, indent=2), flush=True)
    if not ok:
        raise SystemExit(2)


def checkpoint_names(max_word_exposure: int, checkpoint_words: int) -> list[str]:
    if checkpoint_words <= 0:
        return []
    out: list[str] = []
    w = checkpoint_words
    while w <= max_word_exposure:
        if w < 1_000_000:
            out.append("chck_1M")
        elif w % 1_000_000 == 0:
            out.append(f"chck_{w // 1_000_000}M")
        else:
            out.append(f"chck_{w}w")
        w += checkpoint_words
    return out


def smoke_forward(args: argparse.Namespace) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    examples, data_meta = load_examples_jsonl(args.example_jsonl, selected_words=10**18, max_rows=args.smoke_rows)
    reset_rng(args.extra_init_seed if args.extra_init_seed >= 0 else args.seed)
    model = build_model(args, tokenizer)
    reset_rng(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)
    device = torch.device("cpu")
    model.to(device)
    model.train(False)
    dataset = MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, args.smoke_batch_size), shuffle=False, collate_fn=collate, num_workers=0)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    word_group = batch["word_group"].to(device)
    with torch.no_grad():
        masked_inputs, labels, mask_stats = apply_wwm_masking(input_ids, attention_mask, word_group, tokenizer, args.mask_prob, gen)
        out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
        loss = float(out_model.loss.detach().cpu()) if out_model.loss is not None else math.nan
        finite_logits = bool(torch.isfinite(out_model.logits).all().item())
    payload = {
        "status": "ROBERTA_TRANSFER_SMOKE_FORWARD",
        "created_utc": now_utc(),
        "example_jsonl": str(args.example_jsonl),
        "rows_loaded": len(examples),
        "data_meta": data_meta,
        "model_family": args.model_family,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(tokenizer),
        "mask_stats": mask_stats,
        "loss": loss,
        "finite_loss": math.isfinite(loss),
        "finite_logits": finite_logits,
        "device": "cpu",
        "no_training_started": True,
    }
    write_json(out / "smoke_forward.json", payload)
    print(json.dumps({"status": payload["status"], "finite_loss": payload["finite_loss"], "loss": loss, "masked_tokens": mask_stats["masked_tokens"], "out": str(out / "smoke_forward.json")}, indent=2), flush=True)
    if not (payload["finite_loss"] and payload["finite_logits"] and mask_stats["masked_tokens"] > 0):
        raise SystemExit(2)


def train(args: argparse.Namespace) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "train_command.json", {"created_utc": now_utc(), "argv": os.sys.argv, "recipe": summarize_recipe(args)})

    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    examples, data_meta = load_examples_jsonl(args.example_jsonl, selected_words=args.max_word_exposure, max_rows=None)
    actual_words = sum(ex.words for ex in examples)
    if actual_words != args.max_word_exposure:
        raise RuntimeError(f"actual_words {actual_words} != max_word_exposure {args.max_word_exposure}")

    dataset = MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    total_steps = len(loader)
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))

    reset_rng(args.seed)
    if args.extra_init_seed >= 0:
        reset_rng(args.extra_init_seed)
    model = build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        reset_rng(args.train_rng_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    write_json(out / "example_order_manifest.json", {
        "data_source_type": "example_jsonl",
        "example_jsonl": str(args.example_jsonl),
        "example_jsonl_label": args.example_jsonl_label,
        "selected_for_training_words": actual_words,
        "num_consumed_examples": len(examples),
        "source_words_consumed": source_words,
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_json_sha256": sha256_file(Path(args.tokenizer_path) / "tokenizer.json"),
        **data_meta,
    })

    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict[str, Any]] = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    start = time.time()

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            _example_ids = batch.pop("example_id")
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            input_ids = input_ids[:, :args.seq_length].contiguous()
            attention_mask = attention_mask[:, :args.seq_length].contiguous()
            word_group = word_group[:, :args.seq_length].contiguous()
            masked_inputs, labels, mask_stats = apply_wwm_masking(input_ids, attention_mask, word_group, tokenizer, args.mask_prob, gen)

            optim.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            loss = out_model.loss
            if loss is None:
                raise RuntimeError("model returned no loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()

            cumulative_words += words
            loss_float = float(loss.detach().cpu())
            loss_values.append(loss_float)
            rec = {
                "step": step,
                "loss": loss_float,
                "lr": float(sched.get_last_lr()[0]),
                "batch_words": words,
                "cumulative_word_exposure": cumulative_words,
                "seq_len": args.seq_length,
                "masked_tokens": mask_stats["masked_tokens"],
                "candidate_tokens": mask_stats["candidate_tokens"],
                "effective_mask_rate": round(mask_stats["masked_tokens"] / max(1, mask_stats["candidate_tokens"]), 4),
                "mask_mode": "wwm",
                "mask_prob_nominal": args.mask_prob,
                "elapsed_sec": round(time.time() - start, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)

            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                if next_ckpt < 1_000_000:
                    name = "chck_1M"
                elif next_ckpt % 1_000_000 == 0:
                    name = f"chck_{next_ckpt // 1_000_000}M"
                else:
                    name = f"chck_{next_ckpt}w"
                cp = out / "hf_model" / name
                save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({
                    "name": name,
                    "target_word_exposure": next_ckpt,
                    "actual_cumulative_word_exposure": cumulative_words,
                    "path": str(cp),
                })
                print(json.dumps({"event": "checkpoint_saved", "name": name, "cum_words": cumulative_words}), flush=True)
                next_ckpt += args.checkpoint_words

    save_hf_checkpoint(model, tokenizer, out / "hf_model")
    metrics = {
        "status": "ROBERTA_TRANSFER_TRAINING_DONE",
        "variant": f"step209_{args.model_family}_wwm_fixed",
        "backend": "mlm",
        "model_family": f"{args.model_family}_ForMaskedLM",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_path": str(args.tokenizer_path),
        "word_exposure": cumulative_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": total_steps,
        "optimizer": "AdamW betas=(0.9,0.98)",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_fraction": args.warmup_fraction,
        "lr_total_steps": schedule_total,
        "masking_curriculum": "wwm_fixed",
        "mask_prob_start": args.mask_prob,
        "mask_prob_end": args.mask_prob,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "seq_length": args.seq_length,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "ffn_mult": args.ffn_mult,
        "gradient_checkpointing": args.gradient_checkpointing,
        "saved_checkpoints": saved_checkpoints,
        "source_words_consumed": source_words,
        "data_source_type": "example_jsonl",
        "example_jsonl": str(args.example_jsonl),
        "example_jsonl_label": args.example_jsonl_label,
        "elapsed_sec": round(time.time() - start, 1),
    }
    write_json(out / "scientific_metrics.json", metrics)
    print(json.dumps({"event": "done", "status": metrics["status"], "param_count": metrics["parameter_count"], "loss_first": metrics["loss_first"], "loss_last": metrics["loss_last"], "word_exposure": cumulative_words, "checkpoints": [c["name"] for c in saved_checkpoints]}, indent=2), flush=True)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example_jsonl", type=Path, required=True)
    ap.add_argument("--example_jsonl_label", default="")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--tokenizer_path", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--tokenizer_label", default="compliant16k_reinvest10M")
    ap.add_argument("--model_family", choices=TRAIN_MODEL_FAMILIES, default="roberta")
    ap.add_argument("--hidden_size", type=int, default=480)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--ffn_mult", type=int, default=4)
    ap.add_argument("--max_position_embeddings", type=int, default=512)
    ap.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    ap.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    ap.add_argument("--layer_norm_eps", type=float, default=1e-5)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--extra_init_seed", type=int, default=43022)
    ap.add_argument("--train_rng_seed", type=int, default=43023)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seq_length", type=int, default=256)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--learning_rate", type=float, default=0.001)
    ap.add_argument("--warmup_fraction", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--max_word_exposure", type=int, default=40_000_000)
    ap.add_argument("--checkpoint_words", type=int, default=20_000_000)
    ap.add_argument("--lr_total_steps", type=int, default=2529)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--allow_larger_jsonl", action="store_true", help="Permit preflight when the JSONL is larger than selected exposure.")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--smoke-forward-only", action="store_true")
    ap.add_argument("--smoke-rows", type=int, default=4)
    ap.add_argument("--smoke-batch-size", type=int, default=2)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        preflight(args)
        return
    if args.smoke_forward_only:
        smoke_forward(args)
        return
    train(args)


if __name__ == "__main__":
    main()
