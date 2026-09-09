#!/usr/bin/env python3
"""research: corrected macro-update bridge trainer for the legal relation overlay.

This replaces the exploratory research bridge runs, whose optimized loss was an
average of microbatch means and whose ordinary corruption stream differed between
arms.  The corrected trainer makes the optimized objective explicit over each
complete word-paced macro-update:

  L = lambda_rel * S_rel / N_rel + (1 - lambda_rel) * S_ord / N_ord

where S_rel/S_ord are cross-entropy sums over relation-row targets and ordinary-row
targets respectively within the same macro-update.  For the first corrected bridge
comparison, lambda_rel is the relation-word fraction in that macro-update.  Thus
sparse answer labels on relation rows do not silently reduce the relation component
to their token fraction.  Pooled token mean is also available as a transparent
alternative, with its induced lambda logged.

All ordinary rows use row-keyed deterministic whole-word masking.  Their corruption
is identical across arms regardless of whether intervening relation rows use WWM or
answer-span masking.  Relation answer spans must be fully covered by the tokenized
sequence; missing/truncated spans raise an error rather than falling back to MLM.
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
import pathlib
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/frontier_consolidation')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(A02_SCRIPTS))
sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))

import coherent86_continuation_trainer as base_loader  # noqa: E402

PARENT_PATH = _public_path('models/frontier')
MODELING_SRC = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_private_modeling.py')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def reset_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_seed(*parts: Any) -> int:
    txt = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.blake2b(txt.encode("utf-8"), digest_size=8).digest()
    # random.Random accepts arbitrary Python ints; keep it positive and bounded for logs.
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    kind = row.get("bridge_kind")
    if kind == "ordinary_tail":
        return ("ordinary_tail", row.get("orig_row_index"), row.get("example_id"), row.get("presentation_source_index"))
    return ("relation", row.get("pair_id"), row.get("row_type"), row.get("role"), row.get("relation_epoch"), row.get("relation_epoch_index"), row.get("presentation_index"))


def lr_at_update(update_index0: int, total: int, warmup: int, peak: float) -> float:
    if update_index0 < warmup:
        return peak * update_index0 / max(1, warmup)
    p = (update_index0 - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    return peak * 0.5 * (1.0 + math.cos(math.pi * p))


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


class WordGroupBuilder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = set(int(x) for x in tokenizer.all_special_ids)
        self.cache: Dict[int, bool] = {}

    def _word_start(self, tid: int) -> bool:
        v = self.cache.get(int(tid))
        if v is None:
            tok = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(tok is not None and is_word_start(str(tok)))
            self.cache[int(tid)] = v
        return v

    def groups(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        word_group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(int(input_ids.shape[0])):
            if int(attention_mask[i]) == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._word_start(tid) or i == 0:
                gid += 1
            word_group[i] = gid
        return word_group


def tokenize_row(row: Dict[str, Any], tokenizer, seq_length: int, wgb: WordGroupBuilder) -> Dict[str, Any]:
    enc = tokenizer(
        str(row.get("text", "")),
        add_special_tokens=False,
        truncation=True,
        max_length=int(seq_length),
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"].squeeze(0)
    attn = enc["attention_mask"].squeeze(0)
    offsets = enc["offset_mapping"].squeeze(0)
    return {
        "input_ids": ids,
        "attention_mask": attn,
        "offsets": offsets,
        "word_group": wgb.groups(ids, attn),
    }


def apply_wwm_row(input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor,
                  tokenizer, seed: int, mask_prob: float) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Deterministic row-keyed whole-word MLM masking.

    Unlike research's microbatch-level fallback, this row-level version permits a
    row to contribute zero labels if no word group is sampled.  Macro-updates have
    hundreds of rows, so the macro target count remains nonzero while row-keyed
    corruption stays identical across arms.
    """
    rng = random.Random(int(seed))
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    valid_groups = sorted(set(int(x) for x in word_group.tolist() if int(x) >= 0))
    chosen = [g for g in valid_groups if rng.random() < float(mask_prob)]
    chosen_set = set(chosen)
    selected_positions: List[int] = []
    for i in range(int(input_ids.shape[0])):
        if int(attention_mask[i]) == 0:
            continue
        if int(word_group[i]) in chosen_set:
            labels[i] = input_ids[i]
            selected_positions.append(i)
            p = rng.random()
            if p < 0.8:
                masked[i] = int(tokenizer.mask_token_id)
            elif p < 0.9:
                masked[i] = rng.randrange(int(tokenizer.vocab_size))
            # else keep original
    return masked, labels, {
        "n_word_groups": len(valid_groups),
        "n_selected_word_groups": len(chosen),
        "n_target_tokens": len(selected_positions),
        "zero_label_row": len(selected_positions) == 0,
    }


def locate_span_positions(offsets: torch.Tensor, start: int, end: int) -> List[int]:
    pos: List[int] = []
    for i in range(int(offsets.shape[0])):
        s, e = int(offsets[i][0]), int(offsets[i][1])
        if e <= s:
            continue
        if s < end and e > start:
            pos.append(i)
    return pos


def apply_answer_span_row(row: Dict[str, Any], input_ids: torch.Tensor, offsets: torch.Tensor,
                          tokenizer) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if "answer_char_span" not in row:
        raise RuntimeError(f"relation row lacks answer_char_span: {row_key(row)}")
    span = row["answer_char_span"]
    if not isinstance(span, Sequence) or len(span) != 2:
        raise RuntimeError(f"bad answer_char_span {span!r} for {row_key(row)}")
    start, end = int(span[0]), int(span[1])
    text = str(row.get("text", ""))
    ans = str(row.get("answer_text", ""))
    if start < 0 or end <= start or end > len(text):
        raise RuntimeError(f"answer span outside text for {row_key(row)}: span={span}, len={len(text)}")
    if text[start:end] != ans:
        raise RuntimeError(f"answer span text mismatch for {row_key(row)}: text_span={text[start:end]!r}, answer={ans!r}")
    pos = locate_span_positions(offsets, start, end)
    if not pos:
        raise RuntimeError(f"answer span truncated or unavailable for {row_key(row)}: span={span}, answer={ans!r}")
    covered_start = min(int(offsets[p][0]) for p in pos)
    covered_end = max(int(offsets[p][1]) for p in pos)
    if covered_start > start or covered_end < end:
        raise RuntimeError(f"answer span partial for {row_key(row)}: span={(start, end)}, covered={(covered_start, covered_end)}, answer={ans!r}")
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    for p in pos:
        labels[p] = input_ids[p]
        masked[p] = int(tokenizer.mask_token_id)
    return masked, labels, {
        "n_target_tokens": len(pos),
        "zero_label_row": len(pos) == 0,
        "answer_text": ans,
    }


def prepare_targets_for_macro(rows: List[Dict[str, Any]], tokenizer, seq_length: int, wgb: WordGroupBuilder,
                              arm: str, train_seed: int, mask_prob: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    stats = {
        "relation_rows": 0,
        "ordinary_rows": 0,
        "relation_words": 0,
        "ordinary_words": 0,
        "relation_target_tokens": 0,
        "ordinary_target_tokens": 0,
        "relation_zero_label_rows": 0,
        "ordinary_zero_label_rows": 0,
        "relation_selected_word_groups": 0,
        "ordinary_selected_word_groups": 0,
        "answer_span_rows": 0,
        "wwm_relation_rows": 0,
        "wwm_ordinary_rows": 0,
    }
    for presentation_i, row in enumerate(rows):
        is_rel = row.get("bridge_kind") == "relation_answer_packet"
        tok = tokenize_row(row, tokenizer, seq_length, wgb)
        words = int(row.get("words", len(str(row.get("text", "")).split())))
        if is_rel:
            stats["relation_rows"] += 1
            stats["relation_words"] += words
        else:
            stats["ordinary_rows"] += 1
            stats["ordinary_words"] += words

        if is_rel and arm == "answer_allocation":
            masked, labels, mstats = apply_answer_span_row(row, tok["input_ids"], tok["offsets"], tokenizer)
            component = "relation"
            stats["answer_span_rows"] += 1
        else:
            stream = "ordinary_stream" if not is_rel else "relation_stream"
            seed = stable_seed("wwm", stream, int(train_seed), row_key(row))
            masked, labels, mstats = apply_wwm_row(
                tok["input_ids"], tok["attention_mask"], tok["word_group"], tokenizer, seed, mask_prob)
            component = "relation" if is_rel else "ordinary"
            if is_rel:
                stats["wwm_relation_rows"] += 1
                stats["relation_selected_word_groups"] += int(mstats.get("n_selected_word_groups", 0))
            else:
                stats["wwm_ordinary_rows"] += 1
                stats["ordinary_selected_word_groups"] += int(mstats.get("n_selected_word_groups", 0))

        n_targets = int((labels != -100).sum().item())
        if is_rel:
            stats["relation_target_tokens"] += n_targets
            stats["relation_zero_label_rows"] += int(n_targets == 0)
        else:
            stats["ordinary_target_tokens"] += n_targets
            stats["ordinary_zero_label_rows"] += int(n_targets == 0)
        examples.append({
            "input_ids": masked,
            "attention_mask": tok["attention_mask"],
            "labels": labels,
            "component": component,
            "is_relation": bool(is_rel),
            "row_key": row_key(row),
            "words": words,
            "presentation_i": presentation_i,
        })
    return examples, stats


def load_model(device: torch.device, private_scale: float = 0.75):
    model, missing, unexpected = base_loader.load_model(PARENT_PATH, device, 128, float(private_scale))
    bad_missing_private = [k for k in missing if "private_adapter" in k]
    if bad_missing_private:
        raise RuntimeError(f"Missing private-adapter keys when loading parent: {bad_missing_private[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading parent: {unexpected[:10]}")
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(True)
    return model, list(missing), list(unexpected)


def freeze_to_private_optimizer(model, lr: float, wd: float):
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    private_names = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
    if not private_names:
        raise RuntimeError("No private_adapter parameters found")
    for n, p in model.named_parameters():
        p.requires_grad_(n in private_names)
    private_up_ids = {id(p) for n, p in model.named_parameters() if n in private_names and ".private_adapter.up." in n}
    normal = [p for n, p in model.named_parameters() if n in private_names and id(p) not in private_up_ids]
    zero_wd = [p for n, p in model.named_parameters() if n in private_names and id(p) in private_up_ids]
    opt = torch.optim.AdamW([
        {"params": normal, "weight_decay": float(wd)},
        {"params": zero_wd, "weight_decay": 0.0},
    ], lr=float(lr), betas=(0.9, 0.98), eps=1e-6)
    return opt, {
        "trainable_tensors": sum(1 for _, p in model.named_parameters() if p.requires_grad),
        "trainable_params": sum(p.numel() for _, p in model.named_parameters() if p.requires_grad),
        "private_param_names": sorted(private_names),
    }


def model_identity(model) -> Dict[str, Any]:
    named = list(model.named_parameters())
    private = [(n, p) for n, p in named if ".private_adapter." in n]
    scales: List[float] = []
    try:
        scales = [float(layer.private_adapter.scale) for layer in model.deberta.encoder.layer]
    except Exception:
        pass
    return {
        "class": type(model).__name__,
        "module": type(model).__module__,
        "total_params": int(sum(p.numel() for _, p in named)),
        "private_adapter_tensors": len(private),
        "private_adapter_params": int(sum(p.numel() for _, p in private)),
        "executed_private_scales": scales,
    }


def save_checkpoint(model, tokenizer, ckpt_dir: pathlib.Path, metadata: Dict[str, Any], private_scale: float):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model.config, "private_adapter_scale"):
        model.config.private_adapter_scale = float(private_scale)
    try:
        for layer in model.deberta.encoder.layer:
            layer.private_adapter.scale = float(private_scale)
    except Exception:
        pass
    model.eval()
    model.save_pretrained(str(ckpt_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(ckpt_dir))
    if MODELING_SRC.exists():
        dst = ckpt_dir / MODELING_SRC.name
        if not dst.exists():
            shutil.copy2(str(MODELING_SRC), str(dst))
    (ckpt_dir / "bridge_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    model.train()


def relation_lambda_for_macro(args: argparse.Namespace, stats: Dict[str, Any]) -> float:
    rel_words = float(stats["relation_words"])
    ord_words = float(stats["ordinary_words"])
    rel_targets = float(stats["relation_target_tokens"])
    ord_targets = float(stats["ordinary_target_tokens"])
    if args.loss_mode == "explicit_word_fraction":
        denom = rel_words + ord_words
        return rel_words / denom if denom > 0 else 0.0
    if args.loss_mode == "explicit_fixed_lambda":
        return float(args.relation_lambda)
    if args.loss_mode == "pooled_token_mean":
        denom = rel_targets + ord_targets
        return rel_targets / denom if denom > 0 else 0.0
    raise ValueError(args.loss_mode)


def train(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "hf_cache"
    os.environ["HF_HOME"] = str(cache.resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    (cache / "modules").mkdir(parents=True, exist_ok=True)

    reset_all(int(args.train_seed))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    overlay = load_jsonl(pathlib.Path(args.overlay_jsonl))
    total_words = sum(int(r.get("words", len(str(r.get("text", "")).split()))) for r in overlay)
    relation_words_total = sum(int(r.get("words", 0)) for r in overlay if r.get("bridge_kind") == "relation_answer_packet")
    print(json.dumps({
        "event": "data_loaded",
        "rows": len(overlay),
        "relation_rows": sum(1 for r in overlay if r.get("bridge_kind") == "relation_answer_packet"),
        "ordinary_rows": sum(1 for r in overlay if r.get("bridge_kind") != "relation_answer_packet"),
        "total_words": total_words,
        "relation_words_total": relation_words_total,
    }), flush=True)

    model, missing, unexpected = load_model(device, args.private_scale)
    ident = model_identity(model)
    if ident["class"] != "FrozenSlowPrivateDebertaV2ForMaskedLM" or ident["private_adapter_params"] != 995584:
        raise RuntimeError(f"Bad model identity: {ident}")
    print(json.dumps({"event": "model_loaded", **ident, "missing_keys_count": len(missing), "unexpected_keys_count": len(unexpected)}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = WordGroupBuilder(tokenizer)
    opt, opt_info = freeze_to_private_optimizer(model, args.lr, args.weight_decay)
    print(json.dumps({"event": "optimizer", "trainable_tensors": opt_info["trainable_tensors"], "trainable_params": opt_info["trainable_params"]}), flush=True)

    if torch.cuda.is_available():
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    config = {
        "status": "CORRECTED_BRIDGE_CONFIG",
        "scientific_question": "Can relation-aligned state-selection credit be integrated into legal BabyLM continuation under matched macro-update loss accounting?",
        "arm": args.arm,
        "overlay_jsonl": rel(pathlib.Path(args.overlay_jsonl)),
        "parent_path": rel(PARENT_PATH),
        "objective": "complete-macro-update explicit relation/ordinary decomposition",
        "loss_mode": args.loss_mode,
        "ordinary_corruption": "row-keyed deterministic WWM shared across arms for ordinary_tail rows",
        "relation_corruption": "WWM on relation rows for ordinary_wwm; explicit answer-span masking for answer_allocation",
        "answer_span_policy": "missing/truncated/mismatched answer spans raise RuntimeError; no MLM fallback",
        "seq_length": int(args.seq_length),
        "add_special_tokens_for_training": False,
        "words_per_update": int(args.words_per_update),
        "max_updates": int(args.max_updates),
        "train_seed": int(args.train_seed),
        "schedule_total": int(args.schedule_total),
        "schedule_offset": int(args.schedule_offset),
        "warmup": int(args.warmup),
        "lr_peak": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "private_scale": float(args.private_scale),
        "model_identity": ident,
    }
    (out_dir / "bridge_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cursor = 0
    cum_words = 0
    cum_rel_words = 0
    cum_ord_words = 0
    cum_rel_targets = 0
    cum_ord_targets = 0
    logs: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    t0 = time.time()
    model.train()

    for update_i in range(int(args.max_updates)):
        rows: List[Dict[str, Any]] = []
        words = 0
        while cursor < len(overlay) and words < int(args.words_per_update):
            row = overlay[cursor]
            rows.append(row)
            words += int(row.get("words", len(str(row.get("text", "")).split())))
            cursor += 1
        if not rows:
            print(json.dumps({"event": "data_exhausted", "update": update_i}), flush=True)
            break

        schedule_idx = int(args.schedule_offset) + update_i
        lr = lr_at_update(schedule_idx, int(args.schedule_total), int(args.warmup), float(args.lr))
        for pg in opt.param_groups:
            pg["lr"] = lr

        examples, prep_stats = prepare_targets_for_macro(
            rows, tokenizer, int(args.seq_length), wgb, args.arm, int(args.train_seed), float(args.mask_prob))
        rel_n = int(prep_stats["relation_target_tokens"])
        ord_n = int(prep_stats["ordinary_target_tokens"])
        if prep_stats["ordinary_rows"] > 0 and ord_n <= 0:
            raise RuntimeError(f"No ordinary targets in macro-update {update_i}: {prep_stats}")
        if prep_stats["relation_rows"] > 0 and rel_n <= 0:
            raise RuntimeError(f"No relation targets in macro-update {update_i}: {prep_stats}")
        lam_rel = relation_lambda_for_macro(args, prep_stats)
        if prep_stats["relation_rows"] == 0:
            lam_rel = 0.0
        if prep_stats["ordinary_rows"] == 0:
            lam_rel = 1.0
        lam_ord = 1.0 - lam_rel

        opt.zero_grad(set_to_none=True)
        rel_sum_total = 0.0
        ord_sum_total = 0.0
        rel_seen = 0
        ord_seen = 0
        for mb_start in range(0, len(examples), int(args.micro_batch)):
            mb = examples[mb_start:mb_start + int(args.micro_batch)]
            input_ids = torch.stack([x["input_ids"] for x in mb]).to(device)
            attention_mask = torch.stack([x["attention_mask"] for x in mb]).to(device)
            labels = torch.stack([x["labels"] for x in mb]).to(device)
            component_rel = torch.stack([(x["labels"] != -100) if x["component"] == "relation" else torch.zeros_like(x["labels"], dtype=torch.bool) for x in mb]).to(device)
            component_ord = torch.stack([(x["labels"] != -100) if x["component"] == "ordinary" else torch.zeros_like(x["labels"], dtype=torch.bool) for x in mb]).to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            vocab = out.logits.shape[-1]
            ce = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="none").view_as(labels)
            rel_sum = ce[component_rel].sum() if component_rel.any() else torch.tensor(0.0, device=device)
            ord_sum = ce[component_ord].sum() if component_ord.any() else torch.tensor(0.0, device=device)
            loss = torch.tensor(0.0, device=device)
            if rel_n > 0 and lam_rel != 0.0:
                loss = loss + float(lam_rel) * rel_sum / float(rel_n)
            if ord_n > 0 and lam_ord != 0.0:
                loss = loss + float(lam_ord) * ord_sum / float(ord_n)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"bad loss at update {update_i}, mb_start {mb_start}")
            loss.backward()
            rel_sum_total += float(rel_sum.detach().cpu())
            ord_sum_total += float(ord_sum.detach().cpu())
            rel_seen += int(component_rel.sum().detach().cpu())
            ord_seen += int(component_ord.sum().detach().cpu())
            del input_ids, attention_mask, labels, component_rel, component_ord, out, ce, rel_sum, ord_sum, loss

        if rel_seen != rel_n or ord_seen != ord_n:
            raise RuntimeError(f"macro target count mismatch update {update_i}: prepared rel/ord={rel_n}/{ord_n}, seen={rel_seen}/{ord_seen}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.max_grad_norm)).detach().cpu())
        opt.step()

        rel_loss = rel_sum_total / max(1, rel_n)
        ord_loss = ord_sum_total / max(1, ord_n)
        optimized_loss = lam_rel * rel_loss + lam_ord * ord_loss
        cum_words += words
        cum_rel_words += int(prep_stats["relation_words"])
        cum_ord_words += int(prep_stats["ordinary_words"])
        cum_rel_targets += rel_n
        cum_ord_targets += ord_n
        log = {
            "update": update_i + 1,
            "update_index0": update_i,
            "schedule_idx": schedule_idx,
            "lr": lr,
            "rows": len(rows),
            "words": words,
            "cum_words": cum_words,
            "relation_rows": int(prep_stats["relation_rows"]),
            "ordinary_rows": int(prep_stats["ordinary_rows"]),
            "relation_words": int(prep_stats["relation_words"]),
            "ordinary_words": int(prep_stats["ordinary_words"]),
            "relation_word_fraction": float(prep_stats["relation_words"] / max(1, words)),
            "loss_mode": args.loss_mode,
            "lambda_relation": float(lam_rel),
            "lambda_ordinary": float(lam_ord),
            "relation_target_tokens": rel_n,
            "ordinary_target_tokens": ord_n,
            "relation_token_fraction": float(rel_n / max(1, rel_n + ord_n)),
            "relation_loss": rel_loss,
            "ordinary_loss": ord_loss,
            "optimized_loss": optimized_loss,
            "grad_norm_preclip": grad_norm,
            "relation_zero_label_rows": int(prep_stats["relation_zero_label_rows"]),
            "ordinary_zero_label_rows": int(prep_stats["ordinary_zero_label_rows"]),
            "answer_span_rows": int(prep_stats["answer_span_rows"]),
            "wwm_relation_rows": int(prep_stats["wwm_relation_rows"]),
            "wwm_ordinary_rows": int(prep_stats["wwm_ordinary_rows"]),
            "cum_relation_targets": cum_rel_targets,
            "cum_ordinary_targets": cum_ord_targets,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        logs.append(log)
        if update_i == 0 or (update_i + 1) % int(args.log_every) == 0:
            print(json.dumps({"event": "update", **log}), flush=True)

        if ((update_i + 1) % int(args.checkpoint_every) == 0 or
                update_i + 1 == int(args.max_updates) or cursor >= len(overlay)):
            ckpt_name = f"update_{update_i + 1:04d}"
            ckpt_dir = out_dir / "checkpoints" / ckpt_name
            metadata = {
                "arm": args.arm,
                "update": update_i + 1,
                "schedule_idx": schedule_idx,
                "cum_words": cum_words,
                "cum_relation_words": cum_rel_words,
                "cum_ordinary_words": cum_ord_words,
                "cum_relation_targets": cum_rel_targets,
                "cum_ordinary_targets": cum_ord_targets,
                "loss_mode": args.loss_mode,
                "lambda_relation_last_update": float(lam_rel),
                "relation_loss_last_update": rel_loss,
                "ordinary_loss_last_update": ord_loss,
                "optimized_loss_last_update": optimized_loss,
                "train_seed": int(args.train_seed),
                "private_scale": float(args.private_scale),
                "model_identity": ident,
            }
            save_checkpoint(model, tokenizer, ckpt_dir, metadata, float(args.private_scale))
            checkpoints.append({"update": update_i + 1, "path": rel(ckpt_dir), "optimized_loss": optimized_loss})
            print(json.dumps({"event": "checkpoint", "update": update_i + 1, "path": rel(ckpt_dir)}), flush=True)
        if cursor >= len(overlay):
            print(json.dumps({"event": "overlay_exhausted", "update": update_i + 1}), flush=True)
            break

    with (out_dir / "update_log.jsonl").open("w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
    summary = {
        "status": "CORRECTED_BRIDGE_TRAIN_DONE",
        "arm": args.arm,
        "out_dir": rel(out_dir),
        "overlay_jsonl": rel(pathlib.Path(args.overlay_jsonl)),
        "parent_path": rel(PARENT_PATH),
        "loss_mode": args.loss_mode,
        "completed_updates": len(logs),
        "total_words_consumed": cum_words,
        "total_relation_words": cum_rel_words,
        "total_ordinary_words": cum_ord_words,
        "total_relation_targets": cum_rel_targets,
        "total_ordinary_targets": cum_ord_targets,
        "mean_relation_token_fraction": sum(float(x["relation_token_fraction"]) for x in logs) / max(1, len(logs)),
        "mean_lambda_relation": sum(float(x["lambda_relation"]) for x in logs) / max(1, len(logs)),
        "final_update": logs[-1] if logs else None,
        "checkpoints": checkpoints,
        "elapsed_sec": round(time.time() - t0, 1),
        "model_identity": ident,
        "scientific_status": "corrected interpretable macro-update training; research cancelled outputs are exploratory pilots only",
    }
    (out_dir / "bridge_train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlay-jsonl", required=True)
    ap.add_argument("--arm", required=True, choices=["ordinary_wwm", "answer_allocation"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--max-updates", type=int, default=354)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--schedule-total", type=int, default=455)
    ap.add_argument("--schedule-offset", type=int, default=101)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--train-seed", type=int, default=47047)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--loss-mode", choices=["explicit_word_fraction", "explicit_fixed_lambda", "pooled_token_mean"], default="explicit_word_fraction")
    ap.add_argument("--relation-lambda", type=float, default=0.12136590231805529)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
