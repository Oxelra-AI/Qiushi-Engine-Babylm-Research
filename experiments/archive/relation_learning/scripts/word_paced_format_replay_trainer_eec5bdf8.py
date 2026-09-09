#!/usr/bin/env python3
"""research corrected word-paced format-alignment replay trainer.

This repairs the research screen before any score is read from it.  The corrected
trainer keeps the scientific contrast from research/097 but makes two invariants
explicit:

  * whole-word masking is derived from tokenizer word-start tokens (Ġ/▁), not
    character offsets, so roughly 15% of non-special tokens become targets rather
    than entire short rows;
  * private-on and private-off behavior are compared in eval mode at update 1,
    and the coherent neutral readout is measured on a fixed no-gradient set
    separate from the coherent set used as the preservation leash.

The run remains word-paced to coherent86's private suffix:
  - 3,992,800 legal suffix words
  - approximately 101 macro-updates of about 39.5K words each
  - dynamic padding for short isolated rows
  - frozen slow path plus 995,584 trainable private-adapter parameters
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import pathlib
import random
import shutil
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

_SCRIPT = _public_path('experiments/archive/relation_learning/scripts/word_paced_format_replay_trainer_eec5bdf8.py')
_ROOT = _public_path('.')

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(_public_path('experiments/archive/frontier_consolidation/scripts')))
from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402

CHCK82 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')

COHERENT86_WORDS = 3_992_800
COHERENT86_UPDATES = 101
WORDS_PER_UPDATE = COHERENT86_WORDS // COHERENT86_UPDATES
LR_TOTAL_STEPS = 455
WARMUP_FRACTION = 0.06


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(_ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_private_enabled(model, enabled: bool) -> None:
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(bool(enabled))
    elif hasattr(model, "config"):
        model.config.private_adapter_enabled = bool(enabled)


def load_frozen_model(chck_path: pathlib.Path, private_scale: float, device: torch.device):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, DebertaV2Config

    tokenizer = AutoTokenizer.from_pretrained(str(chck_path), use_fast=True, local_files_only=True)
    config = DebertaV2Config.from_pretrained(str(chck_path))
    config.private_adapter_bottleneck = int(getattr(config, "private_adapter_bottleneck", getattr(config, "adapter_bottleneck", 128)))
    config.private_adapter_scale = float(private_scale)
    config.private_adapter_enabled = True
    config.private_adapter_activation = getattr(config, "private_adapter_activation", getattr(config, "adapter_activation", "gelu"))

    model = FrozenSlowPrivateDebertaV2ForMaskedLM(config)
    state_dict = load_file(str(chck_path / "model.safetensors"), device="cpu")
    model.load_state_dict(state_dict, strict=False)

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = 0
    for name, p in model.named_parameters():
        if "private_adapter" in name:
            p.requires_grad_(True)
            trainable += p.numel()
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    return model, tokenizer, total, trainable


def load_jsonl_rows(path: pathlib.Path | str, max_words: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    with pathlib.Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if max_words and total + words > max_words and rows:
                break
            rows.append({"text": text, "words": words, "source": str(obj.get("source", ""))})
            total += words
            if max_words and total >= max_words:
                break
    return rows


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


class WordGrouper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = set(int(x) for x in tokenizer.all_special_ids)
        self.cache: dict[int, bool] = {}

    def word_start(self, tid: int) -> bool:
        v = self.cache.get(int(tid))
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and is_word_start(str(s)))
            self.cache[int(tid)] = v
        return v

    def tokenize(self, text: str, max_length: int = 256, add_special_tokens: bool = True) -> tuple[list[int], list[int]]:
        enc = self.tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=add_special_tokens)
        ids = [int(x) for x in enc["input_ids"]]
        groups = [-1] * len(ids)
        gid = -1
        first_non_special_seen = False
        for i, tid in enumerate(ids):
            if tid in self.special_ids:
                continue
            if gid < 0 or self.word_start(tid) or not first_non_special_seen:
                gid += 1
            groups[i] = gid
            first_non_special_seen = True
        return ids, groups


def collate_dynamic(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    mx = max(len(b["ids"]) for b in batch)
    n = len(batch)
    ids = torch.full((n, mx), int(pad_id), dtype=torch.long)
    att = torch.zeros((n, mx), dtype=torch.long)
    wg = torch.full((n, mx), -1, dtype=torch.long)
    words = torch.zeros(n, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["ids"])
        ids[i, :L] = torch.tensor(b["ids"], dtype=torch.long)
        att[i, :L] = 1
        wg[i, :L] = torch.tensor(b["wg"], dtype=torch.long)
        words[i] = int(b["words"])
    return {"input_ids": ids, "attention_mask": att, "word_group": wg, "words": words}


def apply_wwm(input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor,
              tokenizer, mask_prob: float, gen: torch.Generator):
    """research-style WWM: sample word groups independently, not one per row."""
    device = input_ids.device
    B, S = input_ids.shape
    labels = input_ids.clone()
    special = torch.tensor(sorted(int(x) for x in tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & (word_group >= 0) & ~torch.isin(input_ids, special)
    select = torch.zeros_like(candidate)
    for b in range(B):
        groups = word_group[b]
        valid = torch.unique(groups[groups >= 0])
        if valid.numel() <= 0:
            continue
        r = torch.rand(valid.numel(), generator=gen, device=device)
        chosen = valid[r < mask_prob]
        if chosen.numel() > 0:
            select[b] = torch.isin(groups, chosen) & candidate[b]
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[int(flat[0, 0])] = True
    labels[~select] = -100
    masked = input_ids.clone()
    r = torch.rand(B, S, generator=gen, device=device)
    mask_token_id = int(tokenizer.mask_token_id)
    masked[select & (r < 0.8)] = mask_token_id
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    if rand_tok.any():
        masked[rand_tok] = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=device)
    return masked, labels, select, candidate


def lr_at_update(step: int, total_steps: int, warmup_steps: int, peak_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def ce_value(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    vocab = out.logits.shape[-1]
    loss_sum = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="sum")
    targets = int((labels != -100).sum().item())
    value = float((loss_sum / max(1, targets)).detach().cpu())
    return loss_sum, value


def compute_eval_ce_pair(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    set_private_enabled(model, False)
    with torch.no_grad():
        _slow_sum, slow_val = ce_value(model, input_ids, attention_mask, labels)
    set_private_enabled(model, True)
    with torch.no_grad():
        _on_sum, on_val = ce_value(model, input_ids, attention_mask, labels)
    if was_training:
        model.train()
    return slow_val, on_val


def compute_neutral_loss(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device,
                         subsample: int = 64, require_grad: bool = True) -> tuple[torch.Tensor | None, float]:
    n = min(int(subsample), int(input_ids.shape[0]))
    if n <= 0:
        return None, 0.0
    was_training = model.training
    model.eval()
    ids = input_ids[:n].to(device)
    att = attention_mask[:n].to(device)
    set_private_enabled(model, False)
    with torch.no_grad():
        slow_logits = model(input_ids=ids, attention_mask=att).logits.detach()
    set_private_enabled(model, True)
    if require_grad:
        out = model(input_ids=ids, attention_mask=att)
        private_logits = out.logits
    else:
        with torch.no_grad():
            private_logits = model(input_ids=ids, attention_mask=att).logits
    log_p_private = F.log_softmax(private_logits, dim=-1)
    p_slow = F.softmax(slow_logits, dim=-1)
    kl_tok = F.kl_div(log_p_private, p_slow, reduction="none").sum(-1)
    mask_float = att.float()
    loss = (kl_tok * mask_float).sum() / max(1.0, float(mask_float.sum()))
    value = float(loss.detach().cpu())
    if not require_grad:
        loss = None
    if was_training:
        model.train()
    return loss, value


def build_macro_batches(tokenized: list[dict[str, Any]], words_per_update: int, max_updates: int = 0) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_words = 0
    for t in tokenized:
        cur.append(t)
        cur_words += int(t["words"])
        if cur_words >= words_per_update:
            batches.append(cur)
            cur = []
            cur_words = 0
            if max_updates and len(batches) >= max_updates:
                return batches
    if cur and (not max_updates or len(batches) < max_updates):
        batches.append(cur)
    return batches


def load_fixed_batch(path: str, grouper: WordGrouper, tokenizer, rows: int, max_length: int = 256):
    if not path:
        return None
    objs = load_jsonl_rows(path, max_words=0)[:rows]
    toks = []
    for r in objs:
        ids, wg = grouper.tokenize(r["text"], max_length=max_length, add_special_tokens=True)
        toks.append({"ids": ids, "wg": wg, "words": int(r["words"])})
    batch = collate_dynamic(toks, int(tokenizer.pad_token_id))
    return batch["input_ids"], batch["attention_mask"], sum(int(r["words"]) for r in objs)


def save_checkpoint(model, tokenizer, dst: pathlib.Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst), safe_serialization=True)
    tokenizer.save_pretrained(str(dst))
    src = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_private_modeling.py')
    if src.exists() and not (dst / src.name).exists():
        shutil.copy2(str(src), str(dst / src.name))


def train(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = out_dir / "hf_cache"
    (hf_cache / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_MODULES_CACHE"] = str(hf_cache / "modules")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(json.dumps({"event": "start", "arm": args.arm_label, "stream": rel(args.stream), "device": str(device), "seed": args.seed}), flush=True)

    model, tokenizer, total_params, trainable_params = load_frozen_model(pathlib.Path(args.endpoint), args.private_scale, device)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    print(json.dumps({"event": "model_loaded", "total_params": total_params, "trainable_params": trainable_params,
                      "private_scale": args.private_scale}), flush=True)

    train_max_words = int(args.max_train_words) if int(args.max_train_words) > 0 else 0
    rows = load_jsonl_rows(args.stream, max_words=train_max_words)
    total_words_loaded = sum(int(r["words"]) for r in rows)
    print(json.dumps({"event": "data_loaded", "rows": len(rows), "total_words_loaded": total_words_loaded,
                      "max_train_words": train_max_words}), flush=True)

    grouper = WordGrouper(tokenizer)
    tokenized: list[dict[str, Any]] = []
    for r in rows:
        ids, wg = grouper.tokenize(r["text"], max_length=args.max_length, add_special_tokens=True)
        tokenized.append({"ids": ids, "wg": wg, "words": int(r["words"])})

    leash_batch = load_fixed_batch(args.leash_coherent_stream, grouper, tokenizer, args.neutral_rows) if args.leash_coherent_stream else None
    readout_batch = load_fixed_batch(args.readout_coherent_stream, grouper, tokenizer, args.readout_rows) if args.readout_coherent_stream else None
    if leash_batch is not None:
        print(json.dumps({"event": "leash_coherent_loaded", "rows": args.neutral_rows, "words": leash_batch[2]}), flush=True)
    if readout_batch is not None:
        print(json.dumps({"event": "readout_coherent_loaded", "rows": args.readout_rows, "words": readout_batch[2]}), flush=True)

    max_updates = int(args.max_updates) if int(args.max_updates) > 0 else 0
    macro_batches = build_macro_batches(tokenized, WORDS_PER_UPDATE, max_updates=max_updates)
    n_updates = len(macro_batches)
    actual_total_words = sum(sum(int(t["words"]) for t in b) for b in macro_batches)
    warmup_steps = max(1, int(LR_TOTAL_STEPS * WARMUP_FRACTION))
    print(json.dumps({"event": "batching", "macro_updates": n_updates, "target_words_per_update": WORDS_PER_UPDATE,
                      "actual_total_words": actual_total_words,
                      "actual_mean_words_per_update": actual_total_words / max(1, n_updates),
                      "lr_total_steps": LR_TOTAL_STEPS, "warmup": warmup_steps}), flush=True)

    private_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(private_params, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.weight_decay)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 17)

    config = {
        "status": "FORMAT_REPLAY_CONFIG",
        "created_utc": now(),
        "arm_label": args.arm_label,
        "stream": rel(args.stream),
        "endpoint": rel(args.endpoint),
        "legal_scope": "same coherent86 suffix words; arm/stream define the intervention (row segmentation and/or official special-token exposure through this corrected trainer)",
        "total_words_loaded": total_words_loaded,
        "total_words_scheduled": actual_total_words,
        "macro_updates": n_updates,
        "words_per_update_target": WORDS_PER_UPDATE,
        "lr": args.lr,
        "lr_total_steps": LR_TOTAL_STEPS,
        "warmup_steps": warmup_steps,
        "private_scale": args.private_scale,
        "alpha_endpoint": args.alpha_endpoint,
        "neutral_lambda": args.neutral_lambda,
        "leash_coherent_stream": rel(args.leash_coherent_stream) if args.leash_coherent_stream else "",
        "readout_coherent_stream": rel(args.readout_coherent_stream) if args.readout_coherent_stream else "",
        "mask_prob": args.mask_prob,
        "micro_batch_size": args.micro_batch_size,
        "max_length": args.max_length,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "invariants": {
            "target_ratio_bounds": [args.min_target_ratio, args.max_target_ratio],
            "max_initial_private_slow_ce_absdiff": args.max_initial_ce_absdiff,
            "max_initial_main_ce": args.max_initial_main_ce,
            "max_initial_readout_kl": args.max_initial_readout_kl,
        },
    }
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cum_words = 0
    log_records: list[dict[str, Any]] = []
    model.train()
    pad_id = int(tokenizer.pad_token_id)
    log_path = out_dir / "training_log.jsonl"
    final_mean_ce = None
    final_readout_kl = None
    with log_path.open("w", encoding="utf-8") as logf:
        for update_idx, macro_batch in enumerate(macro_batches):
            lr = lr_at_update(update_idx, LR_TOTAL_STEPS, warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            macro_words = sum(int(t["words"]) for t in macro_batch)
            n_micro = math.ceil(len(macro_batch) / int(args.micro_batch_size))
            macro_non_special_for_scale = sum(sum(1 for g in t["wg"] if int(g) >= 0) for t in macro_batch)
            expected_targets_for_scale = max(1.0, float(args.mask_prob) * float(macro_non_special_for_scale))
            total_targets = 0
            total_non_special = 0
            total_ce_sum = 0.0
            init_slow_vals: list[float] = []
            init_on_vals: list[float] = []

            for mi in range(n_micro):
                start = mi * int(args.micro_batch_size)
                end = min(start + int(args.micro_batch_size), len(macro_batch))
                micro = macro_batch[start:end]
                batch = collate_dynamic(micro, pad_id)
                ids = batch["input_ids"].to(device, non_blocking=True)
                att = batch["attention_mask"].to(device, non_blocking=True)
                wg = batch["word_group"].to(device, non_blocking=True)
                masked, labels, _select, candidate = apply_wwm(ids, att, wg, tokenizer, float(args.mask_prob), gen)
                n_targets = int((labels != -100).sum().item())
                n_non_special = int(candidate.sum().item())
                if n_targets <= 0:
                    continue
                total_targets += n_targets
                total_non_special += n_non_special

                if update_idx == 0 and mi < int(args.initial_ce_micro_batches):
                    slow_val, on_val = compute_eval_ce_pair(model, masked, att, labels)
                    init_slow_vals.append(slow_val)
                    init_on_vals.append(on_val)

                model.train()
                set_private_enabled(model, True)
                out = model(input_ids=masked, attention_mask=att)
                vocab = out.logits.shape[-1]
                ce_sum = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="sum")
                # Approximate research's per-token loss while accumulating over many short-row micro-batches.
                # Accumulate one macro-update whose gradient is close to a research
                # batch loss averaged over all selected targets.  The denominator
                # uses the macro expected target count so short-row micro-batches do
                # not overweight small fragments.
                scaled_ce = ce_sum / expected_targets_for_scale
                (float(args.main_lambda) * scaled_ce).backward()
                total_ce_sum += float(ce_sum.detach().cpu())
                del out, ce_sum, scaled_ce, masked, labels, ids, att, wg

            target_ratio = total_targets / max(1, total_non_special)
            mean_ce = total_ce_sum / max(1, total_targets)
            if not (float(args.min_target_ratio) <= target_ratio <= float(args.max_target_ratio)):
                raise RuntimeError(f"target_ratio_out_of_range update={update_idx+1} ratio={target_ratio:.6f} targets={total_targets} non_special={total_non_special}")
            init_slow_ce = sum(init_slow_vals) / len(init_slow_vals) if init_slow_vals else None
            init_on_ce = sum(init_on_vals) / len(init_on_vals) if init_on_vals else None
            init_absdiff = abs(init_on_ce - init_slow_ce) if init_slow_ce is not None and init_on_ce is not None else None
            if update_idx == 0:
                if init_absdiff is not None and init_absdiff > float(args.max_initial_ce_absdiff):
                    raise RuntimeError(f"initial_private_slow_ce_diff_too_large slow={init_slow_ce} on={init_on_ce} diff={init_absdiff}")
                if mean_ce > float(args.max_initial_main_ce):
                    raise RuntimeError(f"initial_main_ce_too_high ce={mean_ce:.6f} targets={total_targets} ratio={target_ratio:.6f}")

            leash_kl_val = 0.0
            if leash_batch is not None and float(args.neutral_lambda) > 0.0:
                leash_loss, leash_kl_val = compute_neutral_loss(model, leash_batch[0], leash_batch[1], device,
                                                                subsample=int(args.neutral_subsample), require_grad=True)
                if leash_loss is not None:
                    (float(args.neutral_lambda) * leash_loss).backward()
                    del leash_loss

            torch.nn.utils.clip_grad_norm_(private_params, 1.0)
            optimizer.step()

            readout_kl_val = 0.0
            if readout_batch is not None:
                _no_loss, readout_kl_val = compute_neutral_loss(model, readout_batch[0], readout_batch[1], device,
                                                                subsample=int(args.readout_subsample), require_grad=False)
            if update_idx == 0 and readout_kl_val > float(args.max_initial_readout_kl):
                raise RuntimeError(f"initial_readout_kl_too_high kl={readout_kl_val:.8f}")

            cum_words += macro_words
            rec = {
                "update": update_idx + 1,
                "lr": lr,
                "macro_words": macro_words,
                "cum_words": cum_words,
                "macro_rows": len(macro_batch),
                "micro_batches": n_micro,
                "main_targets": total_targets,
                "non_special_tokens": total_non_special,
                "target_ratio": target_ratio,
                "main_ce": mean_ce,
                "initial_eval_slow_ce": init_slow_ce if update_idx == 0 else None,
                "initial_eval_private_on_ce": init_on_ce if update_idx == 0 else None,
                "initial_private_slow_ce_absdiff": init_absdiff if update_idx == 0 else None,
                "leash_neutral_kl": leash_kl_val,
                "readout_neutral_kl": readout_kl_val,
                "private_rms_max": max(model.private_adapter_rms()) if hasattr(model, "private_adapter_rms") else None,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logf.flush()
            log_records.append(rec)
            final_mean_ce = mean_ce
            final_readout_kl = readout_kl_val
            if update_idx == 0 or (update_idx + 1) % int(args.log_every) == 0:
                print(json.dumps({"event": "progress", "arm": args.arm_label, **rec}, ensure_ascii=False), flush=True)

    set_private_enabled(model, True)
    ckpt_dir = out_dir / "checkpoint"
    save_checkpoint(model, tokenizer, ckpt_dir)
    print(json.dumps({"event": "checkpoint_saved", "path": rel(ckpt_dir), "updates": n_updates, "cum_words": cum_words}), flush=True)

    alpha_dir = None
    if float(args.alpha_endpoint) != float(args.private_scale):
        alpha_dir = out_dir / f"alpha_{float(args.alpha_endpoint):.2f}"
        model.config.private_adapter_scale = float(args.alpha_endpoint)
        save_checkpoint(model, tokenizer, alpha_dir)
        print(json.dumps({"event": "alpha_endpoint", "alpha": args.alpha_endpoint, "path": rel(alpha_dir)}), flush=True)

    summary = {
        "status": "FORMAT_REPLAY_DONE",
        "created_utc": now(),
        "arm_label": args.arm_label,
        "updates": n_updates,
        "total_words": cum_words,
        "final_main_ce": final_mean_ce,
        "final_readout_neutral_kl": final_readout_kl,
        "first_update": log_records[0] if log_records else None,
        "last_update": log_records[-1] if log_records else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "checkpoint": rel(ckpt_dir),
        "alpha_endpoint_dir": rel(alpha_dir) if alpha_dir else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm-label", default="isolated_all")
    ap.add_argument("--endpoint", default=str(CHCK82))
    ap.add_argument("--leash-coherent-stream", default="")
    ap.add_argument("--readout-coherent-stream", default="")
    ap.add_argument("--neutral-rows", type=int, default=256)
    ap.add_argument("--readout-rows", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--private-scale", type=float, default=1.0)
    ap.add_argument("--alpha-endpoint", type=float, default=0.75)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--main-lambda", type=float, default=1.0)
    ap.add_argument("--neutral-lambda", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--micro-batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=97097)
    ap.add_argument("--max-train-words", type=int, default=0)
    ap.add_argument("--max-updates", type=int, default=0)
    ap.add_argument("--neutral-subsample", type=int, default=64)
    ap.add_argument("--readout-subsample", type=int, default=64)
    ap.add_argument("--initial-ce-micro-batches", type=int, default=3)
    ap.add_argument("--min-target-ratio", type=float, default=0.08)
    ap.add_argument("--max-target-ratio", type=float, default=0.24)
    ap.add_argument("--max-initial-ce-absdiff", type=float, default=0.05)
    ap.add_argument("--max-initial-main-ce", type=float, default=6.5)
    ap.add_argument("--max-initial-readout-kl", type=float, default=0.005)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
