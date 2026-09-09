#!/usr/bin/env python3
"""research: Formation-stage replay experiment.

Runs private-only MLM formation from chck_82M using two different tail lengths:
  reference_4M : same 4M words as coherent86 (reproduces v4 formation)
  full_18M     : full remaining 18M legal words with natural schedule completion

Both arms use the same chck_82M starting point, same stream, same hyperparameters.
The ONLY difference is the tail length (and consequently, the number of updates and
where the LR schedule ends).

This directly tests whether more formation data helps private adapter learning.
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
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DebertaV2Config

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/frontier_consolidation')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(A02_SCRIPTS))

from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402

DEFAULT_ENDPOINT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
DEFAULT_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')


@dataclass
class Ex:
    text: str
    words: int
    source: str
    example_id: int
    row_index: int


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def reset_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def load_examples_tail(path: Path, skip_rows: int, max_words: int) -> list[Ex]:
    examples: list[Ex] = []
    selected = 0
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skip_rows:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch row={idx}")
            if selected + words > max_words:
                break
            examples.append(Ex(text=text, words=words, source=str(obj.get("source", "")),
                               example_id=int(obj.get("example_id", -1)), row_index=idx))
            selected += words
    return examples


class TailDataset(Dataset):
    def __init__(self, examples: list[Ex], tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._ws_cache: dict[int, bool] = {}

    def _word_start(self, tid: int) -> bool:
        v = self._ws_cache.get(tid)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and is_word_start(str(s)))
            self._ws_cache[tid] = v
        return v

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tokenizer(ex.text, add_special_tokens=False, truncation=True,
                             max_length=self.seq_length, padding="max_length",
                             return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        word_group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if attention_mask[i] == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._word_start(tid) or i == 0:
                gid += 1
            word_group[i] = gid
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "word_group": word_group, "words": ex.words,
                "example_id": ex.example_id, "row_index": ex.row_index,
                "source": ex.source}


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
        "example_id": torch.tensor([x["example_id"] for x in batch], dtype=torch.long),
        "row_index": torch.tensor([x["row_index"] for x in batch], dtype=torch.long),
        "source": [x["source"] for x in batch],
    }


def apply_wwm(input_ids, attention_mask, word_group, tokenizer, mask_prob, gen):
    B, S = input_ids.shape
    device = input_ids.device
    labels = input_ids.clone()
    mask_token_id = int(tokenizer.mask_token_id)
    special = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special)
    select = torch.zeros_like(candidate)
    for b in range(B):
        groups = word_group[b]
        valid = torch.unique(groups[groups >= 0])
        if valid.numel() > 0:
            r = torch.rand(valid.numel(), generator=gen, device=device)
            chosen = valid[r < mask_prob]
            if chosen.numel() > 0:
                select[b] = torch.isin(groups, chosen) & candidate[b]
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
    labels[~select] = -100
    masked = input_ids.clone()
    r = torch.rand(B, S, generator=gen, device=device)
    masked[select & (r < 0.8)] = mask_token_id
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    if rand_tok.any():
        masked[rand_tok] = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),),
                                         generator=gen, device=device)
    return masked, labels, select


def lr_at_update(upd, total, warmup, peak):
    if upd < warmup:
        return peak * upd / max(1, warmup)
    p = (upd - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(max(p, 0.0), 1.0)))


def compute_neutral_loss(model, input_ids, attention_mask, neutral_subsample):
    n = min(int(neutral_subsample), input_ids.shape[0])
    if n <= 0:
        return None, 0.0
    was_training = model.training
    model.eval()
    ids = input_ids[:n]
    att = attention_mask[:n]
    model.set_private_enabled(False)
    with torch.no_grad():
        slow_logits = model(input_ids=ids, attention_mask=att).logits.detach()
    model.set_private_enabled(True)
    out = model(input_ids=ids, attention_mask=att)
    log_p = F.log_softmax(out.logits, dim=-1)
    p_slow = F.softmax(slow_logits, dim=-1)
    kl = F.kl_div(log_p, p_slow, reduction="none").sum(-1)
    mask_f = att.float()
    loss = (kl * mask_f).sum() / max(1.0, float(mask_f.sum()))
    if was_training:
        model.train()
    return loss, float(loss.detach().cpu())


def save_checkpoint(model, tokenizer, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst), safe_serialization=True)
    tokenizer.save_pretrained(str(dst))
    src = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_private_modeling.py')
    d = dst / src.name
    if src.exists() and not d.exists():
        shutil.copy2(str(src), str(d))


def train(args) -> dict:
    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf_cache = out / "hf_cache"
    (hf_cache / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    endpoint = Path(args.endpoint)
    stream = Path(args.stream)

    print(json.dumps({"event": "start", "arm": args.arm,
                      "max_tail_words": args.max_tail_words,
                      "lr_total_steps": args.lr_total_steps,
                      "device": str(device)}), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(endpoint), local_files_only=True)
    examples = load_examples_tail(stream, args.skip_rows, args.max_tail_words)
    total_words = sum(e.words for e in examples)
    print(json.dumps({"event": "data_loaded", "rows": len(examples),
                      "words": total_words,
                      "first_row": examples[0].row_index if examples else None,
                      "last_row": examples[-1].row_index if examples else None}), flush=True)

    dataset = TailDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0, pin_memory=torch.cuda.is_available())

    reset_all(args.seed)
    cfg = DebertaV2Config.from_pretrained(str(endpoint), local_files_only=True)
    cfg.private_adapter_bottleneck = args.bottleneck
    cfg.private_adapter_scale = args.train_scale
    cfg.private_adapter_enabled = True
    model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
    sd = load_file(str(endpoint / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tied_missing = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}
    bad_missing = [x for x in missing if "private_adapter" not in x and x not in tied_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"load mismatch: bad_missing={bad_missing[:10]} unexpected={unexpected[:10]}")
    model.tie_weights()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    if torch.cuda.is_available():
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    pnames = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
    for n, p in model.named_parameters():
        p.requires_grad_(n in pnames)
    up_ids = {id(p) for n, p in model.named_parameters() if ".private_adapter.up." in n}
    p_normal = [p for n, p in model.named_parameters() if n in pnames and id(p) not in up_ids]
    p_zero_wd = [p for n, p in model.named_parameters() if n in pnames and id(p) in up_ids]
    optimizer = torch.optim.AdamW([
        {"params": p_normal, "weight_decay": args.weight_decay},
        {"params": p_zero_wd, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.98), eps=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    private_params = sum(p.numel() for n, p in model.named_parameters() if n in pnames)
    print(json.dumps({"event": "model_loaded", "total_params": total_params,
                      "private_params": private_params}), flush=True)

    schedule_total = args.lr_total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    # Save config
    config = {
        "status": "FORMATION_CONFIG",
        "arm": args.arm, "endpoint": rel(endpoint), "stream": rel(stream),
        "initial_consumed_words": args.initial_words, "max_tail_words": args.max_tail_words,
        "full_cap_words": args.full_cap, "skip_rows": args.skip_rows,
        "lr_total_steps": schedule_total, "warmup_steps": warmup,
        "batch_size": args.batch_size, "seq_length": args.seq_length,
        "lr": args.lr, "weight_decay": args.weight_decay, "mask_prob": args.mask_prob,
        "bottleneck": args.bottleneck, "train_scale": args.train_scale,
        "seed": args.seed, "total_params": total_params, "private_params": private_params,
        "tail_rows": len(examples), "tail_words": total_words,
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    losses_main, losses_neutral = [], []
    cum_main = 0
    updates = 0
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    log_path = out / "training_log.jsonl"

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch["words"].sum().item())
            tail_next = cum_main + words
            total_next = args.initial_words + tail_next
            if tail_next > args.max_tail_words or total_next > args.full_cap:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            masked, labels, _ = apply_wwm(input_ids, attention_mask, word_group,
                                          tokenizer, args.mask_prob, gen)
            n_targets = int((labels != -100).sum().item())

            lr = lr_at_update(updates, schedule_total, warmup, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            model.train()
            model.set_private_enabled(True)
            out_m = model(input_ids=masked, attention_mask=attention_mask)
            vocab = out_m.logits.shape[-1]
            main_loss = F.cross_entropy(out_m.logits.reshape(-1, vocab), labels.reshape(-1),
                                       ignore_index=-100, reduction="sum") / max(1, n_targets)
            ml_val = float(main_loss.detach().cpu())
            main_loss.backward()
            del out_m, main_loss

            nl_val = 0.0
            if args.neutral_lambda > 0:
                nl, nl_val = compute_neutral_loss(model, masked, attention_mask, 32)
                if nl is not None:
                    (args.neutral_lambda * nl).backward()
                    del nl

            torch.nn.utils.clip_grad_norm_(p_normal + p_zero_wd, 1.0)
            optimizer.step()

            updates += 1
            cum_main += words
            total_consumed = args.initial_words + cum_main
            losses_main.append(ml_val)
            losses_neutral.append(nl_val)

            rec = {"update": updates, "lr": round(lr, 8), "words": cum_main,
                   "total": total_consumed, "main_loss": round(ml_val, 6),
                   "neutral_loss": round(nl_val, 6), "targets": n_targets,
                   "elapsed": round(time.time() - t0, 1)}
            logf.write(json.dumps(rec) + "\n")
            if updates == 1 or updates % 25 == 0:
                print(json.dumps({"event": "train", **rec}), flush=True)

            while next_ckpt is not None and cum_main >= next_ckpt:
                approx_total = args.initial_words + next_ckpt
                name = f"chck_{approx_total // 1_000_000}M"
                save_checkpoint(model, tokenizer, out / "hf_model" / name)
                print(json.dumps({"event": "checkpoint", "name": name,
                                  "words": cum_main, "update": updates}), flush=True)
                next_ckpt += args.checkpoint_words
                if next_ckpt > args.max_tail_words:
                    next_ckpt = None

            del input_ids, attention_mask, word_group, masked, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model.set_private_enabled(True)
    save_checkpoint(model, tokenizer, out / "hf_model" / "final")

    metrics = {
        "status": "FORMATION_COMPLETE",
        "arm": args.arm, "updates": updates, "tail_words": cum_main,
        "total_consumed": args.initial_words + cum_main,
        "total_params": total_params, "private_params": private_params,
        "first_loss": losses_main[0] if losses_main else None,
        "final_loss": losses_main[-1] if losses_main else None,
        "mean_loss": sum(losses_main) / len(losses_main) if losses_main else None,
        "first_neutral": losses_neutral[0] if losses_neutral else None,
        "final_neutral": losses_neutral[-1] if losses_neutral else None,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["reference_4M", "full_18M"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--endpoint", default=str(DEFAULT_ENDPOINT))
    p.add_argument("--stream", default=str(DEFAULT_STREAM))
    p.add_argument("--output_dir", default="")
    p.add_argument("--initial_words", type=int, default=82_012_495)
    p.add_argument("--full_cap", type=int, default=100_000_000)
    p.add_argument("--skip_rows", type=int, default=530_944)
    p.add_argument("--lr_total_steps", type=int, default=455)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--warmup_fraction", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--bottleneck", type=int, default=128)
    p.add_argument("--train_scale", type=float, default=1.0)
    p.add_argument("--neutral_lambda", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=43023)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--max_tail_words", type=int, default=0,
                   help="Override tail words (0=use arm default)")
    args = p.parse_args()

    # Set arm-specific defaults
    if args.arm == "reference_4M":
        if not args.output_dir:
            args.output_dir = str(_public_path('experiments/archive/functional_learning/data/formation_experiment/reference_4M'))
        if args.max_tail_words <= 0:
            args.max_tail_words = 3_992_918  # Same as research coherent86
    elif args.arm == "full_18M":
        if not args.output_dir:
            args.output_dir = str(_public_path('experiments/archive/functional_learning/data/formation_experiment/full_18M'))
        if args.max_tail_words <= 0:
            args.max_tail_words = 17_987_505  # Full remaining budget
    else:
        raise ValueError(f"Unknown arm: {args.arm}")

    train(args)


if __name__ == "__main__":
    main()
