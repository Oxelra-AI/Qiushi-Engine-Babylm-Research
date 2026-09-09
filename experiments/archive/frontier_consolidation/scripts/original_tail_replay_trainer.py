#!/usr/bin/env python3
"""research: original post-80M stream replay trainer.

Loads legal chck_80M, discards optimizer state, and continues on the exact rows
that the original legal continuous run consumed after saving chck_80M. This is
only to reproduce an observed research restart movement without the fresh data
order / token-concatenation presentation confound.

The loop mirrors the original masking_curriculum_trainer.py fixed-WWM recipe as
closely as possible while loading an existing checkpoint instead of random init:
- row examples from the frozen 100M JSONL, starting at parent_step * batch_size;
- each row tokenized independently, truncated/padded to seq_len=256;
- batch exposure is the sum of row word counts;
- fixed WWM 0.15 with the same torch-device RNG interface;
- AdamW betas=(0.9,0.98), wd=0.01, grad clipping 1.0;
- save at total exposure milestones 85M/90M/95M/100M.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

SEQ_LENGTH = 256
MASK_PROB = 0.15
BASE_LR = 0.001
WEIGHT_DECAY = 0.01
ORIGINAL_TOTAL_STEPS = 2529
WARMUP_FRACTION = 0.06
PARENT_EXPOSURE = 80_000_000  # nominal checkpoint name
PARENT_ACTUAL_WORDS = 80_034_368  # research training log cumulative_word_exposure at chck_80M
PARENT_STEP = 2024
BATCH_SIZE = 256
TARGET_TOTALS = [85_000_000, 90_000_000, 95_000_000, 100_000_000]
TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


@dataclass
class Example:
    text: str
    words: int
    row_index0: int
    source: str = ""
    example_id: int = -1


class RowDataset(Dataset):
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
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": group,
            "words": ex.words,
            "row_index0": ex.row_index0,
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
        "row_index0": torch.tensor([x["row_index0"] for x in batch], dtype=torch.long),
    }


def compute_matched_continuation_lr() -> float:
    warmup = int(ORIGINAL_TOTAL_STEPS * WARMUP_FRACTION)
    progress_80 = (ORIGINAL_TOTAL_STEPS * 0.8 - warmup) / (ORIGINAL_TOTAL_STEPS - warmup)
    factor = 0.5 * (1.0 + math.cos(math.pi * progress_80))
    return BASE_LR * factor


def apply_wwm_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor,
                   tokenizer, mask_prob: float, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=input_ids.device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)
    for b in range(bsz):
        groups = word_group[b]
        valid_groups = torch.unique(groups[groups >= 0])
        if valid_groups.numel() == 0:
            continue
        gp = torch.rand(valid_groups.numel(), generator=gen, device=input_ids.device)
        chosen = valid_groups[gp < mask_prob]
        if chosen.numel() == 0:
            continue
        select[b] = torch.isin(groups, chosen) & candidate[b]
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
    labels[~select] = -100
    masked = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=input_ids.device)
    mask_id = tokenizer.mask_token_id
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked[mask_tok] = mask_id
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=input_ids.device)
        masked[rand_tok] = rand_ids
    return masked, labels


def save_checkpoint(model, tokenizer, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    tok_cfg = dst / "tokenizer_config.json"
    if tok_cfg.exists():
        cfg = json.loads(tok_cfg.read_text(encoding="utf-8"))
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        tok_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_tail_examples(train_file: Path, row_start: int) -> tuple[list[Example], dict[str, Any]]:
    examples: list[Example] = []
    skipped_words = 0
    selected_words = 0
    source_words: dict[str, int] = {}
    first_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    with train_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch at row {idx}: {words} vs {len(text.split())}")
            if idx < row_start:
                skipped_words += words
                continue
            source = str(obj.get("source", "example_jsonl"))
            example_id = int(obj.get("example_id", idx))
            examples.append(Example(text=text, words=words, row_index0=idx, source=source, example_id=example_id))
            selected_words += words
            source_words[source] = source_words.get(source, 0) + words
            mini = {k: obj.get(k) for k in obj.keys() if k != "text"}
            mini["row_index0"] = idx
            if len(first_rows) < 5:
                first_rows.append(mini)
            last_rows.append(mini)
            if len(last_rows) > 5:
                last_rows.pop(0)
    return examples, {
        "row_start_index0": row_start,
        "skipped_words_before_tail": skipped_words,
        "tail_rows": len(examples),
        "tail_words": selected_words,
        "source_words_consumed": source_words,
        "first_tail_rows_no_text": first_rows,
        "last_tail_rows_no_text": last_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_checkpoint", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--lr_mode", required=True, choices=["matched", "base"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--train_rng_seed", type=int, default=43044)
    ap.add_argument("--row_start", type=int, default=PARENT_STEP * BATCH_SIZE)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--verify_train_hash", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    init_path = Path(args.init_checkpoint)
    train_file = Path(args.train_file)
    run_dir = Path(args.run_dir)
    model_root = run_dir / "hf_model"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    if not init_path.exists():
        raise FileNotFoundError(init_path)
    if not train_file.exists():
        raise FileNotFoundError(train_file)
    if args.verify_train_hash:
        actual = sha256_file(train_file)
        if actual != TRAIN_SHA:
            raise RuntimeError(f"train SHA mismatch {actual} != {TRAIN_SHA}")
        print(json.dumps({"event": "train_hash_ok", "sha256": actual}), flush=True)

    random.seed(args.train_rng_seed)
    torch.manual_seed(args.train_rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_rng_seed)

    tokenizer = AutoTokenizer.from_pretrained(str(init_path), use_fast=True)
    model = DebertaV2ForMaskedLM.from_pretrained(str(init_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    examples, tail_meta = load_tail_examples(train_file, args.row_start)
    dataset = RowDataset(examples, tokenizer, SEQ_LENGTH)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    total_steps = len(loader)

    if args.lr_mode == "matched":
        peak_lr = compute_matched_continuation_lr()
        warmup_steps = 0
        label = "original_tail_replay_matched_lr"
    else:
        peak_lr = BASE_LR
        warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))
        label = "original_tail_replay_base_lr"

    optim = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed)

    print(json.dumps({
        "event": "start", "label": label, "lr_mode": args.lr_mode,
        "init": str(init_path), "train_file": str(train_file), "run_dir": str(run_dir),
        "row_start": args.row_start, "tail_rows": tail_meta["tail_rows"], "tail_words": tail_meta["tail_words"],
        "total_steps": total_steps, "peak_lr": peak_lr, "warmup_steps": warmup_steps,
        "device": str(device), "started_utc": now(),
    }, indent=2), flush=True)

    log_path = run_dir / "training_log.jsonl"
    loss_values: list[float] = []
    saved: list[dict[str, Any]] = []
    cumulative_tail_words = 0
    next_target_idx = 0
    first_row_seen = None
    last_row_seen = None
    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            row_indices = batch.pop("row_index0")
            if first_row_seen is None:
                first_row_seen = int(row_indices[0].item())
            last_row_seen = int(row_indices[-1].item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            masked, labels = apply_wwm_mask(input_ids, attention_mask, word_group, tokenizer, MASK_PROB, gen)
            optim.zero_grad(set_to_none=True)
            out = model(input_ids=masked, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            if loss is None:
                raise RuntimeError("model returned no loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            cumulative_tail_words += words
            loss_float = float(loss.detach().cpu())
            loss_values.append(loss_float)
            total_exposure = PARENT_ACTUAL_WORDS + cumulative_tail_words
            total_exposure_nominal_parent = PARENT_EXPOSURE + cumulative_tail_words
            rec = {
                "step": step,
                "loss": loss_float,
                "lr": float(sched.get_last_lr()[0]),
                "batch_words": words,
                "continuation_words": cumulative_tail_words,
                "total_word_exposure_actual_parent": total_exposure,
                "total_word_exposure_nominal_parent": total_exposure_nominal_parent,
                "row_start_index0": int(row_indices[0].item()),
                "row_end_index0": int(row_indices[-1].item()),
                "masked_tokens": int((labels != -100).sum().item()),
                "effective_mask_rate": round(int((labels != -100).sum().item()) / max(1, int(attention_mask.sum().item())), 4),
            }
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            while next_target_idx < len(TARGET_TOTALS) and total_exposure >= TARGET_TOTALS[next_target_idx]:
                target = TARGET_TOTALS[next_target_idx]
                ckpt_name = f"chck_{target // 1_000_000}M"
                cp = model_root / ckpt_name
                save_checkpoint(model, tokenizer, cp)
                saved.append({
                    "name": ckpt_name,
                    "target_total_word_exposure": target,
                    "actual_total_word_exposure": total_exposure,
                    "nominal_parent_total_word_exposure": total_exposure_nominal_parent,
                    "continuation_words": cumulative_tail_words,
                    "step": step,
                    "path": str(cp),
                    "recent_loss50": sum(loss_values[-50:]) / len(loss_values[-50:]),
                })
                print(json.dumps({"event": "checkpoint_saved", "name": ckpt_name, "total_words": total_exposure, "step": step}), flush=True)
                next_target_idx += 1

    metrics = {
        "status": "ORIGINAL_TAIL_REPLAY_DONE",
        "label": label,
        "lr_mode": args.lr_mode,
        "peak_lr": peak_lr,
        "warmup_steps": warmup_steps,
        "parent_exposure_nominal": PARENT_EXPOSURE,
        "parent_actual_words": PARENT_ACTUAL_WORDS,
        "row_start": args.row_start,
        "tail_metadata": tail_meta,
        "total_steps": total_steps,
        "continuation_words": cumulative_tail_words,
        "total_word_exposure_nominal_parent": PARENT_EXPOSURE + cumulative_tail_words,
        "total_word_exposure_actual_parent": PARENT_ACTUAL_WORDS + cumulative_tail_words,
        "first_row_seen": first_row_seen,
        "last_row_seen": last_row_seen,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "loss_mean_last50": sum(loss_values[-50:]) / len(loss_values[-50:]) if loss_values else None,
        "saved_checkpoints": saved,
        "train_rng_seed": args.train_rng_seed,
        "finished_utc": now(),
    }
    (run_dir / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "train_command.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", **{k: v for k, v in metrics.items() if k != "saved_checkpoints"}}, indent=2), flush=True)
    print(f"Checkpoints saved: {[x['name'] for x in saved]}", flush=True)


if __name__ == "__main__":
    main()
