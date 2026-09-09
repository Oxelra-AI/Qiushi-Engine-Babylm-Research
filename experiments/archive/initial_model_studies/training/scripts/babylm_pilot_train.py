#!/usr/bin/env python3
"""BabyLM 2026 Strict-Small V0 pilot trainer.

Purpose: prove the official 2026 data -> HF causal checkpoint -> BabyLM eval path.
This is intentionally a small dense causal GPT-2-style pilot, not a SOTA model.

Writes all learned state and evidence to QIUSHI_AI_LAB_RUN_DIR.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup


TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]
BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id", default="BabyLM-community/BabyLM-2026-Strict-Small")
    p.add_argument("--dataset_revision", default="c92ab16b4f08858304b0815706065b3354d8fc0a")
    p.add_argument("--variant", choices=["dense_causal"], default="dense_causal")
    p.add_argument("--tokenizer", choices=["baseline16k"], default="baseline16k")
    p.add_argument("--max_word_exposure", type=int, default=1_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--words_per_example", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def run_dir() -> Path:
    rd = os.environ.get("QIUSHI_AI_LAB_RUN_DIR")
    if rd:
        return Path(rd)
    # Direct-run fallback for local debugging.
    return Path("experiments/archive/initial_model_studies/training/runs/direct_babylm_pilot_debug")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_words_in_file(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += len(line.split())
    return total


def download_dataset(args: argparse.Namespace, out: Path) -> tuple[Path, list[dict]]:
    raw_dir = out / "raw_dataset"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local = Path(
        snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            revision=args.dataset_revision,
            allow_patterns=TRAIN_FILES + ["README.md"],
            local_dir=raw_dir,
            local_dir_use_symlinks=False,
        )
    )
    manifest_files = []
    for name in TRAIN_FILES:
        p = local / name
        if not p.exists():
            raise FileNotFoundError(f"required training file missing after download: {p}")
        manifest_files.append(
            {
                "path": str(p),
                "name": name,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "whitespace_words": count_words_in_file(p),
            }
        )
    return local, manifest_files


@dataclass
class Example:
    text: str
    words: int


def iter_examples(files: list[Path], max_words: int, words_per_example: int) -> Iterable[Example]:
    used = 0
    buf: list[str] = []
    for fp in files:
        with fp.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for w in line.split():
                    if used >= max_words:
                        break
                    buf.append(w)
                    used += 1
                    if len(buf) >= words_per_example:
                        yield Example(" ".join(buf), len(buf))
                        buf = []
                if used >= max_words:
                    break
        if used >= max_words:
            break
    if buf:
        yield Example(" ".join(buf), len(buf))


class WordChunkDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
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
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "words": ex.words}


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
    }


def save_hf_checkpoint(model, tokenizer, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    # The organizer baseline tokenizer carries a non-importable tokenizer_class
    # (`TokenizersBackend`) through save_pretrained even after rewrapping. Force a
    # portable class name in the saved config so AutoTokenizer can reload local
    # checkpoints and future HF submissions without custom code.
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


def main() -> None:
    args = parse_args()
    out = run_dir()
    out.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(json.dumps({"event": "start", "run_dir": str(out), "args": vars(args)}, ensure_ascii=False), flush=True)

    raw_dir, manifest_files = download_dataset(args, out)
    total_words = sum(x["whitespace_words"] for x in manifest_files)
    selected_files = [raw_dir / name for name in TRAIN_FILES]
    selected_words = min(args.max_word_exposure, total_words)

    base_tokenizer = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True)
    # Always re-wrap as a portable PreTrainedTokenizerFast. The official baseline
    # tokenizer loads, but its saved init kwargs keep tokenizer_class="TokenizersBackend",
    # which AutoTokenizer cannot import from a clean checkpoint. Reconstructing from
    # the Rust backend and clearing class-specific init kwargs makes local and HF
    # checkpoints portable.
    from transformers import PreTrainedTokenizerFast

    fast_backend = getattr(base_tokenizer, "_tokenizer", None) or getattr(base_tokenizer, "backend_tokenizer", None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast_backend,
        bos_token=base_tokenizer.bos_token if base_tokenizer.bos_token else "<s>",
        eos_token=base_tokenizer.eos_token if base_tokenizer.eos_token else "</s>",
        unk_token=base_tokenizer.unk_token if base_tokenizer.unk_token else "<unk>",
        pad_token=base_tokenizer.pad_token if base_tokenizer.pad_token else "<pad>",
        mask_token=getattr(base_tokenizer, "mask_token", None) or "<mask>",
        model_max_length=1024,
    )
    tokenizer.init_kwargs.pop("tokenizer_class", None)
    tokenizer.init_kwargs["tokenizer_class"] = "PreTrainedTokenizerFast"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer_manifest = {
        "mode": args.tokenizer,
        "source_repo": BASELINE_TOKENIZER_REPO,
        "source_revision": "main",
        "vocab_size": len(tokenizer),
        "tokenizer_language_training_exposure_words": 0,
        "rule_note": "Uses official BabyLM 2026 baseline tokenizer; later SOTA runs should document whether this tokenizer is organizer-provided and accepted as a baseline artifact, or train tokenizer on the official 10M corpus.",
    }

    examples = list(iter_examples(selected_files, selected_words, args.words_per_example))
    actual_words = sum(ex.words for ex in examples)
    if actual_words != selected_words:
        raise RuntimeError(f"internal word selection mismatch: {actual_words} vs {selected_words}")
    # Deterministic order with tiny shuffle improves training without changing exposure.
    rng = random.Random(args.seed)
    rng.shuffle(examples)

    dataset = WordChunkDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=torch.cuda.is_available())

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.seq_length,
        n_ctx=args.seq_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config)
    param_count = sum(p.numel() for p in model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = len(loader)
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    checkpoint_saved = False
    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                "elapsed_sec": time.time() - start_time,
            }
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            if (not checkpoint_saved) and cumulative_words >= args.checkpoint_words:
                save_hf_checkpoint(model, tokenizer, out / "hf_model" / "chck_1M")
                checkpoint_saved = True
                print(json.dumps({"event": "checkpoint_saved", "path": str(out / "hf_model" / "chck_1M"), "cumulative_word_exposure": cumulative_words}), flush=True)

    save_hf_checkpoint(model, tokenizer, out / "hf_model")
    if not checkpoint_saved:
        save_hf_checkpoint(model, tokenizer, out / "hf_model" / "chck_1M")

    data_manifest = {
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "download_dir": str(raw_dir),
        "files": manifest_files,
        "total_official_dataset_whitespace_words_counted": total_words,
        "selected_for_this_run_whitespace_words": actual_words,
        "max_word_exposure_requested": args.max_word_exposure,
        "word_exposure_count_method": "Python str.split() over selected training text; examples truncated by word count before tokenization.",
    }
    (out / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "tokenizer_manifest.json").write_text(json.dumps(tokenizer_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "variant": args.variant,
        "status": "completed",
        "parameter_count": param_count,
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count(),
        "seq_length": args.seq_length,
        "words_per_example": args.words_per_example,
        "batch_size": args.batch_size,
        "steps": total_steps,
        "word_exposure": cumulative_words,
        "requested_max_word_exposure": args.max_word_exposure,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "loss_mean_last_20": sum(loss_values[-20:]) / min(20, len(loss_values)) if loss_values else None,
        "elapsed_sec": time.time() - start_time,
        "hf_model_dir": str(out / "hf_model"),
        "checkpoint_chck_1M_dir": str(out / "hf_model" / "chck_1M"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "completed", "metrics": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
