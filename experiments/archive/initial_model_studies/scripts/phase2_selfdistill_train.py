#!/usr/bin/env python3
"""Phase-2 Self-Distillation trainer: preserve mid-course broad representations.

Scientific hypothesis: The AMLM 40M checkpoint has broader Entity/EWoK/GlobalPIQA
representations than the final 100M model. These decay because the MLM gradient
at late training doesn't require them. Self-distillation from the 40M state
prevents this decay by adding a representation-preservation loss.

Implementation:
- Load AMLM 40M checkpoint as STUDENT (trainable) and TEACHER (frozen)
- Train student with: L = L_WWM + lambda * L_distill
- L_distill = MSE between student and teacher hidden states at target layers
- Continue for 60M words (reaching 100M total) with fresh cosine LR
- Save checkpoints at standard 10M intervals

This is NOT:
- WESS (no external entity slots or labeled addresses)
- R1 (no synthetic state data)
- AMLM (no difficulty-weighted masking; uses flat WWM)
- Hyperparameter tuning (changes the loss function structure)

It IS: a test of whether representation-geometry preservation prevents the
reproducible 40M→100M competence decay on Entity/EWoK/GlobalPIQA.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedTokenizerFast,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"
TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]


# ─── Data loading (reuse exact official-corpus logic) ───────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_words_in_file(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += len(line.split())
    return total


def load_official_corpus(data_dir: Path) -> list[str]:
    """Load official BabyLM train files, return list of documents."""
    docs = []
    for fname in TRAIN_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing: {fpath}")
        with fpath.open("r", encoding="utf-8", errors="replace") as f:
            current_doc = []
            for line in f:
                stripped = line.strip()
                if stripped:
                    current_doc.append(stripped)
                else:
                    if current_doc:
                        docs.append(" ".join(current_doc))
                        current_doc = []
            if current_doc:
                docs.append(" ".join(current_doc))
    return docs


@dataclass
class PackedExample:
    input_ids: list[int]
    word_group_boundaries: list[tuple[int, int]]  # for WWM
    n_words: int


def pack_examples(docs: list[str], tokenizer: PreTrainedTokenizerFast,
                  max_seq_length: int, words_per_example: int,
                  pool_words: int, seed: int) -> list[PackedExample]:
    """Pack documents into fixed-length training examples with word accounting."""
    rng = random.Random(seed)
    shuffled = list(range(len(docs)))
    rng.shuffle(shuffled)

    examples = []
    cumulative_words = 0
    buffer_ids = []
    buffer_boundaries = []
    buffer_words = 0

    for doc_idx in shuffled:
        if cumulative_words >= pool_words:
            break
        doc = docs[doc_idx]
        words = doc.split()
        # Tokenize with word boundaries
        encoded = tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)
        tokens = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        # Build word group boundaries
        word_groups = []
        if offsets:
            current_start = 0
            for i in range(1, len(offsets)):
                if offsets[i][0] > offsets[i-1][1] or (offsets[i][0] == 0 and i > 0):
                    word_groups.append((current_start, i))
                    current_start = i
            word_groups.append((current_start, len(offsets)))

        # Add to buffer
        offset = len(buffer_ids)
        buffer_ids.extend(tokens)
        buffer_boundaries.extend([(s + offset, e + offset) for s, e in word_groups])
        buffer_words += len(words)
        cumulative_words += len(words)

        # Flush full examples
        while len(buffer_ids) >= max_seq_length:
            ex_ids = buffer_ids[:max_seq_length]
            ex_bounds = [(s, e) for s, e in buffer_boundaries if s < max_seq_length]
            # Clip last boundary
            if ex_bounds and ex_bounds[-1][1] > max_seq_length:
                ex_bounds[-1] = (ex_bounds[-1][0], max_seq_length)
            ex_words = sum(1 for s, e in ex_bounds if e - s > 0)
            examples.append(PackedExample(ex_ids, ex_bounds, ex_words))

            buffer_ids = buffer_ids[max_seq_length:]
            buffer_boundaries = [(s - max_seq_length, e - max_seq_length)
                                 for s, e in buffer_boundaries if e > max_seq_length]
            buffer_boundaries = [(max(0, s), e) for s, e in buffer_boundaries]

    return examples


# ─── Masking (WWM only for Phase 2) ────────────────────────────────────────

def apply_wwm_masking(input_ids: torch.Tensor, word_boundaries: list[list[tuple[int, int]]],
                      mask_prob: float, tokenizer: PreTrainedTokenizerFast,
                      rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-word masking. Returns (masked_input_ids, labels)."""
    batch_size, seq_len = input_ids.shape
    labels = torch.full_like(input_ids, -100)
    masked_ids = input_ids.clone()
    mask_token_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size

    for b in range(batch_size):
        boundaries = word_boundaries[b]
        # Select word groups to mask
        selected = [bg for bg in boundaries if rng.random() < mask_prob]
        for start, end in selected:
            for pos in range(start, min(end, seq_len)):
                labels[b, pos] = input_ids[b, pos]
                r = rng.random()
                if r < 0.8:
                    masked_ids[b, pos] = mask_token_id
                elif r < 0.9:
                    masked_ids[b, pos] = rng.randint(0, vocab_size - 1)
                # else keep original (10%)
    return masked_ids, labels


# ─── Self-Distillation Loss ────────────────────────────────────────────────

class SelfDistillationLoss(torch.nn.Module):
    """MSE between student and teacher hidden states at target layers."""

    def __init__(self, teacher_model: torch.nn.Module, target_layers: list[int],
                 lambda_distill: float = 0.5):
        super().__init__()
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.target_layers = target_layers
        self.lambda_distill = lambda_distill

    @torch.no_grad()
    def get_teacher_hiddens(self, input_ids: torch.Tensor,
                            attention_mask: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.teacher(input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        return [outputs.hidden_states[l] for l in self.target_layers]

    def compute(self, student_hiddens: list[torch.Tensor],
                input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        teacher_hiddens = self.get_teacher_hiddens(input_ids, attention_mask)
        loss = torch.tensor(0.0, device=input_ids.device)
        for s_h, t_h in zip(student_hiddens, teacher_hiddens):
            # MSE only on non-padding positions
            mask = attention_mask.unsqueeze(-1).float()
            diff = (s_h - t_h) * mask
            loss += (diff ** 2).sum() / mask.sum() / s_h.shape[-1]
        return self.lambda_distill * loss / len(self.target_layers)


# ─── Main Training Loop ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--student_init", required=True, help="Path to 40M checkpoint (also used as teacher)")
    parser.add_argument("--teacher_path", default=None, help="Teacher path (default: same as student_init)")
    # Training
    parser.add_argument("--max_word_exposure", type=int, default=60000000, help="Words to train in Phase 2")
    parser.add_argument("--example_pool_words", type=int, default=10000000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_fraction", type=float, default=0.06)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    # Distillation
    parser.add_argument("--lambda_distill", type=float, default=0.5)
    parser.add_argument("--target_layers", type=int, nargs="+", default=[2, 4, 6])
    # Checkpointing
    parser.add_argument("--checkpoint_words", type=int, default=10000000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = args.teacher_path or args.student_init

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    mask_rng = random.Random(args.seed + 789)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.student_init)
    print(f"[init] tokenizer loaded from {args.student_init}, vocab={tokenizer.vocab_size}")

    # Load student model
    student = AutoModelForMaskedLM.from_pretrained(args.student_init)
    student.to(device)
    student.train()
    param_count = sum(p.numel() for p in student.parameters())
    print(f"[init] student loaded: {param_count:,} parameters")

    # Load teacher model (frozen copy)
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_path)
    teacher.to(device)
    distill_loss_fn = SelfDistillationLoss(teacher, args.target_layers, args.lambda_distill)
    print(f"[init] teacher loaded from {teacher_path}, layers={args.target_layers}, λ={args.lambda_distill}")

    # Load and pack official corpus
    data_dir = Path(snapshot_download(
        "BabyLM-community/BabyLM-2026-Strict",
        repo_type="dataset", allow_patterns=["train/*.txt"],
        local_dir=str(output_dir / "data_cache")
    )) / "train"
    docs = load_official_corpus(data_dir)
    pool_word_count = sum(len(d.split()) for d in docs)
    print(f"[data] {len(docs)} docs, {pool_word_count:,} words in pool")

    # Build epoch-shuffled examples
    total_epochs = (args.max_word_exposure + args.example_pool_words - 1) // args.example_pool_words
    all_examples = []
    for epoch in range(total_epochs):
        epoch_examples = pack_examples(docs, tokenizer, args.seq_length,
                                        160, args.example_pool_words,
                                        seed=args.seed + epoch * 1000)
        random.Random(args.seed + epoch * 7).shuffle(epoch_examples)
        all_examples.extend(epoch_examples)

    # Compute training geometry
    examples_per_batch = args.batch_size
    total_examples = min(len(all_examples),
                         args.max_word_exposure * len(all_examples) // sum(e.n_words for e in all_examples))
    total_steps = total_examples // examples_per_batch
    warmup_steps = int(total_steps * args.warmup_fraction)
    print(f"[geometry] {total_steps} steps, {warmup_steps} warmup, {total_epochs} epochs")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training state
    cumulative_words = 0
    step = 0
    next_checkpoint_words = args.checkpoint_words
    loss_accum = 0.0
    mlm_loss_accum = 0.0
    distill_loss_accum = 0.0
    log_steps = 0
    t0 = time.time()
    training_log = output_dir / "training_log.jsonl"
    checkpoints_saved = []

    # Starting word offset (we're continuing from 40M)
    word_offset = 40000000  # for checkpoint naming

    print(f"[train] starting Phase 2 self-distillation training")
    print(f"[train] word_offset={word_offset}, max_exposure={args.max_word_exposure}")

    batch_ids = []
    batch_boundaries = []
    batch_words = 0
    example_idx = 0

    while cumulative_words < args.max_word_exposure and example_idx < len(all_examples):
        # Collect batch
        batch_ids.clear()
        batch_boundaries.clear()
        batch_words = 0

        while len(batch_ids) < examples_per_batch and example_idx < len(all_examples):
            ex = all_examples[example_idx]
            example_idx += 1
            batch_ids.append(ex.input_ids)
            batch_boundaries.append(ex.word_group_boundaries)
            batch_words += ex.n_words

        if len(batch_ids) < examples_per_batch:
            break

        # Tensorize
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else torch.ones_like(input_ids)

        # Apply WWM masking
        masked_ids, labels = apply_wwm_masking(input_ids, batch_boundaries,
                                               args.mask_prob, tokenizer, mask_rng)

        # Forward pass (student) with hidden states
        outputs = student(masked_ids, attention_mask=attention_mask, labels=labels,
                         output_hidden_states=True)
        mlm_loss = outputs.loss

        # Distillation loss (on UNMASKED input for clean teacher comparison)
        student_hiddens = [outputs.hidden_states[l] for l in args.target_layers]
        d_loss = distill_loss_fn.compute(student_hiddens, input_ids, attention_mask)

        # Combined loss
        total_loss = mlm_loss + d_loss

        # Backward + step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Accounting
        step += 1
        cumulative_words += batch_words
        loss_accum += total_loss.item()
        mlm_loss_accum += mlm_loss.item()
        distill_loss_accum += d_loss.item()
        log_steps += 1

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            log_entry = {
                "step": step, "loss": loss_accum / log_steps,
                "mlm_loss": mlm_loss_accum / log_steps,
                "distill_loss": distill_loss_accum / log_steps,
                "lr": scheduler.get_last_lr()[0],
                "cumulative_words": cumulative_words,
                "total_words_with_offset": cumulative_words + word_offset,
                "elapsed_sec": elapsed,
            }
            with training_log.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"[step {step}] loss={log_entry['loss']:.4f} mlm={log_entry['mlm_loss']:.4f} "
                  f"distill={log_entry['distill_loss']:.4f} lr={log_entry['lr']:.2e} "
                  f"words={cumulative_words + word_offset:,}", flush=True)
            loss_accum = mlm_loss_accum = distill_loss_accum = 0.0
            log_steps = 0

        # Checkpointing
        if cumulative_words >= next_checkpoint_words:
            total_m = (cumulative_words + word_offset) // 1000000
            ckpt_name = f"chck_{total_m}M"
            ckpt_path = output_dir / "hf_model" / ckpt_name
            student.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            checkpoints_saved.append({"name": ckpt_name, "cum_words": cumulative_words + word_offset,
                                       "path": str(ckpt_path)})
            print(f"[checkpoint] {ckpt_name} saved at {cumulative_words + word_offset:,} total words")
            next_checkpoint_words += args.checkpoint_words

    # Final save
    elapsed = time.time() - t0
    final_ckpt = output_dir / "hf_model" / "chck_100M"
    student.save_pretrained(str(final_ckpt))
    tokenizer.save_pretrained(str(final_ckpt))

    metrics = {
        "variant": "phase2_self_distillation",
        "student_init": args.student_init,
        "teacher_path": teacher_path,
        "parameter_count": param_count,
        "lambda_distill": args.lambda_distill,
        "target_layers": args.target_layers,
        "phase2_word_exposure": cumulative_words,
        "total_word_exposure_with_offset": cumulative_words + word_offset,
        "total_training_steps": step,
        "learning_rate": args.learning_rate,
        "mask_prob": args.mask_prob,
        "mask_mode": "wwm",
        "seed": args.seed,
        "elapsed_sec": elapsed,
        "checkpoints": checkpoints_saved,
    }
    (output_dir / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"event": "done", **metrics}, indent=2))


if __name__ == "__main__":
    main()
