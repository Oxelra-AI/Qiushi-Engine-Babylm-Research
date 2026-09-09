#!/usr/bin/env python3
"""research: Paired continuation from INITIAL_MODEL_STUDIES research chck_70M.

Loads the INITIAL_MODEL_STUDIES 70M checkpoint and continues training for 30M more words (to 100M
total exposure) using the EXACT post-70M example order from INITIAL_MODEL_STUDIES's manifest.

Two arms run from identical weights with identical optimizer restart:
- Arm A (--mask_mode wwm):   fixed whole-word masking (control)
- Arm B (--mask_mode token): token-level masking (treatment)

Both share:
  - Same chck_70M weights (model only, no optimizer state)
  - Fresh AdamW (lr=1e-3, wd=0.01, betas=(0.9, 0.98)) → restart confound controlled
  - LR phase: cosine(warmup=122, total=2442) advanced to research before training
  - Exact post-70M example order from INITIAL_MODEL_STUDIES manifest
  - batch=256, seq=256, mask_prob=0.15
  - Masking RNG seed=43

The ONLY difference is mask_mode (wwm vs token).
This isolates the granularity effect from the optimizer restart confound.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForMaskedLM, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# ─── Constants ───────────────────────────────────────────────────────────────
INITIAL_MODEL_STUDIES_RUN = Path("experiments/archive/initial_model_studies/training/runs"
                 "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256")
CHCK_70M = INITIAL_MODEL_STUDIES_RUN / "hf_model" / "chck_70M"
MANIFEST = INITIAL_MODEL_STUDIES_RUN / "example_order_manifest.json"
DATASET_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
DATASET_REV = "c92ab16b4f08858304b0815706065b3354d8fc0a"

# Original schedule parameters (from INITIAL_MODEL_STUDIES research scientific_metrics)
ORIGINAL_TOTAL_STEPS = 2442
ORIGINAL_WARMUP_STEPS = 122   # int(2442 * 0.05)
RESUME_STEP = 1709            # 70M / 40960 words/step
WORDS_PER_EXAMPLE = 160
POST_70M_START_IDX = 437500   # 70M / 160
POST_70M_END_IDX = 625000     # All remaining


# ─── Data ────────────────────────────────────────────────────────────────────
# Exact file order used by INITIAL_MODEL_STUDIES babylm_masked_train_fullcycle.py
TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]


@dataclass
class Example:
    text: str
    words: int
    example_id: int
    source: str


def iter_examples_initial_model_studies(files: list[Path], max_words: int, words_per_example: int):
    """Exact replica of INITIAL_MODEL_STUDIES iter_examples: word-level streaming across files."""
    used = 0
    buf: list[str] = []
    buf_source = ""
    for fp in files:
        with fp.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for w in line.split():
                    if used >= max_words:
                        break
                    if not buf:
                        buf_source = fp.name
                    buf.append(w)
                    used += 1
                    if len(buf) >= words_per_example:
                        yield Example(" ".join(buf), len(buf), example_id=-1, source=buf_source)
                        buf = []
                        buf_source = ""
                if used >= max_words:
                    break
        if used >= max_words:
            break
    if buf:
        yield Example(" ".join(buf), len(buf), example_id=-1, source=buf_source)


def load_post_70M_examples(cache_dir: Optional[Path] = None) -> list[Example]:
    """Load the exact post-70M examples in INITIAL_MODEL_STUDIES manifest order."""
    print(json.dumps({"event": "loading_manifest", "path": str(MANIFEST)}), flush=True)
    manifest = json.load(MANIFEST.open())
    all_ids = manifest["consumed_example_ids_in_order"]
    post_ids = all_ids[POST_70M_START_IDX:POST_70M_END_IDX]
    assert len(post_ids) == 187500, f"Expected 187500, got {len(post_ids)}"

    # Download official dataset
    from huggingface_hub import snapshot_download
    dl_dir = cache_dir or Path("/tmp/babylm_continuation_cache")
    dl_dir.mkdir(parents=True, exist_ok=True)
    local = Path(snapshot_download(
        repo_id=DATASET_ID, repo_type="dataset",
        revision=DATASET_REV, allow_patterns=TRAIN_FILES + ["README.md"],
        local_dir=str(dl_dir), local_dir_use_symlinks=False,
    ))
    
    # Build pool using EXACT INITIAL_MODEL_STUDIES logic: iter_examples with files in TRAIN_FILES order
    files = [local / name for name in TRAIN_FILES]
    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(f"Required file missing: {fp}")
    
    pool_examples = list(iter_examples_initial_model_studies(files, 10_000_000, WORDS_PER_EXAMPLE))
    for i, ex in enumerate(pool_examples):
        ex.example_id = i
    
    print(json.dumps({"event": "pool_built", "total_examples": len(pool_examples),
                      "total_words": sum(e.words for e in pool_examples)}), flush=True)
    
    # Select post-70M examples in manifest order
    examples = []
    for idx, eid in enumerate(post_ids):
        if eid >= len(pool_examples):
            raise IndexError(f"Example ID {eid} >= pool size {len(pool_examples)}")
        ex = pool_examples[eid]
        examples.append(Example(text=ex.text, words=ex.words,
                               example_id=eid, source=ex.source))
    
    total_words = sum(e.words for e in examples)
    print(json.dumps({"event": "post_70M_loaded", "n_examples": len(examples),
                      "total_words": total_words, "first_id": post_ids[0],
                      "last_id": post_ids[-1]}), flush=True)
    return examples


# ─── Tokenization ────────────────────────────────────────────────────────────
class ContinuationDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id or 0
        self.special_ids = set()
        for sid in [tokenizer.cls_token_id, tokenizer.sep_token_id,
                    tokenizer.pad_token_id, tokenizer.mask_token_id]:
            if sid is not None:
                self.special_ids.add(sid)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(ex.text, truncation=True, max_length=self.max_length,
                            padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        
        # Build word_group for WWM
        word_group = torch.full_like(input_ids, -1)
        gid = -1
        word_ids = enc.word_ids(0) if hasattr(enc, "word_ids") else None
        if word_ids is not None:
            for i, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if i == 0 or wid != word_ids[i-1]:
                    gid += 1
                word_group[i] = gid
        else:
            # Fallback: use subword-start heuristic
            for i in range(len(input_ids)):
                tid = int(input_ids[i])
                if tid in self.special_ids:
                    continue
                token_str = self.tokenizer.convert_ids_to_tokens(tid)
                if token_str and (i == 0 or not token_str.startswith("##")):
                    gid += 1
                if gid >= 0:
                    word_group[i] = gid

        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "word_group": word_group, "words": ex.words}


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
    }


# ─── Masking ─────────────────────────────────────────────────────────────────
def apply_masking(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  word_group: torch.Tensor, tokenizer, mask_mode: str,
                  mask_prob: float, gen: torch.Generator):
    """Apply WWM or token-level masking. Returns (masked_input_ids, labels)."""
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    special_ids = set()
    for sid in [tokenizer.cls_token_id, tokenizer.sep_token_id,
                tokenizer.pad_token_id, tokenizer.mask_token_id]:
        if sid is not None:
            special_ids.add(sid)
    
    # Identify candidate positions (non-special, non-pad)
    candidate = attention_mask.bool().clone()
    for sid in special_ids:
        candidate &= (input_ids != sid)
    
    select = torch.zeros_like(input_ids, dtype=torch.bool)
    
    if mask_mode == "token":
        # Token-level: each candidate token independently selected
        probs = torch.full((bsz, seq), mask_prob, device=device)
        rand = torch.rand(bsz, seq, device=device, generator=gen)
        select = (rand < probs) & candidate
    elif mask_mode == "wwm":
        # Whole-word masking: select word groups, mask all tokens in selected groups
        for b in range(bsz):
            groups = word_group[b]
            valid_groups = torch.unique(groups[groups >= 0])
            if valid_groups.numel() == 0:
                continue
            n_select = max(1, int(valid_groups.numel() * mask_prob + 0.5))
            perm = torch.randperm(valid_groups.numel(), device=device, generator=gen)
            chosen = valid_groups[perm[:n_select]]
            sel_b = torch.isin(groups, chosen) & candidate[b]
            select[b] = sel_b
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")
    
    # Ensure at least one masked position per sequence
    for b in range(bsz):
        if select[b].sum() == 0:
            cands = candidate[b].nonzero(as_tuple=False)
            if cands.numel() > 0:
                select[b, cands[0, 0]] = True
    
    # Apply masking: 80% [MASK], 10% random, 10% keep
    masked_input = input_ids.clone()
    rand2 = torch.rand(bsz, seq, device=device, generator=gen)
    mask_positions = select
    
    # 80% MASK
    mask80 = mask_positions & (rand2 < 0.8)
    masked_input[mask80] = mask_token_id
    
    # 10% random
    mask_rand = mask_positions & (rand2 >= 0.8) & (rand2 < 0.9)
    n_rand = mask_rand.sum().item()
    if n_rand > 0:
        random_tokens = torch.randint(0, tokenizer.vocab_size, (n_rand,),
                                     device=device, generator=gen)
        masked_input[mask_rand] = random_tokens
    
    # 10% keep original (already in masked_input)
    
    # Labels: -100 where not predicted
    labels[~mask_positions] = -100
    
    return masked_input, labels


# ─── Checkpoint saving ───────────────────────────────────────────────────────
def save_hf_checkpoint(model, tokenizer, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)


# ─── Main ────────────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser(description="Paired continuation from INITIAL_MODEL_STUDIES chck_70M")
    p.add_argument("--mask_mode", choices=["wwm", "token"], required=True,
                   help="Masking granularity for this arm")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--max_word_exposure", type=int, default=30_000_000,
                   help="Word exposure for this continuation (default 30M)")
    p.add_argument("--checkpoint_words", type=int, default=10_000_000)
    p.add_argument("--cache_dir", default="/tmp/babylm_continuation_cache")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--smoke_steps", type=int, default=0,
                   help="If >0, stop after this many steps (smoke test)")
    return p.parse_args()


def main():
    args = build_args()
    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ─── Load model from chck_70M ────────────────────────────────────────────
    print(json.dumps({"event": "loading_model", "path": str(CHCK_70M)}), flush=True)
    model = AutoModelForMaskedLM.from_pretrained(str(CHCK_70M))
    tokenizer = AutoTokenizer.from_pretrained(str(CHCK_70M))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(json.dumps({"event": "model_loaded", "params": param_count,
                      "device": str(device)}), flush=True)

    # ─── Load post-70M examples ──────────────────────────────────────────────
    examples = load_post_70M_examples(Path(args.cache_dir))
    
    # Respect max_word_exposure
    max_examples = args.max_word_exposure // WORDS_PER_EXAMPLE
    if max_examples < len(examples):
        examples = examples[:max_examples]
    
    dataset = ContinuationDataset(examples, tokenizer, max_length=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                       collate_fn=collate, num_workers=args.num_workers,
                       pin_memory=torch.cuda.is_available(), drop_last=False)
    total_continuation_steps = len(loader)
    print(json.dumps({"event": "dataloader_ready", "total_steps": total_continuation_steps,
                      "examples": len(examples), "batch_size": args.batch_size}), flush=True)

    # ─── Optimizer + LR schedule (resume from research) ─────────────────────
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=ORIGINAL_WARMUP_STEPS,
        num_training_steps=ORIGINAL_TOTAL_STEPS
    )
    # Advance scheduler to research (the position where 70M was saved)
    for _ in range(RESUME_STEP):
        sched.step()
    
    initial_lr = sched.get_last_lr()[0]
    print(json.dumps({"event": "lr_phase_resumed", "resume_step": RESUME_STEP,
                      "initial_lr": initial_lr, "expected_lr": 0.000227,
                      "continuation_steps": total_continuation_steps}), flush=True)

    # ─── Training ────────────────────────────────────────────────────────────
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    
    log_path = out / "training_log.jsonl"
    cumulative_words = 0  # Words in this continuation only
    global_words = 70_000_000  # Cumulative from start of original training
    loss_values = []
    masked_token_values = []
    saved_checkpoints = []
    next_ckpt = args.checkpoint_words  # First checkpoint at 10M continuation = 80M global
    
    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            if args.smoke_steps > 0 and step > args.smoke_steps:
                break
            
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            
            masked_inputs, labels = apply_masking(
                input_ids, attention_mask, word_group,
                tokenizer, args.mask_mode, args.mask_prob, gen
            )
            
            optim.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask,
                            labels=labels)
            loss = out_model.loss
            if loss is None:
                raise RuntimeError("model returned no loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            
            cumulative_words += words
            global_words += words
            loss_float = float(loss.detach().cpu())
            loss_values.append(loss_float)
            n_pred = int((labels != -100).sum().item())
            masked_token_values.append(n_pred)
            
            rec = {
                "step": step, "global_step": RESUME_STEP + step,
                "loss": loss_float, "lr": float(sched.get_last_lr()[0]),
                "batch_words": words, "continuation_word_exposure": cumulative_words,
                "global_word_exposure": global_words,
                "masked_tokens": n_pred,
                "effective_mask_rate": round(n_pred / (input_ids.shape[0] * input_ids.shape[1]), 4),
                "mask_mode": args.mask_mode,
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            
            if step == 1 or step % args.log_every == 0 or step == total_continuation_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            
            # Checkpointing based on continuation words
            while next_ckpt is not None and cumulative_words >= next_ckpt:
                global_target = 70_000_000 + next_ckpt
                name = f"chck_{global_target // 1_000_000}M"
                cp = out / "hf_model" / name
                save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({
                    "name": name,
                    "target_global_word_exposure": global_target,
                    "actual_global_word_exposure": global_words,
                    "continuation_words": cumulative_words,
                    "path": str(cp),
                })
                print(json.dumps({"event": "checkpoint_saved", "name": name,
                                  "global_words": global_words}), flush=True)
                next_ckpt += args.checkpoint_words
                if 70_000_000 + next_ckpt > 100_000_000:
                    next_ckpt = None

    # Save final model
    final_cp = out / "hf_model" / "chck_100M"
    save_hf_checkpoint(model, tokenizer, final_cp)
    if not saved_checkpoints or saved_checkpoints[-1]["name"] != "chck_100M":
        saved_checkpoints.append({
            "name": "chck_100M",
            "target_global_word_exposure": 100_000_000,
            "actual_global_word_exposure": global_words,
            "continuation_words": cumulative_words,
            "path": str(final_cp),
        })
    
    # ─── Save metrics ────────────────────────────────────────────────────────
    metrics = {
        "experiment": "paired_continuation_from_70M",
        "mask_mode": args.mask_mode,
        "source_checkpoint": str(CHCK_70M),
        "resume_step": RESUME_STEP,
        "original_total_steps": ORIGINAL_TOTAL_STEPS,
        "continuation_steps": step if args.smoke_steps == 0 else min(step, args.smoke_steps),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "mask_prob": args.mask_prob,
        "initial_lr_at_resume": initial_lr,
        "seed": args.seed,
        "parameter_count": param_count,
        "word_exposure_continuation": cumulative_words,
        "word_exposure_global": global_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "masked_tokens_total": sum(masked_token_values),
        "saved_checkpoints": saved_checkpoints,
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(json.dumps({"event": "done", "mask_mode": args.mask_mode,
                      "loss_first": metrics["loss_first"],
                      "loss_last": metrics["loss_last"],
                      "global_words": global_words,
                      "checkpoints": [c["name"] for c in saved_checkpoints]}), flush=True)


if __name__ == "__main__":
    main()
