#!/usr/bin/env python3
"""Unified Masking-Curriculum Trainer for BabyLM Strict-Small.

Implements prediction-granularity curriculum control as a principled framework:
  - WWM→token switching: controls prediction UNIT granularity over training
  - Mask-probability decay: controls prediction DENSITY over training  
  - Per-token difficulty weighting (AMLM-hard): controls which tokens receive
    prediction pressure based on model's current accuracy

These are unified as three orthogonal dimensions of "what the model is asked
to predict" at each training step. The hypothesis: controlling these jointly
creates better reusable competence than any fixed masking recipe because it
matches prediction difficulty to the model's current learning state.

Dynamics traces at each checkpoint record:
  - Per-frequency-band loss and accuracy
  - Effective masking rate vs nominal
  - Prediction entropy distribution
  - Token difficulty distribution evolution

These traces let us verify WHEN and WHY transitions improve cross-task transfer.

Compatible with: DeBERTa-v2 8x480, baseline16k tokenizer, official corpus,
and the COMPACT_EXPERIENCE evaluation pipeline (eval_cs_4m_residualized_fast.py).

Masking curriculum modes:
  wwm_fixed       - Standard fixed WWM at mask_prob (baseline control)
  token_fixed     - Standard fixed token-level at mask_prob
  wwm_to_token    - WWM for first switch_frac, then token-level (leader recipe)
  decay_wwm       - WWM with mask_prob linearly decaying from start to end
  decay_token     - Token-level with mask_prob linearly decaying
  decay_wwm_to_token - Decaying prob + granularity switch at switch_frac
  amlm_hard       - WWM + per-token difficulty weighting + decay
  amlm_hard_switch - AMLM-hard with WWM→token switch at switch_frac

Extends compact_experience with LAMB optimizer and intermediate_size control.
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
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

# LAMB optimizer for leader-recipe replication
try:
    from lamb_optimizer import get_lamb_optimizer
    LAMB_AVAILABLE = True
except ImportError:
    try:
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("lamb_optimizer",
            str(_public_path('experiments/archive/frontier_consolidation/scripts/lamb_optimizer.py')))
        lmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(lmod)
        get_lamb_optimizer = lmod.get_lamb_optimizer
        LAMB_AVAILABLE = True
    except Exception:
        LAMB_AVAILABLE = False

# ─── Constants ────────────────────────────────────────────────────────────────
BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"
TRAIN_FILES = [
    "bnc_spoken.train.txt", "childes.train.txt", "gutenberg.train.txt",
    "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt",
]

MASKING_CURRICULA = [
    "wwm_fixed", "token_fixed",
    "wwm_to_token", "decay_wwm", "decay_token",
    "decay_wwm_to_token", "amlm_hard", "amlm_hard_switch",
]

# ─── Data utilities ───────────────────────────────────────────────────────────
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


@dataclass
class Example:
    text: str
    words: int
    example_id: int = -1
    source: str = ""


def iter_examples(files: list[Path], max_words: int, words_per_example: int) -> Iterable[Example]:
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
                        yield Example(" ".join(buf), len(buf), source=buf_source)
                        buf = []
                        buf_source = ""
                if used >= max_words:
                    break
        if used >= max_words:
            break
    if buf:
        yield Example(" ".join(buf), len(buf), source=buf_source)


def load_examples_jsonl(path: Path, selected_words: int) -> tuple[list[Example], int, int, list[dict]]:
    examples: list[Example] = []
    total_file_words = 0
    total_rows = 0
    sample_rows: list[dict] = []
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
                raise RuntimeError(f"JSONL word-count mismatch at row {total_rows}: field={words} actual={actual}")
            total_file_words += words
            if selected < selected_words:
                ex_id = int(obj.get("example_id", total_rows - 1))
                source = str(obj.get("source", "example_jsonl"))
                if selected + words <= selected_words:
                    examples.append(Example(text=text, words=words, example_id=ex_id, source=source))
                    selected += words
                else:
                    # Partial example to reach exact target (matches official corpus behavior)
                    take = selected_words - selected
                    if take > 0:
                        partial_text = " ".join(text.split()[:take])
                        examples.append(Example(text=partial_text, words=take, example_id=ex_id, source=source))
                        selected += take
                if len(sample_rows) < 10:
                    keep = {k: obj[k] for k in obj.keys() if k != "text"}
                    sample_rows.append(keep)
    if selected != selected_words:
        raise RuntimeError(f"JSONL selected words {selected} != target {selected_words}")
    return examples, total_file_words, total_rows, sample_rows


# ─── Tokenizer ────────────────────────────────────────────────────────────────
def make_portable_tokenizer(tokenizer_path: str = "") -> PreTrainedTokenizerFast:
    if tokenizer_path:
        from pathlib import Path as _P
        tp = str(_P(tokenizer_path).resolve()) if _P(tokenizer_path).exists() else tokenizer_path
        base = AutoTokenizer.from_pretrained(tp, use_fast=True, local_files_only=_P(tokenizer_path).exists())
    else:
        # Try local cache first, then online
        try:
            base = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True, local_files_only=True)
        except Exception:
            base = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True)
    backend = getattr(base, "_tokenizer", None) or getattr(base, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError(f"Tokenizer at {tokenizer_path or BASELINE_TOKENIZER_REPO} has no fast backend")
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


# ─── Dataset ──────────────────────────────────────────────────────────────────
def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


class MaskedChunkDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start = {}

    def _word_start_flag(self, token_id: int) -> bool:
        v = self._word_start.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start[token_id] = v
        return v

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text, add_special_tokens=False, truncation=True,
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
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "word_group": group, "words": ex.words}


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
    }


# ─── Masking Curriculum State ─────────────────────────────────────────────────
@dataclass
class MaskingCurriculumState:
    """Unified state for prediction-granularity curriculum control.
    
    Three orthogonal dimensions:
      1. Granularity: wwm vs token (controlled by switch_frac)
      2. Density: mask_prob decays from start to end
      3. Difficulty: per-token weighting based on model accuracy (AMLM-hard)
    """
    curriculum: str
    mask_prob_start: float = 0.30
    mask_prob_end: float = 0.15
    switch_frac: float = 0.7  # fraction of training before WWM→token switch
    amlm_window: int = 10    # batches per AMLM weight update
    amlm_lambda: float = 0.2  # exponential smoothing for AMLM weights
    
    # Runtime state
    vocab_size: int = 0
    total_steps: int = 0
    current_step: int = 0
    
    # Per-token difficulty weights (AMLM-hard)
    token_weights: torch.Tensor = field(default=None, repr=False)
    correct_counts: torch.Tensor = field(default=None, repr=False)
    total_counts: torch.Tensor = field(default=None, repr=False)
    batches_in_window: int = 0
    n_amlm_updates: int = 0
    
    # Dynamics trace accumulators (reset at each checkpoint)
    trace_losses_by_freq: dict = field(default_factory=dict, repr=False)
    trace_accuracy_by_freq: dict = field(default_factory=dict, repr=False)
    trace_mask_rates: list = field(default_factory=list, repr=False)
    trace_entropy_sum: float = 0.0
    trace_entropy_count: int = 0
    
    def initialize(self, vocab_size: int, total_steps: int):
        """Initialize after vocab size and total steps are known."""
        self.vocab_size = vocab_size
        self.total_steps = total_steps
        if self.uses_amlm:
            self.token_weights = torch.ones(vocab_size) * self.mask_prob_start
            self.correct_counts = torch.zeros(vocab_size, dtype=torch.long)
            self.total_counts = torch.zeros(vocab_size, dtype=torch.long)
    
    @property
    def uses_amlm(self) -> bool:
        return self.curriculum in ("amlm_hard", "amlm_hard_switch")
    
    @property
    def uses_decay(self) -> bool:
        return self.curriculum in ("decay_wwm", "decay_token", "decay_wwm_to_token",
                                   "amlm_hard", "amlm_hard_switch")
    
    @property
    def uses_switch(self) -> bool:
        return self.curriculum in ("wwm_to_token", "decay_wwm_to_token", "amlm_hard_switch")
    
    def get_current_mask_mode(self) -> str:
        """What prediction granularity to use at the current step."""
        frac = self.current_step / max(1, self.total_steps)
        if self.curriculum in ("token_fixed", "decay_token"):
            return "token"
        if self.curriculum in ("wwm_fixed", "decay_wwm", "amlm_hard"):
            return "wwm"
        # Curricula with switching
        if self.uses_switch:
            return "wwm" if frac < self.switch_frac else "token"
        return "wwm"
    
    def get_current_mask_prob(self) -> float:
        """What nominal masking density to use at the current step."""
        if not self.uses_decay:
            return self.mask_prob_start  # fixed throughout
        frac = min(self.current_step / max(1, self.total_steps), 1.0)
        return self.mask_prob_start + (self.mask_prob_end - self.mask_prob_start) * frac
    
    def get_per_token_probs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Get per-token masking probabilities for AMLM-hard, or None for uniform."""
        if not self.uses_amlm or self.token_weights is None:
            return None
        return self.token_weights[input_ids.cpu()].to(input_ids.device)
    
    def record_batch_accuracy(self, input_ids: torch.Tensor, labels: torch.Tensor,
                              logits: torch.Tensor):
        """Track per-token accuracy for AMLM difficulty adaptation."""
        if not self.uses_amlm:
            return
        mask = labels != -100
        if mask.sum() == 0:
            self.batches_in_window += 1
            return
        masked_ids = labels[mask].detach().cpu().long()
        predicted = logits[mask].argmax(dim=-1).detach().cpu().long()
        correct_ids = masked_ids[predicted == masked_ids]
        total_bc = torch.bincount(masked_ids, minlength=self.vocab_size)
        correct_bc = (torch.bincount(correct_ids, minlength=self.vocab_size)
                      if correct_ids.numel() else torch.zeros(self.vocab_size, dtype=torch.long))
        self.total_counts += total_bc
        self.correct_counts += correct_bc
        self.batches_in_window += 1
    
    def maybe_update_amlm_weights(self) -> bool:
        """Update per-token masking weights if AMLM window is complete."""
        if not self.uses_amlm:
            return False
        if self.batches_in_window < self.amlm_window:
            return False
        
        base_prob = self.get_current_mask_prob()
        seen = self.total_counts > 0
        
        if seen.sum() > 0:
            accuracy = torch.zeros(self.vocab_size)
            accuracy[seen] = self.correct_counts[seen].float() / self.total_counts[seen].float()
            # Hard method: tokens with low accuracy get masked MORE
            new_weights = base_prob * (2.0 - 2.0 * accuracy)  # range [0, 2*base_prob]
            new_weights = new_weights.clamp(min=0.02, max=0.6)
            # Exponential smoothing
            self.token_weights[seen] = (
                self.amlm_lambda * self.token_weights[seen] +
                (1.0 - self.amlm_lambda) * new_weights[seen]
            )
        
        # Unseen tokens get base probability
        unseen = ~seen
        if self.token_weights is not None:
            self.token_weights[unseen] = base_prob
        
        # Reset window
        self.correct_counts.zero_()
        self.total_counts.zero_()
        self.batches_in_window = 0
        self.n_amlm_updates += 1
        return True
    
    def record_dynamics_trace(self, input_ids: torch.Tensor, labels: torch.Tensor,
                              logits: torch.Tensor, loss: float,
                              token_freqs: torch.Tensor, effective_mask_rate: float):
        """Accumulate dynamics trace data for checkpoint-level analysis."""
        self.trace_mask_rates.append(effective_mask_rate)
        
        # Loss and accuracy by frequency band
        mask = labels != -100
        if mask.sum() == 0:
            return
        
        masked_ids = labels[mask].detach().cpu().long()
        masked_logits = logits[mask].detach().cpu()
        predicted = masked_logits.argmax(dim=-1)
        correct = (predicted == masked_ids)
        
        # Frequency bands: high (top 1000), mid (1000-5000), low (5000+)
        freqs = token_freqs[masked_ids]
        for band_name, lo, hi in [("high", 0, 1000), ("mid", 1000, 5000), ("low", 5000, float("inf"))]:
            band_mask = (freqs >= lo) & (freqs < hi)
            if band_mask.sum() > 0:
                band_correct = correct[band_mask].float().mean().item()
                # Per-token cross-entropy for this band
                band_logits = masked_logits[band_mask]
                band_targets = masked_ids[band_mask]
                band_loss = F.cross_entropy(band_logits, band_targets, reduction="mean").item()
                if band_name not in self.trace_losses_by_freq:
                    self.trace_losses_by_freq[band_name] = []
                    self.trace_accuracy_by_freq[band_name] = []
                self.trace_losses_by_freq[band_name].append(band_loss)
                self.trace_accuracy_by_freq[band_name].append(band_correct)
        
        # Prediction entropy
        probs = F.softmax(masked_logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
        self.trace_entropy_sum += entropy
        self.trace_entropy_count += 1
    
    def flush_dynamics_trace(self, checkpoint_name: str) -> dict:
        """Produce a dynamics trace summary and reset accumulators."""
        trace = {
            "checkpoint": checkpoint_name,
            "curriculum": self.curriculum,
            "step": self.current_step,
            "mask_mode_at_checkpoint": self.get_current_mask_mode(),
            "mask_prob_at_checkpoint": self.get_current_mask_prob(),
            "n_amlm_updates_total": self.n_amlm_updates,
            "effective_mask_rate_mean": (
                sum(self.trace_mask_rates) / len(self.trace_mask_rates)
                if self.trace_mask_rates else 0.0
            ),
            "effective_mask_rate_std": float(np.std(self.trace_mask_rates)) if self.trace_mask_rates else 0.0,
            "loss_by_freq_band": {
                k: {"mean": sum(v)/len(v), "last": v[-1] if v else None, "n": len(v)}
                for k, v in self.trace_losses_by_freq.items()
            },
            "accuracy_by_freq_band": {
                k: {"mean": sum(v)/len(v), "last": v[-1] if v else None, "n": len(v)}
                for k, v in self.trace_accuracy_by_freq.items()
            },
            "prediction_entropy_mean": (
                self.trace_entropy_sum / self.trace_entropy_count
                if self.trace_entropy_count > 0 else 0.0
            ),
        }
        if self.uses_amlm and self.token_weights is not None:
            tw = self.token_weights.numpy()
            trace["amlm_weight_stats"] = {
                "mean": float(tw.mean()), "std": float(tw.std()),
                "p10": float(np.percentile(tw, 10)), "p50": float(np.percentile(tw, 50)),
                "p90": float(np.percentile(tw, 90)),
            }
        
        # Reset
        self.trace_losses_by_freq = {}
        self.trace_accuracy_by_freq = {}
        self.trace_mask_rates = []
        self.trace_entropy_sum = 0.0
        self.trace_entropy_count = 0
        return trace


# ─── Masking with curriculum control ─────────────────────────────────────────
def apply_masking_curriculum(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor,
    tokenizer, state: MaskingCurriculumState, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply masking according to the current curriculum state."""
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)
    
    mask_mode = state.get_current_mask_mode()
    mask_prob = state.get_current_mask_prob()
    per_token_probs = state.get_per_token_probs(input_ids)  # None if not AMLM
    
    if mask_mode == "token":
        if per_token_probs is not None:
            probs = torch.rand(bsz, seq, generator=gen, device=device)
            select = candidate & (probs < per_token_probs)
        else:
            probs = torch.rand(bsz, seq, generator=gen, device=device)
            select = candidate & (probs < mask_prob)
    elif mask_mode == "wwm":
        for b in range(bsz):
            groups = word_group[b]
            valid_groups = torch.unique(groups[groups >= 0])
            if valid_groups.numel() == 0:
                continue
            if per_token_probs is not None:
                # For AMLM+WWM: use mean token prob within each group
                group_probs = torch.zeros(valid_groups.max().item() + 1, device=device)
                for gid in valid_groups:
                    g_mask = (groups == gid) & candidate[b]
                    if g_mask.any():
                        group_probs[gid] = per_token_probs[b][g_mask].mean()
                gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
                chosen = valid_groups[gp < group_probs[valid_groups]]
            else:
                gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
                chosen = valid_groups[gp < mask_prob]
            if chosen.numel() == 0:
                continue
            sel_b = torch.isin(groups, chosen) & candidate[b]
            select[b] = sel_b
    
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
    
    labels[~select] = -100
    masked_inputs = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = mask_token_id
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids
    return masked_inputs, labels


# ─── Model ────────────────────────────────────────────────────────────────────
def build_model(args: argparse.Namespace, tokenizer):
    max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
    pos_att_type = [x.strip() for x in args.deberta_pos_att_type.split(",") if x.strip()]
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.intermediate_size if args.intermediate_size > 0 else args.hidden_size * args.ffn_mult,
        max_position_embeddings=max_pos,
        max_relative_positions=args.max_relative_positions,
        position_buckets=args.position_buckets,
        relative_attention=True,
        pos_att_type=pos_att_type,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return DebertaV2ForMaskedLM(cfg)


def save_hf_checkpoint(model, tokenizer, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    force_portable_tokenizer_config(dst)


# ─── Sequence-length schedule ─────────────────────────────────────────────────
def seq_length_for_progress(frac: float, schedule: list[tuple[float, int]], default_len: int) -> int:
    length = default_len
    for thresh, L in schedule:
        if frac >= thresh:
            length = L
    return length


# ─── Token frequency computation ─────────────────────────────────────────────
def compute_token_frequency_ranks(examples: list[Example], tokenizer, max_seq_length: int) -> torch.Tensor:
    """Compute frequency rank (0=most frequent) for each token in vocab.
    Used for dynamics trace frequency-band analysis."""
    vocab_size = len(tokenizer)
    counts = torch.zeros(vocab_size, dtype=torch.long)
    for ex in examples[:min(len(examples), 5000)]:  # sample for speed
        enc = tokenizer(ex.text, add_special_tokens=False, truncation=True,
                        max_length=max_seq_length)
        for tid in enc["input_ids"]:
            counts[tid] += 1
    # Rank: 0 = most frequent
    _, indices = counts.sort(descending=True)
    ranks = torch.zeros(vocab_size, dtype=torch.long)
    ranks[indices] = torch.arange(vocab_size)
    return ranks


# ─── Main training loop ──────────────────────────────────────────────────────
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Masking-Curriculum Trainer")
    # Data
    p.add_argument("--dataset_id", default="BabyLM-community/BabyLM-2026-Strict-Small")
    p.add_argument("--dataset_revision", default="c92ab16b4f08858304b0815706065b3354d8fc0a")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_word_exposure", type=int, default=4_000_000)
    p.add_argument("--example_pool_words", type=int, default=10_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--words_per_example", type=int, default=160)
    p.add_argument("--example_jsonl", default="")
    p.add_argument("--example_jsonl_label", default="")
    p.add_argument("--example_jsonl_meta", default="")
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--tokenizer_label", default="baseline16k")
    p.add_argument("--tokenization_summary_limit", type=int, default=500)
    
    # Masking curriculum
    p.add_argument("--masking_curriculum", choices=MASKING_CURRICULA, default="wwm_fixed",
                   help="Masking curriculum mode")
    p.add_argument("--mask_prob_start", type=float, default=0.15,
                   help="Initial masking probability")
    p.add_argument("--mask_prob_end", type=float, default=0.15,
                   help="Final masking probability (same as start for fixed modes)")
    p.add_argument("--switch_frac", type=float, default=0.7,
                   help="Fraction of training before WWM→token granularity switch")
    p.add_argument("--amlm_window", type=int, default=10,
                   help="Batches per AMLM difficulty weight update")
    p.add_argument("--amlm_lambda", type=float, default=0.2,
                   help="Exponential smoothing for AMLM weight updates")
    
    # Sequence length
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--seq_len_schedule", default="",
                   help="e.g. '0.0:64,0.5:128,0.8:256'")
    
    # Model (DeBERTa-v2 8x480 default)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=0,
                   help="Explicit intermediate size; overrides hidden_size*ffn_mult if >0")
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    
    # Optimization
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["adamw", "lamb"], default="adamw",
                   help="Optimizer: adamw (default) or lamb (leader recipe)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping max norm")
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    
    # Reproducibility
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=-1)
    p.add_argument("--train_rng_seed", type=int, default=-1)
    p.add_argument("--log_every", type=int, default=50)
    
    # Dynamics trace
    p.add_argument("--dynamics_trace_every", type=int, default=5,
                   help="Record dynamics trace every N steps (for checkpoint-level analysis)")
    
    return p.parse_args()


def main() -> None:
    args = build_args()
    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pool_words = args.example_pool_words
    selected_words = args.max_word_exposure
    if pool_words < selected_words:
        pool_words = selected_words

    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    example_jsonl_meta: dict = {}

    # ── Load data ──
    if args.example_jsonl:
        jsonl_path = Path(args.example_jsonl)
        examples, jsonl_total_words, jsonl_total_rows, sample_rows = load_examples_jsonl(jsonl_path, selected_words)
        actual_words = sum(ex.words for ex in examples)
        pool_words = jsonl_total_words
        manifest_files = [{
            "path": str(jsonl_path), "name": jsonl_path.name,
            "bytes": jsonl_path.stat().st_size, "sha256": sha256_file(jsonl_path),
            "whitespace_words": jsonl_total_words, "rows": jsonl_total_rows,
        }]
        total_words = jsonl_total_words
        if args.example_jsonl_meta:
            meta_path = Path(args.example_jsonl_meta)
            example_jsonl_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        example_selection_metadata = {
            "data_source_type": "example_jsonl",
            "example_jsonl": str(jsonl_path),
            "example_jsonl_label": args.example_jsonl_label,
        }
    else:
        # Official corpus full-cycle
        from huggingface_hub import snapshot_download
        raw_dir = out / "raw_dataset"
        raw_dir.mkdir(parents=True, exist_ok=True)
        local = Path(snapshot_download(
            repo_id=args.dataset_id, repo_type="dataset",
            revision=args.dataset_revision,
            allow_patterns=TRAIN_FILES + ["README.md"],
            local_dir=raw_dir, local_dir_use_symlinks=False,
        ))
        manifest_files = []
        for name in TRAIN_FILES:
            p = local / name
            if not p.exists():
                raise FileNotFoundError(f"missing: {p}")
            manifest_files.append({
                "path": str(p), "name": name, "bytes": p.stat().st_size,
                "sha256": sha256_file(p), "whitespace_words": count_words_in_file(p),
            })
        total_words = sum(f["whitespace_words"] for f in manifest_files)
        files = [local / n for n in TRAIN_FILES]
        
        if pool_words > total_words:
            pool_words = total_words
        pool_examples = list(iter_examples(files, pool_words, args.words_per_example))
        for i, ex in enumerate(pool_examples):
            ex.example_id = i
        
        if selected_words > total_words * 10:
            raise RuntimeError(f"exposure {selected_words} exceeds 10 epochs ({total_words * 10})")
        
        examples: list[Example] = []
        actual_words = 0
        epoch = 0
        epoch_metadata = []
        while actual_words < selected_words:
            epoch_examples = list(pool_examples)
            shuffle_seed = args.seed + 1000003 * epoch
            random.Random(shuffle_seed).shuffle(epoch_examples)
            before = actual_words
            epoch_take = 0
            for ex in epoch_examples:
                if actual_words >= selected_words:
                    break
                source = f"epoch{epoch + 1}::{ex.source}"
                if actual_words + ex.words <= selected_words:
                    examples.append(Example(ex.text, ex.words, example_id=ex.example_id, source=source))
                    actual_words += ex.words
                    epoch_take += 1
                else:
                    take = selected_words - actual_words
                    if take > 0:
                        examples.append(Example(" ".join(ex.text.split()[:take]), take,
                                                example_id=ex.example_id, source=source))
                        actual_words += take
                        epoch_take += 1
                    break
            epoch_metadata.append({"epoch_index": epoch + 1, "shuffle_seed": shuffle_seed,
                                   "words_added": actual_words - before, "examples_added": epoch_take})
            epoch += 1
        example_selection_metadata = {
            "data_source_type": "official_corpus_fullcycle",
            "selection_epochs": epoch_metadata,
        }

    if actual_words != selected_words:
        raise RuntimeError(f"word selection mismatch {actual_words} vs {selected_words}")

    # ── Dataset and DataLoader ──
    dataset = MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())
    total_steps = len(loader)

    # ── Token frequency ranks for dynamics trace ──
    token_freq_ranks = compute_token_frequency_ranks(examples, tokenizer, args.max_seq_length)

    # ── Initialize masking curriculum state ──
    curriculum_state = MaskingCurriculumState(
        curriculum=args.masking_curriculum,
        mask_prob_start=args.mask_prob_start,
        mask_prob_end=args.mask_prob_end,
        switch_frac=args.switch_frac,
        amlm_window=args.amlm_window,
        amlm_lambda=args.amlm_lambda,
    )
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=total_steps)

    # ── Model ──
    reset_all_rng = lambda s: (random.seed(s), torch.manual_seed(s),
                               torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None)
    reset_all_rng(args.seed)
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)
    
    param_count = sum(p.numel() for p in model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # ── Optimizer ──
    if args.optimizer == "lamb":
        if not LAMB_AVAILABLE:
            raise RuntimeError("LAMB requested but lamb_optimizer.py not found")
        optim = get_lamb_optimizer(model, lr=args.learning_rate, weight_decay=args.weight_decay)
        print(json.dumps({"event": "optimizer", "type": "LAMB", "lr": args.learning_rate}), flush=True)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.98))
        print(json.dumps({"event": "optimizer", "type": "AdamW", "lr": args.learning_rate}), flush=True)
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup,
                                            num_training_steps=schedule_total)

    # ── Seq-length schedule ──
    seq_schedule: list[tuple[float, int]] = []
    if args.seq_len_schedule:
        for part in args.seq_len_schedule.split(","):
            t, L = part.split(":")
            seq_schedule.append((float(t), int(L)))
        seq_schedule.sort()

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    # ── Training loop ──
    log_path = out / "training_log.jsonl"
    dynamics_path = out / "dynamics_traces.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict] = []
    dynamics_traces: list[dict] = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    
    # Record the manifests
    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    (out / "example_order_manifest.json").write_text(json.dumps({
        "seed": args.seed, "selected_for_training_words": actual_words,
        "num_consumed_examples": len(examples),
        "masking_curriculum": args.masking_curriculum,
        "mask_prob_start": args.mask_prob_start,
        "mask_prob_end": args.mask_prob_end,
        "switch_frac": args.switch_frac,
        "source_words_consumed": source_words,
        **example_selection_metadata,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            
            frac = (step - 1) / max(1, schedule_total)
            cur_len = seq_length_for_progress(frac, seq_schedule, args.seq_length) if seq_schedule else args.seq_length
            cur_len = min(cur_len, args.max_seq_length)
            input_ids = input_ids[:, :cur_len].contiguous()
            attention_mask = attention_mask[:, :cur_len].contiguous()
            word_group = word_group[:, :cur_len].contiguous()
            
            # Update curriculum state step counter
            curriculum_state.current_step = step - 1
            
            # Apply curriculum-controlled masking
            masked_inputs, labels = apply_masking_curriculum(
                input_ids, attention_mask, word_group, tokenizer, curriculum_state, gen
            )
            
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
            n_pred = int((labels != -100).sum().item())
            n_candidate = int((attention_mask.bool()).sum().item())
            effective_mask_rate = n_pred / max(1, n_candidate)
            
            # Record per-token accuracy for AMLM
            if curriculum_state.uses_amlm:
                with torch.no_grad():
                    curriculum_state.record_batch_accuracy(input_ids, labels, out_model.logits)
                curriculum_state.maybe_update_amlm_weights()
            
            # Dynamics trace at configurable intervals
            if step % args.dynamics_trace_every == 0:
                with torch.no_grad():
                    curriculum_state.record_dynamics_trace(
                        input_ids, labels, out_model.logits, loss_float,
                        token_freq_ranks, effective_mask_rate
                    )
            
            rec = {
                "step": step, "loss": loss_float, "lr": float(sched.get_last_lr()[0]),
                "batch_words": words, "cumulative_word_exposure": cumulative_words,
                "seq_len": cur_len, "masked_tokens": n_pred,
                "effective_mask_rate": round(effective_mask_rate, 4),
                "mask_mode": curriculum_state.get_current_mask_mode(),
                "mask_prob_nominal": round(curriculum_state.get_current_mask_prob(), 4),
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            
            # Checkpoint saving with dynamics trace flush
            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                if next_ckpt < 1_000_000:
                    name = "chck_1M"
                elif next_ckpt % 1_000_000 == 0:
                    name = f"chck_{next_ckpt // 1_000_000}M"
                else:
                    name = f"chck_{next_ckpt}w"
                cp = out / "hf_model" / name
                save_hf_checkpoint(model, tokenizer, cp)
                # Flush dynamics trace at checkpoint
                trace = curriculum_state.flush_dynamics_trace(name)
                dynamics_traces.append(trace)
                saved_checkpoints.append({
                    "name": name, "target_word_exposure": next_ckpt,
                    "actual_cumulative_word_exposure": cumulative_words, "path": str(cp),
                })
                print(json.dumps({"event": "checkpoint_saved", "name": name,
                                  "cum_words": cumulative_words,
                                  "mask_mode": curriculum_state.get_current_mask_mode(),
                                  "mask_prob": round(curriculum_state.get_current_mask_prob(), 4)}),
                      flush=True)
                next_ckpt += args.checkpoint_words

    # Final model save
    save_hf_checkpoint(model, tokenizer, out / "hf_model")
    if not saved_checkpoints:
        cp = out / "hf_model" / "chck_1M"
        save_hf_checkpoint(model, tokenizer, cp)
        saved_checkpoints.append({"name": "chck_1M", "target_word_exposure": args.checkpoint_words,
                                  "actual_cumulative_word_exposure": cumulative_words, "path": str(cp)})

    # Write dynamics traces
    with dynamics_path.open("w", encoding="utf-8") as f:
        for trace in dynamics_traces:
            f.write(json.dumps(trace) + "\n")

    # Scientific metrics
    metrics = {
        "variant": f"masking_curriculum_{args.masking_curriculum}",
        "backend": "mlm",
        "model_family": "DebertaV2ForMaskedLM",
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "word_exposure": cumulative_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": total_steps,
        "masking_curriculum": args.masking_curriculum,
        "mask_prob_start": args.mask_prob_start,
        "mask_prob_end": args.mask_prob_end,
        "switch_frac": args.switch_frac,
        "amlm_window": args.amlm_window if curriculum_state.uses_amlm else None,
        "n_amlm_updates": curriculum_state.n_amlm_updates,
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "ffn_mult": args.ffn_mult,
        "seed": args.seed,
        "seq_length": args.seq_length,
        "max_seq_length": args.max_seq_length,
        "seq_len_schedule": args.seq_len_schedule,
        "saved_checkpoints": saved_checkpoints,
        "dynamics_traces_file": str(dynamics_path),
        "source_words_consumed": source_words,
        **example_selection_metadata,
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(json.dumps({
        "event": "done", "param_count": param_count,
        "loss_first": metrics["loss_first"], "loss_last": metrics["loss_last"],
        "word_exposure": cumulative_words,
        "masking_curriculum": args.masking_curriculum,
        "n_amlm_updates": curriculum_state.n_amlm_updates,
        "checkpoints": [c["name"] for c in saved_checkpoints],
    }), flush=True)


if __name__ == "__main__":
    main()
