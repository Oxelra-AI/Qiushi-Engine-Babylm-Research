#!/usr/bin/env python3
"""research: AMLM (Adaptive Masked Language Modeling) extension for BabyLM 2x2 screen.

Extends the existing masked trainer with:
1. Mask decay schedule: overall masking probability decays linearly from mask_prob_start
   to mask_prob_end over training (Edman et al. 2025, AMLM).
2. Per-token adaptation (hard method): tracks per-token accuracy over a sliding window,
   increases masking probability for tokens the model gets wrong more often.
   
Timescale fix: window size is configurable to match the step budget.
At 20M exposure (2 epochs, ~397 steps), window=10 gives ~39 AMLM updates.

Usage:
  python amlm_trainer.py \
    --example_jsonl <path> --max_word_exposure 20000000 \
    --mask_mode wwm --amlm_mode hard --amlm_window 10 \
    --mask_prob_start 0.4 --mask_prob_end 0.15 \
    --model_type deberta_v2 --hidden_size 480 --n_layer 8 --n_head 8 \
    --batch_size 256 --max_seq_length 256 --seed 42 ...
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import sys, os, pathlib

# Import the existing trainer's infrastructure
SCRIPTS_DIR = _public_path('experiments/archive/initial_model_studies/training/scripts')
sys.path.insert(0, str(SCRIPTS_DIR))
from babylm_masked_train_fullcycle import (
    build_args, main as original_main,
    apply_masking, MaskedChunkDataset, collate, Example,
    make_portable_tokenizer, download_dataset, iter_examples,
    load_examples_jsonl, count_words_in_file, sha256_file,
    seq_length_for_progress, summarize_tokenization_coupling,
    BASELINE_TOKENIZER_REPO, TRAIN_FILES,
)

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, PreTrainedTokenizerFast,
    BertConfig, BertForMaskedLM,
    DebertaV2Config, DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)


class AMLMState:
    """Tracks per-token masking probabilities and accuracy statistics."""
    
    def __init__(self, vocab_size: int, mode: str = "hard", window: int = 10,
                 mask_prob_start: float = 0.4, mask_prob_end: float = 0.15,
                 lambda_smooth: float = 0.2, total_steps: int = 1000):
        self.vocab_size = vocab_size
        self.mode = mode  # "hard" or "none"
        self.window = window
        self.mask_prob_start = mask_prob_start
        self.mask_prob_end = mask_prob_end
        self.lambda_smooth = lambda_smooth
        self.total_steps = total_steps
        
        # Per-token masking weights (initialized uniform)
        self.token_weights = torch.ones(vocab_size) * mask_prob_start
        
        # Accuracy tracking for current window
        self.correct_counts = torch.zeros(vocab_size, dtype=torch.long)
        self.total_counts = torch.zeros(vocab_size, dtype=torch.long)
        self.batches_in_window = 0
        self.global_step = 0
        self.n_updates = 0
    
    def get_current_base_prob(self) -> float:
        """Linear decay from mask_prob_start to mask_prob_end."""
        if self.total_steps <= 1:
            return self.mask_prob_start
        frac = min(self.global_step / self.total_steps, 1.0)
        return self.mask_prob_start + (self.mask_prob_end - self.mask_prob_start) * frac
    
    def record_batch(self, input_ids: torch.Tensor, labels: torch.Tensor, 
                     logits: torch.Tensor):
        """Record accuracy for masked tokens in this batch."""
        if self.mode == "none":
            return
        
        # Only count positions where labels != -100 (masked positions).
        # Use vectorized bincounts; Python loops over masked tokens are too slow for
        # the 10M factorial screen.
        mask = labels != -100
        if mask.sum() == 0:
            self.batches_in_window += 1
            return
        masked_ids = labels[mask].detach().cpu().long()
        predicted = logits[mask].argmax(dim=-1).detach().cpu().long()
        correct_ids = masked_ids[predicted == masked_ids]
        total_bc = torch.bincount(masked_ids, minlength=self.vocab_size)
        correct_bc = torch.bincount(correct_ids, minlength=self.vocab_size) if correct_ids.numel() else torch.zeros(self.vocab_size, dtype=torch.long)
        self.total_counts += total_bc
        self.correct_counts += correct_bc
        self.batches_in_window += 1
    
    def maybe_update_weights(self):
        """Update per-token masking weights if window is complete."""
        if self.mode == "none":
            self.global_step += 1
            return False
        
        self.global_step += 1
        if self.batches_in_window < self.window:
            return False
        
        # Compute per-token accuracy for tokens that were seen
        base_prob = self.get_current_base_prob()
        seen = self.total_counts > 0
        
        if seen.sum() > 0:
            accuracy = torch.zeros(self.vocab_size)
            accuracy[seen] = self.correct_counts[seen].float() / self.total_counts[seen].float()
            
            # Hard method: weight = base_prob * (1 - accuracy)
            # Tokens with high accuracy get masked LESS; tokens with low accuracy get masked MORE
            new_weights = base_prob * (1.0 - accuracy)
            # Clamp to reasonable range
            new_weights = new_weights.clamp(min=0.01, max=0.8)
            
            # Exponential smoothing
            self.token_weights[seen] = (
                self.lambda_smooth * self.token_weights[seen] + 
                (1 - self.lambda_smooth) * new_weights[seen]
            )
        
        # For unseen tokens, just use base probability
        unseen = ~seen
        self.token_weights[unseen] = base_prob
        
        # Reset window
        self.correct_counts.zero_()
        self.total_counts.zero_()
        self.batches_in_window = 0
        self.n_updates += 1
        return True
    
    def get_per_token_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get masking probability for each position based on token identity."""
        if self.mode == "none":
            return torch.full_like(input_ids, self.get_current_base_prob(), dtype=torch.float32)
        return self.token_weights[input_ids.cpu()].to(input_ids.device)


def apply_masking_amlm(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       word_group: torch.Tensor, tokenizer, mask_mode: str,
                       amlm_state: AMLMState, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """AMLM-aware masking: uses per-token probabilities instead of uniform."""
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)
    
    # Get per-token masking probabilities
    token_probs = amlm_state.get_per_token_probs(input_ids)
    
    if mask_mode == "token":
        probs = torch.rand(bsz, seq, generator=gen, device=device)
        select = candidate & (probs < token_probs)
    elif mask_mode == "wwm":
        # For WWM, use the max probability within each word group
        for b in range(bsz):
            groups = word_group[b]
            valid_groups = torch.unique(groups[groups >= 0])
            if valid_groups.numel() == 0:
                continue
            # For each group, compute group masking probability as max of its tokens
            group_probs = torch.zeros(valid_groups.max().item() + 1, device=device)
            for gid in valid_groups:
                g_mask = (groups == gid) & candidate[b]
                if g_mask.any():
                    group_probs[gid] = token_probs[b][g_mask].max()
            gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
            chosen = valid_groups[gp < group_probs[valid_groups]]
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


def add_amlm_args(parser: argparse.ArgumentParser):
    """Add AMLM-specific arguments to the existing parser."""
    parser.add_argument("--amlm_mode", choices=["none", "hard"], default="none",
                       help="AMLM adaptation mode: none=standard fixed masking, hard=accuracy-based adaptation")
    parser.add_argument("--amlm_window", type=int, default=10,
                       help="Batches per AMLM update window (paper: 200; scaled for short screens)")
    parser.add_argument("--mask_prob_start", type=float, default=0.15,
                       help="Initial masking probability (AMLM: 0.4, standard: 0.15)")
    parser.add_argument("--mask_prob_end", type=float, default=0.15,
                       help="Final masking probability after linear decay (AMLM: 0.15)")
    parser.add_argument("--amlm_lambda", type=float, default=0.2,
                       help="Exponential smoothing factor for AMLM weight updates")
    return parser


if __name__ == "__main__":
    # This file provides the AMLMState class and apply_masking_amlm function
    # to be imported by the 4-arm launcher. It also validates that the import works.
    print("AMLM module loaded successfully.")
    print(f"  AMLMState: vocab_size, mode, window, mask_prob_start/end, lambda_smooth")
    print(f"  apply_masking_amlm: AMLM-aware masking with per-token probabilities")
    print(f"  add_amlm_args: argument parser extension")
