#!/usr/bin/env python3
"""research fallback: Evidence-Visible Masking continuation trainer.

Instead of uniform 15% WWM masking, preferentially masks tokens in entity/relation/
attribute positions while keeping context tokens visible. This focuses the gradient
on learning entity-specific knowledge (EWoK/COMPS/GlobalPIQA targets).

Token priority classification (corpus-internal only):
  HIGH (mask at ~30%): Named entity tokens, numbers in relational context,
                       object nouns after relational verbs
  NORMAL (mask at ~8%): Function words, common verbs, determiners

Total masked tokens per sequence stays approximately equal to standard 15% MLM.

Uses same 80M→100M continuation framework as cluster test.
Does NOT use any downstream evaluation labels, AoA/CDI items.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

SEQ_LENGTH = 256
TARGET_MASK_FRAC = 0.15  # same total masked fraction as standard
HIGH_PRIORITY_MASK_PROB = 0.30  # mask entity/relation positions more
LOW_PRIORITY_MASK_PROB = 0.08   # mask function words less
ORIGINAL_TOTAL_STEPS = 2515
WARMUP_FRACTION = 0.06
BASE_LR = 0.001
WEIGHT_DECAY = 0.01
CONTINUATION_START_EXPOSURE = 80_000_000
CONTINUATION_BUDGET = 20_000_000
CHECKPOINT_INTERVAL = 5_000_000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ─── Token Priority Classification ────────────────────────────────────────────

# Simple heuristics for identifying evidence-requiring positions
COPULA_VERBS = {"is", "are", "was", "were", "be", "been", "being", "became", "becomes"}
RELATIONAL_VERBS = {"built", "discovered", "invented", "founded", "won", "lost",
                    "defeated", "created", "wrote", "composed", "designed", "led",
                    "ruled", "conquered", "established", "produced", "made", "has",
                    "had", "contains", "includes", "reaches", "measures", "weighs"}
FUNCTION_WORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                  "from", "with", "by", "of", "is", "are", "was", "were", "be", "been",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "can", "shall", "that", "this", "it", "its"}


def classify_token_priorities(token_strs: list[str], input_ids: list[int],
                              special_ids: set) -> list[float]:
    """Assign mask priority to each token position based on linguistic role.
    
    Returns a list of priorities (0.0 = never mask, 1.0 = high priority).
    """
    n = len(token_strs)
    priorities = [0.5] * n  # default: medium priority
    
    # Track state for relational patterns
    after_copula = False
    after_relational = False
    
    for i, (tok_str, tid) in enumerate(zip(token_strs, input_ids)):
        if tid in special_ids:
            priorities[i] = 0.0
            continue
        
        # Clean token for analysis
        clean = tok_str.replace("Ġ", "").replace("▁", "").strip()
        lower = clean.lower()
        
        # Named entity detection: capitalized word-start tokens (not sentence-initial)
        is_word_start = tok_str.startswith("Ġ") or tok_str.startswith("▁")
        is_capitalized = bool(clean) and clean[0].isupper() and len(clean) > 1
        is_sentence_start = (i <= 1) or (i > 0 and token_strs[i-1].rstrip().endswith((".", "!", "?")))
        
        if is_capitalized and is_word_start and not is_sentence_start:
            # Named entity token → high priority
            priorities[i] = 1.0
            # Also mark continuation tokens of the entity
            for j in range(i + 1, min(i + 4, n)):
                next_tok = token_strs[j].replace("Ġ", "").replace("▁", "").strip()
                if next_tok and not token_strs[j].startswith("Ġ") and not token_strs[j].startswith("▁"):
                    priorities[j] = 0.9  # continuation of entity
                else:
                    break
        
        # Numbers in context → high priority (facts, measurements)
        elif bool(re.match(r'\d', clean)):
            priorities[i] = 0.9
        
        # Token after copula/relational verb → attribute/object (high priority)
        elif after_copula or after_relational:
            if lower not in FUNCTION_WORDS and len(clean) > 2:
                priorities[i] = 0.85
            after_copula = False
            after_relational = False
        
        # Function words → low priority
        elif lower in FUNCTION_WORDS:
            priorities[i] = 0.2
        
        # Update state
        if lower in COPULA_VERBS:
            after_copula = True
        elif lower in RELATIONAL_VERBS:
            after_relational = True
        else:
            after_copula = False
            after_relational = False
    
    return priorities


def evidence_visible_mask_batch(input_ids: torch.Tensor, tokenizer,
                                target_mask_frac: float, gen: torch.Generator,
                                device: torch.device):
    """Evidence-visible masking: preferentially mask entity/relation positions."""
    batch_size, seq_len = input_ids.shape
    special_ids = set(tokenizer.all_special_ids)
    mask_id = tokenizer.mask_token_id
    
    masked_inputs = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    for i in range(batch_size):
        # Get token strings for priority classification
        ids = input_ids[i].tolist()
        token_strs = tokenizer.convert_ids_to_tokens(ids)
        
        # Classify priorities
        priorities = classify_token_priorities(token_strs, ids, special_ids)
        
        # Compute per-position mask probabilities
        # Scale so total expected masks ≈ target_mask_frac * non-special tokens
        n_valid = sum(1 for p in priorities if p > 0)
        if n_valid == 0:
            continue
        
        # Normalize: high-priority positions get higher prob, low get lower
        # Total budget: target_mask_frac * n_valid tokens
        target_masks = target_mask_frac * n_valid
        raw_probs = []
        for p in priorities:
            if p <= 0:
                raw_probs.append(0.0)
            elif p >= 0.8:
                raw_probs.append(HIGH_PRIORITY_MASK_PROB)
            elif p <= 0.3:
                raw_probs.append(LOW_PRIORITY_MASK_PROB)
            else:
                raw_probs.append(target_mask_frac)
        
        # Scale to match target total masks
        raw_sum = sum(raw_probs)
        if raw_sum > 0:
            scale = target_masks / raw_sum
            mask_probs = [min(p * scale, 0.5) for p in raw_probs]
        else:
            mask_probs = [target_mask_frac] * seq_len
        
        # Sample mask positions
        for j in range(seq_len):
            if mask_probs[j] <= 0:
                continue
            if torch.rand(1, generator=gen).item() < mask_probs[j]:
                labels[i, j] = input_ids[i, j]
                # 80% mask, 10% random, 10% keep
                r = torch.rand(1, generator=gen).item()
                if r < 0.8:
                    masked_inputs[i, j] = mask_id
                elif r < 0.9:
                    masked_inputs[i, j] = torch.randint(len(tokenizer), (1,), generator=gen).item()
    
    return masked_inputs.to(device), attention_mask.to(device), labels.to(device)


# The rest of the training loop would be identical to continuation_trainer.py
# but calling evidence_visible_mask_batch instead of wwm_mask_batch.
# To avoid full duplication, we can import and override.

if __name__ == "__main__":
    print("Evidence-visible masking module ready for integration.")
    print("To use: replace wwm_mask_batch with evidence_visible_mask_batch in continuation trainer.")
    print("Controls needed: C1 (uniform WWM), C2 (random priority), C3 (inverse priority)")
