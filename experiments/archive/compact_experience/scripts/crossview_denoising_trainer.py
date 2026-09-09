#!/usr/bin/env python3
"""research: Cross-view conditional denoising trainer for BabyLM Strict-Small.

Scientific question: does the same-window paired-experience benefit come from
genuine semantic cross-view reconstruction, or from lexical redundancy (copying
identical tokens visible in the partner view)?

Arms (all matched on total mask budget, corpus, tokenizer, model, optimizer, seed):
  baseline       - Standard random WWM everywhere (= clean-Qwen recipe)
  anchor_copy    - On pair rows, preferentially mask content-anchor words (shared
                   between original and rewrite) → model can "copy" from partner
  semantic_xview - On pair rows, preferentially mask non-anchor words (paraphrastic
                   predicates/attributes unique to one view) → model must reconstruct
                   using partner view's different-but-equivalent expression
  partner_shuffle - Same masking as semantic_xview, but partner segments are shuffled
                   (random other pair's rewrite), breaking semantic correspondence

All arms:
  - Same total 15% WWM mask budget per row
  - Same corpus/tokenizer/model/optimizer/seed/exposure
  - Official (non-pair) rows always get standard random WWM
  - Only the distribution of masks within pair rows differs

"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

# ─── Constants ────────────────────────────────────────────────────────────────
MASK_PROB = 0.15
ARMS = ["baseline", "anchor_copy", "semantic_xview", "partner_shuffle"]
WORD_RE = re.compile(r"\S+")

# Top-200 English function words (not useful as copy targets)
FUNCTION_WORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "about", "against", "up", "down", "out", "off", "over", "under",
    "that", "this", "these", "those", "it", "its", "he", "she", "they",
    "them", "his", "her", "their", "we", "us", "our", "you", "your",
    "i", "me", "my", "who", "whom", "which", "what", "that", "s", "t",
    "re", "ve", "ll", "d", "m", "don", "didn", "doesn", "won", "wasn",
    "also", "still", "already", "even", "much", "well", "back", "also",
])


# ─── Utilities ────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_word(w: str) -> str:
    """Normalize for anchor matching: lowercase, strip peripheral punctuation."""
    return w.lower().strip(".,!?;:'\"()-[]{}/*&^%$#@~`+=<>|\\")


def is_content_anchor(word: str) -> bool:
    """True if a shared word is a content anchor (not a function word)."""
    return normalize_word(word) not in FUNCTION_WORDS and len(normalize_word(word)) > 1


# ─── Pair structure analysis ─────────────────────────────────────────────────
@dataclass
class PairSegment:
    pair_id: str
    side: str  # "original" or "rewrite"
    start_char: int
    end_char: int
    words: List[str]
    word_char_spans: List[Tuple[int, int]]  # (start, end) for each word
    anchor_mask: List[bool]  # True if word is an anchor (shared with partner)
    content_anchor_mask: List[bool]  # True if word is a content anchor


@dataclass
class PairRowStructure:
    """Pre-computed structure for one pair row."""
    example_id: int
    segments: List[PairSegment]
    n_pairs: int
    total_words: int
    anchor_words: int
    content_anchor_words: int
    nonanchor_words: int


def build_pair_row_structure(
    row_text: str,
    pair_ids: List[str],
    pairs_db: Dict[str, Dict[str, Any]],
    example_id: int,
) -> PairRowStructure:
    """Reconstruct segment structure and compute anchor labels for a pair row."""
    segments: List[PairSegment] = []
    pos = 0
    
    for pid in pair_ids:
        pair = pairs_db[pid]
        orig_text = str(pair["original"])
        rew_text = str(pair["rewrite"])
        
        # Compute shared normalized words between this pair
        orig_normed = set(normalize_word(w) for w in orig_text.split())
        rew_normed = set(normalize_word(w) for w in rew_text.split())
        shared_normed = orig_normed & rew_normed
        
        for side, seg_text in [("original", orig_text), ("rewrite", rew_text)]:
            # Advance past inter-segment spaces
            while pos < len(row_text) and row_text[pos] == " ":
                pos += 1
            
            # Locate segment in row text
            if row_text.startswith(seg_text, pos):
                start = pos
            else:
                found = row_text.find(seg_text, max(0, pos - 10))
                if found < 0:
                    # Fallback: approximate position
                    start = pos
                else:
                    start = found
            end = start + len(seg_text)
            
            # Parse words and their character spans
            words = []
            word_spans = []
            for m in WORD_RE.finditer(seg_text):
                w = m.group()
                words.append(w)
                word_spans.append((start + m.start(), start + m.end()))
            
            # Compute anchor labels
            anchor_mask = [normalize_word(w) in shared_normed for w in words]
            content_anchor_mask = [
                (normalize_word(w) in shared_normed) and is_content_anchor(w)
                for w in words
            ]
            
            segments.append(PairSegment(
                pair_id=pid, side=side,
                start_char=start, end_char=end,
                words=words, word_char_spans=word_spans,
                anchor_mask=anchor_mask,
                content_anchor_mask=content_anchor_mask,
            ))
            pos = end
    
    total_words = sum(len(s.words) for s in segments)
    anchor_words = sum(sum(s.anchor_mask) for s in segments)
    content_anchor_words = sum(sum(s.content_anchor_mask) for s in segments)
    nonanchor_words = total_words - anchor_words
    
    return PairRowStructure(
        example_id=example_id,
        segments=segments,
        n_pairs=len(pair_ids),
        total_words=total_words,
        anchor_words=anchor_words,
        content_anchor_words=content_anchor_words,
        nonanchor_words=nonanchor_words,
    )


# ─── Tokenizer ───────────────────────────────────────────────────────────────
def make_portable_tokenizer(tokenizer_path: str = "") -> PreTrainedTokenizerFast:
    if tokenizer_path:
        base = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    else:
        base = AutoTokenizer.from_pretrained(
            "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict", revision="main", use_fast=True)
    backend = getattr(base, "_tokenizer", None) or getattr(base, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError(f"No fast backend for {tokenizer_path}")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=base.bos_token or "<s>",
        eos_token=base.eos_token or "</s>",
        unk_token=base.unk_token or "<unk>",
        pad_token=base.pad_token or "<pad>",
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


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


# ─── Dataset ─────────────────────────────────────────────────────────────────
@dataclass
class Example:
    text: str
    words: int
    example_id: int = -1
    source: str = ""
    is_pair_row: bool = False
    pair_structure: Optional[PairRowStructure] = None


class CrossViewDataset(Dataset):
    """Dataset that provides word groups + pair-aware priority labels."""
    
    def __init__(self, examples: List[Example], tokenizer, seq_length: int, arm: str):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.arm = arm
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start_cache: Dict[int, bool] = {}
    
    def _is_word_start(self, token_id: int) -> bool:
        v = self._word_start_cache.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start_cache[token_id] = v
        return v
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def _compute_token_priorities(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        text: str, pair_structure: PairRowStructure
    ) -> torch.Tensor:
        """Map pair structure to per-token priority labels.
        
        Priority values encode both lexical class and view:
          0 = ordinary/function-anchor
          1 = content anchor in original; 2 = non-anchor in original
          3 = content anchor in rewrite;  4 = non-anchor in rewrite
        This lets masking select one target view while preserving the partner view.
        """
        seq_len = input_ids.shape[0]
        priority = torch.zeros(seq_len, dtype=torch.long)
        
        # Get offset mapping for character-to-token alignment
        enc = self.tokenizer(
            text, add_special_tokens=False, truncation=True,
            max_length=self.seq_length, return_offsets_mapping=True
        )
        offsets = enc["offset_mapping"]  # list of (start, end) per token
        
        # Build character→word_label mapping from pair structure
        # Label: 0=not in any segment, 1=content_anchor, 2=non-anchor, 3=function_anchor
        char_labels = [0] * len(text)
        for seg in pair_structure.segments:
            for i, (ws, we) in enumerate(seg.word_char_spans):
                if ws >= len(text) or we > len(text):
                    continue
                if not seg.anchor_mask[i]:
                    # Non-identical paraphrastic target, view-coded.
                    label = 2 if seg.side == "original" else 4
                elif seg.content_anchor_mask[i]:
                    # Identical content anchor, view-coded.
                    label = 1 if seg.side == "original" else 3
                else:
                    # Function anchor: not a useful lexical-copy target.
                    label = 0
                for c in range(ws, min(we, len(text))):
                    char_labels[c] = label
        
        # Map to tokens using offsets
        for tok_idx, (cs, ce) in enumerate(offsets):
            if tok_idx >= seq_len:
                break
            if cs == ce:
                continue
            # Majority label for this token's character span
            span_labels = char_labels[cs:ce]
            if span_labels:
                # Use max label (prefer non-anchor > content_anchor > ordinary)
                priority[tok_idx] = max(span_labels)
        
        return priority
    
    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text, add_special_tokens=False, truncation=True,
            max_length=self.seq_length, padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        
        # Compute word groups (same as masking_curriculum_trainer)
        group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if attention_mask[i] == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._is_word_start(tid) or i == 0:
                gid += 1
            group[i] = gid
        
        # Compute priority labels for pair rows
        if ex.is_pair_row and ex.pair_structure is not None and self.arm != "baseline":
            priority = self._compute_token_priorities(
                input_ids, attention_mask, ex.text, ex.pair_structure
            )
        else:
            priority = torch.zeros_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": group,
            "priority": priority,
            "is_pair_row": torch.tensor(1 if ex.is_pair_row else 0, dtype=torch.long),
            "words": ex.words,
        }


def collate(batch: List[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "priority": torch.stack([x["priority"] for x in batch]),
        "is_pair_row": torch.stack([x["is_pair_row"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
    }


# ─── Priority-based WWM masking ──────────────────────────────────────────────
def apply_crossview_masking(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    priority: torch.Tensor,
    is_pair_row: torch.Tensor,
    arm: str,
    tokenizer,
    gen: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Apply priority-based WWM masking.
    
    Returns: (masked_inputs, labels, stats_dict)
    """
    device = input_ids.device
    bsz, seq = input_ids.shape
    labels = input_ids.clone()
    mask_token_id = tokenizer.mask_token_id
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)
    
    stats = {"anchor_masked": 0, "nonanchor_masked": 0, "other_masked": 0,
             "original_view_masked": 0, "rewrite_view_masked": 0,
             "pair_rows_in_batch": 0, "total_masked": 0}
    
    for b in range(bsz):
        groups = word_group[b]
        valid_groups = torch.unique(groups[groups >= 0])
        if valid_groups.numel() == 0:
            continue
        
        is_pair = bool(is_pair_row[b].item())
        use_priority = is_pair and arm in ("anchor_copy", "semantic_xview", "partner_shuffle")
        
        if is_pair:
            stats["pair_rows_in_batch"] += 1
        
        # Draw the same baseline Bernoulli-WWM selection in every arm. Priority
        # arms may only swap a selected non-target word for an unselected target
        # word with the SAME subword length. Thus both masked-word count and
        # masked-subword-token count remain exactly identical row-by-row.
        gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
        base_chosen = valid_groups[gp < MASK_PROB]
        if base_chosen.numel() == 0:
            base_chosen = valid_groups[:1]

        if not use_priority:
            chosen = base_chosen
        else:
            pri = priority[b]
            group_priorities = torch.zeros(valid_groups.max().item() + 1,
                                          dtype=torch.long, device=device)
            group_sizes = torch.zeros_like(group_priorities)
            for gid in valid_groups:
                g_mask = (groups == gid) & candidate[b]
                if g_mask.any():
                    group_priorities[gid] = pri[g_mask].max()
                    group_sizes[gid] = g_mask.sum()

            # Choose exactly one target view per row; the partner view stays
            # visible. Direction is balanced in expectation and reported.
            target_original = bool(torch.rand((), generator=gen, device=device) < 0.5)
            if arm == "anchor_copy":
                target_priority = 1 if target_original else 3
            else:
                target_priority = 2 if target_original else 4

            chosen_list = [int(x) for x in base_chosen.tolist()]
            chosen_set = set(chosen_list)
            # Randomized candidate order, while matching subword span length.
            target_pool = [int(g) for g in valid_groups.tolist()
                           if int(g) not in chosen_set and int(group_priorities[int(g)]) == target_priority]
            if target_pool:
                perm = torch.randperm(len(target_pool), generator=gen, device=device).tolist()
                target_pool = [target_pool[i] for i in perm]
            selected_non_target = [g for g in chosen_list
                                   if int(group_priorities[g]) != target_priority]
            if selected_non_target:
                perm = torch.randperm(len(selected_non_target), generator=gen, device=device).tolist()
                selected_non_target = [selected_non_target[i] for i in perm]
            used_targets: Set[int] = set()
            for old_gid in selected_non_target:
                old_size = int(group_sizes[old_gid])
                replacement = next((g for g in target_pool
                                    if g not in used_targets and int(group_sizes[g]) == old_size), None)
                if replacement is not None:
                    chosen_list[chosen_list.index(old_gid)] = replacement
                    used_targets.add(replacement)
            chosen = torch.tensor(chosen_list, device=device, dtype=valid_groups.dtype)
        
        if chosen.numel() == 0:
            continue
        
        sel_b = torch.isin(groups, chosen) & candidate[b]
        select[b] = sel_b
        
        # Track stats
        if is_pair and use_priority:
            pri_sel = priority[b][sel_b]
            stats["anchor_masked"] += int(((pri_sel == 1) | (pri_sel == 3)).sum().item())
            stats["nonanchor_masked"] += int(((pri_sel == 2) | (pri_sel == 4)).sum().item())
            stats["original_view_masked"] += int(((pri_sel == 1) | (pri_sel == 2)).sum().item())
            stats["rewrite_view_masked"] += int(((pri_sel == 3) | (pri_sel == 4)).sum().item())
            stats["other_masked"] += int((pri_sel == 0).sum().item())
    
    # Ensure at least one token masked
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
    
    stats["total_masked"] = int(select.sum().item())
    labels[~select] = -100
    
    # 80/10/10 replacement
    masked_inputs = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = mask_token_id
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),),
                                 generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids
    
    return masked_inputs, labels, stats


# ─── Model ────────────────────────────────────────────────────────────────────
def build_model(args, tokenizer):
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * args.ffn_mult,
        max_position_embeddings=max(args.max_position_embeddings, args.seq_length + 8),
        max_relative_positions=args.max_relative_positions,
        position_buckets=args.position_buckets,
        relative_attention=True,
        pos_att_type=[x.strip() for x in args.pos_att_type.split(",") if x.strip()],
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
    # Force portable config
    tok_cfg = dst / "tokenizer_config.json"
    if tok_cfg.exists():
        cfg = json.loads(tok_cfg.read_text(encoding="utf-8"))
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        tok_cfg.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_pairs_db(path: Path) -> Dict[str, Dict[str, Any]]:
    db = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                db[str(p["pair_id"])] = p
    return db


def load_pair_meta(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load pair row metadata indexed by example_id."""
    meta = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                meta[int(row["example_id"])] = row
    return meta


def make_partner_shuffled_db(pairs_db: Dict[str, Dict[str, Any]], seed: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Replace each rewrite by another pair's rewrite of identical word count.

    Equal whitespace-word count preserves every packed row's official word budget.
    Mapping is a deterministic derangement within rewrite-length strata whenever
    a stratum has >1 member. Original strings and pair-row locations are unchanged.
    """
    by_len: Dict[int, List[str]] = defaultdict(list)
    for pid, p in pairs_db.items():
        by_len[len(str(p["rewrite"]).split())].append(pid)
    rng = random.Random(seed + 570057)
    mapping: Dict[str, str] = {}
    for _, ids in sorted(by_len.items()):
        ids = list(ids)
        rng.shuffle(ids)
        if len(ids) > 1:
            donors = ids[1:] + ids[:1]
        else:
            donors = ids
        mapping.update(dict(zip(ids, donors)))
    shuffled: Dict[str, Dict[str, Any]] = {}
    for pid, p in pairs_db.items():
        donor = mapping[pid]
        q = dict(p)
        q["rewrite"] = str(pairs_db[donor]["rewrite"])
        q["shuffle_donor_pair_id"] = donor
        shuffled[pid] = q
    return shuffled, mapping


def materialize_partner_shuffled_text(text: str, pair_ids: List[str], original_db: Dict[str, Dict[str, Any]], shuffled_db: Dict[str, Dict[str, Any]]) -> str:
    """Rebuild one packed row as original_i + shuffled_rewrite_i segments."""
    segments: List[str] = []
    for pid in pair_ids:
        segments.extend([str(original_db[pid]["original"]), str(shuffled_db[pid]["rewrite"])])
    rebuilt = " ".join(segments)
    if len(rebuilt.split()) != len(text.split()):
        raise RuntimeError("Partner shuffle changed whitespace-word count")
    return rebuilt


def load_examples(
    jsonl_path: Path,
    selected_words: int,
    pair_meta: Dict[int, Dict[str, Any]],
    pairs_db: Dict[str, Dict[str, Any]],
    arm: str,
    shuffle_seed: int = -1,
    original_pairs_db: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Example], Dict[str, Any]]:
    """Load examples from JSONL, annotating pair rows with structure info."""
    examples: List[Example] = []
    selected = 0
    total_rows = 0
    pair_rows_found = 0
    pair_structure_errors = 0
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            total_rows += 1
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            
            if selected >= selected_words:
                break
            if selected + words > selected_words:
                raise RuntimeError(
                    f"JSONL selection requires partial example at row {total_rows}: "
                    f"selected={selected} words={words} target={selected_words}")
            
            ex_id = int(obj.get("example_id", total_rows - 1))
            source = str(obj.get("source", ""))
            
            # Check if this is a pair row
            is_pair = ex_id in pair_meta
            pair_struct = None
            
            if is_pair and arm != "baseline":
                meta = pair_meta[ex_id]
                pair_ids = list(meta["pair_ids"])
                try:
                    if arm == "partner_shuffle":
                        if original_pairs_db is None:
                            raise RuntimeError("partner_shuffle requires original_pairs_db")
                        text = materialize_partner_shuffled_text(
                            text, pair_ids, original_pairs_db, pairs_db)
                    pair_struct = build_pair_row_structure(text, pair_ids, pairs_db, ex_id)
                    pair_rows_found += 1
                except Exception:
                    pair_structure_errors += 1
                    is_pair = False
            elif is_pair:
                pair_rows_found += 1
            
            examples.append(Example(
                text=text, words=words, example_id=ex_id, source=source,
                is_pair_row=is_pair, pair_structure=pair_struct,
            ))
            selected += words
    
    if selected != selected_words:
        raise RuntimeError(f"Selected {selected} != target {selected_words}")
    
    load_meta = {
        "total_rows_scanned": total_rows,
        "examples_loaded": len(examples),
        "pair_rows_found": pair_rows_found,
        "pair_structure_errors": pair_structure_errors,
        "selected_words": selected,
    }
    return examples, load_meta


# ─── Main ─────────────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser(description="research Cross-View Denoising Trainer")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--selected_words", type=int, required=True)
    p.add_argument("--pair_meta", required=True, help="Path to qwen_pair_packed_rows_meta.jsonl")
    p.add_argument("--pairs_db", required=True, help="Path to selected_pairs.jsonl")
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    
    # Model (DeBERTa-v2 8x480)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--pos_att_type", default="p2c,c2p")
    p.add_argument("--seq_length", type=int, default=256)
    
    # Optimization
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    
    # Reproducibility
    p.add_argument("--seed", type=int, default=43022)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def main():
    args = build_args()
    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print(json.dumps({"event": "start", "arm": args.arm, "output_dir": str(out),
                      "selected_words": args.selected_words}), flush=True)
    
    # Load tokenizer
    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    
    # Load pair database
    original_pairs_db = load_pairs_db(Path(args.pairs_db))
    pair_meta = load_pair_meta(Path(args.pair_meta))
    shuffle_mapping: Dict[str, str] = {}
    if args.arm == "partner_shuffle":
        pairs_db, shuffle_mapping = make_partner_shuffled_db(original_pairs_db, args.seed)
    else:
        pairs_db = original_pairs_db
    print(json.dumps({"event": "pairs_loaded", "n_pairs": len(pairs_db),
                      "n_pair_rows": len(pair_meta),
                      "partner_shuffle_changed": sum(k != v for k, v in shuffle_mapping.items())}), flush=True)
    
    # Load examples
    shuffle_seed = args.seed if args.arm == "partner_shuffle" else -1
    examples, load_meta = load_examples(
        Path(args.example_jsonl), args.selected_words,
        pair_meta, pairs_db, args.arm, shuffle_seed=shuffle_seed,
        original_pairs_db=original_pairs_db
    )
    print(json.dumps({"event": "data_loaded", **load_meta}), flush=True)
    
    # Compute pair structure statistics
    pair_examples = [e for e in examples if e.is_pair_row and e.pair_structure]
    if pair_examples:
        anchor_total = sum(e.pair_structure.anchor_words for e in pair_examples)
        content_anchor_total = sum(e.pair_structure.content_anchor_words for e in pair_examples)
        nonanchor_total = sum(e.pair_structure.nonanchor_words for e in pair_examples)
        pair_word_total = sum(e.pair_structure.total_words for e in pair_examples)
        structure_stats = {
            "pair_rows_with_structure": len(pair_examples),
            "pair_word_total": pair_word_total,
            "anchor_fraction": anchor_total / max(1, pair_word_total),
            "content_anchor_fraction": content_anchor_total / max(1, pair_word_total),
            "nonanchor_fraction": nonanchor_total / max(1, pair_word_total),
        }
    else:
        structure_stats = {"pair_rows_with_structure": 0}
    print(json.dumps({"event": "structure_stats", **structure_stats}), flush=True)
    
    # Dataset and DataLoader
    dataset = CrossViewDataset(examples, tokenizer, args.seq_length, args.arm)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    total_steps = len(loader)
    
    # Model
    def reset_rng(s):
        random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    
    reset_rng(args.seed)
    model = build_model(args, tokenizer)
    param_count = sum(p.numel() for p in model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay, betas=(0.9, 0.98))
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup,
                                            num_training_steps=schedule_total)
    
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    
    # Training loop
    cumulative_words = 0
    loss_values: List[float] = []
    saved_checkpoints: List[dict] = []
    next_ckpt = args.checkpoint_words
    masking_stats_accum = defaultdict(int)
    
    # Save initial embedding hash for verification
    with torch.no_grad():
        emb = model.deberta.embeddings.word_embeddings.weight
        emb_hash = hashlib.sha256(emb.cpu().numpy().tobytes()).hexdigest()
    
    log_path = out / "training_log.jsonl"
    model.train()
    
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            priority_labels = batch["priority"].to(device, non_blocking=True)
            is_pair = batch["is_pair_row"].to(device, non_blocking=True)
            
            # Apply cross-view masking
            masked_inputs, labels, batch_stats = apply_crossview_masking(
                input_ids, attention_mask, word_group, priority_labels,
                is_pair, args.arm, tokenizer, gen
            )
            
            # Accumulate masking stats
            for k, v in batch_stats.items():
                masking_stats_accum[k] += v
            
            # Forward + backward
            optim.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            loss = out_model.loss
            if loss is None:
                raise RuntimeError("Model returned no loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            
            cumulative_words += words
            loss_float = float(loss.detach().cpu())
            loss_values.append(loss_float)
            
            rec = {"step": step, "loss": loss_float, "lr": float(sched.get_last_lr()[0]),
                   "batch_words": words, "cumulative_word_exposure": cumulative_words,
                   "masked_tokens": batch_stats["total_masked"],
                   "elapsed_sec": round(time.time() - start_time, 1)}
            logf.write(json.dumps(rec) + "\n")
            
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            
            # Checkpoint
            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.selected_words:
                name = f"chck_{next_ckpt // 1_000_000}M" if next_ckpt >= 1_000_000 else "chck_1M"
                cp = out / "hf_model" / name
                save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({"name": name, "cum_words": cumulative_words, "path": str(cp)})
                print(json.dumps({"event": "checkpoint", "name": name, "cum_words": cumulative_words}), flush=True)
                next_ckpt += args.checkpoint_words
    
    # Final save
    save_hf_checkpoint(model, tokenizer, out / "hf_model")
    
    # Metrics
    metrics = {
        "status": f"CROSSVIEW_{args.arm.upper()}_COMPLETE",
        "arm": args.arm,
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "word_exposure": cumulative_words,
        "actual_training_steps": total_steps,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "learning_rate": args.learning_rate,
        "word_embedding_sha256": emb_hash,
        "saved_checkpoints": saved_checkpoints,
        "structure_stats": structure_stats,
        "masking_stats": dict(masking_stats_accum),
        "load_meta": load_meta,
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(json.dumps({"event": "done", "arm": args.arm, "params": param_count,
                      "loss_first": metrics["loss_first"], "loss_last": metrics["loss_last"],
                      "word_exposure": cumulative_words,
                      "masking_stats": dict(masking_stats_accum)}), flush=True)


if __name__ == "__main__":
    main()
