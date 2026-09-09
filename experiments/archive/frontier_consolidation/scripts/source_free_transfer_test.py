#!/usr/bin/env python3
"""research: Source-free transfer test for the directed edit-state signal.

Scientific purpose
------------------
research showed a detached private readout extracts true-vs-decoy advantage from
source-conditioned research hidden states. But official evaluation never supplies the
paired source. This script tests whether the information *transfers* to source-free
inference under document-disjoint splits:

  For each condition (true-aligned, shuffled-aligned, source-free):
    1. Train a regularized private readout on TRAIN-doc source-CONDITIONED hidden states
    2. Evaluate the SAME readout on TEST-doc source-FREE hidden states

The decisive comparison:
  - true-aligned source-free test NLL < shuffled-aligned source-free test NLL
    → transformation structure transfers, route has merit
  - true-aligned ≈ shuffled
    → no transfer, route has not solved complementary acquisition

Also runs a "shared readout" variant that trains on both conditioned and free hidden
states jointly, and a pure source-free baseline.

No pretraining, no official evaluation, no model parameter updates.
"""
from __future__ import annotations

import argparse
import collections
import csv
import difflib
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STUDY = Path("experiments/archive/frontier_consolidation")
WORKSPACE = STUDY
DEFAULT_CACHE = WORKSPACE / "data/hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE / "hf_home"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_CACHE / "transformers"))
os.environ.setdefault("HF_MODULES_CACHE", str(DEFAULT_CACHE / "modules"))
for _p in [os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"], os.environ["HF_MODULES_CACHE"]]:
    Path(_p).mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"

POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
PAIR_JSONL = WORKSPACE / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
ROW_META_CANDIDATES = [
    WORKSPACE / "data/density_core_reinvestment_medium_riskhard/fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl",
    WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl",
]
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
CKPT = WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
FUNCTION_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every", "no", "not",
    "and", "or", "but", "if", "then", "because", "so", "while", "when", "before", "after", "although",
    "as", "than", "of", "in", "on", "at", "to", "for", "from", "by", "with", "without", "about", "into",
    "over", "under", "between", "through", "during", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "have", "has", "had", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "it", "there", "here", "who", "what", "where", "why", "how", "which", "whose",
}
PRONOUNS = {"i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
            "it", "its", "we", "us", "our", "ours", "they", "them", "their", "theirs", "myself",
            "yourself", "himself", "herself", "itself", "ourselves", "themselves", "who", "whom",
            "whose", "which", "that"}
CONNECTIVES = {"and", "or", "but", "because", "so", "if", "then", "while", "when", "before", "after",
               "although", "though", "since", "until"}


# ── data structures ──

@dataclass
class TokSpan:
    text: str; norm: str; start: int; end: int; is_word: bool

@dataclass
class PairRec:
    pair_id: str; row_index: int; example_id: int
    source_text: str; rewrite_text: str
    source_words: int; rewrite_words: int
    doc_id: str; content_overlap: float

@dataclass
class TransferItem:
    item_id: str
    target_id: int
    target_kind: str
    pair_id: str
    doc_id: str
    split: str  # "train" or "test"
    # token-id sequences for each context condition
    ids_free: list[int]      # [CLS] rewrite_with_mask [SEP]
    ids_true: list[int]      # [CLS] true_source [SEP] rewrite_with_mask [SEP]
    ids_shuffled: list[int]  # [CLS] wrong_source [SEP] rewrite_with_mask [SEP]
    # metadata
    source_absent_target: bool  # target token piece absent from true source


# ── text utilities ──

def words_count(t: str) -> int:
    return len(WORD_RE.findall(t))

def token_spans(text: str) -> list[TokSpan]:
    out = []
    for m in WORD_RE.finditer(text):
        t = m.group()
        out.append(TokSpan(t, t.lower(), m.start(), m.end(), bool(re.search(r"[A-Za-z0-9]", t))))
    return out

def target_kind(tok: str) -> str:
    t = tok.lower()
    if t in PRONOUNS: return "pronoun"
    if t in CONNECTIVES: return "connective"
    if t in FUNCTION_WORDS: return "function"
    if any(ch.isalpha() for ch in t): return "content"
    if any(ch.isdigit() for ch in t): return "number"
    return "punct_or_other"


# ── pair loading (reused from research) ──

def norm_pair_id(pid: str) -> str:
    return str(pid).split(":", 1)[-1] if str(pid).startswith("compact:") else str(pid)

def find_row_meta_path() -> Path:
    for p in ROW_META_CANDIDATES:
        if p.exists(): return p
    raise FileNotFoundError("No changed-block row metadata found")

def load_pair_rows() -> dict[str, tuple[int, int]]:
    path = find_row_meta_path()
    out: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            row_index = int(obj.get("row_index", obj.get("row_index_in_pool_0based", 0)))
            example_id = int(obj.get("example_id", -1))
            for pid in obj.get("pair_ids") or []:
                out[norm_pair_id(pid)] = (row_index, example_id)
    return out

def load_pairs() -> list[PairRec]:
    row_of = load_pair_rows()
    recs = []
    with PAIR_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            pid = norm_pair_id(obj.get("pair_id"))
            if pid not in row_of: continue
            src = str(obj.get("source_text") or "")
            rew = str(obj.get("rewrite_text") or "")
            if not src or not rew: continue
            row_index, example_id = row_of[pid]
            recs.append(PairRec(
                pair_id=pid, row_index=row_index, example_id=example_id,
                source_text=src, rewrite_text=rew,
                source_words=int(obj.get("source_words", words_count(src))),
                rewrite_words=int(obj.get("rewrite_words", words_count(rew))),
                doc_id=str(obj.get("doc_id", "")),
                content_overlap=float(obj.get("content_overlap", 0.0)),
            ))
    recs.sort(key=lambda r: (r.row_index, r.pair_id))
    return recs

def pair_edit_changed_spans(pair: PairRec) -> list[TokSpan]:
    s = [t.norm for t in token_spans(pair.source_text)]
    r_spans = token_spans(pair.rewrite_text)
    r = [t.norm for t in r_spans]
    sm = difflib.SequenceMatcher(a=s, b=r, autojunk=False)
    changed = []
    equal_anchor_ge3 = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" and (i2 - i1) >= 3: equal_anchor_ge3 += 1
        if tag in {"insert", "replace"}: changed.extend(r_spans[j1:j2])
    if len([t for t in changed if t.is_word]) < 3 or equal_anchor_ge3 < 1:
        return []
    return changed


# ── item construction ──

def build_transfer_items(tokenizer, pairs: list[PairRec], train_docs: set[str],
                         test_docs: set[str], args) -> list[TransferItem]:
    """Build TransferItem objects with three context types per item,
    doc-disjoint train/test assignment, and shuffled-source selection."""
    rng = random.Random(args.seed)
    mask_id = int(tokenizer.mask_token_id)
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    if cls_id is None or sep_id is None:
        raise RuntimeError(f"Cannot determine CLS/SEP ids: cls={cls_id}, sep={sep_id}")

    # Group pairs by doc for shuffled source selection
    doc_pairs: dict[str, list[PairRec]] = collections.defaultdict(list)
    for p in pairs:
        doc_pairs[p.doc_id].append(p)
    all_pairs_by_len = sorted(pairs, key=lambda p: p.source_words)

    items: list[TransferItem] = []
    skip = collections.Counter()

    shuffled_pairs = list(pairs)
    rng.shuffle(shuffled_pairs)

    for p in shuffled_pairs:
        if len(items) >= args.max_items:
            break
        # Determine split
        if p.doc_id in train_docs:
            split = "train"
        elif p.doc_id in test_docs:
            split = "test"
        else:
            skip["no_split"] += 1
            continue

        changed_spans = pair_edit_changed_spans(p)
        if not changed_spans:
            skip["no_changed_span"] += 1
            continue

        changed_char = set()
        for s in changed_spans:
            for c in range(s.start, s.end):
                changed_char.add(c)

        # Find single-piece targets in changed spans
        rew_enc = tokenizer(p.rewrite_text, add_special_tokens=False, return_offsets_mapping=True)
        rew_ids = list(rew_enc["input_ids"])
        rew_offs = rew_enc["offset_mapping"]

        candidates = []
        for i, (a, b) in enumerate(rew_offs):
            if b <= a: continue
            if a not in changed_char: continue
            piece_text = p.rewrite_text[a:b]
            if not any(ch.isalnum() for ch in piece_text): continue
            # Find the word span containing this piece
            word_spans = [sp for sp in token_spans(p.rewrite_text) if sp.start <= a and sp.end >= b and sp.is_word]
            if not word_spans: continue
            # Drop edge words
            all_word_spans = [sp for sp in token_spans(p.rewrite_text) if sp.is_word]
            if len(all_word_spans) >= 3 and word_spans[0] in (all_word_spans[0], all_word_spans[-1]):
                continue
            candidates.append((a, b, int(rew_ids[i]), target_kind(word_spans[0].text)))

        if not candidates:
            skip["no_candidate_target"] += 1
            continue

        # Pick one target per pair
        rng.shuffle(candidates)
        char_start, char_end, tgt_id, tgt_kind = candidates[0]

        # Check source-token-absent condition
        src_enc = tokenizer(p.source_text, add_special_tokens=False)
        src_ids_set = set(int(x) for x in src_enc["input_ids"])
        source_absent = tgt_id not in src_ids_set

        if args.require_source_absent and not source_absent:
            skip["target_in_source"] += 1
            continue

        # Build token-id sequences
        # Base/free: [CLS] rewrite_with_mask [SEP]
        masked_rew = list(rew_ids)
        tgt_pos_in_rew = [j for j, (a2, b2) in enumerate(rew_offs) if a2 == char_start and b2 == char_end]
        if len(tgt_pos_in_rew) != 1:
            skip["target_pos_ambiguous"] += 1
            continue
        masked_rew[tgt_pos_in_rew[0]] = mask_id

        ids_free = [cls_id] + masked_rew + [sep_id]
        if len(ids_free) > args.max_length:
            skip["free_too_long"] += 1
            continue

        # True-conditioned: [CLS] source [SEP] rewrite_with_mask [SEP]
        src_ids = list(src_enc["input_ids"])
        ids_true = [cls_id] + src_ids + [sep_id] + masked_rew + [sep_id]
        if len(ids_true) > args.max_length:
            skip["true_too_long"] += 1
            continue

        # Shuffled-conditioned: use source from a different-doc pair with similar length
        shuffled_source = None
        for attempt in range(50):
            idx = (hash(p.pair_id) + attempt * 9973) % len(all_pairs_by_len)
            cand = all_pairs_by_len[idx]
            if cand.doc_id != p.doc_id and abs(cand.source_words - p.source_words) <= 15:
                shuffled_source = cand.source_text
                break
        if shuffled_source is None:
            # Fall back to any different-doc pair
            for attempt in range(50):
                idx = (hash(p.pair_id) + 5000 + attempt * 7919) % len(pairs)
                if pairs[idx].doc_id != p.doc_id:
                    shuffled_source = pairs[idx].source_text
                    break
        if shuffled_source is None:
            skip["no_shuffled_source"] += 1
            continue

        shuf_ids = tokenizer(shuffled_source, add_special_tokens=False)["input_ids"]
        ids_shuffled = [cls_id] + list(shuf_ids) + [sep_id] + masked_rew + [sep_id]
        if len(ids_shuffled) > args.max_length:
            skip["shuffled_too_long"] += 1
            continue

        # Validate all have exactly one mask
        for label, seq in [("free", ids_free), ("true", ids_true), ("shuffled", ids_shuffled)]:
            if seq.count(mask_id) != 1:
                skip[f"mask_count_{label}"] += 1
                continue

        items.append(TransferItem(
            item_id=f"transfer:{p.pair_id}:{char_start}-{char_end}",
            target_id=tgt_id,
            target_kind=tgt_kind,
            pair_id=p.pair_id,
            doc_id=p.doc_id,
            split=split,
            ids_free=ids_free,
            ids_true=ids_true,
            ids_shuffled=ids_shuffled,
            source_absent_target=source_absent,
        ))

    return items, skip


# ── model and forward pass ──

def load_model_tokenizer(device: str):
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    if tok.mask_token is None or tok.mask_token_id is None:
        raise RuntimeError("Tokenizer has no mask token")
    model = AutoModelForMaskedLM.from_pretrained(str(CKPT))
    model.eval()
    model.to(device)
    return model, tok


def pad_id_batch(id_lists: list[list[int]], pad_id: int):
    max_len = max(len(x) for x in id_lists)
    input_ids = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((len(id_lists), max_len), dtype=torch.long)
    for i, ids in enumerate(id_lists):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[i, :len(ids)] = 1
    return input_ids, attention


def extract_hidden_and_logits(model, tokenizer, items: list[TransferItem],
                               context_key: str, batch_size: int, device: str):
    """Forward pass for one context type. Returns mask-position hidden states and logits."""
    mask_id = int(tokenizer.mask_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    id_attr = f"ids_{context_key}"
    seqs = [getattr(it, id_attr) for it in items]
    targets = torch.tensor([it.target_id for it in items], dtype=torch.long)

    all_hidden, all_logits, all_nll = [], [], []
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            end = min(len(seqs), start + batch_size)
            input_ids, attn = pad_id_batch(seqs[start:end], pad_id)
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            mask_pos = (input_ids == mask_id).nonzero(as_tuple=False)
            if mask_pos.shape[0] != (end - start):
                raise RuntimeError(f"Mask count mismatch for {context_key}")
            out = model(input_ids=input_ids, attention_mask=attn,
                       output_hidden_states=True, return_dict=True)
            logits = out.logits[torch.arange(end-start, device=device), mask_pos[:, 1]].float()
            hidden = out.hidden_states[-1][torch.arange(end-start, device=device), mask_pos[:, 1]].float()
            tgt = targets[start:end].to(device)
            nll = F.cross_entropy(logits, tgt, reduction="none")
            all_hidden.append(hidden.cpu())
            all_logits.append(logits.cpu())
            all_nll.append(nll.cpu())

    return {
        "hidden": torch.cat(all_hidden, dim=0),
        "logits": torch.cat(all_logits, dim=0),
        "nll": torch.cat(all_nll, dim=0),
    }


# ── probe training and evaluation ──

class LinearProbe(torch.nn.Module):
    """Linear readout with optional bottleneck and dropout."""
    def __init__(self, in_dim: int, bottleneck: int, vocab: int, dropout: float = 0.1):
        super().__init__()
        self.down = torch.nn.Linear(in_dim, bottleneck)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)
        self.up = torch.nn.Linear(bottleneck, vocab)
        # Zero-init output: initial function = base logits only
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.drop(self.act(self.down(x))))


def train_and_eval_probe(label: str,
                          train_hidden: torch.Tensor, train_base_logits: torch.Tensor,
                          train_targets: torch.Tensor,
                          eval_hidden: torch.Tensor, eval_base_logits: torch.Tensor,
                          eval_targets: torch.Tensor,
                          args, device: str, seed_offset: int = 0):
    """Train probe on given hidden states, evaluate on given eval hidden states."""
    # Standardize using train stats
    mu = train_hidden.mean(dim=0, keepdim=True)
    sd = train_hidden.std(dim=0, keepdim=True).clamp_min(1e-4)
    H_train = ((train_hidden - mu) / sd).float()
    H_eval = ((eval_hidden - mu) / sd).float()

    vocab = int(train_base_logits.shape[1])
    probe = LinearProbe(H_train.shape[1], args.probe_dim, vocab, dropout=args.probe_dropout).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_wd)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 200 + seed_offset)

    n_train = H_train.shape[0]
    train_idx = list(range(n_train))
    history = []

    for epoch in range(args.probe_epochs):
        perm = [train_idx[i] for i in torch.randperm(n_train, generator=gen).tolist()]
        probe.train()
        total_loss, nb = 0.0, 0
        for start in range(0, n_train, args.probe_batch):
            ids = perm[start:start + args.probe_batch]
            h = H_train[ids].to(device)
            off = train_base_logits[ids].to(device)
            tgt = train_targets[ids].to(device)
            logits = off + probe(h)
            loss = F.cross_entropy(logits, tgt)
            if args.delta_l2 > 0:
                delta = logits - off
                loss = loss + args.delta_l2 * delta.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), args.grad_clip)
            opt.step()
            total_loss += float(loss.item())
            nb += 1
        if epoch in {0, args.probe_epochs // 2, args.probe_epochs - 1}:
            probe.eval()
            with torch.no_grad():
                e_logits = eval_base_logits.to(device) + probe(H_eval.to(device))
                e_nll = F.cross_entropy(e_logits, eval_targets.to(device), reduction="none")
            history.append({"epoch": epoch+1,
                           "train_loss": total_loss/max(1,nb),
                           "eval_nll_mean": float(e_nll.mean().item())})

    # Final evaluation
    probe.eval()
    with torch.no_grad():
        e_logits = eval_base_logits.to(device) + probe(H_eval.to(device))
        e_nll = F.cross_entropy(e_logits, eval_targets.to(device), reduction="none").cpu()
        t_logits = train_base_logits.to(device) + probe(H_train.to(device))
        t_nll = F.cross_entropy(t_logits, train_targets.to(device), reduction="none").cpu()

    return {
        "label": label,
        "n_train": n_train,
        "n_eval": H_eval.shape[0],
        "train_nll_mean": float(t_nll.mean().item()),
        "eval_nll_mean": float(e_nll.mean().item()),
        "eval_nll_items": [float(x) for x in e_nll.tolist()],
        "history": history,
    }


def train_shared_probe(label: str,
                        train_h_cond: torch.Tensor, train_h_free: torch.Tensor,
                        train_base_logits: torch.Tensor, train_targets: torch.Tensor,
                        eval_h_free: torch.Tensor, eval_base_logits: torch.Tensor,
                        eval_targets: torch.Tensor,
                        args, device: str, alpha: float = 0.5):
    """Shared readout: train on both conditioned and free hidden states, eval on free only."""
    mu = train_h_free.mean(dim=0, keepdim=True)
    sd = train_h_free.std(dim=0, keepdim=True).clamp_min(1e-4)
    H_cond = ((train_h_cond - mu) / sd).float()
    H_free_tr = ((train_h_free - mu) / sd).float()
    H_free_ev = ((eval_h_free - mu) / sd).float()

    vocab = int(train_base_logits.shape[1])
    probe = LinearProbe(H_cond.shape[1], args.probe_dim, vocab, dropout=args.probe_dropout).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_wd)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 300)

    n_train = H_cond.shape[0]

    for epoch in range(args.probe_epochs):
        perm = torch.randperm(n_train, generator=gen).tolist()
        probe.train()
        for start in range(0, n_train, args.probe_batch):
            ids = perm[start:start + args.probe_batch]
            h_c = H_cond[ids].to(device)
            h_f = H_free_tr[ids].to(device)
            off = train_base_logits[ids].to(device)
            tgt = train_targets[ids].to(device)
            logits_c = off + probe(h_c)
            logits_f = off + probe(h_f)
            loss = alpha * F.cross_entropy(logits_c, tgt) + (1 - alpha) * F.cross_entropy(logits_f, tgt)
            if args.delta_l2 > 0:
                loss = loss + args.delta_l2 * ((logits_c - off).square().mean() + (logits_f - off).square().mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), args.grad_clip)
            opt.step()

    probe.eval()
    with torch.no_grad():
        e_logits = eval_base_logits.to(device) + probe(H_free_ev.to(device))
        e_nll = F.cross_entropy(e_logits, eval_targets.to(device), reduction="none").cpu()
    return {
        "label": label,
        "n_train": n_train,
        "n_eval": H_free_ev.shape[0],
        "eval_nll_mean": float(e_nll.mean().item()),
        "eval_nll_items": [float(x) for x in e_nll.tolist()],
    }


# ── statistics ──

def bootstrap_ci(vals: list[float], n_boot: int = 2000, seed: int = 42) -> dict:
    rng = random.Random(seed)
    n = len(vals)
    if n == 0:
        return {"mean": None, "ci05": None, "ci95": None, "n": 0}
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randint(0, n-1)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    return {
        "mean": statistics.mean(vals),
        "ci05": means[int(0.05 * n_boot)],
        "ci95": means[int(0.95 * n_boot)],
        "n": n,
    }


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap CI for mean(a) - mean(b), paired."""
    rng = random.Random(seed)
    n = len(a)
    assert n == len(b)
    diffs = [a[i] - b[i] for i in range(n)]
    return bootstrap_ci(diffs, n_boot, seed)


# ── main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--max-items", type=int, default=8000)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=8123)
    ap.add_argument("--doc-test-frac", type=float, default=0.30)
    ap.add_argument("--require-source-absent", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--forward-batch", type=int, default=128)
    ap.add_argument("--probe-dim", type=int, default=32)
    ap.add_argument("--probe-epochs", type=int, default=30)
    ap.add_argument("--probe-batch", type=int, default=256)
    ap.add_argument("--probe-lr", type=float, default=0.001)
    ap.add_argument("--probe-wd", type=float, default=0.001)
    ap.add_argument("--probe-dropout", type=float, default=0.15)
    ap.add_argument("--delta-l2", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=2.0)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--n-probe-seeds", type=int, default=3)
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(args.device)
    print(json.dumps({"event": "model_loaded", "device": args.device}), flush=True)

    # Load pairs and split documents
    pairs = load_pairs()
    doc_ids = sorted(set(p.doc_id for p in pairs))
    rng = random.Random(args.seed)
    rng.shuffle(doc_ids)
    n_test = max(1, int(len(doc_ids) * args.doc_test_frac))
    test_docs = set(doc_ids[:n_test])
    train_docs = set(doc_ids[n_test:])
    print(json.dumps({"event": "doc_split", "total_docs": len(doc_ids),
                       "train_docs": len(train_docs), "test_docs": len(test_docs)}), flush=True)

    # Build items
    items, skip = build_transfer_items(tokenizer, pairs, train_docs, test_docs, args)
    train_items = [it for it in items if it.split == "train"]
    test_items = [it for it in items if it.split == "test"]
    print(json.dumps({"event": "items_built", "total": len(items),
                       "train": len(train_items), "test": len(test_items),
                       "source_absent": sum(1 for it in items if it.source_absent_target),
                       "skip": dict(skip)}), flush=True)

    if len(train_items) < 20 or len(test_items) < 10:
        raise RuntimeError(f"Too few items: train={len(train_items)}, test={len(test_items)}")

    # Forward passes for all three context types
    all_items = train_items + test_items
    train_slice = slice(0, len(train_items))
    test_slice = slice(len(train_items), len(all_items))

    forward_results = {}
    for ctx in ["free", "true", "shuffled"]:
        print(json.dumps({"event": "forward", "context": ctx, "n_items": len(all_items), "device": args.device}), flush=True)
        forward_results[ctx] = extract_hidden_and_logits(model, tokenizer, all_items, ctx, args.forward_batch, args.device)
        print(json.dumps({"event": "forward_done", "context": ctx,
                           "nll_mean": float(forward_results[ctx]["nll"].mean().item())}), flush=True)

    # Free GPU memory from model
    del model
    torch.cuda.empty_cache()

    targets = torch.tensor([it.target_id for it in all_items], dtype=torch.long)
    train_targets = targets[train_slice]
    test_targets = targets[test_slice]

    # Base logits for probes: source-free logits (this is the eval regime)
    base_logits_all = forward_results["free"]["logits"]
    train_base = base_logits_all[train_slice]
    test_base = base_logits_all[test_slice]

    # Record raw frozen NLL comparison (diagnostic, not the transfer metric)
    raw_summary = {}
    for ctx in ["free", "true", "shuffled"]:
        nlls = forward_results[ctx]["nll"]
        raw_summary[ctx] = {
            "train_nll_mean": float(nlls[train_slice].mean().item()),
            "test_nll_mean": float(nlls[test_slice].mean().item()),
        }
    raw_summary["true_vs_shuffled_test"] = float(
        forward_results["true"]["nll"][test_slice].mean().item() -
        forward_results["shuffled"]["nll"][test_slice].mean().item()
    )

    # ── Run probe training under each condition, with multiple seeds ──
    probe_device = args.device
    results = {}

    for probe_seed in range(args.n_probe_seeds):
        seed_off = probe_seed * 1000

        # Condition 1: Train on TRUE-conditioned, eval on FREE
        r = train_and_eval_probe(
            f"true_conditioned_seed{probe_seed}",
            forward_results["true"]["hidden"][train_slice],
            train_base, train_targets,
            forward_results["free"]["hidden"][test_slice],
            test_base, test_targets,
            args, probe_device, seed_offset=seed_off)
        results[f"true_cond_s{probe_seed}"] = r

        # Condition 2: Train on SHUFFLED-conditioned, eval on FREE
        r = train_and_eval_probe(
            f"shuffled_conditioned_seed{probe_seed}",
            forward_results["shuffled"]["hidden"][train_slice],
            train_base, train_targets,
            forward_results["free"]["hidden"][test_slice],
            test_base, test_targets,
            args, probe_device, seed_offset=seed_off + 100)
        results[f"shuffled_cond_s{probe_seed}"] = r

        # Condition 3: Train on FREE, eval on FREE (baseline)
        r = train_and_eval_probe(
            f"source_free_seed{probe_seed}",
            forward_results["free"]["hidden"][train_slice],
            train_base, train_targets,
            forward_results["free"]["hidden"][test_slice],
            test_base, test_targets,
            args, probe_device, seed_offset=seed_off + 200)
        results[f"free_s{probe_seed}"] = r

        # Condition 4: Shared readout (true + free joint), eval on FREE
        r = train_shared_probe(
            f"shared_true_seed{probe_seed}",
            forward_results["true"]["hidden"][train_slice],
            forward_results["free"]["hidden"][train_slice],
            train_base, train_targets,
            forward_results["free"]["hidden"][test_slice],
            test_base, test_targets,
            args, probe_device, alpha=0.5)
        results[f"shared_true_s{probe_seed}"] = r

        # Condition 5: Shared readout (shuffled + free joint), eval on FREE
        r = train_shared_probe(
            f"shared_shuffled_seed{probe_seed}",
            forward_results["shuffled"]["hidden"][train_slice],
            forward_results["free"]["hidden"][train_slice],
            train_base, train_targets,
            forward_results["free"]["hidden"][test_slice],
            test_base, test_targets,
            args, probe_device, alpha=0.5)
        results[f"shared_shuffled_s{probe_seed}"] = r

    # ── Aggregate across seeds ──
    def agg_condition(prefix: str):
        keys = [k for k in results if k.startswith(prefix)]
        all_eval_nlls = []
        for k in keys:
            all_eval_nlls.extend(results[k]["eval_nll_items"])
        per_seed_means = [results[k]["eval_nll_mean"] for k in keys]
        return {
            "n_seeds": len(keys),
            "per_seed_eval_nll": per_seed_means,
            "mean_eval_nll": statistics.mean(per_seed_means) if per_seed_means else None,
            "std_eval_nll": statistics.stdev(per_seed_means) if len(per_seed_means) > 1 else 0.0,
            "pooled_bootstrap": bootstrap_ci(all_eval_nlls),
        }

    agg = {
        "true_conditioned": agg_condition("true_cond_"),
        "shuffled_conditioned": agg_condition("shuffled_cond_"),
        "source_free": agg_condition("free_"),
        "shared_true": agg_condition("shared_true_"),
        "shared_shuffled": agg_condition("shared_shuffled_"),
    }

    # Decisive comparison: true vs shuffled source-free eval NLL
    true_nlls = []
    shuf_nlls = []
    for s in range(args.n_probe_seeds):
        true_nlls.append(results[f"true_cond_s{s}"]["eval_nll_mean"])
        shuf_nlls.append(results[f"shuffled_cond_s{s}"]["eval_nll_mean"])

    decisive = {
        "true_mean": statistics.mean(true_nlls),
        "shuffled_mean": statistics.mean(shuf_nlls),
        "delta_true_minus_shuffled": statistics.mean(true_nlls) - statistics.mean(shuf_nlls),
        "interpretation": "negative delta means true-aligned transfers better (lower NLL)",
    }

    # Also: paired bootstrap on items from first seed
    if "true_cond_s0" in results and "shuffled_cond_s0" in results:
        decisive["paired_bootstrap_s0"] = paired_bootstrap(
            results["true_cond_s0"]["eval_nll_items"],
            results["shuffled_cond_s0"]["eval_nll_items"],
        )

    # Reference: research baseline (no probe, just frozen logits)
    free_test_nll = float(forward_results["free"]["nll"][test_slice].mean().item())

    output = {
        "status": "SOURCE_FREE_TRANSFER_TEST",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Test whether source-conditioned training transfers to source-free evaluation under document-disjoint splits",
        "args": vars(args),
        "doc_split": {"total_docs": len(doc_ids), "train_docs": len(train_docs), "test_docs": len(test_docs)},
        "items": {"total": len(items), "train": len(train_items), "test": len(test_items),
                  "source_absent": sum(1 for it in items if it.source_absent_target),
                  "skip": dict(skip)},
        "raw_frozen_nll": raw_summary,
        "baseline_test_nll": free_test_nll,
        "aggregated_conditions": agg,
        "decisive_comparison": decisive,
        "elapsed_sec": time.time() - t0,
    }

    # Remove per-item NLLs from conditions to keep JSON manageable
    for k in results:
        results[k].pop("eval_nll_items", None)
    output["per_seed_results"] = results

    out_json = out_dir / "source_free_transfer_test.json"
    out_json.write_text(json.dumps(output, indent=2))

    # Markdown summary
    lines = ["# research — Source-free transfer test\n"]
    lines.append(f"Items: {len(items)} total ({len(train_items)} train-doc, {len(test_items)} test-doc)")
    lines.append(f"Documents: {len(train_docs)} train, {len(test_docs)} test")
    lines.append(f"Source-absent targets: {sum(1 for it in items if it.source_absent_target)}")
    lines.append(f"Probe: bottleneck {args.probe_dim}, dropout {args.probe_dropout}, L2 {args.probe_wd}, {args.probe_epochs} epochs")
    lines.append(f"Probe seeds: {args.n_probe_seeds}\n")
    lines.append("## Raw frozen NLL (diagnostic)")
    for ctx in ["free", "true", "shuffled"]:
        lines.append(f"  {ctx}: train {raw_summary[ctx]['train_nll_mean']:.4f}, test {raw_summary[ctx]['test_nll_mean']:.4f}")
    lines.append(f"  true-vs-shuffled test: {raw_summary['true_vs_shuffled_test']:.4f}\n")
    lines.append(f"## research baseline test NLL (no probe): {free_test_nll:.4f}\n")
    lines.append("## Transfer results (train-conditioned → eval source-free)")
    lines.append("| Condition | Mean eval NLL | Std | vs research baseline |")
    lines.append("|-----------|--------------|-----|-------------------|")
    for cond in ["true_conditioned", "shuffled_conditioned", "source_free", "shared_true", "shared_shuffled"]:
        a = agg[cond]
        vs_base = a["mean_eval_nll"] - free_test_nll if a["mean_eval_nll"] else None
        lines.append(f"| {cond} | {a['mean_eval_nll']:.4f} | {a['std_eval_nll']:.4f} | {vs_base:+.4f} |")
    lines.append(f"\n## Decisive comparison: true vs shuffled (source-free eval)")
    lines.append(f"  True mean: {decisive['true_mean']:.4f}")
    lines.append(f"  Shuffled mean: {decisive['shuffled_mean']:.4f}")
    lines.append(f"  Delta (true - shuffled): {decisive['delta_true_minus_shuffled']:.4f}")
    if "paired_bootstrap_s0" in decisive:
        pb = decisive["paired_bootstrap_s0"]
        lines.append(f"  Paired bootstrap CI: [{pb['ci05']:.4f}, {pb['ci95']:.4f}]")
    lines.append(f"\n  Interpretation: {'TRANSFER EXISTS' if decisive['delta_true_minus_shuffled'] < -0.01 else 'NO CLEAR TRANSFER' if abs(decisive['delta_true_minus_shuffled']) < 0.01 else 'SHUFFLED BETTER (anti-transfer)'}")
    lines.append(f"\nJSON: `{out_json}`")

    out_md = out_dir / "source_free_transfer_test.md"
    out_md.write_text("\n".join(lines) + "\n")

    # Print summary
    print(json.dumps({
        "status": output["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "baseline_nll": free_test_nll,
        "decisive": decisive,
        "elapsed_sec": output["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
