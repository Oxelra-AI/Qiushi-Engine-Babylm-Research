#!/usr/bin/env python3
"""research: three-way discrimination and source-reassignment probe.

Tests whether the acquired model (e80 answer-only private adapters on repaired
relation-first pairs) truly performs entity-conditioned state retrieval or just
update-recipient/query equality gating with generic source-value preference.

Probes:
1. Three-way scoring: for each pair × context, score all three candidates
   (value_a, value_b, shared_new). In RETAIN contexts, entity-conditioned
   retrieval requires correct_source > wrong_source AND correct_source > new.
   Mere gating would show correct_source ≈ wrong_source >> new.

2. Source-reassignment: swap value_a↔value_b in the source context while
   keeping entities, update, and query frames fixed. If the model reads
   entity-value bindings from context, its RETAIN predictions should follow
   the reassignment.

Runs: (a) parent baseline three-way, (b) train e80 answer-only, (c) trained
model three-way, (d) trained model source-reassignment.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import copy
import gc
import json
import math
import pathlib
import random
import re
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/frontier_consolidation')
sys.path.insert(0, str(_public_path('experiments/archive/frontier_consolidation/scripts')))
sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))

import coherent86_continuation_trainer as trusted_loader  # noqa: E402

MODEL_PATH = _public_path('models/frontier')
V1_PAIRS = _public_path('experiments/archive/functional_learning/data/relation_first_packets/relation_first_pairs.jsonl')

REPLACEMENT_CITIES = [
    "Amsterdam", "Barcelona", "Dublin", "Milan", "Tokyo", "Sydney",
    "Vienna", "Copenhagen", "Stockholm", "Athens", "Lisbon", "Prague",
    "Budapest", "Helsinki", "Cairo", "Montreal", "Zurich", "Brisbane",
    "Edinburgh", "Florence", "Geneva", "Hamburg", "Kyoto", "Munich",
    "Osaka", "Salzburg", "Venice", "Brussels", "Oslo", "Marseille",
    "Ankara", "Lima", "Bogota", "Manila", "Jakarta", "Havana",
    "Nairobi", "Doha", "Riyadh", "Bangalore",
]

RELATION_TEMPLATES = {
    "birthplace": {
        "update": "However, recent records show that {ENTITY} was actually born in {NEW_VALUE}.",
        "use": "According to this information, the birthplace of {QUERY} is {STATE}.",
    },
    "death_place": {
        "update": "However, updated records confirm that {ENTITY} actually died in {NEW_VALUE}.",
        "use": "According to this information, the place where {QUERY} died is {STATE}.",
    },
    "located_in": {
        "update": "However, after a boundary change, {ENTITY} is now part of {NEW_VALUE}.",
        "use": "According to current records, {QUERY} is located in {STATE}.",
    },
    "founded_year": {
        "update": "However, newly discovered documents show that {ENTITY} was actually founded in {NEW_VALUE}.",
        "use": "According to the latest records, {QUERY} was founded in {STATE}.",
    },
}


def rel(path):
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def finite_mean(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# Trusted model loading (exact copy from research)
# ---------------------------------------------------------------------------

def load_trusted_model(device, private_scale=0.75):
    model, missing, unexpected = trusted_loader.load_model(MODEL_PATH, device, 128, private_scale)
    if unexpected:
        raise RuntimeError(f"Trusted loader unexpected keys: {unexpected[:10]}")
    bad_missing = [k for k in missing if "private_adapter" in k]
    if bad_missing:
        raise RuntimeError(f"Trusted loader missing private keys: {bad_missing[:10]}")
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(True)
    return model, {"missing": list(missing), "unexpected": list(unexpected)}


def executed_scales(model):
    return [float(layer.private_adapter.scale) for layer in model.deberta.encoder.layer]


def model_identity(model, load_info):
    named = list(model.named_parameters())
    private = [(n, p) for n, p in named if ".private_adapter." in n]
    return {
        "class": type(model).__name__,
        "module": type(model).__module__,
        "total_params": int(sum(p.numel() for _, p in named)),
        "private_params": int(sum(p.numel() for _, p in private)),
        "executed_private_scales": executed_scales(model),
    }


def freeze_to_private_optimizer(model, lr, weight_decay):
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    private_names = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
    if not private_names:
        raise RuntimeError("No private_adapter parameters found")
    for n, p in model.named_parameters():
        p.requires_grad_(n in private_names)
    private_up_ids = {id(p) for n, p in model.named_parameters() if ".private_adapter.up." in n}
    private_normal = [p for n, p in model.named_parameters() if n in private_names and id(p) not in private_up_ids]
    private_zero_wd = [p for n, p in model.named_parameters() if n in private_names and id(p) in private_up_ids]
    opt = torch.optim.AdamW([
        {"params": private_normal, "weight_decay": float(weight_decay)},
        {"params": private_zero_wd, "weight_decay": 0.0},
    ], lr=float(lr), betas=(0.9, 0.98), eps=1e-6)
    return opt, {
        "trainable_param_count": sum(1 for _, p in model.named_parameters() if p.requires_grad),
        "trainable_tensor_count": int(sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)),
        "nonprivate_trainable_tensors": sum(1 for n, p in model.named_parameters() if p.requires_grad and n not in private_names),
    }


# ---------------------------------------------------------------------------
# Repaired construction (from research)
# ---------------------------------------------------------------------------

def choose_shared_new(pair, rng):
    relation = pair["relation"]
    current = {pair["value_a"].lower(), pair["value_b"].lower()}
    source = str(pair.get("source_context", "")).lower()
    pool = REPLACEMENT_CITIES if relation != "founded_year" else [str(y) for y in range(1880, 2010, 5)]
    preferred = [pair.get("new_value_a"), pair.get("new_value_b")]
    candidates = []
    for v in preferred + pool:
        if v is None:
            continue
        vl = str(v).lower()
        if vl in current or vl in source:
            continue
        if v not in candidates:
            candidates.append(str(v))
    if not candidates:
        raise ValueError(f"No valid shared replacement for {pair['pair_id']}")
    return rng.choice(candidates[:min(12, len(candidates))])


def construct_repaired_pairs(v1_pairs, seed):
    rng = random.Random(seed)
    out = []
    for p in v1_pairs:
        tmpl = RELATION_TEMPLATES.get(p["relation"])
        if tmpl is None:
            continue
        q = copy.deepcopy(p)
        q["contract_version"] = "shared_new_recipient_only_final_span"
        q["shared_new_value"] = choose_shared_new(q, rng)
        q["new_value_a"] = q["shared_new_value"]
        q["new_value_b"] = q["shared_new_value"]
        q["update_a_sentence"] = tmpl["update"].format(ENTITY=q["entity_a"], NEW_VALUE=q["shared_new_value"])
        q["update_b_sentence"] = tmpl["update"].format(ENTITY=q["entity_b"], NEW_VALUE=q["shared_new_value"])
        q["use_frame_a"] = tmpl["use"].format(QUERY=q["entity_a"], STATE="{STATE}")
        q["use_frame_b"] = tmpl["use"].format(QUERY=q["entity_b"], STATE="{STATE}")
        q["context_update_a"] = q["source_context"] + " " + q["update_a_sentence"]
        q["context_update_b"] = q["source_context"] + " " + q["update_b_sentence"]
        q["context_neutral"] = q["source_context"]
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# Source-reassignment construction
# ---------------------------------------------------------------------------

def swap_values_in_source(pair):
    """Swap value_a↔value_b in the source context. Entities stay in position."""
    src = pair["source_context"]
    va, vb = pair["value_a"], pair["value_b"]
    # Use placeholder to avoid double-replacement
    placeholder = "\x00SWAP_PLACEHOLDER\x00"
    # First replace va→placeholder, then vb→va, then placeholder→vb
    swapped = src.replace(va, placeholder)
    swapped = swapped.replace(vb, va)
    swapped = swapped.replace(placeholder, vb)
    # Verify the swap happened
    n_va_orig = src.count(va)
    n_vb_orig = src.count(vb)
    n_va_swap = swapped.count(va)
    n_vb_swap = swapped.count(vb)
    valid = (n_va_swap == n_vb_orig and n_vb_swap == n_va_orig and
             n_va_orig > 0 and n_vb_orig > 0 and va != vb)
    return swapped, valid, {
        "orig_va_count": n_va_orig, "orig_vb_count": n_vb_orig,
        "swap_va_count": n_va_swap, "swap_vb_count": n_vb_swap,
    }


def construct_reassignment_contexts(pair):
    """Build contexts with swapped source values for the reassignment test."""
    tmpl = RELATION_TEMPLATES.get(pair["relation"])
    if tmpl is None:
        return None, False, "no_template"
    swapped_src, valid, swap_info = swap_values_in_source(pair)
    if not valid:
        return None, False, swap_info
    # After swap: entity_a's source value is now value_b, entity_b's is value_a
    ctx = {
        "pair_id": pair["pair_id"],
        "relation": pair["relation"],
        "entity_a": pair["entity_a"],
        "entity_b": pair["entity_b"],
        "original_value_a": pair["value_a"],
        "original_value_b": pair["value_b"],
        "reassigned_value_a": pair["value_b"],  # A now has B's value
        "reassigned_value_b": pair["value_a"],  # B now has A's value
        "shared_new_value": pair["shared_new_value"],
        "swapped_source": swapped_src,
        "swap_info": swap_info,
        # Update contexts use swapped source
        "context_update_a_swapped": swapped_src + " " + pair["update_a_sentence"],
        "context_update_b_swapped": swapped_src + " " + pair["update_b_sentence"],
        "use_frame_a": pair["use_frame_a"],
        "use_frame_b": pair["use_frame_b"],
    }
    return ctx, True, swap_info


# ---------------------------------------------------------------------------
# Three-way scorer
# ---------------------------------------------------------------------------

def locate_span_positions(offsets, start, end):
    pos = []
    for i, (s, e) in enumerate(offsets):
        s, e = int(s), int(e)
        if e <= s:
            continue
        if s < end and e > start:
            pos.append(i)
    return pos


def score_candidate_in_context(model, tokenizer, context_with_frame, candidate, device, seq_length):
    """Score a candidate in the {STATE} slot of context_with_frame."""
    if context_with_frame.count("{STATE}") != 1:
        raise ValueError(f"Frame must have exactly one {{STATE}}: {context_with_frame[:100]!r}")
    before, after = context_with_frame.split("{STATE}")
    full = before + candidate + after
    start = len(before)
    end = start + len(candidate)
    enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True,
                    return_tensors="pt", max_length=seq_length, truncation=True)
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
    positions = locate_span_positions(offsets, start, end)
    if not positions:
        raise ValueError(f"Candidate span unavailable: {candidate!r} start={start} end={end}")
    covered_start = min(offsets[p][0] for p in positions)
    covered_end = max(offsets[p][1] for p in positions)
    if covered_start > start or covered_end < end:
        raise ValueError(f"Candidate span partial: {candidate!r} span={start,end} covered={covered_start,covered_end}")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    masked_ids = input_ids.clone()
    target_ids = input_ids[0, positions].detach().clone()
    for p in positions:
        masked_ids[0, p] = int(tokenizer.mask_token_id)
    with torch.no_grad():
        logits = model(input_ids=masked_ids, attention_mask=attention_mask).logits[0, positions]
        lp = F.log_softmax(logits, dim=-1)
        token_logps = lp[torch.arange(len(positions), device=device), target_ids].cpu().tolist()
    return {
        "candidate": candidate,
        "n_tokens": len(positions),
        "sum_logp": float(sum(token_logps)),
        "mean_logp": float(sum(token_logps) / len(token_logps)),
    }


def threeway_score_pair(model, tokenizer, pair, device, seq_length):
    """Score value_a, value_b, shared_new in each of 4+2 contexts."""
    va, vb, vnew = pair["value_a"], pair["value_b"], pair["shared_new_value"]
    candidates = [va, vb, vnew]
    contexts = {
        "update_a_query_a": pair["context_update_a"] + " " + pair["use_frame_a"],
        "update_b_query_a": pair["context_update_b"] + " " + pair["use_frame_a"],
        "update_a_query_b": pair["context_update_a"] + " " + pair["use_frame_b"],
        "update_b_query_b": pair["context_update_b"] + " " + pair["use_frame_b"],
        "neutral_query_a":  pair["context_neutral"] + " " + pair["use_frame_a"],
        "neutral_query_b":  pair["context_neutral"] + " " + pair["use_frame_b"],
    }
    results = {}
    for ctx_name, ctx_frame in contexts.items():
        row_scores = {}
        for cand in candidates:
            cand_label = "value_a" if cand == va else ("value_b" if cand == vb else "shared_new")
            s = score_candidate_in_context(model, tokenizer, ctx_frame, cand, device, seq_length)
            row_scores[cand_label] = s
        results[ctx_name] = row_scores
    return results


def reassignment_score_pair(model, tokenizer, pair, reassignment_ctx, device, seq_length):
    """Score candidates in source-swapped contexts."""
    va_orig = pair["value_a"]  # originally A's value
    vb_orig = pair["value_b"]  # originally B's value
    vnew = pair["shared_new_value"]
    candidates = [va_orig, vb_orig, vnew]

    contexts = {
        "swap_update_b_query_a": reassignment_ctx["context_update_b_swapped"] + " " + reassignment_ctx["use_frame_a"],
        "swap_update_a_query_b": reassignment_ctx["context_update_a_swapped"] + " " + reassignment_ctx["use_frame_b"],
    }
    results = {}
    for ctx_name, ctx_frame in contexts.items():
        row_scores = {}
        for cand in candidates:
            cand_label = "orig_value_a" if cand == va_orig else ("orig_value_b" if cand == vb_orig else "shared_new")
            s = score_candidate_in_context(model, tokenizer, ctx_frame, cand, device, seq_length)
            row_scores[cand_label] = s
        results[ctx_name] = row_scores
    return results


# ---------------------------------------------------------------------------
# Training (exact copy of research answer-only)
# ---------------------------------------------------------------------------

class RowDataset(Dataset):
    def __init__(self, rows, tokenizer, seq_length):
        self.items = []
        for row in rows:
            if row["role"] == "NEUTRAL":
                continue
            frame = row["use_sentence_frame"]
            candidate = str(row["answer_text"])
            before, after = frame.split("{STATE}")
            full = before + candidate + after
            start, end = len(before), len(before) + len(candidate)
            enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True,
                            return_tensors="pt", max_length=seq_length, truncation=True,
                            padding="max_length")
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
            pos = locate_span_positions(offsets, start, end)
            if not pos:
                raise ValueError(f"training answer span unavailable: {row['pair_id']} {row['row_type']}")
            input_ids = enc["input_ids"].squeeze(0)
            labels = torch.full_like(input_ids, -100)
            masked = input_ids.clone()
            for p in pos:
                labels[p] = input_ids[p]
                masked[p] = int(tokenizer.mask_token_id)
            self.items.append({
                "input_ids": masked,
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": labels,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def build_scoring_rows(pairs):
    """Build the standard 6-row scoring format from repaired pairs."""
    rows = []
    for p in pairs:
        def row(row_type, role, orientation, updated_entity, query_entity,
                source_answer, answer, foil, context_prefix, final_frame):
            return {
                "pair_id": p["pair_id"], "split": p.get("split"), "relation": p["relation"],
                "contract_version": p["contract_version"], "row_type": row_type,
                "role": role, "query_orientation": orientation,
                "updated_entity": updated_entity, "query_entity": query_entity,
                "entity_a": p["entity_a"], "entity_b": p["entity_b"],
                "value_a": p["value_a"], "value_b": p["value_b"],
                "source_answer": source_answer, "shared_new_value": p["shared_new_value"],
                "answer_text": answer, "foil_text": foil,
                "context_prefix": context_prefix, "final_frame": final_frame,
                "use_sentence_frame": context_prefix + " " + final_frame,
                "answer_kind": "shared_new" if answer == p["shared_new_value"] else "source_value",
                "foil_kind": "shared_new" if foil == p["shared_new_value"] else "source_value",
            }
        rows.append(row("query_a_update_a", "UPDATE", "query_a", p["entity_a"], p["entity_a"],
                        p["value_a"], p["shared_new_value"], p["value_a"], p["context_update_a"], p["use_frame_a"]))
        rows.append(row("query_a_update_b", "RETAIN", "query_a", p["entity_b"], p["entity_a"],
                        p["value_a"], p["value_a"], p["shared_new_value"], p["context_update_b"], p["use_frame_a"]))
        rows.append(row("query_b_update_a", "RETAIN", "query_b", p["entity_a"], p["entity_b"],
                        p["value_b"], p["value_b"], p["shared_new_value"], p["context_update_a"], p["use_frame_b"]))
        rows.append(row("query_b_update_b", "UPDATE", "query_b", p["entity_b"], p["entity_b"],
                        p["value_b"], p["shared_new_value"], p["value_b"], p["context_update_b"], p["use_frame_b"]))
        rows.append(row("neutral_query_a", "NEUTRAL", "query_a", "none", p["entity_a"],
                        p["value_a"], p["value_a"], p["shared_new_value"], p["context_neutral"], p["use_frame_a"]))
        rows.append(row("neutral_query_b", "NEUTRAL", "query_b", "none", p["entity_b"],
                        p["value_b"], p["value_b"], p["shared_new_value"], p["context_neutral"], p["use_frame_b"]))
    return rows


def train_model(model, tokenizer, train_rows, epochs, batch_size, lr, wd, max_grad_norm, seed, device, seq_length):
    optimizer, trainable_info = freeze_to_private_optimizer(model, lr, wd)
    if trainable_info["nonprivate_trainable_tensors"] != 0:
        raise RuntimeError(f"Non-private tensors trainable: {trainable_info}")
    ds = RowDataset(train_rows, tokenizer, seq_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=generator)
    print(f"Training {epochs} epochs, {len(ds)} examples, private params={trainable_info['trainable_tensor_count']}", flush=True)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, n_tok = 0.0, 0
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            nt = int((batch["labels"] != -100).sum().item())
            loss_sum += float(out.loss.detach().cpu()) * nt
            n_tok += nt
        if epoch % 20 == 0 or epoch == epochs:
            print(f"  e{epoch:04d} loss={loss_sum / max(1,n_tok):.4f}", flush=True)
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s", flush=True)
    return trainable_info, elapsed


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_threeway(pair_results, pairs, label):
    """Analyze three-way scoring results for a set of pairs."""
    records = []
    for pair, tw in zip(pairs, pair_results):
        pid = pair["pair_id"]
        split = pair.get("split")
        va, vb, vnew = pair["value_a"], pair["value_b"], pair["shared_new_value"]

        # RETAIN query_a (B updated): correct = value_a, wrong = value_b
        retain_qa = tw["update_b_query_a"]
        # RETAIN query_b (A updated): correct = value_b, wrong = value_a
        retain_qb = tw["update_a_query_b"]
        # UPDATE query_a (A updated): correct = shared_new
        update_qa = tw["update_a_query_a"]
        # UPDATE query_b (B updated): correct = shared_new
        update_qb = tw["update_b_query_b"]

        # Cross-source discrimination in RETAIN
        retain_qa_correct = retain_qa["value_a"]["mean_logp"]
        retain_qa_wrong   = retain_qa["value_b"]["mean_logp"]
        retain_qa_new     = retain_qa["shared_new"]["mean_logp"]
        retain_qa_cross   = retain_qa_correct - retain_qa_wrong  # >0 = correct source preferred

        retain_qb_correct = retain_qb["value_b"]["mean_logp"]
        retain_qb_wrong   = retain_qb["value_a"]["mean_logp"]
        retain_qb_new     = retain_qb["shared_new"]["mean_logp"]
        retain_qb_cross   = retain_qb_correct - retain_qb_wrong

        # UPDATE: new should dominate
        update_qa_new  = update_qa["shared_new"]["mean_logp"]
        update_qa_srca = update_qa["value_a"]["mean_logp"]
        update_qb_new  = update_qb["shared_new"]["mean_logp"]
        update_qb_srcb = update_qb["value_b"]["mean_logp"]

        rec = {
            "pair_id": pid, "split": split,
            # RETAIN cross-source discrimination
            "retain_qa_cross_source_margin": retain_qa_cross,
            "retain_qb_cross_source_margin": retain_qb_cross,
            "retain_qa_correct_over_new": retain_qa_correct - retain_qa_new,
            "retain_qb_correct_over_new": retain_qb_correct - retain_qb_new,
            # Is this entity-specific retrieval or just gating?
            "retain_qa_entity_retrieval": (retain_qa_cross > 0),
            "retain_qb_entity_retrieval": (retain_qb_cross > 0),
            "retain_qa_full_retrieval": (retain_qa_cross > 0 and retain_qa_correct > retain_qa_new),
            "retain_qb_full_retrieval": (retain_qb_cross > 0 and retain_qb_correct > retain_qb_new),
            # UPDATE correctness
            "update_qa_correct": (update_qa_new > update_qa_srca),
            "update_qb_correct": (update_qb_new > update_qb_srcb),
            # Raw scores for inspection
            "retain_qa_scores": {k: v["mean_logp"] for k, v in retain_qa.items()},
            "retain_qb_scores": {k: v["mean_logp"] for k, v in retain_qb.items()},
            "update_qa_scores": {k: v["mean_logp"] for k, v in update_qa.items()},
            "update_qb_scores": {k: v["mean_logp"] for k, v in update_qb.items()},
        }
        records.append(rec)

    # Aggregate
    n = len(records)
    summary = {
        "label": label,
        "n_pairs": n,
        # Entity-specific retrieval: cross-source margin > 0
        "n_retain_qa_entity_retrieval": sum(1 for r in records if r["retain_qa_entity_retrieval"]),
        "n_retain_qb_entity_retrieval": sum(1 for r in records if r["retain_qb_entity_retrieval"]),
        "n_retain_both_entity_retrieval": sum(1 for r in records if r["retain_qa_entity_retrieval"] and r["retain_qb_entity_retrieval"]),
        # Full retrieval: cross-source > 0 AND correct > new
        "n_retain_qa_full_retrieval": sum(1 for r in records if r["retain_qa_full_retrieval"]),
        "n_retain_qb_full_retrieval": sum(1 for r in records if r["retain_qb_full_retrieval"]),
        "n_retain_both_full_retrieval": sum(1 for r in records if r["retain_qa_full_retrieval"] and r["retain_qb_full_retrieval"]),
        # Mean cross-source margins
        "mean_retain_qa_cross_source": finite_mean([r["retain_qa_cross_source_margin"] for r in records]),
        "mean_retain_qb_cross_source": finite_mean([r["retain_qb_cross_source_margin"] for r in records]),
        "mean_retain_cross_source": finite_mean([r["retain_qa_cross_source_margin"] for r in records] +
                                                 [r["retain_qb_cross_source_margin"] for r in records]),
        # Mean correct-over-new margins in RETAIN
        "mean_retain_correct_over_new": finite_mean([r["retain_qa_correct_over_new"] for r in records] +
                                                     [r["retain_qb_correct_over_new"] for r in records]),
        # UPDATE success
        "n_update_qa_correct": sum(1 for r in records if r["update_qa_correct"]),
        "n_update_qb_correct": sum(1 for r in records if r["update_qb_correct"]),
    }
    return summary, records


def analyze_reassignment(pair_results, pairs, original_tw, label):
    """Analyze source-reassignment results."""
    records = []
    for pair, reass, orig_tw in zip(pairs, pair_results, original_tw):
        pid = pair["pair_id"]
        split = pair.get("split")
        va_orig = pair["value_a"]
        vb_orig = pair["value_b"]

        # After swap: A's source value is now vb_orig, B's source value is now va_orig
        # RETAIN query_a in swapped context (B updated): correct should now be vb_orig
        swap_retain_qa = reass["swap_update_b_query_a"]
        s_qa_orig_va = swap_retain_qa["orig_value_a"]["mean_logp"]  # was A's value
        s_qa_orig_vb = swap_retain_qa["orig_value_b"]["mean_logp"]  # was B's value, now A's
        s_qa_new     = swap_retain_qa["shared_new"]["mean_logp"]

        # After swap, correct answer for query A is orig_value_b (now assigned to A)
        swap_qa_follows_reassignment = (s_qa_orig_vb > s_qa_orig_va)
        swap_qa_correct_source = (s_qa_orig_vb > s_qa_new)

        # RETAIN query_b in swapped context (A updated): correct should now be va_orig
        swap_retain_qb = reass["swap_update_a_query_b"]
        s_qb_orig_va = swap_retain_qb["orig_value_a"]["mean_logp"]  # now B's value
        s_qb_orig_vb = swap_retain_qb["orig_value_b"]["mean_logp"]  # was B's value
        s_qb_new     = swap_retain_qb["shared_new"]["mean_logp"]

        swap_qb_follows_reassignment = (s_qb_orig_va > s_qb_orig_vb)
        swap_qb_correct_source = (s_qb_orig_va > s_qb_new)

        rec = {
            "pair_id": pid, "split": split,
            "swap_qa_follows_reassignment": swap_qa_follows_reassignment,
            "swap_qb_follows_reassignment": swap_qb_follows_reassignment,
            "swap_qa_correct_source": swap_qa_correct_source,
            "swap_qb_correct_source": swap_qb_correct_source,
            "swap_qa_margin": s_qa_orig_vb - s_qa_orig_va,  # >0 = follows swap
            "swap_qb_margin": s_qb_orig_va - s_qb_orig_vb,  # >0 = follows swap
            "swap_retain_qa_scores": {k: v["mean_logp"] for k, v in swap_retain_qa.items()},
            "swap_retain_qb_scores": {k: v["mean_logp"] for k, v in swap_retain_qb.items()},
        }
        records.append(rec)

    n = len(records)
    summary = {
        "label": label,
        "n_pairs": n,
        "n_swap_qa_follows": sum(1 for r in records if r["swap_qa_follows_reassignment"]),
        "n_swap_qb_follows": sum(1 for r in records if r["swap_qb_follows_reassignment"]),
        "n_swap_both_follow": sum(1 for r in records if r["swap_qa_follows_reassignment"] and r["swap_qb_follows_reassignment"]),
        "n_swap_qa_correct_source": sum(1 for r in records if r["swap_qa_correct_source"]),
        "n_swap_qb_correct_source": sum(1 for r in records if r["swap_qb_correct_source"]),
        "mean_swap_qa_margin": finite_mean([r["swap_qa_margin"] for r in records]),
        "mean_swap_qb_margin": finite_mean([r["swap_qb_margin"] for r in records]),
    }
    return summary, records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=40040)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--private-scale", type=float, default=0.75)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    # 1. Construct repaired pairs
    print("=== Phase 1: Constructing repaired pairs ===", flush=True)
    v1_pairs = read_jsonl(V1_PAIRS)
    pairs = construct_repaired_pairs(v1_pairs, args.seed)
    train_pairs = [p for p in pairs if p.get("split") == "train"]
    held_pairs = [p for p in pairs if p.get("split") == "held"]
    scoring_rows = build_scoring_rows(pairs)
    train_rows = [r for r in scoring_rows if r.get("split") == "train" and r.get("role") != "NEUTRAL"]
    print(f"  {len(pairs)} pairs ({len(train_pairs)} train, {len(held_pairs)} held)", flush=True)

    # 2. Build reassignment contexts
    print("=== Phase 2: Building source-reassignment contexts ===", flush=True)
    reassignment_ctxs = []
    reassignment_valid_pairs = []
    n_reassign_valid = 0
    for p in pairs:
        ctx, valid, info = construct_reassignment_contexts(p)
        if valid:
            reassignment_ctxs.append(ctx)
            reassignment_valid_pairs.append(p)
            n_reassign_valid += 1
        else:
            reassignment_ctxs.append(None)
    print(f"  {n_reassign_valid}/{len(pairs)} pairs have valid source swap", flush=True)

    # 3. Load parent and score three-way
    print("=== Phase 3: Parent three-way scoring ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, use_fast=True)
    model, load_info = load_trusted_model(device, private_scale=args.private_scale)
    ident = model_identity(model, load_info)
    print(f"  Parent: {ident['class']}, private_params={ident['private_params']}, scales={ident['executed_private_scales']}", flush=True)

    model.eval()
    parent_tw_all = []
    for i, p in enumerate(pairs):
        tw = threeway_score_pair(model, tokenizer, p, device, args.seq_length)
        parent_tw_all.append(tw)
        if (i + 1) % 30 == 0:
            print(f"  Parent three-way: {i+1}/{len(pairs)}", flush=True)

    parent_held_tw = [tw for tw, p in zip(parent_tw_all, pairs) if p.get("split") == "held"]
    parent_held_pairs = held_pairs
    parent_summary_all, parent_records_all = analyze_threeway(parent_tw_all, pairs, "parent_all")
    parent_summary_held, parent_records_held = analyze_threeway(parent_held_tw, parent_held_pairs, "parent_held")
    print(f"  Parent held: cross-source retrieval both={parent_summary_held['n_retain_both_entity_retrieval']}/{parent_summary_held['n_pairs']}, "
          f"mean cross-source={parent_summary_held['mean_retain_cross_source']:+.3f}", flush=True)

    # 4. Train model (same params as research e80)
    print("=== Phase 4: Training e80 answer-only ===", flush=True)
    trainable_info, train_time = train_model(
        model, tokenizer, train_rows, args.epochs, args.batch_size,
        args.lr, args.weight_decay, args.max_grad_norm, args.seed,
        device, args.seq_length
    )

    # 5. Trained model three-way scoring
    print("=== Phase 5: Trained model three-way scoring ===", flush=True)
    model.eval()
    trained_tw_all = []
    for i, p in enumerate(pairs):
        tw = threeway_score_pair(model, tokenizer, p, device, args.seq_length)
        trained_tw_all.append(tw)
        if (i + 1) % 30 == 0:
            print(f"  Trained three-way: {i+1}/{len(pairs)}", flush=True)

    trained_held_tw = [tw for tw, p in zip(trained_tw_all, pairs) if p.get("split") == "held"]
    trained_train_tw = [tw for tw, p in zip(trained_tw_all, pairs) if p.get("split") == "train"]
    trained_summary_all, trained_records_all = analyze_threeway(trained_tw_all, pairs, "trained_all")
    trained_summary_held, trained_records_held = analyze_threeway(trained_held_tw, held_pairs, "trained_held")
    trained_summary_train, trained_records_train = analyze_threeway(trained_train_tw, train_pairs, "trained_train")
    print(f"  Trained held: cross-source retrieval both={trained_summary_held['n_retain_both_entity_retrieval']}/{trained_summary_held['n_pairs']}, "
          f"mean cross-source={trained_summary_held['mean_retain_cross_source']:+.3f}", flush=True)
    print(f"  Trained train: cross-source retrieval both={trained_summary_train['n_retain_both_entity_retrieval']}/{trained_summary_train['n_pairs']}, "
          f"mean cross-source={trained_summary_train['mean_retain_cross_source']:+.3f}", flush=True)

    # 6. Source-reassignment test on trained model (held pairs only)
    print("=== Phase 6: Source-reassignment test (trained, held) ===", flush=True)
    valid_held_pairs_for_swap = []
    valid_held_reassignment_ctxs = []
    valid_held_tw_orig = []
    for p, ctx, tw in zip(pairs, reassignment_ctxs, trained_tw_all):
        if p.get("split") == "held" and ctx is not None:
            valid_held_pairs_for_swap.append(p)
            valid_held_reassignment_ctxs.append(ctx)
            valid_held_tw_orig.append(tw)
    print(f"  {len(valid_held_pairs_for_swap)} held pairs have valid source swap", flush=True)

    reassignment_results = []
    for i, (p, ctx) in enumerate(zip(valid_held_pairs_for_swap, valid_held_reassignment_ctxs)):
        rr = reassignment_score_pair(model, tokenizer, p, ctx, device, args.seq_length)
        reassignment_results.append(rr)
    reassignment_summary, reassignment_records = analyze_reassignment(
        reassignment_results, valid_held_pairs_for_swap, valid_held_tw_orig, "trained_held_reassignment"
    )
    print(f"  Reassignment held: follows swap both={reassignment_summary['n_swap_both_follow']}/{reassignment_summary['n_pairs']}, "
          f"mean margins qa={reassignment_summary['mean_swap_qa_margin']:+.3f} qb={reassignment_summary['mean_swap_qb_margin']:+.3f}", flush=True)

    # 7. Source-reassignment on parent (held)
    print("=== Phase 7: Source-reassignment test (parent, held) ===", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model_parent, load_info_parent = load_trusted_model(device, private_scale=args.private_scale)
    model_parent.eval()
    parent_reassignment_results = []
    for i, (p, ctx) in enumerate(zip(valid_held_pairs_for_swap, valid_held_reassignment_ctxs)):
        rr = reassignment_score_pair(model_parent, tokenizer, p, ctx, device, args.seq_length)
        parent_reassignment_results.append(rr)
    parent_reassignment_summary, parent_reassignment_records = analyze_reassignment(
        parent_reassignment_results, valid_held_pairs_for_swap,
        [tw for tw, p in zip(parent_tw_all, pairs) if p.get("split") == "held" and
         any(p["pair_id"] == vp["pair_id"] for vp in valid_held_pairs_for_swap)],
        "parent_held_reassignment"
    )
    print(f"  Parent reassignment held: follows swap both={parent_reassignment_summary['n_swap_both_follow']}/{parent_reassignment_summary['n_pairs']}", flush=True)
    del model_parent
    gc.collect()
    torch.cuda.empty_cache()

    # 8. Save everything
    print("=== Phase 8: Saving results ===", flush=True)
    full_result = {
        "status": "THREEWAY_AND_REASSIGNMENT_PROBE",
        "model_identity": ident,
        "training_params": {"epochs": args.epochs, "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size},
        "training_time_sec": train_time,
        "n_pairs": len(pairs),
        "n_reassignment_valid_held": len(valid_held_pairs_for_swap),
        "parent_threeway": {
            "all": parent_summary_all,
            "held": parent_summary_held,
        },
        "trained_threeway": {
            "all": trained_summary_all,
            "train": trained_summary_train,
            "held": trained_summary_held,
        },
        "reassignment": {
            "trained_held": reassignment_summary,
            "parent_held": parent_reassignment_summary,
        },
    }
    (out_dir / "threeway_reassignment_summary.json").write_text(
        json.dumps(full_result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(out_dir / "trained_held_threeway_records.jsonl", trained_records_held)
    write_jsonl(out_dir / "trained_train_threeway_records.jsonl", trained_records_train)
    write_jsonl(out_dir / "parent_held_threeway_records.jsonl", parent_records_held)
    write_jsonl(out_dir / "reassignment_records.jsonl", reassignment_records)
    write_jsonl(out_dir / "parent_reassignment_records.jsonl", parent_reassignment_records)

    # Print key result
    print("\n" + "=" * 60, flush=True)
    print(json.dumps(full_result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
