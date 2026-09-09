#!/usr/bin/env python3
"""research: target-aligned frozen-model source-use probe.

Tests whether the DeBERTa MLM uses ordered view structure to help reconstruct
masked view targets when a source is present, beyond generic fluency preference.

For each fixed-word-multiset family (compact/scrambled, prefix/scrambled,
onegap/scrambled), masks identical stable word occurrences in both ordered and
scrambled views and scores four conditions:
  NLL(V_ord|S)  – ordered view + source
  NLL(V_scr|S)  – scrambled view + source
  NLL(V_ord)    – ordered view alone
  NLL(V_scr)    – scrambled view alone

The principal estimand is the source-conditioned interaction:
  I_f = [NLL(V_scr|S) - NLL(V_ord|S)] - [NLL(V_scr) - NLL(V_ord)]

Positive I_f means ordered structure specifically helps source-conditioned
reconstruction beyond generic fluency.  Stratified by copy zone, lex class,
source decile, source match multiplicity, and distance.  Pair-clustered SE.

Runs on a single GPU with batched forward passes for efficiency.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
CHCK82 = ROOT / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M"
CHCK100 = ROOT / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M"
PAIR_DIR = ROOT / "data/factorial_view_candidate_audit"
TARGET_RECORDS = ROOT / "data/target_level_factor_alignment_audit/view_word_target_records.csv"

FAMILIES = [
    ("compact", "compact_scrambled"),
    ("prefix_fluent", "prefix_scrambled"),
    ("sourcewide_onegap", "sourcewide_onegap_scrambled"),
]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(t: str) -> int:
    return len(t.split())


def load_pairs(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Load pair JSONL, keyed by pair_id."""
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["pair_id"]] = r
    return out


def load_target_strata(path: pathlib.Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load pre-computed target-level stratum annotations, keyed by (variant, pair_id, view_word_pos)."""
    strata: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["variant"], row["pair_id"], int(row["view_word_pos"]))
            strata[key] = row
    return strata


def match_word_positions(ordered_words: list[str], scrambled_words: list[str]) -> list[tuple[int, int]]:
    """Match word positions between ordered and scrambled views (same word multiset).
    
    Returns list of (ordered_pos, scrambled_pos) pairs.
    For multiply-occurring words, matches by occurrence order within each word type.
    """
    # Build occurrence lists for scrambled
    scr_by_word: dict[str, list[int]] = collections.defaultdict(list)
    for i, w in enumerate(scrambled_words):
        scr_by_word[w.lower()].append(i)
    
    # Track consumed positions
    consumed: dict[str, int] = collections.defaultdict(int)  # word -> next index in scr_by_word
    matches = []
    for ord_i, w in enumerate(ordered_words):
        wl = w.lower()
        idx = consumed[wl]
        if wl in scr_by_word and idx < len(scr_by_word[wl]):
            scr_i = scr_by_word[wl][idx]
            consumed[wl] += 1
            matches.append((ord_i, scr_i))
    return matches


def find_word_token_spans(tokenizer, text: str, words: list[str]) -> list[tuple[int, int]]:
    """Find token spans for each whitespace word in text.
    
    Returns list of (start_token_idx, end_token_idx) exclusive-end spans
    relative to the encode_no_special output.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]  # list of (char_start, char_end) per token
    
    # Find character spans for each whitespace word
    word_char_spans = []
    pos = 0
    for w in words:
        idx = text.find(w, pos)
        if idx < 0:
            # Fallback: try case-insensitive or skip
            idx = text.lower().find(w.lower(), pos)
        if idx < 0:
            word_char_spans.append(None)
            continue
        word_char_spans.append((idx, idx + len(w)))
        pos = idx + len(w)
    
    # Map character spans to token spans
    token_spans = []
    for wcs in word_char_spans:
        if wcs is None:
            token_spans.append(None)
            continue
        wstart, wend = wcs
        tstart = None
        tend = None
        for ti, (cs, ce) in enumerate(offsets):
            if cs == ce == 0 and ti > 0:
                continue  # skip special padding offsets
            if ce > wstart and tstart is None:
                tstart = ti
            if cs < wend:
                tend = ti + 1
        if tstart is not None and tend is not None:
            token_spans.append((tstart, tend))
        else:
            token_spans.append(None)
    
    return token_spans


def build_mlm_context(bos_id, eos_id, source_ids, view_ids, max_len=256):
    """Build [BOS] source [SEP] view [SEP], truncated to max_len."""
    ids = []
    if bos_id is not None:
        ids.append(bos_id)
    src_start = len(ids)
    ids.extend(source_ids)
    if eos_id is not None:
        ids.append(eos_id)
    view_start = len(ids)
    ids.extend(view_ids)
    if eos_id is not None:
        ids.append(eos_id)
    # Truncate
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids, src_start, view_start


def build_view_only_context(bos_id, eos_id, view_ids, max_len=256):
    """Build [BOS] view [SEP], truncated to max_len."""
    ids = []
    if bos_id is not None:
        ids.append(bos_id)
    view_start = len(ids)
    ids.extend(view_ids)
    if eos_id is not None:
        ids.append(eos_id)
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids, view_start


def score_masked_word_nll(model, device, ids_list: list[list[int]], 
                          mask_spans: list[tuple[int, int]], mask_id: int,
                          orig_ids_list: list[list[int]], max_len: int = 256) -> list[float]:
    """Score NLL for masked whole-word targets in a batch.
    
    Each element: mask the BPE span in its context, compute mean NLL across span.
    ids_list[i] is the full token sequence, mask_spans[i] is (start, end) of the
    word's BPE tokens, orig_ids_list[i] has the original token ids.
    
    Returns list of mean NLL per word.
    """
    if not ids_list:
        return []
    
    B = len(ids_list)
    L = max(len(ids) for ids in ids_list)
    L = min(L, max_len)
    
    # Pad and create tensors
    input_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    attention_mask = torch.zeros(B, L, dtype=torch.long, device=device)
    
    for i, ids in enumerate(ids_list):
        seq_len = min(len(ids), L)
        input_ids[i, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        # Apply mask
        start, end = mask_spans[i]
        end = min(end, seq_len)
        if start < seq_len:
            input_ids[i, start:end] = mask_id
    
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    results = []
    for i in range(B):
        start, end = mask_spans[i]
        seq_len = min(len(ids_list[i]), L)
        end = min(end, seq_len)
        if start >= seq_len:
            results.append(float("nan"))
            continue
        
        span_nll = 0.0
        span_count = 0
        for pos in range(start, end):
            orig_id = orig_ids_list[i][pos]
            lp = torch.log_softmax(logits[i, pos].float(), dim=-1)[orig_id]
            span_nll += float(-lp.cpu())
            span_count += 1
        
        results.append(span_nll / max(span_count, 1))
    
    return results


def pair_clustered_se(values: list[float], pair_ids: list[str]) -> float:
    """Compute pair-clustered standard error."""
    # Group by pair_id
    clusters: dict[str, list[float]] = collections.defaultdict(list)
    for v, pid in zip(values, pair_ids):
        if math.isfinite(v):
            clusters[pid].append(v)
    
    if len(clusters) < 2:
        return float("nan")
    
    # Cluster means
    cluster_means = [np.mean(vs) for vs in clusters.values()]
    grand_mean = np.mean(cluster_means)
    
    n_clusters = len(cluster_means)
    var = np.sum([(m - grand_mean) ** 2 for m in cluster_means]) / (n_clusters - 1)
    se = np.sqrt(var / n_clusters)
    return float(se)


def compute_stratum_stats(records: list[dict], stratum_key: str) -> list[dict]:
    """Compute I_f statistics grouped by stratum_key."""
    groups: dict[Any, list[dict]] = collections.defaultdict(list)
    for r in records:
        groups[r.get(stratum_key, "unknown")].append(r)
    
    stats = []
    for key, recs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        I_f_vals = [r["I_f"] for r in recs if math.isfinite(r.get("I_f", float("nan")))]
        pair_ids = [r["pair_id"] for r in recs if math.isfinite(r.get("I_f", float("nan")))]
        
        if not I_f_vals:
            continue
        
        nll_ord_s = [r["nll_ord_s"] for r in recs if math.isfinite(r.get("nll_ord_s", float("nan")))]
        nll_scr_s = [r["nll_scr_s"] for r in recs if math.isfinite(r.get("nll_scr_s", float("nan")))]
        nll_ord = [r["nll_ord"] for r in recs if math.isfinite(r.get("nll_ord", float("nan")))]
        nll_scr = [r["nll_scr"] for r in recs if math.isfinite(r.get("nll_scr", float("nan")))]
        
        stats.append({
            "stratum": stratum_key,
            "key": str(key),
            "n_targets": len(I_f_vals),
            "n_pairs": len(set(pair_ids)),
            "mean_I_f": float(np.mean(I_f_vals)),
            "median_I_f": float(np.median(I_f_vals)),
            "se_I_f": pair_clustered_se(I_f_vals, pair_ids),
            "positive_frac": float(np.mean([1 if v > 0 else 0 for v in I_f_vals])),
            "mean_nll_ord_s": float(np.mean(nll_ord_s)) if nll_ord_s else None,
            "mean_nll_scr_s": float(np.mean(nll_scr_s)) if nll_scr_s else None,
            "mean_nll_ord": float(np.mean(nll_ord)) if nll_ord else None,
            "mean_nll_scr": float(np.mean(nll_scr)) if nll_scr else None,
            "mean_source_effect_ord": float(np.mean([r["nll_ord"] - r["nll_ord_s"] for r in recs
                if math.isfinite(r.get("nll_ord", float("nan"))) and math.isfinite(r.get("nll_ord_s", float("nan")))])) if nll_ord else None,
            "mean_source_effect_scr": float(np.mean([r["nll_scr"] - r["nll_scr_s"] for r in recs
                if math.isfinite(r.get("nll_scr", float("nan"))) and math.isfinite(r.get("nll_scr_s", float("nan")))])) if nll_scr else None,
        })
    
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHCK82))
    ap.add_argument("--model-label", default="chck82")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--targets-per-pair", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=183013)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target-records", default=str(TARGET_RECORDS))
    args = ap.parse_args()
    
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    
    # ─── Load pre-computed target strata ───
    print("Loading target strata...", flush=True)
    strata_path = pathlib.Path(args.target_records)
    strata = load_target_strata(strata_path)
    print(f"  Loaded {len(strata)} stratum records", flush=True)
    
    # ─── Load pair files for all families ───
    all_family_data = {}
    for ord_var, scr_var in FAMILIES:
        ord_path = PAIR_DIR / f"{ord_var}_candidate_pairs.jsonl"
        scr_path = PAIR_DIR / f"{scr_var}_candidate_pairs.jsonl"
        ord_pairs = load_pairs(ord_path)
        scr_pairs = load_pairs(scr_path)
        common_ids = sorted(set(ord_pairs.keys()) & set(scr_pairs.keys()))
        all_family_data[(ord_var, scr_var)] = (ord_pairs, scr_pairs, common_ids)
        print(f"  Family {ord_var}/{scr_var}: {len(common_ids)} common pairs", flush=True)
    
    # ─── Setup model ───
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.dry_run else "cpu"
    
    cache = out_dir / ".hf_cache"
    os.environ["HF_HOME"] = str(cache)
    os.environ["TRANSFORMERS_CACHE"] = str(cache)
    os.environ["HF_MODULES_CACHE"] = str(cache / "modules")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    checkpoint = pathlib.Path(args.checkpoint)
    print(f"Loading tokenizer from {checkpoint}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    
    bos_id = getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "cls_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "sep_token_id", None)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("MLM probe requires mask_token_id")
    
    model = None
    if not args.dry_run:
        print(f"Loading model from {checkpoint} with trust_remote_code=True...", flush=True)
        device = torch.device(device_str)
        model = AutoModelForMaskedLM.from_pretrained(
            str(checkpoint), trust_remote_code=True
        ).to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded: {n_params:,} params on {device_str}", flush=True)
    else:
        device = torch.device("cpu")
        print("  Dry run: model not loaded", flush=True)
    
    # ─── Process each family ───
    all_records = []
    family_summaries = {}
    
    for (ord_var, scr_var), (ord_pairs, scr_pairs, common_ids) in all_family_data.items():
        family_label = f"{ord_var}_vs_{scr_var}"
        print(f"\nProcessing family: {family_label}", flush=True)
        
        # Subsample pairs if requested
        if args.max_pairs > 0 and len(common_ids) > args.max_pairs:
            selected_ids = rng.sample(common_ids, args.max_pairs)
        else:
            selected_ids = common_ids
        
        # Collect all scoring tasks
        scoring_tasks = []  # list of dicts with context info and metadata
        
        for pair_id in selected_ids:
            ord_rec = ord_pairs[pair_id]
            scr_rec = scr_pairs[pair_id]
            source_text = ord_rec["source_text"]
            ord_view = ord_rec["view_text"]
            scr_view = scr_rec["view_text"]
            
            # Tokenize segments
            source_ids = tokenizer.encode(source_text, add_special_tokens=False)
            ord_view_ids = tokenizer.encode(ord_view, add_special_tokens=False)
            scr_view_ids = tokenizer.encode(scr_view, add_special_tokens=False)
            
            # Find word-level token spans
            ord_words = ord_view.split()
            scr_words = scr_view.split()
            
            ord_spans = find_word_token_spans(tokenizer, ord_view, ord_words)
            scr_spans = find_word_token_spans(tokenizer, scr_view, scr_words)
            
            # Match word positions between ordered and scrambled
            matches = match_word_positions(ord_words, scr_words)
            
            # Filter to valid targets: both spans found, within max_length
            valid_targets = []
            for ord_i, scr_i in matches:
                if ord_spans[ord_i] is None or scr_spans[scr_i] is None:
                    continue
                # Get stratum info from pre-computed records
                stratum = strata.get((ord_var, pair_id, ord_i), {})
                if not stratum:
                    continue  # skip if no stratum annotation
                valid_targets.append((ord_i, scr_i, stratum))
            
            # Subsample targets
            if len(valid_targets) > args.targets_per_pair:
                # Prefer diverse strata: try to get mix of copy zones
                copy_zone_groups: dict[str, list] = collections.defaultdict(list)
                for vt in valid_targets:
                    cz = vt[2].get("copy_zone", "unknown")
                    copy_zone_groups[cz].append(vt)
                
                selected_targets = []
                remaining = args.targets_per_pair
                zones = list(copy_zone_groups.keys())
                rng.shuffle(zones)
                per_zone = max(1, remaining // max(len(zones), 1))
                for z in zones:
                    take = min(per_zone, len(copy_zone_groups[z]), remaining)
                    selected_targets.extend(rng.sample(copy_zone_groups[z], take))
                    remaining -= take
                    if remaining <= 0:
                        break
                if remaining > 0:
                    leftover = [vt for vt in valid_targets if vt not in selected_targets]
                    if leftover:
                        selected_targets.extend(rng.sample(leftover, min(remaining, len(leftover))))
            else:
                selected_targets = valid_targets
            
            # Build contexts for each target
            for ord_i, scr_i, stratum in selected_targets:
                ord_span = ord_spans[ord_i]
                scr_span = scr_spans[scr_i]
                
                # Build 4 contexts
                # 1. source + ordered view
                ctx_ord_s, _, view_start_ord_s = build_mlm_context(
                    bos_id, eos_id, source_ids, ord_view_ids, args.max_length)
                mask_span_ord_s = (view_start_ord_s + ord_span[0], view_start_ord_s + ord_span[1])
                
                # 2. source + scrambled view
                ctx_scr_s, _, view_start_scr_s = build_mlm_context(
                    bos_id, eos_id, source_ids, scr_view_ids, args.max_length)
                mask_span_scr_s = (view_start_scr_s + scr_span[0], view_start_scr_s + scr_span[1])
                
                # 3. ordered view only
                ctx_ord, view_start_ord = build_view_only_context(
                    bos_id, eos_id, ord_view_ids, args.max_length)
                mask_span_ord = (view_start_ord + ord_span[0], view_start_ord + ord_span[1])
                
                # 4. scrambled view only
                ctx_scr, view_start_scr = build_view_only_context(
                    bos_id, eos_id, scr_view_ids, args.max_length)
                mask_span_scr = (view_start_scr + scr_span[0], view_start_scr + scr_span[1])
                
                # Check all spans are within bounds
                valid = True
                for ctx, span in [(ctx_ord_s, mask_span_ord_s), (ctx_scr_s, mask_span_scr_s),
                                  (ctx_ord, mask_span_ord), (ctx_scr, mask_span_scr)]:
                    if span[0] >= len(ctx) or span[1] > len(ctx) or span[0] >= span[1]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                scoring_tasks.append({
                    "pair_id": pair_id,
                    "family": family_label,
                    "ord_variant": ord_var,
                    "scr_variant": scr_var,
                    "target_word": ord_words[ord_i],
                    "ord_word_pos": ord_i,
                    "scr_word_pos": scr_i,
                    "stratum": stratum,
                    # Contexts and spans
                    "ctx_ord_s": ctx_ord_s,
                    "mask_ord_s": mask_span_ord_s,
                    "ctx_scr_s": ctx_scr_s,
                    "mask_scr_s": mask_span_scr_s,
                    "ctx_ord": ctx_ord,
                    "mask_ord": mask_span_ord,
                    "ctx_scr": ctx_scr,
                    "mask_scr": mask_span_scr,
                })
        
        print(f"  {len(scoring_tasks)} scoring tasks from {len(selected_ids)} pairs", flush=True)
        
        if args.dry_run:
            # Just record task counts
            family_summaries[family_label] = {
                "n_pairs": len(selected_ids),
                "n_tasks": len(scoring_tasks),
                "dry_run": True,
            }
            continue
        
        # ─── Batched scoring ───
        family_records = []
        conditions = ["ord_s", "scr_s", "ord", "scr"]
        ctx_keys = ["ctx_ord_s", "ctx_scr_s", "ctx_ord", "ctx_scr"]
        mask_keys = ["mask_ord_s", "mask_scr_s", "mask_ord", "mask_scr"]
        
        # Score all tasks for each condition
        condition_scores = {cond: [] for cond in conditions}
        
        for cond_idx, (cond, ctx_key, mask_key) in enumerate(zip(conditions, ctx_keys, mask_keys)):
            print(f"    Scoring condition: {cond} ({len(scoring_tasks)} targets)...", flush=True)
            
            batch_ids = []
            batch_spans = []
            batch_orig_ids = []
            batch_indices = []
            
            for task_idx, task in enumerate(scoring_tasks):
                ctx = task[ctx_key]
                span = task[mask_key]
                batch_ids.append(ctx)
                batch_spans.append(span)
                batch_orig_ids.append(ctx)  # original ids before masking
                batch_indices.append(task_idx)
                
                if len(batch_ids) >= args.batch_size or task_idx == len(scoring_tasks) - 1:
                    nlls = score_masked_word_nll(
                        model, device, batch_ids, batch_spans, mask_id,
                        batch_orig_ids, args.max_length
                    )
                    for bi, nll in zip(batch_indices, nlls):
                        condition_scores[cond].append((bi, nll))
                    
                    batch_ids = []
                    batch_spans = []
                    batch_orig_ids = []
                    batch_indices = []
        
        # Build per-target records
        score_lookup = {cond: {} for cond in conditions}
        for cond in conditions:
            for bi, nll in condition_scores[cond]:
                score_lookup[cond][bi] = nll
        
        for task_idx, task in enumerate(scoring_tasks):
            nll_ord_s = score_lookup["ord_s"].get(task_idx, float("nan"))
            nll_scr_s = score_lookup["scr_s"].get(task_idx, float("nan"))
            nll_ord = score_lookup["ord"].get(task_idx, float("nan"))
            nll_scr = score_lookup["scr"].get(task_idx, float("nan"))
            
            # Source-conditioned interaction
            if all(math.isfinite(x) for x in [nll_ord_s, nll_scr_s, nll_ord, nll_scr]):
                I_f = (nll_scr_s - nll_ord_s) - (nll_scr - nll_ord)
                source_effect_ord = nll_ord - nll_ord_s  # positive = source helps
                source_effect_scr = nll_scr - nll_scr_s
                order_effect_with_source = nll_scr_s - nll_ord_s  # positive = ordered better
                order_effect_without_source = nll_scr - nll_ord
            else:
                I_f = float("nan")
                source_effect_ord = float("nan")
                source_effect_scr = float("nan")
                order_effect_with_source = float("nan")
                order_effect_without_source = float("nan")
            
            s = task["stratum"]
            rec = {
                "pair_id": task["pair_id"],
                "family": task["family"],
                "target_word": task["target_word"],
                "ord_word_pos": task["ord_word_pos"],
                "scr_word_pos": task["scr_word_pos"],
                "nll_ord_s": nll_ord_s,
                "nll_scr_s": nll_scr_s,
                "nll_ord": nll_ord,
                "nll_scr": nll_scr,
                "I_f": I_f,
                "source_effect_ord": source_effect_ord,
                "source_effect_scr": source_effect_scr,
                "order_effect_with_source": order_effect_with_source,
                "order_effect_without_source": order_effect_without_source,
                # Stratum labels
                "copy_zone": s.get("copy_zone", "unknown"),
                "lex_class": s.get("lex_class", "unknown"),
                "source_match_bin": s.get("source_match_bin", "unknown"),
                "source_decile_min": s.get("source_decile_min", ""),
                "source_decile_max": s.get("source_decile_max", ""),
                "counterpart_visible": s.get("counterpart_full_visible", ""),
                "complete_bpe_copy": s.get("complete_bpe_copy_any_source", ""),
                "contentlike": s.get("lex_class", "") in ("content", "capitalized_content", "number_like"),
                "min_visible_gap": s.get("min_visible_source_to_view_token_gap", ""),
            }
            family_records.append(rec)
        
        all_records.extend(family_records)
        
        # Family-level statistics
        valid_recs = [r for r in family_records if math.isfinite(r.get("I_f", float("nan")))]
        I_f_vals = [r["I_f"] for r in valid_recs]
        I_f_pair_ids = [r["pair_id"] for r in valid_recs]
        
        family_summaries[family_label] = {
            "n_pairs": len(selected_ids),
            "n_tasks": len(scoring_tasks),
            "n_valid": len(valid_recs),
            "mean_I_f": float(np.mean(I_f_vals)) if I_f_vals else None,
            "median_I_f": float(np.median(I_f_vals)) if I_f_vals else None,
            "se_I_f": pair_clustered_se(I_f_vals, I_f_pair_ids) if I_f_vals else None,
            "positive_frac": float(np.mean([1 if v > 0 else 0 for v in I_f_vals])) if I_f_vals else None,
            "mean_nll_ord_s": float(np.mean([r["nll_ord_s"] for r in valid_recs])),
            "mean_nll_scr_s": float(np.mean([r["nll_scr_s"] for r in valid_recs])),
            "mean_nll_ord": float(np.mean([r["nll_ord"] for r in valid_recs])),
            "mean_nll_scr": float(np.mean([r["nll_scr"] for r in valid_recs])),
        }
        
        print(f"  Family {family_label}: {len(valid_recs)} valid targets, "
              f"mean I_f = {family_summaries[family_label].get('mean_I_f', 'N/A')}", flush=True)
    
    # ─── Global stratified analysis ───
    strata_keys = ["family", "copy_zone", "lex_class", "source_match_bin", 
                   "counterpart_visible", "complete_bpe_copy", "contentlike"]
    stratified_stats = {}
    for sk in strata_keys:
        stratified_stats[sk] = compute_stratum_stats(all_records, sk)
    
    # Cross-strata: family × copy_zone
    for r in all_records:
        r["family_x_copy_zone"] = f"{r['family']}/{r['copy_zone']}"
    stratified_stats["family_x_copy_zone"] = compute_stratum_stats(all_records, "family_x_copy_zone")
    
    # ─── Write outputs ───
    result = {
        "status": "FROZEN_SOURCE_USE_PROBE",
        "checkpoint": str(args.checkpoint),
        "model_label": args.model_label,
        "dry_run": args.dry_run,
        "seed": args.seed,
        "max_pairs": args.max_pairs,
        "targets_per_pair": args.targets_per_pair,
        "n_total_records": len(all_records),
        "family_summaries": family_summaries,
        "stratified_stats": stratified_stats,
        "elapsed_sec": time.time() - t0,
    }
    
    (out_dir / "frozen_source_use_probe.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    
    # Save records as CSV
    if all_records:
        csv_path = out_dir / "source_use_probe_records.csv"
        fieldnames = list(all_records[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_records:
                writer.writerow(r)
    
    # Write readable report
    lines = [f"# research frozen-model source-use probe ({args.model_label})", ""]
    lines.append(f"Checkpoint: `{args.checkpoint}`")
    lines.append(f"Total target records: {len(all_records)}")
    lines.append(f"Dry run: {args.dry_run}")
    lines.append(f"Elapsed: {time.time() - t0:.1f}s")
    lines.append("")
    
    lines.append("## Family summaries")
    lines.append("| family | n_pairs | n_valid | mean I_f | se I_f | positive frac | mean NLL(ord|S) | mean NLL(scr|S) | mean NLL(ord) | mean NLL(scr) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fl, fs in family_summaries.items():
        if fs.get("dry_run"):
            lines.append(f"| {fl} | {fs['n_pairs']} | {fs['n_tasks']} tasks | dry run |  |  |  |  |  |  |")
        else:
            def fmt(x):
                return "" if x is None else f"{float(x):.6f}"
            lines.append(f"| {fl} | {fs['n_pairs']} | {fs.get('n_valid','')} | "
                        f"{fmt(fs.get('mean_I_f'))} | {fmt(fs.get('se_I_f'))} | "
                        f"{fmt(fs.get('positive_frac'))} | {fmt(fs.get('mean_nll_ord_s'))} | "
                        f"{fmt(fs.get('mean_nll_scr_s'))} | {fmt(fs.get('mean_nll_ord'))} | "
                        f"{fmt(fs.get('mean_nll_scr'))} |")
    lines.append("")
    
    lines.append("## Stratified I_f analysis")
    for sk, stats_list in stratified_stats.items():
        lines.append(f"\n### By {sk}")
        lines.append("| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for s in stats_list:
            def fmt(x):
                return "" if x is None else f"{float(x):.4f}"
            lines.append(f"| {s['key']} | {s['n_targets']} | {s['n_pairs']} | "
                        f"{fmt(s.get('mean_I_f'))} | {fmt(s.get('se_I_f'))} | "
                        f"{fmt(s.get('positive_frac'))} | {fmt(s.get('mean_source_effect_ord'))} | "
                        f"{fmt(s.get('mean_source_effect_scr'))} |")
    
    lines.append("")
    lines.append("## Interpretation")
    lines.append("I_f = [NLL(V_scr|S) - NLL(V_ord|S)] - [NLL(V_scr) - NLL(V_ord)]")
    lines.append("")
    lines.append("Positive I_f: ordered structure specifically helps source-conditioned "
                 "reconstruction beyond generic fluency preference.")
    lines.append("Near-zero I_f: ordering helps equally with and without source; "
                 "generic fluency, not source-use structure.")
    lines.append("Negative I_f: ordering helps *less* when source is present; "
                 "source makes ordering irrelevant.")
    lines.append("")
    lines.append(f"Records CSV: `{out_dir}/source_use_probe_records.csv`")
    lines.append(f"JSON: `{out_dir}/frozen_source_use_probe.json`")
    
    (out_dir / "frozen_source_use_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    print(json.dumps({
        "status": result["status"],
        "out_dir": str(out_dir),
        "n_records": len(all_records),
        "families": list(family_summaries.keys()),
        "elapsed_sec": result["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
