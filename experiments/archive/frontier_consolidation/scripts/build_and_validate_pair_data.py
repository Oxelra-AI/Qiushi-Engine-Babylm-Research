#!/usr/bin/env python3
"""research: build and validate pair-row data for dual-view training.

For each of the 3,005 pair rows in the legal cleanqwen overlay pool:
1. Reconstruct which text spans are source vs rewrite using pair JSONL metadata
2. Tokenize both full-row and rewrite-only texts with offset_mapping
3. Build token-level mapping: full-row token idx -> rewrite-only token idx
4. Validate mapping covers all rewrite tokens
5. Save pre-computed pair data for the dual-view trainer

Also validates that the rewrite-only tokenization can be correctly masked using
the same word groups as the full-row tokenization, ensuring "same-target, same-mask"
for the dual-view mechanism.
"""
import json, sys, time, hashlib
from pathlib import Path
from dataclasses import dataclass

import torch

STUDY = Path("experiments/archive/frontier_consolidation")
WORKSPACE = STUDY

POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
PAIR_JSONL = WORKSPACE / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
META_JSONL = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
OUT_DIR = WORKSPACE / "data/pair_data"

SEQ_LENGTH = 256


def is_word_start(tok_str: str) -> bool:
    return tok_str.startswith("Ġ") or tok_str.startswith("▁")


def assign_word_groups(input_ids, attention_mask, special_ids, tokenizer):
    """Replicate MaskedChunkDataset word-group assignment."""
    seq_len = input_ids.shape[0]
    group = torch.full((seq_len,), -1, dtype=torch.long)
    gid = -1
    cache = {}
    for i in range(seq_len):
        if attention_mask[i] == 0:
            continue
        tid = int(input_ids[i])
        if tid in special_ids:
            continue
        if tid not in cache:
            s = tokenizer.convert_ids_to_tokens(tid)
            cache[tid] = bool(s is not None and is_word_start(str(s)))
        if gid < 0 or cache[tid] or i == 0:
            gid += 1
        group[i] = gid
    return group


def main():
    start = time.time()
    
    # Setup writable HF cache
    import os
    hf_cache = str(OUT_DIR / "hf_cache")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache)
    os.environ.setdefault("HF_MODULES_CACHE", str(Path(hf_cache) / "modules"))
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    special_ids = set(tokenizer.all_special_ids)
    mask_token_id = tokenizer.mask_token_id
    
    # Load pair definitions
    pair_by_id = {}
    with open(PAIR_JSONL) as f:
        for line in f:
            p = json.loads(line)
            pair_by_id[p['pair_id']] = p
    print(f"Loaded {len(pair_by_id)} pair definitions", flush=True)
    
    # Load meta (which pairs belong to which row)
    meta_by_eid = {}
    with open(META_JSONL) as f:
        for line in f:
            m = json.loads(line)
            meta_by_eid[int(m['example_id'])] = m
    print(f"Loaded {len(meta_by_eid)} meta rows", flush=True)
    
    # Load pair rows from pool
    pair_rows = {}
    with open(POOL_10M) as f:
        for line in f:
            row = json.loads(line)
            eid = int(row['example_id'])
            if eid in meta_by_eid:
                pair_rows[eid] = row
    print(f"Found {len(pair_rows)} pair rows in pool", flush=True)
    
    # Process each pair row
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    n_ok = 0
    n_fail = 0
    total_mapped = 0
    total_rw_tokens = 0
    
    for eid in sorted(pair_rows.keys()):
        row = pair_rows[eid]
        meta = meta_by_eid[eid]
        full_text = row['text']
        pair_ids = meta['pair_ids']
        
        # Build rewrite-only text and character range mapping
        rewrite_parts = []
        rewrite_char_ranges = []  # (full_start, full_end, rw_start, rw_end)
        
        cursor = 0  # position in full text to search from
        rw_cursor = 0  # position in rewrite-only text
        
        ok = True
        for pid in pair_ids:
            if pid not in pair_by_id:
                ok = False
                break
            p = pair_by_id[pid]
            src = p['source_text']
            rw = p['rewrite_text']
            
            # Find source and rewrite in full text (forward scan)
            src_idx = full_text.find(src, cursor)
            rw_idx = full_text.find(rw, cursor)
            
            if src_idx < 0 or rw_idx < 0:
                ok = False
                break
            
            rw_start_in_full = rw_idx
            rw_end_in_full = rw_idx + len(rw)
            rw_start_in_rw = rw_cursor
            rw_end_in_rw = rw_cursor + len(rw)
            
            rewrite_char_ranges.append((rw_start_in_full, rw_end_in_full, rw_start_in_rw, rw_end_in_rw))
            rewrite_parts.append(rw)
            
            cursor = max(src_idx + len(src), rw_end_in_full)
            rw_cursor = rw_end_in_rw + 1  # +1 for space separator
        
        if not ok or not rewrite_parts:
            n_fail += 1
            continue
        
        rewrite_text = ' '.join(rewrite_parts)
        
        # Tokenize full text (no padding, with offsets)
        full_enc = tokenizer(full_text, add_special_tokens=False, truncation=True,
                            max_length=SEQ_LENGTH, return_offsets_mapping=True)
        full_ids_raw = full_enc['input_ids']
        full_offsets = full_enc['offset_mapping']
        
        # Tokenize rewrite-only (no padding, with offsets)
        rw_enc = tokenizer(rewrite_text, add_special_tokens=False, truncation=True,
                          max_length=SEQ_LENGTH, return_offsets_mapping=True)
        rw_ids_raw = rw_enc['input_ids']
        rw_offsets = rw_enc['offset_mapping']
        
        # Build rw char -> token index
        rw_char_to_tok = {}
        for j, (rs, re) in enumerate(rw_offsets):
            if rs == re:
                continue
            for c in range(rs, re):
                rw_char_to_tok[c] = j
        
        # Build full -> rw token mapping
        full_to_rw = {}
        full_is_rewrite = [False] * len(full_ids_raw)
        
        for i, (fs, fe) in enumerate(full_offsets):
            if fs == fe:
                continue
            for full_rw_start, full_rw_end, rw_start, rw_end in rewrite_char_ranges:
                if fs >= full_rw_start and fe <= full_rw_end:
                    full_is_rewrite[i] = True
                    # Map to rewrite-only char position
                    rw_char = fs - full_rw_start + rw_start
                    rw_j = rw_char_to_tok.get(rw_char)
                    if rw_j is not None:
                        full_to_rw[i] = rw_j
                    break
        
        # Pad to seq_length for training
        def pad_to_seq(ids, seq_len, pad_id):
            t = torch.full((seq_len,), pad_id, dtype=torch.long)
            n = min(len(ids), seq_len)
            t[:n] = torch.tensor(ids[:n], dtype=torch.long)
            return t
        
        pad_id = tokenizer.pad_token_id or 0
        rw_input_ids = pad_to_seq(rw_ids_raw, SEQ_LENGTH, pad_id)
        rw_attention_mask = torch.zeros(SEQ_LENGTH, dtype=torch.long)
        rw_attention_mask[:min(len(rw_ids_raw), SEQ_LENGTH)] = 1
        rw_word_group = assign_word_groups(rw_input_ids, rw_attention_mask, special_ids, tokenizer)
        
        full_input_ids = pad_to_seq(full_ids_raw, SEQ_LENGTH, pad_id)
        full_attention_mask = torch.zeros(SEQ_LENGTH, dtype=torch.long)
        full_attention_mask[:min(len(full_ids_raw), SEQ_LENGTH)] = 1
        full_word_group = assign_word_groups(full_input_ids, full_attention_mask, special_ids, tokenizer)
        
        # Validate: count mapped tokens
        n_rw_tokens = int(rw_attention_mask.sum().item())
        n_mapped = len(full_to_rw)
        n_full_rewrite = sum(full_is_rewrite)
        
        total_mapped += n_mapped
        total_rw_tokens += n_rw_tokens
        
        # Build full rewrite word group set
        full_rw_wg_set = set()
        for i, is_rw in enumerate(full_is_rewrite):
            if is_rw and full_word_group[i].item() >= 0:
                full_rw_wg_set.add(full_word_group[i].item())
        
        # Build word group level mapping: full_wg -> set of rw_wg
        full_wg_to_rw_wg = {}
        for fi, ri in full_to_rw.items():
            fwg = full_word_group[fi].item()
            rwg = rw_word_group[ri].item()
            if fwg >= 0 and rwg >= 0:
                full_wg_to_rw_wg[fwg] = rwg
        
        results[eid] = {
            'rewrite_text': rewrite_text,
            'rewrite_input_ids': rw_input_ids.tolist(),
            'rewrite_attention_mask': rw_attention_mask.tolist(),
            'rewrite_word_group': rw_word_group.tolist(),
            'full_rewrite_token_mask': full_is_rewrite[:min(len(full_ids_raw), SEQ_LENGTH)],
            'full_to_rw_token_map': {str(k): v for k, v in full_to_rw.items()},
            'full_rw_word_groups': sorted(full_rw_wg_set),
            'full_wg_to_rw_wg': {str(k): v for k, v in full_wg_to_rw_wg.items()},
            'n_pairs': len(pair_ids),
            'n_full_tokens': int(full_attention_mask.sum().item()),
            'n_rw_tokens': n_rw_tokens,
            'n_mapped_tokens': n_mapped,
            'n_full_rewrite_tokens': n_full_rewrite,
            'n_rewrite_word_groups': len(full_rw_wg_set),
            'n_wg_mapped': len(full_wg_to_rw_wg),
        }
        n_ok += 1
    
    # Summary
    summary = {
        'status': 'PAIR_DATA',
        'n_pair_rows': len(pair_rows),
        'n_ok': n_ok,
        'n_fail': n_fail,
        'total_mapped_tokens': total_mapped,
        'total_rw_tokens': total_rw_tokens,
        'mapping_coverage': total_mapped / max(1, total_rw_tokens),
        'mean_rw_tokens_per_row': total_rw_tokens / max(1, n_ok),
        'mean_mapped_per_row': total_mapped / max(1, n_ok),
        'tokenizer_sha': hashlib.sha256(
            (TOKENIZER_DIR / "tokenizer.json").read_bytes()
        ).hexdigest(),
        'pool_sha': hashlib.sha256(POOL_10M.read_bytes()).hexdigest(),
        'elapsed_sec': round(time.time() - start, 2),
    }
    
    # Save
    out_json = OUT_DIR / "pair_data.json"
    with open(out_json, 'w') as f:
        json.dump({'summary': summary, 'pair_data': {str(k): v for k, v in results.items()}},
                  f, indent=2, ensure_ascii=False)
    
    # Save summary separately
    summary_json = OUT_DIR / "pair_data_summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(json.dumps(summary, indent=2), flush=True)
    
    # Quick validation: check that word-group mapping preserves mask transfer
    if n_ok > 0:
        sample_eid = sorted(results.keys())[0]
        r = results[sample_eid]
        print(f"\nSample validation (eid={sample_eid}):")
        print(f"  Full tokens: {r['n_full_tokens']}, Rewrite tokens: {r['n_rw_tokens']}")
        print(f"  Mapped tokens: {r['n_mapped_tokens']}, Rewrite WGs: {r['n_rewrite_word_groups']}")
        print(f"  WG mapped: {r['n_wg_mapped']}")
        
        # Check that all rewrite word groups in full text map to rewrite-only word groups
        full_rw_wgs = set(r['full_rw_word_groups'])
        mapped_wgs = set(int(k) for k in r['full_wg_to_rw_wg'].keys())
        unmapped = full_rw_wgs - mapped_wgs
        print(f"  Unmapped rewrite WGs: {len(unmapped)}")


if __name__ == "__main__":
    main()
