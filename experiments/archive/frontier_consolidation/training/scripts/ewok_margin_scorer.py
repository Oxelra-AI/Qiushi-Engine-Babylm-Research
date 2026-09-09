#!/usr/bin/env python3
"""research: per-item EWoK margin scorer for relation-stability analysis.

Replicates the official BabyLM MLM EWoK scoring exactly:
  - For each EWoK item: sentence_0 = Context1 + " " + Target1  (label=0, correct)
                        sentence_1 = Context2 + " " + Target1  (incorrect)
  - Completion tokens (Target1 tokens) are masked one at a time
  - PLL = sum of log-probs at masked positions
  - The model picks the sentence with higher PLL

This script records PLL_0, PLL_1, and margin = PLL_0 - PLL_1 for every item,
enabling per-item, per-domain, per-concept flip and confidence analysis across
all four 100M treatment cells and the raw 90M checkpoint.

Runs on a single GPU; designed to be launched in parallel on two GPUs.
"""
import json, os, sys, time, pathlib
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ── paths ──────────────────────────────────────────────────────────────
EWOK_DIR = pathlib.Path(
    "experiments/archive/representation_and_objectives/data"
    "pristine_official_coordinate/babylm-eval/strict/"
    "evaluation_data/full_eval/ewok_filtered"
)

OUT_ROOT = pathlib.Path(
    "experiments/archive/frontier_consolidation/data/ewok_margins"
)

# ── model registry ─────────────────────────────────────────────────────
ALL_MODELS = {
    "clean43022": "experiments/archive/compact_experience/training/runs"
        "qwen_clean_aligned_16k_seed43022/hf_model/chck_100M",
    "clean43122": "experiments/archive/compact_experience/training/runs"
        "qwen_clean_aligned_16k_seed43122/hf_model/chck_100M",
    "reinvest43022": "experiments/archive/frontier_consolidation/training/runs"
        "cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/"
        "hf_model/chck_100M",
    "reinvest43122": "experiments/archive/representation_and_objectives/training/runs"
        "repl_compact_view_reinvest_seed43122/"
        "hf_model/chck_100M",
    "reinvest43022_90M": "experiments/archive/frontier_consolidation/training/runs"
        "cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/"
        "hf_model/chck_90M",
}

# ── load EWoK data ─────────────────────────────────────────────────────
def load_ewok_items():
    """Load all EWoK items from JSONL files, preserving domain and metadata."""
    items = []
    for fpath in sorted(EWOK_DIR.glob("*.jsonl")):
        domain = fpath.stem
        with open(fpath) as f:
            for line_idx, line in enumerate(f):
                raw = json.loads(line.strip())
                # Official decode_ewok: sentences use Target1 only
                sentence_0 = raw["Context1"] + " " + raw["Target1"]
                sentence_1 = raw["Context2"] + " " + raw["Target1"]
                items.append({
                    "domain": domain,
                    "line_idx": line_idx,
                    "ConceptA": raw.get("ConceptA", ""),
                    "ConceptB": raw.get("ConceptB", ""),
                    "ContextType": raw.get("ContextType", ""),
                    "ContextDiff": raw.get("ContextDiff", ""),
                    "TargetDiff": raw.get("TargetDiff", ""),
                    "sentence_0": sentence_0,
                    "sentence_1": sentence_1,
                    "completion": " " + raw["Target1"],
                })
    return items


def compute_pll_batch(model, tokenizer, sentences, completions, device,
                      max_batch=256):
    """Compute PLL for a list of (sentence, completion) pairs using MLM masking.
    
    Replicates the official BabyLM MLM scoring:
      1. Tokenize the full sentence
      2. Find completion token positions (tokens overlapping completion chars)
      3. For each completion token, mask it and get log-prob of correct token
      4. Sum log-probs = PLL
    
    Returns list of PLL values (floats), one per sentence.
    """
    mask_id = tokenizer.mask_token_id
    plls = []
    
    # Collect all (masked_tokens, attn_mask, mask_pos, target_id) tuples
    # and their item indices
    all_masked_inputs = []  # (tokens_tensor, attn_tensor, mask_pos, target_id)
    item_starts = []        # start index in all_masked_inputs for each item
    
    for sent_idx, (sent, comp) in enumerate(zip(sentences, completions)):
        item_starts.append(len(all_masked_inputs))
        
        enc = tokenizer(sent, return_offsets_mapping=True, return_attention_mask=True)
        tokens = enc["input_ids"]
        attn = enc["attention_mask"]
        offsets = enc["offset_mapping"]
        
        # Find completion token positions (same logic as official pipeline)
        start_char = len(sent) - len(comp)
        phrase_indices = []
        target_tokens = []
        for i, (s, e) in enumerate(offsets):
            if e > start_char:
                phrase_indices.append(i)
                target_tokens.append(tokens[i])
        
        for pos, tgt in zip(phrase_indices, target_tokens):
            masked = list(tokens)
            masked[pos] = mask_id
            all_masked_inputs.append((
                torch.tensor(masked, dtype=torch.long),
                torch.tensor(attn, dtype=torch.long),
                pos,
                tgt,
            ))
    
    item_starts.append(len(all_masked_inputs))  # sentinel
    
    if not all_masked_inputs:
        return [0.0] * len(sentences)
    
    # Process in batches
    all_logprobs = torch.zeros(len(all_masked_inputs), dtype=torch.float32)
    
    for batch_start in range(0, len(all_masked_inputs), max_batch):
        batch_end = min(batch_start + max_batch, len(all_masked_inputs))
        batch = all_masked_inputs[batch_start:batch_end]
        
        # Pad to same length
        max_len = max(t[0].size(0) for t in batch)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        positions = []
        targets = []
        
        for i, (tok, att, pos, tgt) in enumerate(batch):
            input_ids[i, :tok.size(0)] = tok
            attention_mask[i, :att.size(0)] = att
            positions.append(pos)
            targets.append(tgt)
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits if hasattr(output, 'logits') else output[0]
        
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        
        for i in range(len(batch)):
            pos = positions[i]
            tgt = targets[i]
            all_logprobs[batch_start + i] = log_probs[i, pos, tgt].cpu().item()
    
    # Sum per item
    for sent_idx in range(len(sentences)):
        s = item_starts[sent_idx]
        e = item_starts[sent_idx + 1]
        if s < e:
            plls.append(all_logprobs[s:e].sum().item())
        else:
            plls.append(0.0)
    
    return plls


def score_model(model_name, model_path, items, device, out_dir):
    """Score all EWoK items with one model, save per-item margins."""
    print(json.dumps({"event": "model_start", "model": model_name,
                       "device": str(device)}), flush=True)
    t0 = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.cls_token_id
    
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, trust_remote_code=True
    ).to(device)
    model.eval()
    
    # Process in chunks for memory efficiency
    CHUNK = 200
    records = []
    domain_correct = defaultdict(int)
    domain_total = defaultdict(int)
    
    for chunk_start in range(0, len(items), CHUNK):
        chunk = items[chunk_start:chunk_start + CHUNK]
        
        sentences_0 = [it["sentence_0"] for it in chunk]
        sentences_1 = [it["sentence_1"] for it in chunk]
        completions_0 = [it["completion"] for it in chunk]
        completions_1 = [it["completion"] for it in chunk]
        
        plls_0 = compute_pll_batch(model, tokenizer, sentences_0, completions_0,
                                    device, max_batch=256)
        plls_1 = compute_pll_batch(model, tokenizer, sentences_1, completions_1,
                                    device, max_batch=256)
        
        for i, it in enumerate(chunk):
            margin = plls_0[i] - plls_1[i]
            correct = 1 if margin > 0 else (0.5 if margin == 0 else 0)
            records.append({
                "domain": it["domain"],
                "line_idx": it["line_idx"],
                "ConceptA": it["ConceptA"],
                "ConceptB": it["ConceptB"],
                "ContextType": it["ContextType"],
                "ContextDiff": it["ContextDiff"],
                "TargetDiff": it["TargetDiff"],
                "PLL_0": round(plls_0[i], 6),
                "PLL_1": round(plls_1[i], 6),
                "margin": round(margin, 6),
                "is_correct": correct,
            })
            domain_correct[it["domain"]] += correct
            domain_total[it["domain"]] += 1
        
        if (chunk_start // CHUNK) % 5 == 0:
            print(json.dumps({"event": "progress", "model": model_name,
                               "items": chunk_start + len(chunk),
                               "total": len(items)}), flush=True)
    
    elapsed = time.time() - t0
    
    # Domain accuracies
    domain_acc = {}
    for d in sorted(domain_total):
        domain_acc[d] = round(100.0 * domain_correct[d] / domain_total[d], 4)
    
    overall_correct = sum(domain_correct.values())
    overall_total = sum(domain_total.values())
    
    # Macro accuracy (mean of domain accuracies) — matches official scorer
    macro_acc = round(sum(domain_acc.values()) / len(domain_acc), 4) if domain_acc else 0.0
    
    summary = {
        "model": model_name,
        "model_path": model_path,
        "total_items": len(records),
        "overall_micro_accuracy": round(100.0 * overall_correct / overall_total, 4),
        "overall_macro_accuracy": macro_acc,
        "domain_accuracy": domain_acc,
        "elapsed_sec": round(elapsed, 2),
    }
    
    # Save
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "ewok_margins.json", "w") as f:
        json.dump({"summary": summary, "items": records}, f, indent=1)
    
    # Also save compact CSV for quick analysis
    import csv
    with open(model_dir / "ewok_margins.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "domain", "line_idx", "ConceptA", "ConceptB",
            "ContextType", "ContextDiff", "TargetDiff",
            "PLL_0", "PLL_1", "margin", "is_correct",
        ])
        w.writeheader()
        w.writerows(records)
    
    print(json.dumps({"event": "model_done", "model": model_name,
                       "macro_accuracy": macro_acc,
                       "elapsed_sec": round(elapsed, 2)}), flush=True)
    
    del model
    torch.cuda.empty_cache()
    
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", type=str, required=True,
                        help="Comma-separated model names from registry")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}")
    model_names = [m.strip() for m in args.models.split(",")]
    
    # Validate
    for mn in model_names:
        if mn not in ALL_MODELS:
            print(f"ERROR: unknown model '{mn}'. Available: {list(ALL_MODELS)}")
            sys.exit(1)
    
    # Load EWoK data once
    items = load_ewok_items()
    print(json.dumps({"event": "data_loaded", "total_items": len(items),
                       "domains": len(set(it["domain"] for it in items))}),
          flush=True)
    
    if args.dry_run:
        # Validate paths, print config, exit
        for mn in model_names:
            mp = ALL_MODELS[mn]
            exists = (pathlib.Path(mp) / "model.safetensors").exists()
            print(json.dumps({"event": "dry_run", "model": mn,
                               "path": mp, "exists": exists}), flush=True)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"status": "DRY_RUN_OK",
                           "models": model_names,
                           "gpu": args.gpu,
                           "out_dir": str(OUT_ROOT)}), flush=True)
        return
    
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    
    for mn in model_names:
        s = score_model(mn, ALL_MODELS[mn], items, device, OUT_ROOT)
        summaries.append(s)
    
    # Save combined summary
    combined = {
        "status": "EWOK_MARGINS_COMPLETE",
        "gpu": args.gpu,
        "models": model_names,
        "summaries": summaries,
    }
    with open(OUT_ROOT / "ewok_margins_summary.json", "w") as f:
        json.dump(combined, f, indent=2)
    
    print(json.dumps(combined, indent=2), flush=True)


if __name__ == "__main__":
    main()
