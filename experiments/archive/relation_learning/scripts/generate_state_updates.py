#!/usr/bin/env python3
"""research: Generate entity-state-update companions using a local LLM.

Reads state_update_prompts.jsonl and generates companions using Qwen3-8B (or smaller
for pilot). Outputs raw generations for subsequent validation.

Usage:
    python generate_state_updates.py --model qwen3-8b [--pilot N] [--batch-size B]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path.cwd()
DATA_DIR = ROOT / "experiments/archive/relation_learning/data/state_update_generation"
PROMPTS_PATH = DATA_DIR / "state_update_prompts.jsonl"

MODEL_MAP = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b", choices=list(MODEL_MAP))
    ap.add_argument("--pilot", type=int, default=0, help="Generate only first N prompts for pilot")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="auto", help="cuda:0, cuda:1, cpu, or auto")
    args = ap.parse_args()

    model_id = MODEL_MAP[args.model]
    
    # Read prompts
    prompts = []
    with PROMPTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    
    if args.pilot > 0:
        prompts = prompts[:args.pilot]
    
    n_total = len(prompts)
    print(f"Loaded {n_total} prompts, model={model_id}", flush=True)
    
    # Import torch and transformers
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model on {device}...", flush=True)
    t_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    dtype = torch.float16 if "cuda" in str(device) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    
    print(f"Model loaded in {time.time()-t_load:.1f}s", flush=True)
    
    # Determine output path
    suffix = f"_pilot{args.pilot}" if args.pilot > 0 else ""
    out_path = DATA_DIR / f"raw_state_updates_{args.model}{suffix}.jsonl"
    
    # Generate in batches
    t_gen = time.time()
    n_done = 0
    
    # Use chat template if available, otherwise raw
    use_chat = hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None
    
    with out_path.open("w", encoding="utf-8") as fout:
        for batch_start in range(0, n_total, args.batch_size):
            batch = prompts[batch_start:batch_start + args.batch_size]
            
            if use_chat:
                # Format as chat messages
                batch_texts = []
                for p in batch:
                    messages = [{"role": "user", "content": p["prompt"]}]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    batch_texts.append(text)
            else:
                batch_texts = [p["prompt"] for p in batch]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            
            # Decode only new tokens
            input_len = inputs["input_ids"].shape[1]
            for i, (p, out) in enumerate(zip(batch, outputs)):
                new_tokens = out[input_len:]
                generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                # Take only first sentence/line
                generated = generated.split("\n")[0].strip()
                
                rec = {
                    "id": p["id"],
                    "pair_id": p["pair_id"],
                    "source_sentence": p["source_sentence"],
                    "generated": generated,
                    "source_words": p["source_words"],
                    "generated_words": len(generated.split()),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            n_done += len(batch)
            elapsed = time.time() - t_gen
            rate = n_done / elapsed if elapsed > 0 else 0
            if n_done % (args.batch_size * 10) == 0 or n_done >= n_total:
                print(f"  {n_done}/{n_total} ({100*n_done/n_total:.1f}%) "
                      f"rate={rate:.1f} prompts/s", flush=True)
    
    elapsed_total = time.time() - t_gen
    
    metadata = {
        "status": "GENERATION_COMPLETE",
        "model": model_id,
        "model_alias": args.model,
        "n_prompts": n_total,
        "pilot": args.pilot,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": str(device),
        "output_path": str(out_path),
        "elapsed_sec": round(elapsed_total, 1),
        "rate_prompts_per_sec": round(n_total / max(elapsed_total, 0.1), 2),
    }
    
    meta_path = DATA_DIR / f"generation_metadata_{args.model}{suffix}.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
