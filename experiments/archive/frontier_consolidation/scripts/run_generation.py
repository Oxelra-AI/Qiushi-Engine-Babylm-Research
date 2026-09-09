#!/usr/bin/env python3
"""research: Run concise faithful view generation with Qwen3-4B.

Custom script to avoid training generate directory conflict.
Loads model directly, generates outputs for the test prompts,
saves results as JSONL.
"""
from __future__ import annotations
import json
import time
import os
import sys
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    t0 = time.time()
    
    # Load prompts
    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    print(json.dumps({"event": "prompts_loaded", "n": len(prompts)}), flush=True)
    
    # Load model
    print(json.dumps({"event": "loading_model", "model_id": args.model_id}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(json.dumps({"event": "model_loaded", "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
    
    # Ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Generate in batches
    results = []
    for batch_start in range(0, len(prompts), args.batch_size):
        batch = prompts[batch_start:batch_start + args.batch_size]
        batch_texts = [p["prompt"] for p in batch]
        
        # Tokenize with left padding for generation
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                do_sample=args.temperature > 0,
                top_p=0.95 if args.temperature > 0 else 1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode only new tokens
        for i, (prompt_rec, output_ids) in enumerate(zip(batch, outputs)):
            input_len = inputs.input_ids.shape[1]
            new_tokens = output_ids[input_len:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clean up: take only first complete sentence/line
            lines = generated.split("\n")
            generated_clean = lines[0].strip() if lines else generated
            
            results.append({
                "id": prompt_rec["id"],
                "output": generated_clean,
                "output_raw": generated,
            })
        
        if (batch_start // args.batch_size) % 5 == 0:
            print(json.dumps({
                "event": "batch_done", 
                "batch": batch_start // args.batch_size,
                "total_done": len(results),
                "elapsed_sec": round(time.time() - t0, 1),
            }), flush=True)
    
    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(json.dumps({
        "event": "done",
        "n_generated": len(results),
        "output_path": str(out_path),
        "elapsed_sec": round(time.time() - t0, 1),
    }), flush=True)


if __name__ == "__main__":
    main()
