#!/usr/bin/env python3
"""research: Generate entity-state-update companions using Qwen3-8B directly.

Uses direct generation with a distinct output directory.
Uses local_files_only=True to read from pre-cached model.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = pathlib.Path.cwd()
DATA_DIR = ROOT / "experiments/archive/relation_learning/data/state_update_generation"
PROMPTS_PATH = DATA_DIR / "state_update_prompts.jsonl"
OUT_PATH = DATA_DIR / "raw_state_updates_qwen3_1p7b.jsonl"
META_PATH = DATA_DIR / "generation_metadata_qwen3_1p7b.json"

MODEL_ID = "Qwen/Qwen3-1.7B"
BATCH_SIZE = 128
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
TOP_P = 0.9


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Loading {MODEL_ID}...", flush=True)
    t_load = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True, local_files_only=True,
        device_map=device if device == "cuda" else None,
    )
    model.eval()
    print(f"Loaded in {time.time()-t_load:.1f}s", flush=True)

    # Pad token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"  # for batch generation

    # Read prompts
    prompts = []
    with PROMPTS_PATH.open() as f:
        for line in f:
            prompts.append(json.loads(line))
    n_total = len(prompts)
    print(f"Prompts: {n_total}", flush=True)

    t_gen = time.time()
    n_done = 0

    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for batch_start in range(0, n_total, BATCH_SIZE):
            batch = prompts[batch_start:batch_start + BATCH_SIZE]

            # Format as chat
            batch_texts = []
            for p in batch:
                msgs = [{"role": "user", "content": p["prompt"]}]
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
                batch_texts.append(text)

            inputs = tok(batch_texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE, top_p=TOP_P, do_sample=True,
                    pad_token_id=tok.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for p, out in zip(batch, outputs):
                gen = tok.decode(out[input_len:], skip_special_tokens=True).strip()
                gen = gen.split("\n")[0].strip()
                fout.write(json.dumps({
                    "id": p["id"],
                    "pair_id": p["pair_id"],
                    "source_sentence": p["source_sentence"],
                    "output": gen,
                    "source_words": p["source_words"],
                    "generated_words": len(gen.split()),
                }, ensure_ascii=False) + "\n")

            n_done += len(batch)
            elapsed = time.time() - t_gen
            if n_done % (BATCH_SIZE * 5) == 0 or n_done >= n_total:
                print(f"  {n_done}/{n_total} ({100*n_done/n_total:.1f}%) "
                      f"{n_done/elapsed:.1f}/s ETA {(n_total-n_done)/(n_done/elapsed):.0f}s",
                      flush=True)

    total_time = time.time() - t_gen
    meta = {
        "status": "COMPLETE",
        "model": MODEL_ID,
        "n_prompts": n_total,
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "device": device,
        "output_path": str(OUT_PATH),
        "elapsed_sec": round(total_time, 1),
        "rate": round(n_total / max(total_time, 0.01), 2),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
