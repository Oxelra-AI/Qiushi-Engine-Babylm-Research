#!/usr/bin/env python3
"""research: direct local Qwen generation for structured state-update/use packets."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = pathlib.Path.cwd()
DATA_DIR = ROOT / "experiments/archive/relation_learning/data/state_use_generation"
PROMPTS_PATH = DATA_DIR / "state_use_prompts.jsonl"
MODEL_MAP = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3-1.7b", choices=sorted(MODEL_MAP))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.45)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-suffix", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prompts: list[dict] = []
    with PROMPTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    if args.pilot:
        prompts = prompts[: args.pilot]
    n_total = len(prompts)
    model_id = MODEL_MAP[args.model]
    suffix = args.output_suffix or (f"_pilot{args.pilot}" if args.pilot else "")
    out_path = DATA_DIR / f"raw_state_use_{args.model.replace('.', 'p')}{suffix}.jsonl"
    meta_path = DATA_DIR / f"generation_metadata_{args.model.replace('.', 'p')}{suffix}.json"

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Loading {model_id}...", flush=True)
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
        device_map=device if device.startswith("cuda") else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    print(f"Loaded in {time.time()-t_load:.1f}s", flush=True)
    print(f"Prompts: {n_total}", flush=True)

    t_gen = time.time()
    n_done = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for start in range(0, n_total, args.batch_size):
            batch = prompts[start:start+args.batch_size]
            texts = []
            for p in batch:
                msgs = [{"role": "user", "content": p["prompt"]}]
                texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=768)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p,
                    pad_token_id=tok.pad_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            for p, out in zip(batch, outputs):
                gen = tok.decode(out[input_len:], skip_special_tokens=True).strip()
                # Keep the first JSON-looking line/block if possible; validation will parse/repair.
                fout.write(json.dumps({
                    "id": p["id"],
                    "pair_id": p["pair_id"],
                    "packet_type": p["packet_type"],
                    "candidate_entities": p.get("candidate_entities", []),
                    "source_sentence": p["source_sentence"],
                    "source_words": p["source_words"],
                    "qwen_rewrite": p.get("qwen_rewrite", ""),
                    "rewrite_words": p.get("rewrite_words", 0),
                    "source": p.get("source"),
                    "example_id": p.get("example_id"),
                    "cohort": p.get("cohort"),
                    "output": gen,
                    "generated_words": len(gen.split()),
                }, ensure_ascii=False) + "\n")
            n_done += len(batch)
            elapsed = max(time.time() - t_gen, 0.01)
            if n_done % (args.batch_size * 5) == 0 or n_done >= n_total:
                rate = n_done / elapsed
                eta = (n_total - n_done) / max(rate, 1e-9)
                print(f"  {n_done}/{n_total} ({100*n_done/n_total:.1f}%) {rate:.1f}/s ETA {eta:.0f}s", flush=True)
    elapsed = time.time() - t_gen
    meta = {
        "status": "COMPLETE",
        "model": model_id,
        "model_alias": args.model,
        "n_prompts": n_total,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": device,
        "output_path": str(out_path),
        "elapsed_sec": round(elapsed, 1),
        "rate": round(n_total / max(elapsed, 0.01), 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
