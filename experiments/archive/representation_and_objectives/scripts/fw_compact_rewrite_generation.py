#!/usr/bin/env python3
"""research: Generate compact faithful rewrites for FineWeb sources.

Uses Qwen3.5-9B (approved BabyLM 2026 teacher) to generate compact
rewrites for sources that don't have existing compact rewrites.
Validates entity/number/content retention. Saves validated results.

Waits for a GPU with >=20GB free before starting.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import gc
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import time
from typing import Any

_SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
_WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
ROOT = _WORKSPACE
PROMPT_PATH = _public_path('experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_compact_prompts.jsonl')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/fw_compact_rewrite_generation')

# Use exact local snapshot paths to avoid read-only HF cache locks
MODEL_PREFERENCES = [
    "data/external/1cfa9a7208912126459214e8b04321603b3df60c",
]

BATCH_SIZE = 32
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.1
MIN_GPU_FREE_MB = 12000  # 12GB free for Qwen3-4B + overhead
GPU_WAIT_TIMEOUT = 3600  # 1 hour max wait


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def wc(text: str) -> int:
    return len((text or "").split())


def content_recall(source: str, rewrite: str) -> float:
    """Word-level content recall."""
    sw = set(source.lower().split())
    rw = set(rewrite.lower().split())
    if not sw:
        return 1.0
    return len(sw & rw) / len(sw)


def entity_recall(source_text: str, rewrite: str, 
                  entities: list[str] = None) -> float:
    """Check if named entities from source appear in rewrite."""
    if not entities:
        # Extract capitalised multi-word phrases as proxy
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', source_text)
        entities = [c for c in caps if len(c) > 2]
    if not entities:
        return 1.0
    found = sum(1 for e in entities if e.lower() in rewrite.lower())
    return found / len(entities)


def number_recall(source_text: str, rewrite: str) -> float:
    """Check if numbers from source appear in rewrite."""
    src_nums = set(re.findall(r'\b\d[\d,.]*\b', source_text))
    if not src_nums:
        return 1.0
    rw_nums = set(re.findall(r'\b\d[\d,.]*\b', rewrite))
    found = sum(1 for n in src_nums if n in rw_nums)
    return found / len(src_nums)


def wait_for_gpu(min_free_mb: int = MIN_GPU_FREE_MB,
                 timeout: int = GPU_WAIT_TIMEOUT) -> int:
    """Wait until a GPU with enough free memory is available."""
    import subprocess
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = line.split(",")
                    if len(parts) == 2:
                        idx, free = int(parts[0].strip()), int(parts[1].strip())
                        if free >= min_free_mb:
                            print(json.dumps({
                                "event": "gpu_found",
                                "gpu": idx,
                                "free_mb": free,
                                "waited_sec": round(time.time() - t0, 1),
                            }), flush=True)
                            return idx
        except Exception as e:
            print(json.dumps({"event": "gpu_check_error", "error": str(e)}),
                  flush=True)
        time.sleep(30)
    raise RuntimeError(f"No GPU with >={min_free_mb}MB free after {timeout}s")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print(json.dumps({"event": "loading_prompts", "path": str(PROMPT_PATH)}),
          flush=True)
    prompts = read_jsonl(PROMPT_PATH)
    print(json.dumps({"event": "prompts_loaded", "n": len(prompts)}), flush=True)

    if not prompts:
        print(json.dumps({"event": "no_prompts", "status": "done"}), flush=True)
        return

    # Set writable HF cache for any lock files
    import tempfile
    _tmp_hf = tempfile.mkdtemp(prefix="hf_cache_")
    os.environ["HF_HOME"] = _tmp_hf
    os.environ["HF_HUB_CACHE"] = _tmp_hf
    os.environ["TRANSFORMERS_CACHE"] = _tmp_hf

    # Wait for GPU
    print(json.dumps({"event": "waiting_for_gpu",
                       "min_free_mb": MIN_GPU_FREE_MB}), flush=True)
    gpu_idx = wait_for_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    # Import torch after setting CUDA device
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Try models in preference order
    model = None
    tokenizer = None
    model_id = None

    for mid in MODEL_PREFERENCES:
        try:
            print(json.dumps({"event": "trying_model", "model_id": mid}),
                  flush=True)
            tokenizer = AutoTokenizer.from_pretrained(
                mid, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True)
            model.eval()
            model_id = mid
            print(json.dumps({"event": "model_loaded", "model_id": mid,
                               "elapsed_sec": round(time.time() - t0, 1)}),
                  flush=True)
            break
        except Exception as e:
            print(json.dumps({"event": "model_failed", "model_id": mid,
                               "error": str(e)[:200]}), flush=True)
            model = None
            tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if model is None:
        raise RuntimeError(f"Could not load any model: {MODEL_PREFERENCES}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Generate in batches
    results = []
    n_accepted = 0
    n_rejected = 0
    tokenizer.padding_side = "left"

    for batch_start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[batch_start:batch_start + BATCH_SIZE]

        # Build chat messages for each prompt
        batch_texts = []
        for p in batch:
            if hasattr(tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": p.get("system", "")},
                    {"role": "user", "content": p["prompt"]},
                ]
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    text = p["prompt"]
            else:
                text = p["prompt"]
            batch_texts.append(text)

        try:
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE if TEMPERATURE > 0 else 1.0,
                    do_sample=TEMPERATURE > 0,
                    top_p=0.95 if TEMPERATURE > 0 else 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for i, (prompt_rec, output_ids) in enumerate(zip(batch, outputs)):
                input_len = inputs.input_ids.shape[1]
                new_tokens = output_ids[input_len:]
                generated = tokenizer.decode(
                    new_tokens, skip_special_tokens=True).strip()

                # Clean: first line only
                lines = generated.split("\n")
                gen_clean = lines[0].strip() if lines else generated
                # Remove trailing incomplete sentence fragments
                if gen_clean and not gen_clean[-1] in '.!?':
                    # Try to find last complete sentence
                    for end in ['. ', '! ', '? ']:
                        idx = gen_clean.rfind(end)
                        if idx > 0:
                            gen_clean = gen_clean[:idx + 1]
                            break

                source_text = prompt_rec.get("source_text", "")
                rw_words = wc(gen_clean)
                src_words = prompt_rec.get("source_words", wc(source_text))

                # Validate
                cr = content_recall(source_text, gen_clean)
                er = entity_recall(source_text, gen_clean)
                nr = number_recall(source_text, gen_clean)
                ratio = rw_words / max(1, src_words)

                accepted = (
                    rw_words >= 5 and
                    rw_words <= src_words * 1.1 and
                    cr >= 0.35 and
                    er >= 0.5 and
                    nr >= 0.5 and
                    gen_clean.lower() != source_text.lower()
                )

                rec = {
                    "prompt_id": prompt_rec.get("prompt_id", ""),
                    "norm_hash": prompt_rec.get("norm_hash", ""),
                    "source_text": source_text,
                    "rewrite_text": gen_clean,
                    "source_words": src_words,
                    "rewrite_words": rw_words,
                    "compression_ratio": round(ratio, 4),
                    "content_recall": round(cr, 4),
                    "entity_recall": round(er, 4),
                    "number_recall": round(nr, 4),
                    "accepted": accepted,
                    "doc_id": prompt_rec.get("doc_id", ""),
                    "domains": prompt_rec.get("domains", []),
                    "model_id": model_id,
                }
                results.append(rec)
                if accepted:
                    n_accepted += 1
                else:
                    n_rejected += 1

        except Exception as e:
            print(json.dumps({"event": "batch_error",
                               "batch_start": batch_start,
                               "error": str(e)[:300]}), flush=True)
            # Add failed entries
            for p in batch:
                results.append({
                    "prompt_id": p.get("prompt_id", ""),
                    "norm_hash": p.get("norm_hash", ""),
                    "source_text": p.get("source_text", ""),
                    "rewrite_text": "",
                    "accepted": False,
                    "error": str(e)[:200],
                })
                n_rejected += 1

        if (batch_start // BATCH_SIZE) % 20 == 0:
            print(json.dumps({
                "event": "progress",
                "done": len(results),
                "total": len(prompts),
                "accepted": n_accepted,
                "rejected": n_rejected,
                "elapsed_sec": round(time.time() - t0, 1),
            }), flush=True)

    # Save all results
    all_path = _public_path('experiments/archive/representation_and_objectives/data/fw_compact_rewrite_generation/compact_rewrites_all.jsonl')
    with all_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save accepted only
    accepted_path = _public_path('experiments/archive/representation_and_objectives/data/fw_compact_rewrite_generation/compact_rewrites_accepted.jsonl')
    with accepted_path.open("w", encoding="utf-8") as f:
        for r in results:
            if r.get("accepted"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute stats
    acc_ratios = [r["compression_ratio"] for r in results if r.get("accepted")]
    acc_cr = [r["content_recall"] for r in results if r.get("accepted")]
    acc_er = [r["entity_recall"] for r in results if r.get("accepted")]
    acc_nr = [r["number_recall"] for r in results if r.get("accepted")]
    acc_rw = [r["rewrite_words"] for r in results if r.get("accepted")]

    def s(vals):
        if not vals:
            return {}
        import statistics as st
        return {"n": len(vals), "mean": round(st.fmean(vals), 4),
                "median": round(st.median(vals), 1),
                "min": min(vals), "max": max(vals)}

    manifest = {
        "status": "FW_COMPACT_REWRITE_GENERATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "gpu": gpu_idx,
        "n_prompts": len(prompts),
        "n_generated": len(results),
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "acceptance_rate": round(n_accepted / max(1, len(results)), 4),
        "accepted_rewrite_words_total": sum(acc_rw),
        "accepted_stats": {
            "compression_ratio": s(acc_ratios),
            "content_recall": s(acc_cr),
            "entity_recall": s(acc_er),
            "number_recall": s(acc_nr),
            "rewrite_words": s(acc_rw),
        },
        "outputs": {
            "all": str(all_path),
            "accepted": str(accepted_path),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }

    manifest_path = _public_path('experiments/archive/representation_and_objectives/data/fw_compact_rewrite_generation/compact_rewrite_generation_manifest.json')
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    print(json.dumps({
        "status": manifest["status"],
        "model_id": model_id,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "acceptance_rate": manifest["acceptance_rate"],
        "accepted_rewrite_words": sum(acc_rw),
        "elapsed_sec": manifest["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
