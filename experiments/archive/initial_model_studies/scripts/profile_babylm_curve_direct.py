#!/usr/bin/env python3
"""Standalone trajectory evaluator for BabyLM local checkpoint directories.

Loads each checkpoint directory directly with AutoModelForCausalLM and
AutoTokenizer, runs official BabyLM fast/local evaluation tasks, and saves
per-revision JSON scores. Does not rely on the official evaluator's
revision_name mechanism, which has been shown to produce identical scores
for different local checkpoint subdirectories.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

STUDY = pathlib.Path("experiments/archive/initial_model_studies")
STRICT_DIR = STUDY / "repos/babylm-eval/strict"
HF_MODULES_CACHE = STUDY / "training/hf_modules_cache"


def load_task_data(data_path: pathlib.Path) -> list[dict]:
    """Load BabyLM sentence zero-shot task data from a directory of .jsonl files."""
    items: list[dict] = []
    for fp in sorted(data_path.glob("*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def compute_blimp_accuracy(model, tokenizer, task_data: list[dict], batch_size: int, max_length: int, device: torch.device) -> tuple[float, dict[str, float]]:
    """Compute sentence-pair accuracy for BLiMP-style tasks.

    Each item has 'sentence_good' and 'sentence_bad' (or 'good'/'bad').
    Returns overall accuracy and per-field accuracy.
    """
    correct = 0
    total = 0
    field_correct: dict[str, int] = {}
    field_total: dict[str, int] = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(task_data), batch_size):
            batch = task_data[i : i + batch_size]
            good_texts = [x.get("sentence_good", x.get("good", "")) for x in batch]
            bad_texts = [x.get("sentence_bad", x.get("bad", "")) for x in batch]
            field_names = [x.get("field", x.get("UID", x.get("uid", "unknown"))) for x in batch]
            # Tokenize both
            good_enc = tokenizer(good_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            bad_enc = tokenizer(bad_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            good_logits = model(**good_enc).logits
            bad_logits = model(**bad_enc).logits
            # Compute per-token log-probability of the correct next token
            # For causal LM, compute average log-prob of the sequence
            good_loss = torch.nn.functional.cross_entropy(
                good_logits[:, :-1, :].reshape(-1, good_logits.size(-1)),
                good_enc["input_ids"][:, 1:].reshape(-1),
                reduction="none",
            ).view(good_enc["input_ids"].size(0), -1).mean(dim=1)
            bad_loss = torch.nn.functional.cross_entropy(
                bad_logits[:, :-1, :].reshape(-1, bad_logits.size(-1)),
                bad_enc["input_ids"][:, 1:].reshape(-1),
                reduction="none",
            ).view(bad_enc["input_ids"].size(0), -1).mean(dim=1)
            for j in range(len(batch)):
                if good_loss[j] < bad_loss[j]:
                    correct += 1
                    field_correct[field_names[j]] = field_correct.get(field_names[j], 0) + 1
                total += 1
                field_total[field_names[j]] = field_total.get(field_names[j], 0) + 1
    field_acc = {k: round(field_correct.get(k, 0) / max(1, v) * 100, 2) for k, v in field_total.items()}
    return round(correct / max(1, total) * 100, 2), field_acc


def compute_reading_scores(model, tokenizer, task_data: list[dict], batch_size: int, max_length: int, device: torch.device) -> dict[str, float]:
    """Compute reading fast scores from BabyLM reading data.

    Each item has 'eye_tracking_score' and 'self_paced_reading_score' targets
    and 'text' or 'sentence' field. Uses simple next-token log-probability
    as a predictor and computes the Spearman-like correlation.
    """
    # Simplified: compute mean log-prob per token for each text, then report
    # the correlation with the given scores. This is a minimal proxy; the
    # full official evaluator uses a more complex regression.
    scores: dict[str, float] = {"eye_tracking": 0.0, "self_paced": 0.0}
    model.eval()
    with torch.no_grad():
        texts = [x.get("text", x.get("sentence", "")) for x in task_data]
        eye_targets = [x.get("eye_tracking_score", 0.0) for x in task_data]
        sp_targets = [x.get("self_paced_reading_score", 0.0) for x in task_data]
        log_probs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            out = model(**enc).logits
            loss = torch.nn.functional.cross_entropy(
                out[:, :-1, :].reshape(-1, out.size(-1)),
                enc["input_ids"][:, 1:].reshape(-1),
                reduction="none",
            ).view(enc["input_ids"].size(0), -1).mean(dim=1)
            log_probs.extend(-loss.cpu().tolist())
        # Compute Spearman rank correlation
        import numpy as np
        from scipy.stats import spearmanr
        if len(log_probs) > 2 and len(set(log_probs)) > 1 and len(set(eye_targets)) > 1:
            r_eye, _ = spearmanr(log_probs, eye_targets)
            scores["eye_tracking"] = round(float(r_eye) * 100, 2)
        if len(log_probs) > 2 and len(set(log_probs)) > 1 and len(set(sp_targets)) > 1:
            r_sp, _ = spearmanr(log_probs, sp_targets)
            scores["self_paced"] = round(float(r_sp) * 100, 2)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--revisions", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    HF_MODULES_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(HF_MODULES_CACHE.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load task data once
    task_data: dict[str, list[dict]] = {}
    task_paths = {
        "blimp_fast": STRICT_DIR / "evaluation_data/fast_eval/blimp_fast",
        "supplement_fast": STRICT_DIR / "evaluation_data/fast_eval/supplement_fast",
        "ewok_fast": STRICT_DIR / "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast",
        "entity_tracking_fast": STRICT_DIR / "evaluation_data/fast_eval/entity_tracking_fast",
        "comps": STRICT_DIR / "evaluation_data/full_eval/comps",
        "reading": STRICT_DIR / "evaluation_data/fast_eval/reading",
    }
    for name, path in task_paths.items():
        if path.exists():
            task_data[name] = load_task_data(path)
            print(f"Loaded {name}: {len(task_data[name])} items from {path}", flush=True)
        else:
            print(f"WARNING: {path} does not exist, skipping {name}", flush=True)

    results: list[dict[str, Any]] = []
    for rev in args.revisions:
        ckpt_path = run_dir / "hf_model" / rev
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint directory missing: {ckpt_path}")
        print(f"\n=== Loading {rev} from {ckpt_path} ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True).to(device)
        model.eval()
        scores: dict[str, Any] = {}

        for task_name in ["blimp_fast", "supplement_fast", "ewok_fast", "entity_tracking_fast", "comps"]:
            if task_name not in task_data:
                scores[task_name] = None
                continue
            acc, fields = compute_blimp_accuracy(model, tokenizer, task_data[task_name], args.batch_size, args.max_length, device)
            scores[task_name] = acc
            print(f"  {rev} {task_name}: {acc}", flush=True)

        if "reading" in task_data:
            reading_scores = compute_reading_scores(model, tokenizer, task_data["reading"], args.batch_size, args.max_length, device)
            scores["reading_eye_tracking"] = reading_scores["eye_tracking"]
            scores["reading_self_paced"] = reading_scores["self_paced"]
            print(f"  {rev} reading: eye={reading_scores['eye_tracking']}, sp={reading_scores['self_paced']}", flush=True)

        results.append({"revision": rev, "checkpoint_path": str(ckpt_path), "scores": scores})

    out_path = pathlib.Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWROTE {out_path}", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()