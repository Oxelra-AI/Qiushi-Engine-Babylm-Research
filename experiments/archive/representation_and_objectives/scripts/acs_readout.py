#!/usr/bin/env python3
"""research: Fast EWoK + GlobalPIQA readout for ACS mechanism screen.

Evaluates one DeBERTa-v2 checkpoint on:
  - Full EWoK four-cell interaction (7,618 rows)
  - GlobalPIQA parallel (103 rows) with hard52 breakdown
Produces a summary JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DebertaV2ForMaskedLM, AutoTokenizer

USER_ROOT = Path(".").resolve()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
EVAL_EWOK = A01_WS / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
GP_PARALLEL = A01_WS / "data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval/global_piqa_parallel/eng_latn.jsonl"


def load_ewok_rows() -> list[dict]:
    rows = []
    for path in sorted(EVAL_EWOK.glob("*.jsonl")):
        domain = path.stem
        with path.open("r", encoding="utf-8") as f:
            for local_idx, line in enumerate(f):
                raw = json.loads(line)
                raw["_domain"] = domain
                raw["_local_index"] = local_idx
                rows.append(raw)
    return rows


def load_globalpiqa_parallel() -> list[dict]:
    rows = []
    with GP_PARALLEL.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def pseudo_log_prob(model, tokenizer, context: str, target: str, device, max_len=256) -> float:
    """Compute pseudo-log-probability of target given context using masked LM."""
    text = f"{context} {target}"
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len,
                        padding=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    tgt_ids = tokenizer.encode(f" {target}", add_special_tokens=False)
    if not tgt_ids:
        tgt_ids = tokenizer.encode(target, add_special_tokens=False)
    if not tgt_ids:
        return float("-inf")
    seq = input_ids[0].tolist()
    tgt_start = None
    for i in range(len(seq) - len(tgt_ids), -1, -1):
        if seq[i:i+len(tgt_ids)] == tgt_ids:
            tgt_start = i
            break
    if tgt_start is None:
        return float("-inf")
    total_lp = 0.0
    for offset in range(len(tgt_ids)):
        masked = input_ids.clone()
        masked[0, tgt_start + offset] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(input_ids=masked, attention_mask=attention_mask).logits
        lp = F.log_softmax(logits[0, tgt_start + offset], dim=-1)
        total_lp += lp[tgt_ids[offset]].item()
    return total_lp


def score_ewok_row(model, tokenizer, row, device):
    c1, c2 = row["Context1"], row["Context2"]
    t1, t2 = row["Target1"], row["Target2"]
    s11 = pseudo_log_prob(model, tokenizer, c1, t1, device)
    s12 = pseudo_log_prob(model, tokenizer, c1, t2, device)
    s21 = pseudo_log_prob(model, tokenizer, c2, t1, device)
    s22 = pseudo_log_prob(model, tokenizer, c2, t2, device)
    interaction = (s11 - s12) - (s21 - s22)
    both_correct = (s11 > s12) and (s22 > s21)
    stable_reversal = (s12 > s11) and (s21 > s22)
    return {
        "s11": s11, "s12": s12, "s21": s21, "s22": s22,
        "interaction": interaction,
        "both_correct": both_correct,
        "stable_reversal": stable_reversal,
        "domain": row["_domain"],
    }


def score_globalpiqa_row(model, tokenizer, row, device):
    prompt = row["prompt"]
    choices = [row[f"solution{i}"] for i in range(4)]
    correct_idx = int(row["label"])
    log_probs = []
    for ch in choices:
        lp = pseudo_log_prob(model, tokenizer, prompt, ch, device)
        log_probs.append(lp)
    sorted_idx = sorted(range(4), key=lambda i: -log_probs[i])
    correct_rank = sorted_idx.index(correct_idx) + 1
    top_lp = max(log_probs)
    correct_lp = log_probs[correct_idx]
    top_minus_correct = top_lp - correct_lp if correct_lp > float("-inf") else float("inf")
    return {
        "example_id": row.get("example_id", ""),
        "correct_idx": correct_idx,
        "correct_rank": correct_rank,
        "correct": correct_rank == 1,
        "top_minus_correct": top_minus_correct,
        "log_probs": log_probs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_ewok", action="store_true")
    p.add_argument("--skip_globalpiqa", action="store_true")
    p.add_argument("--ewok_log_every", type=int, default=500)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    device = torch.device(args.device)

    print(json.dumps({"event": "loading", "model": args.model_path}), flush=True)
    model_path = Path(args.model_path)
    custom_py = list(model_path.glob("*.py"))
    if custom_py:
        sys.path.insert(0, str(model_path))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(A01_WS / "cache/hf"))
    os.environ.setdefault("HF_HOME", str(A01_WS / "cache/hf"))
    model = DebertaV2ForMaskedLM.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    print(json.dumps({"event": "loaded", "params": sum(p.numel() for p in model.parameters())}), flush=True)

    summary: dict = {"label": args.label, "model_path": str(model_path),
                     "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ── EWoK ──
    if not args.skip_ewok:
        ewok_rows = load_ewok_rows()
        print(json.dumps({"event": "ewok_start", "rows": len(ewok_rows)}), flush=True)
        both_correct_count = 0
        stable_rev_count = 0
        interactions = []
        ewok_row_results = []
        for i, row in enumerate(ewok_rows):
            r = score_ewok_row(model, tokenizer, row, device)
            ewok_row_results.append(r)
            if r["both_correct"]:
                both_correct_count += 1
            if r["stable_reversal"]:
                stable_rev_count += 1
            interactions.append(r["interaction"])
            if (i+1) % args.ewok_log_every == 0:
                print(json.dumps({"event": "ewok_progress", "done": i+1, "total": len(ewok_rows),
                                   "acc": round(both_correct_count/(i+1), 4),
                                   "stable_rev": stable_rev_count}), flush=True)

        n = len(ewok_rows)
        summary["ewok"] = {
            "n_rows": n,
            "accuracy": round(both_correct_count / n, 6),
            "both_correct": both_correct_count,
            "stable_reversals": stable_rev_count,
            "stable_reversal_frac": round(stable_rev_count / n, 6),
            "mean_interaction": round(sum(interactions)/n, 6),
        }
        print(json.dumps({"event": "ewok_done", **summary["ewok"]}), flush=True)
        ewok_path = out / f"{args.label}_ewok_rows.jsonl"
        with ewok_path.open("w") as f:
            for r in ewok_row_results:
                f.write(json.dumps(r) + "\n")

    # ── GlobalPIQA ──
    if not args.skip_globalpiqa:
        gp_rows = load_globalpiqa_parallel()
        print(json.dumps({"event": "globalpiqa_start", "rows": len(gp_rows)}), flush=True)
        gp_results = []
        for row in gp_rows:
            r = score_globalpiqa_row(model, tokenizer, row, device)
            gp_results.append(r)

        n = len(gp_results)
        correct_count = sum(1 for r in gp_results if r["correct"])
        gp_sorted = sorted(gp_results, key=lambda r: r["top_minus_correct"], reverse=True)
        hard52 = gp_sorted[:52]
        hard52_correct = sum(1 for r in hard52 if r["correct"])
        hard52_margins = [r["top_minus_correct"] for r in hard52 if r["top_minus_correct"] < float("inf")]

        summary["globalpiqa"] = {
            "n_parallel": n,
            "parallel_accuracy_pct": round(correct_count / n * 100, 4),
            "parallel_correct": correct_count,
            "hard52_correct": hard52_correct,
            "hard52_accuracy_pct": round(hard52_correct / 52 * 100, 4),
            "hard52_mean_margin": round(sum(hard52_margins)/len(hard52_margins), 6) if hard52_margins else 0,
        }
        print(json.dumps({"event": "globalpiqa_done", **summary["globalpiqa"]}), flush=True)
        gp_path = out / f"{args.label}_globalpiqa_rows.jsonl"
        with gp_path.open("w") as f:
            for r in gp_results:
                f.write(json.dumps(r) + "\n")

    summary["elapsed_sec"] = round(time.time() - start, 1)
    summary_path = out / f"{args.label}_readout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "READOUT_DONE", "label": args.label,
                       "summary": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
