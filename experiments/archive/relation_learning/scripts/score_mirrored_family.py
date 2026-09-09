#!/usr/bin/env python3
"""research: Base readout scorer for mirrored four-row binding family.

Scores all candidate values using pseudo-log-likelihood. Reports per-role
accuracy, per-k breakdown, pair-level joint success, and margins.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json, os, sys, time, argparse
from pathlib import Path
from collections import defaultdict

# Set up cache BEFORE importing transformers
_ROOT = _public_path('.')
CACHE_BASE = str(_public_path('experiments/archive/relation_learning/data/base_readout/hf_cache'))
os.makedirs(CACHE_BASE, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE_BASE
os.environ["HF_HOME"] = CACHE_BASE

import torch
import torch.nn.functional as F

CHCK82_DIR = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
ROWS_DEFAULT = _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family/mirrored_heldout_frame_all.jsonl')
OUT_DEFAULT = _public_path('experiments/archive/relation_learning/data/base_readout')


def load_model(model_dir: str, device: torch.device):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True,
                                               trust_remote_code=True,
                                               local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True,
                                                  local_files_only=True)
    model.eval()
    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_dir}: {n:,} params on {device}", flush=True)
    return model, tokenizer


def score_candidates(model, tokenizer, prefix_text: str, candidates: dict[str, str],
                     device: torch.device, max_length: int = 512) -> dict:
    """Score each candidate value via pseudo-log-likelihood at the answer slot.
    candidates: {value_label: value_string}
    Returns: {value_label: {"logprob": float, "n_tokens": int, "logprob_per_token": float}}
    """
    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 0

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    results = {}

    for label, value_str in candidates.items():
        cand_ids = tokenizer.encode(value_str, add_special_tokens=False)
        if not cand_ids:
            continue

        # Build: [CLS] ...prefix... cand [SEP]
        avail = max_length - len(cand_ids) - 2
        if avail < 1:
            continue
        trunc_prefix = prefix_ids[-avail:] if len(prefix_ids) > avail else prefix_ids
        base = [cls_id] + trunc_prefix + cand_ids + [sep_id]
        cand_start = len(base) - len(cand_ids) - 1

        # Batch all masked positions for this candidate
        batch = []
        for i in range(len(cand_ids)):
            masked = list(base)
            masked[cand_start + i] = mask_id
            batch.append(masked)

        input_ids = torch.tensor(batch, device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids)

        total_lp = 0.0
        for i in range(len(cand_ids)):
            logits = out.logits[i, cand_start + i]
            lp = F.log_softmax(logits, dim=-1)
            total_lp += lp[cand_ids[i]].item()

        results[label] = {
            "logprob": total_lp,
            "n_tokens": len(cand_ids),
            "logprob_per_token": total_lp / len(cand_ids),
        }

    return results


def score_row(model, tokenizer, row: dict, device: torch.device):
    """Score one row: all candidate values at the answer position."""
    prefix = row["context_text"] + " " + row["query_prefix"]
    # Candidates: correct answer + all present values
    candidates = {}
    for v in row["all_present_values"]:
        candidates[v] = v

    scores = score_candidates(model, tokenizer, prefix, candidates, device)
    if not scores:
        return None

    correct = row["answer"]
    # Top by raw logprob
    top_raw = max(scores, key=lambda c: scores[c]["logprob"])
    # Top by per-token normalized logprob
    top_norm = max(scores, key=lambda c: scores[c]["logprob_per_token"])

    correct_lp = scores.get(correct, {}).get("logprob", float("-inf"))
    # Margin vs strongest competitor
    comps = {c: s for c, s in scores.items() if c != correct}
    if comps:
        top_comp = max(comps, key=lambda c: comps[c]["logprob"])
        margin_raw = correct_lp - comps[top_comp]["logprob"]
    else:
        top_comp = None
        margin_raw = float("inf")

    return {
        "row_id": row["row_id"], "quad_id": row["quad_id"],
        "map_id": row["map_id"], "k": row["k"],
        "frame_id": row["frame_id"], "answer_role": row["answer_role"],
        "context_variant": row["context_variant"],
        "entity_queried": row["entity_queried"],
        "correct_answer": correct,
        "top_raw": top_raw, "top_norm": top_norm,
        "correct_raw": top_raw == correct,
        "correct_norm": top_norm == correct,
        "margin_raw": margin_raw,
        "correct_logprob": correct_lp,
        "top_competitor": top_comp,
        "n_candidates": len(scores),
    }


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    n_cr = sum(r["correct_raw"] for r in results)
    n_cn = sum(r["correct_norm"] for r in results)

    by_role = defaultdict(lambda: {"n": 0, "cr": 0, "cn": 0, "margins": []})
    for r in results:
        role = r["answer_role"]
        by_role[role]["n"] += 1
        by_role[role]["cr"] += int(r["correct_raw"])
        by_role[role]["cn"] += int(r["correct_norm"])
        by_role[role]["margins"].append(r["margin_raw"])

    by_k = defaultdict(lambda: {"n": 0, "cr": 0, "upd_n": 0, "upd_cr": 0,
                                 "ret_n": 0, "ret_cr": 0})
    for r in results:
        k = r["k"]
        by_k[k]["n"] += 1
        by_k[k]["cr"] += int(r["correct_raw"])
        if r["answer_role"] == "updated":
            by_k[k]["upd_n"] += 1
            by_k[k]["upd_cr"] += int(r["correct_raw"])
        else:
            by_k[k]["ret_n"] += 1
            by_k[k]["ret_cr"] += int(r["correct_raw"])

    # Shared-context pair analysis
    pair_key = defaultdict(dict)  # (quad_id, cv) -> {role: correct}
    for r in results:
        pair_key[(r["quad_id"], r["context_variant"])][r["answer_role"]] = r["correct_raw"]
    pairs = [v for v in pair_key.values() if "updated" in v and "retained" in v]
    pair_both = sum(1 for p in pairs if p["updated"] and p["retained"])
    pair_none = sum(1 for p in pairs if not p["updated"] and not p["retained"])

    # Full quad analysis
    quad_rows = defaultdict(list)
    for r in results:
        quad_rows[r["quad_id"]].append(r["correct_raw"])
    quads = [(qid, vs) for qid, vs in quad_rows.items() if len(vs) == 4]
    quad_all = sum(1 for _, vs in quads if all(vs))

    summary = {
        "rows": n,
        "accuracy_raw": n_cr / n if n else 0,
        "accuracy_norm": n_cn / n if n else 0,
        "by_role": {},
        "by_k": {},
        "shared_context_pairs": {
            "total": len(pairs), "both_correct": pair_both,
            "both_wrong": pair_none,
            "joint_rate": pair_both / len(pairs) if pairs else 0,
        },
        "full_quads": {
            "total": len(quads), "all_correct": quad_all,
            "rate": quad_all / len(quads) if quads else 0,
        },
    }
    for role, s in by_role.items():
        m = s["margins"]
        summary["by_role"][role] = {
            "n": s["n"], "accuracy_raw": s["cr"] / s["n"] if s["n"] else 0,
            "accuracy_norm": s["cn"] / s["n"] if s["n"] else 0,
            "margin_mean": sum(m) / len(m) if m else 0,
        }
    for k, s in sorted(by_k.items()):
        summary["by_k"][str(k)] = {
            "n": s["n"], "accuracy": s["cr"] / s["n"] if s["n"] else 0,
            "updated_acc": s["upd_cr"] / s["upd_n"] if s["upd_n"] else 0,
            "retained_acc": s["ret_cr"] / s["ret_n"] if s["ret_n"] else 0,
        }

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(CHCK82_DIR))
    ap.add_argument("--rows-file", default=str(ROWS_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--frame-filter", default="")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model_dir, device)

    rows = []
    with open(args.rows_file) as f:
        for line in f:
            r = json.loads(line.strip())
            if args.frame_filter and args.frame_filter not in r["frame_id"]:
                continue
            rows.append(r)
    if args.max_rows > 0:
        rows = rows[:args.max_rows]
    print(f"Scoring {len(rows)} rows", flush=True)

    results = []
    t0 = time.time()
    for i, row in enumerate(rows):
        res = score_row(model, tokenizer, row, device)
        if res:
            results.append(res)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)} {el:.0f}s", flush=True)
    el = time.time() - t0
    print(f"Done: {len(results)}/{len(rows)} in {el:.0f}s", flush=True)

    with open(out / "raw_scores.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate(results)
    summary["model"] = args.model_dir
    summary["elapsed_sec"] = round(el, 1)
    summary["frame_filter"] = args.frame_filter

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
