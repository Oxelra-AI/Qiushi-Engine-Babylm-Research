#!/usr/bin/env python3
"""Step039b: Robust scorer for relation-first packets.

Fixes the offset-mapping bug in the original scorer by using token-ID 
matching instead of character spans. For each scoring row, masks answer
tokens in the full context and computes mean per-token log-probability.
"""

import json, pathlib, sys, collections, math
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_PATH = "models/frontier"
ROWS_PATH = "experiments/archive/functional_learning/data/relation_first_packets/scoring_rows.jsonl"
PAIRS_PATH = "experiments/archive/functional_learning/data/relation_first_packets/relation_first_pairs.jsonl"
OUT_DIR = pathlib.Path("experiments/archive/functional_learning/data/relation_first_scored")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"


def find_answer_token_positions(full_ids, answer_ids, mask_id):
    """Find positions of answer tokens in the full sequence, searching from the end."""
    # Search from the end for the answer token subsequence
    ans_len = len(answer_ids)
    full_len = len(full_ids)
    
    for start in range(full_len - ans_len, -1, -1):
        if full_ids[start:start + ans_len] == answer_ids:
            return list(range(start, start + ans_len))
    
    # Fallback: try matching without leading special tokens
    # Some tokenizers add a leading token to standalone words
    if ans_len >= 2:
        # Try matching the last (ans_len-1) tokens
        short_ids = answer_ids[1:]
        for start in range(full_len - len(short_ids), -1, -1):
            if full_ids[start:start + len(short_ids)] == short_ids:
                return list(range(start, start + len(short_ids)))
    
    return None


def score_row(row, tokenizer, model, device, mask_id):
    """Score a single row: compute mean log-prob of answer and foil tokens."""
    frame = row["use_sentence_frame"]
    answer = row["answer_text"]
    foil = row["foil_text"]
    
    results = {}
    for label, target in [("answer", answer), ("foil", foil)]:
        # Build full text with target inserted
        full_text = frame.replace("{STATE}", target)
        
        # Tokenize full text and target separately
        full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
        full_ids = full_enc["input_ids"][0].tolist()
        
        # Tokenize just the target (with context for proper tokenization)
        # Use the frame ending to get correct tokenization
        prefix = frame.split("{STATE}")[0]
        prefix_enc = tokenizer(prefix, truncation=True, max_length=512)
        prefix_ids = prefix_enc["input_ids"]
        
        # The answer tokens are those after the prefix
        full_ids_list = full_ids
        prefix_len = len(prefix_ids)
        
        # Find answer positions by comparing full encoding with prefix
        # More robust: tokenize target in the context where it appears
        target_enc = tokenizer(target, add_special_tokens=False)
        target_ids = target_enc["input_ids"]
        
        if not target_ids:
            results[f"score_{label}"] = float("nan")
            results[f"n_tokens_{label}"] = 0
            continue
        
        # Find target positions from the end of the sequence
        positions = find_answer_token_positions(full_ids, target_ids, mask_id)
        
        if positions is None:
            # Try with add_special_tokens=True and strip special tokens
            target_enc2 = tokenizer(target)
            target_ids2 = [t for t in target_enc2["input_ids"] 
                          if t != tokenizer.cls_token_id and t != tokenizer.sep_token_id 
                          and t != tokenizer.pad_token_id]
            positions = find_answer_token_positions(full_ids, target_ids2, mask_id)
        
        if positions is None:
            results[f"score_{label}"] = float("nan")
            results[f"n_tokens_{label}"] = 0
            continue
        
        # Mask each target position one at a time and score
        total_logprob = 0.0
        n_scored = 0
        
        with torch.no_grad():
            for pos in positions:
                masked = full_enc["input_ids"].clone().to(device)
                true_id = masked[0, pos].item()
                masked[0, pos] = mask_id
                
                logits = model(
                    input_ids=masked, 
                    attention_mask=full_enc["attention_mask"].to(device)
                ).logits[0]
                
                lp = torch.log_softmax(logits[pos], dim=-1)
                total_logprob += lp[true_id].item()
                n_scored += 1
        
        results[f"score_{label}"] = total_logprob / max(n_scored, 1)
        results[f"n_tokens_{label}"] = n_scored
    
    margin = results.get("score_answer", float("nan")) - results.get("score_foil", float("nan"))
    correct = margin > 0 if not math.isnan(margin) else False
    
    results["margin"] = margin
    results["correct"] = correct
    return results


def main():
    print(f"Loading model on {DEVICE}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, local_files_only=True)
    model.to(DEVICE).eval()
    mask_id = tokenizer.mask_token_id
    
    rows = [json.loads(l) for l in open(ROWS_PATH)]
    pairs_map = {}
    for l in open(PAIRS_PATH):
        p = json.loads(l)
        pairs_map[p["pair_id"]] = p
    
    print(f"Scoring {len(rows)} rows...", flush=True)
    
    scored_rows = []
    n_nan = 0
    for i, row in enumerate(rows):
        scores = score_row(row, tokenizer, model, DEVICE, mask_id)
        scored = dict(row)
        scored.update(scores)
        scored_rows.append(scored)
        
        if math.isnan(scores.get("margin", float("nan"))):
            n_nan += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Scored {i+1}/{len(rows)}, NaN so far: {n_nan}", flush=True)
    
    print(f"\nTotal NaN margins: {n_nan}/{len(rows)}", flush=True)
    
    # Save scored rows
    with open(OUT_DIR / "scored_rows.jsonl", "w") as f:
        for r in scored_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Compute pair metrics
    by_pair = collections.defaultdict(dict)
    for r in scored_rows:
        by_pair[r["pair_id"]][r["row_type"]] = r
    
    pair_metrics = []
    for pid, prows in by_pair.items():
        u_aa = prows.get("update_a_query_a", {}).get("margin", float("nan"))
        u_bb = prows.get("update_b_query_b", {}).get("margin", float("nan"))
        r_ab = prows.get("update_a_query_b", {}).get("margin", float("nan"))
        r_ba = prows.get("update_b_query_a", {}).get("margin", float("nan"))
        n_a = prows.get("neutral_query_a", {}).get("margin", float("nan"))
        n_b = prows.get("neutral_query_b", {}).get("margin", float("nan"))
        
        # Handle NaN: skip pair if any required margin is NaN
        required = [u_aa, u_bb, r_ab, r_ba]
        has_nan = any(math.isnan(v) for v in required)
        
        if has_nan:
            pair_metrics.append({
                "pair_id": pid, "has_nan": True,
                "relation": prows.get("update_a_query_a", {}).get("relation", "?"),
                "split": pairs_map.get(pid, {}).get("split", "?"),
            })
            continue
        
        U = (u_aa + u_bb) / 2
        R = (r_ab + r_ba) / 2
        N = (n_a + n_b) / 2 if not (math.isnan(n_a) or math.isnan(n_b)) else float("nan")
        
        beta = (U + R) / 2
        alpha = (U - R) / 2
        gamma = beta - abs(alpha)
        
        update_correct = (u_aa > 0) and (u_bb > 0)
        retain_correct = (r_ab > 0) and (r_ba > 0)
        joint = update_correct and retain_correct
        
        pair_metrics.append({
            "pair_id": pid, "has_nan": False,
            "relation": prows.get("update_a_query_a", {}).get("relation", "?"),
            "split": pairs_map.get(pid, {}).get("split", "?"),
            "U": round(U, 4), "R": round(R, 4), "N": round(N, 4) if not math.isnan(N) else None,
            "beta": round(beta, 4), "alpha": round(alpha, 4), "gamma": round(gamma, 4),
            "update_correct": update_correct, "retain_correct": retain_correct,
            "joint": joint,
            "u_aa": round(u_aa, 4), "u_bb": round(u_bb, 4),
            "r_ab": round(r_ab, 4), "r_ba": round(r_ba, 4),
            "n_a": round(n_a, 4) if not math.isnan(n_a) else None,
            "n_b": round(n_b, 4) if not math.isnan(n_b) else None,
        })
    
    with open(OUT_DIR / "pair_metrics.jsonl", "w") as f:
        for m in pair_metrics:
            f.write(json.dumps(m) + "\n")
    
    # Compute summaries
    valid_metrics = [m for m in pair_metrics if not m["has_nan"]]
    nan_metrics = [m for m in pair_metrics if m["has_nan"]]
    
    def summarize(metrics, label):
        n = len(metrics)
        if n == 0:
            return {"label": label, "n": 0}
        mean_U = sum(m["U"] for m in metrics) / n
        mean_R = sum(m["R"] for m in metrics) / n
        mean_beta = sum(m["beta"] for m in metrics) / n
        mean_alpha = sum(m["alpha"] for m in metrics) / n
        mean_abs_alpha = sum(abs(m["alpha"]) for m in metrics) / n
        mean_gamma = sum(m["gamma"] for m in metrics) / n
        n_up = sum(1 for m in metrics if m["update_correct"])
        n_ret = sum(1 for m in metrics if m["retain_correct"])
        n_joint = sum(1 for m in metrics if m["joint"])
        return {
            "label": label, "n": n,
            "mean_U": round(mean_U, 4), "mean_R": round(mean_R, 4),
            "mean_beta": round(mean_beta, 4),
            "mean_alpha": round(mean_alpha, 4),
            "mean_abs_alpha": round(mean_abs_alpha, 4),
            "mean_gamma": round(mean_gamma, 4),
            "n_update_correct": n_up, "n_retain_correct": n_ret, "n_joint": n_joint,
            "update_rate": round(n_up / n, 4), "retain_rate": round(n_ret / n, 4),
            "joint_rate": round(n_joint / n, 4),
        }
    
    train_valid = [m for m in valid_metrics if m["split"] == "train"]
    held_valid = [m for m in valid_metrics if m["split"] == "held"]
    
    by_rel = collections.defaultdict(list)
    for m in valid_metrics:
        by_rel[m["relation"]].append(m)
    
    result = {
        "status": "STEP039B_RELATION_FIRST_SCORED",
        "n_pairs_total": len(pair_metrics),
        "n_pairs_valid": len(valid_metrics),
        "n_pairs_nan": len(nan_metrics),
        "n_rows_nan": n_nan,
        "all": summarize(valid_metrics, "all"),
        "train": summarize(train_valid, "train"),
        "held": summarize(held_valid, "held"),
        "by_relation": {rel: summarize(ms, rel) for rel, ms in by_rel.items()},
    }
    
    with open(OUT_DIR / "relation_first_scored_summary.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60, flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
