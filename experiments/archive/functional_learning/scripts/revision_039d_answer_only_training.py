#!/usr/bin/env python3
"""Step039d: Answer-only training pilot on relation-first packets.

Tests whether focused answer-only MLM training can install entity-conditioned
state tracking (recipient-sensitive selection) in coherent86.

Training: 90 pairs × 4 rows = 360 training examples (UPDATE/RETAIN only)
Held-out: 30 pairs × 4 rows = 120 evaluation examples
Masking: answer tokens only (multi-token, using token-ID matching from step039b)
"""

import json, pathlib, sys, collections, math, time, random
import torch
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_PATH = "models/frontier"
ROWS_PATH = "experiments/archive/functional_learning/data/relation_first_packets/scoring_rows.jsonl"
PAIRS_PATH = "experiments/archive/functional_learning/data/relation_first_packets/relation_first_pairs.jsonl"
OUT_DIR = pathlib.Path("experiments/archive/functional_learning/data/answer_only_training")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
SEED = 39039
LR = 5e-5
EPOCHS = 500
EVAL_EVERY = 50
WEIGHT_DECAY = 0.01


def find_answer_positions(full_ids, answer_ids):
    """Find positions of answer token subsequence, searching from end."""
    ans_len = len(answer_ids)
    for start in range(len(full_ids) - ans_len, -1, -1):
        if full_ids[start:start + ans_len] == answer_ids:
            return list(range(start, start + ans_len))
    # Fallback without first token (tokenizer prefix differences)
    if ans_len >= 2:
        short = answer_ids[1:]
        for start in range(len(full_ids) - len(short), -1, -1):
            if full_ids[start:start + len(short)] == short:
                return list(range(start, start + len(short)))
    return None


def prepare_data(rows, tokenizer, max_len=512):
    """Prepare training examples with answer-only masking."""
    mask_id = tokenizer.mask_token_id
    examples = []
    
    for row in rows:
        if row["role"] == "NEUTRAL":
            continue  # Skip neutral rows for training
        
        frame = row["use_sentence_frame"]
        answer = row["answer_text"]
        full_text = frame.replace("{STATE}", answer)
        
        enc = tokenizer(full_text, truncation=True, max_length=max_len,
                        return_tensors="pt", padding=False)
        input_ids = enc["input_ids"][0]
        attn_mask = enc["attention_mask"][0]
        
        # Tokenize answer to find positions
        ans_enc = tokenizer(answer, add_special_tokens=False)
        ans_ids = ans_enc["input_ids"]
        
        positions = find_answer_positions(input_ids.tolist(), ans_ids)
        if positions is None:
            continue
        
        # Create labels: -100 everywhere except answer positions
        labels = torch.full_like(input_ids, -100)
        masked_input = input_ids.clone()
        
        for pos in positions:
            labels[pos] = input_ids[pos]
            masked_input[pos] = mask_id
        
        examples.append({
            "input_ids": masked_input,
            "attention_mask": attn_mask,
            "labels": labels,
            "row": row,
            "n_answer_tokens": len(positions),
        })
    
    return examples


def evaluate(model, eval_examples, tokenizer, device):
    """Evaluate: compute per-row margin and per-pair metrics."""
    mask_id = tokenizer.mask_token_id
    model.eval()
    
    scored_rows = []
    
    for ex in eval_examples:
        row = ex["row"]
        frame = row["use_sentence_frame"]
        answer = row["answer_text"]
        foil = row["foil_text"]
        
        scores = {}
        for label, target in [("answer", answer), ("foil", foil)]:
            full_text = frame.replace("{STATE}", target)
            enc = tokenizer(full_text, truncation=True, max_length=512,
                            return_tensors="pt")
            full_ids = enc["input_ids"][0].tolist()
            
            tgt_enc = tokenizer(target, add_special_tokens=False)
            tgt_ids = tgt_enc["input_ids"]
            positions = find_answer_positions(full_ids, tgt_ids)
            
            if positions is None:
                scores[f"s_{label}"] = float("nan")
                continue
            
            total_lp = 0.0
            with torch.no_grad():
                masked = enc["input_ids"].clone().to(device)
                for pos in positions:
                    true_id = masked[0, pos].item()
                    masked[0, pos] = mask_id
                
                logits = model(
                    input_ids=masked,
                    attention_mask=enc["attention_mask"].to(device)
                ).logits[0]
                
                for pos in positions:
                    true_id = enc["input_ids"][0, pos].item()
                    lp = torch.log_softmax(logits[pos], dim=-1)
                    total_lp += lp[true_id].item()
            
            scores[f"s_{label}"] = total_lp / max(len(positions), 1)
        
        margin = scores.get("s_answer", float("nan")) - scores.get("s_foil", float("nan"))
        scored_rows.append({
            "pair_id": row["pair_id"],
            "row_type": row["row_type"],
            "role": row["role"],
            "margin": margin,
            "correct": margin > 0 if not math.isnan(margin) else False,
        })
    
    # Compute pair metrics
    by_pair = collections.defaultdict(dict)
    for r in scored_rows:
        by_pair[r["pair_id"]][r["row_type"]] = r
    
    metrics = {"n_pairs": 0, "n_update": 0, "n_retain": 0, "n_joint": 0,
               "sum_U": 0, "sum_R": 0}
    
    for pid, prows in by_pair.items():
        u_aa = prows.get("update_a_query_a", {}).get("margin", float("nan"))
        u_bb = prows.get("update_b_query_b", {}).get("margin", float("nan"))
        r_ab = prows.get("update_a_query_b", {}).get("margin", float("nan"))
        r_ba = prows.get("update_b_query_a", {}).get("margin", float("nan"))
        
        required = [u_aa, u_bb, r_ab, r_ba]
        if any(math.isnan(v) for v in required):
            continue
        
        U = (u_aa + u_bb) / 2
        R = (r_ab + r_ba) / 2
        
        metrics["n_pairs"] += 1
        metrics["sum_U"] += U
        metrics["sum_R"] += R
        
        update_ok = (u_aa > 0) and (u_bb > 0)
        retain_ok = (r_ab > 0) and (r_ba > 0)
        
        if update_ok:
            metrics["n_update"] += 1
        if retain_ok:
            metrics["n_retain"] += 1
        if update_ok and retain_ok:
            metrics["n_joint"] += 1
    
    n = max(metrics["n_pairs"], 1)
    return {
        "n_pairs": metrics["n_pairs"],
        "mean_U": round(metrics["sum_U"] / n, 4),
        "mean_R": round(metrics["sum_R"] / n, 4),
        "mean_beta": round((metrics["sum_U"] + metrics["sum_R"]) / (2 * n), 4),
        "mean_alpha": round((metrics["sum_U"] - metrics["sum_R"]) / (2 * n), 4),
        "n_update": metrics["n_update"],
        "n_retain": metrics["n_retain"],
        "n_joint": metrics["n_joint"],
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"Loading model on {DEVICE}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, local_files_only=True)
    model.to(DEVICE)
    
    # Load rows and split
    all_rows = [json.loads(l) for l in open(ROWS_PATH)]
    pairs_map = {}
    for l in open(PAIRS_PATH):
        p = json.loads(l)
        pairs_map[p["pair_id"]] = p
    
    train_rows = [r for r in all_rows if pairs_map.get(r["pair_id"], {}).get("split") == "train"]
    held_rows = [r for r in all_rows if pairs_map.get(r["pair_id"], {}).get("split") == "held"]
    
    print(f"Train rows: {len(train_rows)}, Held rows: {len(held_rows)}", flush=True)
    
    # Prepare data
    train_examples = prepare_data(train_rows, tokenizer)
    # For eval, include UPDATE/RETAIN rows only
    eval_train_rows = [r for r in train_rows if r["role"] != "NEUTRAL"]
    eval_held_rows = [r for r in held_rows if r["role"] != "NEUTRAL"]
    eval_train_examples = prepare_data(eval_train_rows, tokenizer)
    eval_held_examples = prepare_data(eval_held_rows, tokenizer)
    
    print(f"Train examples: {len(train_examples)}", flush=True)
    print(f"Eval train examples: {len(eval_train_examples)}", flush=True)
    print(f"Eval held examples: {len(eval_held_examples)}", flush=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Baseline evaluation
    baseline_train = evaluate(model, eval_train_examples, tokenizer, DEVICE)
    baseline_held = evaluate(model, eval_held_examples, tokenizer, DEVICE)
    print(f"\nBaseline train: U={baseline_train['mean_U']:+.3f} R={baseline_train['mean_R']:+.3f} "
          f"beta={baseline_train['mean_beta']:+.3f} joint={baseline_train['n_joint']}/{baseline_train['n_pairs']}", flush=True)
    print(f"Baseline held:  U={baseline_held['mean_U']:+.3f} R={baseline_held['mean_R']:+.3f} "
          f"beta={baseline_held['mean_beta']:+.3f} joint={baseline_held['n_joint']}/{baseline_held['n_pairs']}", flush=True)
    
    trajectory = [{
        "epoch": 0,
        "train": baseline_train,
        "held": baseline_held,
        "train_loss": None,
    }]
    
    # Training loop
    rng = random.Random(SEED)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        # Shuffle training examples
        indices = list(range(len(train_examples)))
        rng.shuffle(indices)
        
        epoch_loss = 0.0
        n_tokens = 0
        
        for idx in indices:
            ex = train_examples[idx]
            input_ids = ex["input_ids"].unsqueeze(0).to(DEVICE)
            attn_mask = ex["attention_mask"].unsqueeze(0).to(DEVICE)
            labels = ex["labels"].unsqueeze(0).to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item() * ex["n_answer_tokens"]
            n_tokens += ex["n_answer_tokens"]
        
        mean_loss = epoch_loss / max(n_tokens, 1)
        
        # Evaluate periodically
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            tr = evaluate(model, eval_train_examples, tokenizer, DEVICE)
            he = evaluate(model, eval_held_examples, tokenizer, DEVICE)
            
            trajectory.append({
                "epoch": epoch,
                "train": tr,
                "held": he,
                "train_loss": round(mean_loss, 4),
            })
            
            print(f"[e{epoch:04d}] loss={mean_loss:.4f} | "
                  f"tr j={tr['n_joint']}/{tr['n_pairs']} U={tr['mean_U']:+.3f} R={tr['mean_R']:+.3f} "
                  f"beta={tr['mean_beta']:+.3f} | "
                  f"he j={he['n_joint']}/{he['n_pairs']} U={he['mean_U']:+.3f} R={he['mean_R']:+.3f} "
                  f"beta={he['mean_beta']:+.3f}", flush=True)
    
    # Save results
    summary = {
        "status": "STEP039D_ANSWER_ONLY_TRAINING",
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "seed": SEED,
        "lr": LR,
        "epochs": EPOCHS,
        "n_train_examples": len(train_examples),
        "n_eval_train": len(eval_train_examples),
        "n_eval_held": len(eval_held_examples),
        "baseline_train": baseline_train,
        "baseline_held": baseline_held,
        "final_train": trajectory[-1]["train"],
        "final_held": trajectory[-1]["held"],
        "trajectory": trajectory,
    }
    
    with open(OUT_DIR / "answer_only_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Save trajectory
    with open(OUT_DIR / "trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2)
    
    print("\n" + "=" * 60, flush=True)
    print(json.dumps({
        "status": summary["status"],
        "final_train": summary["final_train"],
        "final_held": summary["final_held"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
