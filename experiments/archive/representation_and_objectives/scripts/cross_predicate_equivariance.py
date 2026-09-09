#!/usr/bin/env python3
"""research: Cross-predicate transfer benchmark and equivariance comparison.

Scientific question: Does explicit paired-world equivariance training improve
cross-predicate role-assignment transfer compared to standard sequence learning?

research showed a tiny GRU trained on 'defeated' templates achieves 96.5% on held
families but only 4.5% on 'lost_to' hypotheses. The question is:
1. Does multi-predicate training improve transfer?
2. Does an explicit equivariance loss improve transfer BEYOND multi-predicate exposure?
3. What is the transfer rate under wholly held predicates?

No BabyLM training, DeBERTa/RoBERTa, evaluation, upload, or submission.
CPU only. Entity names are canonicalized to ENTITY_A / ENTITY_B.
"""

import json, os, sys, random, math, hashlib, collections
from pathlib import Path
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────────
WORKSPACE = Path("experiments/archive/representation_and_objectives")
FAMILIES_TRAIN = WORKSPACE / "data/paired_world_pilot/families_train.jsonl"
FAMILIES_HELD  = WORKSPACE / "data/paired_world_pilot/families_held.jsonl"
OUT_DIR        = WORKSPACE / "data/cross_predicate_equivariance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Predicate definitions ───────────────────────────────────────────────────
# Each predicate: (split, polarity, context_pattern, hypothesis_pattern)
#   split: "train" or "held"
#   polarity: "wf" (winner-first: W verb L) or "lf" (loser-first: L verb W)

PREDICATES = {
    # Training predicates
    "defeated":    ("train", "wf", "On {D}, {W} defeated {L} in the {R} at {T}.",
                    "{X} defeated {Y}."),
    "beat":        ("train", "wf", "On {D}, {W} beat {L} in the {R} at {T}.",
                    "{X} beat {Y}."),
    "lost_to":     ("train", "lf", "On {D}, {L} lost to {W} in the {R} at {T}.",
                    "{X} lost to {Y}."),
    "was_beaten":  ("train", "lf", "On {D}, {L} was beaten by {W} in the {R} at {T}.",
                    "{X} was beaten by {Y}."),
    # Held predicates
    "overcame":    ("held", "wf", "On {D}, {W} overcame {L} in the {R} at {T}.",
                    "{X} overcame {Y}."),
    "prevailed":   ("held", "wf", "On {D}, {W} prevailed over {L} in the {R} at {T}.",
                    "{X} prevailed over {Y}."),
    "fell_to":     ("held", "lf", "On {D}, {L} fell to {W} in the {R} at {T}.",
                    "{X} fell to {Y}."),
    "was_defeated":("held", "lf", "On {D}, {L} was defeated by {W} in the {R} at {T}.",
                    "{X} was defeated by {Y}."),
}

TRAIN_PREDS = [k for k, v in PREDICATES.items() if v[0] == "train"]
HELD_PREDS  = [k for k, v in PREDICATES.items() if v[0] == "held"]

# ─── Data generation ─────────────────────────────────────────────────────────
def load_families(path):
    fams = []
    with open(path) as f:
        for line in f:
            fams.append(json.loads(line))
    return fams

def render_context(pred_key, winner, loser, date, round_n, tournament):
    """Render a context using the specified predicate."""
    split, pol, ctx_pat, _ = PREDICATES[pred_key]
    return ctx_pat.format(W=winner, L=loser, D=date, R=round_n, T=tournament)

def render_hypothesis(pred_key, x, y):
    """Render a hypothesis: '{X} [verb] {Y}.'"""
    _, _, _, hyp_pat = PREDICATES[pred_key]
    return hyp_pat.format(X=x, Y=y)

def generate_nli_rows(families, ctx_preds, hyp_preds, canonical=True):
    """Generate NLI rows for all families with given predicate sets.
    
    Returns list of dicts with:
      family_id, world (1 or 2), ctx_pred, hyp_pred, hyp_orient (AB or BA),
      text (context [SEP] hypothesis), label (1=entailed, 0=not),
      family_split (train/held)
    """
    rows = []
    for fam in families:
        A = "ENTITY_A" if canonical else fam["participant_a"]
        B = "ENTITY_B" if canonical else fam["participant_b"]
        
        for world_key in ["context1", "context2"]:
            world_num = 1 if world_key == "context1" else 2
            ctx = fam[world_key]
            W_label = ctx["winner_label"]  # "A" or "B"
            winner = A if W_label == "A" else B
            loser  = B if W_label == "A" else A
            
            date = ctx.get("date_formatted", "an unknown date")
            rnd  = ctx.get("round_normalized", "a match")
            tourn = ctx.get("event", {}).get("tournament", "a tournament")
            
            for cp in ctx_preds:
                context_text = render_context(cp, winner, loser, date, rnd, tourn)
                
                for hp in hyp_preds:
                    hp_split, hp_pol, _, _ = PREDICATES[hp]
                    
                    for orient in ["AB", "BA"]:
                        if orient == "AB":
                            x, y = A, B
                        else:
                            x, y = B, A
                        
                        hyp_text = render_hypothesis(hp, x, y)
                        
                        # Determine label
                        if hp_pol == "wf":
                            # "X [winner-verb] Y" → entailed if X is the winner
                            label = 1 if x == winner else 0
                        else:  # "lf"
                            # "X [loser-verb] Y" → entailed if X is the loser
                            label = 1 if x == loser else 0
                        
                        rows.append({
                            "family_id": fam["family_id"],
                            "world": world_num,
                            "ctx_pred": cp,
                            "hyp_pred": hp,
                            "hyp_orient": orient,
                            "text": context_text + " [SEP] " + hyp_text,
                            "label": label,
                            "family_split": fam.get("family_split", "train"),
                        })
    return rows


# ─── Tokenizer ───────────────────────────────────────────────────────────────
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1, "[SEP]": 2}
        self.frozen = False
    
    def fit(self, texts):
        for t in texts:
            for w in t.lower().split():
                if w not in self.word2idx:
                    self.word2idx[w] = len(self.word2idx)
        self.frozen = True
    
    def encode(self, text, max_len=128):
        ids = []
        for w in text.lower().split():
            ids.append(self.word2idx.get(w, 1))
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        return ids


# ─── PyTorch model ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class GRU_NLI(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.classifier = nn.Linear(hidden_dim * 2, 2)
        
        def encode(self, x):
            emb = self.embedding(x)
            out, _ = self.gru(emb)
            # Use mean-pool of GRU outputs as representation
            mask = (x != 0).unsqueeze(-1).float()
            h = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            return h
        
        def forward(self, x):
            h = self.encode(x)
            return self.classifier(h)


# ─── Training functions ──────────────────────────────────────────────────────
def build_equivariance_pairs(rows, tok, max_len=128):
    """Build explicit (i, j) equivariance pairs from same (family, world, hyp_orient, hyp_pred) 
    but different ctx_pred. Returns list of (idx_i, idx_j)."""
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        key = (r["family_id"], r["world"], r["hyp_orient"], r["hyp_pred"])
        groups[key].append(i)
    
    pairs = []
    for key, indices in groups.items():
        for a in range(len(indices)):
            for b in range(a+1, len(indices)):
                if rows[indices[a]]["ctx_pred"] != rows[indices[b]]["ctx_pred"]:
                    pairs.append((indices[a], indices[b]))
    return pairs

def train_model(model, train_data, train_labels, epochs=20, lr=0.001,
                equiv_pairs=None, lambda_equiv=0.0, batch_size=256,
                pair_batch_size=512, seed=42):
    """Train with optional equivariance loss on explicit pair batches."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(train_data, dtype=torch.long)
    Y = torch.tensor(train_labels, dtype=torch.long)
    
    # Pre-convert equiv pairs to tensor for fast sampling
    eq_tensor = None
    if equiv_pairs and lambda_equiv > 0:
        eq_tensor = torch.tensor(equiv_pairs, dtype=torch.long)  # (N_pairs, 2)
    
    n = len(X)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            xb, yb = X[idx], Y[idx]
            
            logits = model(xb)
            ce_loss = F.cross_entropy(logits, yb)
            
            loss = ce_loss
            
            if eq_tensor is not None and lambda_equiv > 0:
                # Sample explicit equivariance pair batch
                pair_idx = torch.randint(0, len(eq_tensor), (min(pair_batch_size, len(eq_tensor)),))
                pairs = eq_tensor[pair_idx]
                xa = X[pairs[:, 0]]
                xb_eq = X[pairs[:, 1]]
                
                ha = model.encode(xa)
                hb = model.encode(xb_eq)
                eq_loss = F.mse_loss(ha, hb)
                loss = ce_loss + lambda_equiv * eq_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model

def evaluate_model(model, data, labels):
    """Evaluate and return accuracy."""
    model.eval()
    X = torch.tensor(data, dtype=torch.long)
    Y = torch.tensor(labels, dtype=torch.long)
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(1)
        acc = (preds == Y).float().mean().item()
    return acc


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)
    
    print("Loading families...")
    train_fams = load_families(FAMILIES_TRAIN)
    held_fams  = load_families(FAMILIES_HELD)
    print(f"  Train families: {len(train_fams)}, Held families: {len(held_fams)}")
    
    # ── Generate multi-predicate NLI rows ────────────────────────────────────
    print("\nGenerating multi-predicate NLI rows...")
    
    # Training data: train families × train predicates
    train_rows = generate_nli_rows(train_fams, TRAIN_PREDS, TRAIN_PREDS)
    print(f"  Train rows (train_fam × train_pred): {len(train_rows)}")
    
    # Held-family evaluation: held families × train predicates
    held_fam_train_pred = generate_nli_rows(held_fams, TRAIN_PREDS, TRAIN_PREDS)
    print(f"  Held-family train-pred rows: {len(held_fam_train_pred)}")
    
    # Cross-predicate evaluation: train families × held predicates
    cross_pred_train_fam = generate_nli_rows(train_fams, TRAIN_PREDS, HELD_PREDS)
    print(f"  Train-family cross-pred rows: {len(cross_pred_train_fam)}")
    
    # Full held: held families × held predicates
    full_held = generate_nli_rows(held_fams, TRAIN_PREDS, HELD_PREDS)
    print(f"  Full held rows: {len(full_held)}")
    
    # Mixed context eval: held context preds, train hyp preds (tests ctx generalization)
    held_ctx_train_hyp = generate_nli_rows(held_fams, HELD_PREDS, TRAIN_PREDS)
    print(f"  Held-ctx train-hyp rows: {len(held_ctx_train_hyp)}")
    
    # Save benchmark specification
    benchmark_spec = {
        "train_predicates": TRAIN_PREDS,
        "held_predicates": HELD_PREDS,
        "train_fam_count": len(train_fams),
        "held_fam_count": len(held_fams),
        "train_rows": len(train_rows),
        "held_fam_train_pred_rows": len(held_fam_train_pred),
        "cross_pred_train_fam_rows": len(cross_pred_train_fam),
        "full_held_rows": len(full_held),
        "held_ctx_train_hyp_rows": len(held_ctx_train_hyp),
    }
    with open(OUT_DIR / "benchmark_spec.json", "w") as f:
        json.dump(benchmark_spec, f, indent=2)
    
    if not HAS_TORCH:
        print("PyTorch not available. Saving benchmark only.")
        return benchmark_spec
    
    # ── Tokenize ─────────────────────────────────────────────────────────────
    print("\nTokenizing...")
    tok = SimpleTokenizer()
    all_texts = ([r["text"] for r in train_rows] + 
                 [r["text"] for r in held_fam_train_pred] +
                 [r["text"] for r in cross_pred_train_fam] +
                 [r["text"] for r in full_held] +
                 [r["text"] for r in held_ctx_train_hyp])
    tok.fit(all_texts)
    vocab_size = len(tok.word2idx)
    print(f"  Vocab size: {vocab_size}")
    
    MAX_LEN = 96
    train_X = [tok.encode(r["text"], MAX_LEN) for r in train_rows]
    train_Y = [r["label"] for r in train_rows]
    
    eval_sets = {
        "held_fam_train_pred": (held_fam_train_pred, "Held family, train predicates"),
        "cross_pred_train_fam": (cross_pred_train_fam, "Train family, held predicates"),
        "full_held": (full_held, "Held family, held predicates"),
        "held_ctx_train_hyp": (held_ctx_train_hyp, "Held ctx preds, train hyp preds"),
    }
    eval_encoded = {}
    for key, (rows, desc) in eval_sets.items():
        eval_encoded[key] = (
            [tok.encode(r["text"], MAX_LEN) for r in rows],
            [r["label"] for r in rows],
            rows,
            desc
        )
    
    # ── Build equivariance pairs ─────────────────────────────────────────────
    print("\nBuilding equivariance pairs...")
    equiv_pairs = build_equivariance_pairs(train_rows, tok, MAX_LEN)
    print(f"  Equivariance pairs: {len(equiv_pairs)}")
    
    # ── Run experiments ──────────────────────────────────────────────────────
    SEEDS = [42, 137, 2024]
    EPOCHS = 25
    LR = 0.001
    LAMBDA_EQUIV = 1.0
    
    results = {}
    
    for mode in ["standard", "equivariant"]:
        print(f"\n{'='*60}")
        print(f"Training mode: {mode}")
        print(f"{'='*60}")
        
        mode_results = collections.defaultdict(list)
        
        for seed in SEEDS:
            print(f"\n  Seed {seed}...")
            torch.manual_seed(seed)
            model = GRU_NLI(vocab_size, embed_dim=64, hidden_dim=128)
            
            lam = LAMBDA_EQUIV if mode == "equivariant" else 0.0
            train_model(model, train_X, train_Y, epochs=EPOCHS, lr=LR,
                       equiv_pairs=equiv_pairs if mode == "equivariant" else None,
                       lambda_equiv=lam, seed=seed)
            
            # Evaluate on training set
            train_acc = evaluate_model(model, train_X, train_Y)
            mode_results["train"].append(train_acc)
            print(f"    Train acc: {train_acc:.4f}")
            
            # Evaluate on each held set
            for key, (X, Y, rows, desc) in eval_encoded.items():
                acc = evaluate_model(model, X, Y)
                mode_results[key].append(acc)
                print(f"    {desc}: {acc:.4f}")
                
                # Per-polarity breakdown for cross-predicate sets
                if "cross" in key or "full" in key:
                    for pol_name, pol_val in [("wf_hyp", "wf"), ("lf_hyp", "lf")]:
                        pol_indices = [i for i, r in enumerate(rows) 
                                      if PREDICATES[r["hyp_pred"]][1] == pol_val]
                        if pol_indices:
                            pol_X = [X[i] for i in pol_indices]
                            pol_Y = [Y[i] for i in pol_indices]
                            pol_acc = evaluate_model(model, pol_X, pol_Y)
                            mode_results[f"{key}_{pol_name}"].append(pol_acc)
                            print(f"      {pol_name}: {pol_acc:.4f}")
        
        # Summarize
        for key, vals in mode_results.items():
            results[f"{mode}_{key}"] = {
                "mean": np.mean(vals),
                "std": np.std(vals),
                "seeds": vals,
            }
    
    # ── Additional analysis: per-predicate breakdown ─────────────────────────
    print(f"\n{'='*60}")
    print("Per-predicate held-hypothesis accuracy (last seed, both modes)")
    print(f"{'='*60}")
    
    per_pred_results = {}
    for mode in ["standard", "equivariant"]:
        torch.manual_seed(42)
        model = GRU_NLI(vocab_size, embed_dim=64, hidden_dim=128)
        lam = LAMBDA_EQUIV if mode == "equivariant" else 0.0
        train_model(model, train_X, train_Y, epochs=EPOCHS, lr=LR,
                   equiv_pairs=equiv_pairs if mode == "equivariant" else None,
                   lambda_equiv=lam, seed=42)
        
        for pred_key in HELD_PREDS:
            # Filter full_held rows for this predicate
            rows_e = eval_encoded["full_held"]
            pred_indices = [i for i, r in enumerate(rows_e[2]) if r["hyp_pred"] == pred_key]
            if pred_indices:
                pX = [rows_e[0][i] for i in pred_indices]
                pY = [rows_e[1][i] for i in pred_indices]
                acc = evaluate_model(model, pX, pY)
                per_pred_results[f"{mode}_{pred_key}"] = acc
                print(f"  {mode:12s} | {pred_key:15s} | {acc:.4f}")
    
    # ── Single-predicate baseline (defeated only) ────────────────────────────
    print(f"\n{'='*60}")
    print("Single-predicate baseline: train on 'defeated' only")
    print(f"{'='*60}")
    
    single_train = [r for r in train_rows 
                    if r["ctx_pred"] == "defeated" and r["hyp_pred"] == "defeated"]
    single_X = [tok.encode(r["text"], MAX_LEN) for r in single_train]
    single_Y = [r["label"] for r in single_train]
    print(f"  Single-pred training rows: {len(single_train)}")
    
    single_results = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = GRU_NLI(vocab_size, embed_dim=64, hidden_dim=128)
        train_model(model, single_X, single_Y, epochs=EPOCHS, lr=LR, seed=seed)
        
        for key, (X, Y, rows, desc) in eval_encoded.items():
            acc = evaluate_model(model, X, Y)
            if key not in single_results:
                single_results[key] = []
            single_results[key].append(acc)
    
    for key, vals in single_results.items():
        results[f"single_defeated_{key}"] = {
            "mean": np.mean(vals),
            "std": np.std(vals),
            "seeds": vals,
        }
        _, _, _, desc = eval_encoded[key]
        print(f"  {desc}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    # ── Compile and save ─────────────────────────────────────────────────────
    summary = {
        "status": "CROSS_PREDICATE_EQUIVARIANCE",
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": benchmark_spec,
        "model": {"type": "BiGRU", "embed_dim": 64, "hidden_dim": 128,
                  "vocab_size": vocab_size, "canonical_names": True,
                  "max_len": MAX_LEN, "epochs": EPOCHS, "lr": LR},
        "equivariance": {"lambda": LAMBDA_EQUIV, "pairs": len(equiv_pairs)},
        "seeds": SEEDS,
        "results": results,
        "per_predicate_held": per_pred_results,
    }
    
    with open(OUT_DIR / "cross_predicate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Write markdown summary
    md = ["# research Cross-Predicate Equivariance Comparison\n"]
    md.append("## Benchmark")
    md.append(f"- Train families: {len(train_fams)}, Held families: {len(held_fams)}")
    md.append(f"- Training predicates: {TRAIN_PREDS}")
    md.append(f"- Held predicates: {HELD_PREDS}")
    md.append(f"- Training NLI rows: {len(train_rows)}")
    md.append(f"- Equivariance pairs: {len(equiv_pairs)}")
    md.append(f"- Seeds: {SEEDS}, Epochs: {EPOCHS}, λ_equiv: {LAMBDA_EQUIV}\n")
    
    md.append("## Main comparison (mean ± std over 3 seeds)\n")
    md.append("| Condition | Standard | Equivariant | Δ (equiv−std) |")
    md.append("|---|---:|---:|---:|")
    for key in ["train", "held_fam_train_pred", "cross_pred_train_fam",
                "full_held", "held_ctx_train_hyp"]:
        std_m = results.get(f"standard_{key}", {}).get("mean", 0)
        std_s = results.get(f"standard_{key}", {}).get("std", 0)
        eq_m  = results.get(f"equivariant_{key}", {}).get("mean", 0)
        eq_s  = results.get(f"equivariant_{key}", {}).get("std", 0)
        delta = eq_m - std_m
        label = {
            "train": "Train (train fam, train pred)",
            "held_fam_train_pred": "Held family, train pred",
            "cross_pred_train_fam": "**Cross-pred (train fam, held pred)**",
            "full_held": "**Full held (held fam, held pred)**",
            "held_ctx_train_hyp": "Held ctx pred, train hyp pred",
        }.get(key, key)
        md.append(f"| {label} | {std_m:.4f}±{std_s:.4f} | {eq_m:.4f}±{eq_s:.4f} | {delta:+.4f} |")
    
    md.append("\n## Single-predicate baseline (defeated only)\n")
    md.append("| Condition | Acc (mean ± std) |")
    md.append("|---|---:|")
    for key in ["held_fam_train_pred", "cross_pred_train_fam", "full_held"]:
        v = results.get(f"single_defeated_{key}", {})
        m, s = v.get("mean", 0), v.get("std", 0)
        _, _, _, desc = eval_encoded[key]
        md.append(f"| {desc} | {m:.4f}±{s:.4f} |")
    
    md.append("\n## Per-predicate held accuracy (seed 42)\n")
    md.append("| Predicate | Standard | Equivariant |")
    md.append("|---|---:|---:|")
    for pk in HELD_PREDS:
        s = per_pred_results.get(f"standard_{pk}", 0)
        e = per_pred_results.get(f"equivariant_{pk}", 0)
        _, pol, _, _ = PREDICATES[pk]
        md.append(f"| {pk} ({pol}) | {s:.4f} | {e:.4f} |")
    
    md.append("\n## Interpretation")
    md.append("")
    
    # Compute key deltas
    cross_std = results.get("standard_cross_pred_train_fam", {}).get("mean", 0)
    cross_eq  = results.get("equivariant_cross_pred_train_fam", {}).get("mean", 0)
    full_std  = results.get("standard_full_held", {}).get("mean", 0)
    full_eq   = results.get("equivariant_full_held", {}).get("mean", 0)
    
    if cross_eq - cross_std > 0.05:
        md.append("Equivariance training improved cross-predicate transfer by "
                  f"{cross_eq - cross_std:+.4f} on train-family held-predicates "
                  f"and {full_eq - full_std:+.4f} on full held. "
                  "This is evidence for the paired-world equivariance principle.")
    elif cross_eq - cross_std > 0.01:
        md.append("Equivariance training showed a modest improvement in cross-predicate "
                  f"transfer ({cross_eq - cross_std:+.4f}). The effect is small relative "
                  "to the within-predicate accuracy and may not survive broader tests.")
    else:
        md.append("Equivariance training did not meaningfully improve cross-predicate "
                  "transfer over standard multi-predicate training. The bottleneck appears "
                  "to be representation of unseen predicate tokens, not the training objective alone.")
    
    with open((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/cross_predicate_equivariance/cross_predicate_summary.md'), "w") as f:
        f.write("\n".join(md) + "\n")
    
    print(f"\nResults saved to {OUT_DIR}/")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
