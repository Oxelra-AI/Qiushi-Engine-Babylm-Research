#!/usr/bin/env python3
"""research: Lean cross-predicate equivariance comparison (CPU-feasible).

Key question: Does equivariance training improve cross-predicate transfer 
for role assignment beyond standard multi-predicate training?

Uses 100 families (80 train / 20 held), unidirectional GRU, max_len=48.
"""

import json, os, random, collections
from pathlib import Path
import numpy as np

WORKSPACE = Path("experiments/archive/representation_and_objectives")
FAMILIES_TRAIN = WORKSPACE / "data/paired_world_pilot/families_train.jsonl"
FAMILIES_HELD  = WORKSPACE / "data/paired_world_pilot/families_held.jsonl"
OUT_DIR        = WORKSPACE / "data/cross_predicate_equivariance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Predicates ──────────────────────────────────────────────────────────────
# (split, polarity, ctx_pattern, hyp_pattern)
PREDS = {
    "defeated":     ("train", "wf", "On {D}, {W} defeated {L} in the {R} at {T}.",    "{X} defeated {Y}."),
    "beat":         ("train", "wf", "On {D}, {W} beat {L} in the {R} at {T}.",        "{X} beat {Y}."),
    "lost_to":      ("train", "lf", "On {D}, {L} lost to {W} in the {R} at {T}.",     "{X} lost to {Y}."),
    "was_beaten":   ("train", "lf", "On {D}, {L} was beaten by {W} in the {R} at {T}.", "{X} was beaten by {Y}."),
    "overcame":     ("held",  "wf", "On {D}, {W} overcame {L} in the {R} at {T}.",    "{X} overcame {Y}."),
    "prevailed":    ("held",  "wf", "On {D}, {W} prevailed over {L} in the {R} at {T}.", "{X} prevailed over {Y}."),
    "fell_to":      ("held",  "lf", "On {D}, {L} fell to {W} in the {R} at {T}.",     "{X} fell to {Y}."),
    "was_defeated": ("held",  "lf", "On {D}, {L} was defeated by {W} in the {R} at {T}.", "{X} was defeated by {Y}."),
}
TRAIN_P = [k for k, v in PREDS.items() if v[0] == "train"]
HELD_P  = [k for k, v in PREDS.items() if v[0] == "held"]

# ─── Data ────────────────────────────────────────────────────────────────────
def load_fams(path, limit=None):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out

def gen_rows(families, ctx_preds, hyp_preds):
    rows = []
    for fam in families:
        A, B = "ENTITY_A", "ENTITY_B"
        for wk in ["context1", "context2"]:
            ctx = fam[wk]
            W_lab = ctx["winner_label"]
            W = A if W_lab == "A" else B
            L = B if W_lab == "A" else A
            d = ctx.get("date_formatted", "a date")
            r = ctx.get("round_normalized", "a match")
            t = ctx.get("event", {}).get("tournament", "an event")
            world = 1 if wk == "context1" else 2
            
            for cp in ctx_preds:
                _, pol_c, cpat, _ = PREDS[cp]
                ctxt = cpat.format(W=W, L=L, D=d, R=r, T=t)
                
                for hp in hyp_preds:
                    _, pol_h, _, hpat = PREDS[hp]
                    for orient in ["AB", "BA"]:
                        x = A if orient == "AB" else B
                        y = B if orient == "AB" else A
                        hyp = hpat.format(X=x, Y=y)
                        
                        if pol_h == "wf":
                            label = 1 if x == W else 0
                        else:
                            label = 1 if x == L else 0
                        
                        rows.append({
                            "fid": fam["family_id"], "world": world,
                            "cp": cp, "hp": hp, "orient": orient,
                            "text": ctxt + " [SEP] " + hyp,
                            "label": label,
                        })
    return rows

# ─── Tokenizer ───────────────────────────────────────────────────────────────
class Tok:
    def __init__(self):
        self.w2i = {"<pad>": 0, "<unk>": 1}
        self.frozen = False
    def fit(self, texts):
        for t in texts:
            for w in t.lower().split():
                if w not in self.w2i:
                    self.w2i[w] = len(self.w2i)
        self.frozen = True
    def enc(self, text, ml=48):
        ids = [self.w2i.get(w, 1) for w in text.lower().split()][:ml]
        return ids + [0] * (ml - len(ids))

# ─── Model ───────────────────────────────────────────────────────────────────
import torch, torch.nn as nn, torch.nn.functional as F

class GRU_NLI(nn.Module):
    def __init__(self, vs, ed=48, hd=96):
        super().__init__()
        self.emb = nn.Embedding(vs, ed, padding_idx=0)
        self.gru = nn.GRU(ed, hd, batch_first=True)
        self.cls = nn.Linear(hd, 2)
    def encode(self, x):
        e = self.emb(x)
        o, _ = self.gru(e)
        m = (x != 0).unsqueeze(-1).float()
        return (o * m).sum(1) / m.sum(1).clamp(min=1)
    def forward(self, x):
        return self.cls(self.encode(x))

def build_equiv_pairs(rows):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["fid"], r["world"], r["orient"], r["hp"])].append(i)
    pairs = []
    for idxs in groups.values():
        for a in range(len(idxs)):
            for b in range(a+1, len(idxs)):
                if rows[idxs[a]]["cp"] != rows[idxs[b]]["cp"]:
                    pairs.append((idxs[a], idxs[b]))
    return pairs

def train_m(model, X, Y, epochs, lr, eq_pairs=None, lam=0.0, bs=512, pbs=64, seed=42):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, Yt = torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)
    eq_t = torch.tensor(eq_pairs, dtype=torch.long) if eq_pairs and lam > 0 else None
    n = len(Xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            logits = model(Xt[idx])
            loss = F.cross_entropy(logits, Yt[idx])
            if eq_t is not None:
                pi = torch.randint(0, len(eq_t), (min(pbs, len(eq_t)),))
                p = eq_t[pi]
                ha = model.encode(Xt[p[:, 0]])
                hb = model.encode(Xt[p[:, 1]])
                loss = loss + lam * F.mse_loss(ha, hb)
            opt.zero_grad(); loss.backward(); opt.step()

def eval_m(model, X, Y):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.long)
    Yt = torch.tensor(Y, dtype=torch.long)
    with torch.no_grad():
        return (model(Xt).argmax(1) == Yt).float().mean().item()

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    random.seed(42); np.random.seed(42)
    
    # Load subsets
    tr_fams = load_fams(FAMILIES_TRAIN, limit=80)
    he_fams = load_fams(FAMILIES_HELD, limit=20)
    print(f"Train fams: {len(tr_fams)}, Held fams: {len(he_fams)}")
    
    # Generate rows
    tr_rows = gen_rows(tr_fams, TRAIN_P, TRAIN_P)
    he_tp   = gen_rows(he_fams, TRAIN_P, TRAIN_P)     # held fam, train pred
    cp_tf   = gen_rows(tr_fams, TRAIN_P, HELD_P)      # train fam, cross pred
    full_he = gen_rows(he_fams, TRAIN_P, HELD_P)      # held fam, held pred
    he_ctx  = gen_rows(he_fams, HELD_P, TRAIN_P)      # held ctx, train hyp
    
    # Single-pred baseline: only defeated context + defeated hypothesis
    sp_rows = gen_rows(tr_fams, ["defeated"], ["defeated"])
    
    print(f"Train rows: {len(tr_rows)}, Single-pred: {len(sp_rows)}")
    print(f"Eval: he_tp={len(he_tp)}, cp_tf={len(cp_tf)}, full_he={len(full_he)}, he_ctx={len(he_ctx)}")
    
    # Tokenize
    tok = Tok()
    all_t = [r["text"] for r in tr_rows + he_tp + cp_tf + full_he + he_ctx + sp_rows]
    tok.fit(all_t)
    vs = len(tok.w2i)
    ML = 48
    
    def enc_set(rows):
        return [tok.enc(r["text"], ML) for r in rows], [r["label"] for r in rows]
    
    trX, trY = enc_set(tr_rows)
    spX, spY = enc_set(sp_rows)
    evals = {
        "he_fam_tr_pred":   (enc_set(he_tp),   he_tp,   "Held family, train pred"),
        "cross_pred_tr_fam":(enc_set(cp_tf),   cp_tf,   "Train family, held pred"),
        "full_held":        (enc_set(full_he), full_he, "Held family, held pred"),
        "held_ctx":         (enc_set(he_ctx),  he_ctx,  "Held ctx pred, train hyp"),
    }
    
    eq_pairs = build_equiv_pairs(tr_rows)
    print(f"Equivariance pairs: {len(eq_pairs)}")
    
    SEEDS = [42, 137, 2024]
    EPOCHS = 15
    LR = 0.002
    LAM = 1.0
    results = {}
    
    for mode in ["single_defeated", "standard_multi", "equivariant"]:
        print(f"\n{'='*50}\nMode: {mode}\n{'='*50}")
        for key in list(evals.keys()) + ["train"]:
            results[f"{mode}_{key}"] = []
        
        for seed in SEEDS:
            m = GRU_NLI(vs)
            if mode == "single_defeated":
                train_m(m, spX, spY, EPOCHS, LR, seed=seed)
            elif mode == "standard_multi":
                train_m(m, trX, trY, EPOCHS, LR, seed=seed)
            else:
                train_m(m, trX, trY, EPOCHS, LR, eq_pairs=eq_pairs, lam=LAM, pbs=64, seed=seed)
            
            ta = eval_m(m, trX, trY)
            results[f"{mode}_train"].append(ta)
            print(f"  seed={seed} train={ta:.4f}", end="")
            
            for key, ((X, Y), rows, desc) in evals.items():
                a = eval_m(m, X, Y)
                results[f"{mode}_{key}"].append(a)
                print(f" {key}={a:.4f}", end="")
            print()
    
    # Per-held-predicate breakdown (seed 42, standard vs equivariant)
    per_pred = {}
    for mode in ["standard_multi", "equivariant"]:
        m = GRU_NLI(vs)
        if mode == "standard_multi":
            train_m(m, trX, trY, EPOCHS, LR, seed=42)
        else:
            train_m(m, trX, trY, EPOCHS, LR, eq_pairs=eq_pairs, lam=LAM, pbs=64, seed=42)
        
        for pk in HELD_P:
            (allX, allY), allR, _ = evals["full_held"]
            idx = [i for i, r in enumerate(allR) if r["hp"] == pk]
            if idx:
                pX = [allX[i] for i in idx]
                pY = [allY[i] for i in idx]
                per_pred[f"{mode}_{pk}"] = eval_m(m, pX, pY)
    
    # ── Compile ──────────────────────────────────────────────────────────────
    summary = {
        "status": "CROSS_PREDICATE_EQUIVARIANCE",
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {"train_fams": len(tr_fams), "held_fams": len(he_fams),
                   "train_rows": len(tr_rows), "single_pred_rows": len(sp_rows),
                   "equiv_pairs": len(eq_pairs), "vocab": vs,
                   "epochs": EPOCHS, "lr": LR, "lambda": LAM,
                    "seeds": SEEDS, "model": "BiGRU(64,128)"},
        "results": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "seeds": v}
                   for k, v in results.items()},
        "per_predicate": per_pred,
    }
    
    # Build markdown
    md = ["# research Cross-Predicate Equivariance Comparison\n"]
    md.append(f"- Train preds: {TRAIN_P}  |  Held preds: {HELD_P}")
    md.append(f"- {len(tr_fams)} train fams, {len(he_fams)} held fams, {len(eq_pairs)} equiv pairs")
    md.append(f"- GRU(embed=48, hidden=96), {EPOCHS} epochs, lr={LR}, λ={LAM}\n")
    
    md.append("## Transfer comparison (mean ± std over 3 seeds)\n")
    md.append("| Condition | single_defeated | standard_multi | equivariant |")
    md.append("|---|---:|---:|---:|")
    for key in ["train", "he_fam_tr_pred", "cross_pred_tr_fam", "full_held", "held_ctx"]:
        vals = []
        for mode in ["single_defeated", "standard_multi", "equivariant"]:
            r = results.get(f"{mode}_{key}", [0])
            vals.append(f"{np.mean(r):.4f}±{np.std(r):.4f}")
        label = {"train": "Train", "he_fam_tr_pred": "Held fam, train pred",
                "cross_pred_tr_fam": "**Cross-pred (train fam, held hyp)**",
                "full_held": "**Full held**",
                "held_ctx": "Held ctx, train hyp"}.get(key, key)
        md.append(f"| {label} | {' | '.join(vals)} |")
    
    md.append("\n## Per-held-predicate breakdown (seed 42)\n")
    md.append("| Predicate | standard | equivariant |")
    md.append("|---|---:|---:|")
    for pk in HELD_P:
        s = per_pred.get(f"standard_multi_{pk}", 0)
        e = per_pred.get(f"equivariant_{pk}", 0)
        md.append(f"| {pk} ({PREDS[pk][1]}) | {s:.4f} | {e:.4f} |")
    
    # Key deltas
    cp_std = np.mean(results.get("standard_multi_cross_pred_tr_fam", [0]))
    cp_eq  = np.mean(results.get("equivariant_cross_pred_tr_fam", [0]))
    fh_std = np.mean(results.get("standard_multi_full_held", [0]))
    fh_eq  = np.mean(results.get("equivariant_full_held", [0]))
    
    md.append(f"\n## Key deltas")
    md.append(f"- Cross-pred (equiv − std): {cp_eq - cp_std:+.4f}")
    md.append(f"- Full held (equiv − std): {fh_eq - fh_std:+.4f}")
    
    with open(OUT_DIR / "cross_predicate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/cross_predicate_equivariance/cross_predicate_summary.md'), "w") as f:
        f.write("\n".join(md) + "\n")
    
    print(f"\nSaved to {OUT_DIR}/")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
