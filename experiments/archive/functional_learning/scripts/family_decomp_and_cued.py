#!/usr/bin/env python3
"""research: Family decomposition and task-cued experiment.

Resolves whether research's ident>neutral hcg contrast is:
  (a) Genuine facilitation of attribute identification, or
  (b) A task-ambiguity artifact (mixture changes expected output family)

Part A: NLL decomposition
  For each eval probe, decompose -log p(r_i) into:
    -log p(family=RWT) + -log p(r_i | family=RWT)
  Track p_RWT, p_SRC, within-family identification across training.

Part B: Task-cued experiment
  Add explicit CUE token (COPY_CUE or REWRITE_CUE) at position 15.
  Training: correspondence→REWRITE_CUE, identity→COPY_CUE
  Evaluation: always REWRITE_CUE for correspondence probes.
  This removes the output-family ambiguity. If facilitation survives,
  it's genuine representational transfer, not task confusion.

Key predictions:
  If task ambiguity drives research → cued ident≈neutral, decomp shows
    ident reduces p_RWT but doesn't improve within-family identification.
  If genuine facilitation → cued ident>neutral on held correspondence,
    decomp shows ident improves within-family identification.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, argparse
from pathlib import Path

# ─── Shared vocab from research ──────────────────────────────────────
BOS, PAD, SEP, HAS, IS = 0, 1, 2, 3, 4
N_ENT, N_ATTR = 8, 10
ENT  = lambda e: 5 + e
SRC  = lambda a: 13 + a
RWT  = lambda a: 23 + a
VOCAB_BASE = 33  # original research

# New cue tokens for Part B
COPY_CUE    = 33
REWRITE_CUE = 34
VOCAB_CUED  = 35

TRAIN_E = list(range(6))
HELD_E  = [6, 7]
ALL_E   = list(range(N_ENT))
N_CTX   = 4

# ─── Part A: uncued (same as research, with decomposition) ──────────
# Sequence: BOS E HAS S E HAS S E HAS S E HAS S SEP Eq IS TGT PAD
# Pos:       0  1  2  3 4  5  6 7  8  9 10 11 12 13 14 15  16  17
SEQLEN_A  = 18
IS_POS_A  = 15
TGT_POS_A = 16

# ─── Part B: cued
# Sequence: BOS E HAS S E HAS S E HAS S E HAS S SEP Eq CUE IS TGT PAD
# Pos:       0  1  2  3 4  5  6 7  8  9 10 11 12 13 14  15 16  17  18
SEQLEN_B  = 19
CUE_POS_B = 15
IS_POS_B  = 16
TGT_POS_B = 17

SRC_TOKS = list(range(13, 23))  # s0..s9
RWT_TOKS = list(range(23, 33))  # r0..r9

def mk_seq_a(ce, ca, qi, tgt):
    s = [BOS]
    for i in range(N_CTX):
        s += [ENT(ce[i]), HAS, SRC(ca[i])]
    s += [SEP, ENT(ce[qi]), IS, tgt]
    return (s + [PAD]*SEQLEN_A)[:SEQLEN_A]

def mk_seq_b(ce, ca, qi, cue, tgt):
    s = [BOS]
    for i in range(N_CTX):
        s += [ENT(ce[i]), HAS, SRC(ca[i])]
    s += [SEP, ENT(ce[qi]), cue, IS, tgt]
    return (s + [PAD]*SEQLEN_B)[:SEQLEN_B]

# ─── Data generator ───────────────────────────────────────────────
class DG:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def _ctx(self, qpool=TRAIN_E):
        while True:
            ce = self.rng.choice(ALL_E, N_CTX, replace=False).tolist()
            cands = [i for i,e in enumerate(ce) if e in qpool]
            if cands:
                qi = int(self.rng.choice(cands))
                ca = [int(self.rng.integers(N_ATTR)) for _ in range(N_CTX)]
                return ce, ca, qi

    def gen(self, n, kind, part="A"):
        seqs, infos = [], []
        for _ in range(n):
            ce, ca, qi = self._ctx(TRAIN_E)
            a = ca[qi]
            if kind == "corr":
                tgt = RWT(a)
                cue = REWRITE_CUE
            elif kind == "ident":
                tgt = SRC(a)
                cue = COPY_CUE
            elif kind == "neutral":
                tgt = RWT(int(self.rng.integers(N_ATTR)))
                cue = REWRITE_CUE
            elif kind == "wrong":
                wa = a
                while wa == a: wa = int(self.rng.integers(N_ATTR))
                tgt = RWT(wa)
                cue = REWRITE_CUE
            else:
                raise ValueError(kind)
            if part == "A":
                seqs.append(mk_seq_a(ce, ca, qi, tgt))
            else:
                seqs.append(mk_seq_b(ce, ca, qi, cue, tgt))
            infos.append({"t": kind, "sp": 3+qi*3})
        return seqs, infos

    def held_probes(self, n_per=60, part="A"):
        out = []
        for he in HELD_E:
            for _ in range(n_per):
                a = int(self.rng.integers(N_ATTR))
                oth = self.rng.choice(TRAIN_E, N_CTX-1, replace=False).tolist()
                ce = oth + [he]
                ce = self.rng.permutation(ce).tolist()
                ca, qi = [], -1
                for i,e in enumerate(ce):
                    if e == he: ca.append(a); qi=i
                    else: ca.append(int(self.rng.integers(N_ATTR)))
                if part == "A":
                    ssrc = mk_seq_a(ce, ca, qi, PAD)
                else:
                    ssrc = mk_seq_b(ce, ca, qi, REWRITE_CUE, PAD)
                # no-source version
                nce = self.rng.choice(TRAIN_E, N_CTX, replace=False).tolist()
                nca = [int(self.rng.integers(N_ATTR)) for _ in range(N_CTX)]
                ns = [BOS]
                for i in range(N_CTX):
                    ns += [ENT(nce[i]), HAS, SRC(nca[i])]
                if part == "A":
                    ns += [SEP, ENT(he), IS, PAD]
                    ns = (ns+[PAD]*SEQLEN_A)[:SEQLEN_A]
                else:
                    ns += [SEP, ENT(he), REWRITE_CUE, IS, PAD]
                    ns = (ns+[PAD]*SEQLEN_B)[:SEQLEN_B]
                out.append({"ss": ssrc, "ns": ns, "ct": RWT(a), "cp": SRC(a), "sp": 3+qi*3})
        return out

# ─── Model ─────────────────────────────────────────────────────────
class Blk(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0)
        self.ln1 = nn.LayerNorm(d)
        self.ff  = nn.Sequential(nn.Linear(d,4*d), nn.GELU(), nn.Linear(4*d,d))
        self.ln2 = nn.LayerNorm(d)
    def forward(self, x, m):
        h = self.ln1(x)
        h,_ = self.attn(h,h,h, attn_mask=m)
        x = x + h
        return x + self.ff(self.ln2(x))

class CLM(nn.Module):
    def __init__(self, V, d, nh, nl, ml):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ml, d)
        self.blks = nn.ModuleList([Blk(d,nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.nh = nh
    def forward(self, x, m=None):
        B,L = x.shape
        h = self.tok(x) + self.pos(torch.arange(L,device=x.device))
        for b in self.blks: h = b(h, m)
        h = self.ln(h)
        return h @ self.tok.weight.T

def causal(L, dev="cpu"):
    return torch.triu(torch.ones(L,L,dtype=torch.bool,device=dev), diagonal=1)

def build_masks_3d(all_infos, SL, nh):
    N = len(all_infos)
    L = SL - 1
    cm = causal(L)
    masks = cm.unsqueeze(0).expand(N,-1,-1).clone()
    for i, info in enumerate(all_infos):
        if info["t"] == "ident":
            sp = info["sp"]
            if sp < L:
                for bp in [14, 15]:
                    if bp < L:
                        masks[i, bp, sp] = True
    return masks

def build_masks_3d_b(all_infos, SL, nh):
    """For part B (cued), block IS_POS_B-1 and IS_POS_B from src attr."""
    N = len(all_infos)
    L = SL - 1
    cm = causal(L)
    masks = cm.unsqueeze(0).expand(N,-1,-1).clone()
    for i, info in enumerate(all_infos):
        if info["t"] == "ident":
            sp = info["sp"]
            if sp < L:
                # Block cue_pos(15), IS_pos(16) from source attr
                for bp in [CUE_POS_B, IS_POS_B]:
                    if bp < L:
                        masks[i, bp, sp] = True
    return masks

# ─── Training ─────────────────────────────────────────────────────
def train_ep(model, opt, seqs_t, masks_3d, dev, nh, V, bs=64):
    model.train()
    N = seqs_t.size(0)
    L = seqs_t.size(1) - 1
    idx = torch.randperm(N)
    seqs_t = seqs_t[idx]
    if masks_3d is not None:
        masks_3d = masks_3d[idx]
    tot_loss, tot_tok = 0., 0
    for i in range(0, N, bs):
        batch = seqs_t[i:i+bs].to(dev)
        inp, tgt = batch[:,:-1], batch[:,1:]
        bsz = inp.size(0)
        if masks_3d is not None:
            m = masks_3d[i:i+bsz,:L,:L].to(dev)
            m = m.unsqueeze(1).expand(-1,nh,-1,-1).reshape(bsz*nh,L,L)
        else:
            m = causal(L, dev)
        logits = model(inp, m)
        lm = (tgt != PAD).float()
        loss = F.cross_entropy(logits.reshape(-1,V), tgt.reshape(-1), reduction='none')
        loss = (loss.view(tgt.shape)*lm).sum() / max(lm.sum(),1)
        opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item()*lm.sum().item()
        tot_tok  += lm.sum().item()
    return tot_loss / max(tot_tok,1)

# ─── Evaluation with family decomposition ─────────────────────────
def eval_decomposed(model, probes, dev, nh, V, is_pos, src=True):
    """Returns dict with nll_corr, nll_copy, p_rwt, p_src, 
    within_rwt_nll, family_rwt_nll for family decomposition."""
    model.eval()
    key = "ss" if src else "ns"
    seqs = [p[key] for p in probes if key in p]
    tgts = [(p["ct"], p["cp"]) for p in probes if key in p]
    if not seqs:
        return {}
    
    st = torch.tensor(seqs, dtype=torch.long, device=dev)
    L = st.size(1) - 1
    cm = causal(L, dev)
    
    all_corr_nll = []
    all_copy_nll = []
    all_p_rwt = []
    all_p_src = []
    all_within_rwt = []
    all_family_rwt = []
    
    with torch.no_grad():
        for i in range(0, len(seqs), 128):
            b = st[i:i+128,:L]
            logits = model(b, cm)
            probs = F.softmax(logits[:, is_pos, :], dim=-1)  # [B, V]
            lp = F.log_softmax(logits[:, is_pos, :], dim=-1)
            
            for j in range(b.size(0)):
                ct, cp = tgts[i+j]
                corr_nll = -lp[j, ct].item()
                copy_nll = -lp[j, cp].item()
                
                # Family probabilities
                p_rwt = probs[j, RWT_TOKS].sum().item()
                p_src = probs[j, SRC_TOKS].sum().item()
                
                # Within-family identification
                # p(r_i | family=RWT) = p(r_i) / p(family=RWT)
                if p_rwt > 1e-10:
                    p_ri_given_rwt = probs[j, ct].item() / p_rwt
                    within_rwt_nll = -np.log(max(p_ri_given_rwt, 1e-30))
                else:
                    within_rwt_nll = 30.0  # cap
                
                # Family NLL component: -log p(family=RWT)
                family_rwt_nll = -np.log(max(p_rwt, 1e-30))
                
                all_corr_nll.append(corr_nll)
                all_copy_nll.append(copy_nll)
                all_p_rwt.append(p_rwt)
                all_p_src.append(p_src)
                all_within_rwt.append(within_rwt_nll)
                all_family_rwt.append(family_rwt_nll)
    
    return {
        "corr_nll": float(np.mean(all_corr_nll)),
        "copy_nll": float(np.mean(all_copy_nll)),
        "p_rwt": float(np.mean(all_p_rwt)),
        "p_src": float(np.mean(all_p_src)),
        "within_rwt_nll": float(np.mean(all_within_rwt)),
        "family_rwt_nll": float(np.mean(all_family_rwt)),
    }

# ─── Arms ──────────────────────────────────────────────────────────
ARMS = ["all_corr", "ident_full", "ident_masked", "neutral", "wrong"]

def run_experiment(part, seed, args, dev):
    """Run one seed for one part (A=uncued with decomposition, B=cued)."""
    V = VOCAB_BASE if part == "A" else VOCAB_CUED
    SL = SEQLEN_A if part == "A" else SEQLEN_B
    IS_POS = IS_POS_A if part == "A" else IS_POS_B

    dg_bb = DG(seed)
    bb_seqs, bb_infos = dg_bb.gen(args.backbone_n, "corr", part)
    hprobes = dg_bb.held_probes(n_per=60, part=part)

    sub_data = {}
    dg_id = DG(seed*1000 + 1)
    id_seqs, id_infos = dg_id.gen(args.sub_n, "ident", part)
    sub_data["ident_full"]   = (id_seqs, id_infos)
    sub_data["ident_masked"] = (id_seqs, id_infos)
    dg_corr = DG(seed*1000 + 2)
    sub_data["all_corr"] = dg_corr.gen(args.sub_n, "corr", part)
    dg_neut = DG(seed*1000 + 3)
    sub_data["neutral"] = dg_neut.gen(args.sub_n, "neutral", part)
    dg_wr = DG(seed*1000 + 4)
    sub_data["wrong"] = dg_wr.gen(args.sub_n, "wrong", part)

    torch.manual_seed(seed)
    model = CLM(V, args.d, args.nh, args.nl, SL).to(dev)
    init_sd = {k:v.clone() for k,v in model.state_dict().items()}

    results = {}
    for arm in ARMS:
        s_seqs, s_infos = sub_data[arm]
        all_seqs = bb_seqs + s_seqs
        all_infos = bb_infos + s_infos
        seqs_t = torch.tensor(all_seqs, dtype=torch.long)
        L = SL - 1

        m3d = None
        if arm == "ident_masked":
            if part == "A":
                m3d = build_masks_3d(all_infos, SL, args.nh)
            else:
                m3d = build_masks_3d_b(all_infos, SL, args.nh)

        model.load_state_dict(init_sd)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        curves = []
        for ep in range(1, args.epochs+1):
            tl = train_ep(model, opt, seqs_t, m3d, dev, args.nh, V)
            if ep % args.eval_every == 0 or ep == 1:
                d_s = eval_decomposed(model, hprobes, dev, args.nh, V, IS_POS, src=True)
                d_n = eval_decomposed(model, hprobes, dev, args.nh, V, IS_POS, src=False)
                row = {
                    "e": ep, "tl": round(tl, 5),
                    # Standard metrics
                    "hc_s": round(d_s["corr_nll"], 5),
                    "hcp_s": round(d_s["copy_nll"], 5),
                    "hc_n": round(d_n["corr_nll"], 5),
                    "hcg": round(d_n["corr_nll"] - d_s["corr_nll"], 5),
                    # Family decomposition (src present)
                    "p_rwt_s": round(d_s["p_rwt"], 5),
                    "p_src_s": round(d_s["p_src"], 5),
                    "within_s": round(d_s["within_rwt_nll"], 5),
                    "family_s": round(d_s["family_rwt_nll"], 5),
                    # Family decomposition (no src)
                    "p_rwt_n": round(d_n["p_rwt"], 5),
                    "within_n": round(d_n["within_rwt_nll"], 5),
                    "family_n": round(d_n["family_rwt_nll"], 5),
                }
                curves.append(row)
        results[arm] = curves
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 100])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--backbone_n", type=int, default=100)
    ap.add_argument("--sub_n", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nh", type=int, default=2)
    ap.add_argument("--nl", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="data/decomp_cued")
    ap.add_argument("--smoke", action="store_true")
    A = ap.parse_args()
    if A.smoke:
        A.seeds = [42]; A.epochs = 10; A.backbone_n = 20; A.sub_n = 40; A.eval_every = 5

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = Path(A.out); od.mkdir(parents=True, exist_ok=True)

    all_data = {"config": vars(A), "part_A": {}, "part_B": {}}

    for seed in A.seeds:
        print(f"\n{'='*60}\nSeed {seed} PART A (uncued + decomposition)\n{'='*60}")
        all_data["part_A"][f"seed{seed}"] = run_experiment("A", seed, A, dev)

        print(f"\n{'='*60}\nSeed {seed} PART B (task-cued)\n{'='*60}")
        all_data["part_B"][f"seed{seed}"] = run_experiment("B", seed, A, dev)

    # ─── Summary extraction ────────────────────────────────────────
    summary = {"part_A": {}, "part_B": {}}
    for part in ["part_A", "part_B"]:
        for arm in ARMS:
            finals = []
            for seed in A.seeds:
                cs = all_data[part][f"seed{seed}"][arm]
                finals.append(cs[-1])
            avg = {}
            for k in finals[0]:
                if k == "e": avg[k] = finals[0][k]; continue
                vals = [f[k] for f in finals]
                avg[k] = round(np.mean(vals), 5)
                avg[k+"_std"] = round(np.std(vals), 5)
            summary[part][arm] = avg

    # Key contrasts
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for part_name, label in [("part_A", "UNCUED"), ("part_B", "CUED")]:
        print(f"\n--- {label} ---")
        for arm in ARMS:
            a = summary[part_name][arm]
            print(f"  {arm:15s}  hcg={a['hcg']:+.4f}±{a['hcg_std']:.4f}  "
                  f"hc_s={a['hc_s']:.3f}  p_rwt_s={a['p_rwt_s']:.4f}  "
                  f"within_s={a['within_s']:.3f}  family_s={a['family_s']:.3f}")

        # Ident vs neutral
        for m in ["hcg", "hc_s", "within_s", "family_s", "p_rwt_s"]:
            iv = [all_data[part_name][f"seed{s}"]["ident_full"][-1][m] for s in A.seeds]
            nv = [all_data[part_name][f"seed{s}"]["neutral"][-1][m] for s in A.seeds]
            d = np.array(iv) - np.array(nv)
            print(f"  ident-neutral {m}: {np.mean(d):+.4f} ± {np.std(d):.4f}")

    all_data["summary"] = summary
    with open(od / "results.json", "w") as f:
        json.dump(all_data, f, indent=2)

    print(json.dumps({"status": "DONE", "out": str(od)}))

if __name__ == "__main__":
    main()
