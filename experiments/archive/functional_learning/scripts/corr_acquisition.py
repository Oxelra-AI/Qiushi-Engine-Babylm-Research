#!/usr/bin/env python3
"""research: Correspondence acquisition under identity exposure.

Scientific question: Does identity exposure help, hinder, or leave unchanged
the sample efficiency of learning a reusable correspondence, when all arms
share the same backbone correspondence evidence?

Design:
  8 entities (E0-E7), 10 shared attributes
  Systematic correspondence: rewrite_i = f(source_i) for all i
  Train queries: E0-E5; held-out queries: E6,E7 (never queried in training)
  All entities appear in training context (embeddings learned).

  Each arm trains on 500 seqs/epoch: 100 shared backbone + 400 substitution.
  Backbone: correspondence (target = rewrite token)
  Substitution differs by arm:
    all_corr:      400 more correspondence (upper bound, 500 total)
    ident_full:    400 identity (target = source token, full attention)
    ident_masked:  400 identity (target = source, retrieval blocked)
    neutral:       400 shuffled-target (no learnable relation)
    wrong:         400 wrong-correspondence (active interference)

Evaluation (every 10 epochs): held-out entity correspondence NLL.
  corr_nll_src:   NLL of correct rewrite when source entity is in context
  corr_nll_nosrc: NLL of correct rewrite when source entity NOT in context
  corr_gain:      nosrc - src (benefit of having retrievable source)
  copy_nll_src:   NLL of source token at target position (copy tendency)

Decisive contrast: ident_full vs neutral on held_corr_gain learning curves.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, math, time, argparse
from pathlib import Path

# ─── Vocabulary ────────────────────────────────────────────────────
BOS, PAD, SEP, HAS, IS = 0, 1, 2, 3, 4
N_ENT, N_ATTR = 8, 10
ENT  = lambda e: 5 + e          # 5..12
SRC  = lambda a: 13 + a         # 13..22
RWT  = lambda a: 23 + a         # 23..32
VOCAB = 33
TRAIN_E = list(range(6))
HELD_E  = [6, 7]
ALL_E   = list(range(N_ENT))
# Sequence: BOS E HAS S E HAS S E HAS S E HAS S SEP Eq IS TGT PAD
# Pos:       0  1  2  3 4  5  6 7  8  9 10 11 12 13 14 15  16  17
SEQ_LEN, IS_POS, TGT_POS, N_CTX = 18, 15, 16, 4

def mk_seq(ce, ca, qi, tgt):
    s = [BOS]
    for i in range(N_CTX):
        s += [ENT(ce[i]), HAS, SRC(ca[i])]
    s += [SEP, ENT(ce[qi]), IS, tgt]
    return (s + [PAD]*SEQ_LEN)[:SEQ_LEN]

# ─── Data generation ───────────────────────────────────────────────
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

    def gen(self, n, kind, qpool=TRAIN_E):
        seqs, infos = [], []
        for _ in range(n):
            ce, ca, qi = self._ctx(qpool)
            a = ca[qi]
            if kind == "corr":
                tgt = RWT(a)
            elif kind == "ident":
                tgt = SRC(a)
            elif kind == "neutral":
                tgt = RWT(int(self.rng.integers(N_ATTR)))
            elif kind == "wrong":
                wa = a
                while wa == a: wa = int(self.rng.integers(N_ATTR))
                tgt = RWT(wa)
            else:
                raise ValueError(kind)
            seqs.append(mk_seq(ce, ca, qi, tgt))
            infos.append({"t": kind, "sp": 3+qi*3})
        return seqs, infos

    def held_probes(self, n_per=60):
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
                ssrc = mk_seq(ce, ca, qi, PAD)
                # no-source: he not in context
                nce = self.rng.choice(TRAIN_E, N_CTX, replace=False).tolist()
                nca = [int(self.rng.integers(N_ATTR)) for _ in range(N_CTX)]
                ns = [BOS]
                for i in range(N_CTX):
                    ns += [ENT(nce[i]), HAS, SRC(nca[i])]
                ns += [SEP, ENT(he), IS, PAD]
                ns = (ns+[PAD]*SEQ_LEN)[:SEQ_LEN]
                out.append({"ss":ssrc,"ns":ns,"ct":RWT(a),"cp":SRC(a),"sp":3+qi*3})
        return out

    def train_probes(self, n_per=30):
        out = []
        for te in TRAIN_E:
            for _ in range(n_per):
                a = int(self.rng.integers(N_ATTR))
                oth = self.rng.choice([e for e in ALL_E if e!=te], N_CTX-1, replace=False).tolist()
                ce = oth + [te]
                ce = self.rng.permutation(ce).tolist()
                ca, qi = [], -1
                for i,e in enumerate(ce):
                    if e == te: ca.append(a); qi=i
                    else: ca.append(int(self.rng.integers(N_ATTR)))
                out.append({"ss":mk_seq(ce,ca,qi,PAD),"ct":RWT(a),"cp":SRC(a)})
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
        self.hd = nn.Linear(V, d, bias=False)   # transposed for tied weights
        self.nh = nh
    def forward(self, x, m=None):
        B,L = x.shape
        h = self.tok(x) + self.pos(torch.arange(L,device=x.device))
        for b in self.blks: h = b(h, m)
        h = self.ln(h)
        return h @ self.tok.weight.T  # tied: logits = h @ E^T

# ─── Attention masks ──────────────────────────────────────────────
def causal(L, dev="cpu"):
    return torch.triu(torch.ones(L,L,dtype=torch.bool,device=dev), diagonal=1)

def build_masks_3d(all_infos, L, nh):
    """Per-example 3D masks for ident_masked arm.
    Block positions {14,15} from attending to src_attr_pos for identity seqs."""
    N = len(all_infos)
    cm = causal(L)
    masks = cm.unsqueeze(0).expand(N,-1,-1).clone()
    for i, info in enumerate(all_infos):
        if info["t"] == "ident":
            sp = info["sp"]
            if sp < L:
                # Block query entity pos (14) and IS pos (15) from source attr
                for bp in [14, 15]:
                    if bp < L:
                        masks[i, bp, sp] = True
    return masks

# ─── Training ─────────────────────────────────────────────────────
def train_ep(model, opt, seqs_t, masks_3d, dev, nh, bs=64):
    model.train()
    N = seqs_t.size(0)
    idx = torch.randperm(N)
    seqs_t = seqs_t[idx]
    if masks_3d is not None:
        masks_3d = masks_3d[idx]
    tot_loss, tot_tok = 0., 0
    L = SEQ_LEN - 1
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
        loss = F.cross_entropy(logits.reshape(-1,VOCAB), tgt.reshape(-1), reduction='none')
        loss = (loss.view(tgt.shape)*lm).sum() / max(lm.sum(),1)
        opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item()*lm.sum().item()
        tot_tok  += lm.sum().item()
    return tot_loss / max(tot_tok,1)

def eval_probes(model, probes, dev, nh, src=True):
    model.eval()
    key = "ss" if src else "ns"
    seqs = [p[key] for p in probes if key in p]
    tgts = [(p["ct"], p["cp"]) for p in probes if key in p]
    if not seqs: return float('nan'), float('nan')
    st = torch.tensor(seqs, dtype=torch.long, device=dev)
    L = SEQ_LEN - 1
    cm = causal(L, dev)
    corrs, copies = [], []
    with torch.no_grad():
        for i in range(0, len(seqs), 128):
            b = st[i:i+128,:L]
            logits = model(b, cm)
            lp = F.log_softmax(logits[:,IS_POS,:], dim=-1)
            for j in range(b.size(0)):
                ct, cp = tgts[i+j]
                corrs.append(-lp[j,ct].item())
                copies.append(-lp[j,cp].item())
    return float(np.mean(corrs)), float(np.mean(copies))

# ─── Arms configuration ──────────────────────────────────────────
ARMS = ["all_corr","ident_full","ident_masked","neutral","wrong"]
SUB_KINDS = {"all_corr":"corr","ident_full":"ident","ident_masked":"ident",
             "neutral":"neutral","wrong":"wrong"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42,43,100])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--backbone_n", type=int, default=100)
    ap.add_argument("--sub_n", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nh", type=int, default=2)
    ap.add_argument("--nl", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="data/corr_acquisition")
    ap.add_argument("--smoke", action="store_true")
    A = ap.parse_args()
    if A.smoke:
        A.seeds=[42]; A.epochs=5; A.backbone_n=20; A.sub_n=40; A.eval_every=5

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = Path(A.out); od.mkdir(parents=True, exist_ok=True)
    print(f"Device: {dev}, Vocab: {VOCAB}, SEQ_LEN: {SEQ_LEN}")

    all_res = {}
    for seed in A.seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")

        # Shared backbone (identical across arms)
        dg_bb = DG(seed)
        bb_seqs, bb_infos = dg_bb.gen(A.backbone_n, "corr")
        hprobes = dg_bb.held_probes(n_per=60)
        tprobes = dg_bb.train_probes(n_per=30)
        print(f"Backbone: {len(bb_seqs)}, held probes: {len(hprobes)}, train probes: {len(tprobes)}")

        # Pre-generate substitution data (ident_full & ident_masked share same data)
        sub_data = {}
        dg_id = DG(seed*1000 + 1)
        id_seqs, id_infos = dg_id.gen(A.sub_n, "ident")
        sub_data["ident_full"]   = (id_seqs, id_infos)
        sub_data["ident_masked"] = (id_seqs, id_infos)  # same data!
        dg_corr = DG(seed*1000 + 2)
        sub_data["all_corr"] = dg_corr.gen(A.sub_n, "corr")
        dg_neut = DG(seed*1000 + 3)
        sub_data["neutral"] = dg_neut.gen(A.sub_n, "neutral")
        dg_wr = DG(seed*1000 + 4)
        sub_data["wrong"] = dg_wr.gen(A.sub_n, "wrong")

        # Shared initial model weights
        torch.manual_seed(seed)
        model0 = CLM(VOCAB, A.d, A.nh, A.nl, SEQ_LEN).to(dev)
        init_sd = {k:v.clone() for k,v in model0.state_dict().items()}
        n_params = sum(p.numel() for p in model0.parameters())
        print(f"Params: {n_params:,}")

        # Verify ident data identity
        assert sub_data["ident_full"][0] == sub_data["ident_masked"][0], \
            "ident_full and ident_masked must have identical data"
        print("✓ ident_full/ident_masked data identical")

        seed_res = {}
        for arm in ARMS:
            t0 = time.time()
            print(f"\n--- {arm} ---")

            # Combine backbone + substitution
            s_seqs, s_infos = sub_data[arm]
            all_seqs = bb_seqs + s_seqs
            all_infos = bb_infos + s_infos
            seqs_t = torch.tensor(all_seqs, dtype=torch.long)
            L = SEQ_LEN - 1

            # 3D masks for ident_masked
            m3d = None
            if arm == "ident_masked":
                m3d = build_masks_3d(all_infos, L, A.nh)
                n_blocked = sum(1 for info in all_infos if info["t"]=="ident")
                print(f"  3D masks: {n_blocked}/{len(all_infos)} ident seqs masked")

            # Reset model
            model0.load_state_dict(init_sd)
            opt = torch.optim.AdamW(model0.parameters(), lr=A.lr)

            curves = []
            for ep in range(1, A.epochs+1):
                loss = train_ep(model0, opt, seqs_t, m3d, dev, A.nh)
                if ep % A.eval_every == 0 or ep == 1:
                    hc_s, hcp_s = eval_probes(model0, hprobes, dev, A.nh, src=True)
                    hc_n, hcp_n = eval_probes(model0, hprobes, dev, A.nh, src=False)
                    tc_s, tcp_s = eval_probes(model0, tprobes, dev, A.nh, src=True)
                    row = {
                        "e": ep, "tl": round(loss,5),
                        "hc_s": round(hc_s,5),   # held corr nll, source present
                        "hcp_s": round(hcp_s,5),  # held copy nll, source present
                        "hc_n": round(hc_n,5),   # held corr nll, no source
                        "hcg": round(hc_n-hc_s,5), # held corr gain (source benefit)
                        "tc_s": round(tc_s,5),   # train corr nll, source present
                        "tcp_s": round(tcp_s,5),  # train copy nll
                    }
                    curves.append(row)
                    if ep <= 30 or ep % 50 == 0 or ep == A.epochs:
                        print(json.dumps(row))

            elapsed = time.time()-t0
            final = curves[-1] if curves else {}
            seed_res[arm] = {"curves":curves,"final":final,"elapsed":round(elapsed,1)}
            print(f"  → {arm}: hcg={final.get('hcg','?'):+.4f} tc_s={final.get('tc_s','?'):.4f} ({elapsed:.1f}s)")

        all_res[f"seed{seed}"] = seed_res
        # Save per-seed
        sd = od/f"seed{seed}"; sd.mkdir(exist_ok=True)
        with open(sd/"results.json","w") as f: json.dump(seed_res,f,indent=2)

    # ─── Aggregate ─────────────────────────────────────────────────
    summary = {"arms":{}, "config": {"backbone_n":A.backbone_n,"sub_n":A.sub_n,
                "epochs":A.epochs,"d":A.d,"nh":A.nh,"nl":A.nl,"lr":A.lr,
                "seeds":A.seeds, "vocab":VOCAB}}
    for arm in ARMS:
        finals = [all_res[f"seed{s}"][arm]["final"] for s in A.seeds]
        avg = {}
        for k in finals[0]:
            if k == "e": avg[k]=finals[0][k]; continue
            vals = [f[k] for f in finals]
            avg[k] = round(np.mean(vals),5)
            avg[k+"_std"] = round(np.std(vals),5)
        summary["arms"][arm] = avg

    # Key contrasts
    def gf(arm, met): return [all_res[f"seed{s}"][arm]["final"][met] for s in A.seeds]

    contrasts = {}
    for a1,a2,name in [
        ("ident_full","neutral","ident_vs_neutral"),
        ("ident_full","wrong","ident_vs_wrong"),
        ("ident_full","ident_masked","full_vs_masked"),
        ("all_corr","ident_full","allcorr_vs_ident"),
        ("all_corr","neutral","allcorr_vs_neutral"),
    ]:
        for met in ["hcg","hc_s","tc_s"]:
            v1, v2 = gf(a1,met), gf(a2,met)
            d = np.array(v1)-np.array(v2)
            contrasts[f"{name}_{met}"] = {
                f"{a1}": round(np.mean(v1),5),
                f"{a2}": round(np.mean(v2),5),
                "delta": round(np.mean(d),5),
                "delta_std": round(np.std(d),5),
            }
    summary["contrasts"] = contrasts

    with open(od/"summary.json","w") as f:
        json.dump({"summary":summary,"all_results":all_res}, f, indent=2)

    # ─── Print ─────────────────────────────────────────────────────
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for arm, avg in summary["arms"].items():
        print(f"  {arm:15s}  hcg={avg.get('hcg','?'):+.5f}±{avg.get('hcg_std',0):.5f}  "
              f"hc_s={avg.get('hc_s','?'):.4f}  tc_s={avg.get('tc_s','?'):.4f}")
    print()
    for name, c in contrasts.items():
        if "hcg" in name:
            k1,k2 = [k for k in c if k not in ("delta","delta_std")]
            print(f"  {name}: {k1}={c[k1]:+.5f} {k2}={c[k2]:+.5f} Δ={c['delta']:+.5f}±{c['delta_std']:.5f}")

    print(json.dumps({"status":"DONE","out":str(od)}))

if __name__=="__main__":
    main()
