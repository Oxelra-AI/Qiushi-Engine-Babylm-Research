#!/usr/bin/env python3
"""Step014b: Capacity sweep for entity-attribute binding.

research showed that a 3-layer, 2-head, d=64 Transformer (153K params) defaults to
bag-level representation. This script tests whether larger models can learn entity-
specific binding from the same BOUND training signal.

Configurations (BOUND arm only):
  small:  d=64,  nh=2, nl=3  (153K params, research baseline)
  wide:   d=128, nh=4, nl=3  (~600K params)
  deep:   d=64,  nh=2, nl=6  (~300K params)
  large:  d=128, nh=4, nl=6  (~1.2M params)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, argparse
from pathlib import Path

# ─── Constants (same as research) ──────────────────────────────────
K, N_ENT, N_ATTR = 4, 10, 12
TRAIN_E = list(range(8)); HELD_E = [8, 9]
BOS, PAD, SEP, HAS, IS = 0, 1, 2, 3, 4
ENT  = lambda e: 5 + e
ATTR = lambda a: 15 + a
RWT  = lambda a: 27 + a
VOCAB = 39; RWT_TOKS = list(range(27, 39))
SL = 18; IS_POS = 15

def mk_seq(ce, ca, qi, tgt):
    s = [BOS]
    for i in range(K): s += [ENT(ce[i]), HAS, ATTR(ca[i])]
    s += [SEP, ENT(ce[qi]), IS, tgt, PAD]
    return s

class DG:
    def __init__(self, seed): self.rng = np.random.default_rng(seed)
    def _ctx(self):
        ce = self.rng.choice(TRAIN_E, K, replace=False).tolist()
        ca = self.rng.choice(N_ATTR, K, replace=False).tolist()
        return ce, ca, int(self.rng.integers(K))
    def bound(self, n):
        return [mk_seq(ce, ca, qi, RWT(ca[qi]))
                for ce, ca, qi in (self._ctx() for _ in range(n))]

# ─── Probes ───────────────────────────────────────────────────────
def gen_std_probes(rng, n):
    ps = []
    for _ in range(n):
        ce = rng.choice(TRAIN_E, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(rng.integers(K))
        ps.append(dict(s=mk_seq(ce, ca, qi, PAD), correct=ca[qi]))
    return ps

def gen_bswap(rng, n):
    ps = []
    for _ in range(n):
        ce = rng.choice(TRAIN_E, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(rng.integers(K))
        si = qi
        while si == qi: si = int(rng.integers(K))
        ca2 = list(ca); ca2[qi], ca2[si] = ca2[si], ca2[qi]
        ps.append(dict(s1=mk_seq(ce, ca, qi, PAD), s2=mk_seq(ce, ca2, qi, PAD),
                       c1=ca[qi], c2=ca2[qi]))
    return ps

def gen_qswap(rng, n):
    ps = []
    for _ in range(n):
        ce = rng.choice(TRAIN_E, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi1 = int(rng.integers(K)); qi2 = qi1
        while qi2 == qi1: qi2 = int(rng.integers(K))
        ps.append(dict(s1=mk_seq(ce, ca, qi1, PAD), s2=mk_seq(ce, ca, qi2, PAD),
                       c1=ca[qi1], c2=ca[qi2]))
    return ps

# ─── Model ────────────────────────────────────────────────────────
class Blk(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x, m):
        h = self.ln1(x); h, _ = self.attn(h, h, h, attn_mask=m)
        x = x + h; return x + self.ff(self.ln2(x))

class CLM(nn.Module):
    def __init__(self, V, d, nh, nl, ml):
        super().__init__()
        self.tok = nn.Embedding(V, d); self.pos = nn.Embedding(ml, d)
        self.blks = nn.ModuleList([Blk(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
    def forward(self, x, m=None):
        B, L = x.shape
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))
        for b in self.blks: h = b(h, m)
        return self.ln(h) @ self.tok.weight.T

def causal_mask(L, dev="cpu"):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), diagonal=1)

def train_ep(model, opt, seqs, dev, bs=64):
    model.train()
    st = torch.tensor(seqs, dtype=torch.long)
    N = st.size(0); idx = torch.randperm(N); st = st[idx]
    L = SL - 1; cm = causal_mask(L, dev)
    tl, tt = 0., 0
    for i in range(0, N, bs):
        batch = st[i:i+bs].to(dev)
        inp, tgt = batch[:, :L], batch[:, 1:]
        logits = model(inp, cm)
        mask = (tgt != PAD).float()
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction='none')
        loss = (loss.view(tgt.shape) * mask).sum() / max(mask.sum(), 1)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * mask.sum().item(); tt += mask.sum().item()
    return tl / max(tt, 1)

# ─── Evaluation ───────────────────────────────────────────────────
def _logits(model, seqs, dev):
    st = torch.tensor(seqs, dtype=torch.long, device=dev)
    L = SL - 1; cm = causal_mask(L, dev)
    out = []
    with torch.no_grad():
        for i in range(0, st.size(0), 256):
            out.append(model(st[i:i+256, :L], cm)[:, IS_POS, :])
    return torch.cat(out)

def eval_std(model, probes, dev):
    model.eval()
    lo = _logits(model, [p["s"] for p in probes], dev)
    p = F.softmax(lo, dim=-1); lp = F.log_softmax(lo, dim=-1)
    cnll, mrr = [], []
    for j, pr in enumerate(probes):
        c = pr["correct"]; ct = RWT(c)
        cnll.append(-lp[j, ct].item())
        rp = p[j, RWT_TOKS]
        rk = int((rp.argsort(descending=True) == c).nonzero(as_tuple=True)[0].item()) + 1
        mrr.append(1.0 / rk)
    return dict(cnll=round(float(np.mean(cnll)), 5), mrr=round(float(np.mean(mrr)), 5))

def eval_bswap(model, pairs, dev):
    model.eval()
    lo1 = _logits(model, [p["s1"] for p in pairs], dev)
    lo2 = _logits(model, [p["s2"] for p in pairs], dev)
    bs = []
    for j, pr in enumerate(pairs):
        ct1, ct2 = RWT(pr["c1"]), RWT(pr["c2"])
        m1 = lo1[j, ct1].item() - lo1[j, ct2].item()
        m2 = lo2[j, ct2].item() - lo2[j, ct1].item()
        bs.append(0.5 * (m1 + m2))
    bs = np.array(bs)
    return dict(mean=round(float(bs.mean()), 5), frac=round(float((bs > 0).mean()), 5))

def eval_qswap(model, probes, dev):
    model.eval()
    lo1 = _logits(model, [p["s1"] for p in probes], dev)
    lo2 = _logits(model, [p["s2"] for p in probes], dev)
    p1 = F.softmax(lo1, dim=-1); p2 = F.softmax(lo2, dim=-1)
    both = 0
    for j, pr in enumerate(probes):
        t1 = RWT_TOKS[p1[j, RWT_TOKS].argmax().item()]
        t2 = RWT_TOKS[p2[j, RWT_TOKS].argmax().item()]
        both += (t1 == RWT(pr["c1"])) and (t2 == RWT(pr["c2"]))
    return dict(both=round(both / len(probes), 5))

# ─── Main ─────────────────────────────────────────────────────────
CONFIGS = {
    "small":  dict(d=64,  nh=2, nl=3),   # research baseline
    "wide":   dict(d=128, nh=4, nl=3),   # wider only
    "deep":   dict(d=64,  nh=2, nl=6),   # deeper only
    "large":  dict(d=128, nh=4, nl=6),   # both larger
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--configs", default="small,wide,deep,large")
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/revision_014b_capacity")
    ap.add_argument("--fig", default="experiments/archive/functional_learning/figures/revision_014b_capacity.png")
    A = ap.parse_args()
    od = Path(A.data); od.mkdir(parents=True, exist_ok=True)
    cfgs = A.configs.split(",")

    seeds = [42] if A.smoke else [42, 43, 100]
    epochs = 10 if A.smoke else 300
    n_train = 100 if A.smoke else 500
    eval_every = 2 if A.smoke else 15

    erng = np.random.default_rng(9999)
    std_p = gen_std_probes(erng, 200)
    bswap_p = gen_bswap(erng, 200)
    qswap_p = gen_qswap(erng, 100)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dict(seeds=seeds, epochs=epochs, configs={})
    t0 = time.time()

    for cname in cfgs:
        if cname not in CONFIGS:
            print(f"Unknown config: {cname}"); continue
        c = CONFIGS[cname]
        print(f"\n{'='*60}\nConfig: {cname} (d={c['d']}, nh={c['nh']}, nl={c['nl']})\n{'='*60}")

        for sd in seeds:
            torch.manual_seed(sd); np.random.seed(sd)
            model = CLM(VOCAB, c["d"], c["nh"], c["nl"], SL).to(dev)
            npar = sum(p.numel() for p in model.parameters())
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
            dg = DG(sd)
            key = f"{cname}_seed{sd}"
            curves = []; ts = time.time()

            for ep in range(1, epochs + 1):
                tl = train_ep(model, opt, dg.bound(n_train), dev)
                if ep % eval_every == 0 or ep == 1:
                    m = dict(epoch=ep, tl=round(tl, 5),
                             std=eval_std(model, std_p, dev),
                             bswap=eval_bswap(model, bswap_p, dev),
                             qswap=eval_qswap(model, qswap_p, dev))
                    curves.append(m)
                    if ep == 1 or ep % (eval_every * 4) == 0 or ep == epochs:
                        print(f"  s{sd} e={ep:3d} tl={tl:.3f} cnll={m['std']['cnll']:.3f} "
                              f"bswap={m['bswap']['mean']:+.4f}({m['bswap']['frac']:.2f}) "
                              f"mrr={m['std']['mrr']:.3f} qs={m['qswap']['both']:.2f} "
                              f"({time.time()-ts:.0f}s)")

            data["configs"][key] = dict(curves=curves, params=npar,
                                        d=c["d"], nh=c["nh"], nl=c["nl"])
            print(f"  Params: {npar:,d}")

    with open(od / "results.json", "w") as f:
        json.dump(data, f, indent=2)

    # Summary
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for cname in cfgs:
        finals = []
        for sd in seeds:
            k = f"{cname}_seed{sd}"
            if k in data["configs"] and data["configs"][k]["curves"]:
                finals.append(data["configs"][k]["curves"][-1])
        if not finals: continue
        npar = data["configs"][f"{cname}_seed{seeds[0]}"]["params"]
        cnll = np.mean([f["std"]["cnll"] for f in finals])
        mrr = np.mean([f["std"]["mrr"] for f in finals])
        bs = np.mean([f["bswap"]["mean"] for f in finals])
        bf = np.mean([f["bswap"]["frac"] for f in finals])
        qs = np.mean([f["qswap"]["both"] for f in finals])
        print(f"  {cname:8s} ({npar:>8,d}p)  cnll={cnll:.3f}  mrr={mrr:.3f}  "
              f"bswap={bs:+.4f}({bf:.2f})  qswap={qs:.3f}")

    # Figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        colors = dict(small="tab:blue", wide="tab:orange", deep="tab:green", large="tab:red")
        for cname in cfgs:
            cc = [data["configs"][f"{cname}_seed{sd}"]["curves"]
                  for sd in seeds if f"{cname}_seed{sd}" in data["configs"]]
            if not cc: continue
            eps = [c["epoch"] for c in cc[0]]
            for ax_i, (key, title) in enumerate([
                ("cnll", "Correct NLL"), ("mean", "B_swap"), ("both", "Query-swap")
            ]):
                grp = "std" if key == "cnll" else ("bswap" if key == "mean" else "qswap")
                vals = [np.mean([c[ei][grp][key] for c in cc]) for ei in range(len(cc[0]))]
                npar = data["configs"][f"{cname}_seed{seeds[0]}"]["params"]
                axes[ax_i].plot(eps, vals, color=colors.get(cname, "gray"),
                               label=f"{cname} ({npar//1000}K)")
                axes[ax_i].set_title(title); axes[ax_i].set_xlabel("Epoch")
                axes[ax_i].grid(alpha=0.3)
        axes[0].axhline(np.log(4), color="gray", ls="--", alpha=0.4, label="bag-uniform")
        axes[1].axhline(0, color="gray", ls="--", alpha=0.5)
        for ax in axes: ax.legend(fontsize=7)
        fig.suptitle("Capacity sweep: BOUND arm only", fontsize=12)
        fig.tight_layout(); fig.savefig(A.fig, dpi=150, bbox_inches="tight"); plt.close()
        print(f"\nFigure: {A.fig}")
    except Exception as e:
        print(f"\nFigure failed: {e}")

    print(json.dumps(dict(status="STEP014B_DONE", out=str(od),
                          elapsed=round(time.time()-t0, 1))))

if __name__ == "__main__":
    main()
