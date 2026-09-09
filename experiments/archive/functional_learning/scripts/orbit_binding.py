#!/usr/bin/env python3
"""research: Permutation-orbit entity-attribute binding.

Scientific question: Can a small causal LM learn reusable entity-attribute binding
from finite permutation-orbit training? When does binding appear and persist?
What training signal (entity-specific vs bag-level vs marginal) produces
entity-specific context reading?

Design: K=4 entities per context, each with a unique attribute drawn from N_ATTR=12.
The correct target is determined by the query entity's current attribute assignment.
Different permutations of the same entity-attribute sets produce different correct
targets, forcing the model to read the context rather than memorize associations.

Arms (all share entity/attribute/context distributions):
  BOUND:     target = RWT(query entity's attribute)
  BAG_INDEP: target = RWT(random context attribute, query-independent; 1/K match rate)
  MARGINAL:  target = RWT(random attribute from full set)

Evaluation at checkpoints:
  Standard:  correct_nll, family_mass, within_nll, mrr
  Held:      same metrics with held-out query entities
  B_swap:    same token bag + different binding → does prediction follow?
  Q_swap:    same context + different query entity → does prediction change?
  Corrupt:   replace query's attribute with novel → prediction shift to novel?
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, argparse
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────
K = 4           # entities per context
N_ENT = 10      # total entity types
N_ATTR = 12     # total attribute types
TRAIN_E = list(range(8))
HELD_E = [8, 9]

BOS, PAD, SEP, HAS, IS = 0, 1, 2, 3, 4
ENT  = lambda e: 5 + e       # 5..14
ATTR = lambda a: 15 + a      # 15..26
RWT  = lambda a: 27 + a      # 27..38
VOCAB = 39
RWT_TOKS = list(range(27, 39))
SL = 18         # BOS + K*3 + SEP + Eq + IS + TGT + PAD
IS_POS = 15     # model output at this position predicts TGT

def mk_seq(ce, ca, qi, tgt):
    """BOS E0 HAS A0 E1 HAS A1 E2 HAS A2 E3 HAS A3 SEP Eq IS TGT PAD"""
    s = [BOS]
    for i in range(K):
        s += [ENT(ce[i]), HAS, ATTR(ca[i])]
    s += [SEP, ENT(ce[qi]), IS, tgt, PAD]
    assert len(s) == SL
    return s

# ─── Data Generation ──────────────────────────────────────────────
class DG:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def _ctx(self):
        ce = self.rng.choice(TRAIN_E, K, replace=False).tolist()
        ca = self.rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(self.rng.integers(K))
        return ce, ca, qi

    def bound(self, n):
        """BOUND: target = RWT(query entity's attribute)."""
        return [mk_seq(ce, ca, qi, RWT(ca[qi]))
                for ce, ca, qi in (self._ctx() for _ in range(n))]

    def bag_indep(self, n):
        """BAG_INDEPENDENT: target = RWT(random context attribute)."""
        seqs = []
        for _ in range(n):
            ce, ca, qi = self._ctx()
            ti = int(self.rng.integers(K))  # random context slot
            seqs.append(mk_seq(ce, ca, qi, RWT(ca[ti])))
        return seqs

    def marginal(self, n):
        """MARGINAL: target = RWT(random attribute from full set)."""
        seqs = []
        for _ in range(n):
            ce, ca, qi = self._ctx()
            ta = int(self.rng.integers(N_ATTR))
            seqs.append(mk_seq(ce, ca, qi, RWT(ta)))
        return seqs

# ─── Evaluation Probe Generators (fixed across conditions) ────────
def gen_std_probes(rng, n, entities):
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(rng.integers(K))
        ps.append(dict(s=mk_seq(ce, ca, qi, PAD), correct=ca[qi]))
    return ps

def gen_held_probes(rng, n):
    """Query entity from HELD_E, decoys from TRAIN_E."""
    ps = []
    for _ in range(n):
        he = int(rng.choice(HELD_E))
        others = rng.choice(TRAIN_E, K - 1, replace=False).tolist()
        ce = list(rng.permutation(others + [he]))
        qi = ce.index(he)
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        ps.append(dict(s=mk_seq(ce, ca, qi, PAD), correct=ca[qi]))
    return ps

def gen_bswap_probes(rng, n, entities):
    """B_swap pairs: same entity/attribute tokens, different query binding."""
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(rng.integers(K))
        si = qi
        while si == qi:
            si = int(rng.integers(K))
        ca2 = list(ca)
        ca2[qi], ca2[si] = ca2[si], ca2[qi]
        ps.append(dict(s1=mk_seq(ce, ca, qi, PAD),
                       s2=mk_seq(ce, ca2, qi, PAD),
                       c1=ca[qi], c2=ca2[qi]))
    return ps

def gen_qswap_probes(rng, n, entities):
    """Same context, different query entity."""
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi1 = int(rng.integers(K))
        qi2 = qi1
        while qi2 == qi1:
            qi2 = int(rng.integers(K))
        ps.append(dict(s1=mk_seq(ce, ca, qi1, PAD),
                       s2=mk_seq(ce, ca, qi2, PAD),
                       c1=ca[qi1], c2=ca[qi2]))
    return ps

def gen_corrupt_probes(rng, n, entities):
    """Original + query-corrupt + decoy-corrupt triplets."""
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(rng.integers(K))
        novel = int(rng.integers(N_ATTR))
        while novel in ca:
            novel = int(rng.integers(N_ATTR))
        ca_qc = list(ca); ca_qc[qi] = novel
        di = qi
        while di == qi:
            di = int(rng.integers(K))
        ca_dc = list(ca); ca_dc[di] = novel
        ps.append(dict(s_orig=mk_seq(ce, ca, qi, PAD),
                       s_qc=mk_seq(ce, ca_qc, qi, PAD),
                       s_dc=mk_seq(ce, ca_dc, qi, PAD),
                       correct=ca[qi], novel=novel))
    return ps

# ─── Model (same architecture as research) ─────────────────────────
class Blk(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln2 = nn.LayerNorm(d)
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
        for b in self.blks:
            h = b(h, m)
        return self.ln(h) @ self.tok.weight.T

# ─── Training ─────────────────────────────────────────────────────
def causal_mask(L, dev="cpu"):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), diagonal=1)

def train_ep(model, opt, seqs, dev, bs=64):
    model.train()
    st = torch.tensor(seqs, dtype=torch.long)
    N = st.size(0); idx = torch.randperm(N); st = st[idx]
    L = SL - 1; cm = causal_mask(L, dev)
    tl, tt = 0., 0
    for i in range(0, N, bs):
        batch = st[i:i + bs].to(dev)
        inp, tgt = batch[:, :L], batch[:, 1:]
        logits = model(inp, cm)
        mask = (tgt != PAD).float()
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                               reduction='none')
        loss = (loss.view(tgt.shape) * mask).sum() / max(mask.sum(), 1)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * mask.sum().item(); tt += mask.sum().item()
    return tl / max(tt, 1)

# ─── Evaluation Functions ─────────────────────────────────────────
def _get_logits(model, seqs_list, dev):
    """Run model on a list of sequences, return softmax probs and log-probs at IS_POS."""
    st = torch.tensor(seqs_list, dtype=torch.long, device=dev)
    L = SL - 1; cm = causal_mask(L, dev)
    all_p, all_lp = [], []
    with torch.no_grad():
        for i in range(0, st.size(0), 256):
            b = st[i:i + 256, :L]
            logits = model(b, cm)[:, IS_POS, :]
            all_p.append(F.softmax(logits, dim=-1))
            all_lp.append(F.log_softmax(logits, dim=-1))
    return torch.cat(all_p), torch.cat(all_lp)

def _get_raw_logits(model, seqs_list, dev):
    """Return raw logits at IS_POS."""
    st = torch.tensor(seqs_list, dtype=torch.long, device=dev)
    L = SL - 1; cm = causal_mask(L, dev)
    all_lo = []
    with torch.no_grad():
        for i in range(0, st.size(0), 256):
            b = st[i:i + 256, :L]
            all_lo.append(model(b, cm)[:, IS_POS, :])
    return torch.cat(all_lo)

def eval_std(model, probes, dev):
    model.eval()
    p, lp = _get_logits(model, [x["s"] for x in probes], dev)
    M = {"correct_nll": [], "family_mass": [], "within_nll": [], "mrr": []}
    for j, pr in enumerate(probes):
        c = pr["correct"]; ct = RWT(c)
        M["correct_nll"].append(-lp[j, ct].item())
        fm = p[j, RWT_TOKS].sum().item()
        M["family_mass"].append(fm)
        M["within_nll"].append(-np.log(max(p[j, ct].item() / max(fm, 1e-30), 1e-30)))
        rp = p[j, RWT_TOKS]
        rk = int((rp.argsort(descending=True) == c).nonzero(as_tuple=True)[0].item()) + 1
        M["mrr"].append(1.0 / rk)
    return {k: round(float(np.mean(v)), 5) for k, v in M.items()}

def eval_bswap(model, pairs, dev):
    """B_swap: binding swap margin on same-bag paired permutations."""
    model.eval()
    lo1 = _get_raw_logits(model, [x["s1"] for x in pairs], dev)
    lo2 = _get_raw_logits(model, [x["s2"] for x in pairs], dev)
    bs_vals = []
    for j, pr in enumerate(pairs):
        ct1, ct2 = RWT(pr["c1"]), RWT(pr["c2"])
        m1 = lo1[j, ct1].item() - lo1[j, ct2].item()
        m2 = lo2[j, ct2].item() - lo2[j, ct1].item()
        bs_vals.append(0.5 * (m1 + m2))
    bs = np.array(bs_vals)
    return dict(mean=round(float(bs.mean()), 5),
                std=round(float(bs.std()), 5),
                frac_pos=round(float((bs > 0).mean()), 5))

def eval_qswap(model, probes, dev):
    """Query-swap: same context, different query → correct prediction?"""
    model.eval()
    p1, _ = _get_logits(model, [x["s1"] for x in probes], dev)
    p2, _ = _get_logits(model, [x["s2"] for x in probes], dev)
    c1_ok, c2_ok, both_ok = 0, 0, 0
    for j, pr in enumerate(probes):
        t1 = RWT_TOKS[p1[j, RWT_TOKS].argmax().item()]
        t2 = RWT_TOKS[p2[j, RWT_TOKS].argmax().item()]
        ok1 = (t1 == RWT(pr["c1"]))
        ok2 = (t2 == RWT(pr["c2"]))
        c1_ok += ok1; c2_ok += ok2; both_ok += (ok1 and ok2)
    n = len(probes)
    return dict(acc1=round(c1_ok / n, 5), acc2=round(c2_ok / n, 5),
                both=round(both_ok / n, 5))

def eval_corrupt(model, probes, dev):
    """Corruption: query entity fact replaced with novel attribute."""
    model.eval()
    po, _ = _get_logits(model, [x["s_orig"] for x in probes], dev)
    pqc, _ = _get_logits(model, [x["s_qc"] for x in probes], dev)
    pdc, _ = _get_logits(model, [x["s_dc"] for x in probes], dev)
    qd, dd, qnf, dnf = [], [], [], []
    for j, pr in enumerate(probes):
        ct = RWT(pr["correct"]); nt = RWT(pr["novel"])
        qd.append(po[j, ct].item() - pqc[j, ct].item())     # correct-prob drop
        dd.append(po[j, ct].item() - pdc[j, ct].item())      # decoy correct drop
        qnf.append(pqc[j, nt].item() - po[j, nt].item())    # novel follow (query)
        dnf.append(pdc[j, nt].item() - po[j, nt].item())     # novel follow (decoy)
    qd, dd = np.array(qd), np.array(dd)
    qnf, dnf = np.array(qnf), np.array(dnf)
    return dict(query_drop=round(float(qd.mean()), 5),
                decoy_drop=round(float(dd.mean()), 5),
                query_novel=round(float(qnf.mean()), 5),
                decoy_novel=round(float(dnf.mean()), 5),
                drop_ratio=round(float(qd.mean() / max(abs(dd.mean()), 1e-10)), 5),
                novel_ratio=round(float(qnf.mean() / max(abs(dnf.mean()), 1e-10)), 5))

# ─── Main ─────────────────────────────────────────────────────────
ARMS = ["bound", "bag_indep", "marginal"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/orbit_binding")
    ap.add_argument("--fig", default="experiments/archive/functional_learning/figures/orbit_binding.png")
    A = ap.parse_args()
    od = Path(A.data); od.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        seeds=[42] if A.smoke else [42, 43, 100],
        epochs=10 if A.smoke else 200,
        n_train=100 if A.smoke else 500,
        eval_every=2 if A.smoke else 10,
        d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64,
        n_std=200, n_held=100, n_bswap=200, n_qswap=100, n_corrupt=100,
    )

    # ── Fixed eval probes ──
    erng = np.random.default_rng(9999)
    std_p   = gen_std_probes(erng, cfg["n_std"], TRAIN_E)
    held_p  = gen_held_probes(erng, cfg["n_held"])
    bswap_p = gen_bswap_probes(erng, cfg["n_bswap"], TRAIN_E)
    qswap_p = gen_qswap_probes(erng, cfg["n_qswap"], TRAIN_E)
    corr_p  = gen_corrupt_probes(erng, cfg["n_corrupt"], TRAIN_E)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    data = dict(config=cfg, arms={})
    t0 = time.time()

    for sd in cfg["seeds"]:
        print(f"\n{'=' * 60}\nSeed {sd}\n{'=' * 60}")
        for arm in ARMS:
            torch.manual_seed(sd); np.random.seed(sd)
            model = CLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)
            npar = sum(p.numel() for p in model.parameters())
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                    weight_decay=cfg["wd"])
            dg = DG(sd + hash(arm) % 10000)  # arm-specific data stream
            gen_fn = getattr(dg, arm)
            key = f"seed{sd}_{arm}"
            curves = []
            ts = time.time()

            for ep in range(1, cfg["epochs"] + 1):
                tl = train_ep(model, opt, gen_fn(cfg["n_train"]), dev, cfg["bs"])

                if ep % cfg["eval_every"] == 0 or ep == 1:
                    m = dict(
                        epoch=ep, train_loss=round(tl, 5),
                        std=eval_std(model, std_p, dev),
                        held=eval_std(model, held_p, dev),
                        bswap=eval_bswap(model, bswap_p, dev),
                        qswap=eval_qswap(model, qswap_p, dev),
                        corrupt=eval_corrupt(model, corr_p, dev),
                    )
                    curves.append(m)
                    if ep == 1 or ep % (cfg["eval_every"] * 5) == 0 or ep == cfg["epochs"]:
                        print(f"  {arm:12s} e={ep:3d} tl={tl:.3f} "
                              f"cnll={m['std']['correct_nll']:.3f} "
                              f"bswap={m['bswap']['mean']:+.3f}"
                              f"({m['bswap']['frac_pos']:.2f}) "
                              f"mrr={m['std']['mrr']:.3f} "
                              f"qs={m['qswap']['both']:.2f} "
                              f"({time.time() - ts:.1f}s)")

            print(f"  Params: {npar:,d}")
            data["arms"][key] = dict(curves=curves)

    # ── Save results ──
    with open(od / "results.json", "w") as f:
        json.dump(data, f, indent=2)

    # ── Summary ──
    S = compute_summary(data, cfg["seeds"])
    print_summary(S, cfg["seeds"])

    # ── Figure ──
    try:
        make_figure(data, cfg["seeds"], A.fig)
        print(f"\nFigure saved: {A.fig}")
    except Exception as e:
        print(f"\nFigure failed: {e}")

    el = round(time.time() - t0, 1)
    print(json.dumps(dict(status="DONE", out=str(od), elapsed=el)))


def compute_summary(data, seeds):
    S = {}
    for arm in ARMS:
        finals = []
        for sd in seeds:
            k = f"seed{sd}_{arm}"
            if k in data["arms"] and data["arms"][k]["curves"]:
                finals.append(data["arms"][k]["curves"][-1])
        if not finals:
            continue
        a = {}
        for grp in ["std", "held", "bswap", "qswap", "corrupt"]:
            if grp not in finals[0]:
                continue
            a[grp] = {}
            for mk, mv in finals[0][grp].items():
                vals = [f[grp][mk] for f in finals if grp in f and mk in f[grp]]
                a[grp][mk] = dict(mean=round(np.mean(vals), 5),
                                  std=round(np.std(vals), 5))
        S[arm] = a
    return S


def print_summary(S, seeds):
    print(f"\n{'=' * 70}\nSUMMARY ({len(seeds)} seeds)\n{'=' * 70}")
    for arm in ARMS:
        if arm not in S:
            continue
        a = S[arm]
        print(f"\n  {arm}:")
        if "std" in a:
            s = a["std"]
            print(f"    std:   cnll={s['correct_nll']['mean']:.3f}±{s['correct_nll']['std']:.3f}  "
                  f"mrr={s['mrr']['mean']:.3f}±{s['mrr']['std']:.3f}  "
                  f"fam={s['family_mass']['mean']:.3f}  "
                  f"wnll={s['within_nll']['mean']:.3f}")
        if "held" in a:
            h = a["held"]
            print(f"    held:  cnll={h['correct_nll']['mean']:.3f}±{h['correct_nll']['std']:.3f}  "
                  f"mrr={h['mrr']['mean']:.3f}")
        if "bswap" in a:
            b = a["bswap"]
            print(f"    bswap: mean={b['mean']['mean']:+.3f}±{b['mean']['std']:.3f}  "
                  f"frac={b['frac_pos']['mean']:.3f}")
        if "qswap" in a:
            q = a["qswap"]
            print(f"    qswap: both={q['both']['mean']:.3f}±{q['both']['std']:.3f}")
        if "corrupt" in a:
            c = a["corrupt"]
            print(f"    corr:  q_drop={c['query_drop']['mean']:+.4f}  "
                  f"d_drop={c['decoy_drop']['mean']:+.4f}  "
                  f"q_novel={c['query_novel']['mean']:+.4f}  "
                  f"d_novel={c['decoy_novel']['mean']:+.4f}")


def make_figure(data, seeds, figpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = dict(bound="tab:blue", bag_indep="tab:orange", marginal="tab:green")
    labels = dict(bound="BOUND", bag_indep="BAG_INDEP", marginal="MARGINAL")

    def avg_m(arm, *keys):
        curves_all = []
        for sd in seeds:
            k = f"seed{sd}_{arm}"
            if k in data["arms"]:
                curves_all.append(data["arms"][k]["curves"])
        if not curves_all:
            return [], []
        epochs = [c["epoch"] for c in curves_all[0]]
        vals = []
        for ei in range(len(curves_all[0])):
            v = []
            for cc in curves_all:
                obj = cc[ei]
                for ky in keys:
                    obj = obj[ky]
                v.append(obj)
            vals.append(np.mean(v))
        return epochs, vals

    for arm in ARMS:
        c, lb = colors[arm], labels[arm]
        # (0,0) Correct NLL
        ep, v = avg_m(arm, "std", "correct_nll")
        if ep: axes[0, 0].plot(ep, v, color=c, label=lb)
        # (0,1) B_swap
        ep, v = avg_m(arm, "bswap", "mean")
        if ep: axes[0, 1].plot(ep, v, color=c, label=lb)
        # (0,2) MRR
        ep, v = avg_m(arm, "std", "mrr")
        if ep: axes[0, 2].plot(ep, v, color=c, label=lb)
        # (1,0) Query-swap both correct
        ep, v = avg_m(arm, "qswap", "both")
        if ep: axes[1, 0].plot(ep, v, color=c, label=lb)
        # (1,1) Corruption query_drop
        ep, v = avg_m(arm, "corrupt", "query_drop")
        if ep: axes[1, 1].plot(ep, v, color=c, label=lb, ls="-")
        ep, v = avg_m(arm, "corrupt", "decoy_drop")
        if ep: axes[1, 1].plot(ep, v, color=c, label=None, ls="--", alpha=0.5)
        # (1,2) Novel follow (query vs decoy)
        ep, v = avg_m(arm, "corrupt", "query_novel")
        if ep: axes[1, 2].plot(ep, v, color=c, label=lb, ls="-")
        ep, v = avg_m(arm, "corrupt", "decoy_novel")
        if ep: axes[1, 2].plot(ep, v, color=c, label=None, ls="--", alpha=0.5)

    # Reference lines
    axes[0, 0].axhline(np.log(12), color="gray", ls=":", alpha=0.4, label="chance")
    axes[0, 0].axhline(np.log(4), color="gray", ls="--", alpha=0.4, label="bag-uniform")
    axes[0, 1].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[0, 2].axhline(0.521, color="gray", ls="--", alpha=0.4, label="bag-uniform")
    axes[0, 2].axhline(0.259, color="gray", ls=":", alpha=0.4, label="chance")
    axes[1, 0].axhline(0.25, color="gray", ls="--", alpha=0.4, label="chance")

    titles = ["Correct Target NLL", "Binding Swap Margin (B_swap)",
              "MRR within RWT family", "Query-Swap (both correct)",
              "Corruption: p(correct) drop\n(solid=query, dashed=decoy)",
              "Corruption: p(novel) gain\n(solid=query, dashed=decoy)"]
    ylabels = ["NLL", "Logit margin", "MRR", "Fraction", "Probability drop", "Probability gain"]
    for i, ax in enumerate(axes.flat):
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i], fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("research: Permutation-Orbit Entity-Attribute Binding", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
