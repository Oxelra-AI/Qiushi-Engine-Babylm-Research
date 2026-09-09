#!/usr/bin/env python3
"""research: answer-loss allocation for permutation-orbit binding.

Keeps the research data distribution and model architecture fixed, but changes how
much learning pressure reaches the answer token. The scientific question is
whether the same finite orbit experience can produce counterfactual entity-attribute
binding when the training objective allocates sufficient pressure to the answer.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ constants ------------------------------
K = 4
N_ENT = 10
N_ATTR = 12
TRAIN_E = list(range(8))
HELD_E = [8, 9]

BOS, PAD, SEP, HAS, IS = 0, 1, 2, 3, 4
ENT = lambda e: 5 + e
ATTR = lambda a: 15 + a
RWT = lambda a: 27 + a
VOCAB = 39
RWT_TOKS = list(range(27, 39))
SL = 18
IS_POS = 15  # input position whose logits predict target token at sequence pos 16


def mk_seq(ce, ca, qi, tgt):
    s = [BOS]
    for i in range(K):
        s += [ENT(ce[i]), HAS, ATTR(ca[i])]
    s += [SEP, ENT(ce[qi]), IS, tgt, PAD]
    assert len(s) == SL
    return s


class DG:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def _ctx(self):
        ce = self.rng.choice(TRAIN_E, K, replace=False).tolist()
        ca = self.rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(self.rng.integers(K))
        return ce, ca, qi

    def bound(self, n):
        return [mk_seq(ce, ca, qi, RWT(ca[qi])) for ce, ca, qi in (self._ctx() for _ in range(n))]

    def bag_indep(self, n):
        seqs = []
        for _ in range(n):
            ce, ca, qi = self._ctx()
            ti = int(self.rng.integers(K))
            seqs.append(mk_seq(ce, ca, qi, RWT(ca[ti])))
        return seqs


# ------------------------------ probes ------------------------------
def gen_std_probes(rng, n, entities):
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(rng.integers(K))
        ps.append(dict(s=mk_seq(ce, ca, qi, PAD), correct=ca[qi], decoys=[ca[i] for i in range(K) if i != qi]))
    return ps


def gen_held_probes(rng, n):
    ps = []
    for _ in range(n):
        he = int(rng.choice(HELD_E))
        others = rng.choice(TRAIN_E, K - 1, replace=False).tolist()
        ce = list(rng.permutation(others + [he]))
        qi = ce.index(he)
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        ps.append(dict(s=mk_seq(ce, ca, qi, PAD), correct=ca[qi], decoys=[ca[i] for i in range(K) if i != qi]))
    return ps


def gen_bswap_probes(rng, n, entities):
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
        ps.append(dict(s1=mk_seq(ce, ca, qi, PAD), s2=mk_seq(ce, ca2, qi, PAD), c1=ca[qi], c2=ca2[qi]))
    return ps


def gen_qswap_probes(rng, n, entities):
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi1 = int(rng.integers(K))
        qi2 = qi1
        while qi2 == qi1:
            qi2 = int(rng.integers(K))
        ps.append(dict(s1=mk_seq(ce, ca, qi1, PAD), s2=mk_seq(ce, ca, qi2, PAD), c1=ca[qi1], c2=ca[qi2]))
    return ps


def gen_corrupt_probes(rng, n, entities):
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
        ps.append(dict(s_orig=mk_seq(ce, ca, qi, PAD), s_qc=mk_seq(ce, ca_qc, qi, PAD),
                       s_dc=mk_seq(ce, ca_dc, qi, PAD), correct=ca[qi], novel=novel))
    return ps


# ------------------------------ model ------------------------------
class Blk(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, m):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=m)
        x = x + h
        return x + self.ff(self.ln2(x))


class CLM(nn.Module):
    def __init__(self, V, d, nh, nl, ml):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ml, d)
        self.blks = nn.ModuleList([Blk(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)

    def forward(self, x, m=None):
        B, L = x.shape
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))
        for b in self.blks:
            h = b(h, m)
        return self.ln(h) @ self.tok.weight.T


def causal_mask(L, dev="cpu"):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), diagonal=1)


# ------------------------------ training ------------------------------
def weight_vec(mode, answer_weight, dev):
    """Weights for each target position in tgt=batch[:,1:]."""
    L = SL - 1
    w = torch.zeros(L, dtype=torch.float32, device=dev)
    if mode == "full":
        w[:] = 1.0
        w[-1] = 0.0  # PAD target, additionally masked out
    elif mode == "answer_only":
        w[IS_POS] = 1.0
    elif mode == "weighted_full":
        w[:] = 1.0
        w[-1] = 0.0
        w[IS_POS] = float(answer_weight)
    else:
        raise ValueError(mode)
    return w


def train_ep(model, opt, seqs, dev, mode="full", answer_weight=1.0, bs=64):
    model.train()
    st = torch.tensor(seqs, dtype=torch.long)
    N = st.size(0)
    st = st[torch.randperm(N)]
    L = SL - 1
    cm = causal_mask(L, dev)
    pos_w = weight_vec(mode, answer_weight, dev)
    total_loss_sum = 0.0
    total_weight = 0.0
    ans_loss_sum = 0.0
    ans_count = 0
    for i in range(0, N, bs):
        batch = st[i:i + bs].to(dev)
        inp, tgt = batch[:, :L], batch[:, 1:]
        logits = model(inp, cm)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction='none').view(tgt.shape)
        nonpad = (tgt != PAD).float()
        weights = nonpad * pos_w.view(1, -1)
        denom = weights.sum().clamp_min(1.0)
        loss = (ce * weights).sum() / denom
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss_sum += float((ce * weights).sum().item())
        total_weight += float(denom.item())
        ans_loss_sum += float(ce[:, IS_POS].sum().item())
        ans_count += int(ce.size(0))
    return dict(weighted_loss=total_loss_sum / max(total_weight, 1.0), answer_ce=ans_loss_sum / max(ans_count, 1))


# ------------------------------ evaluation ------------------------------
def _get_logits(model, seqs_list, dev):
    st = torch.tensor(seqs_list, dtype=torch.long, device=dev)
    L = SL - 1
    cm = causal_mask(L, dev)
    all_p, all_lp, all_lo = [], [], []
    with torch.no_grad():
        for i in range(0, st.size(0), 256):
            b = st[i:i + 256, :L]
            logits = model(b, cm)[:, IS_POS, :]
            all_lo.append(logits)
            all_p.append(F.softmax(logits, dim=-1))
            all_lp.append(F.log_softmax(logits, dim=-1))
    return torch.cat(all_p), torch.cat(all_lp), torch.cat(all_lo)


def eval_std(model, probes, dev):
    model.eval()
    p, lp, lo = _get_logits(model, [x["s"] for x in probes], dev)
    M = {"correct_nll": [], "family_mass": [], "within_nll": [], "mrr": [], "ctx_top1": [], "query_margin": []}
    for j, pr in enumerate(probes):
        c = pr["correct"]; ct = RWT(c)
        M["correct_nll"].append(-lp[j, ct].item())
        fm = p[j, RWT_TOKS].sum().item()
        M["family_mass"].append(fm)
        M["within_nll"].append(-math.log(max(p[j, ct].item() / max(fm, 1e-30), 1e-30)))
        rp = p[j, RWT_TOKS]
        rk = int((rp.argsort(descending=True) == c).nonzero(as_tuple=True)[0].item()) + 1
        M["mrr"].append(1.0 / rk)
        ctx_attrs = [c] + pr["decoys"]
        ctx_logits = torch.tensor([lo[j, RWT(a)].item() for a in ctx_attrs])
        M["ctx_top1"].append(float(int(ctx_logits.argmax().item() == 0)))
        decoy_mean = float(np.mean([lo[j, RWT(a)].item() for a in pr["decoys"]]))
        M["query_margin"].append(lo[j, ct].item() - decoy_mean)
    return {k: round(float(np.mean(v)), 5) for k, v in M.items()}


def eval_bswap(model, pairs, dev):
    model.eval()
    _, _, lo1 = _get_logits(model, [x["s1"] for x in pairs], dev)
    _, _, lo2 = _get_logits(model, [x["s2"] for x in pairs], dev)
    vals = []
    for j, pr in enumerate(pairs):
        ct1, ct2 = RWT(pr["c1"]), RWT(pr["c2"])
        m1 = lo1[j, ct1].item() - lo1[j, ct2].item()
        m2 = lo2[j, ct2].item() - lo2[j, ct1].item()
        vals.append(0.5 * (m1 + m2))
    arr = np.array(vals)
    return dict(mean=round(float(arr.mean()), 5), std=round(float(arr.std()), 5), frac_pos=round(float((arr > 0).mean()), 5))


def eval_qswap(model, probes, dev):
    model.eval()
    p1, _, lo1 = _get_logits(model, [x["s1"] for x in probes], dev)
    p2, _, lo2 = _get_logits(model, [x["s2"] for x in probes], dev)
    both_ok = 0; msum = []
    for j, pr in enumerate(probes):
        t1 = RWT_TOKS[p1[j, RWT_TOKS].argmax().item()]
        t2 = RWT_TOKS[p2[j, RWT_TOKS].argmax().item()]
        ok1 = (t1 == RWT(pr["c1"])); ok2 = (t2 == RWT(pr["c2"]))
        both_ok += (ok1 and ok2)
        m1 = lo1[j, RWT(pr["c1"])].item() - lo1[j, RWT(pr["c2"])].item()
        m2 = lo2[j, RWT(pr["c2"])].item() - lo2[j, RWT(pr["c1"])].item()
        msum.append(0.5 * (m1 + m2))
    arr = np.array(msum)
    return dict(both=round(both_ok / len(probes), 5), margin=round(float(arr.mean()), 5), margin_std=round(float(arr.std()), 5), frac_pos=round(float((arr > 0).mean()), 5))


def eval_corrupt(model, probes, dev):
    model.eval()
    po, _, _ = _get_logits(model, [x["s_orig"] for x in probes], dev)
    pqc, _, _ = _get_logits(model, [x["s_qc"] for x in probes], dev)
    pdc, _, _ = _get_logits(model, [x["s_dc"] for x in probes], dev)
    qnf, dnf, qd, dd = [], [], [], []
    for j, pr in enumerate(probes):
        ct = RWT(pr["correct"]); nt = RWT(pr["novel"])
        qd.append(po[j, ct].item() - pqc[j, ct].item())
        dd.append(po[j, ct].item() - pdc[j, ct].item())
        qnf.append(pqc[j, nt].item() - po[j, nt].item())
        dnf.append(pdc[j, nt].item() - po[j, nt].item())
    qnf, dnf, qd, dd = map(np.array, [qnf, dnf, qd, dd])
    return dict(query_drop=round(float(qd.mean()), 5), decoy_drop=round(float(dd.mean()), 5),
                query_novel=round(float(qnf.mean()), 5), decoy_novel=round(float(dnf.mean()), 5),
                novel_selectivity=round(float((qnf - dnf).mean()), 5),
                drop_selectivity=round(float((qd - dd).mean()), 5))


def evaluate(model, probes, dev):
    return dict(std=eval_std(model, probes["std"], dev), held=eval_std(model, probes["held"], dev),
                bswap=eval_bswap(model, probes["bswap"], dev), qswap=eval_qswap(model, probes["qswap"], dev),
                corrupt=eval_corrupt(model, probes["corrupt"], dev))


# ------------------------------ experiment ------------------------------
DEFAULT_ARMS = [
    dict(name="full_w1_200", target="bound", phases=[dict(mode="full", answer_weight=1.0, epochs=200)]),
    dict(name="ans_only_13", target="bound", phases=[dict(mode="answer_only", answer_weight=1.0, epochs=13)]),
    dict(name="w16_25", target="bound", phases=[dict(mode="weighted_full", answer_weight=16.0, epochs=25)]),
    dict(name="ans_only_200", target="bound", phases=[dict(mode="answer_only", answer_weight=1.0, epochs=200)]),
    dict(name="w16_200", target="bound", phases=[dict(mode="weighted_full", answer_weight=16.0, epochs=200)]),
    dict(name="bag_ans_only_200", target="bag_indep", phases=[dict(mode="answer_only", answer_weight=1.0, epochs=200)]),
    dict(name="ans50_full150", target="bound", phases=[dict(mode="answer_only", answer_weight=1.0, epochs=50), dict(mode="full", answer_weight=1.0, epochs=150)]),
    dict(name="w16_50_full150", target="bound", phases=[dict(mode="weighted_full", answer_weight=16.0, epochs=50), dict(mode="full", answer_weight=1.0, epochs=150)]),
]


def compact(arm, seed, ep, phase_idx, phase_ep, train_info, metrics):
    out = dict(arm=arm, seed=seed, epoch=ep, phase_idx=phase_idx, phase_epoch=phase_ep,
               weighted_loss=round(train_info["weighted_loss"], 5), answer_ce=round(train_info["answer_ce"], 5))
    out.update(metrics)
    return out


def summary(data):
    S = {}
    names = sorted({r["arm"] for r in data["records"]})
    for name in names:
        finals = []
        for sd in data["config"]["seeds"]:
            rs = [r for r in data["records"] if r["arm"] == name and r["seed"] == sd]
            if rs:
                finals.append(rs[-1])
        if not finals:
            continue
        def collect(path):
            vals = []
            for r in finals:
                obj = r
                for p in path:
                    obj = obj[p]
                vals.append(float(obj))
            return dict(mean=round(float(np.mean(vals)), 5), std=round(float(np.std(vals)), 5), vals=[round(float(v), 5) for v in vals])
        S[name] = dict(
            correct_nll=collect(["std", "correct_nll"]),
            mrr=collect(["std", "mrr"]),
            ctx_top1=collect(["std", "ctx_top1"]),
            query_margin=collect(["std", "query_margin"]),
            bswap=collect(["bswap", "mean"]),
            bswap_frac=collect(["bswap", "frac_pos"]),
            qswap_margin=collect(["qswap", "margin"]),
            qswap_both=collect(["qswap", "both"]),
            corrupt_select=collect(["corrupt", "novel_selectivity"]),
            answer_ce=collect(["answer_ce"]),
            weighted_loss=collect(["weighted_loss"]),
        )
    return S


def make_figure(data, figpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = data["config"]["arms"]
    seeds = data["config"]["seeds"]
    colors = {
        "full_w1_200": "tab:gray", "ans_only_13": "tab:cyan", "w16_25": "tab:olive",
        "ans_only_200": "tab:blue", "w16_200": "tab:purple", "bag_ans_only_200": "tab:orange",
        "ans50_full150": "tab:green", "w16_50_full150": "tab:red",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    metrics = [
        ("std", "correct_nll", "Correct NLL", "NLL"),
        ("std", "ctx_top1", "Top among context attributes", "Accuracy"),
        ("std", "query_margin", "Query vs decoy context margin", "Logit margin"),
        ("bswap", "mean", "Same-bag binding-swap margin", "Logit margin"),
        ("qswap", "margin", "Query-swap paired margin", "Logit margin"),
        ("corrupt", "novel_selectivity", "Query-specific novel follow", "Probability diff"),
    ]
    def avg_series(name, grp, key):
        by_seed = []
        for sd in seeds:
            rs = [r for r in data["records"] if r["arm"] == name and r["seed"] == sd]
            if rs:
                by_seed.append(rs)
        if not by_seed:
            return [], []
        epochs = [r["epoch"] for r in by_seed[0]]
        vals = []
        for i in range(len(epochs)):
            vv = []
            for rs in by_seed:
                if i < len(rs):
                    vv.append(rs[i][grp][key])
            vals.append(float(np.mean(vv)))
        return epochs, vals
    for name in arms:
        for ax, (grp, key, title, ylabel) in zip(axes.flat, metrics):
            ep, vals = avg_series(name, grp, key)
            if ep:
                ax.plot(ep, vals, label=name, color=colors.get(name, None), lw=1.8)
    refs = [(0, np.log(4), "bag-uniform"), (0, np.log(12), "marginal"), (1, 0.25, "chance"), (3, 0, "zero"), (4, 0, "zero"), (5, 0, "zero")]
    for idx, y, lab in refs:
        axes.flat[idx].axhline(y, color="black", ls="--" if y == 0 else ":", alpha=0.35, label=lab)
    for ax, (_, _, title, ylabel) in zip(axes.flat, metrics):
        ax.set_title(title, fontsize=10); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(fontsize=6)
    fig.suptitle("research: Answer-loss allocation in orbit binding", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/loss_allocation_binding")
    ap.add_argument("--fig", default="experiments/archive/functional_learning/figures/loss_allocation_binding.png")
    ap.add_argument("--seeds", default="42,43,100")
    A = ap.parse_args()
    od = Path(A.data); od.mkdir(parents=True, exist_ok=True)
    seeds = [42] if A.smoke else [int(x) for x in A.seeds.split(",") if x.strip()]
    arms = DEFAULT_ARMS[:3] if A.smoke else DEFAULT_ARMS
    n_train = 100 if A.smoke else 500
    eval_every = 2 if A.smoke else 10
    # In smoke mode shorten long phases.
    if A.smoke:
        short_arms = []
        for a in arms:
            ph0 = dict(a["phases"][0])
            ph0["epochs"] = min(ph0["epochs"], 6)
            short_arms.append(dict(name=a["name"], target=a["target"], phases=[ph0]))
        arms = short_arms
    cfg = dict(seeds=seeds, arms=[a["name"] for a in arms], n_train=n_train, eval_every=eval_every,
               d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64,
               n_std=256 if not A.smoke else 64, n_held=128 if not A.smoke else 32,
               n_bswap=256 if not A.smoke else 64, n_qswap=128 if not A.smoke else 32,
               n_corrupt=128 if not A.smoke else 32,
               answer_index_in_tgt=IS_POS, nonpad_targets=16, answer_share_full=1/16,
               log4_over_16=round(math.log(4)/16, 6))
    erng = np.random.default_rng(2026015)
    probes = dict(std=gen_std_probes(erng, cfg["n_std"], TRAIN_E), held=gen_held_probes(erng, cfg["n_held"]),
                  bswap=gen_bswap_probes(erng, cfg["n_bswap"], TRAIN_E),
                  qswap=gen_qswap_probes(erng, cfg["n_qswap"], TRAIN_E),
                  corrupt=gen_corrupt_probes(erng, cfg["n_corrupt"], TRAIN_E))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    data = dict(config=cfg, arm_specs=arms, records=[])
    t0 = time.time()
    for sd in seeds:
        print(f"\n{'='*70}\nSeed {sd}\n{'='*70}")
        for aidx, spec in enumerate(arms):
            name = spec["name"]
            torch.manual_seed(sd); np.random.seed(sd)
            model = CLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            # Stable deterministic per-arm stream offset, avoiding Python's salted hash.
            dg = DG(sd + 1000 * (aidx + 1))
            gen_fn = getattr(dg, spec["target"])
            global_ep = 0
            ts = time.time()
            print(f"  Arm {name}: target={spec['target']} phases={spec['phases']}")
            for pi, ph in enumerate(spec["phases"]):
                for pep in range(1, ph["epochs"] + 1):
                    global_ep += 1
                    info = train_ep(model, opt, gen_fn(n_train), dev, mode=ph["mode"], answer_weight=ph.get("answer_weight", 1.0), bs=cfg["bs"])
                    should_eval = (global_ep == 1 or global_ep % eval_every == 0 or pep == ph["epochs"])
                    if should_eval:
                        met = evaluate(model, probes, dev)
                        rec = compact(name, sd, global_ep, pi, pep, info, met)
                        data["records"].append(rec)
                        print(f"    e={global_ep:3d} p{pi}:{ph['mode']:<13s} ansCE={info['answer_ce']:.3f} "
                              f"cnll={met['std']['correct_nll']:.3f} top4={met['std']['ctx_top1']:.3f} "
                              f"qm={met['std']['query_margin']:+.3f} bs={met['bswap']['mean']:+.3f} "
                              f"qs={met['qswap']['margin']:+.3f} sel={met['corrupt']['novel_selectivity']:+.3f} "
                              f"({time.time()-ts:.1f}s)")
            npar = sum(p.numel() for p in model.parameters())
            print(f"    Params: {npar:,d}")
    data["summary"] = summary(data)
    with open(od / "results.json", "w") as f:
        json.dump(data, f, indent=2)
    make_figure(data, A.fig)
    print("\nSUMMARY")
    for name, s in data["summary"].items():
        print(f"  {name:18s} cnll={s['correct_nll']['mean']:.3f}±{s['correct_nll']['std']:.3f} "
              f"top4={s['ctx_top1']['mean']:.3f} qm={s['query_margin']['mean']:+.3f} "
              f"bs={s['bswap']['mean']:+.3f} qs={s['qswap_margin']['mean']:+.3f} "
              f"sel={s['corrupt_select']['mean']:+.3f}")
    print(json.dumps(dict(status="DONE", out=str(od / "results.json"), figure=A.fig,
                          elapsed=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
