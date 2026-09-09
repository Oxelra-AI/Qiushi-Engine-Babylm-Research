#!/usr/bin/env python3
"""Step015b: paired-stream answer-loss allocation for orbit binding.

This repairs research in two ways:
1. all BOUND objective arms consume the same generated contexts and minibatch order
   for a given seed/epoch, so objective allocation is the principal changed variable;
2. binding metrics separate context-bag retrieval from query-conditioned selection and
   avoid top-1 tie bias.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import loss_allocation_binding as B

# Defensive layout assertion: target answer is predicted by logits at B.IS_POS.
_layout = B.mk_seq([0, 1, 2, 3], [0, 1, 2, 3], 0, B.RWT(0))
assert _layout[B.IS_POS] == B.IS
assert _layout[B.IS_POS + 1] == B.RWT(0)


def make_epoch_rows(seed, epoch, n):
    """Common finite stream for all arms at a seed/epoch."""
    rng = np.random.default_rng(5150000 + seed * 10007 + epoch)
    rows = []
    for _ in range(n):
        ce = rng.choice(B.TRAIN_E, B.K, replace=False).tolist()
        ca = rng.choice(B.N_ATTR, B.K, replace=False).tolist()
        qi = int(rng.integers(B.K))
        bag_ti = int(rng.integers(B.K))
        rows.append((ce, ca, qi, bag_ti))
    return rows


def seqs_from_rows(rows, target):
    seqs = []
    for ce, ca, qi, bag_ti in rows:
        if target == "bound":
            tgt = B.RWT(ca[qi])
        elif target == "bag_indep":
            tgt = B.RWT(ca[bag_ti])
        else:
            raise ValueError(target)
        seqs.append(B.mk_seq(ce, ca, qi, tgt))
    return seqs


def common_order(seed, epoch, n):
    return np.random.default_rng(6160000 + seed * 10007 + epoch).permutation(n)


def train_ep_ordered(model, opt, seqs, order, dev, mode="full", answer_weight=1.0, bs=64):
    model.train()
    st = torch.tensor(seqs, dtype=torch.long)
    st = st[torch.tensor(order, dtype=torch.long)]
    L = B.SL - 1
    cm = B.causal_mask(L, dev)
    pos_w = B.weight_vec(mode, answer_weight, dev)
    total_loss_sum = 0.0
    total_weight = 0.0
    ans_loss_sum = 0.0
    ctx_loss_sum = 0.0
    ctx_weight = 0.0
    ans_count = 0
    for i in range(0, st.size(0), bs):
        batch = st[i:i + bs].to(dev)
        inp, tgt = batch[:, :L], batch[:, 1:]
        logits = model(inp, cm)
        ce = F.cross_entropy(logits.reshape(-1, B.VOCAB), tgt.reshape(-1), reduction="none").view(tgt.shape)
        nonpad = (tgt != B.PAD).float()
        weights = nonpad * pos_w.view(1, -1)
        denom = weights.sum().clamp_min(1.0)
        loss = (ce * weights).sum() / denom
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss_sum += float((ce * weights).sum().item())
        total_weight += float(denom.item())
        ans_loss_sum += float(ce[:, B.IS_POS].sum().item())
        ans_count += int(ce.size(0))
        ctx_mask = nonpad.clone(); ctx_mask[:, B.IS_POS] = 0.0
        ctx_loss_sum += float((ce * ctx_mask).sum().item())
        ctx_weight += float(ctx_mask.sum().item())
    return dict(weighted_loss=total_loss_sum / max(total_weight, 1.0),
                answer_ce=ans_loss_sum / max(ans_count, 1),
                context_ce=ctx_loss_sum / max(ctx_weight, 1.0))


# ------------------------------ probes ------------------------------
def gen_std_probes(rng, n, entities):
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, B.K, replace=False).tolist()
        ca = rng.choice(B.N_ATTR, B.K, replace=False).tolist()
        qi = int(rng.integers(B.K))
        ps.append(dict(s=B.mk_seq(ce, ca, qi, B.PAD), ce=ce, ca=ca, qi=qi, correct=ca[qi]))
    return ps


def gen_held_probes(rng, n):
    ps = []
    for _ in range(n):
        he = int(rng.choice(B.HELD_E))
        others = rng.choice(B.TRAIN_E, B.K - 1, replace=False).tolist()
        ce = list(rng.permutation(others + [he]))
        qi = ce.index(he)
        ca = rng.choice(B.N_ATTR, B.K, replace=False).tolist()
        ps.append(dict(s=B.mk_seq(ce, ca, qi, B.PAD), ce=ce, ca=ca, qi=qi, correct=ca[qi]))
    return ps


def gen_bswap_probes(rng, n, entities):
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, B.K, replace=False).tolist()
        ca = rng.choice(B.N_ATTR, B.K, replace=False).tolist()
        qi = int(rng.integers(B.K))
        si = qi
        while si == qi:
            si = int(rng.integers(B.K))
        ca2 = list(ca)
        ca2[qi], ca2[si] = ca2[si], ca2[qi]
        ps.append(dict(s1=B.mk_seq(ce, ca, qi, B.PAD), s2=B.mk_seq(ce, ca2, qi, B.PAD),
                       c1=ca[qi], c2=ca2[qi], ce=ce, ca1=ca, ca2=ca2, qi=qi))
    return ps


def gen_qswap_probes(rng, n, entities):
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, B.K, replace=False).tolist()
        ca = rng.choice(B.N_ATTR, B.K, replace=False).tolist()
        qi1 = int(rng.integers(B.K))
        qi2 = qi1
        while qi2 == qi1:
            qi2 = int(rng.integers(B.K))
        ps.append(dict(s1=B.mk_seq(ce, ca, qi1, B.PAD), s2=B.mk_seq(ce, ca, qi2, B.PAD),
                       c1=ca[qi1], c2=ca[qi2], ce=ce, ca=ca, qi1=qi1, qi2=qi2))
    return ps


def gen_corrupt_probes(rng, n, entities):
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, B.K, replace=False).tolist()
        ca = rng.choice(B.N_ATTR, B.K, replace=False).tolist()
        qi = int(rng.integers(B.K))
        novel = int(rng.integers(B.N_ATTR))
        while novel in ca:
            novel = int(rng.integers(B.N_ATTR))
        ca_qc = list(ca); ca_qc[qi] = novel
        di = qi
        while di == qi:
            di = int(rng.integers(B.K))
        ca_dc = list(ca); ca_dc[di] = novel
        ps.append(dict(s_orig=B.mk_seq(ce, ca, qi, B.PAD), s_qc=B.mk_seq(ce, ca_qc, qi, B.PAD),
                       s_dc=B.mk_seq(ce, ca_dc, qi, B.PAD), ca=ca, qi=qi, correct=ca[qi], novel=novel))
    return ps


def get_logits(model, seqs, dev):
    st = torch.tensor(seqs, dtype=torch.long, device=dev)
    L = B.SL - 1
    cm = B.causal_mask(L, dev)
    ps, lps, los = [], [], []
    with torch.no_grad():
        for i in range(0, st.size(0), 256):
            logits = model(st[i:i + 256, :L], cm)[:, B.IS_POS, :]
            los.append(logits)
            ps.append(F.softmax(logits, dim=-1))
            lps.append(F.log_softmax(logits, dim=-1))
    return torch.cat(ps), torch.cat(lps), torch.cat(los)


def tie_safe_top(logits, correct_idx):
    maxv = float(torch.max(logits).item())
    winners = [i for i, x in enumerate(logits.tolist()) if abs(float(x) - maxv) <= 1e-10]
    return 1.0 / len(winners) if correct_idx in winners else 0.0


def entropy(probs):
    pp = np.array([float(x) for x in probs if float(x) > 0])
    return float(-(pp * np.log(pp)).sum()) if len(pp) else 0.0


def eval_std(model, probes, dev):
    model.eval()
    p, lp, lo = get_logits(model, [x["s"] for x in probes], dev)
    M = {"correct_nll": [], "family_mass": [], "bag_mass": [], "within_bag_nll": [],
         "within_bag_entropy": [], "ctx_top1": [], "query_margin": [], "mrr": []}
    for j, pr in enumerate(probes):
        ca, qi, corr = pr["ca"], pr["qi"], pr["correct"]
        ct = B.RWT(corr)
        M["correct_nll"].append(-lp[j, ct].item())
        fm = p[j, B.RWT_TOKS].sum().item()
        M["family_mass"].append(fm)
        bag_toks = [B.RWT(a) for a in ca]
        bag_probs = p[j, bag_toks]
        bm = bag_probs.sum().item()
        M["bag_mass"].append(bm)
        M["within_bag_nll"].append(-math.log(max(p[j, ct].item() / max(bm, 1e-30), 1e-30)))
        norm_bag = (bag_probs / max(bm, 1e-30)).cpu().numpy()
        M["within_bag_entropy"].append(entropy(norm_bag))
        bag_logits = torch.tensor([lo[j, B.RWT(a)].item() for a in ca])
        M["ctx_top1"].append(tie_safe_top(bag_logits, qi))
        decoy_logits = [lo[j, B.RWT(a)].item() for ii, a in enumerate(ca) if ii != qi]
        M["query_margin"].append(lo[j, ct].item() - float(np.mean(decoy_logits)))
        rp = p[j, B.RWT_TOKS]
        rank = int((rp.argsort(descending=True) == corr).nonzero(as_tuple=True)[0].item()) + 1
        M["mrr"].append(1.0 / rank)
    out = {}
    for k, vals in M.items():
        arr = np.array(vals, dtype=float)
        out[k] = round(float(arr.mean()), 6)
        out[k + "_sem"] = round(float(arr.std(ddof=0) / math.sqrt(max(len(arr), 1))), 6)
    return out


def eval_bswap(model, pairs, dev):
    _, _, lo1 = get_logits(model, [x["s1"] for x in pairs], dev)
    _, _, lo2 = get_logits(model, [x["s2"] for x in pairs], dev)
    vals = []
    for j, pr in enumerate(pairs):
        ct1, ct2 = B.RWT(pr["c1"]), B.RWT(pr["c2"])
        vals.append(0.5 * ((lo1[j, ct1] - lo1[j, ct2]).item() + (lo2[j, ct2] - lo2[j, ct1]).item()))
    arr = np.array(vals)
    return dict(mean=round(float(arr.mean()), 6), sem=round(float(arr.std(ddof=0) / math.sqrt(len(arr))), 6),
                probe_std=round(float(arr.std(ddof=0)), 6), frac_pos=round(float((arr > 0).mean()), 6))


def eval_qswap(model, probes, dev):
    p1, _, lo1 = get_logits(model, [x["s1"] for x in probes], dev)
    p2, _, lo2 = get_logits(model, [x["s2"] for x in probes], dev)
    vals = []; both = 0; frac = []
    for j, pr in enumerate(probes):
        ct1, ct2 = B.RWT(pr["c1"]), B.RWT(pr["c2"])
        m = 0.5 * ((lo1[j, ct1] - lo1[j, ct2]).item() + (lo2[j, ct2] - lo2[j, ct1]).item())
        vals.append(m); frac.append(float(m > 0))
        top1 = B.RWT_TOKS[p1[j, B.RWT_TOKS].argmax().item()]
        top2 = B.RWT_TOKS[p2[j, B.RWT_TOKS].argmax().item()]
        both += int(top1 == ct1 and top2 == ct2)
    arr = np.array(vals)
    return dict(margin=round(float(arr.mean()), 6), sem=round(float(arr.std(ddof=0) / math.sqrt(len(arr))), 6),
                frac_pos=round(float(np.mean(frac)), 6), both=round(float(both / len(probes)), 6))


def eval_corrupt(model, probes, dev):
    po, _, _ = get_logits(model, [x["s_orig"] for x in probes], dev)
    pqc, _, _ = get_logits(model, [x["s_qc"] for x in probes], dev)
    pdc, _, _ = get_logits(model, [x["s_dc"] for x in probes], dev)
    qnf, dnf, qd, dd = [], [], [], []
    for j, pr in enumerate(probes):
        ct, nt = B.RWT(pr["correct"]), B.RWT(pr["novel"])
        qd.append((po[j, ct] - pqc[j, ct]).item())
        dd.append((po[j, ct] - pdc[j, ct]).item())
        qnf.append((pqc[j, nt] - po[j, nt]).item())
        dnf.append((pdc[j, nt] - po[j, nt]).item())
    qnf, dnf, qd, dd = map(np.array, [qnf, dnf, qd, dd])
    ns = qnf - dnf; ds = qd - dd
    return dict(query_novel=round(float(qnf.mean()), 6), decoy_novel=round(float(dnf.mean()), 6),
                novel_selectivity=round(float(ns.mean()), 6), novel_selectivity_sem=round(float(ns.std(ddof=0) / math.sqrt(len(ns))), 6),
                drop_selectivity=round(float(ds.mean()), 6), drop_selectivity_sem=round(float(ds.std(ddof=0) / math.sqrt(len(ds))), 6))


def evaluate(model, probes, dev):
    return dict(std=eval_std(model, probes["std"], dev), held=eval_std(model, probes["held"], dev),
                bswap=eval_bswap(model, probes["bswap"], dev), qswap=eval_qswap(model, probes["qswap"], dev),
                corrupt=eval_corrupt(model, probes["corrupt"], dev))


ARMS = [
    dict(name="full_w1_200", target="bound", phases=[dict(mode="full", answer_weight=1.0, epochs=200)]),
    dict(name="w16_200", target="bound", phases=[dict(mode="weighted_full", answer_weight=16.0, epochs=200)]),
    dict(name="w64_200", target="bound", phases=[dict(mode="weighted_full", answer_weight=64.0, epochs=200)]),
    dict(name="ans_only_200", target="bound", phases=[dict(mode="answer_only", answer_weight=1.0, epochs=200)]),
    dict(name="ans_only_1000", target="bound", phases=[dict(mode="answer_only", answer_weight=1.0, epochs=1000)]),
    dict(name="bag_ans_only_1000", target="bag_indep", phases=[dict(mode="answer_only", answer_weight=1.0, epochs=1000)]),
    dict(name="ans500_full500", target="bound", phases=[dict(mode="answer_only", answer_weight=1.0, epochs=500), dict(mode="full", answer_weight=1.0, epochs=500)]),
]


def get_path(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def summarize(data):
    S = {}
    for name in data["config"]["arms"]:
        finals = []
        switch = []
        for sd in data["config"]["seeds"]:
            rs = [r for r in data["records"] if r["arm"] == name and r["seed"] == sd]
            if rs:
                finals.append(rs[-1])
                sw = [r for r in rs if r.get("phase_idx") == 0 and r.get("phase_end")]
                if sw:
                    switch.append(sw[-1])
        def coll(records, path):
            vals = [float(get_path(r, path)) for r in records]
            return dict(mean=round(float(np.mean(vals)), 6), std=round(float(np.std(vals)), 6), vals=[round(float(v), 6) for v in vals]) if vals else None
        S[name] = dict(final={
            "correct_nll": coll(finals, ["std", "correct_nll"]),
            "bag_mass": coll(finals, ["std", "bag_mass"]),
            "within_bag_nll": coll(finals, ["std", "within_bag_nll"]),
            "ctx_top1": coll(finals, ["std", "ctx_top1"]),
            "query_margin": coll(finals, ["std", "query_margin"]),
            "bswap": coll(finals, ["bswap", "mean"]),
            "bswap_frac": coll(finals, ["bswap", "frac_pos"]),
            "qswap_margin": coll(finals, ["qswap", "margin"]),
            "qswap_frac": coll(finals, ["qswap", "frac_pos"]),
            "qswap_both": coll(finals, ["qswap", "both"]),
            "corrupt_select": coll(finals, ["corrupt", "novel_selectivity"]),
            "answer_ce": coll(finals, ["answer_ce"]),
            "context_ce": coll(finals, ["context_ce"]),
        })
        if switch:
            S[name]["phase0_end"] = {
                "correct_nll": coll(switch, ["std", "correct_nll"]),
                "ctx_top1": coll(switch, ["std", "ctx_top1"]),
                "bswap": coll(switch, ["bswap", "mean"]),
                "qswap_margin": coll(switch, ["qswap", "margin"]),
                "corrupt_select": coll(switch, ["corrupt", "novel_selectivity"]),
            }
    return S


def make_figure(data, figpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = data["config"]["arms"]
    colors = {"full_w1_200":"tab:gray", "w16_200":"tab:purple", "w64_200":"tab:pink", "ans_only_200":"tab:cyan", "ans_only_1000":"tab:blue", "bag_ans_only_1000":"tab:orange", "ans500_full500":"tab:green"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    plots = [("std","correct_nll","Correct NLL","NLL"), ("std","bag_mass","Mass on four context attributes","Probability"),
             ("std","ctx_top1","Tie-safe top-1 among context attrs","Score"), ("bswap","mean","B-swap margin","Logit margin"),
             ("qswap","margin","Q-swap margin","Logit margin"), ("corrupt","novel_selectivity","Query-novel selectivity","Probability diff")]
    for arm in arms:
        for ax, (grp, key, title, ylabel) in zip(axes.flat, plots):
            by_seed = []
            for sd in data["config"]["seeds"]:
                rs = [r for r in data["records"] if r["arm"] == arm and r["seed"] == sd]
                if rs:
                    by_seed.append(rs)
            if not by_seed:
                continue
            # arms can have different lengths; align by epoch values present in all available records at that epoch.
            epochs = sorted(set(r["epoch"] for rs in by_seed for r in rs))
            xs, ys = [], []
            for ep in epochs:
                vals = [get_path(next((r for r in rs if r["epoch"] == ep), rs[-1]), [grp, key]) for rs in by_seed]
                xs.append(ep); ys.append(float(np.mean(vals)))
            ax.plot(xs, ys, label=arm, color=colors.get(arm, None), lw=1.6)
    refs = [(0, math.log(4), "bag uniform"), (0, math.log(12), "marginal"), (2, 0.25, "chance"), (3, 0, "zero"), (4, 0, "zero"), (5, 0, "zero")]
    for idx, y, lab in refs:
        axes.flat[idx].axhline(y, color="black", ls="--" if y == 0 else ":", alpha=0.35, label=lab)
    for ax, (_, _, title, ylabel) in zip(axes.flat, plots):
        ax.set_title(title, fontsize=10); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(fontsize=6)
    fig.suptitle("Step015b: Paired-stream loss allocation and orbit binding", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/revision_015b_paired_loss_allocation")
    ap.add_argument("--fig", default="experiments/archive/functional_learning/figures/revision_015b_paired_loss_allocation.png")
    ap.add_argument("--seeds", default="42,43,100")
    A = ap.parse_args()
    seeds = [42] if A.smoke else [int(x) for x in A.seeds.split(",") if x.strip()]
    arms = ARMS[:3] if A.smoke else ARMS
    if A.smoke:
        new_arms = []
        for a in arms:
            ph = dict(a["phases"][0]); ph["epochs"] = min(ph["epochs"], 6)
            new_arms.append(dict(name=a["name"], target=a["target"], phases=[ph]))
        arms = new_arms
    n_train = 100 if A.smoke else 500
    eval_every = 2 if A.smoke else 25
    cfg = dict(seeds=seeds, arms=[a["name"] for a in arms], n_train=n_train, eval_every=eval_every,
               d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64,
               n_std=512 if not A.smoke else 96, n_held=256 if not A.smoke else 48,
               n_bswap=512 if not A.smoke else 96, n_qswap=256 if not A.smoke else 48,
               n_corrupt=256 if not A.smoke else 48,
               answer_index_in_tgt=B.IS_POS, nonpad_targets=16,
               coef_full=1/16, coef_w16=16/31, coef_w64=64/79,
               log4=math.log(4), log12=math.log(12), log4_over_16=math.log(4)/16)
    erng = np.random.default_rng(150015)
    probes = dict(std=gen_std_probes(erng, cfg["n_std"], B.TRAIN_E), held=gen_held_probes(erng, cfg["n_held"]),
                  bswap=gen_bswap_probes(erng, cfg["n_bswap"], B.TRAIN_E), qswap=gen_qswap_probes(erng, cfg["n_qswap"], B.TRAIN_E),
                  corrupt=gen_corrupt_probes(erng, cfg["n_corrupt"], B.TRAIN_E))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    data = dict(config=cfg, arm_specs=arms, records=[])
    t0 = time.time()
    Path(A.data).mkdir(parents=True, exist_ok=True)
    for sd in seeds:
        print(f"\n{'='*72}\nSeed {sd}\n{'='*72}")
        for spec in arms:
            torch.manual_seed(sd); np.random.seed(sd)
            model = B.CLM(B.VOCAB, cfg["d"], cfg["nh"], cfg["nl"], B.SL).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            ep_global = 0
            ts = time.time()
            print(f"  Arm {spec['name']}: target={spec['target']} phases={spec['phases']}")
            for pi, ph in enumerate(spec["phases"]):
                for pe in range(1, ph["epochs"] + 1):
                    ep_global += 1
                    rows = make_epoch_rows(sd, ep_global, n_train)
                    seqs = seqs_from_rows(rows, spec["target"])
                    order = common_order(sd, ep_global, n_train)
                    info = train_ep_ordered(model, opt, seqs, order, dev, ph["mode"], ph.get("answer_weight", 1.0), cfg["bs"])
                    phase_end = (pe == ph["epochs"])
                    if ep_global == 1 or ep_global % eval_every == 0 or phase_end:
                        met = evaluate(model, probes, dev)
                        rec = dict(arm=spec["name"], seed=sd, epoch=ep_global, phase_idx=pi, phase_epoch=pe,
                                   phase_end=phase_end, mode=ph["mode"], answer_weight=ph.get("answer_weight", 1.0),
                                   weighted_loss=round(info["weighted_loss"], 6), answer_ce=round(info["answer_ce"], 6),
                                   context_ce=round(info["context_ce"], 6))
                        rec.update(met)
                        data["records"].append(rec)
                        print(f"    e={ep_global:4d} p{pi}:{ph['mode']:<13s} ansCE={info['answer_ce']:.3f} ctxCE={info['context_ce']:.3f} "
                              f"cnll={met['std']['correct_nll']:.3f} bagM={met['std']['bag_mass']:.3f} top4={met['std']['ctx_top1']:.3f} "
                              f"qm={met['std']['query_margin']:+.3f} bs={met['bswap']['mean']:+.3f} "
                              f"qs={met['qswap']['margin']:+.3f} sel={met['corrupt']['novel_selectivity']:+.3f} "
                              f"({time.time()-ts:.1f}s)")
            print(f"    Params: {sum(p.numel() for p in model.parameters()):,d}")
    data["summary"] = summarize(data)
    outp = Path(A.data) / "results.json"
    with open(outp, "w") as f:
        json.dump(data, f, indent=2)
    make_figure(data, A.fig)
    print("\nSUMMARY")
    for name, block in data["summary"].items():
        f = block["final"]
        print(f"  {name:18s} cnll={f['correct_nll']['mean']:.3f} top4={f['ctx_top1']['mean']:.3f} "
              f"bagM={f['bag_mass']['mean']:.3f} wbag={f['within_bag_nll']['mean']:.3f} "
              f"qm={f['query_margin']['mean']:+.3f} bs={f['bswap']['mean']:+.3f} "
              f"qsm={f['qswap_margin']['mean']:+.3f} qfrac={f['qswap_frac']['mean']:.3f} sel={f['corrupt_select']['mean']:+.3f}")
        if "phase0_end" in block:
            sw = block["phase0_end"]
            print(f"    phase0_end cnll={sw['correct_nll']['mean']:.3f} top4={sw['ctx_top1']['mean']:.3f} bs={sw['bswap']['mean']:+.3f} qsm={sw['qswap_margin']['mean']:+.3f} sel={sw['corrupt_select']['mean']:+.3f}")
    print(json.dumps(dict(status="STEP015B_DONE", out=str(outp), figure=A.fig, elapsed=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
