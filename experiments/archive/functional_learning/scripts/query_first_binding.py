#!/usr/bin/env python3
"""research: temporal/order positive control for orbit binding.

Step015b showed that stronger answer-token pressure does not rescue binding in the
original causal order. This script keeps the model architecture and orbit task fixed,
but changes whether the query appears before the context triples. In query-first order,
attribute-position states can attend to the query and to their local entity, so a local
match marker can form before answer prediction.
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
import loss_allocation_binding as Base

K = Base.K
N_ATTR = Base.N_ATTR
TRAIN_E = Base.TRAIN_E
HELD_E = Base.HELD_E
BOS, PAD, SEP, HAS, IS = Base.BOS, Base.PAD, Base.SEP, Base.HAS, Base.IS
ENT, ATTR, RWT = Base.ENT, Base.ATTR, Base.RWT
VOCAB, RWT_TOKS = Base.VOCAB, Base.RWT_TOKS
SL = 18
IS_POS = 15


def mk_seq(order, ce, ca, qi, tgt):
    if order == "original":
        s = [BOS]
        for i in range(K):
            s += [ENT(ce[i]), HAS, ATTR(ca[i])]
        s += [SEP, ENT(ce[qi]), IS, tgt, PAD]
    elif order == "query_first":
        s = [BOS, ENT(ce[qi]), SEP]
        for i in range(K):
            s += [ENT(ce[i]), HAS, ATTR(ca[i])]
        s += [IS, tgt, PAD]
    else:
        raise ValueError(order)
    assert len(s) == SL
    assert s[IS_POS] == IS and s[IS_POS + 1] in [PAD] + RWT_TOKS
    return s


def make_epoch_rows(seed, epoch, n):
    rng = np.random.default_rng(6161600 + seed * 10007 + epoch)
    rows = []
    for _ in range(n):
        ce = rng.choice(TRAIN_E, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(rng.integers(K))
        bag_ti = int(rng.integers(K))
        rows.append((ce, ca, qi, bag_ti))
    return rows


def rows_to_seqs(rows, order, target):
    seqs = []
    for ce, ca, qi, bag_ti in rows:
        if target == "bound":
            tgt = RWT(ca[qi])
        elif target == "bag_indep":
            tgt = RWT(ca[bag_ti])
        else:
            raise ValueError(target)
        seqs.append(mk_seq(order, ce, ca, qi, tgt))
    return seqs


def common_order(seed, epoch, n):
    return np.random.default_rng(7171700 + seed * 10007 + epoch).permutation(n)


def weight_vec(mode, answer_weight, dev):
    L = SL - 1
    w = torch.zeros(L, dtype=torch.float32, device=dev)
    if mode == "full":
        w[:] = 1.0; w[-1] = 0.0
    elif mode == "answer_only":
        w[IS_POS] = 1.0
    elif mode == "weighted_full":
        w[:] = 1.0; w[-1] = 0.0; w[IS_POS] = float(answer_weight)
    else:
        raise ValueError(mode)
    return w


def train_ep(model, opt, seqs, order_idx, dev, mode, answer_weight, bs):
    model.train()
    st = torch.tensor(seqs, dtype=torch.long)[torch.tensor(order_idx, dtype=torch.long)]
    L = SL - 1
    cm = Base.causal_mask(L, dev)
    pos_w = weight_vec(mode, answer_weight, dev)
    total = 0.0; denom_total = 0.0; ans_total = 0.0; ctx_total = 0.0; ctx_denom = 0.0; count = 0
    for i in range(0, st.size(0), bs):
        batch = st[i:i + bs].to(dev)
        inp, tgt = batch[:, :L], batch[:, 1:]
        logits = model(inp, cm)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none").view(tgt.shape)
        nonpad = (tgt != PAD).float()
        weights = nonpad * pos_w.view(1, -1)
        denom = weights.sum().clamp_min(1.0)
        loss = (ce * weights).sum() / denom
        opt.zero_grad(); loss.backward(); opt.step()
        total += float((ce * weights).sum().item()); denom_total += float(denom.item())
        ans_total += float(ce[:, IS_POS].sum().item()); count += int(ce.size(0))
        cmask = nonpad.clone(); cmask[:, IS_POS] = 0.0
        ctx_total += float((ce * cmask).sum().item()); ctx_denom += float(cmask.sum().item())
    return dict(weighted_loss=total / max(denom_total, 1), answer_ce=ans_total / max(count, 1), context_ce=ctx_total / max(ctx_denom, 1))


# ------------------------------ probes ------------------------------
def gen_std(rng, n, entities, order):
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(rng.integers(K))
        ps.append(dict(s=mk_seq(order, ce, ca, qi, PAD), ce=ce, ca=ca, qi=qi, correct=ca[qi]))
    return ps


def gen_held(rng, n, order):
    ps = []
    for _ in range(n):
        he = int(rng.choice(HELD_E))
        others = rng.choice(TRAIN_E, K - 1, replace=False).tolist()
        ce = list(rng.permutation(others + [he]))
        qi = ce.index(he)
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        ps.append(dict(s=mk_seq(order, ce, ca, qi, PAD), ce=ce, ca=ca, qi=qi, correct=ca[qi]))
    return ps


def gen_bswap(rng, n, entities, order):
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(rng.integers(K))
        si = qi
        while si == qi:
            si = int(rng.integers(K))
        ca2 = list(ca); ca2[qi], ca2[si] = ca2[si], ca2[qi]
        ps.append(dict(s1=mk_seq(order, ce, ca, qi, PAD), s2=mk_seq(order, ce, ca2, qi, PAD), c1=ca[qi], c2=ca2[qi]))
    return ps


def gen_qswap(rng, n, entities, order):
    ps = []
    for _ in range(n):
        ce = rng.choice(entities, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi1 = int(rng.integers(K)); qi2 = qi1
        while qi2 == qi1:
            qi2 = int(rng.integers(K))
        ps.append(dict(s1=mk_seq(order, ce, ca, qi1, PAD), s2=mk_seq(order, ce, ca, qi2, PAD), c1=ca[qi1], c2=ca[qi2]))
    return ps


def gen_corrupt(rng, n, entities, order):
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
        ps.append(dict(s_orig=mk_seq(order, ce, ca, qi, PAD), s_qc=mk_seq(order, ce, ca_qc, qi, PAD),
                       s_dc=mk_seq(order, ce, ca_dc, qi, PAD), ca=ca, qi=qi, correct=ca[qi], novel=novel))
    return ps


def get_logits(model, seqs, dev):
    st = torch.tensor(seqs, dtype=torch.long, device=dev)
    L = SL - 1; cm = Base.causal_mask(L, dev)
    ps, lps, los = [], [], []
    with torch.no_grad():
        for i in range(0, st.size(0), 256):
            logits = model(st[i:i + 256, :L], cm)[:, IS_POS, :]
            los.append(logits); ps.append(F.softmax(logits, dim=-1)); lps.append(F.log_softmax(logits, dim=-1))
    return torch.cat(ps), torch.cat(lps), torch.cat(los)


def tie_safe(logits, correct_idx):
    vals = logits.tolist(); mx = max(vals); wins = [i for i, v in enumerate(vals) if abs(v - mx) <= 1e-10]
    return 1.0 / len(wins) if correct_idx in wins else 0.0


def eval_std(model, probes, dev):
    p, lp, lo = get_logits(model, [x["s"] for x in probes], dev)
    M = {"correct_nll": [], "bag_mass": [], "within_bag_nll": [], "ctx_top1": [], "query_margin": [], "mrr": []}
    for j, pr in enumerate(probes):
        ca, qi, corr = pr["ca"], pr["qi"], pr["correct"]
        ct = RWT(corr)
        M["correct_nll"].append(-lp[j, ct].item())
        bag_toks = [RWT(a) for a in ca]
        bm = p[j, bag_toks].sum().item()
        M["bag_mass"].append(bm)
        M["within_bag_nll"].append(-math.log(max(p[j, ct].item() / max(bm, 1e-30), 1e-30)))
        bag_logits = torch.tensor([lo[j, RWT(a)].item() for a in ca])
        M["ctx_top1"].append(tie_safe(bag_logits, qi))
        dec = [lo[j, RWT(a)].item() for ii, a in enumerate(ca) if ii != qi]
        M["query_margin"].append(lo[j, ct].item() - float(np.mean(dec)))
        rank = int((p[j, RWT_TOKS].argsort(descending=True) == corr).nonzero(as_tuple=True)[0].item()) + 1
        M["mrr"].append(1.0 / rank)
    return {k: round(float(np.mean(v)), 6) for k, v in M.items()}


def eval_bswap(model, pairs, dev):
    _, _, lo1 = get_logits(model, [x["s1"] for x in pairs], dev)
    _, _, lo2 = get_logits(model, [x["s2"] for x in pairs], dev)
    vals = []
    for j, pr in enumerate(pairs):
        ct1, ct2 = RWT(pr["c1"]), RWT(pr["c2"])
        vals.append(0.5 * ((lo1[j, ct1] - lo1[j, ct2]).item() + (lo2[j, ct2] - lo2[j, ct1]).item()))
    arr = np.array(vals)
    return dict(mean=round(float(arr.mean()), 6), frac_pos=round(float((arr > 0).mean()), 6), sem=round(float(arr.std(ddof=0) / math.sqrt(len(arr))), 6))


def eval_qswap(model, pairs, dev):
    p1, _, lo1 = get_logits(model, [x["s1"] for x in pairs], dev)
    p2, _, lo2 = get_logits(model, [x["s2"] for x in pairs], dev)
    vals = []; both = 0
    for j, pr in enumerate(pairs):
        ct1, ct2 = RWT(pr["c1"]), RWT(pr["c2"])
        vals.append(0.5 * ((lo1[j, ct1] - lo1[j, ct2]).item() + (lo2[j, ct2] - lo2[j, ct1]).item()))
        top1 = RWT_TOKS[p1[j, RWT_TOKS].argmax().item()]
        top2 = RWT_TOKS[p2[j, RWT_TOKS].argmax().item()]
        both += int(top1 == ct1 and top2 == ct2)
    arr = np.array(vals)
    return dict(margin=round(float(arr.mean()), 6), frac_pos=round(float((arr > 0).mean()), 6), both=round(float(both / len(pairs)), 6), sem=round(float(arr.std(ddof=0) / math.sqrt(len(arr))), 6))


def eval_corrupt(model, probes, dev):
    po, _, _ = get_logits(model, [x["s_orig"] for x in probes], dev)
    pq, _, _ = get_logits(model, [x["s_qc"] for x in probes], dev)
    pd, _, _ = get_logits(model, [x["s_dc"] for x in probes], dev)
    vals = []
    for j, pr in enumerate(probes):
        nt = RWT(pr["novel"])
        vals.append((pq[j, nt] - po[j, nt]).item() - (pd[j, nt] - po[j, nt]).item())
    arr = np.array(vals)
    return dict(novel_selectivity=round(float(arr.mean()), 6), sem=round(float(arr.std(ddof=0) / math.sqrt(len(arr))), 6))


def evaluate(model, probes, order, dev):
    P = probes[order]
    return dict(std=eval_std(model, P["std"], dev), held=eval_std(model, P["held"], dev),
                bswap=eval_bswap(model, P["bswap"], dev), qswap=eval_qswap(model, P["qswap"], dev),
                corrupt=eval_corrupt(model, P["corrupt"], dev))


ARMS = [
    dict(name="orig_ans_only_500", order="original", target="bound", phases=[dict(mode="answer_only", answer_weight=1.0, epochs=500)]),
    dict(name="qfirst_full_500", order="query_first", target="bound", phases=[dict(mode="full", answer_weight=1.0, epochs=500)]),
    dict(name="qfirst_ans_only_500", order="query_first", target="bound", phases=[dict(mode="answer_only", answer_weight=1.0, epochs=500)]),
    dict(name="qfirst_w16_500", order="query_first", target="bound", phases=[dict(mode="weighted_full", answer_weight=16.0, epochs=500)]),
    dict(name="qfirst_bag_ans_500", order="query_first", target="bag_indep", phases=[dict(mode="answer_only", answer_weight=1.0, epochs=500)]),
]


def summarize(data):
    S = {}
    for name in data["config"]["arms"]:
        finals = []
        for sd in data["config"]["seeds"]:
            rs = [r for r in data["records"] if r["arm"] == name and r["seed"] == sd]
            if rs: finals.append(rs[-1])
        def coll(path):
            vals = []
            for r in finals:
                obj = r
                for p in path: obj = obj[p]
                vals.append(float(obj))
            return dict(mean=round(float(np.mean(vals)), 6), std=round(float(np.std(vals)), 6), vals=[round(float(v), 6) for v in vals])
        if finals:
            S[name] = dict(correct_nll=coll(["std", "correct_nll"]), bag_mass=coll(["std", "bag_mass"]),
                           within_bag_nll=coll(["std", "within_bag_nll"]), ctx_top1=coll(["std", "ctx_top1"]),
                           query_margin=coll(["std", "query_margin"]), bswap=coll(["bswap", "mean"]),
                           bswap_frac=coll(["bswap", "frac_pos"]), qswap_margin=coll(["qswap", "margin"]),
                           qswap_frac=coll(["qswap", "frac_pos"]), qswap_both=coll(["qswap", "both"]),
                           corrupt_select=coll(["corrupt", "novel_selectivity"]), answer_ce=coll(["answer_ce"]),
                           context_ce=coll(["context_ce"]))
    return S


def make_figure(data, figpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    plots = [("std", "correct_nll", "Correct NLL", "NLL"), ("std", "ctx_top1", "Tie-safe top among context attrs", "Score"),
             ("std", "query_margin", "Query vs decoy context margin", "Logit margin"), ("bswap", "mean", "B-swap margin", "Logit margin"),
             ("qswap", "margin", "Q-swap margin", "Logit margin"), ("corrupt", "novel_selectivity", "Query-novel selectivity", "Probability diff")]
    colors = {"orig_ans_only_500":"tab:gray", "qfirst_full_500":"tab:green", "qfirst_ans_only_500":"tab:blue", "qfirst_w16_500":"tab:purple", "qfirst_bag_ans_500":"tab:orange"}
    for arm in data["config"]["arms"]:
        seed_records = [[r for r in data["records"] if r["arm"] == arm and r["seed"] == sd] for sd in data["config"]["seeds"]]
        seed_records = [x for x in seed_records if x]
        if not seed_records: continue
        epochs = sorted(set(r["epoch"] for rs in seed_records for r in rs))
        for ax, (grp, key, title, ylabel) in zip(axes.flat, plots):
            xs, ys = [], []
            for ep in epochs:
                vals = []
                for rs in seed_records:
                    rr = next((r for r in rs if r["epoch"] == ep), None)
                    if rr is None: continue
                    obj = rr
                    for p in [grp, key]: obj = obj[p]
                    vals.append(obj)
                if vals:
                    xs.append(ep); ys.append(float(np.mean(vals)))
            ax.plot(xs, ys, label=arm, color=colors.get(arm), lw=1.7)
    axes[0,0].axhline(math.log(4), color="black", ls="--", alpha=0.35, label="bag uniform")
    axes[0,0].axhline(math.log(12), color="black", ls=":", alpha=0.35, label="marginal")
    axes[0,1].axhline(0.25, color="black", ls="--", alpha=0.35, label="chance")
    for idx in [3,4,5]: axes.flat[idx].axhline(0, color="black", ls="--", alpha=0.35, label="zero")
    for ax, (_, _, title, ylabel) in zip(axes.flat, plots):
        ax.set_title(title, fontsize=10); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(fontsize=6)
    fig.suptitle("research: Query-before-context order and orbit binding", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.96])
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/query_first_binding")
    ap.add_argument("--fig", default="experiments/archive/functional_learning/figures/query_first_binding.png")
    ap.add_argument("--seeds", default="42,43,100")
    A = ap.parse_args()
    seeds = [42] if A.smoke else [int(x) for x in A.seeds.split(",") if x.strip()]
    arms = ARMS[:3] if A.smoke else ARMS
    if A.smoke:
        tmp=[]
        for a in arms:
            ph=dict(a["phases"][0]); ph["epochs"]=min(6, ph["epochs"])
            tmp.append(dict(name=a["name"], order=a["order"], target=a["target"], phases=[ph]))
        arms=tmp
    n_train = 100 if A.smoke else 500
    eval_every = 2 if A.smoke else 25
    cfg = dict(seeds=seeds, arms=[a["name"] for a in arms], n_train=n_train, eval_every=eval_every,
               d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64, log4=math.log(4), log12=math.log(12))
    erng = np.random.default_rng(160016)
    probes = {}
    for order in ["original", "query_first"]:
        probes[order] = dict(std=gen_std(erng, 512 if not A.smoke else 96, TRAIN_E, order),
                             held=gen_held(erng, 256 if not A.smoke else 48, order),
                             bswap=gen_bswap(erng, 512 if not A.smoke else 96, TRAIN_E, order),
                             qswap=gen_qswap(erng, 256 if not A.smoke else 48, TRAIN_E, order),
                             corrupt=gen_corrupt(erng, 256 if not A.smoke else 48, TRAIN_E, order))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    data = dict(config=cfg, arm_specs=arms, records=[])
    t0=time.time(); Path(A.data).mkdir(parents=True, exist_ok=True)
    for sd in seeds:
        print(f"\n{'='*72}\nSeed {sd}\n{'='*72}")
        for spec in arms:
            torch.manual_seed(sd); np.random.seed(sd)
            model = Base.CLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            epg=0; ts=time.time()
            print(f"  Arm {spec['name']}: order={spec['order']} target={spec['target']} phases={spec['phases']}")
            for pi, ph in enumerate(spec["phases"]):
                for pe in range(1, ph["epochs"]+1):
                    epg += 1
                    rows = make_epoch_rows(sd, epg, n_train)
                    seqs = rows_to_seqs(rows, spec["order"], spec["target"])
                    idx = common_order(sd, epg, n_train)
                    info = train_ep(model, opt, seqs, idx, dev, ph["mode"], ph.get("answer_weight",1.0), cfg["bs"])
                    if epg == 1 or epg % eval_every == 0 or pe == ph["epochs"]:
                        met = evaluate(model, probes, spec["order"], dev)
                        rec = dict(arm=spec["name"], seed=sd, epoch=epg, phase_idx=pi, phase_epoch=pe,
                                   mode=ph["mode"], weighted_loss=round(info["weighted_loss"],6),
                                   answer_ce=round(info["answer_ce"],6), context_ce=round(info["context_ce"],6))
                        rec.update(met); data["records"].append(rec)
                        print(f"    e={epg:3d} {ph['mode']:<13s} ansCE={info['answer_ce']:.3f} ctxCE={info['context_ce']:.3f} "
                              f"cnll={met['std']['correct_nll']:.3f} bagM={met['std']['bag_mass']:.3f} top4={met['std']['ctx_top1']:.3f} "
                              f"qm={met['std']['query_margin']:+.3f} bs={met['bswap']['mean']:+.3f} qsm={met['qswap']['margin']:+.3f} "
                              f"qfrac={met['qswap']['frac_pos']:.3f} sel={met['corrupt']['novel_selectivity']:+.3f} ({time.time()-ts:.1f}s)")
            print(f"    Params: {sum(p.numel() for p in model.parameters()):,d}")
    data["summary"] = summarize(data)
    outp = Path(A.data)/"results.json"
    with open(outp,"w") as f: json.dump(data,f,indent=2)
    make_figure(data, A.fig)
    print("\nSUMMARY")
    for name, s in data["summary"].items():
        print(f"  {name:20s} cnll={s['correct_nll']['mean']:.3f} bagM={s['bag_mass']['mean']:.3f} top4={s['ctx_top1']['mean']:.3f} "
              f"qm={s['query_margin']['mean']:+.3f} bs={s['bswap']['mean']:+.3f} bfrac={s['bswap_frac']['mean']:.3f} "
              f"qsm={s['qswap_margin']['mean']:+.3f} qfrac={s['qswap_frac']['mean']:.3f} qboth={s['qswap_both']['mean']:.3f} sel={s['corrupt_select']['mean']:+.3f}")
    print(json.dumps(dict(status="DONE", out=str(outp), figure=A.fig, elapsed=round(time.time()-t0,1))))


if __name__ == "__main__":
    main()
