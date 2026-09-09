#!/usr/bin/env python3
"""research: budget-matched full-objective comparison after query-first preparation.

research showed that a query-first answer-only bound checkpoint can lose binding under
full next-token training and then recover by 500 branch epochs.  That result does
not by itself establish data-efficiency because the prepared arm received the
answer-only acquisition epochs *plus* 500 full-objective epochs, while research's
fresh full-objective arm received only 500 epochs.

This experiment compares cumulative budgets directly.  For each seed we train from
identical initialization and identical epoch streams through a common total budget:

  fresh_qfirst_full:
      query-first bound target, full next-token objective for all epochs.
  prep_bound_ans_then_full:
      query-first bound target, answer-only for P seed-specific prep epochs,
      then query-first bound target, full next-token objective.
  prep_bag_ans_then_full:
      query-first bag-independent target, answer-only for the same P epochs,
      then query-first bound target, full next-token objective.  This is an
      equally trained but unbound preparation.
  prep_ctx_then_full:
      query-first bound sequence, context-only (no answer loss) for P epochs,
      then query-first bound target, full objective.  This tests generic
      sequence/model maturation without answer/bag-selector pressure.

The primary readout is not just final success.  We track per-seed acquisition
curves, first strong-binding epoch under full-objective training, matched-budget
values at P+500 (the research comparison point), final values at 1000 epochs, and
held-entity probes including held B-swap and held Q-swap.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import loss_allocation_binding as Base
import query_first_binding as S16
import binding_branching as S17

K, N_ATTR, VOCAB = Base.K, Base.N_ATTR, Base.VOCAB
TRAIN_E, HELD_E = Base.TRAIN_E, Base.HELD_E
PAD = Base.PAD
SL, IS_POS = 18, 15

# research branch-start preparation epochs.  These are the deterministic acquisition
# thresholds from the previous experiment and define the cumulative-budget contrast.
DEFAULT_PREP_EPOCHS = {42: 300, 43: 500, 100: 400}

ARMS = [
    dict(name="fresh_qfirst_full", prep_mode=None, prep_target="bound"),
    dict(name="prep_bound_ans_then_full", prep_mode="answer_only", prep_target="bound"),
    dict(name="prep_bag_ans_then_full", prep_mode="answer_only", prep_target="bag_indep"),
    dict(name="prep_ctx_then_full", prep_mode="context_only", prep_target="bound"),
]


def parse_prep_epochs(text):
    if not text:
        return dict(DEFAULT_PREP_EPOCHS)
    out = {}
    for part in text.split(','):
        if not part.strip():
            continue
        k, v = part.split(':')
        out[int(k)] = int(v)
    return out


def mk_held_bswap(rng, n, order):
    """Binding-swap probes where the queried entity is held-out."""
    ps = []
    for _ in range(n):
        he = int(rng.choice(HELD_E))
        others = rng.choice(TRAIN_E, K - 1, replace=False).tolist()
        ce = list(rng.permutation(others + [he]))
        qi = ce.index(he)
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        si = qi
        while si == qi:
            si = int(rng.integers(K))
        ca2 = list(ca); ca2[qi], ca2[si] = ca2[si], ca2[qi]
        ps.append(dict(s1=S16.mk_seq(order, ce, ca, qi, PAD),
                       s2=S16.mk_seq(order, ce, ca2, qi, PAD),
                       c1=ca[qi], c2=ca2[qi]))
    return ps


def mk_held_qswap(rng, n, order):
    """Query-swap probes with one held query and one trained query in the same bag."""
    ps = []
    for _ in range(n):
        he = int(rng.choice(HELD_E))
        others = rng.choice(TRAIN_E, K - 1, replace=False).tolist()
        ce = list(rng.permutation(others + [he]))
        qi1 = ce.index(he)
        qi2 = qi1
        while qi2 == qi1:
            qi2 = int(rng.integers(K))
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        ps.append(dict(s1=S16.mk_seq(order, ce, ca, qi1, PAD),
                       s2=S16.mk_seq(order, ce, ca, qi2, PAD),
                       c1=ca[qi1], c2=ca[qi2]))
    return ps


def mk_held_corrupt(rng, n, order):
    """Corruption probes where the queried entity is held-out."""
    ps = []
    for _ in range(n):
        he = int(rng.choice(HELD_E))
        others = rng.choice(TRAIN_E, K - 1, replace=False).tolist()
        ce = list(rng.permutation(others + [he]))
        qi = ce.index(he)
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        novel = int(rng.integers(N_ATTR))
        while novel in ca:
            novel = int(rng.integers(N_ATTR))
        ca_qc = list(ca); ca_qc[qi] = novel
        di = qi
        while di == qi:
            di = int(rng.integers(K))
        ca_dc = list(ca); ca_dc[di] = novel
        ps.append(dict(s_orig=S16.mk_seq(order, ce, ca, qi, PAD),
                       s_qc=S16.mk_seq(order, ce, ca_qc, qi, PAD),
                       s_dc=S16.mk_seq(order, ce, ca_dc, qi, PAD),
                       ca=ca, qi=qi, correct=ca[qi], novel=novel))
    return ps


def make_probes(seed=180018, n_std=512, n_small=256):
    rng = np.random.default_rng(seed)
    probes = {}
    for order in ["query_first", "original"]:
        probes[order] = dict(
            std=S16.gen_std(rng, n_std, TRAIN_E, order),
            held=S16.gen_held(rng, n_small, order),
            bswap=S16.gen_bswap(rng, n_std, TRAIN_E, order),
            qswap=S16.gen_qswap(rng, n_small, TRAIN_E, order),
            corrupt=S16.gen_corrupt(rng, n_small, TRAIN_E, order),
            held_bswap=mk_held_bswap(rng, n_small, order),
            held_qswap=mk_held_qswap(rng, n_small, order),
            held_corrupt=mk_held_corrupt(rng, n_small, order),
        )
    return probes


def eval_extended(model, probes, order, dev):
    base = S16.evaluate(model, probes, order, dev)
    P = probes[order]
    base["held_bswap"] = S16.eval_bswap(model, P["held_bswap"], dev)
    base["held_qswap"] = S16.eval_qswap(model, P["held_qswap"], dev)
    base["held_corrupt"] = S16.eval_corrupt(model, P["held_corrupt"], dev)
    return base


def should_eval(epoch, total_epochs, eval_every, prep_epoch):
    special = {1, total_epochs, prep_epoch, prep_epoch + 1, prep_epoch + 25, prep_epoch + 50}
    return epoch in special or (epoch % eval_every == 0)


def strong_binding(met):
    return (met["std"]["ctx_top1"] >= 0.95 and
            met["bswap"]["mean"] >= 5.0 and
            met["qswap"]["margin"] >= 5.0 and
            met["corrupt"]["novel_selectivity"] >= 0.8)


def mid_binding(met):
    return (met["std"]["ctx_top1"] >= 0.50 and
            met["bswap"]["mean"] >= 1.0 and
            met["qswap"]["margin"] >= 1.0)


def train_arm(seed, arm, prep_epoch, total_epochs, n_train, eval_every, probes, cfg, dev, cm_std):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Base.CLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    records = []
    ts = time.time()

    for ep in range(1, total_epochs + 1):
        if arm["prep_mode"] is not None and ep <= prep_epoch:
            mode = arm["prep_mode"]
            target = arm["prep_target"]
            phase = "prep"
        else:
            mode = "full"
            target = "bound"
            phase = "full"
        rows = S16.make_epoch_rows(seed, ep, n_train)
        seqs = S16.rows_to_seqs(rows, "query_first", target)
        idx = S16.common_order(seed, ep, n_train)
        info = S17.train_one_epoch(model, opt, seqs, idx, dev, mode, cfg["bs"], cm_std)

        if should_eval(ep, total_epochs, eval_every, prep_epoch):
            met = eval_extended(model, probes, "query_first", dev)
            rec = dict(seed=seed, arm=arm["name"], epoch=ep, prep_epoch=prep_epoch,
                       phase=phase, mode=mode, target=target,
                       wloss=info["wloss"], answer_ce=info["ans_ce"], context_ce=info["ctx_ce"],
                       elapsed=round(time.time() - ts, 1))
            rec.update(met)
            records.append(rec)
            print(f"  {arm['name']:<26s} e={ep:4d} phase={phase:<4s} mode={mode:<12s} "
                  f"cnll={met['std']['correct_nll']:.3f} top4={met['std']['ctx_top1']:.3f} "
                  f"bs={met['bswap']['mean']:+.3f} qs={met['qswap']['margin']:+.3f} "
                  f"held4={met['held']['ctx_top1']:.3f} hbs={met['held_bswap']['mean']:+.3f} "
                  f"sel={met['corrupt']['novel_selectivity']:+.3f} ({time.time()-ts:.0f}s)", flush=True)
    return records, model.state_dict()


def first_epoch(records, predicate, min_epoch=1):
    for r in sorted(records, key=lambda x: x["epoch"]):
        if r["epoch"] >= min_epoch and predicate(r):
            return int(r["epoch"])
    return None


def nearest_record(records, epoch):
    if not records:
        return None
    return min(records, key=lambda r: abs(int(r["epoch"]) - epoch))


def collect_metric(r, path):
    obj = r
    for k in path:
        obj = obj[k]
    return float(obj)


def summarize(data):
    seeds = data["config"]["seeds"]
    arms = [a["name"] for a in data["arm_specs"]]
    summary = {"by_seed": {}, "by_arm": {}}
    for sd in seeds:
        summary["by_seed"][str(sd)] = {}
        prep = data["config"]["prep_epochs"][str(sd)]
        total = prep + 500
        for arm in arms:
            rs = [r for r in data["records"] if r["seed"] == sd and r["arm"] == arm]
            if not rs:
                continue
            final = max(rs, key=lambda r: r["epoch"])
            at_p500 = nearest_record(rs, total)
            first_mid = first_epoch(rs, lambda r: mid_binding(r), min_epoch=1)
            first_strong = first_epoch(rs, lambda r: strong_binding(r), min_epoch=1)
            first_strong_after_full = first_epoch(rs, lambda r: strong_binding(r), min_epoch=prep + 1)
            first_mid_after_full = first_epoch(rs, lambda r: mid_binding(r), min_epoch=prep + 1)
            summary["by_seed"][str(sd)][arm] = dict(
                prep_epoch=prep,
                total=total,
                final_epoch=int(final["epoch"]),
                first_mid_epoch=first_mid,
                first_strong_epoch=first_strong,
                first_mid_after_full_epoch=first_mid_after_full,
                first_strong_after_full_epoch=first_strong_after_full,
                final=dict(top4=collect_metric(final,["std","ctx_top1"]),
                           bswap=collect_metric(final,["bswap","mean"]),
                           qswap=collect_metric(final,["qswap","margin"]),
                           selectivity=collect_metric(final,["corrupt","novel_selectivity"]),
                           nll=collect_metric(final,["std","correct_nll"]),
                           held_top4=collect_metric(final,["held","ctx_top1"]),
                           held_bswap=collect_metric(final,["held_bswap","mean"]),
                           held_qswap=collect_metric(final,["held_qswap","margin"])),
                at_step017_total=None if at_p500 is None else dict(
                           epoch=int(at_p500["epoch"]),
                           top4=collect_metric(at_p500,["std","ctx_top1"]),
                           bswap=collect_metric(at_p500,["bswap","mean"]),
                           qswap=collect_metric(at_p500,["qswap","margin"]),
                           selectivity=collect_metric(at_p500,["corrupt","novel_selectivity"]),
                           nll=collect_metric(at_p500,["std","correct_nll"]),
                           held_top4=collect_metric(at_p500,["held","ctx_top1"]),
                           held_bswap=collect_metric(at_p500,["held_bswap","mean"]),
                           held_qswap=collect_metric(at_p500,["held_qswap","margin"])),
            )
    # Arm-level counts/means.
    for arm in arms:
        finals = []
        p500s = []
        strongs = []
        mid_after = []
        strong_after = []
        for sd in seeds:
            b = summary["by_seed"].get(str(sd), {}).get(arm)
            if not b:
                continue
            finals.append(b["final"])
            if b["at_step017_total"] is not None:
                p500s.append(b["at_step017_total"])
            strongs.append(b["first_strong_epoch"])
            mid_after.append(b["first_mid_after_full_epoch"])
            strong_after.append(b["first_strong_after_full_epoch"])
        def mean(vals, key):
            if not vals:
                return None
            return round(float(np.mean([v[key] for v in vals])), 6)
        summary["by_arm"][arm] = dict(
            n_seeds=len(finals),
            final_success_count=sum(1 for v in finals if v["top4"] >= 0.95 and v["bswap"] >= 5 and v["qswap"] >= 5),
            p500_success_count=sum(1 for v in p500s if v["top4"] >= 0.95 and v["bswap"] >= 5 and v["qswap"] >= 5),
            final_mean_top4=mean(finals,"top4"), final_mean_bswap=mean(finals,"bswap"), final_mean_qswap=mean(finals,"qswap"),
            final_mean_held_top4=mean(finals,"held_top4"), final_mean_held_bswap=mean(finals,"held_bswap"), final_mean_held_qswap=mean(finals,"held_qswap"),
            p500_mean_top4=mean(p500s,"top4"), p500_mean_bswap=mean(p500s,"bswap"), p500_mean_qswap=mean(p500s,"qswap"),
            p500_mean_held_top4=mean(p500s,"held_top4"), p500_mean_held_bswap=mean(p500s,"held_bswap"), p500_mean_held_qswap=mean(p500s,"held_qswap"),
            first_strong_epochs=strongs,
            first_mid_after_full_epochs=mid_after,
            first_strong_after_full_epochs=strong_after,
        )
    return summary


def make_figure(data, figpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a["name"] for a in data["arm_specs"]]
    colours = {
        "fresh_qfirst_full": "tab:green",
        "prep_bound_ans_then_full": "tab:blue",
        "prep_bag_ans_then_full": "tab:orange",
        "prep_ctx_then_full": "tab:purple",
    }
    plots = [
        (["std","ctx_top1"], "Tie-safe top4", "score", 0.25, 0.95),
        (["bswap","mean"], "B-swap margin", "logit margin", 0.0, 5.0),
        (["qswap","margin"], "Q-swap margin", "logit margin", 0.0, 5.0),
        (["held","ctx_top1"], "Held-query top4", "score", 0.25, 0.95),
        (["held_bswap","mean"], "Held-query B-swap", "logit margin", 0.0, 5.0),
        (["std","correct_nll"], "Correct NLL", "nats", math.log(4), None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    seeds = data["config"]["seeds"]

    for ax, (path, title, ylabel, base_line, strong_line) in zip(axes.flat, plots):
        for arm in arms:
            xs = sorted(set(r["epoch"] for r in data["records"] if r["arm"] == arm))
            xout, yout = [], []
            for ep in xs:
                vals = []
                for sd in seeds:
                    rr = [r for r in data["records"] if r["arm"] == arm and r["seed"] == sd and r["epoch"] == ep]
                    if rr:
                        vals.append(collect_metric(rr[0], path))
                if vals:
                    xout.append(ep); yout.append(float(np.mean(vals)))
            ax.plot(xout, yout, color=colours.get(arm), lw=1.8, label=arm.replace('_',' '))
        if base_line is not None:
            ax.axhline(base_line, color="black", ls="--", alpha=0.30)
        if strong_line is not None:
            ax.axhline(strong_line, color="black", ls=":", alpha=0.25)
        ax.set_title(title, fontsize=10); ax.set_xlabel("cumulative epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(fontsize=6)
    fig.suptitle("research: Budget-matched query-first full objective after preparation", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.96])
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_note(data, out_path):
    S = data["summary"]
    lines = []
    lines.append('# research result: budget-matched full-objective comparison')
    lines.append('')
    lines.append('This note is auto-generated from `data/budget_matched_full_objective/results.json`. It compares fresh query-first full-objective training with bound and unbound preparation under the same cumulative epoch budget.')
    lines.append('')
    lines.append('## Arm-level summary')
    lines.append('')
    lines.append('| arm | success at P+500 | final success | mean P+500 top4 | mean final top4 | mean held final top4 | first strong epochs | first strong after full epochs |')
    lines.append('|---|---:|---:|---:|---:|---:|---|---|')
    for arm, sm in S['by_arm'].items():
        lines.append(f"| {arm} | {sm['p500_success_count']}/{sm['n_seeds']} | {sm['final_success_count']}/{sm['n_seeds']} | {sm['p500_mean_top4']:.3f} | {sm['final_mean_top4']:.3f} | {sm['final_mean_held_top4']:.3f} | {sm['first_strong_epochs']} | {sm['first_strong_after_full_epochs']} |")
    lines.append('')
    lines.append('## Per-seed matched research-total readout (epoch P+500)')
    lines.append('')
    lines.append('| seed | arm | epoch | top4 | B-swap | Q-swap | held_top4 | held_B-swap | held_Q-swap |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|')
    for sd, armsm in S['by_seed'].items():
        for arm, sm in armsm.items():
            r = sm['at_step017_total']
            if r is None:
                continue
            lines.append(f"| {sd} | {arm} | {r['epoch']} | {r['top4']:.3f} | {r['bswap']:.3f} | {r['qswap']:.3f} | {r['held_top4']:.3f} | {r['held_bswap']:.3f} | {r['held_qswap']:.3f} |")
    lines.append('')
    lines.append('## Per-seed final readout')
    lines.append('')
    lines.append('| seed | arm | epoch | top4 | B-swap | Q-swap | held_top4 | held_B-swap | held_Q-swap | first strong | first strong after full |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for sd, armsm in S['by_seed'].items():
        for arm, sm in armsm.items():
            r = sm['final']
            fs = sm['first_strong_epoch']
            fsaf = sm['first_strong_after_full_epoch']
            lines.append(f"| {sd} | {arm} | {sm['final_epoch']} | {r['top4']:.3f} | {r['bswap']:.3f} | {r['qswap']:.3f} | {r['held_top4']:.3f} | {r['held_bswap']:.3f} | {r['held_qswap']:.3f} | {'' if fs is None else fs} | {'' if fsaf is None else fsaf} |")
    lines.append('')
    lines.append('## Interpretation slots for the researcher')
    lines.append('')
    lines.append('- If fresh full reaches the same strong-binding state by the same cumulative budget, the endpoint 3/3 versus 1/3 result from research cannot be used as a curriculum-efficiency result; only timing and held/generalization differences remain.')
    lines.append('- If bound preparation reaches strong full-objective binding earlier than fresh full and earlier than unbound bag/context preparation, that supports retained selector structure increasing the value of later full-objective examples.')
    lines.append('- If unbound preparation matches bound preparation, the advantage is more likely generic representation/bag-output pretraining rather than retained entity-specific selector knowledge.')
    lines.append('- Held-entity B/Q-swap probes are included here because held standard top4 alone can mix token-generalization with bag/family effects.')
    Path(out_path).write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--data', default=None)
    ap.add_argument('--fig', default=None)
    ap.add_argument('--seeds', default='42,43,100')
    ap.add_argument('--prep-epochs', default='')
    ap.add_argument('--total-epochs', type=int, default=1000)
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--n-train', type=int, default=500)
    ap.add_argument('--save-checkpoints', action='store_true')
    A = ap.parse_args()

    ws = _public_path('experiments/archive/functional_learning')
    if A.data is None:
        A.data = str(ws / 'data' / 'budget_matched_full_objective')
    if A.fig is None:
        A.fig = str(ws / 'figures' / 'budget_matched_full_objective.png')

    seeds = [42] if A.smoke else [int(x) for x in A.seeds.split(',') if x.strip()]
    prep_epochs_in = parse_prep_epochs(A.prep_epochs)
    if A.smoke:
        total_epochs = 20
        eval_every = 5
        n_train = 50
        prep_epochs = {sd: 8 for sd in seeds}
        arms = ARMS[:3]
        n_std, n_small = 96, 48
    else:
        total_epochs = A.total_epochs
        eval_every = A.eval_every
        n_train = A.n_train
        prep_epochs = {sd: int(prep_epochs_in.get(sd, DEFAULT_PREP_EPOCHS.get(sd, 400))) for sd in seeds}
        arms = ARMS
        n_std, n_small = 512, 256

    cfg = dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64,
               seeds=seeds, total_epochs=total_epochs, eval_every=eval_every,
               n_train=n_train, prep_epochs={str(k): int(v) for k, v in prep_epochs.items()},
               threshold=dict(top4=0.95, bswap=5.0, qswap=5.0, selectivity=0.8))

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cm_std = S17.standard_causal_mask(SL - 1, dev)
    probes = make_probes(n_std=n_std, n_small=n_small)
    out_dir = Path(A.data); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    if A.save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {dev}")
    print(f"Seeds: {seeds}; total_epochs={total_epochs}; prep_epochs={prep_epochs}; n_train={n_train}")
    t0 = time.time()
    data = dict(config=cfg, arm_specs=copy.deepcopy(arms), records=[])

    for sd in seeds:
        print('\n' + '='*80)
        print(f'Seed {sd} (prep_epoch={prep_epochs[sd]})')
        print('='*80)
        for arm in arms:
            print(f"\nArm {arm['name']}")
            recs, state = train_arm(sd, arm, prep_epochs[sd], total_epochs, n_train,
                                    eval_every, probes, cfg, dev, cm_std)
            data['records'].extend(recs)
            if A.save_checkpoints:
                torch.save(state, ckpt_dir / f"seed{sd}_{arm['name']}_final.pt")

    data['summary'] = summarize(data)
    outp = out_dir / 'results.json'
    with open(outp, 'w') as f:
        json.dump(data, f, indent=2)
    make_figure(data, A.fig)
    note = ws / 'notes' / 'budget_matched_full_objective_result.md'
    write_note(data, note)

    print('\nSUMMARY')
    for arm, sm in data['summary']['by_arm'].items():
        print(f"  {arm:<28s} P+500 success={sm['p500_success_count']}/{sm['n_seeds']} "
              f"final success={sm['final_success_count']}/{sm['n_seeds']} "
              f"P+500 top4={sm['p500_mean_top4']:.3f} final top4={sm['final_mean_top4']:.3f} "
              f"held final top4={sm['final_mean_held_top4']:.3f} strong_epochs={sm['first_strong_epochs']}")
    print(json.dumps(dict(status='DONE', out=str(outp), figure=A.fig,
                          note=str(note), elapsed=round(time.time()-t0, 1))))


if __name__ == '__main__':
    main()
