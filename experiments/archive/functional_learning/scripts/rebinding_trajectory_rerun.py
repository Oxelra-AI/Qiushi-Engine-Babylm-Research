#!/usr/bin/env python3
"""Step013b: narrow typed-cue rerun of research with rebinding probes over time.

Purpose: determine whether the research typed-cue models ever acquire reusable
entity--attribute rebinding, or whether the late source-specific interaction is
only a contrast between overfit/confident heuristics after held-out competence
has deteriorated.

This script imports the exact research training substrate and the research paired
rebinding probe. It does NOT introduce within-sequence heterogeneity or any new
training ingredient.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import source_trigger as st
import rebinding_analysis as rb


def summarize(vals):
    arr = np.asarray(vals, dtype=float)
    return {"mean": round(float(arr.mean()), 5), "std": round(float(arr.std()), 5),
            "vals": [round(float(x), 5) for x in arr.tolist()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[42,43,100])
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--eval-epochs', nargs='+', type=int, default=[1,25,50,75,100,150,200,250,300])
    ap.add_argument('--backbone_n', type=int, default=100)
    ap.add_argument('--sub_n', type=int, default=400)
    ap.add_argument('--probe-n-per-held', type=int, default=160)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--nh', type=int, default=2)
    ap.add_argument('--nl', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='experiments/archive/functional_learning/data/rebinding_trajectory_typed')
    args = ap.parse_args()
    eval_epochs = sorted(set([e for e in args.eval_epochs if 1 <= e <= args.epochs] + [args.epochs]))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_data = {'status':'REBINDING_TRAJECTORY_TYPED', 'config':vars(args), 'device':dev, 'cue_typed':{}}
    cue_mode = 'typed'; eval_cue = st.REWRITE_CUE

    for seed in args.seeds:
        print(f"\n{'='*60}\nSeed {seed} typed rebinding trajectory\n{'='*60}", flush=True)
        dg_bb = st.DG(seed)
        bb_seqs = dg_bb.backbone(args.backbone_n, cue_mode)
        dg_sub = st.DG(seed * 1000 + 1)
        sub_draws = dg_sub.base_draws(args.sub_n)
        id_seqs = st.DG.ident_seqs(sub_draws, cue_mode)
        drng = np.random.default_rng(seed * 1000 + 5)
        up_seqs = st.DG.unpaired_seqs(sub_draws, cue_mode, drng)
        dg_probe = st.DG(seed * 2000)
        old_probes = dg_probe.probes_tu(60)
        rb_probes = rb.generate_rebinding_probes(seed * 3000 + 17, args.probe_n_per_held)
        L = st.SL - 1
        blocked_masks = st.build_blocked_masks(sub_draws, args.backbone_n, L)

        torch.manual_seed(seed)
        base_model = st.CLM(st.VOCAB, args.d, args.nh, args.nl, st.SL).to(dev)
        init_sd = {k:v.clone() for k,v in base_model.state_dict().items()}
        if seed == args.seeds[0]:
            print(f"Params: {sum(p.numel() for p in base_model.parameters()):,}", flush=True)
        seed_res = {}
        for arm in st.ARMS:
            t1 = time.time()
            arm_seqs = id_seqs if arm in ('ident_full','ident_blocked') else up_seqs
            seqs_t = torch.tensor(bb_seqs + arm_seqs, dtype=torch.long)
            m3d = blocked_masks if arm == 'ident_blocked' else None
            model = st.CLM(st.VOCAB, args.d, args.nh, args.nl, st.SL).to(dev)
            model.load_state_dict(init_sd)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
            curves = []
            for ep in range(1, args.epochs+1):
                tl = st.train_ep(model, opt, seqs_t, m3d, dev, args.nh)
                if ep in eval_epochs:
                    mt = st.eval_probes(model, old_probes, 's_t', eval_cue, dev)
                    mu = st.eval_probes(model, old_probes, 's_u', eval_cue, dev)
                    mns = st.eval_probes(model, old_probes, 's_ns', eval_cue, dev)
                    rr = rb.eval_rebinding(model, rb_probes, eval_cue, dev)
                    row = {'e':ep, 'tl':round(float(tl),5), 'T_old_scored':mt, 'U_old_scored':mu, 'NS':mns,
                           'rebinding':{k:{'mean':round(v['mean'],5),'std':round(v['std'],5)} for k,v in rr.items()}}
                    curves.append(row)
                    print(f"  {arm:15s} e={ep:3d} tl={tl:.3f} T_old_nll={mt['rwt_nll']:.3f} MRR={mt['rwt_mrr']:.3f} "
                          f"rb_Tmargin={rr['T_logit_old_minus_new']['mean']:+.3f} rb_Smargin={rr['S_logit_new_minus_old']['mean']:+.3f} "
                          f"pair={rr['rebinding_pair_success']['mean']:.3f}", flush=True)
            seed_res[arm] = {'curves':curves, 'elapsed':round(time.time()-t1,1)}
            del model
            if dev == 'cuda': torch.cuda.empty_cache()
        all_data['cue_typed'][f'seed{seed}'] = seed_res
        with open(outdir/'partial_results.json','w') as f: json.dump(all_data,f,indent=2)

    # Aggregate across seeds by epoch/arm
    agg = {}
    metrics = ['T_old_scored.rwt_nll','T_old_scored.rwt_wnll','T_old_scored.rwt_mrr',
               'rebinding.T_logit_old_minus_new','rebinding.S_logit_new_minus_old',
               'rebinding.binding_margin_sum','rebinding.rebinding_pair_success',
               'rebinding.S_new_nll','rebinding.S_new_minus_S_old_nll','rebinding.S_top_new']
    def get(row, path):
        x = row
        for p in path.split('.'):
            x = x[p]
        return x['mean'] if isinstance(x, dict) and 'mean' in x else x
    for ep_i, ep in enumerate(eval_epochs):
        er = {'e':ep}
        for arm in st.ARMS:
            ar = {}
            for m in metrics:
                vals = [get(all_data['cue_typed'][f'seed{s}'][arm]['curves'][ep_i], m) for s in args.seeds]
                ar[m] = summarize(vals)
            er[arm] = ar
        # old-target interaction and rebinding contrasts (not causal-isolated because UP is anti-identity)
        inter = {}
        for ia in ['ident_full','ident_blocked']:
            for m in ['rwt_nll','rwt_wnll','rwt_mrr']:
                vals=[]
                for s in args.seeds:
                    rowi = all_data['cue_typed'][f'seed{s}'][ia]['curves'][ep_i]
                    rowu = all_data['cue_typed'][f'seed{s}']['unpaired_src']['curves'][ep_i]
                    vals.append((rowi['T_old_scored'][m]-rowu['T_old_scored'][m]) - (rowi['U_old_scored'][m]-rowu['U_old_scored'][m]))
                inter[f'D_{ia}.{m}'] = summarize(vals)
        for comp in ['unpaired_src','ident_blocked']:
            for m in ['rebinding.S_logit_new_minus_old','rebinding.binding_margin_sum','rebinding.rebinding_pair_success','rebinding.S_new_nll']:
                vals=[]
                for s in args.seeds:
                    vals.append(get(all_data['cue_typed'][f'seed{s}']['ident_full']['curves'][ep_i], m) -
                                get(all_data['cue_typed'][f'seed{s}'][comp]['curves'][ep_i], m))
                inter[f'ident_full_minus_{comp}.{m}'] = summarize(vals)
        er['interactions'] = inter
        agg[str(ep)] = er
    all_data['aggregate'] = agg
    all_data['elapsed'] = round(time.time()-t0,1)
    with open(outdir/'results.json','w') as f: json.dump(all_data,f,indent=2)

    print('\nAGGREGATE typed trajectory')
    for arm in st.ARMS:
        print(f"\n{arm}")
        for ep in eval_epochs:
            a=agg[str(ep)][arm]
            print(f"  e{ep:3d} oldNLL={a['T_old_scored.rwt_nll']['mean']:.2f} oldMRR={a['T_old_scored.rwt_mrr']['mean']:.2f} "
                  f"Tmargin={a['rebinding.T_logit_old_minus_new']['mean']:+.2f} Smargin={a['rebinding.S_logit_new_minus_old']['mean']:+.2f} "
                  f"pair={a['rebinding.rebinding_pair_success']['mean']:.3f} SnewNLL={a['rebinding.S_new_nll']['mean']:.2f}")
    print(json.dumps({'status':'REBINDING_TRAJECTORY_DONE','out':str(outdir/'results.json'),'elapsed':all_data['elapsed']}))

if __name__ == '__main__':
    main()
