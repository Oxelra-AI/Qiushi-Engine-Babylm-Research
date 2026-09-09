#!/usr/bin/env python3
"""Step013c: existing final-checkpoint panel for train-vs-held rebinding.

Loads research saved final models and evaluates same-token-bag rebinding for:
  * train_query_alltrain: query entity was trained in query role; all context entities train.
  * held_query_traindecoys: held query entity with trained decoys (research-style transfer).
  * train_query_helddecoy: trained query with at least one held nonquery entity.
  * held_query_heldpartner: held query swapped with the other held entity.

This checks whether weak research rebinding is caused by lack of binding in the
substrate or by failure to transfer query role to held entities.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import rebinding_analysis as rb

TRAIN_E = rb.TRAIN_E
HELD_E = rb.HELD_E
ALL_E = rb.ALL_E
N_ATTR = rb.N_ATTR
N_CTX = rb.N_CTX


def mk_group_probes(seed, group, n_per_query=100):
    rng = np.random.default_rng(seed)
    probes=[]
    if group == 'train_query_alltrain':
        query_entities = TRAIN_E
        for q in query_entities:
            for _ in range(n_per_query):
                others = rng.choice([e for e in TRAIN_E if e != q], N_CTX-1, replace=False).tolist()
                probes.extend(make_one(rng, q, others))
    elif group == 'held_query_traindecoys':
        for q in HELD_E:
            for _ in range(n_per_query):
                others = rng.choice(TRAIN_E, N_CTX-1, replace=False).tolist()
                probes.extend(make_one(rng, q, others))
    elif group == 'train_query_helddecoy':
        for q in TRAIN_E:
            for _ in range(max(1, n_per_query//2)):
                one_held = int(rng.choice(HELD_E))
                train_others = rng.choice([e for e in TRAIN_E if e != q], N_CTX-2, replace=False).tolist()
                others = [one_held] + train_others
                probes.extend(make_one(rng, q, others))
    elif group == 'held_query_heldpartner':
        for q in HELD_E:
            partner = [h for h in HELD_E if h != q][0]
            for _ in range(n_per_query):
                train_others = rng.choice(TRAIN_E, N_CTX-2, replace=False).tolist()
                others = [partner] + train_others
                probes.extend(make_one(rng, q, others, force_swap_entity=partner))
    else:
        raise ValueError(group)
    return probes


def make_one(rng, q, others, force_swap_entity=None):
    old = int(rng.integers(N_ATTR))
    new = int(rng.integers(N_ATTR-1))
    if new >= old: new += 1
    ce = list(rng.permutation(list(others)+[q]))
    qi = ce.index(q)
    if force_swap_entity is not None and force_swap_entity in ce:
        si = ce.index(force_swap_entity)
    else:
        si = int(rng.choice([i for i in range(N_CTX) if i != qi]))
    choices = [x for x in range(N_ATTR) if x not in (old,new)]
    ca=[]
    for i,e in enumerate(ce):
        if i == qi: ca.append(old)
        elif i == si: ca.append(new)
        else: ca.append(int(rng.choice(choices)))
    ca_sw=list(ca); ca_sw[qi], ca_sw[si] = ca[si], ca[qi]
    s_t = rb.mk_seq(ce, ca, qi, rb.PAD, rb.PAD)
    s_s = rb.mk_seq(ce, ca_sw, qi, rb.PAD, rb.PAD)
    return [{
        's_t':s_t, 's_s':s_s,
        'old_attr':old, 'new_attr':new,
        'old_rwt':rb.RWT(old), 'new_rwt':rb.RWT(new),
        'old_src':rb.SRC(old), 'new_src':rb.SRC(new),
        'q':q, 'ce':ce, 'qi':qi, 'swap_i':si, 'group':None,
    }]


def compact(v): return {'mean':round(float(v['mean']),5),'std':round(float(v['std']),5)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--research', default='experiments/archive/functional_learning/data/source_trigger/results.json')
    ap.add_argument('--out', default='experiments/archive/functional_learning/data/existing_checkpoint_panel/results.json')
    ap.add_argument('--n-per-query', type=int, default=100)
    args=ap.parse_args()
    with open(args.research) as f: base=json.load(f)
    seeds=base['config']['seeds']; cfg=base['config']
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    groups=['train_query_alltrain','held_query_traindecoys','train_query_helddecoy','held_query_heldpartner']
    out={'status':'EXISTING_CHECKPOINT_PANEL_DONE','input':args.research,'device':dev,'config':{'n_per_query':args.n_per_query},'results':{},'aggregate':{}}
    for cue in ['typed','uninformative']:
        cue_tok = rb.REWRITE_CUE if cue=='typed' else rb.CONST_CUE
        ck='cue_'+cue; out['results'][ck]={}; out['aggregate'][ck]={}
        for seed in seeds:
            out['results'][ck][f'seed{seed}']={}
            probe_sets={g:mk_group_probes(seed*5000+13, g, args.n_per_query) for g in groups}
            for arm in ['ident_full','ident_blocked','unpaired_src']:
                mpath=Path(args.research).parent/f'models_{cue}_s{seed}'/f'{arm}.pt'
                model=rb.CLM(rb.VOCAB,cfg['d'],cfg['nh'],cfg['nl'],rb.SL).to(dev)
                model.load_state_dict(torch.load(mpath,map_location=dev))
                out['results'][ck][f'seed{seed}'][arm]={}
                for g,probes in probe_sets.items():
                    res=rb.eval_rebinding(model, probes, cue_tok, dev)
                    out['results'][ck][f'seed{seed}'][arm][g]={k:compact(v) for k,v in res.items()}
                del model
                if dev=='cuda': torch.cuda.empty_cache()
        # aggregate across seeds
        key_metrics=['T_logit_old_minus_new','S_logit_new_minus_old','binding_margin_sum','rebinding_pair_success','T_old_nll','S_new_nll','S_new_minus_S_old_nll','T_old_minus_T_new_nll','T_top_old','S_top_new']
        for arm in ['ident_full','ident_blocked','unpaired_src']:
            out['aggregate'][ck][arm]={}
            for g in groups:
                out['aggregate'][ck][arm][g]={}
                for km in key_metrics:
                    vals=[out['results'][ck][f'seed{s}'][arm][g][km]['mean'] for s in seeds]
                    out['aggregate'][ck][arm][g][km]={'mean':round(float(np.mean(vals)),5),'std_across_seeds':round(float(np.std(vals)),5),'seed_means':[round(float(v),5) for v in vals]}
        # contrasts ident_full vs others
        out['aggregate'][ck]['contrasts']={}
        for comp in ['ident_blocked','unpaired_src']:
            out['aggregate'][ck]['contrasts'][f'ident_full_minus_{comp}']={}
            for g in groups:
                out['aggregate'][ck]['contrasts'][f'ident_full_minus_{comp}'][g]={}
                for km in ['S_logit_new_minus_old','binding_margin_sum','rebinding_pair_success','S_new_nll']:
                    vals=[out['results'][ck][f'seed{s}']['ident_full'][g][km]['mean'] - out['results'][ck][f'seed{s}'][comp][g][km]['mean'] for s in seeds]
                    out['aggregate'][ck]['contrasts'][f'ident_full_minus_{comp}'][g][km]={'mean':round(float(np.mean(vals)),5),'std_across_seeds':round(float(np.std(vals)),5),'seed_vals':[round(float(v),5) for v in vals]}
    op=Path(args.out); op.parent.mkdir(parents=True,exist_ok=True)
    with open(op,'w') as f: json.dump(out,f,indent=2)
    print(json.dumps({'status':out['status'],'out':str(op)},indent=2))
    for ck in ['cue_typed','cue_uninformative']:
        print('\n'+ck)
        for arm in ['ident_full','ident_blocked','unpaired_src']:
            print(' ',arm)
            for g in groups:
                a=out['aggregate'][ck][arm][g]
                print(f"   {g:25s} Tm {a['T_logit_old_minus_new']['mean']:+.3f} Sm {a['S_logit_new_minus_old']['mean']:+.3f} sum {a['binding_margin_sum']['mean']:+.3f} pair {a['rebinding_pair_success']['mean']:.3f} SnewNLL {a['S_new_nll']['mean']:.2f}")

if __name__=='__main__': main()
