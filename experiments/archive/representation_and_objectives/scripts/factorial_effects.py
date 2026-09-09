#!/usr/bin/env python3
"""Compute factorial orientation effects for research runs.

For response Y(E,R), where E/R are true or flipped sparse orientations:
Delta_E = ([Y(T,T)-Y(F,T)] + [Y(T,F)-Y(F,F)]) / 2
Delta_R = ([Y(T,T)-Y(T,F)] + [Y(F,T)-Y(F,F)]) / 2
Interaction = Y(T,T) - Y(F,T) - Y(T,F) + Y(F,F)
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

QUERIES=["event_role","focal_state","untouched_state"]
EVALS=[
 "atp_trainTrain_trainHyp", "atp_trainTrain_heldHyp",
 "atp_heldHeld_trainHyp", "atp_heldHeld_heldHyp",
 "atp_dirDir_trainHyp", "atp_dirDir_heldHyp",
 "atp_trainEvent_dirState_trainHyp", "atp_dirEvent_trainState_trainHyp",
 "atp_trainEvent_nonState_trainHyp", "atp_nonEvent_trainState_trainHyp",
]

def load(p): return json.loads(Path(p).read_text('utf-8'))

def per_seed_metric(seed_result, ev, q):
    return seed_result['evals'][ev]['contrastive']['by_query'].get(q, float('nan'))

def stat(xs):
    xs=[float(x) for x in xs if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return {'mean':float(np.mean(xs)) if xs else float('nan'), 'std':float(np.std(xs)) if xs else float('nan'), 'n':len(xs), 'vals':xs}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('summary')
    ap.add_argument('--out_dir', default='experiments/archive/representation_and_objectives/data/factorial_effects')
    args=ap.parse_args()
    d=load(args.summary)
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    by_seed={}
    for r in d['per_seed']:
        by_seed.setdefault(r['seed'], {})[r['arm']]=r
    effects={}
    for ev in EVALS:
        effects[ev]={}
        for q in QUERIES:
            de=[]; dr=[]; ix=[]; cells=[]
            for seed, arms in sorted(by_seed.items()):
                needed=['Etrue_Rtrue','Eflip_Rtrue','Etrue_Rflip','Eflip_Rflip']
                if not all(a in arms for a in needed):
                    continue
                tt=per_seed_metric(arms['Etrue_Rtrue'], ev, q)
                ft=per_seed_metric(arms['Eflip_Rtrue'], ev, q)
                tf=per_seed_metric(arms['Etrue_Rflip'], ev, q)
                ff=per_seed_metric(arms['Eflip_Rflip'], ev, q)
                de.append(((tt-ft)+(tf-ff))/2)
                dr.append(((tt-tf)+(ft-ff))/2)
                ix.append(tt-ft-tf+ff)
                cells.append({'seed':seed,'TT':tt,'FT':ft,'TF':tf,'FF':ff})
            effects[ev][q]={'delta_event':stat(de),'delta_rank':stat(dr),'interaction':stat(ix),'cells':cells}
    (out/'factorial_effects.json').write_text(json.dumps({'source':args.summary,'effects':effects}, indent=2)+'\n','utf-8')
    lines=[f"# Factorial effects for {args.summary}\n\n"]
    lines.append("Delta_E and Delta_R are balanced factorial effects over Etrue/Eflip and Rtrue/Rflip arms. They are more appropriate than comparing only against Etrue_Rtrue.\n\n")
    for ev in EVALS:
        lines.append(f"## {ev}\n\n")
        lines.append("| query | Delta_E | Delta_R | interaction |\n|---|---:|---:|---:|\n")
        for q in QUERIES:
            e=effects[ev][q]
            lines.append(f"| {q} | {e['delta_event']['mean']:.3f}±{e['delta_event']['std']:.3f} | {e['delta_rank']['mean']:.3f}±{e['delta_rank']['std']:.3f} | {e['interaction']['mean']:.3f}±{e['interaction']['std']:.3f} |\n")
        lines.append("\n")
    (out/'factorial_effects.md').write_text(''.join(lines),'utf-8')
    print(json.dumps({'status':'factorial_effects_done','md':str(out/'factorial_effects.md'),'json':str(out/'factorial_effects.json')}))
if __name__=='__main__': main()
