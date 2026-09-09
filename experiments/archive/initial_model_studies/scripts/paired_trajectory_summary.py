#!/usr/bin/env python3
"""Compute paired WWM-vs-AMLM trajectory deltas after research trajectories finish."""
from __future__ import annotations
import json, pathlib, statistics
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
IN=ROOT/'data/existing_runs_exposure_trajectory.json'
OUT=ROOT/'data/paired_wwm_amlm_trajectory_summary.json'
NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/paired_wwm_amlm_trajectory_summary.md')
CKPTS=['chck_5M','chck_10M','chck_20M','chck_40M','chck_60M','chck_80M','chck_100M']
KEYS=['blimp','supplement','entity_tracking','ewok','comps','GlobalPIQA_mean','Reading_mean','NLP_mean_no_superglue_aoa']
PAIRS={'seed42':('amlm_seed42','wwm_seed42'),'seed43':('amlm_seed43','wwm_seed43')}

def fmt(x): return f'{x:+.3f}'

def main():
    p=json.loads(IN.read_text())
    runs=p.get('runs',{})
    missing=[r for pair in PAIRS.values() for r in pair if r not in runs]
    if missing: raise RuntimeError(f'missing runs in trajectory: {missing}')
    deltas={}
    for seed,(a,b) in PAIRS.items():
        deltas[seed]={}
        for c in CKPTS:
            if c not in runs[a] or c not in runs[b]: continue
            sa=runs[a][c]['scores']; sb=runs[b][c]['scores']
            deltas[seed][c]={k:sa[k]-sb[k] for k in KEYS}
    mean={}
    for c in CKPTS:
        mean[c]={}
        for k in KEYS:
            vals=[deltas[s][c][k] for s in deltas if c in deltas[s]]
            if vals:
                mean[c][k]={'n':len(vals),'mean':statistics.mean(vals),'values':vals}
                if len(vals)>1: mean[c][k]['stdev']=statistics.stdev(vals)
    payload={'status':'PAIRED_WWM_AMLM_TRAJECTORY','input':str(IN),'pairs':PAIRS,'deltas':deltas,'mean_deltas':mean,'interpretation_hint':'Focus on whether AMLM-WWM Entity/EWoK/GlobalPIQA advantages persist with exposure in both seeds, not on a single checkpoint.'}
    OUT.write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# research paired WWM-AMLM exposure trajectory','',f'Input: `{IN}`','','## Per-seed AMLM - WWM deltas','']
    for seed in ['seed42','seed43']:
        lines += [f'### {seed}', '', '| ckpt | '+' | '.join(KEYS)+' |', '|---|'+'|'.join(['---:']*len(KEYS))+'|']
        for c in CKPTS:
            if c in deltas.get(seed,{}): lines.append('| '+c+' | '+' | '.join(fmt(deltas[seed][c][k]) for k in KEYS)+' |')
        lines.append('')
    lines += ['## Mean deltas across available paired seeds','', '| ckpt | '+' | '.join(KEYS)+' |', '|---|'+'|'.join(['---:']*len(KEYS))+'|']
    for c in CKPTS:
        if c in mean: lines.append('| '+c+' | '+' | '.join(fmt(mean[c][k]['mean']) for k in KEYS if k in mean[c])+' |')
    NOTE.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'out':str(OUT),'note':str(NOTE)},indent=2))
if __name__=='__main__': main()
