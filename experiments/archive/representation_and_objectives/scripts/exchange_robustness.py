#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import numpy as np

ROOT=Path('.').resolve()
DEFAULT=ROOT/'experiments/archive/representation_and_objectives/data/counterfactual_exchange_clean_subset_full/clean_subset_scores.jsonl'
OUT=ROOT/'experiments/archive/representation_and_objectives/data/counterfactual_exchange_robustness'
NOTE=ROOT/'research/notes/representation_and_objectives/counterfactual_exchange_robustness.md'

def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
def q(vals):
    xs=np.asarray([float(v) for v in vals if math.isfinite(float(v))],dtype=float)
    if xs.size==0: return {'n':0}
    return {'n':int(xs.size),'mean':float(xs.mean()),'std':float(xs.std()),'stderr':float(xs.std()/math.sqrt(xs.size)),'median':float(np.quantile(xs,.5)),'p05':float(np.quantile(xs,.05)),'p25':float(np.quantile(xs,.25)),'p75':float(np.quantile(xs,.75)),'p95':float(np.quantile(xs,.95)),'min':float(xs.min()),'max':float(xs.max()),'success_gt0':float((xs>0).mean())}
def load(path):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows
def paired(rows,fam,metric='true_m'):
    by=defaultdict(dict)
    meta={}
    for r in rows:
        if fam and r.get('clean_family')!=fam: continue
        by[r['case_id']][r['arm']]=float(r[metric])
        meta[r['case_id']]=r
    out=[]
    for cid,d in by.items():
        if {'compact','rowblock','interleaved'}<=set(d):
            m=meta[cid]
            out.append({'case_id':cid,'sentence':m.get('sentence',''),'family':m.get('clean_family'),'rb_c':d['rowblock']-d['compact'],'rb_i':d['rowblock']-d['interleaved'],'int_c':d['interleaved']-d['compact'],'compact':d['compact'],'rowblock':d['rowblock'],'interleaved':d['interleaved']})
    return out
def cluster_boot(vals_by_cluster, nboot=4000, seed=130):
    rng=random.Random(seed); keys=list(vals_by_cluster)
    if len(keys)<2: return {'n_clusters':len(keys)}
    boots=[]
    for _ in range(nboot):
        sample=[rng.choice(keys) for _ in keys]
        vals=[]
        for k in sample: vals.extend(vals_by_cluster[k])
        boots.append(float(np.mean(vals)))
    arr=np.asarray(boots)
    return {'n_clusters':len(keys),'boot_mean':float(arr.mean()),'ci05':float(np.quantile(arr,.05)),'ci50':float(np.quantile(arr,.5)),'ci95':float(np.quantile(arr,.95)),'p_gt0':float((arr>0).mean())}
def influence(vals):
    xs=np.asarray(vals,dtype=float)
    if xs.size==0: return {}
    order=np.argsort(xs)
    trim05=xs[order[int(0.05*len(xs)): int(0.95*len(xs))]] if len(xs)>=20 else xs
    top_pos=np.sort(xs)[-max(1,int(.05*len(xs))):]
    top_neg=np.sort(xs)[:max(1,int(.05*len(xs)))]
    return {'trim05_95_mean':float(trim05.mean()),'top5pct_pos_mean':float(top_pos.mean()),'top5pct_neg_mean':float(top_neg.mean()),'top5pct_pos_sum_frac':float(top_pos.sum()/xs.sum()) if abs(xs.sum())>1e-9 else None,'top5pct_abs_frac':float(np.sort(np.abs(xs))[-max(1,int(.05*len(xs))):].sum()/np.abs(xs).sum()) if np.abs(xs).sum()>0 else None}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scores',type=Path,default=DEFAULT); ap.add_argument('--out-dir',type=Path,default=OUT); ap.add_argument('--note',type=Path,default=NOTE); args=ap.parse_args()
    rows=load(args.scores); args.out_dir.mkdir(parents=True, exist_ok=True)
    fams=sorted({r.get('clean_family') for r in rows})
    summary={'status':'COUNTERFACTUAL_EXCHANGE_ROBUSTNESS','created_utc':now(),'scores':str(args.scores),'families':{}}
    for fam in fams:
        ps=paired(rows,fam)
        fd={'n_cases':len(ps),'metrics':{}}
        for key in ['rb_c','rb_i','int_c']:
            vals=[p[key] for p in ps]
            clusters=defaultdict(list)
            for p in ps: clusters[p['sentence']].append(p[key])
            fd['metrics'][key]={'stats':q(vals),'cluster_bootstrap':cluster_boot(clusters),'influence':influence(vals)}
        summary['families'][fam]=fd
    ps_all=paired(rows,None)
    summary['all']={'n_cases':len(ps_all),'metrics':{}}
    for key in ['rb_c','rb_i','int_c']:
        vals=[p[key] for p in ps_all]; clusters=defaultdict(list)
        for p in ps_all: clusters[p['sentence']].append(p[key])
        summary['all']['metrics'][key]={'stats':q(vals),'cluster_bootstrap':cluster_boot(clusters),'influence':influence(vals)}
    (args.out_dir/'exchange_robustness_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # examples of largest spatial positive/negative
    for fam in ['clean_spatial_vertical','clean_temporal_order']:
        ps=paired(rows,fam)
        with (args.out_dir/f'{fam}_top_rowblock_minus_interleaved.jsonl').open('w',encoding='utf-8') as f:
            for p in sorted(ps,key=lambda x:x['rb_i'],reverse=True)[:60]: f.write(json.dumps(p,ensure_ascii=False)+'\n')
        with (args.out_dir/f'{fam}_bottom_rowblock_minus_interleaved.jsonl').open('w',encoding='utf-8') as f:
            for p in sorted(ps,key=lambda x:x['rb_i'])[:60]: f.write(json.dumps(p,ensure_ascii=False)+'\n')
    note=['# research — counterfactual exchange robustness\n',f'Created: {summary["created_utc"]}\n\n']
    note.append('| family | n | rb-compact mean | rb-compact sign | rb-compact cluster p>0 | rb-interleaved mean | rb-interleaved sign | rb-interleaved cluster p>0 | trim rb-interleaved |\n')
    note.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for fam,fd in summary['families'].items():
        mc=fd['metrics']['rb_c']; mi=fd['metrics']['rb_i']
        note.append(f"| {fam} | {fd['n_cases']} | {mc['stats'].get('mean')} | {mc['stats'].get('success_gt0')} | {mc['cluster_bootstrap'].get('p_gt0')} | {mi['stats'].get('mean')} | {mi['stats'].get('success_gt0')} | {mi['cluster_bootstrap'].get('p_gt0')} | {mi['influence'].get('trim05_95_mean')} |\n")
    note.append(f"\nSummary JSON: `{args.out_dir/'exchange_robustness_summary.json'}`\n")
    args.note.parent.mkdir(parents=True,exist_ok=True); args.note.write_text(''.join(note),encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary':str(args.out_dir/'exchange_robustness_summary.json'),'note':str(args.note)},indent=2),flush=True)
if __name__=='__main__': main()
