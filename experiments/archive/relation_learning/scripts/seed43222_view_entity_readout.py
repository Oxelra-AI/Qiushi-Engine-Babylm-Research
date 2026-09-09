#!/usr/bin/env python3
"""Read seed43222 VIEW Entity predictions by relevant-update group.

This script uses the official Entity predictions produced by
`eval_seed43222_entity.py` for D_V_43222 and compares its absolute
late relevant-update profile to prior VIEW seeds 43022/43122. It does not use
REPEAT/CLEAN seed43222 because those training results are not yet available.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, math, pathlib, re, statistics, time
from collections import defaultdict
from typing import Any

ROOT=_public_path('experiments/archive/relation_learning/scripts/seed43222_view_entity_readout.py')
ROOT = _PUBLIC_ROOT
WS=_public_path('experiments/archive/relation_learning')
OUT=_public_path('experiments/archive/relation_learning/data/seed43222_view_entity_readout')
META=_public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_item_metadata.csv')
EVAL=_public_path('experiments/archive/relation_learning/data/seed43222_entity_eval/per_target')
SUM=_public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_relevant_update_summary.csv')
CKS=['chck_80M','chck_90M','chck_100M']

def rel(p:pathlib.Path)->str:
    try: return str(p.relative_to(ROOT))
    except Exception: return str(p)

def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def norm(s:str)->str:
    return re.sub(r'\s+',' ',str(s).strip().lower()).strip(' .')

def mean(xs):
    xs=[float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(xs) if xs else float('nan')

def read_csv(p):
    with open(p, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f): yield r

def read_json(p): return json.loads(p.read_text(encoding='utf-8'))

def load_meta_by_uid():
    by=defaultdict(list)
    for r in read_csv(META):
        r=dict(r)
        for k in ['item_index','reported_numops','relevant_updates','total_ops','irrelevant_ops','prefix_words','stale_available','stale_is_gold']:
            r[k]=int(r[k])
        by[r['uid']].append(r)
    for uid in by:
        by[uid].sort(key=lambda x:int(x['item_index']))
    return by

def prediction_path(ck):
    payload=read_json(EVAL/f'D_V_43222_{ck}.json')
    pred=payload['tasks']['Entity']['predictions']
    p=ROOT/pred
    if not p.exists(): raise FileNotFoundError(p)
    return p

def groups(r):
    relu=int(r['relevant_updates']); total=int(r['total_ops'])
    out=['ALL',f'rel_updates_{relu}',f'reported_numops_{r["reported_numops"]}',f'total_ops_{total}',f'type_{r["entity_type"]}']
    if relu==0:
        out.append(f'rel0_total_ops_{total}')
    else:
        out.append('rel_ge1')
        if relu>=2: out.append('rel_ge2')
        if relu>=3: out.append('rel_ge3')
    return out

def summarize(rows):
    d=defaultdict(list)
    for r in rows:
        for g in groups(r): d[(r['seed'],r['arm'],r['checkpoint'],g)].append(r)
    out=[]
    for (seed,arm,ck,g),vals in sorted(d.items()):
        wrong=[v for v in vals if not int(v['correct'])]
        stale_cand=[v for v in vals if int(v['stale_available']) and not int(v['stale_is_gold'])]
        wrong_stale=[v for v in stale_cand if not int(v['correct'])]
        out.append({'seed':seed,'arm':arm,'checkpoint':ck,'group':g,'n':len(vals),
            'accuracy_pct':100*sum(int(v['correct']) for v in vals)/len(vals),
            'mean_reported_numops':mean([v['reported_numops'] for v in vals]),
            'mean_relevant_updates':mean([v['relevant_updates'] for v in vals]),
            'mean_total_ops':mean([v['total_ops'] for v in vals]),
            'mean_prefix_words':mean([v['prefix_words'] for v in vals]),
            'stale_available_not_gold_n':len(stale_cand),
            'stale_pick_pct_among_wrong_stale_available_not_gold':100*sum(int(v['pred_is_stale_initial']) for v in wrong_stale)/len(wrong_stale) if wrong_stale else float('nan'),
            'stale_pick_pct_among_all_wrong':100*sum(int(v['pred_is_stale_initial']) for v in wrong)/len(wrong) if wrong else float('nan')})
    return out

def late(summary):
    d=defaultdict(list)
    for r in summary: d[(r['seed'],r['arm'],r['group'])].append(r)
    out=[]
    for (seed,arm,g),vals in sorted(d.items()):
        out.append({'seed':seed,'arm':arm,'group':g,'n_checkpoints':len(vals),'n':int(vals[0]['n']),
            'late_mean_accuracy_pct':mean([v['accuracy_pct'] for v in vals]),
            'late_mean_stale_pick_wrong_pct':mean([v['stale_pick_pct_among_wrong_stale_available_not_gold'] for v in vals]),
            'mean_reported_numops':mean([v['mean_reported_numops'] for v in vals]),
            'mean_relevant_updates':mean([v['mean_relevant_updates'] for v in vals]),
            'mean_total_ops':mean([v['mean_total_ops'] for v in vals]),
            'mean_prefix_words':mean([v['mean_prefix_words'] for v in vals])})
    return out

def write_csv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text('\n', encoding='utf-8'); return
    fields=sorted(set().union(*(r.keys() for r in rows)))
    pref=['seed','arm','checkpoint','group','n','n_checkpoints','accuracy_pct','late_mean_accuracy_pct','late_mean_stale_pick_wrong_pct','mean_relevant_updates','mean_total_ops','mean_prefix_words','uid','item_index','entity_type','reported_numops','relevant_updates','total_ops','correct','pred','gold']
    fields=[x for x in pref if x in fields]+[x for x in fields if x not in pref]
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def prior_view_late():
    d=defaultdict(list)
    for r in read_csv(SUM):
        if r['arm']=='V' and r['group'] in {'ALL','rel_updates_0','rel_updates_1','rel_updates_2','rel_updates_3','rel_updates_4','rel_updates_5','rel_ge2','rel_ge3'}:
            d[(r['seed'],r['group'])].append(float(r['accuracy_pct']))
    return {(seed,g):mean(v) for (seed,g),v in d.items()}

def make_note(late_rows):
    prior=prior_view_late()
    idx={(r['group']):r for r in late_rows if r['seed']=='43222'}
    lines=[]
    lines.append('# research seed43222 VIEW Entity readout')
    lines.append('')
    lines.append('Seed43222 VIEW finished and was evaluated on official Entity only; REPEAT/CLEAN seed43222 were still pending when this readout was produced. Values are late means over 80M/90M/100M.')
    lines.append('')
    lines.append('| group | seed43022 V acc | seed43122 V acc | seed43222 V acc | n |')
    lines.append('|---|---:|---:|---:|---:|')
    for g in ['ALL','rel_updates_0','rel_updates_1','rel_updates_2','rel_updates_3','rel_updates_4','rel_updates_5','rel_ge2','rel_ge3']:
        r=idx.get(g,{})
        lines.append(f"| {g} | {prior.get(('43022',g),float('nan')):.2f} | {prior.get(('43122',g),float('nan')):.2f} | {float(r.get('late_mean_accuracy_pct',float('nan'))):.2f} | {r.get('n','')} |")
    lines.append('')
    lines.append('## Scientific reading')
    lines.append('')
    lines.append('The absolute seed43222 VIEW Entity profile is lower overall than the two earlier VIEW seeds, so VIEW alone cannot confirm the three-arm crossover. The important question remains the within-seed contrast against pending seed43222 REPEAT and CLEAN. If REPEAT/CLEAN also move down together, the contrast may still replicate; if VIEW alone lost the multi-update advantage, the current mechanism must be revised.')
    (_public_path('research/notes/relation_learning/seed43222_view_entity_readout.md')).write_text('\n'.join(lines)+'\n', encoding='utf-8')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta=load_meta_by_uid(); rows=[]
    path_map={}
    for ck in CKS:
        p=prediction_path(ck); path_map[ck]=rel(p); obj=read_json(p)
        for uid, items in sorted(meta.items()):
            preds=obj[uid]['predictions']
            if len(preds)!=len(items): raise RuntimeError((uid,len(preds),len(items),rel(p)))
            for item,predrec in zip(items,preds):
                pred=str(predrec.get('pred',''))
                correct=int(norm(pred)==norm(item['gold']))
                pred_stale=int(bool(item.get('stale_initial','')) and norm(pred)==norm(item.get('stale_initial','')))
                rows.append({**item,'seed':'43222','arm':'V','checkpoint':ck,'pred':pred,'correct':correct,'pred_is_stale_initial':pred_stale})
    summary=summarize(rows); late_rows=late(summary)
    write_csv(_public_path('experiments/archive/relation_learning/data/seed43222_view_entity_readout/seed43222_view_entity_item_rows.csv'), rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/seed43222_view_entity_readout/seed43222_view_entity_summary.csv'), summary)
    write_csv(_public_path('experiments/archive/relation_learning/data/seed43222_view_entity_readout/seed43222_view_entity_late_summary.csv'), late_rows)
    (_public_path('experiments/archive/relation_learning/data/seed43222_view_entity_readout/seed43222_view_prediction_paths.json')).write_text(json.dumps(path_map,indent=2)+'\n', encoding='utf-8')
    make_note(late_rows)
    res={'status':'SEED43222_VIEW_ENTITY_READOUT_DONE','finished_utc':now(),'prediction_paths':path_map,'files':{'item_rows':rel(_public_path('experiments/archive/relation_learning/data/seed43222_view_entity_readout/seed43222_view_entity_item_rows.csv')),'summary':rel(_public_path('experiments/archive/relation_learning/data/seed43222_view_entity_readout/seed43222_view_entity_summary.csv')),'late_summary':rel(_public_path('experiments/archive/relation_learning/data/seed43222_view_entity_readout/seed43222_view_entity_late_summary.csv')),'note':rel(_public_path('research/notes/relation_learning/seed43222_view_entity_readout.md'))}}
    (_public_path('experiments/archive/relation_learning/data/seed43222_view_entity_readout/seed43222_view_entity_readout_result.json')).write_text(json.dumps(res,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(res,indent=2))
    for r in late_rows:
        if r['group'] in {'ALL','rel_updates_0','rel_updates_2','rel_updates_3','rel_updates_4','rel_updates_5','rel_ge2','rel_ge3'}:
            print(f"{r['group']} acc={r['late_mean_accuracy_pct']:.2f} n={r['n']}")
if __name__=='__main__': main()
