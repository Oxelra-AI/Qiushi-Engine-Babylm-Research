#!/usr/bin/env python3
"""Compare wwm_seed43 chck_80M vs chck_100M as single official endpoints.

Combines existing 7-column zero-shot/Reading trajectory scores with newly run
SuperGLUE scores. AoA is not recomputed here; the script reports endpoint NLP
average and an AoA-using estimate only if an AoA value is supplied.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, pathlib

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
TRAJ = ROOT/'data/existing_runs_exposure_trajectory.json'
PROTECTED = ROOT/'data/debertav2_b256_true_9of9_coordinate.json'
DATA = ROOT/'repos/babylm-eval/strict/evaluation_data/full_eval/glue_filtered'
OUT = ROOT/'data/wwm_seed43_endpoint_comparison.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/wwm_seed43_endpoint_comparison.md')
COLS7 = ['blimp','supplement','entity_tracking','ewok','comps','GlobalPIQA_mean','Reading_mean']
TASKS = ['boolq','multirc','rte','wsc','mrpc','qqp','mnli']

def read_json(p):
    return json.loads(pathlib.Path(p).read_text())

def labels_for(task):
    labels=[]
    for line in (DATA/f'{task}.valid.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip(): labels.append(int(json.loads(line)['label']))
    return labels

def pred_stats(row):
    pp = pathlib.Path(row['predictions'])
    obj = json.loads(pp.read_text(encoding='utf-8'))
    task = row['task']
    preds = [int(r['pred']) for r in obj[task]['predictions']]
    labels = labels_for(task)
    counts = dict(collections.Counter(preds))
    lab_counts = dict(collections.Counter(labels))
    maj = max(lab_counts.values()) / len(labels) * 100.0
    h = hashlib.sha256(','.join(map(str,preds)).encode()).hexdigest()[:16]
    return {'pred_counts':counts,'label_counts':lab_counts,'majority_accuracy':maj,'hash':h,'all_one_label':len(counts)==1}

def endpoint_payload(name, ckpt, sg_path, aoa=None):
    traj=read_json(TRAJ)
    scores=dict(traj['runs']['wwm_seed43'][ckpt]['scores'])
    sg=read_json(sg_path)
    superglue=sg['superglue_mean']
    endpoint={k:scores[k] for k in COLS7}
    endpoint['superglue']=superglue
    endpoint['endpoint_8col_mean']=sum(endpoint.values())/8
    if aoa is not None:
        endpoint['aoa']=aoa
        endpoint['overall_9col_mean']=(endpoint['endpoint_8col_mean']*8 + aoa)/9
    task_rows=[]
    for row in sg['tasks']:
        r=dict(row); r.update(pred_stats(row)); task_rows.append(r)
    return {'name':name,'ckpt':ckpt,'superglue_path':str(sg_path),'scores':endpoint,'superglue_tasks':task_rows}

def protected_scores():
    p=read_json(PROTECTED)
    # robustly search known fields
    if 'scores_official_columns' in p:
        s=p['scores_official_columns']
    elif 'scores' in p:
        s=p['scores']
    elif 'protected_true_9of9' in p:
        s=p['protected_true_9of9']['scores']
    else:
        s=p
    def get(*names):
        for n in names:
            if n in s: return s[n]
        return None
    ep={
        'blimp':get('BLiMP','blimp'),
        'supplement':get('BLiMP Supplement','supplement'),
        'ewok':get('EWoK','ewok'),
        'entity_tracking':get('Entity Tracking','entity_tracking'),
        'comps':get('COMPS','comps'),
        'superglue':get('(Super)GLUE','superglue'),
        'GlobalPIQA_mean':get('GlobalPIQA','GlobalPIQA_mean'),
        'Reading_mean':get('Reading','Reading_mean'),
        'aoa':get('AoA','aoa'),
    }
    vals8=[ep[k] for k in ['blimp','supplement','ewok','entity_tracking','comps','superglue','GlobalPIQA_mean','Reading_mean'] if ep[k] is not None]
    ep['endpoint_8col_mean']=sum(vals8)/len(vals8)
    if ep.get('aoa') is not None and len(vals8)==8:
        ep['overall_9col_mean']=(sum(vals8)+ep['aoa'])/9
    return ep

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sg80', required=True)
    ap.add_argument('--sg100', required=True)
    ap.add_argument('--aoa80', type=float, default=None)
    ap.add_argument('--aoa100', type=float, default=None)
    args=ap.parse_args()
    ep80=endpoint_payload('wwm_seed43_chck80M','chck_80M',pathlib.Path(args.sg80),args.aoa80)
    ep100=endpoint_payload('wwm_seed43_chck100M','chck_100M',pathlib.Path(args.sg100),args.aoa100)
    prot=protected_scores()
    keys=['blimp','supplement','entity_tracking','ewok','comps','GlobalPIQA_mean','Reading_mean','superglue','endpoint_8col_mean']
    delta80_100={k:ep80['scores'][k]-ep100['scores'][k] for k in keys}
    delta80_prot={k:ep80['scores'][k]-prot[k] for k in keys if k in prot and prot[k] is not None}
    payload={'status':'ENDPOINT_COMPARISON','endpoint80':ep80,'endpoint100':ep100,'protected_reference':prot,'delta_80_minus_100':delta80_100,'delta_80_minus_protected':delta80_prot}
    OUT.write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# research wwm_seed43 endpoint comparison','','Evidence JSON: `data/wwm_seed43_endpoint_comparison.json`','','## Endpoint scores','', '| endpoint | '+' | '.join(keys)+' |','|---|'+'|'.join(['---:']*len(keys))+'|']
    for ep in [ep80, ep100]:
        lines.append('| '+ep['name']+' | '+' | '.join(f"{ep['scores'][k]:.3f}" for k in keys)+' |')
    lines.append('| protected_seed42_100M | '+' | '.join(f"{prot[k]:.3f}" if k in prot and prot[k] is not None else '' for k in keys)+' |')
    lines += ['','## Delta: 80M - 100M','', '| column | delta |','|---|---:|']
    for k in keys: lines.append(f'| {k} | {delta80_100[k]:+.3f} |')
    lines += ['','## SuperGLUE prediction concentration','']
    for ep in [ep80, ep100]:
        lines += [f"### {ep['name']}", '', '| task | accuracy | majority_acc | pred_counts | all_one_label |','|---|---:|---:|---|---|']
        for r in ep['superglue_tasks']:
            lines.append(f"| {r['task']} | {r['accuracy']:.3f} | {r['majority_accuracy']:.3f} | {r['pred_counts']} | {r['all_one_label']} |")
        lines.append('')
    NOTE.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'out':str(OUT),'note':str(NOTE),'delta_80_minus_100':delta80_100},indent=2))
if __name__=='__main__': main()
