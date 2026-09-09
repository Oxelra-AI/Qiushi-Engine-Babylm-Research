#!/usr/bin/env python3
"""Validate research SuperGLUE outputs before scientific interpretation.

Checks:
- each seed/arm has all 7 task prediction files
- prediction files exist and are arm-specific paths
- prediction hash / predicted-label distribution by task and arm
- comparison to validation label majority accuracy
- whether arms collapse to identical predictions
"""
from __future__ import annotations
import hashlib, json, pathlib, collections

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT/'repos/babylm-eval/strict'
DATA = STRICT/'evaluation_data/full_eval/glue_filtered'
FILES = {
    '42': ROOT/'data/superglue_seed42_allarms.json',
    '43': ROOT/'data/superglue_seed43_allarms.json',
}
OUT = ROOT/'data/superglue_output_validation.json'
TASKS=['boolq','multirc','rte','wsc','mrpc','qqp','mnli']
ARMS=['A_official_wwm','B_structured_wwm','C_official_amlm','D_structured_amlm']

def root_path(s: str) -> pathlib.Path:
    p=pathlib.Path(s)
    if p.is_absolute(): return p
    return ROOT.parent.parent / p if False else pathlib.Path(s)

def read_labels(task: str):
    labels=[]
    for line in (DATA/f'{task}.valid.jsonl').read_text().splitlines():
        if line.strip(): labels.append(int(json.loads(line)['label']))
    return labels

def read_preds(path: pathlib.Path, task: str):
    p=path
    if not p.exists():
        # p is usually user-root-relative; resolve from current root by leaving as-is.
        raise FileNotFoundError(str(p))
    obj=json.loads(p.read_text())
    preds=[int(r['pred']) for r in obj[task]['predictions']]
    return preds

def digest(preds):
    h=hashlib.sha256(','.join(map(str,preds)).encode()).hexdigest()[:16]
    return h

def majority_acc(labels):
    c=collections.Counter(labels)
    n=len(labels); return max(c.values())/n*100.0, dict(c)

def main():
    report={'status':'SUPERGLUE_OUTPUT_VALIDATION','seeds':{},'label_majorities':{}}
    for t in TASKS:
        labels=read_labels(t)
        acc,dist=majority_acc(labels)
        report['label_majorities'][t]={'num_examples':len(labels),'label_counts':dist,'majority_accuracy':acc}
    for seed,path in FILES.items():
        payload=json.loads(path.read_text())
        seed_rep={'input':str(path),'arms':{},'identical_groups_by_task':{}}
        for arm in ARMS:
            rec=payload['models'][arm]
            tasks={r['task']:r for r in rec['tasks']}
            arm_rep={'superglue':rec['superglue'],'tasks':{}}
            for t in TASKS:
                row=tasks[t]
                pp=pathlib.Path(row['predictions'])
                preds=read_preds(pp,t)
                labels=read_labels(t)
                correct=sum(int(a==b) for a,b in zip(preds,labels))
                arm_rep['tasks'][t]={
                    'prediction_path':str(pp),
                    'exists':pp.exists(),
                    'json_accuracy':row['accuracy'],
                    'recomputed_accuracy':correct/len(labels)*100.0,
                    'pred_counts':dict(collections.Counter(preds)),
                    'hash':digest(preds),
                    'matches_json_accuracy':abs(row['accuracy']-correct/len(labels)*100.0)<1e-9,
                }
            seed_rep['arms'][arm]=arm_rep
        for t in TASKS:
            groups=collections.defaultdict(list)
            for arm in ARMS:
                groups[seed_rep['arms'][arm]['tasks'][t]['hash']].append(arm)
            seed_rep['identical_groups_by_task'][t]=dict(groups)
        report['seeds'][seed]=seed_rep
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    # compact stdout
    compact={'out':str(OUT),'majority':report['label_majorities'],'identical_groups':{s:report['seeds'][s]['identical_groups_by_task'] for s in report['seeds']}}
    print(json.dumps(compact,indent=2))
if __name__=='__main__': main()
