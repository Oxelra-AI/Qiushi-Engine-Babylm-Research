#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv,json,pathlib,time
ROOT=_public_path('experiments/archive/frontier_consolidation/scripts/update_prediction_commitment_strict.py')
ROOT = _PUBLIC_ROOT
WS=_public_path('experiments/archive/frontier_consolidation')
commit_path=_public_path('experiments/archive/frontier_consolidation/data/distribution_proximity_prediction/prediction_commitment.json')
strict_dir=_public_path('experiments/archive/frontier_consolidation/data/strict_eval_text_prediction')
commit=json.loads(commit_path.read_text(encoding='utf-8'))
fits=[]
with (_public_path('experiments/archive/frontier_consolidation/data/strict_eval_text_prediction/fit_summary_rows.csv')).open(encoding='utf-8',newline='') as f:
    for r in csv.DictReader(f):
        fits.append({k:(float(v) if k in {'beta','pearson','spearman','rmse','r2_zero','r2_centered','x_min','x_max','x_mean','y_mean'} and v!='' else int(v) if k=='n' else v) for k,v in r.items()})
preds={}
with (_public_path('experiments/archive/frontier_consolidation/data/strict_eval_text_prediction/prediction_rows.csv')).open(encoding='utf-8',newline='') as f:
    for r in csv.DictReader(f):
        preds.setdefault(r['metric'],{})[r['family']]=float(r['pred_points'])
aggs={}
with (_public_path('experiments/archive/frontier_consolidation/data/strict_eval_text_prediction/aggregate_prediction_rows.csv')).open(encoding='utf-8',newline='') as f:
    for r in csv.DictReader(f):
        aggs.setdefault(r['metric'],{})[r['quantity']]=float(r['pred_points'])
commit['strict_eval_text_extraction_robustness']={
    'created_before_register_scores': True,
    'source_summary_md': 'research/documents/frontier_consolidation/data/strict_eval_text_prediction/strict_eval_text_prediction_summary.md',
    'strict_extraction_rule': 'BLiMP/Supplement sentence_good/bad; EWoK Context1/2 and Target1/2 only; COMPS prefix+property candidate strings; Entity input_prefix/options/prefix+option; Reading unique sentence.',
    'fit_summary_rows': fits,
    'family_predictions': preds,
    'aggregate_predictions': aggs,
    'scientific_implication': 'The original positive profile-register prediction survives task-aware extraction: profile exEntity5 +0.7697 and cheap6 +0.8441, while word-JS remains a failed lexical control. Thus the prediction is not driven by broad EWoK/COMPS metadata key heuristics.'
}
commit['last_updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
commit_path.write_text(json.dumps(commit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':'commitment_strict_updated','path':str(commit_path.relative_to(ROOT)),'strict_aggregates':aggs},indent=2))
