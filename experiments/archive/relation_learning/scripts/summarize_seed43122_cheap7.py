#!/usr/bin/env python3
"""Summarize research seed43122 base/dose cheap7 results and compare with seed43022 predictions."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, statistics, time
ROOT0 = _public_path('experiments/archive/relation_learning/scripts/summarize_seed43122_cheap7.py')
USER_ROOT = _PUBLIC_ROOT
OUT = USER_ROOT / 'experiments/archive/relation_learning/data/seed43122_cheap7'
TARGETS = {
    'base0': OUT / 'base43122/base43122_cheap7_summary.json',
    'dose21': OUT / 'dose21_43122/dose21_43122_cheap7_summary.json',
    'dose25': OUT / 'dose25_43122/dose25_43122_cheap7_summary.json',
}
CHEAP = ['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA','Reading']
rows=[]
for dose,path in TARGETS.items():
    rec=json.loads(path.read_text(encoding='utf-8'))['record']
    scores=rec.get('scores') or {}
    row={'dose':dose,'target':rec.get('target'),'returncode':rec.get('returncode'),'cheap7':rec.get('cheap7'),'summary':str(path.relative_to(USER_ROOT)),'per_target':rec.get('per_target')}
    for c in CHEAP:
        row[c]=scores.get(c)
    rows.append(row)
idx={r['dose']:r for r in rows}
contrasts=[]
for a,b in [('dose21','base0'),('dose25','base0'),('dose25','dose21')]:
    cr={'contrast':f'{a}-minus-{b}'}
    cr['cheap7_delta']=idx[a]['cheap7']-idx[b]['cheap7'] if idx[a].get('cheap7') is not None and idx[b].get('cheap7') is not None else None
    for c in CHEAP:
        va,vb=idx[a].get(c),idx[b].get(c)
        cr[f'{c}_delta']=va-vb if va is not None and vb is not None else None
    contrasts.append(cr)
# monotone seed43022 predictions: BLiMP down and EWoK up base->dose21->dose25.
sequence={c:[idx[d][c] for d in ['base0','dose21','dose25']] for c in CHEAP if all(idx[d].get(c) is not None for d in ['base0','dose21','dose25'])}
prediction_tests={
    'BLiMP_monotone_down': bool(sequence.get('BLiMP') and sequence['BLiMP'][0] > sequence['BLiMP'][1] > sequence['BLiMP'][2]),
    'EWoK_monotone_up': bool(sequence.get('EWoK') and sequence['EWoK'][0] < sequence['EWoK'][1] < sequence['EWoK'][2]),
    'Reading_monotone_down': bool(sequence.get('Reading') and sequence['Reading'][0] > sequence['Reading'][1] > sequence['Reading'][2]),
}
# Mean of the non-GlobalPIQA columns is useful because GP was the noisiest seed43022 scalar mover.
for r in rows:
    vals=[r[c] for c in ['BLiMP','Supplement','EWoK','Entity','COMPS','Reading'] if r.get(c) is not None]
    r['cheap6_no_GlobalPIQA']=float(statistics.mean(vals)) if vals else None
for cr in contrasts:
    a,b=cr['contrast'].split('-minus-')
    cr['cheap6_no_GlobalPIQA_delta']=idx[a]['cheap6_no_GlobalPIQA']-idx[b]['cheap6_no_GlobalPIQA']
summary={'status':'SEED43122_CHEAP7_SUMMARY','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),'rows':rows,'contrasts':contrasts,'sequence':sequence,'prediction_tests':prediction_tests,'seed43022_prediction_note':'Before scoring seed43122, research registered the seed43022 practical prediction: BLiMP should decline monotonically while EWoK should rise monotonically across base0->dose21->dose25; scalar cheap7 is expected to be vulnerable to GlobalPIQA noise.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
lines=['# research seed43122 cheap7 summary','','| dose | cheap7 | cheap6(no GP) | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append('| {dose} | {cheap7:.4f} | {cheap6_no_GlobalPIQA:.4f} | {BLiMP:.3f} | {Supplement:.3f} | {EWoK:.3f} | {Entity:.3f} | {COMPS:.3f} | {GlobalPIQA:.3f} | {Reading:.3f} |'.format(**r))
lines += ['','## Contrasts','']
for c in contrasts:
    lines.append(f"- {c['contrast']}: cheap7 {c['cheap7_delta']:+.4f}, cheap6(no GP) {c['cheap6_no_GlobalPIQA_delta']:+.4f}, BLiMP {c['BLiMP_delta']:+.3f}, EWoK {c['EWoK_delta']:+.3f}, Reading {c['Reading_delta']:+.3f}, GlobalPIQA {c['GlobalPIQA_delta']:+.3f}")
lines += ['','## Registered seed43022 pattern tested on seed43122','',json.dumps(prediction_tests, indent=2)]
((_PUBLIC_ROOT / 'research/documents/relation_learning/data/seed43122_cheap7/summary.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':summary['status'],'summary':str((OUT/'summary.json').relative_to(USER_ROOT)),'md':str(((_PUBLIC_ROOT / 'research/documents/relation_learning/data/seed43122_cheap7/summary.md')).relative_to(USER_ROOT)),'prediction_tests':prediction_tests},indent=2),flush=True)
