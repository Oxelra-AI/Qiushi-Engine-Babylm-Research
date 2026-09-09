#!/usr/bin/env python3
"""Summarize research seed43022 base/dose cheap7 results."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, statistics, time
ROOT = _public_path('experiments/archive/relation_learning/scripts/summarize_seed43022_cheap7.py')
USER_ROOT = _PUBLIC_ROOT
OUT = USER_ROOT / 'experiments/archive/relation_learning/data/seed43022_cheap7'
TARGETS = {
    'base0': OUT / 'base43022/base43022_cheap7_summary.json',
    'dose21': OUT / 'dose21_43022/dose21_43022_cheap7_summary.json',
    'dose25': OUT / 'dose25_43022/dose25_43022_cheap7_summary.json',
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
    if a in idx and b in idx:
        cr={'contrast':f'{a}-minus-{b}','cheap7_delta':idx[a]['cheap7']-idx[b]['cheap7'] if idx[a].get('cheap7') is not None and idx[b].get('cheap7') is not None else None}
        for c in CHEAP:
            va,vb=idx[a].get(c),idx[b].get(c)
            cr[f'{c}_delta']=va-vb if va is not None and vb is not None else None
        contrasts.append(cr)
summary={'status':'SEED43022_CHEAP7_SUMMARY','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),'rows':rows,'contrasts':contrasts}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
lines=['# research seed43022 cheap7 summary','']
lines.append('| dose | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for r in rows:
    lines.append('| {dose} | {cheap7:.4f} | {BLiMP:.3f} | {Supplement:.3f} | {EWoK:.3f} | {Entity:.3f} | {COMPS:.3f} | {GlobalPIQA:.3f} | {Reading:.3f} |'.format(**r))
lines += ['','## Contrasts','']
for c in contrasts:
    lines.append(f"- {c['contrast']}: cheap7 {c['cheap7_delta']:+.4f}, Entity {c['Entity_delta']:+.3f}, Reading {c['Reading_delta']:+.3f}, EWoK {c['EWoK_delta']:+.3f}")
((_PUBLIC_ROOT / 'research/documents/relation_learning/data/seed43022_cheap7/summary.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':summary['status'],'summary':str((OUT/'summary.json').relative_to(USER_ROOT)),'md':str(((_PUBLIC_ROOT / 'research/documents/relation_learning/data/seed43022_cheap7/summary.md')).relative_to(USER_ROOT))},indent=2),flush=True)
