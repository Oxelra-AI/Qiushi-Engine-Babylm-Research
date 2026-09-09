#!/usr/bin/env python3
"""Summarize SHUF seed43122 cheap7 and compare with COMPACT_EXPERIENCE OFF/ALN seed43122 columns."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, statistics, time
ROOT0 = _public_path('experiments/archive/relation_learning/scripts/summarize_shuf_seed43122_cheap7.py')
ROOT = _PUBLIC_ROOT
OUT = ROOT / 'experiments/archive/relation_learning/data/shuf_seed43122_cheap7'
SHUF_SUM = OUT / 'shuf43122/shuf_seed43122_cheap7_summary.json'
COMPACT_EXPERIENCE_SUM = ROOT / 'experiments/archive/compact_experience/data/control_eval_summary.json'
CHEAP = ['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA','Reading']
rec = json.loads(SHUF_SUM.read_text(encoding='utf-8'))['record']
shuf = dict(rec.get('scores') or {})
shuf['cheap7'] = rec.get('cheap7')
compact_experience = json.loads(COMPACT_EXPERIENCE_SUM.read_text(encoding='utf-8'))['targets']
off = {c: compact_experience['official_lengthmatched_seed43122'].get(c) for c in CHEAP}; off['cheap7'] = statistics.mean([off[c] for c in CHEAP])
aln = {c: compact_experience['qwen_clean_aligned_seed43122'].get(c) for c in CHEAP}; aln['cheap7'] = statistics.mean([aln[c] for c in CHEAP])
def contrast(a: dict, b: dict, name: str):
    row={'contrast':name}
    for c in CHEAP+['cheap7']:
        va, vb = a.get(c), b.get(c)
        row[c+'_delta'] = va - vb if va is not None and vb is not None else None
    row['cheap6_no_GlobalPIQA_delta'] = statistics.mean([a[c] for c in ['BLiMP','Supplement','EWoK','Entity','COMPS','Reading']]) - statistics.mean([b[c] for c in ['BLiMP','Supplement','EWoK','Entity','COMPS','Reading']])
    return row
summary = {
    'status': 'SHUF_SEED43122_CHEAP7_SUMMARY',
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'targets': {
        'OFF43122_from_COMPACT_EXPERIENCE_step030': off,
        'ALN43122_from_COMPACT_EXPERIENCE_step030': aln,
        'SHUF43122_step075': shuf,
    },
    'contrasts': [
        contrast(shuf, off, 'SHUF-minus-OFF'),
        contrast(aln, shuf, 'ALN-minus-SHUF'),
        contrast(aln, off, 'ALN-minus-OFF'),
    ],
    'record': rec,
    'scientific_note': 'SHUF holds the selected original/rewrite text multiset and same-window displacement structure while breaking original-rewrite correspondence. Comparing ALN-SHUF at the same seed tests correct-pair relation effects in cheap columns; this remains a different COMPACT_EXPERIENCE substrate from FUNCTIONAL_RELATION_STUDIES marginal dose.'
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
lines=['# research SHUF seed43122 cheap7','','| arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,row in [('OFF43122',off),('SHUF43122',shuf),('ALN43122',aln)]:
    vals={c: row.get(c) for c in CHEAP}; vals['cheap7']=row.get('cheap7')
    lines.append('| {name} | {cheap7:.4f} | {BLiMP:.3f} | {Supplement:.3f} | {EWoK:.3f} | {Entity:.3f} | {COMPS:.3f} | {GlobalPIQA:.3f} | {Reading:.3f} |'.format(name=name, **vals))
lines += ['','## Contrasts']
for cr in summary['contrasts']:
    lines.append(f"- {cr['contrast']}: cheap7 {cr['cheap7_delta']:+.4f}, cheap6(no GP) {cr['cheap6_no_GlobalPIQA_delta']:+.4f}, BLiMP {cr['BLiMP_delta']:+.3f}, Supplement {cr['Supplement_delta']:+.3f}, EWoK {cr['EWoK_delta']:+.3f}, Entity {cr['Entity_delta']:+.3f}, Reading {cr['Reading_delta']:+.3f}")
lines += ['', summary['scientific_note']]
((_PUBLIC_ROOT / 'research/documents/relation_learning/data/shuf_seed43122_cheap7/summary.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':summary['status'],'summary':str((OUT/'summary.json').relative_to(ROOT)),'md':str(((_PUBLIC_ROOT / 'research/documents/relation_learning/data/shuf_seed43122_cheap7/summary.md')).relative_to(ROOT))},indent=2),flush=True)
