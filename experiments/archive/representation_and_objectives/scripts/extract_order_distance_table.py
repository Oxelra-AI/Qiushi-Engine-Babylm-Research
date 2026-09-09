#!/usr/bin/env python3
"""Extract order/distance residual table from research overwrite summary."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
from pathlib import Path
USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
INP = _public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/overwrite_decomposition_summary.json')
OUT_MD = _public_path('research/notes/representation_and_objectives/order_distance_residual_table.md')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/order_distance_residual_table.json')
LABELS = ['direct_adjacent','direct_then','direct_distance','targetfree_adjacent','targetfree_distance']
obj=json.loads(INP.read_text(encoding='utf-8'))
rows=[]
for s in obj['targets']:
    dec=s['decomposition']['by_label']
    row={'target':s['target']}
    for lab in LABELS:
        d=dec.get(lab,{})
        row[lab+'_resid_med']=d.get('residual_mean',{}).get('median')
        row[lab+'_neg_frac']=d.get('negative_nonadditive_frac')
        row[lab+'_combined_crossed']=d.get('combined_crossed_frac')
        row[lab+'_action_crossed']=d.get('action_crossed_frac')
    rows.append(row)
OUT_JSON.write_text(json.dumps({'rows':rows,'labels':LABELS},indent=2,sort_keys=True)+'\n',encoding='utf-8')
def fmt(x): return 'n/a' if x is None else f'{float(x):.4f}'
lines=['# research order/distance residual table','', '| target | direct adj R | direct then R | direct distance R | tf adj R | tf distance R | direct adj neg | direct dist neg | tf adj neg | tf dist neg |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['target']} | {fmt(r['direct_adjacent_resid_med'])} | {fmt(r['direct_then_resid_med'])} | {fmt(r['direct_distance_resid_med'])} | {fmt(r['targetfree_adjacent_resid_med'])} | {fmt(r['targetfree_distance_resid_med'])} | {fmt(r['direct_adjacent_neg_frac'])} | {fmt(r['direct_distance_neg_frac'])} | {fmt(r['targetfree_adjacent_neg_frac'])} | {fmt(r['targetfree_distance_neg_frac'])} |")
lines.append('')
lines.append(f"JSON: `{_public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/order_distance_residual_table.json').relative_to(USER_ROOT)}`")
OUT_MD.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':'DONE','out_md':str(_public_path('research/notes/representation_and_objectives/order_distance_residual_table.md').relative_to(USER_ROOT)),'out_json':str(_public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/order_distance_residual_table.json').relative_to(USER_ROOT))}))
