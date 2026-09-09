#!/usr/bin/env python3
"""Compare research matched-probe frozen scores between chck82 and chck100."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, math, statistics
from collections import defaultdict
from pathlib import Path
ROOT=_public_path('.')
BASE=_public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe')
P82=_public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck82_scores_v2/matched_probe_paired_effects.csv')
P100=_public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck100_scores_v2/matched_probe_paired_effects.csv')
OUT=_public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_82_100_comparison')
OUT.mkdir(parents=True, exist_ok=True)
def read(p):
    rows=[]
    with p.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            for k,v in list(r.items()):
                if k not in {'prompt_id','family'}:
                    try: r[k]=float(v)
                    except Exception: pass
            rows.append(r)
    return {r['prompt_id']:r for r in rows}
def mean(xs):
    xs=[float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(xs) if xs else None
def summ(xs):
    xs=[float(x) for x in xs if math.isfinite(float(x))]
    if not xs: return {'n':0}
    xs=sorted(xs)
    return {'n':len(xs),'mean':mean(xs),'min':min(xs),'max':max(xs),'median':xs[len(xs)//2]}
r82=read(P82); r100=read(P100)
keys=sorted(set(r82)&set(r100))
metrics=['lift_delta_structured_minus_neutral','sensitivity_delta_structured_minus_reversed','target_nll_improvement_neutral_minus_structured','target_nll_improvement_reversed_minus_structured','structured_delta','neutral_delta','reversed_delta','query_only_delta']
rows=[]
for k in keys:
    a=r82[k]; b=r100[k]
    row={'prompt_id':k,'family':a.get('family')}
    for m in metrics:
        row[f'{m}_82']=a[m]; row[f'{m}_100']=b[m]; row[f'{m}_82_minus_100']=a[m]-b[m]
    rows.append(row)
outcsv=_public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_82_100_comparison/matched_probe_82_100_by_item.csv')
with outcsv.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader(); w.writerows(rows)
summary={'status':'MATCHED_PROBE_CHCK82_VS_CHCK100','n_common':len(keys),'input_82':str(P82.relative_to(ROOT)),'input_100':str(P100.relative_to(ROOT)),'by_metric':{}}
for m in metrics:
    summary['by_metric'][m]={'chck82':summ([r82[k][m] for k in keys]),'chck100':summ([r100[k][m] for k in keys]),'delta_82_minus_100':summ([r82[k][m]-r100[k][m] for k in keys])}
byfam=defaultdict(list)
for r in rows: byfam[r['family']].append(r)
summary['by_family_delta']={}
for fam,rs in byfam.items():
    summary['by_family_delta'][fam]={m:mean([r[f'{m}_82_minus_100'] for r in rs]) for m in metrics[:4]}
outjson=_public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_82_100_comparison/matched_probe_82_100_comparison.json'); outmd=_public_path('research/documents/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_82_100_comparison/matched_probe_82_100_comparison.md')
outjson.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
lines=['# research matched probe: chck82 vs chck100','',f'Common items: {len(keys)}','']
for m in metrics[:4]:
    d=summary['by_metric'][m]['delta_82_minus_100']
    lines.append(f'- {m}: chck82 mean {summary["by_metric"][m]["chck82"].get("mean")}, chck100 mean {summary["by_metric"][m]["chck100"].get("mean")}, 82-100 mean {d.get("mean")}, range [{d.get("min")}, {d.get("max")}]')
lines.append('')
lines.append('Interpretation: if this diagnostic captured the protected 82M relation/state competence, 82M should dominate 100M on structured lift/sensitivity. Similar or higher 100M values indicate the probe is mostly synthetic/query-prior/local relation-word behavior, not the official late-loss phenomenon.')
lines.append(f'\nBy-item CSV: `{outcsv.relative_to(ROOT)}`')
lines.append(f'JSON: `{outjson.relative_to(ROOT)}`')
outmd.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':summary['status'],'out_json':str(outjson.relative_to(ROOT)),'out_md':str(outmd.relative_to(ROOT))},indent=2))
