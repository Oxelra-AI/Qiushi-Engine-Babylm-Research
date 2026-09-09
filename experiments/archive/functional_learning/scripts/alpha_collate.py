#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
EQ7=["BLiMP","Supplement","EWoK","Entity","COMPS","GlobalPIQA_mean","Reading"]
roots=[Path('experiments/archive/functional_learning/data/alpha_eval'),Path('experiments/archive/functional_learning/data/common_eval')]
rows=[]
for root in roots:
    if not root.exists():
        continue
    for p in sorted(root.rglob('*_eval.json')):
        if p.name.startswith('collated'):
            continue
        try:
            o=json.loads(p.read_text())
        except Exception:
            continue
        s=o.get('scores',{})
        vals=[s.get(k) for k in EQ7 if s.get(k) is not None]
        eq=sum(map(float, vals))/len(vals) if vals else None
        rows.append({'tag':o.get('tag',p.stem),'path':str(p),'n':len(vals),'eq':eq,'scores':{k:s.get(k) for k in EQ7},'alpha':o.get('alpha')})
print('tag\tn\teq\talpha\t'+'\t'.join(EQ7))
for r in rows:
    def fmt(x): return 'NA' if x is None else f'{float(x):.4f}'
    print('\t'.join([r['tag'],str(r['n']),fmt(r['eq']),str(r.get('alpha'))]+[fmt(r['scores'].get(k)) for k in EQ7]))
out=Path('experiments/archive/functional_learning/data/alpha_eval/collated_alpha_eval.json')
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'status':'ALPHA_COLLATION','rows':rows},indent=2),encoding='utf-8')
print('wrote',out)
