#!/usr/bin/env python3
import importlib.util
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

p = Path('experiments/archive/representation_and_objectives/training/scripts/temporal_indirect_address_probe.py')
spec = importlib.util.spec_from_file_location('ind267_audit', p)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)

class A: pass

def make_args(train_families, eval_families='original_current_exact'):
    a=A(); a.base_stable=20; a.sparse_changed=8; a.sparse_stable=8
    a.eval_held_changed=10; a.eval_held_stable=10; a.eval_train_changed=10; a.eval_train_stable=10
    a.train_families=train_families; a.eval_families=eval_families; a.train_direct_tag=True
    return a

for tf in ['background_update','background_update,initial_latest,old_new']:
    args=make_args(tf)
    base_rows, update_rows, evals, construction=mod.build(args)
    rows=list(base_rows)+list(update_rows)
    by_text=defaultdict(list)
    for r in rows:
        by_text[r['text']].append(r)
    conflicts=[]; dup=0
    for txt, rs in by_text.items():
        if len(rs)>1:
            dup+=len(rs)
            labs=Counter(r['label'] for r in rs)
            if len(labs)>1:
                conflicts.append({'n':len(rs),'labels':dict(labs),'text':txt[:300],'ids':[r['id'] for r in rs[:5]],'kinds':[r['train_kind'] for r in rs[:5]]})
    kind=Counter(r['train_kind'] for r in rows)
    labels=Counter(r['label'] for r in rows)
    qfam=Counter(r['query_family'] for r in rows)
    print(json.dumps({'train_families':tf,'n_rows':len(rows),'unique_texts':len(by_text),'dup_rows':dup,'n_conflict_texts':len(conflicts),'labels':dict(labels),'query_family':dict(qfam),'kind_counts':dict(kind)}))
    if conflicts:
        print(json.dumps({'first_conflict':conflicts[0]}, indent=2))
