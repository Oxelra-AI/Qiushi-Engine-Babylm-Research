#!/usr/bin/env python3
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

p = Path('experiments/archive/representation_and_objectives/training/scripts/temporal_indirect_address_probe.py')
spec = importlib.util.spec_from_file_location('ind267_dbg', p)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)

class A:
    pass

args = A()
args.base_stable = 40
args.sparse_changed = 16
args.sparse_stable = 16
args.eval_held_changed = 40
args.eval_held_stable = 40
args.eval_train_changed = 40
args.eval_train_stable = 40
args.train_families = 'background_update,initial_latest,old_new'
args.eval_families = 'original_current_exact,prior_revised_to_original_current,earlier_later_to_original_current'
args.train_direct_tag = True
base_rows, update_rows, evals, construction = mod.build(args)
tok = AutoTokenizer.from_pretrained(str(mod.DEFAULT_MODEL))
rows = list(base_rows) + list(update_rows)
lengths = [len(tok(r['text'], add_special_tokens=True)['input_ids']) for r in rows]
print(json.dumps({
    'n_train': len(rows),
    'min': min(lengths),
    'p50': float(np.percentile(lengths, 50)),
    'p90': float(np.percentile(lengths, 90)),
    'p99': float(np.percentile(lengths, 99)),
    'max': max(lengths),
    'frac_gt200': sum(l > 200 for l in lengths) / len(lengths),
    'frac_gt256': sum(l > 256 for l in lengths) / len(lengths),
    'frac_gt320': sum(l > 320 for l in lengths) / len(lengths),
}))
for ml in [200, 256, 320, 384]:
    miss_hyp = 0
    miss_sep = 0
    examples = []
    for r, l in zip(rows[:300], lengths[:300]):
        enc = tok(r['text'], truncation=True, max_length=ml)
        txt = tok.decode(enc['input_ids'])
        if 'ranked higher' not in txt and 'outranked' not in txt:
            miss_hyp += 1
            if len(examples) < 1:
                examples.append({'len': l, 'orig_tail': r['text'][-160:], 'decoded_tail': txt[-220:]})
        if '[SEP]' not in txt:
            miss_sep += 1
    print(json.dumps({'max_len': ml, 'first300_missing_hyp': miss_hyp, 'first300_missing_sep': miss_sep, 'examples': examples}))
for r, l in zip(rows, lengths):
    if 'sparse_changed' in r['train_kind']:
        print(json.dumps({'sample_kind': r['train_kind'], 'len': l, 'text': r['text'][:800]}))
        break
