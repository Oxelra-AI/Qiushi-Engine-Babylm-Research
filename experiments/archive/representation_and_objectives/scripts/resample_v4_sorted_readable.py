#!/usr/bin/env python3
"""Create readable sorted samples from the research v4 selector pool."""
from __future__ import annotations
import json, pathlib, random
from collections import defaultdict
ROOT=pathlib.Path('experiments/archive/representation_and_objectives')
IN=ROOT/'data/live_fineweb_selector_v4_high_precision/live_fineweb_selector_v4_high_precision_kept_sorted.jsonl'
OUT=ROOT/'data/live_fineweb_selector_v4_high_precision/live_fineweb_selector_v4_sorted_samples.json'
NOTE=(ROOT.parents[2] / 'research/notes/representation_and_objectives/live_fineweb_selector_v4_sample_reading.md')
rows=[]
with IN.open(encoding='utf-8') as f:
    for line in f:
        if line.strip(): rows.append(json.loads(line))
# first rows in sorted order are highest-priority definitions/causal/spatial/historical, not random tail.
head=rows[:80]
by_type=defaultdict(list)
for r in rows:
    for t in r.get('selector_v4_types',[]):
        if len(by_type[t])<30: by_type[t].append(r)
rng=random.Random(18022)
random_sample=rng.sample(rows, min(80,len(rows)))
# compressed view for human reading
def slim(r):
    return {
        'text': r.get('selector_v4_text') or r.get('text'),
        'words': r.get('selector_v4_words'),
        'types': r.get('selector_v4_types'),
        'anchors': r.get('selector_v4_caps', [])[:6] + r.get('selector_v4_numbers', [])[:6],
        'doc_id': r.get('doc_id'),
        'sentence_index': r.get('sentence_index'),
    }
payload={'status':'V4_SORTED_SAMPLES','n_rows':len(rows),'head_sorted':[slim(r) for r in head],'by_type':{k:[slim(r) for r in v] for k,v in by_type.items()},'random_sample':[slim(r) for r in random_sample]}
OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')
lines=['# research selector-v4 sample reading\n\n',f'Built compact sample bundle from `{IN}` with {len(rows):,} rows. The `head_sorted` list shows the deterministic high-priority materialization order (definitions/causal/spatial/historical before shallow relational rows); random JSON samples may overrepresent long-tail v4 imperfections.\n\n',f'Sample JSON: `{OUT}`\n']
NOTE.write_text(''.join(lines),encoding='utf-8')
print(json.dumps({'status':payload['status'],'out':str(OUT),'note':str(NOTE),'n_rows':len(rows)}, indent=2))
if __name__=='__main__':
    pass
