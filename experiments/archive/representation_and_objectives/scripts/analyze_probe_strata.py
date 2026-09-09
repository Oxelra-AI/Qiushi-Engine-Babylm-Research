#!/usr/bin/env python3
"""Stratify research symmetric cue-only probe by same-pivot shuffled donors."""
import json
import collections
from pathlib import Path

P = Path('experiments/archive/representation_and_objectives/data/symmetric_cue_view_likelihood_probe/scored_events.jsonl')
OUT = Path('experiments/archive/representation_and_objectives/data/symmetric_cue_view_likelihood_probe/stratified_samepivot_analysis.json')
NOTE = Path('research/notes/representation_and_objectives/probe_stratified_samepivot_analysis.md')
rows = [json.loads(line) for line in P.open(encoding='utf-8') if line.strip()]

def summarize(sub):
    if not sub:
        return {'n': 0}
    ds = [float(e['logp_semantic']) - float(e['logp_shuffle_samepos']) for e in sub]
    da = [float(e['logp_semantic']) - float(e['logp_anchor_samepos']) for e in sub]
    dn = [float(e['logp_semantic']) - float(e['logp_no_cue']) for e in sub]
    same = [str(e.get('pivot_norm')) == str(e.get('shuffle_pivot_norm')) for e in sub]
    return {
        'n': len(sub),
        'same_pivot_frac': sum(same) / len(sub),
        'sem_minus_shuffle_mean': sum(ds) / len(ds),
        'sem_gt_shuffle_success': sum(d > 1e-9 for d in ds) / len(ds),
        'sem_shuffle_exact_tie_frac': sum(abs(d) <= 1e-9 for d in ds) / len(ds),
        'sem_minus_anchor_mean': sum(da) / len(da),
        'sem_gt_anchor_success': sum(d > 0 for d in da) / len(da),
        'sem_minus_no_cue_mean': sum(dn) / len(dn),
        'sem_gt_no_cue_success': sum(d > 0 for d in dn) / len(dn),
    }

out = {'overall': {}, 'by_pivot_norm': {}, 'by_target_class': {}, 'by_source': {}}
for cat in ['ALL', 'comparative', 'physical_change', 'causal_connector', 'temporal', 'spatial', 'negation']:
    sub = rows if cat == 'ALL' else [e for e in rows if e['category'] == cat]
    diff = [e for e in sub if str(e.get('pivot_norm')) != str(e.get('shuffle_pivot_norm'))]
    same = [e for e in sub if str(e.get('pivot_norm')) == str(e.get('shuffle_pivot_norm'))]
    out['overall'][cat] = {'all': summarize(sub), 'diffpivot_only': summarize(diff), 'samepivot_only': summarize(same)}
    for field, target in [('pivot_norm', out['by_pivot_norm']), ('target_class', out['by_target_class']), ('source', out['by_source'])]:
        ctr = collections.Counter(str(e.get(field)) for e in sub)
        target[cat] = {}
        for key, _ in ctr.most_common(30):
            ssub = [e for e in sub if str(e.get(field)) == key]
            target[cat][key] = summarize(ssub)
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
lines = []
lines.append('# research probe stratification by same-pivot shuffled donors\n')
for cat in ['ALL', 'comparative', 'physical_change', 'causal_connector', 'temporal', 'spatial', 'negation']:
    item = out['overall'][cat]
    lines.append(f"- {cat}: all={item['all']}\n")
    lines.append(f"  - diffpivot_only={item['diffpivot_only']}\n")
    lines.append(f"  - samepivot_only={item['samepivot_only']}\n")
lines.append('\n## Comparative by pivot_norm\n')
for key, item in out['by_pivot_norm']['comparative'].items():
    lines.append(f"- {key}: {item}\n")
NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'status': 'ok', 'out': str(OUT), 'note': str(NOTE), 'comparative': out['overall']['comparative'], 'physical_change': out['overall']['physical_change']}, ensure_ascii=False))
