#!/usr/bin/env python3
"""research synthesis for intra-row two-argument four-cell calibration.

Reads pair-level no-update scores and asks whether any subset of the same-context
left/right role objective behaves like the known relation failure surfaces rather
than broad compact context fit. No model is loaded.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCORE_CSV = Path('experiments/archive/representation_and_objectives/data/intrarow_twoarg_calibration_pilot20k/fourcell_pair_scores_summary.csv')
PAIR_POOL = Path('experiments/archive/representation_and_objectives/data/intrarow_twoarg_relation_funnel_pilot20k/intrarow_twoarg_pair_pool.jsonl')
OUT = Path('experiments/archive/representation_and_objectives/data/intrarow_calibration_synthesis')
NOTE = Path('research/notes/representation_and_objectives/intrarow_calibration_synthesis.md')

BAD_TARGETS = {
    'can','could','will','would','should','may','might','must','shall','other','now','even','already','best','keep','help','little','play','goes',
    "you're","we'll","he's","there's",'about','rather','however','means','would','went','shall','number','still','again','mhm','yeah','okay','ok',
}


def qstats(vals: list[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {'n': 0}
    arr = np.asarray(xs, dtype=np.float64)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p*(len(xs)-1); lo=int(math.floor(idx)); hi=int(math.ceil(idx))
        if lo == hi: return xs[lo]
        return xs[lo]*(hi-idx)+xs[hi]*(idx-lo)
    return {'n': len(xs), 'mean': float(arr.mean()), 'std': float(arr.std()), 'stderr': float(arr.std()/math.sqrt(len(xs))), 'median': q(0.5), 'p05': q(0.05), 'p25': q(0.25), 'p75': q(0.75), 'p95': q(0.95), 'min': xs[0], 'max': xs[-1], 'success_gt0': float((arr>0).mean())}


def load_pairs() -> dict[str, dict[str, Any]]:
    out={}
    with PAIR_POOL.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                p=json.loads(line); out[p['pair_id']]=p
    return out


def load_scores(pairs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by=defaultdict(dict)
    with SCORE_CSV.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pid=r['pair_id']; arm=r['arm']
            rec=dict(pairs.get(pid, {})); rec.update(r)
            for k in ['true_m','true_m_a','true_m_b','tperm_m','cperm_m','pair_cost']:
                rec[k]=float(rec[k])
            by[pid][arm]=rec
    return {pid: arms for pid, arms in by.items() if all(a in arms for a in ['compact','rowblock','interleaved'])}


def subset_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {'n': 0}
    out={'n': len(items)}
    for metric in ['true_m','tperm_m','cperm_m']:
        for arm in ['compact','rowblock','interleaved']:
            out[f'{arm}_{metric}']=qstats([it[f'{arm}_{metric}'] for it in items])
        means={arm: out[f'{arm}_{metric}']['mean'] for arm in ['compact','rowblock','interleaved']}
        out[f'{metric}_order']=' > '.join(sorted(means, key=lambda a: means[a], reverse=True))
        out[f'{metric}_range']=max(means.values())-min(means.values())
        out[f'rowblock_minus_compact_{metric}']=qstats([it[f'rowblock_{metric}']-it[f'compact_{metric}'] for it in items])
        out[f'interleaved_minus_compact_{metric}']=qstats([it[f'interleaved_{metric}']-it[f'compact_{metric}'] for it in items])
    out['mean_true_m_allarms']=qstats([it['mean_true_m'] for it in items])
    out['compact_low_success_true_lt_2']=sum(1 for it in items if it['compact_true_m'] < 2)/len(items)
    out['rowblock_better_than_compact_true_frac']=sum(1 for it in items if it['rowblock_true_m'] > it['compact_true_m'])/len(items)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs=load_pairs(); scores=load_scores(pairs)
    rows=[]
    for pid, arms in scores.items():
        base=arms['compact']
        rec={
            'pair_id': pid, 'category': base.get('category'), 'pivot': base.get('pivot_norm_a'), 'target_a': base.get('target_norm_a'), 'target_b': base.get('target_norm_b'),
            'target_class': base.get('target_class'), 'target_token_len': int(base.get('target_token_len', 0)), 'source': base.get('source_a'),
            'left_distance': int(pairs[pid].get('left_distance', -1)), 'right_distance': int(pairs[pid].get('right_distance', -1)), 'between_content_count': int(pairs[pid].get('between_content_count', 0)),
            'bad_target': base.get('target_norm_a') in BAD_TARGETS or base.get('target_norm_b') in BAD_TARGETS,
        }
        for arm in ['compact','rowblock','interleaved']:
            for m in ['true_m','tperm_m','cperm_m']:
                rec[f'{arm}_{m}']=float(arms[arm][m])
        rec['mean_true_m']=(rec['compact_true_m']+rec['rowblock_true_m']+rec['interleaved_true_m'])/3
        rec['rowblock_minus_compact_true_m']=rec['rowblock_true_m']-rec['compact_true_m']
        rows.append(rec)
    subsets={
        'all': rows,
        'not_bad_targets': [r for r in rows if not r['bad_target']],
        'low_mean_true_lt_2': [r for r in rows if r['mean_true_m'] < 2],
        'low_mean_true_lt_5': [r for r in rows if r['mean_true_m'] < 5],
        'compact_true_lt_2': [r for r in rows if r['compact_true_m'] < 2],
        'rowblock_beats_compact_true': [r for r in rows if r['rowblock_true_m'] > r['compact_true_m']],
        'rowblock_beats_compact_true_not_bad': [r for r in rows if r['rowblock_true_m'] > r['compact_true_m'] and not r['bad_target']],
        'balanced_distance': [r for r in rows if abs(r['left_distance']-r['right_distance']) <= 1],
        'comparative': [r for r in rows if r['category']=='comparative'],
        'spatial': [r for r in rows if r['category']=='spatial'],
        'temporal': [r for r in rows if r['category']=='temporal'],
        'causal_connector': [r for r in rows if r['category']=='causal_connector'],
    }
    summary={'status':'INTRAROW_CALIBRATION_SYNTHESIS','scores':str(SCORE_CSV),'pair_pool':str(PAIR_POOL),'n_common_pairs':len(rows),'subsets':{k: subset_summary(v) for k,v in subsets.items()}}
    # Pivot-level table for pivots with enough examples.
    byp=defaultdict(list)
    for r in rows:
        byp[str(r['pivot'])].append(r)
    pivots=[]
    for p, xs in byp.items():
        if len(xs) >= 50:
            s=subset_summary(xs)
            pivots.append({'pivot':p,'n':len(xs),'true_order':s.get('true_m_order'),'true_range':s.get('true_m_range'),'rowblock_minus_compact_true_mean':s.get('rowblock_minus_compact_true_m',{}).get('mean'),'compact_true_mean':s.get('compact_true_m',{}).get('mean'),'mean_true_lt_5_frac':sum(1 for x in xs if x['mean_true_m']<5)/len(xs)})
    pivots.sort(key=lambda x:(x['rowblock_minus_compact_true_mean'] if x['rowblock_minus_compact_true_mean'] is not None else -999), reverse=True)
    summary['pivot_table_ge50']=pivots
    # Save low-margin and rowblock-positive examples for inspection.
    for name, xs in [('low_mean_true_lt_5', subsets['low_mean_true_lt_5']), ('rowblock_beats_compact_true_not_bad', subsets['rowblock_beats_compact_true_not_bad'])]:
        path=OUT/f'{name}.jsonl'
        with path.open('w',encoding='utf-8') as f:
            for r in sorted(xs, key=lambda z:(z['mean_true_m'], -z['rowblock_minus_compact_true_m']))[:500]:
                f.write(json.dumps(r, ensure_ascii=False)+'\n')
    (OUT/'intrarow_calibration_synthesis.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research — intra-row calibration synthesis','']
    lines.append(f"Common scored pairs: {len(rows):,}. This is a file-level synthesis of the pilot no-update four-cell scores; no model is loaded.")
    lines.append('')
    lines.append('## Key subset table')
    lines.append('| subset | n | true order | true range | rowblock-compact true mean | rowblock>true frac | compact true mean | low compact<2 frac | tperm range | cperm range |')
    lines.append('|---|---:|---|---:|---:|---:|---:|---:|---:|---:|')
    for k in ['all','not_bad_targets','low_mean_true_lt_2','low_mean_true_lt_5','compact_true_lt_2','rowblock_beats_compact_true','rowblock_beats_compact_true_not_bad','balanced_distance','causal_connector','comparative','spatial','temporal']:
        s=summary['subsets'][k]
        if not s.get('n'):
            lines.append(f'| {k} | 0 | | | | | | | | |'); continue
        lines.append(f"| {k} | {s['n']} | {s.get('true_m_order')} | {s.get('true_m_range'):.4f} | {s.get('rowblock_minus_compact_true_m',{}).get('mean'):.4f} | {s.get('rowblock_better_than_compact_true_frac'):.3f} | {s.get('compact_true_m',{}).get('mean'):.4f} | {s.get('compact_low_success_true_lt_2'):.3f} | {s.get('tperm_m_range'):.4f} | {s.get('cperm_m_range'):.4f} |")
    lines.append('')
    lines.append('## Top pivots by rowblock-minus-compact true margin, n>=50')
    lines.append('| pivot | n | true order | rb-compact | compact true mean | low mean<5 frac |')
    lines.append('|---|---:|---|---:|---:|---:|')
    for p in pivots[:20]:
        lines.append(f"| {p['pivot']} | {p['n']} | {p['true_order']} | {p['rowblock_minus_compact_true_mean']:.4f} | {p['compact_true_mean']:.4f} | {p['mean_true_lt_5_frac']:.3f} |")
    lines.append('')
    lines.append('Files:')
    lines.append(f"- summary JSON: `{OUT/'intrarow_calibration_synthesis.json'}`")
    lines.append(f"- low-margin examples: `{OUT/'low_mean_true_lt_5.jsonl'}`")
    lines.append(f"- rowblock-positive examples: `{OUT/'rowblock_beats_compact_true_not_bad.jsonl'}`")
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'],'n_common_pairs':len(rows),'summary':str(OUT/'intrarow_calibration_synthesis.json'),'note':str(NOTE)}, indent=2), flush=True)


if __name__=='__main__':
    main()
