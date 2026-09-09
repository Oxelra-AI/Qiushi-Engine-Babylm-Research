#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
FILES = {
    'consistency_10M': ROOT/'data/entity_consistency_consistency_10M_available_coordinate.json',
    'consistency_20M': ROOT/'data/entity_consistency_consistency_20M_available_coordinate.json',
    'shuffled_10M': ROOT/'data/entity_consistency_shuffled_10M_available_coordinate.json',
    'shuffled_20M': ROOT/'data/entity_consistency_shuffled_20M_available_coordinate.json',
}
TRAIN_VAL = ROOT/'data/entity_consistency_20m_training_validation.json'
OUT = ROOT/'data/entity_consistency_available_coordinate_comparison.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/entity_consistency_available_coordinate_comparison.md')
COLS = [
    ('blimp','BLiMP'), ('supplement','Supplement'), ('entity_tracking','Entity'), ('comps','COMPS'),
    ('global_piqa_parallel','GlobalPIQA parallel'), ('global_piqa_nonparallel','GlobalPIQA nonparallel'),
    ('global_piqa_mean','GlobalPIQA mean'), ('reading_eye_tracking','Reading eye'),
    ('reading_self_paced','Reading self-paced'), ('reading_mean','Reading mean')]

def load(p: pathlib.Path) -> dict:
    d = json.loads(p.read_text(encoding='utf-8'))
    s = dict(d['scores'])
    der = d.get('derived_columns', {})
    s['global_piqa_mean'] = der.get('GlobalPIQA_mean_parallel_nonparallel', (s['global_piqa_parallel']+s['global_piqa_nonparallel'])/2)
    s['reading_mean'] = der.get('Reading_mean_eye_self_paced', (s['reading_eye_tracking']+s['reading_self_paced'])/2)
    return {'raw': d, 'scores': s}

def delta(a,b):
    return {k: a['scores'][k]-b['scores'][k] for k,_ in COLS}

def sum_keys(d, keys):
    return sum(d[k] for k in keys)

def main():
    coords = {k: load(p) for k,p in FILES.items()}
    train = json.loads(TRAIN_VAL.read_text(encoding='utf-8')) if TRAIN_VAL.exists() else None
    d10 = delta(coords['consistency_10M'], coords['shuffled_10M'])
    d20 = delta(coords['consistency_20M'], coords['shuffled_20M'])
    prog_c = delta(coords['consistency_20M'], coords['consistency_10M'])
    prog_s = delta(coords['shuffled_20M'], coords['shuffled_10M'])
    protected_100m = json.loads((ROOT/'data/debertav2_b256_true_9of9_coordinate.json').read_text(encoding='utf-8'))
    summary = {
        'known_six_sum_delta_10M': sum_keys(d10, ['blimp','supplement','entity_tracking','comps','global_piqa_mean','reading_mean']),
        'known_six_sum_delta_20M': sum_keys(d20, ['blimp','supplement','entity_tracking','comps','global_piqa_mean','reading_mean']),
        'target_entity_globalpiqa_sum_delta_10M': sum_keys(d10, ['entity_tracking','global_piqa_mean']),
        'target_entity_globalpiqa_sum_delta_20M': sum_keys(d20, ['entity_tracking','global_piqa_mean']),
        'guard_supp_blimp_reading_sum_delta_10M': sum_keys(d10, ['supplement','blimp','reading_mean']),
        'guard_supp_blimp_reading_sum_delta_20M': sum_keys(d20, ['supplement','blimp','reading_mean']),
        'entity_delta_10M': d10['entity_tracking'],
        'entity_delta_20M': d20['entity_tracking'],
        'globalpiqa_mean_delta_10M': d10['global_piqa_mean'],
        'globalpiqa_mean_delta_20M': d20['global_piqa_mean'],
        'reading_mean_delta_10M': d10['reading_mean'],
        'reading_mean_delta_20M': d20['reading_mean'],
    }
    payload = {
        'status': 'ENTITY_CONSISTENCY_AVAILABLE_COORDINATE_COMPARISON',
        'files': {k: str(v) for k,v in FILES.items()},
        'scores': {k: coords[k]['scores'] for k in coords},
        'consistency_minus_shuffled_10M': d10,
        'consistency_minus_shuffled_20M': d20,
        'progression_consistency_20M_minus_10M': prog_c,
        'progression_shuffled_20M_minus_10M': prog_s,
        'summary': summary,
        'training_validation': train,
        'protected_100m_reference': {'Overall Average': protected_100m['Overall Average'], 'scores_official_columns': protected_100m['scores_official_columns']},
        'interpretation': 'Higher is better for all listed columns including Reading. 100M scaling requires multi-column improvement, not isolated same-form Entity movement.',
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# research — Entity Mention Consistency available-coordinate comparison', '', f'Evidence JSON: `{OUT}`', '', 'Higher is better for all columns including Reading.', '']
    for exposure, ck, sk, dd in [('10M','consistency_10M','shuffled_10M',d10), ('20M','consistency_20M','shuffled_20M',d20)]:
        lines += [f'## {exposure}: consistency vs shuffled_pair', '', '| column | consistency | shuffled_pair | delta |', '|---|---:|---:|---:|']
        for key,label in COLS:
            lines.append(f"| {label} | {coords[ck]['scores'][key]:.2f} | {coords[sk]['scores'][key]:.2f} | {dd[key]:+.2f} |")
        lines.append('')
    lines += ['## Summary', '']
    for k,v in summary.items():
        lines.append(f'- {k}: {v}')
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'out': str(OUT), 'summary': summary}, indent=2))

if __name__ == '__main__':
    main()
