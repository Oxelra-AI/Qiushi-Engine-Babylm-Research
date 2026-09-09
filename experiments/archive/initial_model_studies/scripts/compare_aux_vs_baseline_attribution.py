#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
OUT = ROOT/'data/entity_consistency_vs_wwm_attribution.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/entity_consistency_vs_wwm_attribution.md')

FILES = {
    'wwm_10M': ROOT/'data/baseline_wwm_baseline_wwm_10M_available_coordinate.json',
    'wwm_20M': ROOT/'data/baseline_wwm_baseline_wwm_20M_available_coordinate.json',
    'seed1_consistency_10M': ROOT/'data/entity_consistency_consistency_10M_available_coordinate.json',
    'seed1_consistency_20M': ROOT/'data/entity_consistency_consistency_20M_available_coordinate.json',
    'seed1_shuffled_10M': ROOT/'data/entity_consistency_shuffled_10M_available_coordinate.json',
    'seed1_shuffled_20M': ROOT/'data/entity_consistency_shuffled_20M_available_coordinate.json',
    'seed2_consistency_10M': ROOT/'data/seed2_entity_consistency_consistency_10M_available_coordinate.json',
    'seed2_consistency_20M': ROOT/'data/seed2_entity_consistency_consistency_20M_available_coordinate.json',
    'seed2_shuffled_10M': ROOT/'data/seed2_entity_consistency_shuffled_10M_available_coordinate.json',
    'seed2_shuffled_20M': ROOT/'data/seed2_entity_consistency_shuffled_20M_available_coordinate.json',
}
COLS = [('blimp','BLiMP'),('supplement','Supplement'),('entity_tracking','Entity'),('comps','COMPS'),('global_piqa_mean','GlobalPIQA mean'),('reading_mean','Reading mean')]
TARGET = ['entity_tracking','global_piqa_mean']
GUARD = ['supplement','blimp','reading_mean']
SIX = [k for k,_ in COLS]

def load(path: pathlib.Path) -> dict:
    d = json.loads(path.read_text(encoding='utf-8'))
    s = dict(d['scores'])
    der = d.get('derived_columns', {})
    s['global_piqa_mean'] = der.get('GlobalPIQA_mean_parallel_nonparallel', (s['global_piqa_parallel'] + s['global_piqa_nonparallel'])/2.0)
    s['reading_mean'] = der.get('Reading_mean_eye_self_paced', (s['reading_eye_tracking'] + s['reading_self_paced'])/2.0)
    return s

def delta(a: dict, b: dict) -> dict:
    return {k: a[k]-b[k] for k,_ in COLS}

def pack_delta(d: dict) -> dict:
    return {**d, 'six_sum': sum(d[k] for k in SIX), 'target_entity_globalpiqa_sum': sum(d[k] for k in TARGET), 'guard_supp_blimp_reading_sum': sum(d[k] for k in GUARD)}

def main():
    scores = {k: load(p) for k,p in FILES.items()}
    comparisons = {}
    for seed in ['seed1','seed2']:
        for exp in ['10M','20M']:
            c = scores[f'{seed}_consistency_{exp}']; sh = scores[f'{seed}_shuffled_{exp}']; w = scores[f'wwm_{exp}']
            comparisons[f'{seed}_{exp}_consistency_minus_wwm'] = pack_delta(delta(c,w))
            comparisons[f'{seed}_{exp}_shuffled_minus_wwm'] = pack_delta(delta(sh,w))
            comparisons[f'{seed}_{exp}_consistency_minus_shuffled'] = pack_delta(delta(c,sh))
    # cross-seed mean for consistency-vs-WWM and shuffled-vs-WWM at each exposure
    means = {}
    for exp in ['10M','20M']:
        for arm in ['consistency','shuffled']:
            keys = [f'seed1_{exp}_{arm}_minus_wwm', f'seed2_{exp}_{arm}_minus_wwm']
            means[f'{exp}_{arm}_minus_wwm_mean_over_2seeds'] = {m: sum(comparisons[k][m] for k in keys)/2.0 for m in comparisons[keys[0]]}
    interpretation = {
        'main_point': 'The shuffled_pair wrong-pair auxiliary is not a neutral no-auxiliary baseline. Absolute comparison to ordinary WWM shows both auxiliary arms often raise GlobalPIQA relative to WWM, while guard columns and Reading vary strongly by seed. This confirms Entity Mention Consistency cannot be judged solely by consistency-minus-shuffled.',
        'scaling_decision': 'Do not scale repeated-form consistency to 100M: seed2 failed consistency-vs-shuffled replication and consistency-vs-WWM does not show a stable broad guard-safe improvement.',
    }
    payload = {'status':'AUX_VS_WWM_ATTRIBUTION','files':{k:str(v) for k,v in FILES.items()},'scores':scores,'comparisons':comparisons,'mean_over_2seeds':means,'interpretation':interpretation}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# research — Entity Consistency absolute attribution vs ordinary WWM','',f'Evidence JSON: `{OUT}`','','Higher is better for all columns, including Reading. Deltas are arm minus ordinary WWM at matched exposure.','']
    for exp in ['10M','20M']:
        lines += [f'## {exp}: arm minus protected ordinary WWM','', '| arm | BLiMP | Supp | Entity | COMPS | GPIQA mean | Reading mean | six-sum | target E+G | guard S+B+R |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for seed in ['seed1','seed2']:
            for arm in ['consistency','shuffled']:
                d = comparisons[f'{seed}_{exp}_{arm}_minus_wwm']
                lines.append(f"| {seed} {arm} | {d['blimp']:+.2f} | {d['supplement']:+.2f} | {d['entity_tracking']:+.2f} | {d['comps']:+.2f} | {d['global_piqa_mean']:+.2f} | {d['reading_mean']:+.2f} | {d['six_sum']:+.2f} | {d['target_entity_globalpiqa_sum']:+.2f} | {d['guard_supp_blimp_reading_sum']:+.2f} |")
        lines.append('')
    lines += ['## Interpretation','','- The wrong-pair `shuffled_pair` arm itself often changes scores substantially relative to ordinary WWM, especially GlobalPIQA and Reading, so it is not a neutral no-auxiliary baseline.','- Entity Mention Consistency remains closed for 100M scaling: its consistency-minus-shuffled effect did not replicate and its absolute pattern versus WWM is not a stable guard-safe broad improvement.','- The next route should use a structure-destroyed control that preserves lexical/frequency statistics but does not impose a harmful wrong-pair contrast objective; ordinary WWM must remain an explicit absolute baseline.']
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':'AUX_VS_WWM_ATTRIBUTION','out':str(OUT)}, indent=2))

if __name__=='__main__':
    main()
