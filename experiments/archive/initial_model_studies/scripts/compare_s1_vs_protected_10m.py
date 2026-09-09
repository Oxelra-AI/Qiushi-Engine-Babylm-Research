#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
S1 = ROOT / 'data/s1_10m_available_coordinate.json'
P10 = ROOT / 'data/protected_8x480_10m_available_coordinate.json'
P100 = ROOT / 'data/debertav2_b256_true_9of9_coordinate.json'
OUT = ROOT / 'data/s1_vs_protected_10m_comparison.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/s1_vs_protected_10m_comparison.md')

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def r(x): return None if x is None else round(float(x), 4)

def main():
    s1=load(S1); p10=load(P10); p100=load(P100)
    s1_scores=s1['scores']; p10_scores=p10['scores']
    cols=[
        ('blimp','BLiMP'),('supplement','Supplement'),('entity_tracking','Entity'),('comps','COMPS'),
        ('global_piqa_parallel','GlobalPIQA parallel'),('global_piqa_nonparallel','GlobalPIQA nonparallel'),
        ('global_piqa_mean','GlobalPIQA mean'),('reading_eye_tracking','Reading eye'),('reading_self_paced','Reading self-paced'),('reading_mean','Reading mean')]
    s1_all={**s1_scores,'global_piqa_mean':s1['derived_columns']['GlobalPIQA_mean_parallel_nonparallel'],'reading_mean':s1['derived_columns']['Reading_mean_eye_selfpaced']}
    p10_all={**p10_scores,'global_piqa_mean':p10['derived_columns']['GlobalPIQA_mean_parallel_nonparallel'],'reading_mean':p10['derived_columns']['Reading_mean_eye_selfpaced']}
    p100_all={
        'blimp':p100['scores_official_columns']['BLiMP'],
        'supplement':p100['scores_official_columns']['BLiMP Supplement'],
        'entity_tracking':p100['scores_official_columns']['Entity Tracking'],
        'comps':p100['scores_official_columns']['COMPS'],
        'global_piqa_mean':p100['scores_official_columns']['GlobalPIQA'],
        'global_piqa_parallel':p100['subcolumns']['global_piqa_parallel'],
        'global_piqa_nonparallel':p100['subcolumns']['global_piqa_nonparallel'],
        'reading_eye_tracking':p100['subcolumns']['reading_eye_tracking'],
        'reading_self_paced':p100['subcolumns']['reading_self_paced'],
        'reading_mean':p100['scores_official_columns']['Reading'],
    }
    rows=[]
    for key,label in cols:
        rows.append({'key':key,'label':label,'s1_12x384_10m':r(s1_all.get(key)),'protected_8x480_10m':r(p10_all.get(key)),'delta_s1_minus_protected10m':r(s1_all.get(key)-p10_all.get(key)),'protected_8x480_100m':r(p100_all.get(key)),'protected_gain_10m_to_100m':r(p100_all.get(key)-p10_all.get(key))})
    target_keys=['entity_tracking','comps','global_piqa_mean']
    guard_keys=['blimp','supplement','reading_mean']
    target_delta=sum(s1_all[k]-p10_all[k] for k in target_keys)/len(target_keys)
    guard_delta=sum(s1_all[k]-p10_all[k] for k in guard_keys)/len(guard_keys)
    proj={}
    for key,label in cols:
        # crude same-gain projection: S1_10M + protected(100M-10M) gain, not a claim.
        if p100_all.get(key) is not None:
            proj[key]=s1_all[key]+(p100_all[key]-p10_all[key])
    payload={
        'status':'S1_VS_PROTECTED_10M_COMPARISON',
        's1_file':str(S1),'protected_10m_file':str(P10),'protected_100m_file':str(P100),
        'rows':rows,
        'target_cluster_mean_delta_s1_minus_protected10m_entity_comps_globalpiqa':target_delta,
        'guard_cluster_mean_delta_s1_minus_protected10m_blimp_supp_reading':guard_delta,
        'same_gain_projection_not_claim':{k:r(v) for k,v in proj.items()},
        'interpretation':[
            'S1 shape alone is not a broad 10M winner: it loses BLiMP (-0.80) and Supplement (-1.07) versus protected 8x480 at the same 10M exposure.',
            'S1 has a meaningful target-cluster sign at 10M: GlobalPIQA mean +4.445, Entity +0.29, COMPS +0.39, and Reading mean +0.125 versus protected 8x480 10M.',
            'The GlobalPIQA 10M gain is almost as large as the protected 8x480 model\'s entire 10M-to-100M GlobalPIQA gain (+4.87), so 12x384 allocation changes the physical/commonsense profile rather than merely noise.',
            'Entity remains weak: +0.29 at 10M does not approach the leader gap. Shape alone probably does not reproduce the leader package, but it may be a useful base for curriculum/tokenizer/data factors.',
            'A same-gain extrapolation is only a route heuristic, not evidence; it suggests S1 could reach strong GlobalPIQA (~40.08) while still needing help on Entity and grammar/supplement. Full S1 100M evidence is scientifically worthwhile before closing the shape branch.'
        ]
    }
    OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — S1 12×384 vs protected 8×480 at matched 10M','',f'Evidence JSON: `{OUT}`','','| column | S1 12×384 10M | protected 8×480 10M | S1 - protected10M | protected 8×480 100M | protected gain 10M→100M |','|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['label']} | {row['s1_12x384_10m']:.2f} | {row['protected_8x480_10m']:.2f} | {row['delta_s1_minus_protected10m']:+.2f} | {row['protected_8x480_100m']:.2f} | {row['protected_gain_10m_to_100m']:+.2f} |")
    lines += ['','## Interpretation',''] + [f'- {x}' for x in payload['interpretation']]
    lines += ['','## Next execution implication','','S1 is not sufficient as a 10M endpoint, but its +4.45 GlobalPIQA mean over the protected 10M baseline is a real target-cluster signal. The next strongest execution is to run S1 shape-only to 100M (with the same legal official corpus, flat WWM, effective batch 256/micro 128, isolated fork, preferably 1M checkpoints if storage permits) alongside a proposed S2 curriculum arm. If S1 100M preserves the GlobalPIQA advantage and recovers grammar with exposure, add leader-style WWM→token and length curriculum. If it loses the target signal, prioritize curriculum/tokenizer/data reconstruction or hybrid/MNTP.']
    NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'note':str(NOTE),'target_delta':target_delta,'guard_delta':guard_delta,'key_deltas':{r['key']:r['delta_s1_minus_protected10m'] for r in rows}},indent=2))
if __name__=='__main__': main()
