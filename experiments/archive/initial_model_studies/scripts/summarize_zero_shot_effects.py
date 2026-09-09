#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
AGG=ROOT/'data/2x2_two_seed_zero_shot_aggregate.json'
OUT=(ROOT.parents[2] / 'research/notes/initial_model_studies/two_seed_zero_shot_summary.md')
KEYS=['blimp','supplement','entity_tracking','ewok','comps','GlobalPIQA_mean','Reading_mean','NLP_mean_no_superglue_aoa']
CONTRASTS=['data_effect_wwm_BminusA','data_effect_amlm_DminusC','amlm_effect_official_CminusA','amlm_effect_structured_DminusB','interaction_DC_minus_BA']
LABELS={'data_effect_wwm_BminusA':'structured data under WWM (B-A)','data_effect_amlm_DminusC':'structured data under AMLM (D-C)','amlm_effect_official_CminusA':'AMLM on official data (C-A)','amlm_effect_structured_DminusB':'AMLM on structured data (D-B)','interaction_DC_minus_BA':'interaction'}

def fmt(x): return f'{x:+.3f}'

def main():
    p=json.loads(AGG.read_text())
    lines=['# research two-seed zero-shot structured-experience summary','',f'Input: `{AGG}`','', '## Mean effects across seeds 42 and 43', '']
    lines.append('| contrast | ' + ' | '.join(KEYS) + ' |')
    lines.append('|---|' + '|'.join(['---:']*len(KEYS)) + '|')
    for c in CONTRASTS:
        row=[LABELS[c]]
        for k in KEYS:
            rec=p['summary']['mean_factorial'][c][k]
            row.append(fmt(rec['mean']))
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', '## Per-seed values for central contrasts', '']
    for c in CONTRASTS:
        lines.append(f'### {LABELS[c]}')
        lines.append('| column | seed42 | seed43 | mean |')
        lines.append('|---|---:|---:|---:|')
        for k in KEYS:
            vals=p['summary']['seed_values'][c][k]
            mean=p['summary']['mean_factorial'][c][k]['mean']
            lines.append(f'| {k} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(mean)} |')
        lines.append('')
    OUT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out':str(OUT)}, indent=2))
if __name__=='__main__': main()
