#!/usr/bin/env python3
"""Analyze research strict ordered target surface survivors under artifact filters."""
from __future__ import annotations
import csv, json, pathlib
from collections import Counter
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
ROWS=ROOT/'data/strict_ordered_target_surface_rows.csv'
OUT=ROOT/'data/strict_surface_filter_analysis.json'
NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/strict_surface_filter_analysis.md')

def f(x): return float(x)

def main():
    rows=list(csv.DictReader(open(ROWS,newline='',encoding='utf-8')))
    local=[r for r in rows if r.get('select_ordered_local_pass')=='1']
    surv=[r for r in local if r.get('heldout_ordered_local_pass')=='1']
    def count_subset(rs, name):
        no_pref=[r for r in rs if 'target_in_prefix' not in r['tags'].split(';')]
        no_leak=[r for r in no_pref if r['local_leak_10tok']=='0']
        no_rep_cap=[r for r in no_leak if r['repeated_capitalized_prefix_suffix']=='0']
        return {
            'name':name,'n':len(rs),
            'no_target_in_prefix':len(no_pref),
            'no_target_in_prefix_no_local_leak':len(no_leak),
            'no_target_in_prefix_no_local_leak_no_repeated_cap':len(no_rep_cap),
            'sources':dict(Counter(r['source'] for r in rs)),
            'groups':dict(Counter(r['case_group'] for r in rs)),
            'targets_top20':Counter(r['target_surface'] for r in rs).most_common(20),
            'local_distance_top':Counter(str(r.get('same_local_id_distance','')) for r in rs).most_common(20),
        }
    # Survivors with stricter clean filters
    clean=[r for r in surv if 'target_in_prefix' not in r['tags'].split(';') and r['local_leak_10tok']=='0' and r['repeated_capitalized_prefix_suffix']=='0']
    examples=[]
    for r in sorted(clean, key=lambda x:min(f(x['heldout_wwm43_100M_full_minus_same_local']), f(x['heldout_wwm43_100M_full_minus_block_shuffled'])), reverse=True)[:20]:
        examples.append({
            'case_id':r['case_id'],'group':r['case_group'],'source':r['source'],'target':r['target_surface'],
            'local_distance':r['same_local_id_distance'],'tags':r['tags'],
            'select_same_local':f(r['select_wwm42_80M_full_minus_same_local']),
            'select_block':f(r['select_wwm42_80M_full_minus_block_shuffled']),
            'heldout_same_local':f(r['heldout_wwm43_100M_full_minus_same_local']),
            'heldout_block':f(r['heldout_wwm43_100M_full_minus_block_shuffled']),
            'heldout_deleted':f(r['heldout_wwm43_100M_full_minus_deleted']),
            'heldout_cross':f(r['heldout_wwm43_100M_full_minus_cross']),
            'prefix_tail':r['prefix_text_tail'],'same_local_tail':r['same_local_text_tail'],'suffix_window':r['suffix_window']
        })
    payload={
        'status':'STRICT_SURFACE_FILTER_ANALYSIS',
        'all_cases':len(rows),
        'selected_local':count_subset(local,'selected_local'),
        'heldout_local_survivors':count_subset(surv,'heldout_local_survivors'),
        'clean_heldout_local_survivors_count':len(clean),
        'clean_heldout_local_survivors_fraction_all':len(clean)/len(rows),
        'clean_examples':examples,
    }
    OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research strict surface filter analysis','',f'JSON: `{OUT}`','',f"All cases: {len(rows)}",f"Selected local ordered surface: {len(local)}",f"Heldout local survivors: {len(surv)}",f"Heldout survivors without target-in-prefix: {payload['heldout_local_survivors']['no_target_in_prefix']}",f"Heldout survivors without target-in-prefix and local leak: {payload['heldout_local_survivors']['no_target_in_prefix_no_local_leak']}",f"Clean heldout survivors also without repeated-capitalized proxy: {len(clean)} ({len(clean)/len(rows):.4f} of all cases)",'','## Interpretation prompt','','The clean count is a conservative lower bound after removing explicit lexical repetition and simple entity-name repetition proxies. Read decoded examples before treating them as ordered-state targets.']
    NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT),'note':str(NOTE),'clean_count':len(clean),'heldout_survivors':len(surv),'selected_local':len(local)},indent=2))
if __name__=='__main__': main()
