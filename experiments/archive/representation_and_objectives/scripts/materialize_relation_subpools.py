#!/usr/bin/env python3
"""Materialize filtered relation-pair subpools from research model-free audit."""
from __future__ import annotations
import csv, json, time
from collections import Counter
from pathlib import Path

ROOT=Path('.').resolve(); WS=ROOT/'experiments/archive/representation_and_objectives'
PAIR_POOL=WS/'data/relation_active_pair_funnel/active_relation_pair_pool.jsonl'
AUDIT_CSV=WS/'data/relation_pair_pool_semantic_audit/relation_pair_pool_semantic_audit_rows.csv'
OUT=WS/'data/relation_filtered_pair_pools'; NOTE=(ROOT / 'research/notes/representation_and_objectives/relation_filtered_pair_pools.md')

def load_audit():
    d={}
    with AUDIT_CSV.open('r',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rr=dict(r)
            for k in ['normalized_levenshtein','char_jaccard','context_word_jaccard','local_window_jaccard','cross_context_target_leaks']:
                rr[k]=float(rr[k])
            for k in ['either_generic','both_generic','maybe_antonym','same_prefix3','same_suffix3']:
                rr[k]=str(rr[k]).lower()=='true'
            d[rr['pair_id']]=rr
    return d

def load_pairs():
    pairs=[]
    with PAIR_POOL.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): pairs.append(json.loads(line))
    return pairs

def main():
    OUT.mkdir(parents=True,exist_ok=True); audit=load_audit(); pairs=load_pairs()
    preds={
        'non_generic_local10': lambda r: (not r['either_generic']) and r['local_window_jaccard']>=0.10,
        'non_generic_local15': lambda r: (not r['either_generic']) and r['local_window_jaccard']>=0.15,
        'non_generic_local20': lambda r: (not r['either_generic']) and r['local_window_jaccard']>=0.20,
        'candidate_competitive_strict': lambda r: (not r['either_generic']) and (r['local_window_jaccard']>=0.15 or r['char_jaccard']>=0.50 or r['maybe_antonym']),
        'candidate_competitive_broad': lambda r: (not r['either_generic']) and (r['local_window_jaccard']>=0.10 or r['context_word_jaccard']>=0.14 or r['char_jaccard']>=0.40 or r['maybe_antonym']),
        'comparative_all': lambda r: r['category']=='comparative',
    }
    summary={'status':'RELATION_FILTERED_PAIR_POOLS','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'inputs':{'pair_pool':str(PAIR_POOL),'audit_csv':str(AUDIT_CSV)},'subpools':{}}
    pair_by_id={p['pair_id']:p for p in pairs}
    for name,pred in preds.items():
        ids=[pid for pid,r in audit.items() if pred(r)]
        out_path=OUT/f'{name}.jsonl'
        with out_path.open('w',encoding='utf-8') as f:
            for pid in ids:
                f.write(json.dumps(pair_by_id[pid],ensure_ascii=False)+'\n')
        bc=Counter(pair_by_id[pid]['category'] for pid in ids)
        tc=Counter([pair_by_id[pid]['target_norm_a'] for pid in ids]+[pair_by_id[pid]['target_norm_b'] for pid in ids])
        summary['subpools'][name]={'n':len(ids),'path':str(out_path),'by_category':dict(bc),'top_targets':tc.most_common(15)}
    json_path=OUT/'filtered_pair_pools_summary.json'; json_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — filtered relation-pair pools','', 'These files preserve stricter model-free subpools for no-update calibration or later training design. They are not training inputs by themselves.', '', '| subpool | n | causal | spatial | temporal | negation | physical | comparative | path |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,d in summary['subpools'].items():
        bc=d['by_category']; lines.append(f"| {name} | {d['n']} | {bc.get('causal_connector',0)} | {bc.get('spatial',0)} | {bc.get('temporal',0)} | {bc.get('negation',0)} | {bc.get('physical_change',0)} | {bc.get('comparative',0)} | `{d['path']}` |")
    lines += ['', f'JSON: `{json_path}`']
    NOTE.parent.mkdir(parents=True,exist_ok=True); NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'json':str(json_path),'note':str(NOTE)},indent=2),flush=True)
if __name__=='__main__': main()
