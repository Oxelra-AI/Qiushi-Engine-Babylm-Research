#!/usr/bin/env python3
"""Count stricter candidate subpools inside the research active relation pairs."""
from __future__ import annotations
import csv, json, time, textwrap
from collections import defaultdict, Counter
from pathlib import Path

ROOT=Path('.').resolve(); WS=ROOT/'experiments/archive/representation_and_objectives'
AUDIT_CSV=WS/'data/relation_pair_pool_semantic_audit/relation_pair_pool_semantic_audit_rows.csv'
PAIR_POOL=WS/'data/relation_active_pair_funnel/active_relation_pair_pool.jsonl'
OUT=WS/'data/relation_pair_subpool_counts'; NOTE=(ROOT / 'research/notes/representation_and_objectives/relation_pair_subpool_counts.md')

def load_rows():
    with AUDIT_CSV.open('r',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['normalized_levenshtein','char_jaccard','context_word_jaccard','local_window_jaccard','cross_context_target_leaks']:
            r[k]=float(r[k])
        for k in ['either_generic','both_generic','maybe_antonym','same_prefix3','same_suffix3']:
            r[k]=str(r[k]).lower()=='true'
    return rows

def load_pairs_by_id():
    d={}
    with PAIR_POOL.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip():
                p=json.loads(line); d[p['pair_id']]=p
    return d

def excerpt(s,n=190):
    s=' '.join(str(s).split())
    return s[:n] + ('...' if len(s)>n else '')

def count_by_cat(rows, pred):
    c=Counter()
    for r in rows:
        if pred(r): c[r['category']]+=1
    return dict(c)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=load_rows(); pair_by_id=load_pairs_by_id()
    tests={
        'all': lambda r: True,
        'non_generic': lambda r: not r['either_generic'],
        'local_jaccard_ge_0p10': lambda r: r['local_window_jaccard']>=0.10,
        'local_jaccard_ge_0p15': lambda r: r['local_window_jaccard']>=0.15,
        'local_jaccard_ge_0p20': lambda r: r['local_window_jaccard']>=0.20,
        'context_jaccard_ge_0p14': lambda r: r['context_word_jaccard']>=0.14,
        'char_jaccard_ge_0p40': lambda r: r['char_jaccard']>=0.40,
        'char_jaccard_ge_0p50': lambda r: r['char_jaccard']>=0.50,
        'edit_distance_le_0p60': lambda r: r['normalized_levenshtein']<=0.60,
        'same_suffix3': lambda r: r['same_suffix3'],
        'antonym_hint': lambda r: r['maybe_antonym'],
        'non_generic_and_local_ge_0p10': lambda r: (not r['either_generic']) and r['local_window_jaccard']>=0.10,
        'non_generic_and_context_ge_0p14': lambda r: (not r['either_generic']) and r['context_word_jaccard']>=0.14,
        'non_generic_and_char_ge_0p40': lambda r: (not r['either_generic']) and r['char_jaccard']>=0.40,
        'candidate_competitive_broad': lambda r: (not r['either_generic']) and (r['local_window_jaccard']>=0.10 or r['context_word_jaccard']>=0.14 or r['char_jaccard']>=0.40 or r['maybe_antonym']),
        'candidate_competitive_strict': lambda r: (not r['either_generic']) and (r['local_window_jaccard']>=0.15 or r['char_jaccard']>=0.50 or r['maybe_antonym']),
    }
    summary={'status':'RELATION_PAIR_SUBPOOL_COUNTS','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'inputs':{'audit_csv':str(AUDIT_CSV),'pair_pool':str(PAIR_POOL)},'total_pairs':len(rows),'tests':{}}
    for name,pred in tests.items():
        xs=[r for r in rows if pred(r)]
        summary['tests'][name]={'n':len(xs),'fraction':len(xs)/len(rows) if rows else None,'by_category':count_by_cat(rows,pred),'target_top10':Counter([r['target_a'] for r in xs]+[r['target_b'] for r in xs]).most_common(10),'pivot_top10':Counter([r['pivot_a'] for r in xs]+[r['pivot_b'] for r in xs]).most_common(10)}
    # Save inspectable samples from candidate subsets, especially comparative.
    samples={}
    sample_specs={
        'candidate_competitive_broad': tests['candidate_competitive_broad'],
        'candidate_competitive_strict': tests['candidate_competitive_strict'],
        'comparative_all': lambda r: r['category']=='comparative',
        'high_local_overlap': tests['local_jaccard_ge_0p15'],
        'high_target_similarity': tests['char_jaccard_ge_0p50'],
        'antonym_hint': tests['antonym_hint'],
    }
    for name,pred in sample_specs.items():
        out=[]
        for r in rows:
            if pred(r):
                p=pair_by_id.get(r['pair_id'],{})
                out.append({k:r[k] for k in ['pair_id','category','target_a','target_b','pivot_a','pivot_b','either_generic','maybe_antonym','char_jaccard','normalized_levenshtein','context_word_jaccard','local_window_jaccard']} | {'text_a_excerpt':excerpt(p.get('text_a','')),'text_b_excerpt':excerpt(p.get('text_b',''))})
                if len(out)>=40: break
        samples[name]=out
    summary['samples']=samples
    json_path=OUT/'relation_pair_subpool_counts.json'; json_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — stricter subpool counts inside active relation pairs','',f'Total pairs: **{len(rows)}**','', '| subset | n | frac | causal | spatial | temporal | negation | physical | comparative | top targets |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,d in summary['tests'].items():
        bc=d['by_category']; tops=', '.join([f'{t}:{c}' for t,c in d['target_top10'][:5]])
        lines.append(f"| {name} | {d['n']} | {d['fraction']:.3f} | {bc.get('causal_connector',0)} | {bc.get('spatial',0)} | {bc.get('temporal',0)} | {bc.get('negation',0)} | {bc.get('physical_change',0)} | {bc.get('comparative',0)} | {tops} |")
    lines += ['', 'Interpretation:', '- If full no-update four-cell margins are huge and arm-separation does not survive permutation controls, the broad pool should not become a training objective.', '- Any rebuilt pool should increase true competition of cross-targets: higher local-slot overlap, explicit relation-opposition pairs, or repeated target-alternative sets, while preserving legality and avoiding official-example wording.', '', f'JSON: `{json_path}`']
    NOTE.parent.mkdir(parents=True,exist_ok=True); NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'json':str(json_path),'note':str(NOTE)},indent=2),flush=True)
if __name__=='__main__': main()
