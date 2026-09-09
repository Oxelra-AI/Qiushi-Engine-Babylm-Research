#!/usr/bin/env python3
"""research: stratify held-out recombination pairs and export examples.

The main stratification is whether the updated entity
(the B entity in a DISTRACTOR pair) is already mentioned in the source sentence.
If yes, the pair tests within-source identity discrimination after a later update.
If no, it partly tests uptake of an entity introduced by the update sentence.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, pathlib, re, time
from collections import defaultdict

ROOT0=_public_path('experiments/archive/relation_learning/scripts/binding_pair_strata_and_export.py')
ROOT = _PUBLIC_ROOT
DATA=ROOT/'experiments/archive/relation_learning/data/recombination_rows'
OUT=ROOT/'experiments/archive/relation_learning/data/binding_pair_strata_and_export'
OUT.mkdir(parents=True, exist_ok=True)

def read_jsonl(p):
    with p.open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def norm(s): return re.sub(r'\s+',' ',str(s or '').strip().lower())
def entity_in_text(entity,text):
    e=norm(entity); t=norm(text)
    if not e: return False
    # substring with loose punctuation/space equivalence; robust enough for names.
    return e in t

def rel(p):
    try: return str(p.relative_to(ROOT))
    except Exception: return str(p)

def main():
    held=read_jsonl(DATA/'recombination_heldout.jsonl')
    pairs=read_jsonl(DATA/'binding_pairs_heldout.jsonl')
    rows={r['row_id']:r for r in held}
    out_rows=[]
    strata=defaultdict(int)
    source_stats=defaultdict(int)
    for bp in pairs:
        a=rows[bp['row_a_id']]; b=rows[bp['row_b_id']]
        assert a['pair_id']==b['pair_id']==bp['pair_id']
        source=a['source_sentence']
        updated_entity=b['query_entity']
        target_entity=a['query_entity']
        updated_in_source=entity_in_text(updated_entity, source)
        target_in_source=entity_in_text(target_entity, source)
        same_context = a['source_sentence']==b['source_sentence'] and a['update_sentence']==b['update_sentence']
        rec={
            'pair_id':bp['pair_id'],
            'target_entity':target_entity,
            'updated_entity':updated_entity,
            'target_entity_in_source':target_in_source,
            'updated_entity_in_source':updated_in_source,
            'both_entities_in_source':target_in_source and updated_in_source,
            'same_source_update_context':same_context,
            'source_state':a['source_state'],
            'new_state':b['new_state'],
            'source_sentence':source,
            'update_sentence':a['update_sentence'],
            'row_a_id':a['row_id'],
            'row_b_id':b['row_id'],
        }
        out_rows.append(rec)
        strata['updated_entity_in_source' if updated_in_source else 'updated_entity_not_in_source'] += 1
        strata['target_entity_in_source' if target_in_source else 'target_entity_not_in_source'] += 1
        strata['both_entities_in_source' if (target_in_source and updated_in_source) else 'not_both_entities_in_source'] += 1
        if not same_context: strata['different_context_error'] += 1
        # crude source-prefix identifier from row ids; detailed corpus source was not retained in recombination rows.
        source_stats['total'] += 1
    fields=list(out_rows[0].keys())
    with (OUT/'heldout_binding_pair_source_presence.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)
    # Export first ten diverse examples with exact schema.
    examples=[]
    # prefer both types if available
    for want in [True, False]:
        for rec in out_rows:
            if rec['updated_entity_in_source'] is want and len(examples)<10:
                a=rows[rec['row_a_id']]; b=rows[rec['row_b_id']]
                examples.append({'pair':rec,'row_a':a,'row_b':b})
            if len(examples)>=5 and want is True: break
            if len(examples)>=10: break
    if len(examples)<10:
        seen={ex['pair']['pair_id'] for ex in examples}
        for rec in out_rows:
            if rec['pair_id'] in seen: continue
            examples.append({'pair':rec,'row_a':rows[rec['row_a_id']],'row_b':rows[rec['row_b_id']]})
            if len(examples)>=10: break
    export={
        'status':'A01_RECOMBINATION_EXPORT',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'schema':{
            'train_rows':'JSONL records with row_id,pair_id,split,packet_type,query_entity,answer_text,answer_kind,role,pair_half,source_sentence,update_sentence,source_state,new_state,target_entity,updated_entity,context_text,answer_char_start,answer_char_end.',
            'heldout_rows':'same schema as train rows; held-out contains 400 DISTRACTOR rows (200 pairs) plus 200 UPDATED single rows.',
            'binding_pairs':'JSONL pair records with pair_id,row_a_id,row_b_id,entity_a,entity_b,answer_a,answer_b. row_a is unchanged_entity/source_state; row_b is updated_entity/new_state. The source_sentence and update_sentence are identical across the two rows of each pair; only query_entity and final answer differ.',
            'contract':'Full-phrase candidate comparison should score source_state and new_state in the same final frame The relevant state of {query_entity} is ___. Joint success requires row_a source_state>new_state and row_b new_state>source_state.'
        },
        'paths':{
            'train':rel(DATA/'recombination_train.jsonl'),
            'heldout':rel(DATA/'recombination_heldout.jsonl'),
            'binding_pairs':rel(DATA/'binding_pairs_heldout.jsonl'),
            'strata_csv':rel(OUT/'heldout_binding_pair_source_presence.csv'),
        },
        'strata_counts':dict(strata),
        'n_examples':len(examples),
        'examples':examples,
    }
    (OUT/'a01_recombination_schema_and_examples.json').write_text(json.dumps(export,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    summary={k:v for k,v in export.items() if k!='examples'}
    summary['examples_path']=rel(OUT/'a01_recombination_schema_and_examples.json')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
