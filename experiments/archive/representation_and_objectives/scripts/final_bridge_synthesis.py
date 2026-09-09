#!/usr/bin/env python3
"""Final research synthesis: raw-token interfaces vs hard-coordinate ceiling."""
from __future__ import annotations
import json, hashlib
from pathlib import Path

ROOT=Path("experiments/archive/representation_and_objectives")
RUNS={
 "vanilla_base20": ROOT/"data/rawtoken_vanilla_full20/result.json",
 "rawmem_base20": ROOT/"data/rawtoken_rawmem_full20/result.json",
 "rawmem_noevent_base20": ROOT/"data/rawtoken_noevent_full20/result.json",
 "lexmem_base20": ROOT/"data/lexmem_full20/result.json",
 "lexmem_noevent_base20": ROOT/"data/lexmem_noevent_full20/result.json",
 "vanilla_aug20": ROOT/"data/lexrec_vanilla_aug20/result.json",
 "lexrec_aug20": ROOT/"data/lexrec_aug20/result.json",
 "hardcoord_shared_base20": ROOT/"data/hardcoord_shared_full20/result.json",
 "hardcoord_shared_aug20": ROOT/"data/hardcoord_shared_aug20/result.json",
}
OUT=ROOT/"data/final_bridge_synthesis"; OUT.mkdir(parents=True,exist_ok=True)

def load(p): return json.loads(Path(p).read_text())
def get(d,path):
    x=d
    for k in path:
        if not isinstance(x,dict) or k not in x: return None
        x=x[k]
    return x

def row(name,r):
    hard = name.startswith('hardcoord')
    b=r.get('baseline',{}); wp=r.get('write_permutation',{})
    # research evaluator names selective metrics differently
    sel = b.get('selective_eval_held_recomb') or b.get('selective_updating') or {}
    psel = b.get('selective_eval_held_paraphrase') or {}
    return {
        'run': name, 'interface': 'hard_coordinate_ceiling' if hard else 'raw_token', 'arm': r.get('label') or r.get('arm'),
        'train_records': r.get('train_records'), 'eval_records': r.get('eval_records'), 'epochs': r.get('epochs'),
        'loss_final': r.get('losses',[None])[-1], 'runtime_sec': r.get('runtime_sec'),
        'held_recomb': get(b,['eval_held_recomb','accuracy']), 'held_affected': sel.get('affected_accuracy'), 'held_unaffected': sel.get('unaffected_accuracy'),
        'held_composite': sel.get('composite') if sel.get('composite') is not None else get(b,['eval_held_recomb','accuracy']),
        'held_quartet_all': sel.get('quartet_all_rate'),
        'paraphrase': get(b,['eval_held_paraphrase','accuracy']), 'paraphrase_affected': psel.get('affected_accuracy'), 'paraphrase_unaffected': psel.get('unaffected_accuracy'), 'paraphrase_quartet_all': psel.get('quartet_all_rate'),
        'multi_event': get(b,['multi_event','accuracy']),
        'wp_held': get(wp,['eval_held_recomb','accuracy']), 'wp_paraphrase': get(wp,['eval_held_paraphrase','accuracy']), 'wp_multi_event': get(wp,['multi_event','accuracy']),
    }

def main():
    rows=[]
    for k,p in RUNS.items():
        if p.exists(): rows.append(row(k,load(p)))
    for r in rows:
        if r['wp_held'] is not None and r['held_recomb'] is not None: r['wp_delta_held']=r['wp_held']-r['held_recomb']
        if r['wp_paraphrase'] is not None and r['paraphrase'] is not None: r['wp_delta_paraphrase']=r['wp_paraphrase']-r['paraphrase']
        if r['wp_multi_event'] is not None and r['multi_event'] is not None: r['wp_delta_multi_event']=r['wp_multi_event']-r['multi_event']
    by={r['run']:r for r in rows}
    comparisons={}
    def diff(a,b,keys):
        return {k:(by[a].get(k)-by[b].get(k) if a in by and b in by and by[a].get(k) is not None and by[b].get(k) is not None else None) for k in keys}
    keys=['held_recomb','held_affected','held_unaffected','held_quartet_all','paraphrase','multi_event']
    comparisons['hardcoord_base20_minus_vanilla_base20']=diff('hardcoord_shared_base20','vanilla_base20',keys)
    comparisons['hardcoord_base20_minus_best_raw_base_held']={
        'best_raw_base_held':max(by[n]['held_recomb'] for n in ['rawmem_base20','lexmem_base20','rawmem_noevent_base20','lexmem_noevent_base20']),
        'hardcoord_held':by.get('hardcoord_shared_base20',{}).get('held_recomb'),
        'delta':by.get('hardcoord_shared_base20',{}).get('held_recomb')-max(by[n]['held_recomb'] for n in ['rawmem_base20','lexmem_base20','rawmem_noevent_base20','lexmem_noevent_base20'])
    }
    comparisons['hardcoord_aug20_minus_lexrec_aug20']=diff('hardcoord_shared_aug20','lexrec_aug20',keys)
    comparisons['hardcoord_aug20_minus_vanilla_aug20']=diff('hardcoord_shared_aug20','vanilla_aug20',keys)
    payload={
        'status':'FINAL_BRIDGE_SYNTHESIS',
        'runs':{k:str(p) for k,p in RUNS.items()},
        'table':rows,
        'comparisons':comparisons,
        'decision':{
            'regex_assisted_phase2_interpretation':'integration_ceiling_only',
            'natural_bridge_authorized':False,
            'reason':'Hard-coordinate EntityMemory succeeds strongly under matched full-data settings and collapses under write permutation, while all tested raw-token interfaces remain near chance/vanilla and show negligible write-permutation dependence. The missing problem is latent occurrence-role assignment, not memory capacity or training duration.',
            'safe_next':'Design a raw-token latent entity/event/query assignment mechanism with paired counterfactual/equivariant objectives before frozen-DeBERTa EWoK transfer or fresh Strict-Small training.'
        }
    }
    outj=OUT/'final_bridge_synthesis.json'; outj.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    md=(OUT.parents[4] / 'research/documents/representation_and_objectives/data/final_bridge_synthesis/final_bridge_synthesis.md')
    lines=['# research final bridge synthesis','', '## Runs', '', '| run | interface | held | aff | unaff | quartet_all | paraphrase | multi | wp_delta_held | wp_delta_para | wp_delta_multi |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['run']} | {r['interface']} | {r.get('held_recomb')} | {r.get('held_affected')} | {r.get('held_unaffected')} | {r.get('held_quartet_all')} | {r.get('paraphrase')} | {r.get('multi_event')} | {r.get('wp_delta_held')} | {r.get('wp_delta_paraphrase')} | {r.get('wp_delta_multi_event')} |")
    lines += ['', '## Key comparisons', '', '```json', json.dumps(comparisons,indent=2), '```', '', '## Decision', '', json.dumps(payload['decision'],indent=2)]
    md.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':payload['status'],'out_json':str(outj),'out_md':str(md),'comparisons':comparisons,'decision':payload['decision']},indent=2))
if __name__=='__main__': main()
