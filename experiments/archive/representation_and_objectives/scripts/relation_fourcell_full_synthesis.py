#!/usr/bin/env python3
"""Synthesize research full four-cell calibration with semantic subpool labels.

Reads the full all-pair score file produced by relation_fourcell_calibration.py
and the model-free semantic audit, then asks whether any subpool's arm separation
matches the known relation tradeoff more than same-stratum permutation nulls.
"""
from __future__ import annotations
import csv, json, math, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT=Path('.').resolve(); WS=ROOT/'experiments/archive/representation_and_objectives'
FULL_DIR=WS/'data/relation_fourcell_calibration'
FULL_SUMMARY=FULL_DIR/'fourcell_calibration_summary.json'
FULL_SCORES=FULL_DIR/'fourcell_pair_scores.jsonl'
AUDIT_CSV=WS/'data/relation_pair_pool_semantic_audit/relation_pair_pool_semantic_audit_rows.csv'
OUT=WS/'data/relation_fourcell_synthesis'; NOTE=(ROOT / 'research/notes/representation_and_objectives/relation_fourcell_synthesis.md')

KNOWN_ORDER={'gp_parallel':'rowblock > interleaved > compact or rowblock strongest; compact weakest on parallel movement', 'ewok_aggregate':'interleaved > rowblock≈compact aggregate; four-cell rowblock less stable-failure than compact'}

def load_json(path: Path)->dict[str,Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def load_scores(path: Path):
    rows=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def load_audit(path: Path):
    d={}
    with path.open('r',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rr=dict(r)
            for k in ['normalized_levenshtein','char_jaccard','context_word_jaccard','local_window_jaccard','cross_context_target_leaks']:
                rr[k]=float(rr[k])
            for k in ['either_generic','both_generic','maybe_antonym','same_prefix3','same_suffix3']:
                rr[k]=str(rr[k]).lower()=='true'
            d[rr['pair_id']]=rr
    return d

def qstats(vals):
    xs=sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs: return {'n':0}
    def q(p):
        if len(xs)==1: return xs[0]
        idx=p*(len(xs)-1); lo=int(math.floor(idx)); hi=int(math.ceil(idx))
        return xs[lo] if lo==hi else xs[lo]*(hi-idx)+xs[hi]*(idx-lo)
    mean=statistics.fmean(xs); sd=statistics.pstdev(xs)
    return {'n':len(xs),'mean':mean,'std':sd,'stderr':sd/math.sqrt(len(xs)),'median':q(.5),'p05':q(.05),'p95':q(.95),'success_gt0':sum(v>0 for v in xs)/len(xs)}

def pearson(xs,ys):
    if len(xs)<3 or len(xs)!=len(ys): return None
    mx=statistics.fmean(xs); my=statistics.fmean(ys)
    vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    if vx<=1e-12 or vy<=1e-12: return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(vx*vy)

def subset_pred(name, audit):
    if name=='all': return lambda r: True
    if name=='non_generic': return lambda r: not audit[r['pair_id']]['either_generic']
    if name=='candidate_broad': return lambda r: (not audit[r['pair_id']]['either_generic']) and (audit[r['pair_id']]['local_window_jaccard']>=0.10 or audit[r['pair_id']]['context_word_jaccard']>=0.14 or audit[r['pair_id']]['char_jaccard']>=0.40 or audit[r['pair_id']]['maybe_antonym'])
    if name=='candidate_strict': return lambda r: (not audit[r['pair_id']]['either_generic']) and (audit[r['pair_id']]['local_window_jaccard']>=0.15 or audit[r['pair_id']]['char_jaccard']>=0.50 or audit[r['pair_id']]['maybe_antonym'])
    if name=='non_generic_local10': return lambda r: (not audit[r['pair_id']]['either_generic']) and audit[r['pair_id']]['local_window_jaccard']>=0.10
    if name=='non_generic_local15': return lambda r: (not audit[r['pair_id']]['either_generic']) and audit[r['pair_id']]['local_window_jaccard']>=0.15
    if name=='non_generic_char50': return lambda r: (not audit[r['pair_id']]['either_generic']) and audit[r['pair_id']]['char_jaccard']>=0.50
    if name=='comparative_all': return lambda r: r['category']=='comparative'
    raise KeyError(name)

def arm_order(means):
    return sorted(means, key=lambda a: means[a], reverse=True)

def summarize_subset(rows, audit, name):
    pred=subset_pred(name,audit)
    rs=[r for r in rows if r['pair_id'] in audit and pred(r)]
    arms=sorted(set(r['arm'] for r in rs)); byarm={a:[r for r in rs if r['arm']==a] for a in arms}
    out={'n_rows':len(rs),'n_pairs':len(set(r['pair_id'] for r in rs)),'arms':{},'arm_order':{},'relation_deltas':{},'by_category':{}}
    for a,ar in byarm.items():
        out['arms'][a]={m:qstats([r[m] for r in ar]) for m in ['true_m','tperm_m','cperm_m','true_m_a','true_m_b']}
    for m in ['true_m','tperm_m','cperm_m']:
        means={a:out['arms'][a][m]['mean'] for a in arms if out['arms'][a][m].get('n')}
        out['arm_order'][m]={'means':means,'rank_high_to_low':arm_order(means),'range':(max(means.values())-min(means.values()) if means else None)}
    d={a:{r['pair_id']:r for r in byarm.get(a,[])} for a in arms}
    if all(a in d for a in ['compact','rowblock','interleaved']):
        common=sorted(set(d['compact'])&set(d['rowblock'])&set(d['interleaved']))
        out['common_pairs']=len(common)
        for hi,lo in [('rowblock','compact'),('interleaved','compact'),('rowblock','interleaved')]:
            key=f'{hi}_minus_{lo}'
            out['relation_deltas'][key]={m:qstats([d[hi][pid][m]-d[lo][pid][m] for pid in common]) for m in ['true_m','tperm_m','cperm_m']}
            out['relation_deltas'][key]['true_minus_perm_range_excess']={
                'vs_tperm_absrange': abs(out['relation_deltas'][key]['true_m']['mean'])-abs(out['relation_deltas'][key]['tperm_m']['mean']),
                'vs_cperm_absrange': abs(out['relation_deltas'][key]['true_m']['mean'])-abs(out['relation_deltas'][key]['cperm_m']['mean']),
            }
        out['correlations']={}
        for a,b in [('compact','rowblock'),('compact','interleaved'),('rowblock','interleaved')]:
            out['correlations'][f'{a}__{b}']={m:pearson([d[a][pid][m] for pid in common],[d[b][pid][m] for pid in common]) for m in ['true_m','tperm_m','cperm_m']}
    for cat in sorted(set(r['category'] for r in rs)):
        cr=[r for r in rs if r['category']==cat]
        out['by_category'][cat]=summarize_subset_category(cr)
    return out

def summarize_subset_category(rs):
    arms=sorted(set(r['arm'] for r in rs)); byarm={a:[r for r in rs if r['arm']==a] for a in arms}
    out={'n_pairs':len(set(r['pair_id'] for r in rs)),'arm_order':{},'arms':{}}
    for a,ar in byarm.items(): out['arms'][a]={m:qstats([r[m] for r in ar]) for m in ['true_m','tperm_m','cperm_m']}
    for m in ['true_m','tperm_m','cperm_m']:
        means={a:out['arms'][a][m]['mean'] for a in arms if out['arms'][a][m].get('n')}
        out['arm_order'][m]={'means':means,'rank_high_to_low':arm_order(means),'range':(max(means.values())-min(means.values()) if means else None)}
    return out

def main():
    if not FULL_SUMMARY.exists() or not FULL_SCORES.exists():
        raise SystemExit(f'Full calibration not ready: {FULL_SUMMARY} / {FULL_SCORES}')
    OUT.mkdir(parents=True,exist_ok=True)
    full=load_json(FULL_SUMMARY); rows=load_scores(FULL_SCORES); audit=load_audit(AUDIT_CSV)
    subsets=['all','non_generic','candidate_broad','candidate_strict','non_generic_local10','non_generic_local15','non_generic_char50','comparative_all']
    syn={'status':'RELATION_FOURCELL_SYNTHESIS','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'inputs':{'full_summary':str(FULL_SUMMARY),'full_scores':str(FULL_SCORES),'audit_csv':str(AUDIT_CSV)},'known_tradeoff_reference':KNOWN_ORDER,'subsets':{}}
    for s in subsets: syn['subsets'][s]=summarize_subset(rows,audit,s)
    # High-level assessment fields for quick routing.
    assessments=[]
    for s,rec in syn['subsets'].items():
        if rec.get('common_pairs',0)<=0: continue
        true_order=rec['arm_order']['true_m']['rank_high_to_low']
        trange=rec['arm_order']['true_m']['range']; pr=max(rec['arm_order']['tperm_m']['range'] or 0, rec['arm_order']['cperm_m']['range'] or 0)
        assessments.append({'subset':s,'n_pairs':rec['n_pairs'],'true_order':true_order,'true_range':trange,'max_perm_range':pr,'true_range_exceeds_perm':None if trange is None else trange>pr})
    syn['assessment_compact']=assessments
    json_path=OUT/'relation_fourcell_full_synthesis.json'; json_path.write_text(json.dumps(syn,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    # CSV compact table
    csv_path=OUT/'relation_fourcell_subset_table.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        fields=['subset','n_pairs','true_order','true_range','tperm_range','cperm_range','rowblock_minus_compact_true','rowblock_minus_compact_tperm','rowblock_minus_compact_cperm','interleaved_minus_compact_true','interleaved_minus_compact_tperm','interleaved_minus_compact_cperm']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for s,rec in syn['subsets'].items():
            rd=rec.get('relation_deltas',{})
            w.writerow({'subset':s,'n_pairs':rec.get('n_pairs'),'true_order':' > '.join(rec['arm_order']['true_m']['rank_high_to_low']),'true_range':rec['arm_order']['true_m']['range'],'tperm_range':rec['arm_order']['tperm_m']['range'],'cperm_range':rec['arm_order']['cperm_m']['range'],'rowblock_minus_compact_true':rd.get('rowblock_minus_compact',{}).get('true_m',{}).get('mean'),'rowblock_minus_compact_tperm':rd.get('rowblock_minus_compact',{}).get('tperm_m',{}).get('mean'),'rowblock_minus_compact_cperm':rd.get('rowblock_minus_compact',{}).get('cperm_m',{}).get('mean'),'interleaved_minus_compact_true':rd.get('interleaved_minus_compact',{}).get('true_m',{}).get('mean'),'interleaved_minus_compact_tperm':rd.get('interleaved_minus_compact',{}).get('tperm_m',{}).get('mean'),'interleaved_minus_compact_cperm':rd.get('interleaved_minus_compact',{}).get('cperm_m',{}).get('mean')})
    lines=['# research — full relation four-cell synthesis','', 'This synthesis joins the full no-update four-cell scores with model-free semantic subpool labels. The scientific question is whether the pair pool tracks the known FW relation tradeoff more than matched target/context permutation nulls.', '', '| subset | n pairs | true order | true range | target-perm range | context-perm range | rowblock-compact true | rowblock-compact target-null | rowblock-compact context-null | interleaved-compact true |', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for s,rec in syn['subsets'].items():
        rd=rec.get('relation_deltas',{}); rbc=rd.get('rowblock_minus_compact',{}); ic=rd.get('interleaved_minus_compact',{})
        lines.append(f"| {s} | {rec.get('n_pairs')} | {' > '.join(rec['arm_order']['true_m']['rank_high_to_low'])} | {rec['arm_order']['true_m']['range']} | {rec['arm_order']['tperm_m']['range']} | {rec['arm_order']['cperm_m']['range']} | {rbc.get('true_m',{}).get('mean')} | {rbc.get('tperm_m',{}).get('mean')} | {rbc.get('cperm_m',{}).get('mean')} | {ic.get('true_m',{}).get('mean')} |")
    lines += ['', 'Family tables and exact paired deltas are in the JSON.', '', f'JSON: `{json_path}`', f'CSV: `{csv_path}`']
    NOTE.parent.mkdir(parents=True,exist_ok=True); NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':syn['status'],'json':str(json_path),'note':str(NOTE)},indent=2),flush=True)
if __name__=='__main__': main()
