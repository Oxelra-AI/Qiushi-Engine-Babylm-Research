#!/usr/bin/env python3
"""research critical review of research v2 fixed-position prefix target census.

Reads v2 rows, computes whether same-source-near effects survive deleted/cross/block
controls, and decodes representative true/same/cross prefixes plus suffix context.
"""
from __future__ import annotations
import csv, json, pathlib, statistics, sys, random, math
from collections import defaultdict

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
ROWS=ROOT/'data/prefix_target_census_v2_fixedpos_rows.csv'
OUT=ROOT/'data/prefix_census_review.json'
NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/prefix_census_review.md')
sys.path.insert(0,str((ROOT/'scripts').resolve()))
import prefix_target_census_v2_fixedpos as v2

TH=0.05

def stable_pos(r, exp, variant, th=TH):
    return (float(r[f'wwm42_{exp}_full_minus_{variant}_fixedpos']) >= th and
            float(r[f'wwm43_{exp}_full_minus_{variant}_fixedpos']) >= th)

def stable_neg(r, exp, variant, th=TH):
    return (float(r[f'wwm42_{exp}_full_minus_{variant}_fixedpos']) <= -th and
            float(r[f'wwm43_{exp}_full_minus_{variant}_fixedpos']) <= -th)

def signed_min(r, exp, variant):
    a=float(r[f'wwm42_{exp}_full_minus_{variant}_fixedpos'])
    b=float(r[f'wwm43_{exp}_full_minus_{variant}_fixedpos'])
    return math.copysign(min(abs(a),abs(b)), a+b) if a*b>0 else 0.0

def corr(a,b):
    ma=sum(a)/len(a); mb=sum(b)/len(b)
    va=sum((x-ma)**2 for x in a); vb=sum((y-mb)**2 for y in b)
    if va==0 or vb==0: return float('nan')
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/((va*vb)**0.5)

def decode(tok, ids, maxchars=260):
    s=tok.decode(ids, skip_special_tokens=True)
    s=' '.join(s.split())
    return s[-maxchars:]

def rebuild_cases(tok):
    exact=v2.reconstruct_exact_1m(); broad=v2.broad_examples(); wf=v2.wordfreq(exact+broad); rng=random.Random(3162)
    return v2.make_cases(exact,tok,300,'exact_1m_slice',rng,wf)+v2.make_cases(broad,tok,300,'broad_official',rng,wf)

def represent(r, cases, tok):
    i=int(r['case_id']); c=cases[i]
    target=int(c['target_idx'])
    lo=max(0,target-18); hi=min(len(c['lids']), target+24)
    out={k:r[k] for k in ['case_id','case_group','source','target_token','target_surface','target_freq_bucket','local_leak_10tok','repeated_capitalized_prefix_suffix']}
    out.update({
        'stable_same_100M': signed_min(r,'100M','same_source_near'),
        'stable_deleted_100M': signed_min(r,'100M','deleted'),
        'stable_cross_100M': signed_min(r,'100M','cross_source_near'),
        'stable_block_100M': signed_min(r,'100M','block_shuffled'),
        'wwm42_same_100M': float(r['wwm42_100M_full_minus_same_source_near_fixedpos']),
        'wwm43_same_100M': float(r['wwm43_100M_full_minus_same_source_near_fixedpos']),
        'true_prefix_tail': decode(tok,c['pids'][-80:]),
        'same_prefix_tail': decode(tok,c['same_raw'][-80:]),
        'cross_prefix_tail': decode(tok,c['cross_raw'][-80:]),
        'suffix_window': tok.decode(c['lids'][lo:hi], skip_special_tokens=True),
        'target_idx_in_later': target,
    })
    return out

def main():
    rows=list(csv.DictReader(open(ROWS, newline='', encoding='utf-8')))
    summary={}
    for exp in ['40M','80M','100M']:
        same=[r for r in rows if stable_pos(r,exp,'same_source_near')]
        same_not_del=[r for r in same if not stable_pos(r,exp,'deleted')]
        same_not_cross=[r for r in same if not stable_pos(r,exp,'cross_source_near')]
        same_not_block=[r for r in same if not stable_pos(r,exp,'block_shuffled')]
        robust_all=[r for r in same if stable_pos(r,exp,'deleted') and stable_pos(r,exp,'cross_source_near') and stable_pos(r,exp,'block_shuffled')]
        vals={v:[(float(r[f'wwm42_{exp}_full_minus_{v}_fixedpos'])+float(r[f'wwm43_{exp}_full_minus_{v}_fixedpos']))/2 for r in rows]
              for v in ['same_source_near','deleted','cross_source_near','block_shuffled']}
        summary[exp]={
            'n':len(rows),
            'same_pos_ge_0p05':len(same),
            'same_pos_frac':len(same)/len(rows),
            'same_pos_but_deleted_not':len(same_not_del),
            'same_pos_but_cross_not':len(same_not_cross),
            'same_pos_but_block_not':len(same_not_block),
            'robust_same_deleted_cross_block_all_pos':len(robust_all),
            'robust_all_frac':len(robust_all)/len(rows),
            'same_neg_le_minus_0p05':sum(stable_neg(r,exp,'same_source_near') for r in rows),
            'corr_same_deleted':corr(vals['same_source_near'], vals['deleted']),
            'corr_same_cross':corr(vals['same_source_near'], vals['cross_source_near']),
            'corr_same_block':corr(vals['same_source_near'], vals['block_shuffled']),
        }
    # Representative cases using tokenizer/cases regenerated exactly.
    v2.setup_env()
    tok=v2.AutoTokenizer.from_pretrained(str(v2.CKPTS['wwm42_40M'].resolve()), use_fast=True)
    cases=rebuild_cases(tok)
    pos=[r for r in rows if stable_pos(r,'100M','same_source_near')]
    pos_not_deleted=[r for r in pos if not stable_pos(r,'100M','deleted')]
    pos_robust=[r for r in pos if stable_pos(r,'100M','deleted') and stable_pos(r,'100M','cross_source_near') and stable_pos(r,'100M','block_shuffled')]
    neg=[r for r in rows if stable_neg(r,'100M','same_source_near')]
    near=[r for r in rows if abs(signed_min(r,'100M','same_source_near'))<0.005]
    def top(rs, reverse=True, n=8):
        return sorted(rs, key=lambda r: signed_min(r,'100M','same_source_near'), reverse=reverse)[:n]
    reps={
        'positive_same_not_deleted': [represent(r,cases,tok) for r in top(pos_not_deleted, True, 8)],
        'positive_robust_all_controls': [represent(r,cases,tok) for r in top(pos_robust, True, 8)],
        'negative_same': [represent(r,cases,tok) for r in top(neg, False, 8)],
        'near_zero': [represent(r,cases,tok) for r in near[:8]],
    }
    payload={'status':'PREFIX_CENSUS_REVIEW','threshold':TH,'rows_file':str(ROWS),'summary':summary,'representatives':reps}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research critical review of research v2 prefix census','',f'JSON: `{OUT}`',f'Rows reviewed: `{ROWS}`','','## Control-overlap summary (threshold 0.05, sign-stable across seed42/seed43)','', '| exposure | same+ | same+ frac | same+ but deleted not | same+ but cross not | same+ but block not | robust all controls + | corr same/deleted | corr same/cross | corr same/block |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for exp,s in summary.items():
        lines.append(f"| {exp} | {s['same_pos_ge_0p05']} | {s['same_pos_frac']:.3f} | {s['same_pos_but_deleted_not']} | {s['same_pos_but_cross_not']} | {s['same_pos_but_block_not']} | {s['robust_same_deleted_cross_block_all_pos']} ({s['robust_all_frac']:.3f}) | {s['corr_same_deleted']:.3f} | {s['corr_same_cross']:.3f} | {s['corr_same_block']:.3f} |")
    lines += ['','## Review interpretation','', 'The fixed-position repair removes the largest v1 artifact, but the v2 positive population must not yet be treated as semantic/entity-state evidence. The same-source effect is strongly correlated with deleted, cross-source, and block-shuffled variants; a large robust-all-controls subset means many targets are helped by the true local prefix relative to every corrupted prefix, but this can still reflect local lexical/topic/history coherence rather than entity-state binding. The positives not shared by deleted/cross/block are the more relevant targets for high-precision route design and require row-level inspection.','', 'Representative decoded cases are stored in the JSON for qualitative inspection.']
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out':str(OUT),'note':str(NOTE),'summary':summary}, indent=2))
if __name__=='__main__': main()
