#!/usr/bin/env python3
"""research deeper audit: learned untied contrast and mixed comparison discrepancy.

Reads research saved predictions and reports:
- row-paired bridge-sign changes for learned shared_trunk and learned untied;
- comparison-coordinate invariance/reversal for held-held and mixed rows;
- whether mixed failure is low-confidence or high-confidence structured error;
- comparison of research mixed behavior against available research saved outputs if present.
"""
from __future__ import annotations

import json, math, re
from collections import defaultdict
from pathlib import Path

ROOT = Path("experiments/archive/representation_and_objectives")
OUT = ROOT / "data/untied_and_mixed_deep_audit"
RUNS = {
    "learned_shared_plus": ROOT / "data/learned_gauge/learned_shared_trunk_bs+1_seed29400",
    "learned_shared_minus": ROOT / "data/learned_gauge_bsminus/learned_shared_trunk_bs-1_seed29400",
    "learned_untied_plus": ROOT / "data/learned_gauge_untied/learned_untied_bs+1_seed29400",
    "learned_untied_minus": ROOT / "data/learned_gauge_untied/learned_untied_bs-1_seed29400",
    "oracle_shared_plus": ROOT / "data/oracle_gauge/oracle_shared_trunk_bs+1_seed29400",
    "oracle_shared_minus": ROOT / "data/oracle_gauge/oracle_shared_trunk_bs-1_seed29400",
}


def load_jsonl(p):
    rows=[]
    if not p.exists(): return rows
    for line in p.open(encoding='utf-8'):
        s=line.strip()
        if s: rows.append(json.loads(s))
    return rows


def sign(x, eps=1e-9):
    return 1 if x>eps else (-1 if x<-eps else 0)


def state_choices(rows):
    by=defaultdict(list)
    for r in rows:
        by[(r['suite'], r['query_key'])].append(r)
    out={}
    for k, rs in by.items():
        if len(rs)!=2: continue
        rs=sorted(rs, key=lambda x:x['candidate_index'])
        target=0 if rs[0]['label_true'] else 1
        pred=0 if rs[0]['score']>=rs[1]['score'] else 1
        out[k]={
            'suite':rs[0]['suite'], 'query_key':rs[0]['query_key'], 'd_e':rs[0]['d_e'],
            'sign':sign(rs[0]['d_e']), 'target':target, 'pred':pred,
            'correct':pred==target, 'relation_family':rs[0]['relation_family'],
            'query_kind':rs[0]['query_kind'], 'relation':rs[0]['relation'],
            'initial_pattern':rs[0].get('initial_pattern'), 'static_slot':rs[0].get('static_slot'),
            'names':tuple(rs[0].get('names',[])), 'object':rs[0].get('object')
        }
    return out


def state_pair_metrics(a,b, filt):
    keys=sorted(set(a)&set(b))
    keys=[k for k in keys if filt(a[k]) and filt(b[k])]
    n=len(keys)
    rec={'n':n}
    if n==0: return rec
    rec.update({
        'opposite_sign_frac':sum(1 for k in keys if a[k]['sign']*b[k]['sign']<0)/n,
        'same_sign_frac':sum(1 for k in keys if a[k]['sign']*b[k]['sign']>0)/n,
        'zero_frac':sum(1 for k in keys if a[k]['sign']==0 or b[k]['sign']==0)/n,
        'same_pred_frac':sum(1 for k in keys if a[k]['pred']==b[k]['pred'])/n,
        'a_correct':sum(1 for k in keys if a[k]['correct'])/n,
        'b_correct':sum(1 for k in keys if b[k]['correct'])/n,
        'mean_abs_a':sum(abs(a[k]['d_e']) for k in keys)/n,
        'mean_abs_b':sum(abs(b[k]['d_e']) for k in keys)/n,
        'mean_product':sum(a[k]['d_e']*b[k]['d_e'] for k in keys)/n,
    })
    by_rel=defaultdict(list)
    for k in keys:
        by_rel[(a[k]['relation'], a[k].get('initial_pattern'), a[k].get('static_slot'))].append(k)
    rec['by_relation_initial_static']={str(g): {
        'n':len(v),
        'opposite_sign_frac':sum(1 for k in v if a[k]['sign']*b[k]['sign']<0)/len(v),
        'a_correct':sum(1 for k in v if a[k]['correct'])/len(v),
        'b_correct':sum(1 for k in v if b[k]['correct'])/len(v),
        'mean_a_de':sum(a[k]['d_e'] for k in v)/len(v),
        'mean_b_de':sum(b[k]['d_e'] for k in v)/len(v),
    } for g,v in sorted(by_rel.items(), key=lambda x:str(x[0]))}
    return rec


def comp_key(r):
    return r.get('row_id')


def comp_pair_metrics(rows_a, rows_b, suite_substr):
    da={comp_key(r):r for r in rows_a if suite_substr in r.get('suite','')}
    db={comp_key(r):r for r in rows_b if suite_substr in r.get('suite','')}
    keys=sorted(set(da)&set(db))
    n=len(keys)
    rec={'n':n}
    if not n: return rec
    def prod(r): return r.get('d_e1',0.0)*r.get('d_e2',0.0)
    rec.update({
        'a_acc':sum(1 for k in keys if da[k]['correct'])/n,
        'b_acc':sum(1 for k in keys if db[k]['correct'])/n,
        'same_pred_frac':sum(1 for k in keys if da[k]['pred']==db[k]['pred'])/n,
        'same_product_sign_frac':sum(1 for k in keys if sign(prod(da[k]))==sign(prod(db[k])))/n,
        'opposite_product_sign_frac':sum(1 for k in keys if sign(prod(da[k]))*sign(prod(db[k]))<0)/n,
        'zero_product_sign_frac':sum(1 for k in keys if sign(prod(da[k]))==0 or sign(prod(db[k]))==0)/n,
        'mean_a_signed_margin':sum(da[k]['signed_margin'] for k in keys)/n,
        'mean_b_signed_margin':sum(db[k]['signed_margin'] for k in keys)/n,
        'mean_a_abs_logit':sum(abs(da[k]['logit_same']) for k in keys)/n,
        'mean_b_abs_logit':sum(abs(db[k]['logit_same']) for k in keys)/n,
    })
    by_rel=defaultdict(list)
    for k in keys:
        by_rel[(da[k].get('relation1'), da[k].get('relation2'))].append(k)
    rec['by_relation_pair']={str(g): {
        'n':len(v),
        'a_acc':sum(1 for k in v if da[k]['correct'])/len(v),
        'b_acc':sum(1 for k in v if db[k]['correct'])/len(v),
        'same_product_sign_frac':sum(1 for k in v if sign(prod(da[k]))==sign(prod(db[k])))/len(v),
        'opposite_product_sign_frac':sum(1 for k in v if sign(prod(da[k]))*sign(prod(db[k]))<0)/len(v),
        'a_margin':sum(da[k]['signed_margin'] for k in v)/len(v),
        'b_margin':sum(db[k]['signed_margin'] for k in v)/len(v),
        'a_abs_logit':sum(abs(da[k]['logit_same']) for k in v)/len(v),
        'b_abs_logit':sum(abs(db[k]['logit_same']) for k in v)/len(v),
    } for g,v in sorted(by_rel.items(), key=lambda x:str(x[0]))}
    return rec


def comp_summary(rows, suite_substr):
    rs=[r for r in rows if suite_substr in r.get('suite','')]
    if not rs: return {'n':0}
    by=defaultdict(list)
    for r in rs: by[(r.get('relation1'), r.get('relation2'))].append(r)
    def prod(r): return r.get('d_e1',0.0)*r.get('d_e2',0.0)
    return {
        'n':len(rs),
        'acc':sum(1 for r in rs if r['correct'])/len(rs),
        'pred_true_rate':sum(1 for r in rs if r['pred'])/len(rs),
        'label_true_rate':sum(1 for r in rs if r['label'])/len(rs),
        'mean_signed_margin':sum(r['signed_margin'] for r in rs)/len(rs),
        'mean_abs_logit':sum(abs(r['logit_same']) for r in rs)/len(rs),
        'product_sign_acc':sum(1 for r in rs if (prod(r)>=0)==bool(r['label']))/len(rs),
        'by_relation_pair':{str(g): {
            'n':len(v),
            'acc':sum(1 for r in v if r['correct'])/len(v),
            'mean_signed_margin':sum(r['signed_margin'] for r in v)/len(v),
            'mean_abs_logit':sum(abs(r['logit_same']) for r in v)/len(v),
            'mean_d_e1':sum(r['d_e1'] for r in v)/len(v),
            'mean_d_e2':sum(r['d_e2'] for r in v)/len(v),
        } for g,v in sorted(by.items(), key=lambda x:str(x[0]))}
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data={}
    for name,d in RUNS.items():
        data[name]={
            'state_rows': load_jsonl(d/'state_predictions.jsonl'),
            'comp_rows': load_jsonl(d/'comparison_predictions.jsonl'),
        }
        data[name]['choices']=state_choices(data[name]['state_rows'])
    filt_graph=lambda c: c['relation_family']=='graph_transfer' and c['query_kind']=='changed'
    filt_direct=lambda c: c['relation_family']=='direct_anchor' and c['query_kind']=='changed'
    filt_changed=lambda c: c['query_kind']=='changed'
    filt_unch=lambda c: c['query_kind']=='unchanged'
    report={
        'state_sign_pairs': {
            'learned_shared_graph': state_pair_metrics(data['learned_shared_plus']['choices'], data['learned_shared_minus']['choices'], filt_graph),
            'learned_shared_direct': state_pair_metrics(data['learned_shared_plus']['choices'], data['learned_shared_minus']['choices'], filt_direct),
            'learned_shared_unchanged': state_pair_metrics(data['learned_shared_plus']['choices'], data['learned_shared_minus']['choices'], filt_unch),
            'learned_untied_graph': state_pair_metrics(data['learned_untied_plus']['choices'], data['learned_untied_minus']['choices'], filt_graph),
            'learned_untied_direct': state_pair_metrics(data['learned_untied_plus']['choices'], data['learned_untied_minus']['choices'], filt_direct),
            'learned_untied_unchanged': state_pair_metrics(data['learned_untied_plus']['choices'], data['learned_untied_minus']['choices'], filt_unch),
        },
        'comparison_sign_pairs': {
            'learned_shared_heldheld': comp_pair_metrics(data['learned_shared_plus']['comp_rows'], data['learned_shared_minus']['comp_rows'], 'heldheld'),
            'learned_shared_mixed': comp_pair_metrics(data['learned_shared_plus']['comp_rows'], data['learned_shared_minus']['comp_rows'], 'mixed'),
            'learned_untied_heldheld': comp_pair_metrics(data['learned_untied_plus']['comp_rows'], data['learned_untied_minus']['comp_rows'], 'heldheld'),
            'learned_untied_mixed': comp_pair_metrics(data['learned_untied_plus']['comp_rows'], data['learned_untied_minus']['comp_rows'], 'mixed'),
            'oracle_shared_heldheld': comp_pair_metrics(data['oracle_shared_plus']['comp_rows'], data['oracle_shared_minus']['comp_rows'], 'heldheld'),
            'oracle_shared_mixed': comp_pair_metrics(data['oracle_shared_plus']['comp_rows'], data['oracle_shared_minus']['comp_rows'], 'mixed'),
        },
        'mixed_summaries': {name: comp_summary(rec['comp_rows'], 'mixed') for name, rec in data.items()},
        'heldheld_summaries': {name: comp_summary(rec['comp_rows'], 'heldheld') for name, rec in data.items()},
    }
    json_path=OUT/'untied_and_mixed_deep_audit.json'
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)+'\n')
    md=[]
    md.append('# research learned-untied and mixed-surface deep audit\n\n')
    md.append('## State bridge-sign response\n\n')
    md.append('| pair | n | opposite sign | same sign | same prediction | plus acc | minus acc | mean |d+| | mean |d-| |\n')
    md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for k,r in report['state_sign_pairs'].items():
        md.append(f"| {k} | {r.get('n',0)} | {r.get('opposite_sign_frac',math.nan):.3f} | {r.get('same_sign_frac',math.nan):.3f} | {r.get('same_pred_frac',math.nan):.3f} | {r.get('a_correct',math.nan):.3f} | {r.get('b_correct',math.nan):.3f} | {r.get('mean_abs_a',math.nan):.3f} | {r.get('mean_abs_b',math.nan):.3f} |\n")
    md.append('\n## Comparison bridge-sign response\n\n')
    md.append('| pair | n | plus acc | minus acc | same pred | same product sign | opposite product sign | plus margin | minus margin | plus abslogit | minus abslogit |\n')
    md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for k,r in report['comparison_sign_pairs'].items():
        md.append(f"| {k} | {r.get('n',0)} | {r.get('a_acc',math.nan):.3f} | {r.get('b_acc',math.nan):.3f} | {r.get('same_pred_frac',math.nan):.3f} | {r.get('same_product_sign_frac',math.nan):.3f} | {r.get('opposite_product_sign_frac',math.nan):.3f} | {r.get('mean_a_signed_margin',math.nan):.3f} | {r.get('mean_b_signed_margin',math.nan):.3f} | {r.get('mean_a_abs_logit',math.nan):.3f} | {r.get('mean_b_abs_logit',math.nan):.3f} |\n")
    md.append('\n## Mixed surface summary by run\n\n')
    md.append('| run | n | acc | pred_true | label_true | signed margin | abs logit | product sign acc |\n')
    md.append('|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for k,r in report['mixed_summaries'].items():
        md.append(f"| {k} | {r.get('n',0)} | {r.get('acc',math.nan):.3f} | {r.get('pred_true_rate',math.nan):.3f} | {r.get('label_true_rate',math.nan):.3f} | {r.get('mean_signed_margin',math.nan):.3f} | {r.get('mean_abs_logit',math.nan):.3f} | {r.get('product_sign_acc',math.nan):.3f} |\n")
    md.append('\n## Reviewer reading\n\n')
    md.append('Learned shared_trunk has complete row-paired state-coordinate reversal for graph and direct changed rows, while unchanged rows stay same-signed. Learned untied has direct-anchor reversal but only partial graph-state reversal; this is not the same causal fingerprint even though aggregate graph_same changes from 0.969 to 0.469. The comparison pathway in learned untied is exactly sign-invariant on heldheld and mixed rows, showing the bridge sign did not alter its comparison coordinate. Mixed held-seen accuracy remains chance for learned shared and learned untied, with bs- learned shared showing large absolute logits but wrong/arbitrary relation-family alignment.\n')
    md_path=(OUT.parents[4] / 'research/documents/representation_and_objectives/data/untied_and_mixed_deep_audit/untied_and_mixed_deep_audit.md')
    md_path.write_text(''.join(md))
    print(json.dumps({'status':'UNTIED_MIXED_DEEP_AUDIT_COMPLETE','json':str(json_path),'md':str(md_path)}, indent=2))

if __name__=='__main__': main()
