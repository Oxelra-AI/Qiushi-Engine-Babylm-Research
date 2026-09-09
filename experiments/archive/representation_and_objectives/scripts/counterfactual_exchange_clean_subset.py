#!/usr/bin/env python3
"""research: clean-subset synthesis for counterfactual exchange scores.

Reads v3 counterfactual exchange cases and frozen checkpoint scores, applies
stricter hand-coded semantic filters without rerunning models, and reports whether
a less-contaminated subset preserves the unsaturated relation signal and the
rowblock spatial direction.
"""
from __future__ import annotations

import argparse, csv, json, math, re, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

USER_ROOT = Path('.').resolve()
DEFAULT_CASES = USER_ROOT / 'experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_v3_pilot20k/counterfactual_exchange_cases.jsonl'
DEFAULT_SCORES = USER_ROOT / 'experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_v3_pilot20k_scored/counterfactual_exchange_scores.jsonl'
DEFAULT_OUT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/counterfactual_exchange_clean_subset'
DEFAULT_NOTE = USER_ROOT / 'research/notes/representation_and_objectives/counterfactual_exchange_clean_subset.md'

ADJ_OR_ABSTRACT_BAD = {
    'high','higher','highest','low','lower','lowest','long','longer','longest','short','shorter','strong','stronger','weak','weaker','better','worse','best','worst','large','larger','small','smaller','great','greater','lesser','little','public','private','recent','past','future','present','current','previous','next','new','old','ancient','modern','local','global','national','international','major','minor','central','important','significant','precise','precisely','obvious','different','similar','same','possible','likely','unlikely','usual','common','general','specific','basic','virtual','physical','natural','human','social','political','economic','medical','legal','left','right','front','rear','back','top','bottom','middle','upper','lower','far','near','now','then','there','here','ever','never','always','often','usually','maybe','probably',
    'amount','quarter','percent','rate','level','levels','type','types','kind','sort','part','parts','side','sides','area','areas','effect','effects','field','fields','property','properties','support','progress','existence','obvious','shows','born','increase','decrease','growth','loss','problem','problems','damage','death','cause','result','results','reason','reasons','case','cases','point','points','period','time','times','year','years','month','months','week','weeks','day','days','hour','hours','minute','minutes','number','numbers','million','thousand','hundred','seconds','miles','kilometers','acres','degrees','temperature','percent','section','chapter','page','paper','study','research','survey','report'
}
FUNCTION_BAD = {'some','any','all','each','every','both','either','neither','few','several','many','much','more','less','most','least','one','ones','other','another','others','yourself','himself','herself','itself','themselves','myself','ourselves','your','yours','their','theirs','our','ours','his','her','hers','its','this','that','these','those'}
NOUNISH_GOOD_SUFFIX = ('tion','sion','ment','ness','ity','ism','ist','ers','ors','ies')
SPATIAL_TRUE = {'above','below','beneath'}
TEMPORAL_TRUE = {'before','after'}
COMPARATIVE_TRUE = {'higher','lower','larger','smaller','greater','older','younger','better','worse','faster','slower','longer','shorter','stronger','weaker'}


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def qstats(vals: Iterable[float]) -> dict[str, Any]:
    xs=[float(v) for v in vals if math.isfinite(float(v))]
    if not xs: return {'n':0}
    arr=np.asarray(xs,dtype=np.float64)
    return {'n':len(xs),'mean':float(arr.mean()),'std':float(arr.std()),'stderr':float(arr.std()/math.sqrt(len(xs))),'median':float(np.quantile(arr,.5)),'p05':float(np.quantile(arr,.05)),'p25':float(np.quantile(arr,.25)),'p75':float(np.quantile(arr,.75)),'p95':float(np.quantile(arr,.95)),'min':float(arr.min()),'max':float(arr.max()),'success_gt0':float((arr>0).mean())}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def is_bad_arg(w: str) -> bool:
    x=str(w).lower()
    if x in ADJ_OR_ABSTRACT_BAD or x in FUNCTION_BAD: return True
    if len(x) < 3 or not re.fullmatch(r"[a-z][a-z'\-]*", x): return True
    if x.endswith(('ed','ing','ly')): return True
    return False


def nounish(c: dict[str, Any], which: str) -> bool:
    cls=c.get('arg_class_'+which)
    w=c.get('arg_'+which)
    if cls in {'proper','plural','det_noun'}: return True
    if cls == 'head_noun' and str(w).endswith(NOUNISH_GOOD_SUFFIX): return True
    return False


def has_quantity_marker(c: dict[str, Any]) -> bool:
    sent=str(c.get('sentence','')).lower()
    piv=str(c.get('attested_pivot','')).split('_')[0]
    # reject spatial/comparative pivots adjacent to numerals, percentages, units, or quantity nouns.
    return bool(re.search(rf"\b{re.escape(piv)}\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d|[0-9,.]+|a\s+quarter|quarter|half|hundred|thousand|million|billion|percent|%)", sent))


def clean_family(c: dict[str, Any]) -> str | None:
    fam=c.get('family'); piv=str(c.get('attested_pivot','')).lower()
    a=str(c.get('arg_a','')).lower(); b=str(c.get('arg_b','')).lower()
    if is_bad_arg(a) or is_bad_arg(b): return None
    if not (nounish(c,'a') and nounish(c,'b')): return None
    if fam == 'spatial_vertical':
        if piv not in SPATIAL_TRUE: return None
        if has_quantity_marker(c): return None
        # Need direct spatial phrasing, not "above, showed" or list marker.
        sent=str(c.get('sentence','')).lower()
        if re.search(r"\b(above|below|beneath)\s*[,;]", sent): return None
        return 'clean_spatial_vertical'
    if fam == 'temporal_order':
        if piv not in TEMPORAL_TRUE: return None
        sent=str(c.get('sentence','')).lower()
        if 'before and after' in sent or 'after and before' in sent or re.search(r"\b(before|after)\s*[,;]", sent): return None
        # Prefer real event/object nouns; abstract temporal words already rejected.
        return 'clean_temporal_order'
    if fam == 'comparative_scalar':
        if piv not in COMPARATIVE_TRUE: return None
        if has_quantity_marker(c): return None
        return 'clean_comparative_scalar'
    if fam == 'comparative_more_less':
        if piv.startswith(('more_','less_')) and not has_quantity_marker(c): return 'clean_comparative_more_less'
    if fam == 'causal_direction':
        if piv not in {'caused','causes'}: return None
        # caused-by construction is cleaner than generic because; require nonabstract noun-like args.
        return 'clean_causal_direction'
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    byarm=defaultdict(list)
    for r in rows: byarm[r['arm']].append(r)
    summary={'n_rows':len(rows),'n_cases':len({r['case_id'] for r in rows}),'arms':{},'arm_order':{},'paired_deltas':{}}
    metrics=['true_m','target_perm_m','ctx0_margin','ctx1_margin','target_main_bias']
    for arm,xs in sorted(byarm.items()):
        asum={'n_cases':len(xs),'overall':{},'by_clean_family':{}}
        for m in metrics: asum['overall'][m]=qstats([r[m] for r in xs])
        fams=sorted({r['clean_family'] for r in xs})
        for fam in fams:
            fx=[r for r in xs if r['clean_family']==fam]
            fsum={m:qstats([r[m] for r in fx]) for m in metrics}; fsum['n_cases']=len(fx)
            asum['by_clean_family'][fam]=fsum
        summary['arms'][arm]=asum
    for m in metrics:
        means={a:summary['arms'][a]['overall'][m]['mean'] for a in byarm if summary['arms'][a]['overall'][m].get('n')}
        if means:
            summary['arm_order'][m]={'means':means,'rank_high_to_low':sorted(means,key=lambda a:means[a], reverse=True),'range':max(means.values())-min(means.values())}
            byfam={}
            for fam in sorted({r['clean_family'] for r in rows}):
                fm={a:summary['arms'][a]['by_clean_family'].get(fam,{}).get(m,{}).get('mean') for a in byarm}
                fm={k:v for k,v in fm.items() if v is not None}
                if fm: byfam[fam]={'means':fm,'rank_high_to_low':sorted(fm,key=lambda a:fm[a], reverse=True),'range':max(fm.values())-min(fm.values())}
            summary['arm_order'][m]['by_clean_family']=byfam
    # paired deltas
    bykey={(r['arm'],r['case_id']):r for r in rows}
    arms=sorted(byarm)
    for i,a in enumerate(arms):
        for b in arms[i+1:]:
            common=sorted({r['case_id'] for r in byarm[a]} & {r['case_id'] for r in byarm[b]})
            d={'n_common':len(common)}
            for m in metrics:
                d[b+'_minus_'+a+'_'+m]=qstats([bykey[(b,k)][m]-bykey[(a,k)][m] for k in common])
            summary['paired_deltas'][b+'_minus_'+a]=d
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    ap.add_argument('--scores', type=Path, default=DEFAULT_SCORES)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--note', type=Path, default=DEFAULT_NOTE)
    args=ap.parse_args()
    cases={c['case_id']:c for c in load_jsonl(args.cases)}
    scores=load_jsonl(args.scores)
    out=[]; rejects=Counter(); fams=Counter()
    for r in scores:
        c=cases.get(r['case_id'])
        if not c:
            rejects['missing_case'] += 1; continue
        cf=clean_family(c)
        if not cf:
            rejects['filter_reject'] += 1; continue
        rr=dict(r); rr['clean_family']=cf
        for k in ['sentence','context0','context1','arg_class_a','arg_class_b','broad_arg_class','certification']:
            rr[k]=c.get(k)
        out.append(rr); fams[cf] += 1
    summary=summarize(out)
    summary.update({'status':'COUNTERFACTUAL_EXCHANGE_CLEAN_SUBSET','created_utc':now(),'cases_path':str(args.cases),'scores_path':str(args.scores),'rejects':dict(rejects),'clean_family_row_counts':dict(fams)})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/'clean_subset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (args.out_dir/'clean_subset_scores.jsonl').open('w', encoding='utf-8') as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    with (args.out_dir/'clean_subset_samples.jsonl').open('w', encoding='utf-8') as f:
        seen=set()
        for r in out:
            if r['case_id'] in seen: continue
            seen.add(r['case_id'])
            f.write(json.dumps({k:r.get(k) for k in ['case_id','clean_family','sentence','context0','target0','context1','target1','arg_a','arg_b','arg_class_a','arg_class_b']}, ensure_ascii=False)+'\n')
            if len(seen)>=80: break
    note=[]
    note.append('# research — clean subset of counterfactual exchange probe\n')
    note.append(f'Created: {summary["created_utc"]}\n')
    note.append(f'Clean cases: **{summary["n_cases"]}** from scored pilot; row counts by clean family: `{summary["clean_family_row_counts"]}`.\n')
    note.append(f'Rejects: `{summary["rejects"]}`.\n')
    note.append('\n## Arm means\n')
    note.append('| metric | compact | rowblock | interleaved | rank | range |\n|---|---:|---:|---:|---|---:|\n')
    for m in ['true_m','target_perm_m','ctx0_margin','ctx1_margin','target_main_bias']:
        order=summary['arm_order'].get(m,{})
        means=order.get('means',{})
        note.append(f"| {m} | {means.get('compact')} | {means.get('rowblock')} | {means.get('interleaved')} | {' > '.join(order.get('rank_high_to_low',[]))} | {order.get('range')} |\n")
    note.append('\n## Clean-family true_m means\n')
    note.append('| family | n | compact | rowblock | interleaved | rank | range |\n|---|---:|---:|---:|---:|---|---:|\n')
    for fam, fo in sorted(summary['arm_order'].get('true_m',{}).get('by_clean_family',{}).items()):
        means=fo['means']; n=None
        for arm in ['compact','rowblock','interleaved']:
            if arm in summary['arms'] and fam in summary['arms'][arm]['by_clean_family']:
                n=summary['arms'][arm]['by_clean_family'][fam]['n_cases']; break
        note.append(f"| {fam} | {n} | {means.get('compact')} | {means.get('rowblock')} | {means.get('interleaved')} | {' > '.join(fo.get('rank_high_to_low',[]))} | {fo.get('range')} |\n")
    note.append('\nFiles: `'+str(args.out_dir/'clean_subset_summary.json')+'`, `'+str(args.out_dir/'clean_subset_scores.jsonl')+'`, `'+str(args.out_dir/'clean_subset_samples.jsonl')+'`.\n')
    args.note.parent.mkdir(parents=True, exist_ok=True); args.note.write_text(''.join(note), encoding='utf-8')
    print(json.dumps({'status':summary['status'],'n_cases':summary['n_cases'],'summary':str(args.out_dir/'clean_subset_summary.json'),'note':str(args.note)}, indent=2), flush=True)

if __name__ == '__main__':
    main()
