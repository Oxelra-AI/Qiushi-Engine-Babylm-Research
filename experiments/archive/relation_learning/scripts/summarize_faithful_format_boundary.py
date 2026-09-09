#!/usr/bin/env python3
"""Summarize research faithful v4, format, no-special, and embedding-only controls."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, pathlib, statistics, time
from typing import Any
ROOT=_public_path('.')
OUT=_public_path('experiments/archive/relation_learning/data/faithful_format_boundary_synthesis')
COHERENT=_public_path('experiments/archive/relation_learning/data/faithful_superglue/coherent86_alpha075/faithful_superglue_summary.json')
CHCK=_public_path('experiments/archive/relation_learning/data/faithful_superglue/chck82_scale1p75/faithful_superglue_summary.json')
FORMAT_ROOT=_public_path('experiments/archive/relation_learning/data/eval_format_replay_corrected')
TRAIN_ROOT=_public_path('experiments/archive/relation_learning/data/format_replay_corrected')
NO_SPEC_ROOT=_public_path('experiments/archive/relation_learning/data/coherent_no_special_control/eval/summary')
EMB_ROOT=_public_path('experiments/archive/relation_learning/data/special_embedding_only_control/eval/summary')
COLS=["BLiMP","Supplement","EWoK","Entity","COMPS","GlobalPIQA","Reading"]

def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)

def read(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text('\n',encoding='utf-8'); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: fields.append(k); seen.add(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def f(x):
    try: return f"{float(x):.4f}"
    except Exception: return ""

def add_summary(rows,label,seed,path,chck_scores,coh_scores,extra=None):
    obj=read(path)
    scores=obj.get('scores',{})
    row={'family':label,'seed':seed,'summary_path':rel(path),'cheap7':obj.get('cheap7'),'delta_cheap7_vs_chck82': None if obj.get('cheap7') is None else obj.get('cheap7')-chck_scores['cheap7'],'delta_cheap7_vs_coherent86': None if obj.get('cheap7') is None else obj.get('cheap7')-coh_scores['cheap7']}
    if extra: row.update(extra)
    for c in COLS:
        row[c]=scores.get(c)
        row[f'{c}_vs_chck82']=None if scores.get(c) is None else scores[c]-chck_scores[c]
        row[f'{c}_vs_coherent86']=None if scores.get(c) is None else scores[c]-coh_scores[c]
    rows.append(row)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    coh=read(COHERENT); ch=read(CHCK)
    ch_scores={**ch['cheap_scores'],'cheap7':ch['cheap7']}
    coh_scores={**coh['cheap_scores'],'cheap7':coh['cheap7']}
    rows=[]
    rows.append({'family':'faithful_reference','seed':'coherent86','summary_path':rel(COHERENT),'cheap7':coh['cheap7'],'SuperGLUE':coh['superglue_faithful_automodel'],'AoA':coh['aoa'],'Overall':coh['overall_with_measured_or_zero_aoa'],'delta_cheap7_vs_chck82':coh['cheap7']-ch['cheap7'],'delta_overall_vs_historical_421210':coh['overall_with_measured_or_zero_aoa']-42.12102470996659})
    rows.append({'family':'faithful_reference','seed':'chck82','summary_path':rel(CHCK),'cheap7':ch['cheap7'],'SuperGLUE':ch['superglue_faithful_automodel'],'AoA':ch['aoa'],'Overall':ch['overall_with_measured_or_zero_aoa']})
    for arm in ['coherent_unsplit_special','half_coherent_half_isolated','isolated_all']:
        for seed in ['98097','98098']:
            p=FORMAT_ROOT/arm/'summary'/f'step098_{arm}_seed{seed}_alpha0p75_summary.json'
            train=TRAIN_ROOT/arm/f'seed{seed}'/'summary.json'
            extra={}
            if train.exists():
                tr=read(train); last=tr.get('last_update',{})
                extra={'train_summary_path':rel(train),'train_updates':tr.get('updates'),'train_words':tr.get('total_words') or tr.get('cum_words') or last.get('cum_words'),'final_readout_kl':last.get('readout_neutral_kl')}
            add_summary(rows,arm,seed,p,ch_scores,coh_scores,extra)
    for seed in ['98097','98098']:
        p=NO_SPEC_ROOT/f'coherent_unsplit_no_special_seed{seed}_alpha0p75_summary.json'
        add_summary(rows,'coherent_unsplit_no_special_control',seed,p,ch_scores,coh_scores)
    for seed in ['98197','98198']:
        p=EMB_ROOT/f'special_embedding_only_seed{seed}_summary.json'
        add_summary(rows,'special_embedding_only_control',seed,p,ch_scores,coh_scores)
    write_csv(_public_path('experiments/archive/relation_learning/data/faithful_format_boundary_synthesis/control_summary_rows.csv'),rows)
    # family aggregate for non-reference rows
    aggs=[]
    for fam in sorted({r['family'] for r in rows if r['family']!='faithful_reference'}):
        fr=[r for r in rows if r['family']==fam]
        a={'family':fam,'n':len(fr)}
        for k in ['cheap7','delta_cheap7_vs_chck82','delta_cheap7_vs_coherent86']+[f'{c}_vs_chck82' for c in COLS]+[f'{c}_vs_coherent86' for c in COLS]:
            vals=[r.get(k) for r in fr if isinstance(r.get(k),(int,float))]
            if vals:
                a[f'mean_{k}']=sum(vals)/len(vals); a[f'min_{k}']=min(vals); a[f'max_{k}']=max(vals)
        aggs.append(a)
    write_csv(_public_path('experiments/archive/relation_learning/data/faithful_format_boundary_synthesis/family_aggregates.csv'),aggs)
    out={'status':'FAITHFUL_FORMAT_BOUNDARY_SYNTHESIS_DONE','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'faithful_v4':{'coherent86_superglue':coh['superglue_faithful_automodel'],'coherent86_overall':coh['overall_with_measured_or_zero_aoa'],'chck82_superglue':ch['superglue_faithful_automodel'],'chck82_overall':ch['overall_with_measured_or_zero_aoa']},'rows':rows,'family_aggregates':aggs,'interpretation':'Faithful coherent86 Overall is 42.02397, not historical stripped-path 42.1210. All listed format/boundary controls are mechanism controls; none is an admitted v5 component because cheap7 or important columns trade off across two seeds.','outputs':{'summary_json':rel(_public_path('experiments/archive/relation_learning/data/faithful_format_boundary_synthesis/summary.json')),'summary_md':rel(_public_path('research/documents/relation_learning/data/faithful_format_boundary_synthesis/summary.md')),'rows_csv':rel(_public_path('experiments/archive/relation_learning/data/faithful_format_boundary_synthesis/control_summary_rows.csv')),'family_csv':rel(_public_path('experiments/archive/relation_learning/data/faithful_format_boundary_synthesis/family_aggregates.csv'))}}
    (_public_path('experiments/archive/relation_learning/data/faithful_format_boundary_synthesis/summary.json')).write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research faithful/format/boundary synthesis','',out['interpretation'],'',f"Faithful coherent86: cheap7 {coh['cheap7']:.4f}, SG {coh['superglue_faithful_automodel']:.4f}, AoA {coh['aoa']:.1f}, Overall {coh['overall_with_measured_or_zero_aoa']:.6f}.",f"Faithful chck82: cheap7 {ch['cheap7']:.4f}, SG {ch['superglue_faithful_automodel']:.4f}, Overall {ch['overall_with_measured_or_zero_aoa']:.6f}.",'','## Family means versus faithful coherent86','','| family | mean cheap7 | Δcheap7 vs coherent86 | ΔBLiMP | ΔSupp | ΔEWoK | ΔEntity | ΔCOMPS | ΔGPIQA | ΔReading |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in aggs:
        lines.append(f"| {a['family']} | {f(a.get('mean_cheap7'))} | {f(a.get('mean_delta_cheap7_vs_coherent86'))} | {f(a.get('mean_BLiMP_vs_coherent86'))} | {f(a.get('mean_Supplement_vs_coherent86'))} | {f(a.get('mean_EWoK_vs_coherent86'))} | {f(a.get('mean_Entity_vs_coherent86'))} | {f(a.get('mean_COMPS_vs_coherent86'))} | {f(a.get('mean_GlobalPIQA_vs_coherent86'))} | {f(a.get('mean_Reading_vs_coherent86'))} |")
    lines += ['','## Per-seed rows','','| family | seed | cheap7 | Δcheap7 vs coherent86 | ΔBLiMP | ΔSupp | ΔEWoK | ΔEntity | ΔCOMPS | ΔGPIQA | ΔReading |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['family']=='faithful_reference': continue
        lines.append(f"| {r['family']} | {r['seed']} | {f(r.get('cheap7'))} | {f(r.get('delta_cheap7_vs_coherent86'))} | {f(r.get('BLiMP_vs_coherent86'))} | {f(r.get('Supplement_vs_coherent86'))} | {f(r.get('EWoK_vs_coherent86'))} | {f(r.get('Entity_vs_coherent86'))} | {f(r.get('COMPS_vs_coherent86'))} | {f(r.get('GlobalPIQA_vs_coherent86'))} | {f(r.get('Reading_vs_coherent86'))} |")
    lines += ['',f"Full JSON: `{rel(_public_path('experiments/archive/relation_learning/data/faithful_format_boundary_synthesis/summary.json'))}`"]
    (_public_path('research/documents/relation_learning/data/faithful_format_boundary_synthesis/summary.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':out['status'],'summary_json':rel(_public_path('experiments/archive/relation_learning/data/faithful_format_boundary_synthesis/summary.json')),'summary_md':rel(_public_path('research/documents/relation_learning/data/faithful_format_boundary_synthesis/summary.md')),'faithful_v4':out['faithful_v4'],'family_aggregates':aggs},indent=2),flush=True)
if __name__=='__main__': main()
