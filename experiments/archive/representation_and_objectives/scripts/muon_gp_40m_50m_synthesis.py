#!/usr/bin/env python3
"""research row-level synthesis of 40M and 50M Muon-switch GlobalPIQA readouts."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

ROOT=Path('.').resolve()
WS=ROOT/'experiments/archive/representation_and_objectives'
IN40=WS/'data/muon_switch_40m_globalpiqa_margin/globalpiqa_margin_reader_results.json'
IN50=WS/'data/muon_switch_50m_globalpiqa_margin/globalpiqa_margin_reader_results.json'
OUT=WS/'data/muon_gp_40m_50m_synthesis'
NOTE=(ROOT / 'research/notes/representation_and_objectives/muon_gp_40m_50m_synthesis.md')
MAP={
    'adamw':('adamw40','adamw50'),
    'continuous_muon':('continuous_muon40','continuous_muon50'),
    'muon20toadamw':('muon20toadamw40','muon20toadamw50'),
    'muon40toadamw':('muon40toadamw40','muon40toadamw50'),
}

def load(path: Path)->dict[str,Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def mode_payload(d, target, mode):
    return d['targets'][target]['modes'][mode]

def rows_by_id(d,target,mode):
    return {r['example_id']:r for r in mode_payload(d,target,mode)['rows']}

def summary(d,target,mode):
    s=mode_payload(d,target,mode)['summary']
    aw=s.get('always_wrong_subset') or {}
    return {
        'accuracy':s.get('accuracy'),
        'correct_rank_counts':s.get('correct_rank_counts'),
        'mean_top_minus_correct_all':(s.get('all_rows_margin_summary') or {}).get('mean_top_minus_correct'),
        'median_top_minus_correct_all':(s.get('all_rows_margin_summary') or {}).get('median_top_minus_correct'),
        'hard52_accuracy':aw.get('accuracy'),
        'hard52_mean_top_minus_correct':aw.get('mean_top_minus_correct'),
        'hard52_median_top_minus_correct':aw.get('median_top_minus_correct'),
        'hard52_rank_counts':aw.get('correct_rank_counts'),
        'small_wrong_le_0p25':(s.get('all_rows_margin_summary') or {}).get('small_wrong_margin_le_0p25_nats'),
        'small_wrong_le_0p50':(s.get('all_rows_margin_summary') or {}).get('small_wrong_margin_le_0p50_nats'),
        'hard52_small_wrong_le_0p25':aw.get('small_wrong_margin_le_0p25_nats'),
        'hard52_small_wrong_le_0p50':aw.get('small_wrong_margin_le_0p50_nats'),
    }

def row_changes(d40,d50,t40,t50,mode):
    r40=rows_by_id(d40,t40,mode); r50=rows_by_id(d50,t50,mode)
    ids=sorted(set(r40)&set(r50))
    gains=[]; losses=[]; same_ok=0; same_bad=0; rank_improve=0; rank_worse=0; margin_improve=0; margin_worse=0
    deltas=[]
    for uid in ids:
        a=r40[uid]; b=r50[uid]
        if (not a['correct']) and b['correct']: gains.append(uid)
        elif a['correct'] and (not b['correct']): losses.append(uid)
        elif a['correct'] and b['correct']: same_ok+=1
        else: same_bad+=1
        if int(b['correct_rank']) < int(a['correct_rank']): rank_improve+=1
        elif int(b['correct_rank']) > int(a['correct_rank']): rank_worse+=1
        md=float(a['top_minus_correct'])-float(b['top_minus_correct']) # positive means 50M less wrong/confident margin
        if md>1e-9: margin_improve+=1
        elif md<-1e-9: margin_worse+=1
        deltas.append({'example_id':uid,'mode':mode,'correct40':a['correct'],'correct50':b['correct'],'rank40':a['correct_rank'],'rank50':b['correct_rank'],'margin40':a['top_minus_correct'],'margin50':b['top_minus_correct'],'margin40_minus50':md,'choice40':a['choice'],'choice50':b['choice'],'label':a['label']})
    return {'n_common':len(ids),'gained_correct':len(gains),'lost_correct':len(losses),'same_correct':same_ok,'same_wrong':same_bad,'rank_improved':rank_improve,'rank_worse':rank_worse,'margin_improved':margin_improve,'margin_worse':margin_worse,'gain_ids':gains,'loss_ids':losses,'row_deltas':deltas}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d40=load(IN40); d50=load(IN50)
    rows=[]; all_row_deltas=[]
    synthesis={'status':'MUON_GP_40M_50M_SYNTHESIS','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'inputs':{'gp40':str(IN40),'gp50':str(IN50)},'arms':{}}
    for arm,(t40,t50) in MAP.items():
        armrec={'targets':{'40M':t40,'50M':t50},'modes':{}}
        for mode in ['parallel','nonparallel']:
            s40=summary(d40,t40,mode); s50=summary(d50,t50,mode)
            ch=row_changes(d40,d50,t40,t50,mode)
            armrec['modes'][mode]={'summary_40M':s40,'summary_50M':s50,'delta_accuracy':None if s40['accuracy'] is None or s50['accuracy'] is None else s50['accuracy']-s40['accuracy'],'delta_hard52_accuracy':None if s40['hard52_accuracy'] is None or s50['hard52_accuracy'] is None else s50['hard52_accuracy']-s40['hard52_accuracy'],'delta_hard52_mean_top_minus_correct':None if s40['hard52_mean_top_minus_correct'] is None or s50['hard52_mean_top_minus_correct'] is None else s50['hard52_mean_top_minus_correct']-s40['hard52_mean_top_minus_correct'],'row_change_summary':{k:v for k,v in ch.items() if k!='row_deltas'}}
            for r in ch['row_deltas']:
                rr={'arm':arm,**r}; all_row_deltas.append(rr)
            rows.append({'arm':arm,'mode':mode,'target40':t40,'target50':t50,'acc40':s40['accuracy'],'acc50':s50['accuracy'],'delta_acc':armrec['modes'][mode]['delta_accuracy'],'hard52_acc40':s40['hard52_accuracy'],'hard52_acc50':s50['hard52_accuracy'],'delta_hard52_acc':armrec['modes'][mode]['delta_hard52_accuracy'],'hard52_margin40':s40['hard52_mean_top_minus_correct'],'hard52_margin50':s50['hard52_mean_top_minus_correct'],'delta_hard52_margin':armrec['modes'][mode]['delta_hard52_mean_top_minus_correct'],'gained_correct':ch['gained_correct'],'lost_correct':ch['lost_correct'],'rank_improved':ch['rank_improved'],'rank_worse':ch['rank_worse'],'margin_improved':ch['margin_improved'],'margin_worse':ch['margin_worse']})
        synthesis['arms'][arm]=armrec
    # same-exposure 50M differences vs AdamW50
    adamw50_s={m:summary(d50,'adamw50',m) for m in ['parallel','nonparallel']}
    synthesis['differences_vs_adamw50']={}
    for arm,(_t40,t50) in MAP.items():
        diffs={}
        for mode in ['parallel','nonparallel']:
            s=summary(d50,t50,mode); a=adamw50_s[mode]
            diffs[mode]={'delta_accuracy':s['accuracy']-a['accuracy'],'delta_hard52_accuracy':None if s['hard52_accuracy'] is None or a['hard52_accuracy'] is None else s['hard52_accuracy']-a['hard52_accuracy'],'delta_hard52_mean_top_minus_correct':None if s['hard52_mean_top_minus_correct'] is None or a['hard52_mean_top_minus_correct'] is None else s['hard52_mean_top_minus_correct']-a['hard52_mean_top_minus_correct']}
        synthesis['differences_vs_adamw50'][arm]=diffs
    out_json=OUT/'muon_gp_40m_50m_synthesis.json'; out_json.write_text(json.dumps(synthesis,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    csv_path=OUT/'muon_gp_40m_50m_mode_deltas.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    row_csv=OUT/'muon_gp_40m_50m_row_deltas.csv'
    with row_csv.open('w',encoding='utf-8',newline='') as f:
        keys=['arm','mode','example_id','correct40','correct50','rank40','rank50','margin40','margin50','margin40_minus50','choice40','choice50','label']
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_row_deltas)
    lines=['# research — Muon-switch GlobalPIQA 40M→50M synthesis','', 'This file compares the completed 40M and 50M GlobalPIQA hard-rank readouts row-by-row. It is evaluation-only and must not be used to tune an official-example scorer.', '', '| arm | mode | acc40 | acc50 | Δacc | hard52 acc40 | hard52 acc50 | Δhard52 | hard52 margin40 | hard52 margin50 | Δmargin | gained | lost | rank↑ | rank↓ |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['arm']} | {r['mode']} | {r['acc40']:.2f} | {r['acc50']:.2f} | {r['delta_acc']:+.2f} | {'' if r['hard52_acc40'] is None else f'{r['hard52_acc40']:.2f}'} | {'' if r['hard52_acc50'] is None else f'{r['hard52_acc50']:.2f}'} | {'' if r['delta_hard52_acc'] is None else f'{r['delta_hard52_acc']:+.2f}'} | {'' if r['hard52_margin40'] is None else f'{r['hard52_margin40']:.3f}'} | {'' if r['hard52_margin50'] is None else f'{r['hard52_margin50']:.3f}'} | {'' if r['delta_hard52_margin'] is None else f'{r['delta_hard52_margin']:+.3f}'} | {r['gained_correct']} | {r['lost_correct']} | {r['rank_improved']} | {r['rank_worse']} |")
    lines += ['', 'Interpretation:', '- At 50M, neither abrupt switch arm solves the GlobalPIQA_parallel hard-rank problem. Muon20→AdamW recovers parallel from its 40M trough but remains below AdamW50 on parallel while only retaining nonparallel breadth. Muon40→AdamW after ~10M AdamW recovery remains poor on both parallel and nonparallel.', '- Hard52 accuracy remains very low and hard52 margins remain deep, so this evidence supports stopping the exact abrupt empty-moment handoff if broad/EWoK readouts show the same tradeoff.', '', f'JSON: `{out_json}`', f'Mode delta CSV: `{csv_path}`', f'Row delta CSV: `{row_csv}`']
    NOTE.parent.mkdir(parents=True,exist_ok=True); NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':synthesis['status'],'summary':str(out_json),'note':str(NOTE)},indent=2),flush=True)

if __name__=='__main__':
    main()
