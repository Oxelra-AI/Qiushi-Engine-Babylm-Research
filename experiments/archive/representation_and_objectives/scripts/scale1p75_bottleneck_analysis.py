#!/usr/bin/env python3
"""research: score-bottleneck arithmetic and route implications for scale1.75."""
from __future__ import annotations
import json, pathlib, statistics, time

ROOT=pathlib.Path('.')
OUT=ROOT/'experiments/archive/representation_and_objectives/data/scale1p75_score_bottleneck'
NOTE=ROOT/'research/notes/representation_and_objectives/scale1p75_bottleneck_and_next_route.md'
OUT.mkdir(parents=True, exist_ok=True); NOTE.parent.mkdir(parents=True, exist_ok=True)

def j(p): return json.load(open(p, encoding='utf-8'))

p80=j(ROOT/'experiments/archive/representation_and_objectives/data/scale1p75_score_collation/scale1p75_80M_collated_score.json')
p100=j(ROOT/'experiments/archive/representation_and_objectives/data/scale1p75_score_collation/scale1p75_100M_collated_score.json')
# 100M reference from the hardened evaluation script.
reference_100={
    'Overall':41.257770896404615,
    'BLiMP':65.8707181799453,
    'Supplement':61.16566092036889,
    'EWoK':50.39323748109589,
    'Entity':27.400833994026197,
    'COMPS':52.00834536316919,
    'SuperGLUE':70.27986764740969,
    'GlobalPIQA':36.0631067961165,
    'Reading':8.13816768550987,
    'AoA':0.0,
}
keys=['BLiMP','Supplement','EWoK','Entity','COMPS','SuperGLUE','GlobalPIQA','Reading','AoA']
cheap=['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA','Reading']

def gap_report(scores, threshold=41.8):
    fixed_sum=sum(float(v) for k,v in scores.items() if k!='SuperGLUE' and v is not None)
    req_sg=9*threshold-fixed_sum if all(scores.get(k) is not None for k in keys if k!='SuperGLUE') else None
    overall=sum(float(scores[k]) for k in keys)/9 if all(scores.get(k) is not None for k in keys) else None
    return {'overall':overall,'margin_vs_41p8': None if overall is None else overall-threshold,
            'points_needed_in_one_column': None if overall is None else max(0,(threshold-overall)*9),
            'required_superglue_for_41p8':req_sg}

s80=p80['official_overall']['scores']; s100=p100['official_overall']['scores']
fixed_delta_100_vs_80={k:(None if s80.get(k) is None or s100.get(k) is None else float(s100[k])-float(s80[k])) for k in keys if k!='SuperGLUE'}
fixed_sum_delta_100_vs_80=sum(v for v in fixed_delta_100_vs_80.values() if v is not None)
sg_details80=p80['tasks']['SuperGLUE'].get('superglue_primary_metric_details', [])
sg_task_scores80={r['task']:{'metric':r['metric'],'score':r['score']} for r in sg_details80}
# Read matched hard/EWoK summaries from research.
ewok=j(ROOT/'experiments/archive/representation_and_objectives/data/scale1p75_matched_ewok_interaction/scale1p75_matched_ewok_summary.json')
gp100=j(ROOT/'experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface/scale1p75_100m_globalpiqa_summary.json')
res={
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'scale1p75_80M': {'scores':s80,'score_report':gap_report(s80),'superglue_task_primary_scores':sg_task_scores80},
    'scale1p75_100M_current': {'scores':s100,'score_report':gap_report(s100)},
    'fixed_column_delta_100M_minus_80M_without_superglue': fixed_delta_100_vs_80,
    'fixed_sum_delta_100M_minus_80M_without_superglue': fixed_sum_delta_100_vs_80,
    'reference_100M_ref': reference_100,
    'scale1p75_100M_if_SuperGLUE_equals_step35_100M': (sum(float(s100[k]) for k in keys if k!='SuperGLUE')+reference_100['SuperGLUE'])/9,
    'scale1p75_100M_superglue_needed_above_step35_100M': gap_report(s100)['required_superglue_for_41p8']-reference_100['SuperGLUE'],
    'matched_ewok': {
        'scale1p75_80M_accuracy': ewok['targets']['scale1p75_80M']['summary']['accuracy'],
        'legal16k_80M_accuracy': ewok['targets']['legal16k_80M']['summary']['accuracy'],
        'scale1p75_100M_accuracy': ewok['targets']['scale1p75_100M']['summary']['accuracy'],
        'legal16k_100M_accuracy': ewok['targets']['legal16k_100M']['summary']['accuracy'],
        'scale1p75_80M_stable_failure': ewok['targets']['scale1p75_80M']['summary']['stable_failure'],
        'legal16k_80M_stable_failure': ewok['targets']['legal16k_80M']['summary']['stable_failure'],
        'scale1p75_100M_stable_failure': ewok['targets']['scale1p75_100M']['summary']['stable_failure'],
        'legal16k_100M_stable_failure': ewok['targets']['legal16k_100M']['summary']['stable_failure'],
    },
    'globalpiqa_100M_matched_warning_path': 'experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface/scale1p75_100m_globalpiqa_summary.json',
    'interpretation': {
        'score': '80M scale1.75 is a real near-miss but below the public 41.80 frontier; 100M cannot be judged until SuperGLUE completes and needs a much stronger SuperGLUE than either 80M scale1.75 or research 100M.',
        'mechanism': 'Matched legal16k EWoK and GlobalPIQA hard-surface readouts do not support scale1.75 as a binding repair. It is broad-score evidence with EWoK/hard-binding tradeoffs.',
        'next_if_100M_fails': 'Do not spend on exposure-only scale1.75 variants. Build a representation/signal change that keeps the scale1.75 broad BLiMP/Supp/Entity improvements while restoring matched EWoK and GlobalPIQA hard-row margins.'
    }
}
(OUT/'scale1p75_bottleneck_analysis.json').write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
lines=[]
lines.append('# research scale1.75 bottleneck and next-route analysis\n\n')
lines.append('The complete 80M scale1.75 score is a near miss, not SOTA. It uses the corrected SuperGLUE primary metrics (MRPC/QQP F1; others accuracy) and AoA=0.0.\n\n')
lines.append(f"- 80M Overall: **{res['scale1p75_80M']['score_report']['overall']:.12f}**, margin vs 41.80 = **{res['scale1p75_80M']['score_report']['margin_vs_41p8']:+.12f}**. Equivalent one-column shortfall: {res['scale1p75_80M']['score_report']['points_needed_in_one_column']:.6f}.\n")
lines.append(f"- 80M SuperGLUE: **{s80['SuperGLUE']:.12f}**, threshold required at fixed other columns: {res['scale1p75_80M']['score_report']['required_superglue_for_41p8']:.12f}.\n")
lines.append(f"- 100M fixed columns require SuperGLUE **{res['scale1p75_100M_current']['score_report']['required_superglue_for_41p8']:.12f}** to exceed 41.80; if it merely matches research 100M SuperGLUE ({reference_100['SuperGLUE']:.6f}), projected Overall is {res['scale1p75_100M_if_SuperGLUE_equals_step35_100M']:.6f}.\n")
lines.append(f"- Moving from 80M to 100M changes non-SuperGLUE column sum by {fixed_sum_delta_100_vs_80:+.6f}: {fixed_delta_100_vs_80}.\n\n")
lines.append('Mechanism interpretation remains negative for binding: matched legal16k EWoK degrades at both 80M and 100M, and research matched GlobalPIQA hard-surface evidence showed scale1.75 100M worse than the matched research legal16k base on parallel accuracy and hard52 margin.\n\n')
lines.append('If 100M SuperGLUE does not exceed the high 71.405 requirement, the route should not continue by exposure-only or simple amplitude variants. The next useful work is a representation or learning-signal change that preserves the broad score gains of the adapter trajectory while repairing matched EWoK and GlobalPIQA hard rows.\n')
NOTE.write_text(''.join(lines),encoding='utf-8')
print(json.dumps({'status':'BOTTLENECK_ANALYSIS_DONE','json':str(OUT/'scale1p75_bottleneck_analysis.json'),'note':str(NOTE),'80M_margin':res['scale1p75_80M']['score_report']['margin_vs_41p8'],'100M_required_sg':res['scale1p75_100M_current']['score_report']['required_superglue_for_41p8']},indent=2))
