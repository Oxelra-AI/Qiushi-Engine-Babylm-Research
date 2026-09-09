#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib, statistics
from collections import defaultdict

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
INP = ROOT/'data/s1_ablation_likelihood_probe.json'
OUT = ROOT/'data/s1_probe_pattern_analysis.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/s1_probe_pattern_analysis.md')

def short(s: str, n: int=150) -> str:
    return s if len(s)<=n else s[:n-3]+'...'

def main():
    d=json.loads(INP.read_text(encoding='utf-8'))
    recs=d['records']
    by=defaultdict(list)
    for r in recs:
        by[r['target_type']].append(r)
    summary={}
    for typ, rs in by.items():
        vals=[r['delta_true_minus_wrong_mean'] for r in rs]
        vals_no=[r['delta_true_minus_no_mean'] for r in rs]
        summary[typ]={
            'n': len(rs),
            'true_minus_wrong_mean': statistics.mean(vals),
            'true_minus_wrong_median': statistics.median(vals),
            'true_minus_wrong_positive_frac': sum(v>0 for v in vals)/len(vals),
            'true_minus_no_mean': statistics.mean(vals_no),
            'true_minus_no_positive_frac': sum(v>0 for v in vals_no)/len(vals_no),
            'top5_true_minus_wrong': [pack(x) for x in sorted(rs, key=lambda r:r['delta_true_minus_wrong_mean'], reverse=True)[:5]],
            'bottom5_true_minus_wrong': [pack(x) for x in sorted(rs, key=lambda r:r['delta_true_minus_wrong_mean'])[:5]],
        }
    # Extract model-independent rule observations from top records.
    observations=[]
    content_types=['first_s2_content','last_s2_content','first_two_s2_content_window']
    top_content=sorted([r for t in content_types for r in by.get(t,[])], key=lambda r:r['delta_true_minus_wrong_mean'], reverse=True)[:20]
    for r in top_content:
        observations.append({
            'target_type':r['target_type'], 'delta_true_minus_wrong_mean':r['delta_true_minus_wrong_mean'],
            'target_text':r['target_text'], 's1':r['s1'], 's2':r['s2'],
            'pattern_guess': guess_pattern(r)
        })
    payload={'status':'S1_PROBE_PATTERN_ANALYSIS','input':str(INP),'summary_by_target_type':summary,'top_content_observations':observations,
             'use_policy':'These protected-model scores are for mechanism discovery only. They must not be used to select training rows. Training materialization should use corpus-intrinsic rules derived from recurring textual relations.'}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research — s1-ablation probe pattern analysis (mechanism discovery only)','',f'Input: `{INP}`',f'Evidence JSON: `{OUT}`','',
           '**Policy:** use these scores only to discover model-independent relation patterns. Do not select training rows by protected-model delta under Strict-Small.','',
           '| target type | n | true-wrong mean | median | positive frac | true-no mean |','|---|---:|---:|---:|---:|---:|']
    for typ,s in summary.items():
        lines.append(f"| {typ} | {s['n']} | {s['true_minus_wrong_mean']:+.3f} | {s['true_minus_wrong_median']:+.3f} | {s['true_minus_wrong_positive_frac']:.3f} | {s['true_minus_no_mean']:+.3f} |")
    lines += ['', '## Mechanism observations from high true-vs-wrong content spans', '']
    for o in observations[:12]:
        lines.append(f"- `{o['target_text']}` ({o['target_type']}, Δ={o['delta_true_minus_wrong_mean']:+.3f}, {o['pattern_guess']}): {short(o['s1'],90)} / {short(o['s2'],110)}")
    lines += ['', '## Rule implication', '',
              '- Initial pronoun/deictic tokens are not useful XSpan targets: they show true-s1 > no-s1 but essentially no true-s1 > wrong-s1 specificity.',
              '- More promising model-independent targets are semantic content spans after the initial dependent: location/prepositional complements, action-result/object phrases, and definition/property complements following a short definition-style s1.',
              '- The next materializer should use only corpus-intrinsic patterns: same-line adjacent Simple-Wiki definition/description pairs, s2 starts with It/They/This/These/That/Those, target is the first semantic content phrase after the dependent/copula/verb, not selected by model score.']
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'target_types':list(summary)}, indent=2))

def pack(r):
    return {'target_text':r['target_text'], 'delta_true_minus_wrong_mean':r['delta_true_minus_wrong_mean'], 'delta_true_minus_no_mean':r['delta_true_minus_no_mean'], 's1':r['s1'], 's2':r['s2'], 'wrong_s1':r['wrong_s1']}

def guess_pattern(r):
    s2=r['s2'].lower()
    txt=r['target_text'].lower()
    if any(p in s2 for p in [' lives in ', ' is in ', ' are in ', ' located ', ' found in ', ' south of ', ' north of ', 'west of', 'east of']): return 'location/spatial complement'
    if any(v in s2 for v in [' accepts ', ' contains ', ' includes ', ' uses ', ' has ', ' have ', ' consists ', ' works ', ' put an end']): return 'object/affordance/result phrase'
    if any(v in s2 for v in [' is a ', ' are a ', ' is an ', ' are the ', ' means ', ' refers ']): return 'definition/property complement'
    if any(ch.isupper() for ch in r['target_text']): return 'named entity continuation'
    return 'semantic content continuation'

if __name__=='__main__': main()
