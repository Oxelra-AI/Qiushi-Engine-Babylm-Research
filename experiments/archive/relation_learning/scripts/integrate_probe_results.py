#!/usr/bin/env python3
"""Integrate research probe results with research official Entity cuts."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, pathlib, statistics, math
from collections import defaultdict

ROOT=_public_path('experiments/archive/relation_learning/scripts/integrate_probe_results.py')
ROOT = _PUBLIC_ROOT
WS=_public_path('experiments/archive/relation_learning')
OUT=_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation')
research=_public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis')
NOTE=_public_path('research/notes/relation_learning/probe_results_interpretation.md')
JSON_OUT=_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/integrated_probe_summary.json')

def rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            yield r

def f(x):
    try: return float(x)
    except Exception: return float('nan')

def mean(xs):
    xs=[x for x in xs if math.isfinite(x)]
    return statistics.mean(xs) if xs else float('nan')

def late_arm_summary(path, group, keys):
    d=defaultdict(list)
    for r in rows(path):
        if r.get('group')==group:
            d[(r['seed'], r['role'])].append(r)
    out={}
    for k, vals in d.items():
        out[k]={key: mean([f(v[key]) for v in vals]) for key in keys}
        out[k]['n']=int(vals[0]['n']) if vals else 0
    return out

copy=late_arm_summary(_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/copy_summary.csv'),'ALL',['mean_gain','mean_a_nll','mean_b_nll'])
copy_span1=late_arm_summary(_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/copy_summary.csv'),'span_1',['mean_gain'])
copy_span4=late_arm_summary(_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/copy_summary.csv'),'span_4',['mean_gain'])
rew_all=late_arm_summary(_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/rewrite_summary.csv'),'ALL',['mean_gain','mean_a_nll','mean_b_nll'])
rew_non=late_arm_summary(_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/rewrite_summary.csv'),'token_nonoverlap',['mean_gain'])
rew_ov=late_arm_summary(_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/rewrite_summary.csv'),'token_overlap',['mean_gain'])
ent_ab_all=late_arm_summary(_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/entity_ablation_summary.csv'),'ALL',['mean_margin_full','mean_effect_no_initial_qbox','mean_effect_no_last_relevant_update','mean_effect_no_all_relevant_updates'])
ent_ab_ge3=late_arm_summary(_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/entity_ablation_summary.csv'),'rel_ge3',['mean_margin_full','mean_effect_no_initial_qbox','mean_effect_no_last_relevant_update','mean_effect_no_all_relevant_updates'])
# official accuracy from research by relevant updates, late over checkpoints
ent_off=defaultdict(list)
for r in rows(_public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_relevant_update_summary.csv')):
    if r['group'] in {'rel_updates_0','rel_ge1','rel_updates_2','rel_updates_3','rel_updates_4','rel_updates_5'}:
        ent_off[(str(r['seed']), r['arm'], r['group'])].append(f(r['accuracy_pct']))
# map arm V/C/R to role already
ent_off_m={(seed,arm,grp):mean(vals) for (seed,arm,grp),vals in ent_off.items()}
# rel>=2 / rel>=3 from augmented summary not present; compute from prediction augmented rows quickly via summary file groups rel_ge? research includes rel_ge1 only. Use relevant-update weighted means from individual rel groups.
counts={}
for r in rows(_public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_relevant_update_summary.csv')):
    if r['group'].startswith('rel_updates_') and r['arm']=='V' and r['checkpoint']=='chck_80M':
        counts[int(r['group'].split('_')[-1])]=int(r['n'])

def weighted_rel(seed, arm, rels):
    num=den=0.0
    for rel in rels:
        vals=[]; n=None
        for r in rows(_public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_relevant_update_summary.csv')):
            if str(r['seed'])==str(seed) and r['arm']==arm and r['group']==f'rel_updates_{rel}':
                vals.append(f(r['accuracy_pct'])); n=int(r['n'])
        if vals and n:
            num += mean(vals)*n; den += n
    return num/den if den else float('nan')

integrated=[]
for seed in ['43022','43122']:
    for role in ['V','C','R']:
        rec={'seed':seed,'role':role}
        for prefix, table in [('copy',copy),('copy_span1',copy_span1),('copy_span4',copy_span4),('rewrite',rew_all),('rewrite_nonoverlap',rew_non),('rewrite_overlap',rew_ov),('ablation_all',ent_ab_all),('ablation_ge3',ent_ab_ge3)]:
            for k,v in table.get((seed,role),{}).items():
                rec[f'{prefix}_{k}']=v
        for grp in ['rel_updates_0','rel_updates_2','rel_updates_3','rel_updates_4','rel_updates_5']:
            rec[f'official_{grp}_acc']=ent_off_m.get((seed,role,grp),float('nan'))
        rec['official_rel_ge2_acc']=weighted_rel(seed, role, [2,3,4,5])
        rec['official_rel_ge3_acc']=weighted_rel(seed, role, [3,4,5])
        integrated.append(rec)

def order_str(seed, key):
    vals=[(r['role'], r[key]) for r in integrated if r['seed']==seed and math.isfinite(r.get(key,float('nan')))]
    vals=sorted(vals, key=lambda x:x[1], reverse=True)
    return ' > '.join([f'{a}({b:.3f})' for a,b in vals])

lines=[]
lines.append('# research integrated probe interpretation')
lines.append('')
lines.append('This note integrates the research held-out copy/rewrite probes and Entity cue ablations with the research official Entity relevant-update split. All values below are late means over 80M/90M/100M checkpoints for the two existing seeds; no seed43222 result has been read here.')
lines.append('')
lines.append('## Arm-level late orderings')
lines.append('')
lines.append('| seed | held-out copy gain | held-out rewrite gain | rewrite nonoverlap gain | Entity rel0 official acc | Entity rel>=2 official acc | stale-item full margin |')
lines.append('|---:|---|---|---|---|---|---|')
for seed in ['43022','43122']:
    lines.append(f"| {seed} | {order_str(seed,'copy_mean_gain')} | {order_str(seed,'rewrite_mean_gain')} | {order_str(seed,'rewrite_nonoverlap_mean_gain')} | {order_str(seed,'official_rel_updates_0_acc')} | {order_str(seed,'official_rel_ge2_acc')} | {order_str(seed,'ablation_all_mean_margin_full')} |")
lines.append('')
lines.append('## Cue-ablation arm effects on stale-non-gold update items')
lines.append('')
lines.append('`no_initial` and `no_last_update` are changes in gold-over-stale margin after removing the queried-box initial clause or the last relevant update sentence. Positive no_initial means removing the stale initial clause helps; negative no_last_update means removing the update hurts.')
lines.append('')
lines.append('| seed | role | full margin | no_initial effect | no_last_update effect | no_all_updates effect | rel>=3 full margin | rel>=3 no_initial | rel>=3 no_last |')
lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|')
for r in integrated:
    lines.append(f"| {r['seed']} | {r['role']} | {r.get('ablation_all_mean_margin_full',float('nan')):+.3f} | {r.get('ablation_all_mean_effect_no_initial_qbox',float('nan')):+.3f} | {r.get('ablation_all_mean_effect_no_last_relevant_update',float('nan')):+.3f} | {r.get('ablation_all_mean_effect_no_all_relevant_updates',float('nan')):+.3f} | {r.get('ablation_ge3_mean_margin_full',float('nan')):+.3f} | {r.get('ablation_ge3_mean_effect_no_initial_qbox',float('nan')):+.3f} | {r.get('ablation_ge3_mean_effect_no_last_relevant_update',float('nan')):+.3f} |")
lines.append('')
lines.append('## Scientific interpretation')
lines.append('')
lines.append('- The held-out natural-copy probe closes the main copy-side ambiguity from research: REPEAT has the largest copy gain on text rows not used to train any arm. VIEW also exceeds CLEAN, so the most accurate statement is not that exact repetition alone creates copying, but that both nonbaseline source+companion interventions improve natural context copying and exact copied companions improve it most. This distinction matters for the principle: the copy competence is not merely memorizing the MAX packets, but the trade-off with state updating is strongest for REPEAT.')
lines.append('- The held-out rewrite-conditioning probe supplies the missing positive quantity for VIEW. On unselected compact pairs, VIEW has the largest source-conditioned rewrite-token gain in both seeds, and the gap is largest on non-overlap rewrite tokens where exact token copying cannot explain the benefit. The ordering V > C > R says varied restatement buys content-conditioned use of an earlier span, while exact copied fragments are worst for that quantity under the same budget.')
lines.append('- The Entity cue ablation is mechanistically sharper than an aggregate loss fit. On stale-non-gold update items, VIEW has the largest full-context gold-over-stale margin and REPEAT the smallest. Removing the queried-box initial clause helps REPEAT relative to VIEW, consistent with stale initial over-anchoring in REPEAT; removing the last relevant update hurts VIEW more than REPEAT, consistent with VIEW relying more on update evidence. Removing all relevant updates hurts REPEAT even more, which is not clean evidence for update reading; it leaves the stale clause as the dominant remaining cue and should be read as a vulnerability of REPEAT under contradictory evidence rather than as VIEW being less update-dependent.')
lines.append('- Together with research, the supported candidate principle has become a pair of checkpoint-measured competences bought by different experience structures under the same finite budget: exact natural recurrence most strengthens copy/use-earlier-span behavior, while nonidentical restatement most strengthens content-conditioned use of an earlier span and improves multi-update state discrimination. The positive VIEW side is now directly measured, but the current evidence remains two-seed until seed43222 checkpoints complete.')
NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
JSON_OUT.write_text(json.dumps({'integrated_rows': integrated, 'note': str(NOTE.relative_to(ROOT))}, indent=2), encoding='utf-8')
print(json.dumps({'status':'INTEGRATED_SUMMARY_DONE','note':str(NOTE.relative_to(ROOT)),'json':str(JSON_OUT.relative_to(ROOT))}, indent=2))
