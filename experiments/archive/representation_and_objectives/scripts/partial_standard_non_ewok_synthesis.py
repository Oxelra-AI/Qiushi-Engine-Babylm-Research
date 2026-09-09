#!/usr/bin/env python3
"""Partial research synthesis using completed non-EWoK readouts only.

No model scoring is run here. It compares the completed standard-legacy
Supplement/Entity and GlobalPIQA files against research reference/control/treatment
while the isolated GPU EWoK readout runs separately.
"""
from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
STD_SUPP = WS / 'data/legacy_80m_readouts/supplement_entity/per_target/standard_legacy_80m.json'
STD_GP = WS / 'data/legacy_80m_readouts/globalpiqa_margin/standard_legacy_80m_margins.json'
STD_GP_ROOT = WS / 'data/legacy_80m_readouts/globalpiqa_margin'
PREV = WS / 'data/pvdm_full_readout_synthesis/pvdm_full_readout_synthesis.json'
PREV_GP_ROOT = WS / 'data/pvdm_80m_readouts/globalpiqa_margin'
ANATOMY = WS / 'data/globalpiqa_parallel_anatomy/globalpiqa_parallel_anatomy.json'
OUT = WS / 'data/partial_standard_non_ewok_synthesis'
NOTE = (ROOT / 'research/notes/representation_and_objectives/121_partial_standard_non_ewok_synthesis.md')
PREV_NAMES = {'reference':'compact_80m_reference','control':'pvdm_control_80m','treatment':'pvdm_treatment_80m'}


def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
def load(p: Path) -> Any: return json.loads(p.read_text(encoding='utf-8'))
def boolv(x: Any) -> bool:
    if isinstance(x, bool): return x
    if isinstance(x, str): return x.strip().lower() == 'true'
    return bool(x)


def hard_ids() -> set[str]:
    d = load(ANATOMY)
    return {r['example_id'] for r in d.get('agreement',{}).get('parallel',{}).get('rows',[]) if r.get('n_ok') == 0}


def std_metrics() -> dict[str, Any]:
    s = load(STD_SUPP).get('tasks', {})
    g = load(STD_GP)
    rec = {'target':'standard_legacy_80m', 'Supplement': s.get('Supplement',{}).get('score'), 'Entity': s.get('Entity',{}).get('score')}
    for mode in ['parallel','nonparallel']:
        sm = g['modes'][mode]['summary']
        rec[f'GlobalPIQA_{mode}'] = sm['accuracy']
        rec[f'GlobalPIQA_{mode}_rank_counts'] = sm['correct_rank_counts']
        if mode == 'parallel':
            aw = sm.get('always_wrong_subset') or {}
            rec['GlobalPIQA_parallel_hard52_accuracy'] = aw.get('accuracy')
            rec['GlobalPIQA_parallel_hard52_mean_top_minus_correct'] = aw.get('mean_top_minus_correct')
            rec['GlobalPIQA_parallel_hard52_rank_counts'] = aw.get('correct_rank_counts')
    return rec


def prev_metrics() -> dict[str, dict[str, Any]]:
    p = load(PREV)
    out = {}
    for name in ['reference','control','treatment']:
        se = p.get('sentinels',{}).get(name,{})
        gp = p.get('globalpiqa',{}).get(name,{})
        rec = {'target': PREV_NAMES[name], 'Supplement': se.get('Supplement'), 'Entity': se.get('Entity')}
        for mode in ['parallel','nonparallel']:
            sm = gp.get(mode) or {}
            rec[f'GlobalPIQA_{mode}'] = sm.get('accuracy')
            rec[f'GlobalPIQA_{mode}_rank_counts'] = sm.get('correct_rank_counts')
            if mode == 'parallel':
                rec['GlobalPIQA_parallel_hard52_accuracy'] = sm.get('hard52_accuracy')
                rec['GlobalPIQA_parallel_hard52_mean_top_minus_correct'] = sm.get('hard52_mean_top_minus_correct')
                rec['GlobalPIQA_parallel_hard52_rank_counts'] = sm.get('hard52_rank_counts')
        out[name] = rec
    return out


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {k: float(v)-float(b[k]) for k,v in a.items() if isinstance(v,(int,float)) and isinstance(b.get(k),(int,float))}


def gp_rows(root: Path, target: str, mode: str) -> dict[str, dict[str, Any]]:
    p = root / f'{target}_{mode}_rows.csv'
    out = {}
    with p.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            rr['correct'] = boolv(rr.get('correct'))
            rr['correct_rank'] = int(rr['correct_rank'])
            rr['top_minus_correct'] = float(rr['top_minus_correct'])
            out[rr['example_id']] = rr
    return out


def pair(a_name: str, a: dict[str, dict[str, Any]], b_name: str, b: dict[str, dict[str, Any]], h: set[str]) -> dict[str, Any]:
    ids = sorted(set(a)&set(b))
    def sub(ii):
        if not ii: return {'n':0}
        return {
            'n':len(ii), 'a':a_name, 'b':b_name,
            'a_correct':sum(a[i]['correct'] for i in ii), 'b_correct':sum(b[i]['correct'] for i in ii),
            'both_correct':sum(a[i]['correct'] and b[i]['correct'] for i in ii),
            'a_only_correct':sum(a[i]['correct'] and not b[i]['correct'] for i in ii),
            'b_only_correct':sum((not a[i]['correct']) and b[i]['correct'] for i in ii),
            'both_wrong':sum((not a[i]['correct']) and (not b[i]['correct']) for i in ii),
            'a_better_rank':sum(a[i]['correct_rank'] < b[i]['correct_rank'] for i in ii),
            'same_rank':sum(a[i]['correct_rank'] == b[i]['correct_rank'] for i in ii),
            'b_better_rank':sum(a[i]['correct_rank'] > b[i]['correct_rank'] for i in ii),
            'a_lower_margin':sum(a[i]['top_minus_correct'] < b[i]['top_minus_correct'] for i in ii),
            'b_lower_margin':sum(a[i]['top_minus_correct'] > b[i]['top_minus_correct'] for i in ii),
            'mean_rank_delta_a_minus_b':statistics.fmean(a[i]['correct_rank']-b[i]['correct_rank'] for i in ii),
            'mean_margin_delta_a_minus_b':statistics.fmean(a[i]['top_minus_correct']-b[i]['top_minus_correct'] for i in ii),
        }
    r = sub(ids); r['hard52'] = sub([i for i in ids if i in h]); return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in [STD_SUPP, STD_GP, PREV, ANATOMY] if not p.exists()]
    if missing: raise FileNotFoundError(missing)
    std = std_metrics(); prev = prev_metrics(); h=hard_ids()
    metrics = {'standard':std, **prev}
    deltas = {f'standard_minus_{name}': delta(std, prev[name]) for name in ['reference','control','treatment']}
    rows = {mode:{'standard':gp_rows(STD_GP_ROOT,'standard_legacy_80m',mode)} for mode in ['parallel','nonparallel']}
    for mode in rows:
        for name,t in PREV_NAMES.items(): rows[mode][name]=gp_rows(PREV_GP_ROOT,t,mode)
    pairwise = {mode:{f'standard_vs_{name}': pair('standard', rows[mode]['standard'], name, rows[mode][name], h) for name in ['reference','control','treatment']} for mode in ['parallel','nonparallel']}
    interpretation = [
        'Before EWoK, standard staged WWM preserves GlobalPIQA_nonparallel exactly relative to the uninterrupted compact reference (53.0 vs 53.0), unlike PVDM control (51.0) and treatment (48.0).',
        'Standard staged WWM improves GlobalPIQA_parallel accuracy over the compact reference/control (27.18 vs 24.27) and treatment (22.33), but hard52 mean top-minus-correct is worse than reference/control (1.800 vs 1.717/1.603) and only slightly better than treatment (1.820). Thus it moves some parallel labels without repairing the deep hard-row margin object.',
        'Supplement/Entity do not favor standard: Supplement 58.72 is below reference 59.21 and control 61.49, Entity 27.70 is essentially reference/control and below treatment 28.78. The non-EWoK evidence suggests ordinary WWM target distribution is safer for broad GlobalPIQA than PVDM redistribution, but EWoK is still needed to decide the shared-damage split.'
    ]
    payload={'status':'PARTIAL_STANDARD_NON_EWOK_SYNTHESIS','created_utc':now(),'metrics':metrics,'deltas':deltas,'pairwise_globalpiqa':pairwise,'interpretation':interpretation,'source_files':{'standard_supp_entity':str(STD_SUPP),'standard_globalpiqa':str(STD_GP),'previous_step120':str(PREV)}}
    out=OUT/'partial_standard_non_ewok_synthesis.json'; out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — partial standard staged-WWM evidence before EWoK\n','No evaluation is rerun here; this only merges completed Supplement/Entity and GlobalPIQA files.\n','\n## Main table\n']
    for name in ['reference','standard','control','treatment']:
        r=metrics[name]
        lines.append(f"- `{name}`: Supplement={r.get('Supplement')}, Entity={r.get('Entity')}, GP_parallel={r.get('GlobalPIQA_parallel')}, GP_nonparallel={r.get('GlobalPIQA_nonparallel')}, hard52_acc={r.get('GlobalPIQA_parallel_hard52_accuracy')}, hard52_margin={r.get('GlobalPIQA_parallel_hard52_mean_top_minus_correct')}, hard52_ranks={r.get('GlobalPIQA_parallel_hard52_rank_counts')}\n")
    lines.append('\n## Standard deltas\n')
    for k,v in deltas.items(): lines.append(f'- `{k}`: {v}\n')
    lines.append('\n## Hard52 row movement\n')
    for name in ['reference','control','treatment']:
        g=pairwise['parallel'][f'standard_vs_{name}']['hard52']
        lines.append(f"- standard vs {name}: standard better rank {g['a_better_rank']}, same {g['same_rank']}, {name} better {g['b_better_rank']}; standard lower margin {g['a_lower_margin']}, {name} lower margin {g['b_lower_margin']}; mean margin delta {g['mean_margin_delta_a_minus_b']}.\n")
    lines.append('\n## Interpretation\n')
    for x in interpretation: lines.append(f'- {x}\n')
    lines.append(f'\nFiles: `{out}`\n')
    NOTE.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out_json':str(out),'note':str(NOTE),'standard_minus_reference':deltas['standard_minus_reference']},indent=2),flush=True)

if __name__=='__main__': main()
