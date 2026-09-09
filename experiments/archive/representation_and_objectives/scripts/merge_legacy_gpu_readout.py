#!/usr/bin/env python3
"""Merge research standard-legacy GPU EWoK with already completed readouts.

This script performs no model scoring. It reads:
- completed standard-legacy Supplement/Entity and GlobalPIQA files from the
  cancelled combined wrapper;
- the isolated GPU EWoK output for standard-legacy;
- the research PVDM/control/reference readouts.
It writes the causal-split synthesis for deciding whether the shared PVDM/control
EWoK weakening came from staged replay or target redistribution/anchor swapping.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
STD_ROOT = WS / 'data/legacy_80m_readouts'
STD_GP = STD_ROOT / 'globalpiqa_margin'
STD_SUPP = STD_ROOT / 'supplement_entity/per_target/standard_legacy_80m.json'
STD_EWOK_ROOT = WS / 'data/legacy_80m_ewok_gpu'
STD_EWOK_TARGET = 'standard_legacy_80m_gpu'
PREV_SYN = WS / 'data/pvdm_full_readout_synthesis/pvdm_full_readout_synthesis.json'
PREV_ROOT = WS / 'data/pvdm_80m_readouts'
ANATOMY = WS / 'data/globalpiqa_parallel_anatomy/globalpiqa_parallel_anatomy.json'
OUT_DIR = WS / 'data/legacy_80m_causal_split'
NOTE = (ROOT / 'research/notes/representation_and_objectives/legacy_80m_causal_split.md')

PREV_NAMES = {
    'reference': 'compact_80m_reference',
    'control': 'pvdm_control_80m',
    'treatment': 'pvdm_treatment_80m',
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def qstats(vals: list[float]) -> dict[str, Any]:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return {'n': 0}
    def q(frac: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = frac * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {'n': len(xs), 'min': xs[0], 'p05': q(0.05), 'mean': statistics.fmean(xs), 'median': statistics.median(xs), 'p95': q(0.95), 'max': xs[-1]}


def boolv(v: Any) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.strip().lower() == 'true'
    return bool(v)


def fl(v: Any) -> float:
    try: return float(v)
    except Exception: return float('nan')


def load_anatomy() -> set[str]:
    if not ANATOMY.exists(): return set()
    d = load_json(ANATOMY)
    rows = d.get('agreement', {}).get('parallel', {}).get('rows', [])
    return {r['example_id'] for r in rows if r.get('n_ok') == 0}


def preflight() -> dict[str, Any]:
    paths = {
        'standard_supplement_entity': STD_SUPP,
        'standard_globalpiqa': STD_GP / 'standard_legacy_80m_margins.json',
        'standard_ewok_summary': STD_EWOK_ROOT / STD_EWOK_TARGET / 'ewok_interaction_summary.json',
        'standard_ewok_records': STD_EWOK_ROOT / STD_EWOK_TARGET / 'ewok_interaction_records.csv',
        'previous_synthesis': PREV_SYN,
    }
    out = {'status': 'READY', 'created_utc': now_utc(), 'paths': {k: str(v) for k,v in paths.items()}, 'missing': []}
    for k, p in paths.items():
        if not p.exists(): out['missing'].append(k)
    if out['missing']: out['status'] = 'NOT_READY'
    return out


def standard_metrics() -> dict[str, Any]:
    supp = load_json(STD_SUPP)
    gp = load_json(STD_GP / 'standard_legacy_80m_margins.json')
    ew = load_json(STD_EWOK_ROOT / STD_EWOK_TARGET / 'ewok_interaction_summary.json')
    tasks = supp.get('tasks', {})
    rec = {
        'target': 'standard_legacy_80m',
        'Supplement': tasks.get('Supplement', {}).get('score'),
        'Entity': tasks.get('Entity', {}).get('score'),
    }
    for mode in ['parallel', 'nonparallel']:
        s = gp['modes'][mode]['summary']
        rec[f'GlobalPIQA_{mode}'] = s.get('accuracy')
        if mode == 'parallel':
            aw = s.get('always_wrong_subset') or {}
            rec['GlobalPIQA_parallel_hard52_mean_top_minus_correct'] = aw.get('mean_top_minus_correct')
            rec['GlobalPIQA_parallel_hard52_rank_counts'] = aw.get('correct_rank_counts')
    s = ew['summary']
    rec['EWoK_fourcell_accuracy'] = s.get('accuracy')
    rec['EWoK_saved_wrong'] = s.get('saved_wrong')
    rec['EWoK_stable_failure'] = s.get('stable_failure')
    rec['EWoK_stable_failure_frac_wrong'] = s.get('stable_failure_frac_wrong')
    rec['EWoK_interaction_sum_wrong_median'] = (s.get('interaction_sum_wrong') or {}).get('median')
    rec['EWoK_interaction_sum_wrong_mean'] = (s.get('interaction_sum_wrong') or {}).get('mean')
    return rec


def previous_metrics() -> dict[str, dict[str, Any]]:
    prev = load_json(PREV_SYN)
    out = {}
    for name in ['reference', 'control', 'treatment']:
        rec = {'target': PREV_NAMES[name]}
        se = prev.get('sentinels', {}).get(name, {})
        gp = prev.get('globalpiqa', {}).get(name, {})
        ew = prev.get('ewok', {}).get(name, {})
        rec['Supplement'] = se.get('Supplement')
        rec['Entity'] = se.get('Entity')
        rec['GlobalPIQA_parallel'] = (gp.get('parallel') or {}).get('accuracy')
        rec['GlobalPIQA_nonparallel'] = (gp.get('nonparallel') or {}).get('accuracy')
        rec['GlobalPIQA_parallel_hard52_mean_top_minus_correct'] = (gp.get('parallel') or {}).get('hard52_mean_top_minus_correct')
        rec['GlobalPIQA_parallel_hard52_rank_counts'] = (gp.get('parallel') or {}).get('hard52_rank_counts')
        rec['EWoK_fourcell_accuracy'] = ew.get('accuracy')
        rec['EWoK_saved_wrong'] = ew.get('saved_wrong')
        rec['EWoK_stable_failure'] = ew.get('stable_failure')
        rec['EWoK_stable_failure_frac_wrong'] = ew.get('stable_failure_frac_wrong')
        rec['EWoK_interaction_sum_wrong_median'] = ew.get('interaction_sum_wrong_median')
        rec['EWoK_interaction_sum_wrong_mean'] = ew.get('interaction_sum_wrong_mean')
        out[name] = rec
    return out


def numeric_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    out = {}
    for k, v in a.items():
        if isinstance(v, (int, float)) and isinstance(b.get(k), (int, float)):
            out[k] = float(v) - float(b[k])
    return out


def load_gp_rows(root: Path, target: str, mode: str) -> dict[str, dict[str, Any]]:
    p = root / f'{target}_{mode}_rows.csv'
    out = {}
    with p.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            rr['correct'] = boolv(rr.get('correct'))
            rr['correct_rank'] = int(rr['correct_rank'])
            rr['top_minus_correct'] = float(rr['top_minus_correct'])
            rr['choice'] = int(rr['choice'])
            out[rr['example_id']] = rr
    return out


def load_ewok_rows(root: Path, target: str) -> dict[str, dict[str, Any]]:
    p = root / target / 'ewok_interaction_records.csv'
    bool_keys = {'saved_model_correct_flag', 'saved_model_wrong_flag', 'conditional_reversal_failure_stable', 'stable_nonpositive_interaction', 'both_official_sum_positive', 'both_within_context_sum_positive', 'both_within_context_mean_positive', 'local_both_actual_over_swapped_positive'}
    float_keys = {'interaction_sum', 'interaction_mean', 'deletion_interaction_sum', 'interaction_minus_deletion_sum'}
    out = {}
    with p.open(newline='', encoding='utf-8') as f:
        for i, r in enumerate(csv.DictReader(f)):
            rr = dict(r); key = rr.get('uid') or rr.get('example_id') or rr.get('row_id') or f'row_{i}'
            rr['_row_index'] = i
            for k in bool_keys:
                if k in rr: rr[k] = boolv(rr[k])
            for k in float_keys:
                if k in rr: rr[k] = fl(rr[k])
            out[key] = rr
    return out


def gp_pair(a_name: str, a_rows: dict[str, dict[str, Any]], b_name: str, b_rows: dict[str, dict[str, Any]], hard_ids: set[str]) -> dict[str, Any]:
    ids = sorted(set(a_rows) & set(b_rows))
    def sub(ii: list[str]) -> dict[str, Any]:
        if not ii: return {'n': 0}
        return {
            'n': len(ii), 'a': a_name, 'b': b_name,
            'a_correct': sum(a_rows[i]['correct'] for i in ii), 'b_correct': sum(b_rows[i]['correct'] for i in ii),
            'both_correct': sum(a_rows[i]['correct'] and b_rows[i]['correct'] for i in ii),
            'a_only_correct': sum(a_rows[i]['correct'] and not b_rows[i]['correct'] for i in ii),
            'b_only_correct': sum((not a_rows[i]['correct']) and b_rows[i]['correct'] for i in ii),
            'both_wrong': sum((not a_rows[i]['correct']) and (not b_rows[i]['correct']) for i in ii),
            'a_better_rank': sum(a_rows[i]['correct_rank'] < b_rows[i]['correct_rank'] for i in ii),
            'same_rank': sum(a_rows[i]['correct_rank'] == b_rows[i]['correct_rank'] for i in ii),
            'b_better_rank': sum(a_rows[i]['correct_rank'] > b_rows[i]['correct_rank'] for i in ii),
            'a_lower_margin': sum(a_rows[i]['top_minus_correct'] < b_rows[i]['top_minus_correct'] for i in ii),
            'b_lower_margin': sum(a_rows[i]['top_minus_correct'] > b_rows[i]['top_minus_correct'] for i in ii),
            'mean_rank_delta_a_minus_b': statistics.fmean(a_rows[i]['correct_rank'] - b_rows[i]['correct_rank'] for i in ii),
            'mean_margin_delta_a_minus_b': statistics.fmean(a_rows[i]['top_minus_correct'] - b_rows[i]['top_minus_correct'] for i in ii),
        }
    out = sub(ids)
    out['hard52'] = sub([i for i in ids if i in hard_ids])
    return out


def ewok_pair(a_name: str, a_rows: dict[str, dict[str, Any]], b_name: str, b_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(set(a_rows) & set(b_rows), key=lambda x: a_rows[x].get('_row_index', 0))
    if not ids: return {'n': 0, 'a': a_name, 'b': b_name}
    stable_a = {i for i in ids if boolv(a_rows[i].get('conditional_reversal_failure_stable'))}
    stable_b = {i for i in ids if boolv(b_rows[i].get('conditional_reversal_failure_stable'))}
    return {
        'n': len(ids), 'a': a_name, 'b': b_name,
        'a_correct': sum(boolv(a_rows[i].get('saved_model_correct_flag')) for i in ids),
        'b_correct': sum(boolv(b_rows[i].get('saved_model_correct_flag')) for i in ids),
        'both_correct': sum(boolv(a_rows[i].get('saved_model_correct_flag')) and boolv(b_rows[i].get('saved_model_correct_flag')) for i in ids),
        'a_only_correct': sum(boolv(a_rows[i].get('saved_model_correct_flag')) and not boolv(b_rows[i].get('saved_model_correct_flag')) for i in ids),
        'b_only_correct': sum((not boolv(a_rows[i].get('saved_model_correct_flag'))) and boolv(b_rows[i].get('saved_model_correct_flag')) for i in ids),
        'both_wrong': sum((not boolv(a_rows[i].get('saved_model_correct_flag'))) and (not boolv(b_rows[i].get('saved_model_correct_flag'))) for i in ids),
        'stable_a': len(stable_a), 'stable_b': len(stable_b), 'stable_both': len(stable_a & stable_b),
        'stable_a_only': len(stable_a - stable_b), 'stable_b_only': len(stable_b - stable_a),
        'interaction_sum_delta_a_minus_b': qstats([fl(a_rows[i].get('interaction_sum')) - fl(b_rows[i].get('interaction_sum')) for i in ids]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pf = preflight()
    (OUT_DIR / 'preflight.json').write_text(json.dumps(pf, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if pf['status'] != 'READY':
        raise RuntimeError({'preflight': str(OUT_DIR / 'preflight.json'), 'missing': pf['missing']})
    hard = load_anatomy()
    std = standard_metrics(); prev = previous_metrics()
    metrics = {'reference': prev['reference'], 'standard': std, 'control': prev['control'], 'treatment': prev['treatment']}
    deltas = {f'standard_minus_{name}': numeric_delta(std, prev[name]) for name in ['reference', 'control', 'treatment']}
    gp_std = {mode: load_gp_rows(STD_GP, 'standard_legacy_80m', mode) for mode in ['parallel', 'nonparallel']}
    gp_prev = {name: {mode: load_gp_rows(PREV_ROOT / 'globalpiqa_margin', t, mode) for mode in ['parallel', 'nonparallel']} for name,t in PREV_NAMES.items()}
    ew_std = load_ewok_rows(STD_EWOK_ROOT, STD_EWOK_TARGET)
    ew_prev = {name: load_ewok_rows(PREV_ROOT / 'ewok_fourcell', t) for name,t in PREV_NAMES.items()}
    pairwise = {'globalpiqa': {}, 'ewok': {}}
    for mode in ['parallel', 'nonparallel']:
        pairwise['globalpiqa'][mode] = {f'standard_vs_{name}': gp_pair('standard', gp_std[mode], name, gp_prev[name][mode], hard) for name in ['reference', 'control', 'treatment']}
    pairwise['ewok'] = {f'standard_vs_{name}': ewok_pair('standard', ew_std, name, ew_prev[name]) for name in ['reference', 'control', 'treatment']}
    interpretation = []
    sr = deltas['standard_minus_reference']
    ew_delta = sr.get('EWoK_fourcell_accuracy'); stable_delta = sr.get('EWoK_stable_failure_frac_wrong')
    if isinstance(ew_delta, float) and isinstance(stable_delta, float):
        if ew_delta >= -0.001 and stable_delta <= 0.002:
            interpretation.append('Standard staged WWM is close to the uninterrupted compact EWoK reference, so the shared EWoK weakening in PVDM/control is mainly attributable to target redistribution and anchor swapping rather than staged replay.')
        else:
            interpretation.append('Standard staged WWM also weakens EWoK relative to the uninterrupted compact reference, so the staged continuation path itself remains part of the causal mixture; relation objectives should not be added until this path is repaired or separated further.')
    if std.get('GlobalPIQA_nonparallel') == prev['reference'].get('GlobalPIQA_nonparallel'):
        interpretation.append('Standard staged WWM preserves nonparallel GlobalPIQA relative to compact reference, unlike PVDM treatment and control; this favors preserving ordinary WWM while adding only sparse relation-specific pressure if later design proceeds.')
    payload = {
        'status': 'LEGACY_80M_CAUSAL_SPLIT', 'created_utc': now_utc(),
        'research_question': 'Does standard staged WWM replay explain the shared PVDM/control EWoK weakening, or is target redistribution/anchor swapping the main source?',
        'preflight': pf, 'metrics': metrics, 'deltas': deltas, 'pairwise': pairwise, 'interpretation': interpretation,
        'source_files': {'standard_supplement_entity': str(STD_SUPP), 'standard_globalpiqa': str(STD_GP / 'standard_legacy_80m_margins.json'), 'standard_ewok_gpu': str(STD_EWOK_ROOT / STD_EWOK_TARGET / 'ewok_interaction_summary.json'), 'previous_step120': str(PREV_SYN)},
    }
    out = OUT_DIR / 'legacy_80m_causal_split.json'
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = []
    lines.append('# research — standard staged-WWM 80M causal split\n')
    lines.append('This note merges only completed files. No scoring is launched here.\n')
    lines.append('\n## Main table\n')
    for name in ['reference', 'standard', 'control', 'treatment']:
        r = metrics[name]
        lines.append(f"- `{name}`: Supplement={r.get('Supplement')}, Entity={r.get('Entity')}; GP_parallel={r.get('GlobalPIQA_parallel')}, GP_nonparallel={r.get('GlobalPIQA_nonparallel')}, GP_hard52_margin={r.get('GlobalPIQA_parallel_hard52_mean_top_minus_correct')}, GP_hard52_ranks={r.get('GlobalPIQA_parallel_hard52_rank_counts')}; EWoK_acc={r.get('EWoK_fourcell_accuracy')}, EWoK_stable_frac_wrong={r.get('EWoK_stable_failure_frac_wrong')}, EWoK_wrong_median={r.get('EWoK_interaction_sum_wrong_median')}\n")
    lines.append('\n## Standard deltas\n')
    for k, v in deltas.items(): lines.append(f'- `{k}`: {v}\n')
    lines.append('\n## Row movement\n')
    for name in ['reference', 'control', 'treatment']:
        gph = pairwise['globalpiqa']['parallel'][f'standard_vs_{name}']['hard52']
        ew = pairwise['ewok'][f'standard_vs_{name}']
        lines.append(f"- GP hard52 standard vs {name}: standard better rank {gph.get('a_better_rank')}, same {gph.get('same_rank')}, {name} better rank {gph.get('b_better_rank')}; standard lower margin {gph.get('a_lower_margin')}, {name} lower margin {gph.get('b_lower_margin')}; mean margin delta {gph.get('mean_margin_delta_a_minus_b')}.\n")
        lines.append(f"- EWoK standard vs {name}: standard correct {ew.get('a_correct')}, {name} correct {ew.get('b_correct')}, standard-only correct {ew.get('a_only_correct')}, {name}-only correct {ew.get('b_only_correct')}; stable standard {ew.get('stable_a')}, stable {name} {ew.get('stable_b')}, stable both {ew.get('stable_both')}.\n")
    lines.append('\n## Interpretation\n')
    for x in interpretation: lines.append(f'- {x}\n')
    lines.append(f"\nFiles: `{out}`\n")
    NOTE.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'out_json': str(out), 'note': str(NOTE), 'standard_minus_reference': deltas['standard_minus_reference'], 'interpretation': interpretation}, indent=2), flush=True)


if __name__ == '__main__':
    main()
