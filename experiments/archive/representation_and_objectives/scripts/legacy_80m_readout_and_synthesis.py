#!/usr/bin/env python3
"""research matched legacy-WWM 70M->80M readout and causal split synthesis.

This script evaluates one staged standard-legacy WWM continuation on the exact same
70M->80M segment used by PVDM treatment/control.  It reuses the established
Supplement/Entity, GlobalPIQA all-option, and EWoK four-cell readers and compares
the result to the existing uninterrupted compact 80M reference plus PVDM arms.

Scientific purpose: separate ordinary staged continuation mechanics from the
PVDM target-redistribution contrast before any new relational objective is built.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
A01_WS = ROOT / 'experiments/archive/representation_and_objectives'
A02_WS = ROOT / 'experiments/archive/frontier_consolidation'
SCRIPT_DIR = A01_WS / 'scripts'
AI_SCRIPT_DIR = A01_WS / 'training/scripts'
OUT_ROOT = A01_WS / 'data/legacy_80m_readouts'
SUPP_ROOT = OUT_ROOT / 'supplement_entity'
GP_ROOT = OUT_ROOT / 'globalpiqa_margin'
EWOK_ROOT = OUT_ROOT / 'ewok_fourcell'
NOTE = (ROOT / 'research/notes/representation_and_objectives/legacy_80m_causal_split.md')
PREV_SYN = A01_WS / 'data/pvdm_full_readout_synthesis/pvdm_full_readout_synthesis.json'
PREV_READOUT = A01_WS / 'data/pvdm_80m_readouts'
ANATOMY = A01_WS / 'data/globalpiqa_parallel_anatomy/globalpiqa_parallel_anatomy.json'

TARGET = 'standard_legacy_80m'
RUN_DIR = A01_WS / 'training/runs/standard_legacy_70M_to_80M_seed43022'
MODEL_PATH = RUN_DIR / 'hf_model/chck_80M'

PREV_TARGET_TO_NAME = {
    'reference': 'compact_80m_reference',
    'control': 'pvdm_control_80m',
    'treatment': 'pvdm_treatment_80m',
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def qstats(vals: list[float]) -> dict[str, Any]:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return {'n': 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {'n': len(xs), 'min': xs[0], 'p05': q(0.05), 'mean': statistics.fmean(xs), 'median': statistics.median(xs), 'p95': q(0.95), 'max': xs[-1]}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def preflight() -> dict[str, Any]:
    rec: dict[str, Any] = {
        'status': 'READY',
        'created_utc': now_utc(),
        'target': TARGET,
        'run_dir': str(RUN_DIR),
        'run_dir_exists': RUN_DIR.exists(),
        'model_path': str(MODEL_PATH),
        'model_path_exists': MODEL_PATH.exists(),
        'scientific_metrics': str(RUN_DIR / 'scientific_metrics.json'),
        'scientific_metrics_exists': (RUN_DIR / 'scientific_metrics.json').exists(),
        'previous_pvdm_synthesis': str(PREV_SYN),
        'previous_pvdm_synthesis_exists': PREV_SYN.exists(),
        'errors': [],
    }
    if not RUN_DIR.exists():
        rec['errors'].append('missing run_dir')
    if not MODEL_PATH.exists():
        rec['errors'].append('missing hf_model/chck_80M')
    if not PREV_SYN.exists():
        rec['errors'].append('missing previous PVDM synthesis')
    mpath = RUN_DIR / 'scientific_metrics.json'
    if mpath.exists():
        try:
            m = load_json(mpath)
            rec['metrics_summary'] = {
                'variant': m.get('variant'),
                'word_exposure': m.get('word_exposure'),
                'continuation_words': m.get('continuation_words'),
                'actual_training_steps': m.get('actual_training_steps'),
                'stage_stop_name': m.get('stage_stop_name'),
                'mode': m.get('mode'),
                'loss_first': m.get('loss_first'),
                'loss_last': m.get('loss_last'),
                'parameter_count': m.get('parameter_count'),
                'vocab_size': m.get('vocab_size'),
                'micro_batch_size': m.get('micro_batch_size'),
            }
            if m.get('stage_stop_name') != 'chck_80M':
                rec['errors'].append(f"stage_stop_name {m.get('stage_stop_name')} != chck_80M")
            if m.get('word_exposure') != 80011326:
                rec['errors'].append(f"word_exposure {m.get('word_exposure')} != 80011326")
            if 'standard_legacy' not in str(m.get('variant')):
                rec['errors'].append(f"variant {m.get('variant')} is not standard_legacy")
        except Exception as exc:
            rec['errors'].append(f'cannot parse scientific_metrics: {exc}')
    else:
        rec['errors'].append('missing scientific_metrics')
    if rec['errors']:
        rec['status'] = 'NOT_READY'
    return rec


def run_supp_entity(gpu: int, force: bool) -> dict[str, Any]:
    mod = import_module(AI_SCRIPT_DIR / 'full_eval_fw_shared_anchor.py', 'full_eval_step121_legacy')
    mod.OUT_ROOT = SUPP_ROOT
    mod.TARGET_REGISTRY[TARGET] = {
        'run_dir': RUN_DIR,
        'endpoint': 'chck_80M',
        'description': 'research staged legacy WWM 70M->80M replay on the exact PVDM segment; used to separate continuation mechanics from target redistribution.',
        'family': 'standard_legacy_staged_80m',
    }
    old_argv = sys.argv
    try:
        sys.argv = ['full_eval_fw_shared_anchor.py', '--target', TARGET, '--endpoint', 'chck_80M', '--gpu', str(gpu), '--columns', 'Supplement', 'Entity'] + (['--force'] if force else [])
        t0 = time.time(); mod.main(); elapsed = time.time() - t0
    finally:
        sys.argv = old_argv
    return {'target': TARGET, 'elapsed_sec': round(elapsed, 3), 'out_root': str(SUPP_ROOT)}


def run_globalpiqa(max_items: int | None, threads: int) -> dict[str, Any]:
    mod = import_module(SCRIPT_DIR / 'globalpiqa_margin_reader.py', 'globalpiqa_step121_legacy')
    mod.OUT_ROOT = GP_ROOT
    mod.NOTE = (ROOT / 'research/notes/representation_and_objectives/legacy_80m_globalpiqa_margin_reader.md')
    mod.TARGETS[TARGET] = {
        'label': 'research staged legacy WWM 70M->80M replay on the exact PVDM segment',
        'model_root': RUN_DIR / 'hf_model',
        'revision': 'chck_80M',
        'official_parallel': None,
        'official_nonparallel': None,
    }
    old_argv = sys.argv
    try:
        sys.argv = ['globalpiqa_margin_reader.py', '--targets', TARGET, '--modes', 'parallel', 'nonparallel', '--batch_size', '8', '--non_causal_batch_size', '32', '--threads', str(threads)]
        if max_items is not None:
            sys.argv += ['--max_items', str(max_items)]
        t0 = time.time(); mod.main(); elapsed = time.time() - t0
    finally:
        sys.argv = old_argv
    return {'target': TARGET, 'elapsed_sec': round(elapsed, 3), 'out_root': str(GP_ROOT)}


def run_ewok(device: str, row_limit: int, threads: int) -> dict[str, Any]:
    mod = import_module(SCRIPT_DIR / 'fw_ewok_interaction_reader.py', 'ewok_step121_legacy')
    mod.OUT_ROOT = EWOK_ROOT
    mod.NOTE = (ROOT / 'research/notes/representation_and_objectives/121_legacy_80m_ewok_fourcell_reader.md')
    mod.DEFAULT_TARGETS.clear()
    mod.DEFAULT_TARGETS[TARGET] = {'model_path': MODEL_PATH, 'label': 'research staged legacy WWM 70M->80M replay'}
    old_argv = sys.argv
    try:
        sys.argv = ['fw_ewok_interaction_reader.py', '--targets', TARGET, '--device', device, '--threads', str(threads), '--row_batch_size', '64', '--masked_batch_size', '128']
        if row_limit:
            sys.argv += ['--row_limit', str(row_limit)]
        t0 = time.time(); mod.main(); elapsed = time.time() - t0
    finally:
        sys.argv = old_argv
    return {'target': TARGET, 'elapsed_sec': round(elapsed, 3), 'out_root': str(EWOK_ROOT)}


def load_gp_rows(path_root: Path, target: str, mode: str) -> dict[str, dict[str, Any]]:
    p = path_root / f'{target}_{mode}_rows.csv'
    if not p.exists():
        return {}
    out = {}
    with p.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            rr['correct'] = str(rr.get('correct')).strip().lower() == 'true'
            for k in ['choice', 'label', 'correct_rank']:
                if k in rr: rr[k] = int(rr[k])
            if 'top_minus_correct' in rr: rr['top_minus_correct'] = float(rr['top_minus_correct'])
            out[rr['example_id']] = rr
    return out


def load_ewok_rows(path_root: Path, target: str) -> dict[str, dict[str, Any]]:
    p = path_root / target / 'ewok_interaction_records.csv'
    if not p.exists():
        return {}
    bool_keys = {'saved_model_correct_flag', 'saved_model_wrong_flag', 'conditional_reversal_failure_stable', 'stable_nonpositive_interaction', 'both_official_sum_positive', 'both_within_context_sum_positive', 'both_within_context_mean_positive', 'local_both_actual_over_swapped_positive', 'deleted_contexts_identical'}
    float_keys = {'interaction_sum', 'interaction_mean', 'deletion_interaction_sum', 'interaction_minus_deletion_sum'}
    out = {}
    with p.open(newline='', encoding='utf-8') as f:
        for i, r in enumerate(csv.DictReader(f)):
            rr = dict(r)
            key = rr.get('uid') or rr.get('example_id') or rr.get('row_id') or f'row_{i}'
            rr['_row_index'] = i
            for k in bool_keys:
                if k in rr: rr[k] = str(rr[k]).strip().lower() == 'true'
            for k in float_keys:
                if k in rr:
                    try: rr[k] = float(rr[k])
                    except Exception: pass
            out[key] = rr
    return out


def boolv(r: dict[str, Any], k: str) -> bool:
    v = r.get(k)
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.strip().lower() == 'true'
    return bool(v)


def fl(r: dict[str, Any], k: str) -> float:
    try: return float(r.get(k))
    except Exception: return float('nan')


def load_anatomy() -> tuple[set[str], dict[str, list[str]]]:
    if not ANATOMY.exists(): return set(), {}
    d = load_json(ANATOMY)
    rows = d.get('agreement', {}).get('parallel', {}).get('rows', [])
    return {r['example_id'] for r in rows if r.get('n_ok') == 0}, {r['example_id']: list(r.get('categories') or []) for r in rows}


def standard_metrics() -> dict[str, Any]:
    rec: dict[str, Any] = {'target': TARGET}
    supp = SUPP_ROOT / 'per_target' / f'{TARGET}.json'
    if supp.exists():
        d = load_json(supp); tasks = d.get('tasks', {})
        rec['Supplement'] = tasks.get('Supplement', {}).get('score')
        rec['Entity'] = tasks.get('Entity', {}).get('score')
        rec['supplement_entity_source'] = str(supp)
    gp = GP_ROOT / f'{TARGET}_margins.json'
    if gp.exists():
        d = load_json(gp); modes = d.get('modes', {})
        for mode in ['parallel', 'nonparallel']:
            s = modes.get(mode, {}).get('summary', {})
            aw = s.get('always_wrong_subset') or {}
            rec[f'GlobalPIQA_{mode}'] = s.get('accuracy')
            if mode == 'parallel':
                rec['GlobalPIQA_parallel_hard52_mean_top_minus_correct'] = aw.get('mean_top_minus_correct')
                rec['GlobalPIQA_parallel_hard52_rank_counts'] = aw.get('correct_rank_counts')
    ew = EWOK_ROOT / TARGET / 'ewok_interaction_summary.json'
    if ew.exists():
        d = load_json(ew); s = d.get('summary', {})
        rec['EWoK_fourcell_accuracy'] = s.get('accuracy')
        rec['EWoK_stable_failure_frac_wrong'] = s.get('stable_failure_frac_wrong')
        rec['EWoK_interaction_sum_wrong_median'] = (s.get('interaction_sum_wrong') or {}).get('median')
        rec['EWoK_saved_wrong'] = s.get('saved_wrong')
        rec['EWoK_stable_failure'] = s.get('stable_failure')
    return rec


def previous_metrics() -> dict[str, dict[str, Any]]:
    prev = load_json(PREV_SYN)
    out: dict[str, dict[str, Any]] = {}
    for name in ['reference', 'control', 'treatment']:
        rec = {'target': PREV_TARGET_TO_NAME[name]}
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
        rec['EWoK_stable_failure_frac_wrong'] = ew.get('stable_failure_frac_wrong')
        rec['EWoK_interaction_sum_wrong_median'] = ew.get('interaction_sum_wrong_median')
        rec['EWoK_saved_wrong'] = ew.get('saved_wrong')
        rec['EWoK_stable_failure'] = ew.get('stable_failure')
        out[name] = rec
    return out


def numeric_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    out = {}
    for k, v in a.items():
        if isinstance(v, (int, float)) and isinstance(b.get(k), (int, float)):
            out[k] = float(v) - float(b[k])
    return out


def gp_pair(a_name: str, a_rows: dict[str, dict[str, Any]], b_name: str, b_rows: dict[str, dict[str, Any]], hard_ids: set[str]) -> dict[str, Any]:
    ids = sorted(set(a_rows) & set(b_rows))
    def sub(ii: list[str]) -> dict[str, Any]:
        if not ii: return {'n': 0}
        return {
            'n': len(ii),
            'a': a_name,
            'b': b_name,
            'a_correct': sum(a_rows[i]['correct'] for i in ii),
            'b_correct': sum(b_rows[i]['correct'] for i in ii),
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
    stable_a = {i for i in ids if boolv(a_rows[i], 'conditional_reversal_failure_stable')}
    stable_b = {i for i in ids if boolv(b_rows[i], 'conditional_reversal_failure_stable')}
    return {
        'n': len(ids), 'a': a_name, 'b': b_name,
        'a_correct': sum(boolv(a_rows[i], 'saved_model_correct_flag') for i in ids),
        'b_correct': sum(boolv(b_rows[i], 'saved_model_correct_flag') for i in ids),
        'both_correct': sum(boolv(a_rows[i], 'saved_model_correct_flag') and boolv(b_rows[i], 'saved_model_correct_flag') for i in ids),
        'a_only_correct': sum(boolv(a_rows[i], 'saved_model_correct_flag') and not boolv(b_rows[i], 'saved_model_correct_flag') for i in ids),
        'b_only_correct': sum((not boolv(a_rows[i], 'saved_model_correct_flag')) and boolv(b_rows[i], 'saved_model_correct_flag') for i in ids),
        'both_wrong': sum((not boolv(a_rows[i], 'saved_model_correct_flag')) and (not boolv(b_rows[i], 'saved_model_correct_flag')) for i in ids),
        'stable_a': len(stable_a),
        'stable_b': len(stable_b),
        'stable_both': len(stable_a & stable_b),
        'stable_a_only': len(stable_a - stable_b),
        'stable_b_only': len(stable_b - stable_a),
        'interaction_sum_delta_a_minus_b': qstats([fl(a_rows[i], 'interaction_sum') - fl(b_rows[i], 'interaction_sum') for i in ids]),
    }


def training_summary() -> dict[str, Any]:
    mpath = RUN_DIR / 'scientific_metrics.json'
    rec = {'scientific_metrics': str(mpath)}
    if mpath.exists():
        m = load_json(mpath)
        rec.update({
            'word_exposure': m.get('word_exposure'),
            'continuation_words': m.get('continuation_words'),
            'actual_training_steps': m.get('actual_training_steps'),
            'effective_batch_size': m.get('effective_batch_size'),
            'micro_batch_size': m.get('micro_batch_size'),
            'gradient_accumulation_steps': m.get('gradient_accumulation_steps'),
            'loss_first': m.get('loss_first'),
            'loss_last': m.get('loss_last'),
            'stage_stop_name': m.get('stage_stop_name'),
            'trainer_state_path': m.get('trainer_state_path'),
            'segment': m.get('segment'),
            'hashes': m.get('hashes'),
        })
    tlog = RUN_DIR / 'training_log.jsonl'
    if tlog.exists():
        rows = [json.loads(x) for x in tlog.read_text(encoding='utf-8').splitlines() if x.strip()]
        rec['training_log_rows'] = len(rows)
        if rows:
            rec['masked_tokens'] = qstats([float(r['masked_tokens']) for r in rows])
            rec['loss'] = qstats([float(r['loss']) for r in rows])
    return rec


def write_row_tables(hard_ids: set[str]) -> dict[str, str]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for mode in ['parallel', 'nonparallel']:
        std = load_gp_rows(GP_ROOT, TARGET, mode)
        others = {name: load_gp_rows(PREV_READOUT / 'globalpiqa_margin', tname, mode) for name, tname in PREV_TARGET_TO_NAME.items()}
        common = sorted(set(std).intersection(*(set(x) for x in others.values()))) if std and all(others.values()) else []
        if not common: continue
        flat = []
        for exid in common:
            rec = {'mode': mode, 'example_id': exid}
            if mode == 'parallel': rec['hard52'] = exid in hard_ids
            for name, rows in [('standard', std), *others.items()]:
                r = rows[exid]
                rec[f'{name}_correct'] = r['correct']; rec[f'{name}_rank'] = r['correct_rank']; rec[f'{name}_margin'] = r['top_minus_correct']; rec[f'{name}_choice'] = r['choice']
            rec['standard_minus_reference_rank'] = std[exid]['correct_rank'] - others['reference'][exid]['correct_rank']
            rec['standard_minus_reference_margin'] = std[exid]['top_minus_correct'] - others['reference'][exid]['top_minus_correct']
            flat.append(rec)
        p = OUT_ROOT / f'globalpiqa_{mode}_standard_vs_previous_rows.csv'
        with p.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader(); w.writerows(flat)
        paths[mode] = str(p)
    return paths


def synthesize() -> dict[str, Any]:
    pf = preflight()
    prev = previous_metrics()
    std = standard_metrics()
    hard_ids, _cats = load_anatomy()
    gp_std = {mode: load_gp_rows(GP_ROOT, TARGET, mode) for mode in ['parallel', 'nonparallel']}
    gp_prev = {name: {mode: load_gp_rows(PREV_READOUT / 'globalpiqa_margin', tname, mode) for mode in ['parallel', 'nonparallel']} for name, tname in PREV_TARGET_TO_NAME.items()}
    ew_std = load_ewok_rows(EWOK_ROOT, TARGET)
    ew_prev = {name: load_ewok_rows(PREV_READOUT / 'ewok_fourcell', tname) for name, tname in PREV_TARGET_TO_NAME.items()}
    pairwise = {'globalpiqa': {}, 'ewok': {}}
    for mode in ['parallel', 'nonparallel']:
        pairwise['globalpiqa'][mode] = {}
        for name in ['reference', 'control', 'treatment']:
            pairwise['globalpiqa'][mode][f'standard_vs_{name}'] = gp_pair('standard', gp_std[mode], name, gp_prev[name][mode], hard_ids)
    for name in ['reference', 'control', 'treatment']:
        pairwise['ewok'][f'standard_vs_{name}'] = ewok_pair('standard', ew_std, name, ew_prev[name])
    metrics = {'standard': std, **prev}
    deltas = {f'standard_minus_{name}': numeric_delta(std, prev[name]) for name in ['reference', 'control', 'treatment']}
    row_tables = write_row_tables(hard_ids)
    interp = []
    dref = deltas.get('standard_minus_reference', {})
    if dref:
        ew_delta = dref.get('EWoK_fourcell_accuracy')
        stable_delta = dref.get('EWoK_stable_failure_frac_wrong')
        if isinstance(ew_delta, float) and isinstance(stable_delta, float):
            if ew_delta >= -0.001 and stable_delta <= 0.002:
                interp.append('The staged legacy-WWM replay is close to the uninterrupted compact reference on EWoK, so the shared PVDM/control EWoK weakening is more likely caused by target redistribution and anchor swapping than by the staged continuation path itself.')
            else:
                interp.append('The staged legacy-WWM replay also weakens EWoK relative to uninterrupted compact, so the staged continuation path itself remains part of the causal mixture and should be repaired or separately isolated before adding a new relation objective.')
    payload = {
        'status': 'LEGACY_80M_READOUT_SYNTHESIS',
        'created_utc': now_utc(),
        'research_question': 'Does ordinary staged legacy-WWM 70M->80M replay on the exact PVDM segment match the uninterrupted compact 80M trajectory, or does the staged path itself explain the shared EWoK weakening seen in PVDM/control?',
        'preflight': pf,
        'training': training_summary(),
        'metrics': metrics,
        'deltas': deltas,
        'pairwise': pairwise,
        'row_tables': row_tables,
        'interpretation': interp,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / 'legacy_80m_readout_synthesis.json'
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = []
    lines.append('# research — staged legacy-WWM 80M causal split\n')
    lines.append('Purpose: compare one standard WWM staged 70M→80M replay against the existing compact reference and PVDM arms on the same readouts.\n')
    lines.append('\n## Integrity\n')
    lines.append(f"- Preflight status: `{pf['status']}`; errors: {pf.get('errors')}\n")
    lines.append(f"- Training: {payload['training']}\n")
    lines.append('\n## Main table\n')
    for name in ['reference', 'standard', 'control', 'treatment']:
        rec = metrics[name]
        lines.append(f"- `{name}`: Supplement={rec.get('Supplement')}, Entity={rec.get('Entity')}; GP_parallel={rec.get('GlobalPIQA_parallel')}, GP_nonparallel={rec.get('GlobalPIQA_nonparallel')}, GP_hard52_margin={rec.get('GlobalPIQA_parallel_hard52_mean_top_minus_correct')}, GP_hard52_ranks={rec.get('GlobalPIQA_parallel_hard52_rank_counts')}; EWoK_acc={rec.get('EWoK_fourcell_accuracy')}, EWoK_stable_frac_wrong={rec.get('EWoK_stable_failure_frac_wrong')}, EWoK_wrong_median={rec.get('EWoK_interaction_sum_wrong_median')}\n")
    lines.append('\n## Deltas from standard\n')
    for k, v in deltas.items(): lines.append(f'- `{k}`: {v}\n')
    lines.append('\n## Row movement\n')
    for name in ['reference', 'control', 'treatment']:
        gph = pairwise['globalpiqa']['parallel'][f'standard_vs_{name}'].get('hard52', {})
        ew = pairwise['ewok'][f'standard_vs_{name}']
        lines.append(f"- GP hard52 standard vs {name}: standard better rank {gph.get('a_better_rank')}, same {gph.get('same_rank')}, {name} better rank {gph.get('b_better_rank')}; standard lower margin {gph.get('a_lower_margin')}, {name} lower margin {gph.get('b_lower_margin')}; mean margin delta {gph.get('mean_margin_delta_a_minus_b')}.\n")
        lines.append(f"- EWoK standard vs {name}: standard correct {ew.get('a_correct')}, {name} correct {ew.get('b_correct')}, standard-only correct {ew.get('a_only_correct')}, {name}-only correct {ew.get('b_only_correct')}; stable standard {ew.get('stable_a')}, stable {name} {ew.get('stable_b')}, stable both {ew.get('stable_both')}.\n")
    lines.append('\n## Interpretation\n')
    for x in interp: lines.append(f'- {x}\n')
    lines.append(f"\nFiles: `{out}`; row tables: {row_tables}\n")
    NOTE.write_text('\n'.join(lines), encoding='utf-8')
    return {'out_json': str(out), 'note': str(NOTE), 'deltas': deltas, 'interpretation': interp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-supp-entity', action='store_true')
    ap.add_argument('--skip-globalpiqa', action='store_true')
    ap.add_argument('--skip-ewok', action='store_true')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--ewok-device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--ewok-row-limit', type=int, default=0)
    ap.add_argument('--globalpiqa-max-items', type=int, default=None)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pf = preflight()
    pf_path = OUT_ROOT / 'preflight.json'
    pf_path.write_text(json.dumps(pf, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN', 'ready': pf['status'], 'preflight': str(pf_path)}, indent=2), flush=True)
        return
    if pf['status'] != 'READY':
        raise RuntimeError({'preflight': str(pf_path), 'errors': pf.get('errors')})
    runs: dict[str, Any] = {}
    if not args.skip_supp_entity:
        runs['supplement_entity'] = run_supp_entity(args.gpu, args.force)
    if not args.skip_globalpiqa:
        runs['globalpiqa'] = run_globalpiqa(args.globalpiqa_max_items, args.threads)
    if not args.skip_ewok:
        runs['ewok'] = run_ewok(args.ewok_device, args.ewok_row_limit, args.threads)
    syn = synthesize()
    run_summary = {'status': 'LEGACY_80M_READOUTS_DONE', 'created_utc': now_utc(), 'preflight': str(pf_path), 'runs': runs, **syn}
    rp = OUT_ROOT / 'legacy_80m_readout_run_summary.json'
    rp.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': run_summary['status'], 'out_json': syn['out_json'], 'note': syn['note'], 'interpretation': syn['interpretation'], 'deltas': syn['deltas'].get('standard_minus_reference')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
