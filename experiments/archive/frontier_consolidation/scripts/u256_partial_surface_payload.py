#!/usr/bin/env python3
"""research: assemble the U256 100M completed non-SuperGLUE surface.

CPU/filesystem only. This script does not query the managed evaluator status and
runs no model inference. It reads split-column payloads already written by the
hardened U256 evaluator, records the completed zero-shot/Reading/AoA surface,
computes what SuperGLUE would need to be for several research-relevant targets,
and writes a candidate payload usable by the saved-prediction item-flip comparator.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path('.').resolve()
BASE_OUT = ROOT / 'experiments/archive/frontier_consolidation/data/u256_100M_full_eval_hardened'
TARGET = 'U256_100M_seed43022'
RUN_DIR = ROOT / 'experiments/archive/frontier_consolidation/training/runs/eu_U256_legal16k_seed43022_100M'
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/u256_100m_partial_surface'
OUT_JSON = OUT_DIR / 'u256_100M_completed_non_superglue_payload.json'
OUT_MD = (ROOT / 'research/documents/frontier_consolidation/data/u256_100m_partial_surface/u256_100M_completed_non_superglue_payload.md')

ZERO_COLUMNS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA_parallel', 'GlobalPIQA_nonparallel', 'Reading']
DISCRETE_PAYLOAD_COLUMNS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA_parallel', 'GlobalPIQA_nonparallel']
CHEAP_COLS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'Reading']
ALL_TARGETS = {
    'overall': 41.257770896404615,
    'scale1p75_overall': 41.57074653643003,
    'sota_41p8': 41.8,
}
SUPERGLUE_TASKS = ['boolq', 'mnli', 'mrpc', 'multirc', 'qqp', 'rte', 'wsc']
REF = {
    'BLiMP': 65.8707181799453,
    'Supplement': 61.16566092036889,
    'EWoK': 50.39323748109589,
    'Entity': 27.400833994026197,
    'COMPS': 52.00834536316919,
    'SuperGLUE': 70.27986764740969,
    'GlobalPIQA': 36.0631067961165,
    'Reading': 8.13816768550987,
    'AoA': 0.0,
    'Overall': 41.257770896404615,
}
REF['cheap7'] = mean(REF[c] for c in CHEAP_COLS)
SCALE1P75_REF = {
    'BLiMP': 68.63175326978445,
    'Supplement': 62.89517138691106,
    'EWoK': 49.08009312557581,
    'Entity': 27.464548753037228,
    'COMPS': 52.30391659251573,
    'SuperGLUE': 69.33459939303675,
    'GlobalPIQA': 36.10679611650485,
    'Reading': 8.319840190504358,
    'AoA': 0.0,
    'Overall': 41.57074653643003,
}
SCALE1P75_REF['cheap7'] = mean(SCALE1P75_REF[c] for c in CHEAP_COLS)


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def part_name(kind: str) -> str:
    return kind.replace('GlobalPIQA_', 'GP_').replace('/', '_')


def part_payload(kind: str) -> Path:
    pt = f'{TARGET}__{part_name(kind)}'
    return BASE_OUT / 'parts' / part_name(kind) / 'eval' / 'per_target' / f'{pt}.json'


def load_part_task(kind: str) -> dict[str, Any]:
    p = part_payload(kind)
    if not p.exists():
        raise FileNotFoundError({'missing_part_payload': rel(p), 'kind': kind})
    payload = read_json(p)
    task = payload.get('tasks', {}).get(kind)
    if not isinstance(task, dict):
        raise RuntimeError({'missing_task_in_part': kind, 'payload': rel(p)})
    if int(task.get('returncode', -999)) != 0:
        raise RuntimeError({'part_not_successful': kind, 'returncode': task.get('returncode'), 'payload': rel(p), 'error': task.get('error')})
    pred = task.get('predictions')
    if pred and not (ROOT / pred).exists():
        raise FileNotFoundError({'missing_predictions': pred, 'kind': kind})
    return task


def task_score(task: dict[str, Any], kind: str) -> float:
    if kind == 'Reading':
        scores = task.get('scores', {})
        if 'Reading' not in scores:
            raise RuntimeError({'missing_reading_score': task})
        return float(scores['Reading'])
    if kind == 'AoA':
        if 'aoa_leaderboard_score' in task:
            return float(task['aoa_leaderboard_score'])
        if 'aoa_official' in task:
            return float(task['aoa_official'])
        raise RuntimeError({'missing_aoa_score': task})
    if 'score' not in task:
        raise RuntimeError({'missing_score': kind, 'task_keys': sorted(task)})
    return float(task['score'])


def load_superglue_partial() -> dict[str, Any]:
    p = part_payload('SuperGLUE')
    if not p.exists():
        return {'payload': rel(p), 'exists': False, 'completed_tasks': [], 'completed_scores': {}, 'observed_partial_mean': None}
    payload = read_json(p)
    rec = payload.get('tasks', {}).get('SuperGLUE')
    if not isinstance(rec, dict):
        return {'payload': rel(p), 'exists': True, 'completed_tasks': [], 'completed_scores': {}, 'observed_partial_mean': None, 'error': 'missing SuperGLUE task dict'}
    completed_scores: dict[str, float] = {}
    raw_records: list[dict[str, Any]] = []
    for tr in rec.get('tasks', []) or []:
        if not isinstance(tr, dict):
            continue
        task = str(tr.get('task'))
        if task not in SUPERGLUE_TASKS:
            continue
        if int(tr.get('returncode', -999)) != 0:
            continue
        # The current U256 partial payload is still accuracy-style for completed BoolQ/MultiRC.
        # Full official scoring will later use the hardened merge/primary-metric path.
        score = tr.get('score', None)
        if score is None:
            if 'accuracy' in tr:
                score = tr['accuracy']
            elif 'f1' in tr:
                score = tr['f1']
        if score is None:
            continue
        completed_scores[task] = float(score)
        raw_records.append(tr)
    observed = mean(completed_scores.values()) if completed_scores else None
    missing = [t for t in SUPERGLUE_TASKS if t not in completed_scores]
    return {
        'payload': rel(p),
        'exists': True,
        'completed_tasks': sorted(completed_scores),
        'missing_tasks': missing,
        'completed_scores': completed_scores,
        'observed_partial_mean': observed,
        'payload_superglue_mean_field': rec.get('superglue_mean'),
        'raw_completed_records': raw_records,
        'note': 'Partial only; full official SuperGLUE requires all seven tasks and primary metrics for MRPC/QQP.',
    }


def required_superglue_analysis(scores: dict[str, float], superglue_partial: dict[str, Any]) -> dict[str, Any]:
    other8_sum = sum(scores[c] for c in ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'Reading', 'AoA'])
    completed_scores = superglue_partial.get('completed_scores') or {}
    n_done = len(completed_scores)
    sum_done = sum(float(v) for v in completed_scores.values())
    n_remaining = len(SUPERGLUE_TASKS) - n_done
    out: dict[str, Any] = {}
    for name, overall_target in ALL_TARGETS.items():
        required_sg = 9.0 * overall_target - other8_sum
        if n_remaining > 0:
            required_remaining_mean = (7.0 * required_sg - sum_done) / n_remaining
        else:
            required_remaining_mean = None
        out[name] = {
            'overall_target': overall_target,
            'required_full_superglue_mean': required_sg,
            'required_remaining_superglue_mean_given_completed_tasks': required_remaining_mean,
            'mathematically_impossible_given_completed_tasks': required_remaining_mean is not None and required_remaining_mean > 100.0,
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, Any] = {}
    part_sources: dict[str, str] = {}
    for kind in ZERO_COLUMNS + ['AoA']:
        tasks[kind] = load_part_task(kind)
        part_sources[kind] = rel(part_payload(kind))

    raw_scores = {kind: task_score(tasks[kind], kind) for kind in ZERO_COLUMNS + ['AoA']}
    scores = {
        'BLiMP': raw_scores['BLiMP'],
        'Supplement': raw_scores['Supplement'],
        'EWoK': raw_scores['EWoK'],
        'Entity': raw_scores['Entity'],
        'COMPS': raw_scores['COMPS'],
        'GlobalPIQA': mean([raw_scores['GlobalPIQA_parallel'], raw_scores['GlobalPIQA_nonparallel']]),
        'Reading': raw_scores['Reading'],
        'AoA': raw_scores['AoA'],
    }
    cheap7 = mean(scores[c] for c in CHEAP_COLS)
    deltas_step35 = {c: scores[c] - REF[c] for c in scores}
    deltas_scale1p75 = {c: scores[c] - SCALE1P75_REF[c] for c in scores}
    superglue_partial = load_superglue_partial()
    required = required_superglue_analysis(scores, superglue_partial)
    projected = {
        'with_step35_superglue': (sum(scores.values()) + REF['SuperGLUE']) / 9.0,
        'with_scale1p75_superglue': (sum(scores.values()) + SCALE1P75_REF['SuperGLUE']) / 9.0,
    }

    # For item-flip comparator compatibility: keep split GlobalPIQA tasks and official_overall scores.
    candidate_payload = {
        'target': TARGET,
        'description': 'U256 100M endpoint completed non-SuperGLUE surface from split-column payloads; SuperGLUE still partial or pending in managed evaluator',
        'family': 'u256_faithful_visibility_100M_completed_non_superglue_surface',
        'run_dir': rel(RUN_DIR),
        'model_root': rel(RUN_DIR / 'hf_model'),
        'model_path': rel(RUN_DIR / 'hf_model/chck_100M'),
        'endpoint': 'chck_100M',
        'created_utc': now(),
        'tasks': {k: tasks[k] for k in DISCRETE_PAYLOAD_COLUMNS},
        'part_sources': part_sources,
        'official_overall': {
            'scores': scores,
            'complete_for_provisional_overall': False,
            'cheap7': cheap7,
            'observed_non_superglue_column_sum': sum(scores.values()),
            'required_superglue': required,
            'projected_overall': projected,
            'official_like_arithmetic': 'Overall requires mean(BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, AoA); SuperGLUE incomplete or absent here',
        },
        'reference_100M_ref': REF,
        'scale1p75_100M_ref': SCALE1P75_REF,
        'deltas_vs_step35': deltas_step35,
        'deltas_vs_scale1p75': deltas_scale1p75,
        'cheap7_delta_vs_step35': cheap7 - REF['cheap7'],
        'cheap7_delta_vs_scale1p75': cheap7 - SCALE1P75_REF['cheap7'],
        'superglue_partial': superglue_partial,
    }
    OUT_JSON.write_text(json.dumps(candidate_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    lines = [
        '# research — U256 100M completed non-SuperGLUE surface',
        '',
        'This is a CPU/filesystem read of split-column artifacts already written by the hardened evaluator. It is not a complete Overall result because SuperGLUE is incomplete or still pending.',
        '',
        f'cheap7 = **{cheap7:.6f}**; delta vs research cheap7 = **{cheap7 - REF["cheap7"]:+.6f}**; delta vs scale1.75 cheap7 = **{cheap7 - SCALE1P75_REF["cheap7"]:+.6f}**.',
        f'Projected Overall with research SuperGLUE ({REF["SuperGLUE"]:.6f}) = **{projected["with_step35_superglue"]:.6f}**.',
        f'Projected Overall with scale1.75 SuperGLUE ({SCALE1P75_REF["SuperGLUE"]:.6f}) = **{projected["with_scale1p75_superglue"]:.6f}**.',
        '',
        '| Target | required full SuperGLUE | required remaining mean given completed SG tasks |',
        '|---|---:|---:|',
    ]
    for name, rec in required.items():
        rem = rec['required_remaining_superglue_mean_given_completed_tasks']
        rem_s = 'n/a' if rem is None else f'{rem:.6f}'
        lines.append(f'| {name} | {rec["required_full_superglue_mean"]:.6f} | {rem_s} |')
    lines.extend(['', '| Column | U256 100M | research 100M | Delta vs research | scale1.75 100M | Delta vs scale1.75 |', '|---|---:|---:|---:|---:|---:|'])
    for c in ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'Reading', 'AoA']:
        lines.append(f'| {c} | {scores[c]:.6f} | {REF[c]:.6f} | {deltas_step35[c]:+.6f} | {SCALE1P75_REF[c]:.6f} | {deltas_scale1p75[c]:+.6f} |')
    lines.extend(['', 'SuperGLUE partial state:'])
    lines.append(f"- completed tasks: {superglue_partial.get('completed_tasks')}")
    lines.append(f"- observed partial mean: {superglue_partial.get('observed_partial_mean')}")
    lines.append(f"- missing tasks: {superglue_partial.get('missing_tasks')}")
    lines.extend(['', f'Payload JSON: `{rel(OUT_JSON)}`'])
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({
        'status': 'U256_COMPLETED_NON_SUPERGLUE_SURFACE',
        'out_json': rel(OUT_JSON),
        'out_md': rel(OUT_MD),
        'cheap7': cheap7,
        'cheap7_delta_vs_step35': cheap7 - REF['cheap7'],
        'cheap7_delta_vs_scale1p75': cheap7 - SCALE1P75_REF['cheap7'],
        'required_superglue_for_41p8': required['sota_41p8']['required_full_superglue_mean'],
        'required_remaining_superglue_mean_for_41p8': required['sota_41p8']['required_remaining_superglue_mean_given_completed_tasks'],
        'projected_overall_with_step35_superglue': projected['with_step35_superglue'],
        'superglue_completed_tasks': superglue_partial.get('completed_tasks'),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
