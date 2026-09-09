#!/usr/bin/env python3
"""research: assemble the completed scale1.75 100M zero-shot/Reading columns.

CPU/filesystem only.  This script does not run model inference and does not touch
active managed evaluation outputs.  It reads the already completed split-column
payloads from the hardened evaluator tree, constructs an isolated merged payload
usable by the saved-prediction item-flip comparator, and records the exact cheap
surface plus the SuperGLUE/AoA amount required to reach Overall 41.8.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path('.').resolve()
BASE_OUT = ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_100M_full_eval_hardened'
TARGET = 'scale1p75_100M_seed43022'
RUN_DIR = ROOT / 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder'
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_100m_completed_columns_payload'
OUT_JSON = OUT_DIR / 'scale1p75_100M_completed_zero_reading_payload.json'
OUT_MD = (ROOT / 'research/documents/frontier_consolidation/data/scale1p75_100m_completed_columns_payload/scale1p75_100M_completed_zero_reading_payload.md')

ZERO_COLUMNS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA_parallel', 'GlobalPIQA_nonparallel', 'Reading']
CHEAP_COLS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'Reading']
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
    if 'score' not in task:
        raise RuntimeError({'missing_score': kind, 'task_keys': sorted(task)})
    return float(task['score'])

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, Any] = {}
    part_sources: dict[str, str] = {}
    for kind in ZERO_COLUMNS:
        tasks[kind] = load_part_task(kind)
        part_sources[kind] = rel(part_payload(kind))

    scores = {
        'BLiMP': task_score(tasks['BLiMP'], 'BLiMP'),
        'Supplement': task_score(tasks['Supplement'], 'Supplement'),
        'EWoK': task_score(tasks['EWoK'], 'EWoK'),
        'Entity': task_score(tasks['Entity'], 'Entity'),
        'COMPS': task_score(tasks['COMPS'], 'COMPS'),
        'GlobalPIQA': mean([task_score(tasks['GlobalPIQA_parallel'], 'GlobalPIQA_parallel'), task_score(tasks['GlobalPIQA_nonparallel'], 'GlobalPIQA_nonparallel')]),
        'Reading': task_score(tasks['Reading'], 'Reading'),
    }
    cheap7 = mean(scores[c] for c in CHEAP_COLS)
    deltas = {c: scores[c] - REF[c] for c in CHEAP_COLS}
    total_needed = 9 * 41.8
    observed_sum = sum(scores[c] for c in CHEAP_COLS)
    required_superglue_plus_aoa = total_needed - observed_sum
    projected_overall_with_step35_sg_aoa = (observed_sum + REF['SuperGLUE'] + REF['AoA']) / 9

    payload = {
        'target': TARGET,
        'description': 'scale1.75 adapter128 100M endpoint, completed zero-shot and Reading split-column payload only; SuperGLUE/AoA pending elsewhere',
        'family': 'scale1p75_residual_adapter_100M_completed_zero_reading_surface',
        'run_dir': rel(RUN_DIR),
        'model_root': rel(RUN_DIR / 'hf_model'),
        'model_path': rel(RUN_DIR / 'hf_model/chck_100M'),
        'endpoint': 'chck_100M',
        'created_utc': now(),
        'tasks': tasks,
        'part_sources': part_sources,
        'official_overall': {
            'scores': scores,
            'complete_for_provisional_overall': False,
            'cheap7': cheap7,
            'observed_cheap_column_sum': observed_sum,
            'required_superglue_plus_aoa_for_overall_41p8': required_superglue_plus_aoa,
            'projected_overall_with_step35_superglue_and_aoa': projected_overall_with_step35_sg_aoa,
            'official_like_arithmetic': 'Overall requires mean(BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, AoA); SuperGLUE/AoA absent here',
        },
        'reference_100M_ref': REF,
        'deltas_vs_step35': deltas,
        'cheap7_delta_vs_step35': cheap7 - REF['cheap7'],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = [
        '# research — scale1.75 100M completed zero-shot/Reading surface',
        '',
        'This is an isolated CPU/filesystem payload assembled from completed split-column outputs. It does not contain SuperGLUE or AoA and is not a complete Overall result.',
        '',
        f'cheap7 = **{cheap7:.6f}**; delta vs research cheap7 = **{cheap7 - REF["cheap7"]:+.6f}**.',
        f'SuperGLUE+AoA required to reach Overall 41.8 from this cheap surface: **{required_superglue_plus_aoa:.6f}**.',
        f'If SuperGLUE and AoA equal research ({REF["SuperGLUE"]:.6f} + 0), projected Overall = **{projected_overall_with_step35_sg_aoa:.6f}**.',
        '',
        '| Column | scale1.75 100M | research 100M | Delta |',
        '|---|---:|---:|---:|',
    ]
    for c in CHEAP_COLS:
        lines.append(f'| {c} | {scores[c]:.6f} | {REF[c]:.6f} | {deltas[c]:+.6f} |')
    lines.append('')
    lines.append(f'Payload JSON: `{rel(OUT_JSON)}`')
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': 'SCALE1P75_COMPLETED_COLUMNS_PAYLOAD',
        'out_json': rel(OUT_JSON),
        'out_md': rel(OUT_MD),
        'cheap7': cheap7,
        'cheap7_delta_vs_step35': cheap7 - REF['cheap7'],
        'required_superglue_plus_aoa_for_41p8': required_superglue_plus_aoa,
        'projected_overall_with_step35_sg_aoa': projected_overall_with_step35_sg_aoa,
    }, indent=2), flush=True)

if __name__ == '__main__':
    main()
