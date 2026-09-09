#!/usr/bin/env python3
"""Compare training dynamics for the compact-view triangle.

Prepared during research while the repeat and adjbreak arms train. It is safe to run
before completion; it reports available rows and marks incomplete arms. After
completion, run it again to align all three training logs and scientific metrics by
word exposure.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import time
from typing import Any

USER_ROOT = pathlib.Path('.')
OUT_DEFAULT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/compact_triangle_dynamics'
RUNS = {
    'compact_view_reinvest': USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022',
    'compact_repeat_reinvest': USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2',
    'adjbreak_reinvest': USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/gc_adjbreak_reinvest_16k_seed43022_r2',
}
REFERENCE_FIRST_UPDATE = {
    'compact_view_reinvest': {'loss': 9.809807777404785, 'batch_words': 39370, 'masked_tokens': 8658, 'effective_mask_rate': 0.1526},
}

MILESTONES = [
    1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000,
    40_000_000, 60_000_000, 80_000_000, 100_000_000,
]


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_log(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if 'step' in rec and 'loss' in rec:
                rows.append(rec)
    return rows


def closest_at_or_after(rows: list[dict[str, Any]], exposure: int) -> dict[str, Any] | None:
    for r in rows:
        if int(r.get('cumulative_word_exposure', -1)) >= exposure:
            return r
    return None


def closest_by_step(rows: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    if not rows:
        return None
    best = min(rows, key=lambda r: abs(int(r.get('step', 0)) - step))
    return best


def moving_mean(vals: list[float], window: int = 50) -> float | None:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return None
    return statistics.mean(vals[-min(window, len(vals)):])


def summarize_run(name: str, run_dir: pathlib.Path) -> dict[str, Any]:
    log_path = run_dir / 'training_log.jsonl'
    metrics_path = run_dir / 'scientific_metrics.json'
    dyn_path = run_dir / 'dynamics_traces.jsonl'
    rows = read_log(log_path)
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    out: dict[str, Any] = {
        'name': name,
        'run_dir': str(run_dir),
        'training_log_path': str(log_path),
        'training_log_exists': log_path.exists(),
        'n_logged_steps': len(rows),
        'metrics_path': str(metrics_path),
        'metrics_exists': metrics_path.exists(),
        'dynamics_traces_path': str(dyn_path),
        'dynamics_traces_exists': dyn_path.exists(),
        'endpoint_chck100_exists': (run_dir / 'hf_model/chck_100M/model.safetensors').exists(),
        'first_row': rows[0] if rows else None,
        'last_row': rows[-1] if rows else None,
        'tail_loss_mean_last50': moving_mean([r.get('loss') for r in rows[-50:]], 50) if rows else None,
        'tail_mask_rate_mean_last50': moving_mean([r.get('effective_mask_rate') for r in rows[-50:]], 50) if rows else None,
        'milestones': {},
    }
    for m in MILESTONES:
        rec = closest_at_or_after(rows, m)
        out['milestones'][str(m)] = rec
    if metrics:
        for k in ['word_exposure', 'loss_first', 'loss_last', 'actual_training_steps', 'parameter_count', 'vocab_size', 'tokenizer_label', 'seed', 'extra_init_seed', 'train_rng_seed', 'example_jsonl_label']:
            if k in metrics:
                out[k] = metrics[k]
        cps = metrics.get('saved_checkpoints', [])
        out['saved_checkpoint_count'] = len(cps)
        out['last_checkpoint'] = cps[-1] if cps else None
    return out


def diff_records(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    if not a or not b:
        return None
    out: dict[str, Any] = {}
    for k in ['step', 'cumulative_word_exposure', 'loss', 'masked_tokens', 'effective_mask_rate', 'lr']:
        av = a.get(k); bv = b.get(k)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            out[k + '_a_minus_b'] = float(av) - float(bv)
    return out


def build_contrasts(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pairs = [
        ('view_minus_repeat', 'compact_view_reinvest', 'compact_repeat_reinvest'),
        ('view_minus_adjbreak', 'compact_view_reinvest', 'adjbreak_reinvest'),
        ('repeat_minus_adjbreak', 'compact_repeat_reinvest', 'adjbreak_reinvest'),
    ]
    for label, a, b in pairs:
        sa = summaries.get(a, {}); sb = summaries.get(b, {})
        c: dict[str, Any] = {
            'last_row_delta': diff_records(sa.get('last_row'), sb.get('last_row')),
            'tail_loss_mean_last50_delta': None,
            'milestone_deltas': {},
        }
        if isinstance(sa.get('tail_loss_mean_last50'), (int, float)) and isinstance(sb.get('tail_loss_mean_last50'), (int, float)):
            c['tail_loss_mean_last50_delta'] = float(sa['tail_loss_mean_last50']) - float(sb['tail_loss_mean_last50'])
        for m in MILESTONES:
            c['milestone_deltas'][str(m)] = diff_records(sa.get('milestones', {}).get(str(m)), sb.get('milestones', {}).get(str(m)))
        out[label] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(OUT_DEFAULT))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {name: summarize_run(name, run_dir) for name, run_dir in RUNS.items()}
    payload = {
        'status': 'COMPACT_TRIANGLE_DYNAMICS',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': 'Training-dynamics comparator for the compact-view mechanism triangle. It may be run during or after training; final mechanism interpretation requires completed runs and no-AoA readouts.',
        'runs': summaries,
        'contrasts': build_contrasts(summaries),
        'reference_first_update': REFERENCE_FIRST_UPDATE,
    }
    out_json = out_dir / 'triangle_dynamics_summary.json'
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'out_json': str(out_json), 'logged_steps': {k: v['n_logged_steps'] for k, v in summaries.items()}, 'metrics_exists': {k: v['metrics_exists'] for k, v in summaries.items()}}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
