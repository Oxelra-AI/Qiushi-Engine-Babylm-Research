#!/usr/bin/env python3
"""research: CPU-only training-dynamics comparison for scale1.75 100M endpoint.

Compare the completed adapter128 scale1.75 100M training log with the research
legal baseline log. This verifies identical data/order/mask/LR accounting and
summarizes loss movement by exposure. No model inference or GPU work is run.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path('.')
BASE_LOG = ROOT / 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/training_log.jsonl'
SCALE_LOG = ROOT / 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/training_log.jsonl'
BASE_METRICS = ROOT / 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/scientific_metrics.json'
SCALE_METRICS = ROOT / 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/scientific_metrics.json'
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_training_dynamics'
NOTE = ROOT / 'research/notes/frontier_consolidation/scale1p75_training_dynamics.md'
TARGETS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
ACCOUNT_FIELDS = ['step', 'lr', 'batch_words', 'cumulative_word_exposure', 'seq_len', 'masked_tokens', 'effective_mask_rate', 'mask_mode', 'mask_prob_nominal']


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out=[]
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def mean(xs: List[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def std(xs: List[float]) -> float | None:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0 if xs else None


def first_at_or_after(rows: List[Dict[str, Any]], target_words: int) -> Dict[str, Any]:
    for r in rows:
        if int(r['cumulative_word_exposure']) >= target_words:
            return r
    return rows[-1]


def summarize_window(rows: List[Dict[str, Any]], start_words: int, end_words: int) -> Dict[str, Any]:
    vals = [float(r['loss']) for r in rows if start_words < int(r['cumulative_word_exposure']) <= end_words]
    return {'n_steps': len(vals), 'loss_mean': mean(vals), 'loss_std': std(vals), 'loss_min': min(vals) if vals else None, 'loss_max': max(vals) if vals else None}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    base = read_jsonl(BASE_LOG)
    scale = read_jsonl(SCALE_LOG)
    base_m = load_json(BASE_METRICS)
    scale_m = load_json(SCALE_METRICS)
    errors=[]
    if len(base) != len(scale):
        errors.append(f'log length differs: base={len(base)} scale={len(scale)}')
    n = min(len(base), len(scale))
    mismatches = {field: 0 for field in ACCOUNT_FIELDS}
    examples = {field: [] for field in ACCOUNT_FIELDS}
    for i in range(n):
        b = base[i]; s = scale[i]
        for field in ACCOUNT_FIELDS:
            if b.get(field) != s.get(field):
                mismatches[field] += 1
                if len(examples[field]) < 3:
                    examples[field].append({'idx': i, 'base': b.get(field), 'scale': s.get(field)})
    if any(mismatches.values()):
        errors.append(f'accounting field mismatches: {mismatches}')

    milestones=[]
    for m in TARGETS:
        target=m*1_000_000
        b=first_at_or_after(base, target)
        s=first_at_or_after(scale, target)
        milestones.append({
            'target_M': m,
            'step': b['step'],
            'cum_words': b['cumulative_word_exposure'],
            'lr': b['lr'],
            'masked_tokens': b['masked_tokens'],
            'base_loss': b['loss'],
            'scale1p75_loss': s['loss'],
            'scale_minus_base_loss': float(s['loss']) - float(b['loss']),
        })
    windows=[]
    for start, end in [(0, 20), (20, 50), (50, 80), (80, 100), (90, 100), (95, 100), (99, 100)]:
        bsum=summarize_window(base, start*1_000_000, end*1_000_000)
        ssum=summarize_window(scale, start*1_000_000, end*1_000_000)
        windows.append({'window_M': f'{start}-{end}', 'base': bsum, 'scale1p75': ssum, 'mean_loss_delta': (ssum['loss_mean'] - bsum['loss_mean']) if ssum['loss_mean'] is not None and bsum['loss_mean'] is not None else None})

    summary={
        'status': 'SCALE1P75_TRAINING_DYNAMICS',
        'valid_identical_accounting': not errors,
        'errors': errors,
        'base_log': str(BASE_LOG),
        'scale_log': str(SCALE_LOG),
        'line_counts': {'base': len(base), 'scale1p75': len(scale)},
        'account_field_mismatch_counts': mismatches,
        'account_field_mismatch_examples': examples,
        'base_metrics_core': {k: base_m.get(k) for k in ['word_exposure','loss_first','loss_last','actual_training_steps','parameter_count','vocab_size','seed','masking_curriculum']},
        'scale_metrics_core': {k: scale_m.get(k) for k in ['word_exposure','loss_first','loss_last','actual_training_steps','parameter_count','vocab_size','seed','masking_curriculum']},
        'milestones': milestones,
        'windows': windows,
        'interpretation': [
            'Identical accounting fields mean the score difference is attributable to model architecture/optimization trajectory, not data order, word exposure, LR schedule, sequence length, or mask-count drift.',
            'Loss deltas are descriptive only; prior experiments showed scalar MLM loss does not reliably select official competence under this setup.',
        ],
    }
    out_json=OUT_DIR/'scale1p75_training_dynamics.json'
    out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines=[]
    lines.append('# research — scale1.75 100M training-dynamics comparison')
    lines.append('')
    lines.append('CPU-only comparison of research legal baseline and completed adapter128 scale1.75 100M endpoint training logs. It does not run model inference.')
    lines.append('')
    lines.append(f"- Identical accounting fields: `{summary['valid_identical_accounting']}`; errors: `{errors}`.")
    lines.append(f"- Log rows: research {len(base)}, scale1.75 {len(scale)}.")
    lines.append(f"- Mismatch counts for step/LR/batch/exposure/seq/mask fields: `{mismatches}`.")
    lines.append(f"- Final loss: research {base_m.get('loss_last')} vs scale1.75 {scale_m.get('loss_last')} (delta {float(scale_m.get('loss_last'))-float(base_m.get('loss_last')):+.6f}).")
    lines.append('')
    lines.append('## Milestones')
    lines.append('')
    lines.append('| target M | step | words | LR | masked | research loss | scale1.75 loss | Δloss |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in milestones:
        lines.append(f"| {r['target_M']} | {r['step']} | {r['cum_words']} | {r['lr']:.6g} | {r['masked_tokens']} | {r['base_loss']:.6f} | {r['scale1p75_loss']:.6f} | {r['scale_minus_base_loss']:+.6f} |")
    lines.append('')
    lines.append('## Loss windows')
    lines.append('')
    lines.append('| window M | base mean | scale mean | Δmean | steps |')
    lines.append('|---|---:|---:|---:|---:|')
    for w in windows:
        lines.append(f"| {w['window_M']} | {w['base']['loss_mean']:.6f} | {w['scale1p75']['loss_mean']:.6f} | {w['mean_loss_delta']:+.6f} | {w['base']['n_steps']} |")
    lines.append('')
    lines.append('Scientific consequence: the endpoint uses the same batch-word exposure, LR schedule, sequence length, and mask-count stream as research for every one of 2,529 updates. The official score difference, once measured, should be read as the effect of the separately routed residual branch on the learned trajectory rather than a hidden data/order/accounting change.')
    lines.append('')
    lines.append(f"JSON: `{out_json}`")
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'identical_accounting': summary['valid_identical_accounting'], 'out_json': str(out_json), 'note': str(NOTE), 'final_loss_delta': float(scale_m.get('loss_last'))-float(base_m.get('loss_last'))}, indent=2), flush=True)


if __name__ == '__main__':
    main()
