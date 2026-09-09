#!/usr/bin/env python3
"""research: compare U256 and research saved training logs.

This CPU-only script reads existing JSONL logs. It does not run training or
evaluation.  It quantifies how the faithful stream-object visibility run differs
from the research row256 reference in update count, batch-word geometry, LR trace,
mask counts, and cumulative exposure milestones.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import statistics as stats
import time
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
LOG = _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/training_log.jsonl')
U256_LOG = _public_path('experiments/archive/frontier_consolidation/training/runs/eu_U256_legal16k_seed43022_100M/training_log.jsonl')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/u256_training_log_comparison')
NOTE = _public_path('research/notes/frontier_consolidation/u256_training_log_comparison.md')
MILESTONES = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(USER_ROOT))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float('nan')


def stdev(xs: list[float]) -> float:
    return float(stats.pstdev(xs)) if xs else float('nan')


def quantiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    ys = sorted(float(x) for x in xs)
    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        idx = p * (len(ys) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return ys[lo]
        return ys[lo] * (hi - idx) + ys[hi] * (idx - lo)
    return {'min': ys[0], 'p05': q(0.05), 'p50': q(0.5), 'p95': q(0.95), 'max': ys[-1]}


def summarize_series(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    xs = [float(r[key]) for r in rows if key in r and isinstance(r[key], (int, float))]
    return {'sum': sum(xs), 'mean': mean(xs), 'std': stdev(xs), 'q': quantiles(xs)}


def milestone_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in MILESTONES:
        target = m * 1_000_000
        chosen = min(rows, key=lambda r: abs(int(r.get('cumulative_word_exposure', -10**18)) - target))
        out[f'{m}M'] = {
            'step': chosen.get('step'),
            'cumulative_word_exposure': chosen.get('cumulative_word_exposure'),
            'distance_to_target': int(chosen.get('cumulative_word_exposure', 0)) - target,
            'lr': chosen.get('lr'),
            'loss': chosen.get('loss'),
            'batch_words': chosen.get('batch_words'),
            'masked_tokens': chosen.get('masked_tokens'),
            'effective_mask_rate': chosen.get('effective_mask_rate'),
        }
    return out


def compare_common_steps(base: list[dict[str, Any]], u256: list[dict[str, Any]]) -> dict[str, Any]:
    n = min(len(base), len(u256))
    fields = ['lr', 'batch_words', 'cumulative_word_exposure', 'masked_tokens', 'effective_mask_rate', 'loss']
    diffs: dict[str, dict[str, Any]] = {}
    for f in fields:
        vals = []
        abs_vals = []
        mismatch = 0
        max_row = None
        max_abs = -1.0
        for i in range(n):
            if f not in base[i] or f not in u256[i]:
                continue
            d = float(u256[i][f]) - float(base[i][f])
            vals.append(d); abs_vals.append(abs(d))
            if abs(d) > 1e-12:
                mismatch += 1
            if abs(d) > max_abs:
                max_abs = abs(d)
                max_row = {'step': i + 1, 'research': base[i].get(f), 'u256': u256[i].get(f), 'u256_minus_step35': d}
        diffs[f] = {
            'mismatch_count': mismatch,
            'mean_diff': mean(vals),
            'std_diff': stdev(vals),
            'mean_abs_diff': mean(abs_vals),
            'max_abs_diff_row': max_row,
            'diff_quantiles': quantiles(vals),
        }
    # Correlate loss deltas with batch/mask/exposure deltas as a rough training-surface clue.
    corr: dict[str, float | None] = {}
    def pearson(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) < 3 or len(xs) != len(ys):
            return None
        mx = mean(xs); my = mean(ys)
        vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
        if vx <= 0 or vy <= 0:
            return None
        return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))
    loss_delta = [float(u256[i]['loss']) - float(base[i]['loss']) for i in range(n) if 'loss' in base[i] and 'loss' in u256[i]]
    for f in ['batch_words', 'cumulative_word_exposure', 'masked_tokens', 'effective_mask_rate']:
        xs = [float(u256[i][f]) - float(base[i][f]) for i in range(n) if f in base[i] and f in u256[i] and 'loss' in base[i] and 'loss' in u256[i]]
        ys = [float(u256[i]['loss']) - float(base[i]['loss']) for i in range(n) if f in base[i] and f in u256[i] and 'loss' in base[i] and 'loss' in u256[i]]
        corr[f'_delta_vs_loss_delta__{f}'] = pearson(xs, ys)
    return {'common_steps': n, 'field_differences': diffs, 'correlations': corr}


def run(log: Path, u256_log: Path, out_dir: Path, note: Path) -> dict[str, Any]:
    base = read_jsonl(log)
    u256 = read_jsonl(u256_log)
    out_dir.mkdir(parents=True, exist_ok=True)
    note.parent.mkdir(parents=True, exist_ok=True)
    base_words = int(base[-1]['cumulative_word_exposure']) if base else None
    u_words = int(u256[-1]['cumulative_word_exposure']) if u256 else None
    base_milestones = milestone_rows(base)
    u_milestones = milestone_rows(u256)
    milestone_delta: dict[str, dict[str, Any]] = {}
    for m in base_milestones:
        bm = base_milestones[m]; um = u_milestones[m]
        milestone_delta[m] = {
            'step_delta': int(um['step']) - int(bm['step']),
            'cum_words_delta': int(um['cumulative_word_exposure']) - int(bm['cumulative_word_exposure']),
            'loss_delta': float(um['loss']) - float(bm['loss']),
            'lr_delta': float(um['lr']) - float(bm['lr']),
            'masked_tokens_delta': int(um['masked_tokens']) - int(bm['masked_tokens']),
            'u256': um,
            'research': bm,
        }
    result = {
        'status': 'U256_TRAINING_LOG_COMPARISON',
        'created_utc': now(),
        'inputs': {'log': rel(log), 'u256_log': rel(u256_log)},
        'line_counts': {'research': len(base), 'u256': len(u256)},
        'final_cumulative_words': {'research': base_words, 'u256': u_words, 'delta': (u_words - base_words) if base_words is not None and u_words is not None else None},
        'final_losses': {'research': base[-1].get('loss') if base else None, 'u256': u256[-1].get('loss') if u256 else None, 'u256_minus_step35': (float(u256[-1]['loss']) - float(base[-1]['loss'])) if base and u256 else None},
        'series': {
            'batch_words': summarize_series(base, 'batch_words'),
            'u256_batch_words': summarize_series(u256, 'batch_words'),
            'masked_tokens': summarize_series(base, 'masked_tokens'),
            'u256_masked_tokens': summarize_series(u256, 'masked_tokens'),
            'effective_mask_rate': summarize_series(base, 'effective_mask_rate'),
            'u256_effective_mask_rate': summarize_series(u256, 'effective_mask_rate'),
        },
        'common_step_comparison': compare_common_steps(base, u256),
        'milestones': milestone_delta,
        'scientific_reading': '',
    }
    cdiff = result['common_step_comparison']['field_differences']
    result['scientific_reading'] = (
        f"U256 and research end at the same counted 100M words, but U256 uses {len(u256)} updates versus research {len(base)}. "
        f"Across common steps, LR is essentially the same at each index (mismatches {cdiff['lr']['mismatch_count']}, max abs {cdiff['lr']['max_abs_diff_row']['u256_minus_step35']:.3g}), "
        f"while batch word counts and mask counts differ at almost every step because chunking changes the training examples. "
        f"U256's final MLM loss is lower by {result['final_losses']['u256_minus_step35']:.6f}, but previous routes showed lower loss can redistribute rather than improve official competence, so the pending full score remains decisive."
    )
    out_json = out_dir / 'u256_training_log_comparison.json'
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = [
        '# research — U256 training-log comparison',
        '',
        f"Status: `{result['status']}`",
        '',
        f"- line counts: research {len(base)}, U256 {len(u256)}",
        f"- final counted words: research {base_words}, U256 {u_words}",
        f"- final losses: research {result['final_losses']['research']}, U256 {result['final_losses']['u256']}, delta {result['final_losses']['u256_minus_step35']:+.6f}",
        f"- common-step LR mismatches: {cdiff['lr']['mismatch_count']}; batch-word mismatches: {cdiff['batch_words']['mismatch_count']}; masked-token mismatches: {cdiff['masked_tokens']['mismatch_count']}",
        '',
        '## Milestones',
        '| target | step delta | cumulative-word delta | loss delta | masked-token delta |',
        '|---|---:|---:|---:|---:|',
    ]
    for m in [f'{x}M' for x in MILESTONES]:
        r = milestone_delta[m]
        lines.append(f"| {m} | {r['step_delta']} | {r['cum_words_delta']} | {r['loss_delta']:+.6f} | {r['masked_tokens_delta']:+d} |")
    lines += ['', '## Scientific reading', result['scientific_reading'], '', f'JSON: `{rel(out_json)}`']
    note.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--research-log', type=Path, default=LOG)
    ap.add_argument('--u256-log', type=Path, default=U256_LOG)
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    ap.add_argument('--note', type=Path, default=NOTE)
    args = ap.parse_args()
    for attr in ['log', 'u256_log', 'out_dir', 'note']:
        v = getattr(args, attr)
        if isinstance(v, Path) and not v.is_absolute():
            setattr(args, attr, USER_ROOT / v)
    result = run(args.log, args.u256_log, args.out_dir, args.note)
    print(json.dumps({
        'status': result['status'],
        'line_counts': result['line_counts'],
        'final_losses': result['final_losses'],
        'out_json': rel(args.out_dir / 'u256_training_log_comparison.json'),
        'note': rel(args.note),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
