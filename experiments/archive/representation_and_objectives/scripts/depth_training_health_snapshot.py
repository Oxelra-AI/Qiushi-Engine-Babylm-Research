#!/usr/bin/env python3
"""research: health snapshot for the running legal40k 12x384 depth training.

CPU/read-only analysis of training logs. This is not a downstream selector: prior
work showed MLM loss is not reliable for choosing endpoint winners. Its purpose is
narrower: catch execution/optimization anomalies before the official evaluation
runs (wrong schedule, missing masked targets, broken exposure accounting, extreme
loss relative to matched baselines).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
OUT_DIR = WS / 'data/depth_training_health_snapshot'
OUT_JSON = OUT_DIR / 'depth_training_health_snapshot.json'
OUT_MD = (ROOT / 'research/notes/representation_and_objectives/depth_training_health_snapshot.md')

RUNS = {
    'depth12x384_43022_current': WS / 'training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/training_log.jsonl',
    'legal40k8x480_43022_complete': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/training_log.jsonl',
    'legal40k8x480_43122_complete': WS / 'training/runs/legal40k_accum_compact_view_reinvest_seed43122/training_log.jsonl',
}


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def load_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            if isinstance(rec, dict) and 'step' in rec and 'cumulative_word_exposure' in rec:
                rows.append(rec)
    return rows


def first_ge(rows: list[dict[str, Any]], words: int) -> dict[str, Any] | None:
    for r in rows:
        if int(r.get('cumulative_word_exposure', 0)) >= words:
            return r
    return None


def window(rows: list[dict[str, Any]], lo_words: int, hi_words: int) -> list[dict[str, Any]]:
    return [r for r in rows if lo_words <= int(r.get('cumulative_word_exposure', 0)) <= hi_words]


def safe_mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    return mean(vals) if vals else None


def summarize(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {'label': label, 'records': 0}
    last = rows[-1]
    total_words = int(last.get('cumulative_word_exposure', 0))
    checkpoints = list(range(10_000_000, min(100_000_000, (total_words // 10_000_000) * 10_000_000) + 1, 10_000_000))
    by10 = []
    for w in checkpoints:
        r = first_ge(rows, w)
        if r:
            win = window(rows, max(0, w - 2_000_000), w)
            by10.append({
                'target_words': w,
                'step': r.get('step'),
                'actual_words': r.get('cumulative_word_exposure'),
                'loss_at_or_after': r.get('loss'),
                'lr': r.get('lr'),
                'mean_loss_last_2M_words': safe_mean([float(x.get('loss')) for x in win if x.get('loss') is not None]),
                'mean_mask_rate_last_2M_words': safe_mean([float(x.get('effective_mask_rate')) for x in win if x.get('effective_mask_rate') is not None]),
                'mean_masked_tokens_last_2M_words': safe_mean([float(x.get('masked_tokens')) for x in win if x.get('masked_tokens') is not None]),
            })
    return {
        'label': label,
        'records': len(rows),
        'last': last,
        'last_words_millions': total_words / 1_000_000,
        'first_record': rows[0],
        'by_10M': by10,
    }


def compare_at_words(summaries: dict[str, dict[str, Any]], words: int) -> dict[str, Any]:
    out: dict[str, Any] = {'target_words': words}
    vals = {}
    for label, s in summaries.items():
        rows10 = {int(x['target_words']): x for x in s.get('by_10M', [])}
        if words in rows10:
            vals[label] = rows10[words]
    out['available'] = vals
    depth = vals.get('depth12x384_43022_current')
    base = vals.get('legal40k8x480_43022_complete')
    if depth and base:
        for metric in ['loss_at_or_after', 'mean_loss_last_2M_words', 'lr', 'mean_mask_rate_last_2M_words', 'mean_masked_tokens_last_2M_words']:
            a = depth.get(metric)
            b = base.get(metric)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                out[f'depth_minus_8x480_seed43022_{metric}'] = a - b
                if b != 0:
                    out[f'depth_over_8x480_seed43022_{metric}_ratio'] = a / b
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = {label: load_log(path) for label, path in RUNS.items()}
    summaries = {label: summarize(label, rows) for label, rows in logs.items()}
    depth_words = int((summaries.get('depth12x384_43022_current', {}).get('last') or {}).get('cumulative_word_exposure', 0))
    common_10m = (depth_words // 10_000_000) * 10_000_000
    comparisons = [compare_at_words(summaries, w) for w in range(10_000_000, common_10m + 1, 10_000_000)]

    anomalies: list[str] = []
    dsum = summaries.get('depth12x384_43022_current', {})
    last = dsum.get('last') or {}
    if not dsum.get('records'):
        anomalies.append('depth training log missing or empty')
    else:
        if int(last.get('cumulative_word_exposure', 0)) < 50_000_000:
            anomalies.append('depth has not yet reached 50M; health snapshot is still early')
        if last.get('seq_len') != 256:
            anomalies.append(f"depth seq_len unexpected: {last.get('seq_len')}")
        if last.get('mask_mode') != 'wwm':
            anomalies.append(f"depth mask_mode unexpected: {last.get('mask_mode')}")
        if last.get('active_microbatches') != 4:
            anomalies.append(f"depth active_microbatches unexpected: {last.get('active_microbatches')}")
        mr = last.get('effective_mask_rate')
        if isinstance(mr, (int, float)) and not (0.12 <= mr <= 0.18):
            anomalies.append(f"depth latest effective_mask_rate outside loose WWM band: {mr}")
        lr = last.get('lr')
        if isinstance(lr, (int, float)) and lr <= 0:
            anomalies.append(f"depth latest lr nonpositive: {lr}")

    # Compare 60M if available; flag only severe divergence because loss is not an endpoint selector.
    for comp in comparisons:
        ratio = comp.get('depth_over_8x480_seed43022_mean_loss_last_2M_words_ratio')
        if isinstance(ratio, (int, float)) and ratio > 1.25:
            anomalies.append(f"depth mean loss in last 2M at {comp['target_words']} words is >25% above matched 8x480 baseline")

    payload = {
        'status': 'DEPTH_TRAINING_HEALTH_SNAPSHOT',
        'purpose': 'Optimization/accounting health only; not a downstream route selector.',
        'run_paths': {k: rel(v) for k, v in RUNS.items()},
        'summaries': summaries,
        'matched_10M_comparisons': comparisons,
        'current_common_10M_word_mark': common_10m,
        'anomalies': anomalies,
        'reading': [
            'Loss is used here only to detect execution anomalies, because prior steps showed MLM loss and early checkpoints do not select downstream winners.',
            'If no severe anomaly is present, the official full evaluation remains the decision-making evidence for depth.',
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    lines = ['# research — Depth training health snapshot\n\n']
    lines.append('CPU-only log analysis. This does not infer downstream score and does not replace official evaluation.\n\n')
    if dsum.get('records'):
        lines.append(f"Depth current: step `{last.get('step')}`, words `{last.get('cumulative_word_exposure')}`, loss `{float(last.get('loss')):.4f}`, lr `{float(last.get('lr')):.6g}`, mask rate `{float(last.get('effective_mask_rate')):.4f}`.\n\n")
    lines.append('## Matched 10M comparisons against legal40k 8x480 seed43022\n')
    for comp in comparisons:
        w = comp['target_words']
        dm = comp.get('depth_minus_8x480_seed43022_mean_loss_last_2M_words')
        dr = comp.get('depth_over_8x480_seed43022_mean_loss_last_2M_words_ratio')
        if dm is not None:
            lines.append(f"- {w//1_000_000}M: depth mean loss(last2M) minus 8x480 = `{dm:+.4f}` (ratio `{dr:.4f}`).\n")
    lines.append('\n## Anomalies\n')
    if anomalies:
        for a in anomalies:
            lines.append(f"- {a}\n")
    else:
        lines.append('- No severe execution/optimization anomaly detected in the current log snapshot.\n')
    lines.append(f"\nJSON: `{rel(OUT_JSON)}`\n")
    OUT_MD.write_text(''.join(lines), encoding='utf-8')

    print(json.dumps({
        'status': payload['status'],
        'depth_words_millions': dsum.get('last_words_millions'),
        'current_common_10M_word_mark': common_10m,
        'anomaly_count': len(anomalies),
        'out_json': rel(OUT_JSON),
        'note': rel(OUT_MD),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
