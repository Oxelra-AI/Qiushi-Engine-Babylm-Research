#!/usr/bin/env python3
"""Posthoc mechanism analyzer for the compact-view triangle.

Runs after the guarded no-AoA readout has produced per-target payloads for
compact_view_reinvest, compact_repeat_reinvest, and adjbreak_reinvest. It does
not evaluate models. It reads manifests, training logs, triangle summary, the
implementation-equivalence guard, and prediction files referenced by the per-
target payloads. Its purpose is to turn a score table into mechanism-relevant
facts: implementation status, broad-vs-targeted movement, loss/mask dynamics,
and item-level prediction turnover where available.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import time
from collections import Counter
from typing import Any

USER_ROOT = pathlib.Path('.')
TRIANGLE_OUT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval'
DYN_GUARDED = USER_ROOT / 'experiments/archive/representation_and_objectives/data/compact_triangle_dynamics_guarded/triangle_dynamics_guarded_summary.json'
EQUIV_SUMMARY = USER_ROOT / 'experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary.json'
OUT_DIR_DEFAULT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/triangle_posthoc_analysis'
TARGETS = ['compact_view_reinvest', 'compact_repeat_reinvest', 'adjbreak_reinvest']
BROAD = ['BLiMP', 'Supplement', 'COMPS', 'GlobalPIQA_nonparallel', 'Reading']
TARGETED = ['EWoK', 'Entity', 'Entity_full', 'GlobalPIQA_parallel']
RUN_DIRS = {
    'compact_view_reinvest': USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022',
    'compact_repeat_reinvest': USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2',
    'adjbreak_reinvest': USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/gc_adjbreak_reinvest_16k_seed43022_r2',
}


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def maybe_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def per_target_payload(target: str) -> dict[str, Any] | None:
    p = TRIANGLE_OUT / 'per_target' / f'{target}.json'
    return maybe_json(p)


def flatten_predictions_obj(obj: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for group, val in obj.items():
            if isinstance(val, dict) and isinstance(val.get('predictions'), list):
                for item in val['predictions']:
                    if isinstance(item, dict):
                        item_id = str(item.get('id', f'{group}_{len(out)}'))
                        pred = item.get('pred')
                        if pred is not None:
                            out[item_id] = str(pred)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, dict):
                        item_id = str(item.get('id', f'{group}_{i}'))
                        pred = item.get('pred')
                        if pred is not None:
                            out[item_id] = str(pred)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                item_id = str(item.get('id', i))
                pred = item.get('pred')
                if pred is not None:
                    out[item_id] = str(pred)
    return out


def prediction_file_for_task(task_payload: dict[str, Any]) -> pathlib.Path | None:
    out_dir = pathlib.Path(str(task_payload.get('output_dir', '')))
    if not out_dir.exists():
        return None
    # Sentence tasks write predictions.json; Reading has both predictions.json and prediction.jsonl.
    candidates = sorted(out_dir.rglob('predictions.json'), key=lambda p: (len(str(p)), str(p)))
    if candidates:
        return candidates[0]
    candidates = sorted(out_dir.rglob('prediction.jsonl'), key=lambda p: (len(str(p)), str(p)))
    return candidates[0] if candidates else None


def target_predictions(target_payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for col, rec in target_payload.get('tasks', {}).items():
        if not isinstance(rec, dict):
            continue
        p = prediction_file_for_task(rec)
        if p is None:
            continue
        if p.suffix == '.json':
            data = maybe_json(p)
            if data is not None:
                out[col] = flatten_predictions_obj(data)
        elif p.suffix == '.jsonl':
            rows = read_jsonl(p)
            # Reading rows contain Index/Logprob rather than a discrete pred; include a compact signature.
            sigs = {}
            for r in rows:
                idx = r.get('Index')
                if idx is not None:
                    sigs[str(idx)] = json.dumps({k: r.get(k) for k in ['Sentence', 'Word', 'Logprob', 'Prev_Logprob']}, sort_keys=True, ensure_ascii=False)
            out[col] = sigs
    return out


def score_table() -> dict[str, Any]:
    s = maybe_json(TRIANGLE_OUT / 'triangle_noaoa_summary.json')
    return s if isinstance(s, dict) else {}


def mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.mean(vals) if vals else None


def diff(a: float | None, b: float | None) -> float | None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) - float(b)
    return None


def summarize_scores(summary: dict[str, Any]) -> dict[str, Any]:
    table = summary.get('table', {}) if isinstance(summary.get('table'), dict) else {}
    out: dict[str, Any] = {'table': table, 'broad_targeted': {}, 'pairwise': {}}
    for t, scores in table.items():
        if not isinstance(scores, dict):
            continue
        broad_vals = [float(scores[c]) for c in BROAD if isinstance(scores.get(c), (int, float))]
        targ_vals = [float(scores[c]) for c in TARGETED if isinstance(scores.get(c), (int, float))]
        out['broad_targeted'][t] = {
            'broad_mean': mean(broad_vals),
            'targeted_mean': mean(targ_vals),
            'equal7_mean': scores.get('equal7_mean'),
            'equal7_full_entity': scores.get('equal7_full_entity'),
        }
    for a, b in [('compact_view_reinvest', 'compact_repeat_reinvest'), ('compact_view_reinvest', 'adjbreak_reinvest'), ('compact_repeat_reinvest', 'adjbreak_reinvest')]:
        if a in table and b in table:
            sa = table[a]; sb = table[b]
            cols = sorted(set(sa) | set(sb))
            out['pairwise'][f'{a}_minus_{b}'] = {c: diff(sa.get(c), sb.get(c)) for c in cols if diff(sa.get(c), sb.get(c)) is not None}
    return out


def summarize_logs() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t, rd in RUN_DIRS.items():
        rows = read_jsonl(rd / 'training_log.jsonl')
        metrics = maybe_json(rd / 'scientific_metrics.json') or {}
        out[t] = {
            'run_dir': str(rd),
            'n_log_rows': len(rows),
            'first_row': rows[0] if rows else None,
            'last_row': rows[-1] if rows else None,
            'metrics_exists': bool(metrics),
            'word_exposure': metrics.get('word_exposure') if isinstance(metrics, dict) else None,
            'actual_training_steps': metrics.get('actual_training_steps') if isinstance(metrics, dict) else None,
            'loss_first': metrics.get('loss_first') if isinstance(metrics, dict) else None,
            'loss_last': metrics.get('loss_last') if isinstance(metrics, dict) else None,
            'checkpoint_count': len(metrics.get('saved_checkpoints', [])) if isinstance(metrics, dict) else 0,
        }
    # Pairwise final-row differences for gross difficulty / masking sanity.
    pairs = {}
    for a, b in [('compact_view_reinvest', 'compact_repeat_reinvest'), ('compact_view_reinvest', 'adjbreak_reinvest'), ('compact_repeat_reinvest', 'adjbreak_reinvest')]:
        ra = out.get(a, {}).get('last_row') or {}
        rb = out.get(b, {}).get('last_row') or {}
        pairs[f'{a}_minus_{b}'] = {k: diff(ra.get(k), rb.get(k)) for k in ['loss', 'cumulative_word_exposure', 'masked_tokens', 'effective_mask_rate', 'lr'] if diff(ra.get(k), rb.get(k)) is not None}
    out['pairwise_last_row_deltas'] = pairs
    return out


def item_turnover(preds: dict[str, dict[str, dict[str, str]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in sorted(set().union(*(set(p.keys()) for p in preds.values())) if preds else []):
        col_preds = {t: p[col] for t, p in preds.items() if col in p}
        col_out: dict[str, Any] = {'targets_with_predictions': list(col_preds.keys()), 'n_by_target': {t: len(v) for t, v in col_preds.items()}, 'pairs': {}}
        for a, b in [('compact_view_reinvest', 'compact_repeat_reinvest'), ('compact_view_reinvest', 'adjbreak_reinvest'), ('compact_repeat_reinvest', 'adjbreak_reinvest')]:
            if a not in col_preds or b not in col_preds:
                continue
            common = sorted(set(col_preds[a]) & set(col_preds[b]))
            same = sum(1 for k in common if col_preds[a][k] == col_preds[b][k])
            changed = len(common) - same
            examples = []
            for k in common:
                if col_preds[a][k] != col_preds[b][k]:
                    examples.append({'id': k, a: col_preds[a][k], b: col_preds[b][k]})
                if len(examples) >= 8:
                    break
            col_out['pairs'][f'{a}_vs_{b}'] = {
                'common_items': len(common),
                'same_prediction': same,
                'changed_prediction': changed,
                'changed_fraction': changed / len(common) if common else None,
                'first_changed_examples': examples,
            }
        out[col] = col_out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = score_table()
    payloads = {t: per_target_payload(t) for t in TARGETS}
    preds = {t: target_predictions(p) for t, p in payloads.items() if isinstance(p, dict)}
    payload = {
        'status': 'TRIANGLE_POSTHOC_ANALYSIS',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'triangle_summary_path': str(TRIANGLE_OUT / 'triangle_noaoa_summary.json'),
        'guarded_dynamics_path': str(DYN_GUARDED),
        'equivalence_summary_path': str(EQUIV_SUMMARY),
        'implementation_equivalence': summary.get('implementation_equivalence') or maybe_json(EQUIV_SUMMARY),
        'causal_interpretation_allowed': summary.get('causal_interpretation_allowed'),
        'payloads_present': {t: isinstance(p, dict) for t, p in payloads.items()},
        'score_summary': summarize_scores(summary),
        'training_log_summary': summarize_logs(),
        'item_prediction_turnover': item_turnover(preds),
        'mechanism_reading_warning': 'If causal_interpretation_allowed is not true, use this file only as raw descriptive evidence; do not attribute endpoint gaps to compact-view data mechanisms.',
    }
    out_json = out_dir / 'triangle_posthoc_analysis.json'
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    note_lines = [
        '# research triangle posthoc analysis',
        '',
        f"Summary JSON: `{out_json}`",
        '',
        f"Implementation causal guard: `{payload.get('causal_interpretation_allowed')}`",
        '',
        'Payloads present: ' + json.dumps(payload['payloads_present'], ensure_ascii=False),
        '',
        'Broad/targeted means:',
    ]
    for t, bt in payload['score_summary'].get('broad_targeted', {}).items():
        note_lines.append(f"- {t}: broad={bt.get('broad_mean')}, targeted={bt.get('targeted_mean')}, equal7={bt.get('equal7_mean')}")
    note_lines += ['', 'Item prediction turnover columns:']
    for col, rec in payload['item_prediction_turnover'].items():
        note_lines.append(f"- {col}: targets={rec.get('targets_with_predictions')}, n={rec.get('n_by_target')}")
    (out_dir / 'triangle_posthoc_analysis_note.md').write_text('\n'.join(note_lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'out_json': str(out_json), 'payloads_present': payload['payloads_present'], 'causal_interpretation_allowed': payload.get('causal_interpretation_allowed'), 'turnover_columns': list(payload['item_prediction_turnover'].keys())}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
