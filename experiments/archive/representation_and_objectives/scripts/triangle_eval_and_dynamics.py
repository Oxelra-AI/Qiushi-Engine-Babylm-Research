#!/usr/bin/env python3
"""Post-training readout for the compact-view mechanism triangle.

This script is prepared while the two missing matched arms train. It
reuses the frontier_consolidation official-compatible no-AoA evaluator, but patches the
target table to the triangle runs:

- compact_view_reinvest: existing trusted reference.
- compact_repeat_reinvest: exact-repeat reinvest control.
- adjbreak_reinvest: same source/rewrite marginals with source-own adjacency broken.

The scientific interpretation is in `plans/compact_view_triangle_protocol.md`.
This script only evaluates, extracts learning dynamics, and computes contrasts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics
import sys
import time
from typing import Any

USER_ROOT = pathlib.Path('.')
A02_EVAL_SCRIPT = USER_ROOT / 'experiments/archive/frontier_consolidation/scripts/fast_eval_density_arms.py'
OUT_ROOT_DEFAULT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval'
REFERENCE_VIEW_PAYLOAD = USER_ROOT / 'experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/per_target/compact_view_reinvest.json'
REFERENCE_VIEW_RUN = USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022'
REPEAT_RUN = USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2'
ADJBREAK_RUN = USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/gc_adjbreak_reinvest_16k_seed43022_r2'
IMPLEMENTATION_EQUIV_SUMMARY = USER_ROOT / 'experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary.json'

SEED_BANDS = {
    'overall_spread': 0.7848952009527181,
    'full7_spread': 0.8468886294750527,
    'fast_equal7_spread': 1.355714285714285,
}

TRIANGLE_TARGETS = {
    'compact_view_reinvest': {
        'run_dir': REFERENCE_VIEW_RUN,
        'endpoint': 'chck_100M',
        'arm': 'cleanqwen_fineweb_compact_view_reinvest',
        'description': 'Existing same-source compact-view reinvest reference; source plus its own compact rewrite, saved words reinvested into more pairs.',
    },
    'compact_repeat_reinvest': {
        'run_dir': REPEAT_RUN,
        'endpoint': 'chck_100M',
        'arm': 'cleanqwen_fineweb_repeat_compact_reinvest',
        'description': 'Exact-repeat reinvest control; tests whether generated semantic second views matter beyond repeated source exposure and reinvested breadth.',
    },
    'adjbreak_reinvest': {
        'run_dir': ADJBREAK_RUN,
        'endpoint': 'chck_100M',
        'arm': 'cleanqwen_fineweb_compact_view_reinvest_adjbreak',
        'description': 'Source/rewrite marginal-preserving control with source-own rewrite adjacency broken; tests local source-own consolidation.',
    },
}

DEFAULT_COLUMNS = [
    'BLiMP', 'Supplement', 'EWoK', 'Entity', 'Entity_full', 'COMPS',
    'GlobalPIQA_parallel', 'GlobalPIQA_nonparallel', 'Reading',
]
BROAD_COLUMNS = ['BLiMP', 'Supplement', 'COMPS', 'GlobalPIQA_nonparallel', 'Reading']
TARGETED_COLUMNS = ['EWoK', 'Entity', 'Entity_full', 'GlobalPIQA_parallel']


def load_eval_module():
    if not A02_EVAL_SCRIPT.exists():
        raise FileNotFoundError(A02_EVAL_SCRIPT)
    spec = importlib.util.spec_from_file_location('density_eval', A02_EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'could not import {A02_EVAL_SCRIPT}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['density_eval'] = mod
    spec.loader.exec_module(mod)
    mod.TARGETS = TRIANGLE_TARGETS
    return mod


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def implementation_equivalence_status() -> dict[str, Any]:
    out: dict[str, Any] = {
        'summary_path': str(IMPLEMENTATION_EQUIV_SUMMARY),
        'exists': IMPLEMENTATION_EQUIV_SUMMARY.exists(),
        'required_before_causal_interpretation': True,
        'allows_historical_reference_for_causal_triangle': False,
    }
    if not IMPLEMENTATION_EQUIV_SUMMARY.exists():
        out['interpretation'] = 'missing_gc_reference_equivalence__raw_scores_only'
        out['rule'] = 'Run compact_view_gc_reference_equivalence.py to chck_1M before interpreting gaps against the historical compact_view reference.'
        return out
    try:
        payload = read_json(IMPLEMENTATION_EQUIV_SUMMARY)
    except Exception as exc:
        out['interpretation'] = 'unreadable_gc_reference_equivalence__raw_scores_only'
        out['read_error'] = repr(exc)
        return out
    comp = payload.get('log_comparison', {}) if isinstance(payload, dict) else {}
    interp = payload.get('interpretation') if isinstance(payload, dict) else None
    out.update({
        'status': payload.get('status') if isinstance(payload, dict) else None,
        'interpretation': interp,
        'allows_historical_reference_for_causal_triangle': interp == 'trajectory_equivalent_to_checked_horizon',
        'n_compared_steps': comp.get('n_compared_steps') if isinstance(comp, dict) else None,
        'max_abs_delta': comp.get('max_abs_delta') if isinstance(comp, dict) else None,
        'gc_reference_run_dir': payload.get('run_dir') if isinstance(payload, dict) else None,
        'triangle_use_rule': payload.get('triangle_use_rule') if isinstance(payload, dict) else None,
    })
    return out


def per_target_path(out_root: pathlib.Path, target: str) -> pathlib.Path:
    return out_root / 'per_target' / f'{target}.json'


def maybe_copy_reference_view(out_root: pathlib.Path) -> pathlib.Path:
    if not REFERENCE_VIEW_PAYLOAD.exists():
        raise FileNotFoundError(f'missing reference view payload: {REFERENCE_VIEW_PAYLOAD}')
    out = per_target_path(out_root, 'compact_view_reinvest')
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = read_json(REFERENCE_VIEW_PAYLOAD)
    payload['reference_reused_from'] = str(REFERENCE_VIEW_PAYLOAD)
    payload['reference_reuse_note'] = 'Copied by research triangle script to avoid re-evaluating the unchanged compact_view_reinvest reference.'
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return out


def read_training_log_tail(path: pathlib.Path, n: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows[-n:]


def summarize_training(run_dir: pathlib.Path) -> dict[str, Any]:
    metrics_path = run_dir / 'scientific_metrics.json'
    log_path = run_dir / 'training_log.jsonl'
    dyn_path = run_dir / 'dynamics_traces.jsonl'
    endpoint = run_dir / 'hf_model' / 'chck_100M'
    out: dict[str, Any] = {
        'run_dir': str(run_dir),
        'metrics_exists': metrics_path.exists(),
        'training_log_exists': log_path.exists(),
        'dynamics_traces_exists': dyn_path.exists(),
        'endpoint_path': str(endpoint),
        'endpoint_exists': (endpoint / 'model.safetensors').exists(),
        'training_log_tail': read_training_log_tail(log_path, n=5),
    }
    if metrics_path.exists():
        m = read_json(metrics_path)
        for k in [
            'variant', 'backend', 'model_family', 'parameter_count', 'vocab_size',
            'tokenizer_label', 'word_exposure', 'loss_first', 'loss_last',
            'actual_training_steps', 'masking_curriculum', 'mask_prob_start',
            'mask_prob_end', 'hidden_size', 'n_layer', 'n_head', 'ffn_mult',
            'seed', 'extra_init_seed', 'train_rng_seed', 'seq_length', 'max_seq_length',
            'data_source_type', 'example_jsonl', 'example_jsonl_label',
        ]:
            if k in m:
                out[k] = m[k]
        cps = m.get('saved_checkpoints', [])
        out['saved_checkpoint_count'] = len(cps)
        out['first_checkpoint'] = cps[0] if cps else None
        out['last_checkpoint'] = cps[-1] if cps else None
        out['dynamics_traces_file'] = m.get('dynamics_traces_file')
    if dyn_path.exists():
        count = 0
        nonzero_entropy = 0
        mask_rates = []
        for line in dyn_path.read_text(encoding='utf-8', errors='replace').splitlines():
            if not line.strip():
                continue
            count += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            em = rec.get('prediction_entropy_mean')
            if em not in (None, 0, 0.0):
                nonzero_entropy += 1
            mr = rec.get('effective_mask_rate_mean')
            if isinstance(mr, (int, float)):
                mask_rates.append(float(mr))
        out['dynamics_trace_count'] = count
        out['dynamics_trace_nonzero_entropy_count'] = nonzero_entropy
        if mask_rates:
            out['dynamics_mask_rate_mean'] = statistics.mean(mask_rates)
    cfg = endpoint / 'config.json'
    if cfg.exists():
        try:
            out['endpoint_config_step202_activation_checkpointing'] = read_json(cfg).get('activation_checkpointing')
        except Exception:
            pass
    return out


def scores_from_payload(eval_mod, path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return eval_mod.target_scores_from_payload(read_json(path))


def diff_scores(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    out = {}
    for k in sorted(set(a) | set(b)):
        av = a.get(k); bv = b.get(k)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            out[k] = round(float(av) - float(bv), 6)
    return out


def mean_existing(scores: dict[str, Any], cols: list[str]) -> float | None:
    vals = [float(scores[c]) for c in cols if isinstance(scores.get(c), (int, float))]
    return statistics.mean(vals) if vals else None


def build_triangle_summary(eval_mod, out_root: pathlib.Path, targets: list[str]) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    implementation_equivalence = implementation_equivalence_status()
    training = {t: summarize_training(pathlib.Path(TRIANGLE_TARGETS[t]['run_dir'])) for t in TRIANGLE_TARGETS}
    raw_paths = {t: per_target_path(out_root, t) for t in TRIANGLE_TARGETS}
    table = {t: scores_from_payload(eval_mod, p) for t, p in raw_paths.items() if p.exists()}
    table = {t: s for t, s in table.items() if s is not None}
    contrasts: dict[str, Any] = {}
    if 'compact_view_reinvest' in table and 'compact_repeat_reinvest' in table:
        contrasts['view_minus_repeat_reinvest'] = diff_scores(table['compact_view_reinvest'], table['compact_repeat_reinvest'])
    if 'compact_view_reinvest' in table and 'adjbreak_reinvest' in table:
        d = diff_scores(table['compact_view_reinvest'], table['adjbreak_reinvest'])
        contrasts['view_minus_adjbreak_reinvest'] = d
        contrasts['adjbreak_profile'] = {
            'broad_delta_mean_view_minus_adjbreak': mean_existing(d, BROAD_COLUMNS),
            'targeted_delta_mean_view_minus_adjbreak': mean_existing(d, TARGETED_COLUMNS),
            'reading_rule': 'If broad columns all drop strongly, adjbreak may be a generic incoherence control rather than a clean source-own consolidation ablation.',
        }
    if 'compact_repeat_reinvest' in table and 'adjbreak_reinvest' in table:
        contrasts['repeat_minus_adjbreak_reinvest'] = diff_scores(table['compact_repeat_reinvest'], table['adjbreak_reinvest'])
    payload = {
        'status': 'COMPACT_TRIANGLE_NOAOA_SUMMARY',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': 'Fast official-compatible no-AoA triangle readout plus training dynamics. Interpret with the research protocol, seed-spread bands, and the research implementation-equivalence guard.',
        'seed_spread_bands': SEED_BANDS,
        'implementation_equivalence': implementation_equivalence,
        'causal_interpretation_allowed': bool(implementation_equivalence.get('allows_historical_reference_for_causal_triangle')),
        'targets_requested_this_run': targets,
        'target_definitions': {t: {**v, 'run_dir': str(v['run_dir'])} for t, v in TRIANGLE_TARGETS.items()},
        'training': training,
        'table': table,
        'contrasts': contrasts,
        'raw_paths': {t: str(p) for t, p in raw_paths.items() if p.exists()},
        'out_root': str(out_root),
    }
    (out_root / 'triangle_noaoa_summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_note(payload, out_root)
    return payload


def fmt(v: Any) -> str:
    if v is None:
        return ''
    if isinstance(v, float):
        return f'{v:.3f}'
    return str(v)


def write_note(summary: dict[str, Any], out_root: pathlib.Path) -> None:
    keys = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'Entity_full', 'COMPS', 'GlobalPIQA_mean', 'Reading', 'equal7_mean', 'equal7_full_entity']
    lines = [
        '# research compact-view triangle no-AoA readout',
        '',
        f"Summary JSON: `{out_root / 'triangle_noaoa_summary.json'}`",
        '',
        'This is a mechanism readout, not a final endpoint package. It compares compact-view reinvestment against exact-repeat reinvestment and source-own adjacency breaking under the same DeBERTa-v2 8x480/baseline16k/WWM/AdamW coordinate.',
        '',
        '| target | ' + ' | '.join(keys) + ' |',
        '|---|' + '|'.join(['---:'] * len(keys)) + '|',
    ]
    for t, scores in summary.get('table', {}).items():
        lines.append('| ' + t + ' | ' + ' | '.join(fmt(scores.get(k)) for k in keys) + ' |')
    lines += ['', '## Training states', '']
    for t, tr in summary.get('training', {}).items():
        lines.append(f"- **{t}**: metrics={tr.get('metrics_exists')}, endpoint={tr.get('endpoint_exists')}, word_exposure={tr.get('word_exposure')}, steps={tr.get('actual_training_steps')}, loss_first={tr.get('loss_first')}, loss_last={tr.get('loss_last')}, checkpoints={tr.get('saved_checkpoint_count')}")
    lines += ['', '## Mechanism contrasts', '']
    for name, delta in summary.get('contrasts', {}).items():
        lines.append(f'- **{name}**: `{json.dumps(delta, ensure_ascii=False)}`')
    lines += [
        '',
        '## Interpretation reminders',
        '',
        f"- Fast equal7 gaps below {SEED_BANDS['fast_equal7_spread']:.3f} are not a single-seed mechanism verdict without full scoring or a tie-breaking seed.",
        '- If `adjbreak_reinvest` drops across nearly every broad column, it may be a destructive incoherence control rather than evidence specifically for source-own rewrite consolidation.',
    ]
    impl = summary.get('implementation_equivalence', {})
    lines += [
        '',
        '## Implementation-equivalence guard',
        '',
        f"- Summary path: `{impl.get('summary_path')}`",
        f"- Interpretation: `{impl.get('interpretation')}`",
        f"- Allows historical compact-view reference for causal triangle interpretation: `{impl.get('allows_historical_reference_for_causal_triangle')}`",
        '- If this guard is false, the table may be used only as raw score evidence; train/evaluate `compact_view_reinvest` under the same GC implementation before mechanism interpretation.',
    ]
    (out_root / 'triangle_noaoa_note.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def preflight(eval_mod, out_root: pathlib.Path, copy_reference_view: bool) -> dict[str, Any]:
    copied = None
    if copy_reference_view:
        copied = str(maybe_copy_reference_view(out_root))
    payload = {
        'status': 'TRIANGLE_EVAL_PREFLIGHT',
        'copied_reference_view_payload': copied,
        'reference_view_payload_exists': REFERENCE_VIEW_PAYLOAD.exists(),
        'targets': {},
    }
    for t, cfg in TRIANGLE_TARGETS.items():
        run_dir = pathlib.Path(cfg['run_dir'])
        endpoint = run_dir / 'hf_model' / str(cfg['endpoint'])
        payload['targets'][t] = {
            'run_dir': str(run_dir),
            'run_dir_exists': run_dir.exists(),
            'metrics_exists': (run_dir / 'scientific_metrics.json').exists(),
            'endpoint': str(endpoint),
            'endpoint_exists': (endpoint / 'model.safetensors').exists(),
            'per_target_payload': str(per_target_path(out_root, t)),
            'per_target_payload_exists': per_target_path(out_root, t).exists(),
        }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='*', default=['compact_repeat_reinvest', 'adjbreak_reinvest'], choices=sorted(TRIANGLE_TARGETS))
    ap.add_argument('--columns', nargs='*', default=DEFAULT_COLUMNS)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out-root', default=str(OUT_ROOT_DEFAULT))
    ap.add_argument('--work-tag', default='compact_triangle_noaoa')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--copy-reference-view', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()

    eval_mod = load_eval_module()
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.preflight:
        print(json.dumps(preflight(eval_mod, out_root, args.copy_reference_view), indent=2, ensure_ascii=False))
        return
    if args.copy_reference_view:
        maybe_copy_reference_view(out_root)
    if not args.summary_only:
        for target in args.targets:
            print(json.dumps({'event': 'triangle_target_start', 'target': target, 'gpu': args.gpu}, ensure_ascii=False), flush=True)
            eval_mod.eval_target(target, args.gpu, args.work_tag, args.columns, out_root, force=args.force)
    summary = build_triangle_summary(eval_mod, out_root, list(args.targets))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
