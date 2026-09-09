#!/usr/bin/env python3
"""research: compact-view activation-checkpoint implementation-equivalence check.

Why this exists
---------------
The research repair makes the two new compact-view triangle controls use an
activation-checkpointed wrapper, while the compact_view_reinvest reference was
trained historically with the original COMPACT_EXPERIENCE trainer. Before endpoint gaps in
the triangle are interpreted as data-mechanism evidence, the implementation
repair itself must be measured on the *reference data* at a meaningful early
checkpoint.

The cheapest reliable comparison is to train compact_view_reinvest with the
checkpointed wrapper to the row-boundary cumulative exposure at which the
historical run saved chck_1M (default: 1,027,470 words, ~26 optimizer updates)
under the exact research recipe, then compare its training log and saved
checkpoint metadata to the historical run. The row-boundary value is necessary
because the COMPACT_EXPERIENCE example_jsonl loader refuses partial examples. If that early
trajectory diverges materially, the full compact_view arm should be retrained
under the same wrapper before reading the triangle causally.

This script is intentionally narrow. It does not evaluate BabyLM tasks and does
not interpret the mechanism; it only launches/compares the implementation
control.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

USER_ROOT = pathlib.Path('.')
GC_TRAINER = USER_ROOT / 'experiments/archive/representation_and_objectives/scripts/gradient_checkpointed_masking_curriculum_trainer.py'
ORIG_TRAINER = USER_ROOT / 'experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py'
TOKENIZER = USER_ROOT / 'experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model'
REFERENCE_RUN = USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022'
REFERENCE_TRAIN_FILE = USER_ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
REFERENCE_META = USER_ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json'
DEFAULT_RUN_DIR = USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/gc_compact_view_reinvest_1M_seed43022'
DEFAULT_OUT_DIR = USER_ROOT / 'experiments/archive/representation_and_objectives/data/gc_reference_equivalence'
# Historical research crosses and saves chck_1M after 26 updates at this exact
# row-boundary exposure. A literal 1,000,000-word cap would require a partial
# example and fail under the trusted trainer's example_jsonl loader.
DEFAULT_ROW_BOUNDARY_1M_EXPOSURE = 1_027_470

FIXED_ARGS = [
    '--example_jsonl', str(REFERENCE_TRAIN_FILE),
    '--example_jsonl_label', 'cleanqwen_fineweb_compact_view_reinvest',
    '--example_jsonl_meta', str(REFERENCE_META),
    '--tokenizer_path', str(TOKENIZER),
    '--tokenizer_label', 'baseline16k',
    '--hidden_size', '480',
    '--n_layer', '8',
    '--n_head', '8',
    '--ffn_mult', '4',
    '--seed', '43',
    '--extra_init_seed', '43022',
    '--train_rng_seed', '43023',
    '--batch_size', '256',
    '--seq_length', '256',
    '--max_seq_length', '256',
    '--learning_rate', '0.001',
    '--warmup_fraction', '0.06',
    '--weight_decay', '0.01',
    '--masking_curriculum', 'wwm_fixed',
    '--mask_prob_start', '0.15',
    '--mask_prob_end', '0.15',
    '--checkpoint_words', '1000000',
    '--num_workers', '0',
    '--log_every', '1',
    '--dynamics_trace_every', '200',
]

NUMERIC_LOG_KEYS = ['loss', 'lr', 'batch_words', 'cumulative_word_exposure', 'masked_tokens', 'effective_mask_rate']


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


def first_reference_steps(n: int) -> list[dict[str, Any]]:
    return read_jsonl(REFERENCE_RUN / 'training_log.jsonl')[:n]


def metrics_summary(run_dir: pathlib.Path) -> dict[str, Any]:
    p = run_dir / 'scientific_metrics.json'
    if not p.exists():
        return {'metrics_exists': False, 'metrics_path': str(p)}
    m = read_json(p)
    cps = m.get('saved_checkpoints', [])
    out = {
        'metrics_exists': True,
        'metrics_path': str(p),
        'word_exposure': m.get('word_exposure'),
        'actual_training_steps': m.get('actual_training_steps'),
        'loss_first': m.get('loss_first'),
        'loss_last': m.get('loss_last'),
        'parameter_count': m.get('parameter_count'),
        'vocab_size': m.get('vocab_size'),
        'tokenizer_label': m.get('tokenizer_label'),
        'seed': m.get('seed'),
        'extra_init_seed': m.get('extra_init_seed'),
        'train_rng_seed': m.get('train_rng_seed'),
        'saved_checkpoint_count': len(cps),
        'first_checkpoint': cps[0] if cps else None,
        'last_checkpoint': cps[-1] if cps else None,
    }
    cfg = run_dir / 'hf_model/chck_1M/config.json'
    if cfg.exists():
        try:
            out['chck_1M_activation_checkpointing_flag'] = read_json(cfg).get('activation_checkpointing')
        except Exception as exc:
            out['chck_1M_config_read_error'] = repr(exc)
    out['chck_1M_model_exists'] = (run_dir / 'hf_model/chck_1M/model.safetensors').exists()
    return out


def log_compare(gc_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = min(len(gc_rows), len(reference_rows))
    per_step: list[dict[str, Any]] = []
    max_abs: dict[str, float] = {k: 0.0 for k in NUMERIC_LOG_KEYS}
    mismatch_count = 0
    for i in range(n):
        g = gc_rows[i]
        r = reference_rows[i]
        rec: dict[str, Any] = {'index': i, 'step_gc': g.get('step'), 'step_reference': r.get('step')}
        exact_row = True
        for k in NUMERIC_LOG_KEYS:
            gv = g.get(k); rv = r.get(k)
            if isinstance(gv, (int, float)) and isinstance(rv, (int, float)):
                delta = float(gv) - float(rv)
                rec[k + '_gc'] = gv
                rec[k + '_reference'] = rv
                rec[k + '_delta'] = delta
                if abs(delta) > max_abs[k]:
                    max_abs[k] = abs(delta)
                if abs(delta) > 1e-12:
                    exact_row = False
        rec['numeric_exact_for_checked_keys'] = exact_row
        if not exact_row:
            mismatch_count += 1
        if i < 5 or i == n - 1:
            per_step.append(rec)
    return {
        'n_compared_steps': n,
        'gc_steps_available': len(gc_rows),
        'reference_steps_available': len(reference_rows),
        'numeric_mismatch_count_at_tol_1e-12': mismatch_count,
        'max_abs_delta': max_abs,
        'selected_step_records': per_step,
    }


def build_command(run_dir: pathlib.Path, max_word_exposure: int, trainer: pathlib.Path) -> list[str]:
    return [sys.executable, '-B', str(trainer), *FIXED_ARGS, '--output_dir', str(run_dir), '--max_word_exposure', str(max_word_exposure)]


def preflight(run_dir: pathlib.Path, out_dir: pathlib.Path, max_word_exposure: int) -> dict[str, Any]:
    required = [GC_TRAINER, ORIG_TRAINER, TOKENIZER / 'tokenizer.json', TOKENIZER / 'tokenizer_config.json', REFERENCE_TRAIN_FILE, REFERENCE_META, REFERENCE_RUN / 'training_log.jsonl', REFERENCE_RUN / 'scientific_metrics.json']
    missing = [str(p) for p in required if not p.exists()]
    reference_rows = first_reference_steps(1000000)
    target_step = None
    target_ref = None
    for r in reference_rows:
        if int(r.get('cumulative_word_exposure', -1)) >= max_word_exposure:
            target_step = int(r.get('step'))
            target_ref = r
            break
    return {
        'status': 'GC_REFERENCE_EQUIVALENCE_PREFLIGHT',
        'missing_required_paths': missing,
        'run_dir': str(run_dir),
        'out_dir': str(out_dir),
        'max_word_exposure': max_word_exposure,
        'reference_run': str(REFERENCE_RUN),
        'reference_train_file': str(REFERENCE_TRAIN_FILE),
        'reference_rows_available': len(reference_rows),
        'reference_first_row': reference_rows[0] if reference_rows else None,
        'reference_row_reaching_target': target_ref,
        'expected_steps_to_target_from_reference': target_step,
        'command': build_command(run_dir, max_word_exposure, GC_TRAINER),
        'scientific_rule': 'If the GC compact-view reference differs materially from the historical compact-view trajectory by the early checkpoint, do not interpret triangle endpoint gaps until compact_view is trained/evaluated under the same GC implementation.',
    }


def run_training(run_dir: pathlib.Path, max_word_exposure: int, gpu: int, allow_nonempty: bool) -> dict[str, Any]:
    if run_dir.exists() and any(run_dir.iterdir()) and not allow_nonempty:
        raise RuntimeError(f'run_dir exists and is non-empty: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(run_dir, max_word_exposure, GC_TRAINER)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env['TMPDIR'] = f'/tmp/q_representation_and_objectives_step203_gc_view_equiv_{int(time.time())}'
    pathlib.Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    command_record = {
        'status': 'GC_REFERENCE_EQUIVALENCE_TRAIN_COMMAND',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'gpu': gpu,
        'run_dir': str(run_dir),
        'max_word_exposure': max_word_exposure,
        'command': cmd,
    }
    (run_dir / 'train_command.json').write_text(json.dumps(command_record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    stdout_path = run_dir / 'train_stdout.log'
    stderr_path = run_dir / 'train_stderr.log'
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with stdout_path.open('w', encoding='utf-8') as out, stderr_path.open('w', encoding='utf-8') as err:
        out.write(json.dumps({'event': 'gc_reference_start', 'started_utc': started, 'gpu': gpu, 'max_word_exposure': max_word_exposure}) + '\n')
        out.flush()
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env)
    finished = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    result = {
        'status': 'GC_REFERENCE_EQUIVALENCE_TRAIN_FINISHED' if proc.returncode == 0 else 'GC_REFERENCE_EQUIVALENCE_TRAIN_FAILED',
        'returncode': proc.returncode,
        'started_utc': started,
        'finished_utc': finished,
        'run_dir': str(run_dir),
        'stdout_log': str(stdout_path),
        'stderr_log': str(stderr_path),
        'metrics_path': str(run_dir / 'scientific_metrics.json'),
        'metrics_exists': (run_dir / 'scientific_metrics.json').exists(),
    }
    (run_dir / 'launcher_result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return result


def compare_and_write(run_dir: pathlib.Path, out_dir: pathlib.Path, max_word_exposure: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gc_rows = read_jsonl(run_dir / 'training_log.jsonl')
    ref_rows = first_reference_steps(max(len(gc_rows), 1))
    comp = log_compare(gc_rows, ref_rows)
    metrics = metrics_summary(run_dir)
    ref_metrics = metrics_summary(REFERENCE_RUN)
    # A deliberately tight default: any exact row mismatch means the repaired implementation
    # is no longer proven bit-identical at this horizon. Later agents may still decide a tiny
    # numerical-only mismatch is harmless, but it must not be hidden.
    all_exact = comp['numeric_mismatch_count_at_tol_1e-12'] == 0 and comp['n_compared_steps'] == metrics.get('actual_training_steps')
    if not metrics.get('metrics_exists'):
        interpretation = 'missing_gc_metrics'
    elif comp['n_compared_steps'] == 0:
        interpretation = 'missing_training_logs'
    elif all_exact:
        interpretation = 'trajectory_equivalent_to_checked_horizon'
    else:
        interpretation = 'trajectory_diverged_before_or_at_checked_horizon__retrain_view_under_gc_before_triangle_interpretation'
    payload = {
        'status': 'GC_REFERENCE_EQUIVALENCE_COMPARISON',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'question': 'Does the activation-checkpointed implementation reproduce the historical compact_view_reinvest trajectory on the reference data at an early official checkpoint horizon?',
        'run_dir': str(run_dir),
        'reference_run': str(REFERENCE_RUN),
        'max_word_exposure_requested': max_word_exposure,
        'gc_metrics': metrics,
        'reference_metrics_summary': {k: ref_metrics.get(k) for k in ['metrics_exists', 'word_exposure', 'actual_training_steps', 'loss_first', 'loss_last', 'parameter_count', 'vocab_size', 'tokenizer_label', 'saved_checkpoint_count', 'first_checkpoint']},
        'log_comparison': comp,
        'interpretation': interpretation,
        'triangle_use_rule': 'Endpoint gaps from the two research GC-trained controls may be interpreted against the historical compact_view reference only if this comparison is trajectory-equivalent to the checked horizon. Otherwise train compact_view_reinvest under the same GC wrapper to 100M or treat the triangle as implementation-confounded.',
    }
    out_json = out_dir / 'gc_reference_equivalence_summary.json'
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = [
        '# research compact-view GC reference equivalence',
        '',
        f"Summary JSON: `{out_json}`",
        '',
        f"Interpretation: **{interpretation}**",
        '',
        f"GC run: `{run_dir}`",
        f"Historical reference: `{REFERENCE_RUN}`",
        f"Compared steps: {comp['n_compared_steps']}",
        '',
        'Max absolute deltas:',
    ]
    for k, v in comp.get('max_abs_delta', {}).items():
        lines.append(f'- `{k}`: {v}')
    lines += ['', 'Selected rows:', '']
    for rec in comp.get('selected_step_records', []):
        lines.append(f"- step gc/ref {rec.get('step_gc')}/{rec.get('step_reference')}: loss_delta={rec.get('loss_delta')}, cum_words_delta={rec.get('cumulative_word_exposure_delta')}, masked_tokens_delta={rec.get('masked_tokens_delta')}, exact={rec.get('numeric_exact_for_checked_keys')}")
    lines += ['', 'Rule: if this file reports divergence, put the compact-view arm on the same GC implementation before using endpoint gaps as mechanism evidence.']
    (out_dir / 'gc_reference_equivalence_note.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'out_json': str(out_json), 'interpretation': interpretation, 'n_compared_steps': comp['n_compared_steps'], 'max_abs_delta': comp['max_abs_delta']}, indent=2, ensure_ascii=False))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=str(DEFAULT_RUN_DIR))
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--max-word-exposure', type=int, default=DEFAULT_ROW_BOUNDARY_1M_EXPOSURE)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--allow-nonempty', action='store_true')
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    out_dir = pathlib.Path(args.out_dir)
    if args.preflight:
        print(json.dumps(preflight(run_dir, out_dir, args.max_word_exposure), indent=2, ensure_ascii=False))
        return
    if args.run:
        run_training(run_dir, args.max_word_exposure, args.gpu, args.allow_nonempty)
    if args.compare or args.run:
        compare_and_write(run_dir, out_dir, args.max_word_exposure)
        return
    raise SystemExit('Choose --preflight, --run, or --compare')


if __name__ == '__main__':
    main()
