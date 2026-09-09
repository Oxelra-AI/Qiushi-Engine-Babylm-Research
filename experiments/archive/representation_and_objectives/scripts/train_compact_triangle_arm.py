#!/usr/bin/env python3
"""research compact-view mechanism triangle training launcher.

Runs one missing matched arm for the compact-view decomposition:
- compact_repeat_reinvest: exact-source repetition with reinvested breadth.
- adjbreak_reinvest: source/rewrite marginals preserved but source-own adjacency broken.

The launcher is intentionally narrow: it preserves the trusted COMPACT_EXPERIENCE DeBERTa-v2
8x480 / baseline16k / fixed WWM / AdamW recipe and records exact file hashes
before handing off to the trainer. It does not evaluate or interpret results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

USER_ROOT = pathlib.Path('.')
TRAINER = USER_ROOT / 'experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py'
TOKENIZER = USER_ROOT / 'experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model'
A02_DENSITY_META = USER_ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json'
A01_ADJBREAK_MEASUREMENT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/compact_reinvest_adjbreak_control/compact_reinvest_adjbreak_control_measurement.json'

ARM_CONFIG = {
    'compact_repeat_reinvest': {
        'example_jsonl_label': 'cleanqwen_fineweb_repeat_compact_reinvest',
        'train_file': USER_ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl',
        'meta': A02_DENSITY_META,
        'expected_sha256': '91e8817e25f234b87481fa6ca96d968d6c460b84317273629eda21ea3be24eaa',
        'default_run_dir': USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/compact_repeat_reinvest_16k_seed43022',
        'scientific_role': 'Exact-repeat reinvest arm: tests whether compact-view gains require generated semantic second views versus added source breadth/exposure only.',
    },
    'adjbreak_reinvest': {
        'example_jsonl_label': 'cleanqwen_fineweb_compact_view_reinvest_adjbreak',
        'train_file': USER_ROOT / 'experiments/archive/representation_and_objectives/data/compact_reinvest_adjbreak_control/cleanqwen_fineweb_compact_view_reinvest_adjbreak_100M.jsonl',
        'meta': A01_ADJBREAK_MEASUREMENT,
        'expected_sha256': '3097308293081a784d383f9bcaaf01a792e3be3484565e3b63d180ebe7017312',
        'default_run_dir': USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/adjbreak_reinvest_16k_seed43022',
        'scientific_role': 'Adjacency-broken reinvest arm: preserves source/rewrite marginals and word/row geometry while breaking source-own rewrite locality.',
    },
}

FIXED_RECIPE = {
    'tokenizer_path': str(TOKENIZER),
    'tokenizer_label': 'baseline16k',
    'model_family': 'DeBERTa-v2 masked LM',
    'hidden_size': 480,
    'n_layer': 8,
    'n_head': 8,
    'ffn_mult': 4,
    'batch_size': 256,
    'seq_length': 256,
    'max_seq_length': 256,
    'learning_rate': 0.001,
    'warmup_fraction': 0.06,
    'weight_decay': 0.01,
    'masking_curriculum': 'wwm_fixed',
    'mask_prob_start': 0.15,
    'mask_prob_end': 0.15,
    'checkpoint_words': 1_000_000,
    'max_word_exposure': 100_000_000,
    'num_workers': 0,
    'log_every': 50,
    'dynamics_trace_every': 200,
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def build_command(cfg: dict[str, Any], run_dir: pathlib.Path, seed: int, extra_init_seed: int, train_rng_seed: int) -> list[str]:
    return [
        sys.executable,
        str(TRAINER),
        '--example_jsonl', str(cfg['train_file']),
        '--example_jsonl_label', cfg['example_jsonl_label'],
        '--example_jsonl_meta', str(cfg['meta']),
        '--output_dir', str(run_dir),
        '--tokenizer_path', str(TOKENIZER),
        '--tokenizer_label', 'baseline16k',
        '--hidden_size', '480',
        '--n_layer', '8',
        '--n_head', '8',
        '--ffn_mult', '4',
        '--seed', str(seed),
        '--extra_init_seed', str(extra_init_seed),
        '--train_rng_seed', str(train_rng_seed),
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
        '--max_word_exposure', '100000000',
        '--num_workers', '0',
        '--log_every', '50',
        '--dynamics_trace_every', '200',
    ]


def preflight(arm: str, run_dir: pathlib.Path, seed: int, extra_init_seed: int, train_rng_seed: int, check_hash: bool) -> dict[str, Any]:
    cfg = ARM_CONFIG[arm]
    required = [TRAINER, TOKENIZER / 'tokenizer.json', TOKENIZER / 'tokenizer_config.json', cfg['train_file'], cfg['meta']]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError({'missing_required_paths': missing})
    actual_sha = sha256_file(cfg['train_file']) if check_hash else None
    if check_hash and actual_sha != cfg['expected_sha256']:
        raise RuntimeError({'arm': arm, 'actual_sha256': actual_sha, 'expected_sha256': cfg['expected_sha256']})
    meta_status = None
    meta_summary: dict[str, Any] = {}
    try:
        meta = read_json(cfg['meta'])
        meta_status = meta.get('status')
        if arm == 'adjbreak_reinvest':
            meta_summary = {
                'pool_words': meta.get('pool', {}).get('words'),
                'training_100M_written': meta.get('files', {}).get('training_100M_written'),
                'pool_10M_sha256': meta.get('files', {}).get('pool_10M_sha256'),
                'row_word_sequence_identical': meta.get('pool', {}).get('row_word_sequence_identical_to_original_view_pool'),
                'full_pair_visible_rate': meta.get('adjbreak_seq256_visibility', {}).get('full_pair_visible_rate'),
                'source_visible_rate': meta.get('adjbreak_seq256_visibility', {}).get('source_visible_rate'),
            }
        else:
            meta_summary = {
                'all_exact_10M': bool(meta.get('all_exact_10M') or meta.get('audit', {}).get('all_exact_10M')),
                'write_training': bool(meta.get('write_training')),
                'expected_sha256_from_meta': meta.get('sha256', {}).get(pathlib.Path(cfg['train_file']).name),
            }
    except Exception as exc:
        meta_summary = {'meta_read_error': repr(exc)}
    cmd = build_command(cfg, run_dir, seed, extra_init_seed, train_rng_seed)
    return {
        'status': 'TRIANGLE_PREFLIGHT_OK',
        'arm': arm,
        'scientific_role': cfg['scientific_role'],
        'train_file': str(cfg['train_file']),
        'train_file_bytes': pathlib.Path(cfg['train_file']).stat().st_size,
        'expected_sha256': cfg['expected_sha256'],
        'actual_sha256': actual_sha,
        'hash_ok': (actual_sha == cfg['expected_sha256']) if check_hash else None,
        'meta': str(cfg['meta']),
        'meta_status': meta_status,
        'meta_summary': meta_summary,
        'run_dir': str(run_dir),
        'seed': seed,
        'extra_init_seed': extra_init_seed,
        'train_rng_seed': train_rng_seed,
        'fixed_recipe': FIXED_RECIPE,
        'command': cmd,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=sorted(ARM_CONFIG))
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--extra-init-seed', type=int, default=43022)
    ap.add_argument('--train-rng-seed', type=int, default=43023)
    ap.add_argument('--run-dir', default='')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--check-hash', action='store_true')
    ap.add_argument('--allow-nonempty', action='store_true')
    args = ap.parse_args()

    cfg = ARM_CONFIG[args.arm]
    run_dir = pathlib.Path(args.run_dir) if args.run_dir else pathlib.Path(cfg['default_run_dir'])
    status = preflight(args.arm, run_dir, args.seed, args.extra_init_seed, args.train_rng_seed, args.check_hash)
    status['cuda_visible_devices'] = str(args.gpu)
    if args.dry_run:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return

    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise RuntimeError(f'run_dir exists and is non-empty: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'train_command.json').write_text(json.dumps(status, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env['TMPDIR'] = f'/tmp/q_representation_and_objectives_step201_{args.arm}_{args.extra_init_seed}_{int(time.time())}'
    pathlib.Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / 'train_stdout.log'
    stderr_path = run_dir / 'train_stderr.log'
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    cmd = status['command']
    with stdout_path.open('w', encoding='utf-8') as out, stderr_path.open('w', encoding='utf-8') as err:
        out.write(json.dumps({'event': 'launcher_start', 'started_utc': started, 'arm': args.arm, 'gpu': args.gpu, 'seed': args.seed, 'extra_init_seed': args.extra_init_seed, 'train_rng_seed': args.train_rng_seed}) + '\n')
        out.flush()
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env)
    finished = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    metrics_path = run_dir / 'scientific_metrics.json'
    result: dict[str, Any] = {
        'status': 'TRIANGLE_TRAIN_FINISHED' if proc.returncode == 0 else 'TRIANGLE_TRAIN_FAILED',
        'arm': args.arm,
        'returncode': proc.returncode,
        'started_utc': started,
        'finished_utc': finished,
        'gpu': args.gpu,
        'run_dir': str(run_dir),
        'stdout_log': str(stdout_path),
        'stderr_log': str(stderr_path),
        'metrics_path': str(metrics_path),
        'metrics_exists': metrics_path.exists(),
        'seed': args.seed,
        'extra_init_seed': args.extra_init_seed,
        'train_rng_seed': args.train_rng_seed,
    }
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        result['metrics'] = {
            'word_exposure': metrics.get('word_exposure'),
            'actual_training_steps': metrics.get('actual_training_steps'),
            'loss_first': metrics.get('loss_first'),
            'loss_last': metrics.get('loss_last'),
            'parameter_count': metrics.get('parameter_count'),
            'vocab_size': metrics.get('vocab_size'),
            'saved_checkpoints': len(metrics.get('saved_checkpoints', [])),
            'first_checkpoint': metrics.get('saved_checkpoints', [{}])[0].get('name') if metrics.get('saved_checkpoints') else None,
            'last_checkpoint': metrics.get('saved_checkpoints', [{}])[-1].get('name') if metrics.get('saved_checkpoints') else None,
        }
    (run_dir / 'launcher_result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
