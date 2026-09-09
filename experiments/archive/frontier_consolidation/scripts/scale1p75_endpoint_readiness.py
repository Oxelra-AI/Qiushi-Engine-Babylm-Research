#!/usr/bin/env python3
"""research: file-only readiness check for completed scale1.75 100M endpoint.

This script does not run model inference. It verifies that the exact deterministic
100M scale1.75 training artifact is present, has the expected checkpoint ladder,
configuration, training metrics, and bit-identical shared checkpoints relative to
previous 20M/50M/80M runs. It writes a compact JSON and Markdown note so the
full official-compatible evaluator can be interpreted from a stable artifact
record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path('.')
RUN = ROOT / 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder'
HF = RUN / 'hf_model'
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_endpoint_readiness'
NOTE = ROOT / 'research/notes/frontier_consolidation/scale1p75_endpoint_readiness.md'
METRICS = RUN / 'scientific_metrics.json'
TRAIN_LOG = RUN / 'training_log.jsonl'
EXPECTED_FIRST_LOSS = 9.837543487548828
EXPECTED_FINAL_WORDS = 100_000_000
EXPECTED_STEPS = 2529
EXPECTED_PARAM_COUNT = 35_463_008
EXPECTED_VOCAB = 16_384
EXPECTED_CHECKPOINTS = [f'chck_{i}M' for i in range(1, 101)]
COMPARES = [
    ('20M', HF / 'chck_20M/model.safetensors', ROOT / 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M20M_seed43022/hf_model/chck_20M/model.safetensors'),
    ('50M', HF / 'chck_50M/model.safetensors', ROOT / 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M50M_seed43022/hf_model/chck_50M/model.safetensors'),
    ('80M', HF / 'chck_80M/model.safetensors', ROOT / 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M/model.safetensors'),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl_tail(path: Path) -> Dict[str, Any]:
    last = None
    n = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
                n += 1
    return {'line_count': n, 'last_record': last}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    metrics = load_json(METRICS)
    config = load_json(HF / 'chck_100M/config.json')
    train_tail = read_jsonl_tail(TRAIN_LOG)

    ckpt_dirs = sorted([p.name for p in HF.iterdir() if p.is_dir() and p.name.startswith('chck_')], key=lambda s: int(s.split('_')[1][:-1]))
    missing = [c for c in EXPECTED_CHECKPOINTS if c not in ckpt_dirs]
    extra = [c for c in ckpt_dirs if c not in EXPECTED_CHECKPOINTS]
    if missing: errors.append(f'missing checkpoints: {missing[:5]}... total={len(missing)}')
    if extra: errors.append(f'extra checkpoints: {extra[:5]}... total={len(extra)}')
    if metrics.get('word_exposure') != EXPECTED_FINAL_WORDS: errors.append(f"word_exposure {metrics.get('word_exposure')} != {EXPECTED_FINAL_WORDS}")
    if metrics.get('actual_training_steps') != EXPECTED_STEPS: errors.append(f"steps {metrics.get('actual_training_steps')} != {EXPECTED_STEPS}")
    if abs(float(metrics.get('loss_first')) - EXPECTED_FIRST_LOSS) > 1e-12: errors.append(f"loss_first {metrics.get('loss_first')} != {EXPECTED_FIRST_LOSS}")
    if metrics.get('parameter_count') != EXPECTED_PARAM_COUNT: errors.append(f"param_count {metrics.get('parameter_count')} != {EXPECTED_PARAM_COUNT}")
    if metrics.get('vocab_size') != EXPECTED_VOCAB: errors.append(f"vocab_size {metrics.get('vocab_size')} != {EXPECTED_VOCAB}")
    if config.get('architectures') != ['AdapterDebertaV2ForMaskedLM']: errors.append(f"architecture {config.get('architectures')}")
    if config.get('auto_map', {}).get('AutoModelForMaskedLM') != 'adapter_scaled_modeling.AdapterDebertaV2ForMaskedLM': errors.append('missing expected AutoModelForMaskedLM auto_map')
    if config.get('adapter_enabled') is not True or float(config.get('adapter_scale')) != 1.75 or int(config.get('adapter_bottleneck')) != 128:
        errors.append(f"adapter config mismatch: enabled={config.get('adapter_enabled')} scale={config.get('adapter_scale')} bottleneck={config.get('adapter_bottleneck')}")
    last = train_tail['last_record'] or {}
    # `training_log.jsonl` contains per-step records only; the final `done` event
    # is written to stdout and summarized in scientific_metrics.json.  Therefore
    # validate the last step record by step, cumulative exposure, and loss rather
    # than expecting an `event=done` field here.
    if int(last.get('step', -1)) != EXPECTED_STEPS: errors.append(f"last log step {last.get('step')}")
    if int(last.get('cumulative_word_exposure', -1)) != EXPECTED_FINAL_WORDS: errors.append(f"last log cumulative_word_exposure {last.get('cumulative_word_exposure')}")
    if abs(float(last.get('loss', float('nan'))) - float(metrics.get('loss_last'))) > 1e-12: errors.append(f"last log loss {last.get('loss')} != metrics loss_last {metrics.get('loss_last')}")

    hash_compares=[]
    for label, new_path, old_path in COMPARES:
        rec = {'label': label, 'new_path': str(new_path), 'old_path': str(old_path), 'new_exists': new_path.exists(), 'old_exists': old_path.exists()}
        if new_path.exists() and old_path.exists():
            rec['new_size'] = new_path.stat().st_size
            rec['old_size'] = old_path.stat().st_size
            rec['new_sha256'] = sha256(new_path)
            rec['old_sha256'] = sha256(old_path)
            rec['sha_equal'] = rec['new_sha256'] == rec['old_sha256']
            if not rec['sha_equal']:
                errors.append(f'{label} checkpoint hash mismatch')
        else:
            errors.append(f'{label} comparison path missing')
        hash_compares.append(rec)

    endpoint_files = {
        'chck_100M_model_safetensors': {
            'path': str(HF / 'chck_100M/model.safetensors'),
            'size': (HF / 'chck_100M/model.safetensors').stat().st_size,
            'sha256': sha256(HF / 'chck_100M/model.safetensors'),
        },
        'chck_100M_config': {
            'path': str(HF / 'chck_100M/config.json'),
            'size': (HF / 'chck_100M/config.json').stat().st_size,
            'sha256': sha256(HF / 'chck_100M/config.json'),
        },
        'chck_100M_tokenizer_json': {
            'path': str(HF / 'chck_100M/tokenizer.json'),
            'size': (HF / 'chck_100M/tokenizer.json').stat().st_size,
            'sha256': sha256(HF / 'chck_100M/tokenizer.json'),
        },
    }
    saved = metrics.get('saved_checkpoints', [])
    ckpt_100 = next((r for r in saved if r.get('name') == 'chck_100M'), {})

    summary = {
        'status': 'SCALE1P75_ENDPOINT_READINESS',
        'run_dir': str(RUN),
        'metrics_path': str(METRICS),
        'training_log_path': str(TRAIN_LOG),
        'valid': not errors,
        'errors': errors,
        'metrics_core': {
            'word_exposure': metrics.get('word_exposure'),
            'actual_training_steps': metrics.get('actual_training_steps'),
            'loss_first': metrics.get('loss_first'),
            'loss_last': metrics.get('loss_last'),
            'parameter_count': metrics.get('parameter_count'),
            'vocab_size': metrics.get('vocab_size'),
            'seed': metrics.get('seed'),
            'masking_curriculum': metrics.get('masking_curriculum'),
            'mask_prob_start': metrics.get('mask_prob_start'),
            'mask_prob_end': metrics.get('mask_prob_end'),
            'example_jsonl': metrics.get('example_jsonl'),
            'example_jsonl_label': metrics.get('example_jsonl_label'),
            'source_words_consumed': metrics.get('source_words_consumed'),
        },
        'config_core': {
            'architectures': config.get('architectures'),
            'auto_map': config.get('auto_map'),
            'adapter_enabled': config.get('adapter_enabled'),
            'adapter_scale': config.get('adapter_scale'),
            'adapter_bottleneck': config.get('adapter_bottleneck'),
            'hidden_size': config.get('hidden_size'),
            'num_hidden_layers': config.get('num_hidden_layers'),
            'num_attention_heads': config.get('num_attention_heads'),
            'vocab_size': config.get('vocab_size'),
        },
        'checkpoint_ladder': {
            'count': len(ckpt_dirs),
            'first': ckpt_dirs[:5],
            'last': ckpt_dirs[-5:],
            'missing': missing,
            'extra': extra,
            'chck_100M_record': ckpt_100,
        },
        'training_log_tail': train_tail,
        'shared_checkpoint_hash_compares': hash_compares,
        'endpoint_files': endpoint_files,
        'official_score_status': 'not_scored_here; endpoint evaluation remains pending',
    }
    out_json = OUT_DIR / 'scale1p75_endpoint_readiness.json'
    out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines: List[str] = []
    lines.append('# research — scale1.75 100M endpoint readiness')
    lines.append('')
    lines.append('This is a file-only training-artifact check. It does not run or replace the official-compatible full evaluator.')
    lines.append('')
    lines.append(f"- Run: `{RUN}`")
    lines.append(f"- Valid artifact checks: `{summary['valid']}`; errors: `{errors}`")
    lines.append(f"- Word exposure: {metrics.get('word_exposure')}; steps: {metrics.get('actual_training_steps')}; loss first/last: {metrics.get('loss_first')} → {metrics.get('loss_last')}.")
    lines.append(f"- Checkpoint ladder: {len(ckpt_dirs)} directories, expected `chck_1M` ... `chck_100M`; missing={len(missing)}, extra={len(extra)}.")
    lines.append(f"- `chck_100M` record: target {ckpt_100.get('target_word_exposure')}, actual {ckpt_100.get('actual_cumulative_word_exposure')}.")
    lines.append(f"- Config: architecture {config.get('architectures')}, AutoModelForMaskedLM `{config.get('auto_map', {}).get('AutoModelForMaskedLM')}`, adapter scale {config.get('adapter_scale')}, bottleneck {config.get('adapter_bottleneck')}, enabled {config.get('adapter_enabled')}.")
    lines.append(f"- 100M model SHA256: `{endpoint_files['chck_100M_model_safetensors']['sha256']}`; size {endpoint_files['chck_100M_model_safetensors']['size']} bytes.")
    lines.append('')
    lines.append('## Deterministic prefix reproduction')
    lines.append('')
    for r in hash_compares:
        lines.append(f"- {r['label']}: new/old model hashes equal = `{r.get('sha_equal')}`; SHA `{r.get('new_sha256')}`.")
    lines.append('')
    lines.append('Scientific consequence: if `valid` is true, the 100M endpoint is a deterministic continuation of the previously interpreted 20M/50M/80M scale1.75 trajectory. The endpoint score itself is still unknown here and must come from the full nine-column official-compatible evaluator.')
    lines.append('')
    lines.append(f"JSON: `{out_json}`")
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'valid': summary['valid'], 'out_json': str(out_json), 'note': str(NOTE), 'loss_last': metrics.get('loss_last'), 'model_sha100M': endpoint_files['chck_100M_model_safetensors']['sha256']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
