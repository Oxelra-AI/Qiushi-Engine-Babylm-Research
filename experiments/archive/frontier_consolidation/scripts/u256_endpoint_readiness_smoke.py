#!/usr/bin/env python3
"""research: U256 endpoint readiness and CPU HF-load smoke.

This is CPU/filesystem-only and does not inspect or interfere with the active
U256 full evaluator.  It normalizes the U256 100M training metric schema,
checks legal corpus/tokenizer/provenance invariants, verifies checkpoint files,
compares endpoint tokenizer vocabulary to the research legal tokenizer, and loads
chck_100M with a writable HF cache to ensure the endpoint is usable for official
compatible evaluation and later packaging if the score warrants it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path('.').resolve()
DEFAULT_RUN_DIR = USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/eu_U256_legal16k_seed43022_100M'
DEFAULT_LEGAL_TOKENIZER = USER_ROOT / 'experiments/archive/frontier_consolidation/data/compliant_tokenizer'
DEFAULT_OUT_DIR = USER_ROOT / 'experiments/archive/frontier_consolidation/data/u256_endpoint_readiness_smoke'
EXPECTED_BASE_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
EXPECTED_STREAM_SHA = '3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691'
EXPECTED_TOKENIZER_SHA = '91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9'
EXPECTED_FIRST_LOSS = 9.826857208144903
EXPECTED_LAST_LOSS = 2.516624725910071
EXPECTED_PARAM_COUNT = 34_467_424
AOA_STEPS = [f'chck_{i}M' for i in range(1, 11)] + [f'chck_{i}M' for i in range(20, 101, 10)]


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(USER_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError({'missing': label, 'path': rel(path)})


def close(a: float | None, b: float, tol: float) -> bool:
    return a is not None and math.isfinite(float(a)) and abs(float(a) - b) <= tol


def model_vocab(tokenizer_json: Path) -> dict[str, int]:
    tok = read_json(tokenizer_json)
    model = tok.get('model') or {}
    vocab = model.get('vocab')
    if not isinstance(vocab, dict):
        raise RuntimeError({'missing_vocab_in_tokenizer_json': rel(tokenizer_json)})
    return {str(k): int(v) for k, v in vocab.items()}


def collect_readiness(run_dir: Path, legal_tokenizer_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / 'scientific_metrics.json'
    endpoint = run_dir / 'hf_model/chck_100M'
    require(metrics_path, 'U256 scientific_metrics.json')
    require(endpoint / 'model.safetensors', 'U256 chck_100M model.safetensors')
    require(endpoint / 'config.json', 'U256 chck_100M config.json')
    require(endpoint / 'tokenizer.json', 'U256 chck_100M tokenizer.json')
    require(legal_tokenizer_dir / 'tokenizer.json', 'research legal tokenizer.json')
    require(legal_tokenizer_dir / 'tokenizer_metadata.json', 'research legal tokenizer metadata')

    metrics = read_json(metrics_path)
    cfg = read_json(endpoint / 'config.json')
    tok_meta = read_json(legal_tokenizer_dir / 'tokenizer_metadata.json')

    checkpoints = metrics.get('checkpoints') or []
    checkpoint_names = [c.get('name') for c in checkpoints if isinstance(c, dict)]
    checkpoint_name_set = set(checkpoint_names)
    saved_checkpoint_dirs = sorted([p.name for p in (run_dir / 'hf_model').glob('chck_*')])
    missing_aoa_model_files = [s for s in AOA_STEPS if not (run_dir / 'hf_model' / s / 'model.safetensors').exists()]
    missing_all_model_files = [name for name in checkpoint_names if not (run_dir / 'hf_model' / str(name) / 'model.safetensors').exists()]

    endpoint_tok_sha = sha256_file(endpoint / 'tokenizer.json')
    legal_tok_sha = sha256_file(legal_tokenizer_dir / 'tokenizer.json')
    endpoint_vocab = model_vocab(endpoint / 'tokenizer.json')
    legal_vocab = model_vocab(legal_tokenizer_dir / 'tokenizer.json')
    vocab_mismatch = []
    if endpoint_vocab != legal_vocab:
        for key in sorted(set(endpoint_vocab) | set(legal_vocab)):
            if endpoint_vocab.get(key) != legal_vocab.get(key):
                vocab_mismatch.append({'token': key, 'endpoint': endpoint_vocab.get(key), 'legal': legal_vocab.get(key)})
                if len(vocab_mismatch) >= 20:
                    break

    verification = metrics.get('verification') or {}
    source_words_consumed = metrics.get('source_words_consumed') or {}

    expected_checks = {
        'arm_is_U256': metrics.get('arm') == 'U256',
        'word_exposure_is_100M': int(metrics.get('word_exposure', -1)) == 100_000_000,
        'total_charged_words_is_100M': int(metrics.get('total_charged_words', -1)) == 100_000_000,
        'total_steps_is_2530': int(metrics.get('total_steps_executed', -1)) == 2530,
        'completed_epochs_is_10': int(verification.get('completed_epochs', -1)) == 10,
        'first_loss_matches_20M_prefix': close(metrics.get('loss_first'), EXPECTED_FIRST_LOSS, 2e-9),
        'last_loss_matches_collected_training': close(metrics.get('loss_last'), EXPECTED_LAST_LOSS, 2e-9),
        'base_sha_matches_step35_pool': metrics.get('base_sha256') == EXPECTED_BASE_SHA,
        'stream_sha_matches_step35_stream': metrics.get('stream_sha256') == EXPECTED_STREAM_SHA,
        'tokenizer_vocab_size_16384': int(metrics.get('vocab_size', -1)) == 16_384 and int(cfg.get('vocab_size', -1)) == 16_384,
        'parameter_count_matches_debertav2_8x480': int(metrics.get('parameter_count', -1)) == EXPECTED_PARAM_COUNT,
        'verification_words_match_100M': verification.get('words_match_100M') is True,
        'verification_stream_blocks_match': verification.get('stream_blocks_multiset_match') is True,
        'verification_all_epoch_words_10M': verification.get('all_epoch_words_10M') is True,
        'verification_all_epoch_steps_253': verification.get('all_epoch_steps_253') is True,
        'verification_all_epoch_active_tokens_equal_base': verification.get('all_epoch_active_tokens_equal_base') is True,
        'checkpoint_count_100_in_metrics': len(checkpoint_names) == 100 and verification.get('checkpoint_count') == 100,
        'checkpoint_dirs_100': len(saved_checkpoint_dirs) == 100,
        'all_metric_checkpoint_models_present': len(missing_all_model_files) == 0,
        'aoa_ladder_model_files_present': len(missing_aoa_model_files) == 0 and all(s in checkpoint_name_set for s in AOA_STEPS),
        'endpoint_tokenizer_sha_matches_legal': endpoint_tok_sha == legal_tok_sha == EXPECTED_TOKENIZER_SHA,
        'endpoint_tokenizer_vocab_equals_legal': endpoint_vocab == legal_vocab,
        'legal_tokenizer_metadata_pool_sha_matches': (((tok_meta.get('training_data') or {}).get('pool_sha256')) == EXPECTED_BASE_SHA),
        'config_shape_matches_step35_backbone': (
            cfg.get('model_type') == 'deberta-v2'
            and int(cfg.get('num_hidden_layers', -1)) == 8
            and int(cfg.get('hidden_size', -1)) == 480
            and int(cfg.get('intermediate_size', -1)) == 1920
            and int(cfg.get('num_attention_heads', -1)) == 8
            and int(cfg.get('max_position_embeddings', -1)) == 512
            and int(cfg.get('pad_token_id', -1)) == 3
        ),
    }

    readiness_errors = [k for k, ok in expected_checks.items() if not ok]
    return {
        'status': 'U256_ENDPOINT_READINESS',
        'created_utc': now(),
        'run_dir': rel(run_dir),
        'metrics_path': rel(metrics_path),
        'endpoint': rel(endpoint),
        'legal_tokenizer_dir': rel(legal_tokenizer_dir),
        'summary': {
            'arm': metrics.get('arm'),
            'word_exposure': metrics.get('word_exposure'),
            'total_charged_words': metrics.get('total_charged_words'),
            'total_steps_executed': metrics.get('total_steps_executed'),
            'loss_first': metrics.get('loss_first'),
            'loss_last': metrics.get('loss_last'),
            'raw_tokens_per_epoch': metrics.get('raw_tokens_per_epoch'),
            'parameter_count': metrics.get('parameter_count'),
            'vocab_size_metric': metrics.get('vocab_size'),
            'vocab_size_config': cfg.get('vocab_size'),
            'checkpoint_count_metrics': len(checkpoint_names),
            'checkpoint_dirs': len(saved_checkpoint_dirs),
            'aoa_ladder_count': len(AOA_STEPS),
            'source_word_total': sum(int(v) for v in source_words_consumed.values()) if isinstance(source_words_consumed, dict) else None,
            'source_label_count': len(source_words_consumed) if isinstance(source_words_consumed, dict) else None,
        },
        'legal_hashes': {
            'base_sha256': metrics.get('base_sha256'),
            'stream_sha256': metrics.get('stream_sha256'),
            'endpoint_tokenizer_json_sha256': endpoint_tok_sha,
            'legal_tokenizer_json_sha256': legal_tok_sha,
            'expected_tokenizer_json_sha256': EXPECTED_TOKENIZER_SHA,
        },
        'checkpoint_samples': {
            'first3': checkpoint_names[:3],
            'last3': checkpoint_names[-3:],
            'aoa_ladder': AOA_STEPS,
            'missing_aoa_model_files': missing_aoa_model_files,
            'missing_metric_checkpoint_model_files': missing_all_model_files[:20],
        },
        'config_core': {k: cfg.get(k) for k in ['architectures','model_type','num_hidden_layers','hidden_size','intermediate_size','num_attention_heads','max_position_embeddings','max_relative_positions','position_buckets','vocab_size','pad_token_id','type_vocab_size']},
        'verification': verification,
        'expected_checks': expected_checks,
        'readiness_ok': len(readiness_errors) == 0,
        'readiness_errors': readiness_errors,
        'tokenizer_vocab_mismatch_sample': vocab_mismatch,
    }


def run_hf_smoke(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    cache = out_dir / 'hf_cache'
    modules = out_dir / 'hf_modules'
    cache.mkdir(parents=True, exist_ok=True)
    modules.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(cache.resolve())
    os.environ['HF_MODULES_CACHE'] = str(modules.resolve())
    os.environ['TRANSFORMERS_CACHE'] = str(cache.resolve())
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import torch  # noqa: WPS433
    from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: WPS433

    endpoint = run_dir / 'hf_model/chck_100M'
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(endpoint), local_files_only=True, use_fast=True, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(str(endpoint), local_files_only=True, trust_remote_code=True)
    model.eval()
    load_sec = time.time() - t0
    text = "The small child put the red ball on the table and then [MASK] smiled."
    if tok.mask_token and tok.mask_token != '[MASK]':
        text = text.replace('[MASK]', tok.mask_token)
    enc = tok(text, return_tensors='pt')
    labels = enc['input_ids'].clone()
    mask_token_id = tok.mask_token_id
    if mask_token_id is None:
        raise RuntimeError('tokenizer has no mask_token_id')
    non_mask = labels != mask_token_id
    labels[non_mask] = -100
    with torch.no_grad():
        out = model(**enc, labels=labels)
    logits = out.logits.detach()
    finite = torch.isfinite(logits).all().item()
    loss_val = float(out.loss.item()) if out.loss is not None else None
    param_count = sum(p.numel() for p in model.parameters())
    mask_positions = (enc['input_ids'] == mask_token_id).nonzero(as_tuple=False)
    top_tokens: list[dict[str, Any]] = []
    if mask_positions.numel() > 0:
        pos = int(mask_positions[0, 1].item())
        probs = torch.softmax(logits[0, pos], dim=-1)
        vals, ids = torch.topk(probs, k=10)
        for v, i in zip(vals.tolist(), ids.tolist()):
            top_tokens.append({'token_id': int(i), 'token': tok.decode([int(i)]), 'prob': float(v)})
    return {
        'status': 'U256_HF_LOAD_SMOKE',
        'created_utc': now(),
        'endpoint': rel(endpoint),
        'hf_home': rel(cache),
        'hf_modules_cache': rel(modules),
        'tokenizer_length': len(tok),
        'mask_token': tok.mask_token,
        'mask_token_id': tok.mask_token_id,
        'pad_token_id': tok.pad_token_id,
        'model_class': type(model).__name__,
        'param_count': param_count,
        'expected_param_count': EXPECTED_PARAM_COUNT,
        'param_count_matches': param_count == EXPECTED_PARAM_COUNT,
        'finite_logits': bool(finite),
        'probe_loss': loss_val,
        'input_shape': list(enc['input_ids'].shape),
        'top_mask_tokens': top_tokens,
        'load_and_probe_sec': load_sec,
        'ok': bool(finite) and param_count == EXPECTED_PARAM_COUNT and loss_val is not None and math.isfinite(loss_val),
    }


def write_outputs(out_dir: Path, readiness: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        'status': 'U256_READINESS_SMOKE_DONE',
        'created_utc': now(),
        'readiness': readiness,
        'hf_smoke': smoke,
        'ok': bool(readiness.get('readiness_ok')) and bool(smoke.get('ok')),
    }
    json_path = out_dir / 'u256_endpoint_readiness_smoke.json'
    md_path = out_dir / 'u256_endpoint_readiness_smoke.md'
    json_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = [
        '# research — U256 endpoint readiness and HF-load smoke',
        '',
        f"Status: `{combined['status']}`; ok={combined['ok']}",
        '',
        '## Readiness',
        f"- run: `{readiness['run_dir']}`",
        f"- arm: {readiness['summary']['arm']}; words: {readiness['summary']['word_exposure']}; steps: {readiness['summary']['total_steps_executed']}",
        f"- losses: {readiness['summary']['loss_first']} -> {readiness['summary']['loss_last']}",
        f"- raw tokens/epoch: {readiness['summary']['raw_tokens_per_epoch']}; checkpoints: {readiness['summary']['checkpoint_count_metrics']}",
        f"- legal pool SHA: `{readiness['legal_hashes']['base_sha256']}`",
        f"- stream SHA: `{readiness['legal_hashes']['stream_sha256']}`",
        f"- endpoint tokenizer SHA: `{readiness['legal_hashes']['endpoint_tokenizer_json_sha256']}`",
        f"- readiness_ok={readiness['readiness_ok']}; errors={readiness['readiness_errors']}",
        '',
        '## HF load smoke',
        f"- model class: {smoke['model_class']}; params: {smoke['param_count']}; finite_logits={smoke['finite_logits']}; probe_loss={smoke['probe_loss']}",
        f"- writable caches: `{smoke['hf_home']}`, `{smoke['hf_modules_cache']}`",
        f"- top mask tokens: {smoke['top_mask_tokens'][:5]}",
        '',
        f"JSON: `{rel(json_path)}`",
    ]
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'json': rel(json_path), 'md': rel(md_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument('--legal-tokenizer-dir', type=Path, default=DEFAULT_LEGAL_TOKENIZER)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument('--skip-hf-smoke', action='store_true')
    args = ap.parse_args()

    readiness = collect_readiness(args.run_dir.resolve(), args.legal_tokenizer_dir.resolve())
    if args.skip_hf_smoke:
        smoke = {'status': 'U256_HF_LOAD_SMOKE_SKIPPED', 'ok': True, 'created_utc': now()}
    else:
        smoke = run_hf_smoke(args.run_dir.resolve(), args.out_dir.resolve())
    outputs = write_outputs(args.out_dir.resolve(), readiness, smoke)
    print(json.dumps({
        'status': 'U256_READINESS_SMOKE_DONE',
        'ok': readiness.get('readiness_ok') and smoke.get('ok'),
        'readiness_ok': readiness.get('readiness_ok'),
        'readiness_errors': readiness.get('readiness_errors'),
        'hf_smoke_ok': smoke.get('ok'),
        'probe_loss': smoke.get('probe_loss'),
        'outputs': outputs,
    }, indent=2), flush=True)
    if not (readiness.get('readiness_ok') and smoke.get('ok')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
