#!/usr/bin/env python3
"""research: CPU HF-load smoke for the completed scale1.75 100M endpoint.

This verifies that the custom adapter checkpoint can be loaded through the same
AutoModelForMaskedLM/trust_remote_code path used by official-compatible eval and
that a tiny deterministic CPU forward pass returns finite MLM logits/loss.  It
is not an evaluation score and does not touch GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path('.').resolve()
WORKSPACE = USER_ROOT / 'experiments/archive/frontier_consolidation'
RUN_DIR = WORKSPACE / 'training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder'
CKPT = RUN_DIR / 'hf_model/chck_100M'
TOKENIZER_DIR = WORKSPACE / 'data/compliant_tokenizer'
BASE_10M = WORKSPACE / 'data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
OUT_DIR = WORKSPACE / 'data/scale1p75_hf_load_smoke'
OUT_JSON = OUT_DIR / 'scale1p75_hf_load_smoke.json'
NOTE = (USER_ROOT / 'research/notes/frontier_consolidation/scale1p75_hf_load_smoke.md')
EXPECTED_MODEL_SHA = '7349475846ef2a193df850cc23876b2a86f99e82b9f051816064e4561b0f6f52'
EXPECTED_TOKENIZER_SHA = '91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9'
EXPECTED_BASE_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(USER_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_examples(n: int = 4) -> list[str]:
    out: list[str] = []
    with BASE_10M.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            txt = str(rec['text'])
            if len(txt.split()) >= 16:
                out.append(txt)
            if len(out) >= n:
                break
    if len(out) < n:
        raise RuntimeError(f'only found {len(out)} examples')
    return out


def count_parameters(model: Any) -> dict[str, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    adapter = 0
    stock = 0
    adapter_names: list[str] = []
    for name, p in model.named_parameters():
        if '.adapter.' in name or 'adapters' in name or 'adapter_' in name:
            adapter += int(p.numel())
            if len(adapter_names) < 12:
                adapter_names.append(name)
        else:
            stock += int(p.numel())
    return {'total': total, 'trainable': trainable, 'adapter_named': adapter, 'stock_named': stock, 'adapter_name_examples': adapter_names}


def vocab_map(tok_json_path: Path) -> dict[str, int]:
    data = json.loads(tok_json_path.read_text(encoding='utf-8'))
    return data.get('model', {}).get('vocab', {})


def main() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    # Official-compatible trust_remote_code loading writes a cached dynamic module.
    # The runtime default HF module cache can be read-only, so make this smoke's
    # cache explicit and local before importing Transformers.
    local_hf_home = OUT_DIR / 'hf_home'
    local_modules = OUT_DIR / 'hf_modules_cache'
    local_hub = OUT_DIR / 'hf_hub_cache'
    local_hf_home.mkdir(parents=True, exist_ok=True)
    local_modules.mkdir(parents=True, exist_ok=True)
    local_hub.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(local_hf_home)
    os.environ['HF_MODULES_CACHE'] = str(local_modules)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(local_hub)
    os.environ['TRANSFORMERS_CACHE'] = str(local_hub)

    errors: list[str] = []
    for p in [CKPT, CKPT / 'config.json', CKPT / 'model.safetensors', CKPT / 'adapter_scaled_modeling.py', CKPT / 'tokenizer.json', BASE_10M, TOKENIZER_DIR / 'tokenizer.json']:
        if not p.exists():
            errors.append(f'missing {rel(p)}')
    if errors:
        raise FileNotFoundError(errors)

    model_sha = sha256_file(CKPT / 'model.safetensors')
    tok_sha = sha256_file(TOKENIZER_DIR / 'tokenizer.json')
    base_sha = sha256_file(BASE_10M)
    endpoint_tok_sha = sha256_file(CKPT / 'tokenizer.json')
    train_vocab = vocab_map(TOKENIZER_DIR / 'tokenizer.json')
    endpoint_vocab = vocab_map(CKPT / 'tokenizer.json')

    # Import after CUDA is hidden.
    import torch  # noqa: WPS433
    from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: WPS433

    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT), use_fast=True, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(str(CKPT), trust_remote_code=True)
    model.eval()
    model.to('cpu')

    cfg = model.config.to_dict()
    params = count_parameters(model)
    texts = read_examples(4)
    enc = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=64, padding='max_length', return_tensors='pt')
    input_ids = enc['input_ids'].clone()
    attention_mask = enc['attention_mask']
    labels = torch.full_like(input_ids, -100)
    mask_id = int(tokenizer.mask_token_id)
    # Deterministic tiny MLM probe: mask three non-pad positions per sequence.
    masked_positions: list[list[int]] = []
    for i in range(input_ids.shape[0]):
        valid = [j for j in range(input_ids.shape[1]) if int(attention_mask[i, j]) == 1 and int(input_ids[i, j]) != int(tokenizer.pad_token_id)]
        chosen = valid[2:5] if len(valid) >= 5 else valid[:3]
        masked_positions.append(chosen)
        for j in chosen:
            labels[i, j] = input_ids[i, j]
            input_ids[i, j] = mask_id
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = out.logits.detach()
    loss = float(out.loss.item()) if out.loss is not None else None
    finite_logits = bool(torch.isfinite(logits).all().item())
    finite_loss = bool(loss is not None and math.isfinite(loss))
    sample_logits = logits[0, masked_positions[0][0], :].detach().cpu()
    top = torch.topk(sample_logits, k=8)
    top_tokens = [tokenizer.convert_ids_to_tokens(int(i)) for i in top.indices.tolist()]

    ok = True
    checks = {
        'model_sha_matches_step113': model_sha == EXPECTED_MODEL_SHA,
        'training_tokenizer_sha_matches_step35': tok_sha == EXPECTED_TOKENIZER_SHA,
        'base_sha_matches_legal10M': base_sha == EXPECTED_BASE_SHA,
        'endpoint_tokenizer_vocab_equals_training_tokenizer': endpoint_vocab == train_vocab,
        'auto_map_present': bool(cfg.get('auto_map', {}).get('AutoModelForMaskedLM')),
        'architecture_adapter': cfg.get('architectures') == ['AdapterDebertaV2ForMaskedLM'],
        'adapter_enabled': bool(cfg.get('adapter_enabled')),
        'adapter_scale_1p75': abs(float(cfg.get('adapter_scale', -999.0)) - 1.75) < 1e-12,
        'adapter_bottleneck_128': int(cfg.get('adapter_bottleneck', -1)) == 128,
        'vocab_16384': int(cfg.get('vocab_size', -1)) == 16384,
        'mask_token_id_4': int(tokenizer.mask_token_id) == 4,
        'finite_logits': finite_logits,
        'finite_loss': finite_loss,
        'expected_total_params': params['total'] == 35_463_008,
        'expected_adapter_params_by_name': params['adapter_named'] == 995_584,
    }
    ok = all(checks.values())

    result = {
        'status': 'SCALE1P75_HF_LOAD_SMOKE',
        'utc': now(),
        'ok': ok,
        'checks': checks,
        'checkpoint': rel(CKPT),
        'model_sha256': model_sha,
        'tokenizer_sha256_training_dir': tok_sha,
        'tokenizer_sha256_endpoint': endpoint_tok_sha,
        'base10m_sha256': base_sha,
        'config_core': {
            'architectures': cfg.get('architectures'),
            'auto_map': cfg.get('auto_map'),
            'model_type': cfg.get('model_type'),
            'hidden_size': cfg.get('hidden_size'),
            'num_hidden_layers': cfg.get('num_hidden_layers'),
            'num_attention_heads': cfg.get('num_attention_heads'),
            'adapter_enabled': cfg.get('adapter_enabled'),
            'adapter_bottleneck': cfg.get('adapter_bottleneck'),
            'adapter_scale': cfg.get('adapter_scale'),
            'vocab_size': cfg.get('vocab_size'),
        },
        'parameter_counts': params,
        'tokenizer_vocab_equality': {
            'training_vocab_size': len(train_vocab),
            'endpoint_vocab_size': len(endpoint_vocab),
            'vocab_equal': endpoint_vocab == train_vocab,
            'endpoint_json_sha_differs_due_to_reserialization': endpoint_tok_sha != tok_sha,
        },
        'probe': {
            'n_texts': len(texts),
            'seq_len': int(input_ids.shape[1]),
            'masked_positions': masked_positions,
            'loss': loss,
            'logits_shape': list(logits.shape),
            'first_mask_top8_ids': [int(i) for i in top.indices.tolist()],
            'first_mask_top8_tokens': top_tokens,
            'first_mask_top8_logits': [float(x) for x in top.values.tolist()],
        },
        'elapsed_sec': round(time.time() - t0, 3),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    NOTE.write_text(
        '# research — scale1.75 100M HF load smoke\n\n'
        f'Checkpoint: `{rel(CKPT)}`\n\n'
        f'Loaded through `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)` on CPU: **{ok}**.\n\n'
        f'Total params: {params["total"]:,}; adapter-named params: {params["adapter_named"]:,}.\n\n'
        f'Tiny deterministic masked-LM probe loss: `{loss}`; finite logits: `{finite_logits}`.\n\n'
        'This is not a BabyLM score. It verifies official-compatible package loadability and a finite forward pass while the managed full evaluator computes the nine-column surface.\n\n'
        f'JSON: `{rel(OUT_JSON)}`\n',
        encoding='utf-8',
    )
    print(json.dumps({'status': result['status'], 'ok': ok, 'loss': loss, 'out_json': rel(OUT_JSON), 'note': rel(NOTE)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
