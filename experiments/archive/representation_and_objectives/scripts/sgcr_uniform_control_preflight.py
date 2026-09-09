#!/usr/bin/env python3
"""research CPU-only preflight for the SGCR mass-matched uniform residual control.

This script does not train a model and does not run evaluation. It verifies that
the existing research SGCR module can express the single comparison that would be
scientifically useful if the pending SGCR endpoint is sub-frontier but shows
on-mechanism COMPS/GlobalPIQA gains: same extra parameters and same training-token
residual mass, but no support-dependent allocation.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
SCRIPTS = WS / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sgcr_module as sgcr_mod  # noqa: E402

TOK40 = WS / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
TOK16 = WS / 'data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer'
POOL = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
OUT_DIR = WS / 'data/sgcr_uniform_control_preflight'
OUT_JSON = OUT_DIR / 'sgcr_uniform_control_preflight.json'
OUT_MD = (ROOT / 'research/notes/representation_and_objectives/sgcr_uniform_control_preflight.md')

EXPECTED = {
    'tok40_sha256': '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758',
    'tok16_sha256': '4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738',
    'decomposition_map_sha256': 'b450cc45f7d66564a894fb8cc12e0b0ab8e3ba7db0339cad5eec6f30c6edfd8b',
    'component_slots': 68660,
    'max_components': 7,
    'legal40_pool_tokens': 13942644,
    'legal40_used_types': 39320,
    'base_param_count': 38421952,
    'sgcr_new_params': 1073536,
    'total_param_count': 39495488,
    'K': 50.0,
    'd_comp': 64,
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def decomposition_map_sha256(decomp_map: dict[int, list[int]]) -> str:
    payload = json.dumps({str(k): decomp_map[k] for k in sorted(decomp_map)}, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def fmean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def quantiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {'min': None, 'p10': None, 'median': None, 'p90': None, 'max': None}
    ys = sorted(float(x) for x in xs)
    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        idx = p * (len(ys) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(ys) - 1)
        w = idx - lo
        return ys[lo] * (1 - w) + ys[hi] * w
    return {'min': ys[0], 'p10': q(0.10), 'median': q(0.50), 'p90': q(0.90), 'max': ys[-1]}


def mass_stats(rho: torch.Tensor, counts: dict[int, int], decomp_lengths: list[int], force_ids: set[int]) -> dict[str, Any]:
    counts_tensor = torch.tensor([counts.get(i, 0) for i in range(len(rho))], dtype=torch.float32)
    force_mask = torch.tensor([(i in force_ids) or decomp_lengths[i] <= 0 for i in range(len(rho))], dtype=torch.bool)
    ordinary_used = (counts_tensor > 0) & ~force_mask
    low = ordinary_used & (counts_tensor < 50)
    common = ordinary_used & (counts_tensor >= 50)
    residual = (1.0 - rho.float()).cpu()

    def subset(mask: torch.Tensor) -> dict[str, Any]:
        vals = residual[mask].tolist()
        token_mass_value = float(counts_tensor[mask].sum().item())
        if mask.any() and token_mass_value > 0.0:
            mass = float((counts_tensor[mask] * residual[mask]).sum() / counts_tensor[mask].sum())
            rho_mass = float((counts_tensor[mask] * rho.float().cpu()[mask]).sum() / counts_tensor[mask].sum())
        else:
            mass = None
            rho_mass = None
        return {
            'types': int(mask.sum().item()),
            'token_mass': int(token_mass_value),
            'rho_mass_weighted': rho_mass,
            'residual_mass_weighted': mass,
            'residual_type_mean': fmean([float(v) for v in vals]),
            'residual_type_quantiles': quantiles([float(v) for v in vals]),
        }

    return {
        'ordinary_used': subset(ordinary_used),
        'low_support_lt50': subset(low),
        'common_ge50': subset(common),
        'force_or_unseen': subset(force_mask | (counts_tensor <= 0)),
        'ordinary_used_rho_min': float(rho[ordinary_used].min().item()) if ordinary_used.any() else None,
        'ordinary_used_rho_max': float(rho[ordinary_used].max().item()) if ordinary_used.any() else None,
        'ordinary_used_unique_rho_rounded_1e12': len({round(float(x), 12) for x in rho[ordinary_used].tolist()}) if ordinary_used.any() else 0,
    }


def build_target_model(vocab_size: int, tokenizer: Any) -> DebertaV2ForMaskedLM:
    cfg = DebertaV2Config(
        vocab_size=vocab_size,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=1280,
        max_position_embeddings=512,
        max_relative_positions=256,
        position_buckets=256,
        relative_attention=True,
        pos_att_type=['p2c', 'c2p'],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return DebertaV2ForMaskedLM(cfg)


def main() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(str(TOK40), use_fast=True)
    force_ids = set(int(x) for x in tok.all_special_ids if x is not None)
    decomp_map = sgcr_mod.build_decomposition_map(TOK40, TOK16)
    counts = sgcr_mod.build_pool_counts(TOK40, POOL)
    decomp_lengths = [len(decomp_map[i]) for i in range(len(tok))]
    hist = dict(sorted(collections.Counter(decomp_lengths).items()))

    treatment_cfg = sgcr_mod.SGCRConfig(
        vocab_40k=len(tok), vocab_16k=16384, hidden_size=384, d_comp=64, K=50.0,
        uniform_gate=False, init_mode='cold', force_standard_ids=sorted(force_ids),
    )
    uniform_cfg = sgcr_mod.SGCRConfig(
        vocab_40k=len(tok), vocab_16k=16384, hidden_size=384, d_comp=64, K=50.0,
        uniform_gate=True, init_mode='cold', force_standard_ids=sorted(force_ids),
    )
    treatment_buffers = sgcr_mod.build_sgcr_buffers(decomp_map, counts, treatment_cfg)
    uniform_buffers = sgcr_mod.build_sgcr_buffers(decomp_map, counts, uniform_cfg)

    # Instantiate one target-geometry model to check exact cold behavior and parameter counts for the actual control.
    torch.manual_seed(43022)
    model = build_target_model(len(tok), tok)
    base_param_count = sum(p.numel() for p in model.parameters())
    model, uniform_emb = sgcr_mod.apply_sgcr_to_model(model, uniform_cfg, decomp_map, counts)
    zsl = sgcr_mod.verify_zero_sharing_limit(model, uniform_emb)
    with torch.no_grad():
        cold_diff = (uniform_emb.effective_embedding_table() - uniform_emb.word_embeddings.weight).abs()
        cold_max_diff = float(cold_diff.max().item())
        cold_mean_diff = float(cold_diff.mean().item())

    uniform_new_params = uniform_emb.new_parameter_count
    total_param_count = base_param_count + uniform_new_params

    treatment_stats = mass_stats(treatment_buffers['rho'], counts, decomp_lengths, force_ids)
    uniform_stats = mass_stats(uniform_buffers['rho'], counts, decomp_lengths, force_ids)
    uniform_absdiff = abs(uniform_stats['ordinary_used']['residual_mass_weighted'] - treatment_stats['ordinary_used']['residual_mass_weighted'])
    low_common_ratio_treatment = (
        treatment_stats['low_support_lt50']['residual_type_mean'] / treatment_stats['common_ge50']['residual_type_mean']
        if treatment_stats['low_support_lt50']['residual_type_mean'] and treatment_stats['common_ge50']['residual_type_mean'] else None
    )
    low_common_ratio_uniform = (
        uniform_stats['low_support_lt50']['residual_type_mean'] / uniform_stats['common_ge50']['residual_type_mean']
        if uniform_stats['low_support_lt50']['residual_type_mean'] and uniform_stats['common_ge50']['residual_type_mean'] else None
    )

    payload = {
        'status': 'SGCR_UNIFORM_CONTROL_PREFLIGHT',
        'created_utc': now_utc(),
        'script_never_trains': True,
        'script_never_runs_evaluation': True,
        'scientific_purpose': 'Prepare the single attribution comparison for a promising-but-ambiguous SGCR endpoint: same extra component estimator and same training-token residual mass as K=50 SGCR, but no support-dependent routing.',
        'use_condition': 'Run the full uniform-control training only if the pending SGCR endpoint is sub-frontier but improves COMPS/GlobalPIQA or Overall over matched depth enough that distinguishing support routing from generic auxiliary capacity will change the next route.',
        'do_not_run_condition': 'Do not run this control if SGCR clears the visible 41.80 leader (reproduce seed43122 instead), or if SGCR is flat/damaging on COMPS and GlobalPIQA relative to matched depth.',
        'paths': {
            'tok40': str(TOK40),
            'tok16': str(TOK16),
            'pool_10m': str(POOL),
        },
        'hashes': {
            'tok40_sha256': sha256_file(TOK40 / 'tokenizer.json'),
            'tok16_sha256': sha256_file(TOK16 / 'tokenizer.json'),
            'decomposition_map_sha256': decomposition_map_sha256(decomp_map),
        },
        'decomposition': {
            'token_count': len(tok),
            'length_histogram': hist,
            'max_components': max(decomp_lengths),
            'component_slots': sum(decomp_lengths),
        },
        'counts': {
            'legal40_pool_tokens': int(sum(counts.values())),
            'legal40_used_types': int(sum(1 for v in counts.values() if v > 0)),
            'force_standard_ids': sorted(force_ids),
        },
        'parameter_check': {
            'base_param_count': base_param_count,
            'sgcr_new_params': uniform_new_params,
            'total_param_count': total_param_count,
        },
        'cold_exactness': {
            'zero_sharing_limit': zsl,
            'uniform_cold_effective_table_max_diff_vs_standard': cold_max_diff,
            'uniform_cold_effective_table_mean_diff_vs_standard': cold_mean_diff,
        },
        'gate_stats': {
            'treatment_K50': treatment_stats,
            'uniform_mass_matched_K50': uniform_stats,
            'uniform_minus_treatment_residual_mass_absdiff': uniform_absdiff,
            'treatment_low_support_to_common_residual_type_mean_ratio': low_common_ratio_treatment,
            'uniform_low_support_to_common_residual_type_mean_ratio': low_common_ratio_uniform,
        },
        'expected': EXPECTED,
        'elapsed_sec': round(time.time() - t0, 3),
    }

    errors: list[str] = []
    if payload['hashes']['tok40_sha256'] != EXPECTED['tok40_sha256']:
        errors.append('tok40 SHA mismatch')
    if payload['hashes']['tok16_sha256'] != EXPECTED['tok16_sha256']:
        errors.append('tok16 SHA mismatch')
    if payload['hashes']['decomposition_map_sha256'] != EXPECTED['decomposition_map_sha256']:
        errors.append('decomposition map SHA mismatch')
    if payload['decomposition']['component_slots'] != EXPECTED['component_slots'] or payload['decomposition']['max_components'] != EXPECTED['max_components']:
        errors.append('decomposition shape mismatch')
    if payload['counts']['legal40_pool_tokens'] != EXPECTED['legal40_pool_tokens'] or payload['counts']['legal40_used_types'] != EXPECTED['legal40_used_types']:
        errors.append('pool count mismatch')
    for k in ['base_param_count', 'sgcr_new_params', 'total_param_count']:
        if payload['parameter_check'][k] != EXPECTED[k]:
            errors.append(f'{k} mismatch: {payload["parameter_check"][k]} != {EXPECTED[k]}')
    if cold_max_diff != 0.0 or not zsl.get('exact_recovery'):
        errors.append('uniform cold exactness failed')
    if uniform_absdiff > 1e-6:
        errors.append(f'uniform residual mass not matched: {uniform_absdiff}')
    if uniform_stats['ordinary_used_unique_rho_rounded_1e12'] != 1:
        errors.append('uniform control has more than one ordinary-used rho value')
    payload['preflight_errors'] = errors
    payload['preflight_passed'] = not errors
    payload['json_output'] = str(OUT_JSON)
    payload['note'] = str(OUT_MD)

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    lines = []
    lines.append('# research — SGCR uniform-control preflight\n\n')
    lines.append('CPU-only; no training and no evaluation. This prepares the mass-matched nonsharing comparison for the pending exact-prefix SGCR endpoint.\n\n')
    lines.append(f"Status: `{'PASSED' if payload['preflight_passed'] else 'FAILED'}`\n\n")
    lines.append('## Why this control would matter\n')
    lines.append('If SGCR improves the measured support-aligned columns (especially COMPS/GlobalPIQA) but remains below the visible leader, the full endpoint alone cannot distinguish support-dependent routing from generic auxiliary embedding capacity. The uniform control keeps the same component table/projection and the same training-token residual mass, but sets one ordinary-token rho value for all tokens.\n\n')
    lines.append('## Preflight numbers\n')
    lines.append(f"- Treatment residual mass: `{treatment_stats['ordinary_used']['residual_mass_weighted']}`\n")
    lines.append(f"- Uniform residual mass: `{uniform_stats['ordinary_used']['residual_mass_weighted']}`\n")
    lines.append(f"- Abs diff: `{uniform_absdiff}`\n")
    lines.append(f"- Treatment low/common residual type-mean ratio: `{low_common_ratio_treatment}`\n")
    lines.append(f"- Uniform low/common residual type-mean ratio: `{low_common_ratio_uniform}`\n")
    lines.append(f"- Ordinary used uniform rho unique count (rounded 1e-12): `{uniform_stats['ordinary_used_unique_rho_rounded_1e12']}`\n")
    lines.append(f"- Params: base `{base_param_count}`, new `{uniform_new_params}`, total `{total_param_count}`\n")
    lines.append(f"- Cold effective-table max diff vs standard: `{cold_max_diff}`\n")
    lines.append(f"- Decomposition SHA: `{payload['hashes']['decomposition_map_sha256']}`; histogram `{hist}`\n\n")
    if errors:
        lines.append('## Problems\n')
        for e in errors:
            lines.append(f'- {e}\n')
        lines.append('\n')
    lines.append('## Use condition\n')
    lines.append(payload['use_condition'] + '\n\n')
    lines.append('## Files\n')
    lines.append(f"- JSON: `{OUT_JSON}`\n")
    OUT_MD.write_text(''.join(lines), encoding='utf-8')

    print(json.dumps({
        'status': payload['status'],
        'preflight_passed': payload['preflight_passed'],
        'preflight_errors': errors,
        'treatment_residual_mass': treatment_stats['ordinary_used']['residual_mass_weighted'],
        'uniform_residual_mass': uniform_stats['ordinary_used']['residual_mass_weighted'],
        'uniform_absdiff': uniform_absdiff,
        'uniform_unique_rho': uniform_stats['ordinary_used_unique_rho_rounded_1e12'],
        'out_json': str(OUT_JSON),
        'note': str(OUT_MD),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
