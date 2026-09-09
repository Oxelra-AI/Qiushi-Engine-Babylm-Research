#!/usr/bin/env python3
"""research SGCR exact-prefix repair tests.

Tests the live research SGCR module after the audit repair.  This is a
pre-launch scientific-engineering test: it verifies that the SGCR map now encodes
legal16k BPE ancestry rather than padded raw-tokenizer decode/re-encode, that the
support gates use real corpus counts, and that exact-preserving training/eval
contracts still hold on a small model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, PreTrainedTokenizerFast

USER_ROOT = Path('.').resolve()
SCRIPT_DIR = USER_ROOT / "experiments/archive" / 'representation_and_objectives' / 'scripts'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import sgcr_module as sgcr_mod  # noqa: E402

OUT_DIR = USER_ROOT / "experiments/archive" / 'representation_and_objectives' / 'data' / 'sgcr_exact_prefix_repair_tests'
NOTE_PATH = (USER_ROOT / 'research/notes/representation_and_objectives/sgcr_exact_prefix_repair_tests.md')
TOK40 = USER_ROOT / "experiments/archive" / 'representation_and_objectives' / 'data' / 'legal_representation_route_map' / 'tokenizers' / 'legal_byte_bpe_40k'
TOK16 = USER_ROOT / "experiments/archive" / 'representation_and_objectives' / 'data' / 'strictsmall_tokenizer_retrain' / 'strictsmall_compact_reinvest_16k_tokenizer'
POOL = USER_ROOT / "experiments/archive" / 'frontier_consolidation' / 'data' / 'density_cleanqwen_overlay_medium_riskhard' / 'cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def map_hash(mapping: dict[int, list[int]]) -> str:
    payload = json.dumps({str(k): mapping[k] for k in sorted(mapping)}, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def tiny_model(vocab: int = 40000, hidden: int = 64, seed: int = 123) -> DebertaV2ForMaskedLM:
    torch.manual_seed(seed)
    cfg = DebertaV2Config(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        relative_attention=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=3,
    )
    return DebertaV2ForMaskedLM(cfg)


def main() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for child in OUT_DIR.glob('*'):
        if child.is_file() or child.is_symlink():
            child.unlink()

    result: dict[str, object] = {
        'status': 'RUNNING',
        'tok40_tokenizer_sha256': sha256_file(TOK40 / 'tokenizer.json'),
        'tok16_tokenizer_sha256': sha256_file(TOK16 / 'tokenizer.json'),
        'module_sha256': sha256_file(SCRIPT_DIR / bytes((115, 116, 101, 112, 48, 56, 52, 95, 115, 103, 99, 114, 95, 109, 111, 100, 117, 108, 101, 46, 112, 121)).decode('utf-8')),
        'trainer_sha256': sha256_file(SCRIPT_DIR / bytes((115, 116, 101, 112, 48, 56, 52, 95, 115, 103, 99, 114, 95, 116, 114, 97, 105, 110, 101, 114, 46, 112, 121)).decode('utf-8')),
    }

    # 1. Exact BPE-prefix decomposition: no padding, no empty rows.
    decomp_map = sgcr_mod.build_decomposition_map(TOK40, TOK16)
    lengths = [len(decomp_map[i]) for i in range(40000)]
    length_hist = {str(k): int(v) for k, v in sorted(Counter(lengths).items())}
    expected_hist = {'1': 16384, '2': 19333, '3': 3629, '4': 571, '5': 63, '6': 16, '7': 4}
    all_ids_ok = all(0 <= c < 16384 for comps in decomp_map.values() for c in comps)
    nonempty = all(len(comps) > 0 for comps in decomp_map.values())
    decomp_sha = map_hash(decomp_map)
    result['decomposition'] = {
        'token_count': len(decomp_map),
        'length_histogram': length_hist,
        'expected_length_histogram': expected_hist,
        'matches_codex_exact_histogram': length_hist == expected_hist,
        'maximum_components': max(lengths),
        'empty_decompositions': int(sum(1 for x in lengths if x == 0)),
        'all_component_ids_below_16384': all_ids_ok,
        'decomposition_map_sha256': decomp_sha,
        'component_slots': int(sum(lengths)),
        'old_padding_dominated_slots_would_have_been': 40000 * 256,
    }
    assert len(decomp_map) == 40000
    assert length_hist == expected_hist, length_hist
    assert all_ids_ok and nonempty

    # 2. Counts disable tokenizer padding/truncation.
    counts = sgcr_mod.build_pool_counts(TOK40, POOL)
    counts16 = sgcr_mod.build_pool_counts(TOK16, POOL)
    total_tokens = int(sum(counts.values()))
    total_tokens16 = int(sum(counts16.values()))
    used_types = int(sum(1 for v in counts.values() if v > 0))
    used_types16 = int(sum(1 for v in counts16.values() if v > 0))
    result['counts'] = {
        'legal40k_total_tokens': total_tokens,
        'legal16k_total_tokens': total_tokens16,
        'legal40k_used_types': used_types,
        'legal16k_used_types': used_types16,
        'special_counts_0_to_4': {str(i): int(counts.get(i, 0)) for i in range(5)},
        'special_counts16_0_to_4': {str(i): int(counts16.get(i, 0)) for i in range(5)},
    }
    assert total_tokens == 13942644, total_tokens
    assert total_tokens16 == 14669276, total_tokens16
    assert used_types == 39320, used_types

    # 3. Gate construction and mass-calibrated uniform control.
    config = sgcr_mod.SGCRConfig(
        vocab_40k=40000,
        vocab_16k=16384,
        hidden_size=64,
        d_comp=8,
        K=50.0,
        force_standard_ids=[0, 1, 2, 3, 4],
    )
    buffers = sgcr_mod.build_sgcr_buffers(decomp_map, counts, config)
    rho = buffers['rho']
    dlen = buffers['decomp_lengths']
    dids = buffers['decomp_ids']
    counts_tensor = torch.tensor([counts.get(i, 0) for i in range(40000)], dtype=torch.float32)
    forced = torch.tensor([(i in {0, 1, 2, 3, 4}) or dlen[i].item() <= 0 for i in range(40000)])
    ordinary_used = (counts_tensor > 0) & ~forced
    mass_rho = float((counts_tensor[ordinary_used] * rho[ordinary_used]).sum() / counts_tensor[ordinary_used].sum())
    mass_resid = float((counts_tensor[ordinary_used] * (1.0 - rho[ordinary_used])).sum() / counts_tensor[ordinary_used].sum())
    low_support = [i for i in range(40000) if 0 < counts.get(i, 0) < 50]
    # Component support is measured under the legal16k tokenizer, not the direct
    # legal40k token count.  The SGCR gate itself still uses legal40k counts.
    low_all_ge50 = [i for i in low_support if min(counts16.get(c, 0) for c in decomp_map[i]) >= 50]

    uniform_config = sgcr_mod.SGCRConfig(
        vocab_40k=40000,
        vocab_16k=16384,
        hidden_size=64,
        d_comp=8,
        K=50.0,
        uniform_gate=True,
        force_standard_ids=[0, 1, 2, 3, 4],
    )
    uniform_buffers = sgcr_mod.build_sgcr_buffers(decomp_map, counts, uniform_config)
    u_rho = uniform_buffers['rho']
    u_mass_resid = float((counts_tensor[ordinary_used] * (1.0 - u_rho[ordinary_used])).sum() / counts_tensor[ordinary_used].sum())

    result['gates'] = {
        'buffer_shape_decomp_ids': list(dids.shape),
        'buffer_max_comp_len': int(dids.shape[1]),
        'treatment_mass_weighted_rho': mass_rho,
        'treatment_mass_weighted_residual': mass_resid,
        'uniform_mass_weighted_residual': u_mass_resid,
        'uniform_matches_treatment_mass_residual_absdiff': abs(u_mass_resid - mass_resid),
        'forced_special_rhos_0_to_4': {str(i): float(rho[i].item()) for i in range(5)},
        'low_support_used_types': len(low_support),
        'low_support_exact_components_ge50_types': len(low_all_ge50),
        'low_support_exact_components_ge50_fraction': len(low_all_ge50) / len(low_support),
    }
    assert list(dids.shape) == [40000, 7]
    assert all(float(rho[i].item()) == 1.0 for i in range(5))
    assert abs(u_mass_resid - mass_resid) < 1e-7
    assert len(low_support) == 24854
    assert len(low_all_ge50) == 23089

    # 4. Preserve SGCR behavioral contracts on a small but 40k-vocab model.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOK40))
    base_model = tiny_model(seed=7)
    baseline = copy.deepcopy(base_model).eval()
    hooked, sgcr = sgcr_mod.apply_sgcr_to_model(base_model, config, decomp_map, counts)
    hooked = sgcr_mod.sgcr_forward_embedding_hook(hooked, sgcr)
    hooked.eval()

    input_ids = torch.tensor([[1, 20, 100, 2000, 39999, 2], [1, 5, 300, 15000, 25000, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        zsl = sgcr_mod.verify_zero_sharing_limit(hooked, sgcr)
        cold_diff = float((sgcr.effective_embedding_table() - sgcr.word_embeddings.weight).abs().max().item())
        baseline_logits = baseline(input_ids=input_ids, attention_mask=attention_mask).logits
        hooked_logits = hooked(input_ids=input_ids, attention_mask=attention_mask).logits
        cold_logit_diff = float((baseline_logits - hooked_logits).abs().max().item())
    assert zsl['exact_recovery']
    assert cold_diff == 0.0
    assert cold_logit_diff == 0.0

    hooked.train()
    labels = input_ids.clone()
    labels[:, ::2] = -100
    loss = hooked(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()
    first_proj_grad = float(sgcr.component_proj.weight.grad.norm().item())
    first_component_grad = float(sgcr.component_embeddings.weight.grad.norm().item())
    assert first_proj_grad > 0.0
    assert first_component_grad == 0.0
    opt = torch.optim.SGD(hooked.parameters(), lr=0.05)
    opt.step(); opt.zero_grad(set_to_none=True)
    loss2 = hooked(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss2.backward()
    second_component_grad = float(sgcr.component_embeddings.weight.grad.norm().item())
    assert second_component_grad > 0.0

    baked_dir = OUT_DIR / 'baked_standard_model'
    save_info = sgcr_mod.save_baked_checkpoint(hooked, sgcr, tokenizer, baked_dir, also_save_sgcr=True)
    state_keys = sorted(load_file(str(baked_dir / 'model.safetensors')).keys())
    nonstandard = [k for k in state_keys if k.startswith('_sgcr')]
    loaded, loading_info = DebertaV2ForMaskedLM.from_pretrained(baked_dir, output_loading_info=True)
    sidecar = torch.load(baked_dir / 'sgcr_components.pt', map_location='cpu', weights_only=True)

    result['behavior'] = {
        'zero_sharing_limit': zsl,
        'cold_effective_table_max_diff': cold_diff,
        'cold_logit_max_diff_vs_standard': cold_logit_diff,
        'first_projection_grad_norm': first_proj_grad,
        'first_component_grad_norm': first_component_grad,
        'second_component_grad_norm': second_component_grad,
        'save_info': save_info,
        'baked_nonstandard_keys': nonstandard,
        'loaded_missing_keys': loading_info['missing_keys'],
        'loaded_unexpected_keys': loading_info['unexpected_keys'],
        'loaded_param_count': sum(p.numel() for p in loaded.parameters()),
        'sidecar_has_base_word_embeddings': 'base_word_embeddings' in sidecar,
    }
    assert save_info['restore_max_diff_after_save'] == 0.0
    assert not nonstandard
    assert loading_info['missing_keys'] == []
    assert loading_info['unexpected_keys'] == []
    assert 'base_word_embeddings' in sidecar

    result['status'] = 'SGCR_EXACT_PREFIX_REPAIR_TESTS_PASSED'
    (OUT_DIR / 'sgcr_exact_prefix_repair_tests.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    NOTE_PATH.write_text(
        '# research SGCR exact-prefix repair tests\n\n'
        f"Status: `{result['status']}`.\n\n"
        f"- Exact decomposition histogram: `{length_hist}`; map SHA `{decomp_sha}`.\n"
        f"- Counts: {total_tokens} legal40k pool tokens across {used_types} used types; {total_tokens16} legal16k component-token counts. Special ids 0-4 legal40k counts `{result['counts']['special_counts_0_to_4']}`.\n"
        f"- Decomposition buffer shape is `{list(dids.shape)}`, not the old 40k x 256 padded map.\n"
        f"- K=50 treatment mass-weighted residual is {mass_resid:.10f}; uniform control residual matches at {u_mass_resid:.10f}.\n"
        f"- Low-support used legal40k types with all exact components >=50: {len(low_all_ge50)}/{len(low_support)} = {len(low_all_ge50)/len(low_support):.6f}.\n"
        f"- Cold exact table/logit diff 0.0; projection gradient live at first backward ({first_proj_grad:.6g}); component gradient live after one update ({second_component_grad:.6g}).\n"
        f"- Baked checkpoint loads as standard HF model with no `_sgcr` keys and sidecar contains `base_word_embeddings`.\n"
        f"- JSON: `{OUT_DIR / 'sgcr_exact_prefix_repair_tests.json'}`\n",
        encoding='utf-8',
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
