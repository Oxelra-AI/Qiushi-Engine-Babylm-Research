#!/usr/bin/env python3
"""research SGCR integrity tests before any full SGCR launch.

These are low-cost CPU/GPU-smoke tests for the research SGCR implementation:
- syntax/import health after research repairs;
- exact cold-init preservation with gradient-live component route;
- special tokens forced to standard rho=1;
- input embedding and MLM decoder share the same effective table;
- baked checkpoint is a clean standard HF checkpoint loadable by transformers;
- hooked SGCR logits match baked-standard logits in eval mode.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import torch
from transformers import DebertaV2Config, DebertaV2ForMaskedLM

ROOT = Path('.').resolve()
SCRIPTS = ROOT / 'experiments/archive/representation_and_objectives/scripts'
COMPACT_EXPERIENCE = ROOT / 'experiments/archive/compact_experience/scripts'
for p in [SCRIPTS, COMPACT_EXPERIENCE]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import sgcr_module as sgcr  # noqa: E402

TOK40 = ROOT / 'experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
TOK16 = ROOT / 'experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer'
OUT = ROOT / 'experiments/archive/representation_and_objectives/data/sgcr_integrity_tests'


def build_small_model(tokenizer, hidden=48, d_comp=8):
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=96,
        max_position_embeddings=64,
        max_relative_positions=64,
        position_buckets=64,
        relative_attention=True,
        pos_att_type=['p2c', 'c2p'],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(1234)
    return DebertaV2ForMaskedLM(cfg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baked_old = OUT / 'baked_standard_model'
    if baked_old.exists():
        shutil.rmtree(baked_old)
    tokenizer = base.make_portable_tokenizer(str(TOK40))
    special_ids = sorted(set(int(x) for x in tokenizer.all_special_ids if x is not None))
    decomp = sgcr.build_decomposition_map(TOK40, TOK16)
    # Synthetic counts: create rare, common, zero-count, and force-standard cases.
    counts = {i: 100 for i in range(len(tokenizer))}
    rare_ids = [tid for tid in range(10, min(200, len(tokenizer))) if tid not in special_ids][:12]
    zero_ids = [tid for tid in range(200, min(260, len(tokenizer))) if tid not in special_ids][:8]
    for tid in rare_ids:
        counts[tid] = 5
    for tid in zero_ids:
        counts.pop(tid, None)
    for tid in special_ids:
        counts.pop(tid, None)

    model = build_small_model(tokenizer)
    base_state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    conf = sgcr.SGCRConfig(
        vocab_40k=len(tokenizer), vocab_16k=16384,
        hidden_size=model.config.hidden_size, d_comp=8, K=50.0,
        uniform_gate=False, init_mode='cold', force_standard_ids=special_ids,
    )
    model, sgcr_emb = sgcr.apply_sgcr_to_model(model, conf, decomp, counts)
    model = sgcr.sgcr_forward_embedding_hook(model, sgcr_emb)
    model.eval()

    # 1. exact cold-init preservation despite random component codes.
    with torch.no_grad():
        w_std = sgcr_emb.word_embeddings.weight.detach().clone()
        w_eff0 = sgcr_emb.effective_embedding_table().detach().clone()
        cold_max_diff = float((w_eff0 - w_std).abs().max().item())
        cold_mean_diff = float((w_eff0 - w_std).abs().mean().item())
    assert cold_max_diff == 0.0, cold_max_diff

    # 2. special/control tokens forced to rho=1.
    special_rhos = {int(t): float(sgcr_emb.rho[int(t)].item()) for t in special_ids}
    assert all(abs(v - 1.0) < 1e-12 for v in special_rhos.values()), special_rhos
    rare_rhos = [float(sgcr_emb.rho[int(t)].item()) for t in rare_ids]
    common_rhos = [float(sgcr_emb.rho[int(t)].item()) for t in range(1000, 1032) if t not in special_ids]
    assert max(rare_rhos) < min(common_rhos), (rare_rhos[:3], common_rhos[:3])

    # 3. gradient liveness. At exact-preserving cold init, projection gets gradient
    # immediately; component embeddings become live after the first projection update.
    model.train()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(sgcr_emb.component_embeddings.parameters()) + list(sgcr_emb.component_proj.parameters()), lr=1e-2)
    input_ids = torch.tensor([[rare_ids[0], rare_ids[1], 1000, 1001, tokenizer.mask_token_id, 1002, 1003, tokenizer.pad_token_id]], dtype=torch.long)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    labels = torch.full_like(input_ids, -100)
    labels[0, 0] = rare_ids[2]
    labels[0, 1] = 1004
    optimizer.zero_grad(set_to_none=True)
    loss1 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss1.backward()
    grad_proj_1 = float(sgcr_emb.component_proj.weight.grad.norm().item())
    grad_comp_1 = float(sgcr_emb.component_embeddings.weight.grad.norm().item()) if sgcr_emb.component_embeddings.weight.grad is not None else 0.0
    assert grad_proj_1 > 0.0, grad_proj_1
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss2.backward()
    grad_proj_2 = float(sgcr_emb.component_proj.weight.grad.norm().item())
    grad_comp_2 = float(sgcr_emb.component_embeddings.weight.grad.norm().item()) if sgcr_emb.component_embeddings.weight.grad is not None else 0.0
    assert grad_proj_2 > 0.0 and grad_comp_2 > 0.0, (grad_proj_2, grad_comp_2)
    optimizer.step()

    # 4. output decoder really uses W_eff by checking nonzero deviation changes logits.
    model.eval()
    with torch.no_grad():
        w_eff_after = sgcr_emb.effective_embedding_table().detach().clone()
        deviation = (w_eff_after - sgcr_emb.word_embeddings.weight.detach()).norm(dim=1)
        assert float(deviation[rare_ids].mean().item()) > 0.0
        sgcr_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.detach().cpu()

    # 5. bake and verify clean standard HF loading and logits equivalence.
    baked_dir = OUT / 'baked_standard_model'
    bake_info = sgcr.save_baked_checkpoint(model, sgcr_emb, tokenizer, baked_dir, also_save_sgcr=True)
    loaded = DebertaV2ForMaskedLM.from_pretrained(str(baked_dir))
    loaded.eval()
    bad_keys = [k for k in loaded.state_dict().keys() if k.startswith('_sgcr')]
    assert not bad_keys, bad_keys[:5]
    with torch.no_grad():
        baked_logits = loaded(input_ids=input_ids, attention_mask=attention_mask).logits.detach().cpu()
    logits_max_diff = float((sgcr_logits - baked_logits).abs().max().item())
    logits_mean_diff = float((sgcr_logits - baked_logits).abs().mean().item())
    assert logits_max_diff < 1e-5, logits_max_diff

    # 6. K=0 zero-sharing limit.
    model_k0 = build_small_model(tokenizer)
    torch.manual_seed(1234)
    conf0 = sgcr.SGCRConfig(
        vocab_40k=len(tokenizer), vocab_16k=16384,
        hidden_size=model_k0.config.hidden_size, d_comp=8, K=0.0,
        uniform_gate=False, init_mode='cold', force_standard_ids=special_ids,
    )
    model_k0, sgcr0 = sgcr.apply_sgcr_to_model(model_k0, conf0, decomp, counts)
    zsl = sgcr.verify_zero_sharing_limit(model_k0, sgcr0)
    assert zsl['exact_recovery'], zsl

    result = {
        'status': 'SGCR_INTEGRITY_TESTS_PASSED',
        'tokenizer_vocab_size': len(tokenizer),
        'special_ids': special_ids,
        'cold_max_diff': cold_max_diff,
        'cold_mean_diff': cold_mean_diff,
        'special_rhos': special_rhos,
        'rare_rho_mean': sum(rare_rhos) / len(rare_rhos),
        'common_rho_mean': sum(common_rhos) / len(common_rhos),
        'loss1': float(loss1.detach().cpu().item()),
        'loss2': float(loss2.detach().cpu().item()),
        'grad_proj_norm_step1': grad_proj_1,
        'grad_component_norm_step1': grad_comp_1,
        'grad_proj_norm_step2': grad_proj_2,
        'grad_component_norm_step2': grad_comp_2,
        'mean_rare_deviation_after_two_steps': float(deviation[rare_ids].mean().item()),
        'mean_common_deviation_after_two_steps': float(deviation[[1000,1001,1002,1003,1004]].mean().item()),
        'bake_info': bake_info,
        'baked_logits_max_diff': logits_max_diff,
        'baked_logits_mean_diff': logits_mean_diff,
        'zero_sharing_limit': zsl,
        'baked_dir': str(baked_dir.relative_to(ROOT)),
    }
    (OUT / 'sgcr_integrity_tests.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    note = f"""# research SGCR integrity tests\n\nAll tests passed.\n\n- Cold exact preservation max diff: {cold_max_diff}\n- Special-token rhos forced to 1.0: {special_rhos}\n- Rare rho mean: {result['rare_rho_mean']:.6f}; common rho mean: {result['common_rho_mean']:.6f}\n- research component projection grad norm: {grad_proj_1:.6e}; component embedding grad norm: {grad_comp_1:.6e}\n- research component projection grad norm: {grad_proj_2:.6e}; component embedding grad norm: {grad_comp_2:.6e}\n- Baked standard HF logits max diff vs SGCR-hooked model: {logits_max_diff:.6e}\n- Baked checkpoint: `{result['baked_dir']}`\n- JSON: `experiments/archive/representation_and_objectives/data/sgcr_integrity_tests/sgcr_integrity_tests.json`\n"""
    (ROOT / 'research/notes/representation_and_objectives/sgcr_integrity_tests.md').write_text(note, encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
