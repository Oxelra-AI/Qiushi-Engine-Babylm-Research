#!/usr/bin/env python3
"""research: zero-sharing SGCR parity test against the standard trainer core.

Scientific purpose: before any full H100 SGCR run, verify that the SGCR trainer
has an exact K=0 limit not only for a single forward pass, but for the actual
training update machinery: same random init, same real stream prefix, same WWM
masking generator, same masked-token-weighted microbatch accumulation, same
AdamW/cosine schedule, same gradient clipping, same model architecture.

If SGCR(K=0) diverges from the standard path after two optimizer steps, then a
K=50 endpoint would not isolate support sharing; it would also test hidden
trainer/hook/optimization differences.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

ROOT = Path('.').resolve()
SCRIPTS = ROOT / 'experiments/archive/representation_and_objectives/scripts'
COMPACT_EXPERIENCE = ROOT / 'experiments/archive/compact_experience/scripts'
for p in [SCRIPTS, COMPACT_EXPERIENCE]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import sgcr_module as sgcr  # noqa: E402

OUT = ROOT / 'experiments/archive/representation_and_objectives/data/sgcr_zero_sharing_parity'
NOTE = ROOT / 'research/notes/representation_and_objectives/sgcr_zero_sharing_parity.md'
TRAIN_100M = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
POOL_10M = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
TOK40 = ROOT / 'experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
TOK16 = ROOT / 'experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer'

MAX_WORDS = 78887  # exact first 512-row boundary: two effective batches
BATCH_SIZE = 256
MICRO_BATCH = 64
SEED = 43
EXTRA_INIT_SEED = 43022
TRAIN_RNG_SEED = 43023


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_depth_model(tokenizer):
    reset_all_rng(SEED)
    reset_all_rng(EXTRA_INIT_SEED)
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
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
    model = DebertaV2ForMaskedLM(cfg)
    reset_all_rng(TRAIN_RNG_SEED)
    return model


def combine_microbatches(micro_batches):
    return {
        'input_ids': torch.cat([b['input_ids'] for b in micro_batches], dim=0),
        'attention_mask': torch.cat([b['attention_mask'] for b in micro_batches], dim=0),
        'word_group': torch.cat([b['word_group'] for b in micro_batches], dim=0),
        'words': torch.cat([b['words'] for b in micro_batches], dim=0),
    }


def effective_batches(loader):
    buf = []
    accum_steps = BATCH_SIZE // MICRO_BATCH
    for batch in loader:
        buf.append(batch)
        if len(buf) == accum_steps:
            yield combine_microbatches(buf)
            buf = []
    if buf:
        yield combine_microbatches(buf)


def dedupe_params(params):
    seen = set()
    out = []
    for p in params:
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            out.append(p)
    return out


def train_two_steps(model, tokenizer, batches, *, use_sgcr: bool = False, sgcr_emb=None):
    # Match the inherited trainer: training stochasticity starts from the
    # train_rng_seed immediately before the optimization loop.  This makes the
    # parity test distinguish model/trainer differences from mere dropout RNG
    # drift caused by constructing the second model or SGCR component path.
    reset_all_rng(TRAIN_RNG_SEED)
    model.train()
    params = list(model.parameters())
    if use_sgcr and sgcr_emb is not None:
        params = params + list(sgcr_emb.component_embeddings.parameters()) + list(sgcr_emb.component_proj.parameters())
    unique_params = dedupe_params(params)
    optim = torch.optim.AdamW(unique_params, lr=0.001, weight_decay=0.01, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=1, num_training_steps=2)
    curriculum_state = base.MaskingCurriculumState(
        curriculum='wwm_fixed', mask_prob_start=0.15, mask_prob_end=0.15,
        switch_frac=0.7, amlm_window=10, amlm_lambda=0.2,
    )
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=2)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(TRAIN_RNG_SEED)
    losses = []
    masked_counts = []
    cumulative_words = 0
    for step, batch in enumerate(batches, 1):
        words = int(batch.pop('words').sum().item())
        input_ids = batch['input_ids'][:, :256].contiguous()
        attention_mask = batch['attention_mask'][:, :256].contiguous()
        word_group = batch['word_group'][:, :256].contiguous()
        curriculum_state.current_step = step - 1
        masked_inputs, labels = base.apply_masking_curriculum(
            input_ids, attention_mask, word_group, tokenizer, curriculum_state, gen
        )
        n_pred_total = int((labels != -100).sum().item())
        optim.zero_grad(set_to_none=True)
        weighted_loss_sum = 0.0
        for start in range(0, input_ids.shape[0], MICRO_BATCH):
            end = min(start + MICRO_BATCH, input_ids.shape[0])
            sl_labels = labels[start:end]
            n_pred_i = int((sl_labels != -100).sum().item())
            if n_pred_i <= 0:
                continue
            out = model(input_ids=masked_inputs[start:end], attention_mask=attention_mask[start:end], labels=sl_labels)
            scale = n_pred_i / n_pred_total
            (out.loss * scale).backward()
            weighted_loss_sum += float(out.loss.detach().cpu()) * scale
        grad_norm = float(torch.nn.utils.clip_grad_norm_(unique_params, 1.0).detach().cpu().item())
        optim.step()
        sched.step()
        cumulative_words += words
        losses.append(float(weighted_loss_sum))
        masked_counts.append(n_pred_total)
        if step >= 2:
            break
    return {
        'losses': losses,
        'masked_counts': masked_counts,
        'cumulative_words': cumulative_words,
        'unique_param_count': len(unique_params),
        'last_grad_norm_before_clip': grad_norm,
    }


def max_state_diff(model_a, model_b):
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    common = sorted(set(sd_a) & set(sd_b))
    diffs = []
    for k in common:
        if sd_a[k].shape != sd_b[k].shape or not torch.is_floating_point(sd_a[k]):
            continue
        d = (sd_a[k].detach().float() - sd_b[k].detach().float()).abs()
        diffs.append((float(d.max().item()), float(d.mean().item()), k))
    diffs.sort(reverse=True)
    return diffs[:20]


def main():
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tokenizer = base.make_portable_tokenizer(str(TOK40))
    examples, _, _, _ = base.load_examples_jsonl(TRAIN_100M, MAX_WORDS)
    actual_words = sum(ex.words for ex in examples)
    if actual_words != MAX_WORDS:
        raise RuntimeError(f'word exposure mismatch {actual_words} vs {MAX_WORDS}')
    dataset = base.MaskedChunkDataset(examples, tokenizer, 256)
    loader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False, collate_fn=base.collate, num_workers=0)
    batches = list(effective_batches(loader))
    if len(batches) != 2:
        raise RuntimeError(f'expected two effective batches, got {len(batches)}')
    # Deepcopy batches so the two trainers see identical tensor objects before pop.
    batches_std = [{k: v.clone() for k, v in b.items()} for b in batches]
    batches_sgcr = [{k: v.clone() for k, v in b.items()} for b in batches]

    std_model = build_depth_model(tokenizer)
    sgcr_model = build_depth_model(tokenizer)
    # initial model state identity
    init_diffs = max_state_diff(std_model, sgcr_model)
    if init_diffs and init_diffs[0][0] != 0.0:
        raise RuntimeError(f'initial models differ: {init_diffs[0]}')

    special_ids = sorted(set(int(x) for x in tokenizer.all_special_ids if x is not None))
    decomp = sgcr.build_decomposition_map(TOK40, TOK16)
    counts = sgcr.build_pool_counts(TOK40, POOL_10M)
    conf = sgcr.SGCRConfig(
        vocab_40k=len(tokenizer), vocab_16k=16384, hidden_size=384,
        d_comp=64, K=0.0, uniform_gate=False, init_mode='cold',
        force_standard_ids=special_ids,
    )
    sgcr_model, sgcr_emb = sgcr.apply_sgcr_to_model(sgcr_model, conf, decomp, counts)
    sgcr_model = sgcr.sgcr_forward_embedding_hook(sgcr_model, sgcr_emb)
    zsl = sgcr.verify_zero_sharing_limit(sgcr_model, sgcr_emb)
    if not zsl['exact_recovery']:
        raise RuntimeError(f'K=0 zero-sharing failed: {zsl}')

    std_result = train_two_steps(std_model, tokenizer, batches_std, use_sgcr=False)
    sgcr_result = train_two_steps(sgcr_model, tokenizer, batches_sgcr, use_sgcr=True, sgcr_emb=sgcr_emb)

    diff_after = max_state_diff(std_model, sgcr_model)
    max_diff = diff_after[0][0] if diff_after else 0.0
    mean_top_diff = diff_after[0][1] if diff_after else 0.0
    loss_abs_diffs = [abs(a-b) for a, b in zip(std_result['losses'], sgcr_result['losses'])]

    # Test baked K=0 also recovers standard logits after training.
    with torch.no_grad():
        probe = batches[0]
        input_ids = probe['input_ids'][:8, :64].contiguous()
        attn = probe['attention_mask'][:8, :64].contiguous()
        std_logits = std_model.eval()(input_ids=input_ids, attention_mask=attn).logits.detach().float()
        sgcr_logits = sgcr_model.eval()(input_ids=input_ids, attention_mask=attn).logits.detach().float()
        logits_max_diff = float((std_logits - sgcr_logits).abs().max().item())
        logits_mean_diff = float((std_logits - sgcr_logits).abs().mean().item())

    result = {
        'status': 'SGCR_ZERO_SHARING_PARITY_PASSED' if max_diff < 1e-12 and max(loss_abs_diffs, default=0.0) < 1e-12 and logits_max_diff < 1e-12 else 'SGCR_ZERO_SHARING_PARITY_NONIDENTICAL',
        'actual_words': actual_words,
        'num_examples': len(examples),
        'num_effective_batches': len(batches),
        'architecture': {'hidden_size': 384, 'n_layer': 12, 'n_head': 12, 'intermediate_size': 1280, 'vocab_size': len(tokenizer)},
        'seeds': {'seed': SEED, 'extra_init_seed': EXTRA_INIT_SEED, 'train_rng_seed': TRAIN_RNG_SEED},
        'zero_sharing_limit': zsl,
        'standard_result': std_result,
        'sgcr_k0_result': sgcr_result,
        'loss_abs_diffs': loss_abs_diffs,
        'max_state_diff_after_two_steps': max_diff,
        'mean_of_top_state_diff_after_two_steps': mean_top_diff,
        'top_state_diffs_after_two_steps': diff_after[:10],
        'logits_max_diff_after_two_steps': logits_max_diff,
        'logits_mean_diff_after_two_steps': logits_mean_diff,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    (OUT / 'sgcr_zero_sharing_parity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    note = f"""# research SGCR zero-sharing training parity\n\nStatus: `{result['status']}`\n\nThis test compared a standard 12x384 DeBERTa-v2 two-step training prefix with SGCR at K=0 on the same first 78,887 words of the real 100M stream. It uses identical seeds, WWM masks, masked-token-weighted microbatch accumulation, AdamW/cosine schedule, and gradient clipping.\n\n- Actual words: {actual_words}\n- Effective batches: {len(batches)}\n- Standard losses: {std_result['losses']}\n- SGCR(K=0) losses: {sgcr_result['losses']}\n- Loss absolute differences: {loss_abs_diffs}\n- Max standard-vs-SGCR state diff after two steps: {max_diff}\n- Probe logits max diff after two steps: {logits_max_diff}\n- JSON: `experiments/archive/representation_and_objectives/data/sgcr_zero_sharing_parity/sgcr_zero_sharing_parity.json`\n"""
    NOTE.write_text(note, encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if result['status'] != 'SGCR_ZERO_SHARING_PARITY_PASSED':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
