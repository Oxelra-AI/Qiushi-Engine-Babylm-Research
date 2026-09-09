#!/usr/bin/env python3
"""research: mechanical parity probe for the repaired curriculum trainer.

Purpose: before spending another 100M-word run, verify that the new curriculum
trainer reduces to the trusted legal40k accumulated trainer when curriculum is
fixed 256 and that the active endpoint uses the same seed coordinate as the
baseline (seed=43, extra/init=43022, train_rng_seed=43023).

The probe checks:
  1. fixed-256 dataset/chunk equality against base.MaskedChunkDataset
  2. short-tail handling: min_chunk_tokens=1 retains corpus tails that min8 drops
  3. model initialization equality against the research creation path
  4. first effective-batch equality and identical WWM masks with train seed 43023
  5. first AdamW update equality under the same dropout RNG and scheduler
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

USER_ROOT = Path('.').resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / 'experiments/archive/compact_experience/scripts'
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))
import masking_curriculum_trainer as base  # noqa: E402

PATH = USER_ROOT / 'experiments/archive/representation_and_objectives/scripts/curriculum_trainer_c2bfc720.py'
spec = importlib.util.spec_from_file_location('curriculum_trainer', PATH)
cur = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(cur)

DEFAULT_JSONL = USER_ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
DEFAULT_TOKENIZER = USER_ROOT / 'experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
DEFAULT_OUTDIR = USER_ROOT / 'experiments/archive/representation_and_objectives/data/curriculum_parity_probe'
DEFAULT_NOTE = USER_ROOT / 'research/notes/representation_and_objectives/curriculum_parity_probe.md'
BASELINE_METRICS = USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/scientific_metrics.json'


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def reset_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha_tensor(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]


def state_digest(model: torch.nn.Module) -> dict[str, Any]:
    h = hashlib.sha256()
    n_tensors = 0
    n_params = 0
    samples = {}
    for k, v in model.state_dict().items():
        vv = v.detach().cpu().contiguous()
        h.update(k.encode('utf-8'))
        h.update(str(tuple(vv.shape)).encode('utf-8'))
        h.update(vv.numpy().tobytes())
        n_tensors += 1
        n_params += vv.numel()
        if len(samples) < 8:
            samples[k] = sha_tensor(vv)
    return {'sha256': h.hexdigest(), 'n_tensors': n_tensors, 'n_params': n_params, 'samples': samples}


def max_state_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    max_abs = 0.0
    max_key = None
    n_bad = 0
    for k, va in a.items():
        vb = b[k]
        d = (va.detach().cpu() - vb.detach().cpu()).abs().max().item()
        if d > max_abs:
            max_abs = float(d); max_key = k
        if d != 0.0:
            n_bad += 1
    return {'max_abs': max_abs, 'max_key': max_key, 'n_nonzero_tensors': n_bad, 'n_tensors': len(a)}


def combine_microbatches(micro_batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        'input_ids': torch.cat([b['input_ids'] for b in micro_batches], dim=0),
        'attention_mask': torch.cat([b['attention_mask'] for b in micro_batches], dim=0),
        'word_group': torch.cat([b['word_group'] for b in micro_batches], dim=0),
        'words': torch.cat([b['words'] for b in micro_batches], dim=0),
    }


def first_effective_batch(loader: DataLoader, accum_steps: int) -> dict[str, torch.Tensor]:
    buf = []
    for b in loader:
        buf.append(b)
        if len(buf) == accum_steps:
            return combine_microbatches(buf)
    if buf:
        return combine_microbatches(buf)
    raise RuntimeError('empty loader')


def build_model_like_step061(tokenizer, seed: int, extra_init_seed: int):
    args = argparse.Namespace(
        hidden_size=480,
        n_layer=8,
        n_head=8,
        ffn_mult=4,
        max_position_embeddings=512,
        max_seq_length=256,
        position_buckets=256,
        max_relative_positions=256,
        deberta_pos_att_type='p2c,c2p',
    )
    reset_rng(seed)
    if extra_init_seed >= 0:
        reset_rng(extra_init_seed)
    return base.build_model(args, tokenizer)


def make_curriculum_state(vocab_size: int, total_steps: int):
    st = base.MaskingCurriculumState(
        curriculum='wwm_fixed',
        mask_prob_start=0.15,
        mask_prob_end=0.15,
        switch_frac=1.0,
    )
    st.initialize(vocab_size=vocab_size, total_steps=total_steps)
    return st


def mask_batch(batch: dict[str, torch.Tensor], tokenizer, device: torch.device, total_steps: int, train_seed: int):
    st = make_curriculum_state(len(tokenizer), total_steps)
    st.current_step = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    word_group = batch['word_group'].to(device)
    masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, st, gen)
    return masked_inputs, attention_mask, labels


def one_update(model: torch.nn.Module, batch: dict[str, torch.Tensor], tokenizer, device: torch.device,
               total_steps: int, train_seed: int) -> tuple[float, dict[str, torch.Tensor], dict[str, Any]]:
    model.to(device)
    model.train()
    reset_rng(train_seed)  # matches trainer state immediately before first dropout draw
    masked_inputs, attention_mask, labels = mask_batch(batch, tokenizer, device, total_steps, train_seed)
    n_pred_total = int((labels != -100).sum().item())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, int(total_steps * 0.05)), num_training_steps=total_steps)
    opt.zero_grad(set_to_none=True)
    weighted_loss_sum = 0.0
    active_micro = 0
    for s in range(0, masked_inputs.shape[0], 64):
        e = min(s + 64, masked_inputs.shape[0])
        sl_labels = labels[s:e]
        n_pred_i = int((sl_labels != -100).sum().item())
        if n_pred_i <= 0:
            continue
        out = model(input_ids=masked_inputs[s:e], attention_mask=attention_mask[s:e], labels=sl_labels)
        loss_i = out.loss
        scale = n_pred_i / n_pred_total
        (loss_i * scale).backward()
        weighted_loss_sum += float(loss_i.detach().cpu()) * scale
        active_micro += 1
        del out, loss_i
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
    opt.step(); sched.step()
    post = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    info = {
        'n_pred_total': n_pred_total,
        'active_microbatches': active_micro,
        'grad_norm_before_clip': grad_norm,
        'lr_after_step': float(sched.get_last_lr()[0]),
        'labels_hash': sha_tensor(labels),
        'masked_inputs_hash': sha_tensor(masked_inputs),
    }
    return float(weighted_loss_sum), post, info


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tokenizer = base.make_portable_tokenizer(str(args.tokenizer_path))
    examples, total_file_words, total_rows, sample_rows = base.load_examples_jsonl(args.example_jsonl, args.max_words)
    baseline_metrics = json.loads(BASELINE_METRICS.read_text())

    # Dataset/chunk parity for fixed 256.
    base_dataset = base.MaskedChunkDataset(examples, tokenizer, 256)
    pre_rows = cur.pre_tokenize_all(examples, tokenizer, 256, 0)
    fixed_chunks = []
    for r in pre_rows:
        fixed_chunks.extend(cur.chunk_row(r, 256, tokenizer.pad_token_id, min_tokens=1))
    cur_dataset = cur.PhaseChunkDataset(fixed_chunks)
    dataset_mismatches = []
    for i in range(len(base_dataset)):
        a = base_dataset[i]
        b = cur_dataset[i]
        for key in ['input_ids', 'attention_mask', 'word_group', 'words']:
            av = a[key]
            bv = b[key]
            if isinstance(av, torch.Tensor):
                ok = torch.equal(av, bv)
            else:
                ok = int(av) == int(bv)
            if not ok:
                dataset_mismatches.append({'idx': i, 'key': key})
                break
        if len(dataset_mismatches) >= 5:
            break

    # Short-tail handling on the same subset under the real curriculum's early 64-token chunking.
    tail_rows_min8 = 0; tail_tokens_min8 = 0; tail_rows_min1 = 0; tail_tokens_min1 = 0
    n_chunks_min8 = 0; n_chunks_min1 = 0
    for r in pre_rows:
        c1 = cur.chunk_row(r, 64, tokenizer.pad_token_id, min_tokens=1)
        c8 = cur.chunk_row(r, 64, tokenizer.pad_token_id, min_tokens=8)
        n_chunks_min1 += len(c1); n_chunks_min8 += len(c8)
        tok1 = sum(int(c['attention_mask'].sum().item()) for c in c1)
        tok8 = sum(int(c['attention_mask'].sum().item()) for c in c8)
        if tok1 < r.n_tokens:
            tail_rows_min1 += 1; tail_tokens_min1 += int(r.n_tokens - tok1)
        if tok8 < r.n_tokens:
            tail_rows_min8 += 1; tail_tokens_min8 += int(r.n_tokens - tok8)

    total_steps = math.ceil(len(base_dataset) / 256)
    accum_steps = 4
    base_loader = DataLoader(base_dataset, batch_size=64, shuffle=False, collate_fn=base.collate, num_workers=0)
    cur_loader = DataLoader(cur_dataset, batch_size=64, shuffle=False, collate_fn=cur.collate_chunks, num_workers=0)
    base_batch = first_effective_batch(base_loader, accum_steps)
    cur_batch = first_effective_batch(cur_loader, accum_steps)
    batch_equal = {k: bool(torch.equal(base_batch[k], cur_batch[k])) for k in base_batch}

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    # First-batch masks.
    m1, am1, lab1 = mask_batch(base_batch, tokenizer, device, total_steps, args.train_rng_seed)
    m2, am2, lab2 = mask_batch(cur_batch, tokenizer, device, total_steps, args.train_rng_seed)
    mask_info = {
        'device': str(device),
        'train_rng_seed': args.train_rng_seed,
        'masked_inputs_equal': bool(torch.equal(m1, m2)),
        'labels_equal': bool(torch.equal(lab1, lab2)),
        'attention_equal': bool(torch.equal(am1, am2)),
        'n_masked': int((lab1 != -100).sum().item()),
        'mask_rate': float((lab1 != -100).sum().item() / max(1, am1.bool().sum().item())),
        'labels_hash': sha_tensor(lab1),
        'masked_inputs_hash': sha_tensor(m1),
    }
    del m1, am1, lab1, m2, am2, lab2
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Initialization parity.
    model_a = build_model_like_step061(tokenizer, seed=43, extra_init_seed=43022)
    model_b = build_model_like_step061(tokenizer, seed=43, extra_init_seed=43022)
    init_digest_a = state_digest(model_a)
    init_digest_b = state_digest(model_b)
    init_diff = max_state_diff(model_a.state_dict(), model_b.state_dict())
    cfg = model_a.config
    config_summary = {
        'hidden_size': cfg.hidden_size,
        'num_hidden_layers': cfg.num_hidden_layers,
        'num_attention_heads': cfg.num_attention_heads,
        'intermediate_size': cfg.intermediate_size,
        'pos_att_type': cfg.pos_att_type,
        'max_relative_positions': cfg.max_relative_positions,
        'pad_token_id': cfg.pad_token_id,
        'bos_token_id': cfg.bos_token_id,
        'eos_token_id': cfg.eos_token_id,
        'relative_attention': cfg.relative_attention,
    }

    # First update parity. This is the expensive part but still one effective batch only.
    loss_a, post_a, update_info_a = one_update(model_a, base_batch, tokenizer, device, total_steps, args.train_rng_seed)
    del model_a
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    loss_b, post_b, update_info_b = one_update(model_b, cur_batch, tokenizer, device, total_steps, args.train_rng_seed)
    update_diff = max_state_diff(post_a, post_b)
    del model_b, post_a, post_b
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    summary = {
        'status': 'CURRICULUM_FIXED256_PARITY_PROBE',
        'created_utc': now(),
        'purpose': 'Verify repaired curriculum trainer fixed-256 parity with the trusted legal40k accumulated trainer before a corrected 100M curriculum run.',
        'inputs': {
            'example_jsonl': str(args.example_jsonl),
            'tokenizer_path': str(args.tokenizer_path),
            'max_words': args.max_words,
            'rows': len(examples),
            'actual_words': sum(ex.words for ex in examples),
            'total_file_words': total_file_words,
            'total_file_rows': total_rows,
        },
        'baseline_seed_record': {
            'seed': baseline_metrics.get('seed'),
            'extra_init_seed': baseline_metrics.get('extra_init_seed'),
            'train_rng_seed': baseline_metrics.get('train_rng_seed'),
            'source': str(BASELINE_METRICS),
        },
        'launch_seed_required': {'seed': 43, 'init_seed': 43022, 'train_rng_seed': 43023},
        'fixed256_dataset_parity': {
            'base_len': len(base_dataset),
            'curriculum_fixed_len': len(cur_dataset),
            'all_rows_equal': len(dataset_mismatches) == 0 and len(base_dataset) == len(cur_dataset),
            'first_mismatches': dataset_mismatches,
            'total_steps': total_steps,
        },
        'short_tail_handling_subset_seq64': {
            'min_chunk_tokens_1': {'chunks': n_chunks_min1, 'tail_rows_losing_tokens': tail_rows_min1, 'tail_tokens_lost': tail_tokens_min1},
            'min_chunk_tokens_8': {'chunks': n_chunks_min8, 'tail_rows_losing_tokens': tail_rows_min8, 'tail_tokens_lost': tail_tokens_min8},
            'interpretation': 'Endpoint runs must use min_chunk_tokens=1 so short tails are retained; min8 was only a smoke-test convenience and changes token exposure.',
        },
        'first_effective_batch_parity': {
            'batch_equal': batch_equal,
            'batch_words': int(base_batch['words'].sum().item()),
            'input_ids_hash': sha_tensor(base_batch['input_ids']),
            'word_group_hash': sha_tensor(base_batch['word_group']),
        },
        'first_mask_parity': mask_info,
        'initialization_parity': {
            'config': config_summary,
            'digest_a': init_digest_a,
            'digest_b': init_digest_b,
            'state_diff': init_diff,
            'identical': init_digest_a['sha256'] == init_digest_b['sha256'] and init_diff['max_abs'] == 0.0,
        },
        'first_update_parity': {
            'loss_base_path': loss_a,
            'loss_curriculum_fixed_path': loss_b,
            'loss_abs_diff': abs(loss_a - loss_b),
            'update_info_base': update_info_a,
            'update_info_curriculum_fixed': update_info_b,
            'post_update_state_diff': update_diff,
            'identical_or_close': update_diff['max_abs'] <= 1e-7 and abs(loss_a-loss_b) <= 1e-7,
        },
        'runtime_sec': round(time.time() - t0, 1),
    }
    (args.outdir / 'curriculum_fixed256_parity_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    samples = {
        'base_batch_words_head': base_batch['words'][:16].tolist(),
        'base_input_ids_row0_head': base_batch['input_ids'][0, :80].tolist(),
        'base_word_group_row0_head': base_batch['word_group'][0, :80].tolist(),
        'cur_input_ids_row0_head': cur_batch['input_ids'][0, :80].tolist(),
        'cur_word_group_row0_head': cur_batch['word_group'][0, :80].tolist(),
    }
    (args.outdir / 'first_batch_samples.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    note = []
    note.append('# research — repaired curriculum fixed-256 parity probe\n\n')
    note.append(f'Created: {summary["created_utc"]}.\n\n')
    note.append('The trusted legal40k baseline records seed=43, extra_init_seed=43022, train_rng_seed=43023. The research launch did not pass train_rng_seed and therefore used 43 for masking/dropout; it is not attributable to curriculum and was stopped/failed before useful output.\n\n')
    note.append(f'Fixed-256 dataset parity: `{summary["fixed256_dataset_parity"]}`.\n\n')
    note.append(f'First effective batch parity: `{summary["first_effective_batch_parity"]}`.\n\n')
    note.append(f'First mask parity: `{summary["first_mask_parity"]}`.\n\n')
    note.append(f'Initialization parity: identical={summary["initialization_parity"]["identical"]}, diff=`{summary["initialization_parity"]["state_diff"]}`.\n\n')
    note.append(f'First update parity: `{summary["first_update_parity"]}`.\n\n')
    note.append(f'Short-tail subset result: `{summary["short_tail_handling_subset_seq64"]}`. Endpoint curriculum runs must use `--min_chunk_tokens 1 --train_rng_seed 43023`.\n\n')
    note.append(f'Summary JSON: `{args.outdir / "curriculum_fixed256_parity_summary.json"}`.\n')
    args.note.parent.mkdir(parents=True, exist_ok=True)
    args.note.write_text(''.join(note), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--example_jsonl', type=Path, default=DEFAULT_JSONL)
    ap.add_argument('--tokenizer_path', type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument('--max_words', type=int, default=200064, help='exact row-boundary subset for fast parity')
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--note', type=Path, default=DEFAULT_NOTE)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--train_rng_seed', type=int, default=43023)
    return ap.parse_args()


if __name__ == '__main__':
    run(parse_args())
