#!/usr/bin/env python3
"""research: one-step role-switch/fixed apparatus alignment under shared tokenizer.

Runs one effective batch for both in-place replacement corpora with the same shared
intersection tokenizer, same model initialization, same train RNG, and the same
research accumulated-trainer path. The first packet row begins at row 3012, so the
first 256-row effective batch is identical text in both corpora. Valid relaunch
requires identical input IDs, word groups, masks/labels, losses, gradients, and
post-update checkpoint tensors; otherwise the route is confounded before role text
is encountered.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

ROOT = pathlib.Path('.')
TRAINER_PATH = ROOT / 'experiments/archive/representation_and_objectives/scripts/accumulated_masking_curriculum_trainer.py'
CORPUS_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/inplace_packed_role_switch_replacement_corpus'
TOKENIZER_PATH = ROOT / 'experiments/archive/representation_and_objectives/data/shared_intersection_tokenizer/tokenizers/shared_intersection_legal_byte_bpe_40k'
OUT_ROOT = ROOT / 'experiments/archive/representation_and_objectives/data/shared_tokenizer_one_step_alignment'
ARMS = {
    'role_switch': CORPUS_DIR / 'role_switch_inplace_packed_replacement_100M.jsonl',
    'role_fixed': CORPUS_DIR / 'role_fixed_inplace_packed_replacement_100M.jsonl',
}
SEED = 43
INIT_SEED = 43022
TRAIN_RNG_SEED = 43023
FIRST_BATCH_WORDS = 35716
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spec = importlib.util.spec_from_file_location('trainer', TRAINER_PATH)
tr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tr)
base = tr.base


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def state_dict_hash(model) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode('utf-8') + b'\0' + arr.tobytes())
    return h.hexdigest()


def grad_hash_and_norm(model) -> tuple[str, float]:
    h = hashlib.sha256()
    norm_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().cpu().contiguous()
        h.update(name.encode('utf-8') + b'\0' + g.numpy().tobytes())
        norm_sq += float((g.float() ** 2).sum().item())
    return h.hexdigest(), norm_sq ** 0.5


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainer_args() -> argparse.Namespace:
    return argparse.Namespace(
        # Data / tokenizer labels are not used by build_model but retained for record parity.
        tokenizer_path=str(TOKENIZER_PATH), tokenizer_label='shared_intersection_legal_byte_bpe_40k',
        max_word_exposure=FIRST_BATCH_WORDS, checkpoint_words=FIRST_BATCH_WORDS,
        # Masking
        masking_curriculum='wwm_fixed', mask_prob_start=0.15, mask_prob_end=0.15,
        switch_frac=0.7, amlm_window=10, amlm_lambda=0.2,
        # Sequence/model
        seq_length=256, max_seq_length=256, max_position_embeddings=512,
        seq_len_schedule='', hidden_size=480, n_layer=8, n_head=8, ffn_mult=4,
        position_buckets=256, max_relative_positions=256, deberta_pos_att_type='p2c,c2p',
        # Optimizer/repro
        batch_size=256, micro_batch_size=64, learning_rate=0.001, weight_decay=0.01,
        warmup_fraction=0.06, lr_total_steps=2529, num_workers=0,
        seed=SEED, extra_init_seed=INIT_SEED, train_rng_seed=TRAIN_RNG_SEED,
    )


def hash_batch(batch: dict[str, torch.Tensor]) -> dict[str, str]:
    out = {}
    for k in ['input_ids', 'attention_mask', 'word_group']:
        out[k + '_hash'] = hashlib.sha256(batch[k].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return out


def first_batch_record(corpus: pathlib.Path, label: str) -> dict[str, Any]:
    args = trainer_args()
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER_PATH))
    examples, total_file_words, total_rows, sample_rows = base.load_examples_jsonl(corpus, FIRST_BATCH_WORDS)
    words = sum(int(ex.words) for ex in examples)
    if words != FIRST_BATCH_WORDS or len(examples) != 256:
        raise RuntimeError(f'{label}: selected {len(examples)} rows/{words} words, expected 256/{FIRST_BATCH_WORDS}')
    packet_examples = [i for i, ex in enumerate(examples) if str(ex.source).startswith('step146_')]
    if packet_examples:
        raise RuntimeError(f'{label}: first batch unexpectedly contains packet rows {packet_examples[:5]}')

    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False, collate_fn=base.collate, num_workers=0, pin_memory=torch.cuda.is_available())
    accum_steps = args.batch_size // args.micro_batch_size
    total_steps = math.ceil(len(dataset) / args.batch_size)
    if total_steps != 1:
        raise RuntimeError(f'{label}: one-step alignment expected total_steps=1, got {total_steps}')
    batch = next(tr.effective_batch_iterator(loader, accum_steps))
    batch_words = int(batch['words'].sum().item())
    if batch_words != FIRST_BATCH_WORDS:
        raise RuntimeError(f'{label}: effective batch words {batch_words} != {FIRST_BATCH_WORDS}')
    batch_hashes = hash_batch(batch)

    reset_all_rng(args.seed)
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = base.build_model(args, tokenizer).to(DEVICE)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)
    curriculum_state = base.MaskingCurriculumState(curriculum=args.masking_curriculum, mask_prob_start=args.mask_prob_start, mask_prob_end=args.mask_prob_end, switch_frac=args.switch_frac, amlm_window=args.amlm_window, amlm_lambda=args.amlm_lambda)
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=total_steps)

    model.train()
    initial_model_hash = state_dict_hash(model)
    input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
    attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
    word_group = batch['word_group'].to(DEVICE, non_blocking=True)
    input_ids = input_ids[:, :args.seq_length].contiguous()
    attention_mask = attention_mask[:, :args.seq_length].contiguous()
    word_group = word_group[:, :args.seq_length].contiguous()
    curriculum_state.current_step = 0
    masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, curriculum_state, gen)
    n_pred_total = int((labels != -100).sum().item())
    n_candidate = int(attention_mask.bool().sum().item())
    if n_pred_total <= 0:
        raise RuntimeError(f'{label}: no masked tokens')

    optim.zero_grad(set_to_none=True)
    weighted_loss_sum = 0.0
    active_microbatches = 0
    batch_rows = input_ids.shape[0]
    for start in range(0, batch_rows, args.micro_batch_size):
        end = min(start + args.micro_batch_size, batch_rows)
        sl_labels = labels[start:end]
        n_pred_i = int((sl_labels != -100).sum().item())
        if n_pred_i <= 0:
            continue
        out_model = model(input_ids=masked_inputs[start:end], attention_mask=attention_mask[start:end], labels=sl_labels)
        loss_i = out_model.loss
        if loss_i is None:
            raise RuntimeError(f'{label}: missing model loss')
        scale = n_pred_i / n_pred_total
        (loss_i * scale).backward()
        weighted_loss_sum += float(loss_i.detach().cpu()) * scale
        active_microbatches += 1
    grad_hash, grad_norm = grad_hash_and_norm(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step(); sched.step()
    post_model_hash = state_dict_hash(model)

    # Do not write a model checkpoint here: hash-level equality is the relevant
    # apparatus evidence, and avoiding checkpoint writes prevents sandbox root-dir
    # replacement issues. The long screen will create real checkpoints.
    return {
        'label': label,
        'corpus': str(corpus),
        'selected_rows': len(examples),
        'selected_words': words,
        'total_file_words': total_file_words,
        'total_file_rows': total_rows,
        'packet_examples_in_prefix': packet_examples,
        'batch_words': batch_words,
        'sample_rows': sample_rows,
        **batch_hashes,
        'masked_input_hash': hashlib.sha256(masked_inputs.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        'label_hash': hashlib.sha256(labels.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        'n_pred_total': n_pred_total,
        'n_candidate': n_candidate,
        'effective_mask_rate': n_pred_total / max(1, n_candidate),
        'initial_model_hash': initial_model_hash,
        'grad_hash': grad_hash,
        'grad_norm': grad_norm,
        'loss': weighted_loss_sum,
        'post_model_hash': post_model_hash,
        'scheduler_lr_after_step': float(sched.get_last_lr()[0]),
        'active_microbatches': active_microbatches,
        'checkpoint': None,
        'checkpoint_model_sha256': None,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = {arm: first_batch_record(path, arm) for arm, path in ARMS.items()}
    equality_keys = [
        'input_ids_hash','attention_mask_hash','word_group_hash','masked_input_hash','label_hash',
        'n_pred_total','n_candidate','effective_mask_rate','initial_model_hash','grad_hash','grad_norm',
        'loss','post_model_hash','scheduler_lr_after_step','active_microbatches',
    ]
    equality = {k: records['role_switch'][k] == records['role_fixed'][k] for k in equality_keys}
    summary = {
        'status': 'SHARED_TOKENIZER_ONE_STEP_ALIGNMENT',
        'scientific_reason': 'first batch precedes packet rows, so treatment/control must be bitwise identical under shared tokenizer before relaunch',
        'device': str(DEVICE),
        'tokenizer_path': str(TOKENIZER_PATH),
        'first_batch_words': FIRST_BATCH_WORDS,
        'records': records,
        'equality': equality,
        'all_alignment_keys_identical': all(equality.values()),
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/shared_tokenizer_one_step_alignment.py')),
    }
    out = OUT_ROOT / 'shared_tokenizer_one_step_alignment_summary.json'
    out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event':'SHARED_TOKENIZER_ONE_STEP_ALIGNMENT_DONE','summary':str(out),'all_identical':summary['all_alignment_keys_identical'],'loss':records['role_switch']['loss'],'masked_tokens':records['role_switch']['n_pred_total']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
