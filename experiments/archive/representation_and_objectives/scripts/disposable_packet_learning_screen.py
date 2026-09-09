#!/usr/bin/env python3
"""research: disposable role-switch packet learning screen.

This is explicitly NOT a submission-facing training route.  It starts from an
already-trained checkpoint only to test whether a small role-switch text packet
can teach context-conditioned binding that transfers to unseen templates and an
unseen relation family.  Because adding these packets after 100M/10M would enlarge
the governing corpus union, the resulting checkpoints must not be interpreted as
legal BabyLM endpoints.

Arms:
  - treatment: role-exchange packets from research expanded suite.
  - role_fixed: structurally matched control preserving context/template/entity
    mass while using one fixed alternative under both context directions.

Readout:
  frozen exchange scores on the research scoring pairs, split by train/held_out
  style and family, using the same target-mask PLL as research.  We report delta
  vs the base reheat checkpoint.
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
import os
import pathlib
import random
import shutil
import sys
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_cosine_schedule_with_warmup

USER_ROOT = pathlib.Path('.')
BASE_TRAINER = USER_ROOT / 'experiments/archive/representation_and_objectives/scripts/accumulated_masking_curriculum_trainer.py'
SCORER = USER_ROOT / 'experiments/archive/representation_and_objectives/scripts/frozen_exchange_scoring.py'
PACKET_DIR = USER_ROOT / 'experiments/archive/representation_and_objectives/data/role_switch_expanded_packets'
TOK_DIR = USER_ROOT / 'experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
BASE_CKPT = USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M'
OUT_ROOT = USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/disposable_role_switch_learning_screen'
SUMMARY_DIR = USER_ROOT / 'experiments/archive/representation_and_objectives/data/disposable_packet_learning_screen'


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = load_module(BASE_TRAINER, 'trainer_for_packet_screen')
scoremod = load_module(SCORER, 'scoremod_for_packet_screen')


class JsonlPacketDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, seq_length: int):
        self.examples = [base.base.Example(text=r['text'], words=int(r['word_count']), example_id=i, source=r.get('family', 'packet')) for i, r in enumerate(rows)]
        self.dataset = base.base.MaskedChunkDataset(self.examples, tokenizer, seq_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def reset_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def packet_iterator(dataset, batch_size, accum_steps, repeats, seed):
    # Deterministic epoch shuffling to avoid learning only file order.  Each
    # yielded object is one effective optimizer batch, possibly with a short last
    # batch; loss normalization uses actual masked-token counts.
    n = len(dataset)
    rng = random.Random(seed)
    for epoch in range(repeats):
        indices = list(range(n))
        rng.shuffle(indices)
        buf = []
        for idx in indices:
            buf.append(dataset[idx])
            if len(buf) == batch_size:
                yield epoch, base.combine_microbatches([base.base.collate(buf[i:i+batch_size//accum_steps]) for i in range(0, len(buf), batch_size//accum_steps)])
                buf = []
        if buf:
            # Pad no examples; accept short final effective batch.
            micro = max(1, batch_size // accum_steps)
            yield epoch, base.combine_microbatches([base.base.collate(buf[i:i+micro]) for i in range(0, len(buf), micro)])


def train_arm(arm: str, rows: list[dict[str, Any]], args) -> pathlib.Path:
    run_dir = OUT_ROOT / arm
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / 'hf_model' / 'screen_final'
    if ckpt_dir.exists() and not args.overwrite:
        print(json.dumps({'event': 'train_skip_existing', 'arm': arm, 'ckpt': str(ckpt_dir)}), flush=True)
        return ckpt_dir

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    tokenizer = base.base.make_portable_tokenizer(str(TOK_DIR))
    dataset = JsonlPacketDataset(rows, tokenizer, args.seq_length)
    reset_all(args.seed)
    model = AutoModelForMaskedLM.from_pretrained(str(BASE_CKPT))
    reset_all(args.train_rng_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.repeats
    warmup = min(args.warmup_steps, max(1, total_steps // 5))
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)
    curr = base.base.MaskingCurriculumState(curriculum='wwm_fixed', mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob, switch_frac=0.7, amlm_window=10, amlm_lambda=0.2)
    curr.initialize(vocab_size=len(tokenizer), total_steps=total_steps)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed)

    manifest = {
        'status': 'DISPOSABLE_PACKET_SCREEN_TRAINING',
        'legal_status': 'not_submission_facing; starts from already-trained 100M checkpoint and adds generated packets after corpus consumption',
        'arm': arm,
        'base_checkpoint': str(BASE_CKPT),
        'tokenizer': str(TOK_DIR),
        'rows': len(rows),
        'unique_packet_words': sum(int(r['word_count']) for r in rows),
        'repeats': args.repeats,
        'word_exposure': sum(int(r['word_count']) for r in rows) * args.repeats,
        'batch_size': args.batch_size,
        'micro_batch_size': args.micro_batch_size,
        'lr': args.lr,
        'warmup_steps': warmup,
        'mask_prob': args.mask_prob,
        'seed': args.seed,
        'train_rng_seed': args.train_rng_seed,
    }
    (run_dir / 'screen_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'train_start', **manifest}), flush=True)

    log_path = run_dir / 'training_log.jsonl'
    step = 0
    with log_path.open('w', encoding='utf-8') as logf:
        for epoch, batch in packet_iterator(dataset, args.batch_size, args.batch_size // args.micro_batch_size, args.repeats, args.seed + 17):
            step += 1
            words = int(batch.pop('words').sum().item())
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_group = batch['word_group'].to(device)
            curr.current_step = step - 1
            masked_inputs, labels = base.base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, curr, gen)
            n_pred_total = int((labels != -100).sum().item())
            if n_pred_total <= 0:
                continue
            optim.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            active_micro = 0
            for start in range(0, input_ids.shape[0], args.micro_batch_size):
                end = min(start + args.micro_batch_size, input_ids.shape[0])
                sl_labels = labels[start:end]
                n_pred_i = int((sl_labels != -100).sum().item())
                if n_pred_i <= 0:
                    continue
                out = model(input_ids=masked_inputs[start:end], attention_mask=attention_mask[start:end], labels=sl_labels)
                loss = out.loss
                scale = n_pred_i / n_pred_total
                (loss * scale).backward()
                weighted_loss += float(loss.detach().cpu()) * scale
                active_micro += 1
                del out, loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            update_lr = optim.param_groups[0]['lr']
            optim.step(); sched.step()
            rec = {
                'event': 'train', 'arm': arm, 'step': step, 'epoch': epoch + 1,
                'loss': weighted_loss, 'update_lr': update_lr, 'batch_words': words,
                'masked_tokens': n_pred_total, 'active_microbatches': active_micro,
            }
            logf.write(json.dumps(rec) + '\n'); logf.flush()
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps(rec), flush=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    print(json.dumps({'event': 'train_done', 'arm': arm, 'steps': step, 'ckpt': str(ckpt_dir)}), flush=True)
    return ckpt_dir


def score_checkpoint(label: str, ckpt: pathlib.Path, pairs: list[dict[str, Any]], gpu: int | None) -> dict[str, Any]:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    device = f'cuda:{gpu}' if (gpu is not None and torch.cuda.is_available()) else ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(str(TOK_DIR), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt)).to(device)
    model.eval()
    results = scoremod.score_all_pairs(model, tokenizer, pairs, device)
    out_path = SUMMARY_DIR / f'{label}_pair_results.jsonl'
    with out_path.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    summary = scoremod.summarize(results, label)
    summary['pair_results'] = str(out_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def delta_summary(base_s: dict[str, Any], arm_s: dict[str, Any]) -> dict[str, Any]:
    def get(path, obj):
        cur = obj
        for p in path:
            cur = cur[p]
        return cur
    d = {'overall': {}}
    for k in ['M_mean', 'M_pos_frac', 'CM_mean', 'bias_abs_mean', 'acc_AB', 'acc_BA', 'both_correct']:
        d['overall'][k] = get(['overall', k], arm_s) - get(['overall', k], base_s)
    d['by_family_both_correct'] = {}
    d['by_family_M_mean'] = {}
    for fam in sorted(set(base_s['by_family']) & set(arm_s['by_family'])):
        d['by_family_both_correct'][fam] = arm_s['by_family'][fam]['both_correct'] - base_s['by_family'][fam]['both_correct']
        d['by_family_M_mean'][fam] = arm_s['by_family'][fam]['M_mean'] - base_s['by_family'][fam]['M_mean']
    d['by_style_both_correct'] = {}
    for st in sorted(set(base_s['by_style']) & set(arm_s['by_style'])):
        d['by_style_both_correct'][st] = arm_s['by_style'][st]['both_correct'] - base_s['by_style'][st]['both_correct']
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['treatment', 'role_fixed', 'both'], default='both')
    ap.add_argument('--repeats', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--micro_batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--warmup_steps', type=int, default=5)
    ap.add_argument('--mask_prob', type=float, default=0.15)
    ap.add_argument('--seq_length', type=int, default=128)
    ap.add_argument('--seed', type=int, default=145145)
    ap.add_argument('--train_rng_seed', type=int, default=145245)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--log_every', type=int, default=20)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--score_only', action='store_true')
    args = ap.parse_args()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    treat_rows = read_jsonl(PACKET_DIR / 'train_treatment.jsonl')
    fixed_rows = read_jsonl(PACKET_DIR / 'train_role_fixed_control.jsonl')
    pairs = read_jsonl(PACKET_DIR / 'scoring_pairs.jsonl')

    ckpts = {'base_reheat': BASE_CKPT}
    if not args.score_only:
        if args.arm in ['treatment', 'both']:
            ckpts['treatment'] = train_arm('treatment', treat_rows, args)
        if args.arm in ['role_fixed', 'both']:
            ckpts['role_fixed'] = train_arm('role_fixed', fixed_rows, args)
    else:
        if args.arm in ['treatment', 'both']:
            ckpts['treatment'] = OUT_ROOT / 'treatment' / 'hf_model' / 'screen_final'
        if args.arm in ['role_fixed', 'both']:
            ckpts['role_fixed'] = OUT_ROOT / 'role_fixed' / 'hf_model' / 'screen_final'

    summaries = {}
    for label, ckpt in ckpts.items():
        print(json.dumps({'event': 'score_start', 'label': label, 'ckpt': str(ckpt)}), flush=True)
        summaries[label] = score_checkpoint(label, ckpt, pairs, args.gpu)
        print(json.dumps({'event': 'score_done', 'label': label, 'overall': summaries[label]['overall']}), flush=True)

    deltas = {}
    if 'base_reheat' in summaries:
        for label in summaries:
            if label != 'base_reheat':
                deltas[label + '_minus_base'] = delta_summary(summaries['base_reheat'], summaries[label])
        if 'treatment' in summaries and 'role_fixed' in summaries:
            deltas['treatment_minus_role_fixed'] = delta_summary(summaries['role_fixed'], summaries['treatment'])

    combined = {
        'status': 'DISPOSABLE_PACKET_LEARNING_SCREEN',
        'legal_status': 'not_submission_facing; causal low-cost screen only',
        'args': vars(args),
        'packet_manifest': str(PACKET_DIR / 'manifest.json'),
        'summaries': summaries,
        'deltas': deltas,
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/disposable_packet_learning_screen.py')),
    }
    combined_path = SUMMARY_DIR / 'learning_screen_summary.json'
    combined_path.write_text(json.dumps(combined, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'SCREEN_DONE', 'summary': str(combined_path)}), flush=True)


if __name__ == '__main__':
    main()
