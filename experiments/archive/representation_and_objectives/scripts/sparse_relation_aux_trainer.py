#!/usr/bin/env python3
"""research sparse WWM-preserving relation auxiliary continuation trainer.

This is the direct successor to the research causal split:
- Standard staged WWM on the exact 70M->80M segment did not damage EWoK.
- Dense PVDM/control target redistribution did damage EWoK/GlobalPIQA surfaces.

Therefore this trainer preserves ordinary WWM corruption and MLM targets, then adds
only a sparse auxiliary loss over legal, hand-coded research event labels.  It does
not alter the input text, word exposure, WWM target distribution, tokenizer, model
architecture, or segment.  Two matched modes are supported:

  semantic: pivot-target binding is positive vs matched nonrelation control.
  placebo: same event positions and computations, but the binary target/control
           label is hash-random, destroying relation semantics while preserving
           auxiliary loss mass and gradient locations in expectation.

Both arms start from compact chck_70M and stop at chck_80M.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

USER_ROOT = Path('.').resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / 'experiments/archive/compact_experience/scripts'
REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = USER_ROOT / 'experiments/archive/representation_and_objectives/scripts'
for p in [str(COMPACT_EXPERIENCE_SCRIPTS), str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_masking_lib as pvdm  # noqa: E402
import pvdm_continuation_trainer as cont  # noqa: E402

STUDY = Path('experiments/archive/representation_and_objectives')
WS = STUDY
DEFAULT_TAIL = cont.DEFAULT_TAIL
DEFAULT_LABELS = cont.DEFAULT_LABELS
DEFAULT_TOKENIZER = cont.DEFAULT_TOKENIZER
DEFAULT_INIT = cont.DEFAULT_INIT
EXPECTED = cont.EXPECTED
DEFAULT_START_TAIL_ROW = cont.DEFAULT_START_TAIL_ROW
DEFAULT_START_TAIL_WORDS = cont.DEFAULT_START_TAIL_WORDS
DEFAULT_INIT_ACTUAL_EXPOSURE = cont.DEFAULT_INIT_ACTUAL_EXPOSURE
DEFAULT_STAGE_WORDS_TO_80M = cont.DEFAULT_STAGE_WORDS_TO_80M
DEFAULT_TOTAL_SCHEDULE_STEPS_70M_TO_100M = cont.DEFAULT_TOTAL_SCHEDULE_STEPS_70M_TO_100M
ALL_CATEGORIES = {'physical_change', 'causal_connector', 'temporal', 'spatial', 'comparative', 'negation'}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RelationAuxHead(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 64, temperature: float = 0.2):
        super().__init__()
        self.q = nn.Linear(hidden_size, rank, bias=False)
        self.k = nn.Linear(hidden_size, rank, bias=False)
        self.temperature = float(temperature)

    def sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        qa = F.normalize(self.q(a), dim=-1)
        kb = F.normalize(self.k(b), dim=-1)
        return (qa * kb).sum(dim=-1) / self.temperature


def group_rep(hidden: torch.Tensor, b: int, positions: list[int]) -> torch.Tensor:
    idx = torch.tensor(positions, dtype=torch.long, device=hidden.device)
    return hidden[b].index_select(0, idx).mean(dim=0)


def select_aux_events(
    word_group: torch.Tensor,
    attention_mask: torch.Tensor,
    label_records: list[dict[str, Any]],
    *,
    seed: int,
    batch_step: int,
    event_prob: float,
    max_events_per_batch: int,
    allowed_categories: set[str],
) -> tuple[list[dict[str, Any]], Counter, Counter]:
    """Select sparse legal events without altering WWM masks.

    At most one event per row is selected.  Selection is deterministic from the
    row/event identity and shared across semantic/placebo modes.
    """
    selected: list[dict[str, Any]] = []
    cat_seen = Counter(); reject = Counter()
    for b, lr in enumerate(label_records):
        group_pos = pvdm.group_positions_for_row(word_group[b], attention_mask[b])
        usable, rej = pvdm.collect_active_events(lr, group_pos, same_length_required=False)
        reject.update(rej)
        candidates = []
        for ev in usable:
            cat_seen[ev.category] += 1
            if ev.category not in allowed_categories:
                reject[f'category_excluded::{ev.category}'] += 1
                continue
            if len({ev.pivot_gid, ev.target_gid, ev.control_gid}) < 3:
                reject['role_overlap'] += 1
                continue
            u = pvdm.hash_uniform(seed, 'sparse_rel_select', int(ev.row_tail_idx), int(ev.event_rank), ev.category)
            if u >= event_prob:
                reject[f'not_selected::{ev.category}'] += 1
                continue
            candidates.append(ev)
        if not candidates:
            continue
        candidates.sort(key=lambda e: pvdm.hash_uniform(seed, 'sparse_rel_rank', int(e.row_tail_idx), int(e.event_rank), e.category, e.target_gid))
        ev = candidates[0]
        selected.append({
            'batch_row': b,
            'row_tail_idx': int(ev.row_tail_idx),
            'event_rank': int(ev.event_rank),
            'category': ev.category,
            'pivot_gid': int(ev.pivot_gid),
            'target_gid': int(ev.target_gid),
            'control_gid': int(ev.control_gid),
            'pivot_positions': group_pos[int(ev.pivot_gid)],
            'target_positions': group_pos[int(ev.target_gid)],
            'control_positions': group_pos[int(ev.control_gid)],
        })
    if len(selected) > max_events_per_batch:
        selected.sort(key=lambda e: pvdm.hash_uniform(seed, 'sparse_rel_batch_cap', batch_step, e['row_tail_idx'], e['event_rank'], e['category']))
        selected = selected[:max_events_per_batch]
    selected.sort(key=lambda e: (e['batch_row'], e['event_rank']))
    return selected, cat_seen, reject


def relation_aux_loss(hidden: torch.Tensor, head: RelationAuxHead, events: list[dict[str, Any]], *, mode: str, seed: int, step: int) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if not events:
        return None, {'events': 0}
    logits = []
    labels = []
    cats = Counter()
    for ev in events:
        b = int(ev['batch_row'])
        p_rep = group_rep(hidden, b, ev['pivot_positions'])
        t_rep = group_rep(hidden, b, ev['target_positions'])
        c_rep = group_rep(hidden, b, ev['control_positions'])
        # Pivot->dependent versus pivot->matched-nonrelation-control.
        s_pos = head.sim(p_rep.unsqueeze(0), t_rep.unsqueeze(0)).squeeze(0)
        s_neg = head.sim(p_rep.unsqueeze(0), c_rep.unsqueeze(0)).squeeze(0)
        # Symmetric dependent->pivot versus dependent->matched-control.  This keeps
        # the auxiliary joint rather than only teaching a pivot-local score.
        s_pos_sym = head.sim(t_rep.unsqueeze(0), p_rep.unsqueeze(0)).squeeze(0)
        s_neg_sym = head.sim(t_rep.unsqueeze(0), c_rep.unsqueeze(0)).squeeze(0)
        logits.append(torch.stack([s_pos, s_neg]))
        logits.append(torch.stack([s_pos_sym, s_neg_sym]))
        if mode == 'semantic':
            lab = 0
        elif mode == 'placebo':
            lab = int(pvdm.hash_uniform(seed, 'placebo_label', step, ev['row_tail_idx'], ev['event_rank'], ev['category']) >= 0.5)
        else:
            raise ValueError(mode)
        labels.append(lab); labels.append(lab)
        cats[str(ev['category'])] += 1
    logit_t = torch.stack(logits, dim=0)
    label_t = torch.tensor(labels, dtype=torch.long, device=hidden.device)
    loss = F.cross_entropy(logit_t, label_t)
    with torch.no_grad():
        pred = logit_t.argmax(dim=-1)
        acc = float((pred == label_t).float().mean().item())
        margin = logit_t[:, 0] - logit_t[:, 1]
        if mode == 'semantic':
            signed_margin = margin
        else:
            sign = torch.where(label_t == 0, torch.ones_like(margin), -torch.ones_like(margin))
            signed_margin = sign * margin
        stats = {
            'events': len(events),
            'terms': int(logit_t.shape[0]),
            'aux_accuracy': acc,
            'mean_semantic_margin': float(margin.mean().item()),
            'mean_signed_margin': float(signed_margin.mean().item()),
            'categories': dict(cats),
            'placebo_label_0_terms': int((label_t == 0).sum().item()),
            'placebo_label_1_terms': int((label_t == 1).sum().item()),
        }
    return loss, stats


def convert_counter_dict(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, Counter): out[k] = dict(v)
        elif isinstance(v, defaultdict): out[k] = {kk: dict(vv) if isinstance(vv, Counter) else vv for kk, vv in v.items()}
        else: out[k] = v
    return out


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Sparse relation auxiliary staged continuation trainer')
    ap.add_argument('--mode', choices=['semantic', 'placebo'], required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--init_checkpoint', default=str(DEFAULT_INIT))
    ap.add_argument('--tail_jsonl', default=str(DEFAULT_TAIL))
    ap.add_argument('--labels_jsonl', default=str(DEFAULT_LABELS))
    ap.add_argument('--tokenizer_path', default=str(DEFAULT_TOKENIZER))
    ap.add_argument('--start_tail_row', type=int, default=DEFAULT_START_TAIL_ROW)
    ap.add_argument('--expected_start_tail_words', type=int, default=DEFAULT_START_TAIL_WORDS)
    ap.add_argument('--initial_actual_word_exposure', type=int, default=DEFAULT_INIT_ACTUAL_EXPOSURE)
    ap.add_argument('--max_word_exposure', type=int, default=DEFAULT_STAGE_WORDS_TO_80M)
    ap.add_argument('--max_rows', type=int, default=0)
    ap.add_argument('--checkpoint_name', default='chck_80M')
    ap.add_argument('--total_schedule_steps', type=int, default=DEFAULT_TOTAL_SCHEDULE_STEPS_70M_TO_100M)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--micro_batch_size', type=int, default=8)
    ap.add_argument('--seq_length', type=int, default=256)
    ap.add_argument('--max_seq_length', type=int, default=256)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--warmup_fraction', type=float, default=0.06)
    ap.add_argument('--mask_prob', type=float, default=0.15)
    ap.add_argument('--aux_weight', type=float, default=0.03)
    ap.add_argument('--aux_rank', type=int, default=64)
    ap.add_argument('--aux_temperature', type=float, default=0.2)
    ap.add_argument('--event_prob', type=float, default=0.35)
    ap.add_argument('--max_events_per_batch', type=int, default=96)
    ap.add_argument('--categories', nargs='*', default=sorted(ALL_CATEGORIES))
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--train_rng_seed', type=int, default=43023)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--log_every', type=int, default=25)
    ap.add_argument('--save_trainer_state', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = build_args()
    if args.batch_size <= 0 or args.micro_batch_size <= 0 or args.batch_size % args.micro_batch_size != 0:
        raise ValueError('batch_size must be a positive multiple of micro_batch_size')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    t0 = time.time(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tail_path = Path(args.tail_jsonl); labels_path = Path(args.labels_jsonl); tok_path = Path(args.tokenizer_path); init_ckpt = Path(args.init_checkpoint)
    hashes = {'tail_sha256': base.sha256_file(tail_path), 'labels_sha256': base.sha256_file(labels_path), 'tokenizer_sha256': base.sha256_file(tok_path / 'tokenizer.json')}
    for k, expected in EXPECTED.items():
        if hashes[k] != expected: raise RuntimeError(f'{k} mismatch {hashes[k]} != {expected}')
    if not init_ckpt.exists(): raise RuntimeError(f'init checkpoint not found: {init_ckpt}')
    tokenizer = base.make_portable_tokenizer(str(tok_path))
    examples, label_records, segment = cont.load_segment(tail_path, labels_path, start_tail_row=args.start_tail_row, expected_start_tail_words=args.expected_start_tail_words, max_word_exposure=args.max_word_exposure, max_rows=args.max_rows)
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    stage_steps = math.ceil(len(dataset) / args.batch_size); schedule_total = max(args.total_schedule_steps, stage_steps); warmup = max(1, int(schedule_total * args.warmup_fraction))
    reset_all_rng(args.seed)
    model = DebertaV2ForMaskedLM.from_pretrained(str(init_ckpt))
    if model.config.vocab_size != len(tokenizer): raise RuntimeError(f'vocab mismatch model {model.config.vocab_size} tokenizer {len(tokenizer)}')
    reset_all_rng(args.train_rng_seed + 1210)
    head = RelationAuxHead(int(model.config.hidden_size), rank=args.aux_rank, temperature=args.aux_temperature)
    reset_all_rng(args.train_rng_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device); head.to(device); model.train(); head.train()
    param_count = sum(p.numel() for p in model.parameters()); head_param_count = sum(p.numel() for p in head.parameters())
    optim = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)
    legacy_state = base.MaskingCurriculumState(curriculum='wwm_fixed', mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob, switch_frac=0.7)
    legacy_state.initialize(vocab_size=len(tokenizer), total_steps=schedule_total)
    legacy_gen = torch.Generator(device=device); legacy_gen.manual_seed(args.train_rng_seed)
    allowed_categories = set(args.categories) & ALL_CATEGORIES
    if not allowed_categories: raise ValueError('no allowed relation categories')
    source_words: dict[str, int] = {}
    for ex in examples: source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    manifest = {
        'status': 'SPARSE_RELATION_AUX_MANIFEST', 'created_utc': now_utc(), 'mode': args.mode,
        'output_dir': str(out), 'init_checkpoint': str(init_ckpt), 'initial_actual_word_exposure': args.initial_actual_word_exposure,
        'target_stage_stop': args.checkpoint_name, 'stage_is_physically_stopped_at_end': True, 'segment': segment, 'hashes': hashes,
        'ordinary_wwm_preserved': True, 'mlm_targets_unchanged_from_standard_legacy': True,
        'semantic_vs_placebo_event_selection_shared': True,
        'effective_batch_size': args.batch_size, 'micro_batch_size': args.micro_batch_size, 'gradient_accumulation_steps': args.batch_size // args.micro_batch_size,
        'stage_steps': stage_steps, 'total_schedule_steps_for_possible_70M_to_100M_resume': schedule_total,
        'optimizer_reset': 'AdamW reset, matched to research standard WWM branch', 'learning_rate': args.learning_rate, 'warmup_fraction': args.warmup_fraction, 'weight_decay': args.weight_decay,
        'mask_prob': args.mask_prob, 'aux_weight': args.aux_weight, 'aux_rank': args.aux_rank, 'aux_temperature': args.aux_temperature,
        'event_prob': args.event_prob, 'max_events_per_batch': args.max_events_per_batch, 'categories': sorted(allowed_categories),
        'seed': args.seed, 'train_rng_seed': args.train_rng_seed, 'source_words_consumed': source_words,
        'causal_split_source': 'research/notes/representation_and_objectives/legacy_80m_causal_split.md',
    }
    (out / 'example_order_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'sparse_relation_aux_start', 'mode': args.mode, 'output_dir': str(out), 'device': str(device), 'param_count': param_count, 'head_param_count': head_param_count, 'stage_rows': len(examples), 'stage_words': segment['selected_words'], 'stage_steps': stage_steps, 'expected_total_at_stage_end': args.initial_actual_word_exposure + segment['selected_words'], 'aux_weight': args.aux_weight, 'event_prob': args.event_prob}), flush=True)
    log_path = out / 'training_log.jsonl'; aux_stats_path = out / 'relation_aux_stats.jsonl'
    loss_values=[]; mlm_loss_values=[]; aux_loss_values=[]; cumulative_words=0
    aggregate = Counter(); cat_stats = Counter(); reject_stats = Counter(); placebo_label_terms=Counter()
    with log_path.open('w', encoding='utf-8') as logf, aux_stats_path.open('w', encoding='utf-8') as auxf:
        for step, batch in enumerate(loader, 1):
            lo = (step - 1) * args.batch_size; hi = lo + int(batch['input_ids'].shape[0])
            words = int(batch.pop('words').sum().item())
            input_ids_cpu = batch['input_ids'][:, :args.seq_length].contiguous(); attention_cpu = batch['attention_mask'][:, :args.seq_length].contiguous(); word_group_cpu = batch['word_group'][:, :args.seq_length].contiguous()
            legacy_state.current_step = step - 1
            masked_dev, labels_dev_full = base.apply_masking_curriculum(input_ids_cpu.to(device, non_blocking=True), attention_cpu.to(device, non_blocking=True), word_group_cpu.to(device, non_blocking=True), tokenizer, legacy_state, legacy_gen)
            masked_cpu, labels_cpu = masked_dev.cpu(), labels_dev_full.cpu(); del masked_dev, labels_dev_full
            n_pred_total = int((labels_cpu != -100).sum().item())
            if n_pred_total <= 0: raise RuntimeError(f'no masked tokens at step {step}')
            all_events, seen, rej = select_aux_events(word_group_cpu, attention_cpu, label_records[lo:hi], seed=args.train_rng_seed, batch_step=step, event_prob=args.event_prob, max_events_per_batch=args.max_events_per_batch, allowed_categories=allowed_categories)
            aggregate['events_selected'] += len(all_events); aggregate['batches'] += 1; aggregate['masked_tokens'] += n_pred_total
            cat_stats.update([e['category'] for e in all_events]); reject_stats.update(rej)
            events_by_mb = defaultdict(list)
            for ev in all_events: events_by_mb[int(ev['batch_row']) // args.micro_batch_size].append(ev)
            optim.zero_grad(set_to_none=True)
            weighted_mlm = 0.0; weighted_aux = 0.0; active_microbatches = 0; total_loss_for_log = 0.0
            for mb_start in range(0, int(masked_cpu.shape[0]), args.micro_batch_size):
                mb_end = min(mb_start + args.micro_batch_size, int(masked_cpu.shape[0])); mb_idx = mb_start // args.micro_batch_size
                sl_labels_cpu = labels_cpu[mb_start:mb_end]; n_pred_i = int((sl_labels_cpu != -100).sum().item())
                if n_pred_i <= 0: continue
                input_ids = masked_cpu[mb_start:mb_end].to(device, non_blocking=True); attention_mask = attention_cpu[mb_start:mb_end].to(device, non_blocking=True); labels_dev = sl_labels_cpu.to(device, non_blocking=True)
                out_model = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_dev, output_hidden_states=True)
                mlm_loss = out_model.loss
                if mlm_loss is None: raise RuntimeError('model returned no MLM loss')
                mlm_scale = n_pred_i / n_pred_total
                hidden = out_model.hidden_states[-1]
                mb_events = []
                for ev in events_by_mb.get(mb_idx, []):
                    ev2 = dict(ev); ev2['batch_row'] = int(ev['batch_row']) - mb_start; mb_events.append(ev2)
                aux_loss, aux_stats = relation_aux_loss(hidden, head, mb_events, mode=args.mode, seed=args.train_rng_seed, step=step)
                loss = mlm_loss * mlm_scale
                aux_float = None
                if aux_loss is not None and all_events:
                    aux_scale = args.aux_weight * (len(mb_events) / len(all_events))
                    loss = loss + aux_loss * aux_scale
                    aux_float = float(aux_loss.detach().cpu())
                    weighted_aux += aux_float * (len(mb_events) / len(all_events))
                    placebo_label_terms['label0'] += int(aux_stats.get('placebo_label_0_terms', 0)); placebo_label_terms['label1'] += int(aux_stats.get('placebo_label_1_terms', 0))
                loss.backward()
                weighted_mlm += float(mlm_loss.detach().cpu()) * mlm_scale
                total_loss_for_log += float(loss.detach().cpu())
                active_microbatches += 1
                del out_model, mlm_loss, loss, input_ids, attention_mask, labels_dev, hidden
            if active_microbatches <= 0: raise RuntimeError(f'no active microbatches at step {step}')
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            optim.step(); sched.step()
            cumulative_words += words
            mlm_loss_values.append(float(weighted_mlm)); aux_loss_values.append(float(weighted_aux) if all_events else 0.0); loss_values.append(float(weighted_mlm + args.aux_weight * weighted_aux))
            rec = {'step': step, 'loss_total_reported': float(weighted_mlm + args.aux_weight * weighted_aux), 'mlm_loss': float(weighted_mlm), 'aux_loss_event_mean': float(weighted_aux) if all_events else None, 'aux_weight': args.aux_weight, 'lr': float(sched.get_last_lr()[0]), 'batch_words': words, 'cumulative_continuation_words': cumulative_words, 'total_actual_word_exposure': args.initial_actual_word_exposure + cumulative_words, 'seq_len': args.seq_length, 'masked_tokens': n_pred_total, 'effective_mask_rate': round(n_pred_total / max(1, int(attention_cpu.sum().item())), 6), 'aux_events': len(all_events), 'mode': args.mode, 'active_microbatches': active_microbatches, 'micro_batch_size': args.micro_batch_size, 'elapsed_sec': round(time.time() - t0, 1)}
            logf.write(json.dumps(rec) + '\n'); logf.flush()
            auxf.write(json.dumps({'step': step, 'events_selected': len(all_events), 'selected_categories': dict(Counter(e['category'] for e in all_events)), 'reject_reasons': dict(rej), 'category_seen': dict(seen), 'placebo_label_terms_cumulative': dict(placebo_label_terms)}, ensure_ascii=False) + '\n'); auxf.flush()
            if step == 1 or step % args.log_every == 0 or step == stage_steps: print(json.dumps({'event': 'train', **rec}), flush=True)
            del masked_cpu, labels_cpu, input_ids_cpu, attention_cpu, word_group_cpu
    ckpt_dir = out / 'hf_model' / args.checkpoint_name
    base.save_hf_checkpoint(model, tokenizer, ckpt_dir); base.save_hf_checkpoint(model, tokenizer, out / 'hf_model')
    aux_dir = out / 'relation_aux_head' / args.checkpoint_name; aux_dir.mkdir(parents=True, exist_ok=True); torch.save({'head': head.state_dict(), 'manifest': manifest}, aux_dir / 'relation_aux_head.pt')
    trainer_state_path = None
    if args.save_trainer_state:
        state_dir = out / 'trainer_state' / args.checkpoint_name; state_dir.mkdir(parents=True, exist_ok=True); trainer_state_path = state_dir / 'optimizer_scheduler_rng.pt'
        torch.save({'optimizer': optim.state_dict(), 'scheduler': sched.state_dict(), 'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, 'python_random_state': random.getstate(), 'numpy_random_state': np.random.get_state(), 'completed_stage_steps': stage_steps, 'cumulative_continuation_words': cumulative_words, 'total_actual_word_exposure': args.initial_actual_word_exposure + cumulative_words, 'last_tail_row': segment['last_tail_row'], 'next_tail_row': int(segment['last_tail_row']) + 1 if segment['last_tail_row'] is not None else None, 'schedule_total_steps': schedule_total, 'relation_aux_head': head.state_dict()}, trainer_state_path)
    metrics = {'variant': f'sparse_relation_aux_{args.mode}', 'backend': 'mlm', 'model_family': 'DebertaV2ForMaskedLM', 'parameter_count': param_count, 'aux_head_parameter_count_not_saved_in_hf_model': head_param_count, 'vocab_size': len(tokenizer), 'word_exposure': args.initial_actual_word_exposure + cumulative_words, 'continuation_words': cumulative_words, 'loss_first': loss_values[0] if loss_values else None, 'loss_last': loss_values[-1] if loss_values else None, 'mlm_loss_first': mlm_loss_values[0] if mlm_loss_values else None, 'mlm_loss_last': mlm_loss_values[-1] if mlm_loss_values else None, 'aux_loss_first': aux_loss_values[0] if aux_loss_values else None, 'aux_loss_last': aux_loss_values[-1] if aux_loss_values else None, 'actual_training_steps': stage_steps, 'effective_batch_size': args.batch_size, 'micro_batch_size': args.micro_batch_size, 'gradient_accumulation_steps': args.batch_size // args.micro_batch_size, 'stage_stop_name': args.checkpoint_name, 'ordinary_wwm_preserved': True, 'saved_checkpoints': [{'name': args.checkpoint_name, 'target_word_exposure': args.initial_actual_word_exposure + args.max_word_exposure, 'actual_cumulative_word_exposure': args.initial_actual_word_exposure + cumulative_words, 'path': str(ckpt_dir)}], 'relation_aux': {'mode': args.mode, 'aux_weight': args.aux_weight, 'aux_rank': args.aux_rank, 'aux_temperature': args.aux_temperature, 'event_prob': args.event_prob, 'max_events_per_batch': args.max_events_per_batch, 'categories': sorted(allowed_categories), 'aggregate': dict(aggregate), 'selected_event_categories': dict(cat_stats), 'reject_reasons': dict(reject_stats), 'placebo_label_terms': dict(placebo_label_terms), 'head_path': str(aux_dir / 'relation_aux_head.pt')}, 'trainer_state_path': str(trainer_state_path) if trainer_state_path else None, 'example_order_manifest': str(out / 'example_order_manifest.json'), 'training_log': str(log_path), 'relation_aux_stats': str(aux_stats_path), **manifest}
    (out / 'scientific_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', 'mode': args.mode, 'output_dir': str(out), 'checkpoint': str(ckpt_dir), 'continuation_words': cumulative_words, 'total_actual_word_exposure': args.initial_actual_word_exposure + cumulative_words, 'loss_first': metrics['loss_first'], 'loss_last': metrics['loss_last'], 'mlm_loss_last': metrics['mlm_loss_last'], 'aux_loss_last': metrics['aux_loss_last'], 'trainer_state_path': metrics['trainer_state_path']}), flush=True)


if __name__ == '__main__':
    main()
