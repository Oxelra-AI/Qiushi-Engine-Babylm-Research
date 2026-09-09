#!/usr/bin/env python3
"""research calibration for the research sparse relation auxiliary.

Purpose: before spending two 70M->80M continuations, quantify whether the
semantic vs anchor_permuted contrast is informative on identical batches.
This script performs no optimizer step.  It reuses the exact research event
selection and microbatch-local cross-target matching, then records:

  * full-segment coverage/retention by relation family and match level;
  * pre-update margins and success rates for semantic and anchor_permuted
    anchors, using both the untrained auxiliary head and raw hidden cosine;
  * WWM visibility/selection state of pivot/control/target/negative target;
  * backbone gradient norms for MLM and for the auxiliary, aggregated and by
    relation family, scaled as in the proposed trainer.

The calibration answers whether a paired continuation would resolve the
mechanism, or whether sparse coverage/asymmetric pressure requires repair first.
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
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaV2ForMaskedLM

USER_ROOT = Path('.').resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / 'experiments/archive/compact_experience/scripts'
REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = USER_ROOT / 'experiments/archive/representation_and_objectives/scripts'
for p in [str(COMPACT_EXPERIENCE_SCRIPTS), str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_continuation_trainer as cont  # noqa: E402
import sparse_relation_aux_trainer as relaux  # noqa: E402


DEFAULT_OUT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/sparse_relation_aux_calibration'
DEFAULT_NOTE = USER_ROOT / 'research/notes/representation_and_objectives/sparse_relation_aux_calibration.md'


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def reset_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nested_counter() -> defaultdict[str, Counter]:
    return defaultdict(Counter)


def add_numeric(stats: dict[str, Any], prefix: str, value: float) -> None:
    stats[f'{prefix}_n'] = int(stats.get(f'{prefix}_n', 0)) + 1
    stats[f'{prefix}_sum'] = float(stats.get(f'{prefix}_sum', 0.0)) + float(value)
    stats[f'{prefix}_sumsq'] = float(stats.get(f'{prefix}_sumsq', 0.0)) + float(value) * float(value)
    stats[f'{prefix}_min'] = min(float(value), float(stats.get(f'{prefix}_min', float('inf'))))
    stats[f'{prefix}_max'] = max(float(value), float(stats.get(f'{prefix}_max', float('-inf'))))


def finalize_numeric_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    prefixes = sorted({k[:-2] for k in d if k.endswith('_n')})
    for prefix in prefixes:
        n = int(d.get(f'{prefix}_n', 0))
        s = float(d.get(f'{prefix}_sum', 0.0))
        ss = float(d.get(f'{prefix}_sumsq', 0.0))
        mean = s / n if n else None
        var = max(0.0, ss / n - (mean or 0.0) ** 2) if n else None
        out[prefix] = {
            'n': n,
            'mean': mean,
            'std': math.sqrt(var) if var is not None else None,
            'min': (None if n == 0 else float(d.get(f'{prefix}_min'))),
            'max': (None if n == 0 else float(d.get(f'{prefix}_max'))),
        }
    for k, v in d.items():
        if not (k.endswith('_n') or k.endswith('_sum') or k.endswith('_sumsq') or k.endswith('_min') or k.endswith('_max')):
            if isinstance(v, Counter):
                out[k] = dict(v)
            else:
                out[k] = v
    return out


def grad_norms_by_block(model: torch.nn.Module) -> dict[str, Any]:
    blocks: dict[str, dict[str, float]] = defaultdict(lambda: {'sq': 0.0, 'n_params_with_grad': 0, 'n_elems_with_grad': 0})
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sq = float(torch.sum(g * g).item())
        if name.startswith('deberta.embeddings'):
            block = 'embeddings'
        elif name.startswith('deberta.encoder'):
            block = 'encoder'
        elif name.startswith('cls'):
            block = 'mlm_head'
        else:
            block = 'other_model'
        for key in ['model_all', block]:
            blocks[key]['sq'] += sq
            blocks[key]['n_params_with_grad'] += 1
            blocks[key]['n_elems_with_grad'] += int(g.numel())
    out = {}
    for k, v in blocks.items():
        out[k] = {
            'l2': math.sqrt(float(v['sq'])),
            'n_params_with_grad': int(v['n_params_with_grad']),
            'n_elems_with_grad': int(v['n_elems_with_grad']),
        }
    for k in ['model_all', 'embeddings', 'encoder', 'mlm_head', 'other_model']:
        out.setdefault(k, {'l2': 0.0, 'n_params_with_grad': 0, 'n_elems_with_grad': 0})
    return out


def head_grad_norm(head: torch.nn.Module) -> dict[str, Any]:
    sq = 0.0; nparams = 0; nelems = 0
    for p in head.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sq += float(torch.sum(g * g).item())
        nparams += 1; nelems += int(g.numel())
    return {'l2': math.sqrt(sq), 'n_params_with_grad': nparams, 'n_elems_with_grad': nelems}


def model_zero(model: torch.nn.Module, head: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)


def group_mask_state(input_ids_cpu: torch.Tensor, masked_cpu: torch.Tensor, labels_cpu: torch.Tensor, batch_row: int, positions: list[int]) -> dict[str, int]:
    pos = torch.tensor(positions, dtype=torch.long)
    orig = input_ids_cpu[batch_row].index_select(0, pos)
    masked = masked_cpu[batch_row].index_select(0, pos)
    labels = labels_cpu[batch_row].index_select(0, pos)
    selected = bool((labels != -100).any().item())
    visible_original = bool(torch.equal(orig, masked))
    has_mask_or_random = bool((masked != orig).any().item())
    return {
        'selected_for_mlm': int(selected),
        'visible_original_all_positions': int(visible_original),
        'changed_by_masking_any_position': int(has_mask_or_random),
    }


def update_visibility_stats(vis: defaultdict[str, Counter], prepared: list[dict[str, Any]], input_ids_cpu: torch.Tensor, masked_cpu: torch.Tensor, labels_cpu: torch.Tensor) -> None:
    for ev in prepared:
        cat = str(ev.get('category'))
        roles = {
            'pivot': (int(ev['batch_row']), ev['pivot_positions']),
            'control_anchor': (int(ev['batch_row']), ev['control_positions']),
            'target': (int(ev['batch_row']), ev['target_positions']),
            'negative_target': (int(ev['negative_target_batch_row']), ev['negative_target_positions']),
        }
        for role, (b, pos) in roles.items():
            state = group_mask_state(input_ids_cpu, masked_cpu, labels_cpu, b, pos)
            for k, val in state.items():
                vis[f'{role}::{k}'][cat] += int(val)
                vis[f'{role}::{k}']['ALL'] += int(val)
        vis['events']['ALL'] += 1
        vis['events'][cat] += 1


def score_preupdate(hidden: torch.Tensor, head: relaux.RelationAuxHead, prepared: list[dict[str, Any]], *, mode: str, stats: dict[str, dict[str, Any]]) -> None:
    if not prepared:
        return
    with torch.no_grad():
        for ev in prepared:
            cat = str(ev.get('category'))
            keys = ['ALL', cat]
            b = int(ev['batch_row'])
            if mode == 'semantic':
                anchor_row = b; anchor_positions = ev['pivot_positions']
            elif mode == 'anchor_permuted':
                anchor_row = b; anchor_positions = ev['control_positions']
            else:
                raise ValueError(mode)
            a_rep = relaux.group_rep(hidden, anchor_row, anchor_positions)
            t_rep = relaux.group_rep(hidden, b, ev['target_positions'])
            n_rep = relaux.group_rep(hidden, int(ev['negative_target_batch_row']), ev['negative_target_positions'])
            # Auxiliary-head logits as the proposed loss sees them at initialization.
            s_pos = head.sim(a_rep.unsqueeze(0), t_rep.unsqueeze(0)).squeeze(0)
            s_neg = head.sim(a_rep.unsqueeze(0), n_rep.unsqueeze(0)).squeeze(0)
            s_pos_sym = head.sim(t_rep.unsqueeze(0), a_rep.unsqueeze(0)).squeeze(0)
            s_neg_sym = head.sim(n_rep.unsqueeze(0), a_rep.unsqueeze(0)).squeeze(0)
            margins = [float((s_pos - s_neg).item()), float((s_pos_sym - s_neg_sym).item())]
            # Raw hidden-space cosine, independent of random auxiliary head.
            ar = F.normalize(a_rep.float(), dim=0)
            tr = F.normalize(t_rep.float(), dim=0)
            nr = F.normalize(n_rep.float(), dim=0)
            raw_margin = float(torch.dot(ar, tr).item() - torch.dot(ar, nr).item())
            for key in keys:
                s = stats[key]
                s['events'] = int(s.get('events', 0)) + 1
                s['head_terms'] = int(s.get('head_terms', 0)) + 2
                for m in margins:
                    add_numeric(s, 'head_margin', m)
                    s['head_success_terms'] = int(s.get('head_success_terms', 0)) + int(m > 0.0)
                add_numeric(s, 'raw_cosine_margin', raw_margin)
                s['raw_cosine_success_events'] = int(s.get('raw_cosine_success_events', 0)) + int(raw_margin > 0.0)
                s.setdefault('match_levels', Counter())[str(ev.get('cross_target_match_level', 'NA'))] += 1
                s.setdefault('negative_target_freq_bins', Counter())[str(ev.get('negative_target_freq_bin', 'NA'))] += 1


def prepared_by_microbatch(all_events: list[dict[str, Any]], micro_batch_size: int, *, seed: int, step: int) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    events_by_mb: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ev in all_events:
        events_by_mb[int(ev['batch_row']) // micro_batch_size].append(ev)
    prepared: dict[int, list[dict[str, Any]]] = {}
    step_perm_stats: dict[str, Any] = defaultdict(float)
    step_perm_stats['match_level_counts'] = Counter(); step_perm_stats['group_size_counts'] = Counter()
    total_in = 0; total_used = 0
    for mb_idx, evs in events_by_mb.items():
        mb_start0 = int(mb_idx) * micro_batch_size
        rel_events = []
        for ev in evs:
            ev2 = dict(ev)
            ev2['batch_row'] = int(ev['batch_row']) - mb_start0
            rel_events.append(ev2)
        p, pstats = relaux.assign_cross_targets(rel_events, seed=seed, step=step, microbatch_index=int(mb_idx))
        prepared[int(mb_idx)] = p
        total_in += len(rel_events); total_used += len(p)
        relaux.merge_perm_stats(step_perm_stats, pstats)
    final = relaux.finalize_perm_stats(step_perm_stats)
    final['events_input_to_cross_target_filter'] = int(total_in)
    final['events_used_after_cross_target_filter'] = int(total_used)
    return prepared, final


def flatten_prepared(prepared: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for k in sorted(prepared):
        for ev in prepared[k]:
            e2 = dict(ev)
            e2['_microbatch_index'] = int(k)
            out.append(e2)
    return out


def forward_hidden(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels_dev: torch.Tensor, *, seed: int) -> Any:
    reset_all(seed)
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_dev, output_hidden_states=True)


def accumulate_mlm_grad(model: torch.nn.Module, head: torch.nn.Module, batch_tensors: dict[str, torch.Tensor], tokenizer: Any, legacy_state: Any, legacy_gen: torch.Generator, device: torch.device, *, step: int, args: argparse.Namespace) -> dict[str, Any]:
    # Masks are supplied by caller in batch_tensors.
    masked_cpu = batch_tensors['masked_cpu']; labels_cpu = batch_tensors['labels_cpu']; attention_cpu = batch_tensors['attention_cpu']
    n_pred_total = int((labels_cpu != -100).sum().item())
    model_zero(model, head)
    weighted_loss = 0.0; active = 0
    for mb_start in range(0, int(masked_cpu.shape[0]), args.micro_batch_size):
        mb_end = min(mb_start + args.micro_batch_size, int(masked_cpu.shape[0]))
        sl_labels = labels_cpu[mb_start:mb_end]
        n_pred_i = int((sl_labels != -100).sum().item())
        if n_pred_i <= 0:
            continue
        input_ids = masked_cpu[mb_start:mb_end].to(device, non_blocking=True)
        attention_mask = attention_cpu[mb_start:mb_end].to(device, non_blocking=True)
        labels_dev = sl_labels.to(device, non_blocking=True)
        out = forward_hidden(model, input_ids, attention_mask, labels_dev, seed=args.train_rng_seed + 810000 + step * 1000 + mb_start)
        loss = out.loss * (n_pred_i / n_pred_total)
        loss.backward()
        weighted_loss += float(out.loss.detach().cpu()) * (n_pred_i / n_pred_total)
        active += 1
        del out, loss, input_ids, attention_mask, labels_dev
    norms = grad_norms_by_block(model)
    model_zero(model, head)
    return {'loss': weighted_loss, 'active_microbatches': active, 'masked_tokens': n_pred_total, 'grad_norms': norms}


def accumulate_aux_grad(model: torch.nn.Module, head: torch.nn.Module, batch_tensors: dict[str, torch.Tensor], prepared: dict[int, list[dict[str, Any]]], device: torch.device, *, mode: str, category: str | None, denom_total_events: int, step: int, args: argparse.Namespace) -> dict[str, Any]:
    masked_cpu = batch_tensors['masked_cpu']; labels_cpu = batch_tensors['labels_cpu']; attention_cpu = batch_tensors['attention_cpu']
    if denom_total_events <= 0:
        return {'events_used': 0, 'loss_event_mean_weighted': None, 'grad_norms': None, 'head_grad_norm': None}
    model_zero(model, head)
    weighted_aux = 0.0; active = 0; used_total = 0; terms_total = 0
    cats: Counter = Counter()
    for mb_start in range(0, int(masked_cpu.shape[0]), args.micro_batch_size):
        mb_end = min(mb_start + args.micro_batch_size, int(masked_cpu.shape[0]))
        mb_idx = mb_start // args.micro_batch_size
        evs = prepared.get(int(mb_idx), [])
        if category is not None:
            evs = [e for e in evs if str(e.get('category')) == category]
        if not evs:
            continue
        sl_labels = labels_cpu[mb_start:mb_end]
        input_ids = masked_cpu[mb_start:mb_end].to(device, non_blocking=True)
        attention_mask = attention_cpu[mb_start:mb_end].to(device, non_blocking=True)
        labels_dev = sl_labels.to(device, non_blocking=True)
        out = forward_hidden(model, input_ids, attention_mask, labels_dev, seed=args.train_rng_seed + 820000 + step * 1000 + mb_start)
        hidden = out.hidden_states[-1]
        aux_loss, aux_stats = relaux.relation_aux_loss_prepared(hidden, head, evs, mode=mode)
        if aux_loss is None:
            del out, input_ids, attention_mask, labels_dev
            continue
        used_here = int(aux_stats.get('events_used', 0) or 0)
        scale = args.aux_weight * (used_here / denom_total_events)
        loss = aux_loss * scale
        loss.backward()
        weighted_aux += float(aux_loss.detach().cpu()) * (used_here / denom_total_events)
        used_total += used_here
        terms_total += int(aux_stats.get('terms', 0) or 0)
        cats.update(aux_stats.get('categories', {}))
        active += 1
        del out, hidden, aux_loss, loss, input_ids, attention_mask, labels_dev
    norms = grad_norms_by_block(model)
    hnorm = head_grad_norm(head)
    model_zero(model, head)
    return {'events_used': int(used_total), 'terms': int(terms_total), 'active_microbatches': int(active), 'categories': dict(cats), 'loss_event_mean_weighted': weighted_aux if used_total else None, 'grad_norms': norms, 'head_grad_norm': hnorm, 'scaled_by_aux_weight': args.aux_weight, 'denom_total_events_all_categories': int(denom_total_events), 'category_filter': category or 'ALL'}


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(b)


def finalize_margin_stats(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for cat, s in stats.items():
        f = finalize_numeric_dict(s)
        events = int(s.get('events', 0))
        head_terms = int(s.get('head_terms', 0))
        f['events'] = events
        f['head_terms'] = head_terms
        f['head_success_rate_terms'] = (int(s.get('head_success_terms', 0)) / head_terms) if head_terms else None
        f['raw_cosine_success_rate_events'] = (int(s.get('raw_cosine_success_events', 0)) / events) if events else None
        out[cat] = f
    return out


def summarize_grad_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[str, Any] = defaultdict(lambda: defaultdict(float))
    counts: Counter = Counter()
    by_key: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = rec['key']
        counts[key] += 1
        by_key.setdefault(key, {'batches': 0, 'events_used': 0, 'terms': 0, 'model_l2_values': [], 'encoder_l2_values': [], 'head_l2_values': [], 'ratios_to_mlm_model': [], 'ratios_to_mlm_encoder': []})
        bk = by_key[key]
        bk['batches'] += 1
        bk['events_used'] += int(rec.get('events_used', 0) or 0)
        bk['terms'] += int(rec.get('terms', 0) or 0)
        g = rec.get('grad_norms') or {}
        h = rec.get('head_grad_norm') or {}
        model_l2 = ((g.get('model_all') or {}).get('l2'))
        encoder_l2 = ((g.get('encoder') or {}).get('l2'))
        head_l2 = h.get('l2')
        for name, val in [('model_l2_values', model_l2), ('encoder_l2_values', encoder_l2), ('head_l2_values', head_l2), ('ratios_to_mlm_model', rec.get('ratio_to_mlm_model_l2')), ('ratios_to_mlm_encoder', rec.get('ratio_to_mlm_encoder_l2'))]:
            if val is not None:
                bk[name].append(float(val))
    out = {}
    for key, bk in by_key.items():
        item = {'batches': bk['batches'], 'events_used': bk['events_used'], 'terms': bk['terms']}
        for vals_key in ['model_l2_values', 'encoder_l2_values', 'head_l2_values', 'ratios_to_mlm_model', 'ratios_to_mlm_encoder']:
            vals = bk[vals_key]
            item[vals_key.replace('_values', '')] = {
                'n': len(vals),
                'mean': float(np.mean(vals)) if vals else None,
                'median': float(np.median(vals)) if vals else None,
                'min': float(np.min(vals)) if vals else None,
                'max': float(np.max(vals)) if vals else None,
            }
        out[key] = item
    return out


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Calibrate research sparse relation auxiliary before full continuations')
    ap.add_argument('--output_dir', default=str(DEFAULT_OUT))
    ap.add_argument('--note_path', default=str(DEFAULT_NOTE))
    ap.add_argument('--init_checkpoint', default=str(relaux.DEFAULT_INIT))
    ap.add_argument('--tail_jsonl', default=str(relaux.DEFAULT_TAIL))
    ap.add_argument('--labels_jsonl', default=str(relaux.DEFAULT_LABELS))
    ap.add_argument('--tokenizer_path', default=str(relaux.DEFAULT_TOKENIZER))
    ap.add_argument('--start_tail_row', type=int, default=relaux.DEFAULT_START_TAIL_ROW)
    ap.add_argument('--expected_start_tail_words', type=int, default=relaux.DEFAULT_START_TAIL_WORDS)
    ap.add_argument('--max_word_exposure', type=int, default=relaux.DEFAULT_STAGE_WORDS_TO_80M)
    ap.add_argument('--coverage_batches', type=int, default=250)
    ap.add_argument('--margin_batches', type=int, default=32)
    ap.add_argument('--grad_batches', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--micro_batch_size', type=int, default=8)
    ap.add_argument('--seq_length', type=int, default=256)
    ap.add_argument('--max_seq_length', type=int, default=256)
    ap.add_argument('--mask_prob', type=float, default=0.15)
    ap.add_argument('--aux_weight', type=float, default=0.03)
    ap.add_argument('--aux_rank', type=int, default=64)
    ap.add_argument('--aux_temperature', type=float, default=0.2)
    ap.add_argument('--event_prob', type=float, default=0.35)
    ap.add_argument('--max_events_per_batch', type=int, default=96)
    ap.add_argument('--max_control_distance_abs_diff', type=int, default=1)
    ap.add_argument('--max_control_freq_bin_abs_diff', type=int, default=1)
    ap.add_argument('--categories', nargs='*', default=sorted(relaux.ALL_CATEGORIES))
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--train_rng_seed', type=int, default=43023)
    ap.add_argument('--dropout_mode', choices=['train', 'eval'], default='train')
    ap.add_argument('--num_workers', type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = build_args()
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError('batch_size must be multiple of micro_batch_size')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    t0 = time.time()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    note_path = Path(args.note_path); note_path.parent.mkdir(parents=True, exist_ok=True)
    max_rows = max(args.coverage_batches, args.margin_batches, args.grad_batches) * args.batch_size

    tail_path = Path(args.tail_jsonl); labels_path = Path(args.labels_jsonl); tok_path = Path(args.tokenizer_path); init_ckpt = Path(args.init_checkpoint)
    hashes = {'tail_sha256': base.sha256_file(tail_path), 'labels_sha256': base.sha256_file(labels_path), 'tokenizer_sha256': base.sha256_file(tok_path / 'tokenizer.json')}
    for k, expected in relaux.EXPECTED.items():
        if hashes[k] != expected:
            raise RuntimeError(f'{k} mismatch {hashes[k]} != {expected}')
    tokenizer = base.make_portable_tokenizer(str(tok_path))
    examples, label_records, segment = cont.load_segment(tail_path, labels_path, start_tail_row=args.start_tail_row, expected_start_tail_words=args.expected_start_tail_words, max_word_exposure=args.max_word_exposure, max_rows=max_rows)
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reset_all(args.seed)
    model = DebertaV2ForMaskedLM.from_pretrained(str(init_ckpt))
    if model.config.vocab_size != len(tokenizer):
        raise RuntimeError(f'vocab mismatch model {model.config.vocab_size} tokenizer {len(tokenizer)}')
    reset_all(args.train_rng_seed + 1220)
    head = relaux.RelationAuxHead(int(model.config.hidden_size), rank=args.aux_rank, temperature=args.aux_temperature)
    reset_all(args.train_rng_seed)
    model.to(device); head.to(device)
    if args.dropout_mode == 'train':
        model.train(); head.train()
    else:
        model.eval(); head.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    legacy_state = base.MaskingCurriculumState(curriculum='wwm_fixed', mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob, switch_frac=0.7)
    schedule_total = max(relaux.DEFAULT_TOTAL_SCHEDULE_STEPS_70M_TO_100M, math.ceil(len(dataset) / args.batch_size))
    legacy_state.initialize(vocab_size=len(tokenizer), total_steps=schedule_total)
    legacy_gen = torch.Generator(device=device); legacy_gen.manual_seed(args.train_rng_seed)
    allowed_categories = set(args.categories) & relaux.ALL_CATEGORIES

    manifest = {
        'status': 'SPARSE_RELATION_AUX_CALIBRATION_START',
        'created_utc': now_utc(), 'output_dir': str(outdir), 'note_path': str(note_path),
        'purpose': 'calibrate research sparse relation auxiliary before two full 70M->80M continuations',
        'init_checkpoint': str(init_ckpt), 'segment': segment, 'hashes': hashes,
        'batch_size': args.batch_size, 'micro_batch_size': args.micro_batch_size,
        'coverage_batches_requested': args.coverage_batches, 'margin_batches_requested': args.margin_batches,
        'grad_batches_requested': args.grad_batches, 'dropout_mode': args.dropout_mode,
        'aux_weight': args.aux_weight, 'aux_rank': args.aux_rank, 'aux_temperature': args.aux_temperature,
        'event_prob': args.event_prob, 'max_events_per_batch': args.max_events_per_batch,
        'max_control_distance_abs_diff': args.max_control_distance_abs_diff,
        'max_control_freq_bin_abs_diff': args.max_control_freq_bin_abs_diff,
        'categories': sorted(allowed_categories),
    }
    (outdir / 'calibration_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'calibration_start', 'device': str(device), 'examples': len(examples), 'segment_words': segment.get('selected_words'), 'max_rows_loaded': max_rows, 'dropout_mode': args.dropout_mode}), flush=True)

    coverage = {
        'batches': 0, 'rows': 0, 'words': 0, 'masked_tokens_in_margin_batches': 0,
        'events_selected_before_cross_target_filter': 0,
        'events_input_to_cross_target_filter': 0,
        'events_used_after_cross_target_filter': 0,
        'selected_categories_before_filter': Counter(), 'used_categories_after_filter': Counter(),
        'seen_categories_before_probability': Counter(), 'reject_reasons': Counter(),
        'match_level_counts': Counter(), 'group_size_counts': Counter(),
        'different_row_donor_pairs': 0, 'same_row_donor_pairs': 0,
    }
    vis_stats: defaultdict[str, Counter] = defaultdict(Counter)
    margin_stats: dict[str, dict[str, dict[str, Any]]] = {'semantic': defaultdict(dict), 'anchor_permuted': defaultdict(dict)}
    batch_records_path = outdir / 'batch_calibration_records.jsonl'
    grad_records_path = outdir / 'gradient_records.jsonl'
    grad_records: list[dict[str, Any]] = []

    with batch_records_path.open('w', encoding='utf-8') as brec, grad_records_path.open('w', encoding='utf-8') as grec:
        for step, batch in enumerate(loader, 1):
            if step > args.coverage_batches:
                break
            lo = (step - 1) * args.batch_size; hi = lo + int(batch['input_ids'].shape[0])
            words = int(batch.pop('words').sum().item())
            input_ids_cpu = batch['input_ids'][:, :args.seq_length].contiguous()
            attention_cpu = batch['attention_mask'][:, :args.seq_length].contiguous()
            word_group_cpu = batch['word_group'][:, :args.seq_length].contiguous()
            legacy_state.current_step = step - 1
            all_events, seen, rej = relaux.select_aux_events(word_group_cpu, attention_cpu, label_records[lo:hi], seed=args.train_rng_seed, batch_step=step, event_prob=args.event_prob, max_events_per_batch=args.max_events_per_batch, allowed_categories=allowed_categories, max_control_distance_abs_diff=args.max_control_distance_abs_diff, max_control_freq_bin_abs_diff=args.max_control_freq_bin_abs_diff)
            prepared, pstats = prepared_by_microbatch(all_events, args.micro_batch_size, seed=args.train_rng_seed, step=step)
            prepared_flat = flatten_prepared(prepared)

            coverage['batches'] += 1; coverage['rows'] += int(input_ids_cpu.shape[0]); coverage['words'] += words
            coverage['events_selected_before_cross_target_filter'] += len(all_events)
            coverage['events_input_to_cross_target_filter'] += int(pstats.get('events_input_to_cross_target_filter', 0))
            coverage['events_used_after_cross_target_filter'] += int(pstats.get('events_used_after_cross_target_filter', 0))
            coverage['selected_categories_before_filter'].update([e['category'] for e in all_events])
            coverage['used_categories_after_filter'].update([e['category'] for e in prepared_flat])
            coverage['seen_categories_before_probability'].update(seen)
            coverage['reject_reasons'].update(rej)
            coverage['match_level_counts'].update(pstats.get('match_level_counts') or {})
            coverage['group_size_counts'].update(pstats.get('group_size_counts') or {})
            coverage['different_row_donor_pairs'] += int(pstats.get('different_row_donor_pairs', 0) or 0)
            coverage['same_row_donor_pairs'] += int(pstats.get('same_row_donor_pairs', 0) or 0)

            batch_record: dict[str, Any] = {
                'step': step, 'rows': int(input_ids_cpu.shape[0]), 'words': words,
                'events_selected_before_filter': len(all_events),
                'events_used_after_filter': len(prepared_flat),
                'used_categories': dict(Counter([e['category'] for e in prepared_flat])),
                'match_levels': dict(pstats.get('match_level_counts') or {}),
                'used_fraction_of_selected': pstats.get('used_fraction_of_selected'),
                'different_row_fraction': pstats.get('different_row_fraction'),
            }

            # Use identical ordinary WWM masks for margin/gradient batches.
            if step <= max(args.margin_batches, args.grad_batches):
                masked_dev, labels_dev = base.apply_masking_curriculum(input_ids_cpu.to(device, non_blocking=True), attention_cpu.to(device, non_blocking=True), word_group_cpu.to(device, non_blocking=True), tokenizer, legacy_state, legacy_gen)
                masked_cpu = masked_dev.cpu(); labels_cpu = labels_dev.cpu()
                del masked_dev, labels_dev
                n_pred_total = int((labels_cpu != -100).sum().item())
                coverage['masked_tokens_in_margin_batches'] += n_pred_total
                batch_record['masked_tokens'] = n_pred_total
                batch_tensors = {'masked_cpu': masked_cpu, 'labels_cpu': labels_cpu, 'attention_cpu': attention_cpu, 'input_ids_cpu': input_ids_cpu}
                update_visibility_stats(vis_stats, prepared_flat, input_ids_cpu, masked_cpu, labels_cpu)

                if step <= args.margin_batches:
                    with torch.no_grad():
                        for mb_start in range(0, int(masked_cpu.shape[0]), args.micro_batch_size):
                            mb_end = min(mb_start + args.micro_batch_size, int(masked_cpu.shape[0]))
                            mb_idx = mb_start // args.micro_batch_size
                            evs = prepared.get(int(mb_idx), [])
                            if not evs:
                                continue
                            input_ids = masked_cpu[mb_start:mb_end].to(device, non_blocking=True)
                            attention_mask = attention_cpu[mb_start:mb_end].to(device, non_blocking=True)
                            labels_tmp = labels_cpu[mb_start:mb_end].to(device, non_blocking=True)
                            reset_all(args.train_rng_seed + 800000 + step * 1000 + mb_start)
                            outm = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_tmp, output_hidden_states=True)
                            hidden = outm.hidden_states[-1]
                            score_preupdate(hidden, head, evs, mode='semantic', stats=margin_stats['semantic'])
                            score_preupdate(hidden, head, evs, mode='anchor_permuted', stats=margin_stats['anchor_permuted'])
                            del outm, hidden, input_ids, attention_mask, labels_tmp

                if step <= args.grad_batches:
                    mlm = accumulate_mlm_grad(model, head, batch_tensors, tokenizer, legacy_state, legacy_gen, device, step=step, args=args)
                    mlm_model_l2 = float(mlm['grad_norms']['model_all']['l2'])
                    mlm_encoder_l2 = float(mlm['grad_norms']['encoder']['l2'])
                    grec.write(json.dumps({'step': step, 'key': 'mlm', **mlm}, ensure_ascii=False) + '\n'); grec.flush()
                    grad_records.append({'step': step, 'key': 'mlm', 'events_used': 0, 'terms': 0, 'grad_norms': mlm['grad_norms'], 'head_grad_norm': {'l2': 0.0}})
                    denom = len(prepared_flat)
                    categories_in_batch = sorted(Counter([str(e.get('category')) for e in prepared_flat]))
                    for mode in ['semantic', 'anchor_permuted']:
                        for cat in [None] + categories_in_batch:
                            key = f'aux::{mode}::' + (cat if cat is not None else 'ALL')
                            aux = accumulate_aux_grad(model, head, batch_tensors, prepared, device, mode=mode, category=cat, denom_total_events=denom, step=step, args=args)
                            if aux.get('grad_norms') is not None:
                                aux['ratio_to_mlm_model_l2'] = ratio(aux['grad_norms']['model_all']['l2'], mlm_model_l2)
                                aux['ratio_to_mlm_encoder_l2'] = ratio(aux['grad_norms']['encoder']['l2'], mlm_encoder_l2)
                            rec = {'step': step, 'key': key, **aux}
                            grec.write(json.dumps(rec, ensure_ascii=False) + '\n'); grec.flush()
                            grad_records.append(rec)
                del masked_cpu, labels_cpu

            brec.write(json.dumps(batch_record, ensure_ascii=False) + '\n'); brec.flush()
            if step == 1 or step % 25 == 0 or step == args.coverage_batches:
                print(json.dumps({'event': 'coverage_progress', 'step': step, 'rows': coverage['rows'], 'selected': coverage['events_selected_before_cross_target_filter'], 'used': coverage['events_used_after_cross_target_filter'], 'used_categories': dict(coverage['used_categories_after_filter']), 'elapsed_sec': round(time.time() - t0, 1)}), flush=True)
            del input_ids_cpu, attention_cpu, word_group_cpu

    # Finalize coverage and summaries.
    used = int(coverage['events_used_after_cross_target_filter'])
    selected = int(coverage['events_selected_before_cross_target_filter'])
    coverage_summary = {
        'batches_scanned': int(coverage['batches']), 'rows_scanned': int(coverage['rows']), 'words_scanned': int(coverage['words']),
        'events_selected_before_cross_target_filter': selected,
        'events_used_after_cross_target_filter': used,
        'used_fraction_of_selected': used / selected if selected else None,
        'used_events_per_256row_batch': used / max(1, int(coverage['batches'])),
        'selected_categories_before_filter': dict(coverage['selected_categories_before_filter']),
        'used_categories_after_filter': dict(coverage['used_categories_after_filter']),
        'used_category_fractions': {k: v / used for k, v in coverage['used_categories_after_filter'].items()} if used else {},
        'seen_categories_before_probability': dict(coverage['seen_categories_before_probability']),
        'reject_reasons_top': dict(coverage['reject_reasons'].most_common(40)),
        'match_level_counts': dict(coverage['match_level_counts']),
        'match_level_fractions': {k: v / used for k, v in coverage['match_level_counts'].items()} if used else {},
        'group_size_counts': dict(coverage['group_size_counts']),
        'different_row_donor_pairs': int(coverage['different_row_donor_pairs']),
        'same_row_donor_pairs': int(coverage['same_row_donor_pairs']),
        'different_row_fraction': int(coverage['different_row_donor_pairs']) / used if used else None,
    }

    visibility_summary = {}
    for role_state, counter in vis_stats.items():
        den = vis_stats['events']
        visibility_summary[role_state] = {}
        for cat, val in counter.items():
            denom = int(den.get(cat, 0)) if role_state != 'events' else 1
            visibility_summary[role_state][cat] = {'count': int(val), 'fraction_of_events': (int(val) / denom if denom else None)} if role_state != 'events' else int(val)

    margins_final = {mode: finalize_margin_stats(stats) for mode, stats in margin_stats.items()}
    grad_summary = summarize_grad_records(grad_records)
    calibration = {
        'status': 'SPARSE_RELATION_AUX_CALIBRATION',
        'created_utc': now_utc(), 'runtime_sec': round(time.time() - t0, 2), 'device': str(device),
        'manifest': manifest,
        'coverage': coverage_summary,
        'visibility': visibility_summary,
        'pre_update_margins': margins_final,
        'gradient_summary': grad_summary,
        'artifacts': {
            'manifest': str(outdir / 'calibration_manifest.json'),
            'batch_records': str(batch_records_path),
            'gradient_records': str(grad_records_path),
            'summary': str(outdir / 'calibration_summary.json'),
            'note': str(note_path),
        },
    }
    # Add a direct mode comparison for the load-bearing aggregate quantities.
    comparison: dict[str, Any] = {}
    for cat in sorted(set(margins_final.get('semantic', {}).keys()) | set(margins_final.get('anchor_permuted', {}).keys())):
        sm = margins_final.get('semantic', {}).get(cat, {})
        am = margins_final.get('anchor_permuted', {}).get(cat, {})
        comparison[cat] = {
            'head_success_semantic': sm.get('head_success_rate_terms'),
            'head_success_anchor_permuted': am.get('head_success_rate_terms'),
            'head_success_delta_sem_minus_anchor': (sm.get('head_success_rate_terms') - am.get('head_success_rate_terms')) if sm.get('head_success_rate_terms') is not None and am.get('head_success_rate_terms') is not None else None,
            'raw_cosine_success_semantic': sm.get('raw_cosine_success_rate_events'),
            'raw_cosine_success_anchor_permuted': am.get('raw_cosine_success_rate_events'),
            'raw_cosine_success_delta_sem_minus_anchor': (sm.get('raw_cosine_success_rate_events') - am.get('raw_cosine_success_rate_events')) if sm.get('raw_cosine_success_rate_events') is not None and am.get('raw_cosine_success_rate_events') is not None else None,
            'raw_cosine_margin_mean_semantic': ((sm.get('raw_cosine_margin') or {}).get('mean')),
            'raw_cosine_margin_mean_anchor_permuted': ((am.get('raw_cosine_margin') or {}).get('mean')),
        }
    calibration['mode_margin_comparison'] = comparison

    # Scientific interpretation from thresholds, kept factual and directly tied to numbers.
    problems = []
    if coverage_summary['used_fraction_of_selected'] is not None and coverage_summary['used_fraction_of_selected'] < 0.30:
        problems.append(f"low cross-target retention {coverage_summary['used_fraction_of_selected']:.3f}")
    for cat in ['physical_change', 'comparative']:
        c = int(coverage_summary['used_categories_after_filter'].get(cat, 0))
        if c < max(25, 0.005 * used):
            problems.append(f"sparse {cat} coverage {c} used events")
    sem_all = comparison.get('ALL', {})
    if sem_all.get('raw_cosine_success_semantic') is not None and sem_all.get('raw_cosine_success_semantic') > 0.65:
        problems.append(f"semantic raw hidden cosine already high {sem_all['raw_cosine_success_semantic']:.3f}")
    if sem_all.get('raw_cosine_success_anchor_permuted') is not None and sem_all.get('raw_cosine_success_anchor_permuted') > 0.65:
        problems.append(f"anchor_permuted raw hidden cosine already high {sem_all['raw_cosine_success_anchor_permuted']:.3f}")
    sem_grad = grad_summary.get('aux::semantic::ALL', {})
    anc_grad = grad_summary.get('aux::anchor_permuted::ALL', {})
    if sem_grad and anc_grad:
        sm = (sem_grad.get('ratios_to_mlm_model') or {}).get('mean')
        am = (anc_grad.get('ratios_to_mlm_model') or {}).get('mean')
        if sm is not None and sm < 0.005:
            problems.append(f"semantic weighted aux/model gradient ratio tiny {sm:.4f}")
        if sm is not None and am is not None and (max(sm, am) / max(1e-12, min(sm, am)) > 2.0):
            problems.append(f"semantic vs anchor_permuted weighted aux/model gradient ratio asymmetric {sm:.4f} vs {am:.4f}")
    calibration['calibration_readout'] = {
        'problems_for_full_two_branch_launch': problems,
        'interpretation': 'launch-ready only if coverage is broad enough, pre-update success is not trivially solved by raw hidden context in both arms, and semantic/anchor_permuted weighted backbone gradient pressure is comparable and non-negligible',
    }

    (outdir / 'calibration_summary.json').write_text(json.dumps(calibration, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    note = []
    note.append('# research — sparse relation auxiliary calibration\n')
    note.append(f"Created: {calibration['created_utc']}  Runtime: {calibration['runtime_sec']} s  Device: {device}  Dropout mode: {args.dropout_mode}\n")
    note.append('## Coverage on exact research/121 70M→80M segment prefix\n')
    note.append(f"Batches/rows/words scanned: {coverage_summary['batches_scanned']} / {coverage_summary['rows_scanned']} / {coverage_summary['words_scanned']}\n")
    note.append(f"Selected before cross-target filtering: {selected}; used after microbatch-local cross-target filtering: {used}; used fraction: {coverage_summary['used_fraction_of_selected']}\n")
    note.append(f"Used events per 256-row batch: {coverage_summary['used_events_per_256row_batch']}\n")
    note.append(f"Used categories: {coverage_summary['used_categories_after_filter']}\n")
    note.append(f"Match levels: {coverage_summary['match_level_counts']}\n")
    note.append('## Pre-update margin comparison\n')
    for cat in ['ALL', 'spatial', 'causal_connector', 'temporal', 'negation', 'physical_change', 'comparative']:
        if cat in comparison:
            note.append(f"- {cat}: {comparison[cat]}\n")
    note.append('## Gradient summary\n')
    for key in sorted(grad_summary):
        if key == 'mlm' or key.endswith('::ALL') or key.endswith('::physical_change') or key.endswith('::comparative'):
            note.append(f"- {key}: {grad_summary[key]}\n")
    note.append('## Calibration readout\n')
    note.append(f"Problems for full launch: {problems}\n")
    note.append('\nFiles: `' + str(outdir / 'calibration_summary.json') + '`, `' + str(batch_records_path) + '`, `' + str(grad_records_path) + '`\n')
    note_path.write_text('\n'.join(note) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'SPARSE_RELATION_AUX_CALIBRATION', 'summary': str(outdir / 'calibration_summary.json'), 'note': str(note_path), 'problems': problems, 'used_events': used, 'used_fraction': coverage_summary['used_fraction_of_selected'], 'runtime_sec': calibration['runtime_sec']}), flush=True)


if __name__ == '__main__':
    main()
