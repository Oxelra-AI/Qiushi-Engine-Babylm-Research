#!/usr/bin/env python3
"""research symmetric cue-only masked-likelihood probe for relation events.

This script tests the corrected symmetric-cue hypothesis.  It does not launch
training and does not alter the base WWM pass.  It builds a separate auxiliary
view in which information flow is controlled identically across arms:

  * the dependent/consequence word group is masked and scored;
  * all ordinary context tokens are removed from attention;
  * exactly one cue subtoken is visible;
  * for semantic, matched-anchor, and shuffled-pivot arms the cue is placed at
    the same target-relative pivot position except for an optional native-anchor
    view recorded separately.

Thus the target state cannot absorb the true pivot through unrestricted
self-attention in the matched-anchor/shuffled arms, and the base input stream is
not manipulated.  The probe measures whether, on legal unseen corpus events, the
true pivot-consequence pairing gives higher masked-token likelihood than matched
anchors and shuffled same-family pivots, especially for physical_change and
comparative events that were missing in the research/123 contrast.
"""
from __future__ import annotations

import argparse
import heapq
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
import pvdm_masking_lib as pvdm  # noqa: E402
import sparse_relation_aux_trainer as relaux  # noqa: E402

DEFAULT_OUT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/symmetric_cue_view_likelihood_probe'
DEFAULT_NOTE = USER_ROOT / 'research/notes/representation_and_objectives/symmetric_cue_view_likelihood_probe.md'
DEFAULT_HELDOUT_START_TAIL_ROW = 64255
DEFAULT_HELDOUT_EXPECTED_START_WORDS = 10011326
DEFAULT_HELDOUT_MAX_ROWS = 64000


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def reset_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def numeric_add(d: dict[str, Any], prefix: str, value: float) -> None:
    if not math.isfinite(float(value)):
        return
    d[f'{prefix}_n'] = int(d.get(f'{prefix}_n', 0)) + 1
    d[f'{prefix}_sum'] = float(d.get(f'{prefix}_sum', 0.0)) + float(value)
    d[f'{prefix}_sumsq'] = float(d.get(f'{prefix}_sumsq', 0.0)) + float(value) * float(value)
    d[f'{prefix}_min'] = min(float(value), float(d.get(f'{prefix}_min', float('inf'))))
    d[f'{prefix}_max'] = max(float(value), float(d.get(f'{prefix}_max', float('-inf'))))


def finalize_numeric(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    prefixes = sorted({k[:-2] for k in d if k.endswith('_n')})
    for pfx in prefixes:
        n = int(d.get(f'{pfx}_n', 0))
        s = float(d.get(f'{pfx}_sum', 0.0))
        ss = float(d.get(f'{pfx}_sumsq', 0.0))
        mean = s / n if n else None
        var = max(0.0, ss / n - (mean or 0.0) ** 2) if n else None
        out[pfx] = {
            'n': n,
            'mean': mean,
            'std': math.sqrt(var) if var is not None else None,
            'min': None if n == 0 else float(d.get(f'{pfx}_min')),
            'max': None if n == 0 else float(d.get(f'{pfx}_max')),
        }
    return out


def raw_event(label_record: dict[str, Any], event_rank: int) -> dict[str, Any]:
    evs = label_record.get('events', [])
    if 0 <= int(event_rank) < len(evs) and isinstance(evs[int(event_rank)], dict):
        return evs[int(event_rank)]
    return {}


def first_non_special_token(input_ids_1d: torch.Tensor, positions: list[int], special_ids: set[int]) -> tuple[int, int] | None:
    for p in positions:
        tid = int(input_ids_1d[int(p)].item())
        if tid not in special_ids:
            return int(p), tid
    return None


class CategoryReservoir:
    """Keep the lowest-hash events per category without storing every candidate."""

    def __init__(self, quota: int):
        self.quota = int(quota)
        self.heaps: dict[str, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
        self.counter = 0

    def add(self, category: str, score: float, event: dict[str, Any]) -> None:
        if self.quota <= 0:
            return
        self.counter += 1
        heap = self.heaps[str(category)]
        item = (-float(score), self.counter, event)
        if len(heap) < self.quota:
            heapq.heappush(heap, item)
        else:
            # Heap top is the largest positive score represented as most negative.
            if item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    def events(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cat in sorted(self.heaps):
            items = sorted(self.heaps[cat], reverse=True)
            for neg_score, _, ev in items:
                ev = dict(ev)
                ev['sample_hash'] = -float(neg_score)
                out.append(ev)
        return out

    def counts(self) -> dict[str, int]:
        return {cat: len(items) for cat, items in self.heaps.items()}


def select_candidate_events(
    examples: list[base.Example],
    label_records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    batch_size: int,
    seq_length: int,
    allowed_categories: set[str],
    quota_per_category: int,
    seed: int,
    max_control_distance_abs_diff: int,
    max_control_freq_bin_abs_diff: int,
    max_target_token_len: int,
    num_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = base.MaskedChunkDataset(examples, tokenizer, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=base.collate, num_workers=num_workers)
    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    mask_id = int(tokenizer.mask_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    special_ids.update([mask_id, pad_id])
    reservoir = CategoryReservoir(quota_per_category)
    stats: dict[str, Any] = {
        'rows_scanned': 0, 'events_raw': 0, 'events_tokenization_usable': 0, 'events_candidate_after_filters': 0,
        'candidate_categories': Counter(), 'seen_categories': Counter(), 'usable_categories': Counter(), 'reject_reasons': Counter(),
    }

    for step, batch in enumerate(loader, 1):
        lo = (step - 1) * batch_size
        hi = lo + int(batch['input_ids'].shape[0])
        input_ids = batch['input_ids'][:, :seq_length].contiguous()
        attention = batch['attention_mask'][:, :seq_length].contiguous()
        word_group = batch['word_group'][:, :seq_length].contiguous()
        stats['rows_scanned'] += int(input_ids.shape[0])
        for b in range(int(input_ids.shape[0])):
            lr = label_records[lo + b]
            events_raw = lr.get('events', []) or []
            stats['events_raw'] += len(events_raw)
            for er in events_raw:
                stats['seen_categories'][str(er.get('category', 'unknown'))] += 1
            group_pos = pvdm.group_positions_for_row(word_group[b], attention[b])
            usable, rej = pvdm.collect_active_events(lr, group_pos, same_length_required=False)
            stats['reject_reasons'].update(rej)
            stats['events_tokenization_usable'] += len(usable)
            for ev in usable:
                stats['usable_categories'][ev.category] += 1
                cat = str(ev.category)
                if cat not in allowed_categories:
                    stats['reject_reasons'][f'category_excluded::{cat}'] += 1
                    continue
                if len({int(ev.pivot_gid), int(ev.target_gid), int(ev.control_gid)}) < 3:
                    stats['reject_reasons'][f'role_overlap::{cat}'] += 1
                    continue
                ev_raw = raw_event(lr, int(ev.event_rank))
                meta = relaux._event_meta(ev_raw)
                cdist = as_int(meta.get('control_distance_abs_diff'), 999)
                cfreq = as_int(meta.get('control_freq_bin_abs_diff'), 999)
                if cdist > max_control_distance_abs_diff:
                    stats['reject_reasons'][f'control_distance_mismatch_gt{max_control_distance_abs_diff}::{cat}'] += 1
                    continue
                if cfreq > max_control_freq_bin_abs_diff:
                    stats['reject_reasons'][f'control_freq_mismatch_gt{max_control_freq_bin_abs_diff}::{cat}'] += 1
                    continue
                target_positions = list(group_pos[int(ev.target_gid)])
                pivot_positions = list(group_pos[int(ev.pivot_gid)])
                control_positions = list(group_pos[int(ev.control_gid)])
                if not target_positions or not pivot_positions or not control_positions:
                    stats['reject_reasons'][f'empty_positions::{cat}'] += 1
                    continue
                if len(target_positions) > max_target_token_len:
                    stats['reject_reasons'][f'target_token_len_gt{max_target_token_len}::{cat}'] += 1
                    continue
                pivot_first = first_non_special_token(input_ids[b], pivot_positions, special_ids)
                control_first = first_non_special_token(input_ids[b], control_positions, special_ids)
                if pivot_first is None or control_first is None:
                    stats['reject_reasons'][f'bad_cue_token::{cat}'] += 1
                    continue
                target_ids = [int(input_ids[b, p].item()) for p in target_positions]
                if any(t in special_ids for t in target_ids):
                    stats['reject_reasons'][f'bad_target_token::{cat}'] += 1
                    continue
                # The symmetric view will expose exactly one cue token.  This permits
                # physical/comparative events that research dropped due cue wordpiece length,
                # while keeping visible-token count identical across arms.
                h = pvdm.hash_uniform(seed, 'event_sample', int(lr.get('tail_row_idx', -1)), int(ev.event_rank), cat, meta.get('target_norm', ''))
                cand = {
                    'tail_row_idx': int(lr.get('tail_row_idx', -1)),
                    'batch_scan_step': int(step),
                    'batch_row': int(b),
                    'event_rank': int(ev.event_rank),
                    'source': str(lr.get('source', examples[lo + b].source)),
                    'category': cat,
                    'input_ids': [int(x) for x in input_ids[b].tolist()],
                    'attention_mask': [int(x) for x in attention[b].tolist()],
                    'pivot_positions': [int(x) for x in pivot_positions],
                    'control_positions': [int(x) for x in control_positions],
                    'target_positions': [int(x) for x in target_positions],
                    'pivot_first_position': int(pivot_first[0]),
                    'pivot_first_id': int(pivot_first[1]),
                    'control_first_position': int(control_first[0]),
                    'control_first_id': int(control_first[1]),
                    'target_ids': target_ids,
                    'pivot_len': int(len(pivot_positions)),
                    'control_len': int(len(control_positions)),
                    'target_len': int(len(target_positions)),
                    **meta,
                    'sample_text_prefix': examples[lo + b].text[:240],
                }
                reservoir.add(cat, h, cand)
                stats['events_candidate_after_filters'] += 1
                stats['candidate_categories'][cat] += 1
        if step == 1 or step % 50 == 0:
            print(json.dumps({'event': 'scan_progress', 'step': step, 'rows': stats['rows_scanned'], 'candidates': stats['events_candidate_after_filters'], 'reservoir': reservoir.counts()}), flush=True)
    selected = reservoir.events()
    stats['reservoir_counts'] = Counter([e['category'] for e in selected])
    return selected, stats


def assign_shuffled_pivots(events: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    # Tight grouping first; later relax if needed.
    for i, ev in enumerate(events):
        groups[(ev.get('category'), ev.get('target_class'), ev.get('distance_bin'))].append(i)
    assignment: dict[int, int] = {}
    level_counts: Counter = Counter()
    remaining = set(range(len(events)))
    for level in range(3):
        groups = defaultdict(list)
        for i in sorted(remaining):
            ev = events[i]
            if level == 0:
                key = (ev.get('category'), ev.get('target_class'), ev.get('distance_bin'))
            elif level == 1:
                key = (ev.get('category'), ev.get('target_class'))
            else:
                key = (ev.get('category'),)
            groups[key].append(i)
        newly: set[int] = set()
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            ordered = sorted(idxs, key=lambda i: pvdm.hash_uniform(seed, 'shuffle_order', level, events[i]['tail_row_idx'], events[i]['event_rank'], events[i]['category']))
            n = len(ordered)
            for pos, rec_idx in enumerate(ordered):
                # Choose a donor from a different row when possible.
                donor = None
                for shift in range(1, n):
                    cand = ordered[(pos + shift) % n]
                    if events[cand]['tail_row_idx'] != events[rec_idx]['tail_row_idx']:
                        donor = cand
                        break
                if donor is None:
                    continue
                assignment[rec_idx] = donor
                newly.add(rec_idx)
            level_counts[str(level)] += len(newly)
        remaining -= newly
    out = []
    for i, ev in enumerate(events):
        if i not in assignment:
            continue
        donor = events[assignment[i]]
        e2 = dict(ev)
        e2['shuffle_pivot_first_id'] = int(donor['pivot_first_id'])
        e2['shuffle_pivot_norm'] = str(donor.get('pivot_norm'))
        e2['shuffle_tail_row_idx'] = int(donor['tail_row_idx'])
        e2['shuffle_event_rank'] = int(donor['event_rank'])
        e2['shuffle_match_category'] = str(donor.get('category'))
        e2['shuffle_match_target_class'] = str(donor.get('target_class'))
        e2['shuffle_match_distance_bin'] = str(donor.get('distance_bin'))
        out.append(e2)
    stats = {
        'input_events': len(events),
        'events_with_shuffled_pivot': len(out),
        'dropped_no_shuffle': len(events) - len(out),
        'shuffle_level_counts': dict(level_counts),
        'shuffle_categories': dict(Counter([e['category'] for e in out])),
    }
    return out, stats


def make_view(ev: dict[str, Any], kind: str, *, seq_length: int, mask_id: int, pad_id: int) -> tuple[list[int], list[int], list[int]]:
    ids = [pad_id] * seq_length
    attn = [0] * seq_length
    labels = [-100] * seq_length
    target_positions = [int(p) for p in ev['target_positions'] if 0 <= int(p) < seq_length]
    for p, tid in zip(target_positions, ev['target_ids']):
        ids[p] = mask_id
        attn[p] = 1
        labels[p] = int(tid)
    if kind == 'semantic':
        cue_pos = int(ev['pivot_first_position']); cue_id = int(ev['pivot_first_id'])
    elif kind == 'anchor_samepos':
        cue_pos = int(ev['pivot_first_position']); cue_id = int(ev['control_first_id'])
    elif kind == 'shuffle_samepos':
        cue_pos = int(ev['pivot_first_position']); cue_id = int(ev['shuffle_pivot_first_id'])
    elif kind == 'anchor_native':
        cue_pos = int(ev['control_first_position']); cue_id = int(ev['control_first_id'])
    elif kind == 'no_cue':
        cue_pos = -1; cue_id = -1
    elif kind == 'full_context_masked':
        ids = [int(x) for x in ev['input_ids'][:seq_length]]
        attn = [int(x) for x in ev['attention_mask'][:seq_length]]
        labels = [-100] * seq_length
        for p, tid in zip(target_positions, ev['target_ids']):
            ids[p] = mask_id
            labels[p] = int(tid)
        return ids, attn, labels
    else:
        raise ValueError(kind)
    if cue_pos >= 0:
        if cue_pos in target_positions:
            raise ValueError('cue position overlaps target')
        ids[cue_pos] = cue_id
        attn[cue_pos] = 1
    return ids, attn, labels


def score_events(events: list[dict[str, Any]], model: torch.nn.Module, tokenizer: Any, *, device: torch.device, seq_length: int, view_batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kinds = ['semantic', 'anchor_samepos', 'shuffle_samepos', 'anchor_native', 'no_cue', 'full_context_masked']
    mask_id = int(tokenizer.mask_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    scored: list[dict[str, Any]] = [dict(e) for e in events]
    # Materialize views in streaming batches; record average log-prob per target subtoken.
    view_items: list[tuple[int, str, list[int], list[int], list[int]]] = []
    for i, ev in enumerate(scored):
        for kind in kinds:
            ids, attn, labels = make_view(ev, kind, seq_length=seq_length, mask_id=mask_id, pad_id=pad_id)
            view_items.append((i, kind, ids, attn, labels))
    model.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for start in range(0, len(view_items), view_batch_size):
            chunk = view_items[start:start + view_batch_size]
            ids_t = torch.tensor([x[2] for x in chunk], dtype=torch.long, device=device)
            attn_t = torch.tensor([x[3] for x in chunk], dtype=torch.long, device=device)
            labels_t = torch.tensor([x[4] for x in chunk], dtype=torch.long, device=device)
            out = model(input_ids=ids_t, attention_mask=attn_t)
            logp = F.log_softmax(out.logits.float(), dim=-1)
            for row, (event_i, kind, _ids, _attn, _labels) in enumerate(chunk):
                pos = (labels_t[row] != -100).nonzero(as_tuple=False).view(-1)
                if pos.numel() == 0:
                    val = float('nan')
                    ntok = 0
                else:
                    target = labels_t[row].index_select(0, pos)
                    vals = logp[row].index_select(0, pos).gather(1, target.view(-1, 1)).view(-1)
                    val = float(vals.mean().detach().cpu().item())
                    ntok = int(pos.numel())
                scored[event_i][f'logp_{kind}'] = val
                scored[event_i][f'ntok_{kind}'] = ntok
            del ids_t, attn_t, labels_t, out, logp
            if start == 0 or (start // view_batch_size) % 20 == 0:
                print(json.dumps({'event': 'score_progress', 'views_done': min(start + len(chunk), len(view_items)), 'views_total': len(view_items)}), flush=True)
    stats = {
        'events_scored': len(scored),
        'views_scored': len(view_items),
        'view_kinds': kinds,
        'cuda_peak_memory_gb': (torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == 'cuda' else None),
    }
    return scored, stats


def summarize_scores(scored: list[dict[str, Any]]) -> dict[str, Any]:
    bycat: dict[str, dict[str, Any]] = defaultdict(dict)
    counters: dict[str, Counter] = defaultdict(Counter)
    score_names = ['semantic', 'anchor_samepos', 'shuffle_samepos', 'anchor_native', 'no_cue', 'full_context_masked']
    delta_pairs = [
        ('sem_minus_anchor_samepos', 'semantic', 'anchor_samepos'),
        ('sem_minus_shuffle_samepos', 'semantic', 'shuffle_samepos'),
        ('sem_minus_anchor_native', 'semantic', 'anchor_native'),
        ('sem_minus_no_cue', 'semantic', 'no_cue'),
        ('full_context_minus_sem', 'full_context_masked', 'semantic'),
    ]
    for ev in scored:
        cats = ['ALL', str(ev.get('category'))]
        for cat in cats:
            d = bycat[cat]
            d['n_events'] = int(d.get('n_events', 0)) + 1
            counters[cat]['source::' + str(ev.get('source'))] += 1
            counters[cat]['target_class::' + str(ev.get('target_class'))] += 1
            counters[cat]['distance_bin::' + str(ev.get('distance_bin'))] += 1
            counters[cat]['target_len::' + str(ev.get('target_len'))] += 1
            counters[cat]['pivot_len::' + str(ev.get('pivot_len'))] += 1
            counters[cat]['control_len::' + str(ev.get('control_len'))] += 1
            for name in score_names:
                numeric_add(d, f'logp_{name}', float(ev.get(f'logp_{name}', float('nan'))))
            for dname, a, b in delta_pairs:
                av = float(ev.get(f'logp_{a}', float('nan')))
                bv = float(ev.get(f'logp_{b}', float('nan')))
                delta = av - bv
                numeric_add(d, dname, delta)
                if math.isfinite(delta):
                    d[f'{dname}_success'] = int(d.get(f'{dname}_success', 0)) + int(delta > 0.0)
                    d[f'{dname}_success_denom'] = int(d.get(f'{dname}_success_denom', 0)) + 1
    out: dict[str, Any] = {}
    for cat, d in bycat.items():
        item = finalize_numeric(d)
        item['n_events'] = int(d.get('n_events', 0))
        for dname, _, _ in delta_pairs:
            den = int(d.get(f'{dname}_success_denom', 0))
            item[f'{dname}_success_rate'] = (int(d.get(f'{dname}_success', 0)) / den) if den else None
        item['top_sources'] = dict(Counter({k.split('source::', 1)[1]: v for k, v in counters[cat].items() if k.startswith('source::')}).most_common(10))
        item['target_class_counts'] = dict(Counter({k.split('target_class::', 1)[1]: v for k, v in counters[cat].items() if k.startswith('target_class::')}))
        item['distance_bin_counts'] = dict(Counter({k.split('distance_bin::', 1)[1]: v for k, v in counters[cat].items() if k.startswith('distance_bin::')}))
        item['target_len_counts'] = dict(Counter({k.split('target_len::', 1)[1]: v for k, v in counters[cat].items() if k.startswith('target_len::')}))
        item['cue_len_pair_counts'] = {
            'pivot_len': dict(Counter({k.split('pivot_len::', 1)[1]: v for k, v in counters[cat].items() if k.startswith('pivot_len::')})),
            'control_len': dict(Counter({k.split('control_len::', 1)[1]: v for k, v in counters[cat].items() if k.startswith('control_len::')})),
        }
        out[cat] = item
    # Direct readiness facts for the families missing from research.
    readiness = {}
    for cat in ['physical_change', 'comparative']:
        item = out.get(cat, {})
        n = int(item.get('n_events', 0) or 0)
        ma = (item.get('sem_minus_anchor_samepos') or {}).get('mean')
        ms = (item.get('sem_minus_shuffle_samepos') or {}).get('mean')
        sa = item.get('sem_minus_anchor_samepos_success_rate')
        ss = item.get('sem_minus_shuffle_samepos_success_rate')
        readiness[cat] = {
            'n_events': n,
            'mean_sem_minus_anchor_samepos': ma,
            'mean_sem_minus_shuffle_samepos': ms,
            'success_sem_gt_anchor_samepos': sa,
            'success_sem_gt_shuffle_samepos': ss,
            'supports_separation': bool(n >= 50 and ma is not None and ms is not None and ma > 0.02 and ms > 0.02 and sa is not None and ss is not None and sa > 0.55 and ss > 0.55),
        }
    out['readiness_by_missing_family'] = readiness
    return out


def write_note(path: Path, summary: dict[str, Any], scored_path: Path, samples_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append('# research — symmetric cue-only relation likelihood probe\n')
    lines.append(f"Created: {summary['created_utc']}  Runtime: {summary['runtime_sec']} s  Device: {summary['device']}\n")
    lines.append('## Symmetric auxiliary view\n')
    lines.append('The base WWM pass is untouched.  For this probe only, each view masks the dependent target, removes ordinary context from attention, and exposes exactly one cue subtoken. Semantic, matched-anchor, and shuffled-pivot views use the same pivot position for the cue; the native-anchor view is recorded separately.\n')
    lines.append('## Segment and coverage\n')
    seg = summary['segment']
    lines.append(f"Segment rows/words: start_tail_row={seg['start_tail_row']}, selected_rows={seg['selected_rows']}, selected_words={seg['selected_words']}, first={seg['first_tail_row']}, last={seg['last_tail_row']}\n")
    lines.append(f"Scan candidates by category: {summary['scan_stats']['candidate_categories']}\n")
    lines.append(f"Reservoir selected by category: {summary['scan_stats']['reservoir_counts']}\n")
    lines.append(f"After shuffled-pivot assignment: {summary['shuffle_stats']}\n")
    lines.append('## Masked-token likelihood separation\n')
    for cat in ['ALL', 'physical_change', 'comparative', 'causal_connector', 'temporal', 'spatial', 'negation']:
        if cat in summary['score_summary']:
            item = summary['score_summary'][cat]
            lines.append(f"- {cat}: n={item.get('n_events')}; sem-anchor_samepos mean={(item.get('sem_minus_anchor_samepos') or {}).get('mean')} success={item.get('sem_minus_anchor_samepos_success_rate')}; sem-shuffle_samepos mean={(item.get('sem_minus_shuffle_samepos') or {}).get('mean')} success={item.get('sem_minus_shuffle_samepos_success_rate')}; sem-anchor_native mean={(item.get('sem_minus_anchor_native') or {}).get('mean')} success={item.get('sem_minus_anchor_native_success_rate')}\n")
    lines.append('## Missing-family readiness\n')
    lines.append(json.dumps(summary['score_summary'].get('readiness_by_missing_family', {}), indent=2, ensure_ascii=False) + '\n')
    lines.append('## Files\n')
    lines.append(f"- summary: `{summary['artifacts']['summary']}`\n")
    lines.append(f"- scored events: `{scored_path}`\n")
    lines.append(f"- samples: `{samples_path}`\n")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Symmetric cue-only masked-likelihood probe on legal relation events')
    ap.add_argument('--output_dir', default=str(DEFAULT_OUT))
    ap.add_argument('--note_path', default=str(DEFAULT_NOTE))
    ap.add_argument('--init_checkpoint', default=str(cont.DEFAULT_INIT))
    ap.add_argument('--checkpoint_label', default='a02_compact_chck70M')
    ap.add_argument('--tail_jsonl', default=str(cont.DEFAULT_TAIL))
    ap.add_argument('--labels_jsonl', default=str(cont.DEFAULT_LABELS))
    ap.add_argument('--tokenizer_path', default=str(cont.DEFAULT_TOKENIZER))
    ap.add_argument('--start_tail_row', type=int, default=DEFAULT_HELDOUT_START_TAIL_ROW)
    ap.add_argument('--expected_start_tail_words', type=int, default=DEFAULT_HELDOUT_EXPECTED_START_WORDS)
    ap.add_argument('--max_rows', type=int, default=DEFAULT_HELDOUT_MAX_ROWS)
    ap.add_argument('--max_word_exposure', type=int, default=9971289)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--view_batch_size', type=int, default=256)
    ap.add_argument('--seq_length', type=int, default=256)
    ap.add_argument('--max_seq_length', type=int, default=256)
    ap.add_argument('--quota_per_category', type=int, default=384)
    ap.add_argument('--max_control_distance_abs_diff', type=int, default=8)
    ap.add_argument('--max_control_freq_bin_abs_diff', type=int, default=2)
    ap.add_argument('--max_target_token_len', type=int, default=6)
    ap.add_argument('--categories', nargs='*', default=sorted(relaux.ALL_CATEGORIES))
    ap.add_argument('--seed', type=int, default=43024)
    ap.add_argument('--num_workers', type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = build_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    t0 = time.time()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    note_path = Path(args.note_path)
    tail_path = Path(args.tail_jsonl); labels_path = Path(args.labels_jsonl); tok_path = Path(args.tokenizer_path); init_ckpt = Path(args.init_checkpoint)
    hashes = {'tail_sha256': base.sha256_file(tail_path), 'labels_sha256': base.sha256_file(labels_path), 'tokenizer_sha256': base.sha256_file(tok_path / 'tokenizer.json')}
    for k, expected in cont.EXPECTED.items():
        if hashes[k] != expected:
            raise RuntimeError(f'{k} mismatch {hashes[k]} != {expected}')
    if not init_ckpt.exists():
        raise RuntimeError(f'init checkpoint not found: {init_ckpt}')
    reset_all(args.seed)
    tokenizer = base.make_portable_tokenizer(str(tok_path))
    examples, labels, segment = cont.load_segment(
        tail_path, labels_path,
        start_tail_row=args.start_tail_row,
        expected_start_tail_words=args.expected_start_tail_words,
        max_word_exposure=args.max_word_exposure,
        max_rows=args.max_rows,
    )
    print(json.dumps({'event': 'loaded_segment', 'checkpoint_label': args.checkpoint_label, 'examples': len(examples), 'segment': segment}), flush=True)
    allowed = set(args.categories) & relaux.ALL_CATEGORIES
    selected, scan_stats = select_candidate_events(
        examples, labels, tokenizer,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        allowed_categories=allowed,
        quota_per_category=args.quota_per_category,
        seed=args.seed,
        max_control_distance_abs_diff=args.max_control_distance_abs_diff,
        max_control_freq_bin_abs_diff=args.max_control_freq_bin_abs_diff,
        max_target_token_len=args.max_target_token_len,
        num_workers=args.num_workers,
    )
    selected, shuffle_stats = assign_shuffled_pivots(selected, args.seed)
    print(json.dumps({'event': 'selected_events', 'selected_after_shuffle': len(selected), 'scan_counts': {k: (dict(v) if isinstance(v, Counter) else v) for k, v in scan_stats.items() if k in ['candidate_categories', 'reservoir_counts']}, 'shuffle': shuffle_stats}), flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reset_all(args.seed + 17)
    model = DebertaV2ForMaskedLM.from_pretrained(str(init_ckpt))
    if model.config.vocab_size != len(tokenizer):
        raise RuntimeError(f'vocab mismatch model {model.config.vocab_size} tokenizer {len(tokenizer)}')
    model.to(device)
    scored, scoring_stats = score_events(selected, model, tokenizer, device=device, seq_length=args.seq_length, view_batch_size=args.view_batch_size)
    score_summary = summarize_scores(scored)
    # Convert Counters in scan_stats for JSON.
    scan_json = {}
    for k, v in scan_stats.items():
        scan_json[k] = dict(v) if isinstance(v, Counter) else v
    summary = {
        'status': 'SYMMETRIC_CUE_VIEW_LIKELIHOOD_PROBE',
        'created_utc': now_utc(),
        'runtime_sec': round(time.time() - t0, 2),
        'device': str(device),
        'purpose': 'test whether true pivot-consequence pairing improves masked-token likelihood under a symmetric cue-only auxiliary view before any training launch',
        'checkpoint_label': args.checkpoint_label,
        'init_checkpoint': str(init_ckpt),
        'hashes': hashes,
        'segment': segment,
        'view_definition': {
            'base_wwm_pass_modified': False,
            'target_group': 'masked and scored',
            'ordinary_context': 'removed from attention in cue-only views',
            'visible_cue_count': 'exactly one cue subtoken for semantic, anchor, and shuffled arms',
            'semantic': 'true pivot first subtoken placed at original pivot first-subtoken position',
            'anchor_samepos': 'matched nonrelation anchor first subtoken placed at the same pivot position',
            'shuffle_samepos': 'same-family shuffled true pivot first subtoken placed at the same pivot position',
            'anchor_native': 'matched anchor first subtoken at its original control-anchor position, recorded as a secondary check',
        },
        'args': vars(args),
        'scan_stats': scan_json,
        'shuffle_stats': shuffle_stats,
        'scoring_stats': scoring_stats,
        'score_summary': score_summary,
        'artifacts': {
            'summary': str(outdir / 'symmetric_cue_view_likelihood_summary.json'),
            'scored_events': str(outdir / 'scored_events.jsonl'),
            'sample_events': str(outdir / 'sample_events.jsonl'),
            'note': str(note_path),
        },
    }
    summary_path = outdir / 'symmetric_cue_view_likelihood_summary.json'
    scored_path = outdir / 'scored_events.jsonl'
    samples_path = outdir / 'sample_events.jsonl'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with scored_path.open('w', encoding='utf-8') as f:
        for ev in scored:
            # Full input_ids are omitted from the detailed event record to keep it readable.
            slim = {k: v for k, v in ev.items() if k not in {'input_ids', 'attention_mask'}}
            f.write(json.dumps(slim, ensure_ascii=False) + '\n')
    with samples_path.open('w', encoding='utf-8') as f:
        for ev in scored[:80]:
            slim = {k: v for k, v in ev.items() if k not in {'input_ids', 'attention_mask'}}
            f.write(json.dumps(slim, ensure_ascii=False) + '\n')
    write_note(note_path, summary, scored_path, samples_path)
    print(json.dumps({'status': summary['status'], 'summary': str(summary_path), 'note': str(note_path), 'events_scored': len(scored), 'readiness': score_summary.get('readiness_by_missing_family'), 'runtime_sec': summary['runtime_sec']}), flush=True)


if __name__ == '__main__':
    main()
