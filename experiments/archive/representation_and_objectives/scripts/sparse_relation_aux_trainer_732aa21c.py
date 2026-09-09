#!/usr/bin/env python3
"""research sparse WWM-preserving relation auxiliary continuation trainer.

Evidence context: research showed that ordinary staged WWM on the exact 70M->80M
segment did not reproduce the PVDM/control relation damage, while dense PVDM
modified target identities and anchor masking.  This script therefore keeps the
ordinary WWM inputs and MLM targets unchanged, and adds only a sparse auxiliary
loss over legal research hand-coded relation events.

The research draft used a hash-random placebo; that is not used here.  research
uses a structure-matched control:

  semantic:        true relation pivot scores its dependent target above a
                   structure-matched cross-event dependent target.
  anchor_permuted: the same recipient target, same cross-event dependent target,
                   same orientation, same event set, and same auxiliary loss mass
                   are used, but the true pivot is replaced by the same-row
                   research matched nonrelation anchor.  True pivot/control anchor
                   token length is equal and distance/support mismatch is bounded.

Thus both arms can exploit generic same-row association equally; only the semantic
arm has the true pivot-to-consequence correspondence.
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
FREQ_BIN_ORDER = ['0', '1-2', '3-5', '6-10', '11-25', '26-50', '51-100', '101-250', '251-500', '501-1000', '1001+']
DIST_BIN_ORDER = ['1', '2', '3-4', '5-8', '9-20', '21+']
FREQ_BIN_ID = {x: i for i, x in enumerate(FREQ_BIN_ORDER)}
DIST_BIN_ID = {x: i for i, x in enumerate(DIST_BIN_ORDER)}


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


def group_rep(hidden: torch.Tensor, batch_row: int, positions: list[int]) -> torch.Tensor:
    idx = torch.tensor(positions, dtype=torch.long, device=hidden.device)
    return hidden[batch_row].index_select(0, idx).mean(dim=0)


def _as_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _bin_id(name: Any, mapping: dict[str, int], default: int = -1) -> int:
    return mapping.get(str(name), default)


def _raw_event(label_record: dict[str, Any], event_rank: int) -> dict[str, Any]:
    evs = label_record.get('events', [])
    if 0 <= int(event_rank) < len(evs) and isinstance(evs[int(event_rank)], dict):
        return evs[int(event_rank)]
    return {}


def _event_meta(ev_raw: dict[str, Any]) -> dict[str, Any]:
    cm = ev_raw.get('control_match', {}) if isinstance(ev_raw.get('control_match', {}), dict) else {}
    pivot_freq = str(ev_raw.get('pivot_freq_bin', cm.get('pivot_freq_bin', 'NA')))
    target_freq = str(ev_raw.get('target_freq_bin', 'NA'))
    control_freq = str(ev_raw.get('control_freq_bin', cm.get('control_freq_bin', 'NA')))
    distance_bin = str(ev_raw.get('distance_bin', 'NA'))
    pivot_target_distance = _as_int(ev_raw.get('pivot_target_distance', cm.get('pivot_target_distance', -1)))
    control_target_distance = _as_int(ev_raw.get('control_target_distance', cm.get('control_distance', -1)))
    control_d_abs = abs(control_target_distance - pivot_target_distance) if pivot_target_distance >= 0 and control_target_distance >= 0 else -1
    return {
        'pivot_norm': str(ev_raw.get('pivot_norm', ev_raw.get('pivot', ''))),
        'target_norm': str(ev_raw.get('target_norm', ev_raw.get('target', ''))),
        'control_norm': str(ev_raw.get('control_norm', ev_raw.get('control', ''))),
        'pivot_anchor_class': str(ev_raw.get('pivot_anchor_class', cm.get('pivot_anchor_class', 'NA'))),
        'control_anchor_class': str(ev_raw.get('control_anchor_class', cm.get('control_class', 'NA'))),
        'target_class': str(ev_raw.get('target_class', 'NA')),
        'pivot_freq_bin': pivot_freq,
        'target_freq_bin': target_freq,
        'control_freq_bin': control_freq,
        'pivot_freq_bin_id': _bin_id(pivot_freq, FREQ_BIN_ID),
        'target_freq_bin_id': _bin_id(target_freq, FREQ_BIN_ID),
        'control_freq_bin_id': _bin_id(control_freq, FREQ_BIN_ID),
        'distance_bin': distance_bin,
        'distance_bin_id': _bin_id(distance_bin, DIST_BIN_ID),
        'pivot_target_distance': pivot_target_distance,
        'control_target_distance': control_target_distance,
        'control_distance_abs_diff': _as_int(cm.get('distance_abs_diff', control_d_abs)),
        'control_freq_bin_abs_diff': _as_int(cm.get('freq_bin_abs_diff', abs(_bin_id(control_freq, FREQ_BIN_ID) - _bin_id(pivot_freq, FREQ_BIN_ID)) if _bin_id(control_freq, FREQ_BIN_ID) >= 0 and _bin_id(pivot_freq, FREQ_BIN_ID) >= 0 else -1)),
        'control_match_score': _as_float(cm.get('score', 0.0)),
    }


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
    max_control_distance_abs_diff: int,
    max_control_freq_bin_abs_diff: int,
) -> tuple[list[dict[str, Any]], Counter, Counter]:
    """Select sparse events; selection is mode-independent and preserves WWM.

    Uses at most one event per row.  Requires pivot/control tokenizer length
    equality, bounded same-row control distance/support mismatch, and disjoint
    pivot/target/control roles.  The downstream cross-event negative is a matched
    dependent target rather than the same-row control anchor.
    """
    selected: list[dict[str, Any]] = []
    cat_seen: Counter = Counter()
    reject: Counter = Counter()
    for b, lr in enumerate(label_records):
        group_pos = pvdm.group_positions_for_row(word_group[b], attention_mask[b])
        usable, rej = pvdm.collect_active_events(lr, group_pos, same_length_required=True)
        reject.update(rej)
        candidates: list[tuple[pvdm.ActiveEvent, dict[str, Any]]] = []
        for ev in usable:
            ev_raw = _raw_event(lr, int(ev.event_rank))
            cat_seen[ev.category] += 1
            if ev.category not in allowed_categories:
                reject[f'category_excluded::{ev.category}'] += 1
                continue
            if len({ev.pivot_gid, ev.target_gid, ev.control_gid}) < 3:
                reject['role_overlap'] += 1
                continue
            meta = _event_meta(ev_raw)
            if int(meta.get('control_distance_abs_diff', 999)) > int(max_control_distance_abs_diff):
                reject[f'control_distance_mismatch_gt{max_control_distance_abs_diff}::{ev.category}'] += 1
                continue
            if int(meta.get('control_freq_bin_abs_diff', 999)) > int(max_control_freq_bin_abs_diff):
                reject[f'control_freq_mismatch_gt{max_control_freq_bin_abs_diff}::{ev.category}'] += 1
                continue
            u = pvdm.hash_uniform(seed, 'sparse_rel_select', int(ev.row_tail_idx), int(ev.event_rank), ev.category)
            if u >= event_prob:
                reject[f'not_selected::{ev.category}'] += 1
                continue
            candidates.append((ev, meta))
        if not candidates:
            continue
        candidates.sort(key=lambda x: pvdm.hash_uniform(seed, 'sparse_rel_rank', int(x[0].row_tail_idx), int(x[0].event_rank), x[0].category, x[0].target_gid))
        ev, meta = candidates[0]
        selected.append({
            'batch_row': b,
            'row_tail_idx': int(ev.row_tail_idx),
            'event_rank': int(ev.event_rank),
            'source': str(lr.get('source', '')),
            'category': ev.category,
            'pivot_gid': int(ev.pivot_gid),
            'target_gid': int(ev.target_gid),
            'control_gid': int(ev.control_gid),
            'pivot_len': int(ev.pivot_len),
            'target_len': int(ev.target_len),
            'control_len': int(ev.control_len),
            'pivot_positions': group_pos[int(ev.pivot_gid)],
            'target_positions': group_pos[int(ev.target_gid)],
            'control_positions': group_pos[int(ev.control_gid)],
            **meta,
        })
    if len(selected) > max_events_per_batch:
        selected.sort(key=lambda e: pvdm.hash_uniform(seed, 'sparse_rel_batch_cap', batch_step, e['row_tail_idx'], e['event_rank'], e['category']))
        selected = selected[:max_events_per_batch]
    selected.sort(key=lambda e: (e['batch_row'], e['event_rank']))
    return selected, cat_seen, reject


def _struct_key(ev: dict[str, Any], level: int) -> tuple[Any, ...]:
    if level == 0:
        return (ev['category'], ev.get('target_class'), ev.get('distance_bin'), ev.get('target_freq_bin'), ev.get('target_len'), ev.get('pivot_anchor_class'), ev.get('pivot_freq_bin'), ev.get('pivot_len'))
    if level == 1:
        return (ev['category'], ev.get('target_class'), ev.get('distance_bin'), ev.get('target_freq_bin'), ev.get('target_len'))
    if level == 2:
        return (ev['category'], ev.get('target_class'), ev.get('distance_bin'), ev.get('target_freq_bin'))
    if level == 3:
        return (ev['category'], ev.get('target_class'), ev.get('distance_bin'))
    return (ev['category'], ev.get('target_class'))


def _sort_key(seed: int, step: int, microbatch_index: int, ev: dict[str, Any]) -> tuple[Any, ...]:
    return (
        ev.get('category'), ev.get('target_class'), ev.get('distance_bin_id', -1),
        ev.get('target_freq_bin_id', -1), ev.get('pivot_freq_bin_id', -1),
        ev.get('target_len'), ev.get('pivot_len'),
        pvdm.hash_uniform(seed, 'cross_target_sort', step, microbatch_index, ev['row_tail_idx'], ev['event_rank'], ev['category']),
    )


def _pair_cost(rec: dict[str, Any], donor: dict[str, Any]) -> float:
    return (
        3.0 * (str(rec.get('category')) != str(donor.get('category'))) +
        2.0 * (str(rec.get('target_class')) != str(donor.get('target_class'))) +
        1.0 * abs(int(rec.get('target_freq_bin_id', -1)) - int(donor.get('target_freq_bin_id', -1))) +
        0.75 * abs(int(rec.get('distance_bin_id', -1)) - int(donor.get('distance_bin_id', -1))) +
        0.50 * abs(int(rec.get('target_len', 1)) - int(donor.get('target_len', 1))) +
        0.25 * abs(int(rec.get('pivot_freq_bin_id', -1)) - int(donor.get('pivot_freq_bin_id', -1))) +
        0.10 * abs(int(rec.get('pivot_target_distance', -1)) - int(donor.get('pivot_target_distance', -1)))
    )


def assign_cross_targets(events: list[dict[str, Any]], *, seed: int, step: int, microbatch_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach a structure-matched cross-event dependent target to each event.

    The same prepared records are consumed by both semantic and anchor_permuted
    modes.  Singleton groups are dropped symmetrically so auxiliary loss mass is
    identical across modes.  Because only one event is selected per row, accepted
    donor targets are from another sequence whenever a group has at least two rows.
    """
    if not events:
        return [], {'input_events': 0, 'used_events': 0}
    remaining: set[int] = set(range(len(events)))
    assignment: dict[int, tuple[int, int]] = {}
    level_counts: Counter = Counter(); group_size_counts: Counter = Counter(); shift_same_row: Counter = Counter()
    for level in range(5):
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for idx in sorted(remaining):
            groups[_struct_key(events[idx], level)].append(idx)
        assigned: set[int] = set()
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            ordered = sorted(idxs, key=lambda i: _sort_key(seed, step, microbatch_index, events[i]))
            n = len(ordered)
            best: tuple[float, float, int] | None = None
            for shift in range(1, n):
                same_row = 0; cost = 0.0
                for pos, rec_idx in enumerate(ordered):
                    donor_idx = ordered[(pos + shift) % n]
                    same_row += int(events[rec_idx]['row_tail_idx'] == events[donor_idx]['row_tail_idx'])
                    cost += _pair_cost(events[rec_idx], events[donor_idx])
                cand = (float(same_row), float(cost), int(shift))
                if best is None or cand < best:
                    best = cand
            assert best is not None
            shift = int(best[2])
            for pos, rec_idx in enumerate(ordered):
                donor_idx = ordered[(pos + shift) % n]
                assignment[rec_idx] = (donor_idx, level)
                assigned.add(rec_idx)
            level_counts[str(level)] += n
            group_size_counts[str(n)] += 1
            shift_same_row[f'level{level}_same_row_pairs'] += int(best[0])
        remaining -= assigned
    out: list[dict[str, Any]] = []
    delta = Counter(); same_row = 0; self_pairs = 0; costs: list[float] = []
    for i, ev in enumerate(events):
        if i not in assignment:
            continue
        donor_idx, level = assignment[i]
        donor = events[donor_idx]
        if donor_idx == i:
            self_pairs += 1
            continue
        e2 = dict(ev)
        e2['negative_target_positions'] = donor['target_positions']
        e2['negative_target_batch_row'] = int(donor['batch_row'])
        e2['negative_target_row_tail_idx'] = int(donor['row_tail_idx'])
        e2['negative_target_event_rank'] = int(donor['event_rank'])
        e2['negative_target_norm'] = donor.get('target_norm')
        e2['negative_target_class'] = donor.get('target_class')
        e2['negative_target_freq_bin'] = donor.get('target_freq_bin')
        e2['negative_target_len'] = donor.get('target_len')
        e2['cross_target_match_level'] = int(level)
        cost = float(_pair_cost(ev, donor)); e2['cross_target_pair_cost'] = cost
        out.append(e2); costs.append(cost)
        same_row += int(ev['row_tail_idx'] == donor['row_tail_idx'])
        delta['target_freq_abs_delta_sum'] += abs(int(ev.get('target_freq_bin_id', -1)) - int(donor.get('target_freq_bin_id', -1)))
        delta['distance_bin_abs_delta_sum'] += abs(int(ev.get('distance_bin_id', -1)) - int(donor.get('distance_bin_id', -1)))
        delta['target_len_abs_delta_sum'] += abs(int(ev.get('target_len', 1)) - int(donor.get('target_len', 1)))
        delta['pivot_freq_abs_delta_sum'] += abs(int(ev.get('pivot_freq_bin_id', -1)) - int(donor.get('pivot_freq_bin_id', -1)))
        delta['pivot_target_distance_abs_delta_sum'] += abs(int(ev.get('pivot_target_distance', -1)) - int(donor.get('pivot_target_distance', -1)))
    used = len(out)
    stats = {
        'input_events': len(events), 'used_events': used, 'dropped_singleton_events': len(events) - used,
        'self_pairs': int(self_pairs), 'same_row_donor_pairs': int(same_row), 'different_row_donor_pairs': int(used - same_row),
        'match_level_counts': dict(level_counts), 'group_size_counts': dict(group_size_counts), 'shift_same_row': dict(shift_same_row),
        'mean_pair_cost': float(sum(costs) / used) if used else None,
        'mean_target_freq_abs_delta': float(delta['target_freq_abs_delta_sum'] / used) if used else None,
        'mean_distance_bin_abs_delta': float(delta['distance_bin_abs_delta_sum'] / used) if used else None,
        'mean_target_len_abs_delta': float(delta['target_len_abs_delta_sum'] / used) if used else None,
        'mean_pivot_freq_abs_delta': float(delta['pivot_freq_abs_delta_sum'] / used) if used else None,
        'mean_pivot_target_distance_abs_delta': float(delta['pivot_target_distance_abs_delta_sum'] / used) if used else None,
    }
    return out, stats


def merge_perm_stats(dst: dict[str, Any], stats: dict[str, Any]) -> None:
    dst['calls'] += 1
    dst['input_events'] += int(stats.get('input_events', 0) or 0)
    dst['used_events'] += int(stats.get('used_events', 0) or 0)
    dst['dropped_singleton_events'] += int(stats.get('dropped_singleton_events', 0) or 0)
    dst['self_pairs'] += int(stats.get('self_pairs', 0) or 0)
    dst['same_row_donor_pairs'] += int(stats.get('same_row_donor_pairs', 0) or 0)
    dst['different_row_donor_pairs'] += int(stats.get('different_row_donor_pairs', 0) or 0)
    for k, v in (stats.get('match_level_counts') or {}).items():
        dst['match_level_counts'][str(k)] += int(v)
    for k, v in (stats.get('group_size_counts') or {}).items():
        dst['group_size_counts'][str(k)] += int(v)
    used = int(stats.get('used_events', 0) or 0)
    for key in ['mean_pair_cost', 'mean_target_freq_abs_delta', 'mean_distance_bin_abs_delta', 'mean_target_len_abs_delta', 'mean_pivot_freq_abs_delta', 'mean_pivot_target_distance_abs_delta']:
        val = stats.get(key)
        if val is not None:
            dst[key + '_weighted_sum'] += float(val) * used


def finalize_perm_stats(dst: dict[str, Any]) -> dict[str, Any]:
    used = int(dst.get('used_events', 0) or 0)
    out = {k: v for k, v in dst.items() if not k.endswith('_weighted_sum') and k not in {'match_level_counts', 'group_size_counts'}}
    out['match_level_counts'] = dict(dst.get('match_level_counts', Counter()))
    out['group_size_counts'] = dict(dst.get('group_size_counts', Counter()))
    for key in ['mean_pair_cost', 'mean_target_freq_abs_delta', 'mean_distance_bin_abs_delta', 'mean_target_len_abs_delta', 'mean_pivot_freq_abs_delta', 'mean_pivot_target_distance_abs_delta']:
        out[key] = (float(dst.get(key + '_weighted_sum', 0.0)) / used) if used else None
    out['used_fraction_of_selected'] = used / max(1, int(dst.get('input_events', 0))) if int(dst.get('input_events', 0) or 0) else None
    out['different_row_fraction'] = int(dst.get('different_row_donor_pairs', 0) or 0) / max(1, used) if used else None
    return out


def relation_aux_loss_prepared(hidden: torch.Tensor, head: RelationAuxHead, prepared: list[dict[str, Any]], *, mode: str) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if not prepared:
        return None, {'events_used': 0}
    logits = []; labels = []
    cats: Counter = Counter(); levels: Counter = Counter(); same_row_pairs = 0
    for ev in prepared:
        b = int(ev['batch_row'])
        if mode == 'semantic':
            anchor_row = b; anchor_positions = ev['pivot_positions']
        elif mode == 'anchor_permuted':
            anchor_row = b; anchor_positions = ev['control_positions']
        else:
            raise ValueError(mode)
        a_rep = group_rep(hidden, anchor_row, anchor_positions)
        t_rep = group_rep(hidden, b, ev['target_positions'])
        n_rep = group_rep(hidden, int(ev['negative_target_batch_row']), ev['negative_target_positions'])
        s_pos = head.sim(a_rep.unsqueeze(0), t_rep.unsqueeze(0)).squeeze(0)
        s_neg = head.sim(a_rep.unsqueeze(0), n_rep.unsqueeze(0)).squeeze(0)
        s_pos_sym = head.sim(t_rep.unsqueeze(0), a_rep.unsqueeze(0)).squeeze(0)
        s_neg_sym = head.sim(n_rep.unsqueeze(0), a_rep.unsqueeze(0)).squeeze(0)
        logits.append(torch.stack([s_pos, s_neg])); logits.append(torch.stack([s_pos_sym, s_neg_sym]))
        labels.append(0); labels.append(0)
        cats[str(ev['category'])] += 1
        levels[str(ev.get('cross_target_match_level', 'NA'))] += 1
        same_row_pairs += int(ev.get('negative_target_row_tail_idx') == ev.get('row_tail_idx'))
    logit_t = torch.stack(logits, dim=0)
    label_t = torch.zeros(len(labels), dtype=torch.long, device=hidden.device)
    loss = F.cross_entropy(logit_t, label_t)
    with torch.no_grad():
        margin = logit_t[:, 0] - logit_t[:, 1]
        pred = logit_t.argmax(dim=-1)
        stats = {
            'events_used': len(prepared), 'terms': int(logit_t.shape[0]),
            'aux_accuracy': float((pred == label_t).float().mean().item()),
            'mean_target_minus_cross_target_margin': float(margin.mean().item()),
            'categories': dict(cats), 'match_levels': dict(levels),
            'same_row_negative_target_pairs': int(same_row_pairs),
            'orientation_label_0_terms': int((label_t == 0).sum().item()),
            'orientation_label_1_terms': int((label_t == 1).sum().item()),
        }
    return loss, stats


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='research WWM-preserving sparse relation auxiliary trainer')
    ap.add_argument('--mode', choices=['semantic', 'anchor_permuted'], required=True)
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
    ap.add_argument('--max_control_distance_abs_diff', type=int, default=1)
    ap.add_argument('--max_control_freq_bin_abs_diff', type=int, default=1)
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
    t0 = time.time()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tail_path = Path(args.tail_jsonl); labels_path = Path(args.labels_jsonl); tok_path = Path(args.tokenizer_path); init_ckpt = Path(args.init_checkpoint)
    hashes = {'tail_sha256': base.sha256_file(tail_path), 'labels_sha256': base.sha256_file(labels_path), 'tokenizer_sha256': base.sha256_file(tok_path / 'tokenizer.json')}
    for k, expected in EXPECTED.items():
        if hashes[k] != expected:
            raise RuntimeError(f'{k} mismatch {hashes[k]} != {expected}')
    if not init_ckpt.exists():
        raise RuntimeError(f'init checkpoint not found: {init_ckpt}')
    tokenizer = base.make_portable_tokenizer(str(tok_path))
    examples, label_records, segment = cont.load_segment(tail_path, labels_path, start_tail_row=args.start_tail_row, expected_start_tail_words=args.expected_start_tail_words, max_word_exposure=args.max_word_exposure, max_rows=args.max_rows)
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    stage_steps = math.ceil(len(dataset) / args.batch_size)
    schedule_total = max(args.total_schedule_steps, stage_steps)
    warmup = max(1, int(schedule_total * args.warmup_fraction))

    reset_all_rng(args.seed)
    model = DebertaV2ForMaskedLM.from_pretrained(str(init_ckpt))
    if model.config.vocab_size != len(tokenizer):
        raise RuntimeError(f'vocab mismatch model {model.config.vocab_size} tokenizer {len(tokenizer)}')
    reset_all_rng(args.train_rng_seed + 1220)
    head = RelationAuxHead(int(model.config.hidden_size), rank=args.aux_rank, temperature=args.aux_temperature)
    reset_all_rng(args.train_rng_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device); head.to(device); model.train(); head.train()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    param_count = sum(p.numel() for p in model.parameters())
    head_param_count = sum(p.numel() for p in head.parameters())
    optim = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)
    legacy_state = base.MaskingCurriculumState(curriculum='wwm_fixed', mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob, switch_frac=0.7)
    legacy_state.initialize(vocab_size=len(tokenizer), total_steps=schedule_total)
    legacy_gen = torch.Generator(device=device); legacy_gen.manual_seed(args.train_rng_seed)
    allowed_categories = set(args.categories) & ALL_CATEGORIES
    if not allowed_categories:
        raise ValueError('no allowed relation categories')
    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    manifest = {
        'status': 'SPARSE_RELATION_AUX_MATCHED_ANCHOR_PERMUTATION_MANIFEST',
        'created_utc': now_utc(), 'mode': args.mode, 'output_dir': str(out),
        'init_checkpoint': str(init_ckpt), 'initial_actual_word_exposure': args.initial_actual_word_exposure,
        'target_stage_stop': args.checkpoint_name, 'stage_is_physically_stopped_at_end': True,
        'segment': segment, 'hashes': hashes,
        'ordinary_wwm_preserved': True, 'mlm_targets_unchanged_from_standard_legacy': True,
        'semantic_vs_anchor_permuted_event_selection_shared': True,
        'control_definition': 'same-row matched nonrelation anchor replaces true pivot; cross-event dependent target supplies the negative in both modes',
        'effective_batch_size': args.batch_size, 'micro_batch_size': args.micro_batch_size,
        'gradient_accumulation_steps': args.batch_size // args.micro_batch_size,
        'stage_steps': stage_steps, 'total_schedule_steps_for_possible_70M_to_100M_resume': schedule_total,
        'optimizer_reset': 'AdamW reset, matched to research standard WWM branch',
        'learning_rate': args.learning_rate, 'warmup_fraction': args.warmup_fraction, 'weight_decay': args.weight_decay,
        'mask_prob': args.mask_prob, 'aux_weight': args.aux_weight, 'aux_rank': args.aux_rank,
        'aux_temperature': args.aux_temperature, 'event_prob': args.event_prob,
        'max_events_per_batch': args.max_events_per_batch,
        'max_control_distance_abs_diff': args.max_control_distance_abs_diff,
        'max_control_freq_bin_abs_diff': args.max_control_freq_bin_abs_diff,
        'categories': sorted(allowed_categories), 'seed': args.seed, 'train_rng_seed': args.train_rng_seed,
        'source_words_consumed': source_words,
        'causal_split_source': 'research/notes/representation_and_objectives/legacy_80m_causal_split.md',
        'flawed_hash_random_placebo_not_used': 'experiments/archive/representation_and_objectives/scripts/sparse_relation_aux_trainer.py',
    }
    (out / 'example_order_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'sparse_relation_aux_start', 'mode': args.mode, 'output_dir': str(out), 'device': str(device), 'param_count': param_count, 'head_param_count': head_param_count, 'stage_rows': len(examples), 'stage_words': segment['selected_words'], 'stage_steps': stage_steps, 'expected_total_at_stage_end': args.initial_actual_word_exposure + segment['selected_words'], 'aux_weight': args.aux_weight, 'event_prob': args.event_prob, 'max_control_distance_abs_diff': args.max_control_distance_abs_diff, 'max_control_freq_bin_abs_diff': args.max_control_freq_bin_abs_diff}), flush=True)

    log_path = out / 'training_log.jsonl'
    aux_stats_path = out / 'relation_aux_stats.jsonl'
    loss_values: list[float] = []; mlm_loss_values: list[float] = []; aux_loss_values: list[float] = []
    cumulative_words = 0
    aggregate: Counter = Counter(); cat_stats: Counter = Counter(); reject_stats: Counter = Counter(); orientation_terms: Counter = Counter()
    perm_aggregate: dict[str, Any] = defaultdict(float)
    perm_aggregate['match_level_counts'] = Counter(); perm_aggregate['group_size_counts'] = Counter()

    with log_path.open('w', encoding='utf-8') as logf, aux_stats_path.open('w', encoding='utf-8') as auxf:
        for step, batch in enumerate(loader, 1):
            lo = (step - 1) * args.batch_size; hi = lo + int(batch['input_ids'].shape[0])
            words = int(batch.pop('words').sum().item())
            input_ids_cpu = batch['input_ids'][:, :args.seq_length].contiguous()
            attention_cpu = batch['attention_mask'][:, :args.seq_length].contiguous()
            word_group_cpu = batch['word_group'][:, :args.seq_length].contiguous()
            legacy_state.current_step = step - 1
            masked_dev, labels_dev_full = base.apply_masking_curriculum(input_ids_cpu.to(device, non_blocking=True), attention_cpu.to(device, non_blocking=True), word_group_cpu.to(device, non_blocking=True), tokenizer, legacy_state, legacy_gen)
            masked_cpu, labels_cpu = masked_dev.cpu(), labels_dev_full.cpu()
            del masked_dev, labels_dev_full
            n_pred_total = int((labels_cpu != -100).sum().item())
            if n_pred_total <= 0:
                raise RuntimeError(f'no masked tokens at step {step}')
            all_events, seen, rej = select_aux_events(word_group_cpu, attention_cpu, label_records[lo:hi], seed=args.train_rng_seed, batch_step=step, event_prob=args.event_prob, max_events_per_batch=args.max_events_per_batch, allowed_categories=allowed_categories, max_control_distance_abs_diff=args.max_control_distance_abs_diff, max_control_freq_bin_abs_diff=args.max_control_freq_bin_abs_diff)
            aggregate['batches'] += 1; aggregate['masked_tokens'] += n_pred_total; aggregate['events_selected_before_cross_target_filter'] += len(all_events)
            reject_stats.update(rej)
            events_by_mb: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for ev in all_events:
                events_by_mb[int(ev['batch_row']) // args.micro_batch_size].append(ev)
            prepared_by_mb: dict[int, list[dict[str, Any]]] = {}
            step_perm_stats: dict[str, Any] = defaultdict(float)
            step_perm_stats['match_level_counts'] = Counter(); step_perm_stats['group_size_counts'] = Counter()
            total_events_input = 0; total_events_used = 0
            for mb_idx, evs in events_by_mb.items():
                mb_start0 = int(mb_idx) * args.micro_batch_size
                rel_events = []
                for ev in evs:
                    ev2 = dict(ev); ev2['batch_row'] = int(ev['batch_row']) - mb_start0; rel_events.append(ev2)
                prepared, pstats = assign_cross_targets(rel_events, seed=args.train_rng_seed, step=step, microbatch_index=int(mb_idx))
                prepared_by_mb[int(mb_idx)] = prepared
                total_events_input += len(rel_events); total_events_used += len(prepared)
                merge_perm_stats(step_perm_stats, pstats)
            for evs in prepared_by_mb.values():
                cat_stats.update([e['category'] for e in evs])

            optim.zero_grad(set_to_none=True)
            weighted_mlm = 0.0; weighted_aux = 0.0; active_microbatches = 0
            for mb_start in range(0, int(masked_cpu.shape[0]), args.micro_batch_size):
                mb_end = min(mb_start + args.micro_batch_size, int(masked_cpu.shape[0]))
                mb_idx = mb_start // args.micro_batch_size
                sl_labels_cpu = labels_cpu[mb_start:mb_end]
                n_pred_i = int((sl_labels_cpu != -100).sum().item())
                if n_pred_i <= 0:
                    continue
                input_ids = masked_cpu[mb_start:mb_end].to(device, non_blocking=True)
                attention_mask = attention_cpu[mb_start:mb_end].to(device, non_blocking=True)
                labels_dev = sl_labels_cpu.to(device, non_blocking=True)
                out_model = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_dev, output_hidden_states=True)
                mlm_loss = out_model.loss
                if mlm_loss is None:
                    raise RuntimeError('model returned no MLM loss')
                mlm_scale = n_pred_i / n_pred_total
                hidden = out_model.hidden_states[-1]
                prepared_events = prepared_by_mb.get(int(mb_idx), [])
                aux_loss, aux_stats = relation_aux_loss_prepared(hidden, head, prepared_events, mode=args.mode)
                loss = mlm_loss * mlm_scale
                if aux_loss is not None and total_events_used > 0:
                    used_here = int(aux_stats.get('events_used', 0) or 0)
                    aux_scale = args.aux_weight * (used_here / total_events_used)
                    loss = loss + aux_loss * aux_scale
                    weighted_aux += float(aux_loss.detach().cpu()) * (used_here / total_events_used)
                    orientation_terms['label0'] += int(aux_stats.get('orientation_label_0_terms', 0)); orientation_terms['label1'] += int(aux_stats.get('orientation_label_1_terms', 0))
                loss.backward()
                weighted_mlm += float(mlm_loss.detach().cpu()) * mlm_scale
                active_microbatches += 1
                del out_model, mlm_loss, loss, input_ids, attention_mask, labels_dev, hidden
            if active_microbatches <= 0:
                raise RuntimeError(f'no active microbatches at step {step}')
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            optim.step(); sched.step()
            cumulative_words += words
            aux_event_mean = weighted_aux if total_events_used else None
            aggregate['events_input_to_cross_target_filter'] += int(total_events_input)
            aggregate['events_used_after_cross_target_filter'] += int(total_events_used)
            for k, v in step_perm_stats.items():
                if k in ('match_level_counts', 'group_size_counts'):
                    perm_aggregate[k].update(v)
                else:
                    perm_aggregate[k] += v
            step_perm_final = finalize_perm_stats(step_perm_stats)
            loss_reported = float(weighted_mlm + args.aux_weight * (aux_event_mean or 0.0))
            loss_values.append(loss_reported); mlm_loss_values.append(float(weighted_mlm)); aux_loss_values.append(float(aux_event_mean) if aux_event_mean is not None else 0.0)
            rec = {'step': step, 'loss_total_reported': loss_reported, 'mlm_loss': float(weighted_mlm), 'aux_loss_event_mean': aux_event_mean, 'aux_weight': args.aux_weight, 'lr': float(sched.get_last_lr()[0]), 'batch_words': words, 'cumulative_continuation_words': cumulative_words, 'total_actual_word_exposure': args.initial_actual_word_exposure + cumulative_words, 'seq_len': args.seq_length, 'masked_tokens': n_pred_total, 'effective_mask_rate': round(n_pred_total / max(1, int(attention_cpu.sum().item())), 6), 'aux_events_selected': len(all_events), 'aux_events_input_to_cross_target_filter': int(total_events_input), 'aux_events_used': int(total_events_used), 'cross_target_used_fraction': step_perm_final.get('used_fraction_of_selected'), 'cross_target_different_row_fraction': step_perm_final.get('different_row_fraction'), 'mode': args.mode, 'active_microbatches': active_microbatches, 'micro_batch_size': args.micro_batch_size, 'elapsed_sec': round(time.time() - t0, 1)}
            logf.write(json.dumps(rec) + '\n'); logf.flush()
            auxf.write(json.dumps({'step': step, 'events_selected_before_filter': len(all_events), 'events_input_to_cross_target_filter': int(total_events_input), 'events_used_after_cross_target_filter': int(total_events_used), 'selected_categories_used_cumulative': dict(cat_stats), 'reject_reasons': dict(rej), 'category_seen': dict(seen), 'orientation_terms_cumulative': dict(orientation_terms), 'cross_target_step': step_perm_final}, ensure_ascii=False) + '\n'); auxf.flush()
            if step == 1 or step % args.log_every == 0 or step == stage_steps:
                print(json.dumps({'event': 'train', **rec}), flush=True)
            del masked_cpu, labels_cpu, input_ids_cpu, attention_cpu, word_group_cpu

    ckpt_dir = out / 'hf_model' / args.checkpoint_name
    base.save_hf_checkpoint(model, tokenizer, ckpt_dir)
    base.save_hf_checkpoint(model, tokenizer, out / 'hf_model')
    aux_dir = out / 'relation_aux_head' / args.checkpoint_name
    aux_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'head': head.state_dict(), 'manifest': manifest}, aux_dir / 'relation_aux_head.pt')
    trainer_state_path = None
    if args.save_trainer_state:
        state_dir = out / 'trainer_state' / args.checkpoint_name
        state_dir.mkdir(parents=True, exist_ok=True)
        trainer_state_path = state_dir / 'optimizer_scheduler_rng.pt'
        torch.save({'optimizer': optim.state_dict(), 'scheduler': sched.state_dict(), 'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, 'python_random_state': random.getstate(), 'numpy_random_state': np.random.get_state(), 'completed_stage_steps': stage_steps, 'cumulative_continuation_words': cumulative_words, 'total_actual_word_exposure': args.initial_actual_word_exposure + cumulative_words, 'last_tail_row': segment['last_tail_row'], 'next_tail_row': int(segment['last_tail_row']) + 1 if segment['last_tail_row'] is not None else None, 'schedule_total_steps': schedule_total, 'relation_aux_head': head.state_dict()}, trainer_state_path)
    perm_final = finalize_perm_stats(perm_aggregate)
    cuda_peak_allocated_mb = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else None
    cuda_peak_reserved_mb = float(torch.cuda.max_memory_reserved() / (1024**2)) if torch.cuda.is_available() else None
    metrics = {
        'variant': f'sparse_relation_aux_{args.mode}_matched_anchor_permutation',
        'backend': 'mlm', 'model_family': 'DebertaV2ForMaskedLM', 'parameter_count': param_count,
        'aux_head_parameter_count_not_saved_in_hf_model': head_param_count, 'vocab_size': len(tokenizer),
        'word_exposure': args.initial_actual_word_exposure + cumulative_words, 'continuation_words': cumulative_words,
        'loss_first': loss_values[0] if loss_values else None, 'loss_last': loss_values[-1] if loss_values else None,
        'mlm_loss_first': mlm_loss_values[0] if mlm_loss_values else None, 'mlm_loss_last': mlm_loss_values[-1] if mlm_loss_values else None,
        'aux_loss_first': aux_loss_values[0] if aux_loss_values else None, 'aux_loss_last': aux_loss_values[-1] if aux_loss_values else None,
        'actual_training_steps': stage_steps, 'effective_batch_size': args.batch_size, 'micro_batch_size': args.micro_batch_size,
        'gradient_accumulation_steps': args.batch_size // args.micro_batch_size, 'cuda_peak_allocated_mb': cuda_peak_allocated_mb, 'cuda_peak_reserved_mb': cuda_peak_reserved_mb,
        'stage_stop_name': args.checkpoint_name, 'ordinary_wwm_preserved': True,
        'saved_checkpoints': [{'name': args.checkpoint_name, 'target_word_exposure': args.initial_actual_word_exposure + args.max_word_exposure, 'actual_cumulative_word_exposure': args.initial_actual_word_exposure + cumulative_words, 'path': str(ckpt_dir)}],
        'relation_aux': {'mode': args.mode, 'aux_weight': args.aux_weight, 'aux_rank': args.aux_rank, 'aux_temperature': args.aux_temperature, 'event_prob': args.event_prob, 'max_events_per_batch': args.max_events_per_batch, 'max_control_distance_abs_diff': args.max_control_distance_abs_diff, 'max_control_freq_bin_abs_diff': args.max_control_freq_bin_abs_diff, 'categories': sorted(allowed_categories), 'aggregate': dict(aggregate), 'selected_event_categories_used': dict(cat_stats), 'reject_reasons': dict(reject_stats), 'orientation_terms': dict(orientation_terms), 'cross_target_summary': perm_final, 'head_path': str(aux_dir / 'relation_aux_head.pt')},
        'trainer_state_path': str(trainer_state_path) if trainer_state_path else None,
        'example_order_manifest': str(out / 'example_order_manifest.json'), 'training_log': str(log_path), 'relation_aux_stats': str(aux_stats_path),
        **manifest,
    }
    (out / 'scientific_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', 'mode': args.mode, 'output_dir': str(out), 'checkpoint': str(ckpt_dir), 'continuation_words': cumulative_words, 'total_actual_word_exposure': args.initial_actual_word_exposure + cumulative_words, 'loss_first': metrics['loss_first'], 'loss_last': metrics['loss_last'], 'mlm_loss_last': metrics['mlm_loss_last'], 'aux_loss_last': metrics['aux_loss_last'], 'events_used': int(aggregate.get('events_used_after_cross_target_filter', 0)), 'cross_target_used_fraction': perm_final.get('used_fraction_of_selected'), 'cross_target_different_row_fraction': perm_final.get('different_row_fraction'), 'cuda_peak_allocated_mb': cuda_peak_allocated_mb, 'cuda_peak_reserved_mb': cuda_peak_reserved_mb, 'trainer_state_path': metrics['trainer_state_path']}), flush=True)


if __name__ == '__main__':
    main()
