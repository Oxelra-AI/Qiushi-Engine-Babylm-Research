#!/usr/bin/env python3
"""Deterministic paired PVDM masking utilities for research.

The functions here implement the *training intervention*, not a learned parser or
external linguistic tool.  They consume research hand-coded labels and construct a
paired masking contrast:

  treatment: dependent target masked, true relation pivot visible, matched
             surrogate anchor masked;
  control:   same dependent target masked, matched surrogate anchor visible,
             true relation pivot masked.

Events are used only when the pivot/control/target whitespace words survive the
actual tokenizer truncation and the pivot/control word-groups have the same
subword-token length.  This makes the anchor swap preserve realized masked token
mass.  Background WWM groups are selected by deterministic hash and are identical
between arms, so the paired arms differ only in the event anchor swap while
sharing dependent targets, background masks, and 80/10/10 replacement draws.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class ActiveEvent:
    row_tail_idx: int
    event_rank: int
    pivot_gid: int
    target_gid: int
    control_gid: int
    category: str
    pivot_len: int
    target_len: int
    control_len: int


def _digest(seed: int, *parts: Any) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(seed).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8", errors="replace"))
    return h.digest()


def hash_uniform(seed: int, *parts: Any) -> float:
    """Stable pseudo-random float in [0,1)."""
    x = int.from_bytes(_digest(seed, *parts)[:8], "big")
    return x / float(1 << 64)


def hash_int(seed: int, modulus: int, *parts: Any) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    x = int.from_bytes(_digest(seed, *parts)[:8], "big")
    return x % modulus


def summarize_counter(counter: Counter, top: int | None = None) -> dict[str, int] | list[tuple[str, int]]:
    if top is None:
        return {str(k): int(v) for k, v in counter.items()}
    return [(str(k), int(v)) for k, v in counter.most_common(top)]


def group_positions_for_row(word_group_1d: torch.Tensor, attention_mask_1d: torch.Tensor) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    wg = word_group_1d.tolist()
    am = attention_mask_1d.tolist()
    for pos, (g, a) in enumerate(zip(wg, am)):
        if int(a) and int(g) >= 0:
            groups[int(g)].append(pos)
    return dict(groups)


def collect_active_events(label_record: dict[str, Any], group_pos: dict[int, list[int]], same_length_required: bool = True) -> tuple[list[ActiveEvent], Counter]:
    """Map research whitespace-index events to actual tokenizer word groups.

    Returns usable active events and rejection counts.  Group ids produced by the
    existing byte-BPE WWM dataset correspond to whitespace word indices for the
    retained, non-truncated prefix.  Events outside the retained prefix are
    rejected rather than silently remapped.
    """
    row_tail_idx = int(label_record.get("tail_row_idx", -1))
    usable: list[ActiveEvent] = []
    reject = Counter()
    used_targets: set[int] = set()
    for rank, ev in enumerate(label_record.get("events", [])):
        try:
            p = int(ev["pivot_i"]); t = int(ev["target_i"]); c = int(ev["control_i"])
        except Exception:
            reject["malformed_event"] += 1
            continue
        cat = str(ev.get("category", "unknown"))
        if p not in group_pos or t not in group_pos or c not in group_pos:
            reject[f"truncated_or_missing::{cat}"] += 1
            reject["truncated_or_missing"] += 1
            continue
        if t in used_targets:
            reject[f"duplicate_target::{cat}"] += 1
            reject["duplicate_target"] += 1
            continue
        p_len = len(group_pos[p]); t_len = len(group_pos[t]); c_len = len(group_pos[c])
        if p_len <= 0 or t_len <= 0 or c_len <= 0:
            reject[f"empty_group::{cat}"] += 1
            reject["empty_group"] += 1
            continue
        if same_length_required and p_len != c_len:
            reject[f"anchor_token_length_mismatch::{cat}"] += 1
            reject["anchor_token_length_mismatch"] += 1
            continue
        usable.append(ActiveEvent(row_tail_idx, rank, p, t, c, cat, p_len, t_len, c_len))
        used_targets.add(t)
    return usable, reject


def _desired_group_count(seed: int, row_tail_idx: int, n_valid_groups: int, mask_prob: float) -> int:
    if n_valid_groups <= 0:
        return 0
    x = mask_prob * n_valid_groups
    k = int(math.floor(x))
    frac = x - k
    if hash_uniform(seed, "row_mask_count", row_tail_idx, n_valid_groups) < frac:
        k += 1
    return max(1, min(n_valid_groups, k))


def _select_events(seed: int, events: list[ActiveEvent], target_prob: float) -> list[ActiveEvent]:
    """Select a row-local subset of events with disjoint pivot/target/control roles.

    research real-batch checking exposed that identical target sequences and
    per-event pivot/control length equality are not enough: if one event's pivot
    or surrogate is also another event's target/pivot/surrogate, set unions can
    change realized masked mass differently across arms.  We therefore require
    all selected event role groups to be disjoint within a row.  This is stricter
    than research labels but preserves a clean causal contrast.
    """
    selected: list[ActiveEvent] = []
    occupied_roles: set[int] = set()
    # Keep deterministic priority but avoid systematic left-to-right only: sort
    # by a stable hash after category/event fields.
    ranked = sorted(
        events,
        key=lambda e: (
            hash_uniform(seed, "event_rank", e.row_tail_idx, e.event_rank, e.category, e.target_gid),
            e.target_gid,
            e.pivot_gid,
            e.control_gid,
        ),
    )
    for ev in ranked:
        roles = {ev.pivot_gid, ev.target_gid, ev.control_gid}
        if len(roles) < 3 or roles & occupied_roles:
            continue
        u = hash_uniform(seed, "target_select", ev.row_tail_idx, ev.event_rank, ev.category, ev.target_gid)
        if u < target_prob:
            selected.append(ev)
            occupied_roles.update(roles)
    # Preserve row-order readability downstream.
    selected.sort(key=lambda e: e.event_rank)
    return selected


def make_paired_group_selections(
    word_group: torch.Tensor,
    attention_mask: torch.Tensor,
    label_records: list[dict[str, Any]],
    *,
    seed: int = 43023,
    mask_prob: float = 0.15,
    target_prob: float = 0.60,
    same_length_required: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return per-row treatment/control selected groups and pair stats.

    The returned records contain group-id sets and event-key maps.  No tensors are
    modified here; `apply_replacements_from_group_records` materializes MLM
    inputs/labels.
    """
    bsz = int(word_group.shape[0])
    row_records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "rows": bsz,
        "valid_groups": 0,
        "label_events_total": 0,
        "events_tokenization_usable": 0,
        "events_selected": 0,
        "rows_with_label_events": 0,
        "rows_with_usable_events": 0,
        "rows_with_selected_events": 0,
        "desired_group_count_sum": 0,
        "treatment_group_count_sum": 0,
        "control_group_count_sum": 0,
        "treatment_token_count_sum": 0,
        "control_token_count_sum": 0,
        "target_group_count_sum": 0,
        "background_group_count_sum": 0,
        "anchor_swap_group_count_sum": 0,
        "label_event_categories": Counter(),
        "usable_event_categories": Counter(),
        "selected_event_categories": Counter(),
        "reject_reasons": Counter(),
        "pivot_visible_at_selected_target": {"treatment_visible": 0, "treatment_total": 0, "control_visible": 0, "control_total": 0},
        "rows_forced_above_desired_k": 0,
    }
    for b in range(bsz):
        lr = label_records[b]
        row_tail_idx = int(lr.get("tail_row_idx", b))
        events_raw = list(lr.get("events", []))
        if events_raw:
            stats["rows_with_label_events"] += 1
        stats["label_events_total"] += len(events_raw)
        for ev in events_raw:
            stats["label_event_categories"][str(ev.get("category", "unknown"))] += 1
        group_pos = group_positions_for_row(word_group[b], attention_mask[b])
        valid_groups = sorted(group_pos)
        stats["valid_groups"] += len(valid_groups)
        usable, reject = collect_active_events(lr, group_pos, same_length_required=same_length_required)
        stats["reject_reasons"].update(reject)
        if usable:
            stats["rows_with_usable_events"] += 1
        stats["events_tokenization_usable"] += len(usable)
        for ev in usable:
            stats["usable_event_categories"][ev.category] += 1
        selected_events = _select_events(seed, usable, target_prob)
        if selected_events:
            stats["rows_with_selected_events"] += 1
        stats["events_selected"] += len(selected_events)
        for ev in selected_events:
            stats["selected_event_categories"][ev.category] += 1

        desired_k = _desired_group_count(seed, row_tail_idx, len(valid_groups), mask_prob)
        selected_targets = {ev.target_gid for ev in selected_events}
        pivots = {ev.pivot_gid for ev in selected_events}
        controls = {ev.control_gid for ev in selected_events}
        anchor_union = pivots | controls
        active_common = set(selected_targets)
        treatment_groups = set(selected_targets) | controls
        control_groups = set(selected_targets) | pivots
        # If the event contrast itself exceeds baseline group-count mass, keep it
        # rather than dropping relational target events, but record this explicitly.
        min_needed = max(len(treatment_groups), len(control_groups))
        if min_needed > desired_k:
            stats["rows_forced_above_desired_k"] += 1
            desired_eff = min_needed
        else:
            desired_eff = desired_k
        bg_candidates = [g for g in valid_groups if g not in selected_targets and g not in anchor_union]
        bg_candidates.sort(key=lambda g: hash_uniform(seed, "background_rank", row_tail_idx, g))
        background: list[int] = []
        for g in bg_candidates:
            if len(treatment_groups) >= desired_eff and len(control_groups) >= desired_eff:
                break
            background.append(g)
            treatment_groups.add(g)
            control_groups.add(g)
        # Construct replacement role keys.  Target/background keys are identical;
        # swapped anchors share an event key so 80/10/10 actions match when group
        # token lengths match.
        treatment_keys: dict[int, tuple[Any, ...]] = {}
        control_keys: dict[int, tuple[Any, ...]] = {}
        for ev in selected_events:
            ev_key = ("event", ev.row_tail_idx, ev.event_rank, ev.category, ev.target_gid)
            treatment_keys[ev.target_gid] = ("target",) + ev_key
            control_keys[ev.target_gid] = ("target",) + ev_key
            treatment_keys[ev.control_gid] = ("anchor_swap",) + ev_key
            control_keys[ev.pivot_gid] = ("anchor_swap",) + ev_key
        for g in background:
            k = ("background", row_tail_idx, g)
            treatment_keys[g] = k
            control_keys[g] = k
        t_tok = sum(len(group_pos[g]) for g in treatment_groups)
        c_tok = sum(len(group_pos[g]) for g in control_groups)
        stats["desired_group_count_sum"] += desired_k
        stats["treatment_group_count_sum"] += len(treatment_groups)
        stats["control_group_count_sum"] += len(control_groups)
        stats["treatment_token_count_sum"] += t_tok
        stats["control_token_count_sum"] += c_tok
        stats["target_group_count_sum"] += len(selected_targets)
        stats["background_group_count_sum"] += len(background)
        stats["anchor_swap_group_count_sum"] += len(selected_events)
        # At every selected event, the treatment pivot is visible; the control
        # pivot is the swapped masked anchor.  These counts are event-level.
        stats["pivot_visible_at_selected_target"]["treatment_visible"] += len(selected_events)
        stats["pivot_visible_at_selected_target"]["treatment_total"] += len(selected_events)
        stats["pivot_visible_at_selected_target"]["control_visible"] += 0
        stats["pivot_visible_at_selected_target"]["control_total"] += len(selected_events)
        row_records.append({
            "row_tail_idx": row_tail_idx,
            "group_pos": group_pos,
            "valid_group_count": len(valid_groups),
            "desired_group_count": desired_k,
            "selected_events": selected_events,
            "selected_targets": selected_targets,
            "background_groups": set(background),
            "treatment_groups": treatment_groups,
            "control_groups": control_groups,
            "treatment_keys": treatment_keys,
            "control_keys": control_keys,
            "treatment_token_count": t_tok,
            "control_token_count": c_tok,
        })
    return row_records, stats


def _materialize_one_arm(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    row_records: list[dict[str, Any]],
    *,
    arm: str,
    tokenizer_len: int,
    mask_token_id: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if arm not in {"treatment", "control"}:
        raise ValueError(f"unknown arm {arm}")
    masked_inputs = input_ids.clone()
    labels = input_ids.clone()
    select = torch.zeros_like(input_ids, dtype=torch.bool)
    action_counts = Counter()
    replacement_digest = hashlib.sha256()
    for b, rr in enumerate(row_records):
        groups = rr[f"{arm}_groups"]
        key_map = rr[f"{arm}_keys"]
        group_pos = rr["group_pos"]
        for g in groups:
            positions = group_pos.get(int(g), [])
            role_key = key_map.get(int(g), ("fallback", rr["row_tail_idx"], int(g)))
            for off, pos in enumerate(positions):
                select[b, pos] = True
                u = hash_uniform(seed, "replace", *role_key, off)
                if u < 0.8:
                    masked_inputs[b, pos] = int(mask_token_id)
                    action = "mask_token"
                elif u < 0.9:
                    rid = hash_int(seed, tokenizer_len, "rand_id", *role_key, off)
                    masked_inputs[b, pos] = int(rid)
                    action = "random_token"
                else:
                    action = "keep_original"
                action_counts[action] += 1
                replacement_digest.update(f"{rr['row_tail_idx']}|{role_key}|{off}|{action}\n".encode())
    labels[~select] = -100
    # Do not allow padding/non-attended positions to become active even if a bug
    # slipped through group construction.
    labels[~attention_mask.bool()] = -100
    return masked_inputs, labels, {
        "selected_tokens": int((labels != -100).sum().item()),
        "replacement_action_counts": dict(action_counts),
        "replacement_action_digest": replacement_digest.hexdigest(),
    }


def apply_paired_pvdm_masking(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    label_records: list[dict[str, Any]],
    tokenizer,
    *,
    arm: str,
    seed: int = 43023,
    mask_prob: float = 0.15,
    target_prob: float = 0.60,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Materialize MLM inputs/labels for one arm of the paired PVDM contrast."""
    if input_ids.device.type != "cpu" or attention_mask.device.type != "cpu" or word_group.device.type != "cpu":
        # CPU materialization keeps deterministic hashing simple and avoids many
        # tiny Python writes into CUDA tensors.  The caller moves outputs to GPU.
        input_ids = input_ids.cpu(); attention_mask = attention_mask.cpu(); word_group = word_group.cpu()
    row_records, pair_stats = make_paired_group_selections(
        word_group, attention_mask, label_records,
        seed=seed, mask_prob=mask_prob, target_prob=target_prob, same_length_required=True,
    )
    masked_inputs, labels, arm_stats = _materialize_one_arm(
        input_ids, attention_mask, row_records,
        arm=arm, tokenizer_len=len(tokenizer), mask_token_id=int(tokenizer.mask_token_id), seed=seed,
    )
    pair_stats = dict(pair_stats)
    # Convert Counters for JSON friendliness where needed.
    for k in ["label_event_categories", "usable_event_categories", "selected_event_categories", "reject_reasons"]:
        if isinstance(pair_stats.get(k), Counter):
            pair_stats[k] = dict(pair_stats[k])
    pair_stats.update({"arm": arm, **arm_stats})
    return masked_inputs, labels, pair_stats


def standard_wwm_exact_count_masking(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    tokenizer,
    *,
    seed: int = 43023,
    mask_prob: float = 0.15,
    row_tail_indices: Iterable[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Hash-based WWM control used only for optional machinery checks.

    The actual K=0 parity smoke in research can instead call the legacy masking
    function.  This function is included for debugging the deterministic 80/10/10
    materialization path without any PVDM labels.
    """
    if input_ids.device.type != "cpu":
        input_ids = input_ids.cpu(); attention_mask = attention_mask.cpu(); word_group = word_group.cpu()
    if row_tail_indices is None:
        row_tail_indices = range(int(input_ids.shape[0]))
    masked_inputs = input_ids.clone()
    labels = input_ids.clone()
    select = torch.zeros_like(input_ids, dtype=torch.bool)
    action_counts = Counter()
    rows = 0; groups_sum = 0; token_sum = 0
    for b, row_tail_idx in enumerate(list(row_tail_indices)):
        group_pos = group_positions_for_row(word_group[b], attention_mask[b])
        valid = sorted(group_pos)
        k = _desired_group_count(seed, int(row_tail_idx), len(valid), mask_prob)
        valid.sort(key=lambda g: hash_uniform(seed, "standard_bg", int(row_tail_idx), g))
        chosen = set(valid[:k])
        rows += 1; groups_sum += len(chosen)
        for g in chosen:
            for off, pos in enumerate(group_pos[g]):
                select[b, pos] = True
                token_sum += 1
                u = hash_uniform(seed, "standard_replace", int(row_tail_idx), g, off)
                if u < 0.8:
                    masked_inputs[b, pos] = int(tokenizer.mask_token_id); action_counts["mask_token"] += 1
                elif u < 0.9:
                    rid = hash_int(seed, len(tokenizer), "standard_rand_id", int(row_tail_idx), g, off)
                    masked_inputs[b, pos] = int(rid); action_counts["random_token"] += 1
                else:
                    action_counts["keep_original"] += 1
    labels[~select] = -100
    labels[~attention_mask.bool()] = -100
    return masked_inputs, labels, {"arm": "standard_exact", "rows": rows, "selected_groups": groups_sum, "selected_tokens": token_sum, "replacement_action_counts": dict(action_counts)}


def aggregate_stat_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    counter_keys = {"label_event_categories", "usable_event_categories", "selected_event_categories", "reject_reasons", "replacement_action_counts"}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = out.get(k, 0) + v
            elif k in counter_keys and isinstance(v, dict):
                c = Counter(out.get(k, {})); c.update({str(kk): int(vv) for kk, vv in v.items()}); out[k] = dict(c)
            elif k == "pivot_visible_at_selected_target" and isinstance(v, dict):
                cur = out.get(k, {"treatment_visible": 0, "treatment_total": 0, "control_visible": 0, "control_total": 0})
                for kk, vv in v.items(): cur[kk] = cur.get(kk, 0) + int(vv)
                out[k] = cur
    return out
