#!/usr/bin/env python3
"""Exact-swap conditional-innovation whole-word masking core for research.

This main-workspace port preserves the exact-swap construction while keeping
it separate from the already-running research probability reallocation.

Mechanism:
  1. Reproduce the baseline 15% Bernoulli WWM selection.
  2. For each changed row, propose at most one source-absent rewrite innovation
     group using stable hashes; proposal ordering consumes no torch RNG.
  3. If the proposed target was not already selected, replace one already-selected
     ordinary non-pair donor group of the same token length elsewhere in the batch.
  4. Keep selected group count and selected token count exactly unchanged per batch.

No strings or categories are model inputs.  Metadata only changes the boolean WWM
selection before the unchanged token-mean MLM objective and inherited tokenwise
80/10/10 corruption.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Iterable

import torch


@dataclasses.dataclass
class SelectionAudit:
    baseline_select: torch.Tensor
    biased_select: torch.Tensor
    events: list[dict[str, Any]]
    counters: dict[str, int]


def stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def group_positions(groups: torch.Tensor, candidate: torch.Tensor) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    for gid_t in torch.unique(groups[(groups >= 0) & candidate]):
        gid = int(gid_t.item())
        pos = tuple(int(x) for x in ((groups == gid) & candidate).nonzero(as_tuple=False).flatten().tolist())
        if pos:
            out[gid] = pos
    return out


def count_selected_groups(select: torch.Tensor, word_group: torch.Tensor, candidate: torch.Tensor) -> int:
    total = 0
    for b in range(select.shape[0]):
        for positions in group_positions(word_group[b], candidate[b]).values():
            if bool(select[b, list(positions)].all()):
                total += 1
    return total


def baseline_wwm_selection(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    special_ids: Iterable[int],
    mask_prob: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the fixed-WWM selection used by the inherited trainer."""
    device = input_ids.device
    special = torch.tensor(sorted(int(x) for x in special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special)
    select = torch.zeros_like(candidate)
    for b in range(input_ids.shape[0]):
        groups = word_group[b]
        valid_groups = torch.unique(groups[groups >= 0])
        if valid_groups.numel() == 0:
            continue
        gp = torch.rand(valid_groups.numel(), generator=generator, device=device)
        chosen = valid_groups[gp < float(mask_prob)]
        if chosen.numel():
            select[b] = torch.isin(groups, chosen) & candidate[b]
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel():
            select.view(-1)[flat[0, 0]] = True
    return select, candidate


def validate_target(target: dict[str, Any], positions_by_gid: dict[int, tuple[int, ...]]) -> tuple[bool, str]:
    gid = int(target["group_id"])
    expected = tuple(int(x) for x in target.get("positions") or [])
    actual = positions_by_gid.get(gid)
    if actual is None:
        return False, "target_group_not_visible"
    if expected != actual:
        return False, "target_group_position_mismatch"
    if int(target.get("token_count", -1)) != len(actual):
        return False, "target_token_count_mismatch"
    if target.get("category") != "innovation":
        return False, "target_not_innovation"
    if not bool(target.get("strict_full_group")):
        return False, "target_not_strict_full_group"
    return True, "ok"


def metadata_sets(meta: dict[str, Any], valid_targets: list[dict[str, Any]]) -> dict[str, set[int]]:
    innovation_gids = {int(t["group_id"]) for t in valid_targets}
    copyable_gids = {int(x) for x in meta.get("copyable_group_ids") or []}
    protected_source_gids = {int(x) for x in meta.get("protected_source_group_ids") or []}
    protected_pair_gids = {int(x) for x in meta.get("protected_pair_group_ids") or []}
    if not protected_pair_gids:
        protected_pair_gids.update(protected_source_gids)
        protected_pair_gids.update(copyable_gids)
        protected_pair_gids.update(innovation_gids)
        for pair in meta.get("pairs") or []:
            for key in ("source_group_ids", "partial_source_group_ids_excluded", "partial_rewrite_group_ids_excluded"):
                protected_pair_gids.update(int(x) for x in pair.get(key) or [])
    return {
        "innovation": innovation_gids,
        "copyable": copyable_gids,
        "protected_source": protected_source_gids,
        "protected_pair": protected_pair_gids,
    }


def innovation_biased_selection(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    special_ids: Iterable[int],
    mask_prob: float,
    generator: torch.Generator,
    row_metadata: list[dict[str, Any] | None],
    priority_salt: str,
    max_swaps_per_changed_row: int = 1,
) -> SelectionAudit:
    """Apply exact same-length WWM swaps on top of baseline WWM selection."""
    if len(row_metadata) != input_ids.shape[0]:
        raise ValueError("row_metadata length must equal batch size")
    baseline, candidate = baseline_wwm_selection(
        input_ids, attention_mask, word_group, special_ids, mask_prob, generator
    )
    biased = baseline.clone()
    events: list[dict[str, Any]] = []
    counters: dict[str, int] = {}

    def inc(key: str, n: int = 1) -> None:
        counters[key] = counters.get(key, 0) + int(n)

    row_info: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for b, meta in enumerate(row_metadata):
        pos_by_gid = group_positions(word_group[b], candidate[b])
        info: dict[str, Any] = {
            "positions": pos_by_gid,
            "meta": meta,
            "protected_pair": set(),
            "protected_source": set(),
            "copyable": set(),
            "innovation": set(),
        }
        row_info.append(info)
        if not meta or max_swaps_per_changed_row <= 0:
            continue
        inc("changed_row_exposures")
        raw_targets = list(meta.get("innovation_groups") or [])
        valid_targets: list[dict[str, Any]] = []
        for target in raw_targets:
            ok, reason = validate_target(target, pos_by_gid)
            if ok:
                valid_targets.append(target)
            else:
                inc(reason)
        if not valid_targets:
            inc("changed_rows_no_valid_innovation")
            continue
        inc("eligible_changed_rows")
        sets = metadata_sets(meta, valid_targets)
        info.update({
            "protected_pair": sets["protected_pair"],
            "protected_source": sets["protected_source"],
            "copyable": sets["copyable"],
            "innovation": sets["innovation"],
        })
        for target in valid_targets:
            gid = int(target["group_id"])
            positions = pos_by_gid[gid]
            inc("eligible_innovation_groups")
            inc("baseline_selected_innovation_groups", int(bool(baseline[b, list(positions)].all())))

        eid = int(meta["example_id"])
        proposals = sorted(
            valid_targets,
            key=lambda t: stable_u64("target", priority_salt, eid, int(t["group_id"])),
        )[: int(max_swaps_per_changed_row)]
        pair_source = {
            str(p.get("pair_id", "")): {int(x) for x in p.get("source_group_ids") or []}
            for p in meta.get("pairs") or []
        }
        for target in proposals:
            inc("proposals")
            target_gid = int(target["group_id"])
            target_pos = pos_by_gid[target_gid]
            if bool(baseline[b, list(target_pos)].all()):
                inc("proposal_baseline_collision")
                events.append({
                    "batch_row": b,
                    "example_id": eid,
                    "outcome": "baseline_collision",
                    "target_group_id": target_gid,
                    "target_token_count": len(target_pos),
                    "target_pair_id": str(target.get("pair_id", "")),
                    "target_cue_class": str(target.get("cue_class", "")),
                    "target_category": str(target.get("category", "")),
                })
                continue
            pending.append({
                "batch_row": b,
                "example_id": eid,
                "target": target,
                "target_gid": target_gid,
                "target_pos": target_pos,
                "pair_source": pair_source,
            })

    pending.sort(key=lambda x: stable_u64("pending", priority_salt, x["example_id"], x["target_gid"]))
    invariant_groups = count_selected_groups(baseline, word_group, candidate)
    invariant_tokens = int(baseline.sum().item())
    for item in pending:
        b = int(item["batch_row"])
        eid = int(item["example_id"])
        target = item["target"]
        target_gid = int(item["target_gid"])
        target_pos = item["target_pos"]
        donors: list[tuple[int, int]] = []
        for donor_b, donor_info in enumerate(row_info):
            for donor_gid, donor_pos in donor_info["positions"].items():
                if donor_gid in donor_info["protected_pair"]:
                    continue
                if len(donor_pos) != len(target_pos):
                    continue
                if bool(biased[donor_b, list(donor_pos)].all()):
                    donors.append((donor_b, donor_gid))
        if not donors:
            inc("proposal_no_equal_length_donor")
            events.append({
                "batch_row": b,
                "example_id": eid,
                "outcome": "no_equal_length_donor",
                "target_group_id": target_gid,
                "target_token_count": len(target_pos),
                "target_pair_id": str(target.get("pair_id", "")),
                "target_cue_class": str(target.get("cue_class", "")),
                "target_category": str(target.get("category", "")),
            })
            continue
        donor_b, donor_gid = min(
            donors,
            key=lambda bg: (
                row_metadata[bg[0]] is not None,
                stable_u64("donor", priority_salt, eid, target_gid, bg[0], bg[1]),
            ),
        )
        donor_info = row_info[donor_b]
        donor_pos = donor_info["positions"][donor_gid]
        before_groups = invariant_groups
        before_tokens = invariant_tokens
        biased[donor_b, list(donor_pos)] = False
        biased[b, list(target_pos)] = True
        if bool(biased[donor_b, list(donor_pos)].any()) or not bool(biased[b, list(target_pos)].all()):
            raise AssertionError("swap did not flip whole donor/target groups")
        # Same-token-length whole-group replacement preserves these invariants by
        # construction.  The final batch-level checks below recompute both totals.
        after_groups = invariant_groups
        after_tokens = invariant_tokens

        source_gids = item["pair_source"].get(str(target.get("pair_id", "")), set())
        target_positions_by_gid = row_info[b]["positions"]
        visible_source = [
            g for g in source_gids
            if g in target_positions_by_gid and not bool(biased[b, list(target_positions_by_gid[g])].all())
        ]
        fully_visible_source = bool(source_gids) and all(
            g not in target_positions_by_gid or not bool(biased[b, list(target_positions_by_gid[g])].any())
            for g in source_gids
        )
        inc("successful_swaps")
        inc("forced_relation_cue", target.get("cue_class") == "relation_cue")
        inc("forced_copyable", target.get("category") == "copyable")
        inc("donor_copyable_rewrite", donor_gid in donor_info["copyable"])
        inc("donor_protected_source", donor_gid in donor_info["protected_source"])
        inc("donor_protected_pair", donor_gid in donor_info["protected_pair"])
        inc("donor_ordinary_row", row_metadata[donor_b] is None)
        inc("donor_changed_nonpair_row", row_metadata[donor_b] is not None)
        inc("forced_target_has_visible_source_group", bool(visible_source))
        inc("forced_target_has_fully_visible_source", fully_visible_source)
        events.append({
            "batch_row": b,
            "example_id": eid,
            "outcome": "swapped",
            "target_group_id": target_gid,
            "target_token_count": len(target_pos),
            "target_positions": list(target_pos),
            "target_pair_id": str(target.get("pair_id", "")),
            "target_cue_class": str(target.get("cue_class", "")),
            "target_category": str(target.get("category", "")),
            "donor_batch_row": donor_b,
            "donor_group_id": donor_gid,
            "donor_positions": list(donor_pos),
            "donor_from_ordinary_row": row_metadata[donor_b] is None,
            "donor_copyable_rewrite": donor_gid in donor_info["copyable"],
            "donor_protected_source": donor_gid in donor_info["protected_source"],
            "donor_protected_pair": donor_gid in donor_info["protected_pair"],
            "paired_source_group_count": len(source_gids),
            "paired_source_visible_group_count": len(visible_source),
            "paired_source_fully_visible": fully_visible_source,
            "selected_groups_before_after": [before_groups, after_groups],
            "selected_tokens_before_after": [before_tokens, after_tokens],
        })

    if int(baseline.sum().item()) != int(biased.sum().item()):
        raise AssertionError("batch selected-token mass changed")
    final_groups = count_selected_groups(biased, word_group, candidate)
    if final_groups != invariant_groups:
        raise AssertionError("batch selected-group mass changed")
    return SelectionAudit(baseline, biased, events, counters)


def apply_inherited_tokenwise_corruption(
    input_ids: torch.Tensor,
    select: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inherited independent-token 80/10/10 corruption for selected labels."""
    labels = input_ids.clone()
    labels[~select] = -100
    masked = input_ids.clone()
    r = torch.rand(input_ids.shape, generator=generator, device=input_ids.device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked[mask_tok] = int(mask_token_id)
    if rand_tok.any():
        random_ids = torch.randint(
            0, int(vocab_size), (int(rand_tok.sum().item()),), generator=generator, device=input_ids.device
        )
        masked[rand_tok] = random_ids
    return masked, labels, r
