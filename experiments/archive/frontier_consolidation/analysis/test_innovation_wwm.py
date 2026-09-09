from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

import pytest
import torch
from transformers import AutoTokenizer


HERE = _public_path('experiments/archive/frontier_consolidation/analysis/work')
sys.path.insert(0, str(HERE))

import build_innovation_metadata as builder
import innovation_biased_wwm as ibw


def test_discontiguous_ranges_exclude_partial_groups() -> None:
    group_at = [0, 0, 1, 2, 2]
    groups = {0: (0, 1), 1: (2,), 2: (3, 4)}
    strict, partial = builder.strict_groups_in_positions(group_at, groups, {0, 2, 3, 4})
    assert strict == [1, 2]
    assert partial == [0]


def minimal_meta(target_gid: int, target_positions: list[int], protected: list[int] | None = None):
    return {
        "example_id": 950000,
        "innovation_groups": [{
            "group_id": target_gid,
            "positions": target_positions,
            "token_count": len(target_positions),
            "category": "innovation",
            "cue_class": "content_or_other",
            "strict_full_group": True,
            "pair_id": "p0",
        }],
        "copyable_group_ids": [],
        "protected_source_group_ids": protected or [],
        "pairs": [{"pair_id": "p0", "source_group_ids": protected or []}],
    }


def find_seed_for_pattern(ids, attn, groups, target_gid: int, donor_gid: int) -> int:
    for seed in range(10000):
        g = torch.Generator().manual_seed(seed)
        selected, candidate = ibw.baseline_wwm_selection(ids, attn, groups, [], 0.5, g)
        pos = ibw.group_positions(groups[0], candidate[0])
        if not bool(selected[0, list(pos[target_gid])].any()) and bool(selected[0, list(pos[donor_gid])].all()):
            return seed
    raise AssertionError("no useful deterministic seed found")


def test_same_length_swap_exact_mass_and_rng_invariance() -> None:
    ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    attn = torch.ones_like(ids)
    groups = torch.tensor([[0, 1, 1, 2, 2, 3]])
    seed = find_seed_for_pattern(ids, attn, groups, target_gid=1, donor_gid=2)
    g1, g2 = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
    audit = ibw.innovation_biased_selection(
        ids, attn, groups, [], 0.5, g1, [minimal_meta(1, [1, 2])], "unit", 1
    )
    baseline, _ = ibw.baseline_wwm_selection(ids, attn, groups, [], 0.5, g2)
    assert torch.equal(audit.baseline_select, baseline)
    assert torch.equal(g1.get_state(), g2.get_state())
    assert audit.counters["successful_swaps"] == 1
    assert int(audit.baseline_select.sum()) == int(audit.biased_select.sum())
    assert bool(audit.biased_select[0, 1:3].all())
    assert not bool(audit.biased_select[0, 3:5].any())


def test_protected_source_cannot_be_donor() -> None:
    ids = torch.tensor([[10, 11, 12, 13]])
    attn = torch.ones_like(ids)
    groups = torch.tensor([[0, 0, 1, 1]])
    seed = find_seed_for_pattern(ids, attn, groups, target_gid=0, donor_gid=1)
    audit = ibw.innovation_biased_selection(
        ids, attn, groups, [], 0.5, torch.Generator().manual_seed(seed),
        [minimal_meta(0, [0, 1], protected=[1])], "unit", 1,
    )
    assert audit.counters.get("successful_swaps", 0) == 0
    assert audit.counters["proposal_no_equal_length_donor"] == 1
    assert torch.equal(audit.baseline_select, audit.biased_select)


def test_truncated_or_partial_target_is_rejected() -> None:
    ids = torch.tensor([[10, 11, 12]])
    attn = torch.ones_like(ids)
    groups = torch.tensor([[0, 1, 1]])
    meta = minimal_meta(1, [1])  # Metadata must name both pieces [1,2].
    audit = ibw.innovation_biased_selection(
        ids, attn, groups, [], 0.15, torch.Generator().manual_seed(3), [meta], "unit", 1
    )
    assert audit.counters["target_group_position_mismatch"] == 1
    assert audit.counters["changed_rows_no_valid_innovation"] == 1


def test_repeated_normalized_word_is_copyable_not_innovation() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(builder.DEFAULT_TOKENIZER), use_fast=True)
    text = "cat cat dog cat bird"
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    source_positions = [i for i, (a, b) in enumerate(offsets) if b <= 11]
    rewrite_positions = [i for i, (a, b) in enumerate(offsets) if a >= 11]
    assert source_positions and rewrite_positions
    span = {
        "token_len_truncated": len(enc["input_ids"]),
        "token_len_full": len(enc["input_ids"]),
        "truncated_by_seq256": False,
        "pair_count": 1,
        "pairs": [{
            "pair_id": "p0", "visibility": "both_visible",
            "source_token_ranges": [[min(source_positions), max(source_positions) + 1]],
            "rewrite_token_ranges": [[min(rewrite_positions), max(rewrite_positions) + 1]],
        }],
    }
    row = {"text": text, "words": 5, "source": builder.CHANGED_SOURCE, "first_global_row_1based": 1}
    rec, _ = builder.build_row_metadata(1, row, span, tokenizer)
    copyable = [x for p in rec["pairs"] for x in p["copyable_groups"]]
    innovation = rec["innovation_groups"]
    cats = {x["normalized_text"]: x for x in copyable + innovation}
    assert cats["cat"]["category"] == "copyable"
    assert cats["cat"]["source_normalized_occurrences"] == 2
    assert cats["bird"]["category"] == "innovation"
    assert rec["audit"]["decoy_used_for_classification"] is False


def test_source_rewrite_overlap_is_rejected() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(builder.DEFAULT_TOKENIZER), use_fast=True)
    text = "cat dog bird"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    span = {
        "token_len_truncated": len(ids), "token_len_full": len(ids), "truncated_by_seq256": False,
        "pair_count": 1,
        "pairs": [{"pair_id": "bad", "visibility": "both_visible", "source_token_ranges": [[0, 2]], "rewrite_token_ranges": [[1, len(ids)]]}],
    }
    row = {"text": text, "words": 3, "source": builder.CHANGED_SOURCE, "first_global_row_1based": 1}
    with pytest.raises(RuntimeError, match="source/rewrite overlap"):
        builder.build_row_metadata(1, row, span, tokenizer)
