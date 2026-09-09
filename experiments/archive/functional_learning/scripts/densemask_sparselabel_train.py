#!/usr/bin/env python3
"""research: dense-mask / sparse-label unchanged-Qwen training control.

Scientific purpose
------------------
Dense unchanged-Qwen focus changed two things at once relative to sparse focus:
(1) many more second-view content groups were masked, reducing local completion clues;
(2) many more of those groups were supervised as focus targets.  If the practical
signal survives official evaluation and seed replication, the clean causal training
contrast is therefore not another evaluation interaction but a training arm that keeps
the dense effective input geometry while withholding the extra dense target coverage.

This wrapper reuses the trusted research trainer but monkey-patches the Qwen focus row
constructor for focus-like objectives:

  - labels: selected with the same sparse policy as research sparse focus
            (`--focus-prob`, `--max-focus-groups-per-row`, deterministic row seed);
  - masks:  all detected Qwen second-view content groups, optionally capped by
            QIUSHI_DENSEMASK_MAX_MASK_GROUPS (default 128) and always including the
            sparse labelled groups.

Run only as a focused causal control after the frontier-validation jobs have been read;
do not treat this as a substitute for official Overall or seed replication evidence.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import atexit
import hashlib
import json
import pathlib
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List

import torch

SCRIPT_PATH = _public_path('experiments/archive/functional_learning/scripts/densemask_sparselabel_train.py')
ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_stream_train_weighted as base  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402

PATCH_COUNTER: Counter = Counter()
PATCH_KIND_COUNTER: Counter = Counter()
PAIR_IDS = set()


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def find_out_dir_from_argv() -> pathlib.Path:
    default = _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train')
    argv = list(sys.argv)
    for i, tok in enumerate(argv):
        if tok == "--out-dir" and i + 1 < len(argv):
            return pathlib.Path(argv[i + 1])
        if tok.startswith("--out-dir="):
            return pathlib.Path(tok.split("=", 1)[1])
    return default


def densemask_patch_stats() -> Dict[str, Any]:
    d = dict(PATCH_COUNTER)
    label_tokens = int(d.get("label_target_tokens", 0))
    mask_tokens = int(d.get("mask_target_tokens", 0))
    return {
        "status": "DENSEMASK_SPARSELABEL_PATCH_STATS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": rel(SCRIPT_PATH),
        "patched_base_script": rel(_public_path('experiments/archive/functional_learning/scripts/real_stream_train_weighted.py')),
        "policy": {
            "labels": "research sparse selection governed by --focus-prob and --max-focus-groups-per-row",
            "masks": "all detected Qwen second-view content groups, capped by QIUSHI_DENSEMASK_MAX_MASK_GROUPS and including all labelled groups",
        },
        "counter": d,
        "selected_candidate_kind_counts": dict(PATCH_KIND_COUNTER),
        "label_to_mask_token_ratio": float(label_tokens / mask_tokens) if mask_tokens else None,
        "selected_pair_refs_total": len(PAIR_IDS),
        "interpretation": "This records input-side dense masking with sparse supervision. It is a future causal training control, not a frontier-validation result.",
    }


def write_patch_stats() -> None:
    try:
        out_dir = find_out_dir_from_argv()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "densemask_sparse_label_patch_stats.json").write_text(
            json.dumps(densemask_patch_stats(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception as exc:  # pragma: no cover - only preserves diagnostics
        print(json.dumps({"event": "densemask_patch_stats_write_failed", "error": repr(exc)}), flush=True)


def apply_densemask_sparse_label_row(row: Dict[str, Any], tok: Dict[str, Any], tokenizer, seed: int,
                                     focus_prob: float, max_focus_groups_per_row: int):
    input_ids: torch.Tensor = tok["input_ids"]
    offsets: torch.Tensor = tok["offsets"]
    attention_mask: torch.Tensor = tok["attention_mask"]
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    all_groups: List[Dict[str, Any]] = []
    text = str(row.get("text", ""))
    for seg_i, seg in enumerate(row.get("qwen_pair_segments") or []):
        view_text = str(seg.get("view_text", ""))
        vstart = int(seg.get("view_start", -1))
        vend = int(seg.get("view_end", -1))
        if vstart < 0 or vend <= vstart or vend > len(text):
            continue
        if text[vstart:vend] != view_text:
            continue
        for a, b, word in base.content_word_spans(view_text):
            pos = base.locate_positions(offsets, vstart + a, vstart + b)
            pos = [p for p in pos if int(attention_mask[p]) == 1]
            if pos:
                all_groups.append({
                    "segment_index": seg_i,
                    "pair_id": seg.get("pair_id"),
                    "view_kind": seg.get("view_kind"),
                    "candidate_kind": seg.get("candidate_kind"),
                    "word": word,
                    "positions": pos,
                })
    rng = random.Random(int(seed))
    label_candidates = list(all_groups)
    if len(label_candidates) > int(max_focus_groups_per_row):
        label_candidates = rng.sample(label_candidates, int(max_focus_groups_per_row))
    label_groups = [g for g in label_candidates if rng.random() < float(focus_prob)]
    if not label_groups and label_candidates:
        label_groups = [rng.choice(label_candidates)]

    max_mask_groups = int(base.os.environ.get("QIUSHI_DENSEMASK_MAX_MASK_GROUPS", "128"))
    mask_groups = list(all_groups)
    if len(mask_groups) > max_mask_groups:
        # Prefer deterministic sampling, then force all labelled groups back in.
        mrng = random.Random(stable_seed("densemask-mask-cap", seed, bridge.row_key(row), max_mask_groups))
        sampled = mrng.sample(mask_groups, max_mask_groups)
        for g in label_groups:
            if g not in sampled:
                sampled.append(g)
        mask_groups = sampled
    label_pos = set()
    mask_pos = set()
    for g in mask_groups:
        for p in g["positions"]:
            if int(attention_mask[p]) == 1:
                masked[p] = int(tokenizer.mask_token_id)
                mask_pos.add(int(p))
    kind_counts = Counter()
    for g in label_groups:
        kind_counts[str(g.get("candidate_kind", "unknown"))] += 1
        if g.get("pair_id"):
            PAIR_IDS.add(str(g["pair_id"]))
        for p in g["positions"]:
            if int(attention_mask[p]) == 1:
                labels[p] = input_ids[p]
                masked[p] = int(tokenizer.mask_token_id)
                mask_pos.add(int(p))
                label_pos.add(int(p))
    PATCH_COUNTER.update({
        "qwen_focus_rows_seen": 1,
        "candidate_groups": len(all_groups),
        "label_candidate_groups_after_cap": len(label_candidates),
        "label_selected_groups": len(label_groups),
        "mask_selected_groups": len(mask_groups),
        "label_target_tokens": len(label_pos),
        "mask_target_tokens": len(mask_pos),
        "zero_label_rows": int(len(label_pos) == 0),
        "rows_with_mask_but_no_label": int(len(mask_pos) > 0 and len(label_pos) == 0),
    })
    PATCH_KIND_COUNTER.update(kind_counts)
    return masked, labels, {
        "n_candidate_groups": len(all_groups),
        "n_selected_groups": len(label_groups),
        "n_target_tokens": len(label_pos),
        "zero_label_row": len(label_pos) == 0,
        "selected_candidate_kind_counts": dict(kind_counts),
        "selected_pair_ids": sorted({str(g["pair_id"]) for g in label_groups if g.get("pair_id")}),
        "densemask_control": True,
        "n_masked_groups": len(mask_groups),
        "n_masked_tokens": len(mask_pos),
    }


base.apply_view_focus_row = apply_densemask_sparse_label_row
atexit.register(write_patch_stats)


if __name__ == "__main__":
    # Reuse the original research CLI exactly.  Extra parameters for the mask cap are
    # provided through QIUSHI_DENSEMASK_MAX_MASK_GROUPS so future training commands
    # remain compatible with existing pipeline wrappers.
    base.main()
