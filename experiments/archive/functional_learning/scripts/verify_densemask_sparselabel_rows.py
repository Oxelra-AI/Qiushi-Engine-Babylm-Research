#!/usr/bin/env python3
"""research row-level invariant check for dense-mask/sparse-label control.

This does not train. It verifies on sampled Qwen rows that the proposed control has
exactly the sparse labels and exactly the dense masks (up to labelled positions
being included, already true in dense) relative to the trusted research sparse and
dense row constructors under the same tokenizer, row seed, and selection policies.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys
import time
from collections import Counter
from typing import Any

import torch

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_stream_train_weighted as s64  # noqa: E402
import densemask_sparselabel_train as dm  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402

OUT_DIR = _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_row_verify')
TAIL = _public_path('experiments/archive/functional_learning/data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def positions(t: torch.Tensor, what: str, mask_id: int | None = None) -> set[int]:
    if what == "labels":
        return set(int(i) for i in torch.nonzero(t != -100, as_tuple=False).flatten().tolist())
    if what == "mask":
        assert mask_id is not None
        return set(int(i) for i in torch.nonzero(t == int(mask_id), as_tuple=False).flatten().tolist())
    raise ValueError(what)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, prefix = s64.load_prefix(TAIL, max_updates=1, words_per_update=39533)
    qwen_rows = [r for r in rows if r.get("source") == "qwen_pair_packed" and r.get("qwen_pair_segments")]
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = bridge.WordGroupBuilder(tokenizer)
    max_rows = min(33, len(qwen_rows))
    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    aggregate = Counter()
    for idx, row in enumerate(qwen_rows[:max_rows]):
        tok = bridge.tokenize_row(row, tokenizer, 512, wgb)
        row_key = bridge.row_key(row)
        # Sparse row constructor: labels and masks identical sparse positions.
        seed_sparse = s64.stable_seed("view-focus", 62064, row_key)
        sparse_masked, sparse_labels, sparse_stats = s64.apply_view_focus_row(row, tok, tokenizer, seed_sparse, 0.35, 16)
        # Dense row constructor: labels and masks all/capped content positions.
        seed_dense = s64.stable_seed("view-focus", 62064, row_key)
        dense_masked, dense_labels, dense_stats = s64.apply_view_focus_row(row, tok, tokenizer, seed_dense, 1.0, 128)
        # Dense-mask/sparse-label control.
        seed_dm = s64.stable_seed("view-focus", 62064, row_key)
        dm_masked, dm_labels, dm_stats = dm.apply_densemask_sparse_label_row(row, tok, tokenizer, seed_dm, 0.35, 16)
        sp_label_pos = positions(sparse_labels, "labels")
        sp_mask_pos = positions(sparse_masked, "mask", tokenizer.mask_token_id)
        de_label_pos = positions(dense_labels, "labels")
        de_mask_pos = positions(dense_masked, "mask", tokenizer.mask_token_id)
        dm_label_pos = positions(dm_labels, "labels")
        dm_mask_pos = positions(dm_masked, "mask", tokenizer.mask_token_id)
        ok_sparse_labels = dm_label_pos == sp_label_pos
        ok_sparse_label_values = bool(torch.equal(dm_labels, sparse_labels))
        ok_dense_masks = dm_mask_pos == de_mask_pos
        labels_subset_masks = dm_label_pos.issubset(dm_mask_pos)
        sparse_labels_subset_dense = sp_label_pos.issubset(de_mask_pos)
        row_summary = {
            "idx": idx,
            "row_key": row_key,
            "sparse_label_tokens": len(sp_label_pos),
            "dense_mask_tokens": len(de_mask_pos),
            "dm_label_tokens": len(dm_label_pos),
            "dm_mask_tokens": len(dm_mask_pos),
            "sparse_groups": sparse_stats.get("n_selected_groups"),
            "dense_groups": dense_stats.get("n_selected_groups"),
            "dm_label_groups": dm_stats.get("n_selected_groups"),
            "dm_mask_groups": dm_stats.get("n_masked_groups"),
            "ok_sparse_labels": ok_sparse_labels,
            "ok_sparse_label_values": ok_sparse_label_values,
            "ok_dense_masks": ok_dense_masks,
            "labels_subset_masks": labels_subset_masks,
            "sparse_labels_subset_dense": sparse_labels_subset_dense,
        }
        summaries.append(row_summary)
        aggregate.update({
            "rows_checked": 1,
            "sparse_label_tokens": len(sp_label_pos),
            "dense_mask_tokens": len(de_mask_pos),
            "dm_label_tokens": len(dm_label_pos),
            "dm_mask_tokens": len(dm_mask_pos),
            "sparse_groups": int(sparse_stats.get("n_selected_groups", 0)),
            "dense_groups": int(dense_stats.get("n_selected_groups", 0)),
            "dm_label_groups": int(dm_stats.get("n_selected_groups", 0)),
            "dm_mask_groups": int(dm_stats.get("n_masked_groups", 0)),
        })
        if not (ok_sparse_labels and ok_sparse_label_values and ok_dense_masks and labels_subset_masks and sparse_labels_subset_dense):
            failures.append(row_summary)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": rel(__file__),
        "tail": rel(TAIL),
        "prefix_info": prefix,
        "rows_checked": max_rows,
        "aggregate": dict(aggregate),
        "label_to_mask_token_ratio": float(aggregate["dm_label_tokens"] / aggregate["dm_mask_tokens"]) if aggregate["dm_mask_tokens"] else None,
        "invariants": {
            "densemask_labels_equal_sparse_labels": not any(not r["ok_sparse_labels"] for r in summaries),
            "densemask_label_values_equal_sparse_labels": not any(not r["ok_sparse_label_values"] for r in summaries),
            "densemask_masks_equal_dense_masks": not any(not r["ok_dense_masks"] for r in summaries),
            "densemask_labels_subset_masks": not any(not r["labels_subset_masks"] for r in summaries),
            "sparse_labels_subset_dense_masks": not any(not r["sparse_labels_subset_dense"] for r in summaries),
        },
        "failures": failures,
        "row_summaries": summaries[:10],
        "interpretation": "If PASS, the first-macro Qwen rows implement the intended causal contrast: sparse target supervision with dense input-side clue removal. This does not establish training outcome.",
    }
    out = _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_row_verify/row_verify.json')
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = _public_path('research/documents/functional_learning/data/densemask_sparselabel_row_verify/row_verify.md')
    md.write_text(
        "# research dense-mask/sparse-label row verification\n\n"
        f"Status: `{report['status']}`\n\n"
        f"Rows checked: `{max_rows}`\n\n"
        f"Aggregate label/mask tokens: `{aggregate['dm_label_tokens']}` / `{aggregate['dm_mask_tokens']}`; ratio `{report['label_to_mask_token_ratio']}`\n\n"
        f"Invariants: `{report['invariants']}`\n\n"
        f"Report: `{rel(out)}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "rows_checked": max_rows, "label_to_mask_token_ratio": report["label_to_mask_token_ratio"], "out": rel(out)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
