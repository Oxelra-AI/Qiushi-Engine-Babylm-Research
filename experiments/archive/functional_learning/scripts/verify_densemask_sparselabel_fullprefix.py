#!/usr/bin/env python3
"""research full-prefix invariant check for dense-mask / sparse-label control.

The research dense-mask/sparse-label script monkeypatches
`real_stream_train_weighted.apply_view_focus_row` at import time.  A
verifier that imports research before saving research's original constructor can
accidentally compare the control to itself.  This script preserves the original
research row constructor first, then imports the dense-mask control and compares
on the same Qwen rows, tokenizer, row seeds, and policy parameters used by the
80-update dense experiments.

It verifies and quantifies:
  * control labels vs original sparse labels: focus_prob=0.35, cap=16;
  * control masks vs original dense masks: focus_prob=1.0, cap=128;
  * whether sparse labels are already included in original dense masks;
  * if not, how many extra mask tokens/groups the control introduces in order to
    keep MLM labels masked.

The scientific purpose is to decide whether the next causal training arm is a
clean exact factorial control or a quantified near-control needing repair before
GPU launch.  It performs no training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import torch

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_stream_train_weighted as s64  # noqa: E402

# Preserve the real research constructor before importing research, because research
# intentionally monkeypatches s64.apply_view_focus_row for training.
ORIGINAL_APPLY_VIEW_FOCUS_ROW = s64.apply_view_focus_row
ORIGINAL_STABLE_SEED = s64.stable_seed
ORIGINAL_LOAD_PREFIX = s64.load_prefix

import corrected_bridge_trainer as bridge  # noqa: E402
import densemask_sparselabel_train as dm  # noqa: E402

OUT_DIR = _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_fullprefix_verify')
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


def group_signature(stats: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(stats.get("n_candidate_groups", 0)),
        int(stats.get("n_selected_groups", 0)),
        int(stats.get("n_target_tokens", 0)),
    )


def summarize_examples(examples: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    return examples[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--train-seed", type=int, default=62064)
    ap.add_argument("--sparse-focus-prob", type=float, default=0.35)
    ap.add_argument("--sparse-cap", type=int, default=16)
    ap.add_argument("--dense-focus-prob", type=float, default=1.0)
    ap.add_argument("--dense-cap", type=int, default=128)
    ap.add_argument("--densemask-max-mask-groups", type=int, default=128)
    ap.add_argument("--max-rows", type=int, default=0, help="Optional row limit after filtering Qwen rows; 0 means all.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QIUSHI_DENSEMASK_MAX_MASK_GROUPS"] = str(int(args.densemask_max_mask_groups))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rows, prefix = ORIGINAL_LOAD_PREFIX(TAIL, max_updates=int(args.max_updates), words_per_update=int(args.words_per_update))
    qwen_rows = [r for r in rows if r.get("source") == "qwen_pair_packed" and r.get("qwen_pair_segments")]
    if args.max_rows and args.max_rows > 0:
        qwen_rows = qwen_rows[: int(args.max_rows)]

    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = bridge.WordGroupBuilder(tokenizer)

    failures: list[dict[str, Any]] = []
    mismatch_examples: list[dict[str, Any]] = []
    aggregate = Counter()
    hist = defaultdict(Counter)
    t0 = time.time()

    for idx, row in enumerate(qwen_rows):
        tok = bridge.tokenize_row(row, tokenizer, 512, wgb)
        row_key = bridge.row_key(row)
        seed = ORIGINAL_STABLE_SEED("view-focus", int(args.train_seed), row_key)

        sparse_masked, sparse_labels, sparse_stats = ORIGINAL_APPLY_VIEW_FOCUS_ROW(
            row, tok, tokenizer, seed, float(args.sparse_focus_prob), int(args.sparse_cap)
        )
        dense_masked, dense_labels, dense_stats = ORIGINAL_APPLY_VIEW_FOCUS_ROW(
            row, tok, tokenizer, seed, float(args.dense_focus_prob), int(args.dense_cap)
        )
        dm_masked, dm_labels, dm_stats = dm.apply_densemask_sparse_label_row(
            row, tok, tokenizer, seed, float(args.sparse_focus_prob), int(args.sparse_cap)
        )

        sp_label_pos = positions(sparse_labels, "labels")
        sp_mask_pos = positions(sparse_masked, "mask", tokenizer.mask_token_id)
        de_label_pos = positions(dense_labels, "labels")
        de_mask_pos = positions(dense_masked, "mask", tokenizer.mask_token_id)
        dm_label_pos = positions(dm_labels, "labels")
        dm_mask_pos = positions(dm_masked, "mask", tokenizer.mask_token_id)

        label_equal = dm_label_pos == sp_label_pos and bool(torch.equal(dm_labels, sparse_labels))
        sparse_mask_equal_sparse_labels = sp_mask_pos == sp_label_pos
        dense_mask_equal_dense_labels = de_mask_pos == de_label_pos
        dm_labels_subset_dm_masks = dm_label_pos.issubset(dm_mask_pos)
        sparse_labels_subset_dense_masks = sp_label_pos.issubset(de_mask_pos)
        dm_masks_equal_dense_masks = dm_mask_pos == de_mask_pos
        dm_masks_dense_plus_labels = dm_mask_pos == (de_mask_pos | sp_label_pos)

        extra_vs_dense = dm_mask_pos - de_mask_pos
        missing_vs_dense = de_mask_pos - dm_mask_pos
        labels_outside_dense = sp_label_pos - de_mask_pos
        dense_outside_dm = de_mask_pos - dm_mask_pos
        sym_mask_diff = len(extra_vs_dense) + len(missing_vs_dense)

        n_candidate = int(dense_stats.get("n_candidate_groups", sparse_stats.get("n_candidate_groups", 0)))
        row_summary = {
            "idx": idx,
            "row_key": row_key,
            "candidate_groups": n_candidate,
            "sparse_signature": group_signature(sparse_stats),
            "dense_signature": group_signature(dense_stats),
            "densemask_label_groups": int(dm_stats.get("n_selected_groups", 0)),
            "densemask_mask_groups": int(dm_stats.get("n_masked_groups", 0)),
            "sparse_label_tokens": len(sp_label_pos),
            "dense_mask_tokens": len(de_mask_pos),
            "densemask_label_tokens": len(dm_label_pos),
            "densemask_mask_tokens": len(dm_mask_pos),
            "label_equal_sparse": label_equal,
            "sparse_mask_equal_sparse_labels": sparse_mask_equal_sparse_labels,
            "dense_mask_equal_dense_labels": dense_mask_equal_dense_labels,
            "densemask_labels_subset_masks": dm_labels_subset_dm_masks,
            "sparse_labels_subset_original_dense_masks": sparse_labels_subset_dense_masks,
            "densemask_masks_equal_original_dense_masks": dm_masks_equal_dense_masks,
            "densemask_masks_equal_original_dense_union_sparse_labels": dm_masks_dense_plus_labels,
            "extra_mask_tokens_vs_original_dense": len(extra_vs_dense),
            "missing_mask_tokens_vs_original_dense": len(missing_vs_dense),
            "sparse_label_tokens_outside_original_dense": len(labels_outside_dense),
            "original_dense_tokens_missing_from_densemask": len(dense_outside_dm),
            "symmetric_mask_token_diff": sym_mask_diff,
        }

        aggregate.update({
            "rows_checked": 1,
            "candidate_groups": n_candidate,
            "rows_candidate_groups_gt_sparse_cap": int(n_candidate > int(args.sparse_cap)),
            "rows_candidate_groups_gt_dense_cap": int(n_candidate > int(args.dense_cap)),
            "sparse_label_tokens": len(sp_label_pos),
            "original_sparse_mask_tokens": len(sp_mask_pos),
            "original_dense_label_tokens": len(de_label_pos),
            "original_dense_mask_tokens": len(de_mask_pos),
            "densemask_label_tokens": len(dm_label_pos),
            "densemask_mask_tokens": len(dm_mask_pos),
            "extra_mask_tokens_vs_original_dense": len(extra_vs_dense),
            "missing_mask_tokens_vs_original_dense": len(missing_vs_dense),
            "sparse_label_tokens_outside_original_dense": len(labels_outside_dense),
            "original_dense_tokens_missing_from_densemask": len(dense_outside_dm),
            "symmetric_mask_token_diff": sym_mask_diff,
            "rows_labels_equal_sparse": int(label_equal),
            "rows_dm_masks_equal_original_dense": int(dm_masks_equal_dense_masks),
            "rows_dm_masks_equal_dense_union_sparse": int(dm_masks_dense_plus_labels),
            "rows_sparse_labels_subset_original_dense": int(sparse_labels_subset_dense_masks),
            "rows_dm_labels_subset_dm_masks": int(dm_labels_subset_dm_masks),
        })
        # coarse histograms for scale without saving all rows
        hist["candidate_groups"][min(n_candidate // 16 * 16, 256)] += 1
        hist["symmetric_mask_token_diff"][min(sym_mask_diff, 64)] += 1
        hist["labels_outside_dense"][min(len(labels_outside_dense), 32)] += 1

        hard_fail = not (label_equal and sparse_mask_equal_sparse_labels and dense_mask_equal_dense_labels and dm_labels_subset_dm_masks)
        exact_mask_fail = not dm_masks_equal_dense_masks
        if hard_fail:
            failures.append(row_summary)
        if exact_mask_fail or labels_outside_dense or dense_outside_dm:
            mismatch_examples.append(row_summary)

        if (idx + 1) % 500 == 0:
            print(json.dumps({"event": "progress", "rows": idx + 1, "elapsed_sec": round(time.time() - t0, 2)}, ensure_ascii=False), flush=True)

    rows_checked = int(aggregate["rows_checked"])
    exact_labels_all = aggregate["rows_labels_equal_sparse"] == rows_checked
    labels_masked_all = aggregate["rows_dm_labels_subset_dm_masks"] == rows_checked
    exact_dense_masks_all = aggregate["rows_dm_masks_equal_original_dense"] == rows_checked
    dense_union_sparse_all = aggregate["rows_dm_masks_equal_dense_union_sparse"] == rows_checked
    sparse_subset_dense_all = aggregate["rows_sparse_labels_subset_original_dense"] == rows_checked

    if exact_labels_all and labels_masked_all and exact_dense_masks_all:
        status = "PASS_EXACT_SPARSE_LABELS_AND_DENSE_MASKS"
        interpretation = "The control exactly preserves original sparse supervision while applying the same input mask set as the original dense policy on every checked Qwen row."
    elif exact_labels_all and labels_masked_all and dense_union_sparse_all:
        status = "PASS_QUANTIFIED_DENSE_UNION_SPARSE_LABELS"
        interpretation = "The control preserves original sparse supervision and keeps every labelled token masked, but its mask set is original dense masks plus any sparse labels omitted by dense sampling. This is a quantified near-control rather than an exact dense-mask factorial."
    elif exact_labels_all and labels_masked_all:
        status = "PASS_LABELS_BUT_MASKS_DIFFER"
        interpretation = "Sparse labels are exact and masked, but the input mask set differs from original dense masking; inspect mismatch summaries before GPU training."
    else:
        status = "FAIL_CORE_LABEL_OR_LABEL_MASK_INVARIANT"
        interpretation = "The control does not reliably preserve sparse labels or keep labels masked; repair before training."

    report = {
        "status": status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": rel(__file__),
        "tail": rel(TAIL),
        "prefix_info": prefix,
        "parameters": {
            "max_updates": int(args.max_updates),
            "words_per_update": int(args.words_per_update),
            "train_seed": int(args.train_seed),
            "sparse_focus_prob": float(args.sparse_focus_prob),
            "sparse_cap": int(args.sparse_cap),
            "dense_focus_prob": float(args.dense_focus_prob),
            "dense_cap": int(args.dense_cap),
            "densemask_max_mask_groups": int(args.densemask_max_mask_groups),
            "max_rows": int(args.max_rows),
        },
        "rows_checked": rows_checked,
        "aggregate": dict(aggregate),
        "fractions": {
            "labels_equal_sparse_rows": aggregate["rows_labels_equal_sparse"] / rows_checked if rows_checked else None,
            "densemask_labels_subset_masks_rows": aggregate["rows_dm_labels_subset_dm_masks"] / rows_checked if rows_checked else None,
            "densemask_masks_equal_original_dense_rows": aggregate["rows_dm_masks_equal_original_dense"] / rows_checked if rows_checked else None,
            "densemask_masks_equal_dense_union_sparse_rows": aggregate["rows_dm_masks_equal_dense_union_sparse"] / rows_checked if rows_checked else None,
            "sparse_labels_subset_original_dense_rows": aggregate["rows_sparse_labels_subset_original_dense"] / rows_checked if rows_checked else None,
            "label_to_densemask_mask_token_ratio": aggregate["densemask_label_tokens"] / aggregate["densemask_mask_tokens"] if aggregate["densemask_mask_tokens"] else None,
        },
        "histograms": {k: dict(v) for k, v in hist.items()},
        "invariants": {
            "labels_exactly_original_sparse": exact_labels_all,
            "densemask_labels_subset_masks": labels_masked_all,
            "masks_exactly_original_dense": exact_dense_masks_all,
            "masks_equal_original_dense_union_sparse_labels": dense_union_sparse_all,
            "sparse_labels_subset_original_dense_masks": sparse_subset_dense_all,
        },
        "failure_examples": summarize_examples(failures),
        "mismatch_examples": summarize_examples(mismatch_examples),
        "interpretation": interpretation,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = args.out_dir / "fullprefix_row_verify.json"
    out_md = args.out_dir / "fullprefix_row_verify.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research dense-mask/sparse-label full-prefix verification\n\n",
        f"Status: `{status}`\n\n",
        f"Rows checked: `{rows_checked}` over prefix `{prefix.get('prefix_rows')}` rows / `{prefix.get('prefix_words')}` words.\n\n",
        "## Core counts\n\n",
        f"- Sparse label tokens: `{aggregate['sparse_label_tokens']}`\n",
        f"- Original dense mask tokens: `{aggregate['original_dense_mask_tokens']}`\n",
        f"- Dense-mask/sparse-label label tokens: `{aggregate['densemask_label_tokens']}`\n",
        f"- Dense-mask/sparse-label mask tokens: `{aggregate['densemask_mask_tokens']}`\n",
        f"- Extra mask tokens vs original dense: `{aggregate['extra_mask_tokens_vs_original_dense']}`\n",
        f"- Missing mask tokens vs original dense: `{aggregate['missing_mask_tokens_vs_original_dense']}`\n",
        f"- Sparse label tokens outside original dense mask: `{aggregate['sparse_label_tokens_outside_original_dense']}`\n\n",
        "## Row-level invariants\n\n",
    ]
    for k, v in report["invariants"].items():
        md.append(f"- `{k}`: `{v}`\n")
    md.extend([
        "\n## Interpretation\n\n",
        interpretation + "\n\n",
        f"Full JSON: `{rel(out_json)}`\n",
    ])
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "rows_checked": rows_checked,
        "exact_labels_all": exact_labels_all,
        "exact_dense_masks_all": exact_dense_masks_all,
        "dense_union_sparse_all": dense_union_sparse_all,
        "sparse_label_tokens_outside_original_dense": int(aggregate["sparse_label_tokens_outside_original_dense"]),
        "extra_mask_tokens_vs_original_dense": int(aggregate["extra_mask_tokens_vs_original_dense"]),
        "missing_mask_tokens_vs_original_dense": int(aggregate["missing_mask_tokens_vs_original_dense"]),
        "out_json": rel(out_json),
        "elapsed_sec": round(time.time() - t0, 3),
    }, indent=2, ensure_ascii=False), flush=True)
    if not (exact_labels_all and labels_masked_all):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
