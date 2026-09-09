#!/usr/bin/env python3
"""research fast full-prefix verification for dense-mask/sparse-label control.

This is a faster replacement for the research tensor-heavy verifier.  It enumerates
Qwen second-view content groups once per row, reproduces the deterministic sparse,
dense, and dense-mask/sparse-label policies, and reports whether the control really
has:

  * the same supervised labels as original sparse focus (seed62064, p=0.35, cap=16);
  * the same masked input positions as original dense focus (seed62064, p=1.0, cap=128),
    or a quantified dense-union-sparse near-control;
  * all supervised labels masked;
  * the actual normalized loss arithmetic implied by lambda_focus=0.15.

To guard against a verifier that only checks a reimplementation, it also preserves
the original research constructor before importing research's monkeypatch and compares
actual tensor outputs on a stratified row sample.  It performs no training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import torch

SCRIPT = _public_path('experiments/archive/functional_learning/scripts/verify_densemask_sparselabel_fast.py')
ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_stream_train_weighted as s64  # noqa: E402

# Preserve originals before research patches research.
ORIGINAL_APPLY_VIEW_FOCUS_ROW = s64.apply_view_focus_row
ORIGINAL_STABLE_SEED = s64.stable_seed
ORIGINAL_LOAD_PREFIX = s64.load_prefix

import corrected_bridge_trainer as bridge  # noqa: E402
import densemask_sparselabel_train as dm  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_fast_verify')
TAIL = _public_path('experiments/archive/functional_learning/data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl')
SPARSE_SUMMARY = _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/correspondence_focus_weighted/train_summary.json')
SPARSE_LOG = _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/correspondence_focus_weighted/update_log.jsonl')
DENSE64_SUMMARY = _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/train_summary.json')
DENSE64_LOG = _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/update_log.jsonl')
DENSE65_SUMMARY = _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/train_summary.json')
DENSE65_LOG = _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/update_log.jsonl')
WORD_NORM_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?")


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_word(w: str) -> str:
    m = WORD_NORM_RE.search((w or "").lower())
    return m.group(0).strip("'’") if m else ""


def source_norm_set(source_text: str) -> set[str]:
    return {norm_word(m.group(0)) for m in WORD_NORM_RE.finditer(source_text or "") if norm_word(m.group(0))}


def locate_groups(row: Dict[str, Any], tok: Dict[str, Any]) -> List[Dict[str, Any]]:
    offsets = tok["offsets"]
    attention_mask = tok["attention_mask"]
    text = str(row.get("text", ""))
    groups: List[Dict[str, Any]] = []
    for seg_i, seg in enumerate(row.get("qwen_pair_segments") or []):
        view_text = str(seg.get("view_text", ""))
        source_text = str(seg.get("source_text", ""))
        vstart = int(seg.get("view_start", -1))
        vend = int(seg.get("view_end", -1))
        if vstart < 0 or vend <= vstart or vend > len(text):
            continue
        if text[vstart:vend] != view_text:
            continue
        source_words = source_norm_set(source_text)
        for a, b, word in s64.content_word_spans(view_text):
            pos = s64.locate_positions(offsets, vstart + a, vstart + b)
            pos = tuple(int(p) for p in pos if int(attention_mask[p]) == 1)
            if not pos:
                continue
            nw = norm_word(word)
            # Signature deliberately excludes positions only after keeping segment/word fields;
            # list equality/membership in research relies on dict equality, so this is stricter
            # than position-set comparison for sampled groups.
            groups.append({
                "segment_index": int(seg_i),
                "pair_id": str(seg.get("pair_id", "")),
                "view_kind": seg.get("view_kind"),
                "candidate_kind": seg.get("candidate_kind"),
                "word": word,
                "norm_word": nw,
                "positions": pos,
                "n_tokens": len(pos),
                "copied_binary": "copied" if nw and nw in source_words else "not_copied",
            })
    return groups


def row_seed(row: Dict[str, Any], train_seed: int) -> int:
    return ORIGINAL_STABLE_SEED("view-focus", int(train_seed), bridge.row_key(row))


def sample_sparse_label_groups(groups: List[Dict[str, Any]], seed: int, focus_prob: float, cap: int) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    candidates = list(groups)
    if len(candidates) > int(cap):
        candidates = rng.sample(candidates, int(cap))
    chosen = [g for g in candidates if rng.random() < float(focus_prob)]
    if not chosen and candidates:
        chosen = [rng.choice(candidates)]
    return chosen


def sample_dense_groups_like_step064(groups: List[Dict[str, Any]], seed: int, cap: int) -> List[Dict[str, Any]]:
    # Original dense focus is research apply_view_focus_row with focus_prob=1.0.
    rng = random.Random(int(seed))
    candidates = list(groups)
    if len(candidates) > int(cap):
        candidates = rng.sample(candidates, int(cap))
    return list(candidates)


def sample_densemask_groups_like_step075(groups: List[Dict[str, Any]], labels: List[Dict[str, Any]], row: Dict[str, Any], seed: int, max_mask_groups: int) -> List[Dict[str, Any]]:
    masks = list(groups)
    if len(masks) > int(max_mask_groups):
        mrng = random.Random(dm.stable_seed("densemask-mask-cap", int(seed), bridge.row_key(row), int(max_mask_groups)))
        sampled = mrng.sample(masks, int(max_mask_groups))
        for g in labels:
            if g not in sampled:
                sampled.append(g)
        masks = sampled
    return masks


def positions_from_groups(groups: Iterable[Dict[str, Any]]) -> set[int]:
    out: set[int] = set()
    for g in groups:
        out.update(int(p) for p in g.get("positions", ()))
    return out


def group_keys(groups: Iterable[Dict[str, Any]]) -> set[Tuple[Any, ...]]:
    return {
        (
            g.get("segment_index"),
            g.get("pair_id"),
            g.get("candidate_kind"),
            g.get("word"),
            tuple(g.get("positions", ())),
        )
        for g in groups
    }


def tensor_label_positions(labels: torch.Tensor) -> set[int]:
    return {int(i) for i in torch.nonzero(labels != -100, as_tuple=False).flatten().tolist()}


def tensor_mask_positions(input_ids: torch.Tensor, mask_id: int) -> set[int]:
    return {int(i) for i in torch.nonzero(input_ids == int(mask_id), as_tuple=False).flatten().tolist()}


def add_group_distribution(counter: Counter, groups: Iterable[Dict[str, Any]], prefix: str) -> None:
    for g in groups:
        cb = str(g.get("copied_binary", "unknown"))
        counter[f"{prefix}_groups_{cb}"] += 1
        counter[f"{prefix}_tokens_{cb}"] += int(g.get("n_tokens", 0))
        kind = str(g.get("candidate_kind", "unknown"))
        counter[f"{prefix}_groups_kind::{kind}"] += 1


def update_policy_counter(c: Counter, raw: List[Dict[str, Any]], labels: List[Dict[str, Any]], masks: List[Dict[str, Any]]) -> None:
    raw_pos = positions_from_groups(raw)
    label_pos = positions_from_groups(labels)
    mask_pos = positions_from_groups(masks)
    c["qwen_rows"] += 1
    c["raw_groups"] += len(raw)
    c["raw_tokens"] += len(raw_pos)
    c["label_groups"] += len(labels)
    c["label_tokens"] += len(label_pos)
    c["mask_groups"] += len(masks)
    c["mask_tokens"] += len(mask_pos)
    c["mask_only_tokens"] += len(mask_pos - label_pos)
    if not labels:
        c["zero_label_qwen_rows"] += 1
    add_group_distribution(c, raw, "raw")
    add_group_distribution(c, labels, "label")
    add_group_distribution(c, masks, "mask")


def make_sample_indices(qwen_rows: list[Dict[str, Any]], raw_group_counts: list[int], n: int) -> list[int]:
    if n <= 0:
        return []
    pool: set[int] = set(range(min(20, len(qwen_rows))))
    # Include the rows with the largest group counts and regularly spaced rows.
    pool.update(i for i, _ in sorted(enumerate(raw_group_counts), key=lambda x: x[1], reverse=True)[: min(60, len(qwen_rows))])
    stride = max(1, len(qwen_rows) // max(1, n))
    pool.update(range(0, len(qwen_rows), stride))
    # Deterministic additional fill.
    rng = random.Random(84084)
    while len(pool) < min(n, len(qwen_rows)):
        pool.add(rng.randrange(len(qwen_rows)))
    return sorted(list(pool))[: min(n, len(qwen_rows))]


def compare_actual_constructors(sample_indices: list[int], qwen_rows: list[Dict[str, Any]], tokenizer, wgb, args: argparse.Namespace) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    aggregate = Counter()
    for idx in sample_indices:
        row = qwen_rows[idx]
        tok = bridge.tokenize_row(row, tokenizer, int(args.seq_length), wgb)
        seed = row_seed(row, int(args.train_seed))
        groups = locate_groups(row, tok)
        sparse_labels_g = sample_sparse_label_groups(groups, seed, float(args.sparse_focus_prob), int(args.sparse_cap))
        dense_g = sample_dense_groups_like_step064(groups, seed, int(args.dense_cap))
        dm_masks_g = sample_densemask_groups_like_step075(groups, sparse_labels_g, row, seed, int(args.densemask_max_mask_groups))
        expect_label_pos = positions_from_groups(sparse_labels_g)
        expect_dense_pos = positions_from_groups(dense_g)
        expect_dm_pos = positions_from_groups(dm_masks_g)
        sparse_masked, sparse_labels, _ = ORIGINAL_APPLY_VIEW_FOCUS_ROW(row, tok, tokenizer, seed, float(args.sparse_focus_prob), int(args.sparse_cap))
        dense_masked, dense_labels, _ = ORIGINAL_APPLY_VIEW_FOCUS_ROW(row, tok, tokenizer, seed, 1.0, int(args.dense_cap))
        dm_masked, dm_labels, _ = dm.apply_densemask_sparse_label_row(row, tok, tokenizer, seed, float(args.sparse_focus_prob), int(args.sparse_cap))
        got_sparse_label = tensor_label_positions(sparse_labels)
        got_sparse_mask = tensor_mask_positions(sparse_masked, tokenizer.mask_token_id)
        got_dense_label = tensor_label_positions(dense_labels)
        got_dense_mask = tensor_mask_positions(dense_masked, tokenizer.mask_token_id)
        got_dm_label = tensor_label_positions(dm_labels)
        got_dm_mask = tensor_mask_positions(dm_masked, tokenizer.mask_token_id)
        ok = (
            got_sparse_label == expect_label_pos
            and got_sparse_mask == expect_label_pos
            and got_dense_label == expect_dense_pos
            and got_dense_mask == expect_dense_pos
            and got_dm_label == expect_label_pos
            and got_dm_mask == expect_dm_pos
            and got_dm_label.issubset(got_dm_mask)
        )
        aggregate.update({
            "sample_rows_checked": 1,
            "sample_rows_ok": int(ok),
            "sample_sparse_label_tokens": len(got_sparse_label),
            "sample_dense_mask_tokens": len(got_dense_mask),
            "sample_dm_label_tokens": len(got_dm_label),
            "sample_dm_mask_tokens": len(got_dm_mask),
        })
        if not ok:
            failures.append({
                "idx": idx,
                "row_key": bridge.row_key(row),
                "raw_groups": len(groups),
                "expected_sparse_label_tokens": len(expect_label_pos),
                "got_sparse_label_tokens": len(got_sparse_label),
                "expected_dense_mask_tokens": len(expect_dense_pos),
                "got_dense_mask_tokens": len(got_dense_mask),
                "expected_dm_mask_tokens": len(expect_dm_pos),
                "got_dm_mask_tokens": len(got_dm_mask),
                "dm_label_symmetric_diff": len(expect_label_pos ^ got_dm_label),
                "dm_mask_symmetric_diff": len(expect_dm_pos ^ got_dm_mask),
            })
            if len(failures) >= 12:
                break
    return {
        "status": "sample_actual_constructor_pass" if not failures and aggregate.get("sample_rows_checked", 0) == len(sample_indices) else "sample_actual_constructor_fail",
        "requested_sample_rows": len(sample_indices),
        "aggregate": dict(aggregate),
        "failures": failures,
    }


def validation_against_actual_summaries(policy_totals: dict[str, Any]) -> dict[str, Any]:
    actuals = {
        "sparse_seed62064": read_json(SPARSE_SUMMARY),
        "dense_seed62064": read_json(DENSE64_SUMMARY),
        "dense_seed62065": read_json(DENSE65_SUMMARY),
    }
    out: dict[str, Any] = {}
    for key, actual in actuals.items():
        prof = policy_totals.get(key)
        if not actual or not prof:
            out[key] = {"status": "missing_actual_or_profile", "actual_path": rel({"sparse_seed62064": SPARSE_SUMMARY, "dense_seed62064": DENSE64_SUMMARY, "dense_seed62065": DENSE65_SUMMARY}[key])}
            continue
        out[key] = {
            "actual_train_summary_path": rel({"sparse_seed62064": SPARSE_SUMMARY, "dense_seed62064": DENSE64_SUMMARY, "dense_seed62065": DENSE65_SUMMARY}[key]),
            "actual_focus_targets": actual.get("total_focus_targets"),
            "profile_label_tokens": prof.get("label_tokens"),
            "focus_target_token_diff_profile_minus_actual": int(prof.get("label_tokens", 0)) - int(actual.get("total_focus_targets", 0)),
            "actual_focus_selected_groups_total": actual.get("focus_selected_groups_total"),
            "profile_label_groups": prof.get("label_groups"),
            "focus_group_diff_profile_minus_actual": int(prof.get("label_groups", 0)) - int(actual.get("focus_selected_groups_total", 0)),
            "actual_focus_candidate_groups_total": actual.get("focus_candidate_groups_total"),
            "profile_label_candidate_groups_after_cap": prof.get("label_candidate_groups_after_cap"),
            "candidate_group_diff_profile_minus_actual": int(prof.get("label_candidate_groups_after_cap", 0)) - int(actual.get("focus_candidate_groups_total", 0)),
        }
    return out


def compute_update_weight_stats(log_path: pathlib.Path, label: str) -> dict[str, Any]:
    rows = read_jsonl(log_path)
    if not rows:
        return {"label": label, "log_path": rel(log_path), "status": "missing"}
    per = []
    for r in rows:
        fn = int(r.get("focus_target_tokens", 0))
        on = int(r.get("ordinary_target_tokens", 0))
        lam = float(r.get("lambda_focus", 0.15))
        olam = float(r.get("lambda_ordinary", 0.85))
        per.append({
            "update": int(r.get("update", len(per) + 1)),
            "focus_n": fn,
            "ordinary_n": on,
            "focus_per_token_weight": lam / fn if fn else None,
            "ordinary_per_token_weight": olam / on if on else None,
            "focus_vs_ordinary_per_token_ratio": (lam / fn) / (olam / on) if fn and on and olam else None,
        })
    vals = [x["focus_per_token_weight"] for x in per if x["focus_per_token_weight"] is not None]
    ratios = [x["focus_vs_ordinary_per_token_ratio"] for x in per if x["focus_vs_ordinary_per_token_ratio"] is not None]
    return {
        "label": label,
        "log_path": rel(log_path),
        "updates": len(per),
        "total_focus_targets": sum(int(x["focus_n"]) for x in per),
        "total_ordinary_targets": sum(int(x["ordinary_n"]) for x in per),
        "mean_focus_targets_per_update": sum(int(x["focus_n"]) for x in per) / len(per),
        "min_focus_targets_per_update": min(int(x["focus_n"]) for x in per),
        "max_focus_targets_per_update": max(int(x["focus_n"]) for x in per),
        "mean_focus_per_token_weight": sum(vals) / len(vals) if vals else None,
        "min_focus_per_token_weight": min(vals) if vals else None,
        "max_focus_per_token_weight": max(vals) if vals else None,
        "mean_focus_vs_ordinary_per_token_ratio": sum(ratios) / len(ratios) if ratios else None,
    }


def finalize_policy_counter(c: Counter) -> dict[str, Any]:
    d = dict(c)
    lt = int(d.get("label_tokens", 0))
    mt = int(d.get("mask_tokens", 0))
    rg = int(d.get("raw_groups", 0))
    return {
        **d,
        "label_to_mask_token_ratio": lt / mt if mt else None,
        "mask_only_token_fraction": int(d.get("mask_only_tokens", 0)) / mt if mt else None,
        "label_group_fraction_of_raw": int(d.get("label_groups", 0)) / rg if rg else None,
        "mask_group_fraction_of_raw": int(d.get("mask_groups", 0)) / rg if rg else None,
        "label_copied_token_fraction": int(d.get("label_tokens_copied", 0)) / lt if lt else None,
        "mask_copied_token_fraction": int(d.get("mask_tokens_copied", 0)) / mt if mt else None,
    }


def write_md(path: pathlib.Path, report: dict[str, Any]) -> None:
    lines = ["# research dense-mask/sparse-label fast verification\n"]
    lines.append(f"Status: `{report['status']}`\n")
    pfx = report.get("prefix_info", {})
    lines.append(f"Prefix: `{pfx.get('prefix_rows')}` rows, `{pfx.get('prefix_words')}` words, Qwen rows `{pfx.get('qwen_rows')}`, Qwen pair segments `{pfx.get('qwen_pair_segments')}`.\n")
    core = report.get("core_invariants", {})
    lines.append("## Core invariants\n")
    for k, v in core.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("\n## Policy totals\n")
    for key, rec in report.get("policy_totals", {}).items():
        lines.append(f"- {key}: labels `{rec.get('label_groups')}` groups / `{rec.get('label_tokens')}` tokens; masks `{rec.get('mask_groups')}` groups / `{rec.get('mask_tokens')}` tokens; mask-only `{rec.get('mask_only_tokens')}` tokens; label/mask token ratio `{rec.get('label_to_mask_token_ratio')}`.")
    obj = report.get("objective_arithmetic", {})
    lines.append("\n## Objective arithmetic\n")
    lines.append(f"- `lambda_focus`: `{obj.get('lambda_focus')}`; focus loss is a mean over retained focus labels, so per-token focus weight scales inversely with focus target count within each macro update.")
    lines.append(f"- Dense target count / sparse target count: `{obj.get('dense64_to_sparse_total_focus_target_ratio')}`; dense-mask/sparse-label target count / dense target count: `{obj.get('dm64_to_dense64_total_focus_target_ratio')}`.")
    lines.append(f"- Against sparse focus, the control keeps the same labels and therefore the same focus-target weight schedule; against dense focus, each retained focus label receives about `{obj.get('dense64_to_dm64_mean_per_token_focus_weight_ratio')}` times the per-label focus weight, while extra dense masked tokens are unsupervised context corruption.")
    lines.append("\n## Interpretation\n")
    lines.append(str(report.get("interpretation", "")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--train-seed", type=int, default=62064)
    ap.add_argument("--second-train-seed", type=int, default=62065)
    ap.add_argument("--sparse-focus-prob", type=float, default=0.35)
    ap.add_argument("--sparse-cap", type=int, default=16)
    ap.add_argument("--dense-cap", type=int, default=128)
    ap.add_argument("--densemask-max-mask-groups", type=int, default=128)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--sample-actual-rows", type=int, default=256)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QIUSHI_DENSEMASK_MAX_MASK_GROUPS"] = str(int(args.densemask_max_mask_groups))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rows, prefix_info = ORIGINAL_LOAD_PREFIX(TAIL, int(args.max_updates), int(args.words_per_update))
    qwen_rows = [r for r in rows if r.get("source") == "qwen_pair_packed" and r.get("qwen_pair_segments")]
    if args.max_rows and int(args.max_rows) > 0:
        qwen_rows = qwen_rows[: int(args.max_rows)]
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = bridge.WordGroupBuilder(tokenizer)

    aggregate = Counter()
    policy_counters = {
        "sparse_seed62064": Counter(),
        "dense_seed62064": Counter(),
        "dense_seed62065": Counter(),
        "densemask_sparselabel_seed62064": Counter(),
        "densemask_sparselabel_seed62065": Counter(),
    }
    mismatch_examples: list[dict[str, Any]] = []
    raw_group_counts: list[int] = []
    macro_focus_counts = defaultdict(lambda: Counter())

    max_qwen_rows = int(args.max_rows) if args.max_rows and int(args.max_rows) > 0 else None
    cursor = 0
    stop_main_pass = False
    for update_i in range(int(args.max_updates)):
        if stop_main_pass:
            break
        words = 0
        current: list[Dict[str, Any]] = []
        while cursor < len(rows) and words < int(args.words_per_update):
            r = rows[cursor]
            current.append(r)
            words += int(r.get("words", s64.wc(r.get("text", ""))))
            cursor += 1
        if not current:
            break
        for r in current:
            if not (r.get("source") == "qwen_pair_packed" and r.get("qwen_pair_segments")):
                continue
            if max_qwen_rows is not None and len(raw_group_counts) >= max_qwen_rows:
                stop_main_pass = True
                break
            tok = bridge.tokenize_row(r, tokenizer, int(args.seq_length), wgb)
            groups = locate_groups(r, tok)
            qidx = len(raw_group_counts)
            raw_group_counts.append(len(groups))
            seed64 = row_seed(r, int(args.train_seed))
            seed65 = row_seed(r, int(args.second_train_seed))

            sparse64 = sample_sparse_label_groups(groups, seed64, float(args.sparse_focus_prob), int(args.sparse_cap))
            dense64 = sample_dense_groups_like_step064(groups, seed64, int(args.dense_cap))
            dm64_masks = sample_densemask_groups_like_step075(groups, sparse64, r, seed64, int(args.densemask_max_mask_groups))
            sparse65 = sample_sparse_label_groups(groups, seed65, float(args.sparse_focus_prob), int(args.sparse_cap))
            dense65 = sample_dense_groups_like_step064(groups, seed65, int(args.dense_cap))
            dm65_masks = sample_densemask_groups_like_step075(groups, sparse65, r, seed65, int(args.densemask_max_mask_groups))

            pos_sparse64 = positions_from_groups(sparse64)
            pos_dense64 = positions_from_groups(dense64)
            pos_dm64 = positions_from_groups(dm64_masks)
            pos_sparse65 = positions_from_groups(sparse65)
            pos_dense65 = positions_from_groups(dense65)
            pos_dm65 = positions_from_groups(dm65_masks)

            aggregate.update({
                "qwen_rows_checked": 1,
                "raw_groups": len(groups),
                "raw_tokens": len(positions_from_groups(groups)),
                "rows_over_sparse_cap": int(len(groups) > int(args.sparse_cap)),
                "rows_over_dense_cap": int(len(groups) > int(args.dense_cap)),
                "seed62064_labels_subset_masks": int(pos_sparse64.issubset(pos_dm64)),
                "seed62064_dm_labels_equal_sparse": 1,  # by construction; actual sample verifies tensors
                "seed62064_dm_masks_equal_dense": int(pos_dm64 == pos_dense64),
                "seed62064_dm_masks_equal_dense_union_sparse": int(pos_dm64 == (pos_dense64 | pos_sparse64)),
                "seed62064_sparse_labels_subset_original_dense": int(pos_sparse64.issubset(pos_dense64)),
                "seed62064_extra_mask_tokens_vs_dense": len(pos_dm64 - pos_dense64),
                "seed62064_missing_mask_tokens_vs_dense": len(pos_dense64 - pos_dm64),
                "seed62065_labels_subset_masks": int(pos_sparse65.issubset(pos_dm65)),
                "seed62065_dm_labels_equal_sparse": 1,
                "seed62065_dm_masks_equal_dense": int(pos_dm65 == pos_dense65),
                "seed62065_dm_masks_equal_dense_union_sparse": int(pos_dm65 == (pos_dense65 | pos_sparse65)),
                "seed62065_sparse_labels_subset_original_dense": int(pos_sparse65.issubset(pos_dense65)),
                "seed62065_extra_mask_tokens_vs_dense": len(pos_dm65 - pos_dense65),
                "seed62065_missing_mask_tokens_vs_dense": len(pos_dense65 - pos_dm65),
            })
            macro_focus_counts["sparse_seed62064"][update_i] += len(pos_sparse64)
            macro_focus_counts["dense_seed62064"][update_i] += len(pos_dense64)
            macro_focus_counts["densemask_sparselabel_seed62064"][update_i] += len(pos_sparse64)
            macro_focus_counts["sparse_seed62065"][update_i] += len(pos_sparse65)
            macro_focus_counts["dense_seed62065"][update_i] += len(pos_dense65)
            macro_focus_counts["densemask_sparselabel_seed62065"][update_i] += len(pos_sparse65)

            update_policy_counter(policy_counters["sparse_seed62064"], groups, sparse64, sparse64)
            policy_counters["sparse_seed62064"]["label_candidate_groups_after_cap"] += len(sample_dense_groups_like_step064(groups, seed64, int(args.sparse_cap)))
            update_policy_counter(policy_counters["dense_seed62064"], groups, dense64, dense64)
            policy_counters["dense_seed62064"]["label_candidate_groups_after_cap"] += len(dense64)
            update_policy_counter(policy_counters["densemask_sparselabel_seed62064"], groups, sparse64, dm64_masks)
            policy_counters["densemask_sparselabel_seed62064"]["label_candidate_groups_after_cap"] += len(sample_dense_groups_like_step064(groups, seed64, int(args.sparse_cap)))
            update_policy_counter(policy_counters["dense_seed62065"], groups, dense65, dense65)
            policy_counters["dense_seed62065"]["label_candidate_groups_after_cap"] += len(dense65)
            update_policy_counter(policy_counters["densemask_sparselabel_seed62065"], groups, sparse65, dm65_masks)
            policy_counters["densemask_sparselabel_seed62065"]["label_candidate_groups_after_cap"] += len(sample_dense_groups_like_step064(groups, seed65, int(args.sparse_cap)))

            if (pos_dm64 != pos_dense64) or (not pos_sparse64.issubset(pos_dm64)) or (pos_dm65 != pos_dense65):
                if len(mismatch_examples) < 20:
                    mismatch_examples.append({
                        "qwen_index": qidx,
                        "row_key": bridge.row_key(r),
                        "raw_groups": len(groups),
                        "seed62064_sparse_label_tokens": len(pos_sparse64),
                        "seed62064_dense_mask_tokens": len(pos_dense64),
                        "seed62064_dm_mask_tokens": len(pos_dm64),
                        "seed62064_extra_vs_dense": len(pos_dm64 - pos_dense64),
                        "seed62064_missing_vs_dense": len(pos_dense64 - pos_dm64),
                        "seed62065_sparse_label_tokens": len(pos_sparse65),
                        "seed62065_dense_mask_tokens": len(pos_dense65),
                        "seed62065_dm_mask_tokens": len(pos_dm65),
                        "seed62065_extra_vs_dense": len(pos_dm65 - pos_dense65),
                        "seed62065_missing_vs_dense": len(pos_dense65 - pos_dm65),
                    })

    sample_indices = make_sample_indices(qwen_rows, raw_group_counts, int(args.sample_actual_rows))
    sample_actual = compare_actual_constructors(sample_indices, qwen_rows, tokenizer, wgb, args)

    rows_checked = int(aggregate.get("qwen_rows_checked", 0))
    core_invariants = {
        "rows_checked": rows_checked,
        "no_rows_over_dense_cap": int(aggregate.get("rows_over_dense_cap", 0)) == 0,
        "seed62064_labels_equal_original_sparse_by_group_logic": int(aggregate.get("seed62064_dm_labels_equal_sparse", 0)) == rows_checked,
        "seed62064_all_labels_masked": int(aggregate.get("seed62064_labels_subset_masks", 0)) == rows_checked,
        "seed62064_masks_equal_original_dense": int(aggregate.get("seed62064_dm_masks_equal_dense", 0)) == rows_checked,
        "seed62064_masks_equal_original_dense_union_sparse": int(aggregate.get("seed62064_dm_masks_equal_dense_union_sparse", 0)) == rows_checked,
        "seed62065_labels_equal_original_sparse_by_group_logic": int(aggregate.get("seed62065_dm_labels_equal_sparse", 0)) == rows_checked,
        "seed62065_all_labels_masked": int(aggregate.get("seed62065_labels_subset_masks", 0)) == rows_checked,
        "seed62065_masks_equal_original_dense": int(aggregate.get("seed62065_dm_masks_equal_dense", 0)) == rows_checked,
        "sample_actual_constructor_status": sample_actual.get("status"),
    }
    if rows_checked and core_invariants["seed62064_all_labels_masked"] and core_invariants["seed62064_masks_equal_original_dense"] and sample_actual.get("status") == "sample_actual_constructor_pass":
        status = "PASS_FULLPREFIX_EXACT_SPARSE_LABELS_DENSE_INPUT"
        interpretation = "The seed62064 causal arm is an exact full-prefix control against sparse focus for labels and against dense focus for input masks. With lambda_focus=0.15 on mean focus loss, it keeps the sparse focus-target weight schedule while exposing the same dense-masked Qwen second-view input context."
    elif rows_checked and core_invariants["seed62064_all_labels_masked"] and core_invariants["seed62064_masks_equal_original_dense_union_sparse"] and sample_actual.get("status") == "sample_actual_constructor_pass":
        status = "PASS_FULLPREFIX_QUANTIFIED_DENSE_UNION_SPARSE_LABELS"
        interpretation = "The causal arm preserves sparse labels and masks every label, but its input mask set is original dense masks plus sparse labels omitted by the dense cap. Interpret as a quantified near-control, not exact input matching."
    elif rows_checked and core_invariants["seed62064_all_labels_masked"] and sample_actual.get("status") == "sample_actual_constructor_pass":
        status = "PASS_LABELS_MASKED_BUT_INPUT_DIFFERS"
        interpretation = "The causal arm preserves sparse labels, but input masks differ from original dense masks. Repair or explicitly reinterpret before GPU training."
    else:
        status = "FAIL_DENSEMASK_SPARSELABEL_CONTROL"
        interpretation = "Do not launch GPU training: sparse-label or label-masking invariants did not pass independent full-prefix/sample checks."

    policy_totals = {k: finalize_policy_counter(v) for k, v in policy_counters.items()}
    validation = validation_against_actual_summaries(policy_totals)
    sparse_w = compute_update_weight_stats(SPARSE_LOG, "actual_sparse_seed62064")
    dense_w = compute_update_weight_stats(DENSE64_LOG, "actual_dense_seed62064")
    dense65_w = compute_update_weight_stats(DENSE65_LOG, "actual_dense_seed62065")
    dm64_focus = [int(macro_focus_counts["densemask_sparselabel_seed62064"][i]) for i in sorted(macro_focus_counts["densemask_sparselabel_seed62064"].keys())]
    dense64_focus = [int(macro_focus_counts["dense_seed62064"][i]) for i in sorted(macro_focus_counts["dense_seed62064"].keys())]
    lambda_focus = 0.15
    dm64_weights = [lambda_focus / x for x in dm64_focus if x]
    dense64_weights = [lambda_focus / x for x in dense64_focus if x]
    pair_ratios = [(lambda_focus / dm64_focus[i]) / (lambda_focus / dense64_focus[i]) for i in range(min(len(dm64_focus), len(dense64_focus))) if dm64_focus[i] and dense64_focus[i]]
    objective_arithmetic = {
        "lambda_focus": lambda_focus,
        "loss_definition": "For correspondence_focus_weighted, each macro update optimizes lambda_focus * mean_CE(focus labels) + (1-lambda_focus) * mean_CE(ordinary labels). Extra dense-mask/sparse-label mask-only tokens affect input context but are not focus CE labels.",
        "sparse64_total_focus_targets": policy_totals["sparse_seed62064"].get("label_tokens"),
        "dense64_total_focus_targets": policy_totals["dense_seed62064"].get("label_tokens"),
        "dm64_total_focus_targets": policy_totals["densemask_sparselabel_seed62064"].get("label_tokens"),
        "dense64_to_sparse_total_focus_target_ratio": (policy_totals["dense_seed62064"].get("label_tokens", 0) / policy_totals["sparse_seed62064"].get("label_tokens", 1)) if policy_totals["sparse_seed62064"].get("label_tokens", 0) else None,
        "dm64_to_dense64_total_focus_target_ratio": (policy_totals["densemask_sparselabel_seed62064"].get("label_tokens", 0) / policy_totals["dense_seed62064"].get("label_tokens", 1)) if policy_totals["dense_seed62064"].get("label_tokens", 0) else None,
        "mean_focus_per_token_weight_dm64": sum(dm64_weights) / len(dm64_weights) if dm64_weights else None,
        "mean_focus_per_token_weight_dense64": sum(dense64_weights) / len(dense64_weights) if dense64_weights else None,
        "dense64_to_dm64_mean_per_token_focus_weight_ratio": sum(pair_ratios) / len(pair_ratios) if pair_ratios else None,
        "actual_sparse_update_weight_stats": sparse_w,
        "actual_dense64_update_weight_stats": dense_w,
        "actual_dense65_update_weight_stats": dense65_w,
    }

    report = {
        "status": status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": rel(SCRIPT),
        "tail": rel(TAIL),
        "prefix_info": prefix_info,
        "parameters": {
            "max_updates": int(args.max_updates),
            "words_per_update": int(args.words_per_update),
            "train_seed": int(args.train_seed),
            "second_train_seed": int(args.second_train_seed),
            "sparse_focus_prob": float(args.sparse_focus_prob),
            "sparse_cap": int(args.sparse_cap),
            "dense_cap": int(args.dense_cap),
            "densemask_max_mask_groups": int(args.densemask_max_mask_groups),
            "seq_length": int(args.seq_length),
            "sample_actual_rows": int(args.sample_actual_rows),
            "max_rows": int(args.max_rows),
        },
        "core_invariants": core_invariants,
        "aggregate": dict(aggregate),
        "policy_totals": policy_totals,
        "validation_against_actual_train_summaries": validation,
        "sample_actual_constructor_check": sample_actual,
        "histograms": {
            "raw_group_count": {str(k): int(v) for k, v in Counter(raw_group_counts).items()},
            "raw_group_count_binned16": {str(k): int(v) for k, v in sorted(Counter((x // 16) * 16 for x in raw_group_counts).items())},
        },
        "mismatch_examples": mismatch_examples,
        "objective_arithmetic": objective_arithmetic,
        "interpretation": interpretation,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_json = args.out_dir / "densemask_sparselabel_fast_verify.json"
    out_md = args.out_dir / "densemask_sparselabel_fast_verify.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out_md, report)
    print(json.dumps({"status": status, "out_json": rel(out_json), "out_md": rel(out_md), "elapsed_sec": report["elapsed_sec"], "rows_checked": rows_checked}, indent=2), flush=True)
    if status.startswith("FAIL"):
        sys.exit(1)


if __name__ == "__main__":
    main()
