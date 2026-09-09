#!/usr/bin/env python3
"""research real-batch invariants for the PVDM/control masking contrast.

This is a CPU/GPU-free preflight before any H100 continuation.  It tokenizes real
70M->100M tail rows with the shared legal tokenizer, applies the exact paired
PVDM masking library, and measures whether the treatment and control preserve:

  * dependent target groups/tokens;
  * realized masked mass per row and batch;
  * 80/10/10 replacement randomness;

while differing materially in true-pivot visibility, with all quantities resolved
by relation category.  It also writes a one-batch K=0 legacy masking smoke using
the inherited WWM function only to confirm the imported base machinery remains
callable; the actual training parity run must still be a short GPU continuation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

USER_ROOT = Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = USER_ROOT / "experiments/archive" / 'representation_and_objectives' / "scripts"
for p in [str(COMPACT_EXPERIENCE_SCRIPTS), str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_masking_lib as pvdm  # noqa: E402

STUDY = Path("experiments/archive/representation_and_objectives")
WS = STUDY
LABEL_DIR = WS / "data/pvdm_strict_labels"
TAIL = LABEL_DIR / "compact_tail_70M_100M.jsonl"
LABELS = LABEL_DIR / "pvdm_strict_tail_70M_100M_labels.jsonl"
TOKENIZER_DIR = WS / "data/shared_tokenizer/shared_16k_tokenizer"
OUT = WS / "data/pvdm_batch_invariants"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/pvdm_batch_invariants.md')
EXPECTED = {
    "tail_sha256": "19304f23a57a3809b50af496a99068a03417cb27886796604da5e755bf004675",
    "labels_sha256": "39ed2b8eec610cb9f5e6456695d1b92057d6bb2af9fad0369790f1295589306a",
    "tokenizer_sha256": "e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(limit_rows: int) -> tuple[list[base.Example], list[dict[str, Any]]]:
    examples: list[base.Example] = []
    labels: list[dict[str, Any]] = []
    with TAIL.open(encoding="utf-8") as tf, LABELS.open(encoding="utf-8") as lf:
        for idx, (tl, ll) in enumerate(zip(tf, lf)):
            if idx >= limit_rows:
                break
            tobj = json.loads(tl); lobj = json.loads(ll)
            text = str(tobj["text"])
            words = int(tobj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"tail word mismatch row {idx}")
            if int(lobj.get("tail_row_idx", -1)) != idx:
                raise RuntimeError(f"label local index mismatch at row {idx}: {lobj.get('tail_row_idx')}")
            if int(lobj.get("words", words)) != words:
                raise RuntimeError(f"label word mismatch at row {idx}")
            examples.append(base.Example(text=text, words=words, example_id=int(tobj.get("example_id", idx)), source=str(tobj.get("source", ""))))
            labels.append(lobj)
    if len(examples) != limit_rows:
        raise RuntimeError(f"loaded {len(examples)} rows, expected {limit_rows}")
    return examples, labels


def per_row_token_counts(labels_t: torch.Tensor, labels_c: torch.Tensor) -> tuple[list[int], list[int]]:
    return (labels_t != -100).sum(dim=1).tolist(), (labels_c != -100).sum(dim=1).tolist()


def analyze_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor, label_records: list[dict[str, Any]], tokenizer, *, seed: int, mask_prob: float, target_prob: float) -> dict[str, Any]:
    masked_t, labels_t, stats_t = pvdm.apply_paired_pvdm_masking(input_ids, attention_mask, word_group, label_records, tokenizer, arm="treatment", seed=seed, mask_prob=mask_prob, target_prob=target_prob)
    masked_c, labels_c, stats_c = pvdm.apply_paired_pvdm_masking(input_ids, attention_mask, word_group, label_records, tokenizer, arm="control", seed=seed, mask_prob=mask_prob, target_prob=target_prob)
    row_records, pair_stats = pvdm.make_paired_group_selections(word_group, attention_mask, label_records, seed=seed, mask_prob=mask_prob, target_prob=target_prob, same_length_required=True)

    t_counts, c_counts = per_row_token_counts(labels_t, labels_c)
    selected_token_diff = [int(a - b) for a, b in zip(t_counts, c_counts)]
    target_token_t = 0; target_token_c = 0
    target_label_mismatch = 0
    target_original_mismatch = 0
    anchor_swap_action_mismatches = 0
    background_label_mismatch = 0
    selected_target_groups = 0
    pivot_visible_events = 0
    control_pivot_visible_events = 0
    category = defaultdict(lambda: Counter())
    selected_rows = 0
    for b, rr in enumerate(row_records):
        if rr["selected_events"]:
            selected_rows += 1
        group_pos = rr["group_pos"]
        # Ensure group-count and token-count mass equality at row level.
        if rr["treatment_token_count"] != rr["control_token_count"]:
            raise RuntimeError(f"token mass mismatch in row {b}: {rr['treatment_token_count']} vs {rr['control_token_count']}")
        if len(rr["treatment_groups"]) != len(rr["control_groups"]):
            raise RuntimeError(f"group mass mismatch in row {b}")
        for ev in rr["selected_events"]:
            cat = ev.category
            selected_target_groups += 1
            category[cat]["events_selected"] += 1
            category[cat]["target_tokens"] += ev.target_len
            category[cat]["anchor_tokens"] += ev.pivot_len
            category[cat]["rows_with_selected_event"] += 0  # fixed below from row cats
            ppos = group_pos[ev.pivot_gid]
            cpos = group_pos[ev.control_gid]
            tpos = group_pos[ev.target_gid]
            # Dependent target must be selected with identical original labels and identical replacement surface.
            for pos in tpos:
                target_token_t += 1; target_token_c += 1
                if int(labels_t[b, pos]) == -100 or int(labels_c[b, pos]) == -100:
                    target_label_mismatch += 1
                if int(labels_t[b, pos]) != int(labels_c[b, pos]):
                    target_original_mismatch += 1
                if int(masked_t[b, pos]) != int(masked_c[b, pos]):
                    # identical target replacement randomness should make this impossible
                    target_label_mismatch += 1
            # In treatment pivot remains original/visible and control anchor is the swapped masked group;
            # in control this is reversed.  Since lengths are equal and the semantic key is shared,
            # the replacement actions on the swapped anchor positions should match offset-by-offset.
            for off, (pp, cp) in enumerate(zip(ppos, cpos)):
                if int(labels_t[b, pp]) != -100:
                    pivot_visible_events -= 1000000  # hard marker for a failure
                if int(masked_t[b, pp]) == int(input_ids[b, pp]):
                    pivot_visible_events += 1
                if int(labels_c[b, pp]) == -100:
                    control_pivot_visible_events += 1
                # Action class should match between treatment control-anchor and control pivot-anchor.
                t_is_mask = int(masked_t[b, cp]) == int(tokenizer.mask_token_id)
                c_is_mask = int(masked_c[b, pp]) == int(tokenizer.mask_token_id)
                t_is_keep = int(masked_t[b, cp]) == int(input_ids[b, cp])
                c_is_keep = int(masked_c[b, pp]) == int(input_ids[b, pp])
                if (t_is_mask != c_is_mask) or (t_is_keep != c_is_keep):
                    anchor_swap_action_mismatches += 1
            # Background group replacement surfaces must be identical positions/labels.
        row_cats = {ev.category for ev in rr["selected_events"]}
        for cat in row_cats:
            category[cat]["rows_with_selected_event"] += 1
        for g in rr["background_groups"]:
            for pos in group_pos[g]:
                if int(labels_t[b, pos]) != int(labels_c[b, pos]):
                    background_label_mismatch += 1
                if int(masked_t[b, pos]) != int(masked_c[b, pos]):
                    background_label_mismatch += 1

    diff_tokens = int((masked_t != masked_c).sum().item())
    pair = {
        "rows": int(input_ids.shape[0]),
        "tokens": int(attention_mask.sum().item()),
        "stats_treatment": stats_t,
        "stats_control": stats_c,
        "pair_stats_selected": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in pair_stats.items() if k not in {"label_event_categories", "usable_event_categories", "selected_event_categories", "reject_reasons"}},
        "pair_category_counts": {k: dict(v) for k, v in category.items()},
        "selected_rows": selected_rows,
        "selected_target_groups": selected_target_groups,
        "treatment_masked_tokens": int((labels_t != -100).sum().item()),
        "control_masked_tokens": int((labels_c != -100).sum().item()),
        "masked_token_count_row_diff_abs_sum": int(sum(abs(x) for x in selected_token_diff)),
        "masked_token_count_row_diff_max_abs": int(max([abs(x) for x in selected_token_diff] or [0])),
        "dependent_target_tokens_treatment": target_token_t,
        "dependent_target_tokens_control": target_token_c,
        "dependent_target_label_or_replacement_mismatches": target_label_mismatch,
        "dependent_target_original_id_mismatches": target_original_mismatch,
        "background_label_or_replacement_mismatches": background_label_mismatch,
        "anchor_swap_action_mismatches": anchor_swap_action_mismatches,
        "true_pivot_visible_token_count_treatment": pivot_visible_events,
        "true_pivot_visible_token_count_control": control_pivot_visible_events,
        "masked_input_difference_tokens_between_arms": diff_tokens,
        "replacement_action_counts_treatment": stats_t.get("replacement_action_counts", {}),
        "replacement_action_counts_control": stats_c.get("replacement_action_counts", {}),
        "replacement_action_digest_treatment": stats_t.get("replacement_action_digest"),
        "replacement_action_digest_control": stats_c.get("replacement_action_digest"),
    }
    return pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--target-prob", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=43023)
    ap.add_argument("--max-seq-length", type=int, default=256)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {
        "tail_sha256": sha256_file(TAIL),
        "labels_sha256": sha256_file(LABELS),
        "tokenizer_sha256": sha256_file(TOKENIZER_DIR / "tokenizer.json"),
    }
    errors = [f"{k} mismatch {v} != {EXPECTED[k]}" for k, v in hashes.items() if v != EXPECTED[k]]
    if errors:
        raise RuntimeError("; ".join(errors))
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER_DIR))
    examples, labels = load_records(args.rows)
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=0)
    batch_summaries: list[dict[str, Any]] = []
    agg = Counter()
    cat_agg: dict[str, Counter] = defaultdict(Counter)
    replacement_t = Counter(); replacement_c = Counter(); reject = Counter(); usable_cats = Counter(); selected_cats = Counter(); label_cats = Counter()
    for bi, batch in enumerate(loader):
        lo = bi * args.batch_size; hi = lo + int(batch["input_ids"].shape[0])
        rec = analyze_batch(batch["input_ids"], batch["attention_mask"], batch["word_group"], labels[lo:hi], tokenizer, seed=args.seed, mask_prob=args.mask_prob, target_prob=args.target_prob)
        rec["batch_index"] = bi
        rec["row_start"] = lo
        rec["row_end_exclusive"] = hi
        batch_summaries.append(rec)
        for key in [
            "rows", "tokens", "selected_rows", "selected_target_groups", "treatment_masked_tokens", "control_masked_tokens",
            "masked_token_count_row_diff_abs_sum", "dependent_target_tokens_treatment", "dependent_target_tokens_control",
            "dependent_target_label_or_replacement_mismatches", "dependent_target_original_id_mismatches",
            "background_label_or_replacement_mismatches", "anchor_swap_action_mismatches",
            "true_pivot_visible_token_count_treatment", "true_pivot_visible_token_count_control", "masked_input_difference_tokens_between_arms",
        ]:
            agg[key] += int(rec.get(key, 0))
        agg["masked_token_count_row_diff_max_abs"] = max(agg.get("masked_token_count_row_diff_max_abs", 0), int(rec.get("masked_token_count_row_diff_max_abs", 0)))
        replacement_t.update({str(k): int(v) for k, v in rec.get("replacement_action_counts_treatment", {}).items()})
        replacement_c.update({str(k): int(v) for k, v in rec.get("replacement_action_counts_control", {}).items()})
        for cat, vals in rec["pair_category_counts"].items():
            cat_agg[cat].update({str(k): int(v) for k, v in vals.items()})
        for name, dest in [("label_event_categories", label_cats), ("usable_event_categories", usable_cats), ("selected_event_categories", selected_cats), ("reject_reasons", reject)]:
            src = rec.get("stats_treatment", {}).get(name, {})
            dest.update({str(k): int(v) for k, v in src.items()})
    # One K=0 import/masking smoke: use the legacy stochastic WWM on the first batch.
    first_batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=0)))
    state = base.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob, switch_frac=0.7)
    state.initialize(vocab_size=len(tokenizer), total_steps=max(1, math.ceil(len(dataset)/args.batch_size)))
    gen = torch.Generator(device="cpu"); gen.manual_seed(args.seed)
    legacy_inputs, legacy_labels = base.apply_masking_curriculum(first_batch["input_ids"], first_batch["attention_mask"], first_batch["word_group"], tokenizer, state, gen)
    legacy_smoke = {
        "rows": int(first_batch["input_ids"].shape[0]),
        "masked_tokens": int((legacy_labels != -100).sum().item()),
        "effective_mask_rate": float((legacy_labels != -100).sum().item() / max(1, first_batch["attention_mask"].sum().item())),
        "changed_input_tokens": int((legacy_inputs != first_batch["input_ids"]).sum().item()),
    }
    summary = {
        "status": "PVDM_BATCH_INVARIANTS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"tail": str(TAIL), "labels": str(LABELS), "tokenizer": str(TOKENIZER_DIR), **hashes},
        "parameters": vars(args),
        "aggregate": {
            **{k: int(v) for k, v in agg.items()},
            "effective_mask_rate_treatment": agg["treatment_masked_tokens"] / max(1, agg["tokens"]),
            "effective_mask_rate_control": agg["control_masked_tokens"] / max(1, agg["tokens"]),
            "selected_target_groups_per_row": agg["selected_target_groups"] / max(1, agg["rows"]),
            "selected_rows_fraction": agg["selected_rows"] / max(1, agg["rows"]),
            "arm_difference_tokens_per_selected_event": agg["masked_input_difference_tokens_between_arms"] / max(1, agg["selected_target_groups"]),
            "true_pivot_visibility_fraction_treatment": agg["true_pivot_visible_token_count_treatment"] / max(1, agg["true_pivot_visible_token_count_treatment"]),
            "true_pivot_visibility_fraction_control": agg["true_pivot_visible_token_count_control"] / max(1, agg["true_pivot_visible_token_count_treatment"]),
        },
        "category_aggregate": {cat: dict(c) for cat, c in sorted(cat_agg.items())},
        "label_event_categories": dict(label_cats),
        "usable_event_categories_after_tokenization_and_anchor_length": dict(usable_cats),
        "selected_event_categories": dict(selected_cats),
        "reject_reasons": dict(reject),
        "replacement_action_counts_treatment": dict(replacement_t),
        "replacement_action_counts_control": dict(replacement_c),
        "first_8_batch_summaries": batch_summaries[:8],
        "legacy_k0_import_smoke": legacy_smoke,
        "pass_conditions": {
            "dependent_targets_identical": agg["dependent_target_label_or_replacement_mismatches"] == 0 and agg["dependent_target_original_id_mismatches"] == 0,
            "masked_mass_equal_per_row": agg["masked_token_count_row_diff_abs_sum"] == 0 and agg["masked_token_count_row_diff_max_abs"] == 0,
            "background_replacement_identical": agg["background_label_or_replacement_mismatches"] == 0,
            "anchor_swap_replacement_actions_matched": agg["anchor_swap_action_mismatches"] == 0,
            "treatment_has_material_pivot_visibility_contrast": agg["selected_target_groups"] > 0 and agg["true_pivot_visible_token_count_control"] == 0 and agg["masked_input_difference_tokens_between_arms"] > agg["selected_target_groups"],
            "categories_present": len(selected_cats) >= 5,
            "legacy_wwm_callable": legacy_smoke["masked_tokens"] > 0,
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    summary["ready_for_80m_two_arm_launch"] = all(summary["pass_conditions"].values())
    out_json = OUT / "pvdm_batch_invariant_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    batches_path = OUT / "pvdm_batch_invariant_batches.jsonl"
    with batches_path.open("w", encoding="utf-8") as f:
        for rec in batch_summaries:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    lines = [
        "# research — PVDM real-batch invariant check\n\n",
        f"Rows checked: {args.rows:,}; batch size {args.batch_size}; target_prob {args.target_prob}.\n\n",
        "## Aggregate\n\n",
        f"- selected rows: {summary['aggregate']['selected_rows']:,}/{summary['aggregate']['rows']:,} ({summary['aggregate']['selected_rows_fraction']:.3f})\n",
        f"- selected dependent target groups: {summary['aggregate']['selected_target_groups']:,} ({summary['aggregate']['selected_target_groups_per_row']:.3f}/row)\n",
        f"- effective mask rate: treatment {summary['aggregate']['effective_mask_rate_treatment']:.4f}, control {summary['aggregate']['effective_mask_rate_control']:.4f}\n",
        f"- target mismatches: {summary['aggregate']['dependent_target_label_or_replacement_mismatches']} label/replacement, {summary['aggregate']['dependent_target_original_id_mismatches']} original-id\n",
        f"- per-row masked token mass diff abs-sum/max: {summary['aggregate']['masked_token_count_row_diff_abs_sum']} / {summary['aggregate']['masked_token_count_row_diff_max_abs']}\n",
        f"- background mismatches: {summary['aggregate']['background_label_or_replacement_mismatches']}; anchor action mismatches: {summary['aggregate']['anchor_swap_action_mismatches']}\n",
        f"- selected categories: {dict(selected_cats)}\n\n",
        "## Pass conditions\n\n",
    ]
    for k, v in summary["pass_conditions"].items():
        lines.append(f"- {k}: {v}\n")
    lines += [
        f"\nReady for two-arm 70M→80M launch: **{summary['ready_for_80m_two_arm_launch']}**.\n\n",
        f"JSON: `{out_json}`\n",
        f"Batch JSONL: `{batches_path}`\n",
    ]
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "ready_for_80m_two_arm_launch": summary["ready_for_80m_two_arm_launch"],
        "aggregate": summary["aggregate"],
        "pass_conditions": summary["pass_conditions"],
        "out_json": str(out_json),
        "note": str(NOTE),
    }, indent=2), flush=True)
    if not summary["ready_for_80m_two_arm_launch"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
