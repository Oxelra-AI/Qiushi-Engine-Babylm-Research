#!/usr/bin/env python3
"""Direct sequence-curriculum exposure measurement.

CPU-only measurement. It does not train a model and does not read official
evaluation text. It answers whether the existing local seq_len_schedule path
actually exposes all charged training words at short sequence lengths, and it
quantifies a compliant word-boundary chunking alternative for the exact
compact-view-reinvest corpus and tokenizers.

Scientific use:
- If the existing schedule slices 256-token rows to L=64/128 while charging
  the full row words, any future run using that path is not a clean test of
  the public leader's 64->256 sequence factor.
- A faithful alternative should partition the same <=10M words per epoch into
  stage-length chunks and debit only the chunk words, preserving 100M total
  charged-word exposure across 10 epochs while revealing suffix words that
  prefix slicing hides.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402

OUT_DIR = WORKSPACE / "data" / "sequence_curriculum_loop_measurement"
NOTE = (USER_ROOT / 'research/notes/frontier_consolidation/sequence_curriculum_loop_measurement.md')
TRAIN_10M = WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
TRAIN_100M = WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
EXPECTED_10M_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_100M_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
TOKENIZERS = {
    "legal16k": WORKSPACE / "data" / "compliant_tokenizer",
    "minfreq50_supportfloor": WORKSPACE / "data" / "supportfloor_tokenizers" / "legal_byte_bpe_40k_minfreq50",
}
LENGTHS = [64, 128, 256]
SCHEDULES = {
    "64x3_128x4_256x3": {64: 3, 128: 4, 256: 3},
    "64x7_256x3": {64: 7, 128: 0, 256: 3},
}
WORD_RE = re.compile(r"\S+")
BASE_ROWS_AT_256 = 256
TRAIN_RNG_SEED = 43023
MASK_PROB = 0.15


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}


def read_examples(path: Path, n_rows: int) -> list[Any]:
    examples: list[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word-count mismatch in {path} line {line_no}: field={words} actual={len(text.split())}")
            examples.append(base.Example(text=text, words=words, example_id=int(obj.get("example_id", line_no - 1)), source=str(obj.get("source", "example_jsonl"))))
            if len(examples) >= n_rows:
                break
    if len(examples) != n_rows:
        raise RuntimeError(f"needed {n_rows} rows from {path}, got {len(examples)}")
    return examples


def first_training_batch(examples: list[Any], tokenizer) -> dict[str, torch.Tensor]:
    dataset = base.MaskedChunkDataset(examples, tokenizer, 256)
    loader = DataLoader(dataset, batch_size=BASE_ROWS_AT_256, shuffle=False, collate_fn=base.collate, num_workers=0)
    return next(iter(loader))


def groups_per_row(word_group: torch.Tensor) -> list[int]:
    counts: list[int] = []
    for row in word_group:
        valid = row[row >= 0]
        counts.append(int(torch.unique(valid).numel()) if valid.numel() else 0)
    return counts


def selected_groups_per_row(word_group: torch.Tensor, labels: torch.Tensor) -> list[int]:
    counts: list[int] = []
    selected = labels != -100
    for row_g, row_s in zip(word_group, selected):
        valid = row_g[row_s & (row_g >= 0)]
        counts.append(int(torch.unique(valid).numel()) if valid.numel() else 0)
    return counts


def quantiles(xs: list[int] | list[float]) -> dict[str, float]:
    if not xs:
        return {"min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    ys = sorted(float(x) for x in xs)

    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        idx = p * (len(ys) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return ys[lo]
        return ys[lo] * (hi - idx) + ys[hi] * (idx - lo)

    return {"min": ys[0], "p10": q(0.10), "median": q(0.50), "p90": q(0.90), "max": ys[-1], "mean": sum(ys) / len(ys)}


def current_loop_first_step_measurement(label: str, tokenizer) -> dict[str, Any]:
    examples = read_examples(TRAIN_100M, BASE_ROWS_AT_256)
    batch = first_training_batch(examples, tokenizer)
    charged_words = int(batch["words"].sum().item())
    full_attention = batch["attention_mask"]
    full_word_group = batch["word_group"]
    full_tokens = int(full_attention.bool().sum().item())
    full_group_counts = groups_per_row(full_word_group)
    full_groups = sum(full_group_counts)
    rows = int(batch["input_ids"].shape[0])
    state = base.MaskingCurriculumState(
        curriculum="wwm_fixed",
        mask_prob_start=MASK_PROB,
        mask_prob_end=MASK_PROB,
        switch_frac=0.7,
        amlm_window=10,
        amlm_lambda=0.2,
    )
    state.initialize(vocab_size=len(tokenizer), total_steps=math.ceil(647_400 / BASE_ROWS_AT_256))
    by_length: dict[str, Any] = {}
    for L in LENGTHS:
        input_ids = batch["input_ids"][:, :L].contiguous()
        attention_mask = batch["attention_mask"][:, :L].contiguous()
        word_group = batch["word_group"][:, :L].contiguous()
        visible_group_counts = groups_per_row(word_group)
        visible_groups = sum(visible_group_counts)
        visible_tokens = int(attention_mask.bool().sum().item())
        state.current_step = 0
        gen = torch.Generator(device="cpu")
        gen.manual_seed(TRAIN_RNG_SEED)
        _masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, state, gen)
        masked_tokens = int((labels != -100).sum().item())
        selected_group_counts = selected_groups_per_row(word_group, labels)
        selected_groups = sum(selected_group_counts)
        by_length[str(L)] = {
            "rows": rows,
            "charged_words_debited_by_current_loop": charged_words,
            "visible_tokens_entering_loss_candidates": visible_tokens,
            "full256_tokens_in_same_rows": full_tokens,
            "visible_token_fraction_vs_full256": visible_tokens / max(1, full_tokens),
            "visible_word_groups_entering_loss_candidates": visible_groups,
            "full256_word_groups_in_same_rows": full_groups,
            "visible_word_group_fraction_vs_full256": visible_groups / max(1, full_groups),
            "word_groups_hidden_but_charged_vs_full256": full_groups - visible_groups,
            "hidden_group_fraction_vs_full256": (full_groups - visible_groups) / max(1, full_groups),
            "masked_tokens_sampled_wwm015_cpu_rng": masked_tokens,
            "selected_word_groups_sampled_wwm015_cpu_rng": selected_groups,
            "selected_groups_per_visible_group": selected_groups / max(1, visible_groups),
            "masked_tokens_per_visible_token": masked_tokens / max(1, visible_tokens),
            "visible_groups_per_debited_word": visible_groups / max(1, charged_words),
            "visible_tokens_per_debited_word": visible_tokens / max(1, charged_words),
            "row_visible_group_quantiles": quantiles(visible_group_counts),
            "row_selected_group_quantiles": quantiles(selected_group_counts),
        }
    return {
        "tokenizer": label,
        "first_effective_batch_rows": rows,
        "first_effective_batch_debited_words": charged_words,
        "first_effective_batch_full256_tokens": full_tokens,
        "first_effective_batch_full256_word_groups": full_groups,
        "measurement_path_matches_trainer_lines": "COMPACT_EXPERIENCE masking_curriculum_trainer lines 852-863 pop full batch words before slicing input_ids/attention_mask/word_group to cur_len; this record instantiates the same dataset, collate, first batch, and WWM mask function without model forward/backward.",
        "rng_note": "Visibility and word debit are deterministic. Mask sample uses the same algorithm and seed on CPU; GPU random placement in a real trainer may differ, but the visibility/debit result does not depend on random masking.",
        "by_length": by_length,
    }


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def find_word_index(starts: list[int], spans: list[tuple[int, int]], s: int, e: int) -> int | None:
    if e <= s or not spans:
        return None
    idx = bisect.bisect_right(starts, s) - 1
    for j in (idx, idx + 1):
        if 0 <= j < len(spans):
            a, b = spans[j]
            if s < b and e > a:
                return j
    mid = (s + e - 1) // 2
    idx = bisect.bisect_right(starts, mid) - 1
    if 0 <= idx < len(spans):
        a, b = spans[idx]
        if s < b and e > a:
            return idx
    return None


def token_counts_by_whitespace_word(text: str, tokenizer) -> tuple[list[int], int, int]:
    spans = word_spans(text)
    if not spans:
        return [], 0, 0
    starts = [s for s, _ in spans]
    enc = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping") or []
    counts = [0 for _ in spans]
    unassigned = 0
    for s, e in offsets:
        idx = find_word_index(starts, spans, int(s), int(e))
        if idx is None:
            unassigned += 1
        else:
            counts[idx] += 1
    return counts, len(offsets), unassigned


def chunk_counts_for_length(word_token_counts: list[int], L: int) -> tuple[int, int, int, int, int, list[int]]:
    chunks = 0
    total_words = 0
    total_tokens = 0
    overlong_words = 0
    zero_token_words = 0
    chunk_word_counts: list[int] = []
    cur_words = 0
    cur_tokens = 0
    for tok_count in word_token_counts:
        if tok_count <= 0:
            zero_token_words += 1
        if tok_count > L:
            overlong_words += 1
            if cur_words:
                chunks += 1
                chunk_word_counts.append(cur_words)
                total_words += cur_words
                total_tokens += cur_tokens
                cur_words = 0
                cur_tokens = 0
            chunks += 1
            chunk_word_counts.append(1)
            total_words += 1
            total_tokens += tok_count
            continue
        if cur_words and cur_tokens + tok_count > L:
            chunks += 1
            chunk_word_counts.append(cur_words)
            total_words += cur_words
            total_tokens += cur_tokens
            cur_words = 0
            cur_tokens = 0
        cur_words += 1
        cur_tokens += tok_count
    if cur_words:
        chunks += 1
        chunk_word_counts.append(cur_words)
        total_words += cur_words
        total_tokens += cur_tokens
    return chunks, total_words, total_tokens, overlong_words, zero_token_words, chunk_word_counts


def streaming_chunk_measurement(label: str, tokenizer) -> dict[str, Any]:
    per_len = {L: {
        "prefix_active_tokens": 0,
        "prefix_visible_words": 0,
        "faithful_chunks": 0,
        "faithful_words_charged": 0,
        "faithful_active_tokens": 0,
        "faithful_overlong_words": 0,
        "faithful_zero_token_words": 0,
        "chunk_word_counts": [],
    } for L in LENGTHS}
    rows = 0
    total_words = 0
    total_raw_tokens = 0
    total_unassigned_offsets = 0
    row_token_len_samples: list[int] = []
    row_word_samples: list[int] = []
    sample_chunks: dict[str, list[dict[str, Any]]] = {str(L): [] for L in LENGTHS}

    with TRAIN_10M.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word-count mismatch at row {rows + 1}: field={words} actual={len(text.split())}")
            counts, raw_tokens, unassigned = token_counts_by_whitespace_word(text, tokenizer)
            if len(counts) != words:
                raise RuntimeError(f"whitespace span count mismatch at row {rows + 1}: {len(counts)} vs {words}")
            rows += 1
            total_words += words
            total_raw_tokens += raw_tokens
            total_unassigned_offsets += unassigned
            if len(row_token_len_samples) < 2000:
                row_token_len_samples.append(raw_tokens)
                row_word_samples.append(words)

            prefix_cum_tokens = 0
            prefix_visible_by_L = {L: 0 for L in LENGTHS}
            for wc in counts:
                for L in LENGTHS:
                    if wc == 0:
                        if prefix_cum_tokens < L:
                            prefix_visible_by_L[L] += 1
                    elif prefix_cum_tokens < L:
                        prefix_visible_by_L[L] += 1
                prefix_cum_tokens += wc
            for L in LENGTHS:
                stats = per_len[L]
                stats["prefix_active_tokens"] += min(raw_tokens, L)
                stats["prefix_visible_words"] += prefix_visible_by_L[L]
                ch, ch_words, ch_tokens, overlong, zero_words, chunk_word_counts = chunk_counts_for_length(counts, L)
                stats["faithful_chunks"] += ch
                stats["faithful_words_charged"] += ch_words
                stats["faithful_active_tokens"] += ch_tokens
                stats["faithful_overlong_words"] += overlong
                stats["faithful_zero_token_words"] += zero_words
                stats["chunk_word_counts"].extend(chunk_word_counts)
                if len(sample_chunks[str(L)]) < 5:
                    sample_chunks[str(L)].append({
                        "row_index": rows - 1,
                        "source": obj.get("source", ""),
                        "row_words": words,
                        "row_tokens": raw_tokens,
                        "chunks_from_this_row": ch,
                        "chunk_word_counts_head": chunk_word_counts[:8],
                    })
            if rows % 10000 == 0:
                print(json.dumps({"event": "progress", "tokenizer": label, "rows": rows, "words": total_words, "elapsed_sec": round(time.time() - START, 1)}), flush=True)

    if total_words != 10_000_000:
        raise RuntimeError(f"expected 10M words, measured {total_words}")

    prefix_rows = rows
    prefix_steps_per_epoch = math.ceil(prefix_rows / BASE_ROWS_AT_256)
    result_by_len: dict[str, Any] = {}
    for L in LENGTHS:
        batch_rows = int(BASE_ROWS_AT_256 * (256 / L))
        stats = per_len[L]
        faithful_steps = math.ceil(int(stats["faithful_chunks"]) / batch_rows)
        prefix_active = int(stats["prefix_active_tokens"])
        faithful_active = int(stats["faithful_active_tokens"])
        prefix_visible_words = int(stats["prefix_visible_words"])
        result_by_len[str(L)] = {
            "stage_length": L,
            "inverse_scaled_row_batch": batch_rows,
            "prefix_rows_per_epoch_current_loop": prefix_rows,
            "prefix_steps_per_epoch_current_loop_batch256": prefix_steps_per_epoch,
            "prefix_words_debited_per_epoch": total_words,
            "prefix_active_tokens_per_epoch": prefix_active,
            "prefix_visible_whitespace_words_per_epoch": prefix_visible_words,
            "prefix_hidden_whitespace_words_but_debited": total_words - prefix_visible_words,
            "prefix_hidden_whitespace_word_fraction": (total_words - prefix_visible_words) / total_words,
            "faithful_chunks_per_epoch": int(stats["faithful_chunks"]),
            "faithful_words_charged_per_epoch": int(stats["faithful_words_charged"]),
            "faithful_active_tokens_per_epoch": faithful_active,
            "faithful_steps_per_epoch_inverse_batch": faithful_steps,
            "faithful_active_token_ratio_vs_prefix": faithful_active / max(1, prefix_active),
            "faithful_step_ratio_vs_prefix": faithful_steps / max(1, prefix_steps_per_epoch),
            "faithful_overlong_words": int(stats["faithful_overlong_words"]),
            "faithful_zero_token_words": int(stats["faithful_zero_token_words"]),
            "chunk_words_quantiles": quantiles(stats["chunk_word_counts"]),
            "sample_chunks": sample_chunks[str(L)],
        }
    schedule_results: dict[str, Any] = {}
    for name, counts_by_L in SCHEDULES.items():
        prefix_tokens = 0
        faithful_tokens = 0
        prefix_steps = 0
        faithful_steps = 0
        for L, epochs in counts_by_L.items():
            if epochs <= 0:
                continue
            r = result_by_len[str(L)]
            prefix_tokens += epochs * int(r["prefix_active_tokens_per_epoch"])
            faithful_tokens += epochs * int(r["faithful_active_tokens_per_epoch"])
            prefix_steps += epochs * int(r["prefix_steps_per_epoch_current_loop_batch256"])
            faithful_steps += epochs * int(r["faithful_steps_per_epoch_inverse_batch"])
        schedule_results[name] = {
            "epochs_by_length": counts_by_L,
            "charged_words_total": total_words * sum(counts_by_L.values()),
            "prefix_active_tokens_total": prefix_tokens,
            "faithful_active_tokens_total": faithful_tokens,
            "faithful_active_token_ratio_vs_prefix": faithful_tokens / max(1, prefix_tokens),
            "prefix_optimizer_steps_total": prefix_steps,
            "faithful_optimizer_steps_total_inverse_batch": faithful_steps,
            "faithful_step_ratio_vs_prefix": faithful_steps / max(1, prefix_steps),
        }
    return {
        "tokenizer": label,
        "rows": rows,
        "words": total_words,
        "raw_tokens_untruncated": total_raw_tokens,
        "raw_tokens_per_word": total_raw_tokens / total_words,
        "unassigned_token_offsets": total_unassigned_offsets,
        "row_token_len_sample_quantiles": quantiles(row_token_len_samples),
        "row_word_sample_quantiles": quantiles(row_word_samples),
        "by_length": result_by_len,
        "schedules": schedule_results,
        "compliance_regime": "Each stage epoch partitions the same 10M training words into word-boundary chunks at the stage length; every whitespace word is charged exactly once per epoch, so a 10-epoch 64/128/256 curriculum remains 100M charged-word exposure. Extra target tokens come from revealing suffix words that prefix slicing hid, not from adding new words or extra epochs.",
    }


def write_csvs(result: dict[str, Any]) -> None:
    loop_csv = OUT_DIR / "current_loop_first_step_by_length.csv"
    with loop_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tokenizer", "length", "rows", "charged_words", "visible_tokens", "full256_tokens",
            "visible_token_fraction", "visible_word_groups", "full256_word_groups", "hidden_group_fraction",
            "masked_tokens_sample", "selected_word_groups_sample", "visible_groups_per_debited_word",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, rec in result["current_loop_first_step"].items():
            for L, r in rec["by_length"].items():
                w.writerow({
                    "tokenizer": label,
                    "length": L,
                    "rows": r["rows"],
                    "charged_words": r["charged_words_debited_by_current_loop"],
                    "visible_tokens": r["visible_tokens_entering_loss_candidates"],
                    "full256_tokens": r["full256_tokens_in_same_rows"],
                    "visible_token_fraction": r["visible_token_fraction_vs_full256"],
                    "visible_word_groups": r["visible_word_groups_entering_loss_candidates"],
                    "full256_word_groups": r["full256_word_groups_in_same_rows"],
                    "hidden_group_fraction": r["hidden_group_fraction_vs_full256"],
                    "masked_tokens_sample": r["masked_tokens_sampled_wwm015_cpu_rng"],
                    "selected_word_groups_sample": r["selected_word_groups_sampled_wwm015_cpu_rng"],
                    "visible_groups_per_debited_word": r["visible_groups_per_debited_word"],
                })

    stage_csv = OUT_DIR / "faithful_chunking_by_length.csv"
    with stage_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tokenizer", "length", "raw_tokens_per_word", "prefix_hidden_word_fraction",
            "prefix_active_tokens_per_epoch", "faithful_active_tokens_per_epoch", "target_token_ratio",
            "prefix_steps_per_epoch", "faithful_steps_per_epoch", "step_ratio", "faithful_chunks_per_epoch",
            "inverse_scaled_row_batch", "faithful_words_charged_per_epoch", "overlong_words",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, rec in result["faithful_streaming_chunking"].items():
            for L, r in rec["by_length"].items():
                w.writerow({
                    "tokenizer": label,
                    "length": L,
                    "raw_tokens_per_word": rec["raw_tokens_per_word"],
                    "prefix_hidden_word_fraction": r["prefix_hidden_whitespace_word_fraction"],
                    "prefix_active_tokens_per_epoch": r["prefix_active_tokens_per_epoch"],
                    "faithful_active_tokens_per_epoch": r["faithful_active_tokens_per_epoch"],
                    "target_token_ratio": r["faithful_active_token_ratio_vs_prefix"],
                    "prefix_steps_per_epoch": r["prefix_steps_per_epoch_current_loop_batch256"],
                    "faithful_steps_per_epoch": r["faithful_steps_per_epoch_inverse_batch"],
                    "step_ratio": r["faithful_step_ratio_vs_prefix"],
                    "faithful_chunks_per_epoch": r["faithful_chunks_per_epoch"],
                    "inverse_scaled_row_batch": r["inverse_scaled_row_batch"],
                    "faithful_words_charged_per_epoch": r["faithful_words_charged_per_epoch"],
                    "overlong_words": r["faithful_overlong_words"],
                })

    sched_csv = OUT_DIR / "schedule_ratios.csv"
    with sched_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tokenizer", "schedule", "charged_words_total", "prefix_active_tokens_total",
            "faithful_active_tokens_total", "target_token_ratio", "prefix_optimizer_steps_total",
            "faithful_optimizer_steps_total", "step_ratio",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, rec in result["faithful_streaming_chunking"].items():
            for schedule, r in rec["schedules"].items():
                w.writerow({
                    "tokenizer": label,
                    "schedule": schedule,
                    "charged_words_total": r["charged_words_total"],
                    "prefix_active_tokens_total": r["prefix_active_tokens_total"],
                    "faithful_active_tokens_total": r["faithful_active_tokens_total"],
                    "target_token_ratio": r["faithful_active_token_ratio_vs_prefix"],
                    "prefix_optimizer_steps_total": r["prefix_optimizer_steps_total"],
                    "faithful_optimizer_steps_total": r["faithful_optimizer_steps_total_inverse_batch"],
                    "step_ratio": r["faithful_step_ratio_vs_prefix"],
                })


def write_note(result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research sequence-curriculum loop measurement\n")
    lines.append("CPU-only direct measurement on A02's exact compact-view-reinvest corpus. No model was trained or evaluated.\n")
    lines.append("\n## Inputs\n")
    lines.append(f"- 10M pool SHA matched: `{result['sha256']['train_10m_matches']}`; 100M train SHA matched: `{result['sha256']['train_100m_matches']}`.\n")
    lines.append(f"- Tokenizers measured: {', '.join(result['tokenizers'].keys())}.\n")
    lines.append("\n## Direct existing-loop first-step measurement\n")
    lines.append("The current local `seq_len_schedule` path constructs 256-token padded/truncated rows, pops and debits the full row word count, then slices tensors to the current length. The table shows the first actual 256-row batch under this path.\n\n")
    lines.append("| tokenizer | L | debited words | visible groups | full256 groups | hidden group frac | visible groups/debited word | selected groups sample | masked tokens sample |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, rec in result["current_loop_first_step"].items():
        for L in ["64", "128", "256"]:
            r = rec["by_length"][L]
            lines.append(
                f"| {label} | {L} | {r['charged_words_debited_by_current_loop']} | {r['visible_word_groups_entering_loss_candidates']} | {r['full256_word_groups_in_same_rows']} | {r['hidden_group_fraction_vs_full256']:.3f} | {r['visible_groups_per_debited_word']:.3f} | {r['selected_word_groups_sampled_wwm015_cpu_rng']} | {r['masked_tokens_sampled_wwm015_cpu_rng']} |\n"
            )
    lines.append("\n## Full-pool faithful chunking/accounting alternative\n")
    lines.append("A faithful stage-length regime partitions the same 10M words into word-boundary chunks for each stage length; every word is debited exactly once per epoch. Extra target tokens are suffix content that prefix slicing hides, not extra words.\n\n")
    lines.append("| tokenizer | L | prefix hidden word frac | faithful chunks/epoch | charged words/epoch | active-token ratio vs prefix | step ratio vs prefix | chunk words median/p90 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, rec in result["faithful_streaming_chunking"].items():
        for L in ["64", "128", "256"]:
            r = rec["by_length"][L]
            q = r["chunk_words_quantiles"]
            lines.append(
                f"| {label} | {L} | {r['prefix_hidden_whitespace_word_fraction']:.3f} | {r['faithful_chunks_per_epoch']} | {r['faithful_words_charged_per_epoch']} | {r['faithful_active_token_ratio_vs_prefix']:.3f} | {r['faithful_step_ratio_vs_prefix']:.3f} | {q['median']:.1f}/{q['p90']:.1f} |\n"
            )
    lines.append("\n## Ten-epoch schedule ratios\n")
    lines.append("| tokenizer | schedule | charged words | target-token ratio faithful/prefix | optimizer-step ratio faithful/prefix |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for label, rec in result["faithful_streaming_chunking"].items():
        for schedule, r in rec["schedules"].items():
            lines.append(
                f"| {label} | {schedule} | {r['charged_words_total']} | {r['faithful_active_token_ratio_vs_prefix']:.3f} | {r['faithful_step_ratio_vs_prefix']:.3f} |\n"
            )
    lines.append("\n## Scientific reading\n")
    lines.append("- A future sequence-curriculum run should not use the existing prefix-slicing path as evidence for a leader-style 64→256 curriculum: short stages hide a large suffix fraction while debiting full row words.\n")
    lines.append("- The compliant faithful-chunking regime is well-defined in word-budget terms: the same 10M words are partitioned differently, and a ten-epoch curriculum remains 100M charged words.\n")
    lines.append("- This measurement does not justify a GPU launch while the word-mean screen is unresolved. If sequence length becomes the next route, the trainer must implement the faithful chunk stream, not patch `seq_len_schedule` by slicing prefixes.\n")
    lines.append(f"\nFull JSON: `{rel(OUT_DIR / 'sequence_curriculum_loop_measurement.json')}`\n")
    lines.append(f"CSV files: `{rel(OUT_DIR / 'current_loop_first_step_by_length.csv')}`, `{rel(OUT_DIR / 'faithful_chunking_by_length.csv')}`, `{rel(OUT_DIR / 'schedule_ratios.csv')}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")


START = time.time()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    sha_10 = sha256_file(TRAIN_10M)
    sha_100 = sha256_file(TRAIN_100M)
    tokenizers = {label: base.make_portable_tokenizer(str(path)) for label, path in TOKENIZERS.items()}
    tokenizer_records = {label: {"path": rel(path), "len": len(tok), "vocab_size": tok.vocab_size, "is_fast": tok.is_fast, "tokenizer_json_sha256": sha256_file(path / "tokenizer.json")} for label, (path, tok) in zip(TOKENIZERS.keys(), zip(TOKENIZERS.values(), tokenizers.values()))}
    result: dict[str, Any] = {
        "status": "SEQUENCE_CURRICULUM_LOOP_MEASUREMENT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Directly measure current prefix-slicing seq_len_schedule debit/visibility and quantify compliant faithful word-boundary chunking for A02 exact corpus/tokenizers before any sequence-curriculum GPU work.",
        "inputs": {
            "train_10m": file_record(TRAIN_10M),
            "train_100m": file_record(TRAIN_100M),
            "trainer_source": rel(COMPACT_EXPERIENCE_SCRIPTS / "masking_curriculum_trainer.py"),
            "trainer_lines_of_interest": "Dataset tokenizes at max_seq_length=256; training loop lines 852-863 pop words, then slice tensors to cur_len; cumulative_words later adds full words.",
        },
        "sha256": {
            "train_10m_expected": EXPECTED_10M_SHA,
            "train_10m_actual": sha_10,
            "train_10m_matches": sha_10 == EXPECTED_10M_SHA,
            "train_100m_expected": EXPECTED_100M_SHA,
            "train_100m_actual": sha_100,
            "train_100m_matches": sha_100 == EXPECTED_100M_SHA,
        },
        "tokenizers": tokenizer_records,
        "current_loop_first_step": {},
        "faithful_streaming_chunking": {},
        "interpretation": {
            "current_prefix_path": "If L<256, actual tensors entering masking/loss are prefixes, but batch words are debited before slicing; this is visible in the direct batch measurement and the inherited training-loop code.",
            "faithful_compliant_path": "For each stage length, split the same 10M words into word-boundary chunks and debit chunk words exactly once; inverse row-batch scaling keeps token work comparable while avoiding hidden charged suffixes.",
            "launch_status": "No GPU training or evaluation is launched by this script. It is an implementation/attribution precondition for a possible future sequence route after the current word-mean and support-floor evidence are read.",
        },
    }
    if not result["sha256"]["train_10m_matches"] or not result["sha256"]["train_100m_matches"]:
        raise RuntimeError("training corpus SHA mismatch")
    for label, tok in tokenizers.items():
        result["current_loop_first_step"][label] = current_loop_first_step_measurement(label, tok)
    for label, tok in tokenizers.items():
        result["faithful_streaming_chunking"][label] = streaming_chunk_measurement(label, tok)
    result["elapsed_sec"] = round(time.time() - START, 3)
    out_json = OUT_DIR / "sequence_curriculum_loop_measurement.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csvs(result)
    write_note(result)
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(NOTE),
        "elapsed_sec": result["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
