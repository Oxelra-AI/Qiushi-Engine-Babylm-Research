#!/usr/bin/env python3
"""research sequence-curriculum loop measurement and faithful chunking prototype.

This script is CPU-only.  It does not train a model and does not inspect official
evaluation text.  It answers two concrete research questions raised by research:

1. In the actual research/research training-loop path, what enters the loss at an
   L64/L128/L256 step, and what word count is debited?
2. What would a compliant stage-length word-boundary chunk stream look like if
   each 10M-word epoch exposes every training word once while using inverse
   row-batch scaling for 64/128/256?

The result is meant to make a future 64->256 experiment scientifically clean:
current prefix slicing should not be confused with the public leader's sequence
factor if it hides suffix words while charging them.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
WORKSPACE = STUDY
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402

OUT_DIR = WORKSPACE / "data" / "sequence_curriculum_loop_measurement"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/sequence_curriculum_loop_measurement.md')
TRAIN_10M = USER_ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
TRAIN_100M = USER_ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
EXPECTED_10M_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_100M_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
TOKENIZERS = {
    "legal40k": WORKSPACE / "data" / "legal_representation_route_map" / "tokenizers" / "legal_byte_bpe_40k",
    "minfreq25": WORKSPACE / "data" / "tokenizer_support_spectrum" / "tokenizers" / "legal_byte_bpe_40k_minfreq25",
}
LENGTHS = [64, 128, 256]
SCHEDULES = {
    "64x3_128x4_256x3": {64: 3, 128: 4, 256: 3},
    "64x7_256x3": {64: 7, 128: 0, 256: 3},
}
WORD_RE = re.compile(r"\S+")
BASE_ROWS_AT_256 = 256
TRAIN_RNG_SEED = 43023


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


def read_first_examples(path: Path, n_rows: int) -> list[Any]:
    examples: list[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word-count mismatch in {path} line {line_no}")
            examples.append(base.Example(text=text, words=words, example_id=int(obj.get("example_id", line_no - 1)), source=str(obj.get("source", "example_jsonl"))))
            if len(examples) >= n_rows:
                break
    if len(examples) != n_rows:
        raise RuntimeError(f"needed {n_rows} rows from {path}, got {len(examples)}")
    return examples


def combine_microbatches(micro_batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.cat([b["input_ids"] for b in micro_batches], dim=0),
        "attention_mask": torch.cat([b["attention_mask"] for b in micro_batches], dim=0),
        "word_group": torch.cat([b["word_group"] for b in micro_batches], dim=0),
        "words": torch.cat([b["words"] for b in micro_batches], dim=0),
    }


def first_effective_batch(examples: list[Any], tokenizer) -> dict[str, torch.Tensor]:
    dataset = base.MaskedChunkDataset(examples, tokenizer, 256)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=base.collate, num_workers=0)
    micros: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        micros.append(batch)
        if len(micros) == 4:
            break
    if len(micros) != 4:
        raise RuntimeError(f"expected 4 microbatches, got {len(micros)}")
    return combine_microbatches(micros)


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
    examples = read_first_examples(TRAIN_100M, BASE_ROWS_AT_256)
    batch = first_effective_batch(examples, tokenizer)
    charged_words = int(batch["words"].sum().item())
    full_attention = batch["attention_mask"]
    full_word_group = batch["word_group"]
    full_tokens = int(full_attention.bool().sum().item())
    full_group_counts = groups_per_row(full_word_group)
    full_groups = sum(full_group_counts)
    rows = int(batch["input_ids"].shape[0])
    state = base.MaskingCurriculumState(
        curriculum="wwm_fixed",
        mask_prob_start=0.15,
        mask_prob_end=0.15,
        switch_frac=0.7,
        amlm_window=10,
        amlm_lambda=0.2,
    )
    # For research, only current_step matters, but use the actual 100M 256-row step count.
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
        "measurement_path_matches_step65_lines": "research lines 731-747 pop full batch words before slicing input_ids/attention_mask/word_group to cur_len; this record uses the same dataset, 64-row microbatches, 4-microbatch combine, and WWM mask function, without model forward/backward.",
        "rng_note": "Visibility and word debit are deterministic. Mask sample uses the same algorithm and seed on CPU; GPU random placement in a real trainer may differ, but the visibility/debit result does not depend on random masking.",
        "by_length": by_length,
    }


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def find_word_index(starts: list[int], spans: list[tuple[int, int]], s: int, e: int) -> int | None:
    if e <= s or not spans:
        return None
    idx = bisect.bisect_right(starts, s) - 1
    candidates = [idx, idx + 1]
    for j in candidates:
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
    for s0, e0 in offsets:
        s, e = int(s0), int(e0)
        idx = find_word_index(starts, spans, s, e)
        if idx is None and 0 <= s <= e <= len(text) and text[s:e].strip() == "":
            # Some byte-BPE tokenizers emit a standalone leading-space token (e.g. "Ġ")
            # with an offset covering only the inter-word blank before numbers or
            # unusual continuations.  The trainer counts that non-special token as a
            # visible candidate.  For faithful word-boundary chunk accounting, attach
            # it to the following whitespace word so active-token totals match the LM
            # input rather than silently dropping visible space tokens.
            nxt = bisect.bisect_left(starts, e)
            if 0 <= nxt < len(spans):
                idx = nxt
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
            # Charge the whitespace word by attaching it to the current chunk; if no
            # chunk exists, it will start a zero-token chunk that immediately accepts
            # following words. This case should be vanishingly rare for real text.
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
                next_tokens = prefix_cum_tokens + wc
                for L in LENGTHS:
                    # A word is counted as visible if at least one of its tokens would
                    # lie inside the current prefix. Zero-token words are counted if the
                    # current prefix has not yet exhausted the row.
                    if wc == 0:
                        if prefix_cum_tokens < L:
                            prefix_visible_by_L[L] += 1
                    elif prefix_cum_tokens < L:
                        prefix_visible_by_L[L] += 1
                prefix_cum_tokens = next_tokens
            for L in LENGTHS:
                per_len[L]["prefix_active_tokens"] += min(raw_tokens, L)
                per_len[L]["prefix_visible_words"] += prefix_visible_by_L[L]
                ch, ch_words, ch_tokens, overlong, zero_words, chunk_word_counts = chunk_counts_for_length(counts, L)
                per_len[L]["faithful_chunks"] += ch
                per_len[L]["faithful_words_charged"] += ch_words
                per_len[L]["faithful_active_tokens"] += ch_tokens
                per_len[L]["faithful_overlong_words"] += overlong
                per_len[L]["faithful_zero_token_words"] += zero_words
                per_len[L]["chunk_word_counts"].extend(chunk_word_counts)
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
                print(json.dumps({"event": "progress", "tokenizer": label, "rows": rows, "words": total_words}), flush=True)

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

    sched_csv = OUT_DIR / "faithful_chunking_by_schedule.csv"
    with sched_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tokenizer", "schedule", "charged_words_total", "prefix_active_tokens_total",
            "faithful_active_tokens_total", "target_token_ratio", "prefix_steps_total",
            "faithful_steps_total", "step_ratio",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, rec in result["faithful_streaming_chunking"].items():
            for name, r in rec["schedules"].items():
                w.writerow({
                    "tokenizer": label,
                    "schedule": name,
                    "charged_words_total": r["charged_words_total"],
                    "prefix_active_tokens_total": r["prefix_active_tokens_total"],
                    "faithful_active_tokens_total": r["faithful_active_tokens_total"],
                    "target_token_ratio": r["faithful_active_token_ratio_vs_prefix"],
                    "prefix_steps_total": r["prefix_optimizer_steps_total"],
                    "faithful_steps_total": r["faithful_optimizer_steps_total_inverse_batch"],
                    "step_ratio": r["faithful_step_ratio_vs_prefix"],
                })


def write_note(result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research — Sequence-curriculum loop measurement")
    lines.append("")
    lines.append("CPU-only measurement on compact_view_reinvest. No model was trained or evaluated.")
    lines.append("")
    lines.append("## Direct current-loop measurement")
    lines.append("")
    for label, rec in result["current_loop_first_step"].items():
        r64 = rec["by_length"]["64"]
        r128 = rec["by_length"]["128"]
        r256 = rec["by_length"]["256"]
        lines.append(f"### {label}")
        lines.append(f"- First effective batch: {rec['first_effective_batch_rows']} rows, {rec['first_effective_batch_debited_words']} words debited.")
        lines.append(f"- L64 sees {r64['visible_word_groups_entering_loss_candidates']} of {r64['full256_word_groups_in_same_rows']} full-256 word groups ({r64['visible_word_group_fraction_vs_full256']:.3f}) while debiting the full words; sampled WWM selected {r64['selected_word_groups_sampled_wwm015_cpu_rng']} groups / {r64['masked_tokens_sampled_wwm015_cpu_rng']} tokens.")
        lines.append(f"- L128 sees {r128['visible_word_group_fraction_vs_full256']:.3f} of full-256 word groups; L256 sees {r256['visible_word_group_fraction_vs_full256']:.3f}.")
        lines.append("")
    lines.append("This directly confirms the code reading: the current trainer debits `batch.words` before slicing to the stage length, so suffix word groups hidden by L64/L128 still consume word exposure.")
    lines.append("")
    lines.append("## Compliant faithful chunking prototype")
    lines.append("")
    lines.append("Regime used here: each stage epoch partitions the same 10M whitespace words into word-boundary chunks at the stage length; every word is charged once per epoch. A 10-epoch 64/128/256 curriculum remains 100M charged-word exposure.")
    lines.append("")
    for label, rec in result["faithful_streaming_chunking"].items():
        lines.append(f"### {label}")
        lines.append(f"- Untruncated tokenization: {rec['raw_tokens_untruncated']} tokens / {rec['words']} words = {rec['raw_tokens_per_word']:.4f} tokens per word; unassigned offsets {rec['unassigned_token_offsets']}.")
        for name, sched in rec["schedules"].items():
            lines.append(f"- Schedule {name}: faithful chunking gives {sched['faithful_active_token_ratio_vs_prefix']:.3f}x active tokens at {sched['faithful_step_ratio_vs_prefix']:.3f}x optimizer steps, with {sched['charged_words_total']} charged words.")
        r64 = rec["by_length"]["64"]
        r128 = rec["by_length"]["128"]
        r256 = rec["by_length"]["256"]
        lines.append(f"- Per epoch hidden-by-prefix words: L64 {r64['prefix_hidden_whitespace_word_fraction']:.3f}, L128 {r128['prefix_hidden_whitespace_word_fraction']:.3f}, L256 {r256['prefix_hidden_whitespace_word_fraction']:.3f}.")
        lines.append(f"- Faithful chunks per epoch: L64 {r64['faithful_chunks_per_epoch']} with batch {r64['inverse_scaled_row_batch']}; L128 {r128['faithful_chunks_per_epoch']} with batch {r128['inverse_scaled_row_batch']}; L256 {r256['faithful_chunks_per_epoch']} with batch {r256['inverse_scaled_row_batch']}.")
        lines.append("")
    lines.append("## Route implication")
    lines.append("")
    lines.append("A future 64->256 route should not use the existing `seq_len_schedule` prefix path as the leader-style factor. The faithful version is a stage-specific chunk stream with inverse row-batch scaling and explicit charged-word accounting. It changes target-token exposure substantially while keeping the same 10M words per epoch; therefore it is a real training intervention and should be launched only after the current depth vector and A02 support-floor vector are read.")
    lines.append("")
    lines.append(f"JSON: `{rel(OUT_DIR / 'sequence_curriculum_loop_measurement.json')}`")
    lines.append(f"CSV: `{rel(OUT_DIR / 'current_loop_first_step_by_length.csv')}`, `{rel(OUT_DIR / 'faithful_chunking_by_length.csv')}`, `{rel(OUT_DIR / 'faithful_chunking_by_schedule.csv')}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    sha10 = sha256_file(TRAIN_10M)
    sha100 = sha256_file(TRAIN_100M)
    if sha10 != EXPECTED_10M_SHA:
        raise RuntimeError(f"10M SHA mismatch: {sha10}")
    if sha100 != EXPECTED_100M_SHA:
        raise RuntimeError(f"100M SHA mismatch: {sha100}")

    tokenizers = {label: base.make_portable_tokenizer(str(path)) for label, path in TOKENIZERS.items()}
    tokenizer_manifest = {
        label: {
            "path": rel(path),
            "tokenizer_json_sha256": sha256_file(path / "tokenizer.json"),
            "vocab_size": tok.vocab_size,
            "len_tokenizer": len(tok),
            "is_fast": bool(tok.is_fast),
            "mask_token_id": tok.mask_token_id,
        }
        for label, (path, tok) in zip(TOKENIZERS.keys(), [(TOKENIZERS[k], tokenizers[k]) for k in TOKENIZERS])
    }

    current_loop = {}
    streaming = {}
    for label, tok in tokenizers.items():
        print(json.dumps({"event": "current_loop_first_step_start", "tokenizer": label}), flush=True)
        current_loop[label] = current_loop_first_step_measurement(label, tok)
        print(json.dumps({"event": "streaming_chunk_measurement_start", "tokenizer": label}), flush=True)
        streaming[label] = streaming_chunk_measurement(label, tok)

    result = {
        "status": "SEQUENCE_CURRICULUM_LOOP_MEASUREMENT",
        "purpose": "Turn research code reading into a direct current-loop measurement and define a compliant faithful 64/128/256 chunk-stream regime without model training.",
        "input_files": {
            "train_10M": rel(TRAIN_10M),
            "train_10M_sha256": sha10,
            "train_100M": rel(TRAIN_100M),
            "train_100M_sha256": sha100,
        },
        "tokenizers": tokenizer_manifest,
        "current_loop_first_step": current_loop,
        "faithful_streaming_chunking": streaming,
        "conclusion": {
            "current_seq_len_schedule_confirmed_as_prefix_slicing_with_full_word_debit": True,
            "faithful_chunking_compliance_regime": "same 10M whitespace words per epoch, partitioned into stage-length word-boundary chunks; 10 epochs = 100M charged words",
            "expensive_work_status": "No new GPU work launched. Use this measurement only to design a future sequence-curriculum route after the running depth result and A02 support-floor evidence are read.",
        },
    }
    out_json = OUT_DIR / "sequence_curriculum_loop_measurement.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csvs(result)
    write_note(result)
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "note": rel(NOTE),
        "legal40k_current_L64_visible_group_fraction": current_loop["legal40k"]["by_length"]["64"]["visible_word_group_fraction_vs_full256"],
        "legal40k_schedule_64x3_128x4_256x3_target_ratio": streaming["legal40k"]["schedules"]["64x3_128x4_256x3"]["faithful_active_token_ratio_vs_prefix"],
        "minfreq25_schedule_64x3_128x4_256x3_target_ratio": streaming["minfreq25"]["schedules"]["64x3_128x4_256x3"]["faithful_active_token_ratio_vs_prefix"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
