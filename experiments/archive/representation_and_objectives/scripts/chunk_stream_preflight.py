#!/usr/bin/env python3
"""research chunk-stream construction check for experience-utilization arms.

CPU-only.  No model training, no official evaluation text, no GPU use.

This constructs the actual word-boundary chunk stream needed for the research
experience-utilization comparisons.  It verifies that charged words are neither
lost nor duplicated, that active token totals match the research measurement, and
that first optimizer batches can be formed with bounded microbatch accumulation.
"""
from __future__ import annotations

from dataclasses import dataclass
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

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
WORKSPACE = STUDY
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))
import masking_curriculum_trainer as base  # noqa: E402

TRAIN_10M = USER_ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
EXPECTED_10M_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
CURRICULUM_MEASUREMENTS = WORKSPACE / "data" / "sequence_curriculum_loop_measurement" / "sequence_curriculum_loop_measurement.json"
DESIGN = WORKSPACE / "data" / "experience_utilization_design" / "experience_utilization_experiment_design.json"
OUT_DIR = WORKSPACE / "data" / "chunk_stream_preflight"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/chunk_stream_preflight.md')
TOKENIZERS = {
    "legal40k": WORKSPACE / "data" / "legal_representation_route_map" / "tokenizers" / "legal_byte_bpe_40k",
    "minfreq25": WORKSPACE / "data" / "tokenizer_support_spectrum" / "tokenizers" / "legal_byte_bpe_40k_minfreq25",
}
LENGTHS = [64, 128, 256]
WORDS_PER_EPOCH = 10_000_000
ROWS_PER_EPOCH = 64_740
STEPS_PER_EPOCH = 253
EPOCHS_TOTAL = 10
BASE_MASK_PROB = 0.15
TRAIN_RNG_SEED = 43023
MICRO_BATCH_CHUNKS = 64
FIRST_CHUNK_COLLECT_LIMIT = {64: 1200, 128: 700, 256: 400}
WORD_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    input_ids: list[int]
    word_group: list[int]
    charged_words: int
    source_row: int
    row_words: int
    start_word: int
    end_word_exclusive: int
    overlong_continuation: bool = False


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


def token_ids_by_whitespace_word(text: str, tokenizer) -> tuple[list[list[int]], int, int, list[dict[str, Any]]]:
    spans = word_spans(text)
    starts = [s for s, _ in spans]
    enc = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = list(enc.get("offset_mapping") or [])
    if len(ids) != len(offsets):
        raise RuntimeError("tokenizer did not return one offset per input id")
    by_word: list[list[int]] = [[] for _ in spans]
    unassigned = 0
    unassigned_examples: list[dict[str, Any]] = []
    for tok_id, (s0, e0) in zip(ids, offsets):
        s, e = int(s0), int(e0)
        idx = find_word_index(starts, spans, s, e)
        if idx is None and 0 <= s <= e <= len(text) and text[s:e].strip() == "":
            # Byte-level BPE can emit a standalone space marker before the next
            # word.  The trainer counts it as a normal visible token, so attach
            # it to that next whitespace word.
            nxt = bisect.bisect_left(starts, e)
            if 0 <= nxt < len(spans):
                idx = nxt
        if idx is None:
            unassigned += 1
            if len(unassigned_examples) < 10:
                unassigned_examples.append({"token_id": int(tok_id), "offset": [s, e], "span_text": text[s:e]})
        else:
            by_word[idx].append(int(tok_id))
    return by_word, len(ids), unassigned, unassigned_examples


def chunks_from_word_tokens(word_tokens: list[list[int]], L: int, row_index: int, row_words: int) -> tuple[list[Chunk], dict[str, Any]]:
    chunks: list[Chunk] = []
    cur_ids: list[int] = []
    cur_groups: list[int] = []
    cur_words = 0
    cur_start_word: int | None = None
    overlong_words = 0
    continuation_chunks = 0
    zero_token_words = 0

    def flush(end_word: int) -> None:
        nonlocal cur_ids, cur_groups, cur_words, cur_start_word
        if cur_words or cur_ids:
            chunks.append(Chunk(
                input_ids=cur_ids,
                word_group=cur_groups,
                charged_words=cur_words,
                source_row=row_index,
                row_words=row_words,
                start_word=cur_start_word if cur_start_word is not None else end_word,
                end_word_exclusive=end_word,
            ))
        cur_ids = []
        cur_groups = []
        cur_words = 0
        cur_start_word = None

    for word_idx, ids in enumerate(word_tokens):
        if not ids:
            zero_token_words += 1
        if len(ids) > L:
            overlong_words += 1
            flush(word_idx)
            # Expose every token of the very rare overlong whitespace word.  The
            # word is charged once on the first piece; continuation pieces carry
            # zero extra charged words and are recorded explicitly.
            for part_start in range(0, len(ids), L):
                part = ids[part_start:part_start + L]
                charged = 1 if part_start == 0 else 0
                continuation = part_start > 0
                if continuation:
                    continuation_chunks += 1
                chunks.append(Chunk(
                    input_ids=part,
                    word_group=[0 for _ in part],
                    charged_words=charged,
                    source_row=row_index,
                    row_words=row_words,
                    start_word=word_idx,
                    end_word_exclusive=word_idx + 1,
                    overlong_continuation=continuation,
                ))
            continue
        if cur_ids and len(cur_ids) + len(ids) > L:
            flush(word_idx)
        if cur_start_word is None:
            cur_start_word = word_idx
        group_id = cur_words
        cur_ids.extend(ids)
        cur_groups.extend([group_id for _ in ids])
        cur_words += 1
    flush(len(word_tokens))
    return chunks, {
        "overlong_words": overlong_words,
        "overlong_continuation_chunks": continuation_chunks,
        "zero_token_words": zero_token_words,
    }


def batch_distribution(n_items: int, n_steps: int = STEPS_PER_EPOCH) -> dict[str, Any]:
    base_n = n_items // n_steps
    rem = n_items % n_steps
    return {
        "items": n_items,
        "steps": n_steps,
        "base_chunks_per_step": base_n,
        "steps_with_base_plus_one": rem,
        "steps_with_base": n_steps - rem,
        "first_step_chunks": base_n + (1 if rem else 0),
        "mean_chunks_per_step": n_items / n_steps,
        "max_accumulation_microbatches": math.ceil((base_n + (1 if rem else 0)) / MICRO_BATCH_CHUNKS),
    }


def collate_chunks(chunks: list[Chunk], L: int) -> dict[str, torch.Tensor]:
    n = len(chunks)
    input_ids = torch.zeros((n, L), dtype=torch.long)
    attention_mask = torch.zeros((n, L), dtype=torch.long)
    word_group = torch.full((n, L), -1, dtype=torch.long)
    words = torch.tensor([c.charged_words for c in chunks], dtype=torch.long)
    for i, ch in enumerate(chunks):
        if len(ch.input_ids) > L:
            raise RuntimeError(f"chunk length {len(ch.input_ids)} exceeds stage length {L}")
        if len(ch.input_ids) != len(ch.word_group):
            raise RuntimeError("input_ids and word_group length mismatch")
        m = len(ch.input_ids)
        if m:
            input_ids[i, :m] = torch.tensor(ch.input_ids, dtype=torch.long)
            attention_mask[i, :m] = 1
            word_group[i, :m] = torch.tensor(ch.word_group, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "word_group": word_group, "words": words}


def count_selected_groups(word_group: torch.Tensor, labels: torch.Tensor) -> int:
    total = 0
    selected = labels != -100
    for g, s in zip(word_group, selected):
        valid = g[s & (g >= 0)]
        total += int(torch.unique(valid).numel()) if valid.numel() else 0
    return total


def first_update_mask_measurement(label: str, tokenizer, chunks: list[Chunk], L: int, total_steps: int) -> dict[str, Any]:
    batch = collate_chunks(chunks, L)
    state = base.MaskingCurriculumState(
        curriculum="wwm_fixed",
        mask_prob_start=BASE_MASK_PROB,
        mask_prob_end=BASE_MASK_PROB,
        switch_frac=0.7,
        amlm_window=10,
        amlm_lambda=0.2,
    )
    state.initialize(vocab_size=len(tokenizer), total_steps=total_steps)
    state.current_step = 0
    gen = torch.Generator(device="cpu")
    gen.manual_seed(TRAIN_RNG_SEED)
    _masked, labels = base.apply_masking_curriculum(batch["input_ids"], batch["attention_mask"], batch["word_group"], tokenizer, state, gen)
    active_tokens = int(batch["attention_mask"].sum().item())
    charged_words = int(batch["words"].sum().item())
    masked_tokens = int((labels != -100).sum().item())
    selected_groups = count_selected_groups(batch["word_group"], labels)
    return {
        "tokenizer": label,
        "length": L,
        "chunks_in_first_update": len(chunks),
        "charged_words_in_first_update": charged_words,
        "active_tokens_in_first_update": active_tokens,
        "mean_active_tokens_per_chunk": active_tokens / max(1, len(chunks)),
        "masked_tokens_sampled_wwm015_cpu_rng": masked_tokens,
        "selected_word_groups_sampled_wwm015_cpu_rng": selected_groups,
        "masked_tokens_per_active_token": masked_tokens / max(1, active_tokens),
        "selected_groups_per_charged_word": selected_groups / max(1, charged_words),
        "microbatch_size": MICRO_BATCH_CHUNKS,
        "accumulation_microbatches": math.ceil(len(chunks) / MICRO_BATCH_CHUNKS),
        "nominal_token_slots": len(chunks) * L,
        "source_row_span": [chunks[0].source_row if chunks else None, chunks[-1].source_row if chunks else None],
        "overlong_continuation_chunks_in_first_update": sum(1 for c in chunks if c.overlong_continuation),
    }


def scan_tokenizer(label: str, tokenizer) -> dict[str, Any]:
    per_len: dict[int, dict[str, Any]] = {
        L: {
            "chunks": 0,
            "charged_words": 0,
            "active_tokens": 0,
            "overlong_words": 0,
            "overlong_continuation_chunks": 0,
            "zero_token_words": 0,
            "first_chunks": [],
            "chunk_word_counts": [],
            "chunk_token_counts": [],
        } for L in LENGTHS
    }
    rows = 0
    words_total = 0
    raw_tokens_total = 0
    unassigned_total = 0
    unassigned_examples: list[dict[str, Any]] = []
    row_token_samples: list[int] = []

    with TRAIN_10M.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch at row {rows}: field {words} actual {len(text.split())}")
            word_tokens, raw_tokens, unassigned, examples = token_ids_by_whitespace_word(text, tokenizer)
            if len(word_tokens) != words:
                raise RuntimeError(f"word span mismatch at row {rows}: {len(word_tokens)} vs {words}")
            rows += 1
            words_total += words
            raw_tokens_total += raw_tokens
            unassigned_total += unassigned
            if examples and len(unassigned_examples) < 10:
                unassigned_examples.extend(examples[:10 - len(unassigned_examples)])
            if len(row_token_samples) < 2000:
                row_token_samples.append(raw_tokens)
            for L in LENGTHS:
                chunks, meta = chunks_from_word_tokens(word_tokens, L, rows - 1, words)
                st = per_len[L]
                st["chunks"] += len(chunks)
                st["charged_words"] += sum(c.charged_words for c in chunks)
                st["active_tokens"] += sum(len(c.input_ids) for c in chunks)
                st["overlong_words"] += meta["overlong_words"]
                st["overlong_continuation_chunks"] += meta["overlong_continuation_chunks"]
                st["zero_token_words"] += meta["zero_token_words"]
                st["chunk_word_counts"].extend(c.charged_words for c in chunks)
                st["chunk_token_counts"].extend(len(c.input_ids) for c in chunks)
                limit = FIRST_CHUNK_COLLECT_LIMIT[L]
                first = st["first_chunks"]
                if len(first) < limit:
                    first.extend(chunks[:max(0, limit - len(first))])
            if rows % 10000 == 0:
                print(json.dumps({"event": "scan_progress", "tokenizer": label, "rows": rows, "words": words_total}), flush=True)

    if rows != ROWS_PER_EPOCH or words_total != WORDS_PER_EPOCH:
        raise RuntimeError(f"unexpected corpus size rows={rows} words={words_total}")
    if unassigned_total:
        raise RuntimeError(f"unassigned tokenizer offsets remain for {label}: {unassigned_total}, examples={unassigned_examples[:3]}")

    curriculum_measurements = json.loads(CURRICULUM_MEASUREMENTS.read_text(encoding="utf-8"))
    rec = curriculum_measurements["faithful_streaming_chunking"][label]
    by_len: dict[str, Any] = {}
    first_update: dict[str, Any] = {}
    for L in LENGTHS:
        st = per_len[L]
        dist = batch_distribution(int(st["chunks"]))
        first_n = int(dist["first_step_chunks"])
        first_chunks = st["first_chunks"][:first_n]
        if len(first_chunks) != first_n:
            raise RuntimeError(f"not enough first chunks for {label} L{L}: {len(first_chunks)} vs {first_n}")
        mask_meas = first_update_mask_measurement(label, tokenizer, first_chunks, L, STEPS_PER_EPOCH * EPOCHS_TOTAL)
        first_update[str(L)] = {**dist, **mask_meas}
        by_len[str(L)] = {
            "length": L,
            "chunks_per_epoch_with_overlong_split": int(st["chunks"]),
            "charged_words_per_epoch": int(st["charged_words"]),
            "active_tokens_per_epoch": int(st["active_tokens"]),
            "active_tokens_per_word": int(st["active_tokens"]) / WORDS_PER_EPOCH,
            "overlong_words": int(st["overlong_words"]),
            "overlong_continuation_chunks": int(st["overlong_continuation_chunks"]),
            "zero_token_words": int(st["zero_token_words"]),
            "chunks_per_epoch_without_overlong_split": int(rec["by_length"][str(L)]["faithful_chunks_per_epoch"]),
            "chunk_delta_vs_step76": int(st["chunks"]) - int(rec["by_length"][str(L)]["faithful_chunks_per_epoch"]),
            "active_tokens_per_epoch": int(rec["by_length"][str(L)]["faithful_active_tokens_per_epoch"]),
            "active_token_delta_vs_step76": int(st["active_tokens"]) - int(rec["by_length"][str(L)]["faithful_active_tokens_per_epoch"]),
            "first_update": first_update[str(L)],
        }
        if int(st["charged_words"]) != WORDS_PER_EPOCH:
            raise RuntimeError(f"charged word mismatch for {label} L{L}: {st['charged_words']}")
        if int(st["active_tokens"]) != raw_tokens_total:
            raise RuntimeError(f"active token mismatch for {label} L{L}: {st['active_tokens']} vs raw {raw_tokens_total}")
    schedule_64_128_256 = {
        "charged_words_total": WORDS_PER_EPOCH * EPOCHS_TOTAL,
        "active_tokens_total": raw_tokens_total * EPOCHS_TOTAL,
        "steps_total_stage_reset": STEPS_PER_EPOCH * EPOCHS_TOTAL,
        "expected_masked_targets_mask015": raw_tokens_total * EPOCHS_TOTAL * BASE_MASK_PROB,
        "per_length_epochs": {"64": 3, "128": 4, "256": 3},
        "chunks_total_with_overlong_split": 3 * by_len["64"]["chunks_per_epoch_with_overlong_split"] + 4 * by_len["128"]["chunks_per_epoch_with_overlong_split"] + 3 * by_len["256"]["chunks_per_epoch_with_overlong_split"],
    }
    schedule_256 = {
        "charged_words_total": WORDS_PER_EPOCH * EPOCHS_TOTAL,
        "active_tokens_total": raw_tokens_total * EPOCHS_TOTAL,
        "steps_total_stage_reset": STEPS_PER_EPOCH * EPOCHS_TOTAL,
        "expected_masked_targets_mask015": raw_tokens_total * EPOCHS_TOTAL * BASE_MASK_PROB,
        "per_length_epochs": {"256": 10},
        "chunks_total_with_overlong_split": 10 * by_len["256"]["chunks_per_epoch_with_overlong_split"],
    }
    return {
        "tokenizer": label,
        "tokenizer_path": rel(TOKENIZERS[label]),
        "tokenizer_json_sha256": sha256_file(TOKENIZERS[label] / "tokenizer.json"),
        "vocab_size": tokenizer.vocab_size,
        "len_tokenizer": len(tokenizer),
        "rows": rows,
        "words_per_epoch": words_total,
        "raw_tokens_per_epoch": raw_tokens_total,
        "raw_tokens_per_word": raw_tokens_total / words_total,
        "unassigned_offsets": unassigned_total,
        "by_length": by_len,
        "schedule_U256": schedule_256,
        "schedule_U64_128_256": schedule_64_128_256,
        "schedules_match_on_charged_words_active_tokens_steps": (
            schedule_256["charged_words_total"] == schedule_64_128_256["charged_words_total"]
            and schedule_256["active_tokens_total"] == schedule_64_128_256["active_tokens_total"]
            and schedule_256["steps_total_stage_reset"] == schedule_64_128_256["steps_total_stage_reset"]
        ),
    }


def write_outputs(result: dict[str, Any]) -> None:
    out_json = OUT_DIR / "chunk_stream_preflight.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    length_csv = OUT_DIR / "chunk_stream_by_length.csv"
    with length_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "tokenizer", "length", "chunks_per_epoch", "charged_words_per_epoch", "active_tokens_per_epoch",
            "overlong_words", "overlong_continuation_chunks", "chunk_delta_vs_step76", "active_token_delta_vs_step76",
            "first_step_chunks", "first_step_charged_words", "first_step_active_tokens", "first_step_masked_tokens",
            "first_step_selected_groups", "first_step_accumulation_microbatches", "first_step_nominal_token_slots",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, rec in result["tokenizers"].items():
            for L, r in rec["by_length"].items():
                fu = r["first_update"]
                w.writerow({
                    "tokenizer": label,
                    "length": L,
                    "chunks_per_epoch": r["chunks_per_epoch_with_overlong_split"],
                    "charged_words_per_epoch": r["charged_words_per_epoch"],
                    "active_tokens_per_epoch": r["active_tokens_per_epoch"],
                    "overlong_words": r["overlong_words"],
                    "overlong_continuation_chunks": r["overlong_continuation_chunks"],
                    "chunk_delta_vs_step76": r["chunk_delta_vs_step76"],
                    "active_token_delta_vs_step76": r["active_token_delta_vs_step76"],
                    "first_step_chunks": fu["chunks_in_first_update"],
                    "first_step_charged_words": fu["charged_words_in_first_update"],
                    "first_step_active_tokens": fu["active_tokens_in_first_update"],
                    "first_step_masked_tokens": fu["masked_tokens_sampled_wwm015_cpu_rng"],
                    "first_step_selected_groups": fu["selected_word_groups_sampled_wwm015_cpu_rng"],
                    "first_step_accumulation_microbatches": fu["accumulation_microbatches"],
                    "first_step_nominal_token_slots": fu["nominal_token_slots"],
                })

    lines: list[str] = []
    lines.append("# research — Chunk-stream construction for experience utilization")
    lines.append("")
    lines.append("CPU-only construction check. No model was trained or evaluated.")
    lines.append("")
    lines.append("## What was verified")
    lines.append("")
    lines.append("The word-boundary chunk stream exposes every tokenizer token from the 10M-word compact_view_reinvest corpus once per epoch, while charging each whitespace word once per epoch. It therefore repairs the prefix path's hidden-word debit without adding new linguistic data or extra epochs.")
    lines.append("")
    for label, rec in result["tokenizers"].items():
        u256 = rec["schedule_U256"]
        uord = rec["schedule_U64_128_256"]
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- Corpus: {rec['rows']:,} rows, {rec['words_per_epoch']:,} words, {rec['raw_tokens_per_epoch']:,} untruncated tokens ({rec['raw_tokens_per_word']:.4f} tokens/word), unassigned offsets {rec['unassigned_offsets']}.")
        for L in LENGTHS:
            r = rec["by_length"][str(L)]
            fu = r["first_update"]
            lines.append(f"- L{L}: {r['chunks_per_epoch_with_overlong_split']:,} chunks/epoch, {r['charged_words_per_epoch']:,} charged words, {r['active_tokens_per_epoch']:,} active tokens, overlong words {r['overlong_words']} with {r['overlong_continuation_chunks']} continuation chunks; first update uses {fu['chunks_in_first_update']} chunks, {fu['active_tokens_in_first_update']} active tokens, {fu['accumulation_microbatches']} microbatches, sampled {fu['masked_tokens_sampled_wwm015_cpu_rng']} masked tokens.")
        lines.append(f"- U256 and U64_128_256 match on 100M charged words, {u256['active_tokens_total']:,} active tokens, and {u256['steps_total_stage_reset']} stage-reset updates: {rec['schedules_match_on_charged_words_active_tokens_steps']}.")
        lines.append(f"- U256 chunks total {u256['chunks_total_with_overlong_split']:,}; U64_128_256 chunks total {uord['chunks_total_with_overlong_split']:,}. The latter changes context/order/packing, not word count, token count, or update count.")
        lines.append("")
    lines.append("## Implementation implication")
    lines.append("")
    lines.append("A future trainer should build stage-specific chunk streams, form exactly 253 optimizer updates per 10M-word stage epoch by distributing chunks across updates, apply full-effective-batch WWM before microbatch forward passes, and sum charged words from chunk metadata. The single overlong whitespace word at L64 should be split into continuation chunks with zero extra charged words rather than creating an over-length tensor.")
    lines.append("")
    lines.append(f"JSON: `{rel(out_json)}`")
    lines.append(f"CSV: `{rel(length_csv)}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    sha = sha256_file(TRAIN_10M)
    if sha != EXPECTED_10M_SHA:
        raise RuntimeError(f"10M corpus SHA mismatch: {sha}")
    if not DESIGN.exists():
        raise RuntimeError(f"missing research design file: {DESIGN}")
    tokenizers = {label: base.make_portable_tokenizer(str(path)) for label, path in TOKENIZERS.items()}
    result: dict[str, Any] = {
        "status": "CHUNK_STREAM_PREFLIGHT",
        "purpose": "Construct stage-length word-boundary chunks for the experience-utilization arms and verify charged-word/token/update matching without training.",
        "input_corpus": rel(TRAIN_10M),
        "input_corpus_sha256": sha,
        "measurement": rel(CURRICULUM_MEASUREMENTS),
        "design": rel(DESIGN),
        "steps_per_epoch": STEPS_PER_EPOCH,
        "epochs_total": EPOCHS_TOTAL,
        "base_mask_prob": BASE_MASK_PROB,
        "micro_batch_chunks": MICRO_BATCH_CHUNKS,
        "tokenizers": {},
    }
    for label, tok in tokenizers.items():
        print(json.dumps({"event": "tokenizer_scan_start", "tokenizer": label}), flush=True)
        result["tokenizers"][label] = scan_tokenizer(label, tok)
    write_outputs(result)
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(OUT_DIR / "chunk_stream_preflight.json"),
        "csv": rel(OUT_DIR / "chunk_stream_by_length.csv"),
        "note": rel(NOTE),
        "legal40k_match": result["tokenizers"]["legal40k"]["schedules_match_on_charged_words_active_tokens_steps"],
        "legal40k_L64_overlong_words": result["tokenizers"]["legal40k"]["by_length"]["64"]["overlong_words"],
        "legal40k_L64_chunk_delta_vs_step76": result["tokenizers"]["legal40k"]["by_length"]["64"]["chunk_delta_vs_step76"],
        "minfreq25_match": result["tokenizers"]["minfreq25"]["schedules_match_on_charged_words_active_tokens_steps"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
