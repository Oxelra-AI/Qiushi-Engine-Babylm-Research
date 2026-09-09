#!/usr/bin/env python3
"""research: CPU accounting audit for a faithful 64->256 sequence curriculum.

Why this exists
---------------
The public 41.80 leader reports a 64->256 sequence-length curriculum with batch size
scaled inversely with sequence length. The COMPACT_EXPERIENCE/research trainer has a `seq_len_schedule`,
but it tokenizes each full row at max length 256, slices the prefix to the short length in
the training loop, and still charges the full row word count. That is a visible-prefix
intervention, not a faithful sequence curriculum.

This script quantifies that difference on the exact legal compact_view_reinvest 10M pool
for legal40k and minfreq25 tokenizers. It trains/evaluates no model and uses no GPU.
It produces route material for deciding how to implement sequence curriculum if the active
12x384 depth run does not cross the frontier.
"""
from __future__ import annotations

import json
import math
import pathlib
import time
from collections import defaultdict
from typing import Any

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
POOL_10M = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
OUT_DIR = ROOT / "data/sequence_curriculum_accounting_audit"
OUT_JSON = OUT_DIR / "sequence_curriculum_accounting_audit.json"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/sequence_curriculum_accounting_audit.md')

TOKENIZERS = {
    "legal40k": ROOT / "data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k",
    "minfreq25": ROOT / "data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq25",
}
LENGTHS = [64, 128, 256]
MASK_PROB = 0.15
BASE_ROWS_PER_STEP_AT_256 = 256

# Candidate schedules expressed as epochs at each sequence length. These are NOT claimed
# to be the leader's hidden exact schedule; they are faithful candidates matching the
# published 64->256 idea and inversely scaled batch rule.
SCHEDULES = {
    "two_stage_64x7_256x3": [(7, 64), (3, 256)],
    "three_stage_64x3_128x4_256x3": [(3, 64), (4, 128), (3, 256)],
    "mild_128x7_256x3": [(7, 128), (3, 256)],
}


def rows_per_step(length: int) -> int:
    return int(BASE_ROWS_PER_STEP_AT_256 * 256 / length)


def token_word_group_lengths(tokenizer, text: str) -> list[int]:
    # Match the existing trainer's word-group heuristic: a new group begins at i=0 or
    # when the byte/SentencePiece token string starts with Ġ/▁.
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if not ids:
        return []
    special = set(tokenizer.all_special_ids)
    lengths: list[int] = []
    current = 0
    started = False
    for i, tid in enumerate(ids):
        if tid in special:
            continue
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        is_start = i == 0 or str(tok).startswith("Ġ") or str(tok).startswith("▁")
        if is_start:
            if started:
                lengths.append(current)
            current = 1
            started = True
        else:
            if not started:
                started = True
                current = 1
            else:
                current += 1
    if started:
        lengths.append(current)
    return lengths


def prefix_visible(group_lengths: list[int], L: int) -> tuple[int, int, bool]:
    # Number of tokenized word-groups and tokens visible in the first L tokens.
    pos = 0
    groups = 0
    toks = 0
    for gl in group_lengths:
        if pos >= L:
            break
        # Include group if at least its first token is visible; target burden in the sliced
        # trainer includes only visible tokens, not the hidden suffix.
        visible = min(gl, max(0, L - pos))
        if visible > 0:
            groups += 1
            toks += visible
        pos += gl
    return groups, toks, (sum(group_lengths) > L)


def greedy_chunk_counts(group_lengths: list[int], L: int) -> tuple[int, int, int]:
    # Split at word-group boundaries so every tokenized word-group is exposed at this stage.
    # If a single word-group exceeds L (rare for byte-BPE), place it alone; no group is dropped.
    if not group_lengths:
        return 0, 0, 0
    chunks = 0
    cur = 0
    max_group = 0
    for gl in group_lengths:
        max_group = max(max_group, gl)
        if cur == 0:
            cur = gl
        elif cur + gl <= L:
            cur += gl
        else:
            chunks += 1
            cur = gl
    if cur > 0:
        chunks += 1
    return chunks, sum(group_lengths), max_group


def load_pool() -> list[dict[str, Any]]:
    rows = []
    with POOL_10M.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({
                "text": str(obj["text"]),
                "words": int(obj.get("words", len(str(obj["text"]).split()))),
                "source": str(obj.get("source", "")),
            })
    return rows


def summarize_tokenizer(label: str, tok_path: pathlib.Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True)
    # Accumulators
    total_words = 0
    total_groups = 0
    total_tokens = 0
    prefix = {L: defaultdict(float) for L in LENGTHS}
    chunked = {L: defaultdict(float) for L in LENGTHS}
    by_source_prefix = {L: defaultdict(lambda: defaultdict(float)) for L in LENGTHS}
    by_source_chunked = {L: defaultdict(lambda: defaultdict(float)) for L in LENGTHS}

    t0 = time.time()
    for idx, r in enumerate(rows, 1):
        glens = token_word_group_lengths(tok, r["text"])
        words = r["words"]
        src = r["source"]
        ntok = sum(glens)
        ng = len(glens)
        total_words += words
        total_groups += ng
        total_tokens += ntok
        for L in LENGTHS:
            pg, pt, truncated = prefix_visible(glens, L)
            prefix[L]["visible_groups"] += pg
            prefix[L]["visible_tokens"] += pt
            prefix[L]["charged_words"] += words
            prefix[L]["rows"] += 1
            prefix[L]["truncated_rows"] += int(truncated)
            by_source_prefix[L][src]["visible_groups"] += pg
            by_source_prefix[L][src]["visible_tokens"] += pt
            by_source_prefix[L][src]["charged_words"] += words
            ch, ctok, maxg = greedy_chunk_counts(glens, L)
            chunked[L]["chunks"] += ch
            chunked[L]["visible_groups"] += ng
            chunked[L]["visible_tokens"] += ntok
            chunked[L]["charged_groups"] += ng
            chunked[L]["max_group_tokens_max"] = max(chunked[L].get("max_group_tokens_max", 0), maxg)
            by_source_chunked[L][src]["chunks"] += ch
            by_source_chunked[L][src]["visible_groups"] += ng
            by_source_chunked[L][src]["visible_tokens"] += ntok
        if idx % 10000 == 0:
            pass

    out = {
        "label": label,
        "tokenizer_path": str(tok_path),
        "vocab_size": tok.vocab_size,
        "len_tokenizer": len(tok),
        "pool_rows": len(rows),
        "pool_words": total_words,
        "total_token_word_groups": total_groups,
        "total_tokens_no_trunc": total_tokens,
        "tokens_per_word": total_tokens / max(1, total_words),
        "groups_per_whitespace_word": total_groups / max(1, total_words),
        "elapsed_sec": round(time.time() - t0, 2),
        "lengths": {},
        "schedules": {},
    }

    for L in LENGTHS:
        p = dict(prefix[L])
        c = dict(chunked[L])
        p["visible_group_fraction_vs_full"] = p["visible_groups"] / max(1, total_groups)
        p["visible_token_fraction_vs_full"] = p["visible_tokens"] / max(1, total_tokens)
        p["charged_words_per_visible_group"] = p["charged_words"] / max(1, p["visible_groups"])
        p["truncated_row_fraction"] = p["truncated_rows"] / max(1, p["rows"])
        p["expected_wwm_target_tokens"] = MASK_PROB * p["visible_tokens"]
        p["fixed_row_steps_per_10M_epoch"] = math.ceil(len(rows) / BASE_ROWS_PER_STEP_AT_256)

        c["rows_per_step_inverse_scaled"] = rows_per_step(L)
        c["steps_per_10M_epoch_inverse_scaled"] = math.ceil(c["chunks"] / c["rows_per_step_inverse_scaled"])
        c["padded_token_slots_per_epoch"] = c["chunks"] * L
        c["mean_chunk_fill"] = c["visible_tokens"] / max(1, c["padded_token_slots_per_epoch"])
        c["expected_wwm_target_tokens"] = MASK_PROB * c["visible_tokens"]
        c["visible_group_fraction_vs_full"] = c["visible_groups"] / max(1, total_groups)
        c["visible_token_fraction_vs_full"] = c["visible_tokens"] / max(1, total_tokens)

        out["lengths"][str(L)] = {
            "current_prefix_slicing_fixed_row_batch": p,
            "faithful_word_boundary_chunking_inverse_batch": c,
        }

    for sname, stages in SCHEDULES.items():
        cur_groups = sum(ep * prefix[L]["visible_groups"] for ep, L in stages)
        cur_tokens = sum(ep * prefix[L]["visible_tokens"] for ep, L in stages)
        faithful_groups = sum(ep * chunked[L]["visible_groups"] for ep, L in stages)
        faithful_tokens = sum(ep * chunked[L]["visible_tokens"] for ep, L in stages)
        cur_steps = sum(ep * math.ceil(len(rows) / BASE_ROWS_PER_STEP_AT_256) for ep, L in stages)
        faithful_steps = sum(ep * math.ceil(chunked[L]["chunks"] / rows_per_step(L)) for ep, L in stages)
        out["schedules"][sname] = {
            "stages_epochs_length": stages,
            "current_prefix_visible_group_fraction_over_10_epochs": cur_groups / max(1, 10 * total_groups),
            "current_prefix_visible_token_fraction_over_10_epochs": cur_tokens / max(1, 10 * total_tokens),
            "current_prefix_missing_group_fraction_over_10_epochs": 1 - cur_groups / max(1, 10 * total_groups),
            "current_prefix_expected_wwm_target_tokens_10ep": MASK_PROB * cur_tokens,
            "current_prefix_optimizer_steps_10ep_fixed_batch256": cur_steps,
            "faithful_visible_group_fraction_over_10_epochs": faithful_groups / max(1, 10 * total_groups),
            "faithful_visible_token_fraction_over_10_epochs": faithful_tokens / max(1, 10 * total_tokens),
            "faithful_expected_wwm_target_tokens_10ep": MASK_PROB * faithful_tokens,
            "faithful_optimizer_steps_10ep_inverse_batch": faithful_steps,
            "optimizer_step_ratio_faithful_vs_current_prefix": faithful_steps / max(1, cur_steps),
            "target_token_ratio_faithful_vs_current_prefix": faithful_tokens / max(1, cur_tokens),
        }

    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_pool()
    summaries = {}
    for label, path in TOKENIZERS.items():
        summaries[label] = summarize_tokenizer(label, path, rows)

    payload = {
        "status": "SEQUENCE_CURRICULUM_ACCOUNTING_AUDIT",
        "pool_10m": str(POOL_10M),
        "pool_rows": len(rows),
        "pool_words": sum(r["words"] for r in rows),
        "published_leader_factor": "sequence length curriculum 64->256, batch size scaled inversely with sequence length (model card lines 203 and 213)",
        "current_trainer_finding": "research/research seq_len_schedule slices max-256 tokenized rows to the current short prefix but still charges full row words; this hides suffix word-groups during short phases while counting them in word exposure.",
        "mask_prob_assumed_for_target_accounting": MASK_PROB,
        "tokenizer_summaries": summaries,
        "scientific_reading": (
            "A faithful sequence curriculum should be implemented as stage-length tokenization/streaming or word-boundary chunking with inverse row-batch scaling, not as prefix slicing of max-256 rows. The audit quantifies how much prefix slicing under-exposes word-groups and target tokens for proposed schedules."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research — Sequence curriculum accounting audit\n")
    lines.append("CPU-only accounting for a faithful 64→256 sequence curriculum on the exact compact_view_reinvest 10M pool. No model was trained or evaluated.\n")
    lines.append("The public leader states `Sequence length curriculum 64 -> 256` and that batch size is scaled inversely with sequence length. The current COMPACT_EXPERIENCE/research schedule instead tokenizes each row at 256, slices a short prefix, and charges the full row words.\n")
    for label, s in summaries.items():
        lines.append(f"## {label}\n")
        lines.append(f"- vocab {s['vocab_size']}; pool tokens/word {s['tokens_per_word']:.4f}; token-word-groups/word {s['groups_per_whitespace_word']:.4f}; tokenization elapsed {s['elapsed_sec']}s")
        lines.append("| L | prefix visible groups | prefix missing groups | prefix charged words / visible group | faithful chunks/epoch | faithful steps/epoch | chunk fill |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for L in LENGTHS:
            p = s["lengths"][str(L)]["current_prefix_slicing_fixed_row_batch"]
            c = s["lengths"][str(L)]["faithful_word_boundary_chunking_inverse_batch"]
            lines.append(
                f"| {L} | {p['visible_group_fraction_vs_full']:.3f} | {1-p['visible_group_fraction_vs_full']:.3f} | {p['charged_words_per_visible_group']:.3f} | {int(c['chunks'])} | {int(c['steps_per_10M_epoch_inverse_scaled'])} | {c['mean_chunk_fill']:.3f} |"
            )
        lines.append("\n### Ten-epoch schedule accounting\n")
        lines.append("| schedule | prefix visible group frac | prefix missing group frac | faithful/current target-token ratio | faithful/current optimizer-step ratio |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, rec in s["schedules"].items():
            lines.append(
                f"| {name} | {rec['current_prefix_visible_group_fraction_over_10_epochs']:.3f} | {rec['current_prefix_missing_group_fraction_over_10_epochs']:.3f} | {rec['target_token_ratio_faithful_vs_current_prefix']:.3f} | {rec['optimizer_step_ratio_faithful_vs_current_prefix']:.3f} |"
            )
        lines.append("")
    lines.append("## Research use\n")
    lines.append("If the active 12×384 fixed-length run does not cross the frontier, a sequence-curriculum route should use a stage-specific chunking/streaming trainer with inverse row-batch scaling. It should not use the existing `seq_len_schedule` path as evidence for the leader-style factor because that path hides suffix word-groups while charging their words.\n")
    lines.append(f"JSON: `{OUT_JSON}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "pool_rows": payload["pool_rows"],
        "pool_words": payload["pool_words"],
        "tokenizers": {k: {
            "tokens_per_word": round(v["tokens_per_word"], 4),
            "L64_prefix_visible_group_frac": round(v["lengths"]["64"]["current_prefix_slicing_fixed_row_batch"]["visible_group_fraction_vs_full"], 4),
            "L128_prefix_visible_group_frac": round(v["lengths"]["128"]["current_prefix_slicing_fixed_row_batch"]["visible_group_fraction_vs_full"], 4),
            "L256_prefix_visible_group_frac": round(v["lengths"]["256"]["current_prefix_slicing_fixed_row_batch"]["visible_group_fraction_vs_full"], 4),
            "three_stage_prefix_missing_group_frac": round(v["schedules"]["three_stage_64x3_128x4_256x3"]["current_prefix_missing_group_fraction_over_10_epochs"], 4),
            "three_stage_target_ratio_faithful_vs_prefix": round(v["schedules"]["three_stage_64x3_128x4_256x3"]["target_token_ratio_faithful_vs_current_prefix"], 4),
        } for k, v in summaries.items()},
        "out_json": str(OUT_JSON),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
