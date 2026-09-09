#!/usr/bin/env python3
"""research tokenizer-group chunk measurement for possible faithful sequence training.

CPU-only, training-side only.  research measured a word-boundary chunking
alternative through whitespace spans and found a small class of byte-level space
tokens outside those spans.  This script measures the same object directly in the
actual WWM group stream used by the inherited trainer: a new group begins at
position 0 or at byte-level word-start tokens (Ġ/▁), and groups are never split.

Scientific use:
- quantify whether tokenizer-group chunking preserves the legal 10M-word / 100M
  exposure accounting while including the separator tokens omitted by the
  whitespace-span approximation;
- measure whether pair-aware group chunking can retain the source+rewrite
  same-window signal in the compact changed rows;
- produce construction facts only.  It launches no training and reads no official
  evaluation text.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
POOL_10M = WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
ROW_META = WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
PAIR_ROWS = WORKSPACE / "data" / "medium_compact_analysis" / "medium_compact_ws_rows.jsonl"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
TOKENIZERS = {
    "legal16k": WORKSPACE / "data" / "compliant_tokenizer",
    "minfreq50_supportfloor": WORKSPACE / "data" / "supportfloor_tokenizers" / "legal_byte_bpe_40k_minfreq50",
}
EXPECTED_TOKENIZER_SHA = {
    "legal16k": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
    "minfreq50_supportfloor": "9900f42b392fb69dd55c9c9fd7f539da09b3a33487e4e3542f9b953f48310922",
}
OUT_DIR = WORKSPACE / "data" / "tokenizer_group_chunk_measurement"
NOTE = (USER_ROOT / 'research/notes/frontier_consolidation/tokenizer_group_chunk_measurement.md')
CHANGED_SOURCE = "cleanqwen_fineweb_compact_view_reinvest"
LENGTHS = [64, 128, 256]
SCHEDULES = {
    "64x3_128x4_256x3": {64: 3, 128: 4, 256: 3},
    "64x7_256x3": {64: 7, 128: 0, 256: 3},
}
BASE_ROWS_AT_256 = 256
WORD_RE = re.compile(r"\S+")
STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
START = time.time()


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_word(w: str) -> str:
    return STRIP_RE.sub("", w).lower().replace("’", "'")


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def row_words(text: str) -> list[str]:
    return [norm_word(m.group(0)) for m in WORD_RE.finditer(text)]


def find_word_index(starts: list[int], spans: list[tuple[int, int]], s: int, e: int) -> int | None:
    if e <= s or not spans:
        return None
    idx = bisect.bisect_right(starts, s) - 1
    for j in (idx - 1, idx, idx + 1):
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


def quantiles(xs: list[int] | list[float]) -> dict[str, float]:
    if not xs:
        return {"min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    ys = sorted(float(x) for x in xs)

    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        pos = p * (len(ys) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ys[lo]
        return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)

    return {"min": ys[0], "p10": q(0.10), "median": q(0.50), "p90": q(0.90), "p95": q(0.95), "max": ys[-1], "mean": sum(ys) / len(ys)}


def stats(xs: list[float]) -> dict[str, float | int]:
    if not xs:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ys = sorted(float(x) for x in xs)

    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        pos = p * (len(ys) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ys[lo]
        return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)

    return {"n": len(ys), "mean": statistics.mean(ys), "median": statistics.median(ys), "p05": q(0.05), "p95": q(0.95), "min": ys[0], "max": ys[-1]}


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def group_stream(text: str, tokenizer) -> dict[str, Any]:
    enc = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc.get("offset_mapping", [])]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    spans = word_spans(text)
    starts = [s for s, _ in spans]
    word_for_token: list[int | None] = []
    for s, e in offsets:
        word_for_token.append(find_word_index(starts, spans, s, e))

    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for i, (tid, tstr) in enumerate(zip(ids, tokens)):
        if current is None or is_word_start(str(tstr)) or i == 0:
            if current is not None:
                groups.append(current)
            current = {"token_start": i, "token_end": i + 1, "token_count": 1, "word_indices": set(), "token_strings": [str(tstr)]}
        else:
            current["token_end"] = i + 1
            current["token_count"] += 1
            if len(current["token_strings"]) < 5:
                current["token_strings"].append(str(tstr))
        wi = word_for_token[i]
        if wi is not None:
            current["word_indices"].add(int(wi))
    if current is not None:
        groups.append(current)

    word_to_groups: dict[int, list[int]] = {i: [] for i in range(len(spans))}
    group_word_index: list[int | None] = []
    multiword_groups = 0
    no_word_groups = 0
    for gi, g in enumerate(groups):
        wis = sorted(int(x) for x in g["word_indices"])
        g["word_indices"] = wis
        if not wis:
            no_word_groups += 1
            group_word_index.append(None)
        else:
            if len(wis) > 1:
                multiword_groups += 1
            group_word_index.append(wis[0])
            for wi in wis:
                word_to_groups[wi].append(gi)
    words_no_group = sum(1 for v in word_to_groups.values() if not v)
    words_multi_group = sum(1 for v in word_to_groups.values() if len(v) > 1)
    no_word_examples = []
    for gi, g in enumerate(groups):
        if not g["word_indices"] and len(no_word_examples) < 8:
            s = offsets[g["token_start"]][0] if offsets else 0
            e = offsets[g["token_end"] - 1][1] if offsets else 0
            no_word_examples.append({
                "group_index": gi,
                "token_start": g["token_start"],
                "token_end": g["token_end"],
                "token_strings": g["token_strings"],
                "substring": text[s:e],
                "offset_start": s,
                "offset_end": e,
            })
    return {
        "ids": ids,
        "offsets": offsets,
        "groups": groups,
        "group_token_counts": [int(g["token_count"]) for g in groups],
        "group_word_index": group_word_index,
        "word_to_groups": word_to_groups,
        "word_count": len(spans),
        "raw_tokens": len(ids),
        "group_count": len(groups),
        "no_word_groups": no_word_groups,
        "multiword_groups": multiword_groups,
        "words_no_group": words_no_group,
        "words_multi_group": words_multi_group,
        "no_word_examples": no_word_examples,
    }


def greedy_chunk_ids_for_groups(group_token_counts: list[int], L: int) -> tuple[list[int], int, int, int]:
    ids: list[int] = []
    chunk = 0
    cur_tokens = 0
    cur_groups = 0
    overlong = 0
    for tok_count in group_token_counts:
        tok_count = int(tok_count)
        if tok_count > L:
            overlong += 1
            if cur_groups:
                chunk += 1
                cur_tokens = 0
                cur_groups = 0
            ids.append(chunk)
            chunk += 1
            cur_tokens = 0
            cur_groups = 0
            continue
        if cur_groups and cur_tokens + tok_count > L:
            chunk += 1
            cur_tokens = 0
            cur_groups = 0
        ids.append(chunk)
        cur_tokens += tok_count
        cur_groups += 1
    n_chunks = (max(ids) + 1) if ids else 0
    return ids, n_chunks, sum(group_token_counts), overlong


def full_pool_measurement(label: str, tokenizer) -> dict[str, Any]:
    by_len: dict[int, dict[str, Any]] = {L: {
        "prefix_active_tokens": 0,
        "prefix_visible_groups": 0,
        "group_chunks": 0,
        "group_active_tokens": 0,
        "group_overlong_groups": 0,
        "chunk_group_counts": [],
        "chunk_word_counts": [],
    } for L in LENGTHS}
    totals = Counter()
    group_count_deltas: list[int] = []
    group_token_counts_sample: list[int] = []
    no_word_examples: list[dict[str, Any]] = []
    rows = 0
    sample_rows: list[dict[str, Any]] = []

    with POOL_10M.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            declared_words = int(obj.get("words", len(text.split())))
            if declared_words != len(text.split()):
                raise RuntimeError(f"word-count mismatch row={rows+1}: field={declared_words} split={len(text.split())}")
            gs = group_stream(text, tokenizer)
            rows += 1
            totals["declared_words"] += declared_words
            totals["raw_tokens"] += gs["raw_tokens"]
            totals["groups"] += gs["group_count"]
            totals["no_word_groups"] += gs["no_word_groups"]
            totals["multiword_groups"] += gs["multiword_groups"]
            totals["words_no_group"] += gs["words_no_group"]
            totals["words_multi_group"] += gs["words_multi_group"]
            totals["over256_rows"] += 1 if gs["raw_tokens"] > 256 else 0
            totals["truncated_tokens_vs256"] += max(0, gs["raw_tokens"] - 256)
            group_count_deltas.append(gs["group_count"] - declared_words)
            if len(group_token_counts_sample) < 10000:
                group_token_counts_sample.extend(gs["group_token_counts"][: max(0, 10000 - len(group_token_counts_sample))])
            if gs["no_word_groups"] and len(no_word_examples) < 8:
                no_word_examples.append({
                    "row_index_1based": rows,
                    "example_id": obj.get("example_id"),
                    "source": obj.get("source"),
                    "examples": gs["no_word_examples"],
                })
            if len(sample_rows) < 6:
                sample_rows.append({
                    "row_index_1based": rows,
                    "example_id": obj.get("example_id"),
                    "source": obj.get("source"),
                    "declared_words": declared_words,
                    "raw_tokens": gs["raw_tokens"],
                    "group_count": gs["group_count"],
                    "group_minus_words": gs["group_count"] - declared_words,
                    "no_word_groups": gs["no_word_groups"],
                    "words_no_group": gs["words_no_group"],
                })
            for L in LENGTHS:
                rec = by_len[L]
                rec["prefix_active_tokens"] += min(gs["raw_tokens"], L)
                rec["prefix_visible_groups"] += sum(1 for g in gs["groups"] if int(g["token_start"]) < L)
                chunk_ids, n_chunks, active_tokens, overlong = greedy_chunk_ids_for_groups(gs["group_token_counts"], L)
                rec["group_chunks"] += n_chunks
                rec["group_active_tokens"] += active_tokens
                rec["group_overlong_groups"] += overlong
                if n_chunks:
                    chunk_groups = [0 for _ in range(n_chunks)]
                    chunk_words_sets: list[set[int]] = [set() for _ in range(n_chunks)]
                    for gi, cid in enumerate(chunk_ids):
                        chunk_groups[cid] += 1
                        for wi in gs["groups"][gi]["word_indices"]:
                            chunk_words_sets[cid].add(int(wi))
                    rec["chunk_group_counts"].extend(chunk_groups)
                    rec["chunk_word_counts"].extend([len(s) for s in chunk_words_sets])
            if rows % 10000 == 0:
                print(json.dumps({"event": "full_pool_progress", "tokenizer": label, "rows": rows, "time_utc": now_utc(), "elapsed_sec": round(time.time() - START, 1)}), flush=True)

    if int(totals["declared_words"]) != 10_000_000:
        raise RuntimeError(f"expected 10M words, measured {totals['declared_words']}")
    prefix_steps_per_epoch = math.ceil(rows / BASE_ROWS_AT_256)
    by_length_out: dict[str, Any] = {}
    for L in LENGTHS:
        rec = by_len[L]
        inverse_batch = int(BASE_ROWS_AT_256 * (256 / L))
        group_steps = math.ceil(int(rec["group_chunks"]) / inverse_batch)
        by_length_out[str(L)] = {
            "stage_length": L,
            "inverse_scaled_row_batch": inverse_batch,
            "prefix_rows_per_epoch_current_loop": rows,
            "prefix_steps_per_epoch_current_loop_batch256": prefix_steps_per_epoch,
            "prefix_active_tokens_per_epoch": int(rec["prefix_active_tokens"]),
            "prefix_visible_groups_per_epoch": int(rec["prefix_visible_groups"]),
            "group_chunks_per_epoch": int(rec["group_chunks"]),
            "declared_words_per_epoch": int(totals["declared_words"]),
            "group_active_tokens_per_epoch": int(rec["group_active_tokens"]),
            "group_active_token_ratio_vs_prefix": rec["group_active_tokens"] / max(1, rec["prefix_active_tokens"]),
            "group_steps_per_epoch_inverse_batch": group_steps,
            "group_step_ratio_vs_prefix": group_steps / max(1, prefix_steps_per_epoch),
            "group_overlong_groups": int(rec["group_overlong_groups"]),
            "chunk_groups_quantiles": quantiles(rec["chunk_group_counts"]),
            "chunk_declared_words_quantiles": quantiles(rec["chunk_word_counts"]),
        }
    schedule_out: dict[str, Any] = {}
    for name, counts_by_L in SCHEDULES.items():
        prefix_tokens = 0
        group_tokens = 0
        prefix_steps = 0
        group_steps = 0
        for L, epochs in counts_by_L.items():
            if epochs <= 0:
                continue
            r = by_length_out[str(L)]
            prefix_tokens += epochs * int(r["prefix_active_tokens_per_epoch"])
            group_tokens += epochs * int(r["group_active_tokens_per_epoch"])
            prefix_steps += epochs * int(r["prefix_steps_per_epoch_current_loop_batch256"])
            group_steps += epochs * int(r["group_steps_per_epoch_inverse_batch"])
        schedule_out[name] = {
            "epochs_by_length": counts_by_L,
            "charged_words_total": int(totals["declared_words"]) * sum(counts_by_L.values()),
            "prefix_active_tokens_total": prefix_tokens,
            "group_active_tokens_total": group_tokens,
            "group_active_token_ratio_vs_prefix": group_tokens / max(1, prefix_tokens),
            "prefix_optimizer_steps_total": prefix_steps,
            "group_optimizer_steps_total_inverse_batch": group_steps,
            "group_step_ratio_vs_prefix": group_steps / max(1, prefix_steps),
        }
    return {
        "tokenizer": label,
        "rows": rows,
        "declared_words": int(totals["declared_words"]),
        "raw_tokens": int(totals["raw_tokens"]),
        "groups": int(totals["groups"]),
        "raw_tokens_per_word": totals["raw_tokens"] / max(1, totals["declared_words"]),
        "groups_per_word": totals["groups"] / max(1, totals["declared_words"]),
        "no_word_groups": int(totals["no_word_groups"]),
        "multiword_groups": int(totals["multiword_groups"]),
        "words_no_group": int(totals["words_no_group"]),
        "words_multi_group": int(totals["words_multi_group"]),
        "over256_rows": int(totals["over256_rows"]),
        "truncated_tokens_vs256": int(totals["truncated_tokens_vs256"]),
        "group_minus_declared_word_quantiles": quantiles(group_count_deltas),
        "group_token_count_sample_quantiles": quantiles(group_token_counts_sample),
        "no_word_group_examples": no_word_examples,
        "sample_rows": sample_rows,
        "by_length": by_length_out,
        "schedules": schedule_out,
        "accounting_reading": "The trainer-visible group stream includes all tokenizer tokens, including standalone separator-space groups when present. Declared BabyLM word exposure is still charged from the original row word counts; the chunks are a different view of the same allowed text, not additional text.",
    }


def load_meta() -> tuple[set[str], dict[int, dict[str, Any]]]:
    selected: set[str] = set()
    by_ex: dict[int, dict[str, Any]] = {}
    for obj in read_jsonl(ROW_META):
        pair_ids = [str(x).split(":", 1)[-1] for x in obj.get("pair_ids") or []]
        selected.update(pair_ids)
        by_ex[int(obj["example_id"])] = {**obj, "pair_ids_norm": pair_ids}
    return selected, by_ex


def load_pairs(selected: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for obj in read_jsonl(PAIR_ROWS):
        pid = str(obj.get("prompt_id") or obj.get("pair_id") or "")
        if pid in selected:
            out[pid] = obj
    return out


def construct_word_roles(meta: dict[str, Any], pair_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    seq: list[dict[str, Any]] = []
    for pid in meta.get("pair_ids_norm") or []:
        p = pair_map[pid]
        for role, key in [("source", "source_text"), ("rewrite", "rewrite_text")]:
            for local_i, w in enumerate(str(p.get(key) or "").split()):
                seq.append({"norm": norm_word(w), "pair_id": pid, "role": role, "local_word_index": local_i})
    return seq


def pair_intervals(meta: dict[str, Any], pair_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    pos = 0
    for pid in meta.get("pair_ids_norm") or []:
        p = pair_map[pid]
        sw = len(str(p.get("source_text") or "").split())
        rw = len(str(p.get("rewrite_text") or "").split())
        intervals.append({
            "pair_id": pid,
            "source_word_start": pos,
            "source_word_end": pos + sw,
            "rewrite_word_start": pos + sw,
            "rewrite_word_end": pos + sw + rw,
            "pair_word_start": pos,
            "pair_word_end": pos + sw + rw,
            "source_words": sw,
            "rewrite_words": rw,
        })
        pos += sw + rw
    return intervals


def groups_for_word_range(word_to_groups: dict[int, list[int]], start: int, end: int) -> list[int]:
    out: set[int] = set()
    for wi in range(start, end):
        out.update(int(g) for g in word_to_groups.get(wi, []))
    return sorted(out)


def prefix_pair_visibility(source_groups: list[int], rewrite_groups: list[int], groups: list[dict[str, Any]], L: int) -> str:
    def any_visible(gis: list[int]) -> bool:
        return any(int(groups[gi]["token_start"]) < L for gi in gis)

    def full_visible(gis: list[int]) -> bool:
        return bool(gis) and all(int(groups[gi]["token_end"]) <= L for gi in gis)

    s_any = any_visible(source_groups)
    r_any = any_visible(rewrite_groups)
    s_full = full_visible(source_groups)
    r_full = full_visible(rewrite_groups)
    if s_full and r_full:
        return "source_rewrite_full_visible"
    if s_any and r_any:
        return "source_rewrite_partial_cooccur"
    if s_any and not r_any:
        return "source_only"
    if r_any and not s_any:
        return "rewrite_only"
    return "invisible"


def greedy_pair_visibility(pair_groups: list[int], source_groups: list[int], rewrite_groups: list[int], chunk_ids: list[int]) -> str:
    pair_chunks = {chunk_ids[g] for g in pair_groups if 0 <= g < len(chunk_ids)}
    source_chunks = {chunk_ids[g] for g in source_groups if 0 <= g < len(chunk_ids)}
    rewrite_chunks = {chunk_ids[g] for g in rewrite_groups if 0 <= g < len(chunk_ids)}
    if pair_chunks and len(pair_chunks) == 1:
        return "source_rewrite_full_same_chunk"
    if source_chunks and rewrite_chunks and (source_chunks & rewrite_chunks):
        return "source_rewrite_partial_cochunk"
    if source_chunks and not rewrite_chunks:
        return "source_only"
    if rewrite_chunks and not source_chunks:
        return "rewrite_only"
    return "split_or_invisible"


def pair_atomic_chunk_ids(group_token_counts: list[int], pair_group_ranges: list[tuple[int, int]], L: int) -> tuple[list[int], set[int], int]:
    chunk_ids = [-1 for _ in group_token_counts]
    current_chunk = 0
    current_tokens = 0
    overlong_pairs: set[int] = set()
    next_unassigned_group = 0

    def flush_if_needed(tok_count: int):
        nonlocal current_chunk, current_tokens
        if current_tokens and current_tokens + tok_count > L:
            current_chunk += 1
            current_tokens = 0

    for pi, (gs, ge) in enumerate(pair_group_ranges):
        # Assign any unpaired gap groups greedily.  Changed rows should not have gaps,
        # but this keeps the measurement robust to separator-only groups.
        while next_unassigned_group < gs:
            tc = int(group_token_counts[next_unassigned_group])
            flush_if_needed(tc)
            chunk_ids[next_unassigned_group] = current_chunk
            current_tokens += tc
            if tc > L:
                current_chunk += 1
                current_tokens = 0
            next_unassigned_group += 1
        atom_tokens = sum(int(x) for x in group_token_counts[gs:ge])
        if atom_tokens <= L:
            flush_if_needed(atom_tokens)
            for gi in range(gs, ge):
                chunk_ids[gi] = current_chunk
            current_tokens += atom_tokens
        else:
            overlong_pairs.add(pi)
            if current_tokens:
                current_chunk += 1
                current_tokens = 0
            for gi in range(gs, ge):
                tc = int(group_token_counts[gi])
                flush_if_needed(tc)
                chunk_ids[gi] = current_chunk
                current_tokens += tc
                if tc > L:
                    current_chunk += 1
                    current_tokens = 0
        next_unassigned_group = max(next_unassigned_group, ge)
    while next_unassigned_group < len(group_token_counts):
        tc = int(group_token_counts[next_unassigned_group])
        flush_if_needed(tc)
        chunk_ids[next_unassigned_group] = current_chunk
        current_tokens += tc
        if tc > L:
            current_chunk += 1
            current_tokens = 0
        next_unassigned_group += 1
    for i, cid in enumerate(chunk_ids):
        if cid < 0:
            raise RuntimeError(f"unassigned group {i}")
    n_chunks = (max(chunk_ids) + 1) if chunk_ids else 0
    return chunk_ids, overlong_pairs, n_chunks


def pair_measurement_for_tokenizer(label: str, tokenizer, meta_by_ex: dict[int, dict[str, Any]], pair_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = 0
    pairs = 0
    exact_alignment_rows = 0
    token_truncation_rows = 0
    pair_token_lengths: list[float] = []
    source_token_lengths: list[float] = []
    rewrite_token_lengths: list[float] = []
    row_group_lengths: list[float] = []
    row_token_lengths: list[float] = []
    group_word_alignment = Counter()
    per_L = {L: {
        "prefix": Counter(),
        "greedy": Counter(),
        "pair_atomic": Counter(),
        "pair_atomic_overlong": 0,
        "pair_atomic_chunks": 0,
        "greedy_chunks": 0,
    } for L in LENGTHS}
    samples: list[dict[str, Any]] = []

    with POOL_10M.open("r", encoding="utf-8") as f:
        row_index = 0
        for line in f:
            if not line.strip():
                continue
            row_index += 1
            obj = json.loads(line)
            if obj.get("source") != CHANGED_SOURCE:
                continue
            rows += 1
            ex_id = int(obj["example_id"])
            meta = meta_by_ex.get(ex_id)
            if meta is None:
                raise RuntimeError(f"changed row example_id={ex_id} missing meta")
            text = str(obj["text"])
            words_norm = row_words(text)
            constructed = construct_word_roles(meta, pair_map)
            if len(words_norm) != len(constructed):
                raise RuntimeError(f"changed row example_id={ex_id} word length mismatch row={len(words_norm)} constructed={len(constructed)}")
            mismatch = next((i for i, w in enumerate(words_norm) if w != constructed[i]["norm"]), None)
            if mismatch is not None:
                raise RuntimeError(f"changed row example_id={ex_id} mismatch at word={mismatch} row={words_norm[mismatch]} constructed={constructed[mismatch]['norm']}")
            exact_alignment_rows += 1
            intervals = pair_intervals(meta, pair_map)
            gs = group_stream(text, tokenizer)
            if gs["raw_tokens"] > 256:
                token_truncation_rows += 1
            row_group_lengths.append(float(gs["group_count"]))
            row_token_lengths.append(float(gs["raw_tokens"]))
            group_word_alignment["no_word_groups"] += gs["no_word_groups"]
            group_word_alignment["multiword_groups"] += gs["multiword_groups"]
            group_word_alignment["words_no_group"] += gs["words_no_group"]
            group_word_alignment["words_multi_group"] += gs["words_multi_group"]

            pair_records = []
            pair_group_ranges: list[tuple[int, int]] = []
            for interval in intervals:
                source_groups = groups_for_word_range(gs["word_to_groups"], interval["source_word_start"], interval["source_word_end"])
                rewrite_groups = groups_for_word_range(gs["word_to_groups"], interval["rewrite_word_start"], interval["rewrite_word_end"])
                pair_groups = sorted(set(source_groups) | set(rewrite_groups))
                if not source_groups or not rewrite_groups or not pair_groups:
                    raise RuntimeError(f"missing groups for pair {interval['pair_id']} example_id={ex_id}")
                gs0, ge0 = min(pair_groups), max(pair_groups) + 1
                # Source and rewrite are contiguous word blocks; direct group stream should
                # make their pair group interval contiguous except for rare no-word separators,
                # which are included if they sit inside the interval.
                pair_group_ranges.append((gs0, ge0))
                stoks = sum(int(gs["groups"][g]["token_count"]) for g in source_groups)
                rtoks = sum(int(gs["groups"][g]["token_count"]) for g in rewrite_groups)
                ptoks = sum(int(gs["groups"][g]["token_count"]) for g in range(gs0, ge0))
                source_token_lengths.append(float(stoks))
                rewrite_token_lengths.append(float(rtoks))
                pair_token_lengths.append(float(ptoks))
                pair_records.append({
                    "pair_id": interval["pair_id"],
                    "source_groups": source_groups,
                    "rewrite_groups": rewrite_groups,
                    "pair_groups": list(range(gs0, ge0)),
                    "source_tokens": stoks,
                    "rewrite_tokens": rtoks,
                    "pair_tokens_including_internal_separators": ptoks,
                })
                pairs += 1

            for L in LENGTHS:
                rec = per_L[L]
                greedy_ids, greedy_chunks, _active, _overlong = greedy_chunk_ids_for_groups(gs["group_token_counts"], L)
                atomic_ids, overlong_set, atomic_chunks = pair_atomic_chunk_ids(gs["group_token_counts"], pair_group_ranges, L)
                rec["greedy_chunks"] += greedy_chunks
                rec["pair_atomic_chunks"] += atomic_chunks
                rec["pair_atomic_overlong"] += len(overlong_set)
                for pi, pair_rec in enumerate(pair_records):
                    pvis = prefix_pair_visibility(pair_rec["source_groups"], pair_rec["rewrite_groups"], gs["groups"], L)
                    gvis = greedy_pair_visibility(pair_rec["pair_groups"], pair_rec["source_groups"], pair_rec["rewrite_groups"], greedy_ids)
                    avis = greedy_pair_visibility(pair_rec["pair_groups"], pair_rec["source_groups"], pair_rec["rewrite_groups"], atomic_ids)
                    if pi in overlong_set:
                        avis = "source_rewrite_overlong"
                    rec["prefix"][pvis] += 1
                    rec["greedy"][gvis] += 1
                    rec["pair_atomic"][avis] += 1
            if len(samples) < 5:
                samples.append({
                    "row_index_1based": row_index,
                    "example_id": ex_id,
                    "words": int(obj.get("words", len(text.split()))),
                    "raw_tokens": gs["raw_tokens"],
                    "group_count": gs["group_count"],
                    "pair_count": len(pair_records),
                    "first_pairs": pair_records[:3],
                    "text_preview": text[:220],
                })

    per_length_out: dict[str, Any] = {}
    for L in LENGTHS:
        rec = per_L[L]
        total = max(1, sum(rec["prefix"].values()))
        per_length_out[str(L)] = {
            "pairs": total,
            "prefix_counts": dict(rec["prefix"]),
            "greedy_counts": dict(rec["greedy"]),
            "pair_atomic_counts": dict(rec["pair_atomic"]),
            "prefix_full_fraction": rec["prefix"].get("source_rewrite_full_visible", 0) / total,
            "prefix_any_cooccur_fraction": (rec["prefix"].get("source_rewrite_full_visible", 0) + rec["prefix"].get("source_rewrite_partial_cooccur", 0)) / total,
            "greedy_full_same_chunk_fraction": rec["greedy"].get("source_rewrite_full_same_chunk", 0) / total,
            "greedy_any_cochunk_fraction": (rec["greedy"].get("source_rewrite_full_same_chunk", 0) + rec["greedy"].get("source_rewrite_partial_cochunk", 0)) / total,
            "pair_atomic_full_same_chunk_fraction": rec["pair_atomic"].get("source_rewrite_full_same_chunk", 0) / total,
            "pair_atomic_overlong_fraction": rec["pair_atomic"].get("source_rewrite_overlong", 0) / total,
            "greedy_chunks_total_changed_rows": int(rec["greedy_chunks"]),
            "pair_atomic_chunks_total_changed_rows": int(rec["pair_atomic_chunks"]),
        }
    schedule_out: dict[str, Any] = {}
    for name, counts_by_L in SCHEDULES.items():
        total_epochs = sum(counts_by_L.values())
        prefix_full = 0.0
        greedy_full = 0.0
        atomic_full = 0.0
        atomic_overlong = 0.0
        for L, epochs in counts_by_L.items():
            if epochs <= 0:
                continue
            r = per_length_out[str(L)]
            prefix_full += epochs * r["prefix_full_fraction"]
            greedy_full += epochs * r["greedy_full_same_chunk_fraction"]
            atomic_full += epochs * r["pair_atomic_full_same_chunk_fraction"]
            atomic_overlong += epochs * r["pair_atomic_overlong_fraction"]
        schedule_out[name] = {
            "epochs_by_length": counts_by_L,
            "prefix_full_fraction_epoch_mean": prefix_full / total_epochs,
            "greedy_full_same_chunk_fraction_epoch_mean": greedy_full / total_epochs,
            "pair_atomic_full_same_chunk_fraction_epoch_mean": atomic_full / total_epochs,
            "pair_atomic_overlong_fraction_epoch_mean": atomic_overlong / total_epochs,
        }
    return {
        "tokenizer": label,
        "changed_rows": rows,
        "pairs": pairs,
        "exact_alignment_rows": exact_alignment_rows,
        "token_truncation_rows_vs256": token_truncation_rows,
        "row_group_stats": stats(row_group_lengths),
        "row_token_stats": stats(row_token_lengths),
        "pair_token_stats_including_internal_separators": stats(pair_token_lengths),
        "source_token_stats": stats(source_token_lengths),
        "rewrite_token_stats": stats(rewrite_token_lengths),
        "group_word_alignment_counts": dict(group_word_alignment),
        "by_length": per_length_out,
        "schedules": schedule_out,
        "samples": samples,
    }


def write_csvs(result: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "group_chunk_by_length.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tokenizer", "length", "declared_words", "raw_tokens", "groups", "prefix_active_tokens",
            "group_active_tokens", "active_token_ratio", "prefix_steps", "group_steps", "step_ratio",
            "group_chunks", "overlong_groups",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, rec in result["full_pool_group_chunking"].items():
            for L, r in rec["by_length"].items():
                w.writerow({
                    "tokenizer": label,
                    "length": L,
                    "declared_words": rec["declared_words"],
                    "raw_tokens": rec["raw_tokens"],
                    "groups": rec["groups"],
                    "prefix_active_tokens": r["prefix_active_tokens_per_epoch"],
                    "group_active_tokens": r["group_active_tokens_per_epoch"],
                    "active_token_ratio": r["group_active_token_ratio_vs_prefix"],
                    "prefix_steps": r["prefix_steps_per_epoch_current_loop_batch256"],
                    "group_steps": r["group_steps_per_epoch_inverse_batch"],
                    "step_ratio": r["group_step_ratio_vs_prefix"],
                    "group_chunks": r["group_chunks_per_epoch"],
                    "overlong_groups": r["group_overlong_groups"],
                })
    with (OUT_DIR / "pair_preservation_by_length.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tokenizer", "length", "pairs", "prefix_full", "prefix_any", "greedy_full", "greedy_any",
            "pair_atomic_full", "pair_atomic_overlong", "greedy_chunks_changed", "pair_atomic_chunks_changed",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, rec in result["changed_pair_group_chunking"].items():
            for L, r in rec["by_length"].items():
                w.writerow({
                    "tokenizer": label,
                    "length": L,
                    "pairs": r["pairs"],
                    "prefix_full": r["prefix_full_fraction"],
                    "prefix_any": r["prefix_any_cooccur_fraction"],
                    "greedy_full": r["greedy_full_same_chunk_fraction"],
                    "greedy_any": r["greedy_any_cochunk_fraction"],
                    "pair_atomic_full": r["pair_atomic_full_same_chunk_fraction"],
                    "pair_atomic_overlong": r["pair_atomic_overlong_fraction"],
                    "greedy_chunks_changed": r["greedy_chunks_total_changed_rows"],
                    "pair_atomic_chunks_changed": r["pair_atomic_chunks_total_changed_rows"],
                })


def write_note(result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research tokenizer-group chunk measurement\n")
    lines.append("CPU-only training-side measurement of faithful sequence chunks in the actual WWM group stream. No model was trained and no official evaluation text was read.\n")
    lines.append("\n## Inputs\n")
    lines.append(f"- Pool SHA matched: `{result['sha256']['pool_matches']}`.\n")
    for label, rec in result["tokenizers"].items():
        lines.append(f"- {label}: vocab {rec['len']}, tokenizer SHA matched `{rec['sha_matches']}`.\n")
    lines.append("\n## Full-pool group stream\n")
    lines.append("| tokenizer | words | raw tokens | groups | groups/word | no-word groups | words without group | over256 rows | truncated tokens vs256 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, rec in result["full_pool_group_chunking"].items():
        lines.append(
            f"| {label} | {rec['declared_words']} | {rec['raw_tokens']} | {rec['groups']} | {rec['groups_per_word']:.6f} | {rec['no_word_groups']} | {rec['words_no_group']} | {rec['over256_rows']} | {rec['truncated_tokens_vs256']} |\n"
        )
    lines.append("\n## Group chunks by length\n")
    lines.append("Tokenizer-group chunks include separator-space tokens and never split WWM groups. Ratios compare against the inherited prefix-slicing path over the same rows and word budget.\n\n")
    lines.append("| tokenizer | L | group chunks/epoch | active-token ratio vs prefix | step ratio vs prefix | chunk groups median/p90 | chunk words median/p90 | overlong groups |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, rec in result["full_pool_group_chunking"].items():
        for L in ["64", "128", "256"]:
            r = rec["by_length"][L]
            qg = r["chunk_groups_quantiles"]
            qw = r["chunk_declared_words_quantiles"]
            lines.append(
                f"| {label} | {L} | {r['group_chunks_per_epoch']} | {r['group_active_token_ratio_vs_prefix']:.3f} | {r['group_step_ratio_vs_prefix']:.3f} | {qg['median']:.1f}/{qg['p90']:.1f} | {qw['median']:.1f}/{qw['p90']:.1f} | {r['group_overlong_groups']} |\n"
            )
    lines.append("\n## Ten-epoch schedule ratios\n")
    lines.append("| tokenizer | schedule | charged words | active-token ratio vs prefix | optimizer-step ratio vs prefix |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for label, rec in result["full_pool_group_chunking"].items():
        for schedule, r in rec["schedules"].items():
            lines.append(f"| {label} | {schedule} | {r['charged_words_total']} | {r['group_active_token_ratio_vs_prefix']:.3f} | {r['group_step_ratio_vs_prefix']:.3f} |\n")
    lines.append("\n## Compact-pair preservation in group chunks\n")
    lines.append("Fractions are over the 12,155 source+rewrite pairs in the compact changed block. Pair-atomic chunks pack each source+rewrite pair as one atom when it fits the stage length.\n\n")
    lines.append("| tokenizer | L | prefix full | greedy full | greedy any cochunk | pair-atomic full | pair-atomic overlong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for label, rec in result["changed_pair_group_chunking"].items():
        for L in ["64", "128", "256"]:
            r = rec["by_length"][L]
            lines.append(
                f"| {label} | {L} | {r['prefix_full_fraction']:.3f} | {r['greedy_full_same_chunk_fraction']:.3f} | {r['greedy_any_cochunk_fraction']:.3f} | {r['pair_atomic_full_same_chunk_fraction']:.3f} | {r['pair_atomic_overlong_fraction']:.3f} |\n"
            )
    lines.append("\n## Pair preservation over ten epochs\n")
    lines.append("| tokenizer | schedule | prefix full | greedy full | pair-atomic full | pair-atomic overlong |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for label, rec in result["changed_pair_group_chunking"].items():
        for schedule, r in rec["schedules"].items():
            lines.append(
                f"| {label} | {schedule} | {r['prefix_full_fraction_epoch_mean']:.3f} | {r['greedy_full_same_chunk_fraction_epoch_mean']:.3f} | {r['pair_atomic_full_same_chunk_fraction_epoch_mean']:.3f} | {r['pair_atomic_overlong_fraction_epoch_mean']:.3f} |\n"
            )
    lines.append("\n## Scientific reading\n")
    lines.append("- Direct tokenizer-group chunks raise the active-token ratios slightly above the whitespace-span estimate because separator-space tokens are now included; the word accounting still remains the same 10M declared words per epoch.\n")
    lines.append("- The group stream is almost one-to-one with declared words, so a faithful group-chunk trainer can preserve the BabyLM word budget while exposing suffix tokens hidden by prefix slicing.\n")
    lines.append("- Pair-aware chunking remains necessary: naive group chunks improve suffix exposure but split many L64 source+rewrite atoms; pair-atomic group chunks preserve most L64 pairs and almost all L128/L256 pairs.\n")
    lines.append("- This is construction evidence for a possible later sequence route, not a reason to interrupt the running word-mean evaluation or the prepared minfreq50 path.\n")
    lines.append(f"\nFull JSON: `{rel(OUT_DIR / 'tokenizer_group_chunk_measurement.json')}`\n")
    lines.append(f"CSV: `{rel(OUT_DIR / 'group_chunk_by_length.csv')}`, `{rel(OUT_DIR / 'pair_preservation_by_length.csv')}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    pool_sha = sha256_file(POOL_10M)
    tokenizers = {}
    tok_records: dict[str, Any] = {}
    for label, path in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        tokenizers[label] = tok
        sha = sha256_file(path / "tokenizer.json")
        tok_records[label] = {
            "path": rel(path),
            "len": len(tok),
            "vocab_size": tok.vocab_size,
            "tokenizer_json_sha256": sha,
            "sha_matches": sha == EXPECTED_TOKENIZER_SHA[label],
        }
    if pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError(f"pool SHA mismatch {pool_sha}")
    if not all(r["sha_matches"] for r in tok_records.values()):
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_records}")

    selected, meta_by_ex = load_meta()
    pair_map = load_pairs(selected)
    missing = selected - set(pair_map)
    if missing:
        raise RuntimeError(f"missing pair records n={len(missing)} sample={sorted(missing)[:5]}")

    result: dict[str, Any] = {
        "status": "TOKENIZER_GROUP_CHUNK_MEASUREMENT",
        "created_utc": now_utc(),
        "purpose": "Measure faithful sequence chunks directly in the inherited trainer's WWM group stream on the frozen compact-view reinvest 10M corpus, including source+rewrite pair preservation, before any sequence GPU work.",
        "inputs": {
            "pool_10m": rel(POOL_10M),
            "row_meta": rel(ROW_META),
            "pair_rows": rel(PAIR_ROWS),
            "changed_source": CHANGED_SOURCE,
            "lengths": LENGTHS,
            "schedules": SCHEDULES,
        },
        "sha256": {
            "pool_expected": EXPECTED_POOL_SHA,
            "pool_actual": pool_sha,
            "pool_matches": pool_sha == EXPECTED_POOL_SHA,
        },
        "tokenizers": tok_records,
        "full_pool_group_chunking": {},
        "changed_pair_group_chunking": {},
        "interpretation": {
            "word_budget": "Declared word counts come from the frozen JSONL rows and sum to 10M per epoch; group chunks reorganize the same text into shorter token windows and do not introduce additional text.",
            "trainer_match": "Groups are formed by the same byte-level word-start rule used by COMPACT_EXPERIENCE masking_curriculum_trainer.MaskedChunkDataset, except here the full untruncated tokenizer stream is used before chunking.",
            "launch_status": "No H100 work is launched. Use this only as construction evidence if sequence length becomes selected after the word-mean and support-floor results are read.",
        },
    }
    for label, tok in tokenizers.items():
        result["full_pool_group_chunking"][label] = full_pool_measurement(label, tok)
    for label, tok in tokenizers.items():
        result["changed_pair_group_chunking"][label] = pair_measurement_for_tokenizer(label, tok, meta_by_ex, pair_map)
    result["elapsed_sec"] = round(time.time() - START, 3)
    out_json = OUT_DIR / "tokenizer_group_chunk_measurement.json"
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
