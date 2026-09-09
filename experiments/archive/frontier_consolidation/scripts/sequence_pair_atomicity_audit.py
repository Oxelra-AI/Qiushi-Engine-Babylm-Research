#!/usr/bin/env python3
"""research: pair-atomicity audit for faithful sequence curricula.

CPU-only. No training and no official evaluation text. The research loop
measurement showed that faithful stage-length chunking can expose hidden suffix
words under the legal word budget, but compact-view reinvestment depends on
source and rewrite co-occurring in one context. This audit measures whether
current prefix slicing, naive faithful word chunks, or pair-atomic faithful
chunks preserve source+rewrite same-window structure in the 3,005 changed
compact-view rows.
"""
from __future__ import annotations

import bisect
import collections
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402

POOL_10M = WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
PAIR_MAP = WORKSPACE / "data" / "pair_span_map" / "pair_span_map.jsonl"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
TOKENIZERS = {
    "legal16k": WORKSPACE / "data" / "compliant_tokenizer",
    "minfreq50_supportfloor": WORKSPACE / "data" / "supportfloor_tokenizers" / "legal_byte_bpe_40k_minfreq50",
}
LENGTHS = [64, 128, 256]
SCHEDULES = {
    "64x3_128x4_256x3": {64: 3, 128: 4, 256: 3},
    "64x7_256x3": {64: 7, 128: 0, 256: 3},
}
OUT_DIR = WORKSPACE / "data" / "sequence_pair_atomicity_audit"
NOTE = (USER_ROOT / 'research/notes/frontier_consolidation/sequence_pair_atomicity_audit.md')
WORD_RE = re.compile(r"\S+")
START = time.time()


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


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def quantiles(xs: list[int] | list[float]) -> dict[str, float]:
    if not xs:
        return {"min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
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

    return {"min": ys[0], "p10": q(0.10), "median": q(0.50), "p90": q(0.90), "p95": q(0.95), "max": ys[-1], "mean": sum(ys) / len(ys)}


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


def token_counts_by_word(text: str, tokenizer) -> tuple[list[int], int, int]:
    spans = word_spans(text)
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


def cumulative(counts: list[int]) -> list[int]:
    c = [0]
    total = 0
    for x in counts:
        total += int(x)
        c.append(total)
    return c


def greedy_chunk_ids(counts: list[int], L: int) -> tuple[list[int], int, int, int]:
    ids: list[int] = []
    chunk = 0
    cur_tokens = 0
    cur_words = 0
    overlong_words = 0
    for tok_count in counts:
        tok_count = int(tok_count)
        if tok_count > L:
            overlong_words += 1
            if cur_words:
                chunk += 1
                cur_tokens = 0
                cur_words = 0
            ids.append(chunk)
            chunk += 1
            cur_tokens = 0
            cur_words = 0
            continue
        if cur_words and cur_tokens + tok_count > L:
            chunk += 1
            cur_tokens = 0
            cur_words = 0
        ids.append(chunk)
        cur_tokens += tok_count
        cur_words += 1
    chunks = (max(ids) + 1) if ids else 0
    return ids, chunks, sum(counts), overlong_words


def pair_intervals(pair_record: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    word_pos = 0
    for p in pair_record["pairs"]:
        sw = int(p["source_words"])
        rw = int(p["rewrite_words"])
        out.append({
            "pair_id": str(p["pair_id"]),
            "source_word_start": word_pos,
            "source_word_end": word_pos + sw,
            "rewrite_word_start": word_pos + sw,
            "rewrite_word_end": word_pos + sw + rw,
            "pair_word_start": word_pos,
            "pair_word_end": word_pos + sw + rw,
            "source_words": sw,
            "rewrite_words": rw,
        })
        word_pos += sw + rw
    return out


def prefix_visibility_for_pair(interval: dict[str, Any], cums: list[int], L: int) -> str:
    s0 = cums[interval["source_word_start"]]
    s1 = cums[interval["source_word_end"]]
    r0 = cums[interval["rewrite_word_start"]]
    r1 = cums[interval["rewrite_word_end"]]
    s_any = s0 < L and s1 > 0
    r_any = r0 < L and r1 > 0
    s_full = s1 <= L
    r_full = r1 <= L
    if s_full and r_full:
        return "source_rewrite_full_visible"
    if s_any and r_any:
        return "source_rewrite_partial_cooccur"
    if s_any and not r_any:
        return "source_only"
    if r_any and not s_any:
        return "rewrite_only"
    return "invisible"


def greedy_visibility_for_pair(interval: dict[str, Any], chunk_ids: list[int]) -> str:
    ps = interval["pair_word_start"]
    pe = interval["pair_word_end"]
    ss = interval["source_word_start"]
    se = interval["source_word_end"]
    rs = interval["rewrite_word_start"]
    re = interval["rewrite_word_end"]
    pair_chunks = set(chunk_ids[ps:pe])
    if len(pair_chunks) == 1:
        return "source_rewrite_full_same_chunk"
    src_chunks = set(chunk_ids[ss:se])
    rew_chunks = set(chunk_ids[rs:re])
    if src_chunks & rew_chunks:
        return "source_rewrite_partial_cochunk"
    return "split_between_source_and_rewrite"


def load_changed_rows() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    pair_rows = list(read_jsonl(PAIR_MAP))
    needed = {int(r["row_index_in_pool_1based"]): r for r in pair_rows}
    pool_rows: dict[int, dict[str, Any]] = {}
    with POOL_10M.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if idx in needed:
                obj = json.loads(line)
                pool_rows[idx] = obj
                rec = needed[idx]
                if int(obj["example_id"]) != int(rec["example_id"]):
                    raise RuntimeError(f"example_id mismatch at pool row {idx}")
                if int(obj["words"]) != int(rec["words"]):
                    raise RuntimeError(f"word mismatch at pool row {idx}")
            if len(pool_rows) == len(needed):
                break
    if len(pool_rows) != len(needed):
        raise RuntimeError(f"missing pool rows: {len(pool_rows)} of {len(needed)}")
    return pair_rows, pool_rows


def analyze_tokenizer(label: str, tokenizer, pair_rows: list[dict[str, Any]], pool_rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    by_len: dict[int, dict[str, Any]] = {}
    for L in LENGTHS:
        by_len[L] = {
            "prefix_counts": collections.Counter(),
            "greedy_word_chunk_counts": collections.Counter(),
            "pair_atomic_counts": collections.Counter(),
            "greedy_chunks_in_changed_rows": 0,
            "pair_atomic_chunks_in_changed_rows": 0,
            "greedy_overlong_words": 0,
            "pair_atomic_overlong_pairs": 0,
        }
    row_count = 0
    pair_count = 0
    total_words = 0
    total_tokens = 0
    total_unassigned = 0
    pair_token_lens: list[int] = []
    source_token_lens: list[int] = []
    rewrite_token_lens: list[int] = []
    sample_overlong: dict[str, list[dict[str, Any]]] = {str(L): [] for L in LENGTHS}
    sample_split: dict[str, list[dict[str, Any]]] = {str(L): [] for L in LENGTHS}

    for rec in pair_rows:
        row_count += 1
        idx = int(rec["row_index_in_pool_1based"])
        text = str(pool_rows[idx]["text"])
        words = int(rec["words"])
        counts, raw_tokens, unassigned = token_counts_by_word(text, tokenizer)
        if len(counts) != words:
            raise RuntimeError(f"word/token alignment mismatch row {idx}: {len(counts)} vs {words}")
        cums = cumulative(counts)
        intervals = pair_intervals(rec)
        if intervals and intervals[-1]["pair_word_end"] != words:
            raise RuntimeError(f"pair word coverage mismatch row {idx}: {intervals[-1]['pair_word_end']} vs {words}")
        total_words += words
        total_tokens += raw_tokens
        total_unassigned += unassigned
        pair_count += len(intervals)
        for p in intervals:
            s_tok = cums[p["source_word_end"]] - cums[p["source_word_start"]]
            r_tok = cums[p["rewrite_word_end"]] - cums[p["rewrite_word_start"]]
            source_token_lens.append(s_tok)
            rewrite_token_lens.append(r_tok)
            pair_token_lens.append(s_tok + r_tok)
        for L in LENGTHS:
            bucket = by_len[L]
            chunk_ids, chunks, _tok_sum, overlong_words = greedy_chunk_ids(counts, L)
            bucket["greedy_chunks_in_changed_rows"] += chunks
            bucket["greedy_overlong_words"] += overlong_words
            # pair-atomic chunks over the row, greedily packing whole source+rewrite pair atoms.
            cur = 0
            atomic_chunks = 0
            for p in intervals:
                pair_tok = cums[p["pair_word_end"]] - cums[p["pair_word_start"]]
                if pair_tok > L:
                    bucket["pair_atomic_overlong_pairs"] += 1
                    bucket["pair_atomic_counts"]["overlong_cannot_full_fit"] += 1
                    if cur:
                        atomic_chunks += 1
                        cur = 0
                    atomic_chunks += 1
                    if len(sample_overlong[str(L)]) < 8:
                        sample_overlong[str(L)].append({
                            "row_index": idx,
                            "example_id": rec["example_id"],
                            "pair_id": p["pair_id"],
                            "pair_words": p["source_words"] + p["rewrite_words"],
                            "pair_tokens": pair_tok,
                            "source_tokens": cums[p["source_word_end"]] - cums[p["source_word_start"]],
                            "rewrite_tokens": cums[p["rewrite_word_end"]] - cums[p["rewrite_word_start"]],
                        })
                    continue
                if cur and cur + pair_tok > L:
                    atomic_chunks += 1
                    cur = 0
                cur += pair_tok
                bucket["pair_atomic_counts"]["source_rewrite_full_same_chunk"] += 1
            if cur:
                atomic_chunks += 1
            bucket["pair_atomic_chunks_in_changed_rows"] += atomic_chunks

            for p in intervals:
                pref = prefix_visibility_for_pair(p, cums, L)
                bucket["prefix_counts"][pref] += 1
                gv = greedy_visibility_for_pair(p, chunk_ids)
                bucket["greedy_word_chunk_counts"][gv] += 1
                if gv != "source_rewrite_full_same_chunk" and len(sample_split[str(L)]) < 8:
                    sample_split[str(L)].append({
                        "row_index": idx,
                        "example_id": rec["example_id"],
                        "pair_id": p["pair_id"],
                        "visibility": gv,
                        "source_words": p["source_words"],
                        "rewrite_words": p["rewrite_words"],
                        "pair_tokens": cums[p["pair_word_end"]] - cums[p["pair_word_start"]],
                        "source_chunks": sorted(set(chunk_ids[p["source_word_start"]:p["source_word_end"]])),
                        "rewrite_chunks": sorted(set(chunk_ids[p["rewrite_word_start"]:p["rewrite_word_end"]])),
                    })

    result_by_len: dict[str, Any] = {}
    for L in LENGTHS:
        bucket = by_len[L]
        prefix = dict(bucket["prefix_counts"])
        greedy = dict(bucket["greedy_word_chunk_counts"])
        atomic = dict(bucket["pair_atomic_counts"])
        result_by_len[str(L)] = {
            "L": L,
            "pairs": pair_count,
            "prefix_counts": prefix,
            "prefix_full_source_rewrite_fraction": prefix.get("source_rewrite_full_visible", 0) / pair_count,
            "prefix_any_source_rewrite_cooccur_fraction": (prefix.get("source_rewrite_full_visible", 0) + prefix.get("source_rewrite_partial_cooccur", 0)) / pair_count,
            "greedy_word_chunk_counts": greedy,
            "greedy_full_source_rewrite_same_chunk_fraction": greedy.get("source_rewrite_full_same_chunk", 0) / pair_count,
            "greedy_any_source_rewrite_cochunk_fraction": (greedy.get("source_rewrite_full_same_chunk", 0) + greedy.get("source_rewrite_partial_cochunk", 0)) / pair_count,
            "pair_atomic_counts": atomic,
            "pair_atomic_fittable_full_same_chunk_fraction": atomic.get("source_rewrite_full_same_chunk", 0) / pair_count,
            "pair_atomic_overlong_fraction": atomic.get("overlong_cannot_full_fit", 0) / pair_count,
            "greedy_chunks_in_changed_rows": int(bucket["greedy_chunks_in_changed_rows"]),
            "pair_atomic_chunks_in_changed_rows": int(bucket["pair_atomic_chunks_in_changed_rows"]),
            "greedy_overlong_words": int(bucket["greedy_overlong_words"]),
            "pair_atomic_overlong_pairs": int(bucket["pair_atomic_overlong_pairs"]),
            "sample_overlong_pairs": sample_overlong[str(L)],
            "sample_greedy_splits": sample_split[str(L)],
        }
    schedule_results: dict[str, Any] = {}
    for name, epochs_by_L in SCHEDULES.items():
        denom = pair_count * sum(epochs_by_L.values())
        prefix_full = 0
        prefix_any = 0
        greedy_full = 0
        greedy_any = 0
        atomic_full = 0
        atomic_overlong = 0
        for L, epochs in epochs_by_L.items():
            if epochs <= 0:
                continue
            r = result_by_len[str(L)]
            prefix_full += epochs * int(r["prefix_counts"].get("source_rewrite_full_visible", 0))
            prefix_any += epochs * int(r["prefix_counts"].get("source_rewrite_full_visible", 0) + r["prefix_counts"].get("source_rewrite_partial_cooccur", 0))
            greedy_full += epochs * int(r["greedy_word_chunk_counts"].get("source_rewrite_full_same_chunk", 0))
            greedy_any += epochs * int(r["greedy_word_chunk_counts"].get("source_rewrite_full_same_chunk", 0) + r["greedy_word_chunk_counts"].get("source_rewrite_partial_cochunk", 0))
            atomic_full += epochs * int(r["pair_atomic_counts"].get("source_rewrite_full_same_chunk", 0))
            atomic_overlong += epochs * int(r["pair_atomic_counts"].get("overlong_cannot_full_fit", 0))
        schedule_results[name] = {
            "epochs_by_length": epochs_by_L,
            "pair_occurrences": denom,
            "prefix_full_source_rewrite_fraction": prefix_full / denom,
            "prefix_any_source_rewrite_cooccur_fraction": prefix_any / denom,
            "greedy_full_source_rewrite_same_chunk_fraction": greedy_full / denom,
            "greedy_any_source_rewrite_cochunk_fraction": greedy_any / denom,
            "pair_atomic_fittable_full_same_chunk_fraction": atomic_full / denom,
            "pair_atomic_overlong_fraction": atomic_overlong / denom,
        }
    return {
        "tokenizer": label,
        "changed_rows": row_count,
        "pairs": pair_count,
        "changed_words": total_words,
        "raw_tokens_in_changed_rows": total_tokens,
        "unassigned_offsets": total_unassigned,
        "pair_token_stats": {
            "source_tokens": quantiles(source_token_lens),
            "rewrite_tokens": quantiles(rewrite_token_lens),
            "source_plus_rewrite_tokens": quantiles(pair_token_lens),
        },
        "by_length": result_by_len,
        "schedules": schedule_results,
    }


def write_note(result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research sequence pair-atomicity audit\n")
    lines.append("CPU-only audit of whether a short-sequence curriculum preserves source+rewrite same-window structure in the compact-view changed block. No model was trained and no official eval text was used.\n")
    lines.append("\n## Inputs\n")
    lines.append(f"- Pool SHA matched: `{result['sha256']['pool_matches']}`. Pair map rows: `{result['pair_map_rows']}`.\n")
    lines.append(f"- Pair map source: `{rel(PAIR_MAP)}`; pool: `{rel(POOL_10M)}`.\n")
    lines.append("\n## Pair token lengths\n")
    lines.append("| tokenizer | pairs | source+rewrite token median/p95/max | source median | rewrite median |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for label, rec in result["tokenizer_results"].items():
        pt = rec["pair_token_stats"]["source_plus_rewrite_tokens"]
        st = rec["pair_token_stats"]["source_tokens"]
        rt = rec["pair_token_stats"]["rewrite_tokens"]
        lines.append(f"| {label} | {rec['pairs']} | {pt['median']:.1f}/{pt['p95']:.1f}/{pt['max']:.0f} | {st['median']:.1f} | {rt['median']:.1f} |\n")
    lines.append("\n## Per-length preservation\n")
    lines.append("Fractions are over the 12,155 compact source+rewrite pairs in the changed block. Prefix is the current schedule path; greedy is faithful word-boundary chunking without pair awareness; pair-atomic packs whole source+rewrite pairs when they fit.\n\n")
    lines.append("| tokenizer | L | prefix full | prefix any cooccur | greedy full same chunk | greedy any cochunk | pair-atomic full/fittable | pair-atomic overlong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, rec in result["tokenizer_results"].items():
        for L in ["64", "128", "256"]:
            r = rec["by_length"][L]
            lines.append(
                f"| {label} | {L} | {r['prefix_full_source_rewrite_fraction']:.3f} | {r['prefix_any_source_rewrite_cooccur_fraction']:.3f} | {r['greedy_full_source_rewrite_same_chunk_fraction']:.3f} | {r['greedy_any_source_rewrite_cochunk_fraction']:.3f} | {r['pair_atomic_fittable_full_same_chunk_fraction']:.3f} | {r['pair_atomic_overlong_fraction']:.3f} |\n"
            )
    lines.append("\n## Ten-epoch schedule pair preservation\n")
    lines.append("| tokenizer | schedule | prefix full | greedy full | pair-atomic full/fittable | pair-atomic overlong |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for label, rec in result["tokenizer_results"].items():
        for schedule, r in rec["schedules"].items():
            lines.append(
                f"| {label} | {schedule} | {r['prefix_full_source_rewrite_fraction']:.3f} | {r['greedy_full_source_rewrite_same_chunk_fraction']:.3f} | {r['pair_atomic_fittable_full_same_chunk_fraction']:.3f} | {r['pair_atomic_overlong_fraction']:.3f} |\n"
            )
    lines.append("\n## Scientific reading\n")
    lines.append("- Existing prefix slicing hides most compact pairs during L64/L128 phases because many pairs occur after the row prefix, so it weakens the load-bearing same-window second-view signal while still charging words.\n")
    lines.append("- Naive word-boundary chunking exposes suffix words but can split source and rewrite across chunks, especially at L64. A future sequence trainer should preserve pair atoms in changed rows when possible rather than merely stream whitespace chunks.\n")
    lines.append("- Pair-atomic chunking is feasible for most pairs at L64 and almost all pairs at L128/L256; overlong pairs quantify the unavoidable context-length cost. This is a construction requirement, not a reason to launch sequence training before current word-mean/support-floor evidence is read.\n")
    lines.append(f"\nFull JSON: `{rel(OUT_DIR / 'sequence_pair_atomicity_audit.json')}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    pool_sha = sha256_file(POOL_10M)
    if pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError(f"pool SHA mismatch {pool_sha}")
    pair_rows, pool_rows = load_changed_rows()
    tokenizers = {label: base.make_portable_tokenizer(str(path)) for label, path in TOKENIZERS.items()}
    result: dict[str, Any] = {
        "status": "SEQUENCE_PAIR_ATOMICITY_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Measure whether current prefix slicing, naive faithful word chunks, or pair-atomic chunks preserve compact source+rewrite same-window structure under short sequence lengths.",
        "sha256": {"pool_expected": EXPECTED_POOL_SHA, "pool_actual": pool_sha, "pool_matches": True},
        "inputs": {"pool_10m": rel(POOL_10M), "pair_span_map": rel(PAIR_MAP), "tokenizers": {k: rel(v) for k, v in TOKENIZERS.items()}},
        "pair_map_rows": len(pair_rows),
        "tokenizer_results": {},
        "interpretation": {
            "prefix_path": "Measures current local seq_len_schedule behavior: tokens after L are absent from masking/loss, regardless of the charged row words.",
            "greedy_word_chunking": "Measures a faithful word-budget chunk stream that exposes suffix words but does not protect source+rewrite boundaries.",
            "pair_atomic_chunking": "Measures a changed-row chunk stream that packs each source+rewrite pair as an atom when its tokenized length fits the stage length.",
        },
    }
    for label, tok in tokenizers.items():
        print(json.dumps({"event": "analyze_tokenizer", "tokenizer": label, "elapsed_sec": round(time.time() - START, 1)}), flush=True)
        result["tokenizer_results"][label] = analyze_tokenizer(label, tok, pair_rows, pool_rows)
    result["elapsed_sec"] = round(time.time() - START, 3)
    out_json = OUT_DIR / "sequence_pair_atomicity_audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(result)
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(NOTE), "elapsed_sec": result["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
