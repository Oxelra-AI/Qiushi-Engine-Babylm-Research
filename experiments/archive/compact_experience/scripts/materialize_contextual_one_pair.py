#!/usr/bin/env python3
"""research: materialize contextual one-pair clean-Qwen corpora.

This constructs a higher-context same-window variant of the clean research Qwen pair route.
For every selected official-sentence/Qwen-rewrite pair, the original sentence and its
rewrite are placed adjacent inside the real 160-word official row context from which the
original sentence was drawn, with exactly one pair per training row.  The remaining word
budget is filled with deterministic official text rows, and a length-matched official-only
control is created from the same treatment row-length sequence.

The script uses only BabyLM training text and research/028 generation metadata already
counted in the training budget.  It does not read official AoA/CDI words, child curves,
AoA predictions, AoA scores, or downstream evaluation outputs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = _public_path('experiments/archive/compact_experience')
OFFICIAL = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
SELECTED = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
PROMPTS = _public_path('experiments/archive/compact_experience/data/qwen_aligned/rewrite_prompts.jsonl')
EXTRA_SHARDS = [
    _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences_shard0.jsonl'),
    _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences_shard1.jsonl'),
]
TOTAL_WORDS = 10_000_000
PASSES = 10
WORDS_PER_OFFICIAL_ROW = 160
RNG_SEED = 282103
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Row:
    text: str
    words: int
    source: str
    example_id: int
    pair_id: str | None = None
    meta: Dict[str, Any] | None = None


def norm_ws(s: str) -> str:
    return " ".join((s or "").replace("\u00a0", " ").split())


def norm_tok(w: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", w.lower())


def split_sentences(text: str) -> List[str]:
    return [norm_ws(x) for x in SENT_SPLIT.split(norm_ws(text)) if norm_ws(x)]


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(vals: Iterable[float]) -> Dict[str, Any]:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        i = p * (len(ys) - 1)
        lo = int(i)
        hi = min(lo + 1, len(ys) - 1)
        a = i - lo
        return ys[lo] * (1 - a) + ys[hi] * a
    return {
        "n": len(ys),
        "min": round(ys[0], 4),
        "p05": round(q(0.05), 4),
        "mean": round(statistics.fmean(ys), 4),
        "median": round(q(0.5), 4),
        "p95": round(q(0.95), 4),
        "max": round(ys[-1], 4),
    }


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_official() -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(OFFICIAL):
        out[int(row["example_id"])] = row
    total = sum(int(r["words"]) for r in out.values())
    if total != TOTAL_WORDS:
        raise RuntimeError(f"official word total {total} != {TOTAL_WORDS}")
    return out


def load_sentence_idx() -> Dict[str, int]:
    idx: Dict[str, int] = {}
    if PROMPTS.exists():
        with PROMPTS.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "source_sentence_idx" in r:
                        idx[str(r["id"])] = int(r["source_sentence_idx"])
    for p in EXTRA_SHARDS:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if "sentence_idx" in r:
                            idx[str(r["id"])] = int(r["sentence_idx"])
    return idx


def find_subseq(row_words: List[str], sent_words: List[str], normalized: bool = False) -> Optional[Tuple[int, int]]:
    if not sent_words:
        return None
    if normalized:
        rw = [norm_tok(x) for x in row_words]
        sw = [norm_tok(x) for x in sent_words]
    else:
        rw, sw = row_words, sent_words
    n = len(sw)
    for i in range(0, len(rw) - n + 1):
        if rw[i:i+n] == sw:
            return i, i + n
    return None


def locate(pair: Dict[str, Any], official: Dict[int, Dict[str, Any]], sent_idx: Dict[str, int]) -> Dict[str, Any]:
    pid = str(pair["pair_id"])
    exid = int(pair["example_id"])
    row = official.get(exid)
    if row is None:
        raise RuntimeError(f"missing official example_id for pair {pid}: {exid}")
    original = norm_ws(str(pair["original"]))
    row_text = norm_ws(str(row["text"]))
    row_words = row_text.split()
    sents = split_sentences(row_text)
    idx = sent_idx.get(pid)
    chosen_j: Optional[int] = None
    start = end = -1
    method = ""
    if idx is not None and 0 <= idx < len(sents) and norm_ws(sents[idx]) == original:
        chosen_j = idx
        sent_words = sents[idx].split()
        start = sum(len(s.split()) for s in sents[:idx])
        end = start + len(sent_words)
        method = "sentence_idx"
        if row_words[start:end] != sent_words:
            found = find_subseq(row_words, sent_words, False) or find_subseq(row_words, sent_words, True)
            if found is not None:
                start, end = found
                method = "sentence_idx_subseq_repair"
    else:
        for j, s in enumerate(sents):
            if norm_ws(s) == original:
                chosen_j = j
                sent_words = s.split()
                start = sum(len(x.split()) for x in sents[:j])
                end = start + len(sent_words)
                method = "sentence_text"
                if row_words[start:end] != sent_words:
                    found = find_subseq(row_words, sent_words, False) or find_subseq(row_words, sent_words, True)
                    if found is not None:
                        start, end = found
                        method = "sentence_text_subseq_repair"
                break
    if chosen_j is None:
        sent_words = original.split()
        found = find_subseq(row_words, sent_words, False)
        method = "exact_word_subsequence"
        if found is None:
            found = find_subseq(row_words, sent_words, True)
            method = "normalized_word_subsequence"
        if found is None:
            raise RuntimeError(f"could not locate original in official row for pair {pid}")
        start, end = found
    original_words = int(pair.get("original_words", len(original.split())))
    if end - start != original_words:
        # The research feasibility pass found exact sentence matches for all selected pairs; stop if this changes.
        raise RuntimeError(f"span length mismatch for pair {pid}: {end-start} vs {original_words}")
    return {
        "pair_id": pid,
        "example_id": exid,
        "source": pair.get("source"),
        "cohort": pair.get("cohort"),
        "row_words": row_words,
        "start_word": start,
        "end_word": end,
        "method": method,
        "sentence_idx_used": chosen_j,
    }


def take_context(row_words: List[str], start: int, end: int, pair_words: int, cap: int, extra: int = 0) -> Dict[str, Any]:
    target = max(pair_words, min(cap, WORDS_PER_OFFICIAL_ROW)) + extra
    target = min(target, WORDS_PER_OFFICIAL_ROW)
    capacity = max(0, target - pair_words)
    left_avail = start
    right_avail = len(row_words) - end
    left_want = capacity // 2
    right_want = capacity - left_want
    left_take = min(left_avail, left_want)
    right_take = min(right_avail, right_want)
    leftover = capacity - left_take - right_take
    if leftover > 0:
        add_left = min(left_avail - left_take, leftover)
        left_take += add_left
        leftover -= add_left
    if leftover > 0:
        add_right = min(right_avail - right_take, leftover)
        right_take += add_right
        leftover -= add_right
    if leftover != 0:
        raise RuntimeError(f"unused context capacity {leftover}")
    left = row_words[start-left_take:start]
    right = row_words[end:end+right_take]
    return {"left": left, "right": right, "left_take": left_take, "right_take": right_take, "target": pair_words + left_take + right_take}


def contextual_row(pair: Dict[str, Any], loc: Dict[str, Any], cap: int, extra: int = 0) -> Row:
    original = norm_ws(str(pair["original"]))
    rewrite = norm_ws(str(pair["rewrite"]))
    ow = int(pair["original_words"])
    rw = int(pair["rewrite_words"])
    pair_words = ow + rw
    ctx = take_context(loc["row_words"], int(loc["start_word"]), int(loc["end_word"]), pair_words, cap, extra)
    parts = ctx["left"] + original.split() + rewrite.split() + ctx["right"]
    text = " ".join(parts)
    words = len(parts)
    if words != ctx["target"]:
        raise RuntimeError(f"row target mismatch {words} vs {ctx['target']}")
    if words > WORDS_PER_OFFICIAL_ROW:
        raise RuntimeError(f"contextual row longer than 160: {words}")
    meta = {
        "pair_id": str(pair["pair_id"]),
        "example_id": int(pair["example_id"]),
        "source": pair.get("source"),
        "cohort": pair.get("cohort"),
        "words": words,
        "original_words": ow,
        "rewrite_words": rw,
        "pair_words": pair_words,
        "context_words": int(ctx["left_take"] + ctx["right_take"]),
        "left_take": int(ctx["left_take"]),
        "right_take": int(ctx["right_take"]),
        "left_available_words": int(loc["start_word"]),
        "right_available_words": int(len(loc["row_words"]) - int(loc["end_word"])),
        "extra_context_for_divisibility": int(extra),
        "locate_method": loc.get("method"),
        "sentence_idx_used": loc.get("sentence_idx_used"),
    }
    return Row(text=text, words=words, source=f"qwen_context_onepair::{pair.get('source')}", example_id=700000 + int(hashlib.sha256(str(pair["pair_id"]).encode()).hexdigest()[:10], 16), pair_id=str(pair["pair_id"]), meta=meta)


def add_mod_context(rows_meta: List[Tuple[Dict[str, Any], Dict[str, Any]]], cap: int) -> Dict[str, int]:
    # rows_meta holds (pair, loc).  Return per-pair extra context so total contextual row words are divisible by 160.
    base_total = 0
    extras: Dict[str, int] = {}
    capacities: List[Tuple[str, int]] = []
    for pair, loc in rows_meta:
        pair_words = int(pair["pair_words"])
        ctx0 = take_context(loc["row_words"], int(loc["start_word"]), int(loc["end_word"]), pair_words, cap, 0)
        base_words = int(ctx0["target"])
        base_total += base_words
        max_extra = WORDS_PER_OFFICIAL_ROW - base_words
        # There is often unused official context because cap120 deliberately leaves room below 160.
        avail_left = int(loc["start_word"]) - int(ctx0["left_take"])
        avail_right = (len(loc["row_words"]) - int(loc["end_word"])) - int(ctx0["right_take"])
        max_extra = min(max_extra, max(0, avail_left + avail_right))
        if max_extra > 0:
            capacities.append((str(pair["pair_id"]), max_extra))
    need = (-base_total) % WORDS_PER_OFFICIAL_ROW
    remaining = need
    for pid, can in capacities:
        if remaining <= 0:
            break
        take = min(can, remaining)
        extras[pid] = take
        remaining -= take
    if remaining != 0:
        raise RuntimeError(f"could not add {need} extra context words to make contextual block divisible by 160; remaining {remaining}")
    return extras


def official_filler_rows(official_pool: List[Dict[str, Any]], excluded_example_ids: set[int], needed_words: int) -> Tuple[List[Row], Dict[str, int]]:
    if needed_words % WORDS_PER_OFFICIAL_ROW != 0:
        raise RuntimeError(f"needed official filler not divisible by 160: {needed_words}")
    needed_rows = needed_words // WORDS_PER_OFFICIAL_ROW
    non_source = [r for r in official_pool if int(r["example_id"]) not in excluded_example_ids]
    source = [r for r in official_pool if int(r["example_id"]) in excluded_example_ids]
    rng = random.Random(RNG_SEED + 1)
    rng.shuffle(non_source)
    rng.shuffle(source)
    chosen = (non_source + source)[:needed_rows]
    if len(chosen) != needed_rows:
        raise RuntimeError(f"needed {needed_rows} filler rows, got {len(chosen)}")
    rows = [Row(text=str(r["text"]), words=int(r["words"]), source=str(r["source"]), example_id=int(r["example_id"])) for r in chosen]
    reused = sum(1 for r in chosen if int(r["example_id"]) in excluded_example_ids)
    return rows, {"needed_rows": needed_rows, "selected_example_rows_reused_after_nonselected_exhausted": reused, "nonselected_available_rows": len(non_source)}


def official_length_matched_control(official_pool: List[Dict[str, Any]], lengths: List[int]) -> List[Row]:
    all_words: List[str] = []
    source_marks: List[str] = []
    example_marks: List[int] = []
    for r in official_pool:
        ws = str(r["text"]).split()
        all_words.extend(ws)
        source_marks.extend([str(r["source"])] * len(ws))
        example_marks.extend([int(r["example_id"])] * len(ws))
    if len(all_words) != TOTAL_WORDS:
        raise RuntimeError(f"official stream {len(all_words)} != {TOTAL_WORDS}")
    offset = 137 * WORDS_PER_OFFICIAL_ROW
    all_words = all_words[offset:] + all_words[:offset]
    source_marks = source_marks[offset:] + source_marks[:offset]
    example_marks = example_marks[offset:] + example_marks[:offset]
    rows: List[Row] = []
    pos = 0
    for i, L in enumerate(lengths):
        seg_words = all_words[pos:pos+L]
        seg_sources = source_marks[pos:pos+L]
        seg_examples = example_marks[pos:pos+L]
        if len(seg_words) != L:
            raise RuntimeError("length sequence exceeded official stream")
        common_source = collections.Counter(seg_sources).most_common(1)[0][0]
        common_example = collections.Counter(seg_examples).most_common(1)[0][0]
        rows.append(Row(text=" ".join(seg_words), words=L, source=f"official_context_lengthmatched::{common_source}", example_id=800000 + i + common_example * 0))
        pos += L
    if pos != TOTAL_WORDS:
        raise RuntimeError(f"control consumed {pos} words")
    return rows


def write_pool(path: pathlib.Path, rows: List[Row], pair_meta_path: pathlib.Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_f = pair_meta_path.open("w", encoding="utf-8") if pair_meta_path else None
    try:
        with path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if r.words != len(r.text.split()):
                    raise RuntimeError(f"row {i} word mismatch {r.words} vs {len(r.text.split())}")
                f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False) + "\n")
                if meta_f and r.meta is not None:
                    meta = dict(r.meta)
                    meta["row_index"] = i
                    meta["example_id_out"] = r.example_id
                    meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    finally:
        if meta_f:
            meta_f.close()


def write_training(path: pathlib.Path, rows: List[Row], pass_orders: List[List[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for order in pass_orders:
            for idx in order:
                r = rows[idx]
                f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False) + "\n")
                total += r.words
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"{path} total {total} != {TOTAL_WORDS * PASSES}")


def build(cap: int, out_dir: pathlib.Path) -> Dict[str, Any]:
    t0 = time.time()
    official_by_id = load_official()
    official_pool = [official_by_id[i] for i in sorted(official_by_id)]
    sent_idx = load_sentence_idx()
    selected = read_jsonl(SELECTED)
    rows_loc: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for p in selected:
        rows_loc.append((p, locate(p, official_by_id, sent_idx)))
    extra_by_pair = add_mod_context(rows_loc, cap)
    context_rows: List[Row] = []
    for pair, loc in rows_loc:
        context_rows.append(contextual_row(pair, loc, cap, extra_by_pair.get(str(pair["pair_id"]), 0)))
    contextual_words = sum(r.words for r in context_rows)
    if contextual_words % WORDS_PER_OFFICIAL_ROW != 0:
        raise RuntimeError(f"contextual block still not divisible by 160: {contextual_words}")
    selected_example_ids = {int(p["example_id"]) for p in selected}
    filler_needed = TOTAL_WORDS - contextual_words
    if filler_needed <= 0:
        raise RuntimeError(f"contextual rows exceed 10M: {contextual_words}")
    filler_rows, filler_info = official_filler_rows(official_pool, selected_example_ids, filler_needed)
    treatment_pool = context_rows + filler_rows
    rng = random.Random(RNG_SEED + 2 + cap)
    rng.shuffle(treatment_pool)
    treatment_lengths = [r.words for r in treatment_pool]
    control_pool = official_length_matched_control(official_pool, treatment_lengths)
    if sum(treatment_lengths) != TOTAL_WORDS or sum(r.words for r in control_pool) != TOTAL_WORDS:
        raise RuntimeError("10M total mismatch")
    if treatment_lengths != [r.words for r in control_pool]:
        raise RuntimeError("control length sequence mismatch")

    pass_orders: List[List[int]] = []
    n = len(treatment_pool)
    for pass_i in range(PASSES):
        order = list(range(n))
        random.Random(RNG_SEED + 100 + pass_i).shuffle(order)
        pass_orders.append(order)

    corpus_dir = out_dir / "training_corpora"
    treat10 = corpus_dir / f"qwen_context_onepair_cap{cap}_10M.jsonl"
    treat100 = corpus_dir / f"qwen_context_onepair_cap{cap}_100M.jsonl"
    ctrl10 = corpus_dir / f"official_context_lengthmatched_cap{cap}_10M.jsonl"
    ctrl100 = corpus_dir / f"official_context_lengthmatched_cap{cap}_100M.jsonl"
    pair_meta = out_dir / f"contextual_pair_rows_cap{cap}_meta.jsonl"
    write_pool(treat10, treatment_pool, pair_meta)
    write_pool(ctrl10, control_pool)
    write_training(treat100, treatment_pool, pass_orders)
    write_training(ctrl100, control_pool, pass_orders)

    meta_rows = [r.meta for r in context_rows if r.meta is not None]
    by_source_words = collections.Counter()
    by_source_pairs = collections.Counter()
    for m in meta_rows:
        by_source_words[str(m["source"])] += int(m["words"])
        by_source_pairs[str(m["source"])] += 1
    treatment_source_words = collections.Counter()
    for r in treatment_pool:
        treatment_source_words[str(r.source)] += int(r.words)
    duplicate_example_counts = collections.Counter(int(p["example_id"]) for p in selected)
    payload = {
        "status": "CONTEXTUAL_ONE_PAIR_MATERIALIZED",
        "non_leakage_statement": "Uses only official training text plus research/028 selected Qwen rewrites already counted in training data. No official AoA/CDI words, child curves, AoA predictions, AoA scores, or downstream evaluation outputs are read or used.",
        "cap": cap,
        "selected_pairs": len(selected),
        "unique_selected_example_ids": len(selected_example_ids),
        "max_pairs_from_one_example_id": max(duplicate_example_counts.values()),
        "contextual_pair_rows": len(context_rows),
        "contextual_rows_total_words": contextual_words,
        "contextual_rows_word_fraction": round(contextual_words / TOTAL_WORDS, 6),
        "qwen_pair_words": sum(int(p["pair_words"]) for p in selected),
        "qwen_pair_word_fraction": round(sum(int(p["pair_words"]) for p in selected) / TOTAL_WORDS, 6),
        "official_context_words_inside_contextual_rows": sum(int(m["context_words"]) for m in meta_rows),
        "extra_context_words_for_160_divisibility": sum(extra_by_pair.values()),
        "rows_receiving_extra_context_for_divisibility": sum(1 for x in extra_by_pair.values() if x > 0),
        "official_filler_words": filler_needed,
        "official_filler_rows": len(filler_rows),
        "official_filler_info": filler_info,
        "treatment_pool_rows": len(treatment_pool),
        "control_pool_rows": len(control_pool),
        "row_length_sequence_matched": treatment_lengths == [r.words for r in control_pool],
        "pair_boundary_preserved": True,
        "one_pair_per_contextual_row": True,
        "row_words_never_exceed_160": max(treatment_lengths) <= WORDS_PER_OFFICIAL_ROW,
        "word_totals": {
            "treatment_10M": sum(treatment_lengths),
            "control_10M": sum(r.words for r in control_pool),
            "treatment_100M": TOTAL_WORDS * PASSES,
            "control_100M": TOTAL_WORDS * PASSES,
        },
        "contextual_row_word_stats": stats([r.words for r in context_rows]),
        "context_word_stats": stats([int(m["context_words"]) for m in meta_rows]),
        "left_context_taken_stats": stats([int(m["left_take"]) for m in meta_rows]),
        "right_context_taken_stats": stats([int(m["right_take"]) for m in meta_rows]),
        "pair_word_stats": stats([int(m["pair_words"]) for m in meta_rows]),
        "contextual_source_row_words": dict(sorted(by_source_words.items())),
        "contextual_source_pairs": dict(sorted(by_source_pairs.items())),
        "treatment_pool_source_words": dict(sorted(treatment_source_words.items())),
        "files": {
            "treatment_10M": str(treat10),
            "treatment_100M": str(treat100),
            "official_lengthmatched_10M": str(ctrl10),
            "official_lengthmatched_100M": str(ctrl100),
            "contextual_pair_meta": str(pair_meta),
        },
        "sha256": {
            "treatment_10M": sha_file(treat10),
            "treatment_100M": sha_file(treat100),
            "official_lengthmatched_10M": sha_file(ctrl10),
            "official_lengthmatched_100M": sha_file(ctrl100),
        },
        "intended_mechanism": "same-window generated second-view correspondence embedded in real official row context, testing whether official context restores BLiMP/EWoK/COMPS/Reading while preserving Supplement/Entity/SuperGLUE gains",
        "elapsed_sec": round(time.time() - t0, 3),
    }
    meta_path = out_dir / f"contextual_one_pair_cap{cap}_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"metadata": str(meta_path), **{k: payload[k] for k in ["cap", "contextual_rows_total_words", "contextual_rows_word_fraction", "qwen_pair_word_fraction", "official_filler_words", "row_length_sequence_matched"]}, "sha256_treatment_100M": payload["sha256"]["treatment_100M"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, nargs="+", default=[120])
    ap.add_argument("--out_root", default=str(_public_path('experiments/archive/compact_experience/data/contextual_one_pair')))
    args = ap.parse_args()
    results = []
    for cap in args.cap:
        if cap <= 0 or cap > WORDS_PER_OFFICIAL_ROW:
            raise SystemExit(f"cap must be in 1..160, got {cap}")
        out_dir = pathlib.Path(args.out_root) / f"cap{cap}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results.append(build(cap, out_dir))
    print(json.dumps({"status": "ok", "results": results}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
