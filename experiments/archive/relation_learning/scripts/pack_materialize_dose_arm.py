#!/usr/bin/env python3
"""research: select, pack, and materialize aligned-restatement dose arms.

The dose experiment asks whether the validated aligned-restatement relation keeps
paying when additional source->rewrite pairs are added above the inherited COMPACT_EXPERIENCE
ALN block.  This materializer preserves the compact-view-reinvest 100M topology:
row count, pass order, inherited qwen_pair_packed rows, compact-view-reinvest rows,
and total words stay fixed.  New pair rows replace ordinary filler slots only.

Each replacement slot remains a 160-word row.  The beginning of the row contains
one or more intact new source+rewrite pairs packed with the COMPACT_EXPERIENCE greedy logic;
any leftover words are filled from the ordinary filler row that was displaced.
Thus the net filler displacement equals the exact added pair-word budget, while
training row count and total word exposure remain identical to the base stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

ROOT = pathlib.Path.cwd()
BASE_STREAM = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
BASE_META = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"
TOKENIZER = ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
OUT_DEFAULT = ROOT / "experiments/archive/relation_learning/data/restatement_dose_streams"
EXPECTED_BASE_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
ROWS_PER_PASS = 64_740
PASSES = 10
TOTAL_WORDS_PER_PASS = 10_000_000
INHERITED_PAIR_WORDS = 1_656_800
ROW_WORDS = 160
MAX_PACK_PAIR_WORDS = 160
FILLER_SOURCES = {"childes", "gutenberg", "open_subtitles", "simple_wiki", "bnc_spoken", "switchboard"}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_jsonl_rows(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON in {path} line {line_no}: {e}") from e
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def wc(x: str) -> int:
    return len(str(x).split())


def stat(xs: list[float]) -> dict[str, Any]:
    ys = [float(x) for x in xs if math.isfinite(float(x))]
    if not ys:
        return {"n": 0}
    ys = sorted(ys)
    p95 = ys[min(len(ys) - 1, int(math.ceil(0.95 * len(ys))) - 1)]
    return {
        "n": len(ys),
        "min": round(min(ys), 4),
        "mean": round(statistics.mean(ys), 4),
        "median": round(statistics.median(ys), 4),
        "p95": round(p95, 4),
        "max": round(max(ys), 4),
    }


def load_accepted(paths: list[pathlib.Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path_i, p in enumerate(paths):
        for rec_i, r in enumerate(read_jsonl(p)):
            original_id = str(r.get("original_id") or r.get("pair_id") or "")
            if not original_id:
                raise RuntimeError(f"missing original_id in {p} row {rec_i}")
            if original_id in seen:
                continue
            seen.add(original_id)
            original = " ".join(str(r.get("original", "")).split())
            rewrite = " ".join(str(r.get("rewrite", "")).split())
            if not original or not rewrite:
                raise RuntimeError(f"empty original/rewrite in {p} row {rec_i}")
            ow, rw = wc(original), wc(rewrite)
            pair_words = int(r.get("pair_words") or (ow + rw))
            if pair_words != ow + rw:
                raise RuntimeError(f"pair_words mismatch {original_id}: {pair_words} vs {ow}+{rw}")
            rr = dict(r)
            rr.update({
                "pair_id": str(r.get("pair_id") or f"dose58_{original_id}"),
                "original_id": original_id,
                "original": original,
                "rewrite": rewrite,
                "original_words": ow,
                "rewrite_words": rw,
                "pair_words": pair_words,
                "input_file_rank": path_i,
                "input_row_rank": rec_i,
            })
            if pair_words > MAX_PACK_PAIR_WORDS:
                raise RuntimeError(f"pair {original_id} has {pair_words} words > max row {MAX_PACK_PAIR_WORDS}")
            out.append(rr)
    return out


def exact_select(pairs: list[dict[str, Any]], target_words: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    unselected: list[dict[str, Any]] = []
    total = 0
    for p in shuffled:
        w = int(p["pair_words"])
        if total + w <= target_words:
            selected.append(p)
            total += w
        else:
            unselected.append(p)
    if total == target_words:
        return selected
    # Remove a suffix and refill the small gap with exact DP.  Since pair weights
    # are small and the greedy remainder is < max pair size, the gap after removing
    # 50-800 pairs is only a few thousand to a few tens of thousands.
    for k in [50, 100, 200, 400, 800, 1200]:
        if k >= len(selected):
            continue
        base = selected[:-k]
        removed = selected[-k:]
        gap = target_words - sum(int(p["pair_words"]) for p in base)
        pool = removed + unselected[:max(2000, 4 * k)]
        reachable: dict[int, tuple[int, int] | None] = {0: None}
        for idx, p in enumerate(pool):
            w = int(p["pair_words"])
            # Snapshot avoids reusing this item.
            for s in list(reachable.keys())[::-1]:
                ns = s + w
                if ns <= gap and ns not in reachable:
                    reachable[ns] = (s, idx)
            if gap in reachable:
                break
        if gap in reachable:
            chosen_idx: set[int] = set()
            cur = gap
            while cur:
                prev = reachable[cur]
                if prev is None:
                    raise RuntimeError("internal DP reconstruction error")
                prev_sum, idx = prev
                chosen_idx.add(idx)
                cur = prev_sum
            return base + [pool[i] for i in sorted(chosen_idx)]
    raise RuntimeError(f"Could not select exact {target_words} words; greedy total was {total}")


def pack_pairs(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cur_segments: list[str] = []
    cur_pair_ids: list[str] = []
    cur_original_ids: list[str] = []
    cur_words = 0
    cur_sources: Counter[str] = Counter()
    for p in selected:
        w = int(p["pair_words"])
        seg = f"{p['original']} {p['rewrite']}"
        if wc(seg) != w:
            raise RuntimeError(f"segment word mismatch {p['pair_id']}")
        if cur_words and cur_words + w > MAX_PACK_PAIR_WORDS:
            rows.append({
                "text_pairs_only": " ".join(cur_segments),
                "pair_words": cur_words,
                "pair_ids": list(cur_pair_ids),
                "original_ids": list(cur_original_ids),
                "n_pairs": len(cur_pair_ids),
                "component_sources": dict(cur_sources),
            })
            cur_segments, cur_pair_ids, cur_original_ids, cur_words, cur_sources = [], [], [], 0, Counter()
        cur_segments.append(seg)
        cur_pair_ids.append(str(p["pair_id"]))
        cur_original_ids.append(str(p["original_id"]))
        cur_words += w
        cur_sources[str(p.get("source", ""))] += w
    if cur_segments:
        rows.append({
            "text_pairs_only": " ".join(cur_segments),
            "pair_words": cur_words,
            "pair_ids": list(cur_pair_ids),
            "original_ids": list(cur_original_ids),
            "n_pairs": len(cur_pair_ids),
            "component_sources": dict(cur_sources),
        })
    for i, r in enumerate(rows):
        if r["pair_words"] != wc(r["text_pairs_only"]):
            raise RuntimeError(f"packed row {i} word mismatch")
        if r["pair_words"] > MAX_PACK_PAIR_WORDS:
            raise RuntimeError(f"packed row {i} overflow {r['pair_words']}")
    return rows


def spaced_indices(indices: list[int], need: int) -> list[int]:
    if need > len(indices):
        raise RuntimeError(f"need {need} filler slots but only {len(indices)} candidates")
    if need == 0:
        return []
    # Choose approximately evenly across the pass to avoid moving a block of the
    # curriculum/order more than necessary.
    chosen = []
    used = set()
    n = len(indices)
    for j in range(need):
        pos = int((j + 0.5) * n / need)
        pos = min(n - 1, max(0, pos))
        # Resolve rare duplicates by walking right then left.
        if pos in used:
            r = pos + 1
            while r < n and r in used:
                r += 1
            if r < n:
                pos = r
            else:
                l = pos - 1
                while l >= 0 and l in used:
                    l -= 1
                if l < 0:
                    raise RuntimeError("could not assign unique spaced index")
                pos = l
        used.add(pos)
        chosen.append(indices[pos])
    return sorted(chosen)


def load_tokenizer(path: pathlib.Path):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    except Exception as e:  # pragma: no cover - tokenizer failure should be recorded, not fatal.
        print(f"[WARN] tokenizer load failed: {e}", flush=True)
        return None


def process_pass(rows: list[dict[str, Any]], pass_i: int, packed_rows: list[dict[str, Any]], dose_name: str, tokenizer: Any | None, write_meta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    before_words = sum(int(r["words"]) for r in rows)
    candidates = [i for i, r in enumerate(rows) if str(r.get("source", "")) in FILLER_SOURCES and int(r.get("words", 0)) == ROW_WORDS]
    replace_idxs = spaced_indices(candidates, len(packed_rows))
    src_displaced: Counter[str] = Counter()
    topup_src: Counter[str] = Counter()
    token_counts: list[int] = []
    token_over_256 = 0
    for local_j, (idx, pack) in enumerate(zip(replace_idxs, packed_rows)):
        old = rows[idx]
        old_words = str(old["text"]).split()
        pair_words = int(pack["pair_words"])
        topup_n = ROW_WORDS - pair_words
        if topup_n < 0:
            raise RuntimeError(f"packed pair row over {ROW_WORDS} words")
        if len(old_words) < topup_n:
            raise RuntimeError(f"not enough topup words in row {idx}")
        topup = " ".join(old_words[:topup_n])
        new_text = pack["text_pairs_only"] if not topup else pack["text_pairs_only"] + " " + topup
        if wc(new_text) != ROW_WORDS:
            raise RuntimeError(f"new row word mismatch pass {pass_i} idx {idx}: {wc(new_text)}")
        old_source = str(old.get("source", ""))
        src_displaced[old_source] += ROW_WORDS
        topup_src[old_source] += topup_n
        rows[idx] = {
            "text": new_text,
            "words": ROW_WORDS,
            "example_id": 880000000 + pass_i * 100000 + local_j,
            "source": f"{dose_name}_qwen_pair_packed",
        }
        tok_len = None
        if tokenizer is not None:
            tok_len = len(tokenizer.encode(new_text, add_special_tokens=False))
            token_counts.append(tok_len)
            if tok_len > 256:
                token_over_256 += 1
        write_meta_rows.append({
            "pass": pass_i,
            "row_index_in_pass": idx,
            "global_row_index": pass_i * ROWS_PER_PASS + idx,
            "new_example_id": rows[idx]["example_id"],
            "old_example_id": old.get("example_id"),
            "old_source": old_source,
            "pair_words": pair_words,
            "topup_words": topup_n,
            "row_words": ROW_WORDS,
            "n_pairs": pack["n_pairs"],
            "pair_ids": pack["pair_ids"],
            "original_ids": pack["original_ids"],
            "component_sources": pack["component_sources"],
            "token_count_no_special": tok_len,
        })
    after_words = sum(int(r["words"]) for r in rows)
    return {
        "pass": pass_i,
        "before_words": before_words,
        "after_words": after_words,
        "candidate_filler_rows_160w": len(candidates),
        "replaced_rows": len(packed_rows),
        "added_pair_words": sum(int(r["pair_words"]) for r in packed_rows),
        "topup_words": sum(ROW_WORDS - int(r["pair_words"]) for r in packed_rows),
        "displaced_source_words": dict(src_displaced),
        "topup_source_words": dict(topup_src),
        "token_count_stats": stat([float(x) for x in token_counts]),
        "token_rows_over_256": token_over_256,
    }


def materialize(base_stream: pathlib.Path, packed_rows: list[dict[str, Any]], dose_name: str, out_stream: pathlib.Path | None, tokenizer: Any | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    pass_stats: list[dict[str, Any]] = []
    row_meta: list[dict[str, Any]] = []
    output_source_words: Counter[str] = Counter()
    fout = out_stream.open("w", encoding="utf-8") if out_stream is not None else None
    try:
        buf: list[dict[str, Any]] = []
        pass_i = 0
        with base_stream.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                buf.append(json.loads(line))
                if len(buf) == ROWS_PER_PASS:
                    ps = process_pass(buf, pass_i, packed_rows, dose_name, tokenizer, row_meta)
                    pass_stats.append(ps)
                    for r in buf:
                        output_source_words[str(r.get("source", ""))] += int(r["words"])
                        if fout is not None:
                            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    pass_i += 1
                    buf = []
        if buf:
            ps = process_pass(buf, pass_i, packed_rows, dose_name, tokenizer, row_meta)
            pass_stats.append(ps)
            for r in buf:
                output_source_words[str(r.get("source", ""))] += int(r["words"])
                if fout is not None:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        if pass_i + (1 if buf else 0) not in {PASSES}:  # normally exactly ten passes
            pass
    finally:
        if fout is not None:
            fout.close()
    return pass_stats, row_meta, output_source_words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accepted", nargs="+", required=True, help="Accepted dose pair JSONL files")
    ap.add_argument("--dose-name", required=True, help="Label, e.g. dose21 or dose25")
    ap.add_argument("--target-pair-words", type=int, required=True, help="Added pair words per 10M pass")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--base-stream", default=str(BASE_STREAM))
    ap.add_argument("--base-meta", default=str(BASE_META))
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--selection-seed", type=int, default=58021)
    ap.add_argument("--write-stream", action="store_true")
    ap.add_argument("--skip-base-sha", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir) / args.dose_name
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_paths = [pathlib.Path(p) for p in args.accepted]
    accepted = load_accepted(accepted_paths)
    available_words = sum(int(p["pair_words"]) for p in accepted)
    if available_words < args.target_pair_words:
        raise RuntimeError(f"available pair words {available_words} < target {args.target_pair_words}")
    selected = exact_select(accepted, args.target_pair_words, args.selection_seed + args.target_pair_words)
    selected_words = sum(int(p["pair_words"]) for p in selected)
    if selected_words != args.target_pair_words:
        raise RuntimeError(f"selection mismatch {selected_words} != {args.target_pair_words}")
    packed = pack_pairs(selected)

    base_stream = pathlib.Path(args.base_stream)
    base_sha = "skipped"
    base_sha_ok = True
    if not args.skip_base_sha:
        base_sha = sha256_file(base_stream)
        base_sha_ok = base_sha == EXPECTED_BASE_SHA
        if not base_sha_ok:
            raise RuntimeError(f"base SHA mismatch {base_sha} != {EXPECTED_BASE_SHA}")

    tokenizer = load_tokenizer(pathlib.Path(args.tokenizer))
    out_stream = out_dir / f"{args.dose_name}_compact_view_reinvest_100M.jsonl" if args.write_stream else None
    pass_stats, row_meta, output_source_words = materialize(base_stream, packed, args.dose_name, out_stream, tokenizer)
    row_meta_path = out_dir / f"{args.dose_name}_dose_rows_meta.jsonl"
    selected_path = out_dir / f"{args.dose_name}_selected_pairs.jsonl"
    packed_path = out_dir / f"{args.dose_name}_packed_pair_rows.jsonl"
    write_jsonl(selected_path, selected)
    write_jsonl(packed_path, packed)
    write_jsonl(row_meta_path, row_meta)

    total_output_words = sum(output_source_words.values())
    if total_output_words != TOTAL_WORDS_PER_PASS * PASSES:
        raise RuntimeError(f"output words {total_output_words} != {TOTAL_WORDS_PER_PASS * PASSES}")
    pass_word_ok = all(ps["before_words"] == TOTAL_WORDS_PER_PASS and ps["after_words"] == TOTAL_WORDS_PER_PASS for ps in pass_stats)
    per_pass_added = [ps["added_pair_words"] for ps in pass_stats]
    if any(x != args.target_pair_words for x in per_pass_added):
        raise RuntimeError(f"per-pass added words not constant: {per_pass_added[:5]}")

    pair_source_words = Counter()
    pair_source_counts = Counter()
    for p in selected:
        pair_source_words[str(p.get("source", ""))] += int(p["pair_words"])
        pair_source_counts[str(p.get("source", ""))] += 1
    meta = {
        "status": "RESTATEMENT_DOSE_MATERIALIZED" if args.write_stream else "RESTATEMENT_DOSE_DRYRUN",
        "created_utc": now_utc(),
        "dose_name": args.dose_name,
        "accepted_files": [str(p) for p in accepted_paths],
        "accepted_pairs_loaded_unique": len(accepted),
        "accepted_pair_words_available": available_words,
        "target_added_pair_words_per_10M": args.target_pair_words,
        "selected_pairs": len(selected),
        "selected_pair_words": selected_words,
        "inherited_pair_words_per_10M": INHERITED_PAIR_WORDS,
        "total_pair_words_per_10M": INHERITED_PAIR_WORDS + selected_words,
        "total_pair_fraction_per_10M": round((INHERITED_PAIR_WORDS + selected_words) / TOTAL_WORDS_PER_PASS, 6),
        "selection_seed_effective": args.selection_seed + args.target_pair_words,
        "selection_sha256_virtual": sha256_jsonl_rows(selected),
        "packed_rows_per_pass": len(packed),
        "packed_pair_word_stats": stat([float(r["pair_words"]) for r in packed]),
        "packed_n_pairs_stats": stat([float(r["n_pairs"]) for r in packed]),
        "topup_words_per_pass": sum(ROW_WORDS - int(r["pair_words"]) for r in packed),
        "replacement_rows_per_pass": len(packed),
        "net_filler_displacement_words_per_pass": selected_words,
        "base_stream": str(base_stream),
        "base_sha256": base_sha,
        "base_sha_ok": base_sha_ok,
        "base_meta": str(args.base_meta),
        "write_stream": bool(args.write_stream),
        "output_stream": str(out_stream) if out_stream else None,
        "output_stream_sha256": sha256_file(out_stream) if out_stream else None,
        "row_count_preserved": True,
        "rows_per_pass": ROWS_PER_PASS,
        "passes": len(pass_stats),
        "pass_word_ok": pass_word_ok,
        "output_total_words": total_output_words,
        "output_source_words": dict(output_source_words),
        "selected_pair_source_counts": dict(pair_source_counts),
        "selected_pair_source_words": dict(pair_source_words),
        "token_rows_over_256_total": sum(int(ps.get("token_rows_over_256", 0)) for ps in pass_stats),
        "pass_stats": pass_stats,
        "outputs": {
            "selected_pairs": str(selected_path),
            "packed_pair_rows": str(packed_path),
            "dose_rows_meta": str(row_meta_path),
        },
        "sha256": {
            "selected_pairs": sha256_file(selected_path),
            "packed_pair_rows": sha256_file(packed_path),
            "dose_rows_meta": sha256_file(row_meta_path),
        },
        "scientific_note": "Additional aligned-restatement pair words replace ordinary filler while inherited ALN qwen_pair_packed rows and compact-view-reinvest rows remain untouched. Rows are topped up with words from displaced filler to preserve row count and total exposure; metadata records pair versus topup accounting.",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path = out_dir / f"{args.dose_name}_materialization_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ["status", "dose_name", "accepted_pairs_loaded_unique", "accepted_pair_words_available", "target_added_pair_words_per_10M", "selected_pairs", "selected_pair_words", "total_pair_fraction_per_10M", "packed_rows_per_pass", "topup_words_per_pass", "token_rows_over_256_total", "write_stream", "output_stream", "elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
