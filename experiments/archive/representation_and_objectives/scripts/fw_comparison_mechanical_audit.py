#!/usr/bin/env python3
"""research: mechanical and provenance audit for compact-view vs source-breadth arms.

CPU-only.  This script checks that the two candidate corpora are genuinely a
matched data-mechanism comparison before any H100 training:
  * exact 10M word totals and identical row word sequence;
  * same shared compliant tokenizer;
  * tokenization/seq256/WWM-target geometry under that tokenizer;
  * source-breadth companion independence from compact-pair source hashes;
  * exact n=7/8/10 scored-text overlap for the components that differ.

The overlap scan uses official score text only as a post-construction provenance
measurement, never as a source-selection signal.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import re
import statistics
import time
from typing import Any, Iterable

from transformers import AutoTokenizer

ROOT = pathlib.Path(".").resolve()
WS = ROOT / "experiments/archive/representation_and_objectives"
COMPACT_10M = WS / "data/fw_full_arms/fw_preserved_compact_view_10M.jsonl"
BREADTH_10M = WS / "data/fw_source_breadth_arm/fw_preserved_source_breadth_10M.jsonl"
COMPACT_100M = WS / "data/fw_source_breadth_arm/fw_preserved_compact_view_100M.jsonl"
BREADTH_100M = WS / "data/fw_source_breadth_arm/fw_preserved_source_breadth_100M.jsonl"
USABLE_PAIRS = WS / "data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl"
BREADTH_SOURCES = WS / "data/fw_source_breadth_arm/source_breadth_companion_sources.jsonl"
BREADTH_ROW_META = WS / "data/fw_source_breadth_arm/source_breadth_row_meta.jsonl"
TOKENIZER_DIR = WS / "data/shared_tokenizer/shared_16k_tokenizer"
TOKENIZER_MANIFEST = WS / "data/shared_tokenizer/shared_tokenizer_manifest.json"
OUT_DIR = WS / "data/fw_comparison_mechanical_audit"
NOTE = (ROOT / 'research/notes/representation_and_objectives/fw_comparison_mechanical_audit.md')

PRISTINE_EVAL_ROOT = WS / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
GLOBALPIQA_EVAL_ROOT = WS / "data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval"
NS = (7, 8, 10)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
TEXT_KEYS = {
    "sentence", "sentence_good", "sentence_bad", "good_sentence", "bad_sentence",
    "context", "question", "passage", "paragraph", "premise", "hypothesis",
    "text", "text_a", "text_b", "input", "target", "option1", "option2",
    "answer", "correct", "incorrect", "choice1", "choice2", "query", "title",
    "article", "summary", "word", "stem", "ending0", "ending1", "ending2", "ending3",
    "prompt", "solution0", "solution1", "solution2", "solution3",
}
SKIP_KEYS = {"id", "idx", "uid", "guid", "label", "labels", "metadata", "meta", "path", "file", "filename", "source", "source_file", "example_id"}
STRICT_SCORE_TOP_DIRS = {"aoa", "blimp_filtered", "supplement_filtered", "ewok_filtered", "entity_tracking", "comps", "reading", "global_piqa_parallel", "global_piqa_nonparallel"}
STRICT_NON_SCORE_TOP_DIRS = {"vqa_filtered", "winoground_filtered"}
TOTAL_WORDS = 10_000_000
PASSES = 10


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wc(text: str) -> int:
    return len((text or "").split())


def toks(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def ngrams(ts: list[str], n: int) -> set[str]:
    if len(ts) < n:
        return set()
    return {" ".join(ts[i:i + n]) for i in range(len(ts) - n + 1)}


def norm_hash(text: str) -> str:
    return hashlib.sha256(" ".join((text or "").lower().split()).encode("utf-8")).hexdigest()[:32]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def arm_summary(path: pathlib.Path) -> dict[str, Any]:
    rows = 0
    words = 0
    source_words: collections.Counter[str] = collections.Counter()
    word_seq: list[int] = []
    source_seq: list[str] = []
    mismatches = []
    for i, r in enumerate(iter_jsonl(path), 1):
        rows += 1
        w = int(r.get("words") or wc(str(r.get("text") or "")))
        actual = wc(str(r.get("text") or ""))
        if w != actual and len(mismatches) < 20:
            mismatches.append({"line": i, "field_words": w, "actual_words": actual, "source": r.get("source")})
        words += w
        word_seq.append(w)
        source = str(r.get("source") or "")
        source_seq.append(source)
        source_words[source] += w
    return {
        "path": str(path),
        "exists": path.exists(),
        "rows": rows,
        "words": words,
        "sha256": sha256_file(path),
        "source_words": dict(source_words),
        "word_seq": word_seq,
        "source_seq": source_seq,
        "word_mismatches": mismatches,
    }


def stream_summary(path: pathlib.Path) -> dict[str, Any]:
    rows = 0
    words = 0
    source_words: collections.Counter[str] = collections.Counter()
    first_sources = []
    for r in iter_jsonl(path):
        rows += 1
        w = int(r.get("words") or wc(str(r.get("text") or "")))
        words += w
        source = str(r.get("source") or "")
        source_words[source] += w
        if len(first_sources) < 8:
            first_sources.append(source)
    return {"path": str(path), "exists": path.exists(), "rows": rows, "words": words, "sha256": sha256_file(path), "source_words": dict(source_words), "first_sources": first_sources}


def token_metrics(path: pathlib.Path, tokenizer, max_length: int = 256) -> dict[str, Any]:
    rows = 0
    words = 0
    tokens_untruncated = 0
    tokens_visible = 0
    rows_over_256 = 0
    wwm_groups_visible = 0
    source_metrics: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    for r in iter_jsonl(path):
        rows += 1
        text = str(r.get("text") or "")
        w = int(r.get("words") or wc(text))
        words += w
        source = str(r.get("source") or "")
        ids_full = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        ids_vis = ids_full[:max_length]
        tokens_untruncated += len(ids_full)
        tokens_visible += len(ids_vis)
        if len(ids_full) > max_length:
            rows_over_256 += 1
        group_count = 0
        prev = False
        for j, tid in enumerate(ids_vis):
            if int(tid) in special_ids:
                prev = False
                continue
            tok = tokenizer.convert_ids_to_tokens(int(tid))
            is_start = j == 0 or str(tok).startswith("Ġ") or str(tok).startswith("▁") or not prev
            if is_start:
                group_count += 1
            prev = True
        wwm_groups_visible += group_count
        source_metrics[source]["rows"] += 1
        source_metrics[source]["words"] += w
        source_metrics[source]["tokens_untruncated"] += len(ids_full)
        source_metrics[source]["tokens_visible"] += len(ids_vis)
        source_metrics[source]["rows_over_256"] += int(len(ids_full) > max_length)
        source_metrics[source]["wwm_groups_visible"] += group_count
    return {
        "rows": rows,
        "words": words,
        "tokens_untruncated": tokens_untruncated,
        "tokens_visible": tokens_visible,
        "tokens_per_word_untruncated": tokens_untruncated / max(1, words),
        "tokens_per_word_visible": tokens_visible / max(1, words),
        "rows_over_256": rows_over_256,
        "over_256_fraction": rows_over_256 / max(1, rows),
        "wwm_groups_visible": wwm_groups_visible,
        "expected_masked_wwm_groups_10_pass_0p15": wwm_groups_visible * PASSES * 0.15,
        "by_source": {k: dict(v) for k, v in source_metrics.items()},
    }


def iter_texts(obj: Any, parent_key: str = "") -> Iterable[str]:
    if isinstance(obj, str):
        pk = parent_key.lower()
        if pk in SKIP_KEYS:
            return
        if pk in TEXT_KEYS or len(toks(obj)) >= 5:
            yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_texts(x, parent_key)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in SKIP_KEYS:
                continue
            yield from iter_texts(v, str(k))


def classify_eval_file(rel: pathlib.Path) -> dict[str, Any]:
    top = rel.parts[0] if rel.parts else rel.name
    name = rel.name
    if top == "glue_filtered":
        if ".valid." in name:
            return {"top_dir": top, "strict_small_score_text": True}
        return {"top_dir": top, "strict_small_score_text": False}
    if top in STRICT_NON_SCORE_TOP_DIRS:
        return {"top_dir": top, "strict_small_score_text": False}
    if top in STRICT_SCORE_TOP_DIRS:
        return {"top_dir": top, "strict_small_score_text": True}
    return {"top_dir": top, "strict_small_score_text": False}


def eval_sources() -> list[tuple[pathlib.Path, pathlib.Path]]:
    srcs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for p in sorted(PRISTINE_EVAL_ROOT.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".csv", ".txt"}:
            srcs.append((p, p.relative_to(PRISTINE_EVAL_ROOT)))
    for task_dir in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        p = GLOBALPIQA_EVAL_ROOT / task_dir / "eng_latn.jsonl"
        if p.exists():
            srcs.append((p, pathlib.Path(task_dir) / "eng_latn.jsonl"))
    return srcs


def build_eval_index() -> tuple[list[dict[str, Any]], dict[int, dict[str, list[int]]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    ng_to_ids: dict[int, dict[str, list[int]]] = {n: collections.defaultdict(list) for n in NS}  # type: ignore[assignment]
    counts: collections.Counter[str] = collections.Counter()

    def add_record(rel: pathlib.Path, locator: str, text: str, cls: dict[str, Any]) -> None:
        if not cls.get("strict_small_score_text"):
            return
        ts = toks(text)
        if len(ts) < min(NS):
            return
        rid = len(records)
        records.append({"id": rid, "file": str(rel), "locator": locator, "text_prefix": text[:240], "num_tokens": len(ts), "top_dir": cls["top_dir"]})
        counts[cls["top_dir"]] += 1
        for n in NS:
            for ng in ngrams(ts, n):
                ng_to_ids[n][ng].append(rid)

    for p, rel in eval_sources():
        cls = classify_eval_file(rel)
        try:
            if p.suffix.lower() == ".jsonl":
                with p.open(encoding="utf-8") as f:
                    for li, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            for text in iter_texts(obj):
                                add_record(rel, f"line:{li}", text, cls)
                        except json.JSONDecodeError:
                            add_record(rel, f"line:{li}", line, cls)
            elif p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                for text in iter_texts(obj):
                    add_record(rel, "json", text, cls)
            elif p.suffix.lower() == ".csv":
                with p.open(encoding="utf-8", newline="") as f:
                    for ri, row in enumerate(csv.DictReader(f), 1):
                        for key, val in row.items():
                            if val and isinstance(val, str):
                                add_record(rel, f"row:{ri}:col:{key}", val, cls)
            else:
                add_record(rel, "txt", p.read_text(encoding="utf-8", errors="ignore"), cls)
        except Exception as exc:
            counts[f"_read_error::{rel}"] += 1
    return records, ng_to_ids, dict(counts)


def component_overlap(name: str, rows: list[dict[str, Any]], text_key: str, word_key: str, records: list[dict[str, Any]], ng_to_ids: dict[int, dict[str, list[int]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"component_kind": name, "components": len(rows), "words": sum(int(r.get(word_key) or wc(str(r.get(text_key) or ""))) for r in rows)}
    details = []
    for n in NS:
        comps_any = 0
        unique_ngrams: set[str] = set()
        hit_count = 0
        eval_hits = 0
        top_dirs: collections.Counter[str] = collections.Counter()
        for idx, r in enumerate(rows):
            text = str(r.get(text_key) or "")
            matched_here = False
            for ng in ngrams(toks(text), n):
                ids = ng_to_ids[n].get(ng)
                if not ids:
                    continue
                matched_here = True
                unique_ngrams.add(ng)
                hit_count += 1
                uniq_ids = set(ids)
                eval_hits += len(uniq_ids)
                tops = collections.Counter(records[rid]["top_dir"] for rid in uniq_ids)
                top_dirs.update(tops)
                if len(details) < 80:
                    first = records[next(iter(uniq_ids))]
                    details.append({"n": n, "ngram": ng, "component_kind": name, "component_index": idx, "component_hash": r.get("norm_hash") or norm_hash(text), "eval_top_dir": first["top_dir"], "eval_file": first["file"], "component_prefix": text[:200], "eval_prefix": first["text_prefix"]})
            if matched_here:
                comps_any += 1
        out[f"components_with_any_match_n{n}"] = comps_any
        out[f"unique_matched_ngrams_n{n}"] = len(unique_ngrams)
        out[f"component_ngram_hit_count_n{n}"] = hit_count
        out[f"eval_record_hit_count_n{n}"] = eval_hits
        out[f"top_dir_hits_n{n}"] = dict(top_dirs)
    out["details_sample"] = details
    return out


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "min": xs[0], "max": xs[-1], "sum": sum(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-overlap", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    compact = arm_summary(COMPACT_10M)
    breadth = arm_summary(BREADTH_10M)
    compact_words_seq = compact.pop("word_seq")
    breadth_words_seq = breadth.pop("word_seq")
    compact_source_seq = compact.pop("source_seq")
    breadth_source_seq = breadth.pop("source_seq")
    word_seq_match = compact_words_seq == breadth_words_seq
    source_seq_match_except_fw = all((a == b) or (a == "fw_preserved_compact_view" and b == "fw_preserved_source_breadth") for a, b in zip(compact_source_seq, breadth_source_seq))

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    tok_manifest = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
    tok_hash = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    compact_tok = token_metrics(COMPACT_10M, tok)
    breadth_tok = token_metrics(BREADTH_10M, tok)

    compact_stream = stream_summary(COMPACT_100M) if COMPACT_100M.exists() else None
    breadth_stream = stream_summary(BREADTH_100M) if BREADTH_100M.exists() else None

    pairs = read_jsonl(USABLE_PAIRS)
    breadth_sources = read_jsonl(BREADTH_SOURCES)
    row_meta = read_jsonl(BREADTH_ROW_META)
    pair_hashes = {str(r.get("norm_hash") or norm_hash(str(r.get("source_text") or ""))) for r in pairs}
    breadth_hashes = {str(r.get("norm_hash") or norm_hash(str(r.get("text") or ""))) for r in breadth_sources}
    hash_overlap = sorted(pair_hashes & breadth_hashes)[:100]

    # Prefix check on the FineWeb rows only: both arm rows should begin with the
    # same packed selected source text stored in row_meta.
    prefix_failures = []
    compact_fw_rows = [r for r in iter_jsonl(COMPACT_10M) if r.get("source") == "fw_preserved_compact_view"]
    breadth_fw_rows = [r for r in iter_jsonl(BREADTH_10M) if r.get("source") == "fw_preserved_source_breadth"]
    for i, m in enumerate(row_meta[:len(compact_fw_rows)]):
        src = str(m.get("source_text") or "")
        if not str(compact_fw_rows[i].get("text") or "").startswith(src):
            prefix_failures.append({"row": i, "arm": "compact"})
        if not str(breadth_fw_rows[i].get("text") or "").startswith(src):
            prefix_failures.append({"row": i, "arm": "breadth"})
        if len(prefix_failures) >= 20:
            break

    overlap_summary: dict[str, Any] | None = None
    if not args.skip_overlap:
        records, ng_to_ids, eval_counts = build_eval_index()
        pair_sources = [{"norm_hash": r.get("norm_hash"), "text": r.get("source_text"), "words": r.get("source_words")} for r in pairs]
        compact_rewrites = [{"norm_hash": r.get("norm_hash"), "text": r.get("rewrite_text"), "words": r.get("rewrite_words")} for r in pairs]
        breadth_companions = [{"norm_hash": r.get("norm_hash"), "text": r.get("text"), "words": r.get("words")} for r in breadth_sources]
        comps = [
            component_overlap("common_fineweb_selected_sources", pair_sources, "text", "words", records, ng_to_ids),
            component_overlap("compact_rewrite_companions", compact_rewrites, "text", "words", records, ng_to_ids),
            component_overlap("source_breadth_companions", breadth_companions, "text", "words", records, ng_to_ids),
        ]
        overlap_summary = {
            "strict_eval_record_counts_by_top": eval_counts,
            "strict_eval_records_total": len(records),
            "strict_eval_unique_ngrams": {str(n): len(ng_to_ids[n]) for n in NS},
            "component_kind_summary": comps,
        }

    payload = {
        "status": "FW_COMPACT_BREADTH_MECHANICAL_AUDIT",
        "created_utc": now_utc(),
        "purpose": "Pre-training mechanical/provenance audit for compact_view vs source_breadth with one shared tokenizer.",
        "arms": {"compact_view": compact, "source_breadth": breadth},
        "matched_structure": {
            "word_seq_match": word_seq_match,
            "source_seq_match_except_fw_label": source_seq_match_except_fw,
            "rows_match": compact["rows"] == breadth["rows"],
            "words_match_10m": compact["words"] == breadth["words"] == TOTAL_WORDS,
            "fineweb_rows_compact": len(compact_fw_rows),
            "fineweb_rows_breadth": len(breadth_fw_rows),
            "common_source_prefix_failures_sample": prefix_failures,
            "breadth_companion_hash_overlap_with_pair_sources_count": len(pair_hashes & breadth_hashes),
            "breadth_companion_hash_overlap_with_pair_sources_sample": hash_overlap,
        },
        "tokenizer": {
            "dir": str(TOKENIZER_DIR),
            "tokenizer_json_sha256": tok_hash,
            "manifest_tokenizer_json_sha256": tok_manifest.get("tokenizer", {}).get("hashes", {}).get("tokenizer.json"),
            "len": len(tok),
            "vocab_size": tok.vocab_size,
            "special_ids": tok.all_special_ids,
        },
        "token_metrics": {"compact_view": compact_tok, "source_breadth": breadth_tok},
        "training_streams": {"compact_view": compact_stream, "source_breadth": breadth_stream},
        "component_sets": {
            "usable_pairs": len(pairs),
            "pair_source_words": sum(int(r.get("source_words") or 0) for r in pairs),
            "compact_rewrite_words": sum(int(r.get("rewrite_words") or 0) for r in pairs),
            "breadth_sources": len(breadth_sources),
            "breadth_source_words": sum(int(r.get("words") or 0) for r in breadth_sources),
            "compact_rewrite_word_stats": stat([int(r.get("rewrite_words") or 0) for r in pairs]),
            "breadth_source_word_stats": stat([int(r.get("words") or 0) for r in breadth_sources]),
        },
        "overlap_summary": overlap_summary,
        "files": {
            "json": str(OUT_DIR / "fw_compact_breadth_mechanical_audit.json"),
            "note": str(NOTE),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    out_json = OUT_DIR / "fw_compact_breadth_mechanical_audit.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def overlap_line(kind: str) -> str:
        if not overlap_summary:
            return "overlap skipped"
        rec = next(x for x in overlap_summary["component_kind_summary"] if x["component_kind"] == kind)
        return ", ".join(f"n{n}: {rec[f'components_with_any_match_n{n}']} comps/{rec[f'unique_matched_ngrams_n{n}']} unique" for n in NS)

    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(
        "# research — compact vs source-breadth mechanical audit\n\n"
        f"- Both 10M arms have {compact['rows']:,} rows and {compact['words']:,} words; row word sequence match: {word_seq_match}.\n"
        f"- Source labels match except the FineWeb arm label: {source_seq_match_except_fw}.\n"
        f"- Shared tokenizer `{TOKENIZER_DIR}` has len={len(tok)}, tokenizer.json SHA `{tok_hash}`.\n"
        f"- Compact tokens/word visible: {compact_tok['tokens_per_word_visible']:.4f}; breadth tokens/word visible: {breadth_tok['tokens_per_word_visible']:.4f}.\n"
        f"- Compact rows over 256 tokens: {compact_tok['rows_over_256']:,}; breadth rows over 256 tokens: {breadth_tok['rows_over_256']:,}.\n"
        f"- WWM groups visible per pass: compact {compact_tok['wwm_groups_visible']:,}, breadth {breadth_tok['wwm_groups_visible']:,}.\n"
        f"- Breadth companion source hashes overlapping selected compact-pair source hashes: {len(pair_hashes & breadth_hashes)}.\n"
        f"- Exact score-text overlap common sources: {overlap_line('common_fineweb_selected_sources')}.\n"
        f"- Exact score-text overlap compact rewrites: {overlap_line('compact_rewrite_companions')}.\n"
        f"- Exact score-text overlap source-breadth companions: {overlap_line('source_breadth_companions')}.\n\n"
        f"JSON: `{out_json}`\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": payload["status"],
        "word_seq_match": word_seq_match,
        "source_seq_match_except_fw_label": source_seq_match_except_fw,
        "compact_words": compact["words"],
        "breadth_words": breadth["words"],
        "compact_visible_tpw": round(compact_tok["tokens_per_word_visible"], 4),
        "breadth_visible_tpw": round(breadth_tok["tokens_per_word_visible"], 4),
        "breadth_pair_hash_overlap": len(pair_hashes & breadth_hashes),
        "overlap_compact_rewrites_n7": None if not overlap_summary else next(x for x in overlap_summary["component_kind_summary"] if x["component_kind"] == "compact_rewrite_companions")["components_with_any_match_n7"],
        "overlap_breadth_n7": None if not overlap_summary else next(x for x in overlap_summary["component_kind_summary"] if x["component_kind"] == "source_breadth_companions")["components_with_any_match_n7"],
        "json": str(out_json),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
