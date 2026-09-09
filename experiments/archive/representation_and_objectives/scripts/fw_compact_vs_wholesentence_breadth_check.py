#!/usr/bin/env python3
"""research: mechanical/provenance check for compact-view vs repaired whole-sentence
source-breadth arms before any H100 training.

This reuses the research comparison machinery but points the breadth side to the
repaired arm from research, where independent FineWeb companions are packed as
whole sentences inside each row.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

from transformers import AutoTokenizer

ROOT = pathlib.Path(".").resolve()
WS = ROOT / "experiments/archive/representation_and_objectives"
AUDIT = WS / "scripts/fw_comparison_mechanical_audit.py"
BUILD = WS / "scripts/fw_source_breadth_arm.py"
COMPACT_10M = WS / "data/fw_full_arms/fw_preserved_compact_view_10M.jsonl"
COMPACT_100M = WS / "data/fw_source_breadth_arm/fw_preserved_compact_view_100M.jsonl"
BREADTH_10M = WS / "data/fw_source_breadth_wholesentence_arm/fw_preserved_source_breadth_wholesentence_10M.jsonl"
BREADTH_100M = WS / "data/fw_source_breadth_wholesentence_arm/fw_preserved_source_breadth_wholesentence_100M.jsonl"
BREADTH_SOURCES = WS / "data/fw_source_breadth_wholesentence_arm/source_breadth_wholesentence_companion_sources.jsonl"
BREADTH_ROW_META = WS / "data/fw_source_breadth_wholesentence_arm/source_breadth_wholesentence_row_meta.jsonl"
REPAIR_MANIFEST = WS / "data/fw_source_breadth_wholesentence_arm/fw_source_breadth_wholesentence_manifest.json"
TOKENIZER_DIR = WS / "data/shared_tokenizer/shared_16k_tokenizer"
TOKENIZER_MANIFEST = WS / "data/shared_tokenizer/shared_tokenizer_manifest.json"
OUT_DIR = WS / "data/fw_compact_vs_wholesentence_breadth_check"
NOTE = (ROOT / 'research/notes/representation_and_objectives/fw_compact_vs_wholesentence_breadth_check.md')
COMPACT_LABEL = "fw_preserved_compact_view"
BREADTH_LABEL = "fw_preserved_source_breadth_wholesentence"
TOTAL_WORDS = 10_000_000
PASSES = 10


def import_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len((text or "").split())


def arm_summary(path: pathlib.Path) -> dict[str, Any]:
    rows = 0
    words = 0
    source_words: collections.Counter[str] = collections.Counter()
    word_seq: list[int] = []
    source_seq: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for i, r in enumerate(iter_jsonl(path), 1):
        rows += 1
        text = str(r.get("text") or "")
        w = int(r.get("words") or wc(text))
        actual = wc(text)
        if w != actual and len(mismatches) < 20:
            mismatches.append({"line": i, "field_words": w, "actual_words": actual, "source": r.get("source")})
        words += w
        word_seq.append(w)
        src = str(r.get("source") or "")
        source_seq.append(src)
        source_words[src] += w
    return {"path": str(path), "exists": path.exists(), "rows": rows, "words": words, "sha256": sha256_file(path), "source_words": dict(source_words), "word_seq": word_seq, "source_seq": source_seq, "word_mismatches": mismatches}


def stream_summary(path: pathlib.Path) -> dict[str, Any]:
    rows = 0
    words = 0
    source_words: collections.Counter[str] = collections.Counter()
    first_example_ids: list[int] = []
    first_sources: list[str] = []
    for r in iter_jsonl(path):
        rows += 1
        w = int(r.get("words") or wc(str(r.get("text") or "")))
        words += w
        source_words[str(r.get("source") or "")] += w
        if len(first_example_ids) < 8:
            first_example_ids.append(int(r.get("example_id") or 0))
            first_sources.append(str(r.get("source") or ""))
    return {"path": str(path), "exists": path.exists(), "rows": rows, "words": words, "sha256": sha256_file(path), "source_words": dict(source_words), "first_example_ids": first_example_ids, "first_sources": first_sources}


def token_metrics(path: pathlib.Path, tokenizer, max_length: int = 256) -> dict[str, Any]:
    rows = 0
    words = 0
    tokens_untruncated = 0
    tokens_visible = 0
    rows_over_256 = 0
    wwm_groups_visible = 0
    by_source: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
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
        over = len(ids_full) > max_length
        rows_over_256 += int(over)
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
        by_source[source]["rows"] += 1
        by_source[source]["words"] += w
        by_source[source]["tokens_untruncated"] += len(ids_full)
        by_source[source]["tokens_visible"] += len(ids_vis)
        by_source[source]["rows_over_256"] += int(over)
        by_source[source]["wwm_groups_visible"] += group_count
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
        "by_source": {k: dict(v) for k, v in by_source.items()},
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aud = import_module(AUDIT, "audit_for_step104")
    build = import_module(BUILD, "build_for_step104_check")

    compact = arm_summary(COMPACT_10M)
    breadth = arm_summary(BREADTH_10M)
    compact_word_seq = compact.pop("word_seq")
    breadth_word_seq = breadth.pop("word_seq")
    compact_source_seq = compact.pop("source_seq")
    breadth_source_seq = breadth.pop("source_seq")
    word_seq_match = compact_word_seq == breadth_word_seq
    source_seq_match_except_fw_label = all((a == b) or (a == COMPACT_LABEL and b == BREADTH_LABEL) for a, b in zip(compact_source_seq, breadth_source_seq)) and len(compact_source_seq) == len(breadth_source_seq)

    compact_stream = stream_summary(COMPACT_100M)
    breadth_stream = stream_summary(BREADTH_100M)
    repair_manifest = json.loads(REPAIR_MANIFEST.read_text(encoding="utf-8"))
    stream_orders_match = bool((repair_manifest.get("training_streams") or {}).get("pass_orders_match_existing_compact"))
    compact_stream_rec = ((repair_manifest.get("training_streams") or {}).get("compact_view_existing") or {})
    breadth_stream_rec = ((repair_manifest.get("training_streams") or {}).get("source_breadth_wholesentence") or {})

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    tok_manifest = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
    compact_tok = token_metrics(COMPACT_10M, tok)
    breadth_tok = token_metrics(BREADTH_10M, tok)

    pairs = build.load_pairs()
    original_row_metas = build.pack_pair_metas(pairs)
    compact_fw_rows = [r for r in iter_jsonl(COMPACT_10M) if r.get("source") == COMPACT_LABEL]
    breadth_fw_rows = [r for r in iter_jsonl(BREADTH_10M) if r.get("source") == BREADTH_LABEL]
    prefix_failures: list[dict[str, Any]] = []
    for i, m in enumerate(original_row_metas[:len(compact_fw_rows)]):
        src = str(m.get("source_text") or "")
        if not str(compact_fw_rows[i].get("text") or "").startswith(src) and len(prefix_failures) < 20:
            prefix_failures.append({"row": i, "arm": "compact"})
        if not str(breadth_fw_rows[i].get("text") or "").startswith(src) and len(prefix_failures) < 20:
            prefix_failures.append({"row": i, "arm": "breadth"})

    breadth_sources = read_jsonl(BREADTH_SOURCES)
    row_meta = read_jsonl(BREADTH_ROW_META)
    pair_hashes = {str(p["norm_hash"]) for p in pairs}
    breadth_hashes = {str(r.get("norm_hash") or aud.norm_hash(str(r.get("text") or ""))) for r in breadth_sources}
    overlap_hashes = sorted(pair_hashes & breadth_hashes)
    whole_sentence_rows = sum(1 for r in row_meta if r.get("whole_sentences") is True)

    records, ng_to_ids, eval_counts = aud.build_eval_index()
    pair_sources = [{"norm_hash": p.get("norm_hash"), "text": p.get("source_text"), "words": p.get("source_words")} for p in pairs]
    compact_rewrites = [{"norm_hash": p.get("norm_hash"), "text": p.get("rewrite_text"), "words": p.get("rewrite_words")} for p in pairs]
    breadth_companions = [{"norm_hash": r.get("norm_hash"), "text": r.get("text"), "words": r.get("words")} for r in breadth_sources]
    comps = [
        aud.component_overlap("common_fineweb_selected_sources", pair_sources, "text", "words", records, ng_to_ids),
        aud.component_overlap("compact_rewrite_companions", compact_rewrites, "text", "words", records, ng_to_ids),
        aud.component_overlap("source_breadth_wholesentence_companions", breadth_companions, "text", "words", records, ng_to_ids),
    ]
    overlap_summary = {
        "strict_eval_record_counts_by_top": eval_counts,
        "strict_eval_records_total": len(records),
        "strict_eval_unique_ngrams": {str(n): len(ng_to_ids[n]) for n in aud.NS},
        "component_kind_summary": comps,
    }

    result = {
        "status": "FW_COMPACT_VS_WHOLESENTENCE_BREADTH_CHECK",
        "arms": {"compact_view": compact, "source_breadth_wholesentence": breadth},
        "matching": {
            "word_seq_match": word_seq_match,
            "source_seq_match_except_fw_label": source_seq_match_except_fw_label,
            "fineweb_rows_compact": len(compact_fw_rows),
            "fineweb_rows_breadth": len(breadth_fw_rows),
            "common_source_prefix_failures_sample": prefix_failures,
            "breadth_companion_hash_overlap_with_pair_sources_count": len(overlap_hashes),
            "breadth_companion_hash_overlap_with_pair_sources_sample": overlap_hashes[:20],
            "breadth_row_meta_rows": len(row_meta),
            "breadth_rows_marked_whole_sentence": whole_sentence_rows,
        },
        "training_streams": {
            "compact_view": compact_stream,
            "source_breadth_wholesentence": breadth_stream,
            "compact_stream_manifest_record": compact_stream_rec,
            "source_breadth_wholesentence_manifest_record": breadth_stream_rec,
            "pass_orders_match_existing_compact": stream_orders_match,
        },
        "tokenizer": {
            "dir": str(TOKENIZER_DIR),
            "manifest": tok_manifest,
            "len": len(tok),
            "vocab_size": tok.vocab_size,
            "is_fast": tok.is_fast,
            "tokenizer_json_sha256": sha256_file(TOKENIZER_DIR / "tokenizer.json"),
        },
        "token_metrics_10m": {
            "compact_view": compact_tok,
            "source_breadth_wholesentence": breadth_tok,
            "delta_breadth_minus_compact": {
                "tokens_per_word_visible": breadth_tok["tokens_per_word_visible"] - compact_tok["tokens_per_word_visible"],
                "rows_over_256": breadth_tok["rows_over_256"] - compact_tok["rows_over_256"],
                "wwm_groups_visible": breadth_tok["wwm_groups_visible"] - compact_tok["wwm_groups_visible"],
            },
        },
        "overlap_with_official_score_text": overlap_summary,
        "repaired_breadth_manifest": repair_manifest,
        "scientific_reading": {
            "what_is_isolated": "The comparison now contrasts source-aligned compact re-expression against coherent whole-sentence independent FineWeb breadth under matched word exposure, row lengths, tokenizer, seed, architecture, and pass order.",
            "remaining_asymmetry": "The breadth companion distribution is independently selected and domain-matched only coarsely; it is not source-aligned to the common FineWeb source rows. That asymmetry is the intended breadth-versus-alignment contrast, but downstream reading should separate broad-data gain from alignment gain.",
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_json = OUT_DIR / "fw_compact_vs_wholesentence_breadth_check.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def overlap_line(name: str) -> str:
        rec = next(x for x in comps if x["component_kind"] == name)
        return f"n7 {rec['components_with_any_match_n7']} comps/{rec['unique_matched_ngrams_n7']} unique, n8 {rec['components_with_any_match_n8']} comps/{rec['unique_matched_ngrams_n8']} unique, n10 {rec['components_with_any_match_n10']} comps/{rec['unique_matched_ngrams_n10']} unique"

    NOTE.write_text(
        "# research — compact vs repaired whole-sentence source-breadth check\n\n"
        f"- Both 10M arms have {compact['rows']:,} rows and {compact['words']:,} words; row word sequence match: {word_seq_match}.\n"
        f"- Source labels match except the FineWeb arm label: {source_seq_match_except_fw_label}.\n"
        f"- FineWeb rows: compact {len(compact_fw_rows):,}, repaired breadth {len(breadth_fw_rows):,}; repaired breadth rows marked whole-sentence: {whole_sentence_rows:,}.\n"
        f"- Shared tokenizer `{TOKENIZER_DIR}` has len={len(tok)}, tokenizer.json SHA `{sha256_file(TOKENIZER_DIR / 'tokenizer.json')}`.\n"
        f"- Compact visible tokens/word: {compact_tok['tokens_per_word_visible']:.4f}; repaired breadth visible tokens/word: {breadth_tok['tokens_per_word_visible']:.4f}.\n"
        f"- Compact rows over 256 tokens: {compact_tok['rows_over_256']:,}; repaired breadth rows over 256 tokens: {breadth_tok['rows_over_256']:,}.\n"
        f"- WWM groups visible per pass: compact {compact_tok['wwm_groups_visible']:,}, repaired breadth {breadth_tok['wwm_groups_visible']:,}.\n"
        f"- Repaired breadth companion source hashes overlapping selected compact-pair source hashes: {len(overlap_hashes)}.\n"
        f"- 100M repaired breadth stream: {breadth_stream['words']:,} words / {breadth_stream['rows']:,} rows; pass orders match existing compact stream: {stream_orders_match}.\n"
        f"- Exact score-text overlap common sources: {overlap_line('common_fineweb_selected_sources')}.\n"
        f"- Exact score-text overlap compact rewrites: {overlap_line('compact_rewrite_companions')}.\n"
        f"- Exact score-text overlap repaired breadth companions: {overlap_line('source_breadth_wholesentence_companions')}.\n\n"
        "Use the repaired whole-sentence breadth stream for the first compact-vs-breadth H100 comparison; do not use the research sliced breadth stream.\n\n"
        f"JSON: `{out_json}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "word_seq_match": word_seq_match,
        "source_seq_match_except_fw_label": source_seq_match_except_fw_label,
        "compact_visible_tpw": round(compact_tok["tokens_per_word_visible"], 4),
        "breadth_visible_tpw": round(breadth_tok["tokens_per_word_visible"], 4),
        "breadth_pair_hash_overlap": len(overlap_hashes),
        "breadth_whole_sentence_rows": whole_sentence_rows,
        "breadth_stream_words": breadth_stream["words"],
        "pass_orders_match_existing_compact": stream_orders_match,
        "overlap_breadth_n7": next(x for x in comps if x["component_kind"] == "source_breadth_wholesentence_companions")["components_with_any_match_n7"],
        "json": str(out_json),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
