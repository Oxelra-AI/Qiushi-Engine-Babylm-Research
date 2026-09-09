#!/usr/bin/env python3
"""research: row-level lineage and 10M->100M multiplicity audit for the active compliant endpoint.

This CPU-only audit protects the corrected-tokenizer compact_view_reinvest route.
It does not read evaluation outputs and does not alter model/data/tokenizer files.
It recomputes exact hashes, row/word/source counts, verifies that the 100M
training stream contains only ten presentations of the exact 10M pool, and ties
changed-block row metadata to the selected compact FineWeb+Qwen rewrite pairs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(".")
STUDY = ROOT / "experiments/archive" / 'representation_and_objectives'
A02_OVERLAY = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_cleanqwen_overlay_medium_riskhard"
A02_DENSITY = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_core_reinvestment_medium_riskhard"
COMPACT_EXPERIENCE_QWEN = ROOT / "experiments/archive" / 'compact_experience' / "data" / "qwen_clean_aligned"
OUT_DIR = STUDY / "data" / "corpus_lineage_multiplicity_audit"
NOTE_PATH = (ROOT / 'research/notes/representation_and_objectives/corpus_lineage_multiplicity_audit.md')

ACTIVE_10M = A02_OVERLAY / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
ACTIVE_100M = A02_OVERLAY / "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
COMMON_FILLER = A02_OVERLAY / "common_filler_rows.jsonl"
HELDOUT = A02_OVERLAY / "heldout_cleanqwen_rows.jsonl"
CHANGED_META = A02_OVERLAY / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
OVERLAY_META = A02_OVERLAY / "density_cleanqwen_rowholdout_overlay_metadata.json"
DENSITY_META = A02_DENSITY / "density_core_reinvestment_metadata.json"
COMPACT_SELECTED = A02_DENSITY / "selected_compact_reinvest_pairs.jsonl"
COMPACT_ACCEPTED = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "medium_compact_analysis" / "medium_compact_ws_accepted_rewrites.jsonl"
COMPACT_PROMPTS = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "medium_density_prompts" / "fineweb_medium_compact_prompts_all.jsonl"
COMPACT_OUTPUTS = ROOT / "experiments/archive" / 'frontier_consolidation' / "training" / "runs" / "medium_compact_qwen_all" / "outputs.jsonl"
QWEN_CLEAN_META = COMPACT_EXPERIENCE_QWEN / "clean_materialization_metadata.json"
QWEN_CLEAN_AUDIT = COMPACT_EXPERIENCE_QWEN / "pretrain_audit.json"
QWEN_PAIR_META = COMPACT_EXPERIENCE_QWEN / "qwen_pair_packed_rows_meta.jsonl"
QWEN_SELECTED_PAIRS = COMPACT_EXPERIENCE_QWEN / "selected_pairs.jsonl"
QWEN_OUTPUTS = ROOT / "experiments/archive" / 'compact_experience' / "training" / "runs" / "qwen_rewrites_full" / "outputs.jsonl"
QWEN_OUTPUTS_METRICS = ROOT / "experiments/archive" / 'compact_experience' / "training" / "runs" / "qwen_rewrites_full" / "metrics.json"
QWEN_EXTRA0 = ROOT / "experiments/archive" / 'compact_experience' / "training" / "runs" / "extra_qwen2_shard0" / "outputs.jsonl"
QWEN_EXTRA1 = ROOT / "experiments/archive" / 'compact_experience' / "training" / "runs" / "extra_qwen2_shard1" / "outputs.jsonl"
TOKENIZER_MANIFEST = STUDY / "data" / "strictsmall_tokenizer_retrain" / "strictsmall_tokenizer_manifest.json"
TRAIN_COMPLETION = STUDY / "data" / "strictsmalltok_training_completion" / "strictsmalltok_training_completion_summary.json"

EXPECTED = {
    "active_10M_sha256": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
    "active_100M_sha256": "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691",
    "rows_10M": 64740,
    "words_10M": 10_000_000,
    "rows_100M": 647400,
    "words_100M": 100_000_000,
    "changed_rows": 3006,
    "changed_pair_rows": 3005,
    "compact_pairs": 12155,
    "compact_pair_words": 423511,
    "compact_source_words": 261803,
    "compact_rewrite_words": 161708,
    "compact_neutral_topup_words": 9,
    "qwen_pair_rows": 12236,
    "qwen_pair_words": 1656800,
    "common_filler_rows": 61734,
    "common_filler_words": 9576480,
    "passes": 10,
}

OFFICIAL_SOURCES = {"childes", "open_subtitles", "bnc_spoken", "gutenberg", "simple_wiki", "switchboard"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_key(obj: dict[str, Any]) -> tuple[str, int, Any, str]:
    return (obj.get("text", ""), int(obj.get("words", 0)), obj.get("example_id"), obj.get("source", ""))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"JSON parse failed at {path}:{line_no}: {e}") from e


def word_count_text(text: str) -> int:
    return len(text.split())


def classify_source(source: str) -> str:
    if source in OFFICIAL_SOURCES:
        return "official_babylm_source_row"
    if source == "qwen_pair_packed":
        return "inherited_official_source_qwen_paraphrase_pair_row"
    if source == "cleanqwen_fineweb_compact_view_reinvest":
        return "fineweb_source_qwen_compact_rewrite_pair_row"
    if source.startswith("neutral_cleanqwen_topup_compact_reinvest::"):
        return "neutral_topup_from_heldout_official_row"
    return "unknown_source_class"


def summarize_text_corpus(path: Path, keep_counter: bool = True, chunk_pass_size: int | None = None) -> dict[str, Any]:
    h = hashlib.sha256()
    rows = 0
    words = 0
    source_rows = Counter()
    source_words = Counter()
    class_rows = Counter()
    class_words = Counter()
    word_mismatch_examples: list[dict[str, Any]] = []
    dup_counter: Counter | None = Counter() if keep_counter else None
    pass_counters: list[Counter] = []
    pass_source_words: list[Counter] = []
    if chunk_pass_size:
        pass_counters = [Counter() for _ in range(EXPECTED["passes"])]
        pass_source_words = [Counter() for _ in range(EXPECTED["passes"])]

    with path.open("rb") as fb:
        for raw in fb:
            h.update(raw)
            if not raw.strip():
                continue
            obj = json.loads(raw)
            rows += 1
            declared_words = int(obj.get("words", 0))
            actual_words = word_count_text(obj.get("text", ""))
            words += declared_words
            src = obj.get("source", "")
            source_rows[src] += 1
            source_words[src] += declared_words
            cls = classify_source(src)
            class_rows[cls] += 1
            class_words[cls] += declared_words
            if actual_words != declared_words and len(word_mismatch_examples) < 20:
                word_mismatch_examples.append({
                    "row_index_1based": rows,
                    "source": src,
                    "example_id": obj.get("example_id"),
                    "declared_words": declared_words,
                    "actual_split_words": actual_words,
                    "text_prefix": obj.get("text", "")[:160],
                })
            key = row_key(obj)
            if dup_counter is not None:
                dup_counter[key] += 1
            if chunk_pass_size:
                pi = (rows - 1) // chunk_pass_size
                if 0 <= pi < EXPECTED["passes"]:
                    pass_counters[pi][key] += 1
                    pass_source_words[pi][src] += declared_words

    out: dict[str, Any] = {
        "path": str(path),
        "sha256": h.hexdigest(),
        "rows": rows,
        "whitespace_words_declared_sum": words,
        "source_rows": dict(source_rows),
        "source_words": dict(source_words),
        "source_class_rows": dict(class_rows),
        "source_class_words": dict(class_words),
        "word_mismatch_count_sampled": len(word_mismatch_examples),
        "word_mismatch_examples": word_mismatch_examples,
    }
    if dup_counter is not None:
        duplicate_keys = sum(1 for v in dup_counter.values() if v > 1)
        duplicate_rows_extra = sum(v - 1 for v in dup_counter.values() if v > 1)
        out["unique_row_keys"] = len(dup_counter)
        out["duplicate_row_keys"] = duplicate_keys
        out["duplicate_row_excess"] = duplicate_rows_extra
        out["counter"] = dup_counter
    if chunk_pass_size:
        out["pass_counters"] = pass_counters
        out["pass_source_words"] = [dict(c) for c in pass_source_words]
    return out


def count_jsonl_and_sha(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    rows = 0
    with path.open("rb") as f:
        for raw in f:
            h.update(raw)
            if raw.strip():
                rows += 1
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size, "rows": rows, "sha256": h.hexdigest()}


def pair_stats(path: Path, id_field: str = "pair_id") -> dict[str, Any]:
    rows = 0
    ids = set()
    source_words = 0
    rewrite_words = 0
    pair_words = 0
    domain_rows = Counter()
    id_duplicates = 0
    examples = []
    for obj in iter_jsonl(path):
        rows += 1
        pid = obj.get(id_field) or obj.get("prompt_id")
        if pid in ids:
            id_duplicates += 1
        ids.add(pid)
        sw = int(obj.get("source_words") or obj.get("original_words") or 0)
        rw = int(obj.get("rewrite_words") or 0)
        pw = int(obj.get("pair_words") or (sw + rw))
        source_words += sw
        rewrite_words += rw
        pair_words += pw
        hits = obj.get("domain_hits") or []
        if not hits:
            domain_rows["no_domain"] += 1
        else:
            for d in hits:
                domain_rows[d] += 1
        if len(examples) < 5:
            examples.append({k: obj.get(k) for k in [id_field, "prompt_id", "key", "source", "example_id", "sentence_id", "doc_id", "source_words", "rewrite_words", "pair_words", "content_recall", "entity_recall", "number_recall"] if k in obj})
    return {
        "path": str(path),
        "rows": rows,
        "unique_ids": len(ids),
        "id_duplicates": id_duplicates,
        "source_words": source_words,
        "rewrite_words": rewrite_words,
        "pair_words": pair_words,
        "domain_row_hits": dict(domain_rows),
        "examples": examples,
        "ids": ids,
    }


def changed_block_stats() -> dict[str, Any]:
    meta_rows = []
    meta_pair_ids = []
    meta_words = 0
    meta_component_words = Counter()
    for obj in iter_jsonl(CHANGED_META):
        meta_rows.append(obj)
        meta_words += int(obj.get("words", 0))
        for pid in obj.get("pair_ids", []):
            meta_pair_ids.append(pid)
        for k, v in (obj.get("component_sources") or {}).items():
            meta_component_words[k] += int(v)
    ten_rows = []
    ten_words = 0
    ten_sources = Counter()
    ten_classes = Counter()
    for i, obj in enumerate(iter_jsonl(ACTIVE_10M)):
        if i >= len(meta_rows):
            break
        ten_rows.append(obj)
        ten_words += int(obj.get("words", 0))
        ten_sources[obj.get("source", "")] += int(obj.get("words", 0))
        ten_classes[classify_source(obj.get("source", ""))] += int(obj.get("words", 0))
    selected_stats = pair_stats(COMPACT_SELECTED)
    accepted_stats = pair_stats(COMPACT_ACCEPTED, id_field="prompt_id")
    selected_ids = selected_stats.pop("ids")
    accepted_ids = accepted_stats.pop("ids")
    meta_id_counts = Counter(meta_pair_ids)
    missing_meta_in_selected = sorted([pid for pid in meta_id_counts if pid not in selected_ids])[:20]
    selected_missing_from_meta = sorted([pid for pid in selected_ids if pid not in meta_id_counts])[:20]
    selected_not_accepted = sorted([pid.replace("compact:", "") for pid in selected_ids if pid.replace("compact:", "") not in accepted_ids])[:20]
    topup_rows = [r for r in meta_rows if not r.get("pair_ids")]
    # Check changed-block 10M rows align with metadata row lengths and IDs.
    alignment_errors = []
    for i, (m, r) in enumerate(zip(meta_rows, ten_rows)):
        if int(m.get("words", -1)) != int(r.get("words", -2)) or m.get("example_id") != r.get("example_id"):
            alignment_errors.append({
                "row_index": i,
                "meta_example_id": m.get("example_id"),
                "row_example_id": r.get("example_id"),
                "meta_words": m.get("words"),
                "row_words": r.get("words"),
            })
            if len(alignment_errors) >= 20:
                break
    return {
        "changed_meta_rows": len(meta_rows),
        "changed_meta_words": meta_words,
        "changed_meta_pair_refs": len(meta_pair_ids),
        "changed_meta_unique_pair_ids": len(meta_id_counts),
        "changed_meta_pair_duplicate_ref_count": sum(v - 1 for v in meta_id_counts.values() if v > 1),
        "changed_meta_component_words": dict(meta_component_words),
        "changed_first_rows_in_10M_rows": len(ten_rows),
        "changed_first_rows_in_10M_words": ten_words,
        "changed_first_rows_sources_words": dict(ten_sources),
        "changed_first_rows_source_class_words": dict(ten_classes),
        "changed_metadata_alignment_error_count_sampled": len(alignment_errors),
        "changed_metadata_alignment_errors_sample": alignment_errors,
        "compact_selected_pair_stats": selected_stats,
        "compact_accepted_pair_stats": accepted_stats,
        "pair_id_coverage": {
            "missing_meta_ids_in_selected_sample": missing_meta_in_selected,
            "selected_ids_missing_from_meta_sample": selected_missing_from_meta,
            "selected_not_in_accepted_after_prefix_strip_sample": selected_not_accepted,
            "all_meta_ids_in_selected": not missing_meta_in_selected,
            "all_selected_ids_in_meta": not selected_missing_from_meta,
            "all_selected_ids_in_accepted_pool": not selected_not_accepted,
        },
        "neutral_topup_rows": topup_rows,
    }


def qwen_pair_block_stats(active_10m_counter: Counter) -> dict[str, Any]:
    qmeta = []
    qmeta_words = 0
    qmeta_pair_refs = []
    for obj in iter_jsonl(QWEN_PAIR_META):
        qmeta.append(obj)
        qmeta_words += int(obj.get("words", 0))
        qmeta_pair_refs.extend(obj.get("pair_ids", []))
    selected_stats = pair_stats(QWEN_SELECTED_PAIRS)
    selected_ids = selected_stats.pop("ids")
    qmeta_counts = Counter(qmeta_pair_refs)
    missing_in_selected = sorted([pid for pid in qmeta_counts if pid not in selected_ids])[:20]
    selected_missing_from_packed = sorted([pid for pid in selected_ids if pid not in qmeta_counts])[:20]
    active_qwen_rows = 0
    active_qwen_words = 0
    for key, count in active_10m_counter.items():
        text, words, example_id, src = key
        if src == "qwen_pair_packed":
            active_qwen_rows += count
            active_qwen_words += words * count
    clean_meta = read_json(QWEN_CLEAN_META)
    clean_audit = read_json(QWEN_CLEAN_AUDIT)
    return {
        "qwen_pair_meta_rows": len(qmeta),
        "qwen_pair_meta_words": qmeta_words,
        "qwen_pair_meta_refs": len(qmeta_pair_refs),
        "qwen_pair_meta_unique_pair_ids": len(qmeta_counts),
        "qwen_pair_meta_duplicate_refs": sum(v - 1 for v in qmeta_counts.values() if v > 1),
        "qwen_selected_pair_stats": selected_stats,
        "pair_id_coverage": {
            "missing_qwen_meta_ids_in_selected_sample": missing_in_selected,
            "selected_ids_missing_from_packed_meta_sample": selected_missing_from_packed,
            "all_qwen_meta_ids_in_selected": not missing_in_selected,
            "all_selected_ids_in_qwen_packed_meta": not selected_missing_from_packed,
        },
        "active_10M_qwen_pair_packed_rows": active_qwen_rows,
        "active_10M_qwen_pair_packed_words": active_qwen_words,
        "clean_qwen_generation_summary": {
            "generation_output_records": clean_meta.get("generation_output_records"),
            "source_sentences": clean_meta.get("source_sentences"),
            "clean_pairs_available": clean_meta.get("clean_pairs_available"),
            "selected_pairs": clean_meta.get("selected_pairs"),
            "selected_pair_words": clean_meta.get("selected_pair_words"),
            "selected_pair_source_pairs": clean_meta.get("selected_pair_source_pairs"),
            "selected_pair_source_words": clean_meta.get("selected_pair_source_words"),
            "validation_thresholds": clean_meta.get("validation_thresholds"),
            "files": clean_meta.get("files"),
            "sha256": clean_meta.get("sha256"),
        },
        "clean_qwen_pretrain_audit_excerpt": {
            "status": clean_audit.get("status"),
            "selected_pair_words": clean_audit.get("selected_pair_words"),
            "row_length_sequence_matched": clean_audit.get("row_length_sequence_matched"),
            "qwen_treatment_source_words": clean_audit.get("source_word_counts_qwen_treatment"),
            "actual_sha256": clean_audit.get("actual_sha256"),
            "sha256_match_metadata": clean_audit.get("sha256_match_metadata"),
        },
        "qwen_generation_artifacts": {
            "base_outputs": count_jsonl_and_sha(QWEN_OUTPUTS),
            "base_metrics": read_json(QWEN_OUTPUTS_METRICS),
            "extra_shard0_outputs": count_jsonl_and_sha(QWEN_EXTRA0),
            "extra_shard1_outputs": count_jsonl_and_sha(QWEN_EXTRA1),
        },
    }


def compact_generation_stats() -> dict[str, Any]:
    summary = read_json(ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "medium_compact_analysis" / "medium_compact_ws_summary.json")
    output_meta = count_jsonl_and_sha(COMPACT_OUTPUTS)
    prompt_meta = count_jsonl_and_sha(COMPACT_PROMPTS)
    accepted_meta = count_jsonl_and_sha(COMPACT_ACCEPTED)
    model_counts = Counter()
    gen_tokens = 0
    prompt_ids_from_prompts = set()
    prompt_source_assets = Counter()
    prompt_source_tiers = Counter()
    for obj in iter_jsonl(COMPACT_PROMPTS):
        prompt_ids_from_prompts.add(obj.get("prompt_id"))
        prompt_source_assets[obj.get("source_asset", "")] += 1
        prompt_source_tiers[obj.get("source_tier", "")] += 1
    for obj in iter_jsonl(COMPACT_OUTPUTS):
        model_counts[obj.get("model", "")] += 1
        gen_tokens += int(obj.get("generated_tokens") or 0)
    accepted_ids = {obj.get("prompt_id") for obj in iter_jsonl(COMPACT_ACCEPTED)}
    selected_ids = {obj.get("pair_id", "").replace("compact:", "") for obj in iter_jsonl(COMPACT_SELECTED)}
    return {
        "prompts": prompt_meta,
        "outputs": output_meta,
        "accepted_rewrites": accepted_meta,
        "output_model_counts": dict(model_counts),
        "output_generated_tokens_sum": gen_tokens,
        "prompt_id_count": len(prompt_ids_from_prompts),
        "accepted_prompt_id_count": len(accepted_ids),
        "selected_prompt_id_count": len(selected_ids),
        "selected_ids_subset_of_accepted": selected_ids.issubset(accepted_ids),
        "accepted_ids_subset_of_prompts": accepted_ids.issubset(prompt_ids_from_prompts),
        "prompt_source_assets": dict(prompt_source_assets),
        "prompt_source_tiers": dict(prompt_source_tiers),
        "analysis_summary_excerpt": {
            "status": summary.get("status"),
            "label": summary.get("label"),
            "prompts_sha256": summary.get("prompts_sha256"),
            "outputs_sha256": summary.get("outputs_sha256"),
            "word_accounting": summary.get("word_accounting"),
            "overall": summary.get("summary", {}).get("overall"),
            "top_hard_reason_prefixes": summary.get("summary", {}).get("top_hard_reason_prefixes"),
            "top_soft_flags": summary.get("summary", {}).get("top_soft_flags"),
        },
    }


def multiplicity_compare(counter_10m: Counter, pass_counters: list[Counter], counter_100m: Counter) -> dict[str, Any]:
    expected10 = Counter({k: v * EXPECTED["passes"] for k, v in counter_10m.items()})
    missing = []
    excess = []
    mismatched = 0
    for key, exp in expected10.items():
        got = counter_100m.get(key, 0)
        if got != exp:
            mismatched += 1
            if len(missing) < 20:
                missing.append({"key_source": key[3], "example_id": key[2], "words": key[1], "expected": exp, "got": got, "text_hash": text_hash(key[0]), "text_prefix": key[0][:100]})
    for key, got in counter_100m.items():
        if key not in expected10 and len(excess) < 20:
            excess.append({"key_source": key[3], "example_id": key[2], "words": key[1], "got": got, "text_hash": text_hash(key[0]), "text_prefix": key[0][:100]})
    per_pass = []
    pass_all_ok = True
    for i, pc in enumerate(pass_counters):
        ok = pc == counter_10m
        pass_all_ok = pass_all_ok and ok
        diff_count = 0
        samples = []
        if not ok:
            all_keys = set(pc) | set(counter_10m)
            for k in all_keys:
                if pc.get(k, 0) != counter_10m.get(k, 0):
                    diff_count += 1
                    if len(samples) < 5:
                        samples.append({"source": k[3], "example_id": k[2], "words": k[1], "expected": counter_10m.get(k, 0), "got": pc.get(k, 0), "text_hash": text_hash(k[0])})
        per_pass.append({"pass_index_0based": i, "rows": sum(pc.values()), "unique_row_keys": len(pc), "is_exact_permutation_of_10M_pool": ok, "diff_key_count": diff_count, "diff_samples": samples})
    return {
        "overall_multiset_is_10x_10M_pool": counter_100m == expected10,
        "overall_mismatched_10M_keys": mismatched,
        "overall_excess_100M_key_sample_count": len(excess),
        "overall_missing_or_count_mismatch_samples": missing,
        "overall_excess_samples": excess,
        "each_100M_tenth_is_permutation_of_10M_pool": pass_all_ok,
        "per_pass": per_pass,
    }


def write_source_class_csv(class_words: dict[str, int], class_rows: dict[str, int], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_class", "rows_10M", "words_10M", "fraction_words_10M"])
        total_words = sum(class_words.values())
        for cls, words in sorted(class_words.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([cls, class_rows.get(cls, 0), words, words / total_words if total_words else 0.0])


def write_source_csv(source_words: dict[str, int], source_rows: dict[str, int], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "rows_10M", "words_10M", "fraction_words_10M", "source_class"])
        total_words = sum(source_words.values())
        for src, words in sorted(source_words.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([src, source_rows.get(src, 0), words, words / total_words if total_words else 0.0, classify_source(src)])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)

    overlay_meta = read_json(OVERLAY_META)
    density_meta = read_json(DENSITY_META)
    tokenizer_manifest = read_json(TOKENIZER_MANIFEST)
    train_completion = read_json(TRAIN_COMPLETION)

    corpus10 = summarize_text_corpus(ACTIVE_10M, keep_counter=True)
    counter10: Counter = corpus10.pop("counter")
    corpus100 = summarize_text_corpus(ACTIVE_100M, keep_counter=True, chunk_pass_size=EXPECTED["rows_10M"])
    counter100: Counter = corpus100.pop("counter")
    pass_counters: list[Counter] = corpus100.pop("pass_counters")

    common = summarize_text_corpus(COMMON_FILLER, keep_counter=True)
    common_counter: Counter = common.pop("counter")
    heldout = summarize_text_corpus(HELDOUT, keep_counter=True)
    heldout_counter: Counter = heldout.pop("counter")
    del common_counter, heldout_counter  # only counts/hashes are needed in durable JSON.

    mult = multiplicity_compare(counter10, pass_counters, counter100)
    changed = changed_block_stats()
    qwen_stats = qwen_pair_block_stats(counter10)
    compact_stats = compact_generation_stats()

    checks: dict[str, bool] = {
        "active_10M_sha_matches_expected": corpus10["sha256"] == EXPECTED["active_10M_sha256"],
        "active_100M_sha_matches_expected": corpus100["sha256"] == EXPECTED["active_100M_sha256"],
        "active_10M_rows_exact": corpus10["rows"] == EXPECTED["rows_10M"],
        "active_10M_words_exact": corpus10["whitespace_words_declared_sum"] == EXPECTED["words_10M"],
        "active_100M_rows_exact": corpus100["rows"] == EXPECTED["rows_100M"],
        "active_100M_words_exact": corpus100["whitespace_words_declared_sum"] == EXPECTED["words_100M"],
        "active_100M_is_exact_10x_multiset": mult["overall_multiset_is_10x_10M_pool"],
        "each_100M_tenth_is_pool_permutation": mult["each_100M_tenth_is_permutation_of_10M_pool"],
        "common_filler_rows_exact": common["rows"] == EXPECTED["common_filler_rows"],
        "common_filler_words_exact": common["whitespace_words_declared_sum"] == EXPECTED["common_filler_words"],
        "changed_rows_exact": changed["changed_meta_rows"] == EXPECTED["changed_rows"],
        "changed_words_exact": changed["changed_meta_words"] == EXPECTED["compact_pair_words"] + EXPECTED["compact_neutral_topup_words"],
        "changed_pair_refs_exact": changed["changed_meta_pair_refs"] == EXPECTED["compact_pairs"],
        "compact_selected_pairs_exact": changed["compact_selected_pair_stats"]["rows"] == EXPECTED["compact_pairs"],
        "compact_selected_pair_words_exact": changed["compact_selected_pair_stats"]["pair_words"] == EXPECTED["compact_pair_words"],
        "compact_selected_source_words_exact": changed["compact_selected_pair_stats"]["source_words"] == EXPECTED["compact_source_words"],
        "compact_selected_rewrite_words_exact": changed["compact_selected_pair_stats"]["rewrite_words"] == EXPECTED["compact_rewrite_words"],
        "compact_selected_ids_all_in_changed_meta": changed["pair_id_coverage"]["all_selected_ids_in_meta"],
        "changed_meta_ids_all_in_compact_selected": changed["pair_id_coverage"]["all_meta_ids_in_selected"],
        "compact_selected_ids_all_in_accepted_pool": changed["pair_id_coverage"]["all_selected_ids_in_accepted_pool"],
        "qwen_pair_rows_exact": qwen_stats["qwen_pair_meta_rows"] == EXPECTED["qwen_pair_rows"],
        "qwen_pair_words_exact": qwen_stats["qwen_pair_meta_words"] == EXPECTED["qwen_pair_words"],
        "active_10M_qwen_pair_rows_exact": qwen_stats["active_10M_qwen_pair_packed_rows"] == EXPECTED["qwen_pair_rows"],
        "active_10M_qwen_pair_words_exact": qwen_stats["active_10M_qwen_pair_packed_words"] == EXPECTED["qwen_pair_words"],
        "qwen_packed_ids_subset_selected_pairs": qwen_stats["pair_id_coverage"]["all_qwen_meta_ids_in_selected"],
        "compact_outputs_all_qwen35_9b": compact_stats["output_model_counts"] == {"Qwen/Qwen3.5-9B": 21465},
        "compact_selected_subset_of_accepted": compact_stats["selected_ids_subset_of_accepted"],
        "compact_accepted_subset_of_prompts": compact_stats["accepted_ids_subset_of_prompts"],
        "tokenizer_manifest_uses_active_10M": tokenizer_manifest.get("tokenizer_training_source", {}).get("sha256") == corpus10["sha256"],
        "train_commands_use_active_100M": all(cmd.get("train_file_sha256") == corpus100["sha256"] and cmd.get("hash_ok") for cmd in read_json(ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "strictsmall_tokenizer_retrain" / "strictsmall_retrain_preflight.json").get("commands", {}).values()),
        "training_completion_exact_100M_both_seeds": all(
            seed.get("complete_training_artifact") is True
            and seed.get("scientific_metrics_selected", {}).get("word_exposure") == EXPECTED["words_100M"]
            and seed.get("training_log_summary", {}).get("last_train", {}).get("cumulative_word_exposure") == EXPECTED["words_100M"]
            and seed.get("completion_checks", {}).get("chck_100M_exists") is True
            and seed.get("completion_checks", {}).get("final_train_cum_words_100M") is True
            for seed in train_completion.get("seeds", {}).values()
        ),
    }

    validation_errors = [k for k, ok in checks.items() if not ok]
    source_rows10 = corpus10["source_rows"]
    source_words10 = corpus10["source_words"]
    class_rows10 = corpus10["source_class_rows"]
    class_words10 = corpus10["source_class_words"]

    write_source_csv(source_words10, source_rows10, OUT_DIR / "source_word_breakdown_10M.csv")
    write_source_class_csv(class_words10, class_rows10, OUT_DIR / "source_class_word_breakdown_10M.csv")

    # Compact row examples for future human inspection without opening large files.
    examples_path = OUT_DIR / "compact_changed_block_examples.csv"
    selected_by_id: dict[str, dict[str, Any]] = {}
    for obj in iter_jsonl(COMPACT_SELECTED):
        selected_by_id[obj.get("pair_id")] = obj
    with examples_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "example_id", "words", "pair_id", "source_words", "rewrite_words", "domain_hits", "content_recall", "entity_recall", "number_recall", "source_text", "rewrite_text"])
        written = 0
        for meta in iter_jsonl(CHANGED_META):
            for pid in meta.get("pair_ids", []):
                pobj = selected_by_id.get(pid)
                if not pobj:
                    continue
                w.writerow([
                    meta.get("row_index"), meta.get("example_id"), meta.get("words"), pid,
                    pobj.get("source_words"), pobj.get("rewrite_words"), ";".join(pobj.get("domain_hits") or []),
                    pobj.get("content_recall"), pobj.get("entity_recall"), pobj.get("number_recall"),
                    pobj.get("source_text", "")[:300], pobj.get("rewrite_text", "")[:300]
                ])
                written += 1
                if written >= 60:
                    break
            if written >= 60:
                break

    result = {
        "status": "CORPUS_LINEAGE_MULTIPLICITY_AUDIT" if not validation_errors else "CORPUS_LINEAGE_MULTIPLICITY_AUDIT_WITH_VALIDATION_ERRORS",
        "purpose": "Verify active compliant-tokenizer endpoint corpus lineage and 10M-to-100M multiplicity while official evaluations run.",
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "checks": checks,
        "expected": EXPECTED,
        "active_10M_corpus": corpus10,
        "active_100M_stream": corpus100,
        "multiplicity_10M_to_100M": mult,
        "common_filler_rows": common,
        "heldout_cleanqwen_rows": heldout,
        "changed_block_and_compact_pairs": changed,
        "inherited_qwen_pair_packed_block": qwen_stats,
        "compact_fineweb_generation_provenance": compact_stats,
        "metadata_cross_checks": {
            "overlay_status": overlay_meta.get("status"),
            "overlay_inputs": overlay_meta.get("inputs"),
            "overlay_base_split": overlay_meta.get("base_split"),
            "overlay_compact_reinvest_family": overlay_meta.get("families", {}).get("compact_reinvest"),
            "overlay_audit": overlay_meta.get("audit"),
            "overlay_source_word_counts_active_arm": overlay_meta.get("source_word_counts_by_arm", {}).get("cleanqwen_fineweb_compact_view_reinvest"),
            "density_status": density_meta.get("status"),
            "density_inputs": density_meta.get("inputs"),
            "density_selection": density_meta.get("selection"),
            "density_compact_reinvest_summary": density_meta.get("compact_reinvest_summary"),
        },
        "tokenizer_and_training_links": {
            "tokenizer_training_source_path": tokenizer_manifest.get("tokenizer_training_source", {}).get("path"),
            "tokenizer_training_source_sha256": tokenizer_manifest.get("tokenizer_training_source", {}).get("sha256"),
            "tokenizer_json_sha256": tokenizer_manifest.get("tokenizer_hashes", {}).get("tokenizer.json"),
            "model_training_source_path": tokenizer_manifest.get("model_training_source", {}).get("path"),
            "model_training_source_sha256": tokenizer_manifest.get("model_training_source", {}).get("sha256"),
            "strict_small_accounting": tokenizer_manifest.get("strict_small_accounting"),
            "training_completion_status": train_completion.get("status"),
            "training_completion_seeds_excerpt": {k: {kk: v.get(kk) for kk in ["exit_code", "word_exposure_final", "final_checkpoint_exists", "checkpoint_count", "tokenizer_sha256_final", "train_file_sha256"]} for k, v in train_completion.get("seeds", {}).items()},
        },
        "interpretation": [
            "This audit establishes file-level and row-level corpus accounting for the active corrected-tokenizer compact_view_reinvest endpoint, not an endpoint score.",
            "If all checks pass, the 100M training stream is exactly ten presentations of the 10M pool: every one-tenth chunk is a permutation of the 64,740-row pool and the full 100M multiset is 10x the 10M row multiset.",
            "The 10M pool source classes separate official BabyLM rows, inherited official-source Qwen paraphrase pair rows, FineWeb source plus Qwen compact rewrite pair rows, and a 9-word neutral official top-up.",
            "Compact FineWeb provenance is tied to A02 medium FineWeb prompt rows and Qwen/Qwen3.5-9B outputs, selected from accepted rewrites; the audit checks selected ids are contained in both changed-row metadata and accepted rewrite pool.",
            "Inherited qwen_pair_packed provenance is tied to COMPACT_EXPERIENCE research selected pairs and generation artifacts; this part is inherited evidence, not newly regenerated here.",
            "This audit does not prove semantic faithfulness beyond the recorded filters and samples; semantic risk remains a scientific interpretation matter, while official legality depends on exact data budget, allowed sources, teacher approval, and reproducible lineage.",
        ],
        "artifacts": {
            "source_word_breakdown_csv": str(OUT_DIR / "source_word_breakdown_10M.csv"),
            "source_class_word_breakdown_csv": str(OUT_DIR / "source_class_word_breakdown_10M.csv"),
            "compact_examples_csv": str(examples_path),
            "note": str(NOTE_PATH),
        },
    }

    out_json = OUT_DIR / "corpus_lineage_multiplicity_audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    class_rows = result["active_10M_corpus"]["source_class_rows"]
    class_words = result["active_10M_corpus"]["source_class_words"]
    note = []
    note.append("# research corpus lineage and multiplicity audit\n")
    note.append("This CPU-only audit protects the active compliant-tokenizer compact_view_reinvest endpoint while full official evaluations run. It is not a BabyLM score.\n")
    note.append("## Pass/fail summary\n")
    note.append(f"- status: `{result['status']}`\n")
    note.append(f"- validation errors: `{len(validation_errors)}` {validation_errors}\n")
    note.append(f"- active 10M SHA: `{corpus10['sha256']}`; rows `{corpus10['rows']}`; words `{corpus10['whitespace_words_declared_sum']}`\n")
    note.append(f"- active 100M SHA: `{corpus100['sha256']}`; rows `{corpus100['rows']}`; words `{corpus100['whitespace_words_declared_sum']}`\n")
    note.append(f"- 100M multiset is 10x 10M pool: `{mult['overall_multiset_is_10x_10M_pool']}`\n")
    note.append(f"- each 64,740-row tenth is a permutation of the 10M pool: `{mult['each_100M_tenth_is_permutation_of_10M_pool']}`\n")
    note.append("\n## 10M source classes\n")
    for cls, words in sorted(class_words.items(), key=lambda kv: (-kv[1], kv[0])):
        note.append(f"- `{cls}`: rows `{class_rows.get(cls, 0)}`, words `{words}`, fraction `{words/EXPECTED['words_10M']:.6f}`\n")
    note.append("\n## Changed block and Qwen-pair lineage\n")
    note.append(f"- compact FineWeb selected pairs: `{changed['compact_selected_pair_stats']['rows']}` pairs, source words `{changed['compact_selected_pair_stats']['source_words']}`, rewrite words `{changed['compact_selected_pair_stats']['rewrite_words']}`, pair words `{changed['compact_selected_pair_stats']['pair_words']}`.\n")
    note.append(f"- changed metadata rows: `{changed['changed_meta_rows']}`, words `{changed['changed_meta_words']}`, pair refs `{changed['changed_meta_pair_refs']}`, top-up rows `{len(changed['neutral_topup_rows'])}`.\n")
    note.append(f"- compact selected ids all in accepted rewrite pool: `{changed['pair_id_coverage']['all_selected_ids_in_accepted_pool']}`; all selected ids represented in changed metadata: `{changed['pair_id_coverage']['all_selected_ids_in_meta']}`.\n")
    note.append(f"- inherited qwen_pair_packed rows in active 10M: `{qwen_stats['active_10M_qwen_pair_packed_rows']}` rows, `{qwen_stats['active_10M_qwen_pair_packed_words']}` words; packed metadata rows `{qwen_stats['qwen_pair_meta_rows']}`.\n")
    note.append("\n## Generation provenance\n")
    note.append(f"- compact FineWeb Qwen output model counts: `{compact_stats['output_model_counts']}`, generated token sum `{compact_stats['output_generated_tokens_sum']}`, output SHA `{compact_stats['outputs']['sha256']}`.\n")
    note.append(f"- compact prompt source assets: `{compact_stats['prompt_source_assets']}`; tiers `{compact_stats['prompt_source_tiers']}`.\n")
    base_metrics = qwen_stats['qwen_generation_artifacts']['base_metrics']
    note.append(f"- inherited clean-Qwen base generation model `{base_metrics.get('model')}`, prompts `{base_metrics.get('prompts')}`, generated tokens `{base_metrics.get('generated_tokens')}`, output SHA `{qwen_stats['qwen_generation_artifacts']['base_outputs']['sha256']}`.\n")
    note.append("\n## Interpretation for the SOTA route\n")
    note.append("The corrected-tokenizer endpoint has a stronger rule-facing data record after this audit: the tokenizer source and training pool are the same exact 10M file, and the 100M model stream is only ten presentations of that pool. This removes a major source of hidden corpus drift before interpreting the pending official scores. The audit does not create a score and does not justify new training; it prepares interpretation of the completed corrected-tokenizer evaluations.\n")
    note.append("\n## Artifacts\n")
    note.append(f"- JSON: `{out_json}`\n")
    note.append(f"- source CSV: `{OUT_DIR / 'source_word_breakdown_10M.csv'}`\n")
    note.append(f"- source-class CSV: `{OUT_DIR / 'source_class_word_breakdown_10M.csv'}`\n")
    note.append(f"- compact examples CSV: `{examples_path}`\n")
    NOTE_PATH.write_text("".join(note), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "validation_error_count": len(validation_errors),
        "active_10M_rows_words": [corpus10["rows"], corpus10["whitespace_words_declared_sum"]],
        "active_100M_rows_words": [corpus100["rows"], corpus100["whitespace_words_declared_sum"]],
        "active_100M_is_10x_multiset": mult["overall_multiset_is_10x_10M_pool"],
        "each_tenth_is_permutation": mult["each_100M_tenth_is_permutation_of_10M_pool"],
        "source_class_words_10M": class_words,
        "out_json": str(out_json),
        "note": str(NOTE_PATH),
    }, indent=2, ensure_ascii=False))

    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
