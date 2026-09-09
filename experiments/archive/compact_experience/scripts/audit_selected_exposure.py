#!/usr/bin/env python3
"""Audit selected-original exposure across clean-Qwen controls.

The key question is whether the clean-Qwen gain could be explained by simply
seeing selected high-quality original rows more often.  This script counts, for
several materialized pools, how many words are explicit selected-pair material and
how many words from official rows whose example_id contains a selected source
sentence remain in filler.  For the main clean treatment, separated control, and
selected-original-dup-all control, the sum should equal the selected official row
budget (unique selected example_ids * 160) if the control substitutes pair material
for selected official-row material rather than adding exposure.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib
from collections import Counter

ROOT = _public_path('experiments/archive/compact_experience')
SEL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
CLEAN_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
CLEAN_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
SHUF_META = _public_path('experiments/archive/compact_experience/data/qwen_shuffled_control/shuffled_control_metadata.json')
SHUF_POOL = _public_path('experiments/archive/compact_experience/data/qwen_shuffled_control/training_corpora/qwen_shuffled_10M.jsonl')
SEP_META = _public_path('experiments/archive/compact_experience/data/qwen_separated_pair_control/qwen_separated_pair_metadata.json')
DUPALL_META = _public_path('experiments/archive/compact_experience/data/selected_original_dup_all_control/selected_original_dup_all_metadata.json')
ORIGDUP_META = _public_path('experiments/archive/compact_experience/data/original_dup_control/original_dup_control_metadata.json')
ORIGDUP_POOL = _public_path('experiments/archive/compact_experience/data/original_dup_control/training_corpora/official_original_dup_10M.jsonl')
OUT = _public_path('experiments/archive/compact_experience/data/selected_exposure_audit.json')


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_info():
    selected_ids = set()
    pairs = []
    with SEL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            selected_ids.add(int(o["example_id"]))
            pairs.append(o)
    by_source = Counter()
    orig_by_source = Counter()
    rewrite_by_source = Counter()
    for p in pairs:
        src = p.get("source", "unknown")
        by_source[src] += int(p["pair_words"])
        orig_by_source[src] += int(p["original_words"])
        rewrite_by_source[src] += int(p["rewrite_words"])
    return pairs, selected_ids, by_source, orig_by_source, rewrite_by_source


def filler_selected_words(pool: pathlib.Path, pair_source_prefixes: tuple[str, ...], selected_ids: set[int]) -> dict:
    words = 0
    rows = 0
    total = 0
    source_words = Counter()
    with pool.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            w = int(o.get("words", len(str(o["text"]).split())))
            total += w
            src = str(o.get("source", ""))
            if any(src.startswith(p) for p in pair_source_prefixes):
                continue
            if int(o.get("example_id", -1)) in selected_ids:
                words += w
                rows += 1
                source_words[src] += w
    return {"filler_selected_id_words": words, "filler_selected_id_rows_or_chunks": rows, "filler_selected_source_words": dict(source_words), "pool_total_words": total}


def main():
    pairs, selected_ids, pair_by_source, orig_by_source, rewrite_by_source = selected_info()
    clean = load_json(CLEAN_META)
    audit = {
        "status": "SELECTED_EXPOSURE_AUDIT",
        "selected_pairs": len(pairs),
        "selected_unique_example_ids": len(selected_ids),
        "selected_official_row_word_budget_unique_ids_times_160": len(selected_ids) * 160,
        "selected_pair_words": sum(int(p["pair_words"]) for p in pairs),
        "selected_original_words": sum(int(p["original_words"]) for p in pairs),
        "selected_rewrite_words": sum(int(p["rewrite_words"]) for p in pairs),
        "selected_pair_words_by_source": dict(pair_by_source),
        "selected_original_words_by_source": dict(orig_by_source),
        "selected_rewrite_words_by_source": dict(rewrite_by_source),
        "arms": {},
    }
    clean_fill = filler_selected_words(CLEAN_POOL, ("qwen_pair_packed",), selected_ids)
    audit["arms"]["qwen_clean_aligned"] = {
        "explicit_pair_words": clean.get("selected_pair_words"),
        "explicit_original_words": audit["selected_original_words"],
        "explicit_rewrite_words": audit["selected_rewrite_words"],
        **clean_fill,
        "explicit_pair_plus_selected_filler_words": int(clean.get("selected_pair_words", 0)) + clean_fill["filler_selected_id_words"],
    }
    if SHUF_META.exists() and SHUF_POOL.exists():
        shuf = load_json(SHUF_META)
        fill = filler_selected_words(SHUF_POOL, ("qwen_shuffled_pair_packed",), selected_ids)
        audit["arms"]["qwen_shuffled_control"] = {"explicit_pair_words": shuf.get("pair_words"), **fill, "explicit_pair_plus_selected_filler_words": int(shuf.get("pair_words", 0)) + fill["filler_selected_id_words"]}
    if SEP_META.exists():
        sep = load_json(SEP_META)
        audit["arms"]["qwen_separated_pair_control"] = {
            "explicit_pair_words": sep.get("pair_words"),
            "selected_id_words_in_filler_after_nonselected_first_policy": sep.get("selected_id_words_in_filler_after_nonselected_first_policy"),
            "explicit_pair_plus_selected_filler_words": int(sep.get("pair_words", 0)) + int(sep.get("selected_id_words_in_filler_after_nonselected_first_policy", 0)),
        }
    if DUPALL_META.exists():
        dup = load_json(DUPALL_META)
        audit["arms"]["selected_original_dup_all_control"] = {
            "explicit_duplicate_pair_words": dup.get("duplicate_pair_words"),
            "selected_id_words_in_filler_after_nonselected_first_policy": dup.get("selected_id_words_in_filler_after_nonselected_first_policy"),
            "explicit_pair_plus_selected_filler_words": int(dup.get("duplicate_pair_words", 0)) + int(dup.get("selected_id_words_in_filler_after_nonselected_first_policy", 0)),
        }
    if ORIGDUP_META.exists() and ORIGDUP_POOL.exists():
        od = load_json(ORIGDUP_META)
        fill = filler_selected_words(ORIGDUP_POOL, ("official_original_dup_packed",), selected_ids)
        audit["arms"]["old_official_original_dup"] = {"explicit_duplicate_pair_words": od.get("duplicate_pair_words"), "selected_duplicate_pairs": od.get("selected_duplicate_pairs"), **fill, "explicit_pair_plus_selected_filler_words": int(od.get("duplicate_pair_words", 0)) + fill["filler_selected_id_words"]}
    # Official full-pool controls see each selected official row once per 10M pool.
    audit["arms"]["official_full_pool_reference"] = {"selected_official_row_words_seen_per_10M_pool": len(selected_ids) * 160}
    budget = audit["selected_official_row_word_budget_unique_ids_times_160"]
    for rec in audit["arms"].values():
        x = rec.get("explicit_pair_plus_selected_filler_words") or rec.get("selected_official_row_words_seen_per_10M_pool")
        if x is not None:
            rec["delta_vs_selected_row_word_budget"] = int(x) - budget
    OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "selected_row_budget": budget, "arm_totals": {k: v.get("explicit_pair_plus_selected_filler_words", v.get("selected_official_row_words_seen_per_10M_pool")) for k, v in audit["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
