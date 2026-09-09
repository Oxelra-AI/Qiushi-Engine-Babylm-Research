#!/usr/bin/env python3
"""research: corrected source-ID extraction for the restatement-dose arm.

research accidentally compared accepted clean-pair ids (rw2s0_..., rw2s1_...)
against the unsplit extra prompt ids (rw2_...), so it counted almost all extra
prompts as rejected and did not apply the COMPACT_EXPERIENCE source-completeness filter that
created the published rejection count.  This script rebuilds the dose candidate
pool from the exact COMPACT_EXPERIENCE research validation sources:
  - research source_sentences.jsonl with rw_* ids;
  - research extra_source_sentences_shard0.jsonl with rw2s0_* ids;
  - research extra_source_sentences_shard1.jsonl with rw2s1_* ids.

It excludes every source whose pair id is in clean_pairs.jsonl and then applies
COMPACT_EXPERIENCE's is_complete_pair_text(..., is_source=True) to remove source-fragment
cases.  It writes corrected prompts and dose-size calculations for 21% and 25%
total aligned-restatement pair-word fractions.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import random
import statistics
import time
from collections import Counter
from typing import Any

ROOT = pathlib.Path.cwd()
COMPACT_EXPERIENCE_WS = ROOT / "experiments/archive/compact_experience"
COMPACT_EXPERIENCE_SCRIPT = COMPACT_EXPERIENCE_WS / "scripts/clean_materialize_qwen_pairs.py"
COMPACT_EXPERIENCE_DATA = COMPACT_EXPERIENCE_WS / "data/qwen_clean_aligned"
CLEAN_PAIRS = COMPACT_EXPERIENCE_DATA / "clean_pairs.jsonl"
SOURCE_SPECS = [
    ("base", COMPACT_EXPERIENCE_WS / "data/qwen_aligned/source_sentences.jsonl"),
    ("extra_shard0", COMPACT_EXPERIENCE_DATA / "extra_source_sentences_shard0.jsonl"),
    ("extra_shard1", COMPACT_EXPERIENCE_DATA / "extra_source_sentences_shard1.jsonl"),
]
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/dose_arm_originals"
RNG_SEED = 56056
PILOT_N = 512

EXISTING_PAIR_WORDS_PER_10M = 1_656_800
TOTAL_WORDS_PER_PASS = 10_000_000
DOSE_TARGETS = [0.210, 0.250]

PROMPT_TEMPLATE = (
    "Rewrite the following sentence to mean exactly the same thing, using different words and sentence structure.\n\n"
    "Rules:\n"
    "- Keep every named entity, proper name, speaker label, number, date, quantity, and quoted word unchanged.\n"
    "- Do not add, remove, or change any facts.\n"
    "- Use different vocabulary and phrasing; do not copy more than three consecutive words from the original.\n"
    "- Output one complete rewritten sentence only. No explanation.\n\n"
    "Original: {source_sentence}\n\n"
    "Rewritten:"
)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_compact_experience():
    spec = importlib.util.spec_from_file_location("compact_experience_step028", COMPACT_EXPERIENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {COMPACT_EXPERIENCE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def load_sources() -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for cohort, path in SOURCE_SPECS:
        loaded = 0
        for r in read_jsonl(path):
            # source files use text/words/source/example_id/sentence_idx, unlike prompt files.
            rows.append({
                "original_id": str(r["id"]),
                "source_sentence": str(r["text"]),
                "source_words": int(r["words"]),
                "source_name": str(r["source"]),
                "source_example_id": r.get("example_id"),
                "source_sentence_idx": r.get("sentence_idx"),
                "cohort": cohort,
            })
            loaded += 1
        counts[cohort] = loaded
    return rows, counts


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compact_experience = import_compact_experience()

    clean_pairs = read_jsonl(CLEAN_PAIRS)
    clean_ids = {str(p.get("pair_id") or p.get("id")) for p in clean_pairs}
    selected_ids = set()
    selected_path = COMPACT_EXPERIENCE_DATA / "selected_pairs.jsonl"
    if selected_path.exists():
        selected_ids = {str(p.get("pair_id") or p.get("id")) for p in read_jsonl(selected_path)}
    clean_unselected = sorted(clean_ids - selected_ids)

    sources, source_counts = load_sources()
    total_sources = len(sources)
    if len({s["original_id"] for s in sources}) != total_sources:
        raise RuntimeError("Source ids are not unique; corrected extraction cannot proceed")

    rejected = [s for s in sources if s["original_id"] not in clean_ids]
    excluded = Counter()
    salvageable: list[dict[str, Any]] = []
    source_complete_count = 0
    for s in rejected:
        src = s["source_sentence"]
        # This is the exact completeness function used to define source_incomplete_or_fragment.
        if not compact_experience.is_complete_pair_text(src, is_source=True):
            excluded["source_incomplete_or_fragment_recomputed"] += 1
            continue
        source_complete_count += 1
        # Additional tiny surface exclusions that protect generation without changing the scientific count.
        if src.count("xxx") > 2:
            excluded["too_many_placeholder"] += 1
            continue
        words = src.split()
        if not any(c.isalpha() for c in src):
            excluded["no_alpha"] += 1
            continue
        if len(set(w.lower().strip('.,!?;:\"\'()[]{}') for w in words)) / max(1, len(words)) < 0.3:
            excluded["low_diversity"] += 1
            continue
        ss = dict(s)
        ss["source_sha256"] = hashlib.sha256(src.encode("utf-8")).hexdigest()
        salvageable.append(ss)

    rng = random.Random(RNG_SEED)
    shuffled = salvageable[:]
    rng.shuffle(shuffled)

    prompts: list[dict[str, Any]] = []
    for s in shuffled:
        prompts.append({
            "id": f"dose56_{s['original_id']}",
            "original_id": s["original_id"],
            "prompt": PROMPT_TEMPLATE.format(source_sentence=s["source_sentence"]),
            "source_sentence": s["source_sentence"],
            "source_words": s["source_words"],
            "source_name": s["source_name"],
            "source_example_id": s.get("source_example_id"),
            "source_sentence_idx": s.get("source_sentence_idx"),
            "cohort": s["cohort"],
        })

    # Stratified pilot: sample approximately proportionally by source_name for a better yield estimate.
    by_source: dict[str, list[dict[str, Any]]] = {}
    for p in prompts:
        by_source.setdefault(str(p["source_name"]), []).append(p)
    pilot: list[dict[str, Any]] = []
    for name, rows in sorted(by_source.items()):
        take = round(PILOT_N * len(rows) / max(1, len(prompts)))
        take = min(take, len(rows))
        pilot.extend(rows[:take])
    if len(pilot) < min(PILOT_N, len(prompts)):
        used = {p["id"] for p in pilot}
        for p in prompts:
            if p["id"] not in used:
                pilot.append(p)
                used.add(p["id"])
                if len(pilot) >= min(PILOT_N, len(prompts)):
                    break
    pilot = pilot[:min(PILOT_N, len(prompts))]

    salvage_path = OUT_DIR / "salvageable_rejected_originals_corrected.jsonl"
    prompts_path = OUT_DIR / "dose_arm_rewrite_prompts_corrected.jsonl"
    pilot_path = OUT_DIR / "dose_arm_rewrite_prompts_corrected_pilot512.jsonl"
    write_jsonl(salvage_path, salvageable)
    write_jsonl(prompts_path, prompts)
    write_jsonl(pilot_path, pilot)

    reg = Counter(s["source_name"] for s in salvageable)
    cohort = Counter(s["cohort"] for s in salvageable)
    reg_words: dict[str, dict[str, Any]] = {}
    for name in sorted(reg):
        ws = [int(s["source_words"]) for s in salvageable if s["source_name"] == name]
        reg_words[name] = {"count": len(ws), "mean_words": round(statistics.mean(ws), 2), "total_words": sum(ws)}

    # Generation sizing before full hours are spent; this will be updated with pilot yield.
    mean_existing_pair_words = sum(int(p["pair_words"]) for p in clean_pairs) / max(1, len(clean_pairs))
    dose_targets = []
    for frac in DOSE_TARGETS:
        target_words = int(round(frac * TOTAL_WORDS_PER_PASS))
        add_words = max(0, target_words - EXISTING_PAIR_WORDS_PER_10M)
        est_pairs = int(math_ceil(add_words / mean_existing_pair_words)) if add_words else 0
        dose_targets.append({
            "target_total_pair_fraction": frac,
            "target_total_pair_words_per_10M": target_words,
            "additional_pair_words_needed_per_10M": add_words,
            "estimated_pairs_needed_at_existing_mean_pair_words": est_pairs,
        })

    meta = {
        "status": "DOSE_ORIGINALS_CORRECTED",
        "created_utc": now_utc(),
        "elapsed_sec": round(time.time() - started, 2),
        "correction": "uses COMPACT_EXPERIENCE research source ids rw/rw2s0/rw2s1, not unsplit prompt ids; recomputes source completeness with COMPACT_EXPERIENCE is_complete_pair_text",
        "compact_experience_counts": {
            "source_counts_by_cohort": source_counts,
            "total_sources": total_sources,
            "clean_pairs_available": len(clean_ids),
            "selected_pairs": len(selected_ids),
            "clean_but_unselected_pairs": len(clean_unselected),
            "rejected_by_exact_id": len(rejected),
        },
        "source_filter_counts": {
            "rejected_sources_passing_compact_experience_source_completeness": source_complete_count,
            "excluded_after_rejected": dict(excluded),
            "salvageable_rejected_originals": len(salvageable),
        },
        "register_distribution": reg_words,
        "cohort_distribution": dict(cohort),
        "dose_targets_before_pilot": dose_targets,
        "prompt_template": PROMPT_TEMPLATE.replace("{source_sentence}", "<source>"),
        "outputs": {
            "salvageable_rejected_originals": str(salvage_path),
            "rewrite_prompts_all": str(prompts_path),
            "pilot_prompts": str(pilot_path),
        },
        "sha256": {
            "salvageable_rejected_originals": sha256_file(salvage_path),
            "rewrite_prompts_all": sha256_file(prompts_path),
            "pilot_prompts": sha256_file(pilot_path),
        },
    }
    meta_path = OUT_DIR / "extraction_metadata_corrected.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


def math_ceil(x: float) -> int:
    import math
    return int(math.ceil(x))


if __name__ == "__main__":
    main()
