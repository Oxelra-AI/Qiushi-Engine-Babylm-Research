#!/usr/bin/env python3
"""research: prepare optional higher-than-MAX compact-view generation prompts.

The fixed-budget dose curve currently has 1x and research MAX=2.64x training
points, plus a prepared no-generation intermediate 1.82x point.  If the MAX
semantic view-minus-repeat leg is still rising, a higher dose is needed to
locate the turnover fraction.  The only remaining sentence inventory does not
come from the already web-artifact-removed tier, so this branch is a boundary
instrument with a stronger source-population warning; it should not be confused
with the cleaner 1x--1.82x--2.64x nested WAR/medium family.

This script performs CPU prompt preparation only.  It uses the identical prompt
word-target rule and system text as research and records the provenance, expected
source shift, and generation commands for a first-free H100 card.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
FULL_SOURCE = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_sentence_sources/fineweb_sentence_sources.jsonl")
MEDIUM_SOURCE = ROOT / "data/fineweb_source_by_rewrite_repair/fineweb_factual_medium_repaired_sources.jsonl"
EXPANSION_PROMPTS = ROOT / "data/dose_expansion_prompts/expansion_compact_prompts_all.jsonl"
OUT_DIR = ROOT / "data/higher_dose_prompts"

BASE_CHANGED_BLOCK_WORDS = 423_520
MAX_PAIR_WORDS = 1_118_587
MAX_CHANGED_BUDGET = 1_118_720
TARGET_HIGH_DOSE = 3.50
COMPACT_SYSTEM = (
    "You rewrite factual English sentences for a small masked language model. Keep the exact same facts and all names, "
    "numbers, dates, quantities, comparisons, negation, and causal relations. Prefer fewer, clearer words. Output only one sentence."
)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def wc(text: str) -> int:
    return len((text or "").split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entity_word_count(entities: list[str]) -> int:
    return sum(len(str(e).split()) for e in entities if str(e).strip())


def compact_target_words(row: dict[str, Any]) -> int:
    src = str(row.get("source_text") or row.get("text") or "")
    source_words = int(row.get("source_words") or row.get("words") or wc(src))
    ent_words = entity_word_count([str(x) for x in (row.get("source_entities") or row.get("entities") or [])])
    nums = len(row.get("source_numbers") or row.get("numbers") or [])
    floor = ent_words + nums + 5
    target = max(8, math.ceil(0.72 * source_words), floor)
    if target >= source_words:
        target = max(8, source_words - 1)
        if floor >= source_words:
            target = source_words
    return int(target)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    src = str(row.get("source_text") or row.get("text") or "").strip()
    return {
        "sentence_id": row.get("sentence_id"),
        "doc_id": str(row.get("doc_id", "")),
        "source_text": src,
        "source_words": int(row.get("source_words") or row.get("words") or wc(src)),
        "source_entities": list(row.get("source_entities") or row.get("entities") or []),
        "source_numbers": list(row.get("source_numbers") or row.get("numbers") or []),
        "domain_hits": list(row.get("domain_hits") or []),
        "relation_count": int(row.get("relation_count") or 0),
        "flags": list(row.get("flags") or []),
        "content_score": float(row.get("content_score") or 0.0),
        "source_row_quality_flags": list(row.get("source_row_quality_flags") or []),
    }


def compact_prompt(row: dict[str, Any]) -> dict[str, Any]:
    sid = int(row.get("sentence_id") or 0)
    src = row["source_text"]
    sw = row["source_words"]
    tgt = compact_target_words(row)
    return {
        "prompt_id": f"frontier_consolidation_fwcompact_highdose_{sid:06d}",
        "typ": "compression",
        "view_regime": "compact_faithful",
        "source": "frontier_consolidation_higher_dose_remaining_fineweb_compact_simplification",
        "sentence_id": sid,
        "doc_id": row["doc_id"],
        "source_text": src,
        "source_words": sw,
        "source_entities": row["source_entities"],
        "source_numbers": row["source_numbers"],
        "domain_hits": row["domain_hits"],
        "relation_count": row["relation_count"],
        "source_flags": row["flags"],
        "source_row_quality_flags": row["source_row_quality_flags"],
        "content_score": row["content_score"],
        "compression_target_words": tgt,
        "target_ratio": tgt / max(1, sw),
        "system": COMPACT_SYSTEM,
        "prompt": (
            f"Rewrite the factual sentence in simpler plain English using at most {tgt} words if the facts allow it. "
            "Do not drop any named entity, number, date, quantity, comparison, negation, or causal relation. "
            "Do not add background knowledge or explanations. Do not copy the original wording when a shorter faithful wording is possible. "
            "Output exactly one grammatical English sentence.\n\nSOURCE SENTENCE:\n" + src
        ),
    }


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    def q(p: float) -> float:
        return xs[int(p * (len(xs) - 1))]
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p95": q(0.95), "max": xs[-1], "sum": sum(xs)}


def sid_set(rows: list[dict[str, Any]]) -> set[Any]:
    return {r.get("sentence_id") for r in rows}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full = read_jsonl(FULL_SOURCE)
    medium = read_jsonl(MEDIUM_SOURCE)
    exp_prompts = read_jsonl(EXPANSION_PROMPTS)
    used = sid_set(medium) | sid_set(exp_prompts)

    normalized: list[dict[str, Any]] = []
    skip_used = skip_empty = skip_words = 0
    for r in full:
        sid = r.get("sentence_id")
        if sid in used:
            skip_used += 1
            continue
        n = normalize_row(r)
        if not n["source_text"]:
            skip_empty += 1
            continue
        if n["source_words"] < 10 or n["source_words"] > 48:
            skip_words += 1
            continue
        normalized.append(n)

    # Sort for reproducibility.  A secondary high-content file lets a later agent
    # generate only the most source-rich subset if one-card time is tight.
    normalized.sort(key=lambda r: (int(r["sentence_id"]), str(r["doc_id"])))
    prompts = [compact_prompt(r) for r in normalized]
    high_content_rows = sorted(normalized, key=lambda r: (-float(r["content_score"]), int(r["sentence_id"])))
    high_content_prompts = [compact_prompt(r) for r in high_content_rows]

    all_path = OUT_DIR / "higher_dose_compact_prompts_all.jsonl"
    firstfree_path = OUT_DIR / "higher_dose_compact_prompts_firstfree_all.jsonl"
    high_content_path = OUT_DIR / "higher_dose_compact_prompts_highcontent_first12000.jsonl"
    gpu0_path = OUT_DIR / "higher_dose_compact_prompts_gpu0.jsonl"
    gpu1_path = OUT_DIR / "higher_dose_compact_prompts_gpu1.jsonl"
    mid = len(prompts) // 2
    write_jsonl(all_path, prompts)
    write_jsonl(firstfree_path, prompts)
    write_jsonl(high_content_path, high_content_prompts[:12_000])
    write_jsonl(gpu0_path, prompts[:mid])
    write_jsonl(gpu1_path, prompts[mid:])

    source_words = [float(r["source_words"]) for r in normalized]
    target_words = [float(compact_target_words(r)) for r in normalized]
    flag_counts: dict[str, int] = {}
    for r in normalized:
        for fl in r["flags"] or ["no_flag"]:
            flag_counts[fl] = flag_counts.get(fl, 0) + 1
    needed_for_3p5 = math.ceil(TARGET_HIGH_DOSE * BASE_CHANGED_BLOCK_WORDS) - MAX_PAIR_WORDS
    meta = {
        "status": "HIGHER_DOSE_PROMPTS_PREPARED",
        "scientific_purpose": "Optional higher-than-2.64x generation branch if MAX semantic view-minus-repeat is still rising; this branch has a stronger source-population warning because all remaining sources fall outside the already web-artifact-removed/medium accepted families.",
        "inputs": {
            "full_source": str(FULL_SOURCE),
            "full_source_sha256": sha256_file(FULL_SOURCE),
            "medium_used": str(MEDIUM_SOURCE),
            "expansion_prompts_used": str(EXPANSION_PROMPTS),
        },
        "counts": {
            "full_rows": len(full),
            "used_sentence_ids": len(used),
            "unused_after_used_filter": len(full) - skip_used,
            "prompts_total": len(prompts),
            "skip_used": skip_used,
            "skip_empty": skip_empty,
            "skip_word_count": skip_words,
        },
        "source_word_stats": stat(source_words),
        "target_word_stats": stat(target_words),
        "flag_counts_top20": sorted(flag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20],
        "capacity_target": {
            "max_pair_words": MAX_PAIR_WORDS,
            "max_changed_budget": MAX_CHANGED_BUDGET,
            "target_high_dose_vs_1x": TARGET_HIGH_DOSE,
            "target_pair_words_for_3p5x": math.ceil(TARGET_HIGH_DOSE * BASE_CHANGED_BLOCK_WORDS),
            "additional_accepted_pair_words_needed_beyond_step256_max_for_3p5x": needed_for_3p5,
            "rough_prompt_source_words_available": int(sum(source_words)),
            "rough_note": "At research acceptance/yield this is plausibly enough for a >3x point, but the actual accepted set must be measured before training.",
        },
        "outputs": {
            "all": str(all_path),
            "firstfree_all": str(firstfree_path),
            "highcontent_first12000": str(high_content_path),
            "gpu0_half": str(gpu0_path),
            "gpu1_half": str(gpu1_path),
        },
        "generation_commands": {
            "first_free_card_all": f"CUDA_VISIBLE_DEVICES=<FREE_GPU> ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {firstfree_path} --output-jsonl experiments/archive/frontier_consolidation/training/runs/higher_dose_compact_qwen_firstfree/outputs.jsonl --batch-size 64 --max-new-tokens 80 --temperature 0.1 --device cuda",
            "first_free_card_highcontent_12000": f"CUDA_VISIBLE_DEVICES=<FREE_GPU> ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {high_content_path} --output-jsonl experiments/archive/frontier_consolidation/training/runs/higher_dose_compact_qwen_highcontent12000/outputs.jsonl --batch-size 64 --max-new-tokens 80 --temperature 0.1 --device cuda",
            "two_card_gpu0": f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {gpu0_path} --output-jsonl experiments/archive/frontier_consolidation/training/runs/higher_dose_compact_qwen_gpu0/outputs.jsonl --batch-size 64 --max-new-tokens 80 --temperature 0.1 --device cuda",
            "two_card_gpu1": f"CUDA_VISIBLE_DEVICES=1 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {gpu1_path} --output-jsonl experiments/archive/frontier_consolidation/training/runs/higher_dose_compact_qwen_gpu1/outputs.jsonl --batch-size 64 --max-new-tokens 80 --temperature 0.1 --device cuda",
        },
        "interpretation_warning": "A higher dose made from this remaining source pool would not by itself be a clean population-controlled extension of the 1x/1.82x/2.64x family. Its view-minus-repeat leg remains internally matched within the higher-dose rows, but comparisons of curve position against lower doses carry stronger source-population uncertainty.",
        "no_generation_training_or_evaluation_started": True,
    }
    meta_path = OUT_DIR / "higher_dose_prompt_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research higher-dose prompt preparation",
        "",
        "This is only a prepared generation branch; no GPU generation, BabyLM training, or evaluation has been started.",
        f"Remaining prompts: {len(prompts):,}; source words {int(sum(source_words)):,}; median source words {statistics.median(source_words) if source_words else None}.",
        f"Additional accepted pair words needed beyond research MAX for a 3.5x point: {needed_for_3p5:,}.",
        "All remaining rows are outside the already used medium/WAR generation inventory, so this branch carries a stronger source-population warning than the nested 1x--1.82x--2.64x family.",
        "",
        f"Metadata: `{meta_path}`",
    ]
    (OUT_DIR / "higher_dose_prompt_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": meta["status"],
        "prompts_total": len(prompts),
        "rough_source_words_available": int(sum(source_words)),
        "additional_pair_words_needed_for_3p5x": needed_for_3p5,
        "metadata": str(meta_path),
        "firstfree_prompts": str(firstfree_path),
        "highcontent_first12000": str(high_content_path),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
