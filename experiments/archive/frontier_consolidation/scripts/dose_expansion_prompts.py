#!/usr/bin/env python3
"""research: Prepare compact-view generation prompts from unused FineWeb sentences.

Scientific purpose: expand the compact-view pair pool to enable a dose-response
experiment testing how the fraction of 10M budget restructured into source+compact-view
packets affects learning.

Sources used:
  - web_artifact_removed pool (39,550 sentences, already web-quality filtered)
  - Exclude all 21,465 medium sentences already generated
  - Apply same word-count and quality filters as research
  - Use identical prompt regime and acceptance criteria
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")

# Input: broader pool and already-used medium pool
WAR_PATH = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_sentence_screens/web_artifact_removed_sources.jsonl")
MEDIUM_PATH = ROOT / "data/fineweb_source_by_rewrite_repair/fineweb_factual_medium_repaired_sources.jsonl"

OUT_DIR = ROOT / "data/dose_expansion_prompts"

# Same prompt regime as research
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
    """Same rule as research."""
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
    """Normalize a WAR source row to medium-compatible format."""
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
    }


def compact_prompt(row: dict[str, Any]) -> dict[str, Any]:
    """Build compact prompt using same format as research."""
    sid = int(row.get("sentence_id") or 0)
    src = row["source_text"]
    sw = row["source_words"]
    tgt = compact_target_words(row)
    return {
        "prompt_id": f"frontier_consolidation_fwcompact_expansion_{sid:06d}",
        "typ": "compression",
        "view_regime": "compact_faithful",
        "source": "frontier_consolidation_expansion_fineweb_compact_simplification",
        "sentence_id": sid,
        "doc_id": row["doc_id"],
        "source_text": src,
        "source_words": sw,
        "source_entities": row["source_entities"],
        "source_numbers": row["source_numbers"],
        "domain_hits": row["domain_hits"],
        "relation_count": row["relation_count"],
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
    return {
        "n": len(xs), "min": xs[0], "mean": statistics.fmean(xs),
        "median": statistics.median(xs), "max": xs[-1], "sum": sum(xs),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load medium IDs to exclude
    medium_ids = set()
    for row in read_jsonl(MEDIUM_PATH):
        medium_ids.add(row.get("sentence_id"))
    print(f"Medium IDs to exclude: {len(medium_ids)}", flush=True)

    # Load WAR pool, exclude medium, filter
    war_rows = read_jsonl(WAR_PATH)
    print(f"WAR pool total: {len(war_rows)}", flush=True)

    unused = []
    skip_in_medium = 0
    skip_word_count = 0
    skip_empty = 0
    for r in war_rows:
        sid = r.get("sentence_id")
        if sid in medium_ids:
            skip_in_medium += 1
            continue
        n = normalize_row(r)
        if not n["source_text"] or n["source_words"] < 10:
            skip_empty += 1
            continue
        if n["source_words"] > 48:
            skip_word_count += 1
            continue
        unused.append(n)

    print(f"Unused after filters: {len(unused)} (skip_medium={skip_in_medium}, skip_wc={skip_word_count}, skip_empty={skip_empty})", flush=True)

    # Sort by sentence_id for reproducibility
    unused.sort(key=lambda r: r["sentence_id"])

    # Build compact prompts
    prompts = [compact_prompt(r) for r in unused]

    # Split into two halves for dual-GPU generation
    mid = len(prompts) // 2
    gpu0_prompts = prompts[:mid]
    gpu1_prompts = prompts[mid:]

    # Write all, gpu0, gpu1 prompt files
    all_path = OUT_DIR / "expansion_compact_prompts_all.jsonl"
    gpu0_path = OUT_DIR / "expansion_compact_prompts_gpu0.jsonl"
    gpu1_path = OUT_DIR / "expansion_compact_prompts_gpu1.jsonl"
    write_jsonl(all_path, prompts)
    write_jsonl(gpu0_path, gpu0_prompts)
    write_jsonl(gpu1_path, gpu1_prompts)

    # Word statistics
    sws = [r["source_words"] for r in unused]
    tgts = [compact_target_words(r) for r in unused]

    meta = {
        "status": "DOSE_EXPANSION_PROMPTS_PREPARED",
        "scientific_purpose": "Expand compact-view pool for dose-response experiment. Uses FineWeb sentences from WAR pool not already in medium tier.",
        "war_input": str(WAR_PATH),
        "war_sha256": sha256_file(WAR_PATH),
        "medium_excluded": str(MEDIUM_PATH),
        "medium_ids_excluded": len(medium_ids),
        "war_total": len(war_rows),
        "unused_after_filters": len(unused),
        "skip_in_medium": skip_in_medium,
        "skip_word_count": skip_word_count,
        "skip_empty": skip_empty,
        "prompts_total": len(prompts),
        "prompts_gpu0": len(gpu0_prompts),
        "prompts_gpu1": len(gpu1_prompts),
        "source_word_stats": stat([float(x) for x in sws]),
        "target_word_stats": stat([float(x) for x in tgts]),
        "compact_prompt_rule": "target=max(8, ceil(0.72*source_words), entity_words+number_count+5), same as research",
        "outputs": {
            "all": str(all_path),
            "gpu0": str(gpu0_path),
            "gpu1": str(gpu1_path),
        },
        "generation_commands": {
            "gpu0": f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {gpu0_path} --output-jsonl experiments/archive/frontier_consolidation/training/runs/expansion_compact_qwen_gpu0/outputs.jsonl --batch-size 64 --max-new-tokens 80 --temperature 0.1 --device cuda",
            "gpu1": f"CUDA_VISIBLE_DEVICES=1 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {gpu1_path} --output-jsonl experiments/archive/frontier_consolidation/training/runs/expansion_compact_qwen_gpu1/outputs.jsonl --batch-size 64 --max-new-tokens 80 --temperature 0.1 --device cuda",
        },
        "no_babylm_evaluation_signal_used": True,
    }
    meta_path = OUT_DIR / "expansion_prompt_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": meta["status"],
        "prompts_total": meta["prompts_total"],
        "prompts_gpu0": meta["prompts_gpu0"],
        "prompts_gpu1": meta["prompts_gpu1"],
        "source_word_stats": meta["source_word_stats"],
        "target_word_stats": meta["target_word_stats"],
        "metadata": str(meta_path),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
