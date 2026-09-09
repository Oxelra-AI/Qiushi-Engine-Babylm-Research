#!/usr/bin/env python3
"""Prepare a brevity-pressure prompt variant for the high-anchor FineWeb slice.

The first research generation showed that Qwen3.5-9B can preserve facts on this
source tier, but often returns near-copies.  This script keeps the same source
sentences and adds a dynamic word target so the next run measures the tradeoff
between stronger view diversity and factual preservation.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
from typing import Any

IN_PATH = pathlib.Path("experiments/archive/frontier_consolidation/data/fineweb_high_anchor_slice/fineweb_high_anchor_simplification_prompts_pilot1024.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/high_anchor_compression_prompt_variant")
OUT_PATH = OUT_DIR / "fineweb_high_anchor_compress_prompts_pilot1024.jsonl"
META_PATH = OUT_DIR / "fineweb_high_anchor_compress_prompt_metadata.json"
NOTE_PATH = pathlib.Path("research/notes/frontier_consolidation/high_anchor_compression_prompt_variant.md")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def entity_word_count(entities: list[str]) -> int:
    words = 0
    for e in entities:
        words += len(str(e).split())
    return words


def target_words(row: dict[str, Any]) -> int:
    source_words = int(row.get("source_words") or len(str(row.get("source_text", "")).split()))
    ent_words = entity_word_count(row.get("source_entities") or [])
    nums = len(row.get("source_numbers") or [])
    floor = ent_words + nums + 5
    target = max(8, math.ceil(0.72 * source_words), floor)
    # Keep pressure to shorten, except when named entities make that impossible.
    if target >= source_words:
        target = max(8, source_words - 1)
        if floor >= source_words:
            target = source_words
    return int(target)


def main() -> None:
    rows = read_jsonl(IN_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_rows = []
    ratios = []
    for r in rows:
        nr = dict(r)
        src = str(r["source_text"])
        tgt = target_words(r)
        sw = int(r.get("source_words") or len(src.split()))
        ratios.append(tgt / max(1, sw))
        nr["prompt_id"] = str(r.get("prompt_id", "")).replace("fwhighanchor", "fwcompress")
        nr["source"] = "frontier_consolidation_high_anchor_fineweb_qwen_compression"
        nr["compression_target_words"] = tgt
        nr["system"] = (
            "You rewrite factual English sentences for a small masked language model. Keep the exact same facts and all names, "
            "numbers, dates, quantities, comparisons, negation, and causal relations. Prefer fewer, clearer words. Output only one sentence."
        )
        nr["prompt"] = (
            f"Rewrite the factual sentence in simpler plain English using at most {tgt} words if the facts allow it. "
            "Do not drop any named entity, number, date, quantity, comparison, negation, or causal relation. "
            "Do not add background knowledge or explanations. Do not copy the original wording when a shorter faithful wording is possible. "
            "Output exactly one grammatical English sentence.\n\n"
            f"SOURCE SENTENCE:\n{src}"
        )
        out_rows.append(nr)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "status": "HIGH_ANCHOR_COMPRESSION_PROMPTS_PREPARED",
        "input": str(IN_PATH),
        "output": str(OUT_PATH),
        "rows": len(out_rows),
        "target_ratio_mean": sum(ratios) / max(1, len(ratios)),
        "target_ratio_min": min(ratios) if ratios else None,
        "target_ratio_max": max(ratios) if ratios else None,
        "purpose": "Measure whether the high-anchor FineWeb source tier can support more information-dense faithful second views than the first near-copy simplification prompt.",
        "generation_command": "CUDA_VISIBLE_DEVICES=<gpu> \"${BABYLM_GENERATOR:?configure-an-external-generator}\" --model qwen3.5-9b --prompts-jsonl " + str(OUT_PATH) + " --output-jsonl experiments/archive/frontier_consolidation/training/runs/high_anchor_fineweb_compress_qwen_pilot1024/outputs.jsonl --batch-size 64 --max-new-tokens 80 --temperature 0.1 --device cuda",
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text(
        "# research high-anchor compression prompt variant\n\n"
        "The first high-anchor FineWeb generation preserved facts well but often copied the source. This variant keeps the same 1,024 sources and adds a dynamic word target so the next run measures factual-preservation cost under stronger brevity pressure.\n\n"
        f"Prompts: `{OUT_PATH}`\n\nMetadata: `{META_PATH}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": meta["status"], "rows": len(out_rows), "out": str(OUT_PATH), "metadata": str(META_PATH), "target_ratio_mean": meta["target_ratio_mean"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
