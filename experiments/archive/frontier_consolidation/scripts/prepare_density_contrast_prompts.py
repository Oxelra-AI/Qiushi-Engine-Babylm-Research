#!/usr/bin/env python3
"""Prepare separated near-length and compact FineWeb view prompts for research.

research showed two distinct regimes on the same 1,024 high-anchor sources:
  * a mild simplification prompt: high acceptance but mostly near-copy / near-length;
  * a brevity-pressure prompt: useful compact views with measurable entity/number loss.

These factors must remain separate rather than being collapsed into one best-view set.
This script therefore materializes prompt files that keep the regimes separate.
It does not use BabyLM evaluation items or scores.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
HP_PROMPTS_IN = ROOT / "data/fineweb_source_by_rewrite_repair/fineweb_factual_high_precision_simplification_prompts_all.jsonl"
HA_SOURCES_IN = ROOT / "data/fineweb_high_anchor_slice/fineweb_high_anchor_sources.jsonl"
OUT_DIR = ROOT / "data/density_contrast_prompts"
NOTE_PATH = (ROOT.parents[2] / 'research/notes/frontier_consolidation/density_contrast_prompt_preparation.md')

NEAR_SYSTEM = (
    "You simplify factual English sentences for a small masked language model. Preserve exactly the same meaning, "
    "including every named entity, number, date, quantity, and relation. Do not add facts. Output only one sentence."
)

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


def entity_word_count(entities: list[str]) -> int:
    return sum(len(str(e).split()) for e in entities if str(e).strip())


def compact_target_words(row: dict[str, Any]) -> int:
    src = str(row.get("source_text") or row.get("text") or "")
    source_words = int(row.get("source_words") or row.get("words") or wc(src))
    ent_words = entity_word_count([str(x) for x in (row.get("source_entities") or row.get("entities") or [])])
    nums = len(row.get("source_numbers") or row.get("numbers") or [])
    # This is the research successful target form.  It creates real brevity pressure
    # while leaving a floor for names/numbers/relations, which were the main failure modes.
    floor = ent_words + nums + 5
    target = max(8, math.ceil(0.72 * source_words), floor)
    if target >= source_words:
        target = max(8, source_words - 1)
        if floor >= source_words:
            target = source_words
    return int(target)


def normalize_prompt_base(row: dict[str, Any], source_tier: str) -> dict[str, Any]:
    src = str(row.get("source_text") or row.get("text") or "").strip()
    if not src:
        raise ValueError("source row has no text")
    return {
        "sentence_id": row.get("sentence_id"),
        "doc_id": str(row.get("doc_id", "")),
        "source_row": row.get("source_row"),
        "sent_index_in_row": row.get("sent_index_in_row"),
        "source_text": src,
        "source_words": int(row.get("source_words") or row.get("words") or wc(src)),
        "source_entities": list(row.get("source_entities") or row.get("entities") or []),
        "source_numbers": list(row.get("source_numbers") or row.get("numbers") or []),
        "domain_hits": list(row.get("domain_hits") or []),
        "relation_count": int(row.get("relation_count") or 0),
        "source_asset": row.get("source_asset") or row.get("source") or source_tier,
        "source_tier": source_tier,
    }


def near_prompt(row: dict[str, Any], source_tier: str) -> dict[str, Any]:
    b = normalize_prompt_base(row, source_tier)
    sid = int(b.get("sentence_id") or 0)
    nr = dict(b)
    nr.update({
        "prompt_id": f"frontier_consolidation_fwnear_{source_tier}_{sid:06d}",
        "typ": "simplification",
        "view_regime": "near_length_faithful",
        "source": f"frontier_consolidation_{source_tier}_fineweb_near_length_simplification",
        "system": NEAR_SYSTEM,
        "prompt": (
            "Rewrite the factual sentence below in simpler plain English while preserving the exact same facts. "
            "Keep every named entity, number, date, quantity, and relation. Do not add facts, explanations, headings, lists, or examples. "
            "Output exactly one grammatical English sentence.\n\nSOURCE SENTENCE:\n" + b["source_text"]
        ),
    })
    return nr


def compact_prompt(row: dict[str, Any], source_tier: str) -> dict[str, Any]:
    b = normalize_prompt_base(row, source_tier)
    sid = int(b.get("sentence_id") or 0)
    tgt = compact_target_words(b)
    nr = dict(b)
    nr.update({
        "prompt_id": f"frontier_consolidation_fwcompact_{source_tier}_{sid:06d}",
        "typ": "compression",
        "view_regime": "compact_faithful",
        "source": f"frontier_consolidation_{source_tier}_fineweb_compact_simplification",
        "compression_target_words": tgt,
        "target_ratio": tgt / max(1, int(b["source_words"])),
        "system": COMPACT_SYSTEM,
        "prompt": (
            f"Rewrite the factual sentence in simpler plain English using at most {tgt} words if the facts allow it. "
            "Do not drop any named entity, number, date, quantity, comparison, negation, or causal relation. "
            "Do not add background knowledge or explanations. Do not copy the original wording when a shorter faithful wording is possible. "
            "Output exactly one grammatical English sentence.\n\nSOURCE SENTENCE:\n" + b["source_text"]
        ),
    })
    return nr


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ws = [int(r.get("source_words") or wc(str(r.get("source_text", "")))) for r in rows]
    targets = [int(r.get("compression_target_words") or 0) for r in rows if r.get("compression_target_words")]
    ratios = [float(r.get("target_ratio") or 0.0) for r in rows if r.get("target_ratio")]
    docs = {str(r.get("doc_id", "")) for r in rows}
    out = {
        "rows": len(rows),
        "source_words_whitespace": sum(ws),
        "unique_docs": len(docs),
        "mean_source_words": statistics.fmean(ws) if ws else None,
        "median_source_words": statistics.median(ws) if ws else None,
    }
    if targets:
        out.update({
            "target_words_sum": sum(targets),
            "mean_target_ratio": statistics.fmean(ratios),
            "median_target_ratio": statistics.median(ratios),
        })
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hp_in = read_jsonl(HP_PROMPTS_IN)
    ha_in = read_jsonl(HA_SOURCES_IN)

    assets = {
        "high_precision": hp_in,
        "high_anchor": ha_in,
    }
    files: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    for tier, rows in assets.items():
        near = [near_prompt(r, tier) for r in rows]
        compact = [compact_prompt(r, tier) for r in rows]
        near_path = OUT_DIR / f"fineweb_{tier}_near_prompts_all.jsonl"
        compact_path = OUT_DIR / f"fineweb_{tier}_compact_prompts_all.jsonl"
        write_jsonl(near_path, near)
        write_jsonl(compact_path, compact)
        files[f"{tier}_near"] = str(near_path)
        files[f"{tier}_compact"] = str(compact_path)
        summaries[f"{tier}_near"] = summarize(near)
        summaries[f"{tier}_compact"] = summarize(compact)
        rng = random.Random(829130 + len(tier))
        samples[f"{tier}_near"] = rng.sample(near, min(5, len(near)))
        samples[f"{tier}_compact"] = rng.sample(compact, min(5, len(compact)))

    meta = {
        "status": "DENSITY_CONTRAST_PROMPTS_PREPARED",
        "purpose": "Keep near-length and compact FineWeb second-view regimes separate so saved rewrite words can be tested as a source-reinvestment mechanism.",
        "inputs": {
            "high_precision_prompts": str(HP_PROMPTS_IN),
            "high_anchor_sources": str(HA_SOURCES_IN),
        },
        "outputs": files,
        "summaries": summaries,
        "compact_prompt_rule": "target=max(8, ceil(0.72*source_words), entity_words+number_count+5), lowered by one word when possible if target reaches source length",
        "no_babylm_evaluation_signal_used": True,
    }
    meta_path = OUT_DIR / "density_contrast_prompt_metadata.json"
    samples_path = OUT_DIR / "density_contrast_prompt_samples.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text(
        "# research density-contrast prompt preparation\n\n"
        "The near-length and compact view regimes are now separate prompt files. This protects the information-density hypothesis: a compact arm can spend the saved rewrite words on additional distinct sources instead of being merged with near-copy rewrites.\n\n"
        f"Metadata: `{meta_path}`\n\nSamples: `{samples_path}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": meta["status"], "metadata": str(meta_path), "outputs": files, "summaries": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
