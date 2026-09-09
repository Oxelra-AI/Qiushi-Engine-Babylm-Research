#!/usr/bin/env python3
"""Prepare separated near-length and compact prompts for the larger repaired FineWeb medium tier.

The high-precision tier proved that Qwen3.5-9B can produce faithful near views and useful compact views, but the source-aligned changed block was only about 217k words.  This script scales the same legal, evaluation-independent prompt regimes to the broader repaired medium source tier so the density contrast can reach a stronger training signal before spending H100 time on BabyLM pretraining.
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
MEDIUM_IN = ROOT / "data/fineweb_source_by_rewrite_repair/fineweb_factual_medium_repaired_sources.jsonl"
OUT_DIR = ROOT / "data/medium_density_prompts"
NOTE_PATH = (ROOT.parents[2] / 'research/notes/frontier_consolidation/medium_density_prompt_preparation.md')

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


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    return {
        "n": len(xs),
        "min": xs[0],
        "p05": xs[int(0.05 * (len(xs) - 1))],
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p95": xs[int(0.95 * (len(xs) - 1))],
        "max": xs[-1],
        "sum": sum(xs),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ws = [int(r.get("source_words") or wc(str(r.get("source_text", "")))) for r in rows]
    targets = [int(r.get("compression_target_words") or 0) for r in rows if r.get("compression_target_words")]
    ratios = [float(r.get("target_ratio") or 0.0) for r in rows if r.get("target_ratio")]
    docs = {str(r.get("doc_id", "")) for r in rows}
    domains: dict[str, int] = {}
    for r in rows:
        for d in r.get("domain_hits") or ["no_domain"]:
            domains[str(d)] = domains.get(str(d), 0) + 1
    out = {
        "rows": len(rows),
        "source_words_whitespace": sum(ws),
        "unique_docs": len(docs),
        "source_word_stats": stat([float(x) for x in ws]),
        "domain_hit_counts": dict(sorted(domains.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    if targets:
        out.update({
            "target_words_sum": sum(targets),
            "target_word_stats": stat([float(x) for x in targets]),
            "target_ratio_stats": stat(ratios),
        })
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(MEDIUM_IN)
    tier = "medium"
    near = [near_prompt(r, tier) for r in source_rows]
    compact = [compact_prompt(r, tier) for r in source_rows]
    near_path = OUT_DIR / "fineweb_medium_near_prompts_all.jsonl"
    compact_path = OUT_DIR / "fineweb_medium_compact_prompts_all.jsonl"
    write_jsonl(near_path, near)
    write_jsonl(compact_path, compact)
    rng = random.Random(829140)
    samples = {
        "near": rng.sample(near, min(8, len(near))),
        "compact": rng.sample(compact, min(8, len(compact))),
    }
    meta = {
        "status": "MEDIUM_DENSITY_PROMPTS_PREPARED",
        "purpose": "Scale the separated near-length and compact FineWeb view regimes to the larger repaired medium source tier before any BabyLM pretraining screen.",
        "input": str(MEDIUM_IN),
        "input_sha256": sha256_file(MEDIUM_IN),
        "outputs": {"near": str(near_path), "compact": str(compact_path)},
        "sha256": {"near": sha256_file(near_path), "compact": sha256_file(compact_path)},
        "summaries": {"near": summarize(near), "compact": summarize(compact)},
        "compact_prompt_rule": "target=max(8, ceil(0.72*source_words), entity_words+number_count+5), lowered by one word when possible if target reaches source length",
        "no_babylm_evaluation_signal_used": True,
    }
    meta_path = OUT_DIR / "medium_density_prompt_metadata.json"
    sample_path = OUT_DIR / "medium_density_prompt_samples.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sample_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text(
        "# research medium FineWeb density prompt preparation\n\n"
        "Prepared separated near-length and compact Qwen3.5 prompts for the repaired medium FineWeb source tier. "
        "These prompts extend the high-precision density contrast without using BabyLM evaluation data.\n\n"
        f"Source rows: {len(source_rows):,}. Source words: {meta['summaries']['near']['source_words_whitespace']:,}.\n\n"
        f"Metadata: `{meta_path}`\n\nSamples: `{sample_path}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": meta["status"], "metadata": str(meta_path), "outputs": meta["outputs"], "summaries": meta["summaries"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
