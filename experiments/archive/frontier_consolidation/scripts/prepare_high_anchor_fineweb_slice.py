#!/usr/bin/env python3
"""Prepare a higher-anchor FineWeb simplification slice from the repaired research tier.

This is a narrower source asset for the first Qwen faithfulness run.  It filters
out first/second-person advice, generic starts, title-like rows, low-anchor rows,
and low content score examples that remained in the broad high-precision tier.
The aim is not to build the final corpus here, but to make the first generation
slice more diagnostic for the source-by-rewrite mechanism.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import re
from typing import Any, Iterable

IN_PATH = pathlib.Path("experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_factual_high_precision_sources.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/fineweb_high_anchor_slice")
NOTE = pathlib.Path("research/notes/frontier_consolidation/fineweb_high_anchor_slice.md")

GENERIC_STARTS = {
    "additionally", "moreover", "currently", "fortunately", "sadly", "unfortunately", "for", "if",
    "today", "people", "such", "these", "this", "when", "where", "while", "first", "also",
    "it", "its", "they", "their", "he", "she", "his", "her", "as", "here", "only", "amazingly",
    "create", "discover", "choose", "use", "find", "learn", "click", "read",
}
PRON = re.compile(r"\b(you|your|we|our|us|i|my)\b", re.I)
TITLE = re.compile(r"^(Signs and Symptoms|Causes of|Types of|Benefits of|How to|What is|Why)\b", re.I)
LOW_VALUE = [
    re.compile(r"\b(says|according to)\b", re.I),
    re.compile(r"\bshould be\b", re.I),
]
SOURCE_NOISE = [
    re.compile(r"\.\.\.|…"),
    re.compile(r"[▪✔•]"),
    re.compile(r"\b(score hour|space of curiosity|thinkng|theres|there expansion|ii is)\b", re.I),
    re.compile(r"___"),
]
SYSTEM = (
    "You simplify factual English sentences for a small masked language model. Preserve exactly the same meaning, "
    "including every named entity, number, date, quantity, and relation. Do not add facts. Output only one sentence."
)


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reject_reasons(r: dict[str, Any]) -> list[str]:
    text = str(r["text"])
    first = re.sub(r"[^A-Za-z]", "", text.split()[0]).lower() if text.split() else ""
    entities = r.get("entities") or []
    numbers = r.get("numbers") or []
    domains = set(r.get("domain_hits") or [])
    non_generic_domains = domains - {"causal_relational", "quant_numeric"}
    score = float(r.get("content_score", 0.0))
    reasons: list[str] = []
    if PRON.search(text):
        reasons.append("first_or_second_person")
    if first in GENERIC_STARTS:
        reasons.append("generic_context_start")
    if TITLE.search(text):
        reasons.append("title_like_start")
    if any(rx.search(text) for rx in SOURCE_NOISE):
        reasons.append("residual_web_or_fragment_noise")
    if not (len(entities) >= 1 or len(numbers) >= 1 or non_generic_domains):
        reasons.append("no_named_numeric_or_domain_anchor")
    if score < 2.5:
        reasons.append("low_content_score")
    # Keep some reported-speech factual rows only when they have enough named/numeric anchors.
    if any(rx.search(text) for rx in LOW_VALUE) and score < 3.2:
        reasons.append("low_value_report_or_advice")
    return reasons


def prompt(row: dict[str, Any]) -> dict[str, Any]:
    text = row["text"]
    return {
        "prompt_id": f"frontier_consolidation_fwhighanchor_{int(row.get('sentence_id', 0)):06d}",
        "typ": "simplification",
        "source": "frontier_consolidation_high_anchor_fineweb_qwen_simplification",
        "sentence_id": row.get("sentence_id"),
        "doc_id": str(row.get("doc_id", "")),
        "source_row": row.get("source_row"),
        "sent_index_in_row": row.get("sent_index_in_row"),
        "source_text": text,
        "source_words": int(row.get("words", len(text.split()))),
        "source_entities": row.get("entities") or [],
        "source_numbers": row.get("numbers") or [],
        "domain_hits": row.get("domain_hits") or [],
        "relation_count": row.get("relation_count"),
        "content_score": row.get("content_score"),
        "system": SYSTEM,
        "prompt": (
            "Rewrite the factual sentence below in simpler plain English while preserving the exact same facts. "
            "Keep every named entity, number, date, quantity, causal relation, comparison, and negation. "
            "Do not add background knowledge, explanations, headings, lists, or examples. "
            "Output exactly one grammatical English sentence.\n\nSOURCE SENTENCE:\n" + text
        ),
    }


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    docs = {str(r.get("doc_id", "")) for r in rows}
    domains = collections.Counter(h for r in rows for h in (r.get("domain_hits") or []))
    words = [int(r.get("words", len(str(r.get("text", "")).split()))) for r in rows]
    return {
        "rows": len(rows),
        "words": sum(words),
        "unique_docs": len(docs),
        "mean_words": sum(words) / max(1, len(words)),
        "min_words": min(words) if words else None,
        "max_words": max(words) if words else None,
        "domain_hit_counts": dict(domains.most_common()),
    }


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_doc: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_doc[str(r.get("doc_id", ""))].append(r)
    docs = list(by_doc); rng.shuffle(docs)
    out = []
    for d in docs:
        out.append(max(by_doc[d], key=lambda r: (float(r.get("content_score", 0.0)), -abs(int(r.get("words", 0)) - 22))))
        if len(out) >= n:
            break
    if len(out) < n:
        seen = {r.get("sentence_id") for r in out}
        for r in sorted(rows, key=lambda r: (-float(r.get("content_score", 0.0)), int(r.get("words", 0)))):
            if r.get("sentence_id") in seen:
                continue
            out.append(r)
            if len(out) >= n:
                break
    rng.shuffle(out)
    return out[:n]


def main() -> None:
    rows = load_rows(IN_PATH)
    kept = []
    reject = collections.Counter()
    examples = []
    for r in rows:
        reasons = reject_reasons(r)
        if reasons:
            for x in reasons:
                reject[x] += 1
            if len(examples) < 24:
                examples.append({"sentence_id": r.get("sentence_id"), "words": r.get("words"), "reasons": reasons, "text": r.get("text")})
        else:
            kept.append(r)
    kept = sorted(kept, key=lambda r: (-float(r.get("content_score", 0.0)), str(r.get("doc_id", "")), int(r.get("sentence_id", 0))))
    pilot1024 = stratified(kept, min(1024, len(kept)), 829211)
    pilot2048 = stratified(kept, min(2048, len(kept)), 829212)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = OUT_DIR / "fineweb_high_anchor_sources.jsonl"
    allp = OUT_DIR / "fineweb_high_anchor_simplification_prompts_all.jsonl"
    p1024 = OUT_DIR / f"fineweb_high_anchor_simplification_prompts_pilot{len(pilot1024)}.jsonl"
    p2048 = OUT_DIR / f"fineweb_high_anchor_simplification_prompts_pilot{len(pilot2048)}.jsonl"
    meta = OUT_DIR / "fineweb_high_anchor_slice_metadata.json"
    samples = OUT_DIR / "fineweb_high_anchor_slice_samples.json"
    write_jsonl(src, kept)
    write_jsonl(allp, [prompt(r) for r in kept])
    write_jsonl(p1024, [prompt(r) for r in pilot1024])
    write_jsonl(p2048, [prompt(r) for r in pilot2048])
    payload = {
        "status": "FINEWEB_HIGH_ANCHOR_SLICE_PREPARED",
        "input": str(IN_PATH),
        "input_rows": len(rows),
        "input_words": sum(int(r.get("words", 0)) for r in rows),
        "kept": stats(kept),
        "pilot1024": stats(pilot1024),
        "pilot2048": stats(pilot2048),
        "rejection_reason_counts": dict(reject.most_common()),
        "reject_examples": examples,
        "outputs": {
            "sources": str(src),
            "all_prompts": str(allp),
            "pilot1024_prompts": str(p1024),
            "pilot2048_prompts": str(p2048),
            "metadata": str(meta),
            "samples": str(samples),
            "note": str(NOTE),
        },
        "generation_command_template": (
            "CUDA_VISIBLE_DEVICES=<free_gpu> \"${BABYLM_GENERATOR:?configure-an-external-generator}\" --model qwen3.5-9b "
            f"--prompts-jsonl {p1024} --output-jsonl experiments/archive/frontier_consolidation/training/runs/high_anchor_fineweb_qwen_pilot1024/outputs.jsonl "
            "--batch-size 64 --max-new-tokens 80 --temperature 0.1 --device cuda"
        ),
        "not_launched": True,
    }
    meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rng = random.Random(829213)
    samples.write_text(json.dumps({"random_kept": rng.sample(kept, min(32, len(kept))), "top_kept": kept[:32], "reject_examples": examples}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(
        "# research high-anchor FineWeb generation slice\n\n"
        "This narrows the repaired high-precision tier for the first Qwen faithfulness run. It removes first/second-person advice, generic starts, title-like rows, low-anchor rows, and low content-score rows that remained after the broader repair.\n\n"
        f"Input: {len(rows):,} rows / {sum(int(r.get('words',0)) for r in rows):,} words. Kept: {payload['kept']['rows']:,} rows / {payload['kept']['words']:,} words / {payload['kept']['unique_docs']:,} docs.\n\n"
        f"Recommended first generation prompts: `{p1024}`; larger pilot: `{p2048}`.\n\n"
        "Use only after the pending SimpleWiki semantic-view evidence supports continuing a generated-view route, or if that evidence is weak and this high-anchor slice is needed to test whether broad factual source-by-rewrite is a better mechanism.\n"
        f"\nJSON: `{meta}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "kept": payload["kept"], "pilot1024": payload["pilot1024"], "metadata": str(meta), "note": str(NOTE), "top_rejections": reject.most_common(10)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
