#!/usr/bin/env python3
"""Prepare a compact FineWeb source-by-rewrite faithfulness slice.

This is the next data-route probe if the SimpleWiki semantic-view contrast is weak.
It samples complete cached FineWeb sentence sources from measured research tiers and
writes Qwen simplification prompts plus metadata.  It does not run generation.

Scientific target:
  Test whether sentence-level FineWeb sources can receive faithful simplifications
  at adequate acceptance rate before spending H100 time on a full source+rewrite
  corpus.  The leader-like structure of interest is original factual source
  sentence + faithful simplified rewrite, contrasted later with packet-local
  same-source repetition.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
SRC_DIR = ROOT / "training" / "data" / "fineweb_sentence_screens"
OUT_DIR = ROOT / "training" / "data" / "fineweb_rewrite_faithfulness_slice"
NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/representation_and_objectives/fineweb_rewrite_faithfulness_slice.md')

BALANCED = SRC_DIR / "balanced_source_by_rewrite_sources.jsonl"
STRICT = SRC_DIR / "strict_factual_expository_sources.jsonl"
WEB_REMOVED = SRC_DIR / "web_artifact_removed_sources.jsonl"

PROMPT_TEMPLATE = (
    "Simplify the following FineWeb sentence so that a young reader can understand it. "
    "Use only information that is present in the sentence. Keep all names, dates, numbers, places, and causal facts. "
    "Do not add background facts, examples, or explanations. Output only one complete simplified sentence.\n\n"
    "Sentence: {text}"
)

TIERS = {
    "strict_factual_expository": STRICT,
    "balanced_source_by_rewrite": BALANCED,
    "web_artifact_removed_extra": WEB_REMOVED,
}
DEFAULT_PLAN = {
    # Strong factual rows: cleaner but narrow.  These estimate upper-bound acceptance.
    "strict_factual_expository": 160,
    # Main future substrate: enough mass but noisier.  This estimates deployable acceptance.
    "balanced_source_by_rewrite": 320,
    # Boundary rows removed from balanced but present after artifact removal.  These show how
    # sensitive generation faithfulness is to looser source screening without using broad noisy rows.
    "web_artifact_removed_extra": 64,
}
DOMAIN_KEYS = [
    "science_physical", "geography_places", "people_history", "institutions_society",
    "causal_relational", "quant_numeric",
]


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def word_count(text: str) -> int:
    return len(text.split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(xs: Iterable[float]) -> dict[str, Any]:
    vals = [float(x) for x in xs]
    if not vals:
        return {"n": 0}
    vals.sort()
    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = p * (len(vals) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)
    return {
        "n": len(vals),
        "min": vals[0],
        "p05": q(0.05),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p95": q(0.95),
        "max": vals[-1],
        "sum": sum(vals),
    }


def source_quality_bucket(row: dict[str, Any]) -> str:
    flags = set(row.get("flags") or [])
    domains = set(row.get("domain_hits") or [])
    rel = int(row.get("relation_count") or 0)
    ents = row.get("entities") or []
    nums = row.get("numbers") or []
    if not flags and (rel >= 2 or (domains & {"science_physical", "causal_relational", "quant_numeric"}) or nums):
        return "high_relation_or_numeric_clean"
    if not flags and ents:
        return "entity_clean"
    if "low_relation" in flags:
        return "low_relation"
    if flags:
        return "screen_edge"
    return "plain_clean"


def sort_key(row: dict[str, Any], rng: random.Random) -> tuple:
    # Deterministic per run, but keeps documents and domains mixed.
    return (
        str(row.get("doc_id")),
        -float(row.get("content_score") or 0.0),
        int(row.get("sentence_id") or 0),
        rng.random(),
    )


def choose_rows(rows: list[dict[str, Any]], n: int, rng: random.Random, exclude_ids: set[int]) -> list[dict[str, Any]]:
    pool = [r for r in rows if int(r.get("sentence_id")) not in exclude_ids and 8 <= word_count(str(r.get("text", ""))) <= 55]
    # Keep sources with explicit quality flags out of the strict/balanced core except for edge bucket.
    by_bucket: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in pool:
        by_bucket[source_quality_bucket(r)].append(r)
    for bucket in by_bucket:
        by_bucket[bucket].sort(key=lambda r: sort_key(r, rng))

    quotas = {
        "high_relation_or_numeric_clean": 0.42,
        "entity_clean": 0.22,
        "plain_clean": 0.16,
        "low_relation": 0.14,
        "screen_edge": 0.06,
    }
    chosen: list[dict[str, Any]] = []
    used_docs: set[str] = set()
    # First pass: favor document diversity while satisfying approximate bucket proportions.
    for bucket, frac in quotas.items():
        target = max(0, int(round(n * frac)))
        for r in by_bucket.get(bucket, []):
            if len([x for x in chosen if source_quality_bucket(x) == bucket]) >= target:
                break
            doc = str(r.get("doc_id"))
            if doc in used_docs and len(used_docs) < n:
                continue
            chosen.append(r)
            used_docs.add(doc)
            exclude_ids.add(int(r.get("sentence_id")))
    # Fill remaining from all buckets, still minimizing duplicate documents when possible.
    all_pool = sorted(pool, key=lambda r: (str(r.get("doc_id")) in used_docs, -float(r.get("content_score") or 0.0), rng.random()))
    chosen_ids = {int(r.get("sentence_id")) for r in chosen}
    for r in all_pool:
        if len(chosen) >= n:
            break
        sid = int(r.get("sentence_id"))
        if sid in chosen_ids or sid in exclude_ids:
            continue
        chosen.append(r)
        chosen_ids.add(sid)
        exclude_ids.add(sid)
        used_docs.add(str(r.get("doc_id")))
    return chosen[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=82911)
    ap.add_argument("--strict", type=int, default=DEFAULT_PLAN["strict_factual_expository"])
    ap.add_argument("--balanced", type=int, default=DEFAULT_PLAN["balanced_source_by_rewrite"])
    ap.add_argument("--web-extra", type=int, default=DEFAULT_PLAN["web_artifact_removed_extra"])
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    loaded = {tier: read_jsonl(path) for tier, path in TIERS.items()}
    plan = {
        "strict_factual_expository": args.strict,
        "balanced_source_by_rewrite": args.balanced,
        "web_artifact_removed_extra": args.web_extra,
    }
    # Extra tier should be rows that web_artifact_removed kept but balanced did not.
    balanced_ids = {int(r.get("sentence_id")) for r in loaded["balanced_source_by_rewrite"]}
    loaded["web_artifact_removed_extra"] = [r for r in loaded["web_artifact_removed_extra"] if int(r.get("sentence_id")) not in balanced_ids]

    chosen_records: list[dict[str, Any]] = []
    used: set[int] = set()
    for tier in ["strict_factual_expository", "balanced_source_by_rewrite", "web_artifact_removed_extra"]:
        selected = choose_rows(loaded[tier], plan[tier], rng, used)
        for rank, row in enumerate(selected):
            text = str(row.get("text", "")).strip()
            rec = dict(row)
            rec.update({
                "selection_tier": tier,
                "selection_rank_in_tier": rank,
                "prompt_id": f"fwfaith_simp_{tier}_{int(row.get('sentence_id')):06d}",
                "typ": "simplification",
                "prompt": PROMPT_TEMPLATE.format(text=text),
                "source_text": text,
                "source_words": word_count(text),
                "source_key": f"fineweb_sent_{int(row.get('sentence_id'))}",
            })
            chosen_records.append(rec)

    # Stable mixed order for generation, preserving an index that analyzer can map.
    rng.shuffle(chosen_records)
    for i, rec in enumerate(chosen_records):
        rec["generation_index"] = i

    prompts = out_dir / "fineweb_rewrite_faithfulness_prompts.jsonl"
    sources = out_dir / "fineweb_rewrite_faithfulness_sources.jsonl"
    samples = out_dir / "fineweb_rewrite_faithfulness_samples.json"
    metadata = out_dir / "fineweb_rewrite_faithfulness_metadata.json"
    generation_config = out_dir / "GENERATION_CONFIG.json"

    with prompts.open("w", encoding="utf-8") as f:
        for rec in chosen_records:
            f.write(json.dumps({
                "id": rec["prompt_id"],
                "prompt": rec["prompt"],
                "type": rec["typ"],
                "source_text": rec["source_text"],
                "source_words": rec["source_words"],
                "source_key": rec["source_key"],
                "selection_tier": rec["selection_tier"],
                "sentence_id": rec.get("sentence_id"),
                "doc_id": rec.get("doc_id"),
                "entities": rec.get("entities") or [],
                "numbers": rec.get("numbers") or [],
                "domain_hits": rec.get("domain_hits") or [],
                "flags": rec.get("flags") or [],
                "content_score": rec.get("content_score"),
                "slice_index": rec["generation_index"],
            }, ensure_ascii=False) + "\n")
    with sources.open("w", encoding="utf-8") as f:
        for rec in sorted(chosen_records, key=lambda r: (r["selection_tier"], int(r.get("sentence_id") or 0))):
            f.write(json.dumps({k: rec.get(k) for k in [
                "prompt_id", "generation_index", "selection_tier", "sentence_id", "doc_id", "source_row", "sent_index_in_row",
                "source_text", "source_words", "entities", "numbers", "domain_hits", "relation_count", "flags", "content_score",
            ]}, ensure_ascii=False) + "\n")

    by_tier = collections.Counter(r["selection_tier"] for r in chosen_records)
    by_bucket = collections.Counter(source_quality_bucket(r) for r in chosen_records)
    by_domain = collections.Counter(d for r in chosen_records for d in (r.get("domain_hits") or []))
    meta = {
        "status": "FINEWEB_REWRITE_FAITHFULNESS_SLICE_PREPARED",
        "not_launched": True,
        "scientific_purpose": "Small Qwen simplification faithfulness slice for a future source+rewrite versus same-source repetition FineWeb contrast, contingent on the active semantic-view result.",
        "seed": args.seed,
        "source_files": {tier: str(path) for tier, path in TIERS.items()},
        "input_sha256": {tier: sha256_file(path) for tier, path in TIERS.items()},
        "loaded_rows_after_extra_filter": {tier: len(rows) for tier, rows in loaded.items()},
        "requested_plan": plan,
        "selected_prompts": len(chosen_records),
        "selected_sources": len(chosen_records),
        "selected_source_words": sum(int(r["source_words"]) for r in chosen_records),
        "unique_docs": len({str(r.get("doc_id")) for r in chosen_records}),
        "selected_by_tier": dict(by_tier),
        "selected_by_quality_bucket": dict(by_bucket),
        "selected_by_domain_hit": dict(by_domain),
        "source_word_stats": stats([r["source_words"] for r in chosen_records]),
        "entity_count_stats": stats([len(r.get("entities") or []) for r in chosen_records]),
        "number_count_stats": stats([len(r.get("numbers") or []) for r in chosen_records]),
        "prompts_jsonl": str(prompts),
        "sources_jsonl": str(sources),
        "samples_json": str(samples),
        "future_generation": {
            "model_alias": "qwen3.5-9b", "prompts_jsonl": str(prompts),
            "output_jsonl": "experiments/archive/representation_and_objectives/training/runs/fineweb_rewrite_faithfulness_slice_qwen/outputs.jsonl",
            "batch_size": 64, "max_new_tokens": 80, "temperature": 0.15,
            "device": "cuda", "status": "proposed_not_executed",
        },
        "future_analysis_script": "experiments/archive/representation_and_objectives/training/scripts/analyze_fineweb_rewrite_faithfulness_slice.py",
        "use_policy": "These are prompts, not a training corpus. Generated outputs require faithfulness analysis before corpus materialization.",
    }
    metadata.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sample_payload = {
        "first_prompts": [
            {k: rec.get(k) for k in ["prompt_id", "selection_tier", "doc_id", "source_words", "entities", "numbers", "domain_hits", "flags", "source_text", "prompt"]}
            for rec in chosen_records[:12]
        ],
        "by_tier_first": {
            tier: [
                {k: rec.get(k) for k in ["prompt_id", "generation_index", "doc_id", "source_words", "entities", "numbers", "domain_hits", "flags", "source_text"]}
                for rec in sorted([r for r in chosen_records if r["selection_tier"] == tier], key=lambda x: x["generation_index"])[:10]
            ] for tier in plan
        },
    }
    samples.write_text(json.dumps(sample_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    generation_config.write_text(json.dumps(meta["future_generation"], indent=2) + "\n", encoding="utf-8")

    lines = ["# research FineWeb rewrite faithfulness slice\n\n"]
    lines.append("This is a prepared small generation test, not a launched job and not a training corpus.\n\n")
    lines.append(f"Prompts: `{prompts}`\n\n")
    lines.append(f"Sources: `{sources}`\n\n")
    lines.append(f"Metadata: `{metadata}`\n\n")
    lines.append(f"Samples: `{samples}`\n\n")
    lines.append(f"Selected {len(chosen_records)} complete-sentence simplification prompts with {meta['selected_source_words']:,} source words across {meta['unique_docs']:,} documents. Tier counts: {dict(by_tier)}.\n\n")
    lines.append("The analyzer should be run on generated outputs before any corpus materialization; acceptance must be interpreted by tier because the balanced source pool intentionally preserves breadth while remaining noisier than the strict factual tier.\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": meta["status"],
        "not_launched": True,
        "prompts": str(prompts),
        "metadata": str(metadata),
        "note": str(NOTE),
        "selected_prompts": len(chosen_records),
        "selected_source_words": meta["selected_source_words"],
        "selected_by_tier": dict(by_tier),
        "unique_docs": meta["unique_docs"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
