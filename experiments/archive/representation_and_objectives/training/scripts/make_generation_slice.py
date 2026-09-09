#!/usr/bin/env python3
"""Build a small, provenance-preserving generation slice from the research prompt pool.

The slice is deliberately small enough for a small generation comparison,
but stratified enough to test the actual scientific risk identified in the source-faithfulness analysis:
are Qwen-generated SimpleWiki derivatives source-faithful, entity/numeric preserving,
and broad enough in factual/causal coverage to justify larger data construction?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("experiments/archive/representation_and_objectives")
SHARD0 = ROOT / "training/data/factual_prompts_shard0_expand.jsonl"
SHARD1 = ROOT / "training/data/factual_prompts_shard1_simpara.jsonl"
OUT_DIR = ROOT / "training/data/generation_slice"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-type", type=int, default=128)
    p.add_argument("--seed", type=int, default=82903)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(SHARD0) + read_jsonl(SHARD1)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[str(r.get("type", "unknown"))].append(r)

    rng = random.Random(args.seed)
    selected: list[dict] = []
    for typ in ["expansion", "simplification", "paraphrase"]:
        pool = by_type[typ]
        if len(pool) < args.per_type:
            raise RuntimeError(f"not enough {typ} rows: {len(pool)} < {args.per_type}")
        # Stratify by source length tercile so the slice is not dominated by short rows.
        sorted_pool = sorted(pool, key=lambda x: int(x.get("source_words", len(str(x.get("source_text", "")).split()))))
        buckets = [sorted_pool[: len(sorted_pool)//3], sorted_pool[len(sorted_pool)//3: 2*len(sorted_pool)//3], sorted_pool[2*len(sorted_pool)//3:]]
        take_counts = [args.per_type // 3, args.per_type // 3, args.per_type - 2*(args.per_type // 3)]
        for bucket, take in zip(buckets, take_counts):
            selected.extend(rng.sample(bucket, take))

    rng.shuffle(selected)
    for i, r in enumerate(selected):
        r["slice_index"] = i

    prompt_path = out_dir / "slice_prompts.jsonl"
    meta_path = out_dir / "slice_prompt_metadata.json"
    with prompt_path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    type_counts = Counter(r["type"] for r in selected)
    source_words_by_type = defaultdict(int)
    article_counts = Counter()
    for r in selected:
        source_words_by_type[r["type"]] += int(r.get("source_words", len(r.get("source_text", "").split())))
        article_counts[str(r.get("source_article", ""))] += 1

    meta = {
        "status": "GENERATION_SLICE_PROMPTS_MATERIALIZED",
        "purpose": "fast source-faithfulness/breadth/provenance test before any 50k generation or full training package",
        "seed": args.seed,
        "per_type": args.per_type,
        "total_prompts": len(selected),
        "type_counts": dict(type_counts),
        "source_words_by_type": dict(source_words_by_type),
        "unique_source_articles": len(article_counts),
        "max_prompts_per_article": max(article_counts.values()) if article_counts else 0,
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "input_shards": {
            "shard0": {"path": str(SHARD0), "sha256": sha256_file(SHARD0)},
            "shard1": {"path": str(SHARD1), "sha256": sha256_file(SHARD1)},
        },
        "sample_ids": [r["id"] for r in selected[:12]],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
