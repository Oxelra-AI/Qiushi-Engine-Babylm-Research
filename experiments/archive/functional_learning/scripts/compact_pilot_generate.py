#!/usr/bin/env python3
"""research: Prepare and launch Qwen-internal compact rewrite generation pilot.

Takes the inherited 512-prompt slice, enriches it with entity/number metadata from
the full selected-pairs file, formats scientific generation prompts, and
saves the generation input JSONL.

This script only prepares the input. Generation is run separately by
the Qwen3.5-9B teacher (model identifier qwen3.5-9b).

Usage:
    python compact_pilot_generate.py prepare
    python compact_pilot_generate.py enrich --generation-output OUTPUT.jsonl
"""
import argparse, json, sys, os, re
from pathlib import Path
from collections import Counter

STUDY = Path("experiments/archive/functional_learning")
WS = STUDY
DATA = WS / "data" / "compact_pilot"
SCRIPTS = WS / "scripts"

# Input paths
PROMPT_SLICE = Path("experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_compaction_prompt_slice512.jsonl")
SELECTED_PAIRS = Path("experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl")
MANIFEST = Path("experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_compaction_manifest.jsonl")


def prepare(args):
    """Prepare generation input from the 512-prompt slice."""
    DATA.mkdir(parents=True, exist_ok=True)

    # Load prompt slice
    with open(PROMPT_SLICE) as f:
        prompts = [json.loads(l) for l in f]
    print(f"Loaded {len(prompts)} prompts from slice", flush=True)

    # Load entity/number metadata from selected pairs
    pair_meta = {}
    with open(SELECTED_PAIRS) as f:
        for line in f:
            row = json.loads(line)
            pair_meta[row["pair_id"]] = {
                "entity_source": row.get("entity_source", []),
                "entity_rewrite": row.get("entity_rewrite", []),
                "num_source": row.get("num_source", []),
                "num_rewrite": row.get("num_rewrite", []),
                "content_overlap": row.get("content_overlap", 0),
                "len_ratio": row.get("len_ratio", 1.0),
            }

    # Enrich prompts with metadata
    enriched = []
    n_entity = 0
    n_number = 0
    for p in prompts:
        pid = p["pair_id"]
        meta = pair_meta.get(pid, {})
        rec = {**p, **meta}
        enriched.append(rec)
        if meta.get("entity_source"):
            n_entity += 1
        if meta.get("num_source"):
            n_number += 1

    print(f"Enriched: {n_entity} with entities, {n_number} with numbers", flush=True)

    # Save enriched metadata (for post-generation analysis)
    meta_path = DATA / "pilot_enriched_metadata.jsonl"
    with open(meta_path, "w") as f:
        for rec in enriched:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save generation input (just the prompt field in the format expected by the generator)
    gen_input_path = DATA / "generation_input.jsonl"
    with open(gen_input_path, "w") as f:
        for rec in enriched:
            f.write(json.dumps({"prompt": rec["prompt"], "pair_id": rec["pair_id"]}, ensure_ascii=False) + "\n")

    # Source distribution
    src_counts = Counter(r["source"] for r in enriched)

    summary = {
        "status": "COMPACT_PILOT_PREPARED",
        "n_prompts": len(enriched),
        "n_with_entities": n_entity,
        "n_with_numbers": n_number,
        "source_distribution": dict(src_counts),
        "generation_input": str(gen_input_path),
        "enriched_metadata": str(meta_path),
        "prompt_slice_source": str(PROMPT_SLICE),
        "selected_pairs_source": str(SELECTED_PAIRS),
    }
    with open(DATA / "preparation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def enrich(args):
    """Join generation output with enriched metadata for quality analysis."""
    gen_path = Path(args.generation_output)
    if not gen_path.exists():
        print(f"ERROR: generation output not found: {gen_path}", file=sys.stderr)
        sys.exit(1)

    meta_path = DATA / "pilot_enriched_metadata.jsonl"
    if not meta_path.exists():
        print(f"ERROR: metadata not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    # Load metadata as ordered list (matches generation input order)
    meta_list = []
    with open(meta_path) as f:
        for line in f:
            meta_list.append(json.loads(line))

    # Load generation output
    gen_results = []
    with open(gen_path) as f:
        for line in f:
            gen_results.append(json.loads(line))

    # Join by index (generation output has 'index' matching input line order)
    joined = []
    for gen in gen_results:
        idx = gen.get("index", len(joined))
        if idx < len(meta_list):
            meta = meta_list[idx]
        else:
            meta = {}
        compact_text = gen.get("output", gen.get("text", gen.get("completion", "")))
        # Clean the output: strip whitespace, remove quotes if wrapped
        compact_text = compact_text.strip()
        if compact_text.startswith('"') and compact_text.endswith('"'):
            compact_text = compact_text[1:-1]

        pid = meta.get("pair_id", gen.get("pair_id", f"idx_{idx}"))
        joined.append({
            "pair_id": pid,
            "source": meta.get("source", ""),
            "example_id": meta.get("example_id", 0),
            "original": meta.get("original", ""),
            "current_rewrite": meta.get("current_rewrite", ""),
            "compact_rewrite": compact_text,
            "target_compact_words": meta.get("target_compact_rewrite_words", 0),
            "expected_saved_words": meta.get("expected_saved_words", 0),
            "entity_source": meta.get("entity_source", []),
            "num_source": meta.get("num_source", []),
            "content_overlap": meta.get("content_overlap", 0),
        })

    out_path = DATA / "pilot_generation_joined.jsonl"
    with open(out_path, "w") as f:
        for rec in joined:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Joined {len(joined)} records -> {out_path}", flush=True)
    return str(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("prepare")
    enrich_p = sub.add_parser("enrich")
    enrich_p.add_argument("--generation-output", required=True)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "enrich":
        enrich(args)
    else:
        ap.print_help()
        sys.exit(1)
