#!/usr/bin/env python3
"""Prepare a compact semantic-review bundle for selected density contrast pairs.

The automatic analyzer is intentionally conservative but lexical. This bundle exposes
actual source/rewrite pairs from the selected high-precision core and compact-added
sets, especially low-content-recall and number/entity-heavy examples, so an independent
scientist/LLM reviewer can judge whether compact views are mostly faithful compression
or harmful information deletion.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_META = ROOT / "data/density_core_reinvestment_full/density_core_reinvestment_metadata.json"
OUT_JSON = ROOT / "data/density_semantic_review/density_semantic_review_bundle.json"
OUT_MD = (ROOT.parents[2] / 'research/notes/frontier_consolidation/density_semantic_review_bundle.md')


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def slim(r: dict[str, Any], group: str) -> dict[str, Any]:
    return {
        "group": group,
        "pair_id": r.get("pair_id"),
        "key": r.get("key"),
        "sentence_id": r.get("sentence_id"),
        "doc_id": r.get("doc_id"),
        "domain_hits": r.get("domain_hits") or [],
        "source_words": r.get("source_words"),
        "rewrite_words": r.get("rewrite_words"),
        "length_ratio": (r.get("rewrite_words") or 0) / max(1, (r.get("source_words") or 0)),
        "content_recall": r.get("content_recall"),
        "content_overlap": r.get("content_overlap"),
        "entity_recall": r.get("entity_recall"),
        "number_recall": r.get("number_recall"),
        "soft_flags": r.get("soft_flags") or [],
        "source_risks": r.get("source_risks") or [],
        "source_text": r.get("source_text"),
        "rewrite_text": r.get("rewrite_text"),
    }


def take_unique(existing: set[str], rows: list[dict[str, Any]], n: int, group: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        pid = str(r.get("pair_id") or r.get("key"))
        if pid in existing:
            continue
        existing.add(pid)
        out.append(slim(r, group))
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=str(DEFAULT_META))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-md", default=str(OUT_MD))
    ap.add_argument("--seed", type=int, default=829142)
    args = ap.parse_args()
    meta = json.loads(pathlib.Path(args.metadata).read_text(encoding="utf-8"))
    files = meta["files"]
    near = read_jsonl(pathlib.Path(files["near_core_pairs"]))
    compact_core = read_jsonl(pathlib.Path(files["compact_core_pairs"]))
    compact_added = read_jsonl(pathlib.Path(files["compact_added_pairs"]))

    rng = random.Random(args.seed)
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    examples += take_unique(seen, sorted(compact_core, key=lambda r: (r.get("content_recall") or 0, r.get("content_overlap") or 0)), 16, "compact_core_low_content")
    examples += take_unique(seen, sorted(compact_added, key=lambda r: (r.get("content_recall") or 0, r.get("content_overlap") or 0)), 12, "compact_added_low_content")
    num_or_ent = [r for r in compact_core + compact_added if "quant_numeric" in (r.get("domain_hits") or []) or (r.get("entity_recall") or 1.0) < 1.0]
    rng.shuffle(num_or_ent)
    examples += take_unique(seen, num_or_ent, 12, "compact_numeric_or_entity")
    random_compact = list(compact_core + compact_added)
    rng.shuffle(random_compact)
    examples += take_unique(seen, random_compact, 12, "compact_random")
    near_copy = [r for r in near if "near_copy_view" in (r.get("soft_flags") or [])]
    rng.shuffle(near_copy)
    examples += take_unique(seen, near_copy, 8, "near_core_near_copy")
    near_random = list(near)
    rng.shuffle(near_random)
    examples += take_unique(seen, near_random, 8, "near_core_random")

    out = {
        "status": "DENSITY_SEMANTIC_REVIEW_BUNDLE_PREPARED",
        "purpose": "Actual selected high-precision source/rewrite examples for judging compact fidelity and near-copy behavior before training.",
        "metadata": str(args.metadata),
        "counts": {
            "near_core_pairs": len(near),
            "compact_core_pairs": len(compact_core),
            "compact_added_pairs": len(compact_added),
            "examples": len(examples),
        },
        "groups": {},
        "examples": examples,
    }
    for e in examples:
        out["groups"].setdefault(e["group"], 0)
        out["groups"][e["group"]] += 1
    out_json = pathlib.Path(args.out_json)
    out_md = pathlib.Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research density semantic review bundle", "", out["purpose"], "", f"Metadata: `{args.metadata}`", "", f"Bundle JSON: `{out_json}`", ""]
    for i, e in enumerate(examples, 1):
        lines.append(f"## {i}. {e['group']} | {e['pair_id']}")
        lines.append(f"Metrics: source_words={e['source_words']} rewrite_words={e['rewrite_words']} ratio={e['length_ratio']:.3f} recall={e['content_recall']} overlap={e['content_overlap']} entity={e['entity_recall']} number={e['number_recall']} domains={e['domain_hits']} flags={e['soft_flags']} risks={e['source_risks']}")
        lines.append(f"SOURCE: {e['source_text']}")
        lines.append(f"REWRITE: {e['rewrite_text']}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": out["status"], "json": str(out_json), "md": str(out_md), "counts": out["counts"], "groups": out["groups"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
