#!/usr/bin/env python3
"""Probe the public leader-linked FineWeb simplification-pair dataset surface.

This is reconnaissance for a later paired-rewrite mechanism screen. It reads only
metadata and a bounded streaming sample, computes word-count/field summaries, and
writes a local JSON/note. It does not train a model or use the sampled
content as a submission artifact by itself.
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import time
from collections import Counter

OUT_DIR = pathlib.Path("experiments/archive/compact_experience/data/fineweb_simplification_probe")
OUT_JSON = OUT_DIR / "probe_summary.json"
OUT_NOTE = pathlib.Path("research/notes/compact_experience/fineweb_simplification_probe.md")
DATASET = "go76dof/Fineweb_simplification_pairs"


def wc(s) -> int:
    return len(str(s).split()) if s is not None else 0


def compact_value(v, limit=400):
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    s = str(v).replace("\n", " ")
    return s[:limit] + ("…" if len(s) > limit else "")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str((pathlib.Path("data/external/hf_probe")).resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str((pathlib.Path("data/external/hub")).resolve()))
    os.environ.setdefault("HF_DATASETS_CACHE", str((pathlib.Path("data/external/datasets")).resolve()))
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    t0 = time.time()
    configs = []
    splits_by_config = {}
    try:
        configs = get_dataset_config_names(DATASET)
    except Exception as e:
        configs = ["<config_error:%s>" % type(e).__name__]
    for cfg in configs[:5]:
        if cfg.startswith("<"):
            continue
        try:
            splits_by_config[cfg] = get_dataset_split_names(DATASET, cfg)
        except Exception as e:
            splits_by_config[cfg] = ["<split_error:%s>" % type(e).__name__]

    ds = load_dataset(DATASET, split="train", streaming=True)
    first_rows = []
    field_counts = Counter()
    pair_candidates = Counter()
    word_stats = Counter()
    lens = []
    ratios = []
    total_rows = 0
    total_words_all_fields = 0
    field_word_sums = Counter()
    # probe 5000 rows if available; enough for format/length, not a corpus estimate.
    for i, x in zip(range(5000), ds):
        total_rows += 1
        field_counts.update(x.keys())
        if i < 8:
            first_rows.append({k: compact_value(v) for k, v in x.items()})
        str_fields = {k: v for k, v in x.items() if isinstance(v, str)}
        for k, v in str_fields.items():
            n = wc(v)
            field_word_sums[k] += n
            total_words_all_fields += n
        keys = set(str_fields)
        # Try common pair-field conventions.
        for a, b in [("original", "simplified"), ("source", "target"), ("text", "simplified"), ("complex", "simple"), ("sentence", "simple_sentence")]:
            if a in keys and b in keys:
                wa, wb = wc(str_fields[a]), wc(str_fields[b])
                lens.append((wa, wb))
                if wa > 0:
                    ratios.append(wb / wa)
                pair_candidates[(a, b)] += 1
        if "text" in keys:
            word_stats["text_words"] += wc(str_fields["text"])
    def stat(vals):
        vals = list(vals)
        if not vals:
            return None
        vals_sorted = sorted(vals)
        return {
            "n": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "min": vals_sorted[0],
            "p10": vals_sorted[int(0.10 * (len(vals_sorted)-1))],
            "p90": vals_sorted[int(0.90 * (len(vals_sorted)-1))],
            "max": vals_sorted[-1],
        }
    orig_lens = [a for a, _ in lens]
    simp_lens = [b for _, b in lens]
    payload = {
        "dataset": DATASET,
        "configs": configs,
        "splits_by_config": splits_by_config,
        "sample_rows": total_rows,
        "first_rows": first_rows,
        "field_counts": dict(field_counts),
        "field_word_sums": dict(field_word_sums),
        "pair_candidates": {str(k): v for k, v in pair_candidates.items()},
        "original_word_stats": stat(orig_lens),
        "simplified_word_stats": stat(simp_lens),
        "simplified_to_original_ratio_stats": stat(ratios),
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# FineWeb simplification-pair probe",
        "",
        f"Dataset: `{DATASET}`",
        f"JSON: `{OUT_JSON}`",
        "",
        f"Sampled streaming rows: {total_rows}",
        f"Fields: `{dict(field_counts)}`",
        f"Pair-field candidates: `{payload['pair_candidates']}`",
        "",
        "## Word statistics over detected pair fields",
        "",
        f"Original: `{payload['original_word_stats']}`",
        f"Simplified: `{payload['simplified_word_stats']}`",
        f"Simplified/original ratio: `{payload['simplified_to_original_ratio_stats']}`",
        "",
        "## First rows (truncated)",
        "",
        "```json",
        json.dumps(first_rows, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(OUT_JSON), "note": str(OUT_NOTE), "sample_rows": total_rows, "pair_candidates": payload["pair_candidates"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
