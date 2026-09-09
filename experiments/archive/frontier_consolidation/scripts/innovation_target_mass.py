#!/usr/bin/env python3
"""research: corpus-side target mass for compact-view conditional innovation.

CPU-only. No model update/evaluation. Counts how many rewrite word groups in all
visible compact source/rewrite pairs are direct copies from the source versus
source-absent innovations, and how much target mass a changed-span objective would
add if applied to the already-counted compact-view rows.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
import json
import pathlib
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

from transformers import AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
COND_SCRIPT = WORKSPACE / "scripts/conditional_innovation_probe.py"
OUT_DIR = WORKSPACE / "data/innovation_target_mass"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cond = load_module("conditional_for_mass", COND_SCRIPT)


def summarize(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(vals), "mean": float(statistics.mean(vals)), "median": float(statistics.median(vals)), "min": float(min(vals)), "max": float(max(vals))}


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sha = cond.collate_mod.sha256_file(cond.collate_mod.TRAIN_100M)
    tok_sha = cond.collate_mod.sha256_file(cond.collate_mod.TOKENIZER_DIR / "tokenizer.json")
    if train_sha != cond.collate_mod.EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {train_sha}")
    if tok_sha != cond.collate_mod.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    span_by_ex = cond.collate_mod.load_span_map(cond.collate_mod.SPAN_JSONL)
    example_ids = sorted(span_by_ex)
    examples = cond.read_examples_by_example_id(cond.collate_mod.TRAIN_100M, example_ids)
    tokenizer = AutoTokenizer.from_pretrained(str(cond.collate_mod.TOKENIZER_DIR), use_fast=True)
    ds = cond.collate_mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, cond.collate_mod.SEQ_LEN)
    totals = Counter()
    by_row = []
    by_pair_counts = []
    innovation_norms = Counter()
    copyable_norms = Counter()
    relation_norms = Counter()
    rows_with_innovation = 0
    pairs_with_innovation = 0
    for idx in range(len(ds)):
        item = ds[idx]
        row_counts = Counter()
        if item.get("aux_error"):
            totals[f"aux_error_{item['aux_error']}"] += 1
            continue
        input_ids = item["input_ids"]
        word_group = item["word_group"]
        for p in item.get("aux_pairs") or []:
            pair_counts = Counter()
            src_norms = set()
            for _, posns in cond.group_positions_in_ranges(word_group, p["source_ranges"]):
                ids = [int(input_ids[q].item()) for q in posns]
                n = cond.token_norm(tokenizer, ids)
                if n:
                    src_norms.add(n)
            for _, posns in cond.group_positions_in_ranges(word_group, p["rewrite_ranges"]):
                ids = [int(input_ids[q].item()) for q in posns]
                n = cond.token_norm(tokenizer, ids)
                if not n or len(n) <= 1:
                    totals["skip_empty_or_single"] += 1
                    continue
                cat = "copyable" if n in src_norms else "innovation"
                first_word = n.split()[0] if n.split() else n
                cue = "relation_cue" if first_word in cond.RELATION_WORDS else "content_or_other"
                tokens = len(posns)
                for key in [cat, cue, f"{cat}:{cue}"]:
                    totals[f"groups:{key}"] += 1
                    totals[f"tokens:{key}"] += tokens
                    row_counts[f"groups:{key}"] += 1
                    pair_counts[f"groups:{key}"] += 1
                totals["groups:all"] += 1
                totals["tokens:all"] += tokens
                row_counts["groups:all"] += 1
                pair_counts["groups:all"] += 1
                if cat == "innovation":
                    innovation_norms[n] += 1
                    if cue == "relation_cue":
                        relation_norms[n] += 1
                else:
                    copyable_norms[n] += 1
            if pair_counts.get("groups:innovation", 0) > 0:
                pairs_with_innovation += 1
            pair_counts["pair_id"] = p.get("pair_id", "")
            by_pair_counts.append(dict(pair_counts))
        if row_counts.get("groups:innovation", 0) > 0:
            rows_with_innovation += 1
        row_counts["global_row_1based"] = int(item["global_row_1based"])
        row_counts["example_id"] = int(item["example_id"])
        by_row.append(dict(row_counts))
    # Convert to one-pass and ten-pass target mass. The same 10M pool is repeated 10 times.
    keys = sorted(k for k in totals if k.startswith("groups:") or k.startswith("tokens:"))
    mass = {k: int(totals[k]) for k in keys}
    for k in list(mass):
        mass[f"ten_pass_{k}"] = int(mass[k] * 10)
    denom_groups = totals.get("groups:all", 1)
    denom_tokens = totals.get("tokens:all", 1)
    fractions = {}
    for k, v in mass.items():
        if k.startswith("groups:"):
            fractions[k.replace("groups:", "group_frac:")] = float(v / denom_groups)
        if k.startswith("tokens:"):
            fractions[k.replace("tokens:", "token_frac:")] = float(v / denom_tokens)
    row_stats = {}
    for key in ["groups:all", "groups:innovation", "groups:copyable", "groups:innovation:relation_cue", "groups:innovation:content_or_other"]:
        row_stats[key] = summarize([float(r.get(key, 0)) for r in by_row])
    summary = {
        "status": "INNOVATION_TARGET_MASS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Count full compact-pair rewrite target mass split into copyable and source-absent innovation groups for potential changed-span objectives.",
        "inputs": {
            "train_sha256": train_sha,
            "tokenizer_sha256": tok_sha,
            "span_examples": len(span_by_ex),
            "examples_loaded": len(examples),
        },
        "totals_one_pass": {k: int(totals[k]) for k in sorted(totals)},
        "target_mass": mass,
        "fractions_within_rewrite_groups": fractions,
        "rows_with_innovation": rows_with_innovation,
        "row_frac_with_innovation": rows_with_innovation / len(by_row),
        "pairs_with_innovation": pairs_with_innovation,
        "pair_frac_with_innovation": pairs_with_innovation / max(1, len(by_pair_counts)),
        "row_stats": row_stats,
        "top_innovation_norms": innovation_norms.most_common(40),
        "top_copyable_norms": copyable_norms.most_common(40),
        "top_relation_innovation_norms": relation_norms.most_common(40),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = OUT_DIR / "innovation_target_mass.json"
    out_md = OUT_DIR / "innovation_target_mass.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — compact-pair innovation target mass",
        "",
        "CPU-only; no model update, no official evaluation, no corpus/tokenizer change, no H100 work.",
        "",
        f"- changed rows loaded: `{len(examples)}` of `{len(span_by_ex)}`",
        f"- train SHA: `{train_sha}`",
        f"- tokenizer SHA: `{tok_sha}`",
        "",
        "## One-pass rewrite word-group mass",
        f"- all rewrite groups: `{totals['groups:all']}`; all rewrite tokens in groups: `{totals['tokens:all']}`",
        f"- copyable groups: `{totals['groups:copyable']}` ({totals['groups:copyable']/denom_groups:.3f}); tokens `{totals['tokens:copyable']}` ({totals['tokens:copyable']/denom_tokens:.3f})",
        f"- innovation groups: `{totals['groups:innovation']}` ({totals['groups:innovation']/denom_groups:.3f}); tokens `{totals['tokens:innovation']}` ({totals['tokens:innovation']/denom_tokens:.3f})",
        f"- innovation relation-cue groups: `{totals['groups:innovation:relation_cue']}` ({totals['groups:innovation:relation_cue']/denom_groups:.3f}); tokens `{totals['tokens:innovation:relation_cue']}` ({totals['tokens:innovation:relation_cue']/denom_tokens:.3f})",
        f"- innovation content/other groups: `{totals['groups:innovation:content_or_other']}` ({totals['groups:innovation:content_or_other']/denom_groups:.3f}); tokens `{totals['tokens:innovation:content_or_other']}` ({totals['tokens:innovation:content_or_other']/denom_tokens:.3f})",
        f"- rows with at least one innovation group: `{rows_with_innovation}` ({rows_with_innovation/len(by_row):.3f}); pairs with innovation: `{pairs_with_innovation}` ({pairs_with_innovation/max(1,len(by_pair_counts)):.3f})",
        "",
        "## Ten-pass exposure equivalent inside the fixed 100M stream",
        f"- innovation groups over ten passes: `{totals['groups:innovation']*10}`; innovation tokens over ten passes: `{totals['tokens:innovation']*10}`",
        f"- relation-cue innovation groups over ten passes: `{totals['groups:innovation:relation_cue']*10}`; tokens: `{totals['tokens:innovation:relation_cue']*10}`",
        "",
        "## Top source-absent innovation normalized words",
    ]
    for word, c in innovation_norms.most_common(25):
        lines.append(f"- `{word}`: {c}")
    lines += ["", "## Top relation-cue innovations"]
    for word, c in relation_norms.most_common(25):
        lines.append(f"- `{word}`: {c}")
    lines += ["", "Full JSON: `" + str(out_json) + "`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
