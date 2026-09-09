#!/usr/bin/env python3
"""research: Build innovation group map for innovation-biased WWM.

Precomputes which word groups in each changed pool row are:
  - innovation: source-absent rewrite groups (normalized text not in paired source)
  - copyable: rewrite groups whose normalized text appears in the paired source
  - source: groups within source token spans

Used at training time to bias WWM group selection toward innovation targets.
CPU-only, processes 3,005 changed rows from the 10M pool.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import pathlib
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

from transformers import AutoTokenizer

# ─── Paths ────────────────────────────────────────────────────────────────────
def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"

POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
SPAN_JSONL = WORKSPACE / "data/pair_span_map/pair_span_map.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
OUT_DIR = WORKSPACE / "data/innovation_group_map"

EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOK_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
SEQ_LEN = 256


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Word-group assignment (matches MaskedChunkDataset.__getitem__) ────────
def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def compute_word_groups(input_ids: list[int], attention_mask: list[int],
                        tokenizer, special_ids: set[int]) -> list[int]:
    """Return per-position word-group IDs, matching the training dataset exactly."""
    ws_cache: dict[int, bool] = {}
    groups = [-1] * len(input_ids)
    gid = -1
    for i in range(len(input_ids)):
        if attention_mask[i] == 0:
            continue
        tid = input_ids[i]
        if tid in special_ids:
            continue
        if tid not in ws_cache:
            s = tokenizer.convert_ids_to_tokens(tid)
            ws_cache[tid] = bool(s is not None and is_word_start(str(s)))
        if gid < 0 or ws_cache[tid] or i == 0:
            gid += 1
        groups[i] = gid
    return groups


# ─── Text normalization (matches research probes) ──────────────────────────
def norm_text(text: str) -> str:
    text = text.lower().replace("Ġ", " ").replace("▁", " ")
    pieces = re.findall(r"[a-z0-9]+", text)
    return " ".join(pieces)


def token_norm(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    return norm_text(tokenizer.decode([int(x) for x in ids], clean_up_tokenization_spaces=False))


# ─── Build the map ─────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify inputs
    pool_sha = sha256_file(POOL_10M)
    tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    assert pool_sha == EXPECTED_POOL_SHA, f"pool SHA mismatch: {pool_sha}"
    assert tok_sha == EXPECTED_TOK_SHA, f"tokenizer SHA mismatch: {tok_sha}"

    # Load pair span map
    span_by_ex: dict[int, dict[str, Any]] = {}
    with SPAN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            span_by_ex[int(obj["example_id"])] = obj

    # Load pool texts for changed rows only
    changed_ids = set(span_by_ex.keys())
    pool_texts: dict[int, str] = {}
    with POOL_10M.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            eid = int(obj.get("example_id", -1))
            if eid in changed_ids:
                pool_texts[eid] = str(obj["text"])

    assert len(pool_texts) == len(changed_ids), f"loaded {len(pool_texts)} != {len(changed_ids)}"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    special_ids = set(tokenizer.all_special_ids)

    # Build map
    innovation_map: dict[int, dict[str, Any]] = {}
    totals = Counter()
    per_row_stats: list[dict[str, int]] = []

    for eid in sorted(changed_ids):
        text = pool_texts[eid]
        span_info = span_by_ex[eid]

        # Tokenize exactly as training does
        enc = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=SEQ_LEN, padding="max_length")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        groups = compute_word_groups(input_ids, attention_mask, tokenizer, special_ids)

        # Build position-to-group map
        n_total_groups = max(groups) + 1 if any(g >= 0 for g in groups) else 0

        # Collect source norm sets and source group IDs per pair
        source_gids: set[int] = set()
        rewrite_gids: set[int] = set()

        innovation_gids: set[int] = set()
        copyable_gids: set[int] = set()

        for pair in span_info.get("pairs", []):
            if pair.get("visibility") != "both_visible":
                continue

            # Source: collect normalized texts and group IDs
            src_norms: set[str] = set()
            for rng in pair.get("source_token_ranges", []):
                start, end = int(rng[0]), int(rng[1])
                # Group positions in this source range
                gid_positions: dict[int, list[int]] = defaultdict(list)
                for pos in range(start, end):
                    if pos < len(groups) and groups[pos] >= 0:
                        gid_positions[groups[pos]].append(pos)
                for gid, posns in gid_positions.items():
                    source_gids.add(gid)
                    ids = [input_ids[p] for p in posns]
                    n = token_norm(tokenizer, ids)
                    if n:
                        src_norms.add(n)

            # Rewrite: classify groups
            for rng in pair.get("rewrite_token_ranges", []):
                start, end = int(rng[0]), int(rng[1])
                gid_positions: dict[int, list[int]] = defaultdict(list)
                for pos in range(start, end):
                    if pos < len(groups) and groups[pos] >= 0:
                        gid_positions[groups[pos]].append(pos)
                for gid, posns in gid_positions.items():
                    rewrite_gids.add(gid)
                    ids = [input_ids[p] for p in posns]
                    n = token_norm(tokenizer, ids)
                    if not n or len(n) <= 1:
                        totals["skip_empty_or_single"] += 1
                        continue
                    if n in src_norms:
                        copyable_gids.add(gid)
                        totals["copyable_groups"] += 1
                    else:
                        innovation_gids.add(gid)
                        totals["innovation_groups"] += 1

        # Groups that appear in both source and rewrite should be classified conservatively
        # (a group spanning a source-rewrite boundary is rare due to word-start splitting)
        # If a group is both innovation and source, keep as innovation
        # If a group is both copyable and source, keep as copyable

        n_innov = len(innovation_gids)
        n_copy = len(copyable_gids)
        n_source = len(source_gids - innovation_gids - copyable_gids)
        n_other = n_total_groups - n_innov - n_copy - n_source

        innovation_map[eid] = {
            "innovation_gids": sorted(innovation_gids),
            "copyable_gids": sorted(copyable_gids),
            "source_gids": sorted(source_gids - innovation_gids - copyable_gids),
            "n_total_groups": n_total_groups,
            "n_innovation": n_innov,
            "n_copyable": n_copy,
            "n_source": n_source,
            "n_other": n_other,
        }
        per_row_stats.append({
            "example_id": eid,
            "n_total": n_total_groups,
            "n_innov": n_innov,
            "n_copy": n_copy,
            "n_source": n_source,
            "n_other": n_other,
        })
        totals["rows_processed"] += 1

    # Save map
    out_json = OUT_DIR / "innovation_group_map.json"
    out_json.write_text(json.dumps(
        {str(k): v for k, v in innovation_map.items()},
        separators=(",", ":"),
    ) + "\n", encoding="utf-8")

    # Summary statistics
    def stat(vals):
        if not vals:
            return {}
        return {"n": len(vals), "mean": round(statistics.mean(vals), 2),
                "median": round(statistics.median(vals), 1),
                "min": min(vals), "max": max(vals)}

    summary = {
        "status": "INNOVATION_GROUP_MAP",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "pool_sha256": pool_sha,
            "tokenizer_sha256": tok_sha,
            "span_rows": len(span_by_ex),
        },
        "totals": dict(totals),
        "per_row_stats": {
            "n_total": stat([r["n_total"] for r in per_row_stats]),
            "n_innovation": stat([r["n_innov"] for r in per_row_stats]),
            "n_copyable": stat([r["n_copy"] for r in per_row_stats]),
            "n_source": stat([r["n_source"] for r in per_row_stats]),
            "n_other": stat([r["n_other"] for r in per_row_stats]),
        },
        "budget_analysis": {
            "note": "With p_innov=0.5 p_copy=0.0 and standard 0.15 total budget",
            "avg_innovation_selected": round(0.5 * statistics.mean([r["n_innov"] for r in per_row_stats]), 2),
            "avg_standard_selected": round(0.15 * statistics.mean([r["n_total"] for r in per_row_stats]), 2),
            "avg_p_other": round(
                (0.15 * statistics.mean([r["n_total"] for r in per_row_stats])
                 - 0.5 * statistics.mean([r["n_innov"] for r in per_row_stats]))
                / max(1, statistics.mean([r["n_source"] + r["n_other"] for r in per_row_stats])),
                4
            ),
        },
        "output_map": str(out_json),
        "map_entries": len(innovation_map),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (OUT_DIR / "innovation_group_map_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Readable note
    note_lines = [
        "# research — innovation group map",
        "",
        "CPU-only; no model, no training, no evaluation.",
        "",
        f"- changed rows processed: `{totals['rows_processed']}`",
        f"- total innovation groups (one pass): `{totals.get('innovation_groups', 0)}`",
        f"- total copyable groups (one pass): `{totals.get('copyable_groups', 0)}`",
        f"- skipped empty/single: `{totals.get('skip_empty_or_single', 0)}`",
        "",
        "## Per-row statistics",
    ]
    for key, label in [("n_total", "total groups"), ("n_innovation", "innovation"),
                        ("n_copyable", "copyable"), ("n_source", "source"), ("n_other", "other")]:
        s = summary["per_row_stats"][key]
        note_lines.append(f"- {label}: mean={s.get('mean')}, median={s.get('median')}, min={s.get('min')}, max={s.get('max')}")
    note_lines.extend([
        "",
        "## Budget analysis (p_innov=0.5, p_copy=0.0)",
        f"- avg innovation selected per changed row: {summary['budget_analysis']['avg_innovation_selected']}",
        f"- avg standard total selected per row: {summary['budget_analysis']['avg_standard_selected']}",
        f"- avg p_other for budget match: {summary['budget_analysis']['avg_p_other']}",
        "",
        f"Map: `{out_json}`",
        f"Summary: `{OUT_DIR / 'innovation_group_map_summary.json'}`",
    ])
    (OUT_DIR / "innovation_group_map.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "map_entries": summary["map_entries"],
        "innovation_groups": totals.get("innovation_groups", 0),
        "copyable_groups": totals.get("copyable_groups", 0),
        "out_map": str(out_json),
        "out_summary": str(OUT_DIR / "innovation_group_map_summary.json"),
        "out_note": str(OUT_DIR / "innovation_group_map.md"),
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
