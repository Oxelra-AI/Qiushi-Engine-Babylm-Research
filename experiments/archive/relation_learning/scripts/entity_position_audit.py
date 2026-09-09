#!/usr/bin/env python3
"""research: Entity position/order shortcut audit for binding factorial pairs.

Each pair shares the same source and update sentences but has a different query
sentence at the end. A position shortcut would exist if the updated entity 
systematically appears in a specific relative position (first/last/closer to
query) compared to the unchanged entity.

Uses pair_half='A' (unchanged_entity query) and pair_half='B' (updated_entity query).
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import statistics
import sys
from collections import Counter

_SCRIPT = _public_path('experiments/archive/relation_learning/scripts/entity_position_audit.py')
ROOT = _public_path('experiments/archive/relation_learning/data')


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_first_position(text: str, entity: str) -> int:
    """Find first position of entity in text (case-insensitive). Returns -1 if not found."""
    idx = text.lower().find(entity.lower())
    return idx


def analyze_pairs(rows: list[dict], split_name: str) -> dict:
    """Analyze entity position patterns in paired rows."""
    # Group by pair_id
    pairs = {}
    for r in rows:
        pid = r["pair_id"]
        ph = r.get("pair_half", "")
        if ph in ("A", "B"):
            pairs.setdefault(pid, {})[ph] = r
    
    # Filter complete pairs
    complete = {k: v for k, v in pairs.items() if "A" in v and "B" in v}
    
    stats = {
        "split": split_name,
        "total_pairs": len(complete),
        "updated_first_in_source": 0,
        "unchanged_first_in_source": 0,
        "same_position_in_source": 0,
        "one_or_both_missing_in_source": 0,
        "updated_first_in_context_prefix": 0,
        "unchanged_first_in_context_prefix": 0,
        "same_in_context_prefix": 0,
        "missing_in_context_prefix": 0,
        "position_diffs_source": [],
        "position_diffs_context": [],
    }
    
    for pid, pair in complete.items():
        row_a = pair["A"]  # queries the unchanged entity
        row_b = pair["B"]  # queries the updated entity
        
        unchanged_entity = row_a["query_entity"]  # = target_entity
        updated_entity = row_b["query_entity"]  # = updated_entity
        
        source_sent = row_a.get("source_sentence", "")
        update_sent = row_a.get("update_sentence", "")
        
        # Shared prefix = source + update sentences (before the query-specific suffix)
        shared_prefix = source_sent + " " + update_sent
        
        # Check source sentence positions
        src_unch = find_first_position(source_sent, unchanged_entity)
        src_upd = find_first_position(source_sent, updated_entity)
        
        if src_unch >= 0 and src_upd >= 0:
            diff = src_upd - src_unch
            stats["position_diffs_source"].append(diff)
            if diff < 0:
                stats["updated_first_in_source"] += 1
            elif diff > 0:
                stats["unchanged_first_in_source"] += 1
            else:
                stats["same_position_in_source"] += 1
        else:
            stats["one_or_both_missing_in_source"] += 1
        
        # Check shared context prefix positions
        ctx_unch = find_first_position(shared_prefix, unchanged_entity)
        ctx_upd = find_first_position(shared_prefix, updated_entity)
        
        if ctx_unch >= 0 and ctx_upd >= 0:
            diff = ctx_upd - ctx_unch
            stats["position_diffs_context"].append(diff)
            if diff < 0:
                stats["updated_first_in_context_prefix"] += 1
            elif diff > 0:
                stats["unchanged_first_in_context_prefix"] += 1
            else:
                stats["same_in_context_prefix"] += 1
        else:
            stats["missing_in_context_prefix"] += 1
    
    # Compute statistics
    for key in ["source", "context"]:
        diffs = stats[f"position_diffs_{key}"]
        n_valid = len(diffs)
        if n_valid > 0:
            stats[f"{key}_n_valid"] = n_valid
            stats[f"{key}_mean_diff"] = round(statistics.mean(diffs), 2)
            stats[f"{key}_median_diff"] = round(statistics.median(diffs), 2)
            stats[f"{key}_std_diff"] = round(statistics.stdev(diffs), 2) if n_valid > 1 else 0
            first_key = f"updated_first_in_{key}" if key == "source" else f"updated_first_in_context_prefix"
            stats[f"{key}_frac_updated_first"] = round(
                stats[first_key] / n_valid, 4
            )
        del stats[f"position_diffs_{key}"]
    
    return stats


def main():
    train_path = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_train.jsonl')
    held_path = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_heldout.jsonl')
    
    if not train_path.exists():
        print(f"ERROR: {train_path} not found", file=sys.stderr)
        sys.exit(1)
    
    train_rows = load_jsonl(train_path)
    held_rows = load_jsonl(held_path)
    
    print(f"Loaded {len(train_rows)} train rows, {len(held_rows)} held rows", flush=True)
    
    train_stats = analyze_pairs(train_rows, "train")
    held_stats = analyze_pairs(held_rows, "heldout")
    
    # Determine risk level
    train_frac = train_stats.get("source_frac_updated_first", 0.5)
    risk = "HIGH" if (train_frac > 0.65 or train_frac < 0.35) else "LOW"
    
    out_dir = _public_path('experiments/archive/relation_learning/data/entity_position_audit')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "status": "ENTITY_POSITION_AUDIT",
        "train": train_stats,
        "heldout": held_stats,
        "shortcut_risk": risk,
    }
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    md_lines = [
        "# research: Entity position/order shortcut audit",
        "",
        "## Purpose",
        "Check whether the updated entity systematically appears before/after the",
        "unchanged entity in context. If so, position could substitute for entity identity.",
        "",
    ]
    for label, st in [("Train", train_stats), ("Heldout", held_stats)]:
        md_lines += [
            f"## {label} ({st['total_pairs']} pairs)",
            "",
            "### Source sentence",
            f"- Updated entity first: {st['updated_first_in_source']}",
            f"- Unchanged entity first: {st['unchanged_first_in_source']}",
            f"- Same position: {st['same_position_in_source']}",
            f"- Missing: {st['one_or_both_missing_in_source']}",
            f"- Fraction updated first: {st.get('source_frac_updated_first', 'N/A')}",
            f"- Mean position diff (upd-unch): {st.get('source_mean_diff', 'N/A')}",
            "",
            "### Shared context prefix (source + update)",
            f"- Updated entity first: {st['updated_first_in_context_prefix']}",
            f"- Unchanged entity first: {st['unchanged_first_in_context_prefix']}",
            f"- Same: {st['same_in_context_prefix']}",
            f"- Missing: {st['missing_in_context_prefix']}",
            f"- Fraction updated first: {st.get('context_frac_updated_first', 'N/A')}",
            f"- Mean position diff (upd-unch): {st.get('context_mean_diff', 'N/A')}",
            "",
        ]
    md_lines += [f"## Risk: {risk}"]
    
    with open(out_dir / "summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
