#!/usr/bin/env python3
"""research extension: within-architecture difficulty and per-column mean deltas.

Tests whether the universal difficulty redistribution:
  (a) holds within each evaluation column (not just cross-column)
  (b) persists when difficulty is defined within each architecture
  (c) produces column-specific mean deltas consistent with score-column analysis
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, sys, math, time, importlib, importlib.util
from pathlib import Path
from collections import defaultdict
from typing import Any

def find_user_root() -> Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
MOVEMENT_READER = WS / "scripts/selected_prediction_movement_reader.py"

CELLS_100M = {
    "full_deberta_compact": WS / "data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json",
    "full_deberta_repeat": WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_100M/eval/per_target/arch_full_repeat_chck_100M.json",
    "cc_nodis_compact": WS / "data/commoncopy_architecture_interaction_selected_panel/nodis_compact/chck_100M/eval/per_target/arch_nodis_compact_chck_100M.json",
    "cc_nodis_repeat": WS / "data/commoncopy_architecture_interaction_selected_panel/nodis_repeat/chck_100M/eval/per_target/arch_nodis_repeat_chck_100M.json",
    "plain_nodis_compact": WS / "data/architecture_interaction_selected_panel_final/nodis_compact/chck_100M/eval/per_target/arch_nodis_compact_chck_100M.json",
    "plain_nodis_repeat": WS / "data/architecture_interaction_selected_panel_final/nodis_repeat/chck_100M/eval/per_target/arch_nodis_repeat_chck_100M.json",
    "roberta_compact": WS / "data/roberta_minimal_selected_eval/compact/chck_100M/eval/per_target/roberta_compact_chck_100M.json",
    "roberta_repeat": WS / "data/roberta_minimal_selected_eval/repeat/chck_100M/eval/per_target/roberta_repeat_chck_100M.json",
}

ARCHITECTURES = {
    "full_deberta": ("full_deberta_compact", "full_deberta_repeat"),
    "cc_nodis": ("cc_nodis_compact", "cc_nodis_repeat"),
    "plain_nodis": ("plain_nodis_compact", "plain_nodis_repeat"),
    "roberta": ("roberta_compact", "roberta_repeat"),
}

STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def resolve(x):
    if x is None or str(x) == "": return None
    p = Path(str(x))
    return ROOT / p if not p.is_absolute() else p

def load_step218_module():
    spec = importlib.util.spec_from_file_location("selected_prediction_movement_reader", MOVEMENT_READER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def load_cell_items(mod, pt_path):
    record = mod.load_eval_record(pt_path)
    items, meta = mod.load_items_from_record(record)
    return {r["item_id"]: r for r in items}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(WS / "data/cross_architecture_item_discrimination"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = load_step218_module()
    cell_items = {}
    for name, path in CELLS_100M.items():
        if path.exists():
            cell_items[name] = load_cell_items(mod, path)

    # Common items across all 100M cells
    avail = [n for n in CELLS_100M if n in cell_items]
    common_ids = sorted(set.intersection(*(set(cell_items[n]) for n in avail)))
    print(f"Common 100M items: {len(common_ids)}", file=sys.stderr, flush=True)

    # ============================================================
    # A: Per-column mean deltas for each architecture
    # ============================================================
    print("\n=== A: Per-column mean deltas (compact-minus-repeat) ===", file=sys.stderr, flush=True)

    column_deltas = defaultdict(lambda: defaultdict(list))  # arch -> col -> [deltas]
    for item_id in common_ids:
        # Get column from first available cell
        col = cell_items[avail[0]][item_id].get("column", "unknown")
        for arch_name, (compact_key, repeat_key) in ARCHITECTURES.items():
            if compact_key in cell_items and repeat_key in cell_items:
                ci = cell_items[compact_key].get(item_id)
                ri = cell_items[repeat_key].get(item_id)
                if ci is not None and ri is not None:
                    d = (1.0 if ci["correct"] else 0.0) - (1.0 if ri["correct"] else 0.0)
                    column_deltas[arch_name][col].append(d)

    col_delta_summary = {}
    for arch_name in ARCHITECTURES:
        row = {}
        for col in STABLE_COLUMNS:
            vals = column_deltas[arch_name].get(col, [])
            if vals:
                row[col] = {"mean": sum(vals)/len(vals), "n": len(vals),
                           "n_pos": sum(1 for v in vals if v>0),
                           "n_neg": sum(1 for v in vals if v<0),
                           "n_zero": sum(1 for v in vals if v==0)}
        col_delta_summary[arch_name] = row
        parts = [f"{col}={row[col]['mean']:.4f}" for col in STABLE_COLUMNS if col in row]
        print(f"  {arch_name}: {', '.join(parts)}", file=sys.stderr, flush=True)

    # ============================================================
    # B: Within-architecture difficulty terciles
    # ============================================================
    print("\n=== B: Within-architecture difficulty terciles ===", file=sys.stderr, flush=True)

    within_arch_difficulty = {}
    for arch_name, (compact_key, repeat_key) in ARCHITECTURES.items():
        if repeat_key not in cell_items:
            continue
        # Define difficulty from this architecture's own repeat accuracy
        difficulties = {}
        for item_id in common_ids:
            ri = cell_items[repeat_key].get(item_id)
            if ri is not None:
                difficulties[item_id] = 1.0 if ri["correct"] else 0.0

        # Sort items by difficulty
        sorted_items = sorted(difficulties.items(), key=lambda x: x[1])
        n = len(sorted_items)
        # Tercile boundaries: hard = wrong on repeat (0.0), easy = right on repeat (1.0)
        # For binary accuracy, items are either 0 or 1, so we split differently
        n_wrong = sum(1 for _, v in sorted_items if v == 0.0)
        n_right = n - n_wrong

        # Compute compact delta for wrong-on-repeat vs right-on-repeat items
        compact_deltas_wrong = []
        compact_deltas_right = []
        if compact_key in cell_items:
            for item_id, diff in sorted_items:
                ci = cell_items[compact_key].get(item_id)
                ri = cell_items[repeat_key].get(item_id)
                if ci is not None and ri is not None:
                    delta = (1.0 if ci["correct"] else 0.0) - (1.0 if ri["correct"] else 0.0)
                    if diff == 0.0:
                        compact_deltas_wrong.append(delta)
                    else:
                        compact_deltas_right.append(delta)

        wrong_mean = sum(compact_deltas_wrong)/len(compact_deltas_wrong) if compact_deltas_wrong else 0
        right_mean = sum(compact_deltas_right)/len(compact_deltas_right) if compact_deltas_right else 0

        within_arch_difficulty[arch_name] = {
            "n_wrong_on_repeat": n_wrong,
            "n_right_on_repeat": n_right,
            "delta_wrong_items": wrong_mean,
            "delta_right_items": right_mean,
            "n_wrong_flipped_positive": sum(1 for d in compact_deltas_wrong if d > 0),
            "n_right_flipped_negative": sum(1 for d in compact_deltas_right if d < 0),
        }
        print(f"  {arch_name}: wrong_on_repeat={n_wrong} delta={wrong_mean:.4f}, "
              f"right_on_repeat={n_right} delta={right_mean:.4f}, "
              f"wrong→right={within_arch_difficulty[arch_name]['n_wrong_flipped_positive']}, "
              f"right→wrong={within_arch_difficulty[arch_name]['n_right_flipped_negative']}",
              file=sys.stderr, flush=True)

    # ============================================================
    # C: Within-column within-architecture difficulty
    # ============================================================
    print("\n=== C: Per-column within-architecture difficulty ===", file=sys.stderr, flush=True)

    col_arch_difficulty = {}
    for col in STABLE_COLUMNS:
        col_items_ids = [item_id for item_id in common_ids
                         if cell_items[avail[0]][item_id].get("column") == col]
        col_results = {}
        for arch_name, (compact_key, repeat_key) in ARCHITECTURES.items():
            if compact_key not in cell_items or repeat_key not in cell_items:
                continue
            wrong_deltas = []
            right_deltas = []
            for item_id in col_items_ids:
                ci = cell_items[compact_key].get(item_id)
                ri = cell_items[repeat_key].get(item_id)
                if ci is None or ri is None:
                    continue
                delta = (1.0 if ci["correct"] else 0.0) - (1.0 if ri["correct"] else 0.0)
                if ri["correct"]:
                    right_deltas.append(delta)
                else:
                    wrong_deltas.append(delta)
            col_results[arch_name] = {
                "n_wrong": len(wrong_deltas),
                "n_right": len(right_deltas),
                "delta_wrong": sum(wrong_deltas)/len(wrong_deltas) if wrong_deltas else 0,
                "delta_right": sum(right_deltas)/len(right_deltas) if right_deltas else 0,
            }
        col_arch_difficulty[col] = col_results

    for col in STABLE_COLUMNS:
        print(f"  {col}:", file=sys.stderr, flush=True)
        for arch_name, r in col_arch_difficulty.get(col, {}).items():
            print(f"    {arch_name}: wrong({r['n_wrong']}) δ={r['delta_wrong']:.4f}, "
                  f"right({r['n_right']}) δ={r['delta_right']:.4f}",
                  file=sys.stderr, flush=True)

    # ============================================================
    # D: Cross-architecture item overlap analysis
    # ============================================================
    print("\n=== D: Cross-architecture flip overlap ===", file=sys.stderr, flush=True)
    
    # Items flipped positive (wrong→right) and negative (right→wrong) for each architecture
    arch_flips = {}
    for arch_name, (compact_key, repeat_key) in ARCHITECTURES.items():
        if compact_key not in cell_items or repeat_key not in cell_items:
            continue
        pos_flips = set()
        neg_flips = set()
        for item_id in common_ids:
            ci = cell_items[compact_key].get(item_id)
            ri = cell_items[repeat_key].get(item_id)
            if ci is None or ri is None:
                continue
            if not ri["correct"] and ci["correct"]:
                pos_flips.add(item_id)
            elif ri["correct"] and not ci["correct"]:
                neg_flips.add(item_id)
        arch_flips[arch_name] = {"pos": pos_flips, "neg": neg_flips}
        print(f"  {arch_name}: {len(pos_flips)} pos flips, {len(neg_flips)} neg flips, "
              f"net={len(pos_flips)-len(neg_flips)}", file=sys.stderr, flush=True)

    # Jaccard similarity of flip sets
    flip_overlap = []
    for a1 in ARCHITECTURES:
        for a2 in ARCHITECTURES:
            if a1 >= a2 or a1 not in arch_flips or a2 not in arch_flips:
                continue
            for flip_type in ["pos", "neg"]:
                s1 = arch_flips[a1][flip_type]
                s2 = arch_flips[a2][flip_type]
                union = len(s1 | s2)
                intersection = len(s1 & s2)
                jaccard = intersection / union if union > 0 else 0
                # Expected Jaccard under independence
                n_total = len(common_ids)
                p1 = len(s1) / n_total
                p2 = len(s2) / n_total
                expected_intersection = p1 * p2 * n_total
                expected_union = (p1 + p2 - p1 * p2) * n_total
                expected_jaccard = expected_intersection / expected_union if expected_union > 0 else 0
                rec = {
                    "pair": f"{a1}_vs_{a2}",
                    "flip_type": flip_type,
                    "size_a1": len(s1),
                    "size_a2": len(s2),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": jaccard,
                    "expected_jaccard_independent": expected_jaccard,
                    "jaccard_ratio": jaccard / expected_jaccard if expected_jaccard > 0 else None,
                }
                flip_overlap.append(rec)
                if flip_type == "pos":
                    print(f"  {a1} ∩ {a2} (pos flips): Jaccard={jaccard:.4f} vs expected={expected_jaccard:.4f} "
                          f"ratio={jaccard/expected_jaccard:.2f}x (|∩|={intersection})",
                          file=sys.stderr, flush=True)

    # ============================================================
    # Save extension results
    # ============================================================
    ext_results = {
        "status": "CROSS_ARCHITECTURE_EXTENSION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "A_per_column_mean_deltas": col_delta_summary,
        "B_within_arch_difficulty": within_arch_difficulty,
        "C_per_column_within_arch_difficulty": col_arch_difficulty,
        "D_flip_overlap": flip_overlap,
        "boundary": "CPU-only extension analysis; no model loading, training, evaluation, "
                     "SuperGLUE, AoA, upload, or leaderboard submission."
    }

    (out_dir / "cross_architecture_extension.json").write_text(
        json.dumps(ext_results, indent=2, default=lambda x: list(x) if isinstance(x, set) else str(x)),
        encoding="utf-8"
    )

    # Summary
    print(json.dumps({
        "status": "CROSS_ARCHITECTURE_EXTENSION",
        "per_column_deltas": {arch: {col: f"{v['mean']:.4f}" for col, v in cols.items()} 
                             for arch, cols in col_delta_summary.items()},
        "within_arch_net_flips": {arch: v["n_wrong_flipped_positive"] - v["n_right_flipped_negative"]
                                  for arch, v in within_arch_difficulty.items()},
        "boundary": "CPU-only extension analysis."
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
