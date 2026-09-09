#!/usr/bin/env python3
"""research review-recommended diagnostics.

Tests recommended by independent verification to distinguish stochastic perturbation
from weak-but-systematic learning:

A. Cross-architecture REPEAT-REPEAT correctness correlation (baseline shared difficulty).
   If repeat cells share item-level difficulty, the near-zero delta correlation is meaningful.
   If they don't, the delta decorrelation is trivially expected.

B. Within-trajectory 80M vs 100M delta stability.
   If delta(80M) correlates with delta(100M), the treatment effect is trajectory-stable.
   If not, the effect is also unstable within a single training trajectory.

C. Same-sign vs opposite-sign flip overlap (directional Jaccard).
   Tests whether shared flips are more same-direction than expected.

D. Cross-architecture correctness correlation for COMPACT vs REPEAT cells separately.
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

# All per-target files
CELLS = {
    # 100M
    "full_deberta_compact_100M": WS / "data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json",
    "full_deberta_repeat_100M": WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_100M/eval/per_target/arch_full_repeat_chck_100M.json",
    "cc_nodis_compact_100M": WS / "data/commoncopy_architecture_interaction_selected_panel/nodis_compact/chck_100M/eval/per_target/arch_nodis_compact_chck_100M.json",
    "cc_nodis_repeat_100M": WS / "data/commoncopy_architecture_interaction_selected_panel/nodis_repeat/chck_100M/eval/per_target/arch_nodis_repeat_chck_100M.json",
    "plain_nodis_compact_100M": WS / "data/architecture_interaction_selected_panel_final/nodis_compact/chck_100M/eval/per_target/arch_nodis_compact_chck_100M.json",
    "plain_nodis_repeat_100M": WS / "data/architecture_interaction_selected_panel_final/nodis_repeat/chck_100M/eval/per_target/arch_nodis_repeat_chck_100M.json",
    "roberta_compact_100M": WS / "data/roberta_minimal_selected_eval/compact/chck_100M/eval/per_target/roberta_compact_chck_100M.json",
    "roberta_repeat_100M": WS / "data/roberta_minimal_selected_eval/repeat/chck_100M/eval/per_target/roberta_repeat_chck_100M.json",
    # 80M (DeBERTa only)
    "full_deberta_compact_80M": WS / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
    "full_deberta_repeat_80M": WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_80M/eval/per_target/arch_full_repeat_chck_80M.json",
    "cc_nodis_compact_80M": WS / "data/commoncopy_architecture_interaction_selected_panel/nodis_compact/chck_80M/eval/per_target/arch_nodis_compact_chck_80M.json",
    "cc_nodis_repeat_80M": WS / "data/commoncopy_architecture_interaction_selected_panel/nodis_repeat/chck_80M/eval/per_target/arch_nodis_repeat_chck_80M.json",
    "plain_nodis_compact_80M": WS / "data/architecture_interaction_selected_panel_final/nodis_compact/chck_80M/eval/per_target/arch_nodis_compact_chck_80M.json",
    "plain_nodis_repeat_80M": WS / "data/architecture_interaction_selected_panel_final/nodis_repeat/chck_80M/eval/per_target/arch_nodis_repeat_chck_80M.json",
}

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

def pearson_r(xs, ys):
    n = len(xs)
    if n < 3: return 0.0
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    syy = sum((y-my)**2 for y in ys)
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    d = math.sqrt(sxx * syy)
    return sxy/d if d > 1e-15 else 0.0

def main():
    out_dir = Path(WS / "data/independent_review_recommended_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = load_step218_module()
    cell_items = {}
    for name, path in CELLS.items():
        if path.exists():
            cell_items[name] = load_cell_items(mod, path)
    
    # Common items across all 100M cells
    cells_100M = [n for n in CELLS if "100M" in n and n in cell_items]
    common_ids = sorted(set.intersection(*(set(cell_items[n]) for n in cells_100M)))
    print(f"Common 100M items: {len(common_ids)}", file=sys.stderr, flush=True)

    # ============================================================
    # A: Cross-architecture REPEAT-REPEAT correctness baseline
    # ============================================================
    print("\n=== A: Cross-architecture repeat-repeat correctness correlation ===", file=sys.stderr, flush=True)
    
    repeat_cells = ["full_deberta_repeat_100M", "cc_nodis_repeat_100M", 
                     "plain_nodis_repeat_100M", "roberta_repeat_100M"]
    compact_cells = ["full_deberta_compact_100M", "cc_nodis_compact_100M",
                      "plain_nodis_compact_100M", "roberta_compact_100M"]
    
    correctness_corr_results = []
    for treatment, cell_list, label in [("repeat", repeat_cells, "REPEAT"), 
                                         ("compact", compact_cells, "COMPACT")]:
        for i, c1 in enumerate(cell_list):
            for c2 in cell_list[i+1:]:
                if c1 not in cell_items or c2 not in cell_items:
                    continue
                xs = [1.0 if cell_items[c1][iid]["correct"] else 0.0 for iid in common_ids]
                ys = [1.0 if cell_items[c2][iid]["correct"] else 0.0 for iid in common_ids]
                r = pearson_r(xs, ys)
                a1_name = c1.replace(f"_{treatment}_100M", "")
                a2_name = c2.replace(f"_{treatment}_100M", "")
                rec = {"treatment": treatment, "a1": a1_name, "a2": a2_name, 
                       "pearson_r": r, "n": len(common_ids)}
                correctness_corr_results.append(rec)
                print(f"  {label} {a1_name} vs {a2_name}: r={r:.4f}", 
                      file=sys.stderr, flush=True)

    # ============================================================
    # B: Within-trajectory 80M vs 100M delta stability
    # ============================================================
    print("\n=== B: Within-trajectory 80M vs 100M delta stability ===", file=sys.stderr, flush=True)
    
    trajectory_stability = []
    archs_80 = {
        "full_deberta": ("full_deberta_compact_80M", "full_deberta_repeat_80M",
                          "full_deberta_compact_100M", "full_deberta_repeat_100M"),
        "cc_nodis": ("cc_nodis_compact_80M", "cc_nodis_repeat_80M",
                      "cc_nodis_compact_100M", "cc_nodis_repeat_100M"),
        "plain_nodis": ("plain_nodis_compact_80M", "plain_nodis_repeat_80M",
                         "plain_nodis_compact_100M", "plain_nodis_repeat_100M"),
    }
    
    for arch_name, (c80, r80, c100, r100) in archs_80.items():
        if not all(k in cell_items for k in [c80, r80, c100, r100]):
            continue
        common_80_100 = sorted(set(cell_items[c80]) & set(cell_items[r80]) & 
                               set(cell_items[c100]) & set(cell_items[r100]))
        delta_80 = [(1.0 if cell_items[c80][i]["correct"] else 0.0) - 
                    (1.0 if cell_items[r80][i]["correct"] else 0.0) for i in common_80_100]
        delta_100 = [(1.0 if cell_items[c100][i]["correct"] else 0.0) - 
                     (1.0 if cell_items[r100][i]["correct"] else 0.0) for i in common_80_100]
        r = pearson_r(delta_80, delta_100)
        # Also compute: correctness stability within same treatment
        compact_80_100_r = pearson_r(
            [1.0 if cell_items[c80][i]["correct"] else 0.0 for i in common_80_100],
            [1.0 if cell_items[c100][i]["correct"] else 0.0 for i in common_80_100]
        )
        repeat_80_100_r = pearson_r(
            [1.0 if cell_items[r80][i]["correct"] else 0.0 for i in common_80_100],
            [1.0 if cell_items[r100][i]["correct"] else 0.0 for i in common_80_100]
        )
        rec = {
            "arch": arch_name, "n": len(common_80_100),
            "delta_80_vs_100_r": r,
            "compact_correctness_80_vs_100_r": compact_80_100_r,
            "repeat_correctness_80_vs_100_r": repeat_80_100_r,
        }
        trajectory_stability.append(rec)
        print(f"  {arch_name}: delta_80v100 r={r:.4f}, compact_corr r={compact_80_100_r:.4f}, "
              f"repeat_corr r={repeat_80_100_r:.4f} (n={len(common_80_100)})",
              file=sys.stderr, flush=True)

    # ============================================================
    # C: Same-sign vs opposite-sign flip overlap
    # ============================================================
    print("\n=== C: Directional flip overlap ===", file=sys.stderr, flush=True)

    arch_pairs = [("full_deberta", "cc_nodis"), ("full_deberta", "roberta"),
                  ("cc_nodis", "plain_nodis"), ("cc_nodis", "roberta")]
    arch_deltas = {}
    for arch_prefix in ["full_deberta", "cc_nodis", "plain_nodis", "roberta"]:
        ck = f"{arch_prefix}_compact_100M"
        rk = f"{arch_prefix}_repeat_100M"
        if ck in cell_items and rk in cell_items:
            arch_deltas[arch_prefix] = {
                iid: (1.0 if cell_items[ck][iid]["correct"] else 0.0) - 
                     (1.0 if cell_items[rk][iid]["correct"] else 0.0)
                for iid in common_ids
            }
    
    directional_results = []
    for a1, a2 in arch_pairs:
        if a1 not in arch_deltas or a2 not in arch_deltas:
            continue
        d1 = arch_deltas[a1]
        d2 = arch_deltas[a2]
        # Items where both have nonzero deltas
        both_nonzero = [(d1[i], d2[i]) for i in common_ids if d1[i] != 0 and d2[i] != 0]
        if not both_nonzero:
            continue
        same_sign = sum(1 for x, y in both_nonzero if (x > 0) == (y > 0))
        opp_sign = len(both_nonzero) - same_sign
        # Specifically: (both +1) vs (one +1, one -1) vs (both -1)
        both_pos = sum(1 for x, y in both_nonzero if x > 0 and y > 0)
        both_neg = sum(1 for x, y in both_nonzero if x < 0 and y < 0)
        pos_neg = sum(1 for x, y in both_nonzero if x > 0 and y < 0)
        neg_pos = sum(1 for x, y in both_nonzero if x < 0 and y > 0)
        # Expected under independence given marginal rates
        n = len(common_ids)
        p1_pos = sum(1 for i in common_ids if d1[i] > 0) / n
        p1_neg = sum(1 for i in common_ids if d1[i] < 0) / n
        p2_pos = sum(1 for i in common_ids if d2[i] > 0) / n
        p2_neg = sum(1 for i in common_ids if d2[i] < 0) / n
        expected_both_pos = p1_pos * p2_pos * n
        expected_both_neg = p1_neg * p2_neg * n
        expected_pos_neg = p1_pos * p2_neg * n
        expected_neg_pos = p1_neg * p2_pos * n
        
        rec = {
            "pair": f"{a1}_vs_{a2}",
            "n_both_nonzero": len(both_nonzero),
            "same_sign": same_sign, "opp_sign": opp_sign,
            "same_sign_rate": same_sign / len(both_nonzero),
            "both_pos": both_pos, "expected_both_pos": expected_both_pos,
            "both_neg": both_neg, "expected_both_neg": expected_both_neg,
            "pos_neg": pos_neg, "expected_pos_neg": expected_pos_neg,
            "neg_pos": neg_pos, "expected_neg_pos": expected_neg_pos,
            "same_sign_enrichment": (both_pos + both_neg) / (expected_both_pos + expected_both_neg) if (expected_both_pos + expected_both_neg) > 0 else None,
        }
        directional_results.append(rec)
        print(f"  {a1} vs {a2}: same_sign={same_sign}/{len(both_nonzero)} ({rec['same_sign_rate']:.3f}), "
              f"enrichment={rec['same_sign_enrichment']:.3f}x",
              file=sys.stderr, flush=True)

    # ============================================================
    # Save
    # ============================================================
    results = {
        "status": "independent_review_RECOMMENDED_DIAGNOSTICS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_common_100M": len(common_ids),
        "A_correctness_correlations": correctness_corr_results,
        "B_trajectory_stability": trajectory_stability,
        "C_directional_overlap": directional_results,
        "boundary": "CPU-only; no model loading, training, evaluation, SuperGLUE, AoA, upload, or submission."
    }
    (out_dir / "independent_review_recommended_diagnostics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    
    print(json.dumps({
        "status": "independent_review_RECOMMENDED_DIAGNOSTICS",
        "out": str(out_dir / "independent_review_recommended_diagnostics.json"),
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
