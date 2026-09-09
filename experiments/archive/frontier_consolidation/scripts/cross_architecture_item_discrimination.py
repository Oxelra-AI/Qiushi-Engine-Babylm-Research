#!/usr/bin/env python3
"""research: Cross-architecture item-level discrimination analysis.

Uses EXISTING per-target evaluation data from all available architecture×data cells
(full DeBERTa, common-copy no-disentangle DeBERTa, plain shifted-init no-disentangle
DeBERTa, stock RoBERTa) to compute item-level compact-minus-repeat correlations.

This is a PURE CPU analysis on saved evaluation artifacts — no model loading, no GPU,
no new training or evaluation.

Discriminating predictions of candidate explanations:
  H1 (relative position encoding): full & common-copy nodis HIGH corr; plain nodis ZERO;
     RoBERTa ZERO.  Key: plain nodis -> zero (basin prevents mechanism entirely)
  H3 (optimization basin): full & common-copy nodis HIGH corr; plain nodis MODERATE;
     RoBERTa LOW.  Key: plain nodis -> moderate (same architecture, different basin,
     partially overlapping benefit pattern)
  Distinguishing observation: corr(full_DeBERTa, plain_nodis) moderate vs near-zero
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, sys, math, time, importlib, importlib.util, hashlib
from pathlib import Path
from collections import defaultdict
from typing import Any

def find_user_root() -> Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
MOVEMENT_READER = WS / "scripts/selected_prediction_movement_reader.py"

# ============================================================
# Per-target file locations for all architecture × data cells
# ============================================================
# Using 100M checkpoint for all comparisons (common endpoint)
CELLS = {
    # Full DeBERTa (p2c + c2p, 34,467,424 params)
    "full_deberta_compact_100M": WS / "data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json",
    "full_deberta_repeat_100M": WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_100M/eval/per_target/arch_full_repeat_chck_100M.json",
    # Common-copy no-disentangle DeBERTa (pos_att_type=[], 30,773,344 params, matched init)
    "cc_nodis_compact_100M": WS / "data/commoncopy_architecture_interaction_selected_panel/nodis_compact/chck_100M/eval/per_target/arch_nodis_compact_chck_100M.json",
    "cc_nodis_repeat_100M": WS / "data/commoncopy_architecture_interaction_selected_panel/nodis_repeat/chck_100M/eval/per_target/arch_nodis_repeat_chck_100M.json",
    # Plain no-disentangle DeBERTa (pos_att_type=[], 30,773,344 params, SHIFTED init)
    "plain_nodis_compact_100M": WS / "data/architecture_interaction_selected_panel_final/nodis_compact/chck_100M/eval/per_target/arch_nodis_compact_chck_100M.json",
    "plain_nodis_repeat_100M": WS / "data/architecture_interaction_selected_panel_final/nodis_repeat/chck_100M/eval/per_target/arch_nodis_repeat_chck_100M.json",
    # Stock RoBERTa (absolute position, ~30,530,000 params)
    "roberta_compact_100M": WS / "data/roberta_minimal_selected_eval/compact/chck_100M/eval/per_target/roberta_compact_chck_100M.json",
    "roberta_repeat_100M": WS / "data/roberta_minimal_selected_eval/repeat/chck_100M/eval/per_target/roberta_repeat_chck_100M.json",
}

# Also at 80M for DeBERTa cells
CELLS_80M = {
    "full_deberta_compact_80M": WS / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
    "full_deberta_repeat_80M": WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_80M/eval/per_target/arch_full_repeat_chck_80M.json",
    "cc_nodis_compact_80M": WS / "data/commoncopy_architecture_interaction_selected_panel/nodis_compact/chck_80M/eval/per_target/arch_nodis_compact_chck_80M.json",
    "cc_nodis_repeat_80M": WS / "data/commoncopy_architecture_interaction_selected_panel/nodis_repeat/chck_80M/eval/per_target/arch_nodis_repeat_chck_80M.json",
    "plain_nodis_compact_80M": WS / "data/architecture_interaction_selected_panel_final/nodis_compact/chck_80M/eval/per_target/arch_nodis_compact_chck_80M.json",
    "plain_nodis_repeat_80M": WS / "data/architecture_interaction_selected_panel_final/nodis_repeat/chck_80M/eval/per_target/arch_nodis_repeat_chck_80M.json",
}

STABLE_COLUMNS = {"BLiMP", "Supplement", "EWoK", "Entity", "COMPS"}
ALL_COLUMNS = {"BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"}

ARCHITECTURES_100M = {
    "full_deberta": ("full_deberta_compact_100M", "full_deberta_repeat_100M"),
    "cc_nodis": ("cc_nodis_compact_100M", "cc_nodis_repeat_100M"),
    "plain_nodis": ("plain_nodis_compact_100M", "plain_nodis_repeat_100M"),
    "roberta": ("roberta_compact_100M", "roberta_repeat_100M"),
}

ARCHITECTURES_80M = {
    "full_deberta": ("full_deberta_compact_80M", "full_deberta_repeat_80M"),
    "cc_nodis": ("cc_nodis_compact_80M", "cc_nodis_repeat_80M"),
    "plain_nodis": ("plain_nodis_compact_80M", "plain_nodis_repeat_80M"),
}

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def resolve(x: str | Path | None) -> Path | None:
    if x is None or str(x) == "":
        return None
    p = Path(str(x))
    if p.is_absolute():
        return p
    return ROOT / p

def load_step218_module():
    spec = importlib.util.spec_from_file_location(
        "selected_prediction_movement_reader", MOVEMENT_READER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {MOVEMENT_READER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def load_cell_items(mod, pt_path: Path) -> dict[str, dict[str, Any]]:
    """Load per-target items into {item_id: {item_fields}}."""
    record = mod.load_eval_record(pt_path)
    items, meta = mod.load_items_from_record(record)
    return {r["item_id"]: r for r in items}

def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation. Return 0 if degenerate."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sxx * syy)
    if denom < 1e-15:
        return 0.0
    return sxy / denom

def bootstrap_correlation(xs: list[float], ys: list[float], n_boot: int = 400, seed: int = 42) -> dict:
    """Bootstrap confidence interval for Pearson r."""
    import random
    rng = random.Random(seed)
    n = len(xs)
    point = pearson_r(xs, ys)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idx]
        by = [ys[i] for i in idx]
        boots.append(pearson_r(bx, by))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return {"r": point, "ci_025": lo, "ci_975": hi, "n": n}

def agreement_rate(xs: list[float], ys: list[float]) -> dict:
    """Fraction of items where both architectures agree on sign of delta."""
    n = len(xs)
    if n == 0:
        return {"n": 0, "agree": 0.0, "n_nonzero_both": 0}
    both_nonzero = [(x, y) for x, y in zip(xs, ys) if x != 0 and y != 0]
    n_bn = len(both_nonzero)
    if n_bn == 0:
        return {"n": n, "agree": 0.0, "n_nonzero_both": 0}
    agree = sum(1 for x, y in both_nonzero if (x > 0) == (y > 0))
    return {"n": n, "agree": agree / n_bn, "n_nonzero_both": n_bn, "n_agree": agree}

def mean_delta(deltas: list[float]) -> float:
    return sum(deltas) / len(deltas) if deltas else 0.0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(WS / "data/cross_architecture_item_discrimination"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify all per-target files exist
    all_cells = {**CELLS, **CELLS_80M}
    file_status = {}
    for name, path in all_cells.items():
        file_status[name] = {"path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}
    
    missing = [n for n, s in file_status.items() if not s["exists"]]
    print(json.dumps({
        "status": "CROSS_ARCHITECTURE_ITEM_DISCRIMINATION",
        "phase": "file_check",
        "n_cells": len(all_cells),
        "n_present": len(all_cells) - len(missing),
        "n_missing": len(missing),
        "missing": missing,
    }, indent=2), flush=True)

    if missing:
        print(f"WARNING: {len(missing)} per-target files missing: {missing}", file=sys.stderr)

    if args.plan_only:
        (out_dir / "plan.json").write_text(json.dumps({
            "file_status": file_status,
            "architectures_100M": {k: [str(CELLS[c]) for c in v] for k, v in ARCHITECTURES_100M.items()},
        }, indent=2), encoding="utf-8")
        print(json.dumps({"plan": str(out_dir / "plan.json")}))
        return

    # Load items for all 100M cells
    mod = load_step218_module()
    print("Loading items from per-target files...", file=sys.stderr, flush=True)

    cell_items: dict[str, dict[str, dict]] = {}
    cell_counts: dict[str, int] = {}
    for name, path in CELLS.items():
        if not path.exists():
            continue
        cell_items[name] = load_cell_items(mod, path)
        cell_counts[name] = len(cell_items[name])
        print(f"  {name}: {cell_counts[name]} items", file=sys.stderr, flush=True)

    # Also load 80M DeBERTa cells
    for name, path in CELLS_80M.items():
        if not path.exists():
            continue
        cell_items[name] = load_cell_items(mod, path)
        cell_counts[name] = len(cell_items[name])
        print(f"  {name}: {cell_counts[name]} items", file=sys.stderr, flush=True)

    # ============================================================
    # Analysis 1: Cross-architecture item-level delta correlations at 100M
    # ============================================================
    print("\n--- Analysis 1: Cross-architecture delta correlations at 100M ---", file=sys.stderr, flush=True)

    # Find common items across ALL 100M cells
    avail_100M = [name for name in CELLS if name in cell_items]
    common_100M_ids = sorted(set.intersection(*(set(cell_items[n]) for n in avail_100M)))
    print(f"  Common items across {len(avail_100M)} 100M cells: {len(common_100M_ids)}", file=sys.stderr, flush=True)

    # Compute per-architecture compact-minus-repeat delta for each item
    arch_deltas_100M: dict[str, dict[str, float]] = {}  # arch -> {item_id: delta}
    for arch_name, (compact_key, repeat_key) in ARCHITECTURES_100M.items():
        if compact_key not in cell_items or repeat_key not in cell_items:
            print(f"  SKIP {arch_name}: missing cells", file=sys.stderr, flush=True)
            continue
        deltas = {}
        for item_id in common_100M_ids:
            ci = cell_items[compact_key].get(item_id)
            ri = cell_items[repeat_key].get(item_id)
            if ci is None or ri is None:
                continue
            deltas[item_id] = (1.0 if ci["correct"] else 0.0) - (1.0 if ri["correct"] else 0.0)
        arch_deltas_100M[arch_name] = deltas

    # Compute pairwise correlations
    arch_pairs = [
        ("full_deberta", "cc_nodis", "Same architecture, matched init (should be HIGH)"),
        ("full_deberta", "plain_nodis", "KEY DISCRIMINATOR: H1 predicts ~0, H3 predicts moderate"),
        ("full_deberta", "roberta", "Cross-architecture (should be LOW)"),
        ("cc_nodis", "plain_nodis", "Same no-disentangle arch, different init"),
        ("cc_nodis", "roberta", "Reduced DeBERTa vs RoBERTa"),
        ("plain_nodis", "roberta", "Both 'failed' architectures (shared failure pattern?)"),
    ]

    correlation_results = []
    for a1, a2, description in arch_pairs:
        if a1 not in arch_deltas_100M or a2 not in arch_deltas_100M:
            continue
        d1 = arch_deltas_100M[a1]
        d2 = arch_deltas_100M[a2]
        common = sorted(set(d1) & set(d2))
        xs = [d1[i] for i in common]
        ys = [d2[i] for i in common]
        
        r_boot = bootstrap_correlation(xs, ys, n_boot=400, seed=hash(f"{a1}_{a2}") & 0xFFFFFFFF)
        agree = agreement_rate(xs, ys)
        
        rec = {
            "pair": f"{a1}_vs_{a2}",
            "description": description,
            "n_common": len(common),
            "pearson_r": r_boot["r"],
            "ci_025": r_boot["ci_025"],
            "ci_975": r_boot["ci_975"],
            "agreement_rate": agree["agree"],
            "n_nonzero_both": agree["n_nonzero_both"],
            "mean_delta_a1": mean_delta(xs),
            "mean_delta_a2": mean_delta(ys),
        }
        correlation_results.append(rec)
        print(f"  {a1} vs {a2}: r={rec['pearson_r']:.4f} [{rec['ci_025']:.4f}, {rec['ci_975']:.4f}] "
              f"agree={rec['agreement_rate']:.3f} mean_delta=({rec['mean_delta_a1']:.4f}, {rec['mean_delta_a2']:.4f})",
              file=sys.stderr, flush=True)

    # ============================================================
    # Analysis 2: Per-column stratification at 100M
    # ============================================================
    print("\n--- Analysis 2: Per-column stratified correlations at 100M ---", file=sys.stderr, flush=True)

    column_corr_results = {}
    for col in sorted(STABLE_COLUMNS):
        col_items = [item_id for item_id in common_100M_ids
                     if any(cell_items[n][item_id].get("column") == col 
                            for n in avail_100M if item_id in cell_items[n])]
        if not col_items:
            continue
        col_results = []
        for a1, a2, desc in arch_pairs:
            if a1 not in arch_deltas_100M or a2 not in arch_deltas_100M:
                continue
            d1 = arch_deltas_100M[a1]
            d2 = arch_deltas_100M[a2]
            common = [i for i in col_items if i in d1 and i in d2]
            if len(common) < 10:
                continue
            xs = [d1[i] for i in common]
            ys = [d2[i] for i in common]
            r = pearson_r(xs, ys)
            agree = agreement_rate(xs, ys)
            col_results.append({
                "pair": f"{a1}_vs_{a2}",
                "column": col,
                "n": len(common),
                "pearson_r": r,
                "agreement_rate": agree["agree"],
                "mean_delta_a1": mean_delta(xs),
                "mean_delta_a2": mean_delta(ys),
            })
        column_corr_results[col] = col_results
        # Print the key discriminator for each column
        for cr in col_results:
            if "full_deberta_vs_plain_nodis" in cr["pair"]:
                print(f"  {col}: full_deberta vs plain_nodis r={cr['pearson_r']:.4f} (n={cr['n']})",
                      file=sys.stderr, flush=True)

    # ============================================================
    # Analysis 3: Difficulty-stratified analysis at 100M
    # ============================================================
    print("\n--- Analysis 3: Difficulty-stratified compact benefit ---", file=sys.stderr, flush=True)

    # Use repeat accuracy as difficulty proxy (average across architectures)
    item_difficulty = {}
    for item_id in common_100M_ids:
        accs = []
        for arch_name, (_, repeat_key) in ARCHITECTURES_100M.items():
            if repeat_key in cell_items:
                ri = cell_items[repeat_key].get(item_id)
                if ri is not None:
                    accs.append(1.0 if ri["correct"] else 0.0)
        if accs:
            item_difficulty[item_id] = sum(accs) / len(accs)

    # Split items into difficulty terciles
    diff_values = sorted(item_difficulty.values())
    n_items = len(diff_values)
    tercile_bounds = [diff_values[n_items // 3], diff_values[2 * n_items // 3]]
    
    difficulty_results = {}
    for arch_name in arch_deltas_100M:
        d = arch_deltas_100M[arch_name]
        for label, lo, hi in [("hard", -0.01, tercile_bounds[0] + 0.01),
                                ("medium", tercile_bounds[0] - 0.01, tercile_bounds[1] + 0.01),
                                ("easy", tercile_bounds[1] - 0.01, 1.01)]:
            items_in = [i for i in d if i in item_difficulty and lo <= item_difficulty[i] <= hi]
            vals = [d[i] for i in items_in]
            difficulty_results[f"{arch_name}_{label}"] = {
                "arch": arch_name, "difficulty": label, "n_items": len(items_in),
                "mean_delta": mean_delta(vals) if vals else None,
                "n_positive": sum(1 for v in vals if v > 0),
                "n_negative": sum(1 for v in vals if v < 0),
            }

    print("  Mean compact-minus-repeat delta by difficulty tercile:", file=sys.stderr, flush=True)
    for arch_name in arch_deltas_100M:
        parts = []
        for label in ["hard", "medium", "easy"]:
            key = f"{arch_name}_{label}"
            if key in difficulty_results and difficulty_results[key]["mean_delta"] is not None:
                parts.append(f"{label}={difficulty_results[key]['mean_delta']:.4f}")
        print(f"    {arch_name}: {', '.join(parts)}", file=sys.stderr, flush=True)

    # ============================================================  
    # Analysis 4: 80M DeBERTa cells (replication at earlier checkpoint)
    # ============================================================
    print("\n--- Analysis 4: DeBERTa-only 80M correlations ---", file=sys.stderr, flush=True)

    arch_deltas_80M: dict[str, dict[str, float]] = {}
    for arch_name, (compact_key, repeat_key) in ARCHITECTURES_80M.items():
        if compact_key not in cell_items or repeat_key not in cell_items:
            continue
        deltas = {}
        common = sorted(set(cell_items[compact_key]) & set(cell_items[repeat_key]))
        for item_id in common:
            ci = cell_items[compact_key][item_id]
            ri = cell_items[repeat_key][item_id]
            deltas[item_id] = (1.0 if ci["correct"] else 0.0) - (1.0 if ri["correct"] else 0.0)
        arch_deltas_80M[arch_name] = deltas

    corr_80M_results = []
    deberta_pairs_80 = [
        ("full_deberta", "cc_nodis", "Same arch, matched init"),
        ("full_deberta", "plain_nodis", "KEY DISCRIMINATOR"),
        ("cc_nodis", "plain_nodis", "Same nodis arch, diff init"),
    ]
    for a1, a2, desc in deberta_pairs_80:
        if a1 not in arch_deltas_80M or a2 not in arch_deltas_80M:
            continue
        d1 = arch_deltas_80M[a1]
        d2 = arch_deltas_80M[a2]
        common = sorted(set(d1) & set(d2))
        xs = [d1[i] for i in common]
        ys = [d2[i] for i in common]
        r = pearson_r(xs, ys)
        corr_80M_results.append({
            "pair": f"{a1}_vs_{a2}",
            "description": desc,
            "n_common": len(common),
            "pearson_r": r,
            "agreement_rate": agreement_rate(xs, ys)["agree"],
        })
        print(f"  80M {a1} vs {a2}: r={r:.4f} (n={len(common)})", file=sys.stderr, flush=True)

    # ============================================================
    # Analysis 5: Cross-architecture absolute accuracy profile
    # ============================================================
    print("\n--- Analysis 5: Architecture-specific accuracy profiles ---", file=sys.stderr, flush=True)

    # For stable columns, compute per-architecture mean accuracy for compact and repeat
    accuracy_profiles = {}
    for arch_name, (compact_key, repeat_key) in ARCHITECTURES_100M.items():
        if compact_key not in cell_items or repeat_key not in cell_items:
            continue
        compact_correct = 0
        repeat_correct = 0
        n_stable = 0
        for item_id in common_100M_ids:
            ci = cell_items[compact_key].get(item_id)
            ri = cell_items[repeat_key].get(item_id)
            if ci is None or ri is None:
                continue
            col = ci.get("column", "")
            if col not in STABLE_COLUMNS:
                continue
            compact_correct += 1 if ci["correct"] else 0
            repeat_correct += 1 if ri["correct"] else 0
            n_stable += 1
        if n_stable > 0:
            accuracy_profiles[arch_name] = {
                "n_stable": n_stable,
                "compact_acc": compact_correct / n_stable,
                "repeat_acc": repeat_correct / n_stable,
                "delta": (compact_correct - repeat_correct) / n_stable,
            }
            print(f"  {arch_name}: compact={accuracy_profiles[arch_name]['compact_acc']:.4f}, "
                  f"repeat={accuracy_profiles[arch_name]['repeat_acc']:.4f}, "
                  f"delta={accuracy_profiles[arch_name]['delta']:.4f} (n={n_stable})",
                  file=sys.stderr, flush=True)

    # ============================================================
    # Save all results
    # ============================================================
    results = {
        "status": "CROSS_ARCHITECTURE_ITEM_DISCRIMINATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "common_100M_items": len(common_100M_ids),
        "cell_counts": cell_counts,
        "analysis_1_correlations_100M": correlation_results,
        "analysis_2_per_column": column_corr_results,
        "analysis_3_difficulty_stratified": difficulty_results,
        "analysis_4_correlations_80M": corr_80M_results,
        "analysis_5_accuracy_profiles": accuracy_profiles,
        "candidate_explanations": {
            "H1_relative_position": {
                "predicts": "plain_nodis vs full_deberta r near ZERO (mechanism entirely absent)",
                "key_test": "corr(full_deberta, plain_nodis) at 100M"
            },
            "H3_optimization_basin": {
                "predicts": "plain_nodis vs full_deberta r MODERATE positive (same arch, degraded basin)",
                "key_test": "corr(full_deberta, plain_nodis) at 100M"  
            },
            "discriminating_threshold": "r < 0.02 favors H1; r > 0.05 favors H3; intermediate ambiguous"
        },
        "boundary": "CPU-only analysis on saved evaluation artifacts; no model loading, training, evaluation, "
                     "SuperGLUE, AoA, upload, or leaderboard submission."
    }

    (out_dir / "cross_architecture_item_discrimination.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    # Write human-readable summary
    lines = ["# research: Cross-Architecture Item-Level Discrimination Analysis\n"]
    lines.append(f"Created: {results['created_utc']}\n")
    lines.append(f"Common 100M items: {results['common_100M_items']}\n")
    
    lines.append("\n## Analysis 1: Cross-architecture delta correlations at 100M\n")
    lines.append("| Pair | r | CI 95% | Agreement | Mean Δ₁ | Mean Δ₂ | n |\n")
    lines.append("|------|---|--------|-----------|---------|---------|---|\n")
    for rec in correlation_results:
        lines.append(f"| {rec['pair']} | {rec['pearson_r']:.4f} | [{rec['ci_025']:.4f}, {rec['ci_975']:.4f}] | "
                     f"{rec['agreement_rate']:.3f} | {rec['mean_delta_a1']:.4f} | {rec['mean_delta_a2']:.4f} | {rec['n_common']} |\n")

    lines.append("\n## Analysis 2: Key per-column correlations (full_deberta vs plain_nodis)\n")
    for col, recs in column_corr_results.items():
        for cr in recs:
            if "full_deberta_vs_plain_nodis" in cr["pair"]:
                lines.append(f"- {col}: r={cr['pearson_r']:.4f} (n={cr['n']})\n")

    lines.append("\n## Analysis 4: DeBERTa 80M correlations\n")
    for rec in corr_80M_results:
        lines.append(f"- {rec['pair']}: r={rec['pearson_r']:.4f} (n={rec['n_common']})\n")

    lines.append("\n## Analysis 5: Stable-column accuracy profiles\n")
    for arch, prof in accuracy_profiles.items():
        lines.append(f"- {arch}: compact={prof['compact_acc']:.4f}, repeat={prof['repeat_acc']:.4f}, delta={prof['delta']:.4f}\n")

    lines.append("\n## Interpretation\n")
    lines.append("See candidate_explanations in JSON for hypothesis comparison.\n")
    lines.append("\n## Boundary\n")
    lines.append("CPU-only analysis on saved evaluation artifacts; no model loading, training, evaluation, SuperGLUE, AoA, upload, or leaderboard submission.\n")

    (out_dir / "cross_architecture_item_discrimination.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "CROSS_ARCHITECTURE_ITEM_DISCRIMINATION",
        "out_dir": str(out_dir),
        "json": str(out_dir / "cross_architecture_item_discrimination.json"),
        "md": str(out_dir / "cross_architecture_item_discrimination.md"),
        "key_result_full_vs_plain_nodis_r": next(
            (r["pearson_r"] for r in correlation_results if "full_deberta_vs_plain_nodis" in r["pair"]), None
        ),
        "key_result_full_vs_cc_nodis_r": next(
            (r["pearson_r"] for r in correlation_results if "full_deberta_vs_cc_nodis" in r["pair"]), None
        ),
        "key_result_full_vs_roberta_r": next(
            (r["pearson_r"] for r in correlation_results if "full_deberta_vs_roberta" in r["pair"]), None
        ),
        "boundary": "CPU-only analysis; no model loading, training, evaluation, SuperGLUE, AoA, upload, or leaderboard submission."
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
