#!/usr/bin/env python3
"""research: Shared-versus-private fitting index analysis.

CPU-only post-processing of per-row MLM losses produced by shared_private_loss.py.
Computes cross-arm correlations, shared/private decomposition, and prediction tests.

No GPU, no training, no official evaluation, no leaderboard, no upload.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from scipy import stats

OUTPUT_DIR = "experiments/archive/relation_learning/data/shared_private_loss"
NOTE_PATH = "research/notes/relation_learning/shared_private_fitting_index.md"

ARMS = ["D_V_43022", "D_V_43122", "D_C_43022", "R_V_43022", "R_C_43022"]
CHECKPOINTS = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
EVAL_SETS = ["heldout", "filler"]

def load_arm_data(arm):
    """Load per-row losses for one arm."""
    npz_path = os.path.join(OUTPUT_DIR, f"{arm}_per_row_losses.npz")
    if not os.path.exists(npz_path):
        return None
    return np.load(npz_path)

def get_losses(data, arm, eset, chck):
    """Extract loss vector for a specific arm/eval_set/checkpoint."""
    key = f"{arm}__{eset}__{chck}__losses"
    if key in data:
        return data[key]
    return None

def get_example_ids(data, arm, eset, chck):
    key = f"{arm}__{eset}__{chck}__example_ids"
    if key in data:
        return data[key]
    return None

def compute_delta(data, arm, eset, chck_early, chck_late):
    """Compute per-row loss change: late - early (negative = improvement)."""
    early = get_losses(data, arm, eset, chck_early)
    late = get_losses(data, arm, eset, chck_late)
    if early is None or late is None:
        return None
    return late - early

def cross_arm_correlation(delta_a, delta_b):
    """Compute Pearson correlation between two delta vectors."""
    valid = ~(np.isnan(delta_a) | np.isnan(delta_b))
    if valid.sum() < 10:
        return {"r": float("nan"), "p": float("nan"), "n": int(valid.sum())}
    r, p = stats.pearsonr(delta_a[valid], delta_b[valid])
    return {"r": float(r), "p": float(p), "n": int(valid.sum())}

def shared_private_decomposition(delta_a, delta_b):
    """Decompose into shared and private components.
    
    shared_i = (delta_a_i + delta_b_i) / 2  (mean across arms)
    private_a_i = delta_a_i - shared_i
    private_b_i = delta_b_i - shared_i
    
    Returns:
      shared_var: variance of shared component
      private_a_var: variance of private component for arm A
      private_b_var: variance of private component for arm B
      shared_fraction: shared_var / total_var
    """
    valid = ~(np.isnan(delta_a) | np.isnan(delta_b))
    da = delta_a[valid]
    db = delta_b[valid]
    
    shared = (da + db) / 2.0
    priv_a = da - shared
    priv_b = db - shared
    
    shared_var = np.var(shared)
    priv_a_var = np.var(priv_a)
    priv_b_var = np.var(priv_b)
    total_var = (np.var(da) + np.var(db)) / 2.0
    
    shared_frac = shared_var / total_var if total_var > 0 else float("nan")
    
    return {
        "shared_var": float(shared_var),
        "private_a_var": float(priv_a_var),
        "private_b_var": float(priv_b_var),
        "total_var_mean": float(total_var),
        "shared_fraction": float(shared_frac),
        "n": int(valid.sum()),
    }

def source_stratified_correlation(delta_a, delta_b, example_ids, rows_by_id):
    """Compute correlations stratified by source."""
    by_source = {}
    for i, eid in enumerate(example_ids):
        src = rows_by_id.get(int(eid), {}).get("source", "unknown")
        if src not in by_source:
            by_source[src] = {"a": [], "b": []}
        if not (np.isnan(delta_a[i]) or np.isnan(delta_b[i])):
            by_source[src]["a"].append(delta_a[i])
            by_source[src]["b"].append(delta_b[i])
    
    results = {}
    for src, vecs in by_source.items():
        if len(vecs["a"]) < 10:
            results[src] = {"r": float("nan"), "p": float("nan"), "n": len(vecs["a"])}
        else:
            a = np.array(vecs["a"])
            b = np.array(vecs["b"])
            r, p = stats.pearsonr(a, b)
            results[src] = {"r": float(r), "p": float(p), "n": len(a)}
    return results

def main():
    # Load all arm data
    arm_data = {}
    for arm in ARMS:
        d = load_arm_data(arm)
        if d is not None:
            arm_data[arm] = d
            print(f"Loaded {arm}: {len(d.files)} arrays", flush=True)
        else:
            print(f"WARNING: {arm} not found, skipping", flush=True)
    
    if len(arm_data) < 2:
        print("ERROR: need at least 2 arms for comparison", flush=True)
        sys.exit(1)
    
    # Load heldout rows for source stratification
    import json as json_mod
    heldout_path = "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl"
    rows_by_id = {}
    if os.path.exists(heldout_path):
        with open(heldout_path) as f:
            for line in f:
                obj = json_mod.loads(line)
                rows_by_id[obj["example_id"]] = {"source": obj["source"], "words": obj["words"]}
    
    analysis = {"comparisons": {}, "decompositions": {}, "source_stratified": {}}
    
    # ── Key comparisons ──────────────────────────────────────────────────
    
    # Define comparison pairs
    comparison_pairs = [
        # Cross-seed (same architecture, same data, different trajectory)
        ("D_V_43022", "D_V_43122", "cross_seed_DeBERTa"),
        # Cross-architecture (same data, same seed, different architecture)
        ("D_V_43022", "R_V_43022", "cross_arch_view"),
        # View vs Clean within DeBERTa seed43022
        ("D_V_43022", "D_C_43022", "view_vs_clean_DeBERTa"),
        # View vs Clean within RoBERTa
        ("R_V_43022", "R_C_43022", "view_vs_clean_RoBERTa"),
    ]
    
    # Delta windows
    delta_windows = [
        ("chck_60M", "chck_80M", "60to80"),
        ("chck_60M", "chck_100M", "60to100"),
        ("chck_80M", "chck_100M", "80to100"),
    ]
    
    for arm_a, arm_b, comp_name in comparison_pairs:
        if arm_a not in arm_data or arm_b not in arm_data:
            print(f"Skipping {comp_name}: missing arm(s)", flush=True)
            continue
        
        for eset in EVAL_SETS:
            for chck_early, chck_late, win_name in delta_windows:
                delta_a = compute_delta(arm_data[arm_a], arm_a, eset, chck_early, chck_late)
                delta_b = compute_delta(arm_data[arm_b], arm_b, eset, chck_early, chck_late)
                
                if delta_a is None or delta_b is None:
                    continue
                
                key = f"{comp_name}__{eset}__{win_name}"
                
                corr = cross_arm_correlation(delta_a, delta_b)
                decomp = shared_private_decomposition(delta_a, delta_b)
                
                analysis["comparisons"][key] = corr
                analysis["decompositions"][key] = decomp
                
                print(f"{key}: r={corr['r']:.4f}, p={corr['p']:.2e}, "
                      f"shared_frac={decomp['shared_fraction']:.4f}, n={corr['n']}", flush=True)
                
                # Source-stratified (heldout only)
                if eset == "heldout":
                    eids_a = get_example_ids(arm_data[arm_a], arm_a, eset, chck_early)
                    if eids_a is not None:
                        strat = source_stratified_correlation(delta_a, delta_b, eids_a, rows_by_id)
                        analysis["source_stratified"][key] = strat
    
    # ── Mean loss trajectories ───────────────────────────────────────────
    
    trajectories = {}
    for arm in arm_data:
        for eset in EVAL_SETS:
            traj = {}
            for chck in CHECKPOINTS:
                losses = get_losses(arm_data[arm], arm, eset, chck)
                if losses is not None:
                    traj[chck] = float(np.nanmean(losses))
            if traj:
                trajectories[f"{arm}__{eset}"] = traj
    
    analysis["trajectories"] = trajectories
    
    # ── Save results ─────────────────────────────────────────────────────
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    json_path = os.path.join(OUTPUT_DIR, "shared_private_analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis: {json_path}", flush=True)
    
    # ── Generate note ────────────────────────────────────────────────────
    
    lines = ["# research: Shared-versus-private fitting index\n"]
    lines.append("This note records the per-row MLM loss cross-arm correlation and "
                 "shared/private decomposition computed from existing frontier_consolidation checkpoints.\n")
    lines.append("## Mean loss trajectories\n")
    for key in sorted(trajectories):
        arm_eset = key
        traj = trajectories[key]
        lines.append(f"**{arm_eset}**: " + ", ".join(f"{c}={v:.4f}" for c, v in sorted(traj.items())) + "\n")
    
    lines.append("\n## Cross-arm correlations of per-row loss change\n")
    lines.append("| comparison | eval_set | window | r | p | n | shared_frac |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|\n")
    for key in sorted(analysis["comparisons"]):
        parts = key.split("__")
        comp, eset, win = parts[0], parts[1], parts[2]
        c = analysis["comparisons"][key]
        d = analysis["decompositions"].get(key, {})
        lines.append(f"| {comp} | {eset} | {win} | {c['r']:.4f} | {c['p']:.2e} | {c['n']} | {d.get('shared_fraction', float('nan')):.4f} |\n")
    
    if analysis["source_stratified"]:
        lines.append("\n## Source-stratified correlations (heldout only)\n")
        for key in sorted(analysis["source_stratified"]):
            lines.append(f"\n**{key}**\n")
            for src, sc in sorted(analysis["source_stratified"][key].items()):
                lines.append(f"  {src}: r={sc['r']:.4f}, p={sc['p']:.2e}, n={sc['n']}\n")
    
    lines.append("\n## Files\n")
    lines.append(f"- Analysis JSON: `{json_path}`\n")
    lines.append(f"- Per-arm NPZ files: `{OUTPUT_DIR}/<arm>_per_row_losses.npz`\n")
    
    os.makedirs(os.path.dirname(NOTE_PATH), exist_ok=True)
    with open(NOTE_PATH, "w") as f:
        f.writelines(lines)
    print(f"Saved note: {NOTE_PATH}", flush=True)
    
    print(json.dumps({"status": "ANALYSIS_DONE", "json": json_path, "note": NOTE_PATH}, indent=2), flush=True)

if __name__ == "__main__":
    main()
