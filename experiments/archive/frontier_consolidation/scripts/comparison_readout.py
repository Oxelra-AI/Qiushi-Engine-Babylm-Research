#!/usr/bin/env python3
"""research: comparison readout for GPU-scored register and sub-dose arms vs clean.

Reads scoring results from gpu_priority_eval/per_target/ and computes
arm-minus-clean contrasts at matched checkpoints. Separates Entity from exEntity4
(mean of BLiMP, Supplement, EWoK, COMPS) using the specified comparison.

Run after GPU scoring produces per_target JSON files.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, math, pathlib, sys


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'

CLEAN_DIR = WS / "data" / "deberta_maxgeom_clean_stable_eval" / "eval" / "per_target"
GPU_DIR = WS / "data" / "gpu_priority_eval" / "per_target"

FAMILIES = ["BLiMP", "Supplement", "EWoK", "COMPS", "Entity"]
EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS"]


def load_scores(d: pathlib.Path) -> dict[str, dict[str, float | None]]:
    """Load {checkpoint: {family: score}} from per_target JSONs."""
    out: dict[str, dict[str, float | None]] = {}
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        data = json.loads(f.read_text())
        ck = data.get("checkpoint") or data.get("endpoint") or f.stem.split("_")[-1]
        tasks = data.get("tasks", {})
        row = {}
        for fam in FAMILIES:
            t = tasks.get(fam, {})
            s = t.get("score")
            row[fam] = float(s) if s is not None and math.isfinite(float(s)) else None
        out[ck] = row
    return out


def load_arm_scores(gpu_dir: pathlib.Path) -> dict[str, dict[str, dict[str, float | None]]]:
    """Load {arm: {checkpoint: {family: score}}} from GPU scoring per_target."""
    arms: dict[str, dict[str, dict[str, float | None]]] = {}
    if not gpu_dir.exists():
        return arms
    for f in sorted(gpu_dir.glob("*.json")):
        data = json.loads(f.read_text())
        arm = data.get("arm", "unknown")
        ck = data.get("checkpoint", f.stem.split("_")[-1])
        tasks = data.get("tasks", {})
        row = {}
        for fam in FAMILIES:
            t = tasks.get(fam, {})
            s = t.get("score")
            row[fam] = float(s) if s is not None and math.isfinite(float(s)) else None
        arms.setdefault(arm, {})[ck] = row
    return arms


def exE4(scores: dict[str, float | None]) -> float | None:
    vals = [scores[f] for f in EX_ENTITY if scores.get(f) is not None]
    return sum(vals) / len(vals) if len(vals) == 4 else None


def cheap5(scores: dict[str, float | None]) -> float | None:
    vals = [scores[f] for f in FAMILIES if scores.get(f) is not None]
    return sum(vals) / len(vals) if len(vals) == 5 else None


def main():
    # Load clean reference
    clean = load_scores(CLEAN_DIR)
    print(f"Clean reference: {len(clean)} checkpoints from {CLEAN_DIR.name}")

    # Load GPU-scored arms
    arm_scores = load_arm_scores(GPU_DIR)
    print(f"GPU-scored arms: {list(arm_scores.keys())} from {GPU_DIR}")
    for arm, ck_data in arm_scores.items():
        print(f"  {arm}: {len(ck_data)} checkpoints")
    print()

    if not arm_scores:
        print("NO SCORED ARMS YET. Run this after GPU scoring produces results.")
        return

    # Compute contrasts
    results: list[dict] = []
    for arm, ck_data in sorted(arm_scores.items()):
        print(f"\n{'='*70}")
        print(f"ARM: {arm}")
        print(f"{'Checkpoint':<12} {'BLiMP':>7} {'Supp':>7} {'EWoK':>7} {'COMPS':>7} {'Entity':>7} {'exE4':>7} {'ch5':>7}")
        print(f"{'':12} {'Δ':>7} {'Δ':>7} {'Δ':>7} {'Δ':>7} {'Δ':>7} {'Δ':>7} {'Δ':>7}")
        print("-" * 70)

        deltas_exE4 = []
        deltas_entity = []
        deltas_ch5 = []

        for ck in sorted(ck_data.keys()):
            arm_row = ck_data[ck]
            clean_row = clean.get(ck, {})

            # Absolute scores
            a_exE4 = exE4(arm_row)
            c_exE4 = exE4(clean_row)
            a_ch5 = cheap5(arm_row)
            c_ch5 = cheap5(clean_row)
            a_ent = arm_row.get("Entity")
            c_ent = clean_row.get("Entity")

            # Deltas
            d = {}
            for fam in FAMILIES:
                a = arm_row.get(fam)
                c = clean_row.get(fam)
                d[fam] = a - c if a is not None and c is not None else None

            d_exE4 = a_exE4 - c_exE4 if a_exE4 is not None and c_exE4 is not None else None
            d_ch5 = a_ch5 - c_ch5 if a_ch5 is not None and c_ch5 is not None else None
            d_ent = d.get("Entity")

            def fmt(v, w=7):
                return f"{v:+{w}.3f}" if v is not None else f"{'n/a':>{w}}"

            def fmta(v, w=7):
                return f"{v:{w}.3f}" if v is not None else f"{'n/a':>{w}}"

            print(f"{ck:<12} {fmta(arm_row.get('BLiMP'))} {fmta(arm_row.get('Supplement'))} "
                  f"{fmta(arm_row.get('EWoK'))} {fmta(arm_row.get('COMPS'))} "
                  f"{fmta(a_ent)} {fmta(a_exE4)} {fmta(a_ch5)}")
            print(f"{'  Δ clean':<12} {fmt(d.get('BLiMP'))} {fmt(d.get('Supplement'))} "
                  f"{fmt(d.get('EWoK'))} {fmt(d.get('COMPS'))} "
                  f"{fmt(d_ent)} {fmt(d_exE4)} {fmt(d_ch5)}")

            if d_exE4 is not None:
                deltas_exE4.append(d_exE4)
            if d_ent is not None:
                deltas_entity.append(d_ent)
            if d_ch5 is not None:
                deltas_ch5.append(d_ch5)

            results.append({
                "arm": arm, "checkpoint": ck,
                "scores": {f: arm_row.get(f) for f in FAMILIES},
                "deltas": {f: d.get(f) for f in FAMILIES},
                "exEntity4": a_exE4, "delta_exE4": d_exE4,
                "cheap5": a_ch5, "delta_ch5": d_ch5,
            })

        if deltas_exE4:
            print(f"\n  Mean Δ exEntity4: {sum(deltas_exE4)/len(deltas_exE4):+.4f} "
                  f"(over {len(deltas_exE4)} checkpoints)")
        if deltas_entity:
            print(f"  Mean Δ Entity:    {sum(deltas_entity)/len(deltas_entity):+.4f}")
        if deltas_ch5:
            print(f"  Mean Δ cheap5:    {sum(deltas_ch5)/len(deltas_ch5):+.4f}")

    # Save results
    out_dir = WS / "data" / "comparison_readout"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "status": "COMPARISON_READOUT",
        "clean_checkpoints": len(clean),
        "arm_count": len(arm_scores),
        "results": results,
    }
    (out_dir / "comparison_readout.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved to {out_dir / 'comparison_readout.json'}")


if __name__ == "__main__":
    main()
