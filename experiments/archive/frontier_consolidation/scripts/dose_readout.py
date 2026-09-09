#!/usr/bin/env python3
"""research: standalone dose-curve readout from CPU-parallel evaluation results.

Reads the unified research CSV and the research seed-spread data to produce
the first complete dose-response profile.  Independent of the managed
integrator chain.

Produces:
  - dose_semantic_profile.csv      V-R at each dose and checkpoint
  - dose_growth_profile.csv        V_dose - V_1x clean-free growth
  - dose_total_profile.csv         V-C total treatment at matched checkpoints
  - dose_readout_summary.json      Complete scientific readout
  - dose_readout_summary.md        Human-readable summary

Usage:
  python3 -B dose_readout.py [--csv PATH]
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import pathlib
import statistics
from typing import Any, Dict, List, Optional, Tuple


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
OUT = WS / "data/dose_readout"
DEFAULT_CSV = WS / "data/cpu_parallel_dose_eval/dose_ladder_stable_rows.csv"
SEED_SPREAD_CSV = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_seed_spread.csv"

FAMILIES = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
PRIMARY = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading"]
SECONDARY = ["EWoK_plus_Entity_sum"]

DOSE_MAP = {
    "clean0": 0.0,
    "dose1_view": 1.0, "dose1_repeat": 1.0,
    "dose1p82_view": 1.82, "dose1p82_repeat": 1.82,
    "max_view": 2.64, "max_repeat": 2.64,
}

CKS = [f"chck_{i}M" for i in range(10, 101, 10)]


def read_csv(path: pathlib.Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FAMILIES + PRIMARY + SECONDARY + ["words"]:
            if k in r and r[k] not in (None, "", "None"):
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
    return rows


def lookup(rows: List[Dict], arm: str, ck: str) -> Optional[Dict]:
    for r in rows:
        if r.get("arm") == arm and r.get("checkpoint") == ck:
            return r
    return None


def safe_float(v) -> Optional[float]:
    if v is None or v == "" or v == "None":
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def delta_row(a: Dict, b: Dict, label: str) -> Dict[str, Any]:
    """Compute a - b for all numeric columns."""
    d = {"label": label, "checkpoint": a.get("checkpoint"), "words": a.get("words")}
    for col in FAMILIES + PRIMARY + SECONDARY:
        va, vb = safe_float(a.get(col)), safe_float(b.get(col))
        if va is not None and vb is not None:
            d[col] = round(va - vb, 6)
        else:
            d[col] = None
    return d


def read_seed_spread() -> Dict[str, Dict[str, float]]:
    """Read research seed spread: per-checkpoint treatment delta spread."""
    if not SEED_SPREAD_CSV.is_file():
        return {}
    result = {}
    with open(SEED_SPREAD_CSV) as f:
        for r in csv.DictReader(f):
            ck = r.get("checkpoint", "")
            result[ck] = {}
            for col in FAMILIES + PRIMARY + SECONDARY:
                key = f"treatment_delta_seed43122_minus_seed43022_{col}"
                if key in r:
                    try:
                        result[ck][col] = float(r[key])
                    except (ValueError, TypeError):
                        pass
    return result


def compute_semantic_profile(rows: List[Dict]) -> List[Dict]:
    """V-R at each dose and checkpoint."""
    profile = []
    for dose_name, dose_val in [("dose1", 1.0), ("dose1p82", 1.82), ("dose2p64", 2.64)]:
        view_arm = {"dose1": "dose1_view", "dose1p82": "dose1p82_view", "dose2p64": "max_view"}[dose_name]
        repeat_arm = {"dose1": "dose1_repeat", "dose1p82": "dose1p82_repeat", "dose2p64": "max_repeat"}[dose_name]
        for ck in CKS:
            v = lookup(rows, view_arm, ck)
            r = lookup(rows, repeat_arm, ck)
            if v and r:
                d = delta_row(v, r, f"VR_{dose_name}")
                d["dose"] = dose_val
                d["dose_name"] = dose_name
                profile.append(d)
    return profile


def compute_growth_profile(rows: List[Dict]) -> List[Dict]:
    """Clean-free view-arm growth: V_dose - V_1x at each checkpoint."""
    profile = []
    for dose_name in ["dose1p82", "dose2p64"]:
        view_arm = {"dose1p82": "dose1p82_view", "dose2p64": "max_view"}[dose_name]
        dose_val = {"dose1p82": 1.82, "dose2p64": 2.64}[dose_name]
        for ck in CKS:
            vd = lookup(rows, view_arm, ck)
            v1 = lookup(rows, "dose1_view", ck)
            if vd and v1:
                d = delta_row(vd, v1, f"growth_V_{dose_name}_minus_V_1x")
                d["dose"] = dose_val
                d["dose_name"] = dose_name
                profile.append(d)
    return profile


def compute_total_profile(rows: List[Dict]) -> List[Dict]:
    """V-C total treatment at matched checkpoints (clean only has 10-80M)."""
    profile = []
    clean_cks = [f"chck_{i}M" for i in range(10, 81, 10)]
    for dose_name, view_arm, dose_val in [
        ("dose1", "dose1_view", 1.0),
        ("dose1p82", "dose1p82_view", 1.82),
        ("dose2p64", "max_view", 2.64),
    ]:
        for ck in clean_cks:
            v = lookup(rows, view_arm, ck)
            c = lookup(rows, "clean0", ck)
            if v and c:
                d = delta_row(v, c, f"VC_{dose_name}")
                d["dose"] = dose_val
                d["dose_name"] = dose_name
                profile.append(d)
    return profile


def profile_summary(profile: List[Dict], label: str) -> Dict[str, Any]:
    """Summarize a profile: mean across checkpoints for primary composites."""
    if not profile:
        return {"label": label, "n": 0}

    by_dose: Dict[float, List[Dict]] = {}
    for p in profile:
        dose = p.get("dose", 0)
        by_dose.setdefault(dose, []).append(p)

    summary = {"label": label, "doses": {}}
    for dose, items in sorted(by_dose.items()):
        dose_sum: Dict[str, Any] = {"n_checkpoints": len(items)}
        for col in PRIMARY + SECONDARY + FAMILIES:
            vals = [safe_float(i.get(col)) for i in items]
            vals = [v for v in vals if v is not None]
            if vals:
                dose_sum[f"{col}_mean"] = round(statistics.mean(vals), 4)
                dose_sum[f"{col}_median"] = round(statistics.median(vals), 4)
                if len(vals) >= 3:
                    dose_sum[f"{col}_stdev"] = round(statistics.stdev(vals), 4)
        # Also compute 80M and 100M endpoints separately
        for ck_label in ["chck_80M", "chck_100M"]:
            ck_items = [i for i in items if i.get("checkpoint") == ck_label]
            if ck_items:
                for col in PRIMARY + SECONDARY:
                    v = safe_float(ck_items[0].get(col))
                    if v is not None:
                        dose_sum[f"{col}_at_{ck_label}"] = round(v, 4)

        summary["doses"][f"dose_{dose:.2f}"] = dose_sum
    return summary


def write_profile_csv(path: pathlib.Path, profile: List[Dict]) -> None:
    if not profile:
        return
    fields = ["label", "dose_name", "dose", "checkpoint", "words"] + FAMILIES + PRIMARY + SECONDARY
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in profile:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=pathlib.Path, default=DEFAULT_CSV)
    args = ap.parse_args()

    if not args.csv.is_file():
        print(json.dumps({"error": f"CSV not found: {args.csv}", "hint": "Run cpu_parallel_dose_eval.py first"}))
        raise SystemExit(1)

    OUT.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.csv)
    seed_spread = read_seed_spread()

    # Check coverage
    coverage = {}
    for r in rows:
        arm = r.get("arm", "unknown")
        coverage.setdefault(arm, []).append(r.get("checkpoint"))

    print(json.dumps({"status": "DOSE_READOUT_START", "input_rows": len(rows),
                       "coverage": {k: len(v) for k, v in coverage.items()}}), flush=True)

    # Compute profiles
    semantic = compute_semantic_profile(rows)
    growth = compute_growth_profile(rows)
    total = compute_total_profile(rows)

    write_profile_csv(OUT / "dose_semantic_profile.csv", semantic)
    write_profile_csv(OUT / "dose_growth_profile.csv", growth)
    write_profile_csv(OUT / "dose_total_profile.csv", total)

    # Build summary
    sem_sum = profile_summary(semantic, "semantic_VR")
    growth_sum = profile_summary(growth, "clean_free_growth")
    total_sum = profile_summary(total, "total_VC")

    # Seed spread context
    spread_ctx = {}
    if seed_spread:
        for col in PRIMARY + SECONDARY:
            vals = []
            for ck, sd in seed_spread.items():
                if col in sd:
                    vals.append(abs(sd[col]))
            if vals:
                spread_ctx[col] = {
                    "mean_abs_spread": round(statistics.mean(vals), 4),
                    "max_abs_spread": round(max(vals), 4),
                }

    # Absolute ladder values for each arm
    abs_ladders = {}
    for arm_name in DOSE_MAP:
        arm_rows = [r for r in rows if r.get("arm") == arm_name]
        if arm_rows:
            arm_rows.sort(key=lambda r: r.get("words", 0))
            ladder = []
            for r in arm_rows:
                entry = {"checkpoint": r.get("checkpoint"), "words": r.get("words")}
                for col in PRIMARY + SECONDARY + FAMILIES:
                    v = safe_float(r.get(col))
                    if v is not None:
                        entry[col] = round(v, 4)
                ladder.append(entry)
            abs_ladders[arm_name] = ladder

    summary = {
        "status": "DOSE_READOUT_COMPLETE",
        "input_csv": str(args.csv),
        "input_rows": len(rows),
        "coverage": {k: len(v) for k, v in coverage.items()},
        "semantic_profile": sem_sum,
        "growth_profile": growth_sum,
        "total_profile": total_sum,
        "seed_spread_context": spread_ctx,
        "absolute_ladders": abs_ladders,
        "reading_order_note": "research frozen order: ladder means before endpoints, "
            "V-R before growth, cheap6/cheap5 must agree, EWoK+Entity secondary",
        "profiles": {
            "semantic": str(OUT / "dose_semantic_profile.csv"),
            "growth": str(OUT / "dose_growth_profile.csv"),
            "total": str(OUT / "dose_total_profile.csv"),
        },
    }

    with open(OUT / "dose_readout_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Write human-readable MD
    md = ["# research Dose-Response Readout\n"]
    md.append(f"Input: {args.csv} ({len(rows)} rows)\n")
    md.append(f"Coverage: {', '.join(f'{k}: {len(v)} ckpts' for k, v in coverage.items())}\n")

    if spread_ctx:
        md.append("\n## Seed spread context (same-architecture, two seeds)\n")
        for col, ctx in spread_ctx.items():
            md.append(f"- {col}: mean |spread| = {ctx['mean_abs_spread']:.4f}, max = {ctx['max_abs_spread']:.4f}\n")

    md.append("\n## Semantic profile (V-R): view minus repeat at each dose\n")
    for dk, ds in sem_sum.get("doses", {}).items():
        md.append(f"\n### {dk} (n={ds.get('n_checkpoints', 0)} checkpoints)\n")
        for col in PRIMARY + SECONDARY:
            m = ds.get(f"{col}_mean")
            s = ds.get(f"{col}_stdev")
            e80 = ds.get(f"{col}_at_chck_80M")
            e100 = ds.get(f"{col}_at_chck_100M")
            parts = [f"mean={m:.4f}" if m is not None else ""]
            if s is not None: parts.append(f"sd={s:.4f}")
            if e80 is not None: parts.append(f"@80M={e80:.4f}")
            if e100 is not None: parts.append(f"@100M={e100:.4f}")
            md.append(f"- {col}: {', '.join(p for p in parts if p)}\n")

    md.append("\n## Clean-free view growth: V_dose - V_1x\n")
    for dk, ds in growth_sum.get("doses", {}).items():
        md.append(f"\n### {dk}\n")
        for col in PRIMARY + SECONDARY:
            m = ds.get(f"{col}_mean")
            s = ds.get(f"{col}_stdev")
            e80 = ds.get(f"{col}_at_chck_80M")
            e100 = ds.get(f"{col}_at_chck_100M")
            parts = [f"mean={m:.4f}" if m is not None else ""]
            if s is not None: parts.append(f"sd={s:.4f}")
            if e80 is not None: parts.append(f"@80M={e80:.4f}")
            if e100 is not None: parts.append(f"@100M={e100:.4f}")
            md.append(f"- {col}: {', '.join(p for p in parts if p)}\n")

    md.append("\n## Total treatment (V-C): view minus clean (10-80M only)\n")
    for dk, ds in total_sum.get("doses", {}).items():
        md.append(f"\n### {dk}\n")
        for col in PRIMARY + SECONDARY:
            m = ds.get(f"{col}_mean")
            e80 = ds.get(f"{col}_at_chck_80M")
            parts = [f"mean={m:.4f}" if m is not None else ""]
            if e80 is not None: parts.append(f"@80M={e80:.4f}")
            md.append(f"- {col}: {', '.join(p for p in parts if p)}\n")

    md.append("\n## Absolute ladder endpoints (chck_100M or last available)\n")
    for arm_name in ["clean0", "dose1_view", "dose1_repeat",
                     "dose1p82_view", "dose1p82_repeat", "max_view", "max_repeat"]:
        ladder = abs_ladders.get(arm_name, [])
        if ladder:
            last = ladder[-1]
            md.append(f"- {arm_name} ({last.get('checkpoint', '?')}): "
                      + ", ".join(f"{c}={last.get(c, '?')}" for c in PRIMARY[:1] + SECONDARY[:1])
                      + "\n")

    with open(OUT / "dose_readout_summary.md", "w") as f:
        f.writelines(md)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
