#!/usr/bin/env python3
"""research: integrated evidence readout across all arms.

Reads scored checkpoints from:
  - parallel_eval/register/  (adultprose, childspeech)
  - parallel_eval/subdose/   (quarter, half, full)
  - reference_decomposition_readout/  (existing V/B/R/clean scores)
  - dose_ladder_stable_eval/  (existing dose scores)

Produces a combined table of V-C, R-C, B-C, register contrasts, sub-dose curve,
and the in-corpus arm (when scored).

Output: data/integrated_readout/
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import collections, csv, json, math, pathlib, statistics, time
from typing import Any

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
OUT = WS / "data/integrated_readout"

STABLE_FAMILIES = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
STABLE_CKS = [f"chck_{i}M" for i in range(10, 81, 10)]
EXENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]

def finite(x):
    try: return math.isfinite(float(x))
    except: return False

def load_per_target(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except:
        return None

def get_score(payload, col):
    if not payload:
        return None
    tasks = payload.get("tasks", {})
    rec = tasks.get(col)
    if not isinstance(rec, dict):
        # Check stable_family_scores
        scores = payload.get("stable_family_scores", {})
        v = scores.get(col)
        return float(v) if finite(v) else None
    if col == "Reading":
        v = (rec.get("scores") or {}).get("Reading")
    else:
        v = rec.get("score")
    return float(v) if finite(v) else None

def aggregate_scores(payload):
    scores = {}
    for col in STABLE_FAMILIES:
        scores[col] = get_score(payload, col)
    vals6 = [scores[c] for c in STABLE_FAMILIES]
    vals5 = [scores[c] for c in EXENTITY]
    scores["cheap6"] = statistics.mean([float(x) for x in vals6]) if all(finite(x) for x in vals6) else None
    scores["exEntity5"] = statistics.mean([float(x) for x in vals5]) if all(finite(x) for x in vals5) else None
    return scores

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ── Source directories ──
SOURCES = {
    "register": WS / "data/parallel_eval/register",
    "subdose": WS / "data/parallel_eval/subdose",
    "existing_dose": WS / "data/dose_ladder_stable_eval/eval",
    "existing_decomp": WS / "data/reference_decomposition_readout",
    "clean_anchor": WS / "data/clean_anchor_commonwindow_eval/eval",
    "incorpus": WS / "data/parallel_eval/incorpus",  # future
}

# ── Arm definitions with label prefix mapping ──
ARM_DEFS = {
    "regmax_adultprose": {
        "search_dirs": ["register"],
        "prefix": "regmax_adultprose_seed43022",
        "description": "MAX register: adult-prose removal",
        "rho": 0.1119,
        "group": "register",
    },
    "regmax_childspeech": {
        "search_dirs": ["register"],
        "prefix": "regmax_childspeech_seed43022",
        "description": "MAX register: childspeech removal",
        "rho": 0.1119,
        "group": "register",
    },
    "subdose_quarter_1x": {
        "search_dirs": ["subdose"],
        "prefix": "subdose_quarter_1x_seed43022",
        "description": "Sub-dose quarter (rho≈0.0106)",
        "rho": 0.0106,
        "group": "subdose",
    },
    "subdose_half_1x": {
        "search_dirs": ["subdose"],
        "prefix": "subdose_half_1x_seed43022",
        "description": "Sub-dose half (rho≈0.0212)",
        "rho": 0.0212,
        "group": "subdose",
    },
    "subdose_full_1x": {
        "search_dirs": ["subdose"],
        "prefix": "subdose_full_1x_seed43022",
        "description": "Sub-dose full (rho≈0.0424)",
        "rho": 0.0424,
        "group": "subdose",
    },
    "max_view": {
        "search_dirs": ["existing_dose"],
        "prefix": "dose_max_view",
        "description": "MAX dose view (rho≈0.1119)",
        "rho": 0.1119,
        "group": "dose",
    },
    "max_repeat": {
        "search_dirs": ["existing_dose"],
        "prefix": "dose_max_repeat",
        "description": "MAX dose repeat (rho≈0.1119)",
        "rho": 0.1119,
        "group": "dose",
    },
    "max_breadth": {
        "search_dirs": ["existing_dose"],
        "prefix": "dose_max_breadth",
        "description": "MAX dose breadth (rho≈0.1119)",
        "rho": 0.1119,
        "group": "dose",
    },
    "clean_maxgeom": {
        "search_dirs": ["clean_anchor"],
        "prefix": "deberta_maxgeom_clean_seed43022",
        "description": "Clean MAX-geometry reference (seed43022)",
        "rho": 0.0,
        "group": "reference",
    },
    "incorpus_adultprose": {
        "search_dirs": ["incorpus"],
        "prefix": "incorpus_adultprose_seed43022",
        "description": "In-corpus adult prose (no FineWeb, rho≈0.0443)",
        "rho": 0.0443,
        "group": "incorpus",
    },
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Collect all scores
    all_scores: dict[str, dict[str, dict]] = {}  # arm -> ck -> {family: score}
    arm_status: dict[str, dict] = {}

    for arm_name, adef in ARM_DEFS.items():
        arm_scores: dict[str, dict] = {}
        for search_key in adef["search_dirs"]:
            search_dir = SOURCES.get(search_key)
            if not search_dir:
                continue
            per_target = search_dir / "per_target"
            if not per_target.exists():
                continue
            for ck in STABLE_CKS:
                target = f"{adef['prefix']}_{ck}"
                payload = load_per_target(per_target / f"{target}.json")
                if payload:
                    scores = aggregate_scores(payload)
                    if any(finite(v) for v in scores.values()):
                        arm_scores[ck] = scores

        all_scores[arm_name] = arm_scores
        complete_cks = [ck for ck, s in arm_scores.items() if finite(s.get("cheap6"))]
        arm_status[arm_name] = {
            "description": adef["description"],
            "rho": adef["rho"],
            "group": adef["group"],
            "scored_checkpoints": len(arm_scores),
            "complete_checkpoints": len(complete_cks),
            "complete_list": complete_cks,
        }

    # ── Write raw scores CSV ──
    csv_path = OUT / "all_arm_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        cols = ["arm", "group", "rho", "checkpoint"] + STABLE_FAMILIES + ["cheap6", "exEntity5"]
        writer.writerow(cols)
        for arm_name in sorted(all_scores.keys()):
            adef = ARM_DEFS[arm_name]
            for ck in STABLE_CKS:
                if ck not in all_scores[arm_name]:
                    continue
                s = all_scores[arm_name][ck]
                row = [arm_name, adef["group"], adef["rho"], ck]
                for col in STABLE_FAMILIES + ["cheap6", "exEntity5"]:
                    row.append(round(s[col], 4) if finite(s.get(col)) else "")
                writer.writerow(row)

    # ── Compute contrasts against clean reference ──
    clean_scores = all_scores.get("clean_maxgeom", {})
    contrast_rows = []

    for arm_name, arm_scores in all_scores.items():
        if arm_name == "clean_maxgeom":
            continue
        for ck in STABLE_CKS:
            if ck not in arm_scores or ck not in clean_scores:
                continue
            s = arm_scores[ck]
            c = clean_scores[ck]
            contrast = {}
            for col in STABLE_FAMILIES + ["cheap6", "exEntity5"]:
                if finite(s.get(col)) and finite(c.get(col)):
                    contrast[col] = round(s[col] - c[col], 4)
            if contrast:
                contrast_rows.append({
                    "arm": arm_name,
                    "group": ARM_DEFS[arm_name]["group"],
                    "rho": ARM_DEFS[arm_name]["rho"],
                    "checkpoint": ck,
                    **contrast,
                })

    contrast_csv = OUT / "arm_minus_clean_contrasts.csv"
    with open(contrast_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arm", "group", "rho", "checkpoint"] + STABLE_FAMILIES + ["cheap6", "exEntity5"])
        writer.writeheader()
        for row in contrast_rows:
            writer.writerow(row)

    # ── Register contrast: childspeech_removed - adultprose_removed ──
    register_contrast = {}
    child_scores = all_scores.get("regmax_childspeech", {})
    adult_scores = all_scores.get("regmax_adultprose", {})
    for ck in STABLE_CKS:
        if ck in child_scores and ck in adult_scores:
            cs = child_scores[ck]
            ads = adult_scores[ck]
            diff = {}
            for col in STABLE_FAMILIES + ["cheap6", "exEntity5"]:
                if finite(cs.get(col)) and finite(ads.get(col)):
                    diff[col] = round(cs[col] - ads[col], 4)
            if diff:
                register_contrast[ck] = diff

    # ── Summary statistics ──
    def window_mean(arm_scores, cks, col):
        vals = [arm_scores[ck][col] for ck in cks if ck in arm_scores and finite(arm_scores[ck].get(col))]
        return round(statistics.mean(vals), 4) if vals else None

    common_cks = [f"chck_{i}M" for i in range(10, 81, 10)]
    late_cks = ["chck_80M"]

    summary_lines = [
        f"# research integrated evidence readout",
        f"Generated: {now()}\n",
        f"## Arm scoring status\n",
    ]
    for arm_name, st in sorted(arm_status.items()):
        summary_lines.append(f"- **{arm_name}** ({st['description']}): {st['complete_checkpoints']}/{len(STABLE_CKS)} complete checkpoints")

    total_scored = sum(st["complete_checkpoints"] for st in arm_status.values())
    total_possible = len(ARM_DEFS) * len(STABLE_CKS)
    summary_lines.append(f"\nTotal: {total_scored}/{total_possible} complete (arm × checkpoint) cells\n")

    # Sub-dose curve
    summary_lines.append("## Sub-dose curve (arm minus clean)\n")
    summary_lines.append("| arm | rho | cheap6 | exEntity5 | Entity |\n|---|---|---|---|---|")
    for arm_name in ["subdose_quarter_1x", "subdose_half_1x", "subdose_full_1x", "max_view"]:
        adef = ARM_DEFS.get(arm_name, {})
        arm_c = [r for r in contrast_rows if r["arm"] == arm_name]
        if arm_c:
            mean_cheap6 = statistics.mean([r["cheap6"] for r in arm_c if "cheap6" in r]) if arm_c else None
            mean_exE5 = statistics.mean([r["exEntity5"] for r in arm_c if "exEntity5" in r]) if arm_c else None
            mean_ent = statistics.mean([r["Entity"] for r in arm_c if "Entity" in r]) if arm_c else None
            c6 = f"{mean_cheap6:.4f}" if mean_cheap6 is not None else "n/a"
            e5 = f"{mean_exE5:.4f}" if mean_exE5 is not None else "n/a"
            ent = f"{mean_ent:.4f}" if mean_ent is not None else "n/a"
            summary_lines.append(f"| {arm_name} | {adef.get('rho', '?')} | {c6} | {e5} | {ent} |")
        else:
            summary_lines.append(f"| {arm_name} | {adef.get('rho', '?')} | no data | no data | no data |")

    # Register contrast
    summary_lines.append("\n## Register contrast (childspeech_removed - adultprose_removed)\n")
    if register_contrast:
        summary_lines.append("| checkpoint | cheap6 | exEntity5 | Entity | BLiMP | Supplement | EWoK | COMPS | Reading |\n|---|---|---|---|---|---|---|---|---|")
        for ck in STABLE_CKS:
            if ck in register_contrast:
                d = register_contrast[ck]
                summary_lines.append(f"| {ck} | {d.get('cheap6', 'n/a')} | {d.get('exEntity5', 'n/a')} | {d.get('Entity', 'n/a')} | {d.get('BLiMP', 'n/a')} | {d.get('Supplement', 'n/a')} | {d.get('EWoK', 'n/a')} | {d.get('COMPS', 'n/a')} | {d.get('Reading', 'n/a')} |")
    else:
        summary_lines.append("No register contrast data available yet.\n")

    # In-corpus vs FineWeb comparison
    summary_lines.append("\n## In-corpus vs FineWeb (both minus clean)\n")
    for arm_name in ["incorpus_adultprose", "subdose_full_1x"]:
        arm_c = [r for r in contrast_rows if r["arm"] == arm_name]
        if arm_c:
            mean_cheap6 = statistics.mean([r["cheap6"] for r in arm_c if "cheap6" in r])
            mean_exE5 = statistics.mean([r["exEntity5"] for r in arm_c if "exEntity5" in r])
            summary_lines.append(f"- **{arm_name}**: mean cheap6 = {mean_cheap6:.4f}, mean exEntity5 = {mean_exE5:.4f}")
        else:
            summary_lines.append(f"- **{arm_name}**: no data yet")

    # Persistence
    summary_lines.append("\n## Persistence note\n")
    summary_lines.append("R-Cmax exEntity5 decays from +0.4361 (common10_80) to +0.0343 (late80_100)")
    summary_lines.append("V-Cmax exEntity5 holds at +0.4998 (common10_80) and +0.3853 (late80_100)")
    summary_lines.append("B-Cmax exEntity5 holds at +0.6691 (common10_80) and +0.3350 (late80_100)")
    summary_lines.append("Any principle must explain why distinct content persists and duplicated content decays.\n")

    summary_text = "\n".join(summary_lines)
    (OUT / "integrated_readout_summary.md").write_text(summary_text)

    # Save metadata
    meta = {
        "status": "INTEGRATED_READOUT",
        "created_utc": now(),
        "arm_status": arm_status,
        "total_scored_cells": total_scored,
        "total_possible_cells": total_possible,
        "register_contrast_checkpoints": list(register_contrast.keys()),
        "csv_files": {
            "scores": str(csv_path.relative_to(ROOT)),
            "contrasts": str(contrast_csv.relative_to(ROOT)),
        },
    }
    (OUT / "readout_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta, indent=2), flush=True)
    print("\n" + summary_text, flush=True)


if __name__ == "__main__":
    main()
