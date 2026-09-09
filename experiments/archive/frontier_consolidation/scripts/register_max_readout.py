#!/usr/bin/env python3
"""research: assemble MAX-register displacement readout once scores exist.

Reads stable-family per-target JSONs for:
  * research MAX-register arms (childspeech removed, adultprose removed)
  * research proportional MAX view midpoint
  * DeBERTa MAX-geometry clean anchors from research/281 outputs

Produces direct childspeech-minus-adultprose contrasts, arm-minus-view placement,
and arm-minus-clean placement over common-window checkpoints.  The script is
file-only and can run before register scores exist; then it truthfully reports
partial data.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import statistics
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
OUT = WS / "data/register_max_readout"
REG_EVAL = WS / "data/register_max_commonwindow_eval/eval/per_target"
MAX_VIEW_EVAL = WS / "data/dose_ladder_stable_eval/eval/per_target"
CLEAN_EVAL_DIRS = [
    WS / "data/deberta_maxgeom_clean_stable_eval/eval/per_target",
    WS / "data/clean_anchor_commonwindow_eval/eval/per_target",
]
META = WS / "data/register_max_rowholdout_pools/register_max_rowholdout_metadata.json"
STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
EX_ENTITY_COLUMNS = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 81, 10)]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def metric_from_payload(path: pathlib.Path, col: str) -> float | None:
    if not path.exists():
        return None
    try:
        d = read_json(path)
    except Exception:
        return None
    ss = d.get("stable_family_scores") or {}
    if finite(ss.get(col)):
        return float(ss[col])
    rec = (d.get("tasks") or {}).get(col) or {}
    if col == "Reading":
        val = (rec.get("scores") or {}).get("Reading") if isinstance(rec.get("scores"), dict) else rec.get("score")
    else:
        val = rec.get("score")
    return float(val) if finite(val) else None


def aggregate(vals: dict[str, float | None]) -> dict[str, float | None]:
    out = dict(vals)
    out["cheap6"] = statistics.mean([float(vals[c]) for c in STABLE_COLUMNS]) if all(finite(vals.get(c)) for c in STABLE_COLUMNS) else None
    out["cheap5"] = statistics.mean([float(vals[c]) for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]]) if all(finite(vals.get(c)) for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]) else None
    out["exEntity5"] = statistics.mean([float(vals[c]) for c in EX_ENTITY_COLUMNS]) if all(finite(vals.get(c)) for c in EX_ENTITY_COLUMNS) else None
    return out


def score_record(label: str, kind: str, ck: str, path: pathlib.Path) -> dict[str, Any]:
    vals = {c: metric_from_payload(path, c) for c in STABLE_COLUMNS}
    agg = aggregate(vals)
    return {"label": label, "kind": kind, "checkpoint": ck, "path": rel(path), "exists": path.exists(), "complete_stable6": all(finite(vals[c]) for c in STABLE_COLUMNS), **agg}


def clean_paths(seed: int, ck: str) -> list[pathlib.Path]:
    names = [
        f"deberta_maxgeom_clean_seed{seed}_{ck}.json",
        f"deberta_maxgeom_clean_seed{seed}_{ck}.json",
    ]
    out: list[pathlib.Path] = []
    for root in CLEAN_EVAL_DIRS:
        for name in names:
            p = root / name
            if p.exists():
                out.append(p)
    # Prefer research if available for the same seed/checkpoint because it is the two-anchor readout root,
    # otherwise keep research older files.
    unique: dict[str, pathlib.Path] = {}
    for p in out:
        key = p.name.replace("step274_", "").replace("step281_", "")
        if "research" in str(p):
            unique[key] = p
        else:
            unique.setdefault(key, p)
    return list(unique.values())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = read_json(META)
    score_rows: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        score_rows.append(score_record("regmax_childspeech_removed", "register_arm", ck, REG_EVAL / f"regmax_childspeech_samefw_seed43022_{ck}.json"))
        score_rows.append(score_record("regmax_adultprose_removed", "register_arm", ck, REG_EVAL / f"regmax_adultprose_samefw_seed43022_{ck}.json"))
        score_rows.append(score_record("proportional_max_view", "view_midpoint", ck, MAX_VIEW_EVAL / f"dose_max_view_{ck}.json"))
        for seed in [43022, 43122]:
            paths = clean_paths(seed, ck)
            if paths:
                # If duplicate path options exist, use first after preference logic.
                score_rows.append(score_record(f"clean_seed{seed}", "clean", ck, paths[0]))
            else:
                score_rows.append(score_record(f"clean_seed{seed}", "clean", ck, CLEAN_EVAL_DIRS[0] / f"missing_seed{seed}_{ck}.json"))

    by = {(r["label"], r["checkpoint"]): r for r in score_rows}
    metrics = STABLE_COLUMNS + ["cheap6", "cheap5", "exEntity5"]
    contrast_rows: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        child = by[("regmax_childspeech_removed", ck)]
        adult = by[("regmax_adultprose_removed", ck)]
        view = by[("proportional_max_view", ck)]
        clean_recs = [by[(f"clean_seed{s}", ck)] for s in [43022, 43122]]
        clean_mean = {}
        clean_spread = {}
        for m in metrics:
            vals = [r.get(m) for r in clean_recs if finite(r.get(m))]
            clean_mean[m] = statistics.mean([float(v) for v in vals]) if vals else None
            clean_spread[m] = (max([float(v) for v in vals]) - min([float(v) for v in vals])) if len(vals) >= 2 else None
        for label, a, b, kind in [
            ("child_minus_adult", child, adult, "primary_register_contrast"),
            ("child_minus_view", child, view, "arm_minus_proportional_view"),
            ("adult_minus_view", adult, view, "arm_minus_proportional_view"),
        ]:
            rec = {"contrast": label, "kind": kind, "checkpoint": ck, "a_complete": a.get("complete_stable6"), "b_complete": b.get("complete_stable6")}
            for m in metrics:
                rec[m] = float(a[m]) - float(b[m]) if finite(a.get(m)) and finite(b.get(m)) else None
            rec["complete_stable6_contrast"] = bool(a.get("complete_stable6") and b.get("complete_stable6"))
            contrast_rows.append(rec)
        for arm_label, arm_rec in [("child_minus_clean_mean", child), ("adult_minus_clean_mean", adult), ("view_minus_clean_mean", view)]:
            rec = {"contrast": arm_label, "kind": "arm_or_view_minus_clean_mean", "checkpoint": ck, "a_complete": arm_rec.get("complete_stable6"), "clean_available_seeds": sum(1 for r in clean_recs if r.get("complete_stable6"))}
            for m in metrics:
                rec[m] = float(arm_rec[m]) - float(clean_mean[m]) if finite(arm_rec.get(m)) and finite(clean_mean[m]) else None
                rec[m + "_clean_spread"] = clean_spread[m]
            rec["complete_stable6_contrast"] = bool(arm_rec.get("complete_stable6") and finite(clean_mean.get("cheap6")))
            contrast_rows.append(rec)

    traj_rows: list[dict[str, Any]] = []
    for contrast in sorted({r["contrast"] for r in contrast_rows}):
        group = [r for r in contrast_rows if r["contrast"] == contrast and r.get("complete_stable6_contrast")]
        rec: dict[str, Any] = {"contrast": contrast, "complete_checkpoint_count": len(group), "complete_checkpoints": [r["checkpoint"] for r in group]}
        for m in metrics:
            vals = [r.get(m) for r in group if finite(r.get(m))]
            rec[m + "_mean"] = statistics.mean([float(v) for v in vals]) if vals else None
            rec[m + "_min"] = min([float(v) for v in vals]) if vals else None
            rec[m + "_max"] = max([float(v) for v in vals]) if vals else None
            if any((m + "_clean_spread") in r for r in group):
                sp = [r.get(m + "_clean_spread") for r in group if finite(r.get(m + "_clean_spread"))]
                rec[m + "_clean_spread_mean"] = statistics.mean([float(x) for x in sp]) if sp else None
        traj_rows.append(rec)

    for path, rows in [(OUT / "score_rows.csv", score_rows), (OUT / "contrast_rows.csv", contrast_rows), (OUT / "trajectory_contrasts.csv", traj_rows)]:
        fields = sorted({k for r in rows for k in r.keys()}) if rows else []
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

    result = {
        "status": "REGISTER_MAX_READOUT_COMPLETE",
        "created_utc": now(),
        "data_state": "complete_register_scores" if all(r.get("complete_stable6") for r in score_rows if r["kind"] == "register_arm") else "partial",
        "boundary": "File-only readout; no model loading, training, official evaluation, GPU, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.",
        "metadata_core": {"geometry": meta.get("geometry"), "position_matching": meta.get("position_matching")},
        "score_rows": len(score_rows),
        "complete_register_cells": sum(1 for r in score_rows if r["kind"] == "register_arm" for c in STABLE_COLUMNS if finite(r.get(c))),
        "register_cell_total": 2 * len(CHECKPOINTS) * len(STABLE_COLUMNS),
        "trajectory_contrasts": traj_rows,
        "files": {"score_rows": rel(OUT / "score_rows.csv"), "contrast_rows": rel(OUT / "contrast_rows.csv"), "trajectory_contrasts": rel(OUT / "trajectory_contrasts.csv"), "summary_json": rel(OUT / "register_max_readout_summary.json"), "summary_md": rel(OUT / "register_max_readout_summary.md")},
    }
    write_json(OUT / "register_max_readout_summary.json", result)

    def fmt(x: Any) -> str:
        return "" if not finite(x) else f"{float(x):.4f}"
    lines = ["# research MAX-register readout", "", result["boundary"], "", f"Data state: `{result['data_state']}`; register stable cells {result['complete_register_cells']}/{result['register_cell_total']}.", "", "## Trajectory contrasts over complete common-window checkpoints", "", "| contrast | complete ck count | cheap6 | exEntity5 | Entity | BLiMP | Supplement | EWoK | COMPS | Reading |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in traj_rows:
        lines.append(f"| {r['contrast']} | {r['complete_checkpoint_count']} | {fmt(r.get('cheap6_mean'))} | {fmt(r.get('exEntity5_mean'))} | {fmt(r.get('Entity_mean'))} | {fmt(r.get('BLiMP_mean'))} | {fmt(r.get('Supplement_mean'))} | {fmt(r.get('EWoK_mean'))} | {fmt(r.get('COMPS_mean'))} | {fmt(r.get('Reading_mean'))} |")
    lines += ["", "## Interpretation", "", "Primary contrast is `child_minus_adult`: positive means removing developmental/speech hurts less or adult-prose removal hurts more; negative means developmental/speech removal has higher opportunity cost. Arm-minus-view locates each pure-register removal arm around the trained proportional MAX view midpoint. Arm/view-minus-clean uses whatever clean anchors are available and must be read with the reported clean spread.", "", f"JSON: `{rel(OUT / 'register_max_readout_summary.json')}`", f"Contrasts: `{rel(OUT / 'contrast_rows.csv')}`"]
    (OUT / "register_max_readout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "data_state": result["data_state"], "complete_register_cells": result["complete_register_cells"], "register_cell_total": result["register_cell_total"], "trajectory_contrasts": traj_rows, "summary_md": rel(OUT / "register_max_readout_summary.md")}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
