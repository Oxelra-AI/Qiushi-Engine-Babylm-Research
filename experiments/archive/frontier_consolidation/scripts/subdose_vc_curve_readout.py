#!/usr/bin/env python3
"""research: assemble the geometry-matched sub-dose V-C curve.

Reads stable-family per_target JSONs from:
  * research clean-anchor common-window scorer (two MAX-geometry clean basins)
  * research sub-dose common-window scorer (quarter/half/full_1x once scored)
  * research existing MAX view scorer (same MAX row geometry)

Outputs a row-matched dose curve over chck_10M..chck_80M.  The old 1.82x point is
not included in the strict curve because research has 65,041 rows whereas the MAX
geometry has 65,313 rows.  It can be compared separately as historical amount
context, not as a strict row-matched interpolation point.

No model loading, training, official evaluation, GPU work, GlobalPIQA, SuperGLUE,
AoA, upload, or leaderboard action.
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
OUT = WS / "data/subdose_vc_curve_readout"
CLEAN_EVAL = WS / "data/clean_anchor_commonwindow_eval/eval/per_target"
SUBDOSE_EVAL = WS / "data/subdose_commonwindow_eval/eval/per_target"
MAX_EVAL = WS / "data/dose_ladder_stable_eval/eval/per_target"
META280 = WS / "data/subdose_ladder_maxgeom_pools/subdose_ladder_metadata.json"
DISPLACED = WS / "data/displaced_clean_block_readout/displaced_clean_block_summary.json"
META = WS / "data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json"
META = WS / "data/dose_1p82x_rowholdout_pools/dose1p82_rowholdout_metadata.json"

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


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def metric_from_payload(payload: dict[str, Any], col: str) -> float | None:
    ss = payload.get("stable_family_scores") or {}
    if finite(ss.get(col)):
        return float(ss[col])
    rec = (payload.get("tasks") or {}).get(col) or {}
    if col == "Reading":
        scores = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
        val = scores.get("Reading") if scores else rec.get("score")
    else:
        val = rec.get("score")
    return float(val) if finite(val) else None


def load_payload(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def row_scores(label: str, kind: str, checkpoint: str, path: pathlib.Path, seed: int | None, rho: float, words_admitted: int, docs: int | None, displaced_words: int | None) -> dict[str, Any]:
    payload = load_payload(path)
    rec: dict[str, Any] = {"label": label, "kind": kind, "checkpoint": checkpoint, "seed": seed, "rho": rho, "words_admitted": words_admitted, "unique_docs": docs, "displaced_clean_words": displaced_words, "path": rel(path), "exists": path.exists()}
    if payload is None:
        rec["complete_stable6"] = False
        return rec
    vals: dict[str, float | None] = {}
    for c in STABLE_COLUMNS:
        vals[c] = metric_from_payload(payload, c)
        rec[c] = vals[c]
    rec["cheap6"] = statistics.mean([float(vals[c]) for c in STABLE_COLUMNS]) if all(finite(vals[c]) for c in STABLE_COLUMNS) else None
    rec["cheap5"] = statistics.mean([float(vals[c]) for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]]) if all(finite(vals[c]) for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]) else None
    rec["exEntity5"] = statistics.mean([float(vals[c]) for c in EX_ENTITY_COLUMNS]) if all(finite(vals[c]) for c in EX_ENTITY_COLUMNS) else None
    rec["complete_stable6"] = all(finite(vals[c]) for c in STABLE_COLUMNS)
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m280 = read_json(META280)
    displaced = read_json(DISPLACED) if DISPLACED.exists() else {}
    m256 = read_json(META)
    m258 = read_json(META) if META.exists() else {}

    dose_meta = {
        "quarter_1x": {"rho": m280["doses"]["quarter_1x"]["rho"], "words": m280["doses"]["quarter_1x"]["active_total_pw"], "docs": m280["doses"]["quarter_1x"]["unique_docs"], "prefix": "subdose_quarter_1x_seed43022"},
        "half_1x": {"rho": m280["doses"]["half_1x"]["rho"], "words": m280["doses"]["half_1x"]["active_total_pw"], "docs": m280["doses"]["half_1x"]["unique_docs"], "prefix": "subdose_half_1x_seed43022"},
        "full_1x_maxgeom": {"rho": m280["doses"]["full_1x"]["rho"], "words": m280["doses"]["full_1x"]["active_total_pw"], "docs": m280["doses"]["full_1x"]["unique_docs"], "prefix": "subdose_full_1x_seed43022"},
        "max_2p64x": {"rho": m256["dose"]["changed_budget_fraction_of_10M"], "words": m256["pair_summary"]["pair_words"], "docs": m256["pair_summary"]["unique_docs"], "prefix": "dose_max_view"},
    }
    displaced_by_label = {
        "quarter_1x": (displaced.get("cumulative_displaced_clean", {}).get("quarter_1x", {}) or {}).get("words"),
        "half_1x": (displaced.get("cumulative_displaced_clean", {}).get("half_1x", {}) or {}).get("words"),
        "full_1x_maxgeom": (displaced.get("cumulative_displaced_clean", {}).get("full_1x", {}) or {}).get("words"),
        "max_2p64x": (displaced.get("cumulative_displaced_clean", {}).get("max_dose2p64", {}) or {}).get("words"),
    }

    score_rows: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        for seed, pref in [(43022, "deberta_maxgeom_clean_seed43022"), (43122, "deberta_maxgeom_clean_seed43122")]:
            score_rows.append(row_scores(f"clean_seed{seed}", "clean", ck, CLEAN_EVAL / f"{pref}_{ck}.json", seed, 0.0, 0, 0, 0))
        for label, dm in dose_meta.items():
            base = SUBDOSE_EVAL if label != "max_2p64x" else MAX_EVAL
            score_rows.append(row_scores(label, "view", ck, base / f"{dm['prefix']}_{ck}.json", 43022, float(dm["rho"]), int(dm["words"]), int(dm["docs"]), displaced_by_label.get(label)))

    # Compute clean mean and V-C deltas for each checkpoint/dose when possible.
    clean_by_ck: dict[str, dict[str, Any]] = {}
    for ck in CHECKPOINTS:
        crs = [r for r in score_rows if r["kind"] == "clean" and r["checkpoint"] == ck]
        clean_by_ck[ck] = {}
        for metric in STABLE_COLUMNS + ["cheap6", "cheap5", "exEntity5"]:
            vals = [r.get(metric) for r in crs if finite(r.get(metric))]
            clean_by_ck[ck][metric + "_mean"] = statistics.mean([float(v) for v in vals]) if vals else None
            clean_by_ck[ck][metric + "_spread"] = (max([float(v) for v in vals]) - min([float(v) for v in vals])) if len(vals) >= 2 else None
            clean_by_ck[ck][metric + "_n"] = len(vals)

    delta_rows: list[dict[str, Any]] = []
    for r in score_rows:
        if r["kind"] != "view":
            continue
        ck = r["checkpoint"]
        rec = {"label": r["label"], "checkpoint": ck, "rho": r["rho"], "words_admitted": r["words_admitted"], "unique_docs": r["unique_docs"], "displaced_clean_words": r["displaced_clean_words"], "score_path": r["path"]}
        complete = True
        for metric in STABLE_COLUMNS + ["cheap6", "cheap5", "exEntity5"]:
            cv = clean_by_ck[ck].get(metric + "_mean")
            rv = r.get(metric)
            rec[metric + "_view"] = rv
            rec[metric + "_clean_mean"] = cv
            rec[metric + "_clean_spread"] = clean_by_ck[ck].get(metric + "_spread")
            rec[metric + "_vc_delta"] = float(rv) - float(cv) if finite(rv) and finite(cv) else None
            rec[metric + "_clean_n"] = clean_by_ck[ck].get(metric + "_n")
            if metric in STABLE_COLUMNS and not (finite(rv) and finite(cv)):
                complete = False
        rec["complete_stable6_delta"] = complete
        delta_rows.append(rec)

    # Trajectory means over common complete window.
    traj_rows: list[dict[str, Any]] = []
    for label in ["quarter_1x", "half_1x", "full_1x_maxgeom", "max_2p64x"]:
        group = [r for r in delta_rows if r["label"] == label]
        rec: dict[str, Any] = {"label": label}
        if group:
            rec.update({k: group[0][k] for k in ["rho", "words_admitted", "unique_docs", "displaced_clean_words"]})
        complete_cks = [r["checkpoint"] for r in group if r.get("complete_stable6_delta")]
        rec["complete_checkpoints"] = complete_cks
        rec["complete_checkpoint_count"] = len(complete_cks)
        for metric in STABLE_COLUMNS + ["cheap6", "cheap5", "exEntity5"]:
            vals = [r.get(metric + "_vc_delta") for r in group if r["checkpoint"] in complete_cks and finite(r.get(metric + "_vc_delta"))]
            rec[metric + "_vc_mean"] = statistics.mean([float(v) for v in vals]) if vals else None
            rec[metric + "_vc_min"] = min([float(v) for v in vals]) if vals else None
            rec[metric + "_vc_max"] = max([float(v) for v in vals]) if vals else None
        traj_rows.append(rec)

    # Save rows.
    fieldnames = sorted({k for r in score_rows for k in r.keys()})
    with (OUT / "raw_score_rows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(score_rows)
    dfields = sorted({k for r in delta_rows for k in r.keys()})
    with (OUT / "vc_delta_rows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dfields); w.writeheader(); w.writerows(delta_rows)
    tfields = sorted({k for r in traj_rows for k in r.keys()})
    with (OUT / "vc_trajectory_means.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tfields); w.writeheader(); w.writerows(traj_rows)

    strict_geometry = {
        "included": "clean, quarter_1x, half_1x, full_1x_maxgeom, max_2p64x all use 65,313-row MAX geometry",
        "excluded_context": {
            "dose1p82_rows": (m258.get("audit", {}).get("row_counts", {}) or {}).get("compact_view_dose1p82x"),
            "max_rows": (m256.get("audit", {}).get("row_counts", {}) or {}).get("compact_view_dose2p64x"),
            "reason": "research 1.82x has a different 65,041-row sequence, so it is not part of the strict row-matched curve.",
        },
    }
    result = {
        "status": "SUBDOSE_VC_CURVE_READOUT_COMPLETE",
        "created_utc": now(),
        "boundary": "File-only readout; no model loading, training, official evaluation, GPU work, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.",
        "strict_geometry": strict_geometry,
        "score_rows": len(score_rows),
        "delta_rows": len(delta_rows),
        "clean_reference_by_checkpoint": clean_by_ck,
        "trajectory_means": traj_rows,
        "files": {"raw_score_rows": rel(OUT / "raw_score_rows.csv"), "vc_delta_rows": rel(OUT / "vc_delta_rows.csv"), "vc_trajectory_means": rel(OUT / "vc_trajectory_means.csv"), "summary_json": rel(OUT / "subdose_vc_curve_summary.json"), "summary_md": rel(OUT / "subdose_vc_curve_summary.md")},
    }
    write_json(OUT / "subdose_vc_curve_summary.json", result)

    lines = ["# research geometry-matched sub-dose V-C curve readout", "", result["boundary"], "", "## Geometry", "", strict_geometry["included"], "", f"Excluded 1.82x context: rows={strict_geometry['excluded_context']['dose1p82_rows']} versus MAX rows={strict_geometry['excluded_context']['max_rows']}; {strict_geometry['excluded_context']['reason']}", "", "## Trajectory means over complete common-window checkpoints", "", "| label | rho | admitted words | docs | complete ck count | cheap6 V-C | ex-Entity V-C | Entity V-C | clean spread cheap6 mean window |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rec in traj_rows:
        # Average the clean cheap6 spread over checkpoints where this label is complete.
        spreads = []
        for ck in rec.get("complete_checkpoints", []):
            sp = clean_by_ck[ck].get("cheap6_spread")
            if finite(sp): spreads.append(float(sp))
        mean_spread = statistics.mean(spreads) if spreads else None
        def fmt(x: Any) -> str:
            return "" if not finite(x) else f"{float(x):.4f}"
        lines.append(f"| {rec['label']} | {fmt(rec.get('rho'))} | {rec.get('words_admitted')} | {rec.get('unique_docs')} | {rec.get('complete_checkpoint_count')} | {fmt(rec.get('cheap6_vc_mean'))} | {fmt(rec.get('exEntity5_vc_mean'))} | {fmt(rec.get('Entity_vc_mean'))} | {fmt(mean_spread)} |")
    lines += ["", f"JSON: `{rel(OUT / 'subdose_vc_curve_summary.json')}`", f"Deltas: `{rel(OUT / 'vc_delta_rows.csv')}`"]
    (OUT / "subdose_vc_curve_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "trajectory_means": traj_rows, "summary_md": rel(OUT / "subdose_vc_curve_summary.md"), "strict_geometry": strict_geometry}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
