#!/usr/bin/env python3
"""research: file-only fixed-budget allocation readout.

This readout turns existing and newly arriving stable-family score payloads into
scientific contrasts for the current BabyLM Strict-Small principle search.  It
is intentionally file-only: no model loading, no training, no evaluation, no
GlobalPIQA, no SuperGLUE, no AoA, no upload, and no leaderboard action.

Quantities
----------
DeBERTa first basin, MAX dose:
  V-R = compact re-expression versus exact repeat.
  V-B = compact re-expression versus same-population sentence breadth.
  B-R = same-population sentence breadth versus exact repeat.
  V-R should equal (V-B)+(B-R) whenever all three scores exist.

RoBERTa MAX geometry:
  V-C = compact re-expression versus clean, same 653,130-row geometry.

DeBERTa MAX geometry clean controls, once trained/scored:
  V-C in seed43022 and seed43122 basins, with ex-Entity summaries.

The script can be rerun after background jobs fill more payloads.  Missing rows
are recorded explicitly rather than imputed.
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
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'

FAMILIES = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
NO_READING = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
EX_ENTITY_NO_READING = ["BLiMP", "Supplement", "EWoK", "COMPS"]
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 101, 10)]

ARMS: dict[str, dict[str, Any]] = {
    "deberta_basin1_view": {
        "architecture": "deberta",
        "seed": 43022,
        "role": "V",
        "root": WS / "data" / "dose_ladder_stable_eval" / "eval" / "per_target",
        "prefix": "dose_max_view",
        "description": "first-basin MAX compact re-expression arm",
    },
    "deberta_basin1_repeat": {
        "architecture": "deberta",
        "seed": 43022,
        "role": "R",
        "root": WS / "data" / "dose_ladder_stable_eval" / "eval" / "per_target",
        "prefix": "dose_max_repeat",
        "description": "first-basin MAX exact-repeat arm",
    },
    "deberta_basin1_breadth": {
        "architecture": "deberta",
        "seed": 43022,
        "role": "B",
        "root": WS / "data" / "breadth_entity_ewok_eval" / "eval" / "per_target",
        "prefix": "max_breadth_seed43022",
        "description": "first-basin MAX same-population sentence-breadth arm",
    },
    "deberta_basin1_clean_maxgeom": {
        "architecture": "deberta",
        "seed": 43022,
        "role": "C",
        "root": WS / "data" / "deberta_maxgeom_clean_stable_eval" / "eval" / "per_target",
        "prefix": "deberta_maxgeom_clean_seed43022",
        "description": "first-basin MAX-geometry clean control, pending training/scoring",
    },
    "deberta_basin2_view": {
        "architecture": "deberta",
        "seed": 43122,
        "role": "V",
        "root": WS / "data" / "second_basin_entity_ewok_eval" / "view_eval" / "per_target",
        "prefix": "second_basin_max_view_seed43122",
        "description": "second-basin MAX compact re-expression arm; currently mainly Entity/EWoK",
    },
    "deberta_basin2_repeat": {
        "architecture": "deberta",
        "seed": 43122,
        "role": "R",
        "root": WS / "data" / "second_basin_entity_ewok_eval" / "repeat_eval" / "per_target",
        "prefix": "second_basin_max_repeat_seed43122",
        "description": "second-basin MAX exact-repeat arm; currently mainly Entity/EWoK",
    },
    "deberta_basin2_clean_maxgeom": {
        "architecture": "deberta",
        "seed": 43122,
        "role": "C",
        "root": WS / "data" / "deberta_maxgeom_clean_stable_eval" / "eval" / "per_target",
        "prefix": "deberta_maxgeom_clean_seed43122",
        "description": "second-basin MAX-geometry clean control, pending training/scoring",
    },
    "roberta_view": {
        "architecture": "roberta",
        "seed": 43022,
        "role": "V",
        "root": WS / "data" / "roberta_viewclean_total_stable_eval" / "eval" / "per_target",
        "prefix": "roberta_viewclean_total_view",
        "description": "RoBERTa MAX compact re-expression arm",
    },
    "roberta_clean": {
        "architecture": "roberta",
        "seed": 43022,
        "role": "C",
        "root": WS / "data" / "roberta_viewclean_total_stable_eval" / "eval" / "per_target",
        "prefix": "roberta_viewclean_total_clean",
        "description": "RoBERTa MAX clean arm at the same 653,130-row geometry",
    },
    "roberta_repeat": {
        "architecture": "roberta",
        "seed": 43022,
        "role": "R",
        "root": WS / "data" / "roberta_maxdose_repeat_eval" / "eval" / "per_target",
        "prefix": "roberta_max_repeat_seed43022",
        "description": "RoBERTa MAX exact-repeat arm; currently late Entity only unless further scored",
    },
}

CONTRASTS = [
    ("deberta_basin1_VminusR", "deberta_basin1_view", "deberta_basin1_repeat"),
    ("deberta_basin1_VminusB", "deberta_basin1_view", "deberta_basin1_breadth"),
    ("deberta_basin1_BminusR", "deberta_basin1_breadth", "deberta_basin1_repeat"),
    ("deberta_basin1_VminusC_maxgeom", "deberta_basin1_view", "deberta_basin1_clean_maxgeom"),
    ("deberta_basin2_VminusR", "deberta_basin2_view", "deberta_basin2_repeat"),
    ("deberta_basin2_VminusC_maxgeom", "deberta_basin2_view", "deberta_basin2_clean_maxgeom"),
    ("roberta_VminusC_maxgeom", "roberta_view", "roberta_clean"),
    ("roberta_VminusR", "roberta_view", "roberta_repeat"),
]

AGGREGATES = {
    "cheap6_no_GlobalPIQA": FAMILIES,
    "cheap5_no_GlobalPIQA_Reading": NO_READING,
    "exEntity5": EX_ENTITY,
    "exEntity4_noReading": EX_ENTITY_NO_READING,
    "EWoK_plus_Entity_sum": ["EWoK", "Entity"],
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


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


def ck_words(ck: str) -> int:
    return int(ck.split("_")[1].rstrip("M")) * 1_000_000


def task_score(payload: dict[str, Any], family: str) -> float | None:
    task = (payload.get("tasks") or {}).get(family)
    if not isinstance(task, dict) or task.get("returncode") != 0:
        return None
    if family == "Reading":
        val = task.get("score")
        if finite(val):
            return float(val)
        scores = task.get("scores") if isinstance(task.get("scores"), dict) else {}
        val = scores.get("Reading") if scores else None
    else:
        val = task.get("score")
    return float(val) if finite(val) else None


def payload_path(arm_key: str, ck: str) -> pathlib.Path:
    cfg = ARMS[arm_key]
    return pathlib.Path(cfg["root"]) / f"{cfg['prefix']}_{ck}.json"


def collect_scores() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for arm_key, cfg in ARMS.items():
        for ck in CHECKPOINTS:
            p = payload_path(arm_key, ck)
            if not p.exists():
                for fam in FAMILIES:
                    missing_rows.append({"arm": arm_key, "checkpoint": ck, "words": ck_words(ck), "family": fam, "reason": "payload_missing", "path": rel(p)})
                continue
            try:
                payload = read_json(p)
            except Exception as exc:
                for fam in FAMILIES:
                    missing_rows.append({"arm": arm_key, "checkpoint": ck, "words": ck_words(ck), "family": fam, "reason": f"payload_unreadable:{exc!r}", "path": rel(p)})
                continue
            for fam in FAMILIES:
                val = task_score(payload, fam)
                row = {
                    "arm": arm_key,
                    "architecture": cfg["architecture"],
                    "seed": cfg["seed"],
                    "role": cfg["role"],
                    "checkpoint": ck,
                    "words": ck_words(ck),
                    "family": fam,
                    "score": val,
                    "payload": rel(p),
                    "description": cfg["description"],
                }
                if val is None:
                    row_m = dict(row)
                    row_m["reason"] = "task_missing_or_no_score"
                    missing_rows.append(row_m)
                else:
                    score_rows.append(row)
    return score_rows, missing_rows


def score_lookup(score_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    return {(r["arm"], r["checkpoint"], r["family"]): float(r["score"]) for r in score_rows if finite(r.get("score"))}


def aggregate(vals: list[float | None]) -> float | None:
    if not vals or any(not finite(v) for v in vals):
        return None
    return statistics.mean([float(v) for v in vals])


def make_contrasts(score_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = score_lookup(score_rows)
    contrast_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for cname, a, b in CONTRASTS:
        for ck in CHECKPOINTS:
            for fam in FAMILIES:
                ka = (a, ck, fam)
                kb = (b, ck, fam)
                if ka in lookup and kb in lookup:
                    contrast_rows.append({
                        "contrast": cname,
                        "arm_a": a,
                        "arm_b": b,
                        "checkpoint": ck,
                        "words": ck_words(ck),
                        "quantity": fam,
                        "delta": lookup[ka] - lookup[kb],
                        "score_a": lookup[ka],
                        "score_b": lookup[kb],
                    })
                else:
                    missing.append({"contrast": cname, "arm_a": a, "arm_b": b, "checkpoint": ck, "words": ck_words(ck), "quantity": fam, "missing_a": ka not in lookup, "missing_b": kb not in lookup})
            for agg_name, fams in AGGREGATES.items():
                vals_a = [lookup.get((a, ck, fam)) for fam in fams]
                vals_b = [lookup.get((b, ck, fam)) for fam in fams]
                agg_a = aggregate(vals_a)
                agg_b = aggregate(vals_b)
                if finite(agg_a) and finite(agg_b):
                    contrast_rows.append({
                        "contrast": cname,
                        "arm_a": a,
                        "arm_b": b,
                        "checkpoint": ck,
                        "words": ck_words(ck),
                        "quantity": agg_name,
                        "delta": float(agg_a) - float(agg_b),
                        "score_a": float(agg_a),
                        "score_b": float(agg_b),
                    })
                else:
                    missing.append({"contrast": cname, "arm_a": a, "arm_b": b, "checkpoint": ck, "words": ck_words(ck), "quantity": agg_name, "missing_a": not finite(agg_a), "missing_b": not finite(agg_b)})
    return contrast_rows, missing


def make_triangle_rows(contrast_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lut = {(r["contrast"], r["checkpoint"], r["quantity"]): float(r["delta"]) for r in contrast_rows if finite(r.get("delta"))}
    rows: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        for q in FAMILIES + list(AGGREGATES.keys()):
            vr = lut.get(("deberta_basin1_VminusR", ck, q))
            vb = lut.get(("deberta_basin1_VminusB", ck, q))
            br = lut.get(("deberta_basin1_BminusR", ck, q))
            if finite(vr) and finite(vb) and finite(br):
                rows.append({
                    "checkpoint": ck,
                    "words": ck_words(ck),
                    "quantity": q,
                    "VminusR": float(vr),
                    "VminusB": float(vb),
                    "BminusR": float(br),
                    "triangle_sum_VB_plus_BR": float(vb) + float(br),
                    "residual": (float(vb) + float(br)) - float(vr),
                    "fraction_of_VminusR_from_BminusR": (float(br) / float(vr)) if abs(float(vr)) > 1e-12 else None,
                })
    return rows


def summarize_contrasts(contrast_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in contrast_rows:
        window = "common10_80" if 10_000_000 <= int(r["words"]) <= 80_000_000 else "late80_100" if 80_000_000 <= int(r["words"]) <= 100_000_000 else "all_available"
        groups.setdefault((r["contrast"], r["quantity"], window), []).append(r)
    out: list[dict[str, Any]] = []
    for (contrast, quantity, window), rows in sorted(groups.items()):
        xs = [float(r["delta"]) for r in rows if finite(r.get("delta"))]
        if not xs:
            continue
        out.append({
            "contrast": contrast,
            "quantity": quantity,
            "window": window,
            "n": len(xs),
            "mean_delta": statistics.mean(xs),
            "min_delta": min(xs),
            "max_delta": max(xs),
            "checkpoints": ";".join(r["checkpoint"] for r in sorted(rows, key=lambda z: int(z["words"]))),
        })
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(x: Any) -> str:
    return "NA" if not finite(x) else f"{float(x):+.4f}"


def selected_lines(summary_rows: list[dict[str, Any]], triangle_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    want = [
        ("deberta_basin1_VminusB", "exEntity5", "late80_100"),
        ("deberta_basin1_BminusR", "exEntity5", "late80_100"),
        ("deberta_basin1_VminusR", "exEntity5", "late80_100"),
        ("deberta_basin1_VminusB", "cheap6_no_GlobalPIQA", "late80_100"),
        ("deberta_basin1_BminusR", "cheap6_no_GlobalPIQA", "late80_100"),
        ("roberta_VminusC_maxgeom", "exEntity5", "late80_100"),
        ("roberta_VminusC_maxgeom", "cheap6_no_GlobalPIQA", "late80_100"),
        ("deberta_basin1_VminusC_maxgeom", "exEntity5", "late80_100"),
        ("deberta_basin2_VminusC_maxgeom", "exEntity5", "late80_100"),
    ]
    lut = {(r["contrast"], r["quantity"], r["window"]): r for r in summary_rows}
    for key in want:
        r = lut.get(key)
        if r:
            lines.append(f"- {key[0]} {key[1]} {key[2]}: n={r['n']} mean={fmt(r['mean_delta'])} range=[{fmt(r['min_delta'])}, {fmt(r['max_delta'])}] checkpoints={r['checkpoints']}")
        else:
            lines.append(f"- {key[0]} {key[1]} {key[2]}: not yet complete")
    tri = [r for r in triangle_rows if r["quantity"] in {"Entity", "exEntity5", "cheap6_no_GlobalPIQA"} and r["checkpoint"] in {"chck_80M", "chck_90M", "chck_100M"}]
    if tri:
        lines.append("")
        lines.append("Triangle rows now complete for selected quantities:")
        for r in tri:
            lines.append(f"- {r['checkpoint']} {r['quantity']}: V-R={fmt(r['VminusR'])}, V-B={fmt(r['VminusB'])}, B-R={fmt(r['BminusR'])}, residual={fmt(r['residual'])}")
    else:
        lines.append("")
        lines.append("No broad DeBERTa V/B/R triangle rows are complete beyond Entity yet.")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(WS / "data" / "fixed_budget_allocation_readout"))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    score_rows, raw_missing = collect_scores()
    contrast_rows, contrast_missing = make_contrasts(score_rows)
    triangle_rows = make_triangle_rows(contrast_rows)
    summary_rows = summarize_contrasts(contrast_rows)

    score_csv = out_dir / "score_rows.csv"
    raw_missing_csv = out_dir / "missing_score_rows.csv"
    contrast_csv = out_dir / "contrast_rows.csv"
    contrast_missing_csv = out_dir / "missing_contrast_rows.csv"
    triangle_csv = out_dir / "triangle_rows.csv"
    summary_csv = out_dir / "summary_rows.csv"
    write_csv(score_csv, score_rows)
    write_csv(raw_missing_csv, raw_missing)
    write_csv(contrast_csv, contrast_rows)
    write_csv(contrast_missing_csv, contrast_missing)
    write_csv(triangle_csv, triangle_rows)
    write_csv(summary_csv, summary_rows)

    complete_broad_triangle = [r for r in triangle_rows if r["quantity"] in {"exEntity5", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading"}]
    roberta_vc = [r for r in summary_rows if r["contrast"] == "roberta_VminusC_maxgeom" and r["quantity"] in {"exEntity5", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading"}]
    clean_deberta_vc = [r for r in summary_rows if r["contrast"] in {"deberta_basin1_VminusC_maxgeom", "deberta_basin2_VminusC_maxgeom"}]
    summary = {
        "status": "FIXED_BUDGET_ALLOCATION_READOUT_INCOMPLETE" if raw_missing or contrast_missing else "FIXED_BUDGET_ALLOCATION_READOUT_COMPLETE",
        "created_utc": now(),
        "score_row_count": len(score_rows),
        "missing_score_row_count": len(raw_missing),
        "contrast_row_count": len(contrast_rows),
        "missing_contrast_row_count": len(contrast_missing),
        "triangle_row_count": len(triangle_rows),
        "broad_triangle_row_count": len(complete_broad_triangle),
        "roberta_geometry_clean_summary_count": len(roberta_vc),
        "deberta_maxgeom_clean_summary_count": len(clean_deberta_vc),
        "scientific_reading": "Use V-C for fixed-budget allocation, V-B for compact re-expression specificity, B-R for same-population breadth versus duplicate recurrence, and the exact V-R=(V-B)+(B-R) identity to avoid treating Entity aggregate movement as a single mechanism.",
        "files": {
            "score_rows_csv": rel(score_csv),
            "missing_score_rows_csv": rel(raw_missing_csv),
            "contrast_rows_csv": rel(contrast_csv),
            "missing_contrast_rows_csv": rel(contrast_missing_csv),
            "triangle_rows_csv": rel(triangle_csv),
            "summary_rows_csv": rel(summary_csv),
            "summary_json": rel(out_dir / "fixed_budget_allocation_readout_summary.json"),
            "summary_md": rel(out_dir / "fixed_budget_allocation_readout_summary.md"),
        },
        "no_model_loading_training_evaluation_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    write_json(out_dir / "fixed_budget_allocation_readout_summary.json", summary)
    lines = [
        "# research fixed-budget allocation readout",
        "",
        summary["scientific_reading"],
        "",
        f"Score rows present: {len(score_rows)}; missing score rows: {len(raw_missing)}.",
        f"Contrast rows present: {len(contrast_rows)}; missing contrast rows: {len(contrast_missing)}.",
        f"Triangle rows present: {len(triangle_rows)}; broad triangle rows: {len(complete_broad_triangle)}.",
        "",
        "## Selected quantities",
    ]
    lines.extend(selected_lines(summary_rows, triangle_rows))
    lines.extend([
        "",
        "## Files",
    ])
    for k, v in summary["files"].items():
        lines.append(f"- {k}: `{v}`")
    (out_dir / "fixed_budget_allocation_readout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "score_row_count": len(score_rows),
        "missing_score_row_count": len(raw_missing),
        "contrast_row_count": len(contrast_rows),
        "triangle_row_count": len(triangle_rows),
        "summary_json": summary["files"]["summary_json"],
        "summary_md": summary["files"]["summary_md"],
        "no_model_loading_training_evaluation_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
