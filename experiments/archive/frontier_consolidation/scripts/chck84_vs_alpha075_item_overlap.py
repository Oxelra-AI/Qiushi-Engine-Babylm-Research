#!/usr/bin/env python3
"""research: compare trained chck_84M endpoint branch with coherent86 alpha=0.75.

CPU/file-only analysis.  It uses already-produced official-compatible prediction
files for the scale1.75 seed43022 reference checkpoints and the truthful
coherent86 alpha0.75 carrier.  The purpose is to separate endpoint arithmetic
from mechanism: chck_84M is an ordinary legal training checkpoint on the same
trajectory, while alpha0.75 is frozen-anchor private-pathway interpolation.  The
script compares the seven cheap columns at item/subtask/family level and records
how much of each endpoint's movement from chck_82M is shared or disjoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

DEFAULT_SELECTED_DIR = pathlib.Path(
    "experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M"
)
DEFAULT_ALPHA_MANIFEST = pathlib.Path(
    "experiments/archive/frontier_consolidation/data/truthful_private_scale_carriers/coherent86_alpha0p75"
    "truthful_coherent86_alpha0p75_carrier_manifest.json"
)
DEFAULT_OUT_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/chck84_vs_alpha075_item_overlap")
SCRIPT = pathlib.Path("experiments/archive/frontier_consolidation/scripts/chck84_item_movement_analyzer.py")

CLASSIFICATION_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
NONVOL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
CHEAP5_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
RELATION_STATE_COLUMNS = ["EWoK", "Entity"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def json_sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_sanitize(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def safe_float(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def fmean(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return statistics.fmean(xs) if xs else None


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else float("nan")


def short(x: Any, n: int = 220) -> str:
    s = "" if x is None else str(x).replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def load_step186_module():
    path = SCRIPT
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("base", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["base"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def load_selected_rows(selected_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with (selected_dir / "selected_trajectory.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("endpoint"):
                rows[r["endpoint"]] = r
    return rows


def load_reference_items(base: Any, selected_dir: pathlib.Path, endpoint: str) -> list[dict[str, Any]]:
    payload_path = selected_dir / "eval" / "per_target" / f"scale1p75_seed43022_reference_{endpoint}.json"
    payload = read_json(payload_path)
    rows = base.endpoint_items(payload)
    # add path provenance to returned rows only through summary, not rowwise
    return rows


def load_alpha_items(base: Any, carrier_path: pathlib.Path) -> list[dict[str, Any]]:
    carrier = read_json(carrier_path)
    rows: list[dict[str, Any]] = []
    rows += base.extract_blimp_like(carrier["blimp"], base.CURRENT_FULL / "blimp_filtered", "BLiMP")
    rows += base.extract_blimp_like(carrier["blimp_supplement"], base.CURRENT_FULL / "supplement_filtered", "Supplement")
    rows += base.extract_ewok(carrier["ewok"], base.CURRENT_FULL / "ewok_filtered")
    rows += base.extract_entity(carrier["entity_tracking_filtered"], base.CURRENT_FULL / "entity_tracking")
    rows += base.extract_comps(carrier["comps"], base.CURRENT_FULL / "comps")
    rows += base.extract_global_piqa(
        carrier["global_piqa_parallel"],
        base.LEGACY_FULL / "global_piqa_parallel" / "eng_latn.jsonl",
        "global_piqa_parallel",
    )
    rows += base.extract_global_piqa(
        carrier["global_piqa_nonparallel"],
        base.LEGACY_FULL / "global_piqa_nonparallel" / "eng_latn.jsonl",
        "global_piqa_nonparallel",
    )
    return rows


def merge_items(label_to_rows: dict[str, list[dict[str, Any]]], suffix: dict[str, str]) -> list[dict[str, Any]]:
    key_sets = {lab: {r["item_key"] for r in rows} for lab, rows in label_to_rows.items()}
    ref_keys = next(iter(key_sets.values()))
    for lab, keys in key_sets.items():
        if keys != ref_keys:
            raise RuntimeError(f"Item-key mismatch for {lab}: {len(keys)} vs {len(ref_keys)}")
    base_meta: dict[str, dict[str, Any]] = {}
    per = {lab: {r["item_key"]: r for r in rows} for lab, rows in label_to_rows.items()}
    for r in next(iter(label_to_rows.values())):
        base_meta[r["item_key"]] = {k: v for k, v in r.items() if k not in {"pred", "correct"}}
    merged: list[dict[str, Any]] = []
    for key in sorted(ref_keys):
        row = dict(base_meta[key])
        for lab in label_to_rows:
            rr = per[lab][key]
            s = suffix[lab]
            row[f"correct_{s}"] = int(bool(rr["correct"]))
            row[f"pred_{s}"] = rr["pred"]
        merged.append(row)
    return merged


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(f, "") for f in fields)].append(r)
    return out


def summarise_groups(rows: list[dict[str, Any]], fields: list[str], suffixes: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, rs in group_rows(rows, fields).items():
        rec = {fields[i]: key[i] for i in range(len(fields))}
        rec["n_items"] = len(rs)
        for s in suffixes:
            rec[f"score_{s}"] = pct(sum(r[f"correct_{s}"] for r in rs), len(rs))
        for a, b in [("82", "84"), ("82", "alpha075"), ("84", "alpha075")]:
            if a in suffixes and b in suffixes:
                rec[f"delta_{b}_minus_{a}"] = rec[f"score_{b}"] - rec[f"score_{a}"]
                rec[f"gains_{a}_to_{b}"] = sum(r[f"correct_{a}"] == 0 and r[f"correct_{b}"] == 1 for r in rs)
                rec[f"losses_{a}_to_{b}"] = sum(r[f"correct_{a}"] == 1 and r[f"correct_{b}"] == 0 for r in rs)
                rec[f"net_items_{a}_to_{b}"] = rec[f"gains_{a}_to_{b}"] - rec[f"losses_{a}_to_{b}"]
        out.append(rec)
    return sorted(out, key=lambda r: tuple(str(r.get(f, "")) for f in fields))


def official_like_scores_from_subtasks(subtask_rows: list[dict[str, Any]], suffixes: list[str]) -> dict[str, dict[str, Any]]:
    by_col: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in subtask_rows:
        by_col[r["column"]].append(r)
    out: dict[str, dict[str, Any]] = {}
    for col, rows in by_col.items():
        rec: dict[str, Any] = {"n_subtasks": len(rows), "n_raw_items": sum(int(r["n_items"]) for r in rows)}
        for s in suffixes:
            rec[f"score_{s}"] = fmean([safe_float(r.get(f"score_{s}")) for r in rows])
        for a, b in [("82", "84"), ("82", "alpha075"), ("84", "alpha075")]:
            if a in suffixes and b in suffixes:
                rec[f"delta_{b}_minus_{a}"] = rec[f"score_{b}"] - rec[f"score_{a}"]
        out[col] = rec
    return out


def aggregate_scores(column_scores: dict[str, dict[str, Any]], reading_scores: dict[str, float]) -> dict[str, Any]:
    by_suffix: dict[str, dict[str, float]] = defaultdict(dict)
    for col in CLASSIFICATION_COLUMNS:
        for s in ["82", "84", "alpha075"]:
            by_suffix[s][col] = float(column_scores[col][f"score_{s}"])
    for s, score in reading_scores.items():
        by_suffix[s]["Reading"] = float(score)
    out: dict[str, Any] = {}
    for s, scores in by_suffix.items():
        out[s] = {
            "scores": scores,
            "cheap7": fmean([scores.get(c) for c in CHEAP_COLUMNS]),
            "cheap6_no_globalpiqa": fmean([scores.get(c) for c in NONVOL_COLUMNS]),
            "cheap5_no_globalpiqa_reading": fmean([scores.get(c) for c in CHEAP5_COLUMNS]),
            "relation_state_ewok_entity": fmean([scores.get(c) for c in RELATION_STATE_COLUMNS]),
            "syntax_surface_blimp_comps": fmean([scores.get(c) for c in ["BLiMP", "COMPS"]]),
        }
    for metric in ["cheap7", "cheap6_no_globalpiqa", "cheap5_no_globalpiqa_reading", "relation_state_ewok_entity", "syntax_surface_blimp_comps"]:
        out.setdefault("deltas", {})[f"84_minus_82_{metric}"] = out["84"][metric] - out["82"][metric]
        out["deltas"][f"alpha075_minus_82_{metric}"] = out["alpha075"][metric] - out["82"][metric]
        out["deltas"][f"alpha075_minus_84_{metric}"] = out["alpha075"][metric] - out["84"][metric]
    return out


def overlap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for col in CLASSIFICATION_COLUMNS + ["ALL"]:
        rs = rows if col == "ALL" else [r for r in rows if r["column"] == col]
        keys = {r["item_key"] for r in rs}
        gain84 = {r["item_key"] for r in rs if r["correct_82"] == 0 and r["correct_84"] == 1}
        gaina = {r["item_key"] for r in rs if r["correct_82"] == 0 and r["correct_alpha075"] == 1}
        loss84 = {r["item_key"] for r in rs if r["correct_82"] == 1 and r["correct_84"] == 0}
        lossa = {r["item_key"] for r in rs if r["correct_82"] == 1 and r["correct_alpha075"] == 0}
        a_not84 = {r["item_key"] for r in rs if r["correct_alpha075"] == 1 and r["correct_84"] == 0}
        u84_nota = {r["item_key"] for r in rs if r["correct_84"] == 1 and r["correct_alpha075"] == 0}
        def jac(a: set[str], b: set[str]) -> float | None:
            return len(a & b) / len(a | b) if (a or b) else None
        summary[col] = {
            "n_items": len(rs),
            "anchor_correct": sum(r["correct_82"] == 1 for r in rs),
            "chck84_correct": sum(r["correct_84"] == 1 for r in rs),
            "alpha075_correct": sum(r["correct_alpha075"] == 1 for r in rs),
            "gain84_vs82": len(gain84),
            "gain_alpha075_vs82": len(gaina),
            "shared_gains_vs82": len(gain84 & gaina),
            "gain_jaccard_vs82": jac(gain84, gaina),
            "chck84_unique_gains_vs82": len(gain84 - gaina),
            "alpha075_unique_gains_vs82": len(gaina - gain84),
            "loss84_vs82": len(loss84),
            "loss_alpha075_vs82": len(lossa),
            "shared_losses_vs82": len(loss84 & lossa),
            "loss_jaccard_vs82": jac(loss84, lossa),
            "chck84_unique_losses_vs82": len(loss84 - lossa),
            "alpha075_unique_losses_vs82": len(lossa - loss84),
            "alpha075_correct_chck84_wrong": len(a_not84),
            "chck84_correct_alpha075_wrong": len(u84_nota),
            "net_alpha075_minus_chck84_items": len(a_not84) - len(u84_nota),
        }
    return summary


def disagreement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        c84 = r["correct_84"]
        ca = r["correct_alpha075"]
        if c84 == ca:
            continue
        if c84 and not ca:
            direction = "chck84_correct_alpha075_wrong"
        else:
            direction = "alpha075_correct_chck84_wrong"
        out.append({
            "direction": direction,
            "anchor_correct_82": r["correct_82"],
            "column": r["column"],
            "subtask": r["subtask"],
            "subgroup": r.get("subgroup", ""),
            "fine_group": r.get("fine_group", ""),
            "item_key": r["item_key"],
            "pred_id": r.get("pred_id", ""),
            "gold": short(r.get("gold")),
            "pred_82": short(r.get("pred_82")),
            "pred_84": short(r.get("pred_84")),
            "pred_alpha075": short(r.get("pred_alpha075")),
            "text_a": r.get("text_a", ""),
            "text_b": r.get("text_b", ""),
        })
    return out


def top_by_delta(rows: list[dict[str, Any]], field: str, n: int = 12, reverse: bool = True) -> list[dict[str, Any]]:
    vals = [r for r in rows if isinstance(r.get(field), (int, float)) and math.isfinite(float(r[field]))]
    return sorted(vals, key=lambda r: float(r[field]), reverse=reverse)[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected-dir", default=str(DEFAULT_SELECTED_DIR))
    ap.add_argument("--alpha-manifest", default=str(DEFAULT_ALPHA_MANIFEST))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    selected_dir = pathlib.Path(args.selected_dir)
    alpha_manifest_path = pathlib.Path(args.alpha_manifest)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base = load_step186_module()
    selected_rows = load_selected_rows(selected_dir)
    alpha_manifest = read_json(alpha_manifest_path)
    alpha_carrier_path = pathlib.Path(alpha_manifest["carrier_path"])

    ref82 = load_reference_items(base, selected_dir, "chck_82M")
    ref84 = load_reference_items(base, selected_dir, "chck_84M")
    alpha = load_alpha_items(base, alpha_carrier_path)
    label_to_rows = {"chck_82M": ref82, "chck_84M": ref84, "alpha075": alpha}
    suffix = {"chck_82M": "82", "chck_84M": "84", "alpha075": "alpha075"}
    merged = merge_items(label_to_rows, suffix)

    suffixes = ["82", "84", "alpha075"]
    subtask = summarise_groups(merged, ["column", "subtask"], suffixes)
    subgroup = summarise_groups(merged, ["column", "subgroup"], suffixes)
    fine = summarise_groups(merged, ["column", "subtask", "fine_group"], suffixes)
    col_scores = official_like_scores_from_subtasks(subtask, suffixes)

    reading_scores = {
        "82": safe_float(selected_rows["chck_82M"].get("Reading")),
        "84": safe_float(selected_rows["chck_84M"].get("Reading")),
        "alpha075": safe_float(alpha_manifest["score_arithmetic_candidate_native"]["scores"].get("Reading")),
    }
    aggregates = aggregate_scores(col_scores, {k: float(v) for k, v in reading_scores.items() if v is not None})
    overlap = overlap_summary(merged)
    disagreements = disagreement_rows(merged)

    by_col_churn: dict[str, Any] = {}
    for col in CLASSIFICATION_COLUMNS:
        rs = [r for r in merged if r["column"] == col]
        by_col_churn[col] = overlap[col]

    # Write detailed tables.
    write_csv(out_dir / "classification_item_records.csv", merged)
    write_csv(out_dir / "chck84_alpha075_disagreements.csv", disagreements)
    write_csv(out_dir / "subtask_overlap_and_movement.csv", subtask)
    write_csv(out_dir / "subgroup_overlap_and_movement.csv", subgroup)
    write_csv(out_dir / "fine_group_overlap_and_movement.csv", fine)

    top_alpha_over_84 = {col: top_by_delta([r for r in subtask if r["column"] == col], "delta_alpha075_minus_84", 8, True) for col in CLASSIFICATION_COLUMNS}
    top_84_over_alpha = {col: top_by_delta([r for r in subtask if r["column"] == col], "delta_alpha075_minus_84", 8, False) for col in CLASSIFICATION_COLUMNS}

    summary = {
        "status": "CHCK84_VS_ALPHA075_ITEM_OVERLAP",
        "created_utc": now_utc(),
        "selected_dir": str(selected_dir),
        "alpha_manifest": str(alpha_manifest_path),
        "alpha_carrier_path": str(alpha_carrier_path),
        "alpha_carrier_sha256": sha256_file(alpha_carrier_path),
        "n_classification_items": len(merged),
        "n_disagreement_items_84_vs_alpha075": len(disagreements),
        "official_like_classification_scores_from_predictions": col_scores,
        "reading_scores": reading_scores,
        "aggregate_scores": aggregates,
        "raw_item_overlap_vs_chck82": overlap,
        "top_subtasks_alpha075_over_chck84": top_alpha_over_84,
        "top_subtasks_chck84_over_alpha075": top_84_over_alpha,
        "output_files": {},
        "elapsed_sec": round(time.time() - t0, 2),
    }
    for name in [
        "classification_item_records.csv",
        "chck84_alpha075_disagreements.csv",
        "subtask_overlap_and_movement.csv",
        "subgroup_overlap_and_movement.csv",
        "fine_group_overlap_and_movement.csv",
    ]:
        p = out_dir / name
        summary["output_files"][name] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}

    lines: list[str] = []
    lines.append("# research `chck_84M` versus coherent86 alpha0.75 cheap-task overlap\n\n")
    lines.append("CPU/file-only comparison of two existing legal endpoint functions. `chck_84M` is an ordinary checkpoint on the trained scale1.75 trajectory; coherent86 alpha0.75 is a frozen-anchor private-pathway interpolation. This analysis does not evaluate SuperGLUE, does not train, and does not submit.\n\n")
    lines.append("## Aggregate cheap columns\n\n")
    lines.append("| endpoint | cheap7 | cheap6 no GlobalPIQA | cheap5 no GlobalPIQA/Reading | EWoK+Entity | BLiMP+COMPS |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for s, label in [("82", "chck82"), ("84", "chck84"), ("alpha075", "alpha0.75")]:
        a = aggregates[s]
        lines.append(f"| {label} | {a['cheap7']:.6f} | {a['cheap6_no_globalpiqa']:.6f} | {a['cheap5_no_globalpiqa_reading']:.6f} | {a['relation_state_ewok_entity']:.6f} | {a['syntax_surface_blimp_comps']:.6f} |\n")
    d = aggregates["deltas"]
    lines.append("\nDeltas alpha0.75 minus chck84: ")
    lines.append(f"cheap7 {d['alpha075_minus_84_cheap7']:+.6f}, cheap6-no-GP {d['alpha075_minus_84_cheap6_no_globalpiqa']:+.6f}, cheap5-no-GP/Reading {d['alpha075_minus_84_cheap5_no_globalpiqa_reading']:+.6f}, EWoK+Entity {d['alpha075_minus_84_relation_state_ewok_entity']:+.6f}, BLiMP+COMPS {d['alpha075_minus_84_syntax_surface_blimp_comps']:+.6f}.\n\n")

    lines.append("## Per-column scores\n\n")
    lines.append("| column | chck82 | chck84 | alpha0.75 | 84-82 | alpha-82 | alpha-84 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for col in CLASSIFICATION_COLUMNS:
        c = col_scores[col]
        lines.append(f"| {col} | {c['score_82']:.4f} | {c['score_84']:.4f} | {c['score_alpha075']:.4f} | {c['delta_84_minus_82']:+.4f} | {c['delta_alpha075_minus_82']:+.4f} | {c['delta_alpha075_minus_84']:+.4f} |\n")
    lines.append(f"| Reading | {reading_scores['82']:.4f} | {reading_scores['84']:.4f} | {reading_scores['alpha075']:.4f} | {reading_scores['84']-reading_scores['82']:+.4f} | {reading_scores['alpha075']-reading_scores['82']:+.4f} | {reading_scores['alpha075']-reading_scores['84']:+.4f} |\n")

    lines.append("\n## Raw item overlap relative to chck82\n\n")
    lines.append("| column | gain84 | gain_alpha | shared gains | gain Jaccard | loss84 | loss_alpha | shared losses | alpha correct / 84 wrong | 84 correct / alpha wrong | net alpha-84 items |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for col in CLASSIFICATION_COLUMNS + ["ALL"]:
        o = overlap[col]
        gj = o['gain_jaccard_vs82']
        lines.append(
            f"| {col} | {o['gain84_vs82']} | {o['gain_alpha075_vs82']} | {o['shared_gains_vs82']} | "
            f"{(gj if gj is not None else float('nan')):.4f} | {o['loss84_vs82']} | {o['loss_alpha075_vs82']} | {o['shared_losses_vs82']} | "
            f"{o['alpha075_correct_chck84_wrong']} | {o['chck84_correct_alpha075_wrong']} | {o['net_alpha075_minus_chck84_items']} |\n"
        )

    lines.append("\n## Interpretation\n\n")
    alpha_minus_84_cheap7 = d["alpha075_minus_84_cheap7"]
    alpha_minus_84_cheap6 = d["alpha075_minus_84_cheap6_no_globalpiqa"]
    alpha_minus_84_rel = d["alpha075_minus_84_relation_state_ewok_entity"]
    if alpha_minus_84_cheap7 > 0 and alpha_minus_84_cheap6 <= 0:
        lines.append("The alpha0.75 endpoint has a small cheap7 advantage over chck84, but that advantage disappears when GlobalPIQA is removed. ")
    elif alpha_minus_84_cheap7 > 0:
        lines.append("The alpha0.75 endpoint has a cheap7 advantage over chck84 that partly survives the non-GlobalPIQA aggregate. ")
    else:
        lines.append("The chck84 endpoint matches or exceeds alpha0.75 on cheap7. ")
    if alpha_minus_84_rel < 0:
        lines.append("Relation/state columns favor the trained chck84 checkpoint rather than the interpolated private endpoint. ")
    lines.append("Thus the two endpoint functions are not a strict dominance relation: alpha0.75 remains the numerically stronger local cheap7/Overall(AoA0) carrier before chck84 SuperGLUE is known, while chck84 is the cleaner ordinary-training endpoint and preserves more relation/state signal. This is endpoint evidence, not a new authorization for alpha tuning or eval-set-driven combination.\n\n")

    lines.append("## Largest subtask differences, alpha0.75 minus chck84\n\n")
    for col in CLASSIFICATION_COLUMNS:
        lines.append(f"### {col}\n")
        lines.append("Alpha0.75 above chck84:\n")
        for r in top_alpha_over_84[col][:5]:
            lines.append(f"- {r['subtask']}: {r['delta_alpha075_minus_84']:+.3f} points (n={r['n_items']}, net={r.get('net_items_84_to_alpha075')})\n")
        lines.append("Chck84 above alpha0.75:\n")
        for r in top_84_over_alpha[col][:5]:
            lines.append(f"- {r['subtask']}: {r['delta_alpha075_minus_84']:+.3f} points (n={r['n_items']}, net={r.get('net_items_84_to_alpha075')})\n")
        lines.append("\n")

    lines.append("## Files\n\n")
    for name, rec in summary["output_files"].items():
        lines.append(f"- `{rec['path']}` ({rec['bytes']} bytes, sha256 `{str(rec['sha256'])[:16]}…`)\n")
    lines.append(f"- JSON summary: `{out_dir / 'chck84_vs_alpha075_item_overlap_summary.json'}`\n")

    write_json(out_dir / "chck84_vs_alpha075_item_overlap_summary.json", summary)
    (out_dir / "chck84_vs_alpha075_item_overlap_summary.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(out_dir),
        "n_classification_items": len(merged),
        "n_disagreements": len(disagreements),
        "alpha075_minus_chck84": {
            "cheap7": round(d["alpha075_minus_84_cheap7"], 6),
            "cheap6_no_globalpiqa": round(d["alpha075_minus_84_cheap6_no_globalpiqa"], 6),
            "cheap5_no_globalpiqa_reading": round(d["alpha075_minus_84_cheap5_no_globalpiqa_reading"], 6),
            "relation_state": round(d["alpha075_minus_84_relation_state_ewok_entity"], 6),
        },
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
