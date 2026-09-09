#!/usr/bin/env python3
"""research/283: read out register-displacement packet-minus-clean and pair curves.

Reads stable-family per_target JSONs from:
  * research clean-anchor common-window scorer (two MAX-geometry clean basins)
  * research register_commonwindow_eval scorer (after register arms are trained/scored)

Combines them with research position-matched direct pool metadata. Before scores
exist, it produces a truthful missing-data summary with data_state=no_scores or
partial, not a misleading scientific completion. No model loading, training,
official evaluation, GPU work, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, math, pathlib, statistics, time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
OUT = WS / "data/register_vc_readout"
CLEAN_EVAL = WS / "data/clean_anchor_commonwindow_eval/eval/per_target"
REG_EVAL = WS / "data/register_commonwindow_eval/eval/per_target"
REG_META = WS / "data/register_position_matched_direct_pools/register_position_matched_direct_pools_metadata.json"

STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
EX_ENTITY_COLUMNS = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
AGGREGATES = ["cheap6", "cheap5", "exEntity5"]
METRICS = STABLE_COLUMNS + AGGREGATES
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 81, 10)]
WINDOWS = {
    "common10_80": CHECKPOINTS,
    "early10_40": [f"chck_{i}M" for i in range(10, 41, 10)],
    "mature50_80": [f"chck_{i}M" for i in range(50, 81, 10)],
    "endpoint80": ["chck_80M"],
}
REG_ARMS = {
    "childsub_posmatched": {
        "prefix": "regpos_childsub_posmatched_seed43022",
        "kind": "child/subtitle-rich clean removed",
        "pair_role": "remove_childsub_rich",
    },
    "adult_posmatched": {
        "prefix": "regpos_adult_posmatched_seed43022",
        "kind": "Gutenberg/SimpleWiki-rich clean removed",
        "pair_role": "remove_gutsimple_rich",
    },
}
CLEAN_PREFIXES = {43022: "deberta_maxgeom_clean_seed43022", 43122: "deberta_maxgeom_clean_seed43122"}


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
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def finite(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def metric_from_payload(payload: dict[str, Any], col: str) -> float | None:
    ss = payload.get("stable_family_scores") or {}
    aliases = {
        "cheap6": "cheap6_no_GlobalPIQA",
        "cheap5": "cheap5_no_GlobalPIQA_Reading",
        "exEntity5": "exEntity5_BLiMP_Supp_EWoK_COMPS_Reading",
    }
    key = aliases.get(col, col)
    if finite(ss.get(key)):
        return float(ss[key])
    rec = (payload.get("tasks") or {}).get(col) or {}
    if col == "Reading":
        scores = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
        val = scores.get("Reading") if scores else rec.get("score")
    elif col in aliases:
        fams = STABLE_COLUMNS if col == "cheap6" else ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"] if col == "cheap5" else EX_ENTITY_COLUMNS
        vals = [metric_from_payload(payload, c) for c in fams]
        return statistics.mean([float(v) for v in vals]) if all(finite(v) for v in vals) else None
    else:
        val = rec.get("score") if isinstance(rec, dict) else None
    return float(val) if finite(val) else None


def load_payload(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = read_json(path)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def score_row(label: str, ck: str, path: pathlib.Path, seed: int | None, kind: str, meta: dict[str, Any] | None) -> dict[str, Any]:
    payload = load_payload(path)
    row: dict[str, Any] = {
        "label": label,
        "checkpoint": ck,
        "seed": seed,
        "kind": kind,
        "path": rel(path),
        "exists": path.exists(),
        "rho": meta.get("rho") if meta else 0.0,
        "active_fineweb_words": meta.get("active_fineweb_words") if meta else 0,
        "displaced_child_sub_fraction": (meta.get("displaced_clean_summary") or {}).get("child_sub_fraction") if meta else None,
        "displaced_adult_fraction": (meta.get("displaced_clean_summary") or {}).get("adult_gut_simple_fraction") if meta else None,
    }
    if payload is None:
        for m in METRICS:
            row[m] = None
        row["complete_stable6"] = False
        row["complete_exEntity5"] = False
        return row
    for c in STABLE_COLUMNS:
        row[c] = metric_from_payload(payload, c)
    row["cheap6"] = metric_from_payload(payload, "cheap6")
    row["cheap5"] = metric_from_payload(payload, "cheap5")
    row["exEntity5"] = metric_from_payload(payload, "exEntity5")
    row["complete_stable6"] = all(finite(row.get(c)) for c in STABLE_COLUMNS)
    row["complete_exEntity5"] = all(finite(row.get(c)) for c in EX_ENTITY_COLUMNS)
    if row["complete_stable6"] and all(finite(row.get(m)) for m in ["cheap6", "exEntity5", "Entity"]):
        row["cheap6_identity_residual"] = 6.0 * float(row["cheap6"]) - 5.0 * float(row["exEntity5"]) - float(row["Entity"])
    else:
        row["cheap6_identity_residual"] = None
    return row


def stats(vals: list[float]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "mean": statistics.mean(vals) if vals else None,
        "median": statistics.median(vals) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "stdev": statistics.stdev(vals) if len(vals) >= 2 else 0.0 if len(vals) == 1 else None,
        "positive": sum(v > 0 for v in vals),
        "negative": sum(v < 0 for v in vals),
        "zero": sum(v == 0 for v in vals),
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for r in rows for k in r.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not REG_META.exists():
        raise FileNotFoundError(REG_META)
    meta = read_json(REG_META)
    arm_meta = {ar["arm_key"]: ar for ar in meta.get("arm_results", [])}
    missing_meta = [a for a in REG_ARMS if a not in arm_meta]
    if missing_meta:
        raise KeyError(f"missing register arm metadata for {missing_meta}")

    score_rows: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        for seed, pref in CLEAN_PREFIXES.items():
            score_rows.append(score_row(f"clean_seed{seed}", ck, CLEAN_EVAL / f"{pref}_{ck}.json", seed, "clean", None))
        for arm, cfg in REG_ARMS.items():
            expected = REG_EVAL / f"{cfg['prefix']}_{ck}.json"
            score_rows.append(score_row(arm, ck, expected, 43022, cfg["kind"], arm_meta[arm]))

    clean_by_ck: dict[str, dict[str, Any]] = {}
    for ck in CHECKPOINTS:
        clean_by_ck[ck] = {}
        crs = [r for r in score_rows if r["kind"] == "clean" and r["checkpoint"] == ck]
        for m in METRICS:
            vals = [r.get(m) for r in crs if finite(r.get(m))]
            clean_by_ck[ck][m+"_mean"] = statistics.mean([float(v) for v in vals]) if vals else None
            clean_by_ck[ck][m+"_spread"] = max([float(v) for v in vals]) - min([float(v) for v in vals]) if len(vals) >= 2 else None
            clean_by_ck[ck][m+"_n"] = len(vals)
            for r in crs:
                clean_by_ck[ck][f"{m}_seed{r['seed']}"] = r.get(m)

    arm_clean_delta_rows: list[dict[str, Any]] = []
    for r in score_rows:
        if r["kind"] == "clean":
            continue
        ck = r["checkpoint"]
        base = {k: r.get(k) for k in ["label", "checkpoint", "kind", "rho", "active_fineweb_words", "displaced_child_sub_fraction", "displaced_adult_fraction", "path", "exists", "complete_stable6", "complete_exEntity5"]}
        for m in METRICS:
            rv = r.get(m)
            for seed in CLEAN_PREFIXES:
                cv = clean_by_ck[ck].get(f"{m}_seed{seed}")
                row = dict(base)
                row.update({
                    "metric": m,
                    "arm_score": rv,
                    "clean_seed": seed,
                    "clean_score": cv,
                    "packet_minus_clean": float(rv) - float(cv) if finite(rv) and finite(cv) else None,
                    "metric_complete": finite(rv) and finite(cv),
                })
                arm_clean_delta_rows.append(row)
            cvm = clean_by_ck[ck].get(m+"_mean")
            row = dict(base)
            row.update({
                "metric": m,
                "arm_score": rv,
                "clean_seed": "mean_of_both_required",
                "clean_score": cvm,
                "clean_n": clean_by_ck[ck].get(m+"_n"),
                "clean_spread": clean_by_ck[ck].get(m+"_spread"),
                "packet_minus_clean": float(rv) - float(cvm) if finite(rv) and finite(cvm) and clean_by_ck[ck].get(m+"_n") == 2 else None,
                "metric_complete": finite(rv) and finite(cvm) and clean_by_ck[ck].get(m+"_n") == 2,
            })
            arm_clean_delta_rows.append(row)

    pair_rows: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        child = next((r for r in score_rows if r["label"] == "childsub_posmatched" and r["checkpoint"] == ck), None)
        adult = next((r for r in score_rows if r["label"] == "adult_posmatched" and r["checkpoint"] == ck), None)
        if not child or not adult:
            continue
        for m in METRICS:
            cv = child.get(m); av = adult.get(m)
            pair_rows.append({
                "pair_label": "remove_childsub_rich_minus_remove_gutsimple_rich",
                "checkpoint": ck,
                "metric": m,
                "childsub_removed_score": cv,
                "adult_removed_score": av,
                "childsub_removed_minus_adult_removed": float(cv) - float(av) if finite(cv) and finite(av) else None,
                "metric_complete": finite(cv) and finite(av),
            })

    trajectory_rows: list[dict[str, Any]] = []
    for label in REG_ARMS:
        for metric in METRICS:
            for window, cks in WINDOWS.items():
                vals = [float(r["packet_minus_clean"]) for r in arm_clean_delta_rows if r["label"] == label and r["metric"] == metric and r["clean_seed"] == "mean_of_both_required" and r["checkpoint"] in cks and finite(r.get("packet_minus_clean"))]
                st = stats(vals)
                trajectory_rows.append({
                    "curve": "packet_minus_clean_two_anchor_mean",
                    "label": label,
                    "removed_register": REG_ARMS[label]["kind"],
                    "metric": metric,
                    "window": window,
                    "required_checkpoints": ";".join(cks),
                    "complete_checkpoint_count": len(vals),
                    "window_complete": len(vals) == len(cks),
                    "rho": arm_meta[label].get("rho"),
                    "active_fineweb_words": arm_meta[label].get("active_fineweb_words"),
                    "displaced_child_sub_fraction": (arm_meta[label].get("displaced_clean_summary") or {}).get("child_sub_fraction"),
                    "displaced_adult_fraction": (arm_meta[label].get("displaced_clean_summary") or {}).get("adult_gut_simple_fraction"),
                    **st,
                })
    pair_window_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        for window, cks in WINDOWS.items():
            vals = [float(r["childsub_removed_minus_adult_removed"]) for r in pair_rows if r["metric"] == metric and r["checkpoint"] in cks and finite(r.get("childsub_removed_minus_adult_removed"))]
            pair_window_rows.append({
                "pair_label": "remove_childsub_rich_minus_remove_gutsimple_rich",
                "metric": metric,
                "window": window,
                "required_checkpoints": ";".join(cks),
                "complete_checkpoint_count": len(vals),
                "window_complete": len(vals) == len(cks),
                **stats(vals),
            })

    # Data-state summary distinguishes script completion from scientific score availability.
    register_metric_cells = [r for r in pair_rows if r["metric"] in STABLE_COLUMNS]
    complete_cells = [r for r in register_metric_cells if r.get("metric_complete")]
    clean_metric_cells = [r for r in score_rows if r["kind"] == "clean" for m in STABLE_COLUMNS if finite(r.get(m))]
    expected_reg_stable_cells = len(CHECKPOINTS) * 2 * len(STABLE_COLUMNS)
    expected_clean_stable_cells = len(CHECKPOINTS) * 2 * len(STABLE_COLUMNS)
    if len(complete_cells) == 0 and len(clean_metric_cells) == 0:
        data_state = "no_scores"
    elif len(complete_cells) == expected_reg_stable_cells // 2 and len(clean_metric_cells) == expected_clean_stable_cells:
        # complete_cells are pair metric rows, one per ck/metric, so divide register cell count by two.
        data_state = "full"
    else:
        data_state = "partial"

    identity_residuals = [abs(float(r["cheap6_identity_residual"])) for r in score_rows if finite(r.get("cheap6_identity_residual"))]
    result = {
        "status": "reference_283_REGISTER_PACKET_CLEAN_READOUT_SCRIPT_COMPLETE",
        "data_state": data_state,
        "created_utc": now(),
        "boundary": "File-only readout; no model loading, training, official evaluation, GPU work, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.",
        "pool_metadata": rel(REG_META),
        "expected_clean_stable_cells": expected_clean_stable_cells,
        "available_clean_stable_cells": len(clean_metric_cells),
        "expected_pair_stable_cells": len(CHECKPOINTS) * len(STABLE_COLUMNS),
        "available_pair_stable_cells": len(complete_cells),
        "score_rows": len(score_rows),
        "arm_clean_delta_rows": len(arm_clean_delta_rows),
        "pair_rows": len(pair_rows),
        "trajectory_window_rows": len(trajectory_rows),
        "pair_window_rows": len(pair_window_rows),
        "max_abs_cheap6_identity_residual": max(identity_residuals) if identity_residuals else None,
        "trajectory_window_summaries": trajectory_rows,
        "pair_window_summaries": pair_window_rows,
        "clean_reference_by_checkpoint": clean_by_ck,
        "files": {
            "raw_score_rows": rel(OUT / "register_raw_score_rows.csv"),
            "arm_clean_delta_rows": rel(OUT / "register_packet_minus_clean_rows.csv"),
            "pair_delta_rows": rel(OUT / "register_pair_delta_rows.csv"),
            "trajectory_windows": rel(OUT / "register_packet_minus_clean_window_summaries.csv"),
            "pair_windows": rel(OUT / "register_pair_window_summaries.csv"),
            "summary_json": rel(OUT / "register_vc_readout_summary.json"),
            "summary_md": rel(OUT / "register_vc_readout_summary.md"),
        },
    }
    write_csv(OUT / "register_raw_score_rows.csv", score_rows)
    write_csv(OUT / "register_packet_minus_clean_rows.csv", arm_clean_delta_rows)
    write_csv(OUT / "register_pair_delta_rows.csv", pair_rows)
    write_csv(OUT / "register_packet_minus_clean_window_summaries.csv", trajectory_rows)
    write_csv(OUT / "register_pair_window_summaries.csv", pair_window_rows)
    write_json(OUT / "register_vc_readout_summary.json", result)

    def fmt(x: Any) -> str:
        return "" if not finite(x) else f"{float(x):+.4f}"
    lines = [
        "# research/283 register packet-minus-clean readout",
        "",
        result["boundary"],
        "",
        f"Data state: `{data_state}`. This states score availability, not scientific completion.",
        "",
        f"Available clean stable cells: {len(clean_metric_cells)} / {expected_clean_stable_cells}.",
        f"Available direct-pair stable cells: {len(complete_cells)} / {len(CHECKPOINTS) * len(STABLE_COLUMNS)}.",
        f"Max cheap6 aggregation residual: {fmt(result['max_abs_cheap6_identity_residual'])}.",
        "",
        "## Pair contrast: remove_childsub_rich_minus_remove_gutsimple_rich",
        "",
        "| metric | window | complete ck | mean | min | max | positive | negative |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pair_window_rows:
        if r["metric"] in ["cheap6", "exEntity5", "Entity", "BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]:
            lines.append(f"| {r['metric']} | {r['window']} | {r['complete_checkpoint_count']} | {fmt(r.get('mean'))} | {fmt(r.get('min'))} | {fmt(r.get('max'))} | {r.get('positive')} | {r.get('negative')} |")
    lines += [
        "",
        "## Packet-minus-clean two-anchor summaries",
        "",
        "| arm | removed register | metric | window | complete ck | mean | clean requirement |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for r in trajectory_rows:
        if r["metric"] in ["cheap6", "exEntity5", "Entity"] and r["window"] in ["common10_80", "mature50_80", "endpoint80"]:
            lines.append(f"| {r['label']} | {r['removed_register']} | {r['metric']} | {r['window']} | {r['complete_checkpoint_count']} | {fmt(r.get('mean'))} | both clean anchors required |")
    lines += [
        "",
        "## Reading reminder",
        "",
        "The direct pair difference is the primary removal-side discriminator. Arm-minus-clean uses both clean anchors and should not be summarized from a single clean seed. A mixed quarter_1x null can be cancellation between removed-register effects. Entity should be separated from exEntity5 because previous Entity movement was operation-allocation-skewed.",
        "",
        f"JSON: `{rel(OUT / 'register_vc_readout_summary.json')}`",
        f"Pair deltas: `{rel(OUT / 'register_pair_delta_rows.csv')}`",
        f"Packet-minus-clean rows: `{rel(OUT / 'register_packet_minus_clean_rows.csv')}`",
    ]
    (OUT / "register_vc_readout_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "data_state": data_state,
        "available_clean_stable_cells": len(clean_metric_cells),
        "available_pair_stable_cells": len(complete_cells),
        "summary_md": rel(OUT / "register_vc_readout_summary.md"),
    }, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
