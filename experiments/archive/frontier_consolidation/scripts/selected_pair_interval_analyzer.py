#!/usr/bin/env python3
"""research: paired selected-prediction interval analyzer for extractive-view readouts.

This is CPU/file-only interpretation of already-produced BabyLM selected-eval
per-target JSON files.  It does not train, score models, run SuperGLUE/AoA,
upload, or submit.  Its purpose is to make the pending research source-only
extractive result interpretable immediately after selected per-target files
exist: for each extractive arm and checkpoint, compare selected predictions
against the legal compact reference on item-level stable families and report
paired target bootstrap intervals.

Two modes:
  * panel mode (default): read an extractive selected panel directory and run
    ready extractive-vs-legal_compact comparisons.
  * direct mode: pass --left-per-target and --right-per-target for a smoke or
    one-off comparison.

Convention: reported deltas are RIGHT minus LEFT in percentage points of raw
item correctness. Official aggregate score deltas are still taken from the
selected panel/per-target records; item intervals are interpretive because some
BabyLM columns average by UID/subtask rather than raw item count.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:  # Use NumPy when available; the fallback keeps this analysis portable.
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - fallback for portability
    np = None  # type: ignore


EXTRACTIVE_ARMS = ["extractive_balanced", "extractive_wide"]
REFERENCE_ARM = "legal_compact"
STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GLOBAL_COLUMNS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
SUBSETS = {
    "stable_five_item_pool": STABLE_COLUMNS,
    "EWoK_plus_Entity_item_pool": ["EWoK", "Entity"],
    "BLiMP": ["BLiMP"],
    "Supplement": ["Supplement"],
    "EWoK": ["EWoK"],
    "Entity": ["Entity"],
    "COMPS": ["COMPS"],
    "GlobalPIQA_item_pool": GLOBAL_COLUMNS,
}


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_STAGE1_PANEL = WS / "data/extractive_selected_eval_stage1_plan"
DEFAULT_OUT = WS / "data/extractive_selected_interval_readiness"
MOVEMENT_READER = WS / "scripts/selected_prediction_movement_reader.py"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve(x: str | Path | None) -> Path | None:
    if x is None or str(x) == "":
        return None
    p = Path(str(x))
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return ROOT / p


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_step218_module():
    if not MOVEMENT_READER.exists():
        raise FileNotFoundError(MOVEMENT_READER)
    spec = importlib.util.spec_from_file_location("selected_prediction_movement_reader", MOVEMENT_READER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {MOVEMENT_READER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def per_target_from_summary_path(summary_path_s: str | None) -> Path | None:
    p = resolve(summary_path_s)
    if p is None or not p.exists():
        return None
    obj = read_json(p)
    if isinstance(obj, dict) and "record" in obj and isinstance(obj["record"], dict):
        pt = obj["record"].get("per_target")
        return resolve(pt) if pt else None
    if isinstance(obj, dict) and "tasks" in obj:
        return p
    return None


def load_rows(panel_dir: Path) -> list[dict[str, str]]:
    rows_csv = panel_dir / "extractive_selected_panel_rows.csv"
    if not rows_csv.exists():
        return []
    with rows_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def checkpoint_key(ck: str) -> int:
    if ck.startswith("chck_") and ck.endswith("M"):
        try:
            return int(ck[len("chck_"):-1])
        except Exception:
            pass
    return 10**9


def build_panel_comparisons(panel_dir: Path) -> list[dict[str, Any]]:
    rows = load_rows(panel_dir)
    by = {(r.get("arm", ""), r.get("checkpoint", "")): r for r in rows}
    summary = {}
    sp = panel_dir / "extractive_selected_panel_summary.json"
    if sp.exists():
        summary = read_json(sp)
    checkpoints = sorted({r.get("checkpoint", "") for r in rows if r.get("checkpoint")}, key=checkpoint_key)
    if not checkpoints:
        checkpoints = list(summary.get("checkpoints", []))
    comparisons: list[dict[str, Any]] = []
    for ck in checkpoints:
        ref = by.get((REFERENCE_ARM, ck))
        ref_pt = per_target_from_summary_path(ref.get("summary_path")) if ref else None
        for arm in EXTRACTIVE_ARMS:
            row = by.get((arm, ck))
            arm_pt = per_target_from_summary_path(row.get("summary_path")) if row else None
            comparisons.append({
                "arm": arm,
                "checkpoint": ck,
                "left_label": REFERENCE_ARM,
                "right_label": arm,
                "left_per_target": str(ref_pt) if ref_pt else None,
                "right_per_target": str(arm_pt) if arm_pt else None,
                "ready": bool(ref_pt and arm_pt and ref_pt.exists() and arm_pt.exists()),
                "missing": {
                    "reference": None if ref_pt and ref_pt.exists() else f"missing {REFERENCE_ARM} {ck}",
                    "extractive": None if arm_pt and arm_pt.exists() else f"missing {arm} {ck}",
                },
            })
    return comparisons


def official_scores_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    # Prefer official aggregate scores where available. For selected eval records
    # generated by the current wrappers, column scores are usually in task_scores.
    task_scores = meta.get("task_scores") or {}
    official = meta.get("official_scores") or {}
    out: dict[str, Any] = {}
    for col in [*STABLE_COLUMNS, *GLOBAL_COLUMNS, "GlobalPIQA", "Reading"]:
        if col in official:
            out[col] = official[col]
        elif col in task_scores:
            out[col] = task_scores[col]
    if "GlobalPIQA" not in out:
        gps = []
        for c in GLOBAL_COLUMNS:
            v = out.get(c)
            if v is not None:
                try:
                    gps.append(float(v))
                except Exception:
                    pass
        if len(gps) == 2:
            out["GlobalPIQA"] = sum(gps) / 2.0
    return out


def score_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    keys = sorted(set(left) | set(right))
    for k in keys:
        try:
            out[k] = float(right.get(k)) - float(left.get(k))
        except Exception:
            out[k] = None
    return out


def cheap_composites(scores: dict[str, Any]) -> dict[str, float | None]:
    def f(vals: list[Any]) -> float | None:
        try:
            xs = [float(v) for v in vals]
            if any(not math.isfinite(x) for x in xs):
                return None
            return sum(xs) / len(xs)
        except Exception:
            return None
    b, s, e, en, c, g, r = [scores.get(k) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]]
    return {
        "cheap7": f([b, s, e, en, c, g, r]),
        "cheap6_no_GlobalPIQA": f([b, s, e, en, c, r]),
        "cheap5_no_GlobalPIQA_Reading": f([b, s, e, en, c]),
        "EWoK_plus_Entity": (float(e) + float(en)) if e is not None and en is not None else None,
    }


def deterministic_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big", signed=False)


def quantiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"p025": None, "p50": None, "p975": None}
    ys = sorted(xs)
    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        pos = p * (len(ys) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return ys[lo]
        frac = pos - lo
        return ys[lo] * (1 - frac) + ys[hi] * frac
    return {"p025": q(0.025), "p50": q(0.5), "p975": q(0.975)}


def bootstrap_mean_pp(diffs: list[int], label: str, n_bootstrap: int) -> dict[str, Any]:
    n = len(diffs)
    point = 100.0 * (sum(diffs) / n) if n else None
    if n == 0 or n_bootstrap <= 0:
        return {"n": n, "point_pp": point, "bootstrap": None}
    if np is not None:
        arr = np.asarray(diffs, dtype=np.float32)
        rng = np.random.default_rng(deterministic_seed(label))
        vals: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n, endpoint=False)
            vals.append(float(arr[idx].mean() * 100.0))
    else:  # deterministic slower fallback
        import random
        rng = random.Random(deterministic_seed(label))
        vals = []
        for _ in range(n_bootstrap):
            vals.append(100.0 * sum(diffs[rng.randrange(n)] for _i in range(n)) / n)
    qs = quantiles(vals)
    return {
        "n": n,
        "point_pp": point,
        "bootstrap": {"n_bootstrap": n_bootstrap, **qs},
    }


def cluster_bootstrap_mean_pp(item_rows: list[dict[str, Any]], label: str, n_bootstrap: int) -> dict[str, Any]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for r in item_rows:
        cid = f"{r.get('column')}::{r.get('subtask')}"
        clusters[cid].append((1 if r.get("right_correct") else 0) - (1 if r.get("left_correct") else 0))
    citems = list(clusters.items())
    n_clusters = len(citems)
    n_items = sum(len(v) for _, v in citems)
    point = 100.0 * sum(sum(v) for _, v in citems) / n_items if n_items else None
    if n_clusters == 0 or n_bootstrap <= 0:
        return {"n_items": n_items, "n_clusters": n_clusters, "point_pp": point, "bootstrap": None}
    if np is not None:
        rng = np.random.default_rng(deterministic_seed(label + ":cluster"))
        sums = np.asarray([sum(v) for _, v in citems], dtype=np.float64)
        sizes = np.asarray([len(v) for _, v in citems], dtype=np.float64)
        vals = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_clusters, size=n_clusters, endpoint=False)
            vals.append(float(100.0 * sums[idx].sum() / sizes[idx].sum()))
    else:
        import random
        rng = random.Random(deterministic_seed(label + ":cluster"))
        vals = []
        for _ in range(n_bootstrap):
            s = 0; z = 0
            for _j in range(n_clusters):
                _, v = citems[rng.randrange(n_clusters)]
                s += sum(v); z += len(v)
            vals.append(100.0 * s / z if z else 0.0)
    qs = quantiles(vals)
    return {"n_items": n_items, "n_clusters": n_clusters, "point_pp": point, "bootstrap": {"n_bootstrap": n_bootstrap, **qs}}


def analyze_pair(left_pt: Path, right_pt: Path, left_label: str, right_label: str, out_dir: Path, n_bootstrap: int, max_examples: int) -> dict[str, Any]:
    mod = load_step218_module()
    left_record = mod.load_eval_record(left_pt)
    right_record = mod.load_eval_record(right_pt)
    left_items, left_meta = mod.load_items_from_record(left_record)
    right_items, right_meta = mod.load_items_from_record(right_record)
    movement = mod.compare(left_items, right_items, left_label, right_label, left_meta, right_meta, max_examples=max_examples)
    item_rows = movement["per_item_rows"]

    subset_results: dict[str, Any] = {}
    for subset_name, cols in SUBSETS.items():
        subset = [r for r in item_rows if r.get("column") in cols]
        diffs = [(1 if r.get("right_correct") else 0) - (1 if r.get("left_correct") else 0) for r in subset]
        subset_results[subset_name] = {
            "item_bootstrap": bootstrap_mean_pp(diffs, f"{left_pt}|{right_pt}|{subset_name}", n_bootstrap),
            "subtask_cluster_bootstrap": cluster_bootstrap_mean_pp(subset, f"{left_pt}|{right_pt}|{subset_name}", n_bootstrap),
        }

    left_scores = official_scores_from_meta(left_meta)
    right_scores = official_scores_from_meta(right_meta)
    left_scores.update(cheap_composites(left_scores))
    right_scores.update(cheap_composites(right_scores))
    official_delta = score_delta(left_scores, right_scores)

    payload = {
        "status": "SELECTED_PAIR_INTERVAL_ANALYSIS",
        "created_utc": now(),
        "meaning": "CPU/file-only paired item interval analysis of existing selected prediction files; RIGHT minus LEFT.",
        "inputs": {
            "left_per_target": str(left_pt),
            "right_per_target": str(right_pt),
            "left_label": left_label,
            "right_label": right_label,
        },
        "official_or_wrapper_scores": {
            "left": left_scores,
            "right": right_scores,
            "delta_right_minus_left": official_delta,
            "note": "Official/wrapper aggregates are the selected score readout. Item bootstrap intervals are raw item-level interpretive summaries and do not replace official group averaging.",
        },
        "coverage": movement["missing_item_diagnostics"],
        "subset_intervals": subset_results,
        "movement_summary_by_column": movement["by_column"],
        "worst_subtasks_by_right_minus_left": movement["worst_subtasks_by_right_minus_left"][:30],
        "best_subtasks_by_right_minus_left": movement["best_subtasks_by_right_minus_left"][:30],
        "examples": movement["examples"],
        "loader_warnings_count": len(left_meta.get("warnings", [])) + len(right_meta.get("warnings", [])),
        "skipped_columns": {"left": left_meta.get("skipped"), "right": right_meta.get("skipped")},
        "boundary": "No training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission was performed.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "selected_pair_interval_analysis.json"
    md_path = out_dir / "selected_pair_interval_analysis.md"
    csv_path = out_dir / "subset_intervals.csv"
    payload["outputs"] = {"json": str(json_path), "md": str(md_path), "csv": str(csv_path)}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["subset", "n", "item_point_pp", "item_p025", "item_p50", "item_p975", "n_clusters", "cluster_point_pp", "cluster_p025", "cluster_p50", "cluster_p975"]
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for subset_name, rec in subset_results.items():
            ib = rec["item_bootstrap"]
            cb = rec["subtask_cluster_bootstrap"]
            wr.writerow({
                "subset": subset_name,
                "n": ib.get("n"),
                "item_point_pp": ib.get("point_pp"),
                "item_p025": (ib.get("bootstrap") or {}).get("p025"),
                "item_p50": (ib.get("bootstrap") or {}).get("p50"),
                "item_p975": (ib.get("bootstrap") or {}).get("p975"),
                "n_clusters": cb.get("n_clusters"),
                "cluster_point_pp": cb.get("point_pp"),
                "cluster_p025": (cb.get("bootstrap") or {}).get("p025"),
                "cluster_p50": (cb.get("bootstrap") or {}).get("p50"),
                "cluster_p975": (cb.get("bootstrap") or {}).get("p975"),
            })
    md_path.write_text(make_md(payload), encoding="utf-8")
    return payload


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def make_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    inp = payload["inputs"]
    lines.append(f"# research selected pair interval: {inp['right_label']} minus {inp['left_label']}")
    lines.append("")
    lines.append(f"Created UTC: `{payload['created_utc']}`")
    lines.append("")
    lines.append("CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.")
    lines.append("")
    lines.append("## Official/wrapper selected deltas")
    lines.append("")
    od = payload["official_or_wrapper_scores"]["delta_right_minus_left"]
    for k in ["cheap7", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity", "BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]:
        if k in od:
            lines.append(f"- {k}: {fmt(od.get(k), 4)}")
    lines.append("")
    lines.append("## Raw item paired intervals, RIGHT minus LEFT (percentage points)")
    lines.append("")
    lines.append("| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for subset_name, rec in payload["subset_intervals"].items():
        ib = rec["item_bootstrap"]; cb = rec["subtask_cluster_bootstrap"]
        ibq = ib.get("bootstrap") or {}; cbq = cb.get("bootstrap") or {}
        lines.append(
            f"| {subset_name} | {ib.get('n')} | {fmt(ib.get('point_pp'), 4)} | [{fmt(ibq.get('p025'), 4)}, {fmt(ibq.get('p975'), 4)}] | "
            f"{cb.get('n_clusters')} | {fmt(cb.get('point_pp'), 4)} | [{fmt(cbq.get('p025'), 4)}, {fmt(cbq.get('p975'), 4)}] |"
        )
    cov = payload["coverage"]
    lines.append("")
    lines.append("## Coverage and caution")
    lines.append(f"- common items: `{cov.get('n_common_items')}`, left-only `{cov.get('left_only_item_ids')}`, right-only `{cov.get('right_only_item_ids')}`.")
    lines.append(f"- loader warnings: `{payload.get('loader_warnings_count')}`; skipped columns: {payload.get('skipped_columns')}.")
    lines.append("- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.")
    lines.append("")
    lines.append("## Files")
    for k, v in payload.get("outputs", {}).items():
        lines.append(f"- {k}: `{v}`")
    return "\n".join(lines) + "\n"


def run_panel(panel_dir: Path, out_dir: Path, n_bootstrap: int, max_examples: int, plan_only: bool, force: bool) -> dict[str, Any]:
    comparisons = build_panel_comparisons(panel_dir)
    ready = [c for c in comparisons if c["ready"]]
    payload: dict[str, Any] = {
        "status": "EXTRACTIVE_INTERVAL_PANEL_PLAN",
        "created_utc": now(),
        "panel_dir": str(panel_dir),
        "comparisons": comparisons,
        "ready_count": len(ready),
        "n_comparisons": len(comparisons),
        "n_bootstrap": n_bootstrap,
        "meaning": "Readiness and optional interval analysis for extractive-vs-legal_compact selected prediction comparisons.",
        "no_training_selected_eval_superglue_aoa_upload_or_leaderboard": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if not plan_only:
        for c in ready:
            subdir = out_dir / c["arm"] / c["checkpoint"]
            done = subdir / "selected_pair_interval_analysis.json"
            if done.exists() and not force:
                results.append({"arm": c["arm"], "checkpoint": c["checkpoint"], "status": "exists", "out_json": str(done)})
                continue
            res = analyze_pair(resolve(c["left_per_target"]), resolve(c["right_per_target"]), c["left_label"], c["right_label"], subdir, n_bootstrap, max_examples)  # type: ignore[arg-type]
            results.append({"arm": c["arm"], "checkpoint": c["checkpoint"], "status": res["status"], "out_json": res["outputs"]["json"]})
    payload["run_results"] = results
    plan_path = out_dir / "interval_panel_plan.json"
    plan_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-dir", type=Path, default=DEFAULT_STAGE1_PANEL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=300)
    ap.add_argument("--max-examples", type=int, default=24)
    ap.add_argument("--left-per-target", type=Path)
    ap.add_argument("--right-per-target", type=Path)
    ap.add_argument("--left-label", default="left")
    ap.add_argument("--right-label", default="right")
    args = ap.parse_args()

    if args.left_per_target or args.right_per_target:
        if not args.left_per_target or not args.right_per_target:
            raise ValueError("direct mode requires both --left-per-target and --right-per-target")
        payload = analyze_pair(resolve(args.left_per_target), resolve(args.right_per_target), args.left_label, args.right_label, args.out_dir, args.n_bootstrap, args.max_examples)  # type: ignore[arg-type]
        print(json.dumps({"status": payload["status"], "out_json": payload["outputs"]["json"], "n_common_items": payload["coverage"].get("n_common_items")}, indent=2), flush=True)
        return

    payload = run_panel(args.panel_dir, args.out_dir, args.n_bootstrap, args.max_examples, args.plan_only, args.force)
    print(json.dumps({"status": payload["status"], "out": str(args.out_dir / "interval_panel_plan.json"), "ready_count": payload["ready_count"], "n_comparisons": payload["n_comparisons"], "ran": not args.plan_only}, indent=2), flush=True)


if __name__ == "__main__":
    main()
