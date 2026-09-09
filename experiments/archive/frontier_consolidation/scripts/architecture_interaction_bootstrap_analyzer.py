#!/usr/bin/env python3
"""research: four-cell item bootstrap for the DeBERTa positional-package interaction.

This CPU/file-only script sharpens the readout prepared in research. After
`architecture_interaction_selected_panel.py` has produced per-target JSON
files for the four cells

  full_compact, full_repeat, nodis_compact, nodis_repeat

it computes the raw selected-item interaction

  (nodis_compact - nodis_repeat) - (full_compact - full_repeat)

on common item IDs for stable BabyLM selected columns and GlobalPIQA. It reports
item and subtask-cluster bootstrap intervals. Official/wrapper aggregate score
interactions from the selected panel remain the primary score readout because
some BabyLM columns use UID/subtask averaging; this script is an interpretive
item-level check of whether the interaction is carried by broad stable-item
movement or by a small/fragile subset.

The script does not train, run selected evaluation, run SuperGLUE/AoA, upload, or
submit to any leaderboard. Use --plan-only while the research H100 jobs are still
unresolved.
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
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_PANEL = WS / "data/architecture_interaction_selected_panel"
DEFAULT_OUT = WS / "data/architecture_interaction_bootstrap"
MOVEMENT_READER = WS / "scripts/selected_prediction_movement_reader.py"

ARMS = ["full_compact", "full_repeat", "nodis_compact", "nodis_repeat"]
COMPACT_ARMS = {"full": "full_compact", "nodis": "nodis_compact"}
REPEAT_ARMS = {"full": "full_repeat", "nodis": "nodis_repeat"}
STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GLOBAL_COLUMNS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
SUBSETS: dict[str, list[str]] = {
    "stable_five_item_pool": STABLE_COLUMNS,
    "EWoK_plus_Entity_item_pool": ["EWoK", "Entity"],
    "BLiMP": ["BLiMP"],
    "Supplement": ["Supplement"],
    "EWoK": ["EWoK"],
    "Entity": ["Entity"],
    "COMPS": ["COMPS"],
    "GlobalPIQA_item_pool": GLOBAL_COLUMNS,
}
SCORE_KEYS = [
    "cheap7",
    "cheap6_no_GlobalPIQA",
    "cheap5_no_GlobalPIQA_Reading",
    "EWoK_plus_Entity",
    "BLiMP",
    "Supplement",
    "EWoK",
    "Entity",
    "COMPS",
    "GlobalPIQA",
    "Reading",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(x: str | Path | None) -> Path | None:
    """Resolve recorded artifact paths deterministically against the user root.

    Do not first probe the caller's current working directory: a future invocation
    from a different cwd could otherwise substitute an unrelated relative file.
    Recorded artifacts use root-relative paths.
    """
    if x is None or str(x) == "":
        return None
    p = Path(str(x))
    if p.is_absolute():
        return p
    return ROOT / p


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
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ys[lo]
        frac = pos - lo
        return ys[lo] * (1.0 - frac) + ys[hi] * frac

    return {"p025": q(0.025), "p50": q(0.5), "p975": q(0.975)}


def bootstrap_values(values: list[float], label: str, n_bootstrap: int) -> dict[str, Any]:
    n = len(values)
    point = 100.0 * (sum(values) / n) if n else None
    if n == 0 or n_bootstrap <= 0:
        return {"n": n, "point_pp": point, "bootstrap": None}
    if np is not None:
        arr = np.asarray(values, dtype=np.float32)
        rng = np.random.default_rng(deterministic_seed(label))
        vals = [float(arr[rng.integers(0, n, size=n, endpoint=False)].mean() * 100.0) for _ in range(n_bootstrap)]
    else:
        import random

        rng = random.Random(deterministic_seed(label))
        vals = [100.0 * sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_bootstrap)]
    return {"n": n, "point_pp": point, "bootstrap": {"n_bootstrap": n_bootstrap, **quantiles(vals)}}


def cluster_bootstrap(rows: list[dict[str, Any]], value_key: str, label: str, n_bootstrap: int) -> dict[str, Any]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        clusters[f"{r.get('column')}::{r.get('subtask')}"].append(float(r[value_key]))
    citems = list(clusters.items())
    n_clusters = len(citems)
    n_items = sum(len(v) for _, v in citems)
    point = 100.0 * sum(sum(v) for _, v in citems) / n_items if n_items else None
    if n_clusters == 0 or n_bootstrap <= 0:
        return {"n_items": n_items, "n_clusters": n_clusters, "point_pp": point, "bootstrap": None}
    if np is not None:
        sums = np.asarray([sum(v) for _, v in citems], dtype=np.float64)
        sizes = np.asarray([len(v) for _, v in citems], dtype=np.float64)
        rng = np.random.default_rng(deterministic_seed(label + ":cluster"))
        vals = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_clusters, size=n_clusters, endpoint=False)
            vals.append(float(100.0 * sums[idx].sum() / sizes[idx].sum()))
    else:
        import random

        rng = random.Random(deterministic_seed(label + ":cluster"))
        vals = []
        for _ in range(n_bootstrap):
            s = 0.0
            z = 0
            for _j in range(n_clusters):
                _, v = citems[rng.randrange(n_clusters)]
                s += sum(v)
                z += len(v)
            vals.append(100.0 * s / z if z else 0.0)
    return {"n_items": n_items, "n_clusters": n_clusters, "point_pp": point, "bootstrap": {"n_bootstrap": n_bootstrap, **quantiles(vals)}}


def load_step218_module():
    spec = importlib.util.spec_from_file_location("selected_prediction_movement_reader", MOVEMENT_READER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {MOVEMENT_READER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def load_panel(panel_dir: Path) -> dict[str, Any]:
    summary = panel_dir / "architecture_interaction_selected_panel_summary.json"
    plan = panel_dir / "selected_panel_plan.json"
    if summary.exists():
        obj = read_json(summary)
        obj["_source_path"] = str(summary)
        return obj
    if plan.exists():
        obj = read_json(plan)
        obj["_source_path"] = str(plan)
        obj.setdefault("rows", [])
        obj.setdefault("interactions", [])
        return obj
    return {"status": "MISSING_PANEL", "_source_path": str(summary), "rows": [], "interactions": [], "checkpoints": []}


def row_by_arm_ck(panel: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r.get("arm")), str(r.get("checkpoint"))): r for r in panel.get("rows", [])}


def checkpoint_key(ck: str) -> int:
    if ck.startswith("chck_") and ck.endswith("M"):
        try:
            return int(ck[len("chck_"):-1])
        except Exception:
            pass
    return 10**9


def requested_checkpoints(panel: dict[str, Any], explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    cks = {str(r.get("checkpoint")) for r in panel.get("rows", []) if r.get("checkpoint")}
    if not cks:
        cks = {str(c) for c in panel.get("checkpoints", []) if c}
    return sorted(cks, key=checkpoint_key)


def per_target_from_row(row: dict[str, Any] | None) -> Path | None:
    if not row:
        return None
    p = resolve(row.get("per_target"))
    if p and p.exists():
        return p
    # Fallback through summary_path if the row points to a research custom-eval summary.
    sp = resolve(row.get("summary_path"))
    if sp is None or not sp.exists():
        return None
    obj = read_json(sp)
    if isinstance(obj, dict) and "tasks" in obj:
        return sp
    if isinstance(obj, dict) and isinstance(obj.get("record"), dict):
        return resolve(obj["record"].get("per_target"))
    return None


def official_panel_interaction(panel: dict[str, Any], ck: str) -> dict[str, Any] | None:
    for rec in panel.get("interactions", []):
        if rec.get("checkpoint") == ck:
            return rec
    return None


def load_arm_items(mod: Any, pt: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    record = mod.load_eval_record(pt)
    items, meta = mod.load_items_from_record(record)
    return {r["item_id"]: r for r in items}, meta


def compute_checkpoint(panel: dict[str, Any], ck: str, out_dir: Path, n_bootstrap: int, max_examples: int) -> dict[str, Any]:
    by = row_by_arm_ck(panel)
    per_targets = {arm: per_target_from_row(by.get((arm, ck))) for arm in ARMS}
    missing = {arm: str(pt) if pt is not None and pt.exists() else None for arm, pt in per_targets.items()}
    ready = all(pt is not None and pt.exists() for pt in per_targets.values())
    rec_base: dict[str, Any] = {
        "checkpoint": ck,
        "ready": bool(ready),
        "per_targets": missing,
        "official_panel_interaction": official_panel_interaction(panel, ck),
    }
    if not ready:
        absent = [arm for arm, pt in per_targets.items() if not (pt is not None and pt.exists())]
        rec_base["status"] = "missing_per_target"
        rec_base["missing_arms"] = absent
        return rec_base

    mod = load_step218_module()
    arm_items: dict[str, dict[str, dict[str, Any]]] = {}
    metas: dict[str, Any] = {}
    for arm, pt in per_targets.items():
        assert pt is not None
        arm_items[arm], metas[arm] = load_arm_items(mod, pt)

    common_ids = sorted(set.intersection(*(set(x) for x in arm_items.values())))
    all_counts = {arm: len(arm_items[arm]) for arm in ARMS}
    missing_by_arm = {arm: len(set.union(*(set(x) for x in arm_items.values())) - set(arm_items[arm])) for arm in ARMS}
    identity_mismatches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for item_id in common_ids:
        f_c = arm_items["full_compact"][item_id]
        f_r = arm_items["full_repeat"][item_id]
        n_c = arm_items["nodis_compact"][item_id]
        n_r = arm_items["nodis_repeat"][item_id]
        identity_fields = ["column", "subtask", "index", "target"]
        field_values = {field: {arm: arm_items[arm][item_id].get(field) for arm in ARMS} for field in identity_fields}
        bad_fields = {field: vals for field, vals in field_values.items() if len({str(v) for v in vals.values()}) != 1}
        if bad_fields:
            if len(identity_mismatches) < 50:
                identity_mismatches.append({"item_id": item_id, "bad_fields": bad_fields})
            continue
        # Use the full-compact metadata as the stable key carrier after four-way identity agreement.
        full_delta = (1 if f_c["correct"] else 0) - (1 if f_r["correct"] else 0)
        nodis_delta = (1 if n_c["correct"] else 0) - (1 if n_r["correct"] else 0)
        interaction = nodis_delta - full_delta
        rows.append({
            "item_id": item_id,
            "column": f_c.get("column"),
            "subtask": f_c.get("subtask"),
            "index": f_c.get("index"),
            "full_repeat_correct": bool(f_r["correct"]),
            "full_compact_correct": bool(f_c["correct"]),
            "nodis_repeat_correct": bool(n_r["correct"]),
            "nodis_compact_correct": bool(n_c["correct"]),
            "full_delta": full_delta,
            "nodis_delta": nodis_delta,
            "interaction": interaction,
            "target": f_c.get("target"),
        })

    if identity_mismatches:
        return {
            **rec_base,
            "status": "four_way_item_identity_mismatch",
            "all_arm_item_counts": all_counts,
            "missing_id_count_by_arm": missing_by_arm,
            "n_common_items_four_cell": len(common_ids),
            "identity_mismatch_examples": identity_mismatches,
            "boundary": "Four-cell item interaction not computed because common item IDs did not agree on column/subtask/index/target across all arms.",
        }

    subset_results: dict[str, Any] = {}
    for subset_name, cols in SUBSETS.items():
        subset = [r for r in rows if r.get("column") in cols]
        subset_results[subset_name] = {
            "n_items": len(subset),
            "interaction_item_bootstrap": bootstrap_values([float(r["interaction"]) for r in subset], f"{ck}|{subset_name}|interaction", n_bootstrap),
            "full_delta_item_bootstrap": bootstrap_values([float(r["full_delta"]) for r in subset], f"{ck}|{subset_name}|full_delta", n_bootstrap),
            "nodis_delta_item_bootstrap": bootstrap_values([float(r["nodis_delta"]) for r in subset], f"{ck}|{subset_name}|nodis_delta", n_bootstrap),
            "interaction_subtask_cluster_bootstrap": cluster_bootstrap(subset, "interaction", f"{ck}|{subset_name}|interaction", n_bootstrap),
            "full_delta_subtask_cluster_bootstrap": cluster_bootstrap(subset, "full_delta", f"{ck}|{subset_name}|full_delta", n_bootstrap),
            "nodis_delta_subtask_cluster_bootstrap": cluster_bootstrap(subset, "nodis_delta", f"{ck}|{subset_name}|nodis_delta", n_bootstrap),
        }

    by_column: dict[str, Any] = {}
    for col in [*STABLE_COLUMNS, *GLOBAL_COLUMNS]:
        subset = [r for r in rows if r.get("column") == col]
        if not subset:
            continue
        by_column[col] = {
            "n": len(subset),
            "full_delta_pp": 100.0 * sum(float(r["full_delta"]) for r in subset) / len(subset),
            "nodis_delta_pp": 100.0 * sum(float(r["nodis_delta"]) for r in subset) / len(subset),
            "interaction_pp": 100.0 * sum(float(r["interaction"]) for r in subset) / len(subset),
        }

    # Keep examples compact and stable-column focused.
    gains = [r for r in rows if r["interaction"] > 0 and r.get("column") in STABLE_COLUMNS]
    losses = [r for r in rows if r["interaction"] < 0 and r.get("column") in STABLE_COLUMNS]
    example_fields = ["item_id", "column", "subtask", "full_delta", "nodis_delta", "interaction", "target"]
    examples = {
        "positive_interaction_stable_examples": [{k: r.get(k) for k in example_fields} for r in gains[:max_examples]],
        "negative_interaction_stable_examples": [{k: r.get(k) for k in example_fields} for r in losses[:max_examples]],
    }

    csv_path = out_dir / ck / "four_cell_item_interaction_rows.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "item_id", "column", "subtask", "index", "full_repeat_correct", "full_compact_correct",
            "nodis_repeat_correct", "nodis_compact_correct", "full_delta", "nodis_delta", "interaction", "target",
        ]
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in fields})

    return {
        **rec_base,
        "status": "ran",
        "all_arm_item_counts": all_counts,
        "missing_id_count_by_arm": missing_by_arm,
        "n_common_items_four_cell": len(rows),
        "identity_mismatch_count": 0,
        "loader_warnings_count": sum(len((metas[a].get("warnings") or [])) for a in ARMS),
        "skipped_columns": {a: metas[a].get("skipped") for a in ARMS},
        "by_column_item_interaction": by_column,
        "subset_results": subset_results,
        "examples": examples,
        "item_rows_csv": str(csv_path),
        "note": "Raw item interaction intervals are interpretive; official selected-panel aggregate interaction remains primary for score columns.",
    }


def summarize_across_checkpoints(checkpoint_results: list[dict[str, Any]], n_bootstrap: int) -> dict[str, Any]:
    ready = [r for r in checkpoint_results if r.get("status") == "ran"]
    if not ready:
        return {}
    out: dict[str, Any] = {}
    for subset_name in SUBSETS:
        vals: list[float] = []
        fulls: list[float] = []
        nodiss: list[float] = []
        for r in ready:
            sub = (r.get("subset_results") or {}).get(subset_name) or {}
            # Store one value per checkpoint as its item-pool point in fraction units.
            ib = sub.get("interaction_item_bootstrap") or {}
            fb = sub.get("full_delta_item_bootstrap") or {}
            nb = sub.get("nodis_delta_item_bootstrap") or {}
            if ib.get("point_pp") is not None:
                vals.append(float(ib["point_pp"]) / 100.0)
            if fb.get("point_pp") is not None:
                fulls.append(float(fb["point_pp"]) / 100.0)
            if nb.get("point_pp") is not None:
                nodiss.append(float(nb["point_pp"]) / 100.0)
        out[subset_name] = {
            "checkpoints": [r.get("checkpoint") for r in ready],
            "mean_checkpoint_interaction_pp": 100.0 * sum(vals) / len(vals) if vals else None,
            "mean_checkpoint_full_delta_pp": 100.0 * sum(fulls) / len(fulls) if fulls else None,
            "mean_checkpoint_nodis_delta_pp": 100.0 * sum(nodiss) / len(nodiss) if nodiss else None,
            "note": "Mean of checkpoint-level raw item-pool points; not an item-resampled longitudinal uncertainty estimate.",
        }
    return out


def write_md(payload: dict[str, Any], out_dir: Path) -> None:
    lines: list[str] = []
    lines.append("# research architecture-interaction four-cell item bootstrap")
    lines.append("")
    lines.append(f"Created UTC: `{payload.get('created_utc')}`")
    lines.append("")
    lines.append("This is CPU/file-only interpretation of existing selected predictions. It does not train, score, run SuperGLUE/AoA, upload, or submit.")
    lines.append("")
    for rec in payload.get("checkpoint_results", []):
        lines.append(f"## {rec.get('checkpoint')}")
        lines.append(f"- status: `{rec.get('status')}`")
        if rec.get("missing_arms"):
            lines.append(f"- missing arms: `{rec.get('missing_arms')}`")
        if rec.get("status") == "ran":
            stable = rec.get("subset_results", {}).get("stable_five_item_pool", {})
            ew_ent = rec.get("subset_results", {}).get("EWoK_plus_Entity_item_pool", {})
            for label, sub in [("stable_five", stable), ("EWoK_plus_Entity", ew_ent)]:
                ib = sub.get("interaction_item_bootstrap", {})
                cb = sub.get("interaction_subtask_cluster_bootstrap", {})
                lines.append(f"- {label} interaction item point pp: `{ib.get('point_pp')}` interval `{(ib.get('bootstrap') or {})}`")
                lines.append(f"- {label} interaction cluster point pp: `{cb.get('point_pp')}` interval `{(cb.get('bootstrap') or {})}`")
        lines.append("")
    lines.append("## Boundary")
    lines.append(payload.get("boundary", ""))
    (out_dir / "architecture_interaction_four_cell_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checkpoints", nargs="+", default=None)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=400)
    ap.add_argument("--max-examples", type=int, default=12)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(args.panel_dir)
    checkpoints = requested_checkpoints(panel, args.checkpoints)
    by = row_by_arm_ck(panel)
    plan_comparisons: list[dict[str, Any]] = []
    for ck in checkpoints:
        for arm in ARMS:
            row = by.get((arm, ck))
            pt = per_target_from_row(row)
            plan_comparisons.append({
                "checkpoint": ck,
                "arm": arm,
                "row_present": row is not None,
                "per_target": str(pt) if pt is not None else None,
                "ready": bool(pt is not None and pt.exists()),
            })

    checkpoint_results: list[dict[str, Any]] = []
    if not args.plan_only:
        for ck in checkpoints:
            checkpoint_results.append(compute_checkpoint(panel, ck, args.out_dir, args.n_bootstrap, args.max_examples))

    payload = {
        "status": "ARCHITECTURE_INTERACTION_FOUR_CELL_BOOTSTRAP",
        "created_utc": now(),
        "panel_dir": str(args.panel_dir),
        "panel_source": panel.get("_source_path"),
        "out_dir": str(args.out_dir),
        "checkpoints": checkpoints,
        "plan_comparisons": plan_comparisons,
        "ready_count": sum(1 for r in plan_comparisons if r["ready"]),
        "needed_count": len(plan_comparisons),
        "ran": not args.plan_only,
        "checkpoint_results": checkpoint_results,
        "mean_checkpoint_item_interactions": summarize_across_checkpoints(checkpoint_results, args.n_bootstrap),
        "meaning": "Four-cell raw item interaction: (nodis_compact-nodis_repeat) - (full_compact-full_repeat), in percentage points. Official selected panel aggregate interaction remains primary for BabyLM score columns.",
        "boundary": "No training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission was performed.",
    }
    out_json = args.out_dir / "architecture_interaction_four_cell_bootstrap.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, args.out_dir)
    print(json.dumps({
        "status": payload["status"],
        "out": str(out_json),
        "ready_count": payload["ready_count"],
        "needed_count": payload["needed_count"],
        "ran": payload["ran"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
