#!/usr/bin/env python3
"""research: Entity depth and V-R/V-C decomposition from existing official reports.

Scientific purpose:
  The replicated Entity advantage must be decomposed by operation depth before it
  can be interpreted. A shallow format/register effect should mostly lift 0-op
  items; a state-tracking/binding-maintenance effect should persist or grow at
  higher operation counts. This script uses existing official Entity reports for
  DeBERTa VIEW/CLEAN/REPEAT seed43022 and seed43122 at 80M/90M/100M.

No training, no benchmark evaluation, no upload.
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
import re
import statistics
from collections import defaultdict
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/entity_depth_vr_readout.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
OUT = _public_path('experiments/archive/relation_learning/data/entity_depth')
OUT.mkdir(parents=True, exist_ok=True)

CKS = ["chck_80M", "chck_90M", "chck_100M"]
ARMS = ["V", "C", "R"]
SEEDS = [43022, 43122]

ENTITY_DIR = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')
S274 = _public_path('experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout')
S268 = _public_path('experiments/archive/frontier_consolidation/data/second_basin_entity_ewok_eval')
S005 = _public_path('experiments/archive/relation_learning/data/vc_seed_replication')


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def load_uid_counts() -> dict[str, int]:
    """Official Entity scoring drops any item whose answer options include 'nothing'.

    The raw JSONL files include many 0-op rows with a gold or foil 'nothing' value;
    the evaluator predictions are generated and scored after the same filter.  Use
    the filtered counts as weights when converting per-UID accuracies into depth
    means, otherwise shallow rows are overweighted.
    """
    counts: dict[str, int] = {}
    for fn in ["regular.jsonl", "ambiref.jsonl", "move_contents.jsonl"]:
        typ = fn[:-6]
        by_depth: dict[int, int] = defaultdict(int)
        with (ENTITY_DIR / fn).open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if any("nothing" in str(option).lower() for option in obj.get("options", [])):
                    continue
                by_depth[int(obj["numops"])] += 1
        for depth, n in sorted(by_depth.items()):
            counts[f"{typ}_{depth}_ops"] = n
    expected = {
        "regular_0_ops": 517, "regular_1_ops": 409, "regular_2_ops": 405, "regular_3_ops": 425, "regular_4_ops": 388, "regular_5_ops": 94,
        "ambiref_0_ops": 508, "ambiref_1_ops": 428, "ambiref_2_ops": 413, "ambiref_3_ops": 409, "ambiref_4_ops": 434, "ambiref_5_ops": 123,
        "move_contents_0_ops": 516, "move_contents_1_ops": 437, "move_contents_2_ops": 399, "move_contents_3_ops": 406, "move_contents_4_ops": 353, "move_contents_5_ops": 116,
    }
    if counts != expected:
        raise RuntimeError({"filtered_counts": counts, "expected_official_counts": expected})
    return counts


def parse_report(report_path: pathlib.Path) -> dict[str, float]:
    text = report_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {}
    for m in re.finditer(r"\b(regular|ambiref|move_contents)_(\d+)_ops:\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text):
        out[f"{m.group(1)}_{m.group(2)}_ops"] = float(m.group(3))
    if len(out) != 18:
        raise ValueError(f"Expected 18 Entity UID rows in {report_path}, found {len(out)}")
    return out


def payload_report(payload_path: pathlib.Path) -> pathlib.Path:
    obj = json.loads(payload_path.read_text(encoding="utf-8"))
    report = obj.get("tasks", {}).get("Entity", {}).get("report")
    if not report:
        raise KeyError(f"No Entity report in {payload_path}")
    p = ROOT / report
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def build_seed43022_payload_map() -> dict[tuple[int, str, str], pathlib.Path]:
    role_to_arm = {"V": "deberta_basin1_view", "C": "deberta_basin1_clean_maxgeom", "R": "deberta_basin1_repeat"}
    out: dict[tuple[int, str, str], pathlib.Path] = {}
    with (_public_path('experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout/score_rows.csv')).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("seed") != "43022" or row.get("family") != "Entity" or row.get("checkpoint") not in CKS:
                continue
            for role, arm in role_to_arm.items():
                if row.get("arm") == arm:
                    out[(43022, role, row["checkpoint"])] = ROOT / row["payload"]
    return out


def find_step005_report(arm: str, ck: str) -> pathlib.Path:
    d = _public_path('experiments/archive/relation_learning/data/vc_seed_replication/outputs') / f"D_{arm}_43122_{ck}" / "Entity"
    # actual arm strings are D_V_43122 etc.; call with V/C.
    if not d.exists():
        d = _public_path('experiments/archive/relation_learning/data/vc_seed_replication/outputs') / f"D_{arm}_43122_{ck}" / "Entity"
    matches = sorted(d.rglob("best_temperature_report.txt"))
    if not matches:
        raise FileNotFoundError(f"No report under {d}")
    return matches[-1]


def get_report(seed: int, role: str, ck: str, seed43022_payloads: dict[tuple[int, str, str], pathlib.Path]) -> pathlib.Path:
    if seed == 43022:
        return payload_report(seed43022_payloads[(seed, role, ck)])
    if seed == 43122 and role in {"V", "C"}:
        return find_step005_report(role, ck)
    if seed == 43122 and role == "R":
        payload = _public_path('experiments/archive/frontier_consolidation/data/second_basin_entity_ewok_eval/repeat_eval/per_target') / f"second_basin_max_repeat_seed43122_{ck}.json"
        return payload_report(payload)
    raise KeyError((seed, role, ck))


def weighted_mean(vals: list[tuple[float, int]]) -> float:
    den = sum(w for _, w in vals)
    return sum(v * w for v, w in vals) / den if den else float("nan")


def main() -> None:
    counts = load_uid_counts()
    seed43022_payloads = build_seed43022_payload_map()

    uid_rows: list[dict[str, Any]] = []
    report_paths: dict[str, str] = {}
    scores: dict[tuple[int, str, str, str], float] = {}

    for seed in SEEDS:
        for role in ARMS:
            for ck in CKS:
                report = get_report(seed, role, ck, seed43022_payloads)
                report_paths[f"{seed}_{role}_{ck}"] = rel(report)
                uid_scores = parse_report(report)
                for uid, acc in sorted(uid_scores.items()):
                    typ, depth_s, _ = uid.rsplit("_", 2)
                    depth = int(depth_s)
                    scores[(seed, role, ck, uid)] = acc
                    uid_rows.append({
                        "seed": seed,
                        "arm": role,
                        "checkpoint": ck,
                        "uid": uid,
                        "entity_type": typ,
                        "numops": depth,
                        "n_items": counts.get(uid, 0),
                        "accuracy": acc,
                        "report": rel(report),
                    })

    # Contrasts by UID.
    contrast_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for ck in CKS:
            for contrast, a, b in [("VminusC", "V", "C"), ("VminusR", "V", "R"), ("CminusR", "C", "R")]:
                for uid in sorted(counts):
                    typ, depth_s, _ = uid.rsplit("_", 2)
                    va = scores[(seed, a, ck, uid)]
                    vb = scores[(seed, b, ck, uid)]
                    contrast_rows.append({
                        "seed": seed,
                        "checkpoint": ck,
                        "contrast": contrast,
                        "uid": uid,
                        "entity_type": typ,
                        "numops": int(depth_s),
                        "n_items": counts[uid],
                        "score_a": va,
                        "score_b": vb,
                        "delta": va - vb,
                    })

    # Weighted summaries by depth and by type/depth.
    summary_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for ck in CKS:
            for contrast in ["VminusC", "VminusR", "CminusR"]:
                subset = [r for r in contrast_rows if r["seed"] == seed and r["checkpoint"] == ck and r["contrast"] == contrast]
                for depth in range(6):
                    vals = [(float(r["delta"]), int(r["n_items"])) for r in subset if r["numops"] == depth]
                    summary_rows.append({
                        "seed": seed,
                        "checkpoint": ck,
                        "contrast": contrast,
                        "group": "ALL_TYPES",
                        "numops": depth,
                        "n_items": sum(w for _, w in vals),
                        "weighted_delta": weighted_mean(vals),
                    })
                    for typ in ["regular", "ambiref", "move_contents"]:
                        vals_t = [(float(r["delta"]), int(r["n_items"])) for r in subset if r["numops"] == depth and r["entity_type"] == typ]
                        summary_rows.append({
                            "seed": seed,
                            "checkpoint": ck,
                            "contrast": contrast,
                            "group": typ,
                            "numops": depth,
                            "n_items": sum(w for _, w in vals_t),
                            "weighted_delta": weighted_mean(vals_t),
                        })

    # Late mean summaries over checkpoints.
    late_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for contrast in ["VminusC", "VminusR", "CminusR"]:
            for group in ["ALL_TYPES", "regular", "ambiref", "move_contents"]:
                for depth in range(6):
                    vals = [float(r["weighted_delta"]) for r in summary_rows if r["seed"] == seed and r["contrast"] == contrast and r["group"] == group and r["numops"] == depth]
                    ns = [int(r["n_items"]) for r in summary_rows if r["seed"] == seed and r["contrast"] == contrast and r["group"] == group and r["numops"] == depth]
                    late_rows.append({
                        "seed": seed,
                        "contrast": contrast,
                        "group": group,
                        "numops": depth,
                        "n_items_per_checkpoint": ns[0] if ns else 0,
                        "late_mean_delta": statistics.mean(vals) if vals else float("nan"),
                        "late_sd_delta": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    })

    # Cross-seed late means.
    cross_seed_rows: list[dict[str, Any]] = []
    for contrast in ["VminusC", "VminusR", "CminusR"]:
        for group in ["ALL_TYPES", "regular", "ambiref", "move_contents"]:
            for depth in range(6):
                vals = [float(r["late_mean_delta"]) for r in late_rows if r["contrast"] == contrast and r["group"] == group and r["numops"] == depth]
                cross_seed_rows.append({
                    "contrast": contrast,
                    "group": group,
                    "numops": depth,
                    "cross_seed_mean_delta": statistics.mean(vals),
                    "cross_seed_spread": max(vals) - min(vals),
                    "seed43022_delta": vals[0],
                    "seed43122_delta": vals[1],
                })

    # Save CSVs.
    def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_uid_scores.csv'), uid_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_uid_contrasts.csv'), contrast_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_depth_checkpoint_summary.csv'), summary_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_depth_late_summary.csv'), late_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_depth_cross_seed_summary.csv'), cross_seed_rows)

    # Slopes for ALL_TYPES only.
    slope_rows = []
    for seed in SEEDS:
        for contrast in ["VminusC", "VminusR", "CminusR"]:
            vals = [(r["numops"], float(r["late_mean_delta"])) for r in late_rows if r["seed"] == seed and r["contrast"] == contrast and r["group"] == "ALL_TYPES"]
            xs = [x for x, _ in vals]
            ys = [y for _, y in vals]
            xbar = statistics.mean(xs)
            ybar = statistics.mean(ys)
            denom = sum((x - xbar) ** 2 for x in xs)
            slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom
            slope_rows.append({"seed": seed, "contrast": contrast, "group": "ALL_TYPES", "slope_delta_per_op": slope, "mean_delta": ybar})
    for contrast in ["VminusC", "VminusR", "CminusR"]:
        vals = [(r["numops"], float(r["cross_seed_mean_delta"])) for r in cross_seed_rows if r["contrast"] == contrast and r["group"] == "ALL_TYPES"]
        xs = [x for x, _ in vals]
        ys = [y for _, y in vals]
        xbar = statistics.mean(xs); ybar = statistics.mean(ys)
        denom = sum((x - xbar) ** 2 for x in xs)
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom
        slope_rows.append({"seed": "cross_seed_mean", "contrast": contrast, "group": "ALL_TYPES", "slope_delta_per_op": slope, "mean_delta": ybar})
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_depth_slopes.csv'), slope_rows)

    summary = {
        "status": "ENTITY_DEPTH_DONE",
        "entity_uid_counts": counts,
        "report_paths": report_paths,
        "files": {
            "uid_scores": rel(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_uid_scores.csv')),
            "uid_contrasts": rel(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_uid_contrasts.csv')),
            "checkpoint_summary": rel(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_depth_checkpoint_summary.csv')),
            "late_summary": rel(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_depth_late_summary.csv')),
            "cross_seed_summary": rel(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_depth_cross_seed_summary.csv')),
            "slopes": rel(_public_path('experiments/archive/relation_learning/data/entity_depth/entity_depth_slopes.csv')),
        },
        "interpretation_hint": "Read cross_seed_summary and slopes: V-R separates presentation diversity from corpus content; depth trend distinguishes shallow format from binding/state-maintenance effects.",
    }
    (_public_path('experiments/archive/relation_learning/data/entity_depth/entity_depth_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Print compact decisive table.
    print("ENTITY DEPTH LATE MEAN DELTAS (80/90/100M mean, ALL_TYPES)")
    print("contrast    seed        op0    op1    op2    op3    op4    op5    slope/op   mean")
    for row in slope_rows:
        seed = row["seed"]
        contrast = row["contrast"]
        if seed == "cross_seed_mean":
            vals = [r for r in cross_seed_rows if r["contrast"] == contrast and r["group"] == "ALL_TYPES"]
            vals = [float(r["cross_seed_mean_delta"]) for r in sorted(vals, key=lambda x: x["numops"])]
        else:
            vals = [r for r in late_rows if r["seed"] == seed and r["contrast"] == contrast and r["group"] == "ALL_TYPES"]
            vals = [float(r["late_mean_delta"]) for r in sorted(vals, key=lambda x: x["numops"])]
        print(f"{contrast:<10s} {str(seed):<10s} " + " ".join(f"{v:+6.2f}" for v in vals) + f" {row['slope_delta_per_op']:+8.3f} {row['mean_delta']:+7.2f}")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
