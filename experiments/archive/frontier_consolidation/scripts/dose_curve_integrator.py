#!/usr/bin/env python3
"""research: integrate the 1x--1.82x--2.64x compact-dose curve.

CPU/file-only readout.  The script requires complete MAX and midpoint selected-
family ladders before it writes a curve and continuation readout.  It treats the
algebraic identity

    V - C = (V - R) + (R - C)

as arithmetic, not as proof of mechanism.  The main clean within-dose contrast is
V-R.  Dose growth is also computed in clean-free form:

    V_d - V_1 = [(V_d-R_d) - (V_1-R_1)] + (R_d - R_1),

so source/total growth can be compared over all ten checkpoints without relying
on the sparse clean rows.

research separately launched RoBERTa view+clean as the amplified total fixed-
budget transfer test, independent of this DeBERTa curve.  research froze the
pre-result reading order and replication rule before any new ladder scores were
read.  This readout therefore does not add another construction vertex.  It
identifies which MAX-dose leg, if any, is strong enough in cheap6 and cheap5 to
merit a second-seed MAX replication across an independent basin.  A RoBERTa
repeat arm is only a later decomposition possibility after the active RoBERTa
view-clean result is read.  EWoK+Entity is reported as secondary evidence and
never drives continuation alone.  No training, GPU evaluation, SuperGLUE, AoA,
upload, or leaderboard action is performed here.
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
WS = ROOT / "experiments/archive/frontier_consolidation"
MAX_DIR_DEFAULT = WS / "data/dose_ladder_stable_eval"
MID_DIR_DEFAULT = WS / "data/intermediate_dose_ladder_stable_eval"
OUT_DIR_DEFAULT = WS / "data/dose_curve_integrator"
SEED_SPREAD_DEFAULT = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_seed_spread.csv"
MAX_DIST_META = WS / "data/dose_distribution_select/dose_distribution_comparison_and_selection.json"
MID_SELECT_META = WS / "data/dose_intermediate_select/dose1p82_selection_metadata.json"
GEOMETRY_RECORD = WS / "data/training_geometry_audit/training_geometry_audit.json"
PRE_RESULT_READOUT_NOTE = WS / "notes/pre_result_reading_order_and_replication_plan.md"
BREADTH_GEOMETRY = WS / "data/breadth_arm_geometry_audit/breadth_geometry_audit.json"
POOL_TOKEN_ACCOUNTING = WS / "data/pool_token_accounting/pool_token_accounting.json"

BASE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
DERIVED = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]
METRICS = [*BASE_COLUMNS, *DERIVED]
PRIMARY = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading"]
SECONDARY = ["EWoK_plus_Entity_sum"]
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 101, 10)]
CLEAN_CHECKPOINTS_EXPECTED = {"chck_20M", "chck_70M", "chck_80M"}
EPS = 1e-9

DOSE_INFO = {
    "clean0": {"dose": 0.0, "rho": 0.0, "data_arm": "clean", "dose_name": "clean0"},
    "dose1_view": {"dose": 1.0, "rho": 0.042352, "data_arm": "view", "dose_name": "dose1"},
    "dose1_repeat": {"dose": 1.0, "rho": 0.042352, "data_arm": "repeat", "dose_name": "dose1"},
    "dose1p82_view": {"dose": 1.8209293539856442, "rho": 0.07712, "data_arm": "view", "dose_name": "dose1p82"},
    "dose1p82_repeat": {"dose": 1.8209293539856442, "rho": 0.07712, "data_arm": "repeat", "dose_name": "dose1p82"},
    "max_view": {"dose": 2.641480921798262, "rho": 0.111872, "data_arm": "view", "dose_name": "dose2p64"},
    "max_repeat": {"dose": 2.641480921798262, "rho": 0.111872, "data_arm": "repeat", "dose_name": "dose2p64"},
}
PAIR_ARMS = {
    "dose1": ("dose1_view", "dose1_repeat"),
    "dose1p82": ("dose1p82_view", "dose1p82_repeat"),
    "dose2p64": ("max_view", "max_repeat"),
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def fnum(x: Any) -> float | None:
    return float(x) if finite(x) else None


def ck_words(ck: str) -> int:
    return int(str(ck)[5:-1]) * 1_000_000


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fields:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def load_summary(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "ok": False, "reason": "missing"}
    try:
        data = read_json(path)
    except Exception as exc:
        return {"path": str(path), "exists": True, "ok": False, "reason": repr(exc)}
    status = str(data.get("status"))
    ok = ("DONE" in status and int(data.get("problem_count") or 0) == 0)
    return {"path": str(path), "exists": True, "ok": ok, "status": status, "problem_count": data.get("problem_count"), "row_count": data.get("row_count"), "raw": data}


def recompute_derived(r: dict[str, Any]) -> dict[str, float | None]:
    vals = {c: fnum(r.get(c)) for c in BASE_COLUMNS}
    out = {c: vals[c] for c in BASE_COLUMNS}
    if all(vals[c] is not None for c in BASE_COLUMNS):
        out["cheap6_no_GlobalPIQA"] = sum(float(vals[c]) for c in BASE_COLUMNS) / 6.0
    else:
        out["cheap6_no_GlobalPIQA"] = None
    if all(vals[c] is not None for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]):
        out["cheap5_no_GlobalPIQA_Reading"] = sum(float(vals[c]) for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]) / 5.0
    else:
        out["cheap5_no_GlobalPIQA_Reading"] = None
    if vals["EWoK"] is not None and vals["Entity"] is not None:
        out["EWoK_plus_Entity_sum"] = float(vals["EWoK"]) + float(vals["Entity"])
    else:
        out["EWoK_plus_Entity_sum"] = None
    return out


def normalize_rows(path: pathlib.Path, family: str) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(path):
        arm = str(raw.get("arm"))
        if arm not in DOSE_INFO:
            continue
        info = DOSE_INFO[arm]
        ck = str(raw.get("checkpoint"))
        rec = {
            "source_file": str(path),
            "source_family": family,
            "arm": arm,
            "dose_name": info["dose_name"],
            "dose": info["dose"],
            "rho": info["rho"],
            "data_arm": info["data_arm"],
            "checkpoint": ck,
            "words": int(float(raw.get("words") or ck_words(ck))),
            "per_target": raw.get("per_target"),
            "run_dir": raw.get("run_dir"),
        }
        derived = recompute_derived(raw)
        for m, v in derived.items():
            rec[m] = v
        # Keep supplied composite mismatch visible, but use recomputed values.
        mismatches = []
        for m in DERIVED:
            if finite(raw.get(m)) and rec[m] is not None and abs(float(raw[m]) - float(rec[m])) > 1e-6:
                mismatches.append({"metric": m, "supplied": float(raw[m]), "recomputed": rec[m]})
        rec["composite_mismatch_json"] = json.dumps(mismatches, ensure_ascii=False) if mismatches else "[]"
        rows.append(rec)
    return rows


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prio = {"midpoint": 2, "max_reuse": 1}
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        key = (str(r["arm"]), str(r["checkpoint"]))
        if key not in by or prio.get(str(r.get("source_family")), 0) > prio.get(str(by[key].get("source_family")), 0):
            by[key] = r
    return sorted(by.values(), key=lambda r: (float(r["rho"]), str(r["data_arm"]), int(r["words"])))


def load_seed_spread(path: pathlib.Path) -> dict[tuple[str, str], float]:
    mapping = {
        "BLiMP": "treatment_delta_seed43122_minus_seed43022_BLiMP",
        "Supplement": "treatment_delta_seed43122_minus_seed43022_Supplement",
        "EWoK": "treatment_delta_seed43122_minus_seed43022_EWoK",
        "Entity": "treatment_delta_seed43122_minus_seed43022_Entity",
        "COMPS": "treatment_delta_seed43122_minus_seed43022_COMPS",
        "Reading": "treatment_delta_seed43122_minus_seed43022_Reading",
        "cheap6_no_GlobalPIQA": "treatment_delta_seed43122_minus_seed43022_cheap6_no_GlobalPIQA",
        "cheap5_no_GlobalPIQA_Reading": "treatment_delta_seed43122_minus_seed43022_cheap5_no_GlobalPIQA_Reading",
        "EWoK_plus_Entity_sum": "treatment_delta_seed43122_minus_seed43022_EWoK_plus_Entity_sum",
    }
    out: dict[tuple[str, str], float] = {}
    for r in read_csv(path):
        ck = str(r.get("checkpoint"))
        for m, col in mapping.items():
            if finite(r.get(col)):
                out[(ck, m)] = abs(float(r[col]))
    return out


def validate_census(rows: list[dict[str, Any]]) -> tuple[bool, list[str], dict[str, Any]]:
    by = {(str(r["arm"]), str(r["checkpoint"])): r for r in rows}
    problems: list[str] = []
    counts: dict[str, Any] = {}
    for arm in ["dose1_view", "dose1_repeat", "dose1p82_view", "dose1p82_repeat", "max_view", "max_repeat"]:
        have = sorted([ck for ck in CHECKPOINTS if (arm, ck) in by], key=ck_words)
        counts[arm] = have
        if have != CHECKPOINTS:
            problems.append(f"{arm} has {len(have)}/10 checkpoints: {have}")
    clean_have = sorted([ck for ck in CHECKPOINTS if ("clean0", ck) in by], key=ck_words)
    counts["clean0"] = clean_have
    if not CLEAN_CHECKPOINTS_EXPECTED.issubset(set(clean_have)):
        problems.append(f"clean0 missing expected sparse clean checkpoints {sorted(CLEAN_CHECKPOINTS_EXPECTED - set(clean_have))}")
    for r in rows:
        mm = json.loads(str(r.get("composite_mismatch_json") or "[]"))
        if mm:
            problems.append(f"composite mismatch {r['arm']} {r['checkpoint']}: {mm[:3]}")
    expected_counts = {"paired_model_rows": 60, "clean_sparse_min_rows": len(CLEAN_CHECKPOINTS_EXPECTED)}
    ok = not problems
    return ok, problems, {"checkpoint_census": counts, "expected_counts": expected_counts, "actual_model_rows_excluding_clean": sum(len(counts[a]) for a in counts if a != "clean0"), "clean_rows": len(clean_have)}


def row_by(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r["arm"]), str(r["checkpoint"])): r for r in rows}


def build_leg_rows(rows: list[dict[str, Any]], spread: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    by = row_by(rows)
    out: list[dict[str, Any]] = []
    for dose_name, (view_arm, repeat_arm) in PAIR_ARMS.items():
        info = DOSE_INFO[view_arm]
        for ck in CHECKPOINTS:
            v = by.get((view_arm, ck)); r = by.get((repeat_arm, ck)); c = by.get(("clean0", ck))
            if v is None or r is None:
                continue
            for metric in METRICS:
                vv, rr = fnum(v.get(metric)), fnum(r.get(metric))
                semantic = vv - rr if vv is not None and rr is not None else None
                total = source = resid = None
                if c is not None:
                    cc = fnum(c.get(metric))
                    total = vv - cc if vv is not None and cc is not None else None
                    source = rr - cc if rr is not None and cc is not None else None
                    resid = total - (semantic + source) if all(x is not None for x in [total, semantic, source]) else None
                ss = spread.get((ck, metric))
                out.append({
                    "dose_name": dose_name, "dose": info["dose"], "rho": info["rho"], "checkpoint": ck, "words": ck_words(ck), "metric": metric,
                    "view_arm": view_arm, "repeat_arm": repeat_arm,
                    "semantic_view_minus_repeat": semantic,
                    "total_view_minus_clean_sparse": total,
                    "source_repeat_minus_clean_sparse": source,
                    "sparse_decomposition_residual": resid,
                    "abs_seed_spread_for_metric_checkpoint": ss,
                    "abs_semantic_over_abs_seed_spread": abs(semantic) / ss if semantic is not None and ss and ss > 1e-9 else None,
                })
    return out


def build_growth_rows(rows: list[dict[str, Any]], spread: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    by = row_by(rows)
    out: list[dict[str, Any]] = []
    for dose_name, (view_arm, repeat_arm) in [("dose1p82", PAIR_ARMS["dose1p82"]), ("dose2p64", PAIR_ARMS["dose2p64"])] :
        info = DOSE_INFO[view_arm]
        for ck in CHECKPOINTS:
            v = by.get((view_arm, ck)); r = by.get((repeat_arm, ck)); v1 = by.get(("dose1_view", ck)); r1 = by.get(("dose1_repeat", ck))
            if not all([v, r, v1, r1]):
                continue
            for metric in METRICS:
                vv, rr, v1v, r1v = [fnum(x.get(metric)) for x in [v, r, v1, r1]]  # type: ignore[union-attr]
                sem_d = vv - rr if vv is not None and rr is not None else None
                sem_1 = v1v - r1v if v1v is not None and r1v is not None else None
                semantic_growth = sem_d - sem_1 if sem_d is not None and sem_1 is not None else None
                total_growth = vv - v1v if vv is not None and v1v is not None else None
                source_growth = rr - r1v if rr is not None and r1v is not None else None
                resid = total_growth - (semantic_growth + source_growth) if all(x is not None for x in [total_growth, semantic_growth, source_growth]) else None
                ss = spread.get((ck, metric))
                out.append({
                    "dose_name": dose_name, "dose": info["dose"], "rho": info["rho"], "rho_minus_1x": float(info["rho"]) - float(DOSE_INFO["dose1_view"]["rho"]),
                    "checkpoint": ck, "words": ck_words(ck), "metric": metric,
                    "semantic_growth_vs_1x_cleanfree": semantic_growth,
                    "total_view_growth_vs_1x_cleanfree": total_growth,
                    "source_repeat_growth_vs_1x_cleanfree": source_growth,
                    "cleanfree_growth_residual": resid,
                    "abs_seed_spread_for_metric_checkpoint": ss,
                    "abs_semantic_growth_over_abs_seed_spread": abs(semantic_growth) / ss if semantic_growth is not None and ss and ss > 1e-9 else None,
                    "abs_total_growth_over_abs_seed_spread": abs(total_growth) / ss if total_growth is not None and ss and ss > 1e-9 else None,
                    "abs_source_growth_over_abs_seed_spread": abs(source_growth) / ss if source_growth is not None and ss and ss > 1e-9 else None,
                })
    return out


def stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "pos_frac": None, "neg_frac": None}
    return {"n": len(vals), "mean": float(statistics.mean(vals)), "median": float(statistics.median(vals)), "min": float(min(vals)), "max": float(max(vals)), "pos_frac": sum(v > 0 for v in vals) / len(vals), "neg_frac": sum(v < 0 for v in vals) / len(vals)}


def vals(rows: list[dict[str, Any]], metric: str, key: str, dose_name: str | None = None) -> list[float]:
    out = []
    for r in rows:
        if r.get("metric") != metric:
            continue
        if dose_name is not None and r.get("dose_name") != dose_name:
            continue
        if finite(r.get(key)):
            out.append(float(r[key]))
    return out


def endpoint(rows: list[dict[str, Any]], metric: str, key: str, dose_name: str, ck: str = "chck_100M") -> float | None:
    for r in rows:
        if r.get("metric") == metric and r.get("dose_name") == dose_name and r.get("checkpoint") == ck and finite(r.get(key)):
            return float(r[key])
    return None


def shape_for_metric(legs: list[dict[str, Any]], growth: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    sem_means = {d: stats(vals(legs, metric, "semantic_view_minus_repeat", d))["mean"] for d in ["dose1", "dose1p82", "dose2p64"]}
    g_mid = stats(vals(growth, metric, "semantic_growth_vs_1x_cleanfree", "dose1p82"))
    g_max = stats(vals(growth, metric, "semantic_growth_vs_1x_cleanfree", "dose2p64"))
    inc_max_mid: list[float] = []
    # incremental semantic MAX-mid at matched checkpoints
    leg_lookup = {(r["dose_name"], r["checkpoint"], r["metric"]): r for r in legs}
    for ck in CHECKPOINTS:
        a = leg_lookup.get(("dose2p64", ck, metric)); b = leg_lookup.get(("dose1p82", ck, metric))
        if a and b and finite(a.get("semantic_view_minus_repeat")) and finite(b.get("semantic_view_minus_repeat")):
            inc_max_mid.append(float(a["semantic_view_minus_repeat"]) - float(b["semantic_view_minus_repeat"]))
    inc = stats(inc_max_mid)
    if all(finite(sem_means[d]) for d in sem_means):
        v1, vm, vx = [float(sem_means[d]) for d in ["dose1", "dose1p82", "dose2p64"]]
        flat_tol = 0.05
        if abs(vm - v1) <= flat_tol and abs(vx - v1) <= flat_tol:
            shape = "flat_at_current_resolution"
        elif vm >= v1 and vx >= vm:
            shape = "monotone_or_saturating_increase"
        elif vm >= v1 and vx < vm:
            shape = "turnover_after_midpoint"
        elif vm < v1 and vx < v1:
            shape = "declines_from_1x"
        else:
            shape = "mixed_nonmonotone"
    else:
        shape = "incomplete"
    return {"semantic_means_by_dose": sem_means, "mid_growth_vs_1x": g_mid, "max_growth_vs_1x": g_max, "max_minus_mid_increment": inc, "shape": shape, "endpoint100_semantic": {d: endpoint(legs, metric, "semantic_view_minus_repeat", d) for d in ["dose1", "dose1p82", "dose2p64"]}}


def seed_scale(spread: dict[tuple[str, str], float], metric: str) -> dict[str, Any]:
    xs = [v for (ck, m), v in spread.items() if m == metric and finite(v) and v > 1e-9]
    return stats(xs)


def metric_component(growth: list[dict[str, Any]], spread: dict[tuple[str, str], float], metric: str, dose_name: str = "dose2p64") -> dict[str, Any]:
    rows = [r for r in growth if r.get("metric") == metric and r.get("dose_name") == dose_name]
    comp = {
        "semantic": stats([float(r["semantic_growth_vs_1x_cleanfree"]) for r in rows if finite(r.get("semantic_growth_vs_1x_cleanfree"))]),
        "total": stats([float(r["total_view_growth_vs_1x_cleanfree"]) for r in rows if finite(r.get("total_view_growth_vs_1x_cleanfree"))]),
        "source": stats([float(r["source_repeat_growth_vs_1x_cleanfree"]) for r in rows if finite(r.get("source_repeat_growth_vs_1x_cleanfree"))]),
        "seed_scale": seed_scale(spread, metric),
    }
    for k in ["semantic", "total", "source"]:
        mn = comp[k]["mean"]
        sd = comp["seed_scale"]["mean"]
        comp[k]["mean_over_seed_scale"] = abs(float(mn)) / float(sd) if finite(mn) and finite(sd) and float(sd) > 1e-9 else None
    return comp


def component_vote(component: dict[str, Any]) -> str:
    sem = component["semantic"]; total = component["total"]; source = component["source"]; scale = component["seed_scale"]["mean"]
    min_mag = max(0.05, 0.5 * float(scale)) if finite(scale) else 0.05
    sem_ok = finite(sem["mean"]) and float(sem["mean"]) > min_mag and float(sem.get("pos_frac") or 0.0) >= 0.70
    src_ok = finite(source["mean"]) and float(source["mean"]) > min_mag and float(source.get("pos_frac") or 0.0) >= 0.70 and finite(total["mean"]) and float(total["mean"]) > 0
    if sem_ok and (not src_ok or abs(float(sem["mean"])) >= abs(float(source["mean"]))):
        return "semantic"
    if src_ok and (not sem_ok or abs(float(source["mean"])) > abs(float(sem["mean"]))):
        return "source_total"
    return "mixed_or_small"


def choose_architecture_leg(summary_by_metric: dict[str, Any], census_ok: bool) -> dict[str, Any]:
    base_note = "research already launched RoBERTa view+clean at 2.64x as the independent amplified total fixed-budget transfer test. research fixed the pre-result reading order: no fifth construction vertex should be added now; a strong carrier leg should instead be replicated at MAX dose with an independent seed/basin."
    if not census_ok:
        return {
            "recommendation": "not_ready",
            "launch_arms": [],
            "replication_target": None,
            "active_independent_roberta_total_arms": ["view", "clean"],
            "reason": "complete three-dose view/repeat census is not present. " + base_note,
        }
    votes = {m: component_vote(summary_by_metric[m]["dose2p64_growth_components"]) for m in PRIMARY}
    shapes = {m: summary_by_metric[m]["semantic_shape"]["shape"] for m in PRIMARY}
    # Cheap6 and cheap5 must agree.  The secondary relational/state sum is shown but cannot decide alone.
    coherent_semantic_shapes = {"monotone_or_saturating_increase", "turnover_after_midpoint"}
    if votes["cheap6_no_GlobalPIQA"] == votes["cheap5_no_GlobalPIQA_Reading"] == "semantic":
        if all(shapes[m] in coherent_semantic_shapes for m in PRIMARY):
            return {
                "recommendation": "replicate_max_view_repeat_second_seed",
                "launch_arms": [],
                "replication_target": {"arms": ["max_view", "max_repeat"], "reason": "semantic V-R is the carrier leg"},
                "active_independent_roberta_total_arms": ["view", "clean"],
                "possible_later_roberta_decomposition_after_viewclean": ["repeat"],
                "reason": "cheap6 and cheap5 both show positive, sign-consistent semantic V-R dose growth at 2.64x and the midpoint gives a coherent rising/saturating or post-midpoint-turnover shape. The next useful expensive action would be a second-seed MAX view/repeat replication, not another construction variant. " + base_note,
            }
    if votes["cheap6_no_GlobalPIQA"] == votes["cheap5_no_GlobalPIQA_Reading"] == "source_total":
        return {
            "recommendation": "replicate_max_repeat_clean_second_seed_if_total_source_leg_remains_primary",
            "launch_arms": [],
            "replication_target": {"arms": ["max_repeat", "max_clean"], "reason": "source/repeat clean-free growth is the carrier leg"},
            "active_independent_roberta_total_arms": ["view", "clean"],
            "reason": "cheap6 and cheap5 both place growth mainly in source/total clean-free movement rather than semantic V-R; the active RoBERTa view+clean pair already tests total cross-architecture transfer, and the next useful expensive action would be second-seed MAX repeat/clean replication if the total-source reading survives full review.",
        }
    return {
        "recommendation": "no_new_expensive_arm_until_family_item_reading",
        "launch_arms": [],
        "replication_target": None,
        "active_independent_roberta_total_arms": ["view", "clean"],
        "reason": f"cheap6/cheap5 component votes or dose shapes are mixed: votes={votes}, semantic_shapes={shapes}. Do not add a RoBERTa repeat arm or any other construction from a small, flat, or nonmonotone aggregate; read family and item movement first. " + base_note,
    }


def compact_population_notes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    if MID_SELECT_META.exists():
        d = read_json(MID_SELECT_META)
        out["dose1p82"] = {"path": str(MID_SELECT_META), "status": d.get("status"), "dose": d.get("dose"), "increment_origin_pair_words": d.get("increment_origin_pair_words"), "full_vs_old1x_standardized_shifts": d.get("full_vs_old1x_standardized_shifts") or d.get("standardized_shifts_full_vs_old1x")}
    if MAX_DIST_META.exists():
        d = read_json(MAX_DIST_META)
        out["dose2p64"] = {"path": str(MAX_DIST_META), "status": d.get("status"), "selected_pair_count": d.get("selected_pair_count"), "selected_pair_words": d.get("selected_pair_words"), "dose_multiple": d.get("dose_multiple"), "old_prefix_preserved": d.get("old_prefix_preserved"), "feature_shift_summary": d.get("complete_max_feature_shift_vs_old_block") or d.get("complete_selected_vs_old_standardized_shift")}
    if GEOMETRY_RECORD.exists():
        d = read_json(GEOMETRY_RECORD)
        out["training_geometry"] = {"path": str(GEOMETRY_RECORD), "status": d.get("status"), "max_updates": d.get("max_updates"), "extra_updates_after_2529": d.get("extra_updates_after_2529"), "summary": d.get("summary") or d.get("rows_updates")}
    if PRE_RESULT_READOUT_NOTE.exists():
        out["pre_result_reading_order"] = {"path": str(PRE_RESULT_READOUT_NOTE), "role": "frozen before new ladder scores were read; primary dose statements use ladder means, clean-free view growth, and cheap6/cheap5 agreement"}
    if POOL_TOKEN_ACCOUNTING.exists():
        d = read_json(POOL_TOKEN_ACCOUNTING)
        out["pool_token_accounting"] = {"path": str(POOL_TOKEN_ACCOUNTING), "relative_shift_vs_max_view": d.get("relative_shift_vs_max_view"), "condition": "word-matched dose arms are not exactly token-matched; WWM group totals remain very close"}
    if BREADTH_GEOMETRY.exists():
        d = read_json(BREADTH_GEOMETRY)
        out["breadth_conditions"] = {"path": str(BREADTH_GEOMETRY), "carried_conditions": d.get("carried_conditions")}
    return out


def build_summary(legs: list[dict[str, Any]], growth: list[dict[str, Any]], spread: dict[tuple[str, str], float], census_ok: bool) -> dict[str, Any]:
    by_metric: dict[str, Any] = {}
    for m in [*PRIMARY, *SECONDARY, "BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]:
        by_metric[m] = {
            "semantic_shape": shape_for_metric(legs, growth, m),
            "dose1p82_growth_components": metric_component(growth, spread, m, "dose1p82"),
            "dose2p64_growth_components": metric_component(growth, spread, m, "dose2p64"),
        }
    arch = choose_architecture_leg(by_metric, census_ok)
    arch["secondary_EWoK_plus_Entity_readout"] = by_metric["EWoK_plus_Entity_sum"]
    return {"metrics": by_metric, "architecture_leg_readout": arch}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dir", default=str(MAX_DIR_DEFAULT))
    ap.add_argument("--mid-dir", default=str(MID_DIR_DEFAULT))
    ap.add_argument("--seed-spread", default=str(SEED_SPREAD_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    max_dir = pathlib.Path(args.max_dir); mid_dir = pathlib.Path(args.mid_dir); spread_path = pathlib.Path(args.seed_spread); out_dir = pathlib.Path(args.out_dir)
    if not max_dir.is_absolute(): max_dir = ROOT / max_dir
    if not mid_dir.is_absolute(): mid_dir = ROOT / mid_dir
    if not spread_path.is_absolute(): spread_path = ROOT / spread_path
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    max_rows_path = max_dir / "dose_ladder_stable_rows.csv"
    mid_rows_path = mid_dir / "intermediate_dose_ladder_rows.csv"
    max_summary = load_summary(max_dir / "dose_ladder_stable_summary.json")
    mid_summary = load_summary(mid_dir / "intermediate_dose_ladder_summary.json")
    plan = {"status": "DOSE_CURVE_INTEGRATOR_PLAN", "created_utc": now(), "inputs": {"max_rows": str(max_rows_path), "mid_rows": str(mid_rows_path), "seed_spread": str(spread_path)}, "exists": {"max_rows": max_rows_path.exists(), "mid_rows": mid_rows_path.exists(), "seed_spread": spread_path.exists()}, "summary_inputs": {"max": {k: v for k, v in max_summary.items() if k != "raw"}, "midpoint": {k: v for k, v in mid_summary.items() if k != "raw"}}, "out_dir": str(out_dir), "no_training_gpu_eval_superglue_aoa_upload_or_leaderboard": True}
    (out_dir / "dose_curve_integrator_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    if not (max_rows_path.exists() and mid_rows_path.exists() and spread_path.exists() and max_summary.get("ok") and mid_summary.get("ok")):
        raise FileNotFoundError("complete MAX rows, midpoint rows, seed spread, and zero-problem summaries are required before integration")

    rows = dedupe(normalize_rows(max_rows_path, "max_reuse") + normalize_rows(mid_rows_path, "midpoint"))
    census_ok, census_problems, census = validate_census(rows)
    spread = load_seed_spread(spread_path)
    legs = build_leg_rows(rows, spread)
    growth = build_growth_rows(rows, spread)
    max_sparse_resid = max([abs(float(r["sparse_decomposition_residual"])) for r in legs if finite(r.get("sparse_decomposition_residual"))], default=None)
    max_growth_resid = max([abs(float(r["cleanfree_growth_residual"])) for r in growth if finite(r.get("cleanfree_growth_residual"))], default=None)
    summary = build_summary(legs, growth, spread, census_ok)
    if census_ok:
        status = "DOSE_CURVE_INTEGRATED"
    else:
        status = "DOSE_CURVE_NOT_INTEGRATED_INCOMPLETE"
        summary["architecture_leg_readout"] = {"recommendation": "not_ready", "launch_arms": [], "reason": "three-dose census incomplete; see census_problems"}

    write_csv(out_dir / "dose_curve_rows.csv", rows)
    write_csv(out_dir / "dose_curve_sparse_clean_legs.csv", legs)
    write_csv(out_dir / "dose_curve_cleanfree_growth_vs_1x.csv", growth)
    payload = {
        "status": status,
        "created_utc": now(),
        "input_status": {"max_summary": {k: v for k, v in max_summary.items() if k != "raw"}, "midpoint_summary": {k: v for k, v in mid_summary.items() if k != "raw"}, "seed_spread_metric_checkpoint_values": len(spread)},
        "census": census,
        "census_problems": census_problems,
        "row_counts": {"normalized_rows": len(rows), "sparse_clean_leg_rows": len(legs), "cleanfree_growth_rows": len(growth)},
        "max_abs_sparse_decomposition_residual": max_sparse_resid,
        "max_abs_cleanfree_growth_residual": max_growth_resid,
        "summary": summary,
        "population_and_training_geometry_notes": compact_population_notes(),
        "files": {"curve_rows_csv": str(out_dir / "dose_curve_rows.csv"), "sparse_clean_legs_csv": str(out_dir / "dose_curve_sparse_clean_legs.csv"), "cleanfree_growth_csv": str(out_dir / "dose_curve_cleanfree_growth_vs_1x.csv"), "pre_result_reading_order": str(PRE_RESULT_READOUT_NOTE)},
        "pre_result_reading_order": {"primary_composites": PRIMARY, "dose_statement": "ladder-mean profile first; clean-free V_d minus V_1 growth over all ten checkpoints; cheap6 and cheap5 must agree", "secondary": "EWoK+Entity and individual family signs explain movement but cannot carry a result alone", "replication_rule": "after identifying the carrier leg, spend the next expensive work on second-seed MAX replication of that exact leg rather than adding another construction vertex"},
        "scope_note": "Selected stable families only: BLiMP, Supplement, EWoK, Entity, COMPS, Reading. GlobalPIQA is not used; SuperGLUE, AoA, upload, and leaderboard submission are not touched.",
    }
    out_json = out_dir / "dose_curve_integrated_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research compact-dose curve integrator", "", payload["scope_note"], "", f"Status: `{status}`", f"Rows: `{payload['row_counts']}`", f"Sparse residual: `{max_sparse_resid}`; clean-free growth residual: `{max_growth_resid}`", "", "## Frozen research reading order", "", f"Primary composites: `{PRIMARY}`", "Dose statement: ladder means first, then clean-free `V_d - V_1` growth over all ten checkpoints; cheap6 and cheap5 must agree.", "EWoK+Entity and individual families explain movement but cannot carry a result alone.", f"Pre-result note: `{PRE_RESULT_READOUT_NOTE}`", "", "## Continuation readout", "", "research RoBERTa view+clean is the independent total-effect transfer test already active. This DeBERTa curve does not add another construction vertex; it identifies whether any carrier leg merits a second-seed MAX replication.", "", f"Recommendation: `{summary['architecture_leg_readout']['recommendation']}`", f"Replication target: `{summary['architecture_leg_readout'].get('replication_target')}`", f"Additional launch arms: `{summary['architecture_leg_readout']['launch_arms']}`", f"Reason: {summary['architecture_leg_readout']['reason']}", "", "## Cheap metric component votes"]
    for m in PRIMARY:
        rec = summary["metrics"][m]
        lines.append(f"- {m}: shape `{rec['semantic_shape']['shape']}`; MAX semantic growth mean `{rec['dose2p64_growth_components']['semantic']['mean']}`; MAX source growth mean `{rec['dose2p64_growth_components']['source']['mean']}`; seed-scale mean `{rec['dose2p64_growth_components']['seed_scale']['mean']}`")
    lines += ["", "EWoK+Entity is secondary and cannot select any expensive continuation by itself.", "", f"JSON: `{out_json}`", f"Clean-free growth CSV: `{out_dir / 'dose_curve_cleanfree_growth_vs_1x.csv'}`"]
    (out_dir / "dose_curve_integrated_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_json": str(out_json), "architecture_recommendation": summary["architecture_leg_readout"], "row_counts": payload["row_counts"], "census_ok": census_ok}, indent=2, ensure_ascii=False), flush=True)
    if not census_ok:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
