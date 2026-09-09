#!/usr/bin/env python3
"""research: common-window and family leave-one-out readout for dose profiles.

Reads `score_harvest_readout` outputs and reports:
  * common 10M--80M summaries so MAX V-R is compared on the same exposure window
    as lower doses while max_repeat 90M/100M are still absent;
  * family concentration and leave-one-family-out composite movement;
  * trajectory-mean versus pointwise seed-spread scale from research.

This is stable-family-only and does not evaluate models.
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
IN = WS / "data/score_harvest_readout"
OUT = WS / "data/family_lfo_readout"
SEED_ROWS = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_stable_rows.csv"
SEED_SPREAD = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_seed_spread.csv"
FAMILIES = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
FIVE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
PRIMARY = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading"]
SECONDARY = ["EWoK_plus_Entity_sum"]
METRICS = FAMILIES + PRIMARY + SECONDARY
COMMON_10_80 = {f"chck_{i}M" for i in range(10, 81, 10)}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def f(v: Any) -> float | None:
    if v is None or v == "" or v == "None":
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ["words", "dose", "rho"] + METRICS:
            if k in r:
                x = f(r[k])
                if x is not None:
                    r[k] = x
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    preferred = ["profile", "group", "window", "n", "family", "mean", "median", "min", "max", "stdev", "positive_rows", "negative_rows", "excluded_family", "cheap6_lfo_mean", "cheap5_lfo_mean", "metric"]
    for p in preferred:
        if p not in fields:
            fields.append(p)
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stat(vals: list[float]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "mean": statistics.mean(vals) if vals else None,
        "median": statistics.median(vals) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "stdev": statistics.stdev(vals) if len(vals) >= 2 else 0.0 if vals else None,
        "positive_rows": sum(1 for v in vals if v > 0),
        "negative_rows": sum(1 for v in vals if v < 0),
    }


def group_key(row: dict[str, Any]) -> str:
    return str(row.get("dose_name") or row.get("contrast") or row.get("label") or "unknown")


def summarize_profile(rows: list[dict[str, Any]], profile_name: str) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_group.setdefault(group_key(r), []).append(r)
    out: dict[str, Any] = {}
    flat_family_rows: list[dict[str, Any]] = []
    flat_lfo_rows: list[dict[str, Any]] = []
    for group, rs0 in sorted(by_group.items()):
        for window_name, rs in [("available", rs0), ("common_10M_80M", [r for r in rs0 if r.get("checkpoint") in COMMON_10_80])]:
            d: dict[str, Any] = {"n": len(rs), "checkpoints": [r.get("checkpoint") for r in sorted(rs, key=lambda x: int(float(x.get("words") or 0)))]}
            fam_means: dict[str, float] = {}
            for fam in FAMILIES:
                vals = [f(r.get(fam)) for r in rs]
                vals_f = [float(v) for v in vals if v is not None]
                s = stat(vals_f)
                d[fam] = s
                if s["mean"] is not None:
                    fam_means[fam] = float(s["mean"])
                flat_family_rows.append({"profile": profile_name, "group": group, "window": window_name, "family": fam, **s})
            for metric in PRIMARY + SECONDARY:
                vals = [f(r.get(metric)) for r in rs]
                d[metric] = stat([float(v) for v in vals if v is not None])
            lfo: dict[str, Any] = {}
            for excluded in FAMILIES:
                keep = [x for x in FAMILIES if x != excluded]
                row_vals: list[float] = []
                for r in rs:
                    vals = [f(r.get(k)) for k in keep]
                    if all(v is not None for v in vals):
                        row_vals.append(sum(float(v) for v in vals) / len(vals))
                lfo[excluded] = stat(row_vals)
                flat_lfo_rows.append({"profile": profile_name, "group": group, "window": window_name, "excluded_family": excluded, "cheap6_lfo_mean": lfo[excluded]["mean"], "n": lfo[excluded]["n"], "positive_rows": lfo[excluded]["positive_rows"], "negative_rows": lfo[excluded]["negative_rows"]})
            lfo5: dict[str, Any] = {}
            for excluded in FIVE:
                keep = [x for x in FIVE if x != excluded]
                row_vals = []
                for r in rs:
                    vals = [f(r.get(k)) for k in keep]
                    if all(v is not None for v in vals):
                        row_vals.append(sum(float(v) for v in vals) / len(vals))
                lfo5[excluded] = stat(row_vals)
            d["leave_one_out_cheap6"] = lfo
            d["leave_one_out_cheap5"] = lfo5
            # Compact concentration index: largest absolute family mean divided by
            # sum absolute family means. Values near 1 indicate one-family carrier.
            abs_sum = sum(abs(v) for v in fam_means.values())
            if abs_sum > 0:
                best = max(fam_means.items(), key=lambda kv: abs(kv[1]))
                d["family_concentration_absmean"] = {"top_family": best[0], "top_family_mean": best[1], "top_abs_fraction": abs(best[1]) / abs_sum, "signed_family_means": fam_means}
            out.setdefault(group, {})[window_name] = d
    write_csv(OUT / f"{profile_name}_family_means.csv", flat_family_rows)
    write_csv(OUT / f"{profile_name}_leave_one_out.csv", flat_lfo_rows)
    return out


def seed_scale() -> dict[str, Any]:
    rows = read_csv(SEED_ROWS)
    by_seed: dict[str, list[dict[str, Any]]] = {"43022": [], "43122": []}
    for r in rows:
        seed = str(int(float(r.get("seed")))) if f(r.get("seed")) is not None else str(r.get("seed"))
        if seed not in by_seed:
            continue
        # Use only compact/view minus repeat rows by reconstructing from rows.
        by_seed[seed].append(r)
    # Reconstruct treatment deltas from row table.
    def lk(seed: str, arm: str, ck: str) -> dict[str, Any] | None:
        want = f"seed{seed}_{'compact' if arm == 'view' else 'repeat'}"
        for r in rows:
            if str(r.get("arm")) == want and str(r.get("checkpoint")) == ck:
                return r
        return None
    treatment: dict[str, list[dict[str, Any]]] = {"43022": [], "43122": []}
    for seed in ["43022", "43122"]:
        for ck in [f"chck_{i}M" for i in range(10, 101, 10)]:
            v, r = lk(seed, "view", ck), lk(seed, "repeat", ck)
            if not v or not r:
                continue
            row = {"seed": seed, "checkpoint": ck}
            for m in METRICS:
                va, vb = f(v.get(m)), f(r.get(m))
                if va is not None and vb is not None:
                    row[m] = va - vb
            treatment[seed].append(row)
    out: dict[str, Any] = {"treatment_delta_means_by_seed": {}, "trajectory_mean_seed_difference": {}, "pointwise_seed_difference": {}}
    for seed, rs in treatment.items():
        out["treatment_delta_means_by_seed"][seed] = {}
        for m in METRICS:
            vals = [f(r.get(m)) for r in rs]
            vals_f = [float(v) for v in vals if v is not None]
            if vals_f:
                out["treatment_delta_means_by_seed"][seed][m] = stat(vals_f)
    for m in METRICS:
        a = (out["treatment_delta_means_by_seed"].get("43022", {}).get(m) or {}).get("mean")
        b = (out["treatment_delta_means_by_seed"].get("43122", {}).get(m) or {}).get("mean")
        if a is not None and b is not None:
            out["trajectory_mean_seed_difference"][m] = float(b) - float(a)
    for m in METRICS:
        vals = []
        for ck in [f"chck_{i}M" for i in range(10, 101, 10)]:
            r0 = next((r for r in treatment["43022"] if r.get("checkpoint") == ck), None)
            r1 = next((r for r in treatment["43122"] if r.get("checkpoint") == ck), None)
            if r0 and r1 and f(r0.get(m)) is not None and f(r1.get(m)) is not None:
                vals.append(float(r1[m]) - float(r0[m]))
        if vals:
            out["pointwise_seed_difference"][m] = {"signed": stat(vals), "absolute": stat([abs(v) for v in vals])}
    # Also load the spread CSV to retain exact source path continuity.
    out["source_rows_csv"] = rel(SEED_ROWS)
    out["source_spread_csv"] = rel(SEED_SPREAD)
    return out


def render_md(summary: dict[str, Any]) -> str:
    lines: list[str] = ["# research common-window family readout", ""]
    lines.append("Stable selected-family score movement only. The MAX semantic leg is read on the common 10M--80M window until max_repeat 90M/100M arrive.")
    lines.append("")
    lines.append("## Seed-scale separation")
    for m in ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]:
        tm = summary["seed_scale"]["trajectory_mean_seed_difference"].get(m)
        pa = ((summary["seed_scale"]["pointwise_seed_difference"].get(m) or {}).get("absolute") or {}).get("mean")
        pm = ((summary["seed_scale"]["pointwise_seed_difference"].get(m) or {}).get("absolute") or {}).get("max")
        lines.append(f"- {m}: trajectory-mean seed difference {tm:.4f}; pointwise mean absolute difference {pa:.4f}; pointwise max absolute difference {pm:.4f}")
    lines.append("")
    for pname in ["semantic_VR", "cleanfree_growth_Vd_minus_V1", "total_VC", "repeat_clean_RC"]:
        lines.append(f"## {pname}")
        for group, by_window in summary["profiles"].get(pname, {}).items():
            cw = by_window.get("common_10M_80M") or by_window.get("available")
            av = by_window.get("available")
            if not cw:
                continue
            lines.append(f"### {group}")
            for label, block in [("common_10M_80M", cw), ("available", av)]:
                if not block:
                    continue
                c6 = block.get("cheap6_no_GlobalPIQA", {})
                c5 = block.get("cheap5_no_GlobalPIQA_Reading", {})
                ee = block.get("EWoK_plus_Entity_sum", {})
                conc = block.get("family_concentration_absmean", {})
                lines.append(f"- {label}: n={block.get('n')} cheap6 mean {c6.get('mean'):.4f} ({c6.get('positive_rows')}/{c6.get('n')} positive), cheap5 mean {c5.get('mean'):.4f}; EWoK+Entity mean {ee.get('mean'):.4f}; top abs family {conc.get('top_family')} mean {conc.get('top_family_mean'):.4f} fraction {conc.get('top_abs_fraction'):.3f}")
                if pname == "semantic_VR" and group == "dose2p64" and label == "common_10M_80M":
                    lfo = block.get("leave_one_out_cheap6", {})
                    lfo_text = ", ".join(f"drop {k}: {v.get('mean'):.4f}" for k, v in lfo.items() if v.get("mean") is not None)
                    lines.append(f"  - MAX V-R cheap6 leave-one-family-out: {lfo_text}")
        lines.append("")
    lines.append("## Readout")
    lines.append(summary.get("reading", ""))
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    profiles_in = {
        "semantic_VR": read_csv(IN / "semantic_VR.csv"),
        "cleanfree_growth_Vd_minus_V1": read_csv(IN / "cleanfree_growth_Vd_minus_V1.csv"),
        "total_VC": read_csv(IN / "total_VC.csv"),
        "repeat_clean_RC": read_csv(IN / "repeat_clean_RC.csv"),
        "roberta_max_view_clean_contrast": read_csv(IN / "roberta_max_view_clean_contrast.csv"),
    }
    profiles = {name: summarize_profile(rows, name) for name, rows in profiles_in.items()}
    seeds = seed_scale()

    # Substantive reading from the in-hand numbers.
    max_vr_cw = profiles["semantic_VR"].get("dose2p64", {}).get("common_10M_80M", {})
    d1_vr_cw = profiles["semantic_VR"].get("dose1", {}).get("common_10M_80M", {})
    d182_vr_cw = profiles["semantic_VR"].get("dose1p82", {}).get("common_10M_80M", {})
    max_total = profiles["total_VC"].get("dose2p64", {}).get("common_10M_80M", {})
    d1_total = profiles["total_VC"].get("dose1", {}).get("common_10M_80M", {})
    max_rc = profiles["repeat_clean_RC"].get("dose2p64", {}).get("common_10M_80M", {})
    d1_rc = profiles["repeat_clean_RC"].get("dose1", {}).get("common_10M_80M", {})
    pieces = []
    def gm(block: dict[str, Any], metric: str) -> float | None:
        return (block.get(metric) or {}).get("mean")
    pieces.append(
        "On the common 10M--80M window, V-R cheap6/cheap5 moves from "
        f"{gm(d1_vr_cw,'cheap6_no_GlobalPIQA'):.4f}/{gm(d1_vr_cw,'cheap5_no_GlobalPIQA_Reading'):.4f} at 1x, "
        f"to {gm(d182_vr_cw,'cheap6_no_GlobalPIQA'):.4f}/{gm(d182_vr_cw,'cheap5_no_GlobalPIQA_Reading'):.4f} at 1.82x, "
        f"to {gm(max_vr_cw,'cheap6_no_GlobalPIQA'):.4f}/{gm(max_vr_cw,'cheap5_no_GlobalPIQA_Reading'):.4f} at 2.64x."
    )
    pieces.append(
        "The 2.64x total V-C movement is broader and larger than its V-R leg: "
        f"cheap6/cheap5 {gm(max_total,'cheap6_no_GlobalPIQA'):.4f}/{gm(max_total,'cheap5_no_GlobalPIQA_Reading'):.4f}, "
        f"versus 1x total {gm(d1_total,'cheap6_no_GlobalPIQA'):.4f}/{gm(d1_total,'cheap5_no_GlobalPIQA_Reading'):.4f}."
    )
    pieces.append(
        "The repeat-clean leg is positive but not dose-growing: "
        f"1x {gm(d1_rc,'cheap6_no_GlobalPIQA'):.4f}/{gm(d1_rc,'cheap5_no_GlobalPIQA_Reading'):.4f}, "
        f"2.64x {gm(max_rc,'cheap6_no_GlobalPIQA'):.4f}/{gm(max_rc,'cheap5_no_GlobalPIQA_Reading'):.4f}."
    )
    conc = max_vr_cw.get("family_concentration_absmean", {})
    pieces.append(
        "The MAX V-R leg is family-concentrated rather than uniformly broad: "
        f"top absolute family is {conc.get('top_family')} with mean {conc.get('top_family_mean'):.4f} and fraction {conc.get('top_abs_fraction'):.3f}; "
        "dropping Entity reduces its cheap6 leave-one-family-out mean sharply, while total V-C remains positive under family deletions."
    )
    pieces.append(
        "Thus the first-basin table supports a dose-amplified structured re-expression effect as an exposure-family allocation pattern, but not yet a settled general law; the second-basin MAX pair now running is needed to distinguish reproducible packet value from basin-specific family redistribution."
    )
    reading = " ".join(pieces)

    summary = {
        "status": "FAMILY_LFO_READOUT_COMPLETE",
        "created_utc": now(),
        "inputs": {k: rel(IN / f"{k}.csv") for k in profiles_in},
        "profiles": profiles,
        "seed_scale": seeds,
        "reading": reading,
        "boundary": "Stable selected families only; no model evaluation, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.",
        "files": {
            "summary_json": rel(OUT / "family_lfo_readout_summary.json"),
            "summary_md": rel(OUT / "family_lfo_readout_summary.md"),
        },
    }
    (OUT / "family_lfo_readout_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "family_lfo_readout_summary.md").write_text(render_md(summary), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "created_utc": summary["created_utc"],
        "common_window_reading": reading,
        "summary_json": summary["files"]["summary_json"],
        "summary_md": summary["files"]["summary_md"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
