#!/usr/bin/env python3
"""research: signed-transition window background for DeBERTa common grids.

This is a CPU/file-only companion to the research signed-transition analyzer.  It
addresses a specific measurement hazard: choosing each trajectory's own best
checkpoint by cheap7 automatically creates a local rise/fall, so similarity of
own-peak windows can be partly selection-induced.

Given a fixed reference window, normally the reference scale1.75 seed43022
`chck_82M -> chck_84M -> chck_86M`, this script compares every eligible late
three-checkpoint window in one or more target trajectories to that fixed
reference signature.  It separately marks the same-coordinate window
(82->84->86) and the target's own-peak window, and ranks those rows against the
target's internal all-window background.

No model inference, training, upload, or leaderboard submission is performed.
It reads already-written selected-grid prediction payloads only.
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
import pathlib
import statistics
import sys
import time
from typing import Any

CLASS_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]
KEY_SIM_LEVELS = {"column_official_subtask_mean", "column_subtask"}
KEY_SIM_WINDOWS = {"rise", "fall", "rise+fall_pattern"}
SIM_METRICS = ["pearson_net_pct", "weighted_pearson_net_pct", "cosine_net_pct", "sign_agreement_ref_movers", "same_signed_pattern_fraction"]


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
PATH = ROOT / "experiments/archive/frontier_consolidation/scripts/signed_transition_signature_analyzer.py"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_step197():
    spec = importlib.util.spec_from_file_location("signed_transition_signature_analyzer", PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


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
                keys.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rel(path: pathlib.Path) -> str:
    p = path.resolve() if path.is_absolute() else path
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(path)


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def pct(x: float) -> float:
    return 100.0 * x


def endpoint_sort_key(ep: str, s196) -> int:
    return int(s196.endpoint_int(ep))


def selected_metric(row: dict[str, Any], metric: str) -> float | None:
    v = row.get(metric)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def parse_labeled_dir(spec: str) -> tuple[str, pathlib.Path]:
    if "=" not in spec:
        raise SystemExit(f"Use LABEL=SELECTED_DIR, got {spec!r}")
    label, d = spec.split("=", 1)
    path = pathlib.Path(d)
    if not path.is_absolute():
        path = ROOT / path
    return label, path


def complete_endpoints(label: str, selected_dir: pathlib.Path, s196) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    traj_rows = s196.read_trajectory_rows(selected_dir)
    complete: list[str] = []
    incomplete: list[dict[str, Any]] = []
    for r in traj_rows:
        ep = r.get("endpoint")
        if not ep:
            continue
        path = s196.per_target_path(selected_dir, label, ep)
        if not path.exists():
            incomplete.append({"endpoint": ep, "reason": "missing_payload", "path": safe_rel(path)})
            continue
        try:
            payload = read_json(path)
            ok, bad = s196.payload_is_complete(payload)
            if ok:
                complete.append(str(ep))
            else:
                incomplete.append({"endpoint": ep, "reason": "incomplete_payload", "bad": bad, "path": safe_rel(path)})
        except Exception as e:
            incomplete.append({"endpoint": ep, "reason": f"exception:{type(e).__name__}", "error": str(e)[:300], "path": safe_rel(path)})
    complete = sorted(set(complete), key=lambda e: endpoint_sort_key(e, s196))
    return complete, traj_rows, incomplete


def choose_best_endpoint(traj_rows: list[dict[str, Any]], complete_eps: list[str], metric: str) -> str:
    usable = set(complete_eps)
    candidates = [r for r in traj_rows if r.get("endpoint") in usable and selected_metric(r, metric) is not None]
    if not candidates:
        raise RuntimeError(f"No selected rows with metric {metric}")
    return str(max(candidates, key=lambda r: float(r[metric]))["endpoint"])


def consecutive_windows(complete_eps: list[str], s196, min_mid_m: int | None = None, max_mid_m: int | None = None) -> list[tuple[str, str, str]]:
    eps = sorted(complete_eps, key=lambda e: endpoint_sort_key(e, s196))
    windows: list[tuple[str, str, str]] = []
    for i in range(1, len(eps) - 1):
        a, b, c = eps[i - 1], eps[i], eps[i + 1]
        # s196.endpoint_int returns the integer M in `chck_84M`, not raw words.
        if endpoint_sort_key(b, s196) - endpoint_sort_key(a, s196) != 2:
            continue
        if endpoint_sort_key(c, s196) - endpoint_sort_key(b, s196) != 2:
            continue
        bm = endpoint_sort_key(b, s196)
        if min_mid_m is not None and bm < min_mid_m:
            continue
        if max_mid_m is not None and bm > max_mid_m:
            continue
        windows.append((a, b, c))
    return windows


def load_window_signature(label: str, selected_dir: pathlib.Path, window: tuple[str, str, str], out_dir: pathlib.Path, s196, late_pair_results, transition_results) -> dict[str, Any]:
    prev_ep, mid_ep, next_ep = window
    payloads = {ep: transition_results.load_payload_for_endpoint(selected_dir, label, ep, s196, late_pair_results) for ep in window}
    items_by_ep = {ep: late_pair_results.endpoint_items(payloads[ep]) for ep in window}
    merged = late_pair_results.merge_endpoint_items(items_by_ep)
    item_rows = transition_results.make_item_transition_rows(label, merged, prev_ep, mid_ep, next_ep, s196)
    group_levels = {
        "column": ["column"],
        "column_subtask": ["column", "subtask"],
        "column_subtask_subgroup": ["column", "subtask", "subgroup"],
        "column_subtask_subgroup_fine": ["column", "subtask", "subgroup", "fine_group"],
    }
    group_summaries: dict[str, list[dict[str, Any]]] = {}
    for name, fields in group_levels.items():
        group_summaries[name] = transition_results.group_summary_rows(item_rows, fields, s196)
    group_summaries["column_official_subtask_mean"] = transition_results.official_like_column_transition_from_subtasks(group_summaries["column_subtask"])
    by_ep = {r.get("endpoint"): r for r in s196.read_trajectory_rows(selected_dir)}
    window_scores = {}
    for ep in window:
        row = by_ep.get(ep, {}) or {}
        window_scores[ep] = {k: row.get(k) for k in ["cheap7", "BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "relation_state_mean"] if k in row}
    summary = {
        "label": label,
        "selected_dir": safe_rel(selected_dir),
        "prev_endpoint": prev_ep,
        "mid_endpoint": mid_ep,
        "next_endpoint": next_ep,
        "window_string": f"{prev_ep}->{mid_ep}->{next_ep}",
        "n_classification_items": len(item_rows),
        "window_scores_selected_rows": window_scores,
    }
    return {"summary": summary, "item_rows": item_rows, "group_summaries": group_summaries}


def item_pattern_metrics(ref_rows: list[dict[str, Any]], oth_rows: list[dict[str, Any]], transition_results) -> dict[str, Any]:
    rb = {r["item_key"]: r for r in ref_rows}
    ob = {r["item_key"]: r for r in oth_rows}
    common = sorted(set(rb) & set(ob))
    out: dict[str, Any] = {"n_common_items": len(common)}
    xs_pat: list[float] = []
    ys_pat: list[float] = []
    ref_changed = 0
    same_on_ref_changed = 0
    opposite_on_ref_changed = 0
    other_zero_on_ref_changed = 0
    for key in common:
        for w in ["rise", "fall"]:
            x = int(rb[key].get(f"delta_{w}", 0))
            y = int(ob[key].get(f"delta_{w}", 0))
            xs_pat.append(float(x)); ys_pat.append(float(y))
            if x != 0:
                ref_changed += 1
                if y == x:
                    same_on_ref_changed += 1
                elif y == -x:
                    opposite_on_ref_changed += 1
                elif y == 0:
                    other_zero_on_ref_changed += 1
    out.update({
        "signed_transition_pattern_cosine": transition_results.cosine(xs_pat, ys_pat),
        "signed_transition_pattern_pearson": transition_results.pearson(xs_pat, ys_pat),
        "ref_changed_transitions": ref_changed,
        "same_signed_transition_on_ref_changed_fraction": same_on_ref_changed / ref_changed if ref_changed else None,
        "opposite_on_ref_changed_fraction": opposite_on_ref_changed / ref_changed if ref_changed else None,
        "other_zero_on_ref_changed_fraction": other_zero_on_ref_changed / ref_changed if ref_changed else None,
    })
    return out


def compare_signature(ref_sig: dict[str, Any], target_sig: dict[str, Any], min_abs_ref_net_pct: float, transition_results) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    group_rows, top_rows = transition_results.compare_group_vectors(ref_sig["group_summaries"], target_sig["group_summaries"], min_abs_ref_net_pct)
    item_rows: list[dict[str, Any]] = []
    for window in ["rise", "fall"]:
        item_rows.extend(transition_results.summarize_item_transition_pair(ref_sig["item_rows"], target_sig["item_rows"], window))
    item_pattern = item_pattern_metrics(ref_sig["item_rows"], target_sig["item_rows"], transition_results)
    return group_rows, item_rows, item_pattern


def sim_value_for_rank(row: dict[str, Any], metric: str) -> float | None:
    v = row.get(metric)
    if v is None:
        return None
    if not isinstance(v, (int, float)) or math.isnan(float(v)):
        return None
    return float(v)


def quantiles(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {"n": len(xs), "min": xs[0], "q10": q(0.10), "q25": q(0.25), "median": q(0.50), "q75": q(0.75), "q90": q(0.90), "max": xs[-1]}


def add_rank_summaries(all_group_rows: list[dict[str, Any]], all_item_pattern_rows: list[dict[str, Any]], target_summaries: dict[str, Any]) -> None:
    labels = sorted({r["target_label"] for r in all_group_rows})
    for label in labels:
        label_rows = [r for r in all_group_rows if r["target_label"] == label]
        label_patterns = [r for r in all_item_pattern_rows if r["target_label"] == label]
        target_summaries.setdefault(label, {})["rank_summaries"] = []
        target_summaries.setdefault(label, {})["item_pattern_rank_summaries"] = []
        for level in sorted(KEY_SIM_LEVELS):
            for window in sorted(KEY_SIM_WINDOWS):
                rows = [r for r in label_rows if r.get("group_level") == level and r.get("window") == window]
                for metric in SIM_METRICS:
                    vals = [sim_value_for_rank(r, metric) for r in rows]
                    vals = [v for v in vals if v is not None]
                    if not vals:
                        continue
                    rec_base = {"group_level": level, "window": window, "metric": metric, "background": quantiles(vals)}
                    for role in ["own_peak", "fixed_82_84_86"]:
                        # A target may peak exactly at the fixed 82->84->86 coordinate.
                        role_rows = [r for r in rows if r.get("target_window_role") == role or r.get("target_window_role") == "own_peak_and_fixed_82_84_86"]
                        if not role_rows:
                            continue
                        val = sim_value_for_rank(role_rows[0], metric)
                        if val is None:
                            continue
                        # Higher similarity is better for all metrics here.
                        rank_desc = 1 + sum(1 for x in vals if x > val)
                        percentile = sum(1 for x in vals if x <= val) / len(vals)
                        rec = dict(rec_base)
                        rec.update({"role": role, "value": val, "rank_descending": rank_desc, "percentile_higher_is_better": percentile})
                        target_summaries[label]["rank_summaries"].append(rec)
        for metric in ["signed_transition_pattern_cosine", "signed_transition_pattern_pearson", "same_signed_transition_on_ref_changed_fraction"]:
            vals = [sim_value_for_rank(r, metric) for r in label_patterns]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            rec_base = {"metric": metric, "background": quantiles(vals)}
            for role in ["own_peak", "fixed_82_84_86"]:
                role_rows = [r for r in label_patterns if r.get("target_window_role") == role or r.get("target_window_role") == "own_peak_and_fixed_82_84_86"]
                if not role_rows:
                    continue
                val = sim_value_for_rank(role_rows[0], metric)
                if val is None:
                    continue
                rank_desc = 1 + sum(1 for x in vals if x > val)
                percentile = sum(1 for x in vals if x <= val) / len(vals)
                rec = dict(rec_base)
                rec.update({"role": role, "value": val, "rank_descending": rank_desc, "percentile_higher_is_better": percentile})
                target_summaries[label]["item_pattern_rank_summaries"].append(rec)


def write_md(summary: dict[str, Any], path: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append("# research signed-transition window background\n\n")
    lines.append("CPU/file-only comparison of every eligible late three-checkpoint window against the fixed reference 82M→84M→86M signed-transition signature. This addresses the selection-induced rise/fall hazard in own-peak windows.\n\n")
    ref = summary["reference"]
    lines.append(f"Reference: `{ref['label']}` fixed window `{ref['window_string']}` from `{ref['selected_dir']}`.\n\n")
    lines.append("## Target summaries\n\n")
    lines.append("| target | complete endpoints | windows | own-peak window | fixed 82→84→86 present? |\n")
    lines.append("|---|---:|---:|---|---|\n")
    for label, ts in summary["targets"].items():
        lines.append(f"| {label} | {len(ts.get('complete_endpoints', []))} | {ts.get('n_windows')} | {ts.get('own_peak_window')} | {ts.get('fixed_window_present')} |\n")
    lines.append("\n## Key ranks against each target's own all-window background\n\n")
    lines.append("Higher similarity is better. Rank 1 means the row is the most similar window to the fixed reference signature within that target trajectory. The fixed row is the same exposure coordinate 82→84→86; own-peak is selected by the target trajectory's cheap7.\n\n")
    lines.append("| target | role | level | window | metric | value | rank | percentile | background median | background max |\n")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|\n")
    for label, ts in summary["targets"].items():
        # Only display the most interpretable pattern metrics to keep the note short.
        key_recs = []
        for r in ts.get("rank_summaries", []):
            if r.get("group_level") in ["column_official_subtask_mean", "column_subtask"] and r.get("window") == "rise+fall_pattern" and r.get("metric") in ["cosine_net_pct", "weighted_pearson_net_pct"]:
                key_recs.append(r)
        for r in ts.get("item_pattern_rank_summaries", []):
            if r.get("metric") in ["signed_transition_pattern_cosine", "same_signed_transition_on_ref_changed_fraction"]:
                rr = dict(r)
                rr["group_level"] = "item_pattern"
                rr["window"] = "rise+fall_pattern"
                key_recs.append(rr)
        for r in key_recs:
            bg = r.get("background", {})
            lines.append(f"| {label} | {r.get('role')} | {r.get('group_level')} | {r.get('window')} | {r.get('metric')} | {fmt(r.get('value'))} | {r.get('rank_descending')} | {fmt(r.get('percentile_higher_is_better'))} | {fmt(bg.get('median'))} | {fmt(bg.get('max'))} |\n")
    lines.append("\n## Best-matching windows by target\n\n")
    lines.append("For a future seed43122 readout, recurring late-phase structure should make the own-peak and/or fixed 82→84→86 window stand out against these internal backgrounds at official-like column and subtask levels, not merely show an automatic rise/fall from peak selection.\n\n")
    lines.append("| target | level | window | metric | best target window | best value | own-peak value | fixed value |\n")
    lines.append("|---|---|---|---|---|---:|---:|---:|\n")
    for label, ts in summary["targets"].items():
        for r in ts.get("best_window_summaries", []):
            lines.append(f"| {label} | {r.get('group_level')} | {r.get('window')} | {r.get('metric')} | {r.get('best_window_string')} | {fmt(r.get('best_value'))} | {fmt(r.get('own_peak_value'))} | {fmt(r.get('fixed_value'))} |\n")
    lines.append("\n## Files\n\n")
    for k, v in summary.get("files", {}).items():
        lines.append(f"- {k}: `{v.get('path')}` ({v.get('size')} bytes, sha256 {v.get('sha256')})\n")
    lines.append("\n## Reading rule\n\n")
    lines.append("When seed43122 arrives, run this same script with seed43122 as a target together with the research/195/196 tools. If its own-peak similarity is ordinary under the all-window background, or if the fixed 82→84→86 coordinate is not similar, the seed43022 chck84 late peak should be treated as a trajectory-specific endpoint asset. If the fixed and own-peak windows both stand out with matching columns/subtasks and item signed transitions, then the residual-capacity late phase is a more robust learning-dynamics object. After that readout, make the route decision rather than adding more analysis layers.\n\n")
    lines.append(f"JSON: `{summary['out_json']}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="LABEL=SELECTED_DIR for the fixed reference trajectory")
    ap.add_argument("--target", action="append", required=True, help="LABEL=SELECTED_DIR for each target trajectory")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--ref-prev", default="chck_82M")
    ap.add_argument("--ref-mid", default="chck_84M")
    ap.add_argument("--ref-next", default="chck_86M")
    ap.add_argument("--fixed-prev", default="chck_82M")
    ap.add_argument("--fixed-mid", default="chck_84M")
    ap.add_argument("--fixed-next", default="chck_86M")
    ap.add_argument("--best-metric", default="cheap7")
    ap.add_argument("--min-mid-m", type=int, default=72)
    ap.add_argument("--max-mid-m", type=int, default=98)
    ap.add_argument("--min-abs-ref-net-pct", type=float, default=0.25)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    transition_results = load_step197()
    s196 = transition_results.load_step196()
    late_pair_results = s196.load_step186()

    ref_label, ref_dir = parse_labeled_dir(args.reference)
    ref_window = (args.ref_prev, args.ref_mid, args.ref_next)
    ref_sig = load_window_signature(ref_label, ref_dir, ref_window, out_dir, s196, late_pair_results, transition_results)
    ref_summary = dict(ref_sig["summary"])

    all_group_rows: list[dict[str, Any]] = []
    all_item_rows: list[dict[str, Any]] = []
    all_item_pattern_rows: list[dict[str, Any]] = []
    targets_summary: dict[str, Any] = {}

    for target_spec in args.target:
        label, selected_dir = parse_labeled_dir(target_spec)
        complete_eps, traj_rows, incomplete = complete_endpoints(label, selected_dir, s196)
        if not complete_eps:
            raise RuntimeError(f"{label}: no complete endpoints")
        best_ep = choose_best_endpoint(traj_rows, complete_eps, args.best_metric)
        windows = consecutive_windows(complete_eps, s196, args.min_mid_m, args.max_mid_m)
        if not windows:
            raise RuntimeError(f"{label}: no eligible consecutive windows")
        fixed_window = (args.fixed_prev, args.fixed_mid, args.fixed_next)
        target_summary = {
            "label": label,
            "selected_dir": safe_rel(selected_dir),
            "complete_endpoints": complete_eps,
            "incomplete_or_missing": incomplete,
            "n_trajectory_rows": len(traj_rows),
            "best_metric": args.best_metric,
            "best_endpoint": best_ep,
            "own_peak_window": None,
            "fixed_window_present": fixed_window in windows,
            "n_windows": len(windows),
            "windows": [],
        }
        by_ep = {r.get("endpoint"): r for r in traj_rows}
        for w in windows:
            role = "background"
            if w == fixed_window:
                role = "fixed_82_84_86"
            if w[1] == best_ep:
                role = "own_peak" if role == "background" else "own_peak_and_fixed_82_84_86"
                target_summary["own_peak_window"] = f"{w[0]}->{w[1]}->{w[2]}"
            sig = load_window_signature(label, selected_dir, w, out_dir, s196, late_pair_results, transition_results)
            group_rows, item_sim_rows, item_pattern = compare_signature(ref_sig, sig, args.min_abs_ref_net_pct, transition_results)
            mid_row = by_ep.get(w[1], {}) or {}
            common = {
                "target_label": label,
                "target_selected_dir": safe_rel(selected_dir),
                "target_prev": w[0],
                "target_mid": w[1],
                "target_next": w[2],
                "target_window_string": f"{w[0]}->{w[1]}->{w[2]}",
                "target_window_role": role,
                "target_mid_selected_cheap7": mid_row.get("cheap7"),
                "target_mid_selected_BLiMP": mid_row.get("BLiMP"),
                "target_mid_selected_Supplement": mid_row.get("Supplement"),
                "target_mid_selected_EWoK": mid_row.get("EWoK"),
                "target_mid_selected_Entity": mid_row.get("Entity"),
                "target_mid_selected_COMPS": mid_row.get("COMPS"),
                "target_mid_selected_GlobalPIQA": mid_row.get("GlobalPIQA"),
                "target_mid_selected_Reading": mid_row.get("Reading"),
            }
            for r in group_rows:
                rr = dict(common)
                rr.update(r)
                all_group_rows.append(rr)
            for r in item_sim_rows:
                rr = dict(common)
                rr.update(r)
                all_item_rows.append(rr)
            ip = dict(common)
            ip.update(item_pattern)
            all_item_pattern_rows.append(ip)
            target_summary["windows"].append({**common, **{"n_classification_items": sig["summary"]["n_classification_items"]}})
        targets_summary[label] = target_summary

    add_rank_summaries(all_group_rows, all_item_pattern_rows, targets_summary)

    # Best window summaries for a compact markdown view.
    for label, ts in targets_summary.items():
        rows = [r for r in all_group_rows if r["target_label"] == label]
        best_summaries: list[dict[str, Any]] = []
        for level in ["column_official_subtask_mean", "column_subtask"]:
            for window in ["rise", "fall", "rise+fall_pattern"]:
                for metric in ["cosine_net_pct", "weighted_pearson_net_pct"]:
                    candidates = [r for r in rows if r.get("group_level") == level and r.get("window") == window and sim_value_for_rank(r, metric) is not None]
                    if not candidates:
                        continue
                    best = max(candidates, key=lambda r: float(r[metric]))
                    own = next((r for r in candidates if r.get("target_window_role") in ["own_peak", "own_peak_and_fixed_82_84_86"]), None)
                    fixed = next((r for r in candidates if r.get("target_window_role") in ["fixed_82_84_86", "own_peak_and_fixed_82_84_86"]), None)
                    best_summaries.append({
                        "group_level": level,
                        "window": window,
                        "metric": metric,
                        "best_window_string": best.get("target_window_string"),
                        "best_window_role": best.get("target_window_role"),
                        "best_value": best.get(metric),
                        "own_peak_value": own.get(metric) if own else None,
                        "fixed_value": fixed.get(metric) if fixed else None,
                    })
        ts["best_window_summaries"] = best_summaries

    group_csv = out_dir / "window_group_signature_similarity.csv"
    item_csv = out_dir / "window_item_signed_transition_similarity.csv"
    pattern_csv = out_dir / "window_item_pattern_similarity.csv"
    write_csv(group_csv, all_group_rows)
    write_csv(item_csv, all_item_rows)
    write_csv(pattern_csv, all_item_pattern_rows)

    out_json = out_dir / "transition_window_background_summary.json"
    out_md = out_dir / "transition_window_background_summary.md"
    summary = {
        "status": "TRANSITION_WINDOW_BACKGROUND",
        "created_utc": now_utc(),
        "elapsed_sec": None,
        "reference": ref_summary,
        "best_metric": args.best_metric,
        "fixed_target_window": f"{args.fixed_prev}->{args.fixed_mid}->{args.fixed_next}",
        "eligible_window_mid_m_range": [args.min_mid_m, args.max_mid_m],
        "targets": targets_summary,
        "files": {
            "window_group_signature_similarity_csv": {"path": safe_rel(group_csv), "size": group_csv.stat().st_size, "sha256": sha256_file(group_csv)},
            "window_item_signed_transition_similarity_csv": {"path": safe_rel(item_csv), "size": item_csv.stat().st_size, "sha256": sha256_file(item_csv)},
            "window_item_pattern_similarity_csv": {"path": safe_rel(pattern_csv), "size": pattern_csv.stat().st_size, "sha256": sha256_file(pattern_csv)},
        },
        "out_json": safe_rel(out_json),
        "out_md": safe_rel(out_md),
        "note": "CPU/file-only; compares every eligible target window against fixed reference transition signature; no model inference/training/upload/submission.",
    }
    summary["elapsed_sec"] = round(time.time() - t0, 3)
    write_json(out_json, summary)
    write_md(summary, out_md)
    print(json.dumps({"status": summary["status"], "out_json": safe_rel(out_json), "out_md": safe_rel(out_md), "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
