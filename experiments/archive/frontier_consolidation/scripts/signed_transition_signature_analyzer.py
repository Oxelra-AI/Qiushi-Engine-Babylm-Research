#!/usr/bin/env python3
"""research: signed transition signatures for DeBERTa selected trajectories.

CPU/file-only.  This tool consumes already-written selected-grid prediction payloads
and compares *changes* around each trajectory's own late aggregate peak: pre-peak
-> peak ("rise") and peak -> post-peak ("fall").  It is designed specifically to
avoid using static best-endpoint correct-set overlap as the main robustness readout.

Scientific purpose: when seed43122 delivers, ask whether the same BabyLM capability
families strengthen and erode together across trajectories.  Endpoint overlap is
supporting context only; signed transition similarity over items, tasks, and
subtasks is the central evidence.
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
from collections import defaultdict
from typing import Any

CLASS_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]
EXPECTED_ENDPOINTS = [f"chck_{m}M" for m in range(70, 102, 2)]


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
PATH = ROOT / "experiments/archive/frontier_consolidation/scripts/multitrajectory_item_dynamics.py"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_step196():
    spec = importlib.util.spec_from_file_location("multitrajectory_item_dynamics", PATH)
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
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
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


def pct(x: float) -> float:
    return 100.0 * x


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def safe_rel(path: pathlib.Path) -> str:
    p = path.resolve() if path.is_absolute() else path
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(path)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def weighted_pearson(xs: list[float], ys: list[float], ws: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys) or len(xs) != len(ws):
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    vx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    vy = sum(w * (y - my) ** 2 for y, w in zip(ys, ws))
    if vx <= 0 or vy <= 0:
        return None
    return sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / math.sqrt(vx * vy)


def cosine(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    dot = sum(x * y for x, y in zip(xs, ys))
    nx = math.sqrt(sum(x * x for x in xs))
    ny = math.sqrt(sum(y * y for y in ys))
    if nx <= 0 or ny <= 0:
        return None
    return dot / (nx * ny)


def endpoint_int(ep: str, s196) -> int:
    return s196.endpoint_int(ep)


def endpoint_suffix(ep: str, s196) -> str:
    return s196.endpoint_suffix(ep)


def selected_metric(row: dict[str, Any], metric: str) -> float | None:
    val = row.get(metric)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def choose_best_endpoint(rows: list[dict[str, Any]], usable_set: set[str], best_metric: str) -> str:
    candidates = [r for r in rows if r.get("endpoint") in usable_set and selected_metric(r, best_metric) is not None]
    if not candidates:
        raise RuntimeError(f"No usable rows with selected metric {best_metric}")
    return str(max(candidates, key=lambda r: float(r[best_metric]))["endpoint"])


def load_payload_for_endpoint(selected_dir: pathlib.Path, label: str, ep: str, s196, late_pair_results) -> dict[str, Any]:
    path = s196.per_target_path(selected_dir, label, ep)
    if not path.exists():
        raise FileNotFoundError(f"Missing per-target payload for {label} {ep}: {path}")
    payload = read_json(path)
    ok, bad = s196.payload_is_complete(payload)
    if not ok:
        raise RuntimeError(f"Incomplete payload for {label} {ep}: {bad}")
    # Force schema alignment immediately.
    _ = late_pair_results.endpoint_items(payload)
    return payload


def group_key(row: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(f, "") for f in fields)


def group_summary_rows(item_rows: list[dict[str, Any]], group_fields: list[str], s196) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in item_rows:
        groups[group_key(r, group_fields)].append(r)
    rows: list[dict[str, Any]] = []
    for key, rs in groups.items():
        rec: dict[str, Any] = {f: key[i] for i, f in enumerate(group_fields)}
        rec["group_level"] = "+".join(group_fields)
        rec["group_id"] = "||".join(str(x) for x in key)
        rec["n_items"] = len(rs)
        for window in ["rise", "fall"]:
            ds = [int(r[f"delta_{window}"]) for r in rs if f"delta_{window}" in r]
            if not ds:
                continue
            gains = sum(1 for d in ds if d == 1)
            losses = sum(1 for d in ds if d == -1)
            unchanged = sum(1 for d in ds if d == 0)
            net = gains - losses
            rec[f"{window}_gains"] = gains
            rec[f"{window}_losses"] = losses
            rec[f"{window}_unchanged"] = unchanged
            rec[f"{window}_net"] = net
            rec[f"{window}_net_pct"] = pct(net / len(ds)) if ds else None
            rec[f"{window}_gain_pct"] = pct(gains / len(ds)) if ds else None
            rec[f"{window}_loss_pct"] = pct(losses / len(ds)) if ds else None
            rec[f"{window}_churn_pct"] = pct((gains + losses) / len(ds)) if ds else None
        rows.append(rec)
    def sort_key(r: dict[str, Any]):
        col = r.get("column", "")
        idx = CLASS_COLUMNS.index(col) if col in CLASS_COLUMNS else 999
        mag = max(abs(float(r.get("rise_net_pct") or 0.0)), abs(float(r.get("fall_net_pct") or 0.0)))
        return (idx, -mag, r.get("subtask", ""), r.get("subgroup", ""), r.get("fine_group", ""))
    rows.sort(key=sort_key)
    return rows


def make_item_transition_rows(label: str, merged: list[dict[str, Any]], prev_ep: str | None, best_ep: str, next_ep: str | None, s196) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    s_best = endpoint_suffix(best_ep, s196)
    s_prev = endpoint_suffix(prev_ep, s196) if prev_ep else None
    s_next = endpoint_suffix(next_ep, s196) if next_ep else None
    for r in merged:
        rec = {
            "label": label,
            "item_key": r["item_key"],
            "column": r["column"],
            "subtask": r.get("subtask", ""),
            "subgroup": r.get("subgroup", ""),
            "fine_group": r.get("fine_group", ""),
            "best_endpoint": best_ep,
            "correct_best": int(r[f"correct_{s_best}"]),
        }
        if prev_ep and s_prev:
            c0 = int(r[f"correct_{s_prev}"])
            c1 = int(r[f"correct_{s_best}"])
            rec["rise_from"] = prev_ep
            rec["rise_to"] = best_ep
            rec["correct_pre"] = c0
            rec["delta_rise"] = c1 - c0
        if next_ep and s_next:
            c0 = int(r[f"correct_{s_best}"])
            c1 = int(r[f"correct_{s_next}"])
            rec["fall_from"] = best_ep
            rec["fall_to"] = next_ep
            rec["correct_post"] = c1
            rec["delta_fall"] = c1 - c0
        rows.append(rec)
    return rows


def analyze_one(label: str, selected_dir: pathlib.Path, out_dir: pathlib.Path, best_metric: str, keep_item_csv: bool, s196, late_pair_results) -> dict[str, Any]:
    trajectory_rows = s196.read_trajectory_rows(selected_dir)
    # First decide which selected rows are complete enough for transition payloads.
    complete_eps: list[str] = []
    incomplete: list[dict[str, Any]] = []
    for r in trajectory_rows:
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
                complete_eps.append(ep)
            else:
                incomplete.append({"endpoint": ep, "reason": "incomplete_payload", "bad": bad, "path": safe_rel(path)})
        except Exception as e:
            incomplete.append({"endpoint": ep, "reason": f"exception:{type(e).__name__}", "error": str(e)[:300], "path": safe_rel(path)})
    complete_eps = sorted(set(complete_eps), key=s196.endpoint_int)
    if not complete_eps:
        raise RuntimeError(f"{label}: no complete endpoints")
    best_ep = choose_best_endpoint(trajectory_rows, set(complete_eps), best_metric)
    i = complete_eps.index(best_ep)
    prev_ep = complete_eps[i - 1] if i > 0 else None
    next_ep = complete_eps[i + 1] if i + 1 < len(complete_eps) else None
    if prev_ep is None or next_ep is None:
        raise RuntimeError(f"{label}: best endpoint {best_ep} lacks adjacent window inside complete endpoints {complete_eps}")
    needed_eps = [prev_ep, best_ep, next_ep]
    payloads = {ep: load_payload_for_endpoint(selected_dir, label, ep, s196, late_pair_results) for ep in needed_eps}
    items_by_ep = {ep: late_pair_results.endpoint_items(payloads[ep]) for ep in needed_eps}
    merged = late_pair_results.merge_endpoint_items(items_by_ep)
    item_rows = make_item_transition_rows(label, merged, prev_ep, best_ep, next_ep, s196)

    group_levels = {
        "column": ["column"],
        "column_subtask": ["column", "subtask"],
        "column_subtask_subgroup": ["column", "subtask", "subgroup"],
        "column_subtask_subgroup_fine": ["column", "subtask", "subgroup", "fine_group"],
    }
    group_summaries: dict[str, list[dict[str, Any]]] = {}
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    for name, fields in group_levels.items():
        rows = group_summary_rows(item_rows, fields, s196)
        group_summaries[name] = rows
        write_csv(label_dir / f"{name}_transition_summary.csv", rows)
    group_summaries["column_official_subtask_mean"] = official_like_column_transition_from_subtasks(group_summaries["column_subtask"])
    write_csv(label_dir / "column_official_subtask_mean_transition_summary.csv", group_summaries["column_official_subtask_mean"])
    if keep_item_csv:
        write_csv(label_dir / "item_signed_transitions.csv", item_rows)

    by_ep = {r.get("endpoint"): r for r in trajectory_rows}
    window_scores = {}
    for ep in needed_eps:
        row = by_ep.get(ep, {})
        window_scores[ep] = {k: row.get(k) for k in ["cheap7", "BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "relation_state_mean"] if k in row}
    summary = {
        "label": label,
        "selected_dir": safe_rel(selected_dir),
        "status": "ok" if len(complete_eps) == len(trajectory_rows) and not incomplete else "partial",
        "n_trajectory_rows": len(trajectory_rows),
        "complete_endpoints": complete_eps,
        "incomplete_or_missing": incomplete,
        "best_metric": best_metric,
        "best_endpoint": best_ep,
        "prev_endpoint": prev_ep,
        "next_endpoint": next_ep,
        "n_classification_items": len(item_rows),
        "window_scores_selected_rows": window_scores,
        "column_transitions": group_summaries["column_official_subtask_mean"],
        "column_transitions_item_weighted": group_summaries["column"],
        "top_subtask_movers": top_movers(group_summaries["column_subtask"], n=20),
        "files": {},
    }
    file_names = [f"{name}_transition_summary.csv" for name in group_levels] + ["column_official_subtask_mean_transition_summary.csv"]
    if keep_item_csv:
        file_names.append("item_signed_transitions.csv")
    for fn in file_names:
        p = label_dir / fn
        summary["files"][fn] = {"path": safe_rel(p), "size": p.stat().st_size, "sha256": sha256_file(p)}
    return {
        "summary": summary,
        "item_rows": item_rows,
        "group_summaries": group_summaries,
    }


def official_like_column_transition_from_subtasks(subtask_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate subtask transition rows to official-like column movement.

    The BabyLM cheap columns are not raw item-count averages: BLiMP/Supplement/EWoK/etc.
    are effectively means over task units/subtasks.  For signed dynamics we therefore
    need both raw item-weighted movement (useful for churn) and subtask-mean movement
    (the score-coordinate capability family movement).  This function produces the
    latter by averaging per-subtask net percentages within each column.
    """
    by_col: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in subtask_rows:
        by_col[str(r.get("column", ""))].append(r)
    out: list[dict[str, Any]] = []
    for col, rs in by_col.items():
        rec: dict[str, Any] = {
            "column": col,
            "group_id": col,
            "group_level": "column_official_subtask_mean",
            "n_subtasks": len(rs),
            "n_items": sum(int(r.get("n_items") or 0) for r in rs),
        }
        for window in ["rise", "fall"]:
            vals = [float(r.get(f"{window}_net_pct")) for r in rs if r.get(f"{window}_net_pct") is not None]
            if vals:
                rec[f"{window}_net_pct"] = statistics.fmean(vals)
                rec[f"{window}_gain_pct_subtask_mean"] = statistics.fmean(float(r.get(f"{window}_gain_pct") or 0.0) for r in rs)
                rec[f"{window}_loss_pct_subtask_mean"] = statistics.fmean(float(r.get(f"{window}_loss_pct") or 0.0) for r in rs)
                rec[f"{window}_churn_pct_subtask_mean"] = statistics.fmean(float(r.get(f"{window}_churn_pct") or 0.0) for r in rs)
                rec[f"{window}_gains"] = sum(int(r.get(f"{window}_gains") or 0) for r in rs)
                rec[f"{window}_losses"] = sum(int(r.get(f"{window}_losses") or 0) for r in rs)
        out.append(rec)
    out.sort(key=lambda r: CLASS_COLUMNS.index(r["column"]) if r.get("column") in CLASS_COLUMNS else 999)
    return out


def top_movers(group_rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    rows = []
    for r in group_rows:
        mag = max(abs(float(r.get("rise_net_pct") or 0.0)), abs(float(r.get("fall_net_pct") or 0.0)))
        rr = {k: r.get(k) for k in ["column", "subtask", "subgroup", "fine_group", "n_items", "rise_net_pct", "fall_net_pct", "rise_churn_pct", "fall_churn_pct"] if k in r}
        rr["max_abs_net_pct"] = mag
        rows.append(rr)
    rows.sort(key=lambda r: float(r.get("max_abs_net_pct") or 0.0), reverse=True)
    return rows[:n]


def summarize_item_transition_pair(ref_rows: list[dict[str, Any]], oth_rows: list[dict[str, Any]], window: str) -> list[dict[str, Any]]:
    ref_by_key = {r["item_key"]: r for r in ref_rows}
    oth_by_key = {r["item_key"]: r for r in oth_rows}
    common = sorted(set(ref_by_key) & set(oth_by_key))
    buckets: dict[str, list[str]] = defaultdict(list)
    buckets["ALL"] = common
    for key in common:
        buckets[ref_by_key[key].get("column", "")].append(key)
    rows: list[dict[str, Any]] = []
    for col, keys in sorted(buckets.items(), key=lambda kv: -1 if kv[0] == "ALL" else (CLASS_COLUMNS.index(kv[0]) if kv[0] in CLASS_COLUMNS else 999)):
        rd: list[int] = []
        od: list[int] = []
        for key in keys:
            if f"delta_{window}" not in ref_by_key[key] or f"delta_{window}" not in oth_by_key[key]:
                continue
            rd.append(int(ref_by_key[key][f"delta_{window}"]))
            od.append(int(oth_by_key[key][f"delta_{window}"]))
        n = len(rd)
        if not n:
            continue
        ref_gain = sum(1 for d in rd if d == 1)
        ref_loss = sum(1 for d in rd if d == -1)
        oth_gain = sum(1 for d in od if d == 1)
        oth_loss = sum(1 for d in od if d == -1)
        ref_changed = sum(1 for d in rd if d != 0)
        oth_changed = sum(1 for d in od if d != 0)
        same_gain = sum(1 for a, b in zip(rd, od) if a == 1 and b == 1)
        same_loss = sum(1 for a, b in zip(rd, od) if a == -1 and b == -1)
        opposite = sum(1 for a, b in zip(rd, od) if a != 0 and b == -a)
        same_on_ref_changed = sum(1 for a, b in zip(rd, od) if a != 0 and b == a)
        other_zero_on_ref_changed = sum(1 for a, b in zip(rd, od) if a != 0 and b == 0)
        gain_union = sum(1 for a, b in zip(rd, od) if a == 1 or b == 1)
        loss_union = sum(1 for a, b in zip(rd, od) if a == -1 or b == -1)
        row = {
            "window": window,
            "column": col,
            "n_items": n,
            "ref_gain": ref_gain,
            "ref_loss": ref_loss,
            "ref_net": ref_gain - ref_loss,
            "ref_net_pct": pct((ref_gain - ref_loss) / n),
            "ref_churn_pct": pct(ref_changed / n),
            "other_gain": oth_gain,
            "other_loss": oth_loss,
            "other_net": oth_gain - oth_loss,
            "other_net_pct": pct((oth_gain - oth_loss) / n),
            "other_churn_pct": pct(oth_changed / n),
            "same_gain": same_gain,
            "same_loss": same_loss,
            "same_signed_transition_on_ref_changed": same_on_ref_changed,
            "opposite_signed_transition_on_ref_changed": opposite,
            "other_zero_on_ref_changed": other_zero_on_ref_changed,
            "sign_agreement_on_ref_changed": same_on_ref_changed / ref_changed if ref_changed else None,
            "opposition_on_ref_changed": opposite / ref_changed if ref_changed else None,
            "other_zero_on_ref_changed_fraction": other_zero_on_ref_changed / ref_changed if ref_changed else None,
            "signed_transition_cosine": cosine([float(x) for x in rd], [float(x) for x in od]),
            "gain_jaccard": same_gain / gain_union if gain_union else None,
            "loss_jaccard": same_loss / loss_union if loss_union else None,
        }
        rows.append(row)
    return rows


def group_map(rows: list[dict[str, Any]], level: str) -> dict[str, dict[str, Any]]:
    return {r["group_id"]: r for r in rows if r.get("group_id")}


def compare_group_vectors(ref_groups: dict[str, list[dict[str, Any]]], oth_groups: dict[str, list[dict[str, Any]]], min_abs_ref_net_pct: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sim_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for level in ["column_official_subtask_mean", "column", "column_subtask", "column_subtask_subgroup"]:
        rg = group_map(ref_groups[level], level)
        og = group_map(oth_groups[level], level)
        common = sorted(set(rg) & set(og))
        for window in ["rise", "fall"]:
            xs: list[float] = []
            ys: list[float] = []
            ws: list[float] = []
            sign_all = 0
            sign_movers = 0
            n_movers = 0
            n_nonzero_ref = 0
            for gid in common:
                x = float(rg[gid].get(f"{window}_net_pct") or 0.0)
                y = float(og[gid].get(f"{window}_net_pct") or 0.0)
                w = math.sqrt(max(1, min(int(rg[gid].get("n_items") or 1), int(og[gid].get("n_items") or 1))))
                xs.append(x); ys.append(y); ws.append(w)
                if x != 0:
                    n_nonzero_ref += 1
                    if (x > 0 and y > 0) or (x < 0 and y < 0) or (x == 0 and y == 0):
                        sign_all += 1
                if abs(x) >= min_abs_ref_net_pct:
                    n_movers += 1
                    if (x > 0 and y > 0) or (x < 0 and y < 0):
                        sign_movers += 1
            sim_rows.append({
                "group_level": level,
                "window": window,
                "n_groups": len(common),
                "pearson_net_pct": pearson(xs, ys),
                "weighted_pearson_net_pct": weighted_pearson(xs, ys, ws),
                "cosine_net_pct": cosine(xs, ys),
                "ref_nonzero_groups": n_nonzero_ref,
                "sign_agreement_ref_nonzero": sign_all / n_nonzero_ref if n_nonzero_ref else None,
                "ref_mover_threshold_abs_net_pct": min_abs_ref_net_pct,
                "ref_mover_groups": n_movers,
                "sign_agreement_ref_movers": sign_movers / n_movers if n_movers else None,
            })
        # Concatenated pattern over rise+fall at this group level.
        xs_pat: list[float] = []
        ys_pat: list[float] = []
        ws_pat: list[float] = []
        n_pattern_movers = 0
        n_pattern_same = 0
        for gid in common:
            rx = float(rg[gid].get("rise_net_pct") or 0.0)
            fx = float(rg[gid].get("fall_net_pct") or 0.0)
            ry = float(og[gid].get("rise_net_pct") or 0.0)
            fy = float(og[gid].get("fall_net_pct") or 0.0)
            w = math.sqrt(max(1, min(int(rg[gid].get("n_items") or 1), int(og[gid].get("n_items") or 1))))
            xs_pat.extend([rx, fx]); ys_pat.extend([ry, fy]); ws_pat.extend([w, w])
            if max(abs(rx), abs(fx)) >= min_abs_ref_net_pct:
                n_pattern_movers += 1
                rise_ok = (rx > 0 and ry > 0) or (rx < 0 and ry < 0) or abs(rx) < min_abs_ref_net_pct
                fall_ok = (fx > 0 and fy > 0) or (fx < 0 and fy < 0) or abs(fx) < min_abs_ref_net_pct
                if rise_ok and fall_ok:
                    n_pattern_same += 1
        sim_rows.append({
            "group_level": level,
            "window": "rise+fall_pattern",
            "n_groups": len(common),
            "pearson_net_pct": pearson(xs_pat, ys_pat),
            "weighted_pearson_net_pct": weighted_pearson(xs_pat, ys_pat, ws_pat),
            "cosine_net_pct": cosine(xs_pat, ys_pat),
            "ref_mover_threshold_abs_net_pct": min_abs_ref_net_pct,
            "ref_pattern_mover_groups": n_pattern_movers,
            "same_signed_pattern_on_ref_movers": n_pattern_same,
            "same_signed_pattern_fraction": n_pattern_same / n_pattern_movers if n_pattern_movers else None,
        })
    # Top reference subtask families: these are the most interpretable capability units.
    rg = group_map(ref_groups["column_subtask"], "column_subtask")
    og = group_map(oth_groups["column_subtask"], "column_subtask")
    rows_for_top = []
    for gid in sorted(set(rg) & set(og)):
        r = rg[gid]; o = og[gid]
        ref_rise = float(r.get("rise_net_pct") or 0.0)
        ref_fall = float(r.get("fall_net_pct") or 0.0)
        oth_rise = float(o.get("rise_net_pct") or 0.0)
        oth_fall = float(o.get("fall_net_pct") or 0.0)
        mag = max(abs(ref_rise), abs(ref_fall))
        rows_for_top.append({
            "column": r.get("column"),
            "subtask": r.get("subtask"),
            "n_items": r.get("n_items"),
            "ref_rise_net_pct": ref_rise,
            "other_rise_net_pct": oth_rise,
            "rise_same_sign": (ref_rise > 0 and oth_rise > 0) or (ref_rise < 0 and oth_rise < 0),
            "ref_fall_net_pct": ref_fall,
            "other_fall_net_pct": oth_fall,
            "fall_same_sign": (ref_fall > 0 and oth_fall > 0) or (ref_fall < 0 and oth_fall < 0),
            "ref_max_abs_net_pct": mag,
            "pattern_same_sign_when_ref_moves": (((abs(ref_rise) < min_abs_ref_net_pct) or ((ref_rise > 0 and oth_rise > 0) or (ref_rise < 0 and oth_rise < 0))) and ((abs(ref_fall) < min_abs_ref_net_pct) or ((ref_fall > 0 and oth_fall > 0) or (ref_fall < 0 and oth_fall < 0)))),
        })
    rows_for_top.sort(key=lambda r: float(r["ref_max_abs_net_pct"]), reverse=True)
    top_rows = rows_for_top[:40]
    return sim_rows, top_rows


def compare_to_reference(reference_label: str, analyses: dict[str, dict[str, Any]], out_dir: pathlib.Path, min_abs_ref_net_pct: float) -> dict[str, Any]:
    if reference_label not in analyses:
        return {"status": "no_reference", "rows": []}
    ref = analyses[reference_label]
    all_item_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    all_top_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for label, an in analyses.items():
        if label == reference_label:
            continue
        item_rows: list[dict[str, Any]] = []
        for window in ["rise", "fall"]:
            item_rows.extend(summarize_item_transition_pair(ref["item_rows"], an["item_rows"], window))
        for r in item_rows:
            r["other_label"] = label
            r["reference_label"] = reference_label
            r["reference_window"] = f"{ref['summary']['prev_endpoint']}->{ref['summary']['best_endpoint']}" if r["window"] == "rise" else f"{ref['summary']['best_endpoint']}->{ref['summary']['next_endpoint']}"
            r["other_window"] = f"{an['summary']['prev_endpoint']}->{an['summary']['best_endpoint']}" if r["window"] == "rise" else f"{an['summary']['best_endpoint']}->{an['summary']['next_endpoint']}"
        all_item_rows.extend(item_rows)
        group_rows, top_rows = compare_group_vectors(ref["group_summaries"], an["group_summaries"], min_abs_ref_net_pct)
        for r in group_rows:
            r["other_label"] = label
            r["reference_label"] = reference_label
            r["reference_rise_window"] = f"{ref['summary']['prev_endpoint']}->{ref['summary']['best_endpoint']}"
            r["reference_fall_window"] = f"{ref['summary']['best_endpoint']}->{ref['summary']['next_endpoint']}"
            r["other_rise_window"] = f"{an['summary']['prev_endpoint']}->{an['summary']['best_endpoint']}"
            r["other_fall_window"] = f"{an['summary']['best_endpoint']}->{an['summary']['next_endpoint']}"
        for r in top_rows:
            r["other_label"] = label
            r["reference_label"] = reference_label
        all_group_rows.extend(group_rows)
        all_top_rows.extend(top_rows)
        summaries[label] = {
            "item_transition_similarity_ALL": [r for r in item_rows if r.get("column") == "ALL"],
            "group_transition_similarity_column_official_subtask_mean": [r for r in group_rows if r.get("group_level") == "column_official_subtask_mean"],
            "group_transition_similarity_column_subtask": [r for r in group_rows if r.get("group_level") == "column_subtask"],
            "top_reference_subtask_transition_families": top_rows[:20],
        }
    write_csv(out_dir / "pairwise_item_signed_transition_similarity.csv", all_item_rows)
    write_csv(out_dir / "pairwise_group_signed_transition_similarity.csv", all_group_rows)
    write_csv(out_dir / "top_reference_subtask_transition_families.csv", all_top_rows)
    return {
        "status": "ok",
        "reference_label": reference_label,
        "min_abs_ref_net_pct": min_abs_ref_net_pct,
        "pairwise_item_signed_transition_similarity_csv": safe_rel(out_dir / "pairwise_item_signed_transition_similarity.csv"),
        "pairwise_group_signed_transition_similarity_csv": safe_rel(out_dir / "pairwise_group_signed_transition_similarity.csv"),
        "top_reference_subtask_transition_families_csv": safe_rel(out_dir / "top_reference_subtask_transition_families.csv"),
        "by_other": summaries,
    }


def write_md(summary: dict[str, Any], path: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append("# research signed transition signature analyzer\n\n")
    lines.append("CPU/file-only analysis of already written selected-grid prediction payloads. It compares signed item transitions, not static endpoint overlap.\n\n")
    lines.append("## Trajectory windows\n\n")
    lines.append("| label | status | best metric | prev→best | best→next | n classification items | selected cheap7(prev,best,next) |\n")
    lines.append("|---|---|---|---|---|---:|---|\n")
    for label, s in summary["trajectories"].items():
        scores = s.get("window_scores_selected_rows", {})
        vals = []
        for ep in [s.get("prev_endpoint"), s.get("best_endpoint"), s.get("next_endpoint")]:
            vals.append(fmt(scores.get(ep, {}).get("cheap7")) if ep else "")
        lines.append(f"| {label} | {s.get('status')} | {s.get('best_metric')} | {s.get('prev_endpoint')}→{s.get('best_endpoint')} | {s.get('best_endpoint')}→{s.get('next_endpoint')} | {s.get('n_classification_items')} | {', '.join(vals)} |\n")
    lines.append("\n## Column signed transitions around each trajectory's own peak\n\n")
    for label, s in summary["trajectories"].items():
        lines.append(f"### {label}\n\n")
        lines.append("| column | n | rise net pct | rise gains/losses | fall net pct | fall gains/losses |\n")
        lines.append("|---|---:|---:|---|---:|---|\n")
        for r in s.get("column_transitions", []):
            lines.append(f"| {r.get('column')} | {r.get('n_items')} | {fmt(r.get('rise_net_pct'))} | {r.get('rise_gains')}/{r.get('rise_losses')} | {fmt(r.get('fall_net_pct'))} | {r.get('fall_gains')}/{r.get('fall_losses')} |\n")
        lines.append("\n")
    pair = summary.get("reference_comparison", {}).get("by_other", {})
    if pair:
        lines.append("## Pairwise signed-transition comparison to reference\n\n")
        for other, rec in pair.items():
            lines.append(f"### {other}\n\n")
            lines.append("Item-level transition similarity over all classification items:\n\n")
            lines.append("| window | ref window | other window | ref net pct | other net pct | ref churn pct | other churn pct | sign agreement on ref-changed | opposite on ref-changed | cosine | gain J | loss J |\n")
            lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in rec.get("item_transition_similarity_ALL", []):
                lines.append(f"| {r.get('window')} | {r.get('reference_window')} | {r.get('other_window')} | {fmt(r.get('ref_net_pct'))} | {fmt(r.get('other_net_pct'))} | {fmt(r.get('ref_churn_pct'))} | {fmt(r.get('other_churn_pct'))} | {fmt(r.get('sign_agreement_on_ref_changed'))} | {fmt(r.get('opposition_on_ref_changed'))} | {fmt(r.get('signed_transition_cosine'))} | {fmt(r.get('gain_jaccard'))} | {fmt(r.get('loss_jaccard'))} |\n")
            lines.append("\nOfficial-like column and subtask-level signed net-vector similarity:\n\n")
            lines.append("| level | window | groups | Pearson | weighted Pearson | cosine | sign agreement on ref movers | ref mover groups |\n")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for level_name, rows in [("column_official_subtask_mean", rec.get("group_transition_similarity_column_official_subtask_mean", [])), ("column_subtask", rec.get("group_transition_similarity_column_subtask", []))]:
                for r in rows:
                    lines.append(f"| {level_name} | {r.get('window')} | {r.get('n_groups')} | {fmt(r.get('pearson_net_pct'))} | {fmt(r.get('weighted_pearson_net_pct'))} | {fmt(r.get('cosine_net_pct'))} | {fmt(r.get('sign_agreement_ref_movers') or r.get('same_signed_pattern_fraction'))} | {r.get('ref_mover_groups') or r.get('ref_pattern_mover_groups')} |\n")
            lines.append("\nTop reference moving subtask families (showing whether the other trajectory moves the same families in the same directions):\n\n")
            lines.append("| column | subtask | n | ref rise | other rise | ref fall | other fall | pattern same? |\n")
            lines.append("|---|---|---:|---:|---:|---:|---:|---|\n")
            for r in rec.get("top_reference_subtask_transition_families", [])[:12]:
                lines.append(f"| {r.get('column')} | {r.get('subtask')} | {r.get('n_items')} | {fmt(r.get('ref_rise_net_pct'))} | {fmt(r.get('other_rise_net_pct'))} | {fmt(r.get('ref_fall_net_pct'))} | {fmt(r.get('other_fall_net_pct'))} | {r.get('pattern_same_sign_when_ref_moves')} |\n")
            lines.append("\n")
    lines.append("## Reading the result\n\n")
    lines.append("For seed43122, use this signed-transition readout before interpreting static correct-set overlap. A robust residual-capacity late phase should show the same task/subtask families strengthening from pre-peak to peak and eroding from peak to post-peak. The column table is official-like (subtask mean) so it aligns with score-coordinate family movement; item-weighted column files are also saved for churn anatomy. If transition signatures do not recur, treat seed43022 `chck_84M` as an endpoint asset and make stochastic competence stabilization the next target rather than tuning the original peak.\n\n")
    lines.append(f"JSON: `{summary['out_json']}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", action="append", required=True, help="LABEL=SELECTED_DIR")
    ap.add_argument("--reference-label", default=None)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--best-metric", default="cheap7")
    ap.add_argument("--min-abs-ref-net-pct", type=float, default=0.25, help="reference subtask movement threshold for sign-pattern summaries")
    ap.add_argument("--keep-item-csv", action="store_true", help="write per-item transition CSVs; large")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    s196 = load_step196()
    late_pair_results = s196.load_step186()
    analyses: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for spec in args.trajectory:
        if "=" not in spec:
            raise SystemExit(f"Use LABEL=SELECTED_DIR for --trajectory, got {spec!r}")
        label, d = spec.split("=", 1)
        selected_dir = pathlib.Path(d)
        if not selected_dir.is_absolute():
            selected_dir = ROOT / selected_dir
        result = analyze_one(label, selected_dir, out_dir, args.best_metric, args.keep_item_csv, s196, late_pair_results)
        analyses[label] = result
        summaries[label] = result["summary"]
    reference_comparison = compare_to_reference(args.reference_label, analyses, out_dir, args.min_abs_ref_net_pct) if args.reference_label else {"status": "no_reference_label"}
    out_json = out_dir / "signed_transition_signature_summary.json"
    out_md = out_dir / "signed_transition_signature_summary.md"
    summary = {
        "status": "SIGNED_TRANSITION_SIGNATURE_ANALYSIS",
        "created_utc": now_utc(),
        "elapsed_sec": None,
        "best_metric": args.best_metric,
        "trajectories": summaries,
        "reference_label": args.reference_label,
        "reference_comparison": reference_comparison,
        "out_json": safe_rel(out_json),
        "out_md": safe_rel(out_md),
        "note": "CPU/file-only; compares signed transitions over pre-peak->peak and peak->post-peak windows. No model inference, upload, or leaderboard submission.",
    }
    summary["elapsed_sec"] = round(time.time() - t0, 3)
    write_json(out_json, summary)
    write_md(summary, out_md)
    print(json.dumps({"status": summary["status"], "out_json": safe_rel(out_json), "out_md": safe_rel(out_md), "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
