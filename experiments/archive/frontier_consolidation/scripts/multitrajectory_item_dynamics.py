#!/usr/bin/env python3
"""research: multi-trajectory item-dynamics analysis for DeBERTa selected grids.

CPU/file-only.  It consumes already-written selected_trajectory.json files and
per-target official-compatible prediction payloads.  It does not run model
inference and does not submit anything.

Scientific purpose: after the reference seed43022 trajectory exposed a narrow
late 84M competence-allocation phase, prepare the same item-level readout for
any completed matched trajectory (especially seed43122 when it delivers) and
measure whether alternative trajectories share the same item/family dynamics or
only score-level coincidences.
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
from collections import Counter, defaultdict
from typing import Any

CLASS_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]
CHEAP_COLUMNS = CLASS_COLUMNS + ["Reading"]
DERIVED = {
    "cheap6_no_GlobalPIQA": ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"],
    "cheap5_no_GlobalPIQA_Reading": ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"],
    "relation_state_mean": ["EWoK", "Entity"],
    "syntax_comp_mean": ["BLiMP", "Supplement", "COMPS"],
    "volatile_mean": ["GlobalPIQA", "Reading"],
}
EXPECTED_ENDPOINTS = [f"chck_{m}M" for m in range(70, 102, 2)]
NEEDED_TASKS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
PATH = ROOT / "experiments/archive/frontier_consolidation/scripts/chck84_item_movement_analyzer.py"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_step186():
    spec = importlib.util.spec_from_file_location("chck84_item_movement_analyzer", PATH)
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


def fmean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def pct(x: float) -> float:
    return 100.0 * x


def endpoint_int(ep: str) -> int:
    if not (ep.startswith("chck_") and ep.endswith("M")):
        raise ValueError(f"bad endpoint {ep!r}")
    return int(ep[5:-1])


def endpoint_words(ep: str) -> int:
    return endpoint_int(ep) * 1_000_000


def endpoint_suffix(ep: str) -> str:
    return str(endpoint_int(ep))


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def trajectory_path(selected_dir: pathlib.Path) -> pathlib.Path:
    return selected_dir / "selected_trajectory.json"


def per_target_path(selected_dir: pathlib.Path, label: str, ep: str) -> pathlib.Path:
    return selected_dir / "eval" / "per_target" / f"{label}_{ep}.json"


def read_trajectory_rows(selected_dir: pathlib.Path) -> list[dict[str, Any]]:
    rows = read_json(trajectory_path(selected_dir))
    if not isinstance(rows, list):
        raise TypeError(f"{trajectory_path(selected_dir)} is not a list")
    out = []
    for r in rows:
        if not isinstance(r, dict):
            raise TypeError(f"bad trajectory row {r!r}")
        rr = dict(r)
        for name, cols in DERIVED.items():
            if all(rr.get(c) is not None for c in cols):
                rr[name] = statistics.fmean(float(rr[c]) for c in cols)
        if rr.get("cheap7") is None and all(rr.get(c) is not None for c in CHEAP_COLUMNS):
            rr["cheap7"] = statistics.fmean(float(rr[c]) for c in CHEAP_COLUMNS)
        out.append(rr)
    return sorted(out, key=lambda r: int(r.get("words") or endpoint_words(str(r.get("endpoint")))))


def extract_payload_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    scores: dict[str, float | None] = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c, {}) if isinstance(tasks, dict) else {}
        scores[c] = float(rec["score"]) if isinstance(rec, dict) and rec.get("score") is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {}) if isinstance(tasks, dict) else {}
        if isinstance(rec, dict) and rec.get("score") is not None:
            gp.append(float(rec["score"]))
    scores["GlobalPIQA"] = statistics.fmean(gp) if len(gp) == 2 else None
    rec = tasks.get("Reading", {}) if isinstance(tasks, dict) else {}
    if isinstance(rec, dict) and isinstance(rec.get("scores"), dict) and rec["scores"].get("Reading") is not None:
        scores["Reading"] = float(rec["scores"]["Reading"])
    elif isinstance(rec, dict) and rec.get("score") is not None:
        scores["Reading"] = float(rec["score"])
    else:
        scores["Reading"] = None
    return scores


def payload_is_complete(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    tasks = payload.get("tasks", {}) if isinstance(payload.get("tasks"), dict) else {}
    bad: list[str] = []
    for name in NEEDED_TASKS:
        rec = tasks.get(name)
        if not isinstance(rec, dict):
            bad.append(f"missing:{name}")
            continue
        if rec.get("returncode") not in (None, 0):
            bad.append(f"returncode:{name}:{rec.get('returncode')}")
        pred = pathlib.Path(rec.get("predictions") or "")
        if not pred.exists() or pred.stat().st_size <= 0:
            bad.append(f"predictions:{name}")
        if name == "Reading":
            if not ((isinstance(rec.get("scores"), dict) and rec["scores"].get("Reading") is not None) or rec.get("score") is not None):
                bad.append("score:Reading")
        else:
            if rec.get("score") is None:
                bad.append(f"score:{name}")
    return not bad, bad


def load_complete_payloads(selected_dir: pathlib.Path, label: str, rows: list[dict[str, Any]], mod) -> tuple[list[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    usable: list[str] = []
    payloads: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    for row in rows:
        ep = row.get("endpoint")
        if not ep:
            continue
        p = per_target_path(selected_dir, label, ep)
        if not p.exists():
            skipped.append({"endpoint": ep, "reason": "missing_payload", "path": str(p.relative_to(ROOT) if p.is_absolute() else p)})
            continue
        try:
            payload = read_json(p)
            ok, bad = payload_is_complete(payload)
            if not ok:
                skipped.append({"endpoint": ep, "reason": "incomplete_payload", "bad": bad, "path": str(p.relative_to(ROOT) if p.is_absolute() else p)})
                continue
            # Force prediction/gold schema alignment now.  This can be expensive but catches stale artifacts.
            _ = mod.endpoint_items(payload)
        except Exception as e:
            skipped.append({"endpoint": ep, "reason": f"exception:{type(e).__name__}", "error": str(e)[:500], "path": str(p.relative_to(ROOT) if p.is_absolute() else p)})
            continue
        usable.append(ep)
        payloads[ep] = payload
    usable = sorted(usable, key=endpoint_int)
    return usable, payloads, skipped


def endpoint_items_by_endpoint(payloads: dict[str, dict[str, Any]], mod) -> dict[str, list[dict[str, Any]]]:
    return {ep: mod.endpoint_items(payload) for ep, payload in payloads.items()}


def merge_items(items_by_ep: dict[str, list[dict[str, Any]]], mod) -> list[dict[str, Any]]:
    return mod.merge_endpoint_items(items_by_ep)


def class_scores_from_merged(merged: list[dict[str, Any]], endpoints: list[str]) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in merged:
        groups[(r["column"], r["subtask"])].append(r)
    subtask_rows = []
    for (col, subtask), rs in sorted(groups.items()):
        rec: dict[str, Any] = {"column": col, "subtask": subtask, "n_items": len(rs)}
        for ep in endpoints:
            s = endpoint_suffix(ep)
            rec[f"score_{s}"] = pct(sum(int(r[f"correct_{s}"]) for r in rs) / len(rs))
        subtask_rows.append(rec)
    by_col: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in subtask_rows:
        by_col[r["column"]].append(r)
    class_scores: dict[str, dict[str, float]] = {}
    for col, rs in by_col.items():
        class_scores[col] = {}
        for ep in endpoints:
            s = endpoint_suffix(ep)
            class_scores[col][ep] = statistics.fmean(float(r[f"score_{s}"]) for r in rs)
    return class_scores, subtask_rows


def reading_scores(payloads: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    out = {}
    for ep, payload in payloads.items():
        task = payload.get("tasks", {}).get("Reading", {})
        val = None
        if isinstance(task, dict) and isinstance(task.get("scores"), dict):
            val = task["scores"].get("Reading")
        elif isinstance(task, dict):
            val = task.get("score")
        out[ep] = float(val) if isinstance(val, (int, float)) else None
    return out


def metric_scores(class_scores: dict[str, dict[str, float]], rd: dict[str, float | None], endpoints: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for ep in endpoints:
        rec = {c: class_scores[c][ep] for c in CLASS_COLUMNS if c in class_scores}
        rec["Reading"] = float(rd[ep]) if rd.get(ep) is not None else float("nan")
        for name, cols in DERIVED.items():
            vals = [rec[c] for c in cols if c in rec and math.isfinite(rec[c])]
            rec[name] = statistics.fmean(vals) if len(vals) == len(cols) else float("nan")
        rec["cheap7"] = statistics.fmean(rec[c] for c in CHEAP_COLUMNS if c in rec and math.isfinite(rec[c]))
        out[ep] = rec
    return out


def trajectory_alignment(trajectory_rows: list[dict[str, Any]], scores: dict[str, dict[str, float]], usable: list[str]) -> dict[str, Any]:
    """Compare item-recomputed scores with selected_trajectory score rows.

    The selected trajectory carries the endpoint arithmetic used elsewhere, while this
    script recomputes item-level official-like accuracies from prediction payloads to
    support churn/overlap analysis.  Small differences can occur from rounded payload
    task scores or tie-breaking; make them explicit.
    """
    by_ep = {r.get("endpoint"): r for r in trajectory_rows}
    records: list[dict[str, Any]] = []
    max_abs_by_metric: dict[str, float] = {m: 0.0 for m in ["cheap7", *CHEAP_COLUMNS]}
    for ep in usable:
        row = by_ep.get(ep, {})
        rec: dict[str, Any] = {"endpoint": ep}
        for m in ["cheap7", *CHEAP_COLUMNS]:
            a = row.get(m)
            b = scores.get(ep, {}).get(m)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(float(b)):
                diff = float(b) - float(a)
                rec[f"selected_{m}"] = float(a)
                rec[f"item_recomputed_{m}"] = float(b)
                rec[f"diff_item_minus_selected_{m}"] = diff
                max_abs_by_metric[m] = max(max_abs_by_metric[m], abs(diff))
        records.append(rec)
    return {
        "note": "Endpoint arithmetic should use selected_trajectory/payload task scores; item-recomputed scores support item/subtask dynamics and may differ at small rounding/tie scales.",
        "max_abs_diff_item_minus_selected": max_abs_by_metric,
        "records": records,
    }


def contiguous_band(endpoints: list[str], scores: dict[str, dict[str, float]], metric: str, drop: float) -> dict[str, Any]:
    vals = [(ep, scores[ep].get(metric)) for ep in endpoints if scores.get(ep, {}).get(metric) is not None]
    if not vals:
        return {"endpoints": [], "span_m": None}
    best_ep, best_val = max(vals, key=lambda x: x[1])
    idx_best = endpoints.index(best_ep)
    good = {ep for ep, val in vals if float(val) >= float(best_val) - drop}
    s = idx_best
    while s - 1 >= 0 and endpoints[s - 1] in good:
        s -= 1
    e = idx_best
    while e + 1 < len(endpoints) and endpoints[e + 1] in good:
        e += 1
    eps = endpoints[s:e+1]
    return {"endpoints": eps, "span_m": (endpoint_int(eps[-1]) - endpoint_int(eps[0])) if eps else None, "best_value": float(best_val), "drop": drop}


def local_excess(endpoints: list[str], scores: dict[str, dict[str, float]], center: str, fields: list[str]) -> dict[str, float | None]:
    if center not in endpoints:
        return {f: None for f in fields}
    i = endpoints.index(center)
    if i <= 0 or i >= len(endpoints) - 1:
        return {f: None for f in fields}
    left, right = endpoints[i - 1], endpoints[i + 1]
    return {f: float(scores[center][f] - 0.5 * (scores[left][f] + scores[right][f])) if all(math.isfinite(scores[e].get(f, float("nan"))) for e in [left, center, right]) else None for f in fields}


def column_peaks(endpoints: list[str], scores: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    out = {}
    for col in CHEAP_COLUMNS + list(DERIVED):
        vals = [(ep, scores[ep].get(col)) for ep in endpoints if scores.get(ep, {}).get(col) is not None and math.isfinite(scores[ep][col])]
        if vals:
            ep, val = max(vals, key=lambda x: x[1])
            out[col] = {"endpoint": ep, "words": endpoint_words(ep), "score": float(val)}
    return out


def correctness(row: dict[str, Any], ep: str) -> int:
    return int(row[f"correct_{endpoint_suffix(ep)}"])


def dynamics_around(merged: list[dict[str, Any]], endpoints: list[str], best_ep: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    idx = endpoints.index(best_ep)
    prev_ep = endpoints[idx - 1] if idx > 0 else None
    next_ep = endpoints[idx + 1] if idx + 1 < len(endpoints) else None
    final_ep = endpoints[-1] if endpoints else None
    dyn_rows: list[dict[str, Any]] = []
    for r in merged:
        rec: dict[str, Any] = {
            "item_key": r["item_key"],
            "column": r["column"],
            "subtask": r["subtask"],
            "subgroup": r.get("subgroup", ""),
            "fine_group": r.get("fine_group", ""),
            "correct_best": correctness(r, best_ep),
            "best_endpoint": best_ep,
            "flip_count_all": sum(correctness(r, a) != correctness(r, b) for a, b in zip(endpoints, endpoints[1:])),
        }
        if prev_ep:
            rec["prev_endpoint"] = prev_ep
            rec["correct_prev"] = correctness(r, prev_ep)
            rec["gain_prev_to_best"] = int(rec["correct_prev"] == 0 and rec["correct_best"] == 1)
            rec["loss_prev_to_best"] = int(rec["correct_prev"] == 1 and rec["correct_best"] == 0)
        if next_ep:
            rec["next_endpoint"] = next_ep
            rec["correct_next"] = correctness(r, next_ep)
            rec["gain_best_to_next"] = int(rec["correct_best"] == 0 and rec["correct_next"] == 1)
            rec["loss_best_to_next"] = int(rec["correct_best"] == 1 and rec["correct_next"] == 0)
        if prev_ep and next_ep:
            rec["transient_gain_best_only_vs_neighbors"] = int(rec["correct_prev"] == 0 and rec["correct_best"] == 1 and rec["correct_next"] == 0)
            rec["recovered_loss_best_only_vs_neighbors"] = int(rec["correct_prev"] == 1 and rec["correct_best"] == 0 and rec["correct_next"] == 1)
            rec["persistent_gain_prev_to_best_through_next"] = int(rec["correct_prev"] == 0 and rec["correct_best"] == 1 and rec["correct_next"] == 1)
        if final_ep and final_ep != best_ep:
            rec["final_endpoint"] = final_ep
            rec["correct_final"] = correctness(r, final_ep)
            rec["gain_best_to_final"] = int(rec["correct_best"] == 0 and rec["correct_final"] == 1)
            rec["loss_best_to_final"] = int(rec["correct_best"] == 1 and rec["correct_final"] == 0)
        dyn_rows.append(rec)
    group_rows = summarize_dyn_groups(dyn_rows, ["column"], include_subtask=False)
    return dyn_rows, group_rows


def summarize_dyn_groups(dyn_rows: list[dict[str, Any]], group_fields: list[str], include_subtask: bool = True) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in dyn_rows:
        groups[tuple(r.get(f, "") for f in group_fields)].append(r)
    rows = []
    count_fields = [
        "gain_prev_to_best", "loss_prev_to_best", "gain_best_to_next", "loss_best_to_next",
        "transient_gain_best_only_vs_neighbors", "persistent_gain_prev_to_best_through_next",
        "recovered_loss_best_only_vs_neighbors", "gain_best_to_final", "loss_best_to_final",
    ]
    for key, rs in groups.items():
        rec = {f: key[i] for i, f in enumerate(group_fields)}
        rec["n_items"] = len(rs)
        for f in count_fields:
            if any(f in r for r in rs):
                val = sum(int(r.get(f, 0)) for r in rs)
                rec[f] = val
                rec[f"{f}_per_item"] = val / len(rs) if rs else 0.0
        if "gain_prev_to_best" in rec and "loss_prev_to_best" in rec:
            rec["net_prev_to_best"] = rec["gain_prev_to_best"] - rec["loss_prev_to_best"]
        if "gain_best_to_next" in rec and "loss_best_to_next" in rec:
            rec["net_best_to_next"] = rec["gain_best_to_next"] - rec["loss_best_to_next"]
        if "gain_best_to_final" in rec and "loss_best_to_final" in rec:
            rec["net_best_to_final"] = rec["gain_best_to_final"] - rec["loss_best_to_final"]
        rec["mean_flip_count_all"] = statistics.fmean(float(r["flip_count_all"]) for r in rs)
        rec["frac_any_flip_all"] = sum(int(r["flip_count_all"] > 0) for r in rs) / len(rs) if rs else 0.0
        rows.append(rec)
    if group_fields == ["column"]:
        rows.sort(key=lambda r: CLASS_COLUMNS.index(r["column"]) if r["column"] in CLASS_COLUMNS else 999)
    else:
        rows.sort(key=lambda r: (r.get("column", ""), -abs(r.get("net_prev_to_best", 0)), r.get(group_fields[-1], "")))
    return rows


def bootstrap_delta(subtask_rows: list[dict[str, Any]], endpoints: list[str], a: str, b: str, reading_delta: float | None, n_boot: int, rng_seed: int) -> dict[str, Any]:
    import random
    by_col: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in subtask_rows:
        by_col[r["column"]].append(r)
    fields = ["cheap7", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "relation_state_mean", "syntax_comp_mean"]
    samples = {f: [] for f in fields}
    sa, sb = endpoint_suffix(a), endpoint_suffix(b)
    rng = random.Random(rng_seed)
    cols_present = [c for c in CLASS_COLUMNS if c in by_col]
    if n_boot <= 0:
        return {"pair": f"{b}_minus_{a}", "n_boot": 0, "note": "bootstrap disabled"}
    for _ in range(n_boot):
        deltas: dict[str, float] = {}
        for col in cols_present:
            units = by_col[col]
            draw = [units[rng.randrange(len(units))] for _ in range(len(units))]
            deltas[col] = statistics.fmean(float(u[f"score_{sb}"]) - float(u[f"score_{sa}"]) for u in draw)
        if all(c in deltas for c in ["EWoK", "Entity"]):
            samples["relation_state_mean"].append(statistics.fmean([deltas["EWoK"], deltas["Entity"]]))
        if all(c in deltas for c in ["BLiMP", "Supplement", "COMPS"]):
            samples["syntax_comp_mean"].append(statistics.fmean([deltas["BLiMP"], deltas["Supplement"], deltas["COMPS"]]))
        class5 = [deltas[c] for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"] if c in deltas]
        samples["cheap5_no_GlobalPIQA_Reading"].append(statistics.fmean(class5))
        cheap6 = class5 + ([reading_delta] if reading_delta is not None else [])
        cheap7 = class5 + ([deltas["GlobalPIQA"]] if "GlobalPIQA" in deltas else []) + ([reading_delta] if reading_delta is not None else [])
        samples["cheap6_no_GlobalPIQA"].append(statistics.fmean(cheap6))
        samples["cheap7"].append(statistics.fmean(cheap7))
    out: dict[str, Any] = {"pair": f"{b}_minus_{a}", "rng_seed": rng_seed, "n_boot": n_boot, "reading_delta_included_exactly": reading_delta}
    for f, vals in samples.items():
        vals = sorted(vals)
        if vals:
            out[f] = {
                "mean": statistics.fmean(vals),
                "p05": vals[int(0.05 * (len(vals) - 1))],
                "p50": vals[int(0.50 * (len(vals) - 1))],
                "p95": vals[int(0.95 * (len(vals) - 1))],
                "p_delta_le_0": sum(x <= 0 for x in vals) / len(vals),
            }
    return out


def analyze_one(label: str, selected_dir: pathlib.Path, out_dir: pathlib.Path, mod, high_band_drop: float, n_boot: int, keep_item_csv: bool) -> dict[str, Any]:
    rows = read_trajectory_rows(selected_dir)
    usable, payloads, skipped = load_complete_payloads(selected_dir, label, rows, mod)
    if not usable:
        raise RuntimeError(f"{label}: no complete endpoints; skipped={skipped}")
    items_by_ep = endpoint_items_by_endpoint(payloads, mod)
    merged = merge_items(items_by_ep, mod)
    class_scores, subtask_rows = class_scores_from_merged(merged, usable)
    rd = reading_scores(payloads)
    scores = metric_scores(class_scores, rd, usable)
    align = trajectory_alignment(rows, scores, usable)
    best_ep = max(usable, key=lambda ep: scores[ep]["cheap7"])
    local = local_excess(usable, scores, best_ep, ["cheap7", *CHEAP_COLUMNS, *DERIVED.keys()])
    dyn_rows, dyn_by_col = dynamics_around(merged, usable, best_ep)
    dyn_by_subtask = summarize_dyn_groups(dyn_rows, ["column", "subtask"])
    peaks = column_peaks(usable, scores)
    score_rows = [{"endpoint": ep, "words_M": endpoint_int(ep), **scores[ep]} for ep in usable]

    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    write_csv(label_dir / "score_curve.csv", score_rows)
    write_csv(label_dir / "subtask_score_curve.csv", subtask_rows)
    write_csv(label_dir / "column_dynamics_around_best.csv", dyn_by_col)
    write_csv(label_dir / "subtask_dynamics_around_best.csv", dyn_by_subtask)
    if keep_item_csv:
        write_csv(label_dir / "item_dynamics_around_best.csv", dyn_rows)

    # Lightweight late-window signature concentration around 80--100M for item churn comparison.
    late_eps = [ep for ep in usable if 80 <= endpoint_int(ep) <= 100]
    sig_by_col = {}
    for col in CLASS_COLUMNS:
        rs = [r for r in merged if r["column"] == col]
        ctr = Counter("".join(str(correctness(r, ep)) for ep in late_eps) for r in rs)
        sig_by_col[col] = [{"signature": sig, "n_items": n, "fraction": n / len(rs)} for sig, n in ctr.most_common(10)]

    def rd_delta(a: str, b: str) -> float | None:
        va, vb = rd.get(a), rd.get(b)
        return float(vb - va) if va is not None and vb is not None else None

    boots = {}
    idx = usable.index(best_ep)
    if idx > 0:
        boots["best_minus_prev"] = bootstrap_delta(subtask_rows, usable, usable[idx - 1], best_ep, rd_delta(usable[idx - 1], best_ep), n_boot, 196000 + endpoint_int(best_ep))
    if idx + 1 < len(usable):
        boots["next_minus_best"] = bootstrap_delta(subtask_rows, usable, best_ep, usable[idx + 1], rd_delta(best_ep, usable[idx + 1]), n_boot, 196100 + endpoint_int(best_ep))
    if usable[-1] != best_ep:
        boots["final_minus_best"] = bootstrap_delta(subtask_rows, usable, best_ep, usable[-1], rd_delta(best_ep, usable[-1]), n_boot, 196200 + endpoint_int(best_ep))

    summary = {
        "label": label,
        "selected_dir": str(selected_dir.relative_to(ROOT) if selected_dir.is_absolute() else selected_dir),
        "status": "ok" if len(usable) == len(rows) and not skipped else "partial",
        "trajectory_n_rows": len(rows),
        "usable_endpoints": usable,
        "skipped_endpoints": skipped,
        "n_classification_items": len(merged),
        "best_endpoint": best_ep,
        "best_words": endpoint_words(best_ep),
        "best_cheap7": scores[best_ep]["cheap7"],
        "final_endpoint": usable[-1],
        "final_cheap7": scores[usable[-1]]["cheap7"],
        "final_minus_best_cheap7": scores[usable[-1]]["cheap7"] - scores[best_ep]["cheap7"],
        "chck_82M_cheap7": scores.get("chck_82M", {}).get("cheap7"),
        "chck_84M_cheap7": scores.get("chck_84M", {}).get("cheap7"),
        "chck_100M_cheap7": scores.get("chck_100M", {}).get("cheap7"),
        "best_local_excess": local,
        "connected_near_best_band": contiguous_band(usable, scores, "cheap7", high_band_drop),
        "column_peaks": peaks,
        "cheap_column_peak_spread_m": (max(v["words"] for k, v in peaks.items() if k in CHEAP_COLUMNS) - min(v["words"] for k, v in peaks.items() if k in CHEAP_COLUMNS)) / 1_000_000 if any(k in CHEAP_COLUMNS for k in peaks) else None,
        "score_curve": score_rows,
        "trajectory_score_alignment": align,
        "dynamics_by_column": dyn_by_col,
        "late_signature_top_by_column": sig_by_col,
        "bootstrap_subtask_unit": boots,
        "files": {},
    }
    for name in ["score_curve.csv", "subtask_score_curve.csv", "column_dynamics_around_best.csv", "subtask_dynamics_around_best.csv"] + (["item_dynamics_around_best.csv"] if keep_item_csv else []):
        p = label_dir / name
        summary["files"][name] = {"path": str(p.relative_to(ROOT)), "size": p.stat().st_size, "sha256": sha256_file(p)}
    return {"summary": summary, "merged": merged, "scores": scores, "endpoints": usable}


def compare_items(reference_label: str, analyses: dict[str, dict[str, Any]], out_dir: pathlib.Path) -> list[dict[str, Any]]:
    if reference_label not in analyses:
        return []
    ref = analyses[reference_label]
    ref_by_key = {r["item_key"]: r for r in ref["merged"]}
    rows_out: list[dict[str, Any]] = []
    for label, an in analyses.items():
        if label == reference_label:
            continue
        oth_by_key = {r["item_key"]: r for r in an["merged"]}
        common_keys = sorted(set(ref_by_key) & set(oth_by_key))
        common_eps = [ep for ep in EXPECTED_ENDPOINTS if ep in ref["endpoints"] and ep in an["endpoints"]]
        # Common endpoint comparisons by column.
        for ep in common_eps:
            s = endpoint_suffix(ep)
            groups: dict[str, list[str]] = defaultdict(list)
            for key in common_keys:
                groups[ref_by_key[key]["column"]].append(key)
            for col, keys in sorted(groups.items(), key=lambda kv: CLASS_COLUMNS.index(kv[0]) if kv[0] in CLASS_COLUMNS else 999):
                both = ref_only = oth_only = both_wrong = 0
                for key in keys:
                    rc = int(ref_by_key[key][f"correct_{s}"])
                    oc = int(oth_by_key[key][f"correct_{s}"])
                    if rc and oc:
                        both += 1
                    elif rc and not oc:
                        ref_only += 1
                    elif oc and not rc:
                        oth_only += 1
                    else:
                        both_wrong += 1
                n = len(keys)
                rows_out.append({
                    "other_label": label,
                    "comparison_kind": "same_endpoint",
                    "endpoint_reference": ep,
                    "endpoint_other": ep,
                    "column": col,
                    "n_items": n,
                    "both_correct": both,
                    "reference_only_correct": ref_only,
                    "other_only_correct": oth_only,
                    "both_wrong": both_wrong,
                    "net_other_minus_reference_items": oth_only - ref_only,
                    "net_other_minus_reference_pct_items": pct((oth_only - ref_only) / n) if n else 0.0,
                    "disagreement_fraction": (ref_only + oth_only) / n if n else 0.0,
                    "jaccard_correct_sets": both / (both + ref_only + oth_only) if (both + ref_only + oth_only) else None,
                })
        # Best-vs-best comparison uses each trajectory's own aggregate best endpoint.
        ref_best = ref["summary"]["best_endpoint"]
        oth_best = an["summary"]["best_endpoint"]
        rs, os = endpoint_suffix(ref_best), endpoint_suffix(oth_best)
        for col in CLASS_COLUMNS:
            keys = [k for k in common_keys if ref_by_key[k]["column"] == col]
            both = ref_only = oth_only = both_wrong = 0
            for key in keys:
                rc = int(ref_by_key[key][f"correct_{rs}"])
                oc = int(oth_by_key[key][f"correct_{os}"])
                if rc and oc:
                    both += 1
                elif rc and not oc:
                    ref_only += 1
                elif oc and not rc:
                    oth_only += 1
                else:
                    both_wrong += 1
            n = len(keys)
            rows_out.append({
                "other_label": label,
                "comparison_kind": "best_vs_best",
                "endpoint_reference": ref_best,
                "endpoint_other": oth_best,
                "column": col,
                "n_items": n,
                "both_correct": both,
                "reference_only_correct": ref_only,
                "other_only_correct": oth_only,
                "both_wrong": both_wrong,
                "net_other_minus_reference_items": oth_only - ref_only,
                "net_other_minus_reference_pct_items": pct((oth_only - ref_only) / n) if n else 0.0,
                "disagreement_fraction": (ref_only + oth_only) / n if n else 0.0,
                "jaccard_correct_sets": both / (both + ref_only + oth_only) if (both + ref_only + oth_only) else None,
            })
    write_csv(out_dir / "pairwise_item_overlap_vs_reference.csv", rows_out)
    return rows_out


def write_md(summary: dict[str, Any], path: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append("# research multi-trajectory DeBERTa item dynamics\n\n")
    lines.append("CPU/file-only analysis of already written selected-grid prediction payloads. It does not run model inference.\n\n")
    lines.append("## Trajectories\n\n")
    lines.append("| label | status | usable | best | best cheap7 | 82M | 84M | 100M | final-best Δ | near-best band | local excess cheap7 | peak spread M |\n")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|\n")
    for label, s in summary["trajectories"].items():
        band = s.get("connected_near_best_band", {})
        lines.append(f"| {label} | {s.get('status')} | {len(s.get('usable_endpoints', []))} | {s.get('best_endpoint')} | {fmt(s.get('best_cheap7'))} | {fmt(s.get('chck_82M_cheap7'))} | {fmt(s.get('chck_84M_cheap7'))} | {fmt(s.get('chck_100M_cheap7'))} | {fmt(s.get('final_minus_best_cheap7'))} | {','.join(band.get('endpoints', []))} | {fmt(s.get('best_local_excess', {}).get('cheap7'))} | {fmt(s.get('cheap_column_peak_spread_m'))} |\n")
    lines.append("\n## Column peak vectors\n\n")
    lines.append("| label | " + " | ".join(CHEAP_COLUMNS) + " |\n")
    lines.append("|---|" + "|".join(["---:"] * len(CHEAP_COLUMNS)) + "|\n")
    for label, s in summary["trajectories"].items():
        vals = []
        for c in CHEAP_COLUMNS:
            rec = s.get("column_peaks", {}).get(c, {})
            vals.append(str(int(rec.get("words", 0) / 1_000_000)) if rec else "")
        lines.append(f"| {label} | " + " | ".join(vals) + " |\n")
    lines.append("\n## Dynamics around each aggregate best endpoint\n\n")
    for label, s in summary["trajectories"].items():
        lines.append(f"### {label}\n\n")
        align = s.get('trajectory_score_alignment', {}).get('max_abs_diff_item_minus_selected', {})
        lines.append(f"Best endpoint: `{s.get('best_endpoint')}`.  Local excess cheap7: {fmt(s.get('best_local_excess', {}).get('cheap7'))}; relation/state: {fmt(s.get('best_local_excess', {}).get('relation_state_mean'))}; cheap5 no GP/Reading: {fmt(s.get('best_local_excess', {}).get('cheap5_no_GlobalPIQA_Reading'))}. Max |item-recomputed minus selected| cheap7: {fmt(align.get('cheap7'))}.\n\n")
        lines.append("| column | n | net prev→best | gains prev→best | losses prev→best | net best→next | gain-best lost at next | net best→final | mean flips |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in s.get("dynamics_by_column", []):
            lines.append(f"| {r.get('column')} | {r.get('n_items')} | {r.get('net_prev_to_best','')} | {r.get('gain_prev_to_best','')} | {r.get('loss_prev_to_best','')} | {r.get('net_best_to_next','')} | {r.get('transient_gain_best_only_vs_neighbors','')} | {r.get('net_best_to_final','')} | {fmt(r.get('mean_flip_count_all'))} |\n")
        lines.append("\n")
    if summary.get("pairwise_item_overlap_vs_reference_summary"):
        lines.append("## Pairwise item-overlap vs reference\n\n")
        lines.append("| other | kind | ref endpoint | other endpoint | column | net other-ref items | net pct | disagreement | correct-set Jaccard |\n")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|\n")
        for r in summary["pairwise_item_overlap_vs_reference_summary"]:
            lines.append(f"| {r.get('other_label')} | {r.get('comparison_kind')} | {r.get('endpoint_reference')} | {r.get('endpoint_other')} | {r.get('column')} | {r.get('net_other_minus_reference_items')} | {fmt(r.get('net_other_minus_reference_pct_items'))} | {fmt(r.get('disagreement_fraction'))} | {fmt(r.get('jaccard_correct_sets'))} |\n")
    lines.append("\n## Reading the result\n\n")
    lines.append("Endpoint arithmetic should still use the selected_trajectory/payload task scores.  This tool recomputes item-level official-like accuracies from prediction files to study churn and overlap, so small rounding/tie differences are reported explicitly in each trajectory's `trajectory_score_alignment` block.\n\n")
    lines.append("This analysis is designed to be rerun on seed43122 after its selected-grid scoring delivers.  Score-level reproduction is not enough: a robust late-phase result should show related column peak structure and item/family dynamics, not only a single volatile endpoint.\n\n")
    lines.append(f"JSON: `{summary['out_json']}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", action="append", required=True, help="LABEL=SELECTED_DIR")
    ap.add_argument("--reference-label", default=None)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--high-band-drop", type=float, default=0.2)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--keep-item-csv", action="store_true", help="write per-item dynamics CSVs; large but useful for inspection")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = load_step186()

    analyses: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for spec in args.trajectory:
        if "=" not in spec:
            raise SystemExit(f"Use LABEL=SELECTED_DIR for --trajectory, got {spec!r}")
        label, d = spec.split("=", 1)
        selected_dir = pathlib.Path(d)
        if not selected_dir.is_absolute():
            selected_dir = ROOT / selected_dir
        result = analyze_one(label, selected_dir, out_dir, mod, args.high_band_drop, args.bootstrap, args.keep_item_csv)
        analyses[label] = result
        summaries[label] = result["summary"]

    pairwise_rows = []
    pairwise_summary = []
    if args.reference_label:
        pairwise_rows = compare_items(args.reference_label, analyses, out_dir)
        # Keep the markdown compact: show best-vs-best and common 84M/86M/100M rows, plus rows with largest abs net pct.
        chosen = [r for r in pairwise_rows if r.get("comparison_kind") == "best_vs_best"]
        for ep in ["chck_84M", "chck_86M", "chck_100M"]:
            chosen += [r for r in pairwise_rows if r.get("comparison_kind") == "same_endpoint" and r.get("endpoint_reference") == ep]
        chosen += sorted(pairwise_rows, key=lambda r: abs(float(r.get("net_other_minus_reference_pct_items") or 0.0)), reverse=True)[:12]
        seen = set()
        for r in chosen:
            key = (r.get("other_label"), r.get("comparison_kind"), r.get("endpoint_reference"), r.get("endpoint_other"), r.get("column"))
            if key not in seen:
                seen.add(key)
                pairwise_summary.append(r)

    out_json = out_dir / "multitrajectory_item_dynamics_summary.json"
    out_md = out_dir / "multitrajectory_item_dynamics_summary.md"
    summary = {
        "status": "MULTITRAJECTORY_ITEM_DYNAMICS",
        "created_utc": now_utc(),
        "elapsed_sec": None,
        "trajectories": summaries,
        "reference_label": args.reference_label,
        "pairwise_item_overlap_path": str((out_dir / "pairwise_item_overlap_vs_reference.csv").relative_to(ROOT)) if pairwise_rows else None,
        "pairwise_item_overlap_vs_reference_summary": pairwise_summary,
        "out_json": str(out_json.relative_to(ROOT)),
        "out_md": str(out_md.relative_to(ROOT)),
        "note": "CPU/file-only analysis over existing selected-grid prediction payloads; no model inference, upload, or leaderboard submission.",
    }
    summary["elapsed_sec"] = round(time.time() - t0, 3)
    write_json(out_json, summary)
    write_md(summary, out_md)
    print(json.dumps({"status": summary["status"], "out_json": str(out_json.relative_to(ROOT)), "out_md": str(out_md.relative_to(ROOT)), "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
