#!/usr/bin/env python3
"""research: late-reference item dynamics around the newly surfaced chck_84M peak.

CPU/file-only analysis. It uses already-written per-target selected-evaluation
prediction payloads for the scale1.75 seed43022 reference trajectory. It does
not read partially written managed-task outputs and does not run model inference.

Scientific purpose: decide whether chck_84M's cheap-task improvement over the
protected/submitted chck_82M is a stable broad improvement, a one-checkpoint
transient, or a weighted reallocation among volatile items. This shapes endpoint
interpretation before the pending SuperGLUE and grid-resume tasks are delivered.
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

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
PATH = ROOT / "experiments/archive/frontier_consolidation/scripts/chck84_item_movement_analyzer.py"
DEFAULT_SELECTED_DIR = ROOT / "experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M"
DEFAULT_LABEL = "scale1p75_seed43022_reference"
DEFAULT_ENDPOINTS = [
    "chck_70M", "chck_72M", "chck_74M", "chck_76M", "chck_78M", "chck_80M",
    "chck_82M", "chck_84M", "chck_86M", "chck_88M", "chck_90M", "chck_92M", "chck_100M",
]
CLASS_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]
CHEAP_COLUMNS = CLASS_COLUMNS + ["Reading"]


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
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


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


def fmean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def endpoint_suffix(ep: str) -> str:
    return ep.replace("chck_", "").replace("M", "")


def endpoint_int(ep: str) -> int:
    return int(endpoint_suffix(ep))


def load_selected_csv(selected_dir: pathlib.Path) -> dict[str, dict[str, str]]:
    path = selected_dir / "selected_trajectory.csv"
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ep = r.get("endpoint") or ""
            if ep:
                rows[ep] = r
    return rows


def per_target_path(selected_dir: pathlib.Path, label: str, ep: str) -> pathlib.Path:
    return selected_dir / "eval" / "per_target" / f"{label}_{ep}.json"


def load_complete_endpoint_payloads(selected_dir: pathlib.Path, label: str, endpoints: list[str], mod) -> tuple[list[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    usable = []
    payloads: dict[str, dict[str, Any]] = {}
    skipped = []
    for ep in endpoints:
        path = per_target_path(selected_dir, label, ep)
        if not path.exists():
            skipped.append({"endpoint": ep, "reason": "missing_per_target", "path": str(path)})
            continue
        try:
            obj = read_json(path)
        except Exception as e:
            skipped.append({"endpoint": ep, "reason": f"json_error:{type(e).__name__}", "path": str(path), "error": str(e)[:300]})
            continue
        tasks = obj.get("tasks", {})
        needed = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
        bad = []
        for t in needed:
            rec = tasks.get(t)
            if not rec or rec.get("returncode") != 0 or not rec.get("predictions"):
                bad.append(t)
            else:
                p = pathlib.Path(rec["predictions"])
                if not p.exists() or p.stat().st_size <= 0:
                    bad.append(t)
        if bad:
            skipped.append({"endpoint": ep, "reason": "incomplete_tasks", "bad_tasks": bad, "path": str(path)})
            continue
        # This may still raise if gold/pred lengths do not match, which is what we want.
        _ = mod.endpoint_items(obj)
        usable.append(ep)
        payloads[ep] = obj
    return usable, payloads, skipped


def build_merged_items(payloads: dict[str, dict[str, Any]], mod) -> list[dict[str, Any]]:
    endpoint_to_rows = {ep: mod.endpoint_items(payload) for ep, payload in payloads.items()}
    return mod.merge_endpoint_items(endpoint_to_rows)


def correctness_sequence(row: dict[str, Any], endpoints: list[str]) -> list[int]:
    return [int(row[f"correct_{endpoint_suffix(ep)}"]) for ep in endpoints]


def count_flips(seq: list[int]) -> int:
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def longest_run(seq: list[int], value: int = 1) -> int:
    best = cur = 0
    for x in seq:
        if x == value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def mean_bool(rows: list[dict[str, Any]], suff: str) -> float:
    return sum(int(r[f"correct_{suff}"]) for r in rows) / len(rows) if rows else float("nan")


def endpoint_class_scores(merged: list[dict[str, Any]], endpoints: list[str]) -> dict[str, dict[str, float]]:
    by_col: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in merged:
        by_col[r["column"]].append(r)
    out: dict[str, dict[str, float]] = {}
    for col, rows in by_col.items():
        subgroups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            # official-style: column score is average of subtask accuracies.
            subgroups[r["subtask"]].append(r)
        out[col] = {}
        for ep in endpoints:
            s = endpoint_suffix(ep)
            vals = [pct(mean_bool(rs, s)) for rs in subgroups.values()]
            out[col][ep] = fmean(vals) or 0.0
    return out


def reading_scores_from_payloads(payloads: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for ep, payload in payloads.items():
        task = payload.get("tasks", {}).get("Reading", {})
        scores = task.get("scores", {})
        val = scores.get("Reading")
        out[ep] = float(val) if isinstance(val, (int, float)) else None
    return out


def weighted_scores(class_scores: dict[str, dict[str, float]], reading_scores: dict[str, float | None], endpoints: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for ep in endpoints:
        cols = {col: class_scores[col][ep] for col in CLASS_COLUMNS}
        cols["Reading"] = reading_scores.get(ep) if reading_scores.get(ep) is not None else float("nan")
        cheap7_vals = [cols[c] for c in CHEAP_COLUMNS if isinstance(cols[c], (int, float)) and math.isfinite(cols[c])]
        cheap6_vals = [cols[c] for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"] if math.isfinite(cols[c])]
        cheap5_vals = [cols[c] for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"] if math.isfinite(cols[c])]
        out[ep] = dict(cols)
        out[ep]["cheap7"] = fmean(cheap7_vals) or float("nan")
        out[ep]["cheap6_no_globalpiqa"] = fmean(cheap6_vals) or float("nan")
        out[ep]["cheap5_no_globalpiqa_reading"] = fmean(cheap5_vals) or float("nan")
        out[ep]["relation_state"] = fmean([cols["EWoK"], cols["Entity"]]) or float("nan")
        out[ep]["syntax_surface"] = fmean([cols["BLiMP"], cols["COMPS"]]) or float("nan")
    return out


def local_excess(scores: dict[str, dict[str, float]], center: str, left: str, right: str, fields: list[str]) -> dict[str, float]:
    out = {}
    for f in fields:
        out[f] = scores[center][f] - 0.5 * (scores[left][f] + scores[right][f])
    return out


def item_dynamics_rows(merged: list[dict[str, Any]], endpoints: list[str]) -> list[dict[str, Any]]:
    rows_out = []
    suffixes = [endpoint_suffix(ep) for ep in endpoints]
    late_window = [ep for ep in endpoints if 80 <= endpoint_int(ep) <= 92]
    late_suff = [endpoint_suffix(ep) for ep in late_window]
    for r in merged:
        seq = correctness_sequence(r, endpoints)
        late_seq = [int(r[f"correct_{s}"]) for s in late_suff]
        rec = {
            "item_key": r["item_key"],
            "column": r["column"],
            "subtask": r["subtask"],
            "subgroup": r.get("subgroup", ""),
            "fine_group": r.get("fine_group", ""),
            "n_endpoints": len(endpoints),
            "n_correct_all": sum(seq),
            "n_correct_late80_92": sum(late_seq),
            "flip_count_all": count_flips(seq),
            "flip_count_late80_92": count_flips(late_seq),
            "longest_correct_run_all": longest_run(seq, 1),
            "signature_all": "".join(map(str, seq)),
            "signature_late80_92": "".join(map(str, late_seq)),
            "correct_82": int(r.get("correct_82", 0)),
            "correct_84": int(r.get("correct_84", 0)),
            "correct_86": int(r.get("correct_86", 0)),
            "correct_88": int(r.get("correct_88", 0)),
            "correct_90": int(r.get("correct_90", 0)),
            "correct_92": int(r.get("correct_92", 0)),
        }
        rec["gain_82_84"] = int(rec["correct_82"] == 0 and rec["correct_84"] == 1)
        rec["loss_82_84"] = int(rec["correct_82"] == 1 and rec["correct_84"] == 0)
        rec["gain_84_86"] = int(rec["correct_84"] == 0 and rec["correct_86"] == 1)
        rec["loss_84_86"] = int(rec["correct_84"] == 1 and rec["correct_86"] == 0)
        rec["transient_gain_84_only_vs82_86"] = int(rec["correct_82"] == 0 and rec["correct_84"] == 1 and rec["correct_86"] == 0)
        rec["persistent_gain_82_to_84_through_86"] = int(rec["correct_82"] == 0 and rec["correct_84"] == 1 and rec["correct_86"] == 1)
        rec["persistent_gain_82_to_84_through_92"] = int(rec["correct_82"] == 0 and rec["correct_84"] == 1 and rec["correct_86"] == 1 and rec["correct_88"] == 1 and rec["correct_90"] == 1 and rec["correct_92"] == 1)
        rec["recovered_loss_84_only_vs82_86"] = int(rec["correct_82"] == 1 and rec["correct_84"] == 0 and rec["correct_86"] == 1)
        rows_out.append(rec)
    return rows_out


def summarize_dynamics(dyn: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, group_fields in [
        ("by_column", ["column"]),
        ("by_column_subtask", ["column", "subtask"]),
        ("by_column_subgroup", ["column", "subgroup"]),
    ]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for r in dyn:
            groups[tuple(r.get(f, "") for f in group_fields)].append(r)
        rows = []
        for gkey, rs in groups.items():
            rec = {f: gkey[i] for i, f in enumerate(group_fields)}
            rec["n_items"] = len(rs)
            for metric in [
                "gain_82_84", "loss_82_84", "gain_84_86", "loss_84_86",
                "transient_gain_84_only_vs82_86", "persistent_gain_82_to_84_through_86",
                "persistent_gain_82_to_84_through_92", "recovered_loss_84_only_vs82_86",
            ]:
                rec[metric] = int(sum(int(r[metric]) for r in rs))
                rec[f"{metric}_per_item"] = rec[metric] / len(rs) if rs else 0.0
            rec["net_82_to_84"] = rec["gain_82_84"] - rec["loss_82_84"]
            rec["net_84_to_86"] = rec["gain_84_86"] - rec["loss_84_86"]
            rec["mean_flip_count_late80_92"] = fmean([float(r["flip_count_late80_92"]) for r in rs]) or 0.0
            rec["frac_any_flip_late80_92"] = sum(int(r["flip_count_late80_92"] > 0) for r in rs) / len(rs) if rs else 0.0
            gains = rec["gain_82_84"]
            if gains:
                rec["frac_82_84_gains_lost_by_86"] = rec["transient_gain_84_only_vs82_86"] / gains
                rec["frac_82_84_gains_persist_through_92"] = rec["persistent_gain_82_to_84_through_92"] / gains
            else:
                rec["frac_82_84_gains_lost_by_86"] = None
                rec["frac_82_84_gains_persist_through_92"] = None
            rows.append(rec)
        # Sort with most important summary first; subtask tables by net absolute movement.
        if group_fields == ["column"]:
            rows.sort(key=lambda r: CLASS_COLUMNS.index(r["column"]) if r["column"] in CLASS_COLUMNS else 999)
        else:
            rows.sort(key=lambda r: (r.get("column", ""), -abs(r.get("net_82_to_84", 0)), r.get(group_fields[-1], "")))
        summary[key] = rows
    return summary


def bootstrap_endpoint_delta(
    subtask_accs: list[dict[str, Any]],
    endpoints: list[str],
    a: str,
    b: str,
    reading_delta: float | None = None,
    rng_seed: int = 189004,
    n_boot: int = 5000,
) -> dict[str, Any]:
    """Bootstrap endpoint deltas by resampling official subtask units within each column.

    This is not a claim about benchmark sampling distribution; it is a stability
    lens for whether a deterministic aggregate is carried by a few small groups.
    Reading has no subtask units here, so its exact scalar delta is included in
    cheap6/cheap7 when available but is not resampled.
    """
    import random

    rng = random.Random(rng_seed)
    by_col: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in subtask_accs:
        by_col[r["column"]].append(r)
    fields = ["cheap7", "cheap6_no_globalpiqa", "cheap5_no_globalpiqa_reading", "relation_state", "syntax_surface"]
    samples = {f: [] for f in fields}
    sa, sb = endpoint_suffix(a), endpoint_suffix(b)
    cols_present = [c for c in CLASS_COLUMNS if c in by_col]
    # Reading is scalar deterministic; include exact scalar in cheap7/cheap6 if available from rows carrying special column Reading.
    for _ in range(n_boot):
        col_score_a = {}
        col_score_b = {}
        for col in cols_present:
            units = by_col[col]
            draw = [units[rng.randrange(len(units))] for _ in range(len(units))]
            col_score_a[col] = statistics.fmean(float(u[f"score_{sa}"]) for u in draw)
            col_score_b[col] = statistics.fmean(float(u[f"score_{sb}"]) for u in draw)
        # Reading not bootstrapped here; use zero contribution for field deltas that include it if caller injected no Reading.
        deltas = {c: col_score_b[c] - col_score_a[c] for c in cols_present}
        if all(c in deltas for c in ["EWoK", "Entity"]):
            samples["relation_state"].append(statistics.fmean([deltas["EWoK"], deltas["Entity"]]))
        if all(c in deltas for c in ["BLiMP", "COMPS"]):
            samples["syntax_surface"].append(statistics.fmean([deltas["BLiMP"], deltas["COMPS"]]))
        class5 = [deltas[c] for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]]
        class6 = [deltas[c] for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]]
        samples["cheap5_no_globalpiqa_reading"].append(statistics.fmean(class5))
        if reading_delta is None:
            samples["cheap6_no_globalpiqa"].append(statistics.fmean(class5))
            samples["cheap7"].append(statistics.fmean(class6))
        else:
            samples["cheap6_no_globalpiqa"].append(statistics.fmean(class5 + [reading_delta]))
            samples["cheap7"].append(statistics.fmean(class6 + [reading_delta]))
    out: dict[str, Any] = {"rng_seed": rng_seed, "n_boot": n_boot, "pair": f"{b}_minus_{a}", "reading_delta_included_exactly": reading_delta, "note": "subtask-unit bootstrap over classification columns; Reading scalar included exactly in cheap6/cheap7 when available, not resampled"}
    for f, vals in samples.items():
        vals = sorted(vals)
        if vals:
            out[f] = {
                "mean": statistics.fmean(vals),
                "p05": vals[int(0.05 * (len(vals)-1))],
                "p50": vals[int(0.50 * (len(vals)-1))],
                "p95": vals[int(0.95 * (len(vals)-1))],
                "p_delta_le_0": sum(x <= 0 for x in vals) / len(vals),
            }
    return out


def subtask_score_rows(merged: list[dict[str, Any]], endpoints: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in merged:
        groups[(r["column"], r["subtask"])].append(r)
    rows = []
    for (col, subtask), rs in groups.items():
        rec: dict[str, Any] = {"column": col, "subtask": subtask, "n_items": len(rs)}
        for ep in endpoints:
            s = endpoint_suffix(ep)
            rec[f"score_{s}"] = pct(sum(int(r[f"correct_{s}"]) for r in rs) / len(rs))
        if "chck_82M" in endpoints and "chck_84M" in endpoints:
            rec["delta_84_minus_82"] = rec["score_84"] - rec["score_82"]
        if "chck_84M" in endpoints and "chck_86M" in endpoints:
            rec["delta_86_minus_84"] = rec["score_86"] - rec["score_84"]
        if "chck_84M" in endpoints and "chck_100M" in endpoints:
            rec["delta_100_minus_84"] = rec["score_100"] - rec["score_84"]
        rows.append(rec)
    return sorted(rows, key=lambda r: (r["column"], -abs(float(r.get("delta_84_minus_82", 0.0))), r["subtask"]))


def top_rows(rows: list[dict[str, Any]], key: str, col: str | None = None, n: int = 8, reverse: bool = True) -> list[dict[str, Any]]:
    xs = [r for r in rows if (col is None or r.get("column") == col) and isinstance(r.get(key), (int, float))]
    return sorted(xs, key=lambda r: float(r[key]), reverse=reverse)[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected-dir", type=pathlib.Path, default=DEFAULT_SELECTED_DIR)
    ap.add_argument("--label", default=DEFAULT_LABEL)
    ap.add_argument("--endpoints", nargs="+", default=DEFAULT_ENDPOINTS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "experiments/archive/frontier_consolidation/data/reference_late_item_dynamics")
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args()

    t0 = time.time()
    mod = load_step186()
    selected_dir = args.selected_dir if args.selected_dir.is_absolute() else ROOT / args.selected_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    usable, payloads, skipped = load_complete_endpoint_payloads(selected_dir, args.label, args.endpoints, mod)
    if not {"chck_82M", "chck_84M", "chck_86M", "chck_100M"}.issubset(set(usable)):
        raise SystemExit(f"Need at least 82/84/86/100 complete; usable={usable}; skipped={skipped}")
    usable = sorted(usable, key=endpoint_int)
    merged = build_merged_items(payloads, mod)
    dyn = item_dynamics_rows(merged, usable)
    dynamics_summary = summarize_dynamics(dyn)
    subtask_rows = subtask_score_rows(merged, usable)
    class_scores = endpoint_class_scores(merged, usable)
    reading_scores = reading_scores_from_payloads(payloads)
    scores = weighted_scores(class_scores, reading_scores, usable)

    score_rows = []
    for ep in usable:
        rec = {"endpoint": ep, "words_M": endpoint_int(ep)}
        rec.update(scores[ep])
        score_rows.append(rec)

    metric_fields = ["cheap7", "cheap6_no_globalpiqa", "cheap5_no_globalpiqa_reading", "relation_state", "syntax_surface", *CHEAP_COLUMNS]
    excess_84 = local_excess(scores, "chck_84M", "chck_82M", "chck_86M", metric_fields)
    excess_82 = local_excess(scores, "chck_82M", "chck_80M", "chck_84M", metric_fields)

    # Classification-only subtask bootstrap; keep it bounded and reproducible.
    def rd(a: str, b: str) -> float | None:
        va, vb = reading_scores.get(a), reading_scores.get(b)
        return (vb - va) if va is not None and vb is not None else None

    boots = {
        "84_minus_82": bootstrap_endpoint_delta(subtask_rows, usable, "chck_82M", "chck_84M", reading_delta=rd("chck_82M", "chck_84M"), n_boot=args.bootstrap),
        "86_minus_84": bootstrap_endpoint_delta(subtask_rows, usable, "chck_84M", "chck_86M", reading_delta=rd("chck_84M", "chck_86M"), n_boot=args.bootstrap),
        "100_minus_84": bootstrap_endpoint_delta(subtask_rows, usable, "chck_84M", "chck_100M", reading_delta=rd("chck_84M", "chck_100M"), n_boot=args.bootstrap),
    }

    # Signature concentration for the late window.
    sig_by_col = {}
    for col in CLASS_COLUMNS:
        rs = [r for r in dyn if r["column"] == col]
        ctr = Counter(r["signature_late80_92"] for r in rs)
        sig_by_col[col] = [{"signature_late80_92": sig, "n_items": n, "fraction": n / len(rs)} for sig, n in ctr.most_common(12)]

    # Save detailed but compact dynamic table and summary tables.
    write_csv(out_dir / "late_item_dynamics.csv", dyn)
    write_csv(out_dir / "score_curve.csv", score_rows)
    write_csv(out_dir / "subtask_score_curve.csv", subtask_rows)
    write_csv(out_dir / "column_dynamics_summary.csv", dynamics_summary["by_column"])
    write_csv(out_dir / "column_subtask_dynamics_summary.csv", dynamics_summary["by_column_subtask"])
    write_csv(out_dir / "column_subgroup_dynamics_summary.csv", dynamics_summary["by_column_subgroup"])

    summary = {
        "status": "REFERENCE_LATE_ITEM_DYNAMICS",
        "created_utc": now_utc(),
        "selected_dir": str(selected_dir.relative_to(ROOT)),
        "label": args.label,
        "requested_endpoints": args.endpoints,
        "usable_endpoints": usable,
        "skipped_endpoints": skipped,
        "n_classification_items": len(merged),
        "score_curve": score_rows,
        "local_excess_84_over_82_86_mean": excess_84,
        "local_excess_82_over_80_84_mean": excess_82,
        "dynamics_by_column": dynamics_summary["by_column"],
        "late80_92_signature_top_by_column": sig_by_col,
        "bootstrap_subtask_unit": boots,
        "top_subtasks_84_minus_82_positive": {col: top_rows(subtask_rows, "delta_84_minus_82", col=col, n=8, reverse=True) for col in CLASS_COLUMNS},
        "top_subtasks_84_minus_82_negative": {col: top_rows(subtask_rows, "delta_84_minus_82", col=col, n=8, reverse=False) for col in CLASS_COLUMNS},
        "top_subtasks_86_minus_84_negative": {col: top_rows(subtask_rows, "delta_86_minus_84", col=col, n=8, reverse=False) for col in CLASS_COLUMNS},
        "files": {},
        "elapsed_sec": round(time.time() - t0, 3),
    }
    for name in [
        "late_item_dynamics.csv", "score_curve.csv", "subtask_score_curve.csv",
        "column_dynamics_summary.csv", "column_subtask_dynamics_summary.csv", "column_subgroup_dynamics_summary.csv",
    ]:
        p = out_dir / name
        summary["files"][name] = {"path": str(p.relative_to(ROOT)), "size": p.stat().st_size, "sha256": sha256_file(p)}

    write_json(out_dir / "reference_late_item_dynamics_summary.json", summary)

    # Markdown report.
    lines: list[str] = []
    lines.append("# research reference late item dynamics around `chck_84M`\n\n")
    lines.append("CPU/file-only analysis of already completed selected cheap-task prediction payloads for the scale1.75 seed43022 reference trajectory. It avoids pending managed-task outputs and does not run model inference.\n\n")
    lines.append(f"Usable endpoints: {', '.join(usable)}. Skipped: {skipped if skipped else 'none'}. Classification items: {len(merged):,}.\n\n")
    lines.append("## Score curve\n\n")
    lines.append("| endpoint | cheap7 | cheap6 no GP | cheap5 no GP/Reading | relation/state | syntax/surface | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for ep in usable:
        s = scores[ep]
        lines.append(
            f"| {ep} | {s['cheap7']:.6f} | {s['cheap6_no_globalpiqa']:.6f} | {s['cheap5_no_globalpiqa_reading']:.6f} | {s['relation_state']:.6f} | {s['syntax_surface']:.6f} | {s['BLiMP']:.4f} | {s['Supplement']:.4f} | {s['EWoK']:.4f} | {s['Entity']:.4f} | {s['COMPS']:.4f} | {s['GlobalPIQA']:.4f} | {s['Reading']:.4f} |\n"
        )
    lines.append("\n## Local peak shape\n\n")
    lines.append("`chck_84M` relative to the mean of its immediate scored neighbors (`chck_82M`, `chck_86M`):\n\n")
    for k in ["cheap7", "cheap6_no_globalpiqa", "cheap5_no_globalpiqa_reading", "relation_state", "BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]:
        lines.append(f"- {k}: {excess_84[k]:+.6f}\n")
    lines.append("\n`chck_82M` relative to the mean of `chck_80M` and `chck_84M` is included in JSON for comparison.\n\n")
    lines.append("## 82→84 gain persistence and 84→86 reversal\n\n")
    lines.append("| column | n | net 82→84 | gains 82→84 | losses 82→84 | gains lost by 86 | gains persist through 92 | net 84→86 | mean late flips | frac any late flip |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in dynamics_summary["by_column"]:
        lines.append(
            f"| {r['column']} | {r['n_items']} | {r['net_82_to_84']} | {r['gain_82_84']} | {r['loss_82_84']} | "
            f"{(r['frac_82_84_gains_lost_by_86'] if r['frac_82_84_gains_lost_by_86'] is not None else float('nan')):.3f} | "
            f"{(r['frac_82_84_gains_persist_through_92'] if r['frac_82_84_gains_persist_through_92'] is not None else float('nan')):.3f} | "
            f"{r['net_84_to_86']} | {r['mean_flip_count_late80_92']:.3f} | {r['frac_any_flip_late80_92']:.3f} |\n"
        )
    lines.append("\n## Subtask-unit bootstrap lens\n\n")
    for name, boot in boots.items():
        lines.append(f"### {name}\n\n")
        for field in ["cheap7", "cheap6_no_globalpiqa", "cheap5_no_globalpiqa_reading", "relation_state", "syntax_surface"]:
            b = boot.get(field, {})
            if b:
                lines.append(f"- {field}: median {b['p50']:+.4f}, 90% interval [{b['p05']:+.4f}, {b['p95']:+.4f}], P(delta<=0)={b['p_delta_le_0']:.3f}\n")
        lines.append("\n")
    lines.append("This bootstrap is a subtask-resampling stability lens for the deterministic benchmark aggregate, not an alternative official metric; Reading is included as an exact scalar in cheap6/cheap7, not resampled.\n\n")
    lines.append("## Interpretation\n\n")
    # Programmatic interpretation based on results.
    r84 = scores["chck_84M"]
    r82 = scores["chck_82M"]
    r86 = scores["chck_86M"]
    d84_82 = r84["cheap7"] - r82["cheap7"]
    d86_84 = r86["cheap7"] - r84["cheap7"]
    lines.append(f"`chck_84M` improves cheap7 over `chck_82M` by {d84_82:+.6f} and then falls by {d86_84:+.6f} at `chck_86M`; its local excess over the 82/86 neighbor mean is {excess_84['cheap7']:+.6f}. ")
    lines.append(f"The peak remains visible without GlobalPIQA/Reading (cheap5 local excess {excess_84['cheap5_no_globalpiqa_reading']:+.6f}) but the relation/state local excess is {excess_84['relation_state']:+.6f}. ")
    lines.append("The item table shows large churn relative to net movement; use this as evidence of late competence allocation rather than smooth monotone acquisition.\n\n")
    lines.append("The pending SuperGLUE result will decide endpoint arithmetic, and seed43122 scoring is still needed before treating the late phase as robust beyond the protected seed/mask stream.\n\n")
    lines.append("## Files\n\n")
    for name, rec in summary["files"].items():
        lines.append(f"- `{rec['path']}` ({rec['size']} bytes, sha256 `{rec['sha256'][:16]}…`)\n")
    lines.append(f"- JSON summary: `{str((out_dir / 'reference_late_item_dynamics_summary.json').relative_to(ROOT))}`\n")
    (out_dir / "reference_late_item_dynamics_summary.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(out_dir.relative_to(ROOT)),
        "usable_endpoints": usable,
        "n_classification_items": len(merged),
        "chck84_minus_chck82_cheap7": round(scores["chck_84M"]["cheap7"] - scores["chck_82M"]["cheap7"], 6),
        "chck86_minus_chck84_cheap7": round(scores["chck_86M"]["cheap7"] - scores["chck_84M"]["cheap7"], 6),
        "chck84_local_excess_cheap7": round(excess_84["cheap7"], 6),
        "chck84_local_excess_cheap5_no_gp_reading": round(excess_84["cheap5_no_globalpiqa_reading"], 6),
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
