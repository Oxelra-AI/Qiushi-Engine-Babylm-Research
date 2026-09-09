#!/usr/bin/env python3
"""research frozen-affine scalar tests from saved candidate-event outputs.

A saved run with no held h0/h2 state anchors supplies a frozen scalar coordinate
for each candidate-event pair.  We fit a one-dimensional affine decision rule
z = a d + b from direct h0/h2 evaluation anchor rows only, then ask whether the
same z predicts unanchored h1/h3 graph-transfer state rows.  This measures
representational sufficiency of a scalar orientation coordinate; it is not a
claim that the model learned the affine map during training.

The script is robust to both research and research saved-output schemas.  It uses
state eval rows and computes d = score(candidate0)-score(candidate1) when the
stored d_e field is absent.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def choice_rows_from_state_predictions(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    by: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r.get("condition"), r.get("arm"), r.get("seed"), r.get("bridge_sign", r.get("source_bridge_sign")), r.get("suite"), r.get("query_key"))].append(r)
    out: List[Dict[str, Any]] = []
    for key, rs in by.items():
        if len(rs) != 2:
            errors.append(f"bad group {key}: {len(rs)} rows")
            continue
        rs = sorted(rs, key=lambda r: int(r.get("candidate_index", 0)))
        if bool(rs[0].get("label_true")) == bool(rs[1].get("label_true")):
            errors.append(f"bad labels {key}: {rs[0].get('label_true')},{rs[1].get('label_true')}")
            continue
        if "d_e" in rs[0]:
            d = float(rs[0]["d_e"])
        else:
            d = float(rs[0].get("score")) - float(rs[1].get("score"))
        y = 1 if bool(rs[0].get("label_true")) else -1
        raw_pred = 1 if d >= 0.0 else -1
        r0 = rs[0]
        out.append({
            "condition": key[0], "arm": key[1], "seed": key[2], "source_bridge_sign": key[3],
            "suite": key[4], "query_key": key[5], "d": d, "y": y,
            "raw_correct": int(raw_pred == y),
            "is_changed": bool(r0.get("is_changed")),
            "relation": r0.get("relation"), "relation_family": r0.get("relation_family"),
            "initial_pattern": r0.get("initial_pattern"), "static_slot": r0.get("static_slot"),
            "is_direct_anchor": bool(r0.get("is_direct_anchor")),
            "global_swap_changes_label": bool(r0.get("global_swap_changes_label", False)),
        })
    return out, errors


def fit_best_threshold(ds: Sequence[float], ys: Sequence[int]) -> Dict[str, float]:
    xs = sorted(set(float(x) for x in ds))
    if not xs:
        raise ValueError("empty calibration set")
    mids = [xs[0] - 1.0] + [(a + b) / 2.0 for a, b in zip(xs, xs[1:])] + [xs[-1] + 1.0]
    best_tuple = None
    best_rec: Dict[str, float] | None = None
    for sign in (-1.0, 1.0):
        for threshold in mids:
            z = [sign * (float(d) - threshold) for d in ds]
            pred = [1 if zz >= 0.0 else -1 for zz in z]
            margins = [float(y) * zz for y, zz in zip(ys, z)]
            acc = sum(int(p == y) for p, y in zip(pred, ys)) / len(ys)
            # Break ties toward larger minimum and mean margin, then lower |threshold|.
            cand = (acc, min(margins), mean(margins) or 0.0, -abs(threshold))
            if best_tuple is None or cand > best_tuple:
                best_tuple = cand
                best_rec = {
                    "kind": "best_threshold", "a": sign, "b": -sign * threshold,
                    "threshold": threshold, "calib_acc": acc,
                    "calib_margin_mean": mean(margins), "calib_margin_min": min(margins),
                }
    assert best_rec is not None
    return best_rec


def fit_least_squares(ds: Sequence[float], ys: Sequence[int]) -> Dict[str, float]:
    x = np.asarray(ds, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    z = a * x + b
    pred = np.where(z >= 0.0, 1, -1)
    margins = y * z
    return {
        "kind": "least_squares", "a": float(a), "b": float(b),
        "calib_acc": float(np.mean(pred == y)),
        "calib_margin_mean": float(np.mean(margins)),
        "calib_margin_min": float(np.min(margins)),
    }


def evaluate_affine(fit: Dict[str, float], rows: Sequence[Dict[str, Any]], arm_sign: int) -> Dict[str, Any]:
    a, b = float(fit["a"]), float(fit["b"])
    groups = {
        "direct_psc_all": lambda r: r["suite"] == "paired_state_conservation" and r["is_changed"] and r["relation_family"] == "direct_anchor",
        "direct_psc_same": lambda r: r["suite"] == "paired_state_conservation" and r["is_changed"] and r["relation_family"] == "direct_anchor" and r.get("initial_pattern") == "same",
        "direct_psc_opp": lambda r: r["suite"] == "paired_state_conservation" and r["is_changed"] and r["relation_family"] == "direct_anchor" and r.get("initial_pattern") == "opposite",
        "graph_psc_all": lambda r: r["suite"] == "paired_state_conservation" and r["is_changed"] and r["relation_family"] == "graph_transfer",
        "graph_psc_same": lambda r: r["suite"] == "paired_state_conservation" and r["is_changed"] and r["relation_family"] == "graph_transfer" and r.get("initial_pattern") == "same",
        "graph_psc_opp": lambda r: r["suite"] == "paired_state_conservation" and r["is_changed"] and r["relation_family"] == "graph_transfer" and r.get("initial_pattern") == "opposite",
        "graph_xt_same": lambda r: r["suite"] == "cross_template_state_readout" and r["is_changed"] and r["relation_family"] == "graph_transfer" and r.get("initial_pattern") == "same",
        "unchanged_psc": lambda r: r["suite"] == "paired_state_conservation" and not r["is_changed"],
    }
    out: Dict[str, Any] = {}
    for name, filt in groups.items():
        xs = [r for r in rows if filt(r)]
        if not xs:
            out[name] = {"n": 0}
            continue
        z = [a * float(r["d"]) + b for r in xs]
        pred = [1 if zz >= 0.0 else -1 for zz in z]
        y = [int(r["y"]) for r in xs]
        canonical = [int(p == yy) for p, yy in zip(pred, y)]
        arm = [int(p == arm_sign * yy) for p, yy in zip(pred, y)]
        out[name] = {
            "n": len(xs),
            "canonical_acc": mean(canonical), "arm_target_acc": mean(arm),
            "canonical_margin_mean": mean([yy * zz for yy, zz in zip(y, z)]),
            "arm_margin_mean": mean([arm_sign * yy * zz for yy, zz in zip(y, z)]),
            "d_mean": mean([float(r["d"]) for r in xs]), "z_mean": mean(z),
        }
    return out


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    choices, errors = choice_rows_from_state_predictions(load_jsonl(run_dir / "eval_state_predictions.jsonl"))
    calib = [r for r in choices if r["suite"] == "paired_state_conservation" and r["is_changed"] and r["relation_family"] == "direct_anchor"]
    graph = [r for r in choices if r["suite"] == "paired_state_conservation" and r["is_changed"] and r["relation_family"] == "graph_transfer"]
    if not calib:
        raise ValueError(f"no calibration rows in {run_dir}")
    analyses = []
    for arm_sign in (1, -1):
        ys = [arm_sign * int(r["y"]) for r in calib]
        for fit in (fit_best_threshold([r["d"] for r in calib], ys), fit_least_squares([r["d"] for r in calib], ys)):
            rec = dict(fit)
            rec["arm_sign"] = arm_sign
            rec["eval"] = evaluate_affine(rec, choices, arm_sign)
            analyses.append(rec)
    central = result.get("central_eval", {})
    return {
        "run_dir": str(run_dir), "condition": result.get("condition"), "arm": result.get("arm"),
        "seed": result.get("seed"), "cell_tag": result.get("cell_tag"),
        "train_cmp_acc": result.get("final_train_metrics", {}).get("train_cmp_acc"),
        "train_state_acc": result.get("final_train_metrics", {}).get("train_state_acc"),
        "central_eval": central,
        "heldheld_closure_acc": central.get("heldheld_unseen_edge_closure_acc"),
        "raw_direct_acc": mean([r["raw_correct"] for r in calib]),
        "raw_graph_acc": mean([r["raw_correct"] for r in graph]),
        "n_choice_rows": len(choices), "n_parse_errors": len(errors), "parse_errors_sample": errors[:5],
        "n_calib_direct": len(calib), "n_graph": len(graph),
        "direct_d_min": min(float(r["d"]) for r in calib), "direct_d_max": max(float(r["d"]) for r in calib),
        "graph_d_min": min(float(r["d"]) for r in graph) if graph else None,
        "graph_d_max": max(float(r["d"]) for r in graph) if graph else None,
        "fits": analyses,
    }


def collect_dirs(roots: Sequence[Path], explicit_dirs: Sequence[Path], condition: str | None) -> List[Path]:
    dirs: List[Path] = []
    for d in explicit_dirs:
        dirs.append(d)
    for root in roots:
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not (d / "result.json").exists() or not (d / "eval_state_predictions.jsonl").exists():
                continue
            if condition:
                rec = json.loads((d / "result.json").read_text(encoding="utf-8"))
                if rec.get("condition") != condition:
                    continue
            dirs.append(d)
    # de-duplicate preserving order
    seen = set(); out = []
    for d in dirs:
        s = str(d)
        if s not in seen:
            out.append(d); seen.add(s)
    return out


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", type=Path, default=[])
    ap.add_argument("--run-dirs", nargs="*", type=Path, default=[])
    ap.add_argument("--condition", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dirs = collect_dirs(args.roots, args.run_dirs, args.condition)
    analyses = [analyze_run(d) for d in dirs]
    payload = {"n_runs": len(analyses), "runs": analyses}
    write_json(args.out / "frozen_affine_saved_outputs.json", payload)

    lines = ["# research frozen-affine scalar tests from saved outputs", "", "Post-training one-dimensional affine calibration from direct h0/h2 state rows; tested on unanchored h1/h3 graph state rows. This is representational sufficiency, not proof that training learned the affine map.", ""]
    for a in analyses:
        lines.append(f"## {Path(a['run_dir']).name}")
        lines.append(f"condition={a.get('condition')} arm={a.get('arm')} seed={a.get('seed')} train_cmp={fmt(a.get('train_cmp_acc'))} hh_closure={fmt(a.get('heldheld_closure_acc'))} raw_direct={fmt(a.get('raw_direct_acc'))} raw_graph={fmt(a.get('raw_graph_acc'))} d_direct=[{fmt(a.get('direct_d_min'))},{fmt(a.get('direct_d_max'))}] d_graph=[{fmt(a.get('graph_d_min'))},{fmt(a.get('graph_d_max'))}]")
        lines.append("")
        lines.append("| affine fit | anchor sign | calib | direct canon | graph same canon | graph same arm | graph all canon | graph all arm | xt graph canon |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for f in a["fits"]:
            ev = f["eval"]
            lines.append("| {kind} | {sgn:+d} | {cal} | {dc} | {gsc} | {gsa} | {gac} | {gaa} | {xt} |".format(
                kind=f["kind"], sgn=int(f["arm_sign"]), cal=fmt(f.get("calib_acc")),
                dc=fmt(ev["direct_psc_all"].get("canonical_acc")),
                gsc=fmt(ev["graph_psc_same"].get("canonical_acc")),
                gsa=fmt(ev["graph_psc_same"].get("arm_target_acc")),
                gac=fmt(ev["graph_psc_all"].get("canonical_acc")),
                gaa=fmt(ev["graph_psc_all"].get("arm_target_acc")),
                xt=fmt(ev["graph_xt_same"].get("canonical_acc")),
            ))
        lines.append("")
    (args.out / "frozen_affine_saved_outputs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN_AFFINE_SAVED_OUTPUTS_COMPLETE", "n_runs": len(analyses), "summary": str(args.out / "frozen_affine_saved_outputs.md"), "json": str(args.out / "frozen_affine_saved_outputs.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
