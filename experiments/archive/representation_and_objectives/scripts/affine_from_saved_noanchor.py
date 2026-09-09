#!/usr/bin/env python3
"""research: frozen affine scalar probe from saved comparison-only predictions.

Scientific purpose
------------------
Use a trained comparison-only research model (no held bridge anchors) as a frozen
source of the candidate-event scalar coordinate d = score(candidate0)-score(candidate1).
Fit only a one-dimensional affine decision rule on direct h0/h2 state-anchor
rows, then test whether the same calibrated scalar predicts h1/h3 graph-transfer
state rows.  This is a representational sufficiency check, not a causal training
claim.

The script reads saved eval_state_predictions.jsonl/result.json from no-anchor
runs.  In research no-anchor runs, h0/h2 bridge anchors were not used for training,
so using the eval direct-anchor rows as calibration rows mimics a post-training
frozen affine fit.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def state_choice_records(state_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in state_rows:
        by[(r.get("condition"), r.get("arm"), r.get("seed"), r.get("bridge_sign"), r.get("suite"), r.get("query_key"))].append(r)
    out: List[Dict[str, Any]] = []
    for key, rows in by.items():
        if len(rows) != 2:
            continue
        rows = sorted(rows, key=lambda x: int(x["candidate_index"]))
        # y=+1 means candidate index 0 is the true final owner; y=-1 means candidate index 1.
        if bool(rows[0].get("label_true")) == bool(rows[1].get("label_true")):
            continue
        y = 1 if bool(rows[0].get("label_true")) else -1
        pred_raw = 1 if float(rows[0]["d_e"]) >= 0.0 else -1
        r0 = rows[0]
        out.append({
            "condition": key[0], "arm": key[1], "seed": key[2], "bridge_sign": key[3],
            "suite": key[4], "query_key": key[5], "d": float(rows[0]["d_e"]), "y": y,
            "raw_correct": int(pred_raw == y), "is_changed": bool(r0.get("is_changed")),
            "relation": r0.get("relation"), "relation_family": r0.get("relation_family"),
            "initial_pattern": r0.get("initial_pattern"), "static_slot": r0.get("static_slot"),
            "is_direct_anchor": bool(r0.get("is_direct_anchor")),
        })
    return out


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def fit_best_threshold(ds: Sequence[float], ys: Sequence[int]) -> Dict[str, float]:
    if not ds:
        raise ValueError("empty calibration set")
    xs = sorted(set(float(x) for x in ds))
    if len(xs) == 1:
        thresholds = [xs[0] - 1.0, xs[0] + 1.0]
    else:
        thresholds = [xs[0] - 1.0] + [(a + b) / 2.0 for a, b in zip(xs, xs[1:])] + [xs[-1] + 1.0]
    best = None
    for s in (1.0, -1.0):
        for t in thresholds:
            preds = [1 if s * (float(d) - t) >= 0.0 else -1 for d in ds]
            acc = sum(int(p == y) for p, y in zip(preds, ys)) / len(ys)
            margin = mean([float(y) * s * (float(d) - t) for d, y in zip(ds, ys)])
            min_margin = min([float(y) * s * (float(d) - t) for d, y in zip(ds, ys)])
            cand = (acc, min_margin, abs(margin or 0.0), s, t, margin)
            if best is None or cand > best:
                best = cand
    assert best is not None
    acc, min_margin, _, s, t, margin = best
    return {"kind": "best_threshold", "a": s, "b": -s * t, "threshold": t, "sign": s, "calib_acc": acc, "calib_margin_mean": margin, "calib_margin_min": min_margin}


def fit_least_squares(ds: Sequence[float], ys: Sequence[int]) -> Dict[str, float]:
    x = np.asarray(ds, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    z = a * x + b
    acc = float(np.mean(np.where(z >= 0.0, 1, -1) == y))
    margins = y * z
    return {"kind": "least_squares", "a": float(a), "b": float(b), "calib_acc": acc, "calib_margin_mean": float(np.mean(margins)), "calib_margin_min": float(np.min(margins))}


def apply_fit(fit: Dict[str, float], rows: Sequence[Dict[str, Any]], bridge_sign: int) -> Dict[str, Any]:
    a, b = float(fit["a"]), float(fit["b"])
    out: Dict[str, Any] = {}
    groups = {
        "direct_all": lambda r: r["is_changed"] and r["relation_family"] == "direct_anchor" and r["suite"] == "paired_state_conservation",
        "direct_same": lambda r: r["is_changed"] and r["relation_family"] == "direct_anchor" and r["suite"] == "paired_state_conservation" and r.get("initial_pattern") == "same",
        "direct_opp": lambda r: r["is_changed"] and r["relation_family"] == "direct_anchor" and r["suite"] == "paired_state_conservation" and r.get("initial_pattern") == "opposite",
        "graph_all": lambda r: r["is_changed"] and r["relation_family"] == "graph_transfer" and r["suite"] == "paired_state_conservation",
        "graph_same": lambda r: r["is_changed"] and r["relation_family"] == "graph_transfer" and r["suite"] == "paired_state_conservation" and r.get("initial_pattern") == "same",
        "graph_opp": lambda r: r["is_changed"] and r["relation_family"] == "graph_transfer" and r["suite"] == "paired_state_conservation" and r.get("initial_pattern") == "opposite",
        "xt_graph_same": lambda r: r["is_changed"] and r["relation_family"] == "graph_transfer" and r["suite"] == "cross_template_state_readout" and r.get("initial_pattern") == "same",
    }
    for name, filt in groups.items():
        xs = [r for r in rows if filt(r)]
        if not xs:
            out[name] = {"n": 0}
            continue
        z = [a * float(r["d"]) + b for r in xs]
        canonical_correct = [int((1 if zz >= 0.0 else -1) == int(r["y"])) for zz, r in zip(z, xs)]
        # bridge_sign=-1 means the hypothetical anchor connector installs the opposite state convention.
        arm_correct = [int((1 if zz >= 0.0 else -1) == int(bridge_sign) * int(r["y"])) for zz, r in zip(z, xs)]
        out[name] = {
            "n": len(xs),
            "canonical_acc": mean(canonical_correct),
            "arm_target_acc": mean(arm_correct),
            "canonical_signed_margin_mean": mean([float(r["y"]) * zz for zz, r in zip(z, xs)]),
            "arm_signed_margin_mean": mean([float(bridge_sign) * float(r["y"]) * zz for zz, r in zip(z, xs)]),
            "z_mean": mean(z),
            "d_mean": mean([float(r["d"]) for r in xs]),
        }
    return out


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    rows = state_choice_records(load_jsonl(run_dir / "eval_state_predictions.jsonl"))
    calib = [r for r in rows if r["suite"] == "paired_state_conservation" and r["is_changed"] and r["relation_family"] == "direct_anchor"]
    graph = [r for r in rows if r["suite"] == "paired_state_conservation" and r["is_changed"] and r["relation_family"] == "graph_transfer"]
    fits = [fit_best_threshold([r["d"] for r in calib], [r["y"] for r in calib]), fit_least_squares([r["d"] for r in calib], [r["y"] for r in calib])]
    fit_results = []
    for fit in fits:
        for bs in (1, -1):
            rec = dict(fit)
            rec["bridge_sign_target"] = bs
            # For bs=-1, the affine should be re-fit with flipped anchor labels, not merely evaluated with flipped targets.
            if bs == -1:
                if fit["kind"] == "best_threshold":
                    rec = fit_best_threshold([r["d"] for r in calib], [-r["y"] for r in calib])
                else:
                    rec = fit_least_squares([r["d"] for r in calib], [-r["y"] for r in calib])
                rec["bridge_sign_target"] = bs
            rec["eval"] = apply_fit(rec, rows, bs)
            fit_results.append(rec)
    return {
        "run_dir": str(run_dir),
        "condition": result.get("condition"), "seed": result.get("seed"),
        "source_bridge_sign": result.get("bridge_sign"), "cell_tag": result.get("cell_tag"),
        "train_cmp_acc": result.get("final_train_metrics", {}).get("train_cmp_acc"),
        "train_state_acc": result.get("final_train_metrics", {}).get("train_state_acc"),
        "heldheld_closure_acc": result.get("central_eval", {}).get("heldheld_unseen_edge_closure_acc"),
        "mixed_acc_model": result.get("central_eval", {}).get("mixed_held_seen_orientation_acc"),
        "model_direct_same": result.get("central_eval", {}).get("direct_same"),
        "model_graph_same": result.get("central_eval", {}).get("graph_same"),
        "n_choice_rows": len(rows), "n_calib_direct": len(calib), "n_graph_eval": len(graph),
        "raw_direct_acc": mean([r["raw_correct"] for r in calib]),
        "raw_graph_acc": mean([r["raw_correct"] for r in graph]),
        "direct_d_min": min([r["d"] for r in calib]) if calib else None,
        "direct_d_max": max([r["d"] for r in calib]) if calib else None,
        "graph_d_min": min([r["d"] for r in graph]) if graph else None,
        "graph_d_max": max([r["d"] for r in graph]) if graph else None,
        "fits": fit_results,
    }


def find_run_dirs(root: Path, condition: str = "tied") -> List[Path]:
    dirs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "result.json").exists():
            continue
        rec = json.loads((d / "result.json").read_text(encoding="utf-8"))
        if rec.get("condition") != condition:
            continue
        # no-anchor bs+ and bs- are identical in training; keep bs+ by default if both exist.
        if rec.get("no_bridge_anchors") is True:
            dirs.append(d)
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--condition", default="tied")
    ap.add_argument("--prefer-bs-plus", action="store_true")
    args = ap.parse_args()

    run_dirs: List[Path] = []
    for root in args.roots:
        run_dirs.extend(find_run_dirs(root, args.condition))
    if args.prefer_bs_plus:
        by_seed: Dict[int, List[Path]] = defaultdict(list)
        for d in run_dirs:
            r = json.loads((d / "result.json").read_text(encoding="utf-8"))
            by_seed[int(r.get("seed"))].append(d)
        kept = []
        for seed, ds in sorted(by_seed.items()):
            plus = []
            for d in ds:
                r = json.loads((d / "result.json").read_text(encoding="utf-8"))
                if int(r.get("bridge_sign")) == 1:
                    plus.append(d)
            kept.append((plus or ds)[0])
        run_dirs = kept

    analyses = [analyze_run(d) for d in run_dirs]
    out = {"n_runs": len(analyses), "runs": analyses}
    write_json(args.out / "affine_from_saved_noanchor.json", out)

    lines = ["# research frozen-affine scalar probe from saved no-anchor outputs", "", "This is a CPU-only representational check: no weights are updated; one-dimensional affine rules are fit from direct h0/h2 eval anchor coordinates and tested on h1/h3 graph-transfer rows.", ""]
    for a in analyses:
        lines.append(f"## {Path(a['run_dir']).name}")
        lines.append(f"train_cmp={a['train_cmp_acc']} heldheld_closure={a['heldheld_closure_acc']} raw_direct={a['raw_direct_acc']} raw_graph={a['raw_graph_acc']} d_direct=[{a['direct_d_min']:.3f},{a['direct_d_max']:.3f}] d_graph=[{a['graph_d_min']:.3f},{a['graph_d_max']:.3f}]")
        lines.append("")
        lines.append("| fit | bs target | calib acc | graph same canon | graph same arm | graph all canon | graph all arm | xt graph canon |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for f in a["fits"]:
            ev = f["eval"]
            lines.append("| {kind} | {bs} | {cal:.3f} | {gs_c:.3f} | {gs_a:.3f} | {ga_c:.3f} | {ga_a:.3f} | {xt:.3f} |".format(
                kind=f["kind"], bs=f["bridge_sign_target"], cal=f["calib_acc"],
                gs_c=ev["graph_same"].get("canonical_acc") or 0.0, gs_a=ev["graph_same"].get("arm_target_acc") or 0.0,
                ga_c=ev["graph_all"].get("canonical_acc") or 0.0, ga_a=ev["graph_all"].get("arm_target_acc") or 0.0,
                xt=ev["xt_graph_same"].get("canonical_acc") or 0.0,
            ))
        lines.append("")
    (args.out / "affine_from_saved_noanchor.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "AFFINE_FROM_SAVED_NOANCHOR_COMPLETE", "n_runs": len(analyses), "summary": str(args.out / "affine_from_saved_noanchor.md"), "json": str(args.out / "affine_from_saved_noanchor.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
