#!/usr/bin/env python3
"""research: compare dense fast-screen item movement with partial full-eval movement.

Inputs are transition-comparison JSONs produced by official_transition_compare.py.
The goal is to quantify which dense-focus movements survive from fast_eval to the
completed official full-eval columns while the full dense evaluation is still running.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

ROOT = pathlib.Path(".").resolve()
FAST = pathlib.Path("experiments/archive/functional_learning/data/dense_fast_transition_compare/dense_vs_coherent86_fast_official_transition.json")
OFFICIAL_PARTIAL = pathlib.Path("experiments/archive/functional_learning/data/dense_official_partial_transition/dense_vs_coherent86_official_partial_official_transition.json")


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rank(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[order[j]] == vals[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson(rank(xs), rank(ys))


def score_delta_rows(trans: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["column"]: r for r in trans["score_summary"]["rows"]}


def subtask_map(trans: dict[str, Any], col: str) -> dict[str, dict[str, Any]]:
    rec = trans.get("column_comparisons", {}).get(col, {})
    return {r["subtask"]: r for r in rec.get("subtasks_by_abs_official_delta", [])}


def summarize_column(fast: dict[str, Any], official: dict[str, Any], col: str) -> dict[str, Any]:
    fscore = score_delta_rows(fast).get(col, {})
    oscore = score_delta_rows(official).get(col, {})
    fcomp = fast["column_comparisons"].get(col, {})
    ocomp = official["column_comparisons"].get(col, {})
    fsubs = subtask_map(fast, col)
    osubs = subtask_map(official, col)
    common_names = sorted(set(fsubs) & set(osubs))
    pairs = []
    for name in common_names:
        fd = float(fsubs[name].get("delta_acc_b_minus_a", 0.0))
        od = float(osubs[name].get("delta_acc_b_minus_a", 0.0))
        pairs.append({
            "subtask": name,
            "fast_delta": fd,
            "official_delta": od,
            "official_minus_fast_delta": od - fd,
            "fast_n": fsubs[name].get("n"),
            "official_n": osubs[name].get("n"),
        })
    xs = [p["fast_delta"] for p in pairs]
    ys = [p["official_delta"] for p in pairs]
    same_sign = None
    if pairs:
        nonzero = [p for p in pairs if p["fast_delta"] != 0 and p["official_delta"] != 0]
        if nonzero:
            same_sign = sum(1 for p in nonzero if (p["fast_delta"] > 0) == (p["official_delta"] > 0)) / len(nonzero)
    largest_disagreements = sorted(pairs, key=lambda p: abs(p["official_minus_fast_delta"]), reverse=True)[:12]
    sign_reversals = [p for p in pairs if p["fast_delta"] * p["official_delta"] < 0]
    sign_reversals = sorted(sign_reversals, key=lambda p: abs(p["official_delta"] - p["fast_delta"]), reverse=True)[:12]
    return {
        "column": col,
        "fast_score_delta_computed": fscore.get("computed_delta_b_minus_a"),
        "official_score_delta_computed": oscore.get("computed_delta_b_minus_a"),
        "official_minus_fast_score_delta": None if fscore.get("computed_delta_b_minus_a") is None or oscore.get("computed_delta_b_minus_a") is None else oscore["computed_delta_b_minus_a"] - fscore["computed_delta_b_minus_a"],
        "fast_net_item_delta": fcomp.get("net_item_delta_b_minus_a"),
        "official_net_item_delta": ocomp.get("net_item_delta_b_minus_a"),
        "common_subtasks": len(pairs),
        "subtask_delta_pearson": pearson(xs, ys),
        "subtask_delta_spearman": spearman(xs, ys),
        "nonzero_same_sign_fraction": same_sign,
        "sign_reversals": sign_reversals,
        "largest_delta_disagreements": largest_disagreements,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", default=str(FAST))
    ap.add_argument("--official", default=str(OFFICIAL_PARTIAL))
    ap.add_argument("--out-root", default="experiments/archive/functional_learning/data/fast_vs_official_partial")
    args = ap.parse_args()
    fast_path = pathlib.Path(args.fast)
    official_path = pathlib.Path(args.official)
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    fast = load(fast_path)
    official = load(official_path)
    common_cols = [c for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"] if c in fast.get("column_comparisons", {}) and c in official.get("column_comparisons", {})]
    result = {
        "status": "FAST_VS_OFFICIAL_PARTIAL_READOUT",
        "fast_transition": rel(fast_path),
        "official_partial_transition": rel(official_path),
        "common_columns": common_cols,
        "columns": {c: summarize_column(fast, official, c) for c in common_cols},
        "interpretation": "Fast_eval and full_eval use different item sets/sizes. Agreement supports using fast movement as a rough direction; disagreement means the official payload must dominate route judgment. This file does not infer unfinished columns.",
    }
    out_json = out_root / "fast_vs_official_partial.json"
    out_md = out_root / "fast_vs_official_partial.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research fast-screen vs partial official movement\n", f"Fast transition: `{rel(fast_path)}`", f"Official partial transition: `{rel(official_path)}`\n"]
    lines.append("| column | fast score delta | full score delta | full-fast | fast net item | full net item | common subtasks | Pearson | Spearman | nonzero same-sign |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in common_cols:
        rec = result["columns"][c]
        def fmt(x: Any) -> str:
            return "" if x is None else f"{float(x):.6f}"
        def fmtd(x: Any) -> str:
            return "" if x is None else f"{float(x):+.6f}"
        lines.append(f"| {c} | {fmtd(rec['fast_score_delta_computed'])} | {fmtd(rec['official_score_delta_computed'])} | {fmtd(rec['official_minus_fast_score_delta'])} | {rec.get('fast_net_item_delta')} | {rec.get('official_net_item_delta')} | {rec.get('common_subtasks')} | {fmt(rec.get('subtask_delta_pearson'))} | {fmt(rec.get('subtask_delta_spearman'))} | {fmt(rec.get('nonzero_same_sign_fraction'))} |")
    for c in common_cols:
        rec = result["columns"][c]
        lines.append(f"\n## {c}\n")
        lines.append("### Largest fast/full delta disagreements\n")
        lines.append("| subtask | fast delta | full delta | full-fast | fast n | full n |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for p in rec["largest_delta_disagreements"][:8]:
            lines.append(f"| {p['subtask']} | {p['fast_delta']:+.6f} | {p['official_delta']:+.6f} | {p['official_minus_fast_delta']:+.6f} | {p['fast_n']} | {p['official_n']} |")
        if rec["sign_reversals"]:
            lines.append("\n### Sign reversals\n")
            lines.append("| subtask | fast delta | full delta | full-fast | fast n | full n |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for p in rec["sign_reversals"][:8]:
                lines.append(f"| {p['subtask']} | {p['fast_delta']:+.6f} | {p['official_delta']:+.6f} | {p['official_minus_fast_delta']:+.6f} | {p['fast_n']} | {p['official_n']} |")
    lines.append("\n## Interpretation\n")
    lines.append(result["interpretation"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "common_columns": common_cols}, indent=2), flush=True)


if __name__ == "__main__":
    main()
