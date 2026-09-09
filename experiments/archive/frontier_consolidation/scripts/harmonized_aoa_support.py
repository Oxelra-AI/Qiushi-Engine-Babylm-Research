#!/usr/bin/env python3
"""research harmonized AoA support analysis.

Reads `aoa_mechanism_analysis.json` produced by analyze_aoa_from_surprisals.py
and adds same-word-support correlations, pairwise valid-word transitions,
leave-one-word-out influence, and bootstrap intervals.  This is CPU-only
interpretation of existing AoA outputs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = _public_path('experiments/archive/frontier_consolidation')
DEFAULT_INPUT = _public_path('experiments/archive/frontier_consolidation/data/aoa_mechanism_analysis/aoa_mechanism_analysis.json')
DEFAULT_OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/harmonized_aoa_support')
DEFAULT_NOTE = _public_path('research/notes/frontier_consolidation/harmonized_aoa_support.md')


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def corr(xs: List[float], ys: List[float]) -> Dict[str, Any]:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"n": len(xs), "pearson": None, "pearson_p": None, "spearman": None, "spearman_p": None}
    pr, pp = pearsonr(xs, ys)
    sr, sp = spearmanr(xs, ys)
    return {"n": len(xs), "pearson": float(pr), "pearson_p": float(pp), "spearman": float(sr), "spearman_p": float(sp)}


def basic_stats(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if finite(v))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "mean": sum(xs) / len(xs), "median": q(0.5), "p95": q(0.95), "max": xs[-1]}


def age_bin(month: float) -> str:
    if month <= 20.0:
        return "early_le20"
    if month <= 25.0:
        return "middle_20to25"
    return "late_gt25"


def load_fit_words(model: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if model.get("status") != "fit_done":
        return out
    for w, rec in (model.get("per_word") or {}).items():
        if finite(rec.get("model_aoa_log10_words")) and finite(rec.get("child_aoa_month")):
            out[str(w)] = {
                "model_aoa": float(rec["model_aoa_log10_words"]),
                "child_aoa": float(rec["child_aoa_month"]),
                "subword_len": float(rec.get("subword_len", 1)),
            }
    return out


def bootstrap_ci(xs: List[float], ys: List[float], n_boot: int, seed: int) -> Dict[str, Any]:
    if len(xs) < 4:
        return {"n_boot": 0}
    rng = random.Random(seed)
    vals: List[float] = []
    n = len(xs)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idx]
        by = [ys[i] for i in idx]
        if len(set(bx)) >= 2 and len(set(by)) >= 2:
            vals.append(float(pearsonr(bx, by)[0]))
    if not vals:
        return {"n_boot": 0}
    st = basic_stats(vals)
    return {"n_boot": len(vals), "pearson_bootstrap_stats": st, "ci95": [st.get("p05"), st.get("p95")]}


def common_model_report(name: str, fit: Dict[str, Dict[str, float]], words: List[str], n_boot: int) -> Dict[str, Any]:
    xs = [fit[w]["model_aoa"] for w in words]
    ys = [fit[w]["child_aoa"] for w in words]
    c = corr(xs, ys)
    c["bootstrap"] = bootstrap_ci(xs, ys, n_boot=n_boot, seed=13017 + abs(hash(name)) % 1000000)
    bins: Dict[str, List[float]] = defaultdict(list)
    for w in words:
        bins[age_bin(fit[w]["child_aoa"])].append(fit[w]["model_aoa"])
    c["model_aoa_stats"] = basic_stats(xs)
    c["model_aoa_by_child_age_bin"] = {k: basic_stats(v) for k, v in bins.items()}
    early = c["model_aoa_by_child_age_bin"].get("early_le20", {}).get("mean")
    late = c["model_aoa_by_child_age_bin"].get("late_gt25", {}).get("mean")
    c["late_minus_early_model_aoa_log10"] = (float(late - early) if early is not None and late is not None else None)
    # LOO influence on Pearson r.
    loo = []
    base_r = c.get("pearson")
    if base_r is not None and len(words) > 4:
        for i, w in enumerate(words):
            xx = xs[:i] + xs[i+1:]
            yy = ys[:i] + ys[i+1:]
            cc = corr(xx, yy)
            rr = cc.get("pearson")
            if rr is not None:
                loo.append({"word": w, "child_aoa_month": fit[w]["child_aoa"], "model_aoa": fit[w]["model_aoa"], "pearson_without_word": rr, "delta_without_minus_full": rr - base_r})
    c["largest_loo_increase_in_r"] = sorted(loo, key=lambda x: x["delta_without_minus_full"], reverse=True)[:20]
    c["largest_loo_decrease_in_r"] = sorted(loo, key=lambda x: x["delta_without_minus_full"])[:20]
    return c


def pairwise_report(anchor_name: str, other_name: str, fits: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    a = fits.get(anchor_name, {})
    b = fits.get(other_name, {})
    common = sorted(set(a) & set(b))
    rows = []
    for w in common:
        rows.append({
            "word": w,
            "child_aoa_month": a[w]["child_aoa"],
            "anchor_model_aoa": a[w]["model_aoa"],
            "other_model_aoa": b[w]["model_aoa"],
            "delta_other_minus_anchor": b[w]["model_aoa"] - a[w]["model_aoa"],
        })
    child = [r["child_aoa_month"] for r in rows]
    delta = [r["delta_other_minus_anchor"] for r in rows]
    by_bin: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_bin[age_bin(r["child_aoa_month"])].append(r["delta_other_minus_anchor"])
    only_anchor = sorted(set(a) - set(b))
    only_other = sorted(set(b) - set(a))
    def trans(words: List[str], src: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        bins: Dict[str, int] = defaultdict(int)
        for w in words:
            bins[age_bin(src[w]["child_aoa"])] += 1
        return {"n": len(words), "by_child_age_bin": dict(bins), "sample": words[:50]}
    return {
        "anchor": anchor_name,
        "other": other_name,
        "common_n": len(common),
        "only_anchor_valid": trans(only_anchor, a),
        "only_other_valid": trans(only_other, b),
        "delta_stats": basic_stats(delta),
        "child_aoa_vs_delta": corr(child, delta),
        "delta_by_child_age_bin": {k: basic_stats(v) for k, v in by_bin.items()},
        "largest_advances_other_vs_anchor": sorted(rows, key=lambda r: r["delta_other_minus_anchor"])[:25],
        "largest_delays_other_vs_anchor": sorted(rows, key=lambda r: r["delta_other_minus_anchor"], reverse=True)[:25],
    }


def make_note(payload: Dict[str, Any]) -> str:
    lines = ["# research harmonized AoA support", ""]
    lines.append("This CPU note uses fitted per-word model AoAs from `analyze_aoa_from_surprisals.py`. It checks same-word support, fit-entry/fit-exit composition, word influence, and bootstrap uncertainty so small raw AoA changes are not overinterpreted.")
    lines.append("")
    lines.append(f"Fit-done models: {', '.join(payload['fit_done_models'])}")
    lines.append(f"Common fitted-word set across fit-done models: {payload['common_support']['n']} words.")
    lines.append("")
    lines.append("## Same-word correlations")
    for name, rec in payload["common_support"]["models"].items():
        lines.append(f"- {name}: r={rec.get('pearson')}, p={rec.get('pearson_p')}, Spearman={rec.get('spearman')}, late-minus-early={rec.get('late_minus_early_model_aoa_log10')}")
    lines.append("")
    lines.append("## Pairwise comparisons to clean_qwen_seed43022")
    for key, rec in payload["pairwise_to_anchor"].items():
        cd = rec["child_aoa_vs_delta"]
        lines.append(f"- {key}: common={rec['common_n']}, mean_delta={rec['delta_stats'].get('mean')}, child-vs-delta r={cd.get('pearson')}, p={cd.get('pearson_p')}; only_anchor={rec['only_anchor_valid']['n']}, only_other={rec['only_other_valid']['n']}")
    lines.append("")
    lines.append(f"JSON: `{payload['out_json']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    ap.add_argument("--anchor", default="clean_qwen_seed43022")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    source = json.loads(pathlib.Path(args.input).read_text(encoding="utf-8"))
    models = source.get("models", {})
    fits = {name: load_fit_words(rec) for name, rec in models.items() if rec.get("status") == "fit_done"}
    fit_done = sorted([n for n, f in fits.items() if f])
    common = sorted(set.intersection(*(set(fits[n]) for n in fit_done))) if fit_done else []
    common_models = {n: common_model_report(n, fits[n], common, args.n_boot) for n in fit_done}
    pairwise = {f"{n}_minus_{args.anchor}": pairwise_report(args.anchor, n, fits) for n in fit_done if n != args.anchor and args.anchor in fits}
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "harmonized_aoa_support.json"
    payload = {
        "status": "HARMONIZED_AOA_SUPPORT",
        "source_analysis": str(pathlib.Path(args.input)),
        "fit_done_models": fit_done,
        "anchor": args.anchor,
        "common_support": {"n": len(common), "words": common, "models": common_models},
        "pairwise_to_anchor": pairwise,
        "out_json": str(out_json),
        "out_note": str(pathlib.Path(args.note)),
        "interpretation": "Same-support and influence analysis of existing AoA outputs. Use after new AoA outputs are incorporated into the source analysis.",
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pathlib.Path(args.note).write_text(make_note(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(out_json), "note": str(pathlib.Path(args.note)), "common_n": len(common), "models": fit_done}, indent=2), flush=True)


if __name__ == "__main__":
    main()
