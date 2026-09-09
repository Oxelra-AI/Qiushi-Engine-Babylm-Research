#!/usr/bin/env python3
"""research: source-only extractive surface/discontinuity atlas.

CPU-only analysis of the already materialized compact candidate pairs and the
research extractive construction functions.  It quantifies a treatment dimension
not captured by density/coverage alone: how much each view turns source words
into discontinuous skip-bigrams / short fragments versus keeping contiguous
source discourse or generating new compact paraphrase text.

This script does not train, evaluate checkpoints, run SuperGLUE/AoA, upload, or
submit anything.  Its purpose is to make the pending extractive selected scores
interpretable once the research trainings finish.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import importlib.util
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
PAIRS_PATH = WS / "data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
SCRIPT = WS / "scripts/extractive_view_pool_preflight.py"
OUT_DIR = WS / "data/extractive_surface_discontinuity_atlas"

VARIANTS = ["compact", "prefix_repeat_local", "extractive_balanced", "extractive_wide"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_step224_module():
    spec = importlib.util.spec_from_file_location("extractive", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def quantiles(vals: list[float]) -> dict[str, float | None]:
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None, "min": None, "max": None}
    vals.sort()
    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = p * (len(vals) - 1)
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)
    return {
        "n": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(q(0.5)),
        "p10": float(q(0.1)),
        "p25": float(q(0.25)),
        "p75": float(q(0.75)),
        "p90": float(q(0.9)),
        "min": float(vals[0]),
        "max": float(vals[-1]),
    }


def monotone_align(src_norms: list[str], view_norms: list[str]) -> list[int | None]:
    positions: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(src_norms):
        if n:
            positions[n].append(i)
    cursor_by_word: dict[str, int] = defaultdict(int)
    prev = -1
    out: list[int | None] = []
    for n in view_norms:
        if not n or n not in positions:
            out.append(None)
            continue
        pos_list = positions[n]
        cur = cursor_by_word[n]
        while cur < len(pos_list) and pos_list[cur] <= prev:
            cur += 1
        if cur < len(pos_list):
            pos = pos_list[cur]
            out.append(pos)
            prev = pos
            cursor_by_word[n] = cur + 1
        else:
            out.append(None)
    return out


def bag_copy_counts(src_norms: list[str], view_norms: list[str], is_content) -> dict[str, Any]:
    src_bag = Counter(n for n in src_norms if n)
    copied_all = 0
    absent_all = 0
    copied_content = 0
    absent_content = 0
    content_total = 0
    for n in view_norms:
        if not n:
            continue
        cont = bool(is_content(n))
        if cont:
            content_total += 1
        if src_bag.get(n, 0) > 0:
            copied_all += 1
            src_bag[n] -= 1
            if cont:
                copied_content += 1
        else:
            absent_all += 1
            if cont:
                absent_content += 1
    norm_total = sum(1 for n in view_norms if n)
    return {
        "norm_word_count": norm_total,
        "content_word_count": content_total,
        "bag_copied_all": copied_all,
        "bag_absent_all": absent_all,
        "bag_copied_content": copied_content,
        "bag_absent_content": absent_content,
        "bag_absent_all_frac": absent_all / norm_total if norm_total else None,
        "bag_absent_content_frac_of_content": absent_content / content_total if content_total else None,
    }


def max_consecutive_run(aligned: list[int | None]) -> tuple[int, int, float | None]:
    runs: list[int] = []
    cur = 0
    prev: int | None = None
    for pos in aligned:
        if pos is None:
            if cur:
                runs.append(cur)
            cur = 0
            prev = None
            continue
        if prev is not None and pos == prev + 1:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 1
        prev = pos
    if cur:
        runs.append(cur)
    return (max(runs) if runs else 0, len(runs), float(statistics.fmean(runs)) if runs else None)


def pair_metrics(pair: dict[str, Any], view_words: list[str], mod) -> dict[str, Any]:
    src_words = mod.split_words(pair["source_text"])
    src_norms = [mod.norm_word(w) for w in src_words]
    view_norms = [mod.norm_word(w) for w in view_words]
    bag = bag_copy_counts(src_norms, view_norms, mod.is_content)
    aligned = monotone_align(src_norms, view_norms)
    aligned_positions = [p for p in aligned if p is not None]
    gaps: list[int] = []
    gap1 = 0
    skip = 0
    adjacent_aligned = 0
    for a, b in zip(aligned, aligned[1:]):
        if a is None or b is None:
            continue
        g = b - a
        if g <= 0:
            continue
        adjacent_aligned += 1
        gaps.append(g)
        if g == 1:
            gap1 += 1
        elif g > 1:
            skip += 1
    max_run, n_runs, mean_run = max_consecutive_run(aligned)
    span = (max(aligned_positions) - min(aligned_positions) + 1) if aligned_positions else 0
    source_len = len(src_words)
    view_len = len(view_words)
    return {
        "pair_id": pair["pair_id"],
        "source_words": source_len,
        "view_words": view_len,
        **bag,
        "content_fraction": bag["content_word_count"] / bag["norm_word_count"] if bag["norm_word_count"] else None,
        "monotone_aligned_words": len(aligned_positions),
        "monotone_aligned_frac": len(aligned_positions) / bag["norm_word_count"] if bag["norm_word_count"] else None,
        "source_span_width": span,
        "source_span_frac": span / source_len if source_len else None,
        "aligned_adjacent_pairs": adjacent_aligned,
        "gap1_adjacent_pairs": gap1,
        "skip_adjacent_pairs": skip,
        "gap1_frac_of_aligned_adjacent": gap1 / adjacent_aligned if adjacent_aligned else None,
        "skip_frac_of_aligned_adjacent": skip / adjacent_aligned if adjacent_aligned else None,
        "mean_positive_gap": float(statistics.fmean(gaps)) if gaps else None,
        "p90_positive_gap": quantiles([float(g) for g in gaps])["p90"],
        "mean_skip_gap": float(statistics.fmean([g for g in gaps if g > 1])) if any(g > 1 for g in gaps) else None,
        "max_consecutive_source_run": max_run,
        "source_run_count": n_runs,
        "mean_source_run_len": mean_run,
        "fragmentation_per_aligned_word": n_runs / len(aligned_positions) if aligned_positions else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_step224_module()
    pairs: list[dict[str, Any]] = []
    with PAIRS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    rows: list[dict[str, Any]] = []
    exemplar_rows: list[dict[str, Any]] = []
    for pair in pairs:
        src_words = mod.split_words(pair["source_text"])
        target_count = int(pair.get("view_words", len(mod.split_words(pair["view_text"]))))
        views = {
            "compact": mod.split_words(pair["view_text"]),
            # Local prefix repeat only: included to anchor the discontinuity metrics.
            # It is distinct from the historical selected repeat arm.
            "prefix_repeat_local": src_words[:target_count],
            "extractive_balanced": mod.build_extractive_balanced(src_words, target_count),
            "extractive_wide": mod.build_extractive_wide(src_words, target_count),
        }
        per_pair: dict[str, dict[str, Any]] = {}
        for variant, view_words in views.items():
            rec = pair_metrics(pair, view_words, mod)
            rec["variant"] = variant
            rec["view_text"] = " ".join(view_words)
            rows.append(rec)
            per_pair[variant] = rec
        # examples where source-only extraction creates many copied skip-bigrams but compact does not
        bal = per_pair["extractive_balanced"]
        comp = per_pair["compact"]
        if len(exemplar_rows) < 20 and bal.get("skip_frac_of_aligned_adjacent") is not None and comp.get("skip_frac_of_aligned_adjacent") is not None:
            if bal["skip_frac_of_aligned_adjacent"] - comp["skip_frac_of_aligned_adjacent"] > 0.35 and bal["view_words"] >= 8:
                exemplar_rows.append({
                    "pair_id": pair["pair_id"],
                    "source": pair["source_text"],
                    "compact": pair["view_text"],
                    "extractive_balanced": " ".join(views["extractive_balanced"]),
                    "extractive_wide": " ".join(views["extractive_wide"]),
                    "balanced_skip_frac": bal["skip_frac_of_aligned_adjacent"],
                    "compact_skip_frac": comp["skip_frac_of_aligned_adjacent"],
                    "balanced_gap1_frac": bal["gap1_frac_of_aligned_adjacent"],
                    "compact_gap1_frac": comp["gap1_frac_of_aligned_adjacent"],
                })

    metric_keys = [
        "content_fraction", "bag_absent_all_frac", "bag_absent_content_frac_of_content",
        "monotone_aligned_frac", "source_span_frac", "gap1_frac_of_aligned_adjacent",
        "skip_frac_of_aligned_adjacent", "mean_positive_gap", "p90_positive_gap",
        "mean_skip_gap", "max_consecutive_source_run", "source_run_count",
        "mean_source_run_len", "fragmentation_per_aligned_word",
    ]
    aggregates: dict[str, Any] = {}
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    for variant in VARIANTS:
        vrecs = by_variant[variant]
        aggregates[variant] = {
            "n_pairs": len(vrecs),
            "metrics": {k: quantiles([r[k] for r in vrecs if r.get(k) is not None]) for k in metric_keys},
            "totals": {
                "view_words": sum(int(r["view_words"]) for r in vrecs),
                "norm_words": sum(int(r["norm_word_count"]) for r in vrecs),
                "content_words": sum(int(r["content_word_count"]) for r in vrecs),
                "bag_copied_content": sum(int(r["bag_copied_content"]) for r in vrecs),
                "bag_absent_content": sum(int(r["bag_absent_content"]) for r in vrecs),
                "gap1_adjacent_pairs": sum(int(r["gap1_adjacent_pairs"]) for r in vrecs),
                "skip_adjacent_pairs": sum(int(r["skip_adjacent_pairs"]) for r in vrecs),
                "aligned_adjacent_pairs": sum(int(r["aligned_adjacent_pairs"]) for r in vrecs),
            },
        }
        t = aggregates[variant]["totals"]
        t["pooled_gap1_frac_of_aligned_adjacent"] = t["gap1_adjacent_pairs"] / t["aligned_adjacent_pairs"] if t["aligned_adjacent_pairs"] else None
        t["pooled_skip_frac_of_aligned_adjacent"] = t["skip_adjacent_pairs"] / t["aligned_adjacent_pairs"] if t["aligned_adjacent_pairs"] else None
        t["pooled_absent_content_frac_of_content"] = t["bag_absent_content"] / t["content_words"] if t["content_words"] else None

    # Compact-relative deltas for the most interpretable aggregate means/totals.
    deltas: dict[str, Any] = {}
    comp = aggregates["compact"]
    for variant in ["extractive_balanced", "extractive_wide", "prefix_repeat_local"]:
        v = aggregates[variant]
        deltas[f"{variant}_minus_compact"] = {
            "mean_skip_frac": v["metrics"]["skip_frac_of_aligned_adjacent"]["mean"] - comp["metrics"]["skip_frac_of_aligned_adjacent"]["mean"],
            "mean_gap1_frac": v["metrics"]["gap1_frac_of_aligned_adjacent"]["mean"] - comp["metrics"]["gap1_frac_of_aligned_adjacent"]["mean"],
            "mean_source_span_frac": v["metrics"]["source_span_frac"]["mean"] - comp["metrics"]["source_span_frac"]["mean"],
            "pooled_absent_content_frac": v["totals"]["pooled_absent_content_frac_of_content"] - comp["totals"]["pooled_absent_content_frac_of_content"],
            "pooled_skip_frac": v["totals"]["pooled_skip_frac_of_aligned_adjacent"] - comp["totals"]["pooled_skip_frac_of_aligned_adjacent"],
        }

    payload = {
        "status": "EXTRACTIVE_SURFACE_DISCONTINUITY_ATLAS",
        "created_utc": now(),
        "meaning": "CPU-only text atlas for interpreting the pending source-only extractive DeBERTa selected results; no training/eval/upload/leaderboard action.",
        "pairs_path": str(PAIRS_PATH),
        "script": str(SCRIPT),
        "variants": VARIANTS,
        "historical_repeat_warning": "prefix_repeat_local is only a local structural baseline. A01 research/240 corrected the historical selected repeat arm to hash-rotated cyclic repetition, not simple prefix-first-N repetition.",
        "aggregate": aggregates,
        "compact_relative_deltas": deltas,
        "examples_high_balanced_skip_vs_compact": exemplar_rows[:10],
        "interpretation": {
            "what_skip_bigrams_mean": "High skip_frac among monotone-aligned copied words indicates a telegraphic source-only view: adjacent view tokens often were separated in the original source. This is distinct from natural compact paraphrase, where novel adjacencies may be fluent generated re-expression rather than deletion artifacts.",
            "how_to_use_with_selected_scores": "If extractive arms score far below compact, this atlas helps attribute the deficit to the bundled absence of generated fluent re-expression/source-absent content/context recomposition, not simply to missing source-tail coverage. If a source-only arm scores near compact despite high skip-bigram fragmentation, source-word selection/coverage is more sufficient than expected.",
            "not_a_downstream_result": "These are corpus geometry metrics only; they do not establish BabyLM competence without the pending selected evaluation.",
        },
        "no_training_selected_eval_upload_aoa_or_leaderboard": True,
    }

    out_json = OUT_DIR / "extractive_surface_discontinuity_atlas.json"
    out_md = OUT_DIR / "extractive_surface_discontinuity_atlas.md"
    out_csv = OUT_DIR / "per_pair_surface_discontinuity.csv"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = ["pair_id", "variant", "source_words", "view_words", *metric_keys]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in fieldnames})

    lines = [
        "# research extractive surface/discontinuity atlas",
        "",
        f"Created UTC: `{payload['created_utc']}`",
        "",
        "CPU-only corpus geometry; no training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission.",
        "",
        "`prefix_repeat_local` is a local structural baseline only; the historical selected repeat arm is hash-rotated cyclic repetition after A01's correction.",
        "",
        "## Pooled copied-adjacency summary",
        "",
        "| variant | absent content/content | pooled gap=1 frac | pooled skip frac | mean skip frac | mean source span frac |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        a = aggregates[variant]
        t = a["totals"]
        m = a["metrics"]
        lines.append(
            f"| {variant} | {t['pooled_absent_content_frac_of_content']:.4f} | "
            f"{t['pooled_gap1_frac_of_aligned_adjacent']:.4f} | {t['pooled_skip_frac_of_aligned_adjacent']:.4f} | "
            f"{m['skip_frac_of_aligned_adjacent']['mean']:.4f} | {m['source_span_frac']['mean']:.4f} |"
        )
    lines += ["", "## Compact-relative deltas", ""]
    for name, rec in deltas.items():
        lines.append(f"- `{name}`: {json.dumps(rec, ensure_ascii=False, sort_keys=True)}")
    lines += ["", "## Interpretation", "", payload["interpretation"]["how_to_use_with_selected_scores"], ""]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "out_csv": str(out_csv), "pairs": len(pairs)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
