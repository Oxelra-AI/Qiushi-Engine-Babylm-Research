#!/usr/bin/env python3
"""research: word-aligned tokenizer seam/phase analysis for old-vs-legal deficit.

Scientific purpose
------------------
research rejected crude aggregate tokenizer explanations (tokens-per-word,
truncation, pool support, token multiset identity).  independent review research sharpened the
remaining representation hypothesis: the nonlegal old tokenizer may preserve
productive word-internal seams and stable local token phases on held-out eval
strings better than a tokenizer fitted only on the narrow legal 10M pool.  This
script tests that exact prediction without training, evaluating a model, or
constructing data from eval text.

It tokenizes the official BLiMP/Supplement/EWoK eval strings with two existing
already-trained tokenizers and correlates word-aligned segmentation metrics with
already-measured old-vs-legal per-UID score deltas from research.
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
import re
import statistics
import time
from collections import defaultdict
from typing import Any, Iterable

from transformers import AutoTokenizer

STUDY = _public_path('experiments/archive/frontier_consolidation')
SESSIONS = _public_path('experiments/archive')
EVAL_ROOT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')
DEFICIT_JSON = _public_path('experiments/archive/frontier_consolidation/data/legal_deficit_subtask_decomposition/legal_deficit_subtask_decomposition.json')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/eval_tokenizer_seam_phase_analysis')

TOKENIZER_PATHS = {
    "old": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M'),
    "legal": _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer'),
}

SPECIAL_IDS = {0, 1, 2, 3, 4}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Suffixes/prefixes are heuristic; this is a frozen measurement, not a training
# construction.  They intentionally include the forms implicated by the losing
# BLiMP/EWoK/Supplement subtasks: reflexives, agreement/plural, participial and
# dynamics/property morphology.
SUFFIXES = [
    "selves", "self",
    "ational", "fulness", "iveness", "ousness", "ization", "isation",
    "ation", "ition", "tion", "sion", "ment", "ness", "less", "able", "ible",
    "ally", "ically", "ing", "ied", "ies", "ied", "ed", "est", "ers", "er", "ly", "es", "s",
]
PREFIXES = ["anti", "auto", "counter", "inter", "intra", "mis", "non", "over", "post", "pre", "pro", "re", "sub", "super", "trans", "ultra", "un", "under", "dis", "im", "in", "ir", "il"]
REFLEXIVE_SEAMS = {
    "myself": [2],
    "yourself": [4],
    "himself": [3],
    "herself": [3],
    "itself": [2],
    "oneself": [3],
    "ourselves": [3],
    "yourselves": [4],
    "themselves": [4],
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_texts(column: str, uid: str) -> list[str]:
    texts: list[str] = []
    if column == "BLiMP":
        p = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/blimp_filtered') / f"{uid}.jsonl"
        for d in read_jsonl(p):
            texts.append(d["sentence_good"])
            texts.append(d["sentence_bad"])
    elif column == "Supplement":
        p = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered') / f"{uid}.jsonl"
        for d in read_jsonl(p):
            texts.append(d["sentence_good"])
            texts.append(d["sentence_bad"])
    elif column == "EWoK":
        p = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered') / f"{uid}.jsonl"
        for d in read_jsonl(p):
            # Same convention as research/research complete-string readings.
            texts.append(" ".join([d["Context1"], d["Target1"]]))
            texts.append(" ".join([d["Context2"], d["Target2"]]))
    return texts


def safe_mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def percentile(xs: list[float], q: float) -> float | None:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return None
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xv, yv = zip(*pairs)
    mx, my = statistics.mean(xv), statistics.mean(yv)
    vx = sum((x - mx) ** 2 for x in xv)
    vy = sum((y - my) ** 2 for y in yv)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        r = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = r
        i = j
    return out


def spearman(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def simple_slope_p(xs: list[float | None], ys: list[float | None]) -> dict[str, Any]:
    """Small helper for descriptive correlation with approximate t-test p."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    r = pearson(xs, ys)
    if r is None or len(pairs) < 4 or abs(r) >= 1:
        return {"n": len(pairs), "pearson": r, "approx_t": None, "approx_p_two_sided": None}
    t = r * math.sqrt((len(pairs) - 2) / max(1e-12, 1 - r * r))
    # Normal approximation is adequate for route screening; exact p not needed.
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return {"n": len(pairs), "pearson": r, "approx_t": t, "approx_p_two_sided_normal_approx": p}


def internal_seams(word: str) -> dict[str, list[int]]:
    w = word.lower()
    seams: dict[str, list[int]] = {"suffix": [], "prefix": [], "reflexive": []}
    if w in REFLEXIVE_SEAMS:
        seams["reflexive"].extend(REFLEXIVE_SEAMS[w])
    for suf in SUFFIXES:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            pos = len(w) - len(suf)
            # Avoid treating every short plural/proper noun as a productive seam.
            if suf in {"s", "es"} and len(w) < 5:
                continue
            seams["suffix"].append(pos)
    for pre in PREFIXES:
        if len(w) > len(pre) + 3 and w.startswith(pre):
            seams["prefix"].append(len(pre))
    # Deduplicate while preserving order.
    for k, vals in seams.items():
        seen = set()
        out = []
        for v in vals:
            if 0 < v < len(w) and v not in seen:
                out.append(v)
                seen.add(v)
        seams[k] = out
    return seams


def token_word_analysis(tokenizer, text: str) -> list[dict[str, Any]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    words: list[dict[str, Any]] = []
    for m in WORD_RE.finditer(text):
        ws, we = m.span()
        word = m.group(0)
        overlap = []
        for i, (ts, te) in enumerate(offsets):
            if te <= ws or ts >= we or te <= ts:
                continue
            if ids[i] in SPECIAL_IDS:
                continue
            overlap.append((i, ts, te, ids[i]))
        boundary_positions = set()
        for _i, ts, te, _tid in overlap:
            if ws < ts < we:
                boundary_positions.add(ts - ws)
            if ws < te < we:
                boundary_positions.add(te - ws)
        words.append({
            "word": word,
            "start": ws,
            "end": we,
            "token_count": len(overlap),
            "token_start_index": overlap[0][0] if overlap else None,
            "token_end_index": overlap[-1][0] if overlap else None,
            "boundaries": sorted(boundary_positions),
            "seams": internal_seams(word),
        })
    return words


def jaccard(a: set[int], b: set[int]) -> float | None:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return None
    return len(a & b) / len(u)


def analyze_texts(tokenizers: dict[str, Any], texts: list[str]) -> dict[str, Any]:
    sums: defaultdict[str, float] = defaultdict(float)
    lists: defaultdict[str, list[float]] = defaultdict(list)
    word_count = 0
    aligned_words = 0
    examples_lost_seam: list[dict[str, Any]] = []
    examples_large_shift: list[dict[str, Any]] = []

    for text_idx, text in enumerate(texts):
        old_words = token_word_analysis(tokenizers["old"], text)
        legal_words = token_word_analysis(tokenizers["legal"], text)
        n = min(len(old_words), len(legal_words))
        word_count += max(len(old_words), len(legal_words))
        for wi in range(n):
            ow, lw = old_words[wi], legal_words[wi]
            if ow["word"].lower() != lw["word"].lower() or ow["start"] != lw["start"]:
                continue
            aligned_words += 1
            old_bounds = set(ow["boundaries"])
            legal_bounds = set(lw["boundaries"])
            old_tc = ow["token_count"]
            legal_tc = lw["token_count"]
            sums["legal_minus_old_token_count"] += legal_tc - old_tc
            lists["abs_legal_minus_old_token_count"].append(abs(legal_tc - old_tc))
            bj = jaccard(old_bounds, legal_bounds)
            if bj is not None:
                lists["boundary_jaccard"].append(bj)
            if old_bounds != legal_bounds:
                sums["boundary_changed_words"] += 1
            if old_tc == 1:
                sums["old_whole_words"] += 1
            if legal_tc == 1:
                sums["legal_whole_words"] += 1
            if old_tc != legal_tc:
                sums["token_count_changed_words"] += 1

            if ow["token_start_index"] is not None and lw["token_start_index"] is not None:
                shift = lw["token_start_index"] - ow["token_start_index"]
                lists["token_start_shift"].append(float(shift))
                lists["abs_token_start_shift"].append(abs(float(shift)))
                if abs(shift) >= 3 and len(examples_large_shift) < 25:
                    examples_large_shift.append({
                        "text_idx": text_idx,
                        "word_idx": wi,
                        "word": ow["word"],
                        "shift": shift,
                        "old_tc": old_tc,
                        "legal_tc": legal_tc,
                        "text_prefix": text[:160],
                    })

            seam_map = ow["seams"]
            for family, seams in seam_map.items():
                for seam in seams:
                    key = family
                    sums[f"{key}_seam_candidates"] += 1
                    old_has = seam in old_bounds
                    legal_has = seam in legal_bounds
                    if old_has:
                        sums[f"{key}_old_has_seam"] += 1
                    if legal_has:
                        sums[f"{key}_legal_has_seam"] += 1
                    if old_has and not legal_has:
                        sums[f"{key}_legal_lost_old_seam"] += 1
                        if len(examples_lost_seam) < 40:
                            examples_lost_seam.append({
                                "text_idx": text_idx,
                                "word": ow["word"],
                                "family": family,
                                "seam": seam,
                                "old_boundaries": sorted(old_bounds),
                                "legal_boundaries": sorted(legal_bounds),
                                "old_tc": old_tc,
                                "legal_tc": legal_tc,
                                "text_prefix": text[:180],
                            })
                    if legal_has and not old_has:
                        sums[f"{key}_legal_gained_seam"] += 1
                    if old_tc == 1:
                        sums[f"{key}_old_whole_candidate_word"] += 1
                    if legal_tc == 1:
                        sums[f"{key}_legal_whole_candidate_word"] += 1
            any_seams = [s for vals in seam_map.values() for s in vals]
            if any_seams:
                sums["any_seam_candidate_words"] += 1
                if old_tc == 1:
                    sums["any_old_whole_candidate_word"] += 1
                if legal_tc == 1:
                    sums["any_legal_whole_candidate_word"] += 1
                # count if any candidate seam is preserved/lost
                old_any = any(s in old_bounds for s in any_seams)
                legal_any = any(s in legal_bounds for s in any_seams)
                if old_any:
                    sums["any_old_has_seam_word"] += 1
                if legal_any:
                    sums["any_legal_has_seam_word"] += 1
                if old_any and not legal_any:
                    sums["any_legal_lost_old_seam_word"] += 1
                if legal_any and not old_any:
                    sums["any_legal_gained_seam_word"] += 1

    out: dict[str, Any] = {
        "texts": len(texts),
        "word_count": word_count,
        "aligned_words": aligned_words,
    }
    denom = max(1, aligned_words)
    out["mean_legal_minus_old_token_count_per_word"] = sums["legal_minus_old_token_count"] / denom
    out["mean_abs_token_count_delta_per_word"] = safe_mean(lists["abs_legal_minus_old_token_count"])
    out["token_count_changed_word_frac"] = sums["token_count_changed_words"] / denom
    out["boundary_changed_word_frac"] = sums["boundary_changed_words"] / denom
    out["mean_boundary_jaccard"] = safe_mean(lists["boundary_jaccard"])
    out["legal_minus_old_whole_word_frac"] = (sums["legal_whole_words"] - sums["old_whole_words"]) / denom
    shifts = lists["token_start_shift"]
    abs_shifts = lists["abs_token_start_shift"]
    out["mean_token_start_shift_legal_minus_old"] = safe_mean(shifts)
    out["mean_abs_token_start_shift"] = safe_mean(abs_shifts)
    out["p90_abs_token_start_shift"] = percentile(abs_shifts, 0.90)
    out["p99_abs_token_start_shift"] = percentile(abs_shifts, 0.99)
    if len(shifts) >= 2:
        out["std_token_start_shift"] = statistics.pstdev(shifts)
    else:
        out["std_token_start_shift"] = None

    for family in ["suffix", "prefix", "reflexive"]:
        cand = sums[f"{family}_seam_candidates"]
        old_has = sums[f"{family}_old_has_seam"]
        legal_has = sums[f"{family}_legal_has_seam"]
        out[f"{family}_seam_candidates"] = cand
        out[f"{family}_old_preserve_rate"] = old_has / cand if cand else None
        out[f"{family}_legal_preserve_rate"] = legal_has / cand if cand else None
        out[f"{family}_legal_minus_old_preserve_rate"] = (legal_has - old_has) / cand if cand else None
        out[f"{family}_legal_lost_old_seam_rate_over_candidates"] = sums[f"{family}_legal_lost_old_seam"] / cand if cand else None
        out[f"{family}_legal_gained_seam_rate_over_candidates"] = sums[f"{family}_legal_gained_seam"] / cand if cand else None
        out[f"{family}_legal_lost_old_seam_rate_given_old"] = sums[f"{family}_legal_lost_old_seam"] / old_has if old_has else None
        out[f"{family}_legal_minus_old_whole_candidate_rate"] = (sums[f"{family}_legal_whole_candidate_word"] - sums[f"{family}_old_whole_candidate_word"]) / cand if cand else None
    candw = sums["any_seam_candidate_words"]
    out["any_seam_candidate_word_count"] = candw
    out["any_old_preserve_word_rate"] = sums["any_old_has_seam_word"] / candw if candw else None
    out["any_legal_preserve_word_rate"] = sums["any_legal_has_seam_word"] / candw if candw else None
    out["any_legal_minus_old_preserve_word_rate"] = (sums["any_legal_has_seam_word"] - sums["any_old_has_seam_word"]) / candw if candw else None
    out["any_legal_lost_old_seam_word_rate"] = sums["any_legal_lost_old_seam_word"] / candw if candw else None
    out["any_legal_gained_seam_word_rate"] = sums["any_legal_gained_seam_word"] / candw if candw else None
    out["any_legal_minus_old_whole_candidate_word_rate"] = (sums["any_legal_whole_candidate_word"] - sums["any_old_whole_candidate_word"]) / candw if candw else None
    out["examples_lost_seam"] = examples_lost_seam
    out["examples_large_shift"] = examples_large_shift
    return out


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deficit = json.loads(DEFICIT_JSON.read_text(encoding="utf-8"))
    tokenizers = {}
    tok_info = {}
    for label, path in TOKENIZER_PATHS.items():
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
        tokenizers[label] = tok
        tok_info[label] = {"path": str(path), "len": len(tok), "is_fast": tok.is_fast}

    per_uid: list[dict[str, Any]] = []
    for column, rows in deficit["columns"].items():
        for row in rows:
            uid = row["uid"]
            delta = row.get("delta_legal_step35")
            if delta is None:
                continue
            texts = iter_texts(column, uid)
            m = analyze_texts(tokenizers, texts)
            rec: dict[str, Any] = {
                "column": column,
                "uid": uid,
                "score_delta_legal_step35_minus_old": delta,
                **{k: v for k, v in m.items() if not k.startswith("examples_")},
                "examples_lost_seam": m["examples_lost_seam"][:5],
                "examples_large_shift": m["examples_large_shift"][:5],
            }
            per_uid.append(rec)
            print(json.dumps({"event": "uid_done", "column": column, "uid": uid, "delta": delta, "any_seams": rec["any_seam_candidate_word_count"]}, ensure_ascii=False), flush=True)

    metric_names = [
        "mean_legal_minus_old_token_count_per_word",
        "mean_abs_token_count_delta_per_word",
        "token_count_changed_word_frac",
        "boundary_changed_word_frac",
        "mean_boundary_jaccard",
        "legal_minus_old_whole_word_frac",
        "mean_abs_token_start_shift",
        "p90_abs_token_start_shift",
        "p99_abs_token_start_shift",
        "std_token_start_shift",
        "suffix_legal_minus_old_preserve_rate",
        "suffix_legal_lost_old_seam_rate_over_candidates",
        "suffix_legal_lost_old_seam_rate_given_old",
        "suffix_legal_minus_old_whole_candidate_rate",
        "prefix_legal_minus_old_preserve_rate",
        "prefix_legal_lost_old_seam_rate_over_candidates",
        "reflexive_legal_minus_old_preserve_rate",
        "reflexive_legal_lost_old_seam_rate_over_candidates",
        "any_legal_minus_old_preserve_word_rate",
        "any_legal_lost_old_seam_word_rate",
        "any_legal_minus_old_whole_candidate_word_rate",
    ]

    correlations: dict[str, Any] = {}
    for column in ["BLiMP", "Supplement", "EWoK", "ALL"]:
        rows = per_uid if column == "ALL" else [r for r in per_uid if r["column"] == column]
        deltas = [r["score_delta_legal_step35_minus_old"] for r in rows]
        correlations[column] = {"n_uid": len(rows), "metrics": {}}
        for metric in metric_names:
            vals = [r.get(metric) for r in rows]
            correlations[column]["metrics"][metric] = {
                "mean": safe_mean([v for v in vals if v is not None]),
                "pearson_delta_vs_metric": pearson(deltas, vals),
                "spearman_delta_vs_metric": spearman(deltas, vals),
                "approx": simple_slope_p(deltas, vals),
            }

    worst = sorted(per_uid, key=lambda r: r["score_delta_legal_step35_minus_old"])[:25]
    best = sorted(per_uid, key=lambda r: r["score_delta_legal_step35_minus_old"], reverse=True)[:15]

    result = {
        "status": "EVAL_TOKENIZER_SEAM_PHASE_ANALYSIS",
        "created_utc": now_utc(),
        "purpose": "Test held-out word-aligned seam and token-phase predictions of the old-vs-legal tokenizer deficit; CPU only, no training/tuning on eval text.",
        "inputs": {
            "deficit_json": str(DEFICIT_JSON),
            "eval_root": str(EVAL_ROOT),
            "tokenizers": tok_info,
        },
        "metrics_interpretation": {
            "score_delta": "legal research score minus old non-submittable score; negative means legal lost.",
            "negative_correlation_with_bad_metric": "If a metric is bad when larger (e.g. lost seam rate or position shift), a representation mechanism predicts Pearson(delta, metric) < 0.",
            "seam_metrics": "Heuristic affix/reflexive/prefix boundaries over held-out eval words; not used to construct training data.",
            "phase_metrics": "Word-aligned token start shifts and boundary changes; captures local position displacement not visible in aggregate tokens-per-word.",
        },
        "correlations": correlations,
        "per_uid": per_uid,
        "worst_25_by_legal_loss": worst,
        "best_15_by_legal_gain": best,
    }

    out_json = _public_path('experiments/archive/frontier_consolidation/data/eval_tokenizer_seam_phase_analysis/eval_tokenizer_seam_phase_analysis.json')
    out_csv = _public_path('experiments/archive/frontier_consolidation/data/eval_tokenizer_seam_phase_analysis/per_uid_seam_phase_metrics.csv')
    out_md = _public_path('research/documents/frontier_consolidation/data/eval_tokenizer_seam_phase_analysis/eval_tokenizer_seam_phase_analysis.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["column", "uid", "score_delta_legal_step35_minus_old"] + metric_names + ["texts", "aligned_words", "any_seam_candidate_word_count", "suffix_seam_candidates", "reflexive_seam_candidates"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in per_uid:
            w.writerow(r)

    md: list[str] = []
    md.append("# research held-out tokenizer seam/phase analysis")
    md.append("")
    md.append("CPU-only measurement of old vs legal research tokenizers on official evaluation strings. It tests word-aligned segmentation mechanisms not measured by research aggregate tokens-per-word/support analyses. No model is trained and no eval text is used to construct training data.")
    md.append("")
    md.append("## Tokenizers")
    for label, info in tok_info.items():
        md.append(f"- `{label}` len={info['len']}: `{info['path']}`")
    md.append("")
    md.append("## Correlations with legal-research minus old score delta")
    md.append("Negative delta means the legal tokenizer endpoint lost. For bad metrics (lost seams, larger local token shifts), a representation explanation predicts negative correlation.")
    md.append("")
    for column in ["BLiMP", "Supplement", "EWoK", "ALL"]:
        md.append(f"### {column} (n={correlations[column]['n_uid']})")
        md.append("| metric | mean | Pearson | Spearman |")
        md.append("|---|---:|---:|---:|")
        for metric in metric_names:
            c = correlations[column]["metrics"][metric]
            md.append(f"| `{metric}` | {fmt(c['mean'], 6)} | {fmt(c['pearson_delta_vs_metric'], 4)} | {fmt(c['spearman_delta_vs_metric'], 4)} |")
        md.append("")

    show_metrics = [
        "mean_abs_token_start_shift", "p90_abs_token_start_shift", "boundary_changed_word_frac",
        "any_legal_lost_old_seam_word_rate", "suffix_legal_lost_old_seam_rate_over_candidates",
        "reflexive_legal_lost_old_seam_rate_over_candidates", "any_legal_minus_old_whole_candidate_word_rate",
    ]
    md.append("## Worst legal-tokenizer losses")
    md.append("| column | uid | Δ score | " + " | ".join(show_metrics) + " |")
    md.append("|---|---|---:|" + "---:|" * len(show_metrics))
    for r in worst[:20]:
        md.append("| {column} | {uid} | {delta} | {vals} |".format(
            column=r["column"], uid=r["uid"], delta=fmt(r["score_delta_legal_step35_minus_old"], 2),
            vals=" | ".join(fmt(r.get(m), 4) for m in show_metrics),
        ))
    md.append("")
    md.append("## Largest legal-tokenizer gains")
    md.append("| column | uid | Δ score | " + " | ".join(show_metrics) + " |")
    md.append("|---|---|---:|" + "---:|" * len(show_metrics))
    for r in best[:12]:
        md.append("| {column} | {uid} | {delta} | {vals} |".format(
            column=r["column"], uid=r["uid"], delta=fmt(r["score_delta_legal_step35_minus_old"], 2),
            vals=" | ".join(fmt(r.get(m), 4) for m in show_metrics),
        ))
    md.append("")
    md.append("## Direct reading")
    all_corr = correlations["ALL"]["metrics"]
    blimp_corr = correlations["BLiMP"]["metrics"]
    ewok_corr = correlations["EWoK"]["metrics"]
    md.append(f"- ALL lost-seam word correlation: Pearson {fmt(all_corr['any_legal_lost_old_seam_word_rate']['pearson_delta_vs_metric'],4)}, Spearman {fmt(all_corr['any_legal_lost_old_seam_word_rate']['spearman_delta_vs_metric'],4)}.")
    md.append(f"- BLiMP suffix lost-seam correlation: Pearson {fmt(blimp_corr['suffix_legal_lost_old_seam_rate_over_candidates']['pearson_delta_vs_metric'],4)}, Spearman {fmt(blimp_corr['suffix_legal_lost_old_seam_rate_over_candidates']['spearman_delta_vs_metric'],4)}.")
    md.append(f"- EWoK local phase-shift correlation: Pearson {fmt(ewok_corr['mean_abs_token_start_shift']['pearson_delta_vs_metric'],4)}, Spearman {fmt(ewok_corr['mean_abs_token_start_shift']['spearman_delta_vs_metric'],4)}.")
    md.append("")
    md.append("Interpretation must be made from the sign and specificity: a strong representation-seam mechanism needs lost seams/phase shifts to be larger in the losing UIDs than in gaining UIDs, not merely nonzero. If correlations are weak or wrong-signed, this particular tokenizer-fit mechanism is not the next expensive route.")
    md.append("")
    md.append(f"JSON: `{out_json}`")
    md.append(f"CSV: `{out_csv}`")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "out_csv": str(out_csv),
        "n_uid": len(per_uid),
        "all_lost_seam_corr": correlations["ALL"]["metrics"]["any_legal_lost_old_seam_word_rate"],
        "blimp_suffix_lost_corr": correlations["BLiMP"]["metrics"]["suffix_legal_lost_old_seam_rate_over_candidates"],
        "ewok_phase_shift_corr": correlations["EWoK"]["metrics"]["mean_abs_token_start_shift"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
