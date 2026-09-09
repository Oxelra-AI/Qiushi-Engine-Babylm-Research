#!/usr/bin/env python3
"""research: preregister dose-overlap structure and reference lines before trusted scoring.

The dose experiment adds aligned-restatement pairs to an inherited COMPACT_EXPERIENCE aligned
block, but the added pairs may differ in source mix and lexical-overlap spectrum.
This script computes inherited and added-pool statistics under one definition and
writes the reference-line expectations used to read later source-conditioned faces.
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
from collections import Counter, defaultdict
from typing import Any

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/preregister_dose_overlap_and_reference_lines.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
OUT = WS / "data/dose_overlap_preregistration"
COMPACT_EXPERIENCE_SELECTED = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl"
DOSE21_SELECTED = WS / "data/probe_clean_nested_dose_streams/dose21_selected_pairs_probe_clean.jsonl"
DOSE25_EXTRA = WS / "data/probe_clean_nested_dose_streams/dose25_extra_selected_pairs_probe_clean_from_shard0.jsonl"
DOSE25_SUPERSET = WS / "data/probe_clean_nested_dose_streams/dose25_selected_pairs_probe_clean_superset.jsonl"
MATERIALIZATION_META = WS / "data/probe_clean_nested_dose_streams/probe_clean_nested_dose_materialization_metadata.json"
COMPACT_EXPERIENCE_INFO = ROOT / "experiments/archive/compact_experience/data/pair_information_audit/pair_information_audit.json"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "my", "no", "not", "of", "on",
    "or", "our", "ours", "she", "should", "so", "such", "than", "that", "the", "their", "theirs", "them",
    "then", "there", "these", "they", "this", "those", "to", "too", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "will", "with", "would", "you", "your", "about", "above", "after", "again", "against",
    "all", "am", "any", "because", "before", "below", "between", "both", "cannot", "down", "during", "each",
    "few", "further", "here", "how", "more", "most", "other", "out", "over", "same", "some", "through", "under",
    "until", "up", "very", "while", "why", "now", "only", "also", "than", "like", "one", "two", "three"
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_tokens(text: str) -> list[str]:
    toks: list[str] = []
    for m in WORD_RE.finditer(str(text or "")):
        t = m.group(0).lower().strip("'\u2019")
        if t:
            toks.append(t)
    return toks


def content_set(text: str) -> set[str]:
    return {t for t in norm_tokens(text) if len(t) >= 3 and t not in STOPWORDS}


def word_count(text: str) -> int:
    return len(str(text or "").split())


def lcs_words(a: str, b: str) -> int:
    x = norm_tokens(a); y = norm_tokens(b)
    if not x or not y:
        return 0
    prev = [0] * (len(y) + 1)
    best = 0
    for xi in x:
        cur = [0] * (len(y) + 1)
        for j, yj in enumerate(y, start=1):
            if xi == yj:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def metrics(pair: dict[str, Any]) -> dict[str, Any]:
    orig = str(pair.get("original", "")); rew = str(pair.get("rewrite", ""))
    os = content_set(orig); rs = content_set(rew)
    inter = len(os & rs); union = len(os | rs)
    overlap_min = inter / max(1, min(len(os), len(rs))) if (os or rs) else 0.0
    jac = inter / max(1, union) if (os or rs) else 0.0
    ow = int(pair.get("original_words") or word_count(orig))
    rw = int(pair.get("rewrite_words") or word_count(rew))
    return {
        "pair_id": pair.get("pair_id"),
        "source": pair.get("source", ""),
        "original_words": ow,
        "rewrite_words": rw,
        "pair_words": int(pair.get("pair_words") or (ow + rw)),
        "len_ratio": (rw / ow) if ow else None,
        "content_jaccard_common_def": jac,
        "content_overlap_min_common_def": overlap_min,
        "stored_content_overlap": pair.get("content_overlap"),
        "stored_content_jaccard": pair.get("content_jaccard"),
        "longest_common_contiguous_words_common_def": lcs_words(orig, rew),
    }


def stat(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    vals.sort()
    if not vals:
        return {"n": 0}
    def q(frac: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = frac * (len(vals) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {
        "n": len(vals),
        "min": vals[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "mean": statistics.fmean(vals),
        "median": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": vals[-1],
    }


def bin_key(x: float, cuts: list[float]) -> str:
    lo = 0.0
    for c in cuts:
        if x < c:
            return f"[{lo:.2f},{c:.2f})"
        lo = c
    return f"[{lo:.2f},inf)"


def summarize_pool(name: str, path: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [metrics(p) for p in iter_jsonl(path)]
    by_source_counts = Counter(str(r["source"]) for r in rows)
    by_source_words: dict[str, int] = defaultdict(int)
    for r in rows:
        by_source_words[str(r["source"])] += int(r["pair_words"])
    hist = Counter(bin_key(float(r["content_jaccard_common_def"]), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) for r in rows)
    lcs_hist = Counter(bin_key(float(r["longest_common_contiguous_words_common_def"]), [1, 2, 3, 4, 5, 6, 8, 10, 15]) for r in rows)
    summary = {
        "pool": name,
        "path": rel(path),
        "n_pairs": len(rows),
        "pair_words": int(sum(int(r["pair_words"]) for r in rows)),
        "word_stats": {
            "original_words": stat([r["original_words"] for r in rows]),
            "rewrite_words": stat([r["rewrite_words"] for r in rows]),
            "pair_words": stat([r["pair_words"] for r in rows]),
            "len_ratio": stat([r["len_ratio"] for r in rows if r["len_ratio"] is not None]),
            "content_jaccard_common_def": stat([r["content_jaccard_common_def"] for r in rows]),
            "content_overlap_min_common_def": stat([r["content_overlap_min_common_def"] for r in rows]),
            "longest_common_contiguous_words_common_def": stat([r["longest_common_contiguous_words_common_def"] for r in rows]),
            "stored_content_overlap": stat([r["stored_content_overlap"] for r in rows if r["stored_content_overlap"] is not None]),
            "stored_content_jaccard": stat([r["stored_content_jaccard"] for r in rows if r["stored_content_jaccard"] is not None]),
        },
        "source_pair_counts": dict(sorted(by_source_counts.items())),
        "source_pair_words": dict(sorted(by_source_words.items())),
        "content_jaccard_histogram_common_def": dict(sorted(hist.items())),
        "lcs_histogram_common_def": dict(sorted(lcs_hist.items())),
        "frac_lcs_ge6_common_def": sum(1 for r in rows if int(r["longest_common_contiguous_words_common_def"]) >= 6) / max(1, len(rows)),
        "frac_content_jaccard_ge0p6_common_def": sum(1 for r in rows if float(r["content_jaccard_common_def"]) >= 0.6) / max(1, len(rows)),
        "frac_content_jaccard_lt0p3_common_def": sum(1 for r in rows if float(r["content_jaccard_common_def"]) < 0.3) / max(1, len(rows)),
    }
    return rows, summary


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8"); return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pools = {
        "inherited_compact_experience_aln_selected": COMPACT_EXPERIENCE_SELECTED,
        "added_dose21_selected": DOSE21_SELECTED,
        "added_dose25_extra": DOSE25_EXTRA,
        "added_dose25_superset": DOSE25_SUPERSET,
    }
    summaries = {}
    for name, path in pools.items():
        rows, summ = summarize_pool(name, path)
        summaries[name] = summ
        write_csv(OUT / f"{name}_pair_overlap_metrics.csv", rows)
    mat = read_json(MATERIALIZATION_META)
    compact_experience_info = read_json(COMPACT_EXPERIENCE_INFO)
    # Reference lines are preregistered expectations, not fitted laws. Positive numbers mean expected increase in true-source benefit.
    aln_off_overlap_mean = (3.5975 + 4.1998) / 2.0
    aln_off_overlap_se_pair_ref = (0.1236 + 0.1379) / 2.0
    aln_off_compact_overlap_ref = 1.67
    inherited_pair_words_per_10m = float(mat["targets"]["inherited_pair_words_per_10M"])
    inherited_simplewiki_words = float(compact_experience_info["pair_audit"]["source_word_accounting"]["simple_wiki"]["pair"])
    dose21_words = float(mat["targets"]["dose21_added_pair_words_per_10M"])
    dose25_words = float(mat["targets"]["dose25_added_pair_words_per_10M"])
    dose21_simple = float(mat["dose21_selection"]["pair_source"]["words"].get("simple_wiki", 0))
    dose25_simple = float(mat["dose21_selection"]["pair_source"]["words"].get("simple_wiki", 0)) + float(mat["dose25_selection"]["extra_pair_source"]["words"].get("simple_wiki", 0))
    ref = {
        "status": "PREREGISTERED_BEFORE_TRUSTED_DOSE_FACE_SCORING",
        "created_utc": now(),
        "reference_lines_are_calibrated_expectations_not_a_fitted_law": True,
        "compact_experience_reference": {
            "ALN_minus_OFF_wikipedia_overlap_gain_T_vs_N_seed_mean": aln_off_overlap_mean,
            "seed43022": 3.5975,
            "seed43122": 4.1998,
            "approx_pair_SE_reference_mean": aln_off_overlap_se_pair_ref,
            "inherited_pair_words_per_10M": inherited_pair_words_per_10m,
            "inherited_simple_wiki_pair_words_per_10M": inherited_simplewiki_words,
            "ALN_minus_OFF_compact_overlap_reference_approx": aln_off_compact_overlap_ref,
        },
        "dose_word_inputs": {
            "dose21_added_pair_words_per_10M": dose21_words,
            "dose25_added_pair_words_per_10M": dose25_words,
            "dose21_simple_wiki_pair_words_per_10M": dose21_simple,
            "dose25_simple_wiki_pair_words_per_10M": dose25_simple,
        },
        "expected_wikipedia_overlap_delta_gain_T_vs_N": {
            "total_aligned_words_scaling": {
                "dose21": aln_off_overlap_mean * dose21_words / inherited_pair_words_per_10m,
                "dose25": aln_off_overlap_mean * dose25_words / inherited_pair_words_per_10m,
            },
            "simple_wiki_words_scaling": {
                "dose21": aln_off_overlap_mean * dose21_simple / inherited_simplewiki_words,
                "dose25": aln_off_overlap_mean * dose25_simple / inherited_simplewiki_words,
            },
        },
        "expected_compact_overlap_delta_gain_reference": {
            "dose21": aln_off_compact_overlap_ref * dose21_words / inherited_pair_words_per_10m,
            "dose25": aln_off_compact_overlap_ref * dose25_words / inherited_pair_words_per_10m,
        },
        "source_absent_nonoverlap_branches": {
            "flat": "asymmetric reach: added aligned restatement sharpens source-recurring readout without changing source-absent substitutions",
            "degrading": "restatement-scaling liability: added relation practice consumes or biases fixed-budget work against source-absent changed-form targets",
            "rising": "practiced-relation transfer: lower-overlap added pairs place more rewrite content in a source-absent/substitution class, so improving nonoverlap is an intended relation-specific effect rather than generic broad generalization",
        },
        "interpretation_rules": [
            "compact overlap is the register-neutral comparator for the Wikipedia source-recurring face",
            "Wikipedia near the SimpleWiki-scaled line with compact near the total-word line supports register/source amplification rather than saturation",
            "both Wikipedia and compact below their calibrated expectations is where bounded gain or saturation becomes plausible",
            "source-absent nonoverlap movement must be read through the three preregistered branches before assigning liability or transfer",
        ],
    }
    out = {"status": "DOSE_OVERLAP_PREREGISTRATION_DONE", "created_utc": now(), "pools": summaries, "reference_lines": ref}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research dose overlap preregistration",
        "",
        f"Created: {now()}",
        "",
        "The repaired aligned-restatement dose should not be read as merely more of the inherited relation. The added pairs differ in source mix and in lexical overlap. This note is written before trusted dose-face scores are consumed.",
        "",
        "## Same-definition overlap summaries",
        "",
        "| pool | pairs | pair words | mean Jaccard | median Jaccard | p25-p75 Jaccard | mean min-overlap | mean LCS | frac LCS>=6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summaries.items():
        js = s["word_stats"]["content_jaccard_common_def"]; os = s["word_stats"]["content_overlap_min_common_def"]; ls = s["word_stats"]["longest_common_contiguous_words_common_def"]
        lines.append(f"| {name} | {s['n_pairs']} | {s['pair_words']} | {js['mean']:.4f} | {js['median']:.4f} | {js['p25']:.4f}-{js['p75']:.4f} | {os['mean']:.4f} | {ls['mean']:.2f} | {s['frac_lcs_ge6_common_def']:.4f} |")
    lines += [
        "",
        "## Calibrated expectations for reading faces",
        "",
        f"COMPACT_EXPERIENCE ALN--OFF Wikipedia overlap reference uses seed mean {aln_off_overlap_mean:.4f} from +3.5975/+4.1998 at {inherited_pair_words_per_10m/1_000_000:.4f}M inherited pair words per 10M. Total aligned-word scaling gives dose21 {ref['expected_wikipedia_overlap_delta_gain_T_vs_N']['total_aligned_words_scaling']['dose21']:.3f} and dose25 {ref['expected_wikipedia_overlap_delta_gain_T_vs_N']['total_aligned_words_scaling']['dose25']:.3f}. SimpleWiki-word scaling gives dose21 {ref['expected_wikipedia_overlap_delta_gain_T_vs_N']['simple_wiki_words_scaling']['dose21']:.3f} and dose25 {ref['expected_wikipedia_overlap_delta_gain_T_vs_N']['simple_wiki_words_scaling']['dose25']:.3f}.",
        f"Compact overlap uses the approximate ALN--OFF reference {aln_off_compact_overlap_ref:.3f}; total aligned-word scaling gives dose21 {ref['expected_compact_overlap_delta_gain_reference']['dose21']:.3f} and dose25 {ref['expected_compact_overlap_delta_gain_reference']['dose25']:.3f}.",
        "",
        "For source-absent/nonoverlap targets, preregister three branches: flat = asymmetric reach; degrading = restatement-side fixed-budget liability; rising = practiced substitution-like relation transfer because lower-overlap added pairs contain more rewrite tokens absent from the source. A nonoverlap rise is not generic broad generalization by itself.",
        "",
        f"Detailed metrics: `{rel(OUT)}`.",
    ]
    ((_PUBLIC_ROOT / 'research/documents/relation_learning/data/dose_overlap_preregistration/preregistration.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "summary": rel(OUT / "summary.json"), "note": rel((_PUBLIC_ROOT / 'research/documents/relation_learning/data/dose_overlap_preregistration/preregistration.md'))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
