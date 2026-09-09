#!/usr/bin/env python3
"""research: quantify copy-like geometry in the research accepted packet pool.

research's lexical state checks accepted many UNCHANGED_DISTRACTOR_USE examples.  A
manual read suggested that most of those use sentences were copies or near-copies
of the source.  This script measures that directly using transparent lexical
features before any training stream is assembled.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path.cwd()
DATA = ROOT / "experiments/archive/relation_learning/data/state_use_generation"
IN = DATA / "validated_state_use_packets_strict_all.jsonl"
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/copy_like_diagnosis"
NOTE = ROOT / "research/notes/relation_learning/copy_like_diagnosis.md"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "may", "might", "must", "not", "of", "on",
    "or", "our", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "under", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "would", "you", "your", "one", "only",
    "more", "most", "all", "any", "both", "each", "every", "own", "same", "other", "through", "now",
    "still", "continues", "continue", "continued", "currently", "former", "formerly",
}


def toks(text: str) -> list[str]:
    return [m.group(0).lower().strip("'\u2019") for m in WORD_RE.finditer(text or "")]


def stem(w: str) -> str:
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 6 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 5 and w.endswith("ied"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 5 and w.endswith("es"):
        return w[:-2]
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def content(text: str) -> list[str]:
    out = []
    for t in toks(text):
        s = stem(t)
        if len(s) >= 3 and s not in STOP:
            out.append(s)
    return out


def jaccard(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    if not (A or B):
        return 0.0
    return len(A & B) / len(A | B)


def overlap_min(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def longest_common_contiguous(a: list[str], b: list[str]) -> tuple[int, str]:
    best = 0
    best_i = 0
    # rows are short; O(n*m) is fine
    dp = [0] * (len(b) + 1)
    for i, x in enumerate(a, start=1):
        ndp = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                ndp[j] = dp[j - 1] + 1
                if ndp[j] > best:
                    best = ndp[j]
                    best_i = i - best
        dp = ndp
    return best, " ".join(a[best_i: best_i + best]) if best else ""


def normalized_exact(a: str, b: str) -> bool:
    return " ".join(toks(a)) == " ".join(toks(b))


def metrics(a: str, b: str) -> dict[str, Any]:
    ta, tb = toks(a), toks(b)
    ca, cb = content(a), content(b)
    lcs, span = longest_common_contiguous(ta, tb)
    return {
        "all_token_jaccard": jaccard(ta, tb),
        "content_jaccard": jaccard(ca, cb),
        "content_overlap_min": overlap_min(ca, cb),
        "longest_common_contiguous_words": lcs,
        "longest_common_span": span,
        "use_len": len(tb),
        "source_len": len(ta),
        "exact_norm": normalized_exact(a, b),
        "b_is_substring_of_a_norm": " ".join(tb) in " ".join(ta) if tb else False,
    }


def pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos); hi = math.ceil(pos)
    return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def stat(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"n": 0, "min": None, "p05": None, "mean": None, "median": None, "p95": None, "max": None}
    return {"n": len(vals), "min": round(min(vals), 4), "p05": round(pct(vals, 0.05), 4), "mean": round(statistics.mean(vals), 4), "median": round(statistics.median(vals), 4), "p95": round(pct(vals, 0.95), 4), "max": round(max(vals), 4)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ["use_source_content_jaccard", "use_source_content_overlap_min", "use_source_lcs", "use_update_content_jaccard", "use_update_lcs"]:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        out[key] = stat(vals)
    n = len(rows) or 1
    out["frac_exact_use_source"] = round(sum(r["use_source_exact"] for r in rows) / n, 4)
    out["frac_use_substring_source"] = round(sum(r["use_substring_source"] for r in rows) / n, 4)
    out["frac_lcs_source_ge_6"] = round(sum(r["use_source_lcs"] >= 6 for r in rows) / n, 4)
    out["frac_lcs_source_ge_8"] = round(sum(r["use_source_lcs"] >= 8 for r in rows) / n, 4)
    out["frac_content_overlap_min_ge_0p94"] = round(sum(r["use_source_content_overlap_min"] >= 0.94 for r in rows) / n, 4)
    out["frac_content_jaccard_ge_0p80"] = round(sum(r["use_source_content_jaccard"] >= 0.80 for r in rows) / n, 4)
    # DUP-like catches the manual failure mode: a source sentence fragment or a long shared span.
    out["frac_dup_like_lcs6_or_exact_or_substring"] = round(sum(r["dup_like_lcs6_or_exact_or_substring"] for r in rows) / n, 4)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with IN.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    audited = []
    for r in rows:
        us = metrics(r["source_sentence"], r["use_sentence"])
        uu = metrics(r["update_sentence"], r["use_sentence"])
        rec = {
            "pair_id": r["pair_id"],
            "packet_type": r["packet_type"],
            "source_sentence": r["source_sentence"],
            "update_sentence": r["update_sentence"],
            "use_sentence": r["use_sentence"],
            "source_state": r["source_state"],
            "new_state": r["new_state"],
            "use_source_content_jaccard": us["content_jaccard"],
            "use_source_content_overlap_min": us["content_overlap_min"],
            "use_source_all_token_jaccard": us["all_token_jaccard"],
            "use_source_lcs": us["longest_common_contiguous_words"],
            "use_source_lcs_span": us["longest_common_span"],
            "use_source_exact": us["exact_norm"],
            "use_substring_source": us["b_is_substring_of_a_norm"],
            "use_update_content_jaccard": uu["content_jaccard"],
            "use_update_lcs": uu["longest_common_contiguous_words"],
            "use_update_lcs_span": uu["longest_common_span"],
        }
        rec["dup_like_lcs6_or_exact_or_substring"] = bool(rec["use_source_lcs"] >= 6 or rec["use_source_exact"] or rec["use_substring_source"])
        audited.append(rec)
    by_type = defaultdict(list)
    for r in audited:
        by_type[r["packet_type"]].append(r)
    summary = {
        "status": "STEP044_COPY_LIKE_DIAGNOSIS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(IN),
        "total_packets": len(audited),
        "type_counts": dict(Counter(r["packet_type"] for r in audited)),
        "overall": summarize(audited),
        "by_type": {k: summarize(v) for k, v in by_type.items()},
        "threshold_reading": {
            "compact_experience_qwen_copy_filter": "copy_overlap if content-overlap-min > 0.94 and rewrite length >= 0.85 original length",
            "changed_form_training_filter_used_for_next_validator": "reject final use sentence if exact/substring source repeat or all-token LCS with source >= 6; also monitor content Jaccard and source-state exact phrase reuse",
        },
    }
    with (OUT_DIR / "copy_like_packet_metrics.jsonl").open("w", encoding="utf-8") as f:
        for r in audited:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT_DIR / "copy_like_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # Save representative worst cases from each type.
    examples = []
    for typ, rs in by_type.items():
        rs2 = sorted(rs, key=lambda x: (x["dup_like_lcs6_or_exact_or_substring"], x["use_source_lcs"], x["use_source_content_overlap_min"], x["use_source_content_jaccard"]), reverse=True)
        examples.extend(rs2[:20])
    with (OUT_DIR / "copy_like_worst_examples.jsonl").open("w", encoding="utf-8") as f:
        for r in examples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = ["# research diagnosis of research accepted packet geometry", ""]
    lines.append("The hardened research lexical state validator is not sufficient for the intended relation-retargeted arm, because a final use sentence can satisfy the state-content rule by repeating the source sentence or a long source fragment. This note quantifies that failure before any training is launched.\n")
    lines.append("## Aggregate copy-like measurements\n")
    lines.append("| subset | n | exact use=source | use substring of source | LCS>=6 | LCS>=8 | content-overlap-min>=0.94 | content-Jaccard>=0.80 | combined DUP-like | mean content Jaccard | mean LCS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, rs in [("ALL", audited)] + sorted(by_type.items()):
        s = summarize(rs)
        lines.append(f"| {label} | {len(rs)} | {s['frac_exact_use_source']:.3f} | {s['frac_use_substring_source']:.3f} | {s['frac_lcs_source_ge_6']:.3f} | {s['frac_lcs_source_ge_8']:.3f} | {s['frac_content_overlap_min_ge_0p94']:.3f} | {s['frac_content_jaccard_ge_0p80']:.3f} | {s['frac_dup_like_lcs6_or_exact_or_substring']:.3f} | {s['use_source_content_jaccard']['mean']:.3f} | {s['use_source_lcs']['mean']:.2f} |")
    lines.append("\n## Interpretation\n")
    lines.append("The intended zero-relevant-update packet should train retained-state readout under changed form after an irrelevant update. Packets with exact/substring use sentences or long source spans instead train source copying with an intervening sentence, close to the DUP/REPEAT behavior already known to create changed-form liability. These packets must not be assembled into the practical SOTA arm. The research prompt and validator therefore require changed-form use sentences, no long shared source/update spans, and balanced subsampling of UPDATED_USE and UNCHANGED_DISTRACTOR_USE after validation.\n")
    lines.append(f"Machine-readable summary: `{OUT_DIR / 'copy_like_summary.json'}`. Per-packet metrics: `{OUT_DIR / 'copy_like_packet_metrics.jsonl'}`. Worst examples: `{OUT_DIR / 'copy_like_worst_examples.jsonl'}`.\n")
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
