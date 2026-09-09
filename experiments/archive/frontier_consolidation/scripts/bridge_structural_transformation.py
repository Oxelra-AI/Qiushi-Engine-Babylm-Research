#!/usr/bin/env python3
"""research: structural-transformation analysis of the 103 retained research bridge outputs.

The research closure used source_copy_degree (does each output content word appear
anywhere in source), which is near-1 by construction under a source-attested-lemma
instruction and says nothing about contiguity, order, function-word restructuring, or
predicate-argument recoding. This analyzer measures the actual structural dimensions
that distinguish fluent proposition-preserving TRANSFORMATION from fluent EXTRACTION.

For each accepted pair it computes, on tokenized word sequences:
  - longest_source_contiguous_run / gen_len: how much of the output is a verbatim
    contiguous source substring (high = extraction)
  - source_subsequence_order_fraction: fraction of gen content words that appear in
    the same relative order as in source (high = order-preserving = extractive)
  - kept_source_word_fraction: gen words that are literally source words
  - novel_word_fraction: gen words NOT literally in source (function-word insertion,
    inflection change, or new lemma)
  - function_word_edit_rate: change in stopword multiset relative to source span
  - reordering_index: 1 - Kendall-tau-like order agreement of shared content words
  - leading_deletion_only: whether the output equals a source suffix (prefix trimmed)
  - is_verbatim_source_substring: output is an exact contiguous source window
  - edit_class in {verbatim_substring, prefix_trim_only, deletion_reorder,
    light_restructure, substantive_restructure}

An extraction-like output keeps source order, has a long verbatim run, minimal
function-word change, and low reordering. A genuine transformation reorders clauses,
changes function words / inflection, and breaks long verbatim runs even while using
mostly source lemmas.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import re
import statistics as st
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
ACCEPTED = _public_path('experiments/archive/frontier_consolidation/data/improved_fluent_bridge/final_accepted.jsonl')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/bridge_structural_transformation')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/bridge_structural_transformation/structural_transformation_analysis.json')
OUT_CSV = _public_path('experiments/archive/frontier_consolidation/data/bridge_structural_transformation/per_pair_structural.csv')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/bridge_structural_transformation/structural_transformation_analysis.md')

STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "as", "of", "to",
    "in", "on", "at", "by", "for", "with", "from", "into", "over", "under", "is",
    "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have",
    "has", "had", "that", "which", "who", "whom", "whose", "this", "these", "those",
    "it", "its", "they", "them", "their", "he", "she", "his", "her", "we", "you",
    "i", "not", "no", "can", "could", "would", "should", "will", "shall", "may",
    "might", "must", "so", "such", "there", "here", "also", "when", "while", "where",
    "how", "why", "what", "though", "although", "however", "yet", "because", "since",
    "about", "up", "down", "out", "off", "very", "more", "most", "some", "any", "all",
    "each", "both", "either", "neither", "only", "just", "even", "still",
}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-\.']*", text.lower())


def content(seq: list[str]) -> list[str]:
    return [w for w in seq if w not in STOP and len(w) >= 2]


def longest_contiguous_run(gen: list[str], src: list[str]) -> int:
    """Longest contiguous gen substring that is also a contiguous src substring."""
    n, m = len(gen), len(src)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    best = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gen[i - 1] == src[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > best:
                    best = dp[i][j]
    return best


def lcs_len(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return prev[m]


def order_agreement(gen_c: list[str], src_c: list[str]) -> float:
    """Fraction of shared content words appearing in same relative source order.

    Uses LCS of gen content words against source content words: high means the
    output preserves source content ordering (extractive); low means reordering.
    """
    if not gen_c:
        return 1.0
    # map each gen content word to whether it is in source; keep only shared, in gen order
    src_set = set(src_c)
    gen_shared = [w for w in gen_c if w in src_set]
    if not gen_shared:
        return 0.0
    l = lcs_len(gen_shared, src_c)
    return l / len(gen_shared)


def is_source_suffix(gen: list[str], src: list[str]) -> bool:
    """Output equals a contiguous source suffix (only leading words trimmed)."""
    if not gen or len(gen) > len(src):
        return False
    return src[len(src) - len(gen):] == gen


def is_verbatim_substring(gen: list[str], src: list[str]) -> bool:
    if not gen or len(gen) > len(src):
        return False
    for i in range(0, len(src) - len(gen) + 1):
        if src[i:i + len(gen)] == gen:
            return True
    return False


def stopword_multiset_delta(gen: list[str], src: list[str]) -> float:
    from collections import Counter
    gs = Counter(w for w in gen if w in STOP)
    ss = Counter(w for w in src if w in STOP)
    keys = set(gs) | set(ss)
    if not keys:
        return 0.0
    diff = sum(abs(gs[k] - ss[k]) for k in keys)
    total = sum(ss[k] for k in keys) or 1
    return diff / total


def classify(rec: dict[str, Any]) -> str:
    if rec["is_verbatim_source_substring"]:
        return "verbatim_substring"
    if rec["leading_deletion_only"]:
        return "prefix_trim_only"
    # order preserved and long verbatim run -> deletion+minor reorder (extractive)
    if rec["order_agreement"] >= 0.9 and rec["longest_run_frac"] >= 0.5 and rec["novel_word_fraction"] <= 0.12:
        return "deletion_reorder"
    if rec["order_agreement"] >= 0.75 and rec["novel_word_fraction"] <= 0.2:
        return "light_restructure"
    return "substantive_restructure"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ACCEPTED.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    recs: list[dict[str, Any]] = []
    for r in rows:
        gen_text = r.get("generated_text") or r.get("raw_output") or ""
        src_text = r.get("source_text") or ""
        nat_text = r.get("natural_compact_text") or ""
        gen = words(gen_text)
        src = words(src_text)
        gen_c = content(gen)
        src_c = content(src)
        src_set = set(src)
        kept = [w for w in gen if w in src_set]
        novel = [w for w in gen if w not in src_set]
        run = longest_contiguous_run(gen, src)
        rec = {
            "pair_id": r.get("pair_id"),
            "bucket": r.get("prototype_bucket"),
            "regime": r.get("regime"),
            "source_words": len(src),
            "gen_words": len(gen),
            "gen_content_words": len(gen_c),
            "kept_source_word_fraction": round(len(kept) / max(1, len(gen)), 4),
            "novel_word_fraction": round(len(novel) / max(1, len(gen)), 4),
            "longest_source_contiguous_run": run,
            "longest_run_frac": round(run / max(1, len(gen)), 4),
            "order_agreement": round(order_agreement(gen_c, src_c), 4),
            "function_word_edit_rate": round(stopword_multiset_delta(gen, src), 4),
            "leading_deletion_only": is_source_suffix(gen, src),
            "is_verbatim_source_substring": is_verbatim_substring(gen, src),
            "source_copy_degree_reported": r.get("source_copy_degree"),
            "compact_absent_content_lemma_count": r.get("compact_absent_content_lemma_count"),
            "gen_text": gen_text,
            "source_text": src_text,
            "natural_compact_text": nat_text,
        }
        rec["reordering_index"] = round(1.0 - rec["order_agreement"], 4)
        rec["edit_class"] = classify(rec)
        recs.append(rec)

    from collections import Counter
    cls_counts = Counter(r["edit_class"] for r in recs)

    def summ(key: str) -> dict[str, float]:
        vals = [float(r[key]) for r in recs]
        return {
            "n": len(vals),
            "mean": round(st.mean(vals), 4),
            "median": round(st.median(vals), 4),
            "p10": round(sorted(vals)[max(0, int(0.1 * len(vals)) - 1)], 4),
            "p90": round(sorted(vals)[min(len(vals) - 1, int(0.9 * len(vals)))], 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    extractive_like = cls_counts.get("verbatim_substring", 0) + cls_counts.get("prefix_trim_only", 0) + cls_counts.get("deletion_reorder", 0)
    transform_like = cls_counts.get("light_restructure", 0) + cls_counts.get("substantive_restructure", 0)

    payload = {
        "status": "BRIDGE_STRUCTURAL_TRANSFORMATION",
        "accepted_pairs": len(recs),
        "edit_class_counts": dict(cls_counts),
        "extractive_like_count": extractive_like,
        "transformation_like_count": transform_like,
        "extractive_like_fraction": round(extractive_like / max(1, len(recs)), 4),
        "transformation_like_fraction": round(transform_like / max(1, len(recs)), 4),
        "distributions": {
            "longest_run_frac": summ("longest_run_frac"),
            "order_agreement": summ("order_agreement"),
            "reordering_index": summ("reordering_index"),
            "novel_word_fraction": summ("novel_word_fraction"),
            "kept_source_word_fraction": summ("kept_source_word_fraction"),
            "function_word_edit_rate": summ("function_word_edit_rate"),
        },
        "interpretation": (
            "This measures structural transformation, not lemma overlap. High longest_run_frac + high "
            "order_agreement + low novel_word_fraction indicates fluent extraction (source substring with "
            "trimming/minor reorder). Low order_agreement, broken verbatim runs, and function-word/inflection "
            "edits indicate genuine proposition-preserving re-expression. source_copy_degree cannot separate "
            "these because it is near-1 by construction."
        ),
        "by_bucket": {},
    }
    for b in sorted({r["bucket"] for r in recs}):
        sub = [r for r in recs if r["bucket"] == b]
        bc = Counter(r["edit_class"] for r in sub)
        payload["by_bucket"][b] = {
            "n": len(sub),
            "edit_class_counts": dict(bc),
            "mean_longest_run_frac": round(st.mean([r["longest_run_frac"] for r in sub]), 4),
            "mean_order_agreement": round(st.mean([r["order_agreement"] for r in sub]), 4),
            "mean_novel_word_fraction": round(st.mean([r["novel_word_fraction"] for r in sub]), 4),
        }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cols = ["pair_id", "bucket", "regime", "edit_class", "gen_words", "longest_run_frac",
            "order_agreement", "reordering_index", "novel_word_fraction",
            "kept_source_word_fraction", "function_word_edit_rate",
            "leading_deletion_only", "is_verbatim_source_substring"]
    lines = [",".join(cols)]
    for r in recs:
        lines.append(",".join(str(r[c]) for c in cols))
    OUT_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# research bridge structural-transformation analysis",
        "",
        f"Accepted pairs analyzed: {len(recs)}",
        "",
        "## Edit-class distribution (structural, not lemma overlap)",
        "",
        "| class | n | meaning |",
        "|---|---:|---|",
        f"| verbatim_substring | {cls_counts.get('verbatim_substring',0)} | output is an exact contiguous source window |",
        f"| prefix_trim_only | {cls_counts.get('prefix_trim_only',0)} | output is a source suffix (leading words dropped) |",
        f"| deletion_reorder | {cls_counts.get('deletion_reorder',0)} | source order preserved, long verbatim run, minimal novel words |",
        f"| light_restructure | {cls_counts.get('light_restructure',0)} | some reordering / function-word edits |",
        f"| substantive_restructure | {cls_counts.get('substantive_restructure',0)} | clause reorder + function-word/inflection change |",
        "",
        f"Extraction-like (verbatim + prefix_trim + deletion_reorder): **{extractive_like}/{len(recs)} = {payload['extractive_like_fraction']}**",
        f"Transformation-like (light + substantive restructure): **{transform_like}/{len(recs)} = {payload['transformation_like_fraction']}**",
        "",
        "## Structural distributions",
        "",
        "| metric | mean | median | p10 | p90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in ["longest_run_frac", "order_agreement", "reordering_index", "novel_word_fraction", "kept_source_word_fraction", "function_word_edit_rate"]:
        s = payload["distributions"][k]
        md.append(f"| {k} | {s['mean']} | {s['median']} | {s['p10']} | {s['p90']} |")
    md += ["", "## By bucket", "", "| bucket | n | mean_run_frac | mean_order_agree | mean_novel_frac |", "|---|---:|---:|---:|---:|"]
    for b, v in payload["by_bucket"].items():
        md.append(f"| {b} | {v['n']} | {v['mean_longest_run_frac']} | {v['mean_order_agreement']} | {v['mean_novel_word_fraction']} |")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "accepted_pairs": len(recs),
        "edit_class_counts": dict(cls_counts),
        "extractive_like_fraction": payload["extractive_like_fraction"],
        "transformation_like_fraction": payload["transformation_like_fraction"],
        "mean_longest_run_frac": payload["distributions"]["longest_run_frac"]["mean"],
        "mean_order_agreement": payload["distributions"]["order_agreement"]["mean"],
        "mean_novel_word_fraction": payload["distributions"]["novel_word_fraction"]["mean"],
        "out_json": str(OUT_JSON.relative_to(ROOT)),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
