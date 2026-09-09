#!/usr/bin/env python3
"""research audit: information value and coverage of the current clean-Qwen second view.

This is a research-facing analysis, not an evaluator.  It quantifies how much of the
10M word budget is spent on original-side words versus generated-side words, how many
distinct official sentences are exposed, what kind of surface transformations the
current Qwen paraphrases contain, and how shorter proposition-preserving second views
could change distinct-original coverage at fixed same-window pair budget.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = _public_path('experiments/archive/compact_experience')
PAIR_PATH = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
META_PATH = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
OFFICIAL_POOL = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
OUT_DIR = _public_path('experiments/archive/compact_experience/data/pair_information_audit')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = _public_path('experiments/archive/compact_experience/data/pair_information_audit/pair_information_audit.json')

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
TERMINAL_OK = re.compile(r"[.!?][\"'”’\)]*$")
BAD_START = re.compile(r"^(?:and|but|or|which|that|whose|whom|soon|of)\b", re.I)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "may", "might", "must", "not", "of", "on",
    "or", "our", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "under", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "would", "you", "your", "one", "only",
    "more", "most", "all", "any", "both", "each", "every", "own", "same", "other", "through",
}


def norm_ws(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def split_sentences(text: str) -> list[str]:
    return [norm_ws(x.strip()) for x in SENT_SPLIT.split(text or "") if norm_ws(x.strip())]


def word_count(text: str) -> int:
    return len((text or "").split())


def norm_tokens(text: str) -> list[str]:
    return [m.group(0).lower().strip("'\"“”‘’.,!?;:()[]{}") for m in WORD_RE.finditer(text or "")]


def content_set(text: str) -> set[str]:
    return {t for t in norm_tokens(text) if len(t) >= 3 and t not in STOPWORDS}


def pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def stats(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"n": 0, "min": None, "p05": None, "mean": None, "median": None, "p95": None, "max": None, "sum": 0}
    return {
        "n": len(vals),
        "min": round(min(vals), 4),
        "p05": round(pct(vals, 0.05), 4),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "p95": round(pct(vals, 0.95), 4),
        "max": round(max(vals), 4),
        "sum": round(sum(vals), 4),
    }


def bin_key(x: float, bins: list[float]) -> str:
    last = 0.0
    for b in bins:
        if x < b:
            return f"[{last:.2f},{b:.2f})"
        last = b
    return f"[{last:.2f},inf)"


def is_candidate_sentence(sent: str) -> bool:
    n = word_count(sent)
    if n < 12 or n > 55:
        return False
    s = sent.strip()
    if not TERMINAL_OK.search(s):
        return False
    if BAD_START.match(s.lstrip("\"'“”‘’([* ")):
        return False
    if "..." in s or "http" in s.lower() or "/CHILDES" in s or ".cha" in s.lower() or "= =" in s:
        return False
    if not any(ch.isalpha() for ch in s):
        return False
    first = s.lstrip("\"'“”‘’([ ").split()[0] if s.split() else ""
    if first and first[0].islower() and not first.startswith("*"):
        return False
    return True


def load_pairs() -> list[dict]:
    out = []
    with PAIR_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def audit_pairs(pairs: list[dict], meta: dict) -> dict:
    source = Counter(p["source"] for p in pairs)
    source_words = defaultdict(lambda: {"orig": 0, "rew": 0, "pair": 0, "n": 0})
    op_bins = Counter()
    overlap_bins = Counter()
    len_ratio_bins = Counter()
    examples = {"very_high_overlap": [], "compact": [], "low_overlap": [], "long": []}
    orig_ws, rew_ws, pair_ws, ratios, overlaps, jaccards = [], [], [], [], [], []
    unique_example_ids = set()
    for p in pairs:
        ow = int(p["original_words"]); rw = int(p["rewrite_words"]); pw = int(p["pair_words"])
        orig_ws.append(ow); rew_ws.append(rw); pair_ws.append(pw)
        ratios.append(float(p["len_ratio"]))
        overlaps.append(float(p["content_overlap"]))
        unique_example_ids.add(p.get("example_id"))
        s = source_words[p["source"]]
        s["orig"] += ow; s["rew"] += rw; s["pair"] += pw; s["n"] += 1
        if rw <= 0.75 * ow:
            op_bins["compression_like_rw_le_0p75_orig"] += 1
        elif rw >= 1.25 * ow:
            op_bins["expansion_like_rw_ge_1p25_orig"] += 1
        else:
            op_bins["length_near_preserving"] += 1
        if p.get("content_overlap", 0) >= 0.82:
            op_bins["high_lexical_overlap_ge_0p82"] += 1
        if p.get("content_overlap", 0) <= 0.45:
            op_bins["lower_lexical_overlap_le_0p45"] += 1
        if p.get("entity_source"):
            op_bins["has_entity_anchor"] += 1
        if p.get("num_source"):
            op_bins["has_number_anchor"] += 1
        overlap_bins[bin_key(float(p["content_overlap"]), [0.2, 0.4, 0.6, 0.8, 0.9, 0.95])] += 1
        len_ratio_bins[bin_key(float(p["len_ratio"]), [0.6, 0.75, 0.9, 1.1, 1.25, 1.5])] += 1
        oset, rset = content_set(p["original"]), content_set(p["rewrite"])
        denom = len(oset | rset) or 1
        jaccards.append(len(oset & rset) / denom)
        ex = {k: p[k] for k in ["pair_id", "source", "original", "rewrite", "original_words", "rewrite_words", "content_overlap", "len_ratio"]}
        if len(examples["very_high_overlap"]) < 8 and float(p["content_overlap"]) >= 0.90:
            examples["very_high_overlap"].append(ex)
        if len(examples["compact"]) < 8 and rw <= 0.75 * ow and float(p["content_overlap"]) >= 0.50:
            examples["compact"].append(ex)
        if len(examples["low_overlap"]) < 8 and float(p["content_overlap"]) <= 0.45:
            examples["low_overlap"].append(ex)
        if len(examples["long"]) < 8 and pw >= 85:
            examples["long"].append(ex)
    total_pair = sum(pair_ws); total_orig = sum(orig_ws); total_rew = sum(rew_ws)
    return {
        "pair_count": len(pairs),
        "unique_official_example_ids": len(unique_example_ids),
        "total_pair_words": total_pair,
        "total_original_side_words": total_orig,
        "total_generated_side_words": total_rew,
        "training_word_budget": 10_000_000,
        "pair_word_fraction_of_10M": round(total_pair / 10_000_000, 6),
        "generated_word_fraction_of_10M": round(total_rew / 10_000_000, 6),
        "original_side_fraction_of_10M": round(total_orig / 10_000_000, 6),
        "pairs_per_100k_pair_words": round(len(pairs) / total_pair * 100_000, 2),
        "distinct_originals_per_100k_generated_words": round(len(pairs) / total_rew * 100_000, 2),
        "word_stats": {"original": stats(orig_ws), "rewrite": stats(rew_ws), "pair": stats(pair_ws), "len_ratio": stats(ratios), "content_overlap": stats(overlaps), "content_jaccard": stats(jaccards)},
        "source_pair_counts": dict(source),
        "source_word_accounting": {k: dict(v) for k, v in sorted(source_words.items())},
        "surface_operation_counts": dict(op_bins),
        "overlap_bins": dict(overlap_bins),
        "len_ratio_bins": dict(len_ratio_bins),
        "examples": examples,
        "metadata_crosscheck": {
            "selected_pairs_metadata": meta.get("selected_pairs"),
            "selected_pair_words_metadata": meta.get("selected_pair_words"),
            "selected_pair_word_fraction_metadata": meta.get("selected_pair_word_fraction"),
            "qwen_pair_rows_metadata": meta.get("qwen_pair_rows"),
        },
    }


def audit_candidate_pool() -> dict:
    by_source = Counter(); word_by_source = Counter(); lengths = []
    total_rows = 0; total_sents = 0
    with OFFICIAL_POOL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line); total_rows += 1
            source = row.get("source", "unknown")
            for sent in split_sentences(row.get("text", "")):
                total_sents += 1
                if is_candidate_sentence(sent):
                    n = word_count(sent)
                    by_source[source] += 1
                    word_by_source[source] += n
                    lengths.append(n)
    return {
        "official_pool_rows": total_rows,
        "all_split_sentence_count": total_sents,
        "candidate_sentence_count_12_55w_cleanish": sum(by_source.values()),
        "candidate_word_count": sum(word_by_source.values()),
        "candidate_length_stats": stats(lengths),
        "candidate_counts_by_source": dict(by_source),
        "candidate_words_by_source": dict(word_by_source),
    }


def coverage_projection(candidate_lengths: list[int], pair_budget: int, synthetic_ratios: list[float]) -> dict:
    # Greedy in the existing official order approximates distinct originals possible at fixed total pair budget.
    # Minimum generated view length protects against telegraphic degeneracy.
    out = {}
    for r in synthetic_ratios:
        used = 0; n = 0; gen = 0; orig = 0
        for ow in candidate_lengths:
            gw = max(6, int(round(ow * r)))
            if used + ow + gw > pair_budget:
                break
            used += ow + gw; gen += gw; orig += ow; n += 1
        out[f"synthetic_len_ratio_{r:g}"] = {
            "distinct_originals": n,
            "pair_words_used": used,
            "original_words": orig,
            "synthetic_words": gen,
            "mean_original_words": round(orig / n, 3) if n else None,
            "mean_synthetic_words": round(gen / n, 3) if n else None,
            "distinct_originals_per_current_pair_count": round(n / 37594, 3) if n else 0,
        }
    return out


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    pairs = load_pairs()
    pair_audit = audit_pairs(pairs, meta)
    candidate = audit_candidate_pool()

    # Candidate lengths in deterministic broad-source order: simply all cleanish lengths sorted by source/name then as encountered.
    lengths = []
    with OFFICIAL_POOL.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for sent in split_sentences(row.get("text", "")):
                if is_candidate_sentence(sent):
                    lengths.append(word_count(sent))
    # Keep the original corpus order for a conservative projection and a shortest-first upper reference.
    current_pair_budget = int(pair_audit["total_pair_words"])
    target_pair_budget = int(0.25 * 10_000_000)
    projections = {
        "fixed_current_pair_budget_1p6568M_corpus_order": coverage_projection(lengths, current_pair_budget, [0.35, 0.5, 0.65, 0.8, 1.0]),
        "fixed_target_pair_budget_2p5M_corpus_order": coverage_projection(lengths, target_pair_budget, [0.35, 0.5, 0.65, 0.8, 1.0]),
        "fixed_current_pair_budget_shortest_first_upper_reference": coverage_projection(sorted(lengths), current_pair_budget, [0.35, 0.5, 0.65, 0.8, 1.0]),
    }

    report = {
        "status": "PAIR_INFORMATION_AUDIT",
        "scientific_question": "How efficiently does the current same-window Qwen second view spend the strict 10M word budget, and what is the headroom for concise proposition-preserving second views?",
        "pair_audit": pair_audit,
        "official_candidate_pool": candidate,
        "coverage_projections": projections,
        "interpretation": {
            "current_anchor": "clean-Qwen same-window second view remains the only replicated positive mechanism, but it spends about half of pair-row words on generated rewrites of near-original length and often high lexical overlap.",
            "next_route_implication": "At a fixed same-window pair budget, a second view with generated length 0.35-0.65x of the original could expose many more distinct official originals and relations if semantic fidelity can be filtered; this motivates transformation-family screens rather than another optimizer/masking/tokenizer knob.",
        },
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
