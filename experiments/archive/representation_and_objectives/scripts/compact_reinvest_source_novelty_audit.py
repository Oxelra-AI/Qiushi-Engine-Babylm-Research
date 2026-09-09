#!/usr/bin/env python3
"""Source-novelty audit for compact_view_reinvest added pairs.

independent review noted that reinvestment increases source/pair counts, but count alone does
not prove new propositions.  This CPU audit compares the 2,061 added compact
source-view pairs against the 10,094 core compact pairs by document identity,
exact normalized text, and lexical n-gram similarity.  It is a conservative
near-duplicate screen, not a semantic entailment model.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path(".")
A01 = ROOT / "experiments/archive" / 'representation_and_objectives'
DENSITY_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_core_reinvestment_medium_riskhard"
OUT_DIR = A01 / "data" / "compact_reinvest_source_novelty_audit"
OUT_JSON = OUT_DIR / "compact_reinvest_source_novelty_audit.json"
NOTE = (ROOT / 'research/notes/representation_and_objectives/compact_reinvest_source_novelty_audit.md')

STOP = {
    "the","a","an","and","or","but","if","then","than","that","this","these","those","to","of","in","on","for","with","as","by","from","at","is","are","was","were","be","been","being","it","its","their","they","them","he","she","his","her","we","you","your","our","not","no","can","could","would","should","will","may","might","have","has","had","do","does","did","because","while","where","when","which","who","whom","whose","into","within","about","over","under","between","after","before","during","through","also","more","most","some","many","much","one","two","new","other","such"
}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def iter_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_tokens(text: str, keep_stop: bool = False) -> list[str]:
    toks = TOKEN_RE.findall(text.lower())
    if keep_stop:
        return toks
    return [t for t in toks if (t not in STOP and (len(t) > 2 or t.isdigit()))]


def norm_string(text: str) -> str:
    return " ".join(norm_tokens(text, keep_stop=True))


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}


def jacc(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def quant(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(float(x) for x in vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs)-1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo]*(hi-pos) + xs[hi]*(pos-lo)
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "p25": q(0.25), "mean": sum(xs)/len(xs), "median": q(0.5), "p75": q(0.75), "p95": q(0.95), "p99": q(0.99), "max": xs[-1], "sum": sum(xs)}


def prep(pair: dict[str, Any]) -> dict[str, Any]:
    source = str(pair.get("source_text", ""))
    rewrite = str(pair.get("rewrite_text", ""))
    toks_all = norm_tokens(source, keep_stop=True)
    toks_content = norm_tokens(source, keep_stop=False)
    return {
        "pair": pair,
        "norm": norm_string(source),
        "all_tokens": toks_all,
        "content_tokens": set(toks_content),
        "bigrams_all": ngrams(toks_all, 2),
        "trigrams_all": ngrams(toks_all, 3),
        "source_text": source,
        "rewrite_text": rewrite,
        "doc_id": str(pair.get("doc_id")),
        "sentence_id": str(pair.get("sentence_id")),
        "domain_hits": pair.get("domain_hits") or ["no_domain"],
        "source_words": int(pair.get("source_words", len(source.split()))),
        "rewrite_words": int(pair.get("rewrite_words", len(rewrite.split()))),
    }


def main() -> None:
    core_pairs = [prep(p) for p in iter_jsonl(DENSITY_DIR / "selected_compact_core_pairs.jsonl")]
    added_pairs = [prep(p) for p in iter_jsonl(DENSITY_DIR / "selected_compact_added_pairs.jsonl")]
    reinvest_pairs = list(iter_jsonl(DENSITY_DIR / "selected_compact_reinvest_pairs.jsonl"))

    core_norm_to_indices: dict[str, list[int]] = defaultdict(list)
    doc_to_core: dict[str, list[int]] = defaultdict(list)
    token_index: dict[str, set[int]] = defaultdict(set)
    bigram_index: dict[tuple[str, ...], set[int]] = defaultdict(set)
    trigram_index: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for i, p in enumerate(core_pairs):
        core_norm_to_indices[p["norm"]].append(i)
        doc_to_core[p["doc_id"]].append(i)
        for t in p["content_tokens"]:
            token_index[t].add(i)
        for bg in p["bigrams_all"]:
            bigram_index[bg].add(i)
        for tg in p["trigrams_all"]:
            trigram_index[tg].add(i)

    records: list[dict[str, Any]] = []
    exact_norm = 0
    same_doc = 0
    for ap in added_pairs:
        candidates: set[int] = set()
        for tg in ap["trigrams_all"]:
            candidates.update(trigram_index.get(tg, ()))
        if len(candidates) < 50:
            for bg in ap["bigrams_all"]:
                candidates.update(bigram_index.get(bg, ()))
        if len(candidates) < 50:
            for t in ap["content_tokens"]:
                candidates.update(token_index.get(t, ()))
        candidates.update(doc_to_core.get(ap["doc_id"], ()))
        if not candidates:
            # There is no shared content word/ngram/doc, so all similarities are zero.
            best = None
            rec = {
                "pair_id": ap["pair"].get("pair_id"),
                "doc_id": ap["doc_id"],
                "domain_hits": ap["domain_hits"],
                "source_words": ap["source_words"],
                "rewrite_words": ap["rewrite_words"],
                "content_recall": ap["pair"].get("content_recall"),
                "entity_recall": ap["pair"].get("entity_recall"),
                "number_recall": ap["pair"].get("number_recall"),
                "same_doc_core_count": 0,
                "exact_norm_source_in_core": False,
                "candidate_core_count": 0,
                "best_core_pair_id": None,
                "best_core_doc_id": None,
                "best_content_jaccard": 0.0,
                "best_bigram_jaccard": 0.0,
                "best_trigram_jaccard": 0.0,
                "best_combined_score": 0.0,
                "source_prefix": ap["source_text"][:240],
                "best_core_source_prefix": None,
            }
            records.append(rec)
            continue
        exact = ap["norm"] in core_norm_to_indices
        if exact:
            exact_norm += 1
        sd_count = len(doc_to_core.get(ap["doc_id"], []))
        if sd_count:
            same_doc += 1
        best_rec = None
        best_score = -1.0
        for ci in candidates:
            cp = core_pairs[ci]
            cj = jacc(ap["content_tokens"], cp["content_tokens"])
            bj = jacc(ap["bigrams_all"], cp["bigrams_all"])
            tj = jacc(ap["trigrams_all"], cp["trigrams_all"])
            contain = 0.0
            if ap["norm"] and cp["norm"] and (ap["norm"] in cp["norm"] or cp["norm"] in ap["norm"]):
                contain = min(len(ap["norm"]), len(cp["norm"])) / max(len(ap["norm"]), len(cp["norm"]))
            score = max(tj, bj * 0.75, cj * 0.45, contain * 0.9)
            if score > best_score:
                best_score = score
                best_rec = (ci, cj, bj, tj, contain, cp)
        assert best_rec is not None
        ci, cj, bj, tj, contain, cp = best_rec
        rec = {
            "pair_id": ap["pair"].get("pair_id"),
            "doc_id": ap["doc_id"],
            "domain_hits": ap["domain_hits"],
            "source_words": ap["source_words"],
            "rewrite_words": ap["rewrite_words"],
            "content_recall": ap["pair"].get("content_recall"),
            "entity_recall": ap["pair"].get("entity_recall"),
            "number_recall": ap["pair"].get("number_recall"),
            "same_doc_core_count": sd_count,
            "exact_norm_source_in_core": exact,
            "candidate_core_count": len(candidates),
            "best_core_pair_id": cp["pair"].get("pair_id"),
            "best_core_doc_id": cp["doc_id"],
            "best_same_doc": ap["doc_id"] == cp["doc_id"],
            "best_content_jaccard": cj,
            "best_bigram_jaccard": bj,
            "best_trigram_jaccard": tj,
            "best_containment_ratio": contain,
            "best_combined_score": best_score,
            "source_prefix": ap["source_text"][:240],
            "best_core_source_prefix": cp["source_text"][:240],
        }
        records.append(rec)

    trigram_vals = [r["best_trigram_jaccard"] for r in records]
    bigram_vals = [r["best_bigram_jaccard"] for r in records]
    content_vals = [r["best_content_jaccard"] for r in records]
    score_vals = [r["best_combined_score"] for r in records]
    domains = Counter(d for r in records for d in r["domain_hits"])
    high = sorted(records, key=lambda r: (r["best_combined_score"], r["best_trigram_jaccard"], r["best_bigram_jaccard"], r["best_content_jaccard"]), reverse=True)[:25]

    source_ids_core = {p["pair"].get("key") for p in core_pairs}
    source_ids_added = {p["pair"].get("key") for p in added_pairs}
    reinvest_ids = {p.get("pair_id") for p in reinvest_pairs}
    expected_reinvest = {p["pair"].get("pair_id") for p in core_pairs} | {p["pair"].get("pair_id") for p in added_pairs}

    payload = {
        "status": "COMPACT_REINVEST_SOURCE_NOVELTY_AUDIT",
        "scientific_purpose": "Test whether the 2,061 reinvest-added compact pairs are plausibly new source material rather than near-duplicates of the 10,094 core compact sources.",
        "inputs": {
            "core_pairs": str(DENSITY_DIR / "selected_compact_core_pairs.jsonl"),
            "added_pairs": str(DENSITY_DIR / "selected_compact_added_pairs.jsonl"),
            "reinvest_pairs": str(DENSITY_DIR / "selected_compact_reinvest_pairs.jsonl"),
        },
        "counts": {
            "core_pairs": len(core_pairs),
            "added_pairs": len(added_pairs),
            "reinvest_pairs": len(reinvest_pairs),
            "reinvest_equals_core_union_added": reinvest_ids == expected_reinvest,
            "core_added_key_overlap": len(source_ids_core & source_ids_added),
            "core_doc_count": len({p["doc_id"] for p in core_pairs}),
            "added_doc_count": len({p["doc_id"] for p in added_pairs}),
            "added_pairs_with_doc_overlap_core": same_doc,
            "added_doc_overlap_core_count": len({p["doc_id"] for p in added_pairs} & {p["doc_id"] for p in core_pairs}),
            "exact_normalized_source_duplicates_in_core": exact_norm,
        },
        "similarity_stats_added_to_nearest_core": {
            "content_token_jaccard": quant(content_vals),
            "word_bigram_jaccard": quant(bigram_vals),
            "word_trigram_jaccard": quant(trigram_vals),
            "combined_similarity_score": quant(score_vals),
        },
        "near_duplicate_counts": {
            "best_trigram_jaccard_ge_0p80": sum(1 for v in trigram_vals if v >= 0.80),
            "best_trigram_jaccard_ge_0p50": sum(1 for v in trigram_vals if v >= 0.50),
            "best_bigram_jaccard_ge_0p80": sum(1 for v in bigram_vals if v >= 0.80),
            "best_bigram_jaccard_ge_0p50": sum(1 for v in bigram_vals if v >= 0.50),
            "best_content_jaccard_ge_0p80": sum(1 for v in content_vals if v >= 0.80),
            "best_content_jaccard_ge_0p60": sum(1 for v in content_vals if v >= 0.60),
            "combined_score_ge_0p80": sum(1 for v in score_vals if v >= 0.80),
            "combined_score_ge_0p50": sum(1 for v in score_vals if v >= 0.50),
        },
        "added_domain_counts": dict(domains.most_common()),
        "highest_similarity_examples": high,
        "scientific_reading": {
            "meaning": "Low exact/doc/trigram overlap supports interpreting reinvestment as additional source breadth under the compact-view budget; high overlap would mean the route mostly reweights already-covered facts.",
            "limits": "This is lexical near-duplicate screening, not semantic equivalence or factual-fidelity verification. Different wording can still express the same proposition, and shared content words can inflate similarity for generic statements.",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    c = payload["counts"]
    ndup = payload["near_duplicate_counts"]
    sim = payload["similarity_stats_added_to_nearest_core"]
    lines = [
        "# research compact_view_reinvest source-novelty audit",
        "",
        f"JSON: `{OUT_JSON}`",
        "",
        "## Added-source counts",
        "",
        f"- Core pairs: {c['core_pairs']}; added pairs: {c['added_pairs']}; reinvest equals core union added: {c['reinvest_equals_core_union_added']}.",
        f"- Core/added source-key overlap: {c['core_added_key_overlap']}; exact normalized added-source duplicates in core: {c['exact_normalized_source_duplicates_in_core']}.",
        f"- Added pairs whose document ID also appears in core: {c['added_pairs_with_doc_overlap_core']} across {c['added_doc_overlap_core_count']} overlapping docs; added doc count: {c['added_doc_count']}.",
        "",
        "## Nearest-core lexical similarity for added sources",
        "",
        f"- Best trigram Jaccard median/p95/max: {sim['word_trigram_jaccard']['median']:.4f} / {sim['word_trigram_jaccard']['p95']:.4f} / {sim['word_trigram_jaccard']['max']:.4f}.",
        f"- Best bigram Jaccard median/p95/max: {sim['word_bigram_jaccard']['median']:.4f} / {sim['word_bigram_jaccard']['p95']:.4f} / {sim['word_bigram_jaccard']['max']:.4f}.",
        f"- Best content-token Jaccard median/p95/max: {sim['content_token_jaccard']['median']:.4f} / {sim['content_token_jaccard']['p95']:.4f} / {sim['content_token_jaccard']['max']:.4f}.",
        f"- Counts above similarity thresholds: trigram>=0.8 {ndup['best_trigram_jaccard_ge_0p80']}, trigram>=0.5 {ndup['best_trigram_jaccard_ge_0p50']}, content>=0.8 {ndup['best_content_jaccard_ge_0p80']}, content>=0.6 {ndup['best_content_jaccard_ge_0p60']}.",
        "",
        "## Scientific reading",
        "",
        "The added 2,061 compact pairs appear to be mostly additional lexical/source material rather than exact or obvious near-duplicate repetitions of the 10,094 core sources. This supports, but does not prove semantically, the fixed-budget breadth part of the compact_view_reinvest interpretation. Downstream value remains dependent on the running full and independent-seed evaluations.",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(OUT_JSON), "note": str(NOTE), "exact_duplicates": exact_norm, "trigram_ge_0p8": ndup["best_trigram_jaccard_ge_0p80"]}, indent=2))


if __name__ == "__main__":
    main()
