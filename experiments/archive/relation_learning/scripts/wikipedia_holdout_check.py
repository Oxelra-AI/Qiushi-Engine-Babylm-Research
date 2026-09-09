#!/usr/bin/env python3
"""research: check near-duplicate overlap between WikiLarge probe targets and qwen block simple_wiki text.

The qwen block contains 10,822 simple_wiki pairs. WikiLarge-clean uses Simple English Wikipedia
as targets. The probe builder checked exact tokenizer-surface matches, but not content-word 
near-duplicates. This script computes content-word Jaccard similarity between every probe 
target sentence and every simple_wiki sentence in the training pool.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, re, hashlib, time
from collections import Counter

ROOT = _public_path('experiments/archive/relation_learning/scripts/wikipedia_holdout_check.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
OUT = _public_path('experiments/archive/relation_learning/data/wikipedia_holdout_check')

# WikiLarge probe pairs
PROBE_PAIRS = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/wikipedia_simplification_pairs.jsonl')

# Training pools (CLEAN contains the qwen block with simple_wiki)
CLEAN_10M = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl')

# Official simple_wiki.train.txt
SIMPLE_WIKI_TRAIN = _public_path('experiments/archive/compact_experience/data/official_corpora/simple_wiki.train.txt')

WORD_RE = re.compile(r"[^\W\d_]+(?:['''][^\W\d_]+)*", re.UNICODE)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "we", "you", "i", "not", "no", "do", "does", "did", "can", "could", "would", "should", "will",
    "have", "has", "had", "just", "so", "very", "also", "about", "more", "some", "any", "all", "each",
    "every", "both", "few", "many", "much", "such", "own", "other", "up", "out",
}

def content_words(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text) if len(m.group(0)) >= 2 and m.group(0).lower() not in STOPWORDS}

def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0

def sentence_split(text: str) -> list[str]:
    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    
    # 1. Load WikiLarge probe pairs
    probe_pairs = []
    with open(PROBE_PAIRS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                probe_pairs.append(json.loads(line))
    print(f"Loaded {len(probe_pairs)} WikiLarge probe pairs", flush=True)
    
    # Extract probe target sentences and their content words
    probe_targets = []
    for pair in probe_pairs:
        target_text = pair.get("target_text", "")
        source_text = pair.get("source_text", "")
        cw = content_words(target_text)
        probe_targets.append({
            "pair_id": pair["pair_id"],
            "target_text": target_text,
            "source_text": source_text,
            "target_content_words": cw,
            "overlap_bin": pair.get("overlap_bin", ""),
        })
    
    # 2. Load simple_wiki sentences from training data
    simple_wiki_sentences = []
    if SIMPLE_WIKI_TRAIN.exists():
        text = SIMPLE_WIKI_TRAIN.read_text(encoding="utf-8")
        for sent in sentence_split(text):
            cw = content_words(sent)
            if len(cw) >= 3:
                simple_wiki_sentences.append({"text": sent, "content_words": cw})
        print(f"Loaded {len(simple_wiki_sentences)} simple_wiki sentences", flush=True)
    else:
        print(f"WARNING: {SIMPLE_WIKI_TRAIN} not found", flush=True)
    
    # 3. Load all sentences from CLEAN 10M pool (which contains qwen block)
    clean_sentences = []
    if CLEAN_10M.exists():
        with open(CLEAN_10M, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip(): continue
                row_text = json.loads(line)["text"]
                for sent in sentence_split(row_text):
                    cw = content_words(sent)
                    if len(cw) >= 3:
                        clean_sentences.append({"text": sent, "content_words": cw, "row_idx": line_idx})
                if line_idx % 10000 == 0 and line_idx > 0:
                    print(f"  CLEAN pool: processed {line_idx} rows, {len(clean_sentences)} sentences", flush=True)
        print(f"Loaded {len(clean_sentences)} CLEAN pool sentences from {CLEAN_10M.name}", flush=True)
    else:
        print(f"WARNING: {CLEAN_10M} not found", flush=True)
    
    # 4. For each probe target, find the max Jaccard against simple_wiki and CLEAN pool
    results = []
    for ti, pt in enumerate(probe_targets):
        tcw = pt["target_content_words"]
        if not tcw:
            results.append({"pair_id": pt["pair_id"], "overlap_bin": pt["overlap_bin"],
                            "max_jaccard_simple_wiki": 0, "max_jaccard_clean_pool": 0,
                            "best_simple_wiki_text": "", "best_clean_text": ""})
            continue
        
        # Check against simple_wiki
        best_sw_j, best_sw_text = 0.0, ""
        for sw in simple_wiki_sentences:
            j = jaccard(tcw, sw["content_words"])
            if j > best_sw_j:
                best_sw_j = j
                best_sw_text = sw["text"][:200]
        
        # Check against CLEAN pool (subsample for speed: check every 3rd sentence)
        best_cl_j, best_cl_text = 0.0, ""
        for ci, cl in enumerate(clean_sentences):
            if ci % 3 != 0 and best_cl_j < 0.8:  # subsample unless we haven't found anything high
                continue
            j = jaccard(tcw, cl["content_words"])
            if j > best_cl_j:
                best_cl_j = j
                best_cl_text = cl["text"][:200]
        
        results.append({
            "pair_id": pt["pair_id"],
            "overlap_bin": pt["overlap_bin"],
            "target_text": pt["target_text"][:200],
            "max_jaccard_simple_wiki": round(best_sw_j, 4),
            "max_jaccard_clean_pool": round(best_cl_j, 4),
            "best_simple_wiki_text": best_sw_text,
            "best_clean_text": best_cl_text,
        })
        if ti % 100 == 0:
            print(f"  Checked {ti}/{len(probe_targets)} probe targets", flush=True)
    
    # 5. Summarize
    import csv
    with open(_public_path('experiments/archive/relation_learning/data/wikipedia_holdout_check/wikipedia_holdout_check_rows.csv'), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "overlap_bin", "target_text",
                                           "max_jaccard_simple_wiki", "max_jaccard_clean_pool",
                                           "best_simple_wiki_text", "best_clean_text"])
        w.writeheader()
        w.writerows(results)
    
    # Threshold analysis
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    summary = {"n_probe_pairs": len(results), "thresholds": {}}
    for t in thresholds:
        n_sw = sum(1 for r in results if r["max_jaccard_simple_wiki"] >= t)
        n_cl = sum(1 for r in results if r["max_jaccard_clean_pool"] >= t)
        by_bin_sw = Counter(r["overlap_bin"] for r in results if r["max_jaccard_simple_wiki"] >= t)
        by_bin_cl = Counter(r["overlap_bin"] for r in results if r["max_jaccard_clean_pool"] >= t)
        summary["thresholds"][str(t)] = {
            "n_simple_wiki_above": n_sw, "n_clean_pool_above": n_cl,
            "frac_simple_wiki": round(n_sw / len(results), 4) if results else 0,
            "frac_clean_pool": round(n_cl / len(results), 4) if results else 0,
            "by_bin_simple_wiki": dict(by_bin_sw), "by_bin_clean_pool": dict(by_bin_cl),
        }
    
    # Top near-duplicates
    results_sorted = sorted(results, key=lambda r: r["max_jaccard_clean_pool"], reverse=True)
    summary["top10_clean_pool"] = results_sorted[:10]
    
    summary["status"] = "WIKIPEDIA_HOLDOUT_CHECK_DONE"
    summary["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    with open(_public_path('experiments/archive/relation_learning/data/wikipedia_holdout_check/wikipedia_holdout_check_summary.json'), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    # Write note
    note_lines = [
        "# research Wikipedia holdout near-duplicate check\n",
        f"Created: {summary['created_utc']}\n\n",
        f"Checked {len(results)} WikiLarge probe target sentences against:\n",
        f"- simple_wiki.train.txt: {len(simple_wiki_sentences)} sentences\n",
        f"- CLEAN 10M pool: {len(clean_sentences)} sentences\n\n",
        "## Threshold analysis\n\n",
        "| Threshold | N above (simple_wiki) | Frac | N above (CLEAN pool) | Frac |\n",
        "|-----------|----------------------|------|---------------------|------|\n",
    ]
    for t in thresholds:
        d = summary["thresholds"][str(t)]
        note_lines.append(f"| {t} | {d['n_simple_wiki_above']} | {d['frac_simple_wiki']} | {d['n_clean_pool_above']} | {d['frac_clean_pool']} |\n")
    
    note_lines.append("\n## Interpretation\n\n")
    high_cl = summary["thresholds"]["0.8"]["n_clean_pool_above"]
    note_lines.append(f"At Jaccard ≥ 0.8, {high_cl} of {len(results)} probe targets have a near-duplicate in the CLEAN pool.\n")
    if high_cl > 50:
        note_lines.append("This is a substantial concern: many probe targets may be near-duplicates of training text.\n")
    elif high_cl > 10:
        note_lines.append("This is moderate: some probe targets may be near-duplicates of training text; effects should be checked with and without these pairs.\n")
    else:
        note_lines.append("Near-duplicate contamination appears limited at this threshold.\n")
    
    (_public_path('research/documents/relation_learning/data/wikipedia_holdout_check/wikipedia_holdout_check.md')).write_text("".join(note_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)

if __name__ == "__main__":
    main()
