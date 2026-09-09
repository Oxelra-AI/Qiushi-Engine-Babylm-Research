#!/usr/bin/env python3
"""research: Validate dose-arm rewrite outputs.

Applies COMPACT_EXPERIENCE-equivalent quality filters plus improvements from the state-update
work. Produces clean pairs ready for packing into the dose-arm stream.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path.cwd()
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/dose_arm_validated"

# Filter thresholds (from COMPACT_EXPERIENCE research + improvements)
CONTENT_OVERLAP_MIN = 0.14
CONTENT_OVERLAP_MAX = 0.94
ENTITY_RECALL_LARGE = 0.75
ENTITY_RECALL_SMALL = 1.0
LEN_RATIO_MIN = 0.45
LEN_RATIO_MAX = 2.1
MIN_REWRITE_WORDS = 6
MAX_REWRITE_WORDS = 70
MAX_LCS = 5          # Reject if longest common subsequence ≥ 6 tokens
MAX_JACCARD = 0.80   # Reject if content-token Jaccard ≥ 0.80

WORD_RE = re.compile(r"\b[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)*\b")
NUM_RE = re.compile(r"\b\d[\d,.]*\b")
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "from", "with",
    "by", "as", "into", "onto", "over", "under", "through", "during", "after", "before", "within",
    "without", "near", "about", "around", "is", "are", "was", "were", "be", "been", "being", "has",
    "have", "had", "do", "does", "did", "this", "that", "these", "those", "his", "her", "their",
    "its", "our", "your", "my", "he", "she", "it", "they", "we", "you", "i", "not", "no", "than",
    "then", "there", "here", "which", "who", "whom", "whose", "what", "where", "when", "why", "how",
    "one", "two", "three", "so", "if",
}
TRANSCRIPT_MARKERS = {"*mot:", "*chi:", "*fat:", "*inv:", "*exp:", "*adu:", "*obs:", "@end", "@begin"}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def words(text: str) -> list[str]:
    return text.split()


def norm_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def content_tokens(text: str) -> set[str]:
    return {t for t in norm_tokens(text) if len(t) >= 3 and t not in STOPWORDS}


def entity_tokens(text: str) -> list[str]:
    """Extract capitalized multi-word entities and proper names."""
    ents = []
    raw = words(text)
    for i, w in enumerate(raw):
        s = w.strip("\"\"''.,!?;:()[]{}—–-")
        if not s:
            continue
        # Speaker labels like *MOT:
        if s.startswith("*") and len(s) > 2:
            ents.append(s.lower().rstrip(":"))
            continue
        letters = re.sub(r"[^A-Za-z]", "", s)
        if not letters:
            continue
        is_cap = letters[0].isupper() and len(letters) > 1
        has_internal_cap = any(c.isupper() for c in letters[1:])
        all_caps = len(letters) >= 2 and letters.isupper()
        if is_cap or has_internal_cap or all_caps:
            if i == 0 and not has_internal_cap and not all_caps and not s.startswith("*"):
                continue
            ents.append(s.lower())
    return ents


def number_tokens(text: str) -> list[str]:
    return [re.sub(r"\s+", "", m.group(0).lower()) for m in NUM_RE.finditer(text)]


def lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (token-level)."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    if m > 60 or n > 60:
        # Fast approximation for long sequences
        prev = [0] * (n + 1)
        for i in range(m):
            cur = [0] * (n + 1)
            for j in range(n):
                if a[i] == b[j]:
                    cur[j + 1] = prev[j] + 1
                else:
                    cur[j + 1] = max(cur[j], prev[j + 1])
            prev = cur
        return prev[n]
    prev = [0] * (n + 1)
    for i in range(m):
        cur = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]:
                cur[j + 1] = prev[j] + 1
            else:
                cur[j + 1] = max(cur[j], prev[j + 1])
        prev = cur
    return prev[n]


def validate_pair(source: str, rewrite: str) -> tuple[bool, str, dict]:
    """Validate a single source-rewrite pair. Returns (accepted, reason, stats)."""
    stats = {}
    src_words = words(source)
    rew_words = words(rewrite)
    src_n = len(src_words)
    rew_n = len(rew_words)
    stats["source_words"] = src_n
    stats["rewrite_words"] = rew_n

    # Basic length checks
    if rew_n < MIN_REWRITE_WORDS:
        return False, "too_short", stats
    if rew_n > MAX_REWRITE_WORDS:
        return False, "too_long", stats

    # Length ratio
    ratio = rew_n / max(1, src_n)
    stats["len_ratio"] = round(ratio, 4)
    if ratio < LEN_RATIO_MIN or ratio > LEN_RATIO_MAX:
        return False, "len_ratio", stats

    # Transcript marker check
    rew_lower = rewrite.lower()
    for marker in TRANSCRIPT_MARKERS:
        if marker in rew_lower:
            return False, "transcript_marker", stats

    # Meta prefix
    if rewrite.strip().startswith(("Here is", "Here's", "Sure", "I'll", "The rewritten")):
        return False, "meta_prefix", stats

    # Content overlap (Jaccard on content tokens)
    src_ct = content_tokens(source)
    rew_ct = content_tokens(rewrite)
    if not src_ct or not rew_ct:
        return False, "no_content_tokens", stats
    jaccard = len(src_ct & rew_ct) / len(src_ct | rew_ct) if (src_ct | rew_ct) else 0
    stats["content_jaccard"] = round(jaccard, 4)
    if jaccard < CONTENT_OVERLAP_MIN:
        return False, "low_content_overlap", stats
    if jaccard > CONTENT_OVERLAP_MAX:
        return False, "copy_overlap", stats

    # High Jaccard check (from state-update validator)
    if jaccard > MAX_JACCARD:
        return False, "high_jaccard", stats

    # LCS check (token-level)
    src_toks = norm_tokens(source)
    rew_toks = norm_tokens(rewrite)
    lcs = lcs_length(src_toks, rew_toks)
    stats["lcs"] = lcs
    if lcs > MAX_LCS:
        return False, "high_lcs", stats

    # Entity recall
    src_ents = entity_tokens(source)
    if src_ents:
        recalled = sum(1 for e in src_ents if e in rew_lower)
        recall = recalled / len(src_ents)
        stats["entity_recall"] = round(recall, 4)
        stats["entity_count"] = len(src_ents)
        threshold = ENTITY_RECALL_SMALL if len(src_ents) <= 2 else ENTITY_RECALL_LARGE
        if recall < threshold:
            return False, "entity_recall", stats

    # Number recall
    src_nums = number_tokens(source)
    if src_nums:
        rew_nums = number_tokens(rewrite)
        num_recalled = sum(1 for n in src_nums if n in rew_nums)
        stats["number_recall"] = round(num_recalled / len(src_nums), 4)
        if num_recalled < len(src_nums):
            return False, "number_mismatch", stats

    # Repetition within rewrite
    rew_toks_lower = [w.lower() for w in rew_words]
    if len(set(rew_toks_lower)) / max(1, rew_n) < 0.4:
        return False, "repetition", stats

    # Fragment/incomplete
    if rewrite.count("...") > 1:
        return False, "fragment", stats

    return True, "accepted", stats


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", required=True, help="Path to generation outputs JSONL")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--max-read-sample", type=int, default=50)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # Read outputs
    outputs_path = pathlib.Path(args.outputs)
    with outputs_path.open(encoding="utf-8") as f:
        outputs = [json.loads(line) for line in f if line.strip()]
    print(f"[LOAD] {len(outputs)} outputs from {outputs_path}", flush=True)

    # Validate each pair
    accepted = []
    rejected_reasons = Counter()
    fingerprints = set()
    all_stats = []
    for rec in outputs:
        rid = rec.get("id", "")
        source = rec.get("source_sentence", "")
        # Extract rewrite from output (first non-empty line after any prefix)
        raw_output = rec.get("output", "").strip()
        # Clean common prefixes
        rewrite = raw_output.split("\n")[0].strip()
        if rewrite.lower().startswith("rewritten:"):
            rewrite = rewrite[len("rewritten:"):].strip()

        # Fingerprint dedup
        fp = hashlib.md5(rewrite.lower().encode()).hexdigest()
        if fp in fingerprints:
            rejected_reasons["duplicate_output"] += 1
            continue
        fingerprints.add(fp)

        ok, reason, stats = validate_pair(source, rewrite)
        stats["id"] = rid
        stats["source_name"] = rec.get("source_name", "")
        all_stats.append(stats)

        if ok:
            accepted.append({
                "pair_id": rid,
                "source_sentence": source,
                "rewrite_sentence": rewrite,
                "source_words": stats["source_words"],
                "rewrite_words": stats["rewrite_words"],
                "content_jaccard": stats.get("content_jaccard", 0),
                "lcs": stats.get("lcs", 0),
                "entity_recall": stats.get("entity_recall"),
                "source_name": rec.get("source_name", ""),
                "source_example_id": rec.get("source_example_id"),
            })
        else:
            rejected_reasons[reason] += 1

    acceptance_rate = len(accepted) / max(1, len(outputs))
    print(f"[RESULT] Accepted: {len(accepted)}/{len(outputs)} ({acceptance_rate:.1%})", flush=True)
    print(f"[REJECTIONS] {dict(rejected_reasons)}", flush=True)

    # Write accepted pairs
    accepted_path = out_dir / "accepted_dose_arm_pairs.jsonl"
    with accepted_path.open("w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write manual read sample
    import random
    sample = random.Random(55).sample(accepted, min(args.max_read_sample, len(accepted)))
    sample_path = out_dir / "accepted_manual_read_sample.md"
    lines = [f"# Dose-arm accepted pairs sample (n={len(sample)})\n\n"]
    for i, s in enumerate(sample):
        lines.append(f"## [{i+1}] {s['pair_id']} ({s['source_name']}, {s['source_words']}→{s['rewrite_words']} words, J={s['content_jaccard']:.3f}, LCS={s['lcs']})\n")
        lines.append(f"**Source:** {s['source_sentence']}\n")
        lines.append(f"**Rewrite:** {s['rewrite_sentence']}\n\n")
    sample_path.write_text("".join(lines), encoding="utf-8")

    # Acceptance stats
    if accepted:
        jaccards = [a["content_jaccard"] for a in accepted]
        lcss = [a["lcs"] for a in accepted]
        acc_stats = {
            "jaccard_mean": round(statistics.mean(jaccards), 4),
            "jaccard_median": round(statistics.median(jaccards), 4),
            "jaccard_p95": round(sorted(jaccards)[int(0.95 * len(jaccards))], 4),
            "lcs_mean": round(statistics.mean(lcss), 2),
            "lcs_median": round(statistics.median(lcss), 2),
            "lcs_max": max(lcss),
        }
    else:
        acc_stats = {}

    elapsed = round(time.time() - started, 1)
    meta = {
        "status": "DOSE_ARM_VALIDATED",
        "created_utc": now_utc(),
        "elapsed_sec": elapsed,
        "total_outputs": len(outputs),
        "accepted": len(accepted),
        "acceptance_rate": round(acceptance_rate, 4),
        "rejection_reasons": dict(rejected_reasons),
        "acceptance_stats": acc_stats,
        "register_distribution": dict(Counter(a["source_name"] for a in accepted)),
        "outputs": {
            "accepted_pairs": str(accepted_path),
            "manual_sample": str(sample_path),
        },
    }
    meta_path = out_dir / "validation_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
