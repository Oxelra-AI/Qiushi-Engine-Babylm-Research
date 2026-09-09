#!/usr/bin/env python3
"""Audit feasibility of a one-pair contextual same-window clean-Qwen arm.

This script does not train and does not create a final corpus.  It checks whether the selected
research clean-Qwen pairs can be placed back into their official 160-word row context, one pair
per row, using only training-corpus artifacts:
  - official_pool.jsonl;
  - selected_pairs.jsonl;
  - rewrite source/prompt metadata with example_id and sentence_idx.

It does not read official AoA/CDI words, child curves, AoA predictions, AoA scores, or downstream
evaluation outputs.  The purpose is to decide whether a scientifically honest contextualized-pair
materializer can be built later, rather than falsely labeling same-source filler as natural context.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import collections
import json
import pathlib
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

ROOT = _public_path('experiments/archive/compact_experience')
OFFICIAL = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
SELECTED = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
PROMPTS = _public_path('experiments/archive/compact_experience/data/qwen_aligned/rewrite_prompts.jsonl')
EXTRA_SHARDS = [
    _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences_shard0.jsonl'),
    _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences_shard1.jsonl'),
]
OUT = _public_path('experiments/archive/compact_experience/data/contextual_one_pair_audit/contextual_one_pair_feasibility.json')
NOTE = _public_path('research/notes/compact_experience/contextual_one_pair_audit.md')
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
CAPS = [80, 120, 160]


def norm_ws(s: str) -> str:
    return " ".join((s or "").replace("\u00a0", " ").split())


def norm_tok(w: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", w.lower())


def split_sentences(text: str) -> List[str]:
    return [norm_ws(x) for x in SENT_SPLIT.split(norm_ws(text)) if norm_ws(x)]


def stats(xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        i = p * (len(ys) - 1)
        lo = int(i)
        hi = min(lo + 1, len(ys) - 1)
        a = i - lo
        return ys[lo] * (1 - a) + ys[hi] * a
    return {
        "n": len(ys),
        "min": ys[0],
        "p05": q(0.05),
        "mean": statistics.fmean(ys),
        "median": q(0.5),
        "p95": q(0.95),
        "max": ys[-1],
    }


def load_official() -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with OFFICIAL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[int(row["example_id"])] = row
    return out


def load_sentence_idx() -> Dict[str, int]:
    idx: Dict[str, int] = {}
    if PROMPTS.exists():
        with PROMPTS.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "source_sentence_idx" in r:
                        idx[str(r["id"])] = int(r["source_sentence_idx"])
    for p in EXTRA_SHARDS:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if "sentence_idx" in r:
                            idx[str(r["id"])] = int(r["sentence_idx"])
    return idx


def find_subseq(row_words: List[str], sent_words: List[str], normalized: bool = False) -> Optional[Tuple[int, int]]:
    if not sent_words:
        return None
    if normalized:
        rw = [norm_tok(x) for x in row_words]
        sw = [norm_tok(x) for x in sent_words]
    else:
        rw, sw = row_words, sent_words
    n = len(sw)
    for i in range(0, len(rw) - n + 1):
        if rw[i:i+n] == sw:
            return i, i + n
    return None


def locate_pair(pair: Dict[str, Any], official: Dict[int, Dict[str, Any]], sent_idx: Dict[str, int]) -> Dict[str, Any]:
    pid = str(pair["pair_id"])
    exid = int(pair["example_id"])
    original = norm_ws(str(pair["original"]))
    row = official.get(exid)
    rec: Dict[str, Any] = {"pair_id": pid, "example_id": exid, "source": pair.get("source"), "cohort": pair.get("cohort")}
    if row is None:
        rec.update({"located": False, "reason": "missing_example_id"})
        return rec
    row_text = norm_ws(row["text"])
    row_words = row_text.split()
    sents = split_sentences(row_text)
    idx = sent_idx.get(pid)
    rec["metadata_sentence_idx"] = idx
    rec["num_split_sentences"] = len(sents)

    # Prefer the recorded sentence index when it exactly matches; otherwise search exact sentence text.
    sent_start_words = 0
    chosen_j: Optional[int] = None
    if idx is not None and 0 <= idx < len(sents) and norm_ws(sents[idx]) == original:
        chosen_j = idx
        sent_start_words = sum(len(s.split()) for s in sents[:idx])
    else:
        for j, s in enumerate(sents):
            if norm_ws(s) == original:
                chosen_j = j
                sent_start_words = sum(len(x.split()) for x in sents[:j])
                break
    if chosen_j is not None:
        sent_words = sents[chosen_j].split()
        start, end = sent_start_words, sent_start_words + len(sent_words)
        if row_words[start:end] != sent_words:
            # Fallback to exact subsequence if simple sentence accumulation drifted.
            loc = find_subseq(row_words, sent_words, normalized=False) or find_subseq(row_words, sent_words, normalized=True)
            if loc is not None:
                start, end = loc
        rec.update({"located": True, "method": "sentence_split", "sentence_idx_used": chosen_j, "start_word": start, "end_word": end})
    else:
        sent_words = original.split()
        loc = find_subseq(row_words, sent_words, normalized=False)
        method = "exact_word_subsequence"
        if loc is None:
            loc = find_subseq(row_words, sent_words, normalized=True)
            method = "normalized_word_subsequence"
        if loc is None:
            rec.update({"located": False, "reason": "original_not_found_in_official_row"})
            return rec
        start, end = loc
        rec.update({"located": True, "method": method, "sentence_idx_used": None, "start_word": start, "end_word": end})

    rec["row_words"] = len(row_words)
    rec["original_words_check"] = int(pair.get("original_words", len(original.split())))
    rec["rewrite_words"] = int(pair.get("rewrite_words", len(str(pair.get("rewrite", "")).split())))
    rec["pair_words"] = int(pair.get("pair_words", rec["original_words_check"] + rec["rewrite_words"]))
    rec["left_available_words"] = int(rec["start_word"])
    rec["right_available_words"] = len(row_words) - int(rec["end_word"])
    # Estimate contextual row geometry for several caps, filling from both sides and borrowing unused capacity.
    cap_rows: Dict[str, Dict[str, int]] = {}
    for cap in CAPS:
        pair_words = rec["pair_words"]
        target = max(pair_words, min(cap, 160))
        capacity = max(0, target - pair_words)
        left_want = capacity // 2
        right_want = capacity - left_want
        left_take = min(rec["left_available_words"], left_want)
        right_take = min(rec["right_available_words"], right_want)
        # Borrow unused capacity from the other side within the same 160-word row.
        leftover = capacity - left_take - right_take
        if leftover > 0:
            add_left = min(rec["left_available_words"] - left_take, leftover)
            left_take += add_left
            leftover -= add_left
        if leftover > 0:
            add_right = min(rec["right_available_words"] - right_take, leftover)
            right_take += add_right
            leftover -= add_right
        cap_rows[str(cap)] = {"cap": cap, "row_words_est": pair_words + left_take + right_take, "context_words": left_take + right_take, "left_take": left_take, "right_take": right_take}
    rec["cap_estimates"] = cap_rows
    return rec


def main() -> None:
    official = load_official()
    sent_idx = load_sentence_idx()
    rows: List[Dict[str, Any]] = []
    with SELECTED.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(locate_pair(json.loads(line), official, sent_idx))

    located = [r for r in rows if r.get("located")]
    failed = [r for r in rows if not r.get("located")]
    by_cohort = collections.defaultdict(lambda: {"n": 0, "located": 0})
    by_source = collections.defaultdict(lambda: {"n": 0, "located": 0})
    methods = collections.Counter()
    for r in rows:
        c = str(r.get("cohort"))
        s = str(r.get("source"))
        by_cohort[c]["n"] += 1
        by_source[s]["n"] += 1
        if r.get("located"):
            by_cohort[c]["located"] += 1
            by_source[s]["located"] += 1
            methods[str(r.get("method"))] += 1

    cap_summary = {}
    for cap in CAPS:
        key = str(cap)
        cap_summary[key] = {
            "row_words_est": stats([float(r["cap_estimates"][key]["row_words_est"]) for r in located]),
            "context_words": stats([float(r["cap_estimates"][key]["context_words"]) for r in located]),
            "total_context_words_if_all_located": sum(int(r["cap_estimates"][key]["context_words"]) for r in located),
            "total_contextual_pair_row_words_if_all_located": sum(int(r["cap_estimates"][key]["row_words_est"]) for r in located),
        }

    payload = {
        "status": "CONTEXTUAL_ONE_PAIR_FEASIBILITY_AUDIT",
        "non_leakage_statement": "Uses only official training pool and research/028 Qwen source metadata; no official AoA/CDI words, child curves, AoA predictions, AoA scores, or downstream eval outputs are read.",
        "selected_pairs": len(rows),
        "located_pairs": len(located),
        "failed_pairs": len(failed),
        "located_fraction": len(located) / len(rows) if rows else 0.0,
        "methods": dict(methods),
        "by_cohort": {k: {**v, "located_fraction": v["located"] / v["n"] if v["n"] else 0.0} for k, v in sorted(by_cohort.items())},
        "by_source": {k: {**v, "located_fraction": v["located"] / v["n"] if v["n"] else 0.0} for k, v in sorted(by_source.items())},
        "left_available_words": stats([float(r["left_available_words"]) for r in located]),
        "right_available_words": stats([float(r["right_available_words"]) for r in located]),
        "pair_words": stats([float(r["pair_words"]) for r in located]),
        "cap_summary": cap_summary,
        "failed_examples": failed[:50],
        "interpretation": "If located_fraction is high, a later materializer can build an honest one-pair contextual arm using real official row context. If low/ambiguous, do not call same-source filler natural context; repair matching first or restrict to high-confidence located pairs with matched controls.",
    }
    _public_path('experiments/archive/compact_experience/data/contextual_one_pair_audit').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research contextual one-pair feasibility audit",
        "",
        payload["non_leakage_statement"],
        "",
        f"Located pairs: {len(located)} / {len(rows)} ({payload['located_fraction']:.4f})",
        f"Methods: {dict(methods)}",
        "",
        "## By cohort",
    ]
    for k, v in payload["by_cohort"].items():
        lines.append(f"- {k}: {v['located']} / {v['n']} ({v['located_fraction']:.4f})")
    lines += ["", "## Cap summaries"]
    for cap, s in cap_summary.items():
        lines.append(f"- cap {cap}: row_words_mean={s['row_words_est'].get('mean')} context_words_mean={s['context_words'].get('mean')} total_context_words={s['total_context_words_if_all_located']}")
    lines += ["", "Output JSON: `" + str(OUT) + "`"]
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "located_fraction": payload["located_fraction"], "methods": dict(methods)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
