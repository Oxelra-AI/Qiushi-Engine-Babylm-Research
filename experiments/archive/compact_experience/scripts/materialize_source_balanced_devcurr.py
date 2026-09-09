#!/usr/bin/env python3
"""research follow-up: source-balanced AoA-safe first-pass curricula.

This builds two matched first-pass-order variants over the exact research clean-Qwen
10M multiset:

  1. source_balanced_easy_to_hard
  2. source_balanced_hard_to_easy

Both keep every row and word exactly once per pass and use the original clean-Qwen
order for passes 2--10.  The first pass is divided into 10 cumulative word bins, each
approximately preserving the corpus source mixture.  Within each source and bin, rows
are ordered by a corpus-only difficulty proxy.  This separates a lexical/developmental
difficulty direction from research's stronger source-block schedule.

No official AoA/CDI words, child curves, AoA predictions, or AoA scores are read or
used.  It is a predeclared legal training-side manipulation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import math
import pathlib
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

ROOT = _public_path('experiments/archive/compact_experience')
IN_10M = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
IN_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
OUT_DIR = _public_path('experiments/archive/compact_experience/data/source_balanced_devcurr')
OUT_META = _public_path('experiments/archive/compact_experience/data/source_balanced_devcurr/source_balanced_devcurr_metadata.json')
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")

@dataclass
class Row:
    index: int
    obj: Dict[str, Any]
    words: int
    source: str
    norm_words: List[str]
    difficulty: float = 0.0
    feature: Dict[str, float] | None = None


def nw(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_digest(obj: Dict[str, Any]) -> str:
    s = json.dumps({"text": obj.get("text"), "words": obj.get("words"), "example_id": obj.get("example_id"), "source": obj.get("source")}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def multiset_digest(objs: Iterable[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for d in sorted(row_digest(o) for o in objs):
        h.update(d.encode("ascii")); h.update(b"\n")
    return h.hexdigest()


def assign_features(rows: List[Row]) -> None:
    freq: Counter[str] = Counter()
    for r in rows:
        freq.update(r.norm_words)
    fvals = sorted(freq.values())
    rare_cut = fvals[max(0, int(0.20 * (len(fvals) - 1)))] if fvals else 1
    # raw features: lower mean log freq, higher rare frac, longer words, higher TTR = harder.
    raws: Dict[str, List[float]] = {"neg_mean_log_freq": [], "rare_frac": [], "avg_chars": [], "ttr": [], "words_norm": []}
    feats: List[Dict[str, float]] = []
    for r in rows:
        ws = r.norm_words
        if ws:
            mean_log = sum(math.log1p(freq[w]) for w in ws) / len(ws)
            rare = sum(1 for w in ws if freq[w] <= rare_cut) / len(ws)
            avg_chars = sum(len(w) for w in ws) / len(ws)
            ttr = len(set(ws)) / len(ws)
        else:
            mean_log, rare, avg_chars, ttr = 0.0, 1.0, 10.0, 1.0
        feat = {
            "neg_mean_log_freq": -mean_log,
            "rare_frac": rare,
            "avg_chars": avg_chars,
            "ttr": ttr,
            "words_norm": r.words / 160.0,
        }
        feats.append(feat)
        for k, v in feat.items():
            raws[k].append(v)
    means = {k: sum(vs)/len(vs) for k, vs in raws.items()}
    stds = {k: (sum((x-means[k])**2 for x in vs)/max(1, len(vs)-1))**0.5 or 1.0 for k, vs in raws.items()}
    # Fixed prospective weights.  Do not tune on official AoA.
    weights = {"neg_mean_log_freq": 1.0, "rare_frac": 0.8, "avg_chars": 0.4, "ttr": 0.3, "words_norm": 0.1}
    for r, feat in zip(rows, feats):
        zsum = 0.0
        for k, w in weights.items():
            zsum += w * ((feat[k] - means[k]) / stds[k])
        r.difficulty = zsum
        r.feature = feat


def source_words(rows: Iterable[Row]) -> Dict[str, int]:
    d: Dict[str, int] = defaultdict(int)
    for r in rows:
        d[r.source] += r.words
    return dict(sorted(d.items()))


def exact_source_targets(total_by_source: Dict[str, int], bins: int = 10) -> List[Dict[str, int]]:
    """Per-bin source targets that sum exactly to source totals.

    Uses floor plus largest remainders in word units.  Row atomicity means realized bins
    are approximate, but each source's full list is consumed exactly once.
    """
    targets = [dict() for _ in range(bins)]
    for src, total in total_by_source.items():
        base = total // bins
        rem = total - base * bins
        for i in range(bins):
            targets[i][src] = base + (1 if i < rem else 0)
    return targets


def build_source_balanced_order(rows: List[Row], easy_first: bool) -> List[Row]:
    by_source: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_source[r.source].append(r)
    for src in by_source:
        by_source[src].sort(key=lambda r: (r.difficulty, r.index), reverse=not easy_first)
    queues = {src: deque(lst) for src, lst in by_source.items()}
    total_by_source = source_words(rows)
    targets = exact_source_targets(total_by_source, bins=10)
    sources = sorted(by_source)
    out: List[Row] = []
    for bi, tgt in enumerate(targets):
        # Within a bin, cycle sources by remaining source deficit. This avoids huge blocks
        # while approximately hitting source word proportions. Rows remain atomic.
        used = {src: 0 for src in sources}
        while True:
            candidates = []
            for src in sources:
                if queues[src]:
                    deficit = tgt.get(src, 0) - used[src]
                    candidates.append((deficit, src))
            if not candidates:
                break
            # Stop bin once all targets are met or only sources with exhausted queues remain.
            if all(deficit <= 0 for deficit, _ in candidates):
                break
            candidates.sort(key=lambda x: (-x[0], x[1]))
            _, src = candidates[0]
            r = queues[src].popleft()
            out.append(r)
            used[src] += r.words
        # If a source was under target due to queue shape, later bins will consume leftovers.
    # Append any leftovers in balanced source cycling order.
    while any(queues[src] for src in sources):
        remaining_words = {src: sum(r.words for r in queues[src]) for src in sources}
        src = max((s for s in sources if queues[s]), key=lambda s: (remaining_words[s], -sources.index(s)))
        out.append(queues[src].popleft())
    if len(out) != len(rows):
        raise RuntimeError(f"order row count mismatch {len(out)} vs {len(rows)}")
    return out


def first_bins(rows: List[Row], bin_words: int = 1_000_000, max_bins: int = 10) -> List[Dict[str, Any]]:
    bins = []
    start = 0
    cur: Dict[str, int] = defaultdict(int)
    cur_words = 0
    b = 1
    for r in rows:
        cur[r.source] += r.words
        cur_words += r.words
        if cur_words >= bin_words and b <= max_bins:
            bins.append({"bin_index": b, "row_atomic_words": cur_words, "source_words": dict(sorted(cur.items()))})
            cur = defaultdict(int)
            cur_words = 0
            b += 1
        if b > max_bins:
            break
    return bins


def write_variant(name: str, ordered: List[Row], original_objs: List[Dict[str, Any]]) -> Dict[str, Any]:
    p10 = OUT_DIR / f"{name}_10M.jsonl"
    p100 = OUT_DIR / f"{name}_100M.jsonl"
    total = 0
    with p10.open("w", encoding="utf-8") as f:
        for r in ordered:
            total += r.words
            f.write(json.dumps(r.obj, ensure_ascii=False) + "\n")
    if total != 10_000_000:
        raise RuntimeError(f"{name} 10M mismatch: {total}")
    with p100.open("w", encoding="utf-8") as f:
        total100 = 0
        # First pass ordered, later passes original order.
        for r in ordered:
            total100 += r.words
            f.write(json.dumps(r.obj, ensure_ascii=False) + "\n")
        for _ in range(9):
            for obj in original_objs:
                total100 += int(obj["words"])
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if total100 != 100_000_000:
        raise RuntimeError(f"{name} 100M mismatch: {total100}")
    return {"name": name, "path_10M": str(p10), "path_100M": str(p100), "sha256_10M": sha_file(p10), "sha256_100M": sha_file(p100), "first_bins": first_bins(ordered)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(IN_META.read_text(encoding="utf-8"))
    if meta.get("status") != "CLEAN_QWEN_CORPORA_MATERIALIZED":
        raise SystemExit("unexpected input metadata status")
    rows: List[Row] = []
    objs: List[Dict[str, Any]] = []
    total = 0
    with IN_10M.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            if len(obj["text"].split()) != int(obj["words"]):
                raise RuntimeError(f"word mismatch row {i}")
            rows.append(Row(i, obj, int(obj["words"]), str(obj.get("source", "unknown")), nw(obj["text"])))
            objs.append(obj)
            total += int(obj["words"])
    if total != 10_000_000:
        raise RuntimeError(f"input word total mismatch: {total}")
    assign_features(rows)
    original_digest = multiset_digest(objs)
    easy = build_source_balanced_order(rows, easy_first=True)
    hard = build_source_balanced_order(rows, easy_first=False)
    if multiset_digest([r.obj for r in easy]) != original_digest or multiset_digest([r.obj for r in hard]) != original_digest:
        raise RuntimeError("multiset digest changed")
    v_easy = write_variant("qwen_sourcebalanced_easy_firstpass", easy, objs)
    v_hard = write_variant("qwen_sourcebalanced_hard_firstpass", hard, objs)
    payload = {
        "status": "SOURCE_BALANCED_DEVCURR_MATERIALIZED",
        "non_leakage_statement": "Uses only research training text, source labels, and corpus-internal lexical frequency/difficulty. It does not read official AoA/CDI evaluation words, AoA predictions, child curves, or AoA scores.",
        "input_10M": str(IN_10M),
        "input_metadata": str(IN_META),
        "word_total_10M": total,
        "row_count": len(rows),
        "row_multiset_digest": original_digest,
        "exact_row_multiset_preserved": True,
        "first_pass_then_nine_original_passes": True,
        "source_words": source_words(rows),
        "difficulty_proxy": "weighted z-score of -mean_log_corpus_freq, rare_frac, avg_word_chars, type_token_ratio, row_words/160; weights fixed before evaluation",
        "variants": [v_easy, v_hard],
        "intended_contrast": "easy_first vs hard_first under approximately source-balanced 1M bins; separates lexical/developmental direction from source-block timing in aoa_developmental_order",
        "baseline_runs": [
            "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022",
            "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43122"
        ]
    }
    OUT_META.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": str(OUT_META), "variants": [{"name": v["name"], "sha256_100M": v["sha256_100M"], "first_bin": v["first_bins"][0]} for v in payload["variants"]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
