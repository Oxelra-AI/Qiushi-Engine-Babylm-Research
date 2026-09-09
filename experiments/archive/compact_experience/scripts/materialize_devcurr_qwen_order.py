#!/usr/bin/env python3
"""research: materialize an AoA-safe developmental-order clean-Qwen corpus.

Scientific purpose
------------------
The research correction made AoA a load-bearing official column.  This script builds
an intervention that is legal and mechanistically interpretable: keep the exact
research clean-Qwen 10M training multiset, tokenizer, architecture, and 10-pass word
budget, but change the *presentation order* of the first pass using only information
available from the training corpus itself:

  * official source labels (CHILDES/simple_wiki/dialogue/prose/...)
  * corpus-internal word frequency
  * row-level lexical/readability proxies computed from the training text

It never reads official AoA/CDI evaluation words or AoA scores.  The intervention is
therefore a developmental prior over experience order, not an evaluation leak.

Output
------
Creates a 10M sorted first-pass file and a 100M training file whose first pass is the
sorted developmental order and whose remaining 9 passes are the original research
clean-Qwen order.  This makes the AoA checkpoint ladder (1M..10M) meaningfully
order-sensitive while preserving most of the baseline final training trajectory.
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

ROOT = _public_path('experiments/archive/compact_experience')
IN_10M = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
IN_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
OUT_DIR = _public_path('experiments/archive/compact_experience/data/aoa_developmental_order')
OUT_10M = _public_path('experiments/archive/compact_experience/data/aoa_developmental_order/qwen_aligned_devcurr_firstpass_10M.jsonl')
OUT_100M = _public_path('experiments/archive/compact_experience/data/aoa_developmental_order/qwen_aligned_devcurr_firstpass_100M.jsonl')
OUT_ALLPASS_100M = _public_path('experiments/archive/compact_experience/data/aoa_developmental_order/qwen_aligned_devcurr_allpasses_100M.jsonl')
OUT_META = _public_path('experiments/archive/compact_experience/data/aoa_developmental_order/devcurr_materialization_metadata.json')

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")

# Lower rank = earlier.  This is a public/training-corpus prior, not tuned on AoA.
# Child-directed and simpler expository text come first; long literary prose comes last.
SOURCE_RANK = {
    "childes": 0,
    "simple_wiki": 1,
    "qwen_pair_packed": 2,
    "open_subtitles": 3,
    "switchboard": 4,
    "bnc_spoken": 5,
    "gutenberg": 6,
}

@dataclass
class Row:
    index: int
    obj: Dict[str, Any]
    text: str
    words: int
    source: str
    norm_words: List[str]
    mean_log_freq: float = 0.0
    rare_frac: float = 0.0
    avg_chars: float = 0.0
    type_token_ratio: float = 0.0
    rank_key: tuple = ()


def norm_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_digest(row: Dict[str, Any]) -> str:
    # Hash the semantic row identity independent of JSON key order.  This makes a
    # multiset-preservation check robust to different output ordering.
    s = json.dumps({"text": row.get("text"), "words": row.get("words"), "example_id": row.get("example_id"), "source": row.get("source")}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def multiset_digest(rows: Iterable[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for d in sorted(row_digest(r) for r in rows):
        h.update(d.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def stats(vals: List[float]) -> Dict[str, float | int | None]:
    if not vals:
        return {"n": 0, "min": None, "p05": None, "mean": None, "median": None, "p95": None, "max": None}
    xs = sorted(vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return float(xs[0])
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return float(xs[lo])
        return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))
    return {"n": len(xs), "min": float(xs[0]), "p05": q(0.05), "mean": float(sum(xs)/len(xs)), "median": q(0.5), "p95": q(0.95), "max": float(xs[-1])}


def source_word_counts(rows: Iterable[Row]) -> Dict[str, int]:
    d: Dict[str, int] = defaultdict(int)
    for r in rows:
        d[r.source] += r.words
    return dict(sorted(d.items()))


def bins_by_words(rows: List[Row], bin_words: int = 1_000_000, max_bins: int = 10) -> List[Dict[str, Any]]:
    bins = []
    cur_sources: Dict[str, int] = defaultdict(int)
    cur_words = 0
    total = 0
    b = 1
    for r in rows:
        remaining_row_words = r.words
        # Keep rows atomic for training; for audit bins, assign the whole row to the
        # bin in which it starts.  This mirrors checkpoint crossing granularity.
        cur_sources[r.source] += remaining_row_words
        cur_words += remaining_row_words
        total += remaining_row_words
        while cur_words >= bin_words and b <= max_bins:
            bins.append({
                "bin_index": b,
                "approx_start_words": (b - 1) * bin_words,
                "approx_end_words": b * bin_words,
                "row_atomic_words_in_bin": cur_words,
                "source_words": dict(sorted(cur_sources.items())),
            })
            b += 1
            cur_sources = defaultdict(int)
            cur_words = 0
        if b > max_bins:
            break
    return bins


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not IN_10M.exists():
        raise SystemExit(f"missing input corpus: {IN_10M}")
    source_meta = json.loads(IN_META.read_text(encoding="utf-8"))
    if source_meta.get("status") != "CLEAN_QWEN_CORPORA_MATERIALIZED":
        raise SystemExit("unexpected clean-Qwen metadata status; refusing to build a derivative corpus")

    rows: List[Row] = []
    raw_objs: List[Dict[str, Any]] = []
    freqs: Counter[str] = Counter()
    total_words = 0
    with IN_10M.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj["text"]
            words = int(obj["words"])
            if len(text.split()) != words:
                raise RuntimeError(f"whitespace word mismatch at row {i}: {len(text.split())} != {words}")
            nw = norm_words(text)
            rows.append(Row(index=i, obj=obj, text=text, words=words, source=str(obj.get("source", "unknown")), norm_words=nw))
            raw_objs.append(obj)
            freqs.update(nw)
            total_words += words
    if total_words != 10_000_000:
        raise RuntimeError(f"input 10M word mismatch: {total_words}")

    # corpus-internal frequency threshold for row difficulty proxies
    positive_freqs = sorted(freqs.values())
    rare_cut = positive_freqs[max(0, int(0.20 * (len(positive_freqs) - 1)))] if positive_freqs else 1

    for r in rows:
        if r.norm_words:
            vals = [math.log1p(freqs[w]) for w in r.norm_words]
            r.mean_log_freq = sum(vals) / len(vals)
            r.rare_frac = sum(1 for w in r.norm_words if freqs[w] <= rare_cut) / len(r.norm_words)
            r.avg_chars = sum(len(w) for w in r.norm_words) / len(r.norm_words)
            r.type_token_ratio = len(set(r.norm_words)) / len(r.norm_words)
        sr = SOURCE_RANK.get(r.source, 9)
        # Earlier = child/simple/aligned-second-view/dialogue before dense prose;
        # within source, high corpus frequency and shorter lexical form first.  The
        # final index makes the sort deterministic and not tuned after evaluation.
        r.rank_key = (
            sr,
            -round(r.mean_log_freq, 8),
            round(r.rare_frac, 8),
            round(r.avg_chars, 8),
            round(r.type_token_ratio, 8),
            r.words,
            r.index,
        )

    sorted_rows = sorted(rows, key=lambda r: r.rank_key)
    sorted_objs = [r.obj for r in sorted_rows]

    in_digest = multiset_digest(raw_objs)
    sorted_digest = multiset_digest(sorted_objs)
    if in_digest != sorted_digest:
        raise RuntimeError("row multiset digest changed after sorting")

    def write_rows(path: pathlib.Path, row_objs: Iterable[Dict[str, Any]]) -> int:
        n_words = 0
        n_rows = 0
        with path.open("w", encoding="utf-8") as f:
            for obj in row_objs:
                n_words += int(obj["words"])
                n_rows += 1
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return n_words

    w10 = write_rows(OUT_10M, sorted_objs)
    if w10 != 10_000_000:
        raise RuntimeError(f"sorted 10M word mismatch: {w10}")

    # First-pass intervention: only the first 10M are sorted; passes 2--10 are the
    # original research order.  This targets AoA checkpoints while minimally changing
    # final optimization exposure relative to the clean-Qwen baseline.
    with OUT_100M.open("w", encoding="utf-8") as f:
        total = 0
        for obj in sorted_objs:
            total += int(obj["words"])
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        for _ in range(9):
            for obj in raw_objs:
                total += int(obj["words"])
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if total != 100_000_000:
        raise RuntimeError(f"firstpass 100M word mismatch: {total}")

    # Also write an all-pass variant for later testing, but do not train it by
    # default; it is a stronger curriculum perturbation and may affect NLP columns.
    with OUT_ALLPASS_100M.open("w", encoding="utf-8") as f:
        total_all = 0
        for _ in range(10):
            for obj in sorted_objs:
                total_all += int(obj["words"])
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if total_all != 100_000_000:
        raise RuntimeError(f"allpass 100M word mismatch: {total_all}")

    source_counts_sorted = source_word_counts(sorted_rows)
    source_counts_original = source_word_counts(rows)
    first_bins_sorted = bins_by_words(sorted_rows, 1_000_000, 10)
    first_bins_original = bins_by_words(rows, 1_000_000, 10)

    meta = {
        "status": "AOA_SAFE_DEVCURR_QWEN_ORDER_MATERIALIZED",
        "scientific_purpose": "Test whether a legal developmental first-pass order can improve the AoA checkpoint trajectory while preserving the clean-Qwen same-window second-view multiset and final training exposure.",
        "non_leakage_statement": "Ordering uses only research training text, official source labels, and corpus-internal lexical statistics. It does not read or use official AoA/CDI evaluation words, AoA predictions, or AoA scores as training/selection signals.",
        "input_10m": str(IN_10M),
        "input_metadata": str(IN_META),
        "output_10m_sorted": str(OUT_10M),
        "output_100m_firstpass": str(OUT_100M),
        "output_100m_allpasses_for_later": str(OUT_ALLPASS_100M),
        "row_count": len(rows),
        "word_total_10m": total_words,
        "word_total_100m_firstpass": 100_000_000,
        "passes": 10,
        "first_pass_sorted_then_nine_original_passes": True,
        "exact_row_multiset_preserved": in_digest == sorted_digest,
        "row_multiset_digest": in_digest,
        "source_rank_earlier_is_smaller": SOURCE_RANK,
        "sort_key": ["source_rank", "-mean_log_corpus_word_frequency", "rare_word_fraction", "avg_word_chars", "type_token_ratio", "whitespace_words", "original_row_index"],
        "source_word_counts_original": source_counts_original,
        "source_word_counts_sorted": source_counts_sorted,
        "first_pass_1M_bins_original_order": first_bins_original,
        "first_pass_1M_bins_developmental_order": first_bins_sorted,
        "feature_stats": {
            "mean_log_freq": stats([r.mean_log_freq for r in rows]),
            "rare_frac": stats([r.rare_frac for r in rows]),
            "avg_chars": stats([r.avg_chars for r in rows]),
            "type_token_ratio": stats([r.type_token_ratio for r in rows]),
            "words": stats([float(r.words) for r in rows]),
        },
        "sha256": {
            OUT_10M.name: sha_file(OUT_10M),
            OUT_100M.name: sha_file(OUT_100M),
            OUT_ALLPASS_100M.name: sha_file(OUT_ALLPASS_100M),
        },
        "training_comparison": {
            "baseline_runs": [
                "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022",
                "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43122"
            ],
            "intended_new_runs": [
                "experiments/archive/compact_experience/training/runs/qwen_devcurr_firstpass_16k_seed43022",
                "experiments/archive/compact_experience/training/runs/qwen_devcurr_firstpass_16k_seed43122"
            ],
            "changed_factor": "first-pass training order only; same 10M clean-Qwen multiset repeated exactly ten times, same baseline16k tokenizer and DeBERTa-v2 8x480 WWM recipe."
        }
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "metadata": str(OUT_META),
        "row_count": len(rows),
        "word_total_10m": total_words,
        "first_1M_devcurr_sources": first_bins_sorted[0]["source_words"] if first_bins_sorted else {},
        "first_1M_original_sources": first_bins_original[0]["source_words"] if first_bins_original else {},
        "sha256_firstpass_100M": meta["sha256"][OUT_100M.name],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
