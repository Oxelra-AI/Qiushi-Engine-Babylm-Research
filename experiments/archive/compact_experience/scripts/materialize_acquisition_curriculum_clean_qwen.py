#!/usr/bin/env python3
"""Materialize legal corpus-only acquisition curricula over the clean-Qwen 10M pool.

The purpose is to test whether an acquisition-paced presentation order derived only
from training-corpus statistics can improve final BabyLM Strict-Small performance,
including the AoA column as a final measurement, without ever reading official AoA/CDI
words, child curves, AoA predictions, AoA scores, or downstream evaluation outputs.

Arms:
  A0 baseline_order: fresh matched-seed control using the research pass shuffle
      convention over the clean-Qwen 10M row multiset.
  A0p repeated_random: one fixed random order repeated for the early passes; isolates
      the repeated-early-order geometry shared by curriculum arms.
  A1 frequency_only: first two passes use easy-to-hard row order by corpus frequency.
  A2 freq_entropy_dispersion_spacing: first two passes use a fuller acquisition score
      combining word frequency, row lexical entropy, source/example dispersion, and a
      spacing heuristic that prevents local repetition of same source/example.
  A3 randomized_label_control: preserves A2 score distribution within strata but
      randomizes row-score assignments.
  A4 reverse_acquisition: hard-to-easy reverse of A2 to test directionality.

All arms write exact 10M pool JSONL and exact 100M training JSONL (ten passes). Passes
3--10 use the research pass shuffle seed convention. Only the early acquisition window
is reordered, but A1/A2/A3/A4 share a repeated deterministic early order, so A0p is the
proper control for that repeated-order geometry.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('experiments/archive/compact_experience')
IN_10M = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
IN_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
OUT_DIR = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_clean_qwen')
TOTAL_WORDS = 10_000_000
PASSES = 10
# research clean materializer used RNG_SEED=28043 and pass shuffles
# random.Random(RNG_SEED + 100 + pass_i).shuffle(order). Use the same
# convention for matched-seed controls, but do not assume byte identity with
# historical streams unless the recorded SHA comparison confirms it.
RNG_SEED = 28043
LOCAL_RNG_SEED = 282103
EARLY_CURRICULUM_PASSES = 2
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")


@dataclass
class Row:
    index: int
    obj: Dict[str, Any]
    text: str
    words: int
    source: str
    example_id: str
    is_qwen_pair: bool
    norm_words: List[str]
    features: Dict[str, float] = field(default_factory=dict)
    a1_score: float = 0.0
    a2_score: float = 0.0
    a3_score: float = 0.0


def norm_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(float(x) for x in vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(pos); hi = min(lo + 1, len(xs) - 1); frac = pos - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac
    return {"n": len(xs), "mean": sum(xs)/len(xs), "min": xs[0], "p05": q(0.05), "p25": q(0.25), "median": q(0.50), "p75": q(0.75), "p95": q(0.95), "max": xs[-1]}


def zscores(raws: Dict[str, List[float]], key: str) -> List[float]:
    vals = raws[key]
    mu = sum(vals) / len(vals)
    sd = (sum((x - mu) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5 or 1.0
    return [(x - mu) / sd for x in vals]


def assign_features(rows: List[Row]) -> None:
    freq: collections.Counter[str] = collections.Counter()
    word_sources: Dict[str, set[str]] = collections.defaultdict(set)
    word_examples: Dict[str, set[str]] = collections.defaultdict(set)
    word_row_counts: collections.Counter[str] = collections.Counter()
    for r in rows:
        freq.update(r.norm_words)
        for w in set(r.norm_words):
            word_sources[w].add(r.source)
            word_examples[w].add(r.example_id)
            word_row_counts[w] += 1
    fvals = sorted(freq.values())
    rare_cut = fvals[max(0, int(0.20 * (len(fvals) - 1)))] if fvals else 1
    source_total = len(set(r.source for r in rows)) or 1
    ex_total = len(set(r.example_id for r in rows)) or 1

    raws: Dict[str, List[float]] = collections.defaultdict(list)
    feats: List[Dict[str, float]] = []
    for r in rows:
        ws = r.norm_words
        if ws:
            counts = collections.Counter(ws)
            n = sum(counts.values())
            probs = [c / n for c in counts.values()]
            entropy = -sum(p * math.log(p + 1e-12) for p in probs) / math.log(max(2, len(counts)))
            mean_log_freq = sum(math.log1p(freq[w]) for w in ws) / len(ws)
            rare_frac = sum(1 for w in ws if freq[w] <= rare_cut) / len(ws)
            avg_chars = sum(len(w) for w in ws) / len(ws)
            ttr = len(counts) / len(ws)
            # Dispersed words are easier in the intended acquisition sense: common across
            # sources/examples rather than bursty in one source/example.
            src_disp = sum(len(word_sources[w]) / source_total for w in set(ws)) / len(set(ws))
            ex_disp = sum(math.log1p(len(word_examples[w])) / math.log1p(ex_total) for w in set(ws)) / len(set(ws))
            row_burst = sum(counts[w] / max(1, word_row_counts[w]) for w in counts) / len(counts)
        else:
            entropy = 1.0; mean_log_freq = 0.0; rare_frac = 1.0; avg_chars = 10.0; ttr = 1.0; src_disp = 0.0; ex_disp = 0.0; row_burst = 1.0
        feat = {
            "mean_log_freq": mean_log_freq,
            "rare_frac": rare_frac,
            "avg_chars": avg_chars,
            "ttr": ttr,
            "entropy": entropy,
            "source_dispersion": src_disp,
            "example_dispersion": ex_disp,
            "burstiness": row_burst,
            "words_norm": r.words / 160.0,
            "is_qwen_pair": 1.0 if r.is_qwen_pair else 0.0,
        }
        feats.append(feat)
        for k, v in feat.items():
            raws[k].append(v)
    z = {k: zscores(raws, k) for k in raws}
    for i, r in enumerate(rows):
        feat = feats[i]
        r.features = feat
        # Higher score = easier/earlier.
        r.a1_score = z["mean_log_freq"][i]
        r.a2_score = (
            1.00 * z["mean_log_freq"][i]
            - 0.55 * z["rare_frac"][i]
            - 0.35 * z["avg_chars"][i]
            - 0.25 * z["ttr"][i]
            - 0.30 * z["entropy"][i]
            + 0.45 * z["source_dispersion"][i]
            + 0.35 * z["example_dispersion"][i]
            - 0.30 * z["burstiness"][i]
            - 0.10 * z["words_norm"][i]
        )


def strata_key(r: Row) -> Tuple[str, int, int]:
    # Preserve source, row-length band, and generated-pair/filler mixture for randomized-label control.
    length_bin = min(7, r.words // 20)
    return (r.source, length_bin, 1 if r.is_qwen_pair else 0)


def assign_randomized_label_control(rows: List[Row]) -> None:
    rng = random.Random(LOCAL_RNG_SEED + 909)
    by_stratum: Dict[Tuple[str, int, int], List[Row]] = collections.defaultdict(list)
    for r in rows:
        by_stratum[strata_key(r)].append(r)
    for key, xs in by_stratum.items():
        # Use the empirical A2 score distribution within stratum, but assign scores to rows randomly.
        scores = sorted([r.a2_score for r in xs], reverse=True)
        perm = list(xs)
        rng.shuffle(perm)
        for r, s in zip(perm, scores):
            r.a3_score = s



def assert_permutation(order: List[int], n_rows: int, label: str) -> None:
    if len(order) != n_rows or sorted(order) != list(range(n_rows)):
        raise RuntimeError(f"{label} is not a full row permutation: len={len(order)} n_rows={n_rows}")
def pass_orders(n_rows: int) -> List[List[int]]:
    orders = []
    for pass_i in range(PASSES):
        order = list(range(n_rows))
        random.Random(RNG_SEED + 100 + pass_i).shuffle(order)
        assert_permutation(order, n_rows, f"default_pass_{pass_i+1}")
        orders.append(order)
    return orders


def repeated_random_order(n_rows: int) -> List[int]:
    order = list(range(n_rows))
    random.Random(LOCAL_RNG_SEED + 777).shuffle(order)
    assert_permutation(order, n_rows, "A0p_repeated_random")
    return order


def greedy_spaced_order(rows: List[Row], score_attr: str, source_window: int = 4, example_window: int = 24) -> List[Row]:
    remaining = sorted(rows, key=lambda r: (-float(getattr(r, score_attr)), r.source, r.example_id, r.index))
    out: List[Row] = []
    recent_sources: collections.deque[str] = collections.deque(maxlen=source_window)
    recent_examples: collections.deque[str] = collections.deque(maxlen=example_window)
    # O(n * small candidate prefix) heuristic; enough for ~70k rows.
    while remaining:
        pick_i = None
        for i, r in enumerate(remaining[:256]):
            if r.source not in recent_sources and r.example_id not in recent_examples:
                pick_i = i
                break
        if pick_i is None:
            for i, r in enumerate(remaining[:512]):
                if r.example_id not in recent_examples:
                    pick_i = i
                    break
        if pick_i is None:
            pick_i = 0
        r = remaining.pop(pick_i)
        out.append(r)
        recent_sources.append(r.source)
        recent_examples.append(r.example_id)
    return out


def write_pool(path: pathlib.Path, ordered: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for r in ordered:
            total += r.words
            f.write(json.dumps(r.obj, ensure_ascii=False) + "\n")
    if total != TOTAL_WORDS:
        raise RuntimeError(f"pool {path} words {total} != {TOTAL_WORDS}")


def write_train(path: pathlib.Path, pool_rows: List[Row], early_orders: List[List[int]], default_orders: List[List[int]]) -> List[Dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    pass_meta: List[Dict[str, Any]] = []
    with path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            before = total
            if pass_i < len(early_orders):
                order = early_orders[pass_i]
                policy = "acquisition_curriculum"
            else:
                order = default_orders[pass_i]
                policy = "matched_shuffle"
            assert_permutation(order, len(pool_rows), f"{path.name}_pass_{pass_i+1}")
            for idx in order:
                r = pool_rows[idx]
                total += r.words
                f.write(json.dumps(r.obj, ensure_ascii=False) + "\n")
            pass_meta.append({"pass_index": pass_i + 1, "policy": policy, "rows_added": len(order), "words_added": total - before})
        raise RuntimeError(f"train {path} words {total} != {TOTAL_WORDS * PASSES}")
    return pass_meta


def bins_by_words(order: List[Row], bin_words: int = 1_000_000, max_bins: int = 10) -> List[Dict[str, Any]]:
    bins = []
    cur_words = 0
    source_words: Dict[str, int] = collections.defaultdict(int)
    qwen_words = 0
    row_scores: List[float] = []
    b = 1
    for r in order:
        cur_words += r.words
        source_words[r.source] += r.words
        qwen_words += r.words if r.is_qwen_pair else 0
        row_scores.append(r.a2_score)
        if cur_words >= bin_words and b <= max_bins:
            bins.append({
                "bin_index": b,
                "row_atomic_words": cur_words,
                "source_words": dict(sorted(source_words.items())),
                "qwen_pair_words": qwen_words,
                "mean_a2_score": sum(row_scores) / len(row_scores) if row_scores else None,
                "rows": len(row_scores),
            })
            cur_words = 0; source_words = collections.defaultdict(int); qwen_words = 0; row_scores = []; b += 1
        if b > max_bins:
            break
    return bins


def source_words(rows: Iterable[Row]) -> Dict[str, int]:
    d: Dict[str, int] = collections.defaultdict(int)
    for r in rows:
        d[r.source] += r.words
    return dict(sorted(d.items()))


def variant_record(name: str, rows: List[Row], early_rank_orders: List[List[int]], default_orders: List[List[int]], description: str, write_files: bool) -> Dict[str, Any]:
    out_root = OUT_DIR / name
    pool_path = out_root / f"{name}_10M.jsonl"
    train_path = out_root / f"{name}_100M.jsonl"
    # Pool file remains in the order used for first curriculum pass for audit convenience.
    first_order = early_rank_orders[0] if early_rank_orders else default_orders[0]
    for pass_i in range(PASSES):
        order = early_rank_orders[pass_i] if pass_i < len(early_rank_orders) else default_orders[pass_i]
        assert_permutation(order, len(rows), f"{name}_pass_{pass_i+1}")
    pool_order = [rows[i] for i in first_order]
    pass_meta = []
    for pass_i in range(PASSES):
        order = early_rank_orders[pass_i] if pass_i < len(early_rank_orders) else default_orders[pass_i]
        pass_meta.append({
            "pass_index": pass_i + 1,
            "policy": "acquisition_curriculum" if pass_i < len(early_rank_orders) else "matched_shuffle",
            "rows_added": len(order),
            "words_added": sum(rows[i].words for i in order),
        })
    meta = {
        "name": name,
        "description": description,
        "pool_10M": str(pool_path),
        "train_100M": str(train_path),
        "words_10M": TOTAL_WORDS,
        "words_100M": TOTAL_WORDS * PASSES,
        "passes": pass_meta,
        "row_multiset_digest": multiset_digest([r.obj for r in rows]),
        "first_pass_bins": bins_by_words(pool_order),
        "source_words_10M": source_words(rows),
        "sha256": {"pool_10M": None, "train_100M": None},
        "files_written": bool(write_files),
    }
    if write_files:
        write_pool(pool_path, pool_order)
        written_pass_meta = write_train(train_path, rows, early_rank_orders, default_orders)
        meta["passes"] = written_pass_meta
        meta["sha256"] = {"pool_10M": sha_file(pool_path), "train_100M": sha_file(train_path)}
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / f"{name}_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--early_passes", type=int, default=EARLY_CURRICULUM_PASSES)
    ap.add_argument("--metadata_only", action="store_true", help="Audit/order metadata only; do not write 10M/100M corpus files.")
    args = ap.parse_args()
    if args.early_passes < 1 or args.early_passes > PASSES:
        raise SystemExit("early_passes must be between 1 and 10")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    in_meta = json.loads(IN_META.read_text(encoding="utf-8"))
    if in_meta.get("status") != "CLEAN_QWEN_CORPORA_MATERIALIZED":
        raise SystemExit(f"unexpected clean metadata status: {in_meta.get('status')}")
    raw_objs = read_jsonl(IN_10M)
    rows: List[Row] = []
    total = 0
    for i, obj in enumerate(raw_objs):
        text = str(obj["text"])
        words = int(obj.get("words", len(text.split())))
        if words != len(text.split()):
            raise RuntimeError(f"word mismatch row {i}: {words} vs {len(text.split())}")
        source = str(obj.get("source", "unknown"))
        eid = str(obj.get("example_id", i))
        is_qwen = source == "qwen_pair_packed" or "qwen" in source or obj.get("contains_qwen_pair") is True
        rows.append(Row(i, obj, text, words, source, eid, is_qwen, norm_words(text)))
        total += words
    if total != TOTAL_WORDS:
        raise RuntimeError(f"input word total {total} != {TOTAL_WORDS}")
    assign_features(rows)
    assign_randomized_label_control(rows)
    default_orders = pass_orders(len(rows))

    a0_early = [default_orders[i] for i in range(args.early_passes)]
    a1_order = [r.index for r in sorted(rows, key=lambda r: (-r.a1_score, r.source, r.words, r.index))]
    a2_order_rows = greedy_spaced_order(rows, "a2_score")
    a2_order = [r.index for r in a2_order_rows]
    a3_order = [r.index for r in sorted(rows, key=lambda r: (-r.a3_score, r.source, r.words, r.index))]
    a4_order = [r.index for r in sorted(rows, key=lambda r: (r.a2_score, r.source, r.words, r.index))]
    a0p_order = repeated_random_order(len(rows))
    for label, order in [("A1", a1_order), ("A2", a2_order), ("A3", a3_order), ("A4", a4_order), ("A0p", a0p_order)]:
        assert_permutation(order, len(rows), label)
    write_files = not args.metadata_only
    variants = []
    variants.append(variant_record("A0_baseline_order", rows, a0_early, default_orders, "Fresh matched-seed retraining control using the research pass-order convention (RNG_SEED base); early two passes use two independent shuffles like research. Compare its 100M hash to the historical clean-Qwen stream before treating it as an exact reproduction.", write_files))
    variants.append(variant_record("A0p_repeated_random", rows, [a0p_order for _ in range(args.early_passes)], default_orders, "Repeated-random control: one fixed random early order repeated for the early passes. Isolates the repeated-early-order geometry that A1/A2/A3/A4 share, so acquisition effects are measured against repeated random rather than against two independent shuffles.", write_files))
    variants.append(variant_record("A1_frequency_only", rows, [a1_order for _ in range(args.early_passes)], default_orders, "Early passes ordered easy-to-hard by corpus-internal mean log word frequency only, repeated for the early passes.", write_files))
    variants.append(variant_record("A2_freq_entropy_dispersion_spacing", rows, [a2_order for _ in range(args.early_passes)], default_orders, "Early passes ordered by fixed corpus-only acquisition score: frequency, rare fraction, word length, lexical entropy, source/example dispersion, burstiness, and spacing heuristic, repeated for the early passes.", write_files))
    variants.append(variant_record("A3_randomized_label_control", rows, [a3_order for _ in range(args.early_passes)], default_orders, "Randomized-label control preserving A2 score distribution inside source/length/generated strata while destroying the exact corpus-derived difficulty ordering; same repeated-early-order geometry as A1/A2/A4.", write_files))
    variants.append(variant_record("A4_reverse_acquisition", rows, [a4_order for _ in range(args.early_passes)], default_orders, "Hard-to-easy reverse of the A2 acquisition score with the same repeated-early-order geometry; tests directionality of the acquisition signal rather than any fixed deterministic order effect.", write_files))

    historical_train = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl')
    a0_sha = next(v["sha256"]["train_100M"] for v in variants if v["name"] == "A0_baseline_order")
    historical_sha = sha_file(historical_train) if (historical_train.exists() and write_files) else None
    payload = {
        "status": "ACQUISITION_CURRICULUM_CLEAN_QWEN_MATERIALIZED",
        "non_leakage_statement": "Uses only research clean-Qwen training text, row metadata, source labels, and corpus-internal lexical statistics. It does not read or use official AoA/CDI evaluation words, child curves, AoA predictions, AoA scores, SuperGLUE labels, or downstream evaluation outputs for data construction or checkpoint selection.",
        "input_10M": str(IN_10M),
        "input_metadata": str(IN_META),
        "row_count": len(rows),
        "word_total_10M": total,
        "passes": PASSES,
        "early_curriculum_passes": args.early_passes,
        "row_multiset_digest": multiset_digest(raw_objs),
        "exact_row_multiset_preserved_in_all_variants": len({v["row_multiset_digest"] for v in variants}) == 1,
        "default_pass_order_seed_rule": "RNG_SEED(28043) + 100 + pass_i, matching research pass shuffle convention for baseline and later passes",
        "a0_reproduction_status": "A0 is a fresh matched-seed retraining control; byte-identity with the historical clean-Qwen 100M stream is NOT proven here. The research pass order was computed over its own qwen_pool row indexing, and this materializer reindexes the same multiset. Compare A0_baseline_order_100M.jsonl sha256 to data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl before claiming exact reproduction.",
        "confound_controls": {"A0p_repeated_random": "repeated random early order isolates repeated-early-order geometry", "A4_reverse_acquisition": "reverse acquisition tests directionality"},
        "score_weights_A2_higher_is_earlier": {"mean_log_freq": 1.0, "rare_frac": -0.55, "avg_chars": -0.35, "ttr": -0.25, "entropy": -0.30, "source_dispersion": 0.45, "example_dispersion": 0.35, "burstiness": -0.30, "words_norm": -0.10},
        "feature_stats": {k: stats([r.features[k] for r in rows]) for k in rows[0].features},
        "source_words_10M": source_words(rows),
        "qwen_pair_words_10M_estimate": sum(r.words for r in rows if r.is_qwen_pair),
        "metadata_only": bool(args.metadata_only),
        "files_written": bool(write_files),
        "historical_train_100M": str(historical_train),
        "historical_train_100M_sha256": historical_sha,
        "a0_train_100M_sha256": a0_sha,
        "a0_matches_historical_train_100M_sha256": (a0_sha == historical_sha) if (a0_sha is not None and historical_sha is not None) else None,
        "variants": variants,
    }
    meta_path = OUT_DIR / ("acquisition_curriculum_metadata_preflight.json" if args.metadata_only else "acquisition_curriculum_metadata.json")
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": str(meta_path), "metadata_only": bool(args.metadata_only), "variants": [{"name": v["name"], "train_100M": v["train_100M"], "sha256": v["sha256"]["train_100M"], "files_written": v.get("files_written")} for v in variants]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
