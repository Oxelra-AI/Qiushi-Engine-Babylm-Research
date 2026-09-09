#!/usr/bin/env python3
"""research: prepare AoA calibration screen inputs.

Two outputs:
1. A reordered 30M-word JSONL for the row-schedule arm
   (best legal schedule from research: source_childes_taper_rowsort_ratio_f40_d40)
2. A per-token enrichment weight JSON for the masking-credit arm

Both arms use the same 100M pool; only the first 30M words are materialized.
The enrichment table maps each vocabulary token to a CHILDES-minus-whole
enrichment z-score for use as a masking weight multiplier.

CPU only; no GPU or training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, csv, hashlib, json, math, pathlib, re, sys, time
from collections import Counter, defaultdict, deque
from typing import Any

import numpy as np

ROOT = _public_path('.')
STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
TOKENIZER_DIR = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_1M')
OUT = _public_path('experiments/archive/relation_learning/data/aoa_calibration_prep')
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
PASS_WORDS = 10_000_000
TARGET_WORDS = 30_000_000  # 3 passes


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


# ── 1. Load all rows and compute word-level features ──────────────────────
def scan_stream():
    """Scan the 100M stream; return rows + word-level frequency tables."""
    rows = []
    source_words: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    word_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    whole_counts: Counter[str] = Counter()
    consumed = 0
    t0 = time.time()
    with STREAM.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            src = str(obj.get("source", "unknown"))
            words = int(obj.get("words", len(text.split())))
            # Count all lower-cased words in this row
            word_counts: Counter[str] = Counter()
            for m in WORD_RE.finditer(text):
                w = m.group(0).lower()
                word_counts[w] += 1
            rows.append({"text": text, "source": src, "words": words,
                         "word_counts": word_counts, "example_id": obj.get("example_id", len(rows))})
            consumed += words
            source_words[src] += words
            source_rows[src] += 1
            for w, c in word_counts.items():
                whole_counts[w] += c
                word_by_source[src][w] += c
            if len(rows) % 100000 == 0:
                print(json.dumps({"event": "scan", "rows": len(rows), "words": consumed,
                                  "sec": round(time.time() - t0, 1)}), flush=True)
    print(json.dumps({"event": "scan_done", "rows": len(rows), "words": consumed,
                      "sec": round(time.time() - t0, 1)}), flush=True)
    return rows, source_words, source_rows, word_by_source, whole_counts, consumed


# ── 2. Schedule reordering (source_childes_taper_rowsort_ratio_f40_d40) ───
def compute_row_scores(rows, word_by_source, whole_counts, source_words):
    """Compute per-row CHILDES enrichment ratio scores for schedule ordering."""
    childes_den = max(source_words.get("childes", 1), 1)
    total = max(sum(source_words.values()), 1)
    for r in rows:
        wc = r["word_counts"]
        if not wc:
            r["score_childes_ratio"] = -1e9
            continue
        vals, weights = [], []
        for w, c in wc.items():
            cf = math.log((word_by_source.get("childes", Counter()).get(w, 0) + 0.5) / childes_den * 1e6)
            wf = math.log((whole_counts.get(w, 0) + 0.5) / total * 1e6)
            vals.append(cf - wf)
            weights.append(c)
        r["score_childes_ratio"] = float(np.average(vals, weights=weights)) if vals else -1e9


def taper_childes_quotas(total_childes: int, floor_frac: float, decay: float) -> list[float]:
    avg = total_childes / 10.0
    floor = floor_frac * avg
    remaining = max(total_childes - 10 * floor, 0.0)
    weights = [math.exp(-decay * p) for p in range(10)]
    sw = sum(weights)
    quotas = [floor + remaining * w / sw for w in weights]
    quotas[-1] += total_childes - sum(quotas)
    return quotas


def schedule_reorder(rows, source_words, floor_frac=0.40, decay=0.40):
    """Reorder rows via source_childes_taper_rowsort_ratio schedule."""
    sources = sorted(source_words)
    child_total = int(source_words.get("childes", 0))
    child_q = taper_childes_quotas(child_total, floor_frac, decay)
    nonchild = [s for s in sources if s != "childes"]
    nonchild_total = max(sum(source_words[s] for s in nonchild), 1)
    quotas = []
    for p in range(10):
        pq: dict[str, float] = {}
        cq = min(child_q[p], PASS_WORDS * 0.985)
        pq["childes"] = cq
        rem = PASS_WORDS - cq
        for s in nonchild:
            pq[s] = rem * float(source_words[s]) / nonchild_total
        quotas.append(pq)
    by_source: dict[str, list[int]] = {s: [] for s in sources}
    for i, r in enumerate(rows):
        by_source[str(r["source"])].append(i)
    # Sort each source queue by descending CHILDES enrichment ratio
    for s in sources:
        by_source[s].sort(key=lambda i: (-float(rows[i].get("score_childes_ratio", -1e9)), i))
    qs: dict[str, deque[int]] = {s: deque(v) for s, v in by_source.items()}
    order = []
    for p in range(10):
        pass_idxs = []
        for s in ["childes"] + [x for x in sources if x != "childes"]:
            target = quotas[p].get(s, 0.0)
            used = 0
            while qs.get(s) and qs[s] and (used < target or (p == 9 and qs[s])):
                idx = qs[s].popleft()
                pass_idxs.append(idx)
                used += int(rows[idx]["words"])
                if p < 9 and used >= target:
                    break
        # Sort within pass by enrichment ratio
        pass_idxs.sort(key=lambda i: (-float(rows[i].get("score_childes_ratio", -1e9)),
                                       str(rows[i]["source"]), i))
        order.extend(pass_idxs)
    leftovers = []
    for s in sources:
        leftovers.extend(list(qs[s]))
    if leftovers:
        leftovers.sort()
        order.extend(leftovers)
    assert len(order) == len(rows) and len(set(order)) == len(rows)
    return order


def materialize_stream(rows, order, target_words, out_path):
    """Write first target_words of the reordered stream as JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written_words = 0
    written_rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for oi in order:
            r = rows[oi]
            if written_words >= target_words:
                break
            obj = {"text": r["text"], "words": r["words"],
                   "example_id": r.get("example_id", oi), "source": r["source"]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written_words += r["words"]
            written_rows += 1
    return written_words, written_rows


# ── 3. Per-token enrichment weight table ──────────────────────────────────
def compute_token_enrichment(whole_counts, word_by_source, source_words, tokenizer_dir):
    """Map CHILDES-minus-whole enrichment from words to token IDs."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    vocab_size = len(tokenizer)
    childes_den = max(source_words.get("childes", 1), 1)
    total = max(sum(source_words.values()), 1)

    # Per-word enrichment for all words in the stream
    word_enrichment: dict[str, float] = {}
    for w, wc in whole_counts.items():
        if wc < 2:
            continue  # skip very rare words
        cf = math.log((word_by_source.get("childes", Counter()).get(w, 0) + 0.5) / childes_den * 1e6)
        wf = math.log((wc + 0.5) / total * 1e6)
        word_enrichment[w] = cf - wf

    # Map words to token IDs: tokenize " word" to get the actual token(s)
    token_enrichment_sum = np.zeros(vocab_size, dtype=np.float64)
    token_enrichment_weight = np.zeros(vocab_size, dtype=np.float64)

    for w, enr in word_enrichment.items():
        freq = whole_counts[w]
        # Tokenize with space prefix (DeBERTa subword style)
        ids = tokenizer.encode(f" {w}", add_special_tokens=False)
        if not ids:
            continue
        # Distribute enrichment across tokens, weighted by word frequency
        for tid in ids:
            if 0 <= tid < vocab_size:
                token_enrichment_sum[tid] += enr * freq
                token_enrichment_weight[tid] += freq

    # Compute per-token enrichment
    enrichment = np.zeros(vocab_size, dtype=np.float64)
    active = token_enrichment_weight > 0
    enrichment[active] = token_enrichment_sum[active] / token_enrichment_weight[active]
    # Leave inactive tokens at 0 (neutral)

    n_active = int(active.sum())
    if n_active > 0:
        active_mean = float(enrichment[active].mean())
        active_std = float(enrichment[active].std())
    else:
        active_mean = 0.0
        active_std = 1.0

    # Z-score normalize
    z_enrichment = np.zeros(vocab_size, dtype=np.float64)
    if active_std > 0:
        z_enrichment[active] = (enrichment[active] - active_mean) / active_std

    return {
        "vocab_size": vocab_size,
        "n_active_tokens": n_active,
        "active_mean": active_mean,
        "active_std": active_std,
        "z_enrichment": z_enrichment.tolist(),
        "raw_enrichment": enrichment.tolist()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(OUT))
    parser.add_argument("--floor-frac", type=float, default=0.40)
    parser.add_argument("--decay", type=float, default=0.40)
    parser.add_argument("--target-words", type=int, default=TARGET_WORDS)
    args = parser.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # research: Scan the full 100M stream
    rows, source_words, source_rows, word_by_source, whole_counts, total_consumed = scan_stream()

    # research: Compute row scores and produce reordered stream
    compute_row_scores(rows, word_by_source, whole_counts, source_words)
    order = schedule_reorder(rows, source_words, args.floor_frac, args.decay)

    schedule_path = out / "schedule_reordered_30M.jsonl"
    written_words, written_rows = materialize_stream(rows, order, args.target_words, schedule_path)
    schedule_hash = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    print(json.dumps({"event": "schedule_stream_written", "path": rel(schedule_path),
                      "words": written_words, "rows": written_rows,
                      "sha256": schedule_hash}), flush=True)

    # Also produce the v4-order first 30M as a separate stream for the masking arm
    v4_path = out / "v4_order_30M.jsonl"
    v4_words, v4_rows = materialize_stream(rows, list(range(len(rows))), args.target_words, v4_path)
    v4_hash = hashlib.sha256(v4_path.read_bytes()).hexdigest()
    print(json.dumps({"event": "v4_stream_written", "path": rel(v4_path),
                      "words": v4_words, "rows": v4_rows,
                      "sha256": v4_hash}), flush=True)

    # research: Compute per-token enrichment weights
    enrichment = compute_token_enrichment(whole_counts, word_by_source, source_words, TOKENIZER_DIR)
    enr_path = out / "token_enrichment_weights.json"
    enr_path.write_text(json.dumps(enrichment, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "enrichment_weights_written", "path": rel(enr_path),
                      "n_active": enrichment["n_active_tokens"],
                      "active_mean": round(enrichment["active_mean"], 4),
                      "active_std": round(enrichment["active_std"], 4)}), flush=True)

    # Save summary
    summary = {
        "status": "AOA_CALIBRATION_PREP_DONE",
        "created_utc": now(),
        "scan": {"rows": len(rows), "words": total_consumed,
                 "source_words": dict(source_words)},
        "schedule": {"path": rel(schedule_path), "words": written_words,
                     "rows": written_rows, "sha256": schedule_hash,
                     "mode": f"source_childes_taper_rowsort_ratio_f{int(args.floor_frac*100)}_d{int(args.decay*100)}"},
        "v4_stream": {"path": rel(v4_path), "words": v4_words, "rows": v4_rows, "sha256": v4_hash},
        "enrichment": {"path": rel(enr_path), "n_active": enrichment["n_active_tokens"],
                       "active_mean": enrichment["active_mean"],
                       "active_std": enrichment["active_std"]},
        "elapsed_sec": round(time.time() - t0, 1)
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
