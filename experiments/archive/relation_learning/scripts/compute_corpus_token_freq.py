#!/usr/bin/env python3
"""Compute corpus token frequencies from the v4-order 30M stream.

Needed for corpus-occurrence-weighted normalization of enrichment masking:
Z(alpha) = sum_t(freq_t * exp(alpha * z_t)) / sum_t(freq_t)

This ensures expected mask rate = 0.15 by construction, regardless of alpha.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, sys, time
from pathlib import Path
from collections import Counter

HERE = _public_path('experiments/archive/relation_learning/scripts')
USER_ROOT = _public_path('.')

def main():
    stream = _public_path('experiments/archive/relation_learning/data/aoa_calibration_prep/v4_order_30M.jsonl')
    tok_path = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
    out_dir = _public_path('experiments/archive/relation_learning/data/corpus_token_freq')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import tokenizer with writable cache
    cache_dir = out_dir / "_hf_cache"
    cache_dir.mkdir(exist_ok=True)
    import os
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path))

    counts = Counter()
    total_tokens = 0
    total_rows = 0
    t0 = time.time()

    with open(stream) as f:
        for line in f:
            row = json.loads(line)
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            for tid in ids:
                counts[tid] += 1
            total_tokens += len(ids)
            total_rows += 1
            if total_rows % 50000 == 0:
                print(json.dumps({"event": "freq_scan", "rows": total_rows,
                                  "tokens": total_tokens,
                                  "sec": round(time.time()-t0, 1)}), flush=True)

    # Build vocab-indexed array
    vocab_size = len(tokenizer)
    freq = [0] * vocab_size
    for tid, cnt in counts.items():
        if tid < vocab_size:
            freq[tid] = cnt

    active = sum(1 for f in freq if f > 0)
    summary = {
        "total_tokens": total_tokens,
        "total_rows": total_rows,
        "active_vocab": active,
        "vocab_size": vocab_size,
        "stream": str(stream.relative_to(USER_ROOT)),
        "tokenizer": str(tok_path.relative_to(USER_ROOT)),
    }

    out = out_dir / "corpus_token_freq.json"
    out.write_text(json.dumps({"token_freq": freq, "summary": summary},
                              ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "CORPUS_FREQ_DONE", **summary,
                      "out": str(out.relative_to(USER_ROOT)),
                      "sec": round(time.time()-t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
