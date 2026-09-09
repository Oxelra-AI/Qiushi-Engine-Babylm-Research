#!/usr/bin/env python3
"""Materialize the exact-ten-pass official arm for the fixed-initialization mixture replication.

The paired arm already exists at data/mixture/training_files/mix_25pct_100M.jsonl.
This script expands the exact 10,000,000-word official pool with the identical epoch-wise
shuffle rule used by the mixture materializer, giving a legal 100,000,000-word exposure.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import hashlib
import json
import random
from pathlib import Path

ROOT = _public_path('experiments/archive/compact_experience')
POOL = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
OUT = _public_path('experiments/archive/compact_experience/data/fixedinit_replication/official_100M.jsonl')
SUMMARY = _public_path('experiments/archive/compact_experience/data/fixedinit_replication/materialization_summary.json')
SEED = 43
EPOCHS = 10
EXPECTED_POOL_WORDS = 10_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows = [json.loads(line) for line in POOL.open(encoding="utf-8")]
    pool_words = sum(int(r["words"]) for r in rows)
    if pool_words != EXPECTED_POOL_WORDS:
        raise RuntimeError(f"official pool words {pool_words} != {EXPECTED_POOL_WORDS}")
    _public_path('experiments/archive/compact_experience/data/fixedinit_replication').mkdir(parents=True, exist_ok=True)
    epoch_hashes = []
    written = 0
    with OUT.open("w", encoding="utf-8") as f:
        for epoch in range(EPOCHS):
            epoch_rows = list(rows)
            shuffle_seed = SEED + 1_000_003 * epoch
            random.Random(shuffle_seed).shuffle(epoch_rows)
            order_h = hashlib.sha256()
            epoch_words = 0
            for row in epoch_rows:
                rec = dict(row)
                rec["source"] = f"epoch{epoch + 1}::{row['source']}"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                epoch_words += int(row["words"])
                written += int(row["words"])
                order_h.update(str(row["example_id"]).encode())
                order_h.update(b"\n")
            if epoch_words != EXPECTED_POOL_WORDS:
                raise RuntimeError(f"epoch {epoch+1} words {epoch_words}")
            epoch_hashes.append({"epoch": epoch + 1, "shuffle_seed": shuffle_seed,
                                 "words": epoch_words, "example_order_sha256": order_h.hexdigest()})
    expected = EXPECTED_POOL_WORDS * EPOCHS
    if written != expected:
        raise RuntimeError(f"written words {written} != {expected}")
    payload = {
        "status": "OFFICIAL_EXACT10_MATERIALIZED",
        "official_pool": str(POOL),
        "official_pool_words": pool_words,
        "official_pool_rows": len(rows),
        "epochs": EPOCHS,
        "training_exposure": written,
        "training_file": str(OUT),
        "training_sha256": sha256(OUT),
        "epoch_order": epoch_hashes,
        "paired_arm": str(_public_path('experiments/archive/compact_experience/data/mixture/training_files/mix_25pct_100M.jsonl')),
        "scientific_contrast": "official-only versus 25%-aligned mixture under identical tokenizer, model initialization, masking RNG, architecture, optimizer, and exposure",
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
