#!/usr/bin/env python3
"""Materialize an official-only 160-word-row 100M training file for geometry controls.

This creates a clean G0 official geometry: the exact 10M official pool rows from
`data/mixture/official_pool.jsonl`, repeated for ten shuffled passes with a
fixed pass-order seed family. It uses only official BabyLM training text and does not
read evaluation outputs or AoA/CDI information.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import pathlib
import random
from typing import Any, Dict, List

ROOT = _public_path('experiments/archive/compact_experience')
OFFICIAL_POOL = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/official_geometry/official160')
TOTAL_WORDS = 10_000_000
PASSES = 10
RNG_SEED = 282103


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                text = str(r["text"])
                words = int(r.get("words", len(text.split())))
                if words != len(text.split()):
                    raise RuntimeError(f"word mismatch in {path}: {words} vs {len(text.split())}")
                out.append({"text": text, "words": words, "example_id": int(r.get("example_id", len(out))), "source": str(r.get("source", "official"))})
    return out


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += int(r["words"])
    if total != TOTAL_WORDS:
        raise RuntimeError(f"pool total {total} != {TOTAL_WORDS}")


def write_train(path: pathlib.Path, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    pass_meta: List[Dict[str, Any]] = []
    with path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            order = list(range(len(rows)))
            seed = RNG_SEED + 100 + pass_i
            random.Random(seed).shuffle(order)
            before = total
            for idx in order:
                r = rows[idx]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += int(r["words"])
            pass_meta.append({"pass_index": pass_i + 1, "shuffle_seed": seed, "words_added": total - before, "rows_added": len(order)})
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"train total {total} != {TOTAL_WORDS * PASSES}")
    return pass_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = pathlib.Path(args.out_root)
    rows = read_jsonl(OFFICIAL_POOL)
    total = sum(int(r["words"]) for r in rows)
    if total != TOTAL_WORDS:
        raise RuntimeError(f"official pool total {total} != {TOTAL_WORDS}")
    if any(int(r["words"]) != 160 for r in rows):
        raise RuntimeError("official160 control expects all pool rows to have 160 words")
    pool = out / "training_corpora" / "official160_10M.jsonl"
    train = out / "training_corpora" / "official160_100M.jsonl"
    write_jsonl(pool, rows)
    pass_meta = write_train(train, rows)
    source_words: Dict[str, int] = {}
    for r in rows:
        source_words[str(r["source"])] = source_words.get(str(r["source"]), 0) + int(r["words"])
    payload = {
        "status": "OFFICIAL160_GEOMETRY_CONTROL_MATERIALIZED",
        "non_leakage_statement": "Uses only official BabyLM training text from the official 10M pool. No official AoA/CDI words, child curves, AoA predictions, AoA scores, or downstream evaluation outputs are read or used.",
        "official_pool_source": str(OFFICIAL_POOL),
        "row_topology": "original 160-whitespace-word official chunks from the official pool, ten shuffled passes",
        "rows_10M": len(rows),
        "words_10M": total,
        "rows_100M": len(rows) * PASSES,
        "words_100M": total * PASSES,
        "passes": pass_meta,
        "source_words_10M": dict(sorted(source_words.items())),
        "files": {"pool_10M": str(pool), "train_100M": str(train)},
        "sha256": {"pool_10M": sha_file(pool), "train_100M": sha_file(train)},
    }
    meta = out / "official160_geometry_metadata.json"
    meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": str(meta), "train_100M": str(train), "sha256_train_100M": payload["sha256"]["train_100M"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
