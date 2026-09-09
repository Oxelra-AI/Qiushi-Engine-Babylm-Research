#!/usr/bin/env python3
"""Step047b: materialize the matched no-substitution legal ordinary tail.

The historical research continuation remains scientifically useful, but it used a
different trainer and cannot be the only matched control for the corrected bridge.
This script writes the exact coherent86 86M->100M reference tail as bridge-format
`ordinary_tail` rows so the research corrected trainer can run the same row-keyed
WWM, word-paced macro-update, checkpoint, and evaluator path without relation
substitution.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import hashlib
import json
import pathlib
import time
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
BASE_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/reference_tail')
INITIAL_WORDS = 86_005_295
FULL_CAP_WORDS = 100_000_000
SKIP_ROWS = 556_791
TAIL_WORDS = FULL_CAP_WORDS - INITIAL_WORDS


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_reference_tail(path: pathlib.Path, skip_rows: int, max_words: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    words_selected = 0
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skip_rows:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch base row {idx}: declared={words} actual={len(text.split())}")
            if words_selected + words > max_words:
                break
            out.append({
                "text": text,
                "words": words,
                "example_id": int(obj.get("example_id", -1)),
                "source": str(obj.get("source", "")),
                "orig_row_index": idx,
                "bridge_kind": "ordinary_tail",
                "reference_tail_position": len(out),
                "presentation_schedule": "reference_tail_wordpaced",
            })
            words_selected += words
    if words_selected != max_words:
        raise RuntimeError(f"selected {words_selected} words, expected {max_words}")
    return out


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> Tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = words = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            words += int(r["words"])
    return n, words


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_reference_tail(BASE_STREAM, SKIP_ROWS, TAIL_WORDS)
    out_jsonl = _public_path('experiments/archive/functional_learning/data/reference_tail/reference_tail_ordinary_wordpaced.jsonl')
    n, words = write_jsonl(out_jsonl, rows)
    source_rows = collections.Counter(str(r.get("source", "")) for r in rows)
    source_words = collections.Counter()
    for r in rows:
        source_words[str(r.get("source", ""))] += int(r["words"])
    manifest = {
        "status": "REFERENCE_TAIL_READY",
        "created_utc": now_utc(),
        "scientific_purpose": "Matched no-substitution ordinary-tail input for the corrected research trainer and evaluator path.",
        "base_stream": rel(BASE_STREAM),
        "base_stream_sha256": sha256_file(BASE_STREAM),
        "initial_consumed_words": INITIAL_WORDS,
        "full_cap_words": FULL_CAP_WORDS,
        "skip_rows": SKIP_ROWS,
        "tail_words": TAIL_WORDS,
        "rows": n,
        "words": words,
        "final_total_words_if_trained_from_parent": INITIAL_WORDS + words,
        "source_rows": dict(source_rows),
        "source_words": dict(source_words),
        "output_jsonl": rel(out_jsonl),
        "trainer_to_use": "experiments/archive/functional_learning/scripts/corrected_bridge_trainer.py with --arm ordinary_wwm; all rows are ordinary_tail so lambda_relation=0.",
    }
    (_public_path('experiments/archive/functional_learning/data/reference_tail/reference_tail_manifest.json')).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
