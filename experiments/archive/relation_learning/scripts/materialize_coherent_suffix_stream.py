#!/usr/bin/env python3
"""research: materialize the unsplit coherent86 private suffix as a research trainer stream.

The isolated and half-format streams already reuse the same 3,992,800 legal suffix
words with altered row boundaries. This file writes the corresponding unchanged-row
stream so the corrected research trainer can test the special-token/official-format
minimal intervention: same coherent rows, same word-paced schedule, but tokenization
adds official [CLS]/[SEP]-style special tokens during private continuation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import pathlib
import time

ROOT = _public_path('.')
BASE_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
OUT_DIR = _public_path('experiments/archive/relation_learning/data/format_control_streams')
OUT_FILE = _public_path('experiments/archive/relation_learning/data/format_control_streams/coherent_unsplit_replay_3992800w.jsonl')
SKIP_ROWS = 530_944
TARGET_WORDS = 3_992_800


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = 0
    words_total = 0
    sha = hashlib.sha256()
    first_example_id = None
    last_example_id = None
    first_row_index = None
    last_row_index = None
    max_words = 0
    with BASE_STREAM.open(encoding="utf-8") as fin, OUT_FILE.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if idx < SKIP_ROWS:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch at base row {idx}: metadata={words} actual={len(text.split())}")
            if words_total + words > TARGET_WORDS:
                break
            out = {
                "text": text,
                "words": words,
                "source": str(obj.get("source", "")) + "|coherent_unsplit_special_control",
                "example_id": int(obj.get("example_id", -1)),
                "row_index_in_stream": idx,
            }
            out_line = json.dumps(out, ensure_ascii=False) + "\n"
            fout.write(out_line)
            sha.update(out_line.encode("utf-8"))
            rows += 1
            words_total += words
            max_words = max(max_words, words)
            if first_example_id is None:
                first_example_id = out["example_id"]
                first_row_index = idx
            last_example_id = out["example_id"]
            last_row_index = idx
    if words_total != TARGET_WORDS:
        raise RuntimeError(f"wrote {words_total} words, expected {TARGET_WORDS}")
    manifest = {
        "status": "COHERENT_UNSPLIT_STREAM_READY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "path": rel(OUT_FILE),
        "sha256": sha.hexdigest(),
        "base_stream": rel(BASE_STREAM),
        "skip_rows": SKIP_ROWS,
        "target_words": TARGET_WORDS,
        "rows": rows,
        "words": words_total,
        "mean_words_per_row": words_total / rows,
        "max_words_per_row": max_words,
        "first_example_id": first_example_id,
        "last_example_id": last_example_id,
        "first_row_index_in_stream": first_row_index,
        "last_row_index_in_stream": last_row_index,
        "legal_scope": "Exact coherent86 private suffix rows and words; no generated text; used only with research trainer add_special_tokens=True to isolate official special-token exposure under otherwise coherent replay.",
    }
    (_public_path('experiments/archive/relation_learning/data/format_control_streams/manifest.json')).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research coherent unsplit special-token control stream", "", manifest["legal_scope"], ""]
    lines.append(f"- Path: `{manifest['path']}`")
    lines.append(f"- Rows/words: `{rows}` / `{words_total}`")
    lines.append(f"- Mean/max words per row: `{manifest['mean_words_per_row']:.3f}` / `{max_words}`")
    lines.append(f"- SHA256: `{manifest['sha256']}`")
    (_public_path('research/documents/relation_learning/data/format_control_streams/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "path": manifest["path"], "rows": rows, "words": words_total, "sha256": manifest["sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
