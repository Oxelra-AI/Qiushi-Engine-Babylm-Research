#!/usr/bin/env python3
"""research: materialize legal isolated-replay private-phase streams.

Scientific purpose
------------------
The successful v4 private phase used 3,992,800 words of coherent 256-token rows.
Stage-II evidence says correct local correspondence increases context-supported
prediction and wrong/non-correspondence can favor isolation-like columns.  This
script constructs a cheap legal screen for whether the private branch can rebias
existing readouts toward evaluation formats without adding non-corpus text:

  * isolated_all: the exact coherent86 tail rows are split at sentence-like
    boundaries, so the same corpus words are practiced with minimal preceding
    document context.
  * half_coherent_half_isolated: the same tail rows are deterministically divided
    into coherent versus sentence-isolated rows by row hash, then interleaved,
    testing whether the private readout can condition on context presence rather
    than moving one global context-reliance setting.

No generated text is introduced; only the row boundaries/order within the private
suffix are changed.  The output JSONL schema is accepted by the research replay
trainer with --skip_rows 0 and --replay_mode coherent_replay.
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
import re
import time
from typing import Any

ROOT = _public_path('.')
BASE_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
OUT_DIR = _public_path('experiments/archive/relation_learning/data/isolated_replay_streams')
SKIP_ROWS = 530_944
TARGET_WORDS = 3_992_800  # actual coherent86 tail_main_word_exposure
SEED = 95097

SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z0-9\"'“‘*]|=))")


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_tail(path: pathlib.Path, skip_rows: int, target_words: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skip_rows:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch at stream row {idx}: metadata {words}, actual {len(text.split())}")
            if total + words > target_words:
                break
            rows.append({
                "text": text,
                "words": words,
                "source": str(obj.get("source", "")),
                "example_id": int(obj.get("example_id", -1)),
                "row_index_in_stream": idx,
            })
            total += words
    if total != target_words:
        raise RuntimeError(f"tail words {total} != requested {target_words}; coherent86 reference should be exact")
    return rows


def split_sentences(text: str) -> list[str]:
    # Keep section-header fragments and dialogue lines as standalone chunks when
    # punctuation is sparse; the splitter is deliberately conservative and never
    # drops tokens.
    pieces = [p.strip() for p in SENT_BOUNDARY.split(text.strip()) if p.strip()]
    return pieces if pieces else [text.strip()]


def isolate_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = split_sentences(row["text"])
    out: list[dict[str, Any]] = []
    for j, ch in enumerate(chunks):
        w = len(ch.split())
        if w <= 0:
            continue
        out.append({
            "text": ch,
            "words": w,
            "source": row["source"] + "|sentence_isolated",
            "example_id": row["example_id"],
            "row_index_in_stream": row["row_index_in_stream"],
            "chunk_index": j,
            "isolation_origin_words": row["words"],
        })
    if sum(x["words"] for x in out) != row["words"]:
        raise RuntimeError(f"split word mismatch row {row['row_index_in_stream']}")
    return out


def write_jsonl(rows: list[dict[str, Any]], path: pathlib.Path) -> str:
    sha = hashlib.sha256()
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            obj = {"text": r["text"], "words": int(r["words"]), "source": r["source"], "example_id": int(r.get("example_id", -1))}
            line = json.dumps(obj, ensure_ascii=False) + "\n"
            f.write(line)
            sha.update(line.encode("utf-8"))
    return sha.hexdigest()


def summarize(rows: list[dict[str, Any]], source_tail: list[dict[str, Any]], label: str) -> dict[str, Any]:
    words = sum(int(r["words"]) for r in rows)
    isolated_words = sum(int(r["words"]) for r in rows if "sentence_isolated" in r["source"])
    return {
        "label": label,
        "rows": len(rows),
        "words": words,
        "isolated_words": isolated_words,
        "isolated_fraction": isolated_words / words if words else 0.0,
        "coherent_or_unsplit_words": words - isolated_words,
        "source_tail_rows": len(source_tail),
        "source_tail_words": sum(int(r["words"]) for r in source_tail),
        "mean_words_per_output_row": words / len(rows) if rows else 0.0,
        "max_words_per_output_row": max([int(r["words"]) for r in rows], default=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-stream", default=str(BASE_STREAM))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--skip-rows", type=int, default=SKIP_ROWS)
    ap.add_argument("--target-words", type=int, default=TARGET_WORDS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tail = read_tail(pathlib.Path(args.base_stream), int(args.skip_rows), int(args.target_words))

    isolated_all: list[dict[str, Any]] = []
    for r in tail:
        isolated_all.extend(isolate_row(r))

    # Deterministic near-half assignment by shuffled row indices.  Keep complete
    # original rows on one side to preserve the exact word multiset once.
    rng = random.Random(int(args.seed))
    order = list(range(len(tail)))
    rng.shuffle(order)
    target_iso = int(args.target_words) // 2
    iso_set: set[int] = set()
    iso_words = 0
    for idx in order:
        if iso_words >= target_iso:
            continue
        iso_set.add(idx)
        iso_words += int(tail[idx]["words"])

    half_rows: list[dict[str, Any]] = []
    for i, r in enumerate(tail):
        if i in iso_set:
            half_rows.extend(isolate_row(r))
        else:
            half_rows.append({
                "text": r["text"], "words": r["words"], "source": r["source"] + "|coherent_unsplit",
                "example_id": r["example_id"], "row_index_in_stream": r["row_index_in_stream"],
            })

    # Interleave half-arm rows so the private phase does not see all one format first.
    rng.shuffle(half_rows)

    outputs = {
        "isolated_all": {"rows": isolated_all, "file": out_dir / "isolated_all_replay_3992800w.jsonl"},
        "half_coherent_half_isolated": {"rows": half_rows, "file": out_dir / "half_coherent_half_isolated_replay_3992800w.jsonl"},
    }
    manifest: dict[str, Any] = {
        "status": "ISOLATED_REPLAY_STREAMS_READY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_stream": rel(args.base_stream),
        "skip_rows": int(args.skip_rows),
        "target_words": int(args.target_words),
        "seed": int(args.seed),
        "legal_scope": "Same BabyLM Strict-Small corpus words from the coherent86 private suffix; no generated text; only sentence-boundary row segmentation and half-arm row-format assignment are changed.",
        "outputs": {},
        "elapsed_sec": None,
    }
    for label, obj in outputs.items():
        sha = write_jsonl(obj["rows"], obj["file"])
        stat = summarize(obj["rows"], tail, label)
        stat.update({"path": rel(obj["file"]), "sha256": sha})
        manifest["outputs"][label] = stat
    manifest["elapsed_sec"] = round(time.time() - t0, 2)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research isolated replay streams", "", manifest["legal_scope"], ""]
    for label, stat in manifest["outputs"].items():
        lines.append(f"## {label}")
        lines.append(f"- Path: `{stat['path']}`")
        lines.append(f"- Rows/words: {stat['rows']} / {stat['words']}")
        lines.append(f"- Isolated fraction: {stat['isolated_fraction']:.4f}")
        lines.append(f"- Mean/max words per row: {stat['mean_words_per_output_row']:.2f} / {stat['max_words_per_output_row']}")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "outputs": {k: {kk: vv for kk, vv in v.items() if kk in {"path", "rows", "words", "isolated_fraction", "sha256"}}
                    for k, v in manifest["outputs"].items()},
        "elapsed_sec": manifest["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
