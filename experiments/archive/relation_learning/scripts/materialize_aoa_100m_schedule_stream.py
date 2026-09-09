#!/usr/bin/env python3
"""research: materialize the legal 100M AoA schedule stream selected by research/110.

This is a CPU artifact only.  It writes the full 100M stream in the same row order as
the validated 30M schedule arm (source_childes_taper_rowsort_ratio_f40_d40 by
default), computes SHA256/word accounting, and compares the first 30M prefix hash
against research when target mode/parameters match.  It does not launch a trunk.
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
import time
from typing import Any

ROOT = _public_path('.')
STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
DEFAULT_ORDER = _public_path('experiments/archive/relation_learning/data/aoa_across_pass_schedule_predictor/order_indices__source_childes_taper_rowsort_ratio_f40_d40.txt')
SUMMARY = _public_path('experiments/archive/relation_learning/data/aoa_calibration_prep/summary.json')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/aoa_schedule_100m_stream')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def wc_obj(obj: dict[str, Any]) -> int:
    try:
        return int(obj.get("words"))
    except Exception:
        return len(str(obj.get("text", "")).split())


def read_order(path: pathlib.Path) -> list[int]:
    order: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                order.append(int(s))
    return order


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize(rows: list[str], order: list[int], target_words: int, out_path: pathlib.Path) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    words = 0
    n_rows = 0
    source_words: dict[str, int] = {}
    source_rows: dict[str, int] = {}
    first_30m_hasher = hashlib.sha256()
    first_30m_words = 0
    first_30m_rows = 0
    with out_path.open("w", encoding="utf-8") as out:
        for idx in order:
            if words >= target_words:
                break
            raw = rows[idx]
            if not raw.endswith("\n"):
                raw_out = raw + "\n"
            else:
                raw_out = raw
            obj = json.loads(raw_out)
            w = wc_obj(obj)
            out.write(raw_out)
            words += w
            n_rows += 1
            src = str(obj.get("source", "unknown"))
            source_words[src] = source_words.get(src, 0) + w
            source_rows[src] = source_rows.get(src, 0) + 1
            if first_30m_words < 30_000_000:
                first_30m_hasher.update(raw_out.encode("utf-8"))
                first_30m_words += w
                first_30m_rows += 1
    return {
        "words": words,
        "rows": n_rows,
        "sha256": sha256_file(out_path),
        "first30m_words": first_30m_words,
        "first30m_rows": first_30m_rows,
        "first30m_sha256": first_30m_hasher.hexdigest(),
        "source_words": source_words,
        "source_rows": source_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--order-indices", type=pathlib.Path, default=DEFAULT_ORDER)
    ap.add_argument("--target-words", type=int, default=100_000_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    order_path = args.order_indices if args.order_indices.is_absolute() else ROOT / args.order_indices
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stream = out_dir / "schedule_reordered_100M.jsonl"
    if out_stream.exists() and not args.force:
        raise SystemExit(f"output exists; pass --force to replace: {out_stream}")

    t0 = time.time()
    rows = STREAM.read_text(encoding="utf-8").splitlines(keepends=True)
    order = read_order(order_path)
    if len(order) != len(rows) or len(set(order)) != len(rows):
        raise SystemExit(f"order length/set mismatch: order {len(order)} unique {len(set(order))}; rows {len(rows)}")
    stats = materialize(rows, order, int(args.target_words), out_stream)

    research = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.is_file() else {}
    sched = research.get("schedule", {}) if isinstance(research, dict) else {}
    prefix_match_step110 = None
    if sched:
        prefix_match_step110 = {
            "mode": sched.get("mode"),
            "words": sched.get("words"),
            "rows": sched.get("rows"),
            "sha256": sched.get("sha256"),
            "first30m_words_match": int(stats["first30m_words"]) == int(sched.get("words", -1)),
            "first30m_rows_match": int(stats["first30m_rows"]) == int(sched.get("rows", -1)),
            "first30m_sha256_match": str(stats["first30m_sha256"]) == str(sched.get("sha256")),
        }

    summary = {
        "status": "AOA_100M_SCHEDULE_STREAM_MATERIALIZED",
        "created_utc": now(),
        "purpose": "CPU-only byte-identical legal full-stream schedule artifact for a possible human-authorized AoA trunk run; not itself a launch decision",
        "input_stream": rel(STREAM),
        "order_indices": rel(order_path),
        "target_words": int(args.target_words),
        "output_stream": rel(out_stream),
        "output": stats,
        "prefix_validation": prefix_match_step110,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research AoA 100M schedule stream",
        "",
        summary["purpose"],
        "",
        f"Output stream: `{rel(out_stream)}`",
        f"Words/rows: {stats['words']} / {stats['rows']}",
        f"SHA256: `{stats['sha256']}`",
        f"First-30M words/rows: {stats['first30m_words']} / {stats['first30m_rows']}",
        f"First-30M SHA256: `{stats['first30m_sha256']}`",
    ]
    if prefix_match_step110:
        lines += [
            "",
            "## research prefix comparison",
            f"Mode: `{prefix_match_step110['mode']}`",
            f"Words match: `{prefix_match_step110['first30m_words_match']}`; rows match: `{prefix_match_step110['first30m_rows_match']}`; hash match: `{prefix_match_step110['first30m_sha256_match']}`",
        ]
    lines += ["", f"Full JSON: `{rel(out_dir / 'summary.json')}`"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "stream": rel(out_stream), "words": stats["words"], "rows": stats["rows"], "sha256": stats["sha256"], "prefix_validation": prefix_match_step110}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
