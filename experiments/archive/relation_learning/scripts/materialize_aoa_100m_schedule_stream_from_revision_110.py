#!/usr/bin/env python3
"""research repair: materialize the full 100M AoA schedule using research code.

The first research full-stream materializer used research saved order indices and preserved
raw input lines.  Its first-30M prefix did not match the actual research calibration stream.
This repair imports research's own scan/score/reorder/materialize functions and writes the
same simplified JSONL schema as research, with target_words=100M.  It validates that the
prefix ending at the research schedule row count has the exact research SHA256.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import importlib.util
import json
import pathlib
import time
from typing import Any

ROOT = _public_path('.')
SCRIPT = _public_path('experiments/archive/relation_learning/scripts/prep_aoa_calibration.py')
SUMMARY = _public_path('experiments/archive/relation_learning/data/aoa_calibration_prep/summary.json')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/aoa_schedule_100m_stream_repair')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_step110():
    spec = importlib.util.spec_from_file_location("prep_aoa_calibration_imported_for_step125", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def prefix_hash(path: pathlib.Path, n_rows: int) -> tuple[str, int, int]:
    h = hashlib.sha256()
    words = 0
    rows = 0
    with path.open("rb") as f:
        for line in f:
            if rows >= n_rows:
                break
            h.update(line)
            rows += 1
            obj = json.loads(line)
            words += int(obj.get("words", len(str(obj.get("text", "")).split())))
    return h.hexdigest(), words, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--floor-frac", type=float, default=0.40)
    ap.add_argument("--decay", type=float, default=0.40)
    ap.add_argument("--target-words", type=int, default=100_000_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stream = out_dir / "schedule_reordered_100M_step110_schema.jsonl"
    if out_stream.exists() and not args.force:
        raise SystemExit(f"output exists; pass --force: {out_stream}")

    mod = load_step110()
    research = json.loads(SUMMARY.read_text(encoding="utf-8"))
    t0 = time.time()
    rows, source_words, source_rows, word_by_source, whole_counts, total_consumed = mod.scan_stream()
    mod.compute_row_scores(rows, word_by_source, whole_counts, source_words)
    order = mod.schedule_reorder(rows, source_words, float(args.floor_frac), float(args.decay))
    written_words, written_rows = mod.materialize_stream(rows, order, int(args.target_words), out_stream)
    out_sha = sha256_file(out_stream)

    sched = research.get("schedule", {})
    rows = int(sched.get("rows", -1))
    pref_sha, pref_words, pref_rows = prefix_hash(out_stream, rows) if rows > 0 else (None, 0, 0)
    prefix_validation = {
        "mode": sched.get("mode"),
        "words": sched.get("words"),
        "rows": sched.get("rows"),
        "sha256": sched.get("sha256"),
        "prefix_rows_checked": pref_rows,
        "prefix_words": pref_words,
        "prefix_sha256": pref_sha,
        "prefix_words_match": pref_words == int(sched.get("words", -1)),
        "prefix_rows_match": pref_rows == int(sched.get("rows", -1)),
        "prefix_sha256_match": pref_sha == str(sched.get("sha256")),
    }
    # Keep a small order hash without writing 647k indices again.
    order_h = hashlib.sha256("\n".join(map(str, order)).encode("utf-8")).hexdigest()
    summary: dict[str, Any] = {
        "status": "AOA_100M_SCHEDULE_STREAM_STEP110_REPAIR_DONE",
        "created_utc": now(),
        "purpose": "full 100M legal schedule stream using the exact research ordering/schema code; CPU artifact only, not a trunk launch",
        "script": rel(SCRIPT),
        "input_stream": rel(mod.STREAM),
        "target_words": int(args.target_words),
        "floor_frac": float(args.floor_frac),
        "decay": float(args.decay),
        "scan": {"rows": len(rows), "words": total_consumed, "source_words": dict(source_words), "source_rows": dict(source_rows)},
        "order_sha256": order_h,
        "output_stream": rel(out_stream),
        "output_words": written_words,
        "output_rows": written_rows,
        "output_sha256": out_sha,
        "prefix_validation": prefix_validation,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research repaired AoA 100M schedule stream",
        "",
        summary["purpose"],
        "",
        f"Output stream: `{rel(out_stream)}`",
        f"Words/rows: {written_words} / {written_rows}",
        f"SHA256: `{out_sha}`",
        "",
        "## research prefix validation",
        f"Words match: `{prefix_validation['prefix_words_match']}`; rows match: `{prefix_validation['prefix_rows_match']}`; hash match: `{prefix_validation['prefix_sha256_match']}`",
        f"Prefix words/rows/hash: {pref_words} / {pref_rows} / `{pref_sha}`",
        f"research words/rows/hash: {sched.get('words')} / {sched.get('rows')} / `{sched.get('sha256')}`",
        "",
        f"Full JSON: `{rel(out_dir / 'summary.json')}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "stream": rel(out_stream), "words": written_words, "rows": written_rows, "sha256": out_sha, "prefix_validation": prefix_validation}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
