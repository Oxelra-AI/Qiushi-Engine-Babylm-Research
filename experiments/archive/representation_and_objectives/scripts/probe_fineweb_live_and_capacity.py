#!/usr/bin/env python3
"""research: tiny live FineWeb-Edu reachability and local capacity probe.

This is CPU/network-only. It does not train. Purpose: decide whether a future
larger source-breadth arm can be built from live FineWeb-Edu or whether local
cached material limits the contrast to the existing 1.75M-word seqsafe96 block.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
from collections import Counter
from typing import Any

OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_live_capacity_probe")
NOTE = pathlib.Path("research/notes/representation_and_objectives/fineweb_live_capacity_probe.md")
LOCAL_CACHED = pathlib.Path("experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_random_quality_3000000w.jsonl")
SEQSAFE_META = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/materialization_metadata.json")
A02_OVERLAY_META = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json")

WORD_RE = re.compile(r"\S+")


def words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def count_jsonl_words(path: pathlib.Path, limit_rows: int | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    rows = 0
    total_words = 0
    sources = Counter()
    doc_ids = set()
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rows += 1
        if limit_rows is not None and rows > limit_rows:
            break
        try:
            o = json.loads(line)
        except Exception:
            continue
        text = o.get("text") or o.get("content") or o.get("source_text") or o.get("source") or ""
        w = int(o.get("words", words(text))) if isinstance(o, dict) else 0
        total_words += w
        src = o.get("source") or o.get("source_asset") or o.get("pool") or "unknown"
        sources[str(src)] += w
        ids = o.get("doc_ids") or o.get("doc_id") or o.get("document_id")
        if isinstance(ids, list):
            for x in ids:
                doc_ids.add(str(x))
        elif ids is not None:
            doc_ids.add(str(ids))
    return {"exists": True, "path": str(path), "rows_scanned": rows, "words_scanned": total_words, "top_sources": sources.most_common(10), "unique_docs_scanned": len(doc_ids)}


def tiny_live_probe(max_rows: int = 8) -> dict[str, Any]:
    t0 = time.time()
    cache_dir = OUT_DIR / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Avoid false read-only failures from the shared runtime cache.
    os.environ["HF_HOME"] = str(cache_dir / "home")
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_MODULES_CACHE"] = str(cache_dir / "modules")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    os.environ["TMPDIR"] = str(cache_dir / "tmp")
    for sub in ["home", "hub", "datasets", "modules", "transformers", "xdg", "tmp"]:
        (cache_dir / sub).mkdir(parents=True, exist_ok=True)
    try:
        import datasets  # type: ignore
    except Exception as e:
        return {"status": "import_error", "error": repr(e), "elapsed_sec": round(time.time() - t0, 3), "cache_dir": str(cache_dir)}
    try:
        # Streaming is the lowest-cost reachability test; do not download a shard.
        ds = datasets.load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True, cache_dir=str(cache_dir / "datasets"))
        rows = []
        total = 0
        for i, ex in enumerate(ds):
            text = str(ex.get("text", ""))
            w = words(text)
            total += w
            rows.append({"i": i, "words": w, "fields": sorted(ex.keys()), "text_excerpt": " ".join(text.split()[:60])})
            if len(rows) >= max_rows:
                break
        return {"status": "ok", "dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "rows": len(rows), "words": total, "samples": rows, "elapsed_sec": round(time.time() - t0, 3), "cache_dir": str(cache_dir)}
    except Exception as e:
        return {"status": "error", "error": repr(e), "elapsed_sec": round(time.time() - t0, 3), "cache_dir": str(cache_dir)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    live = tiny_live_probe(max_rows=8)
    local = count_jsonl_words(LOCAL_CACHED)
    seqsafe = json.loads(SEQSAFE_META.read_text(encoding="utf-8")) if SEQSAFE_META.exists() else None
    a02 = json.loads(A02_OVERLAY_META.read_text(encoding="utf-8")) if A02_OVERLAY_META.exists() else None
    capacity = {
        "local_cached_raw": local,
        "a01_seqsafe96_fineweb_words": (((seqsafe or {}).get("verification") or {}).get("fineweb_words") if seqsafe else None),
        "a01_seqsafe96_fraction_of_10M": ((((seqsafe or {}).get("verification") or {}).get("fineweb_words") or 0) / 10_000_000 if seqsafe else None),
        "a02_changed_block_budget_words": (a02 or {}).get("changed_block_budget_words") if isinstance(a02, dict) else None,
        "a02_fraction_of_10M": ((a02 or {}).get("changed_block_budget_words", 0) / 10_000_000 if isinstance(a02, dict) else None),
    }
    payload = {"status": "FINEWEB_LIVE_CAPACITY_PROBE", "live_probe": live, "capacity": capacity}
    out_json = OUT_DIR / "fineweb_live_capacity_probe.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research FineWeb live reachability and capacity probe\n\n"]
    lines.append(f"Live HuggingFaceFW/fineweb-edu status: `{live.get('status')}`; elapsed {live.get('elapsed_sec')} s.\n\n")
    if live.get("status") == "ok":
        lines.append(f"Tiny streaming sample rows {live.get('rows')} / words {live.get('words')}. This means a larger source-breadth arm can in principle be built from live FineWeb-Edu if rule/accounting checks are preserved.\n\n")
    else:
        lines.append(f"Error: `{live.get('error')}`. If this persists, A01's local clean seqsafe FineWeb source-repetition material is capped by the cached file at the existing 1.753M-word block unless A02 assets are shared/reused.\n\n")
    lines.append(f"Local cached raw scan: {local.get('rows_scanned')} rows / {local.get('words_scanned')} words / {local.get('unique_docs_scanned')} docs (before single-doc + quality + seqsafe filtering).\n\n")
    lines.append(f"A01 seqsafe96 FineWeb block: {capacity['a01_seqsafe96_fineweb_words']} words ({capacity['a01_seqsafe96_fraction_of_10M']:.2%} of 10M).\n\n")
    lines.append(f"A02 density overlay changed block: {capacity['a02_changed_block_budget_words']} words ({capacity['a02_fraction_of_10M']:.2%} of 10M).\n\n")
    lines.append("Interpretation: the queued A01 run remains the largest currently validated source-breadth contrast. A larger-fraction test needs either recovered live FineWeb-Edu access or a new local source asset.\n\n")
    lines.append(f"JSON: `{out_json}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "note": str(NOTE), "live_status": live.get("status"), "local_words": local.get("words_scanned")}, indent=2))


if __name__ == "__main__":
    main()
