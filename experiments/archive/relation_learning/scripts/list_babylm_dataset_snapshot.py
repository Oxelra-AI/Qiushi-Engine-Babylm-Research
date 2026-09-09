#!/usr/bin/env python3
"""research: list the official BabyLM 2026 Strict-Small dataset snapshot.

The goal is to determine whether the pinned dataset revision contains raw
same-source development/validation files, without assuming this from local
data directories.  This script records the repository file list and obvious train /
dev / valid candidates for downstream clean ordinary-fit axis construction.
"""
from __future__ import annotations

import json
import pathlib
import re
import time
from typing import Any

from huggingface_hub import HfApi

ROOT = pathlib.Path.cwd()
OUT = ROOT / "experiments/archive/relation_learning/data/babylm_dataset_snapshot"
DATASET_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
REVISION = "c92ab16b4f08858304b0815706065b3354d8fc0a"
SOURCES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def classify(path: str) -> str:
    s = path.lower()
    if re.search(r"(^|[/_.-])(dev|valid|validation|val)([/_.-]|$)", s):
        return "dev_or_valid_candidate"
    if re.search(r"(^|[/_.-])train([/_.-]|$)", s):
        return "train_candidate"
    if s.endswith("readme.md") or s.endswith("dataset_infos.json"):
        return "metadata"
    return "other"


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files(repo_id=DATASET_ID, repo_type="dataset", revision=REVISION)
    try:
        info = api.dataset_info(repo_id=DATASET_ID, revision=REVISION, files_metadata=True)
        siblings = {s.rfilename: {"size": getattr(s, "size", None), "blob_id": getattr(s, "blob_id", None), "lfs": getattr(s, "lfs", None)} for s in info.siblings}
    except Exception as exc:  # file list is the essential output
        siblings = {}
        info = None
        info_error = {"type": type(exc).__name__, "message": str(exc)}
    else:
        info_error = None
    rows: list[dict[str, Any]] = []
    for f in sorted(files):
        lower = f.lower()
        source_hits = [src for src in SOURCES if src in lower]
        rows.append({
            "path": f,
            "class": classify(f),
            "source_hits": source_hits,
            "size": siblings.get(f, {}).get("size"),
            "blob_id": siblings.get(f, {}).get("blob_id"),
        })
    dev_like = [r for r in rows if r["class"] == "dev_or_valid_candidate"]
    train_like = [r for r in rows if r["class"] == "train_candidate"]
    same_source_dev = [r for r in dev_like if r["source_hits"]]
    same_source_train = [r for r in train_like if r["source_hits"]]
    payload = {
        "status": "BABYLM_DATASET_SNAPSHOT_LISTED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "n_files": len(rows),
        "class_counts": {c: sum(1 for r in rows if r["class"] == c) for c in sorted({r["class"] for r in rows})},
        "dev_or_valid_candidates": dev_like,
        "train_candidates": train_like,
        "same_source_dev_or_valid_candidates": same_source_dev,
        "same_source_train_candidates": same_source_train,
        "dataset_info_error": info_error,
        "files_jsonl": rel(OUT / "snapshot_files.jsonl"),
        "scientific_interpretation": "A raw same-source dev axis is available only if same_source_dev_or_valid_candidates contains actual corpus text files. If absent, a clean broad-fit axis must be drawn from source-matched corpus text outside all lineage streams and screened for tokenizer-training and stream exposure.",
    }
    with (OUT / "snapshot_files.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "snapshot_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
