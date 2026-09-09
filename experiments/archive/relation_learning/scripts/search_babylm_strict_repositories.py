#!/usr/bin/env python3
"""research: check whether a 2026 BabyLM Strict 100M dataset repository exists.

research established that the pinned Strict-Small 10M repo contains only six
train files and no raw dev/valid split.  This script checks Hugging Face for a
corresponding 2026 Strict (100M) dataset repository and records its file list,
so a future same-source unseen axis can be built only if source text outside the
Strict-Small lineage is actually present.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import re
import time
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/search_babylm_strict_repositories.py')
ROOT = _PUBLIC_ROOT

OUT = ROOT / "experiments/archive/relation_learning/data/babylm_strict_repo_search"
SMALL_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
SMALL_REV = "c92ab16b4f08858304b0815706065b3354d8fc0a"
CANDIDATE_IDS = [
    "BabyLM-community/BabyLM-2026-Strict",
    "BabyLM-community/BabyLM-2026-Strict-100M",
    "BabyLM-community/BabyLM-2026-Strict-Large",
    "BabyLM-community/BabyLM-2026",
]
SOURCES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def classify(path: str) -> str:
    s = path.lower()
    if re.search(r"(^|[/_.-])(dev|valid|validation|val)([/_.-]|$)", s):
        return "dev_or_valid_candidate"
    if re.search(r"(^|[/_.-])train([/_.-]|$)", s):
        return "train_candidate"
    if s.endswith("readme.md") or s.endswith("dataset_infos.json") or s.endswith(".json"):
        return "metadata_or_other_json"
    return "other"


def file_rows(api: HfApi, repo_id: str, revision: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
        info = api.dataset_info(repo_id=repo_id, revision=revision, files_metadata=True)
    except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
        return [], {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        return [], {"type": type(exc).__name__, "message": str(exc)}
    siblings = {s.rfilename: {"size": getattr(s, "size", None), "blob_id": getattr(s, "blob_id", None), "lfs": getattr(s, "lfs", None)} for s in info.siblings}
    rows = []
    for f in sorted(files):
        lower = f.lower()
        rows.append({
            "repo_id": repo_id,
            "revision_requested": revision,
            "path": f,
            "class": classify(f),
            "source_hits": [src for src in SOURCES if src in lower],
            "size": siblings.get(f, {}).get("size"),
            "blob_id": siblings.get(f, {}).get("blob_id"),
        })
    return rows, None


def compact_repo_summary(repo_id: str, rows: list[dict[str, Any]], error: dict[str, Any] | None, small_blob_by_path: dict[str, str | None]) -> dict[str, Any]:
    train = [r for r in rows if r["class"] == "train_candidate"]
    dev = [r for r in rows if r["class"] == "dev_or_valid_candidate"]
    same_source_train = [r for r in train if r["source_hits"]]
    same_source_dev = [r for r in dev if r["source_hits"]]
    # Same path/blob comparison is not a subset test, but it quickly separates an
    # exact copy from a larger candidate source.
    exact_same_path_blobs = []
    for r in rows:
        if r["path"] in small_blob_by_path and small_blob_by_path[r["path"]] == r.get("blob_id"):
            exact_same_path_blobs.append(r["path"])
    return {
        "repo_id": repo_id,
        "exists": error is None,
        "error": error,
        "n_files": len(rows),
        "class_counts": {c: sum(1 for r in rows if r["class"] == c) for c in sorted({r["class"] for r in rows})},
        "train_candidates": same_source_train,
        "dev_or_valid_candidates": same_source_dev,
        "total_same_source_train_bytes": sum(int(r.get("size") or 0) for r in same_source_train),
        "strict_small_same_path_same_blob_files": exact_same_path_blobs,
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    small_rows, small_error = file_rows(api, SMALL_ID, SMALL_REV)
    small_blob_by_path = {r["path"]: r.get("blob_id") for r in small_rows}

    candidates: list[str] = []
    for ds in api.list_datasets(author="BabyLM-community", search="BabyLM-2026", limit=200):
        dsid = getattr(ds, "id", None) or getattr(ds, "modelId", None) or str(ds)
        if dsid and dsid not in candidates:
            candidates.append(dsid)
    for cid in CANDIDATE_IDS:
        if cid not in candidates:
            candidates.append(cid)

    repo_summaries = []
    all_rows = []
    for cid in candidates:
        rows, error = file_rows(api, cid, None)
        all_rows.extend(rows)
        repo_summaries.append(compact_repo_summary(cid, rows, error, small_blob_by_path))

    strict_like = [s for s in repo_summaries if s["exists"] and re.search(r"strict", s["repo_id"], flags=re.I)]
    strict_100m_candidates = []
    small_total = sum(int(r.get("size") or 0) for r in small_rows if r["class"] == "train_candidate" and r["source_hits"])
    for s in strict_like:
        rid = s["repo_id"].lower()
        if rid == SMALL_ID.lower():
            continue
        total = s["total_same_source_train_bytes"]
        if total and total > small_total * 1.5:
            strict_100m_candidates.append(s["repo_id"])
        elif "strict-small" not in rid and "strict_small" not in rid:
            strict_100m_candidates.append(s["repo_id"])

    payload = {
        "status": "BABYLM_STRICT_REPOSITORY_SEARCH_DONE",
        "created_utc": now(),
        "elapsed_sec": round(time.time() - t0, 2),
        "small_repo": {
            "repo_id": SMALL_ID,
            "revision": SMALL_REV,
            "error": small_error,
            "n_files": len(small_rows),
            "same_source_train_bytes": small_total,
        },
        "searched_author": "BabyLM-community",
        "candidate_repo_ids": candidates,
        "repo_summaries": repo_summaries,
        "strict_100m_candidate_repo_ids": sorted(set(strict_100m_candidates)),
        "files_jsonl": rel(OUT / "all_candidate_files.jsonl"),
        "summary_json": rel(OUT / "summary.json"),
        "scientific_interpretation": "If strict_100m_candidate_repo_ids contains a same-source train repo larger than Strict-Small, its exact-match-excluded complement may provide the missing same-source unseen fit axis after stream and tokenizer-training exposure screens. If it is empty, no Hugging Face BabyLM-community 2026 Strict 100M source was found by this search.",
    }
    with (OUT / "all_candidate_files.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
