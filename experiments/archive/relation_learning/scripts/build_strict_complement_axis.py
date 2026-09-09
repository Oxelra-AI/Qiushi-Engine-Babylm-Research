#!/usr/bin/env python3
"""research: build a candidate same-source unseen axis from the 2026 Strict complement.

The pinned Strict-Small repo has no dev/valid corpus split.  research found a
larger same-source `BabyLM-community/BabyLM-2026-Strict` repo.  This script
checks whether the Strict-Small files are exact byte prefixes of the Strict
files; if so, it samples deterministic 160-word rows from the post-prefix
complement and screens those rows for exact normalized-row overlap with the
known relation-composition streams.

The output is a candidate scoring axis, not a final generalization result.  It
must still be interpreted with the exact checks recorded here and any additional
future screens against tokenizer-training data or near-duplicate phrase reuse.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/build_strict_complement_axis.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
OUT_DEFAULT = WS / "data/strict_complement_axis"
SMALL_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
SMALL_REV = "c92ab16b4f08858304b0815706065b3354d8fc0a"
STRICT_ID = "BabyLM-community/BabyLM-2026-Strict"
FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]
SOURCE_OF_FILE = {f: f.replace(".train.txt", "") for f in FILES}
FULL_HELDOUT = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl"
SUMMARY = WS / "data/relation_stream_exposure_screen/summary.json"
WORD_RE = re.compile(r"\S+")


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def norm_text(s: str) -> str:
    return " ".join(str(s).lower().split())


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def stat(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float) -> float:
        if len(ys) == 1:
            return float(ys[0])
        k = (len(ys)-1)*p
        lo = int(math.floor(k)); hi = int(math.ceil(k))
        if lo == hi:
            return float(ys[lo])
        return float(ys[lo]*(hi-k) + ys[hi]*(k-lo))
    return {"n": len(xs), "mean": float(statistics.mean(xs)), "median": q(0.5), "min": float(ys[0]), "max": float(ys[-1]), "p10": q(0.1), "p90": q(0.9)}


def get_file_meta(api: HfApi, repo_id: str) -> tuple[str | None, dict[str, dict[str, Any]]]:
    info = api.dataset_info(repo_id=repo_id, files_metadata=True)
    sha = getattr(info, "sha", None)
    by = {}
    for s in info.siblings:
        by[s.rfilename] = {"size": getattr(s, "size", None), "blob_id": getattr(s, "blob_id", None), "lfs": getattr(s, "lfs", None)}
    return sha, by


def get_range(url: str, start: int, end: int, timeout: int = 120) -> bytes:
    if end < start:
        return b""
    headers = {"Range": f"bytes={start}-{end}"}
    r = requests.get(url, headers=headers, stream=True, timeout=timeout)
    r.raise_for_status()
    target_len = end - start + 1
    buf = bytearray()
    for chunk in r.iter_content(chunk_size=1 << 20):
        if not chunk:
            continue
        need = target_len - len(buf)
        if need <= 0:
            break
        buf.extend(chunk[:need])
        if len(buf) >= target_len:
            break
    return bytes(buf)


def heldout_source_targets(n_rows: int) -> dict[str, int]:
    counts = Counter()
    for obj in read_jsonl(FULL_HELDOUT):
        counts[str(obj.get("source", ""))] += 1
    total = sum(counts.values())
    raw = {k: counts[k] * n_rows / total for k in counts}
    targets = {k: int(math.floor(v)) for k, v in raw.items()}
    # ensure each observed source with nonzero desired mass is represented if possible
    for k, v in raw.items():
        if v > 0 and targets[k] == 0:
            targets[k] = 1
    while sum(targets.values()) < n_rows:
        k = max(raw, key=lambda x: raw[x] - targets[x])
        targets[k] += 1
    while sum(targets.values()) > n_rows:
        k = max(targets, key=lambda x: targets[x] - raw[x])
        if targets[k] > 0:
            targets[k] -= 1
        else:
            break
    return dict(sorted(targets.items()))


def words_from_windows(repo_id: str, filename: str, strict_size: int, prefix_size: int, target_words: int) -> tuple[list[str], list[dict[str, Any]]]:
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
    comp_start = prefix_size
    comp_len = max(0, strict_size - prefix_size)
    if comp_len <= 0:
        return [], []
    # Fetch several deterministic windows spread over the complement.  The byte
    # budget is deliberately a little larger than the expected text needed.
    expected_bytes = max(80_000, int(target_words * 8.5))
    windows = max(4, min(32, math.ceil(target_words / 20_000) * 4))
    win_size = max(40_000, min(400_000, math.ceil(expected_bytes / windows * 1.7)))
    records = []
    toks: list[str] = []
    for i in range(windows):
        if comp_len <= win_size:
            start = comp_start
        else:
            # center windows throughout complement; deterministic and not
            # tuned to later outcomes.
            frac = (i + 0.5) / windows
            start = comp_start + int(frac * (comp_len - win_size))
        end = min(strict_size - 1, start + win_size - 1)
        b = get_range(url, start, end)
        text = b.decode("utf-8", errors="ignore")
        # Drop edges so broken UTF-8 or partial words at byte boundaries are not
        # systematically retained.
        if len(text) > 400:
            text = text[200:-200]
        local_toks = WORD_RE.findall(text)
        records.append({"filename": filename, "window_index": i, "byte_start": start, "byte_end": end, "bytes_received": len(b), "tokens_extracted": len(local_toks)})
        toks.extend(local_toks)
        if len(toks) >= target_words + 160:
            # Continue no further than needed; later windows are deterministic
            # but not needed for this row count.
            break
    return toks[:target_words], records


def build_rows(tokens_by_source: dict[str, list[str]], target_rows: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_order = sorted(target_rows)
    for si, source in enumerate(source_order):
        toks = tokens_by_source[source]
        need = target_rows[source]
        if len(toks) < need * 160:
            raise RuntimeError(f"not enough sampled tokens for {source}: {len(toks)} < {need*160}")
        for j in range(need):
            ww = toks[j*160:(j+1)*160]
            rows.append({
                "text": " ".join(ww),
                "words": 160,
                "example_id": 900_000_000 + si * 1_000_000 + j,
                "source": source,
                "origin_repo": STRICT_ID,
                "origin_file": f"{source}.train.txt",
                "origin_region": "strict_post_strict_small_prefix_complement",
                "row_index_within_source_sample": j,
            })
    return rows


def scan_stream_overlap(candidate_rows: list[dict[str, Any]], research: pathlib.Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = json.loads(research.read_text(encoding="utf-8"))
    streams = summary["streams"]
    cand_by_norm = {norm_text(r["text"]): r for r in candidate_rows}
    cand_norms = set(cand_by_norm)
    hits: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    for alias, relpath in streams.items():
        path = ROOT / relpath
        t0 = time.time()
        rows = 0; occ = 0; uniq: set[int] = set()
        if not path.exists():
            metas.append({"stream_alias": alias, "stream_path": relpath, "exists": False})
            continue
        for obj in read_jsonl(path):
            rows += 1
            nt = norm_text(obj.get("text", ""))
            if nt in cand_norms:
                cr = cand_by_norm[nt]
                occ += 1
                uniq.add(int(cr["example_id"]))
                hits.append({
                    "stream_alias": alias,
                    "candidate_example_id": int(cr["example_id"]),
                    "candidate_source": cr["source"],
                    "stream_source": obj.get("source", ""),
                    "stream_example_id": obj.get("example_id"),
                    "match_type": "normalized_full_row_equality",
                })
        metas.append({"stream_alias": alias, "stream_path": relpath, "exists": True, "rows_scanned": rows, "hit_occurrences": occ, "unique_candidate_rows_hit": len(uniq), "elapsed_sec": round(time.time() - t0, 2)})
        print(json.dumps({"event": "stream_candidate_scan", "alias": alias, "rows": rows, "unique_hits": len(uniq), "elapsed_sec": round(time.time()-t0, 2)}, ensure_ascii=False), flush=True)
    return hits, metas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--skip-stream-scan", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    small_sha, small_meta = get_file_meta(api, SMALL_ID)
    strict_sha, strict_meta = get_file_meta(api, STRICT_ID)

    prefix_rows: list[dict[str, Any]] = []
    all_prefix_ok = True
    small_dir = out / "strict_small_snapshot"
    for fn in FILES:
        local = pathlib.Path(hf_hub_download(repo_id=SMALL_ID, repo_type="dataset", revision=SMALL_REV, filename=fn, local_dir=small_dir))
        small_size = local.stat().st_size
        small_hash = sha256_path(local)
        strict_size = int(strict_meta[fn]["size"])
        url = hf_hub_url(repo_id=STRICT_ID, filename=fn, repo_type="dataset")
        b = get_range(url, 0, small_size - 1)
        remote_hash = sha256_bytes(b)
        ok = (len(b) == small_size and remote_hash == small_hash)
        all_prefix_ok = all_prefix_ok and ok
        prefix_rows.append({
            "file": fn,
            "source": SOURCE_OF_FILE[fn],
            "small_local_path": rel(local),
            "small_size": small_size,
            "strict_size": strict_size,
            "strict_minus_small_bytes": strict_size - small_size,
            "small_sha256": small_hash,
            "strict_first_small_bytes_sha256": remote_hash,
            "bytes_received": len(b),
            "prefix_match": ok,
            "small_blob_id": small_meta.get(fn, {}).get("blob_id"),
            "strict_blob_id": strict_meta.get(fn, {}).get("blob_id"),
        })
        print(json.dumps({"event": "prefix_checked", "file": fn, "prefix_match": ok, "small_size": small_size, "strict_size": strict_size}, ensure_ascii=False), flush=True)

    write_csv(out / "strict_small_prefix_check.csv", prefix_rows)
    rows: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    target_rows = heldout_source_targets(args.rows)
    if all_prefix_ok:
        tokens_by_source: dict[str, list[str]] = {}
        for fn in FILES:
            src = SOURCE_OF_FILE[fn]
            need_words = target_rows.get(src, 0) * 160
            if need_words <= 0:
                tokens_by_source[src] = []
                continue
            pr = next(r for r in prefix_rows if r["file"] == fn)
            toks, wins = words_from_windows(STRICT_ID, fn, int(pr["strict_size"]), int(pr["small_size"]), need_words)
            tokens_by_source[src] = toks
            window_records.extend(wins)
            print(json.dumps({"event": "sampled_complement", "source": src, "target_rows": target_rows[src], "tokens": len(toks), "windows": len(wins)}, ensure_ascii=False), flush=True)
        rows = build_rows(tokens_by_source, target_rows)
        write_jsonl(out / "strict_complement_axis_3000_rows.jsonl", rows)
        write_csv(out / "strict_complement_sampling_windows.csv", window_records)

    hits: list[dict[str, Any]] = []
    stream_metas: list[dict[str, Any]] = []
    if rows and not args.skip_stream_scan:
        hits, stream_metas = scan_stream_overlap(rows, SUMMARY)
        write_csv(out / "strict_complement_axis_stream_row_equality_hits.csv", hits)
        write_csv(out / "strict_complement_axis_stream_scan_meta.csv", stream_metas)

    source_counts = Counter(str(r.get("source", "")) for r in rows)
    payload = {
        "status": "STRICT_COMPLEMENT_AXIS_DONE" if all_prefix_ok else "STRICT_COMPLEMENT_PREFIX_FAILED",
        "created_utc": now(),
        "elapsed_sec": round(time.time() - t0, 2),
        "small_repo": {"repo_id": SMALL_ID, "revision": SMALL_REV, "commit_sha": small_sha},
        "strict_repo": {"repo_id": STRICT_ID, "commit_sha": strict_sha},
        "prefix_all_files_match": all_prefix_ok,
        "prefix_check_csv": rel(out / "strict_small_prefix_check.csv"),
        "prefix_rows": prefix_rows,
        "candidate_axis": {
            "path": rel(out / "strict_complement_axis_3000_rows.jsonl") if rows else None,
            "rows": len(rows),
            "words": sum(int(r.get("words", 0)) for r in rows),
            "target_rows_by_source_from_step015_proportions": target_rows,
            "actual_rows_by_source": dict(sorted(source_counts.items())),
            "row_format": "JSONL with text, words=160, source, high example_id; deterministic masks can use example_id without colliding with existing streams",
        },
        "stream_overlap_scan": {
            "skipped": bool(args.skip_stream_scan),
            "streams_scanned": len(stream_metas),
            "hit_occurrences": len(hits),
            "unique_candidate_rows_hit": len({h["candidate_example_id"] for h in hits}),
            "hits_csv": rel(out / "strict_complement_axis_stream_row_equality_hits.csv") if not args.skip_stream_scan else None,
            "scan_meta_csv": rel(out / "strict_complement_axis_stream_scan_meta.csv") if not args.skip_stream_scan else None,
        },
        "sampling_windows_csv": rel(out / "strict_complement_sampling_windows.csv") if rows else None,
        "summary_json": rel(out / "summary.json"),
        "scientific_interpretation": "If all prefix checks match and stream row-equality hits are zero, these rows are a same-source, post-Strict-Small-prefix candidate broad-fit axis from the larger 2026 Strict corpus. It remains a candidate until future checks verify tokenizer-training exclusion and, if needed, stronger near-duplicate or phrase-subsequence exclusion.",
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
