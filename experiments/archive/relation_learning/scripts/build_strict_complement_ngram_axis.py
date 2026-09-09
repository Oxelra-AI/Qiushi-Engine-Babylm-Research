#!/usr/bin/env python3
"""research: build a same-source unseen axis by n-gram excluding Strict-Small text.

The larger `BabyLM-community/BabyLM-2026-Strict` repo exists, but Strict-Small is
not a simple byte prefix for every source.  This script therefore constructs a
candidate broad-fit axis by sampling deterministic 160-word rows from the larger
Strict source files while rejecting any row that shares a same-source contiguous
n-gram with the pinned Strict-Small source file.  It then optionally scans exact
normalized-row equality against the known relation-composition streams.

The resulting axis is a research measurement substrate, not a final conclusion:
it is source-matched and strict-small-ngram-excluded, but downstream use should
retain the recorded n-gram threshold and stream-equality screen.
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
from collections import Counter
from typing import Any

import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/build_strict_complement_ngram_axis.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
OUT_DEFAULT = WS / "data/strict_complement_ngram_axis"
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
FILE_OF_SOURCE = {v: k for k, v in SOURCE_OF_FILE.items()}
FULL_HELDOUT = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl"
SUMMARY = WS / "data/relation_stream_exposure_screen/summary.json"
TOKEN_RE = re.compile(r"\S+")


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm_token(t: str) -> str:
    return t.lower()


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


def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def stable_ngram_hash(tokens: list[str], start: int, n: int) -> int:
    # Deterministic 64-bit digest over normalized whitespace tokens.  hashlib is
    # slower than Python's randomized hash but makes the recorded threshold
    # exactly reproducible and avoids cross-run hash seeds.
    h = hashlib.blake2b(digest_size=8)
    for j in range(start, start + n):
        h.update(tokens[j].encode("utf-8", errors="ignore")); h.update(b"\0")
    return int.from_bytes(h.digest(), "little")


def tokenise_file(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [norm_token(t) for t in TOKEN_RE.findall(text)]


def build_ngram_set(tokens: list[str], n: int) -> set[int]:
    if len(tokens) < n:
        return set()
    return {stable_ngram_hash(tokens, i, n) for i in range(0, len(tokens) - n + 1)}


def row_hits_small_ngram(tokens: list[str], small_ngrams: set[int], n: int) -> bool:
    if len(tokens) < n:
        return False
    for i in range(0, len(tokens) - n + 1):
        if stable_ngram_hash(tokens, i, n) in small_ngrams:
            return True
    return False


def repo_file_meta(repo_id: str) -> tuple[str | None, dict[str, Any]]:
    api = HfApi()
    info = api.dataset_info(repo_id=repo_id, files_metadata=True)
    return getattr(info, "sha", None), {s.rfilename: {"size": getattr(s, "size", None), "blob_id": getattr(s, "blob_id", None)} for s in info.siblings}


def get_range(url: str, start: int, end: int, timeout: int = 120) -> bytes:
    if end < start:
        return b""
    target_len = end - start + 1
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=timeout)
    r.raise_for_status()
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
    for k, v in raw.items():
        if v > 0 and targets[k] == 0:
            targets[k] = 1
    while sum(targets.values()) < n_rows:
        k = max(raw, key=lambda x: raw[x] - targets[x])
        targets[k] += 1
    while sum(targets.values()) > n_rows:
        k = max(targets, key=lambda x: targets[x] - raw[x])
        targets[k] -= 1
    return dict(sorted(targets.items()))


def sample_source_rows(source: str, strict_size: int, target_rows: int, small_ngrams: set[int], ngram_n: int, out_records: list[dict[str, Any]], prefix_skip_bytes: int = 0, max_rounds: int = 80) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fn = FILE_OF_SOURCE[source]
    url = hf_hub_url(repo_id=STRICT_ID, filename=fn, repo_type="dataset")
    rows: list[dict[str, Any]] = []
    seen_norms: set[str] = set()
    reject_ngram = 0
    reject_dup = 0
    candidate_rows = 0
    # Use deterministic windows spread over available bytes.  For prefix-matching
    # files, skip the byte prefix known to be Strict-Small; for non-prefix files,
    # use the full Strict file and rely on n-gram exclusion.
    low = min(max(0, prefix_skip_bytes), max(0, strict_size - 1))
    span = max(1, strict_size - low)
    # Each 160-word row averages roughly 800-900 bytes in these corpora.  Windows
    # are larger than needed to absorb rejections.
    win_size = 450_000
    for wi in range(max_rounds):
        if len(rows) >= target_rows:
            break
        if span <= win_size:
            start = low
        else:
            # Low-discrepancy deterministic positions.  The multiplier by an
            # irrational fraction prevents clustering at early windows.
            frac = ((wi * 0.6180339887498949) % 1.0)
            start = low + int(frac * (span - win_size))
        end = min(strict_size - 1, start + win_size - 1)
        b = get_range(url, start, end)
        text = b.decode("utf-8", errors="ignore")
        if len(text) > 600:
            text = text[300:-300]
        toks = [norm_token(t) for t in TOKEN_RE.findall(text)]
        produced_here = 0
        for off in range(0, max(0, len(toks) - 159), 160):
            if len(rows) >= target_rows:
                break
            row_toks = toks[off:off+160]
            if len(row_toks) < 160:
                continue
            candidate_rows += 1
            if row_hits_small_ngram(row_toks, small_ngrams, ngram_n):
                reject_ngram += 1
                continue
            txt = " ".join(row_toks)
            nt = norm_text(txt)
            if nt in seen_norms:
                reject_dup += 1
                continue
            seen_norms.add(nt)
            rows.append({
                "text": txt,
                "words": 160,
                "example_id": 910_000_000 + len(source) * 1_000_000 + len(rows),
                "source": source,
                "origin_repo": STRICT_ID,
                "origin_file": fn,
                "origin_region": "strict_ngram_excluded_from_pinned_strict_small",
                "strict_byte_window_start": start,
                "strict_byte_window_end": end,
                "window_index": wi,
                "row_offset_in_window_tokens": off,
                "strict_small_ngram_exclusion_n": ngram_n,
            })
            produced_here += 1
        out_records.append({"source": source, "window_index": wi, "byte_start": start, "byte_end": end, "bytes_received": len(b), "tokens": len(toks), "candidate_rows": candidate_rows, "accepted_rows_cumulative": len(rows), "accepted_rows_this_window": produced_here, "reject_ngram_cumulative": reject_ngram, "reject_duplicate_cumulative": reject_dup})
        print(json.dumps({"event": "sample_window", "source": source, "window": wi, "accepted": len(rows), "target": target_rows, "reject_ngram": reject_ngram}, ensure_ascii=False), flush=True)
    meta = {"source": source, "target_rows": target_rows, "accepted_rows": len(rows), "candidate_rows": candidate_rows, "reject_ngram": reject_ngram, "reject_duplicate": reject_dup, "windows_used": len([r for r in out_records if r["source"] == source])}
    if len(rows) < target_rows:
        raise RuntimeError(f"source {source} accepted {len(rows)} < target {target_rows}; meta={meta}")
    return rows, meta


def scan_stream_overlap(candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cand_by_norm = {norm_text(r["text"]): r for r in candidate_rows}
    cand_norms = set(cand_by_norm)
    hits: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    for alias, relpath in summary["streams"].items():
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
        print(json.dumps({"event": "stream_scan", "alias": alias, "unique_hits": len(uniq), "rows_scanned": rows, "elapsed_sec": round(time.time()-t0, 2)}, ensure_ascii=False), flush=True)
    return hits, metas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--ngram-n", type=int, default=16)
    ap.add_argument("--skip-stream-scan", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    small_dir = out / "strict_small_snapshot"

    small_sha, small_meta = repo_file_meta(SMALL_ID)
    strict_sha, strict_meta = repo_file_meta(STRICT_ID)
    target_rows = heldout_source_targets(args.rows)

    prefix_info_path = WS / "data/strict_complement_axis/strict_small_prefix_check.csv"
    prefix_skip: dict[str, int] = {}
    if prefix_info_path.exists():
        with prefix_info_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if str(r.get("prefix_match", "")).lower() == "true":
                    prefix_skip[str(r["source"])] = int(r["small_size"])

    all_rows: list[dict[str, Any]] = []
    small_file_meta: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    source_metas: list[dict[str, Any]] = []
    for fn in FILES:
        source = SOURCE_OF_FILE[fn]
        need = target_rows.get(source, 0)
        if need <= 0:
            continue
        local = pathlib.Path(hf_hub_download(repo_id=SMALL_ID, repo_type="dataset", revision=SMALL_REV, filename=fn, local_dir=small_dir))
        toks = tokenise_file(local)
        ngs = build_ngram_set(toks, args.ngram_n)
        small_file_meta.append({"source": source, "file": fn, "small_path": rel(local), "small_size": local.stat().st_size, "small_sha256": sha256_path(local), "small_tokens": len(toks), "ngram_n": args.ngram_n, "ngram_count": len(ngs), "prefix_skip_bytes_if_available": prefix_skip.get(source, 0)})
        rows, meta = sample_source_rows(source, int(strict_meta[fn]["size"]), need, ngs, args.ngram_n, window_records, prefix_skip_bytes=prefix_skip.get(source, 0))
        all_rows.extend(rows)
        source_metas.append(meta)
        # release ngram set before the next source
        del ngs, toks

    # Reassign example ids source-stably and uniquely after concatenation.
    for i, r in enumerate(all_rows):
        r["example_id"] = 920_000_000 + i
        r["axis_row_index"] = i

    axis_path = out / "strict_complement_ngram_axis_3000_rows.jsonl"
    write_jsonl(axis_path, all_rows)
    write_csv(out / "strict_small_ngram_source_meta.csv", small_file_meta)
    write_csv(out / "strict_complement_sampling_windows.csv", window_records)
    write_csv(out / "strict_complement_source_sampling_meta.csv", source_metas)

    hits: list[dict[str, Any]] = []
    stream_metas: list[dict[str, Any]] = []
    if not args.skip_stream_scan:
        hits, stream_metas = scan_stream_overlap(all_rows)
        write_csv(out / "strict_complement_ngram_axis_stream_row_equality_hits.csv", hits)
        write_csv(out / "strict_complement_ngram_axis_stream_scan_meta.csv", stream_metas)

    # Confirm each candidate row still passes its same-source n-gram screen by
    # reusing accepted counts from construction; full rescreen would rebuild the
    # large n-gram sets and is unnecessary here.
    source_counts = Counter(r["source"] for r in all_rows)
    text_lens = [len(r["text"]) for r in all_rows]
    payload = {
        "status": "STRICT_COMPLEMENT_NGRAM_AXIS_DONE",
        "created_utc": now(),
        "elapsed_sec": round(time.time() - t0, 2),
        "small_repo": {"repo_id": SMALL_ID, "revision": SMALL_REV, "commit_sha": small_sha},
        "strict_repo": {"repo_id": STRICT_ID, "commit_sha": strict_sha},
        "ngram_exclusion": {"n_tokens": args.ngram_n, "same_source_only": True, "rejection_rule": "reject any 160-word candidate row containing a normalized same-source contiguous n-gram from the pinned Strict-Small file"},
        "candidate_axis": {"path": rel(axis_path), "rows": len(all_rows), "words": sum(int(r["words"]) for r in all_rows), "target_rows_by_source_from_step015_proportions": target_rows, "actual_rows_by_source": dict(sorted(source_counts.items())), "text_char_stats": stat([float(x) for x in text_lens])},
        "small_file_meta_csv": rel(out / "strict_small_ngram_source_meta.csv"),
        "sampling_windows_csv": rel(out / "strict_complement_sampling_windows.csv"),
        "source_sampling_meta_csv": rel(out / "strict_complement_source_sampling_meta.csv"),
        "source_sampling_meta": source_metas,
        "stream_overlap_scan": {"skipped": bool(args.skip_stream_scan), "streams_scanned": len(stream_metas), "hit_occurrences": len(hits), "unique_candidate_rows_hit": len({h["candidate_example_id"] for h in hits}), "hits_csv": rel(out / "strict_complement_ngram_axis_stream_row_equality_hits.csv") if not args.skip_stream_scan else None, "scan_meta_csv": rel(out / "strict_complement_ngram_axis_stream_scan_meta.csv") if not args.skip_stream_scan else None},
        "summary_json": rel(out / "summary.json"),
        "scientific_interpretation": "This file is a candidate same-source broad-fit axis from the larger 2026 Strict corpus, source-balanced like the research 2,647 rows and excluding pinned Strict-Small same-source 16-token spans. Zero exact stream-row equality, if observed, supports using it as an additional fit axis, but the n-gram threshold and the fact that this is sampled from the larger Strict training release must be stated when interpreting scores.",
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
