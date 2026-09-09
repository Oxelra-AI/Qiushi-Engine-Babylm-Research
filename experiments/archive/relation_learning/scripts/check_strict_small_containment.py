#!/usr/bin/env python3
"""research: exact containment check of Strict-Small source files in larger Strict files.

The larger 2026 Strict repository can provide a same-source unseen axis only if
we know how the Strict-Small text is embedded in it.  Prefix checks failed for
OpenSubtitles and Switchboard, so this script downloads the larger files for a
specified source subset and searches for the exact Strict-Small byte sequence.
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
import pathlib
import time
from typing import Any

from huggingface_hub import hf_hub_download

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/check_strict_small_containment.py')
ROOT = _PUBLIC_ROOT

OUT = ROOT / "experiments/archive/relation_learning/data/strict_small_containment"
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


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="*", default=["open_subtitles.train.txt", "switchboard.train.txt"])
    ap.add_argument("--keep-strict-files", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for fn in args.files:
        if fn not in FILES:
            raise ValueError(fn)
        print(json.dumps({"event": "download_start", "file": fn}, ensure_ascii=False), flush=True)
        small_path = pathlib.Path(hf_hub_download(repo_id=SMALL_ID, repo_type="dataset", revision=SMALL_REV, filename=fn, local_dir=OUT / "strict_small"))
        strict_path = pathlib.Path(hf_hub_download(repo_id=STRICT_ID, repo_type="dataset", filename=fn, local_dir=OUT / "strict_full"))
        small_b = small_path.read_bytes()
        strict_b = strict_path.read_bytes()
        pos = strict_b.find(small_b)
        # Also search after normalizing CRLF to LF in case of line-ending mismatch.
        pos_lf = -1
        if pos < 0:
            small_lf = small_b.replace(b"\r\n", b"\n")
            strict_lf = strict_b.replace(b"\r\n", b"\n")
            pos_lf = strict_lf.find(small_lf)
        row = {
            "file": fn,
            "small_path": rel(small_path),
            "strict_path": rel(strict_path) if args.keep_strict_files else rel(strict_path),
            "small_size": len(small_b),
            "strict_size": len(strict_b),
            "small_sha256": sha256_bytes(small_b),
            "strict_sha256": sha256_bytes(strict_b),
            "exact_byte_containment_offset": pos,
            "lf_normalized_containment_offset": pos_lf,
            "contains_exact": pos >= 0,
            "contains_lf_normalized": pos_lf >= 0,
            "prefix_match": pos == 0,
        }
        rows.append(row)
        print(json.dumps({"event": "containment_checked", "file": fn, "contains_exact": row["contains_exact"], "offset": pos, "contains_lf_normalized": row["contains_lf_normalized"], "offset_lf": pos_lf}, ensure_ascii=False), flush=True)
        if not args.keep_strict_files and len(strict_b) > 10_000_000:
            # The exact result is recorded; retain small files and delete the large
            # local copy to save workspace storage unless explicitly requested.
            try:
                strict_path.unlink()
            except Exception:
                pass
    write_csv(OUT / "strict_small_containment.csv", rows)
    payload = {
        "status": "STRICT_SMALL_CONTAINMENT_DONE",
        "created_utc": now(),
        "elapsed_sec": round(time.time() - t0, 2),
        "small_repo": {"repo_id": SMALL_ID, "revision": SMALL_REV},
        "strict_repo": {"repo_id": STRICT_ID},
        "files_checked": args.files,
        "rows": rows,
        "csv": rel(OUT / "strict_small_containment.csv"),
        "summary_json": rel(OUT / "summary.json"),
        "scientific_interpretation": "Exact containment identifies whether the Strict-Small source file can be removed as one contiguous byte block from the larger Strict file. If containment fails, a complement axis for that source needs row- or phrase-level exclusion rather than simple prefix/postfix sampling.",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
