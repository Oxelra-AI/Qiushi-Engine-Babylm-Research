#!/usr/bin/env python3
"""research: build non-launched compact ordered/scrambled factorial pool scaffolds.

Creates exact 10M and 40M JSONL files for the cleanest available order contrast:
compact ordered vs compact word-multiset scrambled.  This is CPU/file work only and
is explicitly not a training launch.  The pools are for future use only if pending
DeBERTa trajectory and directional evidence leave this route strongest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("experiments/archive/frontier_consolidation")
BASE_DIR = ROOT / "data/density_cleanqwen_overlay_medium_riskhard"
PAIR_DIR = ROOT / "data/factorial_view_candidate_audit"
DEFAULT_OUT = ROOT / "data/compact_order_factorial_pool_scaffold"
META_PATH = BASE_DIR / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
BASE_10M = BASE_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
FILLER = BASE_DIR / "common_filler_rows.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(text.split())


def read_jsonl(path: Path, limit: int | None = None) -> List[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def load_pair_file(path: Path) -> Dict[str, dict[str, Any]]:
    out: Dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["pair_id"])] = r
    return out


def write_jsonl(rows: List[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_changed_rows(meta_rows: List[dict[str, Any]], candidate_rows: Dict[str, dict[str, Any]], existing_changed: Dict[int, dict[str, Any]], label: str, preserve_compact_source: bool) -> List[dict[str, Any]]:
    out: List[dict[str, Any]] = []
    mismatches = []
    for meta in meta_rows:
        row_index = int(meta.get("row_index", len(out)))
        pair_ids = [str(x) for x in meta.get("pair_ids", [])]
        existing = existing_changed.get(row_index)
        if not pair_ids:
            if existing is None:
                raise ValueError(f"missing top-up existing changed row {row_index}")
            # Preserve the 9-word top-up/non-pair row exactly.
            obj = dict(existing)
            out.append(obj)
            continue
        parts: List[str] = []
        for pid in pair_ids:
            r = candidate_rows[pid]
            parts.append(str(r["source_text"]).strip())
            parts.append(str(r["view_text"]).strip())
        text = " ".join(p for p in parts if p)
        words = wc(text)
        if words != int(meta["words"]):
            raise ValueError(f"word mismatch row {row_index}: {words} != {meta['words']}")
        source = str(existing.get("source", label)) if (preserve_compact_source and existing is not None) else label
        obj = {"text": text, "words": words, "example_id": int(meta["example_id"]), "source": source}
        if preserve_compact_source and existing is not None:
            if text != str(existing.get("text", "")).strip() or words != int(existing.get("words", -1)) or int(existing.get("example_id", -1)) != int(meta["example_id"]):
                mismatches.append(row_index)
        out.append(obj)
    if mismatches:
        raise RuntimeError(f"compact reconstruction mismatched existing changed rows: first {mismatches[:10]} total {len(mismatches)}")
    return out


def repeat_rows(rows: List[dict[str, Any]], passes: int) -> List[dict[str, Any]]:
    out: List[dict[str, Any]] = []
    for p in range(passes):
        for r in rows:
            rr = dict(r)
            # Text/words/example_id are what training uses.  Source is only manifest metadata.
            rr["source"] = f"pass{p+1}::{rr.get('source', '')}"
            out.append(rr)
    return out


def summarize(rows: List[dict[str, Any]], path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "words": sum(int(r["words"]) for r in rows),
        "first_example_ids": [r.get("example_id") for r in rows[:5]],
        "last_example_ids": [r.get("example_id") for r in rows[-5:]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--build-40m", action="store_true", default=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = read_jsonl(META_PATH)
    existing_changed = {i: r for i, r in enumerate(read_jsonl(BASE_10M, limit=len(meta_rows)))}
    filler = read_jsonl(FILLER)
    compact_pairs = load_pair_file(PAIR_DIR / "compact_candidate_pairs.jsonl")
    scrambled_pairs = load_pair_file(PAIR_DIR / "compact_scrambled_candidate_pairs.jsonl")

    variants = {
        "compact_ordered": build_changed_rows(meta_rows, compact_pairs, existing_changed, "compact_ordered_factorial", preserve_compact_source=True),
        "compact_scrambled": build_changed_rows(meta_rows, scrambled_pairs, existing_changed, "compact_scrambled_factorial", preserve_compact_source=False),
    }

    manifest: dict[str, Any] = {
        "status": "COMPACT_ORDER_FACTORIAL_POOL_SCAFFOLD",
        "not_launched": True,
        "scientific_boundary": "Pool scaffold only. Do not train until delivered DeBERTa grids and A01 compact directional fork/triangle evidence are read.",
        "base_10M": str(BASE_10M),
        "base_10M_sha256": sha256_file(BASE_10M),
        "meta_path": str(META_PATH),
        "meta_sha256": sha256_file(META_PATH),
        "filler_path": str(FILLER),
        "filler_sha256": sha256_file(FILLER),
        "changed_rows": len(meta_rows),
        "changed_words": sum(int(r["words"]) for r in meta_rows),
        "filler_rows": len(filler),
        "filler_words": sum(int(r["words"]) for r in filler),
        "variants": {},
        "checks": [],
    }

    for v, changed in variants.items():
        ten_rows = changed + filler
        ten_path = args.out_dir / f"{v}_10M.jsonl"
        write_jsonl(ten_rows, ten_path)
        s10 = summarize(ten_rows, ten_path)
        s10["exact_10M_words"] = (s10["words"] == 10_000_000)
        s10["row_count_matches_step015"] = (s10["rows"] == 64_740)
        if not s10["exact_10M_words"]:
            manifest["checks"].append(f"{v} 10M word mismatch {s10['words']}")
        if not s10["row_count_matches_step015"]:
            manifest["checks"].append(f"{v} row mismatch {s10['rows']}")
        entry: dict[str, Any] = {"10M": s10}
        if v == "compact_ordered":
            # Compare generated compact-order file to original at text/words/example_id level.
            original_rows = read_jsonl(BASE_10M)
            text_equal = all(str(a.get("text", "")).strip() == str(b.get("text", "")).strip() and int(a.get("words", -1)) == int(b.get("words", -2)) and int(a.get("example_id", -1)) == int(b.get("example_id", -2)) for a, b in zip(ten_rows, original_rows))
            entry["matches_original_text_words_example_id"] = text_equal and len(ten_rows) == len(original_rows)
        if args.build_40m:
            forty_rows = repeat_rows(ten_rows, 4)
            forty_path = args.out_dir / f"{v}_40M.jsonl"
            write_jsonl(forty_rows, forty_path)
            s40 = summarize(forty_rows, forty_path)
            s40["exact_40M_words"] = (s40["words"] == 40_000_000)
            s40["passes"] = 4
            if not s40["exact_40M_words"]:
                manifest["checks"].append(f"{v} 40M word mismatch {s40['words']}")
            entry["40M"] = s40
        manifest["variants"][v] = entry

    manifest["ok"] = not manifest["checks"]
    mpath = args.out_dir / "compact_order_factorial_pool_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research compact order factorial pool scaffold", "", "Non-launched CPU scaffold for compact ordered vs compact scrambled.", "", f"OK: `{manifest['ok']}`", "", "| variant | 10M rows | 10M words | 10M sha | 40M rows | 40M words | 40M sha |", "|---|---:|---:|---|---:|---:|---|"]
    for v, e in manifest["variants"].items():
        s10 = e["10M"]; s40 = e.get("40M", {})
        lines.append(f"| {v} | {s10['rows']} | {s10['words']} | `{s10['sha256']}` | {s40.get('rows','')} | {s40.get('words','')} | `{s40.get('sha256','')}` |")
    lines += ["", "## Boundary", "These files are not a training decision.  They are usable only after the delivered DeBERTa common-grid results and A01 directional fork/triangle evidence are read.  The current clean estimand is ordered compact context at fixed compact lexical multiset; it does not isolate tail coverage, budget efficiency, or semantic transformation.", "", f"JSON: `{mpath}`"]
    (args.out_dir / "compact_order_factorial_pool_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "ok": manifest["ok"], "out_dir": str(args.out_dir), "variants": list(manifest["variants"].keys())}, indent=2))


if __name__ == "__main__":
    main()
