#!/usr/bin/env python3
"""research: build exact 10M DeBERTa-style reinvest pools for skeleton dissection.

This reconstructs the compact-view changed block from the pair-row metadata and
replaces each compact rewrite with one of the audited exact-length source-derived
views.  The purpose is to prove a future skeleton-vs-compact-vs-prefix screen can
be made mechanically matched: same source text, same pair row grouping, same row
word counts, same filler rows, same exact 10M legal words.

No tokenizer training, model training, or evaluation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
BASE = ROOT / "data/density_cleanqwen_overlay_medium_riskhard"
PAIRS = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
VARIANT_DIR = ROOT / "data/extractive_skeleton_variant_audit"
DEFAULT_OUT = ROOT / "data/skeleton_reinvest_pool_scaffold"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(text.split())


def load_pairs(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pid = str(r.get("pair_id"))
                out[pid] = r
    return out


def load_variant_views(variant_dir: pathlib.Path, variants: list[str]) -> dict[str, dict[str, str]]:
    views: dict[str, dict[str, str]] = {v: {} for v in variants}
    for v in variants:
        p = variant_dir / f"{v}_pairs.jsonl"
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    views[v][str(r["pair_id"])] = str(r["view_text"])
    return views


def load_meta(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_filler(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_existing_changed_rows(path: pathlib.Path, n_changed: int) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_changed:
                break
            if line.strip():
                rows[i] = json.loads(line)
    return rows


def build_changed_rows(meta_rows: list[dict[str, Any]], pairs: dict[str, dict[str, Any]], view_by_pid: dict[str, str], source_label: str, existing_changed: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in meta_rows:
        row_i = int(m.get("row_index", len(out)))
        pair_ids = [str(x) for x in m["pair_ids"]]
        if not pair_ids:
            # Preserve top-up/non-pair changed-block rows exactly; research has a
            # 9-word OpenSubtitles top-up with example_id 1050000.
            existing = existing_changed.get(row_i)
            if existing is None:
                raise ValueError(f"row {row_i} has no pair_ids and no existing changed row")
            text = str(existing["text"]).strip()
            words = wc(text)
        else:
            parts = []
            for pid in pair_ids:
                pr = pairs[pid]
                view = view_by_pid[pid]
                parts.append(str(pr["source_text"]).strip())
                parts.append(str(view).strip())
            text = " ".join(p for p in parts if p)
            words = wc(text)
        if words != int(m["words"]):
            raise ValueError(f"row {m.get('row_index')} word mismatch {words} != {m['words']} for {source_label}")
        out.append({
            "text": text,
            "words": words,
            "example_id": int(m["example_id"]),
            "source": source_label,
        })
    return out


def write_jsonl(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "words": sum(int(r.get("words", 0)) for r in rows),
        "first_example_ids": [r.get("example_id") for r in rows[:5]],
        "last_example_ids": [r.get("example_id") for r in rows[-5:]],
        "source_counts": {k: v for k, v in sorted(__import__('collections').Counter(str(r.get("source")) for r in rows).items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--base-dir", type=pathlib.Path, default=BASE)
    ap.add_argument("--pairs", type=pathlib.Path, default=PAIRS)
    ap.add_argument("--variant-dir", type=pathlib.Path, default=VARIANT_DIR)
    ap.add_argument("--variants", nargs="+", default=["compact", "prefix_repeat", "content_spread", "scored_source_skeleton"])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.pairs)
    views = load_variant_views(args.variant_dir, args.variants)
    meta = load_meta(args.base_dir / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl")
    existing_changed = load_existing_changed_rows(args.base_dir / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl", len(meta))
    filler = load_filler(args.base_dir / "common_filler_rows.jsonl")
    filler_words = sum(int(r["words"]) for r in filler)
    manifest = {
        "status": "SKELETON_REINVEST_POOL_SCAFFOLD",
        "pairs_path": str(args.pairs),
        "pairs_sha256": sha256_file(args.pairs),
        "base_dir": str(args.base_dir),
        "variant_dir": str(args.variant_dir),
        "filler_rows": len(filler),
        "filler_words": filler_words,
        "changed_meta_rows": len(meta),
        "changed_meta_words": sum(int(r["words"]) for r in meta),
        "variants": {},
        "checks": [],
        "notes": [
            "all variant views are exact same word count as compact rewrite by construction",
            "all pool rows are changed block followed by the exact common filler rows, matching research order",
            "files are scaffolds for future low-cost MLM training screens, not trained or evaluated here",
            "legal final use would require explicit tokenizer-coordinate decision before training",
        ],
    }
    for v in args.variants:
        changed = build_changed_rows(meta, pairs, views[v], f"step181_{v}_skeleton_reinvest", existing_changed)
        rows = changed + filler
        out_path = args.out_dir / f"cleanqwen_fineweb_{v}_skeleton_reinvest_10M.jsonl"
        write_jsonl(rows, out_path)
        changed_path = args.out_dir / f"cleanqwen_fineweb_{v}_skeleton_reinvest_changed_block_rows_meta.jsonl"
        # Preserve row-pair metadata exactly, with variant label.
        with changed_path.open("w", encoding="utf-8") as f:
            for m in meta:
                mm = dict(m)
                mm["variant"] = v
                f.write(json.dumps(mm, ensure_ascii=False) + "\n")
        summary = summarize_rows(rows)
        summary.update({
            "path": str(out_path),
            "sha256": sha256_file(out_path),
            "changed_block_path": str(changed_path),
            "changed_block_sha256": sha256_file(changed_path),
            "changed_rows": len(changed),
            "changed_words": sum(int(r["words"]) for r in changed),
            "exact_10M_words": sum(int(r["words"]) for r in rows) == 10_000_000,
            "row_count_matches_step015": len(rows) == 64_740,
        })
        manifest["variants"][v] = summary
        if not summary["exact_10M_words"]:
            manifest["checks"].append(f"{v}: not exact 10M")
        if not summary["row_count_matches_step015"]:
            manifest["checks"].append(f"{v}: row count mismatch")
    manifest["ok"] = not manifest["checks"]
    out_json = args.out_dir / "skeleton_reinvest_pool_manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = args.out_dir / "skeleton_reinvest_pool_manifest.md"
    lines = ["# research skeleton reinvest pool scaffold", ""]
    lines.append("Exact 10M pool scaffolds; no training or evaluation.")
    lines.append("")
    lines.append(f"OK: `{manifest['ok']}`; filler words {filler_words}; changed rows {len(meta)}; changed words {manifest['changed_meta_words']}")
    lines.append("")
    lines.append("| variant | rows | words | changed rows | changed words | exact 10M | sha256 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for v, s in manifest["variants"].items():
        lines.append(f"| {v} | {s['rows']} | {s['words']} | {s['changed_rows']} | {s['changed_words']} | {s['exact_10M_words']} | `{s['sha256']}` |")
    lines.append("")
    lines.append("## Scientific use")
    lines.append("These pools show that a source-wide skeleton dissection can preserve the exact research legal word budget, pair-row grouping, row positions, and common filler. They do not establish a training result. A training comparison remains conditional on the measured distribution shift and the pending DeBERTa common-grid and triangle results.")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "ok": manifest["ok"], "out_dir": str(args.out_dir), "variants": list(manifest["variants"].keys())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
