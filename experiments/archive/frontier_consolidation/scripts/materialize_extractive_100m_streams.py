#!/usr/bin/env python3
"""research: materialize source-only extractive 100M training streams.

Uses the existing compact_view_reinvest 100M stream as the control row order.
Only rows whose example_id is in the FineWeb compact changed block
(950000..953004) are replaced by the corresponding source-only extractive text
from research. All other rows are copied byte-for-byte from the compact stream.

This is CPU/file construction only: no training, no evaluation, no upload, no
AoA, and no leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
COMPACT_100M = WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
COMPACT_10M = WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
DIR = WS / "data/extractive_view_pool_preflight"
PREFLIGHT = DIR / "extractive_view_pool_preflight.json"
OUT_DIR = WS / "data/extractive_view_100m_streams"
CHANGED_MIN = 950000
CHANGED_MAX = 953004
EXPECTED_ROWS = 647400
EXPECTED_WORDS = 100000000
EXPECTED_CHANGED_ROWS = 30050
EXPECTED_CHANGED_UNIQUE = 3005


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_words(text: str) -> list[str]:
    return [w for w in text.split() if w]


def load_variant_map(variant: str) -> dict[int, dict[str, Any]]:
    path = DIR / f"extractive_{variant}_10M.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    mapping: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            eid = obj.get("example_id")
            if isinstance(eid, int) and CHANGED_MIN <= eid <= CHANGED_MAX:
                actual = len(split_words(str(obj.get("text", ""))))
                words = int(obj.get("words", -1))
                if words != actual:
                    raise RuntimeError(f"{variant} 10M word mismatch line {line_no}: field={words} actual={actual}")
                if eid in mapping:
                    raise RuntimeError(f"duplicate changed example_id in {variant} 10M: {eid}")
                mapping[eid] = obj
    if len(mapping) != EXPECTED_CHANGED_UNIQUE:
        raise RuntimeError(f"{variant} mapping has {len(mapping)} changed ids, expected {EXPECTED_CHANGED_UNIQUE}")
    missing = [eid for eid in range(CHANGED_MIN, CHANGED_MAX + 1) if eid not in mapping]
    if missing:
        raise RuntimeError(f"{variant} missing changed ids: {missing[:10]} ...")
    return mapping


def count_compact10_changed_words() -> dict[int, int]:
    counts: dict[int, int] = {}
    with COMPACT_10M.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            eid = obj.get("example_id")
            if isinstance(eid, int) and CHANGED_MIN <= eid <= CHANGED_MAX:
                counts[eid] = int(obj["words"])
    return counts


def materialize_variant(variant: str, mapping: dict[int, dict[str, Any]], compact_changed_words: dict[int, int], out_dir: Path) -> dict[str, Any]:
    out_path = out_dir / f"extractive_{variant}_100M.jsonl"
    rows = 0
    words = 0
    changed_rows = 0
    filler_rows = 0
    unique_changed: dict[int, int] = {}
    changed_word_mismatches: list[dict[str, Any]] = []
    changed_line_samples: list[dict[str, Any]] = []
    filler_byte_mismatch_count = 0  # by construction copied lines are identical

    with COMPACT_100M.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            eid = obj.get("example_id")
            if isinstance(eid, int) and CHANGED_MIN <= eid <= CHANGED_MAX:
                rep = mapping[eid]
                # The extractive row must preserve the legal charged word count of the compact row.
                if int(rep["words"]) != int(obj["words"]):
                    changed_word_mismatches.append({
                        "line_no": line_no,
                        "example_id": eid,
                        "compact_words": obj.get("words"),
                        "extractive_words": rep.get("words"),
                    })
                if compact_changed_words.get(eid) != int(obj["words"]):
                    changed_word_mismatches.append({
                        "line_no": line_no,
                        "example_id": eid,
                        "compact_10m_words": compact_changed_words.get(eid),
                        "compact_100m_words": obj.get("words"),
                    })
                json.dump(rep, fout, ensure_ascii=False)
                fout.write("\n")
                rows += 1
                words += int(rep["words"])
                changed_rows += 1
                unique_changed[eid] = unique_changed.get(eid, 0) + 1
                if len(changed_line_samples) < 8:
                    changed_line_samples.append({
                        "line_no": line_no,
                        "example_id": eid,
                        "words": rep["words"],
                        "old_source": obj.get("source"),
                        "new_source": rep.get("source"),
                        "new_text_prefix": str(rep.get("text", ""))[:160],
                    })
            else:
                # Preserve all filler/topup/qwen/official rows byte-identically.
                fout.write(line)
                rows += 1
                words += int(obj.get("words", len(split_words(str(obj.get("text", ""))))))
                filler_rows += 1

    bad_repeat_counts = {str(eid): c for eid, c in unique_changed.items() if c != 10}
    sha = sha256_file(out_path)
    status_ok = (
        rows == EXPECTED_ROWS and words == EXPECTED_WORDS and changed_rows == EXPECTED_CHANGED_ROWS
        and filler_rows == EXPECTED_ROWS - EXPECTED_CHANGED_ROWS
        and len(unique_changed) == EXPECTED_CHANGED_UNIQUE
        and not bad_repeat_counts and not changed_word_mismatches
    )
    return {
        "variant": f"extractive_{variant}",
        "path": str(out_path),
        "sha256": sha,
        "rows": rows,
        "words": words,
        "changed_rows_replaced": changed_rows,
        "filler_rows_byte_identical_to_compact_by_construction": filler_rows,
        "filler_byte_mismatch_count": filler_byte_mismatch_count,
        "unique_changed_ids": len(unique_changed),
        "changed_id_repeat_count_bad": bad_repeat_counts,
        "changed_word_mismatches": changed_word_mismatches[:20],
        "changed_line_samples": changed_line_samples,
        "status_ok": status_ok,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["balanced", "wide"], choices=["balanced", "wide"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [COMPACT_100M, COMPACT_10M, PREFLIGHT]:
        if not p.exists():
            raise FileNotFoundError(p)

    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    compact_changed_words = count_compact10_changed_words()
    if len(compact_changed_words) != EXPECTED_CHANGED_UNIQUE:
        raise RuntimeError(f"compact 10M changed ids {len(compact_changed_words)} != {EXPECTED_CHANGED_UNIQUE}")

    results = []
    for variant in args.variants:
        mapping = load_variant_map(variant)
        results.append(materialize_variant(variant, mapping, compact_changed_words, out_dir))

    # The research token counts are exact per 10M pool. Since the 100M stream is ten
    # full repetitions of the same rows in shuffled order, the 100M active-token
    # deltas are exactly 10x the 10M deltas.
    stream_tokens_10m = preflight.get("comparison", {}).get("stream_tokens_10M", {})
    changed_tokens_10m = preflight.get("comparison", {}).get("changed_block_active_tokens", {})
    token_summary: dict[str, Any] = {}
    for variant in ["extractive_balanced", "extractive_wide", "compact", "repeat"]:
        if variant in stream_tokens_10m:
            token_summary[variant] = {
                "active_tokens_10M": stream_tokens_10m[variant],
                "active_tokens_100M_x10": int(stream_tokens_10m[variant]) * 10,
                "changed_block_active_tokens_10M": changed_tokens_10m.get(variant),
                "changed_block_active_tokens_100M_x10": (int(changed_tokens_10m[variant]) * 10 if variant in changed_tokens_10m else None),
            }
    for variant in ["extractive_balanced", "extractive_wide"]:
        key = variant
        if key in token_summary and "compact" in token_summary:
            token_summary[key]["delta_vs_compact_active_tokens_100M"] = token_summary[key]["active_tokens_100M_x10"] - token_summary["compact"]["active_tokens_100M_x10"]
            token_summary[key]["delta_vs_compact_changed_block_active_tokens_100M"] = token_summary[key]["changed_block_active_tokens_100M_x10"] - token_summary["compact"]["changed_block_active_tokens_100M_x10"]

    payload = {
        "status": "EXTRACTIVE_100M_STREAMS_MATERIALIZED",
        "created_utc": now(),
        "meaning": "100M source-only extractive training streams built by replacing only FineWeb compact changed-block rows in the compact 100M stream; all other rows are byte-identical to compact. CPU/file-only.",
        "input_compact_100M": str(COMPACT_100M),
        "input_compact_100M_sha256": sha256_file(COMPACT_100M),
        "input_step224_preflight": str(PREFLIGHT),
        "changed_id_range": [CHANGED_MIN, CHANGED_MAX],
        "expected": {
            "rows": EXPECTED_ROWS,
            "words": EXPECTED_WORDS,
            "changed_rows": EXPECTED_CHANGED_ROWS,
            "changed_unique_ids": EXPECTED_CHANGED_UNIQUE,
            "filler_rows": EXPECTED_ROWS - EXPECTED_CHANGED_ROWS,
        },
        "variant_results": results,
        "token_summary_from_step224_x10": token_summary,
        "paired_run_interpretation": {
            "why_two_arms": "balanced and wide span the irreducible density/coverage/token-mass tradeoff; balanced-only compact superiority would be ambiguous.",
            "balanced_role": "closest content-density and token-mass match to compact but higher source coverage, zero source-absent content, telegraphic surface.",
            "wide_role": "content/coverage-maximized source-only view with token mass close to compact but much higher content density and near-complete source coverage.",
            "one_sided_positive_reading": "if either source-only variant approaches compact and beats repeat on late stable selected families, source-word selection can reproduce much of the compact effect.",
            "one_sided_negative_reading": "if both source-only variants lag compact, the deficit is bundle-level evidence for natural generated compact re-expression plus fluency/source-absent/context distribution, not isolation of a single factor.",
        },
        "no_training_official_eval_upload_aoa_or_leaderboard": True,
    }
    all_ok = all(r["status_ok"] for r in results)
    payload["all_status_ok"] = all_ok
    out_json = out_dir / "extractive_100m_streams_manifest.json"
    out_md = out_dir / "extractive_100m_streams_manifest.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research extractive 100M streams",
        "",
        f"Created UTC: `{payload['created_utc']}`",
        f"All status ok: `{all_ok}`",
        "",
        "These streams replace only example_id 950000--953004 rows in the compact 100M stream; every other line is copied byte-for-byte.",
        "",
        "| variant | rows | words | changed rows | filler rows | sha256 | ok |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        lines.append(f"| {r['variant']} | {r['rows']} | {r['words']} | {r['changed_rows_replaced']} | {r['filler_rows_byte_identical_to_compact_by_construction']} | `{r['sha256']}` | {r['status_ok']} |")
    lines += [
        "",
        "## Token-mass summary (research exact 10M counts ×10)",
        "",
        "| arm | active tokens 100M | changed-block active tokens 100M | delta vs compact active | delta vs compact changed |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in ["compact", "repeat", "extractive_balanced", "extractive_wide"]:
        if k in token_summary:
            rec = token_summary[k]
            lines.append(f"| {k} | {rec.get('active_tokens_100M_x10')} | {rec.get('changed_block_active_tokens_100M_x10')} | {rec.get('delta_vs_compact_active_tokens_100M', '')} | {rec.get('delta_vs_compact_changed_block_active_tokens_100M', '')} |")
    lines += [
        "",
        "The paired balanced+wide comparison is required because the extractive construction cannot match compact simultaneously on density, source coverage, source-absent content, fluency, and token mass.",
        f"JSON: `{out_json}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "all_status_ok": all_ok, "variants": results, "no_training_official_eval_upload_aoa_or_leaderboard": True}, indent=2), flush=True)
    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
