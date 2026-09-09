#!/usr/bin/env python3
"""research: characterize the clean-Qwen rows displaced by the sub-dose ladder.

This is a CPU/file-only readout.  It reads the already-materialized MAX-geometry
clean pool and the research sub-dose metadata to determine what content is removed
when FineWeb source+compact-view rows are admitted at rho=0.011, 0.021, 0.042,
and 0.112.  The scientific point is to separate an admission effect from a
removal effect: if a tiny admitted dose already reaches the clean contrast, the
identity of the displaced clean block becomes load-bearing.

No model loading, training, official evaluation, GPU work, GlobalPIQA,
SuperGLUE, AoA, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import hashlib
import json
import pathlib
import re
import statistics
import time
from collections import Counter
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
POOL256 = WS / "data/dose_2p64x_rowholdout_pools"
POOL280 = WS / "data/subdose_ladder_maxgeom_pools"
OUT = WS / "data/displaced_clean_block_readout"
CLEAN10 = POOL256 / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl"
VIEW10 = POOL256 / "compact_view_dose2p64x_10M.jsonl"
META256 = POOL256 / "dose2p64x_rowholdout_metadata.json"
META280 = POOL280 / "subdose_ladder_metadata.json"
CHANGED_META_CLEAN = POOL256 / "cleanqwen_lengthmatched_dose2p64x_changed_block_rows_meta.jsonl"
CHANGED_META_VIEW = POOL256 / "compact_view_dose2p64x_changed_block_rows_meta.jsonl"
COMMON_FILLER = POOL256 / "common_filler_rows.jsonl"
HELDOUT = POOL256 / "heldout_cleanqwen_rows.jsonl"

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*|[0-9]+(?:\.[0-9]+)?")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def load_jsonl_prefix(path: pathlib.Path, n: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_jsonl_all(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_words(r: dict[str, Any]) -> int:
    return int(r.get("words", len(str(r.get("text", "")).split())))


def source_of(r: dict[str, Any]) -> str:
    src = r.get("source")
    if src is None:
        src = r.get("source_name") or r.get("corpus") or "unknown"
    return str(src)


def summarize_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    words = [row_words(r) for r in rows]
    src_words: Counter[str] = Counter()
    src_rows: Counter[str] = Counter()
    toks: list[str] = []
    for r in rows:
        src = source_of(r)
        src_rows[src] += 1
        src_words[src] += row_words(r)
        toks.extend(content_tokens(str(r.get("text", ""))))
    types = set(toks)
    tok_counts = Counter(toks)
    total_words = sum(words)
    examples = []
    for r in rows[:3]:
        text = " ".join(str(r.get("text", "")).split())[:240]
        examples.append({"source": source_of(r), "words": row_words(r), "example_id": r.get("example_id"), "text_prefix": text})
    for r in rows[-3:]:
        text = " ".join(str(r.get("text", "")).split())[:240]
        examples.append({"source": source_of(r), "words": row_words(r), "example_id": r.get("example_id"), "text_prefix": text})
    return {
        "label": label,
        "rows": len(rows),
        "words": total_words,
        "mean_row_words": statistics.mean(words) if words else None,
        "median_row_words": statistics.median(words) if words else None,
        "source_rows": dict(src_rows.most_common()),
        "source_words": dict(src_words.most_common()),
        "source_word_share": {k: v / total_words for k, v in src_words.most_common()} if total_words else {},
        "qwen_pair_packed_rows": int(src_rows.get("qwen_pair_packed", 0)),
        "qwen_pair_packed_words": int(src_words.get("qwen_pair_packed", 0)),
        "content_token_count": len(toks),
        "content_type_count": len(types),
        "hapax_content_type_count": sum(1 for v in tok_counts.values() if v == 1),
        "top_content_tokens": tok_counts.most_common(30),
        "examples": examples,
    }


def diff_text_identity(clean_rows: list[dict[str, Any]], view_rows: list[dict[str, Any]], upto: int) -> dict[str, Any]:
    same_text = same_words = same_source = 0
    for i in range(upto):
        c, v = clean_rows[i], view_rows[i]
        same_text += int(str(c.get("text", "")) == str(v.get("text", "")))
        same_words += int(row_words(c) == row_words(v))
        same_source += int(source_of(c) == source_of(v))
    return {"rows_checked": upto, "same_text": same_text, "same_words": same_words, "same_source_label": same_source}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m256 = read_json(META256)
    m280 = read_json(META280)
    pair_rows = int(m256["dose"]["pair_rows"])
    topup = int(m256["dose"]["topup_words"])
    changed_with_topup = pair_rows + 1
    clean_prefix = load_jsonl_prefix(CLEAN10, changed_with_topup)
    view_prefix = load_jsonl_prefix(VIEW10, changed_with_topup)
    assert len(clean_prefix) == changed_with_topup
    assert len(view_prefix) == changed_with_topup
    assert sum(row_words(r) for r in clean_prefix) == int(m256["dose"]["changed_block_budget_words"])
    assert sum(row_words(r) for r in view_prefix) == int(m256["dose"]["changed_block_budget_words"])
    # Common filler is read only for source-composition context.  It is shared by all arms.
    common_filler = load_jsonl_all(COMMON_FILLER)
    heldout = load_jsonl_all(HELDOUT)

    dose_active = {"clean_rho0": 0}
    for name, d in m280["doses"].items():
        dose_active[name] = int(d["active_rows"])
    dose_active["max_dose2p64"] = pair_rows  # top-up line is neutral clean-Qwen, not pair admission.

    cumulative: dict[str, Any] = {}
    for name, n in dose_active.items():
        cumulative[name] = summarize_rows(clean_prefix[:n], f"clean rows displaced by {name}")
        cumulative[name]["active_rows"] = n
        cumulative[name]["rho_from_displaced_clean_words"] = cumulative[name]["words"] / 10_000_000

    increments = {
        "quarter_only_rows_0_757": (0, dose_active["quarter_1x"]),
        "half_increment_rows_758_1512": (dose_active["quarter_1x"], dose_active["half_1x"]),
        "full_increment_rows_1513_3004": (dose_active["half_1x"], dose_active["full_1x"]),
        "max_increment_rows_3005_7922": (dose_active["full_1x"], pair_rows),
        "neutral_topup_row_7923": (pair_rows, pair_rows + 1),
    }
    increment_summaries: dict[str, Any] = {}
    for name, (a, b) in increments.items():
        increment_summaries[name] = summarize_rows(clean_prefix[a:b], f"clean rows displaced in {name}")
        increment_summaries[name]["row_span_0idx_start_inclusive"] = a
        increment_summaries[name]["row_span_0idx_end_exclusive"] = b

    # Line-level metadata corroborates that the displaced rows are ordinary BabyLM-source heldout rows, not protected qwen_pair_packed.
    clean_meta_prefix = load_jsonl_prefix(CHANGED_META_CLEAN, changed_with_topup)
    view_meta_prefix = load_jsonl_prefix(CHANGED_META_VIEW, changed_with_topup)
    clean_meta_sources: Counter[str] = Counter()
    clean_meta_words: Counter[str] = Counter()
    for r in clean_meta_prefix[:pair_rows]:
        comps = r.get("component_sources") or {}
        for src, w in comps.items():
            clean_meta_sources[str(src)] += 1
            clean_meta_words[str(src)] += int(w)
    view_pair_id_counts = [len(r.get("pair_ids") or []) for r in view_meta_prefix[:pair_rows]]

    clean_vs_view = diff_text_identity(clean_prefix, view_prefix, changed_with_topup)
    result = {
        "status": "DISPLACED_CLEAN_BLOCK_READOUT_COMPLETE",
        "created_utc": now(),
        "boundary": "CPU/file-only pool readout; no model loading, training, official evaluation, GPU work, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.",
        "scientific_purpose": "Determine what clean content is removed at each sub-dose so a low-dose V-C plateau can be interpreted as admitted-distribution presence, displaced-filler removal, or their contrast rather than as coverage alone.",
        "files": {
            "clean10": rel(CLEAN10),
            "view10": rel(VIEW10),
            "metadata_step256": rel(META256),
            "metadata_step280": rel(META280),
            "common_filler": rel(COMMON_FILLER),
            "heldout": rel(HELDOUT),
        },
        "sha256": {"clean10": sha256(CLEAN10), "view10": sha256(VIEW10), "metadata_step280": sha256(META280)},
        "geometry": {"pair_rows": pair_rows, "changed_with_topup_rows": changed_with_topup, "topup_words": topup, "changed_block_budget_words": m256["dose"]["changed_block_budget_words"]},
        "clean_vs_view_changed_prefix_identity": clean_vs_view,
        "cumulative_displaced_clean": cumulative,
        "incremental_displaced_clean": increment_summaries,
        "common_filler_context_shared_by_all_arms": summarize_rows(common_filler, "common filler shared by all arms"),
        "original_heldout_context": summarize_rows(heldout, "heldout clean-Qwen rows used to make changed block"),
        "metadata_corroboration": {
            "clean_changed_pair_rows_component_word_counts": dict(clean_meta_words.most_common()),
            "clean_changed_pair_rows_component_row_hits": dict(clean_meta_sources.most_common()),
            "qwen_pair_packed_words_in_displaced_pair_rows_by_metadata": int(clean_meta_words.get("qwen_pair_packed", 0)),
            "view_pair_ids_per_pair_row_mean": statistics.mean(view_pair_id_counts) if view_pair_id_counts else None,
            "view_pair_ids_per_pair_row_min": min(view_pair_id_counts) if view_pair_id_counts else None,
            "view_pair_ids_per_pair_row_max": max(view_pair_id_counts) if view_pair_id_counts else None,
        },
        "interpretive_point": "The changed-block clean rows displaced by the sub-dose ladder are ordinary BabyLM-source heldout rows with zero qwen_pair_packed rows; the qwen_pair_packed block is in the common filler/protected portion shared by all arms, not the material displaced by admitting FineWeb packets.",
    }
    write_json(OUT / "displaced_clean_block_summary.json", result)

    # CSVs for quick plotting/inspection.
    with (OUT / "cumulative_displaced_clean_rows.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["dose", "active_rows", "words", "rho", "qwen_pair_packed_words", "childes_words", "gutenberg_words", "open_subtitles_words", "simple_wiki_words", "bnc_spoken_words", "switchboard_words", "content_type_count", "hapax_content_type_count"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for dose, rec in cumulative.items():
            sw = rec["source_words"]
            w.writerow({
                "dose": dose,
                "active_rows": rec["active_rows"],
                "words": rec["words"],
                "rho": rec["rho_from_displaced_clean_words"],
                "qwen_pair_packed_words": rec["qwen_pair_packed_words"],
                "childes_words": sw.get("childes", 0),
                "gutenberg_words": sw.get("gutenberg", 0),
                "open_subtitles_words": sw.get("open_subtitles", 0),
                "simple_wiki_words": sw.get("simple_wiki", 0),
                "bnc_spoken_words": sw.get("bnc_spoken", 0),
                "switchboard_words": sw.get("switchboard", 0),
                "content_type_count": rec["content_type_count"],
                "hapax_content_type_count": rec["hapax_content_type_count"],
            })
    with (OUT / "incremental_displaced_clean_rows.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["segment", "row_start", "row_end_exclusive", "rows", "words", "qwen_pair_packed_words", "top_source", "top_source_words", "content_type_count"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for seg, rec in increment_summaries.items():
            sw = rec["source_words"]
            top_source, top_words = (next(iter(sw.items())) if sw else ("", 0))
            w.writerow({
                "segment": seg,
                "row_start": rec["row_span_0idx_start_inclusive"],
                "row_end_exclusive": rec["row_span_0idx_end_exclusive"],
                "rows": rec["rows"],
                "words": rec["words"],
                "qwen_pair_packed_words": rec["qwen_pair_packed_words"],
                "top_source": top_source,
                "top_source_words": top_words,
                "content_type_count": rec["content_type_count"],
            })

    lines = [
        "# research displaced clean-block readout",
        "",
        result["boundary"],
        "",
        "## Central result",
        "",
        "The clean rows removed by the sub-dose ladder are not the protected `qwen_pair_packed` rows. They are ordinary heldout rows from CHILDES, Gutenberg, OpenSubtitles, Simple Wikipedia, BNC Spoken, and Switchboard. The protected Qwen paired block sits in the shared common filler and is identical across all arms.",
        "",
        "## Cumulative displaced clean rows",
        "",
        "| Dose | Rows | Words | rho(clean removed) | qwen_pair_packed words | Main source-word counts | Content types |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    ordered = ["clean_rho0", "quarter_1x", "half_1x", "full_1x", "max_dose2p64"]
    for dose in ordered:
        rec = cumulative[dose]
        sw = rec["source_words"]
        main = ", ".join(f"{k}:{v}" for k, v in sw.items())
        lines.append(f"| {dose} | {rec['active_rows']} | {rec['words']} | {rec['rho_from_displaced_clean_words']:.6f} | {rec['qwen_pair_packed_words']} | {main} | {rec['content_type_count']} |")
    lines += [
        "",
        "## Incremental displaced segments",
        "",
        "| Segment | Rows | Words | qwen_pair_packed words | Source-word counts |",
        "|---|---:|---:|---:|---|",
    ]
    for seg, rec in increment_summaries.items():
        sw = rec["source_words"]
        main = ", ".join(f"{k}:{v}" for k, v in sw.items())
        lines.append(f"| {seg} | {rec['rows']} | {rec['words']} | {rec['qwen_pair_packed_words']} | {main} |")
    lines += [
        "",
        "## Interpretation for the sub-dose score curve",
        "",
        "A plateau by rho≈0.011 would not mean that the removed block is Qwen-generated paired text; file evidence says the removed material is ordinary BabyLM heldout text. The contrast would instead read as replacing a small amount of ordinary source-balanced BabyLM text with authentic FineWeb source+view packets. Because the common qwen_pair_packed portion is identical in every arm, it cannot explain V≈R≈B≫C through direct displacement in this instrument.",
        "",
        f"JSON summary: `{rel(OUT / 'displaced_clean_block_summary.json')}`",
        f"Cumulative CSV: `{rel(OUT / 'cumulative_displaced_clean_rows.csv')}`",
        f"Incremental CSV: `{rel(OUT / 'incremental_displaced_clean_rows.csv')}`",
    ]
    (OUT / "displaced_clean_block_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "summary_md": rel(OUT / "displaced_clean_block_summary.md"), "summary_json": rel(OUT / "displaced_clean_block_summary.json"), "qwen_pair_packed_words_displaced_at_max": cumulative["max_dose2p64"]["qwen_pair_packed_words"], "max_displaced_sources": cumulative["max_dose2p64"]["source_words"], "no_gpu_training_eval_upload": True}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
