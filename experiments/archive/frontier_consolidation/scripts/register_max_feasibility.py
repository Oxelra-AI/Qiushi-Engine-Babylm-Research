#!/usr/bin/env python3
"""research: feasibility for MAX-dose register-displacement pair.

File-only analysis. The scientific question is whether a high-power MAX-dose
register-removal contrast can be built from existing streams: admit the identical
MAX FineWeb compact-view block while selecting same-word-count clean replacement
rows from contrasting clean registers. No generation, streaming, model loading,
training, evaluation, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import collections, json, pathlib, statistics, time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
POOL = WS / "data/dose_2p64x_rowholdout_pools"
OUT = WS / "data/register_max_feasibility"
CLEAN10 = POOL / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl"
VIEW10 = POOL / "compact_view_dose2p64x_10M.jsonl"
CLEAN_META = POOL / "cleanqwen_lengthmatched_dose2p64x_changed_block_rows_meta.jsonl"
VIEW_META = POOL / "compact_view_dose2p64x_changed_block_rows_meta.jsonl"
PAIR_ROWS = 7923
META_ROWS = 7924
TOTAL_ROWS = 65313
DEV_SPEECH = {"childes", "open_subtitles", "bnc_spoken", "switchboard"}
ADULT_PROSE = {"gutenberg", "simple_wiki"}
PROTECTED = {"qwen_pair_packed"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def iter_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def nwords(row: dict[str, Any]) -> int:
    return int(row.get("words", len(str(row.get("text", "")).split())))


def clean_components(rows: list[dict[str, Any]], meta_rows: list[dict[str, Any]]) -> list[dict[str, int]]:
    comps: list[dict[str, int]] = []
    meta_by_idx = {int(r["row_index"]): r for r in meta_rows}
    for i, r in enumerate(rows):
        if i in meta_by_idx:
            c = {str(k): int(v) for k, v in (meta_by_idx[i].get("component_sources") or {}).items()}
        else:
            src = str(r.get("source", r.get("source_name", "unknown"))).split("::")[-1]
            c = {src: nwords(r)}
        comps.append(c)
    return comps


def score(comps: dict[str, int], group: set[str], opposed: set[str], total: int) -> float:
    g = sum(comps.get(k, 0) for k in group)
    o = sum(comps.get(k, 0) for k in opposed)
    p = sum(comps.get(k, 0) for k in PROTECTED)
    # Protected qwen rows must stay common; make them impossible unless forced.
    if p:
        return -10.0 - p / max(1, total)
    return (g - o) / max(1, total)


def summarize_selection(indices: list[int], rows: list[dict[str, Any]], comps_all: list[dict[str, int]]) -> dict[str, Any]:
    wc = [nwords(rows[i]) for i in indices]
    idxs = list(indices)
    comps = collections.Counter()
    for i in indices:
        comps.update(comps_all[i])
    total = sum(wc)
    dev = sum(comps.get(k, 0) for k in DEV_SPEECH)
    adult = sum(comps.get(k, 0) for k in ADULT_PROSE)
    qwen = sum(comps.get(k, 0) for k in PROTECTED)
    return {
        "rows": len(indices),
        "words": total,
        "dev_speech_words": dev,
        "adult_prose_words": adult,
        "qwen_pair_packed_words": qwen,
        "other_words": total - dev - adult - qwen,
        "dev_speech_fraction": dev / total if total else None,
        "adult_prose_fraction": adult / total if total else None,
        "qwen_pair_packed_fraction": qwen / total if total else None,
        "row_index_min": min(idxs) if idxs else None,
        "row_index_max": max(idxs) if idxs else None,
        "row_index_mean": statistics.mean(idxs) if idxs else None,
        "row_index_sd": statistics.pstdev(idxs) if len(idxs) > 1 else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = load_jsonl(CLEAN10)
    view = load_jsonl(VIEW10)
    clean_meta = load_jsonl(CLEAN_META)
    view_meta = load_jsonl(VIEW_META)
    assert len(clean) == len(view) == TOTAL_ROWS
    assert len(clean_meta) == len(view_meta) == META_ROWS
    for i in range(TOTAL_ROWS):
        if nwords(clean[i]) != nwords(view[i]):
            raise AssertionError(("word geometry mismatch", i, nwords(clean[i]), nwords(view[i])))
    comps = clean_components(clean, clean_meta)
    target_hist = collections.Counter(nwords(view[i]) for i in range(PAIR_ROWS))
    prefix_comps = collections.Counter()
    for i in range(PAIR_ROWS):
        prefix_comps.update(comps[i])
    prefix_words = sum(nwords(clean[i]) for i in range(PAIR_ROWS))
    prefix_dev = sum(prefix_comps.get(k, 0) for k in DEV_SPEECH)
    prefix_adult = sum(prefix_comps.get(k, 0) for k in ADULT_PROSE)
    prefix_qwen = sum(prefix_comps.get(k, 0) for k in PROTECTED)

    by_len: dict[int, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(clean):
        if sum(comps[i].get(k, 0) for k in PROTECTED):
            continue
        by_len[nwords(r)].append(i)

    length_rows = []
    child_choice: list[int] = []
    adult_choice: list[int] = []
    for w, need in sorted(target_hist.items()):
        idxs = by_len.get(w, [])
        child_sorted = sorted(idxs, key=lambda i: (score(comps[i], DEV_SPEECH, ADULT_PROSE, w), -abs(i - (PAIR_ROWS/2))), reverse=True)
        adult_sorted = sorted(idxs, key=lambda i: (score(comps[i], ADULT_PROSE, DEV_SPEECH, w), -abs(i - (PAIR_ROWS/2))), reverse=True)
        child_take = child_sorted[:need]
        adult_take = adult_sorted[:need]
        child_choice.extend(child_take)
        adult_choice.extend(adult_take)
        length_rows.append({
            "words": w,
            "need": need,
            "available_nonqwen": len(idxs),
            "child_take_min_score": min((score(comps[i], DEV_SPEECH, ADULT_PROSE, w) for i in child_take), default=None),
            "adult_take_min_score": min((score(comps[i], ADULT_PROSE, DEV_SPEECH, w) for i in adult_take), default=None),
            "child_available_score_ge_0p5": sum(1 for i in idxs if score(comps[i], DEV_SPEECH, ADULT_PROSE, w) >= 0.5),
            "adult_available_score_ge_0p5": sum(1 for i in idxs if score(comps[i], ADULT_PROSE, DEV_SPEECH, w) >= 0.5),
            "feasible_exact_count": len(child_take) == need and len(adult_take) == need,
        })
    child_counter = collections.Counter(nwords(clean[i]) for i in child_choice)
    adult_counter = collections.Counter(nwords(clean[i]) for i in adult_choice)
    target_ok = child_counter == target_hist and adult_counter == target_hist

    # Crude position-matched distance if choices are sorted within each length.
    dists = []
    overlaps = 0
    for w in sorted(target_hist):
        c = sorted([i for i in child_choice if nwords(clean[i]) == w])
        a = sorted([i for i in adult_choice if nwords(clean[i]) == w])
        for ci, ai in zip(c, a):
            dists.append(abs(ci - ai))
            if ci == ai:
                overlaps += 1
    summary = {
        "status": "REGISTER_MAX_FEASIBILITY_COMPLETE",
        "created_utc": now(),
        "purpose": "Check whether MAX-dose same-FineWeb register-skewed clean displacement is feasible from existing rows without touching qwen_pair_packed.",
        "target": {"fineweb_rows": PAIR_ROWS, "fineweb_words": sum(nwords(view[i]) for i in range(PAIR_ROWS)), "rho": sum(nwords(view[i]) for i in range(PAIR_ROWS)) / 10_000_000, "length_types": len(target_hist)},
        "original_max_displaced_prefix": {"rows": PAIR_ROWS, "words": prefix_words, "dev_speech_words": prefix_dev, "adult_prose_words": prefix_adult, "qwen_pair_packed_words": prefix_qwen, "other_words": prefix_words - prefix_dev - prefix_adult - prefix_qwen, "dev_speech_fraction": prefix_dev/prefix_words, "adult_prose_fraction": prefix_adult/prefix_words},
        "exact_word_hist_feasible_nonqwen": target_ok,
        "length_rows": length_rows,
        "naive_position_match": {"mean_abs_row_distance": statistics.mean(dists) if dists else None, "median_abs_row_distance": statistics.median(dists) if dists else None, "p90_abs_row_distance": sorted(dists)[int(0.9*(len(dists)-1))] if dists else None, "max_abs_row_distance": max(dists) if dists else None, "same_position_overlap": overlaps},
        "childspeech_selection_summary": summarize_selection(sorted(child_choice), clean, comps),
        "adultprose_selection_summary": summarize_selection(sorted(adult_choice), clean, comps),
        "no_generation_streaming_model_loading_training_eval_upload_or_leaderboard": True,
    }
    (OUT / "register_max_feasibility_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "length_feasibility_rows.csv").open("w", encoding="utf-8") as f:
        cols = list(length_rows[0].keys()) if length_rows else []
        f.write(",".join(cols) + "\n")
        for r in length_rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    md = [
        "# research MAX register-pair feasibility", "",
        "File-only. No generation, streaming, model loading, training, evaluation, upload, or leaderboard action.", "",
        "## Target MAX FineWeb block",
        f"Rows: {summary['target']['fineweb_rows']}; words: {summary['target']['fineweb_words']}; rho: {summary['target']['rho']:.6f}; word-length types: {summary['target']['length_types']}.", "",
        "## Existing MAX displaced prefix mixture",
        f"Developmental/speech words (CHILDES + OpenSubtitles + BNC Spoken + Switchboard): {prefix_dev} ({prefix_dev/prefix_words:.6f}).",
        f"Adult-prose words (Gutenberg + SimpleWiki): {prefix_adult} ({prefix_adult/prefix_words:.6f}).",
        f"Other/protected qwen words in prefix: {prefix_words - prefix_dev - prefix_adult - prefix_qwen}; qwen: {prefix_qwen}.", "",
        "## Feasibility",
        f"Exact non-qwen word-histogram feasible for both skewed arms: {target_ok}.",
        "Naive sorted-within-length position pairing:", "```json", json.dumps(summary["naive_position_match"], indent=2), "```", "",
        "## Skewed selection summaries", "```json", json.dumps({"childspeech": summary["childspeech_selection_summary"], "adultprose": summary["adultprose_selection_summary"]}, indent=2, ensure_ascii=False), "```", "",
        "Length table: `experiments/archive/frontier_consolidation/data/register_max_feasibility/length_feasibility_rows.csv`",
        "JSON: `experiments/archive/frontier_consolidation/data/register_max_feasibility/register_max_feasibility_summary.json`",
    ]
    (OUT / "register_max_feasibility_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "target_ok": target_ok, "target_words": summary["target"]["fineweb_words"], "original_prefix_dev_speech_words": prefix_dev, "original_prefix_adult_words": prefix_adult, "childspeech_frac": summary["childspeech_selection_summary"]["dev_speech_fraction"], "adultprose_frac": summary["adultprose_selection_summary"]["adult_prose_fraction"], "summary_md": rel(OUT / "register_max_feasibility_summary.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
