#!/usr/bin/env python3
"""Refined exact-overlap audit for compact_view_reinvest.

Separates overlaps with score-bearing evaluation text from SuperGLUE/GLUE train
splits, and compares the FineWeb changed block against the clean-Qwen heldout
official rows it replaced.
"""
from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

STUDY = Path("experiments/archive/representation_and_objectives")
OUT_DIR = STUDY / "data/reinvest_overlap_refined"
OUT_JSON = OUT_DIR / "compact_reinvest_refined_eval_overlap_audit.json"
NOTE = (STUDY.parents[2] / 'research/notes/representation_and_objectives/compact_reinvest_refined_eval_overlap_audit.md')

EVAL_ROOT = Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval")
CANDIDATE_10M = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
CLEAN_QWEN_10M = Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
HELDOUT_CLEAN_ROWS = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl")
COMMON_FILLER_ROWS = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/common_filler_rows.jsonl")
QWEN_PAIR_ROWS_META = Path("experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl")

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
TEXT_KEYS = {
    "sentence", "sentence_good", "sentence_bad", "good_sentence", "bad_sentence",
    "context", "question", "passage", "paragraph", "premise", "hypothesis",
    "text", "text_a", "text_b", "input", "target", "option1", "option2",
    "answer", "correct", "incorrect", "choice1", "choice2", "query", "title",
    "article", "summary", "word", "stem", "ending0", "ending1", "ending2", "ending3",
}
SKIP_KEYS = {"id", "idx", "uid", "guid", "label", "labels", "metadata", "meta", "path", "file", "filename", "source", "source_file"}


def toks(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def ngrams(text: str, n: int = 7) -> List[str]:
    ts = toks(text)
    if len(ts) < n:
        return []
    return [" ".join(ts[i:i+n]) for i in range(len(ts)-n+1)]


def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_texts(obj: Any, parent_key: str = "") -> Iterable[str]:
    if isinstance(obj, str):
        if parent_key.lower() in SKIP_KEYS:
            return
        # Keep strings either under text-like keys or long enough to carry language.
        if parent_key.lower() in TEXT_KEYS or len(toks(obj)) >= 5:
            yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_texts(x, parent_key)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in SKIP_KEYS:
                continue
            yield from iter_texts(v, str(k))


def classify_eval_file(rel: Path) -> Dict[str, str]:
    top = rel.parts[0] if rel.parts else rel.name
    name = rel.name
    if top == "glue_filtered":
        if ".train." in name:
            role = "glue_train_for_finetuning"
            score_bearing = "no_direct_final_score"
        elif ".valid." in name:
            role = "glue_valid_scored"
            score_bearing = "score_bearing"
        else:
            role = "glue_unknown"
            score_bearing = "unknown"
    else:
        role = "zero_shot_or_human_scored"
        score_bearing = "score_bearing"
    return {"top_dir": top, "file_role": role, "score_bearing": score_bearing}


def build_eval_map(n: int = 7) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    mp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    files = 0
    strings = 0
    occurrence_by_top = Counter()
    occurrence_by_role = Counter()
    for p in sorted(EVAL_ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".json", ".jsonl", ".csv", ".txt"}:
            continue
        rel = p.relative_to(EVAL_ROOT)
        cls = classify_eval_file(rel)
        files += 1
        def add_text(text: str, locator: str) -> None:
            nonlocal strings
            strings += 1
            for ng in ngrams(text, n):
                occurrence_by_top[cls["top_dir"]] += 1
                occurrence_by_role[cls["file_role"]] += 1
                if len(mp[ng]) < 16:
                    rec = {"file": str(rel), "locator": locator, **cls}
                    mp[ng].append(rec)
        try:
            if p.suffix.lower() == ".jsonl":
                with p.open("r", encoding="utf-8") as f:
                    for li, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            for text in iter_texts(obj):
                                add_text(text, f"line:{li}")
                        except json.JSONDecodeError:
                            add_text(line, f"line:{li}")
            elif p.suffix.lower() == ".json":
                obj = load_json(p)
                for text in iter_texts(obj):
                    add_text(text, "json")
            elif p.suffix.lower() == ".csv":
                with p.open("r", encoding="utf-8", newline="") as f:
                    rdr = csv.DictReader(f)
                    for ri, row in enumerate(rdr, 1):
                        for key, val in row.items():
                            if val and isinstance(val, str):
                                add_text(val, f"row:{ri}:col:{key}")
            else:
                add_text(p.read_text(encoding="utf-8", errors="ignore"), "txt")
        except Exception as e:
            mp[f"__SKIP__{rel}"] = [{"error": repr(e)}]
    mp = {k: v for k, v in mp.items() if not k.startswith("__SKIP__")}
    meta = {
        "eval_root": str(EVAL_ROOT),
        "n": n,
        "files": files,
        "strings": strings,
        "unique_ngrams": len(mp),
        "occurrence_by_top_dir": dict(occurrence_by_top),
        "occurrence_by_file_role": dict(occurrence_by_role),
    }
    return mp, meta


def scan_jsonl(path: Path, eval_map: Dict[str, List[Dict[str, Any]]], label: str, source_filter: str | None = None, n: int = 7, sample_limit: int = 80) -> Dict[str, Any]:
    rows = 0
    used_rows = 0
    words = 0
    rows_with_any = 0
    rows_with_score = 0
    rows_with_glue_valid = 0
    rows_with_glue_train = 0
    match_unique = set()
    match_occ = 0
    by_top = Counter()
    by_role = Counter()
    by_score_bearing = Counter()
    by_source_rows = Counter()
    by_source_match_rows = Counter()
    samples_score = []
    samples_all = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rows += 1
            rec = json.loads(line)
            src = str(rec.get("source") or "unknown")
            if source_filter is not None and src != source_filter:
                continue
            used_rows += 1
            text = rec.get("text") or ""
            w = int(rec.get("words") or len(text.split()))
            words += w
            by_source_rows[src] += 1
            row_matches: Dict[str, List[Dict[str, Any]]] = {}
            for ng in ngrams(text, n):
                if ng in eval_map:
                    row_matches.setdefault(ng, eval_map[ng])
                    match_occ += 1
            if row_matches:
                rows_with_any += 1
                by_source_match_rows[src] += 1
                has_score = False
                has_glue_valid = False
                has_glue_train = False
                for ng, evs in row_matches.items():
                    match_unique.add(ng)
                    for ev in evs:
                        by_top[ev["top_dir"]] += 1
                        by_role[ev["file_role"]] += 1
                        by_score_bearing[ev["score_bearing"]] += 1
                        if ev["score_bearing"] == "score_bearing":
                            has_score = True
                        if ev["file_role"] == "glue_valid_scored":
                            has_glue_valid = True
                        if ev["file_role"] == "glue_train_for_finetuning":
                            has_glue_train = True
                if has_score:
                    rows_with_score += 1
                if has_glue_valid:
                    rows_with_glue_valid += 1
                if has_glue_train:
                    rows_with_glue_train += 1
                # Save representative samples, prioritizing score-bearing overlaps.
                for ng, evs in list(row_matches.items())[:5]:
                    sample = {
                        "row_index": idx,
                        "source": src,
                        "example_id": rec.get("example_id"),
                        "words": w,
                        "ngram": ng,
                        "eval_records": evs[:8],
                        "score_bearing": any(ev["score_bearing"] == "score_bearing" for ev in evs),
                        "text_prefix": text[:360],
                    }
                    if len(samples_all) < sample_limit:
                        samples_all.append(sample)
                    if sample["score_bearing"] and len(samples_score) < sample_limit:
                        samples_score.append(sample)
    return {
        "label": label,
        "path": str(path),
        "source_filter": source_filter,
        "total_rows_in_file": rows,
        "scanned_rows": used_rows,
        "scanned_words": words,
        "rows_with_any_overlap": rows_with_any,
        "rows_with_score_bearing_overlap": rows_with_score,
        "rows_with_glue_valid_overlap": rows_with_glue_valid,
        "rows_with_glue_train_overlap": rows_with_glue_train,
        "unique_matching_ngrams": len(match_unique),
        "matching_ngram_occurrences": match_occ,
        "eval_record_hits_by_top_dir": dict(by_top),
        "eval_record_hits_by_file_role": dict(by_role),
        "eval_record_hits_by_score_bearing": dict(by_score_bearing),
        "rows_by_train_source": dict(by_source_rows),
        "rows_with_overlap_by_train_source": dict(by_source_match_rows),
        "sample_score_bearing_overlaps": samples_score,
        "sample_all_overlaps": samples_all,
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    eval_map, eval_meta = build_eval_map(7)

    changed_src = "cleanqwen_fineweb_compact_view_reinvest"
    scans = {
        "candidate_all": scan_jsonl(CANDIDATE_10M, eval_map, "candidate_all", None),
        "candidate_changed_block_only": scan_jsonl(CANDIDATE_10M, eval_map, "candidate_changed_block_only", changed_src),
        "clean_qwen_all": scan_jsonl(CLEAN_QWEN_10M, eval_map, "clean_qwen_all", None),
        "heldout_clean_rows_replaced_by_changed_block": scan_jsonl(HELDOUT_CLEAN_ROWS, eval_map, "heldout_clean_rows_replaced_by_changed_block", None),
        "common_filler_rows": scan_jsonl(COMMON_FILLER_ROWS, eval_map, "common_filler_rows", None),
    }

    cb = scans["candidate_changed_block_only"]
    held = scans["heldout_clean_rows_replaced_by_changed_block"]
    changed_minus_heldout = {
        key: cb[key] - held[key]
        for key in [
            "scanned_rows", "scanned_words", "rows_with_any_overlap", "rows_with_score_bearing_overlap",
            "rows_with_glue_valid_overlap", "rows_with_glue_train_overlap", "unique_matching_ngrams",
            "matching_ngram_occurrences",
        ]
        if isinstance(cb.get(key), int) and isinstance(held.get(key), int)
    }

    payload = {
        "status": "COMPACT_REINVEST_REFINED_EVAL_OVERLAP_AUDIT",
        "created_utc": "2026-08-29T18:45:00Z",
        "purpose": "Check whether the SOTA-facing compact_view_reinvest endpoint has direct phrase overlap with local official evaluation texts, separating score-bearing splits from fine-tuning train splits and comparing the FineWeb changed block to the official rows it replaced.",
        "n": 7,
        "eval_map_meta": eval_meta,
        "scans": scans,
        "changed_block_minus_heldout_clean_rows": changed_minus_heldout,
        "interpretation": {
            "score_bearing_definition": "all non-glue_filtered evaluation files plus glue_filtered *.valid.jsonl; glue_filtered *.train.jsonl is the SuperGLUE fine-tuning train split and not directly scored, but is still tracked separately",
            "main_use": "If candidate_changed_block_only has high or targeted score-bearing overlap, the endpoint needs row removal/retraining before submission; if overlaps are sparse/common and comparable to heldout official rows, they are a provenance warning rather than an explanation for the result.",
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research refined eval-overlap audit for compact_view_reinvest")
    lines.append("")
    lines.append("This CPU-only audit separates score-bearing evaluation text from SuperGLUE fine-tuning train splits and focuses on the FineWeb compact changed block, not only the full 10M corpus.")
    lines.append("")
    lines.append("## Main counts")
    lines.append("")
    for key in ["candidate_changed_block_only", "heldout_clean_rows_replaced_by_changed_block", "candidate_all", "clean_qwen_all"]:
        s = scans[key]
        lines.append(f"- {key}: rows {s['scanned_rows']}, words {s['scanned_words']}, any-overlap rows {s['rows_with_any_overlap']}, score-bearing rows {s['rows_with_score_bearing_overlap']}, GLUE-valid rows {s['rows_with_glue_valid_overlap']}, GLUE-train rows {s['rows_with_glue_train_overlap']}, unique 7-grams {s['unique_matching_ngrams']}, occurrences {s['matching_ngram_occurrences']}.")
    lines.append("")
    lines.append("Changed block minus heldout official rows: " + json.dumps(changed_minus_heldout, ensure_ascii=False))
    lines.append("")
    lines.append("## Changed-block hit distribution")
    lines.append("")
    lines.append("- By top dir: " + json.dumps(cb["eval_record_hits_by_top_dir"], sort_keys=True))
    lines.append("- By file role: " + json.dumps(cb["eval_record_hits_by_file_role"], sort_keys=True))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The JSON stores representative score-bearing samples. This audit is a contamination safeguard, not proof of contamination by itself: exact common phrases and public benchmark fragments can overlap in both official and external web-derived text. The actionable comparison is the changed block against the heldout clean-Qwen official rows it replaced.")
    lines.append("")
    lines.append(f"Machine-readable JSON: `{OUT_JSON}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out_json": str(OUT_JSON),
        "out_note": str(NOTE),
        "changed_block_score_rows": cb["rows_with_score_bearing_overlap"],
        "heldout_score_rows": held["rows_with_score_bearing_overlap"],
        "changed_block_glue_valid_rows": cb["rows_with_glue_valid_overlap"],
        "heldout_glue_valid_rows": held["rows_with_glue_valid_overlap"],
        "changed_block_unique_7grams": cb["unique_matching_ngrams"],
        "heldout_unique_7grams": held["unique_matching_ngrams"],
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
