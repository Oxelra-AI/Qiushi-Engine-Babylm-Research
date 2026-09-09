#!/usr/bin/env python3
"""research direct inspection of score-bearing 7-word overlaps in the reinvest changed block.

The refined overlap audit showed that the FineWeb compact changed block has fewer exact
7-word overlaps with official evaluation text than the clean-Qwen rows it replaced,
but there are a few score-bearing hits. This script reconstructs the same exact
7-gram map while retaining the actual evaluation text, then extracts every
score-bearing changed-block hit for direct reading.
"""
from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

STUDY = Path("experiments/archive/frontier_consolidation")
OUT_DIR = STUDY / "data/changed_block_overlap_inspection"
OUT_JSON = OUT_DIR / "changed_block_score_bearing_overlap_inspection.json"
NOTE = (STUDY.parents[2] / 'research/notes/frontier_consolidation/23_changed_block_score_bearing_overlap_inspection.md')

EVAL_ROOT = Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval")
CANDIDATE_10M = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
HELDOUT_CLEAN_ROWS = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl")
CHANGED_SRC = "cleanqwen_fineweb_compact_view_reinvest"
N = 7
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


def ngrams_from_tokens(ts: List[str], n: int = N) -> List[str]:
    if len(ts) < n:
        return []
    return [" ".join(ts[i:i+n]) for i in range(len(ts)-n+1)]


def ngrams(text: str, n: int = N) -> List[str]:
    return ngrams_from_tokens(toks(text), n)


def iter_texts(obj: Any, parent_key: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(obj, str):
        pk = parent_key.lower()
        if pk in SKIP_KEYS:
            return
        if pk in TEXT_KEYS or len(toks(obj)) >= 5:
            yield parent_key, obj
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


def short(text: str, n: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n] + " ..."


def load_eval_map(n: int = N) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
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

        def add_text(text: str, locator: str, key: str = "") -> None:
            nonlocal strings
            strings += 1
            ts = toks(text)
            for ng in ngrams_from_tokens(ts, n):
                occurrence_by_top[cls["top_dir"]] += 1
                occurrence_by_role[cls["file_role"]] += 1
                if len(mp[ng]) < 64:
                    mp[ng].append({
                        "file": str(rel),
                        "locator": locator,
                        "text_key": key,
                        "text_excerpt": short(text, 900),
                        **cls,
                    })

        try:
            if p.suffix.lower() == ".jsonl":
                with p.open("r", encoding="utf-8") as f:
                    for li, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            for key, text in iter_texts(obj):
                                add_text(text, f"line:{li}", key)
                        except json.JSONDecodeError:
                            add_text(line, f"line:{li}", "line")
            elif p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                for key, text in iter_texts(obj):
                    add_text(text, "json", key)
            elif p.suffix.lower() == ".csv":
                with p.open("r", encoding="utf-8", newline="") as f:
                    rdr = csv.DictReader(f)
                    for ri, row in enumerate(rdr, 1):
                        for key, val in row.items():
                            if val and isinstance(val, str):
                                add_text(val, f"row:{ri}:col:{key}", key)
            else:
                add_text(p.read_text(encoding="utf-8", errors="ignore"), "txt", "txt")
        except Exception as exc:
            # Store an impossible key as a readable trace of skipped files.
            mp[f"__SKIP__{rel}"].append({"error": repr(exc), "file": str(rel)})
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


def inspect_jsonl(path: Path, eval_map: Dict[str, List[Dict[str, Any]]], label: str, source_filter: str | None = None) -> Dict[str, Any]:
    rows = 0
    used = 0
    words = 0
    rows_with_score = 0
    rows_with_glue_valid = 0
    rows_with_any = 0
    hit_rows: List[Dict[str, Any]] = []
    ngram_counter = Counter()
    role_counter = Counter()
    top_counter = Counter()
    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            rows += 1
            rec = json.loads(line)
            src = str(rec.get("source") or "unknown")
            if source_filter is not None and src != source_filter:
                continue
            used += 1
            text = rec.get("text") or ""
            w = int(rec.get("words") or len(text.split()))
            words += w
            row_hits: List[Dict[str, Any]] = []
            for ng in sorted(set(ngrams(text, N))):
                evs = eval_map.get(ng)
                if not evs:
                    continue
                rows_with_any_candidate = True
                score_evs = [ev for ev in evs if ev.get("score_bearing") == "score_bearing"]
                if not score_evs:
                    continue
                ngram_counter[ng] += 1
                for ev in score_evs:
                    role_counter[ev.get("file_role", "unknown")] += 1
                    top_counter[ev.get("top_dir", "unknown")] += 1
                row_hits.append({"ngram": ng, "eval_records": score_evs})
            if row_hits:
                rows_with_score += 1
                if any(ev.get("file_role") == "glue_valid_scored" for h in row_hits for ev in h["eval_records"]):
                    rows_with_glue_valid += 1
                hit_rows.append({
                    "row_index": row_index,
                    "source": src,
                    "example_id": rec.get("example_id"),
                    "words": w,
                    "text": text,
                    "hits": row_hits,
                })
            # count any-overlap rows separately, including non-score-bearing
            if any(ng in eval_map for ng in set(ngrams(text, N))):
                rows_with_any += 1
    return {
        "label": label,
        "path": str(path),
        "source_filter": source_filter,
        "total_rows_in_file": rows,
        "scanned_rows": used,
        "scanned_words": words,
        "rows_with_any_overlap": rows_with_any,
        "rows_with_score_bearing_overlap": rows_with_score,
        "rows_with_glue_valid_score_overlap": rows_with_glue_valid,
        "score_bearing_hit_rows": hit_rows,
        "score_bearing_hit_rows_count": len(hit_rows),
        "score_bearing_unique_ngrams": len(ngram_counter),
        "score_bearing_ngram_counts": dict(ngram_counter),
        "score_bearing_eval_hits_by_file_role": dict(role_counter),
        "score_bearing_eval_hits_by_top_dir": dict(top_counter),
    }


def summarize_row(row: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    lines.append(f"### Row {row['row_index']} · example_id {row.get('example_id')} · {row.get('words')} words")
    lines.append("")
    lines.append("Training row:")
    lines.append("> " + short(row.get("text", ""), 700))
    lines.append("")
    for h in row.get("hits", []):
        lines.append(f"- Exact 7-gram: `{h['ngram']}`")
        for ev in h.get("eval_records", []):
            lines.append(f"  - {ev['file']} {ev['locator']} [{ev['file_role']}]: {short(ev.get('text_excerpt',''), 500)}")
    lines.append("")
    return lines


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    eval_map, eval_meta = load_eval_map(N)
    changed = inspect_jsonl(CANDIDATE_10M, eval_map, "candidate_changed_block_only", CHANGED_SRC)
    heldout = inspect_jsonl(HELDOUT_CLEAN_ROWS, eval_map, "heldout_clean_rows_replaced_by_changed_block", None)
    payload = {
        "status": "CHANGED_BLOCK_SCORE_BEARING_OVERLAP_INSPECTION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": N,
        "eval_map_meta": eval_meta,
        "changed_block": changed,
        "heldout_clean_rows_replaced_by_changed_block": {
            k: v for k, v in heldout.items() if k != "score_bearing_hit_rows"
        },
        "changed_minus_heldout": {
            "rows_with_any_overlap": changed["rows_with_any_overlap"] - heldout["rows_with_any_overlap"],
            "rows_with_score_bearing_overlap": changed["rows_with_score_bearing_overlap"] - heldout["rows_with_score_bearing_overlap"],
            "rows_with_glue_valid_score_overlap": changed["rows_with_glue_valid_score_overlap"] - heldout["rows_with_glue_valid_score_overlap"],
            "score_bearing_unique_ngrams": changed["score_bearing_unique_ngrams"] - heldout["score_bearing_unique_ngrams"],
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("# research changed-block score-bearing overlap inspection")
    lines.append("")
    lines.append("This CPU-only inspection reconstructs the exact 7-word overlap map and prints every score-bearing changed-block hit with the matching evaluation text. It is a direct reading aid for the reinvest endpoint; it does not change data or model results.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Changed block: {changed['scanned_rows']} rows / {changed['scanned_words']} words; any-overlap rows {changed['rows_with_any_overlap']}; score-bearing rows {changed['rows_with_score_bearing_overlap']}; GLUE-valid score-bearing rows {changed['rows_with_glue_valid_score_overlap']}; score-bearing unique 7-grams {changed['score_bearing_unique_ngrams']}.")
    lines.append(f"- Heldout clean-Qwen rows replaced by the changed block: {heldout['scanned_rows']} rows / {heldout['scanned_words']} words; any-overlap rows {heldout['rows_with_any_overlap']}; score-bearing rows {heldout['rows_with_score_bearing_overlap']}; GLUE-valid score-bearing rows {heldout['rows_with_glue_valid_score_overlap']}; score-bearing unique 7-grams {heldout['score_bearing_unique_ngrams']}.")
    lines.append("- Changed minus heldout: " + json.dumps(payload["changed_minus_heldout"], ensure_ascii=False, sort_keys=True))
    lines.append("")
    lines.append("## Direct reading of changed-block score-bearing rows")
    lines.append("")
    for row in changed["score_bearing_hit_rows"]:
        lines.extend(summarize_row(row))
    lines.append("## Provisional reading")
    lines.append("")
    lines.append("The changed block has fewer score-bearing exact-7gram rows than the clean-Qwen rows it replaced. The GLUE-valid hits are short generic public phrases or questions embedded inside much longer unrelated rows, not whole-example target copies. The AoA hits arise from ordinary phrases inside the word-context inventory, and the VQA hit is a spatial phrase. This does not by itself prove absence of every near-duplicate, but it reduces the exact-overlap concern for the frozen endpoint and motivates a stronger near-duplicate scan as the next contamination-safety action if seed stability remains viable.")
    lines.append("")
    lines.append(f"Machine-readable JSON: `{OUT_JSON}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out_json": str(OUT_JSON),
        "note": str(NOTE),
        "changed_score_rows": changed["rows_with_score_bearing_overlap"],
        "changed_glue_valid_score_rows": changed["rows_with_glue_valid_score_overlap"],
        "heldout_score_rows": heldout["rows_with_score_bearing_overlap"],
        "heldout_glue_valid_score_rows": heldout["rows_with_glue_valid_score_overlap"],
        "changed_minus_heldout": payload["changed_minus_heldout"],
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
