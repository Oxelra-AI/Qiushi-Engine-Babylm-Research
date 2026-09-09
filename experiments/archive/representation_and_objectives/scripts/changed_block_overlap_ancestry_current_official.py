#!/usr/bin/env python3
"""research: current-official source ancestry for changed-block exact 7-token overlaps.

Uses the current pristine BabyLM Strict evaluation coordinate built in research plus the
research regenerated official GlobalPIQA files. Focuses on the FineWeb compact changed
block of compact_view_reinvest and records the exact source/rewrite pairs for any rows
sharing an exact 7-token span with Strict-Small score text.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import re
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/representation_and_objectives')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/changed_block_overlap_ancestry_current_official')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/changed_block_overlap_ancestry_current_official/changed_block_overlap_ancestry_current_official.json')
OUT_JSONL = _public_path('experiments/archive/representation_and_objectives/data/changed_block_overlap_ancestry_current_official/changed_block_overlap_ancestry_current_official_rows.jsonl')
NOTE = _public_path('research/notes/representation_and_objectives/changed_block_overlap_ancestry_current_official.md')

PRISTINE_EVAL_ROOT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')
GLOBALPIQA_EVAL_ROOT = _public_path('experiments/archive/representation_and_objectives/data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval')
CANDIDATE_10M = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
CHANGED_META = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl')
PAIRS = _public_path('experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl')
PROMPTS = _public_path('experiments/archive/frontier_consolidation/data/medium_density_prompts/fineweb_medium_compact_prompts_all.jsonl')
ACCEPTED = _public_path('experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_accepted_rewrites.jsonl')
RAW_OUTPUTS = _public_path('experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_rows.jsonl')

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
TEXT_KEYS = {
    "sentence", "sentence_good", "sentence_bad", "good_sentence", "bad_sentence",
    "context", "question", "passage", "paragraph", "premise", "hypothesis",
    "text", "text_a", "text_b", "input", "target", "option1", "option2",
    "answer", "correct", "incorrect", "choice1", "choice2", "query", "title",
    "article", "summary", "word", "stem", "ending0", "ending1", "ending2", "ending3",
    "prompt", "solution0", "solution1", "solution2", "solution3",
}
SKIP_KEYS = {"id", "idx", "uid", "guid", "label", "labels", "metadata", "meta", "path", "file", "filename", "source", "source_file", "example_id"}
STRICT_SCORE_TOP_DIRS = {
    "aoa", "blimp_filtered", "supplement_filtered", "ewok_filtered", "entity_tracking",
    "comps", "reading", "global_piqa_parallel", "global_piqa_nonparallel",
}
STRICT_NON_SCORE_TOP_DIRS = {"vqa_filtered", "winoground_filtered"}


def toks(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def ngrams_from_tokens(ts: list[str], n: int = 7) -> list[tuple[str, int]]:
    if len(ts) < n:
        return []
    return [(" ".join(ts[i:i+n]), i) for i in range(len(ts) - n + 1)]


def iter_texts(obj: Any, parent_key: str = ""):
    if isinstance(obj, str):
        pk = parent_key.lower()
        if pk in SKIP_KEYS:
            return
        if pk in TEXT_KEYS or len(toks(obj)) >= 5:
            yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_texts(x, parent_key)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in SKIP_KEYS:
                continue
            yield from iter_texts(v, str(k))


def classify_eval_file(rel: pathlib.Path) -> dict[str, Any]:
    top = rel.parts[0] if rel.parts else rel.name
    name = rel.name
    if top == "glue_filtered":
        if ".valid." in name:
            return {"top_dir": top, "file_role": "superglue_validation", "strict_small_score_text": True}
        if ".train." in name:
            return {"top_dir": top, "file_role": "superglue_finetune_train", "strict_small_score_text": False}
        return {"top_dir": top, "file_role": "superglue_unknown", "strict_small_score_text": False}
    if top in STRICT_NON_SCORE_TOP_DIRS:
        return {"top_dir": top, "file_role": "multimodal_not_used_for_strict_small_text_score", "strict_small_score_text": False}
    if top in STRICT_SCORE_TOP_DIRS:
        return {"top_dir": top, "file_role": "strict_small_score_text", "strict_small_score_text": True}
    return {"top_dir": top, "file_role": "not_in_strict_small_text_overall", "strict_small_score_text": False}


def eval_sources() -> list[tuple[pathlib.Path, pathlib.Path]]:
    srcs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for p in sorted(PRISTINE_EVAL_ROOT.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".csv", ".txt"}:
            srcs.append((p, p.relative_to(PRISTINE_EVAL_ROOT)))
    for task_dir in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        p = GLOBALPIQA_EVAL_ROOT / task_dir / "eng_latn.jsonl"
        if p.exists():
            srcs.append((p, pathlib.Path(task_dir) / "eng_latn.jsonl"))
    return srcs


def build_eval_records(n: int = 7):
    records: list[dict[str, Any]] = []
    ng_to_ids: dict[str, list[int]] = defaultdict(list)
    file_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()

    def add_record(rel: pathlib.Path, locator: str, text: str, cls: dict[str, Any]):
        ts = toks(text)
        if len(ts) < n:
            return
        rid = len(records)
        rec = {"id": rid, "file": str(rel), "locator": locator, "text": text, "num_tokens": len(ts), **cls}
        records.append(rec)
        file_counts[cls["top_dir"]] += 1
        text_counts["strict_score" if cls["strict_small_score_text"] else "non_score"] += 1
        for ng, _ in ngrams_from_tokens(ts, n):
            ng_to_ids[ng].append(rid)

    for p, rel in eval_sources():
        cls = classify_eval_file(rel)
        try:
            if p.suffix.lower() == ".jsonl":
                with p.open(encoding="utf-8") as f:
                    for li, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            for text in iter_texts(obj):
                                add_record(rel, f"line:{li}", text, cls)
                        except json.JSONDecodeError:
                            add_record(rel, f"line:{li}", line, cls)
            elif p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                for text in iter_texts(obj):
                    add_record(rel, "json", text, cls)
            elif p.suffix.lower() == ".csv":
                with p.open(encoding="utf-8", newline="") as f:
                    for ri, row in enumerate(csv.DictReader(f), 1):
                        for key, val in row.items():
                            if val and isinstance(val, str):
                                add_record(rel, f"row:{ri}:col:{key}", val, cls)
            else:
                add_record(rel, "txt", p.read_text(encoding="utf-8", errors="ignore"), cls)
        except Exception as exc:
            add_record(pathlib.Path("_read_error") / rel, "exception", f"{type(exc).__name__}: {exc}", {"top_dir": "_read_error", "file_role": "read_error", "strict_small_score_text": False})
    return records, ng_to_ids, dict(file_counts), dict(text_counts)


def load_jsonl_by(path: pathlib.Path, key: str) -> dict[Any, dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[obj[key]] = obj
    return out


def load_optional_by_prompt(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj.get("prompt_id")
            if not pid and obj.get("sentence_id") is not None:
                pid = f"frontier_consolidation_fwcompact_medium_{int(obj['sentence_id']):06d}"
            if pid:
                out[str(pid)] = obj
    return out


def prompt_id_from_pair_id(pair_id: str) -> str:
    return pair_id.split(":", 1)[1] if ":" in pair_id else pair_id


def longest_common_span(a: list[str], b: list[str]) -> tuple[int, int, int, str]:
    if not a or not b:
        return 0, -1, -1, ""
    prev = [0] * (len(b) + 1)
    best = (0, -1, -1)
    for i, at in enumerate(a, 1):
        cur = [0] * (len(b) + 1)
        for j, bt in enumerate(b, 1):
            if at == bt:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best[0]:
                    best = (cur[j], i - cur[j], j - cur[j])
        prev = cur
    length, ia, ib = best
    return length, ia, ib, " ".join(a[ia:ia + length]) if length else ""


def pair_overlap_summary(pair: dict[str, Any], row_ngrams: set[str]) -> dict[str, Any]:
    src_t = toks(pair.get("source_text", ""))
    rew_t = toks(pair.get("rewrite_text", ""))
    src_ng = {ng for ng, _ in ngrams_from_tokens(src_t, 7)}
    rew_ng = {ng for ng, _ in ngrams_from_tokens(rew_t, 7)}
    return {
        "pair_id": pair.get("pair_id"),
        "key": pair.get("key"),
        "sentence_id": pair.get("sentence_id"),
        "doc_id": pair.get("doc_id"),
        "domain_hits": pair.get("domain_hits"),
        "source_words": pair.get("source_words"),
        "rewrite_words": pair.get("rewrite_words"),
        "content_recall": pair.get("content_recall"),
        "entity_recall": pair.get("entity_recall"),
        "number_recall": pair.get("number_recall"),
        "matched_ngrams_in_source_text": sorted(row_ngrams & src_ng),
        "matched_ngrams_in_rewrite_text": sorted(row_ngrams & rew_ng),
        "source_text": pair.get("source_text"),
        "rewrite_text": pair.get("rewrite_text"),
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records, ng_to_ids, eval_record_counts_by_top, eval_text_counts = build_eval_records(7)
    pairs = load_jsonl_by(PAIRS, "pair_id")
    meta_by_row = load_jsonl_by(CHANGED_META, "row_index")
    prompts = load_optional_by_prompt(PROMPTS)
    accepted = load_optional_by_prompt(ACCEPTED)
    raw_outputs = load_optional_by_prompt(RAW_OUTPUTS)

    rows_any_strict = []
    rows_any_nonscore = []
    scanned_changed_rows = 0
    scanned_changed_words = 0
    with CANDIDATE_10M.open(encoding="utf-8") as f:
        for file_row_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "cleanqwen_fineweb_compact_view_reinvest":
                continue
            scanned_changed_rows += 1
            scanned_changed_words += int(row.get("words") or 0)
            row_tokens = toks(row.get("text", ""))
            row_ngram_positions = dict(ngrams_from_tokens(row_tokens, 7))
            strict_matches: dict[str, list[int]] = {}
            nonscore_matches: dict[str, list[int]] = {}
            for ng in row_ngram_positions:
                ids = ng_to_ids.get(ng)
                if not ids:
                    continue
                sids = [rid for rid in ids if records[rid]["strict_small_score_text"]]
                nids = [rid for rid in ids if not records[rid]["strict_small_score_text"]]
                if sids:
                    strict_matches[ng] = sids
                if nids:
                    nonscore_matches[ng] = nids
            if not strict_matches and not nonscore_matches:
                continue

            meta = meta_by_row.get(file_row_index, {})
            row_pair_ids = meta.get("pair_ids", [])
            row_strict_ngrams = set(strict_matches.keys())
            pair_records = []
            for pid in row_pair_ids:
                pair = pairs.get(pid)
                if not pair:
                    pair_records.append({"pair_id": pid, "missing_pair_record": True})
                    continue
                prec = pair_overlap_summary(pair, row_strict_ngrams)
                prompt_id = prompt_id_from_pair_id(pid)
                pr = prompts.get(prompt_id)
                if pr:
                    prec["prompt_record"] = {k: pr.get(k) for k in ["prompt_id", "sentence_id", "doc_id", "text", "prompt"] if k in pr}
                ar = accepted.get(prompt_id)
                if ar:
                    prec["accepted_rewrite_record"] = {k: ar.get(k) for k in ["prompt_id", "accepted", "rewrite", "raw_output", "selection_reason", "source_risks"] if k in ar}
                rr = raw_outputs.get(prompt_id)
                if rr:
                    prec["raw_generation_record"] = {k: rr.get(k) for k in ["prompt_id", "source_text", "raw_output"] if k in rr}
                pair_records.append(prec)

            def make_hits(matches: dict[str, list[int]]) -> list[dict[str, Any]]:
                hits = []
                for ng, rids in sorted(matches.items()):
                    for rid in sorted(set(rids)):
                        rec = records[rid]
                        length, rstart, estart, span = longest_common_span(row_tokens, toks(rec["text"]))
                        hits.append({
                            "ngram": ng,
                            "eval_file": rec["file"],
                            "locator": rec["locator"],
                            "top_dir": rec["top_dir"],
                            "file_role": rec["file_role"],
                            "strict_small_score_text": rec["strict_small_score_text"],
                            "longest_common_token_span_len": length,
                            "longest_common_token_span": span,
                            "row_start_token": rstart,
                            "eval_start_token": estart,
                            "eval_text_prefix": rec["text"][:500],
                        })
                return hits

            base_row = {
                "file_row_index": file_row_index,
                "row_index": file_row_index,
                "example_id": row.get("example_id"),
                "words": row.get("words"),
                "row_text": row.get("text"),
                "row_text_prefix": row.get("text", "")[:800],
                "row_meta": meta,
                "pair_records": pair_records,
            }
            if strict_matches:
                rows_any_strict.append({
                    **base_row,
                    "matched_strict_score_ngrams": sorted(row_strict_ngrams),
                    "num_matched_strict_score_ngrams": len(row_strict_ngrams),
                    "strict_eval_hits": make_hits(strict_matches),
                })
            if nonscore_matches:
                rows_any_nonscore.append({
                    "file_row_index": file_row_index,
                    "example_id": row.get("example_id"),
                    "words": row.get("words"),
                    "matched_non_score_ngrams": sorted(nonscore_matches.keys()),
                    "num_matched_non_score_ngrams": len(nonscore_matches),
                    "non_score_eval_hits": make_hits(nonscore_matches),
                })

    strict_by_top = Counter(h["top_dir"] for r in rows_any_strict for h in r["strict_eval_hits"])
    strict_by_role = Counter(h["file_role"] for r in rows_any_strict for h in r["strict_eval_hits"])
    strict_longest = sorted([h["longest_common_token_span_len"] for r in rows_any_strict for h in r["strict_eval_hits"]], reverse=True)
    nonscore_by_top = Counter(h["top_dir"] for r in rows_any_nonscore for h in r["non_score_eval_hits"])
    payload = {
        "status": "CHANGED_BLOCK_OVERLAP_ANCESTRY_CURRENT_OFFICIAL",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": 7,
        "eval_coordinate": {
            "pristine_eval_root": str(PRISTINE_EVAL_ROOT),
            "official_generated_globalpiqa_root": str(GLOBALPIQA_EVAL_ROOT),
            "score_text_definition": "Strict-Small text Overall columns: BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE validation, GlobalPIQA, Reading, AoA. VQA/Winoground and SuperGLUE train are retained only as non-score context.",
        },
        "training_block": {
            "candidate_10m": str(CANDIDATE_10M),
            "changed_source_name": "cleanqwen_fineweb_compact_view_reinvest",
            "scanned_changed_rows": scanned_changed_rows,
            "scanned_changed_words": scanned_changed_words,
        },
        "eval_record_counts_by_top": eval_record_counts_by_top,
        "eval_text_counts": eval_text_counts,
        "strict_score_overlap_summary": {
            "rows_with_strict_score_overlap": len(rows_any_strict),
            "unique_strict_score_ngrams": len({ng for r in rows_any_strict for ng in r["matched_strict_score_ngrams"]}),
            "strict_eval_hit_count": sum(len(r["strict_eval_hits"]) for r in rows_any_strict),
            "strict_eval_hit_by_top_dir": dict(strict_by_top),
            "strict_eval_hit_by_file_role": dict(strict_by_role),
            "longest_common_span_lengths_desc": strict_longest[:30],
        },
        "non_score_overlap_context": {
            "rows_with_non_score_overlap": len(rows_any_nonscore),
            "unique_non_score_ngrams": len({ng for r in rows_any_nonscore for ng in r["matched_non_score_ngrams"]}),
            "non_score_eval_hit_by_top_dir": dict(nonscore_by_top),
        },
        "strict_score_rows": rows_any_strict,
        "interpretation": "This file makes the exact changed-block overlaps auditable under the current official coordinate. Sparse overlaps are provenance signals to inspect, not by themselves evidence that the model saw target labels or that the endpoint score is explained by direct text reuse.",
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_JSONL.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_any_strict), encoding="utf-8")

    lines = [
        "# research — Current-official ancestry for changed-block exact 7-token overlaps",
        "",
        f"Machine-readable JSON: `{OUT_JSON}`",
        f"Row JSONL: `{OUT_JSONL}`",
        "",
        "## Current official coordinate used",
        f"- Pristine full-eval root: `{PRISTINE_EVAL_ROOT}`",
        f"- GlobalPIQA generated by official `dl.py`: `{GLOBALPIQA_EVAL_ROOT}`",
        "- Strict-Small text score columns included: BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE validation, GlobalPIQA, Reading, AoA.",
        "- VQA/Winoground and SuperGLUE train are not counted as Strict-Small text score text here.",
        "",
        "## Counts",
        f"- Changed rows scanned: {scanned_changed_rows}, words: {scanned_changed_words}",
        f"- Rows with exact 7-token overlap to Strict-Small score text: **{len(rows_any_strict)}**",
        f"- Unique exact 7-token spans involved: **{payload['strict_score_overlap_summary']['unique_strict_score_ngrams']}**",
        f"- Strict score hits by top dir: `{dict(strict_by_top)}`",
        f"- Largest longest-common-span lengths: `{strict_longest[:10]}`",
        "",
        "## Row summary",
    ]
    for r in rows_any_strict:
        pair_bits = ", ".join(f"{p.get('pair_id')}[doc={p.get('doc_id')},sid={p.get('sentence_id')}]" for p in r["pair_records"])
        hit_bits = "; ".join(f"{h['eval_file']}:{h['locator']} span={h['longest_common_token_span_len']}" for h in r["strict_eval_hits"][:5])
        lines.append(f"- row {r['row_index']} example {r['example_id']} words {r['words']}: {len(r['matched_strict_score_ngrams'])} spans; pairs {pair_bits}; hits {hit_bits}")
    lines += [
        "",
        "## Interpretation",
        "The ancestry table links each overlapping training row to its compact source/rewrite pairs and the exact official-coordinate evaluation text span. It updates the research overlap work to the current official EWoK release and the regenerated official GlobalPIQA files. The overlap set is sparse and contains generic public-language spans; it should be preserved for endpoint protection and later reviewer/lead judgment together with the research changed-block-vs-heldout comparison.",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "rows_with_strict_score_overlap": len(rows_any_strict),
        "unique_strict_score_ngrams": payload["strict_score_overlap_summary"]["unique_strict_score_ngrams"],
        "strict_eval_hit_by_top_dir": dict(strict_by_top),
        "longest_top10": strict_longest[:10],
        "out_json": str(OUT_JSON),
        "out_jsonl": str(OUT_JSONL),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
