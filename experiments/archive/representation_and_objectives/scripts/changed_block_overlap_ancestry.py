#!/usr/bin/env python3
"""research source-ancestry table for changed-block score-bearing exact 7-gram overlaps.

The research refined overlap audit showed only 8 compact_view_reinvest changed-block
rows with exact 7-word overlap against score-bearing evaluation text, fewer than the
held-out clean-Qwen rows they replaced. This script makes those 8 rows auditable:
row_index/example_id, matching ngrams, score-bearing eval file roles, row pair_ids,
source sentence/doc ids, compact rewrite, and longest contiguous row/eval token span.
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
import textwrap
from collections import defaultdict, Counter
from typing import Any

ROOT = _public_path('experiments/archive')
STUDY = _public_path('data/external/representation_and_objectives')
WORKSPACE = _public_path('data/external/workspace')
EVAL_ROOT = _public_path('data/external/full_eval')
CANDIDATE_10M = _public_path('data/external/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
CHANGED_META = _public_path('data/external/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl')
PAIRS = _public_path('data/external/selected_compact_reinvest_pairs.jsonl')
PROMPTS = _public_path('data/external/fineweb_medium_compact_prompts_all.jsonl')
ACCEPTED = _public_path('data/external/medium_compact_ws_accepted_rewrites.jsonl')
RAW_OUTPUTS = _public_path('data/external/medium_compact_ws_rows.jsonl')
OUT_DIR = _public_path('data/external/overlap_ancestry')
OUT_JSON = _public_path('data/external/changed_block_score_overlap_ancestry.json')
OUT_JSONL = _public_path('data/external/changed_block_score_overlap_ancestry_rows.jsonl')
NOTE = _public_path('data/external/changed_block_score_overlap_ancestry.md')

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
TEXT_KEYS = {
    "sentence", "sentence_good", "sentence_bad", "good_sentence", "bad_sentence",
    "context", "question", "passage", "paragraph", "premise", "hypothesis",
    "text", "text_a", "text_b", "input", "target", "option1", "option2",
    "answer", "correct", "incorrect", "choice1", "choice2", "query", "title",
    "article", "summary", "word", "stem", "ending0", "ending1", "ending2", "ending3",
}
SKIP_KEYS = {"id", "idx", "uid", "guid", "label", "labels", "metadata", "meta", "path", "file", "filename", "source", "source_file"}


def toks(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def ngrams_from_tokens(ts: list[str], n: int = 7) -> list[tuple[str, int]]:
    return [(" ".join(ts[i:i+n]), i) for i in range(max(0, len(ts)-n+1))]


def iter_texts(obj: Any, parent_key: str = ""):
    if isinstance(obj, str):
        if parent_key.lower() in SKIP_KEYS:
            return
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


def classify_eval_file(rel: pathlib.Path) -> dict[str, str]:
    top = rel.parts[0] if rel.parts else rel.name
    name = rel.name
    if top == "glue_filtered":
        if ".train." in name:
            return {"top_dir": top, "file_role": "glue_train_for_finetuning", "score_bearing": "no_direct_final_score"}
        if ".valid." in name:
            return {"top_dir": top, "file_role": "glue_valid_scored", "score_bearing": "score_bearing"}
        return {"top_dir": top, "file_role": "glue_unknown", "score_bearing": "unknown"}
    return {"top_dir": top, "file_role": "zero_shot_or_human_scored", "score_bearing": "score_bearing"}


def build_eval_records(n: int = 7):
    records: list[dict[str, Any]] = []
    ng_to_ids: dict[str, list[int]] = defaultdict(list)

    def add_record(rel: pathlib.Path, locator: str, text: str, cls: dict[str, str]):
        rid = len(records)
        records.append({"id": rid, "file": str(rel), "locator": locator, "text": text, **cls})
        for ng, _ in ngrams_from_tokens(toks(text), n):
            ng_to_ids[ng].append(rid)

    for p in sorted(EVAL_ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".json", ".jsonl", ".csv", ".txt"}:
            continue
        rel = p.relative_to(EVAL_ROOT)
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
        except Exception:
            # Keep overlap audit robust; missing/unparseable eval files cannot create matches.
            continue
    return records, ng_to_ids


def load_jsonl_by(path: pathlib.Path, key: str) -> dict[Any, dict[str, Any]]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[obj[key]] = obj
    return out


def load_pairs() -> dict[str, dict[str, Any]]:
    return load_jsonl_by(PAIRS, "pair_id")


def prompt_id_from_pair_id(pair_id: str) -> str:
    return pair_id.split(":", 1)[1] if ":" in pair_id else pair_id


def load_optional_by_prompt(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    out = {}
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


def longest_common_span(a: list[str], b: list[str]) -> tuple[int, int, int, str]:
    # DP over short row/eval token strings; returns length and start positions.
    if not a or not b:
        return 0, -1, -1, ""
    prev = [0] * (len(b) + 1)
    best = (0, -1, -1)
    for i, at in enumerate(a, 1):
        cur = [0] * (len(b) + 1)
        for j, bt in enumerate(b, 1):
            if at == bt:
                cur[j] = prev[j-1] + 1
                if cur[j] > best[0]:
                    best = (cur[j], i-cur[j], j-cur[j])
        prev = cur
    length, ia, ib = best
    return length, ia, ib, " ".join(a[ia:ia+length]) if length > 0 else ""


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _public_path('data/external/notes').mkdir(parents=True, exist_ok=True)
    records, ng_to_ids = build_eval_records(7)
    pairs = load_pairs()
    meta_by_row = load_jsonl_by(CHANGED_META, "row_index")
    prompts = load_optional_by_prompt(PROMPTS)
    accepted = load_optional_by_prompt(ACCEPTED)
    raw_outputs = load_optional_by_prompt(RAW_OUTPUTS)

    ancestry_rows = []
    score_row_count = 0
    with CANDIDATE_10M.open(encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "cleanqwen_fineweb_compact_view_reinvest":
                continue
            row_tokens = toks(row.get("text", ""))
            row_ngram_positions = dict(ngrams_from_tokens(row_tokens, 7))
            matched = {}
            for ng in row_ngram_positions:
                if ng in ng_to_ids:
                    score_ids = [rid for rid in ng_to_ids[ng] if records[rid]["score_bearing"] == "score_bearing"]
                    if score_ids:
                        matched[ng] = score_ids
            if not matched:
                continue
            score_row_count += 1
            meta = meta_by_row.get(row_index, {})
            row_pair_ids = meta.get("pair_ids", [])
            row_ngrams = set(matched.keys())
            pair_records = []
            for pid in row_pair_ids:
                pair = pairs.get(pid)
                if not pair:
                    pair_records.append({"pair_id": pid, "missing_pair_record": True})
                    continue
                prec = pair_overlap_summary(pair, row_ngrams)
                prompt_id = prompt_id_from_pair_id(pid)
                if prompt_id in prompts:
                    pr = prompts[prompt_id]
                    prec["prompt_record"] = {k: pr.get(k) for k in ["sentence_id", "doc_id", "source_row", "text", "prompt"] if k in pr}
                if prompt_id in accepted:
                    ar = accepted[prompt_id]
                    prec["accepted_rewrite_record"] = {k: ar.get(k) for k in ["prompt_id", "accepted", "rewrite", "raw_output", "selection_reason", "source_risks"] if k in ar}
                if prompt_id in raw_outputs:
                    rr = raw_outputs[prompt_id]
                    prec["raw_generation_record"] = {k: rr.get(k) for k in ["prompt_id", "source_text", "raw_output"] if k in rr}
                pair_records.append(prec)

            eval_hits = []
            for ng, rids in sorted(matched.items()):
                seen = set()
                for rid in rids:
                    if rid in seen:
                        continue
                    seen.add(rid)
                    rec = records[rid]
                    ltoks = toks(rec["text"])
                    length, rstart, estart, span = longest_common_span(row_tokens, ltoks)
                    eval_hits.append({
                        "ngram": ng,
                        "eval_file": rec["file"],
                        "locator": rec["locator"],
                        "top_dir": rec["top_dir"],
                        "file_role": rec["file_role"],
                        "longest_common_token_span_len": length,
                        "longest_common_token_span": span,
                        "row_start_token": rstart,
                        "eval_start_token": estart,
                        "eval_text_prefix": rec["text"][:500],
                    })
            ancestry_rows.append({
                "row_index": row_index,
                "example_id": row.get("example_id"),
                "words": row.get("words"),
                "row_text": row.get("text"),
                "row_text_prefix": row.get("text", "")[:700],
                "matched_score_bearing_ngrams": sorted(row_ngrams),
                "num_matched_score_bearing_ngrams": len(row_ngrams),
                "eval_hits": eval_hits,
                "row_meta": meta,
                "pair_records": pair_records,
            })

    by_top = Counter(h["top_dir"] for r in ancestry_rows for h in r["eval_hits"])
    by_role = Counter(h["file_role"] for r in ancestry_rows for h in r["eval_hits"])
    longest = sorted([h["longest_common_token_span_len"] for r in ancestry_rows for h in r["eval_hits"]], reverse=True)
    payload = {
        "status": "CHANGED_BLOCK_SCORE_OVERLAP_ANCESTRY",
        "n": 7,
        "candidate_path": str(CANDIDATE_10M),
        "eval_root": str(EVAL_ROOT),
        "rows_with_score_bearing_overlap": len(ancestry_rows),
        "unique_score_bearing_ngrams": len({ng for r in ancestry_rows for ng in r["matched_score_bearing_ngrams"]}),
        "eval_hit_count": sum(len(r["eval_hits"]) for r in ancestry_rows),
        "eval_hit_by_top_dir": dict(by_top),
        "eval_hit_by_file_role": dict(by_role),
        "longest_common_span_lengths_desc": longest[:20],
        "rows": ancestry_rows,
        "interpretation": (
            "This is a provenance/contamination inspection table for the sparse exact 7-word score-bearing overlaps in the FineWeb compact changed block. "
            "It links each overlapping row to the compact pair IDs and source/rewrite text; it does not by itself imply contamination because public web text can share common phrases with benchmark sources, and the changed block had fewer such overlaps than the official rows it replaced."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_JSONL.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ancestry_rows), encoding="utf-8")

    lines = [
        "# research changed-block score-bearing overlap ancestry",
        "",
        f"Machine-readable JSON: `{OUT_JSON}`",
        f"Row-wise JSONL: `{OUT_JSONL}`",
        "",
        f"Rows with score-bearing exact 7-word overlap: **{len(ancestry_rows)}**.",
        f"Unique matched score-bearing 7-grams: **{payload['unique_score_bearing_ngrams']}**.",
        f"Evaluation hits by top dir: `{dict(by_top)}`.",
        f"Evaluation hits by file role: `{dict(by_role)}`.",
        f"Largest row/eval longest-common-span lengths: `{longest[:10]}`.",
        "",
        "## Row summary",
        "",
    ]
    for r in ancestry_rows:
        pair_bits = ", ".join([f"{p.get('pair_id')}[doc={p.get('doc_id')},sid={p.get('sentence_id')}]" for p in r["pair_records"]])
        eval_bits = "; ".join([f"{h['eval_file']}:{h['locator']} span={h['longest_common_token_span_len']}" for h in r["eval_hits"][:4]])
        lines.append(f"- row {r['row_index']} example {r['example_id']} words {r['words']}: {len(r['matched_score_bearing_ngrams'])} matched 7-grams; pairs {pair_bits}; eval hits {eval_bits}")
    lines += [
        "",
        "## Interpretation",
        "",
        "This table makes the sparse overlaps auditable before submission. The overlaps remain a provenance warning to inspect, not an explanation for the 42.0868 endpoint: research already showed the changed block has fewer score-bearing exact 7-gram overlaps than the held-out official rows it replaced. The most important remaining score question is still the official-min_context=0 AoA rerun.",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "rows_with_score_bearing_overlap": len(ancestry_rows),
        "unique_score_bearing_ngrams": payload["unique_score_bearing_ngrams"],
        "eval_hit_count": payload["eval_hit_count"],
        "eval_hit_by_top_dir": dict(by_top),
        "longest_top10": longest[:10],
        "out_json": str(OUT_JSON),
        "out_jsonl": str(OUT_JSONL),
        "note": str(NOTE),
    }, indent=2))


if __name__ == "__main__":
    main()
