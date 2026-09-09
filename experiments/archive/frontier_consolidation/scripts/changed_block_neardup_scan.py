#!/usr/bin/env python3
"""research near-duplicate scan for the compact_view_reinvest changed block.

This submission-safety scan tests whether individual FineWeb compact packets in the
changed block are near-duplicates of score-bearing official evaluation strings. It
is stronger than the exact 7-gram audit because it uses rare 5-gram retrieval plus
contiguous-span / n-gram-overlap metrics on the source and rewrite sides.

It does not change model/corpus artifacts and does not use evaluation text for
training. It is only a provenance check of the already-frozen endpoint.
"""
from __future__ import annotations

import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

STUDY = Path("experiments/archive/frontier_consolidation")
OUT_DIR = STUDY / "data/changed_block_neardup_scan"
OUT_JSON = OUT_DIR / "changed_block_neardup_scan.json"
OUT_JSONL = OUT_DIR / "changed_block_neardup_hits.jsonl"
NOTE = (STUDY.parents[2] / 'research/notes/frontier_consolidation/changed_block_neardup_scan.md')

EVAL_ROOT = Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval")
PAIRS = Path("experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl")
CHANGED_META = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl")

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
TEXT_KEYS = {
    "sentence", "sentence_good", "sentence_bad", "good_sentence", "bad_sentence",
    "context", "question", "passage", "paragraph", "premise", "hypothesis",
    "text", "text_a", "text_b", "input", "target", "option1", "option2",
    "answer", "correct", "incorrect", "choice1", "choice2", "query", "title",
    "article", "summary", "word", "stem", "ending0", "ending1", "ending2", "ending3",
}
SKIP_KEYS = {"id", "idx", "uid", "guid", "label", "labels", "metadata", "meta", "path", "file", "filename", "source", "source_file"}
STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "in", "on", "to", "for", "with", "by", "as", "at", "from",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those", "there", "their", "his", "her",
    "you", "your", "i", "we", "they", "he", "she", "not", "no", "do", "does", "did", "have", "has", "had", "can", "could",
}
RETR_N = 5
METRIC_N = 5
MAX_DF = 40
MAX_CANDIDATES_PER_PACKET = 450
MAX_HITS_SAVED = 400


def toks(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def ngrams(ts: List[str], n: int) -> List[str]:
    return [" ".join(ts[i:i+n]) for i in range(max(0, len(ts)-n+1))]


def iter_texts(obj: Any, parent_key: str = "") -> Iterable[str]:
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


def classify_eval_file(rel: Path) -> Dict[str, str]:
    top = rel.parts[0] if rel.parts else rel.name
    name = rel.name
    if top == "glue_filtered":
        if ".train." in name:
            return {"top_dir": top, "file_role": "glue_train_for_finetuning", "score_bearing": "no_direct_final_score"}
        if ".valid." in name:
            return {"top_dir": top, "file_role": "glue_valid_scored", "score_bearing": "score_bearing"}
        return {"top_dir": top, "file_role": "glue_unknown", "score_bearing": "unknown"}
    return {"top_dir": top, "file_role": "zero_shot_or_human_scored", "score_bearing": "score_bearing"}


def short(text: str, n: int = 650) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n] + " ..."


def rareish_ngram(ng: str) -> bool:
    ts = ng.split()
    # Avoid purely function-word phrases that create many generic candidates.
    return sum(t not in STOP for t in ts) >= 2


def add_eval_record(records: List[Dict[str, Any]], rel: Path, locator: str, text: str, cls: Dict[str, str]) -> None:
    if cls["score_bearing"] != "score_bearing":
        return
    ts = toks(text)
    if len(ts) < 7:
        return
    records.append({
        "id": len(records),
        "file": str(rel),
        "locator": locator,
        "top_dir": cls["top_dir"],
        "file_role": cls["file_role"],
        "tokens": ts,
        "text": text,
        "text_excerpt": short(text),
    })


def load_score_bearing_eval_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for p in sorted(EVAL_ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".json", ".jsonl", ".csv", ".txt"}:
            continue
        rel = p.relative_to(EVAL_ROOT)
        cls = classify_eval_file(rel)
        if cls["score_bearing"] != "score_bearing":
            continue
        try:
            if p.suffix.lower() == ".jsonl":
                with p.open(encoding="utf-8") as f:
                    for li, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            for text in iter_texts(obj):
                                add_eval_record(records, rel, f"line:{li}", text, cls)
                        except json.JSONDecodeError:
                            add_eval_record(records, rel, f"line:{li}", line, cls)
            elif p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                for text in iter_texts(obj):
                    add_eval_record(records, rel, "json", text, cls)
            elif p.suffix.lower() == ".csv":
                with p.open(encoding="utf-8", newline="") as f:
                    for ri, row in enumerate(csv.DictReader(f), 1):
                        for key, val in row.items():
                            if val and isinstance(val, str):
                                add_eval_record(records, rel, f"row:{ri}:col:{key}", val, cls)
            else:
                add_eval_record(records, rel, "txt", p.read_text(encoding="utf-8", errors="ignore"), cls)
        except Exception:
            continue
    return records


def build_index(records: List[Dict[str, Any]]) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    df: Counter[str] = Counter()
    per_record: List[set[str]] = []
    for rec in records:
        ngs = {ng for ng in ngrams(rec["tokens"], RETR_N) if rareish_ngram(ng)}
        per_record.append(ngs)
        df.update(ngs)
    index: Dict[str, List[int]] = defaultdict(list)
    for rec, ngs in zip(records, per_record):
        rid = rec["id"]
        for ng in ngs:
            if df[ng] <= MAX_DF:
                index[ng].append(rid)
    return index, dict(df)


def longest_common_contiguous(a: List[str], b: List[str]) -> Tuple[int, str]:
    # Rolling dynamic program, safe for short/medium strings after candidate retrieval.
    if not a or not b:
        return 0, ""
    prev = [0] * (len(b) + 1)
    best_len = 0
    best_end = 0
    for i, at in enumerate(a, 1):
        cur = [0] * (len(b) + 1)
        for j, bt in enumerate(b, 1):
            if at == bt:
                cur[j] = prev[j-1] + 1
                if cur[j] > best_len:
                    best_len = cur[j]
                    best_end = i
        prev = cur
    span = " ".join(a[best_end-best_len:best_end]) if best_len else ""
    return best_len, span


def load_row_index_for_pairs() -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    if not CHANGED_META.exists():
        return out
    with CHANGED_META.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            row_index = rec.get("row_index")
            for pid in rec.get("pair_ids", []):
                out[pid].append(row_index)
    return dict(out)


def packet_candidate_ids(ts: List[str], index: Dict[str, List[int]], df: Dict[str, int]) -> Counter[int]:
    ctr: Counter[int] = Counter()
    for ng in set(ngrams(ts, RETR_N)):
        if not rareish_ngram(ng):
            continue
        if df.get(ng, 10**9) <= MAX_DF:
            ctr.update(index.get(ng, []))
    return ctr


def metrics(pkt_toks: List[str], ev_toks: List[str]) -> Dict[str, Any]:
    p5 = set(ngrams(pkt_toks, METRIC_N))
    e5 = set(ngrams(ev_toks, METRIC_N))
    inter5 = p5 & e5
    union5 = p5 | e5
    lcs_len, lcs_span = longest_common_contiguous(pkt_toks, ev_toks)
    p_tok = set(pkt_toks)
    e_tok = set(ev_toks)
    tok_inter = p_tok & e_tok
    return {
        "shared_5grams": len(inter5),
        "packet_5grams": len(p5),
        "eval_5grams": len(e5),
        "jaccard_5gram": (len(inter5) / len(union5)) if union5 else 0.0,
        "packet_5gram_coverage": (len(inter5) / len(p5)) if p5 else 0.0,
        "eval_5gram_coverage": (len(inter5) / len(e5)) if e5 else 0.0,
        "token_jaccard": (len(tok_inter) / len(p_tok | e_tok)) if (p_tok or e_tok) else 0.0,
        "longest_common_contiguous_tokens": lcs_len,
        "longest_common_span": lcs_span,
    }


def is_concerning(m: Dict[str, Any], pkt_len: int, eval_len: int) -> bool:
    # High-recall thresholds: include anything plausibly more than a generic phrase.
    if m["longest_common_contiguous_tokens"] >= 16:
        return True
    if m["longest_common_contiguous_tokens"] >= 12 and m["shared_5grams"] >= 4:
        return True
    if m["jaccard_5gram"] >= 0.22 and m["shared_5grams"] >= 5:
        return True
    if m["eval_5gram_coverage"] >= 0.35 and m["shared_5grams"] >= 5 and eval_len <= 80:
        return True
    if m["packet_5gram_coverage"] >= 0.40 and m["shared_5grams"] >= 5 and pkt_len <= 80:
        return True
    return False


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    records = load_score_bearing_eval_records()
    index, df = build_index(records)
    pair_to_rows = load_row_index_for_pairs()
    hits: List[Dict[str, Any]] = []
    packet_count = 0
    candidate_metric_count = 0
    pair_side_counts = Counter()
    hit_by_top = Counter()
    hit_by_file_role = Counter()
    max_span = 0
    max_jaccard = 0.0

    with PAIRS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pair = json.loads(line)
            pair_id = pair.get("pair_id")
            for side in ["source_text", "rewrite_text"]:
                text = pair.get(side) or ""
                ts = toks(text)
                if len(ts) < 7:
                    continue
                packet_count += 1
                cand_counter = packet_candidate_ids(ts, index, df)
                if not cand_counter:
                    continue
                # Prioritize candidates with more rare shared retrieval grams.
                for rid, retrieval_shared in cand_counter.most_common(MAX_CANDIDATES_PER_PACKET):
                    ev = records[rid]
                    m = metrics(ts, ev["tokens"])
                    candidate_metric_count += 1
                    max_span = max(max_span, int(m["longest_common_contiguous_tokens"]))
                    max_jaccard = max(max_jaccard, float(m["jaccard_5gram"]))
                    if not is_concerning(m, len(ts), len(ev["tokens"])):
                        continue
                    hit = {
                        "pair_id": pair_id,
                        "changed_block_rows": pair_to_rows.get(pair_id, []),
                        "side": side,
                        "source_words": pair.get("source_words"),
                        "rewrite_words": pair.get("rewrite_words"),
                        "sentence_id": pair.get("sentence_id"),
                        "doc_id": pair.get("doc_id"),
                        "domain_hits": pair.get("domain_hits"),
                        "content_recall": pair.get("content_recall"),
                        "entity_recall": pair.get("entity_recall"),
                        "number_recall": pair.get("number_recall"),
                        "packet_text": short(text, 900),
                        "eval_file": ev["file"],
                        "eval_locator": ev["locator"],
                        "eval_top_dir": ev["top_dir"],
                        "eval_file_role": ev["file_role"],
                        "eval_text": ev["text_excerpt"],
                        "retrieval_shared_rare5": retrieval_shared,
                        **m,
                    }
                    hits.append(hit)
                    pair_side_counts[(pair_id, side)] += 1
                    hit_by_top[ev["top_dir"]] += 1
                    hit_by_file_role[ev["file_role"]] += 1

    # Sort by most concerning continuous span then dense n-gram match.
    hits.sort(key=lambda h: (h["longest_common_contiguous_tokens"], h["jaccard_5gram"], h["shared_5grams"]), reverse=True)
    saved_hits = hits[:MAX_HITS_SAVED]
    OUT_JSONL.write_text("".join(json.dumps(h, ensure_ascii=False) + "\n" for h in saved_hits), encoding="utf-8")
    payload = {
        "status": "CHANGED_BLOCK_NEARDUP_SCAN",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Near-duplicate provenance scan of individual compact source/rewrite packets against score-bearing official evaluation strings.",
        "parameters": {
            "retrieval_ngram": RETR_N,
            "metric_ngram": METRIC_N,
            "max_df": MAX_DF,
            "max_candidates_per_packet": MAX_CANDIDATES_PER_PACKET,
            "thresholds": "span>=16 OR span>=12 & shared5>=4 OR jaccard5>=0.22 & shared5>=5 OR eval_coverage>=0.35 & shared5>=5 & eval_len<=80 OR packet_coverage>=0.40 & shared5>=5 & packet_len<=80",
        },
        "eval_records_score_bearing": len(records),
        "index_unique_rare_5grams": len(index),
        "packets_scanned_sides": packet_count,
        "candidate_metric_count": candidate_metric_count,
        "concerning_hit_count": len(hits),
        "concerning_unique_pair_sides": len(pair_side_counts),
        "hits_saved": len(saved_hits),
        "hit_by_top_dir": {str(k): int(v) for k, v in hit_by_top.items()},
        "hit_by_file_role": {str(k): int(v) for k, v in hit_by_file_role.items()},
        "max_longest_common_contiguous_tokens_seen": max_span,
        "max_5gram_jaccard_seen": max_jaccard,
        "top_hits": saved_hits[:40],
        "out_jsonl": str(OUT_JSONL),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("# research changed-block near-duplicate scan")
    lines.append("")
    lines.append("This CPU-only scan indexes score-bearing official evaluation strings by rare 5-grams and compares each individual compact source/rewrite packet in the reinvest changed block against retrieved candidates. It is a provenance check only; no data or model artifact is changed.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Score-bearing eval text records indexed: {len(records)}.")
    lines.append(f"- Unique rare 5-grams in index (df <= {MAX_DF}): {len(index)}.")
    lines.append(f"- Packet sides scanned: {packet_count}.")
    lines.append(f"- Retrieved candidate packet/eval comparisons scored: {candidate_metric_count}.")
    lines.append(f"- Threshold-passing near-duplicate hits: {len(hits)} across {len(pair_side_counts)} pair-sides.")
    lines.append(f"- Hit distribution by top dir: {dict(hit_by_top)}.")
    lines.append(f"- Maximum contiguous token span observed among scored candidates: {max_span}.")
    lines.append(f"- Maximum 5-gram Jaccard observed among scored candidates: {max_jaccard:.4f}.")
    lines.append("")
    lines.append("## Top threshold-passing hits")
    lines.append("")
    for i, h in enumerate(saved_hits[:20], 1):
        lines.append(f"### {i}. {h['pair_id']} {h['side']} → {h['eval_file']} {h['eval_locator']}")
        lines.append(f"- span={h['longest_common_contiguous_tokens']} shared5={h['shared_5grams']} jaccard5={h['jaccard_5gram']:.3f} eval_cov={h['eval_5gram_coverage']:.3f} packet_cov={h['packet_5gram_coverage']:.3f}")
        lines.append(f"- common span: `{h['longest_common_span']}`")
        lines.append(f"- packet: {h['packet_text']}")
        lines.append(f"- eval: {h['eval_text']}")
        lines.append("")
    lines.append("## Reading")
    lines.append("")
    if len(hits) == 0:
        lines.append("No packet/eval pair passed the high-recall near-duplicate thresholds. This strengthens the exact-overlap reading: the changed block has sparse generic exact overlaps and no obvious near-copy of score-bearing eval text at packet level.")
    else:
        lines.append("Threshold-passing hits require direct human/model reading. Generic topical phrases or public named events are less concerning than whole-question/answer, full premise-hypothesis, full passage, or answer-label overlap. Use the JSONL for full hit reading before final packaging.")
    lines.append("")
    lines.append(f"Machine-readable JSON: `{OUT_JSON}`")
    lines.append(f"Hit JSONL: `{OUT_JSONL}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "eval_records_score_bearing": len(records),
        "packets_scanned_sides": packet_count,
        "candidate_metric_count": candidate_metric_count,
        "concerning_hit_count": len(hits),
        "max_span": max_span,
        "max_5gram_jaccard": round(max_jaccard, 6),
        "out_json": str(OUT_JSON),
        "out_jsonl": str(OUT_JSONL),
        "note": str(NOTE),
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
