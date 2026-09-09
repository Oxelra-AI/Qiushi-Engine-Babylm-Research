#!/usr/bin/env python3
"""research: materialize a CLEAN_NATURAL_SPLIT extent arm.

Purpose
-------
The core DeBERTa result uses designed compact relation packets.  This script
starts the natural-composition extent test by splitting naturally co-occurring
row-internal high-overlap sentence spans already present in the CLEAN stream.
It excludes the dense `qwen_pair_packed` block so the arm tests ordinary BabyLM
subcorpus adjacency rather than the already-designed Qwen-pair block.

Construction
------------
For each natural-source row with a refined high-overlap sentence-span pair
(τ >= 0.50, min 8 content tokens), select the best nonidentical pair and move
the later span out of that row by swapping it with a same-source neutral span
from a row without any selected high-overlap pair.  This preserves total text
multiset within each subcorpus and exact 10M/100M word accounting, while
breaking the selected same-row adjacency for one relation-bearing span per hit
row.  It is not a perfect document-level recomposition; it is a cheap
sensitivity arm whose effect size is bounded by the measured natural dose.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/materialize_clean_natural_split.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
IN_10M = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl')
IN_100M_MAT = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_100M.materialization.json')
META = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json')
OUT = _public_path('experiments/archive/relation_learning/data/clean_natural_split_pools')
NOTE = _public_path('research/documents/relation_learning/data/clean_natural_split_pools/clean_natural_split_summary.md')
META_OUT = _public_path('experiments/archive/relation_learning/data/clean_natural_split_pools/clean_natural_split_metadata.json')
OUT_10M = _public_path('experiments/archive/relation_learning/data/clean_natural_split_pools/cleanqwen_natural_sentence_split_tau050_10M.jsonl')
OUT_100M = _public_path('experiments/archive/relation_learning/data/clean_natural_split_pools/cleanqwen_natural_sentence_split_tau050_100M.jsonl')

STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","it","its","itself","just","ll","m","ma","me","might","more","most","mustn","my","myself","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","she","should","shouldn","so","some","such","t","than","that","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","we","were","weren","what","when","where","which","while","who","whom","why","will","with","won","would","wouldn","you","your","yours","yourself","yourselves","yes","yeah","yep","uh","um","erm","er","huh","oh","okay","ok","mm","mmm","mhm","xxx","mot","chi","fat","bro","sis","exp","inv","int","add","act","com","gra","cod","tim","spa","eng","hun","urs","mar"
}
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,:/-]\d+)*")
SPEAKER_SPLIT_RE = re.compile(r"\s+(?=\*[A-Z]{2,4}:)")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\s+(?==\s*=)|\n+")
CHAT_TIER_LABEL_RE = re.compile(r"(?:^|\s)[*%][A-Za-z]{2,6}:\s*")
BRACKET_RE = re.compile(r"\[[^\]]*\]")
ANGLE_RE = re.compile(r"<[^>]*>")
PAREN_TRACK_RE = re.compile(r"\((?:tracks?|track)[^)]+\)", re.IGNORECASE)
NATURAL_SOURCES = {"childes", "open_subtitles", "bnc_spoken", "gutenberg", "simple_wiki", "switchboard"}


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


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def parse_source(src: str) -> str:
    if "::" in src:
        return src.split("::", 1)[1]
    return src or "unknown"


def strip_scaffold(text: str) -> str:
    x = text.replace("\uFFFD", " ")
    x = CHAT_TIER_LABEL_RE.sub(" ", x)
    x = BRACKET_RE.sub(" ", x)
    x = ANGLE_RE.sub(" ", x)
    x = PAREN_TRACK_RE.sub(" ", x)
    x = re.sub(r"%[a-z]+:\s*", " ", x, flags=re.IGNORECASE)
    return x


def norm_token(tok: str) -> str:
    t = tok.lower().strip("'’")
    if t.endswith("'s"):
        t = t[:-2]
    return t


def content_tokens(text: str) -> list[str]:
    out = []
    for m in TOKEN_RE.finditer(strip_scaffold(text)):
        t = norm_token(m.group(0))
        if not t or t in STOPWORDS:
            continue
        if t.isalpha() and len(t) < 3:
            continue
        out.append(t)
    return out


def split_parts(text: str) -> list[str]:
    tmp = SPEAKER_SPLIT_RE.sub("\n", text)
    parts = [p.strip() for p in SENT_SPLIT_RE.split(tmp) if p.strip()]
    return parts if parts else [text]


def join_parts(parts: list[str]) -> str:
    return " ".join(p.strip() for p in parts if p.strip())


@dataclass
class Span:
    part_index: int
    raw: str
    tokens: list[str]
    token_set: set[str]
    n_content: int
    raw_words: int
    seq: tuple[str, ...]


@dataclass
class RowObj:
    line_index: int
    row: dict[str, Any]
    source: str
    is_natural: bool
    parts: list[str]
    spans: list[Span]
    selected: bool = False
    donor: bool = False
    selected_pair: tuple[int, int] | None = None
    selected_overlap: float = 0.0
    selected_jaccard: float = 0.0
    selected_relation_kind: str = ""
    moved_part_index: int | None = None
    donor_part_index: int | None = None


def build_spans(parts: list[str], min_content: int) -> list[Span]:
    spans = []
    for i, p in enumerate(parts):
        toks = content_tokens(p)
        if len(toks) >= min_content:
            spans.append(Span(i, p, toks, set(toks), len(toks), len(p.split()), tuple(toks)))
    return spans


def pair_metrics(a: Span, b: Span) -> tuple[float, float, int, str]:
    inter = len(a.token_set & b.token_set)
    union = len(a.token_set | b.token_set)
    ov = inter / max(1, min(len(a.token_set), len(b.token_set)))
    jc = inter / max(1, union)
    kind = "exact_sequence" if a.seq == b.seq else ("exact_set" if a.token_set == b.token_set else "nonidentical_high_overlap")
    return ov, jc, inter, kind


def find_best_pair(spans: list[Span], thr: float) -> tuple[tuple[int, int] | None, float, float, str]:
    best = None; best_key = (-1.0, -1.0, -1); best_kind = ""
    for ai in range(len(spans)):
        for bi in range(ai + 1, len(spans)):
            ov, jc, inter, kind = pair_metrics(spans[ai], spans[bi])
            if ov >= thr:
                # Prefer nonidentical high-overlap pairs, then larger overlap/jaccard.
                nonid_bonus = 1 if kind == "nonidentical_high_overlap" else 0
                key = (float(nonid_bonus), ov, jc, inter)
                if key > best_key:
                    best_key = key; best = (ai, bi); best_kind = kind
    if best is None:
        return None, 0.0, 0.0, ""
    return best, best_key[1], best_key[2], best_kind


def count_hits(rows: list[RowObj], thr: float, min_content: int, natural_only: bool = True) -> dict[str, Any]:
    hit_pairs = 0; row_hits = 0; exact_seq = 0; set_exact = 0; nonid = 0; overlaps = []
    by_src: dict[str, dict[str, Any]] = defaultdict(lambda: {"hit_pairs":0,"row_hits":0,"exact_seq":0,"nonid":0})
    for ro in rows:
        if natural_only and not ro.is_natural:
            continue
        spans = build_spans(ro.parts, min_content)
        any_hit = False
        for i in range(len(spans)):
            for j in range(i+1, len(spans)):
                ov, jc, inter, kind = pair_metrics(spans[i], spans[j])
                if ov >= thr:
                    hit_pairs += 1; any_hit = True; overlaps.append(ov); by_src[ro.source]["hit_pairs"] += 1
                    exact_seq += int(kind == "exact_sequence"); set_exact += int(kind in {"exact_sequence", "exact_set"}); nonid += int(kind == "nonidentical_high_overlap")
                    by_src[ro.source]["exact_seq"] += int(kind == "exact_sequence"); by_src[ro.source]["nonid"] += int(kind == "nonidentical_high_overlap")
        if any_hit:
            row_hits += 1; by_src[ro.source]["row_hits"] += 1
    return {"threshold": thr, "natural_only": natural_only, "hit_pairs": hit_pairs, "row_hits": row_hits, "exact_sequence": exact_seq, "set_exact": set_exact, "nonidentical": nonid, "mean_overlap": statistics.mean(overlaps) if overlaps else float('nan'), "by_source": by_src}


def load_rows(min_content: int, thr: float) -> list[RowObj]:
    rows: list[RowObj] = []
    with IN_10M.open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            row = json.loads(line)
            src = parse_source(str(row.get("source") or ""))
            parts = split_parts(str(row.get("text") or ""))
            spans = build_spans(parts, min_content)
            ro = RowObj(line_idx, row, src, src in NATURAL_SOURCES, parts, spans)
            if ro.is_natural:
                pair, ov, jc, kind = find_best_pair(spans, thr)
                if pair is not None:
                    ro.selected = True
                    ro.selected_pair = pair
                    ro.selected_overlap = ov
                    ro.selected_jaccard = jc
                    ro.selected_relation_kind = kind
            rows.append(ro)
    return rows


def choose_donors(rows: list[RowObj], min_content: int, max_len_ratio: float) -> list[dict[str, Any]]:
    # Candidate donors are same-source rows without a selected hit. Use one donor row once.
    donor_by_source: dict[str, list[tuple[int, int, int]]] = defaultdict(list)  # (raw_words, row_idx, span_idx)
    for ri, ro in enumerate(rows):
        if not ro.is_natural or ro.selected:
            continue
        for si, sp in enumerate(ro.spans):
            if sp.raw_words >= min_content:
                donor_by_source[ro.source].append((sp.raw_words, ri, si))
    for src in donor_by_source:
        donor_by_source[src].sort()
    swaps: list[dict[str, Any]] = []
    used_donor_rows: set[int] = set()
    selected_rows = [ri for ri, ro in enumerate(rows) if ro.selected and ro.is_natural]
    # Prioritize stronger overlaps. This makes the split arm target the most relation-like cases first if donors are insufficient.
    selected_rows.sort(key=lambda ri: (rows[ri].selected_overlap, rows[ri].selected_jaccard), reverse=True)
    for ri in selected_rows:
        ro = rows[ri]
        assert ro.selected_pair is not None
        _, bi = ro.selected_pair
        move_span = ro.spans[bi]
        target_words = move_span.raw_words
        best = None; best_score = (10**9, 10**9)
        # same-source first, then any natural source if same-source donor exhausted.
        candidate_sources = [ro.source] + [s for s in sorted(donor_by_source) if s != ro.source]
        for src in candidate_sources:
            for raw_words, dri, dsi in donor_by_source.get(src, []):
                if dri in used_donor_rows or dri == ri:
                    continue
                ratio = max(raw_words, target_words) / max(1, min(raw_words, target_words))
                if ratio > max_len_ratio:
                    continue
                score = (0 if src == ro.source else 1, abs(raw_words - target_words), raw_words)
                if score < best_score:
                    best_score = score; best = (src, raw_words, dri, dsi)
            if best is not None and best[0] == ro.source:
                break
        if best is None:
            continue
        donor_src, donor_words, dri, dsi = best
        donor = rows[dri]
        donor.donor = True
        used_donor_rows.add(dri)
        ro.moved_part_index = move_span.part_index
        donor_span = donor.spans[dsi]
        donor.donor_part_index = donor_span.part_index
        swaps.append({
            "selected_row_index": ro.line_index,
            "selected_example_id": ro.row.get("example_id"),
            "selected_source": ro.source,
            "selected_pair_span_indices": f"{ro.spans[ro.selected_pair[0]].part_index},{ro.spans[ro.selected_pair[1]].part_index}",
            "selected_overlap": ro.selected_overlap,
            "selected_jaccard": ro.selected_jaccard,
            "selected_relation_kind": ro.selected_relation_kind,
            "moved_span_part_index": move_span.part_index,
            "moved_span_words": target_words,
            "moved_span_content_tokens": move_span.n_content,
            "donor_row_index": donor.line_index,
            "donor_example_id": donor.row.get("example_id"),
            "donor_source": donor.source,
            "donor_span_part_index": donor_span.part_index,
            "donor_span_words": donor_words,
            "donor_span_content_tokens": donor_span.n_content,
            "same_source_swap": donor_src == ro.source,
            "length_abs_diff": abs(donor_words - target_words),
            "moved_span_text": move_span.raw[:400],
            "donor_span_text": donor_span.raw[:400],
        })
    return swaps


def apply_swaps(rows: list[RowObj], swaps: list[dict[str, Any]]) -> None:
    for sw in swaps:
        sri = int(sw["selected_row_index"]); dri = int(sw["donor_row_index"])
        # line_index equals list index because rows are loaded in stream order.
        sel = rows[sri]; donor = rows[dri]
        mi = int(sw["moved_span_part_index"]); di = int(sw["donor_span_part_index"])
        moved = sel.parts[mi]
        neutral = donor.parts[di]
        sel.parts[mi] = neutral
        donor.parts[di] = moved


def count_words(path: pathlib.Path) -> dict[str, Any]:
    rows = 0; words = 0; by_source = defaultdict(int)
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); rows += 1; w = int(r.get("words") or len(str(r.get("text") or "").split())); words += w; by_source[parse_source(str(r.get("source") or ""))] += w
    return {"rows": rows, "words": words, "by_source_words": dict(by_source), "exact_10M": words == 10_000_000, "exact_100M": words == 100_000_000}


def write_streams(rows: list[RowObj]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with OUT_10M.open("w", encoding="utf-8") as f:
        for ro in rows:
            new_row = dict(ro.row)
            if ro.selected or ro.donor:
                text = join_parts(ro.parts)
                new_row["text"] = text
                new_row["words"] = len(text.split())
                new_row["natural_split_modified"] = True
            else:
                new_row["natural_split_modified"] = False
            f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
    with OUT_100M.open("w", encoding="utf-8") as out:
        data = OUT_10M.read_text(encoding="utf-8")
        for _ in range(10):
            out.write(data)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8"); return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--min-content", type=int, default=8)
    ap.add_argument("--max-len-ratio", type=float, default=1.8)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for p in [IN_10M, IN_100M_MAT, META]:
        if not p.exists():
            raise FileNotFoundError(p)
    mat = read_json(IN_100M_MAT)
    if int(mat.get("repeats", 0)) != 10:
        raise RuntimeError("CLEAN 100M is not repeat10; natural split accounting would need repair")
    rows = load_rows(args.min_content, args.threshold)
    before_nat = count_hits(rows, args.threshold, args.min_content, natural_only=True)
    before_all = count_hits(rows, args.threshold, args.min_content, natural_only=False)
    swaps = choose_donors(rows, args.min_content, args.max_len_ratio)
    if args.plan_only:
        print(json.dumps({"status":"NATURAL_SPLIT_PLAN", "selected_rows":sum(1 for r in rows if r.selected), "planned_swaps":len(swaps), "before_nat_hits":before_nat["hit_pairs"], "before_all_hits":before_all["hit_pairs"], "threshold":args.threshold, "min_content":args.min_content}, indent=2, ensure_ascii=False), flush=True)
        return
    apply_swaps(rows, swaps)
    after_nat = count_hits(rows, args.threshold, args.min_content, natural_only=True)
    after_all = count_hits(rows, args.threshold, args.min_content, natural_only=False)
    write_streams(rows)
    c10 = count_words(OUT_10M); c100 = count_words(OUT_100M)
    sha10 = sha256_file(OUT_10M); sha100 = sha256_file(OUT_100M)
    write_csv(_public_path('experiments/archive/relation_learning/data/clean_natural_split_pools/natural_split_swaps.csv'), swaps)
    by_src_rows=[]
    for src in sorted(set(before_nat["by_source"]) | set(after_nat["by_source"])):
        b=before_nat["by_source"].get(src, {"hit_pairs":0,"row_hits":0,"exact_seq":0,"nonid":0})
        a=after_nat["by_source"].get(src, {"hit_pairs":0,"row_hits":0,"exact_seq":0,"nonid":0})
        by_src_rows.append({"source":src,"before_hit_pairs":b["hit_pairs"],"after_hit_pairs":a["hit_pairs"],"delta_hit_pairs":a["hit_pairs"]-b["hit_pairs"],"before_row_hits":b["row_hits"],"after_row_hits":a["row_hits"],"before_nonidentical":b["nonid"],"after_nonidentical":a["nonid"]})
    write_csv(_public_path('experiments/archive/relation_learning/data/clean_natural_split_pools/natural_split_before_after_by_source.csv'), by_src_rows)
    meta0 = read_json(META)
    result = {
        "status":"CLEAN_NATURAL_SPLIT_MATERIALIZED",
        "created_utc":now(),"elapsed_sec":round(time.time()-t0,2),
        "input_clean_10M":rel(IN_10M),"input_clean_100M_materialization":rel(IN_100M_MAT),
        "threshold_overlap_coef":args.threshold,"min_content_tokens":args.min_content,"max_len_ratio":args.max_len_ratio,
        "construction":"swap selected high-overlap sentence span with same-source neutral span from non-selected row; qwen_pair_packed excluded and left unchanged",
        "selected_natural_rows":sum(1 for r in rows if r.selected),"performed_swaps":len(swaps),
        "before_nat_hits_tau":before_nat,"after_nat_hits_tau":after_nat,"before_all_hits_tau":before_all,"after_all_hits_tau":after_all,
        "word_count_10M":c10,"word_count_100M":c100,
        "sha256":{"cleanqwen_natural_sentence_split_tau050_10M.jsonl":sha10,"cleanqwen_natural_sentence_split_tau050_100M.jsonl":sha100},
        "files":{"stream_10M":rel(OUT_10M),"stream_100M":rel(OUT_100M),"swaps":rel(_public_path('experiments/archive/relation_learning/data/clean_natural_split_pools/natural_split_swaps.csv')),"by_source":rel(_public_path('experiments/archive/relation_learning/data/clean_natural_split_pools/natural_split_before_after_by_source.csv')),"summary":rel(NOTE),"metadata":rel(META_OUT)},
        "designed_reference":{"selected_compact_pairs_per_10M":(meta0.get("pair_summary") or {}).get("pairs"),"selected_compact_pair_words_per_10M":(meta0.get("dose") or {}).get("selected_pair_words")},
        "pre_stated_readout":"Train recipe-matched DeBERTa seed43022 only if compute is free after higher-priority extent tasks. Score Wikipedia T/U/N, natural-copy, compact rewrite T/U/N, and Entity against CLEAN; predicted direction if natural adjacency matters: lower copy gain and improved true-source use on source-absent changed/substituted targets relative to CLEAN_NATURAL_SPLIT, with smaller magnitude than designed REPEAT_SPLIT because dose is much lower.",
        "no_leaderboard_submission":True,
    }
    # Remove sets from nested by_source before JSON serialization.
    for k in ["before_nat_hits_tau","after_nat_hits_tau","before_all_hits_tau","after_all_hits_tau"]:
        by = result[k].get("by_source", {})
        result[k]["by_source"] = {src: dict(vals) for src, vals in by.items()}
    write_json(META_OUT, result)
    lines=[]
    lines.append("# research CLEAN_NATURAL_SPLIT materialization")
    lines.append("")
    lines.append("This is a ready-to-train extent arm, not evidence yet. It begins the natural-composition test motivated by the refined CLEAN adjacency audit.")
    lines.append("")
    lines.append("## Construction")
    lines.append("")
    lines.append(f"Selected natural-source rows with a refined sentence-span content-overlap coefficient at least `{args.threshold}` and at least `{args.min_content}` content tokens per span. `qwen_pair_packed` was excluded and left unchanged. For each selected row, the later span of its strongest high-overlap pair was swapped with a same-source neutral span from a row without a selected hit when possible, using maximum raw word-count ratio `{args.max_len_ratio}`. This preserves the text multiset within natural subcorpora and keeps total word accounting exact, while removing one relation-bearing adjacency per selected row.")
    lines.append("")
    lines.append("## Audit")
    lines.append("")
    lines.append(f"- Selected natural rows: `{result['selected_natural_rows']}`; performed swaps: `{result['performed_swaps']}`.")
    lines.append(f"- Natural-only τ={args.threshold:.2f} hit pairs before/after: `{before_nat['hit_pairs']}` -> `{after_nat['hit_pairs']}`; row hits `{before_nat['row_hits']}` -> `{after_nat['row_hits']}`.")
    lines.append(f"- All-row τ={args.threshold:.2f} hit pairs before/after: `{before_all['hit_pairs']}` -> `{after_all['hit_pairs']}`; the dense qwen-pair block remains intentionally unchanged.")
    lines.append(f"- Word count: 10M exact `{c10['exact_10M']}` ({c10['words']}); 100M exact `{c100['exact_100M']}` ({c100['words']}).")
    lines.append(f"- 100M stream SHA prefix: `{sha100[:12]}`.")
    lines.append("")
    lines.append("## Pre-stated readout")
    lines.append("")
    lines.append("If this arm is trained, score it against CLEAN on the same DeBERTa seed43022 coordinate. The signs predicted by relation-typed composition are smaller versions of the designed exact-adjacency result: removing natural high-overlap adjacency should reduce exact-copy gain and improve true-source use on source-absent changed/substituted targets if baseline adjacency pays a recurrence cost. Because the refined natural dose is around 3.5k selected rows/hits per 10M, versus 33.3k designed compact relations per 10M, a null result would mostly bound natural-dose detectability rather than overturn the compact locality result.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for label,path in result["files"].items(): lines.append(f"- {label}: `{path}`")
    NOTE.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status":result["status"],"summary":rel(NOTE),"metadata":rel(META_OUT),"stream_100M":rel(OUT_100M),"selected_rows":result["selected_natural_rows"],"performed_swaps":result["performed_swaps"],"natural_hits_before":before_nat["hit_pairs"],"natural_hits_after":after_nat["hit_pairs"],"words_100M":c100["words"],"sha100_prefix":sha100[:12],"elapsed_sec":result["elapsed_sec"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
