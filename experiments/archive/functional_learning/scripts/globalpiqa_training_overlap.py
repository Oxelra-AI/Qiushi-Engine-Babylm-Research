#!/usr/bin/env python3
"""research: inspect whether fragile GlobalPIQA flips are directly visible in the dense training prefix.

The dense zero-shot/Reading surplus is largely carried by four shared GlobalPIQA
correcting flips and one shared loss.  This script asks a narrow evidential
question: are those items near-directly represented in the 80-update legal tail
prefix used by dense unchanged-Qwen focus, especially in Qwen focus rows?

It is not a semantic entailment detector.  It records exact normalized substring
matches and high lexical-overlap rows so later interpretation can distinguish a
possible memorization/benchmark-familiarity account from a broader competence
claim.  Absence of lexical matches is not proof of abstraction; presence of close
matches is a warning to interpret the tiny GlobalPIQA aggregate carefully.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import json
import math
import pathlib
import re
import time
from typing import Any, Dict, Iterable, List, Tuple


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
STUDY = ROOT / "experiments/archive/functional_learning"
TAIL = STUDY / "data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl"
FRAGILITY = STUDY / "data/dense_score_fragility_profile/dense_score_fragility_profile.json"
PROFILE = STUDY / "data/dense_input_label_profile/dense_input_label_profile.json"
OUT_DIR = STUDY / "data/globalpiqa_training_overlap"

STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "from", "by", "at", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "you", "your", "i", "we", "they", "their", "then", "when", "what", "which", "who", "how", "do", "does", "did", "will", "would", "should", "could", "can", "may", "might", "more", "most", "same", "not", "no", "yes", "all", "some", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "about", "after", "before", "into", "onto", "over", "under", "through", "than", "there", "here", "have", "has", "had", "get", "gets", "put", "place", "placed", "make", "makes", "use", "using", "up", "out", "off", "down", "first", "next", "best", "likely", "happens", "happen",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.I)


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def norm_text(s: Any) -> str:
    s = "" if s is None else str(s).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9°' /:-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: Any) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(str(s or ""))]


def content_tokens(s: Any) -> list[str]:
    return [t for t in tokens(s) if len(t) >= 4 and t not in STOP]


def read_prefix_rows(n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with TAIL.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if line.strip():
                r = json.loads(line)
                r["_prefix_index"] = i
                rows.append(r)
    return rows


def row_text_parts(row: dict[str, Any]) -> dict[str, str]:
    parts = {"whole_text": row.get("text") or ""}
    qsegs = row.get("qwen_pair_segments") or []
    if qsegs:
        parts["qwen_sources"] = " ".join(str(s.get("source_text") or "") for s in qsegs)
        parts["qwen_views"] = " ".join(str(s.get("view_text") or "") for s in qsegs)
    return parts


def phrase_hits(row_norm_parts: dict[str, str], phrases: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, phrase in phrases:
        np = norm_text(phrase)
        if len(np) < 8:
            continue
        for part_name, part_norm in row_norm_parts.items():
            if np and np in part_norm:
                out.append({"phrase_label": label, "phrase": phrase, "part": part_name})
    return out


def classify_terms(rec: dict[str, Any]) -> dict[str, list[str]]:
    fields = {
        "prompt": rec.get("prompt"),
        "target": rec.get("target"),
        "parent_pred": rec.get("parent_pred"),
        "dense_pred": rec.get("seed62064_pred"),
    }
    return {k: sorted(set(content_tokens(v))) for k, v in fields.items()}


def top_matches_for_flip(rows: list[dict[str, Any]], rec: dict[str, Any], top_k: int = 8) -> dict[str, Any]:
    query_text = " ".join(str(rec.get(k) or "") for k in ["prompt", "target", "parent_pred", "seed62064_pred", "seed62065_pred"])
    q_tokens = sorted(set(content_tokens(query_text)))
    q_set = set(q_tokens)
    phrases = [
        ("prompt", rec.get("prompt") or ""),
        ("target", rec.get("target") or ""),
        ("parent_pred", rec.get("parent_pred") or ""),
        ("dense_pred", rec.get("seed62064_pred") or ""),
    ]
    term_counts_by_source: collections.Counter[str] = collections.Counter()
    exact_rows: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    for row in rows:
        parts = row_text_parts(row)
        norm_parts = {k: norm_text(v) for k, v in parts.items()}
        hits = phrase_hits(norm_parts, phrases)
        row_toks = content_tokens(parts["whole_text"])
        row_set = set(row_toks)
        overlap = sorted(q_set & row_set)
        # Distinguish overlap in qwen sources/views when present.
        q_source_overlap = sorted(q_set & set(content_tokens(parts.get("qwen_sources", ""))))
        q_view_overlap = sorted(q_set & set(content_tokens(parts.get("qwen_views", ""))))
        if overlap or hits:
            term_counts_by_source[str(row.get("source"))] += len(overlap)
        if hits:
            exact_rows.append({
                "prefix_index": row.get("_prefix_index"),
                "source": row.get("source"),
                "example_id": row.get("example_id"),
                "hits": hits,
                "snippet": str(row.get("text") or "")[:600],
                "qwen_pair_ids": row.get("qwen_pair_ids"),
            })
        if overlap:
            score = len(overlap) / math.sqrt(max(1, len(row_set)))
            jacc = len(overlap) / len(q_set | row_set) if q_set | row_set else 0.0
            qwen_bonus = 0.15 if (q_source_overlap or q_view_overlap) else 0.0
            scored.append({
                "prefix_index": row.get("_prefix_index"),
                "source": row.get("source"),
                "example_id": row.get("example_id"),
                "score": score + qwen_bonus,
                "raw_overlap_count": len(overlap),
                "jaccard": jacc,
                "overlap_terms": overlap,
                "qwen_source_overlap_terms": q_source_overlap,
                "qwen_view_overlap_terms": q_view_overlap,
                "qwen_pair_ids": row.get("qwen_pair_ids"),
                "snippet": str(row.get("text") or "")[:750],
            })
    scored.sort(key=lambda x: (x["score"], x["raw_overlap_count"], x["jaccard"]), reverse=True)
    return {
        "query_tokens": q_tokens,
        "field_tokens": classify_terms(rec),
        "exact_phrase_hit_count": len(exact_rows),
        "exact_phrase_hits": exact_rows[:top_k],
        "term_overlap_source_weighted_counts": dict(term_counts_by_source.most_common()),
        "top_lexical_overlap_rows": scored[:top_k],
    }


def make_md(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# research GlobalPIQA flip overlap with dense training prefix")
    lines.append("")
    lines.append(f"Scanned first `{result['prefix_rows_scanned']}` rows of `{result['tail_path']}`, matching the 80-update dense-focus legal prefix. This is lexical/provenance evidence only; it does not prove semantic learning or its absence.")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Shared GlobalPIQA dense gains inspected: `{len(result['flips']['shared_dense_gains'])}`.")
    lines.append(f"- Shared GlobalPIQA dense losses inspected: `{len(result['flips']['shared_dense_losses'])}`.")
    total_exact = sum(v['exact_phrase_hit_count'] for v in result['overlap_by_flip'].values())
    lines.append(f"- Exact normalized prompt/target/prediction phrase hits in the 80-update prefix: `{total_exact}` across all inspected flips.")
    lines.append("- High lexical-overlap rows are listed below. Treat them as possible thematic exposure, not as confirmed benchmark leakage or causal explanation.")
    lines.append("")
    for group in ["shared_dense_gains", "shared_dense_losses"]:
        lines.append(f"## {group}")
        for rec in result['flips'][group]:
            key = f"{rec['column']}::{rec['example_id']}"
            ov = result['overlap_by_flip'][key]
            lines.append("")
            lines.append(f"### `{key}` pattern `{rec['pattern_parent_seed64_seed65']}`")
            lines.append(f"Prompt: {rec['prompt']}")
            lines.append(f"Target: {rec['target']}")
            lines.append(f"Parent pred: {rec['parent_pred']}")
            lines.append(f"Dense pred: {rec['seed62064_pred']}")
            lines.append(f"Query terms: `{', '.join(ov['query_tokens'])}`")
            lines.append(f"Exact phrase hits: `{ov['exact_phrase_hit_count']}`")
            if ov['exact_phrase_hits']:
                for h in ov['exact_phrase_hits']:
                    lines.append(f"- exact row `{h['prefix_index']}` source `{h['source']}` hits `{h['hits']}` snippet={json.dumps(h['snippet'], ensure_ascii=False)}")
            lines.append("Top lexical-overlap rows:")
            for row in ov['top_lexical_overlap_rows'][:5]:
                qnote = ""
                if row.get('qwen_source_overlap_terms') or row.get('qwen_view_overlap_terms'):
                    qnote = f" qwen_source_terms={row.get('qwen_source_overlap_terms')} qwen_view_terms={row.get('qwen_view_overlap_terms')}"
                lines.append(f"- row `{row['prefix_index']}` source `{row['source']}` overlap `{row['raw_overlap_count']}` score `{row['score']:.4f}` terms={row['overlap_terms']}{qnote} snippet={json.dumps(row['snippet'][:360], ensure_ascii=False)}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("The GlobalPIQA contribution remains a tiny, same-example official signal. Direct exact phrase matches would strengthen a benchmark-familiarity concern; purely thematic overlap is weaker and should mainly motivate not letting the GlobalPIQA scalar carry the learning-principle argument. The larger scientific evidence for dense focus remains Entity depth movement and controlled source-altered common-target behavior, while BLiMP/Supplement/EWoK costs define the repair target.")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    prefix_rows = int(profile["prefix_info"]["prefix_rows"])
    frag = json.loads(FRAGILITY.read_text(encoding="utf-8"))
    flips = {
        "shared_dense_gains": frag["globalpiqa_flips"]["shared_dense_gains"],
        "shared_dense_losses": frag["globalpiqa_flips"]["shared_dense_losses"],
    }
    rows = read_prefix_rows(prefix_rows)
    overlap: dict[str, Any] = {}
    for rec in flips["shared_dense_gains"] + flips["shared_dense_losses"]:
        key = f"{rec['column']}::{rec['example_id']}"
        overlap[key] = top_matches_for_flip(rows, rec)
    result: dict[str, Any] = {
        "status": "GLOBALPIQA_TRAINING_OVERLAP",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tail_path": rel(TAIL),
        "prefix_rows_scanned": prefix_rows,
        "prefix_words": profile["prefix_info"].get("prefix_words"),
        "flips": flips,
        "overlap_by_flip": overlap,
        "source_count_in_prefix": dict(collections.Counter(str(r.get("source")) for r in rows).most_common()),
        "interpretation": "Lexical exact/overlap scan of GlobalPIQA flips against dense 80-update training prefix; not semantic proof.",
    }
    out_json = OUT_DIR / "globalpiqa_training_overlap.json"
    out_md = OUT_DIR / "globalpiqa_training_overlap.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(make_md(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "prefix_rows_scanned": prefix_rows}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
