#!/usr/bin/env python3
"""Analyze the frontier_consolidation high-anchor FineWeb simplification pilot.

The script reads the prompt slice made in research and Qwen outputs produced in
research, then estimates whether the generated second views are faithful enough to
justify building a matched FineWeb source+rewrite versus same-source repetition
pretraining contrast.  It is intentionally conservative: the output is evidence
for the next construction choice, not a proof that the generated text is true.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_PROMPTS = ROOT / "data" / "fineweb_high_anchor_slice" / "fineweb_high_anchor_simplification_prompts_pilot1024.jsonl"
DEFAULT_OUTPUTS = ROOT / "training" / "runs" / "high_anchor_fineweb_qwen_pilot1024_qbatch" / "outputs.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "high_anchor_fineweb_generation_analysis"
DEFAULT_NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/frontier_consolidation/high_anchor_fineweb_generation_analysis.md')

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?", re.I)
CAP_SEQ_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9.&'’\-]+(?:\s+|$)){1,7}")
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", "for", "with", "as", "by", "from", "at",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "he", "she", "they", "we",
    "you", "i", "his", "her", "their", "our", "your", "not", "no", "so", "than", "into", "about", "can", "could", "may", "might",
    "will", "would", "should", "has", "have", "had", "do", "does", "did", "who", "which", "what", "when", "where", "why", "how",
    "one", "two", "new", "old", "more", "most", "many", "much", "some", "such", "very", "only", "also", "between", "during", "after",
}
ENTITY_STOP = {
    "The", "A", "An", "This", "That", "These", "Those", "In", "On", "For", "At", "By", "From", "To", "And", "But", "Or", "If",
    "When", "While", "Because", "SOURCE SENTENCE", "FineWeb", "Sentence", "English",
}
BAD_OUTPUT_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"^\s*(sure|here('| i)s|certainly|of course)\b",
        r"as an ai", r"i (cannot|can't)", r"please provide", r"output only", r"rewrite the factual", r"source sentence",
        r"simplified sentence\s*:", r"the original sentence", r"not mentioned in the sentence", r"cannot determine",
        r"\[.*\]", r"^\s*[-*•]", r"\n\s*[-*•]",
    ]
]
SOURCE_RISK_PATTERNS = {
    "angle_heading": re.compile(r">>|<<"),
    "heading_dash_chain": re.compile(r"\s[-–]\s.*\s[-–]\s"),
    "deictic_time": re.compile(r"\b(this month|today|currently|recently)\b", re.I),
    "probability_hedge": re.compile(r"\b(probably|possibly|perhaps)\b", re.I),
    "long_parenthetical": re.compile(r"\([^)]{20,}\)"),
    "apostle_or_title_apposition": re.compile(r"\b(apostle of|known as|called the)\b", re.I),
}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def word_count(text: str) -> int:
    return len(tokens(text))


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if len(t) > 2 and t not in STOPWORDS and not NUM_RE.fullmatch(t)]


def normalize_num(s: str) -> str:
    return re.sub(r"\s+", "", s.lower().replace(",", ""))


def numbers(text: str) -> set[str]:
    return {normalize_num(m.group(0)) for m in NUM_RE.finditer(text or "")}


def normalized_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def sig_entity_tokens(text: str) -> set[str]:
    return {t for t in content_tokens(text) if len(t) > 2}


def capital_entities(text: str) -> set[str]:
    ents: set[str] = set()
    for m in CAP_SEQ_RE.finditer(text or ""):
        e = " ".join(m.group(0).split()).strip(" .,:;!?()[]{}\"'")
        if not e or e in ENTITY_STOP:
            continue
        if len(e) <= 2:
            continue
        ents.add(normalized_phrase(e))
    return ents


def source_entity_preserved(entity: str, output: str) -> bool:
    ent_norm = normalized_phrase(entity)
    out_norm = normalized_phrase(output)
    if not ent_norm:
        return True
    if ent_norm in out_norm:
        return True
    ent_toks = sig_entity_tokens(entity)
    if not ent_toks:
        return True
    out_toks = set(content_tokens(output))
    kept = len(ent_toks & out_toks)
    if len(ent_toks) <= 2:
        return kept == len(ent_toks)
    return kept / len(ent_toks) >= 0.75


def overlap(a: str, b: str) -> float:
    aa, bb = set(content_tokens(a)), set(content_tokens(b))
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def content_recall(source: str, output: str) -> float:
    src = set(content_tokens(source))
    out = set(content_tokens(output))
    if not src:
        return 1.0
    return len(src & out) / len(src)


def stats(xs: Iterable[float]) -> dict[str, Any]:
    vals = [float(x) for x in xs]
    if not vals:
        return {"n": 0}
    vals.sort()

    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = p * (len(vals) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)

    return {
        "n": len(vals), "min": vals[0], "p05": q(0.05), "mean": statistics.fmean(vals),
        "median": statistics.median(vals), "p95": q(0.95), "max": vals[-1], "sum": sum(vals),
    }


def clean_output(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'")


def output_for_row(outputs: list[dict[str, Any]], idx: int) -> str:
    if idx >= len(outputs):
        return ""
    row = outputs[idx]
    return str(row.get("output") or row.get("generated_text") or row.get("text") or row.get("completion") or "")


def structure_flags(raw_output: str, cleaned: str) -> list[str]:
    flags: list[str] = []
    if not cleaned:
        flags.append("empty_output")
        return flags
    if any(rx.search(raw_output or "") for rx in BAD_OUTPUT_PATTERNS):
        flags.append("bad_output_pattern")
    if "\n" in (raw_output or "").strip():
        flags.append("multi_line_output")
    if not (cleaned[0].isupper() or cleaned[0].isdigit() or cleaned[0] in "\"'"):
        flags.append("bad_start")
    if cleaned[-1] not in ".!?\"'":
        flags.append("bad_end")
    # A permissive single-sentence proxy.  Abbreviations can overcount, so this is a warning unless very large.
    terminal_marks = len(re.findall(r"[.!?](?:\s|$)", cleaned))
    if terminal_marks >= 3:
        flags.append("many_sentences")
    if re.search(r"\b(first|second|third|for example)\b", cleaned, re.I) and terminal_marks >= 2:
        flags.append("explanatory_expansion")
    return flags


def analyze_one(prompt: dict[str, Any], raw_output: str, idx: int) -> dict[str, Any]:
    source = str(prompt.get("source_text") or "")
    out = clean_output(raw_output)
    src_words = int(prompt.get("source_words") or word_count(source))
    out_words = word_count(out)
    ratio = out_words / max(1, src_words)
    src_nums = set(map(normalize_num, prompt.get("source_numbers") or [])) | numbers(source)
    out_nums = numbers(out)
    src_entities = [str(x) for x in (prompt.get("source_entities") or []) if str(x).strip()]
    preserved_entities = [e for e in src_entities if source_entity_preserved(e, out)]
    missing_entities = [e for e in src_entities if e not in preserved_entities]
    nrec = len(src_nums & out_nums) / max(1, len(src_nums)) if src_nums else 1.0
    erec = len(preserved_entities) / max(1, len(src_entities)) if src_entities else 1.0
    ov = overlap(source, out)
    crec = content_recall(source, out)
    source_risks = [name for name, rx in SOURCE_RISK_PATTERNS.items() if rx.search(source)]
    out_caps = capital_entities(out)
    src_caps = capital_entities(source)
    # Include provided source entity fragments to avoid penalizing proper names our regex misses.
    src_caps |= {normalized_phrase(e) for e in src_entities if normalized_phrase(e)}
    new_entity_like = sorted(e for e in out_caps if e not in src_caps and not any(e in s or s in e for s in src_caps))

    hard: list[str] = []
    soft: list[str] = []
    hard.extend(structure_flags(raw_output, out))
    if ratio < 0.35 or ratio > 1.45:
        hard.append(f"length_ratio_{ratio:.2f}")
    if nrec < 1.0:
        hard.append(f"number_recall_{nrec:.2f}")
    if out_nums - src_nums:
        hard.append(f"new_numbers_{len(out_nums - src_nums)}")
    if src_entities and erec < (1.0 if len(src_entities) <= 2 else 0.75):
        hard.append(f"entity_recall_{erec:.2f}")
    if len(new_entity_like) > 3:
        hard.append(f"new_entity_like_{len(new_entity_like)}")
    if ov < 0.10 and src_words >= 10:
        hard.append(f"low_overlap_{ov:.2f}")
    if crec < 0.28 and src_words >= 14:
        hard.append(f"low_content_recall_{crec:.2f}")
    if ov > 0.92 and ratio > 0.80 and src_words >= 14:
        soft.append("near_copy_view")
    if ratio > 1.05:
        soft.append("not_shorter_than_source")
    if source_risks:
        soft.extend(f"source_risk_{x}" for x in source_risks)

    return {
        "index": idx,
        "prompt_id": prompt.get("prompt_id") or prompt.get("id"),
        "sentence_id": prompt.get("sentence_id"),
        "doc_id": prompt.get("doc_id"),
        "domain_hits": prompt.get("domain_hits") or [],
        "source_words": src_words,
        "output_words": out_words,
        "pair_words": src_words + out_words,
        "length_ratio": ratio,
        "content_overlap": ov,
        "content_recall": crec,
        "source_numbers": sorted(src_nums),
        "output_numbers": sorted(out_nums),
        "number_recall": nrec,
        "source_entities": src_entities,
        "missing_entities": missing_entities,
        "entity_recall": erec,
        "new_entity_like": new_entity_like,
        "source_risks": source_risks,
        "hard_reasons": hard,
        "soft_flags": soft,
        "accepted_for_next_construction": not hard,
        "source_text": source,
        "rewrite_text": out,
        "raw_output": raw_output,
    }


def reason_prefix(reason: str) -> str:
    for pref in ["length_ratio", "number_recall", "new_numbers", "entity_recall", "new_entity_like", "low_overlap", "low_content_recall"]:
        if reason.startswith(pref):
            return pref
    return reason


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        domains = r.get("domain_hits") or ["no_domain"]
        for d in domains:
            by_domain[str(d)].append(r)
    accepted = [r for r in rows if r.get("accepted_for_next_construction")]
    reasons = collections.Counter(reason_prefix(x) for r in rows for x in r.get("hard_reasons", []))
    soft = collections.Counter(x for r in rows for x in r.get("soft_flags", []))
    risks = collections.Counter(x for r in rows for x in r.get("source_risks", []))

    def small(rs: list[dict[str, Any]]) -> dict[str, Any]:
        acc = [r for r in rs if r.get("accepted_for_next_construction")]
        return {
            "n": len(rs),
            "accepted": len(acc),
            "accepted_rate": len(acc) / max(1, len(rs)),
            "source_words": sum(int(r.get("source_words") or 0) for r in rs),
            "accepted_source_words": sum(int(r.get("source_words") or 0) for r in acc),
            "accepted_pair_words": sum(int(r.get("pair_words") or 0) for r in acc),
            "length_ratio_stats": stats([r.get("length_ratio") or 0.0 for r in rs]),
            "content_recall_stats": stats([r.get("content_recall") or 0.0 for r in rs]),
            "entity_recall_stats": stats([r.get("entity_recall") or 0.0 for r in rs]),
            "number_recall_stats": stats([r.get("number_recall") or 0.0 for r in rs]),
        }

    return {
        "overall": small(rows),
        "by_domain_hit": {k: small(v) for k, v in sorted(by_domain.items(), key=lambda kv: (-len(kv[1]), kv[0]))},
        "top_hard_reason_prefixes": reasons.most_common(30),
        "top_soft_flags": soft.most_common(30),
        "source_risk_counts": risks.most_common(30),
        "accepted_examples": accepted[:12],
        "rejected_examples": [r for r in rows if not r.get("accepted_for_next_construction")][:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--outputs", default=str(DEFAULT_OUTPUTS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    ap.add_argument("--sample-seed", type=int, default=829120)
    args = ap.parse_args()

    prompts_path = pathlib.Path(args.prompts)
    outputs_path = pathlib.Path(args.outputs)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(prompts_path)
    outputs = read_jsonl(outputs_path)
    if len(outputs) < len(prompts):
        raise RuntimeError(f"outputs {len(outputs)} shorter than prompts {len(prompts)}")
    rows = [analyze_one(pr, output_for_row(outputs, i), i) for i, pr in enumerate(prompts)]
    summary = summarize(rows)

    all_rows = out_dir / "fineweb_high_anchor_generation_rows.jsonl"
    accepted_path = out_dir / "fineweb_high_anchor_accepted_rewrites.jsonl"
    summary_path = out_dir / "fineweb_high_anchor_generation_summary.json"
    samples_path = out_dir / "fineweb_high_anchor_generation_review_samples.json"
    with all_rows.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with accepted_path.open("w", encoding="utf-8") as f:
        for r in rows:
            if r.get("accepted_for_next_construction"):
                f.write(json.dumps({
                    "prompt_id": r["prompt_id"],
                    "sentence_id": r["sentence_id"],
                    "doc_id": r["doc_id"],
                    "source_text": r["source_text"],
                    "rewrite_text": r["rewrite_text"],
                    "source_words": r["source_words"],
                    "rewrite_words": r["output_words"],
                    "pair_words": r["pair_words"],
                    "length_ratio": r["length_ratio"],
                    "domain_hits": r["domain_hits"],
                }, ensure_ascii=False) + "\n")
    rng = random.Random(args.sample_seed)
    accepted = [r for r in rows if r.get("accepted_for_next_construction")]
    rejected = [r for r in rows if not r.get("accepted_for_next_construction")]
    samples_payload = {
        "accepted_random": rng.sample(accepted, min(32, len(accepted))) if accepted else [],
        "rejected_random": rng.sample(rejected, min(32, len(rejected))) if rejected else [],
        "highest_new_entity_like": sorted(rows, key=lambda r: len(r.get("new_entity_like") or []), reverse=True)[:16],
        "lowest_content_recall": sorted(rows, key=lambda r: float(r.get("content_recall") or 0.0))[:16],
    }
    samples_path.write_text(json.dumps(samples_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    payload = {
        "status": "FINEWEB_HIGH_ANCHOR_GENERATION_ANALYZED",
        "scientific_purpose": "Estimate whether high-anchor FineWeb factual sentences can support faithful same-window simplified second views before any matched pretraining contrast is materialized.",
        "prompts": str(prompts_path),
        "outputs": str(outputs_path),
        "prompt_sha256": sha256_file(prompts_path),
        "outputs_sha256": sha256_file(outputs_path),
        "prompt_count": len(prompts),
        "output_count": len(outputs),
        "summary": summary,
        "files": {
            "all_rows": str(all_rows),
            "accepted_rewrites": str(accepted_path),
            "review_samples": str(samples_path),
            "note": str(args.note),
        },
        "interpretation_rules": {
            "strong_enough_for_factor_contrast": "Accepted rate above about 0.75 with high number/entity recall and few source-risk failures supports scaling generation for a matched source-breadth and view-structure contrast.",
            "needs_tighter_source_or_prompt": "Accepted rate below about 0.60, many added entities/numbers, or many source-risk failures means do not materialize a training corpus; tighten the source slice or generation prompt first.",
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note = pathlib.Path(args.note)
    note.parent.mkdir(parents=True, exist_ok=True)
    overall = summary["overall"]
    lines = ["# research high-anchor FineWeb generation analysis\n\n"]
    lines.append(f"Prompts: `{prompts_path}`\n\nOutputs: `{outputs_path}`\n\n")
    lines.append(f"Rows: {len(rows):,}; accepted: {overall['accepted']:,}; accepted rate: {overall['accepted_rate']:.3f}.\n\n")
    lines.append(f"Source words: {overall['source_words']:,}; accepted source words: {overall['accepted_source_words']:,}; accepted source+rewrite words: {overall['accepted_pair_words']:,}.\n\n")
    lines.append(f"Top hard reason prefixes: {summary['top_hard_reason_prefixes'][:12]}\n\n")
    lines.append(f"Top soft flags: {summary['top_soft_flags'][:12]}\n\n")
    lines.append("| domain | n | accepted | rate | accepted source words | accepted pair words |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for domain, s in list(summary["by_domain_hit"].items())[:20]:
        lines.append(f"| {domain} | {s['n']} | {s['accepted']} | {s['accepted_rate']:.3f} | {s['accepted_source_words']} | {s['accepted_pair_words']} |\n")
    lines.append(f"\nJSON: `{summary_path}`\n\nAccepted rewrite JSONL: `{accepted_path}`\n\nReview samples: `{samples_path}`\n")
    note.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "summary_json": str(summary_path),
        "accepted_jsonl": str(accepted_path),
        "note": str(note),
        "accepted_rate": overall["accepted_rate"],
        "accepted_source_words": overall["accepted_source_words"],
        "accepted_pair_words": overall["accepted_pair_words"],
        "top_hard_reasons": summary["top_hard_reason_prefixes"][:10],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
