#!/usr/bin/env python3
"""Analyze FineWeb generated second views with BabyLM-style word accounting.

This is a repaired analyzer for the research density experiments.  research's local
faithfulness analyzer was useful, but its saved `source_words`/`rewrite_words`
were regex-token counts whereas BabyLM corpus accounting uses whitespace words.
This script keeps the same conservative semantic checks while writing whitespace
counts as the canonical counts used by corpus materializers.  Lexical-token counts
are retained only as auxiliary analysis fields.
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

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?", re.I)
CAP_SEQ_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9.&'’\-]+(?:\s+|$)){1,7}")
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", "for", "with", "as", "by", "from", "at",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "he", "she", "they", "we",
    "you", "i", "his", "her", "their", "our", "your", "not", "no", "so", "than", "into", "about", "can", "could", "may", "might",
    "will", "would", "should", "has", "have", "had", "do", "does", "did", "who", "which", "what", "when", "where", "why", "how",
    "one", "two", "new", "old", "more", "most", "many", "much", "some", "such", "very", "only", "also", "between", "during", "after",
    "study", "studies", "research", "show", "shows", "using", "used", "use", "made", "make", "called", "known",
}
ENTITY_STOP = {
    "The", "A", "An", "This", "That", "These", "Those", "In", "On", "For", "At", "By", "From", "To", "And", "But", "Or", "If",
    "When", "While", "Because", "SOURCE SENTENCE", "FineWeb", "Sentence", "English", "Study", "Research", "Both", "Although",
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
    "web_or_reference_phrase": re.compile(r"\b(referenced documents|purchase separately|click here|read more|full report|summary)\b", re.I),
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


def lexical_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def lexical_word_count(text: str) -> int:
    return len(lexical_tokens(text))


def whitespace_count(text: str) -> int:
    return len(" ".join((text or "").split()).split())


def content_tokens(text: str) -> list[str]:
    return [t for t in lexical_tokens(text) if len(t) > 2 and t not in STOPWORDS and not NUM_RE.fullmatch(t)]


def normalize_num(s: str) -> str:
    s = str(s).lower().replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("percent", "%")
    return s


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

    return {"n": len(vals), "min": vals[0], "p05": q(0.05), "mean": statistics.fmean(vals),
            "median": statistics.median(vals), "p95": q(0.95), "max": vals[-1], "sum": sum(vals)}


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
    terminal_marks = len(re.findall(r"[.!?](?:\s|$)", cleaned))
    if terminal_marks >= 3:
        flags.append("many_sentences")
    if re.search(r"\b(first|second|third|for example)\b", cleaned, re.I) and terminal_marks >= 2:
        flags.append("explanatory_expansion")
    return flags


def analyze_one(prompt: dict[str, Any], raw_output: str, idx: int) -> dict[str, Any]:
    source = str(prompt.get("source_text") or "")
    source = " ".join(source.split())
    out = clean_output(raw_output)
    src_words_ws = int(prompt.get("source_words") or whitespace_count(source))
    # Preserve BabyLM-style count even if prompt metadata used whitespace count.
    if src_words_ws != whitespace_count(source):
        src_words_ws = whitespace_count(source)
    out_words_ws = whitespace_count(out)
    src_words_lex = lexical_word_count(source)
    out_words_lex = lexical_word_count(out)
    ratio_ws = out_words_ws / max(1, src_words_ws)
    ratio_lex = out_words_lex / max(1, src_words_lex)
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
    src_caps |= {normalized_phrase(e) for e in src_entities if normalized_phrase(e)}
    new_entity_like = sorted(e for e in out_caps if e not in src_caps and not any(e in s or s in e for s in src_caps))

    hard: list[str] = []
    soft: list[str] = []
    hard.extend(structure_flags(raw_output, out))
    if ratio_ws < 0.35 or ratio_ws > 1.45:
        hard.append(f"length_ratio_{ratio_ws:.2f}")
    if nrec < 1.0:
        hard.append(f"number_recall_{nrec:.2f}")
    if out_nums - src_nums:
        hard.append(f"new_numbers_{len(out_nums - src_nums)}")
    if src_entities and erec < (1.0 if len(src_entities) <= 2 else 0.75):
        hard.append(f"entity_recall_{erec:.2f}")
    if len(new_entity_like) > 3:
        hard.append(f"new_entity_like_{len(new_entity_like)}")
    if ov < 0.10 and src_words_ws >= 10:
        hard.append(f"low_overlap_{ov:.2f}")
    if crec < 0.28 and src_words_ws >= 14:
        hard.append(f"low_content_recall_{crec:.2f}")
    if ov > 0.92 and ratio_ws > 0.80 and src_words_ws >= 14:
        soft.append("near_copy_view")
    if ratio_ws > 1.05:
        soft.append("not_shorter_than_source")
    if source_risks:
        soft.extend(f"source_risk_{x}" for x in source_risks)

    return {
        "index": idx,
        "prompt_id": prompt.get("prompt_id") or prompt.get("id"),
        "sentence_id": prompt.get("sentence_id"),
        "doc_id": prompt.get("doc_id"),
        "domain_hits": prompt.get("domain_hits") or [],
        "source_words": src_words_ws,
        "rewrite_words": out_words_ws,
        "output_words": out_words_ws,
        "pair_words": src_words_ws + out_words_ws,
        "source_words_lexical": src_words_lex,
        "rewrite_words_lexical": out_words_lex,
        "pair_words_lexical": src_words_lex + out_words_lex,
        "length_ratio": ratio_ws,
        "length_ratio_lexical": ratio_lex,
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
            "source_words_whitespace": sum(int(r.get("source_words") or 0) for r in rs),
            "accepted_source_words_whitespace": sum(int(r.get("source_words") or 0) for r in acc),
            "accepted_rewrite_words_whitespace": sum(int(r.get("rewrite_words") or 0) for r in acc),
            "accepted_pair_words_whitespace": sum(int(r.get("pair_words") or 0) for r in acc),
            "weighted_rewrite_to_source_ratio_whitespace": (
                sum(int(r.get("rewrite_words") or 0) for r in acc) / max(1, sum(int(r.get("source_words") or 0) for r in acc))
            ),
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
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--label", default="fineweb_generation_density")
    ap.add_argument("--sample-seed", type=int, default=829130)
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

    all_rows = out_dir / f"{args.label}_rows.jsonl"
    accepted_path = out_dir / f"{args.label}_accepted_rewrites.jsonl"
    summary_path = out_dir / f"{args.label}_summary.json"
    samples_path = out_dir / f"{args.label}_review_samples.json"
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
                    "rewrite_words": r["rewrite_words"],
                    "pair_words": r["pair_words"],
                    "length_ratio": r["length_ratio"],
                    "content_recall": r["content_recall"],
                    "content_overlap": r["content_overlap"],
                    "entity_recall": r["entity_recall"],
                    "number_recall": r["number_recall"],
                    "soft_flags": r["soft_flags"],
                    "source_risks": r["source_risks"],
                    "domain_hits": r["domain_hits"],
                    "accepted_for_next_construction": True,
                }, ensure_ascii=False) + "\n")
    rng = random.Random(args.sample_seed)
    accepted = [r for r in rows if r.get("accepted_for_next_construction")]
    rejected = [r for r in rows if not r.get("accepted_for_next_construction")]
    samples = {
        "accepted_random": rng.sample(accepted, min(24, len(accepted))) if accepted else [],
        "rejected_random": rng.sample(rejected, min(24, len(rejected))) if rejected else [],
        "low_content_accepted": sorted(accepted, key=lambda r: (r.get("content_recall") or 0.0, r.get("entity_recall") or 0.0))[:24],
        "near_copy_accepted": [r for r in accepted if "near_copy_view" in (r.get("soft_flags") or [])][:24],
    }
    payload = {
        "status": "GENERATION_DENSITY_ANALYZED",
        "label": args.label,
        "prompts": str(prompts_path),
        "prompts_sha256": sha256_file(prompts_path),
        "outputs": str(outputs_path),
        "outputs_sha256": sha256_file(outputs_path),
        "word_accounting": "Canonical source_words/rewrite_words/pair_words in rows and accepted JSONL are whitespace counts, matching BabyLM corpus accounting; lexical counts are auxiliary only.",
        "summary": summary,
        "files": {
            "all_rows": str(all_rows),
            "accepted_rewrites": str(accepted_path),
            "summary_json": str(summary_path),
            "review_samples": str(samples_path),
            "note": str(args.note),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note_path = pathlib.Path(args.note)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    overall = summary["overall"]
    note_path.write_text(
        f"# research generation density analysis: {args.label}\n\n"
        f"Accepted {overall['accepted']:,}/{overall['n']:,} rows ({overall['accepted_rate']:.3f}). "
        f"Accepted source words {overall['accepted_source_words_whitespace']:,}, rewrite words {overall['accepted_rewrite_words_whitespace']:,}, "
        f"weighted rewrite/source ratio {overall['weighted_rewrite_to_source_ratio_whitespace']:.3f}.\n\n"
        "Counts in the accepted JSONL are whitespace words for BabyLM-style corpus accounting.\n\n"
        f"Summary JSON: `{summary_path}`\n\nReview samples: `{samples_path}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "label": args.label,
        "summary_json": str(summary_path),
        "accepted": overall["accepted"],
        "accepted_rate": overall["accepted_rate"],
        "accepted_pair_words_whitespace": overall["accepted_pair_words_whitespace"],
        "weighted_rewrite_to_source_ratio_whitespace": overall["weighted_rewrite_to_source_ratio_whitespace"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
