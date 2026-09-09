#!/usr/bin/env python3
"""Analyze a FineWeb sentence simplification faithfulness slice after Qwen generation.

Inputs are the prompts prepared by prepare_fineweb_rewrite_faithfulness_slice.py
and the recorded generation outputs.jsonl.  The analysis is conservative and
transparent; it is meant to decide whether a larger source-by-rewrite corpus is worth
materializing, not to certify truth.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
DEFAULT_PROMPTS = ROOT / "training" / "data" / "fineweb_rewrite_faithfulness_slice" / "fineweb_rewrite_faithfulness_prompts.jsonl"
DEFAULT_OUTPUTS = ROOT / "training" / "runs" / "fineweb_rewrite_faithfulness_slice_qwen" / "outputs.jsonl"
DEFAULT_OUT = ROOT / "training" / "data" / "fineweb_rewrite_faithfulness_slice" / "fineweb_rewrite_faithfulness_analysis.json"
DEFAULT_NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/representation_and_objectives/11_fineweb_rewrite_faithfulness_analysis.md')

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?", re.I)
CAP_SEQ_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9.&'’\-]+(?:\s+|$)){1,5}")
BAD_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"^\s*(sure|here('| i)s|certainly|of course)\b",
        r"as an ai", r"i (cannot|can't)", r"please provide", r"output only",
        r"simplified sentence\s*:", r"the original sentence", r"not mentioned in the sentence",
        r"not provided", r"cannot determine", r"\[.*\]", r"\(.*example.*\)",
    ]
]
STOP_ENTS = {
    "The", "A", "An", "This", "That", "These", "Those", "In", "On", "For", "At", "By", "From", "To", "And", "But", "Or", "If", "When", "While", "Because",
    "Sentence", "FineWeb", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", "for", "with", "as", "by", "from", "at", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "he", "she", "they", "we", "you", "i", "his", "her", "their", "our", "your", "not", "no", "so", "than", "into", "about", "can", "could", "may", "might", "will", "would", "should", "has", "have", "had", "do", "does", "did", "who", "which", "what", "when", "where", "why", "how",
}


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_output(text: str) -> str:
    text = (text or "").strip()
    # Keep first line if the model added extra commentary; newline remains a reason.
    text = re.sub(r"\s+", " ", text)
    return text.strip(' "')


def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def word_count(text: str) -> int:
    return len(tokens(text))


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if len(t) > 2 and t not in STOPWORDS and not NUM_RE.fullmatch(t)}


def numbers(text: str) -> set[str]:
    return {m.group(0).lower().replace(",", "") for m in NUM_RE.finditer(text)}


def entities(text: str) -> set[str]:
    ents: set[str] = set()
    for m in CAP_SEQ_RE.finditer(text):
        e = " ".join(m.group(0).split()).strip(" .,:;!?()[]{}\"'")
        if not e:
            continue
        if e in STOP_ENTS:
            continue
        # Drop single capitalized sentence-initial common words by requiring either
        # multi-token or not present in a small stop list.
        ents.add(e.lower())
    return ents


def overlap(a: str, b: str) -> float:
    aa, bb = content_tokens(a), content_tokens(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def starts_complete(text: str) -> bool:
    return bool(text) and (text[0].isupper() or text[0].isdigit() or text[0] in '"\'')


def ends_complete(text: str) -> bool:
    return bool(text) and text[-1] in ".!?\"'"


def analyze_row(prompt: dict[str, Any], output: str) -> dict[str, Any]:
    source = prompt.get("source_text") or ""
    out = clean_output(output)
    src_words = word_count(source)
    out_words = word_count(out)
    src_nums, out_nums = numbers(source), numbers(out)
    src_ents, out_ents = entities(source), entities(out)
    nrec = len(src_nums & out_nums) / max(1, len(src_nums)) if src_nums else 1.0
    erec = len(src_ents & out_ents) / max(1, len(src_ents)) if src_ents else 1.0
    new_nums = sorted(out_nums - src_nums)
    new_ents = sorted(out_ents - src_ents)
    reasons: list[str] = []
    if not out:
        reasons.append("empty_output")
    if any(rx.search(output or "") for rx in BAD_PATTERNS):
        reasons.append("bad_pattern")
    if not starts_complete(out) or not ends_complete(out):
        reasons.append("fragment_output")
    ratio = out_words / max(1, src_words)
    if ratio < 0.35 or ratio > 1.60:
        reasons.append(f"length_ratio_{ratio:.2f}")
    if nrec < 1.0:
        reasons.append(f"number_recall_{nrec:.2f}")
    if len(new_nums) > 0:
        reasons.append(f"new_numbers_{len(new_nums)}")
    if src_ents and erec < (1.0 if len(src_ents) <= 2 else 0.75):
        reasons.append(f"entity_recall_{erec:.2f}")
    if len(new_ents) > 3:
        reasons.append(f"new_entities_{len(new_ents)}")
    ov = overlap(source, out)
    if ov < 0.12 and src_words >= 10:
        reasons.append(f"low_overlap_{ov:.2f}")
    return {
        "prompt_id": prompt.get("id") or prompt.get("prompt_id"),
        "slice_index": prompt.get("slice_index"),
        "selection_tier": prompt.get("selection_tier"),
        "sentence_id": prompt.get("sentence_id"),
        "doc_id": prompt.get("doc_id"),
        "source_words": src_words,
        "output_words": out_words,
        "length_ratio": ratio,
        "source_entities": sorted(src_ents),
        "output_entities": sorted(out_ents),
        "entity_recall": erec,
        "new_entities": new_ents,
        "source_numbers": sorted(src_nums),
        "output_numbers": sorted(out_nums),
        "number_recall": nrec,
        "new_numbers": new_nums,
        "content_overlap": ov,
        "accepted_for_materialization_probe": not reasons,
        "rejection_reasons": reasons,
        "source_text": source,
        "output": out,
    }


def stats(xs: Iterable[float]) -> dict[str, Any]:
    vals = [float(x) for x in xs]
    if not vals:
        return {"n": 0}
    vals.sort()
    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        i = p * (len(vals) - 1)
        lo = math.floor(i)
        hi = math.ceil(i)
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - i) + vals[hi] * (i - lo)
    return {"n": len(vals), "min": vals[0], "p05": q(0.05), "mean": statistics.fmean(vals), "median": statistics.median(vals), "p95": q(0.95), "max": vals[-1]}


def prefix_reason(reason: str) -> str:
    for pref in ["length_ratio", "number_recall", "new_numbers", "entity_recall", "new_entities", "low_overlap"]:
        if reason.startswith(pref):
            return pref
    return reason


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tier: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_tier[str(r.get("selection_tier"))].append(r)
    out: dict[str, Any] = {}
    for key, rs in [("overall", rows)] + sorted(by_tier.items()):
        accepted = [r for r in rs if r.get("accepted_for_materialization_probe")]
        reasons = collections.Counter(prefix_reason(reason) for r in rs for reason in r.get("rejection_reasons", []))
        out[key] = {
            "n": len(rs),
            "accepted": len(accepted),
            "accepted_rate": len(accepted) / max(1, len(rs)),
            "source_words": sum(int(r.get("source_words") or 0) for r in rs),
            "accepted_source_words": sum(int(r.get("source_words") or 0) for r in accepted),
            "output_words_stats": stats([r.get("output_words") or 0 for r in rs]),
            "length_ratio_stats": stats([r.get("length_ratio") or 0 for r in rs]),
            "entity_recall_stats": stats([r.get("entity_recall") or 0 for r in rs]),
            "number_recall_stats": stats([r.get("number_recall") or 0 for r in rs]),
            "content_overlap_stats": stats([r.get("content_overlap") or 0 for r in rs]),
            "top_rejection_reasons": reasons.most_common(20),
            "examples_rejected": [r for r in rs if not r.get("accepted_for_materialization_probe")][:8],
            "examples_accepted": accepted[:8],
        }
    return out


def output_for_index(outputs: list[dict[str, Any]], i: int) -> str:
    if i >= len(outputs):
        return ""
    row = outputs[i]
    return str(row.get("output") or row.get("generated_text") or row.get("text") or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--outputs", default=str(DEFAULT_OUTPUTS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    args = ap.parse_args()

    prompts = load_jsonl(pathlib.Path(args.prompts))
    outputs = load_jsonl(pathlib.Path(args.outputs))
    if len(outputs) < len(prompts):
        raise RuntimeError(f"outputs {len(outputs)} shorter than prompts {len(prompts)}")
    rows = [analyze_row(pr, output_for_index(outputs, i)) for i, pr in enumerate(prompts)]
    summary = summarize(rows)
    payload = {
        "status": "FINEWEB_REWRITE_FAITHFULNESS_ANALYZED",
        "prompts": args.prompts,
        "outputs": args.outputs,
        "prompt_count": len(prompts),
        "output_count": len(outputs),
        "summary": summary,
        "interpretation": {
            "supports_full_generation": "High acceptance in balanced_source_by_rewrite, with number/entity preservation and few new facts, supports scaling to a matched FineWeb source+rewrite contrast.",
            "redirects_or_tightens": "Low acceptance or many entity/number additions means do not build a full corpus from this source screen; tighten source filtering or prompts before any H100 training.",
        },
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note = pathlib.Path(args.note)
    note.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# research FineWeb rewrite faithfulness analysis\n\n"]
    lines.append(f"Prompts: `{args.prompts}`\n\nOutputs: `{args.outputs}`\n\nJSON: `{out_path}`\n\n")
    lines.append("| tier | n | accepted | accepted rate | source words | accepted source words | top reasons |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---|\n")
    for tier, s in summary.items():
        lines.append("| {} | {} | {} | {:.3f} | {} | {} | {} |\n".format(
            tier, s["n"], s["accepted"], s["accepted_rate"], s["source_words"], s["accepted_source_words"], s["top_rejection_reasons"][:5]
        ))
    note.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_path), "note": str(note), "overall_acceptance": summary.get("overall", {}).get("accepted_rate")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
