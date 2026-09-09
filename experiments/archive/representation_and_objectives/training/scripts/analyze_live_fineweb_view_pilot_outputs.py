#!/usr/bin/env python3
"""Analyze generated outputs for the research live-FineWeb view pilot.

This analyzer is prepared before generation so the acceptance definition is fixed.
It reads the research prompts and a future generated-output file,
then measures whether generated views are faithful, substantive, and usable for a
controlled C_view arm.  It does not call a model, train, or evaluate BabyLM.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
DEFAULT_PROMPTS = ROOT / "training/data/live_fineweb_view_pilot/live_fineweb_view_pilot_prompts_512.jsonl"
DEFAULT_OUTPUTS = ROOT / "training/runs/live_fineweb_view_pilot_qwen/outputs.jsonl"
DEFAULT_OUT = ROOT / "training/data/live_fineweb_view_pilot/live_fineweb_view_pilot_analysis.json"
DEFAULT_NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/21_live_fineweb_view_pilot_analysis.md')

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?", re.I)
CAP_SEQ_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9.&'’\-]+(?:\s+|$)){1,6}")
BAD_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"^\s*(sure|here('| i)s|certainly|of course)\b", r"as an ai", r"i (cannot|can't)", r"please provide",
        r"the original sentence", r"not mentioned", r"not provided", r"cannot determine", r"output only", r"rewritten sentence\s*:",
        r"\[.*\]", r"\(.*example.*\)",
    ]
]
STOP_ENTS = {
    "The", "A", "An", "This", "That", "These", "Those", "In", "On", "For", "At", "By", "From", "To", "And", "But", "Or", "If", "When", "While", "Because", "Sentence", "FineWeb",
}
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", "for", "with", "as", "by", "from", "at", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "he", "she", "they", "we", "you", "i", "his", "her", "their", "our", "your", "not", "no", "so", "than", "into", "about", "can", "could", "may", "might", "will", "would", "should", "has", "have", "had", "do", "does", "did", "who", "which", "what", "when", "where", "why", "how", "one", "single", "sentence",
}
RELATION_TERMS = [
    "because", "caused", "cause", "due", "led", "resulted", "therefore", "prevents", "reduces", "increases", "enables", "allows", "became", "created", "formed", "developed", "founded", "located", "contains", "includes", "requires", "designed", "invented", "means", "refers", "consists", "belongs", "crosses", "flows", "borders", "absorbs", "emits", "supplies", "produced", "converted",
]


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_output(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower().startswith("rewritten sentence:"):
        text = text.split(":", 1)[1].strip()
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
        if not e or e in STOP_ENTS:
            continue
        ents.add(e.lower())
    return ents


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def relation_recall(source: str, out: str) -> float:
    src = {t for t in RELATION_TERMS if re.search(r"\b" + re.escape(t) + r"\b", source, re.I)}
    if not src:
        return 1.0
    got = {t for t in src if re.search(r"\b" + re.escape(t) + r"\b", out, re.I)}
    return len(got) / len(src)


def output_for_index(outputs: list[dict[str, Any]], i: int) -> str:
    if i >= len(outputs):
        return ""
    row = outputs[i]
    return str(row.get("output") or row.get("generated_text") or row.get("text") or "")


def analyze_one(prompt: dict[str, Any], raw_output: str) -> dict[str, Any]:
    source = str(prompt.get("source_text") or "")
    out = clean_output(raw_output)
    src_words = word_count(source)
    out_words = word_count(out)
    src_nums, out_nums = numbers(source), numbers(out)
    src_ents, out_ents = entities(source), entities(out)
    src_tok, out_tok = content_tokens(source), content_tokens(out)
    num_recall = len(src_nums & out_nums) / max(1, len(src_nums)) if src_nums else 1.0
    ent_recall = len(src_ents & out_ents) / max(1, len(src_ents)) if src_ents else 1.0
    overlap = jaccard(src_tok, out_tok)
    seq = SequenceMatcher(None, " ".join(tokens(source)), " ".join(tokens(out))).ratio() if out else 0.0
    rel_recall = relation_recall(source, out)
    new_nums = sorted(out_nums - src_nums)
    new_ents = sorted(out_ents - src_ents)
    reasons: list[str] = []
    if not out:
        reasons.append("empty_output")
    if any(rx.search(raw_output or "") for rx in BAD_PATTERNS):
        reasons.append("bad_pattern")
    if not out or out[0:1] and not (out[0].isupper() or out[0].isdigit() or out[0] in '"\''):
        reasons.append("fragment_start")
    if not out or out[-1] not in ".!?\"'":
        reasons.append("fragment_end")
    ratio = out_words / max(1, src_words)
    if ratio < 0.35 or ratio > 1.65:
        reasons.append(f"length_ratio_{ratio:.2f}")
    if num_recall < 1.0:
        reasons.append(f"number_recall_{num_recall:.2f}")
    if new_nums:
        reasons.append(f"new_numbers_{len(new_nums)}")
    if src_ents and ent_recall < (1.0 if len(src_ents) <= 2 else 0.78):
        reasons.append(f"entity_recall_{ent_recall:.2f}")
    if len(new_ents) > 2:
        reasons.append(f"new_entities_{len(new_ents)}")
    if overlap < 0.16 and src_words >= 10:
        reasons.append(f"low_overlap_{overlap:.2f}")
    if rel_recall < 0.50:
        reasons.append(f"relation_recall_{rel_recall:.2f}")
    exact_or_near_copy = seq >= 0.965 or " ".join(tokens(source)) == " ".join(tokens(out))
    too_close = seq >= 0.90
    if exact_or_near_copy:
        reasons.append("exact_or_near_copy")
    accepted_faithful = not [r for r in reasons if r not in {"exact_or_near_copy"}]
    accepted_substantive = accepted_faithful and not too_close and 0.40 <= ratio <= 1.25
    return {
        "id": prompt.get("id"),
        "pair_id": prompt.get("pair_id"),
        "variant": prompt.get("variant"),
        "source_focus": prompt.get("source_focus"),
        "source_class": prompt.get("source_class"),
        "source_words": src_words,
        "target_view_words_simulated": prompt.get("target_view_words_simulated"),
        "output_words": out_words,
        "length_ratio": ratio,
        "content_overlap": overlap,
        "sequence_similarity": seq,
        "relation_recall": rel_recall,
        "entity_recall": ent_recall,
        "number_recall": num_recall,
        "new_entities": new_ents,
        "new_numbers": new_nums,
        "accepted_faithful": accepted_faithful,
        "accepted_substantive_view": accepted_substantive,
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
        idx = p * (len(vals) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)
    return {"n": len(vals), "min": vals[0], "p05": q(0.05), "mean": statistics.fmean(vals), "median": statistics.median(vals), "p95": q(0.95), "max": vals[-1]}


def reason_prefix(reason: str) -> str:
    for pref in ["length_ratio", "number_recall", "new_numbers", "entity_recall", "new_entities", "low_overlap", "relation_recall"]:
        if reason.startswith(pref):
            return pref
    return reason


def summarize(rows: list[dict[str, Any]], label_fn) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(label_fn(r))].append(r)
    out: dict[str, Any] = {}
    for label, rs in sorted(groups.items()):
        faithful = [r for r in rs if r.get("accepted_faithful")]
        substantive = [r for r in rs if r.get("accepted_substantive_view")]
        reasons = Counter(reason_prefix(reason) for r in rs for reason in r.get("rejection_reasons", []))
        out[label] = {
            "n": len(rs),
            "faithful": len(faithful),
            "faithful_rate": len(faithful) / max(1, len(rs)),
            "substantive": len(substantive),
            "substantive_rate": len(substantive) / max(1, len(rs)),
            "source_words": sum(int(r.get("source_words") or 0) for r in rs),
            "substantive_source_words": sum(int(r.get("source_words") or 0) for r in substantive),
            "output_words_stats": stats([r.get("output_words") or 0 for r in rs]),
            "length_ratio_stats": stats([r.get("length_ratio") or 0 for r in rs]),
            "content_overlap_stats": stats([r.get("content_overlap") or 0 for r in rs]),
            "sequence_similarity_stats": stats([r.get("sequence_similarity") or 0 for r in rs]),
            "entity_recall_stats": stats([r.get("entity_recall") or 0 for r in rs]),
            "number_recall_stats": stats([r.get("number_recall") or 0 for r in rs]),
            "top_rejection_reasons": reasons.most_common(20),
        }
    return out


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
    rows = [analyze_one(pr, output_for_index(outputs, i)) for i, pr in enumerate(prompts)]
    payload = {
        "status": "LIVE_FINEWEB_VIEW_PILOT_ANALYZED",
        "prompts": args.prompts,
        "outputs": args.outputs,
        "prompt_count": len(prompts),
        "output_count": len(outputs),
        "summary_by_variant": summarize(rows, lambda r: r.get("variant")),
        "summary_by_focus": summarize(rows, lambda r: r.get("source_focus")),
        "summary_by_source_class": summarize(rows, lambda r: r.get("source_class")),
        "summary_overall": summarize(rows, lambda r: "overall").get("overall"),
        "rows": rows,
        "interpretation": {
            "supports_source_view_pilot": "A high substantive-view rate with high entity/number/relation preservation supports materializing a compact C_view/B_repeat/B_breadth/A_natural family.",
            "repairs_or_redirects": "Low substantive-view rate, exact-copy dominance, or many new facts means change prompt/source mix before training; do not count nominal accepted outputs as second views.",
        },
    }
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note = pathlib.Path(args.note); note.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# research live FineWeb view pilot analysis\n\n"]
    lines.append(f"Prompts: `{args.prompts}`\n\nOutputs: `{args.outputs}`\n\nJSON: `{out}`\n\n")
    lines.append("## Overall\n\n```json\n" + json.dumps(payload["summary_overall"], indent=2, ensure_ascii=False) + "\n```\n\n")
    lines.append("## By variant\n\n```json\n" + json.dumps(payload["summary_by_variant"], indent=2, ensure_ascii=False) + "\n```\n")
    note.write_text("".join(lines), encoding="utf-8")
    overall = payload["summary_overall"] or {}
    print(json.dumps({
        "status": payload["status"],
        "out": str(out),
        "note": str(note),
        "faithful_rate": overall.get("faithful_rate"),
        "substantive_rate": overall.get("substantive_rate"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
