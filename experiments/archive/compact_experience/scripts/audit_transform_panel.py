#!/usr/bin/env python3
"""Audit research transformation-panel Qwen outputs.

This script merges the two generation shards with the panel prompt/original
metadata and applies transparent, task-independent quality checks.  It does not use
BabyLM evaluation data or AoA/CDI information.  The goal is to decide whether
information-efficient second-view families are viable enough for a real fixed-budget
training screen.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = _public_path('experiments/archive/compact_experience')
PANEL_DIR = _public_path('experiments/archive/compact_experience/data/transform_panel')
PROMPTS_PATH = _public_path('experiments/archive/compact_experience/data/transform_panel/panel_prompts.jsonl')
ORIGINALS_PATH = _public_path('experiments/archive/compact_experience/data/transform_panel/panel_originals.jsonl')
OUTPUT_PATHS = [
    _public_path('experiments/archive/compact_experience/training/runs/transform_panel_qwen_shard0/outputs.jsonl'),
    _public_path('experiments/archive/compact_experience/training/runs/transform_panel_qwen_shard1/outputs.jsonl'),
]
OUT_DIR = _public_path('experiments/archive/compact_experience/data/transform_panel/audit')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = _public_path('experiments/archive/compact_experience/data/transform_panel/audit/transform_panel_audit.json')
ACCEPTED_PATH = _public_path('experiments/archive/compact_experience/data/transform_panel/audit/accepted_panel_records.jsonl')
REJECTED_PATH = _public_path('experiments/archive/compact_experience/data/transform_panel/audit/rejected_panel_records.jsonl')
SAMPLE_PATH = _public_path('experiments/archive/compact_experience/data/transform_panel/audit/stratified_examples.json')

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?|\*[A-Za-z]+:?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)
TERMINAL_OK = re.compile(r"[.!?][\"'”’\)]*$")
META_PREFIXES = ("here is", "here's", "sure", "certainly", "of course", "the rewritten", "rewritten sentence", "sentence:", "rewrite:", "output:", "question:", "answer:")
NEG_WORDS = {"no", "not", "never", "nothing", "none", "nobody", "cannot", "can't", "won't", "isn't", "wasn't", "don't", "doesn't", "didn't", "without", "neither", "nor"}
MODAL_WORDS = {"can", "could", "may", "might", "must", "should", "would", "will", "shall", "ought", "need", "needs", "needed"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "may", "might", "must", "not", "of", "on",
    "or", "our", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "under", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "would", "you", "your", "one", "only",
    "more", "most", "all", "any", "both", "each", "every", "own", "same", "other", "through",
}
FIRST_WORD_NOT_ENTITY = {
    "A", "An", "And", "As", "At", "But", "By", "For", "From", "He", "Her", "His", "I", "If", "In",
    "It", "On", "She", "Some", "That", "The", "Then", "There", "They", "This", "To", "Under", "We",
    "When", "Where", "While", "With", "You", "Your",
}

EXPECTED_RATIO = {
    "near_paraphrase": (0.70, 1.30),
    "capsule_065": (0.45, 0.85),
    "capsule_050": (0.30, 0.70),
    "typed_relation_qa": (0.25, 0.95),
}
CONTENT_OVERLAP_RANGE = {
    "near_paraphrase": (0.20, 0.96),
    "capsule_065": (0.12, 0.96),
    "capsule_050": (0.08, 0.96),
    "typed_relation_qa": (0.05, 0.95),
}
MIN_WORDS = {
    "near_paraphrase": 6,
    "capsule_065": 6,
    "capsule_050": 6,
    "typed_relation_qa": 5,
}
MAX_WORDS = {
    "near_paraphrase": 75,
    "capsule_065": 60,
    "capsule_050": 45,
    "typed_relation_qa": 25,
}


def norm_ws(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def clean_output(text: str) -> str:
    t = norm_ws(text or "")
    changed = True
    while changed and len(t) >= 2:
        changed = False
        for a, b in [("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")]:
            if t.startswith(a) and t.endswith(b):
                t = norm_ws(t[1:-1].strip())
                changed = True
    low = t.lower()
    for m in ("rewritten sentence:", "rewrite:", "output:", "sentence:"):
        if low.startswith(m):
            t = norm_ws(t[len(m):].strip())
            low = t.lower()
    return t


def words(text: str) -> list[str]:
    return (text or "").split()


def word_count(text: str) -> int:
    return len(words(text))


def norm_tokens(text: str) -> list[str]:
    out = []
    for m in WORD_RE.finditer(text or ""):
        t = m.group(0).strip("'\"“”‘’.,!?;:()[]{}").lower()
        if t:
            out.append(t)
    return out


def content_tokens(text: str) -> list[str]:
    return [t for t in norm_tokens(text) if len(t) >= 3 and t not in STOPWORDS]


def number_tokens(text: str) -> list[str]:
    return [re.sub(r"\s+", "", m.group(0).lower()) for m in NUM_RE.finditer(text or "")]


def entity_tokens(text: str) -> list[str]:
    ents = []
    raw = words(text)
    for i, w in enumerate(raw):
        stripped = w.strip("\"“”‘’.,!?;:()[]{}").rstrip("*")
        if not stripped:
            continue
        if stripped.startswith("*") and len(stripped) > 2:
            ents.append(stripped.lower().rstrip(":")); continue
        letters = re.sub(r"[^A-Za-z]", "", stripped)
        if not letters:
            continue
        is_cap = len(letters) > 1 and letters[0].isupper()
        has_internal_cap = any(c.isupper() for c in letters[1:])
        all_caps = len(letters) >= 2 and letters.isupper()
        if is_cap or has_internal_cap or all_caps:
            if i == 0 and not has_internal_cap and not all_caps and stripped not in FIRST_WORD_NOT_ENTITY and not stripped.startswith("*"):
                continue
            if stripped.lower() in STOPWORDS:
                continue
            ents.append(stripped.lower())
    seen = set(); out = []
    for e in ents:
        if e not in seen:
            out.append(e); seen.add(e)
    return out


def containment_recall(required: list[str], observed_text: str) -> float:
    if not required:
        return 1.0
    obs = set(norm_tokens(observed_text))
    hit = sum(1 for x in required if x.strip("*:").lower() in obs)
    return hit / len(required)


def counter_recall(required: set[str], observed_text: str) -> float:
    if not required:
        return 1.0
    obs = set(norm_tokens(observed_text))
    return len(required & obs) / len(required)


def content_overlap(src: str, out: str) -> float:
    a = set(content_tokens(src)); b = set(content_tokens(out))
    if not a:
        return 1.0
    return len(a & b) / len(a)


def content_jaccard(src: str, out: str) -> float:
    a = set(content_tokens(src)); b = set(content_tokens(out))
    if not (a or b):
        return 1.0
    return len(a & b) / len(a | b)


def has_repetition(text: str) -> bool:
    toks = norm_tokens(text)
    if len(toks) < 8:
        return False
    run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    grams = [tuple(toks[i:i+4]) for i in range(len(toks) - 3)]
    return any(v >= 2 for v in Counter(grams).values())


def qa_format_ok(text: str) -> bool:
    return "?" in text and ("—" in text or " - " in text or " – " in text) and text.index("?") < max(text.rfind("—"), text.rfind(" - "), text.rfind(" – "))


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals); pos = (len(vals)-1)*q
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi: return vals[lo]
    return vals[lo]*(hi-pos)+vals[hi]*(pos-lo)


def stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "min": None, "p05": None, "mean": None, "median": None, "p95": None, "max": None, "sum": 0}
    return {
        "n": len(vals), "min": round(min(vals), 4), "p05": round(pct(vals, .05), 4),
        "mean": round(statistics.mean(vals), 4), "median": round(statistics.median(vals), 4),
        "p95": round(pct(vals, .95), 4), "max": round(max(vals), 4), "sum": round(sum(vals), 4)
    }


def audit_record(prompt: dict, output_rec: dict) -> dict:
    src = prompt["source_sentence"]
    transform = prompt["transform"]
    out_raw = output_rec.get("output", "")
    out = clean_output(out_raw)
    src_words = int(prompt["source_words"])
    out_words = word_count(out)
    ratio = out_words / max(1, src_words)
    nums = number_tokens(src)
    ents = entity_tokens(src)
    neg = set(t for t in norm_tokens(src) if t in NEG_WORDS)
    modal = set(t for t in norm_tokens(src) if t in MODAL_WORDS)
    overlap = content_overlap(src, out)
    jacc = content_jaccard(src, out)
    lo, hi = EXPECTED_RATIO[transform]
    clo, chi = CONTENT_OVERLAP_RANGE[transform]
    reasons = []
    low = out.lower().strip()
    if not out:
        reasons.append("empty")
    if any(low.startswith(m) for m in META_PREFIXES):
        reasons.append("meta_prefix")
    if out_words < MIN_WORDS[transform]:
        reasons.append("too_short")
    if out_words > MAX_WORDS[transform]:
        reasons.append("too_long")
    if ratio < lo:
        reasons.append("len_ratio_low")
    if ratio > hi:
        reasons.append("len_ratio_high")
    if transform != "typed_relation_qa" and not TERMINAL_OK.search(out):
        reasons.append("bad_terminal")
    if transform == "typed_relation_qa" and not qa_format_ok(out):
        reasons.append("bad_qa_format")
    if number_tokens(src) != number_tokens(out) and transform != "typed_relation_qa":
        reasons.append("number_mismatch")
    if transform == "typed_relation_qa" and nums and containment_recall(nums, out) < 0.5:
        reasons.append("qa_number_loss")
    ent_recall = containment_recall(ents, out)
    if ents and ent_recall < (0.75 if len(ents) > 4 else 1.0) and transform != "typed_relation_qa":
        reasons.append("entity_loss")
    if transform == "typed_relation_qa" and ents and ent_recall < 0.5:
        reasons.append("qa_entity_weak")
    if counter_recall(neg, out) < 1.0:
        reasons.append("negation_loss")
    if counter_recall(modal, out) < 0.8:
        reasons.append("modality_loss")
    if overlap < clo:
        reasons.append("low_content_overlap")
    if overlap > chi:
        reasons.append("copy_overlap")
    if has_repetition(out):
        reasons.append("repetition")
    if "\n" in out_raw.strip():
        reasons.append("multiline_output")
    return {
        "id": prompt["id"],
        "original_id": prompt["original_id"],
        "transform": transform,
        "source": prompt["source_name"],
        "length_bin": prompt["length_bin"],
        "source_words": src_words,
        "output_words": out_words,
        "len_ratio": ratio,
        "content_overlap": overlap,
        "content_jaccard": jacc,
        "entity_count": len(ents),
        "entity_recall": ent_recall,
        "number_count": len(nums),
        "negation_count": len(neg),
        "modality_count": len(modal),
        "accepted": not reasons,
        "reasons": reasons,
        "original": src,
        "output": out,
        "raw_output": out_raw,
    }


def main() -> None:
    prompts = {r["id"]: r for r in load_jsonl(PROMPTS_PATH)}
    outputs = []
    missing_files = []
    for p in OUTPUT_PATHS:
        if not p.exists():
            missing_files.append(str(p.relative_to(ROOT)))
        else:
            outputs.extend(load_jsonl(p))
    if missing_files:
        raise SystemExit(f"missing generation output files: {missing_files}")
    by_prompt_text = defaultdict(list)
    # Generation outputs preserve prompt and include no id; map by prompt text and occurrence index.
    for rec in outputs:
        by_prompt_text[rec.get("prompt", "")].append(rec)
    prompt_occ_idx = Counter()
    audited = []
    missing_outputs = []
    for pid, prompt in prompts.items():
        k = prompt["prompt"]
        idx = prompt_occ_idx[k]
        prompt_occ_idx[k] += 1
        if idx >= len(by_prompt_text.get(k, [])):
            missing_outputs.append(pid)
            continue
        audited.append(audit_record(prompt, by_prompt_text[k][idx]))
    if missing_outputs:
        raise SystemExit(f"missing outputs for {len(missing_outputs)} prompts; first={missing_outputs[:5]}")

    summary_by_transform = {}
    for transform in sorted({a["transform"] for a in audited}):
        rows = [a for a in audited if a["transform"] == transform]
        acc = [a for a in rows if a["accepted"]]
        rej = [a for a in rows if not a["accepted"]]
        reason_counts = Counter(r for a in rej for r in a["reasons"])
        src_counts = Counter(a["source"] for a in acc)
        len_counts = Counter(a["length_bin"] for a in acc)
        summary_by_transform[transform] = {
            "n": len(rows),
            "accepted": len(acc),
            "accept_rate": round(len(acc) / len(rows), 4) if rows else 0,
            "rejected": len(rej),
            "rejection_reasons": dict(reason_counts.most_common()),
            "output_word_stats_all": stats([a["output_words"] for a in rows]),
            "output_word_stats_accepted": stats([a["output_words"] for a in acc]),
            "len_ratio_stats_accepted": stats([a["len_ratio"] for a in acc]),
            "content_overlap_stats_accepted": stats([a["content_overlap"] for a in acc]),
            "content_jaccard_stats_accepted": stats([a["content_jaccard"] for a in acc]),
            "entity_recall_stats_entity_cases": stats([a["entity_recall"] for a in rows if a["entity_count"] > 0]),
            "accepted_source_counts": dict(src_counts),
            "accepted_length_bin_counts": dict(len_counts),
        }

    # Per-original completeness: can we compare transforms on exactly the same originals?
    by_orig = defaultdict(dict)
    for a in audited:
        by_orig[a["original_id"]][a["transform"]] = a
    transforms = sorted(summary_by_transform)
    complete_accept_sets = {}
    for t in transforms:
        complete_accept_sets[t] = {oid for oid, d in by_orig.items() if d.get(t, {}).get("accepted")}
    common_accept_all = set.intersection(*complete_accept_sets.values()) if complete_accept_sets else set()
    common_capsule_current = complete_accept_sets.get("near_paraphrase", set()) & complete_accept_sets.get("capsule_050", set()) & complete_accept_sets.get("capsule_065", set())

    with ACCEPTED_PATH.open("w", encoding="utf-8") as af, REJECTED_PATH.open("w", encoding="utf-8") as rf:
        for a in audited:
            (af if a["accepted"] else rf).write(json.dumps(a, ensure_ascii=False) + "\n")

    samples = {}
    for transform in transforms:
        samples[transform] = {
            "accepted_first10": [{k: r[k] for k in ["id", "source", "source_words", "output_words", "len_ratio", "content_overlap", "original", "output"]} for r in audited if r["transform"] == transform and r["accepted"]][:10],
            "rejected_first10": [{k: r[k] for k in ["id", "source", "source_words", "output_words", "len_ratio", "content_overlap", "reasons", "original", "output"]} for r in audited if r["transform"] == transform and not r["accepted"]][:10],
        }
    SAMPLE_PATH.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "status": "TRANSFORM_PANEL_AUDIT",
        "n_prompts": len(prompts),
        "n_outputs": len(outputs),
        "n_audited": len(audited),
        "output_files": [str(p.relative_to(ROOT)) for p in OUTPUT_PATHS],
        "sha256_outputs": {p.name: sha(p) for p in OUTPUT_PATHS},
        "summary_by_transform": summary_by_transform,
        "accepted_original_sets": {t: len(s) for t, s in complete_accept_sets.items()},
        "common_accept_all_transforms": len(common_accept_all),
        "common_accept_near_and_capsules": len(common_capsule_current),
        "files": {
            "accepted": str(ACCEPTED_PATH.relative_to(ROOT)),
            "rejected": str(REJECTED_PATH.relative_to(ROOT)),
            "examples": str(SAMPLE_PATH.relative_to(ROOT)),
        },
        "interpretation_help": {
            "use": "This is an automatic surface/fidelity pre-audit. It can reject obvious bad families or choose families for human/teacher semantic audit, but it cannot prove semantic adequacy by itself.",
            "next": "If capsule_050 or capsule_065 has strong acceptance and acceptable samples, build a fixed-budget corpus screen with current-Qwen and selected-original controls. If acceptance is low or role/negation errors appear in samples, repair prompts before training.",
        },
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
