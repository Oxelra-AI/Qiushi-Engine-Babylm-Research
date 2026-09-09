#!/usr/bin/env python3
"""research: freeze and apply a shared preservation standard for FW compact views.

CPU-only.  The purpose is to make the FW source-compact mechanism test depend on
faithful compact consolidation, not merely on filling a word budget with any
rewrite-like text.  The same standard is applied to the already validated
Qwen3.5 compact rewrites and to the new Qwen3.5 outputs (pilot now; full run when
available).  The source_repeat arm is defined over exactly the same retained
source set and with the same companion word counts.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time
from typing import Any, Iterable

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
FROZEN_SOURCES = _public_path('experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_frozen_sources.jsonl')
PILOT_ROWS = _public_path('experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery/qwen35_generation_merged_quality.jsonl')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard')
NOTE_PATH = _public_path('research/notes/representation_and_objectives/fw_preservation_standard_and_pair_set.md')

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th|percent)?", re.I)
CAP_SEQ_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9.&'’\-]+(?:\s+|$)){1,8}")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from", "has", "have", "had",
    "he", "her", "his", "i", "in", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "they",
    "this", "to", "was", "were", "which", "who", "will", "with", "would", "you", "your", "our", "we", "but",
    "if", "then", "than", "into", "about", "can", "could", "may", "might", "should", "not", "no", "so", "also",
    "only", "more", "most", "many", "much", "some", "such", "very", "between", "during", "after", "before",
    "while", "where", "when", "because", "due", "led", "leads", "result", "results", "using", "used", "use",
    "study", "studies", "research", "found", "find", "shows", "show", "seen", "get", "gets", "make", "made",
}

ENTITY_STOP = {
    "The", "A", "An", "This", "That", "These", "Those", "In", "On", "For", "At", "By", "From", "To", "And",
    "But", "Or", "If", "When", "While", "Because", "Although", "However", "There", "Since", "Several", "People",
    "Kids", "Species", "Research", "Study", "Studies", "Editor", "Note", "Both", "Many", "More", "Most", "Some",
    "Such", "One", "Two", "First", "Second", "New", "Old", "During", "After", "Before", "Between", "Article",
    "Source", "Sentence", "English", "FineWeb", "Output", "Rewrite", "Answer", "May", "Can", "Could", "Would",
}

BAD_OUTPUT_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"^\s*(sure|here('| i)s|certainly|of course)\b",
        r"as an ai", r"i (cannot|can't)", r"please provide", r"output only", r"rewrite the factual",
        r"source sentence", r"simplified sentence\s*:", r"the original sentence", r"not mentioned in the sentence",
        r"cannot determine", r"\[.*\]", r"^\s*[-*•]", r"\n\s*[-*•]",
    ]
]

NEG_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "without", "lack", "lacks", "lacked", "lacking", "cannot",
    "can't", "won't", "isn't", "aren't", "doesn't", "don't", "didn't", "wasn't", "weren't", "unable", "fail",
    "fails", "failed", "failure", "avoid", "avoids", "avoided", "prevent", "prevents", "prevented", "poor",
    "worse", "worst", "non", "illegal", "absent", "absence", "miss", "missing", "rarely", "seldom",
}
EXPLICIT_NEG = {
    "not", "no", "never", "none", "neither", "nor", "without", "lack", "lacks", "lacked", "lacking", "cannot",
    "can't", "won't", "isn't", "aren't", "doesn't", "don't", "didn't", "wasn't", "weren't", "unable",
}
MODAL_GROUPS = {
    "possibility": {"may", "might", "could", "possible", "possibly", "potential", "potentially", "perhaps"},
    "ability": {"can", "could", "able", "capable", "allow", "allows", "allowed", "let", "lets", "help", "helps", "helped", "enable", "enables", "enabled"},
    "necessity": {"must", "should", "need", "needs", "needed", "required", "requires", "necessary", "important"},
    "likelihood": {"likely", "unlikely", "probable", "probably", "suggest", "suggests", "suggested", "seem", "seems", "appears", "apparently"},
}
CAUSAL_WORDS = {
    "because", "cause", "causes", "caused", "causing", "due", "therefore", "thereby", "hence", "since", "so",
    "led", "leads", "lead", "result", "results", "resulted", "resulting", "allow", "allows", "allowed", "let", "lets", "enable",
    "enables", "enabled", "prevent", "prevents", "prevented", "driven", "drove", "impetus", "reason", "reasons",
}
CAUSAL_LIGHT = {"by", "from", "through", "via", "with", "so"}
COMPARISON_DIRS = {
    "up": {"more", "greater", "higher", "larger", "increase", "increased", "increases", "raise", "raised", "raises", "enhance", "enhanced", "enhances", "rising", "risen", "rose", "over", "above", "exceed", "exceeds", "exceeded", "frequent", "frequently"},
    "down": {"less", "fewer", "lower", "smaller", "reduce", "reduced", "reduces", "decrease", "decreased", "decreases", "drop", "dropped", "decline", "declined", "below", "worse", "poor"},
    "before": {"before", "earlier", "prior", "previous", "previously", "former"},
    "after": {"after", "later", "following", "subsequent", "subsequently"},
    "contrast": {"unlike", "whereas", "compared", "different", "difference", "differences", "rather"},
    "similarity": {"similar", "alike"},
}
# Words whose mere presence can be temporal or discourse, not necessarily contrast.
AMBIGUOUS_COMPARISON = {"while"}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(" ".join((text or "").split()).split())


def words(text: str) -> list[str]:
    return [m.group(0).lower().replace("’", "'") for m in WORD_RE.finditer(text or "")]


def content(text: str) -> set[str]:
    return {w for w in words(text) if len(w) > 2 and w not in STOPWORDS and not NUM_RE.fullmatch(w)}


def normalize_num(s: str) -> str:
    s = str(s).lower().replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("percent", "%")
    return s


def nums(text: str) -> set[str]:
    return {normalize_num(m.group(0)) for m in NUM_RE.finditer(text or "")}


def norm_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def norm_hash(text: str) -> str:
    return hashlib.sha256(norm_phrase(text).encode("utf-8")).hexdigest()[:32]


def norm_hash(text: str) -> str:
    # research frozen source hashes used lowercased whitespace normalization rather than punctuation stripping.
    t = " ".join((text or "").lower().split())
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:32]


def clean_output(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```(?:\w+)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    s = re.sub(r"^(Rewrite|Output|Answer|Simplified sentence|Sentence)\s*:\s*", "", s, flags=re.I).strip()
    if "\n" in s:
        parts = [p.strip() for p in s.splitlines() if p.strip()]
        if parts:
            s = parts[0]
    return re.sub(r"\s+", " ", s).strip(" \t\"'`“”")


def real_entities(text: str) -> set[str]:
    ents: set[str] = set()
    # Multi-token capital spans and non-initial proper-looking singletons.
    for m in CAP_SEQ_RE.finditer(text or ""):
        raw = " ".join(m.group(0).split()).strip(" .,:;!?()[]{}\"'")
        if not raw:
            continue
        toks = raw.split()
        if raw in ENTITY_STOP:
            continue
        if toks[0] in ENTITY_STOP and len(toks) == 1:
            continue
        # Trim generic sentence starters from a multi-token span while preserving real following names.
        while toks and toks[0] in ENTITY_STOP:
            toks = toks[1:]
        if not toks:
            continue
        raw = " ".join(toks)
        start_char = m.start()
        at_sentence_start = start_char == 0 or (start_char >= 2 and text[start_char - 2:start_char] in {". ", "? ", "! "})
        if len(toks) == 1:
            tok = toks[0]
            if tok in ENTITY_STOP:
                continue
            # Keep acronyms, internal-capital tokens, and capitalized singletons away from sentence start.
            if tok.isupper() and len(tok) > 1:
                pass
            elif re.search(r"[A-Za-z].*[0-9]|[0-9].*[A-Za-z]", tok):
                pass
            elif re.search(r"[a-z][A-Z]", tok):
                pass
            elif at_sentence_start:
                continue
        ent = norm_phrase(raw)
        if ent and len(ent) > 1:
            ents.add(ent)
    return ents


def entity_preserved(ent: str, output: str) -> bool:
    e = norm_phrase(ent)
    o = norm_phrase(output)
    if not e:
        return True
    if e in o:
        return True
    etoks = {t for t in e.split() if len(t) > 2 and t not in STOPWORDS}
    otoks = content(output)
    if not etoks:
        return True
    kept = len(etoks & otoks)
    if len(etoks) <= 2:
        return kept == len(etoks)
    return kept / len(etoks) >= 0.75


def content_recall(source: str, rewrite: str) -> float:
    src = content(source)
    out = content(rewrite)
    if not src:
        return 1.0
    return len(src & out) / len(src)


def content_overlap(source: str, rewrite: str) -> float:
    src = content(source)
    out = content(rewrite)
    if not src and not out:
        return 1.0
    if not src or not out:
        return 0.0
    return len(src & out) / len(src | out)


def marker_words(text: str) -> set[str]:
    ws = set(words(text))
    joined = " " + " ".join(words(text)) + " "
    if " due to " in joined:
        ws.add("due")
    if " led to " in joined:
        ws.add("led")
    if " as a result " in joined:
        ws.add("result")
    if " because of " in joined:
        ws.add("because")
    return ws


def neg_signature(text: str) -> dict[str, Any]:
    ws = marker_words(text)
    joined = " " + " ".join(words(text)) + " "
    explicit = sorted(ws & EXPLICIT_NEG)
    conceptual = sorted(ws & NEG_WORDS)
    if re.search(r"\bless\s+than\b|\bfewer\s+than\b|\bno\s+longer\b|\bnot\s+only\b", joined):
        conceptual.append("comparative_negative_phrase")
    return {"has_negative_polarity": bool(conceptual), "explicit_negative": sorted(set(explicit)), "negative_words": sorted(set(conceptual))}


def modality_signature(text: str) -> dict[str, Any]:
    ws = marker_words(text)
    groups = {g: sorted(ws & vals) for g, vals in MODAL_GROUPS.items() if ws & vals}
    return {"has_modality": bool(groups), "groups": groups, "all": sorted({x for xs in groups.values() for x in xs})}


def causal_signature(text: str) -> dict[str, Any]:
    ws = marker_words(text)
    strong = sorted(ws & CAUSAL_WORDS)
    light = sorted(ws & CAUSAL_LIGHT)
    return {"has_causal": bool(strong), "strong": strong, "light": light}


def comparison_signature(text: str) -> dict[str, Any]:
    ws = marker_words(text)
    dirs: dict[str, list[str]] = {}
    for d, vals in COMPARISON_DIRS.items():
        hit = sorted(ws & vals)
        if hit:
            dirs[d] = hit
    # Add phrase-level less/more/fewer-than support; avoid treating discourse "while" alone as contrast.
    t = " " + " ".join(words(text)) + " "
    if re.search(r"\bless\s+than\b|\bunder\s+\d|\bbelow\s+\d", t):
        dirs.setdefault("down", []).append("down_phrase")
    if re.search(r"\bmore\s+than\b|\bover\s+\d|\babove\s+\d", t):
        dirs.setdefault("up", []).append("up_phrase")
    return {"has_comparison": bool(dirs), "directions": {k: sorted(set(v)) for k, v in dirs.items()}}


def side_tokens_around_first_marker(text: str, marker_set: set[str]) -> tuple[set[str], set[str]]:
    toks = words(text)
    idx = None
    for i, t in enumerate(toks):
        if t in marker_set:
            idx = i
            break
    if idx is None:
        return set(), set()
    left = {t for t in toks[max(0, idx - 10):idx] if t not in STOPWORDS and len(t) > 2}
    right = {t for t in toks[idx + 1:idx + 11] if t not in STOPWORDS and len(t) > 2}
    return left, right


def structure_flags(raw_output: str, cleaned: str) -> list[str]:
    flags: list[str] = []
    if not cleaned:
        flags.append("empty_output")
        return flags
    if any(rx.search(raw_output or "") for rx in BAD_OUTPUT_PATTERNS):
        flags.append("bad_output_pattern")
    if "\n" in (raw_output or "").strip():
        flags.append("multi_line_output")
    if cleaned and not (cleaned[0].isupper() or cleaned[0].isdigit() or cleaned[0] in "\"'"):
        flags.append("bad_start")
    if cleaned and cleaned[-1] not in ".!?\"'":
        flags.append("bad_end")
    terminal_marks = len(re.findall(r"[.!?](?:\s|$)", cleaned))
    if terminal_marks >= 3:
        flags.append("many_sentences")
    if re.search(r"\b(first|second|third|for example)\b", cleaned, re.I) and terminal_marks >= 2:
        flags.append("explanatory_expansion")
    return flags


def evaluate_pair(row: dict[str, Any], source_kind: str) -> dict[str, Any]:
    source = " ".join(str(row.get("source_text") or row.get("text") or "").split())
    raw = str(row.get("raw_output") or row.get("output") or row.get("generated_text") or row.get("rewrite_text") or "")
    rewrite = clean_output(raw)
    sw = int(row.get("source_words") or row.get("words") or wc(source))
    sw = wc(source) if sw != wc(source) else sw
    rw = int(row.get("rewrite_words") or wc(rewrite))
    rw = wc(rewrite) if rw != wc(rewrite) else rw
    ratio = rw / max(1, sw)
    src_nums = nums(source)
    out_nums = nums(rewrite)
    src_ents = real_entities(source)
    out_ents = real_entities(rewrite)
    kept_ents = {e for e in src_ents if entity_preserved(e, rewrite)}
    missed_ents = sorted(src_ents - kept_ents)
    new_ents = sorted(e for e in out_ents if e not in src_ents and not any(e in s or s in e for s in src_ents))
    crec = content_recall(source, rewrite)
    cov = content_overlap(source, rewrite)
    sneg, oneg = neg_signature(source), neg_signature(rewrite)
    smod, omod = modality_signature(source), modality_signature(rewrite)
    scau, ocau = causal_signature(source), causal_signature(rewrite)
    scomp, ocomp = comparison_signature(source), comparison_signature(rewrite)
    hard: list[str] = []
    soft: list[str] = []
    hard.extend(structure_flags(raw, rewrite))
    if ratio < 0.33 or ratio > 1.25:
        hard.append(f"length_ratio_{ratio:.2f}")
    elif ratio > 1.05:
        soft.append(f"not_compact_ratio_{ratio:.2f}")
    if norm_phrase(source) == norm_phrase(rewrite):
        hard.append("exact_copy")
    elif cov > 0.88 and ratio > 0.80 and sw >= 14:
        soft.append("near_copy_view")
    missing_nums = sorted(src_nums - out_nums)
    new_nums = sorted(out_nums - src_nums)
    if missing_nums:
        hard.append(f"missing_numbers_{len(missing_nums)}")
    if new_nums:
        hard.append(f"new_numbers_{len(new_nums)}")
    if missed_ents:
        # Single real named entities are important anchors; after false-positive trimming, any miss matters.
        hard.append(f"missing_entities_{len(missed_ents)}")
    if len(new_ents) > max(2, len(src_ents) + 1):
        hard.append(f"new_entity_like_{len(new_ents)}")
    if sw >= 14 and crec < 0.28:
        hard.append(f"low_content_recall_{crec:.2f}")
    if sw >= 10 and cov < 0.10:
        hard.append(f"low_content_overlap_{cov:.2f}")

    if sneg["has_negative_polarity"] and not oneg["has_negative_polarity"]:
        hard.append("polarity_lost")
    if not sneg["has_negative_polarity"] and (set(oneg["explicit_negative"]) & EXPLICIT_NEG):
        hard.append("polarity_added")

    # Preserve modality at the level of an explicit modal/hedge, while allowing category substitutions such as can -> may.
    if smod["has_modality"] and not omod["has_modality"]:
        hard.append("modality_lost")
    if not smod["has_modality"] and omod["has_modality"]:
        hard.append("modality_added")

    if scau["has_causal"]:
        causal_ok = ocau["has_causal"]
        if not causal_ok:
            left, right = side_tokens_around_first_marker(source, CAUSAL_WORDS)
            outc = content(rewrite)
            side_ok = (not left or bool(left & outc)) and (not right or bool(right & outc))
            light_bridge = bool(set(ocau["light"]) & CAUSAL_LIGHT)
            causal_ok = side_ok and light_bridge
        if not causal_ok:
            hard.append("causal_relation_lost")
    if (not scau["has_causal"]) and ocau["has_causal"]:
        hard.append("causal_relation_added")

    src_dirs = set(scomp["directions"].keys())
    out_dirs = set(ocomp["directions"].keys())
    # A contrast marker can be preserved by an explicit lexical contrast or by both sides retained with a comparison word.
    if src_dirs:
        missing_dirs = sorted(src_dirs - out_dirs)
        if missing_dirs:
            hard.append("comparison_direction_lost_" + "+".join(missing_dirs))
    if not src_dirs and out_dirs:
        # Adding before/after or up/down changes quantitative/temporal ordering; a lone similarity/contrast word is softer.
        if out_dirs & {"up", "down", "before", "after"}:
            hard.append("comparison_direction_added_" + "+".join(sorted(out_dirs)))
        else:
            soft.append("comparison_relation_added_" + "+".join(sorted(out_dirs)))

    usable = not hard
    return {
        "norm_hash": row.get("norm_hash") or norm_hash(source),
        "source_kind": source_kind,
        "prompt_id": row.get("prompt_id"),
        "doc_id": row.get("doc_id"),
        "domains": row.get("domains") or row.get("domain_hits") or [],
        "source_text": source,
        "rewrite_text": rewrite,
        "source_words": sw,
        "rewrite_words": rw,
        "pair_words": sw + rw,
        "length_ratio": ratio,
        "content_recall": crec,
        "content_overlap": cov,
        "source_numbers": sorted(src_nums),
        "output_numbers": sorted(out_nums),
        "missing_numbers": missing_nums,
        "new_numbers": new_nums,
        "source_entities": sorted(src_ents),
        "output_entities": sorted(out_ents),
        "missing_entities": missed_ents,
        "new_entity_like": new_ents,
        "source_negative": sneg,
        "output_negative": oneg,
        "source_modality": smod,
        "output_modality": omod,
        "source_causal": scau,
        "output_causal": ocau,
        "source_comparison": scomp,
        "output_comparison": ocomp,
        "hard_reasons": hard,
        "soft_flags": soft,
        "usable_for_compact_pair": usable,
        # Existing materializers used the field name accepted; keep it unambiguous here.
        "accepted": usable,
    }


def qstats(vals: Iterable[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals)
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p95": q(0.95), "max": xs[-1], "sum": sum(xs)}


def reason_prefix(reason: str) -> str:
    for pref in [
        "length_ratio", "missing_numbers", "new_numbers", "missing_entities", "new_entity_like", "low_content_recall",
        "low_content_overlap", "polarity", "modality", "causal_relation", "comparison_direction", "exact_copy",
        "empty_output", "bad_output_pattern", "multi_line_output", "bad_start", "bad_end", "many_sentences",
    ]:
        if reason.startswith(pref):
            return pref
    return reason


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r["usable_for_compact_pair"]]
    reasons = collections.Counter(reason_prefix(x) for r in rows for x in r["hard_reasons"])
    soft = collections.Counter(reason_prefix(x) for r in rows for x in r["soft_flags"])
    by_kind: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_kind[r["source_kind"]].append(r)

    def small(rs: list[dict[str, Any]]) -> dict[str, Any]:
        us = [r for r in rs if r["usable_for_compact_pair"]]
        return {
            "n": len(rs),
            "usable": len(us),
            "usable_rate": len(us) / max(1, len(rs)),
            "source_words": sum(r["source_words"] for r in rs),
            "usable_source_words": sum(r["source_words"] for r in us),
            "usable_rewrite_words": sum(r["rewrite_words"] for r in us),
            "usable_pair_words": sum(r["pair_words"] for r in us),
            "usable_rewrite_to_source_ratio": sum(r["rewrite_words"] for r in us) / max(1, sum(r["source_words"] for r in us)),
            "length_ratio": qstats(r["length_ratio"] for r in rs),
            "content_recall": qstats(r["content_recall"] for r in rs),
            "content_overlap": qstats(r["content_overlap"] for r in rs),
            "with_negative_source": sum(1 for r in rs if r["source_negative"]["has_negative_polarity"]),
            "negative_source_usable_rate": (
                sum(1 for r in rs if r["source_negative"]["has_negative_polarity"] and r["usable_for_compact_pair"])
                / max(1, sum(1 for r in rs if r["source_negative"]["has_negative_polarity"]))
            ),
            "with_modality_source": sum(1 for r in rs if r["source_modality"]["has_modality"]),
            "modality_source_usable_rate": (
                sum(1 for r in rs if r["source_modality"]["has_modality"] and r["usable_for_compact_pair"])
                / max(1, sum(1 for r in rs if r["source_modality"]["has_modality"]))
            ),
            "with_causal_source": sum(1 for r in rs if r["source_causal"]["has_causal"]),
            "causal_source_usable_rate": (
                sum(1 for r in rs if r["source_causal"]["has_causal"] and r["usable_for_compact_pair"])
                / max(1, sum(1 for r in rs if r["source_causal"]["has_causal"]))
            ),
            "with_comparison_source": sum(1 for r in rs if r["source_comparison"]["has_comparison"]),
            "comparison_source_usable_rate": (
                sum(1 for r in rs if r["source_comparison"]["has_comparison"] and r["usable_for_compact_pair"])
                / max(1, sum(1 for r in rs if r["source_comparison"]["has_comparison"]))
            ),
        }

    return {
        "overall": small(rows),
        "by_source_kind": {k: small(v) for k, v in sorted(by_kind.items())},
        "hard_reason_prefixes": reasons.most_common(40),
        "soft_flag_prefixes": soft.most_common(40),
    }


def build_old_rows(frozen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_pair(r, "existing_a02_qwen35") for r in frozen if r.get("has_rewrite") and r.get("rewrite_text")]


def build_pilot_rows(pilot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_pair(r, "new_qwen35_pilot") for r in pilot if r.get("rewrite_text")]


def source_repeat_companion(source: str, n_words: int) -> str:
    return " ".join(source.split()[:max(0, n_words)])


def arm_set_summary(usable_rows: list[dict[str, Any]]) -> dict[str, Any]:
    # This is the exact pair-level comparator used later by the materializer: same retained sources,
    # same companion word counts; only companion content differs.
    comp_words = sum(r["source_words"] + r["rewrite_words"] for r in usable_rows)
    repeat_words = 0
    repeat_exact_same_source_hashes = 0
    repeat_examples = []
    for r in usable_rows:
        rep = source_repeat_companion(r["source_text"], r["rewrite_words"])
        repeat_words += r["source_words"] + wc(rep)
        if norm_hash(r["source_text"]) == r["norm_hash"]:
            repeat_exact_same_source_hashes += 1
        if len(repeat_examples) < 8:
            repeat_examples.append({
                "norm_hash": r["norm_hash"],
                "source_words": r["source_words"],
                "companion_words": r["rewrite_words"],
                "compact": r["rewrite_text"],
                "repeat_prefix": rep,
            })
    return {
        "n_sources": len(usable_rows),
        "compact_pair_words": comp_words,
        "source_repeat_pair_words": repeat_words,
        "word_totals_match": comp_words == repeat_words,
        "source_hashes_match_records": repeat_exact_same_source_hashes,
        "weighted_compact_rewrite_to_source_ratio": sum(r["rewrite_words"] for r in usable_rows) / max(1, sum(r["source_words"] for r in usable_rows)),
        "examples": repeat_examples,
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = read_jsonl(FROZEN_SOURCES)
    pilot = read_jsonl(PILOT_ROWS)
    old_rows = build_old_rows(frozen)
    pilot_rows = build_pilot_rows(pilot)
    combined = old_rows + pilot_rows
    usable = [r for r in combined if r["usable_for_compact_pair"]]

    standard = {
        "name": "fw_compact_preservation_standard_v1_step100",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Use the same preservation standard for old and new Qwen3.5 compact rewrites before constructing compact_view/source_repeat BabyLM corpora.",
        "hard_failure_rules": [
            "empty or malformed non-sentence output",
            "rewrite/source whitespace-word ratio <0.33 or >1.25",
            "exact copy of source text",
            "any missing source number or any new number",
            "any missing real named entity after sentence-starter false positives are trimmed",
            "content recall <0.28 for sources with at least 14 words or content overlap <0.10 for sources with at least 10 words",
            "negative polarity in source lost, or explicit negative polarity introduced when source has none",
            "explicit modality/hedge in source lost, or introduced when source has none",
            "causal relation in source lost unless a light causal bridge preserves both sides; new strong causal relation introduced when source has none",
            "comparison/order direction in source not retained; new up/down/before/after direction introduced when source has none",
        ],
        "soft_annotations": [
            "rewrite/source ratio >1.05", "near-copy view", "new soft contrast/similarity relation"
        ],
        "inputs": {
            "frozen_sources": str(FROZEN_SOURCES),
            "frozen_sources_sha256": sha256_file(FROZEN_SOURCES),
            "pilot_rows": str(PILOT_ROWS),
            "pilot_rows_sha256": sha256_file(PILOT_ROWS),
        },
    }

    write_jsonl(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/existing_a02_qwen35_preservation_rows.jsonl'), old_rows)
    write_jsonl(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/new_qwen35_pilot_preservation_rows.jsonl'), pilot_rows)
    write_jsonl(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/available_preservation_rows.jsonl'), combined)
    write_jsonl(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/available_usable_pairs_for_materializer.jsonl'), usable)
    arm_summary = arm_set_summary(usable)
    summary = {
        "status": "FW_PRESERVATION_STANDARD_APPLIED",
        "created_utc": standard["created_utc"],
        "standard": standard,
        "summary": summarize(combined),
        "arm_source_set_comparison": arm_summary,
        "files": {
            "standard": str(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/preservation_standard_v1.json')),
            "existing_rows": str(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/existing_a02_qwen35_preservation_rows.jsonl')),
            "pilot_rows": str(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/new_qwen35_pilot_preservation_rows.jsonl')),
            "all_available_rows": str(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/available_preservation_rows.jsonl')),
            "usable_pairs": str(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/available_usable_pairs_for_materializer.jsonl')),
            "manual_review_samples": str(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/manual_review_samples.json')),
            "note": str(NOTE_PATH),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    (_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/preservation_standard_v1.json')).write_text(json.dumps(standard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/standard_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rng = random.Random(100829)
    def sample(rs: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        if len(rs) <= n:
            return rs
        return rng.sample(rs, n)
    rejected = [r for r in combined if not r["usable_for_compact_pair"]]
    review = {
        "existing_rejected_random": sample([r for r in old_rows if not r["usable_for_compact_pair"]], 24),
        "pilot_rejected_random": sample([r for r in pilot_rows if not r["usable_for_compact_pair"]], 24),
        "usable_low_content": sorted(usable, key=lambda r: (r["content_recall"], r["content_overlap"]))[:24],
        "causal_rejections": [r for r in rejected if any(x.startswith("causal_relation") for x in r["hard_reasons"])][:24],
        "comparison_rejections": [r for r in rejected if any(x.startswith("comparison_direction") for x in r["hard_reasons"])][:24],
        "polarity_rejections": [r for r in rejected if any(x.startswith("polarity") for x in r["hard_reasons"])][:24],
        "modality_rejections": [r for r in rejected if any(x.startswith("modality") for x in r["hard_reasons"])][:24],
    }
    (_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/manual_review_samples.json')).write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    overall = summary["summary"]["overall"]
    by_kind = summary["summary"]["by_source_kind"]
    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text(
        "# research — FW compact-view preservation standard and identical source-set comparison\n\n"
        "The FW mechanism-scale route should test broad faithful compact consolidation, not simply pair volume. "
        "I froze a single preservation standard and applied it to both already trusted A02/Qwen3.5 rewrites and the new Qwen3.5 pilot outputs. "
        "The same usable source hashes define compact_view and source_repeat arms; source_repeat uses the first `rewrite_words` words of the identical source sentence as the companion text, so pair word totals match exactly.\n\n"
        f"## Overall available measurement\n\n"
        f"- Rows measured: {overall['n']:,}\n"
        f"- Usable rows: {overall['usable']:,} ({overall['usable_rate']:.3f})\n"
        f"- Usable source words: {overall['usable_source_words']:,}\n"
        f"- Usable rewrite words: {overall['usable_rewrite_words']:,}\n"
        f"- Usable pair words: {overall['usable_pair_words']:,}\n"
        f"- Usable rewrite/source ratio: {overall['usable_rewrite_to_source_ratio']:.3f}\n\n"
        "## By source kind\n\n"
        + "".join(
            f"- {k}: {v['usable']:,}/{v['n']:,} usable ({v['usable_rate']:.3f}), pair words {v['usable_pair_words']:,}, ratio {v['usable_rewrite_to_source_ratio']:.3f}\n"
            for k, v in by_kind.items()
        )
        + "\n## Most common hard reasons\n\n"
        + "".join(f"- {k}: {v}\n" for k, v in summary["summary"]["hard_reason_prefixes"][:20])
        + "\n## Identical source-set arm comparison\n\n"
        f"- Sources retained: {arm_summary['n_sources']:,}\n"
        f"- compact_view pair words: {arm_summary['compact_pair_words']:,}\n"
        f"- source_repeat pair words: {arm_summary['source_repeat_pair_words']:,}\n"
        f"- Pair-word totals match: {arm_summary['word_totals_match']}\n\n"
        "## Files\n\n"
        f"- Standard: `{_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/preservation_standard_v1.json')}`\n"
        f"- Summary: `{_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/standard_summary.json')}`\n"
        f"- Usable pair set: `{_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/available_usable_pairs_for_materializer.jsonl')}`\n"
        f"- Review samples: `{_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/manual_review_samples.json')}`\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": summary["status"],
        "old_usable": by_kind.get("existing_a02_qwen35", {}).get("usable"),
        "old_n": by_kind.get("existing_a02_qwen35", {}).get("n"),
        "old_rate": by_kind.get("existing_a02_qwen35", {}).get("usable_rate"),
        "pilot_usable": by_kind.get("new_qwen35_pilot", {}).get("usable"),
        "pilot_n": by_kind.get("new_qwen35_pilot", {}).get("n"),
        "pilot_rate": by_kind.get("new_qwen35_pilot", {}).get("usable_rate"),
        "usable_pair_words": overall["usable_pair_words"],
        "arm_pair_words_match": arm_summary["word_totals_match"],
        "summary": str(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/standard_summary.json')),
        "note": str(NOTE_PATH),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
