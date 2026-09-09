#!/usr/bin/env python3
"""research: stricter corpus-derived relational/causal transition substrate miner.

Scientific purpose
------------------
research showed that GlobalPIQA_parallel failures are usually deep-rank, while
research showed that EWoK errors are dominated by context-conditioned target
reversal failures.  The shared object is not a 103-row benchmark branch, but a
broader weakness: small MLMs fail to bind a local context change to the world
relation or consequence that should change a target preference.

This script mines existing allowed reservoirs for broad, non-evaluation-derived
transition material with stricter structure than the research lexicon miner.  It
requires evidence of a context-conditioned relation/consequence (conditional,
causal, change-of-state, from-to, procedure, or explicit contrast) plus a content
channel (physical/material, spatial, temporal/quantity, affordance, or social/
agent relation).  It records high-purity candidate sentences and a balanced
research slice for possible later low-cost factorial probes.  It does not read
any official evaluation item text and does not launch training.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
A02_WS = USER_ROOT / "experiments/archive/frontier_consolidation"
OUT = A01_WS / "data/strict_transition_substrate"
NOTE = A01_WS / "notes/strict_transition_substrate.md"

RESERVOIRS: dict[str, dict[str, Any]] = {
    "compact_experience_aligned_10m_rows": {
        "path": USER_ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl",
        "fields": ["text"],
        "kind": "current_lineage_10M_pool",
    },
    "fw_frozen_sources_38167": {
        "path": A01_WS / "data/fw_mechanism_source_selection/fw_mechanism_frozen_sources.jsonl",
        "fields": ["text"],
        "kind": "fineweb_candidate_source_reservoir",
    },
    "fw_compact_pair_sources": {
        "path": A01_WS / "data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl",
        "fields": ["source_text"],
        "kind": "selected_fineweb_common_sources",
    },
    "fw_compact_rewrites": {
        "path": A01_WS / "data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl",
        "fields": ["rewrite_text"],
        "kind": "selected_qwen35_compact_rewrites",
    },
    "fw_breadth_whole_sentence_companions": {
        "path": A01_WS / "data/fw_source_breadth_wholesentence_arm/source_breadth_wholesentence_companion_sources.jsonl",
        "fields": ["text"],
        "kind": "selected_fineweb_breadth_companions",
    },
}

WORD_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)?\b", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘\[])|\n+")

# Predeclared broad lexicons.  They are intentionally not copied from official examples.
LEX: dict[str, list[str]] = {
    "condition_cause": [
        r"\bif\b", r"\bwhen\b", r"\bwhenever\b", r"\bunless\b", r"\bonce\b", r"\bafter\b", r"\bbefore\b",
        r"\bbecause\b", r"\bsince\b", r"\bdue to\b", r"\bas a result\b", r"\btherefore\b", r"\bso that\b",
        r"\bin order to\b", r"\bleads? to\b", r"\bcaus(?:e|es|ed|ing)\b", r"\bresult(?:s|ed|ing)?\b",
        r"\bprevent(?:s|ed|ing)?\b", r"\ballow(?:s|ed|ing)?\b", r"\bmake(?:s| made)?\b",
    ],
    "change_transition": [
        r"\bbecome(?:s|ing)?\b", r"\bbecame\b", r"\bturn(?:s|ed|ing)? into\b", r"\bchange(?:s|d|ing)?\b",
        r"\bgo(?:es|ing)? from\b", r"\bwent from\b", r"\bfrom\b.{0,40}\bto\b",
        r"\bincreas(?:e|es|ed|ing)\b", r"\bdecreas(?:e|es|ed|ing)\b", r"\bgrow(?:s|ing|n)?\b",
        r"\bshrink(?:s|ing|ed)?\b", r"\bdouble(?:s|d|ing)?\b", r"\bhalve(?:s|d|ing)?\b",
        r"\bfill(?:s|ed|ing)?\b", r"\bempty(?:ies|ied|ing)?\b", r"\bopen(?:s|ed|ing)?\b", r"\bclos(?:e|es|ed|ing)\b",
        r"\bheat(?:s|ed|ing)?\b", r"\bcool(?:s|ed|ing)?\b", r"\bfreez(?:e|es|ing)\b", r"\bmelt(?:s|ed|ing)?\b",
        r"\bdissolv(?:e|es|ed|ing)\b", r"\bbreak(?:s|ing|s up)?\b", r"\bbroke\b", r"\bbroken\b", r"\bbend(?:s|ing|ed)?\b",
        r"\bstretch(?:es|ed|ing)?\b", r"\btear(?:s|ing|tore)?\b", r"\bwet(?:s|ted|ting)?\b", r"\bdry(?:ies|ied|ing)?\b",
    ],
    "physical_action": [
        r"\bpush(?:ed|es|ing)?\b", r"\bpull(?:ed|s|ing)?\b", r"\bthrow(?:s|ing|n)?\b", r"\bthrew\b",
        r"\bdrop(?:ped|s|ping)?\b", r"\bfall(?:s|ing|en)?\b", r"\bfell\b", r"\bbounce(?:s|d|ing)?\b",
        r"\broll(?:s|ed|ing)?\b", r"\bfloat(?:s|ed|ing)?\b", r"\bsink(?:s|ing|sank)?\b", r"\bslide(?:s|d|ing)?\b",
        r"\bpour(?:s|ed|ing)?\b", r"\bspill(?:s|ed|ing)?\b", r"\bcover(?:s|ed|ing)?\b", r"\bblock(?:s|ed|ing)?\b",
        r"\breflect(?:s|ed|ing)?\b", r"\babsorb(?:s|ed|ing)?\b", r"\bemit(?:s|ted|ting)?\b", r"\bpass(?:es|ed|ing)? through\b",
        r"\bseal(?:s|ed|ing)?\b", r"\btie(?:s|d|ing)?\b", r"\battach(?:es|ed|ing)?\b", r"\bdetach(?:es|ed|ing)?\b",
        r"\blift(?:s|ed|ing)?\b", r"\bcarry(?:ies|ied|ing)?\b", r"\bcut(?:s|ting)?\b", r"\bslice(?:s|d|ing)?\b",
    ],
    "material_object": [
        r"\bwater\b", r"\bair\b", r"\bglass\b", r"\bmetal\b", r"\bplastic\b", r"\bwood(?:en)?\b", r"\bpaper\b",
        r"\bfabric\b", r"\bcloth\b", r"\brubber\b", r"\bliquid\b", r"\bsolid\b", r"\bgas\b", r"\bice\b",
        r"\bsteam\b", r"\bfire\b", r"\blight\b", r"\bshadow\b", r"\bsoil\b", r"\bstone\b", r"\bsand\b",
        r"\bbag\b", r"\bcup\b", r"\bbox\b", r"\bcontainer\b", r"\bbottle\b", r"\bwindow\b", r"\bdoor\b",
        r"\bball\b", r"\bpipe\b", r"\brope\b", r"\bstring\b", r"\bwire\b", r"\bwheel\b", r"\btool\b",
        r"\bhot\b", r"\bcold\b", r"\bwet\b", r"\bdry\b", r"\bsoft\b", r"\bhard\b", r"\bheavy\b", r"\blight\b",
        r"\bfull\b", r"\bempty\b", r"\bsealed\b", r"\btransparent\b", r"\bvisible\b", r"\bhidden\b",
    ],
    "spatial_relation": [
        r"\bleft\b", r"\bright\b", r"\bup\b", r"\bdown\b", r"\babove\b", r"\bbelow\b", r"\bunder\b", r"\bover\b",
        r"\binside\b", r"\boutside\b", r"\bin front of\b", r"\bbehind\b", r"\bbetween\b", r"\bthrough\b", r"\bacross\b",
        r"\baround\b", r"\btowards?\b", r"\baway from\b", r"\bnear\b", r"\bfar\b", r"\bcloser\b", r"\bfurther\b",
        r"\bnorth\b", r"\bsouth\b", r"\beast\b", r"\bwest\b", r"\btop\b", r"\bbottom\b", r"\bcenter\b", r"\bedge\b",
    ],
    "temporal_quantity": [
        r"\byear(?:s)?\b", r"\bmonth(?:s)?\b", r"\bweek(?:s)?\b", r"\bday(?:s)?\b", r"\bhour(?:s)?\b", r"\bminute(?:s)?\b",
        r"\btoday\b", r"\btomorrow\b", r"\byesterday\b", r"\bnext\b", r"\blast\b", r"\bearlier\b", r"\blater\b",
        r"\bfirst\b", r"\bsecond\b", r"\bthird\b", r"\bmore\b", r"\bless\b", r"\bfewer\b", r"\badd(?:s|ed|ing)?\b",
        r"\bsubtract(?:s|ed|ing)?\b", r"\bmultiply\b", r"\bdivide(?:s|d|ing)?\b", r"\btotal\b", r"\bevery\b", r"\beach\b",
        r"\b\d{1,4}\b",
    ],
    "affordance_procedure": [
        r"\bhow to\b", r"\buse(?:d|s|ing)?\b", r"\butensil(?:s)?\b", r"\bwear(?:s|ing)?\b", r"\bhold(?:s|ing)?\b",
        r"\bstore(?:s|d|ing)?\b", r"\bcook(?:s|ed|ing)?\b", r"\bwrite(?:s|ing)?\b", r"\bclean(?:s|ed|ing)?\b",
        r"\bserve(?:s|d|ing)?\b", r"\bmake sure\b", r"\bbe careful\b", r"\bsafe(?:ly|ty)?\b", r"\bbest\b", r"\bcan be used\b",
        r"\bin order to\b", r"\bso that\b", r"\bto prevent\b", r"\bto keep\b", r"\bto make\b",
    ],
    "social_agent_relation": [
        r"\bperson\b", r"\bpeople\b", r"\bchild\b", r"\bboy\b", r"\bgirl\b", r"\bfriend\b", r"\bteacher\b", r"\bparent\b",
        r"\bmother\b", r"\bfather\b", r"\bask(?:s|ed|ing)?\b", r"\btell(?:s|ing)?\b", r"\btold\b", r"\bgive(?:s|n|ing)?\b",
        r"\btake(?:s|n|ing)?\b", r"\bwants?\b", r"\bneed(?:s|ed|ing)?\b", r"\bhelp(?:s|ed|ing)?\b", r"\btrust(?:s|ed|ing)?\b",
        r"\bbelieve(?:s|d|ing)?\b", r"\bknow(?:s|ing)?\b", r"\blearn(?:s|ed|ing)?\b", r"\bremember(?:s|ed|ing)?\b",
    ],
}

CONTRAST_PAIRS = [
    ("more", "less"), ("more", "fewer"), ("full", "empty"), ("hot", "cold"), ("wet", "dry"),
    ("heavy", "light"), ("hard", "soft"), ("push", "pull"), ("open", "close"), ("up", "down"),
    ("above", "below"), ("inside", "outside"), ("front", "back"), ("left", "right"),
    ("north", "south"), ("east", "west"), ("before", "after"), ("first", "last"),
    ("increase", "decrease"), ("float", "sink"), ("break", "bend"), ("visible", "hidden"),
    ("transparent", "opaque"), ("near", "far"), ("closer", "further"), ("on", "off"),
]

NOISE_PATTERNS = {
    "url_or_markup": [r"https?://", r"www\.", r"</?\w+", r"\[[^\]]*(illustration|edit|citation)[^\]]*\]"],
    "finance_or_market_metaphor": [r"\bprice(?:s)?\b", r"\brate(?:s)?\b", r"\binflation\b", r"\bmarket\b", r"\bstock(?:s)?\b", r"\binterest rate"],
    "navigation_only": [r"\bnorth\b", r"\bsouth\b", r"\beast\b", r"\bwest\b", r"\broad\b", r"\bstreet\b", r"\bavenue\b", r"\bprovince\b"],
    "sports_or_game_score": [r"\bscore\b", r"\bteam\b", r"\bgame\b", r"\bmatch\b", r"\bplayer\b", r"\bseason\b"],
    "music_media_title": [r"\balbum\b", r"\bsong\b", r"\bfilm\b", r"\bepisode\b", r"\bnovel\b", r"\bchapter\b", r"\bTikTok\b"],
}

ROUTE_ORDER = [
    "physical_material_transition",
    "spatial_transition",
    "temporal_quantity_transition",
    "affordance_procedure_transition",
    "social_agent_transition",
    "explicit_contrast_transition",
]


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for sent in SENT_SPLIT_RE.split(text):
        sent = sent.strip()
        if not sent:
            continue
        # Split some packed transcript/question rows without needing external metadata.
        subparts = re.split(r"\s+(?=\*[A-Z]{2,5}:)|\s+(?=Q:)|\s+(?=A:)", sent)
        for sub in subparts:
            sub = norm_text(sub)
            if sub:
                parts.append(sub)
    return parts


def regex_count(patterns: Iterable[str], text: str) -> int:
    return sum(len(re.findall(p, text, flags=re.I)) for p in patterns)


def hit_map(text: str) -> dict[str, int]:
    return {k: regex_count(v, text) for k, v in LEX.items()}


def contrast_pairs(text: str) -> list[str]:
    lo = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    out = []
    for a, b in CONTRAST_PAIRS:
        if f" {a} " in lo and f" {b} " in lo:
            out.append(f"{a}/{b}")
    return out


def noise_hits(text: str) -> list[str]:
    out = []
    for name, pats in NOISE_PATTERNS.items():
        if any(re.search(p, text, flags=re.I) for p in pats):
            out.append(name)
    # many proper nouns are a sign of biographical/geographical fact rows rather than transition mechanics.
    toks = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    if len(toks) >= 8:
        out.append("proper_name_dense")
    digit_toks = re.findall(r"\b\d{1,4}\b", text)
    if len(digit_toks) >= 5:
        out.append("digit_dense")
    return out


def features(text: str) -> dict[str, Any]:
    h = hit_map(text)
    cp = contrast_pairs(text)
    lo = text.lower()
    transition_evidence = (
        h["condition_cause"] > 0
        or h["change_transition"] > 0
        or len(cp) > 0
        or bool(re.search(r"\b(from|into|onto|off|out of)\b.{0,45}\b(to|into|onto|off|out of)\b", lo))
    )
    content_channels = []
    if h["physical_action"] or h["material_object"]:
        content_channels.append("physical_material")
    if h["spatial_relation"]:
        content_channels.append("spatial")
    if h["temporal_quantity"]:
        content_channels.append("temporal_quantity")
    if h["affordance_procedure"]:
        content_channels.append("affordance")
    if h["social_agent_relation"]:
        content_channels.append("social_agent")
    routes = []
    if transition_evidence and (h["physical_action"] or h["material_object"]) and (h["condition_cause"] or h["change_transition"] or h["physical_action"] >= 2):
        routes.append("physical_material_transition")
    if transition_evidence and h["spatial_relation"] and (h["condition_cause"] or h["change_transition"] or h["physical_action"] or len(cp) > 0):
        routes.append("spatial_transition")
    if transition_evidence and h["temporal_quantity"] and (h["condition_cause"] or h["change_transition"] or len(cp) > 0 or h["temporal_quantity"] >= 3):
        routes.append("temporal_quantity_transition")
    if transition_evidence and h["affordance_procedure"] and (h["physical_action"] or h["material_object"] or h["spatial_relation"] or h["condition_cause"]):
        routes.append("affordance_procedure_transition")
    if transition_evidence and h["social_agent_relation"] and (h["condition_cause"] or h["change_transition"] or h["affordance_procedure"]):
        routes.append("social_agent_transition")
    if len(cp) > 0 and (h["condition_cause"] or h["change_transition"] or h["physical_action"] or h["affordance_procedure"]):
        routes.append("explicit_contrast_transition")
    noise = noise_hits(text)
    score = 0
    score += 4 * min(h["condition_cause"], 2)
    score += 4 * min(h["change_transition"], 2)
    score += 3 * min(h["physical_action"], 3)
    score += 2 * min(h["material_object"], 4)
    score += 2 * min(h["spatial_relation"], 3)
    score += 2 * min(h["temporal_quantity"], 3)
    score += 3 * min(h["affordance_procedure"], 2)
    score += 2 * min(h["social_agent_relation"], 2)
    score += 4 * min(len(cp), 2)
    score += 3 * len(routes)
    score -= 5 * len(set(noise) & {"url_or_markup", "music_media_title"})
    # Navigation/finance can still contain real relational transitions, but should not dominate high-purity examples.
    score -= 2 * len(set(noise) & {"finance_or_market_metaphor", "navigation_only", "sports_or_game_score", "proper_name_dense", "digit_dense"})
    high_purity = bool(routes and score >= 16 and "url_or_markup" not in noise and "music_media_title" not in noise)
    very_high_purity = bool(high_purity and score >= 22 and len(content_channels) >= 2 and not (set(noise) & {"finance_or_market_metaphor", "navigation_only", "sports_or_game_score"}))
    return {
        "hits": h,
        "contrast_pairs": cp,
        "routes": routes,
        "content_channels": content_channels,
        "transition_evidence": transition_evidence,
        "noise": noise,
        "score": score,
        "high_purity": high_purity,
        "very_high_purity": very_high_purity,
    }


def iter_records(label: str, spec: dict[str, Any]):
    p = Path(spec["path"])
    if not p.exists():
        return
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for field in spec["fields"]:
                text = str(obj.get(field, "") or "").strip()
                if not text:
                    continue
                for sentence_index, sent in enumerate(split_sentences(text)):
                    wc = word_count(sent)
                    if wc < 8 or wc > 85:
                        continue
                    yield {
                        "source_label": label,
                        "source_kind": spec["kind"],
                        "input_path": str(p),
                        "row_index": row_index,
                        "sentence_index": sentence_index,
                        "field": field,
                        "text": sent,
                        "words": wc,
                        "doc_id": obj.get("doc_id"),
                        "norm_hash": obj.get("norm_hash"),
                        "domains": obj.get("domains"),
                        "origin_source": obj.get("source") or obj.get("pool") or obj.get("source_kind") or obj.get("origin_source"),
                    }


def stable_key(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip().lower()).encode("utf-8")).hexdigest()


def qstats(vals: Iterable[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
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
    return {"n": len(xs), "min": xs[0], "p10": q(0.10), "median": statistics.median(xs), "mean": statistics.fmean(xs), "p90": q(0.90), "max": xs[-1]}


def summarize_source(rows: list[dict[str, Any]], scanned_sentences: int, scanned_words: int) -> dict[str, Any]:
    by_route = defaultdict(list)
    by_channel = defaultdict(list)
    for r in rows:
        for route in r["routes"]:
            by_route[route].append(r)
        for ch in r["content_channels"]:
            by_channel[ch].append(r)
    return {
        "scanned_sentences": scanned_sentences,
        "scanned_words": scanned_words,
        "candidate_count": len(rows),
        "candidate_words": sum(r["words"] for r in rows),
        "very_high_purity_count": sum(r["very_high_purity"] for r in rows),
        "very_high_purity_words": sum(r["words"] for r in rows if r["very_high_purity"]),
        "candidate_sentence_fraction": len(rows) / scanned_sentences if scanned_sentences else None,
        "candidate_word_fraction": sum(r["words"] for r in rows) / scanned_words if scanned_words else None,
        "route_counts": {k: len(v) for k, v in sorted(by_route.items())},
        "route_words": {k: sum(r["words"] for r in v) for k, v in sorted(by_route.items())},
        "channel_counts": {k: len(v) for k, v in sorted(by_channel.items())},
        "channel_words": {k: sum(r["words"] for r in v) for k, v in sorted(by_channel.items())},
        "score_stats": qstats(r["score"] for r in rows),
        "top_examples": sorted(rows, key=lambda r: (-r["score"], r["source_label"], r["row_index"], r["sentence_index"]))[:30],
    }


def make_balanced_slice(candidates: list[dict[str, Any]], word_budget: int = 200_000, per_route_budget: int = 45_000) -> list[dict[str, Any]]:
    """Greedy route-balanced candidate slice for future probes; not a training arm."""
    # Prefer very high-purity, short, distinct rows.  Avoid putting many duplicates from one document/hash.
    by_route: dict[str, list[dict[str, Any]]] = {r: [] for r in ROUTE_ORDER}
    for c in candidates:
        for route in c["routes"]:
            if route in by_route:
                by_route[route].append(c)
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    doc_counts: Counter[str] = Counter()
    total_words = 0
    rng = random.Random(108)
    for route in ROUTE_ORDER:
        pool = by_route.get(route, [])
        # stable random tiebreaker after score to avoid only one source style.
        scored = []
        for c in pool:
            tie = int(stable_key(c["text"])[:8], 16) / 0xFFFFFFFF
            scored.append((not c["very_high_purity"], -c["score"], abs(c["words"] - 28), tie, c))
        route_words = 0
        for *_ignore, c in sorted(scored):
            key = c["text_sha256"]
            if key in selected_keys:
                continue
            doc_key = str(c.get("norm_hash") or c.get("doc_id") or f"{c['source_label']}:{c['row_index']}")
            if doc_counts[doc_key] >= 3:
                continue
            if total_words + c["words"] > word_budget or route_words + c["words"] > per_route_budget:
                continue
            out = dict(c)
            out["selected_route_bucket"] = route
            selected.append(out)
            selected_keys.add(key)
            doc_counts[doc_key] += 1
            route_words += c["words"]
            total_words += c["words"]
        if total_words >= word_budget:
            break
    # Fill any unused budget from high purity diverse leftovers.
    leftovers = [c for c in candidates if c["text_sha256"] not in selected_keys]
    leftovers.sort(key=lambda c: (not c["very_high_purity"], -c["score"], abs(c["words"] - 30), c["source_label"], c["row_index"]))
    for c in leftovers:
        if total_words + c["words"] > word_budget:
            continue
        doc_key = str(c.get("norm_hash") or c.get("doc_id") or f"{c['source_label']}:{c['row_index']}")
        if doc_counts[doc_key] >= 3:
            continue
        out = dict(c)
        out.setdefault("selected_route_bucket", "fill_high_purity")
        selected.append(out)
        selected_keys.add(c["text_sha256"])
        doc_counts[doc_key] += 1
        total_words += c["words"]
        if total_words >= word_budget * 0.995:
            break
    return selected


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    all_candidates: list[dict[str, Any]] = []
    per_source: dict[str, Any] = {}
    for label, spec in RESERVOIRS.items():
        scanned_sentences = 0
        scanned_words = 0
        local_candidates: list[dict[str, Any]] = []
        for rec in iter_records(label, spec):
            scanned_sentences += 1
            scanned_words += rec["words"]
            feat = features(rec["text"])
            if not feat["high_purity"]:
                continue
            out = dict(rec)
            out.update(feat)
            out["text_sha256"] = stable_key(rec["text"])
            # Reduce later accidental source leakage: store only text and provenance, no official eval relation.
            local_candidates.append(out)
        per_source[label] = summarize_source(local_candidates, scanned_sentences, scanned_words)
        all_candidates.extend(local_candidates)

    # Deduplicate by sentence text, preserving highest score / richer routes.
    best_by_key: dict[str, dict[str, Any]] = {}
    for c in all_candidates:
        k = c["text_sha256"]
        old = best_by_key.get(k)
        if old is None or (c["score"], len(c["routes"]), c["words"]) > (old["score"], len(old["routes"]), old["words"]):
            best_by_key[k] = c
    dedup = sorted(best_by_key.values(), key=lambda r: (-r["score"], r["source_label"], r["row_index"], r["sentence_index"]))
    very_high = [r for r in dedup if r["very_high_purity"]]
    balanced = make_balanced_slice(dedup, word_budget=200_000, per_route_budget=45_000)

    candidates_path = OUT / "strict_transition_candidates.jsonl"
    very_high_path = OUT / "strict_transition_candidates_very_high.jsonl"
    balanced_path = OUT / "strict_transition_balanced_200k_slice.jsonl"
    write_jsonl(candidates_path, dedup)
    write_jsonl(very_high_path, very_high)
    write_jsonl(balanced_path, balanced)

    summary_rows: list[dict[str, Any]] = []
    for label, summ in per_source.items():
        for route in ROUTE_ORDER:
            summary_rows.append({
                "source_label": label,
                "route": route,
                "scanned_sentences": summ["scanned_sentences"],
                "scanned_words": summ["scanned_words"],
                "candidate_count": summ["route_counts"].get(route, 0),
                "candidate_words": summ["route_words"].get(route, 0),
                "candidate_sentence_fraction": summ["route_counts"].get(route, 0) / summ["scanned_sentences"] if summ["scanned_sentences"] else None,
                "candidate_word_fraction": summ["route_words"].get(route, 0) / summ["scanned_words"] if summ["scanned_words"] else None,
            })
    write_csv(OUT / "strict_transition_summary_by_source_route.csv", summary_rows)

    slice_rows = []
    by_bucket = defaultdict(list)
    for r in balanced:
        by_bucket[r.get("selected_route_bucket", "")].append(r)
    for bucket, rows in sorted(by_bucket.items()):
        slice_rows.append({
            "selected_route_bucket": bucket,
            "sentences": len(rows),
            "words": sum(r["words"] for r in rows),
            "very_high_purity_sentences": sum(r["very_high_purity"] for r in rows),
            "mean_score": statistics.fmean(r["score"] for r in rows) if rows else None,
        })
    write_csv(OUT / "strict_transition_balanced_200k_slice_summary.csv", slice_rows)

    route_counts = Counter()
    route_words = Counter()
    channel_counts = Counter()
    channel_words = Counter()
    contrast_counts = Counter()
    noise_counts = Counter()
    source_counts = Counter()
    source_words = Counter()
    for r in dedup:
        source_counts[r["source_label"]] += 1
        source_words[r["source_label"]] += r["words"]
        for route in r["routes"]:
            route_counts[route] += 1
            route_words[route] += r["words"]
        for ch in r["content_channels"]:
            channel_counts[ch] += 1
            channel_words[ch] += r["words"]
        for cp in r["contrast_pairs"]:
            contrast_counts[cp] += 1
        for n in r["noise"]:
            noise_counts[n] += 1

    payload = {
        "status": "STRICT_TRANSITION_SUBSTRATE_MINED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boundary": "Corpus-derived broad transition mining only; no official evaluation item text is read or used.",
        "inputs": {k: str(v["path"]) for k, v in RESERVOIRS.items()},
        "per_source": per_source,
        "dedup_candidate_count": len(dedup),
        "dedup_candidate_words": sum(r["words"] for r in dedup),
        "very_high_candidate_count": len(very_high),
        "very_high_candidate_words": sum(r["words"] for r in very_high),
        "route_counts": dict(route_counts),
        "route_words": dict(route_words),
        "channel_counts": dict(channel_counts),
        "channel_words": dict(channel_words),
        "contrast_pair_counts_top": contrast_counts.most_common(30),
        "noise_counts": dict(noise_counts),
        "source_counts": dict(source_counts),
        "source_words": dict(source_words),
        "score_stats": qstats(r["score"] for r in dedup),
        "balanced_200k_slice": {
            "path": str(balanced_path),
            "sentences": len(balanced),
            "words": sum(r["words"] for r in balanced),
            "summary_csv": str(OUT / "strict_transition_balanced_200k_slice_summary.csv"),
            "scientific_status": "research substrate only, not a committed training arm",
        },
        "files": {
            "summary_json": str(OUT / "strict_transition_substrate_miner.json"),
            "summary_by_source_route_csv": str(OUT / "strict_transition_summary_by_source_route.csv"),
            "candidates_jsonl": str(candidates_path),
            "very_high_candidates_jsonl": str(very_high_path),
            "balanced_200k_slice_jsonl": str(balanced_path),
            "note": str(NOTE),
        },
        "scientific_reading": {
            "use_after_fw_results": "Use these candidates only after compact/breadth endpoint vectors and EWoK/GlobalPIQA margins show whether corpus allocation moved context-conditioned relation failures.",
            "if_fw_no_relation_movement": "A small shared-checkpoint factorial can test swapping 50k-200k words from the balanced slice or applying a corpus-derived four-cell contrast objective before any 100M route.",
            "risk": "Regex mining still contains noise and metaphors; candidates require manual/automatic semantic filtering before training, especially explicit contrast pairs.",
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (OUT / "strict_transition_substrate_miner.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# research — Strict corpus-derived transition substrate\n\n")
    lines.append("## Purpose\n\n")
    lines.append("research EWoK and research GlobalPIQA point to a common weakness in context-conditioned world relations. This miner asks whether allowed reservoirs contain a cleaner broad substrate for later small probes, without reading official evaluation item text or shaping examples from the 103-row parallel set.\n\n")
    lines.append("## Main counts\n\n")
    lines.append(f"- Deduplicated high-purity candidates: `{len(dedup)}` sentences / `{sum(r['words'] for r in dedup)}` words.\n")
    lines.append(f"- Very-high-purity subset: `{len(very_high)}` sentences / `{sum(r['words'] for r in very_high)}` words.\n")
    lines.append(f"- Balanced research slice: `{len(balanced)}` sentences / `{sum(r['words'] for r in balanced)}` words at `{balanced_path}`; this is not a committed training corpus.\n\n")
    lines.append("## Route words by deduplicated candidates\n\n")
    lines.append("| route | sentences | words |\n|---|---:|---:|\n")
    for route in ROUTE_ORDER:
        lines.append(f"| `{route}` | {route_counts.get(route,0)} | {route_words.get(route,0)} |\n")
    lines.append("\n## Source contribution\n\n")
    lines.append("| source | candidates | words |\n|---|---:|---:|\n")
    for label, count in source_counts.most_common():
        lines.append(f"| `{label}` | {count} | {source_words[label]} |\n")
    lines.append("\n## Scientific use\n\n")
    lines.append("- This does not supersede the running FW compact-vs-breadth experiment. First read whether compact recurrence moves EWoK interaction failures and GlobalPIQA hard-row ranks/margins.\n")
    lines.append("- If the FW arms do not move those relational errors, the next efficient route is a small shared-checkpoint factorial using a manually/automatically filtered 50k–200k word subset and/or a corpus-derived four-cell contrast objective; do not commit to 100M from this miner alone.\n")
    lines.append("- Keep channels separate: physical/material, spatial, temporal/quantity, affordance/procedure, and social/agent transitions may support different EWoK/GlobalPIQA subsets.\n\n")
    lines.append("## Files\n\n")
    for k, p in payload["files"].items():
        lines.append(f"- {k}: `{p}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "dedup_candidate_count": payload["dedup_candidate_count"],
        "dedup_candidate_words": payload["dedup_candidate_words"],
        "very_high_candidate_count": payload["very_high_candidate_count"],
        "very_high_candidate_words": payload["very_high_candidate_words"],
        "balanced_200k_words": payload["balanced_200k_slice"]["words"],
        "summary_json": payload["files"]["summary_json"],
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
