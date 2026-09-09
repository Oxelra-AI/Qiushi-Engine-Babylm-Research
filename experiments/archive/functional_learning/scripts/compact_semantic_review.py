#!/usr/bin/env python3
"""research: semantic risk review for Qwen-internal compact rewrites.

The research pilot showed real shortening, but word savings and low Jaccard overlap do
not establish that the compact text is a useful second view.  This script turns the
pilot into a research-facing semantic review packet and a conservative automatic
admission screen.  The screen is intentionally cautious: high-risk rows keep the
existing rewrite unless an independent semantic review says the compact version keeps
who did what, under what temporal/logical force, and in which order.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

STUDY = Path("experiments/archive/functional_learning")
DEFAULT_INPUT = STUDY / "data/compact_pilot/pilot_generation_joined.jsonl"
DEFAULT_PAIR_ANALYSIS = STUDY / "data/compact_pilot/pilot_pair_analysis.jsonl"
OUT_DIR = STUDY / "data/compact_semantic_review"

KNOWN_ALTERED = {
    "rw2s0_029078": "navigation order reversed near Goddess Mother temple and idol makers",
    "rw_035551": "conditional self-attribution converted toward a direct assertion",
}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
NUM_RE = re.compile(r"\b\d+(?:[.,:]\d+)*(?:st|nd|rd|th)?\b", re.I)

PATTERNS = {
    "negation": re.compile(r"\b(not|n't|no|never|neither|nor|nobody|nothing|nowhere|none|cannot|can't|won't|wouldn't|shouldn't|couldn't|didn't|doesn't|don't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b", re.I),
    "modal": re.compile(r"\b(can|could|may|might|must|shall|should|will|would|ought|need|needs|needed)\b", re.I),
    "conditional": re.compile(r"\b(if|unless|provided|whether|suppose|assuming|would|could|might|should|were)\b", re.I),
    "causal_contrast": re.compile(r"\b(because|since|therefore|thus|hence|so|consequently|although|though|however|but|yet|whereas|while)\b", re.I),
    "temporal_order": re.compile(r"\b(before|after|until|during|meanwhile|then|finally|first|second|third|next|later|already|still|now|once|again|earlier|previously|subsequently)\b", re.I),
    "navigation": re.compile(r"\b(left|right|straight|opposite|turn|take|street|road|avenue|north|south|east|west|up|down|toward|towards|past|near|at|from|to|through|around|across|behind|front)\b", re.I),
    "comparison": re.compile(r"\b(more|less|most|least|better|worse|best|worst|rather|than|as\s+as|bigger|smaller|larger|higher|lower|faster|slower|older|newer|longer|shorter)\b", re.I),
    "role_pronoun": re.compile(r"\b(he|she|they|him|her|them|his|their|hers|theirs|himself|herself|themselves|i|me|my|mine|we|us|our|you|your|who|whom|whose)\b", re.I),
    "speech_question": re.compile(r"[\?\"]|\b(said|says|told|asked|replied|answered|called)\b", re.I),
}


def words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def word_count(text: str) -> int:
    return len((text or "").strip().split())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def marker_counts(text: str) -> Dict[str, int]:
    return {name: len(pat.findall(text or "")) for name, pat in PATTERNS.items()}


def marker_preserved(name: str, original: str, compact: str) -> float:
    pat = PATTERNS[name]
    orig = [x.lower() if isinstance(x, str) else str(x).lower() for x in pat.findall(original or "")]
    comp = [x.lower() if isinstance(x, str) else str(x).lower() for x in pat.findall(compact or "")]
    if not orig:
        return 1.0
    co = Counter(orig)
    cc = Counter(comp)
    return sum(min(co[k], cc[k]) for k in co) / sum(co.values())


def direction_sequence(text: str) -> List[str]:
    toks = words(text)
    seq = []
    for t in toks:
        if t in {"left", "right", "straight", "opposite", "north", "south", "east", "west", "up", "down"}:
            seq.append(t)
    return seq


def number_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in NUM_RE.finditer(text or "")}


def has_number_loss(original: str, compact: str) -> bool:
    nums = number_set(original)
    if not nums:
        return False
    comp = (compact or "").lower()
    return any(n not in comp for n in nums)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def risk_for_record(rec: Dict[str, Any], pair_analysis: Dict[str, Any] | None = None) -> Dict[str, Any]:
    pid = rec["pair_id"]
    original = rec.get("original", "")
    current = rec.get("current_rewrite", "")
    compact = rec.get("compact_rewrite", "")
    ow, cw, kw = word_count(original), word_count(current), word_count(compact)
    comp_orig = jaccard(words(compact), words(original))
    comp_curr = jaccard(words(compact), words(current))
    curr_orig = jaccard(words(current), words(original))
    m_orig = marker_counts(original)
    m_comp = marker_counts(compact)
    tags: List[str] = []
    reasons: List[str] = []
    risk = 0

    if pid in KNOWN_ALTERED:
        tags.append("known_altered_claim")
        reasons.append(KNOWN_ALTERED[pid])
        risk += 100

    if not compact.strip():
        tags.append("empty_compact")
        reasons.append("no compact output")
        risk += 100

    if kw >= cw:
        tags.append("no_word_saving")
        reasons.append(f"compact words {kw} >= current rewrite words {cw}")
        risk += 30

    if kw < 8 or (ow > 0 and kw / max(ow, 1) < 0.30):
        tags.append("severe_summary")
        reasons.append(f"compact length {kw} words, original {ow} words")
        risk += 25
    elif ow > 0 and kw / max(ow, 1) < 0.42:
        tags.append("strong_summary")
        reasons.append(f"compact/original length ratio {kw / max(ow, 1):.2f}")
        risk += 10

    if has_number_loss(original, compact):
        tags.append("number_changed_or_removed")
        reasons.append("one or more source numbers not found in compact text")
        risk += 60

    # Navigation/order is fragile because anchors can be preserved while roles swap.
    nav_seq_orig = direction_sequence(original)
    nav_seq_comp = direction_sequence(compact)
    if m_orig["navigation"] >= 5 or len(nav_seq_orig) >= 3:
        tags.append("ordered_navigation")
        risk += 35
        if nav_seq_orig and nav_seq_comp != nav_seq_orig[: len(nav_seq_comp)]:
            tags.append("navigation_sequence_changed")
            reasons.append(f"direction sequence original={nav_seq_orig} compact={nav_seq_comp}")
            risk += 35

    # Conditional/modality/negation/contrast losses are central to semantic force.
    for name, label, add in [
        ("conditional", "conditional_force", 30),
        ("modal", "modality_force", 25),
        ("negation", "negation_force", 35),
        ("causal_contrast", "causal_or_contrast_force", 20),
        ("temporal_order", "temporal_order", 20),
        ("comparison", "comparison_relation", 20),
    ]:
        if m_orig[name] > 0:
            pres = marker_preserved(name, original, compact)
            if pres < 0.50:
                tags.append(label + "_weakened")
                reasons.append(f"{label} marker preservation {pres:.2f}; original count {m_orig[name]}, compact count {m_comp[name]}")
                risk += add
            elif pres < 1.0:
                tags.append(label + "_partly_changed")
                risk += max(5, add // 3)

    if m_orig["speech_question"] > 0 or "?" in original:
        tags.append("dialogue_or_question")
        risk += 12
        # When dialogue is compressed heavily, add risk because speaker/attitude can be lost.
        if ow > 0 and kw / max(ow, 1) < 0.55:
            tags.append("compressed_dialogue")
            risk += 12

    if m_orig["role_pronoun"] >= 3:
        pres = marker_preserved("role_pronoun", original, compact)
        if pres < 0.35:
            tags.append("role_reference_weakened")
            reasons.append(f"role/pronoun preservation {pres:.2f}")
            risk += 20

    if comp_orig < 0.22 and ow >= 18:
        tags.append("very_low_source_overlap")
        reasons.append(f"compact-original Jaccard {comp_orig:.2f}")
        risk += 18
    if comp_curr < 0.16 and cw >= 18:
        tags.append("very_low_current_overlap")
        reasons.append(f"compact-current Jaccard {comp_curr:.2f}")
        risk += 12

    # The current rewrite may itself be the safer reference for some structural force;
    # if compact loses both source and current markers, strengthen the warning.
    cur_m = marker_counts(current)
    for name in ["conditional", "modal", "negation", "temporal_order", "comparison"]:
        if m_orig[name] > 0 and cur_m[name] > 0 and m_comp[name] == 0:
            tag = f"{name}_lost_from_both_references"
            if tag not in tags:
                tags.append(tag)
                risk += 10

    auto_admit = (
        risk < 25
        and kw >= 8
        and kw < cw
        and not has_number_loss(original, compact)
        and pid not in KNOWN_ALTERED
    )
    needs_review = (not auto_admit) and kw > 0 and kw < cw and pid not in KNOWN_ALTERED and not has_number_loss(original, compact)

    return {
        "pair_id": pid,
        "source": rec.get("source", ""),
        "example_id": rec.get("example_id"),
        "current_words": cw,
        "compact_words": kw,
        "original_words": ow,
        "saved_words_if_used": max(0, cw - kw),
        "compact_original_jaccard": round(comp_orig, 4),
        "current_original_jaccard": round(curr_orig, 4),
        "compact_current_jaccard": round(comp_curr, 4),
        "risk_score": risk,
        "risk_tags": sorted(set(tags)),
        "risk_reasons": reasons[:8],
        "auto_admit_low_risk": auto_admit,
        "needs_semantic_review": needs_review,
        "keep_current_rewrite": not auto_admit,
        "known_altered": pid in KNOWN_ALTERED,
        "original": original,
        "current_rewrite": current,
        "compact_rewrite": compact,
    }


def pick_review_sample(screened: List[Dict[str, Any]], n_low: int = 20, n_mid: int = 20, n_high: int = 40, seed: int = 54054) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    known = [r for r in screened if r["known_altered"]]
    high = [r for r in screened if r["risk_score"] >= 55 and not r["known_altered"]]
    mid = [r for r in screened if 25 <= r["risk_score"] < 55 and not r["known_altered"]]
    low = [r for r in screened if r["risk_score"] < 25 and not r["known_altered"]]

    def take_top(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        rows = sorted(rows, key=lambda r: (-r["risk_score"], -r["saved_words_if_used"], r["pair_id"]))
        return rows[:n]

    def take_random(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        rows = list(rows)
        rng.shuffle(rows)
        return rows[:n]

    sample = []
    sample.extend(known)
    sample.extend(take_top(high, n_high))
    sample.extend(take_top(mid, n_mid))
    sample.extend(take_random(low, n_low))

    seen = set()
    out = []
    for r in sample:
        if r["pair_id"] in seen:
            continue
        seen.add(r["pair_id"])
        rr = {k: r[k] for k in [
            "pair_id", "source", "example_id", "original_words", "current_words", "compact_words",
            "saved_words_if_used", "risk_score", "risk_tags", "risk_reasons",
            "original", "current_rewrite", "compact_rewrite"
        ]}
        out.append(rr)
    return out


def write_packet(path: Path, sample: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# research compact-view semantic review packet\n\n")
        f.write("Purpose: distinguish faithful shortening from information-losing summary and altered source meaning before any compact text is admitted to learner training.\n\n")
        f.write("## Pilot-level summary\n\n")
        for k in ["n_records", "current_number_length_pass", "auto_low_risk", "needs_semantic_review", "known_altered", "auto_low_risk_saved_words", "current_number_length_saved_words"]:
            f.write(f"- {k}: {summary.get(k)}\n")
        f.write("\n## Review instructions\n\n")
        f.write("For each item, compare the compact rewrite primarily with the original source text, using the current rewrite only as an inherited second view. Label it as one of: faithful_shortening, supported_summary_with_lost_detail, altered_meaning, or unclear. Pay special attention to entity roles, numbers, direction/order, conditional force, modality, negation, temporal relations, and who did what.\n\n")
        for i, r in enumerate(sample, 1):
            f.write(f"## {i}. {r['pair_id']} · source={r['source']} · risk={r['risk_score']} · saved={r['saved_words_if_used']}\n\n")
            f.write(f"Risk tags: {', '.join(r['risk_tags']) if r['risk_tags'] else 'none'}\n\n")
            if r.get("risk_reasons"):
                f.write("Risk reasons: " + " | ".join(r["risk_reasons"]) + "\n\n")
            f.write("Original:\n")
            f.write(r["original"].strip() + "\n\n")
            f.write("Current inherited rewrite:\n")
            f.write(r["current_rewrite"].strip() + "\n\n")
            f.write("Compact candidate:\n")
            f.write(r["compact_rewrite"].strip() + "\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--pair-analysis", default=str(DEFAULT_PAIR_ANALYSIS))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=54054)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.input))
    pair_rows = {r["pair_id"]: r for r in load_jsonl(Path(args.pair_analysis))} if Path(args.pair_analysis).exists() else {}
    screened = [risk_for_record(r, pair_rows.get(r.get("pair_id"))) for r in records]

    current_number_length = [
        r for r in screened
        if r["compact_words"] >= 8
        and r["compact_words"] < r["current_words"]
        and not any(t == "number_changed_or_removed" for t in r["risk_tags"])
    ]
    low = [r for r in screened if r["auto_admit_low_risk"]]
    needs = [r for r in screened if r["needs_semantic_review"]]
    known = [r for r in screened if r["known_altered"]]

    tag_counts = Counter(t for r in screened for t in r["risk_tags"])
    source_counts = Counter(r["source"] for r in screened)
    source_low = Counter(r["source"] for r in low)
    risk_bins = Counter()
    for r in screened:
        s = r["risk_score"]
        if s < 25:
            risk_bins["low_lt25"] += 1
        elif s < 55:
            risk_bins["medium_25_54"] += 1
        elif s < 100:
            risk_bins["high_55_99"] += 1
        else:
            risk_bins["known_or_extreme_ge100"] += 1

    summary = {
        "status": "COMPACT_SEMANTIC_REVIEW_PREPARED",
        "n_records": len(screened),
        "current_number_length_pass": len(current_number_length),
        "current_number_length_saved_words": sum(r["saved_words_if_used"] for r in current_number_length),
        "auto_low_risk": len(low),
        "auto_low_risk_saved_words": sum(r["saved_words_if_used"] for r in low),
        "needs_semantic_review": len(needs),
        "needs_semantic_review_saved_words": sum(r["saved_words_if_used"] for r in needs),
        "known_altered": len(known),
        "known_altered_ids": {r["pair_id"]: KNOWN_ALTERED.get(r["pair_id"], "") for r in known},
        "risk_bins": dict(risk_bins),
        "top_risk_tags": tag_counts.most_common(30),
        "source_counts": dict(source_counts),
        "auto_low_risk_by_source": dict(source_low),
        "input": str(args.input),
    }

    with (out_dir / "semantic_risk_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out_dir / "semantic_screen_pilot.jsonl").open("w", encoding="utf-8") as f:
        for r in screened:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sample = pick_review_sample(screened, seed=args.seed)
    with (out_dir / "semantic_review_sample.jsonl").open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_packet(out_dir / "semantic_review_packet.md", sample, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {out_dir / 'semantic_screen_pilot.jsonl'}", flush=True)
    print(f"Wrote {out_dir / 'semantic_review_packet.md'} with {len(sample)} examples", flush=True)


if __name__ == "__main__":
    main()
