#!/usr/bin/env python3
"""research: cost and failure-mode map for stronger post-filters on selected
compact_view_reinvest packets.

This is CPU-only analysis of the already-selected changed block. It does not
change the frozen corpus and does not launch training. It estimates how many
compact packets/words would be removed by mechanically checkable filters inspired
by the research semantic review and current research/29 stability evidence.

Scientific purpose:
  - If temporal/DiD evidence points to treatment-specific Supplement/EWoK fragility,
    future corpus repair should target the semantic failure modes that plausibly
    damage role/relation/force learning, not simply lower compression or retune
    optimization.
  - This script estimates the word-budget cost of such repair and separates core
    vs added sources, to support a matched repair design without
    guessing.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import re
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
DATA_DIR = STUDY / "data" / "density_core_reinvestment_medium_riskhard"
OUT_DIR = STUDY / "data" / "compact_reinvest_filter_cost"
FILES = {
    "core": DATA_DIR / "selected_compact_core_pairs.jsonl",
    "added": DATA_DIR / "selected_compact_added_pairs.jsonl",
    "reinvest": DATA_DIR / "selected_compact_reinvest_pairs.jsonl",
}

MODALS = {
    "may", "might", "can", "could", "should", "would", "possible", "possibly", "perhaps", "likely",
    "unlikely", "sometimes", "often", "usually", "generally", "approximately", "about", "around", "roughly",
    "suggest", "suggests", "suggested", "appear", "appears", "seem", "seems", "probably", "potentially",
}
ATTRIBUTION = {
    "say", "says", "said", "according", "reported", "reports", "report", "claim", "claims", "claimed",
    "researchers", "scientists", "officials", "authors", "study", "studies", "survey", "found", "finds",
}
CAUSAL = {
    "cause", "causes", "caused", "because", "therefore", "thus", "hence", "leads", "led", "leading",
    "result", "results", "resulting", "due", "effect", "affect", "affects", "allows", "requires", "prevents",
    "helps", "helped", "make", "makes", "made", "force", "forces", "forced",
}
NEGATION = {"not", "n't", "never", "no", "none", "without", "neither", "nor", "cannot", "can't", "doesn't", "don't", "didn't"}
PRONOUNS = {"it", "they", "them", "this", "that", "these", "those", "he", "she", "his", "her", "their", "its"}
QUESTION_START = {"who", "what", "when", "where", "why", "how", "did", "do", "does", "can", "could", "is", "are", "was", "were", "will", "would", "should", "has", "have", "had"}


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def toks(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:n't)?|\d+(?:[.,:/-]\d+)*|[%$€£]", text.lower())


def has_any(text: str, vocab: set[str]) -> bool:
    tt = set(toks(text))
    return bool(tt & vocab)


def first_word(text: str) -> str:
    m = re.search(r"[A-Za-z]+", text.lower())
    return m.group(0) if m else ""


def is_question_like(text: str) -> bool:
    return "?" in text or first_word(text) in QUESTION_START


def starts_unresolved_pronoun(text: str) -> bool:
    ts = toks(text)
    return bool(ts and ts[0] in PRONOUNS)


def new_causal_without_source(source: str, rewrite: str) -> bool:
    return (not has_any(source, CAUSAL)) and has_any(rewrite, CAUSAL)


def lost_marker(source: str, rewrite: str, vocab: set[str]) -> bool:
    return has_any(source, vocab) and not has_any(rewrite, vocab)


def gained_marker(source: str, rewrite: str, vocab: set[str]) -> bool:
    return (not has_any(source, vocab)) and has_any(rewrite, vocab)


def feature_flags(r: dict[str, Any]) -> dict[str, bool]:
    src = r.get("source_text", "")
    rew = r.get("rewrite_text", "")
    src_q = is_question_like(src)
    rew_q = is_question_like(rew)
    return {
        "content_recall_lt_0p55": float(r.get("content_recall", 0.0)) < 0.55,
        "content_recall_lt_0p60": float(r.get("content_recall", 0.0)) < 0.60,
        "content_recall_lt_0p65": float(r.get("content_recall", 0.0)) < 0.65,
        "content_recall_lt_0p70": float(r.get("content_recall", 0.0)) < 0.70,
        "length_ratio_lt_0p45": float(r.get("rewrite_words", 0)) / max(float(r.get("source_words", 1)), 1.0) < 0.45,
        "length_ratio_lt_0p50": float(r.get("rewrite_words", 0)) / max(float(r.get("source_words", 1)), 1.0) < 0.50,
        "entity_recall_lt_1": float(r.get("entity_recall", 1.0)) < 1.0,
        "number_recall_lt_1": float(r.get("number_recall", 1.0)) < 1.0,
        "question_force_flip": src_q != rew_q,
        "lost_modal_or_hedge": lost_marker(src, rew, MODALS),
        "lost_attribution": lost_marker(src, rew, ATTRIBUTION),
        "lost_negation": lost_marker(src, rew, NEGATION),
        "gained_negation": gained_marker(src, rew, NEGATION),
        "new_causal_marker_without_source": new_causal_without_source(src, rew),
        "rewrite_starts_unresolved_pronoun": starts_unresolved_pronoun(rew) and not starts_unresolved_pronoun(src),
    }


def summarize_numeric(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "min": None, "p05": None, "p25": None, "p75": None, "p95": None, "max": None}
    s = sorted(xs)
    def q(p: float) -> float:
        if len(s) == 1:
            return s[0]
        idx = p * (len(s) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)
    return {"n": len(xs), "mean": statistics.mean(xs), "median": statistics.median(xs), "min": min(xs), "p05": q(0.05), "p25": q(0.25), "p75": q(0.75), "p95": q(0.95), "max": max(xs)}


def counter_summary(rows: list[dict[str, Any]], labels: Iterable[str]) -> dict[str, Any]:
    c = collections.Counter(labels)
    total = len(rows)
    pair_words_total = sum(int(r.get("pair_words", 0)) for r in rows)
    out = {}
    for k, n in c.most_common():
        pw = sum(int(r.get("pair_words", 0)) for r in rows if k in r.get("_computed_flags", set()))
        out[k] = {"rows": n, "row_fraction": n / total if total else 0.0, "pair_words": pw, "pair_word_fraction": pw / pair_words_total if pair_words_total else 0.0}
    return out


def filter_stats(rows: list[dict[str, Any]], rule: list[str]) -> dict[str, Any]:
    removed = [r for r in rows if any(r["_computed_flags"].get(flag, False) for flag in rule)]
    kept = [r for r in rows if r not in removed]
    total_pw = sum(int(r.get("pair_words", 0)) for r in rows)
    removed_pw = sum(int(r.get("pair_words", 0)) for r in removed)
    return {
        "rule_any_of": rule,
        "removed_rows": len(removed),
        "kept_rows": len(kept),
        "removed_row_fraction": len(removed) / len(rows) if rows else 0.0,
        "removed_pair_words": removed_pw,
        "kept_pair_words": total_pw - removed_pw,
        "removed_pair_word_fraction": removed_pw / total_pw if total_pw else 0.0,
        "kept_pair_word_fraction": (total_pw - removed_pw) / total_pw if total_pw else 0.0,
    }


def summarize_group(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    for r in rows:
        r["_computed_flags"] = feature_flags(r)
    flags = collections.Counter()
    for r in rows:
        flags.update([k for k, v in r["_computed_flags"].items() if v])
    domains = collections.Counter()
    domain_pw = collections.Counter()
    for r in rows:
        dh = r.get("domain_hits") or ["no_domain"]
        if not dh:
            dh = ["no_domain"]
        for d in dh:
            domains[d] += 1
            domain_pw[d] += int(r.get("pair_words", 0))
    total_pw = sum(int(r.get("pair_words", 0)) for r in rows)
    flag_stats = {}
    for flag, count in flags.most_common():
        pw = sum(int(r.get("pair_words", 0)) for r in rows if r["_computed_flags"].get(flag, False))
        flag_stats[flag] = {"rows": count, "row_fraction": count / len(rows) if rows else 0.0, "pair_words": pw, "pair_word_fraction": pw / total_pw if total_pw else 0.0}
    repair_rules = {
        "semantic_force_minimal": ["question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source"],
        "semantic_force_plus_pronoun": ["question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source", "rewrite_starts_unresolved_pronoun"],
        "content_ge_0p60_plus_force": ["content_recall_lt_0p60", "question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source"],
        "content_ge_0p65_plus_force": ["content_recall_lt_0p65", "question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source"],
        "strict_surface_and_force": ["content_recall_lt_0p65", "entity_recall_lt_1", "number_recall_lt_1", "question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source", "rewrite_starts_unresolved_pronoun"],
        "aggressive_content_ge_0p70_surface_force": ["content_recall_lt_0p70", "entity_recall_lt_1", "number_recall_lt_1", "question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source", "rewrite_starts_unresolved_pronoun"],
    }
    return {
        "name": name,
        "rows": len(rows),
        "pair_words": total_pw,
        "source_words": sum(int(r.get("source_words", 0)) for r in rows),
        "rewrite_words": sum(int(r.get("rewrite_words", 0)) for r in rows),
        "length_ratio": summarize_numeric([float(r.get("rewrite_words", 0)) / max(float(r.get("source_words", 1)), 1.0) for r in rows]),
        "content_recall": summarize_numeric([float(r.get("content_recall", 0.0)) for r in rows]),
        "entity_recall": summarize_numeric([float(r.get("entity_recall", 1.0)) for r in rows]),
        "number_recall": summarize_numeric([float(r.get("number_recall", 1.0)) for r in rows]),
        "domain_row_counts": dict(domains.most_common()),
        "domain_pair_words": dict(domain_pw.most_common()),
        "flag_stats": flag_stats,
        "repair_rule_costs": {k: filter_stats(rows, rule) for k, rule in repair_rules.items()},
    }


def example_rows(rows: list[dict[str, Any]], flag: str, n: int = 5) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r["_computed_flags"].get(flag, False):
            out.append({
                "pair_id": r.get("pair_id"),
                "domain_hits": r.get("domain_hits"),
                "source_words": r.get("source_words"),
                "rewrite_words": r.get("rewrite_words"),
                "content_recall": r.get("content_recall"),
                "entity_recall": r.get("entity_recall"),
                "number_recall": r.get("number_recall"),
                "source_text": r.get("source_text"),
                "rewrite_text": r.get("rewrite_text"),
            })
            if len(out) >= n:
                break
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = {name: load_jsonl(path) for name, path in FILES.items()}
    summaries = {name: summarize_group(name, rows) for name, rows in loaded.items()}
    # Cross-overlap sanity: reinvest should be core+added concatenation by key.
    core_keys = {r["key"] for r in loaded["core"]}
    added_keys = {r["key"] for r in loaded["added"]}
    reinvest_keys = {r["key"] for r in loaded["reinvest"]}
    high_value_flags = ["question_force_flip", "lost_modal_or_hedge", "lost_attribution", "new_causal_marker_without_source", "rewrite_starts_unresolved_pronoun", "content_recall_lt_0p60", "entity_recall_lt_1", "number_recall_lt_1"]
    examples = {flag: example_rows(loaded["reinvest"], flag, n=5) for flag in high_value_flags}

    result = {
        "status": "COMPACT_REINVEST_FILTER_COST_ANALYSIS",
        "source_files": {k: str(v) for k, v in FILES.items()},
        "summaries": summaries,
        "overlap_sanity": {
            "core_keys": len(core_keys),
            "added_keys": len(added_keys),
            "reinvest_keys": len(reinvest_keys),
            "core_added_disjoint": len(core_keys & added_keys) == 0,
            "reinvest_equals_core_union_added": reinvest_keys == (core_keys | added_keys),
        },
        "examples_by_flag": examples,
        "interpretation": {
            "what_this_is": "mechanical filter-cost map over selected compact packets; not a semantic-error prevalence estimate",
            "main_repair_signal": "If DiD confirms treatment-specific Supplement/EWoK fragility, a plausible repair is to hard-filter force/attribution/modal/causal/pronoun hazards before reinvesting saved words, while preserving compactness and source multiplier.",
            "risk_of_overfiltering": "Content thresholds above 0.65 may remove many pair words and reduce source-diversity gain; use measured costs before any new training allocation.",
        },
    }
    out_json = OUT_DIR / "compact_reinvest_filter_cost_analysis.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    s = summaries["reinvest"]
    lines = ["# research — compact_view_reinvest selected-packet filter-cost map\n\n"]
    lines.append("CPU-only analysis of the frozen selected compact-reinvest changed block. This does not alter the corpus. It estimates how costly stricter post-filters would be if the temporal DiD points to semantic-view fragility.\n\n")
    lines.append(f"Selected reinvest packets: {s['rows']} pairs, {s['pair_words']} pair words, {s['source_words']} source words, {s['rewrite_words']} rewrite words.\n\n")
    lines.append("## Mechanical hazard flags on selected reinvest packets\n")
    for flag, rec in s["flag_stats"].items():
        lines.append(f"- {flag}: {rec['rows']} rows ({rec['row_fraction']*100:.2f}%), {rec['pair_words']} pair words ({rec['pair_word_fraction']*100:.2f}%)\n")
    lines.append("\n## Candidate repair-rule word costs\n")
    for rule, rec in s["repair_rule_costs"].items():
        lines.append(f"- {rule}: remove {rec['removed_rows']} rows ({rec['removed_row_fraction']*100:.2f}%), {rec['removed_pair_words']} pair words ({rec['removed_pair_word_fraction']*100:.2f}%); keep {rec['kept_pair_words']} pair words\n")
    lines.append("\n## Domain composition by pair words\n")
    for dom, pw in list(s["domain_pair_words"].items())[:12]:
        lines.append(f"- {dom}: {pw} pair words\n")
    lines.append("\n## Interpretation\n")
    for k, v in result["interpretation"].items():
        lines.append(f"- {k}: {v}\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    out_md = OUT_DIR / "compact_reinvest_filter_cost_analysis.md"
    out_md.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "reinvest_rows": s["rows"],
        "reinvest_pair_words": s["pair_words"],
        "top_flags": list(s["flag_stats"].items())[:8],
        "repair_rule_costs": s["repair_rule_costs"],
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
