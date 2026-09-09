#!/usr/bin/env python3
"""research: can mechanically safer compact-view packets refill the reinvest block?

Uses the full analyzed medium compact generation rows (21,465 rows) and the
selected compact_reinvest rows (12,155 selected sources). It estimates whether a
future *matched repair corpus* could replace selected packets rejected by targeted
semantic-force filters without new generation, while preserving the ~423.5k-word
changed-block budget and compact source-diversity idea.

This is only a feasibility/cost map. It does not alter the frozen endpoint and it
is not authorization for new training. A new repaired corpus should be built only
if the exact sparse DiD / official-coordinate evidence shows treatment-specific
semantic fragility worth repairing.
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
MEDIUM_ROWS = STUDY / "data/medium_compact_analysis/medium_compact_ws_rows.jsonl"
SELECTED_REINVEST = STUDY / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
SELECTED_CORE = STUDY / "data/density_core_reinvestment_medium_riskhard/selected_compact_core_pairs.jsonl"
SELECTED_ADDED = STUDY / "data/density_core_reinvestment_medium_riskhard/selected_compact_added_pairs.jsonl"
OUT_DIR = STUDY / "data/filter_refill_feasibility"

MODALS = {"may", "might", "can", "could", "should", "would", "possible", "possibly", "perhaps", "likely", "unlikely", "sometimes", "often", "usually", "generally", "approximately", "about", "around", "roughly", "suggest", "suggests", "suggested", "appear", "appears", "seem", "seems", "probably", "potentially"}
ATTRIBUTION = {"say", "says", "said", "according", "reported", "reports", "report", "claim", "claims", "claimed", "researchers", "scientists", "officials", "authors", "study", "studies", "survey", "found", "finds"}
CAUSAL = {"cause", "causes", "caused", "because", "therefore", "thus", "hence", "leads", "led", "leading", "result", "results", "resulting", "due", "effect", "affect", "affects", "allows", "requires", "prevents", "helps", "helped", "make", "makes", "made", "force", "forces", "forced"}
NEGATION = {"not", "n't", "never", "no", "none", "without", "neither", "nor", "cannot", "can't", "doesn't", "don't", "didn't"}
PRONOUNS = {"it", "they", "them", "this", "that", "these", "those", "he", "she", "his", "her", "their", "its"}
QUESTION_START = {"who", "what", "when", "where", "why", "how", "did", "do", "does", "can", "could", "is", "are", "was", "were", "will", "would", "should", "has", "have", "had"}

RULES = {
    "semantic_force_minimal": ["question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source"],
    "semantic_force_plus_pronoun": ["question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source", "rewrite_starts_unresolved_pronoun"],
    "content_ge_0p60_plus_force": ["content_recall_lt_0p60", "question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source"],
    "content_ge_0p65_plus_force": ["content_recall_lt_0p65", "question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source"],
    "strict_surface_and_force": ["content_recall_lt_0p65", "entity_recall_lt_1", "number_recall_lt_1", "question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source", "rewrite_starts_unresolved_pronoun"],
}


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def toks(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:n't)?|\d+(?:[.,:/-]\d+)*|[%$€£]", text.lower())


def has_any(text: str, vocab: set[str]) -> bool:
    return bool(set(toks(text)) & vocab)


def first_word(text: str) -> str:
    ts = toks(text)
    return ts[0] if ts else ""


def is_question_like(text: str) -> bool:
    return "?" in text or first_word(text) in QUESTION_START


def feature_flags(r: dict[str, Any]) -> dict[str, bool]:
    src = r.get("source_text", "")
    rew = r.get("rewrite_text", "")
    src_q = is_question_like(src)
    rew_q = is_question_like(rew)
    return {
        "content_recall_lt_0p60": float(r.get("content_recall", 0.0)) < 0.60,
        "content_recall_lt_0p65": float(r.get("content_recall", 0.0)) < 0.65,
        "entity_recall_lt_1": float(r.get("entity_recall", 1.0)) < 1.0,
        "number_recall_lt_1": float(r.get("number_recall", 1.0)) < 1.0,
        "question_force_flip": src_q != rew_q,
        "lost_modal_or_hedge": has_any(src, MODALS) and not has_any(rew, MODALS),
        "lost_attribution": has_any(src, ATTRIBUTION) and not has_any(rew, ATTRIBUTION),
        "lost_negation": has_any(src, NEGATION) and not has_any(rew, NEGATION),
        "gained_negation": (not has_any(src, NEGATION)) and has_any(rew, NEGATION),
        "new_causal_marker_without_source": (not has_any(src, CAUSAL)) and has_any(rew, CAUSAL),
        "rewrite_starts_unresolved_pronoun": bool(toks(rew) and toks(rew)[0] in PRONOUNS and not (toks(src) and toks(src)[0] in PRONOUNS)),
    }


def key_of(r: dict[str, Any]) -> str:
    return r.get("key") or f"sid:{r.get('sentence_id')}|doc:{r.get('doc_id')}"


def pair_words(r: dict[str, Any]) -> int:
    return int(r.get("pair_words") or (int(r.get("source_words", 0)) + int(r.get("rewrite_words", 0))))


def accepted(r: dict[str, Any]) -> bool:
    return bool(r.get("accepted_for_next_construction"))


def fails_rule(r: dict[str, Any], rule: list[str]) -> bool:
    flags = r.setdefault("_flags", feature_flags(r))
    return any(flags.get(x, False) for x in rule)


def summarize_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domains = collections.Counter()
    for r in rows:
        for d in (r.get("domain_hits") or ["no_domain"]):
            domains[d] += 1
    return {
        "rows": len(rows),
        "pair_words": sum(pair_words(r) for r in rows),
        "mean_pair_words": statistics.mean([pair_words(r) for r in rows]) if rows else None,
        "domain_counts": dict(domains.most_common()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    medium = load_jsonl(MEDIUM_ROWS)
    selected = load_jsonl(SELECTED_REINVEST)
    core = load_jsonl(SELECTED_CORE)
    added = load_jsonl(SELECTED_ADDED)
    selected_keys = {key_of(r) for r in selected}
    core_keys = {key_of(r) for r in core}
    added_keys = {key_of(r) for r in added}
    accepted_rows = [r for r in medium if accepted(r)]
    unused_rows = [r for r in accepted_rows if key_of(r) not in selected_keys]

    out_rules = {}
    for name, rule in RULES.items():
        removed_selected = [r for r in selected if fails_rule(r, rule)]
        kept_selected = [r for r in selected if not fails_rule(r, rule)]
        passing_unused = [r for r in unused_rows if not fails_rule(r, rule)]
        need_words = sum(pair_words(r) for r in removed_selected)
        avail_words = sum(pair_words(r) for r in passing_unused)
        # Greedy fill using original accepted order. Later materialization would need exact word accounting and row-length controls.
        acc = 0
        chosen = []
        for r in passing_unused:
            if acc >= need_words:
                break
            chosen.append(r)
            acc += pair_words(r)
        out_rules[name] = {
            "rule_any_of": rule,
            "selected_removed": summarize_candidates(removed_selected),
            "selected_kept": summarize_candidates(kept_selected),
            "unused_passing": summarize_candidates(passing_unused),
            "removed_pair_words_to_refill": need_words,
            "unused_passing_pair_words_available": avail_words,
            "available_to_needed_ratio": (avail_words / need_words) if need_words else None,
            "greedy_rows_to_refill_or_exceed": len(chosen),
            "greedy_pair_words": acc,
            "can_refill_from_unused_accepted_without_new_generation": avail_words >= need_words,
            "core_removed_pair_words": sum(pair_words(r) for r in core if fails_rule(r, rule)),
            "added_removed_pair_words": sum(pair_words(r) for r in added if fails_rule(r, rule)),
        }

    result = {
        "status": "FILTER_REFILL_FEASIBILITY",
        "sources": {
            "medium_compact_rows": str(MEDIUM_ROWS),
            "selected_reinvest_pairs": str(SELECTED_REINVEST),
            "selected_core_pairs": str(SELECTED_CORE),
            "selected_added_pairs": str(SELECTED_ADDED),
        },
        "counts": {
            "medium_rows_total": len(medium),
            "accepted_rows_total": len(accepted_rows),
            "selected_rows": len(selected),
            "unused_accepted_rows": len(unused_rows),
            "selected_pair_words": sum(pair_words(r) for r in selected),
            "unused_accepted_pair_words": sum(pair_words(r) for r in unused_rows),
            "selected_equals_core_union_added": selected_keys == (core_keys | added_keys),
            "core_added_disjoint": len(core_keys & added_keys) == 0,
        },
        "rules": out_rules,
        "interpretation": {
            "semantic_force_rules_are_refillable": "The targeted semantic-force filters can be refilled from unused accepted compact rows without new generation if available_to_needed_ratio > 1.",
            "content_thresholds_are_expensive": "Rules with content_recall floors may remove much of the selected compact block and could weaken source-diversity gains even if refillable.",
            "not_training_authorization": "Use this only after exact DiD/full-vector evidence shows semantic repair is worth a new training allocation.",
        },
    }
    out_json = OUT_DIR / "filter_refill_feasibility.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research — compact-view filter refill feasibility\n\n"]
    c = result["counts"]
    lines.append(f"Accepted medium compact rows: {c['accepted_rows_total']}; selected reinvest rows: {c['selected_rows']} ({c['selected_pair_words']} pair words); unused accepted rows: {c['unused_accepted_rows']} ({c['unused_accepted_pair_words']} pair words).\n\n")
    lines.append("## Rule feasibility\n")
    for name, rec in out_rules.items():
        rem = rec["selected_removed"]
        avail = rec["unused_passing"]
        lines.append(
            f"- {name}: remove {rem['rows']} selected rows / {rem['pair_words']} pair words; "
            f"unused passing pool {avail['rows']} rows / {avail['pair_words']} pair words; "
            f"available:needed={rec['available_to_needed_ratio']:.2f}; refill={rec['can_refill_from_unused_accepted_without_new_generation']}; "
            f"core_removed_words={rec['core_removed_pair_words']}, added_removed_words={rec['added_removed_pair_words']}\n"
        )
    lines.append("\n## Interpretation\n")
    for k, v in result["interpretation"].items():
        lines.append(f"- {k}: {v}\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    out_md = OUT_DIR / "filter_refill_feasibility.md"
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "counts": result["counts"],
        "rule_ratios": {name: rec["available_to_needed_ratio"] for name, rec in out_rules.items()},
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
