#!/usr/bin/env python3
"""research: semantic-force repair frontier for compact-view reinvestment.

CPU-only analysis over already generated compact rows. It quantifies which
post-generation row-removal rules are mechanically refillable from unused accepted
compact rows, and how much of the density/reinvestment block they would replace.
It does not change any corpus or authorize training.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import re
import statistics
from typing import Any, Callable

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
MEDIUM_ROWS = STUDY / "data/medium_compact_analysis/medium_compact_ws_rows.jsonl"
SELECTED_REINVEST = STUDY / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
SELECTED_CORE = STUDY / "data/density_core_reinvestment_medium_riskhard/selected_compact_core_pairs.jsonl"
SELECTED_ADDED = STUDY / "data/density_core_reinvestment_medium_riskhard/selected_compact_added_pairs.jsonl"
OUT_DIR = STUDY / "data/semantic_repair_frontier"

MODALS = {"may", "might", "can", "could", "should", "would", "possible", "possibly", "perhaps", "likely", "unlikely", "sometimes", "often", "usually", "generally", "approximately", "about", "around", "roughly", "suggest", "suggests", "suggested", "appear", "appears", "seem", "seems", "probably", "potentially"}
ATTRIBUTION = {"say", "says", "said", "according", "reported", "reports", "report", "claim", "claims", "claimed", "researchers", "scientists", "officials", "authors", "study", "studies", "survey", "found", "finds"}
CAUSAL = {"cause", "causes", "caused", "because", "therefore", "thus", "hence", "leads", "led", "leading", "result", "results", "resulting", "due", "effect", "affect", "affects", "allows", "requires", "prevents", "helps", "helped", "make", "makes", "made", "force", "forces", "forced"}
NEGATION = {"not", "n't", "never", "no", "none", "without", "neither", "nor", "cannot", "can't", "doesn't", "don't", "didn't"}
PRONOUNS = {"it", "they", "them", "this", "that", "these", "those", "he", "she", "his", "her", "their", "its"}
QUESTION_START = {"who", "what", "when", "where", "why", "how", "did", "do", "does", "can", "could", "is", "are", "was", "were", "will", "would", "should", "has", "have", "had"}
FORCE_FLAGS = ["question_force_flip", "lost_modal_or_hedge", "lost_attribution", "lost_negation", "gained_negation", "new_causal_marker_without_source"]
PRONOUN_FLAG = "rewrite_starts_unresolved_pronoun"
SURFACE_FLAGS = ["entity_recall_lt_1", "number_recall_lt_1"]


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    src = r.get("source_text", "") or ""
    rew = r.get("rewrite_text", "") or ""
    src_q = is_question_like(src)
    rew_q = is_question_like(rew)
    return {
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


def content_recall(r: dict[str, Any]) -> float:
    return float(r.get("content_recall", 0.0))


def accepted(r: dict[str, Any]) -> bool:
    return bool(r.get("accepted_for_next_construction"))


def primary_domain(r: dict[str, Any]) -> str:
    hits = r.get("domain_hits") or []
    return hits[0] if hits else "no_domain"


def all_domains(r: dict[str, Any]) -> list[str]:
    hits = r.get("domain_hits") or []
    return list(hits) if hits else ["no_domain"]


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pws = [pair_words(r) for r in rows]
    flags = [r.setdefault("_flags", feature_flags(r)) for r in rows]
    def frac_flag(name: str) -> float | None:
        return mean([1.0 if f.get(name, False) else 0.0 for f in flags])
    primary = collections.Counter()
    primary_pw = collections.Counter()
    multi_pw = collections.Counter()
    for r in rows:
        pd = primary_domain(r)
        primary[pd] += 1
        primary_pw[pd] += pair_words(r)
        for d in all_domains(r):
            multi_pw[d] += pair_words(r)
    return {
        "rows": len(rows),
        "pair_words": sum(pws),
        "mean_pair_words": mean([float(x) for x in pws]),
        "mean_content_recall": mean([content_recall(r) for r in rows]),
        "mean_entity_recall": mean([float(r.get("entity_recall", 1.0)) for r in rows]),
        "mean_number_recall": mean([float(r.get("number_recall", 1.0)) for r in rows]),
        "mean_length_ratio": mean([float(r.get("length_ratio", r.get("rewrite_words", 0) / max(1, r.get("source_words", 1)))) for r in rows]),
        "flag_fractions": {name: frac_flag(name) for name in FORCE_FLAGS + [PRONOUN_FLAG] + SURFACE_FLAGS},
        "primary_domain_rows": dict(primary.most_common()),
        "primary_domain_pair_words": dict(primary_pw.most_common()),
        "multi_domain_pair_words": dict(multi_pw.most_common()),
    }


def make_rule(name: str, content_floor: float | None, include_pronoun: bool, include_surface: bool) -> tuple[str, Callable[[dict[str, Any]], bool], dict[str, Any]]:
    def fails(r: dict[str, Any]) -> bool:
        flags = r.setdefault("_flags", feature_flags(r))
        if any(flags.get(x, False) for x in FORCE_FLAGS):
            return True
        if include_pronoun and flags.get(PRONOUN_FLAG, False):
            return True
        if include_surface and any(flags.get(x, False) for x in SURFACE_FLAGS):
            return True
        if content_floor is not None and content_recall(r) < content_floor:
            return True
        return False
    desc = {"content_floor": content_floor, "force_flags": FORCE_FLAGS, "include_unresolved_pronoun": include_pronoun, "include_entity_number_surface": include_surface}
    return name, fails, desc


def domain_word_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter()
    for r in rows:
        c[primary_domain(r)] += pair_words(r)
    return dict(c.most_common())


def domain_availability_ratio(removed: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    rem = collections.Counter()
    av = collections.Counter()
    for r in removed:
        rem[primary_domain(r)] += pair_words(r)
    for r in candidates:
        av[primary_domain(r)] += pair_words(r)
    domains = sorted(set(rem) | set(av))
    rows = []
    bottlenecks = []
    for d in domains:
        need = rem[d]
        have = av[d]
        ratio = (have / need) if need else None
        rec = {"domain": d, "removed_pair_words": need, "unused_passing_pair_words": have, "ratio": ratio}
        rows.append(rec)
        if need and have < need:
            bottlenecks.append(rec)
    return {"by_primary_domain": rows, "bottlenecks": bottlenecks}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    medium = load_jsonl(MEDIUM_ROWS)
    selected = load_jsonl(SELECTED_REINVEST)
    core = load_jsonl(SELECTED_CORE)
    added = load_jsonl(SELECTED_ADDED)
    selected_keys = {key_of(r) for r in selected}
    accepted_rows = [r for r in medium if accepted(r)]
    unused_rows = [r for r in accepted_rows if key_of(r) not in selected_keys]
    selected_pair_total = sum(pair_words(r) for r in selected)

    rules = []
    rules.append(make_rule("force_only", None, False, False))
    rules.append(make_rule("force_plus_pronoun", None, True, False))
    for floor in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        rules.append(make_rule(f"force_content_ge_{floor:.2f}", floor, False, False))
    for floor in [0.45, 0.50, 0.55, 0.60, 0.65]:
        rules.append(make_rule(f"force_pronoun_content_ge_{floor:.2f}", floor, True, False))
    for floor in [None, 0.55, 0.60, 0.65]:
        suffix = "none" if floor is None else f"{floor:.2f}"
        rules.append(make_rule(f"force_surface_content_ge_{suffix}", floor, False, True))

    out_rules = []
    csv_rows = []
    for name, fails, desc in rules:
        removed = [r for r in selected if fails(r)]
        kept = [r for r in selected if not fails(r)]
        passing_unused = [r for r in unused_rows if not fails(r)]
        need_words = sum(pair_words(r) for r in removed)
        have_words = sum(pair_words(r) for r in passing_unused)
        ratio = have_words / need_words if need_words else None
        final_if_refilled = selected_pair_total if have_words >= need_words else (sum(pair_words(r) for r in kept) + have_words)
        rec = {
            "rule": name,
            "description": desc,
            "selected_removed": summarize_rows(removed),
            "selected_kept": summarize_rows(kept),
            "unused_passing": summarize_rows(passing_unused),
            "removed_pair_words_to_refill": need_words,
            "unused_passing_pair_words_available": have_words,
            "available_to_needed_ratio": ratio,
            "can_refill_pair_words_from_unused_accepted": have_words >= need_words,
            "selected_pair_words_after_unlimited_refill_cap": final_if_refilled,
            "selected_pair_word_fraction_removed": need_words / selected_pair_total,
            "domain_availability": domain_availability_ratio(removed, passing_unused),
            "core_removed_pair_words": sum(pair_words(r) for r in core if fails(r)),
            "added_removed_pair_words": sum(pair_words(r) for r in added if fails(r)),
        }
        out_rules.append(rec)
        csv_rows.append({
            "rule": name,
            "content_floor": desc["content_floor"],
            "include_pronoun": desc["include_unresolved_pronoun"],
            "include_surface": desc["include_entity_number_surface"],
            "removed_rows": rec["selected_removed"]["rows"],
            "removed_pair_words": need_words,
            "removed_fraction": rec["selected_pair_word_fraction_removed"],
            "unused_passing_pair_words": have_words,
            "available_to_needed_ratio": ratio,
            "can_refill": have_words >= need_words,
            "mean_content_removed": rec["selected_removed"]["mean_content_recall"],
            "mean_content_unused_passing": rec["unused_passing"]["mean_content_recall"],
            "domain_bottleneck_count": len(rec["domain_availability"]["bottlenecks"]),
        })

    result = {
        "status": "SEMANTIC_REPAIR_FRONTIER",
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
            "selected_pair_words": selected_pair_total,
            "unused_accepted_rows": len(unused_rows),
            "unused_accepted_pair_words": sum(pair_words(r) for r in unused_rows),
        },
        "rules": out_rules,
        "scientific_read": {
            "route_value": "If exact temporal treatment analysis later supports semantic fragility, this frontier identifies repair strengths that preserve the compactness/source-diversity mechanism without new generation.",
            "main_pattern": "Targeted force/modal/attribution/causal/negation rules are mechanically refillable; adding broad content floors quickly exhausts the unused passing pool.",
            "non_authorization": "This is not a training allocation; it prepares a cheaper fork only if existing-artifact evidence selects semantic repair over optimization or consolidation repair.",
        },
    }

    out_json = OUT_DIR / "semantic_repair_frontier.json"
    out_csv = OUT_DIR / "semantic_repair_frontier.csv"
    out_md = OUT_DIR / "semantic_repair_frontier.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # Human-readable compact table sorted by rule severity.
    lines = ["# research — semantic-force repair frontier\n\n"]
    c = result["counts"]
    lines.append(f"Selected compact-reinvest block: {c['selected_rows']} rows / {c['selected_pair_words']} pair words. Unused accepted compact pool: {c['unused_accepted_rows']} rows / {c['unused_accepted_pair_words']} pair words.\n\n")
    lines.append("## Rule frontier\n")
    for row in csv_rows:
        ratio_s = "NA" if row["available_to_needed_ratio"] is None else f"{row['available_to_needed_ratio']:.2f}"
        lines.append(
            f"- {row['rule']}: removes {row['removed_pair_words']} pair words ({100*row['removed_fraction']:.2f}%); "
            f"unused passing {row['unused_passing_pair_words']} words; available:needed={ratio_s}; "
            f"refill={row['can_refill']}; domain shortfalls={row['domain_bottleneck_count']}\n"
        )
    lines.append("\n## Read for future repair\n")
    lines.append("- The strongest refillable repairs are force_only and force_plus_pronoun; they keep the density intervention mechanically feasible without new teacher generation.\n")
    lines.append("- Content-recall floors at 0.60 or above are not refillable from the unused accepted pool and would also erase much of the compressed-view block.\n")
    lines.append("- If a later exact temporal comparison selects semantic repair, the next construction should use a domain-aware replacement list rather than simple greedy refill, because preserving physical/causal exposure matters for EWoK.\n")
    lines.append(f"\nMachine-readable output: `{out_json}`; table: `{out_csv}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "counts": result["counts"],
        "refillable_rules": [r["rule"] for r in out_rules if r["can_refill_pair_words_from_unused_accepted"]],
        "non_refillable_content_rules": [r["rule"] for r in out_rules if (not r["can_refill_pair_words_from_unused_accepted"] and "content" in r["rule"])],
        "out_json": str(out_json),
        "out_csv": str(out_csv),
        "out_md": str(out_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
