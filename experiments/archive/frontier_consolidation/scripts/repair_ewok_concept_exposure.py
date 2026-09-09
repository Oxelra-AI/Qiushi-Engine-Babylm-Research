#!/usr/bin/env python3
"""research: EWoK concept lexical exposure in candidate semantic repairs.

Compares the frozen selected compact_view_reinvest block with candidate
force-based refill plans. This is a weak lexical proxy for proposition-bearing
exposure, not a semantic-contamination or causal proof. Its purpose is to catch
repairs that preserve word count but remove material/physical/spatial relation
vocabulary needed by EWoK.
"""
from __future__ import annotations

import collections
import csv
import importlib.util
import json
import pathlib
import re
from typing import Any, Callable

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
FRONTIER_SCRIPT = STUDY / "scripts/semantic_repair_frontier.py"
DATA_DIR = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
PLAN_DIR = STUDY / "data/domain_aware_refill_plan"
DID_PATH = STUDY / "data/official_ewok_2x2_domain_did/official_ewok_2x2_domain_did.json"
OUT_DIR = STUDY / "data/repair_ewok_concept_exposure"

spec = importlib.util.spec_from_file_location("frontier", FRONTIER_SCRIPT)
frontier = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(frontier)  # type: ignore[arg-type]

RULES = {
    "force_only": (None, False, False),
    "force_plus_pronoun": (None, True, False),
    "force_surface": (None, False, True),
}


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return frontier.load_jsonl(path)


def key_of(r: dict[str, Any]) -> str:
    return frontier.key_of(r)


def pair_words(r: dict[str, Any]) -> int:
    return frontier.pair_words(r)


def make_rule(content_floor: float | None, include_pronoun: bool, include_surface: bool) -> Callable[[dict[str, Any]], bool]:
    _, fails, _ = frontier.make_rule("tmp", content_floor, include_pronoun, include_surface)
    return fails


def text_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+(?:n't)?|\d+(?:[.,:/-]\d+)*", (text or "").lower()))


def load_ewok_concepts() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for p in sorted(DATA_DIR.glob("*.jsonl")):
        concepts = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for k in ["ConceptA", "ConceptB"]:
                val = str(r.get(k, "")).lower().strip()
                for tok in re.findall(r"[a-z]+(?:n't)?|\d+(?:[.,:/-]\d+)*", val):
                    if len(tok) >= 2:
                        concepts.add(tok)
        out[p.stem] = concepts
    return out


def summarize_block(rows: list[dict[str, Any]], concepts_by_domain: dict[str, set[str]]) -> dict[str, Any]:
    # Count row-level and occurrence-like token hits in source/rewrite text.
    block_words = sum(pair_words(r) for r in rows)
    result = {}
    for domain, concepts in concepts_by_domain.items():
        covered = set()
        src_hit_rows = 0
        rew_hit_rows = 0
        either_hit_rows = 0
        src_occ = 0
        rew_occ = 0
        for r in rows:
            src_toks = text_tokens(r.get("source_text", ""))
            rew_toks = text_tokens(r.get("rewrite_text", ""))
            src_hits = src_toks & concepts
            rew_hits = rew_toks & concepts
            if src_hits:
                src_hit_rows += 1
            if rew_hits:
                rew_hit_rows += 1
            if src_hits or rew_hits:
                either_hit_rows += 1
            covered |= src_hits | rew_hits
            src_occ += len(src_hits)
            rew_occ += len(rew_hits)
        total = len(concepts)
        result[domain] = {
            "concept_count": total,
            "unique_concepts_covered": len(covered),
            "coverage_fraction": len(covered) / total if total else None,
            "source_unique_token_hits": src_occ,
            "rewrite_unique_token_hits": rew_occ,
            "either_hit_rows": either_hit_rows,
            "source_hit_rows": src_hit_rows,
            "rewrite_hit_rows": rew_hit_rows,
            "pair_words": block_words,
            "covered_concepts_sample": sorted(covered)[:50],
        }
    return result


def diff_domain(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for d in sorted(a):
        aa = a[d]
        bb = b[d]
        rows.append({
            "domain": d,
            "coverage_delta": bb["coverage_fraction"] - aa["coverage_fraction"],
            "unique_concepts_delta": bb["unique_concepts_covered"] - aa["unique_concepts_covered"],
            "source_hit_rows_delta": bb["source_hit_rows"] - aa["source_hit_rows"],
            "rewrite_hit_rows_delta": bb["rewrite_hit_rows"] - aa["rewrite_hit_rows"],
            "either_hit_rows_delta": bb["either_hit_rows"] - aa["either_hit_rows"],
            "source_unique_token_hits_delta": bb["source_unique_token_hits"] - aa["source_unique_token_hits"],
            "rewrite_unique_token_hits_delta": bb["rewrite_unique_token_hits"] - aa["rewrite_unique_token_hits"],
        })
    rows.sort(key=lambda x: (x["coverage_delta"], x["either_hit_rows_delta"]))
    return rows


def load_replacement_keys(rule: str) -> set[str]:
    p = PLAN_DIR / f"{rule}_replacement_rows.csv"
    keys = set()
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add(row["key"])
    return keys


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = load_jsonl(frontier.SELECTED_REINVEST)
    medium = load_jsonl(frontier.MEDIUM_ROWS)
    medium_by_key = {key_of(r): r for r in medium}
    concepts = load_ewok_concepts()
    did = json.loads(DID_PATH.read_text(encoding="utf-8"))
    original_summary = summarize_block(selected, concepts)

    rule_results = {}
    for rule, args in RULES.items():
        fails = make_rule(*args)
        kept = [r for r in selected if not fails(r)]
        replacement_keys = load_replacement_keys(rule)
        replacements = [medium_by_key[k] for k in replacement_keys if k in medium_by_key]
        final = kept + replacements
        final_summary = summarize_block(final, concepts)
        removed_summary = summarize_block([r for r in selected if fails(r)], concepts)
        repl_summary = summarize_block(replacements, concepts)
        rule_results[rule] = {
            "kept_rows": len(kept),
            "replacement_rows": len(replacements),
            "final_rows": len(final),
            "kept_pair_words": sum(pair_words(r) for r in kept),
            "replacement_pair_words": sum(pair_words(r) for r in replacements),
            "final_pair_words": sum(pair_words(r) for r in final),
            "removed_summary_by_ewok_domain": removed_summary,
            "replacement_summary_by_ewok_domain": repl_summary,
            "final_summary_by_ewok_domain": final_summary,
            "final_minus_original_by_ewok_domain": diff_domain(original_summary, final_summary),
        }

    # Align the exposure deltas to official EWoK DiD domains.
    did_domains = {r["domain"]: r for r in did["domain_summary_sorted_by_DiD"]}
    aligned = {}
    for rule, rr in rule_results.items():
        aligned_rows = []
        delta_by_domain = {r["domain"]: r for r in rr["final_minus_original_by_ewok_domain"]}
        for d, di in did_domains.items():
            ex = delta_by_domain[d]
            aligned_rows.append({
                "domain": d,
                "official_EWoK_DiD": di["DiD_TE43122_minus_TE43022"],
                "TE43022": di["TE43022_reinvest_minus_clean"],
                "TE43122": di["TE43122_reinvest_minus_clean"],
                "concept_coverage_delta": ex["coverage_delta"],
                "source_hit_rows_delta": ex["source_hit_rows_delta"],
                "rewrite_hit_rows_delta": ex["rewrite_hit_rows_delta"],
                "either_hit_rows_delta": ex["either_hit_rows_delta"],
                "source_unique_token_hits_delta": ex["source_unique_token_hits_delta"],
                "rewrite_unique_token_hits_delta": ex["rewrite_unique_token_hits_delta"],
            })
        aligned_rows.sort(key=lambda x: x["official_EWoK_DiD"])
        aligned[rule] = aligned_rows

    result = {
        "status": "REPAIR_EWOK_CONCEPT_EXPOSURE",
        "purpose": "Check whether candidate semantic-force refills preserve lexical exposure to official EWoK concepts in domains with negative treatment interaction.",
        "sources": {
            "selected_reinvest_pairs": str(frontier.SELECTED_REINVEST),
            "medium_compact_rows": str(frontier.MEDIUM_ROWS),
            "ewok_data_dir": str(DATA_DIR),
            "domain_did": str(DID_PATH),
            "refill_plan_dir": str(PLAN_DIR),
        },
        "original_selected_summary_by_ewok_domain": original_summary,
        "rules": rule_results,
        "aligned_with_official_EWoK_DiD": aligned,
        "interpretation": {
            "weakness": "Lexical concept hits do not prove proposition preservation or absence of contamination; they only flag exposure deletion risk.",
            "use": "Prefer refill rules that do not substantially reduce material/physical/spatial EWoK concept exposure if a later semantic-repair training fork is authorized.",
        },
    }
    out_json = OUT_DIR / "repair_ewok_concept_exposure.json"
    out_md = OUT_DIR / "repair_ewok_concept_exposure.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research — EWoK concept exposure under candidate semantic refills\n\n"]
    lines.append("Weak lexical exposure check: counts official EWoK ConceptA/ConceptB token hits in source/rewrite text for the original selected compact block and candidate repaired blocks.\n\n")
    for rule in RULES:
        rr = rule_results[rule]
        lines.append(f"## {rule}\n")
        lines.append(f"Final rows {rr['final_rows']}; final pair words {rr['final_pair_words']}; replacements {rr['replacement_rows']} rows / {rr['replacement_pair_words']} words.\n")
        lines.append("Most negative official EWoK-DiD domains and exposure deltas:\n")
        for rec in aligned[rule][:5]:
            lines.append(
                f"- {rec['domain']}: official DiD {rec['official_EWoK_DiD']:+.2f}; "
                f"coverage_delta {rec['concept_coverage_delta']:+.3f}; "
                f"source_hit_rows_delta {rec['source_hit_rows_delta']:+d}; rewrite_hit_rows_delta {rec['rewrite_hit_rows_delta']:+d}; either_hit_rows_delta {rec['either_hit_rows_delta']:+d}\n"
            )
        lines.append("\n")
    lines.append("## Scientific read\n")
    lines.append("- This protects against a bad repair that exactly refills word count while removing terms from material/physical/spatial EWoK domains.\n")
    lines.append("- The measurement is lexical and weak; it should be combined with exact DiD and semantic-hazard evidence before any new training.\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "summary": {
            rule: {
                "final_pair_words": rule_results[rule]["final_pair_words"],
                "worst5_aligned": aligned[rule][:5],
            } for rule in RULES
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
