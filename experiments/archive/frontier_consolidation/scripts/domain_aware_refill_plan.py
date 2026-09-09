#!/usr/bin/env python3
"""research: domain-aware refill plans for semantically safer compact blocks.

This CPU-only script prepares replacement plans over existing generated compact
rows. It does not materialize a new training corpus. It answers whether the
low-risk unused accepted pool can refill removed compact-view packets while
keeping domain exposure close enough to the frozen compact_view_reinvest block.
"""
from __future__ import annotations

import collections
import csv
import importlib.util
import json
import pathlib
from typing import Any, Callable

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
FRONTIER_SCRIPT = STUDY / "scripts/semantic_repair_frontier.py"
OUT_DIR = STUDY / "data/domain_aware_refill_plan"

# Load functions/constants from the frontier script to keep hazard definitions identical.
spec = importlib.util.spec_from_file_location("frontier", FRONTIER_SCRIPT)
frontier = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(frontier)  # type: ignore[arg-type]

MEDIUM_ROWS = frontier.MEDIUM_ROWS
SELECTED_REINVEST = frontier.SELECTED_REINVEST
SELECTED_CORE = frontier.SELECTED_CORE
SELECTED_ADDED = frontier.SELECTED_ADDED

MAIN_RULES = [
    ("force_only", None, False, False),
    ("force_plus_pronoun", None, True, False),
    ("force_surface", None, False, True),
    ("force_content_ge_0p45", 0.45, False, False),
]


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return frontier.load_jsonl(path)


def pair_words(r: dict[str, Any]) -> int:
    return frontier.pair_words(r)


def key_of(r: dict[str, Any]) -> str:
    return frontier.key_of(r)


def primary_domain(r: dict[str, Any]) -> str:
    return frontier.primary_domain(r)


def content_recall(r: dict[str, Any]) -> float:
    return frontier.content_recall(r)


def candidate_quality(r: dict[str, Any]) -> float:
    """Rank refill rows: preserve semantic surface while staying compact."""
    entity = float(r.get("entity_recall", 1.0))
    number = float(r.get("number_recall", 1.0))
    lr = float(r.get("length_ratio", r.get("rewrite_words", 0) / max(1, r.get("source_words", 1))))
    # Favor high content/entity/number retention and avoid extremely short rewrites.
    return content_recall(r) + 0.08 * entity + 0.08 * number - 0.03 * abs(lr - 0.62)


def make_rule(content_floor: float | None, include_pronoun: bool, include_surface: bool) -> Callable[[dict[str, Any]], bool]:
    _, fails, _ = frontier.make_rule("tmp", content_floor, include_pronoun, include_surface)
    return fails


def summarize_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pws = [pair_words(r) for r in rows]
    by_domain = collections.Counter()
    by_domain_rows = collections.Counter()
    for r in rows:
        d = primary_domain(r)
        by_domain[d] += pair_words(r)
        by_domain_rows[d] += 1
    def avg(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None
    return {
        "rows": len(rows),
        "pair_words": sum(pws),
        "mean_pair_words": avg([float(x) for x in pws]),
        "mean_content_recall": avg([content_recall(r) for r in rows]),
        "mean_length_ratio": avg([float(r.get("length_ratio", r.get("rewrite_words", 0) / max(1, r.get("source_words", 1)))) for r in rows]),
        "mean_entity_recall": avg([float(r.get("entity_recall", 1.0)) for r in rows]),
        "mean_number_recall": avg([float(r.get("number_recall", 1.0)) for r in rows]),
        "primary_domain_pair_words": dict(by_domain.most_common()),
        "primary_domain_rows": dict(by_domain_rows.most_common()),
    }


def domain_delta(before: dict[str, int], after: dict[str, int]) -> list[dict[str, Any]]:
    domains = sorted(set(before) | set(after))
    total_before = sum(before.values()) or 1
    total_after = sum(after.values()) or 1
    rows = []
    for d in domains:
        b = before.get(d, 0)
        a = after.get(d, 0)
        rows.append({
            "domain": d,
            "before_pair_words": b,
            "after_pair_words": a,
            "delta_pair_words": a - b,
            "before_fraction": b / total_before,
            "after_fraction": a / total_after,
            "delta_fraction": (a / total_after) - (b / total_before),
        })
    rows.sort(key=lambda x: abs(x["delta_pair_words"]), reverse=True)
    return rows


def compact_row_record(r: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "key": key_of(r),
        "prompt_id": r.get("prompt_id"),
        "domain": primary_domain(r),
        "pair_words": pair_words(r),
        "source_words": int(r.get("source_words", 0)),
        "rewrite_words": int(r.get("rewrite_words", 0)),
        "length_ratio": float(r.get("length_ratio", 0.0)),
        "content_recall": content_recall(r),
        "entity_recall": float(r.get("entity_recall", 1.0)),
        "number_recall": float(r.get("number_recall", 1.0)),
        "quality_rank_score": candidate_quality(r),
        "reason": reason,
    }


def build_plan(name: str, fails: Callable[[dict[str, Any]], bool], selected: list[dict[str, Any]], unused_rows: list[dict[str, Any]], core: list[dict[str, Any]], added: list[dict[str, Any]]) -> dict[str, Any]:
    selected_total_words = sum(pair_words(r) for r in selected)
    original_domain_words = collections.Counter()
    for r in selected:
        original_domain_words[primary_domain(r)] += pair_words(r)

    removed = [r for r in selected if fails(r)]
    kept = [r for r in selected if not fails(r)]
    passing_unused = [r for r in unused_rows if not fails(r)]
    removed_need_by_domain = collections.Counter()
    for r in removed:
        removed_need_by_domain[primary_domain(r)] += pair_words(r)
    candidates_by_domain: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in passing_unused:
        candidates_by_domain[primary_domain(r)].append(r)
    for d in candidates_by_domain:
        candidates_by_domain[d].sort(key=lambda r: (candidate_quality(r), -pair_words(r)), reverse=True)

    chosen: list[tuple[dict[str, Any], str]] = []
    chosen_keys: set[str] = set()
    domain_fill = {}
    # Fill each removed domain locally if possible. This keeps the primary domain distribution close.
    for d, need in removed_need_by_domain.most_common():
        acc = 0
        n = 0
        for r in candidates_by_domain.get(d, []):
            if key_of(r) in chosen_keys:
                continue
            if acc >= need:
                break
            chosen.append((r, f"same_domain:{d}"))
            chosen_keys.add(key_of(r))
            acc += pair_words(r)
            n += 1
        have_total = sum(pair_words(r) for r in candidates_by_domain.get(d, []))
        domain_fill[d] = {
            "need_pair_words": need,
            "same_domain_available_pair_words": have_total,
            "same_domain_chosen_pair_words": acc,
            "same_domain_chosen_rows": n,
            "same_domain_shortfall_pair_words": max(0, need - acc),
            "same_domain_overshoot_pair_words": max(0, acc - need),
        }

    replacement_words = sum(pair_words(r) for r, _ in chosen)
    need_total = sum(pair_words(r) for r in removed)
    if replacement_words < need_total:
        rest = [r for r in passing_unused if key_of(r) not in chosen_keys]
        rest.sort(key=lambda r: (candidate_quality(r), -abs(pair_words(r) - (need_total - replacement_words))), reverse=True)
        for r in rest:
            if replacement_words >= need_total:
                break
            chosen.append((r, "global_quality_fill"))
            chosen_keys.add(key_of(r))
            replacement_words += pair_words(r)

    chosen_rows = [r for r, _ in chosen]
    final_rows = kept + chosen_rows
    final_summary = summarize_block(final_rows)
    after_domain_words = final_summary["primary_domain_pair_words"]
    deficits_after_same_domain = [v for v in domain_fill.values() if v["same_domain_shortfall_pair_words"] > 0]

    csv_records = [compact_row_record(r, reason) for r, reason in chosen]

    return {
        "rule": name,
        "selected_original": summarize_block(selected),
        "removed": summarize_block(removed),
        "kept": summarize_block(kept),
        "passing_unused": summarize_block(passing_unused),
        "chosen_replacements": summarize_block(chosen_rows),
        "final_candidate_block": final_summary,
        "word_accounting": {
            "original_selected_pair_words": selected_total_words,
            "removed_pair_words": need_total,
            "kept_pair_words": sum(pair_words(r) for r in kept),
            "replacement_pair_words": replacement_words,
            "final_pair_words": sum(pair_words(r) for r in final_rows),
            "final_minus_original_pair_words": sum(pair_words(r) for r in final_rows) - selected_total_words,
            "can_refill_total_pair_words": replacement_words >= need_total,
            "same_domain_shortfall_total_pair_words_before_global_fill": sum(x["same_domain_shortfall_pair_words"] for x in deficits_after_same_domain),
        },
        "domain_fill": domain_fill,
        "domain_delta_vs_original": domain_delta(dict(original_domain_words), after_domain_words),
        "core_removed_pair_words": sum(pair_words(r) for r in core if fails(r)),
        "added_removed_pair_words": sum(pair_words(r) for r in added if fails(r)),
        "replacement_rows_for_materialization": csv_records,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    medium = load_jsonl(MEDIUM_ROWS)
    selected = load_jsonl(SELECTED_REINVEST)
    core = load_jsonl(SELECTED_CORE)
    added = load_jsonl(SELECTED_ADDED)
    selected_keys = {key_of(r) for r in selected}
    accepted_rows = [r for r in medium if frontier.accepted(r)]
    unused_rows = [r for r in accepted_rows if key_of(r) not in selected_keys]

    plans = []
    for name, content_floor, include_pronoun, include_surface in MAIN_RULES:
        fails = make_rule(content_floor, include_pronoun, include_surface)
        plans.append(build_plan(name, fails, selected, unused_rows, core, added))

    # Save replacement lists for feasible main rules; these are not corpus files.
    csv_paths = {}
    for p in plans:
        csv_path = OUT_DIR / f"{p['rule']}_replacement_rows.csv"
        rows = p["replacement_rows_for_materialization"]
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            csv_path.write_text("", encoding="utf-8")
        csv_paths[p["rule"]] = str(csv_path)

    result = {
        "status": "DOMAIN_AWARE_REFILL_PLAN",
        "sources": {
            "medium_compact_rows": str(MEDIUM_ROWS),
            "selected_reinvest_pairs": str(SELECTED_REINVEST),
            "selected_core_pairs": str(SELECTED_CORE),
            "selected_added_pairs": str(SELECTED_ADDED),
        },
        "plans": [{k: v for k, v in p.items() if k != "replacement_rows_for_materialization"} | {"replacement_rows_csv": csv_paths[p["rule"]]} for p in plans],
        "interpretation": {
            "use": "A future semantic-repair build can start from force_plus_pronoun or force_surface because both are refillable in total words from already accepted unused compact rows.",
            "domain_issue": "The refillable rules still have small primary-domain shortfalls in science_physical, quant_numeric, or media_culture before global fill; replacement should therefore be domain-aware and not merely greedy by quality.",
            "non_training_status": "No corpus was changed and no model work was launched.",
        },
    }
    out_json = OUT_DIR / "domain_aware_refill_plan.json"
    out_md = OUT_DIR / "domain_aware_refill_plan.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research — domain-aware compact-view refill plan\n\n"]
    lines.append("CPU-only replacement planning over existing accepted compact rows. No training corpus was changed.\n\n")
    for p in plans:
        wa = p["word_accounting"]
        lines.append(f"## {p['rule']}\n")
        lines.append(
            f"Removed {wa['removed_pair_words']} pair words; chose {wa['replacement_pair_words']} replacement words; "
            f"final-minus-original {wa['final_minus_original_pair_words']} words; same-domain shortfall before global fill {wa['same_domain_shortfall_total_pair_words_before_global_fill']}.\n"
        )
        final = p["final_candidate_block"]
        chosen = p["chosen_replacements"]
        lines.append(
            f"Final candidate block: {final['rows']} rows / {final['pair_words']} pair words; "
            f"mean content {final['mean_content_recall']:.3f}, mean length ratio {final['mean_length_ratio']:.3f}. "
            f"Replacement rows: {chosen['rows']} / {chosen['pair_words']} words.\n"
        )
        lines.append("Largest primary-domain shifts after refill:\n")
        for d in p["domain_delta_vs_original"][:5]:
            lines.append(
                f"- {d['domain']}: {d['delta_pair_words']} words "
                f"({100*d['delta_fraction']:+.2f} percentage points)\n"
            )
        lines.append(f"Replacement rows: `{csv_paths[p['rule']]}`\n\n")
    lines.append("## Scientific read\n")
    lines.append("- A semantically safer compact-view repair is mechanically possible without new teacher generation for force-only, force-plus-pronoun, and force-plus-entity/number-surface rules.\n")
    lines.append("- Adding even a low 0.45 content floor makes the unused passing pool insufficient, so content-threshold repair would shrink or regenerate the changed block rather than preserve the current rate–distortion idea.\n")
    lines.append("- Because EWoK fragility is concentrated in physical/spatial/material relations, any later corpus repair should protect science_physical and causal_relational exposure while filtering assertion-force errors.\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "summary": {
            p["rule"]: {
                "removed_words": p["word_accounting"]["removed_pair_words"],
                "replacement_words": p["word_accounting"]["replacement_pair_words"],
                "final_minus_original_words": p["word_accounting"]["final_minus_original_pair_words"],
                "same_domain_shortfall_before_global_fill": p["word_accounting"]["same_domain_shortfall_total_pair_words_before_global_fill"],
                "replacement_rows_csv": csv_paths[p["rule"]],
            }
            for p in plans
        },
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
