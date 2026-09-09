#!/usr/bin/env python3
"""research: propose a high-fidelity source-attested bridge subset from automatic structure metrics.

This is a conservative file-level selection for scientific inspection, not a final
annotation. It combines the blinded-reader conclusion with structural metrics to ask:
how many of the 103 retained candidates are plausible enough to seed a stricter
bridge generator/selector?
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
from pathlib import Path

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
ACCEPTED = _public_path('experiments/archive/frontier_consolidation/data/improved_fluent_bridge/final_accepted.jsonl')
STRUCT_CSV = _public_path('experiments/archive/frontier_consolidation/data/bridge_structural_transformation/per_pair_structural.csv')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/bridge_high_fidelity_candidates')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/bridge_high_fidelity_candidates/bridge_high_fidelity_candidates.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/bridge_high_fidelity_candidates/bridge_high_fidelity_candidates.md')

# Cases explicitly named by independent review as meaningful structural transformations.
independent_review_POSITIVE_CASES = {"B006", "B019", "B024", "B039", "B056", "B061", "B067", "B082", "B095", "B096"}
independent_review_BAD_EXAMPLES = {"B009", "B049", "B027"}


def load() -> tuple[list[dict], dict[str, dict]]:
    rows = []
    with ACCEPTED.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                r["case_id"] = f"B{i:03d}"
                rows.append(r)
    sm = {}
    with STRUCT_CSV.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sm[r["pair_id"]] = r
    return rows, sm


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, sm = load()
    records = []
    for r in rows:
        s = sm.get(r["pair_id"], {})
        cid = r["case_id"]
        edit = s.get("edit_class")
        order = float(s.get("order_agreement", 1.0))
        run = float(s.get("longest_run_frac", 1.0))
        novel = float(s.get("novel_word_fraction", 0.0))
        func = float(s.get("function_word_edit_rate", 0.0))
        # Candidate tiers. Tier A = substantive or independent review positive; Tier B = structural edits that
        # should be human/LLM adjudicated; exclude cases independent review flagged as bad examples.
        tier = "reject_or_extract"
        reasons = []
        if cid in independent_review_BAD_EXAMPLES:
            reasons.append("independent_review_named_bad_or_semantically_questionable")
        elif cid in independent_review_POSITIVE_CASES:
            tier = "A_seed_transform"
            reasons.append("independent_review_named_positive_transform")
        elif edit == "substantive_restructure" and run <= 0.75:
            tier = "A_seed_transform"
            reasons.append("automatic_substantive_low_copied_run")
        elif edit == "light_restructure" and (order < 0.92 or func >= 0.45 or run < 0.55) and run <= 0.8:
            tier = "B_adjudicate_transform"
            reasons.append("automatic_light_with_order_or_function_change")
        elif edit in {"verbatim_substring", "prefix_trim_only", "deletion_reorder"}:
            reasons.append("extractive_structural_class")
        else:
            reasons.append("weak_or_ambiguous_light_edit")
        records.append({
            "case_id": cid,
            "pair_id": r["pair_id"],
            "bucket": r.get("prototype_bucket"),
            "regime": r.get("regime"),
            "tier": tier,
            "reasons": reasons,
            "edit_class": edit,
            "order_agreement": order,
            "longest_run_frac": run,
            "novel_word_fraction": novel,
            "function_word_edit_rate": func,
            "source_text": r.get("source_text"),
            "candidate_text": r.get("generated_text") or r.get("raw_output"),
            "natural_compact_text": r.get("natural_compact_text"),
        })
    from collections import Counter
    counts = Counter(x["tier"] for x in records)
    payload = {
        "status": "BRIDGE_HIGH_FIDELITY_CANDIDATE_FILTER",
        "n": len(records),
        "tier_counts": dict(counts),
        "interpretation": "Tier A/B are not training-ready labels; they estimate how much usable structural-transformation seed material exists after rejecting exact/deletion-like extraction and independent_review-named bad cases.",
        "records": records,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research high-fidelity bridge candidate subset",
        "",
        f"Total retained research candidates: {len(records)}",
        "",
        "| tier | n | meaning |",
        "|---|---:|---|",
    ]
    for tier in ["A_seed_transform", "B_adjudicate_transform", "reject_or_extract"]:
        md.append(f"| {tier} | {counts.get(tier,0)} | {'strong seed / independent_review-positive' if tier.startswith('A') else ('needs adjudication' if tier.startswith('B') else 'extractive, damaged, or weak light edit')} |")
    md += ["", "## Tier A examples", ""]
    for rec in [x for x in records if x["tier"] == "A_seed_transform"][:20]:
        md += [
            f"### {rec['case_id']} — {rec['bucket']} / {rec['regime']} / {rec['edit_class']}",
            f"Metrics: run={rec['longest_run_frac']}, order={rec['order_agreement']}, func_edit={rec['function_word_edit_rate']}, novel={rec['novel_word_fraction']}",
            f"Source: {rec['source_text']}",
            f"Candidate: {rec['candidate_text']}",
            f"Natural compact ref: {rec['natural_compact_text']}",
            "",
        ]
    md += ["", "## Tier B examples", ""]
    for rec in [x for x in records if x["tier"] == "B_adjudicate_transform"][:20]:
        md += [
            f"### {rec['case_id']} — {rec['bucket']} / {rec['regime']} / {rec['edit_class']}",
            f"Metrics: run={rec['longest_run_frac']}, order={rec['order_agreement']}, func_edit={rec['function_word_edit_rate']}, novel={rec['novel_word_fraction']}",
            f"Source: {rec['source_text']}",
            f"Candidate: {rec['candidate_text']}",
            "",
        ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "tier_counts": dict(counts), "out_json": str(OUT_JSON.relative_to(ROOT)), "out_md": str(OUT_MD.relative_to(ROOT))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
