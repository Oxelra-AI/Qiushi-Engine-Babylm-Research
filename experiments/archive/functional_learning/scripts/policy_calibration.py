#!/usr/bin/env python3
"""research: calibrate compact-admission policy against independent review semantic labels.

The goal is high-precision admission for a learner comparison, not estimating true
corpus prevalence from a risk-enriched sample.  The calibration asks which automatic
rules would have admitted altered or information-losing pilot examples.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

STUDY = Path("experiments/archive/functional_learning")
DATA = STUDY / "data/compact_semantic_review"
SCREEN = DATA / "semantic_screen_pilot.jsonl"
LABELS = DATA / "independent_review_semantic_labels_pilot_sample.jsonl"
OUT = DATA / "policy_calibration_summary.json"


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def decision_v0(r):
    return bool(r.get("auto_admit_low_risk"))


def decision_v1(r):
    """Stricter automatic admit: only very simple, low-risk, non-dialogue/source-overlap-safe rows."""
    tags = set(r.get("risk_tags", []))
    forbidden = {
        "dialogue_or_question",
        "compressed_dialogue",
        "role_reference_weakened",
        "ordered_navigation",
        "navigation_sequence_changed",
        "conditional_force_weakened",
        "conditional_force_partly_changed",
        "modality_force_weakened",
        "modality_force_partly_changed",
        "negation_force_weakened",
        "negation_force_partly_changed",
        "causal_or_contrast_force_weakened",
        "temporal_order_weakened",
        "comparison_relation_weakened",
        "number_changed_or_removed",
        "severe_summary",
        "very_low_source_overlap",
        "very_low_current_overlap",
    }
    return (
        r.get("risk_score", 999) < 12
        and r.get("compact_words", 0) >= 8
        and r.get("compact_words", 0) < r.get("current_words", 0)
        and not (tags & forbidden)
        and not r.get("known_altered")
        and r.get("compact_original_jaccard", 0.0) >= 0.30
        and r.get("compact_current_jaccard", 0.0) >= 0.18
    )


def decision_v2(r):
    """Almost no automatic semantic admission; admit only atomic factual rows without markers."""
    tags = set(r.get("risk_tags", []))
    # Allow only source rows without any risk tag except possibly none.  This is a
    # production safety fallback; manual/LLM review can admit high-risk but faithful rows.
    return (
        r.get("risk_score", 999) == 0
        and not tags
        and r.get("compact_words", 0) >= 8
        and r.get("compact_words", 0) < r.get("current_words", 0)
        and r.get("compact_original_jaccard", 0.0) >= 0.35
        and r.get("compact_current_jaccard", 0.0) >= 0.20
    )


def is_admissible_label(label: str) -> bool:
    return label == "faithful_shortening"


def main():
    screen = {r["pair_id"]: r for r in load_jsonl(SCREEN)}
    labels = load_jsonl(LABELS)
    joined = []
    for lab in labels:
        r = screen.get(lab["pair_id"])
        if not r:
            continue
        jr = {**r, "semantic_label": lab["label"], "semantic_note": lab.get("note", "")}
        jr["decisions"] = {"v0": decision_v0(r), "v1": decision_v1(r), "v2": decision_v2(r)}
        joined.append(jr)

    label_counts = Counter(j["semantic_label"] for j in joined)
    by_policy = {}
    for pol in ["v0", "v1", "v2"]:
        admitted = [j for j in joined if j["decisions"][pol]]
        bad = [j for j in admitted if not is_admissible_label(j["semantic_label"])]
        good = [j for j in admitted if is_admissible_label(j["semantic_label"])]
        by_policy[pol] = {
            "admitted_labeled_sample": len(admitted),
            "faithful_admitted": len(good),
            "nonfaithful_admitted": len(bad),
            "precision_on_labeled_sample": (len(good) / len(admitted)) if admitted else None,
            "nonfaithful_ids": [{"pair_id": j["pair_id"], "label": j["semantic_label"], "risk_score": j["risk_score"], "tags": j["risk_tags"], "note": j["semantic_note"]} for j in bad],
            "faithful_ids": [j["pair_id"] for j in good],
        }

    # Apply policies to all 512 for budget implications.
    all_rows = list(screen.values())
    policy_all = {}
    for pol, fn in [("v0", decision_v0), ("v1", decision_v1), ("v2", decision_v2)]:
        adm = [r for r in all_rows if fn(r)]
        policy_all[pol] = {
            "n_admitted_all512": len(adm),
            "saved_words_all512": sum(r["saved_words_if_used"] for r in adm),
            "by_source": dict(Counter(r["source"] for r in adm)),
        }

    # Feature/tag failure table: which tags occur in nonfaithful vs faithful labels.
    tag_label = defaultdict(Counter)
    for j in joined:
        for t in j.get("risk_tags", []):
            tag_label[t][j["semantic_label"]] += 1

    summary = {
        "status": "POLICY_CALIBRATED",
        "n_labeled_examples": len(joined),
        "label_counts": dict(label_counts),
        "policy_definitions": {
            "v0": "original research auto_low_risk: risk<25 plus length and number checks",
            "v1": "stricter low-risk auto-admit excluding dialogue/navigation/operator/order/severe-summary/low-overlap tags",
            "v2": "atomic fallback: only risk_score==0 and no tags with adequate overlap/length; intended to be supplemented by semantic review labels",
        },
        "policy_on_labeled_sample": by_policy,
        "policy_on_all512": policy_all,
        "tag_label_counts_top": {t: dict(c) for t, c in sorted(tag_label.items(), key=lambda kv: -sum(kv[1].values()))[:40]},
        "interpretation": "Because the sample is risk-enriched, these figures do not estimate full-corpus error prevalence. They show that automatic lexical/risk heuristics are not safe enough for admitting production compact rewrites without semantic review; v2 is high-safety but low-yield and still not a proof of correctness.",
    }

    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (DATA / "policy_calibration_joined_labels.jsonl").open("w", encoding="utf-8") as f:
        for j in joined:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
