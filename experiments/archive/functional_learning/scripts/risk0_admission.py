#!/usr/bin/env python3
"""research: Build risk=0 no-tag admission decisions from full selective generation.

Creates admission decisions for the legal-stream v5 candidate using the structural
risk=0 no-tag criterion from the semantic screen. These rows passed ALL structural
checks: no number changes, no entity losses, no negation/modality/temporal/causal force
changes, no severe summaries, no known altered claims.

The script also validates a random 30-row sample by printing original/current/compact
for human inspection before committing the full batch.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, random, sys
from collections import Counter

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')

JOINED = _public_path('experiments/archive/functional_learning/data/selective_full_generation/selective_full_joined.jsonl')
SCREEN = _public_path('experiments/archive/functional_learning/data/selective_full_generation/semantic_screen/semantic_screen_pilot.jsonl')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/risk0_admission')


def wc(t): return len((t or "").strip().split())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load screen results
    screen_by_pid = {}
    with SCREEN.open() as f:
        for line in f:
            r = json.loads(line.strip())
            screen_by_pid[r["pair_id"]] = r

    # Load joined generation
    joined = []
    with JOINED.open() as f:
        for line in f:
            joined.append(json.loads(line.strip()))

    # Identify risk=0 no-tag candidates
    decisions = []
    admitted = []
    for r in joined:
        pid = r["pair_id"]
        sc = screen_by_pid.get(pid, {})
        risk = sc.get("risk_score", 999)
        tags = sc.get("risk_tags", ["unknown"])
        keep_current = r.get("selective_keep_current", False)
        comp = str(r.get("compact_rewrite", "")).strip()
        curr = str(r.get("current_rewrite", "")).strip()
        orig = str(r.get("original", "")).strip()

        is_risk0_notag = (risk == 0 and len(tags) == 0 and not keep_current)
        compact_words = wc(comp)
        current_words = wc(curr)
        has_saving = compact_words < current_words and compact_words >= 4

        admit = is_risk0_notag and has_saving and bool(comp)

        d = {
            "pair_id": pid,
            "source": r.get("source", ""),
            "example_id": r.get("example_id"),
            "admit_compact": admit,
            "semantic_label": "faithful_shortening" if admit else ("keep_current" if keep_current else "not_admitted"),
            "semantic_confidence": None,
            "semantic_reason": f"risk=0 no-tag structural screen" if admit else f"risk={risk} tags={len(tags)} keep={keep_current}",
            "current_rewrite_words": current_words,
            "compact_rewrite_words": compact_words,
            "saved_words_unique_pair": max(0, current_words - compact_words) if admit else 0,
            "reject_reasons": [] if admit else [f"risk={risk}", f"tags={len(tags)}"] if not keep_current else ["keep_current"],
            "compact_rewrite": comp,
            "current_rewrite": curr,
            "original": orig,
            "risk_score": risk,
            "risk_tags": tags,
        }
        decisions.append(d)
        if admit:
            admitted.append(d)

    # Write decisions
    out_path = _public_path('experiments/archive/functional_learning/data/risk0_admission/risk0_admission_decisions.jsonl')
    with out_path.open("w") as f:
        for d in decisions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Validate: print 30 random admitted examples
    sample = random.Random(42).sample(admitted, min(30, len(admitted)))
    validation = []
    for s in sample:
        validation.append({
            "pair_id": s["pair_id"],
            "source": s["source"],
            "original": s["original"][:200],
            "current": s["current_rewrite"][:200],
            "compact": s["compact_rewrite"][:200],
            "saved_words": s["saved_words_unique_pair"],
        })

    val_path = _public_path('experiments/archive/functional_learning/data/risk0_admission/risk0_validation_sample.jsonl')
    with val_path.open("w") as f:
        for v in validation:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    by_source = Counter(d["source"] for d in admitted)
    total_saved = sum(d["saved_words_unique_pair"] for d in admitted)
    saved_dist = Counter()
    for d in admitted:
        s = d["saved_words_unique_pair"]
        if s <= 3: saved_dist["1-3"] += 1
        elif s <= 5: saved_dist["4-5"] += 1
        elif s <= 10: saved_dist["6-10"] += 1
        elif s <= 15: saved_dist["11-15"] += 1
        else: saved_dist["16+"] += 1

    summary = {
        "status": "RISK0_ADMISSION_DONE",
        "total_pairs_screened": len(decisions),
        "total_admitted": len(admitted),
        "total_saved_words_unique": total_saved,
        "mean_saved_words": round(total_saved / len(admitted), 2) if admitted else 0,
        "by_source": dict(by_source.most_common()),
        "savings_distribution": dict(sorted(saved_dist.items())),
        "admission_criterion": "risk=0, no risk_tags, not KEEP_CURRENT, compact has fewer words, compact >= 4 words",
        "output": str(out_path.relative_to(ROOT)),
        "validation_sample": str(val_path.relative_to(ROOT)),
        "interpretation": "Risk=0 no-tag means the semantic screen found NO structural red flags. These are the most conservative candidates. Validation sample should be inspected before committing to legal-stream training.",
    }
    (_public_path('experiments/archive/functional_learning/data/risk0_admission/risk0_admission_summary.json')).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
