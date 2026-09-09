#!/usr/bin/env python3
"""research: Audit generated concise views against fidelity requirements.

Loads the test prompts (with extracted entities and targets) and the generated
outputs, runs the verifier, and reports acceptance rate and quality metrics.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import sys
from pathlib import Path
from collections import Counter

# Import the protocol's verify function
sys.path.insert(0, str(_public_path('experiments/archive/frontier_consolidation/scripts')))
from concise_view_protocol import verify_output, word_count, extract_entities


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str,
                        default="experiments/archive/frontier_consolidation/data/concise_view_test/test_prompts.jsonl")
    parser.add_argument("--outputs", type=str,
                        default="experiments/archive/frontier_consolidation/data/concise_view_test/generated_outputs.jsonl")
    parser.add_argument("--report", type=str,
                        default="experiments/archive/frontier_consolidation/data/concise_view_test/audit_report.json")
    args = parser.parse_args()

    # Load prompts (have metadata)
    prompts = {}
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                prompts[rec["id"]] = rec

    # Load outputs
    outputs = {}
    with open(args.outputs, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                outputs[rec["id"]] = rec

    # Run verification
    results = []
    for pid, prompt in prompts.items():
        if pid not in outputs:
            results.append({"id": pid, "accepted": False, "reasons": ["missing_output"], "metrics": {}})
            continue
        
        output_text = outputs[pid]["output"]
        meta = {
            "source_words": prompt["source_words"],
            "target_min_words": prompt["target_min_words"],
            "target_max_words": prompt["target_max_words"],
            "extracted_entities": prompt["extracted_entities"],
        }
        
        verification = verify_output(prompt["source_sentence"], output_text, meta)
        verification["id"] = pid
        verification["source"] = prompt["source_name"]
        verification["source_words"] = prompt["source_words"]
        verification["output_text"] = output_text
        verification["source_sentence"] = prompt["source_sentence"]
        results.append(verification)

    # Compute summary statistics
    n_total = len(results)
    n_accepted = sum(1 for r in results if r["accepted"])
    accept_rate = n_accepted / n_total if n_total > 0 else 0

    # Rejection reason counts
    reason_counts = Counter()
    for r in results:
        if not r["accepted"]:
            for reason in r["reasons"]:
                reason_counts[reason] += 1

    # Length ratio stats for accepted
    accepted_ratios = [r["metrics"]["len_ratio"] for r in results 
                       if r["accepted"] and "len_ratio" in r.get("metrics", {})]
    
    # Content overlap stats for accepted
    accepted_overlaps = [r["metrics"]["content_overlap"] for r in results
                         if r["accepted"] and "content_overlap" in r.get("metrics", {})]
    
    # Word count savings for accepted
    accepted_savings = []
    for r in results:
        if r["accepted"] and "output_words" in r.get("metrics", {}):
            saved = r["source_words"] - r["metrics"]["output_words"]
            accepted_savings.append(saved)

    # By-source acceptance
    source_accept = {}
    source_counts_all = Counter()
    for r in results:
        src = r.get("source", "unknown")
        source_counts_all[src] += 1
        if src not in source_accept:
            source_accept[src] = 0
        if r["accepted"]:
            source_accept[src] += 1

    report = {
        "status": "CONCISE_VIEW_AUDIT",
        "n_total": n_total,
        "n_accepted": n_accepted,
        "accept_rate": round(accept_rate, 4),
        "rejection_reasons": dict(reason_counts.most_common()),
        "accepted_stats": {
            "len_ratio": {
                "min": round(min(accepted_ratios), 4) if accepted_ratios else None,
                "mean": round(sum(accepted_ratios)/len(accepted_ratios), 4) if accepted_ratios else None,
                "max": round(max(accepted_ratios), 4) if accepted_ratios else None,
            },
            "content_overlap": {
                "min": round(min(accepted_overlaps), 4) if accepted_overlaps else None,
                "mean": round(sum(accepted_overlaps)/len(accepted_overlaps), 4) if accepted_overlaps else None,
                "max": round(max(accepted_overlaps), 4) if accepted_overlaps else None,
            },
            "words_saved_per_pair": {
                "min": min(accepted_savings) if accepted_savings else None,
                "mean": round(sum(accepted_savings)/len(accepted_savings), 1) if accepted_savings else None,
                "max": max(accepted_savings) if accepted_savings else None,
                "total": sum(accepted_savings) if accepted_savings else None,
            },
        },
        "source_acceptance": {
            src: {
                "n": source_counts_all[src],
                "accepted": source_accept.get(src, 0),
                "rate": round(source_accept.get(src, 0) / source_counts_all[src], 4) if source_counts_all[src] > 0 else 0
            }
            for src in sorted(source_counts_all)
        },
        "examples": {
            "accepted_first5": [],
            "rejected_first5": [],
        }
    }

    # Add examples
    for r in results:
        if r["accepted"] and len(report["examples"]["accepted_first5"]) < 5:
            report["examples"]["accepted_first5"].append({
                "id": r["id"],
                "source": r.get("source"),
                "source_sentence": r.get("source_sentence", ""),
                "output": r.get("output_text", ""),
                "source_words": r.get("source_words"),
                "output_words": r["metrics"].get("output_words"),
                "len_ratio": r["metrics"].get("len_ratio"),
            })
        elif not r["accepted"] and len(report["examples"]["rejected_first5"]) < 5:
            report["examples"]["rejected_first5"].append({
                "id": r["id"],
                "source": r.get("source"),
                "source_sentence": r.get("source_sentence", ""),
                "output": r.get("output_text", ""),
                "source_words": r.get("source_words"),
                "output_words": r["metrics"].get("output_words"),
                "len_ratio": r["metrics"].get("len_ratio"),
                "reasons": r["reasons"],
            })

    # Save report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Print summary
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
