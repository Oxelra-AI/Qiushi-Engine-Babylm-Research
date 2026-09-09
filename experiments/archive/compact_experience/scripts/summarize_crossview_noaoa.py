#!/usr/bin/env python3
"""Summarize research cross-view no-AoA evaluation results."""
import json
import sys
from pathlib import Path

def main():
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/crossview_noaoa_eval")
    per_target = out_root / "per_target"
    
    arms = ["baseline", "anchor_copy", "semantic_xview", "partner_shuffle"]
    results = {}
    
    for arm in arms:
        matches = list(per_target.glob(f"step057_{arm}_*.json"))
        if not matches:
            continue
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        scores = payload.get("zero_shot_scores", {})
        # Compute equal7 (no-AoA proxy)
        cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
        col_vals = {}
        for c in cols:
            if c == "GlobalPIQA":
                p = scores.get("GlobalPIQA_parallel", {}).get("score")
                np_score = scores.get("GlobalPIQA_nonparallel", {}).get("score")
                if p is not None and np_score is not None:
                    col_vals[c] = (p + np_score) / 2
            else:
                v = scores.get(c, {}).get("score")
                if v is not None:
                    col_vals[c] = v
        
        equal7 = sum(col_vals.values()) / max(1, len(col_vals)) if col_vals else None
        results[arm] = {
            "columns": col_vals,
            "equal7_noaoa": equal7,
            "payload_path": str(matches[0]),
        }
    
    if not results:
        print("NO RESULTS FOUND")
        return
    
    # Compute deltas relative to baseline
    baseline = results.get("baseline", {}).get("equal7_noaoa")
    for arm, r in results.items():
        if r.get("equal7_noaoa") is not None and baseline is not None:
            r["delta_vs_baseline"] = r["equal7_noaoa"] - baseline
    
    # Column-level deltas
    baseline_cols = results.get("baseline", {}).get("columns", {})
    for arm, r in results.items():
        if arm == "baseline":
            continue
        col_deltas = {}
        for c, v in r.get("columns", {}).items():
            if c in baseline_cols:
                col_deltas[c] = round(v - baseline_cols[c], 4)
        r["column_deltas_vs_baseline"] = col_deltas
    
    report = {
        "status": "CROSSVIEW_NOAOA_SUMMARY",
        "results": results,
        "interpretation": {
            "question": "Does semantic cross-view reconstruction (non-anchor masking) outperform "
                       "lexical copying (anchor masking) and random baseline?",
            "acceptance": "semantic_xview > baseline on Entity/EWoK without Supplement/Reading collapse; "
                         "anchor_copy tests whether gains come from copying visible identical tokens.",
            "limits": "This no-AoA screen cannot establish final Overall or AoA safety."
        }
    }
    
    summary_path = out_root / "crossview_noaoa_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
