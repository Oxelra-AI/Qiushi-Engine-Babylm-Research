#!/usr/bin/env python3
"""research posalign analysis: row-paired sign reversal across bridge conditions."""
import json, sys
from collections import defaultdict
from pathlib import Path

def load_preds(d):
    sr = []; cr = []
    for f in sorted(d.rglob("state_predictions.jsonl")):
        with open(f) as fh:
            for line in fh:
                sr.append(json.loads(line))
    for f in sorted(d.rglob("comparison_predictions.jsonl")):
        with open(f) as fh:
            for line in fh:
                cr.append(json.loads(line))
    return sr, cr

def analyze(bsplus_dir, bsminus_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sp, cp = load_preds(bsplus_dir)
    sm, cm = load_preds(bsminus_dir)
    
    # Index state predictions by (query_key, candidate_index, is_changed)
    def idx_state(rows):
        by = {}
        for r in rows:
            k = (r.get("query_key",""), r.get("candidate_index",0))
            by[k] = r
        return by
    
    bp = idx_state(sp)
    bm = idx_state(sm)
    
    # Row-paired sign analysis for changed events
    families = {"direct_anchor": [], "graph_transfer": [], "unchanged": []}
    for k in bp:
        rp = bp[k]; rm = bm.get(k)
        if rm is None: continue
        dp = rp.get("d_e", 0); dm = rm.get("d_e", 0)
        rf = rp.get("relation_family", "")
        is_ch = rp.get("is_changed", False)
        ci = rp.get("candidate_index", 0)
        if ci != 0: continue  # one row per query
        
        if not is_ch:
            families["unchanged"].append({"dp": dp, "dm": dm, "same_sign": int((dp>0)==(dm>0)) if dp!=0 and dm!=0 else -1})
        elif rf == "direct_anchor":
            families["direct_anchor"].append({"dp": dp, "dm": dm, "opp_sign": int((dp>0)!=(dm>0)) if dp!=0 and dm!=0 else -1})
        elif rf == "graph_transfer":
            families["graph_transfer"].append({"dp": dp, "dm": dm, "opp_sign": int((dp>0)!=(dm>0)) if dp!=0 and dm!=0 else -1})
    
    # Comparison analysis
    def idx_comp(rows):
        by = {}
        for r in rows:
            by[r.get("row_id","")] = r
        return by
    
    cbp = idx_comp(cp); cbm = idx_comp(cm)
    comp_pairs = []
    for k in cbp:
        rp = cbp[k]; rm = cbm.get(k)
        if rm is None: continue
        comp_pairs.append({
            "suite": rp.get("suite",""),
            "logit_p": rp.get("logit_same",0), "logit_m": rm.get("logit_same",0),
            "correct_p": rp.get("correct",0), "correct_m": rm.get("correct",0),
        })
    
    def safe_frac(items, key):
        valid = [x[key] for x in items if x[key] >= 0]
        return sum(valid)/len(valid) if valid else float('nan')
    
    summary = {
        "direct_anchor": {"n": len(families["direct_anchor"]),
                          "opposite_sign_frac": safe_frac(families["direct_anchor"], "opp_sign")},
        "graph_transfer": {"n": len(families["graph_transfer"]),
                           "opposite_sign_frac": safe_frac(families["graph_transfer"], "opp_sign")},
        "unchanged": {"n": len(families["unchanged"]),
                      "same_sign_frac": safe_frac(families["unchanged"], "same_sign")},
    }
    
    # Comparison by suite
    for suite in set(r["suite"] for r in comp_pairs):
        cs = [r for r in comp_pairs if r["suite"] == suite]
        summary[f"comp_{suite}"] = {
            "n": len(cs),
            "correct_plus": sum(r["correct_p"] for r in cs)/len(cs) if cs else 0,
            "correct_minus": sum(r["correct_m"] for r in cs)/len(cs) if cs else 0,
        }
    
    write_json(out_dir / "row_paired_analysis.json", summary)
    
    lines = ["# research position-aligned matcher: row-paired sign analysis", "",
             "| family | n | metric | value |",
             "|---|---:|---|---:|"]
    lines.append(f"| direct_anchor | {summary['direct_anchor']['n']} | opposite_sign | {summary['direct_anchor']['opposite_sign_frac']:.4f} |")
    lines.append(f"| graph_transfer | {summary['graph_transfer']['n']} | opposite_sign | {summary['graph_transfer']['opposite_sign_frac']:.4f} |")
    lines.append(f"| unchanged | {summary['unchanged']['n']} | same_sign | {summary['unchanged']['same_sign_frac']:.4f} |")
    for k, v in summary.items():
        if k.startswith("comp_"):
            lines.append(f"| {k} | {v['n']} | correct_plus | {v['correct_plus']:.4f} |")
            lines.append(f"| {k} | {v['n']} | correct_minus | {v['correct_minus']:.4f} |")
    
    (out_dir / "row_paired_analysis.md").write_text("\n".join(lines) + "\n")
    return summary

def write_json(p, o):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2)+"\n")

if __name__ == "__main__":
    bp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/archive/representation_and_objectives/data/posalign_shared_bsplus")
    bm = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("experiments/archive/representation_and_objectives/data/posalign_shared_bsminus")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("experiments/archive/representation_and_objectives/data/posalign_analysis")
    summary = analyze(bp, bm, out)
    print(json.dumps({"status": "ANALYSIS_COMPLETE", "summary": summary}, indent=2))
