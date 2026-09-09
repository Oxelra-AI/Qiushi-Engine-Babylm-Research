#!/usr/bin/env python3
"""research: Analyze oracle and learned SpanMatcher gauge results.

Reads saved prediction files from oracle_gauge/ and learned_gauge/,
computes row-paired sign reversal, matching diagnostics, and comparison with research.
"""
import json, math, os, sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path("experiments/archive/representation_and_objectives")


def load_jsonl(path):
    rows = []
    if not path.exists(): return rows
    for line in path.open():
        s = line.strip()
        if s: rows.append(json.loads(s))
    return rows


def load_summary(d):
    p = Path(d) / "span_matcher_gauge_summary.json"
    if p.exists(): return json.loads(p.read_text())
    return None


def row_paired_analysis(dir1, dir2):
    """Compare bs+1 and bs-1 prediction files for row-paired sign reversal."""
    results = []
    for d in [dir1, dir2]:
        if d is None: continue
        summary = load_summary(d)
        if not summary: continue
        
        # Find prediction directories
        pred_dirs = {}
        for item in sorted(Path(d).iterdir()):
            if item.is_dir() and "bs" in item.name:
                key = item.name  # e.g., "learned_shared_trunk_bs+1_seed29400"
                pred_dirs[key] = item
        
        # Group by condition
        by_condition = defaultdict(dict)
        for key, pdir in pred_dirs.items():
            parts = key.rsplit("_seed", 1)
            base = parts[0] if parts else key
            # Extract bridge sign
            if "bs+1" in base: bs = 1
            elif "bs-1" in base: bs = -1
            else: continue
            cond = base.replace("_bs+1", "").replace("_bs-1", "")
            by_condition[cond][bs] = pdir
        
        # For each condition with both signs
        for cond, signs in by_condition.items():
            if 1 not in signs or -1 not in signs: continue
            
            # Load state predictions
            s_plus = load_jsonl(signs[1] / "state_predictions.jsonl")
            s_minus = load_jsonl(signs[-1] / "state_predictions.jsonl")
            
            # Index by query_key + candidate
            def index_states(rows):
                by = {}
                for r in rows:
                    k = (r["suite"], r["query_key"], r["candidate_index"])
                    by[k] = r
                return by
            
            idx_plus = index_states(s_plus)
            idx_minus = index_states(s_minus)
            
            # Row-paired sign analysis for graph-transfer changed rows
            paired_count = 0
            opposite_sign = 0
            for k in idx_plus:
                if k not in idx_minus: continue
                rp = idx_plus[k]
                rm = idx_minus[k]
                if rp.get("relation_family") != "graph_transfer": continue
                if rp.get("query_kind") != "changed": continue
                if rp["candidate_index"] != 0: continue  # use d_e from index 0
                
                de_plus = rp["d_e"]
                de_minus = rm["d_e"]
                paired_count += 1
                if de_plus * de_minus < 0:
                    opposite_sign += 1
            
            # Comparison predictions
            c_plus = load_jsonl(signs[1] / "comparison_predictions.jsonl")
            c_minus = load_jsonl(signs[-1] / "comparison_predictions.jsonl")
            
            results.append({
                "source": str(d),
                "condition": cond,
                "n_paired_graph_changed": paired_count,
                "n_opposite_sign": opposite_sign,
                "opposite_sign_fraction": opposite_sign / paired_count if paired_count else None,
                "n_state_plus": len(s_plus),
                "n_state_minus": len(s_minus),
                "n_comp_plus": len(c_plus),
                "n_comp_minus": len(c_minus),
            })
    
    return results


def main():
    oracle_dir = PROJECT / "data/oracle_gauge"
    learned_dir = PROJECT / "data/learned_gauge"
    out_dir = PROJECT / "data/span_matcher_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load summaries
    oracle_summary = load_summary(oracle_dir)
    learned_summary = load_summary(learned_dir)
    
    # Compact comparison table
    md_lines = ["# research SpanMatcher gauge analysis\n\n"]
    md_lines.append("## Results comparison\n\n")
    md_lines.append("| matcher | condition | bs | train_st | train_cmp | graph_same | unchanged | hh_closure | mixed_acc | match_st | match_cmp |\n")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    
    all_results = []
    for label, summary in [("oracle", oracle_summary), ("learned", learned_summary)]:
        if not summary: continue
        for r in summary.get("all_results", []):
            ce = r.get("central_eval", {})
            ma = r.get("matching_accuracy", {})
            ts = r.get("train_state_acc", "")
            tc = r.get("train_cmp_acc", "")
            gs = ce.get("graph_same", "")
            un = ce.get("unchanged", "")
            hh = ce.get("hh_closure", "")
            mx = ce.get("mixed_acc", "")
            ms = ma.get("state_match_acc", "")
            mc = ma.get("cmp_match_acc", "")
            
            def fmt(v):
                if isinstance(v, float): return f"{v:.3f}" if not math.isnan(v) else "NaN"
                return str(v)
            
            md_lines.append(f"| {label} | {r['condition']} | {r['bridge_sign']:+d} | "
                           f"{fmt(ts)} | {fmt(tc)} | {fmt(gs)} | {fmt(un)} | "
                           f"{fmt(hh)} | {fmt(mx)} | {fmt(ms)} | {fmt(mc)} |\n")
            all_results.append(r)
    
    # Row-paired analysis
    md_lines.append("\n## Row-paired bridge-sign analysis\n\n")
    paired = row_paired_analysis(oracle_dir, learned_dir)
    for p in paired:
        md_lines.append(f"- **{p['condition']}**: {p['n_opposite_sign']}/{p['n_paired_graph_changed']} "
                       f"graph-transfer rows have opposite d_e signs "
                       f"(fraction = {p['opposite_sign_fraction']:.3f})\n" if p['opposite_sign_fraction'] is not None
                       else f"- **{p['condition']}**: no paired rows\n")
    
    # Key comparison with research/293
    md_lines.append("\n## Comparison with supplied harness (research/293)\n\n")
    md_lines.append("research/293 clean supplied harness:\n")
    md_lines.append("- shared_trunk: graph_same +1=1.0, -1=0.0, hh=1.0, mixed=1.0, unchanged=1.0\n")
    md_lines.append("- untied: graph_same +1=0.75, -1=0.5, hh=1.0, mixed=0.5\n\n")
    
    md_lines.append("research raw-name attention (best):\n")
    md_lines.append("- tied +: graph_same=0.844, hh=0.750, mixed=0.777, unchanged=0.812\n")
    md_lines.append("- shared_trunk +: graph_same=0.688, hh=0.547, mixed=0.539, unchanged=0.812\n\n")
    
    # Matching accuracy diagnostic
    md_lines.append("## Matching accuracy diagnostic\n\n")
    for r in all_results:
        ma = r.get("matching_accuracy", {})
        if ma.get("mode") == "learned":
            md_lines.append(f"- {r['condition']} bs{r['bridge_sign']:+d}: "
                           f"state match={ma.get('state_match_acc','?'):.3f}, "
                           f"cmp match={ma.get('cmp_match_acc','?'):.3f}\n")
    
    md_path = out_dir / "span_matcher_analysis.md"
    md_path.write_text("".join(md_lines))
    
    json_path = out_dir / "span_matcher_analysis.json"
    json_path.write_text(json.dumps({
        "all_results": all_results,
        "paired_analysis": paired,
    }, indent=2, sort_keys=True, default=str) + "\n")
    
    print(json.dumps({
        "status": "ANALYSIS_COMPLETE",
        "summary": str(md_path),
        "json": str(json_path),
        "n_results": len(all_results),
        "n_paired": len(paired),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
