#!/usr/bin/env python3
"""research analysis: row-paired sign reversal, matching diagnostics,
and comparison with research/294 baselines."""
import json, sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path("experiments/archive/representation_and_objectives")

def load_jsonl(path):
    rows = []
    if not path.exists(): return rows
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s: rows.append(json.loads(s))
    return rows

def analyze_pair(dir_plus, dir_minus, label):
    """Analyze one (condition, matcher) pair across bridge signs."""
    sp = load_jsonl(dir_plus / "state_predictions.jsonl")
    sm = load_jsonl(dir_minus / "state_predictions.jsonl")
    cp = load_jsonl(dir_plus / "comparison_predictions.jsonl")
    cm = load_jsonl(dir_minus / "comparison_predictions.jsonl")
    
    if not sp or not sm:
        return {"label": label, "error": "missing predictions"}
    
    # Index by query_key + candidate_index for paired analysis
    def index_state(rows):
        by = {}
        for r in rows:
            k = (r["query_key"], r["candidate_index"])
            by[k] = r
        return by
    
    sp_idx = index_state(sp)
    sm_idx = index_state(sm)
    
    # Row-paired sign reversal for changed events
    categories = {"graph_transfer": {"n": 0, "opposite": 0, "same": 0},
                   "direct_anchor": {"n": 0, "opposite": 0, "same": 0},
                   "unchanged": {"n": 0, "same_sign": 0}}
    
    for k in sp_idx:
        if k not in sm_idx: continue
        rp, rm = sp_idx[k], sm_idx[k]
        if rp["candidate_index"] != 0: continue  # avoid double counting
        
        dep = rp.get("d_e", 0)
        dem = sm_idx.get(k, {}).get("d_e", 0)
        if k not in sm_idx: continue
        dem = sm_idx[k]["d_e"]
        
        if rp.get("is_changed"):
            rf = rp.get("relation_family", "")
            if rf in categories:
                categories[rf]["n"] += 1
                if dep * dem < 0: categories[rf]["opposite"] += 1
                else: categories[rf]["same"] += 1
        else:
            categories["unchanged"]["n"] += 1
            if dep * dem > 0: categories["unchanged"]["same_sign"] += 1
    
    # Matching diagnostics from state rows (cand_index=0 only)
    for split_label, rows in [("eval_plus", sp), ("eval_minus", sm)]:
        n_match = correct_cand = correct_other = 0
        for r in rows:
            if r.get("candidate_index") != 0: continue
            m = r.get("matching")
            if not m: continue
            n_match += 1
            if m.get("cand_p_cand", 0) > 0.5: correct_cand += 1
            if m.get("other_p_other", 0) > 0.5: correct_other += 1
        categories[f"matching_{split_label}"] = {
            "n": n_match, "cand_acc": correct_cand/n_match if n_match else 0,
            "other_acc": correct_other/n_match if n_match else 0
        }
    
    # Comparison sign invariance
    cp_idx = {r["row_id"]: r for r in cp}
    cm_idx = {r["row_id"]: r for r in cm}
    comp_same = comp_diff = 0
    for rid in cp_idx:
        if rid not in cm_idx: continue
        de1p = cp_idx[rid].get("d_e1", 0)
        de1m = cm_idx[rid].get("d_e1", 0)
        if de1p * de1m > 0: comp_same += 1
        else: comp_diff += 1
    categories["comparison_sign"] = {"same": comp_same, "diff": comp_diff,
                                      "n": comp_same+comp_diff}
    
    return {"label": label, "categories": categories}

def main():
    out = PROJECT / "data/span_discovery_analysis"
    out.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Check available result directories
    for base_dir, cond in [
        (PROJECT / "data/learned_shared", "shared_trunk"),
        (PROJECT / "data/learned_untied", "untied"),
        (PROJECT / "data/oracle_shared", "shared_trunk_oracle"),
    ]:
        if not base_dir.exists(): continue
        
        # Find bs+1 and bs-1 directories
        plus_dirs = list(base_dir.glob(f"*{cond.split('_oracle')[0]}*bs+1*"))
        minus_dirs = list(base_dir.glob(f"*{cond.split('_oracle')[0]}*bs-1*"))
        
        if plus_dirs and minus_dirs:
            label = f"{'oracle' if 'oracle' in cond else 'learned'}_{cond.split('_oracle')[0]}"
            r = analyze_pair(plus_dirs[0], minus_dirs[0], label)
            results.append(r)
            print(f"\n{label}:")
            for cat, vals in r.get("categories", {}).items():
                print(f"  {cat}: {vals}")
    
    # Summary
    write_json = lambda p, o: p.write_text(json.dumps(o, indent=2, sort_keys=True, default=str)+"\n")
    write_json(out / "span_discovery_analysis.json", results)
    
    lines = ["# research span discovery analysis", ""]
    for r in results:
        lines.append(f"## {r['label']}")
        cats = r.get("categories", {})
        gt = cats.get("graph_transfer", {})
        if gt.get("n", 0) > 0:
            frac = gt["opposite"] / gt["n"]
            lines.append(f"- Graph-transfer sign reversal: {gt['opposite']}/{gt['n']} = {frac:.3f}")
        da = cats.get("direct_anchor", {})
        if da.get("n", 0) > 0:
            frac = da["opposite"] / da["n"]
            lines.append(f"- Direct-anchor sign reversal: {da['opposite']}/{da['n']} = {frac:.3f}")
        un = cats.get("unchanged", {})
        if un.get("n", 0) > 0:
            frac = un["same_sign"] / un["n"]
            lines.append(f"- Unchanged same-sign: {un['same_sign']}/{un['n']} = {frac:.3f}")
        for mk in ["matching_eval_plus", "matching_eval_minus"]:
            mm = cats.get(mk, {})
            if mm.get("n", 0) > 0:
                lines.append(f"- {mk}: cand_acc={mm['cand_acc']:.3f}, other_acc={mm['other_acc']:.3f} (n={mm['n']})")
        lines.append("")
    
    (out / "span_discovery_analysis.md").write_text("\n".join(lines))
    print(json.dumps({"status": "ANALYSIS_COMPLETE",
                      "md": str(out / "span_discovery_analysis.md")}, indent=2))

if __name__ == "__main__":
    main()
