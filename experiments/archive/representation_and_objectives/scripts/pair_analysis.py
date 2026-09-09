#!/usr/bin/env python3
"""research: row-paired comparison-channel analysis.

Compare two bridge-sign cells from the same intervention.
Key test: do graph-transfer d_e values reverse between signs?
"""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plus_dir", type=Path, help="bs+1 output directory")
    ap.add_argument("minus_dir", type=Path, help="bs-1 output directory")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    
    out = args.out or args.plus_dir.parent / f"pair_analysis_{args.plus_dir.name.rsplit('_seed',1)[-1] if 'seed' in args.plus_dir.name else 'results'}"
    out.mkdir(parents=True, exist_ok=True)
    
    # Find state prediction files
    # Check both direct and subdirectory locations
    plus_files = sorted(args.plus_dir.glob("state_predictions.jsonl")) or sorted(args.plus_dir.glob("*/state_predictions.jsonl"))
    minus_files = sorted(args.minus_dir.glob("state_predictions.jsonl")) or sorted(args.minus_dir.glob("*/state_predictions.jsonl"))
    if not plus_files or not minus_files:
        print(f"ERROR: missing state_predictions.jsonl in {args.plus_dir} or {args.minus_dir}")
        sys.exit(1)
    
    plus_sr = load_jsonl(plus_files[0])
    minus_sr = load_jsonl(minus_files[0])
    
    # Index by query_key + candidate
    def index_rows(rows):
        d = {}
        for r in rows:
            key = (r.get("query_key") or r.get("row_id"), r.get("candidate"))
            d[key] = r
        return d
    
    pidx = index_rows(plus_sr)
    midx = index_rows(minus_sr)
    
    # Analyze row-paired d_e by family
    families = {
        "direct_anchor": lambda r: "h0_dax" in str(r.get("relation","")) or "h2_norp" in str(r.get("relation","")),
        "graph_transfer": lambda r: "h1_mep" in str(r.get("relation","")) or "h3_ziv" in str(r.get("relation","")),
    }
    
    results = {}
    for fam, pred in families.items():
        # Get paired d_e values (one row per event, using candidate_index=0)
        pairs = []
        for key, pr in pidx.items():
            if pr.get("candidate_index") != 0: continue
            if not pred(pr): continue
            if key not in midx: continue
            mr = midx[key]
            pairs.append({
                "key": str(key),
                "relation": pr.get("relation"),
                "is_changed": pr.get("is_changed"),
                "d_e_plus": pr["d_e"],
                "d_e_minus": mr["d_e"],
                "same_sign": (pr["d_e"] >= 0) == (mr["d_e"] >= 0),
                "opposite_sign": (pr["d_e"] >= 0) != (mr["d_e"] >= 0),
            })
        
        n = len(pairs)
        opp = sum(1 for p in pairs if p["opposite_sign"])
        same = sum(1 for p in pairs if p["same_sign"])
        results[fam] = {
            "n": n,
            "opposite_sign_frac": opp/n if n else 0,
            "same_sign_frac": same/n if n else 0,
            "mean_d_e_plus": sum(p["d_e_plus"] for p in pairs)/n if n else 0,
            "mean_d_e_minus": sum(p["d_e_minus"] for p in pairs)/n if n else 0,
        }
    
    # Unchanged analysis (both signs should preserve)
    unchanged_pairs = []
    for key, pr in pidx.items():
        if pr.get("candidate_index") != 0: continue
        if pr.get("is_changed"): continue
        if key not in midx: continue
        mr = midx[key]
        unchanged_pairs.append({
            "d_e_plus": pr["d_e"],
            "d_e_minus": mr["d_e"],
            "same_sign": (pr["d_e"] >= 0) == (mr["d_e"] >= 0),
        })
    
    n_unch = len(unchanged_pairs)
    unch_same = sum(1 for p in unchanged_pairs if p["same_sign"])
    results["unchanged"] = {
        "n": n_unch,
        "same_sign_frac": unch_same/n_unch if n_unch else 0,
    }
    
    # Comparison analysis (if available)
    plus_cf = sorted(args.plus_dir.glob("comparison_predictions.jsonl")) or sorted(args.plus_dir.glob("*/comparison_predictions.jsonl"))
    minus_cf = sorted(args.minus_dir.glob("comparison_predictions.jsonl")) or sorted(args.minus_dir.glob("*/comparison_predictions.jsonl"))
    if plus_cf and minus_cf:
        plus_cr = load_jsonl(plus_cf[0])
        minus_cr = load_jsonl(minus_cf[0])
        cidx_p = {r["row_id"]: r for r in plus_cr}
        cidx_m = {r["row_id"]: r for r in minus_cr}
        cmp_pairs = []
        for rid, pr in cidx_p.items():
            if rid not in cidx_m: continue
            mr = cidx_m[rid]
            product_p = pr.get("d_e1", 0) * pr.get("d_e2", 0)
            product_m = mr.get("d_e1", 0) * mr.get("d_e2", 0)
            cmp_pairs.append({
                "product_plus": product_p,
                "product_minus": product_m,
                "same_sign": (product_p >= 0) == (product_m >= 0),
            })
        n_cmp = len(cmp_pairs)
        cmp_same = sum(1 for p in cmp_pairs if p["same_sign"])
        results["comp_product_stability"] = {
            "n": n_cmp,
            "same_sign_frac": cmp_same/n_cmp if n_cmp else 0,
        }
    
    # Also read central_eval from result.json
    for tag, d in [("plus", args.plus_dir), ("minus", args.minus_dir)]:
        rfiles = sorted(d.glob("result.json")) or sorted(d.glob("*/result.json"))
        if rfiles:
            rj = json.loads(rfiles[0].read_text())
            results[f"central_eval_{tag}"] = rj.get("central_eval", {})
    
    # Summary
    lines = ["# research row-paired sign analysis", ""]
    lines.append("| family | n | metric | value |")
    lines.append("|---|---:|---|---:|")
    for fam, data in results.items():
        if fam.startswith("central_eval"): continue
        for k, v in data.items():
            if k == "n": continue
            lines.append(f"| {fam} | {data['n']} | {k} | {v:.4f} |")
    
    lines.append("")
    lines.append("## Central eval metrics")
    lines.append("")
    for tag in ["plus", "minus"]:
        key = f"central_eval_{tag}"
        if key in results:
            ce = results[key]
            lines.append(f"### Bridge sign {'+'if tag=='plus' else '-'}1")
            for k, v in sorted(ce.items()):
                if isinstance(v, (int, float)):
                    lines.append(f"- {k}: {v}")
    
    (out / "pair_analysis.md").write_text("\n".join(lines) + "\n")
    (out / "pair_analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    
    print(json.dumps({
        "status": "PAIR_ANALYSIS_COMPLETE",
        "summary": results,
    }, indent=2))

if __name__ == "__main__":
    main()
