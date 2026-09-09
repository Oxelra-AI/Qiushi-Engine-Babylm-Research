#!/usr/bin/env python3
"""research: Cross-data binding comparison.

Runs the counterbalanced entity-event-state binding evaluator on stock DeBERTa
models trained on different data treatments at matched exposures.

Tests the question: does compact-reinvest training specifically
strengthen entity binding compared to clean/extractive data?

Models:
  - Stock DeBERTa compact-reinvest 80M (research chck_80M)
  - Stock DeBERTa clean-Qwen 80M (research chck_80M)  
  - Stock DeBERTa extractive-balanced 80M/100M (research)
  - Stock DeBERTa extractive-wide 80M/100M (research)
  - Stock DeBERTa compact-reinvest 100M (research chck_100M)

All CPU-only inference on the annotated binding substrate (1,088 eval items).
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, hashlib, time, os, sys
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = _public_path('.')  # -> project root
EVAL_DATA = _public_path('experiments/archive/representation_and_objectives/data/annotated_corpus')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/cross_data_binding_comparison')

# Model registry: {label: (path_relative_to_root, data_treatment, exposure_M)}
MODELS = {
    "compact_reinvest_stock_80M": (
        "experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
        "compact_reinvest", 80
    ),
    "compact_reinvest_stock_100M": (
        "experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M",
        "compact_reinvest", 100
    ),
    "clean_qwen_stock_80M": (
        "experiments/archive/frontier_consolidation/training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_80M",
        "clean_qwen", 80
    ),
    "extractive_balanced_stock_80M": (
        "experiments/archive/frontier_consolidation/training/runs/extractive_balanced_deberta100M_seed43022/hf_model/chck_80M",
        "extractive_balanced", 80
    ),
    "extractive_balanced_stock_100M": (
        "experiments/archive/frontier_consolidation/training/runs/extractive_balanced_deberta100M_seed43022/hf_model/chck_100M",
        "extractive_balanced", 100
    ),
    "extractive_wide_stock_80M": (
        "experiments/archive/frontier_consolidation/training/runs/extractive_wide_deberta100M_seed43022/hf_model/chck_80M",
        "extractive_wide", 80
    ),
    "extractive_wide_stock_100M": (
        "experiments/archive/frontier_consolidation/training/runs/extractive_wide_deberta100M_seed43022/hf_model/chck_100M",
        "extractive_wide", 100
    ),
}


def evaluate_binding(model_path, eval_rows, device="cpu", batch_size=32):
    """Run binding evaluation on a single model. Returns per-item results."""
    os.environ["HF_HOME"] = str(_public_path('experiments/archive/frontier_consolidation/data/cross_data_binding_comparison/hf_cache'))
    os.environ["TRANSFORMERS_CACHE"] = str(_public_path('experiments/archive/frontier_consolidation/data/cross_data_binding_comparison/hf_cache'))
    
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    mask_id = tok.mask_token_id
    
    # Pre-encode answer/foil tokens
    for r in eval_rows:
        aid = tok.encode(" " + r["answer"], add_special_tokens=False)
        fid = tok.encode(" " + r["foil"], add_special_tokens=False)
        r["_answer_id"] = aid[1] if len(aid) > 1 else aid[0]
        r["_foil_id"] = fid[1] if len(fid) > 1 else fid[0]
    
    results = []
    t0 = time.time()
    for i in range(0, len(eval_rows), batch_size):
        batch = eval_rows[i:i + batch_size]
        texts = [r["text"] for r in batch]
        enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits = model(**enc).logits
        
        for j, r in enumerate(batch):
            ids = enc["input_ids"][j]
            mask_positions = (ids == mask_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                results.append({"id": r.get("id",""), "correct": False, "margin": -999, "error": "no_mask"})
                continue
            mp = mask_positions[0].item()
            ans_logit = logits[j, mp, r["_answer_id"]].item()
            foil_logit = logits[j, mp, r["_foil_id"]].item()
            margin = ans_logit - foil_logit
            results.append({
                "id": r.get("id", ""),
                "kind": r["kind"],
                "split": r["split"],
                "family": r.get("family", ""),
                "is_affected": r.get("is_affected_query"),
                "correct": margin > 0,
                "margin": margin,
            })
    
    elapsed = time.time() - t0
    # Clean up model from memory
    del model
    del tok
    torch.cuda.empty_cache() if device != "cpu" else None
    
    return results, elapsed


def aggregate_binding(results):
    """Compute binding metrics from per-item results."""
    held = [r for r in results if r["split"] == "eval_held_recomb" and r["kind"] == "binding"]
    affected = [r for r in held if r["is_affected"] == True]
    unaffected = [r for r in held if r["is_affected"] == False]
    
    aff_acc = sum(r["correct"] for r in affected) / max(len(affected), 1)
    unaff_acc = sum(r["correct"] for r in unaffected) / max(len(unaffected), 1)
    held_acc = sum(r["correct"] for r in held) / max(len(held), 1)
    
    multi = [r for r in results if r["kind"] == "multi_event"]
    multi_acc = sum(r["correct"] for r in multi) / max(len(multi), 1) if multi else None
    
    # By family
    by_family = defaultdict(lambda: {"aff_c": 0, "aff_n": 0, "unaff_c": 0, "unaff_n": 0})
    for r in held:
        fam = r["family"]
        if r["is_affected"]:
            by_family[fam]["aff_c"] += int(r["correct"])
            by_family[fam]["aff_n"] += 1
        else:
            by_family[fam]["unaff_c"] += int(r["correct"])
            by_family[fam]["unaff_n"] += 1
    
    by_split = defaultdict(lambda: {"correct": 0, "n": 0, "margin_sum": 0.0})
    for r in results:
        s = r["split"]
        by_split[s]["correct"] += int(r["correct"])
        by_split[s]["n"] += 1
        by_split[s]["margin_sum"] += r["margin"]
    
    return {
        "held_recomb_accuracy": held_acc,
        "affected_accuracy": aff_acc,
        "affected_n": len(affected),
        "unaffected_accuracy": unaff_acc,
        "unaffected_n": len(unaffected),
        "EEBF_composite": (aff_acc + unaff_acc) / 2,
        "multi_event_accuracy": multi_acc,
        "multi_event_n": len(multi),
        "affected_margin_mean": sum(r["margin"] for r in affected) / max(len(affected), 1),
        "unaffected_margin_mean": sum(r["margin"] for r in unaffected) / max(len(unaffected), 1),
        "by_family": {fam: {
            "affected_acc": v["aff_c"] / max(v["aff_n"], 1),
            "unaffected_acc": v["unaff_c"] / max(v["unaff_n"], 1),
            "affected_n": v["aff_n"],
            "unaffected_n": v["unaff_n"],
        } for fam, v in sorted(by_family.items())},
        "by_split": {k: {
            "accuracy": v["correct"] / v["n"],
            "n": v["n"],
            "margin_mean": v["margin_sum"] / v["n"]
        } for k, v in by_split.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of model labels to run")
    args = ap.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Validate model paths
    available = {}
    for label, (rel_path, treatment, exposure) in MODELS.items():
        full = ROOT / rel_path
        exists = full.exists()
        available[label] = {
            "path": str(full),
            "treatment": treatment,
            "exposure_M": exposure,
            "exists": exists,
        }
    
    if args.plan_only:
        plan = {
            "status": "CROSS_DATA_BINDING_PLAN",
            "eval_data": str(EVAL_DATA),
            "eval_data_exists": EVAL_DATA.exists(),
            "models": available,
            "models_found": sum(1 for v in available.values() if v["exists"]),
            "models_missing": [k for k, v in available.items() if not v["exists"]],
        }
        plan_path = _public_path('experiments/archive/frontier_consolidation/data/cross_data_binding_comparison/binding_comparison_plan.json')
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
        return
    
    # Load eval data once
    eval_path = _public_path('experiments/archive/representation_and_objectives/data/annotated_corpus/eval.jsonl')
    if not eval_path.exists():
        print(f"ERROR: eval data not found at {eval_path}")
        sys.exit(1)
    raw_rows = [json.loads(x) for x in eval_path.read_text().splitlines() if x.strip()]
    print(f"Loaded {len(raw_rows)} eval records from {eval_path}")
    
    # Select models to run
    run_labels = args.models if args.models else list(MODELS.keys())
    
    all_results = {}
    for label in run_labels:
        if label not in available or not available[label]["exists"]:
            print(f"SKIP {label}: not found")
            continue
        
        model_path = available[label]["path"]
        treatment = available[label]["treatment"]
        exposure = available[label]["exposure_M"]
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {label} ({treatment}, {exposure}M)")
        print(f"  path: {model_path}")
        
        # Deep copy eval rows so _answer_id/_foil_id don't persist
        import copy
        eval_rows = copy.deepcopy(raw_rows)
        
        try:
            results, elapsed = evaluate_binding(model_path, eval_rows, device="cpu", batch_size=32)
            metrics = aggregate_binding(results)
            metrics["label"] = label
            metrics["treatment"] = treatment
            metrics["exposure_M"] = exposure
            metrics["model_path"] = model_path
            metrics["elapsed_sec"] = round(elapsed, 1)
            
            all_results[label] = metrics
            
            print(f"  affected_acc: {metrics['affected_accuracy']:.4f}")
            print(f"  unaffected_acc: {metrics['unaffected_accuracy']:.4f}")
            print(f"  EEBF: {metrics['EEBF_composite']:.4f}")
            print(f"  multi_event: {metrics['multi_event_accuracy']}")
            print(f"  elapsed: {elapsed:.1f}s")
            
            # Save individual result
            ind_path = OUT_DIR / f"{label}_binding.json"
            ind_path.write_text(json.dumps(metrics, indent=2) + "\n")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[label] = {"label": label, "error": str(e)}
    
    # Build comparison table
    comparison = {
        "status": "CROSS_DATA_BINDING_COMPARISON",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "eval_records": len(raw_rows),
        "models_evaluated": len([r for r in all_results.values() if "error" not in r]),
    }
    
    # 80M comparison table
    table_80M = {}
    for label, m in all_results.items():
        if m.get("exposure_M") == 80 and "error" not in m:
            table_80M[label] = {
                "treatment": m["treatment"],
                "affected_acc": round(m["affected_accuracy"], 4),
                "unaffected_acc": round(m["unaffected_accuracy"], 4),
                "EEBF": round(m["EEBF_composite"], 4),
                "multi_event": round(m["multi_event_accuracy"], 4) if m["multi_event_accuracy"] is not None else None,
                "affected_margin": round(m["affected_margin_mean"], 4),
                "unaffected_margin": round(m["unaffected_margin_mean"], 4),
            }
    
    # 100M comparison table
    table_100M = {}
    for label, m in all_results.items():
        if m.get("exposure_M") == 100 and "error" not in m:
            table_100M[label] = {
                "treatment": m["treatment"],
                "affected_acc": round(m["affected_accuracy"], 4),
                "unaffected_acc": round(m["unaffected_accuracy"], 4),
                "EEBF": round(m["EEBF_composite"], 4),
                "multi_event": round(m["multi_event_accuracy"], 4) if m["multi_event_accuracy"] is not None else None,
                "affected_margin": round(m["affected_margin_mean"], 4),
                "unaffected_margin": round(m["unaffected_margin_mean"], 4),
            }
    
    # Compact-vs-clean deltas at 80M
    deltas_80M = {}
    compact_80 = all_results.get("compact_reinvest_stock_80M", {})
    if "error" not in compact_80:
        for label, m in all_results.items():
            if m.get("exposure_M") == 80 and "error" not in m and label != "compact_reinvest_stock_80M":
                deltas_80M[f"compact_minus_{m['treatment']}"] = {
                    "affected_delta": round(compact_80["affected_accuracy"] - m["affected_accuracy"], 4),
                    "unaffected_delta": round(compact_80["unaffected_accuracy"] - m["unaffected_accuracy"], 4),
                    "EEBF_delta": round(compact_80["EEBF_composite"] - m["EEBF_composite"], 4),
                    "multi_event_delta": round(
                        (compact_80["multi_event_accuracy"] or 0) - (m["multi_event_accuracy"] or 0), 4
                    ),
                    "affected_margin_delta": round(
                        compact_80["affected_margin_mean"] - m["affected_margin_mean"], 4
                    ),
                }
    
    comparison["table_80M"] = table_80M
    comparison["table_100M"] = table_100M
    comparison["compact_minus_others_80M"] = deltas_80M
    comparison["all_results"] = {k: {
        "treatment": v.get("treatment"),
        "exposure_M": v.get("exposure_M"),
        "affected_acc": round(v["affected_accuracy"], 4) if "affected_accuracy" in v else None,
        "unaffected_acc": round(v["unaffected_accuracy"], 4) if "unaffected_accuracy" in v else None,
        "EEBF": round(v["EEBF_composite"], 4) if "EEBF_composite" in v else None,
        "multi_event": round(v["multi_event_accuracy"], 4) if v.get("multi_event_accuracy") is not None else None,
    } for k, v in all_results.items()}
    
    # Add adapter results from research for comparison
    comparison["adapter_results_from_step240"] = {
        "scale1p75_adapter_82M": {"affected": 0.3177, "unaffected": 0.7552, "EEBF": 0.5365, "multi": 0.500},
        "scale1p75_adapter_84M": {"affected": 0.2865, "unaffected": 0.7708, "EEBF": 0.5286, "multi": 0.4896},
        "scale1p75_adapter_100M": {"affected": 0.2969, "unaffected": 0.7813, "EEBF": 0.5391, "multi": 0.4896},
    }
    
    comparison["interpretation_boundary"] = (
        "A compact-specific binding advantage would unite the data mechanism with A01's architecture route. "
        "Without it, binding weakness is a generic DeBERTa property at this scale, not connected to compact views. "
        "Do not attribute generic weakness to compact data or use it to justify compact-specific architecture changes."
    )
    
    comp_path = _public_path('experiments/archive/frontier_consolidation/data/cross_data_binding_comparison/cross_data_binding_comparison.json')
    comp_path.write_text(json.dumps(comparison, indent=2) + "\n")
    
    # Write readable summary
    md_lines = ["# research: Cross-data entity-event-state binding comparison\n"]
    md_lines.append("## 80M exposure comparison\n")
    md_lines.append("| Model | Treatment | Affected | Unaffected | EEBF | Multi-event | Aff margin |")
    md_lines.append("|-------|-----------|----------|------------|------|-------------|------------|")
    for label, row in sorted(table_80M.items()):
        md_lines.append(
            f"| {label} | {row['treatment']} | {row['affected_acc']:.4f} | "
            f"{row['unaffected_acc']:.4f} | {row['EEBF']:.4f} | "
            f"{row['multi_event'] if row['multi_event'] is not None else 'N/A'} | {row['affected_margin']:.4f} |"
        )
    
    if table_100M:
        md_lines.append("\n## 100M exposure comparison\n")
        md_lines.append("| Model | Treatment | Affected | Unaffected | EEBF | Multi-event | Aff margin |")
        md_lines.append("|-------|-----------|----------|------------|------|-------------|------------|")
        for label, row in sorted(table_100M.items()):
            md_lines.append(
                f"| {label} | {row['treatment']} | {row['affected_acc']:.4f} | "
                f"{row['unaffected_acc']:.4f} | {row['EEBF']:.4f} | "
                f"{row['multi_event'] if row['multi_event'] is not None else 'N/A'} | {row['affected_margin']:.4f} |"
            )
    
    if deltas_80M:
        md_lines.append("\n## Compact-reinvest minus others at 80M\n")
        for name, d in deltas_80M.items():
            md_lines.append(f"- **{name}**: affected Δ={d['affected_delta']:+.4f}, "
                          f"unaffected Δ={d['unaffected_delta']:+.4f}, EEBF Δ={d['EEBF_delta']:+.4f}")
    
    md_lines.append("\n## Scale1.75 adapter results (from research, for reference)\n")
    md_lines.append("| Checkpoint | Affected | Unaffected | EEBF | Multi-event |")
    md_lines.append("|------------|----------|------------|------|-------------|")
    for k, v in comparison["adapter_results_from_step240"].items():
        md_lines.append(f"| {k} | {v['affected']:.4f} | {v['unaffected']:.4f} | {v['EEBF']:.4f} | {v['multi']:.4f} |")
    
    md_path = _public_path('research/documents/frontier_consolidation/data/cross_data_binding_comparison/cross_data_binding_comparison.md')
    md_path.write_text("\n".join(md_lines) + "\n")
    
    print(f"\nResults saved to {OUT_DIR}/")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
