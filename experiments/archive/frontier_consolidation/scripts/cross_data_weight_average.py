#!/usr/bin/env python3
"""research: Cross-data weight averaging for stock DeBERTa models.

Models trained from the same init on different data treatments may have
complementary representations. Average their weights and evaluate.

Key complementarity from extractive experiment:
- Compact: strong on stable families (EWoK, Entity, Supplement, COMPS)
- Extractive_balanced: strong on GlobalPIQA (+3.263 vs compact at 100M)
- Clean_qwen: strong unaffected binding, different bias profile

All models: DebertaV2ForMaskedLM 8×480, vocab 16384, tok SHA a9cbb830,
seed 43022, legal research tokenizer.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, time, os, hashlib, sys
from pathlib import Path
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = _public_path('.')  # project root

# Source models at 100M (all have this checkpoint)
MODELS_100M = {
    "compact": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M'),
    "ext_balanced": _public_path('experiments/archive/frontier_consolidation/training/runs/extractive_balanced_deberta100M_seed43022/hf_model/chck_100M'),
    "ext_wide": _public_path('experiments/archive/frontier_consolidation/training/runs/extractive_wide_deberta100M_seed43022/hf_model/chck_100M'),
}

# Source models at 80M
MODELS_80M = {
    "compact": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M'),
    "clean": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_80M'),
    "ext_balanced": _public_path('experiments/archive/frontier_consolidation/training/runs/extractive_balanced_deberta100M_seed43022/hf_model/chck_80M'),
    "ext_wide": _public_path('experiments/archive/frontier_consolidation/training/runs/extractive_wide_deberta100M_seed43022/hf_model/chck_80M'),
}

OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/cross_data_weight_average')

# Averaging recipes: list of (name, {model_label: weight}, exposure_M)
RECIPES = [
    # 100M recipes
    ("compact_extbal_avg100M", {"compact": 0.5, "ext_balanced": 0.5}, 100),
    ("compact_extwide_avg100M", {"compact": 0.5, "ext_wide": 0.5}, 100),
    ("compact_extbal_extwide_avg100M", {"compact": 1/3, "ext_balanced": 1/3, "ext_wide": 1/3}, 100),
    ("compact_heavy_extbal_avg100M", {"compact": 0.7, "ext_balanced": 0.3}, 100),
    # 80M recipes
    ("compact_clean_avg80M", {"compact": 0.5, "clean": 0.5}, 80),
    ("compact_extbal_avg80M", {"compact": 0.5, "ext_balanced": 0.5}, 80),
    ("all4_avg80M", {"compact": 0.25, "clean": 0.25, "ext_balanced": 0.25, "ext_wide": 0.25}, 80),
]


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def load_state_dict(model_path):
    """Load state dict from a saved HF model."""
    safetensors_path = model_path / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path))
    bin_path = model_path / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(f"No model weights found in {model_path}")


def average_state_dicts(state_dicts, weights):
    """Weighted average of state dicts."""
    assert len(state_dicts) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6
    
    avg = {}
    keys = state_dicts[0].keys()
    for k in keys:
        avg[k] = sum(sd[k].float() * w for sd, w in zip(state_dicts, weights))
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--recipes", nargs="*", default=None, help="Subset of recipe names")
    ap.add_argument("--save-models", action="store_true", help="Save averaged models to disk")
    args = ap.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.plan_only:
        plan = {"status": "CROSS_DATA_AVG_PLAN", "recipes": []}
        for name, weights, exp in RECIPES:
            models = MODELS_100M if exp == 100 else MODELS_80M
            available = {k: str(models.get(k, "?")) for k in weights}
            all_exist = all((models.get(k, Path("/no")).exists()) for k in weights)
            plan["recipes"].append({
                "name": name, "weights": weights, "exposure_M": exp,
                "models": available, "all_available": all_exist,
            })
        (_public_path('experiments/archive/frontier_consolidation/data/cross_data_weight_average/avg_plan.json')).write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
        return
    
    selected_recipes = args.recipes or [r[0] for r in RECIPES]
    
    # Cache loaded state dicts
    sd_cache = {}
    results = {}
    
    for name, weights_dict, exp in RECIPES:
        if name not in selected_recipes:
            continue
        
        models = MODELS_100M if exp == 100 else MODELS_80M
        
        # Check availability
        missing = [k for k in weights_dict if not models.get(k, Path("/no")).exists()]
        if missing:
            print(f"SKIP {name}: missing {missing}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Building average: {name} (exposure {exp}M)")
        print(f"  Weights: {weights_dict}")
        
        # Load state dicts
        sds = []
        ws = []
        for label, w in weights_dict.items():
            cache_key = f"{label}_{exp}M"
            if cache_key not in sd_cache:
                path = models[label]
                print(f"  Loading {label} from {path}")
                sd_cache[cache_key] = load_state_dict(path)
            sds.append(sd_cache[cache_key])
            ws.append(w)
        
        # Average
        avg_sd = average_state_dicts(sds, ws)
        n_params = sum(p.numel() for p in avg_sd.values())
        print(f"  Averaged {len(avg_sd)} tensors, {n_params:,} parameters")
        
        # Save if requested
        if args.save_models:
            save_dir = OUT_DIR / name / "hf_model"
            save_dir.mkdir(parents=True, exist_ok=True)
            # Copy config and tokenizer from first model
            first_model = models[list(weights_dict.keys())[0]]
            import shutil
            for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
                src = first_model / fname
                if src.exists():
                    shutil.copy2(str(src), str(save_dir / fname))
            # Save averaged weights
            from safetensors.torch import save_file
            save_file(avg_sd, str(save_dir / "model.safetensors"))
            model_sha = sha16(save_dir / "model.safetensors")
            print(f"  Saved to {save_dir}, SHA={model_sha}")
        
        # Load into model for inference
        first_model_path = models[list(weights_dict.keys())[0]]
        os.environ["HF_HOME"] = str(_public_path('experiments/archive/frontier_consolidation/data/cross_data_weight_average/hf_cache'))
        os.environ["TRANSFORMERS_CACHE"] = str(_public_path('experiments/archive/frontier_consolidation/data/cross_data_weight_average/hf_cache'))
        
        model = AutoModelForMaskedLM.from_pretrained(
            str(first_model_path), trust_remote_code=True, local_files_only=True
        )
        # Replace weights with averaged
        model.load_state_dict(avg_sd, strict=True)
        model.eval()
        
        # Quick sanity: check model produces finite logits
        tok = AutoTokenizer.from_pretrained(str(first_model_path), trust_remote_code=True, local_files_only=True)
        test_enc = tok("The cup was [MASK].", return_tensors="pt")
        with torch.no_grad():
            logits = model(**test_enc).logits
        finite = logits.isfinite().all().item()
        print(f"  Sanity: finite_logits={finite}")
        
        if not finite:
            results[name] = {"name": name, "error": "non_finite_logits"}
            del model
            continue
        
        # Run binding evaluation on the averaged model
        eval_path = _public_path('experiments/archive/representation_and_objectives/data/annotated_corpus/eval.jsonl')
        if eval_path.exists():
            rows = [json.loads(x) for x in eval_path.read_text().splitlines() if x.strip()]
            import copy
            eval_rows = copy.deepcopy(rows)
            
            mask_id = tok.mask_token_id
            for r in eval_rows:
                aid = tok.encode(" " + r["answer"], add_special_tokens=False)
                fid = tok.encode(" " + r["foil"], add_special_tokens=False)
                r["_answer_id"] = aid[1] if len(aid) > 1 else aid[0]
                r["_foil_id"] = fid[1] if len(fid) > 1 else fid[0]
            
            binding_results = []
            for i in range(0, len(eval_rows), 32):
                batch = eval_rows[i:i+32]
                texts = [r["text"] for r in batch]
                enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**enc).logits
                for j, r in enumerate(batch):
                    ids = enc["input_ids"][j]
                    mp = (ids == mask_id).nonzero(as_tuple=True)[0]
                    if len(mp) == 0:
                        binding_results.append({"correct": False, "split": r["split"], "kind": r["kind"], "is_affected": r.get("is_affected_query")})
                        continue
                    mp = mp[0].item()
                    margin = logits[j, mp, r["_answer_id"]].item() - logits[j, mp, r["_foil_id"]].item()
                    binding_results.append({
                        "correct": margin > 0, "margin": margin,
                        "split": r["split"], "kind": r["kind"],
                        "is_affected": r.get("is_affected_query"),
                    })
            
            held = [r for r in binding_results if r["split"] == "eval_held_recomb" and r["kind"] == "binding"]
            affected = [r for r in held if r["is_affected"] == True]
            unaffected = [r for r in held if r["is_affected"] == False]
            aff_acc = sum(r["correct"] for r in affected) / max(len(affected), 1)
            unaff_acc = sum(r["correct"] for r in unaffected) / max(len(unaffected), 1)
            binding_eebf = (aff_acc + unaff_acc) / 2
        else:
            binding_eebf = None
            aff_acc = None
            unaff_acc = None
        
        results[name] = {
            "name": name,
            "weights": weights_dict,
            "exposure_M": exp,
            "n_params": n_params,
            "finite_logits": finite,
            "binding_eebf": round(binding_eebf, 4) if binding_eebf else None,
            "binding_affected": round(aff_acc, 4) if aff_acc else None,
            "binding_unaffected": round(unaff_acc, 4) if unaff_acc else None,
        }
        
        print(f"  Binding EEBF: {binding_eebf:.4f}" if binding_eebf else "  No binding eval")
        
        del model
        torch.cuda.empty_cache()
    
    summary = {
        "status": "CROSS_DATA_WEIGHT_AVERAGE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "note": "Binding-only evaluation. Full cheap7 evaluation requires GPU. "
                "If binding profiles look promising (better combined EEBF or preserved "
                "complementary strengths), worth running full cheap7 eval.",
    }
    (_public_path('experiments/archive/frontier_consolidation/data/cross_data_weight_average/cross_data_avg_summary.json')).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
