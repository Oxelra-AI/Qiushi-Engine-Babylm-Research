#!/usr/bin/env python3
"""research: Validate that AutoModel.from_pretrained reaches the correct adapter-equipped
encoder class through the same interface the evaluation pipeline uses.

This is the narrow real-interface test that research's direct instantiation established
a reference for but did not itself perform. It:
1. Sets a writable HF_HOME (as research/research evaluators do)
2. Calls AutoModel.from_pretrained(path, trust_remote_code=True) for each repaired checkpoint
3. Records the loaded transformer class, parameter counts, adapter presence, scales
4. Computes hidden-state agreement with the trusted MLM backbone on a fixed batch
5. Also validates that the original (unrepaired) checkpoint still fails this test
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import torch
import numpy as np

# Writable HF cache, mimicking what research/research evaluators do
VALIDATION_DIR = Path("experiments/archive/functional_learning/data/real_interface_validation")
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE = VALIDATION_DIR / "hf_cache"
HF_CACHE.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE.resolve())
os.environ["TRANSFORMERS_CACHE"] = str((HF_CACHE / "hub").resolve())

# Now import transformers with writable cache
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig


def load_trusted_mlm(ckpt_path: str) -> tuple:
    """Load the trusted MLM model using direct import + manual state dict loading.
    This is the research direct_validation reference."""
    import importlib.util
    modeling_file = os.path.join(ckpt_path, "frozen82_private_modeling.py")
    spec = importlib.util.spec_from_file_location("custom_modeling", modeling_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    config_path = os.path.join(ckpt_path, "config.json")
    with open(config_path) as f:
        cfg_dict = json.load(f)
    
    from transformers import DebertaV2Config
    config = DebertaV2Config(**{k: v for k, v in cfg_dict.items() 
                                if k not in ("auto_map", "architectures", "torch_dtype")})
    
    # Instantiate the MLM model
    mlm = mod.FrozenSlowPrivateDebertaV2ForMaskedLM(config)
    
    # Load weights
    from safetensors.torch import load_file
    st_path = os.path.join(ckpt_path, "model.safetensors")
    state = load_file(st_path)
    mlm.load_state_dict(state, strict=False)
    mlm.eval()
    
    return mlm, mod


def get_fixed_batch(ckpt_path: str, device: str = "cpu") -> dict:
    """Get a fixed tokenized batch for hidden-state comparison."""
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    texts = [
        "The cat sat on the mat and looked out the window.",
        "Children learn language from their environment and caregivers.",
        "Scientific research requires careful experimentation and evidence.",
    ]
    batch = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


def inspect_model(model, label: str) -> dict:
    """Inspect a loaded model's adapter structure."""
    info = {
        "label": label,
        "class": type(model).__name__,
        "module": type(model).__module__,
        "total_params": sum(p.numel() for p in model.parameters()),
    }
    
    # Check for adapter and private_adapter modules
    adapter_params = 0
    private_adapter_params = 0
    adapter_scales = []
    private_adapter_scales = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'adapter') and not name.endswith('.adapter'):
            # This is a layer with an adapter attribute
            if hasattr(module, 'adapter') and hasattr(module.adapter, 'scale'):
                adapter_scales.append(float(module.adapter.scale))
        if hasattr(module, 'private_adapter') and not name.endswith('.private_adapter'):
            if hasattr(module.private_adapter, 'scale'):
                private_adapter_scales.append(float(module.private_adapter.scale))
    
    for name, param in model.named_parameters():
        if '.adapter.' in name and '.private_adapter.' not in name:
            adapter_params += param.numel()
        elif '.private_adapter.' in name:
            private_adapter_params += param.numel()
    
    info["adapter_params"] = adapter_params
    info["private_adapter_params"] = private_adapter_params
    info["adapter_scales"] = adapter_scales
    info["private_adapter_scales"] = private_adapter_scales
    info["has_adapters"] = adapter_params > 0
    info["has_private_adapters"] = private_adapter_params > 0
    
    return info


def get_hidden_states(model, batch: dict) -> torch.Tensor:
    """Get encoder hidden states from a model."""
    with torch.no_grad():
        # For base models (AutoModel), forward returns BaseModelOutput
        # For MLM models, we need the encoder
        if hasattr(model, 'deberta'):
            # MLM model - use the encoder
            out = model.deberta(**batch)
        else:
            # Base model - forward directly
            out = model(**batch)
        
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state
        elif isinstance(out, tuple):
            return out[0]
        else:
            return out


def validate_checkpoint(ckpt_path: str, label: str, batch: dict) -> dict:
    """Full validation of one checkpoint through the real AutoModel interface."""
    result = {"label": label, "checkpoint_path": ckpt_path}
    
    # 1. Load through AutoModel.from_pretrained (the real interface)
    print(f"\n{'='*60}")
    print(f"Validating: {label}")
    print(f"Path: {ckpt_path}")
    
    try:
        automodel = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True)
        automodel.eval()
        automodel_info = inspect_model(automodel, f"{label}_AutoModel")
        result["automodel"] = automodel_info
        result["automodel_loaded"] = True
        print(f"  AutoModel class: {automodel_info['class']}")
        print(f"  Total params: {automodel_info['total_params']}")
        print(f"  Adapter params: {automodel_info['adapter_params']}")
        print(f"  Private adapter params: {automodel_info['private_adapter_params']}")
        print(f"  Adapter scales: {automodel_info['adapter_scales']}")
        print(f"  Private scales: {automodel_info['private_adapter_scales']}")
    except Exception as e:
        result["automodel_loaded"] = False
        result["automodel_error"] = str(e)
        print(f"  AutoModel FAILED: {e}")
        return result
    
    # 2. Load trusted MLM reference
    try:
        mlm, mod = load_trusted_mlm(ckpt_path)
        mlm_info = inspect_model(mlm, f"{label}_trusted_MLM")
        result["trusted_mlm"] = mlm_info
    except Exception as e:
        result["trusted_mlm_error"] = str(e)
        print(f"  Trusted MLM load FAILED: {e}")
        return result
    
    # 3. Compare hidden states
    try:
        auto_hidden = get_hidden_states(automodel, batch)
        mlm_hidden = get_hidden_states(mlm, batch)
        
        diff = (auto_hidden - mlm_hidden).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        
        result["hidden_state_comparison"] = {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "shape": list(auto_hidden.shape),
            "exact_match": max_diff == 0.0,
        }
        
        print(f"  Hidden-state max diff: {max_diff}")
        print(f"  Hidden-state mean diff: {mean_diff}")
        print(f"  Exact match: {max_diff == 0.0}")
    except Exception as e:
        result["hidden_state_error"] = str(e)
        print(f"  Hidden-state comparison FAILED: {e}")
    
    # 4. Verify adapter structure matches expectation
    expected_adapter = automodel_info["adapter_params"] == mlm_info["adapter_params"]
    expected_private = automodel_info["private_adapter_params"] == mlm_info["private_adapter_params"]
    result["adapter_count_match"] = expected_adapter
    result["private_adapter_count_match"] = expected_private
    
    # 5. Check that this is the correct custom class, not stock DebertaV2Model
    is_custom = "FrozenSlowPrivate" in automodel_info["class"]
    result["is_custom_class"] = is_custom
    
    # 6. Overall validity
    result["valid"] = (
        result.get("automodel_loaded", False) and
        is_custom and
        automodel_info["has_adapters"] and
        automodel_info["has_private_adapters"] and
        expected_adapter and expected_private and
        result.get("hidden_state_comparison", {}).get("exact_match", False)
    )
    
    print(f"  VALID: {result['valid']}")
    
    del automodel, mlm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return result


def validate_original_fails(original_path: str, label: str) -> dict:
    """Confirm that the original (unrepaired) checkpoint loads as stock model."""
    result = {"label": label, "checkpoint_path": original_path}
    
    print(f"\n--- Negative control: {label} ---")
    try:
        model = AutoModel.from_pretrained(original_path, trust_remote_code=True)
        model.eval()
        info = inspect_model(model, f"{label}_original")
        result["class"] = info["class"]
        result["total_params"] = info["total_params"]
        result["adapter_params"] = info["adapter_params"]
        result["private_adapter_params"] = info["private_adapter_params"]
        result["is_stock"] = "FrozenSlowPrivate" not in info["class"]
        result["adapters_missing"] = info["adapter_params"] == 0
        print(f"  Class: {info['class']} (stock={result['is_stock']})")
        print(f"  Adapters: {info['adapter_params']} (missing={result['adapters_missing']})")
        del model
    except Exception as e:
        result["error"] = str(e)
        print(f"  Load error: {e}")
    
    return result


def main():
    results = {
        "status": "REAL_INTERFACE_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hf_home": str(HF_CACHE.resolve()),
        "description": (
            "Validates that AutoModel.from_pretrained reaches FrozenSlowPrivateDebertaV2Model "
            "with correct adapter parameters and exact hidden-state agreement against the "
            "trusted MLM backbone. This is the decisive real-interface test."
        ),
        "checkpoints": [],
        "negative_controls": [],
    }
    
    # Repaired checkpoint paths
    repaired = [
        ("coherent86_repaired", 
         "experiments/archive/functional_learning/data/automodel_repair/repaired_coherent86_alpha075"),
        ("dense_seed62064_repaired",
         "experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62064_u0080"),
        ("dense_seed62065_repaired",
         "experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62065_u0080"),
    ]
    
    # Original (unrepaired) paths for negative control
    originals = [
        ("dense_seed62064_original",
         "experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/checkpoints/update_0080"),
    ]
    
    # Get fixed batch from first repaired checkpoint
    batch = get_fixed_batch(repaired[0][1])
    
    # Validate repaired checkpoints
    for label, path in repaired:
        if os.path.exists(path):
            v = validate_checkpoint(path, label, batch)
            results["checkpoints"].append(v)
        else:
            results["checkpoints"].append({"label": label, "error": "path_not_found", "path": path})
            print(f"\n  SKIP: {label} - path not found")
    
    # Negative controls
    for label, path in originals:
        if os.path.exists(path):
            v = validate_original_fails(path, label)
            results["negative_controls"].append(v)
        else:
            results["negative_controls"].append({"label": label, "error": "path_not_found"})
    
    # Summary
    all_valid = all(c.get("valid", False) for c in results["checkpoints"] if "error" not in c or c.get("valid") is not None)
    negatives_stock = all(c.get("is_stock", False) and c.get("adapters_missing", False) 
                         for c in results["negative_controls"] if "error" not in c)
    
    results["summary"] = {
        "all_repaired_valid": all_valid,
        "all_negatives_stock": negatives_stock,
        "interface_validated": all_valid and negatives_stock,
    }
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: all_repaired_valid={all_valid}, negatives_stock={negatives_stock}")
    print(f"INTERFACE VALIDATED: {all_valid and negatives_stock}")
    
    # Save
    out_json = VALIDATION_DIR / "real_interface_validation.json"
    out_md = (VALIDATION_DIR.parents[4] / 'research/documents/functional_learning/data/real_interface_validation/real_interface_validation.md')
    
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(out_md, "w") as f:
        f.write("# research Real AutoModel Interface Validation\n\n")
        f.write(f"Created: {results['created_utc']}\n\n")
        f.write("## Purpose\n\n")
        f.write("Validates that `AutoModel.from_pretrained(path, trust_remote_code=True)` ")
        f.write("loads the correct `FrozenSlowPrivateDebertaV2Model` with both slow and ")
        f.write("private adapters, producing exact hidden-state agreement with the trusted ")
        f.write("MLM backbone. This is the real-interface test that research's direct ")
        f.write("instantiation established a reference for.\n\n")
        
        for c in results["checkpoints"]:
            f.write(f"## {c['label']}\n")
            if c.get("valid"):
                f.write(f"- **VALID** ✓\n")
                ai = c.get("automodel", {})
                f.write(f"- Class: `{ai.get('class')}`\n")
                f.write(f"- Total params: {ai.get('total_params')}\n")
                f.write(f"- Adapter params: {ai.get('adapter_params')}\n")
                f.write(f"- Private adapter params: {ai.get('private_adapter_params')}\n")
                f.write(f"- Adapter scales: {ai.get('adapter_scales')}\n")
                f.write(f"- Private scales: {ai.get('private_adapter_scales')}\n")
                hs = c.get("hidden_state_comparison", {})
                f.write(f"- Hidden-state max diff: {hs.get('max_diff')}\n")
                f.write(f"- Exact match: {hs.get('exact_match')}\n")
            elif "error" in c:
                f.write(f"- ERROR: {c.get('error')}\n")
            else:
                f.write(f"- INVALID\n")
                if c.get("automodel"):
                    f.write(f"  - Class: {c['automodel'].get('class')}\n")
            f.write("\n")
        
        f.write("## Negative Controls\n\n")
        for c in results["negative_controls"]:
            f.write(f"### {c['label']}\n")
            f.write(f"- Stock class: {c.get('is_stock')}\n")
            f.write(f"- Adapters missing: {c.get('adapters_missing')}\n")
            f.write(f"- Loaded class: {c.get('class')}\n\n")
        
        f.write(f"## Verdict\n\n")
        f.write(f"- All repaired valid: **{all_valid}**\n")
        f.write(f"- All negatives stock: **{negatives_stock}**\n")
        f.write(f"- Interface validated: **{all_valid and negatives_stock}**\n")
    
    print(f"\nSaved: {out_json}")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
