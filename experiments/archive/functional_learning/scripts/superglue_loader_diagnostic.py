#!/usr/bin/env python3
"""research: SuperGLUE loader diagnostic and repair validation.

Scientific purpose
------------------
The SuperGLUE finetune classifier calls AutoModel.from_pretrained, but our custom
checkpoints only register AutoModelForMaskedLM.  This script:

1. Loads via AutoModel (the broken path) and records what class/params result.
2. Loads via AutoModelForMaskedLM (the trusted path) and records class/params.
3. Compares parameter names, counts, missing/unexpected keys.
4. Runs a fixed-batch forward comparison on both to demonstrate functional difference.
5. If a repaired modeling file with AutoModel support exists, validates it too.

This is a diagnostic tool, not a repair.  The repair is repair_automodel.py.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import torch

OUT_ROOT = Path("experiments/archive/functional_learning/data/superglue_loader_validation")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def has_adapter_params(model):
    adapter_names = []
    private_adapter_names = []
    for name, _ in model.named_parameters():
        if ".private_adapter." in name:
            private_adapter_names.append(name)
        elif ".adapter." in name:
            adapter_names.append(name)
    return {
        "adapter_param_count": len(adapter_names),
        "private_adapter_param_count": len(private_adapter_names),
        "adapter_examples": adapter_names[:4],
        "private_adapter_examples": private_adapter_names[:4],
    }


def get_model_class_info(model):
    return {
        "class_name": type(model).__name__,
        "module": type(model).__module__,
        "has_deberta_attr": hasattr(model, "deberta"),
        "has_encoder_attr": hasattr(model, "encoder"),
    }


def make_fixed_batch(tokenizer_path, device="cpu"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    texts = [
        "The cat sat on the mat.",
        "A dog chased the ball across the yard.",
    ]
    enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def forward_hidden_states(model, batch, use_deberta_encoder=False):
    """Get last hidden states from a model."""
    model.eval()
    with torch.no_grad():
        if use_deberta_encoder:
            # For MLM model, extract encoder part
            out = model.deberta(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        else:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        elif isinstance(out, tuple):
            return out[0]
        else:
            return None


def diagnose_one_checkpoint(ckpt_path: str, label: str) -> dict:
    """Run full diagnostic on one checkpoint."""
    from transformers import AutoModel, AutoModelForMaskedLM, AutoConfig

    result = {
        "checkpoint": ckpt_path,
        "label": label,
        "config_info": {},
        "automodel_path": {},
        "automodel_for_mlm_path": {},
        "comparison": {},
    }

    # Read config
    config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    result["config_info"] = {
        "architectures": getattr(config, "architectures", None),
        "auto_map": getattr(config, "auto_map", None),
        "model_type": getattr(config, "model_type", None),
        "adapter_enabled": getattr(config, "adapter_enabled", None),
        "adapter_scale": getattr(config, "adapter_scale", None),
        "private_adapter_enabled": getattr(config, "private_adapter_enabled", None),
        "private_adapter_scale": getattr(config, "private_adapter_scale", None),
    }

    # 1. Load via AutoModel (the SuperGLUE finetune path)
    try:
        auto_model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True)
        auto_model.eval()
        result["automodel_path"] = {
            "success": True,
            "error": None,
            **get_model_class_info(auto_model),
            **count_params(auto_model),
            **has_adapter_params(auto_model),
        }
    except Exception as e:
        result["automodel_path"] = {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }
        auto_model = None

    # 2. Load via AutoModelForMaskedLM (the trusted path)
    try:
        mlm_model = AutoModelForMaskedLM.from_pretrained(ckpt_path, trust_remote_code=True)
        mlm_model.eval()
        result["automodel_for_mlm_path"] = {
            "success": True,
            "error": None,
            **get_model_class_info(mlm_model),
            **count_params(mlm_model),
            **has_adapter_params(mlm_model),
        }
    except Exception as e:
        result["automodel_for_mlm_path"] = {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }
        mlm_model = None

    # 3. Compare parameter names
    if auto_model is not None and mlm_model is not None:
        auto_names = set(n for n, _ in auto_model.named_parameters())
        # For MLM model, the encoder part is under .deberta
        mlm_encoder_names = set()
        for n, _ in mlm_model.named_parameters():
            if n.startswith("deberta."):
                mlm_encoder_names.add(n[len("deberta."):])

        only_in_automodel = sorted(auto_names - mlm_encoder_names)
        only_in_mlm_encoder = sorted(mlm_encoder_names - auto_names)

        # Count adapter-related missing
        missing_adapter = [n for n in only_in_mlm_encoder if ".adapter." in n and ".private_adapter." not in n]
        missing_private = [n for n in only_in_mlm_encoder if ".private_adapter." in n]

        result["comparison"] = {
            "automodel_param_names": len(auto_names),
            "mlm_encoder_param_names": len(mlm_encoder_names),
            "only_in_automodel": only_in_automodel[:10],
            "only_in_automodel_count": len(only_in_automodel),
            "only_in_mlm_encoder": only_in_mlm_encoder[:10],
            "only_in_mlm_encoder_count": len(only_in_mlm_encoder),
            "missing_adapter_params": len(missing_adapter),
            "missing_private_adapter_params": len(missing_private),
        }

        # 4. Forward comparison on a fixed batch
        batch = make_fixed_batch(ckpt_path)

        h_auto = forward_hidden_states(auto_model, batch, use_deberta_encoder=False)
        h_mlm = forward_hidden_states(mlm_model, batch, use_deberta_encoder=True)

        if h_auto is not None and h_mlm is not None:
            diff = (h_auto - h_mlm).abs()
            result["comparison"]["forward_max_abs_diff"] = float(diff.max())
            result["comparison"]["forward_mean_abs_diff"] = float(diff.mean())
            result["comparison"]["forward_hidden_identical"] = bool(diff.max() < 1e-6)
            result["comparison"]["automodel_hidden_norm"] = float(h_auto.norm())
            result["comparison"]["mlm_encoder_hidden_norm"] = float(h_mlm.norm())
        else:
            result["comparison"]["forward_error"] = "Could not get hidden states from one or both models"

    return result


def main():
    results = {"status": "SUPERGLUE_LOADER_VALIDATION", "diagnostics": []}

    # Test checkpoints
    checkpoints = [
        {
            "path": "experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/checkpoints/update_0080",
            "label": "dense_seed62064_u0080",
        },
    ]

    # Check if seed62065 checkpoint exists
    seed65_path = "experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/checkpoints/update_0080"
    if os.path.exists(seed65_path):
        checkpoints.append({"path": seed65_path, "label": "dense_seed62065_u0080"})

    # Check coherent86 from frontier_consolidation
    coherent86_path = "models/frontier"
    if os.path.exists(coherent86_path):
        checkpoints.append({"path": coherent86_path, "label": "coherent86_alpha075"})

    for ckpt in checkpoints:
        print(f"\n=== Diagnosing: {ckpt['label']} ===", flush=True)
        diag = diagnose_one_checkpoint(ckpt["path"], ckpt["label"])
        results["diagnostics"].append(diag)

        # Print key findings
        am = diag["automodel_path"]
        mlm = diag["automodel_for_mlm_path"]
        if am.get("success"):
            print(f"  AutoModel: {am['class_name']} | params={am['total']} | adapter={am['adapter_param_count']} | private={am['private_adapter_param_count']}")
        else:
            print(f"  AutoModel: FAILED - {am.get('error')}")
        if mlm.get("success"):
            print(f"  AutoModelForMaskedLM: {mlm['class_name']} | params={mlm['total']} | adapter={mlm['adapter_param_count']} | private={mlm['private_adapter_param_count']}")
        else:
            print(f"  AutoModelForMaskedLM: FAILED - {mlm.get('error')}")

        comp = diag.get("comparison", {})
        if "forward_hidden_identical" in comp:
            print(f"  Forward match: {comp['forward_hidden_identical']} | max_diff={comp.get('forward_max_abs_diff', 'N/A'):.6e} | mean_diff={comp.get('forward_mean_abs_diff', 'N/A'):.6e}")
        if "missing_adapter_params" in comp:
            print(f"  Missing from AutoModel: {comp['missing_adapter_params']} adapter + {comp['missing_private_adapter_params']} private_adapter params")

    # Determine verdict
    verdicts = []
    for diag in results["diagnostics"]:
        am = diag["automodel_path"]
        comp = diag.get("comparison", {})
        if am.get("success") and am.get("adapter_param_count", 0) == 0:
            verdicts.append(f"{diag['label']}: AutoModel drops ALL adapters ({comp.get('missing_adapter_params', '?')} adapter + {comp.get('missing_private_adapter_params', '?')} private)")
        elif am.get("success") and not comp.get("forward_hidden_identical", True):
            verdicts.append(f"{diag['label']}: AutoModel loads but produces different hidden states (max_diff={comp.get('forward_max_abs_diff', 'N/A')})")
        elif am.get("success") and comp.get("forward_hidden_identical"):
            verdicts.append(f"{diag['label']}: AutoModel loads correctly and matches MLM encoder")

    results["verdicts"] = verdicts
    results["repair_needed"] = any("drops ALL" in v for v in verdicts)

    out_json = OUT_ROOT / "loader_diagnostic.json"
    out_md = (OUT_ROOT.parents[4] / 'research/documents/functional_learning/data/superglue_loader_validation/loader_diagnostic.md')
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(out_md, "w") as f:
        f.write("# research SuperGLUE loader diagnostic\n\n")
        for diag in results["diagnostics"]:
            f.write(f"## {diag['label']}\n")
            f.write(f"- Checkpoint: `{diag['checkpoint']}`\n")
            am = diag["automodel_path"]
            mlm = diag["automodel_for_mlm_path"]
            if am.get("success"):
                f.write(f"- AutoModel → `{am['class_name']}`, {am['total']} params, {am['adapter_param_count']} adapter, {am['private_adapter_param_count']} private_adapter\n")
            else:
                f.write(f"- AutoModel → FAILED: {am.get('error')}\n")
            if mlm.get("success"):
                f.write(f"- AutoModelForMaskedLM → `{mlm['class_name']}`, {mlm['total']} params, {mlm['adapter_param_count']} adapter, {mlm['private_adapter_param_count']} private_adapter\n")
            comp = diag.get("comparison", {})
            if "forward_hidden_identical" in comp:
                f.write(f"- Forward match: `{comp['forward_hidden_identical']}` | max_diff=`{comp.get('forward_max_abs_diff', 'N/A'):.6e}` | mean_diff=`{comp.get('forward_mean_abs_diff', 'N/A'):.6e}`\n")
            if "missing_adapter_params" in comp:
                f.write(f"- Missing from AutoModel: `{comp['missing_adapter_params']}` adapter + `{comp['missing_private_adapter_params']}` private_adapter parameter tensors\n")
            f.write("\n")
        f.write("## Verdict\n")
        for v in verdicts:
            f.write(f"- {v}\n")
        f.write(f"\n**Repair needed:** `{results['repair_needed']}`\n")

    print(f"\nSaved: {out_json}")
    print(json.dumps({"status": results["status"], "repair_needed": results["repair_needed"], "verdicts": verdicts}, indent=2))


if __name__ == "__main__":
    main()
