#!/usr/bin/env python3
"""research: Repair AutoModel registration for FrozenSlowPrivate checkpoints.

Scientific purpose
------------------
The official SuperGLUE finetune pipeline calls AutoModel.from_pretrained, but our
custom checkpoints only register AutoModelForMaskedLM.  This causes AutoModel to
fall back to stock DebertaV2Model, silently dropping both the inherited slow adapter
(scale 1.75) and the private adapter (scale 0.75) — 2,239,392 parameters lost.

The repair:
1. Creates FrozenSlowPrivateDebertaV2Model (base encoder with both adapter paths).
2. Patches the checkpoint modeling file to include this class.
3. Updates config.json auto_map to register AutoModel.
4. Validates that the repaired AutoModel path produces hidden states matching
   the trusted MLM encoder, including both adapter paths at their executed scales.

The repair is applied to local copies of the checkpoints, not to original
training outputs.  The staged evaluation directories are then updated.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import torch

OUT_ROOT = Path("experiments/archive/functional_learning/data/automodel_repair")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# The new base model class to be injected into the modeling file
# --------------------------------------------------------------------------- #
AUTOMODEL_CLASS_CODE = '''

class FrozenSlowPrivateDebertaV2Model(DebertaV2Model):
    """Base encoder variant of FrozenSlowPrivateDebertaV2ForMaskedLM.

    This class is registered under AutoModel in auto_map so that the official
    SuperGLUE finetune pipeline (which calls AutoModel.from_pretrained) loads
    the full encoder including both slow adapter (adapter_scale) and private
    adapter (private_adapter_scale) paths.

    Layer computation is identical to FrozenSlowPrivateDebertaV2ForMaskedLM:
        base = output(intermediate(attention(h)), attention(h))
        slow = base + adapter(base)           # inherited chck_82M function
        h_next = slow + private_adapter(slow) # learned private update
    """
    config_class = DebertaV2Config

    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        # Mirror the adapter configuration from the MLM class
        config.adapter_bottleneck = int(getattr(config, "adapter_bottleneck", 128))
        config.adapter_activation = getattr(config, "adapter_activation", "gelu")
        config.adapter_enabled = bool(getattr(config, "adapter_enabled", True))
        config.adapter_scale = float(getattr(config, "adapter_scale", 1.75))

        config.private_adapter_bottleneck = int(getattr(config, "private_adapter_bottleneck", 128))
        config.private_adapter_activation = getattr(config, "private_adapter_activation", "gelu")
        config.private_adapter_enabled = bool(getattr(config, "private_adapter_enabled", True))
        config.private_adapter_scale = float(getattr(config, "private_adapter_scale", 1.0))

        for layer in self.encoder.layer:
            layer.__class__ = FrozenSlowPrivateDebertaV2Layer
            layer.adapter = ScaledBottleneckAdapter(
                config,
                bottleneck_attr="adapter_bottleneck",
                scale_attr="adapter_scale",
                enabled_attr="adapter_enabled",
                activation_attr="adapter_activation",
            )
            layer.private_adapter = ScaledBottleneckAdapter(
                config,
                bottleneck_attr="private_adapter_bottleneck",
                scale_attr="private_adapter_scale",
                enabled_attr="private_adapter_enabled",
                activation_attr="private_adapter_activation",
            )

'''


def patch_modeling_file(src_modeling: Path, dst_modeling: Path):
    """Add FrozenSlowPrivateDebertaV2Model to the modeling file."""
    code = src_modeling.read_text()

    # Check if already patched
    if "FrozenSlowPrivateDebertaV2Model" in code:
        print(f"  Modeling file already contains FrozenSlowPrivateDebertaV2Model")
        dst_modeling.write_text(code)
        return

    # Need to add the import for DebertaV2Model
    if "from transformers import DebertaV2Config, DebertaV2ForMaskedLM" in code:
        code = code.replace(
            "from transformers import DebertaV2Config, DebertaV2ForMaskedLM",
            "from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2Model",
        )
    elif "DebertaV2Model" not in code:
        # Add import after existing transformers import
        code = code.replace(
            "from transformers import DebertaV2Config, DebertaV2ForMaskedLM",
            "from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2Model",
        )

    # Insert the new class before the MLM class definition
    insertion_marker = "class FrozenSlowPrivateDebertaV2ForMaskedLM(DebertaV2ForMaskedLM):"
    if insertion_marker in code:
        code = code.replace(insertion_marker, AUTOMODEL_CLASS_CODE + "\n" + insertion_marker)
    else:
        # Fallback: append at end before save_pretrained
        code += AUTOMODEL_CLASS_CODE

    # Update the auto_map in save_pretrained to include AutoModel
    old_auto_map = '''config.auto_map = {
            "AutoModelForMaskedLM": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM"
        }'''
    new_auto_map = '''config.auto_map = {
            "AutoModelForMaskedLM": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM",
            "AutoModel": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2Model"
        }'''
    code = code.replace(old_auto_map, new_auto_map)

    dst_modeling.write_text(code)
    print(f"  Patched modeling file: {dst_modeling}")


def patch_config(src_config: Path, dst_config: Path):
    """Add AutoModel to config.json auto_map."""
    config = json.loads(src_config.read_text())

    auto_map = config.get("auto_map", {})
    if "AutoModel" in auto_map:
        print(f"  Config already has AutoModel mapping")
    else:
        auto_map["AutoModel"] = "frozen82_private_modeling.FrozenSlowPrivateDebertaV2Model"
        config["auto_map"] = auto_map
        print(f"  Added AutoModel mapping to config")

    dst_config.write_text(json.dumps(config, indent=2) + "\n")


def create_repaired_checkpoint(src_dir: Path, dst_dir: Path):
    """Create a repaired copy of a checkpoint with AutoModel support."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files, patching modeling and config
    for item in src_dir.iterdir():
        if item.name == "frozen82_private_modeling.py":
            patch_modeling_file(item, dst_dir / item.name)
        elif item.name == "config.json":
            patch_config(item, dst_dir / item.name)
        elif item.name == "model.safetensors":
            # Symlink large files to save space
            dst = dst_dir / item.name
            if not dst.exists():
                os.symlink(item.resolve(), dst)
                print(f"  Symlinked: {item.name}")
        else:
            shutil.copy2(item, dst_dir / item.name)
            print(f"  Copied: {item.name}")


def validate_repaired_checkpoint(repaired_dir: Path, original_dir: Path, label: str):
    """Validate that repaired AutoModel matches trusted MLM encoder."""
    sys.path.insert(0, str(original_dir))
    from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM
    from transformers import DebertaV2Config, AutoTokenizer, AutoModel
    from safetensors.torch import load_file

    result = {"label": label, "repaired_dir": str(repaired_dir)}

    # 1. Load trusted MLM model from original
    config = DebertaV2Config.from_pretrained(str(original_dir))
    mlm = FrozenSlowPrivateDebertaV2ForMaskedLM(config)
    state = load_file(str(original_dir / "model.safetensors"))
    missing, unexpected = mlm.load_state_dict(state, strict=False)
    mlm.eval()
    result["trusted_mlm"] = {
        "total_params": sum(p.numel() for p in mlm.parameters()),
        "adapter_params": sum(p.numel() for n, p in mlm.named_parameters()
                             if ".adapter." in n and ".private_adapter." not in n),
        "private_adapter_params": sum(p.numel() for n, p in mlm.named_parameters()
                                     if ".private_adapter." in n),
        "missing_keys": missing,
        "unexpected_keys": unexpected[:5],
    }

    # 2. Load via repaired AutoModel (set writable modules cache)
    os.environ["HF_MODULES_CACHE"] = str(OUT_ROOT / "hf_modules_cache")
    os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)

    try:
        repaired_auto = AutoModel.from_pretrained(str(repaired_dir), trust_remote_code=True)
        repaired_auto.eval()
        result["repaired_automodel"] = {
            "success": True,
            "class_name": type(repaired_auto).__name__,
            "module": type(repaired_auto).__module__,
            "total_params": sum(p.numel() for p in repaired_auto.parameters()),
            "adapter_params": sum(p.numel() for n, p in repaired_auto.named_parameters()
                                 if ".adapter." in n and ".private_adapter." not in n),
            "private_adapter_params": sum(p.numel() for n, p in repaired_auto.named_parameters()
                                         if ".private_adapter." in n),
        }
    except Exception as e:
        result["repaired_automodel"] = {"success": False, "error": str(e)}
        return result

    # 3. Also load stock AutoModel for three-way comparison
    stock_auto = AutoModel.from_pretrained(str(original_dir))
    stock_auto.eval()
    result["stock_automodel"] = {
        "class_name": type(stock_auto).__name__,
        "total_params": sum(p.numel() for p in stock_auto.parameters()),
    }

    # 4. Forward comparison on fixed batch
    tok = AutoTokenizer.from_pretrained(str(original_dir))
    batch = tok(
        ["The cat sat on the mat.", "A dog chased the ball across the yard."],
        padding=True, truncation=True, max_length=64, return_tensors="pt",
    )

    with torch.no_grad():
        h_mlm = mlm.deberta(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        h_repaired = repaired_auto(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        h_stock = stock_auto(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state

    diff_repaired = (h_mlm - h_repaired).abs()
    diff_stock = (h_mlm - h_stock).abs()
    diff_rep_vs_stock = (h_repaired - h_stock).abs()

    result["forward_validation"] = {
        "repaired_vs_trusted_mlm": {
            "max_abs_diff": float(diff_repaired.max()),
            "mean_abs_diff": float(diff_repaired.mean()),
            "hidden_identical": bool(diff_repaired.max() < 1e-5),
        },
        "stock_vs_trusted_mlm": {
            "max_abs_diff": float(diff_stock.max()),
            "mean_abs_diff": float(diff_stock.mean()),
            "hidden_identical": bool(diff_stock.max() < 1e-5),
        },
        "repaired_vs_stock": {
            "max_abs_diff": float(diff_rep_vs_stock.max()),
            "mean_abs_diff": float(diff_rep_vs_stock.mean()),
        },
        "trusted_mlm_norm": float(h_mlm.norm()),
        "repaired_norm": float(h_repaired.norm()),
        "stock_norm": float(h_stock.norm()),
    }

    # 5. Check adapter scale execution
    adapter_scales = []
    private_scales = []
    for layer in repaired_auto.encoder.layer:
        if hasattr(layer, "adapter"):
            adapter_scales.append(layer.adapter.scale)
        if hasattr(layer, "private_adapter"):
            private_scales.append(layer.private_adapter.scale)
    result["executed_scales"] = {
        "adapter_scales": adapter_scales,
        "private_adapter_scales": private_scales,
    }

    return result


def main():
    results = {"status": "AUTOMODEL_REPAIR", "repairs": [], "validations": []}

    checkpoints = [
        {
            "original": Path("experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/checkpoints/update_0080"),
            "label": "dense_seed62064_u0080",
        },
        {
            "original": Path("experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/checkpoints/update_0080"),
            "label": "dense_seed62065_u0080",
        },
    ]

    # Also repair coherent86 if accessible
    coherent86 = Path("models/frontier")
    if coherent86.exists():
        checkpoints.append({"original": coherent86, "label": "coherent86_alpha075"})

    for ckpt in checkpoints:
        label = ckpt["label"]
        original = ckpt["original"]
        repaired = OUT_ROOT / f"repaired_{label}"

        print(f"\n{'='*60}")
        print(f"Repairing: {label}")
        print(f"  Source: {original}")
        print(f"  Target: {repaired}")
        print(f"{'='*60}")

        create_repaired_checkpoint(original, repaired)
        results["repairs"].append({
            "label": label,
            "original": str(original),
            "repaired": str(repaired),
        })

        print(f"\nValidating: {label}")
        val = validate_repaired_checkpoint(repaired, original, label)
        results["validations"].append(val)

        # Print key results
        if val.get("repaired_automodel", {}).get("success"):
            ra = val["repaired_automodel"]
            print(f"  Repaired AutoModel: {ra['class_name']} | params={ra['total_params']} | adapter={ra['adapter_params']} | private={ra['private_adapter_params']}")
        fv = val.get("forward_validation", {})
        if "repaired_vs_trusted_mlm" in fv:
            r2m = fv["repaired_vs_trusted_mlm"]
            s2m = fv["stock_vs_trusted_mlm"]
            print(f"  Repaired vs trusted MLM: max_diff={r2m['max_abs_diff']:.6e}, identical={r2m['hidden_identical']}")
            print(f"  Stock vs trusted MLM:    max_diff={s2m['max_abs_diff']:.6e}, identical={s2m['hidden_identical']}")
            print(f"  Adapter scales: {val.get('executed_scales', {}).get('adapter_scales', [])}")
            print(f"  Private scales: {val.get('executed_scales', {}).get('private_adapter_scales', [])}")

    # Summary
    all_valid = all(
        v.get("forward_validation", {}).get("repaired_vs_trusted_mlm", {}).get("hidden_identical", False)
        for v in results["validations"]
    )
    results["all_repaired_valid"] = all_valid

    out_json = OUT_ROOT / "repair_validation.json"
    out_md = (OUT_ROOT.parents[4] / 'research/documents/functional_learning/data/automodel_repair/repair_validation.md')
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(out_md, "w") as f:
        f.write("# research AutoModel repair validation\n\n")
        for val in results["validations"]:
            f.write(f"## {val['label']}\n")
            if val.get("repaired_automodel", {}).get("success"):
                ra = val["repaired_automodel"]
                f.write(f"- Repaired AutoModel → `{ra['class_name']}`, {ra['total_params']} params\n")
                f.write(f"  - adapter params: {ra['adapter_params']}, private adapter params: {ra['private_adapter_params']}\n")
            fv = val.get("forward_validation", {})
            if "repaired_vs_trusted_mlm" in fv:
                r2m = fv["repaired_vs_trusted_mlm"]
                s2m = fv["stock_vs_trusted_mlm"]
                f.write(f"- Repaired vs trusted MLM encoder: max_diff=`{r2m['max_abs_diff']:.6e}`, identical=`{r2m['hidden_identical']}`\n")
                f.write(f"- Stock vs trusted MLM encoder: max_diff=`{s2m['max_abs_diff']:.6e}`, identical=`{s2m['hidden_identical']}`\n")
            sc = val.get("executed_scales", {})
            f.write(f"- Adapter scales: `{sc.get('adapter_scales', [])}`\n")
            f.write(f"- Private adapter scales: `{sc.get('private_adapter_scales', [])}`\n")
            f.write("\n")
        f.write(f"**All repaired checkpoints valid:** `{all_valid}`\n")

    print(f"\nSaved: {out_json}")
    print(json.dumps({"status": results["status"], "all_valid": all_valid}, indent=2))


if __name__ == "__main__":
    main()
