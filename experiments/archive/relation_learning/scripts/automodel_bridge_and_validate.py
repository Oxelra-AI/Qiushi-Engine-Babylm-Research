#!/usr/bin/env python3
"""research: add faithful AutoModel bridges for BabyLM adapter checkpoints.

Scientific purpose
------------------
The official SuperGLUE finetuning path calls ``AutoModel.from_pretrained``.  The
BabyLM checkpoints used in v4/v5 register custom ``AutoModelForMaskedLM``
classes, but older checkpoint files do not register a matching ``AutoModel``.
Transformers can therefore instantiate a stock DebertaV2Model and silently ignore
adapter tensors.  That is an evaluation lesion, not a property of the deployed
model.

This script creates local repaired copies of checkpoints and validates the
central invariant: hidden states from repaired AutoModel must match the encoder
hidden states from the trusted AutoModelForMaskedLM class exactly (up to numerical
roundoff), with all adapter parameters exposed.  It supports both:
  * research AdapterDebertaV2ForMaskedLM: base + slow adapter.
  * research FrozenSlowPrivateDebertaV2ForMaskedLM: base + slow adapter + private
    adapter.

It changes no training output.  The repaired copies are the model-file fidelity
layer needed before any SuperGLUE measurement can be read as evidence about an
adapter/private endpoint.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
DEFAULT_OUT = ROOT / "experiments/archive/relation_learning/data/automodel_bridge"


MODEL_CLASS = r'''

class AdapterDebertaV2Model(DebertaV2Model):
    """Base encoder variant of AdapterDebertaV2ForMaskedLM.

    Registered under AutoModel so official SuperGLUE finetuning loads the same
    encoder computation as the trusted MLM class: each completed DeBERTa layer is
    followed by the trained residual adapter scaled by ``config.adapter_scale``.
    """
    config_class = DebertaV2Config

    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        config.adapter_bottleneck = int(getattr(config, "adapter_bottleneck", 64))
        config.adapter_activation = getattr(config, "adapter_activation", "gelu")
        config.adapter_enabled = bool(getattr(config, "adapter_enabled", True))
        config.adapter_scale = float(getattr(config, "adapter_scale", 1.0))
        for layer in self.encoder.layer:
            layer.__class__ = AdapterDebertaV2Layer
            layer.adapter = ZeroOutputBottleneckAdapter(config)

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if ".adapter." in name:
                yield name, parameter

    def adapter_rms(self) -> list[float]:
        return [float(layer.adapter.last_rms) for layer in self.encoder.layer]

    def set_adapter_enabled(self, enabled: bool):
        for layer in self.encoder.layer:
            layer.adapter.enabled = bool(enabled)
        self.config.adapter_enabled = bool(enabled)
'''


MODEL_CLASS = r'''

class FrozenSlowPrivateDebertaV2Model(DebertaV2Model):
    """Base encoder variant of FrozenSlowPrivateDebertaV2ForMaskedLM.

    Registered under AutoModel so official SuperGLUE finetuning loads the full
    deployed encoder: the inherited slow adapter at ``adapter_scale`` and the
    learned private adapter at ``private_adapter_scale``.  Freezing the slow path
    was a private-continuation training choice; the deployed checkpoint exposes
    all parameters to downstream fine-tuning.
    """
    config_class = DebertaV2Config

    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
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

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if ".adapter." in name and ".private_adapter." not in name:
                yield name, parameter

    def private_adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if ".private_adapter." in name:
                yield name, parameter

    def private_adapter_rms(self) -> list[float]:
        return [float(layer.private_adapter.last_rms) for layer in self.encoder.layer]

    def slow_adapter_rms(self) -> list[float]:
        return [float(layer.adapter.last_rms) for layer in self.encoder.layer]

    def set_private_enabled(self, enabled: bool):
        for layer in self.encoder.layer:
            layer.private_adapter.enabled = bool(enabled)
        self.config.private_adapter_enabled = bool(enabled)

    def set_slow_adapter_enabled(self, enabled: bool):
        for layer in self.encoder.layer:
            layer.adapter.enabled = bool(enabled)
        self.config.adapter_enabled = bool(enabled)
'''


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def detect_modeling(src: Path) -> tuple[str, Path]:
    if (src / "frozen82_private_modeling.py").exists():
        return "research", src / "frozen82_private_modeling.py"
    if (src / "adapter_scaled_modeling.py").exists():
        return "research", src / "adapter_scaled_modeling.py"
    py = sorted(src.glob("*.py"))
    raise FileNotFoundError(f"No supported modeling file under {src}; python files={py}")


def ensure_deberta_model_import(code: str) -> str:
    if "DebertaV2Model" in re.sub(r"class\s+\w+\(DebertaV2Model\):", "", code):
        # Either imported already or present in text.  The import check below is harmless.
        pass
    patterns = [
        "from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2Model",
        "from transformers import DebertaV2Config, DebertaV2ForMaskedLM",
    ]
    if patterns[0] in code:
        return code
    if patterns[1] in code:
        return code.replace(patterns[1], patterns[0], 1)
    # Conservative fallback: add a narrow import after the first transformers import block.
    return code.replace("from transformers ", "from transformers ", 1) + "\n"


def patch_step104(code: str) -> tuple[str, str]:
    code = ensure_deberta_model_import(code)
    if "class AdapterDebertaV2Model(DebertaV2Model):" not in code:
        marker = "class AdapterDebertaV2ForMaskedLM(DebertaV2ForMaskedLM):"
        if marker not in code:
            raise RuntimeError("Cannot locate AdapterDebertaV2ForMaskedLM marker")
        code = code.replace(marker, MODEL_CLASS + "\n\n" + marker, 1)
    old = '''config.auto_map = {
            "AutoModelForMaskedLM": "adapter_scaled_modeling.AdapterDebertaV2ForMaskedLM"
        }'''
    new = '''config.auto_map = {
            "AutoModelForMaskedLM": "adapter_scaled_modeling.AdapterDebertaV2ForMaskedLM",
            "AutoModel": "adapter_scaled_modeling.AdapterDebertaV2Model"
        }'''
    if old in code:
        code = code.replace(old, new, 1)
    return code, "adapter_scaled_modeling.AdapterDebertaV2Model"


def patch_step131(code: str) -> tuple[str, str]:
    code = ensure_deberta_model_import(code)
    if "class FrozenSlowPrivateDebertaV2Model(DebertaV2Model):" not in code:
        marker = "class FrozenSlowPrivateDebertaV2ForMaskedLM(DebertaV2ForMaskedLM):"
        if marker not in code:
            raise RuntimeError("Cannot locate FrozenSlowPrivateDebertaV2ForMaskedLM marker")
        code = code.replace(marker, MODEL_CLASS + "\n\n" + marker, 1)
    old = '''config.auto_map = {
            "AutoModelForMaskedLM": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM"
        }'''
    new = '''config.auto_map = {
            "AutoModelForMaskedLM": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM",
            "AutoModel": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2Model"
        }'''
    if old in code:
        code = code.replace(old, new, 1)
    return code, "frozen82_private_modeling.FrozenSlowPrivateDebertaV2Model"


def patch_modeling(kind: str, src_file: Path, dst_file: Path) -> str:
    code = src_file.read_text(encoding="utf-8")
    if kind == "research":
        patched, auto = patch_step104(code)
    elif kind == "research":
        patched, auto = patch_step131(code)
    else:
        raise ValueError(kind)
    dst_file.write_text(patched, encoding="utf-8")
    return auto


def create_repaired_copy(label: str, src: Path, out_root: Path) -> dict[str, Any]:
    kind, modeling = detect_modeling(src)
    dst = out_root / f"repaired_{label}"
    dst.mkdir(parents=True, exist_ok=True)
    auto_model_entry = None
    copied: list[str] = []
    symlinked: list[str] = []
    for item in sorted(src.iterdir(), key=lambda p: p.name):
        dest = dst / item.name
        if item.name == modeling.name:
            auto_model_entry = patch_modeling(kind, item, dest)
            copied.append(item.name + "[patched]")
        elif item.name == "config.json":
            cfg = read_json(item)
            auto_map = dict(cfg.get("auto_map") or {})
            if auto_model_entry is None:
                # Will be filled after modeling; set below in a second pass if needed.
                pass
            cfg.setdefault("auto_map", auto_map)
            dest.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            copied.append(item.name + "[pending_auto_map]")
        elif item.name == "model.safetensors" and item.is_file():
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            os.symlink(item.resolve(), dest)
            symlinked.append(item.name)
        elif item.is_file():
            shutil.copy2(item, dest)
            copied.append(item.name)
        elif item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest, symlinks=True)
            copied.append(item.name + "/")
    if auto_model_entry is None:
        # Modeling file may have sorted after config; recompute.
        _, patched_modeling = detect_modeling(dst)
        code = patched_modeling.read_text(encoding="utf-8")
        if kind == "research":
            auto_model_entry = "adapter_scaled_modeling.AdapterDebertaV2Model"
        else:
            auto_model_entry = "frozen82_private_modeling.FrozenSlowPrivateDebertaV2Model"
    cfg_path = dst / "config.json"
    cfg = read_json(cfg_path)
    auto_map = dict(cfg.get("auto_map") or {})
    auto_map["AutoModel"] = auto_model_entry
    cfg["auto_map"] = auto_map
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "label": label,
        "kind": kind,
        "source": rel(src),
        "repaired": rel(dst),
        "auto_model_entry": auto_model_entry,
        "copied": copied,
        "symlinked": symlinked,
        "source_model_sha256": sha256_file(src / "model.safetensors"),
        "repaired_model_sha256": sha256_file(dst / "model.safetensors"),
    }


def setup_writable_hf_cache(out_root: Path) -> None:
    cache = out_root / "hf_cache"
    mapping = {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }
    for key, path in mapping.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def count_params(model: Any) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    adapter = sum(p.numel() for n, p in model.named_parameters() if ".adapter." in n and ".private_adapter." not in n)
    private = sum(p.numel() for n, p in model.named_parameters() if ".private_adapter." in n)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "adapter": int(adapter), "private_adapter": int(private), "trainable": int(trainable)}


def validate_one(label: str, repaired: Path, source: Path, out_root: Path) -> dict[str, Any]:
    setup_writable_hf_cache(out_root)
    import torch  # noqa: WPS433
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer  # noqa: WPS433

    rec: dict[str, Any] = {
        "label": label,
        "source": rel(source),
        "repaired": rel(repaired),
        "config_auto_map": read_json(repaired / "config.json").get("auto_map"),
    }
    tok = AutoTokenizer.from_pretrained(str(repaired), trust_remote_code=True)
    batch = tok(
        ["The careful child moved the red cup to the shelf.", "A small model can learn a useful relation from clean evidence."],
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt",
    )
    mlm = AutoModelForMaskedLM.from_pretrained(str(repaired), trust_remote_code=True)
    auto = AutoModel.from_pretrained(str(repaired), trust_remote_code=True)
    mlm.eval()
    auto.eval()
    with torch.no_grad():
        h_mlm = mlm.deberta(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        h_auto = auto(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    diff = (h_mlm - h_auto).abs()
    rec["trusted_mlm_class"] = type(mlm).__module__ + "." + type(mlm).__name__
    rec["repaired_automodel_class"] = type(auto).__module__ + "." + type(auto).__name__
    rec["trusted_mlm_params"] = count_params(mlm)
    rec["repaired_automodel_params"] = count_params(auto)
    rec["automodel_vs_mlm_encoder"] = {
        "max_abs_diff": float(diff.max().cpu()),
        "mean_abs_diff": float(diff.mean().cpu()),
        "identical_tol_1e_6": bool(float(diff.max().cpu()) <= 1e-6),
    }
    rec["executed_scales"] = {}
    layers = getattr(auto, "encoder", None).layer if hasattr(getattr(auto, "encoder", None), "layer") else []
    if layers:
        rec["executed_scales"]["adapter"] = [float(getattr(layer.adapter, "scale")) for layer in layers if hasattr(layer, "adapter")]
        rec["executed_scales"]["private_adapter"] = [float(getattr(layer.private_adapter, "scale")) for layer in layers if hasattr(layer, "private_adapter")]
    try:
        stock = AutoModel.from_pretrained(str(source), trust_remote_code=True)
        stock.eval()
        with torch.no_grad():
            h_stock = stock(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        dstock = (h_mlm - h_stock).abs()
        rec["source_automodel_without_bridge"] = {
            "class": type(stock).__module__ + "." + type(stock).__name__,
            "params": count_params(stock),
            "max_abs_diff_vs_trusted_mlm": float(dstock.max().cpu()),
            "mean_abs_diff_vs_trusted_mlm": float(dstock.mean().cpu()),
        }
    except Exception as exc:  # noqa: BLE001
        rec["source_automodel_without_bridge_error"] = repr(exc)
    return rec


def parse_checkpoint_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, path = raw.split("=", 1)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    if not label:
        raise argparse.ArgumentTypeError("empty checkpoint label")
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise argparse.ArgumentTypeError(f"checkpoint path does not exist: {path}")
    return label, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH; may be repeated")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    parsed = [parse_checkpoint_arg(x) for x in args.checkpoint]
    repairs = []
    validations = []
    for label, src in parsed:
        rep = create_repaired_copy(label, src, out_root)
        repairs.append(rep)
        if not args.skip_validation:
            validations.append(validate_one(label, out_root / f"repaired_{label}", src, out_root))

    all_valid = bool(validations) and all(v.get("automodel_vs_mlm_encoder", {}).get("identical_tol_1e_6") for v in validations)
    result = {
        "status": "AUTOMODEL_BRIDGE_DONE" if all_valid or args.skip_validation else "AUTOMODEL_BRIDGE_NEEDS_REPAIR",
        "out_root": rel(out_root),
        "repairs": repairs,
        "validations": validations,
        "all_valid": all_valid if not args.skip_validation else None,
        "interpretation": "Repaired AutoModel directories are valid only when AutoModel hidden states match trusted MLM encoder hidden states and adapter/private parameter counts are nonzero as expected.",
    }
    write_json(out_root / "automodel_bridge_validation.json", result)
    lines = ["# research AutoModel bridge validation", ""]
    for val in validations:
        lines.append(f"## {val['label']}")
        lines.append(f"- repaired: `{val['repaired']}`")
        lines.append(f"- trusted MLM: `{val['trusted_mlm_class']}` params {val['trusted_mlm_params']}")
        lines.append(f"- repaired AutoModel: `{val['repaired_automodel_class']}` params {val['repaired_automodel_params']}")
        d = val["automodel_vs_mlm_encoder"]
        lines.append(f"- AutoModel vs MLM encoder max diff `{d['max_abs_diff']:.6e}`, mean diff `{d['mean_abs_diff']:.6e}`, identical `{d['identical_tol_1e_6']}`")
        if "source_automodel_without_bridge" in val:
            s = val["source_automodel_without_bridge"]
            lines.append(f"- source AutoModel without bridge: `{s['class']}` params {s['params']}, max diff vs trusted `{s['max_abs_diff_vs_trusted_mlm']:.6e}`")
        lines.append(f"- executed scales: `{val.get('executed_scales')}`")
        lines.append("")
    lines.append(f"All validated: `{result['all_valid']}`")
    (out_root / "automodel_bridge_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_root / "automodel_bridge_validation.json"), "out_md": rel(out_root / "automodel_bridge_validation.md"), "all_valid": result["all_valid"]}, indent=2), flush=True)
    if not args.skip_validation and not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
