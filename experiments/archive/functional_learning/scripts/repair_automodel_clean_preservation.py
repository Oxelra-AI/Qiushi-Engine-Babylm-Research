#!/usr/bin/env python3
"""research: repair and validate AutoModel loading for clean eval-mode preservation checkpoints.

The clean eval-mode preservation endpoint was saved as an MLM checkpoint whose
config only registers AutoModelForMaskedLM.  The BabyLM SuperGLUE path calls
AutoModel.from_pretrained; without an AutoModel registration it silently loads the
stock DeBERTa encoder and drops both slow/private adapters.  This script creates a
local repaired copy with FrozenSlowPrivateDebertaV2Model registered under
AutoModel and validates the real AutoModel interface against the trusted MLM
backbone on a fixed batch.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Any, Dict, Tuple

import torch
from safetensors.torch import load_file

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import repair_automodel as repair078  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/automodel_repair_clean_preservation')
DEFAULT_CLEAN_EVAL = _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_full80/checkpoints/update_0080')
DEFAULT_CLEAN_TRAIN = _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_train_full80/checkpoints/update_0080')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_cache(out_dir: pathlib.Path) -> Dict[str, str]:
    hf = out_dir / "hf_cache"
    env = {
        "HF_HOME": hf / "hf_home",
        "HF_HUB_CACHE": hf / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": hf / "hf_home" / "hub",
        "HF_DATASETS_CACHE": hf / "datasets",
        "TRANSFORMERS_CACHE": hf / "transformers",
        "HF_MODULES_CACHE": hf / "modules",
        "TMPDIR": hf / "tmp",
    }
    for k, p in env.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import transformers.utils.hub as hub
        import transformers.dynamic_module_utils as dyn
        hub.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]
        dyn.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]
        if hasattr(hub, "TRANSFORMERS_CACHE"):
            hub.TRANSFORMERS_CACHE = os.environ["TRANSFORMERS_CACHE"]
        if hasattr(dyn, "TRANSFORMERS_CACHE"):
            dyn.TRANSFORMERS_CACHE = os.environ["TRANSFORMERS_CACHE"]
    except Exception:
        pass
    return {k: str(p.resolve()) for k, p in env.items()}


def reset_dir(path: pathlib.Path, force: bool) -> None:
    if path.exists() or path.is_symlink():
        if not force:
            return
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)


def create_repaired(source: pathlib.Path, dest: pathlib.Path, force: bool) -> Dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    if not (source / "config.json").is_file() or not (source / "frozen82_private_modeling.py").is_file():
        raise FileNotFoundError(f"checkpoint lacks config/modeling file: {source}")
    reset_dir(dest, force)
    if not dest.exists():
        repair078.create_repaired_checkpoint(source, dest)
    cfg = json.loads((dest / "config.json").read_text(encoding="utf-8"))
    return {
        "source": rel(source),
        "repaired": rel(dest),
        "source_sha256": sha256_file(source / "model.safetensors"),
        "repaired_model_is_symlink": (dest / "model.safetensors").is_symlink(),
        "config_auto_map": cfg.get("auto_map"),
        "config_architectures": cfg.get("architectures"),
    }


def load_trusted_mlm(model_dir: pathlib.Path):
    modeling_file = model_dir / "frozen82_private_modeling.py"
    spec = importlib.util.spec_from_file_location(f"trusted_{abs(hash(str(model_dir))) % 10**12}", modeling_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {modeling_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    from transformers import DebertaV2Config
    cfg = DebertaV2Config.from_pretrained(str(model_dir), local_files_only=True)
    model = mod.FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
    state = load_file(str(model_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, list(missing), list(unexpected)


def inspect_model(model) -> Dict[str, Any]:
    named = list(model.named_parameters())
    adapter_params = int(sum(p.numel() for n, p in named if ".adapter." in n and ".private_adapter." not in n))
    private_params = int(sum(p.numel() for n, p in named if ".private_adapter." in n))
    adapter_scales = []
    private_scales = []
    enc = getattr(model, "encoder", None)
    if enc is None and hasattr(model, "deberta"):
        enc = getattr(model.deberta, "encoder", None)
    try:
        for layer in enc.layer:
            if hasattr(layer, "adapter") and hasattr(layer.adapter, "scale"):
                adapter_scales.append(float(layer.adapter.scale))
            if hasattr(layer, "private_adapter") and hasattr(layer.private_adapter, "scale"):
                private_scales.append(float(layer.private_adapter.scale))
    except Exception:
        pass
    return {
        "class": type(model).__name__,
        "module": type(model).__module__,
        "total_params": int(sum(p.numel() for _, p in named)),
        "adapter_params": adapter_params,
        "private_adapter_params": private_params,
        "adapter_scales": adapter_scales,
        "private_adapter_scales": private_scales,
    }


def hidden(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        if hasattr(model, "deberta"):
            out = model.deberta(**batch)
        else:
            out = model(**batch)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def validate_pair(source: pathlib.Path, repaired: pathlib.Path, out_dir: pathlib.Path, label: str) -> Dict[str, Any]:
    setup_cache(out_dir)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(repaired), trust_remote_code=True, local_files_only=True)
    batch = tok([
        "Children learn language from limited experience.",
        "The source sentence supplies evidence for the masked answer.",
        "Preservation should protect ordinary predictions while allowing useful adaptation.",
    ], padding=True, truncation=True, max_length=80, return_tensors="pt")

    trusted, missing, unexpected = load_trusted_mlm(source)
    auto = AutoModel.from_pretrained(str(repaired), trust_remote_code=True, local_files_only=True)
    auto.eval()
    with torch.no_grad():
        ht = hidden(trusted, batch)
        ha = hidden(auto, batch)
    diff = (ht - ha).abs()

    neg: Dict[str, Any] = {}
    try:
        stock = AutoModel.from_pretrained(str(source), trust_remote_code=True, local_files_only=True)
        stock.eval()
        with torch.no_grad():
            hs = hidden(stock, batch)
        sdiff = (ht - hs).abs()
        neg = {
            "loaded": True,
            "identity": inspect_model(stock),
            "is_stock_or_adapterless": "FrozenSlowPrivate" not in type(stock).__name__ or inspect_model(stock)["private_adapter_params"] == 0,
            "hidden_max_abs_diff_vs_trusted": float(sdiff.max()),
            "hidden_mean_abs_diff_vs_trusted": float(sdiff.mean()),
        }
        del stock
    except Exception as exc:
        neg = {"loaded": False, "error": repr(exc)}

    trusted_info = inspect_model(trusted)
    auto_info = inspect_model(auto)
    res = {
        "label": label,
        "source": rel(source),
        "repaired": rel(repaired),
        "trusted_mlm": {**trusted_info, "missing_keys": missing, "unexpected_keys": unexpected[:10]},
        "automodel": auto_info,
        "negative_original_automodel": neg,
        "hidden_state_comparison": {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "exact_match": bool(float(diff.max()) == 0.0),
            "shape": list(ht.shape),
        },
    }
    res["valid"] = bool(
        "FrozenSlowPrivate" in auto_info["class"]
        and auto_info["adapter_params"] == trusted_info["adapter_params"]
        and auto_info["private_adapter_params"] == trusted_info["private_adapter_params"]
        and auto_info["private_adapter_params"] == 995584
        and auto_info["private_adapter_scales"] == trusted_info["private_adapter_scales"]
        and res["hidden_state_comparison"]["exact_match"]
    )
    del trusted, auto
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--include-train-mode", action="store_true", help="also repair the weaker train-mode comparison endpoint")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = [("clean_pres_lambda1_eval_seed62064_u0080", DEFAULT_CLEAN_EVAL)]
    if args.include_train_mode:
        targets.append(("clean_pres_lambda1_train_seed62064_u0080", DEFAULT_CLEAN_TRAIN))

    repairs = []
    validations = []
    for label, src in targets:
        dst = args.out_dir / f"repaired_{label}"
        print(json.dumps({"event": "repair_start", "label": label, "source": rel(src), "dest": rel(dst)}, ensure_ascii=False), flush=True)
        repairs.append({"label": label, **create_repaired(src, dst, bool(args.force))})
        print(json.dumps({"event": "validate_start", "label": label, "repaired": rel(dst)}, ensure_ascii=False), flush=True)
        validations.append(validate_pair(src, dst, args.out_dir / f"validation_{label}", label))
        print(json.dumps({"event": "validate_done", "label": label, "valid": validations[-1].get("valid"), "class": validations[-1].get("automodel", {}).get("class")}, ensure_ascii=False), flush=True)

    result = {
        "status": "AUTOMODEL_REPAIR_CLEAN_PRESERVATION",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/repair_automodel_clean_preservation.py')),
        "repairs": repairs,
        "validations": validations,
        "all_valid": all(v.get("valid") for v in validations),
        "scientific_use": "Repaired copies are the only valid paths for AutoModel-based SuperGLUE; original clean checkpoints remain valid for MLM/AoA through AutoModelForMaskedLM.",
    }
    out_json = args.out_dir / "repair_validation.json"
    out_md = args.out_dir / "repair_validation.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = ["# research clean-preservation AutoModel repair\n\n", f"Status: `{result['status']}`; all valid `{result['all_valid']}`.\n\n"]
    for val in validations:
        lines.append(f"## {val['label']}\n")
        lines.append(f"- repaired: `{val['repaired']}`\n")
        lines.append(f"- AutoModel class: `{val['automodel'].get('class')}` params `{val['automodel'].get('total_params')}` private `{val['automodel'].get('private_adapter_params')}` scales `{val['automodel'].get('private_adapter_scales')}`\n")
        lines.append(f"- hidden max diff vs trusted MLM: `{val['hidden_state_comparison'].get('max_abs_diff')}` exact `{val['hidden_state_comparison'].get('exact_match')}`\n")
        neg = val.get('negative_original_automodel', {})
        lines.append(f"- original AutoModel negative: loaded `{neg.get('loaded')}`, identity `{neg.get('identity')}`, adapterless/stock `{neg.get('is_stock_or_adapterless')}`\n\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "all_valid": result["all_valid"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)
    if not result["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
