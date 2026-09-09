#!/usr/bin/env python3
"""research: audit coherent86 model identity and invalid research assumptions.

This script checks whether generic AutoModelForMaskedLM loading executes the
inherited coherent86 private-adapter architecture, and records the trusted loader
identity, executed private scales, and trainable-set requirements.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import gc
import json
import pathlib
import sys
from typing import Any, Dict

import torch
from safetensors.torch import load_file
from transformers import AutoModelForMaskedLM, DebertaV2Config

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
MODEL_PATH = _public_path('models/frontier')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/model_identity_audit')
OUT_DIR.mkdir(parents=True, exist_ok=True)

A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
A01_SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(A02_SCRIPTS))
sys.path.insert(0, str(A01_SCRIPTS))


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def count_params(model) -> Dict[str, Any]:
    named = list(model.named_parameters())
    private = [(n, p) for n, p in named if ".private_adapter." in n or "private_adapter" in n]
    trainable = [(n, p) for n, p in named if p.requires_grad]
    return {
        "class": type(model).__name__,
        "module": type(model).__module__,
        "total_tensors": len(named),
        "total_params": int(sum(p.numel() for _, p in named)),
        "private_tensor_count": len(private),
        "private_param_count": int(sum(p.numel() for _, p in private)),
        "trainable_tensor_count_default": len(trainable),
        "trainable_param_count_default": int(sum(p.numel() for _, p in trainable)),
        "private_name_examples": [n for n, _ in private[:4]] + [n for n, _ in private[-4:]],
        "has_set_private_enabled": bool(hasattr(model, "set_private_enabled")),
        "has_private_adapter_rms": bool(hasattr(model, "private_adapter_rms")),
    }


def executed_scales(model) -> list[float]:
    vals = []
    try:
        for layer in model.deberta.encoder.layer:
            if hasattr(layer, "private_adapter") and hasattr(layer.private_adapter, "scale"):
                vals.append(float(layer.private_adapter.scale))
    except Exception:
        pass
    return vals


def load_generic(trust_remote_code: bool) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    try:
        obj = AutoModelForMaskedLM.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            output_loading_info=True,
        )
        if isinstance(obj, tuple):
            model, loading_info = obj
        else:
            model, loading_info = obj, {}
        rec.update(count_params(model))
        rec["executed_private_scales"] = executed_scales(model)
        rec["loading_info"] = {
            "missing_keys_count": len(loading_info.get("missing_keys", [])),
            "unexpected_keys_count": len(loading_info.get("unexpected_keys", [])),
            "mismatched_keys_count": len(loading_info.get("mismatched_keys", [])),
            "error_msgs_count": len(loading_info.get("error_msgs", [])),
            "missing_private_count": sum(1 for k in loading_info.get("missing_keys", []) if "private_adapter" in k),
            "unexpected_private_count": sum(1 for k in loading_info.get("unexpected_keys", []) if "private_adapter" in k),
            "missing_first20": loading_info.get("missing_keys", [])[:20],
            "unexpected_first20": loading_info.get("unexpected_keys", [])[:20],
        }
        del model
        gc.collect()
    except Exception as e:
        rec["error"] = repr(e)
    return rec


def load_trusted() -> Dict[str, Any]:
    rec: Dict[str, Any] = {}
    try:
        import coherent86_continuation_trainer as trusted
        device = torch.device("cpu")
        model, missing, unexpected = trusted.load_model(MODEL_PATH, device, 128, 0.75)
        rec.update(count_params(model))
        rec["executed_private_scales"] = executed_scales(model)
        rec["all_scales_0p75"] = bool(rec["executed_private_scales"] and all(abs(x - 0.75) < 1e-12 for x in rec["executed_private_scales"]))
        rec["missing_count"] = len(missing)
        rec["unexpected_count"] = len(unexpected)
        rec["missing_private_count"] = sum(1 for k in missing if "private_adapter" in k)
        rec["unexpected_private_count"] = sum(1 for k in unexpected if "private_adapter" in k)
        rec["missing_first20"] = list(missing)[:20]
        rec["unexpected_first20"] = list(unexpected)[:20]
        # The trusted trainer then freezes every non-private parameter before optimization.
        private_names = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
        for n, p in model.named_parameters():
            p.requires_grad_(n in private_names)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        rec["after_private_freeze_trainable_tensor_count"] = len(trainable)
        rec["after_private_freeze_trainable_param_count"] = int(sum(p.numel() for _, p in trainable))
        rec["after_private_freeze_nonprivate_trainable_count"] = sum(1 for n, _ in trainable if ".private_adapter." not in n)
        del model
        gc.collect()
    except Exception as e:
        rec["error"] = repr(e)
    return rec


def script_issue_scan() -> Dict[str, Any]:
    targets = [
        _public_path('experiments/archive/functional_learning/scripts/revision_039b_robust_scorer.py'),
        _public_path('experiments/archive/functional_learning/scripts/revision_039d_answer_only_training.py'),
        _public_path('experiments/archive/functional_learning/scripts/relation_first_constructor.py'),
    ]
    out = {}
    for p in targets:
        text = p.read_text(encoding="utf-8")
        out[rel(p)] = {
            "uses_generic_auto_model": "AutoModelForMaskedLM.from_pretrained" in text,
            "uses_trust_remote_code": "trust_remote_code" in text,
            "uses_model_parameters_optimizer": "AdamW(model.parameters()" in text or "AdamW(model.parameters" in text,
            "has_token_search_fallback_drops_first_token": "answer_ids[1:]" in text or "target_ids[1:]" in text,
            "uses_individual_position_scoring_loop": "for pos in positions" in text and "masked[0, pos]" in text,
            "constructs_distinct_new_values": "new_value_a" in text and "new_value_b" in text and "new_b" in text,
        }
    return out


def main() -> None:
    cfg = json.loads((_public_path('models/frontier/config.json')).read_text(encoding="utf-8"))
    sd = load_file(str(_public_path('models/frontier/model.safetensors')), device="cpu")
    private_keys = sorted(k for k in sd if "private_adapter" in k)
    result = {
        "status": "MODEL_IDENTITY_AUDIT",
        "model_path": rel(MODEL_PATH),
        "config_architectures": cfg.get("architectures"),
        "config_auto_map": cfg.get("auto_map"),
        "config_private_adapter_scale": cfg.get("private_adapter_scale"),
        "config_adapter_scale": cfg.get("adapter_scale"),
        "state_dict_tensor_count": len(sd),
        "state_dict_private_key_count": len(private_keys),
        "state_dict_private_key_examples": private_keys[:6] + private_keys[-6:],
        "generic_auto_no_trust": load_generic(False),
        "generic_auto_trust_remote_code": load_generic(True),
        "trusted_loader": load_trusted(),
        "script_issues": script_issue_scan(),
    }
    (_public_path('experiments/archive/functional_learning/data/model_identity_audit/model_identity_audit.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# research coherent86 model identity audit\n\n"]
    lines.append(f"Model endpoint: `{result['model_path']}`\n\n")
    lines.append(f"Config architectures: `{result['config_architectures']}`; auto_map: `{result['config_auto_map']}`\n\n")
    lines.append(f"State dict private-adapter tensors: {result['state_dict_private_key_count']}\n\n")
    for label in ["generic_auto_no_trust", "generic_auto_trust_remote_code", "trusted_loader"]:
        rec = result[label]
        lines.append(f"## {label}\n\n")
        if "error" in rec:
            lines.append(f"ERROR: `{rec['error']}`\n\n")
            continue
        lines.append(f"- class: `{rec.get('module')}.{rec.get('class')}`\n")
        lines.append(f"- total params: {rec.get('total_params')}; private params: {rec.get('private_param_count')} in {rec.get('private_tensor_count')} tensors\n")
        lines.append(f"- default trainable params: {rec.get('trainable_param_count_default')} in {rec.get('trainable_tensor_count_default')} tensors\n")
        lines.append(f"- executed private scales: `{rec.get('executed_private_scales')}`\n")
        if 'loading_info' in rec:
            li = rec['loading_info']
            lines.append(f"- loading missing/unexpected/private-unexpected: {li.get('missing_keys_count')}/{li.get('unexpected_keys_count')}/{li.get('unexpected_private_count')}\n")
        if label == "trusted_loader":
            lines.append(f"- after private freeze trainable params: {rec.get('after_private_freeze_trainable_param_count')} ; non-private trainable tensors: {rec.get('after_private_freeze_nonprivate_trainable_count')}\n")
        lines.append("\n")
    lines.append("## Consequence for research\n\n")
    lines.append("Step039b/Step039d used generic AutoModelForMaskedLM and Step039d optimized model.parameters(). Any completed or partial Step039d training would not be interpretable as private-adapter coherent86 acquisition unless the loaded class and trainable set are repaired.\n\n")
    lines.append("Script scan:\n\n")
    for k, v in result["script_issues"].items():
        lines.append(f"- `{k}`: {v}\n")
    (_public_path('research/documents/functional_learning/data/model_identity_audit/model_identity_audit.md')).write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_dir": rel(OUT_DIR),
        "no_trust_class": result["generic_auto_no_trust"].get("class") or result["generic_auto_no_trust"].get("error"),
        "no_trust_private_params": result["generic_auto_no_trust"].get("private_param_count"),
        "trusted_class": result["trusted_loader"].get("class") or result["trusted_loader"].get("error"),
        "trusted_private_params": result["trusted_loader"].get("private_param_count"),
        "trusted_scales": result["trusted_loader"].get("executed_private_scales"),
        "trusted_frozen_trainable_params": result["trusted_loader"].get("after_private_freeze_trainable_param_count"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
