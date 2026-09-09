#!/usr/bin/env python3
"""research: audit whether dense SuperGLUE finetuning can see private adapters.

The BabyLM sentence-zero-shot and Reading evaluators use AutoModelForMaskedLM with
trust_remote_code, which is the auto class registered by the frozen-slow/private
checkpoints. The official SuperGLUE classifier wrapper uses AutoModel. If the
checkpoint config does not register AutoModel, transformers may instantiate the
stock DebertaV2Model and silently drop private-adapter tensors. This script tests
that failure mode directly on dense checkpoints and reference checkpoints.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
import pathlib
import time
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

ROOT = _public_path('.')
OUT = _public_path('experiments/archive/relation_learning/data/superglue_loader_identity_audit')
MODELS = {
    "chck82": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M'),
    "coherent86_train_scale1p0": _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final'),
    "dense62064_u0080": _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/checkpoints/update_0080'),
    "dense62065_u0080": _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/checkpoints/update_0080'),
}
TEXT = "The child put the marble in the red box and later looked for the marble."


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def set_private_enabled(model, enabled: bool) -> None:
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(bool(enabled))
    elif hasattr(model, "config"):
        model.config.private_adapter_enabled = bool(enabled)


def count_private_keys(model_path: pathlib.Path) -> dict[str, Any]:
    sd = load_file(str(model_path / "model.safetensors"), device="cpu")
    keys = [k for k in sd if "private_adapter" in k]
    return {"private_state_keys": len(keys), "private_state_params": int(sum(sd[k].numel() for k in keys)), "examples": keys[:5]}


def try_load_auto(model_path: pathlib.Path) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    try:
        model, info = AutoModel.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True, output_loading_info=True)
        rec.update({
            "ok": True,
            "class": type(model).__name__,
            "module": type(model).__module__,
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "private_named_params": int(sum(p.numel() for n, p in model.named_parameters() if "private_adapter" in n)),
            "loading_info_missing_count": len(info.get("missing_keys") or []),
            "loading_info_unexpected_count": len(info.get("unexpected_keys") or []),
            "missing_examples": (info.get("missing_keys") or [])[:12],
            "unexpected_examples": (info.get("unexpected_keys") or [])[:12],
        })
        model.eval()
        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        enc = tok(TEXT, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            out = model(**enc)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        rec["probe_hidden_mean"] = float(h.mean())
        rec["probe_hidden_std"] = float(h.std())
        rec["probe_hidden_shape"] = list(h.shape)
        del model
    except Exception as e:
        rec.update({"ok": False, "error": repr(e)})
    return rec


def try_load_mlm(model_path: pathlib.Path) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    try:
        model, info = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True, output_loading_info=True)
        rec.update({
            "ok": True,
            "class": type(model).__name__,
            "module": type(model).__module__,
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "private_named_params": int(sum(p.numel() for n, p in model.named_parameters() if "private_adapter" in n)),
            "loading_info_missing_count": len(info.get("missing_keys") or []),
            "loading_info_unexpected_count": len(info.get("unexpected_keys") or []),
            "missing_examples": (info.get("missing_keys") or [])[:12],
            "unexpected_examples": (info.get("unexpected_keys") or [])[:12],
        })
        model.eval()
        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        enc = tok(TEXT, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            set_private_enabled(model, False)
            logits_off = model(**enc).logits
            set_private_enabled(model, True)
            logits_on = model(**enc).logits
        rec["probe_private_on_off_logit_maxdiff"] = float((logits_on - logits_off).abs().max())
        rec["probe_logit_mean_on"] = float(logits_on.mean())
        rec["probe_logit_std_on"] = float(logits_on.std())
        del model
    except Exception as e:
        rec.update({"ok": False, "error": repr(e)})
    return rec


def hidden_tensor_auto(model_path: pathlib.Path):
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    enc = tok(TEXT, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        out = model(**enc)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return h.detach().cpu()


def logits_tensor_mlm(model_path: pathlib.Path, private_on: bool):
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model.eval()
    set_private_enabled(model, private_on)
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    enc = tok(TEXT, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        logits = model(**enc).logits
    return logits.detach().cpu()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = _public_path('experiments/archive/relation_learning/data/superglue_loader_identity_audit/hf_cache')
    for name, p in {
        "HF_HOME": cache / "hf_home",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(p)

    result: dict[str, Any] = {
        "status": "A01_SUPERGLUE_LOADER_IDENTITY_AUDIT_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Does AutoModel-based SuperGLUE finetuning load the custom private-adapter checkpoints, or does it fall back to the frozen DeBERTa encoder and ignore private tensors?",
        "probe_text": TEXT,
        "models": {},
        "pairwise_probe_diffs": {},
    }
    for label, path in MODELS.items():
        cfg = AutoConfig.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        result["models"][label] = {
            "path": rel(path),
            "exists": path.exists(),
            "config_architectures": getattr(cfg, "architectures", None),
            "config_auto_map": getattr(cfg, "auto_map", None),
            "config_private_adapter_scale": getattr(cfg, "private_adapter_scale", None),
            **count_private_keys(path),
            "AutoModel": try_load_auto(path),
            "AutoModelForMaskedLM": try_load_mlm(path),
        }
        print(json.dumps({"event": "model_done", "label": label, "AutoModel": result["models"][label]["AutoModel"], "MLM_class": result["models"][label]["AutoModelForMaskedLM"].get("class")})[:2000], flush=True)

    # Directly test whether AutoModel sees dense/private differences on the exact hidden states supplied to classifier heads.
    try:
        h_chck = hidden_tensor_auto(MODELS["chck82"])
        for label in ["coherent86_train_scale1p0", "dense62064_u0080", "dense62065_u0080"]:
            h = hidden_tensor_auto(MODELS[label])
            result["pairwise_probe_diffs"][f"AutoModel_hidden_{label}_minus_chck82"] = {
                "max_abs": float((h - h_chck).abs().max()),
                "mean_abs": float((h - h_chck).abs().mean()),
            }
        h64 = hidden_tensor_auto(MODELS["dense62064_u0080"])
        h65 = hidden_tensor_auto(MODELS["dense62065_u0080"])
        result["pairwise_probe_diffs"]["AutoModel_hidden_dense62065_minus_dense62064"] = {
            "max_abs": float((h65 - h64).abs().max()),
            "mean_abs": float((h65 - h64).abs().mean()),
        }
    except Exception as e:
        result["pairwise_probe_diffs"]["AutoModel_error"] = repr(e)

    try:
        l_chck = logits_tensor_mlm(MODELS["chck82"], private_on=False)
        for label in ["coherent86_train_scale1p0", "dense62064_u0080", "dense62065_u0080"]:
            l_on = logits_tensor_mlm(MODELS[label], private_on=True)
            result["pairwise_probe_diffs"][f"AutoModelForMaskedLM_private_on_logits_{label}_minus_chck82_off"] = {
                "max_abs": float((l_on - l_chck).abs().max()),
                "mean_abs": float((l_on - l_chck).abs().mean()),
            }
    except Exception as e:
        result["pairwise_probe_diffs"]["MLM_diff_error"] = repr(e)

    json_path = _public_path('experiments/archive/relation_learning/data/superglue_loader_identity_audit/loader_identity_audit.json')
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research SuperGLUE loader identity audit", "", result["question"], ""]
    for label, rec in result["models"].items():
        am = rec["AutoModel"]
        mlm = rec["AutoModelForMaskedLM"]
        lines.append(f"## {label}")
        lines.append(f"- path: `{rec['path']}`")
        lines.append(f"- auto_map: `{rec['config_auto_map']}`; architectures: `{rec['config_architectures']}`; private state tensors/params: `{rec['private_state_keys']}` / `{rec['private_state_params']}`")
        lines.append(f"- AutoModel: ok `{am.get('ok')}`, class `{am.get('module')}.{am.get('class')}`, private named params `{am.get('private_named_params')}`, unexpected keys `{am.get('loading_info_unexpected_count')}`, examples `{am.get('unexpected_examples')}`")
        lines.append(f"- AutoModelForMaskedLM: ok `{mlm.get('ok')}`, class `{mlm.get('module')}.{mlm.get('class')}`, private named params `{mlm.get('private_named_params')}`, private on/off probe maxdiff `{mlm.get('probe_private_on_off_logit_maxdiff')}`")
        lines.append("")
    lines.append("## Pairwise probe differences")
    for k, v in result["pairwise_probe_diffs"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("Interpretation: if AutoModel has zero private named parameters and its hidden states are identical to chck82 while AutoModelForMaskedLM sees nonzero private-on/off logits, then SuperGLUE finetuning based on AutoModel cannot be used as evidence about dense/private continuation. Zero-shot and Reading remain separate because they call AutoModelForMaskedLM with trust_remote_code.")
    lines.append(f"\nJSON: `{rel(json_path)}`")
    (_public_path('research/documents/relation_learning/data/superglue_loader_identity_audit/loader_identity_audit.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "json": rel(json_path), "md": rel(_public_path('research/documents/relation_learning/data/superglue_loader_identity_audit/loader_identity_audit.md')), "pairwise_probe_diffs": result["pairwise_probe_diffs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
