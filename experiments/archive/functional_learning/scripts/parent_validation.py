#!/usr/bin/env python3
"""Validate exact parent/load invariants for research coherent86 continuation."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = _public_path('.')
A02 = _public_path('experiments/archive/frontier_consolidation')
STUDY = _public_path('experiments/archive/functional_learning')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(A02_SCRIPTS))

from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402

PARENT = _public_path('models/frontier')
SOURCE_ALPHA1 = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final')
ANCHOR82 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
EXPECTED_PARENT_SHA = "e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8"
EXPECTED_CHCK82_SHA = "93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3"
OUT = _public_path('experiments/archive/functional_learning/data/parent_validation')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(path: Path):
    from transformers import DebertaV2Config
    cfg = DebertaV2Config.from_pretrained(str(path), local_files_only=True)
    model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
    sd = load_file(str(path / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.tie_weights()
    return cfg, model, sd, missing, unexpected


def tensor_diff_max(path_a: Path, path_b: Path, prefix_filter: str | None = None) -> dict:
    a = load_file(str(path_a / "model.safetensors"), device="cpu")
    b = load_file(str(path_b / "model.safetensors"), device="cpu")
    keys = sorted(set(a) & set(b))
    if prefix_filter is not None:
        keys = [k for k in keys if prefix_filter in k]
    max_abs = 0.0
    max_key = None
    unequal = 0
    for k in keys:
        if a[k].shape != b[k].shape:
            unequal += 1
            max_key = k
            max_abs = float("inf")
            continue
        d = (a[k].float() - b[k].float()).abs().max().item()
        if d != 0.0:
            unequal += 1
        if d > max_abs:
            max_abs = float(d)
            max_key = k
    return {"n_keys": len(keys), "unequal_keys": unequal, "max_abs": max_abs, "max_key": max_key}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg, model, sd, missing, unexpected = load_model(PARENT)
    private_keys = sorted(k for k in sd if ".private_adapter." in k)
    adapter_keys = sorted(k for k in sd if ".adapter." in k and ".private_adapter." not in k)
    scales = [float(layer.private_adapter.scale) for layer in model.deberta.encoder.layer]
    enabled = [bool(layer.private_adapter.enabled) for layer in model.deberta.encoder.layer]
    parent_sha = sha256(_public_path('models/frontier/model.safetensors'))
    anchor_sha = sha256(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M/model.safetensors')) if (_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M/model.safetensors')).exists() else None
    source_sha = sha256(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final/model.safetensors'))
    rec = {
        "status": "PARENT_VALIDATION_COMPLETE",
        "parent": str(PARENT.relative_to(ROOT)),
        "parent_model_sha256": parent_sha,
        "parent_sha_matches_expected": parent_sha == EXPECTED_PARENT_SHA,
        "source_alpha1_model_sha256": source_sha,
        "parent_tensors_identical_to_step150_alpha1_source": parent_sha == source_sha,
        "chck82": str(ANCHOR82.relative_to(ROOT)),
        "chck82_model_sha256": anchor_sha,
        "chck82_sha_matches_expected": anchor_sha == EXPECTED_CHCK82_SHA,
        "config_private_adapter_scale": float(getattr(cfg, "private_adapter_scale", -1.0)),
        "executed_private_adapter_scales": scales,
        "all_executed_scales_0p75": all(abs(x - 0.75) < 1e-9 for x in scales),
        "private_enabled_flags": enabled,
        "private_key_count": len(private_keys),
        "old_adapter_key_count": len(adapter_keys),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "missing_private_keys": [k for k in missing if "private_adapter" in k],
        "bad_missing_non_decoder_non_private": [k for k in missing if "private_adapter" not in k and k not in {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}],
        "private_tensor_diff_parent_vs_alpha1_source": tensor_diff_max(PARENT, SOURCE_ALPHA1, ".private_adapter."),
        "all_common_tensor_diff_parent_vs_alpha1_source": tensor_diff_max(PARENT, SOURCE_ALPHA1, None),
    }
    out_json = _public_path('experiments/archive/functional_learning/data/parent_validation/parent_validation.json')
    out_json.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# research parent/load validation",
        "",
        f"Parent SHA `{parent_sha}`; expected match: `{rec['parent_sha_matches_expected']}`.",
        f"Parent tensor file identical to research alpha1 source tensor file: `{rec['parent_tensors_identical_to_step150_alpha1_source']}`.",
        f"Config private scale `{rec['config_private_adapter_scale']}`; executed layer scales `{scales}`.",
        f"Private key count `{len(private_keys)}`; missing keys `{missing}`; unexpected keys `{unexpected}`.",
        f"Missing private keys: `{rec['missing_private_keys']}`.",
        "",
        "Interpretation: research can rely on the exact coherent86 alpha0.75 tensor/config parent only if the expected SHA, private-key presence, and executed scale checks are true.",
    ]
    out_md = _public_path('research/documents/functional_learning/data/parent_validation/parent_validation.md')
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": rec["status"], "out_json": str(out_json.relative_to(ROOT)), "out_md": str(out_md.relative_to(ROOT)), "parent_sha_matches_expected": rec["parent_sha_matches_expected"], "missing_private_keys": rec["missing_private_keys"], "all_executed_scales_0p75": rec["all_executed_scales_0p75"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
