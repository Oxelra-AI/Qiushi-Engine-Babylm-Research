#!/usr/bin/env python3
"""research: verify coherent88 private-OFF function recovers the chck_84M anchor.

The research coherent88 run attached a private adapter path to the verified chck_84M
slow function. Before evaluating inference-time alpha variants, verify that disabling
the private path gives exactly the anchor logits on deterministic probe texts.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
ANCHOR = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_84M')
COHERENT = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen84_coherent88_seed43022/hf_model/final')
OUT = _public_path('experiments/archive/frontier_consolidation/data/coherent88_identity_check/coherent88_private_off_vs_chck84.json')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def main() -> None:
    cache = _public_path('experiments/archive/frontier_consolidation/data/coherent88_identity_check/runtime_cache')
    for k, sub in {
        "HF_HOME": "hf_home",
        "HF_HUB_CACHE": "hf_home/hub",
        "HUGGINGFACE_HUB_CACHE": "hf_home/hub",
        "TRANSFORMERS_CACHE": "transformers",
        "HF_MODULES_CACHE": "modules",
        "HF_DATASETS_CACHE": "datasets",
        "TMPDIR": "tmp",
    }.items():
        p = cache / sub
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p.resolve())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if not (_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_84M/model.safetensors')).exists():
        raise FileNotFoundError(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_84M/model.safetensors'))
    if not (_public_path('experiments/archive/frontier_consolidation/training/runs/frozen84_coherent88_seed43022/hf_model/final/model.safetensors')).exists():
        raise FileNotFoundError(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen84_coherent88_seed43022/hf_model/final/model.safetensors'))

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(ANCHOR), local_files_only=True, use_fast=True)
    anchor = AutoModelForMaskedLM.from_pretrained(str(ANCHOR), trust_remote_code=True, local_files_only=True)
    coherent = AutoModelForMaskedLM.from_pretrained(str(COHERENT), trust_remote_code=True, local_files_only=True)
    anchor.eval(); coherent.eval()
    if hasattr(coherent, "set_private_enabled"):
        coherent.set_private_enabled(False)
    else:
        raise RuntimeError("coherent model has no set_private_enabled method")

    texts = [
        "The child put the cup on the table and smiled.",
        "Older printers and table-top scientific devices use parallel ports to communicate.",
        "Hypoglycemia is usually associated with diabetes but can also be caused by a lack of food.",
        "The researchers do not claim a cause and effect connection between stress and age-related mental decline.",
        "Tulare County harvested eggplant, squash, and cucumbers early while other vegetables grew well.",
        "A healthy tree or shrub can generally tolerate total defoliation without permanent damage.",
    ]
    batch = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        a = anchor(**batch).logits.detach().float()
        b = coherent(**batch).logits.detach().float()
    diff = (a - b).abs()
    payload: dict[str, Any] = {
        "status": "PASS" if torch.equal(a, b) else ("ALLCLOSE_1E7" if torch.allclose(a, b, atol=1e-7, rtol=0.0) else "DIFF"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anchor": rel(ANCHOR),
        "coherent": rel(COHERENT),
        "anchor_class": anchor.__class__.__name__,
        "coherent_class": coherent.__class__.__name__,
        "coherent_private_enabled_after_disable": bool(getattr(coherent.config, "private_adapter_enabled", True)),
        "coherent_config_private_adapter_scale": float(getattr(coherent.config, "private_adapter_scale", -1.0)),
        "n_texts": len(texts),
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "exact_tensor_equal": bool(torch.equal(a, b)),
        "allclose_atol_1e_7_rtol0": bool(torch.allclose(a, b, atol=1e-7, rtol=0.0)),
        "anchor_model_sha256": sha256_file(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_84M/model.safetensors')),
        "coherent_model_sha256": sha256_file(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen84_coherent88_seed43022/hf_model/final/model.safetensors')),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    _public_path('experiments/archive/frontier_consolidation/data/coherent88_identity_check').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
