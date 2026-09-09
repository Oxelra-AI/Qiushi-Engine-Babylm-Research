#!/usr/bin/env python3
"""research: inspect whether BabyLM candidate models tie input embeddings to output readouts."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/model_tiedness_probe.py')
ROOT = _PUBLIC_ROOT

MODELS = {
    "deberta_representation_frontier_studies_clean": ("mlm", ROOT / "experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model/chck_100M"),
    "deberta_compact_experience_qwen_aligned": ("mlm", ROOT / "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_100M"),
    "roberta_functional_relation_studies_repeat_split": ("mlm", ROOT / "experiments/archive/relation_learning/training/runs/roberta_repeat_split_dose2p64x_rowholdout_100M_seed43022/hf_model/chck_100M"),
    "causal_gpt_repeat": ("causal", ROOT / "experiments/archive/relation_learning/training/runs/causal_gpt_r_dose2p64x_seed43022/hf_model/chck_100M"),
}

OUT = ROOT / "experiments/archive/relation_learning/data/model_tiedness_probe/model_tiedness.json"
NOTE = ROOT / "research/notes/relation_learning/model_tiedness_probe.md"


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def inspect_model(label: str, kind: str, path: pathlib.Path) -> dict[str, Any]:
    cls = AutoModelForCausalLM if kind == "causal" else AutoModelForMaskedLM
    model = cls.from_pretrained(str(path), torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    rec: dict[str, Any] = {
        "label": label,
        "kind": kind,
        "path": rel(path),
        "model_class": model.__class__.__name__,
        "config_model_type": getattr(model.config, "model_type", None),
        "config_architectures": getattr(model.config, "architectures", None),
        "config_tie_word_embeddings": getattr(model.config, "tie_word_embeddings", None),
        "input_module": inp.__class__.__name__ if inp is not None else None,
        "output_module": out.__class__.__name__ if out is not None else None,
    }
    if inp is not None:
        rec["input_weight_shape"] = list(inp.weight.shape)
    if out is not None and hasattr(out, "weight"):
        rec["output_weight_shape"] = list(out.weight.shape)
        same_ptr = inp is not None and inp.weight.data_ptr() == out.weight.data_ptr()
        rec["same_tensor_data_ptr"] = bool(same_ptr)
        if inp is not None and tuple(inp.weight.shape) == tuple(out.weight.shape):
            # Only compute a bounded exact-difference summary on CPU.
            rec["max_abs_weight_difference"] = float((inp.weight.detach() - out.weight.detach()).abs().max().item())
    del model
    return rec


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, (kind, path) in MODELS.items():
        rows.append(inspect_model(label, kind, path))
    payload = {"status": "MODEL_TIEDNESS_DONE", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": rows}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# research model tiedness probe", "", "This records whether representative BabyLM coordinates tie input embeddings to the output readout matrix, for comparison with functional_learning's tied-softmax mechanism note.", "", "| label | model class | tie_word_embeddings | same tensor | input shape | output shape |", "|---|---|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['model_class']} | {r.get('config_tie_word_embeddings')} | {r.get('same_tensor_data_ptr')} | {r.get('input_weight_shape')} | {r.get('output_weight_shape')} |")
    lines += ["", f"JSON: `{rel(OUT)}`."]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "note": rel(NOTE), "json": rel(OUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
