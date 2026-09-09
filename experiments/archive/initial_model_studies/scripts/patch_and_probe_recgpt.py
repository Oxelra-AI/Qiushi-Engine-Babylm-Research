#!/usr/bin/env python3
"""research — Patch public RecGPT tokenizer loading and smoke-test model.

The public HF repo has tokenizer_config.json with tokenizer_class=TokenizersBackend,
which this runtime cannot import. The actual tokenizer.json is standard. This script
creates a local patched snapshot directory with a standard tokenizer_config and verifies
that AutoTokenizer/AutoModelForCausalLM load from that local directory.
"""
from __future__ import annotations
import json
import pathlib
import shutil
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

MODEL_ID = "Serdar404/RecGPT-10M"
ROOT = pathlib.Path("experiments/archive/initial_model_studies")
OUT = ROOT / "data/recgpt_local"
LOCAL = OUT / "patched_model"


def copytree_flat(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)


def patch_tokenizer_config(local: pathlib.Path) -> None:
    # Validate tokenizer.json directly first.
    tok_file = local / "tokenizer.json"
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_file), pad_token="<pad>")
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "backend": "tokenizers",
        "pad_token": "<pad>",
        "model_max_length": 1024,
        "unk_token": None,
        "bos_token": None,
        "eos_token": None,
        "clean_up_tokenization_spaces": False,
    }
    (local / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (local / "special_tokens_map.json").write_text(json.dumps({"pad_token": "<pad>"}, indent=2) + "\n", encoding="utf-8")
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snap = pathlib.Path(snapshot_download(MODEL_ID, local_dir=str(OUT / "snapshot_raw"), local_dir_use_symlinks=False))
    if LOCAL.exists():
        shutil.rmtree(LOCAL)
    copytree_flat(snap, LOCAL)
    patch_tokenizer_config(LOCAL)

    tok = AutoTokenizer.from_pretrained(str(LOCAL), trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(LOCAL), trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    enc = tok("The cat sat on the mat.", return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        logits = model(**enc).logits
    meta = {
        "status": "RECGPT_PATCHED_LOAD_OK",
        "model_id": MODEL_ID,
        "raw_snapshot": str(snap),
        "patched_model_dir": str(LOCAL),
        "tokenizer_class": tok.__class__.__name__,
        "tokenizer_vocab_size": len(tok),
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "model_class": model.__class__.__name__,
        "config_class": model.config.__class__.__name__,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "logits_shape": list(logits.shape),
        "logits_all_finite": bool(torch.isfinite(logits).all()),
        "config": model.config.to_dict(),
    }
    (OUT / "patched_load_probe.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ["patched_model_dir", "tokenizer_class", "tokenizer_vocab_size", "pad_token_id", "model_class", "num_parameters", "logits_shape", "logits_all_finite"]}, indent=2))
    print(f"Saved: {OUT / 'patched_load_probe.json'}")


if __name__ == "__main__":
    main()
