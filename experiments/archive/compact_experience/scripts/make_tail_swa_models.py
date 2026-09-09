#!/usr/bin/env python3
"""Create tail checkpoint-averaged DeBERTa-v2 MLM models from existing clean-Qwen checkpoints.

This is a no-training optimization/consolidation bet: average weights from several late
checkpoints that have complementary no-AoA profiles, then evaluate the averaged model on
no-AoA columns before any full official measurement. It never combines scores across models;
it creates a single new model artifact per averaged window.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import shutil
import time
from typing import Dict, List

import torch
from safetensors.torch import load_file, save_file

ROOT = _public_path('experiments/archive/compact_experience')
DEFAULT_RUN = _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model')
OUT_ROOT = _public_path('experiments/archive/compact_experience/data/tail_swa_models')
CONFIG_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ckpt_dir(root: pathlib.Path, name: str) -> pathlib.Path:
    p = root / name
    if not p.exists():
        raise FileNotFoundError(p)
    if not (p / "model.safetensors").exists():
        raise FileNotFoundError(p / "model.safetensors")
    return p


def copy_tokenizer_and_config(src_root: pathlib.Path, dst: pathlib.Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    # Prefer root-level tokenizer/config, fall back to first checkpoint file if necessary.
    for fn in CONFIG_FILES:
        src = src_root / fn
        if not src.exists():
            # search one checkpoint fallback
            for ch in sorted(src_root.glob("chck_*M")):
                cand = ch / fn
                if cand.exists():
                    src = cand
                    break
        if src.exists():
            shutil.copy2(src, dst / fn)


def average_window(src_root: pathlib.Path, names: List[str], dst: pathlib.Path) -> Dict[str, object]:
    ckpts = [ckpt_dir(src_root, n) for n in names]
    acc: Dict[str, torch.Tensor] = {}
    dtypes: Dict[str, str] = {}
    keys0 = None
    for i, ch in enumerate(ckpts):
        sd = load_file(str(ch / "model.safetensors"), device="cpu")
        keys = set(sd.keys())
        if keys0 is None:
            keys0 = keys
        elif keys != keys0:
            raise RuntimeError(f"state dict key mismatch at {ch}")
        for k, v in sd.items():
            dtypes.setdefault(k, str(v.dtype))
            vf = v.detach().to(torch.float32)
            if i == 0:
                acc[k] = vf
            else:
                acc[k].add_(vf)
        del sd
    n = float(len(ckpts))
    out_sd = {k: (v / n) for k, v in acc.items()}
    copy_tokenizer_and_config(src_root, dst)
    save_file(out_sd, str(dst / "model.safetensors"), metadata={"format": "pt", "averaged_from": ",".join(names)})
    return {"out_model": str(dst), "checkpoints": names, "n": len(names), "num_tensors": len(out_sd), "dtype_reference_sample": dict(list(dtypes.items())[:5]), "model_bytes": (dst / "model.safetensors").stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default=str(DEFAULT_RUN))
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--windows", nargs="*", default=["90,95,100", "85,90,95,100", "75,90,95,100"], help="Comma-separated M values per averaged model")
    args = ap.parse_args()
    src_root = pathlib.Path(args.src_root)
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for win in args.windows:
        vals = [x.strip() for x in win.split(",") if x.strip()]
        names = [f"chck_{v}M" for v in vals]
        label = "swa_" + "_".join(vals) + "M"
        dst = out_root / label
        rows.append(average_window(src_root, names, dst))
        rows[-1]["label"] = label
    payload = {
        "status": "TAIL_SWA_MODELS_CREATED",
        "created_utc": now(),
        "purpose": "No-training consolidation probe. Average existing clean-Qwen checkpoint weights into single HF-compatible candidate models; evaluate each model directly rather than mixing scores.",
        "non_leakage_statement": "Uses only trained checkpoint weights and tokenizer/config from existing clean-Qwen run. It does not read official AoA/CDI items, child curves, AoA scores, SuperGLUE labels, or downstream evaluation outputs for construction.",
        "src_root": str(src_root),
        "models": rows,
    }
    meta = out_root / "tail_swa_metadata.json"
    meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": str(meta), "models": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
