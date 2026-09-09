#!/usr/bin/env python3
"""research: mature checkpoint averaging for the legal40k fixed-256 anchor.

Scientific purpose:
  Test Route C from the research portfolio without new training: whether nearby mature
  legal40k fixed-256 checkpoints contain complementary predictions that a simple
  loss-agnostic average can expose.  This is not a submission endpoint by itself;
  it preserves the exact legal tokenizer/config and averages only already-trained
  compliant checkpoints from research.

Outputs:
  - one HF-loadable averaged checkpoint per predeclared window under
    training/runs/mature_checkpoint_averages/<spec>/hf_model/chck_avg
  - per-run scientific_metrics.json and a global manifest in data/

No model training is performed.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
SOURCE_RUN = A01_WS / "training/runs/legal40k_accum_compact_view_reinvest_seed43022"
SOURCE_HF = SOURCE_RUN / "hf_model"
OUT_RUN_ROOT = A01_WS / "training/runs/mature_checkpoint_averages"
OUT_DATA = A01_WS / "data/mature_checkpoint_averages"
COPY_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def checkpoint_path(million: int) -> Path:
    return SOURCE_HF / f"chck_{million}M" / "model.safetensors"


def normalize(weights: dict[int, float]) -> dict[int, float]:
    s = sum(weights.values())
    if not s:
        raise ValueError("zero weight sum")
    return {k: float(v) / float(s) for k, v in sorted(weights.items())}


def specs() -> dict[str, dict[str, Any]]:
    uniform_80_100 = normalize({m: 1.0 for m in range(80, 101)})
    uniform_90_100 = normalize({m: 1.0 for m in range(90, 101)})
    # Exponential mature window from 70M to 100M, half-life 10M, late checkpoints weighted more.
    exp_70_100_hl10 = normalize({m: 2.0 ** ((m - 100) / 10.0) for m in range(70, 101)})
    # Linear tail average, emphasizing the final part without using training loss or eval rows.
    linear_90_100 = normalize({m: float(m - 89) for m in range(90, 101)})
    return {
        "avg_uniform_80_100": {
            "description": "uniform average of legal40k fixed-256 checkpoints chck_80M..chck_100M inclusive",
            "weights": uniform_80_100,
        },
        "avg_uniform_90_100": {
            "description": "uniform average of legal40k fixed-256 checkpoints chck_90M..chck_100M inclusive",
            "weights": uniform_90_100,
        },
        "avg_exp_70_100_hl10": {
            "description": "exponentially late-weighted average of chck_70M..chck_100M inclusive; half-life 10M",
            "weights": exp_70_100_hl10,
        },
        "avg_linear_90_100": {
            "description": "linearly late-weighted average of chck_90M..chck_100M inclusive",
            "weights": linear_90_100,
        },
    }


def copy_hf_sidecars(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        src = SOURCE_HF / "chck_100M" / name
        if not src.exists():
            src = SOURCE_HF / name
        if not src.exists():
            raise FileNotFoundError(f"missing HF sidecar {name} in source")
        shutil.copy2(src, dst / name)


def average_one(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    weights: dict[int, float] = spec["weights"]
    missing = [m for m in weights if not checkpoint_path(m).exists()]
    if missing:
        raise FileNotFoundError(f"missing source checkpoints for {name}: {missing}")
    dst_run = OUT_RUN_ROOT / name
    dst_model = dst_run / "hf_model" / "chck_avg"
    dst_model.mkdir(parents=True, exist_ok=True)

    accum: dict[str, torch.Tensor] = {}
    ref_dtypes: dict[str, torch.dtype] = {}
    tensor_keys: set[str] | None = None
    source_hashes: dict[str, str] = {}
    t0 = time.time()
    for idx, (million, w) in enumerate(sorted(weights.items())):
        path = checkpoint_path(million)
        source_hashes[f"chck_{million}M"] = sha256_file(path)
        state = load_file(str(path), device="cpu")
        keys = set(state.keys())
        if tensor_keys is None:
            tensor_keys = keys
        elif keys != tensor_keys:
            raise RuntimeError({"spec": name, "checkpoint": million, "key_mismatch": sorted(keys ^ tensor_keys)[:20], "n_diff": len(keys ^ tensor_keys)})
        for k, v in state.items():
            if idx == 0:
                ref_dtypes[k] = v.dtype
                if torch.is_floating_point(v):
                    accum[k] = v.detach().to(torch.float32).mul(float(w))
                else:
                    # Non-floating buffers are expected to be absent in safetensors LM checkpoints;
                    # if present, keep the latest checkpoint value after verifying equality later.
                    accum[k] = v.detach().clone()
            else:
                if torch.is_floating_point(v):
                    accum[k].add_(v.detach().to(torch.float32), alpha=float(w))
                else:
                    if not torch.equal(accum[k], v):
                        # Fall back to the final checkpoint buffer for non-floats; record below.
                        accum[k] = v.detach().clone()
        del state

    final_state: dict[str, torch.Tensor] = {}
    for k, v in accum.items():
        dtype = ref_dtypes[k]
        if torch.is_floating_point(v) and dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            final_state[k] = v.to(dtype)
        else:
            final_state[k] = v
    save_file(final_state, str(dst_model / "model.safetensors"))
    copy_hf_sidecars(dst_model)

    model_sha = sha256_file(dst_model / "model.safetensors")
    config_sha = sha256_file(dst_model / "config.json")
    tokenizer_sha = sha256_file(dst_model / "tokenizer.json")
    metrics = {
        "status": "MATURE_CHECKPOINT_AVERAGE",
        "created_utc": now_utc(),
        "variant": name,
        "description": spec["description"],
        "not_training": True,
        "legal_status": "Averaged only existing research compliant legal40k fixed-256 checkpoints; no additional data exposure or tokenizer training.",
        "source_run": str(SOURCE_RUN),
        "source_checkpoint_weights": {f"chck_{m}M": w for m, w in sorted(weights.items())},
        "endpoint": "chck_avg",
        "hf_model_path": str(dst_model),
        "model_sha256": model_sha,
        "config_sha256": config_sha,
        "tokenizer_sha256": tokenizer_sha,
        "parameter_count": sum(int(t.numel()) for t in final_state.values() if torch.is_floating_point(t)),
        "tensor_count": len(final_state),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (dst_run / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (dst_model / "averaging_manifest.json").write_text(json.dumps(metrics | {"source_model_hashes": source_hashes}, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    OUT_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    all_specs = specs()
    results: dict[str, Any] = {
        "status": "MATURE_CHECKPOINT_AVERAGES_DONE",
        "created_utc": now_utc(),
        "source_run": str(SOURCE_RUN),
        "route": "cheap mature-window consolidation from existing legal40k fixed-256 checkpoints; no new training",
        "results": {},
    }
    for name, spec in all_specs.items():
        print(json.dumps({"event": "average_start", "name": name, "n_checkpoints": len(spec["weights"])}), flush=True)
        res = average_one(name, spec)
        results["results"][name] = res
        print(json.dumps({"event": "average_done", "name": name, "hf_model_path": res["hf_model_path"], "elapsed_sec": res["elapsed_sec"]}), flush=True)
    out = OUT_DATA / "mature_checkpoint_averages_manifest.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": results["status"], "manifest": str(out), "variants": list(results["results"])}), flush=True)


if __name__ == "__main__":
    main()
