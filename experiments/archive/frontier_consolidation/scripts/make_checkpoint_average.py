#!/usr/bin/env python3
"""research: create arithmetic checkpoint averages for existing MLM checkpoints.

This is an execution utility for the existing-ladder route.  It averages model
weights from already trained Hugging Face checkpoints, copies tokenizer/config
files from the first checkpoint, and writes provenance metadata.  It does not
train, modify corpora, or touch the frozen 100M endpoint.

Intended examples, only after broad no-AoA checkpoint screening supports them:
  python -B experiments/archive/frontier_consolidation/scripts/make_checkpoint_average.py \
    --label reinvest43022_avg_80_90_95_100 \
    --checkpoints <seed43022>/chck_80M <seed43022>/chck_90M <seed43022>/chck_95M <seed43022>/chck_100M

  python -B experiments/archive/frontier_consolidation/scripts/make_checkpoint_average.py \
    --label reinvest43122_avg_45_80_100 \
    --checkpoints <seed43122>/chck_45M <seed43122>/chck_80M <seed43122>/chck_100M
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file, save_file

ROOT = Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
OUT_ROOT_DEFAULT = STUDY / "data/checkpoint_averages"
COPY_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "merges.txt",
]

PRESETS = {
    "reinvest43022_avg_80_90_95_100": [
        "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_80M",
        "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_90M",
        "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_95M",
        "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M",
    ],
    "reinvest43022_avg_80_90_100": [
        "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_80M",
        "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_90M",
        "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M",
    ],
    "reinvest43122_avg_45_80_100": [
        "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_45M",
        "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_80M",
        "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_100M",
    ],
    "reinvest43122_avg_45_80_90_100": [
        "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_45M",
        "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_80M",
        "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_90M",
        "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_100M",
    ],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def sha256_path(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_checkpoint(path: Path) -> Dict[str, object]:
    st = path / "model.safetensors"
    cfg = path / "config.json"
    if not st.exists():
        raise FileNotFoundError(st)
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    return {
        "path": str(path),
        "model_safetensors_size": st.stat().st_size,
        "model_safetensors_sha256": sha256_path(st),
        "config_sha256": sha256_path(cfg),
    }


def resolve_checkpoints(label: str, checkpoints: List[str] | None) -> List[Path]:
    if checkpoints:
        return [Path(x) for x in checkpoints]
    if label not in PRESETS:
        raise ValueError(f"No checkpoints supplied and label {label!r} is not a preset. Known presets: {sorted(PRESETS)}")
    return [ROOT / x for x in PRESETS[label]]


def load_metadata(checkpoints: List[Path]) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    records = [check_checkpoint(p) for p in checkpoints]
    cfg_hashes = {str(r["config_sha256"]) for r in records}
    if len(cfg_hashes) != 1:
        raise ValueError(f"Config hashes differ across checkpoints: {cfg_hashes}")
    # Tokenizers should usually match as well.  Record differences rather than
    # failing on missing optional files.
    token_hashes: Dict[str, str] = {}
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        vals = []
        for p in checkpoints:
            f = p / name
            vals.append(sha256_path(f) if f.exists() else "MISSING")
        uniq = sorted(set(vals))
        token_hashes[name] = uniq[0] if len(uniq) == 1 else "DIFF:" + ",".join(uniq)
    return records, token_hashes


def average_safetensors(checkpoints: List[Path], out_model: Path) -> Dict[str, object]:
    first_path = checkpoints[0] / "model.safetensors"
    avg = load_file(str(first_path), device="cpu")
    keys = sorted(avg)
    original_dtypes = {k: str(v.dtype) for k, v in avg.items()}
    for k in keys:
        avg[k] = avg[k].to(torch.float32)
    for ckpt in checkpoints[1:]:
        cur = load_file(str(ckpt / "model.safetensors"), device="cpu")
        if sorted(cur) != keys:
            missing = sorted(set(keys) - set(cur))[:20]
            extra = sorted(set(cur) - set(keys))[:20]
            raise ValueError(f"State-dict key mismatch for {ckpt}: missing={missing} extra={extra}")
        for k in keys:
            if tuple(cur[k].shape) != tuple(avg[k].shape):
                raise ValueError(f"Shape mismatch for {k}: {cur[k].shape} vs {avg[k].shape}")
            avg[k].add_(cur[k].to(torch.float32))
        del cur
    scale = 1.0 / float(len(checkpoints))
    out_tensors = {}
    for k, v in avg.items():
        v.mul_(scale)
        dtype_name = original_dtypes[k]
        if dtype_name == "torch.bfloat16":
            out_tensors[k] = v.to(torch.bfloat16)
        elif dtype_name == "torch.float16":
            out_tensors[k] = v.to(torch.float16)
        elif dtype_name == "torch.float32":
            out_tensors[k] = v.to(torch.float32)
        else:
            out_tensors[k] = v.to(torch.float32)
    save_file(out_tensors, str(out_model))
    return {
        "n_tensors": len(keys),
        "tensor_keys_sha256": hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest(),
        "output_model_safetensors_size": out_model.stat().st_size,
        "output_model_safetensors_sha256": sha256_path(out_model),
        "dtype_counts": {dtype: list(original_dtypes.values()).count(dtype) for dtype in sorted(set(original_dtypes.values()))},
    }


def copy_hf_files(src: Path, dst: Path) -> List[str]:
    copied = []
    for name in COPY_FILES:
        s = src / name
        if s.exists():
            shutil.copyfile(s, dst / name)
            copied.append(name)
    return copied


def main() -> None:
    args = parse_args()
    checkpoints = resolve_checkpoints(args.label, args.checkpoints)
    out_dir = args.out_root / args.label
    out_model = out_dir / "model.safetensors"
    records, token_hashes = load_metadata(checkpoints)
    payload = {
        "status": "CHECKPOINT_AVERAGE_DRY_RUN" if args.dry_run else "CHECKPOINT_AVERAGE_COMPLETE",
        "label": args.label,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Arithmetic average of existing compact_view_reinvest checkpoints for an existing-ladder stopping/averaging test; no training or corpus modification.",
        "checkpoints": records,
        "tokenizer_hashes": token_hashes,
        "output_dir": str(out_dir),
        "output_model": str(out_model),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoint_average_metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if out_dir.exists() and out_model.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = copy_hf_files(checkpoints[0], out_dir)
    avg_meta = average_safetensors(checkpoints, out_model)
    payload.update({"copied_files": copied, "average_metadata": avg_meta})
    meta_path = out_dir / "checkpoint_average_metadata.json"
    payload["metadata_path"] = str(meta_path)
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "label": args.label,
        "output_dir": str(out_dir),
        "metadata_path": str(meta_path),
        "output_model_safetensors_size": avg_meta["output_model_safetensors_size"],
        "output_model_safetensors_sha256": avg_meta["output_model_safetensors_sha256"],
        "n_tensors": avg_meta["n_tensors"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
