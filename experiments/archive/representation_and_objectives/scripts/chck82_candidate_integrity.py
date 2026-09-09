#!/usr/bin/env python3
"""Integrity record for research scale1.75 chck_82M candidate."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
from pathlib import Path
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
RUN_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"
HF = RUN_DIR / "hf_model"
OUT_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_chck82_integrity"
ENDPOINTS = ["chck_80M", "chck_82M", "chck_100M"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256(path) if path.exists() else None}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((RUN_DIR / "scientific_metrics.json").read_text(encoding="utf-8"))
    saved = metrics.get("saved_checkpoints", [])
    records = {}
    for ep in ENDPOINTS:
        epdir = HF / ep
        records[ep] = {
            "dir": rel(epdir),
            "exists": epdir.exists(),
            "files": {name: file_record(epdir / name) for name in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "adapter_scaled_modeling.py"]},
        }
    saved_lookup = {x.get("name"): x for x in saved if isinstance(x, dict)}
    out = {
        "status": "CHCK82_CANDIDATE_INTEGRITY",
        "created_utc": now(),
        "run_dir": rel(RUN_DIR),
        "candidate_endpoint": "chck_82M",
        "checkpoint_file_records": records,
        "saved_checkpoint_count": len(saved),
        "saved_checkpoint_records": {ep: saved_lookup.get(ep) for ep in ENDPOINTS},
        "training_summary": {k: metrics.get(k) for k in ["variant", "backend", "model_family", "parameter_count", "vocab_size", "tokenizer_label", "word_exposure", "actual_training_steps", "loss_first", "loss_last", "seed", "extra_init_seed", "train_rng_seed", "mask_mode", "seq_length", "batch_size", "optimizer", "learning_rate", "n_layer", "hidden_size", "n_head", "intermediate_size"] if k in metrics},
    }
    j = OUT_DIR / "chck82_candidate_integrity.json"
    j.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = OUT_DIR / "chck82_candidate_integrity.md"
    lines = ["# research chck_82M candidate integrity", "", f"Run: `{rel(RUN_DIR)}`", ""]
    for ep, rec in records.items():
        m = rec["files"]["model.safetensors"]
        lines.append(f"- {ep}: model size {m['size_bytes']} sha256 `{m['sha256']}`")
    lines += ["", f"JSON: `{rel(j)}`"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "candidate_sha256": records["chck_82M"]["files"]["model.safetensors"]["sha256"], "json": rel(j), "md": rel(md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
