#!/usr/bin/env python3
"""Integrity/provenance checks for research parent-anchor continuation.

This script is intentionally CPU-safe.  It records endpoint hashes, private tensor/key
integrity, executed private scales, and first-update pairing information for the three
ordinary continuation references:
  * research standard with carrier-off neutral KL;
  * research parent-on neutral KL;
  * optional no-neutral-KL bracket, if/when launched.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/frontier_consolidation')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))
sys.path.insert(0, str(A02_SCRIPTS))

import torch
from safetensors.torch import load_file

import coherent86_continuation_trainer as prev

PARENT = _public_path('models/frontier')
RUNS = {
    "standard_carrier_anchor": _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_standard_seed43023'),
    "parent_anchor": _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023'),
    "no_kl": _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_no_kl_seed43023'),
}
OUT_DIR = _public_path('experiments/archive/functional_learning/data/integrity_checks')


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"status": "json_read_failed", "error": str(e)}


def first_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def parent_model_integrity(parent: Path) -> dict[str, Any]:
    cfg = load_json(parent / "config.json") or {}
    model_sha = sha256(parent / "model.safetensors")
    sd = load_file(str(parent / "model.safetensors"), device="cpu")
    private_keys = sorted(k for k in sd if ".private_adapter." in k)
    device = torch.device("cpu")
    model, missing, unexpected = prev.load_model(parent, device, int(cfg.get("private_adapter_bottleneck", 128)), 0.75)
    scales = [float(layer.private_adapter.scale) for layer in model.deberta.encoder.layer]
    private_missing = [k for k in missing if "private_adapter" in k]
    tied_missing = [k for k in missing if k in {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}]
    del model
    return {
        "path": rel(parent),
        "model_safetensors_sha256": model_sha,
        "config_private_adapter_scale": cfg.get("private_adapter_scale"),
        "config_adapter_scale": cfg.get("adapter_scale"),
        "config_private_adapter_enabled": cfg.get("private_adapter_enabled"),
        "private_key_count": len(private_keys),
        "private_key_examples": private_keys[:4] + private_keys[-4:],
        "load_missing": missing,
        "load_unexpected": unexpected,
        "missing_private_keys": private_missing,
        "missing_tied_decoder_keys": tied_missing,
        "executed_private_scales_after_load": scales,
        "all_executed_scales_0p75": all(abs(x - 0.75) < 1e-12 for x in scales),
    }


def run_record(name: str, run_dir: Path) -> dict[str, Any]:
    first = first_jsonl(run_dir / "training_log.jsonl")
    last = last_jsonl(run_dir / "training_log.jsonl")
    metrics = load_json(run_dir / "scientific_metrics.json")
    cfg = load_json(run_dir / "train_config.json")
    manifest = load_json(run_dir / "parent_anchor_manifest.json")
    return {
        "name": name,
        "run_dir": rel(run_dir),
        "exists": run_dir.exists(),
        "train_config": cfg,
        "scientific_metrics": metrics,
        "parent_anchor_manifest": manifest,
        "first_update": first,
        "last_update": last,
    }


def compare_first_updates(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    std = records.get("standard_carrier_anchor", {}).get("first_update")
    out = []
    if not std:
        return out
    for name, rec in records.items():
        if name == "standard_carrier_anchor":
            continue
        first = rec.get("first_update")
        if not first:
            continue
        fields = ["source_row_start", "source_row_end", "macro_words", "tail_words", "total_consumed_words", "main_targets", "schedule_index", "lr", "main_loss"]
        out.append({
            "compare_to_step027_standard": name,
            "field_deltas": {f: (None if first.get(f) is None or std.get(f) is None else first.get(f) - std.get(f)) for f in fields if isinstance(first.get(f), (int, float)) and isinstance(std.get(f), (int, float))},
            "field_equalities": {f: first.get(f) == std.get(f) for f in fields},
            "neutral_loss_step027_standard": std.get("neutral_loss"),
            "neutral_loss_other": first.get("neutral_loss"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    result: dict[str, Any] = {
        "status": "INTEGRITY_CHECKS",
        "parent": parent_model_integrity(PARENT),
        "runs": {},
    }
    for name, path in RUNS.items():
        result["runs"][name] = run_record(name, path)
    result["first_update_pairing"] = compare_first_updates(result["runs"])
    out_json = out_dir / "integrity_checks.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# research integrity checks", "", f"Parent: `{rel(PARENT)}`", ""]
    p = result["parent"]
    lines.append(f"- parent SHA256: `{p['model_safetensors_sha256']}`")
    lines.append(f"- private tensor keys: {p['private_key_count']}; missing private keys on load: {len(p['missing_private_keys'])}; unexpected: {len(p['load_unexpected'])}")
    lines.append(f"- executed private scales after load: {p['executed_private_scales_after_load']}")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for name, rec in result["runs"].items():
        met = rec.get("scientific_metrics") or {}
        first = rec.get("first_update") or {}
        lines.append(f"- `{name}` exists={rec['exists']} total={met.get('total_consumed_words')} updates={met.get('updates')} first_main={first.get('main_loss')} first_neutral={first.get('neutral_loss')}")
    lines.append("")
    lines.append("## First-update pairing vs research standard")
    lines.append("")
    for cmp in result["first_update_pairing"]:
        lines.append(f"- `{cmp['compare_to_step027_standard']}` equalities: {cmp['field_equalities']}; neutral losses {cmp['neutral_loss_step027_standard']} vs {cmp['neutral_loss_other']}")
    out_md = out_dir / "integrity_checks.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}), flush=True)


if __name__ == "__main__":
    main()
