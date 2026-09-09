#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_one(path: Path) -> dict:
    rec = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return {**rec, "ok": False, "error": "missing"}
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        rec.update({
            "ok": True,
            "tokenizer_class": tok.__class__.__name__,
            "model_class": model.__class__.__name__,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "n_layer": getattr(model.config, "n_layer", None),
            "n_embd": getattr(model.config, "n_embd", None),
            "n_head": getattr(model.config, "n_head", None),
        })
    except Exception as e:
        rec.update({"ok": False, "error_type": type(e).__name__, "error": str(e)})
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoints", nargs="*", default=[])
    args = ap.parse_args()
    run = Path(args.run_dir)
    metrics_path = run / "scientific_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    names = args.checkpoints or [x.get("name") for x in metrics.get("saved_checkpoints", []) if x.get("name")]
    out = {
        "run_dir": str(run),
        "metrics_present": metrics_path.exists(),
        "saved_checkpoints": metrics.get("saved_checkpoints"),
        "loads": [load_one(run / "hf_model")],
    }
    for name in names:
        out["loads"].append(load_one(run / "hf_model" / name))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
