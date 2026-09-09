#!/usr/bin/env python3
"""Verify BabyLM pilot HF checkpoint portability and print compact evidence JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = {"run_dir": str(run), "files": {}, "loads": []}
    metrics_path = run / "scientific_metrics.json"
    if metrics_path.exists():
        out["metrics"] = json.loads(metrics_path.read_text())
    for rel in [
        "data_manifest.json",
        "tokenizer_manifest.json",
        "hf_model/tokenizer_config.json",
        "hf_model/chck_1M/tokenizer_config.json",
        "hf_model/config.json",
        "hf_model/chck_1M/config.json",
    ]:
        p = run / rel
        entry = {"exists": p.exists()}
        if p.exists():
            entry["bytes"] = p.stat().st_size
            try:
                entry["json"] = json.loads(p.read_text())
            except Exception as exc:
                entry["json_error"] = repr(exc)
        out["files"][rel] = entry
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = run / rel
        rec = {"path": str(p)}
        try:
            tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True)
            rec.update({
                "ok": True,
                "tokenizer_class": tok.__class__.__name__,
                "vocab_size": len(tok),
                "pad_token_id": tok.pad_token_id,
                "bos_token_id": tok.bos_token_id,
                "eos_token_id": tok.eos_token_id,
                "model_class": model.__class__.__name__,
                "parameter_count": sum(x.numel() for x in model.parameters()),
                "n_layer": getattr(model.config, "n_layer", None),
                "n_embd": getattr(model.config, "n_embd", None),
                "n_head": getattr(model.config, "n_head", None),
            })
        except Exception as exc:
            rec.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        out["loads"].append(rec)
    # Local directories do not necessarily support HF branch-style revision names.
    rec = {"path": str(run / "hf_model"), "revision": "chck_1M"}
    try:
        tok = AutoTokenizer.from_pretrained(run / "hf_model", trust_remote_code=True, revision="chck_1M")
        model = AutoModelForCausalLM.from_pretrained(run / "hf_model", trust_remote_code=True, revision="chck_1M")
        rec.update({"ok": True, "tokenizer_class": tok.__class__.__name__, "parameter_count": sum(x.numel() for x in model.parameters())})
    except Exception as exc:
        rec.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
    out["revision_load"] = rec
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
