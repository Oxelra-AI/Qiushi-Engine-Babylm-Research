#!/usr/bin/env python3
"""research: minimal load/logit pilot for the staged alpha0.75 developmental lineage.

The pilot does not train and does not run the full BabyLM evaluation.  It proves the
staged paths can be loaded locally with trust_remote_code and that representative
ancestral/private/final revisions have the intended functions.  It also checks the
private scale re-materialization by comparing an alpha=1.0 coherent-tail checkpoint
against its staged alpha=0.75 copy on a fixed masked input.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
A01 = USER_ROOT / "experiments/archive/representation_and_objectives"
DEFAULT_STAGE = A01 / "data/alpha075_developmental_staging"
DEFAULT_OUT = A01 / "data/alpha075_staged_load_pilot"


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        q = p if p.is_absolute() else USER_ROOT / p
        return str(q.absolute().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_env(out_dir: Path, device: str) -> None:
    cache = out_dir / "runtime_cache"
    mapping = {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }
    for key, path in mapping.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if device.startswith("cuda") and ":" in device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]


def load_one(model_dir: Path, device: str, text: str) -> dict[str, Any]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerFast
    import torch

    cfg = read_json(model_dir / "config.json")
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir.resolve()), trust_remote_code=True, local_files_only=True, padding_side="right")
    except Exception:
        tok = PreTrainedTokenizerFast.from_pretrained(str(model_dir.resolve()), padding_side="right")
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir.resolve()), trust_remote_code=True, local_files_only=True)
    dev = torch.device("cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()
    batch = tok(text, return_tensors="pt")
    batch = {k: v.to(dev) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits.detach().float().cpu()
    finite = bool(torch.isfinite(logits).all().item())
    probe = {
        "shape": list(logits.shape),
        "finite": finite,
        "mean": float(logits.mean().item()),
        "std": float(logits.std().item()),
        "max": float(logits.max().item()),
        "min": float(logits.min().item()),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model_dir": rel(model_dir),
        "loaded_class": cfg.get("architectures"),
        "auto_map": cfg.get("auto_map"),
        "adapter_scale": cfg.get("adapter_scale"),
        "private_adapter_scale": cfg.get("private_adapter_scale"),
        "private_adapter_enabled": cfg.get("private_adapter_enabled"),
        "vocab_size": cfg.get("vocab_size"),
        "probe_logits": probe,
    }


def compare_logits(a_dir: Path, b_dir: Path, device: str, text: str) -> dict[str, Any]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerFast
    import torch

    def load_tok(p: Path):
        try:
            return AutoTokenizer.from_pretrained(str(p.resolve()), trust_remote_code=True, local_files_only=True, padding_side="right")
        except Exception:
            return PreTrainedTokenizerFast.from_pretrained(str(p.resolve()), padding_side="right")

    dev = torch.device("cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    tok = load_tok(a_dir)
    batch = tok(text, return_tensors="pt")
    batch = {k: v.to(dev) for k, v in batch.items()}
    ma = AutoModelForMaskedLM.from_pretrained(str(a_dir.resolve()), trust_remote_code=True, local_files_only=True).to(dev).eval()
    mb = AutoModelForMaskedLM.from_pretrained(str(b_dir.resolve()), trust_remote_code=True, local_files_only=True).to(dev).eval()
    with torch.no_grad():
        la = ma(**batch).logits.detach().float().cpu()
        lb = mb(**batch).logits.detach().float().cpu()
    d = (la - lb).abs()
    out = {
        "a": rel(a_dir),
        "b": rel(b_dir),
        "max_abs_diff": float(d.max().item()),
        "mean_abs_diff": float(d.mean().item()),
        "rms_diff": float(torch.sqrt(torch.mean((la - lb) ** 2)).item()),
        "same_shape": list(la.shape) == list(lb.shape),
        "finite": bool(torch.isfinite(la).all().item() and torch.isfinite(lb).all().item()),
    }
    del ma, mb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default=str(DEFAULT_STAGE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cpu", help="cpu or cuda[:id]; use cpu while H100s are occupied")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_env(out_dir, args.device)
    t0 = time.time()

    stage = Path(args.stage_dir)
    manifest = read_json(stage / "alpha075_developmental_staging_manifest.json")
    hf = stage / "hf_model"
    text = "The child put the [MASK] on the table and smiled."
    revisions = ["chck_1M", "chck_80M", "chck_83M", "main"]
    loads = {}
    for rev in revisions:
        p = hf / rev
        if not (p / "model.safetensors").exists():
            loads[rev] = {"error": f"missing model.safetensors at {rel(p)}"}
            continue
        loads[rev] = load_one(p, args.device, text)

    alpha1_83 = USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/chck_total_83012495w"
    compare_83_alpha = compare_logits(alpha1_83, hf / "chck_83M", args.device, text)
    compare_final_alias = compare_logits(hf / "main", hf / "chck_86M", args.device, text)

    errors = []
    if manifest.get("missing_fast_revisions") != ["chck_90M", "chck_100M"]:
        errors.append(f"unexpected_missing_fast_revisions={manifest.get('missing_fast_revisions')}")
    for rev in revisions:
        rec = loads.get(rev, {})
        if rec.get("probe_logits", {}).get("finite") is not True:
            errors.append(f"nonfinite_or_failed_load_{rev}")
    if loads.get("chck_83M", {}).get("private_adapter_scale") != 0.75:
        errors.append("chck_83M_private_adapter_scale_not_0p75")
    if loads.get("main", {}).get("private_adapter_scale") != 0.75:
        errors.append("main_private_adapter_scale_not_0p75")
    if compare_final_alias.get("max_abs_diff") != 0.0:
        errors.append(f"main_chck86_alias_not_identical_{compare_final_alias.get('max_abs_diff')}")
    if not (compare_83_alpha.get("max_abs_diff", 0.0) > 0.0 and compare_83_alpha.get("finite")):
        errors.append("alpha1_vs_alpha0p75_83M_no_detectable_or_nonfinite_difference")

    payload = {
        "status": "ALPHA075_STAGE_LOAD_PILOT_PASS" if not errors else "ALPHA075_STAGE_LOAD_PILOT_FAIL",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage_manifest": rel(stage / "alpha075_developmental_staging_manifest.json"),
        "hf_model_root": rel(hf),
        "device_requested": args.device,
        "text": text,
        "loads": loads,
        "compare_alpha1_to_alpha0p75_chck83": compare_83_alpha,
        "compare_main_to_chck86_alias": compare_final_alias,
        "errors": errors,
        "elapsed_sec": time.time() - t0,
        "interpretation": "This pilot proves local loading and alpha=0.75 staging for existing truthful revisions only. It does not make missing chck_90M/chck_100M states or full AoA/fast results available.",
    }
    out_json = out_dir / "alpha075_staged_load_pilot.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note = out_dir / "alpha075_staged_load_pilot.md"
    note.write_text("\n".join([
        "# research alpha0.75 staged load pilot",
        "",
        f"Status: **{payload['status']}**",
        f"HF model root: `{payload['hf_model_root']}`",
        f"Errors: `{errors}`",
        f"alpha1 vs alpha0.75 chck83 max_abs_diff: `{compare_83_alpha.get('max_abs_diff')}`",
        f"main vs chck86 alias max_abs_diff: `{compare_final_alias.get('max_abs_diff')}`",
        "",
        f"JSON: `{rel(out_json)}`",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(out_json),
        "errors": errors,
        "alpha1_vs_alpha075_83M_max_abs_diff": compare_83_alpha.get("max_abs_diff"),
        "main_vs_chck86_max_abs_diff": compare_final_alias.get("max_abs_diff"),
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
