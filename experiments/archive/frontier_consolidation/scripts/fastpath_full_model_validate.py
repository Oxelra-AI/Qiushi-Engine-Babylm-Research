#!/usr/bin/env python3
"""Static validation for full research fast-path replay endpoints."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
import time
from pathlib import Path

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/fastpath_full_model_validation')
_PREIMPORT_CACHE = _public_path('experiments/archive/frontier_consolidation/data/fastpath_full_model_validation/preimport_hf_cache')
(_public_path('experiments/archive/frontier_consolidation/data/fastpath_full_model_validation/preimport_hf_cache/modules')).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_PREIMPORT_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(_PREIMPORT_CACHE)
os.environ["HF_MODULES_CACHE"] = str(_public_path('experiments/archive/frontier_consolidation/data/fastpath_full_model_validation/preimport_hf_cache/modules'))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from safetensors.torch import load_file
from transformers import AutoModelForMaskedLM, AutoTokenizer

ENDPOINT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
RUNS = {
    "coherent": _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022'),
    "spanbreak": _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_spanbreak_seed43022'),
}
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/fastpath_full_model_validation')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/fastpath_full_model_validation/fastpath_full_model_validation.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/fastpath_full_model_validation/fastpath_full_model_validation.md')


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_one(name: str, run_dir: Path) -> dict:
    final_dir = run_dir / "hf_model/final"
    final_model = final_dir / "model.safetensors"
    if not final_model.exists():
        raise FileNotFoundError(final_model)
    metrics = read_json(run_dir / "scientific_metrics.json")
    endpoint_sd = load_file(str(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M/model.safetensors')), device="cpu")
    final_sd = load_file(str(final_model), device="cpu")
    max_nonprivate = 0.0
    nonprivate_params = 0
    missing_endpoint_keys = []
    for k, v in endpoint_sd.items():
        if k not in final_sd:
            missing_endpoint_keys.append(k)
            continue
        diff = float((final_sd[k] - v).abs().max().item())
        max_nonprivate = max(max_nonprivate, diff)
        nonprivate_params += int(v.numel())
    private_sq = 0.0
    private_n = 0
    private_absmax = 0.0
    private_key_count = 0
    for k, v in final_sd.items():
        if ".private_adapter." in k:
            vf = v.float()
            private_sq += float(vf.square().sum().item())
            private_n += int(v.numel())
            private_absmax = max(private_absmax, float(vf.abs().max().item()))
            private_key_count += 1
    private_rms = (private_sq / max(1, private_n)) ** 0.5

    cache = _public_path('experiments/archive/frontier_consolidation/data/fastpath_full_model_validation/hf_cache') / name
    for sub in ["modules", "hub", "datasets", "tmp"]:
        (cache / sub).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["TRANSFORMERS_CACHE"] = str(cache)
    # Keep HF_MODULES_CACHE at the pre-import writable location; Transformers caches
    # its dynamic-module root at import time in some versions.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = AutoTokenizer.from_pretrained(str(final_dir), trust_remote_code=True, local_files_only=True)
    m = AutoModelForMaskedLM.from_pretrained(str(final_dir), trust_remote_code=True, local_files_only=True)
    m_ref = AutoModelForMaskedLM.from_pretrained(str(ENDPOINT), trust_remote_code=True, local_files_only=True)
    m.eval(); m_ref.eval()
    text = "The teacher moved the toy from the box to the drawer and then asked where it was."
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        if hasattr(m, "set_private_enabled"):
            m.set_private_enabled(False)
        off = m(**enc).logits
        if hasattr(m_ref, "set_private_enabled"):
            m_ref.set_private_enabled(False)
        ref = m_ref(**enc).logits
        if hasattr(m, "set_private_enabled"):
            m.set_private_enabled(True)
        on = m(**enc).logits
    return {
        "name": name,
        "run_dir": rel(run_dir),
        "final_dir": rel(final_dir),
        "training_metrics": metrics,
        "nonprivate_params_checked": nonprivate_params,
        "missing_endpoint_keys": missing_endpoint_keys[:20],
        "max_nonprivate_diff_vs_chck82": max_nonprivate,
        "private_key_count": private_key_count,
        "private_params": private_n,
        "private_rms": private_rms,
        "private_absmax": private_absmax,
        "trusted_class": type(m).__name__,
        "trusted_total_params": sum(p.numel() for p in m.parameters()),
        "private_off_vs_chck82_logit_maxdiff": float((off - ref).abs().max().item()),
        "private_on_vs_off_logit_maxdiff_probe_text": float((on - off).abs().max().item()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {name: validate_one(name, run) for name, run in RUNS.items()}
    status = "PASS" if all(r["max_nonprivate_diff_vs_chck82"] == 0.0 and r["private_off_vs_chck82_logit_maxdiff"] == 0.0 and r["private_rms"] > 0 for r in results.values()) else "FAIL"
    out = {"status": status, "created_utc": now(), "results": results}
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research fast-path full model validation", "", f"Status: **{status}**", "", "| arm | total words | nonprivate max diff | private rms | private OFF maxdiff | private ON/OFF maxdiff |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        tm = r["training_metrics"]
        lines.append(f"| {name} | {tm.get('total_consumed_words')} | {r['max_nonprivate_diff_vs_chck82']} | {r['private_rms']:.8f} | {r['private_off_vs_chck82_logit_maxdiff']} | {r['private_on_vs_off_logit_maxdiff_probe_text']:.8f} |")
    lines += ["", f"JSON: `{rel(OUT_JSON)}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
