#!/usr/bin/env python3
"""GlobalPIQA hard-surface inference ablation for scale1.75 80M.

No training.  It creates temporary proxy checkpoint directories that symlink the
same 80M adapter model files while patching config.json only for inference:
  - live scale1.75 (original)
  - adapter disabled
  - adapter scale 0.5 / 1.0 / 2.5
and scores GlobalPIQA_parallel/nonparallel with the validated research reader.

This separates direct residual-branch contribution from the trained backbone
trajectory on the known hard GlobalPIQA surface.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any
import importlib.util

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_inference_ablation')
PROXY_ROOT = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_inference_ablation/proxy_models')
HF_CACHE = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_inference_ablation/hf_cache')

# Force writable dynamic-module caches.
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HF_MODULES_CACHE"] = str(_public_path('experiments/archive/representation_and_objectives/data/scale1p75_inference_ablation/hf_cache/modules'))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ADAPTER_80M = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M')
ANCHOR_80M = _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M')

VARIANTS = {
    "scale1p75_live": {"src": ADAPTER_80M, "patch": {}, "label": "A02 scale1.75 live adapter 80M"},
    "scale1p75_disabled": {"src": ADAPTER_80M, "patch": {"adapter_enabled": False}, "label": "same weights, adapter branch disabled at inference"},
    "scale1p75_scale0p5": {"src": ADAPTER_80M, "patch": {"adapter_enabled": True, "adapter_scale": 0.5}, "label": "same weights, adapter scale 0.5 at inference"},
    "scale1p75_scale1p0": {"src": ADAPTER_80M, "patch": {"adapter_enabled": True, "adapter_scale": 1.0}, "label": "same weights, adapter scale 1.0 at inference"},
    "scale1p75_scale2p5": {"src": ADAPTER_80M, "patch": {"adapter_enabled": True, "adapter_scale": 2.5}, "label": "same weights, adapter scale 2.5 at inference"},
    "anchor_legal40k_80M": {"src": ANCHOR_80M, "patch": {}, "label": "A01 fixed-256 legal40k 80M anchor"},
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def make_proxy(name: str, src: Path, patch: dict[str, Any]) -> Path:
    dst = PROXY_ROOT / name
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.name == "config.json":
            cfg = json.loads(item.read_text(encoding="utf-8"))
            cfg.update(patch)
            target.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        elif item.is_file():
            os.symlink(item.resolve(), target)
        elif item.is_dir():
            os.symlink(item.resolve(), target, target_is_directory=True)
    return dst


def load_gp_module():
    spec = importlib.util.spec_from_file_location("gp", str(_public_path('experiments/archive/representation_and_objectives/scripts/globalpiqa_margin_reader.py')))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import research reader")
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    return gp


def extract_compact_summary(res: dict[str, Any]) -> dict[str, Any]:
    out = {"target": res.get("target"), "label": res.get("label"), "model_root": res.get("model_root"), "modes": {}}
    for mode, payload in res.get("modes", {}).items():
        if isinstance(payload, dict) and "summary" in payload:
            out["modes"][mode] = payload["summary"]
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROXY_ROOT.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    proxies: dict[str, Path] = {}
    for name, v in VARIANTS.items():
        src = Path(v["src"])
        if not (src / "model.safetensors").exists():
            raise FileNotFoundError(src / "model.safetensors")
        proxies[name] = make_proxy(name, src, v["patch"])

    gp = load_gp_module()
    gp.OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_inference_ablation/gp_reader_raw')
    gp.NOTE = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_inference_ablation/scale1p75_inference_ablation_gp_note.md')
    gp.TARGETS = {
        name: {
            "model_root": path,
            "revision": None,
            "label": VARIANTS[name]["label"],
            "family": "scale1p75_ablation" if name.startswith("scale1p75") else "anchor",
        }
        for name, path in proxies.items()
    }

    combined: dict[str, Any] = {
        "status": "SCALE1P75_INFERENCE_ABLATION_DONE",
        "created_utc": now(),
        "source_checkpoint": rel(ADAPTER_80M),
        "anchor_checkpoint": rel(ANCHOR_80M),
        "variants": {},
        "deltas_vs_live": {},
        "deltas_vs_anchor": {},
    }
    raw_rows_dir = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_inference_ablation/raw_rows')
    raw_rows_dir.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        print(json.dumps({"event": "score_start", "variant": name, "model_root": rel(proxies[name]), "utc": now()}), flush=True)
        res = gp.run_target(name, ["parallel", "nonparallel"], batch_size=8, non_causal_batch_size=16, max_items=None, threads=8)
        compact = extract_compact_summary(res)
        combined["variants"][name] = compact
        # Save rows separately for later fine-grained analysis without inflating combined JSON.
        rows_payload = {mode: payload.get("rows", []) for mode, payload in res.get("modes", {}).items()}
        (raw_rows_dir / f"{name}_rows.json").write_text(json.dumps(rows_payload, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"event": "score_done", "variant": name, "parallel_acc": compact["modes"].get("parallel", {}).get("accuracy"), "hard52": compact["modes"].get("parallel", {}).get("always_wrong_subset", {}).get("accuracy")}), flush=True)

    def dmetric(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for mode in ["parallel", "nonparallel"]:
            ma = a.get("modes", {}).get(mode, {})
            mb = b.get("modes", {}).get(mode, {})
            od = {}
            for k in ["accuracy", "chance_adjusted_accuracy"]:
                if isinstance(ma.get(k), (int, float)) and isinstance(mb.get(k), (int, float)):
                    od[k] = mb[k] - ma[k]
            sa = ma.get("always_wrong_subset")
            sb = mb.get("always_wrong_subset")
            if isinstance(sa, dict) and isinstance(sb, dict):
                od["hard52_accuracy"] = sb.get("accuracy", 0) - sa.get("accuracy", 0)
                od["hard52_mean_top_minus_correct"] = sb.get("mean_top_minus_correct", 0) - sa.get("mean_top_minus_correct", 0)
                od["hard52_median_top_minus_correct"] = sb.get("median_top_minus_correct", 0) - sa.get("median_top_minus_correct", 0)
            aa = ma.get("all_rows_margin_summary")
            bb = mb.get("all_rows_margin_summary")
            if isinstance(aa, dict) and isinstance(bb, dict):
                od["all_mean_top_minus_correct"] = bb.get("mean_top_minus_correct", 0) - aa.get("mean_top_minus_correct", 0)
            out[mode] = od
        return out

    live = combined["variants"]["scale1p75_live"]
    anchor = combined["variants"]["anchor_legal40k_80M"]
    for name, val in combined["variants"].items():
        if name != "scale1p75_live":
            combined["deltas_vs_live"][name] = dmetric(live, val)
        if name != "anchor_legal40k_80M":
            combined["deltas_vs_anchor"][name] = dmetric(anchor, val)

    out_json = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_inference_ablation/scale1p75_inference_ablation_summary.json')
    out_json.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": combined["status"], "summary": rel(out_json), "variants": list(combined["variants"])}), flush=True)


if __name__ == "__main__":
    main()
