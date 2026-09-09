#!/usr/bin/env python3
"""research: CPU GlobalPIQA hard-surface reader for scale1.75 100M.

This is not official score establishment.  It is a mechanism readout after the
100M endpoint exists, comparing the scale1.75 adapter endpoint against
(1) the matched legal16k base at 100M and (2) the legal40k
anchor at 100M.  The matched legal16k comparison is the attribution-safe one;
the legal40k target is retained only to connect with earlier hard-52
tracking.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
import pathlib
import sys
import time
import importlib.util
from typing import Any

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface')
HF_CACHE = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface/hf_cache')

# Make dynamic custom-code loading writable and isolated.
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HF_MODULES_CACHE"] = str(_public_path('experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface/hf_cache/modules'))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

TARGETS = {
    "scale1p75_100M": {
        "model_root": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M'),
        "revision": None,
        "label": "A02 adapter128 scale1.75 100M official-ladder endpoint",
        "family": "scale1p75_adapter_legal16k",
    },
    "legal16k_100M": {
        "model_root": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M'),
        "revision": None,
        "label": "A02 matched legal16k compact-view reinvest 100M base",
        "family": "matched_legal16k_base",
    },
    "a01_legal40k_100M": {
        "model_root": _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M'),
        "revision": None,
        "label": "A01 legal40k fixed-256 compact-view reinvest 100M anchor",
        "family": "a01_legal40k_anchor",
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def load_gp_module():
    spec = importlib.util.spec_from_file_location("gp", str(_public_path('experiments/archive/representation_and_objectives/scripts/globalpiqa_margin_reader.py')))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import globalpiqa_margin_reader")
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    return gp


def compact(res: dict[str, Any]) -> dict[str, Any]:
    out = {
        "target": res.get("target"),
        "label": res.get("label"),
        "model_root": res.get("model_root"),
        "family": res.get("family"),
        "modes": {},
    }
    for mode, payload in res.get("modes", {}).items():
        if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
            out["modes"][mode] = payload["summary"]
    return out


def delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for mode in ["parallel", "nonparallel"]:
        b = before.get("modes", {}).get(mode, {})
        a = after.get("modes", {}).get(mode, {})
        md: dict[str, Any] = {}
        for k in ["accuracy", "chance_adjusted_accuracy"]:
            if isinstance(b.get(k), (int, float)) and isinstance(a.get(k), (int, float)):
                md[k] = a[k] - b[k]
        bh = b.get("always_wrong_subset")
        ah = a.get("always_wrong_subset")
        if isinstance(bh, dict) and isinstance(ah, dict):
            for k in ["accuracy", "mean_top_minus_correct", "median_top_minus_correct"]:
                if isinstance(bh.get(k), (int, float)) and isinstance(ah.get(k), (int, float)):
                    md[f"hard52_{k}"] = ah[k] - bh[k]
            for k in ["rank1", "rank2", "rank3", "rank4"]:
                if isinstance(bh.get(k), (int, float)) and isinstance(ah.get(k), (int, float)):
                    md[f"hard52_{k}"] = ah[k] - bh[k]
        bm = b.get("all_rows_margin_summary")
        am = a.get("all_rows_margin_summary")
        if isinstance(bm, dict) and isinstance(am, dict):
            for k in ["mean_top_minus_correct", "median_top_minus_correct"]:
                if isinstance(bm.get(k), (int, float)) and isinstance(am.get(k), (int, float)):
                    md[f"all_{k}"] = am[k] - bm[k]
        d[mode] = md
    return d


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    readiness = {}
    for name, meta in TARGETS.items():
        root = pathlib.Path(meta["model_root"])
        readiness[name] = {
            "model_root": rel(root),
            "model_ready": (root / "model.safetensors").exists(),
            "config_ready": (root / "config.json").exists(),
        }
    preflight = {"status": "SCALE1P75_100M_GLOBALPIQA_PREFLIGHT", "created_utc": now(), "targets": readiness}
    (_public_path('experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface/preflight.json')).write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(preflight, ensure_ascii=False), flush=True)
    missing = [k for k, v in readiness.items() if not (v["model_ready"] and v["config_ready"])]
    if missing:
        raise FileNotFoundError(f"Missing ready targets: {missing}")

    gp = load_gp_module()
    gp.OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface/raw_gp_reader')
    gp.NOTE = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface/scale1p75_100m_globalpiqa_note.md')
    gp.TARGETS = TARGETS

    combined: dict[str, Any] = {
        "status": "SCALE1P75_100M_GLOBALPIQA_HARD_SURFACE",
        "created_utc": now(),
        "targets": {},
        "deltas": {},
        "attribution_warning": "Matched research legal16k comparison is mechanism-safe; A01 legal40k comparison crosses tokenizer/vocab/parameter lineage and is only for continuity with earlier hard-52 tracking.",
    }
    raw_rows = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface/raw_rows')
    raw_rows.mkdir(parents=True, exist_ok=True)
    for name in TARGETS:
        print(json.dumps({"event": "target_start", "target": name, "utc": now(), "model_root": rel(TARGETS[name]["model_root"])}), flush=True)
        res = gp.run_target(name, ["parallel", "nonparallel"], batch_size=8, non_causal_batch_size=32, max_items=None, threads=8)
        combined["targets"][name] = compact(res)
        rows_payload = {mode: payload.get("rows", []) for mode, payload in res.get("modes", {}).items()}
        (raw_rows / f"{name}_rows.json").write_text(json.dumps(rows_payload, ensure_ascii=False) + "\n", encoding="utf-8")
        par = combined["targets"][name].get("modes", {}).get("parallel", {})
        print(json.dumps({"event": "target_done", "target": name, "parallel_acc": par.get("accuracy"), "hard52": par.get("always_wrong_subset", {}).get("accuracy"), "hard52_mean_margin": par.get("always_wrong_subset", {}).get("mean_top_minus_correct")}), flush=True)

    base = combined["targets"]["legal16k_100M"]
    cand = combined["targets"]["scale1p75_100M"]
    a01 = combined["targets"]["a01_legal40k_100M"]
    combined["deltas"]["scale1p75_minus_step35_legal16k_100M"] = delta(base, cand)
    combined["deltas"]["scale1p75_minus_a01_legal40k_100M"] = delta(a01, cand)

    out = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface/scale1p75_100m_globalpiqa_summary.json')
    out.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": combined["status"], "summary": rel(out)}), flush=True)


if __name__ == "__main__":
    main()
