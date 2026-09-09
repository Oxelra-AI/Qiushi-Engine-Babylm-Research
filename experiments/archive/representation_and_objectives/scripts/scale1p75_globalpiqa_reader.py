#!/usr/bin/env python3
"""CPU-only GlobalPIQA hard-surface reader for scale-1.75 80M.

Reuses the research all-option scoring infrastructure with the adapter model.
Runs on CPU only (both GPUs are occupied).
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, sys, os, pathlib, importlib.util

# Set HF cache to writable location (custom code needs write access)
os.environ['HF_HOME'] = str(_public_path('experiments/archive/representation_and_objectives/data/scale1p75_globalpiqa_hard_surface/hf_cache'))
os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')

# Import research reader
spec = importlib.util.spec_from_file_location(
    "gp", str(_public_path('experiments/archive/representation_and_objectives/scripts/globalpiqa_margin_reader.py')))
gp = importlib.util.module_from_spec(spec)

# Patch TARGETS before executing the module
ADAPTER_80M = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M')
ANCHOR_80M = _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M')

OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_globalpiqa_hard_surface')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Execute the module to get access to its functions
spec.loader.exec_module(gp)

# Override TARGETS
gp.TARGETS = {
    "scale1p75_80M": {
        "model_root": ADAPTER_80M,
        "revision": None,
        "label": "A02 adapter128 scale1.75 80M endpoint",
        "family": "scale1p75_adapter",
    },
    "anchor_legal40k_80M": {
        "model_root": ANCHOR_80M,
        "revision": None,
        "label": "A01 legal40k fixed-256 compact-view anchor 80M",
        "family": "legal40k_fixed256",
    },
}
gp.OUT_ROOT = OUT_ROOT
gp.NOTE = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_globalpiqa_hard_surface/scale1p75_globalpiqa_note.md')

# Run with default args
class Args:
    targets = ["scale1p75_80M", "anchor_legal40k_80M"]
    modes = ["parallel", "nonparallel"]
    batch_size = 8
    non_causal_batch_size = 32
    max_items = None
    threads = 8
    ready_only = False

args = Args()

# Check readiness
for k, v in gp.TARGETS.items():
    mr = pathlib.Path(v["model_root"])
    ready = (mr / "model.safetensors").exists()
    print(json.dumps({"target": k, "model_root": str(mr), "ready": ready}), flush=True)

if args.ready_only:
    sys.exit(0)

# Score each target
combined = {"status": "SCALE1P75_GLOBALPIQA_HARD_SURFACE", "targets": {}}

for target in args.targets:
    print(json.dumps({"event": "scoring_target", "target": target}), flush=True)
    try:
        res = gp.run_target(target, args.modes, args.batch_size, args.non_causal_batch_size, args.max_items, args.threads)
        summary = {
            "label": res.get("label"),
            "model_root": res.get("model_root"),
            "modes": {}
        }
        for mode in args.modes:
            mode_data = res.get("modes", {}).get(mode, {})
            if "summary" in mode_data:
                summary["modes"][mode] = mode_data["summary"]
        combined["targets"][target] = summary
        print(json.dumps({"event": "target_done", "target": target, "modes": list(summary["modes"].keys())}), flush=True)
    except Exception as e:
        combined["targets"][target] = {"error": str(e)}
        print(json.dumps({"event": "target_error", "target": target, "error": str(e)}), flush=True)

# Save combined
combined_json = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_globalpiqa_hard_surface/scale1p75_globalpiqa_summary.json')
combined_json.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"status": combined["status"], "combined_json": str(combined_json)}), flush=True)
