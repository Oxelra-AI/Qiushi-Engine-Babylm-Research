#!/usr/bin/env python3
"""Check O62065 and MS62065 endpoint identity before evaluation.

This configuration preflight does not load a model or launch evaluation.
Checks repaired AutoModel endpoint configuration and model-path availability.

O62065: ordinary-continuation endpoint identity and AutoModel metadata.
MS62065: acquisition-only endpoint identity and AutoModel metadata.

AoA extraction and assembly are separate follow-up stages.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, os, pathlib, subprocess, sys

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')

# Repaired endpoints
O62065_REPAIRED = _public_path('experiments/archive/relation_learning/data/repair_ordinary62065_bundle/repaired_ordinary62065_u0080')
MS62065_REPAIRED = _public_path('experiments/archive/relation_learning/data/repair_ms62065_bundle/repaired_ms62065_u0080')

# Verify existence
for label, path in [("O62065", O62065_REPAIRED), ("MS62065", MS62065_REPAIRED)]:
    model_file = path / "model.safetensors"
    config_file = path / "config.json"
    if not model_file.exists():
        raise FileNotFoundError(f"{label} model.safetensors not found at {model_file}")
    if not config_file.exists():
        raise FileNotFoundError(f"{label} config.json not found at {config_file}")
    c = json.loads(config_file.read_text())
    auto_map = c.get("auto_map", {})
    if "AutoModel" not in auto_map:
        raise ValueError(f"{label} config missing AutoModel in auto_map: {auto_map}")
    print(json.dumps({"event": "verified", "label": label, "path": str(path.relative_to(ROOT)),
                       "auto_map": auto_map, "model_size": model_file.stat().st_size}))

print(json.dumps({"event": "all_verified", "o62065": str(O62065_REPAIRED.relative_to(ROOT)),
                   "ms62065": str(MS62065_REPAIRED.relative_to(ROOT))}))
