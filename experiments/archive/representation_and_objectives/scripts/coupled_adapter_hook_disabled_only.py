#!/usr/bin/env python3
"""research efficient live-adapter decomposition using research live outputs.

Loads the completed research live readouts for MLM-only, coupled aligned, and
coupled shuffled.  Runs only the missing runtime-hook variants where the coupled
aligned/shuffled adapter output is multiplied by 0.0.  This avoids re-scoring
already established live checkpoints while answering whether hard-surface repair
is carried by live adapter output or by stock/trajectory displacement.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface')
SUMMARY = _public_path('experiments/archive/representation_and_objectives/data/coupled_shuffled_control_readout/coupled_shuffled_control_summary.json')


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

BASE = import_from(_public_path('experiments/archive/representation_and_objectives/scripts/coupled_adapter_hook_hardsurface.py'), "adapter_hook_base")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(p: Path | str | None) -> str | None:
    return BASE.rel(p)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def set_caches(out_root: Path) -> None:
    for d in [out_root, out_root / "home", out_root / "xdg_cache", out_root / "hf_cache", out_root / "hf_cache/modules", out_root / "torch_cache", out_root / "tmp"]:
        d.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str((out_root / "home").resolve())
    os.environ["XDG_CACHE_HOME"] = str((out_root / "xdg_cache").resolve())
    os.environ["HF_HOME"] = str((out_root / "hf_cache").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((out_root / "hf_cache").resolve())
    os.environ["HF_MODULES_CACHE"] = str((out_root / "hf_cache/modules").resolve())
    os.environ["TORCH_HOME"] = str((out_root / "torch_cache").resolve())
    os.environ["TMPDIR"] = str((out_root / "tmp").resolve())


def remap_live_from_step177(research: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    gp = {
        "mlm_live": research["globalpiqa"]["mlm_only_20M"],
        "aligned_live": research["globalpiqa"]["coupled_sparse20_aligned_20M"],
        "shuffled_live": research["globalpiqa"]["coupled_sparse20_shuffled_20M"],
    }
    ew = {
        "mlm_live": research["ewok_stable_subset"]["mlm_only_20M"],
        "aligned_live": research["ewok_stable_subset"]["coupled_sparse20_aligned_20M"],
        "shuffled_live": research["ewok_stable_subset"]["coupled_sparse20_shuffled_20M"],
    }
    variants: dict[str, Any] = {}
    source_map = {
        "mlm_live": "mlm_only_20M",
        "aligned_live": "coupled_sparse20_aligned_20M",
        "shuffled_live": "coupled_sparse20_shuffled_20M",
    }
    for variant, src in source_map.items():
        st = research["targets"][src]
        variants[variant] = {
            "target": src,
            "label": st.get("label"),
            "family": st.get("family"),
            "model_path": st.get("model_path"),
            "adapter_output_scale": 1.0,
            "training_metrics": st.get("training_metrics"),
            "adapter_hook": "not rerun; live output loaded from research readout",
        }
    return gp, ew, variants


def main() -> None:
    set_caches(OUT_ROOT)
    research = read_json(SUMMARY)
    gp_results, ewok_results, variants = remap_live_from_step177(research)

    # Run only the missing disabled variants. Target keys refer to BASE.TARGETS.
    for variant_name, target_name in [("aligned_disabled", "coupled_aligned_20M"), ("shuffled_disabled", "coupled_shuffled_20M")]:
        meta = BASE.TARGETS[target_name]
        scale = 0.0
        metrics = BASE.read_json(Path(meta["metrics_path"]))
        variants[variant_name] = {
            "target": target_name,
            "label": meta["label"],
            "family": meta["family"],
            "model_path": rel(meta["model_path"]),
            "adapter_output_scale": scale,
            "training_metrics": metrics,
            "adapter_hook": None,
        }
        gp, gp_hook = BASE.run_gp_variant(target_name + "__" + variant_name, meta, scale, _public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/globalpiqa'), gp_max_items=None, threads=24)
        ew, ew_hook = BASE.run_ewok_variant(target_name + "__" + variant_name, meta, scale, _public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/ewok_stable_subset'), max_rows=None, threads=24, row_batch_size=64, masked_batch_size=160)
        gp_results[variant_name] = gp
        ewok_results[variant_name] = ew
        variants[variant_name]["adapter_hook"] = {"globalpiqa": gp_hook, "ewok": ew_hook}

    def pair(label: str, base: str, cand: str) -> tuple[str, dict[str, Any]]:
        return label, {"globalpiqa": BASE.gp_delta(gp_results[base], gp_results[cand]), "ewok": BASE.ewok_delta(ewok_results[base], ewok_results[cand])}

    deltas = dict([
        pair("aligned_live_minus_disabled", "aligned_disabled", "aligned_live"),
        pair("shuffled_live_minus_disabled", "shuffled_disabled", "shuffled_live"),
        pair("aligned_disabled_minus_mlm", "mlm_live", "aligned_disabled"),
        pair("shuffled_disabled_minus_mlm", "mlm_live", "shuffled_disabled"),
        pair("aligned_live_minus_mlm", "mlm_live", "aligned_live"),
        pair("shuffled_live_minus_mlm", "mlm_live", "shuffled_live"),
        pair("aligned_live_minus_shuffled_live", "shuffled_live", "aligned_live"),
        pair("aligned_disabled_minus_shuffled_disabled", "shuffled_disabled", "aligned_disabled"),
    ])

    _, ew_meta = BASE.load_stable_indices(max_rows=None)
    summary: dict[str, Any] = {
        "status": "COUPLED_ADAPTER_HOOK_HARDSURFACE_DONE",
        "created_utc": now_utc(),
        "boundary": "existing-checkpoint runtime adapter-output hook; no training; live outputs reused from research; research config adapter_scale is inert; chck_82M untouched",
        "source_live_readout": rel(SUMMARY),
        "variant_order": ["mlm_live", "aligned_live", "aligned_disabled", "shuffled_live", "shuffled_disabled"],
        "variants": variants,
        "globalpiqa_fixed_hard_set": research.get("globalpiqa_fixed_hard_set"),
        "ewok_stable_subset_definition": ew_meta,
        "globalpiqa": gp_results,
        "ewok_stable_subset": ewok_results,
        "deltas": deltas,
        "interpretation": [],
        "summary_json": str(_public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/coupled_adapter_hook_hardsurface_summary.json')),
    }
    summary["interpretation"] = BASE.interpret(summary)
    out_json = _public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/coupled_adapter_hook_hardsurface_summary.json')
    summary["summary_json"] = str(out_json)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    BASE.make_note(summary, _public_path('research/documents/representation_and_objectives/data/coupled_adapter_hook_hardsurface/coupled_adapter_hook_hardsurface_summary.md'))
    print(json.dumps({"status": summary["status"], "summary_json": rel(out_json), "summary_md": rel(_public_path('research/documents/representation_and_objectives/data/coupled_adapter_hook_hardsurface/coupled_adapter_hook_hardsurface_summary.md'))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
