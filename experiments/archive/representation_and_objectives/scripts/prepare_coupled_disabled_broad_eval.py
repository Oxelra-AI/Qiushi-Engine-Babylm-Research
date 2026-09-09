#!/usr/bin/env python3
"""research: prepare inference-disabled proxy runs for coupled sparse20 broad eval.

Creates HF-style proxy run directories with config.adapter_enabled=false and all
weights/tokenizer/code symlinked to the existing coupled aligned/shuffled 20M
checkpoints.  This is for zero-training official-compatible cheap7 evaluation of
whether the hard-surface repair that survives runtime adapter-output scale=0 also
recovers broad BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading.

`adapter_enabled` is active in research modeling; `adapter_scale` is not used.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
import shutil
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/coupled_disabled_broad_eval_prep')

SOURCES = {
    "coupled_aligned_disabled20M_proxy": {
        "source_run": _public_path('experiments/archive/frontier_consolidation/training/runs/coupled_sparse20_aligned_20M_seed43022'),
        "source_ckpt": _public_path('experiments/archive/frontier_consolidation/training/runs/coupled_sparse20_aligned_20M_seed43022/hf_model/final'),
        "dest_run": _public_path('experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_aligned_disabled_20M_seed43022'),
        "mode": "aligned",
    },
    "coupled_shuffled_disabled20M_proxy": {
        "source_run": _public_path('experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_20M_seed43022'),
        "source_ckpt": _public_path('experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_20M_seed43022/hf_model/final'),
        "dest_run": _public_path('experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_disabled_20M_seed43022'),
        "mode": "shuffled",
    },
}
REQUIRED = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "adapter_modeling.py"]


def rel(p: Path | str | None) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(pp)


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src.resolve())
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_proxy(target: str, spec: dict[str, Any]) -> dict[str, Any]:
    src = Path(spec["source_ckpt"])
    source_run = Path(spec["source_run"])
    dest_run = Path(spec["dest_run"])
    dest_ckpt = dest_run / "hf_model" / "final"
    missing = [name for name in REQUIRED if not (src / name).exists()]
    if missing:
        raise FileNotFoundError({"source_ckpt": rel(src), "missing": missing})
    dest_ckpt.mkdir(parents=True, exist_ok=True)
    cfg = read_json(src / "config.json")
    if cfg.get("architectures") != ["AdapterDebertaV2ForMaskedLM"]:
        raise RuntimeError({"source_ckpt": rel(src), "unexpected_architectures": cfg.get("architectures")})
    if cfg.get("auto_map", {}).get("AutoModelForMaskedLM") != "adapter_modeling.AdapterDebertaV2ForMaskedLM":
        raise RuntimeError({"source_ckpt": rel(src), "unexpected_auto_map": cfg.get("auto_map")})
    cfg["adapter_enabled"] = False
    cfg["inference_ablation"] = {
        "source_checkpoint": rel(src),
        "changed_config_field": "adapter_enabled true->false",
        "adapter_scale_warning": "research sparse20 modeling ignores config.adapter_scale; disabling uses adapter.enabled branch, which is active.",
        "scientific_purpose": "zero-training broad check: does coupled-trajectory hard-surface repair survive while broad damage is removed when live adapter output is disabled?",
    }
    (dest_ckpt / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    linked = {}
    for name in REQUIRED:
        if name == "config.json":
            continue
        linked[name] = link_or_copy(src / name, dest_ckpt / name)
    src_metrics = read_json(source_run / "scientific_metrics.json")
    metrics = dict(src_metrics)
    metrics.update({
        "status": "INFERENCE_DISABLED_PROXY",
        "source_status": src_metrics.get("status"),
        "source_mode": src_metrics.get("mode"),
        "mode": f"{src_metrics.get('mode')}_adapter_disabled_inference",
        "source_run": rel(source_run),
        "source_checkpoint": rel(src),
        "inference_only_change": "config.adapter_enabled=false; weights/tokenizer/modeling symlinked unchanged",
        "adapter_scale_warning": "config.adapter_scale is inert in adapter_modeling.py; this proxy disables adapter.enabled, not scaling",
    })
    (dest_run / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "status": "DISABLED_PROXY_READY",
        "target": target,
        "mode": spec["mode"],
        "source_run": rel(source_run),
        "source_ckpt": rel(src),
        "dest_run": rel(dest_run),
        "dest_ckpt": rel(dest_ckpt),
        "config_adapter_enabled": False,
        "linked_or_copied": linked,
        "source_training_metrics": src_metrics,
        "proxy_training_metrics": metrics,
    }
    out = dest_run / "disabled_proxy_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {target: make_proxy(target, spec) for target, spec in SOURCES.items()}
    eval_commands = {}
    for target, spec in SOURCES.items():
        eval_commands[target] = [
            "python3", "-B", "experiments/archive/frontier_consolidation/scripts/dualview_eval_one.py",
            "--run-dir", rel(Path(spec["dest_run"])),
            "--target", target,
            "--endpoint", "final",
            "--out-root", "experiments/archive/representation_and_objectives/data/coupled_disabled_broad_eval/official_outputs",
            "--collate-root", "experiments/archive/representation_and_objectives/data/coupled_disabled_broad_eval/collate",
            "--summary-root", "experiments/archive/representation_and_objectives/data/coupled_disabled_broad_eval/summary",
            "--gpu", "<0-or-1>",
            "--force",
        ]
    summary = {
        "status": "COUPLED_DISABLED_BROAD_EVAL_PROXIES_READY",
        "boundary": "zero-training proxy preparation; no model weights changed; no chck_82M access",
        "proxies": results,
        "planned_eval_commands": eval_commands,
        "decision": "Evaluate cheap7 for aligned/shuffled disabled proxies. If disabled broad recovers while hard repair persists, coupled auxiliary acts as train-time trajectory shaping; if disabled broad remains collapsed, coupled sparse20 route should not be extended.",
    }
    out_json = _public_path('experiments/archive/representation_and_objectives/data/coupled_disabled_broad_eval_prep/coupled_disabled_broad_eval_prep.json')
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": rel(out_json), "targets": list(results)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
