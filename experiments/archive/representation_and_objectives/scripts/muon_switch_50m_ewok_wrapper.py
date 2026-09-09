#!/usr/bin/env python3
"""EWoK four-cell wrapper for Muon/AdamW 50M artifacts.

Reuses the research EWoK four-cell reader without copying scoring logic. The four
50M checkpoints all exist at the deepest common current exposure of the
switch artifacts. Outputs are written under representation_and_objectives only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
A01_WS = ROOT / "experiments/archive/representation_and_objectives"
A02_WS = ROOT / "experiments/archive/frontier_consolidation"
BASE_SCRIPT = A01_WS / "scripts/fw_ewok_interaction_reader.py"
OUT_ROOT = A01_WS / "data/muon_switch_50m_ewok_interaction"
NOTE = A01_WS / "notes/muon_switch_50m_ewok_interaction.md"

TARGETS: dict[str, dict[str, Any]] = {
    "adamw50": {
        "model_path": A02_WS / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_50M",
        "label": "continuous AdamW legal compact reinvest 50M reference",
    },
    "continuous_muon50": {
        "model_path": A02_WS / "training/runs/muon_lr008_wd00125_seed43022_80M/hf_model/chck_50M",
        "label": "continuous matched-decay Muon 50M reference",
    },
    "muon20toadamw50": {
        "model_path": A02_WS / "training/runs/muon20toadamw_seed43022_80M/hf_model/chck_50M",
        "label": "Muon for 20M then AdamW recovery to 50M",
    },
    "muon40toadamw50": {
        "model_path": A02_WS / "training/runs/muon40toadamw_seed43022_80M/hf_model/chck_50M",
        "label": "Muon for 40M then AdamW recovery to 50M",
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("ewok_for_muon_switch50", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def preflight(targets: list[str]) -> dict[str, Any]:
    out = {"status": "READY", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "targets": {}}
    for t in targets:
        meta = TARGETS[t]
        path = Path(meta["model_path"])
        rec = {"model_path": str(path), "exists": path.exists(), "has_config": (path / "config.json").exists(), "has_weights": (path / "model.safetensors").exists(), "errors": []}
        if not path.exists():
            rec["errors"].append("missing checkpoint directory")
        if not (path / "config.json").exists():
            rec["errors"].append("missing config.json")
        if not (path / "model.safetensors").exists():
            rec["errors"].append("missing model.safetensors")
        if rec["errors"]:
            out["status"] = "NOT_READY"
        out["targets"][t] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=sorted(TARGETS))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--row_limit", type=int, default=0)
    ap.add_argument("--row_offset", type=int, default=0)
    ap.add_argument("--row_batch_size", type=int, default=64)
    ap.add_argument("--masked_batch_size", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pf = preflight(args.targets)
    pf_path = OUT_ROOT / "preflight.json"
    pf_path.write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "ready": pf["status"], "preflight": str(pf_path)}, indent=2), flush=True)
        return
    bad = {k: v["errors"] for k, v in pf["targets"].items() if v["errors"]}
    if bad:
        raise RuntimeError({"not_ready": bad, "preflight": str(pf_path)})

    mod = load_base_module()
    mod.OUT_ROOT = OUT_ROOT
    mod.NOTE = NOTE
    mod.DEFAULT_TARGETS.clear()
    mod.DEFAULT_TARGETS.update(TARGETS)
    old_argv = sys.argv
    try:
        sys.argv = [str(BASE_SCRIPT), "--targets", *args.targets, "--device", args.device,
                    "--threads", str(args.threads), "--row_limit", str(args.row_limit),
                    "--row_offset", str(args.row_offset), "--row_batch_size", str(args.row_batch_size),
                    "--masked_batch_size", str(args.masked_batch_size)]
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
