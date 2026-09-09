#!/usr/bin/env python3
"""GlobalPIQA hard-rank wrapper for Muon/AdamW 50M artifacts.

This reuses the validated research all-option GlobalPIQA reader, adding the four
existing 50M checkpoints as targets.  Outputs are written only under
representation_and_objectives.  It is an evaluation-only comparison of existing weights, not training
or official-example tuning.
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
BASE_SCRIPT = A01_WS / "scripts/globalpiqa_margin_reader.py"
OUT_ROOT = A01_WS / "data/muon_switch_50m_globalpiqa_margin"
NOTE = A01_WS / "notes/muon_switch_50m_globalpiqa_margin.md"

TARGETS: dict[str, dict[str, Any]] = {
    "adamw50": {
        "label": "continuous AdamW legal compact reinvest 50M reference",
        "model_root": A02_WS / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_50M",
        "revision": "main",
        "official_parallel": None,
        "official_nonparallel": None,
    },
    "continuous_muon50": {
        "label": "continuous matched-decay Muon 50M reference",
        "model_root": A02_WS / "training/runs/muon_lr008_wd00125_seed43022_80M/hf_model/chck_50M",
        "revision": "main",
        "official_parallel": None,
        "official_nonparallel": None,
    },
    "muon20toadamw50": {
        "label": "Muon for 20M then AdamW recovery to 50M",
        "model_root": A02_WS / "training/runs/muon20toadamw_seed43022_80M/hf_model/chck_50M",
        "revision": "main",
        "official_parallel": None,
        "official_nonparallel": None,
    },
    "muon40toadamw50": {
        "label": "Muon for 40M then AdamW recovery to 50M",
        "model_root": A02_WS / "training/runs/muon40toadamw_seed43022_80M/hf_model/chck_50M",
        "revision": "main",
        "official_parallel": None,
        "official_nonparallel": None,
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("globalpiqa_for_muon_switch50", BASE_SCRIPT)
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
        root = Path(meta["model_root"])
        rec = {"model_root": str(root), "revision": meta["revision"], "exists": root.exists(), "has_config": (root / "config.json").exists(), "has_weights": (root / "model.safetensors").exists(), "errors": []}
        if not root.exists():
            rec["errors"].append("missing checkpoint directory")
        if not (root / "config.json").exists():
            rec["errors"].append("missing config.json")
        if not (root / "model.safetensors").exists():
            rec["errors"].append("missing model.safetensors")
        if rec["errors"]:
            out["status"] = "NOT_READY"
        out["targets"][t] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=sorted(TARGETS))
    ap.add_argument("--modes", nargs="+", default=["parallel"], choices=["parallel", "nonparallel"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--non_causal_batch_size", type=int, default=32)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--threads", type=int, default=12)
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
    mod.TARGETS.clear()
    mod.TARGETS.update(TARGETS)
    old_argv = sys.argv
    try:
        sys.argv = [str(BASE_SCRIPT), "--targets", *args.targets, "--modes", *args.modes,
                    "--batch_size", str(args.batch_size), "--non_causal_batch_size", str(args.non_causal_batch_size),
                    "--threads", str(args.threads)]
        if args.max_items is not None:
            sys.argv.extend(["--max_items", str(args.max_items)])
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
