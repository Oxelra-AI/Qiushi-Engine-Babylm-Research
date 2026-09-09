#!/usr/bin/env python3
"""All-option GlobalPIQA margin reader for FW 70M cheap endpoints.

This wraps the validated CPU margin reader and points it at the two
70M checkpoints whose official cheap predictions are already complete. It is a
scientific readout only: determine whether the FW compact/breadth family changes
correct-option ranks or top-minus-correct margins on the research hard
GlobalPIQA_parallel rows. It must not be used to tune an official-example scorer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
A02_WS = USER_ROOT / "experiments/archive/frontier_consolidation"
BASE_SCRIPT = A01_WS / "scripts/globalpiqa_margin_reader.py"
OUT_ROOT = A01_WS / "data/fw_70m_globalpiqa_margin_reader"
NOTE = A01_WS / "notes/fw_70m_globalpiqa_margin_reader.md"

TARGETS: dict[str, dict[str, Any]] = {
    "a02_fw_compact_70M_seed43022": {
        "label": "A02 compact same-proposition recurrence, shared16k full-batch seed43022, checkpoint 70M",
        "model_root": A02_WS / "training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_70M",
        "revision": "main",
        "official_parallel": 26.21,
        "official_nonparallel": 51.0,
    },
    "a02_fw_breadth_rowblock_70M_seed43022": {
        "label": "A02 row-block whole-sentence source-breadth comparator, shared16k full-batch seed43022, checkpoint 70M",
        "model_root": A02_WS / "training/runs/fw_source_breadth_shared16k_seed43022/hf_model/chck_70M",
        "revision": "main",
        "official_parallel": 26.21,
        "official_nonparallel": 47.0,
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("globalpiqa_margin_reader_for_70m", BASE_SCRIPT)
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
        chck = root / meta["revision"]
        rec = {"model_root": str(root), "revision": meta["revision"], "hf_model_exists": root.exists(), "checkpoint_exists": chck.exists(), "errors": []}
        if not root.exists():
            rec["errors"].append("missing hf_model")
        if not (root / "config.json").exists():
            rec["errors"].append("missing direct checkpoint config.json")
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
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pf = preflight(args.targets)
    pf_path = OUT_ROOT / "preflight.json"
    pf_path.write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "ready": pf["status"], "preflight": str(pf_path)}, indent=2), flush=True)
        return
    not_ready = {t: r["errors"] for t, r in pf["targets"].items() if r["errors"]}
    if not_ready:
        raise RuntimeError({"not_ready": not_ready, "preflight": str(pf_path)})

    mod = load_base_module()
    mod.OUT_ROOT = OUT_ROOT
    mod.NOTE = NOTE
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
