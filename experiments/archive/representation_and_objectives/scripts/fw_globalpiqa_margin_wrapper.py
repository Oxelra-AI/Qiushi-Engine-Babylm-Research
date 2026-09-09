#!/usr/bin/env python3
"""research: FW shared-anchor GlobalPIQA margin wrapper.

This reuses the validated research CPU GlobalPIQA option-margin reader but adds the
three running FineWeb shared-anchor endpoints.  It is a post-endpoint scientific
readout: after compact/breadth training and official-compatible evaluation, it
asks whether any arm improves correct-option ranks and margins on the hard
GlobalPIQA_parallel rows, rather than only changing the aggregate label count.

It does not tune a submission scorer and does not use official examples to shape
training data.
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
OUT_ROOT = A01_WS / "data/fw_globalpiqa_margin_reader"
NOTE = A01_WS / "notes/fw_globalpiqa_margin_reader.md"

FW_TARGETS: dict[str, dict[str, Any]] = {
    "fw_compact_fullbatch_seed43022": {
        "label": "A02 compact same-proposition recurrence anchor, shared16k full-batch seed43022",
        "model_root": A02_WS / "training/runs/fw_compact_view_shared16k_seed43022/hf_model",
        "revision": "chck_100M",
        "official_parallel": None,
        "official_nonparallel": None,
    },
    "fw_breadth_rowblock_fullbatch_seed43022": {
        "label": "A02 row-block whole-sentence source-breadth comparator, shared16k full-batch seed43022",
        "model_root": A02_WS / "training/runs/fw_source_breadth_shared16k_seed43022/hf_model",
        "revision": "chck_100M",
        "official_parallel": None,
        "official_nonparallel": None,
    },
    "fw_breadth_interleaved_fullbatch_seed43022": {
        "label": "A01 interleaved whole-sentence source-breadth comparator, shared16k full-batch seed43022",
        "model_root": A01_WS / "training/runs/fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022/hf_model",
        "revision": "chck_100M",
        "official_parallel": None,
        "official_nonparallel": None,
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("globalpiqa_margin_reader", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def preflight(targets: list[str]) -> dict[str, Any]:
    out = {"status": "READY", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "targets": {}}
    for t in targets:
        meta = FW_TARGETS[t]
        root = Path(meta["model_root"])
        chck = root / meta["revision"]
        rec = {
            "model_root": str(root),
            "revision": meta["revision"],
            "hf_model_exists": root.exists(),
            "checkpoint_exists": chck.exists(),
            "errors": [],
        }
        if not root.exists():
            rec["errors"].append("missing hf_model")
        if not chck.exists():
            rec["errors"].append("missing chck_100M")
        if rec["errors"]:
            out["status"] = "NOT_READY"
        out["targets"][t] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(FW_TARGETS), choices=sorted(FW_TARGETS))
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
    # Redirect outputs and add FW target configs before calling the already-validated runner.
    mod.OUT_ROOT = OUT_ROOT
    mod.NOTE = NOTE
    mod.TARGETS.update(FW_TARGETS)
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
