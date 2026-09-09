#!/usr/bin/env python3
"""EWoK four-cell interaction reader for scale-1.75 80M.

This reuses the validated research/research EWoK interaction implementation but
patches targets to compare the adapter scale=1.75 80M checkpoint against
the legal40k fixed-256 80M anchor.  It performs no training.  The purpose
is to measure whether the first observed GlobalPIQA hard-margin improvement of
scale1.75 is accompanied by repair or damage on EWoK stable conditional
reversals.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Keep HF/transformers cache inside the local workspace; custom model code may
# need a writable cache even when the command itself is otherwise read-only.
USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
# Override rather than setdefault: the managed runtime may predefine read-only
# global HF caches, and transformers dynamic-module loading must be writable for
# the adapter checkpoint's custom code.
WRITABLE_HF_CACHE = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_ewok_hf_cache')
os.environ["HF_HOME"] = str(WRITABLE_HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(WRITABLE_HF_CACHE)
os.environ["HF_MODULES_CACHE"] = str(_public_path('experiments/archive/representation_and_objectives/data/scale1p75_ewok_hf_cache/modules'))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fw_ewok_interaction_reader as ew  # noqa: E402

OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_ewok_interaction')
NOTE = _public_path('research/notes/representation_and_objectives/scale1p75_ewok_interaction.md')

TARGETS: dict[str, dict[str, Any]] = {
    "scale1p75_80M": {
        "label": "A02 adapter128 scale1.75 80M checkpoint",
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M'),
    },
    "anchor_legal40k_80M": {
        "label": "A01 legal40k fixed-256 compact-view anchor chck_80M",
        "model_path": _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M'),
    },
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def delta(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ["accuracy", "wrong", "stable_failure", "stable_failure_frac_all", "within_both_positive_wrong_frac"]:
        if isinstance(base.get(key), (int, float)) and isinstance(other.get(key), (int, float)):
            out[key] = other[key] - base[key]
    if isinstance(base.get("interaction_sum_all"), dict) and isinstance(other.get("interaction_sum_all"), dict):
        out["interaction_sum_all_mean"] = other["interaction_sum_all"].get("mean", 0) - base["interaction_sum_all"].get("mean", 0)
    if isinstance(base.get("interaction_sum_wrong"), dict) and isinstance(other.get("interaction_sum_wrong"), dict):
        out["interaction_sum_wrong_mean"] = other["interaction_sum_wrong"].get("mean", 0) - base["interaction_sum_wrong"].get("mean", 0)
    return out


def keep_summary(res: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in res.items() if k not in {"records", "by_domain", "by_context_diff"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=sorted(TARGETS))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--row_limit", type=int, default=0)
    ap.add_argument("--row_offset", type=int, default=0)
    ap.add_argument("--row_batch_size", type=int, default=32)
    ap.add_argument("--masked_batch_size", type=int, default=128)
    ap.add_argument("--ready-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selected_targets = {k: TARGETS[k] for k in args.targets}
    readiness = {
        k: {
            "model_path": str(v["model_path"]),
            "model_ready": (Path(v["model_path"]) / "model.safetensors").exists(),
            "config_ready": (Path(v["model_path"]) / "config.json").exists(),
        }
        for k, v in selected_targets.items()
    }
    preflight_path = out_root / "preflight.json"
    preflight_path.write_text(json.dumps({"created_utc": now_utc(), "targets": readiness}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "preflight", "ready": readiness, "preflight": str(preflight_path)}, indent=2), flush=True)
    if args.ready_only:
        return
    missing = [k for k, r in readiness.items() if not (r["model_ready"] and r["config_ready"])]
    if missing:
        raise RuntimeError({"missing": missing, "readiness": readiness})

    ew.OUT_ROOT = out_root
    ew.NOTE = NOTE
    ew.DEFAULT_TARGETS = selected_targets
    pf = ew.preflight(args.targets)
    (out_root / "ewok_preflight.json").write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if pf.get("status") != "READY":
        raise RuntimeError({"ewok_preflight": pf})

    import torch
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    all_summary: dict[str, Any] = {
        "status": "SCALE1P75_EWOK_INTERACTION_DONE",
        "created_utc": now_utc(),
        "device": str(device),
        "row_limit": args.row_limit,
        "row_offset": args.row_offset,
        "targets": {},
        "deltas_vs_anchor_legal40k_80M": {},
    }
    raw: dict[str, Any] = {}
    for t in args.targets:
        meta = selected_targets[t]
        print(json.dumps({"event": "target_start", "target": t, "device": str(device), "utc": now_utc()}), flush=True)
        res = ew.run_target(t, Path(meta["model_path"]), device, args.threads, args.row_limit, args.row_offset, args.row_batch_size, args.masked_batch_size)
        tdir = out_root / t
        tdir.mkdir(parents=True, exist_ok=True)
        records = res.pop("records")
        ew.write_csv(tdir / "ewok_interaction_records.csv", records)
        ew.write_csv(tdir / "ewok_interaction_by_domain.csv", res["by_domain"])
        ew.write_csv(tdir / "ewok_interaction_by_context_diff.csv", res["by_context_diff"])
        (tdir / "ewok_interaction_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raw[t] = res
        all_summary["targets"][t] = keep_summary(res)
        print(json.dumps({"event": "target_done", "target": t, "summary": res["summary"], "summary_path": str(tdir / "ewok_interaction_summary.json")}, indent=2), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "anchor_legal40k_80M" in raw:
        base = raw["anchor_legal40k_80M"]["summary"]
        for t, res in raw.items():
            if t != "anchor_legal40k_80M":
                all_summary["deltas_vs_anchor_legal40k_80M"][t] = delta(base, res["summary"])

    combined = out_root / "scale1p75_ewok_summary.json"
    combined.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": all_summary["status"], "combined_summary": str(combined)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
