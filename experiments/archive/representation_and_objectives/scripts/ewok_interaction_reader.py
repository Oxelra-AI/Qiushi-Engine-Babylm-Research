#!/usr/bin/env python3
"""research: EWoK four-cell readout for shared-tokenizer role-switch screen.

Compares role_switch_80M against role_fixed_80M and the matched fixed-256 legal40k
80M anchor on stable conditional-reversal surfaces. No training is performed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fw_ewok_interaction_reader as ew  # noqa: E402

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
DEFAULT_OUT_ROOT = A01_WS / "data" / "ewok_interaction_reader"
DEFAULT_NOTE = A01_WS / "notes" / "147_ewok_interaction_reader.md"

TARGETS: dict[str, dict[str, Any]] = {
    "anchor_fixed256_80M": {
        "label": "Matched legal40k fixed-256 compact-view anchor at chck_80M",
        "model_path": A01_WS / "training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M",
    },
    "role_switch_80M": {
        "label": "Role-switch in-place packed shared-tokenizer 80M screen",
        "model_path": A01_WS / "training/runs/role_switch_sharedtok_80M_legal40k_seed43022/hf_model/chck_80M",
    },
    "role_fixed_80M": {
        "label": "Role-fixed in-place packed shared-tokenizer 80M screen control",
        "model_path": A01_WS / "training/runs/role_fixed_sharedtok_80M_legal40k_seed43022/hf_model/chck_80M",
    },
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Return b - a for key summary metrics."""
    out: dict[str, Any] = {}
    for key in ["accuracy", "stable_failure", "stable_failure_frac_all", "within_both_positive_wrong_frac"]:
        if isinstance(a.get(key), (int, float)) and isinstance(b.get(key), (int, float)):
            out[key] = b[key] - a[key]
    if isinstance(a.get("interaction_sum_all"), dict) and isinstance(b.get("interaction_sum_all"), dict):
        out["interaction_sum_all_mean"] = b["interaction_sum_all"].get("mean", 0) - a["interaction_sum_all"].get("mean", 0)
    if isinstance(a.get("interaction_sum_wrong"), dict) and isinstance(b.get("interaction_sum_wrong"), dict):
        out["interaction_sum_wrong_mean"] = b["interaction_sum_wrong"].get("mean", 0) - a["interaction_sum_wrong"].get("mean", 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=sorted(TARGETS))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--row_limit", type=int, default=0)
    ap.add_argument("--row_offset", type=int, default=0)
    ap.add_argument("--row_batch_size", type=int, default=64)
    ap.add_argument("--masked_batch_size", type=int, default=256)
    ap.add_argument("--ready-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    selected_targets = {k: TARGETS[k] for k in args.targets}
    readiness = {k: {"model_path": str(v["model_path"]), "ready": (Path(v["model_path"]) / "model.safetensors").exists()} for k, v in selected_targets.items()}
    (out_root / "preflight.json").write_text(json.dumps({"targets": readiness}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "preflight", "ready": readiness}, indent=2), flush=True)
    if args.ready_only:
        return
    missing = [k for k, v in readiness.items() if not v["ready"]]
    if missing:
        raise RuntimeError({"missing_checkpoints": missing, "readiness": readiness})

    ew.OUT_ROOT = out_root
    ew.NOTE = DEFAULT_NOTE
    ew.DEFAULT_TARGETS = selected_targets
    pf = ew.preflight(args.targets)
    (out_root / "ewok_preflight.json").write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if pf.get("status") != "READY":
        raise RuntimeError({"preflight": pf})

    import torch
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    all_summary: dict[str, Any] = {
        "status": "EWOK_INTERACTION_READER_DONE",
        "created_utc": now_utc(),
        "legal_status": "80M legal from-scratch screen readout, not final endpoint",
        "device": str(device),
        "targets": {},
        "deltas_vs_anchor_fixed256_80M": {},
    }
    raw: dict[str, Any] = {}
    for t in args.targets:
        meta = selected_targets[t]
        print(json.dumps({"event": "target_start", "target": t, "device": str(device)}), flush=True)
        res = ew.run_target(t, Path(meta["model_path"]), device, args.threads, args.row_limit, args.row_offset, args.row_batch_size, args.masked_batch_size)
        tdir = out_root / t
        tdir.mkdir(parents=True, exist_ok=True)
        records = res.pop("records")
        ew.write_csv(tdir / "ewok_interaction_records.csv", records)
        ew.write_csv(tdir / "ewok_interaction_by_domain.csv", res["by_domain"])
        ew.write_csv(tdir / "ewok_interaction_by_context_diff.csv", res["by_context_diff"])
        (tdir / "ewok_interaction_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raw[t] = res
        all_summary["targets"][t] = {k: v for k, v in res.items() if k not in {"by_domain", "by_context_diff"}}
        print(json.dumps({"event": "target_done", "target": t, "summary": res["summary"], "summary_path": str(tdir / "ewok_interaction_summary.json")}, indent=2), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "anchor_fixed256_80M" in raw:
        a = raw["anchor_fixed256_80M"]["summary"]
        for t, res in raw.items():
            if t != "anchor_fixed256_80M":
                all_summary["deltas_vs_anchor_fixed256_80M"][t] = delta(a, res["summary"])
    if "role_switch_80M" in raw and "role_fixed_80M" in raw:
        all_summary["role_switch_minus_role_fixed"] = delta(raw["role_fixed_80M"]["summary"], raw["role_switch_80M"]["summary"])

    combined = out_root / "ewok_interaction_summary.json"
    combined.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": all_summary["status"], "combined_summary": str(combined)}), flush=True)


if __name__ == "__main__":
    main()
