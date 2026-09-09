#!/usr/bin/env python3
"""research: EWoK four-cell readout for disposable packet-screen checkpoints.

This is a target-swapped wrapper around the research EWoK interaction reader. It
scores existing non-submission screen checkpoints only and writes stable
conditional-reversal summaries.
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
DEFAULT_OUT_ROOT = A01_WS / "data" / "screen_ewok_interaction_reader"
DEFAULT_NOTE = A01_WS / "notes" / "145_screen_ewok_interaction_reader.md"

TARGETS: dict[str, dict[str, Any]] = {
    "base_reheat": {
        "label": "Base reheat 100M before disposable packet screen",
        "model_path": A01_WS / "training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M",
    },
    "screen_treatment": {
        "label": "Disposable role-switch treatment screen final (not submission-legal)",
        "model_path": A01_WS / "training/runs/disposable_role_switch_learning_screen/treatment/hf_model/screen_final",
    },
    "screen_role_fixed": {
        "label": "Disposable role-fixed control screen final (not submission-legal)",
        "model_path": A01_WS / "training/runs/disposable_role_switch_learning_screen/role_fixed/hf_model/screen_final",
    },
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ew.OUT_ROOT = out_root
    ew.NOTE = DEFAULT_NOTE
    ew.DEFAULT_TARGETS = {k: TARGETS[k] for k in args.targets}

    pf = ew.preflight(args.targets)
    (out_root / "preflight.json").write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if pf.get("status") != "READY":
        raise RuntimeError({"preflight": pf})

    import torch
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    all_summary: dict[str, Any] = {
        "status": "SCREEN_EWOK_INTERACTION_READER_DONE",
        "created_utc": now_utc(),
        "legal_status": "not_submission_facing; measures disposable screen checkpoints only",
        "device": str(device),
        "targets": {},
        "deltas_vs_base_reheat": {},
    }
    raw_summaries: dict[str, Any] = {}
    for t in args.targets:
        meta = TARGETS[t]
        print(json.dumps({"event": "target_start", "target": t, "device": str(device)}), flush=True)
        res = ew.run_target(t, Path(meta["model_path"]), device, args.threads, args.row_limit, args.row_offset, args.row_batch_size, args.masked_batch_size)
        tdir = out_root / t
        tdir.mkdir(parents=True, exist_ok=True)
        records = res.pop("records")
        ew.write_csv(tdir / "ewok_interaction_records.csv", records)
        ew.write_csv(tdir / "ewok_interaction_by_domain.csv", res["by_domain"])
        ew.write_csv(tdir / "ewok_interaction_by_context_diff.csv", res["by_context_diff"])
        (tdir / "ewok_interaction_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raw_summaries[t] = res
        all_summary["targets"][t] = {k: v for k, v in res.items() if k not in {"by_domain", "by_context_diff"}}
        print(json.dumps({"event": "target_done", "target": t, "summary": res["summary"], "summary_path": str(tdir / "ewok_interaction_summary.json")}, indent=2), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "base_reheat" in raw_summaries:
        b = raw_summaries["base_reheat"]["summary"]
        for t, res in raw_summaries.items():
            if t == "base_reheat":
                continue
            s = res["summary"]
            all_summary["deltas_vs_base_reheat"][t] = {
                "accuracy": s["accuracy"] - b["accuracy"],
                "stable_failure": s["stable_failure"] - b["stable_failure"],
                "stable_failure_frac_all": s["stable_failure_frac_all"] - b["stable_failure_frac_all"],
                "within_both_positive_wrong_frac": s["within_both_positive_wrong_frac"] - b["within_both_positive_wrong_frac"],
                "interaction_sum_all_mean": s["interaction_sum_all"]["mean"] - b["interaction_sum_all"]["mean"],
            }
    if "screen_treatment" in raw_summaries and "screen_role_fixed" in raw_summaries:
        s = raw_summaries["screen_treatment"]["summary"]
        b = raw_summaries["screen_role_fixed"]["summary"]
        all_summary["treatment_minus_role_fixed"] = {
            "accuracy": s["accuracy"] - b["accuracy"],
            "stable_failure": s["stable_failure"] - b["stable_failure"],
            "stable_failure_frac_all": s["stable_failure_frac_all"] - b["stable_failure_frac_all"],
            "within_both_positive_wrong_frac": s["within_both_positive_wrong_frac"] - b["within_both_positive_wrong_frac"],
            "interaction_sum_all_mean": s["interaction_sum_all"]["mean"] - b["interaction_sum_all"]["mean"],
        }

    combined = out_root / "screen_ewok_interaction_summary.json"
    combined.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": all_summary["status"], "combined_summary": str(combined)}), flush=True)


if __name__ == "__main__":
    main()
