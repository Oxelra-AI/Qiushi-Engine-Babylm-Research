#!/usr/bin/env python3
"""research: EWoK four-cell interaction readout for paired-tail endpoints.

This is a thin, target-swapped wrapper around the validated research EWoK
interaction reader. It scores existing tail endpoints only and writes stable
conditional-reversal summaries. It performs no training and does not use EWoK to
construct submission-time scorers or data.
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
DEFAULT_OUT_ROOT = A01_WS / "data" / "tail_ewok_interaction_reader"
DEFAULT_NOTE = A01_WS / "notes" / "143_tail_ewok_interaction_reader.md"

TARGETS: dict[str, dict[str, Any]] = {
    "tail_residual_low_lr_freshmom": {
        "label": "research residual-low-LR fresh-AdamW tail from legal40k chck_80M",
        "model_path": A01_WS / "training/runs/tail_residual_low_lr_freshmom_from_chck80M/hf_model/chck_100M",
    },
    "tail_reheat_cosine_freshmom": {
        "label": "research reheat-cosine fresh-AdamW tail from legal40k chck_80M",
        "model_path": A01_WS / "training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M",
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
        "status": "TAIL_EWOK_INTERACTION_READER_DONE",
        "created_utc": now_utc(),
        "boundary": "post-endpoint four-cell EWoK readout; no official-example shaping",
        "device": str(device),
        "targets": {},
    }
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
        all_summary["targets"][t] = {k: v for k, v in res.items() if k not in {"by_domain", "by_context_diff"}}
        print(json.dumps({"event": "target_done", "target": t, "summary": res["summary"], "summary_path": str(tdir / "ewok_interaction_summary.json")}, indent=2), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined = out_root / "tail_ewok_interaction_summary.json"
    combined.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": all_summary["status"], "combined_summary": str(combined)}), flush=True)


if __name__ == "__main__":
    main()
