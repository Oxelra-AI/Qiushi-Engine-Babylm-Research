#!/usr/bin/env python3
"""research: GlobalPIQA all-option readout for paired-tail endpoints.

This reuses the research official-compatible MLM completion scorer, but supplies the
fresh-moment residual-low-LR and reheat-cosine tail endpoints as arbitrary targets.
It is a post-training measurement only: no score transformation here is used for
submission or training-data construction.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import globalpiqa_margin_reader as gp  # noqa: E402

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
DEFAULT_OUT_ROOT = A01_WS / "data" / "tail_globalpiqa_margin_reader"

TARGETS: dict[str, dict[str, Any]] = {
    "tail_residual_low_lr_freshmom": {
        "label": "research residual-low-LR fresh-AdamW tail from legal40k chck_80M",
        "model_root": A01_WS / "training/runs/tail_residual_low_lr_freshmom_from_chck80M/hf_model/chck_100M",
        "revision": None,
        "official_parallel": 24.27,
        "official_nonparallel": 47.0,
    },
    "tail_reheat_cosine_freshmom": {
        "label": "research reheat-cosine fresh-AdamW tail from legal40k chck_80M",
        "model_root": A01_WS / "training/runs/tail_reheat_cosine_freshmom_from_chck80M/hf_model/chck_100M",
        "revision": None,
        "official_parallel": 24.27,
        "official_nonparallel": 51.0,
    },
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat: list[dict[str, Any]] = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k not in ("scores", "completions", "metadata")}
        rr.update({f"score_{i}": s for i, s in enumerate(r.get("scores", []))})
        rr.update({f"completion_{i}": c for i, c in enumerate(r.get("completions", []))})
        flat.append(rr)
    keys = sorted({k for rr in flat for k in rr.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=sorted(TARGETS))
    ap.add_argument("--modes", nargs="+", default=["parallel", "nonparallel"], choices=sorted(gp.TASKS))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--non-causal-batch-size", type=int, default=32)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    gp.OUT_ROOT = out_root
    gp.TARGETS = {k: TARGETS[k] for k in args.targets}

    combined: dict[str, Any] = {
        "status": "TAIL_GLOBALPIQA_MARGIN_READER_DONE",
        "created_utc": now_utc(),
        "boundary": "post-endpoint all-option readout; no official-row tuning",
        "targets": {},
    }
    for target in args.targets:
        print(json.dumps({"event": "target_start", "target": target, "modes": args.modes}), flush=True)
        res = gp.run_target(target, args.modes, args.batch_size, args.non_causal_batch_size, args.max_items, args.threads)
        target_json = out_root / f"{target}_margins.json"
        target_json.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary_only = {"label": res.get("label"), "model_root": res.get("model_root"), "revision": res.get("revision"), "modes": {}}
        for mode, md in res.get("modes", {}).items():
            rows = md.get("rows", [])
            write_rows_csv(out_root / f"{target}_{mode}_rows.csv", rows)
            summary_only["modes"][mode] = md.get("summary", {})
        combined["targets"][target] = summary_only
        print(json.dumps({"event": "target_done", "target": target, "summaries": summary_only["modes"]}, indent=2), flush=True)

    combined_json = out_root / "tail_globalpiqa_margin_summary.json"
    combined_json.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": combined["status"], "combined_json": str(combined_json)}), flush=True)


if __name__ == "__main__":
    main()
