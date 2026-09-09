#!/usr/bin/env python3
"""research: CPU GlobalPIQA hard-rank readout for mature checkpoint averages.

Uses the validated research all-option GlobalPIQA reader.  No training and no GPU.
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
DEFAULT_OUT_ROOT = A01_WS / "data/mature_globalpiqa_reader"

TARGETS: dict[str, dict[str, Any]] = {
    "anchor_fixed256_100M": {
        "label": "legal40k fixed-256 compact-view anchor chck_100M",
        "model_root": A01_WS / "training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M",
        "revision": None,
    },
    "avg_uniform_80_100": {
        "label": "uniform average of chck_80M..chck_100M",
        "model_root": A01_WS / "training/runs/mature_checkpoint_averages/avg_uniform_80_100/hf_model/chck_avg",
        "revision": None,
    },
    "avg_uniform_90_100": {
        "label": "uniform average of chck_90M..chck_100M",
        "model_root": A01_WS / "training/runs/mature_checkpoint_averages/avg_uniform_90_100/hf_model/chck_avg",
        "revision": None,
    },
    "avg_exp_70_100_hl10": {
        "label": "exponential late-weight average chck_70M..chck_100M half-life 10M",
        "model_root": A01_WS / "training/runs/mature_checkpoint_averages/avg_exp_70_100_hl10/hf_model/chck_avg",
        "revision": None,
    },
    "avg_linear_90_100": {
        "label": "linear late-weight average chck_90M..chck_100M",
        "model_root": A01_WS / "training/runs/mature_checkpoint_averages/avg_linear_90_100/hf_model/chck_avg",
        "revision": None,
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


def delta_summary(base: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in sorted(set(base.get("modes", {})) & set(target.get("modes", {}))):
        b = base["modes"][mode]
        t = target["modes"][mode]
        d: dict[str, Any] = {}
        for key in ["accuracy", "chance_adjusted_accuracy"]:
            if isinstance(b.get(key), (int, float)) and isinstance(t.get(key), (int, float)):
                d[key] = t[key] - b[key]
        if isinstance(b.get("always_wrong_subset"), dict) and isinstance(t.get("always_wrong_subset"), dict):
            for key in ["accuracy", "mean_top_minus_correct"]:
                if isinstance(b["always_wrong_subset"].get(key), (int, float)) and isinstance(t["always_wrong_subset"].get(key), (int, float)):
                    d[f"always_wrong_subset_{key}"] = t["always_wrong_subset"][key] - b["always_wrong_subset"][key]
            br = b["always_wrong_subset"].get("correct_rank_counts") or b["always_wrong_subset"].get("rank_counts") or {}
            tr = t["always_wrong_subset"].get("correct_rank_counts") or t["always_wrong_subset"].get("rank_counts") or {}
            if isinstance(br, dict) and isinstance(tr, dict):
                d["always_wrong_rank_count_deltas"] = {str(k): tr.get(str(k), tr.get(k, 0)) - br.get(str(k), br.get(k, 0)) for k in sorted(set(map(str, br.keys())) | set(map(str, tr.keys())))}
        if isinstance(b.get("all_rows_margin_summary"), dict) and isinstance(t.get("all_rows_margin_summary"), dict):
            for key in ["mean_top_minus_correct"]:
                if isinstance(b["all_rows_margin_summary"].get(key), (int, float)) and isinstance(t["all_rows_margin_summary"].get(key), (int, float)):
                    d[f"all_rows_{key}"] = t["all_rows_margin_summary"][key] - b["all_rows_margin_summary"][key]
        out[mode] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=sorted(TARGETS))
    ap.add_argument("--modes", nargs="+", default=["parallel", "nonparallel"], choices=sorted(gp.TASKS))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--non-causal-batch-size", type=int, default=32)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ready-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    selected_targets = {k: TARGETS[k] for k in args.targets}
    readiness = {k: {"model_root": str(v["model_root"]), "ready": (Path(v["model_root"]) / "model.safetensors").exists()} for k, v in selected_targets.items()}
    (out_root / "preflight.json").write_text(json.dumps({"targets": readiness}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "preflight", "ready": readiness}, indent=2), flush=True)
    if args.ready_only:
        return
    missing = [k for k, v in readiness.items() if not v["ready"]]
    if missing:
        raise RuntimeError({"missing_checkpoints": missing, "readiness": readiness})

    gp.OUT_ROOT = out_root
    gp.TARGETS = selected_targets
    combined: dict[str, Any] = {
        "status": "MATURE_GLOBALPIQA_READER_DONE",
        "created_utc": now_utc(),
        "route": "mature checkpoint averaging; CPU-only GlobalPIQA all-option readout",
        "targets": {},
        "deltas_vs_anchor_fixed256_100M": {},
    }
    summaries: dict[str, Any] = {}
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
        summaries[target] = summary_only
        print(json.dumps({"event": "target_done", "target": target, "summaries": summary_only["modes"]}, indent=2), flush=True)

    if "anchor_fixed256_100M" in summaries:
        for target in summaries:
            if target != "anchor_fixed256_100M":
                combined["deltas_vs_anchor_fixed256_100M"][target] = delta_summary(summaries["anchor_fixed256_100M"], summaries[target])

    combined_json = out_root / "mature_globalpiqa_summary.json"
    combined_json.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": combined["status"], "combined_json": str(combined_json)}), flush=True)


if __name__ == "__main__":
    main()
