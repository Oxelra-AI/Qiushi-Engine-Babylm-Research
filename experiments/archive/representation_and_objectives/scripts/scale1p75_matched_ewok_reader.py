#!/usr/bin/env python3
"""research: matched-lineage EWoK interaction readout for scale1.75.

No training.  Compares adapter scale1.75 checkpoints with the matched
legal16k compact-view reinvest base at 80M and 100M.  This separates the score
question from mechanism attribution and avoids overreading legal40k cross-
tokenizer comparisons.
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

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_matched_ewok_interaction')
HF_CACHE = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_matched_ewok_interaction/hf_cache')

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HF_MODULES_CACHE"] = str(_public_path('experiments/archive/representation_and_objectives/data/scale1p75_matched_ewok_interaction/hf_cache/modules'))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import fw_ewok_interaction_reader as ew  # noqa: E402

TARGETS: dict[str, dict[str, Any]] = {
    "scale1p75_80M": {
        "label": "A02 adapter128 scale1.75 chck_80M",
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M'),
        "exposure": "80M",
        "role": "candidate",
    },
    "legal16k_80M": {
        "label": "A02 matched legal16k compact-view reinvest chck_80M",
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M'),
        "exposure": "80M",
        "role": "matched_base",
    },
    "scale1p75_100M": {
        "label": "A02 adapter128 scale1.75 official-ladder chck_100M",
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M'),
        "exposure": "100M",
        "role": "candidate",
    },
    "legal16k_100M": {
        "label": "A02 matched legal16k compact-view reinvest chck_100M",
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M'),
        "exposure": "100M",
        "role": "matched_base",
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def delta(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ["accuracy", "wrong", "stable_failure", "stable_failure_frac_all", "within_both_positive_wrong_frac"]:
        if isinstance(base.get(key), (int, float)) and isinstance(other.get(key), (int, float)):
            out[key] = other[key] - base[key]
    for src_key, out_key in [("interaction_sum_all", "interaction_sum_all_mean"), ("interaction_sum_wrong", "interaction_sum_wrong_mean")]:
        if isinstance(base.get(src_key), dict) and isinstance(other.get(src_key), dict):
            out[out_key] = other[src_key].get("mean", 0) - base[src_key].get("mean", 0)
    return out


def keep_summary(res: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in res.items() if k not in {"records", "by_domain", "by_context_diff"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=sorted(TARGETS))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--row_limit", type=int, default=0)
    ap.add_argument("--row_offset", type=int, default=0)
    ap.add_argument("--row_batch_size", type=int, default=32)
    ap.add_argument("--masked_batch_size", type=int, default=128)
    ap.add_argument("--ready-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    selected_targets = {k: TARGETS[k] for k in args.targets}
    readiness = {
        k: {
            "label": v["label"],
            "exposure": v["exposure"],
            "role": v["role"],
            "model_path": rel(Path(v["model_path"])),
            "model_ready": (Path(v["model_path"]) / "model.safetensors").exists(),
            "config_ready": (Path(v["model_path"]) / "config.json").exists(),
        }
        for k, v in selected_targets.items()
    }
    (out_root / "preflight.json").write_text(json.dumps({"created_utc": now(), "targets": readiness}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "preflight", "targets": readiness}, ensure_ascii=False), flush=True)
    if args.ready_only:
        return
    missing = [k for k, r in readiness.items() if not (r["model_ready"] and r["config_ready"])]
    if missing:
        raise RuntimeError({"missing": missing, "readiness": readiness})

    import torch
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    ew.OUT_ROOT = out_root
    ew.NOTE = _public_path('research/notes/representation_and_objectives/scale1p75_matched_ewok_interaction.md')
    ew.DEFAULT_TARGETS = selected_targets
    pf = ew.preflight(args.targets)
    (out_root / "ewok_preflight.json").write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if pf.get("status") != "READY":
        raise RuntimeError({"ewok_preflight": pf})

    all_summary: dict[str, Any] = {
        "status": "SCALE1P75_MATCHED_EWOK_INTERACTION_DONE",
        "created_utc": now(),
        "device": str(device),
        "targets": {},
        "deltas": {},
        "interpretation_warning": "Matched legal16k comparisons are attribution-safe; official EWoK score still comes from the official eval payloads.",
    }
    raw: dict[str, Any] = {}
    for t in args.targets:
        meta = selected_targets[t]
        print(json.dumps({"event": "target_start", "target": t, "device": str(device), "utc": now()}), flush=True)
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
        s = res["summary"]
        print(json.dumps({"event": "target_done", "target": t, "accuracy": s.get("accuracy"), "stable_failure": s.get("stable_failure"), "stable_failure_frac_all": s.get("stable_failure_frac_all")}), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pairs = [("80M", "legal16k_80M", "scale1p75_80M"), ("100M", "legal16k_100M", "scale1p75_100M")]
    for exposure, base_key, cand_key in pairs:
        if base_key in raw and cand_key in raw:
            all_summary["deltas"][f"scale1p75_minus_step35_legal16k_{exposure}"] = delta(raw[base_key]["summary"], raw[cand_key]["summary"])

    out = out_root / "scale1p75_matched_ewok_summary.json"
    out.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": all_summary["status"], "summary": rel(out)}), flush=True)


if __name__ == "__main__":
    main()
