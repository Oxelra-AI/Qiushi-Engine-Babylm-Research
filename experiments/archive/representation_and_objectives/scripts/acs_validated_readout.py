#!/usr/bin/env python3
"""research: validated fixed-coordinate readout for ACS mechanism screen.

This replaces the research ad hoc readout.  It deliberately uses the existing
validated readers:
  * GlobalPIQA: research official-compatible all-option completion reader with
    length-normalized completion scores and the fixed cross-endpoint hard subset
    from `globalpiqa_parallel_anatomy/globalpiqa_parallel_anatomy.json`.
  * EWoK: research four-cell interaction reader, preserving row-level stable
    conditional-reversal records.

The script performs no training and does not tune on official items.  It is a
posthoc mechanism readout for already-existing checkpoints.  Run one target per
GPU for EWoK if parallel resource use matters; GlobalPIQA remains CPU in the
research implementation by design.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')

# Isolated writable caches for any custom-code model loading are placed under
# each output root after argument parsing.  Do not create cache here;
# run_shell write scopes declare the exact output directory only.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat: list[dict[str, Any]] = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k not in ("scores", "completions", "metadata")}
        if isinstance(r.get("scores"), list):
            rr.update({f"score_{i}": s for i, s in enumerate(r["scores"])})
        if isinstance(r.get("completions"), list):
            rr.update({f"completion_{i}": c for i, c in enumerate(r["completions"])})
        if isinstance(r.get("metadata"), dict):
            for k, v in r["metadata"].items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    rr[f"metadata_{k}"] = v
        flat.append(rr)
    keys = sorted({k for rr in flat for k in rr.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat)


def parse_targets(raw: list[list[str]]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for name, model_path, label in raw:
        if name in targets:
            raise ValueError(f"duplicate target name: {name}")
        p = Path(model_path)
        targets[name] = {"model_path": p, "model_root": p, "revision": None, "label": label}
    return targets


def compact_gp(res: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": res.get("target"),
        "label": res.get("label"),
        "model_root": rel(Path(res.get("model_root", "."))),
        "revision": res.get("revision"),
        "device": res.get("device"),
        "modes": {mode: payload.get("summary", {}) for mode, payload in res.get("modes", {}).items()},
    }


def compact_ewok(res: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": res.get("target"),
        "model_path": rel(Path(res.get("model_path", "."))),
        "device": res.get("device"),
        "row_limit": res.get("row_limit"),
        "row_offset": res.get("row_offset"),
        "summary": res.get("summary", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validated fixed-coordinate ACS readout")
    ap.add_argument("--target", nargs=3, action="append", metavar=("NAME", "MODEL_PATH", "LABEL"), required=True,
                    help="Target checkpoint. MODEL_PATH should be an HF checkpoint directory, e.g. .../hf_model/chck_100M")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--skip-globalpiqa", action="store_true")
    ap.add_argument("--skip-ewok", action="store_true")
    ap.add_argument("--gp-modes", nargs="+", default=["parallel", "nonparallel"], choices=["parallel", "nonparallel"])
    ap.add_argument("--gp-batch-size", type=int, default=8)
    ap.add_argument("--gp-non-causal-batch-size", type=int, default=32)
    ap.add_argument("--gp-max-items", type=int, default=None,
                    help="Only for smoke tests. Full readout must leave this unset.")
    ap.add_argument("--gp-threads", type=int, default=8)
    ap.add_argument("--ewok-device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--ewok-threads", type=int, default=8)
    ap.add_argument("--ewok-row-limit", type=int, default=0,
                    help="Only for smoke tests. Full readout must be 0.")
    ap.add_argument("--ewok-row-offset", type=int, default=0)
    ap.add_argument("--ewok-row-batch-size", type=int, default=64)
    ap.add_argument("--ewok-masked-batch-size", type=int, default=256)
    ap.add_argument("--ready-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = out_root / "hf_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root)
    os.environ["HF_MODULES_CACHE"] = str(cache_root / "modules")
    targets = parse_targets(args.target)

    preflight = {
        "status": "ACS_VALIDATED_READOUT_PREFLIGHT",
        "created_utc": now_utc(),
        "boundary": "posthoc readout only; no official-example tuning or custom submission scoring",
        "targets": {},
    }
    for name, meta in targets.items():
        p = Path(meta["model_path"])
        preflight["targets"][name] = {
            "model_path": rel(p),
            "exists": p.exists(),
            "model_safetensors": (p / "model.safetensors").exists(),
            "config_json": (p / "config.json").exists(),
            "tokenizer_json": (p / "tokenizer.json").exists(),
            "label": meta["label"],
        }
    (out_root / "preflight.json").write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(preflight, ensure_ascii=False), flush=True)
    if args.ready_only:
        return
    missing = [k for k, v in preflight["targets"].items() if not (v["exists"] and v["model_safetensors"] and v["config_json"])]
    if missing:
        raise FileNotFoundError(json.dumps({"missing_targets": missing, "preflight": preflight}, ensure_ascii=False))

    summary: dict[str, Any] = {
        "status": "ACS_VALIDATED_READOUT_DONE",
        "created_utc": now_utc(),
        "out_root": rel(out_root),
        "fixed_hard_set": {},
        "globalpiqa": {},
        "ewok": {},
    }

    if not args.skip_globalpiqa:
        gp = import_from(_public_path('experiments/archive/representation_and_objectives/scripts/globalpiqa_margin_reader.py'), "globalpiqa_margin_reader_step162")
        gp.OUT_ROOT = out_root / "globalpiqa" / "raw"
        gp.NOTE = out_root / "globalpiqa" / "globalpiqa_note.md"
        gp.TARGETS = {
            name: {"label": meta["label"], "model_root": Path(meta["model_path"]), "revision": None}
            for name, meta in targets.items()
        }
        hard_ids = gp.load_always_wrong_ids()
        summary["fixed_hard_set"] = {
            "source": rel(gp.ANATOMY_JSON),
            "definition": "example_id rows with n_ok == 0 in research cross-endpoint GlobalPIQA_parallel anatomy; fixed across all targets",
            "n_ids": len(hard_ids),
            "ids_sample": sorted(hard_ids)[:8],
            "length_normalized_reader": rel(_public_path('experiments/archive/representation_and_objectives/scripts/globalpiqa_margin_reader.py')),
        }
        gp_dir = out_root / "globalpiqa"
        gp_dir.mkdir(parents=True, exist_ok=True)
        for name in targets:
            print(json.dumps({"event": "globalpiqa_target_start", "target": name, "utc": now_utc()}), flush=True)
            res = gp.run_target(name, args.gp_modes, args.gp_batch_size, args.gp_non_causal_batch_size, args.gp_max_items, args.gp_threads)
            compact = compact_gp(res)
            summary["globalpiqa"][name] = compact
            (gp_dir / f"{name}_globalpiqa_margins.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            for mode, payload in res.get("modes", {}).items():
                write_rows_csv(gp_dir / f"{name}_{mode}_rows.csv", payload.get("rows", []))
            par = compact.get("modes", {}).get("parallel", {})
            aw = par.get("always_wrong_subset") if isinstance(par, dict) else None
            print(json.dumps({"event": "globalpiqa_target_done", "target": name,
                              "parallel_accuracy": par.get("accuracy") if isinstance(par, dict) else None,
                              "fixed_hard_accuracy": aw.get("accuracy") if isinstance(aw, dict) else None,
                              "fixed_hard_mean_margin": aw.get("mean_top_minus_correct") if isinstance(aw, dict) else None}, ensure_ascii=False), flush=True)
            del res
            gc.collect()

    if not args.skip_ewok:
        import torch
        ew = import_from(_public_path('experiments/archive/representation_and_objectives/scripts/fw_ewok_interaction_reader.py'), "fw_ewok_interaction_reader_step162")
        ew.OUT_ROOT = out_root / "ewok" / "raw"
        ew.NOTE = out_root / "ewok" / "ewok_note.md"
        ew.DEFAULT_TARGETS = {
            name: {"label": meta["label"], "model_path": Path(meta["model_path"])}
            for name, meta in targets.items()
        }
        device = torch.device("cuda" if args.ewok_device == "cuda" and torch.cuda.is_available() else "cpu")
        ew_dir = out_root / "ewok"
        ew_dir.mkdir(parents=True, exist_ok=True)
        for name, meta in targets.items():
            print(json.dumps({"event": "ewok_target_start", "target": name, "device": str(device), "utc": now_utc()}), flush=True)
            res = ew.run_target(name, Path(meta["model_path"]), device, args.ewok_threads,
                                args.ewok_row_limit, args.ewok_row_offset,
                                args.ewok_row_batch_size, args.ewok_masked_batch_size)
            records = res.pop("records")
            tdir = ew_dir / name
            tdir.mkdir(parents=True, exist_ok=True)
            ew.write_csv(tdir / "ewok_interaction_records.csv", records)
            ew.write_csv(tdir / "ewok_interaction_by_domain.csv", res["by_domain"])
            ew.write_csv(tdir / "ewok_interaction_by_context_diff.csv", res["by_context_diff"])
            (tdir / "ewok_interaction_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            summary["ewok"][name] = compact_ewok(res)
            print(json.dumps({"event": "ewok_target_done", "target": name, "summary": res.get("summary", {})}, ensure_ascii=False), flush=True)
            del res, records
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out_path = out_root / "acs_validated_readout_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": rel(out_path)}), flush=True)


if __name__ == "__main__":
    main()
