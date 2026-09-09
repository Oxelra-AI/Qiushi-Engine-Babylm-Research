#!/usr/bin/env python3
"""research: score COMPACT_EXPERIENCE OFF vs ALN on the Strict-complement broad-fit axis.

This is a stock-checkpoint retro-measurement.  It asks whether the inherited
aligned-restatement block bought source-conditioned competence at a clean same-source
ordinary-fit cost.  The row set is the research Strict-complement axis: larger Strict
text sampled after same-source 16-token Strict-Small exclusion and later tokenizer-pool
exposure screening.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_paired_context_off_aln_strict_complement.py')
ROOT = _PUBLIC_ROOT

SCRIPTS = ROOT / "experiments/archive/relation_learning/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import score_dose_ordinary_fit_axes as fit  # noqa: E402

WS = ROOT / "experiments/archive/relation_learning"
COMPACT_EXPERIENCE_RUNS = ROOT / "experiments/archive/compact_experience/training/runs"
OUT_DEFAULT = WS / "data/paired_context_off_aln_strict_complement"
STRICT_AXIS = WS / "data/strict_complement_ngram_axis/strict_complement_ngram_axis_3000_rows.jsonl"
DEFAULT_CKPTS = ["chck_80M", "chck_90M", "chck_100M"]
ARMS: dict[str, dict[str, Any]] = {
    "OFF43022": {
        "relation": "OFF",
        "seed": 43022,
        "run_dir": COMPACT_EXPERIENCE_RUNS / "official_lengthmatched_16k_seed43022",
        "description": "official_lengthmatched no qwen-pair practice seed43022",
    },
    "ALN43022": {
        "relation": "ALN",
        "seed": 43022,
        "run_dir": COMPACT_EXPERIENCE_RUNS / "qwen_clean_aligned_16k_seed43022",
        "description": "qwen_clean_aligned original+own rewrite seed43022",
    },
    "OFF43122": {
        "relation": "OFF",
        "seed": 43122,
        "run_dir": COMPACT_EXPERIENCE_RUNS / "official_lengthmatched_16k_seed43122",
        "description": "official_lengthmatched no qwen-pair practice seed43122",
    },
    "ALN43122": {
        "relation": "ALN",
        "seed": 43122,
        "run_dir": COMPACT_EXPERIENCE_RUNS / "qwen_clean_aligned_16k_seed43122",
        "description": "qwen_clean_aligned original+own rewrite seed43122",
    },
}


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    df = pd.DataFrame(rows)
    by_ck = df.groupby(["seed", "relation", "arm", "checkpoint"], dropna=False).agg(
        n=("example_id", "count"),
        mean_loss=("loss", "mean"),
        sd_loss=("loss", "std"),
        mean_masked=("n_masked", "mean"),
    ).reset_index()
    late = by_ck.groupby(["seed", "relation", "arm"], dropna=False).agg(
        n_min=("n", "min"),
        mean_loss=("mean_loss", "mean"),
        sd_across_checkpoints=("mean_loss", "std"),
        mean_masked=("mean_masked", "mean"),
    ).reset_index()

    ck_contrasts: list[dict[str, Any]] = []
    for (seed, ck), g in by_ck.groupby(["seed", "checkpoint"], dropna=False):
        idx = g.set_index("relation")
        if "ALN" in idx.index and "OFF" in idx.index:
            aln = float(idx.loc["ALN", "mean_loss"])
            off = float(idx.loc["OFF", "mean_loss"])
            ck_contrasts.append({"seed": int(seed), "checkpoint": ck, "ALN_loss": aln, "OFF_loss": off, "ALN_minus_OFF": aln - off})
    late_contrasts: list[dict[str, Any]] = []
    for seed, g in late.groupby("seed", dropna=False):
        idx = g.set_index("relation")
        if "ALN" in idx.index and "OFF" in idx.index:
            aln = float(idx.loc["ALN", "mean_loss"])
            off = float(idx.loc["OFF", "mean_loss"])
            late_contrasts.append({"seed": int(seed), "checkpoint": "late_mean_80_90_100", "ALN_loss": aln, "OFF_loss": off, "ALN_minus_OFF": aln - off})

    rowpair_rows: list[dict[str, Any]] = []
    wide = df.groupby(["example_id", "source", "words", "seed", "relation"], dropna=False)["loss"].mean().reset_index()
    for seed, gs in wide.groupby("seed", dropna=False):
        piv = gs.pivot(index=["example_id", "source", "words"], columns="relation", values="loss")
        if "ALN" in piv.columns and "OFF" in piv.columns:
            vals = (piv["ALN"] - piv["OFF"]).dropna().to_numpy(dtype=float)
            rowpair_rows.append({
                "seed": int(seed),
                "contrast": "ALN-minus-OFF",
                "n_rows": int(len(vals)),
                "mean_delta_loss": float(vals.mean()) if len(vals) else float("nan"),
                "median_delta_loss": float(np.median(vals)) if len(vals) else float("nan"),
                "se_delta_loss": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
                "fraction_aln_lower_loss": float(np.mean(vals < 0.0)) if len(vals) else float("nan"),
            })
    return {
        "by_checkpoint": by_ck,
        "late_summary": late,
        "checkpoint_contrasts": pd.DataFrame(ck_contrasts),
        "late_contrasts": pd.DataFrame(late_contrasts),
        "rowpaired_contrasts": pd.DataFrame(rowpair_rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT_CKPTS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_axis = fit.load_rows(STRICT_AXIS)
    missing: list[str] = []
    for arm in args.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}")
        for ck in args.checkpoints:
            mp = pathlib.Path(ARMS[arm]["run_dir"]) / "hf_model" / ck
            if not fit.model_file_ready(mp):
                missing.append(rel(mp))
    plan = {
        "status": "COMPACT_EXPERIENCE_OFF_ALN_STRICT_COMPLEMENT_PLAN",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "strict_axis": rel(STRICT_AXIS),
        "axis_rows": len(rows_axis),
        "axis_words": int(sum(int(r["words"]) for r in rows_axis)),
        "arms": {a: {**{k: (rel(v) if isinstance(v, pathlib.Path) else v) for k, v in ARMS[a].items()}} for a in args.arms},
        "checkpoints": args.checkpoints,
        "missing": missing,
        "device": args.device,
        "stock_loader_expected": True,
    }
    (out_dir / "score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    if missing:
        raise FileNotFoundError(missing[0])

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(pathlib.Path(ARMS[args.arms[0]]["run_dir"]) / "hf_model"), use_fast=True)
    scored_rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    ident_rows: list[dict[str, Any]] = []
    for arm in args.arms:
        cfg = ARMS[arm]
        run = pathlib.Path(cfg["run_dir"])
        for ck in args.checkpoints:
            mp = run / "hf_model" / ck
            t0 = time.time()
            print(f"[LOAD] {arm} {ck} {mp}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(mp), torch_dtype=torch.float32).eval().to(device)
            ident = fit.model_identity(mp, model, trust_remote_code=False, local_files_only=True)
            ident.update({"arm": arm, "relation": cfg["relation"], "seed": cfg["seed"], "checkpoint": ck})
            ident_rows.append(ident)
            print(json.dumps({"event": "model_identity", **ident}, ensure_ascii=False), flush=True)
            scored = fit.score_rows(model, tok, rows_axis, device, args.batch_size)
            mean_loss = statistics.mean([x["loss"] for x in scored])
            for r in scored:
                q = dict(r)
                q.update({
                    "axis": "strict_complement_ngram_3000",
                    "axis_path": rel(STRICT_AXIS),
                    "arm": arm,
                    "relation": cfg["relation"],
                    "seed": cfg["seed"],
                    "description": cfg["description"],
                    "checkpoint": ck,
                })
                scored_rows.append(q)
            meta.append({
                "arm": arm,
                "relation": cfg["relation"],
                "seed": cfg["seed"],
                "checkpoint": ck,
                "n": len(scored),
                "mean_loss": mean_loss,
                "elapsed_sec_since_model_load": round(time.time() - t0, 2),
                "device": str(device),
                "loaded_class": ident.get("loaded_class"),
                "total_params_loaded": ident.get("total_params_loaded"),
                "adapter_params_loaded": ident.get("adapter_params_loaded"),
            })
            print(f"[DONE] {arm} {ck} mean={mean_loss:.6f} n={len(scored)}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(out_dir / "strict_fit_rows.csv", scored_rows)
    write_csv(out_dir / "strict_fit_score_meta.csv", meta)
    with (out_dir / "model_identity_preamble.jsonl").open("w", encoding="utf-8") as f:
        for row in ident_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summaries = summarize(scored_rows)
    for name, df in summaries.items():
        df.to_csv(out_dir / f"strict_fit_{name}.csv", index=False)
    result = {
        "status": "COMPACT_EXPERIENCE_OFF_ALN_STRICT_COMPLEMENT_DONE",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "rows_scored": len(scored_rows),
        "axis": rel(STRICT_AXIS),
        "model_identity_preamble": rel(out_dir / "model_identity_preamble.jsonl"),
        "late_contrasts": json.loads(summaries["late_contrasts"].to_json(orient="records")) if not summaries["late_contrasts"].empty else [],
        "rowpaired_contrasts": json.loads(summaries["rowpaired_contrasts"].to_json(orient="records")) if not summaries["rowpaired_contrasts"].empty else [],
        "interpretation_note": "Negative ALN_minus_OFF means the aligned-restatement block improves this clean Strict-complement ordinary-fit axis; positive means it costs clean fit relative to OFF.",
    }
    (out_dir / "strict_fit_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
