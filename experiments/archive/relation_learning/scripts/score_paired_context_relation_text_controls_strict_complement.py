#!/usr/bin/env python3
"""research: score COMPACT_EXPERIENCE same-text relation controls on the clean Strict-complement axis.

The research OFF-vs-ALN retro-score showed that the inherited aligned-restatement
block lowers loss on a constructed clean same-source ordinary text axis, but that
fit gain could come from the fluent/simple rewrite text rather than the local
correct-pair relation.  This script scores the stock SHUF and SEP controls on the
same 3,000-row research Strict-complement axis, using the same late checkpoints
(chck_80M/90M/100M) and stock model loading contract as research.

Available controls:
  * OFF: official lengthmatched, no qwen pair practice.
  * ALN: original plus its own rewrite in the same window.
  * SHUF: same selected original and rewrite multisets, wrong correspondence in the
    same window.
  * SEP: same selected originals and rewrites but separated across rows; useful for
    locality, with the known packing caveat.

Seed43122 SHUF exists in relation_learning because it was run later for the replication
probe; SEP43122 was not found in the available run tree.
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
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_paired_context_relation_text_controls_strict_complement.py')
ROOT = _PUBLIC_ROOT

SCRIPTS = ROOT / "experiments/archive/relation_learning/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import score_dose_ordinary_fit_axes as fit  # noqa: E402

WS = ROOT / "experiments/archive/relation_learning"
COMPACT_EXPERIENCE_RUNS = ROOT / "experiments/archive/compact_experience/training/runs"
A02_RUNS = ROOT / "experiments/archive/relation_learning/training/runs"
STRICT_AXIS = WS / "data/strict_complement_ngram_axis/strict_complement_ngram_axis_3000_rows.jsonl"
OUT_DEFAULT = WS / "data/paired_context_relation_text_controls_strict_complement"
DEFAULT_CKPTS = ["chck_80M", "chck_90M", "chck_100M"]

ARMS: dict[str, dict[str, Any]] = {
    "OFF43022": {
        "relation": "OFF", "seed": 43022,
        "run_dir": COMPACT_EXPERIENCE_RUNS / "official_lengthmatched_16k_seed43022",
        "description": "official_lengthmatched no qwen-pair practice seed43022",
    },
    "ALN43022": {
        "relation": "ALN", "seed": 43022,
        "run_dir": COMPACT_EXPERIENCE_RUNS / "qwen_clean_aligned_16k_seed43022",
        "description": "qwen_clean_aligned original+own rewrite seed43022",
    },
    "SHUF43022": {
        "relation": "SHUF", "seed": 43022,
        "run_dir": COMPACT_EXPERIENCE_RUNS / "qwen_shuffled_control_16k_seed43022",
        "description": "qwen_shuffled_control originals paired with wrong rewrites seed43022",
    },
    "SEP43022": {
        "relation": "SEP", "seed": 43022,
        "run_dir": COMPACT_EXPERIENCE_RUNS / "qwen_separated_pair_16k_seed43022",
        "description": "qwen_separated_pair same originals/rewrites separated across rows seed43022",
    },
    "OFF43122": {
        "relation": "OFF", "seed": 43122,
        "run_dir": COMPACT_EXPERIENCE_RUNS / "official_lengthmatched_16k_seed43122",
        "description": "official_lengthmatched no qwen-pair practice seed43122",
    },
    "ALN43122": {
        "relation": "ALN", "seed": 43122,
        "run_dir": COMPACT_EXPERIENCE_RUNS / "qwen_clean_aligned_16k_seed43122",
        "description": "qwen_clean_aligned original+own rewrite seed43122",
    },
    "SHUF43122": {
        "relation": "SHUF", "seed": 43122,
        "run_dir": A02_RUNS / "qwen_shuffled_control_16k_seed43122",
        "description": "qwen_shuffled_control originals paired with wrong rewrites seed43122, later replication run",
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


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
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def summarize(scored_rows: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    df = pd.DataFrame(scored_rows)
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
    late_contrasts: list[dict[str, Any]] = []
    for (seed, ck), g in by_ck.groupby(["seed", "checkpoint"], dropna=False):
        rel_to_loss = {str(r["relation"]): float(r["mean_loss"]) for _, r in g.iterrows()}
        for a, b in combinations(sorted(rel_to_loss), 2):
            # emit both scientific direction names in a stable order by relation labels
            ck_contrasts.append({
                "seed": int(seed), "checkpoint": ck,
                "contrast": f"{a}-minus-{b}",
                "a_relation": a, "b_relation": b,
                "a_loss": rel_to_loss[a], "b_loss": rel_to_loss[b],
                "delta_loss": rel_to_loss[a] - rel_to_loss[b],
            })
    for seed, g in late.groupby("seed", dropna=False):
        rel_to_loss = {str(r["relation"]): float(r["mean_loss"]) for _, r in g.iterrows()}
        for a, b in combinations(sorted(rel_to_loss), 2):
            late_contrasts.append({
                "seed": int(seed), "checkpoint": "late_mean_80_90_100",
                "contrast": f"{a}-minus-{b}",
                "a_relation": a, "b_relation": b,
                "a_loss": rel_to_loss[a], "b_loss": rel_to_loss[b],
                "delta_loss": rel_to_loss[a] - rel_to_loss[b],
            })

    wide = df.groupby(["example_id", "source", "words", "seed", "relation"], dropna=False)["loss"].mean().reset_index()
    rowpair_rows: list[dict[str, Any]] = []
    per_source_rows: list[dict[str, Any]] = []
    for seed, gs in wide.groupby("seed", dropna=False):
        piv = gs.pivot(index=["example_id", "source", "words"], columns="relation", values="loss")
        for a, b in combinations(sorted(list(piv.columns)), 2):
            vals = (piv[a] - piv[b]).dropna().to_numpy(dtype=float)
            if not len(vals):
                continue
            rowpair_rows.append({
                "seed": int(seed), "contrast": f"{a}-minus-{b}",
                "n_rows": int(len(vals)),
                "mean_delta_loss": float(vals.mean()),
                "median_delta_loss": float(np.median(vals)),
                "se_delta_loss": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
                "fraction_a_lower_loss": float(np.mean(vals < 0.0)),
            })
            by_source = pd.DataFrame({"source": [idx[1] for idx in vals_to_index(piv, a, b)], "delta": (piv[a] - piv[b]).dropna().to_numpy(dtype=float)})
            # The helper above preserves index order but avoid trusting it for summary; recompute by source directly.
            diff = (piv[a] - piv[b]).dropna().reset_index(name="delta")
            for src, gg in diff.groupby("source", dropna=False):
                arr = gg["delta"].to_numpy(dtype=float)
                per_source_rows.append({
                    "seed": int(seed), "contrast": f"{a}-minus-{b}", "source": src,
                    "n_rows": int(len(arr)),
                    "mean_delta_loss": float(arr.mean()),
                    "se_delta_loss": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan"),
                    "fraction_a_lower_loss": float(np.mean(arr < 0.0)),
                })
    return {
        "by_checkpoint": by_ck,
        "late_summary": late,
        "checkpoint_contrasts": pd.DataFrame(ck_contrasts),
        "late_contrasts": pd.DataFrame(late_contrasts),
        "rowpaired_contrasts": pd.DataFrame(rowpair_rows),
        "rowpaired_contrasts_by_source": pd.DataFrame(per_source_rows),
    }


def vals_to_index(piv: pd.DataFrame, a: str, b: str):
    # Tiny compatibility helper used only for an intermediate vector; returns the
    # row index of rows where both columns are present.
    return list((piv[a] - piv[b]).dropna().index)


def concise_selected_contrasts(late_contrasts: pd.DataFrame, rowpaired: pd.DataFrame) -> list[dict[str, Any]]:
    want = {
        "ALN-minus-OFF", "SHUF-minus-OFF", "ALN-minus-SHUF",
        "SEP-minus-OFF", "ALN-minus-SEP", "SEP-minus-SHUF",
    }
    rows: list[dict[str, Any]] = []
    # Our generated combinations are alphabetical, so reconstruct requested signs
    # from relation late means and rowpair summaries below.
    # This routine is filled from late_summary after the CSVs are written in main.
    return rows


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
    selected_arms = []
    for arm in args.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}")
        selected_arms.append(arm)
        for ck in args.checkpoints:
            mp = pathlib.Path(ARMS[arm]["run_dir"]) / "hf_model" / ck
            if not fit.model_file_ready(mp):
                missing.append(rel(mp))
    plan = {
        "status": "COMPACT_EXPERIENCE_RELATION_TEXT_CONTROLS_STRICT_COMPLEMENT_PLAN",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "strict_axis": rel(STRICT_AXIS),
        "axis_rows": len(rows_axis),
        "axis_words": int(sum(int(r["words"]) for r in rows_axis)),
        "arms": {a: {k: (rel(v) if isinstance(v, pathlib.Path) else v) for k, v in ARMS[a].items()} for a in selected_arms},
        "checkpoints": args.checkpoints,
        "missing": missing,
        "device": args.device,
        "stock_loader_expected": True,
        "scientific_question": "Does the block-level Strict-complement fit gain follow rewrite text/register or the correct local relation? SHUF controls same text with wrong same-window pairing; SEP controls coexistence without same-window pairing, with a packing caveat.",
    }
    (out_dir / "score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    if missing:
        raise FileNotFoundError(missing[0])

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(pathlib.Path(ARMS[selected_arms[0]]["run_dir"]) / "hf_model"), use_fast=True)
    scored_rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    ident_rows: list[dict[str, Any]] = []
    for arm in selected_arms:
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
                "arm": arm, "relation": cfg["relation"], "seed": cfg["seed"], "checkpoint": ck,
                "n": len(scored), "mean_loss": mean_loss,
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

    # Create a human-readable selected-sign contrast table from late means and rowpaired means.
    late = summaries["late_summary"]
    selected_rows: list[dict[str, Any]] = []
    desired = [("ALN", "OFF"), ("SHUF", "OFF"), ("ALN", "SHUF"), ("SEP", "OFF"), ("ALN", "SEP"), ("SEP", "SHUF")]
    for seed, g in late.groupby("seed", dropna=False):
        rel_to_loss = {str(r["relation"]): float(r["mean_loss"]) for _, r in g.iterrows()}
        for a, b in desired:
            if a in rel_to_loss and b in rel_to_loss:
                selected_rows.append({
                    "seed": int(seed), "contrast": f"{a}-minus-{b}",
                    "a_loss": rel_to_loss[a], "b_loss": rel_to_loss[b],
                    "delta_loss": rel_to_loss[a] - rel_to_loss[b],
                })
    write_csv(out_dir / "strict_fit_selected_late_contrasts.csv", selected_rows)

    result = {
        "status": "COMPACT_EXPERIENCE_RELATION_TEXT_CONTROLS_STRICT_COMPLEMENT_DONE",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "rows_scored": len(scored_rows),
        "axis": rel(STRICT_AXIS),
        "model_identity_preamble": rel(out_dir / "model_identity_preamble.jsonl"),
        "late_summary": json.loads(summaries["late_summary"].to_json(orient="records")) if not summaries["late_summary"].empty else [],
        "selected_late_contrasts": selected_rows,
        "rowpaired_contrasts": json.loads(summaries["rowpaired_contrasts"].to_json(orient="records")) if not summaries["rowpaired_contrasts"].empty else [],
        "outputs": {
            "rows": rel(out_dir / "strict_fit_rows.csv"),
            "meta": rel(out_dir / "strict_fit_score_meta.csv"),
            "late_summary_csv": rel(out_dir / "strict_fit_late_summary.csv"),
            "selected_late_contrasts_csv": rel(out_dir / "strict_fit_selected_late_contrasts.csv"),
            "rowpaired_by_source_csv": rel(out_dir / "strict_fit_rowpaired_contrasts_by_source.csv"),
        },
        "interpretation_note": "Negative A-minus-B means A has lower MLM loss on the clean Strict-complement axis. If SHUF is close to ALN, the clean-fit gain is mostly rewrite/register text; if ALN is uniquely lower than SHUF, correct correspondence contributes to ordinary fit. SEP separates coexistence without same-window pairing but has its known packing caveat.",
    }
    (out_dir / "strict_fit_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
