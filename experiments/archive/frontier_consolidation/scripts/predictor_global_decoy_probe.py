#!/usr/bin/env python3
"""research: broad exact-changed-row predictor diagnostic for source-view consistency.

Imports the repaired global same-row decoy audit machinery, extracts row-centered projected
source/rewrite residuals from exact changed rows, and tests whether a closed-form asymmetric
ridge predictor source->rewrite improves held-out true-vs-same-row-decoy separation over the
identity source residual.

CPU-only. No model update, official evaluation, corpus/tokenizer change, or H100 work.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import math
import pathlib
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

import torch


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
GLOBAL_AUDIT_PATH = WORKSPACE / "scripts/same_row_decoy_global_audit.py"
OUT_DIR = WORKSPACE / "data/predictor_global_decoy_probe"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


global_audit = load_module("same_row_decoy_global_audit_import", GLOBAL_AUDIT_PATH)


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "std": float(statistics.pstdev(vals)),
    }


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=1, eps=1e-6)


def within_row_metrics(P: torch.Tensor, W: torch.Tensor, row_key: torch.Tensor) -> dict[str, Any]:
    Pn = normalize(P)
    Wn = normalize(W)
    n = Pn.shape[0]
    by_row: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        by_row[int(row_key[i].item())].append(i)
    true_vals: list[float] = []
    decoy_vals: list[float] = []
    true_minus_decoy_mean: list[float] = []
    true_minus_decoy_max: list[float] = []
    ranks: list[int] = []
    top1 = 0
    for i in range(n):
        members = by_row[int(row_key[i].item())]
        decoys = [j for j in members if j != i]
        if not decoys:
            continue
        tv = float((Pn[i] * Wn[i]).sum().item())
        dvs = [float((Pn[i] * Wn[j]).sum().item()) for j in decoys]
        true_vals.append(tv)
        decoy_vals.extend(dvs)
        true_minus_decoy_mean.append(tv - statistics.mean(dvs))
        true_minus_decoy_max.append(tv - max(dvs))
        cands = [tv] + dvs
        order = sorted(range(len(cands)), key=lambda k: -cands[k])
        rank = order.index(0) + 1
        ranks.append(rank)
        if rank == 1:
            top1 += 1
    return {
        "n_pairs_with_decoys": len(true_vals),
        "n_decoy_comparisons": len(decoy_vals),
        "true_cos": summarize(true_vals),
        "same_row_decoy_cos": summarize(decoy_vals),
        "true_minus_same_row_mean": summarize(true_minus_decoy_mean),
        "true_minus_same_row_max": summarize(true_minus_decoy_max),
        "within_row_top1_acc": top1 / len(ranks) if ranks else None,
        "within_row_mean_rank": statistics.mean(ranks) if ranks else None,
    }


def ridge_predict(S_train: torch.Tensor, W_train: torch.Tensor, S_test: torch.Tensor, alpha: float) -> torch.Tensor:
    H = S_train.shape[1]
    s_mean = S_train.mean(dim=0, keepdim=True)
    w_mean = W_train.mean(dim=0, keepdim=True)
    X = S_train - s_mean
    Y = W_train - w_mean
    A = X.t() @ X + alpha * torch.eye(H)
    B = X.t() @ Y
    Wmat = torch.linalg.solve(A, B)
    return (S_test - s_mean) @ Wmat + w_mean


def split_row_masks(row_key: torch.Tensor, train_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = sorted(set(int(r.item()) for r in row_key))
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    perm = torch.randperm(len(rows), generator=g).tolist()
    n_train = max(2, int(round(len(rows) * train_frac)))
    train_rows = set(rows[i] for i in perm[:n_train])
    train_mask = torch.tensor([int(r.item()) in train_rows for r in row_key], dtype=torch.bool)
    return train_mask, ~train_mask


def predictor_splits(S: torch.Tensor, W: torch.Tensor, row_key: torch.Tensor, args) -> dict[str, Any]:
    runs = []
    for k in range(args.splits):
        train_mask, test_mask = split_row_masks(row_key, args.train_frac, args.split_seed + k)
        if int(test_mask.sum().item()) < 4 or int(train_mask.sum().item()) < 4:
            runs.append({"status": "too_few_pairs"})
            continue
        P = ridge_predict(S[train_mask], W[train_mask], S[test_mask], args.ridge_alpha)
        pred = within_row_metrics(P, W[test_mask], row_key[test_mask])
        ident = within_row_metrics(S[test_mask], W[test_mask], row_key[test_mask])
        adv_mean = None
        adv_top1 = None
        if pred["true_minus_same_row_mean"]["mean"] is not None and ident["true_minus_same_row_mean"]["mean"] is not None:
            adv_mean = pred["true_minus_same_row_mean"]["mean"] - ident["true_minus_same_row_mean"]["mean"]
        if pred["within_row_top1_acc"] is not None and ident["within_row_top1_acc"] is not None:
            adv_top1 = pred["within_row_top1_acc"] - ident["within_row_top1_acc"]
        runs.append({
            "status": "ok",
            "split_index": k,
            "n_train_pairs": int(train_mask.sum().item()),
            "n_test_pairs": int(test_mask.sum().item()),
            "predictor": pred,
            "identity": ident,
            "predictor_advantage_true_minus_decoy_mean": adv_mean,
            "predictor_advantage_top1": adv_top1,
        })
    ok = [r for r in runs if r.get("status") == "ok"]
    def agg(path: list[str]) -> dict[str, Any]:
        vals = []
        for r in ok:
            cur: Any = r
            for p in path:
                cur = cur.get(p) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return summarize(vals)
    summary = {
        "n_ok_splits": len(ok),
        "predictor_true_minus_decoy_mean": agg(["predictor", "true_minus_same_row_mean", "mean"]),
        "identity_true_minus_decoy_mean": agg(["identity", "true_minus_same_row_mean", "mean"]),
        "predictor_true_minus_decoy_max": agg(["predictor", "true_minus_same_row_max", "mean"]),
        "identity_true_minus_decoy_max": agg(["identity", "true_minus_same_row_max", "mean"]),
        "predictor_top1": agg(["predictor", "within_row_top1_acc"]),
        "identity_top1": agg(["identity", "within_row_top1_acc"]),
        "predictor_advantage_true_minus_decoy_mean": agg(["predictor_advantage_true_minus_decoy_mean"]),
        "predictor_advantage_top1": agg(["predictor_advantage_top1"]),
    }
    return {"summary": summary, "runs": runs}


def run_checkpoint(label: str, ckpt_path: pathlib.Path, examples, tokenizer, span_by_ex, args) -> dict[str, Any]:
    if not ckpt_path.exists():
        return {"label": label, "status": "missing", "path": str(ckpt_path)}
    data = global_audit.collect_vectors(ckpt_path, examples, tokenizer, span_by_ex, args)
    row_key = data["row_keys"]
    by_dim = {}
    for d in args.proj_dims_list:
        S = data["srcs_by_dim"][str(d)]
        W = data["rews_by_dim"][str(d)]
        baseline = within_row_metrics(S, W, row_key)
        ridge = predictor_splits(S, W, row_key, args)
        by_dim[str(d)] = {"baseline_identity_all_pairs": baseline, "ridge_holdout": ridge}
    return {
        "label": label,
        "status": "ok",
        "path": str(ckpt_path),
        "n_pairs": data["n_pairs"],
        "n_rows": data["n_rows"],
        "center_mode": args.center_mode,
        "metrics_by_projection_dim": by_dim,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--sample-rows", type=int, default=768)
    ap.add_argument("--sample-mode", choices=["front", "stride", "random"], default="stride")
    ap.add_argument("--sample-seed", type=int, default=71071)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--train-rng-seed", type=int, default=43023)
    ap.add_argument("--center-mode", choices=["raw", "row_centered"], default="row_centered")
    ap.add_argument("--proj-dims", default="64,128,480")
    ap.add_argument("--proj-seed", type=int, default=71072)
    ap.add_argument("--checkpoints", default="tokenmean_80M,tokenmean_100M")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=71171)
    args = ap.parse_args()
    args.proj_dims_list = [int(x) for x in args.proj_dims.split(",") if x.strip()]
    t0 = time.time()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_sha = global_audit.collate_mod.sha256_file(global_audit.collate_mod.TRAIN_100M)
    tok_sha = global_audit.collate_mod.sha256_file(global_audit.collate_mod.TOKENIZER_DIR / "tokenizer.json")
    if train_sha != global_audit.collate_mod.EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {train_sha}")
    if tok_sha != global_audit.collate_mod.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    span_by_ex = global_audit.collate_mod.load_span_map(global_audit.collate_mod.SPAN_JSONL)
    example_ids = global_audit.choose_span_example_ids(span_by_ex, args.sample_rows, args.sample_mode, args.sample_seed)
    examples = global_audit.read_examples_by_example_id(global_audit.collate_mod.TRAIN_100M, example_ids)
    tokenizer = global_audit.collate_mod.AutoTokenizer.from_pretrained(str(global_audit.collate_mod.TOKENIZER_DIR), use_fast=True)
    results = []
    for label in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
        if label not in global_audit.CHECKPOINTS:
            raise ValueError(label)
        print(json.dumps({"event": "checkpoint_start", "label": label, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        r = run_checkpoint(label, global_audit.CHECKPOINTS[label], examples, tokenizer, span_by_ex, args)
        results.append(r)
        print(json.dumps({"event": "checkpoint_done", "label": label, "status": r.get("status"), "pairs": r.get("n_pairs"), "rows": r.get("n_rows")}), flush=True)
    interpretation = [
        "This broad predictor diagnostic uses stable example_id sampling from the changed-row span map, avoiding the sparse/front-stream issue in the first predictor probe.",
        "A predictor is useful for pair-specificity only if its held-out true-minus-same-row-decoy margin or top1 exceeds the identity source residual. If identity is already stronger, pair-specific separation is already present in the residual geometry and the predictor is not justified as a correspondence extractor.",
        "This is still only representation evidence. A future trainer must decide whether gentle agreement improves downstream learning without erasing private details; it cannot claim expected score gain from this probe alone.",
    ]
    summary = {
        "status": "PREDICTOR_GLOBAL_DECOY_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Broad exact-changed-row test of whether a ridge source->rewrite predictor improves held-out true-vs-same-row-decoy separation over identity residuals.",
        "inputs": {
            "train_sha256": train_sha,
            "tokenizer_sha256": tok_sha,
            "span_examples": len(span_by_ex),
            "sample_rows_loaded": len(examples),
            "sample_mode": args.sample_mode,
            "center_mode": args.center_mode,
            "proj_dims": args.proj_dims_list,
            "train_frac": args.train_frac,
            "ridge_alpha": args.ridge_alpha,
            "splits": args.splits,
        },
        "checkpoint_results": results,
        "scientific_interpretation": interpretation,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "predictor_global_decoy_probe.json"
    out_md = out_dir / "predictor_global_decoy_probe.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research predictor global decoy probe",
        "",
        summary["purpose"],
        "",
        "CPU-only. No model update, official evaluation, corpus/tokenizer change, or H100 work.",
        "",
        "## Inputs",
        f"- changed rows loaded: `{len(examples)}`; center_mode `{args.center_mode}`; dims `{args.proj_dims}`",
        f"- ridge_alpha `{args.ridge_alpha}`, train_frac `{args.train_frac}`, splits `{args.splits}`",
        f"- train SHA: `{train_sha}`",
        f"- tokenizer SHA: `{tok_sha}`",
        "",
        "## Results",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"### {r.get('label')}: status `{r.get('status')}`")
            continue
        lines.append(f"### {r['label']} (pairs={r['n_pairs']}, rows={r['n_rows']})")
        for d in sorted(r["metrics_by_projection_dim"], key=lambda x: int(x)):
            m = r["metrics_by_projection_dim"][d]
            base = m["baseline_identity_all_pairs"]
            ridge = m["ridge_holdout"]["summary"]
            lines.append(
                f"- d{d} identity all: true-decoy-mean={base['true_minus_same_row_mean']['mean']:.4f}, "
                f"true-decoy-max={base['true_minus_same_row_max']['mean']:.4f}, top1={base['within_row_top1_acc']:.4f}"
            )
            lines.append(
                f"  ridge holdout: pred true-decoy-mean={ridge['predictor_true_minus_decoy_mean']['mean']}, "
                f"identity true-decoy-mean={ridge['identity_true_minus_decoy_mean']['mean']}, "
                f"pred top1={ridge['predictor_top1']['mean']}, identity top1={ridge['identity_top1']['mean']}, "
                f"adv_mean={ridge['predictor_advantage_true_minus_decoy_mean']['mean']}, adv_top1={ridge['predictor_advantage_top1']['mean']}"
            )
    lines += ["", "## Interpretation"]
    for item in interpretation:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"Full JSON: `{out_json}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "checkpoint_status": {r.get("label"): r.get("status") for r in results},
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
