#!/usr/bin/env python3
"""research: Is source<->rewrite pair-specific structure recoverable by an ASYMMETRIC
predictor that a symmetric agreement loss would collapse on?

CPU-only.  No model update, official evaluation, corpus/tokenizer change, or H100 work.

Motivation:
  research found that in row-centered residual space the model already encodes strong
  shared-row/topic residual (same-row-other margin > true-pair shuffle margin).  A naive
  positive-only symmetric agreement loss is therefore maximized by encoding row/topic
  identity, not per-proposition source<->rewrite correspondence -- it could collapse to a
  topic embedding and add nothing.

  The decisive question for whether a collapse-resistant construction is worth
  building: does a LEARNABLE ASYMMETRIC predictor (closed-form ridge W mapping source
  residual -> rewrite residual) recover pair-SPECIFIC structure -- i.e. after W, does the
  true rewrite beat SAME-ROW decoy rewrites -- on held-out pairs?

  - If W-predicted source beats same-row decoys on held-out data (out-of-sample), then
    genuine pair-specific correspondence exists that a symmetric loss cannot exploit but a
    predictor-based / shared-private objective can.  The construction has real headroom.
  - If W cannot beat same-row decoys out-of-sample (only in-sample by overfit), then the
    residual carries no recoverable pair-specific signal and the consistency route is weak.

  Same-row decoys provide the following diagnostic: for each source, the
  hard negatives are the OTHER rewrites from the same pool row (same topic, different
  proposition).

This probe reuses the research/research collate machinery, the frozen 100M reinvest stream,
the research legal tokenizer, the research pair-span map, and the actual token-mean checkpoints.
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
from transformers import AutoModelForMaskedLM


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
COLLATE_SCRIPT = WORKSPACE / "scripts/consistency_collate_smoke.py"
BASE_TRAINER_PATH = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
OUT_DIR = WORKSPACE / "data/pair_specificity_predictor_probe"
CHECKPOINTS = {
    "tokenmean_20M": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_20M",
    "tokenmean_80M": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
    "tokenmean_100M": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M",
}


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


collate_mod = load_module("consistency_collate_step071", COLLATE_SCRIPT)
base = load_module("compact_experience_base_step071", BASE_TRAINER_PATH)


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


def pool_ranges(hidden: torch.Tensor, batch_row: int, ranges: list[list[int]]) -> torch.Tensor:
    chunks = [hidden[batch_row, int(a):int(b), :] for a, b in ranges]
    return torch.cat(chunks, dim=0).mean(dim=0)


def span_residual(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    rec: dict[str, Any],
    side: str,
    center_mode: str,
) -> torch.Tensor:
    br = int(rec["batch_row"])
    ranges = rec["source_ranges"] if side == "source" else rec["rewrite_ranges"]
    v = pool_ranges(hidden, br, ranges)
    if center_mode == "row_centered":
        valid = attention_mask[br].bool()
        row_mean = hidden[br, valid, :].mean(dim=0).detach()
        v = v - row_mean
    elif center_mode != "raw":
        raise ValueError(center_mode)
    return v


def mask_for_training(batch, tokenizer, state, gen):
    input_ids = batch["input_ids"][:, :collate_mod.SEQ_LEN].contiguous()
    attention_mask = batch["attention_mask"][:, :collate_mod.SEQ_LEN].contiguous()
    word_group = batch["word_group"][:, :collate_mod.SEQ_LEN].contiguous()
    masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, state, gen)
    return masked_inputs, labels, attention_mask


def extract_residuals(label, ckpt_path, examples, tokenizer, span_by_ex, args):
    """Return per-pair row-centered source/rewrite residual vectors with row keys."""
    if not ckpt_path.exists():
        return None
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt_path))
    model.eval()
    model.to("cpu")
    state = base.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15,
                                        switch_frac=0.7, amlm_window=10, amlm_lambda=0.2)
    state.initialize(vocab_size=len(tokenizer), total_steps=max(1, math.ceil(len(examples) / args.batch_size)))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.train_rng_seed)
    ds = collate_mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, collate_mod.SEQ_LEN)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                         collate_fn=collate_mod.consistency_collate)
    srcs: list[torch.Tensor] = []
    rews: list[torch.Tensor] = []
    row_key: list[int] = []           # unique per pool row (batch_index, batch_row)
    ex_id: list[int] = []
    hidden_size = None
    row_counter = 0
    seen_rows: dict[tuple[int, int], int] = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            aux = batch["aux_records"]
            state.current_step = batch_index
            masked_inputs, labels, attention_mask = mask_for_training(batch, tokenizer, state, gen)
            if not aux:
                continue
            out = model(input_ids=masked_inputs, attention_mask=attention_mask,
                        output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1]
            hidden_size = int(hidden.shape[-1])
            for r in aux:
                s = span_residual(hidden, attention_mask, r, "source", args.center_mode).float()
                w = span_residual(hidden, attention_mask, r, "rewrite", args.center_mode).float()
                key = (batch_index, int(r["batch_row"]))
                if key not in seen_rows:
                    seen_rows[key] = row_counter
                    row_counter += 1
                srcs.append(s)
                rews.append(w)
                row_key.append(seen_rows[key])
                ex_id.append(int(r["example_id"]))
    del model
    if not srcs:
        return None
    S = torch.stack(srcs)   # [N, H]  raw residual (not normalized) for ridge
    W = torch.stack(rews)   # [N, H]
    return {
        "label": label,
        "path": str(ckpt_path),
        "hidden_size": hidden_size,
        "S": S,
        "W": W,
        "row_key": torch.tensor(row_key, dtype=torch.long),
        "ex_id": ex_id,
        "n_pairs": S.shape[0],
        "n_rows": row_counter,
    }


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=1, eps=1e-6)


def same_row_decoy_diag(S: torch.Tensor, W: torch.Tensor, row_key: torch.Tensor) -> dict[str, Any]:
    """Symmetric-agreement baseline: for each source i, cosine to true W[i], to random-other
    rewrite, and to SAME-ROW-other rewrites.  Also retrieval rank of the true rewrite among
    same-row candidates.  This is what a naive positive-only agreement loss rewards."""
    Sn = normalize_rows(S)
    Wn = normalize_rows(W)
    n = Sn.shape[0]
    true_cos = (Sn * Wn).sum(dim=1)  # [N]
    by_row: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        by_row[int(row_key[i].item())].append(i)
    true_vals, decoy_vals, rand_vals = [], [], []
    rank1_within_row, ranks, row_group_sizes = 0, [], []
    for i in range(n):
        r = int(row_key[i].item())
        members = by_row[r]
        # same-row decoys: rewrites from the same row that are NOT the true partner
        decoys = [j for j in members if j != i]
        rand_j = (i + 1) % n
        rand_vals.append(float((Sn[i] * Wn[rand_j]).sum().item()))
        if not decoys:
            continue
        row_group_sizes.append(len(members))
        dvals = [float((Sn[i] * Wn[j]).sum().item()) for j in decoys]
        decoy_vals.extend(dvals)
        true_vals.append(float(true_cos[i].item()))
        # rank of true among {true} U decoys
        cands = [float(true_cos[i].item())] + dvals
        order = sorted(range(len(cands)), key=lambda k: -cands[k])
        rank = order.index(0) + 1  # 1-based rank of the true (index 0)
        ranks.append(rank)
        if rank == 1:
            rank1_within_row += 1
    mean_true = statistics.mean(true_vals) if true_vals else None
    mean_decoy = statistics.mean(decoy_vals) if decoy_vals else None
    mean_rand = statistics.mean(rand_vals) if rand_vals else None
    return {
        "n_pairs_with_decoys": len(true_vals),
        "true_cos": summarize(true_vals),
        "same_row_decoy_cos": summarize(decoy_vals),
        "random_other_cos": summarize(rand_vals),
        "true_minus_same_row_decoy": (mean_true - mean_decoy) if (mean_true is not None and mean_decoy is not None) else None,
        "true_minus_random_other": (mean_true - mean_rand) if (mean_true is not None and mean_rand is not None) else None,
        "within_row_top1_acc": (rank1_within_row / len(ranks)) if ranks else None,
        "within_row_mean_rank": (statistics.mean(ranks) if ranks else None),
        "within_row_group_size": summarize([float(x) for x in row_group_sizes]),
    }


def ridge_predictor_test(data: dict[str, Any], args, rng: torch.Generator) -> dict[str, Any]:
    """Learnable asymmetric predictor: closed-form ridge W_hat = argmin ||S W_hat - W||^2 + a||W_hat||^2.
    Train on a split of ROWS (so held-out rows never share a row with train), then evaluate on
    held-out rows whether the PREDICTED target beats same-row decoys."""
    S = data["S"]
    W = data["W"]
    row_key = data["row_key"]
    unique_rows = sorted(set(int(r.item()) for r in row_key))
    n_rows = len(unique_rows)
    if n_rows < 8:
        return {"status": "too_few_rows", "n_rows": n_rows}
    perm = torch.randperm(n_rows, generator=rng).tolist()
    n_train = max(4, int(round(n_rows * args.train_frac)))
    train_rows = set(unique_rows[perm[i]] for i in range(n_train))
    train_mask = torch.tensor([int(r.item()) in train_rows for r in row_key])
    test_mask = ~train_mask
    if int(test_mask.sum().item()) < 4:
        return {"status": "too_few_test", "n_rows": n_rows}
    S_tr, W_tr = S[train_mask], W[train_mask]
    S_te, W_te = S[test_mask], W[test_mask]
    rk_te = row_key[test_mask]
    H = S.shape[1]
    # center residuals with train mean (predictor operates on centered features)
    s_mean = S_tr.mean(dim=0, keepdim=True)
    w_mean = W_tr.mean(dim=0, keepdim=True)
    Xtr = S_tr - s_mean
    Ytr = W_tr - w_mean
    A = Xtr.t() @ Xtr + args.ridge_alpha * torch.eye(H)
    B = Xtr.t() @ Ytr
    What = torch.linalg.solve(A, B)  # [H, H]
    # predictions
    def predict(Xc):
        return Xc @ What + w_mean
    P_te = predict(S_te - s_mean)          # predicted rewrite for held-out sources
    # baseline: identity (no predictor) = just the source residual as its own predictor of rewrite
    P_id = S_te                            # treat source residual directly
    Wn_te = normalize_rows(W_te)
    Pn_te = normalize_rows(P_te)
    Pn_id = normalize_rows(P_id)
    n = S_te.shape[0]
    by_row: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        by_row[int(rk_te[i].item())].append(i)

    def within_row_metrics(Pn):
        true_vals, decoy_vals, ranks, top1 = [], [], [], 0
        for i in range(n):
            r = int(rk_te[i].item())
            members = by_row[r]
            decoys = [j for j in members if j != i]
            if not decoys:
                continue
            tv = float((Pn[i] * Wn_te[i]).sum().item())
            dv = [float((Pn[i] * Wn_te[j]).sum().item()) for j in decoys]
            true_vals.append(tv)
            decoy_vals.extend(dv)
            cands = [tv] + dv
            order = sorted(range(len(cands)), key=lambda k: -cands[k])
            rank = order.index(0) + 1
            ranks.append(rank)
            if rank == 1:
                top1 += 1
        mt = statistics.mean(true_vals) if true_vals else None
        md = statistics.mean(decoy_vals) if decoy_vals else None
        return {
            "n_pairs_with_decoys": len(true_vals),
            "pred_true_cos": summarize(true_vals),
            "pred_same_row_decoy_cos": summarize(decoy_vals),
            "true_minus_same_row_decoy": (mt - md) if (mt is not None and md is not None) else None,
            "within_row_top1_acc": (top1 / len(ranks)) if ranks else None,
            "within_row_mean_rank": (statistics.mean(ranks) if ranks else None),
        }

    pred_metrics = within_row_metrics(Pn_te)
    id_metrics = within_row_metrics(Pn_id)
    return {
        "status": "ok",
        "n_rows": n_rows,
        "n_train_rows": n_train,
        "n_train_pairs": int(train_mask.sum().item()),
        "n_test_pairs": int(test_mask.sum().item()),
        "ridge_alpha": args.ridge_alpha,
        "hidden_size": H,
        "predictor_holdout": pred_metrics,
        "identity_holdout": id_metrics,
        "predictor_advantage_true_minus_decoy": (
            (pred_metrics["true_minus_same_row_decoy"] - id_metrics["true_minus_same_row_decoy"])
            if (pred_metrics.get("true_minus_same_row_decoy") is not None
                and id_metrics.get("true_minus_same_row_decoy") is not None) else None
        ),
        "predictor_advantage_top1": (
            (pred_metrics["within_row_top1_acc"] - id_metrics["within_row_top1_acc"])
            if (pred_metrics.get("within_row_top1_acc") is not None
                and id_metrics.get("within_row_top1_acc") is not None) else None
        ),
    }


def run_checkpoint(label, ckpt_path, examples, tokenizer, span_by_ex, args):
    data = extract_residuals(label, ckpt_path, examples, tokenizer, span_by_ex, args)
    if data is None:
        return {"label": label, "status": "missing_or_empty", "path": str(ckpt_path)}
    diag = same_row_decoy_diag(data["S"], data["W"], data["row_key"])
    # average ridge test over several row splits for stability
    ridge_runs = []
    for k in range(args.ridge_splits):
        rng = torch.Generator(device="cpu")
        rng.manual_seed(args.split_seed + k)
        ridge_runs.append(ridge_predictor_test(data, args, rng))
    ok_runs = [r for r in ridge_runs if r.get("status") == "ok"]
    def agg(field_path):
        vals = []
        for r in ok_runs:
            cur = r
            for p in field_path:
                cur = cur.get(p) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return summarize(vals)
    ridge_summary = {
        "n_ok_splits": len(ok_runs),
        "predictor_holdout_true_minus_decoy": agg(["predictor_holdout", "true_minus_same_row_decoy"]),
        "identity_holdout_true_minus_decoy": agg(["identity_holdout", "true_minus_same_row_decoy"]),
        "predictor_holdout_top1": agg(["predictor_holdout", "within_row_top1_acc"]),
        "identity_holdout_top1": agg(["identity_holdout", "within_row_top1_acc"]),
        "predictor_advantage_true_minus_decoy": agg(["predictor_advantage_true_minus_decoy"]),
        "predictor_advantage_top1": agg(["predictor_advantage_top1"]),
    }
    return {
        "label": label,
        "status": "ok",
        "path": str(ckpt_path),
        "hidden_size": data["hidden_size"],
        "n_pairs": data["n_pairs"],
        "n_rows": data["n_rows"],
        "center_mode": args.center_mode,
        "symmetric_agreement_diag": diag,
        "ridge_predictor_summary": ridge_summary,
        "ridge_runs": ridge_runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--sample-mode", choices=["front", "stride"], default="front")
    ap.add_argument("--sample-rows", type=int, default=960)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=30)
    ap.add_argument("--train-rng-seed", type=int, default=43023)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--center-mode", choices=["raw", "row_centered"], default="row_centered")
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--ridge-splits", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=71071)
    ap.add_argument("--checkpoints", default="tokenmean_80M,tokenmean_100M")
    args = ap.parse_args()
    t0 = time.time()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_sha = collate_mod.sha256_file(collate_mod.TRAIN_100M)
    tok_sha = collate_mod.sha256_file(collate_mod.TOKENIZER_DIR / "tokenizer.json")
    if train_sha != collate_mod.EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {train_sha}")
    if tok_sha != collate_mod.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    span_by_ex = collate_mod.load_span_map(collate_mod.SPAN_JSONL)
    examples = collate_mod.read_stream_sample(collate_mod.TRAIN_100M, args.sample_mode, args.sample_rows)
    tokenizer = collate_mod.AutoTokenizer.from_pretrained(str(collate_mod.TOKENIZER_DIR), use_fast=True)
    ckpt_labels = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    results = []
    for label in ckpt_labels:
        if label not in CHECKPOINTS:
            raise ValueError(f"unknown checkpoint {label}")
        print(json.dumps({"event": "ckpt_start", "label": label,
                          "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        r = run_checkpoint(label, CHECKPOINTS[label], examples, tokenizer, span_by_ex, args)
        results.append(r)
        print(json.dumps({"event": "ckpt_done", "label": label, "status": r.get("status"),
                          "n_pairs": r.get("n_pairs"), "n_rows": r.get("n_rows")}), flush=True)
    interpretation = [
        "Symmetric agreement diag: if true_minus_same_row_decoy <= 0 in row-centered space, a "
        "naive positive-only symmetric agreement loss is maximized by encoding row/topic identity, "
        "not pair-specific correspondence -- collapse risk confirmed.",
        "Ridge predictor holdout: predictor_holdout_true_minus_decoy is the out-of-sample separation "
        "an asymmetric learnable predictor achieves against SAME-ROW decoys. If it is clearly > 0 and "
        "> the identity_holdout value, genuine pair-specific structure is recoverable and a "
        "predictor-based / shared-private construction has real headroom.",
        "If the predictor cannot beat same-row decoys out-of-sample, source-view consistency lacks a "
        "recoverable pair-specific signal on this substrate and should not be the successor GPU screen.",
        "This is representation-geometry evidence, not BabyLM task-score evidence, and does not by "
        "itself authorize any GPU run.",
    ]
    summary = {
        "status": "PAIR_SPECIFICITY_PREDICTOR_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Test whether an asymmetric learnable predictor recovers pair-specific source<->rewrite "
                   "structure that a symmetric agreement loss would collapse on, using same-row decoys.",
        "inputs": {
            "train_100m": str(collate_mod.TRAIN_100M),
            "train_sha256": train_sha,
            "tokenizer_sha256": tok_sha,
            "span_examples": len(span_by_ex),
            "sample_rows_loaded": len(examples),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "center_mode": args.center_mode,
            "ridge_alpha": args.ridge_alpha,
            "train_frac": args.train_frac,
            "ridge_splits": args.ridge_splits,
            "checkpoints": ckpt_labels,
        },
        "checkpoint_results": results,
        "scientific_interpretation": interpretation,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "pair_specificity_predictor_probe.json"
    out_md = out_dir / "pair_specificity_predictor_probe.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research pair-specificity predictor probe",
        "",
        summary["purpose"],
        "",
        "CPU-only. No model update, official evaluation, corpus/tokenizer change, or H100 work.",
        "",
        "## Inputs",
        f"- center_mode: `{args.center_mode}`; ridge_alpha `{args.ridge_alpha}`; train_frac `{args.train_frac}`; splits `{args.ridge_splits}`",
        f"- sample: `{len(examples)}` stream rows, batch `{args.batch_size}`, max batches `{args.max_batches}`",
        f"- train SHA: `{train_sha}`",
        f"- tokenizer SHA: `{tok_sha}`",
        "",
        "## Per-checkpoint results",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"### {r.get('label')}: status `{r.get('status')}`")
            continue
        d = r["symmetric_agreement_diag"]
        rs = r["ridge_predictor_summary"]
        lines.append(f"### {r['label']}  (pairs={r['n_pairs']}, rows={r['n_rows']}, mode={r['center_mode']})")
        lines.append(f"- symmetric: true_cos={d['true_cos']['mean']:.4f}, same_row_decoy_cos={d['same_row_decoy_cos']['mean']:.4f}, "
                     f"true-minus-decoy={d['true_minus_same_row_decoy']:.4f}, within_row_top1={d['within_row_top1_acc']:.4f}, "
                     f"mean_rank={d['within_row_mean_rank']:.3f} (group size mean={d['within_row_group_size']['mean']:.2f})")
        lines.append(f"- ridge predictor holdout: true-minus-decoy={rs['predictor_holdout_true_minus_decoy']['mean']}, "
                     f"top1={rs['predictor_holdout_top1']['mean']}")
        lines.append(f"- identity holdout:        true-minus-decoy={rs['identity_holdout_true_minus_decoy']['mean']}, "
                     f"top1={rs['identity_holdout_top1']['mean']}")
        lines.append(f"- predictor advantage: true-minus-decoy={rs['predictor_advantage_true_minus_decoy']['mean']}, "
                     f"top1={rs['predictor_advantage_top1']['mean']} (n_ok_splits={rs['n_ok_splits']})")
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
