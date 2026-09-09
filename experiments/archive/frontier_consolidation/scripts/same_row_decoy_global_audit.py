#!/usr/bin/env python3
"""research: Broad same-row decoy audit for source<->rewrite residual pair specificity.

CPU-only. No model update, official eval, corpus/tokenizer change, or H100 work.

Purpose: correct and stress-test the research interpretation.  research reported fields named
"pair-same-row-other margin"; those are actual_minus_same_row_other, not same-row-other
cosines.  If the margin is positive and large, same-row decoys are NOT closer than the true
rewrite; true source<->rewrite residuals are already pair-specific against matched same-row
hard negatives.

This script samples exact changed rows from the research pair-span map (not sparse front/stride
stream rows), reads the corresponding 100M stream rows, extracts row-centered residual span
vectors from actual token-mean legal checkpoints, and ranks true rewrites against same-row
decoys within each row.
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
OUT_DIR = WORKSPACE / "data/same_row_decoy_global_audit"
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


collate_mod = load_module("consistency_collate_step071_global", COLLATE_SCRIPT)
base = load_module("compact_experience_base_step071_global", BASE_TRAINER_PATH)


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "std": None, "p05": None, "p95": None}
    s = sorted(vals)
    def pct(q: float) -> float:
        if len(s) == 1:
            return float(s[0])
        pos = q * (len(s) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return float(s[lo])
        return float(s[lo] * (hi - pos) + s[hi] * (pos - lo))
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "std": float(statistics.pstdev(vals)),
        "p05": pct(0.05),
        "p95": pct(0.95),
    }


def choose_span_example_ids(span_by_ex: dict[int, dict[str, Any]], sample_rows: int, mode: str, seed: int) -> list[int]:
    """Choose changed examples by stable example_id, not by materialized stream row.

    The 100M stream is repeated/shuffled relative to the 10M pool, so span-map
    row_index_in_pool_1based is not a safe row number into TRAIN_100M.  The stable join key
    is example_id, exactly as the research collate path uses.
    """
    ids = sorted(int(k) for k in span_by_ex.keys())
    if sample_rows <= 0 or sample_rows >= len(ids):
        return ids
    if mode == "front":
        return ids[:sample_rows]
    if mode == "stride":
        return [ids[round(i * (len(ids) - 1) / (sample_rows - 1))] for i in range(sample_rows)] if sample_rows > 1 else [ids[0]]
    if mode == "random":
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        perm = torch.randperm(len(ids), generator=g).tolist()[:sample_rows]
        return sorted(ids[i] for i in perm)
    raise ValueError(mode)


def read_examples_by_example_id(path: pathlib.Path, example_ids: list[int]) -> list[Any]:
    """Read the first changed-source occurrence for each requested example_id from the 100M stream."""
    wanted = set(int(x) for x in example_ids)
    found: dict[int, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for row_1, line in enumerate(f, 1):
            obj = json.loads(line)
            exid = int(obj.get("example_id", row_1 - 1))
            if exid not in wanted or exid in found:
                continue
            if str(obj.get("source", "")) != collate_mod.CHANGED_SOURCE:
                continue
            text = str(obj["text"])
            found[exid] = collate_mod.StreamExample(
                text=text,
                words=int(obj.get("words", len(text.split()))),
                example_id=exid,
                source=str(obj.get("source", "")),
                global_row_1based=row_1,
            )
            if len(found) >= len(wanted):
                break
    examples = [found[eid] for eid in example_ids if eid in found]
    examples.sort(key=lambda x: x.global_row_1based)
    return examples


def pool_ranges(hidden: torch.Tensor, batch_row: int, ranges: list[list[int]]) -> torch.Tensor:
    chunks = [hidden[batch_row, int(a):int(b), :] for a, b in ranges]
    return torch.cat(chunks, dim=0).mean(dim=0)


def span_vec(hidden: torch.Tensor, attention_mask: torch.Tensor, rec: dict[str, Any], side: str, mode: str) -> torch.Tensor:
    br = int(rec["batch_row"])
    ranges = rec["source_ranges"] if side == "source" else rec["rewrite_ranges"]
    v = pool_ranges(hidden, br, ranges).float()
    if mode == "row_centered":
        valid = attention_mask[br].bool()
        row_mean = hidden[br, valid, :].mean(dim=0).detach().float()
        v = v - row_mean
    elif mode != "raw":
        raise ValueError(mode)
    return v


def make_projection(hidden_dim: int, proj_dim: int, seed: int) -> torch.Tensor:
    if proj_dim == hidden_dim:
        return torch.eye(hidden_dim, dtype=torch.float32)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    mat = torch.randn(hidden_dim, proj_dim, generator=g, dtype=torch.float32)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q[:, :proj_dim].contiguous()


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=1, eps=1e-6)


def mask_for_training(batch, tokenizer, state, gen):
    input_ids = batch["input_ids"][:, :collate_mod.SEQ_LEN].contiguous()
    attention_mask = batch["attention_mask"][:, :collate_mod.SEQ_LEN].contiguous()
    word_group = batch["word_group"][:, :collate_mod.SEQ_LEN].contiguous()
    masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, state, gen)
    return masked_inputs, labels, attention_mask


def collect_vectors(ckpt_path: pathlib.Path, examples, tokenizer, span_by_ex, args) -> dict[str, Any]:
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt_path))
    model.eval(); model.to("cpu")
    state = base.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15,
                                        switch_frac=0.7, amlm_window=10, amlm_lambda=0.2)
    state.initialize(vocab_size=len(tokenizer), total_steps=max(1, math.ceil(len(examples) / args.batch_size)))
    gen = torch.Generator(device="cpu"); gen.manual_seed(args.train_rng_seed)
    ds = collate_mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, collate_mod.SEQ_LEN)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                         collate_fn=collate_mod.consistency_collate)
    srcs_by_dim: dict[int, list[torch.Tensor]] = defaultdict(list)
    rews_by_dim: dict[int, list[torch.Tensor]] = defaultdict(list)
    row_keys: list[int] = []
    global_rows: list[int] = []
    pair_ids: list[str] = []
    hidden_dim_seen = None
    projections: dict[int, torch.Tensor] = {}
    row_key_map: dict[int, int] = {}
    row_counter = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            state.current_step = batch_index
            masked_inputs, labels, attention_mask = mask_for_training(batch, tokenizer, state, gen)
            aux = batch["aux_records"]
            if not aux:
                continue
            out = model(input_ids=masked_inputs, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1]
            hidden_dim = int(hidden.shape[-1])
            hidden_dim_seen = hidden_dim
            for d in args.proj_dims_list:
                if d not in projections:
                    projections[d] = make_projection(hidden_dim, d, args.proj_seed + d)
            for rec in aux:
                gr = int(rec["global_row_1based"])
                if gr not in row_key_map:
                    row_key_map[gr] = row_counter
                    row_counter += 1
                raw_s = span_vec(hidden, attention_mask, rec, "source", args.center_mode)
                raw_w = span_vec(hidden, attention_mask, rec, "rewrite", args.center_mode)
                for d, proj in projections.items():
                    srcs_by_dim[d].append(raw_s @ proj)
                    rews_by_dim[d].append(raw_w @ proj)
                row_keys.append(row_key_map[gr])
                global_rows.append(gr)
                pair_ids.append(str(rec.get("pair_id", "")))
    del model
    return {
        "hidden_dim": hidden_dim_seen,
        "row_keys": torch.tensor(row_keys, dtype=torch.long),
        "global_rows": global_rows,
        "pair_ids": pair_ids,
        "srcs_by_dim": {str(d): torch.stack(v) for d, v in srcs_by_dim.items()},
        "rews_by_dim": {str(d): torch.stack(v) for d, v in rews_by_dim.items()},
        "n_pairs": len(row_keys),
        "n_rows": row_counter,
    }


def decoy_metrics(S: torch.Tensor, W: torch.Tensor, row_keys: torch.Tensor) -> dict[str, Any]:
    Sn = normalize(S); Wn = normalize(W)
    n = Sn.shape[0]
    by_row: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        by_row[int(row_keys[i].item())].append(i)
    true_vals: list[float] = []
    decoy_vals: list[float] = []
    rand_vals: list[float] = []
    true_minus_row_mean: list[float] = []
    true_minus_row_max: list[float] = []
    ranks: list[int] = []
    top1 = 0
    hard_errors = []
    for i in range(n):
        r = int(row_keys[i].item())
        members = by_row[r]
        decoys = [j for j in members if j != i]
        rand_vals.append(float((Sn[i] * Wn[(i + 1) % n]).sum().item()))
        if not decoys:
            continue
        tv = float((Sn[i] * Wn[i]).sum().item())
        dvs = [float((Sn[i] * Wn[j]).sum().item()) for j in decoys]
        true_vals.append(tv)
        decoy_vals.extend(dvs)
        true_minus_row_mean.append(tv - statistics.mean(dvs))
        true_minus_row_max.append(tv - max(dvs))
        cands = [tv] + dvs
        order = sorted(range(len(cands)), key=lambda k: -cands[k])
        rank = order.index(0) + 1
        ranks.append(rank)
        if rank == 1:
            top1 += 1
        elif len(hard_errors) < 12:
            hard_errors.append({"i": i, "rank": rank, "true": tv, "max_decoy": max(dvs), "row_size": len(members)})
    return {
        "n_pairs_with_decoys": len(true_vals),
        "n_decoy_comparisons": len(decoy_vals),
        "true_cos": summarize(true_vals),
        "same_row_decoy_cos": summarize(decoy_vals),
        "random_other_cos": summarize(rand_vals),
        "true_minus_same_row_mean": summarize(true_minus_row_mean),
        "true_minus_same_row_max": summarize(true_minus_row_max),
        "within_row_top1_acc": (top1 / len(ranks)) if ranks else None,
        "within_row_mean_rank": statistics.mean(ranks) if ranks else None,
        "hard_errors_sample": hard_errors,
    }


def run_checkpoint(label: str, ckpt_path: pathlib.Path, examples, tokenizer, span_by_ex, args) -> dict[str, Any]:
    if not ckpt_path.exists():
        return {"label": label, "status": "missing", "path": str(ckpt_path)}
    print(json.dumps({"event": "collect_vectors_start", "label": label, "examples": len(examples)}), flush=True)
    data = collect_vectors(ckpt_path, examples, tokenizer, span_by_ex, args)
    print(json.dumps({"event": "collect_vectors_done", "label": label, "pairs": data["n_pairs"], "rows": data["n_rows"]}), flush=True)
    by_dim = {}
    for d in args.proj_dims_list:
        S = data["srcs_by_dim"][str(d)]
        W = data["rews_by_dim"][str(d)]
        by_dim[str(d)] = decoy_metrics(S, W, data["row_keys"])
    return {
        "label": label,
        "status": "ok",
        "path": str(ckpt_path),
        "hidden_dim": data["hidden_dim"],
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
    args = ap.parse_args()
    args.proj_dims_list = [int(x) for x in args.proj_dims.split(",") if x.strip()]
    t0 = time.time()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_sha = collate_mod.sha256_file(collate_mod.TRAIN_100M)
    tok_sha = collate_mod.sha256_file(collate_mod.TOKENIZER_DIR / "tokenizer.json")
    if train_sha != collate_mod.EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {train_sha}")
    if tok_sha != collate_mod.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    span_by_ex = collate_mod.load_span_map(collate_mod.SPAN_JSONL)
    example_ids = choose_span_example_ids(span_by_ex, args.sample_rows, args.sample_mode, args.sample_seed)
    examples = read_examples_by_example_id(collate_mod.TRAIN_100M, example_ids)
    tokenizer = collate_mod.AutoTokenizer.from_pretrained(str(collate_mod.TOKENIZER_DIR), use_fast=True)
    ckpt_labels = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    results = []
    for label in ckpt_labels:
        if label not in CHECKPOINTS:
            raise ValueError(f"unknown checkpoint {label}")
        print(json.dumps({"event": "checkpoint_start", "label": label, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        r = run_checkpoint(label, CHECKPOINTS[label], examples, tokenizer, span_by_ex, args)
        results.append(r)
        print(json.dumps({"event": "checkpoint_done", "label": label, "status": r.get("status")}), flush=True)
    interpretation = [
        "The printed research 'pair-same-row-other margin' is actual_minus_same_row_other, not the same-row-other cosine itself. A positive value means true source<->rewrite pairs are closer than same-row decoys.",
        "This audit uses exact changed rows from the pair-span map, so same-row decoys are available for nearly every pair and are matched by topic/row.",
        "If within_row_top1 is near 1 and true_minus_same_row_max is strongly positive, the collapse story that positive-only consistency merely learns generic row/topic agreement is not supported by representation geometry.",
        "Even if collapse-to-row is not supported, this is not BabyLM score evidence. The remaining question is whether forcing already-strong pair-specific residual agreement improves downstream learning without erasing private details; a proposed comparison conditional on weak minfreq50 results.",
    ]
    summary = {
        "status": "SAME_ROW_DECOY_GLOBAL_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Broad changed-row audit of true source<->rewrite residual similarity versus same-row decoys, correcting the research margin interpretation.",
        "inputs": {
            "train_100m": str(collate_mod.TRAIN_100M),
            "train_sha256": train_sha,
            "tokenizer_sha256": tok_sha,
            "span_examples": len(span_by_ex),
            "sample_mode": args.sample_mode,
            "sample_rows_requested": args.sample_rows,
            "sample_rows_loaded": len(examples),
            "requested_example_ids_first": example_ids[:5],
            "requested_example_ids_last": example_ids[-5:],
            "loaded_global_rows_first": [int(x.global_row_1based) for x in examples[:5]],
            "loaded_global_rows_last": [int(x.global_row_1based) for x in examples[-5:]],
            "center_mode": args.center_mode,
            "proj_dims": args.proj_dims_list,
            "batch_size": args.batch_size,
            "checkpoints": ckpt_labels,
        },
        "checkpoint_results": results,
        "scientific_interpretation": interpretation,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "same_row_decoy_global_audit.json"
    out_md = out_dir / "same_row_decoy_global_audit.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research same-row decoy global audit",
        "",
        summary["purpose"],
        "",
        "CPU-only. No model update, official evaluation, corpus/tokenizer change, or H100 work.",
        "",
        "## Inputs",
        f"- changed rows loaded: `{len(examples)}` of `{len(span_by_ex)}` span-map rows, sample mode `{args.sample_mode}`",
        f"- center_mode `{args.center_mode}`, projection dims `{args.proj_dims}`",
        f"- train SHA: `{train_sha}`",
        f"- tokenizer SHA: `{tok_sha}`",
        "",
        "## Per-checkpoint / per-projection results",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"### {r.get('label')}: status `{r.get('status')}`")
            continue
        lines.append(f"### {r['label']}  (pairs={r['n_pairs']}, rows={r['n_rows']}, mode={r['center_mode']})")
        for d in sorted(r["metrics_by_projection_dim"], key=lambda x: int(x)):
            m = r["metrics_by_projection_dim"][d]
            lines.append(
                f"- d{d}: true_cos={m['true_cos']['mean']:.4f}, same_row_decoy_cos={m['same_row_decoy_cos']['mean']:.4f}, "
                f"random_other_cos={m['random_other_cos']['mean']:.4f}, true-decoy-mean={m['true_minus_same_row_mean']['mean']:.4f}, "
                f"true-decoy-max={m['true_minus_same_row_max']['mean']:.4f}, top1={m['within_row_top1_acc']:.4f}, mean_rank={m['within_row_mean_rank']:.3f}"
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
