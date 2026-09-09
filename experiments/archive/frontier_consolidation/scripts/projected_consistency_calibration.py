#!/usr/bin/env python3
"""research: calibrate fixed-subspace source-view consistency on actual checkpoints.

This is CPU-only.  It does not update model parameters, start training, read evaluation
examples, or change the corpus/tokenizer.  It measures whether a future source/rewrite
auxiliary loss can be restricted to a small fixed hidden subspace and row-centered
vectors, instead of forcing the whole hidden vector of already-close paired views.
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
OUT_DIR = WORKSPACE / "data/projected_consistency_calibration"
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


collate_mod = load_module("consistency_collate_step070", COLLATE_SCRIPT)
base = load_module("compact_experience_base_step070_projected_consistency", BASE_TRAINER_PATH)


def summarize(vals: list[float]) -> dict[str, float | int | None]:
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


def grad_l2(grad: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    g = grad.detach().float()
    if mask is not None:
        m = mask.bool()
        if m.dim() == 2:
            m = m.unsqueeze(-1).expand_as(g)
        g = g[m]
    if g.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(g).item())


def cosine_flat(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float | None:
    x = a.detach().float()
    y = b.detach().float()
    if mask is not None:
        m = mask.bool()
        if m.dim() == 2:
            m = m.unsqueeze(-1).expand_as(x)
        x = x[m]
        y = y[m]
    if x.numel() == 0:
        return None
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) == 0.0:
        return None
    return float((x.flatten() @ y.flatten()).item() / float(denom.item()))


def pool_ranges(hidden: torch.Tensor, batch_row: int, ranges: list[list[int]]) -> torch.Tensor:
    chunks = [hidden[batch_row, int(a):int(b), :] for a, b in ranges]
    return torch.cat(chunks, dim=0).mean(dim=0)


def aux_token_mask(shape: tuple[int, int], aux_records: list[dict[str, Any]]) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool)
    for r in aux_records:
        br = int(r["batch_row"])
        for a, b in r["source_ranges"] + r["rewrite_ranges"]:
            mask[br, int(a):int(b)] = True
    return mask


def make_projection(dim_hidden: int, dim_proj: int, seed: int) -> torch.Tensor:
    if dim_proj == dim_hidden:
        return torch.eye(dim_hidden, dtype=torch.float32)
    if dim_proj <= 0 or dim_proj > dim_hidden:
        raise ValueError(f"bad projection dimension {dim_proj} for hidden {dim_hidden}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    mat = torch.randn(dim_hidden, dim_proj, generator=gen, dtype=torch.float32)
    # QR gives fixed orthonormal columns; sign is deterministic after torch QR for fixed seed.
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q[:, :dim_proj].contiguous()


def span_vector(
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


def projected_pair_vectors(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    aux_records: list[dict[str, Any]],
    proj: torch.Tensor,
    center_mode: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int]]]:
    srcs: list[torch.Tensor] = []
    rews: list[torch.Tensor] = []
    row_keys: list[tuple[int, int]] = []
    for r in aux_records:
        src = span_vector(hidden, attention_mask, r, "source", center_mode).float() @ proj
        rew = span_vector(hidden, attention_mask, r, "rewrite", center_mode).float() @ proj
        srcs.append(torch.nn.functional.normalize(src, dim=0, eps=1e-6))
        rews.append(torch.nn.functional.normalize(rew, dim=0, eps=1e-6))
        row_keys.append((int(r["batch_row"]), int(r["example_id"])))
    return srcs, rews, row_keys


def row_normalized_projected_loss(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    aux_records: list[dict[str, Any]],
    proj: torch.Tensor,
    center_mode: str,
) -> torch.Tensor:
    row_losses: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
    for r in aux_records:
        br = int(r["batch_row"])
        src = span_vector(hidden, attention_mask, r, "source", center_mode).float() @ proj
        rew = span_vector(hidden, attention_mask, r, "rewrite", center_mode).float() @ proj
        src_n = torch.nn.functional.normalize(src, dim=0, eps=1e-6)
        rew_n = torch.nn.functional.normalize(rew, dim=0, eps=1e-6)
        cos_sr = (src_n * rew_n.detach()).sum()
        cos_rs = (rew_n * src_n.detach()).sum()
        row_losses[(br, int(r["example_id"]))].append(0.5 * ((1.0 - cos_sr) + (1.0 - cos_rs)))
    if not row_losses:
        return hidden.sum() * 0.0
    return torch.stack([torch.stack(v).mean() for v in row_losses.values()]).mean()


def geometry_for_mode(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    aux_records: list[dict[str, Any]],
    proj: torch.Tensor,
    center_mode: str,
) -> dict[str, Any]:
    if not aux_records:
        return {"n": 0}
    srcs, rews, row_keys = projected_pair_vectors(hidden.detach(), attention_mask, aux_records, proj, center_mode)
    n = len(srcs)
    actual = [float((srcs[i] * rews[i]).sum().item()) for i in range(n)]
    shuffled = [float((srcs[i] * rews[(i + 1) % n]).sum().item()) for i in range(n)] if n > 1 else []
    same_row_other: list[float] = []
    by_row: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(row_keys):
        by_row[key].append(i)
    for idxs in by_row.values():
        if len(idxs) < 2:
            continue
        for j, i in enumerate(idxs):
            other = idxs[(j + 1) % len(idxs)]
            if other != i:
                same_row_other.append(float((srcs[i] * rews[other]).sum().item()))
    mean_actual = statistics.mean(actual) if actual else None
    mean_shuffled = statistics.mean(shuffled) if shuffled else None
    mean_other = statistics.mean(same_row_other) if same_row_other else None
    return {
        "n": n,
        "actual_cos": summarize(actual),
        "shuffled_cos": summarize(shuffled),
        "same_row_other_cos": summarize(same_row_other),
        "actual_minus_shuffled_mean": (float(mean_actual - mean_shuffled) if mean_actual is not None and mean_shuffled is not None else None),
        "actual_minus_same_row_other_mean": (float(mean_actual - mean_other) if mean_actual is not None and mean_other is not None else None),
    }


def mask_for_training(batch: dict[str, Any], tokenizer, state, gen) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"][:, :collate_mod.SEQ_LEN].contiguous()
    attention_mask = batch["attention_mask"][:, :collate_mod.SEQ_LEN].contiguous()
    word_group = batch["word_group"][:, :collate_mod.SEQ_LEN].contiguous()
    masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, state, gen)
    return masked_inputs, labels, attention_mask, word_group


def run_checkpoint(label: str, ckpt_path: pathlib.Path, examples, tokenizer, span_by_ex, args) -> dict[str, Any]:
    if not ckpt_path.exists():
        return {"label": label, "status": "missing", "path": str(ckpt_path)}
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt_path))
    model.eval()
    model.to("cpu")
    state = base.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15, switch_frac=0.7, amlm_window=10, amlm_lambda=0.2)
    state.initialize(vocab_size=len(tokenizer), total_steps=max(1, math.ceil(len(examples) / args.batch_size)))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.train_rng_seed)
    ds = collate_mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, collate_mod.SEQ_LEN)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_mod.consistency_collate)
    proj_dims = [int(x) for x in args.proj_dims.split(",") if x.strip()]
    center_modes = [x.strip() for x in args.center_modes.split(",") if x.strip()]
    projections: dict[int, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    hidden_size_seen = None
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        aux = batch["aux_records"]
        state.current_step = batch_index
        masked_inputs, labels, attention_mask, _word_group = mask_for_training(batch, tokenizer, state, gen)
        if not aux:
            records.append({"batch_index": batch_index, "aux_records": 0, "skipped": "no_aux"})
            continue
        out = model(input_ids=masked_inputs, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        logits = out.logits
        hidden = out.hidden_states[-1]
        hidden_size = int(hidden.shape[-1])
        hidden_size_seen = hidden_size
        for d in proj_dims:
            if d not in projections:
                projections[d] = make_projection(hidden_size, d, args.proj_seed + d)
        mlm_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        mlm_grad = torch.autograd.grad(mlm_loss, hidden, retain_graph=True, allow_unused=True)[0]
        if mlm_grad is None:
            raise RuntimeError(f"MLM hidden gradient unavailable for {label}")
        masked_pos = labels != -100
        aux_pos = aux_token_mask(tuple(labels.shape), aux)
        mlm_l2_all = grad_l2(mlm_grad)
        mlm_l2_aux = grad_l2(mlm_grad, aux_pos)
        mode_records: list[dict[str, Any]] = []
        for mode in center_modes:
            for d in proj_dims:
                proj = projections[d]
                aux_loss = row_normalized_projected_loss(hidden, attention_mask, aux, proj, mode)
                aux_grad = torch.autograd.grad(aux_loss, hidden, retain_graph=True, allow_unused=True)[0]
                if aux_grad is None:
                    raise RuntimeError(f"aux hidden gradient unavailable for {label} {mode} d{d}")
                aux_l2_all = grad_l2(aux_grad)
                aux_l2_aux = grad_l2(aux_grad, aux_pos)
                geom = geometry_for_mode(hidden, attention_mask, aux, proj, mode)
                mode_records.append({
                    "center_mode": mode,
                    "proj_dim": d,
                    "aux_loss": float(aux_loss.detach().item()),
                    "aux_over_mlm_l2_all_hidden_lambda1": (float(aux_l2_all) / float(mlm_l2_all)) if mlm_l2_all else None,
                    "aux_over_mlm_l2_aux_spans_lambda1": (float(aux_l2_aux) / float(mlm_l2_aux)) if mlm_l2_aux else None,
                    "grad_cosine_all_hidden": cosine_flat(mlm_grad, aux_grad),
                    "grad_cosine_aux_spans": cosine_flat(mlm_grad, aux_grad, aux_pos),
                    "geometry": geom,
                })
        row_ids = {(int(r["batch_row"]), int(r["example_id"])) for r in aux}
        records.append({
            "batch_index": batch_index,
            "rows": int(labels.shape[0]),
            "tokens_per_row": int(labels.shape[1]),
            "aux_rows": len(row_ids),
            "aux_records": len(aux),
            "selected_mlm_tokens": int(masked_pos.sum().item()),
            "aux_span_tokens": int(aux_pos.sum().item()),
            "mlm_loss": float(mlm_loss.detach().item()),
            "mode_records": mode_records,
        })
    del model
    usable = [r for r in records if not r.get("skipped")]
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    geom_grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in usable:
        for mr in r["mode_records"]:
            key = f"{mr['center_mode']}|d{mr['proj_dim']}"
            for field in ["aux_loss", "aux_over_mlm_l2_all_hidden_lambda1", "aux_over_mlm_l2_aux_spans_lambda1", "grad_cosine_all_hidden", "grad_cosine_aux_spans"]:
                if mr.get(field) is not None:
                    grouped[key][field].append(float(mr[field]))
            geom = mr.get("geometry") or {}
            for field in ["actual_minus_shuffled_mean", "actual_minus_same_row_other_mean"]:
                if geom.get(field) is not None:
                    geom_grouped[key][field].append(float(geom[field]))
            for gname, fname in [("actual_cos", "mean"), ("shuffled_cos", "mean"), ("same_row_other_cos", "mean")]:
                val = (geom.get(gname) or {}).get(fname)
                if val is not None:
                    geom_grouped[key][f"{gname}_mean"].append(float(val))
    summary_by_mode: dict[str, Any] = {}
    for key in sorted(grouped.keys() | geom_grouped.keys()):
        entry: dict[str, Any] = {}
        for field, vals in grouped.get(key, {}).items():
            entry[field] = summarize(vals)
        for field, vals in geom_grouped.get(key, {}).items():
            entry[field] = summarize(vals)
        ratio_vals = grouped.get(key, {}).get("aux_over_mlm_l2_all_hidden_lambda1", [])
        ratio_aux_vals = grouped.get(key, {}).get("aux_over_mlm_l2_aux_spans_lambda1", [])
        mean_ratio = statistics.mean(ratio_vals) if ratio_vals else None
        mean_aux_ratio = statistics.mean(ratio_aux_vals) if ratio_aux_vals else None
        entry["lambda_for_5pct_all_hidden_l2"] = (0.05 / mean_ratio) if mean_ratio and mean_ratio > 0 else None
        entry["lambda_for_10pct_all_hidden_l2"] = (0.10 / mean_ratio) if mean_ratio and mean_ratio > 0 else None
        entry["lambda_for_5pct_aux_span_l2"] = (0.05 / mean_aux_ratio) if mean_aux_ratio and mean_aux_ratio > 0 else None
        summary_by_mode[key] = entry
    return {
        "label": label,
        "path": str(ckpt_path),
        "status": "ok",
        "hidden_size": hidden_size_seen,
        "batches_seen": len(records),
        "usable_batches": len(usable),
        "records": records,
        "summary_by_mode": summary_by_mode,
    }


def choose_lines_for_note(results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"- {r.get('label')}: status `{r.get('status')}`")
            continue
        lines.append(f"### {r['label']}")
        for key in sorted(r["summary_by_mode"].keys()):
            s = r["summary_by_mode"][key]
            mode, dim = key.split("|")
            loss = (s.get("aux_loss") or {}).get("mean")
            ratio = (s.get("aux_over_mlm_l2_all_hidden_lambda1") or {}).get("mean")
            auxratio = (s.get("aux_over_mlm_l2_aux_spans_lambda1") or {}).get("mean")
            cos = (s.get("grad_cosine_all_hidden") or {}).get("mean")
            margin = (s.get("actual_minus_shuffled_mean") or {}).get("mean")
            other_margin = (s.get("actual_minus_same_row_other_mean") or {}).get("mean")
            lam5 = s.get("lambda_for_5pct_all_hidden_l2")
            lines.append(
                f"- {mode} {dim}: aux_loss={loss:.4f}, pair-shuffle margin={margin:.4f}, pair-same-row-other margin={other_margin:.4f}, "
                f"aux/MLM L2 all={ratio:.4f}, aux-span={auxratio:.4f}, grad_cos={cos:.4f}, λ@5%all≈{lam5:.4g}"
            )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--sample-mode", choices=["front", "stride"], default="front")
    ap.add_argument("--sample-rows", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=4)
    ap.add_argument("--train-rng-seed", type=int, default=43023)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--proj-dims", default="32,64,128,480")
    ap.add_argument("--center-modes", default="raw,row_centered")
    ap.add_argument("--proj-seed", type=int, default=70070)
    ap.add_argument("--checkpoints", default="tokenmean_20M,tokenmean_80M,tokenmean_100M")
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
    results: list[dict[str, Any]] = []
    for label in ckpt_labels:
        if label not in CHECKPOINTS:
            raise ValueError(f"unknown checkpoint label {label}; choices {sorted(CHECKPOINTS)}")
        print(json.dumps({"event": "projected_consistency_checkpoint_start", "label": label, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        result = run_checkpoint(label, CHECKPOINTS[label], examples, tokenizer, span_by_ex, args)
        results.append(result)
        print(json.dumps({"event": "projected_consistency_checkpoint_done", "label": label, "status": result.get("status")}), flush=True)
    interpretation = [
        "This CPU run measures local hidden-state pressure and pair geometry for future construction; it is not BabyLM task-score evidence and does not justify a GPU run by itself.",
        "Raw cosine mostly measures already high same-row/topic anisotropy; row-centered variants subtract each row's hidden mean before span pooling, so they put more pressure on pair-specific residual content while still using only training-side source/rewrite spans.",
        "A future consistency trainer should use these measured λ scales with a small auxiliary weight and should not combine the auxiliary loss with fractional credit, minfreq50, static masking, or sequence changes in the first screen.",
    ]
    summary = {
        "status": "PROJECTED_CONSISTENCY_CALIBRATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "CPU-only calibration of fixed-subspace source/rewrite consistency losses on actual legal16k token-mean compact-view checkpoints.",
        "inputs": {
            "train_100m": str(collate_mod.TRAIN_100M),
            "train_sha256": train_sha,
            "span_jsonl": str(collate_mod.SPAN_JSONL),
            "span_examples": len(span_by_ex),
            "tokenizer_dir": str(collate_mod.TOKENIZER_DIR),
            "tokenizer_sha256": tok_sha,
            "sample_mode": args.sample_mode,
            "sample_rows_loaded": len(examples),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "train_rng_seed": args.train_rng_seed,
            "proj_dims": [int(x) for x in args.proj_dims.split(",") if x.strip()],
            "center_modes": [x.strip() for x in args.center_modes.split(",") if x.strip()],
            "proj_seed": args.proj_seed,
            "torch_threads": args.torch_threads,
            "checkpoints": ckpt_labels,
        },
        "checkpoint_results": results,
        "scientific_interpretation": interpretation,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "projected_consistency_calibration.json"
    out_md = out_dir / "projected_consistency_calibration.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research projected source-view consistency calibration",
        "",
        summary["purpose"],
        "",
        "No model update, official evaluation, corpus change, tokenizer change, or H100 work was performed.",
        "",
        "## Inputs",
        f"- sample: `{summary['inputs']['sample_rows_loaded']}` stream rows, batch `{args.batch_size}`, max batches `{args.max_batches}`",
        f"- projections: `{args.proj_dims}`; modes: `{args.center_modes}`",
        f"- train SHA: `{train_sha}`",
        f"- tokenizer SHA: `{tok_sha}`",
        "",
        "## Per-checkpoint summaries",
    ]
    lines.extend(choose_lines_for_note(results))
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
