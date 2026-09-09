#!/usr/bin/env python3
"""research: CPU hidden-gradient probe for a source-view consistency loss.

This probes actual trained DeBERTa checkpoints on frozen training rows without any
parameter update or official evaluation.  It compares the local gradient pressure
of token-mean MLM with a row-normalized positive-only source/rewrite cosine loss
computed from the research pair-span map.  The measurement is only for route design:
if a shared-subspace consistency branch is later selected, the auxiliary weight
should be small enough to avoid overriding MLM and full-vector forcing.
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
OUT_DIR = WORKSPACE / "data/consistency_gradient_probe"
CHECKPOINTS = {
    "tokenmean_80M": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
    "wordmean_80M": WORKSPACE / "training/runs/wordmean_mlm_complianttok_reinvest_seed43022_80M/hf_model/chck_80M",
}


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


collate_mod = load_module("consistency_collate_for_grad_probe", COLLATE_SCRIPT)
base = load_module("compact_experience_base_step069_consistency_grad", BASE_TRAINER_PATH)


def summarize(vals: list[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {"n": len(vals), "mean": float(statistics.mean(vals)), "median": float(statistics.median(vals)), "min": float(min(vals)), "max": float(max(vals)), "std": float(statistics.pstdev(vals))}


def pool_ranges(hidden: torch.Tensor, batch_row: int, ranges: list[list[int]]) -> torch.Tensor:
    chunks = [hidden[batch_row, int(a):int(b), :] for a, b in ranges]
    return torch.cat(chunks, dim=0).mean(dim=0)


def row_normalized_pair_cos_loss(last_hidden: torch.Tensor, aux_records: list[dict[str, Any]]) -> torch.Tensor:
    row_losses: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
    for r in aux_records:
        br = int(r["batch_row"])
        src = pool_ranges(last_hidden, br, r["source_ranges"])
        rew = pool_ranges(last_hidden, br, r["rewrite_ranges"])
        src_n = torch.nn.functional.normalize(src.float(), dim=0, eps=1e-6)
        rew_n = torch.nn.functional.normalize(rew.float(), dim=0, eps=1e-6)
        cos_sr = (src_n * rew_n.detach()).sum()
        cos_rs = (rew_n * src_n.detach()).sum()
        loss = 0.5 * ((1.0 - cos_sr) + (1.0 - cos_rs))
        row_losses[(br, int(r["example_id"]))].append(loss)
    if not row_losses:
        return last_hidden.sum() * 0.0
    return torch.stack([torch.stack(v).mean() for v in row_losses.values()]).mean()


def mask_for_training(batch: dict[str, Any], tokenizer, state, gen) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"][:, :collate_mod.SEQ_LEN].contiguous()
    attention_mask = batch["attention_mask"][:, :collate_mod.SEQ_LEN].contiguous()
    word_group = batch["word_group"][:, :collate_mod.SEQ_LEN].contiguous()
    masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, state, gen)
    return masked_inputs, labels, attention_mask, word_group


def grad_stats(grad: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float | int | None]:
    g = grad.detach().float()
    if mask is not None:
        m = mask.bool()
        if m.dim() == 2:
            m = m.unsqueeze(-1).expand_as(g)
        g = g[m]
    if g.numel() == 0:
        return {"numel": 0, "l2": 0.0, "rms": None, "mean_abs": None, "max_abs": None}
    return {
        "numel": int(g.numel()),
        "l2": float(torch.linalg.vector_norm(g).item()),
        "rms": float(torch.sqrt((g * g).mean()).item()),
        "mean_abs": float(g.abs().mean().item()),
        "max_abs": float(g.abs().max().item()),
    }


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


def aux_token_mask(shape: tuple[int, int], aux_records: list[dict[str, Any]]) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool)
    for r in aux_records:
        br = int(r["batch_row"])
        for a, b in r["source_ranges"] + r["rewrite_ranges"]:
            mask[br, int(a):int(b)] = True
    return mask


def run_checkpoint(label: str, ckpt_path: pathlib.Path, examples, tokenizer, span_by_ex, args) -> dict[str, Any]:
    if not ckpt_path.exists():
        return {"label": label, "status": "missing", "path": str(ckpt_path)}
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt_path))
    model.eval()
    model.to("cpu")
    state = base.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15, switch_frac=0.7, amlm_window=10, amlm_lambda=0.2)
    total_steps = math.ceil(len(examples) / args.batch_size)
    state.initialize(vocab_size=len(tokenizer), total_steps=max(total_steps, 1))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.train_rng_seed)
    ds = collate_mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, collate_mod.SEQ_LEN)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_mod.consistency_collate)
    records = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        aux = batch["aux_records"]
        state.current_step = batch_index
        masked_inputs, labels, attention_mask, word_group = mask_for_training(batch, tokenizer, state, gen)
        if not aux:
            records.append({"batch_index": batch_index, "aux_records": 0, "skipped": "no_aux"})
            continue
        out = model(input_ids=masked_inputs, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        logits = out.logits
        last_hidden = out.hidden_states[-1]
        mlm_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        aux_loss = row_normalized_pair_cos_loss(last_hidden, aux)
        mlm_grad = torch.autograd.grad(mlm_loss, last_hidden, retain_graph=True, allow_unused=True)[0]
        aux_grad = torch.autograd.grad(aux_loss, last_hidden, retain_graph=False, allow_unused=True)[0]
        if mlm_grad is None or aux_grad is None:
            raise RuntimeError(f"gradient unavailable for {label}: mlm_grad={mlm_grad is not None}, aux_grad={aux_grad is not None}")
        masked_pos = labels != -100
        aux_pos = aux_token_mask(tuple(labels.shape), aux)
        row_ids = {(int(r["batch_row"]), int(r["example_id"])) for r in aux}
        mlm_l2_all = grad_stats(mlm_grad)["l2"]
        aux_l2_all = grad_stats(aux_grad)["l2"]
        mlm_l2_aux = grad_stats(mlm_grad, aux_pos)["l2"]
        aux_l2_aux = grad_stats(aux_grad, aux_pos)["l2"]
        record = {
            "batch_index": batch_index,
            "rows": int(labels.shape[0]),
            "tokens_per_row": int(labels.shape[1]),
            "aux_rows": len(row_ids),
            "aux_records": len(aux),
            "selected_mlm_tokens": int(masked_pos.sum().item()),
            "aux_span_tokens": int(aux_pos.sum().item()),
            "mlm_loss": float(mlm_loss.detach().item()),
            "aux_cos_loss": float(aux_loss.detach().item()),
            "grad_all": {"mlm": grad_stats(mlm_grad), "aux_lambda1": grad_stats(aux_grad), "aux_over_mlm_l2": (float(aux_l2_all) / float(mlm_l2_all)) if mlm_l2_all else None, "cosine": cosine_flat(mlm_grad, aux_grad)},
            "grad_aux_spans": {"mlm": grad_stats(mlm_grad, aux_pos), "aux_lambda1": grad_stats(aux_grad, aux_pos), "aux_over_mlm_l2": (float(aux_l2_aux) / float(mlm_l2_aux)) if mlm_l2_aux else None, "cosine": cosine_flat(mlm_grad, aux_grad, aux_pos)},
            "grad_masked_tokens": {"mlm": grad_stats(mlm_grad, masked_pos), "aux_lambda1": grad_stats(aux_grad, masked_pos), "cosine": cosine_flat(mlm_grad, aux_grad, masked_pos)},
        }
        records.append(record)
    del model
    usable = [r for r in records if not r.get("skipped")]
    ratio_all = [r["grad_all"]["aux_over_mlm_l2"] for r in usable if r["grad_all"]["aux_over_mlm_l2"] is not None]
    ratio_aux = [r["grad_aux_spans"]["aux_over_mlm_l2"] for r in usable if r["grad_aux_spans"]["aux_over_mlm_l2"] is not None]
    cos_all = [r["grad_all"]["cosine"] for r in usable if r["grad_all"]["cosine"] is not None]
    cos_aux = [r["grad_aux_spans"]["cosine"] for r in usable if r["grad_aux_spans"]["cosine"] is not None]
    aux_losses = [r["aux_cos_loss"] for r in usable]
    mlm_losses = [r["mlm_loss"] for r in usable]
    return {
        "label": label,
        "path": str(ckpt_path),
        "status": "ok",
        "batches_seen": len(records),
        "usable_batches": len(usable),
        "records": records,
        "summary": {
            "mlm_loss": summarize(mlm_losses),
            "aux_cos_loss": summarize(aux_losses),
            "aux_over_mlm_l2_all_hidden_lambda1": summarize(ratio_all),
            "aux_over_mlm_l2_aux_spans_lambda1": summarize(ratio_aux),
            "grad_cosine_all_hidden": summarize(cos_all),
            "grad_cosine_aux_spans": summarize(cos_aux),
            "lambda_for_5pct_all_hidden_l2": (0.05 / statistics.mean(ratio_all)) if ratio_all and statistics.mean(ratio_all) > 0 else None,
            "lambda_for_10pct_all_hidden_l2": (0.10 / statistics.mean(ratio_all)) if ratio_all and statistics.mean(ratio_all) > 0 else None,
            "lambda_for_5pct_aux_span_l2": (0.05 / statistics.mean(ratio_aux)) if ratio_aux and statistics.mean(ratio_aux) > 0 else None,
            "lambda_for_10pct_aux_span_l2": (0.10 / statistics.mean(ratio_aux)) if ratio_aux and statistics.mean(ratio_aux) > 0 else None,
        },
    }


def make_note(summary: dict[str, Any], out_json: pathlib.Path) -> str:
    lines = [
        "# research source-view consistency hidden-gradient probe",
        "",
        "CPU-only actual-checkpoint probe. No parameter update, official evaluation, corpus change, or GPU use.",
        "",
        "## Sample",
        f"- stream sample mode: `{summary['inputs']['sample_mode']}`",
        f"- rows loaded: `{summary['inputs']['sample_rows_loaded']}`",
        f"- batch size: `{summary['inputs']['batch_size']}`; max batches per checkpoint: `{summary['inputs']['max_batches']}`",
        f"- train SHA: `{summary['inputs']['train_sha256']}`",
        f"- tokenizer SHA: `{summary['inputs']['tokenizer_sha256']}`",
        "",
        "## Gradient summaries",
    ]
    for r in summary["checkpoint_results"]:
        if r.get("status") != "ok":
            lines.append(f"- {r['label']}: status `{r.get('status')}`")
            continue
        s = r["summary"]
        lines.append(
            f"- {r['label']}: usable_batches={r['usable_batches']}, mlm_loss_mean={s['mlm_loss']['mean']:.4f}, "
            f"aux_cos_loss_mean={s['aux_cos_loss']['mean']:.4f}, aux/MLM hidden-L2 λ=1 mean={s['aux_over_mlm_l2_all_hidden_lambda1']['mean']:.3f}, "
            f"aux-span ratio={s['aux_over_mlm_l2_aux_spans_lambda1']['mean']:.3f}, grad cosine all={s['grad_cosine_all_hidden']['mean']:.4f}, "
            f"suggested λ for 5% all-hidden L2≈{s['lambda_for_5pct_all_hidden_l2']:.4g}"
        )
    lines += ["", "## Interpretation"]
    for item in summary["scientific_interpretation"]:
        lines.append(f"- {item}")
    lines += ["", f"Full JSON: `{out_json}`"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--sample-mode", choices=["front", "stride"], default="front")
    ap.add_argument("--sample-rows", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=6)
    ap.add_argument("--train-rng-seed", type=int, default=43023)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--checkpoints", default="tokenmean_80M,wordmean_80M")
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
            raise ValueError(f"unknown checkpoint label {label}; choices {sorted(CHECKPOINTS)}")
        print(json.dumps({"event": "grad_probe_checkpoint_start", "label": label, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        results.append(run_checkpoint(label, CHECKPOINTS[label], examples, tokenizer, span_by_ex, args))
        print(json.dumps({"event": "grad_probe_checkpoint_done", "label": label, "status": results[-1].get("status")}), flush=True)
    interpretation = [
        "The probe compares gradient pressure at the final hidden tensor, not full parameter updates. It is a scale and compatibility measurement for future construction, not score evidence.",
        "If source-view consistency is later selected, λ should be chosen so its hidden-state gradient is a small fraction of MLM on ordinary mixed batches, because actual paired representations are already close and full-vector forcing risks damaging source-specific syntax/entities.",
    ]
    by_label = {r.get("label"): r for r in results if r.get("status") == "ok"}
    if "tokenmean_80M" in by_label and "wordmean_80M" in by_label:
        tm = by_label["tokenmean_80M"]["summary"]
        wm = by_label["wordmean_80M"]["summary"]
        interpretation.append(
            f"Word-mean versus token-mean on the same probe rows has aux cosine loss delta {wm['aux_cos_loss']['mean'] - tm['aux_cos_loss']['mean']:+.4f} and aux/MLM hidden-gradient-ratio delta {wm['aux_over_mlm_l2_all_hidden_lambda1']['mean'] - tm['aux_over_mlm_l2_all_hidden_lambda1']['mean']:+.3f}; combine this with research representation alignment and research scores before attributing GlobalPIQA/COMPS movement to pair abstraction."
        )
    summary = {
        "status": "CONSISTENCY_GRADIENT_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "CPU-only actual-checkpoint local gradient measurement for a possible source-view consistency objective.",
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
            "torch_threads": args.torch_threads,
            "checkpoints": ckpt_labels,
        },
        "checkpoint_results": results,
        "scientific_interpretation": interpretation,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "consistency_gradient_probe.json"
    out_md = out_dir / "consistency_gradient_probe.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(make_note(summary, out_json), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "checkpoint_status": {r.get("label"): r.get("status") for r in results},
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
