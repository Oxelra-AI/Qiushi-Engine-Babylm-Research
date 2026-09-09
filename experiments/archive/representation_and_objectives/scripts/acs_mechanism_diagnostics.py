#!/usr/bin/env python3
"""research: posthoc mechanism diagnostics for the ACS fork.

Measurements:
  1. Training-log dynamics: CE/ACS losses, selected-set margin/top1, grad norms,
     and clip frequency (grad_norm > 1.0 before clipping in the research trainer).
  2. One matched masked-batch diagnostic from a checkpoint: probability mass in the
     ACS selected set (correct + top-K negatives), CE-vs-ACS gradient cosine and
     norm ratio on parameter subsets, and top-K label inclusion/top1 statistics.

This script is readout/analysis only; it does not modify checkpoints or train.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaV2ForMaskedLM

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
COMPACT_EXPERIENCE_SCRIPTS = _public_path('experiments/archive/compact_experience/scripts')
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))
import masking_curriculum_trainer as base  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SOURCE_METRICS = _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/scientific_metrics.json')
SOURCE_LOG = _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/training_log.jsonl')
SOURCE_CKPT = _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M')
DEFAULT_EXAMPLES = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def qstats(vals) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v)))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "min": xs[0], "p10": q(0.10), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p90": q(0.90), "max": xs[-1]}


def summarize_training_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": rel(path)}
    rows = read_jsonl(path)
    out: dict[str, Any] = {"exists": True, "path": rel(path), "n_steps": len(rows)}
    if not rows:
        return out
    for key in ["ce_loss", "acs_loss", "total_loss", "grad_norm", "masked_tokens", "update_lr", "acs_correct_top1_frac", "acs_mean_margin", "acs_margin_negative_frac"]:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            out[f"{key}_stats"] = qstats(vals)
            out[f"{key}_first"] = vals[0]
            out[f"{key}_last"] = vals[-1]
    g = [float(r["grad_norm"]) for r in rows if isinstance(r.get("grad_norm"), (int, float))]
    if g:
        out["grad_clip_threshold"] = 1.0
        out["clip_count_preclip_grad_norm_gt_1"] = sum(v > 1.0 for v in g)
        out["clip_frac_preclip_grad_norm_gt_1"] = sum(v > 1.0 for v in g) / len(g)
    out["first_record"] = rows[0]
    out["last_record"] = rows[-1]
    return out


def read_source_checkpoint(metrics_path: Path, checkpoint_name: str) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for rec in metrics.get("saved_checkpoints", []):
        if rec.get("name") == checkpoint_name:
            return dict(rec)
    raise RuntimeError(f"checkpoint {checkpoint_name!r} not found")


def read_source_step(training_log_path: Path, consumed_words: int) -> dict[str, Any]:
    for r in read_jsonl(training_log_path):
        if int(r.get("cumulative_word_exposure", -1)) == consumed_words:
            return dict(r)
    raise RuntimeError(f"no source log row at {consumed_words}")


def select_tail_examples(examples: list, consumed_words: int, total_target: int):
    cumulative = 0
    start_idx = None
    for i, ex in enumerate(examples):
        cumulative += int(ex.words)
        if cumulative == consumed_words:
            start_idx = i + 1
            break
        if cumulative > consumed_words:
            raise RuntimeError(f"consumed_words={consumed_words} inside example {i}; cumulative={cumulative}")
    if start_idx is None:
        raise RuntimeError(f"cannot find consumed_words={consumed_words}")
    tail = []
    running = consumed_words
    for ex in examples[start_idx:]:
        w = int(ex.words)
        if running + w > total_target:
            raise RuntimeError(f"partial example: running={running} next={w} target={total_target}")
        tail.append(ex)
        running += w
        if running == total_target:
            break
    if running != total_target:
        raise RuntimeError(f"tail ended at {running}")
    return tail, start_idx, running - consumed_words


def combine_microbatches(micro_batches):
    return {
        "input_ids": torch.cat([b["input_ids"] for b in micro_batches], dim=0),
        "attention_mask": torch.cat([b["attention_mask"] for b in micro_batches], dim=0),
        "word_group": torch.cat([b["word_group"] for b in micro_batches], dim=0),
        "words": torch.cat([b["words"] for b in micro_batches], dim=0),
    }


def get_effective_batch(loader, accum_steps: int, local_step: int):
    buf = []
    step = 0
    for batch in loader:
        buf.append(batch)
        if len(buf) == accum_steps:
            step += 1
            if step == local_step:
                return combine_microbatches(buf)
            buf = []
    raise RuntimeError(f"could not reach effective local_step={local_step}")


def compute_acs(logits_masked: torch.Tensor, labels_masked: torch.Tensor, topk: int) -> tuple[torch.Tensor, dict[str, Any]]:
    n, V = logits_masked.shape
    correct_logits = logits_masked.gather(1, labels_masked.unsqueeze(1))
    with torch.no_grad():
        sel = logits_masked.detach().clone()
        sel.scatter_(1, labels_masked.unsqueeze(1), float("-inf"))
        _, neg_idx = sel.topk(topk, dim=1)
    neg_logits = logits_masked.gather(1, neg_idx)
    contrastive = torch.cat([correct_logits, neg_logits], dim=1)
    targets = torch.zeros(n, dtype=torch.long, device=logits_masked.device)
    acs = F.cross_entropy(contrastive, targets)
    with torch.no_grad():
        full_probs = F.softmax(logits_masked, dim=-1)
        correct_prob = full_probs.gather(1, labels_masked.unsqueeze(1)).squeeze(1)
        neg_probs = full_probs.gather(1, neg_idx)
        selected_mass = correct_prob + neg_probs.sum(dim=1)
        margin = correct_logits.squeeze(1) - neg_logits.max(dim=1).values
        acs_probs = F.softmax(contrastive, dim=1)
        st = {
            "n_masked": int(n),
            "vocab_size": int(V),
            "topk": int(topk),
            "ce_full_loss": float(F.cross_entropy(logits_masked, labels_masked).item()),
            "acs_loss": float(acs.item()),
            "correct_top1_frac": float((margin >= 0).float().mean().item()),
            "negative_margin_frac": float((margin < 0).float().mean().item()),
            "margin_stats": qstats(margin.detach().cpu().tolist()),
            "full_correct_prob_stats": qstats(correct_prob.detach().cpu().tolist()),
            "selected_set_prob_mass_stats": qstats(selected_mass.detach().cpu().tolist()),
            "selected_negative_prob_mass_stats": qstats(neg_probs.sum(dim=1).detach().cpu().tolist()),
            "acs_correct_softmax_prob_stats": qstats(acs_probs[:, 0].detach().cpu().tolist()),
            "selected_mass_mean": float(selected_mass.mean().item()),
            "selected_mass_median": float(selected_mass.median().item()),
            "correct_prob_mean": float(correct_prob.mean().item()),
        }
    return acs, st


def selected_params(model) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {"all_trainable": [], "embeddings": [], "encoder": [], "lm_head": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        groups["all_trainable"].append(p)
        lname = name.lower()
        if "embeddings" in lname:
            groups["embeddings"].append(p)
        elif "cls" in lname or "lm_predictions" in lname or "predictions" in lname:
            groups["lm_head"].append(p)
        elif "encoder" in lname or "deberta" in lname:
            groups["encoder"].append(p)
    return groups


def grad_stats_for_loss(loss: torch.Tensor, params: list[torch.nn.Parameter], retain_graph: bool) -> dict[str, Any]:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    flat_parts = []
    for g in grads:
        if g is not None:
            flat_parts.append(g.detach().float().flatten().cpu())
    if not flat_parts:
        return {"n_params_with_grad": 0, "norm": 0.0, "flat": torch.empty(0)}
    flat = torch.cat(flat_parts)
    return {"n_params_with_grad": len(flat_parts), "norm": float(torch.linalg.vector_norm(flat).item()), "flat": flat}


def gradient_alignment(model, ce_loss: torch.Tensor, acs_loss: torch.Tensor) -> dict[str, Any]:
    groups = selected_params(model)
    out: dict[str, Any] = {}
    for gname, params in groups.items():
        if not params:
            out[gname] = {"n_params_with_grad": 0}
            continue
        ce = grad_stats_for_loss(ce_loss, params, retain_graph=True)
        acs = grad_stats_for_loss(acs_loss, params, retain_graph=True)
        a = ce.pop("flat")
        b = acs.pop("flat")
        if a.numel() and b.numel():
            denom = float(torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item())
            cos = float(torch.dot(a, b).item() / denom) if denom > 0 else float("nan")
        else:
            cos = float("nan")
        out[gname] = {
            "ce_n_params_with_grad": ce["n_params_with_grad"],
            "acs_n_params_with_grad": acs["n_params_with_grad"],
            "ce_grad_norm": ce["norm"],
            "acs_grad_norm": acs["norm"],
            "acs_to_ce_norm_ratio": acs["norm"] / ce["norm"] if ce["norm"] else None,
            "cosine": cos,
        }
        del a, b
    return out


def reset_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def matched_batch_diagnostic(model_path: Path, example_jsonl: Path, topk: int, local_step: int,
                             batch_size: int, micro_batch_size: int, max_seq_length: int,
                             train_rng_seed: int, seed: int, device: torch.device) -> dict[str, Any]:
    source_rec = read_source_checkpoint(SOURCE_METRICS, "chck_80M")
    consumed_words = int(source_rec["actual_cumulative_word_exposure"])
    source_step = int(read_source_step(SOURCE_LOG, consumed_words)["step"])
    tokenizer = base.make_portable_tokenizer(str(SOURCE_CKPT))
    all_examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(example_jsonl, 100_000_000)
    tail_examples, tail_start_idx, continuation_words = select_tail_examples(all_examples, consumed_words, 100_000_000)
    dataset = base.MaskedChunkDataset(tail_examples, tokenizer, max_seq_length)
    loader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=False, collate_fn=base.collate, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    accum_steps = batch_size // micro_batch_size
    batch = get_effective_batch(loader, accum_steps, local_step)
    words = int(batch.pop("words").sum().item())
    reset_rng(seed)
    model = DebertaV2ForMaskedLM.from_pretrained(model_path, local_files_only=True)
    model.to(device).train()
    reset_rng(train_rng_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_rng_seed)
    curriculum_state = base.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15, switch_frac=1.0)
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=math.ceil(len(dataset) / batch_size))
    curriculum_state.current_step = local_step - 1
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    word_group = batch["word_group"].to(device)
    masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, curriculum_state, gen)
    n_pred_total = int((labels != -100).sum().item())
    # Use the first microbatch only for gradient alignment to bound memory, but record exact masked count.
    sl_labels = labels[:micro_batch_size]
    out_model = model(input_ids=masked_inputs[:micro_batch_size], attention_mask=attention_mask[:micro_batch_size], labels=sl_labels)
    ce_loss = out_model.loss
    logits = out_model.logits
    mask_pos = sl_labels != -100
    logits_masked = logits[mask_pos]
    labels_masked = sl_labels[mask_pos]
    acs_loss, acs_st = compute_acs(logits_masked, labels_masked, topk)
    grad_align = gradient_alignment(model, ce_loss, acs_loss)
    result = {
        "model_path": rel(model_path),
        "source_checkpoint_consumed_words": consumed_words,
        "source_step": source_step,
        "example_jsonl": rel(example_jsonl),
        "jsonl_total_words": jsonl_total_words,
        "jsonl_total_rows": jsonl_total_rows,
        "tail_start_idx": tail_start_idx,
        "tail_examples": len(tail_examples),
        "continuation_words": continuation_words,
        "diagnostic_local_step": local_step,
        "effective_batch_words": words,
        "effective_batch_rows": int(input_ids.shape[0]),
        "effective_batch_masked_tokens": n_pred_total,
        "micro_batch_masked_tokens": int(labels_masked.shape[0]),
        "acs_selection": acs_st,
        "gradient_alignment": grad_align,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="ACS mechanism diagnostics")
    ap.add_argument("--treatment-run", required=True)
    ap.add_argument("--control-run", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--example-jsonl", default=str(DEFAULT_EXAMPLES))
    ap.add_argument("--checkpoint-name", default="chck_100M")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--diagnostic-local-step", type=int, default=252)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--micro-batch-size", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--train-rng-seed", type=int, default=53023)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--logs-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    treatment_run = Path(args.treatment_run)
    control_run = Path(args.control_run)
    runs = {
        "treatment": treatment_run,
        "control": control_run,
    }
    summary: dict[str, Any] = {
        "status": "ACS_MECHANISM_DIAGNOSTICS_DONE",
        "created_utc": now_utc(),
        "diagnostic_boundary": "posthoc readout on a fixed matched masked batch and training logs; no training or tuning",
        "runs": {},
    }
    for label, run in runs.items():
        metrics_path = run / "scientific_metrics.json"
        log_path = run / "training_log.jsonl"
        summary["runs"][label] = {
            "run_dir": rel(run),
            "metrics_exists": metrics_path.exists(),
            "model_exists": (run / "hf_model" / args.checkpoint_name / "model.safetensors").exists(),
            "training_log_summary": summarize_training_log(log_path),
        }
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary["runs"][label]["metrics"] = {
                "status": metrics.get("status"),
                "variant": metrics.get("variant"),
                "word_exposure": metrics.get("word_exposure"),
                "tail_word_exposure": metrics.get("tail_word_exposure"),
                "actual_training_steps": metrics.get("actual_training_steps"),
                "parameter_count": metrics.get("parameter_count"),
                "saved_checkpoints": [c.get("name") for c in metrics.get("saved_checkpoints", [])],
            }

    if not args.logs_only:
        device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        for label, run in runs.items():
            ckpt = run / "hf_model" / args.checkpoint_name
            if not (ckpt / "model.safetensors").exists():
                summary["runs"][label]["matched_batch_diagnostic"] = {"status": "missing_model", "model_path": rel(ckpt)}
                continue
            print(json.dumps({"event": "matched_batch_start", "label": label, "model": rel(ckpt), "device": str(device)}), flush=True)
            diag = matched_batch_diagnostic(
                ckpt, Path(args.example_jsonl), args.topk, args.diagnostic_local_step,
                args.batch_size, args.micro_batch_size, args.max_seq_length,
                args.train_rng_seed, args.seed, device,
            )
            summary["runs"][label]["matched_batch_diagnostic"] = diag
            print(json.dumps({"event": "matched_batch_done", "label": label,
                              "ce": diag["acs_selection"].get("ce_full_loss"),
                              "acs": diag["acs_selection"].get("acs_loss"),
                              "selected_mass_mean": diag["acs_selection"].get("selected_mass_mean"),
                              "all_grad_cos": diag["gradient_alignment"].get("all_trainable", {}).get("cosine")}, ensure_ascii=False), flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Direct deltas where meaningful.
    tr = summary["runs"].get("treatment", {})
    co = summary["runs"].get("control", {})
    deltas: dict[str, Any] = {}
    tr_log = tr.get("training_log_summary", {})
    co_log = co.get("training_log_summary", {})
    for key in ["ce_loss_last", "grad_norm_last", "clip_frac_preclip_grad_norm_gt_1"]:
        if isinstance(tr_log.get(key), (int, float)) and isinstance(co_log.get(key), (int, float)):
            deltas[key] = tr_log[key] - co_log[key]
    if not args.logs_only:
        tr_diag = tr.get("matched_batch_diagnostic", {})
        co_diag = co.get("matched_batch_diagnostic", {})
        for key in ["selected_mass_mean", "correct_prob_mean"]:
            tv = (tr_diag.get("acs_selection") or {}).get(key)
            cv = (co_diag.get("acs_selection") or {}).get(key)
            if isinstance(tv, (int, float)) and isinstance(cv, (int, float)):
                deltas[f"matched_batch_{key}_treatment_minus_control"] = tv - cv
        tg = (tr_diag.get("gradient_alignment") or {}).get("all_trainable", {})
        cg = (co_diag.get("gradient_alignment") or {}).get("all_trainable", {})
        for key in ["cosine", "acs_to_ce_norm_ratio"]:
            if isinstance(tg.get(key), (int, float)) and isinstance(cg.get(key), (int, float)):
                deltas[f"matched_batch_all_grad_{key}_treatment_minus_control"] = tg[key] - cg[key]
    summary["deltas_treatment_minus_control"] = deltas

    out_path = out_root / "acs_mechanism_diagnostics_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": rel(out_path)}), flush=True)


if __name__ == "__main__":
    main()
