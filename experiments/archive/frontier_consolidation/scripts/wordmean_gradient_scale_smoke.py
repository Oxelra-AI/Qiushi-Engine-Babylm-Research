#!/usr/bin/env python3
"""research: actual-gradient smoke for word-mean vs token-mean MLM scaling.

This is a small, bounded gradient inspection, not training.  It uses the same
initialization and actual legal training rows, applies one shared WWM mask per
sampled microbatch, and computes parameter-gradient norm/cosine for token-mean and
word-mean losses with identical dropout randomness.  It complements the CPU
logit-weight analysis: unchanged learning rate cannot be read as pure relative
word-credit isolation if parameter-gradient scale or direction changes materially.
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
import os
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np
import torch


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
BASE_TRAINER_PATH = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
WORDMEAN_TRAINER_PATH = WORKSPACE / "scripts/wordmean_mlm_trainer.py"
TRAIN_FILE = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
DEFAULT_OUT = WORKSPACE / "data/wordmean_gradient_scale_smoke"


def import_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = import_module(BASE_TRAINER_PATH, "compact_experience_base_for_step062_gradient")
wordmod = import_module(WORDMEAN_TRAINER_PATH, "frontier_consolidation_step061_wordmean_for_step062_gradient")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_batch_examples(path: pathlib.Path, batch_indices: list[int], full_batch_size: int, micro_batch_size: int) -> list[dict[str, Any]]:
    starts = {bi: bi * full_batch_size for bi in batch_indices}
    needed = set()
    for bi, st in starts.items():
        needed.update(range(st, st + micro_batch_size))
    out: dict[int, Any] = {}
    max_idx = max(needed) if needed else -1
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx > max_idx:
                break
            if idx not in needed:
                continue
            obj = json.loads(line)
            out[idx] = base.Example(text=str(obj["text"]), words=int(obj["words"]), example_id=int(obj.get("example_id", idx)), source=str(obj.get("source", "")))
    records = []
    for bi in batch_indices:
        st = starts[bi]
        exs = [out[i] for i in range(st, st + micro_batch_size) if i in out]
        if len(exs) != micro_batch_size:
            raise RuntimeError(f"missing rows for batch {bi}: have {len(exs)} expected {micro_batch_size}")
        records.append({"batch_index0": bi, "examples": exs})
    return records


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grad_list_and_norm(model) -> tuple[list[torch.Tensor | None], float]:
    grads: list[torch.Tensor | None] = []
    ss = 0.0
    for p in model.parameters():
        if p.grad is None:
            grads.append(None)
            continue
        g = p.grad.detach().float().cpu().clone()
        grads.append(g)
        ss += float((g * g).sum().item())
    return grads, math.sqrt(ss)


def dot_with(model, ref_grads: list[torch.Tensor | None]) -> tuple[float, float]:
    dot = 0.0
    ss = 0.0
    for p, rg in zip(model.parameters(), ref_grads):
        if p.grad is None or rg is None:
            continue
        g = p.grad.detach().float().cpu()
        dot += float((g * rg).sum().item())
        ss += float((g * g).sum().item())
    return dot, math.sqrt(ss)


def selected_lengths(word_group: torch.Tensor, labels: torch.Tensor) -> list[int]:
    out = []
    selected = labels != -100
    for b in range(labels.shape[0]):
        groups = word_group[b]
        for gid in torch.unique(groups[selected[b] & (groups >= 0)]).tolist():
            k = int(((groups == int(gid)) & selected[b]).sum().item())
            if k > 0:
                out.append(k)
    return out


def build_model_args() -> argparse.Namespace:
    return argparse.Namespace(
        hidden_size=480,
        n_layer=8,
        n_head=8,
        ffn_mult=4,
        max_position_embeddings=512,
        max_seq_length=256,
        max_relative_positions=256,
        position_buckets=256,
        deberta_pos_att_type="p2c,c2p",
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu >= 0:
        # CUDA_VISIBLE_DEVICES should normally be set by the launcher, but this is
        # still useful for the provenance record.  If torch already initialized CUDA,
        # it will not remap devices here.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER_DIR))
    batch_records = load_batch_examples(TRAIN_FILE, args.batch_indices, args.full_batch_size, args.micro_batch_size)
    dataset = base.MaskedChunkDataset([], tokenizer, args.max_seq_length)

    set_seed(43)
    set_seed(43022)
    model = base.build_model(build_model_args(), tokenizer).to(device)
    set_seed(43023)
    model.train()

    state = base.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15, switch_frac=0.7, amlm_window=10, amlm_lambda=0.2)
    state.initialize(vocab_size=len(tokenizer), total_steps=2024)
    gen = torch.Generator(device=device)
    gen.manual_seed(43023)

    results = []
    for j, br in enumerate(batch_records):
        dataset.examples = br["examples"]
        items = [dataset[i] for i in range(len(dataset.examples))]
        batch = base.collate(items)
        input_ids = batch["input_ids"][:, :args.seq_length].contiguous().to(device)
        attention_mask = batch["attention_mask"][:, :args.seq_length].contiguous().to(device)
        word_group = batch["word_group"][:, :args.seq_length].contiguous().to(device)
        state.current_step = int(br["batch_index0"])
        masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, state, gen)
        lens = selected_lengths(word_group.detach().cpu(), labels.detach().cpu())

        # Token mean backward with one fixed dropout realization.
        model.zero_grad(set_to_none=True)
        set_seed(args.dropout_seed + j)
        out_token = model(input_ids=masked_inputs, attention_mask=attention_mask)
        loss_token, _ = wordmod.token_mean_mlm_loss(out_token.logits, labels)
        loss_token.backward()
        token_grads, token_norm = grad_list_and_norm(model)

        # Word mean backward with the same dropout seed, identical masked input/labels.
        model.zero_grad(set_to_none=True)
        set_seed(args.dropout_seed + j)
        out_word = model(input_ids=masked_inputs, attention_mask=attention_mask)
        loss_word, wstats = wordmod.word_mean_mlm_loss(out_word.logits, labels, word_group)
        loss_word.backward()
        dot, word_norm = dot_with(model, token_grads)
        cosine = dot / (token_norm * word_norm) if token_norm > 0 and word_norm > 0 else None

        mean_len = sum(lens) / len(lens) if lens else None
        e_inv = sum(1.0 / k for k in lens) / len(lens) if lens else None
        logit_rms = math.sqrt(mean_len * e_inv) if mean_len and e_inv else None
        results.append({
            "batch_index0": int(br["batch_index0"]),
            "rows": len(br["examples"]),
            "words": int(sum(ex.words for ex in br["examples"])),
            "selected_tokens": int((labels != -100).sum().item()),
            "selected_word_groups": int(wstats["selected_word_groups"]),
            "mean_tokens_per_selected_group": float(wstats["mean_tokens_per_selected_group"]),
            "token_mean_loss": float(loss_token.detach().cpu().item()),
            "word_mean_loss": float(loss_word.detach().cpu().item()),
            "loss_delta_word_minus_token": float((loss_word - loss_token).detach().cpu().item()),
            "parameter_grad_norm_tokenmean": token_norm,
            "parameter_grad_norm_wordmean": word_norm,
            "parameter_grad_norm_ratio_wordmean_over_tokenmean": word_norm / token_norm if token_norm > 0 else None,
            "parameter_grad_cosine_wordmean_vs_tokenmean": cosine,
            "logit_weight_rms_scale_uncorrelated_tokens": logit_rms,
        })
        del token_grads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ratios = [r["parameter_grad_norm_ratio_wordmean_over_tokenmean"] for r in results if r["parameter_grad_norm_ratio_wordmean_over_tokenmean"] is not None]
    cosines = [r["parameter_grad_cosine_wordmean_vs_tokenmean"] for r in results if r["parameter_grad_cosine_wordmean_vs_tokenmean"] is not None]
    return {
        "status": "WORDMEAN_GRADIENT_SCALE_SMOKE",
        "created_utc": now_utc(),
        "purpose": "Small bounded actual-gradient check to help separate relative word-length credit from total update-scale/direction effects for the research word-mean screen.",
        "scope": "No optimizer step and no training; same initialized model, same actual training rows, same masks and dropout seeds for token-mean and word-mean backward passes.",
        "device": str(device),
        "gpu_requested": args.gpu,
        "train_file": str(TRAIN_FILE),
        "tokenizer_dir": str(TOKENIZER_DIR),
        "batch_indices": args.batch_indices,
        "full_batch_size_for_indices": args.full_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "results": results,
        "summary": {
            "n_microbatches": len(results),
            "mean_parameter_grad_norm_ratio_wordmean_over_tokenmean": float(np.mean(ratios)) if ratios else None,
            "p10_parameter_grad_norm_ratio": float(np.percentile(ratios, 10)) if ratios else None,
            "p90_parameter_grad_norm_ratio": float(np.percentile(ratios, 90)) if ratios else None,
            "mean_parameter_grad_cosine": float(np.mean(cosines)) if cosines else None,
            "mean_logit_weight_rms_scale_uncorrelated_tokens": float(np.mean([r["logit_weight_rms_scale_uncorrelated_tokens"] for r in results if r["logit_weight_rms_scale_uncorrelated_tokens"] is not None])),
        },
        "interpretation": "If word-mean and token-mean parameter gradients have norm ratio far from 1 or cosine materially below 1, the research unchanged-LR training result combines relative credit reweighting with update-scale/direction changes. This does not invalidate the screen, but it constrains mechanism attribution and motivates scale-aware follow-up only if scores improve.",
    }


def write_outputs(summary: dict[str, Any], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "wordmean_gradient_scale_smoke.json"
    out_md = out_dir / "wordmean_gradient_scale_smoke.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    s = summary["summary"]
    lines = [
        "# research word-mean actual-gradient scale smoke",
        "",
        summary["scope"],
        "",
        f"Device: `{summary['device']}`; microbatches: `{s['n_microbatches']}`; micro batch size: `{summary['micro_batch_size']}`",
        "",
        f"- Mean parameter gradient norm ratio wordmean/tokenmean: `{s['mean_parameter_grad_norm_ratio_wordmean_over_tokenmean']:.4f}` (p10 `{s['p10_parameter_grad_norm_ratio']:.4f}`, p90 `{s['p90_parameter_grad_norm_ratio']:.4f}`)",
        f"- Mean parameter gradient cosine: `{s['mean_parameter_grad_cosine']:.4f}`",
        f"- Mean uncorrelated logit-weight RMS scale: `{s['mean_logit_weight_rms_scale_uncorrelated_tokens']:.4f}`",
        "",
        "| batch idx | rows | words | selected tokens | selected groups | mean k | token loss | word loss | grad norm ratio | grad cosine | logit RMS scale |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["results"]:
        lines.append(f"| {r['batch_index0']} | {r['rows']} | {r['words']} | {r['selected_tokens']} | {r['selected_word_groups']} | {r['mean_tokens_per_selected_group']:.4f} | {r['token_mean_loss']:.4f} | {r['word_mean_loss']:.4f} | {r['parameter_grad_norm_ratio_wordmean_over_tokenmean']:.4f} | {r['parameter_grad_cosine_wordmean_vs_tokenmean']:.4f} | {r['logit_weight_rms_scale_uncorrelated_tokens']:.4f} |")
    lines += ["", "## Interpretation", "", summary["interpretation"], "", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "summary": summary["summary"]}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch-indices", type=int, nargs="*", default=[0, 256, 1024, 1800])
    ap.add_argument("--full-batch-size", type=int, default=256)
    ap.add_argument("--micro-batch-size", type=int, default=8)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--dropout-seed", type=int, default=620620)
    args = ap.parse_args()
    summary = analyze(args)
    write_outputs(summary, pathlib.Path(args.out_dir))


if __name__ == "__main__":
    main()
