#!/usr/bin/env python3
"""research: matched-batch credit-scale analysis for word-mean MLM.

Purpose
-------
The research word-mean MLM run changes the selected-token loss from a token mean

    L_token = (1/T) sum_{selected tokens t} CE_t

to a selected-whole-word-group mean

    L_word = (1/G) sum_{selected groups g} (1/k_g) sum_{t in g} CE_t.

This is not only a relative reweighting of BPE pieces: on a matched masked batch it
also changes the squared weight norm of the logit-level gradient whenever selected
word groups have unequal BPE lengths.  This CPU-only script quantifies that
confound on actual research-tokenizer compact-view-reinvest training batches before
any official score interpretation is made.

No model forward pass, no official evaluation text, no GPU, and no training-data
change are used here.  The analysis is about the weighting geometry induced by the
objective on the existing legal training stream.
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
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
BASE_TRAINER_PATH = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TRAIN_FILE = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
DEFAULT_OUT = WORKSPACE / "data/wordmean_credit_scale_analysis"


def load_base_trainer():
    spec = importlib.util.spec_from_file_location("compact_experience_masking_curriculum_trainer_base", BASE_TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base trainer from {BASE_TRAINER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_base_trainer()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def finite(x: float | None) -> float | None:
    if x is None:
        return None
    return float(x) if math.isfinite(float(x)) else None


def pct(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:.3f}%"


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.{nd}f}"


def choose_sample_batches(total_batches: int, front_batches: int, stride_batches: int, seed: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    front_n = min(front_batches, total_batches)
    for i in range(front_n):
        chosen.append({"batch_index0": i, "sample_kind": "front"})
    if stride_batches > 0 and total_batches > 0:
        if stride_batches == 1:
            idxs = [total_batches // 2]
        else:
            idxs = sorted({int(round(x)) for x in np.linspace(0, total_batches - 1, stride_batches)})
        # Avoid duplicating front batches unless the whole run is tiny.
        front_set = {x["batch_index0"] for x in chosen}
        for i in idxs:
            if i not in front_set:
                chosen.append({"batch_index0": int(i), "sample_kind": "stride"})
    # deterministic order by actual training order; masking RNG is only used for sampled batches
    chosen.sort(key=lambda r: r["batch_index0"])
    return chosen


def tensor_group_lengths(word_group_1d: torch.Tensor, attention_mask_1d: torch.Tensor, input_ids_1d: torch.Tensor, special_ids: set[int]) -> list[int]:
    counts: dict[int, int] = defaultdict(int)
    for gid, attn, tid in zip(word_group_1d.tolist(), attention_mask_1d.tolist(), input_ids_1d.tolist()):
        if attn == 0 or int(gid) < 0 or int(tid) in special_ids:
            continue
        counts[int(gid)] += 1
    return [counts[k] for k in sorted(counts)]


def selected_group_lengths(word_group: torch.Tensor, labels: torch.Tensor) -> list[int]:
    out: list[int] = []
    bsz = labels.shape[0]
    selected = labels != -100
    for b in range(bsz):
        groups = word_group[b]
        sel_groups = torch.unique(groups[selected[b] & (groups >= 0)])
        for gid in sel_groups.tolist():
            k = int(((groups == int(gid)) & selected[b]).sum().item())
            if k > 0:
                out.append(k)
    return out


def summarize_counter(counter: Counter[int]) -> dict[str, Any]:
    n_groups = int(sum(counter.values()))
    n_tokens = int(sum(k * v for k, v in counter.items()))
    if n_groups == 0:
        return {"n_groups": 0, "n_tokens": 0}
    mean_len = n_tokens / n_groups
    e_inv = sum((1.0 / k) * v for k, v in counter.items()) / n_groups
    e_k2 = sum((k * k) * v for k, v in counter.items()) / n_groups
    # If per-position logit gradients are treated as independent with comparable norms,
    # token-mean has squared weight norm 1/T and word-mean has sum_g k*(1/G/k)^2.
    rms_scale_uncorrelated = math.sqrt(mean_len * e_inv)
    # If all token gradients inside a selected word group were perfectly collinear and
    # equal-norm, a group-level aggregate update would scale as below. Real parameter
    # gradients lie between/around these approximations depending on correlations.
    group_coherent_scale = mean_len / math.sqrt(e_k2) if e_k2 > 0 else None
    table = []
    for k in sorted(counter):
        group_frac = counter[k] / n_groups
        token_frac = (k * counter[k]) / n_tokens
        wordmean_credit_frac = group_frac
        tokenmean_credit_frac = token_frac
        per_token_weight_ratio = mean_len / k
        table.append({
            "k_tokens_in_word_group": int(k),
            "n_groups": int(counter[k]),
            "group_fraction": group_frac,
            "token_fraction_or_tokenmean_credit_fraction": token_frac,
            "wordmean_credit_fraction": wordmean_credit_frac,
            "credit_fraction_delta_wordmean_minus_tokenmean": wordmean_credit_frac - tokenmean_credit_frac,
            "per_token_weight_ratio_wordmean_over_tokenmean": per_token_weight_ratio,
        })
    return {
        "n_groups": n_groups,
        "n_tokens": n_tokens,
        "mean_tokens_per_group": mean_len,
        "mean_inverse_group_length": e_inv,
        "mean_group_length_squared": e_k2,
        "rms_logit_gradient_weight_scale_uncorrelated_tokens": rms_scale_uncorrelated,
        "lr_multiplier_to_match_uncorrelated_logit_weight_rms": 1.0 / rms_scale_uncorrelated if rms_scale_uncorrelated else None,
        "group_coherent_gradient_weight_scale_approx": group_coherent_scale,
        "lr_multiplier_to_match_group_coherent_weight_rms": 1.0 / group_coherent_scale if group_coherent_scale else None,
        "length_table": table,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER_DIR))
    examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(TRAIN_FILE, int(args.max_word_exposure))
    actual_words = sum(ex.words for ex in examples)
    if actual_words != int(args.max_word_exposure):
        raise RuntimeError(f"Loaded words {actual_words} != requested {args.max_word_exposure}")
    total_batches = math.ceil(len(examples) / args.batch_size)
    chosen_batches = choose_sample_batches(total_batches, args.front_batches, args.stride_batches, args.sample_seed)
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    state = base.MaskingCurriculumState(
        curriculum="wwm_fixed",
        mask_prob_start=args.mask_prob,
        mask_prob_end=args.mask_prob,
        switch_frac=0.7,
        amlm_window=10,
        amlm_lambda=0.2,
    )
    state.initialize(vocab_size=len(tokenizer), total_steps=total_batches)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.train_rng_seed)

    candidate_counter: Counter[int] = Counter()
    selected_counter: Counter[int] = Counter()
    per_batch = []
    special_ids = set(tokenizer.all_special_ids)

    t0 = time.time()
    for j, rec in enumerate(chosen_batches, 1):
        bi = int(rec["batch_index0"])
        start = bi * args.batch_size
        end = min(len(examples), start + args.batch_size)
        items = [dataset[i] for i in range(start, end)]
        batch = base.collate(items)
        input_ids = batch["input_ids"][:, : args.seq_length].contiguous()
        attention_mask = batch["attention_mask"][:, : args.seq_length].contiguous()
        word_group = batch["word_group"][:, : args.seq_length].contiguous()

        for b in range(input_ids.shape[0]):
            for k in tensor_group_lengths(word_group[b], attention_mask[b], input_ids[b], special_ids):
                candidate_counter[k] += 1

        state.current_step = bi
        masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, state, gen)
        del masked_inputs
        sel_lengths = selected_group_lengths(word_group, labels)
        for k in sel_lengths:
            selected_counter[k] += 1
        T = int((labels != -100).sum().item())
        G = len(sel_lengths)
        if G > 0 and T > 0:
            mean_len = T / G
            e_inv = sum(1.0 / k for k in sel_lengths) / G
            e_k2 = sum(k * k for k in sel_lengths) / G
            rms_uncorr = math.sqrt(mean_len * e_inv)
            group_coh = mean_len / math.sqrt(e_k2) if e_k2 > 0 else None
        else:
            mean_len = None
            rms_uncorr = None
            group_coh = None
        per_batch.append({
            "batch_index0": bi,
            "sample_kind": rec["sample_kind"],
            "rows": end - start,
            "words": int(sum(ex.words for ex in examples[start:end])),
            "selected_tokens": T,
            "selected_word_groups": G,
            "mean_tokens_per_selected_group": finite(mean_len),
            "rms_logit_gradient_weight_scale_uncorrelated_tokens": finite(rms_uncorr),
            "group_coherent_gradient_weight_scale_approx": finite(group_coh),
        })
        if args.progress and (j == 1 or j % args.progress == 0 or j == len(chosen_batches)):
            print(json.dumps({"event": "sample_progress", "sampled_batches": j, "total_sample_batches": len(chosen_batches), "elapsed_sec": round(time.time()-t0, 1)}), flush=True)

    selected_summary = summarize_counter(selected_counter)
    candidate_summary = summarize_counter(candidate_counter)
    batch_means = {
        "mean_selected_tokens": float(np.mean([r["selected_tokens"] for r in per_batch])) if per_batch else None,
        "mean_selected_word_groups": float(np.mean([r["selected_word_groups"] for r in per_batch])) if per_batch else None,
        "mean_tokens_per_selected_group": float(np.mean([r["mean_tokens_per_selected_group"] for r in per_batch if r["mean_tokens_per_selected_group"] is not None])) if per_batch else None,
        "mean_rms_logit_gradient_weight_scale_uncorrelated_tokens": float(np.mean([r["rms_logit_gradient_weight_scale_uncorrelated_tokens"] for r in per_batch if r["rms_logit_gradient_weight_scale_uncorrelated_tokens"] is not None])) if per_batch else None,
        "p10_rms_scale_uncorrelated": float(np.percentile([r["rms_logit_gradient_weight_scale_uncorrelated_tokens"] for r in per_batch if r["rms_logit_gradient_weight_scale_uncorrelated_tokens"] is not None], 10)) if per_batch else None,
        "p90_rms_scale_uncorrelated": float(np.percentile([r["rms_logit_gradient_weight_scale_uncorrelated_tokens"] for r in per_batch if r["rms_logit_gradient_weight_scale_uncorrelated_tokens"] is not None], 90)) if per_batch else None,
        "mean_group_coherent_gradient_weight_scale_approx": float(np.mean([r["group_coherent_gradient_weight_scale_approx"] for r in per_batch if r["group_coherent_gradient_weight_scale_approx"] is not None])) if per_batch else None,
    }
    return {
        "status": "WORDMEAN_CREDIT_SCALE_ANALYSIS",
        "created_utc": now_utc(),
        "purpose": "Separate relative whole-word credit reweighting from total gradient-scale effects before interpreting the research word-mean training screen.",
        "scope": "CPU-only actual training-stream batches; no model forward pass, official evaluation text, GPU, corpus/tokenizer change, or new training.",
        "train_file": str(TRAIN_FILE),
        "tokenizer_dir": str(TOKENIZER_DIR),
        "max_word_exposure": int(args.max_word_exposure),
        "loaded_examples": len(examples),
        "loaded_words": actual_words,
        "jsonl_total_rows_seen_by_loader": jsonl_total_rows,
        "jsonl_total_words_seen_by_loader": jsonl_total_words,
        "sample_rows_without_text": sample_rows,
        "batch_size": args.batch_size,
        "total_batches": total_batches,
        "sampled_batches": len(chosen_batches),
        "front_batches": args.front_batches,
        "stride_batches_requested": args.stride_batches,
        "sample_seed": args.sample_seed,
        "train_rng_seed_for_sampled_masks": args.train_rng_seed,
        "mask_prob": args.mask_prob,
        "selected_group_length_summary": selected_summary,
        "candidate_visible_group_length_summary": candidate_summary,
        "per_batch_summary": batch_means,
        "per_batch_records": per_batch,
        "interpretation": {
            "loss_weight_identity": "Token-mean weight per selected BPE token is 1/T; word-mean weight is 1/(G*k_g), so the per-token weight ratio is mean_selected_tokens_per_group/k_g. Sum of weights remains one, but squared weight norm generally changes.",
            "gradient_scale_caution": "The uncorrelated-position logit-gradient scale is sqrt(mean_k * E_group[1/k]); a scalar learning-rate match would require roughly its inverse. Parameter-gradient scale also depends on within-word gradient correlations, so a single unchanged learning rate does not isolate relative reweighting from update-scale effects.",
            "route_use": "If research improves cheap columns, follow-up should check whether the gain survives a scale-matched or learning-rate-adjusted variant before attributing it solely to tokenizer-invariant credit allocation. If it fails, the compact-view data mechanism remains preserved and this objective should not be extended by retuning alone.",
        },
    }


def write_outputs(summary: dict[str, Any], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "wordmean_credit_scale_analysis.json"
    out_md = out_dir / "wordmean_credit_scale_analysis.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sel = summary["selected_group_length_summary"]
    cand = summary["candidate_visible_group_length_summary"]
    pb = summary["per_batch_summary"]
    lines = [
        "# research word-mean MLM credit-scale analysis",
        "",
        summary["scope"],
        "",
        "## Why this matters",
        "",
        "On the same masked batch, token-mean MLM uses weight `1/T` for every selected BPE token. The research word-mean objective uses `1/(G*k)` for a selected token inside a selected whole-word group of BPE length `k`. The per-token ratio is therefore `mean(k_selected)/k`. This changes both relative credit across word lengths and the squared weight norm of the logit-level gradient, so unchanged learning rate is not a pure isolation of credit allocation.",
        "",
        "## Sample",
        "",
        f"- Loaded words: `{summary['loaded_words']}` from `{summary['train_file']}`",
        f"- Total batches: `{summary['total_batches']}`; sampled batches: `{summary['sampled_batches']}` (front `{summary['front_batches']}`, stride request `{summary['stride_batches_requested']}`)",
        f"- Mask probability: `{summary['mask_prob']}`; tokenizer: `{summary['tokenizer_dir']}`",
        "",
        "## Selected masked word groups",
        "",
        f"- Selected groups: `{sel.get('n_groups')}`; selected tokens: `{sel.get('n_tokens')}`; mean tokens/group: `{fmt(sel.get('mean_tokens_per_group'), 4)}`",
        f"- Uncorrelated-position logit-gradient RMS weight scale, wordmean/tokenmean: `{fmt(sel.get('rms_logit_gradient_weight_scale_uncorrelated_tokens'), 4)}`; LR multiplier to match that scale: `{fmt(sel.get('lr_multiplier_to_match_uncorrelated_logit_weight_rms'), 4)}`",
        f"- Perfect within-word group-coherent approximation scale, wordmean/tokenmean: `{fmt(sel.get('group_coherent_gradient_weight_scale_approx'), 4)}`; LR multiplier under that approximation: `{fmt(sel.get('lr_multiplier_to_match_group_coherent_weight_rms'), 4)}`",
        "",
        "### Credit by selected word length",
        "",
        "| k BPE tokens in selected word | groups | group frac = wordmean credit | token frac = tokenmean credit | Δ wordmean-tokenmean | per-token weight ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sel.get("length_table", []):
        lines.append(
            f"| {row['k_tokens_in_word_group']} | {row['n_groups']} | {pct(row['group_fraction'])} | {pct(row['token_fraction_or_tokenmean_credit_fraction'])} | {pct(row['credit_fraction_delta_wordmean_minus_tokenmean'])} | {fmt(row['per_token_weight_ratio_wordmean_over_tokenmean'], 3)} |"
        )
    lines += [
        "",
        "## Visible candidate word groups before masking",
        "",
        f"- Candidate visible groups: `{cand.get('n_groups')}`; candidate tokens: `{cand.get('n_tokens')}`; mean tokens/group: `{fmt(cand.get('mean_tokens_per_group'), 4)}`",
        "",
        "| k BPE tokens in visible word | groups | group frac | token frac | wordmean-tokenmean credit shift if selected uniformly | per-token ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cand.get("length_table", []):
        lines.append(
            f"| {row['k_tokens_in_word_group']} | {row['n_groups']} | {pct(row['group_fraction'])} | {pct(row['token_fraction_or_tokenmean_credit_fraction'])} | {pct(row['credit_fraction_delta_wordmean_minus_tokenmean'])} | {fmt(row['per_token_weight_ratio_wordmean_over_tokenmean'], 3)} |"
        )
    lines += [
        "",
        "## Per-batch scale summary",
        "",
        f"- Mean selected tokens/batch: `{fmt(pb.get('mean_selected_tokens'), 2)}`; mean selected groups/batch: `{fmt(pb.get('mean_selected_word_groups'), 2)}`; mean selected tokens/group: `{fmt(pb.get('mean_tokens_per_selected_group'), 4)}`",
        f"- Mean uncorrelated logit-gradient RMS scale: `{fmt(pb.get('mean_rms_logit_gradient_weight_scale_uncorrelated_tokens'), 4)}` (p10 `{fmt(pb.get('p10_rms_scale_uncorrelated'), 4)}`, p90 `{fmt(pb.get('p90_rms_scale_uncorrelated'), 4)}`)",
        f"- Mean group-coherent approximation scale: `{fmt(pb.get('mean_group_coherent_gradient_weight_scale_approx'), 4)}`",
        "",
        "## Interpretation for the running research screen",
        "",
        "The research run remains a useful 80M screen of a general legal objective, but its result must not be read as pure evidence for relative word credit alone. It upweights one-piece words and downweights multi-piece words on matched selected WWM batches, while also changing the RMS norm of the logit-level loss weights. A positive result should be followed by scale-aware comparison before firm mechanism attribution; a negative result should stop this objective family rather than erode the already validated compact-view reinvestment data mechanism.",
        "",
        f"Full JSON: `{out_json}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "sampled_batches": summary["sampled_batches"]}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-word-exposure", type=int, default=80_000_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--front-batches", type=int, default=32)
    ap.add_argument("--stride-batches", type=int, default=96)
    ap.add_argument("--sample-seed", type=int, default=62062)
    ap.add_argument("--train-rng-seed", type=int, default=43023)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--progress", type=int, default=16)
    args = ap.parse_args()
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    summary = analyze(args)
    write_outputs(summary, pathlib.Path(args.out_dir))


if __name__ == "__main__":
    main()
