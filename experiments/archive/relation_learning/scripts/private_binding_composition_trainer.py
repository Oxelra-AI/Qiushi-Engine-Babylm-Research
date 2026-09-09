#!/usr/bin/env python3
"""research/083: private-phase binding/composition trainer.

This trainer tests whether the mechanism learned in the local binding coordinate can
be made useful for the BabyLM Strict-Small endpoint under the same frozen trunk and
private-parameter accounting as coherent86.  It can start either from the verified
chck_82M slow function with a fresh private branch, or from coherent86's trained
private branch.  It interleaves:

  * ordinary suffix WWM CE, optionally with private-on/private-off KL, as in coherent86;
  * KL-only ordinary preservation batches;
  * balanced binding answer-credit CE.

research changes the binding objective from a packet-format specialist objective to a
leashed relation objective: on binding rows, private-off/private-on KL is applied to
all non-answer positions while answer CE is applied only to the answer tokens.  The
default binding rows are the deterministic frame-varied rows built in research when
available.  A frame-varied "epoch" is charged as a single-frame reference epoch by
sampling a balanced subset across frames, so 25 reference epochs on five frames has
approximately the same word charge as 25 single-frame epochs but distributes practice
across phrasings.
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
import pathlib
import random
import shutil
import sys
import time
from collections import Counter, defaultdict, deque
from typing import Any

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/relation_learning')
WS = _public_path('experiments/archive/relation_learning')
REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
FUNCTIONAL_RELATION_STUDIES_SCRIPTS = _public_path('experiments/archive/relation_learning/scripts')
sys.path.insert(0, str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS))
sys.path.insert(0, str(FUNCTIONAL_RELATION_STUDIES_SCRIPTS))

import frozen82_fastpath_replay_trainer as S150  # noqa: E402
S73 = None  # imported inside main after a local writable cache is configured

CHCK82 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
COHERENT86 = _public_path('models/frontier')
DEFAULT_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
ORIGINAL_TRAIN_ROWS = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_train.jsonl')
ORIGINAL_HELD_ROWS = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_heldout.jsonl')
ORIGINAL_BINDING_PAIRS = _public_path('experiments/archive/relation_learning/data/recombination_rows/binding_pairs_heldout.jsonl')
FRAME_DIR = _public_path('experiments/archive/relation_learning/data/frame_varied_recombination_rows')
FRAME_TRAIN_ROWS = _public_path('experiments/archive/relation_learning/data/frame_varied_recombination_rows/recombination_train_frame_seen.jsonl')
FRAME_HELD_ROWS = _public_path('experiments/archive/relation_learning/data/frame_varied_recombination_rows/recombination_heldout_frame_all.jsonl')
FRAME_BINDING_PAIRS = _public_path('experiments/archive/relation_learning/data/frame_varied_recombination_rows/binding_pairs_heldout_frame_all.jsonl')

CHCK82_WORDS = 82_012_495
COHERENT86_WORDS = 86_005_295
BINDING_EPOCH_WORDS_REFERENCE = 156_634
BINDING_REFERENCE_PAIRS = 1_666


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def default_binding_train_rows() -> pathlib.Path:
    return FRAME_TRAIN_ROWS if FRAME_TRAIN_ROWS.exists() else ORIGINAL_TRAIN_ROWS


def default_binding_held_rows() -> pathlib.Path:
    return FRAME_HELD_ROWS if FRAME_HELD_ROWS.exists() else ORIGINAL_HELD_ROWS


def default_binding_pairs() -> pathlib.Path:
    return FRAME_BINDING_PAIRS if FRAME_BINDING_PAIRS.exists() else ORIGINAL_BINDING_PAIRS


def norm_path(pathlike: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(pathlike)
    return p if p.is_absolute() else ROOT / p


def setup_cache(out: pathlib.Path) -> None:
    base = out / "hf_cache"
    for k, p in {
        "HF_HOME": base / "hf_home",
        "HF_HUB_CACHE": base / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": base / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": base / "transformers",
        "HF_MODULES_CACHE": base / "modules",
        "HF_DATASETS_CACHE": base / "datasets",
        "TMPDIR": base / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def reset(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_start_model(endpoint: pathlib.Path, args: argparse.Namespace, device: torch.device):
    # Reuse the research loader.  For chck82 it creates a fresh private branch; for a
    # frozen-private endpoint such as coherent86 it loads the existing private branch.
    model, missing, unexpected = S150.load_frozen_private_model(endpoint, args)
    model.to(device)
    if torch.cuda.is_available():
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    private_names = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
    for n, p in model.named_parameters():
        p.requires_grad_(n in private_names)
    identity = {
        "start_endpoint": rel(endpoint),
        "model_class": type(model).__name__,
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "private_params": int(sum(p.numel() for n, p in model.named_parameters() if n in private_names)),
        "frozen_slow_params": int(sum(p.numel() for n, p in model.named_parameters() if n not in private_names)),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "private_adapter_scale": float(args.private_adapter_scale),
    }
    return model, private_names, identity


def load_ordinary_batches(tokenizer, args: argparse.Namespace) -> list[dict[str, Any]]:
    max_words = max(int(args.main_words), int(args.kl_words), 0)
    if max_words <= 0:
        return []
    examples = S150.load_examples_tail(pathlib.Path(args.example_jsonl), int(args.skip_rows), max_words)
    ds = S150.TailDataset(examples, tokenizer, int(args.seq_length))
    loader = torch.utils.data.DataLoader(ds, batch_size=int(args.batch_size), shuffle=False,
                                         collate_fn=S150.collate, num_workers=0,
                                         pin_memory=torch.cuda.is_available())
    return list(loader)


def tokenize_binding_rows(tokenizer, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    train_path = norm_path(args.binding_train_rows)
    train_rows = S73.read_jsonl(train_path)
    train_rows = [r for r in train_rows if r.get("pair_half") in ("A", "B")]
    toks: list[dict[str, Any]] = []
    raw_by_id: dict[str, dict[str, Any]] = {}
    for r in train_rows:
        t = S73.tokenize_row(tokenizer, r, int(args.seq_length))
        if t is None:
            continue
        # Preserve frame/base metadata discarded by the research tokenizer.
        t["frame_id"] = str(r.get("frame_id") or "single_frame")
        t["frame_split"] = str(r.get("frame_split") or "single_frame")
        t["base_pair_id"] = str(r.get("base_pair_id") or r.get("pair_id"))
        t["context_words"] = int(len(str(r.get("context_text", "")).split()))
        toks.append(t)
        raw_by_id[str(r["row_id"])] = r
    return toks, raw_by_id


def select_epoch_pair_indices(complete: list[tuple[str, list[dict[str, Any]]]], args: argparse.Namespace, rng: random.Random) -> tuple[list[int], dict[str, Any]]:
    if not complete:
        return [], {"selection_mode": "empty"}
    frames: dict[str, list[int]] = defaultdict(list)
    for i, (_pid, rows) in enumerate(complete):
        frame = str(rows[0].get("frame_id") or "single_frame")
        frames[frame].append(i)
    n_frames = len(frames)
    explicit_budget = int(args.binding_reference_pairs_per_epoch)
    if explicit_budget > 0:
        pair_budget = min(explicit_budget, len(complete))
    elif n_frames > 1 and str(args.binding_frame_schedule) == "reference_mixed":
        # One reference epoch means about the same number of pair contexts as the
        # original single-frame row set, spread over all frames.
        pair_budget = min(BINDING_REFERENCE_PAIRS, len(complete))
    else:
        pair_budget = len(complete)
    if pair_budget >= len(complete):
        order = list(range(len(complete)))
        rng.shuffle(order)
        by_frame = Counter(str(complete[i][1][0].get("frame_id") or "single_frame") for i in order)
        return order, {"selection_mode": "full", "pair_budget": pair_budget, "available_pairs": len(complete), "n_frames": n_frames, "selected_by_frame": dict(by_frame)}
    if n_frames <= 1 or str(args.binding_frame_schedule) == "random_subset":
        order = list(range(len(complete)))
        rng.shuffle(order)
        selected = order[:pair_budget]
    else:
        # Balanced frame mixture.  Remainder frame changes naturally with RNG shuffle.
        frame_names = sorted(frames)
        rng.shuffle(frame_names)
        base = pair_budget // n_frames
        rem = pair_budget % n_frames
        selected = []
        for j, fr in enumerate(frame_names):
            ids = list(frames[fr])
            rng.shuffle(ids)
            k = min(len(ids), base + (1 if j < rem else 0))
            selected.extend(ids[:k])
        if len(selected) < pair_budget:
            remaining = [i for i in range(len(complete)) if i not in set(selected)]
            rng.shuffle(remaining)
            selected.extend(remaining[:pair_budget - len(selected)])
        rng.shuffle(selected)
    by_frame = Counter(str(complete[i][1][0].get("frame_id") or "single_frame") for i in selected)
    return selected, {"selection_mode": str(args.binding_frame_schedule), "pair_budget": pair_budget, "available_pairs": len(complete), "n_frames": n_frames, "selected_by_frame": dict(by_frame)}


def binding_epoch_batches(tokenizer, args: argparse.Namespace, epoch: int, rng: random.Random) -> tuple[list[list[dict[str, Any]]], int, dict[str, Any]]:
    train_toks, _raw_by_id = tokenize_binding_rows(tokenizer, args)
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in train_toks:
        pair_groups[t["pair_id"]].append(t)
    complete = [(pid, rows) for pid, rows in sorted(pair_groups.items()) if len(rows) == 2]
    selected_idx, selection = select_epoch_pair_indices(complete, args, rng)
    batches: list[list[dict[str, Any]]] = []
    charged = 0
    for st in range(0, len(selected_idx), int(args.pair_batch_size)):
        idxs = selected_idx[st:st + int(args.pair_batch_size)]
        rows: list[dict[str, Any]] = []
        for idx in idxs:
            _pid, rs = complete[idx]
            rows.extend(rs)
            charged += sum(int(r.get("context_words") or 0) for r in rs)
        if rows:
            batches.append(rows)
    frame_count = Counter(str(r.get("frame_id") or "single_frame") for idx in selected_idx for r in complete[idx][1][:1])
    audit = {
        "train_rows_path": rel(norm_path(args.binding_train_rows)),
        "train_token_rows": len(train_toks),
        "complete_pairs_total": len(complete),
        "selected_pairs": len(selected_idx),
        "charged_words_this_epoch": charged,
        "frames_selected": dict(frame_count),
        **selection,
    }
    return batches, charged, audit


def prepare_binding_eval(tokenizer, max_length: int, args: argparse.Namespace):
    held_rows = S73.read_jsonl(norm_path(args.binding_held_rows))
    held_rows = [r for r in held_rows if r.get("pair_half") in ("A", "B")]
    pairs = S73.read_jsonl(norm_path(args.binding_pairs))
    return held_rows, pairs


def optimizer_for(model, private_names: set[str], args: argparse.Namespace):
    private_up_ids = {id(p) for n, p in model.named_parameters() if n in private_names and ".private_adapter.up." in n}
    normal = [p for n, p in model.named_parameters() if n in private_names and id(p) not in private_up_ids]
    zero_wd = [p for n, p in model.named_parameters() if n in private_names and id(p) in private_up_ids]
    return torch.optim.AdamW([
        {"params": normal, "weight_decay": float(args.weight_decay)},
        {"params": zero_wd, "weight_decay": 0.0},
    ], lr=float(args.learning_rate), betas=(0.9, 0.98), eps=1e-6)


def lr_at(research: int, total: int, warmup_frac: float, peak: float) -> float:
    warmup = max(1, int(total * warmup_frac))
    if research < warmup:
        return peak * research / max(1, warmup)
    p = (research - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(max(p, 0.0), 1.0)))


def ordinary_update(model, tokenizer, optimizer, batch, device: torch.device, gen: torch.Generator, args: argparse.Namespace, update_i: int, total_updates: int, main_ce: bool, neutral_only: bool) -> dict[str, Any]:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention = batch["attention_mask"].to(device, non_blocking=True)
    wg = batch["word_group"].to(device, non_blocking=True)
    words = int(batch["words"].sum().item())
    lr = lr_at(update_i, total_updates, float(args.warmup_fraction), float(args.learning_rate))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss_terms: dict[str, float] = {}
    total_loss = None
    if main_ce:
        masked, labels, _select, _groups = S150.apply_wwm(input_ids, attention, wg, tokenizer, float(args.mask_prob), gen)
        S150.set_private_enabled(model, True)
        out = model(input_ids=masked, attention_mask=attention)
        vocab = out.logits.shape[-1]
        main_sum = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="sum")
        n_targets = int((labels != -100).sum().item())
        main_loss = main_sum / max(1, n_targets)
        total_loss = float(args.main_lambda) * main_loss
        loss_terms["main_loss"] = float(main_loss.detach().cpu())
        loss_terms["main_targets"] = n_targets
        neutral_inputs = masked
    else:
        neutral_inputs = input_ids
    if float(args.neutral_lambda) > 0:
        neutral_loss, neutral_value = S150.compute_neutral_loss(model, neutral_inputs, attention, int(args.neutral_subsample))
        if neutral_loss is not None:
            total_loss = (float(args.neutral_lambda) * neutral_loss) if total_loss is None else total_loss + float(args.neutral_lambda) * neutral_loss
        loss_terms["neutral_loss"] = float(neutral_value)
    if total_loss is None:
        return {"kind": "ordinary_skip", "words": words, "lr": lr}
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    return {"kind": "ordinary_kl" if neutral_only else "ordinary_main", "words": words, "lr": lr, **loss_terms}


def binding_nonanswer_kl(model, masked: torch.Tensor, attention: torch.Tensor, answer_mask: torch.Tensor, max_rows: int) -> tuple[torch.Tensor | None, float, int]:
    n = masked.shape[0] if int(max_rows) <= 0 else min(int(max_rows), masked.shape[0])
    if n <= 0:
        return None, 0.0, 0
    ids = masked[:n]
    att = attention[:n]
    ans = answer_mask[:n]
    keep = att.bool() & ~ans.bool()
    n_pos = int(keep.sum().item())
    if n_pos <= 0:
        return None, 0.0, 0
    was_training = model.training
    model.eval()
    S150.set_private_enabled(model, False)
    with torch.no_grad():
        slow_logits = model(input_ids=ids, attention_mask=att).logits.detach()
    S150.set_private_enabled(model, True)
    out = model(input_ids=ids, attention_mask=att)
    log_p_private = F.log_softmax(out.logits, dim=-1)
    p_slow = F.softmax(slow_logits, dim=-1)
    kl_tok = F.kl_div(log_p_private, p_slow, reduction="none").sum(-1)
    mask_float = keep.float()
    loss = (kl_tok * mask_float).sum() / max(1.0, float(mask_float.sum().item()))
    if was_training:
        model.train()
    return loss, float(loss.detach().cpu()), n_pos


def binding_update(model, tokenizer, optimizer, rows, device: torch.device, pad_id: int, gen: torch.Generator, args: argparse.Namespace, update_i: int, total_updates: int) -> dict[str, Any]:
    batch_t = S73.collate(rows, pad_id)
    batch_t = {k: v.to(device) for k, v in batch_t.items()}
    masked, labels, st = S73.apply_answer_arm(batch_t, tokenizer, "answer_clean")
    lr = lr_at(update_i, total_updates, float(args.warmup_fraction), float(args.learning_rate))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    S150.set_private_enabled(model, True)
    out = model(input_ids=masked, attention_mask=batch_t["attention_mask"], labels=labels)
    answer_loss = out.loss
    total_loss = answer_loss
    kl_value = 0.0
    kl_positions = 0
    if float(args.binding_neutral_lambda) > 0:
        kl_loss, kl_value, kl_positions = binding_nonanswer_kl(
            model, masked, batch_t["attention_mask"], batch_t["answer_mask"], int(args.binding_neutral_subsample)
        )
        if kl_loss is not None:
            total_loss = total_loss + float(args.binding_neutral_lambda) * kl_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    labeled = int((labels != -100).sum().item())
    frames = Counter(str(r.get("frame_id") or "single_frame") for r in rows)
    return {
        "kind": "binding",
        "lr": lr,
        "loss": float(total_loss.detach().cpu()),
        "answer_loss": float(answer_loss.detach().cpu()),
        "binding_nonanswer_kl_loss": kl_value,
        "binding_nonanswer_kl_positions": kl_positions,
        "binding_neutral_lambda": float(args.binding_neutral_lambda),
        "labeled_positions": labeled,
        "frames_in_batch": dict(frames),
        **st,
    }


def main() -> None:
    global S73
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start", choices=["chck82_fresh", "coherent86"], default="chck82_fresh")
    ap.add_argument("--main-words", type=int, default=0)
    ap.add_argument("--kl-words", type=int, default=0)
    ap.add_argument("--binding-epochs", type=int, default=0)
    ap.add_argument("--binding-train-rows", default=str(default_binding_train_rows()))
    ap.add_argument("--binding-held-rows", default=str(default_binding_held_rows()))
    ap.add_argument("--binding-pairs", default=str(default_binding_pairs()))
    ap.add_argument("--binding-frame-schedule", choices=["reference_mixed", "full", "random_subset"], default="reference_mixed")
    ap.add_argument("--binding-reference-pairs-per-epoch", type=int, default=0)
    ap.add_argument("--binding-neutral-lambda", type=float, default=1.0)
    ap.add_argument("--binding-neutral-subsample", type=int, default=0, help="0 means use every row in the binding batch for non-answer KL")
    ap.add_argument("--example-jsonl", default=str(DEFAULT_STREAM))
    ap.add_argument("--skip-rows", type=int, default=530944)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--pair-batch-size", type=int, default=8)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--warmup-fraction", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--main-lambda", type=float, default=1.0)
    ap.add_argument("--neutral-lambda", type=float, default=1.0)
    ap.add_argument("--neutral-subsample", type=int, default=64)
    ap.add_argument("--private-adapter-bottleneck", type=int, default=128)
    ap.add_argument("--private-adapter-scale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=82082)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    setup_cache(out)
    os.environ["CACHE_BASE"] = str((out / "import_cache").resolve())
    import binding_factorial_train as _S73  # noqa: E402
    S73 = _S73
    reset(int(args.seed))
    if args.smoke:
        args.main_words = min(int(args.main_words), 2000)
        args.kl_words = min(int(args.kl_words), 2000)
        args.binding_epochs = min(int(args.binding_epochs), 1)
        args.batch_size = min(int(args.batch_size), 8)
        args.pair_batch_size = min(int(args.pair_batch_size), 2)
        args.neutral_subsample = min(int(args.neutral_subsample), 4)
        if int(args.binding_reference_pairs_per_epoch) <= 0:
            args.binding_reference_pairs_per_epoch = 4
        args.binding_neutral_subsample = min(int(args.binding_neutral_subsample), 4) if int(args.binding_neutral_subsample) > 0 else 4
        args.device = "cpu"

    from transformers import AutoTokenizer

    endpoint = CHCK82 if args.start == "chck82_fresh" else COHERENT86
    initial_words = CHCK82_WORDS if args.start == "chck82_fresh" else COHERENT86_WORDS
    tokenizer = AutoTokenizer.from_pretrained(str(endpoint), local_files_only=True, use_fast=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, private_names, identity = load_start_model(endpoint, args, device)
    optimizer = optimizer_for(model, private_names, args)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))
    rng = random.Random(int(args.seed))

    ordinary_batches = load_ordinary_batches(tokenizer, args)
    # Select ordinary batches by word budgets separately for main and KL-only use.
    main_q: deque[dict[str, Any]] = deque()
    kl_q: deque[dict[str, Any]] = deque()
    w_main = 0
    w_kl = 0
    for b in ordinary_batches:
        bw = int(b["words"].sum().item())
        if w_main < int(args.main_words) and w_main + bw <= int(args.main_words):
            main_q.append(b); w_main += bw
        if w_kl < int(args.kl_words) and w_kl + bw <= int(args.kl_words):
            kl_q.append(b); w_kl += bw
    binding_batches_by_epoch: list[list[list[dict[str, Any]]]] = []
    binding_words_by_epoch: list[int] = []
    binding_audits: list[dict[str, Any]] = []
    for ep in range(int(args.binding_epochs)):
        batches, charged, audit = binding_epoch_batches(tokenizer, args, ep, rng)
        if args.smoke:
            batches = batches[:2]
            charged = sum(sum(int(r.get("context_words") or 0) for r in batch) for batch in batches)
            audit["smoke_truncated_batches"] = len(batches)
            audit["charged_words_this_epoch"] = charged
        binding_batches_by_epoch.append(batches)
        binding_words_by_epoch.append(charged)
        binding_audits.append(audit)

    n_binding_updates = sum(len(x) for x in binding_batches_by_epoch)
    total_updates = n_binding_updates + len(main_q) + len(kl_q)
    if total_updates <= 0:
        raise SystemExit("No updates requested")

    config = {
        "status": "PRIVATE_BINDING_COMPOSITION_CONFIG",
        "created_utc": now(),
        "start": args.start,
        "start_endpoint": rel(endpoint),
        "output_dir": rel(out),
        "initial_consumed_words": initial_words,
        "requested_main_words": int(args.main_words),
        "selected_main_words": w_main,
        "requested_kl_words": int(args.kl_words),
        "selected_kl_words": w_kl,
        "binding_train_rows": rel(norm_path(args.binding_train_rows)),
        "binding_held_rows": rel(norm_path(args.binding_held_rows)),
        "binding_pairs": rel(norm_path(args.binding_pairs)),
        "binding_frame_schedule": str(args.binding_frame_schedule),
        "binding_reference_pairs_per_epoch": int(args.binding_reference_pairs_per_epoch),
        "binding_epochs": int(args.binding_epochs),
        "binding_epoch_words_reference": BINDING_EPOCH_WORDS_REFERENCE,
        "binding_words_by_epoch": binding_words_by_epoch,
        "binding_audits": binding_audits,
        "binding_total_words": int(sum(binding_words_by_epoch)),
        "total_requested_or_selected_tail_words": int(w_main + w_kl + sum(binding_words_by_epoch)),
        "planned_updates": total_updates,
        "identity": identity,
        "objective": "ordinary suffix CE/KL plus binding answer CE with private-off KL on every non-answer binding token; frame-varied rows are default when present",
        "counting_note": "ordinary WWM CE words, ordinary KL-only words, and selected binding row words are all charged as private-phase exposure; frame-varied reference_mixed epochs use a reference-sized balanced subset over frames.",
        "args": vars(args),
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "config", **{k: config[k] for k in ["start", "initial_consumed_words", "selected_main_words", "selected_kl_words", "binding_total_words", "planned_updates", "binding_train_rows", "binding_frame_schedule"]}, "identity": identity}, indent=2), flush=True)

    log_path = out / "training_log.jsonl"
    losses = Counter()
    counts = Counter()
    update_i = 0
    charged_main = charged_kl = charged_binding_running = 0

    # Flatten binding batches with epoch markers and exact batch charges for running logs.
    bind_items: deque[tuple[int, list[dict[str, Any]], int]] = deque()
    for ep, batches in enumerate(binding_batches_by_epoch, 1):
        for b in batches:
            bind_items.append((ep, b, sum(int(r.get("context_words") or 0) for r in b)))

    def run_main_one(logf) -> bool:
        nonlocal update_i, charged_main
        if not main_q:
            return False
        b = main_q.popleft()
        words = int(b["words"].sum().item())
        model.train(); S150.set_private_enabled(model, True)
        rec = ordinary_update(model, tokenizer, optimizer, b, device, gen, args, update_i, total_updates, main_ce=True, neutral_only=False)
        update_i += 1; charged_main += words
        for k in ["main_loss", "neutral_loss"]:
            if k in rec and math.isfinite(float(rec[k])):
                losses[k] += float(rec[k]); counts[k] += 1
        rec.update({"update": update_i, "charged_main_words": charged_main, "charged_kl_words": charged_kl, "charged_binding_words": charged_binding_running})
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if update_i == 1 or update_i % int(args.log_every) == 0:
            print(json.dumps({"event": "train", **rec}), flush=True)
        return True

    def run_kl_one(logf) -> bool:
        nonlocal update_i, charged_kl
        if not kl_q:
            return False
        b = kl_q.popleft()
        words = int(b["words"].sum().item())
        model.train(); S150.set_private_enabled(model, True)
        rec = ordinary_update(model, tokenizer, optimizer, b, device, gen, args, update_i, total_updates, main_ce=False, neutral_only=True)
        update_i += 1; charged_kl += words
        if "neutral_loss" in rec and math.isfinite(float(rec["neutral_loss"])):
            losses["kl_only_loss"] += float(rec["neutral_loss"]); counts["kl_only_loss"] += 1
        rec.update({"update": update_i, "charged_main_words": charged_main, "charged_kl_words": charged_kl, "charged_binding_words": charged_binding_running})
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if update_i == 1 or update_i % int(args.log_every) == 0:
            print(json.dumps({"event": "train", **rec}), flush=True)
        return True

    def run_bind_one(logf) -> bool:
        nonlocal update_i, charged_binding_running
        if not bind_items:
            return False
        ep, rows, bwords = bind_items.popleft()
        model.train(); S150.set_private_enabled(model, True)
        rec = binding_update(model, tokenizer, optimizer, rows, device, pad_id, gen, args, update_i, total_updates)
        update_i += 1
        charged_binding_running += int(bwords)
        rec.update({"update": update_i, "binding_epoch": ep, "charged_main_words": charged_main, "charged_kl_words": charged_kl, "charged_binding_words": charged_binding_running})
        for k, name in [("answer_loss", "binding_answer_loss"), ("binding_nonanswer_kl_loss", "binding_nonanswer_kl_loss"), ("loss", "binding_loss")]:
            if k in rec and math.isfinite(float(rec[k])):
                losses[name] += float(rec[k]); counts[name] += 1
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if update_i == 1 or update_i % int(args.log_every) == 0:
            print(json.dumps({"event": "train", **rec}), flush=True)
        return True

    # Interleave: spread ordinary main and KL-only batches across many small binding updates.
    n_binding_updates_for_schedule = max(1, n_binding_updates)
    main_interval = max(1, math.ceil(n_binding_updates_for_schedule / max(1, len(main_q)))) if main_q else 10**9
    kl_interval = max(1, math.ceil(n_binding_updates_for_schedule / max(1, len(kl_q)))) if kl_q else 10**9
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        bind_seen = 0
        while bind_items or main_q or kl_q:
            did = False
            if main_q and (not bind_items or bind_seen % main_interval == 0):
                did = run_main_one(logf) or did
            if kl_q and (not bind_items or bind_seen % kl_interval == 0):
                did = run_kl_one(logf) or did
            if bind_items:
                did = run_bind_one(logf) or did
                bind_seen += 1
            elif not did:
                # drain whichever ordinary queues remain
                did = run_main_one(logf) or run_kl_one(logf)
            if not did:
                break
            if torch.cuda.is_available() and update_i % 100 == 0:
                torch.cuda.empty_cache()

    charged_binding = int(sum(binding_words_by_epoch))
    # End-of-training frame-aware held-out readout.
    held_rows, pairs = prepare_binding_eval(tokenizer, int(args.seq_length), args)
    eval_result = None
    if not args.smoke:
        eval_result = S73.score_margin_eval(model, tokenizer, held_rows, pairs, device, int(args.seq_length), batch_size=32)
    S150.set_private_enabled(model, True)
    S150.save_checkpoint(model, tokenizer, out / "hf_model" / "final")
    total_tail = int(charged_main + charged_kl + charged_binding)
    metrics = {
        "status": "PRIVATE_BINDING_COMPOSITION",
        "created_utc": now(),
        "mode": pathlib.Path(args.output_dir).name,
        "start": args.start,
        "endpoint": rel(endpoint),
        "initial_consumed_words": initial_words,
        "skip_rows": int(args.skip_rows),
        "tail_main_word_exposure": int(charged_main),
        "tail_aux_word_exposure": int(charged_binding),
        "tail_kl_word_exposure": int(charged_kl),
        "tail_charged_words": total_tail,
        "total_consumed_words": int(initial_words + total_tail),
        "full_cap_words": 100_000_000,
        "stopped_before_cap": bool(initial_words + total_tail < 100_000_000),
        "updates": int(update_i),
        "schedule_total": int(total_updates),
        "trainable": "private_adapter_only",
        "total_params": identity["total_params"],
        "private_params": identity["private_params"],
        "frozen_slow_params": identity["frozen_slow_params"],
        "private_adapter_scale": float(args.private_adapter_scale),
        "binding_train_rows": rel(norm_path(args.binding_train_rows)),
        "binding_held_rows": rel(norm_path(args.binding_held_rows)),
        "binding_pairs": rel(norm_path(args.binding_pairs)),
        "binding_frame_schedule": str(args.binding_frame_schedule),
        "binding_neutral_lambda": float(args.binding_neutral_lambda),
        "first_main_loss": None,
        "final_main_loss": None,
        "mean_main_loss": losses["main_loss"] / counts["main_loss"] if counts["main_loss"] else None,
        "first_aux_loss": None,
        "final_aux_loss": None,
        "mean_aux_loss": losses["binding_loss"] / counts["binding_loss"] if counts["binding_loss"] else None,
        "mean_binding_answer_loss": losses["binding_answer_loss"] / counts["binding_answer_loss"] if counts["binding_answer_loss"] else None,
        "mean_binding_nonanswer_kl_loss": losses["binding_nonanswer_kl_loss"] / counts["binding_nonanswer_kl_loss"] if counts["binding_nonanswer_kl_loss"] else None,
        "aux_loss_batches": int(counts["binding_loss"]),
        "first_neutral_loss": None,
        "final_neutral_loss": None,
        "mean_neutral_loss": losses["neutral_loss"] / counts["neutral_loss"] if counts["neutral_loss"] else (losses["kl_only_loss"] / counts["kl_only_loss"] if counts["kl_only_loss"] else 0.0),
        "deterministic_neutrality_eval_mode": True,
        "binding_eval_final": eval_result,
        "identity": identity,
        "train_config": rel(out / "train_config.json"),
        "training_log": rel(log_path),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": metrics["status"], "out": rel(out), "tail_charged_words": total_tail, "total_consumed_words": metrics["total_consumed_words"], "updates": update_i, "binding_joint": None if eval_result is None else eval_result.get("binding_joint_correct"), "elapsed_sec": metrics["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
