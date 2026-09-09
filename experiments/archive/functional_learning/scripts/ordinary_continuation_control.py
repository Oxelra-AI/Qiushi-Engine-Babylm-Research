#!/usr/bin/env python3
"""research: Matched ordinary continuation control.

Same parent (coherent86), same 80-update row schedule, same optimizer settings,
standard 15% WWM on ALL rows uniformly. NO dense-mask clue suppression, NO
focus/ordinary weight split, NO parent-function preservation KL.

Purpose: isolate whether the v5 improvement comes from the specific intervention
(clue suppression + anchoring) versus simply more training on the same rows.

Key design:
- Imports research infrastructure WITHOUT the research dense-mask monkeypatch
- Uses the SAME prefix construction and row order as the clean trainer
- Applies standard 15% WWM to all positions in all rows
- Computes UNIFORM CE loss (no focus λ weighting)
- No teacher model, no KL term
- Same learning rate schedule, optimizer, and checkpointing

The comparison:
- ordinary_control vs coherent86 → effect of 80 more training updates
- clean vs ordinary_control → effect of clue suppression + preservation
- exact (M,S) vs ordinary_control → effect of clue suppression alone
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Import base infrastructure — deliberately NOT importing research (no dense-mask patch)
import real_stream_train_weighted as s64  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/ordinary_continuation_control')
PARENT_EXPOSURE_WORDS = 86_005_295


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def setup_cache(out_dir: pathlib.Path) -> None:
    hf = out_dir / "hf_cache"
    os.environ["HF_HOME"] = str(hf.resolve())
    os.environ["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    os.environ["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for sub in ["hub", "datasets", "transformers", "modules"]:
        (hf / sub).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Matched ordinary continuation control")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--train-seed", type=int, default=62064)
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--private-scale", type=float, default=0.75)
    # Same schedule offset as clean trainer for identical LR trajectory
    ap.add_argument("--schedule-offset", type=int, default=100)
    ap.add_argument("--schedule-total", type=int, default=200)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--rng-record-updates", type=int, default=3)
    # focus_lambda is exposed for documentation; forced to 1.0 (uniform) internally
    ap.add_argument("--focus-lambda", type=float, default=0.15,
                    help="Documented: what the clean trainer used. This script forces uniform weighting internally.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_cache(out_dir)

    device = torch.device(args.device)
    objective = "correspondence_focus_weighted"

    # ── Load tokenizer and prefix rows (same API as clean trainer) ────────
    TAIL_JSONL = _public_path('experiments/archive/functional_learning/data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl')
    rows, prefix_info = s64.load_prefix(str(TAIL_JSONL), int(args.max_updates), int(args.words_per_update))
    tokenizer = bridge.AutoTokenizer.from_pretrained(
        str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = bridge.WordGroupBuilder(tokenizer)
    print(json.dumps({"event": "prefix_loaded", "rows": len(rows),
                       "prefix_words": prefix_info.get("prefix_words"),
                       "qwen_rows": prefix_info.get("qwen_rows", 0)}, ensure_ascii=False), flush=True)

    # ── Load model (student only, no teacher) ────────────────────────────
    model, _missing, _unexpected = bridge.load_model(device, float(args.private_scale))
    ident = bridge.model_identity(model)
    print(json.dumps({"event": "model_loaded", "class": ident["class"],
                       "params": ident["total_params"], "private_params": ident["private_adapter_params"],
                       "private_scales": ident["executed_private_scales"]}, ensure_ascii=False), flush=True)

    # ── Same seed as the clean trainer for identical row schedule ─────────
    acq_seed = int(s64.stable_seed(bytes((115, 116, 101, 112, 48, 54, 52, 45, 114, 101, 97, 108, 45, 115, 116, 114, 101, 97, 109)).decode('utf-8'), objective,
                                    int(args.train_seed), float(args.focus_lambda)))
    random.seed(acq_seed)
    torch.manual_seed(acq_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(acq_seed)

    # ── Setup optimizer (same structure as clean trainer) ─────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=0.01)
    opt_info = {"trainable_tensors": sum(1 for p in model.parameters() if p.requires_grad),
                "trainable_params": sum(p.numel() for p in trainable)}

    # ── Record config ────────────────────────────────────────────────────
    config = {
        "status": "ORDINARY_CONTINUATION_CONTROL",
        "created_utc": now(),
        "method": "ordinary_continuation_uniform_wwm_no_clue_suppression_no_preservation",
        "purpose": "Matched control for the clean preservation policy: same rows, same optimizer, standard 15% WWM, uniform loss, no dense-mask patch, no parent KL",
        "comparison_targets": [
            "clean (M,S)+KL: clue suppression + preservation",
            "exact (M,S): clue suppression only",
            "coherent86: parent (no additional training)",
        ],
        "masking": "standard 15% WWM on ALL rows uniformly (NO dense-mask patch)",
        "loss_weighting": "uniform CE on all masked positions (NO focus/ordinary split)",
        "preservation": "NONE (no teacher, no KL)",
        "train_seed": int(args.train_seed),
        "acquisition_global_seed": acq_seed,
        "max_updates": int(args.max_updates),
        "mask_prob": float(args.mask_prob),
        "lr": float(args.lr),
        "schedule_offset": int(args.schedule_offset),
        "schedule_total": int(args.schedule_total),
        "words_per_update": int(args.words_per_update),
        "private_scale": float(args.private_scale),
        "parent_path": rel(bridge.PARENT_PATH),
        "student_identity": ident,
        "optimizer": {"trainable_tensors": opt_info["trainable_tensors"],
                      "trainable_params": opt_info["trainable_params"]},
    }
    (out_dir / "train_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "train_start", "method": config["method"],
                       "rows": len(rows), "prefix_words": prefix_info.get("prefix_words"),
                       "device": str(device), "seed": acq_seed}, ensure_ascii=False), flush=True)

    # ── Training loop ────────────────────────────────────────────────────
    cursor = 0
    cum_words = 0
    logs: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    t0 = time.time()
    micro_bs = int(args.micro_batch)
    model.train()

    for update_i in range(int(args.max_updates)):
        # Same macro-batch composition as clean trainer
        macro_rows: List[Dict[str, Any]] = []
        words = 0
        while cursor < len(rows) and words < int(args.words_per_update):
            r = rows[cursor]
            macro_rows.append(r)
            words += int(r.get("words", s64.wc(r.get("text", ""))))
            cursor += 1
        if not macro_rows:
            print(json.dumps({"event": "data_exhausted", "update": update_i}), flush=True)
            break

        schedule_idx = int(args.schedule_offset) + update_i
        lr = s64.lr_at_update(schedule_idx, int(args.schedule_total),
                               int(args.warmup), float(args.lr))
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Standard prepare_macro WITHOUT research dense-mask monkeypatch.
        # This applies ordinary 15% WWM to all rows.
        examples, prep = s64.prepare_macro(
            macro_rows, tokenizer, int(args.seq_length), wgb, objective,
            int(args.train_seed), float(args.mask_prob),
            # focus_prob and max_focus_groups_per_row are passed but irrelevant
            # because without the s75 patch, standard masking applies
            0.15, 999
        )

        # Count ALL masked targets (ignore focus/ordinary component labels)
        total_targets = 0
        for ex in examples:
            total_targets += int((ex["labels"] != -100).sum())
        if total_targets <= 0:
            raise RuntimeError(f"No targets at update {update_i}")

        # Forward and backward with UNIFORM loss
        opt.zero_grad(set_to_none=True)
        total_loss_sum = 0.0
        total_seen = 0

        for start in range(0, len(examples), micro_bs):
            mb = examples[start:start + micro_bs]
            inp = torch.stack([x["input_ids"] for x in mb]).to(device)
            att = torch.stack([x["attention_mask"] for x in mb]).to(device)
            lab = torch.stack([x["labels"] for x in mb]).to(device)
            out = model(input_ids=inp, attention_mask=att)
            V = out.logits.shape[-1]
            ce = F.cross_entropy(out.logits.reshape(-1, V), lab.reshape(-1),
                                 ignore_index=-100, reduction="none").view_as(lab)
            mask = lab != -100
            batch_sum = ce[mask].sum()
            batch_seen = int(mask.sum())
            # UNIFORM: loss = sum(CE) / total_targets
            loss = batch_sum / float(total_targets)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"bad loss at update {update_i}")
            loss.backward()
            total_loss_sum += float(batch_sum.detach().cpu())
            total_seen += batch_seen
            del inp, att, lab, out, ce, mask, batch_sum, loss

        if total_seen != total_targets:
            raise RuntimeError(f"target mismatch update {update_i}: {total_targets} vs {total_seen}")

        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(args.max_grad_norm)).detach().cpu())
        opt.step()
        cum_words += words
        mean_ce = total_loss_sum / total_targets

        log = {
            "update": update_i + 1,
            "schedule_idx": schedule_idx,
            "lr": lr,
            "rows": len(macro_rows),
            "words": words,
            "cum_words": cum_words,
            "total_targets": total_targets,
            "mean_ce": mean_ce,
            "grad_norm_preclip": grad_norm,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        logs.append(log)

        if (update_i + 1) % 10 == 0 or update_i == 0:
            print(json.dumps({"event": "update", **log}, ensure_ascii=False), flush=True)

        # Checkpoint
        if ((update_i + 1) % int(args.checkpoint_every) == 0 or
                update_i + 1 == int(args.max_updates)):
            ckpt_dir = out_dir / "checkpoints" / f"update_{update_i + 1:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            ckpt_info = {
                "update": update_i + 1,
                "cum_words": cum_words,
                "endpoint_exposure_conservative": PARENT_EXPOSURE_WORDS + cum_words,
                "mean_ce": mean_ce,
                "path": str(ckpt_dir),
            }
            checkpoints.append(ckpt_info)
            print(json.dumps({"event": "checkpoint", **ckpt_info}, ensure_ascii=False), flush=True)

    # ── Save summary ─────────────────────────────────────────────────────
    summary = {
        "status": "ORDINARY_CONTINUATION_COMPLETE",
        "method": config["method"],
        "train_seed": int(args.train_seed),
        "acquisition_global_seed": acq_seed,
        "update": len(logs),
        "cum_words": cum_words,
        "endpoint_exposure_conservative": PARENT_EXPOSURE_WORDS + cum_words,
        "final_mean_ce": logs[-1]["mean_ce"] if logs else None,
        "model_identity": ident,
        "prefix_info": {
            "prefix_rows": prefix_info.get("prefix_rows"),
            "prefix_words": prefix_info.get("prefix_words"),
            "qwen_rows": prefix_info.get("qwen_rows", 0),
            "source_rows": prefix_info.get("source_rows"),
        },
        "checkpoints": checkpoints,
        "training_logs": logs,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "training_complete",
                       "cum_words": cum_words,
                       "updates": len(logs),
                       "final_ce": logs[-1]["mean_ce"] if logs else None,
                       "endpoint_exposure": PARENT_EXPOSURE_WORDS + cum_words},
                      ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
