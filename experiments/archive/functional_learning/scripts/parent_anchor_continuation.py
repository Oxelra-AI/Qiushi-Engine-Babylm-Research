#!/usr/bin/env python3
"""research: parent-anchored continuation from exact coherent86.

The research exact-parent continuation kept the learned private tensors but used a neutral
KL target equal to the private-off carrier.  That target can oppose part of the useful
private correction already present in the coherent86 parent.  This wrapper reuses the
research trainer and changes only the neutral KL reference: on the same neutral examples,
the current private-on model is pulled toward a frozen copy of the coherent86 private-on
parent rather than toward the private-off carrier.

The resulting arm is paired with research ordinary continuation: same endpoint, suffix
rows, legal word accounting, trainable private-adapter path, WWM stream, schedule,
optimizer, and random seed.  A better result would support cumulative consolidation of an
already useful correction; a flat result would mean parent anchoring preserves but does not
turn the remaining suffix into a stronger transfer model.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import coherent86_continuation_trainer as prev

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')

TEACHER_MODEL = None
TEACHER_META: dict[str, Any] = {}


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def compute_parent_anchor_kl(model, masked_inputs, attention_mask, n_sub: int):
    """KL(frozen coherent86 private-on parent || current private-on) on neutral tokens."""
    global TEACHER_MODEL
    if TEACHER_MODEL is None:
        raise RuntimeError("TEACHER_MODEL was not initialized")
    n = min(int(n_sub), masked_inputs.shape[0])
    if n <= 0:
        return None, 0.0
    was_training = bool(model.training)
    flags = prev.private_flags(model)
    ids = masked_inputs[:n]
    att = attention_mask[:n]
    target_model = TEACHER_MODEL
    try:
        model.eval()
        model.set_private_enabled(True)
        target_model.eval()
        target_model.set_private_enabled(True)
        with torch.no_grad():
            target_logits = target_model(input_ids=ids, attention_mask=att).logits.detach()
        out = model(input_ids=ids, attention_mask=att)
        log_p = F.log_softmax(out.logits, dim=-1)
        p_target = F.softmax(target_logits, dim=-1)
        kl = F.kl_div(log_p, p_target, reduction="none").sum(-1)
        mask_f = att.float()
        loss = (kl * mask_f).sum() / max(1.0, float(mask_f.sum()))
        val = float(loss.detach().cpu())
        del target_logits, out, log_p, p_target, kl, mask_f
    finally:
        prev.restore_private_flags(model, flags)
        if was_training:
            model.train()
        else:
            model.eval()
    return loss, val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", default=str(_public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023')))
    p.add_argument("--endpoint", default=str(prev.DEFAULT_PARENT))
    p.add_argument("--example_jsonl", default=str(prev.DEFAULT_STREAM))
    p.add_argument("--initial_consumed_words", type=int, default=prev.DEFAULT_INITIAL_WORDS)
    p.add_argument("--full_cap_words", type=int, default=prev.DEFAULT_FULL_CAP_WORDS)
    p.add_argument("--max_tail_charged_words", type=int, default=prev.DEFAULT_REMAINING_WORDS)
    p.add_argument("--skip_rows", type=int, default=prev.DEFAULT_SKIP_ROWS)
    p.add_argument("--macro_batch", type=int, default=256)
    p.add_argument("--micro_batch", type=int, default=32)
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--warmup_fraction", type=float, default=0.06)
    p.add_argument("--schedule_total", type=int, default=prev.DEFAULT_SCHEDULE_TOTAL)
    p.add_argument("--schedule_offset", type=int, default=prev.DEFAULT_SCHEDULE_OFFSET)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--main_lambda", type=float, default=1.0)
    p.add_argument("--neutral_lambda", type=float, default=1.0)
    p.add_argument("--neutral_subsample", type=int, default=32)
    p.add_argument("--private_adapter_bottleneck", type=int, default=128)
    p.add_argument("--private_scale", type=float, default=0.75)
    p.add_argument("--train_seed", type=int, default=43023)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--max_updates", type=int, default=0)
    # The reused research trainer expects this field; the parent-anchor script always uses
    # ordinary unweighted MLM for the main objective.
    p.set_defaults(credit_mode="standard")
    return p.parse_args()


def annotate_outputs(out_dir: Path, metrics: dict[str, Any] | None = None) -> None:
    cfg_path = out_dir / "train_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        cfg.update({
            "status": "PARENT_ANCHOR_CONTINUATION_CONFIG",
            "scientific_question": "Does anchoring neutral examples to the frozen coherent86 private-on parent let legal suffix learning add useful competence while preserving the inherited correction?",
            "credit_mode": "standard_parent_anchor",
            "objective": "main_mlm_ce_plus_frozen_parent_private_on_to_current_private_on_kl",
            "neutral_anchor": {
                "type": "frozen_parent_private_on",
                "endpoint": cfg.get("endpoint"),
                "private_scale": cfg.get("private_scale_train_and_saved"),
                "kl_direction": "KL(parent_private_on || current_private_on)",
                "same_neutral_examples_as_step027": True,
            },
            "paired_reference": {
                "carrier_anchor_run": "experiments/archive/functional_learning/training/runs/coherent86_continue_standard_seed43023",
                "changed_factor": "neutral_KL_target_only",
            },
        })
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    met_path = out_dir / "scientific_metrics.json"
    if met_path.exists():
        met = json.loads(met_path.read_text())
        met.update({
            "status": "PARENT_ANCHOR_CONTINUATION_COMPLETE",
            "base_trainer_status": met.get("status"),
            "credit_mode": "standard_parent_anchor",
            "neutral_anchor": "frozen_parent_private_on",
            "paired_carrier_anchor_run": "experiments/archive/functional_learning/training/runs/coherent86_continue_standard_seed43023",
        })
        met_path.write_text(json.dumps(met, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if metrics is not None:
        (out_dir / "parent_anchor_returned_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    global TEACHER_MODEL, TEACHER_META
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = out_dir / "hf_cache"
    (hf_cache / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_MODULES_CACHE"] = str(hf_cache / "modules")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    TEACHER_MODEL, missing, unexpected = prev.load_model(Path(args.endpoint), device, int(args.private_adapter_bottleneck), float(args.private_scale))
    TEACHER_MODEL.eval()
    TEACHER_MODEL.set_private_enabled(True)
    for p in TEACHER_MODEL.parameters():
        p.requires_grad_(False)
    TEACHER_META = {
        "status": "PARENT_ANCHOR_TEACHER_READY",
        "teacher_endpoint": rel(args.endpoint),
        "device": str(device),
        "private_scale": float(args.private_scale),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "neutral_anchor": "frozen_parent_private_on",
        "kl_direction": "KL(parent_private_on || current_private_on)",
    }
    (out_dir / "parent_anchor_manifest.json").write_text(json.dumps(TEACHER_META, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(TEACHER_META), flush=True)

    prev.compute_neutral_kl = compute_parent_anchor_kl
    metrics = prev.train(args)
    annotate_outputs(out_dir, metrics)
    print(json.dumps({"status": "PARENT_ANCHOR_WRAPPER_DONE", "out_dir": rel(out_dir)}), flush=True)


if __name__ == "__main__":
    main()
