#!/usr/bin/env python3
"""research function-preserving residual-adapter trainer.

Thin wrapper around the exact research masking-curriculum trainer.  Changes only:
1. Model construction: AdapterDebertaV2ForMaskedLM instead of DebertaV2ForMaskedLM
2. Optimizer grouping: adapter zero-output tensors (W_up, b_up) get weight_decay=0
3. Checkpoint saving: ensures adapter_modeling.py is beside config.json

Corpus, tokenizer, ordering, masking, all stock DeBERTa tensors, MLM head,
AdamW betas/LR, cosine schedule, gradient clipping, RNG seeds, and checkpoint
cadence are inherited byte-for-byte from the base trainer.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import torch

HERE = _public_path('experiments/archive/frontier_consolidation/scripts')
USER_ROOT = _public_path('.')
BASE_TRAINER = _public_path('experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py')
MODELING_SRC = _public_path('experiments/archive/frontier_consolidation/scripts/adapter_modeling.py')

sys.path.insert(0, str(HERE))
from adapter_modeling import AdapterDebertaV2ForMaskedLM


# ═══════════════════════════════════════════════════════════════════════
# Dynamic import of the base trainer as a module
# ═══════════════════════════════════════════════════════════════════════
def _load_base():
    spec = importlib.util.spec_from_file_location("base", BASE_TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base trainer: {BASE_TRAINER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Consume adapter-specific args before the base trainer sees sys.argv ──
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--adapter_bottleneck", type=int, default=64)
    ap.add_argument("--adapter_activation", default="gelu")
    ap.add_argument("--adapter_enabled", type=int, choices=[0, 1], default=1)
    ap.add_argument("--gpu", type=int, default=0)
    extra, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    os.environ["CUDA_VISIBLE_DEVICES"] = str(extra.gpu)

    base = _load_base()
    original_adamw = torch.optim.AdamW
    original_save  = base.save_hf_checkpoint
    model_ref: dict = {}

    # ── 1. Replace build_model ────────────────────────────────────────
    def adapter_build_model(args, tokenizer):
        """Construct AdapterDebertaV2ForMaskedLM under the exact RNG state
        that the base trainer provides (seed / extra_init_seed already set)."""
        max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
        pos_att_type = [x.strip()
                        for x in args.deberta_pos_att_type.split(",") if x.strip()]
        cfg = base.DebertaV2Config(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.n_layer,
            num_attention_heads=args.n_head,
            intermediate_size=args.hidden_size * args.ffn_mult,
            max_position_embeddings=max_pos,
            max_relative_positions=args.max_relative_positions,
            position_buckets=args.position_buckets,
            relative_attention=True,
            pos_att_type=pos_att_type,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        cfg.adapter_bottleneck  = extra.adapter_bottleneck
        cfg.adapter_activation  = extra.adapter_activation
        cfg.adapter_enabled     = bool(extra.adapter_enabled)
        model = AdapterDebertaV2ForMaskedLM(cfg)
        model.register_for_auto_class("AutoModelForMaskedLM")
        model_ref["m"] = model

        adapter_n = sum(p.numel() for n, p in model.named_parameters()
                        if ".adapter." in n)
        stock_n   = sum(p.numel() for n, p in model.named_parameters()
                        if ".adapter." not in n)
        # Gradient checkpointing: halves activation memory so we fit
        # alongside other GPU processes while keeping batch_size=256.
        model.gradient_checkpointing_enable()

        print(json.dumps({
            "event": "adapter_model_built",
            "adapter_bottleneck": cfg.adapter_bottleneck,
            "adapter_enabled": cfg.adapter_enabled,
            "gradient_checkpointing": True,
            "stock_params": stock_n,
            "adapter_params": adapter_n,
            "total_params": stock_n + adapter_n,
        }), flush=True)
        return model

    # ── 2. Replace AdamW to separate adapter-output weight decay ──────
    def grouped_adamw(params, lr, weight_decay, betas, **kw):
        model = model_ref["m"]
        zero_ids = {id(p) for n, p in model.named_parameters()
                    if ".adapter.up." in n}
        normal, zero_wd = [], []
        for p in params:
            (zero_wd if id(p) in zero_ids else normal).append(p)
        groups = [
            {"params": normal,  "weight_decay": weight_decay},
            {"params": zero_wd, "weight_decay": 0.0},
        ]
        print(json.dumps({
            "event": "adapter_optimizer_groups",
            "normal_decay_params": sum(p.numel() for p in normal),
            "zero_decay_params":   sum(p.numel() for p in zero_wd),
        }), flush=True)
        return original_adamw(groups, lr=lr, betas=betas, **kw)

    # ── 3. Patch checkpoint saving to include modeling source ─────────
    def save_with_src(model, tokenizer, dst: Path) -> None:
        original_save(model, tokenizer, dst)
        target = Path(dst) / MODELING_SRC.name
        if not target.exists():
            shutil.copy2(MODELING_SRC, target)

    # ── Apply patches and run ─────────────────────────────────────────
    base.build_model          = adapter_build_model
    torch.optim.AdamW         = grouped_adamw
    base.save_hf_checkpoint   = save_with_src

    try:
        base.main()
    finally:
        torch.optim.AdamW       = original_adamw
        base.save_hf_checkpoint = original_save


if __name__ == "__main__":
    main()
