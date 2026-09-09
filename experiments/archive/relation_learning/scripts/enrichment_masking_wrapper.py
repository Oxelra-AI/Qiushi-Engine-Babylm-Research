#!/usr/bin/env python3
"""research enrichment-masking calibration arm.

Adapter-faithful DeBERTa-v2 trainer with enrichment-weighted WWM masking.
Model construction is identical to adapter_scaled_trainer.py (adapter
scale 1.75, bottleneck 128). The ONLY difference from the control arm is the
masking distribution: per-word-group masking probabilities are weighted by
CHILDES-minus-whole enrichment z-scores, tapering linearly toward uniform
masking by taper_frac of the full training horizon (lr_total_steps).

Normalization is over corpus token occurrence distribution so expected mask
rate equals base_prob (0.15) by construction. Per-batch realized rates are
logged for verification.

LEGALITY: enrichment z-scores are derived solely from Strict-Small corpus
frequency statistics. No child AoA labels, CDI ages, or external child data
enter the training recipe. z_t = (log_freq_CHILDES_t - log_freq_whole_t) is
a corpus-internal statistic measuring relative CHILDES over-representation.
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
import time
from pathlib import Path

import torch
import numpy as np

HERE = _public_path('experiments/archive/relation_learning/scripts')
USER_ROOT = _public_path('.')
BASE_TRAINER = _public_path('experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py')
MODELING_SRC = _public_path('experiments/archive/frontier_consolidation/scripts/adapter_scaled_modeling.py')

sys.path.insert(0, str(_public_path('experiments/archive/frontier_consolidation/scripts')))
from adapter_scaled_modeling import AdapterDebertaV2ForMaskedLM


# ─── Enrichment masking state ────────────────────────────────────────────────
class EnrichmentMaskState:
    """Per-token masking probabilities from enrichment z-scores with taper."""
    def __init__(self, z_scores: torch.Tensor, corpus_freq: torch.Tensor,
                 alpha_start: float, taper_steps: int,
                 base_prob: float = 0.15, clamp_lo: float = 0.02, clamp_hi: float = 0.60):
        self.z = z_scores          # [vocab_size] float32
        self.freq = corpus_freq     # [vocab_size] float32, corpus occurrence counts
        self.freq_sum = corpus_freq.sum()
        self.alpha_start = alpha_start
        self.taper_steps = taper_steps
        self.base_prob = base_prob
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi
        self.step = 0
        # Pre-clamped probs for the initial alpha
        self._last_alpha = None
        self._last_probs = None

    def get_probs(self) -> tuple[torch.Tensor, float, float, float]:
        """Return (probs_vocab, alpha, Z, pre_clamp_mean)."""
        self.step += 1
        if self.step <= self.taper_steps:
            alpha = self.alpha_start * (1.0 - self.step / self.taper_steps)
        else:
            alpha = 0.0

        # Cache: recompute only when alpha changes meaningfully
        if self._last_alpha is not None and abs(alpha - self._last_alpha) < 1e-7:
            return self._last_probs, alpha, 0.0, 0.0

        log_w = alpha * self.z
        exp_w = torch.exp(log_w)
        # Corpus-occurrence-weighted normalization
        Z = (self.freq * exp_w).sum() / self.freq_sum
        probs = self.base_prob * exp_w / Z

        # Pre-clamp mean (should be ~base_prob)
        pre_clamp_mean = float((self.freq * probs).sum() / self.freq_sum)

        probs = probs.clamp(self.clamp_lo, self.clamp_hi)
        self._last_alpha = alpha
        self._last_probs = probs
        return probs, alpha, float(Z), pre_clamp_mean


# Global state — set during main(), read during masking hook
_ESTATE: EnrichmentMaskState | None = None


def _load_base():
    spec = importlib.util.spec_from_file_location("base", BASE_TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base trainer: {BASE_TRAINER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    # ─── Parse enrichment + adapter args before base trainer sees argv ────────
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--adapter_bottleneck", type=int, default=128)
    ap.add_argument("--adapter_activation", default="gelu")
    ap.add_argument("--adapter_enabled", type=int, choices=[0, 1], default=1)
    ap.add_argument("--adapter_scale", type=float, default=1.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--enrichment_weights", required=True,
                    help="JSON with z_enrichment array, indexed by token ID")
    ap.add_argument("--enrichment_corpus_freq", required=True,
                    help="JSON with token_freq array, indexed by token ID")
    ap.add_argument("--enrichment_alpha_start", type=float, default=1.5)
    ap.add_argument("--enrichment_taper_frac", type=float, default=0.67)
    extra, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    os.environ["CUDA_VISIBLE_DEVICES"] = str(extra.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ─── Load enrichment data ─────────────────────────────────────────────────
    ew = json.loads(Path(extra.enrichment_weights).read_text())
    z_scores = torch.tensor(ew["z_enrichment"], dtype=torch.float32)

    cf = json.loads(Path(extra.enrichment_corpus_freq).read_text())
    corpus_freq = torch.tensor(cf["token_freq"], dtype=torch.float32)

    print(json.dumps({
        "event": "enrichment_loaded",
        "vocab_size": len(z_scores),
        "z_active": int((z_scores != 0).sum()),
        "z_mean": round(float(z_scores[z_scores != 0].mean()), 4),
        "z_std": round(float(z_scores[z_scores != 0].std()), 4),
        "freq_total": int(corpus_freq.sum()),
        "freq_active": int((corpus_freq > 0).sum()),
        "alpha_start": extra.enrichment_alpha_start,
        "taper_frac": extra.enrichment_taper_frac,
    }), flush=True)

    # ─── Load base trainer ────────────────────────────────────────────────────
    base = _load_base()
    original_adamw = torch.optim.AdamW
    original_save = base.save_hf_checkpoint
    original_masking = base.apply_masking_curriculum
    model_ref: dict = {}

    # ─── Adapter model build (identical to research wrapper) ───────────────────
    def adapter_build_model(args, tokenizer):
        max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
        pos_att_type = [x.strip() for x in args.deberta_pos_att_type.split(",") if x.strip()]
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
        cfg.adapter_bottleneck = extra.adapter_bottleneck
        cfg.adapter_activation = extra.adapter_activation
        cfg.adapter_enabled = bool(extra.adapter_enabled)
        cfg.adapter_scale = float(extra.adapter_scale)
        model = AdapterDebertaV2ForMaskedLM(cfg)
        model.register_for_auto_class("AutoModelForMaskedLM")
        model_ref["m"] = model
        adapter_n = sum(p.numel() for n, p in model.named_parameters() if ".adapter." in n)
        stock_n = sum(p.numel() for n, p in model.named_parameters() if ".adapter." not in n)
        model.gradient_checkpointing_enable()

        # ─── Initialize enrichment masking state ──────────────────────────────
        global _ESTATE
        lr_total = args.lr_total_steps if args.lr_total_steps > 0 else 2529
        taper_steps = int(extra.enrichment_taper_frac * lr_total)
        _ESTATE = EnrichmentMaskState(
            z_scores=z_scores,
            corpus_freq=corpus_freq,
            alpha_start=extra.enrichment_alpha_start,
            taper_steps=taper_steps,
        )

        # Verify initial enrichment rates
        probs0, alpha0, Z0, pcm0 = _ESTATE.get_probs()
        _ESTATE.step = 0  # reset after verification probe
        _ESTATE._last_alpha = None

        print(json.dumps({
            "event": "adapter_scaled_model_built",
            "adapter_bottleneck": cfg.adapter_bottleneck,
            "adapter_enabled": cfg.adapter_enabled,
            "adapter_scale": cfg.adapter_scale,
            "gradient_checkpointing": True,
            "stock_params": stock_n,
            "adapter_params": adapter_n,
            "total_params": stock_n + adapter_n,
            "enrichment_alpha_start": extra.enrichment_alpha_start,
            "enrichment_taper_frac": extra.enrichment_taper_frac,
            "enrichment_taper_steps": taper_steps,
            "enrichment_initial_Z": round(Z0, 6),
            "enrichment_initial_pre_clamp_mean": round(pcm0, 6),
            "enrichment_prob_p05": round(float(probs0.quantile(0.05)), 6),
            "enrichment_prob_p50": round(float(probs0.quantile(0.50)), 6),
            "enrichment_prob_p95": round(float(probs0.quantile(0.95)), 6),
        }), flush=True)
        return model

    # ─── Enrichment masking hook ──────────────────────────────────────────────
    def enrichment_masking(input_ids, attention_mask, word_group, tokenizer, state, gen):
        """Replace base apply_masking_curriculum with enrichment-weighted WWM."""
        if _ESTATE is None:
            return original_masking(input_ids, attention_mask, word_group, tokenizer, state, gen)

        probs, alpha, Z, pcm = _ESTATE.get_probs()

        device = input_ids.device
        per_tok = probs[input_ids.cpu()].to(device)

        bsz, seq = input_ids.shape
        labels = input_ids.clone()
        mask_id = tokenizer.mask_token_id
        special = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
        cand = attention_mask.bool() & ~torch.isin(input_ids, special)
        sel = torch.zeros_like(cand)

        # WWM: select whole word groups by mean per-token probability
        for b in range(bsz):
            g = word_group[b]
            vg = torch.unique(g[g >= 0])
            if vg.numel() == 0:
                continue
            gp_val = torch.zeros(vg.max().item() + 1, device=device)
            for gid in vg:
                m = (g == gid) & cand[b]
                if m.any():
                    gp_val[gid] = per_tok[b][m].mean()
            r = torch.rand(vg.numel(), generator=gen, device=device)
            ch = vg[r < gp_val[vg]]
            if ch.numel() > 0:
                sel[b] = torch.isin(g, ch) & cand[b]

        # Fallback: at least one token
        if sel.sum() == 0:
            flat = cand.view(-1).nonzero(as_tuple=False)
            if flat.numel() > 0:
                sel.view(-1)[flat[0, 0]] = True

        labels[~sel] = -100

        # 80/10/10 corruption (identical to base)
        mi = input_ids.clone()
        r = torch.rand(bsz, seq, generator=gen, device=device)
        mi[sel & (r < 0.8)] = mask_id
        rand_mask = sel & (r >= 0.8) & (r < 0.9)
        if rand_mask.any():
            mi[rand_mask] = torch.randint(
                0, len(tokenizer), (int(rand_mask.sum()),),
                generator=gen, device=device)

        eff = float(sel.sum()) / max(float(cand.sum()), 1.0)

        # Log every 10 steps + first 3
        if _ESTATE.step % 10 == 1 or _ESTATE.step <= 3:
            print(json.dumps({
                "event": "enrichment_mask",
                "step": _ESTATE.step,
                "alpha": round(alpha, 4),
                "eff_rate": round(eff, 6),
                "Z": round(Z, 6),
                "pre_clamp_mean": round(pcm, 6),
            }), flush=True)

        return mi, labels

    # ─── Optimizer groups (identical to research) ──────────────────────────────
    def grouped_adamw(params, lr, weight_decay, betas, **kw):
        model = model_ref["m"]
        zero_ids = {id(p) for n, p in model.named_parameters() if ".adapter.up." in n}
        normal, zero_wd = [], []
        for p in params:
            (zero_wd if id(p) in zero_ids else normal).append(p)
        print(json.dumps({
            "event": "adapter_scaled_optimizer_groups",
            "normal_decay_params": sum(p.numel() for p in normal),
            "zero_decay_params": sum(p.numel() for p in zero_wd),
        }), flush=True)
        return original_adamw([
            {"params": normal, "weight_decay": weight_decay},
            {"params": zero_wd, "weight_decay": 0.0},
        ], lr=lr, betas=betas, **kw)

    def save_with_src(model, tokenizer, dst: Path) -> None:
        original_save(model, tokenizer, dst)
        target = Path(dst) / MODELING_SRC.name
        if not target.exists():
            shutil.copy2(MODELING_SRC, target)

    # ─── Apply patches and run ────────────────────────────────────────────────
    base.build_model = adapter_build_model
    base.apply_masking_curriculum = enrichment_masking
    torch.optim.AdamW = grouped_adamw
    base.save_hf_checkpoint = save_with_src
    try:
        base.main()
    finally:
        torch.optim.AdamW = original_adamw
        base.save_hf_checkpoint = original_save
        base.apply_masking_curriculum = original_masking


if __name__ == "__main__":
    main()
