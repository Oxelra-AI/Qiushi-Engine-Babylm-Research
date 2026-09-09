#!/usr/bin/env python3
"""research smoke test for MLM pairwise attention masking in DeBERTa-v2.

The scientific use is narrow: the proposed compact-view causal-separation
study needs to preserve each view's internal context while blocking source<->view
attention. Hugging Face DebertaV2ForMaskedLM cannot receive a square attention
mask directly through its public forward because embeddings also consume the same
mask. This smoke test verifies a custom forward path that sends the ordinary 2D
padding mask to embeddings and a 3D pairwise reachability mask to the encoder.

Checks:
1. A fully visible pairwise mask is numerically equivalent to the ordinary public
   DebertaV2ForMaskedLM forward in eval mode.
2. A boundary-blocked pairwise mask changes logits relative to fully visible.
3. A small backward pass through the custom path has finite gradients.
"""
from __future__ import annotations

import json
import math
import pathlib

import torch
import torch.nn.functional as F
from transformers import DebertaV2Config, DebertaV2ForMaskedLM


def custom_pairmask_forward(model, input_ids, pad_mask_2d, pair_mask_3d, labels=None):
    """Forward DeBERTa MLM with 2D embedding mask and 3D encoder mask."""
    # Match DebertaV2Model.forward enough for the current non-legacy MLM head.
    emb = model.deberta.embeddings(
        input_ids=input_ids,
        token_type_ids=None,
        position_ids=None,
        mask=pad_mask_2d,
        inputs_embeds=None,
    )
    enc = model.deberta.encoder(
        emb,
        pair_mask_3d,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
    seq = enc[0]
    if getattr(model, "legacy", False):
        logits = model.cls(seq)
    else:
        logits = model.lm_predictions(seq, model.deberta.embeddings.word_embeddings)
    loss = None
    if labels is not None:
        loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), labels.view(-1), ignore_index=-100)
    return logits, loss


def main():
    out_dir = pathlib.Path("experiments/archive/representation_and_objectives/data/deberta_pair_attention_mask_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(219)
    cfg = DebertaV2Config(
        vocab_size=257,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        max_relative_positions=64,
        position_buckets=64,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        pad_token_id=0,
    )
    model = DebertaV2ForMaskedLM(cfg)
    model.eval()
    bsz, seq = 3, 18
    input_ids = torch.randint(5, cfg.vocab_size, (bsz, seq))
    # Add padding to one example to make sure padding and pair masking compose.
    input_ids[2, 15:] = 0
    pad = (input_ids != 0).long()
    full_pair = (pad[:, :, None] * pad[:, None, :]).long()

    labels = input_ids.clone()
    labels[:] = -100
    labels[0, 4] = input_ids[0, 4]
    labels[1, 11] = input_ids[1, 11]
    labels[2, 8] = input_ids[2, 8]

    with torch.no_grad():
        std = model(input_ids=input_ids, attention_mask=pad, labels=labels)
        cust_logits, cust_loss = custom_pairmask_forward(model, input_ids, pad, full_pair, labels)
    max_abs = (std.logits - cust_logits).abs().max().item()
    loss_delta = abs(std.loss.item() - cust_loss.item())

    blocked = full_pair.clone()
    # Per-row source/view boundaries; block only real-token cross-view attention.
    boundaries = [8, 10, 7]
    for b, bd in enumerate(boundaries):
        real = int(pad[b].sum().item())
        blocked[b, :bd, bd:real] = 0
        blocked[b, bd:real, :bd] = 0
    with torch.no_grad():
        block_logits, block_loss = custom_pairmask_forward(model, input_ids, pad, blocked, labels)
    blocked_delta = (cust_logits - block_logits).abs().max().item()
    masked_logit_delta = (cust_logits[labels != -100] - block_logits[labels != -100]).abs().max().item()

    # Gradient smoke in train mode on the blocked path.
    model.train()
    logits, loss = custom_pairmask_forward(model, input_ids, pad, blocked, labels)
    loss.backward()
    total_norm_sq = 0.0
    n_grad = 0
    all_finite = True
    for p in model.parameters():
        if p.grad is not None:
            n_grad += 1
            g = p.grad.detach()
            all_finite = all_finite and bool(torch.isfinite(g).all().item())
            total_norm_sq += float(g.float().pow(2).sum().item())
    grad_norm = math.sqrt(total_norm_sq)

    summary = {
        "status": "DEBERTA_PAIR_ATTENTION_MASK_SMOKE",
        "purpose": "Verify that MLM cross-view boundary blocking can be implemented by giving embeddings a 2D padding mask and the encoder a 3D pairwise mask.",
        "standard_vs_custom_full_visible": {
            "max_abs_logit_delta": max_abs,
            "loss_delta": loss_delta,
            "pass": max_abs < 1e-6 and loss_delta < 1e-7,
        },
        "blocked_changes_logits": {
            "max_abs_logit_delta_full_vs_blocked": blocked_delta,
            "masked_positions_max_abs_delta": masked_logit_delta,
            "pass": blocked_delta > 1e-7 and masked_logit_delta > 1e-7,
        },
        "gradient_smoke": {
            "loss": float(loss.detach().cpu()),
            "n_tensors_with_grad": n_grad,
            "grad_norm": grad_norm,
            "all_finite": all_finite,
            "pass": n_grad > 0 and all_finite and math.isfinite(grad_norm) and grad_norm > 0,
        },
        "implementation_note": "Do not pass a 3D/4D mask into DebertaV2ForMaskedLM.forward; wrap the MLM head or trainer so embeddings receive the ordinary 2D padding mask and encoder receives the pairwise reachability mask.",
    }
    summary["pass"] = all(x["pass"] for x in [summary["standard_vs_custom_full_visible"], summary["blocked_changes_logits"], summary["gradient_smoke"]])
    out_path = out_dir / "deberta_pair_attention_mask_smoke.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = out_dir / "deberta_pair_attention_mask_smoke.md"
    md.write_text(
        "# research DeBERTa pair attention mask smoke\n\n"
        f"Pass: {summary['pass']}\n\n"
        f"Full-visible custom vs public forward max logit delta: {max_abs:.3e}; loss delta: {loss_delta:.3e}.\n\n"
        f"Blocked vs visible max logit delta: {blocked_delta:.3e}; masked-position delta: {masked_logit_delta:.3e}.\n\n"
        f"Gradient finite: {all_finite}; grad norm: {grad_norm:.6g}.\n\n"
        "A pair-aware MLM trainer requires a custom forward path; sending a square mask to the stock MLM forward does not implement the intended comparison.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "pass": summary["pass"], "json": str(out_path), "md": str(md), "max_abs": max_abs, "blocked_delta": blocked_delta}, indent=2))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
