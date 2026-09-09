#!/usr/bin/env python3
"""Support-Gated Compositional Residual (SGCR) module for DeBERTa-v2.

Keeps exact legal40k tokenization, WWM labels, data order, and stream.
Adds a jointly-trained low-dimensional legal16k component table and a
static support-dependent gate so rare tokens borrow representation
strength from well-supported components.

Zero-sharing limit: when K=0, rho=1 for all tokens, model is exactly
the standard legal40k DeBERTa-v2.

Parameter-matched nonsharing control: same architecture but rho=mean(rho)
for all tokens (uniform gate), so any benefit is from support routing.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGCRConfig:
    """Configuration for the SGCR embedding layer."""
    def __init__(
        self,
        vocab_40k: int = 40000,
        vocab_16k: int = 16384,
        hidden_size: int = 384,
        d_comp: int = 64,
        K: float = 50.0,
          uniform_gate: bool = False,  # True = parameter-matched control
          init_mode: str = "cold",     # "cold" = exact-preserving gradient-live residual
          force_standard_ids: Optional[list[int]] = None,
      ):
          self.vocab_40k = vocab_40k
          self.vocab_16k = vocab_16k
          self.hidden_size = hidden_size
          self.d_comp = d_comp
          self.K = K
          self.uniform_gate = uniform_gate
          self.init_mode = init_mode
          self.force_standard_ids = set(int(x) for x in (force_standard_ids or []))


def _tokenizer_json_path(path: str | Path) -> Path:
    """Return a tokenizer.json path for a tokenizer directory or JSON file."""
    p = Path(path)
    return p if p.name == "tokenizer.json" else p / "tokenizer.json"


def _normalise_bpe_merge(merge: Any) -> tuple[str, str]:
    """Normalise tokenizers BPE merge records across JSON encodings."""
    if isinstance(merge, str):
        parts = merge.split()
        if len(parts) != 2:
            raise ValueError(f"cannot parse BPE merge record {merge!r}")
        return parts[0], parts[1]
    if isinstance(merge, (list, tuple)) and len(merge) == 2:
        return str(merge[0]), str(merge[1])
    raise ValueError(f"cannot parse BPE merge record {merge!r}")


def build_decomposition_map(
    tok40k_path: str | Path,
    tok16k_path: str | Path,
) -> dict[int, list[int]]:
    """Exactly split legal40k BPE tokens at the legal16k merge frontier.

    The legal16k tokenizer is an id-preserving vocabulary prefix of the legal40k
    tokenizer, and the legal16k merge list is an exact prefix of the legal40k
    merge list. Recursively undoing the post-16k legal40k merges gives the
    intended legal16k component ancestry for every legal40k token. This avoids
    the raw-tokenizer padding/truncation and isolated byte-level decode/re-encode
    defects found by the pre-launch audit.
    """
    with _tokenizer_json_path(tok16k_path).open(encoding="utf-8") as f:
        raw16 = json.load(f)
    with _tokenizer_json_path(tok40k_path).open(encoding="utf-8") as f:
        raw40 = json.load(f)

    if raw16.get("model", {}).get("type") != "BPE" or raw40.get("model", {}).get("type") != "BPE":
        raise ValueError("SGCR prefix decomposition requires BPE tokenizers")

    vocab16: dict[str, int] = raw16["model"]["vocab"]
    vocab40: dict[str, int] = raw40["model"]["vocab"]
    merges16 = [_normalise_bpe_merge(m) for m in raw16["model"].get("merges", [])]
    merges40 = [_normalise_bpe_merge(m) for m in raw40["model"].get("merges", [])]

    if len(vocab16) > len(vocab40):
        raise ValueError("legal16k vocabulary is larger than legal40k vocabulary")
    if merges16 != merges40[: len(merges16)]:
        raise ValueError("legal16k merges are not an exact prefix of legal40k merges")
    bad_vocab_prefix = [piece for piece, tid in vocab16.items() if vocab40.get(piece) != tid]
    if bad_vocab_prefix:
        raise ValueError(f"legal16k vocabulary ids are not an exact legal40k prefix; examples={bad_vocab_prefix[:5]}")

    cutoff = len(vocab16)
    id_to_piece = {tid: piece for piece, tid in vocab40.items()}
    missing_ids = [tid for tid in range(len(vocab40)) if tid not in id_to_piece]
    if missing_ids:
        raise ValueError(f"legal40k vocabulary ids are not contiguous; examples={missing_ids[:5]}")

    reverse_merge: dict[str, tuple[str, str]] = {}
    for left, right in merges40:
        reverse_merge[left + right] = (left, right)

    memo: dict[str, tuple[int, ...]] = {}

    def split(piece: str) -> tuple[int, ...]:
        if piece in memo:
            return memo[piece]
        token_id = vocab40[piece]
        if token_id < cutoff:
            result = (token_id,)
        else:
            if piece not in reverse_merge:
                raise ValueError(f"no merge ancestry for legal40k token {piece!r} id={token_id}")
            left, right = reverse_merge[piece]
            result = split(left) + split(right)
        memo[piece] = result
        return result

    cmap: dict[int, list[int]] = {}
    for tid40 in range(len(vocab40)):
        comps = list(split(id_to_piece[tid40]))
        if not comps:
            raise ValueError(f"empty SGCR decomposition for legal40k token {tid40}")
        if any(c < 0 or c >= cutoff for c in comps):
            raise ValueError(f"component out of legal16k range for legal40k token {tid40}: {comps}")
        cmap[tid40] = comps
    return cmap


def build_pool_counts(
    tok_path: str | Path,
    pool_jsonl: str | Path,
) -> dict[int, int]:
    """Count each token id occurrence in the exact 10M training pool.

    Raw tokenizer.json files can carry persistent padding/truncation settings.
    Disable both and, as a second guard, ignore positions whose attention mask is
    zero. Counts are used by SGCR gates, so padded ids must never contribute.
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(_tokenizer_json_path(tok_path)))
    if hasattr(tok, "no_padding"):
        tok.no_padding()
    if hasattr(tok, "no_truncation"):
        tok.no_truncation()

    counts: dict[int, int] = collections.defaultdict(int)

    def add_encoding(enc) -> None:
        mask = getattr(enc, "attention_mask", None)
        if mask is None:
            for tid in enc.ids:
                counts[int(tid)] += 1
        else:
            for tid, keep in zip(enc.ids, mask):
                if keep:
                    counts[int(tid)] += 1

    with open(pool_jsonl, encoding="utf-8") as f:
        batch = []
        for line in f:
            row = json.loads(line)
            batch.append(row["text"])
            if len(batch) >= 512:
                for enc in tok.encode_batch(batch, add_special_tokens=False):
                    add_encoding(enc)
                batch = []
        if batch:
            for enc in tok.encode_batch(batch, add_special_tokens=False):
                add_encoding(enc)
    return dict(counts)


def build_sgcr_buffers(
    decomposition_map: dict[int, list[int]],
    counts_40k: dict[int, int],
    config: SGCRConfig,
) -> dict[str, torch.Tensor]:
    """Build static buffers for SGCR.

    Returns:
        decomp_ids: (vocab_40k, max_comp_len) padded legal16k component ids
        decomp_lengths: (vocab_40k,) valid component count per token
        rho: (vocab_40k,) support gate values
    """
    max_comp_len = max(
        (len(comps) for comps in decomposition_map.values() if comps),
        default=1
    )

    decomp_ids = torch.zeros(config.vocab_40k, max_comp_len, dtype=torch.long)
    decomp_lengths = torch.zeros(config.vocab_40k, dtype=torch.long)

    for t in range(config.vocab_40k):
        comps = decomposition_map.get(t, [])
        if comps:
            decomp_ids[t, :len(comps)] = torch.tensor(comps, dtype=torch.long)
            decomp_lengths[t] = len(comps)
        else:
            # Tokens with no decomposition must stay on the standard path.
            # Keeping length 0 lets rho logic force rho=1 and prevents accidental
            # use of component id 0 after the component table starts learning.
            decomp_lengths[t] = 0

    # Compute rho values. Force special/control tokens (pad/mask/bos/eos/etc.)
    # to remain standard: they are not lexical evidence-bearing rare tokens, and
    # the [MASK] token receives massive input exposure through MLM corruption even
    # though it has zero corpus count. Letting it use rho≈0 would confound SGCR.
    rho = torch.zeros(config.vocab_40k)
    for t in range(config.vocab_40k):
        if t in config.force_standard_ids or decomp_lengths[t].item() <= 0:
            rho[t] = 1.0
            continue
        n_t = counts_40k.get(t, 0)
        rho[t] = n_t / (n_t + config.K) if config.K > 0 else 1.0

    if config.uniform_gate:
        # Parameter-matched control: match the routed treatment's training-token
        # mass of residual exposure, not the type-mean rho. The type mean makes
        # the residual path about 8.5x too strong for K=50 on this corpus.
        force_mask = torch.tensor([(t in config.force_standard_ids) or (decomp_lengths[t].item() <= 0) for t in range(config.vocab_40k)])
        counts_tensor = torch.tensor([counts_40k.get(t, 0) for t in range(config.vocab_40k)], dtype=torch.float32)
        used_mask = (counts_tensor > 0) & ~force_mask
        if used_mask.any():
            residual_mass = (counts_tensor[used_mask] * (1.0 - rho[used_mask])).sum() / counts_tensor[used_mask].sum()
            mean_rho = float(1.0 - residual_mass.item())
        else:
            mean_rho = 0.5
        rho[~force_mask] = mean_rho
        rho[force_mask] = 1.0

    return {
        "decomp_ids": decomp_ids,
        "decomp_lengths": decomp_lengths,
        "rho": rho,
        "max_comp_len": torch.tensor(max_comp_len),
    }


class SGCREmbedding(nn.Module):
    """Support-Gated Compositional Residual embedding.

    Replaces the standard nn.Embedding for word_embeddings and provides
    an effective embedding table that is also used by the tied decoder.

    Forward: input_ids → embedded vectors using effective embeddings.
    effective_embedding_table(): returns (vocab_40k, hidden_size) tensor.
    """

    def __init__(
        self,
        word_embeddings: nn.Embedding,
        config: SGCRConfig,
        decomp_ids: torch.Tensor,
        decomp_lengths: torch.Tensor,
        rho: torch.Tensor,
    ):
        super().__init__()
        self.word_embeddings = word_embeddings  # shared with base model
        self.config = config

        # New learnable parameters
        self.component_embeddings = nn.Embedding(config.vocab_16k, config.d_comp)
        self.component_proj = nn.Linear(config.d_comp, config.hidden_size, bias=True)

        # Static buffers
        self.register_buffer("decomp_ids", decomp_ids)
        self.register_buffer("decomp_lengths", decomp_lengths)
        self.register_buffer("rho", rho)

        # Initialize new parameters. The default "cold" mode is exact-preserving
        # but gradient-live: random component codes with a zero projection give
        # comp_correction=0 and W_eff=W_std exactly, while the projection receives
        # nonzero gradients immediately. Zeroing both factors would leave the
        # component path dead except for the projection bias.
        if config.init_mode == "cold":
            nn.init.normal_(self.component_embeddings.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.component_proj.weight)
            nn.init.zeros_(self.component_proj.bias)
        elif config.init_mode == "warm":
            # Warm start: component embeddings from SVD of pretrained word embeddings
            self._warm_init()
        else:
            raise ValueError(f"Unknown init_mode: {config.init_mode}")

    def _warm_init(self):
        """Initialize component embeddings from pretrained word embeddings.

        For each legal16k component c, find the legal40k token that has
        the most similar surface form and use a low-rank projection of
        its pretrained embedding.
        """
        # Use truncated SVD of pretrained embeddings to define the component space
        with torch.no_grad():
            W = self.word_embeddings.weight.float()  # (40k, hidden)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # Use first d_comp singular vectors as component space basis
            basis = Vh[:self.config.d_comp, :]  # (d_comp, hidden)
            # Component proj: maps d_comp → hidden via these basis vectors
            self.component_proj.weight.copy_(basis.T.to(self.component_proj.weight.dtype))
            nn.init.zeros_(self.component_proj.bias)
            # Component embeddings: project pretrained 16k-aligned embeddings
            # For now, initialize to zero (cold) - warm init needs explicit mapping
            nn.init.zeros_(self.component_embeddings.weight)

    @property
    def new_parameter_count(self) -> int:
        """Count of parameters added by SGCR."""
        return (self.component_embeddings.weight.numel() +
                self.component_proj.weight.numel() +
                self.component_proj.bias.numel())

    def effective_embedding_table(self) -> torch.Tensor:
        """Compute the full effective embedding table.

        Uses ADDITIVE RESIDUAL formulation:
            W_eff = W_std + (1 - rho) * comp_correction

        This ensures:
        - Cold init (zeros): W_eff = W_std exactly (preserves pretrained or random init)
        - After training: rare tokens get larger component corrections
        - Zero-sharing (K=0, rho=1): W_eff = W_std exactly (standard model)

        Gradient routing:
        - ∂L/∂W_std = ∂L/∂W_eff (always full gradient, like standard model)
        - ∂L/∂comp = (1-rho) * ∂L/∂W_eff (stronger for rare tokens)

        Returns:
            W_eff: (vocab_40k, hidden_size) effective embeddings
        """
        W_std = self.word_embeddings.weight  # (vocab_40k, hidden_size)

        # Look up all decomposition component embeddings
        comp_embs = self.component_embeddings(self.decomp_ids)  # (vocab_40k, max_len, d_comp)

        # Mask for valid positions
        max_len = self.decomp_ids.size(1)
        mask = torch.arange(max_len, device=self.decomp_ids.device).unsqueeze(0) < self.decomp_lengths.unsqueeze(1)

        # Masked mean over components
        comp_embs_masked = comp_embs * mask.unsqueeze(-1).float()
        lengths_safe = self.decomp_lengths.clamp(min=1).unsqueeze(1).float()
        comp_mean = comp_embs_masked.sum(1) / lengths_safe  # (vocab_40k, d_comp)

        # Project to model dimension: the component correction
        comp_correction = self.component_proj(comp_mean)  # (vocab_40k, hidden_size)

        # Additive residual with support gate
        weight = (1 - self.rho).unsqueeze(1)  # (vocab_40k, 1)
        W_eff = W_std + weight * comp_correction

        return W_eff

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up effective embeddings for input token ids.

        Args:
            input_ids: (batch, seq_len) token ids

        Returns:
            embeddings: (batch, seq_len, hidden_size)
        """
        W_eff = self.effective_embedding_table()
        return F.embedding(input_ids, W_eff)


class SGCRDecoder(nn.Module):
    """SGCR-aware output decoder that uses the effective embedding table.

    Replaces the standard tied decoder weight with the SGCR effective table.
    """

    def __init__(self, sgcr_embedding: SGCREmbedding, bias: nn.Parameter):
        super().__init__()
        self.sgcr_embedding = sgcr_embedding
        self.bias = bias  # (vocab_40k,) — the decoder bias stays independent

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute output logits using effective embeddings.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            logits: (batch, seq_len, vocab_40k)
        """
        W_eff = self.sgcr_embedding.effective_embedding_table()
        return F.linear(hidden_states, W_eff, self.bias)


def apply_sgcr_to_model(
    model,  # DebertaV2ForMaskedLM
    sgcr_config: SGCRConfig,
    decomposition_map: dict[int, list[int]],
    counts_40k: dict[int, int],
) -> tuple[Any, SGCREmbedding]:
    """Modify a DebertaV2ForMaskedLM in-place to use SGCR embeddings.

    Returns the model and the SGCREmbedding module.

    Zero-sharing guarantee: when sgcr_config.K == 0, all rho values are 1.0,
    so effective_embedding_table() returns exactly word_embeddings.weight,
    and the model is identically the standard DeBERTa-v2.
    """
    # Build static buffers
    buffers = build_sgcr_buffers(decomposition_map, counts_40k, sgcr_config)

    # Get the original word embeddings
    word_emb = model.deberta.embeddings.word_embeddings

    # Create SGCR embedding module
    sgcr_emb = SGCREmbedding(
        word_embeddings=word_emb,
        config=sgcr_config,
        decomp_ids=buffers["decomp_ids"],
        decomp_lengths=buffers["decomp_lengths"],
        rho=buffers["rho"],
    )

    # Move to same device/dtype as model
    device = word_emb.weight.device
    dtype = word_emb.weight.dtype
    sgcr_emb = sgcr_emb.to(device=device)
    # Keep component params in same dtype
    sgcr_emb.component_embeddings = sgcr_emb.component_embeddings.to(dtype=dtype)
    sgcr_emb.component_proj = sgcr_emb.component_proj.to(dtype=dtype)

    # Replace the embedding module
    # We need to intercept the forward of deberta.embeddings
    # The standard DeBERTa-v2 embeddings: word_embeddings(input_ids) + position_embeddings + LayerNorm + dropout
    # We replace word_embeddings lookup with our effective embedding lookup

    # Store SGCR embedding as a named attribute
    model._sgcr_embedding = sgcr_emb

    # Replace the decoder in cls.predictions
    original_bias = model.cls.predictions.bias  # nn.Parameter(vocab_40k,)
    # The decoder is model.cls.predictions.decoder
    # Its forward: linear(hidden, weight=word_emb.weight, bias=predictions.bias)
    # We need to replace it with our SGCR decoder
    sgcr_decoder = SGCRDecoder(sgcr_emb, original_bias)
    model._sgcr_decoder = sgcr_decoder

    return model, sgcr_emb


def sgcr_forward_embedding_hook(model, sgcr_emb: SGCREmbedding):
    """Register a forward hook to intercept embedding computation.

    This modifies the embedding lookup to use SGCR effective embeddings
    without changing the DeBERTa-v2 forward pass structure.
    """
    original_emb_forward = model.deberta.embeddings.word_embeddings.forward

    def sgcr_word_embedding_forward(input_ids):
        return sgcr_emb(input_ids)

    model.deberta.embeddings.word_embeddings.forward = sgcr_word_embedding_forward

    # Also hook the decoder
    original_decoder = model.cls.predictions.decoder

    def sgcr_decoder_forward(hidden_states):
        W_eff = sgcr_emb.effective_embedding_table()
        return F.linear(hidden_states, W_eff, model.cls.predictions.bias)

    model.cls.predictions.decoder.forward = sgcr_decoder_forward

    return model


# ── Verification utilities ──

def verify_zero_sharing_limit(model, sgcr_emb: SGCREmbedding) -> dict:
    """Verify that K=0 recovers the standard model exactly."""
    with torch.no_grad():
        W_std = sgcr_emb.word_embeddings.weight.clone()
        # Save original rho
        original_rho = sgcr_emb.rho.clone()
        # Set rho = 1 everywhere
        sgcr_emb.rho.fill_(1.0)
        W_eff = sgcr_emb.effective_embedding_table()
        max_diff = (W_eff - W_std).abs().max().item()
        mean_diff = (W_eff - W_std).abs().mean().item()
        # Restore
        sgcr_emb.rho.copy_(original_rho)
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "exact_recovery": max_diff == 0.0,
    }


def sgcr_diagnostic(sgcr_emb: SGCREmbedding) -> dict:
    """Compute diagnostic statistics for the SGCR module."""
    with torch.no_grad():
        rho = sgcr_emb.rho
        W_eff = sgcr_emb.effective_embedding_table()
        W_std = sgcr_emb.word_embeddings.weight

        # Embedding deviation from standard
        deviation = (W_eff - W_std).norm(dim=1)  # (vocab,)

        # Mask for used tokens (rho < 1)
        used = rho < 1.0

        return {
            "new_params": sgcr_emb.new_parameter_count,
            "mean_rho": rho.mean().item(),
            "min_rho": rho.min().item(),
            "tokens_below_0.5": (rho < 0.5).sum().item(),
            "mean_deviation_all": deviation.mean().item(),
            "mean_deviation_used": deviation[used].mean().item() if used.any() else 0,
            "max_deviation": deviation.max().item(),
            "component_emb_norm": sgcr_emb.component_embeddings.weight.norm().item(),
            "proj_weight_norm": sgcr_emb.component_proj.weight.norm().item(),
        }


# ── Baking: convert SGCR model to standard HF model for evaluation ──

def bake_sgcr_to_standard(model, sgcr_emb: SGCREmbedding) -> None:
    """Replace word_embeddings.weight with the effective embedding table.

    After baking, the model is a standard DeBERTa-v2 with modified embeddings
    that produce identical outputs to the SGCR model. The baked model can be
    evaluated by the standard BabyLM pipeline without SGCR hooks.

    Since word_embeddings.weight is tied to decoder.weight, this also updates
    the output decoder. The model can be saved with model.save_pretrained()
    and loaded as a standard DeBERTa-v2.

    WARNING: This is a one-way operation. The original word_embeddings are lost.
    Call this on a copy of the model or at evaluation time only.
    """
    with torch.no_grad():
        W_eff = sgcr_emb.effective_embedding_table()
        # Replace the tied word_embeddings.weight with W_eff
        # Since decoder.weight IS word_embeddings.weight (same object),
        # this updates both input and output
        model.deberta.embeddings.word_embeddings.weight.copy_(W_eff)


def save_baked_checkpoint(model, sgcr_emb: SGCREmbedding, tokenizer,
                          save_dir: Path, also_save_sgcr: bool = True) -> dict:
    """Save a clean baked standard HF checkpoint and restore the live SGCR model.

    The baked model contains the effective SGCR input/output embedding table and
    no `_sgcr*` state keys, so the BabyLM evaluator can load it as an ordinary
    DebertaV2ForMaskedLM. The sidecar SGCR state is diagnostic/resume material;
    it includes the pre-bake base word table to avoid double-applying residuals.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        original_word_emb = model.deberta.embeddings.word_embeddings.weight.detach().clone()
        W_eff_prebake = sgcr_emb.effective_embedding_table().detach().clone()

    if also_save_sgcr:
        sgcr_state = {
            "base_word_embeddings": original_word_emb.cpu(),
            "component_embeddings": sgcr_emb.component_embeddings.state_dict(),
            "component_proj": sgcr_emb.component_proj.state_dict(),
            "rho": sgcr_emb.rho.cpu(),
            "decomp_ids": sgcr_emb.decomp_ids.cpu(),
            "decomp_lengths": sgcr_emb.decomp_lengths.cpu(),
            "state_role": "diagnostic_and_partial_resume_sidecar_without_optimizer_scheduler_rng",
        }
        torch.save(sgcr_state, save_dir / "sgcr_components.pt")

    removed_nonstandard_keys: list[str] = []
    try:
        with torch.no_grad():
            model.deberta.embeddings.word_embeddings.weight.copy_(W_eff_prebake)

        full_state_dict = model.state_dict()
        clean_state_dict = {
            k: v for k, v in full_state_dict.items()
            if not k.startswith("_sgcr")
        }
        removed_nonstandard_keys = sorted(k for k in full_state_dict if k.startswith("_sgcr"))
        model.save_pretrained(save_dir, state_dict=clean_state_dict, safe_serialization=True)
        tokenizer.save_pretrained(str(save_dir))
    finally:
        # Restore the live base table even if save_pretrained/tokenizer saving fails;
        # otherwise subsequent training would double-apply the SGCR residual.
        with torch.no_grad():
            model.deberta.embeddings.word_embeddings.weight.copy_(original_word_emb)

    with torch.no_grad():
        W_eff = sgcr_emb.effective_embedding_table()
        deviation = (W_eff - original_word_emb).norm(dim=1)
        restore_max_diff = (model.deberta.embeddings.word_embeddings.weight - original_word_emb).abs().max().item()

    return {
        "baked": True,
        "save_dir": str(save_dir),
        "mean_deviation": deviation.mean().item(),
        "max_deviation": deviation.max().item(),
        "tokens_with_deviation_gt_0.01": int((deviation > 0.01).sum().item()),
        "removed_nonstandard_state_keys": len(removed_nonstandard_keys),
        "removed_nonstandard_state_key_examples": removed_nonstandard_keys[:10],
        "saved_base_word_embeddings_in_sidecar": bool(also_save_sgcr),
        "restore_max_diff_after_save": restore_max_diff,
    }
