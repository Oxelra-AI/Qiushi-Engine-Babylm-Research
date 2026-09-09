"""Minimal, testable SGCR repair candidate for the research audit.

This file is intentionally independent of Qiushi's live scripts.  It implements
the launch-critical mechanics only: an exact-preserving but gradient-live
initialization, special/missing-token protection, shared effective input/output
weights, and clean baking to a standard Hugging Face checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SGCRConfig:
    vocab_40k: int
    vocab_16k: int
    hidden_size: int
    d_comp: int = 64
    K: float = 50.0
    uniform_gate: bool = False
    force_standard_ids: set[int] = field(default_factory=set)


def build_prefix_decomposition_map(
    tok40k_path: str | Path,
    tok16k_path: str | Path,
) -> dict[int, list[int]]:
    """Exactly split legal40k BPE tokens at the legal16k merge frontier.

    The two supplied legal tokenizers share identical vocab ids through 16,384
    and the legal16k merge list is an exact prefix of the legal40k list.  Using
    that structural relation avoids two defects in decode/re-encode mapping:
    fixed padding in the raw legal16k tokenizer and lossy/context-sensitive
    decoding of isolated byte-level tokens.
    """
    def tokenizer_json(path: str | Path) -> Path:
        path = Path(path)
        return path if path.name == "tokenizer.json" else path / "tokenizer.json"

    with tokenizer_json(tok16k_path).open(encoding="utf-8") as handle:
        raw16 = json.load(handle)
    with tokenizer_json(tok40k_path).open(encoding="utf-8") as handle:
        raw40 = json.load(handle)
    if raw16["model"]["type"] != "BPE" or raw40["model"]["type"] != "BPE":
        raise ValueError("prefix decomposition requires BPE tokenizers")
    vocab16: dict[str, int] = raw16["model"]["vocab"]
    vocab40: dict[str, int] = raw40["model"]["vocab"]
    merges16 = raw16["model"]["merges"]
    merges40 = raw40["model"]["merges"]
    if merges16 != merges40[: len(merges16)]:
        raise ValueError("legal16k merges are not a prefix of legal40k merges")
    if any(vocab40.get(piece) != token_id for piece, token_id in vocab16.items()):
        raise ValueError("legal16k vocabulary ids are not an exact legal40k prefix")

    id_to_piece = {token_id: piece for piece, token_id in vocab40.items()}
    reverse_merge = {left + right: (left, right) for left, right in merges40}
    cutoff = len(vocab16)
    memo: dict[str, tuple[int, ...]] = {}

    def split(piece: str) -> tuple[int, ...]:
        if piece in memo:
            return memo[piece]
        token_id = vocab40[piece]
        if token_id < cutoff:
            result = (token_id,)
        else:
            if piece not in reverse_merge:
                raise ValueError(f"no merge ancestry for legal40k token {piece!r}")
            left, right = reverse_merge[piece]
            result = split(left) + split(right)
        memo[piece] = result
        return result

    return {
        token_id: list(split(id_to_piece[token_id]))
        for token_id in range(len(vocab40))
    }


def build_sgcr_buffers(
    decomposition_map: dict[int, list[int]],
    counts_40k: dict[int, int],
    config: SGCRConfig,
) -> dict[str, torch.Tensor]:
    """Build validated decomposition and gate buffers.

    Tokens without a legal decomposition and special/control ids are forced to
    rho=1, so they retain the standard embedding exactly.
    """
    if config.K < 0:
        raise ValueError("K must be nonnegative")
    max_len = max((len(v) for v in decomposition_map.values()), default=1)
    max_len = max(max_len, 1)
    decomp_ids = torch.zeros(config.vocab_40k, max_len, dtype=torch.long)
    decomp_lengths = torch.zeros(config.vocab_40k, dtype=torch.long)
    for token_id in range(config.vocab_40k):
        components = decomposition_map.get(token_id, [])
        if any(c < 0 or c >= config.vocab_16k for c in components):
            raise ValueError(f"component id out of range for token {token_id}")
        if components:
            decomp_ids[token_id, : len(components)] = torch.tensor(components)
            decomp_lengths[token_id] = len(components)

    forced = torch.tensor(
        [
            token_id in config.force_standard_ids
            or decomp_lengths[token_id].item() == 0
            for token_id in range(config.vocab_40k)
        ],
        dtype=torch.bool,
    )
    counts = torch.tensor(
        [counts_40k.get(token_id, 0) for token_id in range(config.vocab_40k)],
        dtype=torch.float32,
    )
    if config.K == 0:
        rho = torch.ones(config.vocab_40k, dtype=torch.float32)
    else:
        rho = counts / (counts + config.K)
        rho[forced] = 1.0

    if config.uniform_gate:
        ordinary_used = (counts > 0) & ~forced
        if ordinary_used.any():
            # Match the average residual exposure over training-token mass.  A
            # type mean would make K=50 far stronger than the routed treatment.
            mass = counts[ordinary_used]
            uniform_residual = (
                (mass * (1.0 - rho[ordinary_used])).sum() / mass.sum()
            )
            rho[~forced] = 1.0 - uniform_residual
        rho[forced] = 1.0

    return {
        "decomp_ids": decomp_ids,
        "decomp_lengths": decomp_lengths,
        "rho": rho,
    }


class SGCREmbedding(nn.Module):
    """Additive SGCR table with semi-cold exact-preserving initialization."""

    def __init__(
        self,
        word_embeddings: nn.Embedding,
        config: SGCRConfig,
        decomp_ids: torch.Tensor,
        decomp_lengths: torch.Tensor,
        rho: torch.Tensor,
    ) -> None:
        super().__init__()
        if word_embeddings.num_embeddings != config.vocab_40k:
            raise ValueError("40k vocabulary size does not match base embedding")
        if word_embeddings.embedding_dim != config.hidden_size:
            raise ValueError("hidden size does not match base embedding")
        self.word_embeddings = word_embeddings
        self.config = config
        self.component_embeddings = nn.Embedding(config.vocab_16k, config.d_comp)
        self.component_proj = nn.Linear(config.d_comp, config.hidden_size, bias=True)
        self.register_buffer("decomp_ids", decomp_ids)
        self.register_buffer("decomp_lengths", decomp_lengths)
        self.register_buffer("rho", rho)

        # W_eff is exactly W_std because component means and bias are zero, but
        # the nonzero orthogonal projection makes dL/d(component_embeddings)
        # nonzero on the first backward pass.  Zeroing both factors is dead.
        nn.init.zeros_(self.component_embeddings.weight)
        nn.init.orthogonal_(self.component_proj.weight)
        nn.init.zeros_(self.component_proj.bias)

    def effective_embedding_table(self) -> torch.Tensor:
        component_values = self.component_embeddings(self.decomp_ids)
        max_len = self.decomp_ids.shape[1]
        valid = (
            torch.arange(max_len, device=self.decomp_ids.device)[None, :]
            < self.decomp_lengths[:, None]
        )
        # Use the parameter dtype rather than forcing float32, so this remains
        # valid if a future trainer intentionally uses bf16.
        valid_f = valid.unsqueeze(-1).to(component_values.dtype)
        denom = self.decomp_lengths.clamp(min=1).unsqueeze(1).to(component_values.dtype)
        component_mean = (component_values * valid_f).sum(1) / denom
        correction = self.component_proj(component_mean)
        residual_gate = (1.0 - self.rho).unsqueeze(1).to(correction.dtype)
        return self.word_embeddings.weight + residual_gate * correction

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.effective_embedding_table())


def apply_sgcr_to_model(
    model: Any,
    config: SGCRConfig,
    decomposition_map: dict[int, list[int]],
    counts_40k: dict[int, int],
) -> tuple[Any, SGCREmbedding]:
    buffers = build_sgcr_buffers(decomposition_map, counts_40k, config)
    word_embeddings = model.deberta.embeddings.word_embeddings
    sgcr = SGCREmbedding(word_embeddings, config, **buffers)
    sgcr.to(device=word_embeddings.weight.device, dtype=word_embeddings.weight.dtype)
    model._sgcr_embedding = sgcr

    # Instance-level closures keep the normal DeBERTa-v2 forward topology.  The
    # same SGCR object supplies both effective tables, so input and decoder are
    # value-tied and receive joint gradients.
    def input_forward(input_ids: torch.Tensor) -> torch.Tensor:
        return sgcr(input_ids)

    def decoder_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(
            hidden_states,
            sgcr.effective_embedding_table(),
            model.cls.predictions.bias,
        )

    word_embeddings.forward = input_forward
    model.cls.predictions.decoder.forward = decoder_forward
    return model, sgcr


def save_baked_checkpoint(
    model: Any,
    sgcr: SGCREmbedding,
    tokenizer: Any,
    save_dir: str | Path,
    save_resume_state: bool = True,
) -> dict[str, Any]:
    """Save a clean standard checkpoint and restore the live training model."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        base_weight = sgcr.word_embeddings.weight.detach().clone()
        effective_weight = sgcr.effective_embedding_table().detach().clone()

    if save_resume_state:
        # The base table is essential: resuming from the baked W_eff and adding
        # the saved residual again would double-apply SGCR.
        torch.save(
            {
                "base_word_embeddings": base_weight.cpu(),
                "component_embeddings": sgcr.component_embeddings.state_dict(),
                "component_proj": sgcr.component_proj.state_dict(),
                "rho": sgcr.rho.cpu(),
                "decomp_ids": sgcr.decomp_ids.cpu(),
                "decomp_lengths": sgcr.decomp_lengths.cpu(),
            },
            save_dir / "sgcr_resume_state.pt",
        )

    removed: list[str] = []
    try:
        with torch.no_grad():
            sgcr.word_embeddings.weight.copy_(effective_weight)
        full_state = model.state_dict()
        removed = sorted(k for k in full_state if k.startswith("_sgcr"))
        clean_state = {k: v for k, v in full_state.items() if not k.startswith("_sgcr")}
        model.save_pretrained(
            save_dir,
            state_dict=clean_state,
            safe_serialization=True,
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(str(save_dir))
    finally:
        with torch.no_grad():
            sgcr.word_embeddings.weight.copy_(base_weight)

    return {
        "baked": True,
        "removed_nonstandard_state_keys": len(removed),
        "max_embedding_delta": float((effective_weight - base_weight).abs().max()),
    }
