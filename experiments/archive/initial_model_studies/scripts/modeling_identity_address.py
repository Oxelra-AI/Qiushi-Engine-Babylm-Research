"""Official-loadable DeBERTa-v2 models with identity-addressed state retrieval.

The mechanism is parameter-neutral.  At prediction positions, a repeated
content cue in the local prefix addresses its previous occurrence.  A compact
continuation memory is written into a fixed residual-stream subspace after an
early encoder layer.  Address construction and retrieval are vectorized on the
model device; there are no Python loops over examples or token positions.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2ForMaskedLM, DebertaV2Model


class IdentityAddressController(nn.Module):
    MODES = {"off", "shuffled", "true"}

    def __init__(self, config, encoder: nn.Module) -> None:
        super().__init__()
        mode = str(getattr(config, "identity_address_mode", "off"))
        if mode not in self.MODES:
            raise ValueError(f"unknown identity_address_mode: {mode}")
        self.mode = mode
        self.mask_token_id = int(getattr(config, "identity_address_mask_token_id"))
        self.local_window = int(getattr(config, "identity_address_local_window", 24))
        self.continuation_window = int(
            getattr(config, "identity_address_continuation_window", 10)
        )
        self.state_dimensions = int(
            getattr(config, "identity_address_state_dimensions", config.hidden_size // 3)
        )
        self.insert_after_layer = int(
            getattr(config, "identity_address_insert_after_layer", config.num_hidden_layers // 2 - 1)
        )
        if not 0 <= self.insert_after_layer < config.num_hidden_layers:
            raise ValueError(f"invalid identity_address_insert_after_layer: {self.insert_after_layer}")
        if not 0 < self.state_dimensions < config.hidden_size:
            raise ValueError(f"invalid identity_address_state_dimensions: {self.state_dimensions}")
        eligible = list(getattr(config, "identity_address_eligible_token_ids", []))
        lookup = torch.zeros(config.vocab_size, dtype=torch.bool)
        if eligible:
            lookup[torch.tensor(eligible, dtype=torch.long)] = True
        self.register_buffer("eligible_lookup", lookup, persistent=False)
        self.register_buffer("active_target_count", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("total_target_count", torch.zeros((), dtype=torch.long), persistent=False)
        self._input_ids: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._hook = encoder.layer[self.insert_after_layer].register_forward_hook(
            self._hook_hidden
        )

    def set_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> None:
        self._input_ids = input_ids
        self._attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        )
        self._labels = labels

    def clear_context(self) -> None:
        self._input_ids = None
        self._attention_mask = None
        self._labels = None

    @torch.no_grad()
    def address_map(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._input_ids is None or self._attention_mask is None:
            raise RuntimeError("identity-address context is not set")
        input_ids = self._input_ids
        attention = self._attention_mask.bool()
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        if self._labels is None:
            targets = attention & input_ids.eq(self.mask_token_id)
        else:
            targets = attention & self._labels.ne(-100)

        safe_ids = input_ids.clamp(min=0, max=self.eligible_lookup.numel() - 1)
        cue_valid = attention & self.eligible_lookup[safe_ids]
        equal = input_ids.unsqueeze(2).eq(input_ids.unsqueeze(1))
        cue_position = positions.view(length, 1)
        prior_position = positions.view(1, length)
        legal_prior = prior_position <= cue_position - 3
        previous = torch.where(
            equal & legal_prior.unsqueeze(0) & attention.unsqueeze(1),
            prior_position.view(1, 1, length),
            torch.full((1, 1, 1), -1, device=input_ids.device, dtype=torch.long),
        ).amax(dim=2)

        target_position = positions.view(length, 1)
        local_cue_position = positions.view(1, length)
        in_local_prefix = (
            (local_cue_position < target_position)
            & (local_cue_position >= target_position - self.local_window)
        )
        candidates = (
            targets.unsqueeze(2)
            & cue_valid.unsqueeze(1)
            & previous.ge(0).unsqueeze(1)
            & in_local_prefix.unsqueeze(0)
        )
        cue_indices = torch.where(
            candidates,
            positions.view(1, 1, length),
            torch.full((1, 1, 1), -1, device=input_ids.device, dtype=torch.long),
        ).amax(dim=2)
        active = cue_indices.ge(0)
        starts = previous.gather(1, cue_indices.clamp_min(0))
        if self.mode == "shuffled" and active.any():
            values = starts[active]
            starts = starts.clone()
            starts[active] = torch.roll(values, shifts=1, dims=0)
        self.active_target_count.add_(active.sum())
        self.total_target_count.add_(targets.sum())
        return starts, active, targets

    def telemetry(self) -> dict[str, float | int]:
        active = int(self.active_target_count.detach().cpu())
        total = int(self.total_target_count.detach().cpu())
        return {
            "active_targets": active,
            "total_targets": total,
            "activation_rate": active / max(1, total),
        }

    def _hook_hidden(self, module, inputs, output):
        del module, inputs
        if self.mode == "off" or self._input_ids is None or self._attention_mask is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        starts, active, _ = self.address_map()
        active_indices = active.nonzero(as_tuple=False)
        if active_indices.numel() == 0:
            return output

        batch, length, width = hidden.shape
        row_indices = active_indices[:, 0]
        target_indices = active_indices[:, 1]
        active_starts = starts[row_indices, target_indices]
        offsets = torch.arange(self.continuation_window, device=hidden.device)
        source_positions = active_starts.unsqueeze(1) + offsets.unsqueeze(0)
        in_range = source_positions.lt(length)
        source_positions = source_positions.clamp(max=length - 1)
        source_valid = self._attention_mask[row_indices.unsqueeze(1), source_positions].bool()
        valid = in_range & source_valid
        flat_indices = row_indices.unsqueeze(1) * length + source_positions
        gathered = hidden.reshape(batch * length, width)[flat_indices]
        weights = valid.unsqueeze(-1).to(hidden.dtype)
        memory = (gathered * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        state = F.layer_norm(
            memory[:, -self.state_dimensions :], (self.state_dimensions,)
        )
        transported = hidden.clone()
        transported[row_indices, target_indices, -self.state_dimensions :] = state
        if isinstance(output, tuple):
            return (transported,) + output[1:]
        return transported


class IdentityAddressDebertaForMaskedLM(DebertaV2ForMaskedLM):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.identity_address = IdentityAddressController(config, self.deberta.encoder)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if input_ids is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        self.identity_address.set_context(input_ids, attention_mask, labels)
        try:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            self.identity_address.clear_context()


class IdentityAddressDebertaModel(DebertaV2Model):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.identity_address = IdentityAddressController(config, self.encoder)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if input_ids is None:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
        self.identity_address.set_context(input_ids, attention_mask, None)
        try:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
        finally:
            self.identity_address.clear_context()
