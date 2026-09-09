#!/usr/bin/env python3
"""Frozen-slow + fresh-private residual DeBERTa-v2 MLM for research.

This model preserves the verified scale1.75 chck_82M function as the slow path and
adds a second, fresh zero-output private adapter after each completed encoder layer.
The existing research adapter remains in the state-dict under `.adapter.*` and uses
`config.adapter_scale` (1.75 for the chck_82M endpoint).  The new trainable pathway
is under `.private_adapter.*` and uses `config.private_adapter_scale`.

Layer computation:

    base = output(intermediate(attention(h)), attention(h))
    slow = base + old_adapter(base)             # loaded/frozen chck_82M function
    h_next = slow + private_adapter(slow)       # zero at attachment time

Because private_adapter.up and its bias are initialized to zero, enabling the private
path at initialization is exactly function-preserving.  Training scripts can freeze all
non-private parameters and update only `.private_adapter.*` while keeping the slow
function loadable and recoverable by disabling private adapters.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import shutil
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import DebertaV2Config, DebertaV2ForMaskedLM
from transformers.activations import ACT2FN
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Layer

_THIS_FILE = _public_path('experiments/archive/relation_learning/data/format_replay_corrected/isolated_all/seed98098/alpha_0.75/frozen82_private_modeling.py')


class ScaledBottleneckAdapter(nn.Module):
    def __init__(self, config: DebertaV2Config, *, bottleneck_attr: str, scale_attr: str,
                 enabled_attr: str, activation_attr: str = "adapter_activation"):
        super().__init__()
        width = int(getattr(config, bottleneck_attr, getattr(config, "adapter_bottleneck", 64)))
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.down = nn.Linear(config.hidden_size, width)
        self.up = nn.Linear(width, config.hidden_size)
        self.activation = ACT2FN[getattr(config, activation_attr, "gelu")]
        self.enabled = bool(getattr(config, enabled_attr, True))
        self.scale = float(getattr(config, scale_attr, 1.0))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.last_rms = 0.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.scale == 0.0:
            self.last_rms = 0.0
            return torch.zeros_like(hidden_states)
        update = self.up(self.activation(self.down(self.layer_norm(hidden_states)))) * self.scale
        if not torch.jit.is_tracing():
            self.last_rms = float(update.detach().float().square().mean().sqrt().cpu())
        return update


class FrozenSlowPrivateDebertaV2Layer(DebertaV2Layer):
    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_output, att_matrix = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        intermediate_output = self.intermediate(attention_output)
        base_output = self.output(intermediate_output, attention_output)
        slow_output = base_output + self.adapter(base_output)
        layer_output = slow_output + self.private_adapter(slow_output)
        return (layer_output, att_matrix if output_attentions else None)


class FrozenSlowPrivateDebertaV2ForMaskedLM(DebertaV2ForMaskedLM):
    config_class = DebertaV2Config

    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        # Existing slow-path adapter fields; for chck_82M these are loaded from
        # research and should be adapter_bottleneck=128, adapter_scale=1.75.
        config.adapter_bottleneck = int(getattr(config, "adapter_bottleneck", 128))
        config.adapter_activation = getattr(config, "adapter_activation", "gelu")
        config.adapter_enabled = bool(getattr(config, "adapter_enabled", True))
        config.adapter_scale = float(getattr(config, "adapter_scale", 1.75))

        # Fresh private pathway fields.
        config.private_adapter_bottleneck = int(getattr(config, "private_adapter_bottleneck", 128))
        config.private_adapter_activation = getattr(config, "private_adapter_activation", "gelu")
        config.private_adapter_enabled = bool(getattr(config, "private_adapter_enabled", True))
        config.private_adapter_scale = float(getattr(config, "private_adapter_scale", 1.0))

        config.architectures = [self.__class__.__name__]
        config.auto_map = {
            "AutoModelForMaskedLM": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM"
        }
        for layer in self.deberta.encoder.layer:
            layer.__class__ = FrozenSlowPrivateDebertaV2Layer
            layer.adapter = ScaledBottleneckAdapter(
                config,
                bottleneck_attr="adapter_bottleneck",
                scale_attr="adapter_scale",
                enabled_attr="adapter_enabled",
                activation_attr="adapter_activation",
            )
            layer.private_adapter = ScaledBottleneckAdapter(
                config,
                bottleneck_attr="private_adapter_bottleneck",
                scale_attr="private_adapter_scale",
                enabled_attr="private_adapter_enabled",
                activation_attr="private_adapter_activation",
            )

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if ".adapter." in name and ".private_adapter." not in name:
                yield name, parameter

    def private_adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if ".private_adapter." in name:
                yield name, parameter

    def private_adapter_rms(self) -> list[float]:
        return [float(layer.private_adapter.last_rms) for layer in self.deberta.encoder.layer]

    def slow_adapter_rms(self) -> list[float]:
        return [float(layer.adapter.last_rms) for layer in self.deberta.encoder.layer]

    def set_private_enabled(self, enabled: bool):
        for layer in self.deberta.encoder.layer:
            layer.private_adapter.enabled = bool(enabled)
        self.config.private_adapter_enabled = bool(enabled)

    def set_slow_adapter_enabled(self, enabled: bool):
        for layer in self.deberta.encoder.layer:
            layer.adapter.enabled = bool(enabled)
        self.config.adapter_enabled = bool(enabled)

    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)
        dst = Path(save_directory) / _THIS_FILE.name
        if not dst.exists():
            shutil.copy2(_THIS_FILE, dst)
