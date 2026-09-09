#!/usr/bin/env python3
"""Function-preserving residual-adapter DeBERTa-v2 MLM.

The stock DeBERTa-v2 encoder layer names are preserved.  Each layer receives one
post-layer bottleneck branch

    h+ = h + W_up(GELU(W_down(LN(h))))

where W_up and its bias are exactly zero at initialization.  Thus the augmented
model is exactly the stock model at initialization while W_up receives a useful
first-step gradient.  This file is copied beside config.json by the research
trainer so AutoModelForMaskedLM can load checkpoints with trust_remote_code=True.
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

_THIS_FILE = _public_path('experiments/archive/frontier_consolidation/scripts/adapter_modeling.py')


class ZeroOutputBottleneckAdapter(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        width = int(getattr(config, "adapter_bottleneck", 64))
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.down = nn.Linear(config.hidden_size, width)
        self.up = nn.Linear(width, config.hidden_size)
        self.activation = ACT2FN[getattr(config, "adapter_activation", "gelu")]
        self.enabled = bool(getattr(config, "adapter_enabled", True))
        # Exact zero admission; down remains normally initialized by nn.Linear.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.last_rms = 0.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(hidden_states)
        update = self.up(self.activation(self.down(self.layer_norm(hidden_states))))
        if not torch.jit.is_tracing():
            self.last_rms = float(update.detach().float().square().mean().sqrt().cpu())
        return update


class AdapterDebertaV2Layer(DebertaV2Layer):
    """DebertaV2Layer-compatible forward with one separately routed branch."""
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
        layer_output = self.output(intermediate_output, attention_output)
        layer_output = layer_output + self.adapter(layer_output)
        return (layer_output, att_matrix if output_attentions else None)


class AdapterDebertaV2ForMaskedLM(DebertaV2ForMaskedLM):
    """Stock DeBERTa-v2 MLM plus zero-output bottleneck adapters."""
    config_class = DebertaV2Config

    def __init__(self, config: DebertaV2Config):
        # Stock construction occurs first, preserving every original RNG draw.
        super().__init__(config)
        config.adapter_bottleneck = int(getattr(config, "adapter_bottleneck", 64))
        config.adapter_activation = getattr(config, "adapter_activation", "gelu")
        config.adapter_enabled = bool(getattr(config, "adapter_enabled", True))
        config.architectures = [self.__class__.__name__]
        config.auto_map = {
            "AutoModelForMaskedLM": "adapter_modeling.AdapterDebertaV2ForMaskedLM"
        }
        for layer in self.deberta.encoder.layer:
            # Preserve all stock parameter paths; add only `.adapter.*` names.
            layer.__class__ = AdapterDebertaV2Layer
            layer.adapter = ZeroOutputBottleneckAdapter(config)

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if ".adapter." in name:
                yield name, parameter

    def adapter_rms(self) -> list[float]:
        return [float(layer.adapter.last_rms) for layer in self.deberta.encoder.layer]

    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)
        # Ensure the custom modeling source is beside config.json so
        # AutoModelForMaskedLM(..., trust_remote_code=True) can reload.
        dst = Path(save_directory) / _THIS_FILE.name
        if not dst.exists():
            shutil.copy2(_THIS_FILE, dst)
