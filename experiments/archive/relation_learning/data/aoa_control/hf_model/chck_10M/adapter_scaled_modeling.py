#!/usr/bin/env python3
"""research train-time scaled residual-adapter DeBERTa-v2 MLM.

State-dict layout matches the research/103 adapter model.  Each post-layer adapter
update is multiplied by `config.adapter_scale`, allowing a trajectory-consistent
training test of the amplitude window found by the research inference-scale map.
Because W_up and its bias are zero at initialization, all adapter scales preserve
the exact stock function at research; the scale only changes the branch gradient and
future forward amplitude once the branch recruits.
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

_THIS_FILE = _public_path('experiments/archive/relation_learning/data/aoa_control/hf_model/chck_10M/adapter_scaled_modeling.py')


class ZeroOutputBottleneckAdapter(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        width = int(getattr(config, "adapter_bottleneck", 64))
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.down = nn.Linear(config.hidden_size, width)
        self.up = nn.Linear(width, config.hidden_size)
        self.activation = ACT2FN[getattr(config, "adapter_activation", "gelu")]
        self.enabled = bool(getattr(config, "adapter_enabled", True))
        self.scale = float(getattr(config, "adapter_scale", 1.0))
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


class AdapterDebertaV2Layer(DebertaV2Layer):
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
    config_class = DebertaV2Config

    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        config.adapter_bottleneck = int(getattr(config, "adapter_bottleneck", 64))
        config.adapter_activation = getattr(config, "adapter_activation", "gelu")
        config.adapter_enabled = bool(getattr(config, "adapter_enabled", True))
        config.adapter_scale = float(getattr(config, "adapter_scale", 1.0))
        config.architectures = [self.__class__.__name__]
        config.auto_map = {
            "AutoModelForMaskedLM": "adapter_scaled_modeling.AdapterDebertaV2ForMaskedLM"
        }
        for layer in self.deberta.encoder.layer:
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
        dst = Path(save_directory) / _THIS_FILE.name
        if not dst.exists():
            shutil.copy2(_THIS_FILE, dst)
