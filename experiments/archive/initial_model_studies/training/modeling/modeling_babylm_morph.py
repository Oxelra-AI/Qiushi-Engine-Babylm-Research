from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

try:
    from .configuration_babylm_morph import BabyLMMorphConfig
except Exception:
    from configuration_babylm_morph import BabyLMMorphConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: BabyLMMorphConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq, dim = x.shape
        q, k, v = self.c_attn(x).split(dim, dim=2)
        q = q.view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal = torch.tril(torch.ones(seq, seq, device=x.device, dtype=torch.bool)).view(1, 1, seq, seq)
        att = att.masked_fill(~causal, torch.finfo(att.dtype).min)
        if attention_mask is not None:
            key_mask = attention_mask.to(torch.bool).view(bsz, 1, 1, seq)
            att = att.masked_fill(~key_mask, torch.finfo(att.dtype).min)
        att = F.softmax(att.float(), dim=-1).to(x.dtype)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq, dim)
        return self.resid_drop(self.c_proj(y))


class DenseFFN(nn.Module):
    def __init__(self, config: BabyLMMorphConfig):
        super().__init__()
        hidden = config.ffn_mult * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseBlock(nn.Module):
    def __init__(self, config: BabyLMMorphConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ffn = DenseFFN(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ln_2(x))
        return x


class BabyLMMorphForCausalLM(PreTrainedModel):
    config_class = BabyLMMorphConfig
    base_model_prefix = "babylm_morph"
    supports_gradient_checkpointing = False

    def __init__(self, config: BabyLMMorphConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        # Persistent side channel: fixed token-form features are saved as a buffer,
        # then projected and fused on every forward pass.
        self.register_buffer("morph_features", torch.zeros(config.vocab_size, config.morph_dim), persistent=True)
        self.morph_proj = nn.Linear(config.morph_dim, config.n_embd, bias=False)
        self.morph_fuse = nn.Linear(2 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([DenseBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()
        self._last_morph_stats: dict[str, Any] = {}

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        bsz, seq = input_ids.shape
        if seq > self.config.n_positions:
            raise ValueError(f"sequence length {seq} exceeds n_positions {self.config.n_positions}")
        pos = torch.arange(0, seq, device=input_ids.device).unsqueeze(0)
        base = self.wte(input_ids)
        side = self.morph_proj(self.morph_features[input_ids].to(base.dtype))
        # Learned fusion is persistent and token-specific through the side features;
        # this is not a one-time embedding perturbation.
        mix = torch.sigmoid(self.morph_fuse(torch.cat([base, side], dim=-1)))
        x = base + mix * side + self.wpe(pos)
        x = self.drop(x)
        for block in self.h:
            x = block(x, attention_mask=attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        self._last_morph_stats = {
            "morph_mix_mean": float(mix.detach().float().mean().cpu()),
            "morph_side_norm": float(side.detach().float().norm(dim=-1).mean().cpu()),
            "morph_base_norm": float(base.detach().float().norm(dim=-1).mean().cpu()),
        }
        return CausalLMOutput(loss=loss, logits=logits)
