from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

try:
    from .configuration_babylm_memory import BabyLMMemoryConfig
except Exception:
    from configuration_babylm_memory import BabyLMMemoryConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: BabyLMMemoryConfig):
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
    def __init__(self, config: BabyLMMemoryConfig):
        super().__init__()
        hidden = config.ffn_mult * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden), nn.GELU(), nn.Linear(hidden, config.n_embd), nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrefixMemoryAdapter(nn.Module):
    """Vectorized causal prefix memory.

    For each token, build a gated prefix average of projected token states and
    feed it back through a learned gate. This gives a compact left-to-right
    persistent state without external entity labels or non-causal lookahead.
    """

    def __init__(self, config: BabyLMMemoryConfig):
        super().__init__()
        self.write = nn.Linear(config.n_embd, config.memory_dim)
        self.write_gate = nn.Linear(config.n_embd, config.memory_dim)
        self.read = nn.Linear(config.memory_dim, config.n_embd)
        self.fuse = nn.Linear(2 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.memory_dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        value = torch.tanh(self.write(x))
        gate = torch.sigmoid(self.write_gate(x))
        if attention_mask is not None:
            gate = gate * attention_mask.to(gate.dtype).unsqueeze(-1)
        num = torch.cumsum(gate * value, dim=1)
        den = torch.cumsum(gate, dim=1).clamp_min(1e-4)
        mem = num / den
        read = self.read(mem)
        fuse = torch.sigmoid(self.fuse(torch.cat([x, read], dim=-1)))
        delta = self.drop(fuse * read)
        y = x + delta
        stats = {
            "memory_write_gate_mean": gate.detach().float().mean(),
            "memory_fuse_mean": fuse.detach().float().mean(),
            "memory_state_norm": mem.detach().float().norm(dim=-1).mean(),
            "memory_delta_norm": delta.detach().float().norm(dim=-1).mean(),
        }
        return y, stats


class MemoryBlock(nn.Module):
    def __init__(self, config: BabyLMMemoryConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ffn = DenseFFN(config)
        self.ln_mem = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.memory = PrefixMemoryAdapter(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ln_2(x))
        mem_in = self.ln_mem(x)
        mem_out, stats = self.memory(mem_in, attention_mask=attention_mask)
        # Keep the main residual stream intact; add only the causal memory delta.
        x = x + (mem_out - mem_in)
        return x, stats


class BabyLMMemoryForCausalLM(PreTrainedModel):
    config_class = BabyLMMemoryConfig
    base_model_prefix = "babylm_memory"
    supports_gradient_checkpointing = False

    def __init__(self, config: BabyLMMemoryConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([MemoryBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()
        self._last_memory_stats: dict[str, Any] = {}

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

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None, labels: torch.LongTensor | None = None, **kwargs) -> CausalLMOutput:
        bsz, seq = input_ids.shape
        if seq > self.config.n_positions:
            raise ValueError(f"sequence length {seq} exceeds n_positions {self.config.n_positions}")
        pos = torch.arange(0, seq, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.wte(input_ids) + self.wpe(pos))
        stat_list = []
        for block in self.h:
            x, st = block(x, attention_mask=attention_mask)
            stat_list.append(st)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[..., :-1, :].contiguous().view(-1, logits.size(-1)), labels[..., 1:].contiguous().view(-1), ignore_index=-100)
        if stat_list:
            self._last_memory_stats = {k: float(torch.stack([s[k] for s in stat_list]).mean().cpu()) for k in stat_list[0]}
        return CausalLMOutput(loss=loss, logits=logits)
