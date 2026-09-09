from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

try:
    from .configuration_babylm_sparse import BabyLMSparseConfig
except Exception:  # local direct import fallback used by trust_remote_code copies
    from configuration_babylm_sparse import BabyLMSparseConfig


@dataclass
class RouterStats:
    entropy: torch.Tensor
    load: torch.Tensor
    top1_load: torch.Tensor
    aux_loss: torch.Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, config: BabyLMSparseConfig):
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


class RoutedFFN(nn.Module):
    def __init__(self, config: BabyLMSparseConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = min(config.top_k, config.n_experts)
        hidden = config.ffn_mult * config.n_embd
        self.router = nn.Linear(config.n_embd, config.n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.n_embd),
                nn.Dropout(config.resid_pdrop),
            )
            for _ in range(config.n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RouterStats]:
        logits = self.router(x)
        probs = F.softmax(logits.float(), dim=-1).to(x.dtype)
        top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_weights = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        y = torch.zeros_like(x)
        # Simple expert loop is fine for BabyLM-scale pilots and keeps behavior transparent.
        for e, expert in enumerate(self.experts):
            mask = (top_idx == e)
            if not mask.any():
                continue
            weights = (mask.to(x.dtype) * top_weights).sum(dim=-1, keepdim=True)
            y = y + expert(x) * weights
        mean_prob = probs.float().mean(dim=(0, 1))
        top1 = top_idx[..., 0]
        top1_load = F.one_hot(top1, num_classes=self.n_experts).float().mean(dim=(0, 1))
        entropy = (-(probs.float() * probs.float().clamp_min(1e-9).log()).sum(dim=-1)).mean()
        # Encourage avoiding collapse, but keep coefficient small in the trainer.
        aux_loss = self.n_experts * torch.sum(mean_prob * top1_load)
        return y, RouterStats(entropy=entropy, load=mean_prob, top1_load=top1_load, aux_loss=aux_loss)


class SparseBlock(nn.Module):
    def __init__(self, config: BabyLMSparseConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ffn = RoutedFFN(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, RouterStats]:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        ffn_out, stats = self.ffn(self.ln_2(x))
        x = x + ffn_out
        return x, stats


class BabyLMSparseForCausalLM(PreTrainedModel):
    config_class = BabyLMSparseConfig
    base_model_prefix = "babylm_sparse"
    supports_gradient_checkpointing = False

    def __init__(self, config: BabyLMSparseConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([SparseBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()
        self._last_router_stats: dict[str, Any] = {}

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
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)
        stats = []
        aux_loss = input_ids.new_tensor(0.0, dtype=torch.float32)
        for block in self.h:
            x, st = block(x, attention_mask=attention_mask)
            stats.append(st)
            aux_loss = aux_loss + st.aux_loss
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            if self.config.router_aux_coef:
                loss = loss + float(self.config.router_aux_coef) * aux_loss / max(1, len(self.h))
        if stats:
            load = torch.stack([s.load for s in stats]).mean(dim=0)
            top1_load = torch.stack([s.top1_load for s in stats]).mean(dim=0)
            entropy = torch.stack([s.entropy for s in stats]).mean()
            self._last_router_stats = {
                "router_entropy": float(entropy.detach().cpu()),
                "router_load": [float(x) for x in load.detach().cpu()],
                "router_top1_load": [float(x) for x in top1_load.detach().cpu()],
                "router_aux_loss": float((aux_loss / max(1, len(self.h))).detach().cpu()),
            }
        return CausalLMOutput(loss=loss, logits=logits)
