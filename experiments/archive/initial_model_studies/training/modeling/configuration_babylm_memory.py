from __future__ import annotations

from transformers import PretrainedConfig


class BabyLMMemoryConfig(PretrainedConfig):
    """Dense causal Transformer with a compact left-to-right state memory.

    The memory channel is intended to test whether BabyLM Entity/Reading weakness
    reflects insufficient discourse-state persistence rather than only exposure or
    model size. It uses no external labels or tools and preserves a causal LM
    interface for official BabyLM evaluation.
    """

    model_type = "babylm_memory"

    def __init__(
        self,
        vocab_size: int = 16384,
        n_positions: int = 256,
        n_embd: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        ffn_mult: int = 4,
        memory_dim: int = 128,
        memory_layers: str = "all",
        memory_dropout: float = 0.0,
        memory_enabled: bool = True,
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        pad_token_id: int | None = 3,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.ffn_mult = ffn_mult
        self.memory_dim = memory_dim
        self.memory_layers = memory_layers
        self.memory_dropout = memory_dropout
        self.memory_enabled = memory_enabled
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        # Custom model heads are untied to avoid the current Transformers custom
        # safe-serialization shared-tensor failure; dense GPT2 controls are tied.
        self.tie_word_embeddings = False
        self.auto_map = {
            "AutoConfig": "configuration_babylm_memory.BabyLMMemoryConfig",
            "AutoModelForCausalLM": "modeling_babylm_memory.BabyLMMemoryForCausalLM",
        }
        self.architectures = ["BabyLMMemoryForCausalLM"]
