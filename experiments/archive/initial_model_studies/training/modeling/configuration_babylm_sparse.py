from __future__ import annotations

from transformers import PretrainedConfig


class BabyLMSparseConfig(PretrainedConfig):
    """Minimal causal Transformer config with routed sparse FFN blocks.

    This config is intentionally small and HF AutoModel-compatible so BabyLM's
    official evaluator can load local checkpoints with trust_remote_code=True.
    """

    model_type = "babylm_sparse"

    def __init__(
        self,
        vocab_size: int = 16384,
        n_positions: int = 256,
        n_embd: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        ffn_mult: int = 4,
        n_experts: int = 4,
        top_k: int = 2,
        router_aux_coef: float = 0.01,
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
        self.n_experts = n_experts
        self.top_k = top_k
        self.router_aux_coef = router_aux_coef
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        # Untied custom LM head avoids fragile shared-weight save metadata in the
        # current Transformers runtime; dense GPT2 controls remain tied.
        self.tie_word_embeddings = False
        self.auto_map = {
            "AutoConfig": "configuration_babylm_sparse.BabyLMSparseConfig",
            "AutoModelForCausalLM": "modeling_babylm_sparse.BabyLMSparseForCausalLM",
        }
        self.architectures = ["BabyLMSparseForCausalLM"]
