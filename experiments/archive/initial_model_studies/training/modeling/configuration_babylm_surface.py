from __future__ import annotations

from transformers import PretrainedConfig


class BabyLMSurfaceConfig(PretrainedConfig):
    """Dense causal Transformer with a gated token-surface side channel.

    surface_mode="char_ngram" uses a learned character/n-gram embedding table
    shared across tokenizer vocabulary items. Tokenizer strings are mapped to a
    fixed list of n-gram IDs, then composed by learned embeddings and projection.
    surface_mode="lookup" uses a low-rank learned per-token adapter without
    cross-token character sharing, serving as an added-capacity control with the
    same fusion path.
    """

    model_type = "babylm_surface"

    def __init__(
        self,
        vocab_size: int = 16384,
        n_positions: int = 256,
        n_embd: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        ffn_mult: int = 4,
        surface_mode: str = "char_ngram",
        surface_dim: int = 64,
        ngram_vocab_size: int = 1024,
        max_ngrams_per_token: int = 16,
        lookup_rank: int = 5,
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
        if surface_mode not in {"char_ngram", "lookup"}:
            raise ValueError(f"unknown surface_mode {surface_mode}")
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.ffn_mult = ffn_mult
        self.surface_mode = surface_mode
        self.surface_dim = surface_dim
        self.ngram_vocab_size = ngram_vocab_size
        self.max_ngrams_per_token = max_ngrams_per_token
        self.lookup_rank = lookup_rank
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.tie_word_embeddings = False
        self.auto_map = {
            "AutoConfig": "configuration_babylm_surface.BabyLMSurfaceConfig",
            "AutoModelForCausalLM": "modeling_babylm_surface.BabyLMSurfaceForCausalLM",
        }
        self.architectures = ["BabyLMSurfaceForCausalLM"]
