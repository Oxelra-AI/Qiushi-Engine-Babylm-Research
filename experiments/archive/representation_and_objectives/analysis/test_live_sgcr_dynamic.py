"""CPU behavioral tests against the live research SGCR module."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM


ROOT = _public_path('.')
sys.path.insert(0, str(_public_path('experiments/archive/representation_and_objectives/scripts')))
import sgcr_module as live  # noqa: E402
sys.path.insert(0, str(_public_path('experiments/archive/compact_experience/scripts')))
import masking_curriculum_trainer as base  # noqa: E402


VOCAB = 29
COMP_VOCAB = 17
HIDDEN = 16


class NoOpTokenizer:
    def save_pretrained(self, path):
        return (str(path),)


def tiny_model(seed: int = 123) -> DebertaV2ForMaskedLM:
    torch.manual_seed(seed)
    config = DebertaV2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=32,
        relative_attention=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
    )
    return DebertaV2ForMaskedLM(config)


def decomposition() -> dict[int, list[int]]:
    return {token_id: [token_id % COMP_VOCAB, (token_id + 3) % COMP_VOCAB] for token_id in range(VOCAB)}


def counts() -> dict[int, int]:
    return {token_id: token_id + 1 for token_id in range(VOCAB)}


def attach(model: DebertaV2ForMaskedLM, K: float = 7.0):
    config = live.SGCRConfig(
        vocab_40k=VOCAB,
        vocab_16k=COMP_VOCAB,
        hidden_size=HIDDEN,
        d_comp=5,
        K=K,
        force_standard_ids={0, 1, 2, 3, 4},
    )
    model, sgcr = live.apply_sgcr_to_model(model, config, decomposition(), counts())
    return live.sgcr_forward_embedding_hook(model, sgcr), sgcr


@pytest.fixture
def batch():
    input_ids = torch.tensor([[1, 5, 6, 7, 2], [1, 8, 9, 10, 2]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, 5, -100, 7, -100], [-100, 8, 9, -100, -100]])
    return input_ids, attention_mask, labels


def test_exact_standard_recovery_at_K_zero_and_zero_component_output(batch):
    input_ids, attention_mask, _ = batch
    for K in (0.0, 7.0):
        raw = tiny_model(seed=10)
        baseline = copy.deepcopy(raw).eval()
        hooked, sgcr = attach(raw, K=K)
        hooked.eval()
        assert torch.equal(sgcr.effective_embedding_table(), sgcr.word_embeddings.weight)
        with torch.no_grad():
            baseline_logits = baseline(input_ids=input_ids, attention_mask=attention_mask).logits
            hooked_logits = hooked(input_ids=input_ids, attention_mask=attention_mask).logits
        assert torch.equal(hooked_logits, baseline_logits)


def test_component_path_is_gradient_live_at_exact_initialization(batch):
    input_ids, attention_mask, labels = batch
    model, sgcr = attach(tiny_model(seed=20), K=7.0)
    model.train()
    assert torch.equal(sgcr.effective_embedding_table(), sgcr.word_embeddings.weight)
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()
    # Live revision uses random component codes and a zero projection: the
    # projection, rather than component table, must be live on step one.
    assert sgcr.component_proj.weight.grad.norm().item() > 0.0
    assert sgcr.component_embeddings.weight.grad.norm().item() == 0.0

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.backward()
    assert sgcr.component_embeddings.weight.grad.norm().item() > 0.0


def test_input_and_decoder_use_same_effective_values():
    model, sgcr = attach(tiny_model(seed=30), K=7.0)
    with torch.no_grad():
        sgcr.component_proj.weight.normal_(mean=0.0, std=0.2)
        sgcr.component_proj.bias.normal_(mean=0.0, std=0.1)
        expected_table = sgcr.effective_embedding_table()
        ids = torch.tensor([[5, 8, 13]])
        hidden = torch.randn(2, 3, HIDDEN)
        actual_input = model.deberta.embeddings.word_embeddings(ids)
        actual_output = model.cls.predictions.decoder(hidden)
    assert torch.equal(actual_input, F.embedding(ids, expected_table))
    assert torch.equal(actual_output, F.linear(hidden, expected_table, model.cls.predictions.bias))


def test_baked_checkpoint_is_clean_standard_loadable_and_logit_equivalent(tmp_path, batch):
    input_ids, attention_mask, _ = batch
    model, sgcr = attach(tiny_model(seed=40), K=7.0)
    with torch.no_grad():
        sgcr.component_proj.weight.normal_(mean=0.0, std=0.2)
        sgcr.component_proj.bias.normal_(mean=0.0, std=0.1)
    model.eval()
    with torch.no_grad():
        hooked_before = model(input_ids=input_ids, attention_mask=attention_mask).logits
    info = live.save_baked_checkpoint(model, sgcr, NoOpTokenizer(), tmp_path, also_save_sgcr=True)
    state_keys = sorted(load_file(str(Path(tmp_path) / "model.safetensors")).keys())
    assert info["removed_nonstandard_state_keys"] > 0
    assert not [key for key in state_keys if key.startswith("_sgcr")]

    loaded, loading_info = DebertaV2ForMaskedLM.from_pretrained(tmp_path, output_loading_info=True)
    assert loading_info["missing_keys"] == []
    assert loading_info["unexpected_keys"] == []
    loaded.eval()
    with torch.no_grad():
        baked_logits = loaded(input_ids=input_ids, attention_mask=attention_mask).logits
        hooked_after = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.allclose(hooked_before, baked_logits, atol=1e-6, rtol=1e-6)
    assert torch.equal(hooked_before, hooked_after)


def test_missing_decomposition_is_forced_standard():
    config = live.SGCRConfig(
        vocab_40k=4,
        vocab_16k=3,
        hidden_size=8,
        d_comp=2,
        K=5.0,
        force_standard_ids=set(),
    )
    buffers = live.build_sgcr_buffers(
        {0: [0], 1: [], 2: [1], 3: [2]}, {0: 1, 1: 0, 2: 10, 3: 100}, config
    )
    assert buffers["rho"][1].item() == 1.0
    assert buffers["decomp_lengths"][1].item() == 0


def test_uniform_gate_matches_training_mass_mean():
    config = live.SGCRConfig(
        vocab_40k=3,
        vocab_16k=3,
        hidden_size=8,
        d_comp=2,
        K=50.0,
        uniform_gate=True,
    )
    counts_local = {0: 1, 1: 100, 2: 10000}
    buffers = live.build_sgcr_buffers({0: [0], 1: [1], 2: [2]}, counts_local, config)
    expected_rho = sum(
        n * (n / (n + 50.0)) for n in counts_local.values()
    ) / sum(counts_local.values())
    assert torch.allclose(buffers["rho"], torch.full((3,), expected_rho))


def test_saved_component_state_is_sufficient_to_reconstruct_training_state(tmp_path):
    model, sgcr = attach(tiny_model(seed=50), K=7.0)
    with torch.no_grad():
        sgcr.component_proj.weight.normal_(mean=0.0, std=0.2)
    live.save_baked_checkpoint(model, sgcr, NoOpTokenizer(), tmp_path, also_save_sgcr=True)
    state = torch.load(Path(tmp_path) / "sgcr_components.pt", weights_only=True)
    assert "base_word_embeddings" in state


def test_actual_legal40k_tokenizer_saves_and_loads_portably(tmp_path):
    tokenizer_path = _public_path('experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k')
    tokenizer = base.make_portable_tokenizer(str(tokenizer_path))
    tokenizer.save_pretrained(tmp_path)
    reloaded = AutoTokenizer.from_pretrained(tmp_path, use_fast=True)
    assert len(reloaded) == 40000
    assert reloaded.pad_token_id == tokenizer.pad_token_id
    assert reloaded.mask_token_id == tokenizer.mask_token_id
