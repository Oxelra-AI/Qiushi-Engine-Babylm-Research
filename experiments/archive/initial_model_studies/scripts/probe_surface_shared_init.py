#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, "experiments/archive/initial_model_studies/training/scripts")
import babylm_compare_train as tr  # noqa: E402


class TinyTok:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3

    def __len__(self) -> int:
        return 512

    def convert_ids_to_tokens(self, idx: int) -> str:
        base = ["<pad>", "<s>", "</s>", "Ġthe", "ĠAlice", "Ġwalked", "ing", "Ġcat", "s", "ĠBob"]
        if idx < len(base):
            return base[idx]
        return f"Ġtok{idx}"


def make_args(variant: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_id="", dataset_revision="", variant=variant, tokenizer="baseline16k",
        max_word_exposure=1000, example_pool_words=1000, order_mode="random", checkpoint_words=1000,
        seed=42, seq_length=64, words_per_example=80, batch_size=4, learning_rate=5e-4,
        weight_decay=0.1, warmup_fraction=0.05, lr_total_steps=0,
        shared_core_seed=123, extra_init_seed=456, train_rng_seed=789,
        n_layer=2, n_embd=64, n_head=4, ffn_mult=4,
        n_experts=4, top_k=2, router_aux_coef=0.01,
        morph_init_scale=0.02, morph_dim=64, morph_feature_scale=1.0,
        memory_dim=16, memory_dropout=0.0,
        surface_dim=32, surface_feature_scale=1.0, ngram_vocab_size=128, max_ngrams_per_token=12, lookup_rank=2,
        log_every=1,
    )


def build(variant: str, root: pathlib.Path):
    args = make_args(variant)
    out = root / variant
    out.mkdir(parents=True, exist_ok=True)
    tr.reset_all_rng(args.extra_init_seed)
    model, extra = tr.build_model(args, TinyTok(), out)
    info = tr.apply_shared_gpt2_core_init(model, args, TinyTok())
    extra.update(info)
    return model, extra, out


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max())


def check_core_equal(dense, surf, label: str) -> list[tuple[str, float]]:
    checks = [
        (f"{label}_wte", maxdiff(dense.transformer.wte.weight, surf.wte.weight)),
        (f"{label}_wpe", maxdiff(dense.transformer.wpe.weight, surf.wpe.weight)),
        (f"{label}_ln_f_w", maxdiff(dense.transformer.ln_f.weight, surf.ln_f.weight)),
        (f"{label}_ln_f_b", maxdiff(dense.transformer.ln_f.bias, surf.ln_f.bias)),
        (f"{label}_lm_head", maxdiff(dense.lm_head.weight, surf.lm_head.weight)),
    ]
    for i in range(2):
        gd = dense.transformer.h[i]
        gs = surf.h[i]
        checks.extend([
            (f"{label}_b{i}_ln1_w", maxdiff(gd.ln_1.weight, gs.ln_1.weight)),
            (f"{label}_b{i}_ln1_b", maxdiff(gd.ln_1.bias, gs.ln_1.bias)),
            (f"{label}_b{i}_ln2_w", maxdiff(gd.ln_2.weight, gs.ln_2.weight)),
            (f"{label}_b{i}_ln2_b", maxdiff(gd.ln_2.bias, gs.ln_2.bias)),
            (f"{label}_b{i}_attn_qkv_w", maxdiff(gd.attn.c_attn.weight.t(), gs.attn.c_attn.weight)),
            (f"{label}_b{i}_attn_qkv_b", maxdiff(gd.attn.c_attn.bias, gs.attn.c_attn.bias)),
            (f"{label}_b{i}_attn_proj_w", maxdiff(gd.attn.c_proj.weight.t(), gs.attn.c_proj.weight)),
            (f"{label}_b{i}_attn_proj_b", maxdiff(gd.attn.c_proj.bias, gs.attn.c_proj.bias)),
            (f"{label}_b{i}_ffn_fc_w", maxdiff(gd.mlp.c_fc.weight.t(), gs.ffn.net[0].weight)),
            (f"{label}_b{i}_ffn_fc_b", maxdiff(gd.mlp.c_fc.bias, gs.ffn.net[0].bias)),
            (f"{label}_b{i}_ffn_proj_w", maxdiff(gd.mlp.c_proj.weight.t(), gs.ffn.net[2].weight)),
            (f"{label}_b{i}_ffn_proj_b", maxdiff(gd.mlp.c_proj.bias, gs.ffn.net[2].bias)),
        ])
    return checks


def save_reload(model, out: pathlib.Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model.save_pretrained(out, safe_serialization=True)
    tr.copy_custom_code_if_needed(model, out)
    loaded = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True)
    ids = torch.randint(0, model.config.vocab_size, (2, 16))
    y = loaded(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
    assert y.loss is not None


def main() -> None:
    root = pathlib.Path("experiments/archive/initial_model_studies/staging/surface_probe")
    root.mkdir(parents=True, exist_ok=True)
    dense, dense_info, _ = build("dense_untied_causal", root)
    char, char_info, char_dir = build("char_surface_causal", root)
    lookup, lookup_info, lookup_dir = build("lookup_adapter_causal", root)
    checks = check_core_equal(dense, char, "char") + check_core_equal(dense, lookup, "lookup")
    assert all(v == 0.0 for _, v in checks), checks
    assert int(char.surface.surface_ngram_mask.sum().item()) > 0
    assert int((char.surface.surface_ngram_ids > 0).sum().item()) > 0
    assert hasattr(char.surface, "ngram_embedding") and char.surface.ngram_embedding.weight.requires_grad
    assert hasattr(lookup.surface, "lookup") and lookup.surface.lookup.weight.requires_grad
    ids = torch.randint(0, 512, (2, 16))
    forward = {}
    for name, model in [("dense", dense), ("char", char), ("lookup", lookup)]:
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        forward[name] = {"loss": float(out.loss.detach()), "surface": getattr(model, "_last_surface_stats", None)}
    save_reload(char, root / "char_reload")
    save_reload(lookup, root / "lookup_reload")
    report = {
        "dense_info": dense_info,
        "char_info": {k: v for k, v in char_info.items() if k != "surface_manifest"},
        "lookup_info": {k: v for k, v in lookup_info.items() if k != "surface_manifest"},
        "char_manifest_unique_ids": char_info["surface_manifest"]["unique_hashed_ngram_ids_used"],
        "core_max_diffs": checks,
        "forward": forward,
        "char_params": sum(p.numel() for p in char.parameters()),
        "lookup_params": sum(p.numel() for p in lookup.parameters()),
        "dense_params": sum(p.numel() for p in dense.parameters()),
    }
    path = root / "probe_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("SURFACE_PROBE_OK", path)


if __name__ == "__main__":
    main()
