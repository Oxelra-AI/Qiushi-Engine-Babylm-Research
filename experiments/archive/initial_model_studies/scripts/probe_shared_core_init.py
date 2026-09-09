#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, "experiments/archive/initial_model_studies/training/scripts")
import babylm_compare_train as tr  # noqa: E402


class Tok:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3

    def __len__(self) -> int:
        return 512


def make_args(variant: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_id="",
        dataset_revision="",
        variant=variant,
        tokenizer="baseline16k",
        max_word_exposure=1000,
        example_pool_words=0,
        checkpoint_words=1000,
        seed=42,
        seq_length=64,
        words_per_example=80,
        batch_size=4,
        learning_rate=5e-4,
        weight_decay=0.1,
        warmup_fraction=0.05,
        lr_total_steps=0,
        shared_core_seed=123,
        extra_init_seed=456,
        train_rng_seed=789,
        n_layer=2,
        n_embd=64,
        n_head=4,
        ffn_mult=4,
        n_experts=4,
        top_k=2,
        router_aux_coef=0.01,
        morph_init_scale=0.02,
        morph_dim=64,
        morph_feature_scale=1.0,
        memory_dim=16,
        memory_dropout=0.0,
        log_every=1,
    )


def build(variant: str):
    a = make_args(variant)
    out = pathlib.Path("experiments/archive/initial_model_studies/staging/shared_init_probe")
    out.mkdir(parents=True, exist_ok=True)
    if a.extra_init_seed >= 0:
        tr.reset_all_rng(a.extra_init_seed)
    model, extra = tr.build_model(a, Tok(), out)
    info = tr.apply_shared_gpt2_core_init(model, a, Tok())
    extra.update(info)
    if a.train_rng_seed >= 0:
        tr.reset_all_rng(a.train_rng_seed)
    return model, extra


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max())


def main() -> None:
    dense, dinfo = build("dense_untied_causal")
    mem, minfo = build("memory_causal")
    nop, ninfo = build("memory_nopersist_causal")
    print("INFO", json.dumps({"dense": dinfo, "memory": minfo, "nopersist": ninfo}, indent=2, default=str))
    assert dinfo["shared_core_init_applied"] and minfo["shared_core_init_applied"] and ninfo["shared_core_init_applied"]

    checks: list[tuple[str, float]] = []
    checks.append(("wte_dense_mem", maxdiff(dense.transformer.wte.weight, mem.wte.weight)))
    checks.append(("wte_mem_nopersist", maxdiff(mem.wte.weight, nop.wte.weight)))
    checks.append(("wpe_dense_mem", maxdiff(dense.transformer.wpe.weight, mem.wpe.weight)))
    checks.append(("ln_f_weight_dense_mem", maxdiff(dense.transformer.ln_f.weight, mem.ln_f.weight)))
    checks.append(("ln_f_bias_dense_mem", maxdiff(dense.transformer.ln_f.bias, mem.ln_f.bias)))
    checks.append(("lm_head_dense_mem", maxdiff(dense.lm_head.weight, mem.lm_head.weight)))
    checks.append(("lm_head_mem_nopersist", maxdiff(mem.lm_head.weight, nop.lm_head.weight)))
    for i in range(2):
        gd = dense.transformer.h[i]
        gm = mem.h[i]
        gn = nop.h[i]
        checks.extend([
            (f"b{i}_ln1_weight_dense_mem", maxdiff(gd.ln_1.weight, gm.ln_1.weight)),
            (f"b{i}_ln1_bias_dense_mem", maxdiff(gd.ln_1.bias, gm.ln_1.bias)),
            (f"b{i}_ln2_weight_dense_mem", maxdiff(gd.ln_2.weight, gm.ln_2.weight)),
            (f"b{i}_ln2_bias_dense_mem", maxdiff(gd.ln_2.bias, gm.ln_2.bias)),
            (f"b{i}_attn_qkv_weight_dense_mem", maxdiff(gd.attn.c_attn.weight.t(), gm.attn.c_attn.weight)),
            (f"b{i}_attn_qkv_bias_dense_mem", maxdiff(gd.attn.c_attn.bias, gm.attn.c_attn.bias)),
            (f"b{i}_attn_proj_weight_dense_mem", maxdiff(gd.attn.c_proj.weight.t(), gm.attn.c_proj.weight)),
            (f"b{i}_attn_proj_bias_dense_mem", maxdiff(gd.attn.c_proj.bias, gm.attn.c_proj.bias)),
            (f"b{i}_ffn_fc_weight_dense_mem", maxdiff(gd.mlp.c_fc.weight.t(), gm.ffn.net[0].weight)),
            (f"b{i}_ffn_fc_bias_dense_mem", maxdiff(gd.mlp.c_fc.bias, gm.ffn.net[0].bias)),
            (f"b{i}_ffn_proj_weight_dense_mem", maxdiff(gd.mlp.c_proj.weight.t(), gm.ffn.net[2].weight)),
            (f"b{i}_ffn_proj_bias_dense_mem", maxdiff(gd.mlp.c_proj.bias, gm.ffn.net[2].bias)),
            (f"b{i}_attn_qkv_mem_nopersist", maxdiff(gm.attn.c_attn.weight, gn.attn.c_attn.weight)),
            (f"b{i}_ffn_fc_mem_nopersist", maxdiff(gm.ffn.net[0].weight, gn.ffn.net[0].weight)),
        ])
    print("CORE_MAX_DIFFS", json.dumps(checks, indent=2))
    assert all(v == 0.0 for _, v in checks), checks

    adapter_checks: list[tuple[str, float]] = []
    for i in range(2):
        adapter_checks.append((f"b{i}_memory_write_mem_nopersist", maxdiff(mem.h[i].memory.write.weight, nop.h[i].memory.write.weight)))
        adapter_checks.append((f"b{i}_memory_gate_mem_nopersist", maxdiff(mem.h[i].memory.write_gate.weight, nop.h[i].memory.write_gate.weight)))
        adapter_checks.append((f"b{i}_memory_read_mem_nopersist", maxdiff(mem.h[i].memory.read.weight, nop.h[i].memory.read.weight)))
        adapter_checks.append((f"b{i}_memory_fuse_mem_nopersist", maxdiff(mem.h[i].memory.fuse.weight, nop.h[i].memory.fuse.weight)))
    print("ADAPTER_MAX_DIFFS", json.dumps(adapter_checks, indent=2))
    assert all(v == 0.0 for _, v in adapter_checks), adapter_checks
    assert mem.config.memory_enabled is True and nop.config.memory_enabled is False

    ids = torch.randint(0, 512, (2, 16))
    forwards = {}
    for name, model in [("dense", dense), ("memory", mem), ("nopersist", nop)]:
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        forwards[name] = {
            "loss": float(out.loss.detach()),
            "memory_stats": getattr(model, "_last_memory_stats", None),
        }
    print("FORWARDS", json.dumps(forwards, indent=2, default=str))
    print("PROBE_OK")


if __name__ == "__main__":
    main()
