#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

from transformers import BertConfig, BertForMaskedLM, DebertaV2Config, DebertaV2ForMaskedLM

OUT = pathlib.Path("experiments/archive/initial_model_studies/data/deberta_param_probe.json")
VOCAB = 16384
BASELINE_PARAMS = 10727168


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def embedding_params(model) -> int:
    # Count all word embedding parameters; used to separate capacity from relative-attention changes.
    total = 0
    for name, p in model.named_parameters():
        if "word_embeddings" in name or "cls.predictions.decoder" in name or "lm_predictions.lm_head.dense" in name:
            pass
    return model.get_input_embeddings().weight.numel()


def build_deberta(hidden: int, layers: int, heads: int, inter: int, rel: bool = True):
    cfg = DebertaV2Config(
        vocab_size=VOCAB,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=inter,
        max_position_embeddings=512,
        max_relative_positions=256,
        position_buckets=256,
        relative_attention=rel,
        pos_att_type=["p2c", "c2p"] if rel else [],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=3,
        bos_token_id=0,
        eos_token_id=2,
    )
    return DebertaV2ForMaskedLM(cfg)


def main() -> None:
    bert_cfg = BertConfig(
        vocab_size=VOCAB,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
        pad_token_id=3,
        bos_token_id=0,
        eos_token_id=2,
    )
    bert = BertForMaskedLM(bert_cfg)
    rows = []
    for hidden in [192, 208, 224, 240, 256, 272, 288]:
        for layers in [6, 7, 8, 9, 10]:
            for inter_mult in [3, 4, 5]:
                inter = hidden * inter_mult
                if hidden % 8 != 0:
                    continue
                # choose heads that divide hidden and keep head dim 32 or 64 where possible
                candidates = [h for h in [4, 6, 7, 8, 10, 12] if hidden % h == 0]
                for heads in candidates:
                    if hidden // heads not in [32, 40, 48, 56, 64, 68, 72]:
                        continue
                    try:
                        model = build_deberta(hidden, layers, heads, inter, True)
                    except Exception as e:
                        rows.append({"hidden": hidden, "layers": layers, "heads": heads, "intermediate": inter, "error": repr(e)})
                        continue
                    params = count_params(model)
                    rows.append({
                        "model": "DebertaV2ForMaskedLM",
                        "hidden": hidden,
                        "layers": layers,
                        "heads": heads,
                        "head_dim": hidden // heads,
                        "intermediate": inter,
                        "params": params,
                        "input_embedding_params": embedding_params(model),
                        "delta_vs_bert_params": params - BASELINE_PARAMS,
                        "pct_vs_bert": 100.0 * (params - BASELINE_PARAMS) / BASELINE_PARAMS,
                        "relative_attention": True,
                        "pos_att_type": ["p2c", "c2p"],
                        "position_buckets": 256,
                        "max_relative_positions": 256,
                    })
    feasible = [r for r in rows if "params" in r and abs(r["pct_vs_bert"]) <= 5.0]
    feasible_sorted = sorted(feasible, key=lambda r: (abs(r["pct_vs_bert"]), r["layers"], r["hidden"]))[:30]
    payload = {
        "baseline": {
            "model": "BertForMaskedLM",
            "hidden": 256,
            "layers": 8,
            "heads": 8,
            "intermediate": 1024,
            "params_from_existing_runs": BASELINE_PARAMS,
            "params_instantiated_here": count_params(bert),
            "input_embedding_params": embedding_params(bert),
        },
        "feasible_within_5pct_sorted": feasible_sorted,
        "all_rows_count": len(rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2)[:8000])
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
