#!/usr/bin/env python3
from __future__ import annotations

import json
from transformers import BertConfig, BertForMaskedLM, DebertaV2Config, DebertaV2ForMaskedLM

VOCAB = 16384
PAD = 0
BOS = 1
EOS = 2
MAX_POS = 512


def count(model):
    emb = model.get_input_embeddings().weight.numel()
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "embedding": emb,
        "non_embedding": sum(p.numel() for p in model.parameters()) - emb,
    }


def bert(L, d, heads, ffn_mult=4):
    cfg = BertConfig(
        vocab_size=VOCAB, hidden_size=d, num_hidden_layers=L, num_attention_heads=heads,
        intermediate_size=d * ffn_mult, max_position_embeddings=MAX_POS,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS,
        position_embedding_type="absolute",
    )
    return BertForMaskedLM(cfg)


def deberta(L, d, heads, ffn_mult=4, rel=True, pos="p2c,c2p"):
    cfg = DebertaV2Config(
        vocab_size=VOCAB, hidden_size=d, num_hidden_layers=L, num_attention_heads=heads,
        intermediate_size=d * ffn_mult, max_position_embeddings=MAX_POS,
        max_relative_positions=256, position_buckets=256,
        relative_attention=rel, pos_att_type=[x for x in pos.split(',') if x],
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS,
    )
    return DebertaV2ForMaskedLM(cfg)


def main():
    rows=[]
    for L in [6,8,10,12]:
        for d in [320,352,384,416,448,480,512]:
            for h in [5,6,7,8,10,12,16]:
                if d % h != 0:
                    continue
                for typ, fn in [("bert", bert), ("deberta_v2", deberta)]:
                    try:
                        c=count(fn(L,d,h))
                    except Exception as e:
                        continue
                    rows.append({"model_type":typ,"layers":L,"hidden":d,"heads":h,"head_dim":d//h,"ffn_mult":4,**c,"abs_diff_from_34_7M":abs(c["total"]-34_700_000)})
    rows=sorted(rows, key=lambda r:(r["abs_diff_from_34_7M"], r["model_type"], r["layers"], r["hidden"]))
    print(json.dumps(rows[:80], indent=2))

if __name__ == "__main__":
    main()
