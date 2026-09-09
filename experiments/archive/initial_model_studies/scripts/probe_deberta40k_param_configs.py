#!/usr/bin/env python3
from __future__ import annotations
import json
from transformers import DebertaV2Config, DebertaV2ForMaskedLM

VOCAB=40000
PAD=0; BOS=1; EOS=2; MAX_POS=512

def count(L,d,h,ffn_mult=4):
    cfg=DebertaV2Config(vocab_size=VOCAB, hidden_size=d, num_hidden_layers=L, num_attention_heads=h, intermediate_size=d*ffn_mult, max_position_embeddings=MAX_POS, max_relative_positions=256, position_buckets=256, relative_attention=True, pos_att_type=["p2c","c2p"], hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS)
    m=DebertaV2ForMaskedLM(cfg)
    emb=m.get_input_embeddings().weight.numel()
    return {"total":sum(p.numel() for p in m.parameters()), "embedding":emb, "non_embedding":sum(p.numel() for p in m.parameters())-emb}

def main():
    rows=[]
    for L in [8]:
        for d in range(320, 481, 16):
            for h in [5,6,8,10,12,16]:
                if d%h: continue
                c=count(L,d,h)
                rows.append({"layers":L,"hidden":d,"heads":h,"head_dim":d//h,**c,"diff_total_from_34_467_424":c['total']-34467424,"diff_nonemb_from_26_603_104":c['non_embedding']-26603104})
    rows=sorted(rows, key=lambda r:(abs(r['diff_total_from_34_467_424']), abs(r['diff_nonemb_from_26_603_104'])))
    print(json.dumps(rows[:60], indent=2))
if __name__=='__main__': main()
