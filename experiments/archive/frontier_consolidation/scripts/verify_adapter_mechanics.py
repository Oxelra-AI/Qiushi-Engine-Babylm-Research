#!/usr/bin/env python3
"""Mechanical verification for research residual adapters; no corpus training."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, os, random, shutil, sys
from pathlib import Path
import numpy as np
import torch

# Writable HF dynamic-module cache
os.environ["HF_MODULES_CACHE"] = str(_public_path('data/external/hf_modules_cache'))
from transformers import AutoModelForMaskedLM, DebertaV2Config, DebertaV2ForMaskedLM

HERE = _public_path('experiments/archive/frontier_consolidation/scripts')
ROOT = _public_path('.')
sys.path.insert(0, str(HERE))
from adapter_modeling import AdapterDebertaV2ForMaskedLM

OUT = _public_path('experiments/archive/frontier_consolidation/data/adapter_mechanics')


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def config():
    c = DebertaV2Config(vocab_size=16384, hidden_size=480, num_hidden_layers=8,
        num_attention_heads=8, intermediate_size=1920, max_position_embeddings=512,
        max_relative_positions=256, position_buckets=256, relative_attention=True,
        pos_att_type=["p2c","c2p"], hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    c.adapter_bottleneck=64; c.adapter_activation="gelu"; c.adapter_enabled=True
    return c


def maxdiff(a,b): return float((a.detach()-b.detach()).abs().max().cpu())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(43022); base=DebertaV2ForMaskedLM(config()).to(device)
    seed_all(43022); live=AdapterDebertaV2ForMaskedLM(config()).to(device)
    stock_live={n:p for n,p in live.named_parameters() if ".adapter." not in n}
    stock_base=dict(base.named_parameters())
    name_equal=set(stock_base)==set(stock_live)
    param_max=max(maxdiff(stock_base[n],stock_live[n]) for n in stock_base)

    g=torch.Generator().manual_seed(777)
    ids=torch.randint(5,16384,(2,32),generator=g).to(device)
    attn=torch.ones_like(ids); labels=torch.full_like(ids,-100); labels[:,5:10]=ids[:,5:10]
    masked=ids.clone(); masked[:,5:10]=3

    base.eval(); live.eval()
    with torch.no_grad():
        bo=base(input_ids=masked,attention_mask=attn,labels=labels)
        lo=live(input_ids=masked,attention_mask=attn,labels=labels)
    eval_logit=maxdiff(bo.logits,lo.logits); eval_loss=abs(float(bo.loss-lo.loss))

    base.train(); live.train(); seed_all(9001)
    bo=base(input_ids=masked,attention_mask=attn,labels=labels); seed_all(9001)
    lo=live(input_ids=masked,attention_mask=attn,labels=labels)
    train_logit=maxdiff(bo.logits,lo.logits); train_loss=abs(float(bo.loss-lo.loss))
    base.zero_grad(); live.zero_grad(); bo.loss.backward(); lo.loss.backward()
    grad_max=max(maxdiff(stock_base[n].grad,stock_live[n].grad) for n in stock_base if stock_base[n].grad is not None)
    adapter_grads={n: (None if p.grad is None else float(p.grad.float().norm().cpu())) for n,p in live.named_parameters() if ".adapter." in n}
    first_up=sum(v or 0 for n,v in adapter_grads.items() if ".up." in n)
    first_down=sum(v or 0 for n,v in adapter_grads.items() if ".down." in n)

    # Two deterministic AdamW steps: first recruits W_up, second propagates into down/LN.
    seed_all(43023); live2=AdapterDebertaV2ForMaskedLM(config()).to(device).train()
    normal=[]; zero=[]
    for n,p in live2.named_parameters(): (zero if ".adapter.up." in n else normal).append(p)
    opt=torch.optim.AdamW([{"params":normal,"weight_decay":.01},{"params":zero,"weight_decay":0.}],lr=1e-3,betas=(.9,.98))
    step_records=[]
    for step in range(1,3):
        seed_all(9100+step); opt.zero_grad(set_to_none=True)
        out=live2(input_ids=masked,attention_mask=attn,labels=labels); out.loss.backward()
        up=sum(float(p.grad.float().norm()) for n,p in live2.named_parameters() if ".adapter.up." in n and p.grad is not None)
        down=sum(float(p.grad.float().norm()) for n,p in live2.named_parameters() if ".adapter.down." in n and p.grad is not None)
        torch.nn.utils.clip_grad_norm_(live2.parameters(),1.0); opt.step()
        live2.eval()
        with torch.no_grad(): live2(input_ids=masked,attention_mask=attn)
        rms=live2.adapter_rms(); live2.train()
        step_records.append({"step":step,"loss":float(out.loss),"up_grad_norm_sum":up,"down_grad_norm_sum":down,"adapter_rms_mean":sum(rms)/len(rms),"adapter_rms_max":max(rms)})

    # Native HF custom-code round trip.
    save=_public_path('experiments/archive/frontier_consolidation/data/adapter_mechanics/hf_roundtrip'); shutil.rmtree(save,ignore_errors=True)
    live2.register_for_auto_class("AutoModelForMaskedLM"); live2.save_pretrained(save,safe_serialization=True)
    # Ensure the modeling source file lives in the checkpoint directory
    src_file = _public_path('experiments/archive/frontier_consolidation/scripts/adapter_modeling.py')
    dst_file = save / "adapter_modeling.py"
    if not dst_file.exists():
        shutil.copy2(src_file, dst_file)
    reloaded=AutoModelForMaskedLM.from_pretrained(save,trust_remote_code=True).to(device).eval()
    live2.eval()
    with torch.no_grad():
        a=live2(input_ids=masked,attention_mask=attn).logits
        b=reloaded(input_ids=masked,attention_mask=attn).logits
    roundtrip=maxdiff(a,b)
    result={"status":"ADAPTER_MECHANICS","device":str(device),
      "stock_name_sets_equal":name_equal,"stock_parameter_max_abs_diff":param_max,
      "eval_logit_max_abs_diff":eval_logit,"eval_loss_abs_diff":eval_loss,
      "train_logit_max_abs_diff":train_logit,"train_loss_abs_diff":train_loss,
      "stock_gradient_max_abs_diff":grad_max,
      "first_backward_adapter_up_grad_norm_sum":first_up,
      "first_backward_adapter_down_grad_norm_sum":first_down,
      "two_step_recruitment":step_records,"roundtrip_logit_max_abs_diff":roundtrip,
      "reloaded_class":reloaded.__class__.__name__,
      "parameter_counts":{"base":sum(p.numel() for p in base.parameters()),"adapter":sum(p.numel() for p in live.parameters()),"added":sum(p.numel() for p in live.parameters())-sum(p.numel() for p in base.parameters())}}
    passed=(name_equal and param_max==0 and eval_logit==0 and eval_loss==0 and train_logit==0 and train_loss==0 and grad_max==0 and first_up>0 and first_down==0 and step_records[1]["down_grad_norm_sum"]>0 and roundtrip==0)
    result["passed"]=passed
    (_public_path('experiments/archive/frontier_consolidation/data/adapter_mechanics/adapter_mechanics.json')).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not passed: raise SystemExit(2)

if __name__=="__main__": main()
