#!/usr/bin/env python3
"""research: WESS-to-base distillation screen.

Problem from research-194: WESS-on annotated inference is perfect, but the exported
WESS-disabled base has zero controlled binding. This script tests whether an explicit
base-transfer loss can move binding into the ordinary DeBERTa MLM backbone.

Method:
- Same 384x4 smoke scale and same independent-template heldout as research.
- wess_aux_distill trains WESS-on CE loss plus a base-only CE loss on the same synthetic
  binding targets (and optional KL from detached WESS-on distribution to base-only logits).
- Exports and evaluates the plain HF backbone with WESS disabled.

This is not an official candidate; it is a mechanism transfer test.
"""
from __future__ import annotations
import importlib.util, json, pathlib, random, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
SRC=ROOT/'scripts/wess_aux_base_transfer_screen.py'
OUT=ROOT/'data/wess_to_base_distill_screen.json'
RUN=ROOT/'training/runs/wess_to_base_distill_screen'

spec=importlib.util.spec_from_file_location('screen',SRC)
screen=importlib.util.module_from_spec(spec); sys.modules[spec.name]=screen; spec.loader.exec_module(screen)
bridge=screen.bridge; dyn=screen.dyn

class Args:
    hidden_size=384; n_layer=4; n_head=12; intermediate_size=1280
    steps=1000; batch_size=32; episode_ratio=0.5; anneal=True
    encoder_lr=2e-4; slot_lr=1e-3; gate_init=0.03; probe_every=250; seed=42
    train_pairs=1200; eval_pairs=200; official_examples=5000; run_name='distill_384x4_anneal'
    base_ce_weight=1.0; kl_weight=0.25; temperature=2.0


def synthetic_mask_indices(eps, device):
    idx=[]
    for b,ep in enumerate(eps):
        if ep is not None: idx.append((b, ep.mask_pos))
    if not idx:
        return None
    return torch.tensor(idx, dtype=torch.long, device=device)


def train_distill(tok,args,train_eps,official,device):
    torch.manual_seed(args.seed); random.seed(args.seed)
    model=screen.AuxWESSForMLM(tok,'wess_aux',hidden=args.hidden_size,layers=args.n_layer,heads=args.n_head,intermediate=args.intermediate_size,gate_init=args.gate_init).to(device)
    opt=torch.optim.AdamW(screen.param_groups(model,args.encoder_lr,args.slot_lr),weight_decay=0.01)
    sched=get_cosine_schedule_with_warmup(opt,max(1,int(args.steps*0.05)),args.steps)
    rng=random.Random(args.seed+71); meta=[]
    for step in range(args.steps):
        ratio=screen.ratio_at(step,args.steps) if args.anneal else args.episode_ratio
        n_ep=round(args.batch_size*ratio)
        items=[rng.choice(train_eps) for _ in range(n_ep)]+[rng.choice(official) for _ in range(args.batch_size-n_ep)]
        rng.shuffle(items)
        x,a,l,_,eps=bridge.pad_features(items,tok.pad_token_id,device)
        logits_w,loss_w=model(x,a,l,eps,wess_on=True)
        logits_b,loss_b_all=model(x,a,l,eps,wess_on=False)
        loss=loss_w
        idx=synthetic_mask_indices(eps,device)
        base_ce=None; kl=None
        if idx is not None:
            bidx=idx[:,0]; midx=idx[:,1]
            labels=l[bidx,midx]
            base_logits=logits_b[bidx,midx]
            wess_logits=logits_w[bidx,midx].detach()
            base_ce=F.cross_entropy(base_logits,labels)
            T=args.temperature
            kl=F.kl_div(F.log_softmax(base_logits/T,dim=-1),F.softmax(wess_logits/T,dim=-1),reduction='batchmean')*(T*T)
            loss=loss + args.base_ce_weight*base_ce + args.kl_weight*kl
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        if (step+1)%args.probe_every==0 or step==args.steps-1:
            rec={'step':step+1,'ratio':ratio,'loss_total':float(loss.detach()),'loss_wess':float(loss_w.detach()),'loss_base_all':float(loss_b_all.detach()),'base_ce_syn':float(base_ce.detach()) if base_ce is not None else None,'kl_syn':float(kl.detach()) if kl is not None else None,'gate':float(model.fusion_gate.detach())}
            meta.append(rec)
            print(json.dumps({'event':'train','rec':rec}),flush=True)
    return model,meta


def main():
    args=Args(); t0=time.time(); bridge.setup_env(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(str(bridge.TOK_PATH),use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.mask_token
    train_eps=dyn.build_pairs_tmpl(tok,args.train_pairs,args.seed*10+1,dyn.TRAIN_TEMPLATES,dyn.TRAIN_QUERY)
    eval_eps=dyn.build_pairs_tmpl(tok,args.eval_pairs,9990,dyn.HELDOUT_TEMPLATES,dyn.HELDOUT_QUERY)
    official=screen.load_official(tok,args.official_examples,args.seed*10+3)
    audit=bridge.shortcut_audit(eval_eps)
    model,meta=train_distill(tok,args,train_eps,official,device)
    rec={'train_meta':meta,
         'official_loss_base_only':screen.eval_official_loss(tok,model,official,device,wess_on=False),
         'synthetic_base_only':screen.eval_binding(tok,model,eval_eps,device,wess_on=False),
         'synthetic_wess_on':screen.eval_binding(tok,model,eval_eps,device,wess_on=True),
         'interventions_wess_on':screen.eval_interventions(tok,model,eval_eps[:120],device),
         'params_plain_hf':sum(p.numel() for p in model.mlm.parameters()),
         'params_with_wess':sum(p.numel() for p in model.parameters())}
    export_dir=RUN/'distill_384x4_anneal/hf_model'
    model.export_plain(tok,export_dir); rec['exported_plain_hf_model']=str(export_dir)
    payload={'status':'WESS_TO_BASE_DISTILL_SCREEN','description':'Explicitly trains base-only logits on synthetic binding targets alongside WESS-on loss to test transfer into official-compatible backbone.','args':{k:v for k,v in Args.__dict__.items() if not k.startswith('_') and not callable(v)},'device':str(device),'shortcut_audit':audit,'distill':rec,'elapsed_sec':round(time.time()-t0,1)}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2)+'\n')
    (RUN/'distill_384x4_anneal/screen_result.json').parent.mkdir(parents=True,exist_ok=True)
    (RUN/'distill_384x4_anneal/screen_result.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'out':str(OUT),'export':str(export_dir),'summary':rec},indent=2)[:5000],flush=True)

if __name__=='__main__': main()
