#!/usr/bin/env python3
"""research: make persistent S1-depth WESS learning seed-reliable.

Builds on research's independent-template trajectory test. Tests whether a modest
positive residual initialization plus faster slot-path optimization makes the
persistent regime reliable, and whether it survives annealing synthetic episodes
from 50% to 25%.

All scientific measurements use held-out verbs/query phrasing never used for training.
"""
from __future__ import annotations
import importlib.util, json, pathlib, random, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
SRC=ROOT/'scripts/wess_dynamics.py'
OUT=ROOT/'data/wess_reliability_schedule.json'
spec=importlib.util.spec_from_file_location('dyn',SRC)
dyn=importlib.util.module_from_spec(spec); sys.modules[spec.name]=dyn; spec.loader.exec_module(dyn)
bridge=dyn.bridge


def set_gate(model, value:float):
    with torch.no_grad(): model.fusion_gate.fill_(value)


def parameter_groups(model, enc_lr:float, slot_lr:float):
    enc=[]; slot=[]
    for name,p in model.named_parameters():
        if name.startswith('enc.') or name=='bias': enc.append(p)
        else: slot.append(p)
    return [{'params':enc,'lr':enc_lr},{'params':slot,'lr':slot_lr}]


def ratio_at(step:int, steps:int, mode:str):
    if mode=='fixed50': return 0.50
    if mode=='anneal50to25':
        # Strong binding phase through 60%; then linearly reduce to 25% by 85%, hold.
        a=int(0.60*steps); b=int(0.85*steps)
        if step<a: return 0.50
        if step>=b: return 0.25
        return 0.50-0.25*(step-a)/max(1,b-a)
    raise ValueError(mode)


@torch.no_grad()
def official_loss(tok,model,official,device,n=256):
    model.eval(); vals=[]
    for i in range(0,min(n,len(official)),32):
        items=official[i:i+32]
        x,a,l,_,eps=bridge.pad_features(items,tok.pad_token_id,device)
        _,loss=model(x,a,l,eps); vals.append(float(loss))
    model.train(); return sum(vals)/len(vals)


def run(tok,train_eps,heldout,official,cfg,seed,device):
    torch.manual_seed(seed); random.seed(seed)
    model=dyn.GatedWESS(tok,'wess_gold').to(device)
    set_gate(model,cfg['gate_init'])
    groups=parameter_groups(model,cfg['encoder_lr'],cfg['slot_lr'])
    opt=torch.optim.AdamW(groups,weight_decay=0.01)
    steps=cfg['steps']; warm=max(1,int(0.05*steps))
    # Scheduler scales each group's own initial LR.
    sched=get_cosine_schedule_with_warmup(opt,warm,steps)
    rng=random.Random(seed+31); traj=[]; losses=[]
    for step in range(steps):
        ratio=ratio_at(step,steps,cfg['schedule'])
        n_ep=max(1,min(31,round(32*ratio)))
        items=[rng.choice(train_eps) for _ in range(n_ep)]+[rng.choice(official) for _ in range(32-n_ep)]
        rng.shuffle(items)
        x,a,l,_,eps=bridge.pad_features(items,tok.pad_token_id,device)
        _,loss=model(x,a,l,eps)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        losses.append(float(loss.detach()))
        if (step+1)%250==0 or step==steps-1:
            p=dyn.probe(tok,model,heldout,device)
            p.update({'step':step+1,'gate':float(model.fusion_gate.detach()),
                      'episode_ratio':ratio,'train_loss':losses[-1],
                      'official_mlm_loss':official_loss(tok,model,official,device,n=128)})
            traj.append(p)
            print(f"  {cfg['name']} seed{seed} s{step+1}: pa={p['pair_acc']:.3f} lo={p['mean_logodds']:.2f} gate={p['gate']:.3f} ratio={ratio:.2f} off={p['official_mlm_loss']:.3f}",flush=True)
    final=dyn.probe(tok,model,heldout,device)
    final['interventions']=bridge.eval_interventions(tok,model,heldout[:120],device)
    final['learned_gate']=float(model.fusion_gate.detach())
    final['official_mlm_loss']=official_loss(tok,model,official,device,n=256)
    final['loss_mean_last20']=sum(losses[-20:])/len(losses[-20:])
    return {'trajectory':traj,'final':final}


def main():
    bridge.setup_env(); t0=time.time(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(str(bridge.TOK_PATH),use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.mask_token
    train=dyn.build_pairs_tmpl(tok,1200,4210,dyn.TRAIN_TEMPLATES,dyn.TRAIN_QUERY)
    held=dyn.build_pairs_tmpl(tok,200,9990,dyn.HELDOUT_TEMPLATES,dyn.HELDOUT_QUERY)
    official=dyn.load_official(tok,5000,4230)
    audit=bridge.shortcut_audit(held)
    configs=[
      {'name':'fixed50_sepLR_gate003','schedule':'fixed50','steps':1500,'encoder_lr':2e-4,'slot_lr':1e-3,'gate_init':0.03},
      {'name':'anneal50to25_sepLR_gate003','schedule':'anneal50to25','steps':2000,'encoder_lr':2e-4,'slot_lr':1e-3,'gate_init':0.03},
    ]
    seeds=[42,123,456,789]; results={}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    for cfg in configs:
        results[cfg['name']]={}
        for seed in seeds:
            print(f"\n=== {cfg['name']} seed {seed} ===",flush=True)
            results[cfg['name']][str(seed)]=run(tok,train,held,official,cfg,seed,device)
            partial={'status':'PARTIAL','configs':configs,'seeds':seeds,'heldout_audit':audit,'results':results,'elapsed_sec':round(time.time()-t0,1)}
            OUT.write_text(json.dumps(partial,indent=2)+'\n')
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    payload={'status':'COMPLETE','description':'S1-depth WESS reliability across four seeds on independent templates: separate encoder/slot LR, nonzero gate initialization, fixed50 versus anneal50to25.','configs':configs,'seeds':seeds,'heldout_audit':audit,'results':results,'elapsed_sec':round(time.time()-t0,1)}
    OUT.write_text(json.dumps(payload,indent=2)+'\n')
    print(f'\nSaved {OUT}',flush=True)

if __name__=='__main__': main()
