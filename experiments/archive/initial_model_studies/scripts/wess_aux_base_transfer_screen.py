#!/usr/bin/env python3
"""research: WESS auxiliary base-transfer screen.

Transfer-scope distinction:
- WESS can use annotated synthetic spans during auxiliary training/probing.
- Official BabyLM inputs are unlabeled, so the immediate official-compatible artifact must
  be a plain HuggingFace DebertaV2ForMaskedLM with WESS disabled.

This script trains matched models on the same official+synthetic stream:
1. plain_mixed: DeBERTa MLM on mixed stream; no WESS.
2. wess_aux_anneal: DeBERTa MLM + WESS residual only on annotated synthetic episodes,
   with reliable research schedule (50% warm-start annealed to 25%, separate LRs, gate init .03).

It reports:
- synthetic WESS-on heldout binding (annotation-dependent mechanism probe),
- base-only heldout binding with WESS disabled (checks transfer into ordinary weights),
- official-text MLM loss with WESS disabled,
- exported plain HF checkpoints for official evaluator compatibility.

This is a bounded screen; not a 100M candidate.
"""
from __future__ import annotations
import argparse, importlib.util, json, pathlib, random, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
DYN=ROOT/'scripts/wess_dynamics.py'
OUTROOT=ROOT/'training/runs/wess_aux_base_transfer_screen'
DATAOUT=ROOT/'data/wess_aux_base_transfer_screen.json'

spec=importlib.util.spec_from_file_location('dyn',DYN)
dyn=importlib.util.module_from_spec(spec); sys.modules[spec.name]=dyn; spec.loader.exec_module(dyn)
bridge=dyn.bridge


def force_portable_tokenizer_config(dst:pathlib.Path):
    p=dst/'tokenizer_config.json'
    if p.exists():
        cfg=json.loads(p.read_text())
        cfg['tokenizer_class']='PreTrainedTokenizerFast'
        p.write_text(json.dumps(cfg,indent=2)+'\n')


def ratio_at(step:int, steps:int):
    a=int(0.60*steps); b=int(0.85*steps)
    if step<a: return 0.50
    if step>=b: return 0.25
    return 0.50-0.25*(step-a)/max(1,b-a)


def load_official(tok,n,seed,max_len=192):
    return dyn.load_official(tok,n,seed,max_len=max_len)


class AuxWESSForMLM(nn.Module):
    def __init__(self,tok,arm,hidden=384,layers=4,heads=12,intermediate=1280,gate_init=0.03):
        super().__init__(); self.arm=arm; self.hidden=hidden; self.tok=tok
        cfg=DebertaV2Config(vocab_size=len(tok),hidden_size=hidden,num_hidden_layers=layers,
            num_attention_heads=heads,intermediate_size=intermediate,max_position_embeddings=512,
            position_buckets=256,relative_attention=True,pos_att_type=['p2c','c2p'],pad_token_id=tok.pad_token_id)
        self.mlm=DebertaV2ForMaskedLM(cfg)
        self.slot_init=nn.Linear(hidden,hidden)
        self.event_mlp=nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.gru=nn.GRUCell(hidden,hidden)
        self.slot_proj=nn.Linear(hidden,hidden)
        self.fusion_gate=nn.Parameter(torch.tensor(gate_init))
    def route(self,ei):
        return 0 if self.arm=='no_address' else ei
    def run_memory(self,h,ep,swap=False,ablate=False):
        if self.arm in {'plain_mixed','base_only'}: return None
        ent0=h[ep.event_entity_pos[0]]; ent1=None
        for idx,(ei,_) in enumerate(ep.events):
            if ei==1: ent1=h[ep.event_entity_pos[idx]]; break
        if ent1 is None: ent1=ent0
        sl=[self.slot_init(ent0),self.slot_init(ent1)]
        if self.arm=='no_address': sl[1]=sl[0]
        last_q=max(i for i,(ei,_) in enumerate(ep.events) if ei==ep.query_entity)
        for t,(ei,_) in enumerate(ep.events):
            if ablate and t==last_q: continue
            upd=self.event_mlp(torch.cat([h[ep.event_entity_pos[t]],h[ep.event_state_pos[t]]]))
            si=self.route(ei)
            sl[si]=self.gru(upd.unsqueeze(0),sl[si].unsqueeze(0)).squeeze(0)
            if self.arm=='no_address': sl[1]=sl[0]
        if swap and self.arm=='wess_aux': sl[0],sl[1]=sl[1],sl[0]
        return sl[ep.query_entity if self.arm!='no_address' else 0]
    def forward(self,input_ids,attention_mask,labels=None,eps=None,wess_on=True,intervention=None):
        h=self.mlm.deberta(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        if wess_on and eps is not None and self.arm in {'wess_aux','no_address'}:
            h_new=h.clone()
            for b,ep in enumerate(eps):
                if ep is None: continue
                slot=self.run_memory(h[b],ep,swap=(intervention=='swap'),ablate=(intervention=='ablate'))
                if slot is not None:
                    m=ep.mask_pos
                    h_new=h_new.clone(); h_new[b,m]=h[b,m]+self.fusion_gate*self.slot_proj(slot)
            h=h_new
        logits=self.mlm.cls(h)
        loss=None
        if labels is not None:
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100)
        return logits,loss
    def export_plain(self,tok,dst:pathlib.Path):
        dst.mkdir(parents=True,exist_ok=True)
        self.mlm.save_pretrained(dst,safe_serialization=True)
        tok.save_pretrained(dst)
        force_portable_tokenizer_config(dst)


def param_groups(model,enc_lr,slot_lr):
    enc=[]; slot=[]
    for n,p in model.named_parameters():
        if n.startswith('mlm.'):
            enc.append(p)
        else:
            slot.append(p)
    return [{'params':enc,'lr':enc_lr},{'params':slot,'lr':slot_lr}]


def train(tok,arm,args,train_eps,official,device):
    torch.manual_seed(args.seed); random.seed(args.seed)
    model=AuxWESSForMLM(tok,arm,hidden=args.hidden_size,layers=args.n_layer,heads=args.n_head,intermediate=args.intermediate_size,gate_init=args.gate_init).to(device)
    if arm=='plain_mixed':
        opt=torch.optim.AdamW(model.mlm.parameters(),lr=args.encoder_lr,weight_decay=0.01)
    else:
        opt=torch.optim.AdamW(param_groups(model,args.encoder_lr,args.slot_lr),weight_decay=0.01)
    sched=get_cosine_schedule_with_warmup(opt,max(1,int(args.steps*0.05)),args.steps)
    rng=random.Random(args.seed+51); meta=[]
    for step in range(args.steps):
        ratio=ratio_at(step,args.steps) if args.anneal else args.episode_ratio
        n_ep=round(args.batch_size*ratio)
        items=[rng.choice(train_eps) for _ in range(n_ep)]+[rng.choice(official) for _ in range(args.batch_size-n_ep)]
        rng.shuffle(items)
        x,a,l,_,eps=bridge.pad_features(items,tok.pad_token_id,device)
        logits,loss=model(x,a,l,eps,wess_on=(arm!='plain_mixed'))
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        if (step+1)%args.probe_every==0 or step==args.steps-1:
            meta.append({'step':step+1,'ratio':ratio,'train_loss':float(loss.detach()),'gate':float(model.fusion_gate.detach())})
            print(f'{arm} step {step+1}/{args.steps} ratio={ratio:.2f} loss={float(loss):.3f} gate={float(model.fusion_gate.detach()):.3f}',flush=True)
    return model,meta

@torch.no_grad()
def eval_binding(tok,model,eps,device,wess_on=True):
    model.eval(); by={}; correct=0; lo=0.0
    for e in eps:
        x,a,l,_,el=bridge.pad_features([e],tok.pad_token_id,device)
        logits,_=model(x,a,None,el,wess_on=wess_on)
        lp=F.log_softmax(logits[0,e.mask_pos],dim=-1)
        pred=int(lp.argmax()); correct+=int(pred==e.answer_id); lo+=float(lp[e.answer_id]-lp[e.counter_id])
        by.setdefault(e.pair_id,[]).append(pred==e.answer_id)
    pairs=[all(v) for v in by.values() if len(v)==2]
    model.train(); return {'example_acc':correct/len(eps),'pair_acc':sum(pairs)/len(pairs),'mean_logodds':lo/len(eps),'n_pairs':len(pairs)}

@torch.no_grad()
def eval_interventions(tok,model,eps,device):
    model.eval(); sw=[]; ab=[]
    for e in eps:
        x,a,l,_,el=bridge.pad_features([e],tok.pad_token_id,device)
        logits,_=model(x,a,None,el,wess_on=True); lp=F.log_softmax(logits[0,e.mask_pos],dim=-1)
        logits2,_=model(x,a,None,el,wess_on=True,intervention='swap'); lp2=F.log_softmax(logits2[0,e.mask_pos],dim=-1)
        sw.append(float((lp2[e.counter_id]-lp2[e.answer_id])-(lp[e.counter_id]-lp[e.answer_id])))
        logits3,_=model(x,a,None,el,wess_on=True,intervention='ablate'); lp3=F.log_softmax(logits3[0,e.mask_pos],dim=-1)
        ab.append(float((lp3[e.prev_id]-lp3[e.answer_id])-(lp[e.prev_id]-lp[e.answer_id])))
    model.train(); return {'swap_delta':sum(sw)/len(sw),'ablation_delta':sum(ab)/len(ab)}

@torch.no_grad()
def eval_official_loss(tok,model,official,device,wess_on=False,n=512):
    model.eval(); vals=[]
    for i in range(0,min(n,len(official)),32):
        items=official[i:i+32]
        x,a,l,_,eps=bridge.pad_features(items,tok.pad_token_id,device)
        _,loss=model(x,a,l,eps,wess_on=wess_on); vals.append(float(loss))
    model.train(); return sum(vals)/len(vals)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--hidden_size',type=int,default=384); p.add_argument('--n_layer',type=int,default=4); p.add_argument('--n_head',type=int,default=12); p.add_argument('--intermediate_size',type=int,default=1280)
    p.add_argument('--steps',type=int,default=1000); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--episode_ratio',type=float,default=0.5); p.add_argument('--anneal',action='store_true')
    p.add_argument('--encoder_lr',type=float,default=2e-4); p.add_argument('--slot_lr',type=float,default=1e-3); p.add_argument('--gate_init',type=float,default=0.03); p.add_argument('--probe_every',type=int,default=250); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train_pairs',type=int,default=1200); p.add_argument('--eval_pairs',type=int,default=200); p.add_argument('--official_examples',type=int,default=5000)
    p.add_argument('--run_name',default='smoke_384x4_anneal')
    return p.parse_args()


def main():
    args=parse_args(); bridge.setup_env(); t0=time.time(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(str(bridge.TOK_PATH),use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.mask_token
    train_eps=dyn.build_pairs_tmpl(tok,args.train_pairs,args.seed*10+1,dyn.TRAIN_TEMPLATES,dyn.TRAIN_QUERY)
    eval_eps=dyn.build_pairs_tmpl(tok,args.eval_pairs,9990,dyn.HELDOUT_TEMPLATES,dyn.HELDOUT_QUERY)
    official=load_official(tok,args.official_examples,args.seed*10+3)
    audit=bridge.shortcut_audit(eval_eps)
    run_dir=OUTROOT/args.run_name; run_dir.mkdir(parents=True,exist_ok=True)
    results={'args':vars(args),'device':str(device),'shortcut_audit':audit,'train_eps':len(train_eps),'eval_eps':len(eval_eps),'official_examples':len(official),'arms':{}}
    for arm in ['plain_mixed','wess_aux']:
        print(f'\nARM {arm}',flush=True)
        model,meta=train(tok,arm,args,train_eps,official,device)
        rec={'train_meta':meta,'official_loss_base_only':eval_official_loss(tok,model,official,device,wess_on=False),
             'synthetic_base_only':eval_binding(tok,model,eval_eps,device,wess_on=False)}
        if arm=='wess_aux':
            rec['synthetic_wess_on']=eval_binding(tok,model,eval_eps,device,wess_on=True)
            rec['interventions_wess_on']=eval_interventions(tok,model,eval_eps[:120],device)
        export_dir=run_dir/arm/'hf_model'
        model.export_plain(tok,export_dir)
        rec['exported_plain_hf_model']=str(export_dir)
        rec['params_plain_hf']=sum(p.numel() for p in model.mlm.parameters())
        rec['params_with_wess']=sum(p.numel() for p in model.parameters())
        results['arms'][arm]=rec
        DATAOUT.parent.mkdir(parents=True,exist_ok=True); DATAOUT.write_text(json.dumps({'status':'PARTIAL',**results,'elapsed_sec':round(time.time()-t0,1)},indent=2)+'\n')
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    results['elapsed_sec']=round(time.time()-t0,1); results['status']='WESS_AUX_BASE_TRANSFER_SCREEN'
    DATAOUT.write_text(json.dumps(results,indent=2)+'\n')
    (run_dir/'screen_result.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'out':str(DATAOUT),'run_dir':str(run_dir),'summary':{a:results['arms'][a] for a in results['arms']}},indent=2)[:4000],flush=True)

if __name__=='__main__': main()
