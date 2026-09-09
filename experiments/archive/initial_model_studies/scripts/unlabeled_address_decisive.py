#!/usr/bin/env python3
"""research decisive unlabeled-address WESS experiment.

Train-time gold metadata supervises a token-role router only. At predicted-route inference,
WESS receives input_ids, attention_mask and router logits; it does NOT read gold event
positions, entity addresses, query position, or event assignments. EpisodeFeat is retained
only for answer scoring and router accuracy comparison.

Predicted roles: 0 other, 1 event-entity, 2 event-state, 3 query-entity.
Predicted event-state tokens are paired with the nearest preceding predicted event entity.
Slots are created by first occurrence of distinct entity token IDs; query slot is selected by
matching the predicted query-entity token ID. This directly tests learned unlabeled routing.
"""
from __future__ import annotations
import importlib.util, json, pathlib, random, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
DYN=ROOT/'scripts/wess_dynamics.py'
OUT=ROOT/'data/unlabeled_address_decisive.json'
spec=importlib.util.spec_from_file_location('dyn',DYN)
dyn=importlib.util.module_from_spec(spec); sys.modules[spec.name]=dyn; spec.loader.exec_module(dyn)
bridge=dyn.bridge

class PredictedRouterWESS(nn.Module):
    def __init__(self,tok,hidden=384,layers=4,heads=12,intermediate=1280):
        super().__init__(); self.tok=tok
        cfg=DebertaV2Config(vocab_size=len(tok),hidden_size=hidden,num_hidden_layers=layers,
            num_attention_heads=heads,intermediate_size=intermediate,max_position_embeddings=512,
            position_buckets=256,relative_attention=True,pos_att_type=['p2c','c2p'],pad_token_id=tok.pad_token_id)
        self.mlm=DebertaV2ForMaskedLM(cfg)
        self.router=nn.Linear(hidden,4)
        self.slot_init=nn.Linear(hidden,hidden)
        self.event_mlp=nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.gru=nn.GRUCell(hidden,hidden); self.slot_proj=nn.Linear(hidden,hidden)
        self.gate=nn.Parameter(torch.tensor(0.03))

    def gold_roles(self,ep,L,device):
        y=torch.zeros(L,dtype=torch.long,device=device)
        for p in ep.event_entity_pos: y[p]=1
        for p in ep.event_state_pos: y[p]=2
        y[ep.query_entity_pos]=3
        return y

    def parse_predicted(self,ids,role_logits,mask_pos,mode='predicted'):
        roles=role_logits.argmax(-1).tolist(); ids_l=ids.tolist()
        ents=[i for i,r in enumerate(roles) if r==1 and i<mask_pos]
        states=[i for i,r in enumerate(roles) if r==2 and i<mask_pos]
        qs=[i for i,r in enumerate(roles) if r==3]
        # Pair every predicted state with nearest preceding predicted entity.
        events=[]
        for sp in states:
            before=[p for p in ents if p<sp]
            if before: events.append((before[-1],sp))
        # Stable token-identity addresses by first event occurrence.
        unique=[]
        for ep,_ in events:
            tid=ids_l[ep]
            if tid not in unique: unique.append(tid)
        if len(unique)<2 or not qs or not events: return None
        unique=unique[:2]
        qpos=min(qs,key=lambda p:abs(p-mask_pos)); qid=ids_l[qpos]
        if qid not in unique: return None
        qslot=unique.index(qid)
        routed=[]
        for ep,sp in events:
            tid=ids_l[ep]
            if tid in unique: routed.append((ep,sp,unique.index(tid)))
        if mode=='random': routed=[(ep,sp,(i*17+3)%2) for i,(ep,sp,_) in enumerate(routed)]
        if mode=='no_address': routed=[(ep,sp,0) for ep,sp,_ in routed]; qslot=0
        return routed,qslot,unique,qpos

    def memory_from_route(self,h,ids,route,intervention=None):
        routed,qslot,unique,qpos=route
        first=[]
        for slot in range(2):
            pos=next((ep for ep,sp,si in routed if si==slot),None)
            if pos is None:return None
            first.append(self.slot_init(h[pos]))
        slots=first
        last_q=max((i for i,(_,_,si) in enumerate(routed) if si==qslot),default=-1)
        for i,(ep,sp,si) in enumerate(routed):
            if intervention=='ablate' and i==last_q: continue
            upd=self.event_mlp(torch.cat([h[ep],h[sp]]))
            slots[si]=self.gru(upd.unsqueeze(0),slots[si].unsqueeze(0)).squeeze(0)
        if intervention=='swap': slots[0],slots[1]=slots[1],slots[0]
        return slots[qslot]

    def memory_gold(self,h,ep,intervention=None):
        sl=[]
        for ei in range(2):
            p=next(ep.event_entity_pos[t] for t,(x,_) in enumerate(ep.events) if x==ei)
            sl.append(self.slot_init(h[p]))
        last_q=max(i for i,(ei,_) in enumerate(ep.events) if ei==ep.query_entity)
        for t,(ei,_) in enumerate(ep.events):
            if intervention=='ablate' and t==last_q: continue
            upd=self.event_mlp(torch.cat([h[ep.event_entity_pos[t]],h[ep.event_state_pos[t]]]))
            sl[ei]=self.gru(upd.unsqueeze(0),sl[ei].unsqueeze(0)).squeeze(0)
        if intervention=='swap': sl[0],sl[1]=sl[1],sl[0]
        return sl[ep.query_entity]

    def forward(self,input_ids,attention_mask,labels=None,eps=None,route_mode='predicted',intervention=None,router_weight=0.0):
        h=self.mlm.deberta(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        rlog=self.router(h); rloss=torch.tensor(0.,device=h.device); nroute=0
        h2=h.clone()
        if eps is not None:
            for b,ep in enumerate(eps):
                if ep is None: continue
                gold=self.gold_roles(ep,h.shape[1],h.device)
                rloss=rloss+F.cross_entropy(rlog[b],gold); nroute+=1
                if route_mode=='gold': slot=self.memory_gold(h[b],ep,intervention)
                else:
                    route=self.parse_predicted(input_ids[b],rlog[b],ep.mask_pos,route_mode)
                    slot=self.memory_from_route(h[b],input_ids[b],route,intervention) if route else None
                if slot is not None:
                    h2=h2.clone(); h2[b,ep.mask_pos]=h[b,ep.mask_pos]+self.gate*self.slot_proj(slot)
        logits=self.mlm.cls(h2); loss=None
        if labels is not None:
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100)
            if nroute: loss=loss+router_weight*(rloss/nroute)
        return logits,loss,rlog

def ratio_at(step,steps):
    a=int(.6*steps); b=int(.85*steps)
    if step<a:return .5
    if step>=b:return .25
    return .5-.25*(step-a)/max(1,b-a)

def load_official(tok,n,seed): return dyn.load_official(tok,n,seed,max_len=160)

def train(tok,train_eps,official,device,steps=1200):
    torch.manual_seed(42); random.seed(42); model=PredictedRouterWESS(tok).to(device)
    enc=[]; aux=[]
    for n,p in model.named_parameters(): (enc if n.startswith('mlm.') else aux).append(p)
    opt=torch.optim.AdamW([{'params':enc,'lr':2e-4},{'params':aux,'lr':1e-3}],weight_decay=.01)
    sch=get_cosine_schedule_with_warmup(opt,int(.05*steps),steps); rng=random.Random(93); meta=[]
    for step in range(steps):
        ratio=ratio_at(step,steps); ne=round(32*ratio)
        items=[rng.choice(train_eps) for _ in range(ne)]+[rng.choice(official) for _ in range(32-ne)]; rng.shuffle(items)
        x,a,l,_,eps=bridge.pad_features(items,tok.pad_token_id,device)
        # Gold route early for a stable slot algorithm; predicted route during final 40%.
        mode='gold' if step<int(.6*steps) else 'predicted'
        _,loss,_=model(x,a,l,eps,route_mode=mode,router_weight=1.0)
        assert torch.isfinite(loss); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
        if (step+1)%300==0:
            meta.append({'step':step+1,'ratio':ratio,'route_mode':mode,'loss':float(loss.detach()),'gate':float(model.gate.detach())}); print(meta[-1],flush=True)
    return model,meta

@torch.no_grad()
def router_metrics(model,eps,tok,device):
    counts={'token_correct':0,'token_total':0,'entity_correct':0,'entity_total':0,'state_correct':0,'state_total':0,'query_correct':0,'query_total':0,'parse_success':0,'route_exact':0}
    model.eval()
    for e in eps:
        x,a,l,_,el=bridge.pad_features([e],tok.pad_token_id,device)
        h=model.mlm.deberta(input_ids=x,attention_mask=a).last_hidden_state; rl=model.router(h)[0]; pred=rl.argmax(-1)
        gold=model.gold_roles(e,len(pred),device); valid=a[0].bool(); counts['token_correct']+=int((pred[valid]==gold[valid]).sum()); counts['token_total']+=int(valid.sum())
        for lab,name in [(1,'entity'),(2,'state'),(3,'query')]:
            m=gold==lab; counts[name+'_correct']+=int((pred[m]==lab).sum()); counts[name+'_total']+=int(m.sum())
        route=model.parse_predicted(x[0],rl,e.mask_pos,'predicted')
        if route:
            counts['parse_success']+=1; routed,qslot,unique,qpos=route
            gold_pairs={(e.event_entity_pos[t],e.event_state_pos[t],ei) for t,(ei,_) in enumerate(e.events)}
            if set(routed)==gold_pairs and qslot==e.query_entity: counts['route_exact']+=1
    n=len(eps); return {**counts,'token_acc':counts['token_correct']/counts['token_total'],'entity_recall':counts['entity_correct']/counts['entity_total'],'state_recall':counts['state_correct']/counts['state_total'],'query_recall':counts['query_correct']/counts['query_total'],'parse_success_rate':counts['parse_success']/n,'route_exact_rate':counts['route_exact']/n}

@torch.no_grad()
def evaluate(model,eps,tok,device,mode,intervention=None):
    model.eval(); corr=0; lo=0.; by={}
    for e in eps:
        x,a,l,_,el=bridge.pad_features([e],tok.pad_token_id,device); z,_,_=model(x,a,None,el,route_mode=mode,intervention=intervention)
        lp=F.log_softmax(z[0,e.mask_pos],-1); pred=int(lp.argmax()); corr+=pred==e.answer_id; lo+=float(lp[e.answer_id]-lp[e.counter_id]); by.setdefault(e.pair_id,[]).append(pred==e.answer_id)
    pairs=[all(v) for v in by.values() if len(v)==2]
    return {'example_acc':corr/len(eps),'pair_acc':sum(pairs)/len(pairs),'mean_logodds':lo/len(eps)}

def main():
    bridge.setup_env(); t0=time.time(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tok=AutoTokenizer.from_pretrained(str(bridge.TOK_PATH),use_fast=True)
    if tok.pad_token is None:tok.pad_token=tok.mask_token
    train_eps=dyn.build_pairs_tmpl(tok,1600,1951,dyn.TRAIN_TEMPLATES,dyn.TRAIN_QUERY)
    held=dyn.build_pairs_tmpl(tok,300,1952,dyn.HELDOUT_TEMPLATES,dyn.HELDOUT_QUERY)
    official=load_official(tok,5000,1953); model,meta=train(tok,train_eps,official,device)
    rm=router_metrics(model,held,tok,device); modes={}
    for mode in ['gold','predicted','no_address','random']:
        base=evaluate(model,held,tok,device,mode); sw=evaluate(model,held,tok,device,mode,'swap'); ab=evaluate(model,held,tok,device,mode,'ablate')
        modes[mode]={**base,'swap_delta':sw['mean_logodds']-base['mean_logodds'],'ablation_delta':ab['mean_logodds']-base['mean_logodds']}
    payload={'status':'UNLABELED_ADDRESS_DECISIVE','description':'Predicted-route inference uses no gold episode positions/addresses; metadata only scores answers/router accuracy.','train_meta':meta,'shortcut_audit':bridge.shortcut_audit(held),'router_metrics':rm,'modes':modes,'elapsed_sec':round(time.time()-t0,1)}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2),flush=True)

if __name__=='__main__':main()
