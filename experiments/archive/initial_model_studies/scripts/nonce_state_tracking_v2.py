#!/usr/bin/env python3
"""Correct v2 pilot for nonce entity-state tracking.

Uses episode-local candidate retrieval, so unseen state symbols are evaluable.
Arms: endpoint Transformer; endpoint + random intermediate state supervision; WESS
with gold participant routing and strictly slot-only readout.
"""
from __future__ import annotations
import json, math, pathlib, random, time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
OUT=ROOT/'data/nonce_state_tracking/v2_pilot.json'
ENT_TR='dax wug blicket toma kiki bouba zup nib fep gax horp jiv lem quog rav sib'.split()
ENT_HO='vex yim zol pav ket mur fid gon'.split()
ST_TR='lup fen mor tib cav dren gol hax jep klin nuf pev rax sov tul wem'.split()
ST_HO='yib zaf brin cux dop elv fug hiv'.split()
SPECIAL=['[PAD]','[EV]','[Q]']
WORDS=['is','at','moves','to','goes','now','where']
VOCAB=SPECIAL+WORDS+ENT_TR+ENT_HO+ST_TR+ST_HO
T2I={x:i for i,x in enumerate(VOCAB)}
PAD=T2I['[PAD]']; EV=T2I['[EV]']; Q=T2I['[Q]']

@dataclass
class Ep:
    entities:list[str]; candidates:list[str]; events:list[tuple[int,int]]
    query:int; answer:int; tables:list[list[int]]


def make_ep(rng, ent_pool, state_pool, n_events, min_over=1):
    ents=rng.sample(ent_pool,4); states=rng.sample(state_pool,4)
    events=[]; cur=[None]*4; tables=[]
    # initialize every entity, each with one candidate state
    perm=list(range(4)); rng.shuffle(perm)
    for e,s in enumerate(perm):
        cur[e]=s; events.append((e,s)); tables.append(cur.copy())
    q=rng.randrange(4); over=0
    while len(events)<n_events:
        e=q if over<min_over else rng.randrange(4)
        choices=[s for s in range(4) if s!=cur[e]]; s=rng.choice(choices)
        if e==q: over+=1
        cur[e]=s; events.append((e,s)); tables.append(cur.copy())
    return Ep(ents,states,events,q,cur[q],tables)


def dataset(seed,n,ent_pool,state_pool,event_range,min_over=1):
    r=random.Random(seed)
    return [make_ep(r,ent_pool,state_pool,r.randint(*event_range),min_over) for _ in range(n)]


def seq(ep, upto=None, query=None):
    evs=ep.events if upto is None else ep.events[:upto+1]
    ids=[]
    for e,s in evs: ids += [EV,T2I[ep.entities[e]],T2I[ep.candidates[s]]]
    q=ep.query if query is None else query
    ids += [Q,T2I[ep.entities[q]]]
    return ids


def pad(seqs,device):
    m=max(map(len,seqs)); x=torch.full((len(seqs),m),PAD,dtype=torch.long,device=device)
    a=torch.zeros_like(x)
    for i,s in enumerate(seqs): x[i,:len(s)]=torch.tensor(s,device=device); a[i,:len(s)]=1
    return x,a

class Encoder(nn.Module):
    def __init__(self,d=96,layers=2):
        super().__init__(); self.emb=nn.Embedding(len(VOCAB),d); self.pos=nn.Embedding(128,d)
        lay=nn.TransformerEncoderLayer(d,4,192,.1,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(lay,layers); self.norm=nn.LayerNorm(d)
    def forward(self,x,a):
        p=torch.arange(x.size(1),device=x.device)[None]; h=self.norm(self.emb(x)+self.pos(p))
        return self.enc(h,src_key_padding_mask=(a==0))
    def candidates(self,eps):
        ids=torch.tensor([[T2I[s] for s in e.candidates] for e in eps],device=self.emb.weight.device)
        return self.emb(ids)

class Endpoint(nn.Module):
    def __init__(self,step=False): super().__init__(); self.base=Encoder(); self.step=step
    def score(self,eps,seqs):
        x,a=pad(seqs,self.base.emb.weight.device); h=self.base(x,a)
        # Q is penultimate token, entity is final; pool both
        q=torch.stack([(h[i,len(s)-2]+h[i,len(s)-1])/2 for i,s in enumerate(seqs)])
        c=self.base.candidates(eps); return torch.einsum('bd,bkd->bk',q,c)/math.sqrt(q.size(-1))
    def forward(self,eps): return self.score(eps,[seq(e) for e in eps])

class WESS(nn.Module):
    def __init__(self,d=96):
        super().__init__(); self.emb=nn.Embedding(len(VOCAB),d); self.init=nn.Linear(d,d)
        self.event=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,d))
        self.gru=nn.GRUCell(d,d); self.out=nn.Linear(d,d,bias=False)
    def run_slots(self,ep,skip_event=None):
        dev=self.emb.weight.device
        entids=torch.tensor([T2I[x] for x in ep.entities],device=dev)
        slots=self.init(self.emb(entids)); history=[slots]
        for t,(ei,si) in enumerate(ep.events):
            if t==skip_event: history.append(slots); continue
            v=self.event(torch.cat([self.emb(entids[ei]),self.emb(torch.tensor(T2I[ep.candidates[si]],device=dev))]))
            new=self.gru(v[None],slots[ei:ei+1])[0]
            slots=slots.clone(); slots[ei]=new; history.append(slots)
        return slots,history
    def score(self,eps,swap=None,skip=None):
        qs=[]; cs=[]
        for i,e in enumerate(eps):
            slots,_=self.run_slots(e,None if skip is None else skip[i])
            if swap is not None:
                a,b=swap[i]; slots=slots.clone(); slots[[a,b]]=slots[[b,a]]
            qs.append(self.out(slots[e.query]))
            ids=torch.tensor([T2I[s] for s in e.candidates],device=slots.device); cs.append(self.emb(ids))
        q=torch.stack(qs); c=torch.stack(cs); return torch.einsum('bd,bkd->bk',q,c)/math.sqrt(q.size(-1))
    def forward(self,eps): return self.score(eps)


def batches(data,b,rng):
    ix=list(range(len(data))); rng.shuffle(ix)
    for i in range(0,len(ix),b): yield [data[j] for j in ix[i:i+b]]

def train(model,arm,data,seed,epochs=18):
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.01); rng=random.Random(seed+991)
    for epc in range(epochs):
        model.train(); losses=[]
        for es in batches(data,32,rng):
            logits=model(es); y=torch.tensor([e.answer for e in es],device=logits.device)
            loss=F.cross_entropy(logits,y)
            if arm=='step':
                # random time/entity intermediate query, same candidate set
                ts=[rng.randrange(3,len(e.events)) for e in es]; ks=[rng.randrange(4) for _ in es]
                ss=[seq(e,upto=t,query=k) for e,t,k in zip(es,ts,ks)]
                li=model.score(es,ss); yi=torch.tensor([e.tables[t][k] for e,t,k in zip(es,ts,ks)],device=logits.device)
                loss=loss+F.cross_entropy(li,yi)
            assert torch.isfinite(loss),f'nonfinite {arm} epoch {epc}'
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(loss.item())
    return sum(losses)/len(losses)

@torch.no_grad()
def acc(model,data):
    model.eval(); n=c=0
    for i in range(0,len(data),64):
        es=data[i:i+64]; p=model(es).argmax(-1).cpu().tolist(); c+=sum(a==e.answer for a,e in zip(p,es)); n+=len(es)
    return c/n

@torch.no_grad()
def interventions(model,data):
    # Slot swap: swap queried slot with another; expected answer becomes other's final state.
    es=data[:100]; swaps=[]; expected=[]
    for e in es:
        b=(e.query+1)%4; swaps.append((e.query,b)); expected.append(e.tables[-1][b])
    ps=model.score(es,swap=swaps).argmax(-1).cpu().tolist(); swap_acc=sum(a==b for a,b in zip(ps,expected))/len(es)
    # Remove last query-entity write; expected state is state before that write.
    skips=[]; expected2=[]; valid=[]
    for e in es:
        idx=max(i for i,(k,_) in enumerate(e.events) if k==e.query)
        prior=None
        for j in range(idx-1,-1,-1):
            if e.events[j][0]==e.query: prior=e.events[j][1]; break
        if prior is not None: valid.append(e); skips.append(idx); expected2.append(prior)
    p2=model.score(valid,skip=skips).argmax(-1).cpu().tolist(); abl=sum(a==b for a,b in zip(p2,expected2))/len(valid)
    return {'slot_swap_transfer':swap_acc,'last_write_ablation_revert':abl,'n':len(valid)}

def main():
    t=time.time(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); seed=42
    trainset=dataset(seed,500,ENT_TR,ST_TR,(5,7),1)
    splits={
      'iid':dataset(1001,200,ENT_TR,ST_TR,(5,7),1),
      'new_ent':dataset(1002,200,ENT_HO,ST_TR,(5,7),1),
      'new_state':dataset(1003,200,ENT_TR,ST_HO,(5,7),1),
      'new_both':dataset(1004,200,ENT_HO,ST_HO,(5,7),1),
      'long':dataset(1005,200,ENT_TR,ST_TR,(9,12),3),
      'overwrite':dataset(1006,200,ENT_TR,ST_TR,(7,9),3)}
    result={'status':'V2_PILOT','seed':seed,'random_baseline':.25,'splits':{}}
    for arm in ['endpoint','step','wess']:
        torch.manual_seed(seed); m=(WESS() if arm=='wess' else Endpoint(step=(arm=='step'))).to(device)
        loss=train(m,arm,trainset,seed)
        scores={k:acc(m,v) for k,v in splits.items()}
        rec={'loss_last':loss,'params':sum(p.numel() for p in m.parameters()),'accuracy':scores}
        if arm=='wess': rec['interventions']=interventions(m,splits['overwrite'])
        result['splits'][arm]=rec; print(arm,rec,flush=True)
    result['elapsed_sec']=time.time()-t; OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
