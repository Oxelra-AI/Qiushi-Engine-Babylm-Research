#!/usr/bin/env python3
"""research — WESS slot-signal DYNAMICS: why does binding wash out, and can it persist?

Stability concern: the 1000-step peak / 2000-step collapse means we found a
transient, not a stable mechanism. We must NOT pick a peak on the same synthetic suite.

This experiment:
1. Uses DIVERSE templates for training and a SEPARATE independent-template held-out suite
   (different verbs, different query phrasing) so the tracked signal is not the training
   distribution.
2. Tracks the binding signal TRAJECTORY on the held-out suite every `probe_every` steps,
   for the full training run — so we SEE the wash-out dynamics rather than picking endpoints.
3. Compares stabilization hypotheses at S1 depth (12x384):
   - baseline: gated residual, LR 5e-4, constant-through-cosine (reproduce wash-out)
   - two_phase: freeze encoder after step F, train only slot pathway + gate (protect binding)
   - gate_l1: small L1 penalty pushing gate away from 0 once binding appears (anti-decay)
   - higher_episode: 50% episode ratio (more binding gradient per step)
4. Runs 2 seeds to check the trajectory pattern is not seed-specific.

Output: per-config, per-seed trajectory of held-out pair_acc / logodds / gate, plus
final endpoint. The scientific question is persistence, not peak.
"""
from __future__ import annotations
import importlib.util, json, pathlib, random, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model, get_cosine_schedule_with_warmup

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
SRC = ROOT / 'scripts/wess_mlm_transfer_pilot.py'
OUT = ROOT / 'data/wess_dynamics.json'

spec = importlib.util.spec_from_file_location('bridge', SRC)
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)

EpisodeFeat = bridge.EpisodeFeat
ENTITY_NAMES = bridge.ENTITY_NAMES
STATE_CANDIDATES = bridge.STATE_CANDIDATES
one_token_id = bridge.one_token_id
tok_pos_from_span = bridge.tok_pos_from_span

# Diverse templates: (verb phrase producing "<ent> <verb...> the <state>", ent_word_idx_in_sent, state marker)
# Each template renders an event sentence and we recover positions by offsets, so we just vary surface form.
TRAIN_TEMPLATES = [
    ("{e} moved to the {s}.",),
    ("{e} went into the {s}.",),
    ("{e} walked to the {s}.",),
    ("{e} is now in the {s}.",),
]
# Independent held-out templates: DIFFERENT verbs/phrasing not seen in training.
HELDOUT_TEMPLATES = [
    ("{e} entered the {s}.",),
    ("{e} stayed inside the {s}.",),
    ("{e} arrived at the {s}.",),
]
TRAIN_QUERY = "Where is {e}? {m}."
HELDOUT_QUERY = "{e} can be found in the {m}."


def make_member_tmpl(tok, pair_id, member, ents, states, assignment, query, templates, query_tmpl, rng):
    parts=[]; events=[]; ent_char=[]; st_char=[]; cur=[None]*2; prev=None
    for ei,si in assignment:
        e=ents[ei]; s=states[si]
        tmpl = rng.choice(templates)[0]
        sent = tmpl.format(e=e, s=s)
        off=sum(len(p)+1 for p in parts) if parts else 0
        e_start=off + sent.index(e); e_end=e_start+len(e)
        s_start=off + sent.rindex(s); s_end=s_start+len(s)
        parts.append(sent); events.append((ei,s)); ent_char.append((e_start,e_end)); st_char.append((s_start,s_end))
        if ei==query and cur[query] is not None:
            prev=cur[query]
        cur[ei]=s
    qsent=query_tmpl.format(e=ents[query], m=tok.mask_token)
    off=sum(len(p)+1 for p in parts)
    q_ent_start=off + qsent.index(ents[query]); q_ent_end=q_ent_start+len(ents[query])
    text=' '.join(parts+[qsent])
    enc=tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=160)
    ids=enc['input_ids']; offsets=enc['offset_mapping']
    mask_positions=[i for i,x in enumerate(ids) if x==tok.mask_token_id]
    if len(mask_positions)!=1: return None
    epos=[]; spos=[]
    for cs,ce in ent_char:
        p=tok_pos_from_span(offsets,cs,ce)
        if not p: return None
        epos.append(p[0])
    for cs,ce in st_char:
        p=tok_pos_from_span(offsets,cs,ce)
        if not p: return None
        spos.append(p[0])
    qpos=tok_pos_from_span(offsets,q_ent_start,q_ent_end)
    if not qpos: return None
    answer=cur[query]; other=1-query; counter=cur[other]
    if answer is None or counter is None or answer==counter or prev is None or prev==answer: return None
    aid=one_token_id(tok,answer); cid=one_token_id(tok,counter); pid=one_token_id(tok,prev)
    if aid is None or cid is None or pid is None: return None
    labels=[-100]*len(ids); labels[mask_positions[0]]=aid
    return EpisodeFeat(pair_id,member,text,query,other,answer,counter,prev,events,epos,spos,qpos[0],mask_positions[0],aid,cid,pid,ids,labels)


def build_pairs_tmpl(tok, n_pairs, seed, templates, query_tmpl):
    rng=random.Random(seed); feats=[]; pid=0
    good_states=[s for s in STATE_CANDIDATES if one_token_id(tok,s) is not None]
    good_ents=[e for e in ENTITY_NAMES if len(tok(' '+e,add_special_tokens=False)['input_ids'])<=2]
    while len(feats)<2*n_pairs and pid<20*n_pairs:
        ents=rng.sample(good_ents,2); states=rng.sample(good_states,4)
        a=make_member_tmpl(tok,pid,0,ents,states,[(0,0),(1,1),(0,2),(1,3)],0,templates,query_tmpl,rng)
        b=make_member_tmpl(tok,pid,1,ents,states,[(1,0),(0,1),(1,2),(0,3)],0,templates,query_tmpl,rng)
        if a and b: feats.extend([a,b])
        pid+=1
    return feats[:2*n_pairs]


def load_official(tok, n, seed, max_len=160):
    rng=random.Random(seed); rows=[]
    for fn in ['childes.train.txt','bnc_spoken.train.txt','simple_wiki.train.txt','open_subtitles.train.txt','qed.train.txt']:
        p=bridge.RAW/fn
        if not p.exists(): continue
        with p.open('r',encoding='utf-8',errors='replace') as f:
            for line in f:
                text=line.strip(); wc=len(text.split())
                if 6<=wc<=120: rows.append(text)
                if len(rows)>=n*8: break
    rng.shuffle(rows); ex=[]
    for text in rows:
        enc=tok(text,add_special_tokens=False,truncation=True,max_length=max_len)
        ids=list(enc['input_ids'])
        if len(ids)<8: continue
        labels=[-100]*len(ids)
        cand=[i for i,t in enumerate(ids) if t not in {tok.pad_token_id,tok.mask_token_id}]
        if not cand: continue
        m=rng.choice(cand); labels[m]=ids[m]; ids[m]=tok.mask_token_id
        ex.append({'input_ids':ids,'labels':labels})
        if len(ex)>=n: break
    return ex


class GatedWESS(nn.Module):
    def __init__(self, tok, arm, hidden=384, layers=12, heads=12, intermediate=1280):
        super().__init__(); self.arm=arm; self.tok=tok; self.hidden=hidden
        cfg=DebertaV2Config(vocab_size=len(tok),hidden_size=hidden,num_hidden_layers=layers,
            num_attention_heads=heads,intermediate_size=intermediate,max_position_embeddings=512,
            position_buckets=256,relative_attention=True,pos_att_type=['p2c','c2p'],pad_token_id=tok.pad_token_id)
        self.enc=DebertaV2Model(cfg)
        self.bias=nn.Parameter(torch.zeros(len(tok)))
        self.slot_init=nn.Linear(hidden,hidden)
        self.event_mlp=nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.gru=nn.GRUCell(hidden,hidden)
        self.slot_proj=nn.Linear(hidden,hidden)
        self.fusion_gate=nn.Parameter(torch.tensor(0.0))
    def route_slot(self,ep,ei,t):
        if self.arm=='wess_gold': return ei
        if self.arm=='no_address': return 0
        return ei
    def run_memory(self,h,ep,skip_last_query=False,swap=False):
        if self.arm=='plain_mlm': return None
        ent0=h[ep.event_entity_pos[0]]; ent1=None
        for idx,(ei,_) in enumerate(ep.events):
            if ei==1: ent1=h[ep.event_entity_pos[idx]]; break
        if ent1 is None: ent1=ent0
        sl=[self.slot_init(ent0),self.slot_init(ent1)]
        if self.arm=='no_address': sl[1]=sl[0]
        last_q=max(i for i,(ei,_) in enumerate(ep.events) if ei==ep.query_entity)
        for t,(ei,_) in enumerate(ep.events):
            if skip_last_query and t==last_q: continue
            upd=self.event_mlp(torch.cat([h[ep.event_entity_pos[t]],h[ep.event_state_pos[t]]]))
            si=self.route_slot(ep,ei,t)
            sl[si]=self.gru(upd.unsqueeze(0),sl[si].unsqueeze(0)).squeeze(0)
            if self.arm=='no_address': sl[1]=sl[0]
        if swap and self.arm=='wess_gold': sl[0],sl[1]=sl[1],sl[0]
        return sl[ep.query_entity if self.arm!='no_address' else 0]
    def forward(self,input_ids,attention_mask,labels=None,eps=None,intervention=None,freeze_enc=False):
        if freeze_enc:
            with torch.no_grad():
                h=self.enc(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
            h=h.detach()
        else:
            h=self.enc(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        if eps is not None and self.arm!='plain_mlm':
            h_new=h.clone()
            for b,ep in enumerate(eps):
                if ep is None: continue
                slot=self.run_memory(h[b],ep,skip_last_query=(intervention=='ablate'),swap=(intervention=='swap'))
                if slot is not None:
                    m=ep.mask_pos
                    h_new=h_new.clone(); h_new[b,m]=h[b,m]+self.fusion_gate*self.slot_proj(slot)
            h=h_new
        logits=torch.matmul(h,self.enc.embeddings.word_embeddings.weight.t())+self.bias
        loss=None
        if labels is not None:
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100)
        return logits,loss


@torch.no_grad()
def probe(tok, model, eps, device):
    model.eval(); by_pair={}; correct=0; lo=0.0
    for e in eps:
        x,a,l,_,epl=bridge.pad_features([e],tok.pad_token_id,device)
        logits,_=model(x,a,None,epl)
        lp=F.log_softmax(logits[0,e.mask_pos],dim=-1)
        pred=int(lp.argmax()); correct+=int(pred==e.answer_id)
        lo+=float(lp[e.answer_id]-lp[e.counter_id])
        by_pair.setdefault(e.pair_id,[]).append(pred==e.answer_id)
    pair=[all(v) for v in by_pair.values() if len(v)==2]
    model.train()
    return {'example_acc':correct/len(eps),'pair_acc':sum(pair)/max(1,len(pair)),'mean_logodds':lo/len(eps)}


def train_with_trajectory(tok, arm, cfg, train_eps, official, heldout, device, seed, probe_every=200):
    torch.manual_seed(seed); random.seed(seed)
    model=GatedWESS(tok, arm).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=0.01)
    steps=cfg['steps']; warmup=max(1,int(steps*0.05))
    sched=get_cosine_schedule_with_warmup(opt,warmup,steps)
    rng=random.Random(seed+31)
    ep_ratio=cfg.get('episode_ratio',0.25); n_ep=max(1,round(32*ep_ratio))
    freeze_at=cfg.get('freeze_enc_at',None); gate_l1=cfg.get('gate_l1',0.0)
    traj=[]
    for step in range(steps):
        fr = (freeze_at is not None and step>=freeze_at and arm=='wess_gold')
        items=[rng.choice(train_eps) for _ in range(n_ep)]+[rng.choice(official) for _ in range(32-n_ep)]
        rng.shuffle(items)
        x,a,l,_,eps=bridge.pad_features(items,tok.pad_token_id,device)
        _,loss=model(x,a,l,eps,freeze_enc=fr)
        if gate_l1>0 and arm=='wess_gold':
            loss = loss - gate_l1*torch.abs(model.fusion_gate)  # reward larger |gate| once useful
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        if (step+1)%probe_every==0 or step==steps-1:
            p=probe(tok,model,heldout,device)
            p['step']=step+1; p['gate']=float(model.fusion_gate.detach()); p['loss']=float(loss.detach())
            traj.append(p)
    final=probe(tok,model,heldout,device)
    if arm=='wess_gold':
        final['interventions']=bridge.eval_interventions(tok,model,heldout[:120],device)
        final['learned_gate']=float(model.fusion_gate.detach())
    return traj, final


def main():
    bridge.setup_env(); t0=time.time(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(str(bridge.TOK_PATH),use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.mask_token

    train_eps=build_pairs_tmpl(tok,1200,4210,TRAIN_TEMPLATES,TRAIN_QUERY)
    heldout=build_pairs_tmpl(tok,200,9990,HELDOUT_TEMPLATES,HELDOUT_QUERY)
    official=load_official(tok,5000,4230)
    audit_ho=bridge.shortcut_audit(heldout)
    print(f'train={len(train_eps)} heldout={len(heldout)} official={len(official)}',flush=True)
    print(f'heldout shortcut audit: {json.dumps(audit_ho)}',flush=True)

    configs={
        'baseline_lr5e4_1500': {'lr':5e-4,'steps':1500,'episode_ratio':0.25},
        'twophase_freeze750': {'lr':5e-4,'steps':1500,'episode_ratio':0.25,'freeze_enc_at':750},
        'higher_episode50': {'lr':5e-4,'steps':1500,'episode_ratio':0.50},
    }
    seeds=[42,123]
    results={}
    for cname,cfg in configs.items():
        results[cname]={}
        for seed in seeds:
            print(f'\n=== {cname} seed {seed} ===',flush=True)
            arm_res={}
            for arm in (['plain_mlm','wess_gold'] if seed==seeds[0] else ['wess_gold']):
                traj,final=train_with_trajectory(tok,arm,cfg,train_eps,official,heldout,device,seed)
                arm_res[arm]={'trajectory':traj,'final':final}
                tj=' '.join(f"s{t['step']}:pa={t['pair_acc']:.2f},lo={t['mean_logodds']:.2f},g={t['gate']:.3f}" for t in traj)
                print(f'  {arm}: {tj}',flush=True)
            results[cname][str(seed)]=arm_res
            OUT.write_text(json.dumps({'status':'PARTIAL','heldout_audit':audit_ho,'configs':configs,'results':results,'elapsed_sec':round(time.time()-t0,1)},indent=2)+'\n')
    payload={'status':'WESS_DYNAMICS_COMPLETE','description':'Held-out independent-template binding trajectory across training for S1-depth WESS stabilization hypotheses.','heldout_audit':audit_ho,'configs':configs,'seeds':seeds,'results':results,'elapsed_sec':round(time.time()-t0,1)}
    OUT.write_text(json.dumps(payload,indent=2)+'\n')
    print(f'\nSaved {OUT}',flush=True)

if __name__=='__main__':
    main()
