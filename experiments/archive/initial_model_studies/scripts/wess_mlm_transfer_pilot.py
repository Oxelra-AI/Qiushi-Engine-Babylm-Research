#!/usr/bin/env python3
"""research short-range DeBERTa+WESS MLM transfer pilot.

This is NOT a BabyLM SOTA run. It is the bridge test specified in
plans/short_range_wess_transfer_contract_v2.md:
- balanced counterfactual paired entity-state episodes;
- shortcut audit before training;
- real HF tokenizer + DeBERTa-v2 MLM backbone;
- matched arms: plain MLM, no-address memory, WESS gold, random-route, wrong-entity;
- binding-pair accuracy/log-odds and directed slot interventions.

The experiment is intentionally small (one seed by default) so it can reveal
whether WESS can affect masked natural word prediction under MLM before any 100M run.
"""
from __future__ import annotations
import json, math, os, pathlib, random, time
from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
OUT = ROOT / 'data/wess_mlm_transfer_pilot.json'
RAW = ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/raw_dataset'
TOK_PATH = ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M'

ENTITY_NAMES = ['Alice','Bob','Charlie','Daisy','Emma','Frank','Grace','Henry','Iris','Jack','Lily','Martin']
STATE_CANDIDATES = ['garden','kitchen','attic','garage','office','porch','closet','basement','hallway','bedroom','cellar','pantry']

@dataclass
class EpisodeFeat:
    pair_id: int
    member: int
    text: str
    query_entity: int
    other_entity: int
    answer_state: str
    counter_state: str
    prev_state: str
    events: list[tuple[int, str]]  # entity idx, state word
    event_entity_pos: list[int]
    event_state_pos: list[int]
    query_entity_pos: int
    mask_pos: int
    answer_id: int
    counter_id: int
    prev_id: int
    input_ids: list[int]
    labels: list[int]


def setup_env():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def one_token_id(tok, word: str) -> int | None:
    """Return the lexical token id for a short state word.

    The BabyLM baseline ByteLevel tokenizer often represents a prefixed word as
    [standalone-space-token, word-token], e.g. ' garden' -> ['Ġ','Ġgarden'].
    Our query uses a single <mask>, so score the lexical content token and ignore
    the standalone spacing token when this exact pattern occurs.
    """
    ids_pref = tok(' ' + word, add_special_tokens=False)['input_ids']
    toks_pref = tok.convert_ids_to_tokens(ids_pref)
    if len(ids_pref) == 1:
        return ids_pref[0]
    if len(ids_pref) == 2 and toks_pref[0] in {'Ġ', 'Ċ'}:
        return ids_pref[1]
    ids_bare = tok(word, add_special_tokens=False)['input_ids']
    if len(ids_bare) == 1:
        return ids_bare[0]
    return None


def tok_pos_from_span(offsets, start, end):
    poss=[]
    for i,(s,e) in enumerate(offsets):
        if s < end and e > start:
            poss.append(i)
    return poss


def make_member(tok, pair_id:int, member:int, ents:list[str], states:list[str], assignment:list[tuple[int,int]], query:int) -> EpisodeFeat | None:
    """Render one member and recover token positions via offsets.

    assignment: list of (entity_idx, state_idx) events. Final state of query is answer.
    """
    parts=[]; events=[]; ent_char=[]; st_char=[]; cur=[None]*2; prev=None
    for ei,si in assignment:
        e=ents[ei]; s=states[si]
        sent=f"{e} moved to the {s}."
        off=sum(len(p)+1 for p in parts) if parts else 0
        e_start=off + sent.index(e); e_end=e_start+len(e)
        s_start=off + sent.index(s); s_end=s_start+len(s)
        parts.append(sent); events.append((ei,s)); ent_char.append((e_start,e_end)); st_char.append((s_start,s_end))
        if ei==query and cur[query] is not None:
            prev=cur[query]
        cur[ei]=s
    qsent=f"Where is {ents[query]}? {tok.mask_token}."
    off=sum(len(p)+1 for p in parts)
    q_ent_start=off + qsent.index(ents[query]); q_ent_end=q_ent_start+len(ents[query])
    mask_start=off + qsent.index(tok.mask_token); mask_end=mask_start+len(tok.mask_token)
    text=' '.join(parts+[qsent])
    enc=tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=128)
    ids=enc['input_ids']; offsets=enc['offset_mapping']
    mask_positions=[i for i,x in enumerate(ids) if x==tok.mask_token_id]
    if len(mask_positions)!=1:
        return None
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
    if answer is None or counter is None or answer==counter or prev is None or prev==answer:
        return None
    aid=one_token_id(tok,answer); cid=one_token_id(tok,counter); pid=one_token_id(tok,prev)
    if aid is None or cid is None or pid is None:
        return None
    labels=[-100]*len(ids); labels[mask_positions[0]]=aid
    return EpisodeFeat(pair_id,member,text,query,other,answer,counter,prev,events,epos,spos,qpos[0],mask_positions[0],aid,cid,pid,ids,labels)


def build_pairs(tok, n_pairs:int, seed:int) -> list[EpisodeFeat]:
    rng=random.Random(seed)
    feats=[]; pid=0
    good_states=[s for s in STATE_CANDIDATES if one_token_id(tok,s) is not None]
    good_ents=[e for e in ENTITY_NAMES if len(tok(' '+e, add_special_tokens=False)['input_ids'])==1 or len(tok(e, add_special_tokens=False)['input_ids'])==1]
    while len(feats)<2*n_pairs and pid<10*n_pairs:
        ents=rng.sample(good_ents,2); states=rng.sample(good_states,4)
        # Pair with same event positions/token multiset; query entity fixed as entity 0.
        # Member A answer=s2, counter=s3; member B answer=s3, counter=s2.
        # Both have old states and distractor update after final query-entity write in A/B balanced by pair set.
        assign_A=[(0,0),(1,1),(0,2),(1,3)]
        assign_B=[(1,0),(0,1),(1,2),(0,3)]
        # query entity 0 in both; answers differ: A states[2], B states[3].
        a=make_member(tok,pid,0,ents,states,assign_A,0)
        b=make_member(tok,pid,1,ents,states,assign_B,0)
        if a and b:
            feats.extend([a,b])
        pid+=1
    return feats[:2*n_pairs]


def shortcut_audit(eps:list[EpisodeFeat]) -> dict[str,Any]:
    by_pair={}
    for e in eps: by_pair.setdefault(e.pair_id,[]).append(e)
    def state_seq(ep): return [s for _,s in ep.events]
    preds={
        'first_state': lambda e: state_seq(e)[0],
        'last_state': lambda e: state_seq(e)[-1],
        'nearest_state': lambda e: state_seq(e)[-1],
        'majority_fixed_first_visible': lambda e: sorted(set(state_seq(e)))[0],
        'prev_query_state': lambda e: e.prev_state,
    }
    out={}
    for name,fn in preds.items():
        ex=[]; pair=[]
        for pid,members in by_pair.items():
            if len(members)!=2: continue
            cor=[fn(e)==e.answer_state for e in members]
            ex.extend(cor); pair.append(all(cor))
        out[name]={'example_acc':sum(ex)/len(ex),'pair_acc':sum(pair)/len(pair),'n_pairs':len(pair)}
    return out


def load_official_texts(tok, n:int, seed:int, max_len:int=96):
    rng=random.Random(seed); lines=[]
    for fn in ['childes.train.txt','bnc_spoken.train.txt','simple_wiki.train.txt']:
        p=RAW/fn
        if p.exists():
            with p.open('r',encoding='utf-8',errors='replace') as f:
                for line in f:
                    line=line.strip()
                    if 6<=len(line.split())<=80: lines.append(line)
                    if len(lines)>=n*5: break
    rng.shuffle(lines); examples=[]
    for line in lines[:n]:
        enc=tok(line, add_special_tokens=False, truncation=True, max_length=max_len)
        ids=enc['input_ids']
        if len(ids)<8: continue
        labels=[-100]*len(ids); ids=list(ids)
        cand=[i for i,t in enumerate(ids) if t not in {tok.pad_token_id, tok.mask_token_id}]
        if not cand: continue
        mpos=rng.choice(cand); labels[mpos]=ids[mpos]; ids[mpos]=tok.mask_token_id
        examples.append({'input_ids':ids,'labels':labels})
    return examples[:n]


def pad_features(items, pad_id, device):
    max_len=max(len(it['input_ids']) if isinstance(it,dict) else len(it.input_ids) for it in items)
    B=len(items)
    input_ids=torch.full((B,max_len),pad_id,dtype=torch.long,device=device)
    labels=torch.full((B,max_len),-100,dtype=torch.long,device=device)
    is_ep=[]; eps=[]
    for i,it in enumerate(items):
        if isinstance(it,dict):
            ids=it['input_ids']; labs=it['labels']; is_ep.append(False); eps.append(None)
        else:
            ids=it.input_ids; labs=it.labels; is_ep.append(True); eps.append(it)
        input_ids[i,:len(ids)]=torch.tensor(ids,device=device)
        labels[i,:len(labs)]=torch.tensor(labs,device=device)
    att=(input_ids!=pad_id).long()
    return input_ids,att,labels,is_ep,eps


class MLMWESS(nn.Module):
    def __init__(self, tok, arm:str, hidden=128, layers=2):
        super().__init__(); self.arm=arm; self.tok=tok; self.hidden=hidden
        cfg=DebertaV2Config(vocab_size=len(tok), hidden_size=hidden, num_hidden_layers=layers,
                            num_attention_heads=4, intermediate_size=hidden*4, max_position_embeddings=256,
                            position_buckets=64, relative_attention=True, pos_att_type=['p2c','c2p'],
                            pad_token_id=tok.pad_token_id)
        self.enc=DebertaV2Model(cfg)
        self.bias=nn.Parameter(torch.zeros(len(tok)))
        self.slot_init=nn.Linear(hidden,hidden)
        self.event_mlp=nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.gru=nn.GRUCell(hidden,hidden)
        self.fuse=nn.Linear(2*hidden,hidden)
    def route_slot(self, ep:EpisodeFeat, ei:int, t:int):
        if self.arm=='wess_gold': return ei
        if self.arm=='wess_eventwise_random': return random.Random(ep.pair_id*1000+t*17+3).randrange(2)
        if self.arm=='wess_wrong_entity': return 1-ei
        if self.arm=='no_address': return 0
        return ei
    def run_memory(self,h,ep:EpisodeFeat, skip_last_query=False, swap=False):
        if self.arm=='plain_mlm': return None
        # Use a list of separate tensors to avoid in-place mutation on graph tensors
        ent0=h[ep.event_entity_pos[0]]; ent1=None
        for idx,(ei,_) in enumerate(ep.events):
            if ei==1: ent1=h[ep.event_entity_pos[idx]]; break
        if ent1 is None: ent1=ent0
        slot_list = [self.slot_init(ent0), self.slot_init(ent1)]
        if self.arm=='no_address': slot_list[1]=slot_list[0]
        last_q=max(i for i,(ei,_) in enumerate(ep.events) if ei==ep.query_entity)
        for t,(ei,_) in enumerate(ep.events):
            if skip_last_query and t==last_q: continue
            e_repr=h[ep.event_entity_pos[t]]; s_repr=h[ep.event_state_pos[t]]
            upd=self.event_mlp(torch.cat([e_repr,s_repr]))
            si=self.route_slot(ep,ei,t)
            new_val=self.gru(upd.unsqueeze(0),slot_list[si].unsqueeze(0)).squeeze(0)
            slot_list[si]=new_val
            if self.arm=='no_address': slot_list[1]=slot_list[0]
        if swap and self.arm in {'wess_gold','wess_eventwise_random','wess_wrong_entity'}:
            slot_list[0], slot_list[1] = slot_list[1], slot_list[0]
        return slot_list[ep.query_entity if self.arm!='no_address' else 0]
    def forward(self,input_ids,attention_mask,labels=None,eps=None,intervention=None):
        h=self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if eps is not None and self.arm!='plain_mlm':
            # Build a new h tensor with fused slot readout at mask positions
            # Avoid in-place mutation: scatter fused values into a copy
            h_new = h.clone()
            for b,ep in enumerate(eps):
                if ep is None: continue
                slot=self.run_memory(h[b],ep, skip_last_query=(intervention=='ablate'), swap=(intervention=='swap'))
                if slot is not None:
                    m=ep.mask_pos
                    fused = self.fuse(torch.cat([h[b,m], slot]))
                    # Use scatter to avoid indexed assignment on graph tensor
                    h_new = h_new.clone()
                    h_new[b, m] = fused
            h = h_new
        logits=torch.matmul(h,self.enc.embeddings.word_embeddings.weight.t())+self.bias
        loss=None
        if labels is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),labels.view(-1),ignore_index=-100)
        return logits,loss


def train_arm(tok,arm,train_eps,official,eval_eps,seed,device,steps=180,batch=16):
    torch.manual_seed(seed); random.seed(seed)
    model=MLMWESS(tok,arm).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=0.01)
    rng=random.Random(seed+19)
    for step in range(steps):
        ep_batch=[rng.choice(train_eps) for _ in range(batch//2)]
        off_batch=[rng.choice(official) for _ in range(batch-len(ep_batch))]
        items=ep_batch+off_batch; rng.shuffle(items)
        x,a,l,is_ep,eps=pad_features(items,tok.pad_token_id,device)
        _,loss=model(x,a,l,eps)
        assert torch.isfinite(loss),f'nonfinite {arm} step {step}'
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model

@torch.no_grad()
def eval_binding(tok,model,eps,device):
    model.eval(); by_pair={}; rows=[]
    for e in eps:
        x,a,l,_,ep_list=pad_features([e],tok.pad_token_id,device)
        logits,_=model(x,a,None,ep_list)
        lp=F.log_softmax(logits[0,e.mask_pos],dim=-1)
        pred=int(lp.argmax())
        rows.append({'pair_id':e.pair_id,'member':e.member,'correct':pred==e.answer_id,
                     'logp_correct':float(lp[e.answer_id]),'logp_counter':float(lp[e.counter_id]),
                     'logodds':float(lp[e.answer_id]-lp[e.counter_id])})
        by_pair.setdefault(e.pair_id,[]).append(pred==e.answer_id)
    ex_acc=sum(r['correct'] for r in rows)/len(rows)
    pair_vals=[all(v) for v in by_pair.values() if len(v)==2]
    return {'example_acc':ex_acc,'pair_acc':sum(pair_vals)/len(pair_vals),'mean_logodds':sum(r['logodds'] for r in rows)/len(rows),'n_pairs':len(pair_vals),'n_examples':len(rows)}

@torch.no_grad()
def eval_interventions(tok,model,eps,device):
    model.eval(); deltas=[]; transfers=[]; abl_deltas=[]; abl_transfers=[]
    for e in eps:
        x,a,l,_,ep_list=pad_features([e],tok.pad_token_id,device)
        logits,_=model(x,a,None,ep_list); lp=F.log_softmax(logits[0,e.mask_pos],dim=-1)
        logitss,_=model(x,a,None,ep_list,intervention='swap'); lps=F.log_softmax(logitss[0,e.mask_pos],dim=-1)
        before=float(lp[e.counter_id]-lp[e.answer_id]); after=float(lps[e.counter_id]-lps[e.answer_id])
        deltas.append(after-before); transfers.append(int(int(lps.argmax())==e.counter_id))
        logitsa,_=model(x,a,None,ep_list,intervention='ablate'); lpa=F.log_softmax(logitsa[0,e.mask_pos],dim=-1)
        before2=float(lp[e.prev_id]-lp[e.answer_id]); after2=float(lpa[e.prev_id]-lpa[e.answer_id])
        abl_deltas.append(after2-before2); abl_transfers.append(int(int(lpa.argmax())==e.prev_id))
    return {'swap_logodds_delta_other_minus_answer':sum(deltas)/len(deltas),'swap_top1_to_other':sum(transfers)/len(transfers),
            'ablation_logodds_delta_prev_minus_answer':sum(abl_deltas)/len(abl_deltas),'ablation_top1_to_prev':sum(abl_transfers)/len(abl_transfers)}

@torch.no_grad()
def eval_official_loss(tok,model,official,device,n=100):
    model.eval(); losses=[]
    for i in range(0,min(n,len(official)),32):
        items=official[i:i+32]
        x,a,l,_,eps=pad_features(items,tok.pad_token_id,device)
        _,loss=model(x,a,l,eps); losses.append(float(loss))
    return sum(losses)/len(losses)


def main():
    setup_env(); t0=time.time(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(str(TOK_PATH),use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.mask_token
    train_eps=build_pairs(tok,240,1851); eval_eps=build_pairs(tok,120,2851)
    official=load_official_texts(tok,600,1852)
    audit=shortcut_audit(eval_eps)
    
    # Save audit early so it's preserved even if training crashes
    OUT.parent.mkdir(parents=True, exist_ok=True)
    early_info = {
        'status': 'AUDIT_ONLY',
        'train_episodes': len(train_eps), 'eval_episodes': len(eval_eps),
        'official_examples': len(official),
        'shortcut_audit': audit,
        'example_episode': {'text': train_eps[0].text, 'answer': train_eps[0].answer_state,
                           'counter': train_eps[0].counter_state, 'prev': train_eps[0].prev_state} if train_eps else {},
    }
    print('SHORTCUT AUDIT:', json.dumps(audit, indent=2), flush=True)
    audit_path = OUT.parent / 'early_audit.json'
    audit_path.write_text(json.dumps(early_info, indent=2) + '\n')
    print(f'Early audit saved: {audit_path}', flush=True)
    
    # Quick smoke test: 3 steps of wess_gold to verify no autograd errors
    print('SMOKE TEST: wess_gold 3 steps...', flush=True)
    torch.manual_seed(185)
    smoke_model = MLMWESS(tok, 'wess_gold').to(device)
    smoke_opt = torch.optim.AdamW(smoke_model.parameters(), lr=8e-4)
    for i in range(3):
        items = [train_eps[i], official[i]]
        x, a, l, _, eps_l = pad_features(items, tok.pad_token_id, device)
        _, loss = smoke_model(x, a, l, eps_l)
        assert torch.isfinite(loss), f'Smoke test failed step {i}: non-finite loss'
        smoke_opt.zero_grad(); loss.backward(); smoke_opt.step()
    print(f'SMOKE TEST PASSED (loss={float(loss):.4f})', flush=True)
    del smoke_model, smoke_opt
    
    arms=['plain_mlm','no_address','wess_gold','wess_eventwise_random','wess_wrong_entity']
    results={}
    for arm in arms:
        print(f'ARM {arm}',flush=True)
        model=train_arm(tok,arm,train_eps,official,eval_eps,seed=185,device=device)
        rec={'binding':eval_binding(tok,model,eval_eps,device),'official_mlm_loss':eval_official_loss(tok,model,official,device)}
        if arm!='plain_mlm': rec['interventions']=eval_interventions(tok,model,eval_eps[:80],device)
        rec['params']=sum(p.numel() for p in model.parameters())
        results[arm]=rec
        print(json.dumps({arm:rec},indent=2),flush=True)
        # Save incrementally after each arm
        partial = {**early_info, 'status': 'PARTIAL', 'arms': results, 'elapsed_sec': round(time.time()-t0, 1)}
        OUT.write_text(json.dumps(partial, indent=2) + '\n')
    payload={'status':'WESS_MLM_TRANSFER_PILOT','description':'short bridge pilot: DeBERTa-v2 MLM with counterfactual paired natural-language episodes and WESS/no-address/destroyed routing arms','device':str(device),'train_episode_examples':len(train_eps),'eval_episode_examples':len(eval_eps),'official_examples':len(official),'shortcut_audit':audit,'arms':results,'elapsed_sec':round(time.time()-t0,1)}
    OUT.write_text(json.dumps(payload,indent=2)+'\n')
    print(f'Saved {OUT}')

if __name__=='__main__': main()
