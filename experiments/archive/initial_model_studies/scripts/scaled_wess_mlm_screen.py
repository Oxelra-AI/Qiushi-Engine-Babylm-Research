#!/usr/bin/env python3
"""research scaled WESS-MLM screen runner.

Purpose: move the replicated research WESS bridge toward a BabyLM-relevant short-budget
screen without committing to a full 100M candidate. The script is configurable:
- DeBERTa-v2 shape (hidden/layers/heads/intermediate)
- matched arms: plain_mlm, no_address, wess_gold, wess_eventwise_random, optional wrong_entity
- official-text + counterfactual episode mixture
- mechanism eval: binding example/pair accuracy, log-odds, slot swap, write ablation
- official-text MLM validation loss
- explicit accounting of approximate unique/training words, token counts, masked targets

It imports the repaired research bridge implementation for the counterfactual paired suite
and WESS memory/fusion logic, but uses a configurable model class for larger screens.
"""
from __future__ import annotations
import argparse, importlib.util, json, pathlib, random, statistics, sys, time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model, get_cosine_schedule_with_warmup

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
SRC = ROOT / 'scripts/wess_mlm_transfer_pilot.py'
DEFAULT_OUT = ROOT / 'data/scaled_wess_mlm_screen.json'

spec = importlib.util.spec_from_file_location('bridge', SRC)
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)  # type: ignore


def load_official_texts_with_accounting(tok, n:int, seed:int, max_len:int=192):
    rng=random.Random(seed); rows=[]
    for fn in ['childes.train.txt','bnc_spoken.train.txt','simple_wiki.train.txt','open_subtitles.train.txt','qed.train.txt']:
        p=bridge.RAW/fn
        if not p.exists():
            continue
        with p.open('r',encoding='utf-8',errors='replace') as f:
            for line in f:
                text=line.strip()
                wc=len(text.split())
                if 6 <= wc <= 120:
                    rows.append((text,wc,fn))
                if len(rows)>=n*8:
                    break
    rng.shuffle(rows)
    examples=[]; total_words=0; total_tokens=0
    for text,wc,src in rows:
        enc=tok(text, add_special_tokens=False, truncation=True, max_length=max_len)
        ids=list(enc['input_ids'])
        if len(ids)<8:
            continue
        labels=[-100]*len(ids)
        cand=[i for i,t in enumerate(ids) if t not in {tok.pad_token_id, tok.mask_token_id}]
        if not cand:
            continue
        mpos=rng.choice(cand)
        labels[mpos]=ids[mpos]
        ids[mpos]=tok.mask_token_id
        examples.append({'input_ids':ids,'labels':labels,'words':wc,'source':src,'tokens':len(ids),'masked_targets':1})
        total_words += wc; total_tokens += len(ids)
        if len(examples)>=n:
            break
    return examples, {'official_examples':len(examples),'official_unique_words_loaded':total_words,'official_tokens_loaded':total_tokens}


def episode_words(ep) -> int:
    # Count the unmasked answer as the corpus word rather than the literal <mask> token.
    return len(ep.text.replace(ep.tok.mask_token if hasattr(ep, 'tok') else '<mask>', ep.answer_state).split()) if hasattr(ep,'tok') else len(ep.text.split())


class ScaledMLMWESS(nn.Module):
    def __init__(self, tok, arm:str, hidden:int, layers:int, heads:int, intermediate:int, max_pos:int=512, position_buckets:int=256):
        super().__init__(); self.arm=arm; self.tok=tok; self.hidden=hidden
        cfg=DebertaV2Config(vocab_size=len(tok), hidden_size=hidden, num_hidden_layers=layers,
                            num_attention_heads=heads, intermediate_size=intermediate,
                            max_position_embeddings=max_pos, position_buckets=position_buckets,
                            relative_attention=True, pos_att_type=['p2c','c2p'],
                            pad_token_id=tok.pad_token_id)
        self.enc=DebertaV2Model(cfg)
        self.bias=nn.Parameter(torch.zeros(len(tok)))
        self.slot_init=nn.Linear(hidden,hidden)
        self.event_mlp=nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.gru=nn.GRUCell(hidden,hidden)
        self.fuse=nn.Linear(2*hidden,hidden)
    def route_slot(self, ep, ei:int, t:int):
        if self.arm=='wess_gold': return ei
        if self.arm=='wess_eventwise_random': return random.Random(ep.pair_id*1000+t*17+3).randrange(2)
        if self.arm=='wess_wrong_entity': return 1-ei
        if self.arm=='no_address': return 0
        return ei
    def run_memory(self,h,ep, skip_last_query=False, swap=False):
        if self.arm=='plain_mlm': return None
        ent0=h[ep.event_entity_pos[0]]; ent1=None
        for idx,(ei,_) in enumerate(ep.events):
            if ei==1: ent1=h[ep.event_entity_pos[idx]]; break
        if ent1 is None: ent1=ent0
        slot_list=[self.slot_init(ent0), self.slot_init(ent1)]
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
            h_new=h.clone()
            for b,ep in enumerate(eps):
                if ep is None: continue
                slot=self.run_memory(h[b],ep, skip_last_query=(intervention=='ablate'), swap=(intervention=='swap'))
                if slot is not None:
                    m=ep.mask_pos
                    fused=self.fuse(torch.cat([h[b,m], slot]))
                    h_new=h_new.clone(); h_new[b,m]=fused
            h=h_new
        logits=torch.matmul(h,self.enc.embeddings.word_embeddings.weight.t())+self.bias
        loss=None
        if labels is not None:
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return logits, loss


def train_arm(tok, arm:str, train_eps, official, args, device):
    torch.manual_seed(args.seed); random.seed(args.seed)
    model=ScaledMLMWESS(tok, arm, args.hidden_size, args.n_layer, args.n_head, args.intermediate_size,
                        max_pos=args.max_position_embeddings, position_buckets=args.position_buckets).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup=max(1, int(args.steps*args.warmup_fraction))
    sched=get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=args.steps)
    rng=random.Random(args.seed+31)
    losses=[]; masked_targets=0; official_batches=0; episode_batches=0; train_word_exposure=0; train_token_exposure=0
    n_ep=max(0, min(args.batch_size, round(args.batch_size*args.episode_ratio)))
    for step in range(args.steps):
        ep_batch=[rng.choice(train_eps) for _ in range(n_ep)]
        off_batch=[rng.choice(official) for _ in range(args.batch_size-len(ep_batch))]
        items=ep_batch+off_batch; rng.shuffle(items)
        x,a,l,_,eps=bridge.pad_features(items,tok.pad_token_id,device)
        _,loss=model(x,a,l,eps)
        assert torch.isfinite(loss),f'nonfinite {arm} step {step}'
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        losses.append(float(loss.detach().cpu()))
        masked_targets += int((l!=-100).sum().item())
        episode_batches += sum(1 for it in items if not isinstance(it,dict))
        official_batches += sum(1 for it in items if isinstance(it,dict))
        train_token_exposure += int(a.sum().item())
        for it in items:
            if isinstance(it,dict): train_word_exposure += int(it.get('words',0))
            else: train_word_exposure += len(it.text.split())
        if args.log_every and (step+1)%args.log_every==0:
            print(f'  {arm} step {step+1}/{args.steps} loss={losses[-1]:.4f}', flush=True)
    train_meta={'steps':args.steps,'batch_size':args.batch_size,'episode_examples_seen':episode_batches,
                'official_examples_seen':official_batches,'approx_word_exposure':train_word_exposure,
                'token_exposure':train_token_exposure,'masked_targets':masked_targets,
                'loss_first':losses[0] if losses else None,'loss_last':losses[-1] if losses else None,
                'loss_mean_last20':sum(losses[-20:])/max(1,len(losses[-20:]))}
    return model, train_meta


@torch.no_grad()
def eval_official_loss(tok,model,official,device,n=256):
    model.eval(); losses=[]
    for i in range(0,min(n,len(official)),32):
        items=official[i:i+32]
        x,a,l,_,eps=bridge.pad_features(items,tok.pad_token_id,device)
        _,loss=model(x,a,l,eps); losses.append(float(loss))
    return sum(losses)/len(losses)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--output', default=str(DEFAULT_OUT))
    p.add_argument('--hidden_size', type=int, default=384)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=12)
    p.add_argument('--intermediate_size', type=int, default=1280)
    p.add_argument('--max_position_embeddings', type=int, default=512)
    p.add_argument('--position_buckets', type=int, default=256)
    p.add_argument('--steps', type=int, default=240)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--episode_ratio', type=float, default=0.10)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_fraction', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_pairs', type=int, default=1200)
    p.add_argument('--eval_pairs', type=int, default=240)
    p.add_argument('--official_examples', type=int, default=5000)
    p.add_argument('--max_official_len', type=int, default=192)
    p.add_argument('--arms', default='plain_mlm,no_address,wess_gold,wess_eventwise_random')
    p.add_argument('--log_every', type=int, default=50)
    return p.parse_args()


def main():
    args=parse_args(); bridge.setup_env(); t0=time.time()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(str(bridge.TOK_PATH), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.mask_token
    arms=[a.strip() for a in args.arms.split(',') if a.strip()]
    out=pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    train_eps=bridge.build_pairs(tok,args.train_pairs,args.seed*10+1)
    eval_eps=bridge.build_pairs(tok,args.eval_pairs,args.seed*10+2)
    official, official_meta=load_official_texts_with_accounting(tok,args.official_examples,args.seed*10+3,args.max_official_len)
    audit=bridge.shortcut_audit(eval_eps)
    episode_words_train=sum(len(e.text.split()) for e in train_eps)
    episode_words_eval=sum(len(e.text.split()) for e in eval_eps)
    early={'status':'SCALED_WESS_SCREEN_STARTED','args':vars(args),'device':str(device),'arms':arms,
           'train_episode_examples':len(train_eps),'eval_episode_examples':len(eval_eps),'official_meta':official_meta,
           'train_episode_words_unique_approx':episode_words_train,'eval_episode_words_approx':episode_words_eval,
           'shortcut_audit_eval':audit,'source_bridge':str(SRC),'elapsed_sec':round(time.time()-t0,1)}
    out.write_text(json.dumps(early, indent=2)+'\n')
    print('EARLY', json.dumps({k:early[k] for k in ['device','arms','train_episode_examples','eval_episode_examples','official_meta','shortcut_audit_eval']}, indent=2), flush=True)

    results={}
    for arm in arms:
        print(f'\nARM {arm}', flush=True)
        model, train_meta=train_arm(tok,arm,train_eps,official,args,device)
        rec={'train_meta':train_meta,
             'binding':bridge.eval_binding(tok,model,eval_eps,device),
             'official_mlm_loss':eval_official_loss(tok,model,official,device,n=min(512,len(official))),
             'params':sum(p.numel() for p in model.parameters())}
        if arm!='plain_mlm':
            rec['interventions']=bridge.eval_interventions(tok,model,eval_eps[:min(160,len(eval_eps))],device)
        results[arm]=rec
        partial={**early,'status':'SCALED_WESS_SCREEN_PARTIAL','arms_results':results,'elapsed_sec':round(time.time()-t0,1)}
        out.write_text(json.dumps(partial,indent=2)+'\n')
        print(json.dumps({arm:rec},indent=2), flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    summary={}
    for arm,rec in results.items():
        summary[arm]={'mean_logodds':rec['binding']['mean_logodds'],'example_acc':rec['binding']['example_acc'],
                      'pair_acc':rec['binding']['pair_acc'],'official_mlm_loss':rec['official_mlm_loss']}
        if 'interventions' in rec:
            summary[arm]['swap_delta']=rec['interventions']['swap_logodds_delta_other_minus_answer']
            summary[arm]['ablation_delta']=rec['interventions']['ablation_logodds_delta_prev_minus_answer']
    payload={**early,'status':'SCALED_WESS_SCREEN_COMPLETE','arms_results':results,'summary':summary,'elapsed_sec':round(time.time()-t0,1)}
    out.write_text(json.dumps(payload,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)
    print(f'Saved {out}',flush=True)

if __name__=='__main__':
    main()
