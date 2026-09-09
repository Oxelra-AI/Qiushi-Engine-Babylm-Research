#!/usr/bin/env python3
"""research: left-to-right lexical-occurrence raw-token memory.

This is the third and most mechanistically direct cheap raw-token screen. It uses
no template metadata. Candidate cells are raw token identities/positions; every
occurrence of a token id updates all same-id cells in sequence order using that
occurrence's contextual representation. The mask position reads a cell through
learned attention. This supplies a natural temporal-overwrite operator and tests
whether entity-state memory can emerge from raw token recurrence rather than from
hard research regex coordinates.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()
def model_sha(m):
    h=hashlib.sha256()
    for n,p in sorted(m.state_dict().items()): h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]
def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text('utf-8').splitlines() if x.strip()]
def encode_row(row,tok,max_len):
    enc=tok(row['text'],padding='max_length',truncation=True,max_length=max_len); mid=tok.mask_token_id
    try: mpos=enc['input_ids'].index(mid)
    except ValueError: raise ValueError('mask absent '+row['text'])
    aid=tok.encode(' '+row['answer'],add_special_tokens=False); fid=tok.encode(' '+row['foil'],add_special_tokens=False)
    if len(aid)<2 or len(fid)<2: raise ValueError(f"target short {row['answer']} {aid} {row['foil']} {fid}")
    return {**enc,'mask_pos':mpos,'answer_id':aid[1],'foil_id':fid[1]}
class Rows(Dataset):
    def __init__(self,rows,tok,max_len):
        self.items=[]
        for r in rows:
            x=encode_row(r,tok,max_len); x['meta']=r; self.items.append(x)
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]
def collate(xs):
    out={}
    for k in ['input_ids','attention_mask','mask_pos','answer_id','foil_id']:
        out[k]=torch.tensor([x[k] for x in xs],dtype=torch.long)
    out['meta']=[x['meta'] for x in xs]
    return out

class LexicalRecurrentMemory(nn.Module):
    def __init__(self,d,special_ids=(0,1,2,3,4),residual_scale=1.0):
        super().__init__(); self.d=d; self.scale=d**-0.5; self.special_ids=tuple(int(x) for x in special_ids); self.residual_scale=residual_scale
        self.init_cell=nn.Parameter(torch.zeros(d))
        self.local=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.LayerNorm(d))
        self.update_value=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,d))
        self.update_gate=nn.Sequential(nn.Linear(2*d,d//2),nn.GELU(),nn.Linear(d//2,1))
        self.entity_bias=nn.Linear(d,1)
        self.read_query=nn.Linear(d,d,bias=False); self.read_key=nn.Linear(d,d,bias=False)
        self.out=nn.Linear(d,d); self.norm=nn.LayerNorm(d)
    def forward(self,h,input_ids,mask_pos,pad,permute_write=False,return_aux=False):
        B,L,D=h.shape; dev=h.device; ar=torch.arange(B,device=dev); neg=torch.finfo(h.dtype).min
        left=torch.cat([h[:,:1],h[:,:-1]],1); right=torch.cat([h[:,1:],h[:,-1:]],1)
        loc=self.local(torch.cat([left,h,right],-1))
        content=~pad
        for sid in self.special_ids: content=content & input_ids.ne(sid)
        slots=self.init_cell[None,None,:].expand(B,L,D).clone()
        # Sequential same-id recurrence. For each occurrence p, update all candidate
        # cells whose token id matches token p; this creates an identity-indexed state.
        write_means=[]
        for p in range(L):
            cur_valid=content[:,p]
            same=input_ids.eq(input_ids[:,p:p+1]) & content & cur_valid[:,None]  # (B,L) candidate cells updated by occurrence p
            if not same.any():
                continue
            ctx=loc[:,p,:]
            ctx_rep=ctx[:,None,:].expand(B,L,D)
            old=slots
            inp=torch.cat([ctx_rep,old],-1)
            val=self.update_value(inp)
            gate=torch.sigmoid(self.update_gate(inp)).squeeze(-1)
            if permute_write:
                # Send this occurrence's update to a reverse-position identity cell.
                same=same.flip(1)
            new=old + gate[:,:,None]*(val-old)
            slots=torch.where(same[:,:,None], new, slots)
            if return_aux: write_means.append((gate*same.float()).sum(1)/(same.float().sum(1).clamp_min(1.0)))
        q=self.read_query(h[ar,mask_pos]); rk=self.read_key(loc)
        rlog=(q[:,None,:]*rk).sum(-1)*self.scale + self.entity_bias(loc).squeeze(-1)
        rlog=rlog.masked_fill(~content,neg)
        ra=torch.softmax(rlog,dim=-1)
        read=torch.einsum('bl,bld->bd',ra,slots)
        hout=h.clone(); hout[ar,mask_pos]=self.norm(hout[ar,mask_pos]+self.residual_scale*self.out(read))
        aux={'read_attention':ra,'slots':slots}
        if return_aux and write_means: aux['write_mean']=torch.stack(write_means,1)
        return (hout,aux) if return_aux else (hout,None)

class TinyMLM(nn.Module):
    def __init__(self,vocab,arm,d=192,layers=4,heads=6,ff=768,max_len=80):
        super().__init__(); self.arm=arm
        self.word=nn.Embedding(vocab,d); self.pos=nn.Embedding(max_len,d); self.embnorm=nn.LayerNorm(d)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d,heads,ff,0.1,batch_first=True,norm_first=True,activation='gelu') for _ in range(layers)])
        self.memory=None if arm=='vanilla' else LexicalRecurrentMemory(d)
        self.lm=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.LayerNorm(d),nn.Linear(d,vocab,bias=False)); self.lm[-1].weight=self.word.weight
    def forward(self,b,permute_write=False,return_aux=False):
        ids=b['input_ids']; B,L=ids.shape; pos=torch.arange(L,device=ids.device)[None]
        h=self.embnorm(self.word(ids)+self.pos(pos)); pad=b['attention_mask'].eq(0); aux={}
        for i,layer in enumerate(self.layers):
            h=layer(h,src_key_padding_mask=pad)
            if i==1 and self.memory is not None:
                h,aux=self.memory(h,ids,b['mask_pos'],pad,permute_write,return_aux)
        return self.lm(h),aux

def todev(b,dev): return {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}
@torch.no_grad()
def evaluate(model,loader,dev,permute=False):
    model.eval(); rows=[]; sums=defaultdict(lambda:[0,0,0.0])
    for b in loader:
        meta=b['meta']; b=todev(b,dev); logits,aux=model(b,permute_write=permute,return_aux=False)
        ar=torch.arange(logits.shape[0],device=dev); z=logits[ar,b['mask_pos']]; margin=z[ar,b['answer_id']]-z[ar,b['foil_id']]; ok=margin.gt(0)
        for i,m in enumerate(meta):
            sp=m['split']; sums[sp][0]+=int(ok[i]); sums[sp][1]+=1; sums[sp][2]+=float(margin[i])
            rows.append({'id':m['id'],'split':sp,'kind':m['kind'],'family':m['family'],'is_affected':m.get('is_affected_query'),
                         'quartet_id':m.get('quartet_id',''),'correct':bool(ok[i]),'margin':float(margin[i]),'permuted':permute})
    summary={k:{'correct':v[0],'n':v[1],'accuracy':v[0]/v[1],'margin_mean':v[2]/v[1]} for k,v in sums.items()}
    def sel(sp):
        hr=[r for r in rows if r['split']==sp and r['kind']=='binding']
        if not hr: return None
        aff={r['quartet_id']:r['correct'] for r in hr if r.get('is_affected')==True}; una={r['quartet_id']:r['correct'] for r in hr if r.get('is_affected')==False}
        aa=sum(aff.values())/max(len(aff),1); ua=sum(una.values())/max(len(una),1); qg=defaultdict(list)
        for r in hr:
            base='_'.join(r['quartet_id'].rsplit('_',2)[:1]); qg[base].append(r)
        pb=sum(1 for g in qg.values() if any(r['correct'] for r in g if r.get('is_affected')==True) and any(r['correct'] for r in g if r.get('is_affected')==False))
        qa=sum(1 for g in qg.values() if all(r['correct'] for r in g))
        return {'affected_accuracy':aa,'unaffected_accuracy':ua,'composite':0.5*(aa+ua),'pair_both_correct':pb,'pair_total':len(qg),'pair_both_rate':pb/max(len(qg),1),'quartet_all_correct':qa,'quartet_all_rate':qa/max(len(qg),1),'n_affected':len(aff),'n_unaffected':len(una)}
    for sp in ['eval_held_recomb','eval_held_paraphrase']:
        s=sel(sp)
        if s is not None: summary['selective_'+sp]=s
    mr=[r for r in rows if r['kind']=='multi_event']
    if mr: summary['multi_event']={'correct':sum(r['correct'] for r in mr),'n':len(mr),'accuracy':sum(r['correct'] for r in mr)/len(mr)}
    return summary,rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--arm',choices=['vanilla','lexrec'],required=True); ap.add_argument('--data-dir',required=True); ap.add_argument('--tokenizer',required=True); ap.add_argument('--out-dir',default='')
    ap.add_argument('--seed',type=int,default=43022); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--lr',type=float,default=5e-4); ap.add_argument('--max-len',type=int,default=96); ap.add_argument('--d-model',type=int,default=192); ap.add_argument('--layers',type=int,default=4); ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(); seed_all(args.seed); out=Path(args.out_dir or os.environ.get('QIUSHI_AI_LAB_RUN_DIR','.')); out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(args.tokenizer,trust_remote_code=True,local_files_only=True)
    train_rows=read_jsonl(Path(args.data_dir)/'train.jsonl'); eval_rows=read_jsonl(Path(args.data_dir)/'eval.jsonl')
    if args.dry_run: train_rows=train_rows[:64]; eval_rows=eval_rows[:64]; args.epochs=1
    train=Rows(train_rows,tok,args.max_len); ev=Rows(eval_rows,tok,args.max_len); g=torch.Generator().manual_seed(args.seed)
    tl=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=g,collate_fn=collate,num_workers=0); el=DataLoader(ev,batch_size=args.batch_size,shuffle=False,collate_fn=collate,num_workers=0)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=TinyMLM(len(tok),args.arm,args.d_model,args.layers,max_len=args.max_len).to(dev); nparams=sum(p.numel() for p in model.parameters())
    print(json.dumps({'arm':args.arm,'parameters':nparams,'device':str(dev),'train_records':len(train),'eval_records':len(ev)}),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01); losses=[]; t0=time.time()
    for ep in range(args.epochs):
        model.train(); total=0.0; n=0
        for b in tl:
            b=todev(b,dev); opt.zero_grad(set_to_none=True); logits,_=model(b); ar=torch.arange(logits.shape[0],device=dev)
            loss=F.cross_entropy(logits[ar,b['mask_pos']],b['answer_id']); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=float(loss.detach())*len(ar); n+=len(ar)
        losses.append(total/n); print(json.dumps({'epoch':ep+1,'loss':losses[-1]}),flush=True)
    base,rows=evaluate(model,el,dev,False); perm={} if args.arm=='vanilla' else evaluate(model,el,dev,True)[0]
    torch.save({'model':model.state_dict(),'args':vars(args)},out/'model.pt')
    with (out/'eval_rows.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r)+'\n')
    result={'status':'LEXICAL_RECURRENT_MEMORY','arm':args.arm,'seed':args.seed,'device':str(dev),'dry_run':args.dry_run,'train_records':len(train),'eval_records':len(ev),'epochs':args.epochs,'losses':losses,'parameters':nparams,'model_sha256':model_sha(model),'data_sha256':{'train':sha(Path(args.data_dir)/'train.jsonl'),'eval':sha(Path(args.data_dir)/'eval.jsonl')},'baseline':base,'write_permutation':perm,'runtime_sec':time.time()-t0}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
