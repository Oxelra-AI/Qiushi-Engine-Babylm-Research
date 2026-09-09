#!/usr/bin/env python3
"""research: activation-level query-match marker probe.

Train selected continuations from query-first bound preparation checkpoints and
probe hidden states at the four context attribute positions.  A shared scalar
linear probe is trained to select which context attribute belongs to the query.
It is trained on train-entity rows and evaluated on held-entity rows.  This tests
whether behavioral preservation tracks a reusable query-match signal in context
attribute states.

This is a representational probe, not a causal intervention.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import loss_allocation_binding as Base
import query_first_binding as S16
import binding_branching as S17
import budget_matched_full_objective as S18
import revision_019b_embedding_role_decomposition as S19b

SL=18; IS_POS=15; VOCAB=S16.VOCAB; PAD=S16.PAD; RWT=S16.RWT
ATTR_POS=[5,8,11,14]
DEFAULT_PREP_EPOCHS={42:300,43:500,100:400}

ARMS=[
    dict(name="direct_full", target="bound", kind="static", lr=3e-4, ctx_weight=1.0),
    dict(name="static_1over17", target="bound", kind="static", lr=3e-4, ctx_weight=1.0/17.0),
    dict(name="context_only_lr_half", target="bound", kind="context_only", lr=1.5e-4, ctx_weight=1.0),
    dict(name="slot0_static_1over17", target="slot0", kind="static", lr=3e-4, ctx_weight=1.0/17.0),
]
ARM_INDEX={a['name']:i for i,a in enumerate(ARMS)}


def custom_seqs(rows, target):
    seqs=[]
    for ce, ca, qi, bag_ti in rows:
        if target=="bound": tgt=RWT(ca[qi])
        elif target=="slot0": tgt=RWT(ca[0])
        else: raise ValueError(target)
        seqs.append(S16.mk_seq("query_first", ce, ca, qi, tgt))
    return seqs


def train_epoch(model,opt,seqs,order_idx,dev,bs,mask,kind,ctx_weight):
    st=torch.tensor(seqs,dtype=torch.long)[torch.tensor(order_idx,dtype=torch.long)]
    L=SL-1; model.train(); tot=den=ans=ctx=ctxden=0.0; cnt=0
    for i in range(0, st.size(0), bs):
        b=st[i:i+bs].to(dev); inp,tgt=b[:,:L],b[:,1:]
        logits=model(inp,mask)
        ce=F.cross_entropy(logits.reshape(-1,VOCAB), tgt.reshape(-1), reduction='none').view(tgt.shape)
        nonpad=(tgt!=PAD).float()
        if kind=="context_only":
            wt=nonpad.clone(); wt[:,IS_POS]=0.0; wt[:,-1]=0.0
        else:
            pw=torch.full((L,), float(ctx_weight), dtype=torch.float32, device=dev)
            pw[IS_POS]=1.0; pw[-1]=0.0
            wt=nonpad*pw.unsqueeze(0)
        loss=(ce*wt).sum()/wt.sum().clamp_min(1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tot+=float((ce*wt).sum()); den+=float(wt.sum())
            ans+=float(ce[:,IS_POS].sum()); cnt+=int(ce.size(0))
            cm=nonpad.clone(); cm[:,IS_POS]=0.0; cm[:,-1]=0.0
            ctx+=float((ce*cm).sum()); ctxden+=float(cm.sum())
    return dict(wloss=tot/max(den,1), ans_ce=ans/max(cnt,1), ctx_ce=ctx/max(ctxden,1))


def final_hidden(model, inp, mask):
    B,L=inp.shape
    h=model.tok(inp)+model.pos(torch.arange(L,device=inp.device))
    for b in model.blks:
        h=b(h,mask)
    return model.ln(h)


def gen_probe_rows(seed, n, mode):
    rng=np.random.default_rng(seed)
    rows=[]
    for _ in range(n):
        if mode=="train":
            ce=rng.choice(Base.TRAIN_E, Base.K, replace=False).tolist()
        elif mode=="held":
            he=int(rng.choice(Base.HELD_E))
            others=rng.choice(Base.TRAIN_E, Base.K-1, replace=False).tolist()
            ce=list(rng.permutation(others+[he]))
        else:
            raise ValueError(mode)
        ca=rng.choice(Base.N_ATTR, Base.K, replace=False).tolist()
        qi=int(rng.integers(Base.K)) if mode=="train" else ce.index(next(e for e in ce if e in Base.HELD_E))
        bag_ti=int(rng.integers(Base.K))
        rows.append((ce,ca,qi,bag_ti))
    return rows


def collect_feats(model, rows, dev, mask, bs=128):
    seqs=custom_seqs(rows,"bound")
    feats=[]; labels=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(seqs),bs):
            bseq=torch.tensor(seqs[i:i+bs],dtype=torch.long,device=dev)
            h=final_hidden(model,bseq[:,:SL-1],mask)
            feats.append(h[:,ATTR_POS,:].detach().cpu())
            labels.extend([r[2] for r in rows[i:i+bs]])
    return torch.cat(feats,0), torch.tensor(labels,dtype=torch.long)


def train_probe(feat, lab, feat_val, lab_val, steps=400):
    # feat: N x 4 x d. Shared scalar readout over slots.
    d=feat.shape[-1]
    lin=nn.Linear(d,1,bias=False)
    opt=torch.optim.AdamW(lin.parameters(), lr=0.05, weight_decay=1e-3)
    x=feat.float(); y=lab
    xv=feat_val.float(); yv=lab_val
    best=None
    for step in range(steps):
        score=lin(x).squeeze(-1)
        loss=F.cross_entropy(score,y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step%25==0 or step==steps-1:
            with torch.no_grad():
                val_score=lin(xv).squeeze(-1)
                val_acc=float((val_score.argmax(1)==yv).float().mean().item())
                val_loss=float(F.cross_entropy(val_score,yv).item())
                if best is None or val_acc>best['val_acc'] or (val_acc==best['val_acc'] and val_loss<best['val_loss']):
                    best=dict(step=step,val_acc=val_acc,val_loss=val_loss,state=copy.deepcopy(lin.state_dict()))
    lin.load_state_dict(best['state'])
    return lin, {k:v for k,v in best.items() if k!='state'}


def eval_probe(lin, feat, lab):
    with torch.no_grad():
        score=lin(feat.float()).squeeze(-1)
        pred=score.argmax(1)
        acc=float((pred==lab).float().mean().item())
        corr=score[torch.arange(score.size(0)), lab]
        masked=score.clone(); masked[torch.arange(score.size(0)), lab]=-1e9
        decmax=masked.max(1).values
        decmean=(score.sum(1)-corr)/3.0
        return dict(acc=acc, margin_max=float((corr-decmax).mean().item()), margin_mean=float((corr-decmean).mean().item()), loss=float(F.cross_entropy(score,lab).item()))


def behavior(model, probes, dev, cm_blk, prep_tok, prep_out):
    pack=S19b.eval_pack(model,probes,dev,cm_blk,prep_tok,prep_out)
    return S19b.add_drift_to_summary(S19b.metric_summary(pack),pack)


def run_one(seed,P,arm,tied_state,cfg,probes,probe_rows,dev,cm_std,cm_blk,n_train,total_epochs):
    torch.manual_seed(seed+240000+ARM_INDEX.get(arm['name'],0)*43)
    np.random.seed(seed+240000+ARM_INDEX.get(arm['name'],0))
    model=S19b.make_untied_from_tied_state(tied_state,cfg,dev)
    prep_tok=tied_state['tok.weight'].detach().clone().to(dev)
    prep_out=model.out.weight.detach().clone()
    opt=torch.optim.AdamW(model.parameters(), lr=arm['lr'], weight_decay=cfg['wd'])
    for be in range(1,total_epochs+1):
        ep=P+be
        rows=S16.make_epoch_rows(seed,ep,n_train)
        seqs=custom_seqs(rows, arm['target'])
        idx=S16.common_order(seed,ep,n_train)
        info=train_epoch(model,opt,seqs,idx,dev,cfg['bs'],cm_std,arm['kind'],arm['ctx_weight'])
    beh=behavior(model,probes,dev,cm_blk,prep_tok,prep_out)
    probe=run_probe_for_model(model,probe_rows,dev,cm_std)
    return dict(seed=seed, arm=arm['name'], target=arm['target'], kind=arm['kind'], lr=arm['lr'], ctx_weight=arm['ctx_weight'], epochs=total_epochs, train_info=info, behavior=beh, probe=probe)


def run_probe_for_model(model,probe_rows,dev,cm_std):
    ftr, ytr=collect_feats(model,probe_rows['train'],dev,cm_std)
    fva, yva=collect_feats(model,probe_rows['val'],dev,cm_std)
    fhe, yhe=collect_feats(model,probe_rows['held'],dev,cm_std)
    lin,best=train_probe(ftr,ytr,fva,yva)
    return dict(best=best, train=eval_probe(lin,ftr,ytr), val=eval_probe(lin,fva,yva), held=eval_probe(lin,fhe,yhe))


def summarize(records):
    out={}
    arms=sorted(set(r['arm'] for r in records))
    for a in arms:
        rs=[r for r in records if r['arm']==a]
        def mean(path):
            vals=[]
            for r in rs:
                x=r
                for p in path: x=x[p]
                vals.append(float(x))
            return round(float(np.mean(vals)),6)
        out[a]=dict(n=len(rs),
                    held_top4=mean(['behavior','held_top4']), held_b=mean(['behavior','held_b']), held_sel=mean(['behavior','held_sel']), train_top4=mean(['behavior','train_top4']), ctx_ce=mean(['train_info','ctx_ce']), ans_ce=mean(['train_info','ans_ce']),
                    probe_train_acc=mean(['probe','train','acc']), probe_val_acc=mean(['probe','val','acc']), probe_held_acc=mean(['probe','held','acc']), probe_held_margin=mean(['probe','held','margin_max']))
    return out

class NpEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj,(np.integer,)): return int(obj)
        if isinstance(obj,(np.floating,)): return float(obj)
        if isinstance(obj,np.ndarray): return obj.tolist()
        return super().default(obj)


def write_note(data,path):
    lines=["# research activation query-match marker probe", "", "Models are continuations from query-first answer-only preparation. A shared scalar linear probe is trained on train-entity context-attribute hidden states to select the queried entity's attribute slot, then evaluated on held-entity rows. The probe is representational evidence only, not a causal intervention.", "", "## Means", "", "| arm | epochs | held h4 | held B | train h4 | ctx CE | probe train acc | probe val acc | probe held acc | probe held margin |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a,s in data['summary'].items():
        ep=next(r['epochs'] for r in data['records'] if r['arm']==a)
        lines.append(f"| {a} | {ep} | {s['held_top4']:.3f} | {s['held_b']:+.3f} | {s['train_top4']:.3f} | {s['ctx_ce']:.3f} | {s['probe_train_acc']:.3f} | {s['probe_val_acc']:.3f} | {s['probe_held_acc']:.3f} | {s['probe_held_margin']:+.3f} |")
    lines += ["", "## Per-seed", "", "| seed | arm | held h4 | held B | probe held acc | probe held margin |", "|---:|---|---:|---:|---:|---:|"]
    for r in sorted(data['records'], key=lambda x:(x['seed'],x['arm'])):
        lines.append(f"| {r['seed']} | {r['arm']} | {r['behavior']['held_top4']:.3f} | {r['behavior']['held_b']:+.3f} | {r['probe']['held']['acc']:.3f} | {r['probe']['held']['margin_max']:+.3f} |")
    lines += ["", "## Interpretation", "", "If the probe accuracy is high in static_1over17 and low in direct/context_only/slot0, behavioral preservation is associated with retention of a linearly accessible query-match signal at context attribute positions. If the probe remains high in collapsed arms, the marker may be present but decoupled from the answer readout; that would require causal readout or activation interventions. This note records the observed association but does not by itself prove causality.", ""]
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text("\n".join(lines))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir',default='experiments/archive/functional_learning/data/revision_019b_embedding_role_decomposition/checkpoints')
    ap.add_argument('--data',default='experiments/archive/functional_learning/data/activation_marker_probe')
    ap.add_argument('--note',default='research/notes/functional_learning/activation_marker_probe.md')
    ap.add_argument('--seeds',default='43,100')
    ap.add_argument('--arms',default='direct_full,static_1over17,context_only_lr_half,slot0_static_1over17')
    ap.add_argument('--epochs',type=int,default=250)
    ap.add_argument('--n-train',type=int,default=500)
    ap.add_argument('--probe-seed',type=int,default=180018)
    ap.add_argument('--smoke',action='store_true')
    A=ap.parse_args()
    if A.smoke:
        A.seeds='100'; A.arms='direct_full,static_1over17'; A.epochs=5; A.n_train=64; n_std,n_small=96,48; npr=256; nval=128; nheld=128
    else:
        n_std,n_small=512,256; npr=2048; nval=512; nheld=512
    seeds=[int(x) for x in A.seeds.split(',') if x.strip()]
    want={x.strip() for x in A.arms.split(',') if x.strip()}
    arms=[copy.deepcopy(a) for a in ARMS if a['name'] in want]
    cfg=dict(d=64,nh=2,nl=3,lr=3e-4,wd=0.01,bs=64)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cm_std=S17.standard_causal_mask(SL-1,dev); cm_blk=S17.block_query_ctx_mask(SL-1,dev)
    probes=S18.make_probes(seed=A.probe_seed,n_std=n_std,n_small=n_small)
    probe_rows=dict(train=gen_probe_rows(A.probe_seed+1,npr,'train'), val=gen_probe_rows(A.probe_seed+2,nval,'train'), held=gen_probe_rows(A.probe_seed+3,nheld,'held'))
    ckpt_dir=Path(A.checkpoint_dir)
    records=[]; t0=time.time()
    print(f"Device: {dev}; seeds={seeds}; arms={[a['name'] for a in arms]}; epochs={A.epochs}", flush=True)
    for sd in seeds:
        P=DEFAULT_PREP_EPOCHS[sd]
        tied_state=torch.load(ckpt_dir/f'seed{sd}_prep_tied.pt',map_location=dev)
        # prep baseline
        model=S19b.make_untied_from_tied_state(tied_state,cfg,dev)
        prep_tok=tied_state['tok.weight'].detach().clone().to(dev); prep_out=model.out.weight.detach().clone()
        beh=behavior(model,probes,dev,cm_blk,prep_tok,prep_out)
        pr=run_probe_for_model(model,probe_rows,dev,cm_std)
        records.append(dict(seed=sd,arm='prep',target='bound',kind='none',lr=0.0,ctx_weight=0.0,epochs=0,train_info=dict(ans_ce=0.0,ctx_ce=0.0,wloss=0.0),behavior=beh,probe=pr))
        print(f"seed={sd} prep: held {beh['held_top4']:.3f}/{beh['held_b']:+.2f} probe_held={pr['held']['acc']:.3f}", flush=True)
        for arm in arms:
            rec=run_one(sd,P,arm,tied_state,cfg,probes,probe_rows,dev,cm_std,cm_blk,A.n_train,A.epochs)
            records.append(rec)
            print(f"seed={sd} {arm['name']}: held {rec['behavior']['held_top4']:.3f}/{rec['behavior']['held_b']:+.2f} probe_held={rec['probe']['held']['acc']:.3f} ctx={rec['train_info']['ctx_ce']:.3f}", flush=True)
    data=dict(config=dict(seeds=seeds,arms=[a['name'] for a in arms],epochs=A.epochs,n_train=A.n_train,probe_seed=A.probe_seed,n_probe_train=npr,n_probe_val=nval,n_probe_held=nheld), records=records)
    data['summary']=summarize(records)
    out_dir=Path(A.data); out_dir.mkdir(parents=True,exist_ok=True)
    outp=out_dir/'results.json'; outp.write_text(json.dumps(data,indent=2,cls=NpEncoder))
    write_note(data,A.note)
    print(json.dumps({'status':'ok','out':str(outp),'note':A.note,'elapsed':round(time.time()-t0,1)},indent=2))

if __name__=='__main__': main()
