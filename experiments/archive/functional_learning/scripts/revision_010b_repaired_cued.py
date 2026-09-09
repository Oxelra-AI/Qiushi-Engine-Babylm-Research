#!/usr/bin/env python3
"""Step010b: Repaired cued experiment with NEUTRAL_CUE for neutral arm.

Fixes the confound identified by independent review verification: in the original Part B,
neutral's 400 sub-sequences used REWRITE_CUE + random RWT, which actively
corrupted the REWRITE_CUE → correct RWT conditional learned from backbone.

Fix: add NEUTRAL_CUE (token 35) for neutral/wrong sub-sequences.
Now the model's REWRITE_CUE conditional is based only on backbone correspondence
(same for all arms), and ident's COPY_CUE teaches a separate relation.

Also adds a CONSTANT_CUE control: same extra token for ALL sequences,
providing no task-type information. If facilitation survives with constant cue,
it is not due to task specification.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, argparse
from pathlib import Path

BOS, PAD, SEP, HAS, IS = 0, 1, 2, 3, 4
N_ENT, N_ATTR = 8, 10
ENT  = lambda e: 5 + e
SRC  = lambda a: 13 + a
RWT  = lambda a: 23 + a
COPY_CUE, REWRITE_CUE, NEUTRAL_CUE, CONST_CUE = 33, 34, 35, 36
VOCAB = 37
TRAIN_E = list(range(6)); HELD_E = [6,7]; ALL_E = list(range(8))
N_CTX = 4
SRC_TOKS = list(range(13,23)); RWT_TOKS = list(range(23,33))
SL = 19  # BOS 4*(E HAS S) SEP E CUE IS TGT PAD

def mk_seq(ce, ca, qi, cue, tgt):
    s = [BOS]
    for i in range(N_CTX): s += [ENT(ce[i]), HAS, SRC(ca[i])]
    s += [SEP, ENT(ce[qi]), cue, IS, tgt, PAD]
    return s[:SL]

class DG:
    def __init__(self, seed): self.rng = np.random.default_rng(seed)
    def _ctx(self):
        while True:
            ce = self.rng.choice(ALL_E, N_CTX, replace=False).tolist()
            c = [i for i,e in enumerate(ce) if e in TRAIN_E]
            if c:
                qi = int(self.rng.choice(c))
                ca = [int(self.rng.integers(N_ATTR)) for _ in range(N_CTX)]
                return ce, ca, qi

    def gen(self, n, kind, cue_mode="typed"):
        """cue_mode: 'typed' uses COPY/REWRITE/NEUTRAL per kind; 'constant' uses CONST_CUE for all."""
        seqs = []
        for _ in range(n):
            ce, ca, qi = self._ctx()
            a = ca[qi]
            if kind == "corr":
                tgt = RWT(a)
                cue = REWRITE_CUE if cue_mode == "typed" else CONST_CUE
            elif kind == "ident":
                tgt = SRC(a)
                cue = COPY_CUE if cue_mode == "typed" else CONST_CUE
            elif kind == "neutral":
                tgt = RWT(int(self.rng.integers(N_ATTR)))
                cue = NEUTRAL_CUE if cue_mode == "typed" else CONST_CUE
            else:
                raise ValueError(kind)
            seqs.append(mk_seq(ce, ca, qi, cue, tgt))
        return seqs

    def held_probes(self, n_per=60, cue=REWRITE_CUE):
        out = []
        for he in HELD_E:
            for _ in range(n_per):
                a = int(self.rng.integers(N_ATTR))
                oth = self.rng.choice(TRAIN_E, N_CTX-1, replace=False).tolist()
                ce = oth + [he]; ce = self.rng.permutation(ce).tolist()
                ca, qi = [], -1
                for i,e in enumerate(ce):
                    if e==he: ca.append(a); qi=i
                    else: ca.append(int(self.rng.integers(N_ATTR)))
                ssrc = mk_seq(ce, ca, qi, cue, PAD)
                nce = self.rng.choice(TRAIN_E, N_CTX, replace=False).tolist()
                nca = [int(self.rng.integers(N_ATTR)) for _ in range(N_CTX)]
                ns = [BOS]
                for i in range(N_CTX): ns += [ENT(nce[i]),HAS,SRC(nca[i])]
                ns += [SEP,ENT(he),cue,IS,PAD]; ns=(ns+[PAD]*SL)[:SL]
                out.append({"ss":ssrc,"ns":ns,"ct":RWT(a),"cp":SRC(a)})
        return out

class Blk(nn.Module):
    def __init__(self,d,nh):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,nh,batch_first=True,dropout=0)
        self.ln1=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)); self.ln2=nn.LayerNorm(d)
    def forward(self,x,m):
        h=self.ln1(x); h,_=self.attn(h,h,h,attn_mask=m); x=x+h; return x+self.ff(self.ln2(x))

class CLM(nn.Module):
    def __init__(self,V,d,nh,nl,ml):
        super().__init__()
        self.tok=nn.Embedding(V,d); self.pos=nn.Embedding(ml,d)
        self.blks=nn.ModuleList([Blk(d,nh) for _ in range(nl)]); self.ln=nn.LayerNorm(d)
    def forward(self,x,m=None):
        B,L=x.shape; h=self.tok(x)+self.pos(torch.arange(L,device=x.device))
        for b in self.blks: h=b(h,m)
        return self.ln(h) @ self.tok.weight.T

def causal(L,dev="cpu"): return torch.triu(torch.ones(L,L,dtype=torch.bool,device=dev),diagonal=1)

def train_ep(model, opt, seqs_t, dev, V, bs=64):
    model.train(); N=seqs_t.size(0); L=seqs_t.size(1)-1
    idx=torch.randperm(N); seqs_t=seqs_t[idx]
    tl,tt=0.,0; cm=causal(L,dev)
    for i in range(0,N,bs):
        batch=seqs_t[i:i+bs].to(dev); inp,tgt=batch[:,:-1],batch[:,1:]
        logits=model(inp,cm); lm=(tgt!=PAD).float()
        loss=F.cross_entropy(logits.reshape(-1,V),tgt.reshape(-1),reduction='none')
        loss=(loss.view(tgt.shape)*lm).sum()/max(lm.sum(),1)
        opt.zero_grad(); loss.backward(); opt.step()
        tl+=loss.item()*lm.sum().item(); tt+=lm.sum().item()
    return tl/max(tt,1)

IS_POS = 16  # position where model predicts the target

def eval_decomp(model, probes, dev, V, src=True):
    model.eval(); key="ss" if src else "ns"
    seqs=[p[key] for p in probes]; tgts=[(p["ct"],p["cp"]) for p in probes]
    st=torch.tensor(seqs,dtype=torch.long,device=dev); L=st.size(1)-1; cm=causal(L,dev)
    cnll,cpnll,prwt,psrc,wnll,fnll=[],[],[],[],[],[]
    # Also track top-1 accuracy and MRR within RWT family
    top1_correct, mrr_vals = [], []
    with torch.no_grad():
        for i in range(0,len(seqs),128):
            b=st[i:i+128,:L]; logits=model(b,cm)
            probs=F.softmax(logits[:,IS_POS,:],dim=-1); lp=F.log_softmax(logits[:,IS_POS,:],dim=-1)
            for j in range(b.size(0)):
                ct,cp=tgts[i+j]
                cnll.append(-lp[j,ct].item()); cpnll.append(-lp[j,cp].item())
                pr=probs[j,RWT_TOKS].sum().item(); ps=probs[j,SRC_TOKS].sum().item()
                prwt.append(pr); psrc.append(ps)
                wnll.append(-np.log(max(probs[j,ct].item()/max(pr,1e-30),1e-30)))
                fnll.append(-np.log(max(pr,1e-30)))
                # Top-1 accuracy within RWT family
                rwt_probs = probs[j, RWT_TOKS]
                top1_correct.append(int(rwt_probs.argmax().item() == (ct - 23)))
                # MRR within family
                sorted_idx = rwt_probs.argsort(descending=True)
                rank = (sorted_idx == (ct - 23)).nonzero(as_tuple=True)[0].item() + 1
                mrr_vals.append(1.0 / rank)
    return {"cnll":float(np.mean(cnll)),"cpnll":float(np.mean(cpnll)),
            "prwt":float(np.mean(prwt)),"psrc":float(np.mean(psrc)),
            "wnll":float(np.mean(wnll)),"fnll":float(np.mean(fnll)),
            "top1":float(np.mean(top1_correct)),"mrr":float(np.mean(mrr_vals))}

ARMS = ["all_corr","ident_full","neutral"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",nargs="+",type=int,default=[42,43,100])
    ap.add_argument("--epochs",type=int,default=300)
    ap.add_argument("--backbone_n",type=int,default=100)
    ap.add_argument("--sub_n",type=int,default=400)
    ap.add_argument("--eval_every",type=int,default=25)
    ap.add_argument("--d",type=int,default=64)
    ap.add_argument("--nh",type=int,default=2)
    ap.add_argument("--nl",type=int,default=3)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--out",type=str,default="data/revision_010b_repaired_cued")
    ap.add_argument("--smoke",action="store_true")
    A = ap.parse_args()
    if A.smoke: A.seeds=[42]; A.epochs=10; A.eval_every=5

    dev="cuda" if torch.cuda.is_available() else "cpu"
    od=Path(A.out); od.mkdir(parents=True, exist_ok=True)
    t0=time.time()
    all_data={"config":vars(A)}

    # Run two cue modes: "typed" (repaired with NEUTRAL_CUE) and "constant" (no task info)
    for cue_mode in ["typed", "constant"]:
        mkey = f"cue_{cue_mode}"
        all_data[mkey] = {}
        for seed in A.seeds:
            print(f"\n{'='*50}\nSeed {seed} cue_mode={cue_mode}\n{'='*50}",flush=True)
            dg_bb = DG(seed)
            bb_seqs = dg_bb.gen(A.backbone_n, "corr", cue_mode)
            eval_cue = REWRITE_CUE if cue_mode == "typed" else CONST_CUE
            hprobes = dg_bb.held_probes(60, cue=eval_cue)

            sub_data = {}
            sub_data["all_corr"] = DG(seed*1000+2).gen(A.sub_n, "corr", cue_mode)
            sub_data["ident_full"] = DG(seed*1000+1).gen(A.sub_n, "ident", cue_mode)
            sub_data["neutral"] = DG(seed*1000+3).gen(A.sub_n, "neutral", cue_mode)

            torch.manual_seed(seed)
            model = CLM(VOCAB, A.d, A.nh, A.nl, SL).to(dev)
            init_sd = {k:v.clone() for k,v in model.state_dict().items()}

            seed_res = {}
            for arm in ARMS:
                t1 = time.time()
                all_seqs = bb_seqs + sub_data[arm]
                seqs_t = torch.tensor(all_seqs, dtype=torch.long)
                model.load_state_dict(init_sd)
                opt = torch.optim.AdamW(model.parameters(), lr=A.lr)
                curves = []
                for ep in range(1, A.epochs+1):
                    tl = train_ep(model, opt, seqs_t, dev, VOCAB)
                    if ep % A.eval_every == 0 or ep == 1:
                        ds = eval_decomp(model, hprobes, dev, VOCAB, True)
                        dn = eval_decomp(model, hprobes, dev, VOCAB, False)
                        row = {"e":ep,"tl":round(tl,5),
                               "hc_s":round(ds["cnll"],5),"hcp_s":round(ds["cpnll"],5),
                               "hc_n":round(dn["cnll"],5),"hcg":round(dn["cnll"]-ds["cnll"],5),
                               "prwt_s":round(ds["prwt"],5),"psrc_s":round(ds["psrc"],5),
                               "within_s":round(ds["wnll"],5),"family_s":round(ds["fnll"],5),
                               "prwt_n":round(dn["prwt"],5),
                               "within_n":round(dn["wnll"],5),"family_n":round(dn["fnll"],5),
                               "top1_s":round(ds["top1"],5),"mrr_s":round(ds["mrr"],5),
                               "top1_n":round(dn["top1"],5),"mrr_n":round(dn["mrr"],5)}
                        curves.append(row)
                elapsed = time.time()-t1
                seed_res[arm] = curves
                f = curves[-1]
                print(f"  {arm:12s} hc_s={f['hc_s']:.2f} hcg={f['hcg']:+.2f} "
                      f"within={f['within_s']:.2f} family={f['family_s']:.2f} "
                      f"top1={f['top1_s']:.3f} mrr={f['mrr_s']:.3f} ({elapsed:.1f}s)",flush=True)
            all_data[mkey][f"seed{seed}"] = seed_res

    # Summary
    summary = {}
    for mkey in [k for k in all_data if k.startswith("cue_")]:
        summary[mkey] = {}
        for arm in ARMS:
            finals = [all_data[mkey][f"seed{s}"][arm][-1] for s in A.seeds]
            avg = {}
            for k in finals[0]:
                if k=="e": avg[k]=finals[0][k]; continue
                vs=[f[k] for f in finals]
                avg[k]=round(np.mean(vs),5); avg[k+"_std"]=round(np.std(vs),5)
            summary[mkey][arm] = avg
        # Contrasts
        for m in ["hcg","hc_s","within_s","family_s","prwt_s","top1_s","mrr_s"]:
            iv=[all_data[mkey][f"seed{s}"]["ident_full"][-1][m] for s in A.seeds]
            nv=[all_data[mkey][f"seed{s}"]["neutral"][-1][m] for s in A.seeds]
            d=np.array(iv)-np.array(nv)
            summary[mkey][f"ident_minus_neutral_{m}"] = {
                "delta":round(float(np.mean(d)),5),"std":round(float(np.std(d)),5)}

    all_data["summary"] = summary
    print(f"\n{'='*70}\nFINAL SUMMARY (elapsed {time.time()-t0:.1f}s)\n{'='*70}")
    for mkey in sorted(summary):
        print(f"\n--- {mkey} ---")
        for arm in ARMS:
            a=summary[mkey][arm]
            print(f"  {arm:12s}  hc_s={a['hc_s']:.3f}±{a['hc_s_std']:.3f}  "
                  f"hcg={a['hcg']:+.3f}  within={a['within_s']:.3f}  "
                  f"family={a['family_s']:.3f}  top1={a['top1_s']:.3f}  mrr={a['mrr_s']:.3f}")
        for cn in sorted(k for k in summary[mkey] if k.startswith("ident_minus")):
            cv = summary[mkey][cn]
            print(f"  {cn}: {cv['delta']:+.4f} ± {cv['std']:.4f}")

    with open(od/"results.json","w") as f: json.dump(all_data,f,indent=2)
    print(json.dumps({"status":"STEP010B_DONE","out":str(od),"elapsed":round(time.time()-t0,1)}))

if __name__=="__main__":
    main()
