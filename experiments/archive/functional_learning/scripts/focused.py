#!/usr/bin/env python3
"""research focused: Family decomposition + task-cued for key arms only.
Runs Part A and Part B for: ident_full, neutral, all_corr.
Evaluates every 20 epochs to reduce overhead.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, argparse, sys
from pathlib import Path

BOS, PAD, SEP, HAS, IS = 0, 1, 2, 3, 4
N_ENT, N_ATTR = 8, 10
ENT  = lambda e: 5 + e
SRC  = lambda a: 13 + a
RWT  = lambda a: 23 + a
VOCAB_BASE = 33
COPY_CUE, REWRITE_CUE = 33, 34
VOCAB_CUED = 35
TRAIN_E = list(range(6)); HELD_E = [6,7]; ALL_E = list(range(8))
N_CTX = 4
SRC_TOKS = list(range(13,23)); RWT_TOKS = list(range(23,33))

def mk_seq_a(ce, ca, qi, tgt):
    s = [BOS]
    for i in range(N_CTX): s += [ENT(ce[i]), HAS, SRC(ca[i])]
    s += [SEP, ENT(ce[qi]), IS, tgt, PAD]
    return s[:18]

def mk_seq_b(ce, ca, qi, cue, tgt):
    s = [BOS]
    for i in range(N_CTX): s += [ENT(ce[i]), HAS, SRC(ca[i])]
    s += [SEP, ENT(ce[qi]), cue, IS, tgt, PAD]
    return s[:19]

class DG:
    def __init__(self, seed): self.rng = np.random.default_rng(seed)
    def _ctx(self, qp=TRAIN_E):
        while True:
            ce = self.rng.choice(ALL_E, N_CTX, replace=False).tolist()
            c = [i for i,e in enumerate(ce) if e in qp]
            if c:
                qi = int(self.rng.choice(c))
                ca = [int(self.rng.integers(N_ATTR)) for _ in range(N_CTX)]
                return ce, ca, qi
    def gen(self, n, kind, part):
        seqs, infos = [], []
        for _ in range(n):
            ce, ca, qi = self._ctx()
            a = ca[qi]
            if kind=="corr": tgt,cue = RWT(a),REWRITE_CUE
            elif kind=="ident": tgt,cue = SRC(a),COPY_CUE
            elif kind=="neutral": tgt,cue = RWT(int(self.rng.integers(N_ATTR))),REWRITE_CUE
            else: raise ValueError(kind)
            if part=="A": seqs.append(mk_seq_a(ce,ca,qi,tgt))
            else: seqs.append(mk_seq_b(ce,ca,qi,cue,tgt))
            infos.append({"t":kind,"sp":3+qi*3})
        return seqs, infos
    def held_probes(self, n_per=60, part="A"):
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
                if part=="A": ssrc = mk_seq_a(ce,ca,qi,PAD)
                else: ssrc = mk_seq_b(ce,ca,qi,REWRITE_CUE,PAD)
                nce = self.rng.choice(TRAIN_E, N_CTX, replace=False).tolist()
                nca = [int(self.rng.integers(N_ATTR)) for _ in range(N_CTX)]
                ns = [BOS]
                for i in range(N_CTX): ns += [ENT(nce[i]),HAS,SRC(nca[i])]
                if part=="A": ns+=[SEP,ENT(he),IS,PAD]; ns=(ns+[PAD]*18)[:18]
                else: ns+=[SEP,ENT(he),REWRITE_CUE,IS,PAD]; ns=(ns+[PAD]*19)[:19]
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
        self.blks=nn.ModuleList([Blk(d,nh) for _ in range(nl)]); self.ln=nn.LayerNorm(d); self.nh=nh
    def forward(self,x,m=None):
        B,L=x.shape; h=self.tok(x)+self.pos(torch.arange(L,device=x.device))
        for b in self.blks: h=b(h,m)
        return self.ln(h) @ self.tok.weight.T

def causal(L,dev="cpu"): return torch.triu(torch.ones(L,L,dtype=torch.bool,device=dev),diagonal=1)

def train_ep(model, opt, seqs_t, dev, nh, V, bs=64):
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

def eval_decomp(model, probes, dev, nh, V, is_pos, src=True):
    model.eval(); key="ss" if src else "ns"
    seqs=[p[key] for p in probes]; tgts=[(p["ct"],p["cp"]) for p in probes]
    st=torch.tensor(seqs,dtype=torch.long,device=dev); L=st.size(1)-1; cm=causal(L,dev)
    cnll,cpnll,prwt,psrc,wnll,fnll=[],[],[],[],[],[]
    with torch.no_grad():
        for i in range(0,len(seqs),128):
            b=st[i:i+128,:L]; logits=model(b,cm)
            probs=F.softmax(logits[:,is_pos,:],dim=-1); lp=F.log_softmax(logits[:,is_pos,:],dim=-1)
            for j in range(b.size(0)):
                ct,cp=tgts[i+j]
                cnll.append(-lp[j,ct].item()); cpnll.append(-lp[j,cp].item())
                pr=probs[j,RWT_TOKS].sum().item(); ps=probs[j,SRC_TOKS].sum().item()
                prwt.append(pr); psrc.append(ps)
                wnll.append(-np.log(max(probs[j,ct].item()/max(pr,1e-30),1e-30)))
                fnll.append(-np.log(max(pr,1e-30)))
    return {"cnll":float(np.mean(cnll)),"cpnll":float(np.mean(cpnll)),
            "prwt":float(np.mean(prwt)),"psrc":float(np.mean(psrc)),
            "wnll":float(np.mean(wnll)),"fnll":float(np.mean(fnll))}

ARMS = ["all_corr","ident_full","neutral"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",nargs="+",type=int,default=[42,43,100])
    ap.add_argument("--epochs",type=int,default=300)
    ap.add_argument("--backbone_n",type=int,default=100)
    ap.add_argument("--sub_n",type=int,default=400)
    ap.add_argument("--eval_every",type=int,default=20)
    ap.add_argument("--d",type=int,default=64)
    ap.add_argument("--nh",type=int,default=2)
    ap.add_argument("--nl",type=int,default=3)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--out",type=str,default="data/decomp_cued")
    ap.add_argument("--part",type=str,default="both",choices=["A","B","both"])
    ap.add_argument("--smoke",action="store_true")
    A = ap.parse_args()
    if A.smoke: A.seeds=[42]; A.epochs=10; A.eval_every=5

    dev="cuda" if torch.cuda.is_available() else "cpu"
    od=Path(A.out); od.mkdir(parents=True,exist_ok=True)
    t0_all=time.time()

    all_data={"config":vars(A)}
    parts = ["A","B"] if A.part=="both" else [A.part]

    for part in parts:
        pkey = f"part_{part}"
        all_data[pkey] = {}
        V = VOCAB_BASE if part=="A" else VOCAB_CUED
        SL = 18 if part=="A" else 19
        IS_POS = 15 if part=="A" else 16

        for seed in A.seeds:
            print(f"\n{'='*50}\nSeed {seed} Part {part} (V={V})\n{'='*50}",flush=True)
            dg_bb = DG(seed)
            bb_seqs, bb_infos = dg_bb.gen(A.backbone_n, "corr", part)
            hprobes = dg_bb.held_probes(60, part)

            sub_data = {}
            sub_data["all_corr"] = DG(seed*1000+2).gen(A.sub_n, "corr", part)
            sub_data["ident_full"] = DG(seed*1000+1).gen(A.sub_n, "ident", part)
            sub_data["neutral"] = DG(seed*1000+3).gen(A.sub_n, "neutral", part)

            torch.manual_seed(seed)
            model = CLM(V, A.d, A.nh, A.nl, SL).to(dev)
            init_sd = {k:v.clone() for k,v in model.state_dict().items()}

            seed_res = {}
            for arm in ARMS:
                t0 = time.time()
                s_seqs, _ = sub_data[arm]
                all_seqs = bb_seqs + s_seqs
                seqs_t = torch.tensor(all_seqs, dtype=torch.long)

                model.load_state_dict(init_sd)
                opt = torch.optim.AdamW(model.parameters(), lr=A.lr)

                curves = []
                for ep in range(1, A.epochs+1):
                    tl = train_ep(model, opt, seqs_t, dev, A.nh, V)
                    if ep % A.eval_every == 0 or ep == 1:
                        ds = eval_decomp(model, hprobes, dev, A.nh, V, IS_POS, True)
                        dn = eval_decomp(model, hprobes, dev, A.nh, V, IS_POS, False)
                        row = {"e":ep,"tl":round(tl,5),
                               "hc_s":round(ds["cnll"],5),"hcp_s":round(ds["cpnll"],5),
                               "hc_n":round(dn["cnll"],5),"hcg":round(dn["cnll"]-ds["cnll"],5),
                               "prwt_s":round(ds["prwt"],5),"psrc_s":round(ds["psrc"],5),
                               "within_s":round(ds["wnll"],5),"family_s":round(ds["fnll"],5),
                               "prwt_n":round(dn["prwt"],5),
                               "within_n":round(dn["wnll"],5),"family_n":round(dn["fnll"],5)}
                        curves.append(row)
                elapsed = time.time()-t0
                seed_res[arm] = curves
                f = curves[-1]
                print(f"  {arm:12s} hcg={f['hcg']:+.3f} hc_s={f['hc_s']:.2f} "
                      f"prwt={f['prwt_s']:.4f} within={f['within_s']:.2f} "
                      f"family={f['family_s']:.2f} ({elapsed:.1f}s)",flush=True)
            all_data[pkey][f"seed{seed}"] = seed_res

    # ─── Aggregate & contrasts ──────────────────────────────────
    summary = {}
    for pkey in [k for k in all_data if k.startswith("part_")]:
        summary[pkey] = {}
        for arm in ARMS:
            finals = [all_data[pkey][f"seed{s}"][arm][-1] for s in A.seeds]
            avg = {}
            for k in finals[0]:
                if k=="e": avg[k]=finals[0][k]; continue
                vs = [f[k] for f in finals]
                avg[k]=round(np.mean(vs),5); avg[k+"_std"]=round(np.std(vs),5)
            summary[pkey][arm] = avg

        # Ident vs neutral contrast
        contrasts = {}
        for m in ["hcg","hc_s","within_s","family_s","prwt_s"]:
            iv = [all_data[pkey][f"seed{s}"]["ident_full"][-1][m] for s in A.seeds]
            nv = [all_data[pkey][f"seed{s}"]["neutral"][-1][m] for s in A.seeds]
            d = np.array(iv)-np.array(nv)
            contrasts[f"ident_minus_neutral_{m}"] = {
                "delta":round(float(np.mean(d)),5),"std":round(float(np.std(d)),5)}
        # All_corr vs neutral
        for m in ["hcg","hc_s","within_s","family_s","prwt_s"]:
            iv = [all_data[pkey][f"seed{s}"]["all_corr"][-1][m] for s in A.seeds]
            nv = [all_data[pkey][f"seed{s}"]["neutral"][-1][m] for s in A.seeds]
            d = np.array(iv)-np.array(nv)
            contrasts[f"allcorr_minus_neutral_{m}"] = {
                "delta":round(float(np.mean(d)),5),"std":round(float(np.std(d)),5)}
        summary[pkey]["contrasts"] = contrasts

    all_data["summary"] = summary

    print(f"\n{'='*70}\nFINAL SUMMARY (elapsed {time.time()-t0_all:.1f}s)\n{'='*70}")
    for pkey in sorted(summary):
        label = "UNCUED" if "A" in pkey else "CUED"
        print(f"\n--- {label} ({pkey}) ---")
        for arm in ARMS:
            a = summary[pkey][arm]
            print(f"  {arm:12s}  hcg={a['hcg']:+.4f}±{a['hcg_std']:.4f}  "
                  f"hc_s={a['hc_s']:.3f}±{a['hc_s_std']:.3f}  "
                  f"prwt={a['prwt_s']:.4f}  within={a['within_s']:.3f}  family={a['family_s']:.3f}")
        print(f"  --- Contrasts ---")
        for cn, cv in summary[pkey].get("contrasts",{}).items():
            print(f"  {cn}: {cv['delta']:+.4f} ± {cv['std']:.4f}")

    with open(od/"results.json","w") as f:
        json.dump(all_data,f,indent=2)
    print(json.dumps({"status":"DONE","out":str(od),"elapsed":round(time.time()-t0_all,1)}))

if __name__=="__main__":
    main()
