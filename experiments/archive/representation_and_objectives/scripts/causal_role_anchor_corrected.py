#!/usr/bin/env python3
"""research corrected causal role-anchor experiment.

This is the cheap repair requested by independent review verifier: remove world/domain markers,
make the composition label depend on BOTH context and hypothesis predicate role
maps, use atomic predicate tokens with identical syntax, keep held predicates out
of composition training, and compare true anchors against exposure-only and
shuffled anchors.

It is a controlled synthetic mechanism test, not natural-source training. Its job
is to decide whether the research "role-word anchor" positive survives the main
code defect in the earlier held-predicate experiment.
"""
from __future__ import annotations

import argparse, collections, copy, json, math, os, random, re, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

OUT_DIR = Path("experiments/archive/representation_and_objectives/data/causal_role_anchor_corrected")

SEEN = ["dax", "feg", "mip", "lor"]
HELD = ["zup", "niv", "kem", "rox"]
# +1 = subject bears ROLE+ (winner/source); -1 = object bears ROLE+.
ROLE_MAP = {"dax": 1, "feg": 1, "mip": -1, "lor": -1, "zup": 1, "niv": 1, "kem": -1, "rox": -1}
SHUFFLED_HELD_MAP = {p: -ROLE_MAP[p] for p in HELD}
DUMMY_PREDS = ["tav", "gop", "sul", "bem"]


def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def toks(s: str) -> list[str]: return re.findall(r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]", s.lower())

def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

def write_jsonl(p: Path, rows: list[dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")


def roleplus_entity(pred: str, orient: str, role_map: dict[str, int]) -> str:
    subj, obj = ("ENTITY_A", "ENTITY_B") if orient == "AB" else ("ENTITY_B", "ENTITY_A")
    return subj if role_map[pred] == 1 else obj


def comp_label(cp: str, c_orient: str, hp: str, h_orient: str, role_map: dict[str, int]) -> int:
    return int(roleplus_entity(cp, c_orient, role_map) == roleplus_entity(hp, h_orient, role_map))


def sent(pred: str, orient: str, case: int, distractors: bool = True) -> str:
    subj, obj = ("ENTITY_A", "ENTITY_B") if orient == "AB" else ("ENTITY_B", "ENTITY_A")
    tail = " Other named people were ENTITY_C and ENTITY_D." if distractors else ""
    return f"Case {case}: {subj} {pred} {obj}.{tail}"


def hyp(pred: str, orient: str) -> str:
    subj, obj = ("ENTITY_A", "ENTITY_B") if orient == "AB" else ("ENTITY_B", "ENTITY_A")
    return f"{subj} {pred} {obj}."


def anchor_hyp(entity: str, role: str) -> str:
    return f"{entity} was the {role}."


def mentioned_hyp(entity: str) -> str:
    return f"{entity} was mentioned."


def filler_text(token: str, case: int) -> str:
    return f"Case {case}: ENTITY_A {token} ENTITY_B. Other named people were ENTITY_C and ENTITY_D."


def make_seen_composition(n_cases: int) -> list[dict[str, Any]]:
    rows=[]
    for case in range(n_cases):
        for cp in SEEN:
            for hp in SEEN:
                for co in ["AB","BA"]:
                    for ho in ["AB","BA"]:
                        rows.append({"id":f"seencomp_{case}_{cp}_{co}_{hp}_{ho}","kind":"seen_composition","cp":cp,"hp":hp,"co":co,"ho":ho,
                                     "text":sent(cp,co,case)+" [SEP] "+hyp(hp,ho),"label":comp_label(cp,co,hp,ho,ROLE_MAP)})
    return rows


def make_composition_eval(cps: list[str], hps: list[str], n_cases: int, kind: str) -> list[dict[str, Any]]:
    rows=[]
    for case in range(1000, 1000+n_cases):
        for cp in cps:
            for hp in hps:
                for co in ["AB","BA"]:
                    for ho in ["AB","BA"]:
                        rows.append({"id":f"{kind}_{case}_{cp}_{co}_{hp}_{ho}","kind":kind,"cp":cp,"hp":hp,"co":co,"ho":ho,
                                     "text":sent(cp,co,case)+" [SEP] "+hyp(hp,ho),"label":comp_label(cp,co,hp,ho,ROLE_MAP)})
    return rows


def make_anchor_rows(preds: list[str], n_cases: int, kind: str, map_for_labels: dict[str, int]) -> list[dict[str, Any]]:
    rows=[]
    for case in range(2000, 2000+n_cases):
        for p in preds:
            for orient in ["AB","BA"]:
                rp = roleplus_entity(p, orient, map_for_labels)
                rm = "ENTITY_B" if rp == "ENTITY_A" else "ENTITY_A"
                for ent, role, lab in [(rp,"winner",1),(rm,"winner",0),(rp,"loser",0),(rm,"loser",1)]:
                    rows.append({"id":f"{kind}_{case}_{p}_{orient}_{ent}_{role}","kind":kind,"pred":p,"orient":orient,
                                 "text":sent(p,orient,case)+" [SEP] "+anchor_hyp(ent,role),"label":lab})
    return rows


def make_exposure_rows(preds: list[str], n_cases: int, kind: str) -> list[dict[str, Any]]:
    rows=[]
    for case in range(3000, 3000+n_cases):
        for p in preds:
            for orient in ["AB","BA"]:
                # Same predicate/context exposure, but the labels are about
                # mention membership, not roles. Balanced and true from text.
                for ent, lab in [("ENTITY_A",1),("ENTITY_B",1),("ENTITY_E",0),("ENTITY_F",0)]:
                    rows.append({"id":f"{kind}_{case}_{p}_{orient}_{ent}","kind":kind,"pred":p,"orient":orient,
                                 "text":sent(p,orient,case)+" [SEP] "+mentioned_hyp(ent),"label":lab})
    return rows


def make_filler_rows(n_rows: int, kind: str) -> list[dict[str, Any]]:
    rows=[]
    for i in range(n_rows):
        token=DUMMY_PREDS[i % len(DUMMY_PREDS)]
        ent = ["ENTITY_A","ENTITY_B","ENTITY_E","ENTITY_F"][i % 4]
        lab = 1 if ent in {"ENTITY_A","ENTITY_B"} else 0
        rows.append({"id":f"{kind}_{i}","kind":kind,"pred":token,
                     "text":filler_text(token,4000+i)+" [SEP] "+mentioned_hyp(ent),"label":lab})
    return rows


def build_arm(arm: str, n_comp_cases: int, n_anchor_cases: int) -> list[dict[str, Any]]:
    base = make_seen_composition(n_comp_cases)
    seen_anchor = make_anchor_rows(SEEN, n_anchor_cases, "seen_anchor", ROLE_MAP)
    held_anchor_n = len(make_anchor_rows(HELD, n_anchor_cases, "tmp", ROLE_MAP))
    if arm == "noheld_filler":
        extra = make_filler_rows(held_anchor_n, "matched_filler")
    elif arm == "exposure_only":
        extra = make_exposure_rows(HELD, n_anchor_cases, "held_exposure_only")
    elif arm == "true_anchor":
        extra = make_anchor_rows(HELD, n_anchor_cases, "held_true_anchor", ROLE_MAP)
    elif arm == "shuffled_anchor":
        extra = make_anchor_rows(HELD, n_anchor_cases, "held_shuffled_anchor", {**ROLE_MAP, **SHUFFLED_HELD_MAP})
    elif arm == "coverage_only":
        # A small ordinary-coverage arm: no role anchors for held predicates, but
        # a minimal number of held composition examples. This is not token-matched
        # to the anchor arms and is reported separately.
        extra = make_composition_eval(HELD, SEEN, 1, "held_sparse_coverage") + make_filler_rows(max(0, held_anchor_n - len(make_composition_eval(HELD, SEEN, 1, "x"))), "coverage_filler")
    else:
        raise ValueError(arm)
    return base + seen_anchor + extra


class Vocab:
    def __init__(self): self.w2i={"<pad>":0,"<unk>":1}
    def fit(self, rows):
        for r in rows:
            for t in toks(r['text']):
                if t not in self.w2i: self.w2i[t]=len(self.w2i)
    def enc(self, text, max_len):
        ids=[self.w2i.get(t,1) for t in toks(text)][:max_len]
        return ids+[0]*(max_len-len(ids))

def encode(rows, vocab, max_len):
    return torch.tensor([vocab.enc(r['text'],max_len) for r in rows], dtype=torch.long), torch.tensor([int(r['label']) for r in rows], dtype=torch.long)

class BiGRU(nn.Module):
    def __init__(self, vs, emb=64, hid=96):
        super().__init__(); self.emb=nn.Embedding(vs,emb,padding_idx=0); self.gru=nn.GRU(emb,hid,batch_first=True,bidirectional=True); self.head=nn.Sequential(nn.LayerNorm(2*hid),nn.Linear(2*hid,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,x):
        _,h=self.gru(self.emb(x)); return self.head(torch.cat([h[-2],h[-1]],dim=-1))

def acc(model,x,y,batch=1024):
    model.eval(); ok=0; n=0
    with torch.no_grad():
        for s in range(0,len(x),batch):
            pred=model(x[s:s+batch]).argmax(1); ok+=int((pred==y[s:s+batch]).sum().item()); n+=len(pred)
    return ok/max(1,n)

def train_eval(arm, seed, args, eval_rows_by_name):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(min(8, os.cpu_count() or 1))
    train_rows=build_arm(arm,args.comp_cases,args.anchor_cases)
    all_eval=[r for rows in eval_rows_by_name.values() for r in rows]
    vocab=Vocab(); vocab.fit(train_rows)  # eval-only held tokens are in-vocab only if exposed/anchored/covered by arm.
    max_len=min(96,max(len(toks(r['text'])) for r in train_rows+all_eval))
    tr_x,tr_y=encode(train_rows,vocab,max_len)
    ev={k:encode(v,vocab,max_len) for k,v in eval_rows_by_name.items()}
    model=BiGRU(len(vocab.w2i)); opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader=DataLoader(TensorDataset(tr_x,tr_y), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best=-1; best_state=None; hist=[]; t0=time.time()
    for ep in range(1,args.epochs+1):
        model.train()
        for xb,yb in loader:
            loss=F.cross_entropy(model(xb),yb); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if ep<=3 or ep%2==0 or ep==args.epochs:
            ta=acc(model,tr_x,tr_y); hb=acc(model,*ev['held_both'])
            rec={"epoch":ep,"train_acc":ta,"held_both":hb,"seconds":time.time()-t0}; hist.append(rec); print(json.dumps({"arm":arm,"seed":seed,**rec}), flush=True)
            if ta>best: best=ta; best_state=copy.deepcopy(model.state_dict())
            if ta>=0.999: break
    if best_state: model.load_state_dict(best_state)
    out={"arm":arm,"seed":seed,"train_rows":len(train_rows),"train_label_frac":float(np.mean([r['label'] for r in train_rows])),"vocab_size":len(vocab.w2i),"max_len":max_len,"final_train_acc":acc(model,tr_x,tr_y),"history":hist,"eval":{k:acc(model,*v) for k,v in ev.items()},"kind_counts":dict(collections.Counter(r['kind'] for r in train_rows))}
    return out


def exposure_ledger(rows: list[dict[str,Any]]) -> dict[str,Any]:
    cnt=collections.Counter()
    for r in rows:
        ts=toks(r['text'])
        for p in HELD+SEEN+DUMMY_PREDS:
            cnt[p]+=ts.count(p)
    return {"held_predicate_token_counts":{p:cnt[p] for p in HELD},"seen_predicate_token_counts":{p:cnt[p] for p in SEEN},"dummy_predicate_token_counts":{p:cnt[p] for p in DUMMY_PREDS},"total_tokens":sum(len(toks(r['text'])) for r in rows),"n_rows":len(rows)}


def audit(rows: list[dict[str,Any]], name: str) -> dict[str,Any]:
    c=collections.Counter(r['label'] for r in rows)
    textlabs=collections.defaultdict(set)
    cells=collections.defaultdict(collections.Counter)
    for r in rows:
        textlabs[r['text']].add(r['label'])
        cells[(r.get('cp'),r.get('hp'),r.get('co'),r.get('ho'))][r['label']]+=1
    return {"name":name,"n":len(rows),"label_counts":dict(c),"true_frac":c[1]/max(1,len(rows)),"duplicate_text_conflicts":sum(len(v)>1 for v in textlabs.values()),"distinct_texts":len(textlabs),"cell_count":len(cells)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--comp-cases',type=int,default=6); ap.add_argument('--anchor-cases',type=int,default=4)
    ap.add_argument('--seeds',type=str,default='254,255,256,257,258')
    ap.add_argument('--arms',type=str,default='noheld_filler,exposure_only,true_anchor,shuffled_anchor,coverage_only')
    args=ap.parse_args(); OUT_DIR.mkdir(parents=True,exist_ok=True)
    evals={
        'seen_heldfamily': make_composition_eval(SEEN,SEEN,2,'eval_seen'),
        'held_hyp_only': make_composition_eval(SEEN,HELD,2,'eval_held_hyp'),
        'held_ctx_only': make_composition_eval(HELD,SEEN,2,'eval_held_ctx'),
        'held_both': make_composition_eval(HELD,HELD,2,'eval_held_both'),
    }
    for k,v in evals.items(): write_jsonl(OUT_DIR/f'{k}.jsonl',v)
    cases=[]; ledgers={}
    for arm in [a for a in args.arms.split(',') if a.strip()]:
        arm_rows=build_arm(arm,args.comp_cases,args.anchor_cases)
        write_jsonl(OUT_DIR/f'train_{arm}.jsonl', arm_rows)
        ledgers[arm]=exposure_ledger(arm_rows)
        for seed in [int(s) for s in args.seeds.split(',') if s.strip()]:
            cases.append(train_eval(arm,seed,args,evals))
    surfaces=list(evals)
    agg={}
    for arm in sorted({c['arm'] for c in cases}):
        sub=[c for c in cases if c['arm']==arm]
        conv=[c for c in sub if c['final_train_acc']>=0.95]
        agg[arm]={"n":len(sub),"n_converged":len(conv),"train_acc_seeds":[c['final_train_acc'] for c in sub],
                  "raw":{s:{"mean":float(np.mean([c['eval'][s] for c in sub])),"seeds":[c['eval'][s] for c in sub]} for s in surfaces},
                  "converged_only":{s:{"mean":float(np.mean([c['eval'][s] for c in conv])) if conv else float('nan'),"seeds":[c['eval'][s] for c in conv]} for s in surfaces}}
    summary={"status":"CAUSAL_ROLE_ANCHOR_CORRECTED","created_utc":now(),"role_map":ROLE_MAP,"seen":SEEN,"held":HELD,"config":vars(args),"ledgers":ledgers,"eval_audits":{k:audit(v,k) for k,v in evals.items()},"train_audits":{arm:audit(build_arm(arm,args.comp_cases,args.anchor_cases),arm) for arm in agg},"aggregates":agg,"cases":cases,"interpretation_boundary":"The composition label requires both context and hypothesis predicate role maps. Held predicates have identical syntax and no world/domain marker. A true-anchor advantage over exposure_only and shuffled_anchor would support a causal role-word anchoring mechanism in this controlled coordinate; absence of such contrast refutes that mechanism here."}
    write_json(OUT_DIR/'causal_role_anchor_summary.json',summary)
    md=['# research corrected causal role-anchor experiment','',f'Seen: {SEEN}; Held: {HELD}. All predicates use identical syntax; no world/domain markers.','', '| arm | runs | converged | seen | held hyp | held ctx | held both |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ['noheld_filler','exposure_only','true_anchor','shuffled_anchor','coverage_only']:
        if arm not in agg: continue
        a=agg[arm]; co=a['converged_only']
        md.append(f"| {arm} | {a['n']} | {a['n_converged']} | {co['seen_heldfamily']['mean']:.3f} | {co['held_hyp_only']['mean']:.3f} | {co['held_ctx_only']['mean']:.3f} | {co['held_both']['mean']:.3f} |")
    md += ['', 'Converged-only means use train accuracy >= 0.95 and training-fit-only selection.', '', f"Summary JSON: `{OUT_DIR/'causal_role_anchor_summary.json'}`"]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/causal_role_anchor_corrected/causal_role_anchor_summary.md')).write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary_json':str(OUT_DIR/'causal_role_anchor_summary.json'),
                      'held_both_converged':{a:agg[a]['converged_only']['held_both']['mean'] for a in agg},
                      'converged_counts':{a:agg[a]['n_converged'] for a in agg}}, indent=2), flush=True)

if __name__=='__main__': main()
