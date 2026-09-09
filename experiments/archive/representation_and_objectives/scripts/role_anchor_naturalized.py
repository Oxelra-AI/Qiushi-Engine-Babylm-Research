#!/usr/bin/env python3
"""research: naturalized, cue-stripped role-anchor transfer test.

research's corrected atomic predicate script removed the known leakage but then did
not fit its own training rows. This script keeps the important repairs while
removing the artificial ENTITY_A/ENTITY_B symmetry that made optimization
uninformative:

- no world/domain/case markers in text;
- all relation predicates use identical surface syntax: "name predicate name";
- names and distractors vary across pair-held cases;
- held predicates never occur in composition training;
- true, exposure-only, shuffled-anchor, no-held, and sparse-coverage arms are
  compared at small CPU scale;
- model selection uses train fit only, and train accuracy is also reported by
  row kind so a non-fitting run is not interpreted as a mechanism result.

The scientific question is narrow: can flat role-word labels for a held predicate
make that predicate composable in context/hypothesis positions better than mere
exposure or wrong role labels, under a sequence learner rather than a hard-coded
parser?
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

STUDY = Path("experiments/archive/representation_and_objectives")
OUT_DIR = STUDY / "data/role_anchor_naturalized"

SEEN = ["dax", "feg", "mip", "lor"]
HELD = ["zup", "niv", "kem", "rox"]
DUMMY = ["tav", "gop", "sul", "bem"]
# +1 means the grammatical subject bears the positive role (winner); -1 means the object does.
ROLE_MAP = {"dax": 1, "feg": 1, "mip": -1, "lor": -1, "zup": 1, "niv": 1, "kem": -1, "rox": -1}
SHUFFLED_HELD_MAP = {p: -ROLE_MAP[p] for p in HELD}
NAMES = [
    "alisa", "bren", "cato", "dora", "elin", "fara", "gilo", "hani",
    "ivan", "jora", "kira", "lena", "milo", "nora", "orin", "pava",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def toks(s: str) -> list[str]:
    return re.findall(r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]", str(s).lower())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_pairs(n_train: int, n_eval: int, seed: int = 25501) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    """Return train/eval (A,B,D1,D2) cases with held pair combinations.

    Individual names recur across both splits so the test is not OOV lexical
    recognition; pair combinations are disjoint.
    """
    rng = random.Random(seed)
    pairs = [(a, b) for i, a in enumerate(NAMES) for j, b in enumerate(NAMES) if i < j]
    rng.shuffle(pairs)
    chosen_train: list[tuple[str, str]] = []
    covered: set[str] = set()
    # First cover every name at least once in training.
    for p in pairs:
        if len(chosen_train) >= n_train:
            break
        if p[0] not in covered or p[1] not in covered:
            chosen_train.append(p); covered.update(p)
    for p in pairs:
        if len(chosen_train) >= n_train:
            break
        if p not in chosen_train:
            chosen_train.append(p)
    used = set(chosen_train)
    chosen_eval = [p for p in pairs if p not in used][:n_eval]
    def with_distractors(pl: list[tuple[str, str]]) -> list[tuple[str, str, str, str]]:
        out=[]
        for a,b in pl:
            rest=[x for x in NAMES if x not in {a,b}]
            d1,d2=rng.sample(rest,2)
            out.append((a,b,d1,d2))
        return out
    return with_distractors(chosen_train), with_distractors(chosen_eval)


def oriented_pair(a: str, b: str, orient: str) -> tuple[str, str]:
    return (a, b) if orient == "AB" else (b, a)


def roleplus_entity(pred: str, subj: str, obj: str, role_map: dict[str, int]) -> str:
    return subj if role_map[pred] == 1 else obj


def sent(pred: str, a: str, b: str, orient: str, d1: str, d2: str) -> str:
    subj, obj = oriented_pair(a, b, orient)
    return f"{subj} {pred} {obj}. {d1} and {d2} watched."


def hyp(pred: str, a: str, b: str, orient: str) -> str:
    subj, obj = oriented_pair(a, b, orient)
    return f"{subj} {pred} {obj}."


def role_hyp(entity: str, role: str) -> str:
    return f"{entity} was the {role}."


def mention_hyp(entity: str) -> str:
    return f"{entity} was mentioned."


def comp_label(cp: str, co: str, hp: str, ho: str, a: str, b: str, role_map: dict[str, int]) -> int:
    cs, co_obj = oriented_pair(a, b, co)
    hs, ho_obj = oriented_pair(a, b, ho)
    return int(roleplus_entity(cp, cs, co_obj, role_map) == roleplus_entity(hp, hs, ho_obj, role_map))


def comp_rows(cases: list[tuple[str, str, str, str]], cps: list[str], hps: list[str], kind: str) -> list[dict[str, Any]]:
    rows=[]
    for ci,(a,b,d1,d2) in enumerate(cases):
        for cp in cps:
            for hp in hps:
                for co in ["AB","BA"]:
                    for ho in ["AB","BA"]:
                        rows.append({
                            "id": f"{kind}_{ci}_{a}_{b}_{cp}_{co}_{hp}_{ho}",
                            "kind": kind, "a": a, "b": b, "d1": d1, "d2": d2,
                            "cp": cp, "hp": hp, "co": co, "ho": ho,
                            "text": sent(cp,a,b,co,d1,d2) + " [SEP] " + hyp(hp,a,b,ho),
                            "label": comp_label(cp,co,hp,ho,a,b,ROLE_MAP),
                        })
    return rows


def anchor_rows(cases: list[tuple[str, str, str, str]], preds: list[str], kind: str, label_map: dict[str, int]) -> list[dict[str, Any]]:
    rows=[]
    for ci,(a,b,d1,d2) in enumerate(cases):
        for p in preds:
            for orient in ["AB","BA"]:
                subj,obj=oriented_pair(a,b,orient)
                rp=roleplus_entity(p,subj,obj,label_map)
                rm=obj if rp == subj else subj
                for ent, role, lab in [(rp,"winner",1),(rm,"winner",0),(rp,"loser",0),(rm,"loser",1)]:
                    rows.append({
                        "id": f"{kind}_{ci}_{a}_{b}_{p}_{orient}_{ent}_{role}",
                        "kind": kind, "a": a, "b": b, "d1": d1, "d2": d2,
                        "pred": p, "orient": orient,
                        "text": sent(p,a,b,orient,d1,d2) + " [SEP] " + role_hyp(ent,role),
                        "label": lab,
                    })
    return rows


def exposure_rows(cases: list[tuple[str, str, str, str]], preds: list[str], kind: str) -> list[dict[str, Any]]:
    rows=[]
    for ci,(a,b,d1,d2) in enumerate(cases):
        # negatives are names not in the context sentence.
        negs=[x for x in NAMES if x not in {a,b,d1,d2}]
        n1,n2=negs[(2*ci)%len(negs)], negs[(2*ci+1)%len(negs)]
        for p in preds:
            for orient in ["AB","BA"]:
                for ent, lab in [(a,1),(b,1),(n1,0),(n2,0)]:
                    rows.append({
                        "id": f"{kind}_{ci}_{a}_{b}_{p}_{orient}_{ent}",
                        "kind": kind, "a": a, "b": b, "d1": d1, "d2": d2,
                        "pred": p, "orient": orient,
                        "text": sent(p,a,b,orient,d1,d2) + " [SEP] " + mention_hyp(ent),
                        "label": lab,
                    })
    return rows


def filler_rows(cases: list[tuple[str, str, str, str]], n_rows: int, kind: str) -> list[dict[str, Any]]:
    rows=[]
    i=0
    while len(rows) < n_rows:
        a,b,d1,d2 = cases[i % len(cases)]
        p = DUMMY[i % len(DUMMY)]
        orient = "AB" if (i//len(DUMMY)) % 2 == 0 else "BA"
        negs=[x for x in NAMES if x not in {a,b,d1,d2}]
        ent,lab = [(a,1),(b,1),(negs[0],0),(negs[1],0)][i % 4]
        rows.append({"id": f"{kind}_{i}", "kind": kind, "pred": p, "orient": orient,
                     "text": sent(p,a,b,orient,d1,d2) + " [SEP] " + mention_hyp(ent), "label": lab})
        i += 1
    return rows


def build_arm(arm: str, train_cases: list[tuple[str,str,str,str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    comp = comp_rows(train_cases[:args.comp_cases], SEEN, SEEN, "seen_composition")
    seen_anchor = anchor_rows(train_cases[:args.anchor_cases], SEEN, "seen_anchor", ROLE_MAP)
    held_anchor_n = len(anchor_rows(train_cases[:args.anchor_cases], HELD, "tmp", ROLE_MAP))
    if arm == "noheld_filler":
        extra = filler_rows(train_cases[:args.anchor_cases], held_anchor_n, "matched_filler")
    elif arm == "exposure_only":
        extra = exposure_rows(train_cases[:args.anchor_cases], HELD, "held_exposure_only")
    elif arm == "true_anchor":
        extra = anchor_rows(train_cases[:args.anchor_cases], HELD, "held_true_anchor", ROLE_MAP)
    elif arm == "shuffled_anchor":
        extra = anchor_rows(train_cases[:args.anchor_cases], HELD, "held_shuffled_anchor", {**ROLE_MAP, **SHUFFLED_HELD_MAP})
    elif arm == "coverage_only":
        # Same order of rows as other arms, but spend the budget on a small number
        # of held-predicate composition examples plus filler. This is an ordinary
        # coverage comparison, not token-matched to role anchors.
        cov = comp_rows(train_cases[:max(1,args.anchor_cases//6)], HELD, SEEN, "held_sparse_coverage")
        extra = cov + filler_rows(train_cases[:args.anchor_cases], max(0, held_anchor_n - len(cov)), "coverage_filler")
    else:
        raise ValueError(arm)
    return comp + seen_anchor + extra


class Vocab:
    def __init__(self):
        self.w2i={"<pad>":0,"<unk>":1}
    def fit(self, rows: list[dict[str, Any]]):
        for r in rows:
            for t in toks(r["text"]):
                if t not in self.w2i: self.w2i[t]=len(self.w2i)
    def enc(self, text: str, max_len: int) -> list[int]:
        ids=[self.w2i.get(t,1) for t in toks(text)][:max_len]
        return ids + [0]*(max_len-len(ids))


def encode(rows: list[dict[str, Any]], vocab: Vocab, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([vocab.enc(r["text"], max_len) for r in rows], dtype=torch.long), torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)


class AttnBiGRU(nn.Module):
    def __init__(self, vs: int, emb: int = 72, hid: int = 128, dropout: float = 0.02):
        super().__init__()
        self.emb=nn.Embedding(vs, emb, padding_idx=0)
        self.gru=nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.query=nn.Parameter(torch.randn(2*hid) * 0.02)
        self.head=nn.Sequential(nn.LayerNorm(2*hid), nn.Dropout(dropout), nn.Linear(2*hid,96), nn.ReLU(), nn.Linear(96,2))
    def forward(self, x):
        mask=x.ne(0)
        h,_=self.gru(self.emb(x))
        score=torch.matmul(h,self.query).masked_fill(~mask, -1e4)
        w=torch.softmax(score, dim=1).unsqueeze(-1)
        pooled=(h*w).sum(1)
        return self.head(pooled)


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 1024) -> float:
    model.eval(); ok=0; n=0
    with torch.no_grad():
        for s in range(0,len(x),batch):
            pred=model(x[s:s+batch]).argmax(1)
            ok += int((pred == y[s:s+batch]).sum().item()); n += len(pred)
    return ok/max(1,n)


def kind_accuracy(model: nn.Module, rows: list[dict[str,Any]], x: torch.Tensor, y: torch.Tensor) -> dict[str,float]:
    out={}
    by=collections.defaultdict(list)
    for i,r in enumerate(rows): by[r["kind"]].append(i)
    for k,idxs in by.items():
        ids=torch.tensor(idxs,dtype=torch.long)
        out[k]=accuracy(model, x[ids], y[ids])
    return out


def train_one(arm: str, seed: int, train_rows: list[dict[str,Any]], evals: dict[str,list[dict[str,Any]]], args: argparse.Namespace) -> dict[str,Any]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(min(8, os.cpu_count() or 1))
    vocab=Vocab(); vocab.fit(train_rows)
    all_eval=[r for rows in evals.values() for r in rows]
    max_len=min(80, max(len(toks(r["text"])) for r in train_rows + all_eval))
    tr_x,tr_y=encode(train_rows,vocab,max_len)
    enc={k: encode(v,vocab,max_len) for k,v in evals.items()}
    model=AttnBiGRU(len(vocab.w2i))
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader=DataLoader(TensorDataset(tr_x,tr_y), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best=-1.0; best_state=None; hist=[]; t0=time.time()
    for ep in range(1,args.epochs+1):
        model.train()
        for xb,yb in loader:
            loss=F.cross_entropy(model(xb),yb)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if ep <= 3 or ep % args.eval_every == 0 or ep == args.epochs:
            ta=accuracy(model,tr_x,tr_y); hb=accuracy(model,*enc["held_both"])
            rec={"epoch":ep,"train_acc":ta,"held_both":hb,"seconds":time.time()-t0}
            hist.append(rec); print(json.dumps({"arm":arm,"seed":seed,**rec}), flush=True)
            if ta > best:
                best=ta; best_state=copy.deepcopy(model.state_dict())
            if ta >= args.early_stop:
                break
    if best_state: model.load_state_dict(best_state)
    out={
        "arm": arm, "seed": seed, "train_rows": len(train_rows), "vocab_size": len(vocab.w2i), "max_len": max_len,
        "final_train_acc": accuracy(model,tr_x,tr_y), "train_kind_acc": kind_accuracy(model, train_rows, tr_x, tr_y),
        "history": hist, "eval": {k: accuracy(model,*v) for k,v in enc.items()},
    }
    return out


def audit(rows: list[dict[str,Any]], name: str) -> dict[str,Any]:
    labels=collections.Counter(int(r["label"]) for r in rows)
    texts=collections.defaultdict(set); cells=collections.defaultdict(collections.Counter)
    for r in rows:
        texts[r["text"]].add(int(r["label"]))
        cells[(r.get("cp"),r.get("hp"),r.get("co"),r.get("ho"),r.get("pred"),r.get("orient"))][int(r["label"])] += 1
    return {"name":name,"n":len(rows),"label_counts":dict(labels),"true_frac":labels.get(1,0)/max(1,len(rows)),
            "duplicate_text_conflicts":sum(1 for v in texts.values() if len(v)>1),"distinct_texts":len(texts),"cell_count":len(cells)}


def ledger(rows: list[dict[str,Any]]) -> dict[str,Any]:
    cnt=collections.Counter()
    for r in rows:
        ts=toks(r["text"])
        for p in SEEN+HELD+DUMMY: cnt[p]+=ts.count(p)
    return {"seen_pred_counts":{p:cnt[p] for p in SEEN},"held_pred_counts":{p:cnt[p] for p in HELD},"dummy_pred_counts":{p:cnt[p] for p in DUMMY},"tokens":sum(len(toks(r["text"])) for r in rows),"rows":len(rows)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--comp-cases", type=int, default=48)
    ap.add_argument("--anchor-cases", type=int, default=24)
    ap.add_argument("--eval-cases", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--early-stop", type=float, default=0.997)
    ap.add_argument("--seeds", type=str, default="255,256,257")
    ap.add_argument("--arms", type=str, default="noheld_filler,exposure_only,true_anchor,shuffled_anchor,coverage_only")
    args=ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_cases, eval_cases = make_pairs(max(args.comp_cases,args.anchor_cases), args.eval_cases)
    evals={
        "seen_comp": comp_rows(eval_cases, SEEN, SEEN, "eval_seen_comp"),
        "held_hyp_only": comp_rows(eval_cases, SEEN, HELD, "eval_held_hyp_only"),
        "held_ctx_only": comp_rows(eval_cases, HELD, SEEN, "eval_held_ctx_only"),
        "held_both": comp_rows(eval_cases, HELD, HELD, "eval_held_both"),
    }
    write_jsonl(OUT_DIR/"train_cases.jsonl", [{"i":i,"a":a,"b":b,"d1":d1,"d2":d2} for i,(a,b,d1,d2) in enumerate(train_cases)])
    write_jsonl(OUT_DIR/"eval_cases.jsonl", [{"i":i,"a":a,"b":b,"d1":d1,"d2":d2} for i,(a,b,d1,d2) in enumerate(eval_cases)])
    for k,v in evals.items(): write_jsonl(OUT_DIR/f"eval_{k}.jsonl", v)
    cases=[]; train_audits={}; ledgers={}
    arms=[a for a in args.arms.split(',') if a.strip()]
    seeds=[int(s) for s in args.seeds.split(',') if s.strip()]
    for arm in arms:
        rows=build_arm(arm,train_cases,args)
        write_jsonl(OUT_DIR/f"train_{arm}.jsonl", rows)
        train_audits[arm]=audit(rows,arm); ledgers[arm]=ledger(rows)
        for seed in seeds:
            cases.append(train_one(arm, seed, rows, evals, args))
    surfaces=list(evals)
    agg={}
    for arm in arms:
        sub=[c for c in cases if c["arm"] == arm]
        conv=[c for c in sub if c["final_train_acc"] >= 0.95]
        agg[arm]={
            "n": len(sub), "n_converged": len(conv),
            "train_acc": {"mean": float(np.mean([c["final_train_acc"] for c in sub])), "seeds": [c["final_train_acc"] for c in sub]},
            "train_kind_acc_mean": {k: float(np.mean([c["train_kind_acc"].get(k, float('nan')) for c in sub])) for k in sorted({kk for c in sub for kk in c["train_kind_acc"]})},
            "raw": {s: {"mean": float(np.mean([c["eval"][s] for c in sub])), "seeds": [c["eval"][s] for c in sub]} for s in surfaces},
            "converged_only": {s: {"mean": float(np.mean([c["eval"][s] for c in conv])) if conv else float('nan'), "seeds": [c["eval"][s] for c in conv]} for s in surfaces},
        }
    summary={
        "status":"ROLE_ANCHOR_NATURALIZED",
        "created_utc":now(),"config":vars(args),"seen":SEEN,"held":HELD,"role_map":ROLE_MAP,
        "train_pair_design":{"train_case_count":len(train_cases),"eval_case_count":len(eval_cases),"individual_names_reused":True,"pair_combinations_disjoint":True,"no_world_or_domain_markers":True,"identical_predicate_syntax":True},
        "eval_audits":{k:audit(v,k) for k,v in evals.items()},"train_audits":train_audits,"ledgers":ledgers,
        "aggregates":agg,"cases":cases,
        "interpretation":"A true_anchor advantage over exposure_only and shuffled_anchor on held_hyp_only and held_ctx_only, conditional on train fit, means flat role-word support can make held predicate tokens composable for this sequence learner. held_both alone is not sufficient because inverting all held predicate polarities preserves held-held equality.",
    }
    write_json(OUT_DIR/"role_anchor_naturalized_summary.json", summary)
    md=["# research naturalized role-anchor transfer test","",f"Seen predicates: {SEEN}",f"Held predicates: {HELD}","No world/domain/case markers; identical relation syntax; pair combinations held out but names reused.","", "| arm | runs | fit runs | train | seen comp | held hyp | held ctx | held both |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        a=agg[arm]; co=a["converged_only"]
        md.append(f"| {arm} | {a['n']} | {a['n_converged']} | {a['train_acc']['mean']:.3f} | {co['seen_comp']['mean']:.3f} | {co['held_hyp_only']['mean']:.3f} | {co['held_ctx_only']['mean']:.3f} | {co['held_both']['mean']:.3f} |")
    md += ["", "Numbers in the four right columns are train-fit-only selected and fit-run means. Raw seed values and train-kind accuracies are in the JSON.", "", f"Summary JSON: `{OUT_DIR/'role_anchor_naturalized_summary.json'}`"]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/role_anchor_naturalized/role_anchor_naturalized_summary.md')).write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps({"status":summary["status"],"summary_json":str(OUT_DIR/"role_anchor_naturalized_summary.json"),"fit_counts":{a:agg[a]["n_converged"] for a in arms},"held_mixed_fit_means":{a:{"held_hyp":agg[a]["converged_only"]["held_hyp_only"]["mean"],"held_ctx":agg[a]["converged_only"]["held_ctx_only"]["mean"]} for a in arms}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
