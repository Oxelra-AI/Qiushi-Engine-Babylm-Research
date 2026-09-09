#!/usr/bin/env python3
"""research: factorized role-anchor learner.

The raw sequence tests did not fit the controlled rows, so they cannot decide the
scientific mechanism. This script tests a more precise architecture/training
principle: represent each binary relation predicate by a learned role coordinate
(subject bears positive role vs object bears positive role), and compose two
relations by equality of their positive-role entity. The question is whether flat
role anchors for held predicates supply the missing coordinates more efficiently
than ordinary composition rows.

This is intentionally not a large LM experiment. It is a compact mechanistic
algorithm test with transparent equations and exposure-matched control arms.
"""
from __future__ import annotations

import argparse, json, math, random, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

STUDY = Path("experiments/archive/representation_and_objectives")
OUT_DIR = STUDY / "data/factorized_role_anchor"

SEEN = ["dax", "feg", "mip", "lor"]
HELD = ["zup", "niv", "kem", "rox"]
ALL = SEEN + HELD
ROLE_MAP = {"dax": 1, "feg": 1, "mip": -1, "lor": -1, "zup": 1, "niv": 1, "kem": -1, "rox": -1}
SHUFFLED_HELD = {p: -ROLE_MAP[p] for p in HELD}
P2I = {p:i for i,p in enumerate(ALL)}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def orient_factor(o: str) -> int:
    return 1 if o == "AB" else -1


def comp_label(cp: str, co: str, hp: str, ho: str, role_map: dict[str,int]) -> int:
    return int(ROLE_MAP[cp] * orient_factor(co) == ROLE_MAP[hp] * orient_factor(ho))


def make_comp(cps: list[str], hps: list[str], cases: int, kind: str) -> list[dict[str,Any]]:
    rows=[]
    for case in range(cases):
        for cp in cps:
            for hp in hps:
                for co in ["AB","BA"]:
                    for ho in ["AB","BA"]:
                        rows.append({"kind":kind,"case":case,"type":"comp","cp":cp,"hp":hp,"co":co,"ho":ho,
                                     "label":comp_label(cp,co,hp,ho,ROLE_MAP)})
    return rows


def make_anchor(preds: list[str], cases: int, kind: str, label_map: dict[str,int]) -> list[dict[str,Any]]:
    rows=[]
    for case in range(cases):
        for p in preds:
            for o in ["AB","BA"]:
                # query entity + role. entity is A/B. label_map supplies the arm's supervised role meaning.
                pos = label_map[p] * orient_factor(o)  # +1 means A is positive-role entity, -1 means B.
                for ent in ["A","B"]:
                    for role in ["winner","loser"]:
                        ent_sign = 1 if ent == "A" else -1
                        y = int((ent_sign == pos) == (role == "winner"))
                        rows.append({"kind":kind,"case":case,"type":"anchor","p":p,"o":o,"ent":ent,"role":role,"label":y})
    return rows


def make_dummy(rows: int, kind: str) -> list[dict[str,Any]]:
    # Non-role rows in matched exposure arms carry no gradient in this factorized role model.
    return [{"kind":kind,"type":"dummy","label":i%2} for i in range(rows)]


def build_arm(arm: str, comp_cases: int, anchor_cases: int, sparse_cases: int) -> list[dict[str,Any]]:
    seen_comp = make_comp(SEEN, SEEN, comp_cases, "seen_composition")
    seen_anchor = make_anchor(SEEN, anchor_cases, "seen_anchor", ROLE_MAP)
    held_anchor_n = len(make_anchor(HELD, anchor_cases, "tmp", ROLE_MAP))
    if arm == "noheld_filler":
        extra = make_dummy(held_anchor_n, "matched_filler")
    elif arm == "exposure_only":
        extra = make_dummy(held_anchor_n, "held_exposure_only")
    elif arm == "true_anchor":
        extra = make_anchor(HELD, anchor_cases, "held_true_anchor", ROLE_MAP)
    elif arm == "shuffled_anchor":
        extra = make_anchor(HELD, anchor_cases, "held_shuffled_anchor", {**ROLE_MAP, **SHUFFLED_HELD})
    elif arm == "sparse_ctx_coverage":
        # Coverage spends comparable rows on held predicates in context position with seen hypotheses.
        cov = make_comp(HELD, SEEN, sparse_cases, "held_sparse_ctx_composition")
        extra = cov + make_dummy(max(0, held_anchor_n - len(cov)), "coverage_filler")
    elif arm == "sparse_hyp_coverage":
        cov = make_comp(SEEN, HELD, sparse_cases, "held_sparse_hyp_composition")
        extra = cov + make_dummy(max(0, held_anchor_n - len(cov)), "coverage_filler")
    else:
        raise ValueError(arm)
    return seen_comp + seen_anchor + extra


class RoleFactorModel(nn.Module):
    def __init__(self, init_scale: float = 0.01):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(len(ALL)) * init_scale)
        # Dummy branch can absorb non-role exposure rows without changing role coordinates.
        self.dummy_logit = nn.Parameter(torch.tensor(0.0))

    def p_A_positive(self, pred: str, orient: str) -> torch.Tensor:
        return torch.sigmoid(self.theta[P2I[pred]] * orient_factor(orient))

    def row_prob(self, row: dict[str,Any]) -> torch.Tensor:
        if row["type"] == "comp":
            pc = self.p_A_positive(row["cp"], row["co"])
            ph = self.p_A_positive(row["hp"], row["ho"])
            return pc*ph + (1-pc)*(1-ph)
        if row["type"] == "anchor":
            pA = self.p_A_positive(row["p"], row["o"])
            if row["ent"] == "A":
                pwinner = pA
            else:
                pwinner = 1-pA
            return pwinner if row["role"] == "winner" else 1-pwinner
        if row["type"] == "dummy":
            return torch.sigmoid(self.dummy_logit)
        raise ValueError(row["type"])

    def loss(self, rows: list[dict[str,Any]]) -> torch.Tensor:
        probs=[]; ys=[]
        for r in rows:
            probs.append(self.row_prob(r))
            ys.append(float(r["label"]))
        p=torch.stack(probs).clamp(1e-5, 1-1e-5)
        y=torch.tensor(ys, dtype=torch.float32, device=p.device)
        return F.binary_cross_entropy(p, y)

    def predict(self, rows: list[dict[str,Any]]) -> np.ndarray:
        with torch.no_grad():
            ps=torch.stack([self.row_prob(r) for r in rows]).cpu().numpy()
        return (ps >= 0.5).astype(int)


def accuracy(model: RoleFactorModel, rows: list[dict[str,Any]]) -> float:
    pred = model.predict(rows)
    y = np.array([int(r["label"]) for r in rows])
    return float((pred == y).mean())


def train(rows: list[dict[str,Any]], seed: int, epochs: int, lr: float) -> tuple[RoleFactorModel, list[dict[str,Any]]]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model=RoleFactorModel()
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    hist=[]
    for ep in range(1, epochs+1):
        loss=model.loss(rows)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ep in {1,2,3,5,10,20,50,100,200,epochs}:
            hist.append({"epoch":ep,"loss":float(loss.detach().cpu()),"train_acc":accuracy(model, rows)})
        if accuracy(model, rows) >= 0.999 and ep >= 20:
            break
    return model,hist


def by_kind_acc(model: RoleFactorModel, rows: list[dict[str,Any]]) -> dict[str,float]:
    out={}
    kinds=sorted(set(r["kind"] for r in rows))
    for k in kinds:
        sub=[r for r in rows if r["kind"] == k]
        out[k]=accuracy(model, sub) if sub else float('nan')
    return out


def param_summary(model: RoleFactorModel) -> dict[str,Any]:
    th=model.theta.detach().cpu().numpy()
    return {p:{"theta":float(th[P2I[p]]),"learned_sign":int(1 if th[P2I[p]]>=0 else -1),"true_sign":ROLE_MAP[p]} for p in ALL}


def audit_rows(rows: list[dict[str,Any]], name: str) -> dict[str,Any]:
    counts={0:0,1:0}; kind={}
    for r in rows:
        counts[int(r["label"])] += 1
        k=r["kind"]; kind.setdefault(k,{0:0,1:0}); kind[k][int(r["label"])] += 1
    return {"name":name,"n":len(rows),"label_counts":counts,"true_frac":counts[1]/max(1,len(rows)),"kind_label_counts":kind}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--comp-cases", type=int, default=4)
    ap.add_argument("--anchor-cases", type=int, default=2)
    ap.add_argument("--sparse-cases", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--seeds", type=str, default="11,12,13,14,15")
    ap.add_argument("--arms", type=str, default="noheld_filler,exposure_only,true_anchor,shuffled_anchor,sparse_ctx_coverage,sparse_hyp_coverage")
    args=ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    evals={
        "seen_comp": make_comp(SEEN, SEEN, 1, "eval_seen_comp"),
        "held_hyp_only": make_comp(SEEN, HELD, 1, "eval_held_hyp_only"),
        "held_ctx_only": make_comp(HELD, SEEN, 1, "eval_held_ctx_only"),
        "held_both": make_comp(HELD, HELD, 1, "eval_held_both"),
    }
    cases=[]
    arms=[a for a in args.arms.split(',') if a.strip()]
    seeds=[int(s) for s in args.seeds.split(',') if s.strip()]
    train_audits={}
    for arm in arms:
        rows=build_arm(arm,args.comp_cases,args.anchor_cases,args.sparse_cases)
        train_audits[arm]=audit_rows(rows,arm)
        for seed in seeds:
            model,hist=train(rows,seed,args.epochs,args.lr)
            cases.append({
                "arm":arm,"seed":seed,"train_acc":accuracy(model,rows),"train_kind_acc":by_kind_acc(model,rows),
                "eval":{k:accuracy(model,v) for k,v in evals.items()},"params":param_summary(model),"history":hist,
            })
    agg={}
    for arm in arms:
        sub=[c for c in cases if c["arm"] == arm]
        fit=[c for c in sub if c["train_acc"] >= 0.99]
        agg[arm]={
            "n":len(sub),"n_fit":len(fit),"train_acc":{"mean":float(np.mean([c["train_acc"] for c in sub])),"seeds":[c["train_acc"] for c in sub]},
            "eval_fit":{k:{"mean":float(np.mean([c["eval"][k] for c in fit])) if fit else float('nan'),"seeds":[c["eval"][k] for c in fit]} for k in evals},
            "eval_raw":{k:{"mean":float(np.mean([c["eval"][k] for c in sub])),"seeds":[c["eval"][k] for c in sub]} for k in evals},
            "held_sign_accuracy_fit": float(np.mean([np.mean([1.0 if c["params"][p]["learned_sign"] == ROLE_MAP[p] else 0.0 for p in HELD]) for c in fit])) if fit else float('nan'),
        }
    summary={
        "status":"FACTORIZED_ROLE_ANCHOR","created_utc":now(),"config":vars(args),"seen":SEEN,"held":HELD,"role_map":ROLE_MAP,
        "equations":{"p_A_positive":"sigmoid(theta_pred * orient_factor)","composition_prob":"p_ctx*p_hyp + (1-p_ctx)*(1-p_hyp)","anchor_prob":"probability queried entity has requested role under predicate coordinate"},
        "train_audits":train_audits,"eval_audits":{k:audit_rows(v,k) for k,v in evals.items()},"aggregates":agg,"cases":cases,
        "interpretation":"If true_anchor fits and reaches held_hyp_only/held_ctx_only while exposure/noheld remain at chance and shuffled_anchor inverts mixed held-seen surfaces, flat role anchors are sufficient coordinates for compositional reuse in this factorized learner. Sparse coverage shows how ordinary composition rows compare when spent directly on held predicates.",
    }
    write_json(OUT_DIR/"factorized_role_anchor_summary.json", summary)
    md=["# research factorized role-anchor learner","", "Role-coordinate model: each predicate has one learned signed coordinate; composition predicts whether the positive-role entity matches across context and hypothesis.", "", "| arm | runs | fit | held sign fit | seen | held hyp | held ctx | held both |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        a=agg[arm]; ef=a["eval_fit"]
        md.append(f"| {arm} | {a['n']} | {a['n_fit']} | {a['held_sign_accuracy_fit']:.3f} | {ef['seen_comp']['mean']:.3f} | {ef['held_hyp_only']['mean']:.3f} | {ef['held_ctx_only']['mean']:.3f} | {ef['held_both']['mean']:.3f} |")
    md += ["", "The mixed held-seen surfaces are the important surfaces; held-both can remain high under a global inversion of all held predicates.", "", f"Summary JSON: `{OUT_DIR/'factorized_role_anchor_summary.json'}`"]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/factorized_role_anchor/factorized_role_anchor_summary.md')).write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps({"status":summary["status"],"summary_json":str(OUT_DIR/"factorized_role_anchor_summary.json"),"fit_counts":{a:agg[a]["n_fit"] for a in arms},"held_mixed_fit_means":{a:{"held_hyp":agg[a]["eval_fit"]["held_hyp_only"]["mean"],"held_ctx":agg[a]["eval_fit"]["held_ctx_only"]["mean"],"held_sign":agg[a]["held_sign_accuracy_fit"]} for a in arms}}, indent=2), flush=True)

if __name__ == "__main__":
    main()
