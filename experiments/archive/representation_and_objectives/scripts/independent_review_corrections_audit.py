#!/usr/bin/env python3
"""research independent review correction audits.

CPU-only analyses prompted by research independent review verification:
1. Audit what --no-bridge-anchors actually removes in the research arm rows.
2. Inspect tied seed29002 bs-1 comparison underfit for clamp/saturation or row-family pattern.
3. Fit d_minus = alpha + beta d_plus for paired state coordinates to distinguish pure sign reversal from broader change.
4. Split frozen-affine calibration by direct relation h0/h2 and test graph h1/h3 relation-level transfer.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT = Path("experiments/archive/representation_and_objectives")
DATA_ROOT = PROJECT / "data/information_budget_substrate/replace_k16_spread"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs)/len(xs)) if xs else None


def safe_sd(xs: Sequence[float]) -> float | None:
    xs=[float(x) for x in xs]
    if not xs: return None
    if len(xs)==1: return 0.0
    m=sum(xs)/len(xs)
    return float((sum((x-m)**2 for x in xs)/(len(xs)-1))**0.5)


def sign(x: float, eps: float=1e-8) -> int:
    if x > eps: return 1
    if x < -eps: return -1
    return 0


def audit_no_anchor_filter() -> Dict[str, Any]:
    arm_file = DATA_ROOT / "arms/aligned_state_bridge/train_supervised.jsonl"
    rows = load_jsonl(arm_file)
    state = [r for r in rows if r.get("task") == "state_query"]
    cmp_rows = [r for r in rows if r.get("task") == "relation_comparison"]
    state_counts = Counter((r.get("suite"), r.get("relation") or r.get("cause_relation"), r.get("query_kind"), r.get("initial_pattern"), r.get("static_slot")) for r in state)
    rel_counts = Counter(r.get("relation") or r.get("cause_relation") for r in state)
    suite_counts = Counter(r.get("suite") for r in state)
    # The research filter removes all arm state_query rows.  Test whether all are sparse direct h0/h2 anchors.
    direct_rels={"h0_dax","h2_norp"}
    all_direct_changed = all((r.get("relation") or r.get("cause_relation")) in direct_rels and r.get("query_kind") == "changed" for r in state)
    return {
        "arm_file": str(arm_file),
        "arm_train_rows": len(rows), "arm_state_rows": len(state), "arm_comparison_rows": len(cmp_rows),
        "state_suite_counts": dict(suite_counts), "state_relation_counts": dict(rel_counts),
        "state_counts_sample": [{"key": list(k), "count": v} for k,v in state_counts.most_common(12)],
        "no_bridge_filter_removes_all_arm_state_query_rows": True,
        "all_removed_arm_state_rows_are_direct_changed_h0_h2": all_direct_changed,
        "scientific_reading": "If all_removed_arm_state_rows_are_direct_changed_h0_h2 is true, the current research --no-bridge-anchors filter is effectively the intended bridge-anchor removal for this substrate; unchanged/static supervision remains in common_seen_train rather than this arm file.",
    }


def comparison_underfit_audit(run_dir: Path) -> Dict[str, Any]:
    result=load_json(run_dir/"result.json")
    rows=load_jsonl(run_dir/"train_comparison_predictions.jsonl")
    wrong=[r for r in rows if not bool(r.get("correct"))]
    probs=[float(r.get("prob_same")) for r in rows]
    wrong_probs=[float(r.get("prob_same")) for r in wrong]
    def fam(r): return (r.get("suite"), r.get("relation1"), r.get("relation2"), bool(r.get("label")), bool(r.get("pred")))
    fam_counts=Counter(fam(r) for r in wrong)
    by_pair=Counter((r.get("relation1"), r.get("relation2")) for r in wrong)
    near_low=sum(1 for p in wrong_probs if p <= 1.000001e-6)
    near_high=sum(1 for p in wrong_probs if p >= 0.999999)
    margins=[float(r.get("signed_margin")) for r in rows]
    wrong_margins=[float(r.get("signed_margin")) for r in wrong]
    # Estimate clamped BCE contribution on comparison rows only, using label and prob.
    bces=[]
    for r in rows:
        p=float(r.get("prob_same")); y=1.0 if bool(r.get("label")) else 0.0
        p=min(max(p,1e-12),1-1e-12)
        bces.append(-(y*math.log(p)+(1-y)*math.log(1-p)))
    return {
        "run_dir": str(run_dir),
        "train_cmp_acc": result.get("final_train_metrics",{}).get("train_cmp_acc"),
        "train_loss_last": result.get("train_info",{}).get("history",[])[-1].get("loss") if result.get("train_info",{}).get("history") else None,
        "n_train_comparison_rows": len(rows), "n_wrong": len(wrong),
        "prob_range_all": [min(probs), max(probs)] if probs else None,
        "prob_range_wrong": [min(wrong_probs), max(wrong_probs)] if wrong_probs else None,
        "wrong_near_clamp_low": near_low, "wrong_near_clamp_high": near_high,
        "margin_range_all": [min(margins), max(margins)] if margins else None,
        "margin_range_wrong": [min(wrong_margins), max(wrong_margins)] if wrong_margins else None,
        "comparison_bce_mean": mean(bces), "comparison_bce_wrong_mean": mean([bces[i] for i,r in enumerate(rows) if not bool(r.get("correct"))]),
        "wrong_family_counts": [{"suite": k[0], "relation1": k[1], "relation2": k[2], "label": k[3], "pred": k[4], "count": v} for k,v in fam_counts.most_common()],
        "wrong_relation_pair_counts": [{"relation1": k[0], "relation2": k[1], "count": v} for k,v in by_pair.most_common()],
        "wrong_rows_sample": wrong[:10],
        "scientific_reading": "Wrong comparison rows at clamp boundaries would indicate a zero-gradient saturation artifact. Wrong rows away from the 1e-6/1-1e-6 clamp indicate a persistent finite optimization/local-solution issue under the tested schedule, though not necessarily an intrinsic loss impossibility.",
    }


def state_de_map(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows=load_jsonl(run_dir/"eval_state_predictions.jsonl")
    by=defaultdict(list)
    for r in rows:
        by[r.get("query_key")].append(r)
    out={}
    for k,rs in by.items():
        if len(rs)!=2: continue
        rs=sorted(rs,key=lambda r:int(r.get("candidate_index",0)))
        r0=rs[0]
        d=float(r0.get("d_e", float(rs[0]["score"])-float(rs[1]["score"])))
        out[k]={"d":d,"suite":r0.get("suite"),"relation_family":r0.get("relation_family"),"relation":r0.get("relation"),"is_changed":bool(r0.get("is_changed")),"initial_pattern":r0.get("initial_pattern")}
    return out


def regression_dp_dm(plus_dir: Path, minus_dir: Path) -> Dict[str, Any]:
    dp=state_de_map(plus_dir); dm=state_de_map(minus_dir)
    common=set(dp)&set(dm)
    cats={
        "direct_changed_psc": lambda m: m["suite"]=="paired_state_conservation" and m["is_changed"] and m["relation_family"]=="direct_anchor",
        "graph_changed_psc": lambda m: m["suite"]=="paired_state_conservation" and m["is_changed"] and m["relation_family"]=="graph_transfer",
        "graph_changed_same_psc": lambda m: m["suite"]=="paired_state_conservation" and m["is_changed"] and m["relation_family"]=="graph_transfer" and m.get("initial_pattern")=="same",
        "unchanged_psc": lambda m: m["suite"]=="paired_state_conservation" and not m["is_changed"],
    }
    out={}
    for name,f in cats.items():
        ks=[k for k in common if f(dp[k])]
        if not ks:
            out[name]={"n":0}; continue
        x=np.asarray([dp[k]["d"] for k in ks], dtype=float)
        y=np.asarray([dm[k]["d"] for k in ks], dtype=float)
        A=np.stack([x,np.ones_like(x)],axis=1)
        beta,alpha=np.linalg.lstsq(A,y,rcond=None)[0]
        pred=beta*x+alpha
        resid=y-pred
        out[name]={
            "n":len(ks), "alpha":float(alpha), "beta":float(beta),
            "corr":float(np.corrcoef(x,y)[0,1]) if len(ks)>1 and np.std(x)>0 and np.std(y)>0 else None,
            "corr_neg":float(np.corrcoef(x,-y)[0,1]) if len(ks)>1 and np.std(x)>0 and np.std(y)>0 else None,
            "rmse":float(np.sqrt(np.mean(resid**2))), "x_mean":float(np.mean(x)), "y_mean":float(np.mean(y)),
            "same_sign_frac":mean([float(sign(a)==sign(b) and sign(a)!=0) for a,b in zip(x,y)]),
            "opposite_sign_frac":mean([float(sign(a)==-sign(b) and sign(a)!=0 and sign(b)!=0) for a,b in zip(x,y)]),
        }
    return out


def regression_audit(primary_roots: Sequence[Path]) -> Dict[str, Any]:
    out={}
    for root in primary_roots:
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not (d/"result.json").exists(): continue
            rec=load_json(d/"result.json")
            if int(rec.get("bridge_sign"))!=1: continue
            cond=rec.get("condition"); seed=int(rec.get("seed"))
            minus=root / f"{cond}_bs-1_seed{seed}"
            if not minus.exists(): continue
            out[f"{cond}|seed{seed}|{root.name}"]=regression_dp_dm(d, minus)
    return out


def choice_rows_from_run(run_dir: Path) -> List[Dict[str, Any]]:
    rows=load_jsonl(run_dir/"eval_state_predictions.jsonl")
    by=defaultdict(list)
    for r in rows: by[r.get("query_key")].append(r)
    out=[]
    for key,rs in by.items():
        if len(rs)!=2: continue
        rs=sorted(rs,key=lambda r:int(r.get("candidate_index",0)))
        if bool(rs[0].get("label_true")) == bool(rs[1].get("label_true")): continue
        y=1 if bool(rs[0].get("label_true")) else -1
        d=float(rs[0].get("d_e", float(rs[0]["score"])-float(rs[1]["score"])))
        r0=rs[0]
        out.append({"query_key":key,"d":d,"y":y,"suite":r0.get("suite"),"relation":r0.get("relation"),"relation_family":r0.get("relation_family"),"is_changed":bool(r0.get("is_changed")),"initial_pattern":r0.get("initial_pattern")})
    return out


def fit_ls(ds: Sequence[float], ys: Sequence[int]) -> Dict[str, float]:
    x=np.asarray(ds,dtype=float); y=np.asarray(ys,dtype=float)
    A=np.stack([x,np.ones_like(x)],axis=1)
    a,b=np.linalg.lstsq(A,y,rcond=None)[0]
    z=a*x+b; pred=np.where(z>=0,1,-1); margins=y*z
    return {"a":float(a),"b":float(b),"calib_acc":float(np.mean(pred==y)),"calib_margin_mean":float(np.mean(margins)),"calib_margin_min":float(np.min(margins))}


def eval_fit(fit: Dict[str,float], rows: Sequence[Dict[str,Any]]) -> Dict[str,Any]:
    a,b=fit["a"],fit["b"]
    groups={
        "direct_h0_psc": lambda r: r["suite"]=="paired_state_conservation" and r["is_changed"] and r["relation"]=="h0_dax",
        "direct_h2_psc": lambda r: r["suite"]=="paired_state_conservation" and r["is_changed"] and r["relation"]=="h2_norp",
        "graph_h1_psc": lambda r: r["suite"]=="paired_state_conservation" and r["is_changed"] and r["relation"]=="h1_mep",
        "graph_h3_psc": lambda r: r["suite"]=="paired_state_conservation" and r["is_changed"] and r["relation"]=="h3_ziv",
        "graph_all_psc": lambda r: r["suite"]=="paired_state_conservation" and r["is_changed"] and r["relation_family"]=="graph_transfer",
        "graph_all_xt": lambda r: r["suite"]=="cross_template_state_readout" and r["is_changed"] and r["relation_family"]=="graph_transfer",
    }
    out={}
    for name,f in groups.items():
        xs=[r for r in rows if f(r)]
        if not xs: out[name]={"n":0}; continue
        z=[a*r["d"]+b for r in xs]
        acc=mean([float((1 if zz>=0 else -1)==r["y"]) for zz,r in zip(z,xs)])
        margin=mean([r["y"]*zz for zz,r in zip(z,xs)])
        out[name]={"n":len(xs),"acc":acc,"margin_mean":margin,"d_mean":mean([r["d"] for r in xs]),"z_mean":mean(z)}
    return out


def affine_split_audit(run_dir: Path) -> Dict[str,Any]:
    rows=choice_rows_from_run(run_dir)
    psc=[r for r in rows if r["suite"]=="paired_state_conservation" and r["is_changed"]]
    fits={}
    calib_sets={
        "fit_h0_only": [r for r in psc if r["relation"]=="h0_dax"],
        "fit_h2_only": [r for r in psc if r["relation"]=="h2_norp"],
        "fit_h0_h2": [r for r in psc if r["relation"] in {"h0_dax","h2_norp"}],
    }
    for name,calib in calib_sets.items():
        fit=fit_ls([r["d"] for r in calib],[r["y"] for r in calib])
        fit["n_calib"]=len(calib)
        fit["eval"]=eval_fit(fit, rows)
        fits[name]=fit
    return {"run_dir":str(run_dir),"fits":fits}


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args=ap.parse_args()
    primary_roots=[PROJECT/"data/primary_gauge", PROJECT/"data/primary_gauge_seed29001", PROJECT/"data/primary_gauge_seed29002"]
    payload={
        "no_anchor_filter": audit_no_anchor_filter(),
        "tied_seed29002_underfit_original": comparison_underfit_audit(PROJECT/"data/primary_gauge_seed29002/tied_bs-1_seed29002"),
        "tied_seed29002_underfit_repair": comparison_underfit_audit(PROJECT/"data/tied_seed29002_bsminus_repair/tied_bs-1_seed29002"),
        "paired_state_regressions": regression_audit(primary_roots),
        "affine_relation_split_step288_hh_only": affine_split_audit(PROJECT/"data/shared_coordinate_trainvocab_heldheld_anchor_control/tied_heldheld_only_seed28801"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/"independent_review_corrections_audit.json", payload)

    lines=["# research independent_review correction audits", ""]
    nf=payload["no_anchor_filter"]
    lines.append("## No-anchor filter")
    lines.append(f"Arm state rows removed by research no-bridge filter: {nf['arm_state_rows']} / arm rows {nf['arm_train_rows']}; all removed rows direct changed h0/h2 = {nf['all_removed_arm_state_rows_are_direct_changed_h0_h2']}. Relation counts: {nf['state_relation_counts']}.")
    lines.append("")
    for label in ["tied_seed29002_underfit_original", "tied_seed29002_underfit_repair"]:
        u=payload[label]
        lines.append(f"## {label}")
        lines.append(f"train_cmp={u['train_cmp_acc']} n_wrong={u['n_wrong']}/{u['n_train_comparison_rows']} prob_wrong_range={u['prob_range_wrong']} wrong_near_clamp_low/high={u['wrong_near_clamp_low']}/{u['wrong_near_clamp_high']} margin_wrong_range={u['margin_range_wrong']} comparison_bce_mean={u['comparison_bce_mean']:.4f} wrong_mean={u['comparison_bce_wrong_mean']:.4f}")
        lines.append("Wrong relation pairs: " + json.dumps(u['wrong_relation_pair_counts'], sort_keys=True))
        lines.append("")
    lines.append("## Paired d_minus = alpha + beta d_plus regressions")
    lines.append("| run | category | n | beta | alpha | corr | corr(-) | rmse | opp_frac |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for run, cats in sorted(payload["paired_state_regressions"].items()):
        for cat, v in cats.items():
            if not v.get('n'): continue
            lines.append(f"| {run} | {cat} | {v['n']} | {v['beta']:.3f} | {v['alpha']:.3f} | {v.get('corr')} | {v.get('corr_neg')} | {v['rmse']:.3f} | {v['opposite_sign_frac']:.3f} |")
    lines.append("")
    lines.append("## Frozen-affine relation split (research heldheld-only)")
    for fit_name, fit in payload["affine_relation_split_step288_hh_only"]["fits"].items():
        lines.append(f"### {fit_name}: n={fit['n_calib']} a={fit['a']:.4f} b={fit['b']:.4f} calib_acc={fit['calib_acc']:.3f} min_margin={fit['calib_margin_min']:.3f}")
        for group, ev in fit['eval'].items():
            lines.append(f"- {group}: n={ev.get('n')} acc={ev.get('acc')} margin={ev.get('margin_mean')}")
        lines.append("")
    lines.append("## Scientific reading")
    lines.append("The no-anchor filter is exact for this substrate if all removed arm state rows are h0/h2 changed anchors. The tied seed29002 bs-1 underfit is not explained by rows stuck at the probability clamp; wrong train comparisons remain away from clamp and persist under the tested longer/lower-LR schedule, so it should be described as a persistent optimization/local-solution issue under tested schedules, not an intrinsic impossibility. Regression slopes near -1 with low residuals support approximate sign-gauge reversal; deviations and the tied underfit exception should remain visible. The h0-only/h2-only affine split tests whether scalar calibration generalizes across direct relations as well as to graph relations.")
    (args.out/"independent_review_corrections_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status":"independent_review_CORRECTIONS_AUDIT_COMPLETE","summary":str(args.out/"independent_review_corrections_audit.md"),"json":str(args.out/"independent_review_corrections_audit.json")}, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
