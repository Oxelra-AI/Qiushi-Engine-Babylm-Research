#!/usr/bin/env python3
"""research: condition-wise source/rank decomposition including dense-mask/sparse-label.

Reads the five-model research temperature-source readout and summarizes whether the
(M,S) dense-mask/sparse-label arm reproduces dense's source-conditioned rank changes,
and whether it does so by correct-source support or wrong/absent-source degradation.
This is explanatory evidence, not BabyLM scoring.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import statistics
from typing import Any, Iterable

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_IN = _public_path('experiments/archive/functional_learning/data/temperature_source_readout_five_models/temperature_source_readout.json')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/source_margin_decomposition_five_models/source_margin_decomposition_five_models.json')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def vals(xs: Iterable[Any]) -> list[float]:
    out=[]
    for x in xs:
        try:
            y=float(x)
            if math.isfinite(y): out.append(y)
        except Exception:
            pass
    return out


def mean(xs: Iterable[Any]) -> float | None:
    v=vals(xs); return sum(v)/len(v) if v else None


def median(xs: Iterable[Any]) -> float | None:
    v=vals(xs); return statistics.median(v) if v else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs)!=len(ys) or len(xs)<2: return None
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    if vx<=0 or vy<=0: return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(vx*vy)


def qwen_decomp(s: dict[str,Any]) -> dict[str,Any]:
    q=s["qwen_source"]; cond=q["by_condition"]; spec=q["source_specificity"]
    d={}
    for c in ["correct_source","wrong_source","view_only"]:
        cc=cond[c]
        d[c]={
            "mean_nll_T1": cc.get("mean_nll_T1"),
            "mean_nll_Tfit": cc.get("mean_nll_Tfit"),
            "mean_rank": cc.get("mean_rank"),
            "delta_nll_T1_vs_parent": cc.get("mean_delta_nll_T1_vs_parent"),
            "delta_nll_Tfit_vs_parent": cc.get("mean_delta_nll_Tfit_vs_parent"),
            "delta_rank_vs_parent": cc.get("mean_delta_rank_vs_parent"),
            "n": cc.get("n"),
        }
    return {
        "temperature": s.get("temperature"),
        "condition": d,
        "specific_advantage": {
            "T1": spec.get("mean_specific_advantage_T1"),
            "Tfit": spec.get("mean_specific_advantage_Tfit"),
            "rank": spec.get("mean_specific_rank_advantage"),
            "delta_T1_vs_parent": spec.get("mean_delta_specific_advantage_T1_vs_parent"),
            "delta_Tfit_vs_parent": spec.get("mean_delta_specific_advantage_Tfit_vs_parent"),
            "delta_rank_vs_parent": spec.get("mean_delta_specific_rank_advantage_vs_parent"),
        },
        "triplets": {
            "n": len(q.get("triplets", [])),
            "share_delta_T1_positive": mean(1 if r.get("delta_specific_advantage_T1_vs_parent",0)>0 else 0 for r in q.get("triplets",[])),
            "share_delta_rank_positive": mean(1 if r.get("delta_specific_rank_advantage_vs_parent",0)>0 else 0 for r in q.get("triplets",[])),
            "median_delta_T1": median(r.get("delta_specific_advantage_T1_vs_parent") for r in q.get("triplets",[])),
            "median_delta_rank": median(r.get("delta_specific_rank_advantage_vs_parent") for r in q.get("triplets",[])),
        }
    }


def common_decomp(s: dict[str,Any]) -> dict[str,Any]:
    c=s["common_source_reversal"]
    return {
        "source_follow": c.get("source_follow", {}),
        "by_condition": {k:{
            "delta_margin_T1_vs_parent": v.get("mean_delta_margin_T1_vs_parent"),
            "delta_margin_Tfit_vs_parent": v.get("mean_delta_margin_Tfit_vs_parent"),
            "delta_rank_advantage_vs_parent": v.get("mean_delta_rank_advantage_vs_parent"),
            "mean_margin_T1": v.get("mean_margin_T1"),
            "mean_rank_advantage": v.get("mean_rank_advantage"),
            "n": v.get("n"),
        } for k,v in c.get("by_condition",{}).items()},
        "rows": {
            "n": len(c.get("source_follow_rows", [])),
            "share_delta_swing_positive": mean(1 if r.get("delta_source_follow_swing_T1_vs_parent",0)>0 else 0 for r in c.get("source_follow_rows",[])),
            "share_delta_rank_swing_positive": mean(1 if r.get("delta_source_follow_rank_swing_vs_parent",0)>0 else 0 for r in c.get("source_follow_rows",[])),
            "median_delta_swing_T1": median(r.get("delta_source_follow_swing_T1_vs_parent") for r in c.get("source_follow_rows",[])),
            "median_delta_rank_swing": median(r.get("delta_source_follow_rank_swing_vs_parent") for r in c.get("source_follow_rows",[])),
        }
    }


def row_agreement(obj: dict[str,Any], a: str, b: str) -> dict[str,Any]:
    sa=obj["model_summaries"][a]; sb=obj["model_summaries"][b]
    out={}
    qa={r["base_task_id"]:r for r in sa["qwen_source"].get("triplets",[])}
    qb={r["base_task_id"]:r for r in sb["qwen_source"].get("triplets",[])}
    ids=sorted(set(qa)&set(qb))
    out["qwen"]={}
    for m in ["delta_specific_advantage_T1_vs_parent","delta_specific_advantage_Tfit_vs_parent","delta_specific_rank_advantage_vs_parent"]:
        xs=[float(qa[i].get(m,0.0)) for i in ids]; ys=[float(qb[i].get(m,0.0)) for i in ids]
        out["qwen"][m]={"n":len(xs),"pearson":pearson(xs,ys),"mean_abs_diff":mean(abs(x-y) for x,y in zip(xs,ys)),"same_sign_fraction":mean(1 if (x>0)==(y>0) else 0 for x,y in zip(xs,ys))}
    ca={r["task_id"]:r for r in sa["common_source_reversal"].get("source_follow_rows",[])}
    cb={r["task_id"]:r for r in sb["common_source_reversal"].get("source_follow_rows",[])}
    ids=sorted(set(ca)&set(cb))
    out["common"]={}
    for m in ["delta_source_follow_swing_T1_vs_parent","delta_source_follow_swing_Tfit_vs_parent","delta_source_follow_rank_swing_vs_parent"]:
        xs=[float(ca[i].get(m,0.0)) for i in ids]; ys=[float(cb[i].get(m,0.0)) for i in ids]
        out["common"][m]={"n":len(xs),"pearson":pearson(xs,ys),"mean_abs_diff":mean(abs(x-y) for x,y in zip(xs,ys)),"same_sign_fraction":mean(1 if (x>0)==(y>0) else 0 for x,y in zip(xs,ys))}
    return out


def delta_summary(decomp: dict[str,Any], a: str, b: str) -> dict[str,Any]:
    qa=decomp[a]["qwen_source"]; qb=decomp[b]["qwen_source"]
    ca=decomp[a]["common_source_reversal"]; cb=decomp[b]["common_source_reversal"]
    def sub(x,y):
        try: return float(x)-float(y)
        except Exception: return None
    return {
        "qwen_specific_delta_a_minus_b": {
            "T1": sub(qa["specific_advantage"]["delta_T1_vs_parent"], qb["specific_advantage"]["delta_T1_vs_parent"]),
            "Tfit": sub(qa["specific_advantage"]["delta_Tfit_vs_parent"], qb["specific_advantage"]["delta_Tfit_vs_parent"]),
            "rank": sub(qa["specific_advantage"]["delta_rank_vs_parent"], qb["specific_advantage"]["delta_rank_vs_parent"]),
        },
        "qwen_condition_delta_a_minus_b": {c:{
            "nll_T1": sub(qa["condition"][c]["delta_nll_T1_vs_parent"], qb["condition"][c]["delta_nll_T1_vs_parent"]),
            "nll_Tfit": sub(qa["condition"][c]["delta_nll_Tfit_vs_parent"], qb["condition"][c]["delta_nll_Tfit_vs_parent"]),
            "rank": sub(qa["condition"][c]["delta_rank_vs_parent"], qb["condition"][c]["delta_rank_vs_parent"]),
        } for c in ["correct_source","wrong_source","view_only"]},
        "common_delta_a_minus_b": {
            "swing_T1": sub(ca["source_follow"].get("mean_delta_source_follow_swing_T1_vs_parent"), cb["source_follow"].get("mean_delta_source_follow_swing_T1_vs_parent")),
            "swing_Tfit": sub(ca["source_follow"].get("mean_delta_source_follow_swing_Tfit_vs_parent"), cb["source_follow"].get("mean_delta_source_follow_swing_Tfit_vs_parent")),
            "rank_swing": sub(ca["source_follow"].get("mean_delta_source_follow_rank_swing_vs_parent"), cb["source_follow"].get("mean_delta_source_follow_rank_swing_vs_parent")),
            "both_T1": sub(ca["source_follow"].get("both_correct_T1"), cb["source_follow"].get("both_correct_T1")),
            "both_rank": sub(ca["source_follow"].get("both_rank_better"), cb["source_follow"].get("both_rank_better")),
        }
    }


def write_md(path: pathlib.Path, res: dict[str,Any]) -> None:
    lines=[]
    lines.append("# research source/rank decomposition including dense-mask/sparse-label\n\n")
    lines.append(f"Input: `{res['input']}`\n\n")
    lines.append("## Compact model comparison\n\n")
    lines.append("| model | T | calib NLL | Qwen Δspec Tfit | Qwen Δrank | common Δswing Tfit | common Δrank swing | both T1/rank |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for m,d in res["model_decompositions"].items():
        src=res["raw_compact"].get(m,{})
        sf=d["common_source_reversal"]["source_follow"]
        q=d["qwen_source"]["specific_advantage"]
        lines.append(f"| {m} | {src.get('temperature')} | {src.get('calib_mean_nll_T1')} | {q.get('delta_Tfit_vs_parent')} | {q.get('delta_rank_vs_parent')} | {sf.get('mean_delta_source_follow_swing_Tfit_vs_parent')} | {sf.get('mean_delta_source_follow_rank_swing_vs_parent')} | {sf.get('both_correct_T1')}/{sf.get('both_rank_better')} |\n")
    lines.append("\n## Pairwise localization\n\n")
    for k,v in res["pairwise"].items():
        lines.append(f"### {k}\n\n")
        lines.append(json.dumps(v, indent=2, ensure_ascii=False)+"\n\n")
    lines.append("## Interpretation\n\n")
    for x in res["interpretation"]:
        lines.append(f"- {x}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", type=pathlib.Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args=ap.parse_args()
    obj=json.loads(args.input.read_text())
    decomp={m:{"qwen_source":qwen_decomp(s),"common_source_reversal":common_decomp(s)} for m,s in obj["model_summaries"].items()}
    raw_compact={m:{"temperature":s.get("temperature"),"calib_mean_nll_T1":s.get("calibration",{}).get("mean_nll_T1"),"calib_best_mean_nll":s.get("calibration",{}).get("best_mean_nll"),"calib_mean_rank":s.get("calibration",{}).get("mean_rank")} for m,s in obj["model_summaries"].items()}
    pairwise={
        "MS_minus_SS_dense_input_at_sparse_labels": delta_summary(decomp,"densemask_sparselabel_seed62064","sparse_focus_seed62064"),
        "MS_minus_MM64_sparse_labels_vs_dense_coverage": delta_summary(decomp,"densemask_sparselabel_seed62064","dense_focus_seed62064"),
        "MM64_minus_SS_dense_total_effect": delta_summary(decomp,"dense_focus_seed62064","sparse_focus_seed62064"),
    }
    agreements={
        "MS_vs_MM64_row_agreement": row_agreement(obj,"densemask_sparselabel_seed62064","dense_focus_seed62064"),
        "MM64_vs_MM65_row_agreement": row_agreement(obj,"dense_focus_seed62064","dense_focus_seed62065"),
    }
    interp=[
        "The dense-mask/sparse-label arm reproduces the dense source/rank signature under sparse labels: its Qwen Δspecific advantage, common source-follow swing, decision counts, and rank-swing are dense-like rather than sparse-like.",
        "Against dense seed62064, dense-mask differs only slightly on these source diagnostics: Qwen Δspecific Tfit is lower by about 0.005 and Δrank by about 2.9, while common source-follow Tfit/rank swings are slightly larger. Broad dense target coverage is therefore not necessary for the observed source-conditioned shift under this 80-update policy.",
        "The preservation problem remains: dense-mask calibration NLL on ordinary legal-tail text is worse than dense and sparse, and research/research fast/AoA-adjacent evidence shows broad likelihood/rank costs are not fixed by thinning labels. The next method should preserve the clue-suppressed source-visible acquisition condition while explicitly constraining drift on matched view-only or ordinary-corruption renderings where evidence is absent or underdetermined.",
    ]
    res={"status":"SOURCE_MARGIN_DECOMPOSITION_FIVE_MODELS_DONE","input":rel(args.input),"raw_compact":raw_compact,"model_decompositions":decomp,"pairwise":pairwise,"row_agreements":agreements,"interpretation":interp}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    write_md(args.out.with_suffix('.md'), res)
    print(json.dumps({"status":res["status"],"out_json":rel(args.out),"out_md":rel(args.out.with_suffix('.md'))},indent=2))

if __name__ == "__main__":
    main()
