#!/usr/bin/env python3
"""Parallel research pass-1 AoA sensitivity.

Scores candidate first-pass reorderings by applying beta-scaled log cumulative
exposure deltas to the actual coherent86 mean-surprisal curves.  Model AoAs are
computed by the same threshold/sigmoid rule as BabyLM AoA, with child AoAs taken
from research so child CDI curves are not re-fit for every candidate.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import importlib.util
import json
import math
import pathlib
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

ROOT = _public_path('.')
PREDICTOR = _public_path('experiments/archive/relation_learning/scripts/aoa_pass1_order_predictor.py')
OUT = _public_path('experiments/archive/relation_learning/data/aoa_pass1_sensitivity_parallel')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_mod():
    spec = importlib.util.spec_from_file_location("pred", PREDICTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PREDICTOR}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pred"] = mod
    spec.loader.exec_module(mod)
    return mod


def safe_float(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8"); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def fmt(x: Any) -> str:
    v=safe_float(x)
    return "NA" if v is None else f"{v:.4f}"


def sigmoid_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a / (1 + np.exp(-b * (x - c))) + d


def model_aoa_from_curve(vals_in: list[float], steps_in: list[int], vocab_size: int, n_subword_tokens: int) -> tuple[float | None, str]:
    vals=np.asarray(vals_in,dtype=float); st=np.asarray(steps_in,dtype=float)
    m=np.isfinite(vals)
    if int(m.sum())<3:
        return None,"lt3"
    vals=vals[m]; st=st[m]
    random_chance=float(n_subword_tokens*math.log(vocab_size))
    min_surprisal=float(np.min(vals))
    threshold=float(random_chance-0.5*(random_chance-min_surprisal))
    neg=-vals; log_steps=np.log10(st+1)
    rng=float(np.max(neg)-np.min(neg))
    if not math.isfinite(rng):
        return None,"nonfinite"
    p0=[rng,1.0,float(np.mean(log_steps)),float(np.min(neg))]
    lower=[0.0,0.0,float(np.min(log_steps)-1),float(np.min(neg)-2*rng-1)]
    upper=[10*rng+1,100.0,float(np.max(log_steps)+1),float(np.max(neg)+1)]
    try:
        popt,_=curve_fit(sigmoid_function,log_steps,neg,p0=p0,bounds=(lower,upper),maxfev=20000)
        a,b,c,d=[float(z) for z in popt]
    except Exception:
        return None,"fit_exception"
    neg_threshold=-threshold
    if b<=1e-6 or a<=1e-6:
        return None,"flat_or_zero_amplitude"
    if neg_threshold<=d:
        return None,"threshold_below_lower_asymptote"
    if neg_threshold>=a+d:
        return None,"threshold_above_upper_asymptote"
    try:
        log_aoa=float(c-math.log((a/(neg_threshold-d))-1)/b)
        aoa_step=float(10**log_aoa-1)
    except Exception:
        return None,"solve_exception"
    if aoa_step<float(st[0]):
        return None,"before_first"
    if aoa_step>float(st[-1]):
        return None,"after_last"
    return log_aoa,"ok"


def score_job(job: dict[str, Any]) -> dict[str, Any]:
    mode=job["mode"]; beta=float(job["beta"]); checkpoints=job["checkpoints"]; words=job["words"]
    means=job["means"]; dlog=job.get("dlog"); swl=job["swl"]; child=job["child_aoa"]; vocab_size=int(job["vocab_size"])
    model_aoas=[]; child_vals=[]; statuses=Counter()
    for w in words:
        if w not in child:
            statuses["no_child_aoa"]+=1; continue
        if dlog is None:
            vals=[means[(w,ck)] for ck in checkpoints]
        else:
            vals=[means[(w,ck)] + beta*dlog[(w,ck)] for ck in checkpoints]
        aoa,status=model_aoa_from_curve(vals,checkpoints,vocab_size,swl[w])
        statuses[status]+=1
        if aoa is not None:
            model_aoas.append(float(aoa)); child_vals.append(float(child[w]))
    out={"mode":mode,"beta":beta,"oracle_only": bool(job.get("oracle_only",False)),"n_words":len(model_aoas),"status_counts":dict(statuses)}
    if len(model_aoas)>=3 and len(set(model_aoas))>1 and len(set(child_vals))>1:
        pr=pearsonr(model_aoas,child_vals); sr=spearmanr(model_aoas,child_vals)
        out.update({"unclipped_r":float(pr.statistic),"p_value":float(pr.pvalue),"spearman_r":float(sr.statistic),"spearman_p":float(sr.pvalue),"curve_fitness_clipped":0.0 if pr.pvalue>0.1 else float(pr.statistic)})
    else:
        out.update({"unclipped_r":None,"p_value":None,"spearman_r":None,"spearman_p":None,"curve_fitness_clipped":0.0})
    return out


def corr(xs: list[Any], ys: list[Any]) -> dict[str, Any]:
    vals=[]
    for x,y in zip(xs,ys):
        fx=safe_float(x); fy=safe_float(y)
        if fx is not None and fy is not None:
            vals.append((fx,fy))
    if len(vals)<3 or len({x for x,_ in vals})<2 or len({y for _,y in vals})<2:
        return {"n":len(vals),"pearson_r":None,"pearson_p":None,"spearman_r":None,"spearman_p":None}
    pr=pearsonr([x for x,_ in vals],[y for _,y in vals]); sr=spearmanr([x for x,_ in vals],[y for _,y in vals])
    return {"n":len(vals),"pearson_r":float(pr.statistic),"pearson_p":float(pr.pvalue),"spearman_r":float(sr.statistic),"spearman_p":float(sr.pvalue)}


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir",default=str(OUT))
    ap.add_argument("--betas",default="-0.10,-0.20,-0.36,-0.50,-0.75,-1.00,-1.50")
    ap.add_argument("--modes",default="source_childes_only_then_current,source_childes_first,sort__childes_freq,sort__childes_ratio,sort__spoken_freq,sort__childes_composite,sort__whole_freq,source_written_first_control,sort__child_early")
    ap.add_argument("--workers",type=int,default=12)
    args=ap.parse_args()
    out_dir=pathlib.Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    mod=load_mod()
    words_obs, checkpoints, mean_records=mod.load_aggregate_surprisals()
    means={k:float(v["mean_surprisal"]) for k,v in mean_records.items()}
    research=mod.load_step106_records()
    target_words=sorted(set(words_obs)|set(mod.load_cdi_words()))
    print(json.dumps({"event":"start","words_obs":len(words_obs),"target_words":len(target_words),"utc":now()}),flush=True)
    scan=mod.scan_stream_for_exposures(mod.STREAM,target_words,checkpoints)
    feats=mod.build_word_features(target_words,research,scan)
    mod.score_pass1_rows(scan["pass1_rows"],feats)
    tok,_=mod.load_tokenizer_and_evaluator(out_dir)
    swl=mod.subword_lengths(tok,words_obs)
    vocab_size=int(getattr(tok,"vocab_size",len(tok)))
    child_aoa={w:float(v) for w,rec in research.items() if (v:=safe_float(rec.get("child_aoa"))) is not None}
    measured_model={w:float(v) for w,rec in research.items() if (v:=safe_float(rec.get("model_aoa"))) is not None}

    modes=[m.strip() for m in args.modes.split(',') if m.strip()]
    betas=[float(b) for b in args.betas.split(',') if b.strip()]
    jobs=[{"mode":"current_actual","beta":0.0,"checkpoints":checkpoints,"words":words_obs,"means":means,"dlog":None,"swl":swl,"child_aoa":child_aoa,"vocab_size":vocab_size,"oracle_only":False}]
    exposure_delta_summary=[]
    for mode in modes:
        order=mod.candidate_order(scan["pass1_rows"],mode)
        exp=mod.pass1_candidate_exposures(scan["pass1_rows"],order,checkpoints,scan["exposures"])
        dlog={(w,ck): math.log1p(float(exp[ck].get(w,0.0)))-math.log1p(float(scan["exposures"][ck].get(w,0.0))) for w in words_obs for ck in checkpoints}
        exposure_delta_summary.append({"mode":mode,"mean_abs_dlog_1M":float(np.mean([abs(dlog[(w,1_000_000)]) for w in words_obs])),"mean_dlog_1M":float(np.mean([dlog[(w,1_000_000)] for w in words_obs])),"max_abs_dlog_after10M":float(max(abs(dlog[(w,ck)]) for w in words_obs for ck in checkpoints if ck>10_000_000))})
        for beta in betas:
            jobs.append({"mode":mode,"beta":beta,"checkpoints":checkpoints,"words":words_obs,"means":means,"dlog":dlog,"swl":swl,"child_aoa":child_aoa,"vocab_size":vocab_size,"oracle_only":mode=="sort__child_early"})
    results=[]
    t0=time.time()
    with ProcessPoolExecutor(max_workers=max(1,args.workers)) as ex:
        futs=[ex.submit(score_job,j) for j in jobs]
        for fut in as_completed(futs):
            res=fut.result(); results.append(res)
            print(json.dumps({"event":"job_done","mode":res["mode"],"beta":res["beta"],"clipped":res.get("curve_fitness_clipped"),"r":res.get("unclipped_r"),"p":res.get("p_value"),"n":res.get("n_words"),"elapsed_sec":round(time.time()-t0,1)}),flush=True)
    # add delta values to candidate beta rows
    delta_by_mode={r["mode"]:r for r in exposure_delta_summary}
    for r in results:
        d=delta_by_mode.get(r["mode"])
        if d:
            beta=float(r["beta"])
            r["mean_abs_delta_surprisal_1M"]=abs(beta)*d["mean_abs_dlog_1M"]
            r["mean_delta_surprisal_1M"]=beta*d["mean_dlog_1M"]
            r["max_abs_delta_after_10M"]=abs(beta)*d["max_abs_dlog_after10M"]
    rows_sorted=sorted(results,key=lambda r:(bool(r.get("oracle_only")), -(float(r.get("curve_fitness_clipped") or 0.0)), -(float(r.get("unclipped_r") or -999)), str(r.get("mode")), float(r.get("beta",0.0))))
    write_csv(out_dir/"sensitivity_parallel_scores.csv",rows_sorted)
    write_csv(out_dir/"exposure_delta_summary.csv",exposure_delta_summary)
    cur=next(r for r in results if r["mode"]=="current_actual")
    legal=[r for r in rows_sorted if not r.get("oracle_only") and r["mode"]!="current_actual"][:16]
    oracle=[r for r in rows_sorted if r.get("oracle_only")][:10]
    out={"status":"AOA_PASS1_SENSITIVITY_PARALLEL_DONE","created_utc":now(),"scope":"Actual coherent86 mean-surprisal curves plus beta times pass-1 candidate-vs-current log exposure deltas; after 10M exposure deltas are zero. Model AoA fits match the official threshold/sigmoid rule; child AoAs are research precomputed values.","scan":{"rows":scan['rows'],"consumed_words":scan['consumed_words'],"pass1_rows":len(scan['pass1_rows']),"pass1_words":scan['pass1_words'],"elapsed_sec":round(scan['elapsed_sec'],3)},"validation":{"current_actual":cur,"expected":{"n_words":225,"pearson_r":-0.03485418140387646,"pearson_p":0.6030225874932932},"whole_stream_frequency_vs_measured_model_aoa":corr([feats[w].get('whole_logfreq') for w in measured_model],[measured_model[w] for w in measured_model]),"childes_frequency_vs_child_aoa":corr([feats[w].get('childes_logfreq') for w in child_aoa],[child_aoa[w] for w in child_aoa]),"childes_minus_whole_vs_child_aoa":corr([feats[w].get('childes_minus_whole') for w in child_aoa],[child_aoa[w] for w in child_aoa])},"best_legal":legal,"best_oracle":oracle,"outputs":{"summary_json":rel(out_dir/'summary.json'),"summary_md":rel(out_dir/'summary.md'),"scores_csv":rel(out_dir/'sensitivity_parallel_scores.csv'),"delta_csv":rel(out_dir/'exposure_delta_summary.csv')}}
    (out_dir/'summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=["# research AoA pass-1 sensitivity parallel","",out['scope'],"",f"Current actual reproduction: clipped {fmt(cur.get('curve_fitness_clipped'))}, unclipped r {fmt(cur.get('unclipped_r'))}, p {fmt(cur.get('p_value'))}, n {cur.get('n_words')}; research expected r -0.0349, p 0.6030, n 225.",f"Whole-stream frequency vs measured model AoA r {fmt(out['validation']['whole_stream_frequency_vs_measured_model_aoa'].get('pearson_r'))}; CHILDES frequency vs child AoA r {fmt(out['validation']['childes_frequency_vs_child_aoa'].get('pearson_r'))}; CHILDES-minus-whole vs child AoA r {fmt(out['validation']['childes_minus_whole_vs_child_aoa'].get('pearson_r'))}.","","## Best legal rows","","| mode | beta | clipped score | unclipped r | p | n | mean |ΔS_1M| | after10M |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in legal:
        lines.append(f"| {r['mode']} | {r['beta']} | {fmt(r.get('curve_fitness_clipped'))} | {fmt(r.get('unclipped_r'))} | {fmt(r.get('p_value'))} | {r.get('n_words')} | {fmt(r.get('mean_abs_delta_surprisal_1M'))} | {fmt(r.get('max_abs_delta_after_10M'))} |")
    lines += ["","## Oracle-only rows","","| mode | beta | clipped score | unclipped r | p | n | mean |ΔS_1M| |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in oracle:
        lines.append(f"| {r['mode']} | {r['beta']} | {fmt(r.get('curve_fitness_clipped'))} | {fmt(r.get('unclipped_r'))} | {fmt(r.get('p_value'))} | {r.get('n_words')} | {fmt(r.get('mean_abs_delta_surprisal_1M'))} |")
    lines += ["",f"Full JSON: `{rel(out_dir/'summary.json')}`"]
    (out_dir/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({"status":out["status"],"summary_json":rel(out_dir/'summary.json'),"summary_md":rel(out_dir/'summary.md'),"current":cur,"best_legal":legal[:8],"best_oracle":oracle[:5]},indent=2,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
