#!/usr/bin/env python3
"""Fast research pass-1 AoA sensitivity using precomputed child AoAs.

This imports the research pass-1 exposure machinery, applies candidate first-pass
exposure deltas to the actual coherent86 mean-surprisal curves, and computes the
same model-AoA threshold fit plus Pearson p>0.1 clipping semantics. It avoids
re-fitting child CDI curves for every candidate by using research's child_aoa.
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
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = _public_path('.')
PREDICTOR = _public_path('experiments/archive/relation_learning/scripts/aoa_pass1_order_predictor.py')
OUT = _public_path('experiments/archive/relation_learning/data/aoa_pass1_sensitivity_fast')


def load_mod():
    spec = importlib.util.spec_from_file_location("pred", PREDICTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PREDICTOR}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pred"] = mod
    spec.loader.exec_module(mod)
    return mod


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


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
        path.write_text("\n", encoding="utf-8")
        return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def fmt(x: Any) -> str:
    v = safe_float(x)
    return "NA" if v is None else f"{v:.4f}"


def score_curves(mod: Any, curves: dict[tuple[str,int], float], words: list[str], checkpoints: list[int], child_aoa: dict[str,float], swl: dict[str,int], vocab_size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_aoas=[]; child=[]; rows=[]; status_counts=Counter()
    for w in words:
        if w not in child_aoa:
            status_counts["no_child_aoa"] += 1
            continue
        vals=[curves[(w, ck)] for ck in checkpoints]
        aoa,status = mod.model_aoa_from_curve(vals, checkpoints, vocab_size, swl[w])
        status_counts[status] += 1
        rows.append({"word": w, "pred_model_aoa": aoa, "child_aoa": child_aoa[w], "status": status})
        if aoa is not None:
            model_aoas.append(float(aoa)); child.append(float(child_aoa[w]))
    out={"n_words": len(model_aoas), "status_counts": dict(status_counts)}
    if len(model_aoas) >= 3 and len(set(model_aoas)) > 1 and len(set(child)) > 1:
        pr=pearsonr(model_aoas, child); sr=spearmanr(model_aoas, child)
        out.update({"unclipped_r": float(pr.statistic), "p_value": float(pr.pvalue), "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue), "curve_fitness_clipped": 0.0 if pr.pvalue > 0.1 else float(pr.statistic)})
    else:
        out.update({"unclipped_r": None, "p_value": None, "spearman_r": None, "spearman_p": None, "curve_fitness_clipped": 0.0})
    return out, rows


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
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--betas", default="-0.10,-0.20,-0.36,-0.50,-0.75,-1.00,-1.50")
    ap.add_argument("--modes", default="source_childes_only_then_current,source_childes_first,sort__childes_freq,sort__childes_ratio,sort__spoken_freq,sort__childes_composite,sort__whole_freq,source_written_first_control,sort__child_early")
    args=ap.parse_args()
    out_dir=pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mod=load_mod()
    betas=[float(x) for x in args.betas.split(',') if x.strip()]
    modes=[x.strip() for x in args.modes.split(',') if x.strip()]
    words_obs, checkpoints, mean_records = mod.load_aggregate_surprisals()
    means = {k: float(v["mean_surprisal"]) for k, v in mean_records.items()}
    research = mod.load_step106_records()
    target_words=sorted(set(words_obs)|set(mod.load_cdi_words()))
    print(json.dumps({"event":"start","words_obs":len(words_obs),"target_words":len(target_words),"betas":betas,"modes":modes,"utc":now()}), flush=True)
    scan = mod.scan_stream_for_exposures(mod.STREAM, target_words, checkpoints)
    feats = mod.build_word_features(target_words, research, scan)
    mod.score_pass1_rows(scan["pass1_rows"], feats)
    tok,_ev = mod.load_tokenizer_and_evaluator(out_dir)
    swl = mod.subword_lengths(tok, words_obs)
    vocab_size=int(getattr(tok,"vocab_size",len(tok)))
    child_aoa={w: float(v) for w,rec in research.items() if (v:=safe_float(rec.get('child_aoa'))) is not None}
    measured_model={w: float(v) for w,rec in research.items() if (v:=safe_float(rec.get('model_aoa'))) is not None}
    current_score, current_word_rows = score_curves(mod, means, words_obs, checkpoints, child_aoa, swl, vocab_size)
    # Validation against research measured model AoA is direct because current_score uses actual mean curves.
    cur_model_by_word={r['word']:r['pred_model_aoa'] for r in current_word_rows if r.get('pred_model_aoa') is not None}
    validation_corr=corr([cur_model_by_word.get(w) for w in measured_model], [measured_model[w] for w in measured_model])
    summary_rows=[{"mode":"current_actual","beta":0.0,"oracle_only":False,**current_score,"pred_vs_step106_model_aoa_r":validation_corr.get('pearson_r'),"pred_vs_step106_model_aoa_spearman":validation_corr.get('spearman_r')}]
    word_rows=[]
    for mode in modes:
        order=mod.candidate_order(scan["pass1_rows"], mode)
        exp=mod.pass1_candidate_exposures(scan["pass1_rows"], order, checkpoints, scan["exposures"])
        dlog={(w,ck): math.log1p(float(exp[ck].get(w,0.0)))-math.log1p(float(scan["exposures"][ck].get(w,0.0))) for w in words_obs for ck in checkpoints}
        for beta in betas:
            curves={(w,ck): float(means[(w,ck)] + beta*dlog[(w,ck)]) for w in words_obs for ck in checkpoints}
            sc, wr = score_curves(mod, curves, words_obs, checkpoints, child_aoa, swl, vocab_size)
            sc.update({
                "mode": mode,
                "beta": beta,
                "oracle_only": mode == "sort__child_early",
                "mean_abs_delta_surprisal_1M": float(np.mean([abs(beta*dlog[(w,1_000_000)]) for w in words_obs])),
                "mean_delta_surprisal_1M": float(np.mean([beta*dlog[(w,1_000_000)] for w in words_obs])),
                "max_abs_delta_after_10M": float(max(abs(beta*dlog[(w,ck)]) for w in words_obs for ck in checkpoints if ck>10_000_000)),
            })
            summary_rows.append(sc)
            for r in wr:
                if r["word"] in {"airplane","dog","mommy","car","book","zoo","bath","spaghetti"}:
                    word_rows.append({"mode":mode,"beta":beta,**r})
        print(json.dumps({"event":"mode_done","mode":mode}), flush=True)
    rows_sorted=sorted(summary_rows, key=lambda r:(bool(r.get('oracle_only')), -(float(r.get('curve_fitness_clipped') or 0.0)), -(float(r.get('unclipped_r') or -999)), str(r.get('mode')), float(r.get('beta',0))))
    write_csv(out_dir/"sensitivity_fast_scores.csv", rows_sorted)
    write_csv(out_dir/"selected_word_predictions.csv", word_rows)
    legal=[r for r in rows_sorted if not r.get('oracle_only') and r.get('mode')!='current_actual'][:12]
    oracle=[r for r in rows_sorted if r.get('oracle_only')][:12]
    out={
        "status":"AOA_PASS1_SENSITIVITY_FAST_DONE",
        "created_utc":now(),
        "scope":"Actual coherent86 mean-surprisal curves plus beta times candidate-vs-current log cumulative exposure changes at official pass-1 checkpoints; after 10M changes are zero. Child AoAs are precomputed research values; model AoA and Pearson p-value semantics match the official implementation.",
        "inputs":{"predictor":rel(PREDICTOR),"stream":rel(mod.STREAM),"surprisal":rel(mod.SURPRISAL),"words":rel(mod.WORDS)},
        "scan":{"rows":scan['rows'],"consumed_words":scan['consumed_words'],"pass1_rows":len(scan['pass1_rows']),"pass1_words":scan['pass1_words'],"elapsed_sec":round(scan['elapsed_sec'],3)},
        "validation":{"current_actual":summary_rows[0],"expected":{"n_words":225,"pearson_r":-0.03485418140387646,"pearson_p":0.6030225874932932},"whole_stream_frequency_vs_measured_model_aoa":corr([feats[w].get('whole_logfreq') for w in measured_model],[measured_model[w] for w in measured_model]),"childes_frequency_vs_child_aoa":corr([feats[w].get('childes_logfreq') for w in child_aoa],[child_aoa[w] for w in child_aoa]),"childes_minus_whole_vs_child_aoa":corr([feats[w].get('childes_minus_whole') for w in child_aoa],[child_aoa[w] for w in child_aoa])},
        "best_legal":legal,
        "best_oracle":oracle,
        "outputs":{"summary_json":rel(out_dir/'summary.json'),"summary_md":rel(out_dir/'summary.md'),"scores_csv":rel(out_dir/'sensitivity_fast_scores.csv'),"selected_word_predictions":rel(out_dir/'selected_word_predictions.csv')},
    }
    (out_dir/'summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=["# research AoA pass-1 sensitivity fast","",out['scope'],"",f"Current actual reproduction: clipped {fmt(summary_rows[0].get('curve_fitness_clipped'))}, unclipped r {fmt(summary_rows[0].get('unclipped_r'))}, p {fmt(summary_rows[0].get('p_value'))}, n {summary_rows[0].get('n_words')}; research expected r -0.0349, p 0.6030, n 225.",f"Whole-stream frequency vs measured model AoA r {fmt(out['validation']['whole_stream_frequency_vs_measured_model_aoa'].get('pearson_r'))}; CHILDES frequency vs child AoA r {fmt(out['validation']['childes_frequency_vs_child_aoa'].get('pearson_r'))}; CHILDES-minus-whole vs child AoA r {fmt(out['validation']['childes_minus_whole_vs_child_aoa'].get('pearson_r'))}.","","## Best legal sensitivity rows","","| mode | beta | clipped score | unclipped r | p | n | mean |ΔS_1M| | after10M |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in legal:
        lines.append(f"| {r['mode']} | {r['beta']} | {fmt(r.get('curve_fitness_clipped'))} | {fmt(r.get('unclipped_r'))} | {fmt(r.get('p_value'))} | {r.get('n_words')} | {fmt(r.get('mean_abs_delta_surprisal_1M'))} | {fmt(r.get('max_abs_delta_after_10M'))} |")
    lines += ["","## Oracle-only sensitivity rows","","| mode | beta | clipped score | unclipped r | p | n | mean |ΔS_1M| |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in oracle:
        lines.append(f"| {r['mode']} | {r['beta']} | {fmt(r.get('curve_fitness_clipped'))} | {fmt(r.get('unclipped_r'))} | {fmt(r.get('p_value'))} | {r.get('n_words')} | {fmt(r.get('mean_abs_delta_surprisal_1M'))} |")
    lines += ["",f"Full JSON: `{rel(out_dir/'summary.json')}`"]
    (out_dir/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({"status":out['status'],"summary_json":rel(out_dir/'summary.json'),"summary_md":rel(out_dir/'summary.md'),"current":summary_rows[0],"best_legal":legal[:6],"best_oracle":oracle[:6]},indent=2,ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
