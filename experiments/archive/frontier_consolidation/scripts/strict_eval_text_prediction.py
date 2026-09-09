#!/usr/bin/env python3
"""research strict-evaluation-text variant of the register predictor.

This CPU-only script is run before MAX-register scores are read.  It repairs one
limitation of `distribution_proximity_prediction.py`: EWoK/COMPS/Entity
texts are extracted with task-aware stimulus fields rather than broad key-name
heuristics.  Training blocks and calibration equation remain identical.

No model loading, training, official evaluation, GPU work, GlobalPIQA,
SuperGLUE, AoA, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import importlib.util
import json
import math
import pathlib
import statistics
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
PRED_SCRIPT = WS / "scripts/distribution_proximity_prediction.py"
OUT = WS / "data/strict_eval_text_prediction"
CONTRAST_CSV = WS / "data/reference_decomposition_readout/contrast_window_summaries.csv"
EVAL_ROOT = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
PAIR_ROWS = 7923
FAMILIES = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading", "Entity"]
EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
ARM_BLOCK = {"V": "admitted_view", "B": "admitted_breadth", "R": "admitted_repeat"}
ARM_CONTRAST = {"V": "D1_VminusCmax", "B": "D1_BminusCmax", "R": "D1_RminusCmax"}
SPREAD = {"exEntity5": 0.1765, "cheap6": 0.4199}


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_pred_module():
    spec = importlib.util.spec_from_file_location("distribution_proximity_prediction", PRED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PRED_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def clean(s: Any) -> str:
    return " ".join(str(s).split()) if s is not None else ""


def iter_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def strict_eval_family_texts(family: str) -> tuple[list[str], dict[str, Any]]:
    texts: list[str] = []
    files: list[str] = []
    rows = 0
    if family in {"BLiMP", "Supplement"}:
        sub = "blimp_filtered" if family == "BLiMP" else "supplement_filtered"
        d = EVAL_ROOT / sub
        for p in sorted(d.glob("*.jsonl")):
            files.append(rel(p))
            for r in iter_jsonl(p):
                rows += 1
                for k in ["sentence_good", "sentence_bad"]:
                    s = clean(r.get(k))
                    if s:
                        texts.append(s)
        return texts, {"family": family, "dir": rel(d), "files": files, "rows": rows, "texts": len(texts), "strict_rule": "sentence_good/sentence_bad only"}
    if family == "EWoK":
        d = EVAL_ROOT / "ewok_filtered"
        for p in sorted(d.glob("*.jsonl")):
            files.append(rel(p))
            for r in iter_jsonl(p):
                rows += 1
                for k in ["Context1", "Context2", "Target1", "Target2"]:
                    s = clean(r.get(k))
                    if s:
                        texts.append(s)
        return texts, {"family": family, "dir": rel(d), "files": files, "rows": rows, "texts": len(texts), "strict_rule": "Context1/Context2/Target1/Target2 only; metadata Concept/Type/Diff excluded"}
    if family == "COMPS":
        d = EVAL_ROOT / "comps"
        for p in sorted(d.glob("*.jsonl")):
            files.append(rel(p))
            for r in iter_jsonl(p):
                rows += 1
                prop = clean(r.get("property_phrase") or r.get("property"))
                for pref_key in ["prefix_acceptable", "prefix_unacceptable"]:
                    pref = clean(r.get(pref_key))
                    if pref and prop:
                        sent = clean(pref + " " + prop)
                    else:
                        sent = pref or prop
                    if sent:
                        texts.append(sent)
        return texts, {"family": family, "dir": rel(d), "files": files, "rows": rows, "texts": len(texts), "strict_rule": "prefix_acceptable/property_phrase and prefix_unacceptable/property_phrase composed as scorer-like candidates"}
    if family == "Entity":
        d = EVAL_ROOT / "entity_tracking"
        for p in sorted(d.glob("*.jsonl")):
            files.append(rel(p))
            for r in iter_jsonl(p):
                rows += 1
                prefix = clean(r.get("input_prefix"))
                if prefix:
                    texts.append(prefix)
                opts = r.get("options") or []
                if isinstance(opts, list):
                    for opt in opts:
                        opt_s = clean(opt)
                        if opt_s:
                            # Include both option alone and prefix+option because the evaluator scores full candidates.
                            texts.append(opt_s)
                            if prefix:
                                texts.append(clean(prefix + " " + opt_s))
        return texts, {"family": family, "dir": rel(d), "files": files, "rows": rows, "texts": len(texts), "strict_rule": "input_prefix plus options and prefix+option candidate strings"}
    if family == "Reading":
        p = EVAL_ROOT / "reading/reading_data.csv"
        seen: set[str] = set()
        with p.open(encoding="utf-8", errors="replace", newline="") as f:
            for r in csv.DictReader(f):
                rows += 1
                s = clean(r.get("sentence"))
                if s and s not in seen:
                    seen.add(s)
                    texts.append(s)
        files.append(rel(p))
        return texts, {"family": family, "files": files, "rows": rows, "texts": len(texts), "strict_rule": "unique sentence column"}
    raise ValueError(family)


def js(c1: collections.Counter[str], c2: collections.Counter[str]) -> float:
    n1, n2 = float(sum(c1.values())), float(sum(c2.values()))
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    s = 0.0
    for k in set(c1) | set(c2):
        p = c1.get(k, 0) / n1
        q = c2.get(k, 0) / n2
        m = 0.5 * (p + q)
        if p > 0:
            s += 0.5 * p * math.log(p/m, 2)
        if q > 0:
            s += 0.5 * q * math.log(q/m, 2)
    return s


def read_scores() -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    with CONTRAST_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[(r["contrast"], r["quantity"], r["window"])] = float(r["mean"])
            except Exception:
                pass
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x-mx)**2 for x in xs)
    vy = sum((y-my)**2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys))/math.sqrt(vx*vy)


def ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0]*len(vals)
    i=0
    while i < len(order):
        j=i+1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        r=(i+1+j)/2
        for k in range(i,j):
            out[order[k]]=r
        i=j
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(ranks(xs), ranks(ys)) if len(xs) >= 2 and len(xs) == len(ys) else None


def fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    xs=[r['x'] for r in rows]; ys=[r['y'] for r in rows]
    denom=sum(x*x for x in xs)
    beta=sum(x*y for x,y in zip(xs,ys))/denom if denom else float('nan')
    preds=[beta*x for x in xs]
    sse=sum((y-p)**2 for y,p in zip(ys,preds))
    syy0=sum(y*y for y in ys)
    syyc=sum((y-statistics.mean(ys))**2 for y in ys) if len(ys)>1 else float('nan')
    return {'beta':beta,'n':len(rows),'pearson':pearson(xs,ys),'spearman':spearman(xs,ys),'rmse':math.sqrt(sse/len(rows)) if rows else None,'r2_zero':1-sse/syy0 if syy0 else None,'r2_centered':1-sse/syyc if syyc and syyc>0 else None,'x_min':min(xs),'x_max':max(xs),'x_mean':statistics.mean(xs),'y_mean':statistics.mean(ys)}


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main() -> None:
    t0=time.time(); OUT.mkdir(parents=True, exist_ok=True)
    mod=load_pred_module()
    texts={}
    meta={}
    for name,path in [('admitted_view',mod.VIEW10),('admitted_repeat',mod.REPEAT10),('admitted_breadth',mod.BREADTH10)]:
        texts[name], meta[name]=mod.first_n_texts(path, PAIR_ROWS)
    texts['removed_proportional'], meta['removed_proportional']=mod.all_jsonl_texts(mod.HELDOUT_PROP)
    child_idx, adult_idx, selection_meta=mod.load_selection_indices()
    selected, selected_meta=mod.selected_base_texts(child_idx, adult_idx)
    texts['removed_childspeech']=selected['removed_childspeech']; meta['removed_childspeech']=selected_meta['removed_childspeech']
    texts['removed_adultprose']=selected['removed_adultprose']; meta['removed_adultprose']=selected_meta['removed_adultprose']
    eval_meta={}
    for fam in FAMILIES:
        texts[f'eval_{fam}'], eval_meta[fam]=strict_eval_family_texts(fam)
    counters_word={name: mod.word_counter_from_texts(ts) for name,ts in texts.items()}
    counters_profile={name: mod.profile_counter_from_texts(ts) for name,ts in texts.items()}
    scores=read_scores()
    all_rows=[]; pred_rows=[]; agg_rows=[]; fit_rows=[]; distance_rows=[]
    for metric,counters in [('word_js',counters_word),('profile_js',counters_profile)]:
        distances={fam:{} for fam in FAMILIES}
        for fam in FAMILIES:
            ev=counters[f'eval_{fam}']
            for block in ['admitted_view','admitted_breadth','admitted_repeat','removed_proportional','removed_childspeech','removed_adultprose']:
                d=js(ev,counters[block]); distances[fam][block]=d
                distance_rows.append({'metric':metric,'family':fam,'block':block,'js_bits':d})
        cal=[]
        for arm in ['V','B','R']:
            for fam in EX_ENTITY:
                y=scores[(ARM_CONTRAST[arm],fam,'common10_80')]
                x=distances[fam]['removed_proportional']-distances[fam][ARM_BLOCK[arm]]
                row={'metric':metric,'arm':arm,'family':fam,'x':x,'y':y}
                cal.append(row); all_rows.append(row)
        fs=fit(cal); fs.update({'metric':metric,'fit_spec':'strict_eval_common10_80_all3_exEntity'})
        fit_rows.append(fs)
        fam_pred={}
        for fam in FAMILIES:
            xreg=distances[fam]['removed_childspeech']-distances[fam]['removed_adultprose']
            p=fs['beta']*xreg; fam_pred[fam]=p
            pred_rows.append({'metric':metric,'family':fam,'distance_child_minus_adult':xreg,'pred_points':p,'beta':fs['beta']})
        for qty,fams in [('exEntity5',EX_ENTITY),('cheap6',FAMILIES)]:
            val=statistics.mean(fam_pred[f] for f in fams)
            agg_rows.append({'metric':metric,'quantity':qty,'pred_points':val,'min_family_pred':min(fam_pred[f] for f in fams),'max_family_pred':max(fam_pred[f] for f in fams),'positive_family_count':sum(1 for f in fams if fam_pred[f]>0),'negative_family_count':sum(1 for f in fams if fam_pred[f]<0),'over_spread_ref':val/SPREAD[qty]})
    write_csv(OUT/'distance_rows.csv', distance_rows)
    write_csv(OUT/'calibration_rows.csv', all_rows)
    write_csv(OUT/'fit_summary_rows.csv', fit_rows)
    write_csv(OUT/'prediction_rows.csv', pred_rows)
    write_csv(OUT/'aggregate_prediction_rows.csv', agg_rows)
    summary={'status':'STRICT_EVAL_TEXT_PREDICTION_COMPLETE','created_utc':now(),'purpose':'Task-aware strict eval text extraction variant before register scores','selection_meta':selection_meta,'eval_meta':eval_meta,'fit_summary_rows':fit_rows,'aggregate_prediction_rows':agg_rows,'files':{'summary_json':rel(OUT/'strict_eval_text_prediction_summary.json'),'summary_md':rel(OUT/'strict_eval_text_prediction_summary.md'),'distance_rows':rel(OUT/'distance_rows.csv'),'prediction_rows':rel(OUT/'prediction_rows.csv'),'aggregate_prediction_rows':rel(OUT/'aggregate_prediction_rows.csv')},'no_model_loading_training_official_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard':True,'elapsed_sec':round(time.time()-t0,3)}
    (OUT/'strict_eval_text_prediction_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    def fmt(x):
        try: return f'{float(x):.4f}'
        except Exception: return str(x)
    byfit={r['metric']:r for r in fit_rows}; byagg={(r['metric'],r['quantity']):r for r in agg_rows}
    lines=['# research strict evaluation-text prediction','', 'CPU-only strict extraction variant before MAX-register scores. EWoK uses only Context1/2 and Target1/2; COMPS uses prefix+property candidate strings; Entity uses input_prefix/options/prefix+option; BLiMP/Supplement use sentence_good/bad; Reading uses unique sentence.', '', '| metric | beta | Pearson | Spearman | RMSE | exEntity pred | cheap6 pred |', '|---|---:|---:|---:|---:|---:|---:|']
    for metric in ['word_js','profile_js']:
        f=byfit[metric]
        lines.append(f"| {metric} | {fmt(f['beta'])} | {fmt(f['pearson'])} | {fmt(f['spearman'])} | {fmt(f['rmse'])} | {fmt(byagg[(metric,'exEntity5')]['pred_points'])} | {fmt(byagg[(metric,'cheap6')]['pred_points'])} |")
    lines += ['', 'This variant checks whether the original broad key-name heuristics were driving the prediction. The profile model remains scientifically useful only insofar as the positive sign/magnitude survive this stricter extraction.', '', f"JSON: `{rel(OUT/'strict_eval_text_prediction_summary.json')}`"]
    (OUT/'strict_eval_text_prediction_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'fit_summary_rows':fit_rows,'aggregate_prediction_rows':agg_rows,'summary_md':summary['files']['summary_md'],'elapsed_sec':summary['elapsed_sec']},indent=2,ensure_ascii=False), flush=True)

if __name__=='__main__':
    main()
