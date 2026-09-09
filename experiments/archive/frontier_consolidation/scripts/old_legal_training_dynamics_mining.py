#!/usr/bin/env python3
"""research: mine old-tokenizer vs legal-tokenizer training dynamics.

No new training/evaluation. Compare existing training logs and dynamics traces for the
same compact-view reinvest stream, recipe, architecture, and seed, differing only in
tokenizer. The aim is to determine whether the legal deficit looks like optimization
underfit, unstable dynamics, or lower-loss over-consolidation of the repeated stream.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import statistics
import time
from typing import Any

STUDY = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/old_legal_training_dynamics_mining')
RUNS = {
    "old_nonlegal_tok": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022'),
    "legal_step35_tok": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2'),
    "bytealpha_tok": _public_path('experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022'),
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def smooth(xs: list[float], w: int) -> list[float]:
    out=[]
    for i in range(len(xs)):
        lo=max(0,i-w+1)
        out.append(sum(xs[lo:i+1])/(i-lo+1))
    return out


def window_stats(rows: list[dict[str, Any]], start_m: float, end_m: float) -> dict[str, Any]:
    sub=[r for r in rows if start_m*1e6 <= r["cumulative_word_exposure"] <= end_m*1e6]
    if not sub:
        return {"n":0}
    losses=[float(r["loss"]) for r in sub]
    lrs=[float(r["lr"]) for r in sub]
    masks=[float(r.get("effective_mask_rate",0.0)) for r in sub]
    return {
        "n": len(sub),
        "loss_mean": statistics.mean(losses),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_std": statistics.pstdev(losses) if len(losses)>1 else 0.0,
        "lr_mean": statistics.mean(lrs),
        "mask_mean": statistics.mean(masks),
    }


def load_run(path: pathlib.Path) -> dict[str, Any]:
    log=read_jsonl(path/"training_log.jsonl")
    dyn=read_jsonl(path/"dynamics_traces.jsonl") if (path/"dynamics_traces.jsonl").exists() else []
    metrics=json.loads((path/"scientific_metrics.json").read_text(encoding="utf-8"))
    return {"path":str(path),"log":log,"dyn":dyn,"metrics":metrics}


def extract_dyn(dyn: list[dict[str,Any]]) -> list[dict[str,Any]]:
    rows=[]
    for d in dyn:
        if not d.get("loss_by_freq_band"):
            continue
        rec={"checkpoint":d.get("checkpoint"),"step":d.get("step"),"prediction_entropy_mean":d.get("prediction_entropy_mean"),"effective_mask_rate_mean":d.get("effective_mask_rate_mean")}
        for band in ["high","mid","low"]:
            rec[f"{band}_loss"] = d.get("loss_by_freq_band",{}).get(band,{}).get("mean")
            rec[f"{band}_acc"] = d.get("accuracy_by_freq_band",{}).get(band,{}).get("mean")
        rows.append(rec)
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs={name:load_run(path) for name,path in RUNS.items() if path.exists()}
    # Per-window stats in 10M blocks and early/mid/late blocks.
    windows=[]
    for a,b in [(0,1),(1,5),(5,10),(10,20),(20,40),(40,60),(60,80),(80,100),(90,100)]:
        rec={"window_m":f"{a}-{b}"}
        for name,r in runs.items():
            ws=window_stats(r["log"],a,b)
            for k,v in ws.items():
                rec[f"{name}_{k}"]=v
        # deltas legal-old and byte-old where present
        if "old_nonlegal_tok" in runs and "legal_step35_tok" in runs:
            for k in ["loss_mean","loss_first","loss_last","loss_min","loss_std"]:
                rec[f"legal_minus_old_{k}"]=rec.get(f"legal_step35_tok_{k}")-rec.get(f"old_nonlegal_tok_{k}") if rec.get(f"legal_step35_tok_{k}") is not None and rec.get(f"old_nonlegal_tok_{k}") is not None else None
        if "old_nonlegal_tok" in runs and "bytealpha_tok" in runs:
            for k in ["loss_mean","loss_first","loss_last","loss_min","loss_std"]:
                rec[f"bytealpha_minus_old_{k}"]=rec.get(f"bytealpha_tok_{k}")-rec.get(f"old_nonlegal_tok_{k}") if rec.get(f"bytealpha_tok_{k}") is not None and rec.get(f"old_nonlegal_tok_{k}") is not None else None
        windows.append(rec)

    # Matched per-step deltas.
    matched=[]
    oldlog={int(r["step"]):r for r in runs["old_nonlegal_tok"]["log"]}
    for step,oldr in oldlog.items():
        rec={"step":step,"exposure":oldr["cumulative_word_exposure"],"old_loss":oldr["loss"],"old_lr":oldr["lr"]}
        for name in ["legal_step35_tok","bytealpha_tok"]:
            if name in runs and step<=len(runs[name]["log"]):
                rr=runs[name]["log"][research]
                rec[f"{name}_loss"]=rr["loss"]
                rec[f"{name}_minus_old_loss"]=rr["loss"]-oldr["loss"]
                rec[f"{name}_masked_tokens_minus_old"]=rr.get("masked_tokens",0)-oldr.get("masked_tokens",0)
        matched.append(rec)

    # Dynamics trace deltas at every 8M-ish checkpoint with frequency bands.
    dyns={name:extract_dyn(r["dyn"]) for name,r in runs.items()}
    dyn_by_name={name:{d["checkpoint"]:d for d in rows} for name,rows in dyns.items()}
    dyn_delta=[]
    for ck,dold in dyn_by_name.get("old_nonlegal_tok",{}).items():
        rec={"checkpoint":ck,"step":dold.get("step")}
        for band in ["high","mid","low"]:
            rec[f"old_{band}_loss"]=dold.get(f"{band}_loss")
            rec[f"old_{band}_acc"]=dold.get(f"{band}_acc")
        rec["old_entropy"]=dold.get("prediction_entropy_mean")
        for name in ["legal_step35_tok","bytealpha_tok"]:
            d=dyn_by_name.get(name,{}).get(ck)
            if not d: continue
            rec[f"{name}_entropy_minus_old"]=d.get("prediction_entropy_mean")-dold.get("prediction_entropy_mean")
            for band in ["high","mid","low"]:
                if d.get(f"{band}_loss") is not None and dold.get(f"{band}_loss") is not None:
                    rec[f"{name}_minus_old_{band}_loss"]=d.get(f"{band}_loss")-dold.get(f"{band}_loss")
                if d.get(f"{band}_acc") is not None and dold.get(f"{band}_acc") is not None:
                    rec[f"{name}_minus_old_{band}_acc"]=d.get(f"{band}_acc")-dold.get(f"{band}_acc")
        dyn_delta.append(rec)

    # Summaries of sign persistence.
    legal_deltas=[r["legal_step35_tok_minus_old_loss"] for r in matched if "legal_step35_tok_minus_old_loss" in r]
    byte_deltas=[r["bytealpha_tok_minus_old_loss"] for r in matched if "bytealpha_tok_minus_old_loss" in r]
    def sign_summary(vals):
        return {"n":len(vals),"mean":statistics.mean(vals),"median":statistics.median(vals),"frac_negative":sum(v<0 for v in vals)/len(vals),"min":min(vals),"max":max(vals)}
    sign={"legal_minus_old_loss":sign_summary(legal_deltas),"bytealpha_minus_old_loss":sign_summary(byte_deltas) if byte_deltas else None}
    # Exposure where smoothed legal loss first becomes lower than old for 20 consecutive steps.
    for key,name in [("legal","legal_step35_tok"),("bytealpha","bytealpha_tok")]:
        vals=[r.get(f"{name}_minus_old_loss") for r in matched]
        sm=smooth([v if v is not None else 0.0 for v in vals],25)
        first=None
        for i in range(len(sm)-20):
            if all(x<0 for x in sm[i:i+20]):
                first=matched[i]["exposure"]
                break
        sign[f"{key}_smoothed_loss_lower_from_exposure"] = first

    result={
        "status":"OLD_LEGAL_TRAINING_DYNAMICS_MINING",
        "created_utc":now_utc(),
        "purpose":"No-training comparison of existing old/nonlegal, legal research, and bytealphabet training dynamics; tests underfit vs over-consolidation signatures.",
        "runs":{name:{"path":r["path"],"tokenizer_label":r["metrics"].get("tokenizer_label"),"loss_first":r["metrics"].get("loss_first"),"loss_last":r["metrics"].get("loss_last"),"word_exposure":r["metrics"].get("word_exposure"),"steps":r["metrics"].get("actual_training_steps")} for name,r in runs.items()},
        "sign_summary":sign,
        "window_stats":windows,
        "frequency_band_dynamics_delta":dyn_delta,
        "matched_step_loss_delta_first_200":matched[:200],
        "matched_step_loss_delta_every_100th":[r for r in matched if r["step"]%100==0],
    }
    out_json=_public_path('experiments/archive/frontier_consolidation/data/old_legal_training_dynamics_mining/old_legal_training_dynamics_mining.json')
    out_md=_public_path('research/documents/frontier_consolidation/data/old_legal_training_dynamics_mining/old_legal_training_dynamics_mining.md')
    out_csv=_public_path('experiments/archive/frontier_consolidation/data/old_legal_training_dynamics_mining/matched_step_loss_delta.csv')
    out_json.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    with out_csv.open("w",newline="",encoding="utf-8") as f:
        fields=list(matched[0].keys())
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(matched)

    md=[]
    md.append("# research old-vs-legal training dynamics mining")
    md.append("")
    md.append("No new training/evaluation. Same compact-view reinvest stream, architecture, seed, and recipe; tokenizer differs. This tests whether the legal endpoint is under-optimized or instead has lower MLM loss while worse downstream competence.")
    md.append("")
    md.append("## Endpoint run facts")
    md.append("| run | tokenizer | first loss | final loss | steps |")
    md.append("|---|---|---:|---:|---:|")
    for name,info in result["runs"].items():
        md.append(f"| {name} | {info['tokenizer_label']} | {info['loss_first']:.4f} | {info['loss_last']:.4f} | {info['steps']} |")
    md.append("")
    md.append("## Per-step loss delta sign")
    md.append("| contrast | mean delta | median | frac candidate lower loss than old | min | max | smoothed lower from exposure |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for key,label,fromkey in [("legal_minus_old_loss","legal_step35 - old","legal_smoothed_loss_lower_from_exposure"),("bytealpha_minus_old_loss","bytealpha - old","bytealpha_smoothed_loss_lower_from_exposure")]:
        s=sign.get(key)
        if s:
            md.append(f"| {label} | {s['mean']:.5f} | {s['median']:.5f} | {s['frac_negative']:.3f} | {s['min']:.5f} | {s['max']:.5f} | {sign.get(fromkey)} |")
    md.append("")
    md.append("## Window mean loss deltas")
    md.append("| window M words | old loss | legal loss | legal-old | bytealpha loss | bytealpha-old |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in windows:
        md.append(f"| {r['window_m']} | {r.get('old_nonlegal_tok_loss_mean',float('nan')):.4f} | {r.get('legal_step35_tok_loss_mean',float('nan')):.4f} | {r.get('legal_minus_old_loss_mean',float('nan')):.4f} | {r.get('bytealpha_tok_loss_mean',float('nan')):.4f} | {r.get('bytealpha_minus_old_loss_mean',float('nan')):.4f} |")
    md.append("")
    md.append("## Frequency-band dynamics where available")
    md.append("| checkpoint | legal-old high loss | mid loss | low loss | entropy | bytealpha-old high loss | mid loss | low loss | entropy |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in dyn_delta:
        md.append(f"| {r['checkpoint']} | {r.get('legal_step35_tok_minus_old_high_loss',float('nan')):.4f} | {r.get('legal_step35_tok_minus_old_mid_loss',float('nan')):.4f} | {r.get('legal_step35_tok_minus_old_low_loss',float('nan')):.4f} | {r.get('legal_step35_tok_entropy_minus_old',float('nan')):.4f} | {r.get('bytealpha_tok_minus_old_high_loss',float('nan')):.4f} | {r.get('bytealpha_tok_minus_old_mid_loss',float('nan')):.4f} | {r.get('bytealpha_tok_minus_old_low_loss',float('nan')):.4f} | {r.get('bytealpha_tok_entropy_minus_old',float('nan')):.4f} |")
    md.append("")
    md.append("## Scientific reading")
    if sign["legal_minus_old_loss"]["frac_negative"]>0.6:
        md.append("- Legal research has lower MLM training loss than the old nonlegal tokenizer for most training steps, despite much worse official-compatible downstream score. This is not an underfit/too-low-lr signature; it is more consistent with a narrower/easier tokenizer coordinate that over-consolidates the repeated stream or loses useful regularization.")
    else:
        md.append("- Legal research does not have persistently lower MLM training loss; under-optimization remains plausible.")
    md.append("- Any bounded optimization experiment should therefore test a specific regularization/dynamics hypothesis, not a generic lr sweep. A 40M task-score ranking remains unsafe unless a dynamics quantity, not downstream score, is the target.")
    md.append("")
    md.append(f"JSON: `{out_json}`")
    md.append(f"CSV: `{out_csv}`")
    out_md.write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"out_json":str(out_json),"out_md":str(out_md),"legal_loss_summary":sign["legal_minus_old_loss"],"bytealpha_loss_summary":sign.get("bytealpha_minus_old_loss")},indent=2),flush=True)

if __name__=="__main__":
    main()
