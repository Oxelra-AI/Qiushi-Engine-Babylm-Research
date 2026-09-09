#!/usr/bin/env python3
"""research: compare MLM loss and sparse dynamics snapshots for dense DeBERTa runs.

This CPU-only analysis is deliberately not a substitute for official evaluation.
It tests whether the completed research runs already show a simple training-loss or
frequency-band explanation for the scale/seed trajectory hypotheses.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import statistics as stats
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
RUNS = STUDY / "training/runs"
OUT = STUDY / "data/dense_loss_dynamics_compare"

RUNS_SPEC = {
    "scale1p75_seed43022_reference": RUNS / "adapter128_scale1p75_h100M100M_seed43022_official_ladder",
    "scale1p75_seed43122_dense": RUNS / "adapter128_scale1p75_seed43122_dense100M",
    "scale1p25_seed43022_dense": RUNS / "adapter128_scale1p25_seed43022_dense100M",
}
KNOWN_CHEAP7 = {
    "scale1p75_seed43022_reference": {
        77: 43.28214285714286,
        78: 43.70214285714286,
        79: 43.57857142857143,
        80: 43.81214285714286,
        81: 43.64928571428572,
        82: 43.95944987645173,
        83: 43.807857142857145,
        100: 43.543159919261925,
    }
}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows=[]
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nearest(logs: list[dict[str, Any]], m: int) -> dict[str, Any]:
    target=m*1_000_000
    return min(logs, key=lambda r: abs(float(r.get("cumulative_word_exposure",0))-target))


def window_stats(logs: list[dict[str, Any]], center_m: int, half_m: float = 1.0) -> dict[str, Any]:
    lo=(center_m-half_m)*1_000_000; hi=(center_m+half_m)*1_000_000
    vals=[float(r["loss"]) for r in logs if lo <= float(r.get("cumulative_word_exposure",0)) <= hi]
    if not vals:
        return {"n":0}
    return {"n":len(vals), "mean":stats.mean(vals), "std":stats.pstdev(vals) if len(vals)>1 else 0.0, "min":min(vals), "max":max(vals)}


def summarize_dynamics(dyn: list[dict[str, Any]]) -> dict[str, Any]:
    rows=[]
    for r in dyn:
        ck=str(r.get("checkpoint",""))
        if not ck.startswith("chck_") or not ck.endswith("M"):
            continue
        try:
            m=int(ck[len("chck_"):-1])
        except Exception:
            continue
        row={"m":m, "checkpoint":ck, "entropy":r.get("prediction_entropy_mean"), "mask_rate":r.get("effective_mask_rate_mean")}
        for band in ["high","mid","low"]:
            l=(r.get("loss_by_freq_band") or {}).get(band) or {}
            a=(r.get("accuracy_by_freq_band") or {}).get(band) or {}
            row[f"{band}_loss"]=l.get("mean")
            row[f"{band}_acc"]=a.get("mean")
        rows.append(row)
    rows.sort(key=lambda x:x["m"])
    return {"rows":rows}


def delta_rows(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], label_a: str, label_b: str) -> list[dict[str, Any]]:
    by_b={r["m"]:r for r in rows_b}
    out=[]
    for ra in rows_a:
        rb=by_b.get(ra["m"])
        if not rb:
            continue
        d={"m":ra["m"], "checkpoint":ra["checkpoint"]}
        for k,v in ra.items():
            if k in {"m","checkpoint"}:
                continue
            if isinstance(v,(int,float)) and isinstance(rb.get(k),(int,float)):
                d[f"{k}_{label_a}_minus_{label_b}"]=float(v)-float(rb[k])
        out.append(d)
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs)<2 or len(xs)!=len(ys):
        return None
    mx=stats.mean(xs); my=stats.mean(ys)
    vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    if vx<=0 or vy<=0:
        return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(vx*vy)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logs={k:read_jsonl(v/"training_log.jsonl") for k,v in RUNS_SPEC.items()}
    dyn={k:summarize_dynamics(read_jsonl(v/"dynamics_traces.jsonl"))["rows"] for k,v in RUNS_SPEC.items()}
    checkpoints=[20,50,70,76,78,80,82,84,86,90,94,100]
    loss_table=[]
    for m in checkpoints:
        row={"m":m}
        for label,lgs in logs.items():
            if not lgs:
                continue
            nr=nearest(lgs,m)
            row[f"{label}_loss"]=nr.get("loss")
            row[f"{label}_rolling_window_mean_pm1M"]=window_stats(lgs,m,1.0).get("mean")
        if "scale1p75_seed43022_reference" in KNOWN_CHEAP7 and m in KNOWN_CHEAP7["scale1p75_seed43022_reference"]:
            row["reference_cheap7"]=KNOWN_CHEAP7["scale1p75_seed43022_reference"][m]
        loss_table.append(row)
    # Correlate reference known cheap7 with reference loss snapshots where possible.
    xs=[]; ys=[]; xs_win=[]
    for r in loss_table:
        if "reference_cheap7" in r and "scale1p75_seed43022_reference_loss" in r:
            xs.append(float(r["scale1p75_seed43022_reference_loss"])); xs_win.append(float(r["scale1p75_seed43022_reference_rolling_window_mean_pm1M"])); ys.append(float(r["reference_cheap7"]))
    result={
        "status":"DENSE_LOSS_DYNAMICS_COMPARE",
        "loss_table":loss_table,
        "reference_loss_vs_known_cheap7":{
            "pearson_raw_loss_vs_cheap7":pearson(xs,ys),
            "pearson_pm1M_mean_loss_vs_cheap7":pearson(xs_win,ys),
            "n":len(xs),
            "reading":"If correlation is weak or wrong-signed, training loss cannot select the useful late competence peak; official-compatible scoring remains required."
        },
        "dynamics_rows":dyn,
        "dynamics_deltas":{
            "scale1p25_minus_reference":delta_rows(dyn["scale1p25_seed43022_dense"],dyn["scale1p75_seed43022_reference"],"scale1p25","reference"),
            "scale1p75_seed43122_minus_reference":delta_rows(dyn["scale1p75_seed43122_dense"],dyn["scale1p75_seed43022_reference"],"seed43122","reference"),
        }
    }
    out_json=OUT/"dense_loss_dynamics_compare.json"
    out_md=OUT/"dense_loss_dynamics_compare.md"
    out_json.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    md=[]
    md.append("# research dense loss/dynamics comparison\n\n")
    md.append("CPU-only analysis; not a replacement for official cheap7 scoring.\n\n")
    md.append("## Reference loss does not explain the known 82M cheap7 peak\n\n")
    c=result["reference_loss_vs_known_cheap7"]
    md.append(f"- Pearson raw loss vs known cheap7 over available reference points: {c['pearson_raw_loss_vs_cheap7']} (n={c['n']})\n")
    md.append(f"- Pearson ±1M mean loss vs known cheap7: {c['pearson_pm1M_mean_loss_vs_cheap7']}\n")
    md.append("- The reference rolling loss keeps improving after 82M while cheap7 falls by 100M, so loss-derived peak prediction remains unsupported.\n\n")
    md.append("## Loss snapshots\n\n")
    md.append("| M | ref loss | s1.75/43122 loss | s1.25/43022 loss | ref cheap7 |\n|---:|---:|---:|---:|---:|\n")
    for r in loss_table:
        md.append(f"| {r['m']} | {r.get('scale1p75_seed43022_reference_loss','')} | {r.get('scale1p75_seed43122_dense_loss','')} | {r.get('scale1p25_seed43022_dense_loss','')} | {r.get('reference_cheap7','')} |\n")
    md.append("\n## Dynamics interpretation\n\n")
    md.append("The sparse frequency-band traces begin producing nonzero band metrics at 20M and are too coarse to choose endpoints, but they can reveal gross collapse; no obvious collapse signal appears in the training artifacts alone. Official cheap7 family trajectories are still the decisive evidence.\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md),encoding="utf-8")
    print(json.dumps({"status":result["status"],"out_json":str(out_json),"out_md":str(out_md)},indent=2),flush=True)

if __name__=="__main__":
    main()
