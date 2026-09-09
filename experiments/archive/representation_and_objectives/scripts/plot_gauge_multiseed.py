#!/usr/bin/env python3
"""Plot research multi-seed causal gauge and frozen-affine readouts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IN_JSON = Path("experiments/archive/representation_and_objectives/data/multiseed_gauge_affine_full/multiseed_gauge_and_affine_summary.json")
OUT_DIR = Path("experiments/archive/representation_and_objectives/figures")


def mean(vals):
    vals=[v for v in vals if isinstance(v,(int,float))]
    return float(sum(vals)/len(vals)) if vals else np.nan

def sd(vals):
    vals=[float(v) for v in vals if isinstance(v,(int,float))]
    if len(vals)<=1: return 0.0 if vals else np.nan
    m=sum(vals)/len(vals)
    return float((sum((v-m)**2 for v in vals)/(len(vals)-1))**0.5)

def get_stat(payload, cond, key):
    v=payload["by_condition"][cond][key]
    return v["mean"], v["sd"], v["values"]

def affine_rows(payload):
    rows=[]
    for item in payload.get("affine_summaries", []):
        for run in item["payload"].get("runs", []):
            for f in run.get("fits", []):
                if f.get("kind") == "least_squares" and f.get("arm_sign") == 1:
                    ev=f["eval"]
                    rows.append({
                        "name": Path(run["run_dir"]).name,
                        "train_cmp": run.get("train_cmp_acc"),
                        "hh": run.get("heldheld_closure_acc"),
                        "direct": ev["direct_psc_all"].get("canonical_acc"),
                        "graph": ev["graph_psc_all"].get("canonical_acc"),
                        "graph_same": ev["graph_psc_same"].get("canonical_acc"),
                    })
    return rows

def main():
    payload=json.loads(IN_JSON.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conds=["tied","shared_trunk","untied"]
    colors={"tied":"#4C78A8","shared_trunk":"#54A24B","untied":"#E45756"}
    fig=plt.figure(figsize=(13.5,8.0), dpi=180)
    gs=fig.add_gridspec(2,2, height_ratios=[1.2,1.0], width_ratios=[1.25,1.0], hspace=0.38, wspace=0.28)

    ax=fig.add_subplot(gs[0,0])
    x=np.arange(len(conds)); width=0.32
    plus=[]; minus=[]; eplus=[]; eminus=[]
    for c in conds:
        m,s,_=get_stat(payload,c,"graph_same_plus"); plus.append(m); eplus.append(s)
        m,s,_=get_stat(payload,c,"graph_same_minus"); minus.append(m); eminus.append(s)
    ax.bar(x-width/2, plus, width, yerr=eplus, capsize=3, label="bridge_sign +1", color="#72B7B2")
    ax.bar(x+width/2, minus, width, yerr=eminus, capsize=3, label="bridge_sign -1", color="#F58518")
    for i,c in enumerate(conds):
        vals_p=get_stat(payload,c,"graph_same_plus")[2]
        vals_m=get_stat(payload,c,"graph_same_minus")[2]
        ax.scatter(np.full(len(vals_p), x[i]-width/2), vals_p, color="black", s=14, zorder=3)
        ax.scatter(np.full(len(vals_m), x[i]+width/2), vals_m, color="black", s=14, zorder=3)
    ax.set_ylim(-0.05,1.05); ax.set_ylabel("h1/h3 graph-state canonical accuracy")
    ax.set_xticks(x); ax.set_xticklabels(["tied", "shared\ntrunk", "untied"])
    ax.set_title("Bridge-sign flips transfer through shared representation")
    ax.legend(frameon=False, fontsize=9)

    ax=fig.add_subplot(gs[0,1])
    vals=[]; errs=[]
    for c in conds:
        m,s,_=get_stat(payload,c,"graph_changed_opposite_frac"); vals.append(m); errs.append(s)
    ax.bar(x, vals, yerr=errs, capsize=3, color=[colors[c] for c in conds])
    for i,c in enumerate(conds):
        vs=get_stat(payload,c,"graph_changed_opposite_frac")[2]
        ax.scatter(np.full(len(vs),x[i]),vs,color="black",s=14,zorder=3)
    ax.set_ylim(-0.05,1.05); ax.set_xticks(x); ax.set_xticklabels(["tied", "shared\ntrunk", "untied"])
    ax.set_ylabel("fraction with d_e sign reversed")
    ax.set_title("Row-paired h1/h3 event-coordinate reversal")

    ax=fig.add_subplot(gs[1,0])
    metrics=["mixed_acc_plus","mixed_acc_minus","hh_closure_plus","hh_closure_minus"]
    labels=["mixed +", "mixed -", "HH +", "HH -"]
    offsets=np.linspace(-0.3,0.3,len(metrics))
    for j,(metric,label) in enumerate(zip(metrics,labels)):
        ys=[]; es=[]
        for c in conds:
            m,s,_=get_stat(payload,c,metric); ys.append(m); es.append(s)
        ax.bar(x+offsets[j], ys, 0.15, yerr=es, capsize=2, label=label)
    ax.set_ylim(-0.05,1.05); ax.set_xticks(x); ax.set_xticklabels(["tied", "shared\ntrunk", "untied"])
    ax.set_ylabel("comparison accuracy")
    ax.set_title("Held-held closure preserved; mixed held/seen sign follows held gauge")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5,1.18))

    ax=fig.add_subplot(gs[1,1])
    ar=affine_rows(payload)
    names=["research\nHH-only", "research\nno-anchor"]
    graph=[ar[0]["graph"] if len(ar)>0 else np.nan, ar[1]["graph"] if len(ar)>1 else np.nan]
    direct=[ar[0]["direct"] if len(ar)>0 else np.nan, ar[1]["direct"] if len(ar)>1 else np.nan]
    hh=[ar[0]["hh"] if len(ar)>0 else np.nan, ar[1]["hh"] if len(ar)>1 else np.nan]
    xx=np.arange(2); w=0.25
    ax.bar(xx-w, direct, w, label="direct h0/h2", color="#B279A2")
    ax.bar(xx, graph, w, label="graph h1/h3", color="#59A14F")
    ax.bar(xx+w, hh, w, label="HH closure", color="#9D755D")
    ax.set_ylim(-0.05,1.05); ax.set_xticks(xx); ax.set_xticklabels(names)
    ax.set_ylabel("accuracy")
    ax.set_title("Frozen one-dimensional affine calibration")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("research evidence: finite transfer requires anchors + graph + shared representation; scalar coordinate is sufficient when calibrated", fontsize=12)
    fig.text(0.01,0.01,"Data: saved research/291 primary gauge runs (seeds 29000-29002) and research frozen-affine saved-output analysis. Tied seed29002 bs− underfit train comparison (0.875), shown in means rather than removed.", fontsize=7)
    out=OUT_DIR/"gauge_transport_multiseed_affine.png"
    fig.savefig(out, bbox_inches="tight")
    print(json.dumps({"status":"GAUGE_FIGURE_COMPLETE","figure":str(out),"bytes":out.stat().st_size}, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
