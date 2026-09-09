#!/usr/bin/env python3
"""Summarize central metrics for research temporal-change bridge runs."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

CENTRAL = [
    "heldChanged_heldStable_trainCtx_trainHyp",
    "heldChanged_heldStable_trainCtx_heldHyp",
    "heldChanged_heldStable_heldCtx_trainHyp",
    "heldStable_heldStable_trainCtx_trainHyp",
    "heldStable_heldStable_trainCtx_heldHyp",
]
QUERIES = ["focal_before", "focal_after", "secondary_before", "secondary_after"]


def get(ev, q, margin=False):
    key = "contrastive_by_query_margin" if margin else "contrastive_by_query"
    return ev.get(key, {}).get(q, {}).get("mean", float("nan"))


def per_seed_table(summary):
    rows=[]
    for r in summary.get("per_seed", []):
        for en in CENTRAL:
            ev = r["evals"].get(en, {}).get("contrastive", {})
            row={"arm":r["arm"],"seed":r["seed"],"train_acc":r["best_train_acc"],"eval_set":en,"con_acc":ev.get("contrastive_acc",float('nan'))}
            for q in QUERIES:
                row[q]=ev.get("by_query",{}).get(q,float('nan'))
                row[q+"_margin"]=ev.get("by_query_margin",{}).get(q,float('nan'))
            rows.append(row)
    return rows


def main():
    if len(sys.argv)<2:
        print("usage: analyze_temporal_bridge.py summary.json [out_dir]")
        raise SystemExit(2)
    p=Path(sys.argv[1])
    s=json.loads(p.read_text())
    out=Path(sys.argv[2]) if len(sys.argv)>2 else p.parent/"analysis"
    out.mkdir(parents=True,exist_ok=True)
    rows=per_seed_table(s)
    (out/"central_metrics.json").write_text(json.dumps(rows,indent=2)+"\n")
    lines=[f"# research temporal bridge central analysis\n\nSource: `{p}`\n\n"]
    cons=s.get('construction',{})
    if cons:
        lines.append("## Construction snapshot\n\n")
        lines.append(f"Base rows {cons['base']['n']} labels {cons['base']['by_label']} queries {cons['base']['by_query']}\n\n")
        for arm,d in cons.get('updates',{}).items():
            lines.append(f"- {arm}: rows {d['n']} train queries {d['by_query']} labels {d['by_label']} changed_focal {d['by_changed_focal']}\n")
        lines.append("\n")
    lines.append("## Central held changed-focal / stable-secondary readouts\n\n")
    for arm,item in sorted(s.get('aggregate',{}).items()):
        lines.append(f"### {arm} — train_acc {item['train_acc_mean']:.3f}\n\n")
        lines.append("| eval_set | con_acc | fb | fa | sb | sa | fa_m | sa_m |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for en in CENTRAL:
            if en not in item['evals']:
                continue
            ev=item['evals'][en]
            lines.append(f"| {en} | {ev['contrastive_acc']['mean']:.3f} | {get(ev,'focal_before'):.3f} | {get(ev,'focal_after'):.3f} | {get(ev,'secondary_before'):.3f} | {get(ev,'secondary_after'):.3f} | {get(ev,'focal_after',True):.2f} | {get(ev,'secondary_after',True):.2f} |\n")
        lines.append("\n")
        # compact interpretation numbers
        en="heldChanged_heldStable_trainCtx_trainHyp"
        if en in item['evals']:
            ev=item['evals'][en]
            lines.append(f"On `{en}`: focal_after={get(ev,'focal_after'):.3f} (margin {get(ev,'focal_after',True):.2f}), secondary_after={get(ev,'secondary_after'):.3f} (margin {get(ev,'secondary_after',True):.2f}).\n\n")
    # histories
    lines.append("## Train histories\n\n")
    for r in s.get('per_seed',[]):
        hist=r.get('history',[])
        if hist:
            vals=', '.join([f"e{h['epoch']}:{h['train_acc']:.3f}" for h in hist])
            lines.append(f"- {r['arm']} seed {r['seed']}: best {r['best_train_acc']:.3f}; {vals}\n")
    (out/"central_analysis.md").write_text(''.join(lines))
    print(json.dumps({"status":"analysis_done","md":str(out/"central_analysis.md"),"json":str(out/"central_metrics.json")}))

if __name__=='__main__': main()
