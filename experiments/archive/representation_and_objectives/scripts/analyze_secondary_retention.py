#!/usr/bin/env python3
"""Analyze research no-direct-secondary retention summaries."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

QUERIES = ["event_role", "focal_state", "untouched_state"]
KEY_EVALS = [
    "atp_trainTrain_trainHyp",
    "atp_trainTrain_heldHyp",
    "atp_stateOnly_train_trainHyp",
    "atp_eventOnly_train_trainHyp",
    "atp_trainEvent_nonState_trainHyp",
    "atp_nonEvent_trainState_trainHyp",
]

def get(d, path, default=float('nan')):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def arm_metric(agg, arm, ev, q, which='margin'):
    return get(agg, ['arms', arm, 'evals', ev, q, which, 'mean'])

def arm_acc(agg, arm, ev, q):
    return get(agg, ['arms', arm, 'evals', ev, q, 'acc', 'mean'])

def base_metric(agg, ev, q, which='margin'):
    return get(agg, ['base', ev, q, which, 'mean'])

def base_acc(agg, ev, q):
    return get(agg, ['base', ev, q, 'acc', 'mean'])

def fmt(x):
    if x is None or (isinstance(x,float) and math.isnan(x)): return 'nan'
    return f'{float(x):.3f}'

def load_summary(path: Path):
    d = json.loads(path.read_text('utf-8'))
    return d

def table_for_summary(name, d):
    agg = d.get('aggregate', {})
    arms = d.get('arms', [])
    lines=[]
    btr=get(agg,['base','train_acc','mean'])
    lines.append(f"## {name}\n\n")
    lines.append(f"Source: `{d.get('_path','')}`. Base train acc {fmt(btr)}; arms: {', '.join(arms)}.\n\n")
    lines.append("### Core familiar trainTrain surface\n\n")
    lines.append("| arm | update acc | base-after | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    ev='atp_trainTrain_trainHyp'
    bsec=base_metric(agg, ev, 'untouched_state')
    lines.append(f"| BASE before | - | - | {fmt(base_acc(agg,ev,'event_role'))}/{fmt(base_metric(agg,ev,'event_role'))} | {fmt(base_acc(agg,ev,'focal_state'))}/{fmt(base_metric(agg,ev,'focal_state'))} | {fmt(base_acc(agg,ev,'untouched_state'))}/{fmt(bsec)} | 0.000 |\n")
    for arm in arms:
        sec=arm_metric(agg,arm,ev,'untouched_state')
        lines.append(f"| {arm} | {fmt(get(agg,['arms',arm,'update_train_acc','mean']))} | {fmt(get(agg,['arms',arm,'base_anchor_acc_after','mean']))} | {fmt(arm_acc(agg,arm,ev,'event_role'))}/{fmt(arm_metric(agg,arm,ev,'event_role'))} | {fmt(arm_acc(agg,arm,ev,'focal_state'))}/{fmt(arm_metric(agg,arm,ev,'focal_state'))} | {fmt(arm_acc(agg,arm,ev,'untouched_state'))}/{fmt(sec)} | {fmt(sec-bsec if not math.isnan(sec) and not math.isnan(bsec) else float('nan'))} |\n")
    lines.append("\n")
    for ev in KEY_EVALS[1:]:
        lines.append(f"### {ev}: secondary margin and accuracy\n\n")
        lines.append("| arm | event acc | focal acc | secondary acc | secondary margin | delta from before |\n|---|---:|---:|---:|---:|---:|\n")
        b=base_metric(agg,ev,'untouched_state')
        lines.append(f"| BASE before | {fmt(base_acc(agg,ev,'event_role'))} | {fmt(base_acc(agg,ev,'focal_state'))} | {fmt(base_acc(agg,ev,'untouched_state'))} | {fmt(b)} | 0.000 |\n")
        for arm in arms:
            sec=arm_metric(agg,arm,ev,'untouched_state')
            lines.append(f"| {arm} | {fmt(arm_acc(agg,arm,ev,'event_role'))} | {fmt(arm_acc(agg,arm,ev,'focal_state'))} | {fmt(arm_acc(agg,arm,ev,'untouched_state'))} | {fmt(sec)} | {fmt(sec-b if not math.isnan(sec) and not math.isnan(b) else float('nan'))} |\n")
        lines.append("\n")
    # explicit contrasts where available
    lines.append("### Mechanism contrasts on trainTrain/trainHyp\n\n")
    if 'Etrue_Rtrue' in arms and 'Eflip_Rtrue' in arms:
        lines.append(f"- Event flip with rank true: event margin {fmt(arm_metric(agg,'Etrue_Rtrue',KEY_EVALS[0],'event_role'))} -> {fmt(arm_metric(agg,'Eflip_Rtrue',KEY_EVALS[0],'event_role'))}; secondary margin {fmt(arm_metric(agg,'Etrue_Rtrue',KEY_EVALS[0],'untouched_state'))} -> {fmt(arm_metric(agg,'Eflip_Rtrue',KEY_EVALS[0],'untouched_state'))}.\n")
    if 'Etrue_Rtrue' in arms and 'Etrue_Rflip' in arms:
        lines.append(f"- Rank flip with event true: focal margin {fmt(arm_metric(agg,'Etrue_Rtrue',KEY_EVALS[0],'focal_state'))} -> {fmt(arm_metric(agg,'Etrue_Rflip',KEY_EVALS[0],'focal_state'))}; secondary margin {fmt(arm_metric(agg,'Etrue_Rtrue',KEY_EVALS[0],'untouched_state'))} -> {fmt(arm_metric(agg,'Etrue_Rflip',KEY_EVALS[0],'untouched_state'))}.\n")
    if 'Etrue_only' in arms and 'Eflip_only' in arms:
        lines.append(f"- Event-only true vs flipped: secondary margin {fmt(arm_metric(agg,'Etrue_only',KEY_EVALS[0],'untouched_state'))} -> {fmt(arm_metric(agg,'Eflip_only',KEY_EVALS[0],'untouched_state'))}; focal margin {fmt(arm_metric(agg,'Etrue_only',KEY_EVALS[0],'focal_state'))} -> {fmt(arm_metric(agg,'Eflip_only',KEY_EVALS[0],'focal_state'))}.\n")
    if 'Rtrue_only' in arms and 'Rflip_only' in arms:
        lines.append(f"- Rank-only true vs flipped: event margin {fmt(arm_metric(agg,'Rtrue_only',KEY_EVALS[0],'event_role'))} -> {fmt(arm_metric(agg,'Rflip_only',KEY_EVALS[0],'event_role'))}; focal margin {fmt(arm_metric(agg,'Rtrue_only',KEY_EVALS[0],'focal_state'))} -> {fmt(arm_metric(agg,'Rflip_only',KEY_EVALS[0],'focal_state'))}; secondary margin {fmt(arm_metric(agg,'Rtrue_only',KEY_EVALS[0],'untouched_state'))} -> {fmt(arm_metric(agg,'Rflip_only',KEY_EVALS[0],'untouched_state'))}.\n")
    lines.append("\n")
    return lines

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('summaries', nargs='+')
    ap.add_argument('--out_dir', default='experiments/archive/representation_and_objectives/data/secondary_retention_analysis')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lines=["# research secondary-retention analysis\n\n",
           "This file compares no-direct-secondary sparse-update probes. The readout is whether a secondary ranking relation learned before the sparse update stays true after update rows that never directly query that secondary relation. Because these pilots use staged updating, they measure retention/consolidation and orientation interference; a joint no-secondary variant is needed to separate this from ordinary sequential forgetting.\n\n"]
    loaded=[]
    for s in args.summaries:
        p=Path(s); d=load_summary(p); d['_path']=str(p); loaded.append(d)
        lines.extend(table_for_summary(p.parent.name, d))
    (out/'secondary_retention_analysis.md').write_text(''.join(lines),'utf-8')
    (out/'secondary_retention_analysis.json').write_text(json.dumps({'sources':args.summaries,'n':len(loaded)},indent=2)+'\n','utf-8')
    print(json.dumps({'status':'secondary_retention_analysis_done','md':str(out/'secondary_retention_analysis.md')}))
if __name__=='__main__': main()
