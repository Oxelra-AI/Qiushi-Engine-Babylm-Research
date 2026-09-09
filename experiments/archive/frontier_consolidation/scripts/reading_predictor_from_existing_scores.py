#!/usr/bin/env python3
"""research: inspect whether 20M Reading damage predicts mature cheap7 failure.

Uses existing official-compatible per_target JSON files. It is not a new eval; it
summarizes already produced score tables to contextualize the Muon 20M Reading
loss before the 80M result arrives.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
from pathlib import Path
from statistics import mean

ROOT = _public_path('.')
S = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/existing_reading_predictor')
CHEAP = ["BLiMP","Supplement","EWoK","Entity","COMPS","GlobalPIQA","Reading"]
ZERO = ["BLiMP","Supplement","EWoK","Entity","COMPS"]

CANDIDATES = {
    "reinvest": {
        20: _public_path('experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json'),
        70: _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json'),
        80: _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json'),
        100: _public_path('experiments/archive/frontier_consolidation/data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json'),
    },
    "clean_control": {
        20: _public_path('experiments/archive/frontier_consolidation/data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_20M.json'),
        70: _public_path('experiments/archive/frontier_consolidation/data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_70M.json'),
        80: _public_path('experiments/archive/frontier_consolidation/data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_80M.json'),
    },
    "wordmean": {
        70: _public_path('experiments/archive/frontier_consolidation/data/wordmean_70_80M_eval/per_target/wordmean_seed43022_70M.json'),
        80: _public_path('experiments/archive/frontier_consolidation/data/wordmean_70_80M_eval/per_target/wordmean_seed43022_80M.json'),
    },
    "minfreq50": {
        70: _public_path('experiments/archive/frontier_consolidation/data/minfreq50_initmatched_70_80M_eval/per_target/minfreq50_initmatched_seed43022_70M.json'),
        80: _public_path('experiments/archive/frontier_consolidation/data/minfreq50_initmatched_70_80M_eval/per_target/minfreq50_initmatched_seed43022_80M.json'),
    },
    "innovation": {
        70: _public_path('experiments/archive/frontier_consolidation/data/strict_innovation_70_80M_eval/per_target/strict_content_innovation_seed43022_70M.json'),
        80: _public_path('experiments/archive/frontier_consolidation/data/strict_innovation_70_80M_eval/per_target/strict_content_innovation_seed43022_80M.json'),
    },
    "fw_compact": {
        70: _public_path('experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/fw_compact_view_shared16k_seed43022_70M.json'),
        80: _public_path('experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/fw_compact_view_shared16k_seed43022_80M.json'),
        100: _public_path('experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/fw_compact_view_shared16k_seed43022_100M.json'),
    },
    "fw_breadth": {
        70: _public_path('experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/fw_source_breadth_shared16k_seed43022_70M.json'),
        80: _public_path('experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/fw_source_breadth_shared16k_seed43022_80M.json'),
        100: _public_path('experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/fw_source_breadth_shared16k_seed43022_100M.json'),
    },
    "rtd20": {
        20: _public_path('experiments/archive/frontier_consolidation/data/mlm_rtd_20M_eval/per_target/mlm_rtd_lambda1_seed43022_20M.json'),
    },
    "muon008_wd01": {
        20: _public_path('experiments/archive/frontier_consolidation/data/muon_20M_eval/per_target/muon_lr008_20M.json'),
    },
    "muon012_wd01": {
        20: _public_path('experiments/archive/frontier_consolidation/data/muon_20M_eval/per_target/muon_lr012_20M.json'),
    },
    "muon008_wd00125": {
        20: _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_20M_eval/per_target/muon_lr008_wd00125_20M.json'),
    },
}


def load(path):
    return json.loads(path.read_text()) if path.exists() else None

def scores(payload):
    if payload is None: return None
    tasks = payload.get('tasks', {})
    out = {}
    for c in ZERO:
        rec = tasks.get(c, {})
        out[c] = float(rec['score']) if rec.get('score') is not None else None
    gp = []
    for c in ['GlobalPIQA_parallel','GlobalPIQA_nonparallel']:
        rec = tasks.get(c, {})
        if rec.get('score') is not None: gp.append(float(rec['score']))
    out['GlobalPIQA'] = mean(gp) if len(gp)==2 else None
    rd = tasks.get('Reading', {})
    if isinstance(rd.get('scores'), dict) and rd['scores'].get('Reading') is not None:
        out['Reading'] = float(rd['scores']['Reading'])
    elif rd.get('score') is not None:
        out['Reading'] = float(rd['score'])
    else:
        out['Reading'] = None
    return out

def c7(sc):
    if sc is None: return None
    vals=[sc.get(c) for c in CHEAP]
    if any(v is None for v in vals): return None
    return mean(float(v) for v in vals)

def corr(xs, ys):
    n=len(xs)
    if n<3: return None
    mx=mean(xs); my=mean(ys)
    vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    if vx<=0 or vy<=0: return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(vx*vy)**0.5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = {}
    for name, ckpts in CANDIDATES.items():
        table[name] = {}
        for step, path in ckpts.items():
            sc = scores(load(path))
            table[name][str(step)] = {"path": str(path), "exists": path.exists(), "scores": sc, "cheap7": c7(sc)}
    # within-model changes from earliest to latest available
    changes=[]
    for name, rec in table.items():
        steps=sorted(int(k) for k,v in rec.items() if v.get('cheap7') is not None)
        if len(steps)>=2:
            a,b=steps[0],steps[-1]
            sa,sb=rec[str(a)]['scores'],rec[str(b)]['scores']
            changes.append({
                "name": name, "from": a, "to": b,
                "reading_delta": sb['Reading']-sa['Reading'],
                "cheap7_delta": rec[str(b)]['cheap7']-rec[str(a)]['cheap7'],
                "ewok_delta": sb['EWoK']-sa['EWoK'],
                "globalpiqa_delta": sb['GlobalPIQA']-sa['GlobalPIQA'],
            })
    xs=[x['reading_delta'] for x in changes]
    ys=[x['cheap7_delta'] for x in changes]
    summary={"status":"EXISTING_READING_PREDICTOR", "table":table, "changes":changes, "reading_delta_vs_cheap7_delta_corr": corr(xs,ys)}
    (_public_path('experiments/archive/frontier_consolidation/data/existing_reading_predictor/existing_reading_predictor.json')).write_text(json.dumps(summary, indent=2)+"\n")
    lines=["# research existing score trajectories: Reading as predictor", "", f"Correlation of within-run Reading change vs cheap7 change (available multi-checkpoint runs): {summary['reading_delta_vs_cheap7_delta_corr']}", "", "| run | from→to | ΔReading | Δcheap7 | ΔEWoK | ΔGlobalPIQA |", "|---|---:|---:|---:|---:|---:|"]
    for ch in changes:
        lines.append(f"| {ch['name']} | {ch['from']}→{ch['to']} | {ch['reading_delta']:+.3f} | {ch['cheap7_delta']:+.3f} | {ch['ewok_delta']:+.3f} | {ch['globalpiqa_delta']:+.3f} |")
    lines += ["", "## 20M rows", "", "| run | Reading | cheap7 | EWoK | GlobalPIQA |", "|---|---:|---:|---:|---:|"]
    for name, rec in table.items():
        if '20' in rec and rec['20'].get('scores'):
            sc=rec['20']['scores']
            lines.append(f"| {name} | {sc['Reading']:.3f} | {rec['20']['cheap7']:.3f} | {sc['EWoK']:.2f} | {sc['GlobalPIQA']:.2f} |")
    (_public_path('research/documents/frontier_consolidation/data/existing_reading_predictor/existing_reading_predictor.md')).write_text("\n".join(lines)+"\n")
    print(json.dumps({"status": summary['status'], "corr": summary['reading_delta_vs_cheap7_delta_corr'], "changes": changes}, indent=2))

if __name__ == '__main__': main()
