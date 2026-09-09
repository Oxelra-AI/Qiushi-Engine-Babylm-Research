#!/usr/bin/env python3
"""research: compare selected causal GPT compact-vs-repeat trajectories.

Consumes selected_causal_trajectory.json files produced by
selected_causal_checkpoint_eval.py and writes a mechanism-focused contrast.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from statistics import mean
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EFFECTIVE_TOKEN_REL_DELTA = 0.002216  # research compact active 256-token positions per epoch vs repeat.


def read_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        data = data["rows"]
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [r for r in data if r.get("cheap7") is not None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", type=pathlib.Path, required=True)
    ap.add_argument("--repeat", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    compact = {int(r["words"]): r for r in read_rows(args.compact)}
    repeat = {int(r["words"]): r for r in read_rows(args.repeat)}
    common = sorted(set(compact) & set(repeat))
    contrasts=[]
    for w in common:
        c=compact[w]; r=repeat[w]
        row={"words":w,"endpoint":c.get("endpoint"),"compact_cheap7":c.get("cheap7"),"repeat_cheap7":r.get("cheap7"),"delta_cheap7":float(c["cheap7"])-float(r["cheap7"])}
        for col in CHEAP_COLUMNS:
            if c.get(col) is not None and r.get(col) is not None:
                row[f"delta_{col}"]=float(c[col])-float(r[col])
        contrasts.append(row)
    broad_positive=[]
    for row in contrasts:
        deltas=[row.get(f"delta_{c}") for c in CHEAP_COLUMNS if row.get(f"delta_{c}") is not None]
        broad_positive.append(sum(1 for d in deltas if d>0))
        row["n_positive_columns"]=sum(1 for d in deltas if d>0)
        row["n_negative_columns"]=sum(1 for d in deltas if d<0)
    best=max(contrasts,key=lambda r:r["delta_cheap7"]) if contrasts else None
    result={
        "status":"CAUSAL_COMPACT_REPEAT_CONTRAST" if contrasts else "NO_COMMON_POINTS",
        "compact_path":str(args.compact),
        "repeat_path":str(args.repeat),
        "n_common":len(common),
        "effective_token_relative_delta_compact_minus_repeat":EFFECTIVE_TOKEN_REL_DELTA,
        "important_interpretation":"A compact advantage must be broad across checkpoints/families and materially larger than what a +0.2216% effective-token asymmetry could plausibly explain. A one-column or one-checkpoint spike is redistribution, not architecture-transfer evidence.",
        "best_delta_endpoint":best,
        "mean_delta_cheap7":mean([r["delta_cheap7"] for r in contrasts]) if contrasts else None,
        "mean_positive_columns":mean(broad_positive) if broad_positive else None,
        "contrasts":contrasts,
    }
    out_json=args.out_dir/"causal_compact_repeat_contrast.json"
    out_md=args.out_dir/"causal_compact_repeat_contrast.md"
    out_json.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    md=["# Causal GPT compact-vs-repeat selected trajectory contrast\n\n"]
    md.append(f"Common checkpoints: {len(common)}. Compact effective active-token exposure is +{EFFECTIVE_TOKEN_REL_DELTA*100:.4f}% relative to repeat at equal legal words (research audit).\n\n")
    if best:
        md.append(f"Best selected delta: {best['endpoint']} Δcheap7={best['delta_cheap7']:+.6f}, positive columns={best['n_positive_columns']}/7.\n\n")
        md.append("| M | compact | repeat | Δcheap7 | +cols | ΔBLiMP | ΔSupp | ΔEWoK | ΔEntity | ΔCOMPS | ΔGPIQA | ΔReading |\n")
        md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in contrasts:
            md.append(f"| {row['words']/1e6:.0f} | {row['compact_cheap7']:.6f} | {row['repeat_cheap7']:.6f} | {row['delta_cheap7']:+.6f} | {row['n_positive_columns']} | " + " | ".join(f"{row.get('delta_'+c,0):+.3f}" for c in CHEAP_COLUMNS) + " |\n")
    md.append("\nInterpretation: broad sustained positive deltas support transfer of the compact semantic-view mechanism into the causal coordinate; isolated spikes or small deltas comparable to the token asymmetry do not.\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md),encoding="utf-8")
    print(json.dumps({"status":result["status"],"out_json":str(out_json),"out_md":str(out_md)},indent=2),flush=True)

if __name__=="__main__":
    main()
