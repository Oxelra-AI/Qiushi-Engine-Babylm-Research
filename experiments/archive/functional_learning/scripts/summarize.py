#!/usr/bin/env python3
"""Summarize research transition and held-entity metrics."""
import csv, json
from pathlib import Path
import numpy as np

INP = Path("experiments/archive/functional_learning/data/query_first_binding/results.json")
OUT = Path("experiments/archive/functional_learning/data/query_first_binding/compact_summary.csv")
NOTE = Path("research/notes/functional_learning/query_first_binding_compact_summary.md")

d = json.load(open(INP))
rows = []
lines = ["# research compact summary", "", "This file records per-seed transition times and held-entity metrics for the query-first binding result.", ""]
for arm in d["config"]["arms"]:
    lines.append(f"## {arm}")
    vals_final = []
    vals_held = []
    for sd in d["config"]["seeds"]:
        rs = [r for r in d["records"] if r["arm"] == arm and r["seed"] == sd]
        if not rs:
            continue
        f = rs[-1]
        mid = None
        strong = None
        for r in rs:
            if mid is None and r["std"]["ctx_top1"] >= 0.5 and r["qswap"]["frac_pos"] >= 0.7:
                mid = r["epoch"]
            if strong is None and r["std"]["ctx_top1"] >= 0.8 and r["qswap"]["frac_pos"] >= 0.9 and r["bswap"]["frac_pos"] >= 0.9:
                strong = r["epoch"]
        rec = dict(
            arm=arm, seed=sd, mid_epoch=mid if mid is not None else "",
            strong_epoch=strong if strong is not None else "",
            std_cnll=f["std"]["correct_nll"], std_bag_mass=f["std"]["bag_mass"],
            std_top4=f["std"]["ctx_top1"], std_query_margin=f["std"]["query_margin"],
            bswap=f["bswap"]["mean"], bswap_frac=f["bswap"]["frac_pos"],
            qswap_margin=f["qswap"]["margin"], qswap_frac=f["qswap"]["frac_pos"],
            qswap_both=f["qswap"]["both"], corrupt_select=f["corrupt"]["novel_selectivity"],
            held_cnll=f["held"]["correct_nll"], held_bag_mass=f["held"]["bag_mass"],
            held_top4=f["held"]["ctx_top1"], held_query_margin=f["held"]["query_margin"],
        )
        rows.append(rec)
        vals_final.append((rec["std_cnll"], rec["std_top4"], rec["bswap"], rec["qswap_margin"], rec["corrupt_select"]))
        vals_held.append((rec["held_cnll"], rec["held_top4"], rec["held_query_margin"]))
        lines.append(
            f"- seed {sd}: mid={rec['mid_epoch'] or 'none'}, strong={rec['strong_epoch'] or 'none'}, "
            f"standard top4={rec['std_top4']:.3f}, Bswap={rec['bswap']:+.3f}, Qswap={rec['qswap_margin']:+.3f}, "
            f"held top4={rec['held_top4']:.3f}, held NLL={rec['held_cnll']:.3f}"
        )
    if vals_final:
        a = np.array(vals_final, dtype=float)
        h = np.array(vals_held, dtype=float)
        lines.append(
            f"- mean standard: NLL={a[:,0].mean():.3f}, top4={a[:,1].mean():.3f}, "
            f"Bswap={a[:,2].mean():+.3f}, Qswap={a[:,3].mean():+.3f}, selectivity={a[:,4].mean():+.3f}"
        )
        lines.append(
            f"- mean held: NLL={h[:,0].mean():.3f}, top4={h[:,1].mean():.3f}, query margin={h[:,2].mean():+.3f}"
        )
    lines.append("")
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
NOTE.write_text("\n".join(lines))
print({"status": "ok", "csv": str(OUT), "note": str(NOTE)})
