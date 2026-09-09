#!/usr/bin/env python3
"""Training-loss based internal signal around the scale1.75 77M-83M peak.

Uses only the existing legal run's training_log.jsonl and checkpoint exposure records.
No model inference, no official evaluation data.  The purpose is to test whether
moving averages/slopes of the self-supervised training loss could have selected
or warned about the 82M competence peak.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import statistics
import time
from typing import Any

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
A01 = _public_path('experiments/archive/representation_and_objectives')
RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder')
METRICS = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/scientific_metrics.json')
LOG = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/training_log.jsonl')
SWEEP = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_checkpoint_sweep/summary/scale1p75_checkpoint_sweep_summary.json')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_training_loss_peak_signal')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_training_loss_peak_signal/scale1p75_training_loss_peak_signal.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/scale1p75_training_loss_peak_signal/scale1p75_training_loss_peak_signal.md')
WINDOW = [f"chck_{m}M" for m in range(77, 84)]


def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def rel(p):
    pp = pathlib.Path(p)
    try: return str(pp.resolve().relative_to(ROOT))
    except Exception: return str(pp)


def mean(vals):
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def slope_ols(xs, ys):
    if len(xs) < 2: return None
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0: return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3: return None
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0: return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rank(vals, reverse=False):
    return sorted(vals, key=lambda x: (float("inf") if x[1] is None else x[1]), reverse=reverse)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS.read_text())
    logs = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    by_step = {int(r["step"]): r for r in logs}
    checkpoints = {c["name"]: c for c in metrics["saved_checkpoints"]}
    sweep = json.loads(SWEEP.read_text())
    cheap = {r["endpoint"]: r.get("cheap7") for r in sweep["rows"] if r["endpoint"] in WINDOW}
    rows=[]
    for ck in WINDOW:
        rec = checkpoints[ck]
        # Find the training-log update that matches this checkpoint's cumulative exposure.
        target_words = rec["actual_cumulative_word_exposure"]
        idx = min(range(len(logs)), key=lambda i: abs(int(logs[i]["cumulative_word_exposure"]) - int(target_words)))
        r = logs[idx]
        def losses(a,b):
            lo=max(0,idx+a); hi=min(len(logs),idx+b)
            return [logs[j]["loss"] for j in range(lo,hi)]
        row={
            "checkpoint": ck,
            "cheap7": cheap.get(ck),
            "checkpoint_step_record": rec,
            "matched_log_index0": idx,
            "matched_step": r["step"],
            "matched_cumulative_words": r["cumulative_word_exposure"],
            "matched_loss": r["loss"],
            "matched_lr": r["lr"],
            "trailing_loss_10": mean(losses(-9,1)),
            "trailing_loss_25": mean(losses(-24,1)),
            "trailing_loss_50": mean(losses(-49,1)),
            "trailing_loss_100": mean(losses(-99,1)),
            "centered_loss_21": mean(losses(-10,11)),
            "centered_loss_51": mean(losses(-25,26)),
            "forward_loss_25": mean(losses(1,26)),
            "forward_loss_50": mean(losses(1,51)),
        }
        for w in (25,50,100):
            sub = logs[max(0,idx-w+1):idx+1]
            row[f"trailing_loss_slope_per_step_{w}"] = slope_ols([x["step"] for x in sub], [x["loss"] for x in sub])
        rows.append(row)
    correlations = {}
    fields=[k for k in rows[0].keys() if k not in {"checkpoint","cheap7","checkpoint_step_record"} and isinstance(rows[0].get(k),(int,float))]
    for f in fields:
        correlations[f] = pearson([r[f] for r in rows], [r["cheap7"] for r in rows])
    # Which checkpoint would each loss statistic choose if minimized/maximized?
    selectors = {}
    for f in fields:
        vals=[(r["checkpoint"], r[f]) for r in rows]
        selectors[f] = {
            "min_checkpoint": min(vals, key=lambda x: x[1] if x[1] is not None else float("inf"))[0],
            "min_value": min(vals, key=lambda x: x[1] if x[1] is not None else float("inf"))[1],
            "max_checkpoint": max(vals, key=lambda x: -float("inf") if x[1] is None else x[1])[0],
            "max_value": max(vals, key=lambda x: -float("inf") if x[1] is None else x[1])[1],
        }
    result={
        "status":"SCALE1P75_TRAINING_LOSS_PEAK_SIGNAL",
        "created_utc": now(),
        "purpose":"Test whether legal self-supervised training-loss statistics around checkpoint boundaries identify the chck_82M competence peak.",
        "run_dir": rel(RUN),
        "rows": rows,
        "correlations_with_cheap7": correlations,
        "selector_extrema": selectors,
        "peak_by_cheap7":"chck_82M",
        "scientific_reading":"Training loss would be a useful legal stopping signal only if its extrema or slope singled out chck_82M, or at least warned against continuing. If loss keeps improving or chooses another checkpoint while cheap7 peaks at 82M, then simple MLM-loss monitoring is not a sufficient benchmark-independent selector.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    lines=["# research scale1.75 training-loss peak signal", "", f"Status: `{result['status']}`", "", "| checkpoint | cheap7 | step | lr | loss | trail25 | trail50 | trail100 | centered51 | slope50 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['checkpoint']} | {r['cheap7']} | {r['matched_step']} | {r['matched_lr']:.8g} | {r['matched_loss']:.6f} | {r['trailing_loss_25']:.6f} | {r['trailing_loss_50']:.6f} | {r['trailing_loss_100']:.6f} | {r['centered_loss_51']:.6f} | {r['trailing_loss_slope_per_step_50']:.6g} |")
    lines += ["", "## Selector extrema", "", "```json", json.dumps(selectors, indent=2), "```", "", "## Correlations with cheap7", "", "```json", json.dumps(correlations, indent=2), "```", "", result["scientific_reading"], "", f"JSON: `{rel(OUT_JSON)}`"]
    OUT_MD.write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":result["status"],"out_json":rel(OUT_JSON),"out_md":rel(OUT_MD)},indent=2), flush=True)

if __name__ == "__main__":
    main()
