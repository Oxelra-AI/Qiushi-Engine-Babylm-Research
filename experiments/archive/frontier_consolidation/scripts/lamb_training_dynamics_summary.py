#!/usr/bin/env python3
"""research: summarize LAMB training dynamics vs research and Muon.

This is a low-cost CPU analysis of existing training logs. It asks whether the
LAMB arms are optimizing in the same regime as research/Muon or whether cheap-column
behavior should be interpreted as underlearning/overdamping.
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
STUDY = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/lamb_training_dynamics')

RUNS = {
    "adamw_20M": {
        "label": "research AdamW legal reference",
        "path": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/training_log.jsonl'),
    },
    "muon_wdmatched_20M": {
        "label": "Muon hidden wd-matched",
        "path": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_20M/training_log.jsonl'),
    },
    "lamb005_partial": {
        "label": "LAMB lr=0.005 partial timeout",
        "path": _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M/training_log.jsonl'),
    },
    "lamb007_20M": {
        "label": "LAMB lr=0.007 complete",
        "path": _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr007_seed43022_20M/training_log.jsonl'),
    },
}
TARGETS = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 14_000_000, 15_000_000, 20_000_000]


def load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def nearest(rows: list[dict], target: int) -> dict | None:
    if not rows:
        return None
    eligible = [r for r in rows if int(r.get("cumulative_word_exposure", 0)) >= target]
    if eligible:
        return eligible[0]
    return rows[-1]


def tail_mean(rows: list[dict], last_n: int = 20) -> float | None:
    if not rows:
        return None
    vals = [float(r["loss"]) for r in rows[-last_n:]]
    return float(mean(vals)) if vals else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"status": "LAMB_TRAINING_DYNAMICS", "targets": TARGETS, "runs": {}, "matched_loss_table": {}}
    logs = {}
    for key, meta in RUNS.items():
        rows = load_log(meta["path"])
        logs[key] = rows
        if rows:
            result["runs"][key] = {
                "label": meta["label"],
                "path": str(meta["path"]),
                "n_log_rows": len(rows),
                "first": rows[0],
                "last": rows[-1],
                "tail20_loss_mean": tail_mean(rows, 20),
                "tail50_loss_mean": tail_mean(rows, 50),
                "complete_20M": int(rows[-1].get("cumulative_word_exposure", 0)) >= 20_000_000,
            }
        else:
            result["runs"][key] = {"label": meta["label"], "path": str(meta["path"]), "missing": True}

    for target in TARGETS:
        row = {}
        for key, rows in logs.items():
            nr = nearest(rows, target)
            if nr:
                row[key] = {
                    "step": int(nr["step"]),
                    "words": int(nr["cumulative_word_exposure"]),
                    "loss": float(nr["loss"]),
                    "lr": float(nr["lr"]),
                }
        if "adamw_20M" in row:
            base_loss = row["adamw_20M"]["loss"]
            for key in row:
                row[key]["loss_delta_vs_step35"] = row[key]["loss"] - base_loss
        result["matched_loss_table"][str(target)] = row

    # A compact derived interpretation carrier: final/partial losses at the same target.
    interp = []
    for key in ["lamb005_partial", "lamb007_20M", "muon_wdmatched_20M"]:
        if key not in result["runs"] or result["runs"][key].get("missing"):
            continue
        last = result["runs"][key]["last"]
        target = int(last["cumulative_word_exposure"])
        b = nearest(logs["adamw_20M"], target)
        interp.append({
            "run": key,
            "words": target,
            "loss": float(last["loss"]),
            "loss_at_same_or_nearest_words": float(b["loss"]) if b else None,
            "loss_delta_vs_step35_nearest": float(last["loss"] - b["loss"]) if b else None,
            "last_lr": float(last["lr"]),
        })
    result["derived_last_vs_step35"] = interp

    (_public_path('experiments/archive/frontier_consolidation/data/lamb_training_dynamics/lamb_training_dynamics.json')).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = ["# research LAMB training dynamics", "", "Existing logs only; no new training.", "", "## Run endpoints", "", "| run | rows | last step | words | loss | tail20 loss | complete 20M |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, rec in result["runs"].items():
        if rec.get("missing"):
            lines.append(f"| {rec['label']} | — | — | — | — | — | False |")
            continue
        last = rec["last"]
        lines.append(f"| {rec['label']} | {rec['n_log_rows']} | {last['step']} | {last['cumulative_word_exposure']} | {float(last['loss']):.4f} | {rec['tail20_loss_mean']:.4f} | {rec['complete_20M']} |")
    lines += ["", "## Matched/nearest losses", "", "| target words | research loss | Muon loss Δ | LAMB005 loss Δ | LAMB007 loss Δ |", "|---:|---:|---:|---:|---:|"]
    for target in TARGETS:
        row = result["matched_loss_table"][str(target)]
        def v(key, field="loss_delta_vs_step35"):
            if key not in row:
                return "—"
            if field == "loss":
                return f"{row[key]['loss']:.4f}"
            return f"{row[key].get(field, 0):+.4f}"
        lines.append(f"| {target} | {v('adamw_20M', 'loss')} | {v('muon_wdmatched_20M')} | {v('lamb005_partial')} | {v('lamb007_20M')} |")
    lines += ["", "## Last available comparison", "", "| run | words | loss | research loss at nearest words | Δloss | last LR |", "|---|---:|---:|---:|---:|---:|"]
    for rec in interp:
        lines.append(f"| {rec['run']} | {rec['words']} | {rec['loss']:.4f} | {rec['loss_at_same_or_nearest_words']:.4f} | {rec['loss_delta_vs_step35_nearest']:+.4f} | {rec['last_lr']:.6g} |")
    lines += ["", "Interpretation: lr0.007 completing 20M with a high tail loss would mean this arm is not a mature-candidate optimizer even if some cheap columns move. lr0.005 timing out after 14M is a resource/runtime fact; its scientific status should be judged by the 14M cheap evaluation and by whether its update-allocation readout preserves the intended Adam-direction mechanism."]
    (_public_path('research/documents/frontier_consolidation/data/lamb_training_dynamics/lamb_training_dynamics.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": str(OUT), "derived_last_vs_step35": interp}, indent=2))


if __name__ == "__main__":
    main()
