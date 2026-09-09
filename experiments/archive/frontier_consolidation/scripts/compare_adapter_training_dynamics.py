#!/usr/bin/env python3
"""Compare research, research compressed adapters, and research matched-horizon adapter logs."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
from pathlib import Path
from statistics import mean

ROOT = _public_path('experiments/archive/frontier_consolidation')
RUNS = {
    "reference_100M_horizon": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2'),
    "adapter128_compressed20M": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_seed43022_20M'),
    "adapter64_compressed20M": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter64_seed43022_20M'),
    "adapter128_live_100M_horizon": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022'),
    "adapter128_disabled_100M_horizon": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_disabled_h100M20M_seed43022'),
}
OUT = _public_path('experiments/archive/frontier_consolidation/data/adapter_training_dynamics')
WATCH = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 506]


def load_log(run_dir: Path):
    p = run_dir / "training_log.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_metrics(run_dir: Path):
    p = run_dir / "scientific_metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def summarize_run(label, run_dir):
    log = load_log(run_dir)
    metrics = load_metrics(run_dir)
    if not log:
        return {"label": label, "run_dir": str(run_dir), "status": "missing_log"}
    by_step = {int(r["step"]): r for r in log}
    rows = {str(s): by_step.get(s) for s in WATCH if s in by_step}
    return {
        "label": label,
        "run_dir": str(run_dir),
        "status": "ok",
        "n_logged_rows": len(log),
        "last_logged_step": int(log[-1]["step"]),
        "loss_first": log[0].get("loss"),
        "loss_last_logged": log[-1].get("loss"),
        "lr_first": log[0].get("lr"),
        "lr_last_logged": log[-1].get("lr"),
        "word_last_logged": log[-1].get("cumulative_word_exposure"),
        "metrics_word_exposure": metrics.get("word_exposure") if metrics else None,
        "metrics_loss_last": metrics.get("loss_last") if metrics else None,
        "metrics_actual_training_steps": metrics.get("actual_training_steps") if metrics else None,
        "parameter_count": metrics.get("parameter_count") if metrics else None,
        "watch_steps": rows,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {k: summarize_run(k, p) for k, p in RUNS.items()}
    # Pairwise deltas against research for steps that exist in both logs.
    ref = payload.get("reference_100M_horizon", {})
    ref_rows = ref.get("watch_steps", {}) if ref.get("status") == "ok" else {}
    for k, rec in payload.items():
        if k == "reference_100M_horizon" or rec.get("status") != "ok":
            continue
        deltas = {}
        for s, row in rec.get("watch_steps", {}).items():
            rr = ref_rows.get(s)
            if row and rr:
                deltas[s] = {
                    "loss_delta_vs_step35": row["loss"] - rr["loss"],
                    "lr_delta_vs_step35": row["lr"] - rr["lr"],
                    "masked_tokens_delta_vs_step35": row.get("masked_tokens", 0) - rr.get("masked_tokens", 0),
                    "batch_words_delta_vs_step35": row.get("batch_words", 0) - rr.get("batch_words", 0),
                }
        rec["deltas_vs_step35_watch_steps"] = deltas
    (_public_path('experiments/archive/frontier_consolidation/data/adapter_training_dynamics/adapter_training_dynamics.json')).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# research adapter training dynamics", "", "## Watch-step comparison", "", "| run | step | loss | Δloss vs research | lr | Δlr vs research | words | masked |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, rec in payload.items():
        if rec.get("status") != "ok":
            lines.append(f"| {k} | — | — | — | — | — | — | — |")
            continue
        for s in ["1", "50", "100", "150", "200", "250", "300", "350", "400", "450", "500", "506"]:
            row = rec.get("watch_steps", {}).get(s)
            if not row: continue
            d = rec.get("deltas_vs_step35_watch_steps", {}).get(s, {})
            dl = d.get("loss_delta_vs_step35")
            dls = "" if dl is None else f"{dl:+.6f}"
            dlr = d.get("lr_delta_vs_step35")
            dlrs = "" if dlr is None else f"{dlr:+.9f}"
            lines.append(f"| {k} | {s} | {row['loss']:.6f} | {dls} | {row['lr']:.9f} | {dlrs} | {row.get('cumulative_word_exposure')} | {row.get('masked_tokens')} |")
    lines += ["", "## Key interpretation", "", "- research compressed adapters used a 506-step schedule, causing LR to approach zero by 20M; their final loss stayed near 6.88.", "- research live adapter uses `lr_total_steps=2529`, matching the research 100M horizon; if its watch-step LR/masks match research and final loss returns to ~3.75, the research loss collapse was a schedule artifact.", "- The disabled control is the exact check for gradient checkpointing/custom-model execution once its log exists.", ""]
    (_public_path('research/documents/frontier_consolidation/data/adapter_training_dynamics/adapter_training_dynamics.md')).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "out_json": str(_public_path('experiments/archive/frontier_consolidation/data/adapter_training_dynamics/adapter_training_dynamics.json')), "runs": list(payload)}, indent=2))

if __name__ == "__main__":
    main()
