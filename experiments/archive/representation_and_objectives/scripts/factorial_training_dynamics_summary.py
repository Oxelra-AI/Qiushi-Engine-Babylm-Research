#!/usr/bin/env python3
"""Summarize HS/LS/HD/LD training dynamics from repaired RoBERTa logs."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


RUNS = {
    "hs": Path("experiments/archive/representation_and_objectives/training/runs/factorial_hs_roberta_100M_accum_mb16"),
    "ls": Path("experiments/archive/representation_and_objectives/training/runs/factorial_ls_roberta_100M_accum_mb16"),
    "hd": Path("experiments/archive/representation_and_objectives/training/runs/factorial_hd_roberta_100M_accum_mb16"),
    "ld": Path("experiments/archive/representation_and_objectives/training/runs/factorial_ld_roberta_100M_accum_mb16"),
}
OUT_DIR = Path("experiments/archive/representation_and_objectives/data/factorial_training_dynamics")


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def by_window(logs: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    rows = [r for r in logs if start <= int(r["step"]) <= end]
    return {
        "step_start": start,
        "step_end": end,
        "n": len(rows),
        "loss_mean": mean([float(r["loss"]) for r in rows]),
        "mask_rate_mean": mean([float(r["effective_mask_rate"]) for r in rows]),
        "active_tokens_mean": mean([float(r["active_label_tokens"]) for r in rows]),
        "batch_words_mean": mean([float(r["batch_words"]) for r in rows]),
    }


def delta(a: float | None, b: float | None) -> float | None:
    return float(a - b) if a is not None and b is not None else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = {arm: read_jsonl(run / "training_log.jsonl") for arm, run in RUNS.items()}
    metrics = {arm: json.loads((run / "scientific_metrics.json").read_text(encoding="utf-8")) for arm, run in RUNS.items()}
    windows = [(1, 50), (451, 550), (951, 1050), (1451, 1550), (1951, 2050), (2480, 2529)]
    summaries: dict[str, Any] = {}
    for arm, rows in logs.items():
        summaries[arm] = {
            "n_steps": len(rows),
            "loss_first": rows[0]["loss"],
            "loss_last": rows[-1]["loss"],
            "loss_mean_all": mean([float(r["loss"]) for r in rows]),
            "mask_rate_mean_all": mean([float(r["effective_mask_rate"]) for r in rows]),
            "active_tokens_mean_all": mean([float(r["active_label_tokens"]) for r in rows]),
            "windows": [by_window(rows, s, e) for s, e in windows],
            "metrics_loss_first": metrics[arm].get("loss_first"),
            "metrics_loss_last": metrics[arm].get("loss_last"),
        }
    contrasts: dict[str, Any] = {"windows": []}
    for idx, (s, e) in enumerate(windows):
        vals = {arm: summaries[arm]["windows"][idx]["loss_mean"] for arm in RUNS}
        rec = {
            "step_start": s,
            "step_end": e,
            "loss_mean": vals,
            "HS_minus_LS": delta(vals["hs"], vals["ls"]),
            "HD_minus_LD": delta(vals["hd"], vals["ld"]),
        }
        rec["interaction"] = delta(rec["HS_minus_LS"], rec["HD_minus_LD"])
        contrasts["windows"].append(rec)
    final_vals = {arm: summaries[arm]["loss_last"] for arm in RUNS}
    contrasts["final_step_loss"] = {
        "loss": final_vals,
        "HS_minus_LS": delta(final_vals["hs"], final_vals["ls"]),
        "HD_minus_LD": delta(final_vals["hd"], final_vals["ld"]),
    }
    contrasts["final_step_loss"]["interaction"] = delta(contrasts["final_step_loss"]["HS_minus_LS"], contrasts["final_step_loss"]["HD_minus_LD"])
    payload = {
        "status": "FACTORIAL_TRAINING_DYNAMICS",
        "meaning": "Training-loss and mask-stat comparison for repaired RoBERTa HS/LS/HD/LD arms. Interpret only as reconstruction difficulty, not downstream competence.",
        "runs": {k: str(v) for k, v in RUNS.items()},
        "summaries": summaries,
        "contrasts": contrasts,
        "boundary": "Official-compatible selected evaluation remains the consequential mechanism readout.",
    }
    out_json = OUT_DIR / "factorial_training_dynamics.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research factorial training dynamics",
        "",
        f"JSON: `{out_json}`",
        "",
        "Final losses:",
    ]
    for arm in ["hs", "ls", "hd", "ld"]:
        md.append(f"- {arm}: {final_vals[arm]:.6f}")
    md.append("")
    md.append(f"Final loss interaction `(HS-LS)-(HD-LD)`: {contrasts['final_step_loss']['interaction']:.6f}")
    md.append("")
    md.append("This is a reconstruction-difficulty measurement only; do not use it as downstream evidence.")
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/factorial_training_dynamics/factorial_training_dynamics.md')).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "final_loss_interaction": contrasts["final_step_loss"]["interaction"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
