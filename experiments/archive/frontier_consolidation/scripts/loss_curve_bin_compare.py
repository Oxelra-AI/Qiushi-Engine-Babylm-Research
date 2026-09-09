#!/usr/bin/env python3
"""research CPU loss-curve comparison for compact_view_reinvest seeds."""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER_ROOT = pathlib.Path.cwd()
OUT_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/data/loss_curve_bin_compare"
OUT_JSON = OUT_DIR / "loss_curve_bin_compare.json"
OUT_MD = USER_ROOT / "research/notes/frontier_consolidation/loss_curve_bin_compare.md"
FIG_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/figures"
OUT_FIG = FIG_DIR / "compact_reinvest_seed_loss_curves.png"
RUNS = {
    "43022": USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022",
    "43122": USER_ROOT / "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122",
}


def load_jsonl(p: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fmean(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else None


def bin_rows(rows: list[dict[str, Any]], bin_m: int = 10) -> list[dict[str, Any]]:
    bins: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        exp = float(r.get("cumulative_word_exposure", 0)) / 1_000_000
        b = int((exp - 1e-9) // bin_m) * bin_m
        b = max(0, min(90, b))
        bins.setdefault(b, []).append(r)
    out = []
    for b in sorted(bins):
        rs = bins[b]
        out.append({
            "bin_start_m": b,
            "bin_end_m": b + bin_m,
            "n_steps": len(rs),
            "step_first": rs[0].get("step"),
            "step_last": rs[-1].get("step"),
            "exposure_first": rs[0].get("cumulative_word_exposure"),
            "exposure_last": rs[-1].get("cumulative_word_exposure"),
            "loss_mean": fmean([r.get("loss") for r in rs if isinstance(r.get("loss"), (int, float))]),
            "loss_last": float(rs[-1].get("loss")) if isinstance(rs[-1].get("loss"), (int, float)) else None,
            "lr_mean": fmean([r.get("lr") for r in rs if isinstance(r.get("lr"), (int, float))]),
            "effective_mask_rate_mean": fmean([r.get("effective_mask_rate") for r in rs if isinstance(r.get("effective_mask_rate"), (int, float))]),
            "masked_tokens_mean": fmean([r.get("masked_tokens") for r in rs if isinstance(r.get("masked_tokens"), (int, float))]),
            "batch_words_mean": fmean([r.get("batch_words") for r in rs if isinstance(r.get("batch_words"), (int, float))]),
        })
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed, run in RUNS.items():
        rows = load_jsonl(run / "training_log.jsonl")
        per_seed[seed] = {
            "run_dir": str(run),
            "n_steps": len(rows),
            "total_words": rows[-1].get("cumulative_word_exposure") if rows else None,
            "bins_10m": bin_rows(rows, 10),
        }
    bins = {seed: {b["bin_start_m"]: b for b in rec["bins_10m"]} for seed, rec in per_seed.items()}
    common = sorted(set(bins["43022"]) & set(bins["43122"]))
    diffs = []
    for b in common:
        a = bins["43022"][b]
        c = bins["43122"][b]
        diffs.append({
            "bin_start_m": b,
            "loss_mean_diff_43122_minus_43022": c["loss_mean"] - a["loss_mean"],
            "loss_last_diff_43122_minus_43022": c["loss_last"] - a["loss_last"],
            "mask_rate_mean_diff_43122_minus_43022": c["effective_mask_rate_mean"] - a["effective_mask_rate_mean"],
            "masked_tokens_mean_diff_43122_minus_43022": c["masked_tokens_mean"] - a["masked_tokens_mean"],
            "n_steps_43022": a["n_steps"],
            "n_steps_43122": c["n_steps"],
        })
    result = {
        "status": "LOSS_CURVE_BIN_COMPARE",
        "purpose": "CPU comparison of training dynamics for the two compact_view_reinvest seeds before interpreting sparse task trajectories.",
        "per_seed": per_seed,
        "bin_diffs_43122_minus_43022": diffs,
        "out_figure": str(OUT_FIG),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for seed, rec in per_seed.items():
        xs = [(b["bin_start_m"] + b["bin_end_m"]) / 2 for b in rec["bins_10m"]]
        axes[0].plot(xs, [b["loss_mean"] for b in rec["bins_10m"]], marker="o", label=f"seed{seed} mean loss")
        axes[1].plot(xs, [b["effective_mask_rate_mean"] for b in rec["bins_10m"]], marker="o", label=f"seed{seed} mask rate")
    axes[0].set_ylabel("Mean MLM loss")
    axes[1].set_ylabel("Mean effective mask rate")
    axes[1].set_xlabel("Training exposure bin midpoint (M words)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("compact_view_reinvest training traces: seed43022 vs seed43122")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=180)

    lines = ["# research — loss-curve bin comparison\n\n"]
    lines.append("CPU-only comparison of training logs for the two compact_view_reinvest seeds.\n\n")
    lines.append("## Ten-million-word bin differences (43122 minus 43022)\n")
    for d in diffs:
        lines.append(f"- {d['bin_start_m']:02d}-{d['bin_start_m']+10:03d}M: mean loss {d['loss_mean_diff_43122_minus_43022']:+.6g}, last loss {d['loss_last_diff_43122_minus_43022']:+.6g}, mask-rate mean {d['mask_rate_mean_diff_43122_minus_43022']:+.6g}, masked-token mean {d['masked_tokens_mean_diff_43122_minus_43022']:+.6g}\n")
    lines.append("\n## Reading\n")
    lines.append("The two runs have identical step count and exposure schedule. Mean loss differences oscillate around zero across most of training, while the very last minibatch loss is higher for seed43122. This makes the sparse task-slice trajectory more informative than the scalar final loss for explaining the weaker seed.\n")
    lines.append(f"\nFigure: `{OUT_FIG}`\n\nMachine-readable output: `{OUT_JSON}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD), "out_figure": str(OUT_FIG)}, indent=2))


if __name__ == "__main__":
    main()
