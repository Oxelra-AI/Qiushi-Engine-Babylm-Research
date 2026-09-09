#!/usr/bin/env python3
"""Compare recorded 1M pilot metrics across eight optimizer-data arms.

Reads only the existing per-arm scientific_metrics.json files. Training events
in the combined log cannot be uniquely assigned to simultaneous arms, so this
summary does not estimate training dynamics or stability from that log.
"""
import json
import sys
from pathlib import Path

ARMS = [
    ("adamw", "0.001", "official"),
    ("lamb",  "0.001", "official"),
    ("adamw", "0.001", "qwen"),
    ("lamb",  "0.001", "qwen"),
    ("adamw", "0.007", "official"),
    ("lamb",  "0.007", "official"),
    ("adamw", "0.007", "qwen"),
    ("lamb",  "0.007", "qwen"),
]


def run_dir_name(opt: str, lr: str, corpus: str) -> str:
    return f"{opt}_lr{lr}_{corpus}_1M_seed43022"


def analyze_arm(base: Path, opt: str, lr: str, corpus: str) -> dict:
    run_name = run_dir_name(opt, lr, corpus)
    run_path = base / "training" / "runs" / run_name
    result = {"arm": run_name, "optimizer": opt, "lr": float(lr), "corpus": corpus}

    metrics_path = run_path / "scientific_metrics.json"
    if not metrics_path.exists():
        result["status"] = "missing"
        return result

    metrics = json.loads(metrics_path.read_text())
    result["status"] = "complete"
    result["loss_last"] = metrics.get("loss_last")
    result["total_steps"] = metrics.get("total_steps")
    result["cumulative_words"] = metrics.get("cumulative_words")
    result["param_count"] = metrics.get("param_count")
    result["n_pair_rows"] = metrics.get("n_pair_rows")
    result["pair_words"] = metrics.get("pair_words")
    result["elapsed_sec"] = metrics.get("elapsed_sec")
    return result


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]

    results = []
    for opt, lr, corpus in ARMS:
        results.append(analyze_arm(base, opt, lr, corpus))

    summary = {
        "status": "OPTIMIZER_PILOT_METRICS_ONLY_SUMMARY",
        "arms": results,
        "analysis_scope": "Comparison of existing per-arm metrics; no new training or evaluation.",
        "data_availability": {
            "per_arm_training_events": "unavailable",
            "limitation": (
                "Training events in the combined log cannot be uniquely assigned to the "
                "simultaneously trained arms. Loss trajectories, gradient-norm ranges, "
                "trust-ratio dynamics, pair/nonpair loss dynamics and stability verdicts "
                "are therefore not estimated from that log."
            ),
            "missing_metrics": (
                "Missing files are reported as missing; absent fields remain null. "
                "Loss comparisons use only recorded loss_last values."
            ),
        },
    }

    # Loss comparison at 1M
    loss_by_setting = {}
    for r in results:
        if r.get("loss_last") is not None:
            key = f"{r['optimizer']}_lr{r['lr']}"
            if key not in loss_by_setting:
                loss_by_setting[key] = {}
            loss_by_setting[key][r["corpus"]] = r["loss_last"]
    summary["loss_at_1M"] = loss_by_setting

    # Compute data effects at 1M (Qwen - Official loss, lower is better)
    data_effects = {}
    for setting, losses in loss_by_setting.items():
        if "official" in losses and "qwen" in losses:
            data_effects[setting] = round(losses["qwen"] - losses["official"], 4)
    summary["data_effects_loss_1M"] = data_effects

    # Keep the historical summary and any previous metrics-only output intact.
    out_path = base / "data" / "optimizer_pilot_metrics_only_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
