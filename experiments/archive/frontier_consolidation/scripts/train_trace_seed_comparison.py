#!/usr/bin/env python3
"""research CPU comparison of compact_view_reinvest seed training traces.

Compares training logs, checkpoint exposure schedules, model metadata, and
training-trace summaries for seed43022 and seed43122 without evaluating models.
The goal is to see whether the weaker seed43122 fast surface is accompanied by a
plain optimization/loss/exposure difference or instead requires task-level
trajectory analysis.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Any, Dict, List

USER_ROOT = pathlib.Path.cwd()
OUT_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/data/train_trace_seed_comparison"
OUT_JSON = OUT_DIR / "train_trace_seed_comparison.json"
OUT_MD = USER_ROOT / "research/notes/frontier_consolidation/train_trace_seed_comparison.md"
RUNS = {
    "43022": USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022",
    "43122": USER_ROOT / "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122",
}


def jload(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def load_jsonl(p: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def numeric_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    return sorted(keys)


def extract_loss_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep rows with any ordinary scalar loss-like field.
    loss_keys = [k for k in numeric_keys(rows) if "loss" in k.lower() or k.lower() in {"step", "word_exposure", "epoch"}]
    out = []
    for r in rows:
        if any(("loss" in k.lower()) for k in r if isinstance(r.get(k), (int, float))):
            out.append({k: r.get(k) for k in loss_keys if k in r})
    return out


def summarize_loss(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_loss_rows": 0}
    loss_candidates = [k for k in numeric_keys(rows) if "loss" in k.lower()]
    chosen = None
    for k in ["loss", "train_loss", "mlm_loss", "batch_loss"]:
        if k in loss_candidates:
            chosen = k
            break
    if chosen is None and loss_candidates:
        chosen = loss_candidates[0]
    vals = [float(r[chosen]) for r in rows if chosen in r and math.isfinite(float(r[chosen]))] if chosen else []
    steps = [r.get("step") for r in rows if isinstance(r.get("step"), (int, float))]
    out = {"n_loss_rows": len(rows), "loss_key": chosen, "numeric_loss_keys": loss_candidates[:20]}
    if vals:
        out.update({
            "first": vals[0],
            "last": vals[-1],
            "mean": statistics.fmean(vals),
            "min": min(vals),
            "max": max(vals),
            "last10_mean": statistics.fmean(vals[-10:]),
            "last50_mean": statistics.fmean(vals[-50:]) if len(vals) >= 50 else statistics.fmean(vals),
        })
    if steps:
        out["step_min"] = min(steps); out["step_max"] = max(steps)
    return out


def checkpoint_schedule(metrics: dict[str, Any]) -> dict[str, Any]:
    ckpts = metrics.get("saved_checkpoints", [])
    exposure_by_name = {c["name"]: c.get("actual_cumulative_word_exposure") for c in ckpts}
    deviations = {}
    for c in ckpts:
        name = c.get("name", "")
        if not name.startswith("chck_"):
            continue
        target = c.get("target_word_exposure")
        actual = c.get("actual_cumulative_word_exposure")
        if target is not None and actual is not None:
            deviations[name] = actual - target
    return {
        "n_checkpoints": len(ckpts),
        "first_names": [c.get("name") for c in ckpts[:5]],
        "last_names": [c.get("name") for c in ckpts[-5:]],
        "deviation_min": min(deviations.values()) if deviations else None,
        "deviation_max": max(deviations.values()) if deviations else None,
        "deviation_mean": statistics.fmean(deviations.values()) if deviations else None,
        "selected_actual_exposures": {k: exposure_by_name.get(k) for k in ["chck_1M", "chck_10M", "chck_40M", "chck_100M"]},
    }


def read_run(seed: str, run: pathlib.Path) -> dict[str, Any]:
    metrics = jload(run / "scientific_metrics.json")
    train_cmd = jload(run / "train_command.json") if (run / "train_command.json").exists() else {}
    order = jload(run / "example_order_manifest.json") if (run / "example_order_manifest.json").exists() else {}
    training_rows = load_jsonl(run / "training_log.jsonl") if (run / "training_log.jsonl").exists() else []
    dynamics_rows = load_jsonl(run / "dynamics_traces.jsonl") if (run / "dynamics_traces.jsonl").exists() else []
    loss_rows = extract_loss_rows(training_rows)
    dyn_nonempty = [r for r in dynamics_rows if r.get("loss_by_freq_band") or r.get("accuracy_by_freq_band") or r.get("prediction_entropy_mean")]
    return {
        "seed": seed,
        "run_dir": str(run),
        "metadata": {k: metrics.get(k) for k in [
            "parameter_count", "vocab_size", "tokenizer_label", "word_exposure", "loss_first", "loss_last", "actual_training_steps", "masking_curriculum", "mask_prob_start", "mask_prob_end", "switch_frac", "hidden_size", "n_layer", "n_head", "seed", "extra_init_seed", "train_rng_seed", "data_source_type", "example_jsonl", "example_jsonl_label"
        ]},
        "train_command_selected": {k: train_cmd.get(k) for k in sorted(train_cmd.keys()) if k in {"cmd", "argv", "extra_init_seed", "train_rng_seed", "seed", "run_id", "script", "started_utc", "finished_utc"}},
        "example_order_manifest": order,
        "checkpoint_schedule": checkpoint_schedule(metrics),
        "training_log": {
            "n_rows": len(training_rows),
            "numeric_keys": numeric_keys(training_rows)[:40],
            "loss_summary": summarize_loss(loss_rows),
            "first_row": training_rows[0] if training_rows else None,
            "last_row": training_rows[-1] if training_rows else None,
        },
        "dynamics": {
            "n_rows": len(dynamics_rows),
            "n_nonempty_rows": len(dyn_nonempty),
            "nonempty_checkpoints": [r.get("checkpoint") for r in dyn_nonempty[:20]],
            "nonempty_rows_head": dyn_nonempty[:5],
            "nonempty_rows_tail": dyn_nonempty[-5:],
        },
        "stderr_size": (run / "train_stderr.log").stat().st_size if (run / "train_stderr.log").exists() else None,
        "stdout_size": (run / "train_stdout.log").stat().st_size if (run / "train_stdout.log").exists() else None,
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ma, mb = a["metadata"], b["metadata"]
    keys = sorted(set(ma) | set(mb))
    out["metadata_differences"] = {k: {"43022": ma.get(k), "43122": mb.get(k)} for k in keys if ma.get(k) != mb.get(k)}
    out["metadata_same_keys"] = [k for k in keys if ma.get(k) == mb.get(k)]
    la = a["training_log"]["loss_summary"]; lb = b["training_log"]["loss_summary"]
    diffs = {}
    for k in ["first", "last", "mean", "min", "max", "last10_mean", "last50_mean"]:
        if isinstance(la.get(k), (int, float)) and isinstance(lb.get(k), (int, float)):
            diffs[k] = lb[k] - la[k]
    out["loss_43122_minus_43022"] = diffs
    # Check selected checkpoint actual exposure differences.
    ca = a["checkpoint_schedule"]["selected_actual_exposures"]
    cb = b["checkpoint_schedule"]["selected_actual_exposures"]
    out["selected_exposure_actual_diff_43122_minus_43022"] = {k: cb.get(k) - ca.get(k) if isinstance(ca.get(k), int) and isinstance(cb.get(k), int) else None for k in sorted(set(ca) | set(cb))}
    # Dynamics trace comparisons where nonempty.
    da = {r.get("checkpoint"): r for r in a["dynamics"]["nonempty_rows_head"] + a["dynamics"]["nonempty_rows_tail"]}
    db = {r.get("checkpoint"): r for r in b["dynamics"]["nonempty_rows_head"] + b["dynamics"]["nonempty_rows_tail"]}
    common = sorted(set(da) & set(db))
    dyn = {}
    for ck in common:
        ra, rb = da[ck], db[ck]
        dyn[ck] = {
            "prediction_entropy_43122_minus_43022": (rb.get("prediction_entropy_mean") - ra.get("prediction_entropy_mean")) if isinstance(ra.get("prediction_entropy_mean"), (int, float)) and isinstance(rb.get("prediction_entropy_mean"), (int, float)) else None,
            "freq_band_loss_diff": {},
            "freq_band_acc_diff": {},
        }
        for band in sorted(set((ra.get("loss_by_freq_band") or {}).keys()) | set((rb.get("loss_by_freq_band") or {}).keys())):
            va = (ra.get("loss_by_freq_band") or {}).get(band, {}).get("mean")
            vb = (rb.get("loss_by_freq_band") or {}).get(band, {}).get("mean")
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                dyn[ck]["freq_band_loss_diff"][band] = vb - va
        for band in sorted(set((ra.get("accuracy_by_freq_band") or {}).keys()) | set((rb.get("accuracy_by_freq_band") or {}).keys())):
            va = (ra.get("accuracy_by_freq_band") or {}).get(band, {}).get("mean")
            vb = (rb.get("accuracy_by_freq_band") or {}).get(band, {}).get("mean")
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                dyn[ck]["freq_band_acc_diff"][band] = vb - va
    out["dynamics_selected_common"] = dyn
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {seed: read_run(seed, run) for seed, run in RUNS.items()}
    comp = compare(runs["43022"], runs["43122"])
    result = {
        "status": "TRAIN_TRACE_SEED_COMPARISON",
        "purpose": "CPU-only comparison of training metadata and logs for compact_view_reinvest seeds 43022 and 43122 before interpreting task temporal trajectories.",
        "runs": runs,
        "comparison": comp,
        "interpretation": {
            "same_exposure_schedule": all(v == 0 for v in comp["selected_exposure_actual_diff_43122_minus_43022"].values() if v is not None),
            "loss_last_diff_43122_minus_43022": comp["loss_43122_minus_43022"].get("last"),
            "metadata_differences": comp["metadata_differences"],
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research — compact_view_reinvest seed training-trace comparison\n\n"]
    lines.append("CPU/file-only comparison of seed43022 and seed43122 run metadata, checkpoint exposures, training logs, and sparse training dynamics. No model evaluation is performed here.\n\n")
    mdiff = comp["metadata_differences"]
    lines.append("## Main run-level differences\n")
    for k, v in mdiff.items():
        lines.append(f"- {k}: 43022={v['43022']} ; 43122={v['43122']}\n")
    lines.append("\n## Loss differences (43122 minus 43022)\n")
    for k, v in comp["loss_43122_minus_43022"].items():
        lines.append(f"- {k}: {v:+.6g}\n")
    lines.append("\n## Selected checkpoint exposure differences (43122 minus 43022)\n")
    for k, v in comp["selected_exposure_actual_diff_43122_minus_43022"].items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## Selected dynamics rows\n")
    if comp["dynamics_selected_common"]:
        for ck, v in comp["dynamics_selected_common"].items():
            lines.append(f"- {ck}: entropy diff {v['prediction_entropy_43122_minus_43022']}, loss-band diff {v['freq_band_loss_diff']}, acc-band diff {v['freq_band_acc_diff']}\n")
    else:
        lines.append("- No common nonempty dynamics rows beyond sparse trainer probes.\n")
    lines.append("\n## Reading\n")
    lines.append("The two runs share the same checkpoint exposure schedule and corpus path. The final MLM loss is higher for seed43122, but this scalar alone cannot explain which downstream abilities diverged. Use the sparse task-slice temporal probe to determine whether seed43122 is behind early or loses specific abilities later.\n")
    lines.append(f"\nMachine-readable output: `{OUT_JSON}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
