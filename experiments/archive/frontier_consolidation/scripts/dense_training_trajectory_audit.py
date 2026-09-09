#!/usr/bin/env python3
"""research: CPU audit and selected-evaluation plan for research dense DeBERTa runs.

This script does not evaluate models. It reads completed training artifacts and
research launch traces to determine what the runs actually vary, how their MLM
loss trajectories relate to the protected seed43022 reference, and which small
set of checkpoints should be evaluated first to test the fixed seed/scale
hypotheses before spending GPU time on dense full trajectories.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import pathlib
import re
import shlex
import statistics as stats
from typing import Any, Dict, Iterable, List

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
RUNS = STUDY / "training/runs"
TRACE = STUDY / "trace/steps"
OUT_DEFAULT = STUDY / "data/dense_training_trajectory_audit"

REF = {
    "label": "scale1p75_seed43022_reference",
    "run_dir": RUNS / "adapter128_scale1p75_h100M100M_seed43022_official_ladder",
    "launch_trace": None,
    "known_cheap7": {
        "chck_77M": 43.28214285714286,
        "chck_78M": 43.70214285714286,
        "chck_79M": 43.57857142857143,
        "chck_80M": 43.81214285714286,
        "chck_81M": 43.64928571428572,
        "chck_82M": 43.95944987645173,
        "chck_83M": 43.807857142857145,
        "chck_100M": 43.543159919261925,
    },
}
RUN_SPECS = [
    REF,
    {
        "label": "scale1p75_seed43122_dense",
        "run_dir": RUNS / "adapter128_scale1p75_seed43122_dense100M",
        "launch_trace": TRACE / "research/turn_0017/tool_0001.json",
        "expected_role": "same scale as protected reference, changed extra_init_seed and train_rng_seed",
    },
    {
        "label": "scale1p25_seed43022_dense",
        "run_dir": RUNS / "adapter128_scale1p25_seed43022_dense100M",
        "launch_trace": TRACE / "research/turn_0018/tool_0001.json",
        "expected_role": "same seed/mask stream as protected reference, changed adapter_scale",
    },
]

ARG_KEYS = [
    "--adapter_scale", "--seed", "--extra_init_seed", "--train_rng_seed", "--batch_size",
    "--seq_length", "--learning_rate", "--warmup_fraction", "--weight_decay",
    "--checkpoint_words", "--max_word_exposure", "--lr_total_steps",
    "--example_jsonl", "--tokenizer_path", "--adapter_bottleneck", "--masking_curriculum",
]

EVAL_PRIORITY = {
    "scale1p75_seed43122_dense": ["chck_70M", "chck_76M", "chck_78M", "chck_80M", "chck_82M", "chck_84M", "chck_86M", "chck_90M", "chck_100M"],
    "scale1p25_seed43022_dense": ["chck_70M", "chck_76M", "chck_80M", "chck_82M", "chck_86M", "chck_90M", "chck_94M", "chck_100M"],
}


def sha256_file(path: pathlib.Path, max_bytes: int | None = None) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def parse_launch(trace_path: pathlib.Path | None) -> dict[str, Any]:
    if trace_path is None or not trace_path.exists():
        return {"available": False}
    raw = json.loads(trace_path.read_text(encoding="utf-8"))
    args = raw.get("tool_call", {}).get("arguments", {}).get("arguments", {})
    cmd = args.get("command", "")
    toks = shlex.split(cmd)
    parsed: dict[str, str] = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                parsed[t] = toks[i + 1]
                i += 2
            else:
                parsed[t] = "true"
                i += 1
        else:
            i += 1
    return {
        "available": True,
        "trace_path": str(trace_path),
        "command": cmd,
        "selected_args": {k: parsed.get(k) for k in ARG_KEYS if k in parsed},
        "all_args": parsed,
    }


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_dirs(run_dir: pathlib.Path) -> list[dict[str, Any]]:
    hf = run_dir / "hf_model"
    out = []
    if not hf.exists():
        return out
    for p in hf.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"chck_(\d+)M", p.name)
        if not m:
            continue
        out.append({
            "endpoint": p.name,
            "words_nominal": int(m.group(1)) * 1_000_000,
            "path": str(p),
            "has_model": (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists(),
            "config_sha256": sha256_file(p / "config.json"),
            "model_sha256": sha256_file(p / "model.safetensors"),
        })
    return sorted(out, key=lambda r: r["words_nominal"])


def nearest_by_words(logs: list[dict[str, Any]], target: int) -> dict[str, Any] | None:
    if not logs:
        return None
    return min(logs, key=lambda r: abs(float(r.get("cumulative_word_exposure", 0)) - target))


def rolling_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    out = []
    s = 0.0
    q: list[float] = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def summarize_run(spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = pathlib.Path(spec["run_dir"])
    metrics = read_json(run_dir / "scientific_metrics.json")
    logs = read_jsonl(run_dir / "training_log.jsonl")
    dyn = read_jsonl(run_dir / "dynamics_traces.jsonl")
    ckpts = checkpoint_dirs(run_dir)
    launch = parse_launch(spec.get("launch_trace"))
    losses = [float(r.get("loss")) for r in logs if r.get("loss") is not None and math.isfinite(float(r.get("loss")))]
    roll50 = rolling_mean(losses, 50)
    checkpoints_from_metrics = metrics.get("saved_checkpoints", []) if isinstance(metrics.get("saved_checkpoints"), list) else []
    late_targets = [70, 76, 78, 80, 82, 84, 86, 90, 94, 100]
    late_loss = {}
    for m in late_targets:
        row = nearest_by_words(logs, m * 1_000_000)
        if row:
            idx = int(row.get("step", 1)) - 1
            late_loss[f"{m}M"] = {
                "step": row.get("step"),
                "words": row.get("cumulative_word_exposure"),
                "loss": row.get("loss"),
                "rolling50_loss": roll50[idx] if 0 <= idx < len(roll50) else None,
                "lr": row.get("lr"),
            }
    return {
        "label": spec["label"],
        "expected_role": spec.get("expected_role"),
        "run_dir": str(run_dir),
        "launch": launch,
        "metrics_core": {k: metrics.get(k) for k in [
            "variant", "backend", "model_family", "parameter_count", "vocab_size", "tokenizer_label",
            "word_exposure", "loss_first", "loss_last", "actual_training_steps", "hidden_size", "n_layer",
            "n_head", "seed", "seq_length", "example_jsonl_label", "example_jsonl"
        ]},
        "adapter_scale_from_launch": (launch.get("selected_args", {}) or {}).get("--adapter_scale"),
        "extra_init_seed_from_launch": (launch.get("selected_args", {}) or {}).get("--extra_init_seed"),
        "train_rng_seed_from_launch": (launch.get("selected_args", {}) or {}).get("--train_rng_seed"),
        "n_training_log_rows": len(logs),
        "n_dynamics_rows": len(dyn),
        "n_checkpoint_dirs": len(ckpts),
        "checkpoint_endpoints": [c["endpoint"] for c in ckpts],
        "checkpoint_model_hashes_selected": {c["endpoint"]: c.get("model_sha256") for c in ckpts if c["endpoint"] in {"chck_20M", "chck_80M", "chck_82M", "chck_100M"}},
        "checkpoint_metrics_count": len(checkpoints_from_metrics),
        "final_log_row": logs[-1] if logs else None,
        "loss_summary": {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "min_raw": min(losses) if losses else None,
            "min_raw_step": logs[losses.index(min(losses))].get("step") if losses else None,
            "min_roll50": min(roll50) if roll50 else None,
            "min_roll50_step": logs[roll50.index(min(roll50))].get("step") if roll50 else None,
            "late_loss_at_nominal_words": late_loss,
        },
    }


def compare_logs(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    n = min(len(a), len(b))
    same_words = 0
    same_masked = 0
    same_loss = 0
    max_loss_abs = 0.0
    diffs = []
    for i in range(n):
        ra, rb = a[i], b[i]
        if ra.get("batch_words") == rb.get("batch_words") and ra.get("cumulative_word_exposure") == rb.get("cumulative_word_exposure"):
            same_words += 1
        if ra.get("masked_tokens") == rb.get("masked_tokens") and ra.get("effective_mask_rate") == rb.get("effective_mask_rate"):
            same_masked += 1
        la = float(ra.get("loss")); lb = float(rb.get("loss"))
        d = la - lb
        diffs.append(d)
        if la == lb:
            same_loss += 1
        max_loss_abs = max(max_loss_abs, abs(d))
    return {
        "n_compared": n,
        "same_batch_and_cumulative_words_rows": same_words,
        "same_mask_rows": same_masked,
        "same_loss_rows": same_loss,
        "mean_loss_a_minus_b": stats.mean(diffs) if diffs else None,
        "last_loss_a_minus_b": diffs[-1] if diffs else None,
        "max_abs_loss_diff": max_loss_abs if diffs else None,
        "first_nonidentical_mask_row": next((i + 1 for i in range(n) if a[i].get("masked_tokens") != b[i].get("masked_tokens") or a[i].get("effective_mask_rate") != b[i].get("effective_mask_rate")), None),
        "first_nonidentical_loss_row": next((i + 1 for i in range(n) if float(a[i].get("loss")) != float(b[i].get("loss"))), None),
    }


def make_eval_plan(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # Minimum reliable plan: evaluate around known reference peak/decline first, then expand only if ambiguous.
    commands = []
    for label in ["scale1p75_seed43122_dense", "scale1p25_seed43022_dense"]:
        s = summaries[label]
        run_dir = pathlib.Path(s["run_dir"])
        ckpt_set = set(s.get("checkpoint_endpoints", []))
        endpoints = [e for e in EVAL_PRIORITY[label] if e in ckpt_set]
        # Use the existing batch evaluator over the late span, but also include exact selected endpoint list for a future wrapper.
        min_words = min(int(e.split("_")[1][:-1]) for e in endpoints) * 1_000_000
        max_words = max(int(e.split("_")[1][:-1]) for e in endpoints) * 1_000_000
        out_dir = STUDY / "data" / f"selected_trajectory_eval_{label}"
        commands.append({
            "label": label,
            "reason": "Tests seed/scale prediction at low evaluation cost before dense full-trajectory evaluation.",
            "selected_endpoints_first": endpoints,
            "batch_evaluator_range_command": (
                f"PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=<GPU> python -B "
                f"experiments/archive/frontier_consolidation/scripts/batch_trajectory_eval.py "
                f"--run-dir {run_dir} --gpu <GPU> --min-words {min_words} --max-words {max_words} "
                f"--out-dir {out_dir} --label {label}"
            ),
            "note": "The current batch evaluator evaluates every 2M checkpoint in the range; if GPU time is tight, build an endpoint-list wrapper to evaluate only selected_endpoints_first.",
        })
    return {
        "purpose": "Evaluate just enough official-compatible cheap7 trajectory to decide whether research seed/scale hypotheses survive.",
        "commands": commands,
        "interpretation": {
            "scale1p75_seed43122_dense": "Support seed robustness only if best/near-best lies near 80-84M with late decline and similar family movement; a far-away or one-column peak weakens it.",
            "scale1p25_seed43022_dense": "Support adapter-scale timing only if lower scale shifts the near-best band later and/or widens it without turning into a different capability tradeoff.",
        }
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {spec["label"]: summarize_run(spec) for spec in RUN_SPECS}
    logs = {spec["label"]: read_jsonl(pathlib.Path(spec["run_dir"]) / "training_log.jsonl") for spec in RUN_SPECS}
    comparisons = {
        "scale1p25_seed43022_vs_reference_scale1p75_seed43022": compare_logs(logs["scale1p25_seed43022_dense"], logs["scale1p75_seed43022_reference"]),
        "scale1p75_seed43122_vs_reference_scale1p75_seed43022": compare_logs(logs["scale1p75_seed43122_dense"], logs["scale1p75_seed43022_reference"]),
        "scale1p75_seed43122_vs_scale1p25_seed43022": compare_logs(logs["scale1p75_seed43122_dense"], logs["scale1p25_seed43022_dense"]),
    }
    plan = make_eval_plan(summaries)

    result = {
        "status": "DENSE_TRAINING_TRAJECTORY_AUDIT",
        "scientific_purpose": "Separate what the completed research trainings actually varied from what must still be learned by official-compatible checkpoint evaluation.",
        "summaries": summaries,
        "log_stream_comparisons": comparisons,
        "selected_gpu_eval_plan": plan,
        "important_correction": "The scale1.75 seed43122 run changes both extra_init_seed and train_rng_seed relative to protected seed43022; the scale1.25 run preserves seed/train_rng relative to protected seed43022 while changing adapter scale. Therefore scale1.75 seed43122 is a joint init+mask-stream robustness contrast, not pure initialization only.",
    }
    out_json = args.out_dir / "dense_training_trajectory_audit.json"
    out_md = args.out_dir / "dense_training_trajectory_audit.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = []
    md.append("# research dense training trajectory audit\n\n")
    md.append("This CPU/file-only audit inspects the completed research DeBERTa dense runs while the causal GPT arms occupy both H100s. It does not score checkpoints.\n\n")
    md.append("## What the two runs actually vary\n\n")
    for label in ["scale1p75_seed43122_dense", "scale1p25_seed43022_dense"]:
        s = summaries[label]
        la = s.get("launch", {}).get("selected_args", {})
        md.append(f"### {label}\n")
        md.append(f"- adapter_scale={la.get('--adapter_scale')} extra_init_seed={la.get('--extra_init_seed')} train_rng_seed={la.get('--train_rng_seed')}\n")
        md.append(f"- checkpoints={s['n_checkpoint_dirs']} final_loss={s['metrics_core'].get('loss_last')} steps={s['metrics_core'].get('actual_training_steps')}\n")
        md.append(f"- selected hashes: {json.dumps(s['checkpoint_model_hashes_selected'], ensure_ascii=False)}\n\n")
    md.append("## Log-stream comparisons\n\n")
    for k, v in comparisons.items():
        md.append(f"- {k}: same words rows {v['same_batch_and_cumulative_words_rows']}/{v['n_compared']}, same mask rows {v['same_mask_rows']}/{v['n_compared']}, same loss rows {v['same_loss_rows']}/{v['n_compared']}, last loss diff {v['last_loss_a_minus_b']}\n")
    md.append("\nThe key correction is that the scale1.75 cross-seed run changes both initialization and train RNG relative to the protected seed43022 reference, while scale1.25 preserves the protected seed stream and changes only adapter amplitude.\n\n")
    md.append("## Loss trajectory snapshots\n\n")
    for label, s in summaries.items():
        md.append(f"### {label}\n")
        ll = s["loss_summary"].get("late_loss_at_nominal_words", {})
        for m in ["70M", "76M", "80M", "82M", "86M", "90M", "94M", "100M"]:
            if m in ll:
                r = ll[m]
                md.append(f"- {m}: loss={r['loss']:.6f}, roll50={r['rolling50_loss']:.6f}, lr={r['lr']:.3g}, words={r['words']}\n")
        md.append("\n")
    md.append("## Selected GPU scoring plan\n\n")
    for c in plan["commands"]:
        md.append(f"### {c['label']}\n")
        md.append(f"- first endpoints: {', '.join(c['selected_endpoints_first'])}\n")
        md.append(f"- command: `{c['batch_evaluator_range_command']}`\n")
        md.append(f"- note: {c['note']}\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
