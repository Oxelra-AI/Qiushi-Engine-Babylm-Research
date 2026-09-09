#!/usr/bin/env python3
"""research audit of dual-view 20M training runs.

Reads completed run directories and verifies that the scientific comparison is what
it is supposed to be.  In particular aligned and shuffled arms should match in
update count, charged/main/aux exposure, LR sequence, target counts, pair-unit
counts, and checkpoint availability; any score difference should not be explained
by accounting drift.  The mlm_only reference has no auxiliary exposure by design
but must share schedule/initialization/model shape and the same total charged cap.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()

KEYS_EQUAL_ALIGNED_SHUFFLED = [
    "updates", "total_main_word_exposure", "total_aux_word_exposure",
    "total_charged_words", "schedule_total", "total_params", "adapter_params",
    "aux_micro_batch_size", "aux_loss_batches",
]
LOG_KEYS_EQUAL_ALIGNED_SHUFFLED = [
    "batch_words", "aux_words", "cumulative_main_words", "cumulative_aux_words",
    "cumulative_charged_words", "masked_tokens", "aux_targets", "aux_units",
    "aux_conditioned_views", "aux_free_views", "lr",
]
CONFIG_KEYS_ALL = [
    "seed", "extra_init_seed", "train_rng_seed", "batch_size", "seq_length",
    "aux_max_length", "learning_rate", "warmup_fraction", "weight_decay",
    "mask_prob", "lr_total_steps", "adapter_bottleneck", "adapter_scale",
]


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_log(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def summarize_run(label: str, run_dir: pathlib.Path) -> dict[str, Any]:
    sm_path = run_dir / "scientific_metrics.json"
    cfg_path = run_dir / "train_config.json"
    log_path = run_dir / "training_log.jsonl"
    model_root = run_dir / "hf_model"
    metrics = read_json(sm_path) if sm_path.exists() else None
    config = read_json(cfg_path) if cfg_path.exists() else None
    log = read_log(log_path)
    ckpts = sorted([p.name for p in model_root.iterdir() if p.is_dir()]) if model_root.exists() else []
    last = log[-1] if log else None
    first = log[0] if log else None
    aux_targets = [int(x.get("aux_targets", 0)) for x in log]
    masked_tokens = [int(x.get("masked_tokens", 0)) for x in log]
    return {
        "label": label,
        "run_dir": rel(run_dir),
        "exists": run_dir.exists(),
        "scientific_metrics_exists": sm_path.exists(),
        "train_config_exists": cfg_path.exists(),
        "training_log_exists": log_path.exists(),
        "checkpoint_dirs": ckpts,
        "has_chck_20M": "chck_20M" in ckpts,
        "has_final": "final" in ckpts,
        "metrics": metrics,
        "config": config,
        "log_n": len(log),
        "log_first": first,
        "log_last": last,
        "log_sums": {
            "batch_words": sum(int(x.get("batch_words", 0)) for x in log),
            "aux_words": sum(int(x.get("aux_words", 0)) for x in log),
            "masked_tokens": sum(masked_tokens),
            "aux_targets": sum(aux_targets),
            "aux_units": sum(int(x.get("aux_units", 0)) for x in log),
        },
        "log_stats": {
            "mean_loss": statistics.mean(float(x.get("loss")) for x in log) if log else None,
            "mean_aux_loss_nonzero": statistics.mean(float(x.get("aux_loss")) for x in log if float(x.get("aux_loss", 0.0)) > 0.0) if any(float(x.get("aux_loss", 0.0)) > 0.0 for x in log) else 0.0,
            "mean_masked_tokens": statistics.mean(masked_tokens) if masked_tokens else None,
            "mean_aux_targets": statistics.mean(aux_targets) if aux_targets else None,
            "nonzero_aux_batches": sum(1 for x in aux_targets if x > 0),
        },
    }


def same_value(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
        except Exception:
            return False
    return a == b


def compare_metric_keys(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    ma = a.get("metrics") or {}
    mb = b.get("metrics") or {}
    return {k: {"a": ma.get(k), "b": mb.get(k), "equal": same_value(ma.get(k), mb.get(k))} for k in keys}


def compare_config_keys(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    ca = a.get("config") or {}
    cb = b.get("config") or {}
    return {k: {"a": ca.get(k), "b": cb.get(k), "equal": same_value(ca.get(k), cb.get(k))} for k in keys}


def compare_logs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    la = read_log(pathlib.Path(a["run_dir"]) / "training_log.jsonl")
    lb = read_log(pathlib.Path(b["run_dir"]) / "training_log.jsonl")
    n = min(len(la), len(lb))
    diffs = {k: 0 for k in LOG_KEYS_EQUAL_ALIGNED_SHUFFLED}
    first_diff_examples = {}
    for i in range(n):
        for k in LOG_KEYS_EQUAL_ALIGNED_SHUFFLED:
            if not same_value(la[i].get(k), lb[i].get(k), tol=1e-10):
                diffs[k] += 1
                first_diff_examples.setdefault(k, {"index": i, "a": la[i].get(k), "b": lb[i].get(k)})
    return {
        "n_a": len(la),
        "n_b": len(lb),
        "n_common": n,
        "length_equal": len(la) == len(lb),
        "diff_counts_by_key": diffs,
        "first_diff_examples": first_diff_examples,
        "all_accounting_equal": len(la) == len(lb) and all(v == 0 for v in diffs.values()),
    }


def parse_arm(spec: str) -> tuple[str, pathlib.Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        p = pathlib.Path(spec)
        label, path = p.name, spec
    return label, pathlib.Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="label=run_dir; may repeat")
    ap.add_argument("--label", default="dualview_training_panel")
    ap.add_argument("--out-dir", default="experiments/archive/frontier_consolidation/data/dualview_training_audit")
    args = ap.parse_args()

    arms = [parse_arm(x) for x in args.arm]
    runs = {lab: summarize_run(lab, p) for lab, p in arms}
    comparisons: dict[str, Any] = {}
    labels = list(runs.keys())
    if "aligned" in runs and "shuffled" in runs:
        comparisons["aligned_vs_shuffled"] = {
            "metric_equality": compare_metric_keys(runs["aligned"], runs["shuffled"], KEYS_EQUAL_ALIGNED_SHUFFLED),
            "config_equality": compare_config_keys(runs["aligned"], runs["shuffled"], CONFIG_KEYS_ALL + ["aux_lambda", "aux_pair_shuffle_seed"]),
            "log_equality": compare_logs(runs["aligned"], runs["shuffled"]),
        }
    if "mlm_only" in runs:
        for lab in labels:
            if lab == "mlm_only":
                continue
            comparisons[f"{lab}_vs_mlm_only_config"] = {
                "config_equality_common_scaffold": compare_config_keys(runs[lab], runs["mlm_only"], CONFIG_KEYS_ALL),
                "charged_words": {
                    lab: (runs[lab].get("metrics") or {}).get("total_charged_words"),
                    "mlm_only": (runs["mlm_only"].get("metrics") or {}).get("total_charged_words"),
                },
                "main_words": {
                    lab: (runs[lab].get("metrics") or {}).get("total_main_word_exposure"),
                    "mlm_only": (runs["mlm_only"].get("metrics") or {}).get("total_main_word_exposure"),
                },
                "aux_words": {
                    lab: (runs[lab].get("metrics") or {}).get("total_aux_word_exposure"),
                    "mlm_only": (runs["mlm_only"].get("metrics") or {}).get("total_aux_word_exposure"),
                },
            }
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "status": "DUALVIEW_TRAINING_AUDIT",
        "label": args.label,
        "runs": runs,
        "comparisons": comparisons,
    }
    out_json = out_dir / f"{args.label}.json"
    out_md = out_dir / f"{args.label}.md"
    out["out_json"] = rel(out_json)
    out["out_md"] = rel(out_md)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# research dual-view training audit — {args.label}", ""]
    for lab, r in runs.items():
        m = r.get("metrics") or {}
        lines += [f"## {lab}", f"Run: `{r['run_dir']}`", f"metrics exists: `{r['scientific_metrics_exists']}`; log lines: `{r['log_n']}`; chck_20M: `{r['has_chck_20M']}`; final: `{r['has_final']}`", f"charged/main/aux words: `{m.get('total_charged_words')}` / `{m.get('total_main_word_exposure')}` / `{m.get('total_aux_word_exposure')}`; updates `{m.get('updates')}`; final loss `{m.get('final_loss')}`", ""]
    for name, c in comparisons.items():
        lines += [f"## {name}", "```json", json.dumps(c, indent=2, ensure_ascii=False)[:6000], "```", ""]
    lines.append(f"JSON: `{rel(out_json)}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    status_bits = {}
    if "aligned_vs_shuffled" in comparisons:
        avs = comparisons["aligned_vs_shuffled"]
        status_bits["aligned_shuffled_log_all_accounting_equal"] = avs["log_equality"]["all_accounting_equal"]
        status_bits["aligned_shuffled_metric_equal_keys"] = {k:v["equal"] for k,v in avs["metric_equality"].items()}
    print(json.dumps({"status": out["status"], "label": args.label, **status_bits, "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
