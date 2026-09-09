#!/usr/bin/env python3
"""research: verify completed RoBERTa compact-vs-repeat training pair integrity.

Reads completed training logs/metrics only.  No training/evaluation/upload/submission.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_COMPACT = WS / "training/runs/roberta_compact_reinvest_100M_seed43022"
DEFAULT_REPEAT = WS / "training/runs/roberta_repeat_compact_reinvest_100M_seed43022"
DEFAULT_OUT = WS / "data/roberta_training_pair_integrity"

KEYS_RECIPE = [
    "status", "variant", "backend", "model_family", "parameter_count", "vocab_size",
    "tokenizer_label", "tokenizer_path", "word_exposure", "actual_training_steps", "optimizer",
    "learning_rate", "weight_decay", "warmup_fraction", "lr_total_steps", "masking_curriculum",
    "mask_prob_start", "mask_prob_end", "seed", "extra_init_seed", "train_rng_seed",
    "seq_length", "max_seq_length", "batch_size", "hidden_size", "n_layer", "n_head",
    "ffn_mult", "gradient_checkpointing", "data_source_type",
]
NUMERIC_LOG_FIELDS = ["loss", "lr", "batch_words", "cumulative_word_exposure", "masked_tokens", "candidate_tokens", "effective_mask_rate"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def read_log(p: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # DeBERTa logs carry event="train"; this RoBERTa trainer writes bare
            # training rows.  Accept both schemas and reject non-training status rows.
            is_train_event = obj.get("event") == "train"
            is_bare_train_row = "event" not in obj and {"step", "loss", "batch_words", "cumulative_word_exposure"}.issubset(obj)
            if is_train_event or is_bare_train_row:
                rows.append(obj)
    return rows


def normalize_path_value(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v
    root_s = str(ROOT)
    if s.startswith(root_s + "/"):
        return s[len(root_s) + 1:]
    try:
        p = Path(s)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            try:
                return str(p.resolve().relative_to(ROOT))
            except Exception:
                return str(p.resolve())
    except Exception:
        pass
    return s.lstrip("./")


def close(a: Any, b: Any, tol: float = 1e-9, field: str | None = None) -> bool:
    if field and field.endswith("path"):
        return normalize_path_value(a) == normalize_path_value(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def stats(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(xs), "mean": mean(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def checkpoint_names(metrics: dict[str, Any]) -> list[str]:
    return [c.get("name") for c in metrics.get("saved_checkpoints", [])]


def checkpoint_exposures(metrics: dict[str, Any]) -> dict[str, int]:
    return {c.get("name"): int(c.get("actual_cumulative_word_exposure")) for c in metrics.get("saved_checkpoints", [])}


def compare(compact: Path, repeat: Path) -> dict[str, Any]:
    cm = read_json(compact / "scientific_metrics.json")
    rm = read_json(repeat / "scientific_metrics.json")
    clog = read_log(compact / "training_log.jsonl")
    rlog = read_log(repeat / "training_log.jsonl")
    recipe_mismatches = []
    for k in KEYS_RECIPE:
        if not close(cm.get(k), rm.get(k), field=k):
            recipe_mismatches.append({"field": k, "compact": cm.get(k), "repeat": rm.get(k)})
    # example_jsonl and source_words_consumed are expected to differ only in the changed generated block name.
    ccks = checkpoint_exposures(cm)
    rcks = checkpoint_exposures(rm)
    checkpoint_mismatches = {k: {"compact": ccks.get(k), "repeat": rcks.get(k)} for k in sorted(set(ccks) | set(rcks)) if ccks.get(k) != rcks.get(k)}
    paired = []
    for cr, rr in zip(clog, rlog):
        row = {"step": cr.get("step"), "repeat_step": rr.get("step")}
        for f in NUMERIC_LOG_FIELDS:
            cv = cr.get(f); rv = rr.get(f)
            row[f"compact_{f}"] = cv
            row[f"repeat_{f}"] = rv
            if isinstance(cv, (int, float)) and isinstance(rv, (int, float)):
                row[f"delta_{f}_compact_minus_repeat"] = float(cv) - float(rv)
        paired.append(row)
    deltas = {}
    for f in NUMERIC_LOG_FIELDS:
        key = f"delta_{f}_compact_minus_repeat"
        vals = [float(r[key]) for r in paired if key in r]
        deltas[f] = stats(vals)
    windows = {}
    for name, lo, hi in [
        ("early_1_500", 1, 500),
        ("mid_501_1500", 501, 1500),
        ("late_1501_2529", 1501, 2529),
        ("last_250", 2280, 2529),
        ("last_50", 2480, 2529),
    ]:
        subset = [r for r in paired if isinstance(r.get("step"), int) and lo <= int(r["step"]) <= hi]
        windows[name] = {
            "n": len(subset),
            "loss_delta_compact_minus_repeat": mean([float(r["delta_loss_compact_minus_repeat"]) for r in subset if "delta_loss_compact_minus_repeat" in r]),
            "masked_tokens_delta_compact_minus_repeat": mean([float(r["delta_masked_tokens_compact_minus_repeat"]) for r in subset if "delta_masked_tokens_compact_minus_repeat" in r]),
            "candidate_tokens_delta_compact_minus_repeat": mean([float(r["delta_candidate_tokens_compact_minus_repeat"]) for r in subset if "delta_candidate_tokens_compact_minus_repeat" in r]),
            "effective_mask_rate_delta_compact_minus_repeat": mean([float(r["delta_effective_mask_rate_compact_minus_repeat"]) for r in subset if "delta_effective_mask_rate_compact_minus_repeat" in r]),
        }
    exact_batch_word_match = len(clog) == len(rlog) and all(cr.get("batch_words") == rr.get("batch_words") for cr, rr in zip(clog, rlog))
    exact_cumulative_word_match = len(clog) == len(rlog) and all(cr.get("cumulative_word_exposure") == rr.get("cumulative_word_exposure") for cr, rr in zip(clog, rlog))
    same_lr = len(clog) == len(rlog) and all(close(cr.get("lr"), rr.get("lr"), tol=1e-15) for cr, rr in zip(clog, rlog))
    expected_corpus_identity_differences = {
        "compact_example_jsonl": normalize_path_value(cm.get("example_jsonl")),
        "repeat_example_jsonl": normalize_path_value(rm.get("example_jsonl")),
        "compact_example_jsonl_label": cm.get("example_jsonl_label"),
        "repeat_example_jsonl_label": rm.get("example_jsonl_label"),
    }
    payload = {
        "status": "ROBERTA_TRAINING_PAIR_INTEGRITY_REPAIRED",
        "created_utc": now(),
        "meaning": "Completed RoBERTa compact-vs-repeat training-pair integrity and training-surface comparison. This is not downstream evaluation evidence.",
        "compact_run": str(compact),
        "repeat_run": str(repeat),
        "mechanical_integrity": {
            "compact_metrics_exists": (compact / "scientific_metrics.json").exists(),
            "repeat_metrics_exists": (repeat / "scientific_metrics.json").exists(),
            "compact_training_log_rows": len(clog),
            "repeat_training_log_rows": len(rlog),
            "recipe_mismatches_excluding_expected_corpus_identity": recipe_mismatches,
            "expected_corpus_identity_differences": expected_corpus_identity_differences,
            "checkpoint_names_match": checkpoint_names(cm) == checkpoint_names(rm),
            "checkpoint_exposure_mismatches": checkpoint_mismatches,
            "exact_batch_word_match_all_steps": exact_batch_word_match,
            "exact_cumulative_word_match_all_steps": exact_cumulative_word_match,
            "same_lr_all_steps": same_lr,
            "compact_word_exposure": cm.get("word_exposure"),
            "repeat_word_exposure": rm.get("word_exposure"),
            "compact_loss_first_last": [cm.get("loss_first"), cm.get("loss_last")],
            "repeat_loss_first_last": [rm.get("loss_first"), rm.get("loss_last")],
            "compact_all_checkpoints_present": all((compact / "hf_model" / ck).exists() for ck in checkpoint_names(cm)),
            "repeat_all_checkpoints_present": all((repeat / "hf_model" / ck).exists() for ck in checkpoint_names(rm)),
        },
        "training_surface": {
            "deltas_compact_minus_repeat": deltas,
            "windows": windows,
            "total_masked_tokens_compact": sum(int(r.get("masked_tokens", 0)) for r in clog),
            "total_masked_tokens_repeat": sum(int(r.get("masked_tokens", 0)) for r in rlog),
            "total_candidate_tokens_compact": sum(int(r.get("candidate_tokens", 0)) for r in clog),
            "total_candidate_tokens_repeat": sum(int(r.get("candidate_tokens", 0)) for r in rlog),
        },
        "scientific_reading": "The two RoBERTa arms are mechanically matched on exposure, checkpoint cadence, LR, seeds, tokenizer, architecture, and batch words. Loss/masked-token differences are realized consequences of compact vs first-N-repeat text, not recipe drift; selected official-compatible scores and the late local bridge remain necessary before judging transfer.",
    }
    return payload


def make_md(payload: dict[str, Any]) -> str:
    mi = payload["mechanical_integrity"]
    ts = payload["training_surface"]
    lines = []
    lines.append("# research RoBERTa training-pair integrity")
    lines.append("")
    lines.append(f"Created: `{payload['created_utc']}`")
    lines.append("")
    lines.append("This reads completed training logs only; it does not run evaluation or start any training.")
    lines.append("")
    lines.append("## Mechanical pair status")
    lines.append(f"- Compact log rows `{mi['compact_training_log_rows']}`, repeat log rows `{mi['repeat_training_log_rows']}`.")
    lines.append(f"- Exact batch-word match all steps: `{mi['exact_batch_word_match_all_steps']}`; exact cumulative exposure match: `{mi['exact_cumulative_word_match_all_steps']}`; LR match: `{mi['same_lr_all_steps']}`.")
    lines.append(f"- Checkpoint names match: `{mi['checkpoint_names_match']}`; checkpoint exposure mismatches: `{mi['checkpoint_exposure_mismatches']}`.")
    lines.append(f"- Compact exposure `{mi['compact_word_exposure']}`, repeat exposure `{mi['repeat_word_exposure']}`; all checkpoints present compact/repeat `{mi['compact_all_checkpoints_present']}`/`{mi['repeat_all_checkpoints_present']}`.")
    lines.append(f"- Recipe mismatches excluding expected corpus identity: `{mi['recipe_mismatches_excluding_expected_corpus_identity']}`.")
    lines.append("")
    lines.append("## Training surface (compact minus repeat)")
    lines.append(f"- Endpoint losses: compact {mi['compact_loss_first_last'][1]}, repeat {mi['repeat_loss_first_last'][1]}, delta {float(mi['compact_loss_first_last'][1]) - float(mi['repeat_loss_first_last'][1]):+.6f}.")
    lines.append(f"- Total masked tokens: compact {ts['total_masked_tokens_compact']}, repeat {ts['total_masked_tokens_repeat']}, delta {ts['total_masked_tokens_compact'] - ts['total_masked_tokens_repeat']:+d}.")
    lines.append(f"- Total candidate tokens: compact {ts['total_candidate_tokens_compact']}, repeat {ts['total_candidate_tokens_repeat']}, delta {ts['total_candidate_tokens_compact'] - ts['total_candidate_tokens_repeat']:+d}.")
    lines.append("")
    lines.append("| window | n | mean loss Δ | mean masked Δ | mean candidate Δ | mean mask-rate Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, w in ts["windows"].items():
        def f(x):
            return "NA" if x is None else f"{float(x):.6f}"
        lines.append(f"| {name} | {w['n']} | {f(w['loss_delta_compact_minus_repeat'])} | {f(w['masked_tokens_delta_compact_minus_repeat'])} | {f(w['candidate_tokens_delta_compact_minus_repeat'])} | {f(w['effective_mask_rate_delta_compact_minus_repeat'])} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append(payload["scientific_reading"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact-run", type=Path, default=DEFAULT_COMPACT)
    ap.add_argument("--repeat-run", type=Path, default=DEFAULT_REPEAT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    payload = compare(args.compact_run, args.repeat_run)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jp = args.out_dir / "roberta_training_pair_integrity.json"
    mp = args.out_dir / "roberta_training_pair_integrity.md"
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    mp.write_text(make_md(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "json": str(jp),
        "md": str(mp),
          "mechanically_matched": (
              not payload["mechanical_integrity"]["recipe_mismatches_excluding_expected_corpus_identity"]
              and not payload["mechanical_integrity"]["checkpoint_exposure_mismatches"]
              and payload["mechanical_integrity"]["exact_batch_word_match_all_steps"]
              and payload["mechanical_integrity"]["exact_cumulative_word_match_all_steps"]
              and payload["mechanical_integrity"]["same_lr_all_steps"]
        ),
        "endpoint_loss_delta_compact_minus_repeat": payload["mechanical_integrity"]["compact_loss_first_last"][1] - payload["mechanical_integrity"]["repeat_loss_first_last"][1],
        "total_masked_token_delta": payload["training_surface"]["total_masked_tokens_compact"] - payload["training_surface"]["total_masked_tokens_repeat"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
