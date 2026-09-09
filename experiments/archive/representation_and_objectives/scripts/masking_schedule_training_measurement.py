#!/usr/bin/env python3
"""research: measure the fixed-data WWM-to-token training trace before downstream eval.

This is CPU-only.  It checks whether the completed WWM->token run is truly the
same training process as the COMPACT_EXPERIENCE clean-Qwen fixed-WWM run until the 70M-word
switch, then identifies the smallest set of post-switch checkpoints that need
GPU downstream evaluation.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
BASE_ROOT = pathlib.Path("experiments/archive/compact_experience")
TARGET_RUN = ROOT / "training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022"
BASE_RUN = BASE_ROOT / "training/runs/qwen_clean_aligned_16k_seed43022"
OUT_DIR = ROOT / "data/wwm_to_token_training_measurement"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/wwm_to_token_training_measurement.md')

COMPARE_FIELDS = [
    "step",
    "loss",
    "lr",
    "batch_words",
    "cumulative_word_exposure",
    "seq_len",
    "masked_tokens",
    "effective_mask_rate",
    "mask_mode",
    "mask_prob_nominal",
]


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nearly_same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-10)
        except Exception:
            return False
    return a == b


def summarize_loss(rows: list[dict[str, Any]], lo_step: int, hi_step: int) -> dict[str, Any]:
    vals = [float(r["loss"]) for r in rows if lo_step <= int(r["step"]) <= hi_step and r.get("loss") is not None]
    return {
        "step_window": [lo_step, hi_step],
        "n": len(vals),
        "mean": statistics.fmean(vals) if vals else None,
        "first": vals[0] if vals else None,
        "last": vals[-1] if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def mask_mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        mode = str(r.get("mask_mode", ""))
        out[mode] = out.get(mode, 0) + 1
    return out


def checkpoint_names(metrics: dict[str, Any]) -> list[str]:
    return [str(x.get("name")) for x in metrics.get("saved_checkpoints", [])]


def main() -> None:
    target_log = read_jsonl(TARGET_RUN / "training_log.jsonl")
    base_log = read_jsonl(BASE_RUN / "training_log.jsonl")
    target_metrics = load_json(TARGET_RUN / "scientific_metrics.json")
    base_metrics = load_json(BASE_RUN / "scientific_metrics.json")
    target_dyn = read_jsonl(TARGET_RUN / "dynamics_traces.jsonl") if (TARGET_RUN / "dynamics_traces.jsonl").exists() else []
    base_dyn = read_jsonl(BASE_RUN / "dynamics_traces.jsonl") if (BASE_RUN / "dynamics_traces.jsonl").exists() else []

    n = min(len(target_log), len(base_log))
    first_diff: dict[str, Any] | None = None
    same_prefix_steps = 0
    for i in range(n):
        tr = target_log[i]
        br = base_log[i]
        diffs = {}
        for field in COMPARE_FIELDS:
            if not nearly_same(tr.get(field), br.get(field)):
                diffs[field] = {"wwm_to_token": tr.get(field), "fixed_wwm": br.get(field)}
        if diffs:
            first_diff = {"line_index_1based": i + 1, "step": tr.get("step"), "diffs": diffs}
            break
        same_prefix_steps = i + 1

    last_same_row = target_log[same_prefix_steps - 1] if same_prefix_steps else None
    first_token_row = next((r for r in target_log if r.get("mask_mode") == "token"), None)
    token_steps = [int(r["step"]) for r in target_log if r.get("mask_mode") == "token"]
    wwm_steps = [int(r["step"]) for r in target_log if r.get("mask_mode") == "wwm"]

    target_ck = checkpoint_names(target_metrics)
    base_ck = checkpoint_names(base_metrics)
    post_switch_ckpts = [ck for ck in ["chck_80M", "chck_90M", "chck_100M"] if ck in target_ck and ck in base_ck]
    optional_identity_ckpt = "chck_70M" if "chck_70M" in target_ck and "chck_70M" in base_ck else None

    dyn_by_ck_target = {str(r.get("checkpoint")): r for r in target_dyn}
    dyn_by_ck_base = {str(r.get("checkpoint")): r for r in base_dyn}
    dyn_compare: dict[str, Any] = {}
    for ck in [x for x in [optional_identity_ckpt, *post_switch_ckpts] if x]:
        dyn_compare[ck] = {
            "wwm_to_token": dyn_by_ck_target.get(ck),
            "fixed_wwm": dyn_by_ck_base.get(ck),
        }

    payload: dict[str, Any] = {
        "status": "WWM_TO_TOKEN_TRAINING_MEASUREMENT",
        "target_run": str(TARGET_RUN),
        "fixed_wwm_run": str(BASE_RUN),
        "shared_design": {
            "data": target_metrics.get("example_jsonl"),
            "parameter_count_target": target_metrics.get("parameter_count"),
            "parameter_count_fixed_wwm": base_metrics.get("parameter_count"),
            "tokenizer_label_target": target_metrics.get("tokenizer_label"),
            "tokenizer_label_fixed_wwm": base_metrics.get("tokenizer_label"),
            "word_exposure_target": target_metrics.get("word_exposure"),
            "word_exposure_fixed_wwm": base_metrics.get("word_exposure"),
            "actual_training_steps_target": target_metrics.get("actual_training_steps"),
            "actual_training_steps_fixed_wwm": base_metrics.get("actual_training_steps"),
            "seed_target": target_metrics.get("seed"),
            "seed_fixed_wwm": base_metrics.get("seed"),
        },
        "first_scientific_difference": first_diff,
        "same_prefix_steps_excluding_elapsed": same_prefix_steps,
        "last_same_training_row_excluding_elapsed": last_same_row,
        "first_token_training_row": first_token_row,
        "target_mask_mode_counts": mask_mode_counts(target_log),
        "fixed_wwm_mask_mode_counts": mask_mode_counts(base_log),
        "target_token_step_range": [min(token_steps), max(token_steps)] if token_steps else None,
        "target_wwm_step_range": [min(wwm_steps), max(wwm_steps)] if wwm_steps else None,
        "loss_windows": {
            "pre_switch_last_100_steps_target": summarize_loss(target_log, max(1, (first_token_row or {}).get("step", 1) - 100), max(1, (first_token_row or {}).get("step", 1) - 1)) if first_token_row else None,
            "pre_switch_last_100_steps_fixed_wwm": summarize_loss(base_log, max(1, (first_token_row or {}).get("step", 1) - 100), max(1, (first_token_row or {}).get("step", 1) - 1)) if first_token_row else None,
            "post_switch_first_100_steps_target": summarize_loss(target_log, int(first_token_row["step"]), int(first_token_row["step"]) + 99) if first_token_row else None,
            "post_switch_same_100_steps_fixed_wwm": summarize_loss(base_log, int(first_token_row["step"]), int(first_token_row["step"]) + 99) if first_token_row else None,
            "final_100_steps_target": summarize_loss(target_log, max(1, len(target_log) - 99), len(target_log)),
            "final_100_steps_fixed_wwm": summarize_loss(base_log, max(1, len(base_log) - 99), len(base_log)),
        },
        "saved_checkpoints_target": target_ck,
        "saved_checkpoints_fixed_wwm_count": len(base_ck),
        "minimal_downstream_checkpoints": post_switch_ckpts,
        "optional_identity_checkpoint": optional_identity_ckpt,
        "dynamics_at_identity_and_post_switch": dyn_compare,
        "interpretation": {
            "what_this_establishes": "The fixed-data WWM->token run has the same training trace as fixed WWM until the late masking switch, so downstream evaluation can focus on post-switch checkpoints rather than remeasuring the identical prefix.",
            "what_this_does_not_establish": "Lower token-level MLM loss after the switch is not downstream BabyLM competence; official-compatible zero-shot plus Reading evaluation is still needed.",
            "lowest_cost_next_eval": "Evaluate chck_80M, chck_90M, and chck_100M against the existing clean-Qwen fixed-WWM trajectory summary; optionally include chck_70M only as an identity anchor if a reviewer needs it.",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "wwm_to_token_training_measurement.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fnum(x: Any) -> str:
        return "NA" if x is None else f"{float(x):.6f}"

    lines = []
    lines.append("# research WWM-to-token training measurement\n\n")
    lines.append("This CPU-only measurement prepares the orthogonal masking-schedule route without using an H100. It compares the completed WWM->token run with the COMPACT_EXPERIENCE clean-Qwen fixed-WWM run.\n\n")
    lines.append("## Shared setup\n\n")
    lines.append(f"- Target run: `{TARGET_RUN}`\n")
    lines.append(f"- Fixed-WWM reference: `{BASE_RUN}`\n")
    lines.append(f"- Both report 100M word exposure and {target_metrics.get('actual_training_steps')} training steps with baseline16k / DeBERTa-v2 8x480 / clean-Qwen data.\n\n")
    lines.append("## Training trace comparison\n\n")
    if first_diff:
        lines.append(f"- First non-elapsed training-log difference appears at step {first_diff['step']}.\n")
    if last_same_row:
        lines.append(f"- Rows are identical through step {same_prefix_steps}, cumulative exposure {last_same_row.get('cumulative_word_exposure')} words, with mask mode `{last_same_row.get('mask_mode')}`.\n")
    if first_token_row:
        lines.append(f"- WWM->token first token-masking row: step {first_token_row.get('step')}, cumulative exposure {first_token_row.get('cumulative_word_exposure')}, loss {first_token_row.get('loss')}.\n")
    lines.append(f"- Target mask-mode counts: {payload['target_mask_mode_counts']}.\n")
    lines.append(f"- Fixed-WWM mask-mode counts: {payload['fixed_wwm_mask_mode_counts']}.\n\n")
    lines.append("## Loss windows\n\n")
    lw = payload["loss_windows"]
    for key, val in lw.items():
        if val:
            lines.append(f"- {key}: n={val['n']}, first={fnum(val['first'])}, last={fnum(val['last'])}, mean={fnum(val['mean'])}.\n")
    lines.append("\nThe post-switch token objective gives a much lower MLM loss, but this changes the prediction target and cannot be read as BabyLM competence by itself.\n\n")
    lines.append("## Smallest downstream conversion\n\n")
    lines.append(f"Evaluate only post-switch checkpoints `{', '.join(post_switch_ckpts)}` first. The optional identity anchor is `{optional_identity_ckpt}`. This should decide whether the leader-style WWM7->Token3 factor is worth combining with whichever data mechanism survives.\n\n")
    lines.append(f"JSON: `{out_json}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "same_prefix_steps_excluding_elapsed": same_prefix_steps,
        "first_difference_step": first_diff.get("step") if first_diff else None,
        "first_token_step": first_token_row.get("step") if first_token_row else None,
        "minimal_downstream_checkpoints": post_switch_ckpts,
        "out_json": str(out_json),
        "note": str(NOTE),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
