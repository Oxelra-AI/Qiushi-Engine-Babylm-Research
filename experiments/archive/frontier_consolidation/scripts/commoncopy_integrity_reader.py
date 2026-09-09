#!/usr/bin/env python3
"""research: file-only integrity reader for common-copied no-disentangle runs.

Checks the two research common-copy DeBERTa trainings:
  * compact: commoncopy_nodis_compact_deberta100M_seed43022
  * repeat:  commoncopy_nodis_repeat_deberta100M_seed43022

The common-copy runs are produced by common_copy_nodis_trainer_wrapper.py.
Unlike research launcher runs, they do not need a research launcher_result.json; the
load-bearing initialization proof is common_copy_initialization.json.

This reader verifies:
  * common-copy record: source p2c,c2p; target []; target 30,773,344 params;
    all 140 common tensors copied; only-left 32 positional-projection tensors;
    target is common-copied from full init.
  * training metrics: 100M words, 2,529 steps, legal tokenizer label, WWM 0.15,
    DeBERTa 8x480/8 heads, batch256/seq256, finite losses, 10 checkpoints.
  * checkpoint files/configs for chck_80M and chck_100M exist and retain
    relative attention, absolute input positions, and pos_att_type=[].

No selected evaluation, SuperGLUE, AoA, upload, submission, or training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import shlex
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_OUT = WS / "data/commoncopy_integrity"
REQUIRED_CHECKPOINTS = ["chck_80M", "chck_100M"]
NODIS_PARAM_COUNT = 30_773_344
FULL_PARAM_COUNT = 34_467_424

ARMS: dict[str, dict[str, Any]] = {
    "commoncopy_compact": {
        "run_dir": WS / "training/runs/commoncopy_nodis_compact_deberta100M_seed43022",
        "data_arm": "compact",
        "expected_label": "common_copy_nodis_compact",
        "launch_trace": ROOT / "data/external/tool_0001.json",
    },
    "commoncopy_repeat": {
        "run_dir": WS / "training/runs/commoncopy_nodis_repeat_deberta100M_seed43022",
        "data_arm": "repeat",
        "expected_label": "common_copy_nodis_repeat",
        "launch_trace": ROOT / "data/external/tool_0002.json",
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def model_file(ck_dir: Path) -> Path | None:
    for name in ["model.safetensors", "pytorch_model.bin"]:
        p = ck_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def tail(path: Path, n: int = 20) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return lines[-n:]


def names_from_saved_checkpoints(metrics: dict[str, Any] | None) -> list[str]:
    if not metrics:
        return []
    out: list[str] = []
    for rec in metrics.get("saved_checkpoints", []) or []:
        if isinstance(rec, dict) and rec.get("name"):
            out.append(str(rec["name"]))
        elif isinstance(rec, str):
            out.append(rec)
    return out


def check_init_record(init: dict[str, Any] | None, run_dir: Path) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    checks: dict[str, Any] = {"common_copy_initialization_exists": init is not None}
    problems: list[str] = []
    if init is None:
        return checks, [f"common_copy_initialization.json missing at {run_dir}"], {}
    brief = {
        "status": init.get("status"),
        "mode": init.get("mode"),
        "source_pos_att_type": init.get("source_pos_att_type"),
        "target_pos_att_type": init.get("target_pos_att_type"),
        "source_parameter_count": init.get("source_parameter_count"),
        "target_parameter_count": init.get("target_parameter_count"),
        "after_copy_common_key_count": (init.get("after_copy_state_comparison") or {}).get("common_key_count"),
        "after_copy_nonexact_common_count": (init.get("after_copy_state_comparison") or {}).get("nonexact_common_count"),
        "only_left_count": (init.get("after_copy_state_comparison") or {}).get("only_left_count"),
        "initial_logit_mean_abs": (init.get("initial_logit_difference_source_full_vs_target_common_copied") or {}).get("mean_abs_logit_diff"),
    }
    after = init.get("after_copy_state_comparison") or {}
    copy = init.get("copy_record") or {}
    target_cfg = init.get("target_config") or {}
    source_cfg = init.get("source_config") or {}
    expected = {
        "status_ok": init.get("status") == "COMMON_COPIED_NODIS_INITIALIZATION",
        "training_mode_ok": init.get("mode") == "training_build_model_patch",
        "source_pos_full_ok": list(init.get("source_pos_att_type") or []) == ["p2c", "c2p"],
        "target_pos_empty_ok": list(init.get("target_pos_att_type") or []) == [],
        "target_common_copied_flag_ok": bool(init.get("target_is_common_copied_from_source_full_init")) is True,
        "source_param_count_ok": init.get("source_parameter_count") == FULL_PARAM_COUNT,
        "target_param_count_ok": init.get("target_parameter_count") == NODIS_PARAM_COUNT,
        "after_copy_common_count_ok": after.get("common_key_count") == 140,
        "after_copy_nonexact_zero_ok": after.get("nonexact_common_count") == 0,
        "copy_all_target_tensors_ok": bool(copy.get("all_target_tensors_copied")) is True,
        "only_left_pos_projection_count_ok": after.get("only_left_count") == 32,
        "source_config_pos_ok": list(source_cfg.get("pos_att_type") or []) == ["p2c", "c2p"],
        "target_config_pos_ok": list(target_cfg.get("pos_att_type") or []) == [],
        "target_hidden_ok": target_cfg.get("hidden_size") == 480 and target_cfg.get("num_hidden_layers") == 8,
    }
    checks.update(expected)
    for key, ok in expected.items():
        if not ok:
            problems.append(f"init {key} failed; brief={brief}")
    return checks, problems, brief

def command_from_launch_trace(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        return str((((rec.get("tool_call") or {}).get("arguments") or {}).get("arguments") or {}).get("command") or "")
    except Exception:
        return ""


def has_cli_arg(command: str, name: str, expected: str) -> bool:
    if not command:
        return False
    try:
        parts = shlex.split(command)
    except Exception:
        parts = command.split()
    flag = f"--{name}"
    for i, part in enumerate(parts):
        if part == flag and i + 1 < len(parts) and parts[i + 1] == expected:
            return True
        if part.startswith(flag + "=") and part.split("=", 1)[1] == expected:
            return True
    return False


def check_metrics(metrics: dict[str, Any] | None, run_dir: Path, data_arm: str, launch_trace: Path | None) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    checks: dict[str, Any] = {"scientific_metrics_exists": metrics is not None}
    problems: list[str] = []
    if metrics is None:
        return checks, [f"scientific_metrics.json missing at {run_dir}"], {}
    saved_names = names_from_saved_checkpoints(metrics)
    command = command_from_launch_trace(launch_trace)
    batch_size_ok = metrics.get("batch_size") == 256 or has_cli_arg(command, "batch_size", "256")
    brief = {
        "word_exposure": metrics.get("word_exposure"),
        "actual_training_steps": metrics.get("actual_training_steps"),
        "loss_first": metrics.get("loss_first"),
        "loss_last": metrics.get("loss_last"),
        "parameter_count": metrics.get("parameter_count"),
        "vocab_size": metrics.get("vocab_size"),
        "tokenizer_label": metrics.get("tokenizer_label"),
        "saved_checkpoint_count": len(saved_names),
        "first_checkpoint": saved_names[0] if saved_names else None,
        "last_checkpoint": saved_names[-1] if saved_names else None,
        "example_jsonl_label": metrics.get("example_jsonl_label"),
        "example_jsonl": metrics.get("example_jsonl"),
        "batch_size_metric": metrics.get("batch_size"),
        "batch_size_from_launch_trace": has_cli_arg(command, "batch_size", "256"),
        "launch_trace": str(launch_trace) if launch_trace else None,
    }
    expected = {
        "word_exposure_ok": metrics.get("word_exposure") == 100_000_000,
        "actual_training_steps_ok": metrics.get("actual_training_steps") == 2529,
        "parameter_count_ok": metrics.get("parameter_count") == NODIS_PARAM_COUNT,
        "vocab_size_ok": metrics.get("vocab_size") == 16_384,
        "tokenizer_label_ok": metrics.get("tokenizer_label") == "compliant16k_reinvest10M",
        "model_family_ok": metrics.get("model_family") == "DebertaV2ForMaskedLM",
        "hidden_size_ok": metrics.get("hidden_size") == 480,
        "n_layer_ok": metrics.get("n_layer") == 8,
        "n_head_ok": metrics.get("n_head") == 8,
        "masking_curriculum_ok": metrics.get("masking_curriculum") == "wwm_fixed",
        "mask_prob_start_ok": float(metrics.get("mask_prob_start", -1.0)) == 0.15,
        "mask_prob_end_ok": float(metrics.get("mask_prob_end", -1.0)) == 0.15,
        "batch_size_ok": batch_size_ok,
        "seq_length_ok": metrics.get("seq_length") == 256,
        "max_seq_length_ok": metrics.get("max_seq_length") == 256,
        "loss_first_finite": finite(metrics.get("loss_first")),
        "loss_last_finite": finite(metrics.get("loss_last")),
        "saved_checkpoint_count_ok": len(saved_names) == 10,
        "required_checkpoints_in_metrics": all(ck in saved_names for ck in REQUIRED_CHECKPOINTS),
        "data_arm_label_ok": (data_arm in str(metrics.get("example_jsonl_label", "")) or data_arm in str(metrics.get("example_jsonl", ""))),
    }
    checks.update(expected)
    for key, ok in expected.items():
        if not ok:
            problems.append(f"metrics {key} failed; brief={brief}")
    return checks, problems, brief


def check_checkpoints(run_dir: Path) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    checks: dict[str, Any] = {}
    problems: list[str] = []
    brief: dict[str, Any] = {}
    for ck in REQUIRED_CHECKPOINTS:
        ck_dir = run_dir / "hf_model" / ck
        cfg = read_json(ck_dir / "config.json")
        mf = model_file(ck_dir)
        rec = {
            "dir": str(ck_dir),
            "model_file": str(mf) if mf else None,
            "model_file_size": mf.stat().st_size if mf else None,
            "config": None,
        }
        checks[f"{ck}_dir_exists"] = ck_dir.exists()
        checks[f"{ck}_model_file_exists"] = mf is not None
        checks[f"{ck}_config_exists"] = cfg is not None
        if not ck_dir.exists():
            problems.append(f"checkpoint dir missing: {ck_dir}")
        if mf is None:
            problems.append(f"checkpoint model file missing/empty: {ck_dir}")
        if cfg is None:
            problems.append(f"checkpoint config missing: {ck_dir / 'config.json'}")
        else:
            rec["config"] = {
                "hidden_size": cfg.get("hidden_size"),
                "num_hidden_layers": cfg.get("num_hidden_layers"),
                "num_attention_heads": cfg.get("num_attention_heads"),
                "intermediate_size": cfg.get("intermediate_size"),
                "vocab_size": cfg.get("vocab_size"),
                "relative_attention": cfg.get("relative_attention"),
                "position_biased_input": cfg.get("position_biased_input"),
                "pos_att_type": cfg.get("pos_att_type"),
            }
            expected = {
                f"{ck}_hidden_size_ok": cfg.get("hidden_size") == 480,
                f"{ck}_num_hidden_layers_ok": cfg.get("num_hidden_layers") == 8,
                f"{ck}_num_attention_heads_ok": cfg.get("num_attention_heads") == 8,
                f"{ck}_intermediate_size_ok": cfg.get("intermediate_size") == 1920,
                f"{ck}_vocab_size_ok": cfg.get("vocab_size") == 16_384,
                f"{ck}_relative_attention_ok": cfg.get("relative_attention") is True,
                f"{ck}_position_biased_input_ok": cfg.get("position_biased_input") is True,
                f"{ck}_pos_att_type_empty_ok": list(cfg.get("pos_att_type") or []) == [],
            }
            checks.update(expected)
            for key, ok in expected.items():
                if not ok:
                    problems.append(f"checkpoint config {key} failed; got={rec['config']}")
        brief[ck] = rec
    return checks, problems, brief


def check_arm(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(spec["run_dir"])
    checks: dict[str, Any] = {"run_dir_exists": run_dir.exists()}
    problems: list[str] = []
    if not run_dir.exists():
        problems.append(f"run_dir missing: {run_dir}")
    init_checks, init_problems, init_brief = check_init_record(read_json(run_dir / "common_copy_initialization.json"), run_dir)
    launch_trace = Path(spec.get("launch_trace")) if spec.get("launch_trace") else None
    metric_checks, metric_problems, metrics_brief = check_metrics(read_json(run_dir / "scientific_metrics.json"), run_dir, spec["data_arm"], launch_trace)
    ck_checks, ck_problems, ck_brief = check_checkpoints(run_dir)
    checks.update(init_checks)
    checks.update(metric_checks)
    checks.update(ck_checks)
    problems.extend(init_problems)
    problems.extend(metric_problems)
    problems.extend(ck_problems)
    return {
        "arm": name,
        "data_arm": spec["data_arm"],
        "run_dir": str(run_dir),
        "checks": checks,
        "common_copy_initialization_brief": init_brief,
        "metrics_brief": metrics_brief,
        "checkpoint_brief": ck_brief,
        "stdout_tail": tail(run_dir / "train_stdout.log"),
        "stderr_tail": tail(run_dir / "train_stderr.log"),
        "problems": problems,
        "ok_for_selected_eval": len(problems) == 0,
    }


def write_md(payload: dict[str, Any], out_dir: Path) -> None:
    lines = ["# research common-copy no-disentangle integrity", "", f"Created UTC: `{payload.get('created_utc')}`", ""]
    lines.append(f"all_commoncopy_ready_for_selected_eval: `{payload.get('all_commoncopy_ready_for_selected_eval')}`")
    lines.append("")
    for name, rec in payload.get("arm_results", {}).items():
        lines.append(f"## {name}")
        lines.append(f"- ok_for_selected_eval: `{rec.get('ok_for_selected_eval')}`")
        lines.append(f"- run_dir: `{rec.get('run_dir')}`")
        lines.append(f"- init: `{json.dumps(rec.get('common_copy_initialization_brief'), sort_keys=True)}`")
        lines.append(f"- metrics: `{json.dumps(rec.get('metrics_brief'), sort_keys=True)}`")
        if rec.get("problems"):
            lines.append("- problems:")
            for p in rec["problems"]:
                lines.append(f"  - {p}")
        lines.append("")
    lines.append("## Boundary")
    lines.append(payload.get("boundary", ""))
    (out_dir / "commoncopy_integrity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arm_results = {name: check_arm(name, spec) for name, spec in ARMS.items()}
    payload = {
        "status": "COMMONCOPY_INTEGRITY",
        "created_utc": now(),
        "out_dir": str(args.out_dir),
        "arm_results": arm_results,
        "all_commoncopy_ready_for_selected_eval": all(r["ok_for_selected_eval"] for r in arm_results.values()),
        "readout_condition": "Run the research common-copy selected panel only after both arms pass this reader.",
        "boundary": "File-only integrity reader; no training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission.",
    }
    out_json = args.out_dir / "commoncopy_integrity.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, args.out_dir)
    print(json.dumps({
        "status": payload["status"],
        "json": str(out_json),
        "md": str(args.out_dir / "commoncopy_integrity.md"),
        "all_ready": payload["all_commoncopy_ready_for_selected_eval"],
        "problem_count": sum(len(r["problems"]) for r in arm_results.values()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
