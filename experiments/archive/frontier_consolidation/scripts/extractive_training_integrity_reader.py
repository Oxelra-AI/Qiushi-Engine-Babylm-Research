#!/usr/bin/env python3
"""research: CPU integrity reader for research extractive DeBERTa trainings.

This checker is intentionally read-only with respect to model training. It can be run
before the managed H100 jobs finish (then it reports pending) or after they finish
(to verify that both source-only arms are mechanically comparable before selected
evaluation). It does not run selected eval, SuperGLUE, AoA, upload, or leaderboard
submission.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
WS = STUDY
DEFAULT_OUT = WS / "data/extractive_training_integrity"
MANIFEST = WS / "data/extractive_view_100m_streams/extractive_100m_streams_manifest.json"
STREAM_DIR = WS / "data/extractive_view_100m_streams"

ARMS: dict[str, dict[str, Any]] = {
    "balanced": {
        "run_dir": WS / "training/runs/extractive_balanced_deberta100M_seed43022",
        "stream_name": "extractive_balanced_100M.jsonl",
        "expected_sha256": "8a4e77af3ae7049b889e5d8f174616606a930bcd8371188ea35db3197175776c",
        "role": "density/token-mass-nearest source-only compact proxy",
    },
    "wide": {
        "run_dir": WS / "training/runs/extractive_wide_deberta100M_seed43022",
        "stream_name": "extractive_wide_100M.jsonl",
        "expected_sha256": "b9e4666844f966a2fedbe6a37eb61aa3b1bcd5fe08cdc738cd0a15fda4258753",
        "role": "coverage/content-maximized source-only compact proxy",
    },
}

EXPECTED = {
    "word_exposure": 100_000_000,
    "actual_training_steps": 2529,
    "parameter_count": 34_467_424,
    "vocab_size": 16_384,
    "tokenizer_label": "compliant16k_reinvest10M",
    "model_family": "DebertaV2ForMaskedLM",
    "hidden_size": 480,
    "n_layer": 8,
    "n_head": 8,
    "ffn_mult": 4,
    "seed": 43,
    "seq_length": 256,
    "max_seq_length": 256,
    "masking_curriculum": "wwm_fixed",
    "mask_prob_start": 0.15,
    "mask_prob_end": 0.15,
    # research trains with checkpoint_words=10,000,000, so a valid 100M run saves
    # chck_10M ... chck_100M (10 checkpoint directories), unlike older 1M-grid
    # reference trajectories.
    "saved_checkpoint_count": 10,
}

REQUIRED_CHECKPOINTS = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_record(stream_name: str) -> dict[str, Any] | None:
    m = load_json(MANIFEST)
    if not m:
        return None
    for rec in m.get("variant_results", []):
        if Path(str(rec.get("path", ""))).name == stream_name:
            return rec
    return None


def tail(path: Path, n: int = 3000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-n:]


def check_float_equal(a: Any, b: float, tol: float = 1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def check_arm(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(cfg["run_dir"])
    metrics_path = run_dir / "scientific_metrics.json"
    launcher_result_path = run_dir / "launcher_result.json"
    train_command_path = run_dir / "train_command.json"
    stdout_log = run_dir / "train_stdout.log"
    stderr_log = run_dir / "train_stderr.log"
    metrics = load_json(metrics_path)
    launcher = load_json(launcher_result_path)
    train_command = load_json(train_command_path)
    rec = manifest_record(str(cfg["stream_name"]))

    problems: list[str] = []
    checks: dict[str, Any] = {
        "run_dir_exists": run_dir.exists(),
        "metrics_exists": metrics is not None,
        "launcher_result_exists": launcher is not None,
        "train_command_exists": train_command is not None,
        "manifest_record_exists": rec is not None,
    }
    if rec is None:
        problems.append("missing stream manifest record")
    else:
        checks["manifest_status_ok"] = bool(rec.get("status_ok"))
        checks["manifest_sha_match_expected"] = rec.get("sha256") == cfg["expected_sha256"]
        checks["manifest_words_exact_100M"] = rec.get("words") == EXPECTED["word_exposure"]
        checks["manifest_rows_647400"] = rec.get("rows") == 647_400
        if not checks["manifest_status_ok"]:
            problems.append("stream manifest status_ok false")
        if not checks["manifest_sha_match_expected"]:
            problems.append(f"stream manifest sha mismatch: {rec.get('sha256')} != {cfg['expected_sha256']}")
        if not checks["manifest_words_exact_100M"] or not checks["manifest_rows_647400"]:
            problems.append("stream manifest does not show exact 647400 rows / 100M words")

    if launcher is not None:
        checks["launcher_returncode_zero"] = launcher.get("returncode") == 0
        checks["launcher_metrics_exists_flag"] = bool(launcher.get("metrics_exists"))
        if launcher.get("returncode") not in (0, None):
            problems.append(f"launcher returncode {launcher.get('returncode')}")
    else:
        problems.append("launcher_result.json missing (run may still be pending)")

    if train_command is not None:
        checks["train_command_train_file_sha_match"] = train_command.get("train_file_sha256") == cfg["expected_sha256"]
        cmd_recipe = train_command.get("recipe", {})
        command_expected = {
            "max_word_exposure": EXPECTED["word_exposure"],
            "lr_total_steps": EXPECTED["actual_training_steps"],
            "parameter_count_expected": EXPECTED["parameter_count"],
            "hidden_size": EXPECTED["hidden_size"],
            "n_layer": EXPECTED["n_layer"],
            "n_head": EXPECTED["n_head"],
            "ffn_mult": EXPECTED["ffn_mult"],
            "seed": EXPECTED["seed"],
            "extra_init_seed": 43022,
            "train_rng_seed": 43023,
            "batch_size": 256,
            "seq_length": EXPECTED["seq_length"],
            "max_seq_length": EXPECTED["max_seq_length"],
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "warmup_fraction": 0.06,
            "checkpoint_words": 10_000_000,
            "masking_curriculum": EXPECTED["masking_curriculum"],
            "mask_prob_start": EXPECTED["mask_prob_start"],
            "mask_prob_end": EXPECTED["mask_prob_end"],
        }
        checks["train_command_recipe_expected"] = {k: (cmd_recipe.get(k) == v) for k, v in command_expected.items()}
        checks["train_command_expected_exposure"] = checks["train_command_recipe_expected"].get("max_word_exposure", False)
        checks["train_command_lr_total_steps"] = checks["train_command_recipe_expected"].get("lr_total_steps", False)
        checks["train_command_tokenizer_sha"] = train_command.get("tokenizer_metadata", {}).get("tokenizer_json_sha256_observed")
        if not checks["train_command_train_file_sha_match"]:
            problems.append("train_command stream sha mismatch")
        for k, ok in checks["train_command_recipe_expected"].items():
            if not ok:
                problems.append(f"train_command recipe {k} mismatch: {cmd_recipe.get(k)} != {command_expected[k]}")
    else:
        problems.append("train_command.json missing")

    metrics_brief: dict[str, Any] = {}
    if metrics is not None:
        for key, val in EXPECTED.items():
            if key == "saved_checkpoint_count":
                continue
            observed = metrics.get(key)
            if isinstance(val, float):
                ok = check_float_equal(observed, val)
            else:
                ok = observed == val
            checks[f"metrics_{key}_ok"] = ok
            metrics_brief[key] = observed
            if not ok:
                problems.append(f"metrics {key} mismatch: {observed} != {val}")
        saved = metrics.get("saved_checkpoints", [])
        metrics_brief["saved_checkpoint_count"] = len(saved)
        checks["metrics_saved_checkpoint_count_ok"] = len(saved) == EXPECTED["saved_checkpoint_count"]
        if len(saved) != EXPECTED["saved_checkpoint_count"]:
            problems.append(f"saved checkpoint count {len(saved)} != {EXPECTED['saved_checkpoint_count']}")
        names = {x.get("name") for x in saved if isinstance(x, dict)}
        checks["required_checkpoints_present"] = {ck: ck in names and ((run_dir / "hf_model" / ck / "model.safetensors").exists() or (run_dir / "hf_model" / ck / "pytorch_model.bin").exists()) for ck in REQUIRED_CHECKPOINTS}
        for ck, ok in checks["required_checkpoints_present"].items():
            if not ok:
                problems.append(f"required checkpoint missing or no weight file: {ck}")
        metrics_brief["loss_first"] = metrics.get("loss_first")
        metrics_brief["loss_last"] = metrics.get("loss_last")
        metrics_brief["first_checkpoint"] = saved[0].get("name") if saved else None
        metrics_brief["last_checkpoint"] = saved[-1].get("name") if saved else None
    else:
        problems.append("scientific_metrics.json missing (run may still be pending)")

    return {
        "arm": name,
        "role": cfg["role"],
        "run_dir": str(run_dir),
        "stream_name": cfg["stream_name"],
        "expected_stream_sha256": cfg["expected_sha256"],
        "files": {
            "metrics": str(metrics_path),
            "launcher_result": str(launcher_result_path),
            "train_command": str(train_command_path),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        },
        "metrics_brief": metrics_brief,
        "checks": checks,
        "problems": problems,
        "ready_for_selected_eval": metrics is not None and not problems,
        "stdout_tail": tail(stdout_log, 1200) if metrics is None else "",
        "stderr_tail": tail(stderr_log, 1200) if metrics is None or problems else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = {name: check_arm(name, cfg) for name, cfg in ARMS.items()}
    all_ready = all(rec["ready_for_selected_eval"] for rec in arms.values())
    payload = {
        "status": "EXTRACTIVE_TRAINING_INTEGRITY",
        "created_utc": now(),
        "meaning": "CPU integrity reader for research balanced/wide source-only DeBERTa trainings before selected readout.",
        "manifest": str(MANIFEST),
        "stream_dir": str(STREAM_DIR),
        "expected": EXPECTED,
        "required_selected_checkpoints": REQUIRED_CHECKPOINTS,
        "arms": arms,
        "all_ready_for_selected_eval": all_ready,
        "next_when_ready": "Run extractive_selected_eval_panel.py on balanced, wide, and legal_compact late-band checkpoints; then, after per-target predictions exist, optionally run selected_prediction_movement_reader.py for the most informative extractive-vs-compact checkpoint contrasts. No SuperGLUE, AoA, upload, or leaderboard submission.",
        "no_training_selected_eval_upload_aoa_or_leaderboard": True,
    }
    out_json = args.out_dir / "extractive_training_integrity.json"
    out_md = args.out_dir / "extractive_training_integrity.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research extractive training integrity", "", f"Created UTC: `{payload['created_utc']}`", "", f"All ready for selected eval: **{all_ready}**", ""]
    for name, rec in arms.items():
        lines.append(f"## {name}")
        lines.append(f"- ready_for_selected_eval: `{rec['ready_for_selected_eval']}`")
        lines.append(f"- metrics: `{rec['files']['metrics']}`")
        lines.append(f"- problems: {json.dumps(rec['problems'], ensure_ascii=False)}")
        if rec.get("metrics_brief"):
            lines.append(f"- metrics_brief: `{json.dumps(rec['metrics_brief'], ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "all_ready_for_selected_eval": all_ready, "ready": {k: v["ready_for_selected_eval"] for k, v in arms.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
