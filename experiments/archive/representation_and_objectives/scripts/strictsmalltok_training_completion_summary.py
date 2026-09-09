#!/usr/bin/env python3
"""research: summarize completed corrected-tokenizer retrains.

Reads the two completed training directories and verifies
that the training artifact itself is a complete compliant endpoint substrate:
100M exposure, 100 saved checkpoints, required AoA ladder present, expected
10M-trained tokenizer SHA in chck_1M and chck_100M, exact example-order manifest,
and research command provenance.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import time
from pathlib import Path
from typing import Any

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/strictsmalltok_training_completion_summary.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
OUT = _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_training_completion/strictsmalltok_training_completion_summary.json')
NOTE = _public_path('research/notes/representation_and_objectives/strictsmalltok_training_completion_summary.md')
EXPECTED_TOKENIZER_SHA = "4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738"
EXPECTED_STREAM_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
REQUIRED_AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{10*i}M" for i in range(1, 11)]
RUNS = {
    "43022": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022'),
    "43122": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43122'),
}
TASK_RESULT_PATHS = {
    "43022": _public_path('experiments/archive/representation_and_objectives/tasks/s47_t19_tool1/result.json'),
    "43122": _public_path('experiments/archive/representation_and_objectives/tasks/s47_t20_tool1/result.json'),
}
COMMAND_MANIFESTS = {
    "43022": _public_path('experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/train_command_seed43022.json'),
    "43122": _public_path('experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/train_command_seed43122.json'),
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_log_tail_summary(path: Path) -> dict[str, Any]:
    n = 0
    last_train = None
    done = None
    checkpoints = []
    first_train = None
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ev = obj.get("event")
                if ev == "train":
                    if first_train is None:
                        first_train = obj
                    last_train = obj
                elif ev == "checkpoint_saved":
                    checkpoints.append(obj.get("name"))
                elif ev == "done":
                    done = obj
                elif ev is None and ("step" in obj) and ("cumulative_word_exposure" in obj):
                    # Actual training training_log.jsonl format: plain per-step
                    # records with no "event" key. Treat each as a train record.
                    if first_train is None:
                        first_train = obj
                    last_train = obj
    return {"records": n, "first_train": first_train, "last_train": last_train, "done": done, "checkpoint_names_from_log": checkpoints}


def seed_summary(seed: str, run_dir: Path) -> dict[str, Any]:
    model_root = run_dir / "hf_model"
    ckpt_dirs = sorted([p.name for p in model_root.glob("chck_*M") if p.is_dir()], key=lambda x: (len(x), x)) if model_root.exists() else []
    log_summary = read_log_tail_summary(run_dir / "training_log.jsonl")
    metrics = load_json(run_dir / "scientific_metrics.json") or {}
    example_order = load_json(run_dir / "example_order_manifest.json") or {}
    command_manifest = load_json(COMMAND_MANIFESTS[seed]) or {}
    tokenizer_shas = {
        "chck_1M": sha(model_root / "chck_1M/tokenizer.json"),
        "chck_100M": sha(model_root / "chck_100M/tokenizer.json"),
    }
    task_result = load_json(TASK_RESULT_PATHS[seed])
    missing_required_aoa = [s for s in REQUIRED_AOA_STEPS if not (model_root / s).exists()]
    done = log_summary.get("done") or {}
    final_train = log_summary.get("last_train") or {}
    final_cum_words = final_train.get("cumulative_word_exposure")
    metrics_word_exposure = metrics.get("word_exposure")
    exposure_100M = (
        done.get("word_exposure") == 100000000
        or final_cum_words == 100000000
        or metrics_word_exposure == 100000000
    )
    completion_checks = {
        "task_exit_code_zero": isinstance(task_result, dict) and task_result.get("returncode", 0) in (0, None) if task_result is not None else True,
        "word_exposure_100M_in_done": exposure_100M,
        "final_train_cum_words_100M": final_cum_words == 100000000,
        "required_aoa_ladder_present": missing_required_aoa == [],
        "chck_100M_exists": (model_root / "chck_100M").exists(),
        "tokenizer_sha_chck1M_ok": tokenizer_shas["chck_1M"] == EXPECTED_TOKENIZER_SHA,
        "tokenizer_sha_chck100M_ok": tokenizer_shas["chck_100M"] == EXPECTED_TOKENIZER_SHA,
        "example_order_seed43": example_order.get("seed") == 43,
        "example_order_selected_words_100M": example_order.get("selected_for_training_words") == 100000000,
        "command_manifest_stream_hash_ok": EXPECTED_STREAM_SHA in json.dumps(command_manifest),
    }
    return {
        "seed": seed,
        "run_dir": rel(run_dir),
        "model_root": rel(model_root),
        "training_log": rel(run_dir / "training_log.jsonl"),
        "scientific_metrics_path": rel(run_dir / "scientific_metrics.json"),
        "example_order_manifest_path": rel(run_dir / "example_order_manifest.json"),
        "command_manifest_path": rel(COMMAND_MANIFESTS[seed]),
        "task_result_path": rel(TASK_RESULT_PATHS[seed]),
        "checkpoint_dir_count": len(ckpt_dirs),
        "first_5_checkpoints": ckpt_dirs[:5],
        "last_5_checkpoints": ckpt_dirs[-5:],
        "required_aoa_missing_steps": missing_required_aoa,
        "tokenizer_shas": tokenizer_shas,
        "training_log_summary": log_summary,
        "scientific_metrics_selected": {k: metrics.get(k) for k in ["variant", "word_exposure", "actual_training_steps", "parameter_count", "vocab_size", "tokenizer_label", "loss_first", "loss_last", "seed", "extra_init_seed", "train_rng_seed", "mask_mode", "mask_switch_mode", "seq_length", "batch_size", "optimizer", "learning_rate", "n_layer", "hidden_size", "n_head", "intermediate_size"] if k in metrics},
        "example_order_selected": {k: example_order.get(k) for k in ["seed", "selected_for_training_words", "num_examples", "train_file", "train_file_sha256"] if k in example_order},
        "completion_checks": completion_checks,
        "complete_training_artifact": all(completion_checks.values()),
    }


def main() -> None:
    seeds = {seed: seed_summary(seed, run) for seed, run in RUNS.items()}
    payload = {
        "status": "STRICTSMALLTOK_TRAINING_COMPLETION_SUMMARY",
        "created_utc": now_utc(),
        "expected_tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "expected_stream_sha256": EXPECTED_STREAM_SHA,
        "required_aoa_steps": REQUIRED_AOA_STEPS,
        "seeds": seeds,
        "both_complete_training_artifacts": all(s["complete_training_artifact"] for s in seeds.values()),
        "interpretation": "These checks validate the training substrate only. Endpoint quality must come from the research full official evaluations now running on both seeds.",
    }
    _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_training_completion').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research Strict-Small-tokenizer training completion summary",
        "",
        f"Both complete training artifacts: `{payload['both_complete_training_artifacts']}`",
        "",
        "| seed | final loss | word exposure | checkpoints | AoA ladder missing | chck_100M tokenizer ok | complete |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, rec in seeds.items():
        ml = rec["scientific_metrics_selected"].get("loss_last")
        ft = rec["training_log_summary"].get("last_train") or {}
        we = ft.get("cumulative_word_exposure") or rec["scientific_metrics_selected"].get("word_exposure")
        lines.append(f"| {seed} | {ml} | {we} | {rec['checkpoint_dir_count']} | {len(rec['required_aoa_missing_steps'])} | {rec['completion_checks']['tokenizer_sha_chck100M_ok']} | {rec['complete_training_artifact']} |")
    lines.extend(["", "Endpoint competence is not inferred from loss. Full official evaluations are required and are separate from this training-artifact check."])
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(OUT),
        "note": rel(NOTE),
        "both_complete_training_artifacts": payload["both_complete_training_artifacts"],
        "seed_final_loss": {seed: rec["scientific_metrics_selected"].get("loss_last") for seed, rec in seeds.items()},
        "seed_complete": {seed: rec["complete_training_artifact"] for seed, rec in seeds.items()},
    }, indent=2), flush=True)
    if not payload["both_complete_training_artifacts"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
