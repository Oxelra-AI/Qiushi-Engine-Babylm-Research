#!/usr/bin/env python3
"""research CPU smoke for the init-matched minfreq50 trainer wrapper.

This creates a tiny prefix JSONL and runs one CPU-only miniature MLM training job
through `minfreq50_initmatched_trainer.py`.  It is not performance
evidence.  It checks that the monkeypatched build_model path actually executes,
copies same-shape tensors, leaves only vocab-shaped tensors skipped, and produces
valid trainer artifacts/metrics without touching GPUs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
OUT_DIR = WORKSPACE / "data/minfreq50_initmatched_smoke"
TRAINER = WORKSPACE / "scripts/minfreq50_initmatched_trainer.py"
TOKENIZER = WORKSPACE / "data/supportfloor_tokenizers/legal_byte_bpe_40k_minfreq50"
TRAIN_FILE = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
META = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"


def make_smoke_jsonl(n_rows: int) -> tuple[pathlib.Path, int, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / "smoke_prefix_rows.jsonl"
    rows = 0
    words = 0
    with TRAIN_FILE.open("r", encoding="utf-8") as src, dst.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            w = int(obj.get("words", 0)) or len(text.split())
            if w != len(text.split()):
                raise RuntimeError(f"word mismatch in source prefix row {rows+1}")
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            rows += 1
            words += w
            if rows >= n_rows:
                break
    if rows != n_rows:
        raise RuntimeError(f"needed {n_rows} rows, got {rows}")
    return dst, rows, words


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke_jsonl, rows, words = make_smoke_jsonl(8)
    run_dir = OUT_DIR / "tiny_cpu_run"
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    cmd = [
        sys.executable,
        "-B",
        str(TRAINER),
        "--example_jsonl", str(smoke_jsonl),
        "--example_jsonl_label", "initmatched_cpu_smoke",
        "--example_jsonl_meta", str(META),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "legal_byte_bpe_40k_minfreq50_initmatched_smoke",
        "--hidden_size", "64",
        "--n_layer", "1",
        "--n_head", "4",
        "--ffn_mult", "2",
        "--seed", "43",
        "--extra_init_seed", "43022",
        "--train_rng_seed", "43023",
        "--batch_size", "4",
        "--seq_length", "96",
        "--max_seq_length", "96",
        "--learning_rate", "0.001",
        "--warmup_fraction", "0.06",
        "--weight_decay", "0.01",
        "--masking_curriculum", "wwm_fixed",
        "--mask_prob_start", "0.15",
        "--mask_prob_end", "0.15",
        "--checkpoint_words", str(words + 1),
        "--max_word_exposure", str(words),
        "--num_workers", "0",
        "--log_every", "1",
        "--dynamics_trace_every", "1",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TOKENIZERS_PARALLELISM"] = "false"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, text=True, capture_output=True, timeout=600)
    metrics_path = run_dir / "scientific_metrics.json"
    metrics: dict[str, Any] | None = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    init_events = []
    for line in p.stdout.splitlines():
        if "initmatched_model_build" in line:
            try:
                init_events.append(json.loads(line))
            except Exception:
                pass
    status = {
        "status": "INITMATCHED_TRAINER_SMOKE_OK" if p.returncode == 0 and metrics is not None and init_events else "INITMATCHED_TRAINER_SMOKE_FAILED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_utc": started,
        "returncode": p.returncode,
        "smoke_rows": rows,
        "smoke_words": words,
        "cmd": cmd,
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics is not None,
        "init_events": init_events,
        "metrics_digest": None if metrics is None else {
            "word_exposure": metrics.get("word_exposure"),
            "actual_training_steps": metrics.get("actual_training_steps"),
            "loss_first": metrics.get("loss_first"),
            "loss_last": metrics.get("loss_last"),
            "parameter_count": metrics.get("parameter_count"),
            "vocab_size": metrics.get("vocab_size"),
            "tokenizer_label": metrics.get("tokenizer_label"),
            "saved_checkpoints": [x.get("name") for x in metrics.get("saved_checkpoints", [])],
        },
        "stdout_tail": p.stdout[-6000:],
        "stderr_tail": p.stderr[-4000:],
    }
    out_json = OUT_DIR / "initmatched_trainer_smoke.json"
    out_md = OUT_DIR / "initmatched_trainer_smoke.md"
    out_json.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research init-matched trainer CPU smoke",
        "",
        f"Status: `{status['status']}` returncode `{p.returncode}`.",
        f"Tiny run rows/words: `{rows}` / `{words}`.",
        "",
    ]
    if init_events:
        ev = init_events[-1]
        lines += [
            "## Init-matched build event",
            "",
            f"Target/ref vocab: `{ev.get('target_vocab_size')}` / `{ev.get('reference_vocab_size')}`.",
            f"Copied same-shape tensors: `{ev.get('copied_same_shape_tensor_count')}`; skipped vocab-shaped tensors: `{ev.get('skipped_shape_mismatch_count')}`.",
            f"Same-shape exact numel fraction after copy: `{ev.get('same_shape_exact_numel_fraction_after_copy')}`.",
            f"Random-like exact tensors after copy: `{ev.get('random_like_exact_tensor_count_after_copy')}/{ev.get('random_like_tensor_count_after_copy')}`.",
            "",
        ]
    if metrics:
        lines += [
            "## Tiny metrics",
            "",
            f"Word exposure `{metrics.get('word_exposure')}`, steps `{metrics.get('actual_training_steps')}`, loss_first `{metrics.get('loss_first')}`, loss_last `{metrics.get('loss_last')}`, params `{metrics.get('parameter_count')}`.",
            "",
        ]
    lines += ["This is only a software smoke, not model-quality evidence.", "", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)
    if status["status"] != "INITMATCHED_TRAINER_SMOKE_OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
