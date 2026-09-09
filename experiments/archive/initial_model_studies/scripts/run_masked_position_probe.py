#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess

from training_process import run_training
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAIN = ROOT / "training/scripts/babylm_masked_train.py"
STRICT_DIR = ROOT / "repos/babylm-eval/strict"
RUN_ROOT = ROOT / "training/runs"
RUN_ID = "babylm_masked_token_pos512_smoke10k"
RUN_DIR = RUN_ROOT / RUN_ID
OUT_JSON = ROOT / "data/masked_position_probe.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/masked_position_probe.md')


def setup_env() -> dict[str, str]:
    env = os.environ.copy()
    hf_home = ROOT / "training/hf_home"
    env["HF_HOME"] = str(hf_home.resolve())
    env["HF_HUB_CACHE"] = str((hf_home / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((ROOT / "training/hf_modules_cache").resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run(cmd: list[str], env: dict[str, str], cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None)
    print(p.stdout[-5000:], flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed {p.returncode}: {' '.join(cmd)}\n{p.stdout[-12000:]}")
    return p


def read_avg(report: pathlib.Path) -> float | None:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    return float(m.group(1)) if m else None


def main() -> None:
    env = setup_env()
    run_training([
        sys.executable, str(TRAIN),
        "--output_dir", str(RUN_DIR),
        "--mask_mode", "token",
        "--max_word_exposure", "10000",
        "--example_pool_words", "20000",
        "--checkpoint_words", "10000",
        "--words_per_example", "80",
        "--mask_prob", "0.15",
        "--seq_length", "64",
        "--max_seq_length", "128",
        "--max_position_embeddings", "512",
        "--batch_size", "16",
        "--lr_total_steps", "16",
        "--hidden_size", "128",
        "--n_layer", "2",
        "--n_head", "4",
        "--learning_rate", "0.001",
        "--seed", "42",
        "--extra_init_seed", "456",
        "--train_rng_seed", "789",
        "--log_every", "4",
    ], output_dir=RUN_DIR, timeout=900, env=env)
    checks = []
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = RUN_DIR / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        max_pos = int(model.config.max_position_embeddings)
        assert max_pos >= 512, max_pos
        seq_len = 200
        ids = torch.randint(low=0, high=len(tok), size=(2, seq_len))
        labels = ids.clone()
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
        checks.append({
            "rel": rel,
            "model_class": model.__class__.__name__,
            "tokenizer_class": tok.__class__.__name__,
            "max_position_embeddings": max_pos,
            "seq_len_forward_tested": seq_len,
            "forward_loss": float(out.loss.detach()),
            "parameter_count": sum(x.numel() for x in model.parameters()),
        })
    outdir = (RUN_DIR / "eval_results_position_probe").resolve()
    model_path = (RUN_DIR / "hf_model").resolve()
    run([
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path),
        "--backend", "mlm",
        "--task", "entity_tracking",
        "--data_path", "evaluation_data/fast_eval/entity_tracking_fast",
        "--save_predictions",
        "--revision_name", "chck_1M",
        "--batch_size", "64",
        "--output_dir", str(outdir),
    ], env, cwd=STRICT_DIR)
    report = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / "entity_tracking" / "entity_tracking_fast" / "best_temperature_report.txt"
    score = read_avg(report)
    assert score is not None, report
    metrics = json.loads((RUN_DIR / "scientific_metrics.json").read_text(encoding="utf-8"))
    payload = {
        "status": "POSITION_PROBE_OK",
        "run_id": RUN_ID,
        "checks": checks,
        "entity_tracking_fast_mlm": score,
        "entity_report": str(report),
        "metrics_subset": {
            "word_exposure": metrics["word_exposure"],
            "loss_first": metrics["loss_first"],
            "loss_last": metrics["loss_last"],
            "max_seq_length": metrics["max_seq_length"],
            "max_position_embeddings": metrics["max_position_embeddings"],
            "parameter_count": metrics["parameter_count"],
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text(
        "# research — Masked max-position repair probe\n\n"
        f"Evidence JSON: `{OUT_JSON}`\n\n"
        "A 10k token-masked BertForMaskedLM checkpoint was trained with `max_position_embeddings=512`, loaded from root and `chck_1M`, passed a 200-token forward test, and completed official fast Entity Tracking with backend `mlm`.\n\n"
        f"Entity Tracking fast (`mlm`): {score:.2f}.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("WROTE", OUT_JSON)
    print("WROTE", OUT_NOTE)


if __name__ == "__main__":
    main()
