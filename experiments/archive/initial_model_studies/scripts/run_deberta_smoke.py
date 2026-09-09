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
RUN_ID = "babylm_debertav2_wwm_smoke10k"
RUN_DIR = ROOT / "training/runs" / RUN_ID
OUT_JSON = ROOT / "data/deberta_smoke_summary.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/deberta_smoke.md')


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
    print(p.stdout[-6000:], flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed with {p.returncode}: {' '.join(cmd)}\n{p.stdout[-12000:]}")
    return p


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if not m:
        raise RuntimeError(f"could not parse {report}\n{txt[:500]}")
    return float(m.group(1))


def main() -> None:
    env = setup_env()
    run_training([
        sys.executable, str(TRAIN),
        "--output_dir", str(RUN_DIR),
        "--model_type", "deberta_v2",
        "--hidden_size", "240",
        "--n_layer", "8",
        "--n_head", "6",
        "--ffn_mult", "4",
        "--position_buckets", "256",
        "--max_relative_positions", "256",
        "--deberta_relative_attention", "true",
        "--deberta_pos_att_type", "p2c,c2p",
        "--max_word_exposure", "10000",
        "--example_pool_words", "20000",
        "--checkpoint_words", "10000",
        "--words_per_example", "80",
        "--tokenizer_label", "baseline16k",
        "--tokenization_summary_limit", "0",
        "--mask_mode", "wwm",
        "--mask_prob", "0.15",
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--max_position_embeddings", "512",
        "--batch_size", "16",
        "--lr_total_steps", "16",
        "--learning_rate", "0.001",
        "--seed", "42",
        "--extra_init_seed", "456",
        "--train_rng_seed", "789",
        "--log_every", "4",
    ], output_dir=RUN_DIR, timeout=1200, env=env)

    checks = []
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = RUN_DIR / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        assert model.__class__.__name__ == "DebertaV2ForMaskedLM", model.__class__.__name__
        assert int(model.config.hidden_size) == 240
        assert int(model.config.num_hidden_layers) == 8
        assert int(model.config.num_attention_heads) == 6
        assert getattr(model.config, "relative_attention") is True
        ids = torch.randint(low=0, high=len(tok), size=(2, 180))
        y = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        checks.append({
            "rel": rel,
            "tokenizer_class": tok.__class__.__name__,
            "model_class": model.__class__.__name__,
            "parameter_count": sum(x.numel() for x in model.parameters()),
            "input_embedding_params": model.get_input_embeddings().weight.numel(),
            "hidden_size": int(model.config.hidden_size),
            "layers": int(model.config.num_hidden_layers),
            "heads": int(model.config.num_attention_heads),
            "intermediate_size": int(model.config.intermediate_size),
            "relative_attention": bool(getattr(model.config, "relative_attention", False)),
            "pos_att_type": list(getattr(model.config, "pos_att_type", [])),
            "position_buckets": int(getattr(model.config, "position_buckets", 0)),
            "max_relative_positions": int(getattr(model.config, "max_relative_positions", 0)),
            "seq_len_forward_tested": 180,
            "forward_loss": float(y.loss.detach()),
        })

    outdir = (RUN_DIR / "eval_results_deberta_smoke").resolve()
    model_path = (RUN_DIR / "hf_model").resolve()
    scores = {}
    reports = {}
    for task_name, task, data_path in [
        ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
        ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ]:
        run([
            sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(model_path),
            "--backend", "mlm",
            "--task", task,
            "--data_path", data_path,
            "--save_predictions",
            "--revision_name", "chck_1M",
            "--batch_size", "64",
            "--output_dir", str(outdir),
        ], env, cwd=STRICT_DIR)
        report = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / task / task_name / "best_temperature_report.txt"
        scores[task_name] = read_avg(report)
        reports[task_name] = str(report)

    metrics = json.loads((RUN_DIR / "scientific_metrics.json").read_text(encoding="utf-8"))
    summary = {
        "status": "DEBERTA_V2_WWM_SMOKE_OK",
        "run_id": RUN_ID,
        "checks": checks,
        "scores": scores,
        "reports": reports,
        "metrics_subset": {
            "model_family": metrics["model_family"],
            "model_type": metrics["model_type"],
            "parameter_count": metrics["parameter_count"],
            "embedding_parameter_count": metrics["embedding_parameter_count"],
            "non_embedding_parameter_count": metrics["non_embedding_parameter_count"],
            "vocab_size": metrics["vocab_size"],
            "loss_first": metrics["loss_first"],
            "loss_last": metrics["loss_last"],
            "word_exposure": metrics["word_exposure"],
            "masked_tokens_per_whitespace_word": metrics["masked_tokens_per_whitespace_word"],
            "hidden_size": metrics["hidden_size"],
            "n_layer": metrics["n_layer"],
            "n_head": metrics["n_head"],
            "intermediate_size": metrics["intermediate_size"],
            "position_buckets": metrics["position_buckets"],
            "max_relative_positions": metrics["max_relative_positions"],
            "deberta_pos_att_type": metrics["deberta_pos_att_type"],
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text(
        "# research — DeBERTa-v2 WWM smoke\n\n"
        f"Evidence JSON: `{OUT_JSON}`\n\n"
        "A primary parameter-matched DeBERTa-v2 WWM checkpoint trained for 10k official words, saved root/`chck_1M`, loaded with `AutoModelForMaskedLM`, passed 180-token forward tests, and completed official fast BLiMP and Entity with backend `mlm`.\n\n"
        f"BLiMP fast: {scores['blimp_fast']:.2f}; Entity fast: {scores['entity_tracking_fast']:.2f}.\n"
        f"Params: {metrics['parameter_count']}; embedding params: {metrics['embedding_parameter_count']}; non-embedding params: {metrics['non_embedding_parameter_count']}.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUT_JSON)
    print("WROTE", OUT_NOTE)


if __name__ == "__main__":
    main()
