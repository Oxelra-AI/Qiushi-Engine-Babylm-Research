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
OUT_JSON = ROOT / "data/masked_smoke_summary.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/masked_smoke_and_mlm_eval.md')

RUN_SPECS = [
    ("token", "babylm_masked_token_smoke10k"),
    ("wwm", "babylm_masked_wwm_smoke10k"),
]

COMMON_ARGS = [
    "--max_word_exposure", "10000",
    "--example_pool_words", "20000",
    "--checkpoint_words", "10000",
    "--words_per_example", "80",
    "--mask_prob", "0.15",
    "--seq_length", "64",
    "--max_seq_length", "128",
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
]


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


def load_check(run_dir: pathlib.Path, mask_mode: str) -> dict:
    checks = []
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = run_dir / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        ids = torch.randint(0, len(tok), (2, 16))
        labels = ids.clone()
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
        checks.append({
            "rel": rel,
            "ok": True,
            "tokenizer_class": tok.__class__.__name__,
            "model_class": model.__class__.__name__,
            "vocab_size": len(tok),
            "mask_token_id": tok.mask_token_id,
            "pad_token_id": tok.pad_token_id,
            "parameter_count": sum(x.numel() for x in model.parameters()),
            "forward_loss": float(out.loss.detach()),
        })
    metrics = json.loads((run_dir / "scientific_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "example_order_manifest.json").read_text(encoding="utf-8"))
    assert metrics["word_exposure"] == 10000
    assert metrics["backend"] == "mlm"
    assert metrics["mask_mode"] == mask_mode
    assert manifest["selected_for_training_words"] == 10000
    rec = {
        "checks": checks,
        "metrics_subset": {
            "variant": metrics["variant"],
            "backend": metrics["backend"],
            "mask_mode": metrics["mask_mode"],
            "mask_prob": metrics["mask_prob"],
            "parameter_count": metrics["parameter_count"],
            "loss_first": metrics["loss_first"],
            "loss_last": metrics["loss_last"],
            "word_exposure": metrics["word_exposure"],
            "actual_training_steps": metrics["actual_training_steps"],
        },
        "first12_examples": manifest["consumed_example_ids_in_order"][:12],
        "source_words": manifest["source_words_consumed"],
    }
    (run_dir / "masked_load_check.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return rec


def read_avg(report: pathlib.Path) -> float | None:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    return float(m.group(1)) if m else None


def eval_blimp_fast(run_dir: pathlib.Path, env: dict[str, str]) -> dict:
    model_path = (run_dir / "hf_model").resolve()
    outdir = (run_dir / "eval_results_mlm_smoke").resolve()
    run([
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path),
        "--backend", "mlm",
        "--task", "blimp",
        "--data_path", "evaluation_data/fast_eval/blimp_fast",
        "--save_predictions",
        "--revision_name", "chck_1M",
        "--batch_size", "64",
        "--output_dir", str(outdir),
    ], env, cwd=STRICT_DIR)
    report = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / "blimp" / "blimp_fast" / "best_temperature_report.txt"
    score = read_avg(report)
    assert score is not None, report
    return {"blimp_fast": score, "report": str(report)}


def main() -> None:
    env = setup_env()
    rows = []
    for mask_mode, run_id in RUN_SPECS:
        run_dir = RUN_ROOT / run_id
        cmd = [
            sys.executable, str(TRAIN),
            "--output_dir", str(run_dir),
            "--mask_mode", mask_mode,
            *COMMON_ARGS,
        ]
        run_training(cmd, output_dir=run_dir, timeout=900, env=env)
        load = load_check(run_dir, mask_mode)
        ev = eval_blimp_fast(run_dir, env)
        rows.append({"mask_mode": mask_mode, "run_id": run_id, "load_check": load, "eval": ev})
        print("MASKED_SMOKE_OK", run_id, "loss", load["metrics_subset"]["loss_first"], "->", load["metrics_subset"]["loss_last"], "BLiMP", ev["blimp_fast"])

    same_order = rows[0]["load_check"]["first12_examples"] == rows[1]["load_check"]["first12_examples"]
    same_source = rows[0]["load_check"]["source_words"] == rows[1]["load_check"]["source_words"]
    summary = {"status": "MASKED_SMOKE_AND_MLM_EVAL_OK", "same_first12_order": same_order, "same_source_words": same_source, "rows": rows}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# research — Masked-LM smoke and official `mlm` evaluator check",
        "",
        f"Evidence JSON: `{OUT_JSON}`",
        "",
        "Two 10k-word masked-LM smokes were trained with the new standalone `babylm_masked_train.py`: token masking and whole-word masking. Both load with `AutoModelForMaskedLM` at root and `chck_1M`; both were evaluated on official fast BLiMP using backend `mlm`.",
        "",
        "| mask mode | run id | params | loss first | loss last | BLiMP fast (`mlm`) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        m = r["load_check"]["metrics_subset"]
        lines.append(f"| {r['mask_mode']} | `{r['run_id']}` | {m['parameter_count']} | {m['loss_first']:.4f} | {m['loss_last']:.4f} | {r['eval']['blimp_fast']:.2f} |")
    lines += ["", f"Same first 12 example IDs: `{same_order}`. Same source-word mix: `{same_source}`."]
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUT_JSON)
    print("WROTE", OUT_NOTE)


if __name__ == "__main__":
    main()
