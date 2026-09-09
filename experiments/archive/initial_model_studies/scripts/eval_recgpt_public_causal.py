#!/usr/bin/env python3
"""research — Local official-compatible causal evaluation for public RecGPT-10M.

Uses the patched local copy of Serdar404/RecGPT-10M created in research.
Runs the main zero-shot columns needed to verify the public phenotype under the
local BabyLM strict evaluator: BLiMP, Supplement, Entity, COMPS,
GlobalPIQA parallel/nonparallel, EWoK, and Reading.

SuperGLUE/AoA are intentionally left for a later full coordinate pass after the
core causal phenotype is verified, because RecGPT FlexAttention is slow without
compilation in this runtime.
"""
from __future__ import annotations
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
FULL = STRICT / "evaluation_data/full_eval"
MODEL = ROOT / "data/recgpt_local/patched_model"
OUT_JSON = ROOT / "data/recgpt_public_causal_scores.json"
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/recgpt_public_causal_eval.log')
OUTDIR = ROOT / "training/runs/recgpt_public_causal_eval"

TASKS = [
    ("blimp", "blimp", FULL / "blimp_filtered", "blimp_filtered", 8),
    ("supplement", "blimp", FULL / "supplement_filtered", "supplement_filtered", 8),
    ("entity_tracking", "entity_tracking", FULL / "entity_tracking", "entity_tracking", 8),
    ("comps", "comps", FULL / "comps", "comps", 8),
    ("global_piqa_parallel", "global_piqa_parallel", FULL / "global_piqa_parallel", "global_piqa_parallel", 8),
    ("global_piqa_nonparallel", "global_piqa_nonparallel", FULL / "global_piqa_nonparallel", "global_piqa_nonparallel", 8),
    ("ewok", "ewok", FULL / "ewok_filtered_word_tokenize", "ewok_filtered_word_tokenize", 8),
]

PUBLIC_CARD = {
    "blimp": 73.11,
    "supplement": 61.73,
    "ewok": 52.62,
    "entity_tracking": 16.59,
    "comps": 55.43,
    "GlobalPIQA_mean": 40.68,
    "SuperGLUE": 66.64,
    "NLP_Average": 52.40,
}


def env_setup() -> dict[str, str]:
    env = os.environ.copy()
    root_abs = pathlib.Path.cwd().resolve()
    env["NLTK_DATA"] = str((root_abs / ROOT / "data/nltk_data").resolve())
    hf = root_abs / ROOT / "training/hf_home"
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((root_abs / ROOT / "training/hf_modules_cache").resolve())
    env["PYTHONPATH"] = str((root_abs / STRICT).resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def run(cmd: list[str], env: dict[str, str], logf) -> None:
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    logf.write("\n" + line + "\n")
    logf.flush()
    p = subprocess.run(cmd, cwd=str(pathlib.Path.cwd()), env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-2500:], flush=True)
    logf.write(p.stdout + f"\n[rc={p.returncode}]\n")
    logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {line}\n{p.stdout[-8000:]}")


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if m:
        return float(m.group(1))
    vals = re.findall(r"(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)", txt, flags=re.I)
    if vals:
        v = float(vals[-1])
        return v * (100 if v <= 1 else 1)
    raise RuntimeError(f"Could not parse score from {report}\n{txt[:1000]}")


def read_reading(report: pathlib.Path) -> dict[str, float]:
    txt = report.read_text(encoding="utf-8", errors="replace")
    out = {}
    for label, key in [("EYE TRACKING SCORE", "reading_eye_tracking"), ("SELF-PACED READING SCORE", "reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        if not m:
            raise RuntimeError(f"Could not parse {label} from {report}")
        out[key] = float(m.group(1))
    return out


def main() -> None:
    t0 = time.time()
    if not MODEL.exists():
        raise RuntimeError(f"Missing patched RecGPT model dir: {MODEL}")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    env = env_setup()
    root_abs = pathlib.Path.cwd().resolve()
    model_abs = root_abs / MODEL
    out_abs = root_abs / OUTDIR
    payload = {
        "status": "RECGPT_PUBLIC_CAUSAL_EVAL_RUNNING",
        "model_path": str(MODEL),
        "public_model_card_scores": PUBLIC_CARD,
        "scores": {},
        "reports": {},
        "notes": "Patched only tokenizer_config and a Transformers API import/copy mismatch; weights and architecture unchanged.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with LOG.open("w", encoding="utf-8") as logf:
        for col, task, data, dataset, bs in TASKS:
            run([
                sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
                "--model_path_or_name", str(model_abs),
                "--backend", "causal",
                "--task", task,
                "--data_path", str((root_abs / data).resolve()),
                "--save_predictions",
                "--revision_name", "recgpt_public",
                "--batch_size", str(bs),
                "--output_dir", str(out_abs),
            ], env, logf)
            reports = sorted(out_abs.rglob(f"zero_shot/causal/{task}/{dataset}/best_temperature_report.txt"), key=lambda p: str(p))
            if not reports and col == "ewok":
                reports = sorted(out_abs.rglob("zero_shot/causal/ewok/*/best_temperature_report.txt"), key=lambda p: str(p))
            if not reports:
                raise RuntimeError(f"No report found for {col}")
            report = reports[-1]
            payload["reports"][col] = str(report)
            payload["scores"][col] = read_avg(report)
            if "global_piqa_parallel" in payload["scores"] and "global_piqa_nonparallel" in payload["scores"]:
                payload["scores"]["GlobalPIQA_mean"] = (payload["scores"]["global_piqa_parallel"] + payload["scores"]["global_piqa_nonparallel"]) / 2
            OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        # Reading
        reading = FULL / "reading/reading_data.csv"
        run([
            sys.executable, "-m", "evaluation_pipeline.reading.run",
            "--model_path_or_name", str(model_abs),
            "--backend", "causal",
            "--data_path", str((root_abs / reading).resolve()),
            "--revision_name", "recgpt_public",
            "--output_dir", str(out_abs),
        ], env, logf)
        reports = sorted(out_abs.rglob("zero_shot/causal/reading/report.txt"), key=lambda p: str(p))
        if not reports:
            raise RuntimeError("No Reading report found")
        payload["reports"]["reading"] = str(reports[-1])
        payload["scores"].update(read_reading(reports[-1]))
        payload["scores"]["Reading_mean"] = (payload["scores"]["reading_eye_tracking"] + payload["scores"]["reading_self_paced"]) / 2
    if all(k in payload["scores"] for k in ["blimp", "supplement", "ewok", "entity_tracking", "comps", "GlobalPIQA_mean", "Reading_mean"]):
        payload["scores"]["NLP_mean_no_superglue_aoa"] = sum(payload["scores"][k] for k in ["blimp", "supplement", "ewok", "entity_tracking", "comps", "GlobalPIQA_mean", "Reading_mean"]) / 7
    payload["status"] = "RECGPT_PUBLIC_CAUSAL_EVAL_COMPLETE"
    payload["elapsed_sec"] = round(time.time() - t0, 1)
    payload["deltas_vs_public_card"] = {k: payload["scores"][k] - v for k, v in PUBLIC_CARD.items() if k in payload["scores"]}
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT_JSON), "scores": payload["scores"], "deltas_vs_public_card": payload["deltas_vs_public_card"]}, indent=2))

if __name__ == "__main__":
    main()
