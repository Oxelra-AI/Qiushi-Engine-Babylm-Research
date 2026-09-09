#!/usr/bin/env python3
from __future__ import annotations

import json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
MODEL = (ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model").resolve()
OUTDIR = (ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_aoa").resolve()
OUT_JSON = ROOT / "data/debertav2_b256_aoa_result.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_b256_aoa_result.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_b256_aoa_eval.log')
WORD_PATH = (STRICT / "evaluation_data/full_eval/aoa/cdi_childes.json").resolve()
SCORE_REL = pathlib.Path("hf_model/main/zero_shot/mlm/AoA_word/aoa_score.json")
SURPRISAL_REL = pathlib.Path("hf_model/main/zero_shot/mlm/AoA_word/surprisal.json")


def setup_env():
    env = os.environ.copy()
    hf_home = ROOT / "training/hf_home"
    env["HF_HOME"] = str(hf_home.resolve())
    env["HF_HUB_CACHE"] = str((hf_home / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((ROOT / "training/hf_modules_cache").resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = setup_env()
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.AoA_word.run",
        "--model_name", str(MODEL),
        "--backend", "mlm",
        "--track_name", "strict-small",
        "--word_path", str(WORD_PATH),
        "--output_dir", str(OUTDIR),
        "--min_context", "20",
    ]
    with LOG.open("a", encoding="utf-8") as logf:
        logf.write("$ " + " ".join(cmd) + "\n")
        p = subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(p.stdout[-8000:])
        logf.write(p.stdout + f"\n[returncode={p.returncode}]\n")
        if p.returncode != 0:
            raise SystemExit(p.returncode)
    score_path = OUTDIR / SCORE_REL
    if not score_path.exists():
        hits = list(OUTDIR.rglob("aoa_score.json"))
        if not hits:
            raise FileNotFoundError(f"no aoa_score.json under {OUTDIR}")
        score_path = hits[0]
    score_data = json.loads(score_path.read_text())
    aoa = float(score_data.get("aoa", score_data.get("curve_fitness")))
    payload = {"model_path": str(MODEL), "backend": "mlm", "track_name": "strict-small", "word_path": str(WORD_PATH), "output_dir": str(OUTDIR), "score_path": str(score_path), "surprisal_path": str(score_path.parent / 'surprisal.json'), "aoa": aoa}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    OUT_NOTE.write_text(f"# research DeBERTa-v2 b256 AoA\n\nAoA: {aoa:.4f}\n\nEvidence: `{OUT_JSON}`\nRaw score: `{score_path}`\n", encoding="utf-8")
    print(json.dumps({"status":"AOA_DONE","aoa":aoa,"out":str(OUT_JSON)}, indent=2))

if __name__ == "__main__":
    main()