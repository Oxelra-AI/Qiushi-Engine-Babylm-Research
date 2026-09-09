#!/usr/bin/env python3
"""research common BabyLM screen for chck_82M, coherent86 alpha0.75, and replay arms.

Extends the research fast evaluator in two ways needed for trustworthy comparison:
  * Reading receives the required --data_path argument.
  * A model's executed private-adapter scale is changed by setting every module's
    private_adapter.scale, not only config.private_adapter_scale.

This is still a fast/cheap7 research screen, not the complete official stack.  It uses
fast_eval splits for BLiMP/Supp/EWoK/Entity/GlobalPIQA and full_eval COMPS, matching the
inherited fast-screen convention; complete official evaluation must follow for surviving
candidates.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

ROOT = _public_path('.')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
OUT_DEFAULT = _public_path('experiments/archive/functional_learning/data/common_eval')

TASKS = [
    ("BLiMP", "blimp", "evaluation_data/fast_eval/blimp_fast", 32),
    ("Supplement", "blimp", "evaluation_data/fast_eval/supplement_fast", 32),
    ("EWoK", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", 16),
    ("Entity", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast", 32),
    ("COMPS", "comps", "evaluation_data/full_eval/comps", 32),
    ("GlobalPIQA_parallel", "global_piqa_parallel", "evaluation_data/fast_eval/global_piqa_parallel", 32),
    ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", "evaluation_data/fast_eval/global_piqa_nonparallel", 32),
]
EQ7_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]


def parse_score(text: str) -> Optional[float]:
    m = re.search(r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    for line in reversed(text.splitlines()):
        s = line.strip()
        m2 = re.match(r"^(?:[0-9.]+\s+)?([+-]?[0-9]+(?:\.[0-9]+)?)$", s)
        if m2:
            val = float(m2.group(1))
            if -1000 < val < 1000:
                return val
    vals = re.findall(r"(?:accuracy|score|acc|acc_norm)[^0-9+\-]*([+-]?[0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if vals:
        val = float(vals[-1])
        return val * (100 if 0 <= val <= 1 else 1)
    return None


def parse_reading(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def setup_env(out_root: pathlib.Path, tag: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    hf = out_root / "hf_cache" / tag
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    nltk_path = _public_path('experiments/archive/initial_model_studies/data/nltk_data')
    if nltk_path.exists():
        env["NLTK_DATA"] = str(nltk_path.resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: pathlib.Path, timeout: int = 2400) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ " + " ".join(cmd) + "\n")
    p = subprocess.run(cmd, cwd=str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')), env=env, capture_output=True, text=True, timeout=timeout)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(p.stdout[-6000:] if len(p.stdout) > 6000 else p.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(p.stderr[-4000:] if len(p.stderr) > 4000 else p.stderr)
        f.write(f"\n[returncode={p.returncode}]\n")
    return p


def materialize_scaled_model(src: pathlib.Path, alpha: Optional[float], out_root: pathlib.Path, tag: str) -> pathlib.Path:
    """Create a small wrapper checkpoint with the executed private_adapter scale set."""
    if alpha is None:
        return src
    dst = out_root / "scaled_models" / f"{tag}_alpha{str(alpha).replace('.', 'p')}"
    done = dst / ".complete"
    if done.exists() and (dst / "model.safetensors").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    code = f'''
import os, pathlib, sys
hf = pathlib.Path({str((out_root / "hf_cache" / (tag + "_materialize")).resolve())!r})
os.environ["HF_HOME"] = str(hf)
os.environ["HF_HUB_CACHE"] = str(hf / "hub")
os.environ["HF_DATASETS_CACHE"] = str(hf / "datasets")
os.environ["TRANSFORMERS_CACHE"] = str(hf / "transformers")
os.environ["HF_MODULES_CACHE"] = str(hf / "modules")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
    pathlib.Path(os.environ[k]).mkdir(parents=True, exist_ok=True)
from transformers import AutoModelForMaskedLM, AutoTokenizer
src = {str(src.resolve())!r}
dst = {str(dst.resolve())!r}
alpha = {alpha!r}
model = AutoModelForMaskedLM.from_pretrained(src, trust_remote_code=True, local_files_only=True)
if hasattr(model.config, "private_adapter_scale"):
    model.config.private_adapter_scale = float(alpha)
if hasattr(model, "deberta"):
    for layer in model.deberta.encoder.layer:
        if hasattr(layer, "private_adapter"):
            layer.private_adapter.scale = float(alpha)
model.save_pretrained(dst, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
tok.save_pretrained(dst)
for extra in ["frozen82_private_modeling.py", "adapter_scaled_modeling.py"]:
    p = pathlib.Path(src) / extra
    if p.exists():
        q = pathlib.Path(dst) / extra
        if not q.exists():
            q.write_bytes(p.read_bytes())
(pathlib.Path(dst) / "executed_scale_manifest.json").write_text(__import__("json").dumps({{"source": src, "alpha": alpha, "changed_module_private_adapter_scale": True, "hf_modules_cache": os.environ["HF_MODULES_CACHE"]}}, indent=2))
'''
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        subprocess.check_call([sys.executable, tmp])
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    done.write_text("ok\n")
    return dst


def eval_sentence(model_path: pathlib.Path, col: str, task: str, data: str, batch: int,
                  tag: str, env: Dict[str, str], out_root: pathlib.Path) -> Dict[str, Any]:
    task_out = out_root / "outputs" / tag / col
    log = out_root / "logs" / tag / f"{col}.log"
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", task,
        "--data_path", data,
        "--revision_name", f"step026_{tag}_{col}",
        "--save_predictions",
        "--batch_size", str(batch),
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    try:
        p = run_cmd(cmd, env, log)
    except subprocess.TimeoutExpired:
        return {"column": col, "score": None, "error": "timeout"}
    score = None
    for txt_file in sorted(task_out.rglob("best_temperature_report.txt")):
        score = parse_score(txt_file.read_text(errors="replace"))
        if score is not None:
            break
    if score is None:
        for txt_file in sorted(task_out.rglob("*.txt")):
            score = parse_score(txt_file.read_text(errors="replace"))
            if score is not None:
                break
    return {"column": col, "score": score, "returncode": p.returncode, "elapsed_sec": round(time.time() - t0, 1)}


def eval_reading(model_path: pathlib.Path, tag: str, env: Dict[str, str], out_root: pathlib.Path,
                 reading_data: str) -> Dict[str, Any]:
    task_out = out_root / "outputs" / tag / "Reading"
    log = out_root / "logs" / tag / "Reading.log"
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--data_path", reading_data,
        "--revision_name", f"step026_{tag}_Reading",
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    try:
        p = run_cmd(cmd, env, log)
    except subprocess.TimeoutExpired:
        return {"Reading": None, "error": "timeout"}
    scores = parse_reading((p.stdout or "") + "\n" + (p.stderr or ""))
    if not scores:
        for rpt in sorted(task_out.rglob("report.txt")):
            scores = parse_reading(rpt.read_text(errors="replace"))
            if scores:
                break
    return {**scores, "returncode": p.returncode, "elapsed_sec": round(time.time() - t0, 1), "data_path": reading_data}


def evaluate_one(model_path: pathlib.Path, tag: str, gpu: int, out_root: pathlib.Path,
                 alpha: Optional[float], reading_data: str, only_reading: bool = False) -> Dict[str, Any]:
    src = model_path
    model_path = materialize_scaled_model(model_path, alpha, out_root, tag) if alpha is not None else model_path
    env = setup_env(out_root, tag, gpu)
    results: Dict[str, Any] = {"source_model_path": str(src), "executed_model_path": str(model_path), "tag": tag, "gpu": gpu, "alpha": alpha}
    scores: Dict[str, Optional[float]] = {}
    if not only_reading:
        for col, task, data, batch in TASKS:
            print(f"[{tag}] Evaluating {col}...", flush=True)
            r = eval_sentence(model_path, col, task, data, batch, tag, env, out_root)
            scores[col] = r.get("score")
            results[col] = r
            print(f"  {col} = {r.get('score')} ({r.get('elapsed_sec', '?')}s)", flush=True)
        gp = scores.get("GlobalPIQA_parallel")
        gnp = scores.get("GlobalPIQA_nonparallel")
        scores["GlobalPIQA_mean"] = (gp + gnp) / 2.0 if gp is not None and gnp is not None else (gp if gp is not None else gnp)
    print(f"[{tag}] Evaluating Reading...", flush=True)
    rr = eval_reading(model_path, tag, env, out_root, reading_data)
    scores.update({"Reading": rr.get("Reading"), "Reading_eye": rr.get("Reading_eye"), "Reading_self_paced": rr.get("Reading_self_paced")})
    results["Reading"] = rr
    print(f"  Reading = {rr.get('Reading')} ({rr.get('elapsed_sec', '?')}s)", flush=True)
    if only_reading:
        # If an older full result exists, merge its non-reading columns.
        pass
    eq_vals = [scores.get(k) for k in EQ7_KEYS]
    valid = [v for v in eq_vals if v is not None]
    if valid:
        scores["equal_valid_mean"] = sum(valid) / len(valid)
        scores["n_valid_equal_columns"] = len(valid)
    results["scores"] = scores
    out_file = out_root / f"{tag}_eval.json"
    out_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": "eval_complete", "tag": tag, "scores": scores, "out": str(out_file)}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=None, help="Optional executed private adapter scale to materialize and evaluate")
    parser.add_argument("--out_root", default=str(OUT_DEFAULT))
    parser.add_argument("--reading_data", default="evaluation_data/fast_eval/reading/reading_data.csv")
    parser.add_argument("--only_reading", action="store_true")
    args = parser.parse_args()
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    evaluate_one(pathlib.Path(args.model_path), args.tag, args.gpu, out_root, args.alpha, args.reading_data, args.only_reading)


if __name__ == "__main__":
    main()
