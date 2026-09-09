#!/usr/bin/env python3
"""research: full Entity Tracking probe for the paired-alignment Wave-1 result.

The fast Wave-1 screen showed the central effect on Entity (+10.64 aligned-mismatched,
+6.50 aligned-baseline).  This probe reruns Entity Tracking on the full evaluation
set for aligned, mismatched, and the inherited INITIAL_MODEL_STUDIES baseline to check whether the
large signal is only a fast-subset artifact.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Dict, Optional

ROOT_INITIAL_MODEL_STUDIES = pathlib.Path("experiments/archive/initial_model_studies")
ROOT_COMPACT_EXPERIENCE = pathlib.Path("experiments/archive/compact_experience")
STRICT = ROOT_INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
OUT = ROOT_COMPACT_EXPERIENCE / "data/paired_alignment_full_entity"
NOTE = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/paired_alignment_full_entity_probe.md')

TARGETS = {
    "aligned": ROOT_COMPACT_EXPERIENCE / "training/runs/aligned_100M_seed43/hf_model/chck_100M",
    "mismatched": ROOT_COMPACT_EXPERIENCE / "training/runs/mismatched_100M_seed43/hf_model/chck_100M",
    "initial_model_baseline": ROOT_INITIAL_MODEL_STUDIES / "training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M",
}


def parse_score(text: str) -> Optional[float]:
    m = re.search(r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    for line in reversed(text.splitlines()):
        s = line.strip()
        m = re.match(r"^(?:[0-9.]+\s+)?([+-]?[0-9]+(?:\.[0-9]+)?)$", s)
        if m:
            val = float(m.group(1))
            if -1000 < val < 1000:
                return val
    vals = re.findall(r"(?:accuracy|score|acc|acc_norm)[^0-9+\-]*([+-]?[0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if vals:
        val = float(vals[-1])
        return val * (100 if 0 <= val <= 1 else 1)
    return None


def read_report_score(task_out: pathlib.Path) -> Optional[float]:
    candidates = sorted(task_out.rglob("best_temperature_report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    if not candidates:
        candidates = sorted(task_out.rglob("*.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(candidates):
        score = parse_score(p.read_text(encoding="utf-8", errors="replace"))
        if score is not None:
            return score
    return None


def eval_one(name: str, model_path: pathlib.Path, gpu: int) -> Dict:
    env = os.environ.copy()
    hf = OUT / "hf_cache" / name
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NLTK_DATA"] = str((ROOT_INITIAL_MODEL_STUDIES / "data/nltk_data").resolve())
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)

    task_out = OUT / "eval_outputs" / name / "Entity_full"
    task_out.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "logs" / f"{name}_Entity_full.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", "entity_tracking",
        "--data_path", "evaluation_data/full_eval/entity_tracking",
        "--revision_name", f"step018_{name}_Entity_full",
        "--save_predictions",
        "--batch_size", "128",
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n")
    p = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    log_path.write_text("CMD: " + " ".join(cmd) + f"\nRC: {p.returncode}\nELAPSED: {elapsed:.1f}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}\n", encoding="utf-8")
    stdout_score = parse_score(p.stdout)
    report_score = read_report_score(task_out)
    return {
        "target": name,
        "model_path": str(model_path),
        "score": stdout_score if stdout_score is not None else report_score,
        "stdout_score": stdout_score,
        "report_score": report_score,
        "returncode": p.returncode,
        "elapsed_sec": round(elapsed, 3),
        "log": str(log_path),
        "output_dir": str(task_out),
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Dict] = {}
    for name, path in TARGETS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        print(json.dumps({"event": "eval_start", "target": name}), flush=True)
        rec = eval_one(name, path, args.gpu)
        records[name] = rec
        print(json.dumps({"event": "eval_done", "target": name, "score": rec.get("score"), "rc": rec.get("returncode")}), flush=True)
    scores = {k: v.get("score") for k, v in records.items()}
    deltas = {}
    if scores.get("aligned") is not None and scores.get("mismatched") is not None:
        deltas["aligned_minus_mismatched"] = scores["aligned"] - scores["mismatched"]
    if scores.get("aligned") is not None and scores.get("initial_model_baseline") is not None:
        deltas["aligned_minus_initial_model_studies_baseline"] = scores["aligned"] - scores["initial_model_baseline"]
    if scores.get("mismatched") is not None and scores.get("initial_model_baseline") is not None:
        deltas["mismatched_minus_initial_model_studies_baseline"] = scores["mismatched"] - scores["initial_model_baseline"]
    payload = {
        "status": "FULL_ENTITY_PROBE_DONE",
        "note": "Full Entity Tracking only, to check whether the Wave-1 fast Entity signal survives full data.",
        "scores": scores,
        "deltas": deltas,
        "records": records,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = OUT / "full_entity_probe.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — Full Entity Tracking probe",
        "",
        f"JSON: `{out_path}`",
        "",
        "| target | full Entity score |",
        "|---|---:|",
    ]
    for name in ["aligned", "mismatched", "initial_model_baseline"]:
        lines.append(f"| {name} | {scores.get(name):.3f} |" if scores.get(name) is not None else f"| {name} | NA |")
    lines += ["", "## Deltas", "", "| contrast | delta |", "|---|---:|"]
    for k, v in deltas.items():
        lines.append(f"| {k} | {v:.3f} |")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
