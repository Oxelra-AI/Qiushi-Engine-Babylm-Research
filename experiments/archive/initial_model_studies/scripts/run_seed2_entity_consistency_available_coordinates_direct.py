#!/usr/bin/env python3
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
FULL_EVAL = STRICT / "evaluation_data" / "full_eval"
BACKEND = "mlm"
RUNNER_NOTE = "Direct local checkpoint evaluation: model_path_or_name is hf_model/chck_*M; no local revision selection is used."
OUT_MANIFEST = ROOT / "data/seed2_entity_consistency_available_coordinate_manifest.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/seed2_entity_consistency_available_coordinate_manifest.md')

TASKS = [
    ("blimp", "blimp", FULL_EVAL / "blimp_filtered", "blimp_filtered"),
    ("supplement", "blimp", FULL_EVAL / "supplement_filtered", "supplement_filtered"),
    ("entity_tracking", "entity_tracking", FULL_EVAL / "entity_tracking", "entity_tracking"),
    ("comps", "comps", FULL_EVAL / "comps", "comps"),
    ("global_piqa_parallel", "global_piqa_parallel", FULL_EVAL / "global_piqa_parallel", "global_piqa_parallel"),
    ("global_piqa_nonparallel", "global_piqa_nonparallel", FULL_EVAL / "global_piqa_nonparallel", "global_piqa_nonparallel"),
]

JOBS = [
    ("consistency_10M", ROOT / "training/runs/babylm_seed2_entity_consistency_deberta_20M/hf_model/chck_10M"),
    ("consistency_20M", ROOT / "training/runs/babylm_seed2_entity_consistency_deberta_20M/hf_model/chck_20M"),
    ("shuffled_10M", ROOT / "training/runs/babylm_seed2_entity_shuffled_pair_deberta_20M/hf_model/chck_10M"),
    ("shuffled_20M", ROOT / "training/runs/babylm_seed2_entity_shuffled_pair_deberta_20M/hf_model/chck_20M"),
]


def setup_env() -> dict:
    env = os.environ.copy()
    hf_home = ROOT / "training/hf_home"
    env["HF_HOME"] = str(hf_home.resolve())
    env["HF_HUB_CACHE"] = str((hf_home / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((ROOT / "training/hf_modules_cache").resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def run(cmd: list[str], env: dict, logf) -> None:
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    logf.write("\n" + line + "\n")
    logf.flush()
    p = subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-3000:], flush=True)
    logf.write(p.stdout + f"\n[returncode={p.returncode}]\n")
    logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"failed {p.returncode}: {' '.join(cmd)}\n{p.stdout[-8000:]}")


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if not m:
        raise RuntimeError(f"cannot parse average from {report}\n{txt[:1000]}")
    return float(m.group(1))


def read_reading(report: pathlib.Path) -> dict[str, float]:
    txt = report.read_text(encoding="utf-8", errors="replace")
    out = {}
    for label, key in [("EYE TRACKING SCORE", "reading_eye_tracking"), ("SELF-PACED READING SCORE", "reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        if not m:
            raise RuntimeError(f"cannot parse {label} from {report}\n{txt}")
        out[key] = float(m.group(1))
    return out


def find_one(root: pathlib.Path, suffix_parts: list[str]) -> pathlib.Path:
    hits = []
    for p in root.rglob(suffix_parts[-1]):
        s = str(p)
        if all(part in s for part in suffix_parts[:-1]):
            hits.append(p)
    if len(hits) != 1:
        raise RuntimeError(f"expected one report under {root} for {suffix_parts}, got {len(hits)}: {[str(x) for x in hits[:10]]}")
    return hits[0]


def eval_job(key: str, model_path: pathlib.Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    env = setup_env()
    outdir = ROOT / f"training/runs/seed2_entity_consistency_eval_{key}"
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / f"notes/seed2_entity_consistency_eval_{key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    scores = {}; reports = {}
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(json.dumps({"job": key, "model_path": str(model_path), "note": RUNNER_NOTE}, indent=2) + "\n")
        for col, task, data_path, dataset_name in TASKS:
            if not data_path.exists() or not any(data_path.rglob("*")):
                raise RuntimeError(f"missing data {data_path}")
            run([sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
                 "--model_path_or_name", str(model_path.resolve()), "--backend", BACKEND, "--task", task,
                 "--data_path", str(data_path.resolve()), "--save_predictions", "--batch_size", "64",
                 "--output_dir", str(outdir.resolve())], env, logf)
            report = find_one(outdir, [task, dataset_name, "best_temperature_report.txt"])
            scores[col] = read_avg(report)
            reports[col] = str(report)
        reading_csv = FULL_EVAL / "reading" / "reading_data.csv"
        run([sys.executable, "-m", "evaluation_pipeline.reading.run", "--model_path_or_name", str(model_path.resolve()),
             "--backend", BACKEND, "--data_path", str(reading_csv.resolve()), "--output_dir", str(outdir.resolve())], env, logf)
        reading_report = find_one(outdir, ["reading", "report.txt"])
        scores.update(read_reading(reading_report)); reports["reading"] = str(reading_report)
    derived = {
        "GlobalPIQA_mean_parallel_nonparallel": (scores["global_piqa_parallel"] + scores["global_piqa_nonparallel"]) / 2.0,
        "Reading_mean_eye_self_paced": (scores["reading_eye_tracking"] + scores["reading_self_paced"]) / 2.0,
    }
    out_json = ROOT / f"data/seed2_entity_consistency_{key}_available_coordinate.json"
    out_note = ROOT / f"notes/105_seed2_entity_consistency_{key}_available_coordinate.md"
    payload = {"run_name": key, "model_path": str(model_path), "backend": BACKEND, "scores": scores, "derived_columns": derived, "reports": reports, "note": RUNNER_NOTE}
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# research seed2 available coordinate — {key}", "", f"Evidence JSON: `{out_json}`", "", "| column/task | score |", "|---|---:|"]
    for k, lab in [("blimp", "BLiMP"), ("supplement", "Supplement"), ("entity_tracking", "Entity"), ("comps", "COMPS"), ("global_piqa_parallel", "GlobalPIQA parallel"), ("global_piqa_nonparallel", "GlobalPIQA nonparallel")]:
        lines.append(f"| {lab} | {scores[k]:.2f} |")
    lines += [f"| GlobalPIQA mean | {derived['GlobalPIQA_mean_parallel_nonparallel']:.2f} |", f"| Reading eye | {scores['reading_eye_tracking']:.2f} |", f"| Reading self-paced | {scores['reading_self_paced']:.2f} |", f"| Reading mean | {derived['Reading_mean_eye_self_paced']:.2f} |"]
    out_note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"key": key, "model_path": str(model_path), "out_json": str(out_json), "out_note": str(out_note), "eval_dir": str(outdir), "log": str(log_path), "scores": scores, "derived_columns": derived}


def main() -> None:
    t0 = time.time()
    results = []
    for key, mp in JOBS:
        print(json.dumps({"event": "start", "key": key, "model_path": str(mp)}), flush=True)
        results.append(eval_job(key, mp))
    manifest = {"status": "complete", "note": RUNNER_NOTE, "elapsed_sec": time.time() - t0, "results": results}
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research — seed2 Entity consistency available-coordinate manifest", "", f"Evidence JSON: `{OUT_MANIFEST}`", "", RUNNER_NOTE, ""]
    for r in results:
        d = r["derived_columns"]; s = r["scores"]
        lines.append(f"- {r['key']}: BLiMP {s['blimp']:.2f}, Supp {s['supplement']:.2f}, Entity {s['entity_tracking']:.2f}, COMPS {s['comps']:.2f}, GPIQA {d['GlobalPIQA_mean_parallel_nonparallel']:.2f}, Reading {d['Reading_mean_eye_self_paced']:.2f}")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SEED2_ENTITY_AVAILABLE_COORDINATES_DONE", "manifest": str(OUT_MANIFEST), "elapsed_sec": time.time() - t0}, indent=2))

if __name__ == "__main__":
    main()
