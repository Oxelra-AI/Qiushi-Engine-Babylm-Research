#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
FULL_EVAL = STRICT / "evaluation_data" / "full_eval"
MODEL = (ROOT / "training/runs/babylm_fullcycle_wwm_seed42_100M/hf_model").resolve()
OUTDIR = (ROOT / "training/runs/babylm_fullcycle_wwm_seed42_100M/eval_results_available_official").resolve()
PREV_OUTDIR = (ROOT / "training/runs/babylm_fullcycle_wwm_seed42_100M/eval_results_official_zeroshot").resolve()
OUT_JSON = ROOT / "data/wwm100m_available_official_coordinate.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/wwm100m_available_official_coordinate.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/wwm100m_available_official_coordinate.log')
BACKEND = "mlm"
REVISION = "chck_100M"

RUN_TASKS = [
    ("entity_tracking", "entity_tracking", FULL_EVAL / "entity_tracking", "entity_tracking"),
    ("comps", "comps", FULL_EVAL / "comps", "comps"),
    ("global_piqa_parallel", "global_piqa_parallel", FULL_EVAL / "global_piqa_parallel", "global_piqa_parallel"),
    ("global_piqa_nonparallel", "global_piqa_nonparallel", FULL_EVAL / "global_piqa_nonparallel", "global_piqa_nonparallel"),
]
PREVIOUS_REPORTS = {
    "blimp": PREV_OUTDIR / "hf_model" / REVISION / "zero_shot" / BACKEND / "blimp" / "blimp_filtered" / "best_temperature_report.txt",
    "supplement": PREV_OUTDIR / "hf_model" / REVISION / "zero_shot" / BACKEND / "blimp" / "supplement_filtered" / "best_temperature_report.txt",
}


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


def run(cmd: list[str], env: dict[str, str], cwd: pathlib.Path | None = None, logf=None) -> None:
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    if logf:
        logf.write("\n" + line + "\n"); logf.flush()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None)
    print(p.stdout[-3000:], flush=True)
    if logf:
        logf.write(p.stdout + f"\n[returncode={p.returncode}]\n"); logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"command failed with {p.returncode}: {' '.join(cmd)}\n{p.stdout[-8000:]}")


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if m:
        return float(m.group(1))
    vals = re.findall(r"(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)", txt, flags=re.I)
    if vals:
        v = float(vals[-1])
        return v * (100.0 if v <= 1.0 else 1.0)
    raise RuntimeError(f"could not parse average accuracy from {report}\n{txt[:1200]}")


def read_reading(report: pathlib.Path) -> dict[str, float]:
    txt = report.read_text(encoding="utf-8", errors="replace")
    out = {}
    for label, key in [("EYE TRACKING SCORE", "reading_eye_tracking"), ("SELF-PACED READING SCORE", "reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        if not m:
            raise RuntimeError(f"could not parse {label} from {report}\n{txt}")
        out[key] = float(m.group(1))
    return out


def ensure_available_paths() -> dict[str, object]:
    status = {}
    for col, _task, data_path, _dataset in RUN_TASKS:
        status[col] = {"path": str(data_path), "exists": data_path.exists(), "num_files": sum(1 for _ in data_path.rglob('*')) if data_path.exists() else 0}
    ewok = FULL_EVAL / "ewok_filtered"
    status["ewok_filtered"] = {"path": str(ewok), "exists": ewok.exists(), "num_files": sum(1 for _ in ewok.rglob('*')) if ewok.exists() else 0, "skipped_reason": "empty/missing local official data; handled separately"}
    for col, rp in PREVIOUS_REPORTS.items():
        status[f"previous_{col}_report"] = {"path": str(rp), "exists": rp.exists(), "bytes": rp.stat().st_size if rp.exists() else 0}
    return status


def main() -> None:
    env = setup_env()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    path_status = ensure_available_paths()
    for col, _task, data_path, _dataset in RUN_TASKS:
        if not data_path.exists() or not any(data_path.rglob('*')):
            raise RuntimeError(f"required available data path absent or empty for {col}: {data_path}")
    scores = {}
    reports = {}
    provenance = {}
    for col, report in PREVIOUS_REPORTS.items():
        if report.exists():
            scores[col] = read_avg(report)
            reports[col] = str(report)
            provenance[col] = "recovered_from_step72_failed_runner"
        else:
            provenance[col] = "missing_previous_report"
    with LOG.open("a", encoding="utf-8") as logf:
        for col, task, data_path, dataset_name in RUN_TASKS:
            run([
                sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
                "--model_path_or_name", str(MODEL), "--backend", BACKEND, "--task", task,
                "--data_path", str(data_path.resolve()), "--save_predictions", "--revision_name", REVISION,
                "--batch_size", "64", "--output_dir", str(OUTDIR),
            ], env, cwd=STRICT, logf=logf)
            report = OUTDIR / "hf_model" / REVISION / "zero_shot" / BACKEND / task / dataset_name / "best_temperature_report.txt"
            reports[col] = str(report)
            scores[col] = read_avg(report)
            provenance[col] = "fresh_run"
        reading_csv = FULL_EVAL / "reading" / "reading_data.csv"
        run([
            sys.executable, "-m", "evaluation_pipeline.reading.run",
            "--model_path_or_name", str(MODEL), "--backend", BACKEND,
            "--data_path", str(reading_csv.resolve()), "--revision_name", REVISION, "--output_dir", str(OUTDIR),
        ], env, cwd=STRICT, logf=logf)
        reading_report = OUTDIR / "hf_model" / REVISION / "zero_shot" / BACKEND / "reading" / "report.txt"
        reports["reading"] = str(reading_report)
        scores.update(read_reading(reading_report))
        provenance["reading"] = "fresh_run"
    if "global_piqa_parallel" in scores and "global_piqa_nonparallel" in scores:
        global_piqa_mean = (scores["global_piqa_parallel"] + scores["global_piqa_nonparallel"]) / 2.0
    else:
        global_piqa_mean = None
    available_cols = [k for k in ["blimp", "supplement", "entity_tracking", "comps", "global_piqa_parallel", "global_piqa_nonparallel", "reading_eye_tracking", "reading_self_paced"] if k in scores]
    aggregate = {
        "model_path": str(MODEL),
        "revision": REVISION,
        "backend": BACKEND,
        "scores": scores,
        "derived_columns": {
            "GlobalPIQA_mean_parallel_nonparallel": global_piqa_mean,
            "Reading_eye_tracking": scores.get("reading_eye_tracking"),
            "Reading_self_paced": scores.get("reading_self_paced"),
        },
        "reports": reports,
        "provenance": provenance,
        "path_status": path_status,
        "output_dir": str(OUTDIR),
        "missing_columns": ["EWoK", "AoA", "SuperGLUE"],
        "note": "Available official coordinate: BLiMP/Supplement recovered from research, Entity/COMPS/GlobalPIQA/Reading fresh research. EWoK local directory is empty; AoA and SuperGLUE not run yet.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — WWM 100M available official coordinate", "",
        f"Evidence JSON: `{OUT_JSON}`", "",
        "Final checkpoint: `chck_100M`, backend `mlm`.", "",
        "| column/task | score | provenance |", "|---|---:|---|",
    ]
    for key, label in [("blimp", "BLiMP"), ("supplement", "Supplement"), ("entity_tracking", "Entity Tracking"), ("comps", "COMPS"), ("global_piqa_parallel", "GlobalPIQA parallel"), ("global_piqa_nonparallel", "GlobalPIQA nonparallel")]:
        if key in scores:
            lines.append(f"| {label} | {scores[key]:.2f} | {provenance.get(key,'')} |")
    if global_piqa_mean is not None:
        lines.append(f"| GlobalPIQA mean | {global_piqa_mean:.2f} | derived |")
    if "reading_eye_tracking" in scores:
        lines.append(f"| Reading eye | {scores['reading_eye_tracking']:.2f} | fresh_run |")
    if "reading_self_paced" in scores:
        lines.append(f"| Reading self-paced | {scores['reading_self_paced']:.2f} | fresh_run |")
    lines += ["", "EWoK is intentionally missing because `evaluation_data/full_eval/ewok_filtered` exists but contains no JSONL files after official download; AoA and (Super)GLUE are not included in this runner."]
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "WWM100M_AVAILABLE_COORDINATE_DONE", "out": str(OUT_JSON), "available_columns": available_cols, "scores": scores, "global_piqa_mean": global_piqa_mean, "missing": aggregate["missing_columns"]}, indent=2))


if __name__ == "__main__":
    main()
