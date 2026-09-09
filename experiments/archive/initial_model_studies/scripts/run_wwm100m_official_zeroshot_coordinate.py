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
MODEL = (ROOT / "training/runs/babylm_fullcycle_wwm_seed42_100M/hf_model").resolve()
OUTDIR = (ROOT / "training/runs/babylm_fullcycle_wwm_seed42_100M/eval_results_official_zeroshot").resolve()
OUT_JSON = ROOT / "data/wwm100m_official_zeroshot_coordinate.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/72_wwm100m_official_zeroshot_coordinate.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/wwm100m_official_zeroshot_coordinate.log')
BACKEND = "mlm"
REVISION = "chck_100M"

FULL_EVAL = STRICT / "evaluation_data" / "full_eval"
TASKS = [
    ("blimp", "blimp", str(FULL_EVAL / "blimp_filtered"), "blimp_filtered"),
    ("supplement", "blimp", str(FULL_EVAL / "supplement_filtered"), "supplement_filtered"),
    ("ewok", "ewok", str(FULL_EVAL / "ewok_filtered"), "ewok_filtered"),
    ("entity_tracking", "entity_tracking", str(FULL_EVAL / "entity_tracking"), "entity_tracking"),
    ("comps", "comps", str(FULL_EVAL / "comps"), "comps"),
    ("global_piqa_parallel", "global_piqa_parallel", str(FULL_EVAL / "global_piqa_parallel"), "global_piqa_parallel"),
    ("global_piqa_nonparallel", "global_piqa_nonparallel", str(FULL_EVAL / "global_piqa_nonparallel"), "global_piqa_nonparallel"),
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
    if not m:
        # GlobalPIQA reports may still use the same format; keep a fallback.
        vals = re.findall(r"(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)", txt, flags=re.I)
        if vals:
            return float(vals[-1]) * (100.0 if float(vals[-1]) <= 1.0 else 1.0)
        raise RuntimeError(f"could not parse average accuracy from {report}\n{txt[:1000]}")
    return float(m.group(1))


def read_reading(report: pathlib.Path) -> dict[str, float]:
    txt = report.read_text(encoding="utf-8", errors="replace")
    out = {}
    for label, key in [("EYE TRACKING SCORE", "reading_eye_tracking"), ("SELF-PACED READING SCORE", "reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        if not m:
            raise RuntimeError(f"could not parse {label} from {report}\n{txt}")
        out[key] = float(m.group(1))
    return out


def main() -> None:
    env = setup_env()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    scores = {}
    reports = {}
    with LOG.open("a", encoding="utf-8") as logf:
        for col, task, data_path, dataset_name in TASKS:
            run([
                sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
                "--model_path_or_name", str(MODEL), "--backend", BACKEND, "--task", task,
                "--data_path", str(pathlib.Path(data_path).resolve()), "--save_predictions", "--revision_name", REVISION,
                "--batch_size", "64", "--output_dir", str(OUTDIR),
            ], env, cwd=STRICT, logf=logf)
            report = OUTDIR / "hf_model" / REVISION / "zero_shot" / BACKEND / task / dataset_name / "best_temperature_report.txt"
            reports[col] = str(report)
            scores[col] = read_avg(report)
        reading_csv = FULL_EVAL / "reading" / "reading_data.csv"
        run([
            sys.executable, "-m", "evaluation_pipeline.reading.run",
            "--model_path_or_name", str(MODEL), "--backend", BACKEND,
            "--data_path", str(reading_csv.resolve()),
            "--revision_name", REVISION, "--output_dir", str(OUTDIR),
        ], env, cwd=STRICT, logf=logf)
        reading_report = OUTDIR / "hf_model" / REVISION / "zero_shot" / BACKEND / "reading" / "report.txt"
        reports["reading"] = str(reading_report)
        scores.update(read_reading(reading_report))
    global_piqa = (scores["global_piqa_parallel"] + scores["global_piqa_nonparallel"]) / 2.0
    aggregate = {
        "model_path": str(MODEL),
        "revision": REVISION,
        "backend": BACKEND,
        "scores": scores,
        "derived_columns": {
            "GlobalPIQA_mean_parallel_nonparallel": global_piqa,
            "Reading_eye_tracking": scores["reading_eye_tracking"],
            "Reading_self_paced": scores["reading_self_paced"],
        },
        "reports": reports,
        "output_dir": str(OUTDIR),
        "note": "Zero-shot/reading coordinate only; AoA and (Super)GLUE fine-tuning still missing for full nine-column Overall.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — WWM 100M official zero-shot coordinate", "",
        f"Evidence JSON: `{OUT_JSON}`", "",
        "Final checkpoint: `chck_100M`, backend `mlm`.", "",
        "| column/task | score |", "|---|---:|",
        f"| BLiMP | {scores['blimp']:.2f} |",
        f"| Supplement | {scores['supplement']:.2f} |",
        f"| EWoK | {scores['ewok']:.2f} |",
        f"| Entity Tracking | {scores['entity_tracking']:.2f} |",
        f"| COMPS | {scores['comps']:.2f} |",
        f"| GlobalPIQA parallel | {scores['global_piqa_parallel']:.2f} |",
        f"| GlobalPIQA nonparallel | {scores['global_piqa_nonparallel']:.2f} |",
        f"| GlobalPIQA mean | {global_piqa:.2f} |",
        f"| Reading eye | {scores['reading_eye_tracking']:.2f} |",
        f"| Reading self-paced | {scores['reading_self_paced']:.2f} |",
        "", "AoA and (Super)GLUE are not included yet.",
    ]
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "WWM100M_ZERO_SHOT_COORDINATE_DONE", "out": str(OUT_JSON), "scores": scores, "global_piqa_mean": global_piqa}, indent=2))


if __name__ == "__main__":
    main()
