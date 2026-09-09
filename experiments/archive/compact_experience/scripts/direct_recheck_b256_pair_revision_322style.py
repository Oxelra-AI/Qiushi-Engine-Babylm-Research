#!/usr/bin/env python3
"""research exact-path direct recheck for the b256/fixed-seq pair.

This adapts INITIAL_MODEL_STUDIES's `current_best_direct_checkpoint_recheck.py` to the
COMPACT_EXPERIENCE b256/fixed-seq fixed-WWM and WWM->token chck_100M checkpoints.  The goal
is to localize the large Supplement discrepancy: if this research-style evaluator
still gives fixed-WWM Supplement around 57, the discrepancy is in training/runtime
or data/order; if it returns around 65, the research wrapper/evaluator path was the
problem.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path("experiments/archive/compact_experience")
ROOT_INITIAL_MODEL_STUDIES = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT_INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
RUN_BASE = ROOT / "training/runs"
CKPTS = {
    "compact_experience_b256_fixed_chck_100M": RUN_BASE / "wwm_fixed_100M_b256_seq256_seed43/hf_model/chck_100M",
    "compact_experience_b256_wwm_to_token_chck_100M": RUN_BASE / "wwm_to_token_100M_b256_seq256_seed43/hf_model/chck_100M",
}
OUT = ROOT / "data/b256_pair_eval/b256_pair_revision_322style_direct_recheck.json"
NOTE = (ROOT.parents[2] / 'research/notes/compact_experience/b256_pair_revision_322style_direct_recheck.md')
LOG = (ROOT.parents[2] / 'research/notes/compact_experience/13_b256_pair_step322style_direct_recheck.log')
OUTDIR_ROOT = ROOT / "data/b256_pair_eval/revision_322style_eval_outputs"

TASKS = [
    ("BLiMP", "blimp", "evaluation_data/fast_eval/blimp_fast", "blimp_fast"),
    ("Supplement", "blimp", "evaluation_data/fast_eval/supplement_fast", "supplement_fast"),
    ("EWoK", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", "ewok_fast"),
    ("Entity", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast", "entity_tracking_fast"),
    ("COMPS", "comps", "evaluation_data/full_eval/comps", "comps"),
]

INITIAL_MODEL_STUDIES_REF = pathlib.Path("experiments/archive/initial_model_studies/data/current_best_direct_checkpoint_recheck.json")
WRAPPER = ROOT / "data/b256_pair_eval/b256_pair_100M_eval_summary.json"


def env_setup() -> dict[str, str]:
    env = os.environ.copy()
    hf = ROOT / "data/b256_pair_eval/hf_cache_step322style"
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["NLTK_DATA"] = str((ROOT_INITIAL_MODEL_STUDIES / "data/nltk_data").resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def sha16(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def run_cmd(cmd: list[str], env: dict[str, str], logf, cwd: pathlib.Path | None = None) -> None:
    line = "$ " + " ".join(map(str, cmd))
    print(line, flush=True)
    logf.write("\n" + line + "\n")
    logf.flush()
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-1600:], flush=True)
    logf.write(p.stdout + f"\n[returncode={p.returncode}]\n")
    logf.flush()
    if p.returncode != 0:
        raise RuntimeError(line + "\n" + p.stdout[-6000:])


def parse_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if not m:
        raise RuntimeError(f"parse avg failed {report}")
    return float(m.group(1))


def parse_reading(report: pathlib.Path) -> dict[str, float]:
    txt = report.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {}
    for lab, key in [("EYE TRACKING SCORE", "reading_eye_tracking"), ("SELF-PACED READING SCORE", "reading_self_paced")]:
        m = re.search(re.escape(lab) + r":\s*([0-9.\-]+)", txt)
        if not m:
            raise RuntimeError(f"parse reading failed {report}")
        out[key] = float(m.group(1))
    out["Reading"] = (out["reading_eye_tracking"] + out["reading_self_paced"]) / 2.0
    return out


def eval_ckpt(label: str, ckpt: pathlib.Path, env: dict[str, str], logf) -> dict:
    if not ckpt.exists() or not (ckpt / "model.safetensors").exists():
        raise FileNotFoundError(ckpt)
    outdir = OUTDIR_ROOT / f"eval_step322_direct_{label}"
    scores: dict[str, float] = {}
    reports: dict[str, str] = {}
    for score_name, task, data_path, ds_name in TASKS:
        report = outdir / ckpt.name / "main" / "zero_shot" / "mlm" / task / ds_name / "best_temperature_report.txt"
        if not report.exists():
            run_cmd([
                sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
                "--model_path_or_name", str(ckpt.resolve()),
                "--backend", "mlm",
                "--task", task,
                "--data_path", data_path,
                "--save_predictions",
                "--batch_size", "64",
                "--output_dir", str(outdir.resolve()),
            ], env, logf, cwd=STRICT)
        scores[score_name] = parse_avg(report)
        reports[score_name] = str(report)
    rr = outdir / ckpt.name / "main" / "zero_shot" / "mlm" / "reading" / "report.txt"
    if not rr.exists():
        run_cmd([
            sys.executable, "-m", "evaluation_pipeline.reading.run",
            "--model_path_or_name", str(ckpt.resolve()),
            "--backend", "mlm",
            "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
            "--output_dir", str(outdir.resolve()),
        ], env, logf, cwd=STRICT)
    scores.update(parse_reading(rr))
    reports["Reading"] = str(rr)
    return {"path": str(ckpt), "sha16": sha16(ckpt / "model.safetensors"), "scores": scores, "reports": reports}


def weighted_proxy_fast(row: dict[str, float]) -> float:
    # This direct recheck has no GlobalPIQA, so use only mean6 for comparison with research.
    return sum(row[k] for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]) / 6.0


def main() -> None:
    t0 = time.time()
    env = env_setup()
    profiles: dict[str, dict] = {}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUTDIR_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as logf:
        logf.write("\n===== research research-style direct recheck =====\n")
        for label, ckpt in CKPTS.items():
            profiles[label] = eval_ckpt(label, ckpt, env, logf)
    fixed = profiles["compact_experience_b256_fixed_chck_100M"]["scores"]
    token = profiles["compact_experience_b256_wwm_to_token_chck_100M"]["scores"]
    keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading", "reading_eye_tracking", "reading_self_paced"]
    delta = {k: token[k] - fixed[k] for k in keys}
    mean6 = {
        "compact_experience_b256_fixed_chck_100M": weighted_proxy_fast(fixed),
        "compact_experience_b256_wwm_to_token_chck_100M": weighted_proxy_fast(token),
        "token_minus_fixed": weighted_proxy_fast(token) - weighted_proxy_fast(fixed),
    }
    initial_model_studies_ref = json.loads(INITIAL_MODEL_STUDIES_REF.read_text(encoding="utf-8")) if INITIAL_MODEL_STUDIES_REF.exists() else None
    wrapper = json.loads(WRAPPER.read_text(encoding="utf-8")) if WRAPPER.exists() else None
    payload = {
        "status": "B256_PAIR_STEP322STYLE_DIRECT_RECHECK_DONE",
        "method": "adapted INITIAL_MODEL_STUDIES research exact checkpoint-path evaluator; no revision_name; batch_size 64; same fast task paths",
        "profiles": profiles,
        "token_minus_fixed": delta,
        "mean6": mean6,
        "initial_model_studies_step322_reference": initial_model_studies_ref,
        "wrapper_summary_path": str(WRAPPER),
        "wrapper_summary_present": wrapper is not None,
        "elapsed_sec": time.time() - t0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — b256 pair research-style direct recheck",
        "",
        f"JSON: `{OUT}`",
        f"Log: `{LOG}`",
        "",
        "This uses the INITIAL_MODEL_STUDIES research invocation pattern: exact checkpoint directories as `--model_path_or_name`, no `revision_name`, batch size 64, and the same fast BLiMP/Supplement/EWoK/Entity/COMPS/Reading paths.",
        "",
        "| checkpoint | BLiMP | Supplement | EWoK | Entity | COMPS | Reading | mean6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ["compact_experience_b256_fixed_chck_100M", "compact_experience_b256_wwm_to_token_chck_100M"]:
        s = profiles[label]["scores"]
        lines.append(f"| {label} | {s['BLiMP']:.2f} | {s['Supplement']:.2f} | {s['EWoK']:.2f} | {s['Entity']:.2f} | {s['COMPS']:.2f} | {s['Reading']:.3f} | {mean6[label]:.3f} |")
    lines += [
        "",
        "## WWM→token minus fixed",
        "",
        "| BLiMP | Supplement | EWoK | Entity | COMPS | Reading | mean6 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {delta['BLiMP']:+.2f} | {delta['Supplement']:+.2f} | {delta['EWoK']:+.2f} | {delta['Entity']:+.2f} | {delta['COMPS']:+.2f} | {delta['Reading']:+.3f} | {mean6['token_minus_fixed']:+.3f} |",
        "",
    ]
    if initial_model_studies_ref:
        s100 = initial_model_studies_ref["profiles"]["wwm43_chck_100M"]["scores"]
        lines += [
            "## INITIAL_MODEL_STUDIES research reference fixed-WWM chck_100M",
            "",
            f"INITIAL_MODEL_STUDIES fixed-WWM chck_100M: BLiMP {s100['BLiMP']:.2f}, Supplement {s100['Supplement']:.2f}, EWoK {s100['EWoK']:.2f}, Entity {s100['Entity']:.2f}, COMPS {s100['COMPS']:.2f}, Reading {s100['Reading']:.3f}.",
            f"The fixed-WWM comparison research-style direct score differs by Supplement {fixed['Supplement'] - s100['Supplement']:+.2f}, Entity {fixed['Entity'] - s100['Entity']:+.2f}, Reading {fixed['Reading'] - s100['Reading']:+.3f}; this localizes the Supplement mismatch away from the research wrapper evaluator and toward training, software-environment or data-order differences unless another hidden environment factor is found.",
            "",
        ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "delta": delta, "mean6": mean6}, indent=2))


if __name__ == "__main__":
    main()
