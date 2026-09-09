#!/usr/bin/env python3
"""research: official Entity evaluation for seed43222 mechanism arms.

This is a mechanism-instrument evaluation, not a leaderboard run.  It evaluates
available seed43222 VIEW / REPEAT / CLEAN DeBERTa checkpoints on the official BabyLM
Strict Entity task at 80M/90M/100M, saving prediction files for the same depth,
relevant-update, stale-option, and later cue-use readouts used on seeds 43022/43122.

The script is intentionally arm-selectable so a completed VIEW arm can be evaluated
while REPEAT/CLEAN are still training; rerunning later with all arms skips completed
chunks and fills the remaining payloads.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
from collections import defaultdict
import sys
import time
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
STRICT = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict"
ENTITY_DATA = STRICT / "evaluation_data" / "full_eval" / "entity_tracking"
NLP_DATA_ROOT = ROOT / "experiments/archive" / 'initial_model_studies' / "data" / "nltk_data"
OUT = WS / "data" / "seed43222_entity_eval"
RUNS = WS / "training" / "runs"
ARM_CONFIGS = {
    "D_V_43222": {
        "run_dir": RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43222",
        "data_arm": "view",
        "seed": 43222,
        "description": "DeBERTa MAX VIEW seed43222 trained in relation_learning research",
    },
    "D_R_43222": {
        "run_dir": RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43222",
        "data_arm": "repeat",
        "seed": 43222,
        "description": "DeBERTa MAX REPEAT seed43222 trained in relation_learning research",
    },
    "D_C_43222": {
        "run_dir": RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222",
        "data_arm": "clean",
        "seed": 43222,
        "description": "DeBERTa MAX CLEAN seed43222 trained in relation_learning research",
    },
}
CKS = ["chck_80M", "chck_90M", "chck_100M"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_sentence_score(text: str) -> float | None:
    for pat in [
        r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            return val if math.isfinite(val) and -5.0 <= val <= 105.0 else None
    return None


def payload_path(arm: str, ck: str) -> pathlib.Path:
    return OUT / "per_target" / f"step009_{arm}_{ck}.json"


def output_dir(arm: str, ck: str) -> pathlib.Path:
    return OUT / "outputs" / f"step009_{arm}_{ck}" / "Entity"


def load_or_make_payload(arm: str, ck: str) -> dict[str, Any]:
    p = payload_path(arm, ck)
    if p.exists():
        obj = read_json(p)
        obj.setdefault("tasks", {})
        return obj
    cfg = ARM_CONFIGS[arm]
    model_path = cfg["run_dir"] / "hf_model" / ck
    obj = {
        "target": f"step009_{arm}_{ck}",
        "arm": arm,
        "data_arm": cfg["data_arm"],
        "seed": cfg["seed"],
        "description": cfg["description"],
        "checkpoint": ck,
        "model_path": rel(model_path),
        "created_utc": now(),
        "tasks": {},
        "no_training_upload_leaderboard": True,
        "mechanism_instrument_not_submission": True,
    }
    write_json(p, obj)
    return obj


def task_done(payload: dict[str, Any]) -> bool:
    rec = payload.get("tasks", {}).get("Entity", {})
    return isinstance(rec, dict) and rec.get("returncode") == 0 and rec.get("score") is not None and pathlib.Path(ROOT / str(rec.get("predictions", ""))).exists()


def gpu_env(arm: str, ck: str, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if NLP_DATA_ROOT.exists():
        env["NLTK_DATA"] = str(NLP_DATA_ROOT.resolve())
    cache = OUT / "hf_cache" / f"{arm}_{ck}"
    tmp = OUT / "tmp" / f"{arm}_{ck}"
    for key, p in {
        "HF_HOME": cache,
        "HF_HUB_CACHE": cache / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": tmp,
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        env[key] = str(p.resolve())
    return env


def eval_one(arm: str, ck: str, gpu: int, timeout: int) -> dict[str, Any]:
    cfg = ARM_CONFIGS[arm]
    model_path = cfg["run_dir"] / "hf_model" / ck
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    payload = load_or_make_payload(arm, ck)
    if task_done(payload):
        rec = payload["tasks"]["Entity"]
        print(f"[SKIP] {arm} {ck} Entity already score={rec.get('score')}", flush=True)
        return {"status": "skip", "arm": arm, "checkpoint": ck, "score": rec.get("score")}
    out_dir = output_dir(arm, ck)
    log_path = OUT / "logs" / f"step009_{arm}_{ck}_Entity.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    revision = f"step009_{arm}_{ck}_Entity"
    argv = [
        sys.executable, "-B", "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", "entity_tracking",
        "--data_path", str(ENTITY_DATA.resolve()),
        "--revision_name", revision,
        "--save_predictions",
        "--batch_size", "128",
        "--non_causal_batch_size", "64",
        "--output_dir", str(out_dir.resolve()),
    ]
    print(f"[RUN] {arm} {ck} Entity on GPU {gpu}", flush=True)
    t0 = time.time(); rc = -999; err = None
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "start", "utc": now(), "arm": arm, "checkpoint": ck, "gpu": gpu}) + "\n")
            proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=gpu_env(arm, ck, gpu), stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -124; err = f"timeout after {timeout}s"
    elapsed = round(time.time() - t0, 1)
    report_files = sorted(out_dir.rglob("best_temperature_report.txt"), key=lambda p: p.stat().st_mtime)
    pred_files = sorted(out_dir.rglob("predictions.json"), key=lambda p: p.stat().st_mtime)
    score = None
    if report_files:
        score = parse_sentence_score(report_files[-1].read_text(encoding="utf-8", errors="replace"))
    rec = {
        "column": "Entity",
        "returncode": rc,
        "elapsed_sec": elapsed,
        "gpu": gpu,
        "score": score,
        "log": rel(log_path),
        "report": rel(report_files[-1]) if report_files else None,
        "predictions": rel(pred_files[-1]) if pred_files else None,
    }
    if err:
        rec["error"] = err
    payload.setdefault("tasks", {})["Entity"] = rec
    payload["updated_utc"] = now()
    payload["stable_scores"] = {"Entity": score}
    write_json(payload_path(arm, ck), payload)
    print(f"[DONE] {arm} {ck} score={score} rc={rc} elapsed={elapsed}s pred={rec['predictions']}", flush=True)
    return {"status": "done" if score is not None and rc == 0 else "failed", "arm": arm, "checkpoint": ck, "score": score, "elapsed_sec": elapsed}


def summarize(arms: list[str], checkpoints: list[str]) -> dict[str, Any]:
    rows = []
    for arm in arms:
        for ck in checkpoints:
            p = payload_path(arm, ck)
            if not p.exists():
                continue
            obj = read_json(p)
            rec = obj.get("tasks", {}).get("Entity", {})
            rows.append({
                "arm": arm,
                "seed": obj.get("seed"),
                "data_arm": obj.get("data_arm"),
                "checkpoint": ck,
                "score": rec.get("score"),
                "returncode": rec.get("returncode"),
                "predictions": rec.get("predictions"),
                "payload": rel(p),
            })
    scores_by_arm = defaultdict(list)
    for r in rows:
        if r.get("score") is not None:
            scores_by_arm[r["arm"]].append(float(r["score"]))
    late = {arm: statistics.mean(vals) for arm, vals in scores_by_arm.items() if vals}
    return {
        "status": "SEED43222_ENTITY_EVAL_SUMMARY",
        "finished_utc": now(),
        "rows": rows,
        "late_mean_entity_score": late,
        "out_dir": rel(OUT),
    }


def write_summary_csv(summary: dict[str, Any]) -> None:
    p = OUT / "seed43222_entity_eval_rows.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = summary.get("rows", [])
    if not rows:
        p.write_text("\n", encoding="utf-8")
        return
    with p.open("w", newline="", encoding="utf-8") as f:
        fields = ["arm", "seed", "data_arm", "checkpoint", "score", "returncode", "predictions", "payload"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["D_V_43222"], choices=list(ARM_CONFIGS))
    ap.add_argument("--checkpoints", nargs="+", default=CKS)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "SEED43222_ENTITY_EVAL_PLAN", "arms": args.arms, "checkpoints": args.checkpoints, "gpu": args.gpu, "chunks": len(args.arms) * len(args.checkpoints), "out_dir": rel(OUT)}
    for arm in args.arms:
        for ck in args.checkpoints:
            mp = ARM_CONFIGS[arm]["run_dir"] / "hf_model" / ck
            plan.setdefault("model_exists", {})[f"{arm}_{ck}"] = mp.exists()
    (OUT / "seed43222_entity_eval_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return
    results = []
    for arm in args.arms:
        for ck in args.checkpoints:
            results.append(eval_one(arm, ck, args.gpu, args.timeout))
    summary = summarize(args.arms, args.checkpoints)
    summary["results"] = results
    write_json(OUT / "seed43222_entity_eval_summary.json", summary)
    write_summary_csv(summary)
    print(json.dumps({"status": "SEED43222_ENTITY_EVAL_DONE", "summary": rel(OUT / "seed43222_entity_eval_summary.json"), "rows": rel(OUT / "seed43222_entity_eval_rows.csv"), "late_mean_entity_score": summary["late_mean_entity_score"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
