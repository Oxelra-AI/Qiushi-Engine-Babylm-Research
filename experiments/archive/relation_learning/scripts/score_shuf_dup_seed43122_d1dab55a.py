#!/usr/bin/env python3
"""research: score SHUF and selected-DUP seed43122 replication arms.

This wrapper defines the corrected arm identities and delegates mechanism scoring
to the validated research/032 infrastructure. Run it after the training
tasks deliver terminal results:

- SHUF: qwen_shuffled_control_16k_seed43122
- selected-DUP: selected_original_dup_all_16k_seed43122

The obsolete research official_original_dup run is excluded because research's
identity audit showed that it is not the report's DUP construction.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_shuf_dup_seed43122_d1dab55a.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
COMPACT_EXPERIENCE = ROOT / "experiments/archive/compact_experience"
STRICT = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
ENTITY_DATA = STRICT / "evaluation_data/full_eval/entity_tracking"
NLP_DATA_ROOT = ROOT / "experiments/archive/initial_model_studies/data/nltk_data"
sys.path.insert(0, str(WS / "scripts"))

import score_paired_context_relation_design as mech  # noqa: E402
import ordinary_heldout_price_probe as ordinary  # noqa: E402

OUT = WS / "data/shuf_dup_seed43122_probe"
NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/039_shuf_dup_seed43122_probe.md')
ORD_OUT = WS / "data/shuf_dup_seed43122_ordinary_heldout"
ORD_NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/039_shuf_dup_seed43122_ordinary_heldout.md')
CHECKPOINTS = ["chck_80M", "chck_90M", "chck_100M"]

ARMS: dict[str, dict[str, Any]] = {
    "OFF": {
        "role": "OFF",
        "description": "seed43122 official_lengthmatched baseline (no local qwen-pair practice)",
        "run": COMPACT_EXPERIENCE / "training/runs/official_lengthmatched_16k_seed43122",
        "per_target": COMPACT_EXPERIENCE / "data/full_eval/per_target/official_lengthmatched_seed43122.json",
    },
    "SHUF": {
        "role": "SHUF",
        "description": "seed43122 qwen_shuffled_control local original+wrong rewrite; same stream as seed43022 SHUF, new seed pair",
        "run": WS / "training/runs/qwen_shuffled_control_16k_seed43122",
        "per_target": OUT / "per_target/qwen_shuffled_control_seed43122.json",
    },
    "DUP": {
        "role": "DUP",
        "description": "seed43122 selected_original_dup_all local selected original+original duplication; correct replication of report DUP arm",
        "run": WS / "training/runs/selected_original_dup_all_16k_seed43122",
        "per_target": OUT / "per_target/selected_original_dup_all_seed43122.json",
    },
}
CONTRASTS = [
    ("SHUFminusOFF", "SHUF", "OFF"),
    ("DUPminusOFF", "DUP", "OFF"),
    ("DUPminusSHUF", "DUP", "SHUF"),
    ("SHUFminusDUP", "SHUF", "DUP"),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def model_ckpt(run_dir: pathlib.Path, ck: str) -> pathlib.Path:
    return run_dir / "hf_model" / ck


def ckpt_ready(run_dir: pathlib.Path, ck: str) -> bool:
    d = model_ckpt(run_dir, ck)
    return (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()


def task_done(payload: dict[str, Any]) -> bool:
    rec = payload.get("tasks", {}).get("Entity", {})
    pred = rec.get("predictions")
    return bool(isinstance(rec, dict) and rec.get("returncode") == 0 and rec.get("score") is not None and pred and (ROOT / str(pred)).exists())


def parse_score(text: str) -> float | None:
    for pat in [r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)"]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            return val if math.isfinite(val) and -5.0 <= val <= 105.0 else None
    return None


def eval_env(arm: str) -> dict[str, str]:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if NLP_DATA_ROOT.exists():
        env["NLTK_DATA"] = str(NLP_DATA_ROOT.resolve())
    cache = OUT / "hf_cache" / arm
    tmp = OUT / "tmp" / arm
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


def eval_entity_payload(arm: str, gpu: int, timeout: int) -> dict[str, Any]:
    cfg = ARMS[arm]
    payload_path = pathlib.Path(cfg["per_target"])
    if payload_path.exists():
        payload = read_json(payload_path)
        if task_done(payload):
            return {"arm": arm, "status": "already_present", "payload": rel(payload_path), "score": payload.get("tasks", {}).get("Entity", {}).get("score")}
    model_path = model_ckpt(pathlib.Path(cfg["run"]), "chck_100M")
    if not ckpt_ready(pathlib.Path(cfg["run"]), "chck_100M"):
        raise FileNotFoundError(model_path)
    out_dir = OUT / "official_outputs" / arm / "Entity" / "chck_100M"
    log_path = OUT / "logs" / f"entity_{arm}_chck_100M.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    revision = f"step039_{arm}_seed43122_Entity"
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
    env = eval_env(arm)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "start", "utc": now(), "arm": arm, "gpu": gpu, "model_path": rel(model_path)}) + "\n")
        try:
            proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -124
    report_files = sorted(out_dir.rglob("best_temperature_report.txt"), key=lambda p: p.stat().st_mtime)
    pred_files = sorted(out_dir.rglob("predictions.json"), key=lambda p: p.stat().st_mtime)
    score = parse_score(report_files[-1].read_text(encoding="utf-8", errors="replace")) if report_files else None
    payload = {
        "target": f"{arm.lower()}_seed43122",
        "description": cfg["description"],
        "family": "debertav2_8x480_16k_compact_experience_seed43122_relation_replication",
        "run_dir": rel(pathlib.Path(cfg["run"])),
        "model_path": rel(model_path),
        "endpoint": "chck_100M",
        "created_utc": now(),
        "gpu": gpu,
        "tasks": {
            "Entity": {
                "column": "Entity",
                "task": "entity_tracking",
                "data_path": rel(ENTITY_DATA),
                "revision_name": revision,
                "output_dir": rel(out_dir),
                "returncode": rc,
                "elapsed_sec": round(time.time() - t0, 3),
                "log": rel(log_path),
                "score": score,
                "predictions": rel(pred_files[-1]) if pred_files else None,
                "report": rel(report_files[-1]) if report_files else None,
            }
        },
        "stable_scores": {"Entity": score},
        "no_leaderboard_submission": True,
    }
    write_json(payload_path, payload)
    if not task_done(payload):
        raise RuntimeError({"entity_eval_failed": arm, "payload": rel(payload_path), "returncode": rc, "score": score, "log": rel(log_path)})
    return {"arm": arm, "status": "done", "payload": rel(payload_path), "score": score, "elapsed_sec": payload["tasks"]["Entity"]["elapsed_sec"]}


def ensure_entity_payloads(gpu: int, timeout: int) -> list[dict[str, Any]]:
    rows = []
    for arm in ["SHUF", "DUP"]:
        rows.append(eval_entity_payload(arm, gpu, timeout))
    write_json(OUT / "entity_eval_results.json", {"status": "ENTITY_EVAL_RESULTS", "created_utc": now(), "rows": rows})
    return rows


def readiness() -> dict[str, Any]:
    out: dict[str, Any] = {"arms": {}, "all_model_checkpoints_ready": True, "entity_payloads_ready": True}
    for arm, cfg in ARMS.items():
        ck = {c: ckpt_ready(pathlib.Path(cfg["run"]), c) for c in CHECKPOINTS}
        arm_ready = all(ck.values())
        pt_exists = pathlib.Path(cfg["per_target"]).exists()
        pt_done = False
        if pt_exists:
            try:
                pt_done = task_done(read_json(pathlib.Path(cfg["per_target"])))
            except Exception:
                pt_done = False
        out["arms"][arm] = {
            "run": rel(pathlib.Path(cfg["run"])),
            "description": cfg["description"],
            "checkpoints": ck,
            "all_checkpoints_ready": arm_ready,
            "per_target": rel(pathlib.Path(cfg["per_target"])),
            "per_target_exists": pt_exists,
            "entity_payload_done": pt_done,
            "needs_entity_eval_after_training": arm in {"SHUF", "DUP"} and not pt_done,
        }
        out["all_model_checkpoints_ready"] = out["all_model_checkpoints_ready"] and arm_ready
        out["entity_payloads_ready"] = out["entity_payloads_ready"] and pt_done
    return out


def configure_mechanism_base() -> None:
    mech.DEFAULT_OUT = OUT
    mech.DEFAULT_NOTE = NOTE
    mech.CKPTS = CHECKPOINTS
    mech.ARMS = ARMS
    mech.CONTRASTS = CONTRASTS


def configure_ordinary_base() -> None:
    ordinary.OUT = ORD_OUT
    ordinary.NOTE = ORD_NOTE
    ordinary.CKPTS = CHECKPOINTS
    ordinary.ARM_CONFIGS = {k: pathlib.Path(cfg["run"]) for k, cfg in ARMS.items()}
    ordinary.ROLE = {k: k for k in ARMS}
    ordinary.CONTRASTS = [("SHUF", "OFF"), ("DUP", "OFF"), ("DUP", "SHUF"), ("SHUF", "DUP")]


def write_plan() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "SHUF_DUP_SEED43122_SCORING_PLAN",
        "purpose": "Score corrected SHUF and selected-DUP seed43122 replications against pre-result rules.",
        "prestate": rel((_PUBLIC_ROOT / 'research/notes/relation_learning/shuf_dup_prestate.md')),
        "dup_identity_audit": rel((_PUBLIC_ROOT / 'research/notes/relation_learning/dup_identity_audit.md')),
        "arms": {k: {"description": v["description"], "run": rel(pathlib.Path(v["run"])), "per_target": rel(pathlib.Path(v["per_target"]))} for k, v in ARMS.items()},
        "checkpoints": CHECKPOINTS,
        "contrasts": [c for c, _, _ in CONTRASTS],
        "readiness": readiness(),
        "outputs": {
            "mechanism_out": rel(OUT),
            "mechanism_note": rel(NOTE),
            "ordinary_out": rel(ORD_OUT),
            "ordinary_note": rel(ORD_NOTE),
        },
    }
    write_json(OUT / "scoring_plan.json", plan)
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true", help="Write/readiness plan only; do not score.")
    ap.add_argument("--skip-ordinary", action="store_true", help="Run mechanism probes only.")
    ap.add_argument("--skip-mechanism", action="store_true", help="Run ordinary held-out only.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ordinary-batch-size", type=int, default=128)
    ap.add_argument("--entity-gpu", type=int, default=0)
    ap.add_argument("--entity-timeout", type=int, default=2400)
    args, passthrough = ap.parse_known_args()

    plan = write_plan()
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.check_only:
        return
    rd = plan["readiness"]
    if not rd["all_model_checkpoints_ready"]:
        missing = {k: v for k, v in rd["arms"].items() if not v["all_checkpoints_ready"]}
        raise SystemExit(json.dumps({"error": "model_checkpoints_not_ready", "missing": missing}, indent=2))
    if not rd["entity_payloads_ready"]:
        ensure_entity_payloads(args.entity_gpu, args.entity_timeout)
        rd = readiness()
        if not rd["entity_payloads_ready"]:
            raise SystemExit(json.dumps({"error": "entity_payloads_still_missing", "readiness": rd}, indent=2))

    if not args.skip_mechanism:
        configure_mechanism_base()
        sys.argv = [sys.argv[0], "--arms", "OFF", "SHUF", "DUP", "--device", args.device, "--batch-size", str(args.batch_size), "--out-dir", str(OUT), "--note", str(NOTE)] + passthrough
        mech.main()
    if not args.skip_ordinary:
        configure_ordinary_base()
        sys.argv = [sys.argv[0], "--device", args.device, "--batch-size", str(args.ordinary_batch_size), "--arms", "OFF", "SHUF", "DUP"]
        ordinary.main()


if __name__ == "__main__":
    main()
