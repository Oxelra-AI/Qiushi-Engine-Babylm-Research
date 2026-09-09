#!/usr/bin/env python3
"""research official Entity readout for the tight-private relation operator.

The 480-map tight-private checkpoint learned the controlled literal relation family,
but its direct transfer to BabyLM Entity is unknown.  This script runs the official
Entity evaluator on the scale-0.2 checkpoint, then invokes the existing research
operation-structure readout against chck82 and coherent86 references.
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

ROOT = _public_path('.')
STRICT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')
ENTITY_DATA = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')
NLP_DATA_ROOT = _public_path('experiments/archive/initial_model_studies/data/nltk_data')
MODEL = _public_path('experiments/archive/relation_learning/data/tight_private_pilot/checkpoint')
OUT_DEFAULT = _public_path('experiments/archive/relation_learning/data/tight_private_entity_eval')
research = _public_path('experiments/archive/relation_learning/scripts/entity_recency_diagnostic.py')
CHCK82_PRED = _public_path('experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/eval/official_outputs/scale1p75_seed43022_reference_chck_82M/Entity/chck_82M/full_scale1p75_seed43022_reference_chck_82M_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json')
COHERENT86_PRED = _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/official_outputs/coherent86_private_scale_0p75/Entity/final/full_coherent86_private_scale_0p75_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_score(text: str) -> float | None:
    for pat in [r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)"]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if math.isfinite(val) and -5 <= val <= 105:
                return val
    return None


def eval_env(out_dir: pathlib.Path, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if NLP_DATA_ROOT.exists():
        env["NLTK_DATA"] = str(_public_path('experiments/archive/initial_model_studies/data/nltk_data'))
    cache = out_dir / "cache"
    tmp = out_dir / "tmp"
    for k, p in {
        "HF_HOME": cache / "home",
        "HF_HUB_CACHE": cache / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": tmp,
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        env[k] = str(p.resolve())
    return env


def run_entity(out_dir: pathlib.Path, gpu: int, timeout: int, force: bool = False) -> dict[str, Any]:
    label = "tight_private_scale0p2"
    target_dir = out_dir / "official_outputs" / label / "Entity"
    log_path = out_dir / "logs" / f"{label}_Entity.log"
    payload_path = out_dir / "per_target" / f"{label}.json"
    if payload_path.exists() and not force:
        old = json.loads(payload_path.read_text(encoding="utf-8"))
        rec = old.get("entity_eval", old)
        pred = rec.get("predictions")
        if rec.get("returncode") == 0 and rec.get("score") is not None and pred and (ROOT / pred).exists():
            print(json.dumps({"event": "skip_existing", "score": rec.get("score"), "predictions": pred}), flush=True)
            return rec
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-B", "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(_public_path('experiments/archive/relation_learning/data/tight_private_pilot/checkpoint')),
        "--backend", "mlm",
        "--task", "entity_tracking",
        "--data_path", str(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')),
        "--revision_name", label,
        "--save_predictions",
        "--batch_size", "128",
        "--non_causal_batch_size", "64",
        "--output_dir", str(target_dir.resolve()),
    ]
    print(json.dumps({"event": "entity_eval_start", "model": rel(MODEL), "gpu": gpu, "utc": now(), "cmd": cmd}), flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "start", "utc": now(), "model": rel(MODEL)}) + "\n")
        try:
            p = subprocess.run(cmd, cwd=str(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')), env=eval_env(out_dir, gpu), stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            rc = p.returncode
            error = None
        except subprocess.TimeoutExpired:
            rc = -124
            error = f"timeout after {timeout}s"
    report_files = sorted(target_dir.rglob("best_temperature_report.txt"), key=lambda p: p.stat().st_mtime)
    pred_files = sorted(target_dir.rglob("predictions.json"), key=lambda p: p.stat().st_mtime)
    score = parse_score(report_files[-1].read_text(encoding="utf-8", errors="replace")) if report_files else None
    rec = {
        "status": "ok" if rc == 0 and score is not None and pred_files else "failed",
        "label": label,
        "model_path": rel(MODEL),
        "returncode": rc,
        "score": score,
        "elapsed_sec": round(time.time() - t0, 1),
        "gpu": gpu,
        "log": rel(log_path),
        "report": rel(report_files[-1]) if report_files else None,
        "predictions": rel(pred_files[-1]) if pred_files else None,
        "reference_note": "scale0p2 tight-private checkpoint trained on controlled counterfactual relation facts; mechanism transfer readout, not legal submission endpoint",
    }
    if error:
        rec["error"] = error
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps({"target": label, "entity_eval": rec, "updated_utc": now()}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "entity_eval_done", **rec}, ensure_ascii=False), flush=True)
    return rec


def run_step084(out_dir: pathlib.Path, pred_rel: str, baseline: str) -> dict[str, Any]:
    diag_dir = out_dir / f"recency_vs_{baseline}"
    cmd = [
        sys.executable, "-B", str(research),
        "--target", f"chck82,eval,{CHCK82_PRED}",
        "--target", f"coherent86,eval,{COHERENT86_PRED}",
        "--target", f"tight_private_scale0p2,eval,{ROOT / pred_rel}",
        "--baseline-label", baseline,
        "--out-dir", str(diag_dir),
    ]
    print(json.dumps({"event": "recency_start", "baseline": baseline, "cmd": cmd}), flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=600)
    if p.returncode != 0:
        print(p.stdout[-2000:], flush=True)
        print(p.stderr[-2000:], flush=True)
        raise RuntimeError(f"research recency failed baseline={baseline} rc={p.returncode}")
    return {"baseline": baseline, "out_dir": rel(diag_dir), "summary": rel(diag_dir / "summary.md"), "stdout_tail": p.stdout[-1000:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (_public_path('experiments/archive/relation_learning/data/tight_private_pilot/checkpoint/config.json')).exists():
        raise SystemExit(f"missing model {MODEL}")
    rec = run_entity(out_dir, args.gpu, args.timeout, force=args.force)
    diags = []
    if rec.get("predictions"):
        diags.append(run_step084(out_dir, rec["predictions"], "chck82"))
        diags.append(run_step084(out_dir, rec["predictions"], "coherent86"))
    summary = {"status": "TIGHT_PRIVATE_ENTITY_DONE", "created_utc": now(), "entity_eval": rec, "diagnostics": diags}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
