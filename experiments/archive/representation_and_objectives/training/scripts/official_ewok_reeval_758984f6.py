#!/usr/bin/env python3
"""Generic official-coordinate EWoK re-evaluation for research+ models.

Runs BabyLM sentence_zero_shot EWoK on the pristine current 7,618-row
`ewok_filtered` coordinate reconstructed in research, instead of the older local
INITIAL_MODEL_STUDIES EWoK copy used by the inherited fast/full evaluator. This script is a
minimal non-training repair: it changes only the EWoK evaluation data coordinate
and preserves the supplied local MLM checkpoint and tokenizer.
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
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/representation_and_objectives"
# Use the INITIAL_MODEL_STUDIES code checkout for sentence_zero_shot execution; feed it the
# pristine current official EWoK data directory explicitly.
STRICT_CODE = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
PRISTINE_EWOK = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_") or "model"


def parse_sentence_score(text: str) -> float | None:
    for pat in [
        r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if math.isfinite(val) and -5.0 <= val <= 105.0:
                return val
            return None
    return None


def read_report_score(task_out: Path) -> tuple[float | None, str | None]:
    reports = sorted(task_out.rglob("best_temperature_report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(reports):
        val = parse_sentence_score(p.read_text(encoding="utf-8", errors="replace"))
        if val is not None:
            return val, str(p)
    return None, None


def count_prediction_items(pred_path: Path | None) -> int | None:
    if pred_path is None:
        return None
    try:
        pdata: Any = json.loads(pred_path.read_text(encoding="utf-8"))
        if not isinstance(pdata, dict):
            return None
        return sum(len(v.get("predictions", [])) if isinstance(v, dict) else 0 for v in pdata.values())
    except Exception:
        return None


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model-path", type=Path, required=True, help="Local MLM checkpoint directory, usually .../hf_model/chck_100M")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--tag", default="strictsmalltok", help="Short tag used in revision/log/output filenames")
    ap.add_argument("--dry-run", action="store_true", help="Verify paths and EWoK row counts without loading the model")
    args = ap.parse_args()

    t0 = time.time()
    tag = sanitize_tag(args.tag)
    model_path = args.model_path.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not STRICT_CODE.exists():
        raise FileNotFoundError(STRICT_CODE)
    if not PRISTINE_EWOK.exists():
        raise FileNotFoundError(PRISTINE_EWOK)

    domain_counts = {}
    for p in sorted(PRISTINE_EWOK.glob("*.jsonl")):
        with p.open("r", encoding="utf-8", errors="replace") as f:
            domain_counts[p.stem] = sum(1 for _ in f)
    total_rows = sum(domain_counts.values())
    preflight = {
        "created_utc": now_utc(),
        "model_path": file_record(model_path),
        "strict_code": str(STRICT_CODE),
        "pristine_ewok_dir": str(PRISTINE_EWOK),
        "pristine_ewok_domain_counts": domain_counts,
        "pristine_ewok_total_rows": total_rows,
        "tag": tag,
    }
    if args.dry_run:
        out_json = out_root / f"official_ewok_reeval_{tag}_dryrun.json"
        payload = {"status": "OFFICIAL_EWOK_REEVAL_DRYRUN", "preflight": preflight, "elapsed_sec": time.time() - t0}
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "total_rows": total_rows, "model_exists": model_path.exists(), "out_json": str(out_json)}, indent=2), flush=True)
        return

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if total_rows != 7618:
        raise RuntimeError({"error": "unexpected_pristine_ewok_total_rows", "total_rows": total_rows, "domain_counts": domain_counts})

    hf = out_root / "hf_cache"
    tmp = out_root / "tmp"
    env = os.environ.copy()
    env["HF_HOME"] = str((hf / "home").resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TMPDIR"] = str(tmp.resolve())
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for key in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TMPDIR"]:
        Path(env[key]).mkdir(parents=True, exist_ok=True)

    revision = f"official_ewok_{tag}"
    task_out = out_root / "official_outputs" / "EWoK"
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / f"ewok_reeval_{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path),
        "--backend", "mlm",
        "--task", "ewok",
        "--data_path", str(PRISTINE_EWOK.resolve()),
        "--revision_name", revision,
        "--save_predictions",
        "--batch_size", "64",
        "--non_causal_batch_size", "64",
        "--output_dir", str(task_out.resolve()),
    ]
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now_utc()}] $ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=str(STRICT_CODE.resolve()), env=env, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=3600)

    score, report_path = read_report_score(task_out)
    preds = sorted(task_out.rglob("predictions.json"), key=lambda p: (p.stat().st_mtime, str(p)))
    pred_path = preds[-1] if preds else None
    pred_total = count_prediction_items(pred_path)
    validation_errors = []
    if proc.returncode != 0:
        validation_errors.append(f"returncode={proc.returncode}")
    if pred_total != 7618:
        validation_errors.append(f"prediction_item_total={pred_total} != 7618")
    if score is None:
        validation_errors.append("score_not_parsed")

    payload = {
        "status": "OFFICIAL_EWOK_REEVAL_DONE" if not validation_errors else "OFFICIAL_EWOK_REEVAL_FAILED",
        "created_utc": now_utc(),
        "tag": tag,
        "model_path": str(model_path),
        "strict_code": str(STRICT_CODE),
        "pristine_ewok_dir": str(PRISTINE_EWOK),
        "pristine_ewok_domain_counts": domain_counts,
        "pristine_ewok_total_rows": total_rows,
        "returncode": proc.returncode,
        "official_ewok_score": score,
        "report_path": report_path,
        "predictions_path": str(pred_path) if pred_path else None,
        "prediction_item_total": pred_total,
        "validation_errors": validation_errors,
        "log": str(log),
        "elapsed_sec": time.time() - t0,
        "interpretation": "EWoK re-scored on the current pristine 7,618-row official coordinate. Use predictions_path as --pristine-ewok-predictions for research pristine collation.",
    }
    out_json = out_root / f"official_ewok_reeval_{tag}.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "returncode": proc.returncode,
        "official_ewok_score": score,
        "pristine_ewok_total_rows": total_rows,
        "prediction_item_total": pred_total,
        "predictions_path": str(pred_path) if pred_path else None,
        "validation_errors": validation_errors,
        "out_json": str(out_json),
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2), flush=True)
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
