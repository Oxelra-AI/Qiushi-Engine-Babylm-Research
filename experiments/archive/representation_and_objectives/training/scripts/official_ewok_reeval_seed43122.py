#!/usr/bin/env python3
"""research: OFFICIAL-DATA EWoK re-evaluation for compact_view_reinvest seed43122.

This is a post-run repair wrapper for the independent seed43122 robustness model.
The research full-evaluation controller uses the older INITIAL_MODEL_STUDIES strict checkout for
zero-shot evaluation; research showed that checkout's `ewok_filtered` data has
6,666 rows, while the current pristine official coordinate regenerated 7,618
EWoK rows matching the unmodified official collator constants. Therefore, if the
managed seed43122 full evaluation returns with stale-local EWoK, run this wrapper
once to produce the minimal replacement EWoK predictions for the same endpoint.

Scientific purpose: place seed43122 on exactly the same official EWoK coordinate
as seed43022 without retraining or changing any non-EWoK prediction. This is the
lowest-cost reliable action needed before using seed43122 as robustness evidence.
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


WORKSPACE = _public_path('experiments/archive/representation_and_objectives')  # .../representation_and_objectives/workspace
STUDY = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
PRISTINE_EWOK = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered')
DEFAULT_MODEL_PATH = _public_path('experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_100M')
DEFAULT_OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/official_ewok_reeval_seed43122')


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def count_prediction_items(pred_path: Path) -> int | None:
    try:
        pdata: Any = json.loads(pred_path.read_text(encoding="utf-8"))
        if not isinstance(pdata, dict):
            return None
        return sum(len(v.get("predictions", [])) if isinstance(v, dict) else 0 for v in pdata.values())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()

    t0 = time.time()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
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

    if not STRICT.exists():
        raise FileNotFoundError(STRICT)
    if not PRISTINE_EWOK.exists():
        raise FileNotFoundError(PRISTINE_EWOK)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)

    domain_counts = {}
    for p in sorted(PRISTINE_EWOK.glob("*.jsonl")):
        with p.open("r", encoding="utf-8", errors="replace") as f:
            domain_counts[p.stem] = sum(1 for _ in f)
    total_rows = sum(domain_counts.values())

    revision = "official_ewok_seed43122"
    task_out = out_root / "official_outputs" / "EWoK"
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / "ewok_reeval_seed43122.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(args.model_path.resolve()),
        "--backend", "mlm",
        "--task", "ewok",
        "--data_path", str(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered')),
        "--revision_name", revision,
        "--save_predictions",
        "--batch_size", "64",
        "--non_causal_batch_size", "64",
        "--output_dir", str(task_out.resolve()),
    ]
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now_utc()}] $ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')), env=env, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=3600)

    score, report_path = read_report_score(task_out)
    preds = sorted(task_out.rglob("predictions.json"), key=lambda p: (p.stat().st_mtime, str(p)))
    pred_path = preds[-1] if preds else None
    pred_total = count_prediction_items(pred_path) if pred_path else None

    payload = {
        "status": "OFFICIAL_EWOK_SEED43122_REEVAL_DONE",
        "created_utc": now_utc(),
        "model_path": str(args.model_path),
        "pristine_ewok_dir": str(PRISTINE_EWOK),
        "pristine_ewok_domain_counts": domain_counts,
        "pristine_ewok_total_rows": total_rows,
        "returncode": proc.returncode,
        "official_ewok_score": score,
        "report_path": report_path,
        "predictions_path": str(pred_path) if pred_path else None,
        "prediction_item_total": pred_total,
        "log": str(log),
        "elapsed_sec": time.time() - t0,
        "interpretation": (
            "Seed43122 EWoK re-scored on the same pristine official 7618-row ewok_filtered data used for "
            "seed43022 research. Use predictions_path as --pristine-ewok-predictions for "
            "stage_pristine_collate_seed43122.py if the full seed43122 run used stale local EWoK."
        ),
    }
    out_json = out_root / "official_ewok_reeval_seed43122.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "returncode": proc.returncode,
        "official_ewok_score": score,
        "pristine_ewok_total_rows": total_rows,
        "prediction_item_total": pred_total,
        "predictions_path": str(pred_path) if pred_path else None,
        "out_json": str(out_json),
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
