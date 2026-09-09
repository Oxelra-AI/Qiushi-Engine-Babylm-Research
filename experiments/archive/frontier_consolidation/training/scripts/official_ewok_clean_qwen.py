#!/usr/bin/env python3
"""research: current-official 7618-row EWoK re-evaluation for inherited clean-Qwen seeds.

This is a narrow robustness/coordinate repair evaluator, not training and not a
full official surface. It evaluates only EWoK at 100M for the two inherited
clean-Qwen checkpoints, using the same pristine current-official 7618-row
`ewok_filtered` data directory used for compact_view_reinvest.

Scientific decision it supports:
  - If clean-Qwen current-official EWoK remains near the earlier values, then
    compact_view_reinvest has a positive within-seed EWoK treatment effect in
    both seeds (seed43022 and seed43122), even if absolute Overall robustness is
    limited by inherited seed variance.
  - If clean-Qwen current-official EWoK rises enough to erase seed43122's
    treatment effect, then the EWoK mechanism is weaker than the repaired
    research read and semantic-view/source-selection repair becomes more urgent.

Minimum reliable method:
  - only two 100M checkpoints;
  - only one official task family (EWoK);
  - no finetuning, no AoA, no SuperGLUE, no full evaluation, no new training.

Example from the repository root:
  python3 experiments/archive/frontier_consolidation/training/scripts/official_ewok_clean_qwen.py \
    --seeds 43022 43122 --gpu 0

Use --dry-run for path/command validation without loading models.
"""
from __future__ import annotations

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


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", choices=["43022", "43122"], default=["43022", "43122"])
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--non-causal-batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true", help="validate paths and write planned commands without running EWoK")
    ap.add_argument("--out-root", type=Path, default=None, help="optional output root; default QIUSHI_AI_LAB_RUN_DIR or workspace data preflight")
    return ap.parse_args()


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


def read_report_score(task_out: Path) -> tuple[float | None, str | None]:
    reports = sorted(task_out.rglob("best_temperature_report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(reports):
        score = parse_sentence_score(p.read_text(encoding="utf-8", errors="replace"))
        if score is not None:
            return score, str(p)
    return None, None


def count_prediction_items(pred_path: Path | None) -> int | None:
    if not pred_path or not pred_path.exists():
        return None
    try:
        payload: Any = json.loads(pred_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        total = 0
        for v in payload.values():
            if isinstance(v, dict) and isinstance(v.get("predictions"), list):
                total += len(v["predictions"])
        return total
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    t0 = time.time()
    user_root = Path.cwd()
    session = user_root / "experiments/archive" / 'frontier_consolidation'
    strict = user_root / "experiments/archive" / 'initial_model_studies' / "repos" / "babylm-eval" / "strict"
    pristine_ewok = user_root / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict" / "evaluation_data" / "full_eval" / "ewok_filtered"
    model_paths = {
        "43022": user_root / "experiments/archive" / 'compact_experience' / "training" / "runs" / "qwen_clean_aligned_16k_seed43022" / "hf_model" / "chck_100M",
        "43122": user_root / "experiments/archive" / 'compact_experience' / "training" / "runs" / "qwen_clean_aligned_16k_seed43122" / "hf_model" / "chck_100M",
    }

    if args.out_root is not None:
        out_root = args.out_root
    elif os.environ.get("QIUSHI_AI_LAB_RUN_DIR"):
        out_root = Path(os.environ["QIUSHI_AI_LAB_RUN_DIR"])
    else:
        out_root = session / "data" / "official_ewok_clean_qwen_preflight"
    out_root.mkdir(parents=True, exist_ok=True)

    checks = {
        "strict_exists": strict.exists(),
        "pristine_ewok_exists": pristine_ewok.exists(),
        "model_paths_exist": {seed: p.exists() for seed, p in model_paths.items()},
    }
    missing = []
    if not strict.exists():
        missing.append(str(strict))
    if not pristine_ewok.exists():
        missing.append(str(pristine_ewok))
    for seed in args.seeds:
        if not model_paths[seed].exists():
            missing.append(str(model_paths[seed]))
    if missing:
        raise FileNotFoundError("Missing required paths: " + "; ".join(missing))

    domain_counts = {}
    for p in sorted(pristine_ewok.glob("*.jsonl")):
        with p.open("r", encoding="utf-8", errors="replace") as f:
            domain_counts[p.stem] = sum(1 for _ in f)
    total_rows = sum(domain_counts.values())

    run_records = []
    for seed in args.seeds:
        seed_out = out_root / "official_outputs" / f"clean_qwen_seed{seed}" / "EWoK"
        seed_out.mkdir(parents=True, exist_ok=True)
        hf = out_root / "hf_cache" / seed
        tmp = out_root / "tmp" / seed
        for path in [hf / "home", hf / "hub", hf / "datasets", hf / "transformers", hf / "modules", tmp]:
            path.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["HF_HOME"] = str((hf / "home").resolve())
        env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
        env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
        env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
        env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
        env["TMPDIR"] = str(tmp.resolve())
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        revision = f"official_ewok_clean_qwen_seed{seed}"
        cmd = [
            sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(model_paths[seed].resolve()),
            "--backend", "mlm",
            "--task", "ewok",
            "--data_path", str(pristine_ewok.resolve()),
            "--revision_name", revision,
            "--save_predictions",
            "--batch_size", str(args.batch_size),
            "--non_causal_batch_size", str(args.non_causal_batch_size),
            "--output_dir", str(seed_out.resolve()),
        ]
        log = out_root / "logs" / f"official_ewok_clean_qwen_seed{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "seed": seed,
            "model_path": str(model_paths[seed]),
            "output_dir": str(seed_out),
            "log": str(log),
            "command": cmd,
            "dry_run": bool(args.dry_run),
        }
        if args.dry_run:
            record.update({"returncode": None, "official_ewok_score": None, "predictions_path": None, "prediction_item_total": None})
        else:
            with log.open("a", encoding="utf-8") as f:
                f.write(f"\n[{now_utc()}] $ {' '.join(cmd)}\n")
                proc = subprocess.run(cmd, cwd=str(strict.resolve()), env=env, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=3600)
            score, report_path = read_report_score(seed_out)
            preds = sorted(seed_out.rglob("predictions.json"), key=lambda p: (p.stat().st_mtime, str(p)))
            pred_path = preds[-1] if preds else None
            record.update({
                "returncode": proc.returncode,
                "official_ewok_score": score,
                "report_path": report_path,
                "predictions_path": str(pred_path) if pred_path else None,
                "prediction_item_total": count_prediction_items(pred_path),
            })
        run_records.append(record)

    scores = {r["seed"]: r.get("official_ewok_score") for r in run_records if r.get("official_ewok_score") is not None}
    payload = {
        "status": "OFFICIAL_EWOK_CLEAN_QWEN_DRY_RUN" if args.dry_run else "OFFICIAL_EWOK_CLEAN_QWEN_DONE",
        "created_utc": now_utc(),
        "purpose": "Current-official 7618-row EWoK coordinate for inherited clean-Qwen 100M seeds; supports within-seed compact_view_reinvest treatment-effect interpretation.",
        "decision_rule": {
            "positive_replication": "If clean seed43122 current-official EWoK is below reinvest seed43122 official EWoK=51.8917, the EWoK treatment effect remains positive in seed43122.",
            "mechanism_weakened": "If clean seed43122 current-official EWoK is near or above 51.8917, the repaired seed43122 EWoK treatment effect is not positive and semantic-view/source-selection repair becomes more urgent.",
            "seed_gap_reference": "Compare clean seed43122-minus-seed43022 against reinvest official gap -1.6449 to estimate treatment-specific EWoK seed excess under current official coordinate.",
        },
        "strict_repo": str(strict),
        "pristine_ewok_dir": str(pristine_ewok),
        "pristine_ewok_domain_counts": domain_counts,
        "pristine_ewok_total_rows": total_rows,
        "gpu": str(args.gpu),
        "batch_size": args.batch_size,
        "non_causal_batch_size": args.non_causal_batch_size,
        "checks": checks,
        "records": run_records,
        "score_delta_43122_minus_43022": (scores.get("43122") - scores.get("43022")) if all(k in scores for k in ["43022", "43122"]) else None,
        "elapsed_sec": time.time() - t0,
    }
    out_json = out_root / "official_ewok_clean_qwen_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "pristine_ewok_total_rows": total_rows,
        "records": [{k: r.get(k) for k in ["seed", "dry_run", "returncode", "official_ewok_score", "prediction_item_total"]} for r in run_records],
        "score_delta_43122_minus_43022": payload["score_delta_43122_minus_43022"],
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
