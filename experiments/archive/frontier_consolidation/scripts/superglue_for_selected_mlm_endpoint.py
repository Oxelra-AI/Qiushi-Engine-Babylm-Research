#!/usr/bin/env python3
"""research: SuperGLUE evaluator for an already cheap7-scored DeBERTa MLM endpoint.

This is a generic endpoint-branch tool.  It should be used only after the
selected cheap7 grid identifies a candidate whose missing SuperGLUE column can
change the official Overall ranking.  It runs the same research official-compatible
SuperGLUE evaluator (with current primary metrics: F1 for MRPC/QQP, accuracy
otherwise), then combines the measured SuperGLUE with a selected-grid cheap row
and AoA=0 arithmetic.

It deliberately does not run AoA or leaderboard submission.
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
import subprocess
import sys
import time
from statistics import mean
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
EVAL_SCRIPT = WORKSPACE / "scripts/evaluate_compliant_endpoint.py"
CHCK82_VERIFY = WORKSPACE / "data/chck82_independent_verification/chck82_independent_verification.json"
DEFAULT_RUN_DIR = WORKSPACE / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def fnum(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        y = float(x)
    except Exception:
        return None
    if math.isnan(y):
        return None
    return y


def endpoint_words(endpoint: str) -> int | None:
    if endpoint.startswith("chck_") and endpoint.endswith("M"):
        try:
            return int(endpoint[len("chck_"):-1]) * 1_000_000
        except Exception:
            return None
    return None


def load_chck82_reference() -> dict[str, Any]:
    j = read_json(CHCK82_VERIFY)
    scores = {k: float(v) for k, v in j["score_arithmetic"]["scores"].items() if v is not None}
    cheap7 = mean(scores[c] for c in CHEAP_COLUMNS)
    return {
        "scores": scores,
        "cheap7": cheap7,
        "superglue": scores["SuperGLUE"],
        "aoa": scores.get("AoA", 0.0),
        "overall": float(j["score_arithmetic"]["overall_reported"]),
        "source": rel(CHCK82_VERIFY),
    }


def extract_superglue(payload: dict[str, Any]) -> float | None:
    official = payload.get("official_overall", {}).get("scores", {})
    if isinstance(official, dict) and official.get("SuperGLUE") is not None:
        return float(official["SuperGLUE"])
    sg = payload.get("tasks", {}).get("SuperGLUE")
    if isinstance(sg, dict):
        for key in ("superglue_mean", "superglue_primary_metric_mean", "score"):
            if sg.get(key) is not None:
                return float(sg[key])
    return None


def load_cheap_row(path: pathlib.Path, endpoint: str | None = None) -> dict[str, Any]:
    data = read_json(path)
    rows: list[dict[str, Any]]
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        rows = data["rows"]
    elif isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and data.get("scores"):
        # selected_trajectory_summary-like cheap single record
        row = {"endpoint": data.get("endpoint") or data.get("best_endpoint"), **data.get("scores", data.get("best_scores", {}))}
        if data.get("cheap7") is not None:
            row["cheap7"] = data["cheap7"]
        elif data.get("best_cheap7") is not None:
            row["cheap7"] = data["best_cheap7"]
        rows = [row]
    else:
        raise RuntimeError(f"Cannot parse cheap-row JSON shape: {path}")
    if endpoint:
        matches = [r for r in rows if str(r.get("endpoint")) == endpoint]
        if not matches:
            raise RuntimeError(f"Endpoint {endpoint} not found in {path}; available={[r.get('endpoint') for r in rows]}")
        row = dict(matches[0])
    else:
        if len(rows) != 1:
            raise RuntimeError("Cheap JSON has multiple rows; pass --endpoint")
        row = dict(rows[0])
    scores = {c: fnum(row.get(c)) for c in CHEAP_COLUMNS}
    missing = [c for c, v in scores.items() if v is None]
    if missing:
        raise RuntimeError(f"Cheap row missing columns {missing}: {row}")
    row["scores"] = {c: float(scores[c]) for c in CHEAP_COLUMNS}  # type: ignore[index]
    c7 = fnum(row.get("cheap7"))
    recomputed = mean(row["scores"][c] for c in CHEAP_COLUMNS)
    row["cheap7"] = float(c7 if c7 is not None else recomputed)
    row["cheap7_recomputed"] = float(recomputed)
    row["cheap7_delta_from_recomputed"] = float(row["cheap7"] - recomputed)
    return row


def compute_overall(cheap_scores: dict[str, float], superglue: float, aoa: float = 0.0) -> float:
    return float(mean([cheap_scores[c] for c in CHEAP_COLUMNS] + [float(superglue), float(aoa)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=pathlib.Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--target", required=True, help="Unique evaluation target label")
    ap.add_argument("--cheap-trajectory", type=pathlib.Path, required=True, help="selected_trajectory.json containing this endpoint row")
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    ap.add_argument("--collate-root", type=pathlib.Path, required=True)
    ap.add_argument("--summary-root", type=pathlib.Path, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.collate_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)

    ref = load_chck82_reference()
    cheap = load_cheap_row(args.cheap_trajectory, args.endpoint)
    model_path = args.run_dir / "hf_model" / args.endpoint
    endpoint_actual_words = None
    metrics_path = args.run_dir / "scientific_metrics.json"
    if metrics_path.exists():
        try:
            metrics = read_json(metrics_path)
            for rec in metrics.get("saved_checkpoints", []):
                if isinstance(rec, dict) and rec.get("name") == args.endpoint:
                    endpoint_actual_words = rec.get("actual_cumulative_word_exposure")
                    break
        except Exception:
            endpoint_actual_words = None

    preflight = {
        "status": "SUPERGLUE_SELECTED_ENDPOINT_PREFLIGHT",
        "created_utc": now(),
        "target": args.target,
        "run_dir": rel(args.run_dir),
        "endpoint": args.endpoint,
        "model_path": rel(model_path),
        "model_path_exists": model_path.exists(),
        "metrics_path": rel(metrics_path),
        "metrics_exists": metrics_path.exists(),
        "endpoint_target_words": endpoint_words(args.endpoint),
        "endpoint_actual_words_from_metrics": endpoint_actual_words,
        "cheap_trajectory": rel(args.cheap_trajectory),
        "cheap_row": cheap,
        "protected_chck82": ref,
        "thresholds": {
            "superglue_needed_to_beat_chck82_overall_with_aoa0": float(9.0 * ref["overall"] - 7.0 * cheap["cheap7"]),
            "superglue_needed_for_overall_42p0_with_aoa0": float(9.0 * 42.0 - 7.0 * cheap["cheap7"]),
            "superglue_needed_to_match_alpha075_projected_42p1210247099666_with_aoa0": float(9.0 * 42.1210247099666 - 7.0 * cheap["cheap7"]),
        },
        "command_semantics": "If run, evaluate only SuperGLUE via research; no AoA and no leaderboard submission.",
    }
    preflight_path = args.summary_root / f"{args.target}_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.preflight_only:
        print(json.dumps({"status": preflight["status"], "preflight_json": rel(preflight_path), "model_path_exists": model_path.exists(), "thresholds": preflight["thresholds"]}, indent=2), flush=True)
        return

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cache_root = args.summary_root / "runtime_cache" / args.target
    for k, p in {
        "HF_HOME": cache_root / "hf_home",
        "HF_HUB_CACHE": cache_root / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "HF_MODULES_CACHE": cache_root / "modules",
        "HF_DATASETS_CACHE": cache_root / "datasets",
        "TMPDIR": cache_root / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        env[k] = str(p.resolve())

    cmd = [
        sys.executable, "-B", str(EVAL_SCRIPT),
        "--arm", "reinvest",
        "--run-dir", str(args.run_dir),
        "--target", args.target,
        "--endpoint", args.endpoint,
        "--out-root", str(args.out_root),
        "--collate-root", str(args.collate_root),
        "--gpu", str(args.gpu),
        "--columns", "SuperGLUE",
    ]
    if args.force:
        cmd.append("--force")
    log_dir = args.summary_root / "logs" / args.target
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(json.dumps({"event": "superglue_start", "target": args.target, "endpoint": args.endpoint, "gpu": args.gpu, "cheap7": cheap["cheap7"], "thresholds": preflight["thresholds"], "utc": now()}), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=9000)
    (log_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    print(json.dumps({"event": "superglue_returned", "target": args.target, "returncode": proc.returncode, "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:]}, ensure_ascii=False), flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    payload_path = args.out_root / "per_target" / f"{args.target}.json"
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    payload = read_json(payload_path)
    sg = extract_superglue(payload)
    if sg is None:
        raise RuntimeError(f"SuperGLUE score missing in {payload_path}")
    overall = compute_overall(cheap["scores"], sg, 0.0)
    summary = {
        "status": "SUPERGLUE_SELECTED_ENDPOINT_COMPLETE",
        "created_utc": now(),
        "target": args.target,
        "run_dir": rel(args.run_dir),
        "endpoint": args.endpoint,
        "endpoint_target_words": endpoint_words(args.endpoint),
        "endpoint_actual_words_from_metrics": endpoint_actual_words,
        "cheap_trajectory": rel(args.cheap_trajectory),
        "cheap_scores": cheap["scores"],
        "cheap7": cheap["cheap7"],
        "cheap7_recomputed": cheap["cheap7_recomputed"],
        "superglue": sg,
        "aoa_assumed_for_projection": 0.0,
        "projected_overall_with_aoa0": overall,
        "protected_chck82": ref,
        "thresholds": preflight["thresholds"],
        "deltas_vs_chck82": {
            "cheap7": float(cheap["cheap7"] - ref["cheap7"]),
            "superglue": float(sg - ref["superglue"]),
            "projected_overall_with_aoa0": float(overall - ref["overall"]),
        },
        "deltas_vs_alpha075_projected": {
            "projected_overall_with_aoa0_minus_42p1210247099666": float(overall - 42.1210247099666),
        },
        "payload_path": rel(payload_path),
        "stdout_log": rel(log_dir / "stdout.log"),
        "stderr_log": rel(log_dir / "stderr.log"),
        "elapsed_sec": round(time.time() - t0, 1),
        "scientific_reading": "This completes the missing SuperGLUE column for a cheap7-selected endpoint. It is endpoint evidence only; it does not explain the compact-view mechanism and it does not submit to the leaderboard.",
    }
    out_json = args.summary_root / f"{args.target}_superglue_summary.json"
    out_md = args.summary_root / f"{args.target}_superglue_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(
        f"# research selected endpoint SuperGLUE — {args.target}\n\n"
        f"Endpoint: `{args.endpoint}`. Cheap7: `{cheap['cheap7']}`. SuperGLUE: `{sg}`.\n\n"
        f"Projected Overall(AoA=0): `{overall}` (delta vs chck82 `{overall - ref['overall']:+.6f}`; "
        f"delta vs alpha0.75 projected `{overall - 42.1210247099666:+.6f}`).\n\n"
        f"SuperGLUE thresholds from cheap7: beat chck82 `{preflight['thresholds']['superglue_needed_to_beat_chck82_overall_with_aoa0']:.6f}`, "
        f"reach 42.0 `{preflight['thresholds']['superglue_needed_for_overall_42p0_with_aoa0']:.6f}`, "
        f"match alpha0.75 `{preflight['thresholds']['superglue_needed_to_match_alpha075_projected_42p1210247099666_with_aoa0']:.6f}`.\n\n"
        f"Payload: `{rel(payload_path)}`\nJSON: `{rel(out_json)}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "target": args.target, "endpoint": args.endpoint, "superglue": sg, "projected_overall_with_aoa0": overall, "delta_vs_chck82": overall - ref["overall"], "summary_json": rel(out_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
