#!/usr/bin/env python3
"""research/148: readout for the shared-tokenizer role-switch/fixed 80M screen.

Predeclared natural-transfer question (research plan, research shared-tokenizer update):
Route B continues only if the role-switch arm, relative to the structurally matched
role-fixed arm, moves the natural hard surfaces (GlobalPIQA_parallel hard-52
ranks/margins and EWoK stable conditional reversals) while preserving broad cheap7.
Synthetic packet transfer is no longer a success condition.

This wrapper is intentionally only an evaluation/readout launcher.  It trains nothing.
It repairs the draft interface bug by calling the official-compatible
evaluator with its valid default arm (`reinvest`) while overriding --target,
--run-dir and --endpoint for each local checkpoint.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
A02_SCRIPTS = USER_ROOT / "experiments/archive" / 'frontier_consolidation' / "scripts"
OUT_ROOT = A01_WS / "data" / "screen_readout"

TARGETS: dict[str, dict[str, Any]] = {
    "role_switch_80M": {
        "run_dir": A01_WS / "training/runs/role_switch_sharedtok_80M_legal40k_seed43022",
        "endpoint": "chck_80M",
        "description": "role-switch in-place packed replacement corpus; shared legal tokenizer trained on common 9,977,920-word intersection; 80M from-scratch natural-transfer screen",
    },
    "role_fixed_80M": {
        "run_dir": A01_WS / "training/runs/role_fixed_sharedtok_80M_legal40k_seed43022",
        "endpoint": "chck_80M",
        "description": "role-fixed structurally matched replacement corpus control; shared legal tokenizer trained on common 9,977,920-word intersection; 80M from-scratch natural-transfer screen",
    },
    "anchor_fixed256_80M": {
        "run_dir": A01_WS / "training/runs/legal40k_accum_compact_view_reinvest_seed43022",
        "endpoint": "chck_80M",
        "description": "matched legal40k fixed-256 compact-view AdamW anchor at 80M",
    },
}
EVALUATOR = A02_SCRIPTS / "evaluate_compliant_endpoint.py"
CHEAP_COLUMNS = [
    "BLiMP",
    "Supplement",
    "EWoK",
    "Entity",
    "COMPS",
    "GlobalPIQA_parallel",
    "GlobalPIQA_nonparallel",
    "Reading",
]
CHEAP7_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def model_path(meta: dict[str, Any]) -> Path:
    return Path(meta["run_dir"]) / "hf_model" / str(meta["endpoint"])


def check_ready(selected: list[str]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for label in selected:
        meta = TARGETS[label]
        mp = model_path(meta)
        run_dir = Path(meta["run_dir"])
        status[label] = {
            "run_dir": str(run_dir),
            "endpoint": meta["endpoint"],
            "model_path": str(mp),
            "model_ready": bool(mp.exists() and (mp / "model.safetensors").exists()),
            "scientific_metrics_ready": bool((run_dir / "scientific_metrics.json").exists()),
        }
    status["selected_ready"] = all(v.get("model_ready") and v.get("scientific_metrics_ready") for k, v in status.items() if isinstance(v, dict))
    status["arms_ready"] = all(status[a].get("model_ready") and status[a].get("scientific_metrics_ready") for a in ["role_switch_80M", "role_fixed_80M"] if a in selected)
    return status


def extract_score_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return flat cheap scores from the evaluator payload.

    The official-compatible evaluator stores raw task outputs under `tasks` and a
    flat score dictionary under `official_overall.scores`.  When SuperGLUE/AoA are
    not evaluated those entries remain null; cheap7 is the mean of the seven
    the fixed cheap-column panel.
    """
    if not isinstance(payload, dict):
        return {}
    scores: dict[str, Any] = {}
    oo = payload.get("official_overall") or {}
    if isinstance(oo, dict) and isinstance(oo.get("scores"), dict):
        for k, v in oo["scores"].items():
            scores[k] = v
    tasks = payload.get("tasks") or {}
    for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        if scores.get(k) is None and isinstance(tasks.get(k), dict):
            scores[k] = tasks[k].get("score")
    if scores.get("GlobalPIQA_parallel") is None and isinstance(tasks.get("GlobalPIQA_parallel"), dict):
        scores["GlobalPIQA_parallel"] = tasks["GlobalPIQA_parallel"].get("score")
    if scores.get("GlobalPIQA_nonparallel") is None and isinstance(tasks.get("GlobalPIQA_nonparallel"), dict):
        scores["GlobalPIQA_nonparallel"] = tasks["GlobalPIQA_nonparallel"].get("score")
    if scores.get("GlobalPIQA") is None:
        gp1 = scores.get("GlobalPIQA_parallel")
        gp2 = scores.get("GlobalPIQA_nonparallel")
        if isinstance(gp1, (int, float)) and isinstance(gp2, (int, float)):
            scores["GlobalPIQA"] = (gp1 + gp2) / 2.0
    if scores.get("Reading") is None and isinstance(tasks.get("Reading"), dict):
        rd = tasks["Reading"].get("scores") or {}
        if isinstance(rd, dict):
            scores["Reading"] = rd.get("Reading")
    vals = [scores.get(k) for k in CHEAP7_KEYS]
    if all(isinstance(v, (int, float)) for v in vals):
        scores["cheap7"] = sum(float(v) for v in vals) / len(vals)
    return scores


def run_cheap_eval(label: str, meta: dict[str, Any], out_dir: Path, gpu: int, force: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(meta["run_dir"])
    endpoint = str(meta["endpoint"])
    cmd = [
        sys.executable,
        "-B",
        str(EVALUATOR),
        "--arm",
        "reinvest",
        "--target",
        label,
        "--run-dir",
        str(run_dir),
        "--endpoint",
        endpoint,
        "--out-root",
        str(out_dir),
        "--gpu",
        str(gpu),
        "--columns",
        *CHEAP_COLUMNS,
    ]
    if force:
        cmd.append("--force")
    print(json.dumps({"event": "cheap_eval_start", "label": label, "cmd": " ".join(cmd)}), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / f"{label}_eval_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"{label}_eval_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    per_target = out_dir / "per_target" / f"{label}.json"
    payload = json.loads(per_target.read_text(encoding="utf-8")) if per_target.exists() else None
    return {
        "label": label,
        "exit_code": proc.returncode,
        "per_target": str(per_target),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "scores": extract_score_payload(payload),
        "payload": payload,
    }


def numeric_deltas(base: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in sorted(set(base) | set(target)):
        a = base.get(k)
        b = target.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[k] = float(b) - float(a)
    return out


def run_reader(script: Path, args: list[str], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", str(script), *args]
    print(json.dumps({"event": "reader_start", "script": str(script), "cmd": " ".join(cmd)}), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / f"{script.stem}_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"{script.stem}_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    return {"script": str(script), "exit_code": proc.returncode, "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ready", "cheap", "hard", "all"], default="ready")
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=sorted(TARGETS))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ewok-device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ready = check_ready(args.targets)
    (out_root / "readiness.json").write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "readiness", "ready": ready}, indent=2), flush=True)
    if args.stage == "ready":
        return
    missing = [k for k, v in ready.items() if isinstance(v, dict) and (not v.get("model_ready") or not v.get("scientific_metrics_ready"))]
    if missing:
        raise RuntimeError({"not_ready": missing, "readiness": ready})

    summary: dict[str, Any] = {
        "status": "SCREEN_READOUT",
        "created_utc": now_utc(),
        "readiness": ready,
        "cheap_eval": {},
        "cheap_deltas": {},
        "hard_readers": {},
        "interpretation": "Route B continues only if role_switch improves natural hard surfaces above role_fixed while broad cheap7 is preserved; synthetic packet transfer is not a success condition.",
    }

    if args.stage in {"cheap", "all"}:
        cheap_dir = out_root / "cheap_eval"
        for label in args.targets:
            summary["cheap_eval"][label] = run_cheap_eval(label, TARGETS[label], cheap_dir, args.gpu, args.force)
        if "role_switch_80M" in summary["cheap_eval"] and "role_fixed_80M" in summary["cheap_eval"]:
            summary["cheap_deltas"]["role_switch_minus_role_fixed"] = numeric_deltas(
                summary["cheap_eval"]["role_fixed_80M"].get("scores", {}),
                summary["cheap_eval"]["role_switch_80M"].get("scores", {}),
            )
        if "anchor_fixed256_80M" in summary["cheap_eval"]:
            base = summary["cheap_eval"]["anchor_fixed256_80M"].get("scores", {})
            for label, rec in summary["cheap_eval"].items():
                if label != "anchor_fixed256_80M":
                    summary["cheap_deltas"][f"{label}_minus_anchor_fixed256_80M"] = numeric_deltas(base, rec.get("scores", {}))

    if args.stage in {"hard", "all"}:
        gp_script = A01_WS / "scripts/globalpiqa_margin_reader.py"
        ew_script = A01_WS / "scripts/ewok_interaction_reader.py"
        summary["hard_readers"]["globalpiqa"] = run_reader(
            gp_script,
            ["--targets", *args.targets, "--modes", "parallel", "nonparallel", "--out-root", str(out_root / "globalpiqa")],
            out_root / "logs",
        )
        summary["hard_readers"]["ewok"] = run_reader(
            ew_script,
            ["--targets", *args.targets, "--out-root", str(out_root / "ewok"), "--device", args.ewok_device],
            out_root / "logs",
        )

    summary_path = out_root / "screen_readout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary_path": str(summary_path), "cheap_deltas": summary["cheap_deltas"], "hard_readers": summary["hard_readers"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
