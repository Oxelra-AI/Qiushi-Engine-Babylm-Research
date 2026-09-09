#!/usr/bin/env python3
"""research: integrity check for selected causal GPT cheap7 evaluations.

This verifies that the selected compact/repeat trajectories used for the causal
architecture-transfer decision are internally consistent: each endpoint has seven
cheap columns, cheap7 equals the mean of those columns, task prediction/report paths
exist, and model metadata is present. It does not run model evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import pathlib
from statistics import mean
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_one(label: str, root: pathlib.Path, traj_path: pathlib.Path) -> dict[str, Any]:
    rows = read_json(traj_path)
    out: dict[str, Any] = {"label": label, "root": str(root), "trajectory": str(traj_path), "n_rows": len(rows), "rows": []}
    bad = []
    for row in rows:
        ep = row.get("endpoint")
        rec: dict[str, Any] = {"endpoint": ep, "trajectory_cheap7": row.get("cheap7")}
        present_cols = [c for c in CHEAP_COLUMNS if row.get(c) is not None]
        rec["present_columns"] = present_cols
        rec["n_present_columns"] = len(present_cols)
        recomputed = mean(float(row[c]) for c in CHEAP_COLUMNS) if len(present_cols) == len(CHEAP_COLUMNS) else None
        rec["recomputed_cheap7"] = recomputed
        rec["cheap7_abs_error"] = abs(float(row["cheap7"]) - recomputed) if recomputed is not None and row.get("cheap7") is not None else None
        summary_path = pathlib.Path(row.get("summary_path", ""))
        rec["summary_path"] = str(summary_path)
        rec["summary_exists"] = summary_path.exists()
        if summary_path.exists():
            summary = read_json(summary_path)
            rec["summary_status"] = summary.get("status")
            rec["summary_cheap7"] = summary.get("cheap7")
            rec["summary_model_dir"] = summary.get("model_dir")
            rec["summary_model_sha256"] = (summary.get("model") or {}).get("model_safetensors_sha256")
            rec["summary_config_sha256"] = (summary.get("model") or {}).get("config_sha256")
            tasks = summary.get("tasks") or {}
            rec["n_tasks"] = len(tasks)
            rec["task_statuses"] = {k: v.get("status") for k, v in tasks.items() if isinstance(v, dict)}
            missing_artifacts = []
            pred_hashes = {}
            report_hashes = {}
            for k, task in tasks.items():
                if not isinstance(task, dict):
                    continue
                pred = pathlib.Path(task.get("predictions") or "")
                rep = pathlib.Path(task.get("report") or "")
                if not pred.exists():
                    missing_artifacts.append(f"{k}:predictions")
                if not rep.exists():
                    missing_artifacts.append(f"{k}:report")
                pred_hashes[k] = sha256_file(pred)
                report_hashes[k] = sha256_file(rep)
            rec["missing_artifacts"] = missing_artifacts
            rec["prediction_sha256"] = pred_hashes
            rec["report_sha256"] = report_hashes
            if len(tasks) != 8:  # GlobalPIQA is split into two tasks plus six other tasks
                bad.append(f"{label}:{ep}:expected 8 task records got {len(tasks)}")
            if missing_artifacts:
                bad.append(f"{label}:{ep}:missing artifacts {missing_artifacts}")
            if any(s not in {"done", "skipped_existing"} for s in rec["task_statuses"].values()):
                bad.append(f"{label}:{ep}:bad task statuses {rec['task_statuses']}")
            if summary.get("cheap7") is None or row.get("cheap7") is None or abs(float(summary.get("cheap7")) - float(row.get("cheap7"))) > 1e-9:
                bad.append(f"{label}:{ep}:trajectory/summary cheap7 mismatch")
        else:
            bad.append(f"{label}:{ep}:missing summary")
        if len(present_cols) != len(CHEAP_COLUMNS):
            bad.append(f"{label}:{ep}:missing columns")
        if rec["cheap7_abs_error"] is None or rec["cheap7_abs_error"] > 1e-9:
            bad.append(f"{label}:{ep}:cheap7 recompute error {rec['cheap7_abs_error']}")
        out["rows"].append(rec)
    out["issues"] = bad
    out["ok"] = not bad
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact-root", type=pathlib.Path, required=True)
    ap.add_argument("--repeat-root", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    compact = check_one("compact", args.compact_root, args.compact_root / "selected_causal_trajectory.json")
    repeat = check_one("repeat", args.repeat_root, args.repeat_root / "selected_causal_trajectory.json")
    summary = {"status": "CAUSAL_SELECTED_INTEGRITY", "ok": compact["ok"] and repeat["ok"], "compact": compact, "repeat": repeat}
    out_json = args.out_dir / "causal_selected_integrity_check.json"
    out_md = args.out_dir / "causal_selected_integrity_check.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research causal selected evaluation integrity check\n\n", f"Overall OK: **{summary['ok']}**\n\n"]
    for block in [compact, repeat]:
        lines.append(f"## {block['label']}\n\n")
        lines.append(f"Rows: {block['n_rows']}; ok: {block['ok']}\n\n")
        if block["issues"]:
            lines.append("Issues:\n")
            for issue in block["issues"]:
                lines.append(f"- {issue}\n")
        else:
            lines.append("No integrity issues found.\n")
        lines.append("\n| endpoint | cheap7 | recomputed | abs err | tasks | model sha prefix |\n")
        lines.append("|---|---:|---:|---:|---:|---|\n")
        for r in block["rows"]:
            sha = r.get("summary_model_sha256") or ""
            lines.append(f"| {r['endpoint']} | {float(r['trajectory_cheap7']):.9f} | {float(r['recomputed_cheap7']):.9f} | {float(r['cheap7_abs_error']):.2e} | {r.get('n_tasks')} | {sha[:12]} |\n")
        lines.append("\n")
    lines.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "ok": summary["ok"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)
    if not summary["ok"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
