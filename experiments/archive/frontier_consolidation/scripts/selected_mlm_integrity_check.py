#!/usr/bin/env python3
"""research: integrity check for selected DeBERTa MLM trajectory outputs.

Run after selected_mlm_checkpoint_eval.py finishes. It recomputes cheap7 from
trajectory rows, verifies the per-target payloads exist and contain the expected eight
cheap-task records (GlobalPIQA split), parses scores from the payload using the same
logic as batch_trajectory_eval, and reports cache/fresh provenance.
"""
from __future__ import annotations
import argparse
import json
import pathlib
from statistics import mean
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GP_COLS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
EXPECTED_TASKS = set(ZERO_COLUMNS + GP_COLS + ["Reading"])


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    out: dict[str, float | None] = {}
    for c in ZERO_COLUMNS:
        rec = tasks.get(c, {}) if isinstance(tasks, dict) else {}
        out[c] = float(rec["score"]) if isinstance(rec, dict) and rec.get("score") is not None else None
    gp_vals = []
    for c in GP_COLS:
        rec = tasks.get(c, {}) if isinstance(tasks, dict) else {}
        if isinstance(rec, dict) and rec.get("score") is not None:
            gp_vals.append(float(rec["score"]))
    out["GlobalPIQA"] = mean(gp_vals) if len(gp_vals) == 2 else None
    rd = tasks.get("Reading", {}) if isinstance(tasks, dict) else {}
    if isinstance(rd, dict) and isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif isinstance(rd, dict) and rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def check_task_artifacts(tasks: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for name in EXPECTED_TASKS:
        rec = tasks.get(name)
        if not isinstance(rec, dict):
            issues.append(f"missing_task:{name}")
            continue
        if rec.get("returncode") not in (None, 0):
            issues.append(f"bad_returncode:{name}:{rec.get('returncode')}")
        # cache-seeded payloads and fresh evaluator payloads both expose predictions/report.
        pred = pathlib.Path(rec.get("predictions") or "")
        report = pathlib.Path(rec.get("report") or "")
        if not pred.exists():
            issues.append(f"missing_predictions:{name}")
        if not report.exists():
            issues.append(f"missing_report:{name}")
        if name == "Reading":
            if not (isinstance(rec.get("scores"), dict) and rec["scores"].get("Reading") is not None) and rec.get("score") is None:
                issues.append(f"missing_reading_score:{name}")
        else:
            if rec.get("score") is None:
                issues.append(f"missing_score:{name}")
    # Older full-eval cache-seeded payloads may include SuperGLUE/AoA in addition to
    # the eight cheap-task records. Extra tasks are provenance, not a cheap7-integrity
    # failure; the caller records task_count so this remains visible.
    return issues


def check_one(label: str, root: pathlib.Path) -> dict[str, Any]:
    traj_path = root / "selected_trajectory.json"
    out: dict[str, Any] = {"label": label, "root": str(root), "trajectory_path": str(traj_path), "exists": traj_path.exists()}
    if not traj_path.exists():
        out["ok"] = False
        out["issues"] = ["missing_selected_trajectory"]
        return out
    rows = read_json(traj_path)
    out["n_rows"] = len(rows) if isinstance(rows, list) else None
    out["rows"] = []
    issues: list[str] = []
    if not isinstance(rows, list):
        out["ok"] = False
        out["issues"] = ["trajectory_not_list"]
        return out
    for row in rows:
        ep = row.get("endpoint")
        rec: dict[str, Any] = {"endpoint": ep, "trajectory_cheap7": row.get("cheap7")}
        if row.get("error"):
            rec["error"] = row.get("error")
            issues.append(f"{ep}:trajectory_error")
            out["rows"].append(rec)
            continue
        present = [c for c in CHEAP_COLUMNS if row.get(c) is not None]
        rec["present_columns"] = present
        rec["n_present_columns"] = len(present)
        row_scores = {c: (float(row[c]) if row.get(c) is not None else None) for c in CHEAP_COLUMNS}
        row_c7 = cheap7(row_scores)
        rec["row_recomputed_cheap7"] = row_c7
        rec["row_cheap7_abs_error"] = abs(float(row["cheap7"]) - row_c7) if row_c7 is not None and row.get("cheap7") is not None else None
        payload_path = root / "eval" / "per_target" / f"{label}_{ep}.json"
        rec["payload_path"] = str(payload_path)
        rec["payload_exists"] = payload_path.exists()
        if not payload_path.exists():
            issues.append(f"{ep}:missing_payload")
            out["rows"].append(rec)
            continue
        payload = read_json(payload_path)
        p_scores = extract_scores(payload)
        p_c7 = cheap7(p_scores)
        rec["payload_scores"] = p_scores
        rec["payload_cheap7"] = p_c7
        rec["payload_task_count"] = len(payload.get("tasks", {})) if isinstance(payload.get("tasks"), dict) else None
        rec["payload_task_issues"] = check_task_artifacts(payload.get("tasks", {}) if isinstance(payload.get("tasks"), dict) else {})
        rec["payload_description"] = payload.get("description")
        rec["payload_family"] = payload.get("family")
        rec["cached_seeded_from"] = payload.get("cache_seeded_from") or payload.get("description")
        if rec["payload_task_issues"]:
            issues.append(f"{ep}:payload_task_issues:{rec['payload_task_issues']}")
        if p_c7 is None:
            issues.append(f"{ep}:payload_no_cheap7")
        elif abs(float(row["cheap7"]) - p_c7) > 1e-9:
            issues.append(f"{ep}:payload_row_cheap7_mismatch:{float(row['cheap7'])}:{p_c7}")
        for c in CHEAP_COLUMNS:
            if p_scores.get(c) is None or row.get(c) is None or abs(float(row[c]) - float(p_scores[c])) > 1e-9:
                issues.append(f"{ep}:score_mismatch:{c}:{row.get(c)}:{p_scores.get(c)}")
        if len(present) != len(CHEAP_COLUMNS):
            issues.append(f"{ep}:missing_columns")
        if rec["row_cheap7_abs_error"] is None or rec["row_cheap7_abs_error"] > 1e-9:
            issues.append(f"{ep}:row_cheap7_error:{rec['row_cheap7_abs_error']}")
        out["rows"].append(rec)
    out["issues"] = issues
    out["ok"] = not issues
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", action="append", required=True, help="LABEL=OUT_DIR")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    for spec in args.trajectory:
        if "=" not in spec:
            raise SystemExit("Use --trajectory LABEL=OUT_DIR")
        label, root = spec.split("=", 1)
        blocks.append(check_one(label, pathlib.Path(root)))
    summary = {"status": "SELECTED_MLM_INTEGRITY", "ok": all(b.get("ok") for b in blocks), "trajectories": blocks}
    out_json = args.out_dir / "selected_mlm_integrity_check.json"
    out_md = args.out_dir / "selected_mlm_integrity_check.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research selected MLM trajectory integrity check\n\n", f"Overall OK: **{summary['ok']}**\n\n"]
    for b in blocks:
        lines.append(f"## {b['label']}\n\n")
        lines.append(f"Rows: {b.get('n_rows')}; ok: {b.get('ok')}\n\n")
        if b.get("issues"):
            lines.append("Issues:\n")
            for issue in b["issues"]:
                lines.append(f"- {issue}\n")
            lines.append("\n")
        else:
            lines.append("No integrity issues found.\n\n")
        lines.append("| endpoint | cheap7 | recomputed | payload c7 | task count |\n")
        lines.append("|---|---:|---:|---:|---:|\n")
        for r in b.get("rows", []):
            if r.get("trajectory_cheap7") is None:
                lines.append(f"| {r.get('endpoint')} |  |  |  | {r.get('payload_task_count','')} |\n")
            else:
                lines.append(f"| {r.get('endpoint')} | {float(r['trajectory_cheap7']):.9f} | {float(r.get('row_recomputed_cheap7') or 0.0):.9f} | {float(r.get('payload_cheap7') or 0.0):.9f} | {r.get('payload_task_count','')} |\n")
        lines.append("\n")
    lines.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "ok": summary["ok"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)
    if not summary["ok"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
