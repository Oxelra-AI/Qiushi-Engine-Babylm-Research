#!/usr/bin/env python3
"""Integrity verifier for compact-view triangle no-AoA payloads.

Use after the guarded triangle readout has produced per-target JSON files. It checks
that every expected fast/no-AoA column is present, every subprocess returned 0, every
score is finite, output directories and prediction artifacts exist, the implementation
guard permits historical compact_view reference use, and the readiness verifier passed.
It does not evaluate models.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
TRIANGLE_OUT = _public_path('experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval')
READY_JSON = _public_path('experiments/archive/representation_and_objectives/data/triangle_readiness/triangle_readiness.json')
OUT_DIR_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/triangle_eval_integrity')
TARGETS = ["compact_view_reinvest", "compact_repeat_reinvest", "adjbreak_reinvest"]
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
SENTENCE_COLUMNS = [c for c in COLUMNS if c != "Reading"]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def prediction_artifacts(output_dir: Path, reading: bool) -> dict[str, Any]:
    if not output_dir.exists():
        return {"output_dir_exists": False, "prediction_files": [], "report_files": []}
    pred_patterns = ["prediction.jsonl", "predictions.json"] if reading else ["predictions.json"]
    preds: list[Path] = []
    for pat in pred_patterns:
        preds.extend(output_dir.rglob(pat))
    reports = list(output_dir.rglob("*.txt"))
    return {
        "output_dir_exists": True,
        "prediction_files": [rel(p) for p in sorted(preds)],
        "nonempty_prediction_files": [rel(p) for p in sorted(preds) if p.stat().st_size > 0],
        "report_files": [rel(p) for p in sorted(reports)],
    }


def check_task(col: str, rec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"column": col, "present": bool(rec)}
    if not rec:
        out["ok"] = False
        out["problem"] = "missing_task_record"
        return out
    out["returncode"] = rec.get("returncode")
    out["returncode_ok"] = rec.get("returncode") == 0
    data_path = rec.get("data_path")
    if data_path:
        out["data_path"] = data_path
        out["data_path_exists_under_strict"] = (STRICT / str(data_path)).exists()
    score_ok = False
    if col == "Reading":
        scores = rec.get("scores", {}) if isinstance(rec.get("scores"), dict) else {}
        out["scores"] = scores
        score_ok = all(is_finite_number(scores.get(k)) for k in ["Reading", "Reading_eye", "Reading_self_paced"])
        out["score_ok"] = score_ok
    else:
        out["score"] = rec.get("score")
        out["stdout_score"] = rec.get("stdout_score")
        out["report_score"] = rec.get("report_score")
        score_ok = is_finite_number(rec.get("score"))
        out["score_ok"] = score_ok
    output_dir = Path(str(rec.get("output_dir", "")))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    arts = prediction_artifacts(output_dir, reading=(col == "Reading"))
    out.update(arts)
    out["prediction_artifact_ok"] = bool(arts.get("nonempty_prediction_files"))
    # Reading sometimes writes only prediction.jsonl + report; sentence tasks should write predictions.json.
    out["ok"] = bool(out["returncode_ok"] and score_ok and out["prediction_artifact_ok"])
    if data_path:
        out["ok"] = bool(out["ok"] and out.get("data_path_exists_under_strict"))
    return out


def target_payload_path(target: str, triangle_out: Path) -> Path:
    return triangle_out / "per_target" / f"{target}.json"


def check_target(target: str, triangle_out: Path) -> dict[str, Any]:
    path = target_payload_path(target, triangle_out)
    out: dict[str, Any] = {"target": target, "path": rel(path), "exists": path.exists()}
    if not path.exists():
        out["ok"] = False
        return out
    payload = read_json(path)
    out["model_path"] = payload.get("model_path")
    mp = Path(str(payload.get("model_path", "")))
    if not mp.is_absolute():
        mp = ROOT / mp
    out["model_exists"] = (mp / "model.safetensors").exists()
    out["started_utc"] = payload.get("started_utc")
    out["finished_utc"] = payload.get("finished_utc")
    tasks = payload.get("tasks", {}) if isinstance(payload.get("tasks"), dict) else {}
    out["task_checks"] = {col: check_task(col, tasks.get(col, {})) for col in COLUMNS}
    out["ok"] = bool(out["model_exists"] and all(ch["ok"] for ch in out["task_checks"].values()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triangle-out", default=str(TRIANGLE_OUT))
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()
    triangle_out = Path(args.triangle_out)
    if not triangle_out.is_absolute():
        triangle_out = ROOT / triangle_out
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = triangle_out / "triangle_noaoa_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    guard = summary.get("implementation_equivalence", {}) if isinstance(summary.get("implementation_equivalence"), dict) else {}
    ready = read_json(READY_JSON) if READY_JSON.exists() else {}
    targets = {t: check_target(t, triangle_out) for t in TARGETS}
    table = summary.get("table", {}) if isinstance(summary.get("table"), dict) else {}
    table_ok = all(t in table for t in TARGETS)
    finite_table_ok = table_ok and all(is_finite_number(table[t].get(k)) for t in TARGETS for k in ["BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean", "Reading", "equal7_mean", "equal7_full_entity"])
    all_ok = bool(
        summary_path.exists()
        and guard.get("allows_historical_reference_for_causal_triangle") is True
        and ready.get("all_ok") is True
        and all(t["ok"] for t in targets.values())
        and finite_table_ok
    )
    payload = {
        "status": "TRIANGLE_EVAL_PAYLOAD_INTEGRITY_PASS" if all_ok else "TRIANGLE_EVAL_PAYLOAD_INTEGRITY_NOT_READY_OR_FAIL",
        "triangle_summary": rel(summary_path),
        "summary_exists": summary_path.exists(),
        "summary_created_utc": summary.get("created_utc") if isinstance(summary, dict) else None,
        "guard_allows": guard.get("allows_historical_reference_for_causal_triangle"),
        "readiness_path": rel(READY_JSON),
        "readiness_all_ok": ready.get("all_ok") if isinstance(ready, dict) else None,
        "targets": targets,
        "table_targets": sorted(table.keys()) if isinstance(table, dict) else [],
        "table_ok": table_ok,
        "finite_table_ok": finite_table_ok,
        "all_ok": all_ok,
        "interpretation_rule": "Only read the triangle mechanism if this file reports PASS; otherwise inspect failed task records/logs or rerun the guarded no-AoA readout with force after fixing the issue.",
    }
    out_json = out_dir / "triangle_eval_payload_integrity.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = out_dir / "triangle_eval_payload_integrity.md"
    lines = ["# research triangle eval payload integrity", "", f"Status: `{payload['status']}`", f"all_ok: `{all_ok}`", f"guard_allows: `{payload['guard_allows']}`", f"readiness_all_ok: `{payload['readiness_all_ok']}`", "", "## Targets"]
    for t, ch in targets.items():
        lines.append(f"- `{t}`: ok={ch.get('ok')}, exists={ch.get('exists')}, model_exists={ch.get('model_exists')}, finished={ch.get('finished_utc')}")
    lines.append("")
    lines.append(f"JSON: `{rel(out_json)}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "all_ok": all_ok, "out_json": rel(out_json), "out_md": rel(md)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
