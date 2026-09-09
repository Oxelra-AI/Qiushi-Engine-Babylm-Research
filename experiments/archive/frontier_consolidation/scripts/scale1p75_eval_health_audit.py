#!/usr/bin/env python3
"""research: non-invasive health/resumability audit for scale1.75 full-eval output.

This script reads only files already written under the race-free full-evaluation
output root.  It does not wait for a task, run evaluation, invoke CUDA, or modify
any active part output.  The goal is to preserve what is complete, what is still
absent, and whether any observed failure is an environment/cache issue that a
future resume/merge-only invocation can repair.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import re
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
BASE_OUT = WORKSPACE / "data/scale1p75_100M_full_eval_hardened"
OUT_DIR = WORKSPACE / "data/scale1p75_eval_health_audit"
NOTE = WORKSPACE / "notes/eval_path_hardening_and_health.md"
TARGET = "scale1p75_100M_seed43022"
KIND_ORDER = [
    "BLiMP",
    "Supplement",
    "EWoK",
    "Entity",
    "COMPS",
    "GP_parallel",
    "GP_nonparallel",
    "Reading",
    "SuperGLUE",
    "AoA",
]
KIND_TO_TASK = {
    "BLiMP": "BLiMP",
    "Supplement": "Supplement",
    "EWoK": "EWoK",
    "Entity": "Entity",
    "COMPS": "COMPS",
    "GP_parallel": "GlobalPIQA_parallel",
    "GP_nonparallel": "GlobalPIQA_nonparallel",
    "Reading": "Reading",
    "SuperGLUE": "SuperGLUE",
    "AoA": "AoA",
}
ISSUE_PATTERNS = {
    "read_only_fs": re.compile(r"Read-only file system|Errno 30", re.I),
    "hf_module_cache": re.compile(r"HF_MODULES_CACHE|transformers_modules|dynamic module", re.I),
    "traceback": re.compile(r"Traceback \(most recent call last\)", re.I),
    "cuda_oom": re.compile(r"out of memory|CUDA error|CUBLAS_STATUS_ALLOC_FAILED", re.I),
    "missing_file": re.compile(r"FileNotFoundError|No such file or directory", re.I),
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def part_dir(kind: str) -> Path:
    return BASE_OUT / "parts" / kind


def payload_path(kind: str) -> Path:
    task = KIND_TO_TASK[kind]
    return part_dir(kind) / "eval" / "per_target" / f"{TARGET}__{kind}.json"


def scan_text_file(path: Path, max_chars: int = 12000) -> dict[str, Any]:
    rec: dict[str, Any] = {"path": rel(path), "exists": path.exists()}
    if not path.exists() or not path.is_file():
        return rec
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        rec["read_error"] = repr(exc)
        return rec
    tail = text[-max_chars:]
    rec["size_bytes"] = path.stat().st_size
    rec["issue_hits"] = {name: bool(pat.search(tail)) for name, pat in ISSUE_PATTERNS.items()}
    # Keep a small tail only if a suspicious pattern exists.
    if any(rec["issue_hits"].values()):
        rec["tail"] = tail[-3000:]
    return rec


def compact_payload(kind: str, path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"kind": kind, "payload": rel(path), "payload_exists": path.exists()}
    if not path.exists():
        return rec
    try:
        data = read_json(path)
    except Exception as exc:
        rec["payload_error"] = repr(exc)
        return rec
    task_key = KIND_TO_TASK[kind]
    tasks = data.get("tasks", {}) if isinstance(data, dict) else {}
    task = tasks.get(task_key) if isinstance(tasks, dict) else None
    rec["target"] = data.get("target") if isinstance(data, dict) else None
    rec["official_overall_present"] = isinstance(data.get("official_overall"), dict) if isinstance(data, dict) else False
    if not isinstance(task, dict):
        rec["task_present"] = False
        return rec
    rec["task_present"] = True
    rec["returncode"] = task.get("returncode")
    rec["status"] = task.get("status")
    rec["score"] = task.get("score")
    rec["scores"] = task.get("scores")
    rec["superglue_mean"] = task.get("superglue_mean")
    rec["aoa_leaderboard_score"] = task.get("aoa_leaderboard_score")
    rec["aoa_raw_correlation"] = task.get("aoa_raw_correlation")
    rec["num_rows"] = task.get("num_rows")
    rec["predictions"] = task.get("predictions")
    rec["surprisal_path"] = task.get("surprisal_path")
    rec["log"] = task.get("log")
    rec["error"] = task.get("error")
    if task_key == "SuperGLUE":
        sg_tasks = task.get("tasks") or []
        rec["superglue_tasks"] = [
            {"task": x.get("task"), "returncode": x.get("returncode"), "accuracy": x.get("accuracy"), "predictions": x.get("predictions"), "error": x.get("error")}
            for x in sg_tasks if isinstance(x, dict)
        ]
    return rec


def part_complete(kind: str, rec: dict[str, Any]) -> bool:
    if not rec.get("task_present"):
        return False
    if kind in {"BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GP_parallel", "GP_nonparallel"}:
        return rec.get("returncode") == 0 and rec.get("score") is not None
    if kind == "Reading":
        scores = rec.get("scores")
        return rec.get("returncode") == 0 and isinstance(scores, dict) and scores.get("Reading") is not None
    if kind == "SuperGLUE":
        return rec.get("superglue_mean") is not None and len(rec.get("superglue_tasks") or []) == 7
    if kind == "AoA":
        return rec.get("returncode") == 0 and rec.get("aoa_leaderboard_score") is not None
    return False


def collect_logs(kind: str) -> list[dict[str, Any]]:
    root = part_dir(kind)
    candidates: list[Path] = []
    if root.exists():
        candidates.extend(sorted((root / "driver_logs").glob("*.log")))
        candidates.extend(sorted((root / "eval" / "logs").rglob("*.log")))
        candidates.extend(sorted((root / "eval" / "logs").rglob("*.txt")))
    return [scan_text_file(p) for p in candidates]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_dir = BASE_OUT / "summary"
    final_summary = summary_dir / "scale1p75_100M_full_eval_hardened_summary.json"
    staged_payload = BASE_OUT / "staged_full_eval" / "per_target" / f"{TARGET}.json"
    collation_summary = BASE_OUT / "collate" / TARGET / f"pristine_collate_{TARGET}_summary.json"

    parts: list[dict[str, Any]] = []
    for kind in KIND_ORDER:
        rec = compact_payload(kind, payload_path(kind))
        rec["part_dir_exists"] = part_dir(kind).exists()
        rec["complete_by_step111_logic"] = part_complete(kind, rec)
        rec["logs"] = collect_logs(kind)
        rec["suspicious_log_issue_hits"] = {
            name: any((log.get("issue_hits") or {}).get(name) for log in rec["logs"])
            for name in ISSUE_PATTERNS
        }
        parts.append(rec)

    completed = [r["kind"] for r in parts if r.get("complete_by_step111_logic")]
    missing = [r["kind"] for r in parts if not r.get("complete_by_step111_logic")]
    issue_hits = {
        name: [r["kind"] for r in parts if r.get("suspicious_log_issue_hits", {}).get(name)]
        for name in ISSUE_PATTERNS
    }
    top = {
        "status": "SCALE1P75_EVAL_HEALTH_AUDIT",
        "created_utc": now(),
        "purpose": "Non-invasive file audit of the active scale1.75 full official-compatible evaluation output tree; no waiting or evaluation was performed.",
        "base_out": rel(BASE_OUT),
        "target": TARGET,
        "endpoint_ready_exists": (summary_dir / "endpoint_ready.json").exists(),
        "final_summary_exists": final_summary.exists(),
        "stage_summary_exists": (summary_dir / "stage_summary.json").exists(),
        "staged_payload_exists": staged_payload.exists(),
        "collation_summary_exists": collation_summary.exists(),
        "completed_parts": completed,
        "missing_or_incomplete_parts": missing,
        "issue_hits_by_pattern": issue_hits,
        "read_only_or_hf_cache_issue_detected": bool(issue_hits["read_only_fs"] or issue_hits["hf_module_cache"]),
        "parts": parts,
        "resumability_interpretation": None,
    }
    if top["final_summary_exists"]:
        try:
            fs = read_json(final_summary)
            top["final_summary_brief"] = {
                "Overall": fs.get("Overall"),
                "cheap7": fs.get("cheap7"),
                "overall_margin_vs_41p8": fs.get("overall_margin_vs_41p8"),
                "score_signal": fs.get("score_signal"),
            }
        except Exception as exc:
            top["final_summary_error"] = repr(exc)
    if not missing:
        top["resumability_interpretation"] = "All split parts complete; if the managed process fails after this point, a merge-only rerun should stage/collate existing predictions rather than repeat evaluation."
    elif completed:
        top["resumability_interpretation"] = "Some split parts complete and can be reused by the hardened evaluator when force is not set; only incomplete parts should run in a later resume."
    else:
        top["resumability_interpretation"] = "No split part is complete yet, or no part payloads are visible; continue waiting for the managed evaluator unless it returns a terminal failure."
    if top["read_only_or_hf_cache_issue_detected"]:
        top["resumability_interpretation"] += " A cache/path issue appears in logs; patched research evaluators now attach writable HF_HOME/HF_MODULES_CACHE at the outer subprocess boundary for future reruns."

    out_json = OUT_DIR / "scale1p75_eval_health_audit.json"
    write_json(out_json, top)

    lines = [
        "# research — scale1.75 full-eval path hardening and health audit",
        "",
        "This is a file-only audit of the active scale1.75 full-evaluation output tree. It did not run evaluation or invoke CUDA.",
        "",
        f"Base output: `{rel(BASE_OUT)}`",
        f"Endpoint-ready artifact exists: **{top['endpoint_ready_exists']}**",
        f"Final full-eval summary exists: **{top['final_summary_exists']}**",
        f"Completed split parts: `{completed}`",
        f"Missing/incomplete split parts: `{missing}`",
        f"Read-only/HF-cache issue detected in visible logs: **{top['read_only_or_hf_cache_issue_detected']}**",
        "",
        f"Interpretation: {top['resumability_interpretation']}",
        "",
        f"JSON: `{rel(out_json)}`",
    ]
    if top.get("final_summary_brief"):
        b = top["final_summary_brief"]
        lines += [
            "",
            "## Final summary already visible",
            f"Overall: `{b.get('Overall')}`; cheap7: `{b.get('cheap7')}`; margin vs 41.8: `{b.get('overall_margin_vs_41p8')}`; signal: `{b.get('score_signal')}`.",
        ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": top["status"],
        "final_summary_exists": top["final_summary_exists"],
        "completed_parts": completed,
        "missing_or_incomplete_parts": missing,
        "read_only_or_hf_cache_issue_detected": top["read_only_or_hf_cache_issue_detected"],
        "out_json": rel(out_json),
        "note": rel(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
