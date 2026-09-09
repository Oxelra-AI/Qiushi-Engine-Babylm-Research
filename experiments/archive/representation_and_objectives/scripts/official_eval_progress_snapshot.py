#!/usr/bin/env python3
"""research: one-time progress snapshot for the two corrected-tokenizer official evals.

This script reads the running evaluation output roots and writes a separate status
snapshot. It does not alter evaluation files, model weights, data, tokenizer, or
controller state. The snapshot is for liveness/stage interpretation only: partial
non-EWoK/non-AoA surfaces are explicitly not endpoint evidence.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
from pathlib import Path
import re
import time
from typing import Any

def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WORKSPACE = ROOT / "experiments/archive/representation_and_objectives"
OUT_DIR = WORKSPACE / "data/official_eval_progress_snapshot"
OUT_JSON = OUT_DIR / "official_eval_progress_snapshot.json"
OUT_MD = WORKSPACE / "notes/official_eval_progress_snapshot.md"

SEEDS = {
    "43022": {
        "task_ref": "s51_t41_tool1",
        "full_root": WORKSPACE / "data/strictsmalltok_seed43022_full_eval",
        "ewok_root": WORKSPACE / "data/strictsmalltok_seed43022_official_ewok",
        "aoa_root": WORKSPACE / "data/strictsmalltok_seed43022_official_aoa_min0",
        "collate_root": WORKSPACE / "data/strictsmalltok_seed43022_pristine_collate",
        "controller_root": WORKSPACE / "data/strictsmalltok_seed43022_posttrain_eval_controller",
        "target": "strictsmalltok_reinvest_seed43022",
    },
    "43122": {
        "task_ref": "s51_t42_tool1",
        "full_root": WORKSPACE / "data/strictsmalltok_seed43122_full_eval",
        "ewok_root": WORKSPACE / "data/strictsmalltok_seed43122_official_ewok",
        "aoa_root": WORKSPACE / "data/strictsmalltok_seed43122_official_aoa_min0",
        "collate_root": WORKSPACE / "data/strictsmalltok_seed43122_pristine_collate",
        "controller_root": WORKSPACE / "data/strictsmalltok_seed43122_posttrain_eval_controller",
        "target": "strictsmalltok_reinvest_seed43122",
    },
}

FULL_WRAPPER_COLUMNS = [
    "BLiMP", "Supplement", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading", "SuperGLUE",
]
OFFICIAL_AFTER_FULL = ["official_ewok_7618", "official_aoa_min0_8005", "pristine_collate"]
SUPERGLUE_TASKS = ["boolq", "multirc", "mrpc", "mnli", "qqp", "rte", "wsc"]


def file_rec(path: Path, now: float) -> dict[str, Any]:
    try:
        st = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": st.st_size,
        "mtime": st.st_mtime,
        "age_sec": round(now - st.st_mtime, 3),
    }


def safe_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {"_read_error": repr(exc), "path": str(path)}


def tail_text(path: Path, max_bytes: int = 60000) -> str:
    try:
        with path.open("rb") as f:
            try:
                f.seek(-max_bytes, os.SEEK_END)
            except OSError:
                f.seek(0)
            return f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


def latest_files(root: Path, now: float, n: int = 12) -> list[dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            rows.append((st.st_mtime, st.st_size, str(p.relative_to(root))))
    rows.sort(reverse=True)
    return [{"relpath": rel, "size_bytes": size, "age_sec": round(now - mt, 3)} for mt, size, rel in rows[:n]]


def parse_tqdm_progress(text: str) -> dict[str, Any] | None:
    # Examples contain "58%|...| 9810/17020 [03:31<15:15,  7.88it/s,...]".
    matches = re.findall(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]", text)
    if not matches:
        return None
    pct, cur, total, bracket = matches[-1]
    return {"percent": int(pct), "current": int(cur), "total": int(total), "progress_bracket": bracket}


def summarize_superglue_logs(full_root: Path, target: str, now: float) -> dict[str, Any]:
    log_root = full_root / "logs" / target
    out: dict[str, Any] = {}
    for task in SUPERGLUE_TASKS:
        log = log_root / f"superglue_{task}.log"
        txt = tail_text(log)
        rec = file_rec(log, now)
        if txt:
            rec["has_traceback_or_oom"] = any(s in txt for s in ["Traceback", "RuntimeError", "CUDA out of memory", "returncode=1", "returncode=2"])
            rec["returncode_lines_tail"] = [ln[-300:] for ln in txt.splitlines() if "returncode=" in ln][-3:]
            rec["latest_tqdm_progress"] = parse_tqdm_progress(txt)
            rec["last_nonempty_lines"] = [ln[-300:] for ln in txt.splitlines() if ln.strip()][-6:]
        out[task] = rec
    return out


def summarize_payload(full_root: Path, target: str) -> dict[str, Any]:
    payload_path = full_root / "per_target" / f"{target}.json"
    p = safe_json(payload_path)
    rec: dict[str, Any] = {"payload_path": str(payload_path), "exists": payload_path.exists()}
    if not isinstance(p, dict):
        rec["payload"] = p
        return rec
    tasks = p.get("tasks") or {}
    task_summary: dict[str, Any] = {}
    for col in FULL_WRAPPER_COLUMNS:
        t = tasks.get(col)
        if isinstance(t, dict):
            task_summary[col] = {
                "present": True,
                "returncode": t.get("returncode"),
                "score": t.get("score"),
                "scores": t.get("scores"),
                "superglue_mean": t.get("superglue_mean"),
                "superglue_task_count": len(t.get("tasks") or []) if col == "SuperGLUE" else None,
                "superglue_tasks": [{"task": r.get("task"), "accuracy": r.get("accuracy"), "returncode": r.get("returncode")} for r in (t.get("tasks") or [])] if col == "SuperGLUE" else None,
                "error": t.get("error"),
            }
        else:
            task_summary[col] = {"present": False}
    rec["tasks"] = task_summary
    rec["official_overall"] = p.get("official_overall")
    return rec


def summarize_controller(root: Path, seed: str, now: float) -> dict[str, Any]:
    summary = root / f"posttrain_eval_seed{seed}_summary.json"
    preflight = root / f"posttrain_eval_seed{seed}_preflight.json"
    logs = root / "logs"
    return {
        "root": file_rec(root, now),
        "summary": safe_json(summary) if summary.exists() else None,
        "summary_record": file_rec(summary, now),
        "preflight_record": file_rec(preflight, now),
        "controller_logs_latest": latest_files(logs, now, 8),
    }


def count_files(root: Path) -> dict[str, Any]:
    c = 0
    total = 0
    if root.exists():
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    st = p.stat(); c += 1; total += st.st_size
                except OSError:
                    pass
    return {"path": str(root), "exists": root.exists(), "file_count": c, "total_bytes": total}


def summarize_seed(seed: str, cfg: dict[str, Any], now: float) -> dict[str, Any]:
    full_root = cfg["full_root"]
    target = cfg["target"]
    rec: dict[str, Any] = {
        "seed": seed,
        "task_ref": cfg["task_ref"],
        "target": target,
        "full_eval_root": count_files(full_root),
        "official_ewok_root": count_files(cfg["ewok_root"]),
        "official_aoa_root": count_files(cfg["aoa_root"]),
        "pristine_collate_root": count_files(cfg["collate_root"]),
        "latest_full_eval_files": latest_files(full_root, now, 12),
        "payload_summary": summarize_payload(full_root, target),
        "superglue_logs": summarize_superglue_logs(full_root, target, now),
        "controller": summarize_controller(cfg["controller_root"], seed, now),
    }
    # Conservative stage inference.
    payload_tasks = rec["payload_summary"].get("tasks", {}) if isinstance(rec.get("payload_summary"), dict) else {}
    full_done = all(isinstance(payload_tasks.get(col), dict) and payload_tasks[col].get("present") for col in FULL_WRAPPER_COLUMNS[:-1])
    sg = payload_tasks.get("SuperGLUE") if isinstance(payload_tasks.get("SuperGLUE"), dict) else {}
    sg_count = sg.get("superglue_task_count", 0) if isinstance(sg, dict) else 0
    if rec["pristine_collate_root"]["file_count"]:
        stage = "collate_or_done"
    elif rec["official_aoa_root"]["file_count"]:
        stage = "official_aoa_or_collate"
    elif rec["official_ewok_root"]["file_count"]:
        stage = "official_ewok_or_aoa"
    elif sg_count:
        stage = f"full_wrapper_superglue_{sg_count}_of_{len(SUPERGLUE_TASKS)}_tasks_recorded"
    elif full_done:
        stage = "full_wrapper_after_reading_before_superglue_record"
    else:
        stage = "full_wrapper_non_ewok_non_aoa_in_progress"
    rec["stage_inference"] = stage
    rec["partial_scores_not_endpoint_evidence"] = True
    return rec


def write_note(snapshot: dict[str, Any]) -> None:
    lines = []
    lines.append("# research corrected-tokenizer official-evaluation progress snapshot")
    lines.append("")
    lines.append("This snapshot is a liveness and controller-stage record only. It must not be used as a corrected-tokenizer endpoint score because official EWoK, official min-context-zero AoA, and pristine collation are not complete unless the controller summary and collate roots say so.")
    lines.append("")
    lines.append(f"Created UTC: `{snapshot['created_utc']}`")
    lines.append("")
    for seed, rec in snapshot["seeds"].items():
        lines.append(f"## Seed {seed} ({rec['task_ref']})")
        lines.append(f"- Stage inference: `{rec['stage_inference']}`")
        full = rec["full_eval_root"]
        lines.append(f"- Full-wrapper root: exists={full['exists']} files={full['file_count']} bytes={full['total_bytes']}")
        lines.append(f"- Official EWoK root: files={rec['official_ewok_root']['file_count']}; AoA root: files={rec['official_aoa_root']['file_count']}; collate root: files={rec['pristine_collate_root']['file_count']}")
        tasks = rec["payload_summary"].get("tasks", {})
        visible = []
        for col in FULL_WRAPPER_COLUMNS:
            t = tasks.get(col, {}) if isinstance(tasks, dict) else {}
            if t.get("present"):
                if col == "SuperGLUE":
                    visible.append(f"{col}: {t.get('superglue_task_count')}/{len(SUPERGLUE_TASKS)} tasks")
                elif col == "Reading":
                    visible.append(f"{col}: scores={t.get('scores')}")
                else:
                    visible.append(f"{col}: {t.get('score')}")
        lines.append("- Wrapper-recorded columns so far: " + ("; ".join(visible) if visible else "none"))
        for task, logrec in rec["superglue_logs"].items():
            prog = logrec.get("latest_tqdm_progress")
            if prog or logrec.get("returncode_lines_tail"):
                lines.append(f"- SuperGLUE `{task}` log: age={logrec.get('age_sec')}s size={logrec.get('size_bytes')} progress={prog} return_lines={logrec.get('returncode_lines_tail')}")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("Both evaluations had recent SuperGLUE log updates in this snapshot and the downstream official EWoK/AoA/collate roots were still empty, consistent with the controller still inside the full-wrapper SuperGLUE phase. Seven-column wrapper scores are useful only for knowing that upstream columns have run; they are not official corrected endpoints and should not drive route changes before complete evaluation summaries are available.")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    now = time.time()
    snapshot = {
        "status": "OFFICIAL_EVAL_PROGRESS_SNAPSHOT",
        "created_unix": now,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "purpose": "Record completion status of the two corrected-tokenizer evaluations without treating partial results as endpoint evidence.",
        "seeds": {seed: summarize_seed(seed, cfg, now) for seed, cfg in SEEDS.items()},
        "interpretation": [
            "Both managed evaluations should remain the single decisive evidence source for the corrected-tokenizer route.",
            "Do not train or launch relation-retention repairs before full official scores expose a decision-changing weakness on the compliant coordinate.",
            "If controller summaries and pristine collations are absent, any visible zero-shot/Reading/SuperGLUE fragments are not official endpoints.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(snapshot)
    print(json.dumps({
        "status": snapshot["status"],
        "out_json": str(OUT_JSON),
        "note": str(OUT_MD),
        "stage_inference": {seed: rec["stage_inference"] for seed, rec in snapshot["seeds"].items()},
        "ewok_files": {seed: rec["official_ewok_root"]["file_count"] for seed, rec in snapshot["seeds"].items()},
        "aoa_files": {seed: rec["official_aoa_root"]["file_count"] for seed, rec in snapshot["seeds"].items()},
        "collate_files": {seed: rec["pristine_collate_root"]["file_count"] for seed, rec in snapshot["seeds"].items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
