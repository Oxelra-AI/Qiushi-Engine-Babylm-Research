#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RUNNER = ROOT / "scripts/run_available_coordinate_for_model.py"
OUT_MANIFEST = ROOT / "data/relation_wwm_available_coordinate_manifest.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_available_coordinate_manifest.md')

JOBS = [
    {
        "key": "relation_10M",
        "run_name": "babylm_step93_relation_wwm_deberta_20M_chck_10M",
        "model_path": ROOT / "training/runs/babylm_relation_wwm_deberta_20M/hf_model",
        "revision": "chck_10M",
        "eval_dir": ROOT / "training/runs/babylm_relation_wwm_deberta_20M/eval_results_available_chck_10M",
        "out_json": ROOT / "data/relation_wwm_chck_10M_available_coordinate.json",
        "out_note": (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_chck_10M_available_coordinate.md'),
        "log": (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_chck_10M_available_coordinate.log'),
    },
    {
        "key": "relation_20M",
        "run_name": "babylm_step93_relation_wwm_deberta_20M_chck_20M",
        "model_path": ROOT / "training/runs/babylm_relation_wwm_deberta_20M/hf_model",
        "revision": "chck_20M",
        "eval_dir": ROOT / "training/runs/babylm_relation_wwm_deberta_20M/eval_results_available_chck_20M",
        "out_json": ROOT / "data/relation_wwm_chck_20M_available_coordinate.json",
        "out_note": (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_chck_20M_available_coordinate.md'),
        "log": (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_chck_20M_available_coordinate.log'),
    },
    {
        "key": "shuffled_10M",
        "run_name": "babylm_step93_relation_wwm_shuffled_deberta_20M_chck_10M",
        "model_path": ROOT / "training/runs/babylm_relation_wwm_shuffled_deberta_20M/hf_model",
        "revision": "chck_10M",
        "eval_dir": ROOT / "training/runs/babylm_relation_wwm_shuffled_deberta_20M/eval_results_available_chck_10M",
        "out_json": ROOT / "data/relation_wwm_shuffled_chck_10M_available_coordinate.json",
        "out_note": (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_shuffled_chck_10M_available_coordinate.md'),
        "log": (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_shuffled_chck_10M_available_coordinate.log'),
    },
    {
        "key": "shuffled_20M",
        "run_name": "babylm_step93_relation_wwm_shuffled_deberta_20M_chck_20M",
        "model_path": ROOT / "training/runs/babylm_relation_wwm_shuffled_deberta_20M/hf_model",
        "revision": "chck_20M",
        "eval_dir": ROOT / "training/runs/babylm_relation_wwm_shuffled_deberta_20M/eval_results_available_chck_20M",
        "out_json": ROOT / "data/relation_wwm_shuffled_chck_20M_available_coordinate.json",
        "out_note": (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_shuffled_chck_20M_available_coordinate.md'),
        "log": (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_shuffled_chck_20M_available_coordinate.log'),
    },
]


def run_job(job: dict) -> dict:
    cmd = [
        sys.executable, str(RUNNER),
        "--model_path", str(job["model_path"]),
        "--run_name", job["run_name"],
        "--out_json", str(job["out_json"]),
        "--out_note", str(job["out_note"]),
        "--log", str(job["log"]),
        "--output_dir", str(job["eval_dir"]),
        "--revision", job["revision"],
    ]
    t0 = time.time()
    print(json.dumps({"event": "start_job", "key": job["key"], "cmd": cmd}), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(p.stdout[-4000:], flush=True)
    rec = {
        "key": job["key"],
        "run_name": job["run_name"],
        "revision": job["revision"],
        "model_path": str(job["model_path"]),
        "eval_dir": str(job["eval_dir"]),
        "out_json": str(job["out_json"]),
        "out_note": str(job["out_note"]),
        "log": str(job["log"]),
        "returncode": p.returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": p.stdout[-4000:],
    }
    if p.returncode != 0:
        rec["status"] = "failed"
        return rec
    payload = json.loads(job["out_json"].read_text(encoding="utf-8"))
    rec["status"] = "ok"
    rec["scores"] = payload.get("scores", {})
    rec["derived_columns"] = payload.get("derived_columns", {})
    return rec


def main() -> None:
    results = []
    for job in JOBS:
        rec = run_job(job)
        results.append(rec)
        if rec["status"] != "ok":
            break
    manifest = {
        "status": "complete" if all(r["status"] == "ok" for r in results) and len(results) == len(JOBS) else "incomplete_or_failed",
        "jobs_requested": [j["key"] for j in JOBS],
        "results": results,
    }
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research — relation-WWM available-coordinate manifest", "", f"Evidence JSON: `{OUT_MANIFEST}`", ""]
    for r in results:
        lines.append(f"## {r['key']} — {r['status']}")
        lines.append("")
        lines.append(f"- coordinate JSON: `{r['out_json']}`")
        if r.get("scores"):
            d = r.get("derived_columns", {})
            s = r["scores"]
            lines.append(f"- BLiMP {s.get('blimp')}, Supplement {s.get('supplement')}, Entity {s.get('entity_tracking')}, COMPS {s.get('comps')}")
            lines.append(f"- GlobalPIQA mean {d.get('GlobalPIQA_mean_parallel_nonparallel')}, Reading mean {d.get('Reading_mean_eye_self_paced')}")
        lines.append("")
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(OUT_MANIFEST), "completed": len(results)}, indent=2))
    if manifest["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
