#!/usr/bin/env python3
"""research: compare clean lambda=0 replay against original research (M,S).

The preservation trainer was found to change the global
seed and inserted auxiliary forwards, confounding causal interpretation of lambda
strengths. This script runs a bounded acquisition-only replay through:
  A) original research wrapper (research trainer after dense-mask/sparse-label patch)
  B) research clean preservation trainer with lambda_pres=0
and compares logs plus checkpoint tensor hashes.

By default it disables CUDA to avoid contending with long H100 jobs. The purpose is
not frontier evaluation; it is a short implementation equivalence control.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/zero_preservation_replay_compare')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_cmd(cmd: List[str], env: Dict[str, str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> Dict[str, Any]:
    t0 = time.time()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open("w", encoding="utf-8") as se:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=so, stderr=se, text=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - t0, 1),
        "stdout": rel(stdout_path),
        "stderr": rel(stderr_path),
    }


def extract_overlap(log: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "update", "schedule_idx", "lr", "rows", "words", "cum_words",
        "ordinary_target_tokens", "focus_target_tokens", "qwen_rows", "qwen_focus_rows",
        "ordinary_wwm_rows", "focus_selected_groups", "focus_candidate_groups",
        "focus_loss", "ordinary_loss", "pooled_loss", "grad_norm_preclip",
    ]
    out = {k: log.get(k) for k in keys}
    # research clean trainer names the same scalar pooled_ce; use one field for comparison.
    if out.get("pooled_loss") is None and "pooled_ce" in log:
        out["pooled_loss"] = log.get("pooled_ce")
    # research names optimized_loss; research has optimized_acq.
    out["optimized_acquisition_loss"] = log.get("optimized_loss", log.get("optimized_acq"))
    return out


def compare_values(a: Any, b: Any, path: str, diffs: List[Dict[str, Any]], tol: float) -> None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(float(a) - float(b)) > tol:
            diffs.append({"path": path, "a": a, "b": b, "abs_diff": abs(float(a) - float(b))})
    elif a != b:
        diffs.append({"path": path, "a": a, "b": b})


def compare_logs(orig_logs: List[Dict[str, Any]], clean_logs: List[Dict[str, Any]], tol: float) -> List[Dict[str, Any]]:
    diffs: List[Dict[str, Any]] = []
    if len(orig_logs) != len(clean_logs):
        diffs.append({"path": "len(update_log)", "a": len(orig_logs), "b": len(clean_logs)})
        return diffs
    for i, (o, c) in enumerate(zip(orig_logs, clean_logs), start=1):
        oo = extract_overlap(o)
        cc = extract_overlap(c)
        for k in sorted(set(oo) | set(cc)):
            compare_values(oo.get(k), cc.get(k), f"update_{i}.{k}", diffs, tol)
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--max-updates", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--cpu", action="store_true", default=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-10)
    args = ap.parse_args()

    out_dir = args.out_dir
    orig_root = out_dir / "original_revision_075"
    clean_root = out_dir / "clean_lambda0"
    logs_dir = out_dir / "run_logs"
    for p in [orig_root, clean_root, logs_dir]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["QIUSHI_TORCH_THREADS"] = env.get("QIUSHI_TORCH_THREADS", "8")
    cwd = ROOT

    common_args = [
        "--max-updates", str(int(args.max_updates)),
        "--checkpoint-every", str(int(args.checkpoint_every)),
        "--log-every", "1",
        "--train-seed", "62064",
        "--gpu", str(int(args.gpu)),
    ]
    orig_cmd = [sys.executable, str(_public_path('experiments/archive/functional_learning/scripts/densemask_sparselabel_train.py')),
                "--objectives", "correspondence_focus_weighted",
                "--out-dir", str(orig_root)] + common_args
    clean_cmd = [sys.executable, str(_public_path('experiments/archive/functional_learning/scripts/clean_preservation_train.py')),
                 "--lambda-pres", "0.0", "--pres-student-mode", "eval",
                 "--out-dir", str(clean_root)] + common_args

    runs = {
        "original_revision_075": run_cmd(orig_cmd, env, cwd, logs_dir / "original_stdout.log", logs_dir / "original_stderr.log"),
    }
    if runs["original_revision_075"]["returncode"] != 0:
        status = "ZERO_REPLAY_ORIGINAL_FAILED"
        report = {"status": status, "created_utc": now(), "runs": runs}
        (out_dir / "zero_preservation_replay_compare.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": status, "report": rel(out_dir / "zero_preservation_replay_compare.json")}, indent=2), flush=True)
        raise SystemExit(1)
    runs["clean_lambda0"] = run_cmd(clean_cmd, env, cwd, logs_dir / "clean_stdout.log", logs_dir / "clean_stderr.log")
    if runs["clean_lambda0"]["returncode"] != 0:
        status = "ZERO_REPLAY_CLEAN_FAILED"
        report = {"status": status, "created_utc": now(), "runs": runs}
        (out_dir / "zero_preservation_replay_compare.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": status, "report": rel(out_dir / "zero_preservation_replay_compare.json")}, indent=2), flush=True)
        raise SystemExit(1)

    orig_train = orig_root / "correspondence_focus_weighted"
    orig_summary = read_json(orig_train / "train_summary.json")
    clean_summary = read_json(clean_root / "train_summary.json")
    orig_logs = read_jsonl(orig_train / "update_log.jsonl")
    clean_logs = read_jsonl(clean_root / "update_log.jsonl")
    diffs = compare_logs(orig_logs, clean_logs, float(args.tolerance))

    ck = f"update_{int(args.max_updates):04d}"
    orig_ckpt = orig_train / "checkpoints" / ck / "model.safetensors"
    clean_ckpt = clean_root / "checkpoints" / ck / "model.safetensors"
    orig_hash = sha256_file(orig_ckpt)
    clean_hash = sha256_file(clean_ckpt)
    if orig_hash != clean_hash:
        diffs.append({"path": f"checkpoint_{ck}.model_safetensors_sha256", "a": orig_hash, "b": clean_hash})

    status = "ZERO_PRESERVATION_REPLAY_MATCH" if not diffs else "ZERO_PRESERVATION_REPLAY_DIFFERS"
    report = {
        "status": status,
        "created_utc": now(),
        "scientific_question": "Does the clean lambda=0 trainer reproduce the original (M,S) acquisition-only implementation on a short replay?",
        "max_updates": int(args.max_updates),
        "device_policy": "CPU/no CUDA" if args.cpu else f"cuda:{args.gpu}",
        "runs": runs,
        "original_summary_path": rel(orig_train / "train_summary.json"),
        "clean_summary_path": rel(clean_root / "train_summary.json"),
        "original_completed_updates": orig_summary.get("completed_updates"),
        "clean_completed_updates": clean_summary.get("completed_updates"),
        "checkpoint_hashes": {
            "original": orig_hash,
            "clean": clean_hash,
            "match": orig_hash == clean_hash,
        },
        "log_diffs": diffs,
        "original_final_overlap": extract_overlap(orig_logs[-1]) if orig_logs else None,
        "clean_final_overlap": extract_overlap(clean_logs[-1]) if clean_logs else None,
        "interpretation": "A match supports using research clean trainer for controlled preservation tests. It does not validate nonzero preservation strength; that still needs direct training/evaluation.",
    }
    out_json = out_dir / "zero_preservation_replay_compare.json"
    out_md = out_dir / "zero_preservation_replay_compare.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research zero-preservation replay compare",
        "",
        f"Status: `{status}`",
        f"Max updates: `{args.max_updates}`",
        f"Original elapsed: `{runs['original_revision_075']['elapsed_sec']}` sec; clean elapsed: `{runs['clean_lambda0']['elapsed_sec']}` sec",
        f"Checkpoint hash match: `{orig_hash == clean_hash}`",
        f"Original hash: `{orig_hash}`",
        f"Clean hash: `{clean_hash}`",
        f"Log diff count: `{len(diffs)}`",
        "",
        "This checks short-horizon equivalence of the clean lambda=0 trainer to the original (M,S) acquisition-only implementation.",
    ]
    if diffs[:10]:
        lines.append("\n## First diffs")
        for d in diffs[:10]:
            lines.append(f"- `{d['path']}`: {d.get('a')} vs {d.get('b')}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_json": rel(out_json), "out_md": rel(out_md), "hash_match": orig_hash == clean_hash, "diff_count": len(diffs), "elapsed": {k: v["elapsed_sec"] for k, v in runs.items()}}, indent=2), flush=True)
    if diffs:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
