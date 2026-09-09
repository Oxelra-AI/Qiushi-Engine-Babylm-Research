#!/usr/bin/env python3
"""Collect and evaluate Muon→AdamW 40M artifacts.

The switch runs launched by frontier_consolidation research currently expose chck_10M..chck_40M,
not 70M/80M.  This script evaluates the shared 40M surface for four loadable
checkpoints without modifying the source artifacts:

  - continuous AdamW legal compact reinvest (research/36 r2) at chck_40M
  - continuous matched-decay Muon at chck_40M
  - Muon20M→AdamW at chck_40M (about 20M Muon + 20M AdamW)
  - Muon40M→AdamW at chck_40M (switch-boundary / continuous-Muon-like control)

Outputs are written only under representation_and_objectives data/muon_switch_40m_eval.
A proxy run directory contains a symlink to the source hf_model plus a small metrics
manifest so the existing official-compatible evaluator can run without modifying
the custom trainer directories.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


def find_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_root()
A01 = ROOT / "experiments/archive/representation_and_objectives"
A02 = ROOT / "experiments/archive/frontier_consolidation"
EVALUATOR = A02 / "scripts/evaluate_compliant_endpoint.py"
OUT_ROOT = A01 / "data/muon_switch_40m_eval"
NOTE_PATH = A01 / "notes/muon_switch_40m_harvest.md"

EVAL_COLUMNS = [
    "BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
    "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading",
]
SCORE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GP_COLS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]

ARMS: dict[str, dict[str, Any]] = {
    "adamw40": {
        "label": "continuous_AdamW_40M",
        "source_run_dir": A02 / "training/runs/complianttok_reinvest_seed43022_r2",
        "endpoint": "chck_40M",
        "target": "a02_continuous_adamw_reinvest_40M",
        "role": "matched legal compact-view AdamW reference",
    },
    "muon40_cont": {
        "label": "continuous_Muon_wdmatch_40M",
        "source_run_dir": A02 / "training/runs/muon_lr008_wd00125_seed43022_80M",
        "endpoint": "chck_40M",
        "target": "a02_continuous_muon_wdmatch_40M",
        "role": "continuous matched-decay Muon reference at the same 40M exposure",
    },
    "muon20toadamw40": {
        "label": "Muon20M_to_AdamW_40M",
        "source_run_dir": A02 / "training/runs/muon20toadamw_seed43022_80M",
        "endpoint": "chck_40M",
        "target": "a02_muon20toadamw_40M",
        "role": "tests whether the strong 20M Muon geometry survives 20M of AdamW consolidation",
    },
    "muon40toadamw40": {
        "label": "Muon40M_to_AdamW_boundary_40M",
        "source_run_dir": A02 / "training/runs/muon40toadamw_seed43022_80M",
        "endpoint": "chck_40M",
        "target": "a02_muon40toadamw_boundary_40M",
        "role": "switch-boundary control; nearly continuous Muon before AdamW has real post-switch exposure",
    },
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_proxy(arm_key: str) -> Path:
    spec = ARMS[arm_key]
    src = Path(spec["source_run_dir"])
    ck = src / "hf_model" / spec["endpoint"] / "model.safetensors"
    if not ck.exists():
        raise FileNotFoundError(f"missing checkpoint for {arm_key}: {ck}")
    proxy = OUT_ROOT / "proxy_runs" / arm_key
    proxy.mkdir(parents=True, exist_ok=True)
    hf_link = proxy / "hf_model"
    desired = src / "hf_model"
    if hf_link.exists() or hf_link.is_symlink():
        if not hf_link.is_symlink() or hf_link.resolve() != desired.resolve():
            if hf_link.is_symlink() or hf_link.is_file():
                hf_link.unlink()
            else:
                raise RuntimeError(f"proxy hf_model exists and is not the expected symlink: {hf_link}")
    if not hf_link.exists():
        os.symlink(desired, hf_link, target_is_directory=True)
    metrics = {
        "status": "A01_PROXY_FOR_A02_EVALUATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm": arm_key,
        "label": spec["label"],
        "source_run_dir": str(src),
        "source_checkpoint": str(src / "hf_model" / spec["endpoint"]),
        "purpose": "Allows the existing official-compatible evaluator to read custom A02 checkpoints without editing A02 run directories. The proxy is not a training metric record.",
    }
    (proxy / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (proxy / "proxy_manifest.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return proxy


def arm_out_root(arm_key: str) -> Path:
    return OUT_ROOT / "eval_parts" / arm_key


def arm_collate_root(arm_key: str) -> Path:
    return OUT_ROOT / "collate" / arm_key


def per_target_path(arm_key: str) -> Path:
    return arm_out_root(arm_key) / "per_target" / f"{ARMS[arm_key]['target']}.json"


def run_arm(arm_key: str, gpu: int, force: bool = False, timeout: int = 2400) -> dict[str, Any]:
    spec = ARMS[arm_key]
    out_json = per_target_path(arm_key)
    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if out_json.exists() and not force:
        return {"arm": arm_key, "status": "exists", "per_target": str(out_json)}
    proxy = ensure_proxy(arm_key)
    arm_out_root(arm_key).mkdir(parents=True, exist_ok=True)
    arm_collate_root(arm_key).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-B", str(EVALUATOR), "--arm", "reinvest",
        "--run-dir", str(proxy),
        "--target", str(spec["target"]),
        "--endpoint", str(spec["endpoint"]),
        "--out-root", str(arm_out_root(arm_key)),
        "--collate-root", str(arm_collate_root(arm_key)),
        "--gpu", str(gpu),
        "--columns", *EVAL_COLUMNS,
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path = log_dir / f"{arm_key}.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] $ {' '.join(cmd)}\n")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(proc.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr)
        f.write(f"\n[returncode={proc.returncode} elapsed_sec={time.time()-t0:.2f}]\n")
    if proc.returncode != 0:
        return {"arm": arm_key, "status": "failed", "returncode": proc.returncode, "log": str(log_path), "stderr_tail": proc.stderr[-2000:], "stdout_tail": proc.stdout[-2000:]}
    if not out_json.exists():
        return {"arm": arm_key, "status": "missing_output", "log": str(log_path), "expected": str(out_json)}
    return {"arm": arm_key, "status": "done", "per_target": str(out_json), "elapsed_sec": round(time.time()-t0, 3), "log": str(log_path)}


def extract_scores(payload: dict[str, Any] | None) -> dict[str, float | None] | None:
    if payload is None:
        return None
    tasks = payload.get("tasks", {})
    out: dict[str, float | None] = {}
    for c in ZERO_COLUMNS:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if rec.get("score") is not None else None
    gp_vals = []
    for c in GP_COLS:
        rec = tasks.get(c, {})
        if rec.get("score") is not None:
            gp_vals.append(float(rec["score"]))
    out["GlobalPIQA"] = mean(gp_vals) if len(gp_vals) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores: dict[str, float | None] | None) -> float | None:
    if scores is None:
        return None
    vals = [scores.get(c) for c in SCORE_COLUMNS]
    if any(v is None for v in vals):
        return None
    return mean(float(v) for v in vals)


def summarize() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for arm_key, spec in ARMS.items():
        payload = read_json(per_target_path(arm_key))
        scores = extract_scores(payload)
        c7 = cheap7(scores)
        rows[arm_key] = {
            "label": spec["label"],
            "role": spec["role"],
            "source_run_dir": str(spec["source_run_dir"]),
            "endpoint": spec["endpoint"],
            "target": spec["target"],
            "per_target": str(per_target_path(arm_key)),
            "complete": c7 is not None,
            "scores": scores,
            "cheap7": c7,
        }
    base = rows.get("adamw40", {}).get("scores")
    cont = rows.get("muon40_cont", {}).get("scores")
    for arm_key, row in rows.items():
        sc = row.get("scores")
        if sc and base:
            row["delta_vs_adamw40"] = {c: (sc[c] - base[c]) if sc.get(c) is not None and base.get(c) is not None else None for c in SCORE_COLUMNS}
            if row.get("cheap7") is not None and rows["adamw40"].get("cheap7") is not None:
                row["delta_vs_adamw40"]["cheap7"] = row["cheap7"] - rows["adamw40"]["cheap7"]
        else:
            row["delta_vs_adamw40"] = None
        if sc and cont:
            row["delta_vs_continuous_muon40"] = {c: (sc[c] - cont[c]) if sc.get(c) is not None and cont.get(c) is not None else None for c in SCORE_COLUMNS}
            if row.get("cheap7") is not None and rows["muon40_cont"].get("cheap7") is not None:
                row["delta_vs_continuous_muon40"]["cheap7"] = row["cheap7"] - rows["muon40_cont"]["cheap7"]
        else:
            row["delta_vs_continuous_muon40"] = None
    summary = {
        "status": "MUON_SWITCH_40M_SUMMARY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Same-exposure 40M comparison of A02 continuous AdamW, continuous Muon, and Muon-to-AdamW switch artifacts. The switch runs do not yet expose 70M/80M checkpoints.",
        "columns": SCORE_COLUMNS,
        "rows": rows,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_ROOT / "muon_switch_40m_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = OUT_ROOT / "muon_switch_40m_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "label", *SCORE_COLUMNS, "cheap7", "delta_cheap7_vs_adamw40", "delta_cheap7_vs_continuous_muon40", "complete"])
        for arm_key, row in rows.items():
            sc = row.get("scores") or {}
            writer.writerow([
                arm_key, row["label"], *[sc.get(c) for c in SCORE_COLUMNS], row.get("cheap7"),
                (row.get("delta_vs_adamw40") or {}).get("cheap7"),
                (row.get("delta_vs_continuous_muon40") or {}).get("cheap7"), row.get("complete"),
            ])
    lines = [
        "# research — Muon→AdamW 40M artifact harvest",
        "",
        "A02's switch runs currently expose checkpoints only through 40M, so this is a same-exposure 40M comparison rather than the intended 70M/80M mature comparison.",
        "",
        "| arm | role | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 vs AdamW40 | Δcheap7 vs cont-Muon40 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_key, row in rows.items():
        sc = row.get("scores")
        if sc and row.get("cheap7") is not None:
            lines.append(
                f"| {arm_key} | {row['role']} | {sc['BLiMP']:.2f} | {sc['Supplement']:.2f} | {sc['EWoK']:.2f} | {sc['Entity']:.2f} | {sc['COMPS']:.2f} | {sc['GlobalPIQA']:.2f} | {sc['Reading']:.3f} | {row['cheap7']:.4f} | "
                f"{(row.get('delta_vs_adamw40') or {}).get('cheap7', float('nan')):+.4f} | {(row.get('delta_vs_continuous_muon40') or {}).get('cheap7', float('nan')):+.4f} |"
            )
        else:
            lines.append(f"| {arm_key} | {row['role']} | incomplete |  |  |  |  |  |  |  |  |  |")
    lines += [
        "",
        f"Summary JSON: `{summary_path}`",
        f"CSV: `{csv_path}`",
    ]
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["csv_path"] = str(csv_path)
    summary["note_path"] = str(NOTE_PATH)
    print(json.dumps({"status": summary["status"], "complete_arms": [k for k, v in rows.items() if v.get("complete")], "summary": str(summary_path), "note": str(NOTE_PATH)}, indent=2), flush=True)
    return summary


def run_all(gpus: list[int], force: bool, timeout: int, parallel: int) -> None:
    # Small two-GPU scheduler. Each arm writes a disjoint out_root/collate/proxy subdir.
    pending = list(ARMS)
    active: list[tuple[str, subprocess.Popen[str], Path]] = []
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    while pending or active:
        while pending and len(active) < max(1, parallel):
            arm = pending.pop(0)
            gpu = gpus[len(active) % len(gpus)]
            log = OUT_ROOT / "logs" / f"scheduler_{arm}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [py, "-B", str(_public_path('experiments/archive/representation_and_objectives/scripts/muon_switch_40m_eval.py')), "--run", "--arm", arm, "--gpu", str(gpu), "--timeout", str(timeout)]
            if force:
                cmd.append("--force")
            f = log.open("a", encoding="utf-8")
            f.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] $ {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
            # Keep file handle alive by storing as attribute.
            proc._qiushi_log_handle = f  # type: ignore[attr-defined]
            active.append((arm, proc, log))
            print(json.dumps({"event": "launched", "arm": arm, "gpu": gpu, "log": str(log)}), flush=True)
        time.sleep(5)
        still: list[tuple[str, subprocess.Popen[str], Path]] = []
        for arm, proc, log in active:
            rc = proc.poll()
            if rc is None:
                still.append((arm, proc, log))
            else:
                try:
                    proc._qiushi_log_handle.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
                print(json.dumps({"event": "finished", "arm": arm, "returncode": rc, "log": str(log)}), flush=True)
                if rc != 0:
                    raise SystemExit(rc)
        active = still
    summarize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--arm", choices=sorted(ARMS), default="adamw40")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.run:
        result = run_arm(args.arm, args.gpu, force=args.force, timeout=args.timeout)
        print(json.dumps(result, indent=2), flush=True)
    if args.run_all:
        run_all(args.gpus, force=args.force, timeout=args.timeout, parallel=args.parallel)
    if args.summarize:
        summarize()
    if not (args.run or args.run_all or args.summarize):
        ap.error("choose --run, --run-all, or --summarize")


if __name__ == "__main__":
    main()
