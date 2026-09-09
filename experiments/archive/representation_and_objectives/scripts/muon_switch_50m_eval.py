#!/usr/bin/env python3
"""Collect and evaluate Muon→AdamW 50M artifacts.

The comparison requires a later checkpoint because 40M does not test optimizer
consolidation: Muon40→AdamW at 40M is only the switch boundary, and both switch
runs already expose 50M checkpoints.  This wrapper reuses the validated
40M evaluator, but retargets it to the deepest common 50M checkpoints and
writes only under representation_and_objectives.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import importlib.util
import json
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
BASE = A01 / "scripts/muon_switch_40m_eval.py"
OUT_ROOT = A01 / "data/muon_switch_50m_eval"
NOTE_PATH = A01 / "notes/muon_switch_50m_harvest.md"

ARMS50: dict[str, dict[str, Any]] = {
    "adamw50": {
        "label": "continuous_AdamW_50M",
        "source_run_dir": A02 / "training/runs/complianttok_reinvest_seed43022_r2",
        "endpoint": "chck_50M",
        "target": "a02_continuous_adamw_reinvest_50M",
        "role": "matched legal compact-view AdamW reference at 50M",
    },
    "muon50_cont": {
        "label": "continuous_Muon_wdmatch_50M",
        "source_run_dir": A02 / "training/runs/muon_lr008_wd00125_seed43022_80M",
        "endpoint": "chck_50M",
        "target": "a02_continuous_muon_wdmatch_50M",
        "role": "continuous matched-decay Muon reference at 50M",
    },
    "muon20toadamw50": {
        "label": "Muon20M_to_AdamW_50M",
        "source_run_dir": A02 / "training/runs/muon20toadamw_seed43022_80M",
        "endpoint": "chck_50M",
        "target": "a02_muon20toadamw_50M",
        "role": "20M Muon followed by about 30M AdamW recovery/consolidation",
    },
    "muon40toadamw50": {
        "label": "Muon40M_to_AdamW_50M",
        "source_run_dir": A02 / "training/runs/muon40toadamw_seed43022_80M",
        "endpoint": "chck_50M",
        "target": "a02_muon40toadamw_50M",
        "role": "40M Muon followed by about 10M AdamW recovery after the observed handoff shock",
    },
}


def load_base():
    spec = importlib.util.spec_from_file_location("muon_switch_40m_eval_as_50m", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.OUT_ROOT = OUT_ROOT
    mod.NOTE_PATH = NOTE_PATH
    mod.ARMS.clear()
    mod.ARMS.update(ARMS50)
    return mod


def write_50m_summary(mod) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for arm_key, spec in mod.ARMS.items():
        payload = mod.read_json(mod.per_target_path(arm_key))
        scores = mod.extract_scores(payload)
        c7 = mod.cheap7(scores)
        rows[arm_key] = {
            "label": spec["label"],
            "role": spec["role"],
            "source_run_dir": str(spec["source_run_dir"]),
            "endpoint": spec["endpoint"],
            "target": spec["target"],
            "per_target": str(mod.per_target_path(arm_key)),
            "complete": c7 is not None,
            "scores": scores,
            "cheap7": c7,
        }
    base = rows.get("adamw50", {}).get("scores")
    cont = rows.get("muon50_cont", {}).get("scores")
    score_cols = mod.SCORE_COLUMNS
    for arm_key, row in rows.items():
        sc = row.get("scores")
        if sc and base:
            row["delta_vs_adamw50"] = {c: (sc[c] - base[c]) if sc.get(c) is not None and base.get(c) is not None else None for c in score_cols}
            if row.get("cheap7") is not None and rows["adamw50"].get("cheap7") is not None:
                row["delta_vs_adamw50"]["cheap7"] = row["cheap7"] - rows["adamw50"]["cheap7"]
        else:
            row["delta_vs_adamw50"] = None
        if sc and cont:
            row["delta_vs_continuous_muon50"] = {c: (sc[c] - cont[c]) if sc.get(c) is not None and cont.get(c) is not None else None for c in score_cols}
            if row.get("cheap7") is not None and rows["muon50_cont"].get("cheap7") is not None:
                row["delta_vs_continuous_muon50"]["cheap7"] = row["cheap7"] - rows["muon50_cont"]["cheap7"]
        else:
            row["delta_vs_continuous_muon50"] = None
    summary = {
        "status": "MUON_SWITCH_50M_SUMMARY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Deepest common existing-weight 50M comparison of A02 continuous AdamW, continuous Muon, and abrupt Muon-to-AdamW switch artifacts. This tests post-switch recovery/consolidation without launching new training.",
        "boundary": "Evaluation-only harvest of existing checkpoints; do not infer every possible Muon-to-AdamW consolidation scheme from this abrupt empty-moment handoff.",
        "columns": score_cols,
        "rows": rows,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_ROOT / "muon_switch_50m_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = OUT_ROOT / "muon_switch_50m_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "label", *score_cols, "cheap7", "delta_cheap7_vs_adamw50", "delta_cheap7_vs_continuous_muon50", "complete"])
        for arm_key, row in rows.items():
            sc = row.get("scores") or {}
            writer.writerow([
                arm_key, row["label"], *[sc.get(c) for c in score_cols], row.get("cheap7"),
                (row.get("delta_vs_adamw50") or {}).get("cheap7"),
                (row.get("delta_vs_continuous_muon50") or {}).get("cheap7"), row.get("complete"),
            ])
    lines = [
        "# research — Muon→AdamW 50M artifact harvest",
        "",
        "The 40M comparison could not test AdamW recovery for `muon40toadamw`, because 40M is the switch boundary. This table uses the deepest common existing 50M checkpoints without launching new training.",
        "",
        "| arm | role | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 vs AdamW50 | Δcheap7 vs cont-Muon50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_key, row in rows.items():
        sc = row.get("scores")
        if sc and row.get("cheap7") is not None:
            lines.append(
                f"| {arm_key} | {row['role']} | {sc['BLiMP']:.2f} | {sc['Supplement']:.2f} | {sc['EWoK']:.2f} | {sc['Entity']:.2f} | {sc['COMPS']:.2f} | {sc['GlobalPIQA']:.2f} | {sc['Reading']:.3f} | {row['cheap7']:.4f} | "
                f"{(row.get('delta_vs_adamw50') or {}).get('cheap7', float('nan')):+.4f} | {(row.get('delta_vs_continuous_muon50') or {}).get('cheap7', float('nan')):+.4f} |"
            )
        else:
            lines.append(f"| {arm_key} | {row['role']} | incomplete |  |  |  |  |  |  |  |  |  |")
    lines += [
        "",
        "Interpretive boundary: negative results close only these abrupt switch artifacts with empty AdamW hidden-matrix moments, not smoother moment-preserving or blended consolidation.",
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


def run_all(mod, gpus: list[int], force: bool, timeout: int, parallel: int) -> None:
    pending = list(mod.ARMS)
    active: list[tuple[str, subprocess.Popen[str], Path]] = []
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    while pending or active:
        while pending and len(active) < max(1, parallel):
            arm = pending.pop(0)
            gpu = gpus[len(active) % len(gpus)]
            log = OUT_ROOT / "logs" / f"scheduler_{arm}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [py, "-B", str(_public_path('experiments/archive/representation_and_objectives/scripts/muon_switch_50m_eval.py')), "--run", "--arm", arm, "--gpu", str(gpu), "--timeout", str(timeout)]
            if force:
                cmd.append("--force")
            f = log.open("a", encoding="utf-8")
            f.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] $ {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
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
    write_50m_summary(mod)


def main() -> None:
    mod = load_base()
    mod.summarize = lambda: write_50m_summary(mod)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--arm", choices=sorted(ARMS50), default="adamw50")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.run:
        result = mod.run_arm(args.arm, args.gpu, force=args.force, timeout=args.timeout)
        print(json.dumps(result, indent=2), flush=True)
    if args.run_all:
        run_all(mod, args.gpus, args.force, args.timeout, args.parallel)
    if args.summarize:
        write_50m_summary(mod)
    if not (args.run or args.run_all or args.summarize):
        ap.error("choose --run, --run-all, or --summarize")


if __name__ == "__main__":
    main()
